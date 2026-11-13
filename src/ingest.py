# ingest.py : télécharge le RGPD (EN/FR), le découpe par article et écrit data/chunks.jsonl
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = {
    "en": "https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng",
    "fr": "https://eur-lex.europa.eu/eli/reg/2016/679/oj/fra",
}
RAW_DIR = Path("data/raw")
SORTIE = Path("data/chunks.jsonl")
MAX_MOTS = 350  # au-delà, on découpe l'article par paragraphe

# "Article 17" (en/fr) ou "Article premier" (fr), seul sur sa ligne
RE_ARTICLE = re.compile(r"^Article\s+(premier|\d+)$", re.IGNORECASE)
# "CHAPTER II", "CHAPITRE II", "SECTION 1" : en-têtes à ignorer (la ligne suivante est leur titre)
RE_CHAPITRE = re.compile(r"^(CHAPTER|CHAPITRE|SECTION)\s+([IVXL]+|\d+)$", re.IGNORECASE)
# début de paragraphe numéroté : "1. texte" ou "(1) texte" (définitions de l'article 4)
RE_MARQUEUR = re.compile(r"^(?:(\d+)\.|\((\d+)\))\s+")
# puce isolée dans une cellule de tableau : "(a)", "(1)", "1."
RE_PUCE_SEULE = re.compile(r"^(\d+\.|\([a-zA-Z0-9]+\))$")
# fin du texte : formule de signature
RE_FIN = re.compile(r"^(Done at|Fait à)\b")


def telecharger(langue):
    chemin = RAW_DIR / f"gdpr_{langue}.html"
    if chemin.exists():
        return chemin
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entetes = {"User-Agent": "Mozilla/5.0 (projet gdpr-rag)"}
    reponse = requests.get(URLS[langue], headers=entetes, timeout=60)
    if reponse.status_code != 200 or b"Article 17" not in reponse.content:
        raise RuntimeError(
            f"Téléchargement bloqué (HTTP {reponse.status_code}). "
            f"Ouvrir {URLS[langue]} dans le navigateur, enregistrer en HTML "
            f"sous {chemin}, puis relancer."
        )
    chemin.write_bytes(reponse.content)
    return chemin


def extraire_lignes(chemin):
    html = chemin.read_text(encoding="utf-8")
    soupe = BeautifulSoup(html, "html.parser")
    lignes = []
    for balise in soupe.find_all("p"):
        # split() normalise aussi les espaces insécables
        texte = " ".join(balise.get_text(" ").split())
        if texte:
            lignes.append(texte)

    # fusion des puces isolées avec la ligne suivante : "(a)" + "texte"
    fusionnees = []
    i = 0
    while i < len(lignes):
        ligne = lignes[i]
        if RE_PUCE_SEULE.match(ligne) and i + 1 < len(lignes):
            fusionnees.append(ligne + " " + lignes[i + 1])
            i += 2
        else:
            fusionnees.append(ligne)
            i += 1
    return fusionnees


def decouper_articles(lignes):
    articles = []
    courant = None
    sauter_suivante = False
    for ligne in lignes:
        if RE_FIN.match(ligne):
            break
        if sauter_suivante:
            sauter_suivante = False
            continue
        if RE_CHAPITRE.match(ligne):
            sauter_suivante = True  # la ligne suivante est le titre du chapitre
            continue
        correspondance = RE_ARTICLE.match(ligne)
        if correspondance:
            brut = correspondance.group(1).lower()
            numero = 1 if brut == "premier" else int(brut)
            courant = {"numero": numero, "titre": None, "lignes": []}
            articles.append(courant)
            continue
        if courant is None:
            continue  # considérants, avant l'article 1
        if courant["titre"] is None:
            courant["titre"] = ligne
            continue
        courant["lignes"].append(ligne)
    return articles


def regrouper_paragraphes(lignes):
    """Regroupe les lignes en blocs [numero_marqueur, texte]."""
    blocs = []
    for ligne in lignes:
        correspondance = RE_MARQUEUR.match(ligne)
        if correspondance:
            numero = correspondance.group(1) or correspondance.group(2)
            blocs.append([numero, ligne])
        elif blocs:
            blocs[-1][1] += "\n" + ligne
        else:
            blocs.append(["intro", ligne])
    return blocs


def construire_chunks(articles, langue):
    chunks = []
    refs_vues = {}

    def ajouter(numero, titre, ref, corps):
        if ref in refs_vues:  # garde-fou : identifiant unique
            refs_vues[ref] += 1
            ref = f"{ref}#{refs_vues[ref]}"
        else:
            refs_vues[ref] = 1
        # le titre est répété dans chaque chunk pour donner du contexte à la recherche
        texte = f"Article {numero} - {titre}\n{corps}"
        chunks.append({
            "id": f"{langue}:{ref}",
            "article": numero,
            "ref": ref,
            "lang": langue,
            "title": titre,
            "text": texte,
        })

    for article in articles:
        numero = article["numero"]
        titre = article["titre"] or ""
        corps = "\n".join(article["lignes"])

        if len(corps.split()) <= MAX_MOTS:
            ajouter(numero, titre, f"Art.{numero}", corps)
            continue

        for numero_bloc, texte_bloc in regrouper_paragraphes(article["lignes"]):
            ajouter(numero, titre, f"Art.{numero}({numero_bloc})", texte_bloc)
    return chunks


def main():
    tous = []
    for langue in URLS:
        chemin = telecharger(langue)
        lignes = extraire_lignes(chemin)
        articles = decouper_articles(lignes)
        chunks = construire_chunks(articles, langue)
        print(f"[{langue}] {len(articles)} articles (attendu : 99), {len(chunks)} chunks")
        tous.extend(chunks)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE.open("w", encoding="utf-8") as f:
        for chunk in tous:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Écrit : {SORTIE}")


if __name__ == "__main__":
    main()