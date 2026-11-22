# label.py : outil d'annotation manuelle (à faire AVANT de regarder les verdicts du juge)
import json
import random
from pathlib import Path

from outils import lire_jsonl

REPONSES = Path("results/answers.jsonl")
SORTIE = Path("data/judge_labels.jsonl")
NB_A_ETIQUETER = 20
NB_ECHECS_RECHERCHE = 6
GRAINE = 7

CONSIGNES = """
faithful : toutes les affirmations de la réponse sont soutenues par les extraits fournis.
           Une information ajoutée qui n'est pas dans les extraits, ou un sens déformé -> n.
           Une réponse qui dit honnêtement que les extraits ne permettent pas de répondre -> y.
citation_ok : chaque [Art.N] cité figure dans les extraits ET soutient l'affirmation qu'il accompagne.
              Citation absente des extraits, qui ne soutient pas l'affirmation, ou réponse
              affirmative sans aucune citation -> n. Réponse "je ne peux pas répondre" sans citation -> y.
"""


def selectionner(reponses):
    random.seed(GRAINE)
    echecs = []
    autres = []
    for r in reponses:
        tous_trouves = True
        for article_attendu in r["gold"]:
            if article_attendu not in r["articles"]:
                tous_trouves = False
        if tous_trouves:
            autres.append(r)
        else:
            echecs.append(r)
    random.shuffle(echecs)
    random.shuffle(autres)

    choisis = echecs[:NB_ECHECS_RECHERCHE]
    reste = NB_A_ETIQUETER - len(choisis)
    choisis = choisis + autres[:reste]
    return choisis


def demander(texte):
    while True:
        saisie = input(texte + " (y/n) : ").strip().lower()
        if saisie in ("y", "n"):
            return saisie == "y"


def main():
    reponses = lire_jsonl(REPONSES)
    choisis = selectionner(reponses)

    deja_faits = set()
    if SORTIE.exists():
        for ligne in lire_jsonl(SORTIE):
            deja_faits.add(ligne["id"])

    langues = {}
    for r in choisis:
        langues[r["lang"]] = langues.get(r["lang"], 0) + 1
    print(f"Sélection : {len(choisis)} questions, répartition {langues}")
    print(CONSIGNES)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE.open("a", encoding="utf-8") as f:
        for numero, r in enumerate(choisis, start=1):
            if r["id"] in deja_faits:
                continue
            print("=" * 80)
            print(f"[{numero}/{len(choisis)}] {r['id']} ({r['lang']})")
            print("QUESTION :", r["question"])
            print("\nEXTRAITS FOURNIS :")
            for article, texte in zip(r["articles"], r["contextes"]):
                print(f"\n[Art.{article}] {texte}")
            print("\nRÉPONSE :\n" + r["reponse"])
            print()
            faithful = demander("faithful")
            citation_ok = demander("citation_ok")
            note = input("note (raison courte, facultatif) : ").strip()
            ligne = {"id": r["id"], "faithful": faithful, "citation_ok": citation_ok, "note": note}
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()