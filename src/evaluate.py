# evaluate.py : recall@k et MRR pour BM25, dense et hybride, par langue
import json
from pathlib import Path

from retrieve import Retriever, articles_uniques

EVAL = Path("data/eval.jsonl")
SORTIE_MD = Path("results/metrics.md")
SORTIE_DETAIL = Path("results/detail.jsonl")

METHODES = ["bm25", "dense", "hybride", "hybride+rerank"]
KS = [1, 3, 5, 10]
K_RECHERCHE = 50  # nombre de chunks récupérés avant repli par article


def charger_questions():
    questions = []
    with EVAL.open(encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                questions.append(json.loads(ligne))
    return questions


def evaluer_question(articles, gold):
    """articles : numéros d'articles classés, sans doublon. gold : articles attendus."""
    resultat = {}

    # recall@k : part des articles attendus retrouvés dans les k premiers
    for k in KS:
        top_k = articles[:k]
        trouves = 0
        for article_attendu in gold:
            if article_attendu in top_k:
                trouves += 1
        resultat[f"recall@{k}"] = trouves / len(gold)

    # MRR : inverse du rang du premier article attendu
    rang_premier = None
    for rang, article in enumerate(articles, start=1):
        if article in gold:
            rang_premier = rang
            break
    resultat["mrr"] = 1.0 / rang_premier if rang_premier else 0.0
    resultat["rang_premier"] = rang_premier
    return resultat


def moyenne(valeurs):
    if not valeurs:
        return 0.0
    return sum(valeurs) / len(valeurs)


def agreger(details, methode, langue):
    """Moyenne des métriques pour une méthode et une langue (None = toutes)."""
    lignes = []
    for d in details:
        if d["methode"] != methode:
            continue
        if langue is not None and d["lang"] != langue:
            continue
        lignes.append(d)

    agregat = {"n": len(lignes)}
    for k in KS:
        agregat[f"recall@{k}"] = moyenne([d[f"recall@{k}"] for d in lignes])
    agregat["mrr"] = moyenne([d["mrr"] for d in lignes])
    return agregat


def tableau_markdown(details, langue, titre):
    entetes = ["méthode"] + [f"R@{k}" for k in KS] + ["MRR", "n"]
    sortie = [f"### {titre}", ""]
    sortie.append("| " + " | ".join(entetes) + " |")
    sortie.append("|" + "---|" * len(entetes))
    for methode in METHODES:
        a = agreger(details, methode, langue)
        cellules = [methode]
        for k in KS:
            cellules.append(f"{a[f'recall@{k}']:.2f}")
        cellules.append(f"{a['mrr']:.2f}")
        cellules.append(str(a["n"]))
        sortie.append("| " + " | ".join(cellules) + " |")
    sortie.append("")
    return "\n".join(sortie)


def main():
    questions = charger_questions()
    retriever = Retriever()

    details = []
    for question in questions:
        for methode in METHODES:
            chunks = retriever.chercher(question["question"], methode, k=K_RECHERCHE)
            articles = articles_uniques(chunks)
            metriques = evaluer_question(articles, question["gold"])
            details.append({
                "id": question["id"],
                "lang": question["lang"],
                "methode": methode,
                "gold": question["gold"],
                "top5": articles[:5],
                **metriques,
            })
        print(f"{question['id']} évaluée")

    # détail par question, pour l'analyse des échecs
    SORTIE_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE_DETAIL.open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # tableaux : une section par langue, puis le total
    sections = ["# Résultats (niveau article)", ""]
    langues = []
    for question in questions:
        if question["lang"] not in langues:
            langues.append(question["lang"])
    for langue in langues:
        sections.append(tableau_markdown(details, langue, f"Questions en {langue}"))
    sections.append(tableau_markdown(details, None, "Toutes langues"))

    texte = "\n".join(sections)
    SORTIE_MD.write_text(texte, encoding="utf-8")
    print("\n" + texte)


if __name__ == "__main__":
    main()