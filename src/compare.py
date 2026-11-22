# compare.py : compare les résultats avant/après fine-tuning
import json
from pathlib import Path

BASE = Path("results/detail_base.jsonl")
FINETUNE = Path("results/detail_finetune.jsonl")
METHODES = ["bm25", "dense", "hybride", "hybride+rerank"]
LANGUES = ["fr", "en", "zh", None]  # None = toutes langues


def lire(chemin):
    lignes = []
    with chemin.open(encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
    return lignes


def moyenne(valeurs):
    if not valeurs:
        return 0.0
    return sum(valeurs) / len(valeurs)


def metrique(details, methode, langue, cle):
    valeurs = []
    for d in details:
        if d["methode"] != methode:
            continue
        if langue is not None and d["lang"] != langue:
            continue
        valeurs.append(d[cle])
    return moyenne(valeurs)


def rang(d):
    # rang absent = très mauvais, pour pouvoir comparer
    return d["rang_premier"] if d["rang_premier"] is not None else 999


def main():
    avant = lire(BASE)
    apres = lire(FINETUNE)

    print("| méthode | langue | MRR avant | MRR après | R@3 avant | R@3 après |")
    print("|---|---|---|---|---|---|")
    for methode in METHODES:
        for langue in LANGUES:
            nom = langue if langue else "toutes"
            print(
                f"| {methode} | {nom} "
                f"| {metrique(avant, methode, langue, 'mrr'):.2f} "
                f"| {metrique(apres, methode, langue, 'mrr'):.2f} "
                f"| {metrique(avant, methode, langue, 'recall@3'):.2f} "
                f"| {metrique(apres, methode, langue, 'recall@3'):.2f} |"
            )

    # détail des questions dont le rang a changé (méthode dense seulement)
    index_avant = {}
    for d in avant:
        if d["methode"] == "dense":
            index_avant[d["id"]] = d

    ameliorees = []
    degradees = []
    for d in apres:
        if d["methode"] != "dense":
            continue
        ancien = index_avant[d["id"]]
        if rang(d) < rang(ancien):
            ameliorees.append((d["id"], d["lang"], rang(ancien), rang(d)))
        elif rang(d) > rang(ancien):
            degradees.append((d["id"], d["lang"], rang(ancien), rang(d)))

    print("\nQuestions améliorées (dense) : id, langue, rang avant -> après")
    for ligne in ameliorees:
        print(f"  {ligne[0]} {ligne[1]} {ligne[2]} -> {ligne[3]}")
    print("\nQuestions dégradées (dense) :")
    for ligne in degradees:
        print(f"  {ligne[0]} {ligne[1]} {ligne[2]} -> {ligne[3]}")


if __name__ == "__main__":
    main()