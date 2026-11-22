# agreement.py : accord entre le juge et l'annotation humaine
from pathlib import Path

from outils import lire_jsonl

LABELS = Path("data/judge_labels.jsonl")
JUGE = Path("results/judge_results.jsonl")
CRITERES = ["faithful", "citation_ok"]


def kappa_cohen(humain, juge):
    n = len(humain)
    accords = 0
    for h, j in zip(humain, juge):
        if h == j:
            accords += 1
    po = accords / n
    p_h = sum(humain) / n
    p_j = sum(juge) / n
    pe = p_h * p_j + (1 - p_h) * (1 - p_j)
    if pe == 1:
        return None  # toutes les valeurs identiques : kappa indéfini
    return (po - pe) / (1 - pe)


def main():
    labels = {}
    for ligne in lire_jsonl(LABELS):
        labels[ligne["id"]] = ligne
    verdicts = {}
    for ligne in lire_jsonl(JUGE):
        verdicts[ligne["id"]] = ligne

    ids = []
    for identifiant in labels:
        if identifiant in verdicts:
            ids.append(identifiant)
    print(f"{len(ids)} questions annotées et jugées\n")

    for critere in CRITERES:
        humain = []
        juge = []
        for i in ids:
            humain.append(labels[i][critere])
            juge.append(verdicts[i][critere])

        accords = 0
        for h, j in zip(humain, juge):
            if h == j:
                accords += 1
        kappa = kappa_cohen(humain, juge)
        kappa_txt = f"{kappa:.2f}" if kappa is not None else "indéfini"
        print(f"== {critere} : accord {accords}/{len(ids)} ({accords / len(ids):.0%}), kappa {kappa_txt}")

        # matrice de confusion
        vrai_vrai = faux_faux = humain_vrai_juge_faux = humain_faux_juge_vrai = 0
        for h, j in zip(humain, juge):
            if h and j:
                vrai_vrai += 1
            elif not h and not j:
                faux_faux += 1
            elif h and not j:
                humain_vrai_juge_faux += 1
            else:
                humain_faux_juge_vrai += 1
        print(f"   humain=oui/juge=oui {vrai_vrai} | humain=non/juge=non {faux_faux}")
        print(f"   juge trop sévère (humain=oui, juge=non) : {humain_vrai_juge_faux}")
        print(f"   juge trop indulgent (humain=non, juge=oui) : {humain_faux_juge_vrai}")

        # accord par langue
        for langue in ["fr", "en", "zh"]:
            n = 0
            ok = 0
            for i in ids:
                if verdicts[i]["lang"] != langue:
                    continue
                n += 1
                if labels[i][critere] == verdicts[i][critere]:
                    ok += 1
            if n > 0:
                print(f"   {langue} : {ok}/{n}")
        print()

    print("== Désaccords à analyser ==")
    for i in ids:
        for critere in CRITERES:
            if labels[i][critere] != verdicts[i][critere]:
                print(f"\n{i} ({verdicts[i]['lang']}) {critere} : humain={labels[i][critere]}, juge={verdicts[i][critere]}")
                print(f"  note humaine : {labels[i]['note']}")
                print(f"  raison du juge : {verdicts[i]['reason']}")
                if verdicts[i]["cites_hors_contexte"]:
                    print(f"  citations hors extraits : {verdicts[i]['cites_hors_contexte']}")


if __name__ == "__main__":
    main()