# judge.py : LLM-as-a-judge pour la fidélité et la qualité des citations
import json
import os
import re
import time
from pathlib import Path

from google import genai

from outils import appeler, lire_jsonl

REPONSES = Path("results/answers.jsonl")
SORTIE = Path("results/judge_results.jsonl")
JUGE_MODELE = "gemini-3.5-flash-lite"  
PAUSE = 6

RE_CITATION = re.compile(r"Art\.\s*(\d+)")

PROMPT_JUGE = """Tu évalues la réponse d'un assistant juridique (RGPD). Tu disposes de la question, des extraits fournis à l'assistant et de sa réponse. Juge uniquement d'après les extraits, pas d'après tes connaissances personnelles.

Critère faithful : true si toutes les affirmations factuelles de la réponse sont soutenues par les extraits. false si la réponse ajoute une information absente des extraits ou en déforme le sens. Une réponse qui déclare honnêtement que les extraits ne permettent pas de répondre est fidèle (true).

Critère citation_ok : true si chaque article cité [Art.N] figure parmi les extraits ET soutient réellement l'affirmation qu'il accompagne. false s'il y a une citation absente des extraits, une citation qui ne soutient pas l'affirmation, ou une réponse affirmative sans aucune citation. Une réponse qui déclare ne pas pouvoir répondre et ne cite rien est true.

Question : {question}

Extraits fournis :
{contexte}

Réponse à évaluer :
{reponse}

Réponds uniquement par un JSON, avec la raison avant les verdicts :
{{"reason": "une ou deux phrases", "faithful": true, "citation_ok": true}}"""


def articles_cites(reponse):
    cites = []
    for numero in RE_CITATION.findall(reponse):
        article = int(numero)
        if article not in cites:
            cites.append(article)
    return cites


def juger(client, ligne):
    blocs = []
    for article, texte in zip(ligne["articles"], ligne["contextes"]):
        blocs.append(f"[Art.{article}] {texte}")
    prompt = PROMPT_JUGE.format(
        question=ligne["question"],
        contexte="\n\n".join(blocs),
        reponse=ligne["reponse"],
    )
    texte = appeler(client, JUGE_MODELE, prompt, json_mode=True)
    verdict = json.loads(texte)

    # contrôle déterministe : citations absentes des extraits fournis
    hors_contexte = []
    for article in articles_cites(ligne["reponse"]):
        if article not in ligne["articles"]:
            hors_contexte.append(article)

    return {
        "id": ligne["id"],
        "lang": ligne["lang"],
        "faithful": bool(verdict["faithful"]),
        "citation_ok": bool(verdict["citation_ok"]),
        "reason": verdict["reason"],
        "cites_hors_contexte": hors_contexte,
    }


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY absente.")
    client = genai.Client()

    deja_faits = set()
    if SORTIE.exists():
        for ligne in lire_jsonl(SORTIE):
            deja_faits.add(ligne["id"])

    with SORTIE.open("a", encoding="utf-8") as f:
        for ligne in lire_jsonl(REPONSES):
            if ligne["id"] in deja_faits:
                continue
            try:
                verdict = juger(client, ligne)
            except Exception as erreur:
                print(f"Échec {ligne['id']} : {erreur}")
                continue
            f.write(json.dumps(verdict, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{ligne['id']} faithful={verdict['faithful']} citation_ok={verdict['citation_ok']}")
            time.sleep(PAUSE)

    # synthèse par langue : qualité des réponses générées (pour le README)
    resultats = lire_jsonl(SORTIE)
    for langue in ["fr", "en", "zh", None]:
        n = 0
        fideles = 0
        citations = 0
        for r in resultats:
            if langue is not None and r["lang"] != langue:
                continue
            n += 1
            if r["faithful"]:
                fideles += 1
            if r["citation_ok"]:
                citations += 1
        if n > 0:
            nom = langue if langue else "toutes"
            print(f"{nom:7s} n={n}  fidélité={fideles / n:.2f}  citations={citations / n:.2f}")


if __name__ == "__main__":
    main()