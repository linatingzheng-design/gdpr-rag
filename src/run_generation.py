# run_generation.py : génère une réponse RAG pour chaque question de l'évaluation
import os
import time
from pathlib import Path

from google import genai

from generate import repondre
from outils import lire_jsonl
from retrieve import Retriever

EVAL = Path("data/eval.jsonl")
SORTIE = Path("results/answers.jsonl")
PAUSE = 6  # secondes entre deux appels


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY absente.")
    client = genai.Client()
    retriever = Retriever()

    deja_faits = set()
    if SORTIE.exists():
        for ligne in lire_jsonl(SORTIE):
            deja_faits.add(ligne["id"])

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE.open("a", encoding="utf-8") as f:
        for question in lire_jsonl(EVAL):
            if question["id"] in deja_faits:
                continue
            try:
                resultat = repondre(retriever, client, question["question"])
            except Exception as erreur:
                print(f"Échec {question['id']} : {erreur}")
                continue
            ligne = {
                "id": question["id"],
                "lang": question["lang"],
                "question": question["question"],
                "gold": question["gold"],
                "reponse": resultat["reponse"],
                "articles": resultat["articles"],
                "contextes": resultat["contextes"],
            }
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{question['id']} OK")
            time.sleep(PAUSE)


if __name__ == "__main__":
    import json
    main()