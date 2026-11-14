# synth.py : génère des questions synthétiques pour les articles absents de l'évaluation
import json
import re
from pathlib import Path

import os
import time

from google import genai

CHUNKS = Path("data/chunks.jsonl")
EVAL = Path("data/eval.jsonl")
SORTIE = Path("data/synth_train.jsonl")
MODELE_LLM = "gemini-3.5-flash-lite"  
PAUSE_ENTRE_APPELS = 6  # secondes, pour rester sous la limite de requêtes par minute
MAX_TENTATIVES = 3
MIN_MOTS = 30  # on ignore les chunks trop courts

PROMPT = """Tu es un salarié ou un citoyen qui pose une question à un juriste, à propos du texte ci-dessous (extrait du RGPD).

Écris exactement 4 questions réalistes auxquelles ce texte permet de répondre :
- 2 questions en français, 1 en anglais, 1 en chinois simplifié ;
- langage courant, décrivant une situation concrète, comme le ferait un non-juriste ;
- ne recopie pas les termes du texte et ne cite jamais un numéro d'article ;
- chaque question doit être compréhensible sans voir le texte.

Réponds uniquement par un tableau JSON, sans autre texte :
[{{"lang": "fr", "question": "..."}}, {{"lang": "fr", "question": "..."}}, {{"lang": "en", "question": "..."}}, {{"lang": "zh", "question": "..."}}]

Texte :
{texte}"""


def lire_jsonl(chemin):
    lignes = []
    with chemin.open(encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
    return lignes


def articles_de_evaluation():
    gold = set()
    for question in lire_jsonl(EVAL):
        for article in question["gold"]:
            gold.add(article)
    return gold


def extraire_json(texte):
    # retire d'éventuelles balises ```json autour de la réponse
    nettoye = re.sub(r"^```(?:json)?\s*|\s*```$", "", texte.strip())
    return json.loads(nettoye)

def generer_questions(client, texte):
    """Appelle Gemini, avec nouvelle tentative en cas de limite de débit (429)."""
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            reponse = client.models.generate_content(
                model=MODELE_LLM,
                contents=PROMPT.format(texte=texte),
                config={"response_mime_type": "application/json"},
            )
            return extraire_json(reponse.text)
        except Exception as erreur:
            if "429" in str(erreur) and tentative < MAX_TENTATIVES:
                print(f"Limite de débit, attente 60 s (tentative {tentative})")
                time.sleep(60)
            else:
                raise


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY absente : définir la variable d'environnement avant de lancer.")
    client = genai.Client()

    # test rapide : on s'arrête tout de suite si le modèle est inaccessible
    try:
        client.models.generate_content(model=MODELE_LLM, contents="Réponds : ok")
    except Exception as erreur:
        raise SystemExit(f"Modèle inaccessible ({MODELE_LLM}) : {erreur}")
    
    gold = articles_de_evaluation()

    candidats = []
    for chunk in lire_jsonl(CHUNKS):
        if chunk["lang"] != "en":
            continue
        if chunk["article"] in gold:
            continue  # anti-fuite : aucun article de l'évaluation dans l'entraînement
        if len(chunk["text"].split()) < MIN_MOTS:
            continue
        candidats.append(chunk)

    # reprise possible : on saute les chunks déjà traités
    deja_faits = set()
    if SORTIE.exists():
        for ligne in lire_jsonl(SORTIE):
            deja_faits.add(ligne["positive_id"])

    print(f"{len(candidats)} chunks candidats, {len(deja_faits)} déjà traités")

    with SORTIE.open("a", encoding="utf-8") as f:
        for chunk in candidats:
            if chunk["id"] in deja_faits:
                continue
            
            try:
                questions = generer_questions(client, chunk["text"])
            except Exception as erreur:
                print(f"Échec {chunk['id']} : {erreur}")
                continue
           
            for q in questions:
                ligne = {
                    "question": q["question"],
                    "lang": q["lang"],
                    "positive_id": chunk["id"],
                    "article": chunk["article"],
                    "positive": chunk["text"],
                }
                f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{chunk['id']} : {len(questions)} questions")
            time.sleep(PAUSE_ENTRE_APPELS)


if __name__ == "__main__":
    main()