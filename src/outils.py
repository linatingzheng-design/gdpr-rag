# outils.py : fonctions communes (lecture jsonl, appel LLM avec nouvelle tentative)
import json
import time

MAX_TENTATIVES = 3


def lire_jsonl(chemin):
    lignes = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
    return lignes


def appeler(client, modele, contenu, json_mode=False):
    """Appelle Gemini ; en cas de limite de débit (429), attend et réessaie."""
    config = None
    if json_mode:
        config = {"response_mime_type": "application/json"}
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            reponse = client.models.generate_content(
                model=modele, contents=contenu, config=config
            )
            return reponse.text
        except Exception as erreur:
            if "429" in str(erreur) and tentative < MAX_TENTATIVES:
                print(f"Limite de débit, attente 60 s (tentative {tentative})")
                time.sleep(60)
            else:
                raise