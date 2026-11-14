# finetune.py : fine-tuning de multilingual-e5-small avec MultipleNegativesRankingLoss
import json
import random
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, losses

TRAIN = Path("data/synth_train.jsonl")
BASE = "intfloat/multilingual-e5-small"
SORTIE = "models/e5-finetune"
TAILLE_BATCH = 16
LONGUEUR_MAX = 256
EPOCHS = 2
LR = 2e-5
GRAINE = 42


def lire_paires():
    paires = []
    with TRAIN.open(encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                paires.append(json.loads(ligne))
    return paires


def construire_batches(paires, taille):
    """Batches sans passage en double (sinon faux négatifs dans la loss)."""
    random.shuffle(paires)
    batches = []
    restantes = list(paires)
    while restantes:
        batch = []
        passages_vus = set()
        suivantes = []
        for paire in restantes:
            if len(batch) < taille and paire["positive_id"] not in passages_vus:
                batch.append(paire)
                passages_vus.add(paire["positive_id"])
            else:
                suivantes.append(paire)
        batches.append(batch)
        restantes = suivantes
    return batches


def vers_appareil(features, appareil):
    deplace = {}
    for cle, valeur in features.items():
        deplace[cle] = valeur.to(appareil)
    return deplace


def main():
    random.seed(GRAINE)
    torch.manual_seed(GRAINE)

    paires = lire_paires()
    print(f"{len(paires)} paires d'entraînement")

    modele = SentenceTransformer(BASE)
    modele.max_seq_length = LONGUEUR_MAX
    fonction_perte = losses.MultipleNegativesRankingLoss(modele)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=LR)
    appareil = modele.device

    modele.train()
    for epoch in range(EPOCHS):
        batches = construire_batches(paires, TAILLE_BATCH)
        somme_pertes = 0.0
        for numero, batch in enumerate(batches, start=1):
            requetes = []
            passages = []
            for paire in batch:
                requetes.append("query: " + paire["question"])
                passages.append("passage: " + paire["positive"])

            features_requetes = vers_appareil(modele.tokenize(requetes), appareil)
            features_passages = vers_appareil(modele.tokenize(passages), appareil)

            perte = fonction_perte([features_requetes, features_passages], None)
            perte.backward()
            optimiseur.step()
            optimiseur.zero_grad()

            somme_pertes += perte.item()
            print(f"epoch {epoch + 1}/{EPOCHS}  batch {numero}/{len(batches)}  perte {perte.item():.4f}")

        print(f"epoch {epoch + 1} : perte moyenne {somme_pertes / len(batches):.4f}")

    modele.eval()
    Path(SORTIE).parent.mkdir(parents=True, exist_ok=True)
    modele.save(SORTIE)
    print(f"Modèle enregistré : {SORTIE}")


if __name__ == "__main__":
    main()