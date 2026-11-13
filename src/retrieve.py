# retrieve.py : BM25, recherche dense (multilingual-e5-small) et fusion RRF
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer,CrossEncoder

CHUNKS = Path("data/chunks.jsonl")
EMBEDDINGS = Path("data/emb.npy")
MODELE = "intfloat/multilingual-e5-small"

MODELE_RERANK = "BAAI/bge-reranker-v2-m3"
N_RERANK = 20  # nombre de candidats hybrides soumis au reranker


def charger_chunks():
    chunks = []
    with CHUNKS.open(encoding="utf-8") as f:
        for ligne in f:
            chunks.append(json.loads(ligne))
    return chunks


def tokeniser(texte):
    # minuscules, suppression des accents, découpage sur les caractères de mot
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", texte)


def fusion_rrf(listes, k_rrf=60):
    """Fusionne plusieurs classements (listes d'indices) par Reciprocal Rank Fusion."""
    scores = {}
    for liste in listes:
        for rang, indice in enumerate(liste):
            scores[indice] = scores.get(indice, 0.0) + 1.0 / (k_rrf + rang + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self):
        self.chunks = charger_chunks()        

        # index BM25 sur tous les chunks (en + fr)
        corpus_tokens = []
        for chunk in self.chunks:
            corpus_tokens.append(tokeniser(chunk["text"]))
        self.bm25 = BM25Okapi(corpus_tokens)

        # modèle dense et embeddings (calculés une seule fois, puis mis en cache)
        self.modele = SentenceTransformer(MODELE)
        self.modele.max_seq_length = 512
        self.embeddings = self._charger_ou_calculer_embeddings()
        self.reranker = None

    def _charger_ou_calculer_embeddings(self):
        if EMBEDDINGS.exists():
            emb = np.load(EMBEDDINGS)
            if emb.shape[0] == len(self.chunks):
                return emb
            print("Cache obsolète (nombre de chunks différent), recalcul.")
        # e5 exige le préfixe "passage: " côté documents
        textes = []
        for chunk in self.chunks:
            textes.append("passage: " + chunk["text"])
        emb = self.modele.encode(
            textes, batch_size=16, normalize_embeddings=True, show_progress_bar=True
        )
        np.save(EMBEDDINGS, emb)
        return emb

    def bm25_indices(self, question, k=50):
        scores = self.bm25.get_scores(tokeniser(question))
        ordre = np.argsort(scores)[::-1]
        indices = []
        for i in ordre[:k]:
            if scores[i] > 0:  # on ignore les chunks sans aucun terme commun
                indices.append(int(i))
        return indices

    def dense_indices(self, question, k=50):
        # e5 exige le préfixe "query: " côté requête
        vecteur = self.modele.encode("query: " + question, normalize_embeddings=True)
        scores = self.embeddings @ vecteur  # vecteurs normalisés : produit scalaire = cosinus
        ordre = np.argsort(scores)[::-1]
        return [int(i) for i in ordre[:k]]

    def hybride_indices(self, question, k=50):
        liste_bm25 = self.bm25_indices(question, k)
        liste_dense = self.dense_indices(question, k)
        return fusion_rrf([liste_bm25, liste_dense])[:k]

    def rerank_indices(self, question, k=50):
        candidats = self.hybride_indices(question, k)
        tete = candidats[:N_RERANK]
        reste = candidats[N_RERANK:]

        # chargement paresseux du cross-encoder
        if self.reranker is None:
            self.reranker = CrossEncoder(MODELE_RERANK, max_length=512)

        paires = []
        for i in tete:
            paires.append((question, self.chunks[i]["text"]))
        scores = self.reranker.predict(paires, batch_size=8)

        # tri des candidats de tête par score décroissant
        ordre = np.argsort(scores)[::-1]
        tete_reordonnee = []
        for position in ordre:
            tete_reordonnee.append(tete[int(position)])

        # les candidats au-delà du rang 20 restent dans leur ordre d'origine
        return tete_reordonnee + reste

    def chercher(self, question, methode="hybride", k=10):
        fonctions = {
            "bm25": self.bm25_indices,
            "dense": self.dense_indices,
            "hybride": self.hybride_indices,
            "hybride+rerank": self.rerank_indices,
        }
        indices = fonctions[methode](question, k)
        return [self.chunks[i] for i in indices]


def articles_uniques(resultats):
    """Ramène une liste de chunks à une liste de numéros d'articles sans doublon.
    À utiliser dans evaluate.py : un article apparaît en en et en fr, on garde le premier rang."""
    vus = []
    for chunk in resultats:
        if chunk["article"] not in vus:
            vus.append(chunk["article"])
    return vus


if __name__ == "__main__":
    retriever = Retriever()
    while True:
        question = input("\nQuestion (vide pour quitter) : ").strip()
        if not question:
            break
        for methode in ["bm25", "dense", "hybride"]:
            resultats = retriever.chercher(question, methode, k=5)
            refs = [chunk["id"] for chunk in resultats]
            print(f"  {methode:8s} {refs}")