# generate.py : RAG complet = recherche + génération avec citations
import os

from google import genai

from retrieve import Retriever

from outils import appeler

MODELE_LLM = "gemini-3.5-flash-lite"  # le nom qui a fonctionné dans synth.py
METHODE = "hybride+rerank"  # mettre "hybride" si le reranker est trop lent
K_ARTICLES = 5

PROMPT = """Tu es un assistant juridique spécialisé dans le RGPD. Réponds à la question en t'appuyant UNIQUEMENT sur les extraits ci-dessous.

Règles :
- réponds dans la langue de la question ;
- après chaque affirmation, cite l'article utilisé entre crochets, par exemple [Art.17] ;
- si les extraits ne permettent pas de répondre, dis-le clairement au lieu d'inventer ;
- reste concis.

Extraits :
{contexte}

Question : {question}"""


def choisir_contextes(retriever, question):
    chunks = retriever.chercher(question, METHODE, k=20)
    contextes = []
    articles_vus = []
    for chunk in chunks:
        if chunk["article"] in articles_vus:
            continue  # un article apparaît en en et en fr : on n'en garde qu'un
        articles_vus.append(chunk["article"])
        contextes.append(chunk)
        if len(contextes) == K_ARTICLES:
            break
    return contextes


def repondre(retriever, client, question):
    contextes = choisir_contextes(retriever, question)

    blocs = []
    for chunk in contextes:
        blocs.append(f"[Art.{chunk['article']}] {chunk['text']}")
    contexte = "\n\n".join(blocs)

    texte_reponse = appeler(
        client, MODELE_LLM, PROMPT.format(contexte=contexte, question=question)
    )

    articles = []
    textes = []
    for chunk in contextes:
        articles.append(chunk["article"])
        textes.append(chunk["text"])
    return {
        "question": question,
        "reponse": texte_reponse,
        "articles": articles,  # utile pour le juge LLM plus tard
        "contextes": textes,
    }


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY absente.")
    client = genai.Client()
    retriever = Retriever()
    while True:
        question = input("\nQuestion (vide pour quitter) : ").strip()
        if not question:
            break
        resultat = repondre(retriever, client, question)
        print("\n" + resultat["reponse"])
        print("Articles fournis au modèle :", resultat["articles"])