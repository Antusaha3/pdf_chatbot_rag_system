import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

def embed_text(text: str):
    return embedding_model.embed_query(text)

def evaluate_groundedness(answer: str, chunks: list) -> float:
    """How well is the answer grounded in the top chunks?"""
    ans_vec = embed_text(answer)
    chunk_vecs = [embed_text(doc.page_content) for doc in chunks]
    sims = cosine_similarity([ans_vec], chunk_vecs)[0]
    return float(np.max(sims))  # highest support among chunks

def evaluate_relevance(question: str, chunks: list) -> float:
    """How relevant are the retrieved chunks to the query?"""
    q_vec = embed_text(question)
    chunk_vecs = [embed_text(doc.page_content) for doc in chunks]
    sims = cosine_similarity([q_vec], chunk_vecs)[0]
    return float(np.mean(sims))  # avg relevance
