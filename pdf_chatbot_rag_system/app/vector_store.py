import os
import json
import logging
from typing import List
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- CONFIG ----------
CHUNKS_JSON_PATH = "data/chunks.json"
VECTOR_STORE_DIR = "data/faiss_langchain_index"
EMBED_MODEL = "sentence-transformers/LaBSE"
# ----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chunks_as_documents(json_path: str) -> List[Document]:
    """Load text chunks from JSON and wrap them as LangChain Documents."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Chunk file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    if not raw_chunks:
        raise ValueError("❌ Chunk file is empty.")

    documents = [
        Document(page_content=chunk["text"], metadata=chunk.get("metadata", {}))
        for chunk in raw_chunks
    ]
    return documents

def build_faiss_vector_store(documents: List[Document]) -> FAISS:
    """Embed documents using HuggingFaceEmbeddings and create a FAISS vector store."""
    logger.info(f"🔍 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    logger.info(f"🔢 Embedding and indexing {len(documents)} documents...")
    return FAISS.from_documents(documents, embedding=embeddings)

def save_faiss_store(vectorstore: FAISS, save_dir: str):
    """Save the FAISS vector store to disk."""
    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)
    logger.info(f"✅ Vector store saved at: {save_dir}")

def load_faiss_retriever(k: int = 10):
    """Load the FAISS vector store and return a retriever with configurable top-k."""
    logger.info("📦 Loading FAISS vector store for retrieval...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
        raise FileNotFoundError(f"❌ FAISS index not found in directory: {VECTOR_STORE_DIR}")

    vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": k})

def run_vector_store_pipeline():
    """Main pipeline to generate vector store from chunks.json and save to disk."""
    logger.info("📂 Loading chunks as LangChain Documents...")
    docs = load_chunks_as_documents(CHUNKS_JSON_PATH)
    logger.info(f"✅ Loaded {len(docs)} documents.")

    store = build_faiss_vector_store(docs)
    save_faiss_store(store, VECTOR_STORE_DIR)

# --- Optional: CLI Test ---
if __name__ == "__main__":
    run_vector_store_pipeline()
    # To test retrieval manually:
    # retriever = load_faiss_retriever(k=3)
    # results = retriever.invoke("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
    # for i, doc in enumerate(results):
    #     print(f"Chunk {i+1}: {doc.page_content[:250]}")
