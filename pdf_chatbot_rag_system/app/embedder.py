import os
import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS  # ✅ Updated import
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- CONFIGURATION ----------
CHUNKS_PATH = "data/chunks.json"
FAISS_INDEX_DIR = "data/faiss_langchain_index"
EMBED_MODEL_NAME = "sentence-transformers/LaBSE"  # Multilingual, supports Bangla + English
# -----------------------------------

def load_chunks(json_path):
    """Load structured chunks from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_embedding_pipeline():
    print("📥 Loading chunks from JSON...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"✅ Loaded {len(chunks)} chunks.")

    print(f"🔍 Loading embedding model: {EMBED_MODEL_NAME}")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    print("📄 Converting to LangChain Document format...")
    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk.get("metadata", {})
        )
        for chunk in chunks
    ]

    print("📈 Creating FAISS index with LangChain...")
    vectorstore = FAISS.from_documents(documents, embedding)

    print(f"💽 Saving FAISS index to: {FAISS_INDEX_DIR}")
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print("✅ Embedding + Indexing complete with LangChain!")

if __name__ == "__main__":
    run_embedding_pipeline()
