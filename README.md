# 📚 Multilingual Bangla-English RAG QA System

A simple, robust **Retrieval-Augmented Generation (RAG)** pipeline for answering Bangla and English questions from your own PDF knowledge base.  
Built with **FastAPI**, **LangChain**, **FAISS**, and **HuggingFace Transformers**.

---

## ✨ Features

- Accepts both Bangla and English queries
- Retrieves from your supplied Bangla PDF (e.g., textbook, exam guide)
- Modern multilingual embeddings (LaBSE)
- Robust chunking for accurate fact retrieval
- Extracts answers using LLM QA and regex fallback for factual precision
- REST API with `/query` endpoint
- Ready for frontend or chatbot integration

---

## 🚀 Quickstart

### 1. **Clone the Repository**

git clone https://github.com/yourusername/bangla-rag-qa.git
cd bangla-rag-qa

## 2. Set Up Python Environment
python -m venv venv
source venv/bin/activate         # or 'venv\Scripts\activate' on Windows
pip install -r requirements.txt

-----

fastapi
uvicorn
langchain
langchain-community
langchain-huggingface
pdfplumber
transformers
sentence-transformers
huggingface-hub
python-dotenv

-----

3. Prepare Data
Put your Bangla PDF (e.g., HSC26-Bangla1st-Paper.pdf) in the data/ folder.

----

5. Chunk the PDF
   run python app/chunker.py
----

6. Embed Chunks and Build Vector Store
python app/embedding.py

----

7. Run the FastAPI Server
uvicorn main:app --reload
Visit http://127.0.0.1:8000/docs for the API Swagger UI.
 
-----

🔍 How It Works
1. Chunking: PDF is split into overlapping, QA-friendly text chunks.

2. Embedding: Chunks are embedded with LaBSE for multilingual search.

3. Retrieval: On query, top-k similar chunks are retrieved via FAISS.

4. QA Answering: A QA model answers from the chunks. If it fails, a regex fallback tries to extract numbers/facts from the text.

5. API: /query endpoint returns the answer and source chunks.
----
👤 Author
Antu Saha


