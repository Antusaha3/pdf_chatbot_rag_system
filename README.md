
# 📚 Multilingual Retrieval-Augmented Generation (RAG) System

## 🔍 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient multilingual document processing and intelligent question answering. It enables querying Bangla-language educational content using both **Bangla** and **English**, powered by semantic search and large language models.

The pipeline performs:
- **Text extraction** from Bangla PDFs using advanced document parsing
- **Semantic indexing** using sentence embeddings
- **Vector-based retrieval** using PGVector and Redis
- **Context-aware answer generation** via a GPT model

---

## 📖 About This Project

This project was developed as part of a Level-1 AI Engineer Technical Assessment and demonstrates the ability to design and implement a multilingual **Retrieval-Augmented Generation (RAG)** system tailored for educational content in Bangla. The objective was to build an end-to-end pipeline that could ingest Bangla-language PDF textbooks, extract and structure relevant information, and enable **semantic question answering** in both **Bangla and English**.

### 🎯 Key Goals:
- Build a knowledge base from a Bangla PDF corpus (HSC26 Bangla 1st Paper).
- Allow users to query the system in Bangla or English.
- Ensure answers are **grounded in retrieved context**, not hallucinated.
- Maintain long-term memory from the document corpus and short-term conversational memory.
- Provide both a **developer API** (FastAPI) and an **interactive UI** (Streamlit) for end-user access.

### 🧠 Why This Project Matters
Many educational institutions in Bangladesh rely on printed or scanned textbooks in Bangla. However, digital tools that support **automated question answering** in Bangla remain underdeveloped. By applying RAG principles and combining them with multilingual NLP models (like LaBSE and GPT), this system enables:
- **Contextually accurate answers**
- **Multilingual support**
- **Scalable architecture for educational AI applications**

This work also serves as a foundation for more advanced NLP applications in the Bangla language, such as:
- Intelligent tutoring systems
- Language translation evaluation
- Educational content search engines

---

## 🖼️ System Architecture

```
📄 Bangla PDF (HSC Book)
    |
    ↓
🧹 Preprocessing & Cleaning (Regex + fitz)
    |
    ↓
✂️ Chunking (MCQ + Paragraphs)
    |
    ↓
🔢 Embedding (LaBSE)
    |
    ↓
📦 Storage
    ├── Redis (Raw PDFs)
    └── PGVector (Embeddings in PostgreSQL)
    |
    ↓
🔎 Retrieval (MultiVector Retriever)
    |
    ↓
🤖 LLM Querying (GPT via OpenAI or local model)
```

---

## 🚀 Features

- **📄 Unstructured Document Processing**  
  Extracts meaningful content including **text and tables** from Bangla PDFs.

- **🗃️ Redis for Raw Storage**  
  Stores raw PDFs in a fast, in-memory cache with persistence, enabling quick retrieval and debugging.

- **📊 PGVector for Semantic Search**  
  High-dimensional embedding vectors are stored and queried using **PostgreSQL + PGVector** for robust similarity matching.

- **🔁 MultiVector Retriever**  
  Retrieves the most relevant document chunks across contexts and languages to ensure accurate grounding.

- **🧠 LLM Integration**  
  Uses a **GPT model** (OpenAI API or local) to generate answers grounded in the retrieved Bangla/English context.

---

## 🛠️ Tech Stack

### 🔤 Programming Language
- Python

### 🧰 Libraries & Frameworks
- `unstructured`
- `pgvector`
- `redis`
- `langchain`
- `openai`
- `fitz` (PyMuPDF)
- `transformers`

### 🗄️ Databases
- **Redis**: For storing raw PDF documents.
- **PostgreSQL + PGVector**: For storing and querying semantic vector embeddings.

### 🤖 LLMs
- **GPT (via OpenAI API)** – for answer generation.
- Future support for **local LLMs** (e.g., LLama or Mistral) is planned.

---

## 📬 Usage

1. **Extract PDF content:**  
   Run `chunker.py` to parse and clean the Bangla textbook into short-answer and paragraph chunks.

2. **Vectorize Chunks:**  
   Use `vector.py` to embed the text using LaBSE and store it in PGVector.

3. **Ask a Question (via API):**  
   Run FastAPI app or Streamlit interface and ask a question like:  
   > `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`  
   Answer: `শুম্ভুনাথ`

---

## 📈 Future Enhancements

- Fine-tuned **Bangla LLM** for direct question answering.
- Better OCR for scanned textbook PDFs.
- Support for **cross-document reasoning** across multiple PDFs.
- Integration with **N8n or LangFlow** for no-code RAG workflows.

---

## 🤝 Contributing

Pull requests are welcome! Feel free to suggest improvements to chunking logic, evaluation metrics, or translation pipelines.

---

## 📝 License

MIT License © 2025  
Developed as part of an AI Engineering Level-1 Technical Assessment.


