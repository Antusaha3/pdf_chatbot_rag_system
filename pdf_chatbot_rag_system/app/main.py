import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from fastapi import File, UploadFile
import json
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

from app.language_utils import detect_language  # <- use your actual module name
from app.vector_store import load_faiss_retriever
from app.llm_generator import generate_answer

# ---------- App Setup ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Models ----------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    language: str
    answer: str
    source_chunks: List[str]

# ---------- Load Retriever ----------
retriever = load_faiss_retriever()

# ---------- Health Check ----------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ---------- Endpoint ----------
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question provided.")

    logger.info(f"\n[🟡 Incoming Question]: {question}")

    lang = detect_language(question)
    logger.info(f"[🌐 Detected Language]: {lang}")

    try:
        docs = retriever.invoke(question)
        logger.info(f"[🔍 Retrieved {len(docs)} documents]")
    # 🔍 Print the full retrieved chunk contents for debugging
        for i, doc in enumerate(docs):
            print(f"\n[🔎 Chunk {i+1}]\n{doc.page_content}\n")

        # 🔎 Log the retrieved chunks for debugging (first 200 chars)
        for i, doc in enumerate(docs):
            logger.info(f"[🔎 Chunk {i+1}]: {doc.page_content[:200]}")

        if not docs:
            logger.warning("[⚠️ No relevant documents found]")
            return QueryResponse(
                question=question,
                language=lang,
                answer="প্রাসঙ্গিক কোনো তথ্য পাওয়া যায়নি।" if lang == "bn" else "No relevant context found.",
                source_chunks=[]
            )

        answer = generate_answer(query=question, chunks=docs, lang=lang)
        logger.info(f"[✅ Answer]: {answer}")

        return QueryResponse(
            question=question,
            language=lang,
            answer=answer,
            source_chunks=[doc.page_content for doc in docs]
        )

    except Exception as e:
        logger.exception("[❌ ERROR]")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
@app.post("/evaluate")
async def evaluate_rag(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        test_data = json.loads(contents.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    total = len(test_data)
    correct = 0
    evaluations = []

    for entry in test_data:
        question = entry.get("question")
        expected_answer = entry.get("expected_answer")
        lang = detect_language(question)
        docs = retriever.invoke(question)
        predicted = generate_answer(query=question, chunks=docs, lang=lang)

        # Basic fuzzy check (case-insensitive containment)
        is_correct = expected_answer.lower() in predicted.lower()
        if is_correct:
            correct += 1

        evaluations.append({
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": predicted,
            "matched": is_correct,
            "retrieved_chunks": [doc.page_content[:200] for doc in docs]
        })

    accuracy = round((correct / total) * 100, 2)

    return JSONResponse(content={
        "total_questions": total,
        "correct_answers": correct,
        "accuracy (%)": accuracy,
        "evaluations": evaluations
    })
