from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.language_detect import detect_language
from app.vector_store import retrieve_similar_chunks
from app.llm_generator import generate_answer
import logging
from time import time

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_CHUNKS = 5

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    language: str
    retrieved_chunks: list

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Received query: {query}")
    start_time = time()

    try:
        lang = detect_language(query)
        logger.info(f"Detected language: {lang}")

        chunks = retrieve_similar_chunks(query)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        logger.info(f"Retrieved {len(chunks)} chunks")

        answer = generate_answer(query, chunks, lang=lang)
        logger.info(f"Generated answer: {answer}")

    except Exception as e:
        logger.exception("RAG processing error")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=answer,
        language=lang,
        retrieved_chunks=chunks[:MAX_CHUNKS]
    )
