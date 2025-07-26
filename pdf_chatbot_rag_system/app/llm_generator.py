import logging
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.language_utils import detect_language, translate_bn_to_en, translate_en_to_bn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load English QA pipeline once
logger.info("[🤖] Loading English QA model (roberta-base-squad2)...")
qa_en_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

def generate_answer(query: str, chunks, lang: str = "en") -> str:
    """
    Generate a grounded answer from provided chunks.
    For Bangla queries, uses Bangla → English translation for QA, then translates back.
    """
    logger.info(f"[🔍] Generating answer for: {query} ({lang})")

    if not chunks:
        return "প্রাসঙ্গিক তথ্য পাওয়া যায়নি।" if lang == "bn" else "No relevant context found."

    # Combine top chunks into context (limit to 3500 characters)
    context = " ".join([doc.page_content for doc in chunks[:2]])[:1500]

    try:
        if lang == "bn":
            query_en = translate_bn_to_en(query)
            context_en = translate_bn_to_en(context)
            logger.info(f"[→EN] Q: {query_en}")
            logger.info(f"[→EN] Context: {context_en[:256]}...")

            result = qa_en_pipeline(question=query_en, context=context_en)
            answer_en = result.get("answer", "").strip()
            logger.info(f"[←EN] Extracted English answer: {answer_en}")

            answer_bn = translate_en_to_bn(answer_en)
            logger.info(f"[←BN] Translated Bangla answer: {answer_bn}")
            return answer_bn or "⚠️ কোনো উত্তর পাওয়া যায়নি।"

        else:
            result = qa_en_pipeline(question=query, context=context)
            answer = result.get("answer", "").strip()
            logger.info(f"[EN] English answer: {answer}")
            return answer or "⚠️ Could not generate answer."

    except Exception as e:
        logger.error(f"[QA/Translation Error]: {e}")
        return "⚠️ An error occurred while generating the answer."
