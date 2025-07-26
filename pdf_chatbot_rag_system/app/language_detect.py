from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
import re

# Ensure consistent results
DetectorFactory.seed = 42

SUPPORTED_LANGS = {"en", "bn"}

logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns:
        'bn' for Bangla, 'en' for English, or 'unknown' if not supported.
    """
    try:
        cleaned_text = text.strip()
        if not cleaned_text or len(cleaned_text) < 2 or not re.search(r'\w', cleaned_text):
            logger.warning("[LangDetect] Empty or invalid input received.")
            return "unknown"

        lang = detect(cleaned_text)

        if lang in SUPPORTED_LANGS:
            logger.info(f"[LangDetect] Detected: {lang}")
            return lang
        else:
            logger.warning(f"[LangDetect] Detected unsupported language: {lang}")
            return "unknown"
    except LangDetectException as e:
        logger.error(f"[LangDetect Error] Failed to detect language: {e}")
        return "unknown"
