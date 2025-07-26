import os
import logging
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from transformers import pipeline, T5Tokenizer, AutoModelForSeq2SeqLM

# ---------- Setup ----------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HUGGINGFACE_TOKEN is not set in .env or environment variables.")

DetectorFactory.seed = 42
SUPPORTED_LANGS = {"en", "bn"}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Model Config ----------
BN_EN_MODEL = "csebuetnlp/banglat5_nmt_bn_en"
EN_BN_MODEL = "csebuetnlp/banglat5_nmt_en_bn"

_bn2en_pipeline = None
_en2bn_pipeline = None

# ---------- Language Detection ----------
def detect_language(text: str) -> str:
    try:
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 2:
            logger.warning("[LangDetect] Input too short for language detection.")
            return "unknown"

        lang = detect(cleaned)
        return lang if lang in SUPPORTED_LANGS else "unknown"
    except Exception as e:
        logger.error(f"[LangDetect Error] {e}")
        return "unknown"

# ---------- Translation Model Loaders ----------
def load_bn2en():
    global _bn2en_pipeline
    if _bn2en_pipeline is None:
        logger.info("🔁 Loading Bangla → English model...")
        tokenizer = T5Tokenizer.from_pretrained(BN_EN_MODEL, token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(BN_EN_MODEL, token=HF_TOKEN)
        _bn2en_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return _bn2en_pipeline

def load_en2bn():
    global _en2bn_pipeline
    if _en2bn_pipeline is None:
        logger.info("🔁 Loading English → Bangla model...")
        tokenizer = T5Tokenizer.from_pretrained(EN_BN_MODEL, token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(EN_BN_MODEL, token=HF_TOKEN)
        _en2bn_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return _en2bn_pipeline

# ---------- Translation Functions ----------
def translate_bn_to_en(text: str) -> str:
    try:
        pipeline_bn2en = load_bn2en()
        result = pipeline_bn2en(text.strip(), max_length=256, clean_up_tokenization_spaces=True)
        return result[0]["generated_text"].strip()
    except Exception as e:
        logger.warning(f"[⚠️ Translation Error] BN → EN: {e}")
        return text

def translate_en_to_bn(text: str) -> str:
    try:
        pipeline_en2bn = load_en2bn()
        result = pipeline_en2bn(text.strip(), max_length=256, clean_up_tokenization_spaces=True)
        return result[0]["generated_text"].strip()
    except Exception as e:
        logger.warning(f"[⚠️ Translation Error] EN → BN: {e}")
        return text

# ---------- General Translator ----------
def translate(text: str, direction: str = "bn2en") -> str:
    if direction == "bn2en":
        return translate_bn_to_en(text)
    elif direction == "en2bn":
        return translate_en_to_bn(text)
    else:
        raise ValueError(f"Unsupported translation direction: {direction}")


