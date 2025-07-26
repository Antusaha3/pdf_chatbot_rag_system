# app/chunker.py

import fitz  # PyMuPDF
import re
import json
import os
from typing import List

# ---------- CONFIG ----------
PDF_PATH = "D:/pdf_chatbot_rag_system/data/HSC26-Bangla1st-Paper.pdf"
OUTPUT_JSON_PATH = "data/chunks.json"
# ----------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all raw text from the PDF.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def clean_text(text: str) -> str:
    """
    Remove unnecessary symbols and normalize spacing.
    """
    text = re.sub(r'[^\u0980-\u09FFA-Za-z0-9।,.!?()\[\]{}\'" \n]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_mcq_qa_pairs(text: str) -> List[str]:
    """
    Extract MCQ-style Bangla QA chunks.
    Format: "৪১। কাকে অনুপমের ভাগ্য দেবতা বলা হয়েছে? (ক) পিতা (খ) ভাই (গ) মামা (ঘ) শিক্ষক"
    Output: "কাকে অনুপমের ভাগ্য দেবতা বলা হয়েছে: মামা"
    """
    mcq_lines = re.findall(r'\d+[।.][^\n]+', text)
    option_pattern = re.compile(r'[কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়]+[)]\s*([^()]+)')

    qa_chunks = []
    for line in mcq_lines:
        parts = line.split('(')
        question_part = parts[0].strip()
        options_joined = '(' + '('.join(parts[1:]) if len(parts) > 1 else ""
        options = option_pattern.findall(options_joined)

        if question_part and options:
            # Use shortest option as heuristic
            answer = min(options, key=len)
            qa_chunks.append(f"{question_part}: {answer.strip()}")
    return qa_chunks

def split_paragraphs(text: str) -> List[str]:
    """
    Break cleaned text into paragraph-level chunks (30+ characters).
    """
    paras = re.split(r'[।\n]', text)
    paras = [p.strip() for p in paras if len(p.strip()) > 30]
    return paras

def run_chunking_pipeline():
    print("📄 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("🧼 Cleaning text...")
    cleaned = clean_text(raw_text)

    print("✂️ Extracting chunks...")
    qa_chunks = extract_mcq_qa_pairs(cleaned)
    para_chunks = split_paragraphs(cleaned)

    # Combine and deduplicate
    all_chunks = list(set(qa_chunks + para_chunks))

    print(f"💾 Saving {len(all_chunks)} chunks to {OUTPUT_JSON_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(
            [{"text": chunk, "metadata": {}} for chunk in all_chunks],
            f,
            ensure_ascii=False,
            indent=2
        )

    # Preview
    print("📌 Sample Chunks:")
    for i, chunk in enumerate(all_chunks[:5]):
        print(f"[Sample {i+1}] {chunk}")

    print("✅ Chunking complete.")

if __name__ == "__main__":
    run_chunking_pipeline()
