import streamlit as st
import requests

# --------------- CONFIG ---------------
API_URL = "http://localhost:8000/query"
st.set_page_config(page_title="📚 Bangla-English RAG Chatbot", page_icon="🤖", layout="centered")
# --------------------------------------

st.title("📚 Bangla-English RAG Chatbot")
st.markdown("Ask a question from the **Bangla textbook PDF** and get answers using AI-powered retrieval.")

# Sidebar example questions
with st.sidebar:
    st.subheader("💡 Example Questions")
    examples = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    for ex in examples:
        if st.button(ex):
            st.session_state["question"] = ex

# --- Input ---
question = st.text_input("❓ Enter your question", value=st.session_state.get("question", ""), key="input")

if st.button("🔍 Get Answer") and question:
    with st.spinner("Thinking..."):
        try:
            res = requests.post(API_URL, json={"question": question})
            res.raise_for_status()
            data = res.json()

            # --- Show Answer ---
            st.success("✅ Answer generated successfully!")
            st.markdown(f"""
                <div style="background-color:#1a1a1a;padding:20px 20px;border-radius:8px;margin-top:10px;color:#fff">
                    <h4 style="color:#fff">📖 Answer ({data['language']})</h4>
                    <p style="font-size:20px;color:#0bf">{data['answer']}</p>
                </div>
            """, unsafe_allow_html=True)

            # --- Show Retrieved Chunks ---
            if data["source_chunks"]:
                st.markdown("### 📚 Retrieved Context")
                for i, chunk in enumerate(data["source_chunks"], 1):
                    with st.expander(f"Chunk {i}"):
                        st.markdown(chunk)

        except Exception as e:
            st.error(f"Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Powered by FastAPI, LangChain, FAISS, HuggingFace, and Streamlit | © 2025")
