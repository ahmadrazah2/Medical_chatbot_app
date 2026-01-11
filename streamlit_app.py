from pathlib import Path
import streamlit as st

from rag.config import AppConfig
from rag.loader import DataLoader
from rag.cleaner import TextCleaner
from rag.embeddings import EmbeddingFactory
from rag.vectordb import ChromaStore
from rag.llm import HuggingFaceLLM
from rag.rag_chain import RAGPipeline

import os

# ---------------------------------
# Load secrets (if available)
# ---------------------------------
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Medical Book Assistant",
    page_icon="🏥",
    layout="centered",
)

# ---------------------------------
# Custom CSS (ChatGPT-like light UI)
# ---------------------------------
st.markdown(
    """
<style>
/* Light background */
.stApp {
    background: #f7f7f8;
    color: #111827;
}

header, footer {
    background: #ffffff !important;
    color: #111827 !important;
}

/* Reduce default padding */
.block-container {
    padding-top: 2.5rem;
}

/* Title area */
.header {
    max-width: 900px;
    margin: auto;
}

/* CPU warning box */
.cpu-warning {
    background-color: #fff7ed;
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 16px;
    margin-bottom: 26px;
    font-size: 0.95rem;
    color: #9a3412;
    border: 1px solid #fed7aa;
}

/* Pills */
div[data-testid="stButton"] > button {
    border-radius: 999px !important;
    padding: 0.45rem 1rem !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #111827 !important;
    font-size: 0.9rem !important;
    white-space: normal;
}
div[data-testid="stButton"] > button:hover {
    background: #f9fafb !important;
}

/* Center chips */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 22px;
}

/* Let pills size to text */
.chips div[data-testid="stButton"] {
    width: auto !important;
    display: inline-block !important;
}

div[data-testid="stButton"] > button {
    width: auto !important;
}

/* Chat width */
.chat-area {
    max-width: 900px;
    margin: auto;
}

/* Clear button */
.clear-btn button {
    background: #ffffff !important;
    color: #111827 !important;
}

/* Toggle alignment */
.toggle-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Chat input styling */
[data-testid="stChatInput"] textarea {
    border-radius: 999px !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #111827 !important;
    caret-color: #111827 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280 !important;
}
[data-testid="stChatInput"] button {
    border-radius: 999px !important;
    background: #111827 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Chat message text */
[data-testid="stChatMessageContent"] {
    color: #111827 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------
# Build RAG once
# ---------------------------------
@st.cache_resource
def build_pipeline():
    cfg = AppConfig()

    loader = DataLoader(Path("data"))
    raw_docs = loader.load()

    cleaner = TextCleaner()
    clean_docs = cleaner.clean_docs(raw_docs)

    embeddings = EmbeddingFactory(cfg.embedding_model).build()

    store = ChromaStore(cfg.chroma_dir, cfg.collection_name)
    chunks = store.split(clean_docs, cfg.chunk_size, cfg.chunk_overlap)
    vectordb = store.build_or_load(chunks, embeddings)

    llm = HuggingFaceLLM(
        repo_id=cfg.repo_id,
    ).build()

    return RAGPipeline(vectordb, llm, cfg.k)

# ---------------------------------
# Session state
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""


# ---------------------------------
# Header
# ---------------------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown("## 🏥 Medical Chatbot")
st.caption("Ask questions about medical topics (RAG + Chroma + Local Mistral).")


st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# Medical suggestion pills (wrap to text size)
# ---------------------------------
suggestions = [
    "What is insulin resistance?",
    "Explain HbA1c in simple words.",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "What are common side effects of metformin?",
]

st.markdown('<div class="chips">', unsafe_allow_html=True)
for i, text in enumerate(suggestions):
    if st.button(text, key=f"suggest_{i}"):
        st.session_state.pending_input = text
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# Controls row
# ---------------------------------
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🧹 Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------
# Chat history
# ---------------------------------
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# Chat input
# ---------------------------------
user_input = st.chat_input("Ask a medical question ")

if not user_input and st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = ""

# ---------------------------------
# RAG pipeline
# ---------------------------------
rag = build_pipeline()

# ---------------------------------
# Run chat
# ---------------------------------
if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            answer, docs = rag.answer(user_input)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
