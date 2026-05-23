"""
RAG Document Q&A Chatbot
LangChain + FAISS + Groq + Streamlit
Run: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #07070f; color: #e8e8f0; }
header[data-testid="stHeader"] { background: transparent; }
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #07070f 60%, #1a0d1a 100%);
    border: 1px solid #1a1a2e; border-radius: 24px;
    padding: 3rem 2rem; margin-bottom: 2rem;
    position: relative; overflow: hidden; text-align: center;
}
.hero::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 25% 50%, rgba(168,85,247,0.06) 0%, transparent 45%),
                radial-gradient(circle at 75% 50%, rgba(0,200,255,0.06) 0%, transparent 45%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 10px; color: #a855f7; border: 1px solid #a855f740;
    background: #a855f710; padding: 5px 14px; border-radius: 999px;
    margin-bottom: 1.2rem; letter-spacing: 3px; text-transform: uppercase;
}
.hero h1 {
    font-size: 3.5rem; font-weight: 800; margin: 0.4rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #a855f7 50%, #00c8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.05;
}
.hero-sub { color: #555570; font-size: 1rem; margin-top: 0.8rem; }
.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2rem; }
.stat-card {
    background: #0d0d1a; border: 1px solid #1a1a2e; border-radius: 16px;
    padding: 1.2rem; text-align: center; position: relative; overflow: hidden;
}
.stat-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #a855f760, transparent);
}
.stat-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #a855f7; line-height: 1; }
.stat-label { font-size: 0.72rem; color: #444460; margin-top: 5px; letter-spacing: 2px; text-transform: uppercase; }
.section-label {
    font-family: 'Space Mono', monospace; font-size: 10px; color: #444460;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #1a1a2e; }
.chat-bubble-user {
    background: linear-gradient(135deg, #a855f720, #00c8ff10);
    border: 1px solid #a855f730; border-radius: 16px 16px 4px 16px;
    padding: 12px 16px; margin: 8px 0; margin-left: 20%;
    font-size: 14px; color: #e0e0f0; line-height: 1.6;
}
.chat-bubble-ai {
    background: #0d0d1a; border: 1px solid #1a1a2e;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px; margin: 8px 0; margin-right: 20%;
    font-size: 14px; color: #c0c0d8; line-height: 1.6;
}
.chat-label-user { font-family: 'Space Mono', monospace; font-size: 10px; color: #a855f7; letter-spacing: 2px; text-align: right; margin-bottom: 2px; }
.chat-label-ai { font-family: 'Space Mono', monospace; font-size: 10px; color: #444460; letter-spacing: 2px; margin-bottom: 2px; }
section[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid #1a1a2e !important; }
.stButton > button {
    background: linear-gradient(135deg, #a855f7, #00c8ff) !important;
    color: #07070f !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 12px !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 2rem !important; letter-spacing: 1px !important;
}
.footer { text-align: center; font-family: 'Space Mono', monospace; font-size: 10px; color: #222235; padding: 2rem 0 1rem; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 RAG · LangChain · Groq · FAISS</div>
    <h1>DocMind AI</h1>
    <p class="hero-sub">Upload any PDF — ask questions, get instant answers grounded in your document</p>
</div>
<div class="stats-row">
    <div class="stat-card"><div class="stat-value">RAG</div><div class="stat-label">Architecture</div></div>
    <div class="stat-card"><div class="stat-value">FAISS</div><div class="stat-label">Vector Search</div></div>
    <div class="stat-card"><div class="stat-value">Groq</div><div class="stat-label">LLM Backend</div></div>
    <div class="stat-card"><div class="stat-value">∞</div><div class="stat-label">Page Support</div></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")
st.sidebar.markdown("---")
st.sidebar.markdown("""<div style='font-family:Space Mono,monospace;font-size:10px;color:#444460;letter-spacing:1px;line-height:2'>
BUILT BY<br><span style='color:#a855f7'>KUNALJIT DAS</span><br>B.TECH CSE · AKTU<br>LANGCHAIN · GROQ · RAG</div>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Build vectorstore (free HuggingFace embeddings — no API key needed) ───────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(pdf_path):
    loader   = PyPDFLoader(pdf_path)
    pages    = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(pages)
    embeddings  = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ── Answer function ───────────────────────────────────────────────────────────
def get_answer(question, vectorstore, chat_history, api_key):
    docs    = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.
Always base your answers on the context below. If the answer is not in the context, say so clearly.

Document Context:
{context}

Conversation History:
{history_text}

User Question: {question}

Answer:"""

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=api_key)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ── Main UI ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_pdf and api_key:
        if st.session_state.doc_name != uploaded_pdf.name:
            with st.spinner("🧠 Reading & indexing document…"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tfile.write(uploaded_pdf.read())
                tfile.close()
                try:
                    st.session_state.vectorstore  = build_vectorstore(tfile.name)
                    st.session_state.doc_name     = uploaded_pdf.name
                    st.session_state.chat_history = []
                    st.success(f"✅ **{uploaded_pdf.name}** ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tfile.name)
    elif uploaded_pdf and not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        st.markdown("""
        <div style="border:2px dashed #1a1a2e;border-radius:20px;padding:3rem 2rem;text-align:center;color:#333350">
            <div style="font-size:2.5rem;margin-bottom:1rem">📄</div>
            <div style="font-family:Space Mono,monospace;font-size:11px;letter-spacing:2px">DROP A PDF TO BEGIN</div>
            <div style="font-size:12px;margin-top:0.5rem;color:#222235">Research papers · Books · Reports · Manuals</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.doc_name:
        st.markdown(f"""<div style="background:#a855f710;border:1px solid #a855f730;border-radius:10px;padding:10px 14px;margin-top:12px;font-size:12px;color:#a855f7;font-family:Space Mono,monospace">
            📄 {st.session_state.doc_name}</div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear & Upload New"):
            st.session_state.vectorstore  = None
            st.session_state.doc_name     = None
            st.session_state.chat_history = []
            st.rerun()

with col_right:
    st.markdown('<div class="section-label">Chat with your Document</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div style="text-align:center;padding:3rem 1rem;color:#333350">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="font-family:Space Mono,monospace;font-size:11px;letter-spacing:2px">UPLOAD A PDF TO START CHATTING</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-label-user">YOU</div><div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label-ai">DOCMIND AI</div><div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    question = st.text_input("", placeholder="Ask anything about your document…", label_visibility="collapsed")

    col_btn1, col_btn2 = st.columns([3,1])
    with col_btn1:
        ask = st.button("Ask DocMind →", use_container_width=True)
    with col_btn2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask and question:
        if not st.session_state.vectorstore:
            st.warning("Please upload a PDF and enter your API key first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("🧠 Thinking…"):
                try:
                    answer = get_answer(question, st.session_state.vectorstore, st.session_state.chat_history, api_key)
                except Exception as e:
                    answer = f"Error: {e}"
            st.session_state.chat_history.append({"role": "ai", "content": answer})
            st.rerun()

st.markdown("""<div class="footer">DOCMIND AI · RAG + LANGCHAIN + GROQ · BUILT BY KUNALJIT DAS · ASSAM KAZIRANGA UNIVERSITY</div>
""", unsafe_allow_html=True)
