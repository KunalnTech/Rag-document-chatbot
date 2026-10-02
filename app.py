"""DocMind: chat with a PDF.  LangChain + FAISS + Gemini + Streamlit.
Run: streamlit run app.py
"""

import html
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import GEMINI_MODEL, answer_question, build_vectorstore

load_dotenv()

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")
css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
for key, default in {
    "history": [],
    "store": None,
    "doc": None,   # dict: name, pages, chunks, sig
}.items():
    st.session_state.setdefault(key, default)


def reset_document():
    st.session_state.store = None
    st.session_state.doc = None
    st.session_state.history = []


def render_message(msg: dict):
    who = "You" if msg["role"] == "user" else "DocMind"
    cls = "user" if msg["role"] == "user" else "ai"
    body = html.escape(msg["content"]).replace("\n", "<br>")
    src = ""
    if msg.get("pages"):
        pages = ", ".join(str(p) for p in msg["pages"])
        src = f'<div class="sources">From page {pages}</div>'
    st.markdown(
        f'<div class="who {cls}">{who}</div>'
        f'<div class="bubble {cls}">{body}{src}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Create one free in Google AI Studio. Or set GOOGLE_API_KEY in .env.",
    )
    st.caption(f"Model: `{GEMINI_MODEL}`")
    st.divider()
    st.caption("Built by Kunaljit Das")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><h1>DocMind</h1>'
    "<p>Upload a PDF and ask questions. Answers come only from your document, "
    "with the pages they were found on.</p></div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.7], gap="large")

# ── Left: document ────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="label">Document</div>', unsafe_allow_html=True)
    pdf = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if pdf is not None:
        sig = (pdf.name, pdf.size)
        doc = st.session_state.doc
        if doc is None or doc["sig"] != sig:
            with st.spinner("Reading and indexing the PDF..."):
                path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                        t.write(pdf.getvalue())
                        path = t.name
                    store, n_pages, n_chunks = build_vectorstore(path)
                    st.session_state.store = store
                    st.session_state.doc = {
                        "name": pdf.name, "pages": n_pages,
                        "chunks": n_chunks, "sig": sig,
                    }
                    st.session_state.history = []
                except Exception as e:
                    reset_document()
                    st.error(f"Could not index this PDF: {e}")
                finally:
                    if path and os.path.exists(path):
                        os.unlink(path)
    elif st.session_state.doc is not None:
        reset_document()  # uploader was cleared with the X

    doc = st.session_state.doc
    if doc:
        st.markdown(
            f'<div class="docinfo"><b>{html.escape(doc["name"])}</b><br>'
            f'{doc["pages"]} pages · {doc["chunks"]} searchable sections</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Text-based PDFs only. Scanned images have no text to search.")

# ── Right: chat ───────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="label">Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div class="empty">No questions yet. Upload a PDF, then ask '
            "something like “What is the main conclusion?”</div>",
            unsafe_allow_html=True,
        )
    for m in st.session_state.history:
        render_message(m)

    with st.form("ask", clear_on_submit=True):
        question = st.text_input(
            "Question",
            placeholder="Ask about your document",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([3, 1])
        ask = c1.form_submit_button("Ask", use_container_width=True)
        clear = c2.form_submit_button("Clear chat", use_container_width=True)

    if clear:
        st.session_state.history = []
        st.rerun()

    if ask and question.strip():
        if st.session_state.store is None:
            st.warning("Upload a PDF first.")
        elif not api_key:
            st.warning("Enter your Gemini API key in the sidebar.")
        else:
            prior = list(st.session_state.history)  # excludes the new question
            with st.spinner("Searching the document..."):
                try:
                    text, pages = answer_question(
                        question.strip(), st.session_state.store, prior, api_key
                    )
                except Exception as e:
                    st.error(f"Gemini request failed: {e}")
                else:
                    st.session_state.history += [
                        {"role": "user", "content": question.strip()},
                        {"role": "ai", "content": text, "pages": pages},
                    ]
                    st.rerun()
