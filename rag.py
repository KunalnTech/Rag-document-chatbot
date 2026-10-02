"""RAG core: PDF -> chunks -> FAISS index -> Gemini answer with page citations."""

import os
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
HISTORY_TURNS = 6

PROMPT = """You answer questions using only the document excerpts below.
If the answer is not in the excerpts, say you could not find it in the document.
Do not use outside knowledge.

Document excerpts:
{context}

Conversation so far:
{history}

Question: {question}

Answer:"""


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local embedding model, loaded once. No API key needed."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(pdf_path: str) -> tuple[FAISS, int, int]:
    """Index a PDF. Returns (vectorstore, page_count, chunk_count)."""
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No text found in this PDF. It may be a scan; try a text-based PDF."
        )
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(pages), len(chunks)


def _to_text(content) -> str:
    """Gemini can return a string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else p.get("text", "") for p in content
        )
    return str(content)


def answer_question(question: str, store: FAISS, history: list[dict], api_key: str):
    """Return (answer_text, sorted_page_numbers) for a question."""
    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )
    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-HISTORY_TURNS:]
    ) or "(none)"

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, temperature=0.2, google_api_key=api_key
    )
    prompt = PROMPT.format(context=context, history=history_text, question=question)
    reply = llm.invoke([HumanMessage(content=prompt)])
    return _to_text(reply.content).strip(), pages
