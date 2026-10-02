# DocMind: RAG Document Q&A Chatbot

Upload a PDF and ask questions. Answers are grounded in retrieved chunks of your
document and cite the pages they came from.

**Stack:** LangChain · FAISS · Google Gemini · Hugging Face embeddings · Streamlit

## How it works

1. `PyPDFLoader` reads the PDF; `RecursiveCharacterTextSplitter` cuts it into
   1000-character chunks with 200 overlap.
2. Chunks are embedded locally with `all-MiniLM-L6-v2` (no API key) and stored in FAISS.
3. For each question, the 4 most similar chunks are retrieved.
4. Gemini answers using only those chunks plus the last few turns of chat.
   If the answer is not in the document, it says so.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your key from aistudio.google.com
streamlit run app.py
```

You can also paste the key into the sidebar instead of using `.env`.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `rag.py` | Loading, chunking, FAISS index, Gemini call |
| `styles.css` | Page styling |

## Limitations

- Text-based PDFs only; scanned images need OCR first.
- One document at a time; the index lives in memory for the session.
- Model names change. If you get "model not found", set `GEMINI_MODEL` in `.env`.
