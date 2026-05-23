# DocMind AI — RAG Document Q&A Chatbot 🧠

An intelligent document chatbot built with **Retrieval-Augmented Generation (RAG)**. Upload any PDF and ask questions in natural language — answers are grounded in your document, not hallucinated.

## How It Works

```
PDF → Text Chunks → Gemini Embeddings → FAISS Vector Store
                                                ↓
User Question → Embed Query → Retrieve Top-4 Chunks → Gemini LLM → Answer
```

## Features

- 📄 Upload any PDF (research papers, books, reports, manuals)
- 🔍 Semantic search via FAISS vector similarity
- 🧠 Answers grounded in document context (no hallucination)
- 💬 Multi-turn conversation with memory
- ⚡ Powered by Gemini 1.5 Flash

## Project Structure

```
rag-document-chatbot/
├── app.py            # Streamlit web app (RAG pipeline + UI)
├── requirements.txt  # Dependencies
├── .env.example      # API key template
└── README.md
```

## Setup & Run

### 1. Get a Gemini API Key
Go to [aistudio.google.com](https://aistudio.google.com) → Get API Key (free)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use the app
1. Enter your Gemini API key in the sidebar
2. Upload any PDF
3. Ask questions about the document!

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 1.5 Flash (Google) |
| Embeddings | Gemini Embedding-001 |
| Vector Store | FAISS |
| RAG Pipeline | LangChain |
| Frontend | Streamlit |
| PDF Loader | PyPDF |

## Why RAG?

Standard LLMs hallucinate when asked about specific documents. RAG fixes this by:
1. Converting document text into vector embeddings
2. Storing them in FAISS for fast similarity search
3. Retrieving the most relevant chunks for each question
4. Grounding the LLM's answer in real document content

## Author

**Kunaljit Das** — B.Tech CSE, The Assam Kaziranga University  
[LinkedIn](https://linkedin.com/in/kunaljit-das) · [GitHub](https://github.com/KunalnTech)
