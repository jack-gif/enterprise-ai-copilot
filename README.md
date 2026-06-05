# Enterprise AI Copilot 🚀

An enterprise-grade AI system built using LLMs, RAG, and multi-agent orchestration.

## 🔹 Features

- Retrieval-Augmented Generation (RAG) using FAISS
- Gemini LLM integration
- Multi-tool AI Agent using LangGraph
- Semantic search with embeddings
- Conversational memory support
- LLM observability using LangSmith
- FastAPI-based scalable backend

## 🔹 Architecture

User Query → Agent → Decision (RAG / API) → Response

## 🔹 Tech Stack

- Python, FastAPI
- LangChain, LangGraph
- Google Gemini (LLM + Embeddings)
- FAISS (Vector DB)
- LangSmith (Tracing & Observability)

## 🔹 Use Cases

- Enterprise knowledge assistant
- Policy/document Q&A
- Transaction monitoring assistant
- AI-powered copilots

## 🔹 How to Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🔹 How to Test Copilot

- http://127.0.0.1:8000/ask?query=how%20much%20required%20for%20dual%20approval
