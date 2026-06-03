# PDF RAG Chatbot

A RAG-based PDF Q&A chatbot built with FastAPI, LangChain, OpenAI, and ChromaDB.

## Tech Stack
- FastAPI — backend API
- LangChain — RAG pipeline
- OpenAI GPT-3.5 — language model
- ChromaDB — vector database
- React.js — frontend (coming soon)

## How it works
1. Upload any PDF
2. System chunks and embeds it into ChromaDB
3. Ask questions in natural language
4. System retrieves relevant chunks and answers using GPT

## Setup
```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn main:app --reload
```

## API Endpoints
- POST /upload-pdf → upload PDF
- POST /ask → ask question
- GET / → health check