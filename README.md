# FastAPI RAG API

A production-grade **Retrieval-Augmented Generation (RAG)** REST API built with FastAPI, LangChain, ChromaDB, and OpenAI. Upload PDF documents and query them in natural language — the API retrieves the most relevant context and generates accurate, grounded answers.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

This API implements a full RAG pipeline in four steps:

1. **Ingest** — Upload a PDF; the file is parsed and split into overlapping text chunks.
2. **Embed** — Each chunk is converted into a vector using a local SentenceTransformer model (`all-MiniLM-L6-v2`).
3. **Store** — Vectors are persisted in a local ChromaDB collection.
4. **Retrieve & Generate** — At query time, the question is embedded, the most similar chunks are retrieved, and OpenAI generates a grounded answer from that context.

---

## Architecture

```
┌─────────────┐     PDF Upload      ┌──────────────────────────────────────┐
│   Client    │ ─────────────────►  │            FastAPI Layer             │
│  (curl /    │                     │  POST /api/v1/upload                 │
│  Swagger)   │ ◄─────────────────  │  POST /api/v1/query                  │
└─────────────┘     JSON Response   │  GET  /api/v1/documents              │
                                    │  DELETE /api/v1/clear                │
                                    └────────────────┬─────────────────────┘
                                                     │
                          ┌──────────────────────────▼──────────────────────────┐
                          │                  Core Pipeline                      │
                          │                                                     │
                          │  ┌─────────────┐    ┌──────────────┐               │
                          │  │PyMuPDF      │    │Sentence      │               │
                          │  │Loader       │───►│Transformer   │               │
                          │  │(PDF Parser) │    │(Embeddings)  │               │
                          │  └─────────────┘    └──────┬───────┘               │
                          │                            │                       │
                          │                    ┌───────▼───────┐               │
                          │                    │   ChromaDB    │               │
                          │                    │ (Vector Store)│               │
                          │                    └───────┬───────┘               │
                          │                            │                       │
                          │                    ┌───────▼───────┐               │
                          │                    │ RAGRetriever  │               │
                          │                    │ (Top-K Docs)  │               │
                          │                    └───────┬───────┘               │
                          │                            │                       │
                          │                    ┌───────▼───────┐               │
                          │                    │ RAGGenerator  │               │
                          │                    │  (OpenAI LLM) │               │
                          │                    └───────────────┘               │
                          └─────────────────────────────────────────────────────┘
```

---

## Features

- **PDF ingestion** — Upload any PDF; automatic parsing, chunking, and embedding
- **Semantic search** — Cosine-similarity retrieval via ChromaDB with configurable result count and score threshold
- **Grounded generation** — Answers are generated strictly from retrieved context, minimizing hallucination
- **Local embeddings** — `all-MiniLM-L6-v2` runs on-device; no embedding API calls or costs
- **Persistent vector store** — ChromaDB persists embeddings to disk across restarts
- **Structured responses** — All endpoints return fully typed Pydantic models
- **Interactive docs** — Swagger UI at `/docs`, ReDoc at `/redoc`
- **Structured logging** — Timestamp-formatted log output with per-module loggers

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ASGI server | [Uvicorn](https://www.uvicorn.org/) |
| LLM | [OpenAI](https://platform.openai.com/) (`gpt-3.5-turbo`) |
| Embeddings | [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Document loading | [LangChain](https://python.langchain.com/) + PyMuPDF |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Environment config | python-dotenv |

---

## Project Structure

```
fastapi-rag-api/
├── app/
│   ├── main.py               # FastAPI app factory, middleware, metadata
│   ├── api/
│   │   └── routes.py         # All API endpoint definitions
│   └── core/
│       ├── embeddings.py     # EmbeddingManager — SentenceTransformer wrapper
│       ├── vectorstore.py    # VectorStore — ChromaDB CRUD operations
│       ├── retriever.py      # RAGRetriever — semantic search
│       └── generator.py      # RAGGenerator — context assembly + OpenAI call
├── data/
│   ├── pdf_files/            # Uploaded PDFs (auto-created)
│   └── vector_store/         # ChromaDB persistence directory (auto-created)
├── requirements.txt
├── .env                      # Environment variables (not committed)
└── README.md
```

---

## Prerequisites

- Python **3.10+**
- An **OpenAI API key** — [get one here](https://platform.openai.com/api-keys)
- `pip` and `venv`

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/saif321761/fastapi-rag-api.git
cd fastapi-rag-api
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB) automatically.

---

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

> **Never commit `.env` to version control.** It is already listed in `.gitignore`.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI secret key |

---

## Running the API

```bash
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`.

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc documentation |
| `http://localhost:8000/health` | Health check |

---

## API Reference

### `POST /api/v1/upload`

Upload a PDF file and index it into the vector store.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | A `.pdf` file |

**Response** `201 Created`

```json
{
  "message": "report.pdf uploaded and processed successfully.",
  "pages": 12,
  "chunks": 47,
  "total_docs": 47
}
```

**Error responses**

| Status | Condition |
|---|---|
| `415 Unsupported Media Type` | File is not a PDF |
| `500 Internal Server Error` | Parsing or embedding failure |

---

### `POST /api/v1/query`

Submit a natural-language question and receive a generated answer.

**Request body** — `application/json`

```json
{
  "query": "What are the main risk factors mentioned in the report?",
  "n_results": 3
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The question to ask |
| `n_results` | integer | `3` | Number of chunks to retrieve (1–20) |

**Response** `200 OK`

```json
{
  "query": "What are the main risk factors mentioned in the report?",
  "answer": "The report identifies three main risk factors: ...",
  "sources": ["report.pdf"],
  "docs_used": 3
}
```

---

### `GET /api/v1/documents`

Return the total number of indexed document chunks.

**Response** `200 OK`

```json
{
  "total_documents": 47
}
```

---

### `DELETE /api/v1/clear`

Remove all documents from the vector store.

**Response** `200 OK`

```json
{
  "message": "Vector store cleared successfully."
}
```

---

### `GET /health`

Basic health probe.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## Usage Examples

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@./my_document.pdf"
```

### Query the knowledge base

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the conclusion of the document?", "n_results": 5}'
```

### Check indexed document count

```bash
curl http://localhost:8000/api/v1/documents
```

### Clear the vector store

```bash
curl -X DELETE http://localhost:8000/api/v1/clear
```

---

## How It Works

### Document Ingestion Pipeline

```
PDF File
   │
   ▼
PyMuPDFLoader          — Extracts raw text page by page
   │
   ▼
RecursiveCharacterTextSplitter
  chunk_size=1000       — Splits text into overlapping chunks
  chunk_overlap=200     — Preserves context across boundaries
   │
   ▼
EmbeddingManager       — Encodes each chunk with all-MiniLM-L6-v2
  (SentenceTransformer) — Produces 384-dimensional vectors
   │
   ▼
VectorStore (ChromaDB) — Persists vectors + metadata to disk
```

### Query Pipeline

```
User Question
   │
   ▼
EmbeddingManager       — Embeds the question into a 384-d vector
   │
   ▼
RAGRetriever           — Queries ChromaDB for top-K nearest chunks
  score_threshold=0.1  — Filters out low-relevance results
   │
   ▼
RAGGenerator           — Builds a context-grounded prompt
   │
   ▼
OpenAI Chat API        — Generates answer from context only
  model=gpt-3.5-turbo
  temperature=0.1
   │
   ▼
JSON Response          — answer + sources + docs_used
```

The system message instructs the model to answer **only from the provided context** and respond with "I don't know" if the answer is absent, keeping responses factual and grounded.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
