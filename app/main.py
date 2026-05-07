"""FastAPI RAG API - Application entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="RAG API",
    description=(
        "A production-grade Retrieval-Augmented Generation (RAG) API "
        "built with FastAPI, LangChain, ChromaDB, and OpenAI. "
        "Upload PDF documents and query them using natural language."
    ),
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "documents",
            "description": "Upload and manage PDF documents in the vector store.",
        },
        {
            "name": "query",
            "description": "Submit natural-language questions and receive AI-generated answers.",
        },
        {
            "name": "system",
            "description": "System health and metadata endpoints.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["system"], summary="API root")
def root():
    """Return API metadata and available endpoint paths."""
    return {
        "message": "RAG API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "documents": "/api/v1/documents",
            "clear": "/api/v1/clear",
        },
    }


@app.get("/health", tags=["system"], summary="Health check")
def health_check():
    """Return the current health status of the API."""
    return {
        "status": "healthy",
        "version": "1.0.0",
    }
