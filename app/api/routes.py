"""API route definitions for the RAG system."""
import os
import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.core.embeddings import EmbeddingManager
from app.core.generator import RAGGenerator
from app.core.retriever import RAGRetriever
from app.core.vectorstore import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

embedding_manager = EmbeddingManager()
vector_store = VectorStore()
retriever = RAGRetriever(vector_store, embedding_manager)
generator = RAGGenerator(retriever)


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to ask the RAG system.")
    n_results: int = Field(3, ge=1, le=20, description="Number of documents to retrieve.")


class QueryResponse(BaseModel):
    query: str = Field(..., description="The original question.")
    answer: str = Field(..., description="AI-generated answer based on retrieved context.")
    sources: List[str] = Field(..., description="Source files referenced in the answer.")
    docs_used: int = Field(..., description="Number of documents used to generate the answer.")


class UploadResponse(BaseModel):
    message: str = Field(..., description="Result summary of the upload operation.")
    pages: int = Field(..., description="Number of pages parsed from the PDF.")
    chunks: int = Field(..., description="Number of text chunks stored in the vector store.")
    total_docs: int = Field(..., description="Total documents now in the vector store.")


class DocumentCountResponse(BaseModel):
    total_documents: int = Field(..., description="Current document count in the vector store.")


class ClearResponse(BaseModel):
    message: str = Field(..., description="Confirmation message after clearing the vector store.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    tags=["documents"],
)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file into the vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted.",
        )

    try:
        upload_dir = "data/pdf_files"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)

        logger.info(
            "Uploaded '%s': %d pages, %d chunks.", file.filename, len(documents), len(chunks)
        )

        return UploadResponse(
            message=f"{file.filename} uploaded and processed successfully.",
            pages=len(documents),
            chunks=len(chunks),
            total_docs=vector_store.get_count(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing upload for '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    tags=["query"],
)
async def query_rag(request: QueryRequest):
    """Submit a question and receive an AI-generated answer with source references."""
    try:
        result = generator.generate(request.query, request.n_results)
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            docs_used=result["docs_used"],
        )
    except Exception as e:
        logger.exception("Error processing query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/documents",
    response_model=DocumentCountResponse,
    summary="Get document count",
    tags=["documents"],
)
async def get_documents():
    """Return the total number of documents currently in the vector store."""
    return DocumentCountResponse(total_documents=vector_store.get_count())


@router.delete(
    "/clear",
    response_model=ClearResponse,
    summary="Clear the vector store",
    tags=["documents"],
)
async def clear_vectorstore():
    """Remove all documents from the vector store."""
    vector_store.clear()
    return ClearResponse(message="Vector store cleared successfully.")
