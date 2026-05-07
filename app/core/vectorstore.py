"""ChromaDB vector store management."""
import os
import uuid
import logging
from typing import Any, List

import chromadb
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and retrieval using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "data/vector_store",
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client: Any = None
        self.collection: Any = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"},
            )
            logger.info(
                "VectorStore initialized. Collection: '%s', documents: %d",
                self.collection_name,
                self.collection.count(),
            )
        except Exception as e:
            logger.exception("Failed to initialize VectorStore: %s", e)
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        """Add documents with their embeddings to the collection.

        Args:
            documents: LangChain Document objects with page_content and metadata.
            embeddings: NumPy array of shape (len(documents), embedding_dim).
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document count ({len(documents)}) must match "
                f"embedding count ({len(embeddings)})."
            )

        ids: List[str] = []
        metadatas: List[dict] = []
        documents_text: List[str] = []
        embeddings_list: List[List[float]] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )
        logger.info(
            "Added %d documents. Total in collection: %d",
            len(documents),
            self.collection.count(),
        )

    def get_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Remove all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings"},
        )
        logger.info("Collection '%s' cleared.", self.collection_name)
