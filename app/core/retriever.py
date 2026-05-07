"""RAG document retrieval from ChromaDB."""
import logging
from typing import Any, Dict, List

from app.core.embeddings import EmbeddingManager
from app.core.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves semantically relevant documents from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        logger.info("RAGRetriever initialized.")

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        score_threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: The user's search query.
            n_results: Maximum number of documents to retrieve.
            score_threshold: Minimum cosine-similarity score (0–1) to include.

        Returns:
            List of dicts with keys: content, metadata, relevance_score.
        """
        try:
            query_embedding = self.embedding_manager.generate_embeddings([query])

            results = self.vector_store.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_docs: List[Dict[str, Any]] = []
            for i in range(len(results["documents"][0])):
                relevance_score = 1 - results["distances"][0][i]
                if relevance_score >= score_threshold:
                    retrieved_docs.append(
                        {
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "relevance_score": round(relevance_score, 4),
                        }
                    )

            logger.info("Retrieved %d documents for query.", len(retrieved_docs))
            return retrieved_docs

        except Exception as e:
            logger.exception("Error retrieving documents: %s", e)
            raise
