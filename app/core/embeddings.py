"""Embedding generation using SentenceTransformers."""
import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embedding generation via SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model from disk or cache."""
        try:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info(
                "Embedding model loaded. Dimension: %d",
                self.model.get_embedding_dimension(),
            )
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input strings to embed.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        if self.model is None:
            raise ValueError("Embedding model is not loaded.")
        return self.model.encode(texts, show_progress_bar=False)
