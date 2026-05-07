"""RAG answer generation using OpenAI."""
import os
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from app.core.retriever import RAGRetriever

load_dotenv()

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generates answers by combining retrieved documents with an OpenAI model."""

    def __init__(
        self,
        retriever: RAGRetriever,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("RAGGenerator initialized with model: %s", model)

    def generate(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Retrieve relevant documents and generate an answer.

        Args:
            query: The user's question.
            n_results: Number of documents to retrieve for context.

        Returns:
            Dict with keys: answer (str), sources (List[str]), docs_used (int).
        """
        try:
            retrieved_docs = self.retriever.retrieve(query, n_results)

            if not retrieved_docs:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "sources": [],
                    "docs_used": 0,
                }

            context = "\n\n".join(
                f"Source: {doc['metadata'].get('source_file', 'unknown')}\n"
                f"Content: {doc['content']}"
                for doc in retrieved_docs
            )

            prompt = (
                "You are a helpful AI assistant.\n"
                "Answer the question based ONLY on the context below.\n"
                'If the answer is not in the context, say "I don\'t know."\n\n'
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers "
                            "based on provided context only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            answer = response.choices[0].message.content
            sources: List[str] = list(
                {doc["metadata"].get("source_file", "unknown") for doc in retrieved_docs}
            )

            return {
                "answer": answer,
                "sources": sources,
                "docs_used": len(retrieved_docs),
            }

        except Exception as e:
            logger.exception("Error generating answer: %s", e)
            raise
