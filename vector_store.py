"""
A simple in-memory vector store for managing and querying text embeddings,
backed by a JSON file for persistence.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constants import LOGGER
from openai_clients import call_openai_embedding

class VectorStore:
    """A file-backed vector store for managing text embeddings."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load vectors and metadata from the JSON file if it exists."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.vectors = data.get("vectors", {})
                    self.metadata = data.get("metadata", {})
                    LOGGER.info(f"Loaded {len(self.vectors)} vectors from {self.file_path}")
            except (json.JSONDecodeError, IOError) as e:
                LOGGER.error(f"Failed to load vector store from {self.file_path}: {e}")

    def _save(self) -> None:
        """Save the current vectors and metadata to the JSON file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({"vectors": self.vectors, "metadata": self.metadata}, f)
        except IOError as e:
            LOGGER.error(f"Failed to save vector store to {self.file_path}: {e}")

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate the cosine similarity between two vectors."""
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = math.sqrt(sum(v**2 for v in vec1))
        norm2 = math.sqrt(sum(v**2 for v in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def add(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Generate embeddings for a list of texts and add them to the store.

        Args:
            texts: A list of strings to embed and store.
            metadatas: A list of metadata dictionaries, one for each text.
        """
        if len(texts) != len(metadatas):
            raise ValueError("The number of texts and metadatas must be the same.")

        for i, text in enumerate(texts):
            embedding_response = call_openai_embedding(text)
            if embedding_response and "embedding" in embedding_response:
                vector = embedding_response["embedding"]
                # Use the text itself as a simple key
                self.vectors[text] = vector
                self.metadata[text] = metadatas[i]
            else:
                LOGGER.warning(f"Failed to generate embedding for text: {text}")
        
        self._save()

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query the vector store to find the most similar texts.

        Args:
            query_text: The text to find similar entries for.
            top_k: The number of top results to return.

        Returns:
            A list of tuples, each containing (text, similarity_score, metadata).
        """
        query_embedding_response = call_openai_embedding(query_text)
        if not query_embedding_response or "embedding" not in query_embedding_response:
            LOGGER.error("Failed to generate embedding for the query text.")
            return []

        query_vector = query_embedding_response["embedding"]
        
        similarities = []
        for key, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((key, similarity, self.metadata.get(key, {})))

        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
