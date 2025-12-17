"""
Embedder
========

Handles embedding generation using Ollama or other models.
Converts semantic chunks into high-dimensional vectors for FAISS indexing.
"""

import time
from typing import List, Dict


class Embedder:
    """Generates embeddings for text chunks using the configured model."""

    def __init__(self, embedding_model, batch_size: int = 8, max_retries: int = 3):
        """
        Initialize Embedder.

        Args:
            embedding_model: LangChain-compatible embedding model
            batch_size (int): Number of chunks to embed per batch
            max_retries (int): Retries for failed embeddings
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries

    # =====================================================
    # Internal Utilities
    # =====================================================

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                vectors = self.embedding_model.embed_documents(texts)
                return vectors
            except Exception as e:
                print(f"[Embedder] Embedding batch failed (attempt {attempt}): {e}")
                time.sleep(1.5 * attempt)
        print("[Embedder] Batch failed after retries; returning empty embeddings.")
        return [[] for _ in texts]

    # =====================================================
    # Public Methods
    # =====================================================

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks (List[dict]): List of {"text": str, "meta": {...}}
        Returns:
            List[dict]: List of {"vector": [...], "meta": {...}}
        """
        if not chunks:
            print("[Embedder] No chunks to embed.")
            return []

        print(f"[Embedder] Embedding {len(chunks)} chunks in batches of {self.batch_size}...")

        embedded = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c["text"] for c in batch]

            vectors = self._embed_batch(texts)
            for j, vector in enumerate(vectors):
                if not vector:
                    continue
                embedded.append({
                    "vector": vector,
                    "meta": batch[j]["meta"],
                    "text": batch[j]["text"]
                })

        print(f"[Embedder] Completed embeddings for {len(embedded)} chunks.")
        return embedded

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string (for search).

        Args:
            query (str): The query text
        Returns:
            List[float]: Embedding vector
        """
        try:
            return self.embedding_model.embed_query(query)
        except Exception as e:
            print(f"[Embedder] Query embedding failed: {e}")
            return []
