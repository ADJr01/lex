"""
QueryService
============

Provides semantic query interface over FAISS VectorStore.
Supports FAST (single-pass) and DEEP (multi-stage) modes with scoring and filters.
"""

import time
import numpy as np


class QueryService:
    """Performs advanced vector queries with scoring and filters."""

    def __init__(self, vector_store, embedder, response_mode="FAST"):
        """
        Initialize QueryService.

        Args:
            vector_store (VectorStore): The FAISS-backed vector store
            embedder (Embedder): Embedding model wrapper
            response_mode (str): 'FAST' or 'DEEP'
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.response_mode = response_mode

    # =====================================================
    # Core Query Execution
    # =====================================================

    def invoke(self, query: str, filter=None, top_k: int = 5, with_score: bool = True):
        """
        Execute a semantic query.

        Args:
            query (str): Natural language query
            filter (dict): Optional metadata filter (e.g. {'source': 'sales.csv'})
            top_k (int): Number of top results
            with_score (bool): Whether to include similarity scores

        Returns:
            List[dict]: Retrieved results sorted by similarity
        """
        start_time = time.time()
        print(f"[QueryService] Running {self.response_mode} query: '{query}'")

        # 1. Embed query
        q_vector = self.embedder.embed_query(query)
        if not q_vector:
            print("[QueryService] Failed to embed query.")
            return []

        # 2. FAST mode → single retrieval
        if self.response_mode == "FAST":
            results = self.vector_store.query(q_vector, top_k=top_k, filter=filter, with_score=with_score)
            print(f"[QueryService] FAST retrieval completed ({len(results)} results).")
            return results

        # 3. DEEP mode → multi-step retrieval and re-ranking
        if self.response_mode == "DEEP":
            return self._deep_query(query, q_vector, filter, top_k, with_score)

        # Invalid mode
        print(f"[QueryService] Unknown mode: {self.response_mode}")
        return []


    # Deep Query Logic

    def _deep_query(self, query, q_vector, filter, top_k, with_score):
        """
        Perform deeper, multi-pass query with query expansion and re-ranking.
        """
        print("[QueryService] Starting DEEP query mode...")
        start_time = time.time()
        # Step 1: Initial retrieval
        primary_results = self.vector_store.query(q_vector, top_k=top_k * 2, filter=filter, with_score=True)

        # Step 2: Expand query with top retrieved content
        if not primary_results:
            return []

        top_texts = [r["meta"].get("preview", r["meta"].get("source", "")) for r in primary_results[:3]]
        expanded_query = f"{query}. Context hints: {' '.join(top_texts)}"

        q2_vector = self.embedder.embed_query(expanded_query)

        # Step 3: Second retrieval (refined)
        refined_results = self.vector_store.query(q2_vector, top_k=top_k, filter=filter, with_score=True)

        # Step 4: Merge and rerank
        combined = {r["meta"]["source"] + str(r["meta"].get("chunk_index", "")): r for r in (primary_results + refined_results)}

        results = sorted(
            combined.values(),
            key=lambda x: x.get("score", float("inf"))
        )[:top_k]

        elapsed = time.time() - start_time
        print(f"[QueryService] DEEP retrieval completed in {elapsed:.2f}s ({len(results)} results).")

        return results


    # =====================================================
    # GET AS RETRIEVER
    # =====================================================

    def store_as_retriever(self,k:int):
        return  self.vector_store.get_as_retriever(k=k,embeddings_model=self.embedder)
