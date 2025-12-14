"""
VectorStore
===========

Manages FAISS vector database (L2 or IVF based on mode).
Supports insertion, deletion, querying, and persistent metadata mapping.
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime


class VectorStore:
    """Handles FAISS index creation, updates, queries, and persistence."""

    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.json"

    def __init__(self, store_dir: str, mode: str, embedding_dim: int = 1536):
        """
        Initialize VectorStore.

        Args:
            store_dir (str): Directory where FAISS files are stored
            mode (str): 'LEX_NANO' or 'LEX_LDS'
            embedding_dim (int): Dimension of embedding vectors
        """
        self.store_dir = Path(store_dir)
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.index_path = self.store_dir / self.INDEX_FILE
        self.meta_path = self.store_dir / self.META_FILE

        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata = {}

        # Initialize or load FAISS index
        self.index = self._load_or_initialize_index()
        self._load_metadata()

    # =====================================================
    # Initialization
    # =====================================================

    def _load_or_initialize_index(self):
        """Load existing FAISS index or create a new one."""
        if self.index_path.exists():
            print(f"[VectorStore] Loading existing FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))

        print(f"[VectorStore] Creating new FAISS index for mode={self.mode}")
        if self.mode == "LEX_NANO":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.mode == "LEX_LDS":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 256)
            index.nprobe = 10
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return index

    def _load_metadata(self):
        """Load metadata mapping from disk."""
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    # =====================================================
    # Save / Persist
    # =====================================================

    def _save(self):
        """Persist FAISS index and metadata."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

    # =====================================================
    # Vector Operations
    # =====================================================

    def add_vectors(self, embeddings, metadata_list):
        """
        Insert vectors and associate metadata.
        """
        if not embeddings:
            print("[VectorStore] No embeddings to add.")
            return []

        vectors = np.array(embeddings).astype("float32")

        # Train IVF if necessary
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            print("[VectorStore] Training IVF index before insertion...")
            self.index.train(vectors)

        start_id = len(self.metadata)
        ids = np.arange(start_id, start_id + len(vectors)).astype("int64")

        # 🔧 FIX: use different insertion method depending on index type
        if isinstance(self.index, faiss.IndexFlatL2):
            # IndexFlatL2 doesn’t support add_with_ids()
            self.index.add(vectors)
            assigned_ids = list(range(start_id, start_id + vectors.shape[0]))
        else:
            self.index.add_with_ids(vectors, ids)
            assigned_ids = ids.tolist()

        # Store metadata mapping
        for i, mid in enumerate(assigned_ids):
            meta = {
                **metadata_list[i],
                "inserted_at": datetime.utcnow().isoformat()
            }
            self.metadata[str(mid)] = meta

        self._save()
        print(f"[VectorStore] Added {len(vectors)} vectors to FAISS.")
        return assigned_ids

    def delete_vectors(self, vector_ids):
        """Remove vectors from index and metadata."""
        if not vector_ids:
            return
        print(f"[VectorStore] Deleting {len(vector_ids)} vectors...")
        try:
            id_array = np.array(vector_ids).astype("int64")
            self.index.remove_ids(id_array)
            for vid in vector_ids:
                self.metadata.pop(str(vid), None)
            self._save()
        except Exception as e:
            print(f"[VectorStore] Vector deletion failed: {e}")

    def query(self, query_vector, top_k=5, filter=None, with_score=True):
        """
        Perform similarity query.

        Args:
            query_vector (List[float]): The query embedding vector
            top_k (int): Number of nearest neighbors
            filter (dict): Optional metadata filter
            with_score (bool): Include similarity scores
        Returns:
            List[dict]: Ranked results with metadata
        """
        if self.index.ntotal == 0:
            print("[VectorStore] Index empty.")
            return []

        qv = np.array([query_vector]).astype("float32")
        distances, ids = self.index.search(qv, top_k)

        results = []
        for i, idx in enumerate(ids[0]):
            if idx == -1:
                continue
            meta = self.metadata.get(str(idx))
            if not meta:
                continue
            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue
            entry = {"meta": meta}
            if with_score:
                entry["score"] = float(distances[0][i])
            results.append(entry)

        return results

    # =====================================================
    # Utility
    # =====================================================

    def clear(self):
        """Reset index and metadata."""
        self.index = self._load_or_initialize_index()
        self.metadata = {}
        self._save()
        print("[VectorStore] Cleared all stored vectors.")
