import os
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class VectorStore:
    """
    Handles FAISS index creation, updates, queries, and persistence.
    Supports both L2 and Cosine similarity with automatic normalization.
    """

    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.json"
    STORE_META_FILE = "store_meta.json"

    def __init__(
        self,
        store_dir: str,
        mode: str,
        embedding_dim: int = 1536,
        similarity_metric: str = "cosine"
    ):
        """
        Initialize VectorStore.

        Args:
            store_dir (str): Directory where FAISS files are stored
            mode (str): 'LEX_NANO' or 'LEX_LDS'
            embedding_dim (int): Dimension of embedding vectors
            similarity_metric (str): 'cosine' or 'l2' - distance metric to use
        """
        self.store_dir = Path(store_dir)
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric.lower()
        
        if self.similarity_metric not in ["cosine", "l2"]:
            raise ValueError(f"similarity_metric must be 'cosine' or 'l2', got '{similarity_metric}'")
        
        self.index_path = self.store_dir / self.INDEX_FILE
        self.meta_path = self.store_dir / self.META_FILE
        self.store_meta_path = self.store_dir / self.STORE_META_FILE
        
        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata = {}

        # Validate mode and similarity metric consistency
        self._validate_store_config()

        # Initialize or load FAISS index
        self.index = self._load_or_initialize_index()
        self._load_metadata()
        
        print(f"[VectorStore] Initialized with mode={self.mode}, similarity={self.similarity_metric}")

    # =====================================================
    # Configuration Validation
    # =====================================================

    def _validate_store_config(self):
        """
        Ensure that an existing FAISS store is used only with the same mode and similarity metric.
        """
        if self.store_meta_path.exists():
            with open(self.store_meta_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            
            existing_mode = existing_meta.get("mode")
            existing_metric = existing_meta.get("similarity_metric", "l2")  # Default to l2 for backward compatibility

            # Check mode mismatch
            if existing_mode and existing_mode != self.mode:
                raise RuntimeError(
                    f"[VectorStore] Mode mismatch detected! "
                    f"This FAISS store was created with mode '{existing_mode}', "
                    f"but you are trying to load it with '{self.mode}'. "
                    f"Please delete or use a new store directory."
                )
            
            # Check similarity metric mismatch
            if existing_metric != self.similarity_metric:
                raise RuntimeError(
                    f"[VectorStore] Similarity metric mismatch detected! "
                    f"This FAISS store was created with '{existing_metric}', "
                    f"but you are trying to load it with '{self.similarity_metric}'. "
                    f"Please delete or use a new store directory."
                )
        else:
            # Save configuration on first creation
            with open(self.store_meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "mode": self.mode,
                    "similarity_metric": self.similarity_metric,
                    "embedding_dim": self.embedding_dim,
                    "created_at": datetime.utcnow().isoformat()
                }, f, indent=4)

    # =====================================================
    # Initialization
    # =====================================================

    def _load_or_initialize_index(self):
        """Load existing FAISS index or create a new one."""
        if self.index_path.exists():
            print(f"[VectorStore] Loading existing FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))

        print(f"[VectorStore] Creating new FAISS index for mode={self.mode}, similarity={self.similarity_metric}")
        
        if self.mode == "LEX_NANO":
            # Flat index - exact search
            if self.similarity_metric == "cosine":
                # For cosine similarity with flat index, use IndexFlatIP with normalized vectors
                index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)
                
        elif self.mode == "LEX_LDS":
            # IVF index - approximate search
            nlist = min(256, max(8, int(self.embedding_dim / 4)))  # adaptive
            
            if self.similarity_metric == "cosine":
                # For cosine similarity with IVF, use IndexIVFFlat with Inner Product
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
            
            index.nprobe = min(10, nlist)
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
    # Vector Normalization (for Cosine Similarity)
    # =====================================================

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.
        
        Args:
            vectors: numpy array of shape (n, embedding_dim)
            
        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

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

    def add_vectors(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Insert vectors and associate metadata.
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries (one per embedding)
            
        Returns:
            List of assigned vector IDs
        """
        if not embeddings:
            print("[VectorStore] No embeddings to add.")
            return []

        vectors = np.array(embeddings).astype("float32")
        
        # Normalize vectors if using cosine similarity
        if self.similarity_metric == "cosine":
            vectors = self._normalize_vectors(vectors)

        # Train IVF if necessary
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            nlist = self.index.nlist
            if len(vectors) < nlist:
                print(
                    f"[VectorStore] Too few vectors ({len(vectors)}) to train IVF with {nlist} clusters. "
                    f"Adjusting nlist..."
                )
                # Rebuild index with fewer clusters
                nlist = max(4, len(vectors) // 2)
                
                if self.similarity_metric == "cosine":
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(
                        quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
                    )
                else:
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(
                        quantizer, self.embedding_dim, nlist, faiss.METRIC_L2
                    )
                
                self.index.nprobe = min(10, nlist)
            
            print(f"[VectorStore] Training IVF index with {len(vectors)} vectors and {nlist} clusters...")
            self.index.train(vectors)

        start_id = len(self.metadata)
        ids = np.arange(start_id, start_id + len(vectors)).astype("int64")

        # Insert vectors based on index type
        if isinstance(self.index, (faiss.IndexFlatL2, faiss.IndexFlatIP)):
            # Flat indexes don't support add_with_ids()
            self.index.add(vectors)
            assigned_ids = list(range(start_id, start_id + vectors.shape[0]))
        else:
            # IVF and other indexes support add_with_ids()
            self.index.add_with_ids(vectors, ids)
            assigned_ids = ids.tolist()

        # Store metadata mapping
        for i, mid in enumerate(assigned_ids):
            meta = {
                **metadata_list[i],
                "inserted_at": datetime.utcnow().isoformat()
            }
            # Ensure page_content is stored
            if "text" in metadata_list[i]:
                meta["page_content"] = metadata_list[i]["text"]
            elif "page_content" not in meta:
                meta["page_content"] = ""
            
            self.metadata[str(mid)] = meta

        self._save()
        print(f"[VectorStore] Added {len(vectors)} vectors to FAISS (similarity={self.similarity_metric}).")
        return assigned_ids

    def delete_vectors(self, vector_ids: List[int]):
        """
        Remove vectors from index and metadata.
        
        Args:
            vector_ids: List of vector IDs to delete
        """
        if not vector_ids:
            return
        
        print(f"[VectorStore] Deleting {len(vector_ids)} vectors...")
        try:
            id_array = np.array(vector_ids).astype("int64")
            
            # Check if index supports removal
            if hasattr(self.index, 'remove_ids'):
                self.index.remove_ids(id_array)
            else:
                print("[VectorStore] WARNING: This index type doesn't support removal. "
                      "Only metadata will be deleted.")
            
            # Remove metadata
            for vid in vector_ids:
                self.metadata.pop(str(vid), None)
            
            self._save()
            print(f"[VectorStore] Successfully deleted {len(vector_ids)} vectors.")
        except Exception as e:
            print(f"[VectorStore] Vector deletion failed: {e}")

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        with_score: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity query.

        Args:
            query_vector: The query embedding vector
            top_k: Number of nearest neighbors to retrieve
            filter: Optional metadata filter (e.g., {"source": "doc1.pdf"})
            with_score: Include similarity scores in results
            score_threshold: Minimum similarity score to include
                - For cosine: higher is better (0-1 range), filter scores >= threshold
                - For L2: lower is better, filter scores <= threshold

        Returns:
            List of dictionaries containing metadata and content
        """
        if self.index.ntotal == 0:
            print("[VectorStore] Index is empty. No vectors to search.")
            return []

        qv = np.array([query_vector]).astype("float32")
        
        # Normalize query vector if using cosine similarity
        if self.similarity_metric == "cosine":
            qv = self._normalize_vectors(qv)

        # Perform search
        distances, ids = self.index.search(qv, min(top_k * 2, self.index.ntotal))  # Fetch extra for filtering

        results = []
        for i, idx in enumerate(ids[0]):
            if idx == -1:  # Invalid index
                continue
            
            meta = self.metadata.get(str(idx))
            if not meta:
                continue
            
            # Apply metadata filter
            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue
            
            score = float(distances[0][i])
            
            # Apply score threshold
            if score_threshold is not None:
                if self.similarity_metric == "cosine":
                    # For cosine (inner product): higher is better
                    if score < score_threshold:
                        continue
                else:
                    # For L2: lower is better
                    if score > score_threshold:
                        continue
            
            entry = {
                "meta": meta,
                "page_content": meta.get("page_content", "")
            }
            
            if with_score:
                entry["score"] = score
                entry["similarity_metric"] = self.similarity_metric
            
            results.append(entry)
            
            # Stop once we have enough results
            if len(results) >= top_k:
                break

        return results

    def query_with_debug(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Perform query and print detailed debug information.
        
        Useful for diagnosing retrieval issues.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to retrieve and display
            filter: Optional metadata filter
        """
        print(f"\n{'='*80}")
        print(f"DEBUG QUERY (Mode: {self.mode}, Similarity: {self.similarity_metric})")
        print(f"Total vectors in index: {self.index.ntotal}")
        print(f"{'='*80}\n")
        
        results = self.query(query_vector, top_k=top_k, filter=filter, with_score=True)
        
        if not results:
            print("❌ NO RESULTS FOUND!")
            print("\nPossible issues:")
            print("1. Index is empty or not properly trained")
            print("2. Query vector dimension mismatch")
            print("3. Metadata filter is too restrictive")
            print("4. Score threshold too strict")
            return
        
        for i, result in enumerate(results, 1):
            meta = result["meta"]
            score = result.get("score", 0)
            content = result["page_content"]
            
            print(f"Result #{i}")
            print(f"  Score: {score:.4f} ({self.similarity_metric.upper()})")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  File Type: {meta.get('file_type', 'N/A')}")
            print(f"  Chunk Index: {meta.get('chunk_index', 'N/A')}")
            print(f"  Content Length: {len(content)} chars")
            print(f"  Content Preview:")
            print(f"    {content[:200]}...")
            print(f"{'-'*80}\n")

    # =====================================================
    # Utility
    # =====================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing store statistics
        """
        return {
            "mode": self.mode,
            "similarity_metric": self.similarity_metric,
            "embedding_dim": self.embedding_dim,
            "total_vectors": self.index.ntotal,
            "metadata_count": len(self.metadata),
            "index_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "store_dir": str(self.store_dir)
        }

    def clear(self):
        """Reset index and metadata."""
        # Recreate index with same configuration
        self.index = self._load_or_initialize_index()
        self.metadata = {}
        self._save()
        print("[VectorStore] Cleared all stored vectors.")

    def rebuild_with_new_metric(self, new_similarity_metric: str) -> None:
        """
        Rebuild the index with a different similarity metric.
        WARNING: This will clear all existing vectors!
        
        Args:
            new_similarity_metric: 'cosine' or 'l2'
        """
        print(f"[VectorStore] Rebuilding index with new similarity metric: {new_similarity_metric}")
        
        # Update configuration
        self.similarity_metric = new_similarity_metric.lower()
        
        # Update store metadata
        with open(self.store_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "mode": self.mode,
                "similarity_metric": self.similarity_metric,
                "embedding_dim": self.embedding_dim,
                "rebuilt_at": datetime.utcnow().isoformat()
            }, f, indent=4)
        
        # Clear and reinitialize
        self.clear()
        
        print(f"[VectorStore] Index rebuilt successfully with {new_similarity_metric} similarity.")

#
# # =====================================================
# # Usage Examples
# # =====================================================
#
# if __name__ == "__main__":
#     # Example 1: Create vector store with cosine similarity (RECOMMENDED)
#     store_cosine = VectorStore(
#         store_dir="./faiss_store_cosine",
#         mode="LEX_NANO",
#         embedding_dim=1536,
#         similarity_metric="cosine"  # Better for text embeddings
#     )
#
#     # Example 2: Create vector store with L2 distance
#     store_l2 = VectorStore(
#         store_dir="./faiss_store_l2",
#         mode="LEX_NANO",
#         embedding_dim=1536,
#         similarity_metric="l2"
#     )
#
#     # Example 3: Add vectors with metadata
#     embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
#     metadata_list = [
#         {"source": "doc1.pdf", "page": 1, "text": "Sample content 1"},
#         {"source": "doc1.pdf", "page": 2, "text": "Sample content 2"},
#         {"source": "doc2.pdf", "page": 1, "text": "Sample content 3"}
#     ]
#
#     ids = store_cosine.add_vectors(embeddings, metadata_list)
#     print(f"Added vectors with IDs: {ids}")
#
#     # Example 4: Query with debug
#     query_vec = [0.15] * 1536
#     store_cosine.query_with_debug(query_vec, top_k=3)
#
#     # Example 5: Query with score threshold
#     results = store_cosine.query(
#         query_vec,
#         top_k=5,
#         score_threshold=0.5,  # For cosine: only scores >= 0.5
#         with_score=True
#     )
#
#     # Example 6: Get store statistics
#     stats = store_cosine.get_stats()
#     print(f"Store stats: {stats}")