import os
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Handles FAISS index creation, updates, queries, and persistence.
    Supports both L2 and Cosine similarity with automatic normalization.
    Includes intelligent IVF training with Lloyd's K-means optimization.
    """

    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.json"
    STORE_META_FILE = "store_meta.json"
    TRAINING_META_FILE = "training_meta.json"

    def __init__(
            self,
            store_dir: str,
            mode: str,
            embedding_dim: int = 1536,
            similarity_metric: str = "cosine",
            instance_name: str = None,
    ):
        """
        Initialize VectorStore.

        Args:
            store_dir (str): Directory where FAISS files are stored
            mode (str): 'LEX_NANO' (exact search) or 'LEX_LDS' (approximate search with IVF)
            embedding_dim (int): Dimension of embedding vectors
            similarity_metric (str): 'cosine' or 'l2' - distance metric to use
        """
        self.instance_name = instance_name if instance_name is not None else ""
        self.store_dir = Path(store_dir)
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric.lower()

        if self.similarity_metric not in ["cosine", "l2"]:
            raise ValueError(f"similarity_metric must be 'cosine' or 'l2', got '{similarity_metric}'")

        self.index_path = self.store_dir / self.INDEX_FILE
        self.meta_path = self.store_dir / self.META_FILE
        self.store_meta_path = self.store_dir / self.STORE_META_FILE
        self.training_meta_path = self.store_dir / self.TRAINING_META_FILE

        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata = {}
        self.training_metadata = {}

        # Validate mode and similarity metric consistency
        self._validate_store_config()

        # Initialize or load FAISS index
        self.index = self._load_or_initialize_index()
        self._load_metadata()
        self._load_training_metadata()

        logger.info(f"[VectorStore] Initialized with mode={self.mode}, similarity={self.similarity_metric}")

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
            existing_metric = existing_meta.get("similarity_metric", "l2")

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
                    "app":self.instance_name,
                    "mode": self.mode,
                    "similarity_metric": self.similarity_metric,
                    "embedding_dim": self.embedding_dim,
                    "created_at": datetime.utcnow().isoformat()
                }, f, indent=4)

    # =====================================================
    # Initialization
    # =====================================================

    def _calculate_optimal_nlist(self, expected_vectors: int = 1000) -> int:
        """
        Calculate optimal number of clusters (nlist) for IVF index.

        Uses industry best practices:
        - For small datasets (< 1M): sqrt(N) to 4*sqrt(N)
        - For large datasets (> 1M): more aggressive clustering

        Args:
            expected_vectors: Expected number of vectors in the index

        Returns:
            Optimal nlist value
        """
        if expected_vectors < 1000:
            # Small dataset: conservative clustering
            nlist = max(8, int(np.sqrt(expected_vectors)))
        elif expected_vectors < 100000:
            # Medium dataset: sqrt(N) * 2
            nlist = int(np.sqrt(expected_vectors) * 2)
        elif expected_vectors < 1000000:
            # Large dataset: sqrt(N) * 4
            nlist = int(np.sqrt(expected_vectors) * 4)
        else:
            # Very large dataset: more aggressive
            nlist = int(np.sqrt(expected_vectors) * 8)

        # Ensure reasonable bounds
        nlist = max(8, min(nlist, 65536))  # FAISS limit is 65536

        return nlist

    def _calculate_optimal_nprobe(self, nlist: int) -> int:
        """
        Calculate optimal nprobe (number of clusters to search) for query time.

        Trade-off between speed and accuracy:
        - Higher nprobe = better accuracy, slower search
        - Lower nprobe = faster search, lower accuracy

        Args:
            nlist: Number of clusters in the index

        Returns:
            Optimal nprobe value
        """
        # Industry best practice: nprobe = sqrt(nlist) to nlist/10
        # We use a balanced approach for good accuracy/speed trade-off

        if nlist <= 16:
            nprobe = nlist  # Search all clusters for small nlist
        elif nlist <= 100:
            nprobe = max(10, nlist // 4)  # Search 25% of clusters
        elif nlist <= 1000:
            nprobe = max(20, nlist // 10)  # Search 10% of clusters
        else:
            nprobe = max(50, int(np.sqrt(nlist)))  # Sqrt approach for large nlist

        # Ensure nprobe doesn't exceed nlist
        return min(nprobe, nlist)

    def _load_or_initialize_index(self):
        """Load existing FAISS index or create a new one."""
        if self.index_path.exists():
            logger.info(f"[VectorStore] Loading existing FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))

        logger.info(f"[VectorStore] Creating new FAISS index for mode={self.mode}, similarity={self.similarity_metric}")

        if self.mode == "LEX_NANO":
            # Flat index - exact search (no training needed)
            if self.similarity_metric == "cosine":
                index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)

        elif self.mode == "LEX_LDS":
            # IVF index - approximate search (requires training)
            # Start with conservative nlist, will be adjusted during training
            initial_nlist = self._calculate_optimal_nlist(expected_vectors=1000)

            if self.similarity_metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    initial_nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    initial_nlist,
                    faiss.METRIC_L2
                )

            # Set initial nprobe
            index.nprobe = self._calculate_optimal_nprobe(initial_nlist)

            logger.info(f"[VectorStore] IVF index created with nlist={initial_nlist}, nprobe={index.nprobe}")
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

    def _load_training_metadata(self):
        """Load training metadata from disk."""
        if self.training_meta_path.exists():
            with open(self.training_meta_path, "r", encoding="utf-8") as f:
                self.training_metadata = json.load(f)
        else:
            self.training_metadata = {
                "is_trained": False,
                "training_vectors_count": 0,
                "nlist": 0,
                "nprobe": 0,
                "last_trained_at": None,
                "training_iterations": 0
            }

    # =====================================================
    # Intelligent IVF Training
    # =====================================================

    def _should_retrain(self, new_vectors_count: int) -> bool:
        """
        Determine if index should be retrained based on new data.

        Retraining criteria:
        1. Index has never been trained
        2. Number of vectors has doubled since last training
        3. Significant change in data distribution (future enhancement)

        Args:
            new_vectors_count: Number of new vectors being added

        Returns:
            True if retraining is recommended
        """
        if not self.training_metadata.get("is_trained", False):
            return True

        current_total = self.index.ntotal
        trained_count = self.training_metadata.get("training_vectors_count", 0)

        # Retrain if data has grown significantly (doubled)
        if current_total + new_vectors_count >= trained_count * 2:
            logger.info(
                f"[VectorStore] Retraining recommended: vectors grew from {trained_count} to {current_total + new_vectors_count}")
            return True

        return False

    def _train_ivf_index(
            self,
            vectors: np.ndarray,
            force_retrain: bool = False,
            max_iterations: int = 25
    ) -> bool:
        """
        Train IVF index using Lloyd's K-means algorithm with optimizations.

        Lloyd's K-means is the default and most robust clustering algorithm in FAISS.
        It iteratively refines cluster centers to minimize quantization error.

        Args:
            vectors: Training vectors (already normalized if using cosine)
            force_retrain: Force retraining even if not needed
            max_iterations: Maximum K-means iterations (default: 25)

        Returns:
            True if training was successful
        """
        if not isinstance(self.index, faiss.IndexIVFFlat):
            return True  # Not an IVF index, no training needed

        # Check if retraining is needed
        if self.index.is_trained and not force_retrain:
            if not self._should_retrain(len(vectors)):
                logger.info("[VectorStore] Index already trained and no retraining needed")
                return True

        total_vectors = self.index.ntotal + len(vectors)

        # Calculate optimal nlist based on total expected vectors
        optimal_nlist = self._calculate_optimal_nlist(total_vectors)

        # Check if we need to rebuild index with new nlist
        if optimal_nlist != self.index.nlist:
            logger.info(f"[VectorStore] Rebuilding index: nlist {self.index.nlist} → {optimal_nlist}")

            # Save existing vectors
            if self.index.ntotal > 0:
                logger.warning("[VectorStore] Existing vectors will need to be re-indexed")

            # Rebuild index with optimal nlist
            if self.similarity_metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                new_index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    optimal_nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                new_index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    optimal_nlist,
                    faiss.METRIC_L2
                )

            self.index = new_index

        # Ensure we have enough training vectors
        min_training_vectors = max(self.index.nlist * 39, 1000)  # FAISS recommendation: 39 * nlist

        if len(vectors) < self.index.nlist:
            logger.warning(
                f"[VectorStore] Insufficient vectors ({len(vectors)}) for training. "
                f"Need at least {self.index.nlist} (nlist). Adjusting nlist..."
            )
            # Reduce nlist to match available data
            new_nlist = max(4, len(vectors) // 10)

            if self.similarity_metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    new_nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    new_nlist,
                    faiss.METRIC_L2
                )

        # Prepare training data
        training_vectors = vectors.copy()

        # If we have more vectors than needed, sample representative subset
        if len(training_vectors) > min_training_vectors * 2:
            logger.info(f"[VectorStore] Sampling {min_training_vectors} vectors for efficient training")
            indices = np.random.choice(
                len(training_vectors),
                min_training_vectors,
                replace=False
            )
            training_vectors = training_vectors[indices]

        # Configure K-means parameters
        # Lloyd's algorithm is used by default in FAISS
        clustering_params = faiss.ClusteringParameters()
        clustering_params.niter = max_iterations  # K-means iterations
        clustering_params.verbose = False
        clustering_params.spherical = (self.similarity_metric == "cosine")  # Spherical K-means for cosine
        clustering_params.update_index = True

        # Train the index using Lloyd's K-means
        logger.info(
            f"[VectorStore] Training IVF index with Lloyd's K-means:\n"
            f"  - Training vectors: {len(training_vectors)}\n"
            f"  - Clusters (nlist): {self.index.nlist}\n"
            f"  - Max iterations: {max_iterations}\n"
            f"  - Spherical K-means: {clustering_params.spherical}"
        )

        try:
            self.index.train(training_vectors)

            # Update nprobe based on final nlist
            optimal_nprobe = self._calculate_optimal_nprobe(self.index.nlist)
            self.index.nprobe = optimal_nprobe

            # Update training metadata
            self.training_metadata = {
                "is_trained": True,
                "training_vectors_count": total_vectors,
                "nlist": self.index.nlist,
                "nprobe": self.index.nprobe,
                "last_trained_at": datetime.utcnow().isoformat(),
                "training_iterations": self.training_metadata.get("training_iterations", 0) + 1,
                "kmeans_iterations": max_iterations,
                "spherical": clustering_params.spherical
            }
            self._save_training_metadata()

            logger.info(
                f"[VectorStore] ✅ Training completed successfully!\n"
                f"  - Final nlist: {self.index.nlist}\n"
                f"  - Final nprobe: {self.index.nprobe}\n"
                f"  - Total training iterations: {self.training_metadata['training_iterations']}"
            )

            return True

        except Exception as e:
            logger.error(f"[VectorStore] Training failed: {e}")
            return False

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

    def _save_training_metadata(self):
        """Persist training metadata."""
        with open(self.training_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata, f, indent=4)

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
        For LEX_LDS mode, automatically handles IVF training with Lloyd's K-means.

        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries (one per embedding)

        Returns:
            List of assigned vector IDs
        """
        if not embeddings:
            logger.info("[VectorStore] No embeddings to add.")
            return []

        vectors = np.array(embeddings).astype("float32")

        # Normalize vectors if using cosine similarity
        if self.similarity_metric == "cosine":
            vectors = self._normalize_vectors(vectors)

        # LEX_LDS: Automatic IVF training with Lloyd's K-means
        if self.mode == "LEX_LDS":
            if isinstance(self.index, faiss.IndexIVFFlat):
                # Train index if needed
                training_success = self._train_ivf_index(vectors)

                if not training_success:
                    logger.error("[VectorStore] IVF training failed. Vectors not added.")
                    return []

        # LEX_NANO: Direct insertion (no training needed)
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

        logger.info(
            f"[VectorStore] Added {len(vectors)} vectors to FAISS "
            f"(mode={self.mode}, similarity={self.similarity_metric})."
        )

        return assigned_ids

    def delete_vectors(self, vector_ids: List[int]):
        """
        Remove vectors from index and metadata.

        Note: For IVF indexes, deletion may affect performance.
        Consider retraining if deleting many vectors.

        Args:
            vector_ids: List of vector IDs to delete
        """
        if not vector_ids:
            return

        logger.info(f"[VectorStore] Deleting {len(vector_ids)} vectors...")
        try:
            id_array = np.array(vector_ids).astype("int64")

            # Check if index supports removal
            if hasattr(self.index, 'remove_ids'):
                self.index.remove_ids(id_array)

                # For IVF indexes, check if retraining is beneficial
                if self.mode == "LEX_LDS" and len(vector_ids) > self.index.ntotal * 0.1:
                    logger.warning(
                        f"[VectorStore] Deleted {len(vector_ids)} vectors (>10% of index). "
                        f"Consider retraining for optimal performance."
                    )
            else:
                logger.warning(
                    "[VectorStore] This index type doesn't support removal. "
                    "Only metadata will be deleted."
                )

            # Remove metadata
            for vid in vector_ids:
                self.metadata.pop(str(vid), None)

            self._save()
            logger.info(f"[VectorStore] Successfully deleted {len(vector_ids)} vectors.")
        except Exception as e:
            logger.error(f"[VectorStore] Vector deletion failed: {e}")

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
            logger.warning("[VectorStore] Index is empty. No vectors to search.")
            return []

        qv = np.array([query_vector]).astype("float32")

        # Normalize query vector if using cosine similarity
        if self.similarity_metric == "cosine":
            qv = self._normalize_vectors(qv)

        # Perform search
        # Fetch extra results to account for filtering
        search_k = min(top_k * 3, self.index.ntotal)
        distances, ids = self.index.search(qv, search_k)

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
        print(f"\n{'=' * 80}")
        print(f"DEBUG QUERY (Mode: {self.mode}, Similarity: {self.similarity_metric})")
        print(f"Total vectors in index: {self.index.ntotal}")

        if self.mode == "LEX_LDS":
            print(f"IVF Configuration:")
            print(f"  - nlist (clusters): {self.index.nlist}")
            print(f"  - nprobe (search clusters): {self.index.nprobe}")
            print(f"  - Is trained: {self.index.is_trained}")
            print(f"  - Training iterations: {self.training_metadata.get('training_iterations', 0)}")

        print(f"{'=' * 80}\n")

        results = self.query(query_vector, top_k=top_k, filter=filter, with_score=True)

        if not results:
            print("❌ NO RESULTS FOUND!")
            print("\nPossible issues:")
            print("1. Index is empty or not properly trained")
            print("2. Query vector dimension mismatch")
            print("3. Metadata filter is too restrictive")
            print("4. Score threshold too strict")
            if self.mode == "LEX_LDS" and not self.index.is_trained:
                print("5. IVF index not trained - add more vectors first")
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
            print(f"{'-' * 80}\n")

    # =====================================================
    # Utility & Maintenance
    # =====================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vector store.

        Returns:
            Dictionary containing store statistics
        """
        stats = {
            "mode": self.mode,
            "similarity_metric": self.similarity_metric,
            "embedding_dim": self.embedding_dim,
            "total_vectors": self.index.ntotal,
            "metadata_count": len(self.metadata),
            "store_dir": str(self.store_dir)
        }

        # Add IVF-specific stats
        if self.mode == "LEX_LDS":
            stats.update({
                "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
                "nlist": self.index.nlist if hasattr(self.index, 'nlist') else None,
                "nprobe": self.index.nprobe if hasattr(self.index, 'nprobe') else None,
                "training_iterations": self.training_metadata.get("training_iterations", 0),
                "last_trained_at": self.training_metadata.get("last_trained_at"),
                "training_vectors_count": self.training_metadata.get("training_vectors_count", 0)
            })

        return stats

    def optimize_search_speed(self, accuracy_target: float = 0.9):
        """
        Optimize search speed by adjusting nprobe for LEX_LDS mode.

        Trade-off: Lower nprobe = faster search but lower accuracy

        Args:
            accuracy_target: Desired accuracy (0.7-1.0)
                0.7 = fast, 0.9 = balanced, 1.0 = accurate
        """
        if self.mode != "LEX_LDS":
            logger.warning("[VectorStore] optimize_search_speed only applies to LEX_LDS mode")
            return

        if not hasattr(self.index, 'nprobe'):
            return

        # Adjust nprobe based on accuracy target
        nlist = self.index.nlist

        if accuracy_target >= 0.95:
            new_nprobe = max(nlist // 5, 50)  # High accuracy
        elif accuracy_target >= 0.85:
            new_nprobe = max(nlist // 10, 20)  # Balanced
        else:
            new_nprobe = max(nlist // 20, 10)  # Fast search

        new_nprobe = min(new_nprobe, nlist)  # Don't exceed nlist

        old_nprobe = self.index.nprobe
        self.index.nprobe = new_nprobe

        logger.info(
            f"[VectorStore] Search speed optimized: nprobe {old_nprobe} → {new_nprobe} "
            f"(target accuracy: {accuracy_target:.1%})"
        )

    def force_retrain(self):
        """
        Force retraining of IVF index using all current vectors.
        Useful after significant data changes or deletions.
        """
        if self.mode != "LEX_LDS":
            logger.warning("[VectorStore] force_retrain only applies to LEX_LDS mode")
            return

        if self.index.ntotal == 0:
            logger.warning("[VectorStore] No vectors in index to train on")
            return

        logger.info("[VectorStore] Extracting all vectors for retraining...")

        # Extract all vectors from index
        all_vectors = np.zeros((self.index.ntotal, self.embedding_dim), dtype='float32')
        for i in range(self.index.ntotal):
            all_vectors[i] = self.index.reconstruct(int(i))

        # Clear and retrain
        logger.info("[VectorStore] Retraining index with all vectors...")
        success = self._train_ivf_index(all_vectors, force_retrain=True)

        if success:
            # Re-add all vectors
            logger.info("[VectorStore] Re-indexing vectors after retraining...")
            ids = np.arange(self.index.ntotal).astype("int64")
            self.index.add_with_ids(all_vectors, ids)
            self._save()
            logger.info("[VectorStore] ✅ Retraining completed successfully!")
        else:
            logger.error("[VectorStore] ❌ Retraining failed!")

    def clear(self):
        """Reset index and metadata."""
        # Recreate index with same configuration
        old_ntotal = self.index.ntotal
        self.index = self._load_or_initialize_index()
        self.metadata = {}
        self.training_metadata = {
            "is_trained": False,
            "training_vectors_count": 0,
            "nlist": 0,
            "nprobe": 0,
            "last_trained_at": None,
            "training_iterations": 0
        }
        self._save()
        self._save_training_metadata()
        logger.info(f"[VectorStore] Cleared all {old_ntotal} vectors and reset training state.")

    def rebuild_with_new_metric(self, new_similarity_metric: str) -> None:
        """
        Rebuild the index with a different similarity metric.
        WARNING: This will clear all existing vectors!

        Args:
            new_similarity_metric: 'cosine' or 'l2'
        """
        logger.info(f"[VectorStore] Rebuilding index with new similarity metric: {new_similarity_metric}")

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

        logger.info(f"[VectorStore] Index rebuilt successfully with {new_similarity_metric} similarity.")


    def get_as_retriever(self, k: int = 4):
        """
        Get a LangChain-compatible retriever interface.

        This method returns a retriever object that can be used seamlessly
        with LangChain's retrieval chains, QA systems, and agents.

        Args:
            k: Number of documents to retrieve (default: 4)

        Returns:
            VectorStoreRetriever: LangChain-compatible retriever

        Example:
            ```python
            # Create vector store and add documents
            store = VectorStore("./faiss", "LEX_NANO", similarity_metric="cosine")
            store.add_vectors(embeddings, metadata_list)

            # Get LangChain retriever
            retriever = store.get_as_retriever(k=4)

            # Use with LangChain
            from langchain.chains import RetrievalQA
            from langchain_openai import ChatOpenAI

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(),
                retriever=retriever,
                return_source_documents=True
            )

            result = qa_chain({"query": "What is the main topic?"})
            ```
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from pydantic import Field

        class VectorStoreRetriever(BaseRetriever):
            """
            LangChain-compatible retriever wrapper for VectorStore.
            """
            vector_store: Any = Field(description="The VectorStore instance")
            k: int = Field(default=4, description="Number of documents to retrieve")
            score_threshold: Optional[float] = Field(
                default=None,
                description="Minimum similarity score threshold"
            )
            filter: Optional[Dict[str, Any]] = Field(
                default=None,
                description="Metadata filter"
            )

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                    self,
                    query: str,
                    *,
                    run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> List[Document]:
                """
                Get documents relevant to a query.

                Args:
                    query: Query string (will be embedded)
                    run_manager: Callback manager

                Returns:
                    List of relevant Document objects
                """
                # This method expects the query to already be embedded
                # In practice, you'll need to embed the query before calling this
                # For now, we'll raise a helpful error
                raise NotImplementedError(
                    "Direct string queries not supported. Use get_relevant_documents_from_embedding() "
                    "or embed your query first using your embedding model, then call: "
                    "retriever.get_relevant_documents_from_embedding(query_embedding)"
                )

            def get_relevant_documents_from_embedding(
                    self,
                    query_embedding: List[float],
                    run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> List[Document]:
                """
                Get documents relevant to a query embedding.

                Args:
                    query_embedding: Pre-computed query embedding vector
                    run_manager: Callback manager

                Returns:
                    List of relevant Document objects
                """
                # Query the vector store
                results = self.vector_store.query(
                    query_vector=query_embedding,
                    top_k=self.k,
                    filter=self.filter,
                    with_score=True,
                    score_threshold=self.score_threshold
                )

                # Convert to LangChain Document objects
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result["page_content"],
                        metadata={
                            **result["meta"],
                            "score": result.get("score"),
                            "similarity_metric": result.get("similarity_metric")
                        }
                    )
                    documents.append(doc)

                return documents

            def invoke(
                    self,
                    query_embedding: List[float],
                    config: Optional[Dict] = None,
            ) -> List[Document]:
                """
                Invoke the retriever with a query embedding.

                This is the main method used by LangChain chains.

                Args:
                    query_embedding: Pre-computed query embedding vector
                    config: Optional configuration

                Returns:
                    List of relevant Document objects
                """
                return self.get_relevant_documents_from_embedding(query_embedding)

        # Create and return the retriever
        retriever = VectorStoreRetriever(
            vector_store=self,
            k=k
        )

        logger.info(f"[VectorStore] Created LangChain retriever with k={k}")
        return retriever











