"""
Lex (Core Engine)
=================

Orchestrates the entire Chrono Engine:
- Detects and tracks file changes
- Performs semantic chunking
- Generates embeddings
- Updates FAISS vector store
- Provides unified query interface
"""

import os
from lex.core.record_manager import RecordManager
from lex.core.chrono_watcher import ChronoWatcher
from lex.core.chunker import Chunker
from lex.core.embedder import Embedder
from lex.core.vector_store import VectorStore
from lex.core.query_service import QueryService


class Lex:
    """Main entry point for the Lexi Chrono Engine."""

    def __init__(self, config):
        self.config = config
        self.config.validate()
        self.record_manager = RecordManager(config.record_path)
        self.chunker = Chunker(strategy=config.chunking_strategy)
        self.embedder = Embedder(config.embedding)
        test_vec = self.embedder.embed_query("dimension check")
        embedding_dim = len(test_vec) if test_vec else 1536
        self.vector_store = VectorStore(
            config.vector_store_dir,
            mode=config.mode["mode"],
            embedding_dim=embedding_dim
        )
        self.query_service = QueryService(
            self.vector_store,
            self.embedder,
            response_mode=config.mode["response_mode"]
        )

        self.watcher = ChronoWatcher(config.storage_dir, self.record_manager)
        self.is_running = False

        print(f"[Lex] Initialized instance '{config.instance_name}' successfully.")

    # Lifecycle Methods

    def start(self, watch: bool = False):
        """
        Start synchronization and optionally watch directories for live changes.

        Args:
            watch (bool): Whether to continuously watch directories
        """
        print(f"[Lex] Starting sync for instance: {self.config.instance_name}")
        self.sync()  # initial sync
        if watch:
            self.watcher.start()
            self.is_running = True
            print("[Lex] Live watcher enabled.")
        return self

    def sync(self):
        """
        Perform a manual synchronization of all supported files.
        Detects new/changed/deleted files and updates FAISS accordingly.
        """
        print("[Lex] Performing full manual sync...")
        self.watcher.scan_once()
        all_files = self.record_manager.get_all_files()

        for file_path, meta in all_files.items():
            status = meta.get("status")
            if status in ["new", "changed"]:
                self._process_file(file_path)
            elif status == "deleted":
                ids = meta.get("vector_ids", [])
                if ids:
                    self.vector_store.delete_vectors(ids)
                self.record_manager.remove_entry(file_path)

        print("[Lex] Manual sync complete.")

    def close(self):
        """Close all Lex components gracefully."""
        if self.is_running:
            self.watcher.stop()
        self.record_manager.save()
        print(f"[Lex] Instance '{self.config.instance_name}' closed cleanly.")

    # =====================================================
    # Internal Utilities
    # =====================================================

    def _process_file(self, file_path: str):
        """Process a single file (chunk → embed → store → mark synced)."""
        print(f"[Lex] Processing {file_path}...")

        chunks = self.chunker.chunk_file(file_path, self.config.embedding)
        if not chunks:
            print(f"[Lex] No chunks produced for {file_path}.")
            return

        embedded_chunks = self.embedder.embed_chunks(chunks)
        if not embedded_chunks:
            print(f"[Lex] Failed to embed {file_path}.")
            return

        vectors = [ec["vector"] for ec in embedded_chunks]
        metadata_list = [ec["meta"] for ec in embedded_chunks]

        vector_ids = self.vector_store.add_vectors(vectors, metadata_list)
        self.record_manager.mark_synced(file_path, vector_ids)
        print(f"[Lex] File {file_path} synced successfully with {len(vector_ids)} vectors.")

    # =====================================================
    # Query Interface
    # =====================================================

    def vector_store_api(self):
        """
        Return a QueryService interface for querying the vector store.
        Example:
            query = lex.vector_store_api()
            query.invoke("Find documents about sales")
        """
        return self.query_service
