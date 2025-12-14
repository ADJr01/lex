"""
InstanceConfig
==============

Provides a fluent interface for configuring Lexi instances.
Handles embedding, storage paths, mode selection, and sync strategy.
"""

import os
from pathlib import Path


class InstanceConfig:
    """Handles configuration for a Lexi Chrono instance."""

    class MODES:
        LEX_NANO = "LEX_NANO"  # Lightweight, in-memory FAISS (L2)
        LEX_LDS = "LEX_LDS"    # Large-scale FAISS (IVF)

    class RESPONSE_MODES:
        FAST = "FAST"  # Prioritize speed
        DEEP = "DEEP"  # Prioritize semantic accuracy

    class CHUNK_MECHANISM:
        SEMANTIC_CHUNK = "SEMANTIC_CHUNK"  # Semantic boundary-based
        HYBRID = "HYBRID"                  # Semantic + token fallback

    def __init__(self, instance_name: str):
        self.instance_name = instance_name
        self.embedding = None
        self.record_path = None
        self.storage_dir = None
        self.vector_store_dir = None
        self.chunking_strategy = self.CHUNK_MECHANISM.SEMANTIC_CHUNK
        self.mode = {
            "mode": self.MODES.LEX_NANO,
            "response_mode": self.RESPONSE_MODES.FAST
        }

    # =========================
    #   Fluent Setters
    # =========================

    def set_embbeding(self, embedding_instance):
        """Attach embedding model instance."""
        self.embedding = embedding_instance
        return self

    def set_record_path(self, path: str):
        """Set path for record.json and related metadata."""
        self.record_path = str(Path(path).resolve())
        os.makedirs(self.record_path, exist_ok=True)
        return self

    def set_storage_dir(self, path: str):
        """Set recursive storage directory for source files."""
        self.storage_dir = str(Path(path).resolve())
        os.makedirs(self.storage_dir, exist_ok=True)
        return self

    def set_vector_store_dir(self, path: str):
        """Set directory where FAISS index files are stored."""
        self.vector_store_dir = str(Path(path).resolve())
        os.makedirs(self.vector_store_dir, exist_ok=True)
        return self

    def set_chunking_strategy(self, strategy: str):
        """Set chunking strategy (semantic or hybrid)."""
        valid = [self.CHUNK_MECHANISM.SEMANTIC_CHUNK, self.CHUNK_MECHANISM.HYBRID]
        if strategy not in valid:
            raise ValueError(f"Invalid chunking strategy: {strategy}")
        self.chunking_strategy = strategy
        return self

    def set_mode(self, mode_config: dict):
        """Set Lex operational mode and response mode."""
        mode = mode_config.get("mode")
        response_mode = mode_config.get("response_mode")

        if mode not in [self.MODES.LEX_NANO, self.MODES.LEX_LDS]:
            raise ValueError("Invalid mode. Must be LEX_NANO or LEX_LDS.")
        if response_mode not in [self.RESPONSE_MODES.FAST, self.RESPONSE_MODES.DEEP]:
            raise ValueError("Invalid response_mode. Must be FAST or DEEP.")

        self.mode = {
            "mode": mode,
            "response_mode": response_mode
        }
        return self

    # =========================
    #   Validation & Accessors
    # =========================

    def validate(self):
        """Validate that all essential parameters are configured."""
        missing = []
        if not self.embedding:
            missing.append("embedding")
        if not self.record_path:
            missing.append("record_path")
        if not self.storage_dir:
            missing.append("storage_dir")
        if not self.vector_store_dir:
            missing.append("vector_store_dir")

        if missing:
            raise ValueError(f"Missing required configuration fields: {missing}")
        return True

    def as_dict(self):
        """Return configuration as a serializable dictionary."""
        return {
            "instance_name": self.instance_name,
            "record_path": self.record_path,
            "storage_dir": self.storage_dir,
            "vector_store_dir": self.vector_store_dir,
            "chunking_strategy": self.chunking_strategy,
            "mode": self.mode,
            "embedding": str(type(self.embedding)) if self.embedding else None,
        }

    def __repr__(self):
        return (
            f"<InstanceConfig name={self.instance_name} "
            f"mode={self.mode['mode']} strategy={self.chunking_strategy}>"
        )
