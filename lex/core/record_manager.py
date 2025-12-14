"""
RecordManager
==============

Maintains `record.json` for tracking file changes, hashes,
vector embeddings, and sync status.
"""

import json
import os
import hashlib
import time
from pathlib import Path


class RecordManager:
    """Manages file change records stored in record.json."""

    DEFAULT_FILE = "record.json"

    def __init__(self, record_path: str):
        self.record_dir = Path(record_path)
        self.record_file = self.record_dir / self.DEFAULT_FILE
        self.data = {"files": {}, "last_sync": None}

        self._ensure_record_file()
        self.load()

    # =====================================================
    # Internal Utilities
    # =====================================================

    def _ensure_record_file(self):
        """Ensure record.json exists and is initialized."""
        os.makedirs(self.record_dir, exist_ok=True)
        if not self.record_file.exists():
            with open(self.record_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)

    def _atomic_write(self):
        """Safely write to record.json atomically."""
        temp_file = self.record_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
        os.replace(temp_file, self.record_file)

    def _hash_file(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            return ""

    # =====================================================
    # Core Methods
    # =====================================================

    def load(self):
        """Load record.json from disk."""
        try:
            with open(self.record_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Reinitialize if corrupted
            self.data = {"files": {}, "last_sync": None}
            self._atomic_write()

    def save(self):
        """Persist record.json to disk."""
        self._atomic_write()

    def update_entry(self, file_path: str, metadata: dict):
        """Update a single file entry."""
        abs_path = str(Path(file_path).resolve())
        self.data["files"][abs_path] = {
            **self.data["files"].get(abs_path, {}),
            **metadata,
        }
        self.data["last_sync"] = int(time.time())
        self.save()

    def remove_entry(self, file_path: str):
        """Remove file record (e.g., when deleted)."""
        abs_path = str(Path(file_path).resolve())
        if abs_path in self.data["files"]:
            del self.data["files"][abs_path]
            self.data["last_sync"] = int(time.time())
            self.save()

    def get_entry(self, file_path: str) -> dict:
        """Return metadata for a file, or None if not tracked."""
        abs_path = str(Path(file_path).resolve())
        return self.data["files"].get(abs_path)

    def get_all_files(self) -> dict:
        """Return all tracked file metadata."""
        return self.data["files"]

    def mark_deleted(self, file_path: str):
        """Mark a file as deleted in record.json."""
        abs_path = str(Path(file_path).resolve())
        if abs_path in self.data["files"]:
            self.data["files"][abs_path]["status"] = "deleted"
            self.save()

    def mark_synced(self, file_path: str, vector_ids=None):
        """Mark file as fully synced with optional FAISS vector IDs."""
        abs_path = str(Path(file_path).resolve())
        if abs_path in self.data["files"]:
            entry = self.data["files"][abs_path]
            entry["status"] = "synced"
            entry["vector_ids"] = vector_ids or []
            self.save()

    def register_file(self, file_path: str):
        """Register or update file status by hash comparison."""
        abs_path = str(Path(file_path).resolve())
        file_hash = self._hash_file(abs_path)
        last_mod = os.path.getmtime(abs_path) if os.path.exists(abs_path) else 0

        existing = self.data["files"].get(abs_path)
        if not existing:
            status = "new"
        elif existing.get("hash") != file_hash:
            status = "changed"
        else:
            status = "synced"

        entry = {
            "status": status,
            "hash": file_hash,
            "last_modified": int(last_mod),
            "vector_ids": existing.get("vector_ids", []) if existing else [],
            "meta": existing.get("meta", {}) if existing else {},
        }

        self.update_entry(abs_path, entry)
        return status

    def cleanup_stale_records(self):
        """Remove entries whose files no longer exist."""
        to_remove = []
        for path in list(self.data["files"].keys()):
            if not os.path.exists(path):
                to_remove.append(path)
        for path in to_remove:
            self.mark_deleted(path)
        if to_remove:
            self.save()
        return to_remove
