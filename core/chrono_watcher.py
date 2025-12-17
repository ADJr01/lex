"""
ChronoWatcher
=============

Observes storage directories and detects file changes.
Integrates with RecordManager to keep `record.json` updated.
Supports both live watching and manual sync.
"""

import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from util.Contants import EXTENSION_SUPPORTED_CONSTANT


class ChronoWatcher(FileSystemEventHandler):
    """Watches directories for changes and updates RecordManager."""

    SUPPORTED_EXTENSIONS = EXTENSION_SUPPORTED_CONSTANT

    def __init__(self, storage_dir, record_manager, auto_start=False):
        """
        Initialize the watcher.

        Args:
            storage_dir (str): Directory to monitor (recursively)
            record_manager (RecordManager): The record.json handler
            auto_start (bool): Whether to begin watching immediately
        """
        self.storage_dir = Path(storage_dir)
        self.record_manager = record_manager
        self.observer = None
        self.is_running = False

        if auto_start:
            self.start()

    # =====================================================
    # Utility Methods
    # =====================================================

    def _is_supported(self, file_path: str) -> bool:
        """Check if file type is supported."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.SUPPORTED_EXTENSIONS

    # =====================================================
    # Watchdog Event Handlers
    # =====================================================

    def on_created(self, event):
        """Handle file creation."""
        if event.is_directory or not self._is_supported(event.src_path):
            return
        status = self.record_manager.register_file(event.src_path)
        print(f"[ChronoWatcher] New file detected: {event.src_path} → {status}")

    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory or not self._is_supported(event.src_path):
            return
        status = self.record_manager.register_file(event.src_path)
        print(f"[ChronoWatcher] File modified: {event.src_path} → {status}")

    def on_deleted(self, event):
        """Handle file deletion."""
        if event.is_directory or not self._is_supported(event.src_path):
            return
        self.record_manager.mark_deleted(event.src_path)
        print(f"[ChronoWatcher] File deleted: {event.src_path}")

    # =====================================================
    # Manual Sync / Full Scan
    # =====================================================

    def scan_once(self):
        """
        Perform a one-time scan of the storage directory.

        Detects new/changed/deleted files and updates record.json.
        """
        print("[ChronoWatcher] Performing manual scan...")

        existing_paths = set(self.record_manager.get_all_files().keys())
        found_paths = set()

        for root, _, files in os.walk(self.storage_dir):
            for fname in files:
                file_path = os.path.join(root, fname)
                if not self._is_supported(file_path):
                    continue
                found_paths.add(str(Path(file_path).resolve()))
                status = self.record_manager.register_file(file_path)
                if status in ["new", "changed"]:
                    print(f"[ChronoWatcher] {status.upper()} → {file_path}")

        # Handle deletions (files no longer exist)
        deleted = existing_paths - found_paths
        for path in deleted:
            self.record_manager.mark_deleted(path)
            print(f"[ChronoWatcher] DELETED → {path}")

        self.record_manager.save()
        print("[ChronoWatcher] Manual scan completed.")

    # =====================================================
    # Watcher Lifecycle
    # =====================================================

    def start(self):
        """Begin watching storage directories recursively."""
        if self.is_running:
            print("[ChronoWatcher] Already running.")
            return

        print(f"[ChronoWatcher] Starting watcher on {self.storage_dir} ...")
        self.observer = Observer()
        self.observer.schedule(self, str(self.storage_dir), recursive=True)
        self.observer.start()
        self.is_running = True
        print("[ChronoWatcher] Watcher started.")

    def stop(self):
        """Stop watching."""
        if not self.is_running or not self.observer:
            print("[ChronoWatcher] Watcher not running.")
            return

        print("[ChronoWatcher] Stopping watcher...")
        self.observer.stop()
        self.observer.join()
        self.is_running = False
        print("[ChronoWatcher] Watcher stopped.")

    def restart(self):
        """Restart the watcher."""
        self.stop()
        time.sleep(0.5)
        self.start()
