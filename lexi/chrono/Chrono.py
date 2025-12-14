import os
import sys
import sqlite3
import hashlib
import logging
from threading import Lock
from pathlib import Path
from typing import List, Dict, Optional, Any

from lexi.core.constants import FileStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Chrono:
    """
    Synchronize file changes in directories with an SQLite database.
    Tracks additions, modifications, and deletions for RAG applications.
    """

    # Status codes
    STATUS_NEW = 0
    STATUS_CHANGED = 1
    STATUS_UNCHANGED = -1
    STATUS_DELETED = 2

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Chrono with configuration.

        Args:
            config: Dictionary with keys:
                - supported_file_types (List[str]): File extensions to track (e.g., ['.txt', '.pdf'])
                - db_path (str): Path to SQLite database file
                - scan_dirs (List[str]): Directories to scan
                - in_memory (bool, optional): Use in-memory DB (default: False)
                - use_hash_for_changes (bool, optional): Use SHA-256 for change detection (default: False)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = Lock()
        self.scanned_changes: List[Dict[str, Any]] = []

        # Validate and extract config
        self._validate_config(config)

        self.supported_file_types = {ext.lower() for ext in config['supported_file_types']}
        self.db_path = config['db_path']
        self.scan_dirs = [os.path.abspath(d) for d in config['scan_dirs']]
        self.in_memory = config.get('in_memory', False)
        self.use_hash_for_changes = config.get('use_hash_for_changes', False)

        # Initialize database
        self._init_db()

        self.logger.info(f"Chrono initialized: {len(self.scan_dirs)} dirs, "
                         f"{len(self.supported_file_types)} file types, "
                         f"hash={self.use_hash_for_changes}, in_memory={self.in_memory}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration dictionary."""
        required_keys = ['supported_file_types', 'db_path', 'scan_dirs']

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: '{key}'")
            if config[key] is None:
                raise ValueError(f"Config key '{key}' cannot be None")

        if not config['supported_file_types']:
            raise ValueError("supported_file_types cannot be empty")

        if not isinstance(config['supported_file_types'], list):
            raise ValueError("supported_file_types must be a list")

        for ext in config['supported_file_types']:
            if not isinstance(ext, str) or not ext.startswith('.'):
                raise ValueError(f"Invalid file extension: '{ext}'. Must start with '.'")

        # Validate db_path
        if not isinstance(config['db_path'], str):
            raise ValueError(f"db_path must be a string, got {type(config['db_path'])}")

        # Check if db_path is a directory (common mistake)
        if config['db_path'].endswith(os.sep) or config['db_path'].endswith('/') or config['db_path'].endswith('\\'):
            raise ValueError(f"db_path must be a file path, not a directory: '{config['db_path']}'. "
                             f"Example: 'D:/chrono/chrono.db' not 'D:/chrono/'")

        # Check if db_path has a file extension
        if '.' not in os.path.basename(config['db_path']):
            raise ValueError(f"db_path should include a file name with extension: '{config['db_path']}'. "
                             f"Example: '{config['db_path']}/chrono.db'")

        if not config['scan_dirs']:
            raise ValueError("scan_dirs cannot be empty")

        if not isinstance(config['scan_dirs'], list):
            raise ValueError("scan_dirs must be a list")

        for directory in config['scan_dirs']:
            if not os.path.isabs(directory):
                raise ValueError(f"Directory must be absolute path: '{directory}'")
            if not os.path.exists(directory):
                raise ValueError(f"Directory does not exist: '{directory}'")
            if not os.path.isdir(directory):
                raise ValueError(f"Path is not a directory: '{directory}'")

    def _init_db(self) -> None:
        """Initialize SQLite database and create schema."""
        db_location = ':memory:' if self.in_memory else self.db_path

        # Create parent directory if needed (only for file-based DB)
        if not self.in_memory and self.db_path:
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    self.logger.info(f"Created database directory: {db_dir}")
                except OSError as e:
                    raise ValueError(f"Cannot create database directory '{db_dir}': {e}")

        try:
            self.conn = sqlite3.connect(db_location, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError as e:
            raise ValueError(f"Cannot open database at '{db_location}': {e}. "
                             f"Ensure the directory exists and you have write permissions.")

        cursor = self.conn.cursor()

        # Create table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS file_records
                       (
                           file_path
                           TEXT
                           PRIMARY
                           KEY,
                           file_name
                           TEXT
                           NOT
                           NULL,
                           file_type
                           TEXT
                           NOT
                           NULL,
                           file_size_kb
                           INTEGER
                           NOT
                           NULL,
                           last_modified
                           REAL
                           NOT
                           NULL,
                           content_hash
                           TEXT,
                           file_id
                           TEXT
                       )
                       """)

        # Create index
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_file_path ON file_records(file_path)
                       """)

        self.conn.commit()
        self.logger.debug(f"Database initialized at {db_location}")

    def _generate_file_id(self, file_path: str) -> str:
        """Generate unique file ID based on inode and device."""
        try:
            stat = os.stat(file_path)
            return f"{stat.st_ino}-{stat.st_dev}"
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()  # fallback

    def _compute_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA-256 hash of file contents."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(65536), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (IOError, OSError) as e:
            self.logger.warning(f"Failed to hash file {file_path}: {e}")
            return None

    def _get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata for a file."""
        try:
            stat = os.stat(file_path)
            size_kb = round(stat.st_size / 1024)
            last_modified = stat.st_mtime

            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_type': os.path.splitext(file_path)[1].lower(),
                'file_size_kb': size_kb,
                'last_modified': last_modified,
                'content_hash': None,
                'file_id': self._generate_file_id(file_path),
            }

            if self.use_hash_for_changes:
                metadata['content_hash'] = self._compute_hash(file_path)

            return metadata

        except (IOError, OSError, PermissionError) as e:
            self.logger.warning(f"Cannot access file {file_path}: {e}")
            return None

    def status(self) -> Dict[str, Any]:
        """
        Return current Chrono status and resource usage.
        """
        # Directories tracked
        tracked_dirs = self.scan_dirs

        # DB size
        if self.in_memory:
            db_size = "in-memory"
        else:
            try:
                db_size = os.path.getsize(self.db_path)
            except OSError:
                db_size = 0

        # Process memory (try psutil, fallback to sys.getsizeof)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_bytes = process.memory_info().rss
        except (ImportError, Exception):
            memory_bytes = sys.getsizeof(self.__dict__)

        return {
            "tracked_directories": tracked_dirs,
            "database_path": self.db_path,
            "database_size_bytes": db_size,
            "process_memory_bytes": memory_bytes,
            "use_hash_for_changes": self.use_hash_for_changes,
            "in_memory": self.in_memory
        }

    def _load_existing_records(self) -> Dict[str, Dict[str, Any]]:
        """Load existing file records from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM file_records")

        records = {}
        for row in cursor.fetchall():
            records[row['file_path']] = dict(row)

        return records

    def _compare_metadata(self, current: Dict[str, Any], existing: Dict[str, Any]) -> int:
        """
        Compare current file metadata with existing record.
        Returns status code.
        """
        # Check size and modification time
        if current['file_size_kb'] != existing['file_size_kb']:
            return self.STATUS_CHANGED

        if abs(current['last_modified'] - existing['last_modified']) > 0.01:
            return self.STATUS_CHANGED

        # Check hash if enabled
        if self.use_hash_for_changes:
            if current['content_hash'] != existing['content_hash']:
                return self.STATUS_CHANGED

        return self.STATUS_UNCHANGED

    def scan(self) -> List[Dict[str, Any]]:
        """
        Scan directories for file changes.

        Returns:
            List of dictionaries containing file metadata and status codes.
        """
        with self._lock:
            self.logger.info("Starting directory scan...")
            self.scanned_changes = []

            # Load existing records
            existing_records = self._load_existing_records()
            scanned_paths = set()

            # Scan directories
            for scan_dir in self.scan_dirs:
                self._scan_directory(scan_dir, existing_records, scanned_paths)

            # Detect deletions
            self._detect_deletions(existing_records, scanned_paths)

            self.logger.info(f"Scan complete: {len(self.scanned_changes)} changes detected")
            self._log_summary()

            return self.scanned_changes

    def _scan_directory(self, root_dir: str, existing_records: Dict[str, Dict],
                        scanned_paths: set) -> None:
        """Scan a single directory recursively."""
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
                # Filter out hidden directories
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]

                for filename in filenames:
                    # Skip hidden files
                    if filename.startswith('.'):
                        continue

                    file_path = os.path.join(dirpath, filename)
                    file_ext = os.path.splitext(filename)[1].lower()

                    # Check if supported file type
                    if file_ext not in self.supported_file_types:
                        continue

                    scanned_paths.add(file_path)

                    # Get metadata
                    metadata = self._get_file_metadata(file_path)
                    if not metadata:
                        continue

                    # Determine status
                    if file_path in existing_records:
                        status = self._compare_metadata(metadata, existing_records[file_path])
                    else:
                        status = self.STATUS_NEW

                    # Only add if changed
                    if status != self.STATUS_UNCHANGED:
                        metadata['status'] = status
                        self.scanned_changes.append(metadata)

        except (IOError, OSError, PermissionError) as e:
            self.logger.error(f"Error scanning directory {root_dir}: {e}")

    def _detect_deletions(self, existing_records: Dict[str, Dict],
                          scanned_paths: set) -> None:
        """Detect files that were deleted from disk."""
        for file_path, record in existing_records.items():
            # Check if file is under scan_dirs
            is_under_scan_dir = any(
                file_path.startswith(scan_dir) for scan_dir in self.scan_dirs
            )

            if is_under_scan_dir and file_path not in scanned_paths:
                deletion_record = {
                    'file_path': file_path,
                    'file_name': record['file_name'],
                    'file_type': record['file_type'],
                    'file_size_kb': record['file_size_kb'],
                    'status': self.STATUS_DELETED
                }
                self.scanned_changes.append(deletion_record)

    def _log_summary(self) -> None:
        """Log summary of detected changes."""
        summary = {
            self.STATUS_NEW: 0,
            self.STATUS_CHANGED: 0,
            self.STATUS_DELETED: 0
        }

        for change in self.scanned_changes:
            status = change['status']
            if status in summary:
                summary[status] += 1

        self.logger.info(f"Summary - New: {summary[self.STATUS_NEW]}, "
                         f"Changed: {summary[self.STATUS_CHANGED]}, "
                         f"Deleted: {summary[self.STATUS_DELETED]}")



    def commit(self) -> None:
        """
        Commit scanned changes to database.
        Applies INSERT, UPDATE, and DELETE operations based on status codes.
        """
        with self._lock:
            if not self.scanned_changes:
                self.logger.info("No changes to commit")
                return

            self.logger.info(f"Committing {len(self.scanned_changes)} changes...")

            try:
                cursor = self.conn.cursor()

                # Group changes by status for batch operations
                new_files = []
                changed_files = []
                deleted_files = []

                for change in self.scanned_changes:
                    if change['status'] == self.STATUS_NEW:
                        new_files.append(change)
                    elif change['status'] == self.STATUS_CHANGED:
                        changed_files.append(change)
                    elif change['status'] == self.STATUS_DELETED:
                        deleted_files.append(change)

                # Begin transaction
                cursor.execute("BEGIN TRANSACTION")

                # Insert new files
                if new_files:
                    cursor.executemany("""
                                       INSERT INTO file_records
                                       (file_path, file_name, file_type, file_size_kb, last_modified, content_hash,
                                        file_id)
                                       VALUES (?, ?, ?, ?, ?, ?, ?)
                                       """, [(f['file_path'], f['file_name'], f['file_type'],
                                              f['file_size_kb'], f['last_modified'], f['content_hash'], f['file_id'])
                                             for f in new_files])
                    self.logger.info(f"Inserted {len(new_files)} new records")

                # Update changed files
                if changed_files:
                    cursor.executemany("""
                                       UPDATE file_records
                                       SET file_name     = ?,
                                           file_type     = ?,
                                           file_size_kb  = ?,
                                           last_modified = ?,
                                           content_hash  = ?,
                                           file_id       = ?
                                       WHERE file_path = ?
                                       """, [(f['file_name'], f['file_type'], f['file_size_kb'],
                                              f['last_modified'], f['content_hash'], f['file_id'], f['file_path'])
                                             for f in changed_files])
                    self.logger.info(f"Updated {len(changed_files)} records")

                # Delete removed files
                if deleted_files:
                    cursor.executemany("""
                                       DELETE
                                       FROM file_records
                                       WHERE file_path = ?
                                       """, [(f['file_path'],) for f in deleted_files])
                    self.logger.info(f"Deleted {len(deleted_files)} records")

                # Commit transaction
                self.conn.commit()
                self.logger.info("Commit successful")

                # Clear scanned changes
                self.scanned_changes = []

            except sqlite3.Error as e:
                self.logger.error(f"Database error during commit: {e}")
                self.conn.rollback()
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during commit: {e}")
                self.conn.rollback()
                raise
    def get_active_records(self):
        """
        Return all file records that are NEW or CHANGED.
        """
        cursor = self.conn.execute(
            "SELECT rowid, path, status FROM file_records WHERE status IN (?, ?)",
            (FileStatus.STATUS_NEW, FileStatus.STATUS_CHANGED)
        )
        return [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]

    def update_status(self, record_id: int, new_status: int):
        """
        Update the sync status of a file record in ChronoDB.
        Used by Lexi to mark records as SYNCED or ERROR.
        """
        with self._lock:
            try:
                self.conn.execute(
                    "UPDATE file_records SET status=? WHERE rowid=?",
                    (new_status, record_id)
                )
            except Exception as e:
                print(f"[Chrono] Failed to update status for record {record_id}: {e}")

    def mark_synced(self, file_path: str):
        """
        Mark a file as successfully processed (SYNCED) by its path.
        """
        with self._lock:
            try:
                self.conn.execute(
                    "UPDATE file_records SET status=? WHERE path=?",
                    (FileStatus.STATUS_SYNCED, file_path)
                )
            except Exception as e:
                print(f"[Chrono] Failed to mark {file_path} as synced: {e}")

    def remove_record(self, record_id: int):
        """
        Permanently remove a record from ChronoDB.
        Used when Lexi confirms a file has been deleted and vectors removed.
        """
        with self._lock:
            try:
                self.conn.execute("DELETE FROM file_records WHERE rowid=?", (record_id,))
            except Exception as e:
                print(f"[Chrono] Failed to delete record {record_id}: {e}")

    def query_records(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query file records from scanned list or database.

        Args:
            directory: Optional directory path to filter records.
                      If None, returns all records.

        Returns:
            List of file records matching the directory filter.
        """
        with self._lock:
            # First check if we have scanned changes
            if self.scanned_changes:
                self.logger.debug(f"Returning records from scanned list")
                results = self.scanned_changes
            else:
                # Query from database
                self.logger.debug(f"Querying records from database")
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM file_records")

                results = []
                for row in cursor.fetchall():
                    record = dict(row)
                    # Add status as unchanged since these are existing records
                    record['status'] = self.STATUS_UNCHANGED
                    results.append(record)

            # Filter by directory if specified
            if directory:
                # Normalize directory path
                directory = os.path.abspath(directory)
                if not directory.endswith(os.sep):
                    directory += os.sep

                filtered_results = [
                    record for record in results
                    if record['file_path'].startswith(directory)
                ]

                self.logger.info(f"Found {len(filtered_results)} records in directory: {directory}")
                return filtered_results

            self.logger.info(f"Found {len(results)} total records")
            return results

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception:
            pass  # Silently ignore errors during cleanup


