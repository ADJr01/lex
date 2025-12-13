import sqlite3
from typing import Dict, Any, List


class MetadataStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT,
                directory TEXT,
                content_type TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def add_chunk(self, chunk_id: str, record: Dict[str, Any]):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)
        """, (
            chunk_id,
            record["file_path"],
            record["directory"],
            record["content_type"],
            str(record)
        ))
        self.conn.commit()

    def delete_by_file(self, file_path: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM chunks WHERE file_path = ?",
            (file_path,)
        )
        self.conn.commit()

    def query_by_directory(self, directory: str) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT chunk_id FROM chunks WHERE directory LIKE ?",
            (f"{directory}%",)
        )
        return [row[0] for row in cursor.fetchall()]

    def stats(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]

    def close(self):
        self.conn.close()
