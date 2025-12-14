"""
Chunker
=======

Performs semantic chunking using LangChain-supported loaders and chunkers.
Supports semantic or hybrid strategies for different file types.
"""

import os
import json
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from semantic_chunker_langchain.chunker import SemanticChunker
import csv


class Chunker:
    """Handles file chunking for supported formats using LangChain."""

    SUPPORTED_EXTENSIONS = [".txt", ".json", ".pdf", ".csv", ".docx", ".xls", ".xlsx"]

    def __init__(self, strategy="SEMANTIC_CHUNK"):
        """
        Initialize chunker.

        Args:
            strategy (str): 'SEMANTIC_CHUNK' or 'HYBRID'
        """
        self.strategy = strategy

    # =====================================================
    # File Loading Utilities
    # =====================================================

    def _load_text(self, file_path: str) -> str:
        """Read plain text or JSON file."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            print(f"[Chunker] Failed to load {file_path}: {e}")
            return ""

    def _load_csv(self, file_path: str) -> str:
        """Convert CSV rows to text."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                lines = [" | ".join(row) for row in reader]
            return "\n".join(lines)
        except Exception as e:
            print(f"[Chunker] Failed to load CSV {file_path}: {e}")
            return ""

    def _load_doc(self, file_path: str) -> str:
        """Use LangChain document loaders for structured files."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext in [".xls", ".xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[Chunker] Failed to load document {file_path}: {e}")
            return ""

    def _load_file(self, file_path: str) -> str:
        """Auto-select appropriate loader based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".json"]:
            return self._load_text(file_path)
        elif ext == ".csv":
            return self._load_csv(file_path)
        else:
            return self._load_doc(file_path)

    # =====================================================
    # Chunking Logic
    # =====================================================

    def chunk_file(self, file_path: str, embedding_model=None):
        """
        Chunk a file into semantic or hybrid chunks.

        Args:
            file_path (str): Path to the file
            embedding_model: Optional embedding model for semantic chunking
        Returns:
            List[dict]: [{"text": str, "meta": {...}}]
        """
        if not os.path.exists(file_path):
            print(f"[Chunker] File not found: {file_path}")
            return []

        text = self._load_file(file_path)
        if not text.strip():
            print(f"[Chunker] Empty or unreadable file: {file_path}")
            return []

        try:
            if self.strategy == "SEMANTIC_CHUNK" and embedding_model:
                splitter = SemanticChunker(embedding_model)
                chunks = splitter.split_text(text)
            else:
                # Hybrid: use semantic if possible, else fallback to token-based
                try:
                    splitter = SemanticChunker(embedding_model)
                    chunks = splitter.split_text(text)
                except Exception:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=100
                    )
                    chunks = splitter.split_text(text)
        except Exception as e:
            print(f"[Chunker] Chunking failed for {file_path}: {e}")
            chunks = [text]

        result = []
        for idx, chunk in enumerate(chunks):
            result.append({
                "text": chunk,
                "meta": {
                    "source": str(Path(file_path).resolve()),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                },
            })
        print(f"[Chunker] Chunked {file_path} → {len(result)} chunks")
        return result
