"""
Chunker
=======

Performs intelligent semantic chunking using LangChain-supported loaders and chunkers.
Enhanced with proper error handling and dependency validation.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from lex.core.Parsers.doc_paresr import DocParser

# Conditional imports with helpful error messages
try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyMuPDFLoader,
    )
except ImportError:
    raise ImportError("Please install: pip install langchain-community")

try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[Warning] PyPDFLoader unavailable. Install: pip install pypdf")

try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("[Warning] Docx2txtLoader unavailable. Install: pip install docx2txt")

try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("[Warning] UnstructuredExcelLoader unavailable. Install: pip install unstructured")

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from semantic_chunker_langchain.chunker import SemanticChunker
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[Warning] SemanticChunker unavailable. Install: pip install semantic-chunker-langchain")


class Chunker:
    """Handles intelligent file chunking for supported formats using LangChain."""

    SUPPORTED_EXTENSIONS = [".txt", ".json", ".pdf", ".csv", ".docx", ".xls", ".xlsx"]

    def __init__(self, strategy="SEMANTIC_CHUNK", adaptive=True):
        """
        Initialize chunker with enhanced capabilities.

        Args:
            strategy (str): 'SEMANTIC_CHUNK' or 'HYBRID'
            adaptive (bool): Enable adaptive chunking based on content structure
        """
        self.strategy = strategy
        self.adaptive = adaptive
        self.chunk_cache = {}
        self._validate_dependencies()
        self.parser = parser = DocParser(None,2000,200,100)


    def _validate_dependencies(self):
        """Check which loaders are available and warn about missing ones."""
        status = {
            "PDF": PDF_AVAILABLE,
            "DOCX": DOCX_AVAILABLE,
            "Excel": EXCEL_AVAILABLE,
            "Semantic": SEMANTIC_AVAILABLE
        }
        missing = [k for k, v in status.items() if not v]
        if missing:
            print(f"[Chunker] Missing optional dependencies for: {', '.join(missing)}")

    # =====================================================
    # File Loading Utilities (FIXED)
    # =====================================================


    def _load_file(self, file_path: str) -> str:
        """Auto-select appropriate loader based on extension."""
        ext = os.path.splitext(file_path)[1].lower()

        self.parser.select_file(file_path)
        docs = self.parser.process_document()

        text_content = []
        for doc in docs:
            # Document objects have page_content attribute
            if hasattr(doc, 'page_content'):
                text_content.append(doc.page_content)
            else:
                text_content.append(str(doc))

        return "\n\n".join(text_content)

    # =====================================================
    # Content Analysis & Smart Preprocessing
    # =====================================================

    def _analyze_content_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze text structure to determine optimal chunking strategy.

        Returns:
            Dict with structure metadata
        """
        analysis = {
            "has_code": bool(re.search(r'```|class |def |function |import |require\(', text)),
            "has_lists": bool(re.search(r'^\s*[-*•]\s+|\d+\.\s+', text, re.MULTILINE)),
            "has_headers": bool(re.search(r'^#{1,6}\s+|^[A-Z][^.!?]*$', text, re.MULTILINE)),
            "has_tables": bool(re.search(r'\|.*\||\t.*\t', text)),
            "has_json": bool(re.search(r'^\s*[{\[]', text.strip())),
            "avg_line_length": len(text) / max(text.count('\n'), 1),
            "total_length": len(text),
            "paragraph_count": len(re.findall(r'\n\s*\n', text)) + 1,
        }
        return analysis

    def _smart_preprocess(self, text: str, analysis: Dict[str, Any]) -> str:
        """Intelligently preprocess text based on structure analysis."""
        # Preserve code blocks
        if analysis["has_code"]:
            text = self._mark_code_boundaries(text)

        # Normalize excessive whitespace while preserving structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # Preserve list structures
        if analysis["has_lists"]:
            text = self._enhance_list_markers(text)

        return text

    def _mark_code_boundaries(self, text: str) -> str:
        """Add boundary markers for code blocks to prevent splitting."""
        pattern = r'(```[\s\S]*?```)'
        return re.sub(pattern, r'\n[CODE_BLOCK_START]\n\1\n[CODE_BLOCK_END]\n', text)

    def _enhance_list_markers(self, text: str) -> str:
        """Enhance list item detection for better chunk boundaries."""
        text = re.sub(r'^(\s*[-*•]\s+)', r'[LIST_ITEM]\1', text, flags=re.MULTILINE)
        return text

    # =====================================================
    # Enhanced Chunking Logic
    # =====================================================

    def _get_adaptive_chunk_size(self, analysis: Dict[str, Any]) -> int:
        """Determine optimal chunk size based on content analysis."""
        base_size = 1000

        if analysis["has_code"]:
            base_size = 1500
        elif analysis["has_tables"]:
            base_size = 800
        elif analysis["avg_line_length"] > 100:
            base_size = 1200
        elif analysis["paragraph_count"] < 5:
            base_size = 2000

        return base_size

    def _create_smart_splitter(self, text: str, embedding_model=None) -> Any:
        """Create an intelligent text splitter based on content analysis."""
        analysis = self._analyze_content_structure(text)
        chunk_size = self._get_adaptive_chunk_size(analysis)

        separators = [
            "\n[CODE_BLOCK_END]\n",
            "\n\n\n",
            "\n\n",
            "\n[LIST_ITEM]",
            "\n",
            ". ",
            " ",
            "",
        ]

        if self.strategy == "SEMANTIC_CHUNK" and embedding_model and SEMANTIC_AVAILABLE:
            try:
                return SemanticChunker(embedding_model)
            except Exception as e:
                print(f"[Chunker] Semantic chunker failed, using hybrid: {e}")

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(200, int(chunk_size * 0.15)),
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def _post_process_chunks(self, chunks: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Post-process chunks to clean up markers and ensure quality."""
        processed = []
        for chunk in chunks:
            chunk = chunk.replace('[CODE_BLOCK_START]', '')
            chunk = chunk.replace('[CODE_BLOCK_END]', '')
            chunk = chunk.replace('[LIST_ITEM]', '')
            chunk = chunk.strip()

            if chunk:
                processed.append(chunk)

        return processed

    def _merge_small_chunks(self, chunks: List[str], min_size: int = 100) -> List[str]:
        """Intelligently merge chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current) < min_size:
                current += "\n\n" + next_chunk
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    def _add_context_overlap(self, chunks: List[str], overlap_sentences: int = 2) -> List[str]:
        """Add contextual overlap between chunks for better retrieval."""
        if len(chunks) <= 1:
            return chunks

        enhanced = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]

            sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
            overlap = ' '.join(sentences[-overlap_sentences:]) if len(sentences) > overlap_sentences else ''

            if overlap:
                enhanced_chunk = f"[Previous context: {overlap}]\n\n{current_chunk}"
                enhanced.append(enhanced_chunk)
            else:
                enhanced.append(current_chunk)

        return enhanced

    # =====================================================
    # Main Chunking Interface
    # =====================================================

    def chunk_file(self, file_path: str, embedding_model=None,
                   add_overlap_context: bool = False) -> List[Dict[str, Any]]:
        """
        Chunk a file into intelligent semantic or hybrid chunks.

        Args:
            file_path (str): Path to the file
            embedding_model: Optional embedding model for semantic chunking
            add_overlap_context (bool): Add contextual overlap between chunks

        Returns:
            List[dict]: [{"text": str, "meta": {...}}]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load file content
        try:
            text = self._load_file(file_path)
        except Exception as e:
            print(f"[Chunker] Error loading {file_path}: {e}")
            raise

        if not text.strip():
            print(f"[Chunker] Warning: Empty or unreadable file: {file_path}")
            return []

        # Analyze content structure
        analysis = self._analyze_content_structure(text)

        # Smart preprocessing
        if self.adaptive:
            text = self._smart_preprocess(text, analysis)

        # Create appropriate splitter and chunk
        try:
            splitter = self._create_smart_splitter(text, embedding_model)
            chunks = splitter.split_text(text)

            # Post-process chunks
            chunks = self._post_process_chunks(chunks, analysis)
            chunks = self._merge_small_chunks(chunks)

            if add_overlap_context and len(chunks) > 1:
                chunks = self._add_context_overlap(chunks)

        except Exception as e:
            print(f"[Chunker] Chunking failed for {file_path}: {e}")
            chunks = [text]

        # Build result with enhanced metadata
        result = []
        for idx, chunk in enumerate(chunks):
            result.append({
                "text": chunk,
                "meta": {
                    "source": str(Path(file_path).resolve()),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "has_code": analysis["has_code"],
                    "has_tables": analysis["has_tables"],
                    "strategy_used": self.strategy,
                    "adaptive_enabled": self.adaptive,
                },
            })

        print(f"[Chunker] ✅ Chunked {file_path} → {len(result)} chunks (strategy: {self.strategy})")
        return result

    def chunk_multiple_files(self, file_paths: List[str],
                           embedding_model=None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk multiple files efficiently.

        Returns:
            Dict mapping file paths to their chunks
        """
        results = {}
        for file_path in file_paths:
            try:
                results[file_path] = self.chunk_file(file_path, embedding_model)
            except Exception as e:
                print(f"[Chunker] ❌ Skipping {file_path}: {e}")
                results[file_path] = []
        return results