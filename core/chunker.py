"""
Enhanced Chunker with Improved Semantic Chunking Strategy
=========================================================

Performs intelligent semantic chunking with multiple strategies for optimal retrieval.
Integrated with DocParser for better document parsing and cleaning.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from core.Parsers.doc_paresr import DocParser
from util.Contants import EXTENSION_SUPPORTED_CONSTANT

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

from langchain_experimental.text_splitter import SemanticChunker
SEMANTIC_AVAILABLE = True



class Chunker:
    """
    Enhanced intelligent file chunker with improved semantic strategies.

    Supports multiple chunking strategies optimized for retrieval accuracy:
    - SEMANTIC_CHUNK: Content-aware semantic boundaries
    - HYBRID: Combines semantic and structural chunking
    - FIXED: Traditional fixed-size chunks with smart overlap
    """

    SUPPORTED_EXTENSIONS = EXTENSION_SUPPORTED_CONSTANT

    def __init__(
        self,
        strategy: str = "SEMANTIC_CHUNK",
        adaptive: bool = True,
        chunk_size: int = 400,  # Optimized for retrieval
        chunk_overlap: int = 120,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
    ):
        """
        Initialize chunker with enhanced capabilities.

        Args:
            strategy: 'SEMANTIC_CHUNK', 'HYBRID', or 'FIXED'
            adaptive: Enable adaptive chunking based on content structure
            chunk_size: Target chunk size (smaller = better retrieval precision)
            chunk_overlap: Overlap between chunks for context preservation
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum chunk size before forcing split
        """
        self.strategy = strategy
        self.adaptive = adaptive
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_cache = {}

        self._validate_dependencies()

        # Initialize DocParser with optimized settings for retrieval
        self.parser = DocParser(
            file_path=None,
            chunk_size=self.chunk_size,  # Use consistent chunk size
            chunk_overlap=self.chunk_overlap,
            min_content_length=self.min_chunk_size
        )

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
    # File Loading with DocParser Integration
    # =====================================================

    def _load_file(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Load file using DocParser and return both text and structured chunks.

        Returns:
            Tuple of (full_text, parsed_documents_with_metadata)
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return '', []

        # Use DocParser for optimal file parsing
        self.parser.select_file(file_path)
        docs = self.parser.process_document()

        if not docs:
            return '', []

        # Extract text while preserving document structure
        text_content = []
        doc_metadata = []

        for doc in docs:
            if hasattr(doc, 'page_content'):
                text_content.append(doc.page_content)
                doc_metadata.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
            else:
                content = str(doc)
                text_content.append(content)
                doc_metadata.append({'content': content, 'metadata': {}})

        full_text = "\n\n".join(text_content)
        return full_text, doc_metadata

    # =====================================================
    # Enhanced Content Analysis
    # =====================================================

    def _analyze_content_structure(self, text: str) -> Dict[str, Any]:
        """
        Deep analysis of text structure for optimal chunking strategy.

        Returns:
            Comprehensive structure metadata
        """
        # Basic structure detection
        analysis = {
            "has_code": bool(re.search(
                r'```|class |fn |def |function |import |require\(|const |let |var |=>',
                text
            )),
            "has_lists": bool(re.search(
                r'^\s*[-*•]\s+|\d+\.\s+',
                text,
                re.MULTILINE
            )),
            "has_headers": bool(re.search(
                r'^#{1,6}\s+|^[A-Z][^.!?]{3,50}$',
                text,
                re.MULTILINE
            )),
            "has_tables": bool(re.search(r'\|.*\||\t.*\t', text)),
            "has_json": bool(re.search(r'^\s*[{\[]', text.strip())),
            "avg_line_length": len(text) / max(text.count('\n'), 1),
            "total_length": len(text),
            "paragraph_count": len(re.findall(r'\n\s*\n', text)) + 1,
        }

        # Advanced analysis
        sentences = re.split(r'[.!?]+\s+', text)
        analysis["sentence_count"] = len(sentences)
        analysis["avg_sentence_length"] = len(text) / max(len(sentences), 1)

        # Detect document type
        analysis["document_type"] = self._infer_document_type(analysis)

        # Calculate complexity score (0-1)
        analysis["complexity"] = self._calculate_complexity(analysis)

        return analysis

    def _infer_document_type(self, analysis: Dict[str, Any]) -> str:
        """Infer document type from structural analysis."""
        if analysis["has_code"] and analysis["avg_line_length"] < 60:
            return "code"
        elif analysis["has_tables"] and analysis["paragraph_count"] < 10:
            return "tabular"
        elif analysis["has_json"]:
            return "structured_data"
        elif analysis["has_headers"] and analysis["paragraph_count"] > 5:
            return "article"
        elif analysis["paragraph_count"] > 20:
            return "long_form"
        else:
            return "general"

    def _calculate_complexity(self, analysis: Dict[str, Any]) -> float:
        """Calculate content complexity score (0-1)."""
        score = 0.0

        if analysis["has_code"]:
            score += 0.3
        if analysis["has_tables"]:
            score += 0.2
        if analysis["avg_sentence_length"] > 30:
            score += 0.2
        if analysis["paragraph_count"] > 50:
            score += 0.2
        if analysis["has_headers"]:
            score += 0.1

        return min(score, 1.0)

    # =====================================================
    # Smart Preprocessing
    # =====================================================

    def _smart_preprocess(self, text: str, analysis: Dict[str, Any]) -> str:
        """Intelligently preprocess text based on structure analysis."""
        # Preserve code blocks with enhanced markers
        if analysis["has_code"]:
            text = self._mark_code_boundaries(text)

        # Normalize excessive whitespace while preserving structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r' {3,}', '  ', text)

        # Preserve and enhance list structures
        if analysis["has_lists"]:
            text = self._enhance_list_markers(text)

        # Preserve headers
        if analysis["has_headers"]:
            text = self._enhance_header_markers(text)

        # Preserve tables
        if analysis["has_tables"]:
            text = self._mark_table_boundaries(text)

        return text

    def _mark_code_boundaries(self, text: str) -> str:
        """Add boundary markers for code blocks to prevent splitting."""
        # Markdown code blocks
        text = re.sub(
            r'(```[\s\S]*?```)',
            r'\n[CODE_BLOCK_START]\n\1\n[CODE_BLOCK_END]\n',
            text
        )

        # Indented code blocks (4+ spaces or tabs)
        text = re.sub(
            r'(\n(?:[ ]{4,}|\t).+(?:\n(?:[ ]{4,}|\t).+)*)',
            r'\n[CODE_BLOCK_START]\1\n[CODE_BLOCK_END]\n',
            text
        )

        return text

    def _enhance_list_markers(self, text: str) -> str:
        """Enhance list item detection for better chunk boundaries."""
        text = re.sub(
            r'^(\s*[-*•]\s+)',
            r'[LIST_ITEM]\1',
            text,
            flags=re.MULTILINE
        )
        return text

    def _enhance_header_markers(self, text: str) -> str:
        """Add markers for headers to preserve them in chunks."""
        # Markdown headers
        text = re.sub(
            r'^(#{1,6}\s+.+)$',
            r'[HEADER]\1',
            text,
            flags=re.MULTILINE
        )
        return text

    def _mark_table_boundaries(self, text: str) -> str:
        """Mark table boundaries to keep them together."""
        # Simple table detection (lines with pipes or tabs)
        lines = text.split('\n')
        in_table = False
        result = []

        for line in lines:
            if '|' in line or '\t' in line:
                if not in_table:
                    result.append('[TABLE_START]')
                    in_table = True
                result.append(line)
            else:
                if in_table:
                    result.append('[TABLE_END]')
                    in_table = False
                result.append(line)

        if in_table:
            result.append('[TABLE_END]')

        return '\n'.join(result)

    # =====================================================
    # Enhanced Chunking Strategies
    # =====================================================

    def _get_adaptive_chunk_size(self, analysis: Dict[str, Any]) -> int:
        """Determine optimal chunk size based on content analysis."""
        doc_type = analysis["document_type"]
        complexity = analysis["complexity"]

        # Base sizes optimized for retrieval
        size_map = {
            "code": 600,           # Larger for code context
            "tabular": 400,        # Smaller to isolate table rows
            "structured_data": 300,  # Very small for JSON objects
            "article": 500,        # Balanced for articles
            "long_form": 550,      # Slightly larger for narrative
            "general": 500         # Default balanced size
        }

        base_size = size_map.get(doc_type, 500)

        # Adjust based on complexity
        if complexity > 0.7:
            base_size = int(base_size * 1.2)  # Increase for complex content
        elif complexity < 0.3:
            base_size = int(base_size * 0.9)  # Decrease for simple content

        # Ensure within bounds
        return max(self.min_chunk_size, min(base_size, self.max_chunk_size))

    def _create_smart_splitter(
        self,
        text: str,
        analysis: Dict[str, Any],
        embedding_model=None
    ) -> Any:
        """Create an intelligent text splitter based on content analysis."""
        chunk_size = self._get_adaptive_chunk_size(analysis)

        # Calculate adaptive overlap (20-25% of chunk size)
        overlap = min(
            int(chunk_size * 0.25),
            self.chunk_overlap
        )

        # Priority-based separators for semantic boundaries
        separators = self._get_smart_separators(analysis)

        # Try semantic chunking first if available and appropriate
        if (self.strategy == "SEMANTIC_CHUNK" and
            embedding_model and
            SEMANTIC_AVAILABLE and
            analysis["document_type"] not in ["structured_data", "tabular"]):
            try:
                print("[Chunker] Using semantic chunking strategy")
                return SemanticChunker(
                    embedding_model,
                    breakpoint_threshold_type="percentile",  # More stable
                    breakpoint_threshold_amount=0.7  # Adjust sensitivity
                )
            except Exception as e:
                print(f"[Chunker] Semantic chunker failed, using hybrid: {e}")

        # Fallback to recursive character splitter
        print(f"[Chunker] Using recursive splitter (chunk_size={chunk_size}, overlap={overlap})")
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def _get_smart_separators(self, analysis: Dict[str, Any]) -> List[str]:
        """Get priority-ordered separators based on content type."""
        doc_type = analysis["document_type"]

        # Base separators (high to low priority)
        base_separators = [
            "\n[CODE_BLOCK_END]\n",  # Code blocks (highest priority)
            "\n[TABLE_END]\n",       # Tables
            "\n\n\n",                # Section breaks
            "\n[HEADER]",            # Headers
            "\n\n",                  # Paragraph breaks
            "\n[LIST_ITEM]",         # List items
            "\n",                    # Line breaks
            ". ",                    # Sentence ends
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            " ",                     # Word breaks
            "",                      # Character breaks (last resort)
        ]

        # Adjust separator priority based on document type
        if doc_type == "code":
            # Prioritize code block boundaries
            base_separators = [
                "\n[CODE_BLOCK_END]\n",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        elif doc_type == "tabular":
            # Prioritize table and row boundaries
            base_separators = [
                "\n[TABLE_END]\n",
                "\n",
                "\t",
                " ",
                ""
            ]
        elif doc_type == "article":
            # Prioritize semantic boundaries
            base_separators = [
                "\n\n\n",
                "\n[HEADER]",
                "\n\n",
                ". ",
                "\n",
                " ",
                ""
            ]

        return base_separators

    # =====================================================
    # Post-Processing
    # =====================================================

    def _post_process_chunks(
        self,
        chunks: List[str],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Post-process chunks to clean up markers and ensure quality."""
        processed = []

        for chunk in chunks:
            # Remove boundary markers
            chunk = chunk.replace('[CODE_BLOCK_START]', '')
            chunk = chunk.replace('[CODE_BLOCK_END]', '')
            chunk = chunk.replace('[TABLE_START]', '')
            chunk = chunk.replace('[TABLE_END]', '')
            chunk = chunk.replace('[LIST_ITEM]', '')
            chunk = chunk.replace('[HEADER]', '')

            # Clean up excessive whitespace
            chunk = re.sub(r'\n{3,}', '\n\n', chunk)
            chunk = chunk.strip()

            # Only keep chunks that meet minimum size
            if len(chunk) >= self.min_chunk_size:
                processed.append(chunk)

        return processed

    def _merge_small_chunks(
        self,
        chunks: List[str],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Intelligently merge chunks that are too small."""
        if not chunks:
            return chunks

        # Adjust min size based on document type
        doc_type = analysis["document_type"]
        min_size = self.min_chunk_size

        if doc_type == "structured_data":
            min_size = 50  # Allow smaller chunks for JSON/CSV

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current) < min_size:
                # Merge with appropriate separator
                separator = "\n\n" if not current.endswith('\n') else ""
                current += separator + next_chunk
            else:
                merged.append(current)
                current = next_chunk

        # Don't forget the last chunk
        if current:
            merged.append(current)

        return merged

    def _add_context_overlap(
        self,
        chunks: List[str],
        overlap_sentences: int = 2
    ) -> List[str]:
        """Add contextual overlap between chunks for better retrieval."""
        if len(chunks) <= 1:
            return chunks

        enhanced = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]

            # Extract last N sentences from previous chunk
            sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
            overlap_text = ' '.join(
                sentences[-overlap_sentences:]
            ) if len(sentences) > overlap_sentences else ''

            if overlap_text and len(overlap_text) > 20:
                # Add as context prefix
                enhanced_chunk = f"[Context: ...{overlap_text}]\n\n{current_chunk}"
                enhanced.append(enhanced_chunk)
            else:
                enhanced.append(current_chunk)

        return enhanced

    def _ensure_chunk_quality(self, chunks: List[str]) -> List[str]:
        """Final quality check and enhancement of chunks."""
        quality_chunks = []

        for chunk in chunks:
            # Skip empty or too-short chunks
            if len(chunk.strip()) < self.min_chunk_size:
                continue

            # Ensure chunks don't exceed max size
            if len(chunk) > self.max_chunk_size:
                # Split oversized chunks
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                sub_chunks = sub_splitter.split_text(chunk)
                quality_chunks.extend(sub_chunks)
            else:
                quality_chunks.append(chunk)

        return quality_chunks

    # =====================================================
    # Main Chunking Interface
    # =====================================================

    def chunk_file(
        self,
        file_path: str,
        embedding_model=None,
        add_overlap_context: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Chunk a file into intelligent semantic or hybrid chunks.

        Args:
            file_path: Path to the file
            embedding_model: Optional embedding model for semantic chunking
            add_overlap_context: Add contextual overlap between chunks

        Returns:
            List of dicts with 'text' and 'meta' keys
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load file content using DocParser
        try:
            text, doc_metadata = self._load_file(file_path)
        except Exception as e:
            print(f"[Chunker] Error loading {file_path}: {e}")
            raise

        if not text.strip():
            print(f"[Chunker] Warning: Empty or unreadable file: {file_path}")
            return []

        # Analyze content structure
        analysis = self._analyze_content_structure(text)
        print(f"[Chunker] Document type: {analysis['document_type']}, "
              f"Complexity: {analysis['complexity']:.2f}")

        # Smart preprocessing
        if self.adaptive:
            text = self._smart_preprocess(text, analysis)

        # Create appropriate splitter and chunk
        try:
            splitter = self._create_smart_splitter(text, analysis, embedding_model)
            chunks = splitter.split_text(text)

            # Post-process chunks
            chunks = self._post_process_chunks(chunks, analysis)
            chunks = self._merge_small_chunks(chunks, analysis)
            chunks = self._ensure_chunk_quality(chunks)

            if add_overlap_context and len(chunks) > 1:
                chunks = self._add_context_overlap(chunks)

        except Exception as e:
            print(f"[Chunker] Chunking failed for {file_path}: {e}")
            # Fallback: return entire text as single chunk
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
                    "document_type": analysis["document_type"],
                    "complexity": analysis["complexity"],
                    "has_code": analysis["has_code"],
                    "has_tables": analysis["has_tables"],
                    "strategy_used": self.strategy,
                    "adaptive_enabled": self.adaptive,
                    "file_extension": os.path.splitext(file_path)[1].lower(),
                },
            })

        print(f"[Chunker] ✅ Chunked {file_path} → {len(result)} chunks "
              f"(strategy: {self.strategy}, type: {analysis['document_type']})")

        return result

    def chunk_multiple_files(
        self,
        file_paths: List[str],
        embedding_model=None
    ) -> Dict[str, List[Dict[str, Any]]]:
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