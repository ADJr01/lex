import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocParser:
    """
    Universal document parser that automatically detects file type and
    processes it into chunked documents.

    Supports: .txt, .json, .pdf, .csv, .docx, .xls, .xlsx
    """

    SUPPORTED_EXTENSIONS = [".txt", ".json", ".pdf", ".csv", ".docx", ".xls", ".xlsx"]

    # Mapping of file extensions to their respective loaders
    LOADER_MAP = {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".json": JSONLoader,
        ".csv": CSVLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    def __init__(
            self,
            file_path: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            min_content_length: int = 50,
            csv_encoding: str = "utf-8",
            json_jq_schema: str = ".",
            json_text_content: bool = True,
    ):
        """
        Initialize the DocParser.

        Args:
            file_path: Path to the document file
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            min_content_length: Minimum content length to keep a chunk
            csv_encoding: Encoding for CSV files
            json_jq_schema: JQ schema for extracting content from JSON
            json_text_content: Whether JSON loader should extract text content
        """
        self.selected_file_path = None
        self.file_extension = None
        self.min_content_length = min_content_length
        self.csv_encoding = csv_encoding
        self.json_jq_schema = json_jq_schema
        self.json_text_content = json_text_content

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n","\n\n", "\n", ". ", "! ", "? ", ";", ":", " "],
            length_function=len,
        )

        if file_path is not None:
            self.select_file(file_path)

    def _validate_file(self, file_path: str) -> tuple[bool, str]:
        """
        Validate if file exists and has supported extension.

        Returns:
            Tuple of (is_valid, extension)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")

        extension = Path(file_path).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                f"Unsupported file type '{extension}' for file: {file_path}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
            return False, extension

        return True, extension

    def select_file(self, file_path: str) -> None:
        """
        Select a file for processing.

        Args:
            file_path: Path to the document file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        is_valid, extension = self._validate_file(file_path)

        if not is_valid:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        self.selected_file_path = file_path
        self.file_extension = extension
        logger.info(f"Selected file: {file_path} (type: {extension})")

    def _get_loader(self):
        """
        Get the appropriate document loader based on file extension.

        Returns:
            Configured document loader instance
        """
        loader_class = self.LOADER_MAP.get(self.file_extension)

        if loader_class is None:
            raise ValueError(f"No loader available for {self.file_extension}")

        # Configure loader based on file type
        if self.file_extension == ".json":
            return loader_class(
                file_path=self.selected_file_path,
                jq_schema=self.json_jq_schema,
                text_content=self.json_text_content
            )
        elif self.file_extension == ".csv":
            return loader_class(
                file_path=self.selected_file_path,
                encoding=self.csv_encoding
            )
        else:
            return loader_class(self.selected_file_path)

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content based on file type.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Route to appropriate cleaning method based on file type
        if self.file_extension == ".pdf":
            return self._clean_pdf_text(text)
        elif self.file_extension in [".csv", ".xls", ".xlsx"]:
            return self._clean_structured_text(text)
        elif self.file_extension == ".json":
            return self._clean_json_text(text)
        elif self.file_extension in [".txt", ".docx"]:
            return self._clean_general_text(text)
        else:
            return self._clean_general_text(text)

    def _clean_pdf_text(self, text: str) -> str:
        """
        Aggressive cleaning for PDF text (handles OCR artifacts, ligatures, etc.)
        Optimized for programming books and technical documents.
        """
        import unicodedata
        import re

        # Ligature map
        LIGATURES = {
            "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
            "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
        }
        LIGATURE_TRANS = str.maketrans(LIGATURES)

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Replace ligatures
        text = text.translate(LIGATURE_TRANS)

        # Remove font artifacts (preserve code syntax)
        text = re.sub(r"\{(?:sifd|cid\d+|[a-z]{2,4}\d*)\}|\[(?:sfi\d+|[a-z]{2,4}\d*)\]", "", text)

        # Remove private use area and control chars (preserve \t, \n, \r)
        text = re.sub(r"[\ue000-\uf8ff\x00-\x08\x0b-\x0c\x0e-\x1f\x7f■◆●◼◾◽▪▫•◊]", "", text)

        # Remove invalid chars but preserve programming symbols
        text = re.sub(r"[^\w\s.,!?:;()'\"/%\-–—\n\{\}\[\]<>=+*&|^~`#@$\\]", "", text)

        # Fix hyphenations: inter-\nnational -> international
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

        # Merge broken lines inside paragraphs
        text = re.sub(r"(?<![\.\?!:\}])\n(?=[a-zA-Z])", " ", text)

        # Remove page numbers and headers/footers
        text = re.sub(
            r"^\s*(?:Page\s*\d+|\d+|Copyright.*|All rights reserved.*|Chapter\s+\d+.*)\s*$",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # Normalize spaces and newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Clean up lines
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        return text.strip()

    def _clean_structured_text(self, text: str) -> str:
        """
        Minimal cleaning for CSV/Excel data to preserve structure.
        Only removes control characters and normalizes whitespace.
        """
        import unicodedata
        import re

        # Unicode normalization (preserve structure)
        text = unicodedata.normalize("NFKC", text)

        # Remove only control characters (preserve tabs, newlines)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize excessive spaces within cells (but preserve line structure)
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()

    def _clean_json_text(self, text: str) -> str:
        """
        Minimal cleaning for JSON to preserve valid syntax.
        """
        import unicodedata
        import re

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Remove only control characters
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

        return text.strip()

    def _clean_general_text(self, text: str) -> str:
        """
        Moderate cleaning for TXT/DOCX files.
        Removes excessive whitespace while preserving intentional formatting.
        """
        import unicodedata
        import re

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters (preserve \t, \n, \r)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize multiple spaces (but preserve single spaces and newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Collapse excessive newlines (3+ becomes 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Clean up lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)

        return text.strip()

    def _enhance_metadata(
            self,
            metadata: Dict[str, Any],
            doc_index: int,
            total_docs: int
    ) -> Dict[str, Any]:
        """
        Enhance document metadata with additional information.

        Args:
            metadata: Original metadata
            doc_index: Index of current document
            total_docs: Total number of documents

        Returns:
            Enhanced metadata dictionary
        """
        enhanced = {
            **metadata,
            "source": self.selected_file_path,
            "file_type": self.file_extension,
            "doc_index": doc_index,
            "total_docs": total_docs,
        }

        # Add file-specific metadata
        if self.file_extension == ".pdf" and "page" in metadata:
            enhanced["page_number"] = metadata.get("page", 0) + 1

        return enhanced

    def process_document(self) -> List[Document]:
        """
        Process the selected document and return chunked documents.

        Returns:
            List of Document objects with cleaned, chunked content

        Raises:
            ValueError: If no file is selected
            Exception: For document loading or processing errors
        """
        if self.selected_file_path is None:
            raise ValueError("No file selected. Use select_file() first.")

        try:
            logger.info(f"Processing document: {self.selected_file_path}")

            # Load document using appropriate loader
            loader = self._get_loader()
            raw_documents = loader.load()

            if not raw_documents:
                logger.warning(f"No content extracted from: {self.selected_file_path}")
                return []

            logger.info(f"Loaded {len(raw_documents)} document(s) from file")

            all_chunks = []

            for doc_index, doc in enumerate(raw_documents):
                # Skip documents with minimal content
                if len(doc.page_content.strip()) < self.min_content_length:
                    logger.debug(f"Skipping document {doc_index} due to insufficient content")
                    continue

                # Clean the text
                cleaned_text = self._clean_text(doc.page_content)

                if not cleaned_text or len(cleaned_text.strip()) < self.min_content_length:
                    logger.debug(f"Skipping document {doc_index} after cleaning")
                    continue

                # Enhance metadata
                enhanced_metadata = self._enhance_metadata(
                    doc.metadata,
                    doc_index,
                    len(raw_documents)
                )

                # Split document into chunks
                chunks = self.text_splitter.create_documents(
                    texts=[cleaned_text],
                    metadatas=[enhanced_metadata]
                )

                # Add chunk-specific metadata
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = chunk_idx
                    chunk.metadata["total_chunks"] = len(chunks)

                all_chunks.extend(chunks)

            logger.info(f"Created {len(all_chunks)} chunks from {len(raw_documents)} document(s)")

            if not all_chunks:
                logger.warning(f"No valid content extracted after processing: {self.selected_file_path}")
                return []

            return all_chunks

        except Exception as e:
            logger.error(f"Error processing document '{self.selected_file_path}': {str(e)}")
            raise

    def process_document_without_chunking(self) -> List[Document]:
        """
        Process document keeping original structure intact (no chunking).
        Useful when you want to preserve document boundaries.

        Returns:
            List of Document objects without chunking
        """
        if self.selected_file_path is None:
            raise ValueError("No file selected. Use select_file() first.")

        try:
            logger.info(f"Processing document without chunking: {self.selected_file_path}")

            loader = self._get_loader()
            documents = loader.load()

            processed_docs = []

            for doc_index, doc in enumerate(documents):
                if len(doc.page_content.strip()) < self.min_content_length:
                    continue

                cleaned_text = self._clean_text(doc.page_content)

                if not cleaned_text or len(cleaned_text.strip()) < self.min_content_length:
                    continue

                enhanced_metadata = self._enhance_metadata(
                    doc.metadata,
                    doc_index,
                    len(documents)
                )

                processed_doc = Document(
                    page_content=cleaned_text,
                    metadata=enhanced_metadata
                )

                processed_docs.append(processed_doc)

            logger.info(f"Processed {len(processed_docs)} document(s) without chunking")
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    @classmethod
    def process_file(
            cls,
            file_path: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            **kwargs
    ) -> List[Document]:
        """
        Convenience method to process a file in one call.

        Args:
            file_path: Path to the document file
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            **kwargs: Additional arguments for DocParser initialization

        Returns:
            List of processed Document objects
        """
        parser = cls(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        return parser.process_document()


