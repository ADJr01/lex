from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from core.Parsers.helper.parsing_helper import (
    clean_pdf_text,
    is_pdf_file
)


class PDFParser:
    def __init__(
            self,
            pdf_path: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,  # Increased for better context preservation
            min_page_length: int = 50,  # More reasonable minimum
            pdf_parser=PyMuPDFLoader
    ):
        self.selected_pdf_path = None
        self.parser = pdf_parser
        self.min_page_length = min_page_length

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        if pdf_path is not None and is_pdf_file(pdf_path):
            self.selected_pdf_path = pdf_path

    def select_pdf(self, pdf_path: str) -> None:
        if not is_pdf_file(pdf_path):
            raise ValueError(f"Invalid PDF file: {pdf_path}")
        self.selected_pdf_path = pdf_path

    def process_pdf(self) -> List[Document]:
        """
        Process PDF and return chunked documents with metadata.

        Returns:
            List of Document objects with cleaned, chunked content

        Raises:
            ValueError: If no PDF is selected or parser is not configured
            Exception: For PDF loading or processing errors
        """
        if self.selected_pdf_path is None:
            raise ValueError("No PDF file selected. Use select_pdf() first.")

        if self.parser is None:
            raise ValueError("No PDF parser configured.")

        try:
            # Load PDF pages
            pages = self.parser(self.selected_pdf_path).load()

            if not pages:
                raise ValueError(f"No content extracted from PDF: {self.selected_pdf_path}")

            all_chunks = []

            for page_index, page in enumerate(pages):
                # Skip pages with minimal content
                if len(page.page_content.strip()) < self.min_page_length:
                    continue

                # Clean the text
                cleaned_text = clean_pdf_text(page.page_content)

                # Skip if cleaning resulted in empty content
                if not cleaned_text or len(cleaned_text.strip()) < self.min_page_length:
                    continue

                # Split page into chunks
                page_chunks = self.text_splitter.create_documents(
                    texts=[cleaned_text],
                    metadatas=[{
                        **page.metadata,
                        "source": self.selected_pdf_path,
                        "page_number": page_index + 1,
                        "total_pages": len(pages)
                    }]
                )

                all_chunks.extend(page_chunks)

            if not all_chunks:
                raise ValueError(f"No valid content extracted from PDF after processing: {self.selected_pdf_path}")

            return all_chunks

        except Exception as e:
            # Log the error with more context
            print(f"Error processing PDF '{self.selected_pdf_path}': {str(e)}")
            raise  # Re-raise to allow caller to handle

    def process_pdf_without_chunking(self) -> List[Document]:
        """
        Process PDF keeping full pages intact (no chunking).
        Useful when you want to preserve page boundaries.
        """
        if self.selected_pdf_path is None:
            raise ValueError("No PDF file selected.")

        try:
            pages = self.parser(self.selected_pdf_path).load()
            documents = []

            for page_index, page in enumerate(pages):
                if len(page.page_content.strip()) < self.min_page_length:
                    continue

                cleaned_text = clean_pdf_text(page.page_content)

                if not cleaned_text or len(cleaned_text.strip()) < self.min_page_length:
                    continue

                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        **page.metadata,
                        "source": self.selected_pdf_path,
                        "page_number": page_index + 1,
                        "total_pages": len(pages)
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise