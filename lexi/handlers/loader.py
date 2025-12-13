from lexi.handlers.base import LexiHandler
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, JSONLoader

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    JSONLoader
)


class UniversalLoaderHandler(LexiHandler):

    def _process(self, context: dict):
        file_path = context["file_path"]
        ext = file_path.split(".")[-1].lower()

        if ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")

        elif ext == "pdf":
            loader = PyPDFLoader(file_path)

        elif ext in ("doc", "docx"):
            loader = UnstructuredWordDocumentLoader(file_path)

        elif ext in ("xlsx", "xls"):
            loader = UnstructuredExcelLoader(file_path)

        elif ext == "json":
            loader = JSONLoader(file_path, jq_schema=".", text_content=False)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        context["documents"] = loader.load()


class LoaderHandler(LexiHandler):

    def _process(self, context: dict):
        file_path = context["file_path"]
        ext = file_path.split(".")[-1].lower()

        if ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")

        elif ext == "pdf":
            loader = PyPDFLoader(file_path)

        elif ext in ("doc", "docx"):
            loader = UnstructuredWordDocumentLoader(file_path)

        elif ext in ("xlsx", "xls"):
            loader = UnstructuredExcelLoader(file_path)

        elif ext == "json":
            loader = JSONLoader(file_path, jq_schema=".", text_content=False)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        documents = loader.load()

        context["documents"] = documents
