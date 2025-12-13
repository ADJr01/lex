import os
import numpy as np
from typing import Dict, Any, List
from lexi.handlers.base import LexiHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _inject_canonical_metadata(
    chunk: Document,
    *,
    file_path: str,
    chunk_index: int
) -> None:
    """
    Inject mandatory metadata fields required by Lexi.
    """
    chunk.metadata.update({
        "file_path": file_path,
        "directory": os.path.dirname(file_path),
        "content_type": os.path.splitext(file_path)[1].lstrip("."),
        "chunk_index": chunk_index,
    })




class RecursiveChunkerHandler(LexiHandler):
    """
    Standard recursive text chunking for LEXI_NANO.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def _process(self, context: Dict[str, Any]) -> None:
        file_path = context["file_path"]
        documents: List[Document] = context["documents"]

        chunks = self.splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            _inject_canonical_metadata(
                chunk,
                file_path=file_path,
                chunk_index=idx
            )

        context["chunks"] = chunks



class SemanticChunkerHandler(LexiHandler):
    """
    Max–Min semantic chunking for LEXI_LDS.
    """

    def __init__(
        self,
        embedding,
        max_tokens: int = 800,
        min_similarity: float = 0.75
    ):
        super().__init__()
        self.embedding = embedding
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity

        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

    # --------------------------------------------------
    # Math utilities
    # --------------------------------------------------

    @staticmethod
    def _to_vector(embedding) -> np.ndarray:
        """
        Normalize embedding output to numpy array.
        """
        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # --------------------------------------------------
    # Core processing
    # --------------------------------------------------

    def _process(self, context: Dict[str, Any]) -> None:
        file_path = context["file_path"]
        documents: List[Document] = context["documents"]

        base_chunks = self.base_splitter.split_documents(documents)

        final_chunks: List[Document] = []
        buffer: List[Document] = []
        buffer_embedding: np.ndarray | None = None

        for chunk in base_chunks:
            emb = self._to_vector(
                self.embedding.embed_query(chunk.page_content)
            )

            if not buffer:
                buffer = [chunk]
                buffer_embedding = emb
                continue

            similarity = self._cosine_similarity(buffer_embedding, emb)

            if (
                similarity >= self.min_similarity
                and len(chunk.page_content) <= self.max_tokens
            ):
                buffer.append(chunk)
                buffer_embedding = (buffer_embedding + emb) / 2
            else:
                final_chunks.append(self._merge(buffer, file_path))
                buffer = [chunk]
                buffer_embedding = emb

        if buffer:
            final_chunks.append(self._merge(buffer, file_path))

        context["chunks"] = final_chunks

    # --------------------------------------------------
    # Merge helper
    # --------------------------------------------------

    def _merge(self, chunks: List[Document], file_path: str) -> Document:
        text = "\n".join(c.page_content for c in chunks)
        metadata = dict(chunks[0].metadata)

        merged = Document(page_content=text, metadata=metadata)
        merged.metadata["semantic"] = True

        _inject_canonical_metadata(
            merged,
            file_path=file_path,
            chunk_index=merged.metadata.get("chunk_index", 0)
        )

        return merged


