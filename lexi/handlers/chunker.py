from lexi.handlers.base import LexiHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunkerHandler(LexiHandler):

    def __init__(self, chunk_size=1000, overlap=200):
        super().__init__()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def _process(self, context: dict):
        chunks = self.splitter.split_documents(context["documents"])

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["file_path"] = context["file_path"]

        context["chunks"] = chunks

from lexi.handlers.base import LexiHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


class SemanticChunkerHandler(LexiHandler):
    """
    Max–Min semantic chunking
    """

    def __init__(self, embedding, max_tokens=800, min_similarity=0.75):
        super().__init__()
        self.embedding = embedding
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _process(self, context: dict):
        docs = context["documents"]
        base_chunks = self.base_splitter.split_documents(docs)

        final_chunks = []
        buffer = []
        buffer_emb = None

        for chunk in base_chunks:
            emb = self.embedding.embed_query(chunk.page_content)

            if not buffer:
                buffer = [chunk]
                buffer_emb = emb
                continue

            sim = self._cosine_sim(buffer_emb, emb)

            if sim >= self.min_similarity and len(chunk.page_content) < self.max_tokens:
                buffer.append(chunk)
                buffer_emb = (buffer_emb + emb) / 2
            else:
                final_chunks.append(self._merge(buffer))
                buffer = [chunk]
                buffer_emb = emb

        if buffer:
            final_chunks.append(self._merge(buffer))

        context["chunks"] = final_chunks

        for idx, chunk in enumerate(context["chunks"]):
            chunk.metadata.update({
                "chunk_index": idx,
                "file_path": context["file_path"],
                "directory": "/".join(context["file_path"].split("/")[:-1]),
                "content_type": context["file_path"].split(".")[-1]
            })

    def _merge(self, chunks):
        text = "\n".join(c.page_content for c in chunks)
        meta = chunks[0].metadata.copy()
        meta["semantic"] = True

        from langchain.schema import Document
        return Document(page_content=text, metadata=meta)

