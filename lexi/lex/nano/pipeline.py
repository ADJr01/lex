from lexi.lex.base import LexBase
from lexi.handlers.loader import LoaderHandler
from lexi.handlers.cleaner import BasicCleanerHandler
from lexi.handlers.chunker import RecursiveChunkerHandler
from lexi.handlers.embedder import EmbedderHandler
from lexi.handlers.vectorstore import FaissNanoHandler
from lexi.storage.faiss_store import FaissStore


class LexiNanoPipeline(LexBase):

    def __init__(self, config):
        self.embedding = config.embedding
        self.faiss_store = FaissStore(self.embedding)
        self._build_chain()

    def _build_chain(self):
        self.loader = LoaderHandler()
        self.cleaner = BasicCleanerHandler()
        self.chunker = RecursiveChunkerHandler()
        self.embedder = EmbedderHandler(self.embedding)
        self.vector = FaissNanoHandler(self.faiss_store)

        self.loader \
            .set_next(self.cleaner) \
            .set_next(self.chunker) \
            .set_next(self.embedder) \
            .set_next(self.vector)

    def ingest(self, record: dict):
        context = {"file_path": record["file_path"]}
        self.loader.handle(context)

    def delete(self, file_path: str):
        self.faiss_store.delete(
            lambda m: m.get("file_path") == file_path
        )

    def search(self, query: str, **kwargs):
        return self.faiss_store.similarity_search(query, k=kwargs.get("k", 5))
