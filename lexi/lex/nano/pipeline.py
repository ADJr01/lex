from lexi.lex.base import LexBase
from lexi.handlers.loader import LoaderHandler
from lexi.handlers.cleaner import BasicCleanerHandler
from lexi.handlers.chunker import RecursiveChunkerHandler
from lexi.handlers.embedder import EmbedderHandler
from lexi.handlers.vectorstore import FaissNanoHandler
from lexi.storage.faiss_store import FaissStore
from lexi.storage.metadata_store import MetadataStore
import os
class LexiNanoPipeline(LexBase):

    def __init__(self, config):
        self.embedding = config.embedding
        base_dir = os.path.dirname(config.db_path)
        self.index_path = os.path.join(base_dir, "lexi_index", "nano")
        os.makedirs(self.index_path, exist_ok=True)
        self.metastore = MetadataStore(
            db_path=os.path.join(self.index_path, "metadata.db")
        )
        self.faiss_store = FaissStore(self.embedding)
        self._build_chain()

    def _build_chain(self):
        self.loader = LoaderHandler()
        self.cleaner = BasicCleanerHandler()
        self.chunker = RecursiveChunkerHandler()
        self.embedder = EmbedderHandler(self.embedding)
        self.vector = FaissNanoHandler(self.faiss_store,self.metastore)

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

    def persist(self):
        """
        Persist FAISS index and metadata.
        """
        if hasattr(self.faiss_store, "save"):
            self.faiss_store.save(self.index_path)

    def load(self):
        """
        Load FAISS index if it exists.
        """
        try:
            if hasattr(self.faiss_store, "load"):
                self.faiss_store.load(self.index_path)
        except Exception:
            # First run – nothing to load
            pass
