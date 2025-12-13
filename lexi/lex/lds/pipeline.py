import os
from lexi.lex.base import LexBase
from lexi.handlers.loader import UniversalLoaderHandler
from lexi.handlers.cleaner import AdvancedCleanerHandler
from lexi.handlers.chunker import SemanticChunkerHandler
from lexi.handlers.embedder import EmbedderHandler
from lexi.handlers.vectorstore import FaissNanoHandler
from lexi.storage.faiss_store import FaissHNSWStore
from lexi.storage.metadata_store import MetadataStore

class LexiLDSPipeline(LexBase):

    def __init__(self, config):
        self.embedding = config.embedding
        dim = len(self.embedding.embed_query("test"))
        self.index_path = os.path.join("lexi_index", "nano")
        os.makedirs(self.index_path, exist_ok=True)
        self.metastore = MetadataStore(
            db_path=os.path.join(self.index_path, "metadata.db")
        )
        self.faiss_store = FaissHNSWStore(self.embedding, dim)
        self._build_chain()

    def _build_chain(self):
        self.loader = UniversalLoaderHandler()
        self.cleaner = AdvancedCleanerHandler()
        self.chunker = SemanticChunkerHandler(self.embedding)
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
        return self.faiss_store.similarity_search(query, k=kwargs.get("k", 10))
