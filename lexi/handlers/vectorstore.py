from lexi.handlers.base import LexiHandler
from langchain_community.vectorstores import FAISS
from lexi.storage.metadata_store import MetadataStore

class FaissNanoHandler(LexiHandler):

    def __init__(self, faiss_store):
        super().__init__()
        self.store = faiss_store

    def _process(self, context: dict):
        self.store.add_documents(context["chunks"])


from lexi.utils.hashing import chunk_id


class FaissNanoHandler(LexiHandler):

    def __init__(self, faiss_store, metastore):
        super().__init__()
        self.store = faiss_store
        self.metastore = metastore

    def _process(self, context: dict):
        for chunk in context["chunks"]:
            cid = chunk_id(
                chunk.metadata["file_path"],
                chunk.metadata["chunk_index"],
                chunk.page_content
            )

            chunk.metadata["chunk_id"] = cid
            REQUIRED_FIELDS = ("file_path", "directory", "content_type", "chunk_index")
            for field in REQUIRED_FIELDS:
                if field not in chunk.metadata:
                    raise ValueError(
                        f"Missing required metadata field '{field}' "
                        f"for chunk from {chunk.metadata.get('file_path')}"
                    )
            self.metastore.add_chunk(cid, chunk.metadata)

        self.store.add_documents(context["chunks"])
