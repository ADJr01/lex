import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

class PersistentFaissMixin:

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.store.index, f"{path}/index.faiss")

    def load(self, path: str):
        self.store.index = faiss.read_index(f"{path}/index.faiss")

class FaissStore:
    def __init__(self, embedding):
        dim = len(embedding.embed_query("test"))

        index = faiss.IndexFlatL2(dim)

        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        self.store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    def add_documents(self, docs):
        self.store.add_documents(docs)

    def delete(self, filter_func):
        self.store.delete(filter_func)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k)

    def save(self, path: str):
        self.store.save_local(path)

    def load(self, path: str):
        self.store = FAISS.load_local(
            path,
            self.store.embedding_function,
            allow_dangerous_deserialization=True,
        )


class FaissHNSWStore:
    def __init__(self, embedding, dim):
        docstore = InMemoryDocstore({})
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 200
        index_to_docstore_id = {}
        self.store = FAISS(embedding, index, docstore, index_to_docstore_id)

    def add_documents(self, docs):
        self.store.add_documents(docs)

    def delete(self, filter_func):
        self.store.delete(filter_func)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k)

    def save(self, path: str):
        self.store.save_local(path)

    def load(self, path: str):
        self.store = FAISS.load_local(
            path,
            self.store.embedding_function,
            allow_dangerous_deserialization=True,
        )


