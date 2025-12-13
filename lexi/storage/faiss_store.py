import faiss
from langchain.vectorstores import FAISS

class PersistentFaissMixin:

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.store.index, f"{path}/index.faiss")

    def load(self, path: str):
        self.store.index = faiss.read_index(f"{path}/index.faiss")

class FaissStore:
    def __init__(self, embedding):
        index = faiss.IndexFlatL2(embedding.embed_query("test").__len__())
        self.store = FAISS(embedding, index, {}, [])

    def add_documents(self, docs):
        self.store.add_documents(docs)

    def delete(self, filter_func):
        self.store.delete(filter_func)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k)


class FaissHNSWStore:
    def __init__(self, embedding, dim):
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 200

        self.store = FAISS(embedding, index, {}, [])

    def add_documents(self, docs):
        self.store.add_documents(docs)

    def delete(self, filter_func):
        self.store.delete(filter_func)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k)
