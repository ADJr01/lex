class LexiRetriever:

    def __init__(self, vectorstore, metastore):
        self.vectorstore = vectorstore
        self.metastore = metastore

    def search(self, query: str, directory: str = None, k: int = 5):
        results = self.vectorstore.similarity_search(query, k * 2)

        if not directory:
            return results[:k]

        allowed_ids = set(self.metastore.query_by_directory(directory))

        filtered = [
            r for r in results
            if r.metadata.get("chunk_id") in allowed_ids
        ]

        return filtered[:k]
