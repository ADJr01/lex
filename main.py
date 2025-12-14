from lex.config.InstanceConfig import InstanceConfig
from lex.core.Lex import Lex

class DummyEmbedding:
    def embed_documents(self, texts): return [[1.0]*3 for _ in texts]
    def embed_query(self, text): return [1.0]*3

def test_lex_sync_and_query(tmp_path):
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    (storage_dir / "test.txt").write_text("Test content for Lex.")

    config = (
        InstanceConfig("test_lex")
        .set_embbeding(DummyEmbedding())
        .set_record_path(tmp_path / "record")
        .set_storage_dir(storage_dir)
        .set_vector_store_dir(tmp_path / "faiss")
    )

    lex = Lex(config)
    lex.sync()

    query_api = lex.vector_store_api()
    results = query_api.invoke("test content")
    assert len(results) > 0
    lex.close()




if __name__ == "__main__":
    dir = 'D:\\test\\'
    test_lex_sync_and_query(dir)