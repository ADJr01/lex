from lex.config.InstanceConfig import InstanceConfig
from langchain_ollama import OllamaEmbeddings
from lex.core.Lex import Lex

def get_ollama_embeddong():
    return OllamaEmbeddings(model="qwen3-embedding:0.6b")


if __name__ == "__main__":
    lexi_conf = (InstanceConfig('my_app')
                 .set_embbeding(get_ollama_embeddong())
                 .set_vector_store_dir("D:\\Projects\\Personal\\LLM\\Lexi\\test\\persist")
                 .set_storage_dir("D:\\Projects\\Personal\\LLM\\Lexi\\test\\storage")
                 .set_record_path("D:\\Projects\\Personal\\LLM\\Lexi\\test\\record")
                 .set_chunking_strategy(InstanceConfig.CHUNK_MECHANISM.SEMANTIC_CHUNK)
                 .set_mode({'mode':InstanceConfig.MODES.LEX_NANO,'response_mode':InstanceConfig.RESPONSE_MODES.DEEP}))
    lexi = Lex(lexi_conf).start()
    print(lexi.is_running)
    query_api = lexi.vector_store_api()
    results = query_api.invoke("what is machine learning",top_k=10)
    for res in results:
        if res['score'] > 0.9:
            print(res['page_content'])
            print("=============="*10,end='\n\n')
    lexi.close()
    print(lexi.is_running)

