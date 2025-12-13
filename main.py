from lexi.controller.lexi_controller import LexiController
from langchain_ollama import OllamaEmbeddings
def get_ollama_embeddong():
    return OllamaEmbeddings(model="qwen3-embedding:0.6b")

def main():
    lexi_config = {
        'mode': 'LEXI_NANO',
        'embedding': get_ollama_embeddong(),
        'sync_dirs':['D:\\Training\\test'],
        "in_memory":True,
        'use_hash_for_changes':True,

    }
    lex = LexiController(lexi_config)
    lex.sync()
    result = lex.search(query="array in js",kwargs=1)
    print(lex.stats())
    print(result)
    lex.shutdown()



if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
