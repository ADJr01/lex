from lexi.handlers.base import LexiHandler


class EmbedderHandler(LexiHandler):

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def _process(self, context: dict):
        context["embeddings"] = self.embedding
