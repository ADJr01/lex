import re
from lexi.handlers.base import LexiHandler



class BasicCleanerHandler(LexiHandler):

    def _process(self, context: dict):
        cleaned_docs = []

        for doc in context["documents"]:
            text = doc.page_content
            text = text.replace("\x00", "").strip()

            doc.page_content = text
            cleaned_docs.append(doc)

        context["documents"] = cleaned_docs



class AdvancedCleanerHandler(LexiHandler):

    def _process(self, context: dict):
        cleaned = []

        for doc in context["documents"]:
            text = doc.page_content

            # remove nulls, extra spaces
            text = text.replace("\x00", " ")
            text = re.sub(r"\s+", " ", text)

            # remove page numbers / headers (basic heuristic)
            text = re.sub(r"Page \d+", "", text)

            doc.page_content = text.strip()
            cleaned.append(doc)

        context["documents"] = cleaned
