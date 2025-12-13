from abc import ABC, abstractmethod


class LexBase(ABC):

    @abstractmethod
    def ingest(self, record: dict):
        pass

    @abstractmethod
    def delete(self, file_path: str):
        pass

    @abstractmethod
    def search(self, query: str, **kwargs):
        pass
