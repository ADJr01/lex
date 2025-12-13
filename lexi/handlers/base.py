from abc import ABC, abstractmethod


class LexiHandler(ABC):
    def __init__(self):
        self._next = None

    def set_next(self, handler: "LexiHandler") -> "LexiHandler":
        self._next = handler
        return handler

    def handle(self, context: dict):
        self._process(context)
        if self._next:
            self._next.handle(context)

    @abstractmethod
    def _process(self, context: dict):
        pass
