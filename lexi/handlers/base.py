from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LexiHandler(ABC):
    """
    Base class for all Lexi handlers.

    Implements Chain of Responsibility pattern.
    Each handler:
      - receives a mutable `context` dict
      - processes part of the pipeline
      - forwards context to the next handler
    """

    def __init__(self, name: Optional[str] = None):
        self._next: Optional["LexiHandler"] = None
        self.name = name or self.__class__.__name__

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def set_next(self, handler: "LexiHandler") -> "LexiHandler":
        """
        Set the next handler in the chain.
        Returns the handler to allow fluent chaining.
        """
        self._next = handler
        return handler

    def get_next(self) -> Optional["LexiHandler"]:
        return self._next

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def handle(self, context: Dict[str, Any]) -> None:
        """
        Execute this handler and forward to the next handler.

        This method should NOT be overridden.
        Override `_process()` instead.
        """
        if context is None:
            raise ValueError("LexiHandler received a null context")

        try:
            self._process(context)
        except Exception as e:
            raise RuntimeError(
                f"[{self.name}] failed while processing context keys={list(context.keys())}"
            ) from e

        if self._next:
            self._next.handle(context)

    # ------------------------------------------------------------------
    # Handler logic (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _process(self, context: Dict[str, Any]) -> None:
        """
        Implement handler-specific logic here.

        Rules:
        - Must mutate context in-place
        - Must NOT return anything
        - Must NOT call next handler directly
        """
        pass

    # ------------------------------------------------------------------
    # Optional lifecycle hooks (future use)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset internal state if the handler is stateful.
        Default: no-op.
        """
        pass

    def supports(self, context: Dict[str, Any]) -> bool:
        """
        Whether this handler supports the given context.
        Can be overridden for conditional chains.
        Default: always True.
        """
        return True

    # ------------------------------------------------------------------
    # Debugging helpers
    # ------------------------------------------------------------------

    def describe_chain(self) -> str:
        """
        Return a human-readable representation of the chain.
        """
        chain = [self.name]
        current = self._next

        while current:
            chain.append(current.name)
            current = current.get_next()

        return " -> ".join(chain)
