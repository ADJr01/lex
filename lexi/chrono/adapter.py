from typing import List, Optional
from Chrono import Chrono


class ChronoAdapter:
    """
    Thin adapter around Chrono.
    Manages Chrono lifecycle and exposes a stable interface for LexiController.
    """

    def __init__(self, chrono_config: dict):
        self.chrono_config = chrono_config
        self._chrono: Optional[Chrono] = None

    # -------------------------
    # Lifecycle management
    # -------------------------

    def __enter__(self):
        self._chrono = Chrono(self.chrono_config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._chrono:
            self._chrono.close()
            self._chrono = None
        return False  # do not suppress exceptions

    # -------------------------
    # Chrono operations
    # -------------------------

    def scan(self) -> List[dict]:
        self._ensure_initialized()
        return self._chrono.scan()

    def commit(self) -> None:
        self._ensure_initialized()
        self._chrono.commit()

    def query_records(self, directory: Optional[str] = None) -> List[dict]:
        self._ensure_initialized()
        return self._chrono.query_records(directory)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _ensure_initialized(self):
        if self._chrono is None:
            raise RuntimeError(
                "ChronoAdapter not initialized. "
                "Use it within a context manager: `with ChronoAdapter(...) as chrono:`"
            )
