from lexi.chrono.Chrono import Chrono
from typing import List, Optional


class ChronoAdapter:
    """
    Thin adapter around Chrono.
    Manages Chrono lifecycle and exposes a stable interface for LexiController.
    """

    def __init__(self, chrono_config: dict):
        self.chrono_config = chrono_config
        self._chrono: Optional[Chrono] = Chrono(self.chrono_config)



    def scan(self) -> List[dict]:
        self._ensure_initialized()
        return self._chrono.scan()

    def commit(self) -> None:
        self._ensure_initialized()
        self._chrono.commit()

    def query_records(self, directory: Optional[str] = None) -> List[dict]:
        self._ensure_initialized()
        return self._chrono.query_records(directory)

    def close(self) -> None:
        if self._chrono:
            self._chrono.close()
            self._chrono = None
        return False  # do not suppress exceptions

    # -------------------------
    # Internal helpers
    # -------------------------

    def _ensure_initialized(self):
        if self._chrono is None:
            raise RuntimeError(
                "ChronoAdapter not initialized. "
                "Use it within a context manager: `with ChronoAdapter(...) as chrono:`"
            )
