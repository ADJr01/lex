from typing import Dict, Any
from lexi.core.constants import LexiMode
from lexi.core.exceptions import ConfigurationError


class LexiConfig:
    def __init__(self, config: Dict[str, Any]):
        self.raw = config
        self._validate()
        self._apply_defaults()

    def _validate(self):
        if "mode" not in self.raw:
            raise ConfigurationError("`mode` is compulsory")

        if self.raw["mode"] not in LexiMode.__members__:
            raise ConfigurationError("Invalid Lexi mode")

        if "embedding" not in self.raw:
            raise ConfigurationError("`embedding` function is compulsory")

        if "sync_dirs" not in self.raw or not self.raw["sync_dirs"]:
            raise ConfigurationError("`sync_dirs` must be a non-empty list")

    def _apply_defaults(self):
        self.raw.setdefault("db_path", None)
        self.raw.setdefault("in_memory", False)
        self.raw.setdefault("use_hash_for_changes", True)

    @property
    def mode(self) -> LexiMode:
        return LexiMode[self.raw["mode"]]

    @property
    def embedding(self):
        return self.raw["embedding"]

    @property
    def sync_dirs(self):
        return self.raw["sync_dirs"]

    @property
    def db_path(self):
        return self.raw["db_path"]

    @property
    def in_memory(self):
        return self.raw["in_memory"]

    @property
    def use_hash_for_changes(self):
        return self.raw["use_hash_for_changes"]
