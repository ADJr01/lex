from enum import Enum


class LexiMode(str, Enum):
    LEXI_NANO = "LEXI_NANO"
    LEXI_LDS = "LEXI_LDS"


class FileStatus:
    NEW = 0
    CHANGED = 1
    UNCHANGED = -1
    DELETED = 2
