from enum import Enum


class LexiMode(str, Enum):
    LEXI_NANO = "LEXI_NANO"
    LEXI_LDS = "LEXI_LDS"


class FileStatus:
    STATUS_NEW = 0
    STATUS_CHANGED = 1
    STATUS_DELETED = 2
    STATUS_UNCHANGED = -1
    STATUS_SYNCED = 3
    STATUS_ERROR = 4