class LexiError(Exception):
    """Base Lexi exception"""


class ConfigurationError(LexiError):
    """Invalid configuration"""


class PipelineError(LexiError):
    """Pipeline execution error"""
