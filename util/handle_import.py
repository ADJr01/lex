import importlib
from typing import Any, List, Optional


def handle_import(
        from_module: str,
        import_names: List[str],
        on_fail_prefix: Optional[str] = None
) -> dict[str, Any]:
    """
    Dynamically import specified names from a module with fallback support.

    Args:
        from_module: The module path to import from (e.g., 'util.Constants')
        import_names: List of names to import (e.g., ['INVALID_INSTANCE_NAME_CONSTANT'])
        on_fail_prefix: Optional prefix to prepend to module path on import failure

    Returns:
        Dictionary mapping import names to their imported objects

    Raises:
        ImportError: If both primary and fallback imports fail

    Example:
        >>> imports = handle_import(
        ...     'util.Constants',
        ...     ['INVALID_INSTANCE_NAME_CONSTANT', 'SUPPORTED_METRIC_CONSTANT'],
        ...     'lex'
        ... )
        >>> INVALID_INSTANCE_NAME_CONSTANT = imports['INVALID_INSTANCE_NAME_CONSTANT']
    """
    try:
        # Try primary import
        module = importlib.import_module(from_module)
        return {name: getattr(module, name) for name in import_names}
    except (ImportError, ModuleNotFoundError) as e:
        if on_fail_prefix:
            # Try fallback with prefix
            fallback_module = f"{on_fail_prefix}.{from_module}"
            try:
                module = importlib.import_module(fallback_module)
                return {name: getattr(module, name) for name in import_names}
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    f"Failed to import from both '{from_module}' and '{fallback_module}'"
                ) from e
        raise