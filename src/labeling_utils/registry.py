"""
Backend registry for CTC model backends.

This module provides a plugin-style registration system that allows:
1. Auto-discovery of backends from the backends/ directory
2. Manual registration of custom backends
3. Lazy loading to avoid importing unavailable dependencies
"""

from typing import Dict, List, Type, Optional, Callable
import logging
import importlib

from .base import CTCModelBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Registry for CTC model backends.

    Supports both eager and lazy registration:
    - Eager: register(name, BackendClass)
    - Lazy: register_lazy(name, module_path, class_name)

    Lazy registration avoids importing backends until they're needed,
    which prevents ImportError for backends with unavailable dependencies.
    """

    def __init__(self):
        self._backends: Dict[str, Type[CTCModelBackend]] = {}
        self._lazy_backends: Dict[str, tuple] = {}  # name -> (module_path, class_name)
        self._aliases: Dict[str, str] = {}  # alias -> canonical name

    def register(
        self,
        name: str,
        backend_class: Type[CTCModelBackend],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a backend class.

        Args:
            name: Canonical name for the backend (e.g., "huggingface")
            backend_class: The backend class to register
            aliases: Optional list of alternative names (e.g., ["hf", "transformers"])
        """
        name = name.lower()
        self._backends[name] = backend_class
        logger.debug(f"Registered backend: {name}")

        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name
                logger.debug(f"Registered alias: {alias} -> {name}")

    def register_lazy(
        self,
        name: str,
        module_path: str,
        class_name: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a backend for lazy loading.

        The backend class won't be imported until get() is called.
        This is useful for backends with optional dependencies.

        Args:
            name: Canonical name for the backend
            module_path: Full module path (e.g., "labeling_utils.backends.huggingface")
            class_name: Class name within the module
            aliases: Optional list of alternative names
        """
        name = name.lower()
        self._lazy_backends[name] = (module_path, class_name)
        logger.debug(f"Registered lazy backend: {name} -> {module_path}.{class_name}")

        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name

    def get(self, name: str) -> Type[CTCModelBackend]:
        """
        Get a backend class by name.

        Args:
            name: Backend name or alias

        Returns:
            The backend class

        Raises:
            ValueError: If backend not found
            ImportError: If backend dependencies unavailable
        """
        name = name.lower()

        # Resolve alias
        if name in self._aliases:
            name = self._aliases[name]

        # Check eager registrations first
        if name in self._backends:
            return self._backends[name]

        # Try lazy loading
        if name in self._lazy_backends:
            module_path, class_name = self._lazy_backends[name]
            try:
                module = importlib.import_module(module_path)
                backend_class = getattr(module, class_name)
                # Cache for future use
                self._backends[name] = backend_class
                return backend_class
            except ImportError as e:
                raise ImportError(
                    f"Backend '{name}' requires additional dependencies: {e}"
                ) from e

        # Not found
        available = self.list_backends()
        raise ValueError(
            f"Unknown backend: {name}. Available backends: {available}"
        )

    def list_backends(self) -> List[str]:
        """List all registered backend names (excluding aliases)."""
        all_names = set(self._backends.keys()) | set(self._lazy_backends.keys())
        return sorted(all_names)

    def list_aliases(self) -> Dict[str, str]:
        """List all aliases and their canonical names."""
        return dict(self._aliases)

    def is_available(self, name: str) -> bool:
        """
        Check if a backend is available (dependencies installed).

        Args:
            name: Backend name or alias

        Returns:
            True if backend can be loaded, False otherwise
        """
        try:
            self.get(name)
            return True
        except (ValueError, ImportError):
            return False

    def __contains__(self, name: str) -> bool:
        name = name.lower()
        if name in self._aliases:
            name = self._aliases[name]
        return name in self._backends or name in self._lazy_backends

    def __iter__(self):
        return iter(self.list_backends())


# Global registry instance
_registry = BackendRegistry()


def register_backend(
    name: str,
    backend_class: Type[CTCModelBackend],
    aliases: Optional[List[str]] = None,
) -> None:
    """
    Register a backend class globally.

    Args:
        name: Backend name (e.g., "huggingface")
        backend_class: The backend class
        aliases: Optional alternative names
    """
    _registry.register(name, backend_class, aliases)


def register_backend_lazy(
    name: str,
    module_path: str,
    class_name: str,
    aliases: Optional[List[str]] = None,
) -> None:
    """
    Register a backend for lazy loading.

    Args:
        name: Backend name
        module_path: Module containing the backend class
        class_name: Name of the backend class
        aliases: Optional alternative names
    """
    _registry.register_lazy(name, module_path, class_name, aliases)


def get_backend(name: str = "huggingface") -> Type[CTCModelBackend]:
    """
    Get a backend class by name.

    Args:
        name: Backend name or alias (default: "huggingface")

    Returns:
        The backend class

    Raises:
        ValueError: If backend not found
        ImportError: If dependencies unavailable
    """
    return _registry.get(name)


def list_backends() -> List[str]:
    """List all available backend names."""
    return _registry.list_backends()


def is_backend_available(name: str) -> bool:
    """Check if a backend is available."""
    return _registry.is_available(name)


# Decorator for easy backend registration
def backend(
    name: str,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[CTCModelBackend]], Type[CTCModelBackend]]:
    """
    Decorator to register a backend class.

    Example:
        @backend("mybackend", aliases=["mb"])
        class MyBackend(CTCModelBackend):
            ...
    """
    def decorator(cls: Type[CTCModelBackend]) -> Type[CTCModelBackend]:
        register_backend(name, cls, aliases)
        return cls
    return decorator
