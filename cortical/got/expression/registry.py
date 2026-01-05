"""
Function registry for extensible query functions.

Functions are registered by name and can be looked up at runtime.
This allows new functions to be added without modifying core code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.got.api import GoTManager


@dataclass
class FunctionSignature:
    """Describes a registered function's interface."""
    name: str
    description: str
    required_args: List[str]
    optional_args: Dict[str, Any]  # name -> default value
    returns: str  # description of return type


class QueryFunction(ABC):
    """Base class for all query functions."""

    @classmethod
    @abstractmethod
    def signature(cls) -> FunctionSignature:
        """Return function signature for validation and help."""
        pass

    @abstractmethod
    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Any:
        """Execute the function and return results."""
        pass


class FunctionRegistry:
    """
    Registry for query functions.

    Functions are registered by name and can be looked up at runtime.
    This allows new functions to be added without modifying core code.
    """

    _instance: Optional["FunctionRegistry"] = None
    _functions: Dict[str, Type[QueryFunction]] = {}

    @classmethod
    def instance(cls) -> "FunctionRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(cls, name: str) -> Callable[[Type[QueryFunction]], Type[QueryFunction]]:
        """Decorator to register a function."""
        def decorator(func_class: Type[QueryFunction]) -> Type[QueryFunction]:
            cls._functions[name.lower()] = func_class
            return func_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[QueryFunction]]:
        """Look up a function by name (case-insensitive)."""
        return cls._functions.get(name.lower())

    @classmethod
    def list_functions(cls) -> List[FunctionSignature]:
        """List all registered functions."""
        return [f.signature() for f in cls._functions.values()]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered functions (for testing)."""
        cls._functions.clear()
