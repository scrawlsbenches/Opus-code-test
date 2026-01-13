"""
Function registry for extensible CDG query functions.

Functions are registered by name and can be looked up at runtime.
This allows new functions to be added without modifying core code.

The registry supports two types of functions:
1. Core CDG functions (count, exists, type_of)
2. Extension functions registered by other modules (e.g., GoT's blockers, infer)

See: docs/design/cdg-query-language.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.cdg.storage import CDGStore
    from cortical.cdg.index_manager import CDGIndexManager
    from cortical.cdg.schema import SchemaRegistry


# Common fields valid for all entity types (used by CLI --list-fields)
COMMON_FIELDS: Set[str] = {'id', 'title', 'status', 'created_at', 'modified_at'}


@dataclass
class FunctionSignature:
    """
    Describes a registered function's interface.

    Attributes:
        name: Function name (case-insensitive for lookup)
        description: Human-readable description
        required_args: List of required positional argument names
        optional_args: Dict of optional arguments with default values
        returns: Description of return type
        category: Category for grouping in help (e.g., 'core', 'graph', 'filter')
    """
    name: str
    description: str
    required_args: List[str] = field(default_factory=list)
    optional_args: Dict[str, Any] = field(default_factory=dict)
    returns: str = ""
    category: str = "core"


class QueryFunction(ABC):
    """
    Base class for all CDG query functions.

    Subclasses must implement:
    - signature(): Return function metadata
    - execute(): Execute the function with given context

    The execute() method receives a QueryContext with access to:
    - CDGStore for entity access
    - CDGIndexManager for indexed lookups
    - SchemaRegistry for schema validation
    - Any extension-specific context (e.g., GoTManager)
    """

    @classmethod
    @abstractmethod
    def signature(cls) -> FunctionSignature:
        """Return function signature for validation and help."""
        pass

    @abstractmethod
    def execute(
        self,
        context: "QueryContext",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute the function and return results.

        Args:
            context: Query execution context with store, index, schema access
            args: Positional arguments (already resolved from AST)
            kwargs: Keyword arguments (already resolved from AST)

        Returns:
            Function result (typically List[Entity] or scalar)
        """
        pass


@dataclass
class QueryContext:
    """
    Context passed to query functions during execution.

    This provides functions with access to CDG infrastructure
    without tightly coupling them to specific implementations.
    """
    store: Optional["CDGStore"] = None
    index_manager: Optional["CDGIndexManager"] = None
    schema_registry: Optional["SchemaRegistry"] = None

    # Extension context - modules can attach their own context here
    # e.g., GoT attaches GoTManager as extensions['got_manager']
    extensions: Dict[str, Any] = field(default_factory=dict)

    def get_extension(self, key: str) -> Any:
        """Get an extension context by key, or None if not present."""
        return self.extensions.get(key)

    def require_extension(self, key: str) -> Any:
        """Get an extension context by key, raising if not present."""
        if key not in self.extensions:
            raise RuntimeError(
                f"Required extension '{key}' not available in query context. "
                f"Available extensions: {list(self.extensions.keys())}"
            )
        return self.extensions[key]


class FunctionRegistry:
    """
    Registry for CDG query functions.

    Functions are registered by name and can be looked up at runtime.
    This allows new functions to be added without modifying core code.

    Usage:
        # Register a function class
        @FunctionRegistry.register('count')
        class CountFunction(QueryFunction):
            ...

        # Look up a function
        func_class = FunctionRegistry.get('count')

        # List all functions
        signatures = FunctionRegistry.list_functions()
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
        """
        Decorator to register a function.

        Usage:
            @FunctionRegistry.register('blockers')
            class BlockersFunction(QueryFunction):
                ...
        """
        def decorator(func_class: Type[QueryFunction]) -> Type[QueryFunction]:
            cls._functions[name.lower()] = func_class
            return func_class
        return decorator

    @classmethod
    def register_function(cls, name: str, func_class: Type[QueryFunction]) -> None:
        """
        Register a function class directly (non-decorator form).

        Usage:
            FunctionRegistry.register_function('blockers', BlockersFunction)
        """
        cls._functions[name.lower()] = func_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[QueryFunction]]:
        """Look up a function by name (case-insensitive)."""
        return cls._functions.get(name.lower())

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a function is registered."""
        return name.lower() in cls._functions

    @classmethod
    def list_functions(cls) -> List[FunctionSignature]:
        """List all registered functions."""
        return [f.signature() for f in cls._functions.values()]

    @classmethod
    def list_by_category(cls, category: str) -> List[FunctionSignature]:
        """List functions in a specific category."""
        return [
            f.signature() for f in cls._functions.values()
            if f.signature().category == category
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered functions (for testing)."""
        cls._functions.clear()

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a function by name. Returns True if it existed."""
        return cls._functions.pop(name.lower(), None) is not None
