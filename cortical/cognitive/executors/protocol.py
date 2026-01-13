"""
Query Executor Protocol and Base Classes.

Defines the interface that all query executors must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ExecutionResult:
    """
    Result from a query executor.

    Attributes:
        items: List of result items (files, documents, entities, etc.)
        confidence: Overall confidence in the results (0.0-1.0)
        source: Which executor produced this result
        explanation: Optional natural language explanation
        metadata: Additional executor-specific metadata
    """
    items: List[Any] = field(default_factory=list)
    confidence: float = 0.5
    source: str = "unknown"
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Check if result has no items."""
        return len(self.items) == 0

    def __len__(self) -> int:
        """Number of result items."""
        return len(self.items)


@runtime_checkable
class QueryExecutorProtocol(Protocol):
    """
    Protocol for query executors.

    All executors must implement:
    - execute(query) -> ExecutionResult
    - format_result(result) -> str
    """

    def execute(self, query: Any) -> ExecutionResult:
        """
        Execute a query and return results.

        Args:
            query: The parsed query (type depends on executor)

        Returns:
            ExecutionResult with items, confidence, and metadata
        """
        ...

    def format_result(self, result: ExecutionResult) -> str:
        """
        Format an execution result as natural language.

        Args:
            result: The execution result to format

        Returns:
            Human-readable string representation
        """
        ...


class BaseExecutor(ABC):
    """
    Abstract base class for query executors.

    Provides common functionality and enforces the protocol.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Executor name for identification."""
        ...

    @abstractmethod
    def execute(self, query: Any) -> ExecutionResult:
        """Execute a query and return results."""
        ...

    def format_result(self, result: ExecutionResult) -> str:
        """
        Default result formatting.

        Subclasses can override for custom formatting.
        """
        if result.is_empty:
            return f"No results found ({self.name})."

        lines = [f"Found {len(result)} results:"]
        for i, item in enumerate(result.items[:10], 1):
            lines.append(f"  {i}. {self._format_item(item)}")

        if len(result.items) > 10:
            lines.append(f"  ... and {len(result.items) - 10} more")

        if result.explanation:
            lines.append(f"\n{result.explanation}")

        return "\n".join(lines)

    def _format_item(self, item: Any) -> str:
        """Format a single result item."""
        if isinstance(item, dict):
            # Try common keys
            if "file" in item:
                return item["file"]
            if "name" in item:
                return item["name"]
            if "title" in item:
                return item["title"]
            if "doc_id" in item:
                return item["doc_id"]
            return str(item)
        if hasattr(item, "name"):
            return item.name
        return str(item)
