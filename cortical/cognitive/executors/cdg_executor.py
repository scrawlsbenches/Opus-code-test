"""
CDG Executor: SQL-like queries against CDG storage.

Handles queries routed as CDG type:
- "FROM task WHERE status = 'pending'"
- "blockers('T-123')"
- "depends_on('T-456')"

Note: This is a stub that will be fully implemented when
CDG infrastructure is available.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .protocol import BaseExecutor, ExecutionResult

if TYPE_CHECKING:
    from cortical.cdg.storage import CDGStore
    from cortical.cdg.query.executor import QueryExecutor


class CDGExecutor(BaseExecutor):
    """
    Executes CDG queries using SQL-like syntax.

    Capabilities:
    - FROM/WHERE/ORDER BY queries
    - Built-in functions (blockers, depends_on)
    - Index-optimized lookups

    Note: Requires CDGStore and QueryExecutor to be configured.
    """

    def __init__(
        self,
        store: Optional["CDGStore"] = None,
        query_executor: Optional["QueryExecutor"] = None,
    ):
        """
        Initialize CDG executor.

        Args:
            store: CDGStore for entity storage
            query_executor: QueryExecutor for running queries
        """
        self._store = store
        self._query_executor = query_executor

    @property
    def name(self) -> str:
        return "cdg"

    def execute(self, query: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a CDG query.

        Args:
            query: Dict with 'raw' key containing the query string

        Returns:
            ExecutionResult with matching entities
        """
        raw_query = query.get("raw", "")

        # Check if CDG infrastructure is available
        if self._store is None or self._query_executor is None:
            return ExecutionResult(
                items=[],
                confidence=0.1,
                source=self.name,
                explanation=(
                    "CDG query infrastructure not configured. "
                    "Initialize with CDGStore and QueryExecutor."
                ),
                metadata={"error": "not_configured", "raw_query": raw_query}
            )

        # Parse and execute the query
        try:
            from cortical.cdg.query.parser import Parser
            from cortical.cdg.query.planner import QueryPlanner

            parser = Parser()
            ast = parser.parse(raw_query)

            planner = QueryPlanner()
            plan = planner.plan(ast)

            results = self._query_executor.execute(plan)

            return ExecutionResult(
                items=results,
                confidence=0.9,
                source=self.name,
                explanation=f"Found {len(results)} results for: {raw_query}",
                metadata={"raw_query": raw_query, "result_count": len(results)}
            )

        except ImportError:
            return ExecutionResult(
                items=[],
                confidence=0.1,
                source=self.name,
                explanation="CDG query modules not available.",
                metadata={"error": "import_error", "raw_query": raw_query}
            )
        except Exception as e:
            return ExecutionResult(
                items=[],
                confidence=0.2,
                source=self.name,
                explanation=f"Query execution failed: {str(e)}",
                metadata={"error": "execution_error", "raw_query": raw_query}
            )

    def format_result(self, result: ExecutionResult) -> str:
        """Format CDG query results."""
        if result.is_empty:
            return result.explanation or "No results found."

        lines = []
        if result.explanation:
            lines.append(result.explanation)
            lines.append("")

        for i, item in enumerate(result.items[:10], 1):
            # Try to extract meaningful representation
            if hasattr(item, "id"):
                lines.append(f"  {i}. {item.id}")
            elif isinstance(item, dict) and "id" in item:
                lines.append(f"  {i}. {item['id']}")
            else:
                lines.append(f"  {i}. {str(item)[:60]}")

        if len(result.items) > 10:
            lines.append(f"  ... and {len(result.items) - 10} more")

        return "\n".join(lines)
