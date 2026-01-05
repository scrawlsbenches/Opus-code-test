"""
Query executor that walks the AST and executes against GoT storage.

Uses the FunctionRegistry for function dispatch.
"""

from typing import Any, List, Optional, TYPE_CHECKING

from .ast import (
    Expression, Query, Literal, Field, Comparison,
    AndExpr, OrExpr, NotExpr, FunctionCall, Op
)
from .registry import FunctionRegistry
from .errors import ExecutionError

if TYPE_CHECKING:
    from cortical.got.api import GoTManager


class QueryExecutor:
    """
    Execute AST against GoT storage using registered functions.

    Usage:
        executor = QueryExecutor(manager)
        results = executor.execute(query)
    """

    def __init__(self, manager: "GoTManager"):
        self.manager = manager
        self.registry = FunctionRegistry.instance()

    def execute(self, query: Query) -> Any:
        """Execute a query and return results."""
        # Implementation will be completed in T-008
        raise NotImplementedError("Executor implementation pending (T-008)")

    def _evaluate(self, expr: Expression) -> Any:
        """Evaluate an expression node."""
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Field):
            return expr.name
        elif isinstance(expr, Comparison):
            return self._evaluate_comparison(expr)
        elif isinstance(expr, AndExpr):
            return self._evaluate_and(expr)
        elif isinstance(expr, OrExpr):
            return self._evaluate_or(expr)
        elif isinstance(expr, NotExpr):
            return self._evaluate_not(expr)
        elif isinstance(expr, FunctionCall):
            return self._apply_function(expr)
        else:
            raise ExecutionError(f"Unknown expression type: {type(expr)}")

    def _evaluate_comparison(self, comp: Comparison) -> Any:
        """Evaluate a comparison expression."""
        # Implementation pending
        raise NotImplementedError()

    def _evaluate_and(self, expr: AndExpr) -> Any:
        """Evaluate an AND expression."""
        # Implementation pending
        raise NotImplementedError()

    def _evaluate_or(self, expr: OrExpr) -> Any:
        """Evaluate an OR expression."""
        # Implementation pending
        raise NotImplementedError()

    def _evaluate_not(self, expr: NotExpr) -> Any:
        """Evaluate a NOT expression."""
        # Implementation pending
        raise NotImplementedError()

    def _apply_function(self, func: FunctionCall) -> Any:
        """Execute a function call using the registry."""
        func_class = self.registry.get(func.name)

        if func_class is None:
            available = [f.name for f in self.registry.list_functions()]
            raise ExecutionError(
                f"Unknown function '{func.name}'. "
                f"Available functions: {', '.join(available) if available else 'none'}"
            )

        # Extract arg values from AST
        arg_values = [self._evaluate(arg) for arg in func.args]
        kwarg_values = {k: self._evaluate(v) for k, v in func.kwargs.items()}

        # Instantiate and execute
        instance = func_class()
        return instance.execute(self.manager, arg_values, kwarg_values)


def execute(manager: "GoTManager", query: Query) -> Any:
    """Convenience function to execute a query."""
    return QueryExecutor(manager).execute(query)
