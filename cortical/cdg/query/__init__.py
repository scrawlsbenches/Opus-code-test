"""
CDG Query Language - Schema-driven queries for any entity type.

This module provides a unified query language for the Cortical Data Graph (CDG).
It supports SQL-like queries with automatic index optimization based on schema
field annotations.

Usage:
    from cortical.cdg.query import CDGQueryEngine, parse

    # Parse a query
    query = parse("FROM task WHERE status = 'pending' AND priority = 'high'")

    # Execute with engine
    engine = CDGQueryEngine(store, index_manager, schema_registry)
    results = engine.query("FROM task WHERE status = 'pending'")

    # Or execute step by step
    query = parse("FROM decision WHERE status = 'draft'")
    plan = engine.plan(query)
    results = engine.execute(plan)

Query Syntax:
    FROM <entity_type> [WHERE <conditions>] [ORDER BY <field> [ASC|DESC]] [LIMIT n] [OFFSET n]

    Examples:
        FROM task WHERE status = 'pending'
        FROM decision WHERE status = 'draft' ORDER BY created_at DESC LIMIT 10
        FROM handoff WHERE status = 'initiated'

    Function calls:
        blockers('T-123')
        infer(commits=10)
        count()

See: docs/design/cdg-query-language.md
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .ast import (
    CDGQuery, Query,  # Query is alias for backwards compat
    Expression, Comparison, AndExpr, OrExpr, NotExpr, FunctionCall,
    Field, Literal, Op
)
from .lexer import Lexer, Token, TokenType, tokenize
from .parser import Parser, parse
from .planner import QueryPlanner, QueryPlan, PlanStrategy, IndexLookup, plan
from .executor import QueryExecutor, execute
from .registry import FunctionRegistry, FunctionSignature, QueryFunction, QueryContext
from .errors import (
    CDGQueryError,
    QueryLexerError,
    QueryParseError,
    QueryValidationError,
    QueryPlanError,
    QueryExecutionError,
    QueryNotImplementedError
)

if TYPE_CHECKING:
    from cortical.cdg.storage import CDGStore
    from cortical.cdg.index_manager import CDGIndexManager
    from cortical.cdg.schema import SchemaRegistry


class CDGQueryEngine:
    """
    High-level query engine for CDG.

    Combines parsing, planning, and execution into a simple interface.

    Usage:
        engine = CDGQueryEngine(store, index_manager, schema_registry)
        results = engine.query("FROM task WHERE status = 'pending'")
    """

    def __init__(
        self,
        store: Optional["CDGStore"] = None,
        index_manager: Optional["CDGIndexManager"] = None,
        schema_registry: Optional["SchemaRegistry"] = None,
        extensions: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the query engine.

        Args:
            store: CDGStore for entity access
            index_manager: CDGIndexManager for indexed lookups
            schema_registry: SchemaRegistry for schema validation
            extensions: Optional dict of extension contexts (e.g., {'got_manager': manager})
        """
        self.store = store
        self.index_manager = index_manager
        self.schema_registry = schema_registry
        self.extensions = extensions or {}

        self._planner = QueryPlanner(schema_registry)
        self._context = QueryContext(
            store=store,
            index_manager=index_manager,
            schema_registry=schema_registry,
            extensions=self.extensions
        )
        self._executor = QueryExecutor(
            store=store,
            index_manager=index_manager,
            schema_registry=schema_registry,
            context=self._context
        )

    def query(self, query_string: str) -> List[Any]:
        """
        Parse and execute a query string.

        Args:
            query_string: The query to execute

        Returns:
            List of entities matching the query
        """
        parsed = parse(query_string)
        planned = self._planner.plan(parsed)
        return self._executor.execute(planned)

    def parse(self, query_string: str) -> CDGQuery:
        """Parse a query string into an AST."""
        return parse(query_string)

    def plan(self, query: CDGQuery) -> QueryPlan:
        """Create an execution plan for a query."""
        return self._planner.plan(query)

    def execute(self, plan: QueryPlan) -> List[Any]:
        """Execute a query plan."""
        return self._executor.execute(plan)

    def register_extension(self, key: str, value: Any) -> None:
        """Register an extension context (e.g., GoTManager)."""
        self.extensions[key] = value
        self._context.extensions[key] = value


# Convenience function for quick queries
def query(
    query_string: str,
    store: Optional["CDGStore"] = None,
    index_manager: Optional["CDGIndexManager"] = None,
    schema_registry: Optional["SchemaRegistry"] = None
) -> List[Any]:
    """
    Parse and execute a query in one step.

    This is a convenience function for simple use cases.
    For repeated queries, create a CDGQueryEngine instance.
    """
    engine = CDGQueryEngine(store, index_manager, schema_registry)
    return engine.query(query_string)


__all__ = [
    # Main engine
    'CDGQueryEngine',
    'query',

    # AST
    'CDGQuery',
    'Query',  # Alias for backwards compat
    'Expression',
    'Comparison',
    'AndExpr',
    'OrExpr',
    'NotExpr',
    'FunctionCall',
    'Field',
    'Literal',
    'Op',

    # Lexer
    'Lexer',
    'Token',
    'TokenType',
    'tokenize',

    # Parser
    'Parser',
    'parse',

    # Planner
    'QueryPlanner',
    'QueryPlan',
    'PlanStrategy',
    'IndexLookup',
    'plan',

    # Executor
    'QueryExecutor',
    'execute',

    # Registry
    'FunctionRegistry',
    'FunctionSignature',
    'QueryFunction',
    'QueryContext',

    # Errors
    'CDGQueryError',
    'QueryLexerError',
    'QueryParseError',
    'QueryValidationError',
    'QueryPlanError',
    'QueryExecutionError',
    'QueryNotImplementedError',
]
