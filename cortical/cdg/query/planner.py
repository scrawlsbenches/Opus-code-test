"""
Query planner for CDG queries.

The planner analyzes a query AST and produces an execution plan that:
1. Determines which fields can use indexes (O(1) lookup)
2. Determines which fields require post-filtering (O(n) scan)
3. Chooses the optimal execution strategy

Strategies:
- index_intersect: Multiple indexed fields, intersect results
- index_scan: Single indexed field, then post-filter
- full_scan: No indexed fields, scan all entities

See: docs/design/cdg-query-language.md
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING
from enum import Enum, auto

from .ast import (
    CDGQuery, Expression, Comparison, AndExpr, OrExpr, NotExpr,
    FunctionCall, Field, Literal, Op
)
from .errors import QueryPlanError, QueryNotImplementedError

if TYPE_CHECKING:
    from cortical.cdg.schema import SchemaRegistry


class PlanStrategy(Enum):
    """Query execution strategy."""
    INDEX_INTERSECT = auto()  # Multiple index lookups, intersect results
    INDEX_SCAN = auto()       # Single index lookup, post-filter remaining
    FULL_SCAN = auto()        # No indexes available, scan all entities
    FUNCTION_CALL = auto()    # Standalone function call (no entity scan)


@dataclass
class IndexLookup:
    """A planned index lookup operation."""
    field: str
    op: Op
    value: Any

    def __repr__(self) -> str:
        return f"IndexLookup({self.field} {self.op.name} {self.value!r})"


@dataclass
class QueryPlan:
    """
    A query execution plan.

    Attributes:
        strategy: The execution strategy to use
        entity_type: The entity type being queried (None for function calls)
        index_lookups: List of index lookup operations
        post_filter: Expression to evaluate after index lookups
        order_by: Optional (field, desc) tuple for sorting
        limit: Optional maximum results
        offset: Optional skip count
        function_call: For FUNCTION_CALL strategy, the function to execute
    """
    strategy: PlanStrategy
    entity_type: Optional[str] = None
    index_lookups: List[IndexLookup] = field(default_factory=list)
    post_filter: Optional[Expression] = None
    order_by: Optional[tuple] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    function_call: Optional[FunctionCall] = None

    def __repr__(self) -> str:
        parts = [f"QueryPlan(strategy={self.strategy.name}"]
        if self.entity_type:
            parts.append(f", entity_type={self.entity_type!r}")
        if self.index_lookups:
            parts.append(f", index_lookups={self.index_lookups}")
        if self.post_filter:
            parts.append(f", post_filter={type(self.post_filter).__name__}")
        if self.order_by:
            parts.append(f", order_by={self.order_by}")
        if self.limit:
            parts.append(f", limit={self.limit}")
        parts.append(")")
        return "".join(parts)


class QueryPlanner:
    """
    Plans query execution based on schema and indexes.

    The planner uses the schema registry to determine which fields
    are indexed and chooses the optimal execution strategy.
    """

    def __init__(self, schema_registry: Optional["SchemaRegistry"] = None):
        self.schema_registry = schema_registry

    def plan(self, query: CDGQuery) -> QueryPlan:
        """
        Create an execution plan for a query.

        Args:
            query: The parsed query AST

        Returns:
            QueryPlan with execution strategy and operations
        """
        # Handle standalone function calls
        if query.is_function_query():
            return QueryPlan(
                strategy=PlanStrategy.FUNCTION_CALL,
                function_call=query.expression,
                order_by=query.order_by,
                limit=query.limit,
                offset=query.offset
            )

        # Handle entity queries
        if query.entity_type is None:
            # Legacy query without FROM clause - treat as task query for backwards compat
            # TODO(cdg-query): Should we require FROM clause?
            entity_type = 'task'
        else:
            entity_type = query.entity_type

        # If no expression, it's a simple "select all"
        if query.expression is None:
            return QueryPlan(
                strategy=PlanStrategy.FULL_SCAN,
                entity_type=entity_type,
                order_by=query.order_by,
                limit=query.limit,
                offset=query.offset
            )

        # Analyze the expression to find indexable conditions
        index_lookups, post_filter = self._analyze_expression(
            query.expression,
            entity_type
        )

        # Choose strategy based on available indexes
        if not index_lookups:
            strategy = PlanStrategy.FULL_SCAN
            post_filter = query.expression  # All conditions as post-filter
        elif len(index_lookups) == 1:
            strategy = PlanStrategy.INDEX_SCAN
        else:
            strategy = PlanStrategy.INDEX_INTERSECT

        return QueryPlan(
            strategy=strategy,
            entity_type=entity_type,
            index_lookups=index_lookups,
            post_filter=post_filter,
            order_by=query.order_by,
            limit=query.limit,
            offset=query.offset
        )

    def _analyze_expression(
        self,
        expr: Expression,
        entity_type: str
    ) -> tuple:
        """
        Analyze an expression to extract indexable conditions.

        Returns:
            (index_lookups, post_filter) tuple
        """
        # Handle AND expressions - can extract multiple index lookups
        if isinstance(expr, AndExpr):
            return self._analyze_and_expr(expr, entity_type)

        # Handle OR expressions - currently requires full scan
        # TODO(cdg-query): OR optimization requires union of index lookups
        # See: docs/design/cdg-query-language.md#open-questions
        if isinstance(expr, OrExpr):
            return ([], expr)  # Full scan with post-filter

        # Handle NOT expressions - requires full scan
        if isinstance(expr, NotExpr):
            return ([], expr)  # Full scan with post-filter

        # Handle simple comparison
        if isinstance(expr, Comparison):
            return self._analyze_comparison(expr, entity_type)

        # Handle function call within expression
        if isinstance(expr, FunctionCall):
            return ([], expr)  # Function calls are post-filter

        # Unknown expression type
        return ([], expr)

    def _analyze_and_expr(
        self,
        expr: AndExpr,
        entity_type: str
    ) -> tuple:
        """Analyze AND expression - can use multiple indexes."""
        index_lookups = []
        post_filter_parts = []

        for child in expr.children:
            if isinstance(child, Comparison):
                lookup = self._try_index_lookup(child, entity_type)
                if lookup:
                    index_lookups.append(lookup)
                else:
                    post_filter_parts.append(child)
            else:
                # Complex sub-expressions go to post-filter
                post_filter_parts.append(child)

        # Build post-filter from remaining conditions
        post_filter = None
        if len(post_filter_parts) == 1:
            post_filter = post_filter_parts[0]
        elif len(post_filter_parts) > 1:
            post_filter = AndExpr(children=tuple(post_filter_parts))

        return (index_lookups, post_filter)

    def _analyze_comparison(
        self,
        expr: Comparison,
        entity_type: str
    ) -> tuple:
        """Analyze a single comparison."""
        lookup = self._try_index_lookup(expr, entity_type)
        if lookup:
            return ([lookup], None)
        else:
            return ([], expr)

    def _try_index_lookup(
        self,
        comp: Comparison,
        entity_type: str
    ) -> Optional[IndexLookup]:
        """
        Try to create an index lookup for a comparison.

        Returns IndexLookup if the field is indexed and operator is supported,
        otherwise None.
        """
        field_name = comp.field.name

        # Check if field is indexed
        if not self._is_field_indexed(entity_type, field_name):
            return None

        # Check if operator is index-compatible
        # Only EQ and IN can use hash indexes
        if comp.op not in (Op.EQ, Op.IN):
            return None

        # Extract literal value
        if not isinstance(comp.value, Literal):
            return None

        return IndexLookup(
            field=field_name,
            op=comp.op,
            value=comp.value.value
        )

    def _is_field_indexed(self, entity_type: str, field_name: str) -> bool:
        """Check if a field is indexed in the schema."""
        if self.schema_registry is None:
            return False

        schema = self.schema_registry.get_schema(entity_type)
        if schema is None:
            return False

        # Check if field exists and is indexed
        field_def = schema.fields.get(field_name)
        if field_def is None:
            return False

        return getattr(field_def, 'indexed', False)


def plan(query: CDGQuery, schema_registry: Optional["SchemaRegistry"] = None) -> QueryPlan:
    """Convenience function to plan a query."""
    return QueryPlanner(schema_registry).plan(query)
