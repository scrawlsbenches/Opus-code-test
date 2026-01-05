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

# Import functions to ensure they are registered with the FunctionRegistry
from .functions import graph, filters  # noqa: F401

if TYPE_CHECKING:
    from cortical.got.api import GoTManager
    from cortical.got.query_builder import Query as QueryBuilder


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
        """
        Execute a query and return results.

        Translates the AST Query into Query builder calls and executes.
        Handles NOT and complex OR expressions through post-filtering when necessary.
        """
        from cortical.got.query_builder import Query as QueryBuilder

        # Handle top-level FunctionCall expressions directly
        if query.expression and isinstance(query.expression, FunctionCall):
            return self._apply_function(query.expression)

        # Check if expression requires post-filtering
        needs_post_filter = query.expression and (
            self._contains_not(query.expression) or
            self._has_complex_or(query.expression) or
            self._contains_function(query.expression)
        )

        if needs_post_filter:
            # For NOT, complex OR, or function expressions, we need post-processing
            return self._execute_with_post_filter(query)

        # Standard path: can use Query builder directly
        entity_type = query.entity_type or 'task'
        q = self._build_base_query(entity_type)

        # Apply expression filters if present
        if query.expression:
            q = self._apply_expression(q, query.expression)

        # Apply ORDER BY if present
        if query.order_by:
            field, desc = query.order_by
            q = q.order_by(field, desc=desc)

        # Apply LIMIT if present
        if query.limit:
            q = q.limit(query.limit)

        # Apply OFFSET if present
        if query.offset:
            q = q.offset(query.offset)

        # Execute and return results
        return q.execute()

    def _contains_not(self, expr: Expression) -> bool:
        """Check if expression tree contains NOT."""
        if isinstance(expr, NotExpr):
            return True
        elif isinstance(expr, AndExpr) or isinstance(expr, OrExpr):
            return any(self._contains_not(child) for child in expr.children)
        return False

    def _contains_function(self, expr: Expression) -> bool:
        """Check if expression tree contains FunctionCall nodes."""
        if isinstance(expr, FunctionCall):
            return True
        elif isinstance(expr, (AndExpr, OrExpr)):
            return any(self._contains_function(child) for child in expr.children)
        elif isinstance(expr, NotExpr):
            return self._contains_function(expr.child)
        return False

    def _has_complex_or(self, expr: Expression) -> bool:
        """Check if expression contains complex OR (with nested AND/OR)."""
        if isinstance(expr, OrExpr):
            # OR is complex if any child is not a simple Comparison
            return any(not isinstance(child, Comparison) for child in expr.children)
        elif isinstance(expr, AndExpr):
            # Recurse into AND children
            return any(self._has_complex_or(child) for child in expr.children)
        return False

    def _execute_with_post_filter(self, query: Query) -> Any:
        """
        Execute query with expressions requiring post-filtering.

        This path is used when:
        - Expression contains NOT (Query builder doesn't support negation)
        - Expression contains complex OR (nested AND/OR)
        - Expression contains function calls mixed with filters

        For function expressions, we evaluate them to get result sets and
        combine them using set operations.
        """
        from cortical.got.query_builder import Query as QueryBuilder

        entity_type = query.entity_type or 'task'

        # If expression contains functions, use special evaluation
        if self._contains_function(query.expression):
            filtered = self._evaluate_expression_with_functions(query.expression, entity_type)
        else:
            # Original path: Get all entities and filter with predicates
            q = self._build_base_query(entity_type)
            all_entities = q.execute()
            filtered = [
                entity for entity in all_entities
                if self._matches_expression(entity, query.expression)
            ]

        # Apply ORDER BY manually if present
        if query.order_by:
            field, desc = query.order_by
            filtered = self._sort_results(filtered, field, desc)

        # Apply OFFSET and LIMIT manually
        if query.offset:
            filtered = filtered[query.offset:]
        if query.limit:
            filtered = filtered[:query.limit]

        return filtered

    def _evaluate_expression_with_functions(self, expr: Expression, entity_type: str) -> List[Any]:
        """
        Evaluate expression containing function calls.

        Returns a list of entities matching the expression.
        Uses set operations to combine results from functions and filters.

        For example:
        - recent(7) AND priority = 'high': Execute function, filter results
        - blocked() OR status = 'pending': Union of function results and filter results
        """
        if isinstance(expr, FunctionCall):
            # Execute function directly
            return self._apply_function(expr)

        elif isinstance(expr, Comparison):
            # Get all entities and filter by comparison
            q = self._build_base_query(entity_type)
            all_entities = q.execute()
            return [e for e in all_entities if self._matches_comparison(e, expr)]

        elif isinstance(expr, AndExpr):
            # Strategy: Execute functions first, then apply filters to results
            func_results = None
            filters = []

            for child in expr.children:
                if self._contains_function(child):
                    # Evaluate this function subtree
                    child_results = self._evaluate_expression_with_functions(child, entity_type)
                    if func_results is None:
                        func_results = child_results
                    else:
                        # Multiple functions: intersection
                        child_ids = {getattr(e, 'id', id(e)) for e in child_results}
                        func_results = [e for e in func_results if getattr(e, 'id', id(e)) in child_ids]
                else:
                    # This is a filter expression
                    filters.append(child)

            # Start with function results or all entities
            if func_results is not None:
                entities = func_results
            else:
                # No functions (shouldn't happen, but handle it)
                q = self._build_base_query(entity_type)
                entities = q.execute()

            # Apply filters
            if filters:
                entities = [e for e in entities
                           if all(self._matches_expression(e, f) for f in filters)]

            return entities

        elif isinstance(expr, OrExpr):
            # Union of all child results
            all_results = []
            seen_ids = set()

            for child in expr.children:
                child_results = self._evaluate_expression_with_functions(child, entity_type)
                for e in child_results:
                    entity_id = getattr(e, 'id', id(e))
                    if entity_id not in seen_ids:
                        seen_ids.add(entity_id)
                        all_results.append(e)

            return all_results

        elif isinstance(expr, NotExpr):
            # Complement: all entities minus child results
            q = self._build_base_query(entity_type)
            all_entities = q.execute()
            child_results = self._evaluate_expression_with_functions(expr.child, entity_type)
            child_ids = {getattr(e, 'id', id(e)) for e in child_results}
            return [e for e in all_entities if getattr(e, 'id', id(e)) not in child_ids]

        else:
            raise ExecutionError(f"Unsupported expression type: {type(expr).__name__}")

    def _matches_expression(self, entity: Any, expr: Expression) -> bool:
        """Check if entity matches an expression (handles NOT)."""
        if isinstance(expr, Comparison):
            return self._matches_comparison(entity, expr)
        elif isinstance(expr, AndExpr):
            return all(self._matches_expression(entity, child) for child in expr.children)
        elif isinstance(expr, OrExpr):
            return any(self._matches_expression(entity, child) for child in expr.children)
        elif isinstance(expr, NotExpr):
            return not self._matches_expression(entity, expr.child)
        else:
            raise ExecutionError(f"Cannot match expression type: {type(expr).__name__}")

    def _matches_comparison(self, entity: Any, comp: Comparison) -> bool:
        """Check if entity matches a comparison."""
        field_name = self._evaluate(comp.field)
        expected_value = self._evaluate(comp.value)

        # Get actual value from entity
        actual_value = getattr(entity, field_name, None)
        if actual_value is None and hasattr(entity, 'properties'):
            actual_value = entity.properties.get(field_name)

        # Apply operator
        if comp.op == Op.EQ:
            return actual_value == expected_value
        elif comp.op == Op.NE:
            return actual_value != expected_value
        elif comp.op == Op.GT:
            return actual_value is not None and actual_value > expected_value
        elif comp.op == Op.LT:
            return actual_value is not None and actual_value < expected_value
        elif comp.op == Op.GTE:
            return actual_value is not None and actual_value >= expected_value
        elif comp.op == Op.LTE:
            return actual_value is not None and actual_value <= expected_value
        elif comp.op == Op.IN:
            return actual_value in expected_value
        elif comp.op == Op.NOT_IN:
            return actual_value not in expected_value
        elif comp.op == Op.LIKE:
            return expected_value in (actual_value or "")
        elif comp.op == Op.NOT_LIKE:
            return expected_value not in (actual_value or "")
        else:
            raise ExecutionError(f"Unknown operator: {comp.op}")

    def _sort_results(self, results: List[Any], field: str, desc: bool) -> List[Any]:
        """Sort results by field."""
        def get_sort_key(entity):
            value = getattr(entity, field, None)
            if value is None and hasattr(entity, 'properties'):
                value = entity.properties.get(field)
            # Handle priority specially
            if field == "priority" and isinstance(value, str):
                priority_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
                value = priority_order.get(value, -1)
            return value if value is not None else ""

        return sorted(results, key=get_sort_key, reverse=desc)

    def _build_base_query(self, entity_type: str) -> "QueryBuilder":
        """
        Build base query for entity type.

        Args:
            entity_type: Entity type name (e.g., 'task', 'decision')

        Returns:
            QueryBuilder instance for the entity type
        """
        from cortical.got.query_builder import Query as QueryBuilder

        q = QueryBuilder(self.manager)

        # Map entity type to query method
        entity_type_lower = entity_type.lower()
        if entity_type_lower == 'task':
            return q.tasks()
        elif entity_type_lower == 'decision':
            return q.decisions()
        elif entity_type_lower == 'sprint':
            return q.sprints()
        elif entity_type_lower == 'edge':
            return q.edges()
        else:
            # Use generic entities() method for other types
            return q.entities(entity_type_lower)

    def _apply_expression(self, q: "QueryBuilder", expr: Expression) -> "QueryBuilder":
        """
        Apply expression filters to query builder.

        Args:
            q: Query builder instance
            expr: Expression AST node

        Returns:
            Modified query builder with filters applied
        """
        if isinstance(expr, Comparison):
            return self._apply_comparison(q, expr)
        elif isinstance(expr, AndExpr):
            return self._apply_and(q, expr)
        elif isinstance(expr, OrExpr):
            return self._apply_or(q, expr)
        elif isinstance(expr, NotExpr):
            return self._apply_not(q, expr)
        elif isinstance(expr, FunctionCall):
            # Function calls return results directly, not Query builders
            raise ExecutionError(
                "Function calls cannot be used as filter expressions. "
                "Use them as top-level queries instead."
            )
        else:
            raise ExecutionError(f"Unsupported expression type: {type(expr).__name__}")

    def _apply_comparison(self, q: "QueryBuilder", comp: Comparison) -> "QueryBuilder":
        """
        Apply comparison to query builder.

        Uses WhereClause directly to support operators beyond EQ, since the Query builder's
        public API only exposes .where() with implicit EQ operator. This is a documented
        workaround until the Query builder adds .where_op() or similar method.

        Args:
            q: Query builder instance
            comp: Comparison AST node

        Returns:
            Query builder with where clause applied
        """
        from cortical.got.query_builder import WhereClause

        # Extract field name and value
        field_name = self._evaluate(comp.field)
        value = self._evaluate(comp.value)

        # Map Op enum to Query builder operator string
        op_map = {
            Op.EQ: "eq",
            Op.NE: "ne",
            Op.GT: "gt",
            Op.LT: "lt",
            Op.GTE: "gte",
            Op.LTE: "lte",
            Op.IN: "in",
            Op.LIKE: "contains",  # Map LIKE to contains for substring matching
        }

        operator = op_map.get(comp.op)
        if operator is None:
            # Unsupported operators
            if comp.op == Op.NOT_IN:
                raise ExecutionError(
                    "NOT IN operator not supported. Use NOT (field IN values) instead."
                )
            elif comp.op == Op.NOT_LIKE:
                raise ExecutionError(
                    "NOT LIKE operator not supported. Use NOT (field LIKE pattern) instead."
                )
            else:
                raise ExecutionError(f"Unknown operator: {comp.op}")

        # NOTE: This directly modifies the query builder's internal state.
        # Ideally, Query builder would expose .where_op(field, operator, value)
        # This workaround will be replaced when that API is added.
        where_clause = WhereClause(field=field_name, value=value, operator=operator)
        q._where_clauses.append(where_clause)
        return q

    def _apply_and(self, q: "QueryBuilder", expr: AndExpr) -> "QueryBuilder":
        """
        Apply AND expression to query builder.

        AND expressions chain multiple where() calls.

        Args:
            q: Query builder instance
            expr: AndExpr AST node

        Returns:
            Query builder with all conditions applied
        """
        # Chain where() calls for each condition
        for child in expr.children:
            q = self._apply_expression(q, child)
        return q

    def _apply_or(self, q: "QueryBuilder", expr: OrExpr) -> "QueryBuilder":
        """
        Apply OR expression to query builder.

        Note: This is only called for simple OR expressions (all children are Comparisons).
        Complex OR expressions (with nested AND/OR) are handled by post-filtering.

        Args:
            q: Query builder instance
            expr: OrExpr AST node

        Returns:
            Query builder with OR conditions applied
        """
        # Use or_where() for each alternative
        for child in expr.children:
            if isinstance(child, Comparison):
                field_name = self._evaluate(child.field)
                value = self._evaluate(child.value)
                if child.op == Op.EQ:
                    q = q.or_where(**{field_name: value})
                else:
                    # For non-EQ operators in OR, we'd need more complex handling
                    # For now, only support EQ in OR expressions that use Query builder
                    raise ExecutionError(
                        f"Operator {child.op.name} in OR expressions requires post-filtering. "
                        "Use simple equality (=) operators in OR for best performance."
                    )
        return q


    def _evaluate(self, expr: Expression) -> Any:
        """Evaluate an expression node to a value."""
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Field):
            return expr.name
        else:
            raise ExecutionError(f"Cannot evaluate {type(expr).__name__} to a value")

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
        # IMPORTANT: func.kwargs is a tuple of tuples, not a dict!
        # Convert: ((key, value), ...) -> {key: value, ...}
        kwarg_values = {k: self._evaluate(v) for k, v in func.kwargs}

        # Instantiate and execute
        instance = func_class()
        return instance.execute(self.manager, arg_values, kwarg_values)


def execute(manager: "GoTManager", query: Query) -> Any:
    """Convenience function to execute a query."""
    return QueryExecutor(manager).execute(query)
