"""
Query Optimizer for GoT Expression System.

The optimizer analyzes query ASTs and generates execution plans that:
1. Use indexes when available (status, priority fields)
2. Estimate query costs
3. Provide EXPLAIN output for debugging
4. Warn about expensive operations

Usage:
    from cortical.got.expression import parse
    from cortical.got.expression.optimizer import QueryOptimizer

    query = parse("status = 'pending' AND priority = 'high'")
    optimizer = QueryOptimizer()
    plan = optimizer.explain(query)
    print(plan)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .ast import (
    Expression,
    Query,
    Comparison,
    AndExpr,
    OrExpr,
    NotExpr,
    FunctionCall,
    Field,
    Literal,
    Op,
)


@dataclass
class QueryPlan:
    """
    Represents an optimized execution plan for a query.

    Attributes:
        steps: Human-readable execution steps
        estimated_cost: Estimated cost (0.0-1.0 scale, lower is better)
        uses_index: Whether any indexes are used
        index_fields: List of indexed fields used
        warnings: List of performance warnings
        requires_post_filter: Whether post-filtering is required
    """
    steps: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    uses_index: bool = False
    index_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_post_filter: bool = False


@dataclass
class SchemaInfo:
    """
    Schema information for optimization decisions.

    Attributes:
        indexed_fields: Set of fields that have indexes
        field_cardinality: Estimated cardinality for enum fields
        total_entities: Approximate count of entities (for cost estimation)
    """
    indexed_fields: Set[str] = field(default_factory=lambda: {'status', 'priority'})
    field_cardinality: Dict[str, int] = field(default_factory=lambda: {
        'status': 4,      # pending, in_progress, completed, blocked
        'priority': 4,    # low, medium, high, critical
    })
    total_entities: int = 100  # Default assumption


class QueryOptimizer:
    """
    Optimizes query ASTs for efficient execution.

    The optimizer is schema-aware and understands:
    - Which fields are indexed (status, priority)
    - Cardinality of enum fields
    - Cost of different operations (index scan vs full scan)

    Example:
        optimizer = QueryOptimizer()
        query = parse("status = 'pending' AND priority = 'high'")
        plan = optimizer.optimize(query)
        explanation = optimizer.explain(query)
        print(explanation)
    """

    def __init__(self, schema_info: Optional[SchemaInfo] = None):
        """
        Initialize the optimizer.

        Args:
            schema_info: Optional schema information for optimization.
                        If None, uses default schema with status/priority indexes.
        """
        self.schema = schema_info or SchemaInfo()

    def optimize(self, query: Query) -> QueryPlan:
        """
        Analyze a query and generate an optimized execution plan.

        Args:
            query: Query AST to optimize

        Returns:
            QueryPlan with execution steps and cost estimates
        """
        plan = QueryPlan()

        if query.expression is None:
            # Empty query - full scan
            plan.steps.append("Full scan (no filters)")
            plan.estimated_cost = 1.0
            plan.warnings.append("No filters specified - will scan all entities")
            return plan

        # Analyze the expression tree
        self._analyze_expression(query.expression, plan)

        # Add ORDER BY cost if present
        if query.order_by:
            field, desc = query.order_by
            plan.steps.append(f"Sort by {field} ({'DESC' if desc else 'ASC'})")
            # Sorting adds moderate cost
            plan.estimated_cost += 0.1

        # Add LIMIT/OFFSET information
        if query.offset:
            plan.steps.append(f"Skip first {query.offset} results")
        if query.limit:
            plan.steps.append(f"Limit to {query.limit} results")
            # LIMIT reduces cost slightly (less data to return)
            plan.estimated_cost *= 0.9

        return plan

    def _analyze_expression(self, expr: Expression, plan: QueryPlan) -> None:
        """
        Recursively analyze an expression tree and update the plan.

        Args:
            expr: Expression node to analyze
            plan: QueryPlan to update with analysis results
        """
        if isinstance(expr, Comparison):
            self._analyze_comparison(expr, plan)

        elif isinstance(expr, AndExpr):
            self._analyze_and(expr, plan)

        elif isinstance(expr, OrExpr):
            self._analyze_or(expr, plan)

        elif isinstance(expr, NotExpr):
            self._analyze_not(expr, plan)

        elif isinstance(expr, FunctionCall):
            self._analyze_function(expr, plan)

        else:
            # Unknown expression type - conservative estimate
            plan.steps.append(f"Evaluate {type(expr).__name__}")
            plan.estimated_cost += 0.5

    def _analyze_comparison(self, comp: Comparison, plan: QueryPlan) -> None:
        """Analyze a comparison expression."""
        field_name = comp.field.name
        value = self._extract_literal_value(comp.value)

        # Check if field is indexed
        if field_name in self.schema.indexed_fields:
            plan.uses_index = True
            plan.index_fields.append(field_name)

            # Estimate selectivity based on field cardinality
            cardinality = self.schema.field_cardinality.get(field_name, 10)
            selectivity = 1.0 / cardinality

            if comp.op == Op.EQ:
                plan.steps.append(f"Index scan on '{field_name}' = {value!r}")
                plan.estimated_cost += 0.1 * selectivity
            elif comp.op in (Op.IN, Op.NOT_IN):
                if isinstance(value, (list, tuple)):
                    count = len(value)
                    plan.steps.append(f"Index scan on '{field_name}' IN {count} values")
                    plan.estimated_cost += 0.1 * selectivity * count
                else:
                    plan.steps.append(f"Index scan on '{field_name}' IN values")
                    plan.estimated_cost += 0.1 * selectivity
            else:
                # Other operators might not use index efficiently
                plan.steps.append(f"Index scan on '{field_name}' {comp.op.name} {value!r}")
                plan.estimated_cost += 0.2
                if comp.op in (Op.GT, Op.LT, Op.GTE, Op.LTE):
                    plan.warnings.append(
                        f"Range operator {comp.op.name} on indexed field '{field_name}' "
                        "may scan multiple index buckets"
                    )
        else:
            # No index - must do full scan with filter
            plan.steps.append(f"Full scan with filter: {field_name} {comp.op.name} {value!r}")
            plan.estimated_cost += 0.8
            plan.warnings.append(
                f"Field '{field_name}' is not indexed - requires full table scan"
            )

    def _analyze_and(self, expr: AndExpr, plan: QueryPlan) -> None:
        """Analyze an AND expression."""
        # AND of indexed fields = intersect index results (efficient)
        # AND with non-indexed = apply filters sequentially

        indexed_children = []
        non_indexed_children = []

        for child in expr.children:
            if self._uses_index(child):
                indexed_children.append(child)
            else:
                non_indexed_children.append(child)

        # Analyze indexed children first (most efficient)
        if indexed_children:
            if len(indexed_children) > 1:
                plan.steps.append(f"Intersect {len(indexed_children)} index scans")

            for child in indexed_children:
                self._analyze_expression(child, plan)

            # AND reduces result set - lower cost multiplier
            plan.estimated_cost *= 0.5

        # Then analyze non-indexed children (filter results)
        if non_indexed_children:
            if indexed_children:
                plan.steps.append(f"Apply {len(non_indexed_children)} additional filters")

            for child in non_indexed_children:
                self._analyze_expression(child, plan)

    def _analyze_or(self, expr: OrExpr, plan: QueryPlan) -> None:
        """Analyze an OR expression."""
        # OR of indexed fields = union index results (efficient)
        # OR with complex expressions = may require post-filtering

        all_indexed = all(self._uses_index(child) for child in expr.children)

        if all_indexed and all(isinstance(child, Comparison) for child in expr.children):
            # Simple OR of indexed comparisons - can use index union
            plan.steps.append(f"Union {len(expr.children)} index scans")
            for child in expr.children:
                self._analyze_expression(child, plan)

            # OR increases result set but index union is still efficient
            plan.estimated_cost *= 1.2
        else:
            # Complex OR - requires post-filtering
            plan.requires_post_filter = True
            plan.steps.append(f"Complex OR with {len(expr.children)} branches (post-filter)")

            for child in expr.children:
                self._analyze_expression(child, plan)

            # Post-filtering is expensive
            plan.estimated_cost *= 2.0
            plan.warnings.append(
                "Complex OR expression requires post-filtering (full scan + filter)"
            )

    def _analyze_not(self, expr: NotExpr, plan: QueryPlan) -> None:
        """Analyze a NOT expression."""
        # NOT always requires post-filtering (can't use indexes for negation)
        plan.requires_post_filter = True
        plan.steps.append("Negation (requires post-filter)")

        self._analyze_expression(expr.child, plan)

        # Negation requires full scan then filter
        plan.estimated_cost = max(plan.estimated_cost, 0.9)
        plan.warnings.append(
            "NOT operator requires full scan with post-filtering"
        )

    def _analyze_function(self, func: FunctionCall, plan: QueryPlan) -> None:
        """Analyze a function call."""
        # Function cost depends on the function
        # Graph traversal functions are potentially expensive

        if func.name in ('connected_to', 'path', 'traverse'):
            plan.steps.append(f"Graph traversal: {func.name}()")
            plan.estimated_cost += 0.6
            plan.warnings.append(
                f"Graph function '{func.name}' may be expensive for large graphs"
            )
        elif func.name in ('blocked', 'recent', 'active'):
            plan.steps.append(f"Function call: {func.name}()")
            plan.estimated_cost += 0.3
        else:
            plan.steps.append(f"Function call: {func.name}()")
            plan.estimated_cost += 0.4

        # Functions require post-filtering when combined with other expressions
        plan.requires_post_filter = True

    def _uses_index(self, expr: Expression) -> bool:
        """Check if an expression can use an index."""
        if isinstance(expr, Comparison):
            field_name = expr.field.name
            return field_name in self.schema.indexed_fields
        elif isinstance(expr, AndExpr):
            # AND uses index if any child does
            return any(self._uses_index(child) for child in expr.children)
        elif isinstance(expr, OrExpr):
            # OR can use index if all children are simple indexed comparisons
            return all(
                isinstance(child, Comparison) and child.field.name in self.schema.indexed_fields
                for child in expr.children
            )
        else:
            return False

    def _extract_literal_value(self, expr: Expression) -> Any:
        """Extract the literal value from an expression."""
        if isinstance(expr, Literal):
            return expr.value
        return "<complex>"

    def explain(self, query: Query) -> str:
        """
        Generate a human-readable query plan explanation.

        Args:
            query: Query AST to explain

        Returns:
            Formatted explanation string
        """
        plan = self.optimize(query)

        lines = []
        lines.append("=" * 70)
        lines.append("QUERY EXECUTION PLAN")
        lines.append("=" * 70)
        lines.append("")

        # Query text (if available)
        lines.append("Steps:")
        for i, step in enumerate(plan.steps, 1):
            lines.append(f"  {i}. {step}")

        lines.append("")
        lines.append(f"Estimated cost: {plan.estimated_cost:.2f}")
        lines.append(f"Uses indexes: {plan.uses_index}")

        if plan.index_fields:
            lines.append(f"Indexed fields: {', '.join(plan.index_fields)}")

        if plan.requires_post_filter:
            lines.append("Requires post-filtering: Yes")

        if plan.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in plan.warnings:
                lines.append(f"  ⚠ {warning}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def estimate_cost(self, query: Query) -> float:
        """
        Estimate the execution cost of a query.

        Args:
            query: Query AST to estimate

        Returns:
            Estimated cost (0.0-1.0+ scale, lower is better)
        """
        plan = self.optimize(query)
        return plan.estimated_cost
