"""
Fluent Query Builder for Graph of Thought.

This module provides a powerful, SQL-like query interface for the GoT graph.
It follows the Builder pattern for intuitive method chaining and supports
lazy evaluation for memory efficiency on large graphs.

USAGE EXAMPLES
--------------

Basic queries:
    >>> Query(manager).tasks().where(status="pending").execute()
    [Task(...), Task(...)]

Complex filters with multiple conditions:
    >>> results = (
    ...     Query(manager)
    ...     .tasks()
    ...     .where(status="pending")      # AND condition
    ...     .where(priority="high")       # AND condition
    ...     .or_where(priority="critical") # OR alternative
    ...     .connected_to(sprint_id)      # Must be connected to sprint
    ...     .order_by("created_at", desc=True)
    ...     .limit(10)
    ...     .execute()
    ... )

Aggregation with GROUP BY:
    >>> counts = (
    ...     Query(manager)
    ...     .tasks()
    ...     .group_by("status")
    ...     .count()
    ...     .execute()
    ... )
    {"pending": 5, "completed": 3, "in_progress": 2}

Lazy iteration (memory efficient):
    >>> for task in Query(manager).tasks().iter():
    ...     process(task)
    ...     if should_stop:
    ...         break  # Stops iteration early, no wasted work

DESIGN PATTERNS
---------------
- Builder Pattern: Method chaining for query construction
- Strategy Pattern: Different execution strategies (eager, lazy, indexed)
- Iterator Pattern: Lazy evaluation with generators
- Composite Pattern: OR groups containing multiple WHERE clauses

FILTER LOGIC
------------
The WHERE/OR logic follows these rules:
- Multiple .where() calls: AND (all must match)
- .or_where(): Creates OR alternative to WHERE
- Combined: (WHERE conditions) OR (any OR group)

Example: .where(status="pending").or_where(priority="critical")
         Matches: status="pending" OR priority="critical"

PERFORMANCE NOTES
-----------------
- Use .limit() to avoid loading all results
- Use .iter() for memory-efficient streaming
- Use .exists() when you only need boolean check
- Use .first() when you only need one result
- .explain() shows the query plan without executing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import time
from .api import GoTManager
from .types import Task, Decision, Edge, Sprint, Entity


T = TypeVar("T", bound=Entity)


# ============================================================================
# QUERY METRICS
# ============================================================================

class QueryMetrics:
    """
    Collects query execution metrics for the GoT Query API.

    Tracks:
    - Query execution times (min, max, avg, total)
    - Entity counts per query
    - Cache hit rates from GoTManager
    - Query counts by type

    Thread Safety: NOT thread-safe. Use external locking for concurrent access.

    Example:
        metrics = QueryMetrics()
        query = Query(manager, metrics=metrics).tasks().execute()
        print(metrics.summary())
    """

    def __init__(self, enabled: bool = True):
        """Initialize metrics collector."""
        self.enabled = enabled
        self._query_times: List[float] = []  # List of execution times in ms
        self._entity_counts: List[int] = []  # Entities returned per query
        self._query_counts: Dict[str, int] = defaultdict(int)  # By entity type
        self._total_queries = 0
        self._cache_hits_before = 0
        self._cache_misses_before = 0

    def start_query(self, manager: GoTManager, entity_type: str) -> float:
        """Record query start. Returns start time."""
        if not self.enabled:
            return 0.0

        # Capture cache state before query
        cache_stats = manager.cache_stats()
        self._cache_hits_before = cache_stats.get('hits', 0)
        self._cache_misses_before = cache_stats.get('misses', 0)

        return time.perf_counter()

    def end_query(
        self,
        manager: GoTManager,
        entity_type: str,
        start_time: float,
        result_count: int
    ) -> None:
        """Record query completion."""
        if not self.enabled:
            return

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        self._query_times.append(duration_ms)
        self._entity_counts.append(result_count)
        self._query_counts[entity_type] += 1
        self._total_queries += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get query metrics statistics."""
        if not self._query_times:
            return {
                'total_queries': 0,
                'total_entities': 0,
                'avg_time_ms': 0.0,
                'min_time_ms': 0.0,
                'max_time_ms': 0.0,
                'queries_by_type': {},
            }

        return {
            'total_queries': self._total_queries,
            'total_entities': sum(self._entity_counts),
            'avg_entities_per_query': sum(self._entity_counts) / len(self._entity_counts),
            'avg_time_ms': sum(self._query_times) / len(self._query_times),
            'min_time_ms': min(self._query_times),
            'max_time_ms': max(self._query_times),
            'queries_by_type': dict(self._query_counts),
        }

    def summary(self) -> str:
        """Get human-readable metrics summary."""
        stats = self.get_stats()
        lines = [
            "GoT Query API Metrics",
            "=" * 40,
            f"Total queries: {stats['total_queries']}",
            f"Total entities: {stats['total_entities']}",
            f"Avg entities/query: {stats.get('avg_entities_per_query', 0):.1f}",
            "",
            "Timing:",
            f"  Avg: {stats['avg_time_ms']:.2f}ms",
            f"  Min: {stats['min_time_ms']:.2f}ms",
            f"  Max: {stats['max_time_ms']:.2f}ms",
            "",
            "Queries by type:",
        ]
        for entity_type, count in stats['queries_by_type'].items():
            lines.append(f"  {entity_type}: {count}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self._query_times.clear()
        self._entity_counts.clear()
        self._query_counts.clear()
        self._total_queries = 0


# Module-level default metrics collector (disabled by default)
_default_metrics = QueryMetrics(enabled=False)


def get_query_metrics() -> QueryMetrics:
    """Get the module-level query metrics collector."""
    return _default_metrics


def enable_query_metrics() -> None:
    """Enable module-level query metrics collection."""
    _default_metrics.enabled = True


def disable_query_metrics() -> None:
    """Disable module-level query metrics collection."""
    _default_metrics.enabled = False


class EntityType(Enum):
    """Types of entities that can be queried."""
    TASK = auto()
    DECISION = auto()
    SPRINT = auto()
    EDGE = auto()
    HANDOFF = auto()


class SortOrder(Enum):
    """Sort order for results."""
    ASC = auto()
    DESC = auto()


@dataclass
class WhereClause:
    """A single WHERE condition."""
    field: str
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, contains


@dataclass
class OrGroup:
    """Group of WHERE clauses joined by OR."""
    clauses: List[WhereClause] = field(default_factory=list)


@dataclass
class OrderByClause:
    """ORDER BY clause."""
    field: str
    order: SortOrder = SortOrder.ASC


@dataclass
class ConnectionFilter:
    """Filter for connected entities."""
    entity_id: str
    edge_type: Optional[str] = None
    direction: str = "any"  # incoming, outgoing, any


@dataclass
class QueryPlan:
    """Execution plan for a query."""
    steps: List[Dict[str, Any]]
    estimated_cost: float
    uses_index: bool
    index_name: Optional[str]

    def __getitem__(self, key: str) -> Any:
        """Support dict-like access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return hasattr(self, key)


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================
#
# These classes implement the Strategy pattern for aggregation operations.
# Each aggregation follows a three-phase lifecycle:
#   1. initial() - Create the starting accumulator value
#   2. accumulate() - Called once per entity, updates accumulator
#   3. finalize() - Convert accumulator to final result
#
# This design allows streaming aggregation without loading all data first.
# ============================================================================


class AggregateFunction(ABC):
    """
    Base class for aggregate functions (Strategy pattern).

    Subclasses implement specific aggregation logic like COUNT, SUM, AVG.
    The three-phase design (initial -> accumulate -> finalize) enables
    streaming aggregation over large result sets.

    Example implementation:
        class Sum(AggregateFunction):
            def initial(self): return 0
            def accumulate(self, acc, entity): return acc + entity.value
            def finalize(self, acc): return acc
    """

    @abstractmethod
    def initial(self) -> Any:
        """Return initial accumulator value (e.g., 0 for Count, [] for Collect)."""
        pass

    @abstractmethod
    def accumulate(self, acc: Any, value: Any) -> Any:
        """Update accumulator with entity. Called once per matching entity."""
        pass

    @abstractmethod
    def finalize(self, acc: Any) -> Any:
        """Convert accumulator to final result (e.g., compute average from sum/count)."""
        pass


class Count(AggregateFunction):
    """Count items in a group."""

    def initial(self) -> int:
        return 0

    def accumulate(self, acc: int, value: Any) -> int:
        return acc + 1

    def finalize(self, acc: int) -> int:
        return acc


class Collect(AggregateFunction):
    """Collect field values into a list."""

    def __init__(self, field: str):
        self.field = field

    def initial(self) -> List:
        return []

    def accumulate(self, acc: List, entity: Any) -> List:
        value = getattr(entity, self.field, None)
        if value is None and hasattr(entity, 'properties'):
            value = entity.properties.get(self.field)
        if value is not None:
            acc.append(value)
        return acc

    def finalize(self, acc: List) -> List:
        return acc


class Sum(AggregateFunction):
    """Sum numeric field values."""

    def __init__(self, field: str):
        self.field = field

    def initial(self) -> float:
        return 0.0

    def accumulate(self, acc: float, entity: Any) -> float:
        value = getattr(entity, self.field, None)
        if value is None and hasattr(entity, 'properties'):
            value = entity.properties.get(self.field)
        if isinstance(value, (int, float)):
            return acc + value
        return acc

    def finalize(self, acc: float) -> float:
        return acc


class Avg(AggregateFunction):
    """
    Calculate average of numeric field values.

    Uses a (sum, count) tuple as accumulator to avoid storing all values.
    This is memory-efficient for large datasets.

    Example:
        Query(manager).tasks().group_by("priority").aggregate(avg_time=Avg("duration"))
    """

    def __init__(self, field: str):
        self.field = field

    def initial(self) -> Tuple[float, int]:
        # Tuple of (running_sum, count) - allows one-pass average calculation
        return (0.0, 0)

    def accumulate(self, acc: Tuple[float, int], entity: Any) -> Tuple[float, int]:
        total, count = acc
        # Try direct attribute first, then properties dict
        value = getattr(entity, self.field, None)
        if value is None and hasattr(entity, 'properties'):
            value = entity.properties.get(self.field)
        # Only accumulate numeric values, skip None/strings
        if isinstance(value, (int, float)):
            return (total + value, count + 1)
        return acc

    def finalize(self, acc: Tuple[float, int]) -> float:
        # Compute average from accumulated sum and count
        total, count = acc
        return total / count if count > 0 else 0.0


class Min(AggregateFunction):
    """Find minimum value of a field."""

    def __init__(self, field: str):
        self.field = field

    def initial(self) -> Optional[Any]:
        return None

    def accumulate(self, acc: Optional[Any], entity: Any) -> Optional[Any]:
        value = getattr(entity, self.field, None)
        if value is None and hasattr(entity, 'properties'):
            value = entity.properties.get(self.field)
        if value is None:
            return acc
        if acc is None:
            return value
        return min(acc, value)

    def finalize(self, acc: Optional[Any]) -> Optional[Any]:
        return acc


class Max(AggregateFunction):
    """Find maximum value of a field."""

    def __init__(self, field: str):
        self.field = field

    def initial(self) -> Optional[Any]:
        return None

    def accumulate(self, acc: Optional[Any], entity: Any) -> Optional[Any]:
        value = getattr(entity, self.field, None)
        if value is None and hasattr(entity, 'properties'):
            value = entity.properties.get(self.field)
        if value is None:
            return acc
        if acc is None:
            return value
        return max(acc, value)

    def finalize(self, acc: Optional[Any]) -> Optional[Any]:
        return acc


# ============================================================================
# QUERY BUILDER
# ============================================================================


class Query(Generic[T]):
    """
    Fluent query builder for GoT entities.

    Example:
        results = (
            Query(manager)
            .tasks()
            .where(status="pending", priority="high")
            .order_by("created_at", desc=True)
            .limit(10)
            .execute()
        )
    """

    def __init__(self, manager: GoTManager, metrics: Optional[QueryMetrics] = None):
        """
        Initialize query builder with GoT manager.

        Args:
            manager: GoTManager instance for entity access
            metrics: Optional QueryMetrics for timing/counting. If None, uses
                     module-level metrics (disabled by default).
        """
        self._manager = manager
        self._metrics = metrics if metrics is not None else _default_metrics
        self._entity_type: Optional[EntityType] = None
        self._where_clauses: List[WhereClause] = []
        self._or_groups: List[OrGroup] = []
        self._order_by: List[OrderByClause] = []
        self._limit_value: Optional[int] = None
        self._offset_value: int = 0
        self._connections: List[ConnectionFilter] = []
        self._group_by_fields: List[str] = []
        self._aggregates: Dict[str, AggregateFunction] = {}
        self._index_hint: Optional[str] = None
        self._executed = False
        self._cached_results: Optional[List[T]] = None
        self._count_mode = False  # When True, execute returns counts

    def tasks(self) -> "Query[Task]":
        """Query tasks."""
        self._entity_type = EntityType.TASK
        return self

    def sprints(self) -> "Query[Sprint]":
        """Query sprints."""
        self._entity_type = EntityType.SPRINT
        return self

    def decisions(self) -> "Query[Decision]":
        """Query decisions."""
        self._entity_type = EntityType.DECISION
        return self

    def edges(self) -> "Query[Edge]":
        """Query edges."""
        self._entity_type = EntityType.EDGE
        return self

    def where(self, **conditions) -> "Query[T]":
        """
        Add WHERE conditions (AND).

        Args:
            **conditions: Field=value pairs to filter by

        Returns:
            Self for chaining
        """
        for field, value in conditions.items():
            self._where_clauses.append(WhereClause(field=field, value=value))
        return self

    def or_where(self, **conditions) -> "Query[T]":
        """
        Add OR conditions.

        Args:
            **conditions: Field=value pairs to filter by

        Returns:
            Self for chaining
        """
        group = OrGroup()
        for field, value in conditions.items():
            group.clauses.append(WhereClause(field=field, value=value))
        self._or_groups.append(group)
        return self

    def connected_to(
        self,
        entity_id: str,
        via: Optional[str] = None,
        direction: str = "any"
    ) -> "Query[T]":
        """
        Filter to entities connected to the given entity.

        Args:
            entity_id: ID of entity to find connections to
            via: Optional edge type to filter by
            direction: "incoming", "outgoing", or "any"

        Returns:
            Self for chaining
        """
        self._connections.append(ConnectionFilter(
            entity_id=entity_id,
            edge_type=via,
            direction=direction
        ))
        return self

    def order_by(self, field: str, desc: bool = False) -> "Query[T]":
        """
        Add ORDER BY clause.

        Args:
            field: Field to sort by
            desc: True for descending order

        Returns:
            Self for chaining
        """
        order = SortOrder.DESC if desc else SortOrder.ASC
        self._order_by.append(OrderByClause(field=field, order=order))
        return self

    def limit(self, n: int) -> "Query[T]":
        """
        Limit number of results.

        Args:
            n: Maximum number of results

        Returns:
            Self for chaining
        """
        self._limit_value = n
        return self

    def offset(self, n: int) -> "Query[T]":
        """
        Skip first N results.

        Args:
            n: Number of results to skip

        Returns:
            Self for chaining
        """
        self._offset_value = n
        return self

    def group_by(self, *fields: str) -> "Query[T]":
        """
        Group results by fields for aggregation.

        Args:
            *fields: Field names to group by

        Returns:
            Self for chaining
        """
        self._group_by_fields.extend(fields)
        return self

    def aggregate(self, **aggregates: AggregateFunction) -> "Query[T]":
        """
        Add aggregate functions.

        Args:
            **aggregates: name=AggregateFunction pairs

        Returns:
            Self for chaining
        """
        self._aggregates.update(aggregates)
        return self

    def use_index(self, index_name: str) -> "Query[T]":
        """
        Hint to use a specific index for optimization.

        Args:
            index_name: Name of index to use

        Returns:
            Self for chaining
        """
        self._index_hint = index_name
        return self

    # ========================================================================
    # EXECUTION METHODS
    # ========================================================================

    def execute(self) -> Union[List[T], Dict[Any, Any]]:
        """
        Execute the query and return results.

        Returns:
            List of matching entities, or dict if using group_by/aggregate
        """
        # Start metrics timing
        entity_type_name = self._entity_type.name if self._entity_type else "UNKNOWN"
        start_time = self._metrics.start_query(self._manager, entity_type_name)

        try:
            if self._count_mode and self._group_by_fields:
                # Count mode with group_by - return counts per group
                groups: Dict[Any, int] = defaultdict(int)
                for entity in self._execute_query():
                    key = self._get_group_key(entity)
                    groups[key] += 1
                result = dict(groups)
                self._metrics.end_query(
                    self._manager, entity_type_name, start_time, sum(groups.values())
                )
                return result

            if self._group_by_fields or self._aggregates:
                result = self._execute_aggregation()
                self._metrics.end_query(
                    self._manager, entity_type_name, start_time, len(result) if isinstance(result, dict) else 0
                )
                return result

            self._executed = True
            results = list(self._execute_query())

            # Apply sorting
            if self._order_by:
                results = self._apply_sorting(results)

            # Apply offset and limit
            if self._offset_value:
                results = results[self._offset_value:]
            if self._limit_value is not None:
                results = results[:self._limit_value]

            self._cached_results = results

            # Record metrics
            self._metrics.end_query(self._manager, entity_type_name, start_time, len(results))

            return results
        except Exception:
            # Record failed query (0 results)
            self._metrics.end_query(self._manager, entity_type_name, start_time, 0)
            raise

    def iter(self) -> Generator[T, None, None]:
        """
        Lazily iterate over results.

        Yields:
            Entities matching the query
        """
        count = 0
        skipped = 0

        for entity in self._execute_query():
            # Handle offset
            if skipped < self._offset_value:
                skipped += 1
                continue

            # Handle limit
            if self._limit_value is not None and count >= self._limit_value:
                break

            yield entity
            count += 1

    def count(self) -> Union[int, Dict[Any, int], "Query[T]"]:
        """
        Count matching entities.

        When used with group_by, returns self for chaining (call execute() after).
        Otherwise returns count directly.

        Returns:
            Count as int, or self for chaining with group_by
        """
        if self._group_by_fields:
            # With group_by, enable count mode and return self for chaining
            self._count_mode = True
            return self

        # Without group_by, return count directly
        return sum(1 for _ in self._execute_query())

    def exists(self) -> bool:
        """
        Check if any entities match.

        Returns:
            True if at least one entity matches
        """
        for _ in self._execute_query():
            return True
        return False

    def first(self) -> Optional[T]:
        """
        Get first matching entity.

        Returns:
            First entity or None
        """
        for entity in self._execute_query():
            return entity
        return None

    def explain(self) -> QueryPlan:
        """
        Get query execution plan without executing.

        Returns:
            QueryPlan describing how the query would be executed
        """
        steps = []

        # Step 1: Entity source
        steps.append({
            "type": "scan",
            "entity_type": self._entity_type.name if self._entity_type else "unknown",
            "index": self._index_hint,
        })

        # Step 2: Filters
        if self._where_clauses:
            steps.append({
                "type": "filter",
                "conditions": [
                    {"field": c.field, "op": c.operator, "value": str(c.value)}
                    for c in self._where_clauses
                ],
            })

        # Step 3: Connection filters
        if self._connections:
            steps.append({
                "type": "connection_filter",
                "connections": [
                    {"entity_id": c.entity_id, "edge_type": c.edge_type}
                    for c in self._connections
                ],
            })

        # Step 4: Sorting
        if self._order_by:
            steps.append({
                "type": "sort",
                "fields": [
                    {"field": o.field, "order": o.order.name}
                    for o in self._order_by
                ],
            })

        # Step 5: Pagination
        if self._limit_value or self._offset_value:
            steps.append({
                "type": "pagination",
                "limit": self._limit_value,
                "offset": self._offset_value,
            })

        return QueryPlan(
            steps=steps,
            estimated_cost=len(steps) * 10.0,  # Simplified cost model
            uses_index=self._index_hint is not None,
            index_name=self._index_hint,
        )

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _execute_query(self) -> Generator[T, None, None]:
        """Execute query and yield matching entities."""
        # Get base entities
        entities = self._get_base_entities()

        # Get connected entity IDs if we have connection filters
        connected_ids: Optional[Set[str]] = None
        if self._connections:
            connected_ids = self._get_connected_ids()

        # Filter entities
        for entity in entities:
            if self._matches_filters(entity, connected_ids):
                yield entity

    def _get_base_entities(self) -> List[Any]:
        """Get base entities based on entity type."""
        if self._entity_type == EntityType.TASK:
            return self._manager.list_all_tasks()
        elif self._entity_type == EntityType.SPRINT:
            return self._manager.list_sprints()
        elif self._entity_type == EntityType.DECISION:
            return self._manager.list_decisions()
        elif self._entity_type == EntityType.EDGE:
            return self._manager.list_edges()
        else:
            return []

    def _get_connected_ids(self) -> Set[str]:
        """Get IDs of entities connected per connection filters."""
        connected = set()

        # Load edges once for all connection filters (query-level caching)
        edges = self._manager.list_edges()

        for conn in self._connections:
            for edge in edges:
                # Check edge type filter
                if conn.edge_type and edge.edge_type != conn.edge_type:
                    continue

                # Check direction and add connected entity
                if conn.direction in ("any", "outgoing"):
                    if edge.source_id == conn.entity_id:
                        connected.add(edge.target_id)
                if conn.direction in ("any", "incoming"):
                    if edge.target_id == conn.entity_id:
                        connected.add(edge.source_id)

        return connected

    def _matches_filters(
        self,
        entity: Any,
        connected_ids: Optional[Set[str]]
    ) -> bool:
        """Check if entity matches all filters."""
        # Check connection filter first
        if connected_ids is not None:
            entity_id = getattr(entity, 'id', None)
            if entity_id not in connected_ids:
                return False

        # If no WHERE or OR clauses, match all
        if not self._where_clauses and not self._or_groups:
            return True

        # Check WHERE clauses (all must match)
        where_matches = all(
            self._matches_clause(entity, clause)
            for clause in self._where_clauses
        ) if self._where_clauses else False

        # Check OR groups (any group can match if all its clauses match)
        or_matches = any(
            all(self._matches_clause(entity, c) for c in group.clauses)
            for group in self._or_groups
        ) if self._or_groups else False

        # If we have both WHERE and OR, it's WHERE OR (any OR group)
        if self._where_clauses and self._or_groups:
            return where_matches or or_matches

        # If only WHERE clauses, all must match
        if self._where_clauses:
            return where_matches

        # If only OR groups, at least one group must match
        return or_matches

    def _matches_clause(self, entity: Any, clause: WhereClause) -> bool:
        """Check if entity matches a single WHERE clause."""
        # Get field value
        value = getattr(entity, clause.field, None)
        if value is None and hasattr(entity, 'properties'):
            value = entity.properties.get(clause.field)

        # Compare based on operator
        if clause.operator == "eq":
            return value == clause.value
        elif clause.operator == "ne":
            return value != clause.value
        elif clause.operator == "gt":
            return value is not None and value > clause.value
        elif clause.operator == "lt":
            return value is not None and value < clause.value
        elif clause.operator == "gte":
            return value is not None and value >= clause.value
        elif clause.operator == "lte":
            return value is not None and value <= clause.value
        elif clause.operator == "in":
            return value in clause.value
        elif clause.operator == "contains":
            return clause.value in (value or "")

        return False

    def _apply_sorting(self, results: List[T]) -> List[T]:
        """Apply ORDER BY clauses to results."""
        if not self._order_by:
            return results

        # Priority order for sorting
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        def sort_key(entity):
            keys = []
            for order_clause in self._order_by:
                value = getattr(entity, order_clause.field, None)
                if value is None and hasattr(entity, 'properties'):
                    value = entity.properties.get(order_clause.field)

                # Handle priority specially
                if order_clause.field == "priority" and isinstance(value, str):
                    value = priority_order.get(value, 99)

                # Handle None values
                if value is None:
                    value = "" if isinstance(value, str) else float('inf')

                keys.append(value)
            return tuple(keys)

        reverse = any(o.order == SortOrder.DESC for o in self._order_by)
        return sorted(results, key=sort_key, reverse=reverse)

    def _get_group_key(self, entity: Any) -> Any:
        """Get grouping key for an entity."""
        if len(self._group_by_fields) == 1:
            field = self._group_by_fields[0]
            value = getattr(entity, field, None)
            if value is None and hasattr(entity, 'properties'):
                value = entity.properties.get(field)
            return value

        # Multiple fields - return tuple
        values = []
        for field in self._group_by_fields:
            value = getattr(entity, field, None)
            if value is None and hasattr(entity, 'properties'):
                value = entity.properties.get(field)
            values.append(value)
        return tuple(values)

    def _execute_aggregation(self) -> Dict[Any, Any]:
        """Execute query with aggregation."""
        groups: Dict[Any, Dict[str, Any]] = defaultdict(
            lambda: {name: agg.initial() for name, agg in self._aggregates.items()}
        )

        # Also track count for simple count() calls
        counts: Dict[Any, int] = defaultdict(int)

        for entity in self._execute_query():
            key = self._get_group_key(entity)
            counts[key] += 1

            for name, agg in self._aggregates.items():
                groups[key][name] = agg.accumulate(groups[key][name], entity)

        # Finalize aggregates
        result = {}
        for key in counts:
            if self._aggregates:
                result[key] = {
                    name: agg.finalize(groups[key][name])
                    for name, agg in self._aggregates.items()
                }
            else:
                result[key] = counts[key]

        return result
