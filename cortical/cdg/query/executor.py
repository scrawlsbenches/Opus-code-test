"""
Query executor for CDG queries.

Executes query plans against CDG storage using:
- CDGIndexManager for indexed field lookups
- CDGStore for entity retrieval
- SchemaRegistry for entity type resolution

See: docs/design/cdg-query-language.md
"""

import fnmatch
import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .ast import (
    CDGQuery, Expression, Comparison, AndExpr, OrExpr, NotExpr,
    FunctionCall, Literal, Op
)
from .planner import QueryPlan, PlanStrategy, IndexLookup
from .registry import FunctionRegistry, QueryContext
from .errors import (
    QueryExecutionError, QueryValidationError, QueryNotImplementedError
)

if TYPE_CHECKING:
    from cortical.cdg.storage import CDGStore
    from cortical.cdg.index_manager import CDGIndexManager
    from cortical.cdg.schema import SchemaRegistry

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Executes CDG query plans.

    The executor takes a QueryPlan and executes it against the CDG
    storage layer, using indexes where available and falling back
    to full scans when necessary.
    """

    def __init__(
        self,
        store: Optional["CDGStore"] = None,
        index_manager: Optional["CDGIndexManager"] = None,
        schema_registry: Optional["SchemaRegistry"] = None,
        context: Optional[QueryContext] = None
    ):
        self.store = store
        self.index_manager = index_manager
        self.schema_registry = schema_registry
        self.context = context or QueryContext(
            store=store,
            index_manager=index_manager,
            schema_registry=schema_registry
        )

    def execute(self, plan: QueryPlan) -> List[Any]:
        """
        Execute a query plan and return results.

        Args:
            plan: The query execution plan

        Returns:
            List of entities matching the query
        """
        if plan.strategy == PlanStrategy.FUNCTION_CALL:
            return self._execute_function_call(plan)

        # Execute entity query based on strategy
        if plan.strategy == PlanStrategy.INDEX_INTERSECT:
            entities = self._execute_index_intersect(plan)
        elif plan.strategy == PlanStrategy.INDEX_UNION:
            entities = self._execute_index_union(plan)
        elif plan.strategy == PlanStrategy.INDEX_SCAN:
            entities = self._execute_index_scan(plan)
        else:  # FULL_SCAN
            entities = self._execute_full_scan(plan)

        # Apply post-filter if present
        if plan.post_filter is not None:
            entities = [e for e in entities if self._evaluate(plan.post_filter, e)]

        # Apply ordering
        if plan.order_by:
            entities = self._apply_order(entities, plan.order_by)

        # Apply limit/offset
        entities = self._apply_limit_offset(entities, plan.limit, plan.offset)

        return entities

    def _execute_function_call(self, plan: QueryPlan) -> List[Any]:
        """Execute a standalone function call."""
        func_call = plan.function_call
        if func_call is None:
            return []

        func_class = FunctionRegistry.get(func_call.name)
        if func_class is None:
            available = [f.name for f in FunctionRegistry.list_functions()]
            raise QueryExecutionError(
                f"Unknown function: {func_call.name}",
                function_name=func_call.name,
                available_functions=available
            )

        # Resolve arguments
        args = [self._resolve_literal(arg) for arg in func_call.args]
        kwargs = {k: self._resolve_literal(v) for k, v in func_call.kwargs}

        # Execute function
        func_instance = func_class()
        result = func_instance.execute(self.context, args, kwargs)

        # Ensure result is a list
        if not isinstance(result, list):
            result = [result]

        # Apply ordering and limit to function results
        if plan.order_by:
            result = self._apply_order(result, plan.order_by)
        result = self._apply_limit_offset(result, plan.limit, plan.offset)

        return result

    def _execute_index_intersect(self, plan: QueryPlan) -> List[Any]:
        """Execute using multiple index lookups with intersection."""
        if self.index_manager is None or self.store is None:
            # Fall back to full scan when no index manager
            return self._execute_full_scan(plan)

        # Perform all index lookups
        result_sets: List[Set[str]] = []
        for lookup in plan.index_lookups:
            ids = self._perform_index_lookup(plan.entity_type, lookup)
            result_sets.append(ids)

        # Intersect all result sets
        if not result_sets:
            return []

        result_ids = result_sets[0]
        for ids in result_sets[1:]:
            result_ids = result_ids & ids

        # Load entities
        return self._load_entities(result_ids)

    def _execute_index_union(self, plan: QueryPlan) -> List[Any]:
        """Execute using index union (OR optimization).

        Each branch's lookups are intersected, then all branches are unioned.
        """
        if self.index_manager is None or self.store is None:
            # Fall back to full scan when no index manager
            return self._execute_full_scan(plan)

        if not plan.union_branches:
            return []

        # Process each branch: intersect lookups within branch, union across branches
        all_result_ids: Set[str] = set()

        for branch_lookups in plan.union_branches:
            # Intersect all lookups in this branch
            branch_sets: List[Set[str]] = []
            for lookup in branch_lookups:
                ids = self._perform_index_lookup(plan.entity_type, lookup)
                branch_sets.append(ids)

            if not branch_sets:
                continue

            # Intersect within branch
            branch_ids = branch_sets[0]
            for ids in branch_sets[1:]:
                branch_ids = branch_ids & ids

            # Union with overall results
            all_result_ids = all_result_ids | branch_ids

        # Load entities
        return self._load_entities(all_result_ids)

    def _execute_index_scan(self, plan: QueryPlan) -> List[Any]:
        """Execute using single index lookup."""
        if self.index_manager is None or self.store is None:
            # Fall back to full scan when no index manager
            return self._execute_full_scan(plan)

        # Perform single index lookup
        if not plan.index_lookups:
            return []

        lookup = plan.index_lookups[0]
        result_ids = self._perform_index_lookup(plan.entity_type, lookup)

        # Load entities
        return self._load_entities(result_ids)

    def _execute_full_scan(self, plan: QueryPlan) -> List[Any]:
        """Execute using full entity scan."""
        if self.store is None:
            raise QueryNotImplementedError(
                "Full scan requires CDGStore",
                doc_reference="docs/design/cdg-query-language.md"
            )

        entity_type = plan.entity_type
        if entity_type is None:
            return []

        # Get ID prefix for entity type
        prefix = self._get_entity_prefix(entity_type)
        if prefix is None:
            raise QueryValidationError(
                f"Unknown entity type: {entity_type}",
                entity_type=entity_type,
                valid_entity_types=self._list_entity_types()
            )

        # Load all entities of this type
        entity_ids = self.store.list_by_prefix(prefix)
        return self._load_entities(set(entity_ids))

    def _perform_index_lookup(
        self,
        entity_type: str,
        lookup: IndexLookup
    ) -> Set[str]:
        """Perform an index lookup operation."""
        if self.index_manager is None:
            return set()

        if lookup.op == Op.EQ:
            # For btree fields, use btree lookup; for hash, use hash lookup
            if self.index_manager.is_btree_indexed(entity_type, lookup.field):
                # BTree can do equality lookup via lookup_range with same start/end
                return self.index_manager.lookup_range(
                    entity_type, lookup.field,
                    start_value=lookup.value,
                    end_value=lookup.value,
                    start_inclusive=True,
                    end_inclusive=True
                )
            return self.index_manager.lookup(entity_type, lookup.field, lookup.value)

        elif lookup.op == Op.IN:
            if isinstance(lookup.value, list):
                return self.index_manager.lookup_multi(entity_type, lookup.field, lookup.value)
            else:
                return self.index_manager.lookup(entity_type, lookup.field, lookup.value)

        elif lookup.op == Op.GT:
            # Range query: field > value (btree only)
            try:
                return self.index_manager.lookup_gt(entity_type, lookup.field, lookup.value)
            except ValueError:
                # Field doesn't have btree index, return empty (will be post-filtered)
                return set()

        elif lookup.op == Op.GTE:
            # Range query: field >= value (btree only)
            try:
                return self.index_manager.lookup_gte(entity_type, lookup.field, lookup.value)
            except ValueError:
                return set()

        elif lookup.op == Op.LT:
            # Range query: field < value (btree only)
            try:
                return self.index_manager.lookup_lt(entity_type, lookup.field, lookup.value)
            except ValueError:
                return set()

        elif lookup.op == Op.LTE:
            # Range query: field <= value (btree only)
            try:
                return self.index_manager.lookup_lte(entity_type, lookup.field, lookup.value)
            except ValueError:
                return set()

        else:
            # Unsupported operator for index lookup
            return set()

    def _load_entities(self, entity_ids: Set[str]) -> List[Any]:
        """Load entities by ID."""
        if self.store is None:
            return []

        entities = []
        for entity_id in entity_ids:
            entity = self.store.read(entity_id)
            if entity is not None:
                entities.append(entity)
        return entities

    def _get_entity_prefix(self, entity_type: str) -> Optional[str]:
        """Get the ID prefix for an entity type."""
        # Try schema registry first (if available)
        if self.schema_registry is not None:
            schema = self.schema_registry.get_schema(entity_type)
            if schema is not None:
                return schema.id_prefix

        # Fall back to entity_schemas module (works without registry)
        try:
            from cortical.got.entity_schemas import get_id_prefix
            return get_id_prefix(entity_type)
        except (ImportError, KeyError):
            return None

    def _list_entity_types(self) -> List[str]:
        """List all known entity types."""
        # Try schema registry first (if available)
        if self.schema_registry is not None:
            return list(self.schema_registry.list_schemas().keys())

        # Fall back to entity_schemas module (works without registry)
        try:
            from cortical.got.entity_schemas import list_entity_types
            return list_entity_types()
        except ImportError:
            # TODO(cdg-query): entity_schemas not available, return empty list
            # This should not happen in normal usage - entity_schemas is part of cortical.got
            return []

    def _evaluate(self, expr: Expression, entity: Any) -> bool:
        """Evaluate an expression against an entity."""
        if isinstance(expr, Comparison):
            return self._evaluate_comparison(expr, entity)
        elif isinstance(expr, AndExpr):
            return all(self._evaluate(child, entity) for child in expr.children)
        elif isinstance(expr, OrExpr):
            return any(self._evaluate(child, entity) for child in expr.children)
        elif isinstance(expr, NotExpr):
            return not self._evaluate(expr.child, entity)
        elif isinstance(expr, FunctionCall):
            # Function calls in filter context return boolean
            # TODO(cdg-query): Support filter functions that return boolean
            raise QueryNotImplementedError(
                f"Function calls in filter context not yet supported: {expr.name}",
                doc_reference="docs/design/cdg-query-language.md"
            )
        else:
            logger.warning(f"Unknown expression type in filter: {type(expr)}")
            return True

    def _evaluate_comparison(self, comp: Comparison, entity: Any) -> bool:
        """Evaluate a comparison against an entity."""
        field_name = comp.field.name
        entity_value = self._get_field_value(entity, field_name)
        compare_value = self._resolve_literal(comp.value)

        op = comp.op

        if op == Op.EQ:
            return entity_value == compare_value
        elif op == Op.NE:
            return entity_value != compare_value
        elif op == Op.GT:
            return entity_value is not None and entity_value > compare_value
        elif op == Op.LT:
            return entity_value is not None and entity_value < compare_value
        elif op == Op.GTE:
            return entity_value is not None and entity_value >= compare_value
        elif op == Op.LTE:
            return entity_value is not None and entity_value <= compare_value
        elif op == Op.IN:
            if isinstance(compare_value, list):
                return entity_value in compare_value
            return entity_value == compare_value
        elif op == Op.NOT_IN:
            if isinstance(compare_value, list):
                return entity_value not in compare_value
            return entity_value != compare_value
        elif op == Op.LIKE:
            if entity_value is None or compare_value is None:
                return False
            # Convert SQL LIKE pattern to fnmatch pattern
            pattern = str(compare_value).replace('%', '*').replace('_', '?')
            return fnmatch.fnmatch(str(entity_value), pattern)
        elif op == Op.NOT_LIKE:
            if entity_value is None or compare_value is None:
                return True
            pattern = str(compare_value).replace('%', '*').replace('_', '?')
            return not fnmatch.fnmatch(str(entity_value), pattern)
        elif op == Op.IS_NULL:
            return entity_value is None
        elif op == Op.IS_NOT_NULL:
            return entity_value is not None
        else:
            logger.warning(f"Unknown comparison operator: {op}")
            return False

    def _get_field_value(self, entity: Any, field_name: str) -> Any:
        """Get a field value from an entity."""
        # Try direct attribute access
        if hasattr(entity, field_name):
            return getattr(entity, field_name)

        # Try dict access
        if isinstance(entity, dict):
            return entity.get(field_name)

        # Try properties dict (common in CDG entities)
        if hasattr(entity, 'properties'):
            props = entity.properties
            if isinstance(props, dict):
                return props.get(field_name)

        # Try data dict
        if hasattr(entity, 'data'):
            data = entity.data
            if isinstance(data, dict):
                return data.get(field_name)

        return None

    def _resolve_literal(self, expr: Any) -> Any:
        """Resolve a Literal expression to its value."""
        if isinstance(expr, Literal):
            return expr.value
        return expr

    def _apply_order(self, entities: List[Any], order_by: tuple) -> List[Any]:
        """Apply ORDER BY to results."""
        field, desc = order_by

        def get_sort_key(entity):
            value = self._get_field_value(entity, field)
            # Handle None values - sort them last
            if value is None:
                return (1, '')
            return (0, value)

        return sorted(entities, key=get_sort_key, reverse=desc)

    def _apply_limit_offset(
        self,
        entities: List[Any],
        limit: Optional[int],
        offset: Optional[int]
    ) -> List[Any]:
        """Apply LIMIT and OFFSET to results."""
        if offset:
            entities = entities[offset:]
        if limit:
            entities = entities[:limit]
        return entities


def execute(
    plan: QueryPlan,
    store: Optional["CDGStore"] = None,
    index_manager: Optional["CDGIndexManager"] = None,
    schema_registry: Optional["SchemaRegistry"] = None,
    context: Optional[QueryContext] = None
) -> List[Any]:
    """Convenience function to execute a query plan."""
    executor = QueryExecutor(store, index_manager, schema_registry, context)
    return executor.execute(plan)
