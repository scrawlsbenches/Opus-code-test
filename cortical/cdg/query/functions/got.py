"""
GoT-specific query functions for CDG.

These functions provide Graph of Thought operations that require
GoTManager access. They are registered with the CDG FunctionRegistry
and use context.require_extension('got_manager') to access GoTManager.

Graph Traversal Functions:
- connected_to(entity_id): All entities connected via any edge
- path(from_id, to_id, max_depth): Shortest path between entities
- children(entity_id): Entities that depend on this one
- parents(entity_id): Entities this one depends on
- descendants(entity_id, max_depth): Transitive dependents
- ancestors(entity_id, max_depth): Transitive dependencies
- orphan_nodes(): Entities with no edges
- blockers(task_id): Tasks blocking this task
- dependents(task_id): Tasks depending on this task
- all_dependencies(entity_id): All transitive dependencies
- cycle_detect(entity_id): Detect circular dependencies

Filter Functions:
- recent(days): Entities created within N days
- stale(days): Entities older than N days
- has_edge(edge_type): Entities with specific edge type
- blocked(): Tasks blocked by others
- blocking(): Tasks blocking others
- in_sprint(sprint_id): Tasks in a sprint
- unassigned(): Tasks without assignee
- overdue(): Tasks past due date

Aggregate Functions:
- aggregate(field, operation): Group by field and aggregate

See: docs/design/cdg-query-language.md
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..registry import FunctionRegistry, FunctionSignature, QueryFunction, QueryContext

if TYPE_CHECKING:
    from cortical.got.api import GoTManager


def _get_manager(context: QueryContext) -> "GoTManager":
    """Get GoTManager from context, raising if not available."""
    return context.require_extension('got_manager')


# ===========================================================================
# Graph Traversal Functions
# ===========================================================================

@FunctionRegistry.register('connected_to')
class ConnectedToFunction(QueryFunction):
    """Find entities connected to a given entity via any edge."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='connected_to',
            description='Find all entities connected to the specified entity',
            required_args=['entity_id'],
            optional_args={},
            returns='List of entities connected via any edge type',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)

        # Get all edges
        edges = manager.list_edges()

        # Find all connected entity IDs
        connected_ids = set()
        for edge in edges:
            if edge.source_id == entity_id:
                connected_ids.add(edge.target_id)
            if edge.target_id == entity_id:
                connected_ids.add(edge.source_id)

        # Get all tasks and filter by connected IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in connected_ids]


@FunctionRegistry.register('path')
class PathFunction(QueryFunction):
    """Find shortest path between two entities."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='path',
            description='Find the shortest path between two entities',
            required_args=['from_id', 'to_id'],
            optional_args={'max_depth': None},
            returns='List of entity IDs representing the path, or None',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[List[str]]:
        # Parse arguments
        if len(args) >= 2:
            from_id = args[0]
            to_id = args[1]
            max_depth = args[2] if len(args) > 2 else kwargs.get('max_depth')
        else:
            from_id = kwargs.get('from_id')
            to_id = kwargs.get('to_id')
            max_depth = kwargs.get('max_depth')

        if not from_id or not to_id:
            raise ValueError("from_id and to_id are required")

        manager = _get_manager(context)

        # Use PathFinder for BFS shortest path
        from cortical.got.path_finder import PathFinder
        finder = PathFinder(manager)
        if max_depth is not None:
            finder = finder.max_length(max_depth)

        return finder.shortest_path(from_id, to_id)


@FunctionRegistry.register('children')
class ChildrenFunction(QueryFunction):
    """Find direct children (entities that depend on this one)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='children',
            description='Find entities that directly depend on the specified entity',
            required_args=['entity_id'],
            optional_args={},
            returns='List of entities where this entity is their dependency',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        # Edge semantics: A -> DEPENDS_ON -> B means "A depends on B"
        # children(B) returns [A] - entities that depend on B
        edges = manager.list_edges()
        child_ids = set()

        for edge in edges:
            if edge.target_id == entity_id and edge.edge_type == EdgeTypes.DEPENDS_ON:
                child_ids.add(edge.source_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in child_ids]


@FunctionRegistry.register('parents')
class ParentsFunction(QueryFunction):
    """Find direct parents (entities this one depends on)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='parents',
            description='Find entities that the specified entity directly depends on',
            required_args=['entity_id'],
            optional_args={},
            returns='List of entities this entity depends on',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        # Edge semantics: A -> DEPENDS_ON -> B means "A depends on B"
        # parents(A) returns [B] - what A depends on
        edges = manager.list_edges()
        parent_ids = set()

        for edge in edges:
            if edge.source_id == entity_id and edge.edge_type == EdgeTypes.DEPENDS_ON:
                parent_ids.add(edge.target_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in parent_ids]


@FunctionRegistry.register('descendants')
class DescendantsFunction(QueryFunction):
    """Find all descendants (entities that transitively depend on this one)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='descendants',
            description='Find all entities that transitively depend on the specified entity',
            required_args=['entity_id'],
            optional_args={'max_depth': None},
            returns='List of all descendant entities',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        # Parse arguments
        if args:
            entity_id = args[0]
            max_depth = args[1] if len(args) > 1 else kwargs.get('max_depth')
        else:
            entity_id = kwargs.get('entity_id')
            max_depth = kwargs.get('max_depth')

        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        # Build reverse adjacency (from target to source)
        edges = manager.list_edges()
        reverse_adjacency: Dict[str, List[str]] = {}

        for edge in edges:
            if edge.edge_type == EdgeTypes.DEPENDS_ON:
                if edge.target_id not in reverse_adjacency:
                    reverse_adjacency[edge.target_id] = []
                reverse_adjacency[edge.target_id].append(edge.source_id)

        # BFS to find descendants
        visited: set = set()
        queue = [entity_id]
        depth_map = {entity_id: 0}

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            current_depth = depth_map[current]
            if max_depth is not None and current_depth >= max_depth:
                continue

            for dependent in reverse_adjacency.get(current, []):
                if dependent not in visited:
                    queue.append(dependent)
                    depth_map[dependent] = current_depth + 1

        visited.discard(entity_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in visited]


@FunctionRegistry.register('ancestors')
class AncestorsFunction(QueryFunction):
    """Find all ancestors (transitive dependencies)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='ancestors',
            description='Find all entities this entity transitively depends on',
            required_args=['entity_id'],
            optional_args={'max_depth': None},
            returns='List of all ancestor entities',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        # Parse arguments
        if args:
            entity_id = args[0]
            max_depth = args[1] if len(args) > 1 else kwargs.get('max_depth')
        else:
            entity_id = kwargs.get('entity_id')
            max_depth = kwargs.get('max_depth')

        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        # Build forward adjacency (from source to target)
        edges = manager.list_edges()
        forward_adjacency: Dict[str, List[str]] = {}

        for edge in edges:
            if edge.edge_type == EdgeTypes.DEPENDS_ON:
                if edge.source_id not in forward_adjacency:
                    forward_adjacency[edge.source_id] = []
                forward_adjacency[edge.source_id].append(edge.target_id)

        # BFS to find ancestors
        visited: set = set()
        queue = [entity_id]
        depth_map = {entity_id: 0}

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            current_depth = depth_map[current]
            if max_depth is not None and current_depth >= max_depth:
                continue

            for dependency in forward_adjacency.get(current, []):
                if dependency not in visited:
                    queue.append(dependency)
                    depth_map[dependency] = current_depth + 1

        visited.discard(entity_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in visited]


@FunctionRegistry.register('orphan_nodes')
class OrphanNodesFunction(QueryFunction):
    """Find entities with no incoming or outgoing edges."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='orphan_nodes',
            description='Find entities with no incoming or outgoing edges',
            required_args=[],
            optional_args={},
            returns='List of isolated entities',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        manager = _get_manager(context)

        edges = manager.list_edges()
        connected_ids = set()
        for edge in edges:
            connected_ids.add(edge.source_id)
            connected_ids.add(edge.target_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id not in connected_ids]


@FunctionRegistry.register('blockers')
class BlockersFunction(QueryFunction):
    """Find tasks that block the given task."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='blockers',
            description='Find tasks that block the specified task',
            required_args=['task_id'],
            optional_args={},
            returns='List of tasks with BLOCKS edge to the specified task',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        task_id = args[0] if args else kwargs.get('task_id')
        if not task_id:
            raise ValueError("task_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        edges = manager.list_edges()
        blocker_ids = set()

        for edge in edges:
            if edge.target_id == task_id and edge.edge_type == EdgeTypes.BLOCKS:
                blocker_ids.add(edge.source_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in blocker_ids]


@FunctionRegistry.register('dependents')
class DependentsFunction(QueryFunction):
    """Find tasks that depend on the given task."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='dependents',
            description='Find tasks that depend on the specified task',
            required_args=['task_id'],
            optional_args={},
            returns='List of tasks with DEPENDS_ON edge to the specified task',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        task_id = args[0] if args else kwargs.get('task_id')
        if not task_id:
            raise ValueError("task_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        edges = manager.list_edges()
        dependent_ids = set()

        for edge in edges:
            if edge.target_id == task_id and edge.edge_type == EdgeTypes.DEPENDS_ON:
                dependent_ids.add(edge.source_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in dependent_ids]


@FunctionRegistry.register('all_dependencies')
class AllDependenciesFunction(QueryFunction):
    """Find all direct and transitive dependencies of an entity."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='all_dependencies',
            description='Find all direct and transitive dependencies',
            required_args=['entity_id'],
            optional_args={},
            returns='List of all entities this entity depends on',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        # Build forward adjacency
        edges = manager.list_edges()
        forward_adjacency: Dict[str, List[str]] = {}

        for edge in edges:
            if edge.edge_type == EdgeTypes.DEPENDS_ON:
                if edge.source_id not in forward_adjacency:
                    forward_adjacency[edge.source_id] = []
                forward_adjacency[edge.source_id].append(edge.target_id)

        # BFS to find all dependencies
        visited: set = set()
        queue = [entity_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for dependency in forward_adjacency.get(current, []):
                if dependency not in visited:
                    queue.append(dependency)

        visited.discard(entity_id)

        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in visited]


@FunctionRegistry.register('cycle_detect')
class CycleDetectFunction(QueryFunction):
    """Detect circular dependencies starting from an entity."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='cycle_detect',
            description='Detect circular dependencies, returns cycle path if found',
            required_args=['entity_id'],
            optional_args={},
            returns='List of entity IDs forming the cycle, or empty list',
            category='graph'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        # Build adjacency list
        edges = manager.list_edges()
        adjacency: Dict[str, List[str]] = {}

        for edge in edges:
            if edge.edge_type == EdgeTypes.DEPENDS_ON:
                if edge.source_id not in adjacency:
                    adjacency[edge.source_id] = []
                adjacency[edge.source_id].append(edge.target_id)

        # DFS with path tracking
        def dfs_cycle(node: str, path: List[str], visited: set) -> Optional[List[str]]:
            if node in visited:
                cycle_start_idx = path.index(node)
                return path[cycle_start_idx:] + [node]

            visited.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                result = dfs_cycle(neighbor, path, visited)
                if result:
                    return result

            path.pop()
            visited.remove(node)
            return None

        result = dfs_cycle(entity_id, [], set())
        return result if result else []


# ===========================================================================
# Filter Functions
# ===========================================================================

@FunctionRegistry.register('recent')
class RecentFunction(QueryFunction):
    """Filter entities created within N days."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='recent',
            description='Find entities created within N days',
            required_args=[],
            optional_args={'days': 7},
            returns='List of recent entities',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        days = args[0] if args else kwargs.get('days', 7)
        if not isinstance(days, (int, float)) or days < 0:
            raise ValueError(f"days must be a non-negative number, got {days}")

        manager = _get_manager(context)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        tasks = manager.list_all_tasks()
        return [t for t in tasks if t.created_at >= cutoff_str]


@FunctionRegistry.register('stale')
class StaleFunction(QueryFunction):
    """Filter entities older than N days."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='stale',
            description='Find entities created more than N days ago',
            required_args=[],
            optional_args={'days': 30},
            returns='List of old entities',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        days = args[0] if args else kwargs.get('days', 30)
        if not isinstance(days, (int, float)) or days < 0:
            raise ValueError(f"days must be a non-negative number, got {days}")

        manager = _get_manager(context)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        tasks = manager.list_all_tasks()
        return [t for t in tasks if t.created_at < cutoff_str]


@FunctionRegistry.register('has_edge')
class HasEdgeFunction(QueryFunction):
    """Filter entities that have at least one edge of the given type."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='has_edge',
            description='Find entities with at least one edge of given type',
            required_args=['edge_type'],
            optional_args={},
            returns='List of entities with the edge type',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        if not args and 'edge_type' not in kwargs:
            raise ValueError("has_edge requires edge_type argument")

        edge_type = args[0] if args else kwargs.get('edge_type')
        if not isinstance(edge_type, str):
            raise ValueError(f"edge_type must be a string, got {type(edge_type)}")

        manager = _get_manager(context)

        edges = manager.list_edges()
        entity_ids = set()

        for edge in edges:
            if edge.edge_type == edge_type:
                entity_ids.add(edge.source_id)
                entity_ids.add(edge.target_id)

        # Load entities (supports any entity type via prefix)
        results = []
        for entity_id in entity_ids:
            if entity_id.startswith('T-'):
                task = manager.get_task(entity_id)
                if task is not None:
                    results.append(task)
            # TODO(cdg-query): Add support for other entity types via CDGStore

        return results


@FunctionRegistry.register('blocked')
class BlockedFunction(QueryFunction):
    """Filter tasks that are blocked by other tasks."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='blocked',
            description='Find tasks that are blocked by other tasks',
            required_args=[],
            optional_args={},
            returns='List of blocked tasks',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        manager = _get_manager(context)
        return manager.query_api.get_blocked_tasks()


@FunctionRegistry.register('blocking')
class BlockingFunction(QueryFunction):
    """Filter tasks that are blocking other tasks."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='blocking',
            description='Find tasks that are blocking other tasks',
            required_args=[],
            optional_args={},
            returns='List of tasks blocking others',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        manager = _get_manager(context)
        from cortical.got.types import EdgeTypes

        edges = manager.list_edges()
        blocker_ids = set()

        for edge in edges:
            if edge.edge_type == EdgeTypes.BLOCKS:
                blocked_task = manager.get_task(edge.target_id)
                if blocked_task is not None and blocked_task.status != 'completed':
                    blocker_ids.add(edge.source_id)

        results = []
        for blocker_id in blocker_ids:
            task = manager.get_task(blocker_id)
            if task is not None:
                results.append(task)

        return results


@FunctionRegistry.register('in_sprint')
class InSprintFunction(QueryFunction):
    """Filter tasks that are in a specific sprint."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='in_sprint',
            description='Find tasks in a specific sprint',
            required_args=['sprint_id'],
            optional_args={},
            returns='List of tasks in the sprint',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        if not args and 'sprint_id' not in kwargs:
            raise ValueError("in_sprint requires sprint_id argument")

        sprint_id = args[0] if args else kwargs.get('sprint_id')
        if not isinstance(sprint_id, str):
            raise ValueError(f"sprint_id must be a string, got {type(sprint_id)}")

        manager = _get_manager(context)
        return manager.query_api.get_sprint_tasks(sprint_id)


@FunctionRegistry.register('unassigned')
class UnassignedFunction(QueryFunction):
    """Filter tasks with no assignee."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='unassigned',
            description='Find tasks with no assignee',
            required_args=[],
            optional_args={},
            returns='List of unassigned tasks',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        manager = _get_manager(context)

        tasks = manager.list_all_tasks()
        results = []

        for task in tasks:
            assignee_prop = task.properties.get('assignee')
            assignee_meta = task.metadata.get('assignee')
            if not assignee_prop and not assignee_meta:
                results.append(task)

        return results


@FunctionRegistry.register('overdue')
class OverdueFunction(QueryFunction):
    """Filter tasks past their due date."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='overdue',
            description='Find tasks past their due date',
            required_args=[],
            optional_args={},
            returns='List of overdue tasks',
            category='filter'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        manager = _get_manager(context)

        now = datetime.now(timezone.utc).isoformat()
        tasks = manager.list_all_tasks()
        results = []

        for task in tasks:
            if task.status == 'completed':
                continue

            due_date = task.properties.get('due_date') or task.metadata.get('due_date')
            if due_date and isinstance(due_date, str) and due_date < now:
                results.append(task)

        return results


# ===========================================================================
# Aggregate Functions
# ===========================================================================

@FunctionRegistry.register('aggregate')
class AggregateFunction(QueryFunction):
    """Aggregate entities by a field."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='aggregate',
            description='Count or group entities by a field',
            required_args=['field'],
            optional_args={'operation': 'count'},
            returns='Dict mapping field values to counts/results',
            category='aggregate'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Dict[Any, int]:
        # Parse arguments
        if args:
            field = args[0]
            operation = args[1] if len(args) > 1 else kwargs.get('operation', 'count')
        else:
            field = kwargs.get('field')
            operation = kwargs.get('operation', 'count')

        if not field:
            raise ValueError("field is required")

        manager = _get_manager(context)

        if operation == 'count':
            from cortical.got.query_builder import Query
            result = Query(manager).tasks().group_by(field).count().execute()
            return result if isinstance(result, dict) else {}

        raise ValueError(f"Unsupported aggregation operation: {operation}")
