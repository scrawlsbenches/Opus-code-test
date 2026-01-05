"""
Graph traversal functions for GoT query expressions.

These functions provide graph navigation capabilities for exploring
entity relationships in the Graph of Thought system.

AVAILABLE FUNCTIONS
-------------------
connected_to(entity_id):
    Returns all entities connected to the specified entity via any edge.

path(from_id, to_id, max_depth=10):
    Finds the shortest path between two entities.
    Returns list of entity IDs, or None if no path exists.

children(entity_id):
    Returns entities that directly depend on the specified entity.
    (Entities where from_id = entity_id in DEPENDS_ON edges)

parents(entity_id):
    Returns entities that the specified entity directly depends on.
    (Entities where to_id = entity_id in DEPENDS_ON edges)

descendants(entity_id, max_depth=None):
    Returns all entities reachable following dependency chains.
    Optional depth limit to prevent deep recursion.

ancestors(entity_id, max_depth=None):
    Returns all entities this entity transitively depends on.
    Optional depth limit to prevent deep recursion.

all_dependencies(entity_id):
    Returns all direct and transitive dependencies of an entity.
    Follows DEPENDS_ON edges to find complete dependency graph.
    Semantically equivalent to ancestors() but more explicit name.

orphan_nodes():
    Returns entities with no incoming or outgoing edges.
    Useful for finding isolated tasks or decisions.

cycle_detect(entity_id):
    Detects circular dependencies starting from an entity.
    Returns the cycle path if found, empty list if no cycle.
    Uses DFS with path tracking.

USAGE EXAMPLES
--------------
Find all connected entities:
    >>> connected_to('T-001')

Find path between tasks:
    >>> path('T-001', 'T-010')

Find dependencies:
    >>> children('S-001')  # Tasks in sprint
    >>> parents('T-005')   # What T-005 depends on

Find transitive relationships:
    >>> descendants('T-001', max_depth=3)  # All tasks blocked by T-001
    >>> ancestors('T-010')  # Full dependency chain
    >>> all_dependencies('T-010')  # Same as ancestors, explicit name

Detect cycles:
    >>> cycle_detect('T-001')  # Returns cycle path or []

Find isolated work:
    >>> orphan_nodes()  # Tasks with no relationships
"""

from typing import Any, Dict, List, Optional

from cortical.got.expression.registry import FunctionRegistry, QueryFunction, FunctionSignature
from cortical.got.path_finder import PathFinder
from cortical.got.query_builder import Query


@FunctionRegistry.register("connected_to")
class ConnectedTo(QueryFunction):
    """Find entities connected to a given entity via any edge."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="connected_to",
            description="Find all entities connected to the specified entity",
            required_args=["entity_id"],
            optional_args={},
            returns="List of entities connected via any edge type"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute connected_to function.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of entities connected to the specified entity
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

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


@FunctionRegistry.register("path")
class Path(QueryFunction):
    """Find shortest path between two entities."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="path",
            description="Find the shortest path between two entities",
            required_args=["from_id", "to_id"],
            optional_args={"max_depth": 10},
            returns="List of entity IDs representing the path, or None if no path exists"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        Execute path function.

        Args:
            manager: GoTManager instance
            args: Positional arguments [from_id, to_id, max_depth]
            kwargs: Keyword arguments {from_id: str, to_id: str, max_depth: int}

        Returns:
            List of entity IDs in path, or None if no path exists
        """
        # Parse arguments
        if len(args) >= 2:
            from_id = args[0]
            to_id = args[1]
            max_depth = args[2] if len(args) > 2 else kwargs.get('max_depth', 10)
        else:
            from_id = kwargs.get('from_id')
            to_id = kwargs.get('to_id')
            max_depth = kwargs.get('max_depth', 10)

        if not from_id or not to_id:
            raise ValueError("from_id and to_id are required")

        # Use PathFinder for BFS shortest path
        finder = PathFinder(manager)
        if max_depth is not None:
            finder = finder.max_length(max_depth)

        return finder.shortest_path(from_id, to_id)


@FunctionRegistry.register("children")
class Children(QueryFunction):
    """Find direct children (entities that depend on this one)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="children",
            description="Find entities that directly depend on the specified entity",
            required_args=["entity_id"],
            optional_args={},
            returns="List of entities where from_id = entity_id in DEPENDS_ON edges"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute children function.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of child entities
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Get edges where this entity is the source (points to children)
        edges = manager.list_edges()
        child_ids = set()

        for edge in edges:
            if edge.source_id == entity_id and edge.edge_type == "DEPENDS_ON":
                child_ids.add(edge.target_id)

        # Get all tasks and filter by child IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in child_ids]


@FunctionRegistry.register("parents")
class Parents(QueryFunction):
    """Find direct parents (entities this one depends on)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="parents",
            description="Find entities that the specified entity directly depends on",
            required_args=["entity_id"],
            optional_args={},
            returns="List of entities where to_id = entity_id in DEPENDS_ON edges"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute parents function.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of parent entities
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Get edges where this entity is the target (depends on parents)
        edges = manager.list_edges()
        parent_ids = set()

        for edge in edges:
            if edge.target_id == entity_id and edge.edge_type == "DEPENDS_ON":
                parent_ids.add(edge.source_id)

        # Get all tasks and filter by parent IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in parent_ids]


@FunctionRegistry.register("descendants")
class Descendants(QueryFunction):
    """Find all descendants (transitive children)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="descendants",
            description="Find all entities reachable following dependency chains",
            required_args=["entity_id"],
            optional_args={"max_depth": None},
            returns="List of all descendant entities (recursive children)"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute descendants function.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id, max_depth]
            kwargs: Keyword arguments {entity_id: str, max_depth: int}

        Returns:
            List of all descendant entities
        """
        # Parse arguments
        if args:
            entity_id = args[0]
            max_depth = args[1] if len(args) > 1 else kwargs.get('max_depth')
        else:
            entity_id = kwargs.get('entity_id')
            max_depth = kwargs.get('max_depth')

        if not entity_id:
            raise ValueError("entity_id is required")

        # Use PathFinder to get reachable nodes following DEPENDS_ON edges
        finder = PathFinder(manager).via_edges("DEPENDS_ON")
        if max_depth is not None:
            finder = finder.max_length(max_depth)

        # Get all reachable nodes
        descendant_ids = finder.reachable_from(entity_id)

        # Remove the entity itself from results
        descendant_ids.discard(entity_id)

        # Get all tasks and filter by descendant IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in descendant_ids]


@FunctionRegistry.register("ancestors")
class Ancestors(QueryFunction):
    """Find all ancestors (transitive parents)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="ancestors",
            description="Find all entities this entity transitively depends on",
            required_args=["entity_id"],
            optional_args={"max_depth": None},
            returns="List of all ancestor entities (recursive parents)"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute ancestors function.

        For ancestors, we need to traverse edges in reverse direction:
        follow edges where the current entity is the target (to_id).

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id, max_depth]
            kwargs: Keyword arguments {entity_id: str, max_depth: int}

        Returns:
            List of all ancestor entities
        """
        # Parse arguments
        if args:
            entity_id = args[0]
            max_depth = args[1] if len(args) > 1 else kwargs.get('max_depth')
        else:
            entity_id = kwargs.get('entity_id')
            max_depth = kwargs.get('max_depth')

        if not entity_id:
            raise ValueError("entity_id is required")

        # Build reverse adjacency (follow edges backwards)
        edges = manager.list_edges()
        reverse_adjacency = {}

        for edge in edges:
            if edge.edge_type == "DEPENDS_ON":
                # Reverse: if A->B, then from B we can reach A
                if edge.target_id not in reverse_adjacency:
                    reverse_adjacency[edge.target_id] = []
                reverse_adjacency[edge.target_id].append(edge.source_id)

        # BFS/DFS to find ancestors
        visited = set()
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

            # Add parents
            for parent in reverse_adjacency.get(current, []):
                if parent not in visited:
                    queue.append(parent)
                    depth_map[parent] = current_depth + 1

        # Remove the entity itself from results
        visited.discard(entity_id)

        # Get all tasks and filter by ancestor IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in visited]


@FunctionRegistry.register("orphan_nodes")
class OrphanNodes(QueryFunction):
    """Find entities with no incoming or outgoing edges."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="orphan_nodes",
            description="Find entities with no incoming or outgoing edges",
            required_args=[],
            optional_args={},
            returns="List of isolated entities (no connections)"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute orphan_nodes function.

        Args:
            manager: GoTManager instance
            args: Not used
            kwargs: Not used

        Returns:
            List of entities with no edges
        """
        # Get all edges
        edges = manager.list_edges()

        # Build set of all entity IDs that have connections
        connected_ids = set()
        for edge in edges:
            connected_ids.add(edge.source_id)
            connected_ids.add(edge.target_id)

        # Get all tasks and filter for orphans
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id not in connected_ids]


@FunctionRegistry.register("blockers")
class Blockers(QueryFunction):
    """Find tasks that block the given task."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="blockers",
            description="Find tasks that block the specified task",
            required_args=["task_id"],
            optional_args={},
            returns="List of tasks with BLOCKS edge TO the specified task"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute blockers function.

        Finds tasks that have a BLOCKS edge pointing TO the given task.
        These are the tasks that are blocking the specified task from proceeding.

        Args:
            manager: GoTManager instance
            args: Positional arguments [task_id]
            kwargs: Keyword arguments {task_id: str}

        Returns:
            List of tasks blocking the specified task
        """
        task_id = args[0] if args else kwargs.get('task_id')
        if not task_id:
            raise ValueError("task_id is required")

        # Get edges where target is the given task and edge type is BLOCKS
        edges = manager.list_edges()
        blocker_ids = set()

        for edge in edges:
            if edge.target_id == task_id and edge.edge_type == "BLOCKS":
                blocker_ids.add(edge.source_id)

        # Get all tasks and filter by blocker IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in blocker_ids]


@FunctionRegistry.register("dependents")
class Dependents(QueryFunction):
    """Find tasks that depend on the given task."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="dependents",
            description="Find tasks that depend on the specified task",
            required_args=["task_id"],
            optional_args={},
            returns="List of tasks with DEPENDS_ON edge FROM the specified task"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute dependents function.

        Finds tasks that have a DEPENDS_ON edge pointing TO them FROM the given task.
        These are the tasks that depend on the specified task completing.

        Semantically equivalent to children() function, but named for clarity in queries.

        Args:
            manager: GoTManager instance
            args: Positional arguments [task_id]
            kwargs: Keyword arguments {task_id: str}

        Returns:
            List of tasks that depend on the specified task
        """
        task_id = args[0] if args else kwargs.get('task_id')
        if not task_id:
            raise ValueError("task_id is required")

        # Get edges where source is the given task and edge type is DEPENDS_ON
        # If A -> DEPENDS_ON -> B, then B depends on A
        edges = manager.list_edges()
        dependent_ids = set()

        for edge in edges:
            if edge.source_id == task_id and edge.edge_type == "DEPENDS_ON":
                dependent_ids.add(edge.target_id)

        # Get all tasks and filter by dependent IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in dependent_ids]


@FunctionRegistry.register("all_dependencies")
class AllDependencies(QueryFunction):
    """Find all direct and transitive dependencies of an entity."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="all_dependencies",
            description="Find all direct and transitive dependencies (complete dependency graph)",
            required_args=["entity_id"],
            optional_args={},
            returns="List of all entities this entity depends on (direct and transitive)"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute all_dependencies function.

        This is semantically equivalent to ancestors() - finds all entities
        that the given entity depends on, following DEPENDS_ON edges backwards.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of all dependency entities (direct and transitive)
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Build reverse adjacency (follow DEPENDS_ON edges backwards)
        edges = manager.list_edges()
        reverse_adjacency = {}

        for edge in edges:
            if edge.edge_type == "DEPENDS_ON":
                # If A -> DEPENDS_ON -> B, then from B we can reach dependency A
                if edge.target_id not in reverse_adjacency:
                    reverse_adjacency[edge.target_id] = []
                reverse_adjacency[edge.target_id].append(edge.source_id)

        # BFS to find all dependencies
        visited = set()
        queue = [entity_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Add dependencies (parents in the dependency graph)
            for dependency in reverse_adjacency.get(current, []):
                if dependency not in visited:
                    queue.append(dependency)

        # Remove the entity itself from results
        visited.discard(entity_id)

        # Get all tasks and filter by dependency IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in visited]


@FunctionRegistry.register("cycle_detect")
class CycleDetect(QueryFunction):
    """Detect circular dependencies starting from an entity."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="cycle_detect",
            description="Detect circular dependencies, returns cycle path if found",
            required_args=["entity_id"],
            optional_args={},
            returns="List of entity IDs forming the cycle, or empty list if no cycle"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Execute cycle_detect function.

        Uses DFS with path tracking to detect cycles in the dependency graph.
        If a cycle is found, returns the path forming the cycle.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of entity IDs forming the cycle path, or empty list if no cycle
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Build adjacency list for DEPENDS_ON edges
        edges = manager.list_edges()
        adjacency = {}

        for edge in edges:
            if edge.edge_type == "DEPENDS_ON":
                if edge.source_id not in adjacency:
                    adjacency[edge.source_id] = []
                adjacency[edge.source_id].append(edge.target_id)

        # DFS with path tracking to detect cycles
        def dfs_cycle(node: str, path: List[str], visited: set) -> Optional[List[str]]:
            """
            DFS to detect cycle.

            Args:
                node: Current node
                path: Current path from start
                visited: Set of nodes in current path

            Returns:
                Cycle path if found, None otherwise
            """
            if node in visited:
                # Found a cycle - return the cycle path
                cycle_start_idx = path.index(node)
                return path[cycle_start_idx:] + [node]

            visited.add(node)
            path.append(node)

            # Explore neighbors
            for neighbor in adjacency.get(node, []):
                result = dfs_cycle(neighbor, path, visited)
                if result:
                    return result

            # Backtrack
            path.pop()
            visited.remove(node)
            return None

        # Start DFS from the given entity
        result = dfs_cycle(entity_id, [], set())
        return result if result else []
