"""
Graph traversal functions for GoT query expressions.

These functions provide graph navigation capabilities for exploring
entity relationships in the Graph of Thought system.

AVAILABLE FUNCTIONS
-------------------
connected_to(entity_id):
    Returns all entities connected to the specified entity via any edge.

path(from_id, to_id, max_depth=None):
    Finds the shortest path between two entities.
    Returns list of entity IDs, or None if no path exists.
    No artificial depth limit by default (design principle: no hardcoded magic numbers).

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
            optional_args={"max_depth": None},  # No artificial limit per design principle
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

        Design principle: No hardcoded magic numbers. Default to unlimited
        traversal; let explicit limits be opt-in. If a query runs forever,
        the developer stops it manually.

        Args:
            manager: GoTManager instance
            args: Positional arguments [from_id, to_id, max_depth]
            kwargs: Keyword arguments {from_id: str, to_id: str, max_depth: int}

        Returns:
            List of entity IDs in path, or None if no path exists
        """
        # Parse arguments - default to None (unlimited) per design principle
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

        Edge semantics: A → DEPENDS_ON → B means "A depends on B"
        children(B) returns [A] - entities that depend on B.
        These are entities where B is the TARGET of their DEPENDS_ON edge.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of child entities (entities that depend on this one)
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Get edges where this entity is the TARGET (things depend on us)
        # If A → DEPENDS_ON → B, then children(B) returns A
        edges = manager.list_edges()
        child_ids = set()

        for edge in edges:
            if edge.target_id == entity_id and edge.edge_type == "DEPENDS_ON":
                child_ids.add(edge.source_id)

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
            returns="List of entities this entity depends on (direct dependencies)"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute parents function.

        Edge semantics: A → DEPENDS_ON → B means "A depends on B"
        parents(A) returns [B] - what A depends on.
        These are entities that are the TARGET of A's DEPENDS_ON edges.

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            List of parent entities (what this entity depends on)
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Get edges where this entity is the SOURCE (we depend on targets)
        # If A → DEPENDS_ON → B, then parents(A) returns B
        edges = manager.list_edges()
        parent_ids = set()

        for edge in edges:
            if edge.source_id == entity_id and edge.edge_type == "DEPENDS_ON":
                parent_ids.add(edge.target_id)

        # Get all tasks and filter by parent IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in parent_ids]


@FunctionRegistry.register("descendants")
class Descendants(QueryFunction):
    """Find all descendants (entities that depend on this one, transitively)."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="descendants",
            description="Find all entities that (transitively) depend on the specified entity",
            required_args=["entity_id"],
            optional_args={"max_depth": None},
            returns="List of all descendant entities (things that depend on us)"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute descendants function.

        Edge semantics: A → DEPENDS_ON → B means "A depends on B"
        descendants(B) returns all entities that (transitively) depend on B.

        We traverse REVERSE through edges where we are the TARGET,
        following to the SOURCE (things that depend on us).

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id, max_depth]
            kwargs: Keyword arguments {entity_id: str, max_depth: int}

        Returns:
            List of all descendant entities (what depends on us)
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

        # Build REVERSE adjacency (from target to source)
        # If A → DEPENDS_ON → B, then from B we can find A (things that depend on B)
        edges = manager.list_edges()
        reverse_adjacency = {}

        for edge in edges:
            if edge.edge_type == "DEPENDS_ON":
                if edge.target_id not in reverse_adjacency:
                    reverse_adjacency[edge.target_id] = []
                reverse_adjacency[edge.target_id].append(edge.source_id)

        # BFS to find descendants (what depends on us)
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

            # Add dependents (sources of edges where we're the target)
            for dependent in reverse_adjacency.get(current, []):
                if dependent not in visited:
                    queue.append(dependent)
                    depth_map[dependent] = current_depth + 1

        # Remove the entity itself from results
        visited.discard(entity_id)

        # Get all tasks and filter by descendant IDs
        all_entities = manager.list_all_tasks()
        return [e for e in all_entities if e.id in visited]


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

        Edge semantics: A → DEPENDS_ON → B means "A depends on B"
        ancestors(A) returns all entities A (transitively) depends on.

        We traverse FORWARD through edges where we are the SOURCE,
        following to the TARGET (our dependencies).

        Args:
            manager: GoTManager instance
            args: Positional arguments [entity_id, max_depth]
            kwargs: Keyword arguments {entity_id: str, max_depth: int}

        Returns:
            List of all ancestor entities (what we depend on)
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

        # Build FORWARD adjacency (from source to target)
        # If A → DEPENDS_ON → B, then from A we can reach B (our dependency)
        edges = manager.list_edges()
        forward_adjacency = {}

        for edge in edges:
            if edge.edge_type == "DEPENDS_ON":
                if edge.source_id not in forward_adjacency:
                    forward_adjacency[edge.source_id] = []
                forward_adjacency[edge.source_id].append(edge.target_id)

        # BFS to find ancestors (what we depend on)
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

            # Add dependencies (targets of our DEPENDS_ON edges)
            for dependency in forward_adjacency.get(current, []):
                if dependency not in visited:
                    queue.append(dependency)
                    depth_map[dependency] = current_depth + 1

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
            returns="List of tasks with DEPENDS_ON edge TO the specified task"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute dependents function.

        Finds tasks that depend on the given task (tasks that have DEPENDS_ON
        edges pointing TO the given task).

        Edge semantics: A → DEPENDS_ON → B means "A depends on B"
        So dependents(B) returns [A] - all tasks that depend on B.

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

        # Get edges where target is the given task and edge type is DEPENDS_ON
        # If A -> DEPENDS_ON -> B, then A depends on B
        # So dependents(B) returns tasks where B is the target
        edges = manager.list_edges()
        dependent_ids = set()

        for edge in edges:
            if edge.target_id == task_id and edge.edge_type == "DEPENDS_ON":
                dependent_ids.add(edge.source_id)

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

        Edge semantics: A → DEPENDS_ON → B means "A depends on B"
        all_dependencies(A) returns all entities A (transitively) depends on.

        This is semantically equivalent to ancestors() - finds all entities
        that the given entity depends on, following DEPENDS_ON edges forward
        from source to target.

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

        # Build FORWARD adjacency (from source to target)
        # If A → DEPENDS_ON → B, then from A we can reach B (our dependency)
        edges = manager.list_edges()
        forward_adjacency = {}

        for edge in edges:
            if edge.edge_type == "DEPENDS_ON":
                if edge.source_id not in forward_adjacency:
                    forward_adjacency[edge.source_id] = []
                forward_adjacency[edge.source_id].append(edge.target_id)

        # BFS to find all dependencies
        visited = set()
        queue = [entity_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Add dependencies (targets of our DEPENDS_ON edges)
            for dependency in forward_adjacency.get(current, []):
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


@FunctionRegistry.register("exists")
class Exists(QueryFunction):
    """
    Check if an entity exists in the Graph of Thought.

    Design Reference: docs/design/got-query-audit-and-design.md T-013
    specifies exists(entity_id) -> bool

    Usage:
        exists('T-001') -> True if task exists
        exists('T-NONEXISTENT') -> False
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="exists",
            description="Check if an entity exists by ID",
            required_args=["entity_id"],
            optional_args={},
            returns="Boolean: True if entity exists, False otherwise"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> bool:
        """
        Execute exists function.

        Checks all entity stores (tasks, sprints, decisions, etc.)
        to determine if the given ID exists.

        Args:
            manager: GoT manager with entity access
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            True if entity exists, False otherwise
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # Check tasks
        if hasattr(manager, 'get_task'):
            task = manager.get_task(entity_id)
            if task is not None:
                return True

        # Check sprints
        if hasattr(manager, 'get_sprint'):
            sprint = manager.get_sprint(entity_id)
            if sprint is not None:
                return True

        # Check decisions
        if hasattr(manager, 'get_decision'):
            decision = manager.get_decision(entity_id)
            if decision is not None:
                return True

        # Check epics
        if hasattr(manager, 'get_epic'):
            epic = manager.get_epic(entity_id)
            if epic is not None:
                return True

        # Check knowledge transfers
        if hasattr(manager, 'get_knowledge_transfer'):
            kt = manager.get_knowledge_transfer(entity_id)
            if kt is not None:
                return True

        # Check handoffs
        if hasattr(manager, 'get_handoff'):
            handoff = manager.get_handoff(entity_id)
            if handoff is not None:
                return True

        return False


@FunctionRegistry.register("type_of")
class TypeOf(QueryFunction):
    """
    Determine the type of an entity in the Graph of Thought.

    Design Reference: docs/design/got-query-audit-and-design.md T-013
    specifies type_of(entity_id) -> str

    Usage:
        type_of('T-001') -> 'task'
        type_of('S-001') -> 'sprint'
        type_of('D-001') -> 'decision'
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="type_of",
            description="Get the type of an entity by ID",
            required_args=["entity_id"],
            optional_args={},
            returns="String: entity type (task, sprint, decision, etc.) or None"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """
        Execute type_of function.

        Determines entity type by checking each store and also
        by examining the ID prefix convention (T-, S-, D-, etc.).

        Args:
            manager: GoT manager with entity access
            args: Positional arguments [entity_id]
            kwargs: Keyword arguments {entity_id: str}

        Returns:
            Entity type as string, or None if not found
        """
        entity_id = args[0] if args else kwargs.get('entity_id')
        if not entity_id:
            raise ValueError("entity_id is required")

        # First, try ID prefix convention for efficiency
        # T- = Task, S- = Sprint, D- = Decision, E- = Edge/Epic
        # KT- = KnowledgeTransfer, H- = Handoff
        prefix_map = {
            'T-': 'task',
            'S-': 'sprint',
            'D-': 'decision',
            'KT-': 'knowledge_transfer',
            'H-': 'handoff',
        }

        for prefix, entity_type in prefix_map.items():
            if entity_id.startswith(prefix):
                # Verify the entity actually exists with this type
                getter_name = f'get_{entity_type}'
                if hasattr(manager, getter_name):
                    entity = getattr(manager, getter_name)(entity_id)
                    if entity is not None:
                        return entity_type

        # Fallback: Check each store explicitly
        if hasattr(manager, 'get_task'):
            task = manager.get_task(entity_id)
            if task is not None:
                return 'task'

        if hasattr(manager, 'get_sprint'):
            sprint = manager.get_sprint(entity_id)
            if sprint is not None:
                return 'sprint'

        if hasattr(manager, 'get_decision'):
            decision = manager.get_decision(entity_id)
            if decision is not None:
                return 'decision'

        if hasattr(manager, 'get_epic'):
            epic = manager.get_epic(entity_id)
            if epic is not None:
                return 'epic'

        if hasattr(manager, 'get_knowledge_transfer'):
            kt = manager.get_knowledge_transfer(entity_id)
            if kt is not None:
                return 'knowledge_transfer'

        if hasattr(manager, 'get_handoff'):
            handoff = manager.get_handoff(entity_id)
            if handoff is not None:
                return 'handoff'

        return None
