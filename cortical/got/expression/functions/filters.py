"""
Filter functions for advanced entity filtering.

Provides registered functions for time-based, relationship-based,
and state-based filtering in GoT queries.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, TYPE_CHECKING

from cortical.got.expression.registry import FunctionRegistry, FunctionSignature, QueryFunction
from cortical.got.types import EdgeTypes

if TYPE_CHECKING:
    from cortical.got.api import GoTManager


@FunctionRegistry.register("recent")
class RecentFunction(QueryFunction):
    """
    Filter entities created within N days.

    Returns entities where created_at is within the specified
    number of days from now.

    Args:
        days: Number of days to look back (default: 7)

    Returns:
        List of entities created within time window

    Example:
        recent(7)      # Entities from last week
        recent(1)      # Entities from last day
        recent(days=30) # Entities from last month
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="recent",
            description="Find entities created/modified within N days",
            required_args=[],
            optional_args={"days": 7},
            returns="List[Entity]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        days = args[0] if args else kwargs.get("days", 7)
        if not isinstance(days, (int, float)) or days < 0:
            raise ValueError(f"days must be a non-negative number, got {days}")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        # Get all tasks and filter by timestamp
        tasks = manager.query_api.list_all_tasks()
        results = []

        for task in tasks:
            # Check created_at (primary) or modified_at (fallback)
            # Use created_at as primary since modified_at gets updated on every write
            if task.created_at >= cutoff_str:
                results.append(task)

        return results


@FunctionRegistry.register("stale")
class StaleFunction(QueryFunction):
    """
    Filter entities NOT created recently (older than N days).

    Returns entities where created_at is older than
    the specified number of days from now.

    Args:
        days: Number of days threshold (default: 30)

    Returns:
        List of old entities

    Example:
        stale(30)      # Created 30+ days ago
        stale(90)      # Created 90+ days ago
        stale(days=7)  # Created 7+ days ago
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="stale",
            description="Find entities created more than N days ago",
            required_args=[],
            optional_args={"days": 30},
            returns="List[Entity]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        days = args[0] if args else kwargs.get("days", 30)
        if not isinstance(days, (int, float)) or days < 0:
            raise ValueError(f"days must be a non-negative number, got {days}")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        # Get all tasks and filter by timestamp
        tasks = manager.query_api.list_all_tasks()
        results = []

        for task in tasks:
            # Check created_at (since modified_at gets updated on every write)
            if task.created_at < cutoff_str:
                results.append(task)

        return results


@FunctionRegistry.register("has_edge")
class HasEdgeFunction(QueryFunction):
    """
    Filter entities that have at least one edge of the given type.

    Args:
        edge_type: Edge type to filter by (e.g., "BLOCKS", "DEPENDS_ON")

    Returns:
        List of entities with at least one edge of specified type

    Example:
        has_edge("BLOCKS")     # Tasks that block something
        has_edge("DEPENDS_ON") # Tasks with dependencies
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="has_edge",
            description="Find entities with at least one edge of given type",
            required_args=["edge_type"],
            optional_args={},
            returns="List[Entity]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        if not args and "edge_type" not in kwargs:
            raise ValueError("has_edge requires edge_type argument")

        edge_type = args[0] if args else kwargs.get("edge_type")
        if not isinstance(edge_type, str):
            raise ValueError(f"edge_type must be a string, got {type(edge_type)}")

        # Get all edges and find entities with this edge type
        all_edges = manager.query_api.list_edges()
        entity_ids = set()

        for edge in all_edges:
            if edge.edge_type == edge_type:
                # Add both source and target
                entity_ids.add(edge.source_id)
                entity_ids.add(edge.target_id)

        # Load the entities (currently only supports tasks)
        results = []
        for entity_id in entity_ids:
            if entity_id.startswith("T-"):
                task = manager.get_task(entity_id)
                if task is not None:
                    results.append(task)

        return results


@FunctionRegistry.register("blocked")
class BlockedFunction(QueryFunction):
    """
    Filter tasks that are blocked by other tasks.

    Returns tasks that have incoming BLOCKS edges from
    incomplete tasks.

    Returns:
        List of blocked tasks

    Example:
        blocked()  # All blocked tasks
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="blocked",
            description="Find tasks that are blocked by other tasks",
            required_args=[],
            optional_args={},
            returns="List[Task]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        # Use the built-in query method
        return manager.query_api.get_blocked_tasks()


@FunctionRegistry.register("blocking")
class BlockingFunction(QueryFunction):
    """
    Filter tasks that are blocking other tasks.

    Returns tasks that have outgoing BLOCKS edges to
    incomplete tasks.

    Returns:
        List of tasks blocking others

    Example:
        blocking()  # All tasks that block others
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="blocking",
            description="Find tasks that are blocking other tasks",
            required_args=[],
            optional_args={},
            returns="List[Task]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        # Find all BLOCKS edges
        all_edges = manager.query_api.list_edges()
        blocker_ids = set()

        for edge in all_edges:
            if edge.edge_type == EdgeTypes.BLOCKS:
                # Check if the blocked task is not completed
                blocked_task = manager.get_task(edge.target_id)
                if blocked_task is not None and blocked_task.status != "completed":
                    blocker_ids.add(edge.source_id)

        # Load the blocker tasks
        results = []
        for blocker_id in blocker_ids:
            task = manager.get_task(blocker_id)
            if task is not None:
                results.append(task)

        return results


@FunctionRegistry.register("in_sprint")
class InSprintFunction(QueryFunction):
    """
    Filter tasks that are in a specific sprint.

    Returns tasks that have a CONTAINS edge from the sprint.

    Args:
        sprint_id: Sprint identifier

    Returns:
        List of tasks in the sprint

    Example:
        in_sprint("S-020")                    # Tasks in sprint S-020
        in_sprint(sprint_id="S-sprint-017")   # Tasks in legacy sprint
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="in_sprint",
            description="Find tasks in a specific sprint",
            required_args=["sprint_id"],
            optional_args={},
            returns="List[Task]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        if not args and "sprint_id" not in kwargs:
            raise ValueError("in_sprint requires sprint_id argument")

        sprint_id = args[0] if args else kwargs.get("sprint_id")
        if not isinstance(sprint_id, str):
            raise ValueError(f"sprint_id must be a string, got {type(sprint_id)}")

        # Use the built-in query method
        return manager.query_api.get_sprint_tasks(sprint_id)


@FunctionRegistry.register("unassigned")
class UnassignedFunction(QueryFunction):
    """
    Filter tasks with no assignee.

    Returns tasks where assignee field is missing, None, or empty string.

    Returns:
        List of unassigned tasks

    Example:
        unassigned()  # All tasks without assignee
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="unassigned",
            description="Find tasks with no assignee",
            required_args=[],
            optional_args={},
            returns="List[Task]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        # Get all tasks and filter by assignee
        tasks = manager.query_api.list_all_tasks()
        results = []

        for task in tasks:
            # Check properties.assignee and metadata.assignee
            assignee_prop = task.properties.get("assignee")
            assignee_meta = task.metadata.get("assignee")

            # Unassigned if both are None/empty
            if not assignee_prop and not assignee_meta:
                results.append(task)

        return results


@FunctionRegistry.register("overdue")
class OverdueFunction(QueryFunction):
    """
    Filter tasks past their due date.

    Returns tasks where due_date field exists and is in the past.
    Checks both properties.due_date and metadata.due_date.

    Returns:
        List of overdue tasks

    Example:
        overdue()  # All tasks past due date
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="overdue",
            description="Find tasks past their due date",
            required_args=[],
            optional_args={},
            returns="List[Task]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        now = datetime.now(timezone.utc).isoformat()

        # Get all tasks and filter by due_date
        tasks = manager.query_api.list_all_tasks()
        results = []

        for task in tasks:
            # Skip completed tasks
            if task.status == "completed":
                continue

            # Check both properties.due_date and metadata.due_date
            due_date = task.properties.get("due_date") or task.metadata.get("due_date")

            if due_date and isinstance(due_date, str):
                # Compare ISO strings (works for ISO 8601 format)
                if due_date < now:
                    results.append(task)

        return results


@FunctionRegistry.register("entity_type")
class EntityTypeFunction(QueryFunction):
    """
    Filter entities by their type.

    Returns entities of the specified type (task, decision, sprint, etc.).

    Args:
        type_name: Entity type to filter by ('task', 'decision', 'sprint', etc.)

    Returns:
        List of entities of the specified type

    Example:
        entity_type('decision')  # All decisions
        entity_type('sprint')    # All sprints
        entity_type('task')      # All tasks
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="entity_type",
            description="Find entities of a specific type",
            required_args=["type_name"],
            optional_args={},
            returns="List[Entity]"
        )

    def execute(
        self,
        manager: "GoTManager",
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        if not args and "type_name" not in kwargs:
            raise ValueError("entity_type requires type_name argument")

        type_name = args[0] if args else kwargs.get("type_name")
        if not isinstance(type_name, str):
            raise ValueError(f"type_name must be a string, got {type(type_name)}")

        type_name = type_name.lower()

        # Route to appropriate manager method based on type
        if type_name == "decision":
            return manager.list_decisions()
        elif type_name == "sprint":
            return manager.list_sprints()
        elif type_name == "task":
            return manager.query_api.list_all_tasks()
        elif type_name == "edge":
            return manager.list_edges()
        elif type_name == "handoff":
            return manager.list_handoffs()
        elif type_name == "kt" or type_name == "knowledge_transfer":
            return manager.list_knowledge_transfers()
        else:
            # Unknown type - return empty list
            return []
