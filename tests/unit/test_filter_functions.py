"""
Unit tests for filter functions.

Tests all registered filter functions: recent, stale, has_edge, blocked,
blocking, in_sprint, unassigned, and overdue.
"""

import pytest
from datetime import datetime, timedelta, timezone

from cortical.got.api import GoTManager
from cortical.got.expression.registry import FunctionRegistry
from cortical.got.expression.functions.filters import (
    RecentFunction,
    StaleFunction,
    HasEdgeFunction,
    BlockedFunction,
    BlockingFunction,
    InSprintFunction,
    UnassignedFunction,
    OverdueFunction,
)


@pytest.fixture
def manager(fresh_got_manager: GoTManager) -> GoTManager:
    """Create a GoT manager with temporary storage."""
    return fresh_got_manager


@pytest.fixture
def manager_with_tasks(manager: GoTManager) -> GoTManager:
    """Create manager with sample tasks for testing."""
    # Create tasks with different timestamps
    now = datetime.now(timezone.utc)

    # Recent task (1 day ago) - create with specific timestamp
    with manager.transaction() as tx:
        from cortical.got.types import Task
        from cortical.utils.id_generation import generate_task_id
        task_id = generate_task_id()
        recent_task = Task(
            id=task_id,
            title="Recent task",
            priority="high",
            status="in_progress",
            created_at=(now - timedelta(days=1)).isoformat(),
            modified_at=(now - timedelta(days=1)).isoformat()
        )
        tx.tx_manager.write(tx.tx, recent_task)

    # Stale task (60 days ago) - create with specific timestamp
    with manager.transaction() as tx:
        from cortical.got.types import Task
        from cortical.utils.id_generation import generate_task_id
        task_id = generate_task_id()
        stale_task = Task(
            id=task_id,
            title="Stale task",
            priority="low",
            status="pending",
            created_at=(now - timedelta(days=60)).isoformat(),
            modified_at=(now - timedelta(days=60)).isoformat()
        )
        tx.tx_manager.write(tx.tx, stale_task)

    # Task with assignee
    assigned_task = manager.create_task(
        "Assigned task",
        priority="medium",
        status="in_progress"
    )
    manager.update_task(
        assigned_task.id,
        properties={"assignee": "agent-001"}
    )

    # Unassigned task
    unassigned_task = manager.create_task(
        "Unassigned task",
        priority="medium",
        status="pending"
    )

    # Overdue task
    overdue_task = manager.create_task(
        "Overdue task",
        priority="critical",
        status="pending"
    )
    overdue_date = (now - timedelta(days=5)).isoformat()
    manager.update_task(
        overdue_task.id,
        properties={"due_date": overdue_date}
    )

    # Not overdue task
    not_overdue_task = manager.create_task(
        "Future task",
        priority="medium",
        status="pending"
    )
    future_date = (now + timedelta(days=5)).isoformat()
    manager.update_task(
        not_overdue_task.id,
        properties={"due_date": future_date}
    )

    # Completed overdue task (should not appear in overdue)
    completed_task = manager.create_task(
        "Completed overdue task",
        priority="high",
        status="completed"
    )
    past_date = (now - timedelta(days=10)).isoformat()
    manager.update_task(
        completed_task.id,
        properties={"due_date": past_date}
    )

    return manager


@pytest.fixture
def manager_with_edges(manager: GoTManager) -> GoTManager:
    """Create manager with tasks and edges."""
    # Create tasks
    blocker = manager.create_task("Blocker task", priority="high")
    blocked = manager.create_task("Blocked task", priority="medium")
    dependency = manager.create_task("Dependency", priority="low")
    independent = manager.create_task("Independent task", priority="medium")

    # Create sprint
    sprint = manager.create_sprint(
        "Test sprint",
        description="Sprint for testing",
        status="in_progress"
    )

    # Create edges
    manager.add_edge(blocker.id, blocked.id, "BLOCKS")
    manager.add_edge(blocked.id, dependency.id, "DEPENDS_ON")
    manager.add_edge(sprint.id, blocker.id, "CONTAINS")
    manager.add_edge(sprint.id, blocked.id, "CONTAINS")

    return manager


class TestRecentFunction:
    """Test recent() function."""

    def test_recent_default_7_days(self, manager_with_tasks):
        """Test recent() with default 7 days."""
        func = RecentFunction()
        results = func.execute(manager_with_tasks, [], {})

        # Should include recent task (1 day ago), not stale (60 days ago)
        assert len(results) >= 1
        task_titles = [t.title for t in results]
        assert "Recent task" in task_titles
        assert "Stale task" not in task_titles

    def test_recent_custom_days(self, manager_with_tasks):
        """Test recent() with custom days parameter."""
        func = RecentFunction()

        # 2 days - should include recent (1 day ago)
        results = func.execute(manager_with_tasks, [2], {})
        task_titles = [t.title for t in results]
        assert "Recent task" in task_titles

        # 0.5 days (12 hours) - should include tasks from today
        results = func.execute(manager_with_tasks, [0.5], {})
        # Tasks created "now" should be included
        assert len(results) >= 5

    def test_recent_with_kwargs(self, manager_with_tasks):
        """Test recent() with keyword arguments."""
        func = RecentFunction()
        # Use 2 days to be safe (task created 1 day ago)
        results = func.execute(manager_with_tasks, [], {"days": 2})

        task_titles = [t.title for t in results]
        assert "Recent task" in task_titles

    def test_recent_negative_days_raises(self, manager_with_tasks):
        """Test recent() with negative days raises ValueError."""
        func = RecentFunction()
        with pytest.raises(ValueError, match="non-negative"):
            func.execute(manager_with_tasks, [-1], {})

    def test_recent_signature(self):
        """Test recent() function signature."""
        sig = RecentFunction.signature()
        assert sig.name == "recent"
        assert sig.optional_args["days"] == 7
        assert len(sig.required_args) == 0


class TestStaleFunction:
    """Test stale() function."""

    def test_stale_default_30_days(self, manager_with_tasks):
        """Test stale() with default 30 days."""
        func = StaleFunction()
        results = func.execute(manager_with_tasks, [], {})

        # Should include stale task (60 days ago)
        task_titles = [t.title for t in results]
        assert "Stale task" in task_titles
        assert "Recent task" not in task_titles

    def test_stale_custom_days(self, manager_with_tasks):
        """Test stale() with custom days parameter."""
        func = StaleFunction()

        # 50 days - should include stale (60 days ago)
        results = func.execute(manager_with_tasks, [50], {})
        task_titles = [t.title for t in results]
        assert "Stale task" in task_titles

        # 70 days - should not include stale (60 days ago)
        results = func.execute(manager_with_tasks, [70], {})
        task_titles = [t.title for t in results]
        assert "Stale task" not in task_titles

    def test_stale_negative_days_raises(self, manager_with_tasks):
        """Test stale() with negative days raises ValueError."""
        func = StaleFunction()
        with pytest.raises(ValueError, match="non-negative"):
            func.execute(manager_with_tasks, [-5], {})

    def test_stale_signature(self):
        """Test stale() function signature."""
        sig = StaleFunction.signature()
        assert sig.name == "stale"
        assert sig.optional_args["days"] == 30
        assert len(sig.required_args) == 0


class TestHasEdgeFunction:
    """Test has_edge() function."""

    def test_has_edge_blocks(self, manager_with_edges):
        """Test has_edge() with BLOCKS edge type."""
        func = HasEdgeFunction()
        results = func.execute(manager_with_edges, ["BLOCKS"], {})

        # Should include blocker and blocked tasks
        task_ids = [t.id for t in results]
        assert len(task_ids) >= 2

        task_titles = [t.title for t in results]
        assert "Blocker task" in task_titles
        assert "Blocked task" in task_titles

    def test_has_edge_depends_on(self, manager_with_edges):
        """Test has_edge() with DEPENDS_ON edge type."""
        func = HasEdgeFunction()
        results = func.execute(manager_with_edges, ["DEPENDS_ON"], {})

        task_titles = [t.title for t in results]
        assert "Blocked task" in task_titles
        assert "Dependency" in task_titles

    def test_has_edge_contains(self, manager_with_edges):
        """Test has_edge() with CONTAINS edge type."""
        func = HasEdgeFunction()
        results = func.execute(manager_with_edges, ["CONTAINS"], {})

        # Should include tasks that are in sprint
        task_titles = [t.title for t in results]
        assert "Blocker task" in task_titles
        assert "Blocked task" in task_titles

    def test_has_edge_nonexistent_type(self, manager_with_edges):
        """Test has_edge() with non-existent edge type returns empty."""
        func = HasEdgeFunction()
        results = func.execute(manager_with_edges, ["NONEXISTENT"], {})
        assert len(results) == 0

    def test_has_edge_missing_argument_raises(self, manager_with_edges):
        """Test has_edge() without edge_type raises ValueError."""
        func = HasEdgeFunction()
        with pytest.raises(ValueError, match="requires edge_type"):
            func.execute(manager_with_edges, [], {})

    def test_has_edge_signature(self):
        """Test has_edge() function signature."""
        sig = HasEdgeFunction.signature()
        assert sig.name == "has_edge"
        assert "edge_type" in sig.required_args
        assert len(sig.optional_args) == 0


class TestBlockedFunction:
    """Test blocked() function."""

    def test_blocked_finds_blocked_tasks(self, manager_with_edges):
        """Test blocked() finds tasks with incoming BLOCKS edges."""
        func = BlockedFunction()
        results = func.execute(manager_with_edges, [], {})

        # Should include blocked task (has incoming BLOCKS edge from incomplete blocker)
        task_titles = [t.title for t in results]
        assert "Blocked task" in task_titles

    def test_blocked_excludes_completed_blockers(self, manager_with_edges):
        """Test blocked() excludes tasks blocked by completed tasks."""
        # Complete the blocker
        tasks = manager_with_edges.query_api.find_tasks(title_contains="Blocker")
        blocker = tasks[0]
        manager_with_edges.update_task(blocker.id, status="completed")

        func = BlockedFunction()
        results = func.execute(manager_with_edges, [], {})

        # Should not include blocked task anymore
        task_titles = [t.title for t in results]
        assert "Blocked task" not in task_titles

    def test_blocked_empty_when_no_blocks(self, manager):
        """Test blocked() returns empty when no BLOCKS edges exist."""
        manager.create_task("Task 1", priority="high")
        manager.create_task("Task 2", priority="medium")

        func = BlockedFunction()
        results = func.execute(manager, [], {})
        assert len(results) == 0

    def test_blocked_signature(self):
        """Test blocked() function signature."""
        sig = BlockedFunction.signature()
        assert sig.name == "blocked"
        assert len(sig.required_args) == 0
        assert len(sig.optional_args) == 0


class TestBlockingFunction:
    """Test blocking() function."""

    def test_blocking_finds_blocker_tasks(self, manager_with_edges):
        """Test blocking() finds tasks with outgoing BLOCKS edges."""
        func = BlockingFunction()
        results = func.execute(manager_with_edges, [], {})

        # Should include blocker task (has outgoing BLOCKS edge to incomplete task)
        task_titles = [t.title for t in results]
        assert "Blocker task" in task_titles

    def test_blocking_excludes_completed_blocked(self, manager_with_edges):
        """Test blocking() excludes blockers of completed tasks."""
        # Complete the blocked task
        tasks = manager_with_edges.query_api.find_tasks(title_contains="Blocked")
        blocked = tasks[0]
        manager_with_edges.update_task(blocked.id, status="completed")

        func = BlockingFunction()
        results = func.execute(manager_with_edges, [], {})

        # Should not include blocker task anymore
        task_titles = [t.title for t in results]
        assert "Blocker task" not in task_titles

    def test_blocking_empty_when_no_blocks(self, manager):
        """Test blocking() returns empty when no BLOCKS edges exist."""
        manager.create_task("Task 1", priority="high")
        manager.create_task("Task 2", priority="medium")

        func = BlockingFunction()
        results = func.execute(manager, [], {})
        assert len(results) == 0

    def test_blocking_signature(self):
        """Test blocking() function signature."""
        sig = BlockingFunction.signature()
        assert sig.name == "blocking"
        assert len(sig.required_args) == 0
        assert len(sig.optional_args) == 0


class TestInSprintFunction:
    """Test in_sprint() function."""

    def test_in_sprint_finds_tasks(self, manager_with_edges):
        """Test in_sprint() finds tasks in specified sprint."""
        # Get sprint ID
        sprints = manager_with_edges.query_api.list_sprints()
        sprint_id = sprints[0].id

        func = InSprintFunction()
        results = func.execute(manager_with_edges, [sprint_id], {})

        # Should include tasks with CONTAINS edge from sprint
        assert len(results) >= 2
        task_titles = [t.title for t in results]
        assert "Blocker task" in task_titles
        assert "Blocked task" in task_titles

    def test_in_sprint_with_kwargs(self, manager_with_edges):
        """Test in_sprint() with keyword arguments."""
        sprints = manager_with_edges.query_api.list_sprints()
        sprint_id = sprints[0].id

        func = InSprintFunction()
        results = func.execute(manager_with_edges, [], {"sprint_id": sprint_id})

        assert len(results) >= 2

    def test_in_sprint_nonexistent_sprint(self, manager_with_edges):
        """Test in_sprint() with non-existent sprint returns empty."""
        func = InSprintFunction()
        results = func.execute(manager_with_edges, ["S-NONEXISTENT"], {})
        assert len(results) == 0

    def test_in_sprint_missing_argument_raises(self, manager_with_edges):
        """Test in_sprint() without sprint_id raises ValueError."""
        func = InSprintFunction()
        with pytest.raises(ValueError, match="requires sprint_id"):
            func.execute(manager_with_edges, [], {})

    def test_in_sprint_signature(self):
        """Test in_sprint() function signature."""
        sig = InSprintFunction.signature()
        assert sig.name == "in_sprint"
        assert "sprint_id" in sig.required_args
        assert len(sig.optional_args) == 0


class TestUnassignedFunction:
    """Test unassigned() function."""

    def test_unassigned_finds_tasks(self, manager_with_tasks):
        """Test unassigned() finds tasks without assignee."""
        func = UnassignedFunction()
        results = func.execute(manager_with_tasks, [], {})

        # Should include unassigned task
        task_titles = [t.title for t in results]
        assert "Unassigned task" in task_titles
        assert "Assigned task" not in task_titles

    def test_unassigned_checks_both_fields(self, manager):
        """Test unassigned() checks both properties and metadata."""
        # Task with assignee in properties
        task1 = manager.create_task("Task 1", priority="high")
        manager.update_task(task1.id, properties={"assignee": "agent-001"})

        # Task with assignee in metadata
        task2 = manager.create_task("Task 2", priority="medium")
        manager.update_task(task2.id, metadata={"assignee": "agent-002"})

        # Task with no assignee
        task3 = manager.create_task("Task 3", priority="low")

        func = UnassignedFunction()
        results = func.execute(manager, [], {})

        task_titles = [t.title for t in results]
        assert "Task 3" in task_titles
        assert "Task 1" not in task_titles
        assert "Task 2" not in task_titles

    def test_unassigned_empty_string_is_unassigned(self, manager):
        """Test unassigned() treats empty string as unassigned."""
        task = manager.create_task("Empty assignee", priority="medium")
        manager.update_task(task.id, properties={"assignee": ""})

        func = UnassignedFunction()
        results = func.execute(manager, [], {})

        task_titles = [t.title for t in results]
        assert "Empty assignee" in task_titles

    def test_unassigned_signature(self):
        """Test unassigned() function signature."""
        sig = UnassignedFunction.signature()
        assert sig.name == "unassigned"
        assert len(sig.required_args) == 0
        assert len(sig.optional_args) == 0


class TestOverdueFunction:
    """Test overdue() function."""

    def test_overdue_finds_overdue_tasks(self, manager_with_tasks):
        """Test overdue() finds tasks past due date."""
        func = OverdueFunction()
        results = func.execute(manager_with_tasks, [], {})

        # Should include overdue task
        task_titles = [t.title for t in results]
        assert "Overdue task" in task_titles
        assert "Future task" not in task_titles

    def test_overdue_excludes_completed(self, manager_with_tasks):
        """Test overdue() excludes completed tasks even if overdue."""
        func = OverdueFunction()
        results = func.execute(manager_with_tasks, [], {})

        task_titles = [t.title for t in results]
        assert "Completed overdue task" not in task_titles

    def test_overdue_checks_both_fields(self, manager):
        """Test overdue() checks both properties and metadata."""
        now = datetime.now(timezone.utc)
        past_date = (now - timedelta(days=5)).isoformat()

        # Task with due_date in properties
        task1 = manager.create_task("Task 1", priority="high", status="pending")
        manager.update_task(task1.id, properties={"due_date": past_date})

        # Task with due_date in metadata
        task2 = manager.create_task("Task 2", priority="medium", status="pending")
        manager.update_task(task2.id, metadata={"due_date": past_date})

        func = OverdueFunction()
        results = func.execute(manager, [], {})

        task_titles = [t.title for t in results]
        assert "Task 1" in task_titles
        assert "Task 2" in task_titles

    def test_overdue_no_due_date_excluded(self, manager):
        """Test overdue() excludes tasks without due_date."""
        manager.create_task("No due date", priority="high", status="pending")

        func = OverdueFunction()
        results = func.execute(manager, [], {})

        task_titles = [t.title for t in results]
        assert "No due date" not in task_titles

    def test_overdue_signature(self):
        """Test overdue() function signature."""
        sig = OverdueFunction.signature()
        assert sig.name == "overdue"
        assert len(sig.required_args) == 0
        assert len(sig.optional_args) == 0


class TestFunctionRegistration:
    """Test that all functions are properly registered."""

    def test_all_functions_registered(self):
        """Test that all 8 filter functions are registered."""
        registry = FunctionRegistry
        # Compare by class name to handle module reloading in tests
        assert registry.get("recent").__name__ == "RecentFunction"
        assert registry.get("stale").__name__ == "StaleFunction"
        assert registry.get("has_edge").__name__ == "HasEdgeFunction"
        assert registry.get("blocked").__name__ == "BlockedFunction"
        assert registry.get("blocking").__name__ == "BlockingFunction"
        assert registry.get("in_sprint").__name__ == "InSprintFunction"
        assert registry.get("unassigned").__name__ == "UnassignedFunction"
        assert registry.get("overdue").__name__ == "OverdueFunction"

    def test_case_insensitive_lookup(self):
        """Test that function lookup is case-insensitive."""
        registry = FunctionRegistry
        # Compare by class name to handle module reloading in tests
        assert registry.get("RECENT").__name__ == "RecentFunction"
        assert registry.get("Stale").__name__ == "StaleFunction"
        assert registry.get("HAS_EDGE").__name__ == "HasEdgeFunction"

    def test_list_functions_includes_filters(self):
        """Test that list_functions includes all filter functions."""
        signatures = FunctionRegistry.list_functions()
        names = [sig.name for sig in signatures]

        assert "recent" in names
        assert "stale" in names
        assert "has_edge" in names
        assert "blocked" in names
        assert "blocking" in names
        assert "in_sprint" in names
        assert "unassigned" in names
        assert "overdue" in names
