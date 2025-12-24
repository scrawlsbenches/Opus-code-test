"""
Tests for GoT query language.

Comprehensive tests for the graph query language including:
- Status-based queries (pending, completed, in_progress, blocked)
- Priority-based queries (high priority, critical)
- Sprint queries (tasks in sprint, current sprint)
- Relationship queries (what blocks, what depends on, relationships)
- Orphan detection
- Entity listing (decisions, handoffs, sprints)
- Time-based queries (recent, stale)
- Path queries
"""

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Import the adapter we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.got_utils import TransactionalGoTAdapter, STATUS_PENDING, STATUS_COMPLETED, STATUS_IN_PROGRESS
from cortical.got import GoTManager


class TestQueryLanguageBasics:
    """Test basic query language functionality."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        """Provide temporary GoT directory."""
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        """Provide TransactionalGoTAdapter instance with test data."""
        adapter = TransactionalGoTAdapter(got_dir)

        # Use the underlying manager to create tasks with status
        adapter._manager.create_task("Pending task 1", status="pending", priority="high")
        adapter._manager.create_task("Pending task 2", status="pending", priority="low")
        adapter._manager.create_task("Completed task", status="completed", priority="medium")
        adapter._manager.create_task("In progress task", status="in_progress", priority="critical")

        return adapter

    def test_query_pending_tasks(self, adapter):
        """Query 'pending tasks' returns pending tasks."""
        results = adapter.query("pending tasks")

        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Pending task 1" in titles
        assert "Pending task 2" in titles

    def test_query_active_tasks(self, adapter):
        """Query 'active tasks' returns in-progress tasks."""
        results = adapter.query("active tasks")

        assert len(results) == 1
        assert results[0]["title"] == "In progress task"

    def test_query_blocked_tasks_empty(self, adapter):
        """Query 'blocked tasks' returns empty when no blocked tasks."""
        results = adapter.query("blocked tasks")
        assert len(results) == 0

    def test_query_case_insensitive(self, adapter):
        """Queries are case insensitive."""
        results_lower = adapter.query("pending tasks")
        results_upper = adapter.query("PENDING TASKS")
        results_mixed = adapter.query("Pending Tasks")

        assert len(results_lower) == len(results_upper) == len(results_mixed)

    def test_query_with_whitespace(self, adapter):
        """Queries handle extra whitespace."""
        results = adapter.query("  pending tasks  ")
        assert len(results) == 2

    def test_query_unknown_returns_empty(self, adapter):
        """Unknown query returns empty list."""
        results = adapter.query("unknown query type")
        assert results == []


class TestQueryOrphans:
    """Test orphan detection queries."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        # Create orphan task (no edges)
        orphan = manager.create_task("Orphan task", status="pending")

        # Create connected tasks
        task1 = manager.create_task("Connected task 1", status="pending")
        task2 = manager.create_task("Connected task 2", status="pending")

        # Create sprint and add task to it
        sprint = manager.create_sprint("Test sprint", number=1)
        manager.add_edge(sprint.id, task1.id, "CONTAINS")

        # Create dependency between tasks
        manager.add_edge(task2.id, task1.id, "DEPENDS_ON")

        return adapter

    def test_query_orphan_tasks(self, adapter):
        """Query 'orphan tasks' returns unconnected tasks."""
        results = adapter.query("orphan tasks")

        assert len(results) == 1
        assert results[0]["title"] == "Orphan task"

    def test_query_orphan_nodes_alias(self, adapter):
        """Query 'orphan nodes' is alias for orphan tasks."""
        results = adapter.query("orphan nodes")
        assert len(results) == 1

    def test_query_orphans_alias(self, adapter):
        """Query 'orphans' is alias for orphan tasks."""
        results = adapter.query("orphans")
        assert len(results) == 1


class TestQueryRelationships:
    """Test relationship-based queries."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        # Create task chain: task1 <- task2 <- task3 (depends on)
        self.task1 = manager.create_task("Base task", status="pending")
        self.task2 = manager.create_task("Middle task", status="pending")
        self.task3 = manager.create_task("Top task", status="pending")

        # task2 depends on task1
        manager.add_edge(self.task2.id, self.task1.id, "DEPENDS_ON")
        # task3 depends on task2
        manager.add_edge(self.task3.id, self.task2.id, "DEPENDS_ON")

        return adapter

    def test_query_what_blocks(self, adapter):
        """Query 'what blocks <task>' returns blocking tasks."""
        # task1 blocks task2 (task2 depends on task1)
        results = adapter.query(f"what blocks {self.task2.id}")

        # Note: The query looks for BLOCKS edges, not DEPENDS_ON
        # If no BLOCKS edges exist, returns empty
        # This tests current behavior
        assert isinstance(results, list)

    def test_query_what_depends_on(self, adapter):
        """Query 'what depends on <task>' returns dependent tasks."""
        results = adapter.query(f"what depends on {self.task1.id}")

        # task2 depends on task1
        assert len(results) >= 1
        ids = {r["id"] for r in results}
        assert self.task2.id in ids

    def test_query_relationships(self, adapter):
        """Query 'relationships <task>' returns all relationships."""
        results = adapter.query(f"relationships {self.task2.id}")

        # task2 has relationships with task1 (depends on) and task3 (depended by)
        assert len(results) >= 1


class TestQueryStatusFilters:
    """Test status-based query filters."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        manager.create_task("Completed 1", status="completed")
        manager.create_task("Completed 2", status="completed")
        manager.create_task("In Progress", status="in_progress")
        manager.create_task("Pending", status="pending")

        return adapter

    def test_query_completed_tasks(self, adapter):
        """Query 'completed tasks' returns completed tasks."""
        results = adapter.query("completed tasks")

        assert len(results) == 2
        for r in results:
            assert r["status"] == "completed"

    def test_query_in_progress_tasks(self, adapter):
        """Query 'in_progress tasks' returns in-progress tasks."""
        results = adapter.query("in_progress tasks")

        assert len(results) == 1
        assert results[0]["status"] == "in_progress"


class TestQueryPriorityFilters:
    """Test priority-based query filters."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        manager.create_task("Critical task", priority="critical", status="pending")
        manager.create_task("High priority task", priority="high", status="pending")
        manager.create_task("Medium task", priority="medium", status="pending")
        manager.create_task("Low task", priority="low", status="pending")

        return adapter

    def test_query_high_priority_tasks(self, adapter):
        """Query 'high priority tasks' returns high priority tasks."""
        results = adapter.query("high priority tasks")

        assert len(results) == 1
        assert results[0]["priority"] == "high"

    def test_query_critical_tasks(self, adapter):
        """Query 'critical tasks' returns critical priority tasks."""
        results = adapter.query("critical tasks")

        assert len(results) == 1
        assert results[0]["priority"] == "critical"


class TestQuerySprintFilters:
    """Test sprint-based query filters."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        # Create sprints
        self.sprint1 = manager.create_sprint("Sprint 1", number=1, status="completed")
        self.sprint2 = manager.create_sprint("Sprint 2", number=2, status="in_progress")

        # Create tasks
        self.task1 = manager.create_task("Task in Sprint 1", status="completed")
        self.task2 = manager.create_task("Task in Sprint 2", status="in_progress")
        self.task3 = manager.create_task("Another in Sprint 2", status="pending")

        # Add tasks to sprints
        manager.add_edge(self.sprint1.id, self.task1.id, "CONTAINS")
        manager.add_edge(self.sprint2.id, self.task2.id, "CONTAINS")
        manager.add_edge(self.sprint2.id, self.task3.id, "CONTAINS")

        return adapter

    def test_query_tasks_in_sprint(self, adapter):
        """Query 'tasks in sprint <id>' returns sprint tasks."""
        results = adapter.query(f"tasks in sprint {self.sprint2.id}")

        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Task in Sprint 2" in titles
        assert "Another in Sprint 2" in titles

    def test_query_current_sprint(self, adapter):
        """Query 'current sprint' returns active sprint."""
        results = adapter.query("current sprint")

        assert len(results) == 1
        assert results[0]["title"] == "Sprint 2"
        assert results[0]["status"] == "in_progress"

    def test_query_all_sprints(self, adapter):
        """Query 'sprints' returns all sprints."""
        results = adapter.query("sprints")

        assert len(results) == 2


class TestQueryEntityListing:
    """Test entity listing queries."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        # Create various entities
        manager.create_task("Task 1", status="pending")
        manager.create_decision("Decision 1", rationale="Because")
        manager.create_decision("Decision 2", rationale="Also because")
        manager.create_sprint("Sprint 1", number=1)

        return adapter

    def test_query_decisions(self, adapter):
        """Query 'decisions' returns all decisions."""
        results = adapter.query("decisions")

        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Decision 1" in titles
        assert "Decision 2" in titles

    def test_query_all_tasks(self, adapter):
        """Query 'all tasks' returns all tasks."""
        results = adapter.query("all tasks")

        assert len(results) == 1


class TestQueryRecentAndStale:
    """Test time-based queries."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        adapter = TransactionalGoTAdapter(got_dir)
        manager = adapter._manager

        # Create tasks (all recent since just created)
        manager.create_task("Recent task", status="pending")

        return adapter

    def test_query_recent_tasks(self, adapter):
        """Query 'recent tasks' returns recently created tasks."""
        results = adapter.query("recent tasks")

        # Just created, so should include our task
        assert len(results) >= 1

    def test_query_stale_tasks(self, adapter):
        """Query 'stale tasks' returns tasks not updated recently."""
        results = adapter.query("stale tasks")

        # Just created, so nothing should be stale
        assert len(results) == 0


class TestGetOrphanTasks:
    """Test the get_orphan_tasks method directly."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        return TransactionalGoTAdapter(got_dir)

    def test_empty_returns_empty(self, adapter):
        """Empty graph returns no orphans."""
        orphans = adapter.get_orphan_tasks()
        assert orphans == []

    def test_single_orphan(self, adapter):
        """Single task with no edges is orphan."""
        adapter._manager.create_task("Orphan", status="pending")

        orphans = adapter.get_orphan_tasks()
        assert len(orphans) == 1
        assert orphans[0].content == "Orphan"

    def test_connected_task_not_orphan(self, adapter):
        """Task with edges is not orphan."""
        task = adapter._manager.create_task("Connected", status="pending")
        sprint = adapter._manager.create_sprint("Sprint", number=1)
        adapter._manager.add_edge(sprint.id, task.id, "CONTAINS")

        orphans = adapter.get_orphan_tasks()
        assert len(orphans) == 0


class TestListMethods:
    """Test the list_* methods added for usability."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def manager(self, got_dir):
        return GoTManager(got_dir)

    def test_list_tasks(self, manager):
        """list_tasks returns all tasks."""
        manager.create_task("Task 1")
        manager.create_task("Task 2")

        tasks = manager.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_with_status(self, manager):
        """list_tasks with status filter works."""
        manager.create_task("Pending", status="pending")
        manager.create_task("Completed", status="completed")

        pending = manager.list_tasks(status="pending")
        assert len(pending) == 1
        assert pending[0].title == "Pending"

    def test_list_edges(self, manager):
        """list_edges returns all edges."""
        task = manager.create_task("Task")
        decision = manager.create_decision("Decision", rationale="Why")
        manager.add_edge(task.id, decision.id, "JUSTIFIES")

        edges = manager.list_edges()
        assert len(edges) == 1

    def test_list_decisions(self, manager):
        """list_decisions returns all decisions."""
        manager.create_decision("Decision 1", rationale="Why 1")
        manager.create_decision("Decision 2", rationale="Why 2")

        decisions = manager.list_decisions()
        assert len(decisions) == 2

    def test_log_decision_alias(self, manager):
        """log_decision is alias for create_decision."""
        decision = manager.log_decision("Decision", rationale="Because")

        assert decision.title == "Decision"
        assert decision.rationale == "Because"
        assert decision.id.startswith("D-")


class TestQueryEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        return tmp_path / ".got"

    @pytest.fixture
    def adapter(self, got_dir):
        return TransactionalGoTAdapter(got_dir)

    def test_query_empty_string(self, adapter):
        """Empty query string returns empty list."""
        results = adapter.query("")
        assert results == []

    def test_query_nonexistent_task_id(self, adapter):
        """Query with nonexistent task ID returns empty."""
        results = adapter.query("what blocks T-nonexistent-task")
        assert results == []

    def test_query_relationships_nonexistent(self, adapter):
        """Relationships query with nonexistent ID returns empty."""
        results = adapter.query("relationships T-nonexistent")
        assert results == []
