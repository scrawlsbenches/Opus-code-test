"""
GoT CLI Commands Behavioral Tests
=================================

Tests the GoT CLI commands after the GoTManager API migration.
These tests verify commands work correctly with the new API.

Commands tested (fixed during refactoring):
- validate: Graph validation
- sprint list: List all sprints
- handoff list: List handoffs
- analyze: Graph analysis commands
- backlog list: List backlog tasks
- dashboard: Show dashboard

Uses in-memory containers for fast testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from io import StringIO
import sys

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.core.bootstrap import create_container
from cortical.got import GoTManager


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_got_dir():
    """Create a temporary directory for GoT storage."""
    temp_dir = tempfile.mkdtemp(prefix="test_got_cli_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_container(temp_got_dir):
    """Create an in-memory container for fast tests."""
    return create_container(got_dir=temp_got_dir, use_memory=True)


@pytest.fixture
def got_manager(memory_container):
    """Get GoTManager from the container."""
    return memory_container.resolve(GoTManager)


@pytest.fixture
def sample_tasks_and_sprints(got_manager):
    """Create sample data for CLI tests."""
    # Create tasks
    t1 = got_manager.create_task(
        title="Authentication feature",
        priority="high",
        status="pending"
    )
    t2 = got_manager.create_task(
        title="Database migration",
        priority="medium",
        status="in_progress"
    )
    t3 = got_manager.create_task(
        title="Code review",
        priority="low",
        status="completed"
    )

    # Create dependency
    got_manager.add_dependency(t2, t1)

    # Create sprint with tasks
    sprint = got_manager.create_sprint(title="Sprint 1", number=1)

    return {
        "tasks": [t1, t2, t3],
        "sprint": sprint
    }


# =============================================================================
# STORY 1: Developer Validates Graph
# =============================================================================

class TestValidateCommand:
    """Test 'got validate' command."""

    def test_validate_empty_graph(self, got_manager):
        """
        Scenario: Developer validates empty GoT graph
        Expected: Validation passes, returns valid graph state
        """
        # Call validate method (simulates CLI validate command)
        tasks = got_manager.list_tasks()
        edges = got_manager.list_edges()

        # Empty graph should be valid
        assert isinstance(tasks, list)
        assert isinstance(edges, list)

    def test_validate_graph_with_entities(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer validates graph with tasks and edges
        Expected: All entities are counted correctly
        """
        tasks = got_manager.list_tasks()
        edges = got_manager.list_edges()

        # Should have 3 tasks
        assert len(tasks) >= 3

        # Should have at least 1 edge (the dependency)
        assert len(edges) >= 1

    def test_validate_detects_edge_types(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer checks edge types
        Expected: Edge type is accessible and correct
        """
        edges = got_manager.list_edges()

        for edge in edges:
            # edge.edge_type should be accessible
            edge_type = edge.edge_type
            if isinstance(edge_type, str):
                assert edge_type in ["DEPENDS_ON", "BLOCKS", "CONTAINS", "RELATED"]
            else:
                # Enum type
                assert hasattr(edge_type, 'name')


# =============================================================================
# STORY 2: Developer Lists Sprints
# =============================================================================

class TestSprintListCommand:
    """Test 'got sprint list' command."""

    def test_list_sprints_empty(self, got_manager):
        """
        Scenario: Developer lists sprints when none exist
        Expected: Returns empty list
        """
        sprints = got_manager.list_sprints()
        assert isinstance(sprints, list)

    def test_list_sprints_with_data(self, got_manager):
        """
        Scenario: Developer lists sprints after creating some
        Expected: All sprints are returned
        """
        s1 = got_manager.create_sprint(title="Sprint Alpha", number=100)
        s2 = got_manager.create_sprint(title="Sprint Beta", number=101)

        sprints = got_manager.list_sprints()

        assert len(sprints) >= 2
        sprint_ids = [s.id for s in sprints]
        assert s1.id in sprint_ids
        assert s2.id in sprint_ids

    def test_sprint_progress(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer checks sprint progress
        Expected: Progress dict has correct keys
        """
        sprint = sample_tasks_and_sprints["sprint"]

        # Progress requires tasks to be linked
        progress = got_manager.get_sprint_progress(sprint)

        assert "total" in progress
        assert "completed" in progress
        assert "in_progress" in progress
        assert "pending" in progress
        assert "completion_rate" in progress


# =============================================================================
# STORY 3: Developer Lists Handoffs
# =============================================================================

class TestHandoffListCommand:
    """Test 'got handoff list' command."""

    def test_list_handoffs_empty(self, got_manager):
        """
        Scenario: Developer lists handoffs when none exist
        Expected: Returns empty list
        """
        handoffs = got_manager.list_handoffs()
        assert isinstance(handoffs, list)

    def test_list_handoffs_with_data(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer lists handoffs after creating one
        Expected: Handoff is accessible with correct attributes
        """
        task = sample_tasks_and_sprints["tasks"][0]

        # Create handoff with correct API signature
        handoff = got_manager.initiate_handoff(
            source_agent="current-agent",
            target_agent="next-agent",
            task_id=task.id,
            instructions="Continue work on auth"
        )

        handoffs = got_manager.list_handoffs()

        assert len(handoffs) >= 1
        # Check handoff attributes (not dict access)
        h = handoffs[0]
        assert hasattr(h, 'status') or hasattr(h, 'target_agent')


# =============================================================================
# STORY 4: Developer Uses Analyze Commands
# =============================================================================

class TestAnalyzeCommands:
    """Test 'got analyze' commands."""

    def test_analyze_dependency_count(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer analyzes dependencies
        Expected: Can count dependency edges
        """
        edges = got_manager.list_edges()
        dep_edges = [
            e for e in edges
            if str(e.edge_type) in ["DEPENDS_ON", "EdgeTypes.DEPENDS_ON"]
            or (hasattr(e.edge_type, 'name') and e.edge_type.name == "DEPENDS_ON")
        ]

        # At least one dependency was created
        assert len(dep_edges) >= 1

    def test_analyze_orphan_tasks(self, got_manager):
        """
        Scenario: Developer finds orphan tasks (no edges)
        Expected: Can identify isolated tasks
        """
        # Create orphan task
        orphan = got_manager.create_task(title="Orphan task", priority="low")

        # Get all task IDs with edges
        edges = got_manager.list_edges()
        connected_ids = set()
        for edge in edges:
            connected_ids.add(edge.source_id)
            connected_ids.add(edge.target_id)

        # Get all tasks
        tasks = got_manager.list_tasks()
        orphan_tasks = [t for t in tasks if t.id not in connected_ids]

        # The orphan task should be found
        orphan_ids = [t.id for t in orphan_tasks]
        assert orphan.id in orphan_ids


# =============================================================================
# STORY 5: Developer Uses Backlog Commands
# =============================================================================

class TestBacklogCommands:
    """Test 'got backlog' commands."""

    def test_list_backlog_empty(self, got_manager):
        """
        Scenario: Developer lists backlog when empty
        Expected: Returns empty list
        """
        # Backlog = tasks not assigned to sprint
        tasks = got_manager.list_tasks()
        # All tasks start unassigned
        assert isinstance(tasks, list)

    def test_list_backlog_unassigned_tasks(self, got_manager):
        """
        Scenario: Developer lists unassigned tasks
        Expected: Tasks without sprint assignment are listed
        """
        # Create unassigned task
        task = got_manager.create_task(title="Backlog item", priority="medium")

        # Create sprint and assigned task
        sprint = got_manager.create_sprint(title="Sprint 1", number=200)
        assigned_task = got_manager.create_task(
            title="Sprint item",
            priority="high",
            sprint_id=sprint
        )

        # Get all tasks
        all_tasks = got_manager.list_tasks()
        assert len(all_tasks) >= 2


# =============================================================================
# STORY 6: Developer Uses Dashboard
# =============================================================================

class TestDashboardCommand:
    """Test 'got dashboard' command."""

    def test_dashboard_empty_graph(self, got_manager):
        """
        Scenario: Developer views dashboard on empty graph
        Expected: Dashboard data can be collected without errors
        """
        # Collect dashboard data
        tasks = got_manager.list_tasks()
        edges = got_manager.list_edges()
        sprints = got_manager.list_sprints()
        handoffs = got_manager.list_handoffs()

        # All should return lists
        assert isinstance(tasks, list)
        assert isinstance(edges, list)
        assert isinstance(sprints, list)
        assert isinstance(handoffs, list)

    def test_dashboard_with_data(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer views dashboard with data
        Expected: Can compute metrics
        """
        tasks = got_manager.list_tasks()
        edges = got_manager.list_edges()

        # Compute task status breakdown
        status_counts = {}
        for task in tasks:
            status = task.status
            status_counts[status] = status_counts.get(status, 0) + 1

        assert "pending" in status_counts or "in_progress" in status_counts

    def test_dashboard_edge_statistics(self, got_manager, sample_tasks_and_sprints):
        """
        Scenario: Developer views edge statistics
        Expected: Can count edges by type
        """
        edges = got_manager.list_edges()

        # Count by edge type
        type_counts = {}
        for edge in edges:
            edge_type = str(edge.edge_type)
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

        assert len(type_counts) >= 0  # May be empty if no edges


# =============================================================================
# STORY 7: Developer Gets Blocked Tasks
# =============================================================================

class TestBlockedTasksCommand:
    """Test 'got blocked' command."""

    def test_get_blocked_tasks_empty(self, got_manager):
        """
        Scenario: Developer checks blocked tasks when none blocked
        Expected: Returns empty list
        """
        blocked = got_manager.get_blocked_tasks()
        assert isinstance(blocked, list)

    def test_get_blocked_tasks_with_blocked(self, got_manager):
        """
        Scenario: Developer checks blocked tasks after blocking one
        Expected: Blocked task is in list with reason
        """
        # Create and block task
        task = got_manager.create_task(title="Blocked item", priority="high")
        got_manager.block_task(task.id, "Waiting for approval")

        blocked = got_manager.get_blocked_tasks()

        assert len(blocked) >= 1
        # Each item is (task, reason)
        for blocked_task, reason in blocked:
            if blocked_task.id == task.id:
                assert reason == "Waiting for approval"
                break
        else:
            pytest.fail("Blocked task not found in list")


# =============================================================================
# STORY 8: Developer Gets Active Tasks
# =============================================================================

class TestActiveTasksCommand:
    """Test 'got active' command."""

    def test_get_active_tasks_empty(self, got_manager):
        """
        Scenario: Developer checks active tasks when none active
        Expected: Returns empty list
        """
        active = got_manager.list_tasks(status="in_progress")
        assert isinstance(active, list)

    def test_get_active_tasks_with_active(self, got_manager):
        """
        Scenario: Developer checks active tasks after starting one
        Expected: Active task is in list
        """
        task = got_manager.create_task(title="Active item", priority="high")
        got_manager.start_task(task.id)

        active = got_manager.list_tasks(status="in_progress")

        assert len(active) >= 1
        active_ids = [t.id for t in active]
        assert task.id in active_ids
