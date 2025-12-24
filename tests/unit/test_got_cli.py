"""
Comprehensive Unit Tests for GoT CLI System
============================================

Tests the CLI commands in scripts/got_utils.py including:
- Task operations (create, list, start, complete, block)
- Decision operations (log, list, why)
- Handoff operations (initiate, accept, complete, list)
- Query language
- Edge inference
- Statistics
- Sprint management
"""

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from cortical.reasoning.graph_of_thought import NodeType, EdgeType, ThoughtNode
from cortical.reasoning.thought_graph import ThoughtGraph

# Import after path setup
import got_utils
from got_utils import (
    GoTProjectManager,
    EventLog,
    cmd_task_create,
    cmd_task_list,
    cmd_task_show,
    cmd_task_start,
    cmd_task_complete,
    cmd_task_block,
    cmd_task_delete,
    cmd_decision_log,
    cmd_decision_list,
    cmd_decision_why,
    cmd_handoff_initiate,
    cmd_handoff_accept,
    cmd_handoff_complete,
    cmd_handoff_list,
    cmd_query,
    cmd_infer,
    cmd_stats,
    cmd_sprint_create,
    cmd_sprint_status,
    cmd_sprint_list,
    generate_task_id,
    generate_sprint_id,
    generate_decision_id,
    STATUS_PENDING,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETED,
    STATUS_BLOCKED,
    got_auto_commit,
    _got_auto_push,
    GOT_AUTO_COMMIT_ENABLED,
    GOT_AUTO_PUSH_ENABLED,
    MUTATING_COMMANDS,
    PROTECTED_BRANCHES,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_got_dir():
    """Create a temporary GoT directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_manager(temp_got_dir):
    """Create a mock GoTProjectManager with pre-configured behavior.

    Note: We don't use spec=GoTProjectManager because the TX backend
    (TransactionalGoTAdapter) adds additional methods like initiate_handoff,
    accept_handoff, complete_handoff, and list_handoffs.
    """
    manager = MagicMock()
    manager.got_dir = temp_got_dir
    manager.events_dir = temp_got_dir / "events"
    manager.events_dir.mkdir(parents=True, exist_ok=True)

    # Mock graph
    manager.graph = ThoughtGraph()

    # Mock event log
    manager.event_log = Mock()

    # Default return values for standard operations
    manager.save.return_value = None
    manager.create_task.return_value = "task:T-20251220-120000-abc123"
    manager.create_sprint.return_value = "sprint:S-001"
    manager.log_decision.return_value = "decision:D-20251220-120000-def456"

    # Default return values for handoff operations (TX backend)
    manager.initiate_handoff.return_value = "H-20251220-120000-abc123"
    manager.accept_handoff.return_value = True
    manager.complete_handoff.return_value = True
    manager.list_handoffs.return_value = []

    return manager


@pytest.fixture
def mock_args():
    """Create a mock args object for argparse."""
    args = Mock()
    return args


@pytest.fixture
def capture_output():
    """Capture stdout for testing CLI output."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    yield sys.stdout
    sys.stdout = old_stdout


# =============================================================================
# TASK COMMAND TESTS
# =============================================================================


class TestTaskCreate:
    """Tests for task create command."""

    def test_create_basic_task(self, mock_manager, mock_args):
        """Create a basic task with minimal arguments."""
        mock_args.title = "Fix bug"
        mock_args.priority = "high"
        mock_args.category = "bugfix"
        mock_args.description = ""
        mock_args.sprint = None
        mock_args.depends = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_create(mock_args, mock_manager)

        assert result == 0
        mock_manager.create_task.assert_called_once()
        assert "Created:" in captured.getvalue()

    def test_create_task_with_all_options(self, mock_manager, mock_args):
        """Create a task with all optional arguments."""
        mock_args.title = "Implement feature"
        mock_args.priority = "critical"
        mock_args.category = "feature"
        mock_args.description = "A detailed description"
        mock_args.sprint = "sprint:S-001"
        mock_args.depends = ["task:T-001", "task:T-002"]

        with patch('sys.stdout', new=StringIO()):
            result = cmd_task_create(mock_args, mock_manager)

        assert result == 0
        mock_manager.create_task.assert_called_once()
        call_kwargs = mock_manager.create_task.call_args[1]
        assert call_kwargs['title'] == "Implement feature"
        assert call_kwargs['priority'] == "critical"
        assert call_kwargs['category'] == "feature"
        mock_manager.save.assert_called_once()

    def test_create_task_prints_id(self, mock_manager, mock_args):
        """Task creation prints the task ID."""
        mock_args.title = "Test"
        mock_args.priority = "medium"
        mock_args.category = "test"
        mock_args.description = ""
        mock_args.sprint = None
        mock_args.depends = None

        mock_manager.create_task.return_value = "task:T-12345"

        with patch('sys.stdout', new=StringIO()) as captured:
            cmd_task_create(mock_args, mock_manager)

        assert "task:T-12345" in captured.getvalue()


class TestTaskList:
    """Tests for task list command."""

    def test_list_all_tasks(self, mock_manager, mock_args):
        """List all tasks with no filters."""
        mock_args.status = None
        mock_args.priority = None
        mock_args.category = None
        mock_args.sprint = None
        mock_args.blocked = False
        mock_args.json = False

        # Create mock tasks with correct attribute names
        task1 = Mock()
        task1.id = "task:T-001"
        task1.content = "Task 1"
        task1.properties = {"status": "pending", "priority": "high"}

        task2 = Mock()
        task2.id = "task:T-002"
        task2.content = "Task 2"
        task2.properties = {"status": "completed", "priority": "low"}

        mock_manager.list_tasks.return_value = [task1, task2]

        with patch('got_utils.format_task_table', return_value="Task Table"):
            with patch('sys.stdout', new=StringIO()):
                result = cmd_task_list(mock_args, mock_manager)

        assert result == 0
        mock_manager.list_tasks.assert_called_once()

    def test_list_tasks_with_filters(self, mock_manager, mock_args):
        """List tasks with status and priority filters."""
        mock_args.status = "pending"
        mock_args.priority = "high"
        mock_args.category = None
        mock_args.sprint = None
        mock_args.blocked = False
        mock_args.json = False

        mock_manager.list_tasks.return_value = []

        with patch('got_utils.format_task_table', return_value=""):
            result = cmd_task_list(mock_args, mock_manager)

        assert result == 0
        mock_manager.list_tasks.assert_called_with(
            status="pending",
            priority="high",
            category=None,
            sprint_id=None,
            blocked_only=False,
        )

    def test_list_tasks_json_output(self, mock_manager, mock_args):
        """List tasks with JSON output format."""
        mock_args.status = None
        mock_args.priority = None
        mock_args.category = None
        mock_args.sprint = None
        mock_args.blocked = False
        mock_args.json = True

        task = Mock()
        task.id = "task:T-001"
        task.content = "Test Task"
        task.properties = {"status": "pending", "priority": "medium"}

        mock_manager.list_tasks.return_value = [task]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_list(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        # Should contain JSON
        assert "[" in output or "{" in output


class TestTaskStart:
    """Tests for task start command."""

    def test_start_existing_task(self, mock_manager, mock_args):
        """Start an existing task."""
        mock_args.task_id = "task:T-001"
        mock_manager.start_task.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_start(mock_args, mock_manager)

        assert result == 0
        mock_manager.start_task.assert_called_once_with("task:T-001")
        mock_manager.save.assert_called_once()
        assert "Started:" in captured.getvalue()

    def test_start_nonexistent_task(self, mock_manager, mock_args):
        """Starting a nonexistent task returns error."""
        mock_args.task_id = "task:T-999"
        mock_manager.start_task.return_value = False

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_start(mock_args, mock_manager)

        assert result == 1
        assert "not found" in captured.getvalue()
        mock_manager.save.assert_not_called()


class TestTaskShow:
    """Tests for task show command - isolated unit tests."""

    def test_show_existing_task(self, mock_manager, mock_args):
        """Show details of an existing task."""
        mock_args.task_id = "task:T-001"

        # Create a mock task node
        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Fix authentication bug",
            properties={
                "status": "in_progress",
                "priority": "high",
                "category": "bugfix",
                "description": "Users cannot login"
            },
            metadata={
                "created_at": "2025-12-20T10:00:00Z",
                "updated_at": "2025-12-20T11:00:00Z"
            }
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_show(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "task:T-001" in output
        assert "Fix authentication bug" in output
        assert "in_progress" in output
        assert "high" in output

    def test_show_nonexistent_task(self, mock_manager, mock_args):
        """Show returns error for nonexistent task."""
        mock_args.task_id = "T-NONEXISTENT"
        mock_manager.get_task.return_value = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_show(mock_args, mock_manager)

        assert result == 1
        assert "not found" in captured.getvalue()

    def test_show_task_with_dependencies(self, mock_manager, mock_args):
        """Show displays task dependencies."""
        mock_args.task_id = "task:T-002"

        # Main task
        task_node = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Implement feature",
            properties={"status": "pending", "priority": "medium", "category": "feature"},
            metadata={}
        )

        # Dependency
        dep_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Design database schema",
            properties={},
            metadata={}
        )

        mock_manager.get_task.return_value = task_node
        mock_manager.get_task_dependencies.return_value = [dep_node]
        mock_manager.what_depends_on.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_show(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Depends On" in output
        assert "task:T-001" in output
        assert "Design database schema" in output

    def test_show_task_with_dependents(self, mock_manager, mock_args):
        """Show displays tasks that depend on this one."""
        mock_args.task_id = "task:T-001"

        # Main task
        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Core library",
            properties={"status": "completed", "priority": "high", "category": "feature"},
            metadata={}
        )

        # Dependent task
        dependent_node = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Build on core",
            properties={},
            metadata={}
        )

        mock_manager.get_task.return_value = task_node
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = [dependent_node]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_show(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Blocks" in output
        assert "task:T-002" in output
        assert "Build on core" in output

    def test_show_with_id_normalization_bare_id(self, mock_manager, mock_args):
        """Show normalizes bare task ID (without task: prefix)."""
        mock_args.task_id = "T-20251220-001"

        # First call with bare ID returns None
        # Second call with prefixed ID returns the task
        task_node = ThoughtNode(
            id="task:T-20251220-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending", "priority": "low", "category": "feature"},
            metadata={}
        )

        # get_task returns None for bare ID, then task for prefixed ID
        mock_manager.get_task.side_effect = [None, task_node]
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_show(mock_args, mock_manager)

        assert result == 0
        # Verify it tried with prefix after bare ID failed
        calls = mock_manager.get_task.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "T-20251220-001"
        assert calls[1][0][0] == "task:T-20251220-001"

    def test_show_with_retrospective(self, mock_manager, mock_args):
        """Show displays retrospective if present."""
        mock_args.task_id = "task:T-001"

        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Completed task",
            properties={
                "status": "completed",
                "priority": "high",
                "category": "bugfix",
                "retrospective": "Fixed by updating dependency"
            },
            metadata={"completed_at": "2025-12-20T15:00:00Z"}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_show(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Retrospective" in output
        assert "Fixed by updating dependency" in output


class TestTaskComplete:
    """Tests for task complete command."""

    def test_complete_task_without_retrospective(self, mock_manager, mock_args):
        """Complete a task without retrospective notes."""
        mock_args.task_id = "task:T-001"
        mock_args.retrospective = None
        mock_manager.complete_task.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_complete(mock_args, mock_manager)

        assert result == 0
        mock_manager.complete_task.assert_called_once_with("task:T-001", None)
        assert "Completed:" in captured.getvalue()

    def test_complete_task_with_retrospective(self, mock_manager, mock_args):
        """Complete a task with retrospective notes."""
        mock_args.task_id = "task:T-001"
        mock_args.retrospective = "Learned about edge cases"
        mock_manager.complete_task.return_value = True

        result = cmd_task_complete(mock_args, mock_manager)

        assert result == 0
        mock_manager.complete_task.assert_called_once_with(
            "task:T-001",
            "Learned about edge cases"
        )
        mock_manager.save.assert_called_once()

    def test_complete_nonexistent_task(self, mock_manager, mock_args):
        """Completing a nonexistent task returns error."""
        mock_args.task_id = "task:T-999"
        mock_args.retrospective = None
        mock_manager.complete_task.return_value = False

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_complete(mock_args, mock_manager)

        assert result == 1
        assert "not found" in captured.getvalue()


class TestTaskBlock:
    """Tests for task block command."""

    def test_block_task_with_reason(self, mock_manager, mock_args):
        """Block a task with a reason."""
        mock_args.task_id = "task:T-001"
        mock_args.reason = "Waiting for API design"
        mock_args.blocker = None
        mock_manager.block_task.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_block(mock_args, mock_manager)

        assert result == 0
        mock_manager.block_task.assert_called_once_with(
            "task:T-001",
            "Waiting for API design",
            None
        )
        assert "Blocked:" in captured.getvalue()

    def test_block_task_with_blocker(self, mock_manager, mock_args):
        """Block a task with a blocking task."""
        mock_args.task_id = "task:T-002"
        mock_args.reason = "Depends on other task"
        mock_args.blocker = "task:T-001"
        mock_manager.block_task.return_value = True

        result = cmd_task_block(mock_args, mock_manager)

        assert result == 0
        mock_manager.block_task.assert_called_once_with(
            "task:T-002",
            "Depends on other task",
            "task:T-001"
        )


class TestTaskDelete:
    """
    Unit tests for task delete command - isolated with mocks.

    Converted from behavioral tests to avoid data creation.
    Tests the transactional safety checks of delete_task.
    """

    def test_delete_standalone_task(self, mock_manager, mock_args):
        """Delete a task with no dependencies."""
        mock_args.task_id = "T-001"
        mock_args.force = False

        # Mock: task exists with no dependents
        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Standalone task",
            properties={"status": STATUS_PENDING},
            metadata={}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.what_depends_on.return_value = []
        mock_manager.delete_task.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        assert result == 0
        mock_manager.delete_task.assert_called_once_with("T-001", force=False)
        output = captured.getvalue()
        assert "Deleted" in output

    def test_delete_nonexistent_task_fails(self, mock_manager, mock_args):
        """Delete returns error for nonexistent task."""
        mock_args.task_id = "T-NONEXISTENT"
        mock_args.force = False

        mock_manager.get_task.return_value = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        assert result == 1
        output = captured.getvalue()
        assert "not found" in output

    def test_delete_with_dependents_fails_without_force(self, mock_manager, mock_args):
        """Delete fails when task has dependents and no --force."""
        mock_args.task_id = "T-001"
        mock_args.force = False

        # Mock: task exists with dependents
        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Prerequisite task",
            properties={"status": STATUS_PENDING},
            metadata={}
        )
        dependent_node = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Depends on T-001",
            properties={},
            metadata={}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.what_depends_on.return_value = [dependent_node]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        assert result == 1
        output = captured.getvalue()
        assert "Cannot delete" in output or "depend" in output.lower()
        # delete_task should NOT have been called
        mock_manager.delete_task.assert_not_called()

    def test_delete_with_force_succeeds(self, mock_manager, mock_args):
        """Force delete bypasses dependency check."""
        mock_args.task_id = "T-001"
        mock_args.force = True

        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Task to force delete",
            properties={"status": STATUS_PENDING},
            metadata={}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.delete_task.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        assert result == 0
        mock_manager.delete_task.assert_called_once_with("T-001", force=True)

    def test_delete_in_progress_fails_without_force(self, mock_manager, mock_args):
        """Delete fails for in-progress task without --force."""
        mock_args.task_id = "T-001"
        mock_args.force = False

        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="In-progress task",
            properties={"status": STATUS_IN_PROGRESS},
            metadata={}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.what_depends_on.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        assert result == 1
        output = captured.getvalue()
        assert "in progress" in output.lower() or "Cannot delete" in output

    def test_delete_completed_task_allowed(self, mock_manager, mock_args):
        """Completed tasks can be deleted without --force."""
        mock_args.task_id = "T-001"
        mock_args.force = False

        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Completed task",
            properties={"status": STATUS_COMPLETED},
            metadata={}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.what_depends_on.return_value = []
        mock_manager.delete_task.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        assert result == 0
        mock_manager.delete_task.assert_called_once()

    def test_delete_shows_force_hint(self, mock_manager, mock_args):
        """When blocked, shows hint about --force."""
        mock_args.task_id = "T-001"
        mock_args.force = False

        task_node = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Task with deps",
            properties={"status": STATUS_PENDING},
            metadata={}
        )
        dependent_node = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Dependent",
            properties={},
            metadata={}
        )
        mock_manager.get_task.return_value = task_node
        mock_manager.what_depends_on.return_value = [dependent_node]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_task_delete(mock_args, mock_manager)

        output = captured.getvalue()
        assert "--force" in output


# =============================================================================
# DECISION COMMAND TESTS
# =============================================================================


class TestDecisionLog:
    """Tests for decision log command."""

    def test_log_basic_decision(self, mock_manager, mock_args):
        """Log a basic decision with rationale."""
        mock_args.decision = "Use PostgreSQL for storage"
        mock_args.rationale = "Better performance for complex queries"
        mock_args.affects = None
        mock_args.alternatives = None
        mock_args.file = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_log(mock_args, mock_manager)

        assert result == 0
        mock_manager.log_decision.assert_called_once()
        output = captured.getvalue()
        assert "Decision logged:" in output
        assert "Use PostgreSQL" in output
        assert "Better performance" in output

    def test_log_decision_with_all_options(self, mock_manager, mock_args):
        """Log a decision with all optional fields."""
        mock_args.decision = "Refactor authentication"
        mock_args.rationale = "Security improvements needed"
        mock_args.affects = ["task:T-001", "task:T-002"]
        mock_args.alternatives = ["Keep current", "Use third-party"]
        mock_args.file = "auth.py"

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_log(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "task:T-001" in output
        assert "Keep current" in output


class TestDecisionList:
    """Tests for decision list command."""

    def test_list_decisions_empty(self, mock_manager, mock_args):
        """List decisions when none exist."""
        mock_manager.get_decisions.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_list(mock_args, mock_manager)

        assert result == 0
        assert "No decisions" in captured.getvalue()

    def test_list_decisions_with_data(self, mock_manager, mock_args):
        """List decisions when they exist."""
        decision1 = Mock()
        decision1.id = "decision:D-001"
        decision1.content = "Use microservices"
        decision1.properties = {
            "rationale": "Better scalability",
            "alternatives": ["Monolith", "Serverless"]
        }

        decision2 = Mock()
        decision2.id = "decision:D-002"
        decision2.content = "Use TypeScript"
        decision2.properties = {"rationale": "Type safety"}

        mock_manager.get_decisions.return_value = [decision1, decision2]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_list(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "decision:D-001" in output
        assert "Use microservices" in output
        assert "Better scalability" in output


class TestDecisionWhy:
    """Tests for decision why command."""

    def test_why_no_decisions(self, mock_manager, mock_args):
        """Query why with no affecting decisions."""
        mock_args.task_id = "task:T-001"
        mock_manager.why.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_why(mock_args, mock_manager)

        assert result == 0
        assert "No decisions found" in captured.getvalue()

    def test_why_with_decisions(self, mock_manager, mock_args):
        """Query why with affecting decisions."""
        mock_args.task_id = "task:T-001"
        mock_manager.why.return_value = [
            {
                "decision_id": "decision:D-001",
                "decision": "Refactor API",
                "rationale": "Improve maintainability",
                "alternatives": ["Keep as-is"]
            }
        ]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_why(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "decision:D-001" in output
        assert "Refactor API" in output


# =============================================================================
# HANDOFF COMMAND TESTS
# =============================================================================


class TestHandoffInitiate:
    """Tests for handoff initiate command.

    Updated to use manager methods (TX backend) instead of EventLog/HandoffManager.
    """

    def test_initiate_handoff_basic(self, mock_manager, mock_args):
        """Initiate a basic handoff."""
        mock_args.task_id = "task:T-001"
        mock_args.source = "main"
        mock_args.target = "sub-agent-1"
        mock_args.instructions = ""

        # Mock task retrieval
        task = Mock()
        task.id = "task:T-001"
        task.content = "Fix authentication"
        task.properties = {"status": "pending", "priority": "high"}

        mock_manager.get_task.return_value = task
        mock_manager.initiate_handoff.return_value = "handoff:H-001"

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_initiate(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "handoff:H-001" in output or "H-001" in output
        assert "main" in output and "sub-agent-1" in output
        mock_manager.initiate_handoff.assert_called_once()

    def test_initiate_handoff_nonexistent_task(self, mock_manager, mock_args):
        """Initiate handoff for nonexistent task fails."""
        mock_args.task_id = "task:T-999"
        mock_args.source = "main"
        mock_args.target = "sub-agent-1"
        mock_args.instructions = ""

        mock_manager.get_task.return_value = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_initiate(mock_args, mock_manager)

        assert result == 1
        assert "not found" in captured.getvalue()


class TestHandoffAccept:
    """Tests for handoff accept command.

    Updated to use manager.accept_handoff() instead of HandoffManager.
    """

    def test_accept_handoff(self, mock_manager, mock_args):
        """Accept a handoff."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.message = "Acknowledged"

        mock_manager.accept_handoff.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_accept(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "H-001" in output
        assert "sub-agent-1" in output
        mock_manager.accept_handoff.assert_called_once_with(
            handoff_id="handoff:H-001",
            agent="sub-agent-1",
            acknowledgment="Acknowledged"
        )

    def test_accept_handoff_failure(self, mock_manager, mock_args):
        """Accept handoff returns error on failure."""
        mock_args.handoff_id = "handoff:H-NONEXISTENT"
        mock_args.agent = "sub-agent-1"
        mock_args.message = ""

        mock_manager.accept_handoff.return_value = False

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_accept(mock_args, mock_manager)

        assert result == 1
        assert "Failed" in captured.getvalue()


class TestHandoffComplete:
    """Tests for handoff complete command.

    Updated to use manager.complete_handoff() instead of HandoffManager.
    """

    def test_complete_handoff_with_json_result(self, mock_manager, mock_args):
        """Complete a handoff with JSON result."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.result = '{"status": "done", "files": ["auth.py"]}'
        mock_args.artifacts = ["commit:abc123"]

        mock_manager.complete_handoff.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_complete(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "H-001" in output
        assert "done" in output
        mock_manager.complete_handoff.assert_called_once()
        # Verify JSON was parsed
        call_kwargs = mock_manager.complete_handoff.call_args[1]
        assert call_kwargs["result"] == {"status": "done", "files": ["auth.py"]}

    def test_complete_handoff_with_invalid_json(self, mock_manager, mock_args):
        """Complete a handoff with invalid JSON falls back to message."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.result = "Task completed successfully"
        mock_args.artifacts = None

        mock_manager.complete_handoff.return_value = True

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_complete(mock_args, mock_manager)

        assert result == 0
        # Verify plain text was wrapped in dict
        call_kwargs = mock_manager.complete_handoff.call_args[1]
        assert call_kwargs["result"] == {"message": "Task completed successfully"}

    def test_complete_handoff_failure(self, mock_manager, mock_args):
        """Complete handoff returns error on failure."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.result = "{}"
        mock_args.artifacts = None

        mock_manager.complete_handoff.return_value = False

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_complete(mock_args, mock_manager)

        assert result == 1
        assert "Failed" in captured.getvalue()


class TestHandoffList:
    """Tests for handoff list command.

    Updated to use manager.list_handoffs() instead of EventLog/HandoffManager.
    """

    def test_list_handoffs_empty(self, mock_manager, mock_args):
        """List handoffs when none exist."""
        mock_args.status = None

        mock_manager.list_handoffs.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_list(mock_args, mock_manager)

        assert result == 0
        assert "No handoffs" in captured.getvalue()
        mock_manager.list_handoffs.assert_called_once_with(status=None)

    def test_list_handoffs_with_data(self, mock_manager, mock_args):
        """List handoffs when they exist."""
        mock_args.status = None

        handoffs = [
            {
                "id": "handoff:H-001",
                "source_agent": "main",
                "target_agent": "sub-agent-1",
                "task_id": "task:T-001",
                "status": "completed",
                "instructions": "Please review the code"
            }
        ]

        mock_manager.list_handoffs.return_value = handoffs

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_list(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "H-001" in output
        assert "main → sub-agent-1" in output

    def test_list_handoffs_filtered_by_status(self, mock_manager, mock_args):
        """List handoffs filtered by status."""
        mock_args.status = "initiated"

        filtered_handoffs = [
            {"id": "H-001", "status": "initiated", "source_agent": "main", "target_agent": "sub-agent-1"},
        ]

        mock_manager.list_handoffs.return_value = filtered_handoffs

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_list(mock_args, mock_manager)

        assert result == 0
        mock_manager.list_handoffs.assert_called_once_with(status="initiated")


# =============================================================================
# QUERY COMMAND TESTS
# =============================================================================


class TestQuery:
    """Tests for query command."""

    def test_query_pending_tasks(self, mock_manager, mock_args):
        """Query for pending tasks."""
        mock_args.query_string = ["pending", "tasks"]
        # Query results can be dicts with id and title
        mock_manager.query.return_value = [
            {"id": "task:T-001", "title": "Task 1"},
            {"id": "task:T-002", "title": "Task 2"}
        ]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_query(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Query: pending tasks" in output

    def test_query_no_results(self, mock_manager, mock_args):
        """Query with no results."""
        mock_args.query_string = ["blocked", "tasks"]
        mock_manager.query.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_query(mock_args, mock_manager)

        assert result == 0
        assert "No results" in captured.getvalue()


# =============================================================================
# INFER COMMAND TESTS
# =============================================================================


class TestInfer:
    """Tests for edge inference command."""

    def test_infer_from_recent_commits(self, mock_manager, mock_args):
        """Infer edges from recent commits."""
        mock_args.commits = 10
        mock_args.message = None

        edges = [
            {
                "type": "implements",
                "commit_hash": "abc123",
                "commit": "abc123",
                "task": "task:T-001"
            }
        ]
        mock_manager.infer_edges_from_recent_commits.return_value = edges

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_infer(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Analyzed last 10 commits" in output
        assert "implements" in output

    def test_infer_from_specific_message(self, mock_manager, mock_args):
        """Infer edges from specific commit message."""
        mock_args.commits = 10
        mock_args.message = "feat: Implement task:T-001"

        edges = [
            {"type": "implements", "from": "commit", "to": "task:T-001"}
        ]
        mock_manager.infer_edges_from_commit.return_value = edges

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_infer(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Analyzing message:" in output

    def test_infer_no_edges_found(self, mock_manager, mock_args):
        """Infer with no edges found."""
        mock_args.commits = 5
        mock_args.message = None

        mock_manager.infer_edges_from_recent_commits.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_infer(mock_args, mock_manager)

        assert result == 0
        assert "No task references found" in captured.getvalue()


# =============================================================================
# STATS COMMAND TESTS
# =============================================================================


class TestStats:
    """Tests for stats command."""

    def test_stats_basic(self, mock_manager, mock_args):
        """Show basic statistics."""
        stats = {
            "total_tasks": 10,
            "total_sprints": 2,
            "total_epics": 1,
            "total_edges": 15,
            "tasks_by_status": {
                "pending": 5,
                "in_progress": 3,
                "completed": 2
            }
        }
        mock_manager.get_stats.return_value = stats

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_stats(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "Total tasks: 10" in output
        assert "Total sprints: 2" in output
        assert "pending: 5" in output
        assert "completed: 2" in output


# =============================================================================
# SPRINT COMMAND TESTS
# =============================================================================


class TestSprintCreate:
    """Tests for sprint create command."""

    def test_create_basic_sprint(self, mock_manager, mock_args):
        """Create a basic sprint."""
        mock_args.name = "Sprint 1"
        mock_args.number = None
        mock_args.epic = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_sprint_create(mock_args, mock_manager)

        assert result == 0
        mock_manager.create_sprint.assert_called_once_with(
            name="Sprint 1",
            number=None,
            epic_id=None
        )
        assert "Created:" in captured.getvalue()

    def test_create_sprint_with_number(self, mock_manager, mock_args):
        """Create a sprint with number."""
        mock_args.name = "Q4 Sprint"
        mock_args.number = 42
        mock_args.epic = "epic:E-abc123"

        result = cmd_sprint_create(mock_args, mock_manager)

        assert result == 0
        mock_manager.create_sprint.assert_called_once_with(
            name="Q4 Sprint",
            number=42,
            epic_id="epic:E-abc123"
        )


class TestSprintList:
    """Tests for sprint list command."""

    def test_list_sprints_empty(self, mock_manager, mock_args):
        """List sprints when none exist."""
        mock_args.status = None
        mock_manager.list_sprints.return_value = []

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_sprint_list(mock_args, mock_manager)

        assert result == 0
        assert "No sprints" in captured.getvalue()

    def test_list_sprints_with_data(self, mock_manager, mock_args):
        """List sprints when they exist."""
        mock_args.status = None

        sprint = Mock()
        sprint.id = "sprint:S-001"
        sprint.content = "Sprint 1"
        sprint.properties = {"status": "active"}

        mock_manager.list_sprints.return_value = [sprint]
        mock_manager.get_sprint_progress.return_value = {"progress_percent": 60.0}

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_sprint_list(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "sprint:S-001" in output
        assert "60%" in output


class TestSprintStatus:
    """Tests for sprint status command."""

    def test_sprint_status_current(self, mock_args):
        """Show current sprint status."""
        mock_args.sprint_id = None

        sprint = Mock()
        sprint.id = "sprint:S-001"
        sprint.content = "Current Sprint"
        sprint.properties = {"status": "active"}

        # Create a less strict mock for this test
        mock_manager = Mock()
        # When sprint_id is None, it calls list_sprints
        mock_manager.list_sprints.return_value = [sprint]
        mock_manager.get_sprint_progress.return_value = {
            "total_tasks": 10,
            "completed_tasks": 6,
            "in_progress_tasks": 3,
            "blocked_tasks": 1,
            "progress_percent": 60.0
        }

        with patch('got_utils.format_sprint_status', return_value="Sprint Status"):
            with patch('sys.stdout', new=StringIO()) as captured:
                result = cmd_sprint_status(mock_args, mock_manager)

        # The actual implementation may vary, check it exists
        assert result == 0


# =============================================================================
# ID GENERATION TESTS
# =============================================================================


class TestIdGeneration:
    """Tests for ID generation functions."""

    def test_generate_task_id_format(self):
        """Task ID has correct format T-YYYYMMDD-HHMMSS-XXXXXXXX."""
        task_id = generate_task_id()
        assert task_id.startswith("T-")
        parts = task_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # hex suffix

    def test_generate_sprint_id_with_number(self):
        """Sprint ID with number has correct format."""
        sprint_id = generate_sprint_id(5)
        assert sprint_id == "S-005"

    def test_generate_sprint_id_without_number(self):
        """Sprint ID without number uses date format."""
        sprint_id = generate_sprint_id()
        assert sprint_id.startswith("S-")
        # Format: S-YYYY-MM
        parts = sprint_id.split("-")
        assert len(parts) == 3

    def test_generate_decision_id_format(self):
        """Decision ID has correct format D-YYYYMMDD-HHMMSS-XXXXXXXX."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("D-")
        parts = decision_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # hex suffix


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in CLI commands."""

    def test_task_start_handles_exception(self, mock_manager, mock_args):
        """Task start handles exceptions gracefully."""
        mock_args.task_id = "task:T-001"
        mock_manager.start_task.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            cmd_task_start(mock_args, mock_manager)

    def test_invalid_task_id_format(self, mock_manager, mock_args):
        """Commands handle invalid task ID format."""
        mock_args.task_id = "invalid-id"
        mock_manager.get_task.return_value = None

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_handoff_initiate(mock_args, mock_manager)

        assert result == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    def test_task_workflow(self, mock_manager, mock_args):
        """Test complete task workflow: create -> start -> complete."""
        # Create task
        mock_args.title = "Test task"
        mock_args.priority = "medium"
        mock_args.category = "test"
        mock_args.description = ""
        mock_args.sprint = None
        mock_args.depends = None

        result = cmd_task_create(mock_args, mock_manager)
        assert result == 0

        # Start task
        mock_args.task_id = "task:T-001"
        mock_manager.start_task.return_value = True
        result = cmd_task_start(mock_args, mock_manager)
        assert result == 0

        # Complete task
        mock_args.retrospective = "Test complete"
        mock_manager.complete_task.return_value = True
        result = cmd_task_complete(mock_args, mock_manager)
        assert result == 0

    def test_decision_affects_task(self, mock_manager, mock_args):
        """Test decision logging that affects a task."""
        # Log decision affecting task
        mock_args.decision = "Use new API"
        mock_args.rationale = "Better performance"
        mock_args.affects = ["task:T-001"]
        mock_args.alternatives = None
        mock_args.file = None

        result = cmd_decision_log(mock_args, mock_manager)
        assert result == 0

        # Query why task exists
        mock_args.task_id = "task:T-001"
        mock_manager.why.return_value = [
            {
                "decision_id": "decision:D-001",
                "decision": "Use new API",
                "rationale": "Better performance",
                "alternatives": []
            }
        ]

        with patch('sys.stdout', new=StringIO()) as captured:
            result = cmd_decision_why(mock_args, mock_manager)

        assert result == 0
        assert "Use new API" in captured.getvalue()


# =============================================================================
# REBUILD TELEMETRY TESTS
# =============================================================================


class TestRebuildTelemetry:
    """
    Unit tests for rebuild_graph_from_events telemetry feature.

    Task T-20251221-020047-ecf6: Add rebuild validation with telemetry.
    These tests verify that edge counts are validated after event replay.
    """

    def test_rebuild_returns_telemetry(self, temp_got_dir):
        """Rebuild should return telemetry with graph and stats."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Task 1"}, "meta": {}},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)

        # Should return a dict with graph and telemetry
        assert isinstance(result, dict), "with_telemetry=True should return dict"
        assert "graph" in result, "Result should contain 'graph'"
        assert "telemetry" in result, "Result should contain 'telemetry'"
        assert isinstance(result["graph"], ThoughtGraph), "graph should be ThoughtGraph"

    def test_telemetry_counts_nodes_created(self, temp_got_dir):
        """Telemetry should count nodes created."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Task 1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "Task 2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-003", "type": "TASK", "data": {"title": "Task 3"}, "meta": {}},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["nodes_created"] == 3, "Should count 3 nodes created"
        assert telemetry["node_create_events"] == 3, "Should count 3 node.create events"

    def test_telemetry_counts_edges_created(self, temp_got_dir):
        """Telemetry should count edges created vs edge events."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Source"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "Target"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.create", "src": "task:T-001", "tgt": "task:T-002", "type": "DEPENDS_ON", "weight": 1.0},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["edge_create_events"] == 1, "Should count 1 edge.create event"
        assert telemetry["edges_created"] == 1, "Should count 1 edge created"
        assert telemetry["edges_skipped"] == 0, "No edges should be skipped"

    def test_telemetry_tracks_skipped_edges(self, temp_got_dir):
        """Telemetry should track edges skipped due to missing nodes."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Source"}, "meta": {}},
            # T-002 NOT created - edge should be skipped
            {"ts": "2025-01-01T00:00:01Z", "event": "edge.create", "src": "task:T-001", "tgt": "task:T-002", "type": "DEPENDS_ON", "weight": 1.0},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["edge_create_events"] == 1, "Should count 1 edge.create event"
        assert telemetry["edges_created"] == 0, "No edges should be created (target missing)"
        assert telemetry["edges_skipped"] == 1, "1 edge should be skipped"

    def test_telemetry_edge_validation_passes(self, temp_got_dir):
        """Telemetry should validate: edges_created == edges expected (no skips)."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "A"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "B"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.create", "src": "task:T-001", "tgt": "task:T-002", "type": "DEPENDS_ON", "weight": 1.0},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["validation_passed"] is True, "Validation should pass when all edges created"

    def test_telemetry_edge_validation_fails_on_skips(self, temp_got_dir):
        """Telemetry should fail validation when edges are skipped."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "A"}, "meta": {}},
            # Missing target node
            {"ts": "2025-01-01T00:00:01Z", "event": "edge.create", "src": "task:T-001", "tgt": "task:T-MISSING", "type": "DEPENDS_ON", "weight": 1.0},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["validation_passed"] is False, "Validation should fail when edges skipped"
        assert len(telemetry["validation_errors"]) > 0, "Should have validation errors"

    def test_telemetry_tracks_comma_split_edges(self, temp_got_dir):
        """Telemetry should correctly count edges from comma-separated IDs."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "S"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "T1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-003", "type": "TASK", "data": {"title": "T2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:03Z",
                "event": "edge.create",
                "src": "task:T-001",
                "tgt": "task:T-002,task:T-003",  # Comma-separated
                "type": "DEPENDS_ON",
                "weight": 1.0
            },
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["edge_create_events"] == 1, "Should count 1 edge.create event"
        assert telemetry["edges_created"] == 2, "Should create 2 edges from comma-split"

    def test_telemetry_tracks_errors(self, temp_got_dir):
        """Telemetry should track processing errors."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create"},  # Missing required fields
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert telemetry["errors"] > 0, "Should count errors"

    def test_backward_compatible_without_telemetry(self, temp_got_dir):
        """Default behavior (no telemetry flag) returns just the graph."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Task 1"}, "meta": {}},
        ]

        result = EventLog.rebuild_graph_from_events(events)

        # Should return just the graph (backward compatible)
        assert isinstance(result, ThoughtGraph), "Default should return ThoughtGraph directly"

    def test_telemetry_summary_string(self, temp_got_dir):
        """Telemetry should include a human-readable summary."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Task 1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "Task 2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.create", "src": "task:T-001", "tgt": "task:T-002", "type": "DEPENDS_ON", "weight": 1.0},
        ]

        result = EventLog.rebuild_graph_from_events(events, with_telemetry=True)
        telemetry = result["telemetry"]

        assert "summary" in telemetry, "Telemetry should include summary"
        assert "nodes" in telemetry["summary"].lower(), "Summary should mention nodes"
        assert "edges" in telemetry["summary"].lower(), "Summary should mention edges"


class TestDecisionNodeType:
    """
    Unit tests for decision node type handling in rebuild_graph_from_events.

    Bug fix: Decisions were being created with NodeType.CONTEXT instead of
    NodeType.DECISION, causing the dashboard to show "Decisions: 0" despite
    19 decision events in the event log.
    """

    def test_decision_create_uses_decision_nodetype(self, temp_got_dir):
        """decision.create events should create nodes with NodeType.DECISION."""
        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "decision.create",
                "id": "decision:D-001",
                "decision": "Use TDD for all changes",
                "rationale": "Improves code quality",
                "affects": [],
                "alternatives": [],
            }
        ]

        graph = EventLog.rebuild_graph_from_events(events)

        assert "decision:D-001" in graph.nodes, "Decision node should exist"
        node = graph.nodes["decision:D-001"]
        assert node.node_type == NodeType.DECISION, \
            f"Node type should be DECISION, got {node.node_type}"

    def test_decision_count_in_rebuilt_graph(self, temp_got_dir):
        """Multiple decisions should all have NodeType.DECISION."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "decision.create", "id": "decision:D-001",
             "decision": "Decision 1", "rationale": "Reason 1", "affects": [], "alternatives": []},
            {"ts": "2025-01-01T00:00:01Z", "event": "decision.create", "id": "decision:D-002",
             "decision": "Decision 2", "rationale": "Reason 2", "affects": [], "alternatives": []},
            {"ts": "2025-01-01T00:00:02Z", "event": "decision.create", "id": "decision:D-003",
             "decision": "Decision 3", "rationale": "Reason 3", "affects": [], "alternatives": []},
        ]

        graph = EventLog.rebuild_graph_from_events(events)

        decision_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.DECISION]
        assert len(decision_nodes) == 3, f"Should have 3 DECISION nodes, got {len(decision_nodes)}"

    def test_decision_affects_creates_motivates_edges(self, temp_got_dir):
        """Decisions with 'affects' should create MOTIVATES edges."""
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001",
             "type": "TASK", "data": {"title": "Task 1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "decision.create", "id": "decision:D-001",
             "decision": "Choose approach A", "rationale": "Faster",
             "affects": ["task:T-001"], "alternatives": []},
        ]

        graph = EventLog.rebuild_graph_from_events(events)

        # Decision should exist with correct type
        assert graph.nodes["decision:D-001"].node_type == NodeType.DECISION

        # Should have MOTIVATES edge from decision to task
        motivates_edges = [e for e in graph.edges
                          if e.source_id == "decision:D-001" and e.edge_type == EdgeType.MOTIVATES]
        assert len(motivates_edges) == 1, "Should have 1 MOTIVATES edge"
        assert motivates_edges[0].target_id == "task:T-001"

    def test_decision_content_preserved(self, temp_got_dir):
        """Decision content and rationale should be preserved in node."""
        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "decision.create",
                "id": "decision:D-001",
                "decision": "Use event sourcing",
                "rationale": "Better for concurrency",
                "affects": [],
                "alternatives": ["Use CRUD", "Use state machine"],
            }
        ]

        graph = EventLog.rebuild_graph_from_events(events)
        node = graph.nodes["decision:D-001"]

        assert "event sourcing" in node.content.lower(), "Decision content should be in node"


class TestAutoTaskHook:
    """
    Unit tests for auto-task creation hook functionality.

    Task T-20251221-020101-afc4: Add auto-task creation hook.
    When committing without a task reference, prompt to create a GoT task.
    """

    def test_detect_task_reference_in_commit_message(self):
        """Commit message with task ID should be detected."""
        from got_utils import has_task_reference

        # Standard format with task prefix
        assert has_task_reference("fix: T-20251221-123456-abcd Fixed bug") is True
        assert has_task_reference("feat: Implement feature (T-20251220-111111-1111)") is True
        assert has_task_reference("[T-20251219-000000-0000] chore: cleanup") is True

    def test_detect_task_reference_case_insensitive(self):
        """Task reference detection should be case insensitive."""
        from got_utils import has_task_reference

        assert has_task_reference("fix: t-20251221-123456-abcd lowercase") is True

    def test_no_task_reference_detected(self):
        """Commit message without task ID should return False."""
        from got_utils import has_task_reference

        assert has_task_reference("fix: Fixed a bug") is False
        assert has_task_reference("chore: Update dependencies") is False
        assert has_task_reference("feat: Add new feature") is False

    def test_malformed_task_reference_not_detected(self):
        """Malformed task IDs should not be detected."""
        from got_utils import has_task_reference

        # Too short
        assert has_task_reference("fix: T-2025-123") is False
        # Missing segments
        assert has_task_reference("fix: T-20251221-abcd") is False
        # Wrong prefix
        assert has_task_reference("fix: X-20251221-123456-abcd") is False

    def test_extract_commit_type_for_task_category(self):
        """Extract commit type prefix to suggest task category."""
        from got_utils import extract_commit_type

        assert extract_commit_type("fix: Fixed something") == "fix"
        assert extract_commit_type("feat: Added feature") == "feat"
        assert extract_commit_type("chore: Cleanup") == "chore"
        assert extract_commit_type("docs: Updated readme") == "docs"
        assert extract_commit_type("refactor: Improved code") == "refactor"
        assert extract_commit_type("test: Added tests") == "test"

    def test_extract_commit_type_no_prefix(self):
        """Commit without conventional prefix should return None."""
        from got_utils import extract_commit_type

        assert extract_commit_type("Just a plain message") is None
        assert extract_commit_type("Updated something") is None

    def test_suggest_task_category_from_commit_type(self):
        """Map commit type to GoT task category."""
        from got_utils import suggest_task_category

        assert suggest_task_category("fix") == "bugfix"
        assert suggest_task_category("feat") == "feature"
        assert suggest_task_category("docs") == "docs"
        assert suggest_task_category("refactor") == "refactor"
        assert suggest_task_category("test") == "testing"
        assert suggest_task_category("chore") == "chore"
        assert suggest_task_category(None) == "general"

    def test_generate_task_title_from_commit(self):
        """Generate a task title from commit message."""
        from got_utils import generate_task_title_from_commit

        assert generate_task_title_from_commit("fix: Fixed login bug") == "Fixed login bug"
        assert generate_task_title_from_commit("feat: Add user auth") == "Add user auth"
        assert generate_task_title_from_commit("Plain message") == "Plain message"


class TestTaskNextCommand:
    """
    Unit tests for 'got task next' command.

    Task T-20251221-112230-db4a: Add 'got task next' command.
    Picks highest priority pending task that isn't blocked.
    """

    def test_next_returns_highest_priority_pending(self, temp_got_dir):
        """Should return high priority task before medium/low."""
        manager = GoTProjectManager(got_dir=temp_got_dir)

        # Create tasks with different priorities
        low_id = manager.create_task("Low priority task", priority="low")
        medium_id = manager.create_task("Medium priority task", priority="medium")
        high_id = manager.create_task("High priority task", priority="high")

        result = manager.get_next_task()

        assert result is not None, "Should return a task"
        assert result["id"] == f"task:{high_id}" or result["id"] == high_id
        assert "High priority" in result["title"]

    def test_next_skips_in_progress_tasks(self, temp_got_dir):
        """Should skip tasks that are already in progress."""
        manager = GoTProjectManager(got_dir=temp_got_dir)

        high_id = manager.create_task("In progress task", priority="high")
        manager.start_task(high_id)
        medium_id = manager.create_task("Pending task", priority="medium")

        result = manager.get_next_task()

        assert result is not None
        assert "Pending task" in result["title"]

    def test_next_skips_completed_tasks(self, temp_got_dir):
        """Should skip completed tasks."""
        manager = GoTProjectManager(got_dir=temp_got_dir)

        high_id = manager.create_task("Completed task", priority="high")
        manager.start_task(high_id)
        manager.complete_task(high_id)
        medium_id = manager.create_task("Pending task", priority="medium")

        result = manager.get_next_task()

        assert result is not None
        assert "Pending task" in result["title"]

    def test_next_skips_blocked_tasks(self, temp_got_dir):
        """Should skip tasks that are blocked."""
        manager = GoTProjectManager(got_dir=temp_got_dir)

        blocker_id = manager.create_task("Blocker task", priority="high")
        blocked_id = manager.create_task("Blocked task", priority="high")
        manager.block_task(blocked_id, reason="Depends on blocker", blocker_id=blocker_id)

        unblocked_id = manager.create_task("Unblocked task", priority="medium")

        result = manager.get_next_task()

        assert result is not None
        # Should return either the blocker (unblocked high) or unblocked medium
        assert "Blocked task" not in result["title"]

    def test_next_returns_none_when_no_pending(self, temp_got_dir):
        """Should return None when no pending tasks exist."""
        manager = GoTProjectManager(got_dir=temp_got_dir)

        # Create and complete a task
        task_id = manager.create_task("Done task", priority="high")
        manager.start_task(task_id)
        manager.complete_task(task_id)

        result = manager.get_next_task()

        assert result is None

    def test_next_returns_oldest_within_same_priority(self, temp_got_dir):
        """When priority is equal, should return oldest task first."""
        manager = GoTProjectManager(got_dir=temp_got_dir)

        first_id = manager.create_task("First high task", priority="high")
        second_id = manager.create_task("Second high task", priority="high")

        result = manager.get_next_task()

        assert result is not None
        assert "First high" in result["title"]


# =============================================================================
# AUTO-COMMIT/PUSH TESTS
# =============================================================================


class TestGotAutoCommit:
    """Tests for got_auto_commit function."""

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', False)
    def test_disabled_returns_false(self):
        """Auto-commit returns False when disabled."""
        result = got_auto_commit("task", "create")
        assert result is False

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', True)
    def test_non_mutating_command_returns_false(self):
        """Auto-commit returns False for non-mutating commands."""
        result = got_auto_commit("dashboard", None)
        assert result is False

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', True)
    def test_invalid_subcommand_returns_false(self):
        """Auto-commit returns False for invalid subcommand."""
        result = got_auto_commit("task", "nonexistent")
        assert result is False

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', True)
    @patch('subprocess.run')
    def test_mutating_command_commits(self, mock_run):
        """Auto-commit triggers commit for mutating commands."""
        # Mock: git add succeeds, git diff --cached shows changes, git commit succeeds
        def run_side_effect(*args, **kwargs):
            if 'diff' in args[0] and '--cached' in args[0]:
                # Return non-zero to indicate there are staged changes
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        result = got_auto_commit("task", "create")

        assert result is True
        # Verify commit was called
        commit_calls = [c for c in mock_run.call_args_list if 'commit' in str(c)]
        assert len(commit_calls) >= 1

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', True)
    @patch('subprocess.run')
    def test_no_changes_returns_false(self, mock_run):
        """Auto-commit returns False when no changes to commit."""
        # Mock: git diff --cached shows no changes (returncode=0)
        mock_run.return_value = MagicMock(returncode=0)

        result = got_auto_commit("task", "complete")

        assert result is False

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', True)
    @patch('subprocess.run')
    def test_git_error_returns_false(self, mock_run):
        """Auto-commit returns False on git error."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, ['git', 'commit'])

        result = got_auto_commit("task", "create")

        assert result is False

    @patch('got_utils.GOT_AUTO_COMMIT_ENABLED', True)
    @patch('got_utils.GOT_AUTO_PUSH_ENABLED', True)
    @patch('got_utils._got_auto_push')
    @patch('subprocess.run')
    def test_auto_push_called_after_commit(self, mock_run, mock_push):
        """Auto-push is called after successful commit when enabled."""
        def run_side_effect(*args, **kwargs):
            if 'diff' in args[0] and '--cached' in args[0]:
                return MagicMock(returncode=1)  # Has changes
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect
        mock_push.return_value = True

        result = got_auto_commit("task", "complete")

        assert result is True
        mock_push.assert_called_once()


class TestGotAutoPush:
    """Tests for _got_auto_push function."""

    @patch('subprocess.run')
    def test_protected_branch_skips_push(self, mock_run):
        """Auto-push skips protected branches (main, master, etc.)."""
        mock_run.return_value = MagicMock(stdout='main\n', returncode=0)

        result = _got_auto_push()

        assert result is False
        # Should only call git rev-parse, not git push
        push_calls = [c for c in mock_run.call_args_list if 'push' in str(c)]
        assert len(push_calls) == 0

    @patch('subprocess.run')
    def test_non_claude_branch_skips_push(self, mock_run):
        """Auto-push skips non-claude/* branches."""
        mock_run.return_value = MagicMock(stdout='feature/my-feature\n', returncode=0)

        result = _got_auto_push()

        assert result is False

    @patch('subprocess.run')
    def test_claude_branch_pushes(self, mock_run):
        """Auto-push works on claude/* branches."""
        def run_side_effect(*args, **kwargs):
            if 'rev-parse' in args[0]:
                return MagicMock(stdout='claude/test-session-abc\n', returncode=0)
            if 'push' in args[0]:
                return MagicMock(returncode=0, stderr='')
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        result = _got_auto_push()

        assert result is True
        push_calls = [c for c in mock_run.call_args_list if 'push' in str(c)]
        assert len(push_calls) == 1

    @patch('subprocess.run')
    def test_network_error_retries(self, mock_run):
        """Auto-push retries on network errors."""
        call_count = [0]

        def run_side_effect(*args, **kwargs):
            if 'rev-parse' in args[0]:
                return MagicMock(stdout='claude/test-session\n', returncode=0)
            if 'push' in args[0]:
                call_count[0] += 1
                if call_count[0] < 3:
                    return MagicMock(returncode=1, stderr='network error')
                return MagicMock(returncode=0, stderr='')
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        result = _got_auto_push()

        assert result is True
        assert call_count[0] == 3  # Retried until success

    @patch('subprocess.run')
    def test_timeout_returns_false(self, mock_run):
        """Auto-push returns False on timeout."""
        import subprocess

        def run_side_effect(*args, **kwargs):
            if 'rev-parse' in args[0]:
                return MagicMock(stdout='claude/test-session\n', returncode=0)
            if 'push' in args[0]:
                raise subprocess.TimeoutExpired(args[0], 30)
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        result = _got_auto_push()

        assert result is False

    @patch('subprocess.run')
    def test_all_protected_branches(self, mock_run):
        """Verify all protected branches are skipped."""
        for branch in PROTECTED_BRANCHES:
            mock_run.return_value = MagicMock(stdout=f'{branch}\n', returncode=0)
            result = _got_auto_push()
            assert result is False, f"Branch {branch} should be protected"


class TestMutatingCommands:
    """Tests for MUTATING_COMMANDS configuration."""

    def test_task_mutating_subcommands(self):
        """Task subcommands that mutate are in the list."""
        task_mutating = MUTATING_COMMANDS.get("task")
        assert "create" in task_mutating
        assert "start" in task_mutating
        assert "complete" in task_mutating
        assert "block" in task_mutating
        assert "delete" in task_mutating
        assert "depends" in task_mutating

    def test_sprint_mutating_subcommands(self):
        """Sprint subcommands that mutate are in the list."""
        sprint_mutating = MUTATING_COMMANDS.get("sprint")
        assert "create" in sprint_mutating
        assert "start" in sprint_mutating
        assert "complete" in sprint_mutating

    def test_always_mutating_commands(self):
        """Commands like compact/migrate are always mutating."""
        assert MUTATING_COMMANDS.get("compact") is True
        assert MUTATING_COMMANDS.get("migrate") is True

    def test_dashboard_not_mutating(self):
        """Dashboard and query commands are not mutating."""
        assert MUTATING_COMMANDS.get("dashboard") is None
        assert MUTATING_COMMANDS.get("query") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
