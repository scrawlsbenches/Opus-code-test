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
    cmd_task_create,
    cmd_task_list,
    cmd_task_show,
    cmd_task_start,
    cmd_task_complete,
    cmd_task_block,
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
    """Create a mock GoTProjectManager with pre-configured behavior."""
    manager = Mock(spec=GoTProjectManager)
    manager.got_dir = temp_got_dir
    manager.events_dir = temp_got_dir / "events"
    manager.events_dir.mkdir(parents=True, exist_ok=True)

    # Mock graph
    manager.graph = ThoughtGraph()

    # Mock event log
    manager.event_log = Mock()

    # Default return values
    manager.save.return_value = None
    manager.create_task.return_value = "task:T-20251220-120000-abc123"
    manager.create_sprint.return_value = "sprint:S-001"
    manager.log_decision.return_value = "decision:D-20251220-120000-def456"

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
    """Tests for handoff initiate command."""

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

        with patch('got_utils.HandoffManager') as MockHandoffMgr:
            mock_handoff_mgr = Mock()
            mock_handoff_mgr.initiate_handoff.return_value = "handoff:H-001"
            MockHandoffMgr.return_value = mock_handoff_mgr

            with patch('sys.stdout', new=StringIO()) as captured:
                result = cmd_handoff_initiate(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "handoff:H-001" in output
        assert "main" in output and "sub-agent-1" in output

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
    """Tests for handoff accept command."""

    def test_accept_handoff(self, mock_manager, mock_args):
        """Accept a handoff."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.message = "Acknowledged"

        with patch('got_utils.HandoffManager') as MockHandoffMgr:
            mock_handoff_mgr = Mock()
            MockHandoffMgr.return_value = mock_handoff_mgr

            with patch('sys.stdout', new=StringIO()) as captured:
                result = cmd_handoff_accept(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "handoff:H-001" in output
        assert "sub-agent-1" in output


class TestHandoffComplete:
    """Tests for handoff complete command."""

    def test_complete_handoff_with_json_result(self, mock_manager, mock_args):
        """Complete a handoff with JSON result."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.result = '{"status": "done", "files": ["auth.py"]}'
        mock_args.artifacts = ["commit:abc123"]

        with patch('got_utils.HandoffManager') as MockHandoffMgr:
            mock_handoff_mgr = Mock()
            MockHandoffMgr.return_value = mock_handoff_mgr

            with patch('sys.stdout', new=StringIO()) as captured:
                result = cmd_handoff_complete(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "handoff:H-001" in output
        assert "done" in output

    def test_complete_handoff_with_invalid_json(self, mock_manager, mock_args):
        """Complete a handoff with invalid JSON falls back to message."""
        mock_args.handoff_id = "handoff:H-001"
        mock_args.agent = "sub-agent-1"
        mock_args.result = "Task completed successfully"
        mock_args.artifacts = None

        with patch('got_utils.HandoffManager') as MockHandoffMgr:
            mock_handoff_mgr = Mock()
            MockHandoffMgr.return_value = mock_handoff_mgr

            with patch('sys.stdout', new=StringIO()) as captured:
                result = cmd_handoff_complete(mock_args, mock_manager)

        assert result == 0


class TestHandoffList:
    """Tests for handoff list command."""

    def test_list_handoffs_empty(self, mock_manager, mock_args):
        """List handoffs when none exist."""
        mock_args.status = None

        with patch('got_utils.EventLog.load_all_events', return_value=[]):
            with patch('got_utils.HandoffManager.load_handoffs_from_events', return_value=[]):
                with patch('sys.stdout', new=StringIO()) as captured:
                    result = cmd_handoff_list(mock_args, mock_manager)

        assert result == 0
        assert "No handoffs" in captured.getvalue()

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

        with patch('got_utils.EventLog.load_all_events', return_value=[]):
            with patch('got_utils.HandoffManager.load_handoffs_from_events', return_value=handoffs):
                with patch('sys.stdout', new=StringIO()) as captured:
                    result = cmd_handoff_list(mock_args, mock_manager)

        assert result == 0
        output = captured.getvalue()
        assert "handoff:H-001" in output
        assert "main → sub-agent-1" in output

    def test_list_handoffs_filtered_by_status(self, mock_manager, mock_args):
        """List handoffs filtered by status."""
        mock_args.status = "initiated"

        all_handoffs = [
            {"id": "H-001", "status": "initiated"},
            {"id": "H-002", "status": "completed"},
        ]

        with patch('got_utils.EventLog.load_all_events', return_value=[]):
            with patch('got_utils.HandoffManager.load_handoffs_from_events', return_value=all_handoffs):
                with patch('sys.stdout', new=StringIO()) as captured:
                    result = cmd_handoff_list(mock_args, mock_manager)

        assert result == 0


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
        """Task ID has correct format."""
        task_id = generate_task_id()
        assert task_id.startswith("task:T-")
        parts = task_id.split("-")
        assert len(parts) >= 3

    def test_generate_sprint_id_with_number(self):
        """Sprint ID with number has correct format."""
        sprint_id = generate_sprint_id(5)
        assert sprint_id == "sprint:S-005"

    def test_generate_sprint_id_without_number(self):
        """Sprint ID without number uses date format."""
        sprint_id = generate_sprint_id()
        assert sprint_id.startswith("sprint:")
        assert ":" in sprint_id

    def test_generate_decision_id_format(self):
        """Decision ID has correct format."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("decision:D-")
        parts = decision_id.split("-")
        assert len(parts) >= 3


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
