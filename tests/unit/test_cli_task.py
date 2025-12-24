"""
Unit tests for cortical.got.cli.task module.

Tests use mocked GoTProjectManager to avoid file system operations.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace

from cortical.got.cli.task import (
    cmd_task_create,
    cmd_task_list,
    cmd_task_next,
    cmd_task_show,
    cmd_task_start,
    cmd_task_complete,
    cmd_task_block,
    cmd_task_depends,
    cmd_task_delete,
    setup_task_parser,
    handle_task_command,
)
from cortical.reasoning.graph_of_thought import ThoughtNode, NodeType


class TestCmdTaskCreate(unittest.TestCase):
    """Test cmd_task_create command handler."""

    def test_create_basic_task(self):
        """Test creating a basic task with minimal arguments."""
        mock_manager = Mock()
        mock_manager.create_task.return_value = "T-20251223-001"

        args = Namespace(
            title="Test task",
            priority="medium",
            category="feature",
            description="",
            sprint=None,
            depends=None,
            blocks=None,
        )

        with patch('builtins.print') as mock_print:
            result = cmd_task_create(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.create_task.assert_called_once_with(
            title="Test task",
            priority="medium",
            category="feature",
            description="",
            sprint_id=None,
            depends_on=None,
            blocks=None,
        )
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with("Created: T-20251223-001")

    def test_create_task_with_high_priority(self):
        """Test creating a high-priority task."""
        mock_manager = Mock()
        mock_manager.create_task.return_value = "T-20251223-002"

        args = Namespace(
            title="Critical bug fix",
            priority="high",
            category="bugfix",
            description="Fix security issue",
            sprint="S-001",
            depends=None,
            blocks=None,
        )

        with patch('builtins.print'):
            result = cmd_task_create(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.create_task.assert_called_once()
        call_args = mock_manager.create_task.call_args[1]
        self.assertEqual(call_args['priority'], 'high')
        self.assertEqual(call_args['category'], 'bugfix')
        self.assertEqual(call_args['description'], 'Fix security issue')
        self.assertEqual(call_args['sprint_id'], 'S-001')

    def test_create_task_with_dependencies(self):
        """Test creating a task with dependencies."""
        mock_manager = Mock()
        mock_manager.create_task.return_value = "T-20251223-003"

        args = Namespace(
            title="Follow-up task",
            priority="medium",
            category="feature",
            description="",
            sprint=None,
            depends=["T-001", "T-002"],
            blocks=["T-004"],
        )

        with patch('builtins.print'):
            result = cmd_task_create(args, mock_manager)

        self.assertEqual(result, 0)
        call_args = mock_manager.create_task.call_args[1]
        self.assertEqual(call_args['depends_on'], ["T-001", "T-002"])
        self.assertEqual(call_args['blocks'], ["T-004"])


class TestCmdTaskList(unittest.TestCase):
    """Test cmd_task_list command handler."""

    def test_list_all_tasks(self):
        """Test listing all tasks without filters."""
        mock_manager = Mock()
        task1 = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="First task",
            properties={"status": "pending", "priority": "high"}
        )
        task2 = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Second task",
            properties={"status": "in_progress", "priority": "medium"}
        )
        mock_manager.list_tasks.return_value = [task1, task2]

        args = Namespace(
            status=None,
            priority=None,
            category=None,
            sprint=None,
            blocked=False,
            json=False,
        )

        with patch('builtins.print') as mock_print:
            result = cmd_task_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_tasks.assert_called_once_with(
            status=None,
            priority=None,
            category=None,
            sprint_id=None,
            blocked_only=False,
        )
        # Should print formatted table
        print_output = str(mock_print.call_args)
        self.assertTrue(mock_print.called)

    def test_list_with_status_filter(self):
        """Test listing tasks filtered by status."""
        mock_manager = Mock()
        mock_manager.list_tasks.return_value = []

        args = Namespace(
            status="in_progress",
            priority=None,
            category=None,
            sprint=None,
            blocked=False,
            json=False,
        )

        with patch('builtins.print'):
            result = cmd_task_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_tasks.assert_called_once()
        call_args = mock_manager.list_tasks.call_args[1]
        self.assertEqual(call_args['status'], 'in_progress')

    def test_list_with_priority_filter(self):
        """Test listing tasks filtered by priority."""
        mock_manager = Mock()
        mock_manager.list_tasks.return_value = []

        args = Namespace(
            status=None,
            priority="high",
            category=None,
            sprint=None,
            blocked=False,
            json=False,
        )

        with patch('builtins.print'):
            result = cmd_task_list(args, mock_manager)

        self.assertEqual(result, 0)
        call_args = mock_manager.list_tasks.call_args[1]
        self.assertEqual(call_args['priority'], 'high')

    def test_list_blocked_only(self):
        """Test listing only blocked tasks."""
        mock_manager = Mock()
        mock_manager.list_tasks.return_value = []

        args = Namespace(
            status=None,
            priority=None,
            category=None,
            sprint=None,
            blocked=True,
            json=False,
        )

        with patch('builtins.print'):
            result = cmd_task_list(args, mock_manager)

        self.assertEqual(result, 0)
        call_args = mock_manager.list_tasks.call_args[1]
        self.assertTrue(call_args['blocked_only'])

    def test_list_json_output(self):
        """Test listing tasks with JSON output."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending", "priority": "medium"}
        )
        mock_manager.list_tasks.return_value = [task]

        args = Namespace(
            status=None,
            priority=None,
            category=None,
            sprint=None,
            blocked=False,
            json=True,
        )

        with patch('builtins.print') as mock_print:
            result = cmd_task_list(args, mock_manager)

        self.assertEqual(result, 0)
        # Should print JSON output
        self.assertTrue(mock_print.called)
        print_output = str(mock_print.call_args)
        # JSON output should contain the task data


class TestCmdTaskNext(unittest.TestCase):
    """Test cmd_task_next command handler."""

    def test_next_task_available(self):
        """Test getting next task when one is available."""
        mock_manager = Mock()
        mock_manager.get_next_task.return_value = {
            'id': 'task:T-001',
            'title': 'Next task',
            'priority': 'high',
            'category': 'feature',
        }

        args = Namespace(start=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_next(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.get_next_task.assert_called_once()
        # Should print task info
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Next task: task:T-001" in call for call in print_calls))

    def test_next_task_with_start_flag(self):
        """Test getting and starting next task with --start flag."""
        mock_manager = Mock()
        mock_manager.get_next_task.return_value = {
            'id': 'task:T-001',
            'title': 'Next task',
            'priority': 'high',
            'category': 'feature',
        }
        mock_manager.start_task.return_value = True

        args = Namespace(start=True)

        with patch('builtins.print') as mock_print:
            result = cmd_task_next(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.start_task.assert_called_once_with('T-001')
        # Should print started confirmation
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Started: task:T-001" in call for call in print_calls))

    def test_next_task_none_available(self):
        """Test when no pending tasks are available."""
        mock_manager = Mock()
        mock_manager.get_next_task.return_value = None

        args = Namespace(start=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_next(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("No pending tasks available.")


class TestCmdTaskShow(unittest.TestCase):
    """Test cmd_task_show command handler."""

    def test_show_task_not_found(self):
        """Test showing a task that doesn't exist."""
        mock_manager = Mock()
        mock_manager.get_task.return_value = None

        args = Namespace(task_id="T-999")

        with patch('builtins.print') as mock_print:
            result = cmd_task_show(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Task not found: T-999")

    def test_show_task_with_task_prefix(self):
        """Test showing a task with 'task:' prefix."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending", "priority": "medium"}
        )
        # First call with "task:T-001" returns None, second call with "T-001" succeeds
        mock_manager.get_task.side_effect = [None, task]
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = []

        args = Namespace(task_id="task:T-001")

        with patch('builtins.print'):
            result = cmd_task_show(args, mock_manager)

        self.assertEqual(result, 0)
        # Should try both with and without prefix
        self.assertEqual(mock_manager.get_task.call_count, 2)

    def test_show_task_success(self):
        """Test showing an existing task."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={
                "status": "pending",
                "priority": "high",
                "category": "feature"
            }
        )
        mock_manager.get_task.return_value = task
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = []

        args = Namespace(task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_task_show(args, mock_manager)

        self.assertEqual(result, 0)
        # Should print task details
        self.assertTrue(mock_print.called)

    def test_show_task_with_dependencies(self):
        """Test showing a task with dependencies."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending", "priority": "high"}
        )
        dep1 = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Dependency 1",
            properties={}
        )
        dep2 = ThoughtNode(
            id="task:T-003",
            node_type=NodeType.TASK,
            content="Dependency 2",
            properties={}
        )
        mock_manager.get_task.return_value = task
        mock_manager.get_task_dependencies.return_value = [dep1, dep2]
        mock_manager.what_depends_on.return_value = []

        args = Namespace(task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_task_show(args, mock_manager)

        self.assertEqual(result, 0)
        # Should show dependencies
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Depends On" in call for call in print_calls))

    def test_show_task_with_dependents(self):
        """Test showing a task with dependents (what it blocks)."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending", "priority": "high"}
        )
        dependent = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Blocked task",
            properties={}
        )
        mock_manager.get_task.return_value = task
        mock_manager.get_task_dependencies.return_value = []
        mock_manager.what_depends_on.return_value = [dependent]

        args = Namespace(task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_task_show(args, mock_manager)

        self.assertEqual(result, 0)
        # Should show what this task blocks
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Blocks" in call for call in print_calls))


class TestCmdTaskStart(unittest.TestCase):
    """Test cmd_task_start command handler."""

    def test_start_task_success(self):
        """Test successfully starting a task."""
        mock_manager = Mock()
        mock_manager.start_task.return_value = True

        args = Namespace(task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_task_start(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.start_task.assert_called_once_with("T-001")
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with("Started: T-001")

    def test_start_task_not_found(self):
        """Test starting a task that doesn't exist."""
        mock_manager = Mock()
        mock_manager.start_task.return_value = False

        args = Namespace(task_id="T-999")

        with patch('builtins.print') as mock_print:
            result = cmd_task_start(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Task not found: T-999")


class TestCmdTaskComplete(unittest.TestCase):
    """Test cmd_task_complete command handler."""

    def test_complete_task_success(self):
        """Test successfully completing a task."""
        mock_manager = Mock()
        mock_manager.complete_task.return_value = True

        args = Namespace(task_id="T-001", retrospective=None)

        with patch('builtins.print') as mock_print:
            result = cmd_task_complete(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.complete_task.assert_called_once_with("T-001", None)
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with("Completed: T-001")

    def test_complete_task_with_retrospective(self):
        """Test completing a task with retrospective notes."""
        mock_manager = Mock()
        mock_manager.complete_task.return_value = True

        args = Namespace(
            task_id="T-001",
            retrospective="Learned a lot about testing"
        )

        with patch('builtins.print'):
            result = cmd_task_complete(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.complete_task.assert_called_once_with(
            "T-001",
            "Learned a lot about testing"
        )

    def test_complete_task_not_found(self):
        """Test completing a task that doesn't exist."""
        mock_manager = Mock()
        mock_manager.complete_task.return_value = False

        args = Namespace(task_id="T-999", retrospective=None)

        with patch('builtins.print') as mock_print:
            result = cmd_task_complete(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Task not found: T-999")


class TestCmdTaskBlock(unittest.TestCase):
    """Test cmd_task_block command handler."""

    def test_block_task_success(self):
        """Test successfully blocking a task."""
        mock_manager = Mock()
        mock_manager.block_task.return_value = True

        args = Namespace(
            task_id="T-001",
            reason="Waiting for dependency",
            blocker=None
        )

        with patch('builtins.print') as mock_print:
            result = cmd_task_block(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.block_task.assert_called_once_with(
            "T-001",
            "Waiting for dependency",
            None
        )
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with("Blocked: T-001")

    def test_block_task_with_blocker(self):
        """Test blocking a task with a blocker task ID."""
        mock_manager = Mock()
        mock_manager.block_task.return_value = True

        args = Namespace(
            task_id="T-001",
            reason="Blocked by another task",
            blocker="T-002"
        )

        with patch('builtins.print'):
            result = cmd_task_block(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.block_task.assert_called_once_with(
            "T-001",
            "Blocked by another task",
            "T-002"
        )

    def test_block_task_not_found(self):
        """Test blocking a task that doesn't exist."""
        mock_manager = Mock()
        mock_manager.block_task.return_value = False

        args = Namespace(
            task_id="T-999",
            reason="Test reason",
            blocker=None
        )

        with patch('builtins.print') as mock_print:
            result = cmd_task_block(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Task not found: T-999")


class TestCmdTaskDepends(unittest.TestCase):
    """Test cmd_task_depends command handler."""

    def test_add_dependency_success(self):
        """Test successfully adding a task dependency."""
        mock_manager = Mock()
        mock_manager.add_dependency.return_value = True

        args = Namespace(task_id="T-001", depends_on_id="T-002")

        with patch('builtins.print') as mock_print:
            result = cmd_task_depends(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.add_dependency.assert_called_once_with("T-001", "T-002")
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with(
            "Created dependency: T-001 depends on T-002"
        )

    def test_add_dependency_failure(self):
        """Test adding dependency when task IDs are invalid."""
        mock_manager = Mock()
        mock_manager.add_dependency.return_value = False

        args = Namespace(task_id="T-001", depends_on_id="T-999")

        with patch('builtins.print') as mock_print:
            result = cmd_task_depends(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with(
            "Failed to create dependency - check that both task IDs exist"
        )

    def test_add_dependency_exception(self):
        """Test handling exception when adding dependency."""
        mock_manager = Mock()
        mock_manager.add_dependency.side_effect = Exception("Circular dependency")

        args = Namespace(task_id="T-001", depends_on_id="T-002")

        with patch('builtins.print') as mock_print:
            result = cmd_task_depends(args, mock_manager)

        self.assertEqual(result, 1)
        print_output = str(mock_print.call_args)
        self.assertIn("Error creating dependency", print_output)


class TestCmdTaskDelete(unittest.TestCase):
    """Test cmd_task_delete command handler."""

    def test_delete_task_not_found(self):
        """Test deleting a task that doesn't exist."""
        mock_manager = Mock()
        mock_manager.get_task.return_value = None

        args = Namespace(task_id="T-999", force=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_delete(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Task not found: T-999")

    def test_delete_task_with_dependents_no_force(self):
        """Test deleting a task with dependents without --force flag."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending"}
        )
        dependent = ThoughtNode(
            id="task:T-002",
            node_type=NodeType.TASK,
            content="Dependent task",
            properties={}
        )
        mock_manager.get_task.return_value = task
        mock_manager.what_depends_on.return_value = [dependent]

        args = Namespace(task_id="T-001", force=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_delete(args, mock_manager)

        self.assertEqual(result, 1)
        # Should print warning about dependents
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Cannot delete" in call for call in print_calls))

    def test_delete_in_progress_task_no_force(self):
        """Test deleting an in-progress task without --force flag."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "in_progress"}
        )
        mock_manager.get_task.return_value = task
        mock_manager.what_depends_on.return_value = []

        args = Namespace(task_id="T-001", force=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_delete(args, mock_manager)

        self.assertEqual(result, 1)
        # Should print warning about in-progress status
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("task is in progress" in call for call in print_calls))

    def test_delete_task_success(self):
        """Test successfully deleting a task."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending"}
        )
        mock_manager.get_task.return_value = task
        mock_manager.what_depends_on.return_value = []
        mock_manager.delete_task.return_value = True

        args = Namespace(task_id="T-001", force=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_delete(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.delete_task.assert_called_once_with("T-001", force=False)
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Deleted: T-001" in call for call in print_calls))

    def test_delete_task_with_force(self):
        """Test deleting a task with --force flag."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "in_progress"}
        )
        mock_manager.get_task.return_value = task
        mock_manager.delete_task.return_value = True

        args = Namespace(task_id="T-001", force=True)

        with patch('builtins.print') as mock_print:
            result = cmd_task_delete(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.delete_task.assert_called_once_with("T-001", force=True)
        # Should show forced deletion message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("forced deletion" in call for call in print_calls))

    def test_delete_task_deletion_failed(self):
        """Test when deletion operation fails."""
        mock_manager = Mock()
        task = ThoughtNode(
            id="task:T-001",
            node_type=NodeType.TASK,
            content="Test task",
            properties={"status": "pending"}
        )
        mock_manager.get_task.return_value = task
        mock_manager.what_depends_on.return_value = []
        mock_manager.delete_task.return_value = False

        args = Namespace(task_id="T-001", force=False)

        with patch('builtins.print') as mock_print:
            result = cmd_task_delete(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_any_call("Failed to delete: T-001")


class TestHandleTaskCommand(unittest.TestCase):
    """Test handle_task_command routing."""

    def test_no_subcommand(self):
        """Test error when no subcommand specified."""
        mock_manager = Mock()
        args = Namespace()  # No task_command attribute

        with patch('builtins.print') as mock_print:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 1)
        self.assertIn("No task subcommand", str(mock_print.call_args))

    def test_route_to_create(self):
        """Test routing to create command."""
        mock_manager = Mock()
        args = Namespace(
            task_command="create",
            title="Test task",
            priority="medium",
            category="feature",
            description="",
            sprint=None,
            depends=None,
            blocks=None,
        )

        with patch('cortical.got.cli.task.cmd_task_create', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_list(self):
        """Test routing to list command."""
        mock_manager = Mock()
        args = Namespace(task_command="list")

        with patch('cortical.got.cli.task.cmd_task_list', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_next(self):
        """Test routing to next command."""
        mock_manager = Mock()
        args = Namespace(task_command="next")

        with patch('cortical.got.cli.task.cmd_task_next', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_show(self):
        """Test routing to show command."""
        mock_manager = Mock()
        args = Namespace(task_command="show", task_id="T-001")

        with patch('cortical.got.cli.task.cmd_task_show', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_start(self):
        """Test routing to start command."""
        mock_manager = Mock()
        args = Namespace(task_command="start", task_id="T-001")

        with patch('cortical.got.cli.task.cmd_task_start', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_complete(self):
        """Test routing to complete command."""
        mock_manager = Mock()
        args = Namespace(task_command="complete", task_id="T-001")

        with patch('cortical.got.cli.task.cmd_task_complete', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_block(self):
        """Test routing to block command."""
        mock_manager = Mock()
        args = Namespace(task_command="block", task_id="T-001", reason="Test")

        with patch('cortical.got.cli.task.cmd_task_block', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_delete(self):
        """Test routing to delete command."""
        mock_manager = Mock()
        args = Namespace(task_command="delete", task_id="T-001", force=False)

        with patch('cortical.got.cli.task.cmd_task_delete', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_route_to_depends(self):
        """Test routing to depends command."""
        mock_manager = Mock()
        args = Namespace(
            task_command="depends",
            task_id="T-001",
            depends_on_id="T-002"
        )

        with patch('cortical.got.cli.task.cmd_task_depends', return_value=0) as mock_cmd:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_unknown_command(self):
        """Test error for unknown subcommand."""
        mock_manager = Mock()
        args = Namespace(task_command="invalid")

        with patch('builtins.print') as mock_print:
            result = handle_task_command(args, mock_manager)

        self.assertEqual(result, 1)
        self.assertIn("Unknown task subcommand", str(mock_print.call_args))


class TestSetupTaskParser(unittest.TestCase):
    """Test setup_task_parser function."""

    def test_parser_setup_create(self):
        """Test create subcommand parser setup."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args([
            'task', 'create', 'Test task',
            '--priority', 'high',
            '--category', 'bugfix',
            '--description', 'Fix critical bug'
        ])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'create')
        self.assertEqual(args.title, 'Test task')
        self.assertEqual(args.priority, 'high')
        self.assertEqual(args.category, 'bugfix')
        self.assertEqual(args.description, 'Fix critical bug')

    def test_parser_setup_list(self):
        """Test list subcommand parser setup."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args([
            'task', 'list',
            '--status', 'in_progress',
            '--priority', 'high',
            '--json'
        ])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'list')
        self.assertEqual(args.status, 'in_progress')
        self.assertEqual(args.priority, 'high')
        self.assertTrue(args.json)

    def test_parser_setup_show(self):
        """Test show subcommand parser setup."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args(['task', 'show', 'T-001'])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'show')
        self.assertEqual(args.task_id, 'T-001')

    def test_parser_setup_next_with_start(self):
        """Test next subcommand parser with --start flag."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args(['task', 'next', '--start'])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'next')
        self.assertTrue(args.start)

    def test_parser_setup_delete_with_force(self):
        """Test delete subcommand parser with --force flag."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args(['task', 'delete', 'T-001', '--force'])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'delete')
        self.assertEqual(args.task_id, 'T-001')
        self.assertTrue(args.force)

    def test_parser_setup_depends(self):
        """Test depends subcommand parser setup."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args(['task', 'depends', 'T-001', '--on', 'T-002'])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'depends')
        self.assertEqual(args.task_id, 'T-001')
        self.assertEqual(args.depends_on_id, 'T-002')

    def test_parser_setup_create_with_dependencies(self):
        """Test create parser with --depends-on and --blocks."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_task_parser(subparsers)

        args = parser.parse_args([
            'task', 'create', 'Test task',
            '--depends-on', 'T-001', 'T-002',
            '--blocks', 'T-003'
        ])

        self.assertEqual(args.command, 'task')
        self.assertEqual(args.task_command, 'create')
        self.assertEqual(args.depends, ['T-001', 'T-002'])
        self.assertEqual(args.blocks, ['T-003'])


if __name__ == '__main__':
    unittest.main()
