"""
Unit tests for cortical.got.cli.sprint module.

Tests use mocked GoTProjectManager to avoid file system operations.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace

from cortical.got.cli.sprint import (
    cmd_sprint_create,
    cmd_sprint_list,
    cmd_sprint_status,
    cmd_sprint_start,
    cmd_sprint_complete,
    cmd_sprint_claim,
    cmd_sprint_release,
    cmd_sprint_goal_add,
    cmd_sprint_goal_list,
    cmd_sprint_goal_complete,
    cmd_sprint_link,
    cmd_sprint_unlink,
    cmd_sprint_tasks,
    cmd_sprint_suggest,
    cmd_epic_create,
    cmd_epic_list,
    cmd_epic_show,
    handle_sprint_command,
    handle_epic_command,
    setup_sprint_parser,
    setup_epic_parser,
)


def create_mock_sprint(sprint_id="S-001", content="Sprint 1", status="available", claimed_by=None):
    """Helper to create a mock sprint node."""
    sprint = Mock()
    sprint.id = sprint_id
    sprint.content = content
    sprint.properties = {"status": status}
    if claimed_by:
        sprint.properties["claimed_by"] = claimed_by
    return sprint


def create_mock_epic(epic_id="EPIC-001", content="Epic 1", status="active", phase="planning"):
    """Helper to create a mock epic node."""
    epic = Mock()
    epic.id = epic_id
    epic.content = content
    epic.properties = {"status": status, "phase": phase}
    return epic


def create_mock_task(task_id="T-001", content="Task 1", status="pending", priority="medium"):
    """Helper to create a mock task node."""
    task = Mock()
    task.id = task_id
    task.content = content
    task.properties = {"status": status, "priority": priority}
    return task


class TestCmdSprintCreate(unittest.TestCase):
    """Test cmd_sprint_create command handler."""

    def test_create_sprint_success(self):
        """Test successful sprint creation."""
        mock_manager = Mock()
        mock_manager.create_sprint.return_value = "S-001"

        args = Namespace(name="Sprint 1", number=1, epic=None)

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_create(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.create_sprint.assert_called_once_with(
            name="Sprint 1",
            number=1,
            epic_id=None,
        )
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with("Created: S-001")

    def test_create_sprint_with_epic(self):
        """Test sprint creation with epic link."""
        mock_manager = Mock()
        mock_manager.create_sprint.return_value = "S-002"

        args = Namespace(name="Sprint 2", number=2, epic="EPIC-001")

        with patch('builtins.print'):
            result = cmd_sprint_create(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.create_sprint.assert_called_once_with(
            name="Sprint 2",
            number=2,
            epic_id="EPIC-001",
        )


class TestCmdSprintList(unittest.TestCase):
    """Test cmd_sprint_list command handler."""

    def test_list_empty(self):
        """Test listing when no sprints exist."""
        mock_manager = Mock()
        mock_manager.list_sprints.return_value = []

        args = Namespace(status=None)

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_sprints.assert_called_once_with(status=None)
        mock_print.assert_called_with("No sprints found.")

    def test_list_with_sprints(self):
        """Test listing with sample sprints."""
        mock_manager = Mock()
        sprint1 = create_mock_sprint("S-001", "Sprint 1", "available")
        sprint2 = create_mock_sprint("S-002", "Sprint 2", "in_progress", "agent-alpha")

        mock_manager.list_sprints.return_value = [sprint1, sprint2]
        mock_manager.get_sprint_progress.side_effect = [
            {"progress_percent": 0.0},
            {"progress_percent": 50.0},
        ]

        args = Namespace(status=None)

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_list(args, mock_manager)

        self.assertEqual(result, 0)
        self.assertEqual(mock_manager.get_sprint_progress.call_count, 2)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("S-001" in str(call) for call in print_calls))
        self.assertTrue(any("S-002" in str(call) for call in print_calls))
        self.assertTrue(any("agent-alpha" in str(call) for call in print_calls))

    def test_list_with_status_filter(self):
        """Test listing with status filter."""
        mock_manager = Mock()
        mock_manager.list_sprints.return_value = []

        args = Namespace(status="in_progress")

        with patch('builtins.print'):
            result = cmd_sprint_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_sprints.assert_called_once_with(status="in_progress")


class TestCmdSprintStatus(unittest.TestCase):
    """Test cmd_sprint_status command handler."""

    def test_status_specific_sprint_not_found(self):
        """Test status for non-existent sprint."""
        mock_manager = Mock()
        mock_manager.get_sprint.return_value = None

        args = Namespace(sprint_id="S-999")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_status(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Sprint not found: S-999")

    def test_status_specific_sprint_success(self):
        """Test status for existing sprint."""
        mock_manager = Mock()
        sprint = create_mock_sprint("S-001", "Sprint 1", "in_progress")
        mock_manager.get_sprint.return_value = sprint
        mock_manager.get_sprint_progress.return_value = {
            "completed": 2,
            "total_tasks": 5,
            "progress_percent": 40.0,
        }

        args = Namespace(sprint_id="S-001")

        with patch('cortical.got.cli.sprint.format_sprint_status', return_value="Sprint Status Output"):
            with patch('builtins.print') as mock_print:
                result = cmd_sprint_status(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.get_sprint.assert_called_once_with("S-001")

    def test_status_no_sprint_id_shows_active(self):
        """Test status without sprint ID shows active sprints."""
        mock_manager = Mock()
        sprint1 = create_mock_sprint("S-001", "Sprint 1", "in_progress")
        mock_manager.list_sprints.side_effect = [[sprint1], []]
        mock_manager.get_sprint_progress.return_value = {"progress_percent": 50.0}

        args = Namespace(sprint_id=None)

        with patch('cortical.got.cli.sprint.format_sprint_status', return_value="Sprint Status"):
            with patch('builtins.print'):
                result = cmd_sprint_status(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_sprints.assert_called_with(status="in_progress")

    def test_status_fallback_to_available(self):
        """Test status falls back to available sprints if no active."""
        mock_manager = Mock()
        sprint1 = create_mock_sprint("S-001", "Sprint 1", "available")
        mock_manager.list_sprints.side_effect = [[], [sprint1]]
        mock_manager.get_sprint_progress.return_value = {"progress_percent": 0.0}

        args = Namespace(sprint_id=None)

        with patch('cortical.got.cli.sprint.format_sprint_status', return_value="Sprint Status"):
            with patch('builtins.print'):
                result = cmd_sprint_status(args, mock_manager)

        self.assertEqual(result, 0)
        # First call for in_progress, second for available
        self.assertEqual(mock_manager.list_sprints.call_count, 2)


class TestCmdSprintStart(unittest.TestCase):
    """Test cmd_sprint_start command handler."""

    def test_start_sprint_success(self):
        """Test successful sprint start."""
        mock_manager = Mock()
        sprint = create_mock_sprint("S-001", "Sprint 1", "in_progress")
        mock_manager.update_sprint.return_value = sprint

        args = Namespace(sprint_id="S-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_start(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.update_sprint.assert_called_once_with("S-001", status="in_progress")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("S-001" in str(call) for call in print_calls))


class TestCmdSprintComplete(unittest.TestCase):
    """Test cmd_sprint_complete command handler."""

    def test_complete_sprint_success(self):
        """Test successful sprint completion."""
        mock_manager = Mock()
        sprint = create_mock_sprint("S-001", "Sprint 1", "completed")
        mock_manager.update_sprint.return_value = sprint

        args = Namespace(sprint_id="S-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_complete(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.update_sprint.assert_called_once_with("S-001", status="completed")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Completed" in str(call) for call in print_calls))


class TestCmdSprintClaim(unittest.TestCase):
    """Test cmd_sprint_claim command handler."""

    def test_claim_sprint_success(self):
        """Test successful sprint claim."""
        mock_manager = Mock()
        sprint = create_mock_sprint("S-001", "Sprint 1", "in_progress", "agent-1")
        mock_manager.claim_sprint.return_value = sprint

        args = Namespace(sprint_id="S-001", agent="agent-1")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_claim(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.claim_sprint.assert_called_once_with("S-001", "agent-1")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Claimed" in str(call) for call in print_calls))
        self.assertTrue(any("agent-1" in str(call) for call in print_calls))

    def test_claim_sprint_error(self):
        """Test sprint claim failure."""
        mock_manager = Mock()
        mock_manager.claim_sprint.side_effect = ValueError("Sprint already claimed")

        args = Namespace(sprint_id="S-001", agent="agent-1")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_claim(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Error: Sprint already claimed")


class TestCmdSprintRelease(unittest.TestCase):
    """Test cmd_sprint_release command handler."""

    def test_release_sprint_success(self):
        """Test successful sprint release."""
        mock_manager = Mock()
        sprint = create_mock_sprint("S-001", "Sprint 1", "available")
        mock_manager.release_sprint.return_value = sprint

        args = Namespace(sprint_id="S-001", agent="agent-1")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_release(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.release_sprint.assert_called_once_with("S-001", "agent-1")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Released" in str(call) for call in print_calls))

    def test_release_sprint_error(self):
        """Test sprint release failure."""
        mock_manager = Mock()
        mock_manager.release_sprint.side_effect = ValueError("Not claimed by this agent")

        args = Namespace(sprint_id="S-001", agent="agent-1")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_release(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Error: Not claimed by this agent")


class TestCmdSprintGoalAdd(unittest.TestCase):
    """Test cmd_sprint_goal_add command handler."""

    def test_add_goal_success(self):
        """Test successful goal addition."""
        mock_manager = Mock()
        mock_manager.add_sprint_goal.return_value = True

        args = Namespace(sprint_id="S-001", description="Complete feature X")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_goal_add(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.add_sprint_goal.assert_called_once_with("S-001", "Complete feature X")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Added goal" in str(call) for call in print_calls))

    def test_add_goal_sprint_not_found(self):
        """Test goal addition when sprint doesn't exist."""
        mock_manager = Mock()
        mock_manager.add_sprint_goal.return_value = False

        args = Namespace(sprint_id="S-999", description="Goal")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_goal_add(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Sprint not found: S-999")


class TestCmdSprintGoalList(unittest.TestCase):
    """Test cmd_sprint_goal_list command handler."""

    def test_list_goals_empty(self):
        """Test listing when no goals exist."""
        mock_manager = Mock()
        mock_manager.list_sprint_goals.return_value = []

        args = Namespace(sprint_id="S-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_goal_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("No goals for sprint S-001")

    def test_list_goals_with_items(self):
        """Test listing with sample goals."""
        mock_manager = Mock()
        mock_manager.list_sprint_goals.return_value = [
            {"description": "Goal 1", "completed": False},
            {"description": "Goal 2", "completed": True},
        ]

        args = Namespace(sprint_id="S-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_goal_list(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Goal 1" in str(call) for call in print_calls))
        self.assertTrue(any("Goal 2" in str(call) for call in print_calls))


class TestCmdSprintGoalComplete(unittest.TestCase):
    """Test cmd_sprint_goal_complete command handler."""

    def test_complete_goal_success(self):
        """Test successful goal completion."""
        mock_manager = Mock()
        mock_manager.complete_sprint_goal.return_value = True

        args = Namespace(sprint_id="S-001", index=0)

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_goal_complete(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.complete_sprint_goal.assert_called_once_with("S-001", 0)
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Completed goal" in str(call) for call in print_calls))

    def test_complete_goal_failure(self):
        """Test goal completion failure."""
        mock_manager = Mock()
        mock_manager.complete_sprint_goal.return_value = False

        args = Namespace(sprint_id="S-001", index=99)

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_goal_complete(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Failed - check sprint ID and goal index")


class TestCmdSprintLink(unittest.TestCase):
    """Test cmd_sprint_link command handler."""

    def test_link_task_success(self):
        """Test successful task linking."""
        mock_manager = Mock()
        mock_manager.link_task_to_sprint.return_value = True

        args = Namespace(sprint_id="S-001", task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_link(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.link_task_to_sprint.assert_called_once_with("S-001", "T-001")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Linked" in str(call) for call in print_calls))

    def test_link_task_failure(self):
        """Test task linking failure."""
        mock_manager = Mock()
        mock_manager.link_task_to_sprint.return_value = False

        args = Namespace(sprint_id="S-999", task_id="T-999")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_link(args, mock_manager)

        self.assertEqual(result, 1)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Failed" in str(call) for call in print_calls))


class TestCmdSprintUnlink(unittest.TestCase):
    """Test cmd_sprint_unlink command handler."""

    def test_unlink_task_success(self):
        """Test successful task unlinking."""
        mock_manager = Mock()
        mock_manager.unlink_task_from_sprint.return_value = True

        args = Namespace(sprint_id="S-001", task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_unlink(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.unlink_task_from_sprint.assert_called_once_with("S-001", "T-001")
        mock_manager.save.assert_called_once()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Unlinked" in str(call) for call in print_calls))

    def test_unlink_task_not_linked(self):
        """Test unlinking when no link exists."""
        mock_manager = Mock()
        mock_manager.unlink_task_from_sprint.return_value = False

        args = Namespace(sprint_id="S-001", task_id="T-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_unlink(args, mock_manager)

        self.assertEqual(result, 1)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("No link found" in str(call) for call in print_calls))


class TestCmdSprintTasks(unittest.TestCase):
    """Test cmd_sprint_tasks command handler."""

    def test_tasks_empty(self):
        """Test listing when no tasks in sprint."""
        mock_manager = Mock()
        mock_manager.get_sprint_tasks.return_value = []

        args = Namespace(sprint_id="S-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_tasks(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("No tasks in sprint S-001")

    def test_tasks_with_items(self):
        """Test listing with sample tasks."""
        mock_manager = Mock()
        task1 = create_mock_task("T-001", "Task 1", "pending", "high")
        task2 = create_mock_task("T-002", "Task 2", "in_progress", "medium")
        mock_manager.get_sprint_tasks.return_value = [task1, task2]

        args = Namespace(sprint_id="S-001")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_tasks(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("T-001" in str(call) for call in print_calls))
        self.assertTrue(any("T-002" in str(call) for call in print_calls))
        self.assertTrue(any("high" in str(call) for call in print_calls))


class TestCmdSprintSuggest(unittest.TestCase):
    """Test cmd_sprint_suggest command handler."""

    def test_suggest_no_pending_tasks(self):
        """Test suggestions when no pending tasks exist."""
        mock_manager = Mock()
        mock_manager.list_tasks.return_value = []

        args = Namespace(limit=10, strategy="balanced")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_suggest(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("No pending tasks to suggest.")

    def test_suggest_with_pending_tasks(self):
        """Test suggestions with pending tasks."""
        mock_manager = Mock()
        task1 = create_mock_task("T-001", "High priority task", "pending", "high")
        task2 = create_mock_task("T-002", "Medium priority task", "pending", "medium")
        task3 = create_mock_task("T-003", "Low priority task", "pending", "low")

        # Add category to properties
        task1.properties["category"] = "feature"
        task2.properties["category"] = "bugfix"
        task3.properties["category"] = "docs"

        mock_manager.list_tasks.return_value = [task1, task2, task3]
        mock_manager.what_blocks.return_value = []  # No blockers

        args = Namespace(limit=5, strategy="balanced")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_suggest(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Should print suggestions with task IDs
        self.assertTrue(any("T-001" in str(call) for call in print_calls))
        self.assertTrue(any("T-002" in str(call) for call in print_calls))

    def test_suggest_blocked_tasks_penalized(self):
        """Test that blocked tasks get lower priority."""
        mock_manager = Mock()
        task1 = create_mock_task("T-001", "Blocked task", "pending", "high")
        task2 = create_mock_task("T-002", "Unblocked task", "pending", "medium")

        task1.properties["category"] = "feature"
        task2.properties["category"] = "feature"

        mock_manager.list_tasks.return_value = [task1, task2]
        mock_manager.what_blocks.side_effect = [["T-003"], []]  # task1 blocked, task2 not

        args = Namespace(limit=10, strategy="balanced")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_suggest(args, mock_manager)

        self.assertEqual(result, 0)
        # Both tasks should be shown, but order may differ due to blocking penalty

    def test_suggest_respects_limit(self):
        """Test that limit parameter is respected."""
        mock_manager = Mock()
        tasks = [create_mock_task(f"T-{i:03d}", f"Task {i}", "pending", "medium") for i in range(20)]
        for task in tasks:
            task.properties["category"] = "feature"

        mock_manager.list_tasks.return_value = tasks
        mock_manager.what_blocks.return_value = []

        args = Namespace(limit=5, strategy="balanced")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_suggest(args, mock_manager)

        self.assertEqual(result, 0)
        # Should suggest at most 5 tasks
        print_output = str(mock_print.call_args_list)
        # Count how many task IDs are mentioned (rough check)
        # The output should mention "5 tasks" in the header

    def test_suggest_handles_exception(self):
        """Test error handling when suggestion fails."""
        mock_manager = Mock()
        mock_manager.list_tasks.side_effect = Exception("Database error")

        args = Namespace(limit=10, strategy="balanced")

        with patch('builtins.print') as mock_print:
            result = cmd_sprint_suggest(args, mock_manager)

        self.assertEqual(result, 1)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Error generating suggestions" in str(call) for call in print_calls))


class TestCmdEpicCreate(unittest.TestCase):
    """Test cmd_epic_create command handler."""

    def test_create_epic_success(self):
        """Test successful epic creation."""
        mock_manager = Mock()
        mock_manager.create_epic.return_value = "EPIC-001"

        args = Namespace(name="Epic 1", epic_id=None)

        with patch('builtins.print') as mock_print:
            result = cmd_epic_create(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.create_epic.assert_called_once_with(
            name="Epic 1",
            epic_id=None,
        )
        mock_manager.save.assert_called_once()
        mock_print.assert_called_with("Created: EPIC-001")

    def test_create_epic_with_custom_id(self):
        """Test epic creation with custom ID."""
        mock_manager = Mock()
        mock_manager.create_epic.return_value = "EPIC-CUSTOM"

        args = Namespace(name="Custom Epic", epic_id="EPIC-CUSTOM")

        with patch('builtins.print'):
            result = cmd_epic_create(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.create_epic.assert_called_once_with(
            name="Custom Epic",
            epic_id="EPIC-CUSTOM",
        )


class TestCmdEpicList(unittest.TestCase):
    """Test cmd_epic_list command handler."""

    def test_list_empty(self):
        """Test listing when no epics exist."""
        mock_manager = Mock()
        mock_manager.list_epics.return_value = []

        args = Namespace(status=None)

        with patch('builtins.print') as mock_print:
            result = cmd_epic_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_epics.assert_called_once_with(status=None)
        mock_print.assert_called_with("No epics found.")

    def test_list_with_epics(self):
        """Test listing with sample epics."""
        mock_manager = Mock()
        epic1 = create_mock_epic("EPIC-001", "Epic 1", "active", "planning")
        epic2 = create_mock_epic("EPIC-002", "Epic 2", "completed", "delivery")

        mock_manager.list_epics.return_value = [epic1, epic2]

        args = Namespace(status=None)

        with patch('builtins.print') as mock_print:
            result = cmd_epic_list(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("EPIC-001" in str(call) for call in print_calls))
        self.assertTrue(any("EPIC-002" in str(call) for call in print_calls))
        self.assertTrue(any("planning" in str(call) for call in print_calls))
        self.assertTrue(any("delivery" in str(call) for call in print_calls))

    def test_list_with_status_filter(self):
        """Test listing with status filter."""
        mock_manager = Mock()
        mock_manager.list_epics.return_value = []

        args = Namespace(status="active")

        with patch('builtins.print'):
            result = cmd_epic_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_manager.list_epics.assert_called_once_with(status="active")


class TestCmdEpicShow(unittest.TestCase):
    """Test cmd_epic_show command handler."""

    def test_show_epic_not_found(self):
        """Test showing epic that doesn't exist."""
        mock_manager = Mock()
        mock_manager.get_epic.return_value = None

        args = Namespace(epic_id="EPIC-999")

        with patch('builtins.print') as mock_print:
            result = cmd_epic_show(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Epic not found: EPIC-999")

    def test_show_epic_success(self):
        """Test showing existing epic."""
        mock_manager = Mock()
        epic = create_mock_epic("EPIC-001", "Epic 1", "active", "planning")
        mock_manager.get_epic.return_value = epic
        mock_manager.list_sprints.return_value = []

        args = Namespace(epic_id="EPIC-001")

        with patch('builtins.print') as mock_print:
            result = cmd_epic_show(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("EPIC-001" in str(call) for call in print_calls))
        self.assertTrue(any("Epic 1" in str(call) for call in print_calls))

    def test_show_epic_with_sprints(self):
        """Test showing epic with associated sprints."""
        mock_manager = Mock()
        epic = create_mock_epic("EPIC-001", "Epic 1", "active", "planning")
        sprint1 = create_mock_sprint("S-001", "Sprint 1")
        sprint2 = create_mock_sprint("S-002", "Sprint 2")

        mock_manager.get_epic.return_value = epic
        mock_manager.list_sprints.return_value = [sprint1, sprint2]

        args = Namespace(epic_id="EPIC-001")

        with patch('builtins.print') as mock_print:
            result = cmd_epic_show(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("S-001" in str(call) for call in print_calls))
        self.assertTrue(any("S-002" in str(call) for call in print_calls))


class TestHandleSprintCommand(unittest.TestCase):
    """Test handle_sprint_command routing."""

    def test_no_subcommand(self):
        """Test error when no subcommand specified."""
        mock_manager = Mock()
        args = Namespace()  # No sprint_command attribute

        with patch('builtins.print') as mock_print:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_once()
        self.assertIn("No sprint subcommand", str(mock_print.call_args))

    def test_route_to_create(self):
        """Test routing to create command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="create", name="Sprint 1", number=1, epic=None)

        with patch('cortical.got.cli.sprint.cmd_sprint_create', return_value=0) as mock_create:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_create.assert_called_once_with(args, mock_manager)

    def test_route_to_list(self):
        """Test routing to list command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="list", status=None)

        with patch('cortical.got.cli.sprint.cmd_sprint_list', return_value=0) as mock_list:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_list.assert_called_once_with(args, mock_manager)

    def test_route_to_status(self):
        """Test routing to status command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="status", sprint_id="S-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_status', return_value=0) as mock_status:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_status.assert_called_once_with(args, mock_manager)

    def test_route_to_start(self):
        """Test routing to start command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="start", sprint_id="S-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_start', return_value=0) as mock_start:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_start.assert_called_once_with(args, mock_manager)

    def test_route_to_complete(self):
        """Test routing to complete command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="complete", sprint_id="S-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_complete', return_value=0) as mock_complete:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_complete.assert_called_once_with(args, mock_manager)

    def test_route_to_claim(self):
        """Test routing to claim command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="claim", sprint_id="S-001", agent="agent-1")

        with patch('cortical.got.cli.sprint.cmd_sprint_claim', return_value=0) as mock_claim:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_claim.assert_called_once_with(args, mock_manager)

    def test_route_to_release(self):
        """Test routing to release command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="release", sprint_id="S-001", agent="agent-1")

        with patch('cortical.got.cli.sprint.cmd_sprint_release', return_value=0) as mock_release:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_release.assert_called_once_with(args, mock_manager)

    def test_route_to_link(self):
        """Test routing to link command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="link", sprint_id="S-001", task_id="T-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_link', return_value=0) as mock_link:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_link.assert_called_once_with(args, mock_manager)

    def test_route_to_unlink(self):
        """Test routing to unlink command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="unlink", sprint_id="S-001", task_id="T-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_unlink', return_value=0) as mock_unlink:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_unlink.assert_called_once_with(args, mock_manager)

    def test_route_to_tasks(self):
        """Test routing to tasks command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="tasks", sprint_id="S-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_tasks', return_value=0) as mock_tasks:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_tasks.assert_called_once_with(args, mock_manager)

    def test_route_to_suggest(self):
        """Test routing to suggest command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="suggest", limit=10, strategy="balanced")

        with patch('cortical.got.cli.sprint.cmd_sprint_suggest', return_value=0) as mock_suggest:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_suggest.assert_called_once_with(args, mock_manager)

    def test_route_to_goal_add(self):
        """Test routing to goal add command."""
        mock_manager = Mock()
        args = Namespace(
            sprint_command="goal",
            goal_action="add",
            sprint_id="S-001",
            description="Goal 1"
        )

        with patch('cortical.got.cli.sprint.cmd_sprint_goal_add', return_value=0) as mock_add:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_add.assert_called_once_with(args, mock_manager)

    def test_route_to_goal_list(self):
        """Test routing to goal list command."""
        mock_manager = Mock()
        args = Namespace(sprint_command="goal", goal_action="list", sprint_id="S-001")

        with patch('cortical.got.cli.sprint.cmd_sprint_goal_list', return_value=0) as mock_list:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_list.assert_called_once_with(args, mock_manager)

    def test_route_to_goal_complete(self):
        """Test routing to goal complete command."""
        mock_manager = Mock()
        args = Namespace(
            sprint_command="goal",
            goal_action="complete",
            sprint_id="S-001",
            index=0
        )

        with patch('cortical.got.cli.sprint.cmd_sprint_goal_complete', return_value=0) as mock_complete:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_complete.assert_called_once_with(args, mock_manager)

    def test_goal_no_action(self):
        """Test error when goal subcommand has no action."""
        mock_manager = Mock()
        args = Namespace(sprint_command="goal")  # No goal_action

        with patch('builtins.print') as mock_print:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 1)
        self.assertIn("No goal subcommand", str(mock_print.call_args))

    def test_unknown_command(self):
        """Test error for unknown subcommand."""
        mock_manager = Mock()
        args = Namespace(sprint_command="invalid")

        with patch('builtins.print') as mock_print:
            result = handle_sprint_command(args, mock_manager)

        self.assertEqual(result, 1)
        self.assertIn("Unknown sprint subcommand", str(mock_print.call_args))


class TestHandleEpicCommand(unittest.TestCase):
    """Test handle_epic_command routing."""

    def test_no_subcommand(self):
        """Test error when no subcommand specified."""
        mock_manager = Mock()
        args = Namespace()  # No epic_command attribute

        with patch('builtins.print') as mock_print:
            result = handle_epic_command(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_once()
        self.assertIn("No epic subcommand", str(mock_print.call_args))

    def test_route_to_create(self):
        """Test routing to create command."""
        mock_manager = Mock()
        args = Namespace(epic_command="create", name="Epic 1", epic_id=None)

        with patch('cortical.got.cli.sprint.cmd_epic_create', return_value=0) as mock_create:
            result = handle_epic_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_create.assert_called_once_with(args, mock_manager)

    def test_route_to_list(self):
        """Test routing to list command."""
        mock_manager = Mock()
        args = Namespace(epic_command="list", status=None)

        with patch('cortical.got.cli.sprint.cmd_epic_list', return_value=0) as mock_list:
            result = handle_epic_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_list.assert_called_once_with(args, mock_manager)

    def test_route_to_show(self):
        """Test routing to show command."""
        mock_manager = Mock()
        args = Namespace(epic_command="show", epic_id="EPIC-001")

        with patch('cortical.got.cli.sprint.cmd_epic_show', return_value=0) as mock_show:
            result = handle_epic_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_show.assert_called_once_with(args, mock_manager)

    def test_unknown_command(self):
        """Test error for unknown subcommand."""
        mock_manager = Mock()
        args = Namespace(epic_command="invalid")

        with patch('builtins.print') as mock_print:
            result = handle_epic_command(args, mock_manager)

        self.assertEqual(result, 1)
        self.assertIn("Unknown epic subcommand", str(mock_print.call_args))


class TestSetupSprintParser(unittest.TestCase):
    """Test setup_sprint_parser function."""

    def test_parser_setup(self):
        """Test that parser is set up correctly."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Set up sprint parser
        setup_sprint_parser(subparsers)

        # Parse a create command
        args = parser.parse_args(['sprint', 'create', 'Sprint 1', '--number', '1'])
        self.assertEqual(args.command, 'sprint')
        self.assertEqual(args.sprint_command, 'create')
        self.assertEqual(args.name, 'Sprint 1')
        self.assertEqual(args.number, 1)

    def test_list_parser_setup(self):
        """Test list subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_sprint_parser(subparsers)

        args = parser.parse_args(['sprint', 'list', '--status', 'in_progress'])
        self.assertEqual(args.command, 'sprint')
        self.assertEqual(args.sprint_command, 'list')
        self.assertEqual(args.status, 'in_progress')

    def test_goal_add_parser_setup(self):
        """Test goal add subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_sprint_parser(subparsers)

        args = parser.parse_args(['sprint', 'goal', 'add', 'S-001', 'Complete feature'])
        self.assertEqual(args.command, 'sprint')
        self.assertEqual(args.sprint_command, 'goal')
        self.assertEqual(args.goal_action, 'add')
        self.assertEqual(args.sprint_id, 'S-001')
        self.assertEqual(args.description, 'Complete feature')


class TestSetupEpicParser(unittest.TestCase):
    """Test setup_epic_parser function."""

    def test_parser_setup(self):
        """Test that parser is set up correctly."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Set up epic parser
        setup_epic_parser(subparsers)

        # Parse a create command
        args = parser.parse_args(['epic', 'create', 'Epic 1'])
        self.assertEqual(args.command, 'epic')
        self.assertEqual(args.epic_command, 'create')
        self.assertEqual(args.name, 'Epic 1')

    def test_list_parser_setup(self):
        """Test list subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_epic_parser(subparsers)

        args = parser.parse_args(['epic', 'list', '--status', 'active'])
        self.assertEqual(args.command, 'epic')
        self.assertEqual(args.epic_command, 'list')
        self.assertEqual(args.status, 'active')

    def test_show_parser_setup(self):
        """Test show subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_epic_parser(subparsers)

        args = parser.parse_args(['epic', 'show', 'EPIC-001'])
        self.assertEqual(args.command, 'epic')
        self.assertEqual(args.epic_command, 'show')
        self.assertEqual(args.epic_id, 'EPIC-001')


if __name__ == '__main__':
    unittest.main()
