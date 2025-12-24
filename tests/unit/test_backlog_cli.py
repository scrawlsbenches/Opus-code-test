"""
Tests for cortical/got/cli/backlog.py

Tests the backlog CLI commands for managing unassigned tasks.
"""

import json
import pytest
from io import StringIO
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass


# Mock data classes
@dataclass
class MockTask:
    id: str
    title: str
    status: str = "pending"
    priority: str = "medium"
    description: str = ""
    created_at: str = "2025-12-24T00:00:00Z"


@dataclass
class MockSprint:
    id: str
    title: str
    is_current: bool = False


@dataclass
class MockSprintSuggestion:
    sprint_id: str
    sprint_title: str
    confidence: float
    reason: str
    is_current: bool = False


@dataclass
class MockConnectionSuggestion:
    target_id: str
    edge_type: str
    confidence: float


@dataclass
class MockOrphanReport:
    orphan_tasks: list
    total_tasks: int
    orphan_rate: float


class TestBacklogList:
    """Tests for cmd_backlog_list."""

    def test_list_empty_backlog(self, capsys):
        """Empty backlog shows appropriate message."""
        from cortical.got.cli.backlog import cmd_backlog_list

        # Mock manager with no orphans
        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = []

        mock_got = Mock()
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = False
            result = cmd_backlog_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower() or "0" in captured.out

    def test_list_with_tasks(self, capsys):
        """List shows tasks sorted by priority."""
        from cortical.got.cli.backlog import cmd_backlog_list

        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = ["T-001", "T-002"]

        mock_got = Mock()
        mock_got.get_task.side_effect = [
            MockTask(id="T-001", title="High priority task", priority="high"),
            MockTask(id="T-002", title="Low priority task", priority="low"),
        ]
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = False
            args.sort = "priority"
            args.status = None
            result = cmd_backlog_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "BACKLOG" in captured.out
        assert "T-001" in captured.out
        assert "T-002" in captured.out

    def test_list_json_output(self, capsys):
        """JSON output flag produces valid JSON."""
        from cortical.got.cli.backlog import cmd_backlog_list

        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = ["T-001"]

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(
            id="T-001", title="Test task", priority="high"
        )
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = True
            args.sort = "priority"
            args.status = None
            result = cmd_backlog_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["id"] == "T-001"

    def test_list_sort_by_created(self, capsys):
        """List can be sorted by creation date."""
        from cortical.got.cli.backlog import cmd_backlog_list

        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = ["T-001", "T-002"]

        mock_got = Mock()
        mock_got.get_task.side_effect = [
            MockTask(id="T-001", title="Old task", created_at="2025-01-01"),
            MockTask(id="T-002", title="New task", created_at="2025-12-01"),
        ]
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = False
            args.sort = "created"
            args.status = None
            result = cmd_backlog_list(args, mock_manager)

        assert result == 0

    def test_list_filter_by_status(self, capsys):
        """List can be filtered by status."""
        from cortical.got.cli.backlog import cmd_backlog_list

        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = ["T-001", "T-002"]

        mock_got = Mock()
        mock_got.get_task.side_effect = [
            MockTask(id="T-001", title="Pending task", status="pending"),
            MockTask(id="T-002", title="In progress", status="in_progress"),
        ]
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = True
            args.sort = "priority"
            args.status = "pending"
            result = cmd_backlog_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Only pending task should be in output
        assert len(data) == 1
        assert data[0]["status"] == "pending"

    def test_list_no_manager(self, capsys):
        """Error when manager cannot be accessed."""
        from cortical.got.cli.backlog import cmd_backlog_list

        mock_manager = Mock()
        mock_manager._manager = None
        mock_manager._got_manager = None

        args = Mock()
        result = cmd_backlog_list(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestBacklogPromote:
    """Tests for cmd_backlog_promote."""

    def test_promote_success(self, capsys):
        """Successfully promote task to sprint."""
        from cortical.got.cli.backlog import cmd_backlog_promote

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(id="T-001", title="Test task")
        mock_got.get_sprint.return_value = MockSprint(id="S-001", title="Sprint 1")
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-001"
        args.sprint = "S-001"

        result = cmd_backlog_promote(args, mock_manager)

        assert result == 0
        mock_got.add_task_to_sprint.assert_called_once_with("T-001", "S-001")
        captured = capsys.readouterr()
        assert "Promoted" in captured.out

    def test_promote_to_current_sprint(self, capsys):
        """Promote to current sprint when none specified."""
        from cortical.got.cli.backlog import cmd_backlog_promote

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(id="T-001", title="Test task")
        mock_got.get_sprint.return_value = MockSprint(id="S-CURRENT", title="Current Sprint")
        mock_got.get_current_sprint.return_value = MockSprint(id="S-CURRENT", title="Current Sprint")
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-001"
        args.sprint = None

        result = cmd_backlog_promote(args, mock_manager)

        assert result == 0
        mock_got.add_task_to_sprint.assert_called_once_with("T-001", "S-CURRENT")

    def test_promote_no_current_sprint(self, capsys):
        """Error when no sprint specified and no current sprint."""
        from cortical.got.cli.backlog import cmd_backlog_promote

        mock_got = Mock()
        mock_got.get_current_sprint.return_value = None
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-001"
        args.sprint = None

        result = cmd_backlog_promote(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No current sprint" in captured.out

    def test_promote_task_not_found(self, capsys):
        """Error when task not found."""
        from cortical.got.cli.backlog import cmd_backlog_promote

        mock_got = Mock()
        mock_got.get_task.return_value = None
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-NOTFOUND"
        args.sprint = "S-001"

        result = cmd_backlog_promote(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_promote_sprint_not_found(self, capsys):
        """Error when sprint not found."""
        from cortical.got.cli.backlog import cmd_backlog_promote

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(id="T-001", title="Test")
        mock_got.get_sprint.return_value = None
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-001"
        args.sprint = "S-NOTFOUND"

        result = cmd_backlog_promote(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Sprint not found" in captured.out

    def test_promote_exception(self, capsys):
        """Error when add_task_to_sprint fails."""
        from cortical.got.cli.backlog import cmd_backlog_promote

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(id="T-001", title="Test")
        mock_got.get_sprint.return_value = MockSprint(id="S-001", title="Sprint")
        mock_got.add_task_to_sprint.side_effect = Exception("Database error")
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-001"
        args.sprint = "S-001"

        result = cmd_backlog_promote(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestBacklogReview:
    """Tests for cmd_backlog_review."""

    def test_review_task(self, capsys):
        """Review task shows details and suggestions."""
        from cortical.got.cli.backlog import cmd_backlog_review

        mock_detector = Mock()
        mock_detector.suggest_sprint.return_value = [
            MockSprintSuggestion(
                sprint_id="S-001",
                sprint_title="Sprint 1",
                confidence=0.8,
                reason="Similar tasks",
                is_current=True
            )
        ]
        mock_detector.suggest_connections.return_value = [
            MockConnectionSuggestion(
                target_id="T-002",
                edge_type="DEPENDS_ON",
                confidence=0.7
            )
        ]

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(
            id="T-001",
            title="Test task",
            description="A test task for review"
        )
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.task_id = "T-001"
            result = cmd_backlog_review(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "BACKLOG REVIEW" in captured.out
        assert "T-001" in captured.out
        assert "SPRINT SUGGESTIONS" in captured.out
        assert "RELATED TASKS" in captured.out

    def test_review_task_not_found(self, capsys):
        """Error when task not found."""
        from cortical.got.cli.backlog import cmd_backlog_review

        mock_got = Mock()
        mock_got.get_task.return_value = None
        mock_manager = Mock()
        mock_manager._manager = mock_got

        args = Mock()
        args.task_id = "T-NOTFOUND"
        result = cmd_backlog_review(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_review_no_suggestions(self, capsys):
        """Review shows message when no suggestions available."""
        from cortical.got.cli.backlog import cmd_backlog_review

        mock_detector = Mock()
        mock_detector.suggest_sprint.return_value = []
        mock_detector.suggest_connections.return_value = []

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(id="T-001", title="Isolated task")
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.task_id = "T-001"
            result = cmd_backlog_review(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No sprint suggestions" in captured.out
        assert "No related tasks" in captured.out


class TestBacklogStats:
    """Tests for cmd_backlog_stats."""

    def test_stats_display(self, capsys):
        """Stats shows statistics in readable format."""
        from cortical.got.cli.backlog import cmd_backlog_stats

        mock_detector = Mock()
        mock_detector.generate_orphan_report.return_value = MockOrphanReport(
            orphan_tasks=["T-001", "T-002"],
            total_tasks=10,
            orphan_rate=20.0
        )

        mock_got = Mock()
        mock_got.get_task.side_effect = [
            MockTask(id="T-001", title="Task 1", priority="high", status="pending"),
            MockTask(id="T-002", title="Task 2", priority="low", status="in_progress"),
        ]
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = False
            result = cmd_backlog_stats(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "BACKLOG STATISTICS" in captured.out
        assert "Total in backlog: 2" in captured.out

    def test_stats_json_output(self, capsys):
        """Stats produces valid JSON output."""
        from cortical.got.cli.backlog import cmd_backlog_stats

        mock_detector = Mock()
        mock_detector.generate_orphan_report.return_value = MockOrphanReport(
            orphan_tasks=["T-001"],
            total_tasks=5,
            orphan_rate=20.0
        )

        mock_got = Mock()
        mock_got.get_task.return_value = MockTask(
            id="T-001", title="Task", priority="high", status="pending"
        )
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = True
            result = cmd_backlog_stats(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_backlog"] == 1
        assert data["total_tasks"] == 5
        assert data["backlog_rate"] == 20.0


class TestSetupParser:
    """Tests for setup_backlog_parser."""

    def test_setup_creates_subcommands(self):
        """Parser setup creates expected subcommands."""
        from cortical.got.cli.backlog import setup_backlog_parser
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        # This should not raise
        setup_backlog_parser(subparsers)


class TestHandleBacklogCommand:
    """Tests for handle_backlog_command."""

    def test_no_subcommand(self, capsys):
        """Error when no subcommand specified."""
        from cortical.got.cli.backlog import handle_backlog_command

        args = Mock()
        args.backlog_command = None
        mock_manager = Mock()

        result = handle_backlog_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No backlog subcommand" in captured.out

    def test_unknown_subcommand(self, capsys):
        """Error when unknown subcommand specified."""
        from cortical.got.cli.backlog import handle_backlog_command

        args = Mock()
        args.backlog_command = "unknown"
        mock_manager = Mock()

        result = handle_backlog_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown backlog subcommand" in captured.out

    def test_routes_to_list(self, capsys):
        """Routes 'list' command to cmd_backlog_list."""
        from cortical.got.cli.backlog import handle_backlog_command

        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = []

        mock_got = Mock()
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.backlog_command = "list"
            args.json = False
            result = handle_backlog_command(args, mock_manager)

        assert result == 0

    def test_routes_to_stats(self, capsys):
        """Routes 'stats' command to cmd_backlog_stats."""
        from cortical.got.cli.backlog import handle_backlog_command

        mock_detector = Mock()
        mock_detector.generate_orphan_report.return_value = MockOrphanReport(
            orphan_tasks=[], total_tasks=0, orphan_rate=0.0
        )

        mock_got = Mock()
        mock_manager = Mock()
        mock_manager._manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.backlog_command = "stats"
            args.json = False
            result = handle_backlog_command(args, mock_manager)

        assert result == 0


class TestManagerAccess:
    """Tests for manager access patterns."""

    def test_uses_got_manager_fallback(self, capsys):
        """Uses _got_manager when _manager is None."""
        from cortical.got.cli.backlog import cmd_backlog_list

        mock_detector = Mock()
        mock_detector.find_orphan_tasks.return_value = []

        mock_got = Mock()
        mock_manager = Mock()
        mock_manager._manager = None
        mock_manager._got_manager = mock_got

        with patch('cortical.got.cli.backlog.OrphanDetector', return_value=mock_detector):
            args = Mock()
            args.json = False
            result = cmd_backlog_list(args, mock_manager)

        assert result == 0
