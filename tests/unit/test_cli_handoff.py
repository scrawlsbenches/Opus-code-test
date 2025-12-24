"""
Unit tests for cortical/got/cli/handoff.py

Tests the CLI command handlers for handoff operations without
using real GoT data or file I/O.

Updated to use TX backend methods (manager.initiate_handoff, etc.)
instead of EventLog/HandoffManager.
"""

import pytest
from unittest.mock import MagicMock
from argparse import Namespace

from cortical.got.cli.handoff import (
    cmd_handoff_initiate,
    cmd_handoff_accept,
    cmd_handoff_complete,
    cmd_handoff_list,
    handle_handoff_command,
)


@pytest.fixture
def mock_manager():
    """Create a mock GoTProjectManager with TX backend methods."""
    manager = MagicMock()
    manager.events_dir = "/fake/events"
    # Mock TX backend methods
    manager.initiate_handoff = MagicMock(return_value="H-20251223-123456-abc123")
    manager.accept_handoff = MagicMock(return_value=True)
    manager.complete_handoff = MagicMock(return_value=True)
    manager.list_handoffs = MagicMock(return_value=[])
    return manager


@pytest.fixture
def mock_task():
    """Create a mock task."""
    task = MagicMock()
    task.content = "Implement feature X"
    task.properties = {
        "status": "in_progress",
        "priority": "high"
    }
    return task


class TestHandoffInitiate:
    """Tests for cmd_handoff_initiate command."""

    def test_initiate_success(self, mock_manager, mock_task, capsys):
        """Test successful handoff initiation."""
        mock_manager.get_task.return_value = mock_task
        mock_manager.initiate_handoff.return_value = "H-20251223-123456-abc123"

        args = Namespace(
            task_id="T-123",
            target="sub-agent-1",
            source="main",
            instructions="Please implement this feature"
        )

        result = cmd_handoff_initiate(args, mock_manager)

        assert result == 0
        mock_manager.get_task.assert_called_once_with("T-123")
        mock_manager.initiate_handoff.assert_called_once_with(
            source_agent="main",
            target_agent="sub-agent-1",
            task_id="T-123",
            context={
                "task_title": "Implement feature X",
                "task_status": "in_progress",
                "task_priority": "high",
            },
            instructions="Please implement this feature",
        )

        captured = capsys.readouterr()
        assert "H-20251223-123456-abc123" in captured.out
        assert "Implement feature X" in captured.out
        assert "From: main" in captured.out
        assert "To: sub-agent-1" in captured.out
        assert "Please implement this feature" in captured.out

    def test_initiate_without_instructions(self, mock_manager, mock_task):
        """Test handoff initiation without instructions."""
        mock_manager.get_task.return_value = mock_task
        mock_manager.initiate_handoff.return_value = "H-123"

        args = Namespace(
            task_id="T-123",
            target="sub-agent-1",
            source="main",
            instructions=""
        )

        result = cmd_handoff_initiate(args, mock_manager)

        assert result == 0
        mock_manager.initiate_handoff.assert_called_once()
        call_kwargs = mock_manager.initiate_handoff.call_args[1]
        assert call_kwargs["instructions"] == ""

    def test_initiate_task_not_found(self, mock_manager, capsys):
        """Test handoff initiation when task doesn't exist."""
        mock_manager.get_task.return_value = None

        args = Namespace(
            task_id="T-NONEXISTENT",
            target="sub-agent-1",
            source="main",
            instructions=""
        )

        result = cmd_handoff_initiate(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Task not found: T-NONEXISTENT" in captured.out


class TestHandoffAccept:
    """Tests for cmd_handoff_accept command."""

    def test_accept_success(self, mock_manager, capsys):
        """Test successful handoff acceptance."""
        mock_manager.accept_handoff.return_value = True

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            message="Got it, starting work"
        )

        result = cmd_handoff_accept(args, mock_manager)

        assert result == 0
        mock_manager.accept_handoff.assert_called_once_with(
            handoff_id="H-123",
            agent="sub-agent-1",
            acknowledgment="Got it, starting work"
        )

        captured = capsys.readouterr()
        assert "Handoff accepted: H-123" in captured.out
        assert "Agent: sub-agent-1" in captured.out

    def test_accept_without_message(self, mock_manager):
        """Test handoff acceptance without acknowledgment message."""
        mock_manager.accept_handoff.return_value = True

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            message=""
        )

        result = cmd_handoff_accept(args, mock_manager)

        assert result == 0
        mock_manager.accept_handoff.assert_called_once()
        call_kwargs = mock_manager.accept_handoff.call_args[1]
        assert call_kwargs["acknowledgment"] == ""

    def test_accept_failure(self, mock_manager, capsys):
        """Test handoff acceptance failure."""
        mock_manager.accept_handoff.return_value = False

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            message=""
        )

        result = cmd_handoff_accept(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to accept handoff" in captured.out


class TestHandoffComplete:
    """Tests for cmd_handoff_complete command."""

    def test_complete_with_json_result(self, mock_manager, capsys):
        """Test handoff completion with valid JSON result."""
        mock_manager.complete_handoff.return_value = True

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result='{"status": "success", "files_modified": 3}',
            artifacts=["file1.py", "file2.py"]
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 0
        mock_manager.complete_handoff.assert_called_once_with(
            handoff_id="H-123",
            agent="sub-agent-1",
            result={"status": "success", "files_modified": 3},
            artifacts=["file1.py", "file2.py"]
        )

        captured = capsys.readouterr()
        assert "Handoff completed: H-123" in captured.out
        assert "Agent: sub-agent-1" in captured.out
        assert "status" in captured.out

    def test_complete_with_plain_text_result(self, mock_manager):
        """Test handoff completion with plain text result (gets wrapped in dict)."""
        mock_manager.complete_handoff.return_value = True

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result="Work completed successfully",
            artifacts=None
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 0
        mock_manager.complete_handoff.assert_called_once()
        call_kwargs = mock_manager.complete_handoff.call_args[1]
        assert call_kwargs["result"] == {"message": "Work completed successfully"}
        assert call_kwargs["artifacts"] == []

    def test_complete_without_artifacts(self, mock_manager):
        """Test handoff completion without artifacts."""
        mock_manager.complete_handoff.return_value = True

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result='{"status": "done"}',
            artifacts=None
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 0
        call_kwargs = mock_manager.complete_handoff.call_args[1]
        assert call_kwargs["artifacts"] == []

    def test_complete_failure(self, mock_manager, capsys):
        """Test handoff completion failure."""
        mock_manager.complete_handoff.return_value = False

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result="{}",
            artifacts=None
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to complete handoff" in captured.out


class TestHandoffList:
    """Tests for cmd_handoff_list command."""

    def test_list_no_handoffs(self, mock_manager, capsys):
        """Test listing when no handoffs exist."""
        mock_manager.list_handoffs.return_value = []

        args = Namespace(status=None)

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No handoffs found" in captured.out

    def test_list_all_handoffs(self, mock_manager, capsys):
        """Test listing all handoffs."""
        mock_manager.list_handoffs.return_value = [
            {
                "id": "H-123",
                "source_agent": "main",
                "target_agent": "sub-agent-1",
                "task_id": "T-456",
                "status": "initiated",
                "instructions": "Implement feature X"
            },
            {
                "id": "H-124",
                "source_agent": "main",
                "target_agent": "sub-agent-2",
                "task_id": "T-789",
                "status": "completed",
                "instructions": "Fix bug Y"
            }
        ]

        args = Namespace(status=None)

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        mock_manager.list_handoffs.assert_called_once_with(status=None)

        captured = capsys.readouterr()
        assert "Handoffs (2):" in captured.out
        assert "H-123" in captured.out
        assert "H-124" in captured.out
        assert "main → sub-agent-1" in captured.out
        assert "main → sub-agent-2" in captured.out

    def test_list_with_status_filter(self, mock_manager, capsys):
        """Test listing handoffs filtered by status."""
        mock_manager.list_handoffs.return_value = [
            {
                "id": "H-123",
                "source_agent": "main",
                "target_agent": "sub-agent-1",
                "task_id": "T-456",
                "status": "completed",
                "instructions": "Done"
            }
        ]

        args = Namespace(status="completed")

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        mock_manager.list_handoffs.assert_called_once_with(status="completed")

    def test_list_in_progress_alias_for_accepted(self, mock_manager, capsys):
        """Test that 'in_progress' is an alias for 'accepted' status.

        This matches the task terminology where users expect 'in_progress'
        to mean 'currently being worked on'. For handoffs, this maps to
        the 'accepted' status.
        """
        mock_manager.list_handoffs.return_value = [
            {
                "id": "H-123",
                "source_agent": "main",
                "target_agent": "sub-agent-1",
                "task_id": "T-456",
                "status": "accepted",
                "instructions": "Working on it"
            }
        ]

        # User passes in_progress (familiar from tasks)
        args = Namespace(status="in_progress")

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        # Should be normalized to 'accepted' when calling manager
        mock_manager.list_handoffs.assert_called_once_with(status="accepted")

        captured = capsys.readouterr()
        assert "Handoffs (1):" in captured.out
        assert "H-123" in captured.out

    def test_list_shows_truncated_instructions(self, mock_manager, capsys):
        """Test that long instructions are truncated in list output."""
        long_instructions = "A" * 100
        mock_manager.list_handoffs.return_value = [
            {
                "id": "H-123",
                "source_agent": "main",
                "target_agent": "sub-agent-1",
                "task_id": "T-456",
                "status": "initiated",
                "instructions": long_instructions
            }
        ]

        args = Namespace(status=None)

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        # Should be truncated to 50 chars + "..."
        assert "..." in captured.out


class TestHandleHandoffCommand:
    """Tests for handle_handoff_command routing."""

    def test_route_to_initiate(self, mock_manager, mock_task):
        """Test routing to initiate command."""
        mock_manager.get_task.return_value = mock_task

        args = Namespace(
            handoff_command="initiate",
            task_id="T-123",
            target="sub-agent-1",
            source="main",
            instructions=""
        )

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_manager.initiate_handoff.assert_called_once()

    def test_route_to_accept(self, mock_manager):
        """Test routing to accept command."""
        mock_manager.accept_handoff.return_value = True

        args = Namespace(
            handoff_command="accept",
            handoff_id="H-123",
            agent="sub-agent-1",
            message=""
        )

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_manager.accept_handoff.assert_called_once()

    def test_route_to_complete(self, mock_manager):
        """Test routing to complete command."""
        mock_manager.complete_handoff.return_value = True

        args = Namespace(
            handoff_command="complete",
            handoff_id="H-123",
            agent="sub-agent-1",
            result="{}",
            artifacts=None
        )

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_manager.complete_handoff.assert_called_once()

    def test_route_to_list(self, mock_manager):
        """Test routing to list command."""
        mock_manager.list_handoffs.return_value = []

        args = Namespace(
            handoff_command="list",
            status=None
        )

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_manager.list_handoffs.assert_called_once()

    def test_missing_subcommand(self, mock_manager, capsys):
        """Test error when handoff subcommand is missing."""
        args = Namespace()  # No handoff_command

        result = handle_handoff_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No handoff subcommand specified" in captured.out

    def test_unknown_subcommand(self, mock_manager, capsys):
        """Test error for unknown subcommand."""
        args = Namespace(handoff_command="unknown")

        result = handle_handoff_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown handoff subcommand" in captured.out
