"""
Unit tests for cortical/got/cli/handoff.py

Tests the CLI command handlers for handoff operations without
using real GoT data or file I/O.
"""

import pytest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from cortical.got.cli.handoff import (
    cmd_handoff_initiate,
    cmd_handoff_accept,
    cmd_handoff_complete,
    cmd_handoff_list,
    handle_handoff_command,
)


class TestHandoffInitiate:
    """Tests for cmd_handoff_initiate command."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        manager = MagicMock()
        manager.events_dir = "/fake/events"
        return manager

    @pytest.fixture
    def mock_task(self):
        """Create a mock task."""
        task = MagicMock()
        task.content = "Implement feature X"
        task.properties = {
            "status": "in_progress",
            "priority": "high"
        }
        return task

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_initiate_success(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, mock_task, capsys):
        """Test successful handoff initiation."""
        # Setup
        mock_manager.get_task.return_value = mock_task
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr.initiate_handoff.return_value = "H-20251223-123456-abc123"
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            task_id="T-123",
            target="sub-agent-1",
            source="main",
            instructions="Please implement this feature"
        )

        # Execute
        result = cmd_handoff_initiate(args, mock_manager)

        # Assert
        assert result == 0
        mock_manager.get_task.assert_called_once_with("T-123")
        mock_handoff_mgr.initiate_handoff.assert_called_once_with(
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

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_initiate_without_instructions(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, mock_task):
        """Test handoff initiation without instructions."""
        mock_manager.get_task.return_value = mock_task
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr.initiate_handoff.return_value = "H-123"
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            task_id="T-123",
            target="sub-agent-1",
            source="main",
            instructions=""
        )

        result = cmd_handoff_initiate(args, mock_manager)

        assert result == 0
        mock_handoff_mgr.initiate_handoff.assert_called_once()
        call_kwargs = mock_handoff_mgr.initiate_handoff.call_args[1]
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

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        manager = MagicMock()
        manager.events_dir = "/fake/events"
        return manager

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_accept_success(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, capsys):
        """Test successful handoff acceptance."""
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            message="Got it, starting work"
        )

        result = cmd_handoff_accept(args, mock_manager)

        assert result == 0
        mock_handoff_mgr.accept_handoff.assert_called_once_with(
            handoff_id="H-123",
            agent="sub-agent-1",
            acknowledgment="Got it, starting work"
        )

        captured = capsys.readouterr()
        assert "Handoff accepted: H-123" in captured.out
        assert "Agent: sub-agent-1" in captured.out

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_accept_without_message(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager):
        """Test handoff acceptance without acknowledgment message."""
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            message=""
        )

        result = cmd_handoff_accept(args, mock_manager)

        assert result == 0
        mock_handoff_mgr.accept_handoff.assert_called_once()
        call_kwargs = mock_handoff_mgr.accept_handoff.call_args[1]
        assert call_kwargs["acknowledgment"] == ""


class TestHandoffComplete:
    """Tests for cmd_handoff_complete command."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        manager = MagicMock()
        manager.events_dir = "/fake/events"
        return manager

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_complete_with_json_result(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, capsys):
        """Test handoff completion with valid JSON result."""
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result='{"status": "success", "files_modified": 3}',
            artifacts=["file1.py", "file2.py"]
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 0
        mock_handoff_mgr.complete_handoff.assert_called_once_with(
            handoff_id="H-123",
            agent="sub-agent-1",
            result={"status": "success", "files_modified": 3},
            artifacts=["file1.py", "file2.py"]
        )

        captured = capsys.readouterr()
        assert "Handoff completed: H-123" in captured.out
        assert "Agent: sub-agent-1" in captured.out
        assert "status" in captured.out

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_complete_with_plain_text_result(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager):
        """Test handoff completion with plain text result (gets wrapped in dict)."""
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result="Work completed successfully",
            artifacts=None
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 0
        mock_handoff_mgr.complete_handoff.assert_called_once()
        call_kwargs = mock_handoff_mgr.complete_handoff.call_args[1]
        assert call_kwargs["result"] == {"message": "Work completed successfully"}
        assert call_kwargs["artifacts"] == []

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_complete_without_artifacts(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager):
        """Test handoff completion without artifacts."""
        mock_handoff_mgr = MagicMock()
        mock_handoff_mgr_cls.return_value = mock_handoff_mgr

        args = Namespace(
            handoff_id="H-123",
            agent="sub-agent-1",
            result="{}",
            artifacts=None
        )

        result = cmd_handoff_complete(args, mock_manager)

        assert result == 0
        call_kwargs = mock_handoff_mgr.complete_handoff.call_args[1]
        assert call_kwargs["artifacts"] == []


class TestHandoffList:
    """Tests for cmd_handoff_list command."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        manager = MagicMock()
        manager.events_dir = "/fake/events"
        return manager

    @pytest.fixture
    def sample_handoffs(self):
        """Sample handoff data."""
        return [
            {
                "id": "H-001",
                "status": "initiated",
                "source_agent": "main",
                "target_agent": "sub-1",
                "task_id": "T-123",
                "instructions": "Please implement authentication feature with OAuth2 support"
            },
            {
                "id": "H-002",
                "status": "accepted",
                "source_agent": "main",
                "target_agent": "sub-2",
                "task_id": "T-456",
                "instructions": "Fix bug in login flow"
            },
            {
                "id": "H-003",
                "status": "completed",
                "source_agent": "main",
                "target_agent": "sub-1",
                "task_id": "T-789",
            },
        ]

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_list_no_handoffs(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, capsys):
        """Test listing when no handoffs exist."""
        mock_event_log_cls.load_all_events.return_value = []
        mock_handoff_mgr_cls.load_handoffs_from_events.return_value = []

        args = Namespace(status=None)

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No handoffs found" in captured.out

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_list_all_handoffs(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, sample_handoffs, capsys):
        """Test listing all handoffs without status filter."""
        mock_event_log_cls.load_all_events.return_value = []
        mock_handoff_mgr_cls.load_handoffs_from_events.return_value = sample_handoffs

        args = Namespace(status=None)

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Handoffs (3)" in captured.out
        assert "H-001" in captured.out
        assert "H-002" in captured.out
        assert "H-003" in captured.out
        assert "→" in captured.out  # initiated icon
        assert "✓" in captured.out  # accepted icon
        assert "✓✓" in captured.out  # completed icon

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_list_with_status_filter(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, sample_handoffs, capsys):
        """Test listing handoffs filtered by status."""
        mock_event_log_cls.load_all_events.return_value = []
        mock_handoff_mgr_cls.load_handoffs_from_events.return_value = sample_handoffs

        args = Namespace(status="initiated")

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Handoffs (1)" in captured.out
        assert "H-001" in captured.out
        assert "H-002" not in captured.out
        assert "H-003" not in captured.out

    @patch('scripts.got_utils.EventLog')
    @patch('scripts.got_utils.HandoffManager')
    def test_list_shows_truncated_instructions(self, mock_handoff_mgr_cls, mock_event_log_cls, mock_manager, sample_handoffs, capsys):
        """Test that long instructions are truncated in list view."""
        mock_event_log_cls.load_all_events.return_value = []
        mock_handoff_mgr_cls.load_handoffs_from_events.return_value = sample_handoffs

        args = Namespace(status=None)

        result = cmd_handoff_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        # First handoff has long instructions
        assert "Please implement authentication feature with OAuth" in captured.out
        assert "..." in captured.out


class TestHandleHandoffCommand:
    """Tests for handle_handoff_command dispatcher."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        return MagicMock()

    def test_no_subcommand(self, mock_manager, capsys):
        """Test when no handoff subcommand is specified."""
        args = Namespace()

        result = handle_handoff_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No handoff subcommand specified" in captured.out

    def test_unknown_subcommand(self, mock_manager, capsys):
        """Test when unknown handoff subcommand is specified."""
        args = Namespace(handoff_command="invalid")

        result = handle_handoff_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown handoff subcommand: invalid" in captured.out

    @patch('cortical.got.cli.handoff.cmd_handoff_initiate')
    def test_routes_to_initiate(self, mock_cmd, mock_manager):
        """Test that 'initiate' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(handoff_command="initiate")

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.handoff.cmd_handoff_accept')
    def test_routes_to_accept(self, mock_cmd, mock_manager):
        """Test that 'accept' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(handoff_command="accept")

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.handoff.cmd_handoff_complete')
    def test_routes_to_complete(self, mock_cmd, mock_manager):
        """Test that 'complete' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(handoff_command="complete")

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.handoff.cmd_handoff_list')
    def test_routes_to_list(self, mock_cmd, mock_manager):
        """Test that 'list' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(handoff_command="list")

        result = handle_handoff_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)
