"""
Unit tests for cortical/got/cli/handoff.py

Tests the handoff CLI command handlers for:
- Initiating handoffs
- Accepting handoffs
- Completing handoffs
- Rejecting handoffs
- Listing and showing handoffs
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from io import StringIO

from cortical.got.cli.handoff import (
    cmd_handoff_initiate,
    cmd_handoff_accept,
    cmd_handoff_complete,
    cmd_handoff_reject,
    cmd_handoff_show,
    cmd_handoff_list,
    setup_handoff_parser,
    handle_handoff_command,
)


class TestCmdHandoffInitiate:
    """Tests for cmd_handoff_initiate command handler."""

    def test_initiate_handoff_success(self, capsys):
        """Successfully initiating a handoff."""
        task = MagicMock()
        task.content = "Fix the bug"
        task.properties = {"status": "pending", "priority": "high"}

        args = SimpleNamespace(
            task_id="T-001",
            source="main",
            target="sub-agent-1",
            instructions="Please review and fix",
        )
        manager = MagicMock()
        manager.get_task.return_value = task
        manager.initiate_handoff.return_value = "H-20251227-001"

        result = cmd_handoff_initiate(args, manager)

        assert result == 0
        manager.initiate_handoff.assert_called_once()
        output = capsys.readouterr().out
        assert "H-20251227-001" in output
        assert "Fix the bug" in output
        assert "main" in output
        assert "sub-agent-1" in output

    def test_initiate_handoff_task_not_found(self, capsys):
        """Initiating handoff for non-existent task."""
        args = SimpleNamespace(
            task_id="T-MISSING",
            source="main",
            target="sub-agent-1",
            instructions="",
        )
        manager = MagicMock()
        manager.get_task.return_value = None

        result = cmd_handoff_initiate(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Task not found" in output

    def test_initiate_handoff_stdin_instructions(self, capsys):
        """Reading instructions from stdin."""
        task = MagicMock()
        task.content = "Task title"
        task.properties = {}

        args = SimpleNamespace(
            task_id="T-001",
            source="main",
            target="sub-agent-1",
            instructions="-",  # Read from stdin
        )
        manager = MagicMock()
        manager.get_task.return_value = task
        manager.initiate_handoff.return_value = "H-001"

        with patch('sys.stdin', StringIO("Instructions from stdin")):
            result = cmd_handoff_initiate(args, manager)

        assert result == 0
        call_args = manager.initiate_handoff.call_args
        assert call_args.kwargs["instructions"] == "Instructions from stdin"

    def test_initiate_handoff_long_instructions_truncated(self, capsys):
        """Long instructions are truncated in display."""
        task = MagicMock()
        task.content = "Task"
        task.properties = {}

        long_instructions = "A" * 150  # > 100 chars
        args = SimpleNamespace(
            task_id="T-001",
            source="main",
            target="agent",
            instructions=long_instructions,
        )
        manager = MagicMock()
        manager.get_task.return_value = task
        manager.initiate_handoff.return_value = "H-001"

        result = cmd_handoff_initiate(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "..." in output  # Truncation


class TestCmdHandoffAccept:
    """Tests for cmd_handoff_accept command handler."""

    def test_accept_handoff_success(self, capsys):
        """Successfully accepting a handoff."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="sub-agent-1",
            message="Acknowledged",
        )
        manager = MagicMock()
        manager.accept_handoff.return_value = True

        result = cmd_handoff_accept(args, manager)

        assert result == 0
        manager.accept_handoff.assert_called_once_with(
            handoff_id="H-001",
            agent="sub-agent-1",
            acknowledgment="Acknowledged",
        )
        output = capsys.readouterr().out
        assert "Handoff accepted" in output

    def test_accept_handoff_failure(self, capsys):
        """Failed handoff acceptance."""
        args = SimpleNamespace(
            handoff_id="H-INVALID",
            agent="agent",
            message="",
        )
        manager = MagicMock()
        manager.accept_handoff.return_value = False

        result = cmd_handoff_accept(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Failed to accept handoff" in output


class TestCmdHandoffComplete:
    """Tests for cmd_handoff_complete command handler."""

    def test_complete_handoff_success_json_result(self, capsys):
        """Successfully completing a handoff with JSON result."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="sub-agent-1",
            result='{"status": "done", "files_changed": 3}',
            artifacts=["commit-abc123", "file.py"],
        )
        manager = MagicMock()
        manager.complete_handoff.return_value = True

        result = cmd_handoff_complete(args, manager)

        assert result == 0
        call_args = manager.complete_handoff.call_args
        assert call_args.kwargs["result"] == {"status": "done", "files_changed": 3}
        assert call_args.kwargs["artifacts"] == ["commit-abc123", "file.py"]
        output = capsys.readouterr().out
        assert "Handoff completed" in output

    def test_complete_handoff_string_result(self, capsys):
        """String result is wrapped in dict."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="agent",
            result="Task completed successfully",  # Not valid JSON
            artifacts=None,
        )
        manager = MagicMock()
        manager.complete_handoff.return_value = True

        result = cmd_handoff_complete(args, manager)

        assert result == 0
        call_args = manager.complete_handoff.call_args
        assert call_args.kwargs["result"] == {"message": "Task completed successfully"}

    def test_complete_handoff_failure(self, capsys):
        """Failed handoff completion."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="agent",
            result="{}",
            artifacts=None,
        )
        manager = MagicMock()
        manager.complete_handoff.return_value = False

        result = cmd_handoff_complete(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Failed to complete handoff" in output


class TestCmdHandoffReject:
    """Tests for cmd_handoff_reject command handler."""

    def test_reject_handoff_success(self, capsys):
        """Successfully rejecting a handoff."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="sub-agent-1",
            reason="Out of scope for this agent",
        )
        manager = MagicMock()
        manager.reject_handoff.return_value = True

        result = cmd_handoff_reject(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Handoff rejected" in output
        assert "Out of scope" in output

    def test_reject_handoff_stdin_reason(self, capsys):
        """Reading rejection reason from stdin."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="agent",
            reason="-",  # Read from stdin
        )
        manager = MagicMock()
        manager.reject_handoff.return_value = True

        with patch('sys.stdin', StringIO("Detailed rejection reason")):
            result = cmd_handoff_reject(args, manager)

        assert result == 0
        call_args = manager.reject_handoff.call_args
        assert call_args.kwargs["reason"] == "Detailed rejection reason"

    def test_reject_handoff_failure(self, capsys):
        """Failed handoff rejection."""
        args = SimpleNamespace(
            handoff_id="H-001",
            agent="agent",
            reason="reason",
        )
        manager = MagicMock()
        manager.reject_handoff.return_value = False

        result = cmd_handoff_reject(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Failed to reject handoff" in output


class TestCmdHandoffShow:
    """Tests for cmd_handoff_show command handler."""

    def test_show_handoff_not_found(self, capsys):
        """Handoff not found."""
        args = SimpleNamespace(handoff_id="H-MISSING")
        manager = MagicMock()
        manager.list_handoffs.return_value = []

        result = cmd_handoff_show(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Handoff not found" in output

    def test_show_handoff_success(self, capsys):
        """Successfully showing handoff details."""
        handoff = {
            "id": "H-001",
            "status": "accepted",
            "source_agent": "main",
            "target_agent": "sub-agent-1",
            "task_id": "T-001",
            "created_at": "2025-12-27T10:00:00",
            "accepted_at": "2025-12-27T10:05:00",
            "instructions": "Please complete this task",
            "context": {"priority": "high"},
        }
        args = SimpleNamespace(handoff_id="H-001")
        manager = MagicMock()
        manager.list_handoffs.return_value = [handoff]

        result = cmd_handoff_show(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "H-001" in output
        assert "accepted" in output
        assert "main" in output
        assert "sub-agent-1" in output
        assert "Please complete this task" in output
        assert "priority" in output

    def test_show_handoff_with_result(self, capsys):
        """Show handoff with completion result."""
        handoff = {
            "id": "H-001",
            "status": "completed",
            "source_agent": "main",
            "target_agent": "agent",
            "task_id": "T-001",
            "completed_at": "2025-12-27T12:00:00",
            "result": {"files_changed": 5},
            "artifacts": ["commit-123"],
        }
        args = SimpleNamespace(handoff_id="H-001")
        manager = MagicMock()
        manager.list_handoffs.return_value = [handoff]

        result = cmd_handoff_show(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "completed" in output
        assert "files_changed" in output
        assert "commit-123" in output


class TestCmdHandoffList:
    """Tests for cmd_handoff_list command handler."""

    def test_list_handoffs_empty(self, capsys):
        """No handoffs found."""
        args = SimpleNamespace(status=None)
        manager = MagicMock()
        manager.list_handoffs.return_value = []

        result = cmd_handoff_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "No handoffs found" in output

    def test_list_handoffs_success(self, capsys):
        """Successfully listing handoffs."""
        handoffs = [
            {
                "id": "H-001",
                "status": "initiated",
                "source_agent": "main",
                "target_agent": "agent-1",
                "task_id": "T-001",
                "instructions": "Do this task",
            },
            {
                "id": "H-002",
                "status": "accepted",
                "source_agent": "main",
                "target_agent": "agent-2",
                "task_id": "T-002",
            },
        ]
        args = SimpleNamespace(status=None)
        manager = MagicMock()
        manager.list_handoffs.return_value = handoffs

        result = cmd_handoff_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Handoffs (2)" in output
        assert "H-001" in output
        assert "H-002" in output

    def test_list_handoffs_with_status_filter(self, capsys):
        """List handoffs with status filter."""
        args = SimpleNamespace(status="accepted")
        manager = MagicMock()
        manager.list_handoffs.return_value = []

        result = cmd_handoff_list(args, manager)

        manager.list_handoffs.assert_called_once_with(status="accepted")

    def test_list_handoffs_in_progress_alias(self, capsys):
        """in_progress is normalized to accepted."""
        args = SimpleNamespace(status="in_progress")
        manager = MagicMock()
        manager.list_handoffs.return_value = []

        result = cmd_handoff_list(args, manager)

        manager.list_handoffs.assert_called_once_with(status="accepted")

    def test_list_handoffs_with_limit(self, capsys):
        """List handoffs with limit."""
        handoffs = [{"id": f"H-{i}", "status": "initiated",
                     "source_agent": "main", "target_agent": "agent",
                     "task_id": f"T-{i}"} for i in range(10)]
        args = SimpleNamespace(status=None, limit=3)
        manager = MagicMock()
        manager.list_handoffs.return_value = handoffs

        result = cmd_handoff_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Handoffs (3)" in output


class TestSetupHandoffParser:
    """Tests for setup_handoff_parser function."""

    def test_setup_creates_handoff_subparser(self):
        """Handoff subparser is created correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_handoff_parser(subparsers)

        # Parse 'handoff initiate' command
        args = parser.parse_args([
            'handoff', 'initiate', 'T-001',
            '--target', 'agent-1',
            '--source', 'main',
            '--instructions', 'Do it'
        ])
        assert args.task_id == 'T-001'
        assert args.target == 'agent-1'
        assert args.source == 'main'
        assert args.instructions == 'Do it'

    def test_setup_handoff_accept(self):
        """Handoff accept parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_handoff_parser(subparsers)

        args = parser.parse_args([
            'handoff', 'accept', 'H-001',
            '--agent', 'agent-1',
            '--message', 'Got it'
        ])
        assert args.handoff_id == 'H-001'
        assert args.agent == 'agent-1'
        assert args.message == 'Got it'

    def test_setup_handoff_complete(self):
        """Handoff complete parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_handoff_parser(subparsers)

        args = parser.parse_args([
            'handoff', 'complete', 'H-001',
            '--agent', 'agent-1',
            '--result', '{"done": true}',
            '--artifacts', 'file1.py', 'file2.py'
        ])
        assert args.handoff_id == 'H-001'
        assert args.result == '{"done": true}'
        assert args.artifacts == ['file1.py', 'file2.py']

    def test_setup_handoff_reject(self):
        """Handoff reject parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_handoff_parser(subparsers)

        args = parser.parse_args([
            'handoff', 'reject', 'H-001',
            '--agent', 'agent-1',
            '--reason', 'Cannot do this'
        ])
        assert args.handoff_id == 'H-001'
        assert args.reason == 'Cannot do this'

    def test_setup_handoff_list(self):
        """Handoff list parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_handoff_parser(subparsers)

        args = parser.parse_args([
            'handoff', 'list',
            '--status', 'accepted',
            '--limit', '5'
        ])
        assert args.status == 'accepted'
        assert args.limit == 5


class TestHandleHandoffCommand:
    """Tests for handle_handoff_command routing function."""

    def test_handle_initiate_command(self):
        """Routes to initiate handler."""
        task = MagicMock()
        task.content = "Task"
        task.properties = {}

        args = SimpleNamespace(
            handoff_command="initiate",
            task_id="T-001",
            source="main",
            target="agent",
            instructions="",
        )
        manager = MagicMock()
        manager.get_task.return_value = task
        manager.initiate_handoff.return_value = "H-001"

        result = handle_handoff_command(args, manager)

        assert result == 0

    def test_handle_accept_command(self):
        """Routes to accept handler."""
        args = SimpleNamespace(
            handoff_command="accept",
            handoff_id="H-001",
            agent="agent",
            message="",
        )
        manager = MagicMock()
        manager.accept_handoff.return_value = True

        result = handle_handoff_command(args, manager)

        assert result == 0

    def test_handle_complete_command(self):
        """Routes to complete handler."""
        args = SimpleNamespace(
            handoff_command="complete",
            handoff_id="H-001",
            agent="agent",
            result="{}",
            artifacts=None,
        )
        manager = MagicMock()
        manager.complete_handoff.return_value = True

        result = handle_handoff_command(args, manager)

        assert result == 0

    def test_handle_reject_command(self):
        """Routes to reject handler."""
        args = SimpleNamespace(
            handoff_command="reject",
            handoff_id="H-001",
            agent="agent",
            reason="No",
        )
        manager = MagicMock()
        manager.reject_handoff.return_value = True

        result = handle_handoff_command(args, manager)

        assert result == 0

    def test_handle_show_command(self):
        """Routes to show handler."""
        args = SimpleNamespace(
            handoff_command="show",
            handoff_id="H-001",
        )
        manager = MagicMock()
        manager.list_handoffs.return_value = []

        result = handle_handoff_command(args, manager)

        assert result == 1  # Not found

    def test_handle_list_command(self):
        """Routes to list handler."""
        args = SimpleNamespace(
            handoff_command="list",
            status=None,
        )
        manager = MagicMock()
        manager.list_handoffs.return_value = []

        result = handle_handoff_command(args, manager)

        assert result == 0

    def test_handle_no_subcommand(self, capsys):
        """No subcommand returns error."""
        args = SimpleNamespace()
        manager = MagicMock()

        result = handle_handoff_command(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "No handoff subcommand specified" in output

    def test_handle_none_subcommand(self, capsys):
        """None subcommand returns error."""
        args = SimpleNamespace(handoff_command=None)
        manager = MagicMock()

        result = handle_handoff_command(args, manager)

        assert result == 1

    def test_handle_unknown_subcommand(self, capsys):
        """Unknown subcommand returns error."""
        args = SimpleNamespace(handoff_command="unknown")
        manager = MagicMock()

        result = handle_handoff_command(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Unknown handoff subcommand" in output
