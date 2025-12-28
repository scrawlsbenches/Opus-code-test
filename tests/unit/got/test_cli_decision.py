"""
Unit tests for cortical/got/cli/decision.py

Tests the decision CLI command handlers for:
- Logging decisions with rationale
- Listing decisions
- Showing decision details
- Querying why tasks exist
"""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from cortical.got.cli.decision import (
    cmd_decision_log,
    cmd_decision_list,
    cmd_decision_show,
    cmd_decision_why,
    setup_decision_parser,
    handle_decision_command,
    _prompt_task_linkage,
)


class TestCmdDecisionLog:
    """Tests for cmd_decision_log command handler."""

    def test_log_decision_basic(self, capsys):
        """Log a basic decision with rationale."""
        args = SimpleNamespace(
            decision="Use PostgreSQL for persistence",
            rationale="Better ACID compliance and mature ecosystem",
            affects=None,
            alternatives=None,
            file=None,
        )
        manager = MagicMock()
        manager.log_decision.return_value = "D-20251227-001"

        result = cmd_decision_log(args, manager)

        assert result == 0
        manager.log_decision.assert_called_once_with(
            decision="Use PostgreSQL for persistence",
            rationale="Better ACID compliance and mature ecosystem",
            affects=None,
            alternatives=None,
            context=None,
        )
        output = capsys.readouterr().out
        assert "D-20251227-001" in output
        assert "PostgreSQL" in output

    def test_log_decision_with_affects(self, capsys):
        """Log decision affecting specific tasks."""
        args = SimpleNamespace(
            decision="Refactor authentication",
            rationale="Security improvement",
            affects=["T-001", "T-002", "T-003"],
            alternatives=None,
            file=None,
        )
        manager = MagicMock()
        manager.log_decision.return_value = "D-001"

        result = cmd_decision_log(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "T-001" in output
        assert "T-002" in output
        assert "T-003" in output

    def test_log_decision_with_alternatives(self, capsys):
        """Log decision with considered alternatives."""
        args = SimpleNamespace(
            decision="Use REST API",
            rationale="Simpler implementation",
            affects=None,
            alternatives=["GraphQL", "gRPC"],
            file=None,
        )
        manager = MagicMock()
        manager.log_decision.return_value = "D-001"

        result = cmd_decision_log(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "GraphQL" in output
        assert "gRPC" in output

    def test_log_decision_with_file_context(self, capsys):
        """Log decision with file context."""
        args = SimpleNamespace(
            decision="Add type hints",
            rationale="Better IDE support",
            affects=None,
            alternatives=None,
            file="cortical/processor.py",
        )
        manager = MagicMock()
        manager.log_decision.return_value = "D-001"

        result = cmd_decision_log(args, manager)

        assert result == 0
        call_args = manager.log_decision.call_args
        assert call_args.kwargs["context"] == {"file": "cortical/processor.py"}

    def test_log_decision_full_options(self, capsys):
        """Log decision with all options."""
        args = SimpleNamespace(
            decision="Use async/await",
            rationale="Better I/O performance",
            affects=["T-001"],
            alternatives=["threading", "multiprocessing"],
            file="cortical/async_api.py",
        )
        manager = MagicMock()
        manager.log_decision.return_value = "D-001"

        result = cmd_decision_log(args, manager)

        assert result == 0
        call_args = manager.log_decision.call_args
        assert call_args.kwargs["affects"] == ["T-001"]
        assert call_args.kwargs["alternatives"] == ["threading", "multiprocessing"]
        assert call_args.kwargs["context"] == {"file": "cortical/async_api.py"}


class TestCmdDecisionList:
    """Tests for cmd_decision_list command handler."""

    def test_list_decisions_empty(self, capsys):
        """No decisions logged yet."""
        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_decisions.return_value = []

        result = cmd_decision_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "No decisions logged" in output

    def test_list_decisions_success(self, capsys):
        """Successfully listing decisions."""
        decision1 = MagicMock()
        decision1.id = "D-001"
        decision1.content = "Use PostgreSQL"
        decision1.properties = {"rationale": "ACID compliance"}

        decision2 = MagicMock()
        decision2.id = "D-002"
        decision2.content = "Use REST API"
        decision2.properties = {
            "rationale": "Simpler",
            "alternatives": ["GraphQL", "gRPC"],
        }

        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_decisions.return_value = [decision1, decision2]

        result = cmd_decision_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Decisions (2)" in output
        assert "D-001" in output
        assert "D-002" in output
        assert "PostgreSQL" in output
        assert "REST API" in output
        assert "GraphQL" in output

    def test_list_decisions_with_limit(self, capsys):
        """List decisions with limit."""
        decisions = [MagicMock(id=f"D-{i}", content=f"Decision {i}",
                               properties={"rationale": "reason"})
                     for i in range(10)]
        args = SimpleNamespace(limit=3)
        manager = MagicMock()
        manager.list_decisions.return_value = decisions

        result = cmd_decision_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Decisions (3)" in output

    def test_list_decisions_fallback_to_get_decisions(self, capsys):
        """Fallback to get_decisions when list_decisions not available."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Decision"
        decision.properties = {"rationale": "reason"}

        args = SimpleNamespace()
        manager = MagicMock(spec=[])  # Empty spec, no list_decisions
        manager.get_decisions = MagicMock(return_value=[decision])
        # Remove list_decisions to force fallback
        del manager.list_decisions

        result = cmd_decision_list(args, manager)

        assert result == 0
        manager.get_decisions.assert_called_once()


class TestCmdDecisionShow:
    """Tests for cmd_decision_show command handler."""

    def test_show_decision_not_found(self, capsys):
        """Decision not found."""
        args = SimpleNamespace(decision_id="D-MISSING")
        manager = MagicMock()
        manager.list_decisions.return_value = []

        result = cmd_decision_show(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Decision not found" in output

    def test_show_decision_basic(self, capsys):
        """Show basic decision details."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Use PostgreSQL"
        decision.properties = {
            "rationale": "ACID compliance",
            "created_at": "2025-12-27T10:00:00",
        }

        args = SimpleNamespace(decision_id="D-001")
        manager = MagicMock()
        manager.list_decisions.return_value = [decision]
        manager.get_task.return_value = None  # No affected tasks

        result = cmd_decision_show(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "D-001" in output
        assert "PostgreSQL" in output
        assert "ACID compliance" in output

    def test_show_decision_with_alternatives(self, capsys):
        """Show decision with alternatives."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Use REST"
        decision.properties = {
            "rationale": "Simpler",
            "alternatives": ["GraphQL", "gRPC"],
        }

        args = SimpleNamespace(decision_id="D-001")
        manager = MagicMock()
        manager.list_decisions.return_value = [decision]

        result = cmd_decision_show(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Alternatives Considered" in output
        assert "GraphQL" in output
        assert "gRPC" in output

    def test_show_decision_with_affected_tasks(self, capsys):
        """Show decision affecting tasks."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Refactor auth"
        decision.properties = {
            "rationale": "Security",
            "affects": ["T-001", "T-002"],
        }

        task1 = MagicMock()
        task1.content = "Implement OAuth"

        args = SimpleNamespace(decision_id="D-001")
        manager = MagicMock()
        manager.list_decisions.return_value = [decision]
        manager.get_task.side_effect = lambda id: task1 if id == "T-001" else None

        result = cmd_decision_show(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Affects" in output
        assert "T-001" in output
        assert "Implement OAuth" in output
        assert "T-002" in output

    def test_show_decision_with_context(self, capsys):
        """Show decision with context."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Add types"
        decision.properties = {
            "rationale": "IDE support",
            "context": {"file": "processor.py", "line": 42},
        }

        args = SimpleNamespace(decision_id="D-001")
        manager = MagicMock()
        manager.list_decisions.return_value = [decision]

        result = cmd_decision_show(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Context" in output
        assert "processor.py" in output

    def test_show_decision_fallback_to_get_decisions(self, capsys):
        """Fallback to get_decisions when list_decisions not available."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Decision"
        decision.properties = {"rationale": "reason"}

        args = SimpleNamespace(decision_id="D-001")
        manager = MagicMock(spec=[])
        manager.get_decisions = MagicMock(return_value=[decision])
        manager.get_task = MagicMock(return_value=None)

        result = cmd_decision_show(args, manager)

        assert result == 0


class TestCmdDecisionWhy:
    """Tests for cmd_decision_why command handler."""

    def test_why_no_decisions(self, capsys):
        """No decisions affecting task."""
        args = SimpleNamespace(task_id="T-001")
        manager = MagicMock()
        manager.why.return_value = []

        result = cmd_decision_why(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "No decisions found" in output

    def test_why_single_decision(self, capsys):
        """Single decision affecting task."""
        reasons = [{
            "decision_id": "D-001",
            "decision": "Use PostgreSQL",
            "rationale": "ACID compliance",
            "alternatives": [],
        }]

        args = SimpleNamespace(task_id="T-001")
        manager = MagicMock()
        manager.why.return_value = reasons

        result = cmd_decision_why(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Why T-001" in output
        assert "D-001" in output
        assert "PostgreSQL" in output
        assert "ACID" in output

    def test_why_multiple_decisions(self, capsys):
        """Multiple decisions affecting task."""
        reasons = [
            {
                "decision_id": "D-001",
                "decision": "Use PostgreSQL",
                "rationale": "ACID compliance",
                "alternatives": ["MySQL", "MongoDB"],
            },
            {
                "decision_id": "D-002",
                "decision": "Use async",
                "rationale": "Performance",
                "alternatives": [],
            },
        ]

        args = SimpleNamespace(task_id="T-001")
        manager = MagicMock()
        manager.why.return_value = reasons

        result = cmd_decision_why(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "D-001" in output
        assert "D-002" in output
        assert "MySQL" in output
        assert "MongoDB" in output


class TestSetupDecisionParser:
    """Tests for setup_decision_parser function."""

    def test_setup_creates_decision_subparser(self):
        """Decision subparser is created correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_decision_parser(subparsers)

        # Parse 'decision log' command
        args = parser.parse_args([
            'decision', 'log', 'Use PostgreSQL',
            '--rationale', 'ACID compliance',
        ])
        assert args.decision == 'Use PostgreSQL'
        assert args.rationale == 'ACID compliance'

    def test_setup_decision_log_with_options(self):
        """Decision log parser handles all options."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_decision_parser(subparsers)

        args = parser.parse_args([
            'decision', 'log', 'Use REST',
            '--rationale', 'Simpler',
            '--affects', 'T-001', 'T-002',
            '--alternatives', 'GraphQL', 'gRPC',
            '--file', 'api.py',
        ])
        assert args.decision == 'Use REST'
        assert args.affects == ['T-001', 'T-002']
        assert args.alternatives == ['GraphQL', 'gRPC']
        assert args.file == 'api.py'

    def test_setup_decision_list(self):
        """Decision list parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_decision_parser(subparsers)

        args = parser.parse_args(['decision', 'list', '--limit', '5'])
        assert args.limit == 5

    def test_setup_decision_show(self):
        """Decision show parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_decision_parser(subparsers)

        args = parser.parse_args(['decision', 'show', 'D-001'])
        assert args.decision_id == 'D-001'

    def test_setup_decision_why(self):
        """Decision why parser works."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_decision_parser(subparsers)

        args = parser.parse_args(['decision', 'why', 'T-001'])
        assert args.task_id == 'T-001'


class TestHandleDecisionCommand:
    """Tests for handle_decision_command routing function."""

    def test_handle_log_command(self):
        """Routes to log handler."""
        args = SimpleNamespace(
            decision_command="log",
            decision="Use PostgreSQL",
            rationale="ACID",
            affects=None,
            alternatives=None,
            file=None,
        )
        manager = MagicMock()
        manager.log_decision.return_value = "D-001"

        result = handle_decision_command(args, manager)

        assert result == 0

    def test_handle_list_command(self):
        """Routes to list handler."""
        args = SimpleNamespace(decision_command="list")
        manager = MagicMock()
        manager.list_decisions.return_value = []

        result = handle_decision_command(args, manager)

        assert result == 0

    def test_handle_show_command(self):
        """Routes to show handler."""
        args = SimpleNamespace(
            decision_command="show",
            decision_id="D-001",
        )
        manager = MagicMock()
        manager.list_decisions.return_value = []

        result = handle_decision_command(args, manager)

        assert result == 1  # Not found

    def test_handle_why_command(self):
        """Routes to why handler."""
        args = SimpleNamespace(
            decision_command="why",
            task_id="T-001",
        )
        manager = MagicMock()
        manager.why.return_value = []

        result = handle_decision_command(args, manager)

        assert result == 0

    def test_handle_no_subcommand(self, capsys):
        """No subcommand returns error."""
        args = SimpleNamespace()
        manager = MagicMock()

        result = handle_decision_command(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "No decision subcommand specified" in output

    def test_handle_none_subcommand(self, capsys):
        """None subcommand returns error."""
        args = SimpleNamespace(decision_command=None)
        manager = MagicMock()

        result = handle_decision_command(args, manager)

        assert result == 1

    def test_handle_unknown_subcommand(self, capsys):
        """Unknown subcommand returns error."""
        args = SimpleNamespace(decision_command="unknown")
        manager = MagicMock()

        result = handle_decision_command(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Unknown decision subcommand" in output


class TestPromptTaskLinkage:
    """Tests for _prompt_task_linkage helper function."""

    def test_prompt_with_tasks_select_by_number(self, monkeypatch, capsys):
        """Select task by number."""
        # Create mock task
        mock_task = MagicMock()
        mock_task.id = "T-20251228-001"
        mock_task.content = "Test task for linking"

        # Create mock manager
        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        # Mock user input (select first task)
        monkeypatch.setattr('builtins.input', lambda _: '1')

        _prompt_task_linkage("D-20251228-001", manager)

        # Verify edge was created
        manager.add_edge.assert_called_once_with(
            source_id="D-20251228-001",
            target_id="T-20251228-001",
            edge_type="JUSTIFIES"
        )

        output = capsys.readouterr().out
        assert "Link to a task" in output
        assert "T-20251228-001" in output
        assert "✓ Linked" in output

    def test_prompt_with_tasks_select_by_task_id(self, monkeypatch, capsys):
        """Select task by entering task ID directly."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-002"
        mock_task.content = "Another test task"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        # Mock user input (enter task ID directly)
        monkeypatch.setattr('builtins.input', lambda _: 'T-20251228-002')

        _prompt_task_linkage("D-20251228-002", manager)

        manager.add_edge.assert_called_once_with(
            source_id="D-20251228-002",
            target_id="T-20251228-002",
            edge_type="JUSTIFIES"
        )

    def test_prompt_with_tasks_skip(self, monkeypatch, capsys):
        """User presses Enter to skip linking."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-003"
        mock_task.content = "Task to skip"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        # Mock user input (empty string = skip)
        monkeypatch.setattr('builtins.input', lambda _: '')

        _prompt_task_linkage("D-20251228-003", manager)

        # Verify no edge was created
        manager.add_edge.assert_not_called()

    def test_prompt_with_no_tasks(self, capsys):
        """No prompt when no in-progress tasks exist."""
        manager = MagicMock()
        manager.list_tasks.return_value = []

        _prompt_task_linkage("D-20251228-004", manager)

        # Verify no prompt was shown
        output = capsys.readouterr().out
        assert "Link to a task" not in output
        manager.add_edge.assert_not_called()

    def test_prompt_with_invalid_number(self, monkeypatch, capsys):
        """Invalid number selection."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-005"
        mock_task.content = "Task"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        # Mock user input (invalid number)
        monkeypatch.setattr('builtins.input', lambda _: '99')

        _prompt_task_linkage("D-20251228-005", manager)

        output = capsys.readouterr().out
        assert "Invalid selection" in output
        manager.add_edge.assert_not_called()

    def test_prompt_with_multiple_tasks(self, monkeypatch, capsys):
        """Display multiple tasks limited to 5."""
        # Create 7 mock tasks
        mock_tasks = []
        for i in range(7):
            task = MagicMock()
            task.id = f"T-20251228-{i:03d}"
            task.content = f"Task {i}"
            mock_tasks.append(task)

        manager = MagicMock()
        manager.list_tasks.return_value = mock_tasks

        # Mock user input (select third task)
        monkeypatch.setattr('builtins.input', lambda _: '3')

        _prompt_task_linkage("D-20251228-006", manager)

        output = capsys.readouterr().out
        # Should only show first 5 tasks
        assert "T-20251228-000" in output
        assert "T-20251228-004" in output
        assert "T-20251228-006" not in output  # 7th task not shown

        # Should create edge to third task
        manager.add_edge.assert_called_once_with(
            source_id="D-20251228-006",
            target_id="T-20251228-002",
            edge_type="JUSTIFIES"
        )

    def test_prompt_with_long_task_title(self, monkeypatch, capsys):
        """Long task titles are truncated."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-007"
        mock_task.content = "This is a very long task title that should be truncated to fit nicely in the display"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        monkeypatch.setattr('builtins.input', lambda _: '')

        _prompt_task_linkage("D-20251228-007", manager)

        output = capsys.readouterr().out
        # Check that title is truncated (should end with "...")
        assert "..." in output
        # Full text shouldn't be in output
        assert "fit nicely in the display" not in output

    def test_prompt_handles_eoferror(self, monkeypatch):
        """Handles EOFError gracefully (non-interactive mode)."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-008"
        mock_task.content = "Task"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        # Mock EOFError (e.g., when stdin is closed)
        def raise_eoferror(_):
            raise EOFError

        monkeypatch.setattr('builtins.input', raise_eoferror)

        # Should not raise exception
        _prompt_task_linkage("D-20251228-008", manager)

        manager.add_edge.assert_not_called()

    def test_prompt_handles_keyboard_interrupt(self, monkeypatch):
        """Handles KeyboardInterrupt gracefully (Ctrl+C)."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-009"
        mock_task.content = "Task"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]

        # Mock KeyboardInterrupt
        def raise_keyboard_interrupt(_):
            raise KeyboardInterrupt

        monkeypatch.setattr('builtins.input', raise_keyboard_interrupt)

        # Should not raise exception
        _prompt_task_linkage("D-20251228-009", manager)

        manager.add_edge.assert_not_called()

    def test_prompt_handles_edge_creation_error(self, monkeypatch, capsys):
        """Handles error during edge creation gracefully."""
        mock_task = MagicMock()
        mock_task.id = "T-20251228-010"
        mock_task.content = "Task"

        manager = MagicMock()
        manager.list_tasks.return_value = [mock_task]
        manager.add_edge.side_effect = Exception("Edge creation failed")

        monkeypatch.setattr('builtins.input', lambda _: '1')

        # Should not raise exception
        _prompt_task_linkage("D-20251228-010", manager)

        output = capsys.readouterr().out
        assert "✗ Failed to create edge" in output
        assert "Edge creation failed" in output
