"""
Unit tests for cortical/got/cli/decision.py

Tests the CLI command handlers for decision operations without
using real GoT data or file I/O.
"""

import pytest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from cortical.got.cli.decision import (
    cmd_decision_log,
    cmd_decision_list,
    cmd_decision_why,
    handle_decision_command,
)


class TestDecisionLog:
    """Tests for cmd_decision_log command."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        manager = MagicMock()
        return manager

    def test_log_basic_decision(self, mock_manager, capsys):
        """Test logging a basic decision."""
        mock_manager.log_decision.return_value = "D-20251223-123456-abc123"

        args = Namespace(
            decision="Use BM25 for scoring",
            rationale="Better relevance than TF-IDF",
            affects=None,
            alternatives=None,
            file=None
        )

        result = cmd_decision_log(args, mock_manager)

        assert result == 0
        mock_manager.log_decision.assert_called_once_with(
            decision="Use BM25 for scoring",
            rationale="Better relevance than TF-IDF",
            affects=None,
            alternatives=None,
            context=None,
        )

        captured = capsys.readouterr()
        assert "Decision logged: D-20251223-123456-abc123" in captured.out
        assert "Use BM25 for scoring" in captured.out
        assert "Better relevance than TF-IDF" in captured.out

    def test_log_with_affects(self, mock_manager, capsys):
        """Test logging a decision that affects specific tasks."""
        mock_manager.log_decision.return_value = "D-123"

        args = Namespace(
            decision="Refactor query module",
            rationale="Improve maintainability",
            affects=["T-001", "T-002", "T-003"],
            alternatives=None,
            file=None
        )

        result = cmd_decision_log(args, mock_manager)

        assert result == 0
        mock_manager.log_decision.assert_called_once()
        call_kwargs = mock_manager.log_decision.call_args[1]
        assert call_kwargs["affects"] == ["T-001", "T-002", "T-003"]

        captured = capsys.readouterr()
        assert "Affects: T-001, T-002, T-003" in captured.out

    def test_log_with_alternatives(self, mock_manager, capsys):
        """Test logging a decision with alternatives considered."""
        mock_manager.log_decision.return_value = "D-123"

        args = Namespace(
            decision="Use JSON for persistence",
            rationale="Human-readable and git-friendly",
            affects=None,
            alternatives=["pickle", "msgpack", "protobuf"],
            file=None
        )

        result = cmd_decision_log(args, mock_manager)

        assert result == 0
        call_kwargs = mock_manager.log_decision.call_args[1]
        assert call_kwargs["alternatives"] == ["pickle", "msgpack", "protobuf"]

        captured = capsys.readouterr()
        assert "Alternatives considered: pickle, msgpack, protobuf" in captured.out

    def test_log_with_file_context(self, mock_manager):
        """Test logging a decision with file context."""
        mock_manager.log_decision.return_value = "D-123"

        args = Namespace(
            decision="Add type hints",
            rationale="Improve IDE support",
            affects=None,
            alternatives=None,
            file="cortical/processor/core.py"
        )

        result = cmd_decision_log(args, mock_manager)

        assert result == 0
        call_kwargs = mock_manager.log_decision.call_args[1]
        assert call_kwargs["context"] == {"file": "cortical/processor/core.py"}

    def test_log_with_all_options(self, mock_manager, capsys):
        """Test logging a decision with all options specified."""
        mock_manager.log_decision.return_value = "D-123"

        args = Namespace(
            decision="Migrate to pytest",
            rationale="Better fixtures and parametrization",
            affects=["T-100", "T-101"],
            alternatives=["unittest", "nose2"],
            file="tests/conftest.py"
        )

        result = cmd_decision_log(args, mock_manager)

        assert result == 0
        mock_manager.log_decision.assert_called_once_with(
            decision="Migrate to pytest",
            rationale="Better fixtures and parametrization",
            affects=["T-100", "T-101"],
            alternatives=["unittest", "nose2"],
            context={"file": "tests/conftest.py"},
        )

        captured = capsys.readouterr()
        assert "Decision logged" in captured.out
        assert "Affects: T-100, T-101" in captured.out
        assert "Alternatives considered: unittest, nose2" in captured.out


class TestDecisionList:
    """Tests for cmd_decision_list command."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        return MagicMock()

    @pytest.fixture
    def sample_decisions(self):
        """Sample decision data."""
        decisions = []
        for i in range(3):
            decision = MagicMock()
            decision.id = f"D-00{i+1}"
            decision.content = f"Decision {i+1}"
            decision.properties = {
                "rationale": f"Rationale {i+1}",
                "alternatives": [f"alt{i+1}-1", f"alt{i+1}-2"]
            }
            decisions.append(decision)
        return decisions

    def test_list_no_decisions(self, mock_manager, capsys):
        """Test listing when no decisions exist."""
        mock_manager.list_decisions.return_value = []

        args = Namespace()

        result = cmd_decision_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No decisions logged yet" in captured.out

    def test_list_with_list_decisions_method(self, mock_manager, sample_decisions, capsys):
        """Test listing using list_decisions method (transactional backend)."""
        mock_manager.list_decisions.return_value = sample_decisions

        args = Namespace()

        result = cmd_decision_list(args, mock_manager)

        assert result == 0
        mock_manager.list_decisions.assert_called_once()

        captured = capsys.readouterr()
        assert "Decisions (3)" in captured.out
        assert "D-001" in captured.out
        assert "Decision 1" in captured.out
        assert "Rationale 1" in captured.out
        assert "Alternatives: alt1-1, alt1-2" in captured.out

    def test_list_with_get_decisions_fallback(self, mock_manager, sample_decisions, capsys):
        """Test listing using get_decisions fallback (legacy backend)."""
        # Simulate manager without list_decisions method
        del mock_manager.list_decisions
        mock_manager.get_decisions.return_value = sample_decisions

        args = Namespace()

        result = cmd_decision_list(args, mock_manager)

        assert result == 0
        mock_manager.get_decisions.assert_called_once()

        captured = capsys.readouterr()
        assert "Decisions (3)" in captured.out

    def test_list_without_alternatives(self, mock_manager, capsys):
        """Test listing decisions without alternatives."""
        decision = MagicMock()
        decision.id = "D-001"
        decision.content = "Simple decision"
        decision.properties = {
            "rationale": "Simple rationale"
        }

        mock_manager.list_decisions.return_value = [decision]

        args = Namespace()

        result = cmd_decision_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "D-001" in captured.out
        assert "Simple decision" in captured.out
        assert "Alternatives:" not in captured.out


class TestDecisionWhy:
    """Tests for cmd_decision_why command."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        return MagicMock()

    @pytest.fixture
    def sample_reasons(self):
        """Sample reasons for task creation."""
        return [
            {
                "decision_id": "D-001",
                "decision": "Implement caching",
                "rationale": "Improve performance",
                "alternatives": ["Redis", "Memcached"]
            },
            {
                "decision_id": "D-002",
                "decision": "Use in-memory cache",
                "rationale": "Simpler deployment",
                "alternatives": []
            }
        ]

    def test_why_no_reasons_found(self, mock_manager, capsys):
        """Test querying why when no reasons exist."""
        mock_manager.why.return_value = []

        args = Namespace(task_id="T-999")

        result = cmd_decision_why(args, mock_manager)

        assert result == 0
        mock_manager.why.assert_called_once_with("T-999")

        captured = capsys.readouterr()
        assert "No decisions found affecting T-999" in captured.out

    def test_why_with_reasons(self, mock_manager, sample_reasons, capsys):
        """Test querying why with reasons found."""
        mock_manager.why.return_value = sample_reasons

        args = Namespace(task_id="T-123")

        result = cmd_decision_why(args, mock_manager)

        assert result == 0
        mock_manager.why.assert_called_once_with("T-123")

        captured = capsys.readouterr()
        assert "Why T-123?" in captured.out
        assert "D-001" in captured.out
        assert "Implement caching" in captured.out
        assert "Improve performance" in captured.out
        assert "Alternatives: Redis, Memcached" in captured.out
        assert "D-002" in captured.out
        assert "Use in-memory cache" in captured.out

    def test_why_without_alternatives(self, mock_manager, capsys):
        """Test querying why when reasons have no alternatives."""
        reasons = [{
            "decision_id": "D-001",
            "decision": "Simple decision",
            "rationale": "Simple rationale",
            "alternatives": []
        }]
        mock_manager.why.return_value = reasons

        args = Namespace(task_id="T-123")

        result = cmd_decision_why(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "D-001" in captured.out
        # Should not show "Alternatives:" when list is empty
        assert "Alternatives:" not in captured.out


class TestHandleDecisionCommand:
    """Tests for handle_decision_command dispatcher."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTProjectManager."""
        return MagicMock()

    def test_no_subcommand(self, mock_manager, capsys):
        """Test when no decision subcommand is specified."""
        args = Namespace()

        result = handle_decision_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No decision subcommand specified" in captured.out

    def test_unknown_subcommand(self, mock_manager, capsys):
        """Test when unknown decision subcommand is specified."""
        args = Namespace(decision_command="invalid")

        result = handle_decision_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown decision subcommand: invalid" in captured.out

    @patch('cortical.got.cli.decision.cmd_decision_log')
    def test_routes_to_log(self, mock_cmd, mock_manager):
        """Test that 'log' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(decision_command="log")

        result = handle_decision_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.decision.cmd_decision_list')
    def test_routes_to_list(self, mock_cmd, mock_manager):
        """Test that 'list' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(decision_command="list")

        result = handle_decision_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.decision.cmd_decision_why')
    def test_routes_to_why(self, mock_cmd, mock_manager):
        """Test that 'why' command is routed correctly."""
        mock_cmd.return_value = 0
        args = Namespace(decision_command="why")

        result = handle_decision_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)
