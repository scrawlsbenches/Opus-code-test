"""
Unit tests for cortical.got.cli.query module.

Tests all query-related CLI commands with mocked manager.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from argparse import Namespace
from datetime import datetime

from cortical.got.cli.query import (
    cmd_query,
    cmd_blocked,
    cmd_active,
    cmd_stats,
    cmd_dashboard,
    cmd_validate,
    cmd_infer,
    cmd_compact,
    cmd_export,
)


class TestCmdQuery:
    """Tests for cmd_query function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_query_with_results(self, mock_manager, capsys):
        """Test query command with results."""
        mock_manager.query.return_value = [
            {"id": "T-001", "title": "Task 1", "status": "pending", "priority": "high"},
            {"id": "T-002", "title": "Task 2", "status": "in_progress"},
        ]
        args = Namespace(query_string=["what", "blocks", "T-001"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        mock_manager.query.assert_called_once_with("what blocks T-001")
        captured = capsys.readouterr()
        assert "Results (2)" in captured.out
        assert "T-001" in captured.out
        assert "T-002" in captured.out

    def test_query_no_results(self, mock_manager, capsys):
        """Test query command with no results."""
        mock_manager.query.return_value = []
        args = Namespace(query_string=["active", "tasks"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_query_path_results(self, mock_manager, capsys):
        """Test query with path results."""
        mock_manager.query.return_value = [
            {"step": 1, "id": "T-001", "title": "Start"},
            {"step": 2, "id": "T-002", "title": "Middle"},
            {"step": 3, "id": "T-003", "title": "End"},
        ]
        args = Namespace(query_string=["path", "from", "T-001", "to", "T-003"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "[2]" in captured.out
        assert "[3]" in captured.out

    def test_query_relationship_results(self, mock_manager, capsys):
        """Test query with relationship results."""
        mock_manager.query.return_value = [
            {"relation": "BLOCKS", "id": "T-001", "title": "Blocker"},
            {"relation": "DEPENDS_ON", "id": "T-002", "title": "Dependency"},
        ]
        args = Namespace(query_string=["relationships", "T-003"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "BLOCKS" in captured.out
        assert "DEPENDS_ON" in captured.out

    def test_query_blocked_results(self, mock_manager, capsys):
        """Test query with blocked task results."""
        mock_manager.query.return_value = [
            {"id": "T-001", "title": "Blocked Task", "reason": "Missing dependency"},
        ]
        args = Namespace(query_string=["blocked", "tasks"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "T-001" in captured.out
        assert "Missing dependency" in captured.out


class TestCmdBlocked:
    """Tests for cmd_blocked function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_blocked_with_tasks(self, mock_manager, capsys):
        """Test blocked command with tasks."""
        mock_task1 = MagicMock()
        mock_task1.id = "T-001"
        mock_task1.content = "Blocked Task 1"

        mock_task2 = MagicMock()
        mock_task2.id = "T-002"
        mock_task2.content = "Blocked Task 2"

        mock_manager.get_blocked_tasks.return_value = [
            (mock_task1, "Missing dependency"),
            (mock_task2, "Waiting for review"),
        ]
        args = Namespace()

        result = cmd_blocked(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Blocked Tasks (2)" in captured.out
        assert "T-001" in captured.out
        assert "Missing dependency" in captured.out
        assert "T-002" in captured.out
        assert "Waiting for review" in captured.out

    def test_blocked_no_tasks(self, mock_manager, capsys):
        """Test blocked command with no tasks."""
        mock_manager.get_blocked_tasks.return_value = []
        args = Namespace()

        result = cmd_blocked(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No blocked tasks" in captured.out


class TestCmdActive:
    """Tests for cmd_active function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    @patch('cortical.got.cli.query.format_task_table')
    def test_active_tasks(self, mock_format, mock_manager, capsys):
        """Test active command."""
        mock_tasks = [MagicMock(), MagicMock()]
        mock_manager.get_active_tasks.return_value = mock_tasks
        mock_format.return_value = "Formatted Task Table"
        args = Namespace()

        result = cmd_active(args, mock_manager)

        assert result == 0
        mock_format.assert_called_once_with(mock_tasks)
        captured = capsys.readouterr()
        assert "Formatted Task Table" in captured.out


class TestCmdStats:
    """Tests for cmd_stats function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_stats_output(self, mock_manager, capsys):
        """Test stats command output."""
        mock_manager.get_stats.return_value = {
            "total_tasks": 10,
            "total_sprints": 3,
            "total_epics": 2,
            "total_edges": 25,
            "tasks_by_status": {
                "pending": 5,
                "in_progress": 3,
                "completed": 2,
            },
        }
        args = Namespace()

        result = cmd_stats(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Total tasks: 10" in captured.out
        assert "Total sprints: 3" in captured.out
        assert "Total epics: 2" in captured.out
        assert "Total edges: 25" in captured.out
        assert "pending: 5" in captured.out
        assert "in_progress: 3" in captured.out
        assert "completed: 2" in captured.out


class TestCmdDashboard:
    """Tests for cmd_dashboard function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_dashboard_success(self, mock_manager, capsys):
        """Test dashboard command success."""
        args = Namespace()

        with patch('scripts.got_dashboard.render_dashboard') as mock_render:
            mock_render.return_value = "Dashboard Content"
            result = cmd_dashboard(args, mock_manager)

        assert result == 0
        mock_render.assert_called_once_with(mock_manager)
        captured = capsys.readouterr()
        assert "Dashboard Content" in captured.out

    def test_dashboard_import_error(self, mock_manager, capsys):
        """Test dashboard command with import error."""
        args = Namespace()

        # Mock the import to fail
        with patch.dict('sys.modules', {'scripts.got_dashboard': None}):
            result = cmd_dashboard(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Could not import dashboard module" in captured.out

    def test_dashboard_render_error(self, mock_manager, capsys):
        """Test dashboard command with render error."""
        args = Namespace()

        with patch('scripts.got_dashboard.render_dashboard', side_effect=Exception("Render failed")):
            result = cmd_dashboard(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error rendering dashboard" in captured.out


class TestCmdValidate:
    """Tests for cmd_validate function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager with graph."""
        from cortical.reasoning.graph_of_thought import NodeType

        manager = MagicMock()
        manager.events_dir = "/fake/events"

        # Create mock nodes
        task1 = MagicMock()
        task1.node_type = NodeType.TASK
        task2 = MagicMock()
        task2.node_type = NodeType.TASK

        manager.graph.nodes = {
            "T-001": task1,
            "T-002": task2,
        }

        # Create mock edges
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"

        manager.graph.edges = [edge1]

        return manager

    def test_validate_healthy(self, mock_manager, capsys):
        """Test validate command with healthy graph."""
        args = Namespace()

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.load_all_events.return_value = [
                {"event": "node.create"},
                {"event": "edge.create"},
            ]
            result = cmd_validate(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "HEALTHY" in captured.out
        assert "Nodes: 2" in captured.out
        assert "Tasks: 2" in captured.out

    def test_validate_edge_discrepancy(self, mock_manager, capsys):
        """Test validate with edge discrepancy."""
        args = Namespace()

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            # Create events that suggest more edges should exist
            mock_event_log.load_all_events.return_value = [
                {"event": "edge.create"},
                {"event": "edge.create"},
                {"event": "edge.create"},
            ]
            result = cmd_validate(args, mock_manager)

        # Should detect discrepancy
        captured = capsys.readouterr()
        assert "EDGE DISCREPANCY" in captured.out or "edge loss" in captured.out

    def test_validate_high_orphan_rate(self, mock_manager, capsys):
        """Test validate with high orphan rate."""
        # Many nodes, few edges
        mock_manager.graph.nodes = {f"T-{i:03d}": MagicMock() for i in range(100)}
        mock_manager.graph.edges = []  # No edges

        args = Namespace()

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.load_all_events.return_value = []
            result = cmd_validate(args, mock_manager)

        # Should detect high orphan rate
        captured = capsys.readouterr()
        assert "ORPHAN RATE" in captured.out or "orphan rate" in captured.out


class TestCmdInfer:
    """Tests for cmd_infer function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_infer_from_commits(self, mock_manager, capsys):
        """Test infer command from commits."""
        mock_manager.infer_edges_from_recent_commits.return_value = [
            {
                "commit_hash": "abc123",
                "type": "IMPLEMENTS",
                "from": "abc123",
                "to": "T-001",
            },
        ]
        args = Namespace(commits=10, message=None)

        result = cmd_infer(args, mock_manager)

        assert result == 0
        mock_manager.infer_edges_from_recent_commits.assert_called_once_with(10)
        captured = capsys.readouterr()
        assert "Analyzed last 10 commits" in captured.out
        assert "abc123" in captured.out

    def test_infer_from_message(self, mock_manager, capsys):
        """Test infer command from specific message."""
        mock_manager.infer_edges_from_commit.return_value = [
            {"type": "IMPLEMENTS", "from": "commit", "to": "T-002"},
        ]
        args = Namespace(commits=10, message="feat: Add feature")

        result = cmd_infer(args, mock_manager)

        assert result == 0
        mock_manager.infer_edges_from_commit.assert_called_once_with("feat: Add feature")
        captured = capsys.readouterr()
        assert "Analyzing message" in captured.out

    def test_infer_no_results(self, mock_manager, capsys):
        """Test infer with no results."""
        mock_manager.infer_edges_from_recent_commits.return_value = []
        args = Namespace(commits=10, message=None)

        result = cmd_infer(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No task references found" in captured.out


class TestCmdCompact:
    """Tests for cmd_compact function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        manager.events_dir = "/fake/events"
        return manager

    def test_compact_dry_run(self, mock_manager, capsys):
        """Test compact command in dry run mode."""
        args = Namespace(dry_run=True, preserve_days=7, no_preserve_handoffs=False)

        mock_now = datetime(2025, 12, 23, 12, 0, 0)

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            with patch('cortical.got.cli.query.datetime') as mock_dt:
                mock_dt.utcnow.return_value = mock_now
                mock_event_log.load_all_events.return_value = [
                    {"event": "node.create", "ts": "2025-12-16T12:00:00Z"},  # Old
                    {"event": "node.create", "ts": "2025-12-22T12:00:00Z"},  # Recent
                    {"event": "handoff.initiate", "ts": "2025-12-15T12:00:00Z"},  # Handoff
                ]
                result = cmd_compact(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out
        assert "Total events: 3" in captured.out

    def test_compact_success(self, mock_manager, capsys):
        """Test compact command success."""
        args = Namespace(dry_run=False, preserve_days=7, no_preserve_handoffs=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.compact_events.return_value = {
                "status": "success",
                "nodes_written": 10,
                "edges_written": 20,
                "handoffs_preserved": 2,
                "files_removed": 5,
                "compact_file": "compact.jsonl",
                "original_event_count": 100,
                "old_events_consolidated": 80,
                "recent_events_kept": 20,
            }
            result = cmd_compact(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Event compaction complete" in captured.out
        assert "Nodes written: 10" in captured.out
        assert "Edges written: 20" in captured.out

    def test_compact_nothing_to_compact(self, mock_manager, capsys):
        """Test compact with nothing to compact."""
        args = Namespace(dry_run=False, preserve_days=7, no_preserve_handoffs=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.compact_events.return_value = {
                "status": "nothing_to_compact",
            }
            result = cmd_compact(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Nothing to compact" in captured.out

    def test_compact_error(self, mock_manager, capsys):
        """Test compact with error."""
        args = Namespace(dry_run=False, preserve_days=7, no_preserve_handoffs=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.compact_events.return_value = {
                "error": "Failed to compact",
            }
            result = cmd_compact(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Failed to compact" in captured.out


class TestCmdExport:
    """Tests for cmd_export function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_export_to_stdout(self, mock_manager, capsys):
        """Test export command to stdout."""
        mock_manager.export_graph.return_value = {
            "nodes": {"T-001": {"title": "Task 1"}},
            "edges": [],
        }
        args = Namespace(output=None)

        result = cmd_export(args, mock_manager)

        assert result == 0
        mock_manager.export_graph.assert_called_once_with(None)
        captured = capsys.readouterr()
        assert '"nodes"' in captured.out
        assert '"T-001"' in captured.out

    def test_export_to_file(self, mock_manager, capsys):
        """Test export command to file."""
        mock_manager.export_graph.return_value = {"nodes": {}, "edges": []}
        args = Namespace(output="/fake/output.json")

        from pathlib import Path
        with patch('pathlib.Path'):
            result = cmd_export(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Exported to:" in captured.out
