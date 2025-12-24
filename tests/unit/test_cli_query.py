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


class TestCmdQueryEdgeCases:
    """Tests for edge cases in cmd_query function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_query_relationship_without_title(self, mock_manager, capsys):
        """Test query with relationship result that has no title."""
        mock_manager.query.return_value = [
            {"relation": "BLOCKS", "id": "T-001"},  # No title field
        ]
        args = Namespace(query_string=["relationships", "T-002"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "BLOCKS" in captured.out
        assert "T-001" in captured.out

    def test_query_generic_result_without_status(self, mock_manager, capsys):
        """Test query with generic result that has no status."""
        mock_manager.query.return_value = [
            {"id": "T-001", "title": "Task without status", "priority": "high"},
        ]
        args = Namespace(query_string=["active", "tasks"])

        result = cmd_query(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "T-001" in captured.out
        assert "Priority: high" in captured.out


class TestCmdValidateEdgeCases:
    """Tests for edge cases in cmd_validate function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager with graph."""
        from cortical.reasoning.graph_of_thought import NodeType

        manager = MagicMock()
        manager.events_dir = "/fake/events"

        # Create mock nodes
        task1 = MagicMock()
        task1.node_type = NodeType.TASK

        manager.graph.nodes = {"T-001": task1}

        # Create mock edges
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"

        manager.graph.edges = [edge1]

        return manager

    def test_validate_edge_loss_warning(self, mock_manager, capsys):
        """Test validate with minor edge loss (between -10% and -5%)."""
        args = Namespace()

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            # Create events suggesting 2 edges should exist (actual is 1)
            # This is a -50% discrepancy, but we need -6% to hit the warning path
            # Expected edges = 17, actual = 16 → -5.88% (within warning range)
            mock_event_log.load_all_events.return_value = [
                {"event": "edge.create"} for _ in range(17)
            ]
            # Set actual edges to 16
            mock_manager.graph.edges = [MagicMock() for _ in range(16)]
            for i, edge in enumerate(mock_manager.graph.edges):
                edge.source_id = f"T-{i}"
                edge.target_id = f"T-{i+1}"

            result = cmd_validate(args, mock_manager)

        # Should have warning, not issue
        captured = capsys.readouterr()
        assert "edge loss" in captured.out or "Minor edge loss" in captured.out

    def test_validate_edge_surplus_warning(self, mock_manager, capsys):
        """Test validate with edge surplus (between +5% and +10%)."""
        args = Namespace()

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            # Expected edges = 10, actual = 11 → +10% (within surplus range)
            mock_event_log.load_all_events.return_value = [
                {"event": "edge.create"} for _ in range(10)
            ]
            # Set actual edges to 11
            mock_manager.graph.edges = [MagicMock() for _ in range(11)]
            for i, edge in enumerate(mock_manager.graph.edges):
                edge.source_id = f"T-{i}"
                edge.target_id = f"T-{i+1}"

            result = cmd_validate(args, mock_manager)

        # Should have warning about surplus
        captured = capsys.readouterr()
        assert "surplus" in captured.out or "Edge surplus" in captured.out

    def test_validate_moderate_orphan_rate(self, mock_manager, capsys):
        """Test validate with moderate orphan rate (between 25% and 50%)."""
        # Create 10 nodes with 3 edges → 4 connected nodes, 6 orphans → 60% orphan rate
        # Actually, let's do 10 nodes, 5 edges connecting 8 nodes → 2 orphans = 20%
        # Need 30% orphan rate: 10 nodes, 3 edges connecting 6 nodes → 4 orphans = 40%
        mock_manager.graph.nodes = {f"T-{i:03d}": MagicMock() for i in range(10)}

        # Create 3 edges connecting 6 nodes
        edges = []
        for i in range(3):
            edge = MagicMock()
            edge.source_id = f"T-{i*2:03d}"
            edge.target_id = f"T-{i*2+1:03d}"
            edges.append(edge)
        mock_manager.graph.edges = edges

        args = Namespace()

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.load_all_events.return_value = []
            result = cmd_validate(args, mock_manager)

        # Should have warning for moderate orphan rate (40% is between 25% and 50%)
        captured = capsys.readouterr()
        assert "Moderate orphan rate" in captured.out or "orphan rate" in captured.out


class TestSetupQueryParser:
    """Tests for setup_query_parser function."""

    def test_setup_query_parser(self):
        """Test setup_query_parser adds all subcommands."""
        from argparse import ArgumentParser
        from cortical.got.cli.query import setup_query_parser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        setup_query_parser(subparsers)

        # Test that we can parse each command
        args = parser.parse_args(["query", "what", "blocks", "T-001"])
        assert args.command == "query"
        assert args.query_string == ["what", "blocks", "T-001"]

        args = parser.parse_args(["blocked"])
        assert args.command == "blocked"

        args = parser.parse_args(["active"])
        assert args.command == "active"

        args = parser.parse_args(["stats"])
        assert args.command == "stats"

        args = parser.parse_args(["dashboard"])
        assert args.command == "dashboard"

        args = parser.parse_args(["validate"])
        assert args.command == "validate"

        args = parser.parse_args(["infer", "--commits", "20"])
        assert args.command == "infer"
        assert args.commits == 20

        args = parser.parse_args(["infer", "-m", "feat: Add feature"])
        assert args.command == "infer"
        assert args.message == "feat: Add feature"

        args = parser.parse_args(["compact", "--preserve-days", "14"])
        assert args.command == "compact"
        assert args.preserve_days == 14

        args = parser.parse_args(["compact", "--no-preserve-handoffs"])
        assert args.command == "compact"
        assert args.no_preserve_handoffs is True

        args = parser.parse_args(["compact", "--dry-run"])
        assert args.command == "compact"
        assert args.dry_run is True

        args = parser.parse_args(["export", "-o", "output.json"])
        assert args.command == "export"
        assert args.output == "output.json"


class TestHandleQueryCommands:
    """Tests for handle_query_commands function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_handle_query_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_query."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.query.return_value = []
        args = Namespace(command="query", query_string=["test"])

        result = handle_query_commands(args, mock_manager)

        assert result == 0
        mock_manager.query.assert_called_once()

    def test_handle_blocked_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_blocked."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.get_blocked_tasks.return_value = []
        args = Namespace(command="blocked")

        result = handle_query_commands(args, mock_manager)

        assert result == 0
        mock_manager.get_blocked_tasks.assert_called_once()

    def test_handle_active_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_active."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.get_active_tasks.return_value = []
        args = Namespace(command="active")

        with patch('cortical.got.cli.query.format_task_table', return_value=""):
            result = handle_query_commands(args, mock_manager)

        assert result == 0
        mock_manager.get_active_tasks.assert_called_once()

    def test_handle_stats_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_stats."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.get_stats.return_value = {
            "total_tasks": 0,
            "total_sprints": 0,
            "total_epics": 0,
            "total_edges": 0,
            "tasks_by_status": {},
        }
        args = Namespace(command="stats")

        result = handle_query_commands(args, mock_manager)

        assert result == 0
        mock_manager.get_stats.assert_called_once()

    def test_handle_dashboard_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_dashboard."""
        from cortical.got.cli.query import handle_query_commands

        args = Namespace(command="dashboard")

        with patch('scripts.got_dashboard.render_dashboard', return_value="Dashboard"):
            result = handle_query_commands(args, mock_manager)

        assert result == 0

    def test_handle_validate_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_validate."""
        from cortical.got.cli.query import handle_query_commands
        from cortical.reasoning.graph_of_thought import NodeType

        mock_manager.graph.nodes = {}
        mock_manager.graph.edges = []
        mock_manager.events_dir = "/fake/events"
        args = Namespace(command="validate")

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.load_all_events.return_value = []
            result = handle_query_commands(args, mock_manager)

        assert result == 0

    def test_handle_infer_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_infer."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.infer_edges_from_recent_commits.return_value = []
        args = Namespace(command="infer", commits=10, message=None)

        result = handle_query_commands(args, mock_manager)

        assert result == 0
        mock_manager.infer_edges_from_recent_commits.assert_called_once()

    def test_handle_compact_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_compact."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.events_dir = "/fake/events"
        args = Namespace(command="compact", dry_run=False, preserve_days=7, no_preserve_handoffs=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log:
            mock_event_log.compact_events.return_value = {"status": "nothing_to_compact"}
            result = handle_query_commands(args, mock_manager)

        assert result == 0

    def test_handle_export_command(self, mock_manager):
        """Test handle_query_commands routes to cmd_export."""
        from cortical.got.cli.query import handle_query_commands

        mock_manager.export_graph.return_value = {}
        args = Namespace(command="export", output=None)

        result = handle_query_commands(args, mock_manager)

        assert result == 0
        mock_manager.export_graph.assert_called_once()

    def test_handle_unknown_command(self, mock_manager):
        """Test handle_query_commands returns None for unknown command."""
        from cortical.got.cli.query import handle_query_commands

        args = Namespace(command="unknown")

        result = handle_query_commands(args, mock_manager)

        assert result is None
