"""
Tests for GoT CLI analyze commands.

Tests the analysis commands that use Query API, GraphWalker,
PathFinder, and PatternMatcher for graph analysis.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch, call
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cortical.got.cli.analyze import (
    cmd_analyze_summary,
    cmd_analyze_dependencies,
    cmd_analyze_patterns,
    cmd_analyze_orphans,
    _status_icon,
    _task_age_str,
    setup_analyze_parser,
    handle_analyze_command,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_manager():
    """Create a mock GoTManager."""
    manager = Mock()
    manager.get_task = Mock()
    return manager


@pytest.fixture
def mock_adapter(mock_manager):
    """Create a mock TransactionalGoTAdapter."""
    adapter = Mock()
    adapter._manager = mock_manager
    return adapter


@pytest.fixture
def mock_task():
    """Create a mock task."""
    task = Mock()
    task.id = "T-20251224-120000-abcd1234"
    task.title = "Test Task"
    task.status = "pending"
    task.priority = "high"
    task.created_at = "2025-12-24T12:00:00+00:00"
    return task


@pytest.fixture
def mock_sprint():
    """Create a mock sprint."""
    sprint = Mock()
    sprint.id = "S-sprint-001-test"
    sprint.title = "Test Sprint"
    sprint.status = "in_progress"
    return sprint


@pytest.fixture
def mock_decision():
    """Create a mock decision."""
    decision = Mock()
    decision.id = "D-20251224-120000-abcd1234"
    decision.title = "Test Decision"
    return decision


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestStatusIcon:
    """Test _status_icon helper function."""

    def test_completed_icon(self):
        """Test completed status returns correct icon."""
        assert _status_icon("completed") == "[x]"

    def test_in_progress_icon(self):
        """Test in_progress status returns correct icon."""
        assert _status_icon("in_progress") == "[~]"

    def test_pending_icon(self):
        """Test pending status returns correct icon."""
        assert _status_icon("pending") == "[ ]"

    def test_blocked_icon(self):
        """Test blocked status returns correct icon."""
        assert _status_icon("blocked") == "[!]"

    def test_unknown_status_icon(self):
        """Test unknown status returns default icon."""
        assert _status_icon("unknown_status") == "[?]"

    def test_empty_status_icon(self):
        """Test empty status returns default icon."""
        assert _status_icon("") == "[?]"


class TestTaskAgeStr:
    """Test _task_age_str helper function."""

    def test_recent_task(self):
        """Test task created less than an hour ago."""
        now = datetime.now(timezone.utc)
        recent = now - timedelta(minutes=30)
        assert _task_age_str(recent.isoformat()) == "recent"

    def test_hours_ago(self):
        """Test task created hours ago."""
        now = datetime.now(timezone.utc)
        hours_ago = now - timedelta(hours=5)
        result = _task_age_str(hours_ago.isoformat())
        assert "h ago" in result

    def test_days_ago(self):
        """Test task created days ago."""
        now = datetime.now(timezone.utc)
        days_ago = now - timedelta(days=5)
        result = _task_age_str(days_ago.isoformat())
        assert "5d ago" == result

    def test_months_ago(self):
        """Test task created months ago."""
        now = datetime.now(timezone.utc)
        months_ago = now - timedelta(days=65)
        result = _task_age_str(months_ago.isoformat())
        assert "2mo ago" == result

    def test_datetime_object_input(self):
        """Test with datetime object instead of string."""
        now = datetime.now(timezone.utc)
        days_ago = now - timedelta(days=3)
        result = _task_age_str(days_ago)
        assert "3d ago" == result

    def test_invalid_date_string(self):
        """Test with invalid date string."""
        result = _task_age_str("not-a-date")
        assert result == "unknown"

    def test_none_input(self):
        """Test with None input."""
        result = _task_age_str(None)
        assert result == "unknown"


# =============================================================================
# COMMAND FUNCTION TESTS
# =============================================================================


class TestCmdAnalyzeSummary:
    """Test cmd_analyze_summary command."""

    @patch('cortical.got.Query')
    @patch('builtins.print')
    def test_summary_with_no_tasks(self, mock_print, mock_query_class, mock_adapter, mock_manager):
        """Test summary with no tasks."""
        # Mock Query to return empty results
        mock_query = Mock()
        mock_query_class.return_value = mock_query
        mock_query.tasks.return_value = mock_query
        mock_query.sprints.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.count.return_value = mock_query
        mock_query.where.return_value = mock_query
        mock_query.or_where.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.connected_to.return_value = mock_query
        mock_query.execute.return_value = {}  # Empty status counts

        args = Namespace()
        result = cmd_analyze_summary(args, mock_adapter)

        assert result == 0
        # Verify print was called (checking header was printed)
        assert any("GoT ANALYSIS SUMMARY" in str(call) for call in mock_print.call_args_list)

    @patch('cortical.got.Query')
    @patch('builtins.print')
    def test_summary_with_tasks(self, mock_print, mock_query_class, mock_adapter, mock_manager, mock_task):
        """Test summary with tasks."""
        # Mock Query to return task data
        mock_query = Mock()
        mock_query_class.return_value = mock_query
        mock_query.tasks.return_value = mock_query
        mock_query.sprints.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.count.return_value = mock_query
        mock_query.where.return_value = mock_query
        mock_query.or_where.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.connected_to.return_value = mock_query

        # Setup different return values for different queries
        call_count = [0]
        def execute_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:  # Status counts
                return {"pending": 5, "in_progress": 2, "completed": 10}
            elif call_count[0] == 2:  # Priority counts
                return {"high": 3, "medium": 8, "low": 6}
            elif call_count[0] == 3:  # Urgent tasks
                return [mock_task]
            elif call_count[0] == 4:  # Sprints
                return []
            else:
                return []

        mock_query.execute.side_effect = execute_side_effect

        args = Namespace()
        result = cmd_analyze_summary(args, mock_adapter)

        assert result == 0
        # Verify headers were printed
        assert any("TASKS BY STATUS" in str(call) for call in mock_print.call_args_list)
        assert any("TASKS BY PRIORITY" in str(call) for call in mock_print.call_args_list)

    @patch('cortical.got.Query')
    @patch('builtins.print')
    def test_summary_with_active_sprint(self, mock_print, mock_query_class, mock_adapter, mock_manager, mock_sprint, mock_task):
        """Test summary with active sprint."""
        mock_query = Mock()
        mock_query_class.return_value = mock_query
        mock_query.tasks.return_value = mock_query
        mock_query.sprints.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.count.return_value = mock_query
        mock_query.where.return_value = mock_query
        mock_query.or_where.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.connected_to.return_value = mock_query

        # Setup different return values
        call_count = [0]
        completed_task = Mock()
        completed_task.status = "completed"

        def execute_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:  # Status counts
                return {"pending": 1, "completed": 1}
            elif call_count[0] == 2:  # Priority counts
                return {"high": 2}
            elif call_count[0] == 3:  # Urgent tasks
                return []
            elif call_count[0] == 4:  # Sprints
                return [mock_sprint]
            elif call_count[0] == 5:  # Sprint tasks
                return [mock_task, completed_task]
            else:
                return []

        mock_query.execute.side_effect = execute_side_effect

        args = Namespace()
        result = cmd_analyze_summary(args, mock_adapter)

        assert result == 0
        assert any("SPRINTS" in str(call) for call in mock_print.call_args_list)


class TestCmdAnalyzeDependencies:
    """Test cmd_analyze_dependencies command."""

    @patch('builtins.print')
    def test_dependencies_task_not_found(self, mock_print, mock_adapter, mock_manager):
        """Test dependencies command with non-existent task."""
        mock_manager.get_task.return_value = None

        args = Namespace(task_id="T-nonexistent")
        result = cmd_analyze_dependencies(args, mock_adapter)

        assert result == 1
        assert any("not found" in str(call).lower() for call in mock_print.call_args_list)

    @patch('cortical.got.GraphWalker')
    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_dependencies_no_upstream(self, mock_print, mock_path_finder_class, mock_walker_class, mock_adapter, mock_manager, mock_task):
        """Test dependencies with no upstream dependencies."""
        mock_manager.get_task.return_value = mock_task

        # Mock GraphWalker
        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.starting_from.return_value = mock_walker
        mock_walker.follow.return_value = mock_walker
        mock_walker.reverse.return_value = mock_walker
        mock_walker.directed.return_value = mock_walker
        mock_walker.bfs.return_value = mock_walker
        mock_walker.max_depth.return_value = mock_walker
        mock_walker.visit.return_value = mock_walker
        mock_walker.run.return_value = []  # No upstream dependencies

        # Mock PathFinder
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.via_edges.return_value = mock_path_finder
        mock_path_finder.shortest_path.return_value = None  # No circular dependency

        args = Namespace(task_id="T-test")
        result = cmd_analyze_dependencies(args, mock_adapter)

        assert result == 0
        assert any("no dependencies" in str(call).lower() for call in mock_print.call_args_list)

    @patch('cortical.got.GraphWalker')
    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_dependencies_with_upstream(self, mock_print, mock_path_finder_class, mock_walker_class, mock_adapter, mock_manager, mock_task):
        """Test dependencies with upstream dependencies."""
        mock_manager.get_task.side_effect = lambda tid: mock_task if tid else None

        # Mock GraphWalker to return upstream dependencies
        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.starting_from.return_value = mock_walker
        mock_walker.follow.return_value = mock_walker
        mock_walker.reverse.return_value = mock_walker
        mock_walker.directed.return_value = mock_walker
        mock_walker.bfs.return_value = mock_walker
        mock_walker.max_depth.return_value = mock_walker
        mock_walker.visit.return_value = mock_walker

        # First run returns upstream, second returns downstream
        mock_walker.run.side_effect = [["T-dep1", "T-dep2"], []]

        # Mock PathFinder
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.via_edges.return_value = mock_path_finder
        mock_path_finder.shortest_path.return_value = None

        args = Namespace(task_id="T-test")
        result = cmd_analyze_dependencies(args, mock_adapter)

        assert result == 0
        assert any("DEPENDS ON" in str(call) for call in mock_print.call_args_list)

    @patch('cortical.got.GraphWalker')
    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_dependencies_circular_detected(self, mock_print, mock_path_finder_class, mock_walker_class, mock_adapter, mock_manager, mock_task):
        """Test detection of circular dependencies."""
        mock_manager.get_task.return_value = mock_task

        # Mock GraphWalker
        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.starting_from.return_value = mock_walker
        mock_walker.follow.return_value = mock_walker
        mock_walker.reverse.return_value = mock_walker
        mock_walker.directed.return_value = mock_walker
        mock_walker.bfs.return_value = mock_walker
        mock_walker.max_depth.return_value = mock_walker
        mock_walker.visit.return_value = mock_walker
        mock_walker.run.side_effect = [[], []]  # No deps

        # Mock PathFinder to return circular path
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.via_edges.return_value = mock_path_finder
        mock_path_finder.shortest_path.return_value = ["T-test", "T-dep1", "T-test"]

        args = Namespace(task_id="T-test")
        result = cmd_analyze_dependencies(args, mock_adapter)

        assert result == 0
        assert any("CIRCULAR DEPENDENCY" in str(call) for call in mock_print.call_args_list)

    @patch('cortical.got.GraphWalker')
    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_dependencies_blocked_by_incomplete(self, mock_print, mock_path_finder_class, mock_walker_class, mock_adapter, mock_manager, mock_task):
        """Test detection of incomplete blocking dependencies."""
        incomplete_task = Mock()
        incomplete_task.id = "T-incomplete"
        incomplete_task.title = "Incomplete Task"
        incomplete_task.status = "pending"

        def get_task_side_effect(tid):
            if tid == "T-test":
                return mock_task
            elif tid == "T-incomplete":
                return incomplete_task
            return None

        mock_manager.get_task.side_effect = get_task_side_effect

        # Mock GraphWalker
        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.starting_from.return_value = mock_walker
        mock_walker.follow.return_value = mock_walker
        mock_walker.reverse.return_value = mock_walker
        mock_walker.directed.return_value = mock_walker
        mock_walker.bfs.return_value = mock_walker
        mock_walker.max_depth.return_value = mock_walker
        mock_walker.visit.return_value = mock_walker
        mock_walker.run.side_effect = [["T-incomplete"], []]

        # Mock PathFinder
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.via_edges.return_value = mock_path_finder
        mock_path_finder.shortest_path.return_value = None

        args = Namespace(task_id="T-test")
        result = cmd_analyze_dependencies(args, mock_adapter)

        assert result == 0
        assert any("Blocked by incomplete" in str(call) for call in mock_print.call_args_list)


class TestCmdAnalyzePatterns:
    """Test cmd_analyze_patterns command."""

    @patch('cortical.got.Pattern')
    @patch('cortical.got.PatternMatcher')
    @patch('builtins.print')
    def test_patterns_no_matches(self, mock_print, mock_matcher_class, mock_pattern_class, mock_adapter, mock_manager):
        """Test pattern analysis with no matches."""
        # Mock Pattern
        mock_pattern = Mock()
        mock_pattern_class.return_value = mock_pattern
        mock_pattern.node.return_value = mock_pattern
        mock_pattern.edge.return_value = mock_pattern

        # Mock PatternMatcher
        mock_matcher = Mock()
        mock_matcher_class.return_value = mock_matcher
        mock_matcher.limit.return_value = mock_matcher
        mock_matcher.find.return_value = []  # No matches

        args = Namespace()
        result = cmd_analyze_patterns(args, mock_adapter)

        assert result == 0
        assert any("no blocking chains found" in str(call).lower() for call in mock_print.call_args_list)

    @patch('cortical.got.Pattern')
    @patch('cortical.got.PatternMatcher')
    @patch('builtins.print')
    def test_patterns_with_blocking_chains(self, mock_print, mock_matcher_class, mock_pattern_class, mock_adapter, mock_manager):
        """Test pattern analysis with blocking chains found."""
        # Mock tasks
        task_a = Mock()
        task_a.title = "Task A"
        task_b = Mock()
        task_b.title = "Task B"
        task_c = Mock()
        task_c.title = "Task C"

        # Mock Pattern
        mock_pattern = Mock()
        mock_pattern_class.return_value = mock_pattern
        mock_pattern.node.return_value = mock_pattern
        mock_pattern.edge.return_value = mock_pattern

        # Mock PatternMatcher
        mock_matcher = Mock()
        mock_matcher_class.return_value = mock_matcher
        mock_matcher.limit.return_value = mock_matcher

        # Return blocking chain on first call, empty on others
        call_count = [0]
        def find_side_effect(pattern):
            call_count[0] += 1
            if call_count[0] == 1:  # Blocking chains
                return [{"a": task_a, "b": task_b, "c": task_c}]
            else:
                return []

        mock_matcher.find.side_effect = find_side_effect

        args = Namespace()
        result = cmd_analyze_patterns(args, mock_adapter)

        assert result == 0
        assert any("Task A" in str(call) for call in mock_print.call_args_list)

    @patch('cortical.got.Pattern')
    @patch('cortical.got.PatternMatcher')
    @patch('builtins.print')
    def test_patterns_with_documented_decisions(self, mock_print, mock_matcher_class, mock_pattern_class, mock_adapter, mock_manager, mock_task, mock_decision):
        """Test pattern analysis with task-decision links."""
        # Mock Pattern
        mock_pattern = Mock()
        mock_pattern_class.return_value = mock_pattern
        mock_pattern.node.return_value = mock_pattern
        mock_pattern.edge.return_value = mock_pattern

        # Mock PatternMatcher
        mock_matcher = Mock()
        mock_matcher_class.return_value = mock_matcher
        mock_matcher.limit.return_value = mock_matcher

        # Return documented decisions on third call
        call_count = [0]
        def find_side_effect(pattern):
            call_count[0] += 1
            if call_count[0] == 3:  # Documented decisions
                return [{"task": mock_task, "decision": mock_decision}]
            else:
                return []

        mock_matcher.find.side_effect = find_side_effect

        args = Namespace()
        result = cmd_analyze_patterns(args, mock_adapter)

        assert result == 0
        assert any("Test Decision" in str(call) for call in mock_print.call_args_list)


class TestCmdAnalyzeOrphans:
    """Test cmd_analyze_orphans command."""

    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_orphans_all_connected(self, mock_print, mock_path_finder_class, mock_adapter, mock_manager):
        """Test orphan analysis when all tasks are connected."""
        # Mock PathFinder
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.connected_components.return_value = [{"T-1", "T-2", "T-3"}]

        args = Namespace()
        result = cmd_analyze_orphans(args, mock_adapter)

        assert result == 0
        assert any("all tasks are connected" in str(call).lower() for call in mock_print.call_args_list)

    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_orphans_with_clusters(self, mock_print, mock_path_finder_class, mock_adapter, mock_manager, mock_task):
        """Test orphan analysis with disconnected clusters."""
        mock_manager.get_task.return_value = mock_task

        # Mock PathFinder to return multiple components
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.connected_components.return_value = [
            {"T-1", "T-2", "T-3"},  # Main component
            {"T-orphan1", "T-orphan2"},  # Orphan cluster
            {"T-orphan3"}  # Single orphan
        ]

        args = Namespace()
        result = cmd_analyze_orphans(args, mock_adapter)

        assert result == 0
        assert any("ORPHAN CLUSTERS" in str(call) for call in mock_print.call_args_list)
        assert any("Cluster 1" in str(call) for call in mock_print.call_args_list)

    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_orphans_empty_graph(self, mock_print, mock_path_finder_class, mock_adapter, mock_manager):
        """Test orphan analysis with empty graph."""
        # Mock PathFinder to return empty components
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        mock_path_finder.connected_components.return_value = []

        args = Namespace()
        result = cmd_analyze_orphans(args, mock_adapter)

        assert result == 0

    @patch('cortical.got.PathFinder')
    @patch('builtins.print')
    def test_orphans_large_cluster(self, mock_print, mock_path_finder_class, mock_adapter, mock_manager, mock_task):
        """Test orphan analysis with large cluster (>5 nodes)."""
        mock_manager.get_task.return_value = mock_task

        # Mock PathFinder
        mock_path_finder = Mock()
        mock_path_finder_class.return_value = mock_path_finder
        # Main component + large orphan cluster (6 items to trigger "and N more")
        large_cluster = {f"T-orphan-{i}" for i in range(7)}
        mock_path_finder.connected_components.return_value = [
            {"T-main"},
            large_cluster
        ]

        args = Namespace()
        result = cmd_analyze_orphans(args, mock_adapter)

        assert result == 0
        # Verify it handled the large cluster without error
        # The exact output format may vary, so just check it completed


# =============================================================================
# PARSER AND ROUTING TESTS
# =============================================================================


class TestSetupAnalyzeParser:
    """Test setup_analyze_parser function."""

    def test_parser_setup(self):
        """Test that parser is set up correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_analyze_parser(subparsers)

        # Parse a summary command
        args = parser.parse_args(['analyze', 'summary'])
        assert args.analyze_command == 'summary'

    def test_dependencies_parser(self):
        """Test dependencies subcommand parser."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_analyze_parser(subparsers)

        # Parse dependencies command
        args = parser.parse_args(['analyze', 'dependencies', 'T-test'])
        assert args.analyze_command == 'dependencies'
        assert args.task_id == 'T-test'

    def test_patterns_parser(self):
        """Test patterns subcommand parser."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_analyze_parser(subparsers)

        args = parser.parse_args(['analyze', 'patterns'])
        assert args.analyze_command == 'patterns'

    def test_orphans_parser(self):
        """Test orphans subcommand parser."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_analyze_parser(subparsers)

        args = parser.parse_args(['analyze', 'orphans'])
        assert args.analyze_command == 'orphans'


class TestHandleAnalyzeCommand:
    """Test handle_analyze_command routing function."""

    @patch('cortical.got.cli.analyze.cmd_analyze_summary')
    def test_route_no_subcommand_defaults_to_summary(self, mock_summary, mock_adapter):
        """Test that no subcommand defaults to summary."""
        mock_summary.return_value = 0
        args = Namespace()

        result = handle_analyze_command(args, mock_adapter)

        assert result == 0
        mock_summary.assert_called_once_with(args, mock_adapter)

    @patch('cortical.got.cli.analyze.cmd_analyze_summary')
    def test_route_summary(self, mock_summary, mock_adapter):
        """Test routing to summary command."""
        mock_summary.return_value = 0
        args = Namespace(analyze_command='summary')

        result = handle_analyze_command(args, mock_adapter)

        assert result == 0
        mock_summary.assert_called_once_with(args, mock_adapter)

    @patch('cortical.got.cli.analyze.cmd_analyze_dependencies')
    def test_route_dependencies(self, mock_deps, mock_adapter):
        """Test routing to dependencies command."""
        mock_deps.return_value = 0
        args = Namespace(analyze_command='dependencies', task_id='T-test')

        result = handle_analyze_command(args, mock_adapter)

        assert result == 0
        mock_deps.assert_called_once_with(args, mock_adapter)

    @patch('cortical.got.cli.analyze.cmd_analyze_patterns')
    def test_route_patterns(self, mock_patterns, mock_adapter):
        """Test routing to patterns command."""
        mock_patterns.return_value = 0
        args = Namespace(analyze_command='patterns')

        result = handle_analyze_command(args, mock_adapter)

        assert result == 0
        mock_patterns.assert_called_once_with(args, mock_adapter)

    @patch('cortical.got.cli.analyze.cmd_analyze_orphans')
    def test_route_orphans(self, mock_orphans, mock_adapter):
        """Test routing to orphans command."""
        mock_orphans.return_value = 0
        args = Namespace(analyze_command='orphans')

        result = handle_analyze_command(args, mock_adapter)

        assert result == 0
        mock_orphans.assert_called_once_with(args, mock_adapter)

    def test_route_unknown_command(self, mock_adapter):
        """Test routing unknown command returns error."""
        args = Namespace(analyze_command='unknown')

        result = handle_analyze_command(args, mock_adapter)

        assert result == 1
