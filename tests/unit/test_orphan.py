"""
Unit tests for orphan detection and auto-linking system.

Tests the OrphanDetector class and related functionality for:
- Detecting orphan tasks (tasks with no connections)
- Generating orphan reports
- Suggesting sprint assignments
- Suggesting task connections
- Auto-linking orphans to sprints
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from cortical.got.orphan import (
    OrphanDetector,
    OrphanReport,
    ConnectionSuggestion,
    SprintSuggestion,
    check_orphan_on_create,
    generate_orphan_report,
)
from cortical.got.types import Task, Sprint, Edge


class TestOrphanReport:
    """Tests for OrphanReport dataclass."""

    def test_empty_report(self):
        """Test empty orphan report."""
        report = OrphanReport()
        assert report.orphan_count == 0
        assert not report.has_orphans
        assert report.orphan_rate == 0.0

    def test_report_with_orphans(self):
        """Test report with orphan tasks."""
        report = OrphanReport(
            orphan_tasks=["T-001", "T-002", "T-003"],
            total_tasks=10,
            orphan_rate=30.0,
        )
        assert report.orphan_count == 3
        assert report.has_orphans
        assert report.orphan_rate == 30.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = OrphanReport(
            orphan_tasks=["T-001"],
            total_tasks=5,
            orphan_rate=20.0,
        )
        d = report.to_dict()
        assert d['orphan_tasks'] == ["T-001"]
        assert d['total_tasks'] == 5
        assert d['orphan_count'] == 1
        assert 'generated_at' in d


class TestConnectionSuggestion:
    """Tests for ConnectionSuggestion dataclass."""

    def test_connection_suggestion(self):
        """Test connection suggestion creation."""
        suggestion = ConnectionSuggestion(
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            confidence=0.8,
            reason="Similar content",
        )
        assert suggestion.source_id == "T-001"
        assert suggestion.edge_type == "DEPENDS_ON"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suggestion = ConnectionSuggestion(
            source_id="T-001",
            target_id="T-002",
            edge_type="RELATED_TO",
            confidence=0.5,
            reason="Test",
        )
        d = suggestion.to_dict()
        assert d['source_id'] == "T-001"
        assert d['confidence'] == 0.5


class TestSprintSuggestion:
    """Tests for SprintSuggestion dataclass."""

    def test_sprint_suggestion(self):
        """Test sprint suggestion creation."""
        suggestion = SprintSuggestion(
            sprint_id="S-001",
            sprint_title="Sprint 1",
            confidence=0.9,
            reason="Current sprint",
            is_current=True,
        )
        assert suggestion.is_current
        assert suggestion.confidence == 0.9

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suggestion = SprintSuggestion(
            sprint_id="S-001",
            sprint_title="Test Sprint",
            confidence=0.7,
            reason="Similar tasks",
            is_current=False,
        )
        d = suggestion.to_dict()
        assert d['sprint_id'] == "S-001"
        assert not d['is_current']


class TestOrphanDetector:
    """Tests for OrphanDetector class."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock GoTManager."""
        manager = Mock()

        # Create mock tasks
        task1 = Mock(spec=Task)
        task1.id = "T-001"
        task1.title = "Implement authentication"
        task1.description = "Add OAuth2 authentication"
        task1.status = "pending"
        task1.priority = "high"

        task2 = Mock(spec=Task)
        task2.id = "T-002"
        task2.title = "Add login page"
        task2.description = "Create login UI"
        task2.status = "pending"
        task2.priority = "medium"

        task3 = Mock(spec=Task)
        task3.id = "T-003"
        task3.title = "Write documentation"
        task3.description = "Document the API"
        task3.status = "completed"
        task3.priority = "low"

        manager.list_all_tasks.return_value = [task1, task2, task3]
        manager.get_task.side_effect = lambda tid: {
            "T-001": task1,
            "T-002": task2,
            "T-003": task3,
        }.get(tid)

        # Mock sprint
        sprint = Mock(spec=Sprint)
        sprint.id = "S-018"
        sprint.title = "Sprint 18"
        sprint.status = "in_progress"
        manager.get_current_sprint.return_value = sprint
        manager.list_sprints.return_value = [sprint]
        manager.get_sprint_tasks.return_value = [task2]  # task2 is in sprint

        return manager

    def test_is_orphan_no_edges(self, mock_manager):
        """Test detecting orphan task with no edges."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        assert detector.is_orphan("T-001") is True

    def test_is_orphan_with_edges(self, mock_manager):
        """Test detecting connected task."""
        edge = Mock(spec=Edge)
        edge.source_id = "S-018"
        edge.target_id = "T-001"
        edge.edge_type = "CONTAINS"
        mock_manager.get_edges_for_task.return_value = ([], [edge])

        detector = OrphanDetector(mock_manager)
        assert detector.is_orphan("T-001") is False

    def test_find_orphan_tasks(self, mock_manager):
        """Test finding all orphan tasks."""
        # T-001 and T-003 are orphans, T-002 is connected
        def get_edges(task_id):
            if task_id == "T-002":
                edge = Mock(spec=Edge)
                edge.source_id = "S-018"
                edge.target_id = "T-002"
                edge.edge_type = "CONTAINS"
                return ([], [edge])
            return ([], [])

        mock_manager.get_edges_for_task.side_effect = get_edges

        detector = OrphanDetector(mock_manager)
        orphans = detector.find_orphan_tasks()

        assert "T-001" in orphans
        assert "T-002" not in orphans
        assert "T-003" in orphans

    def test_generate_orphan_report(self, mock_manager):
        """Test generating orphan report."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        report = detector.generate_orphan_report()

        assert report.total_tasks == 3
        assert len(report.orphan_tasks) == 3  # All are orphans
        assert report.orphan_rate == 100.0
        assert report.has_orphans

    def test_suggest_sprint_current(self, mock_manager):
        """Test suggesting current sprint for task."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        suggestions = detector.suggest_sprint("T-001")

        assert len(suggestions) >= 1
        # Current sprint should be first with high confidence
        current_suggestion = next((s for s in suggestions if s.is_current), None)
        assert current_suggestion is not None
        assert current_suggestion.sprint_id == "S-018"
        assert current_suggestion.confidence >= 0.8

    def test_suggest_sprint_no_current(self, mock_manager):
        """Test sprint suggestions when no current sprint."""
        mock_manager.get_current_sprint.return_value = None
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        suggestions = detector.suggest_sprint("T-001")

        # Should still get suggestions from in_progress sprints
        assert isinstance(suggestions, list)

    def test_suggest_connections_similar_tasks(self, mock_manager):
        """Test suggesting connections based on similarity."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        suggestions = detector.suggest_connections("T-001")

        # Should suggest T-002 (login page) as related to T-001 (authentication)
        assert isinstance(suggestions, list)
        # Check that connections are for different tasks
        for s in suggestions:
            assert s.source_id == "T-001"
            assert s.target_id != "T-001"

    def test_auto_link_to_sprint(self, mock_manager):
        """Test auto-linking orphan tasks to sprint."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        linked = detector.auto_link_to_sprint(["T-001", "T-003"])

        assert len(linked) == 2
        assert all(sprint_id == "S-018" for _, sprint_id in linked)
        # Verify add_task_to_sprint was called
        assert mock_manager.add_task_to_sprint.call_count == 2

    def test_auto_link_skips_already_linked(self, mock_manager):
        """Test that auto-link skips already linked tasks."""
        # T-002 is already in a sprint
        def get_edges(task_id):
            if task_id == "T-002":
                edge = Mock(spec=Edge)
                edge.source_id = "S-018"
                edge.target_id = "T-002"
                edge.edge_type = "CONTAINS"
                return ([], [edge])
            return ([], [])

        mock_manager.get_edges_for_task.side_effect = get_edges

        detector = OrphanDetector(mock_manager)
        linked = detector.auto_link_to_sprint(["T-001", "T-002"])

        # Only T-001 should be linked
        assert len(linked) == 1
        assert linked[0][0] == "T-001"

    def test_check_on_create(self, mock_manager):
        """Test checking task on creation."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        result = detector.check_on_create("T-001")

        assert result['task_id'] == "T-001"
        assert result['is_orphan'] is True
        assert len(result['warnings']) > 0
        assert 'sprint_suggestions' in result
        assert 'connection_suggestions' in result

    def test_get_orphan_summary(self, mock_manager):
        """Test getting human-readable orphan summary."""
        mock_manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(mock_manager)
        summary = detector.get_orphan_summary()

        assert "ORPHAN DETECTION REPORT" in summary
        assert "Total Tasks: 3" in summary
        assert "Orphan Tasks: 3" in summary

    def test_keyword_extraction(self, mock_manager):
        """Test keyword extraction from text."""
        detector = OrphanDetector(mock_manager)

        keywords = detector._extract_keywords("Add user authentication with tokens")
        assert "user" in keywords
        assert "authentication" in keywords
        assert "tokens" in keywords
        # Stop words should be removed
        assert "with" not in keywords
        assert "add" in keywords  # Short but not a stop word

    def test_keyword_similarity(self, mock_manager):
        """Test keyword similarity calculation."""
        detector = OrphanDetector(mock_manager)

        kw1 = {"auth", "user", "login"}
        kw2 = {"auth", "token", "login"}
        kw3 = {"database", "migration", "schema"}

        # kw1 and kw2 should have high similarity
        sim12 = detector._keyword_similarity(kw1, kw2)
        assert sim12 > 0.3

        # kw1 and kw3 should have low similarity
        sim13 = detector._keyword_similarity(kw1, kw3)
        assert sim13 == 0.0

    def test_infer_edge_type_depends(self, mock_manager):
        """Test inferring DEPENDS_ON edge type."""
        detector = OrphanDetector(mock_manager)

        source = Mock()
        source.title = "Implement login requires authentication"
        source.description = ""

        target = Mock()
        target.title = "Add OAuth2"
        target.description = ""

        edge_type, reason = detector._infer_edge_type(source, target)
        assert edge_type == "DEPENDS_ON"

    def test_infer_edge_type_blocks(self, mock_manager):
        """Test inferring BLOCKS edge type."""
        detector = OrphanDetector(mock_manager)

        source = Mock()
        source.title = "This blocks the deployment"
        source.description = ""

        target = Mock()
        target.title = "Deploy to production"
        target.description = ""

        edge_type, reason = detector._infer_edge_type(source, target)
        assert edge_type == "BLOCKS"

    def test_infer_edge_type_related(self, mock_manager):
        """Test inferring RELATED_TO edge type as default."""
        detector = OrphanDetector(mock_manager)

        source = Mock()
        source.title = "Add unit tests"
        source.description = ""

        target = Mock()
        target.title = "Improve code coverage"
        target.description = ""

        edge_type, reason = detector._infer_edge_type(source, target)
        assert edge_type == "RELATED_TO"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_check_orphan_on_create(self):
        """Test check_orphan_on_create convenience function."""
        manager = Mock()
        task = Mock()
        task.id = "T-001"
        task.title = "Test task"
        task.description = ""
        task.status = "pending"
        task.priority = "medium"

        manager.get_task.return_value = task
        manager.list_all_tasks.return_value = [task]
        manager.get_current_sprint.return_value = None
        manager.list_sprints.return_value = []
        manager.get_edges_for_task.return_value = ([], [])

        result = check_orphan_on_create(manager, "T-001")

        assert result['task_id'] == "T-001"
        assert 'warnings' in result

    def test_generate_orphan_report_convenience(self):
        """Test generate_orphan_report convenience function."""
        manager = Mock()
        manager.list_all_tasks.return_value = []

        report = generate_orphan_report(manager)

        assert isinstance(report, OrphanReport)
        assert report.total_tasks == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_task_list(self):
        """Test handling empty task list."""
        manager = Mock()
        manager.list_all_tasks.return_value = []

        detector = OrphanDetector(manager)
        orphans = detector.find_orphan_tasks()

        assert orphans == []

    def test_missing_task(self):
        """Test handling missing task."""
        manager = Mock()
        manager.get_task.return_value = None
        manager.list_all_tasks.return_value = []
        manager.get_edges_for_task.return_value = ([], [])
        manager.get_current_sprint.return_value = None
        manager.list_sprints.return_value = []

        detector = OrphanDetector(manager)
        suggestions = detector.suggest_sprint("T-nonexistent")

        assert suggestions == []

    def test_auto_link_no_current_sprint(self):
        """Test auto-link when no current sprint exists."""
        manager = Mock()
        manager.get_current_sprint.return_value = None
        manager.get_edges_for_task.return_value = ([], [])

        detector = OrphanDetector(manager)
        linked = detector.auto_link_to_sprint(["T-001"])

        assert linked == []

    def test_auto_link_with_error(self):
        """Test auto-link handles errors gracefully."""
        manager = Mock()
        sprint = Mock()
        sprint.id = "S-001"
        manager.get_current_sprint.return_value = sprint
        manager.get_edges_for_task.return_value = ([], [])
        manager.add_task_to_sprint.side_effect = Exception("Database error")

        detector = OrphanDetector(manager)
        linked = detector.auto_link_to_sprint(["T-001"])

        # Should handle error and return empty list
        assert linked == []
