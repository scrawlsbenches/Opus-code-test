"""
Unit tests for cortical/got/cli/shared.py.

Tests all formatting utilities and helper functions.
"""

import pytest
from unittest.mock import Mock
from cortical.got.cli.shared import (
    format_task_table,
    format_sprint_status,
    format_task_details,
    truncate,
    VALID_STATUSES,
    VALID_PRIORITIES,
    VALID_CATEGORIES,
    PRIORITY_SCORES,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_task():
    """Create a mock task with standard properties."""
    task = Mock()
    task.id = "task:T-20251223-120000-abc123"
    task.content = "Implement feature X"
    task.properties = {
        "status": "in_progress",
        "priority": "high",
        "category": "feature",
    }
    task.metadata = {}
    return task


@pytest.fixture
def mock_sprint():
    """Create a mock sprint with standard properties."""
    sprint = Mock()
    sprint.id = "sprint:S-sprint-001-test"
    sprint.content = "Sprint 001: Test Sprint"
    sprint.properties = {
        "status": "in_progress",
    }
    return sprint


# =============================================================================
# TEST: format_task_table
# =============================================================================

class TestFormatTaskTable:
    """Tests for format_task_table()."""

    def test_empty_list(self):
        """Empty task list returns 'No tasks found' message."""
        result = format_task_table([])
        assert result == "No tasks found."

    def test_single_task(self, mock_task):
        """Single task renders correctly in table."""
        result = format_task_table([mock_task])

        # Check table structure
        assert "┌" in result
        assert "┐" in result
        assert "└" in result
        assert "┘" in result

        # Check content
        assert "T-20251223-120000-abc123" in result
        assert "Implement feature X" in result
        # Status column is 10 chars, so "in_progress" gets truncated to "in_progres"
        assert "in_progres" in result
        assert "high" in result

    def test_multiple_tasks(self, mock_task):
        """Multiple tasks render correctly in table."""
        task2 = Mock()
        task2.id = "task:T-20251223-120001-def456"
        task2.content = "Fix bug Y"
        task2.properties = {"status": "pending", "priority": "critical"}

        result = format_task_table([mock_task, task2])

        # Check both tasks present
        assert "T-20251223-120000-abc123" in result
        assert "T-20251223-120001-def456" in result
        assert "Implement feature X" in result
        assert "Fix bug Y" in result
        # Status column is 10 chars, so "in_progress" gets truncated to "in_progres"
        assert "in_progres" in result
        assert "pending" in result
        assert "high" in result
        assert "critical" in result

    def test_long_id_truncation(self):
        """Task ID longer than 26 chars is truncated."""
        task = Mock()
        task.id = "task:" + "x" * 50  # Very long ID
        task.content = "Test"
        task.properties = {"status": "pending", "priority": "low"}

        result = format_task_table([task])

        # ID should be truncated to 26 chars
        lines = result.split("\n")
        # Find the data row (after header and separator)
        data_row = [line for line in lines if line.startswith("│") and "Test" in line][0]
        # Extract ID field (first column)
        id_field = data_row.split("│")[1].strip()
        assert len(id_field) <= 26

    def test_long_title_truncation(self):
        """Task title longer than 33 chars is truncated."""
        task = Mock()
        task.id = "task:T-123"
        task.content = "This is a very long task title that exceeds the maximum allowed width"
        task.properties = {"status": "pending", "priority": "low"}

        result = format_task_table([task])

        # Title should be truncated to 33 chars
        lines = result.split("\n")
        data_row = [line for line in lines if line.startswith("│") and "This is" in line][0]
        # Extract title field (second column)
        title_field = data_row.split("│")[2].strip()
        assert len(title_field) <= 33

    def test_missing_status_property(self):
        """Task without status shows '?' in table."""
        task = Mock()
        task.id = "task:T-123"
        task.content = "Test"
        task.properties = {"priority": "low"}  # No status

        result = format_task_table([task])

        # Should show '?' for missing status
        assert "?" in result

    def test_missing_priority_property(self):
        """Task without priority shows '?' in table."""
        task = Mock()
        task.id = "task:T-123"
        task.content = "Test"
        task.properties = {"status": "pending"}  # No priority

        result = format_task_table([task])

        # Should show '?' for missing priority
        assert "?" in result

    def test_empty_properties(self):
        """Task with empty properties dict shows '?' for all fields."""
        task = Mock()
        task.id = "task:T-123"
        task.content = "Test"
        task.properties = {}

        result = format_task_table([task])

        # Should show '?' for both status and priority
        lines = result.split("\n")
        data_row = [line for line in lines if line.startswith("│") and "Test" in line][0]
        assert data_row.count("?") >= 2

    def test_table_header_format(self, mock_task):
        """Table has correct header with column names."""
        result = format_task_table([mock_task])

        lines = result.split("\n")
        # Header should be second line
        header = lines[1]

        assert "ID" in header
        assert "Title" in header
        assert "Status" in header
        assert "Priority" in header


# =============================================================================
# TEST: format_sprint_status
# =============================================================================

class TestFormatSprintStatus:
    """Tests for format_sprint_status()."""

    def test_basic_sprint_status(self, mock_sprint):
        """Basic sprint status formats correctly."""
        progress = {
            "completed": 5,
            "total_tasks": 10,
            "progress_percent": 50.0,
            "by_status": {
                "pending": 3,
                "in_progress": 2,
                "completed": 5,
            }
        }

        result = format_sprint_status(mock_sprint, progress)

        # Check sprint info
        assert "Sprint 001: Test Sprint" in result
        assert "sprint:S-sprint-001-test" in result
        assert "Status: in_progress" in result

        # Check progress
        assert "5/10 tasks" in result
        assert "50.0%" in result

        # Check status breakdown
        assert "pending: 3" in result
        assert "in_progress: 2" in result
        assert "completed: 5" in result

    def test_sprint_with_claimed_by(self, mock_sprint):
        """Sprint claimed by agent shows claim info."""
        mock_sprint.properties["claimed_by"] = "agent-001"
        mock_sprint.properties["claimed_at"] = "2025-12-23T12:00:00"

        progress = {
            "completed": 0,
            "total_tasks": 5,
            "progress_percent": 0.0,
            "by_status": {"pending": 5}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "Claimed by: agent-001" in result
        assert "Claimed at: 2025-12-23T12:00:00" in result

    def test_sprint_without_claimed_by(self, mock_sprint):
        """Sprint without claim info doesn't show claim fields."""
        progress = {
            "completed": 0,
            "total_tasks": 5,
            "progress_percent": 0.0,
            "by_status": {"pending": 5}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "Claimed by:" not in result
        assert "Claimed at:" not in result

    def test_sprint_claimed_by_without_timestamp(self, mock_sprint):
        """Sprint claimed without timestamp shows only claimed_by."""
        mock_sprint.properties["claimed_by"] = "agent-002"
        # No claimed_at

        progress = {
            "completed": 0,
            "total_tasks": 5,
            "progress_percent": 0.0,
            "by_status": {"pending": 5}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "Claimed by: agent-002" in result
        assert "Claimed at:" not in result

    def test_zero_progress(self, mock_sprint):
        """Sprint with zero progress shows 0.0%."""
        progress = {
            "completed": 0,
            "total_tasks": 10,
            "progress_percent": 0.0,
            "by_status": {"pending": 10}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "0/10 tasks" in result
        assert "0.0%" in result

    def test_full_progress(self, mock_sprint):
        """Sprint with 100% progress shows all completed."""
        progress = {
            "completed": 10,
            "total_tasks": 10,
            "progress_percent": 100.0,
            "by_status": {"completed": 10}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "10/10 tasks" in result
        assert "100.0%" in result

    def test_empty_by_status(self, mock_sprint):
        """Sprint with no status breakdown still renders."""
        progress = {
            "completed": 0,
            "total_tasks": 0,
            "progress_percent": 0.0,
            "by_status": {}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "0/0 tasks" in result
        assert "By Status:" in result

    def test_missing_status_property(self, mock_sprint):
        """Sprint without status shows 'unknown'."""
        mock_sprint.properties = {}  # No status

        progress = {
            "completed": 0,
            "total_tasks": 5,
            "progress_percent": 0.0,
            "by_status": {}
        }

        result = format_sprint_status(mock_sprint, progress)

        assert "Status: unknown" in result


# =============================================================================
# TEST: format_task_details
# =============================================================================

class TestFormatTaskDetails:
    """Tests for format_task_details()."""

    def test_minimal_task_details(self, mock_task):
        """Minimal task with only required fields formats correctly."""
        result = format_task_details(mock_task)

        # Check structure
        assert "=" * 60 in result
        assert "TASK: task:T-20251223-120000-abc123" in result

        # Check basic fields
        assert "Title:    Implement feature X" in result
        assert "Status:   in_progress" in result
        assert "Priority: high" in result
        assert "Category: feature" in result

    def test_task_with_description(self, mock_task):
        """Task with description shows description section."""
        mock_task.properties["description"] = "This is a detailed description"

        result = format_task_details(mock_task)

        assert "Description:" in result
        assert "This is a detailed description" in result

    def test_task_with_retrospective(self, mock_task):
        """Task with retrospective shows retrospective section."""
        mock_task.properties["retrospective"] = "Learned about X and Y"

        result = format_task_details(mock_task)

        assert "Retrospective:" in result
        assert "Learned about X and Y" in result

    def test_task_with_blocked_reason(self, mock_task):
        """Task with blocked_reason shows blocked section."""
        mock_task.properties["blocked_reason"] = "Waiting for API access"

        result = format_task_details(mock_task)

        assert "Blocked Reason:" in result
        assert "Waiting for API access" in result

    def test_task_with_all_optional_fields(self, mock_task):
        """Task with all optional fields shows all sections."""
        mock_task.properties.update({
            "description": "Full description",
            "retrospective": "Full retrospective",
            "blocked_reason": "Full blocked reason"
        })

        result = format_task_details(mock_task)

        assert "Description:" in result
        assert "Full description" in result
        assert "Retrospective:" in result
        assert "Full retrospective" in result
        assert "Blocked Reason:" in result
        assert "Full blocked reason" in result

    def test_task_with_created_timestamp(self, mock_task):
        """Task with created_at shows creation time."""
        mock_task.metadata["created_at"] = "2025-12-23T10:00:00"

        result = format_task_details(mock_task)

        assert "Timestamps:" in result
        assert "Created:   2025-12-23T10:00:00" in result

    def test_task_with_updated_timestamp(self, mock_task):
        """Task with updated_at shows update time."""
        mock_task.metadata["updated_at"] = "2025-12-23T11:00:00"

        result = format_task_details(mock_task)

        assert "Updated:   2025-12-23T11:00:00" in result

    def test_task_with_completed_timestamp(self, mock_task):
        """Task with completed_at shows completion time."""
        mock_task.metadata["completed_at"] = "2025-12-23T12:00:00"

        result = format_task_details(mock_task)

        assert "Completed: 2025-12-23T12:00:00" in result

    def test_task_with_all_timestamps(self, mock_task):
        """Task with all timestamps shows all times."""
        mock_task.metadata.update({
            "created_at": "2025-12-23T10:00:00",
            "updated_at": "2025-12-23T11:00:00",
            "completed_at": "2025-12-23T12:00:00"
        })

        result = format_task_details(mock_task)

        assert "Created:   2025-12-23T10:00:00" in result
        assert "Updated:   2025-12-23T11:00:00" in result
        assert "Completed: 2025-12-23T12:00:00" in result

    def test_task_without_timestamps(self, mock_task):
        """Task without timestamps still shows Timestamps section."""
        mock_task.metadata = {}

        result = format_task_details(mock_task)

        # Should have Timestamps section header but no actual timestamps
        assert "Timestamps:" in result
        assert "Created:" not in result
        assert "Updated:" not in result
        assert "Completed:" not in result

    def test_missing_category_shows_unknown(self, mock_task):
        """Task without category shows 'unknown'."""
        del mock_task.properties["category"]

        result = format_task_details(mock_task)

        assert "Category: unknown" in result

    def test_missing_status_shows_unknown(self, mock_task):
        """Task without status shows 'unknown'."""
        del mock_task.properties["status"]

        result = format_task_details(mock_task)

        assert "Status:   unknown" in result

    def test_missing_priority_shows_unknown(self, mock_task):
        """Task without priority shows 'unknown'."""
        del mock_task.properties["priority"]

        result = format_task_details(mock_task)

        assert "Priority: unknown" in result


# =============================================================================
# TEST: truncate
# =============================================================================

class TestTruncate:
    """Tests for truncate() utility."""

    def test_text_shorter_than_max_returns_unchanged(self):
        """Text shorter than max_length returns unchanged."""
        text = "Hello"
        result = truncate(text, 10)
        assert result == "Hello"

    def test_text_equal_to_max_returns_unchanged(self):
        """Text exactly at max_length returns unchanged."""
        text = "Hello"
        result = truncate(text, 5)
        assert result == "Hello"

    def test_text_longer_than_max_truncates_with_suffix(self):
        """Text longer than max_length truncates and adds suffix."""
        text = "Hello World"
        result = truncate(text, 8)
        # 8 total chars: 5 chars + 3 char suffix "..."
        assert result == "Hello..."
        assert len(result) == 8

    def test_custom_suffix(self):
        """Custom suffix is used when provided."""
        text = "Hello World"
        result = truncate(text, 8, suffix="!!")
        # 8 total chars: 6 chars + 2 char suffix "!!"
        # text[:6] = "Hello " (includes space)
        assert result == "Hello !!"
        assert len(result) == 8

    def test_empty_string(self):
        """Empty string returns empty string."""
        result = truncate("", 10)
        assert result == ""

    def test_empty_suffix(self):
        """Empty suffix works correctly."""
        text = "Hello World"
        result = truncate(text, 5, suffix="")
        assert result == "Hello"
        assert len(result) == 5

    def test_max_length_zero(self):
        """Max length of 0 with default suffix results in negative slice."""
        text = "Hello"
        result = truncate(text, 0)
        # text[:0-3] = text[:-3] = "He"
        assert result == "He..."

    def test_max_length_one(self):
        """Max length of 1 handles suffix correctly."""
        text = "Hello"
        result = truncate(text, 1, suffix=".")
        assert result == "."

    def test_very_long_text(self):
        """Very long text truncates correctly."""
        text = "x" * 1000
        result = truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_unicode_characters(self):
        """Unicode characters truncate correctly."""
        text = "Hello 世界 World"
        result = truncate(text, 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_suffix_longer_than_max_length(self):
        """Suffix longer than max_length handles gracefully."""
        text = "Hello"
        result = truncate(text, 5, suffix="........")
        # Should return suffix only (length 5)
        assert len(result) == 5


# =============================================================================
# TEST: CONSTANTS
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_valid_statuses_defined(self):
        """VALID_STATUSES contains expected status values."""
        assert "pending" in VALID_STATUSES
        assert "in_progress" in VALID_STATUSES
        assert "completed" in VALID_STATUSES
        assert "blocked" in VALID_STATUSES
        assert "deferred" in VALID_STATUSES

    def test_valid_priorities_defined(self):
        """VALID_PRIORITIES contains expected priority values."""
        assert "critical" in VALID_PRIORITIES
        assert "high" in VALID_PRIORITIES
        assert "medium" in VALID_PRIORITIES
        assert "low" in VALID_PRIORITIES

    def test_valid_categories_defined(self):
        """VALID_CATEGORIES contains expected category values."""
        assert "arch" in VALID_CATEGORIES
        assert "feature" in VALID_CATEGORIES
        assert "bugfix" in VALID_CATEGORIES
        assert "test" in VALID_CATEGORIES
        assert "docs" in VALID_CATEGORIES

    def test_priority_scores_mapping(self):
        """PRIORITY_SCORES maps priorities to numeric scores."""
        assert PRIORITY_SCORES["critical"] == 100
        assert PRIORITY_SCORES["high"] == 75
        assert PRIORITY_SCORES["medium"] == 50
        assert PRIORITY_SCORES["low"] == 25

    def test_priority_scores_descending(self):
        """PRIORITY_SCORES values are in descending order."""
        scores = list(PRIORITY_SCORES.values())
        assert scores == sorted(scores, reverse=True)
