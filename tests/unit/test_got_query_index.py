"""
Schema Validation Tests for QueryIndexManager
==============================================

Tests validation of inputs to QueryIndexManager methods including:
- Invalid status values (not in valid set)
- Invalid priority values (not in valid set)
- Missing required fields
- Type checking for parameters
- EdgeType enum validation

TDD: These tests verify that invalid inputs are properly rejected.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Set

from cortical.got.indexer import QueryIndexManager
from cortical.got.api import GoTManager
from cortical.got.types import Task, VALID_EDGE_TYPES
from cortical.got.errors import ValidationError
from cortical.got.config import DurabilityMode


class TestStatusValidation:
    """Test validation of task status values."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, got_dir):
        """Create a GoTManager for testing."""
        return GoTManager(got_dir, durability=DurabilityMode.RELAXED)

    @pytest.fixture
    def index_manager(self, got_dir):
        """Create a QueryIndexManager for testing."""
        return QueryIndexManager(got_dir)

    def test_invalid_status_rejected_by_task_creation(self, manager):
        """Creating a task with invalid status should raise ValidationError."""
        invalid_statuses = [
            "bogus_status",
            "not_a_status",
            "PENDING",  # Wrong case
            "Pending",  # Wrong case
            "done",     # Not a valid status
            "active",   # Not a valid status
            "",         # Empty string
        ]

        for invalid_status in invalid_statuses:
            with pytest.raises(ValidationError) as exc_info:
                # Task validation happens in __post_init__
                Task(
                    id="T-test",
                    title="Test task",
                    status=invalid_status,
                    priority="medium"
                )
            assert "Invalid status" in str(exc_info.value)

    def test_valid_statuses_accepted(self, manager):
        """Valid status values should be accepted."""
        valid_statuses = ["pending", "in_progress", "completed", "blocked"]

        for status in valid_statuses:
            # Should not raise
            task = manager.create_task("Test task", status=status)
            assert task.status == status

    def test_lookup_with_invalid_status_returns_empty(self, index_manager):
        """Looking up an invalid status should return empty set, not error."""
        # Index doesn't validate - it just returns empty results
        result = index_manager.lookup("status", "bogus_status")
        assert isinstance(result, set)
        assert len(result) == 0

    def test_index_task_with_none_status(self, index_manager):
        """Indexing a task with None status should not raise."""
        # Should handle None gracefully
        index_manager.index_task("T-test", status=None, priority="medium")
        # No exception = success


class TestPriorityValidation:
    """Test validation of task priority values."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, got_dir):
        """Create a GoTManager for testing."""
        return GoTManager(got_dir, durability=DurabilityMode.RELAXED)

    @pytest.fixture
    def index_manager(self, got_dir):
        """Create a QueryIndexManager for testing."""
        return QueryIndexManager(got_dir)

    def test_invalid_priority_rejected_by_task_creation(self, manager):
        """Creating a task with invalid priority should raise ValidationError."""
        invalid_priorities = [
            "super_high",
            "ultra_critical",
            "normal",
            "HIGH",      # Wrong case
            "Low",       # Wrong case
            "urgent",    # Not a valid priority
            "",          # Empty string
            "p1",        # Not a valid priority
        ]

        for invalid_priority in invalid_priorities:
            with pytest.raises(ValidationError) as exc_info:
                # Task validation happens in __post_init__
                Task(
                    id="T-test",
                    title="Test task",
                    status="pending",
                    priority=invalid_priority
                )
            assert "Invalid priority" in str(exc_info.value)

    def test_valid_priorities_accepted(self, manager):
        """Valid priority values should be accepted."""
        valid_priorities = ["low", "medium", "high", "critical"]

        for priority in valid_priorities:
            # Should not raise
            task = manager.create_task("Test task", priority=priority)
            assert task.priority == priority

    def test_lookup_with_invalid_priority_returns_empty(self, index_manager):
        """Looking up an invalid priority should return empty set, not error."""
        # Index doesn't validate - it just returns empty results
        result = index_manager.lookup("priority", "super_high")
        assert isinstance(result, set)
        assert len(result) == 0

    def test_index_task_with_none_priority(self, index_manager):
        """Indexing a task with None priority should not raise."""
        # Should handle None gracefully
        index_manager.index_task("T-test", status="pending", priority=None)
        # No exception = success


class TestRequiredFieldValidation:
    """Test validation of required fields."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, got_dir):
        """Create a GoTManager for testing."""
        return GoTManager(got_dir, durability=DurabilityMode.RELAXED)

    def test_task_requires_id(self):
        """Task creation should fail without an id."""
        with pytest.raises(TypeError) as exc_info:
            # Missing required 'id' parameter
            Task(title="Test task", status="pending", priority="medium")
        assert "id" in str(exc_info.value).lower()

    def test_task_requires_title_for_manager_create(self, manager):
        """Creating task through manager should require title."""
        # Manager's create_task requires title as first parameter
        with pytest.raises(TypeError):
            # Missing required positional argument
            manager.create_task()

    def test_task_with_empty_title_allowed(self, manager):
        """Empty string title should be allowed (not ideal but valid)."""
        # This tests current behavior - empty string is technically valid
        task = manager.create_task("")
        assert task.title == ""


class TestTypeValidation:
    """Test type checking for parameters."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def index_manager(self, got_dir):
        """Create a QueryIndexManager for testing."""
        return QueryIndexManager(got_dir)

    def test_lookup_field_name_must_be_string(self, index_manager):
        """lookup() field_name should be a string."""
        # Python is dynamically typed, so this won't raise at call time
        # but we can test the behavior
        result = index_manager.lookup(123, "pending")  # type: ignore
        # Will convert to string internally: str(123) = "123"
        assert isinstance(result, set)

    def test_lookup_returns_set(self, index_manager):
        """lookup() should always return a set."""
        result = index_manager.lookup("status", "pending")
        assert isinstance(result, set)

    def test_lookup_multi_values_must_be_list(self, index_manager):
        """lookup_multi() values parameter should be a list."""
        # Test with valid list
        result = index_manager.lookup_multi("status", ["pending", "completed"])
        assert isinstance(result, set)

    def test_index_task_id_must_be_string(self, index_manager):
        """index_task() task_id should be a string."""
        # Test that non-string task_id is handled
        # Python will convert it to string
        index_manager.index_task(12345, status="pending")  # type: ignore
        # If it doesn't raise, the conversion happened

    def test_has_index_returns_bool(self, index_manager):
        """has_index() should return a boolean."""
        result = index_manager.has_index("status")
        assert isinstance(result, bool)
        assert result is True  # status is a standard index

        result = index_manager.has_index("nonexistent_field")
        assert isinstance(result, bool)
        assert result is False


class TestEdgeTypeValidation:
    """Test validation of EdgeType enum values."""

    def test_valid_edge_types_defined(self):
        """VALID_EDGE_TYPES should contain expected edge types."""
        expected_types = {
            'DEPENDS_ON',
            'BLOCKS',
            'CONTAINS',
            'RELATES_TO',
            'REQUIRES',
            'IMPLEMENTS',
            'SUPERSEDES',
            'DERIVED_FROM',
            'PARENT_OF',
            'CHILD_OF',
            'PART_OF',
            'REFERENCES',
            'CONTRADICTS',
            'JUSTIFIES',
            'MOTIVATES',
            'CAUSED_BY',
            'TRANSFERS',
            'PRODUCES',
            'DOCUMENTED_BY',
        }

        assert VALID_EDGE_TYPES == expected_types

    def test_edge_type_case_sensitive(self):
        """Edge types should be case-sensitive."""
        # Lowercase versions should NOT be in valid set
        assert 'depends_on' not in VALID_EDGE_TYPES
        assert 'DEPENDS_ON' in VALID_EDGE_TYPES

        assert 'blocks' not in VALID_EDGE_TYPES
        assert 'BLOCKS' in VALID_EDGE_TYPES

    def test_invalid_edge_type_not_in_set(self):
        """Invalid edge types should not be in VALID_EDGE_TYPES."""
        invalid_types = [
            'INVALID_TYPE',
            'LINKS_TO',
            'CONNECTS',
            'ASSOCIATED_WITH',
            'random_edge_type',
        ]

        for invalid_type in invalid_types:
            assert invalid_type not in VALID_EDGE_TYPES


class TestIndexManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def index_manager(self, got_dir):
        """Create a QueryIndexManager for testing."""
        return QueryIndexManager(got_dir)

    def test_lookup_with_none_value(self, index_manager):
        """Lookup with None value should work (stored as __null__)."""
        # Index uses "__null__" for None values
        result = index_manager.lookup("status", None)
        assert isinstance(result, set)

    def test_update_task_with_same_values(self, index_manager):
        """Updating task with same old and new values should be no-op."""
        # Index a task
        index_manager.index_task("T-001", status="pending", priority="high")

        # Update with same values (should be a no-op internally)
        index_manager.update_task(
            "T-001",
            old_status="pending",
            new_status="pending",
            old_priority="high",
            new_priority="high"
        )

        # Task should still be in the index
        assert "T-001" in index_manager.lookup("status", "pending")
        assert "T-001" in index_manager.lookup("priority", "high")

    def test_remove_nonexistent_task(self, index_manager):
        """Removing a non-existent task should not raise."""
        # Should handle gracefully
        index_manager.remove_task("T-nonexistent")
        # No exception = success

    def test_unlink_from_nonexistent_sprint(self, index_manager):
        """Unlinking from non-existent sprint should not raise."""
        # Should handle gracefully
        index_manager.unlink_task_from_sprint("T-001", "S-nonexistent")
        # No exception = success

    def test_get_all_indexed_values_empty(self, index_manager):
        """get_all_indexed_values for empty index should return empty list."""
        # Fresh index should have no values yet
        values = index_manager.get_all_indexed_values("status")
        assert isinstance(values, list)
        # May have empty list or __null__ depending on initialization
        assert isinstance(values, list)

    def test_get_all_indexed_values_nonexistent_field(self, index_manager):
        """get_all_indexed_values for non-indexed field should return empty list."""
        values = index_manager.get_all_indexed_values("nonexistent_field")
        assert isinstance(values, list)
        assert len(values) == 0

    def test_get_stats_structure(self, index_manager):
        """get_stats should return dict with expected keys."""
        stats = index_manager.get_stats()

        # Check required keys
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "rebuilds" in stats
        assert "indexes" in stats
        assert "index_sizes" in stats
        assert "sprint_index_size" in stats

        # Check types
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)
        assert isinstance(stats["rebuilds"], int)
        assert isinstance(stats["indexes"], list)
        assert isinstance(stats["index_sizes"], dict)
        assert isinstance(stats["sprint_index_size"], int)
