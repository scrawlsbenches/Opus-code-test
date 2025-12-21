"""
Tests for GoT conflict resolution.
"""

import pytest

from cortical.got.conflict import (
    ConflictResolver,
    ConflictStrategy,
    SyncConflict,
)
from cortical.got.types import Task, Decision, Edge
from cortical.got.errors import ConflictError


class TestConflictStrategy:
    """Test ConflictStrategy enum."""

    def test_strategy_values(self):
        """Verify strategy enum values."""
        assert ConflictStrategy.OURS.value == "ours"
        assert ConflictStrategy.THEIRS.value == "theirs"
        assert ConflictStrategy.MERGE.value == "merge"


class TestSyncConflict:
    """Test SyncConflict dataclass."""

    def test_sync_conflict_creation(self):
        """Test creating sync conflict."""
        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title", "status"]
        )

        assert conflict.entity_id == "T-001"
        assert conflict.entity_type == "task"
        assert conflict.local_version == 2
        assert conflict.remote_version == 3
        assert conflict.conflict_fields == ["title", "status"]


class TestConflictResolver:
    """Test ConflictResolver class."""

    def test_init_default_strategy(self):
        """Test resolver initializes with default strategy."""
        resolver = ConflictResolver()
        assert resolver.strategy == ConflictStrategy.OURS

    def test_init_custom_strategy(self):
        """Test resolver initializes with custom strategy."""
        resolver = ConflictResolver(strategy=ConflictStrategy.THEIRS)
        assert resolver.strategy == ConflictStrategy.THEIRS

    def test_detect_conflicts_no_overlap(self):
        """Test detecting conflicts when no common entities."""
        resolver = ConflictResolver()

        local = {"T-001": Task(id="T-001", title="Local task")}
        remote = {"T-002": Task(id="T-002", title="Remote task")}

        conflicts = resolver.detect_conflicts(local, remote)
        assert conflicts == []

    def test_detect_conflicts_same_version(self):
        """Test no conflict when versions match."""
        resolver = ConflictResolver()

        task = Task(id="T-001", title="Same", version=1)
        local = {"T-001": task}
        remote = {"T-001": task}

        conflicts = resolver.detect_conflicts(local, remote)
        assert conflicts == []

    def test_detect_conflicts_version_mismatch(self):
        """Test conflict detected on version mismatch."""
        resolver = ConflictResolver()

        local_task = Task(id="T-001", title="Local version", version=2)
        remote_task = Task(id="T-001", title="Remote version", version=3)

        local = {"T-001": local_task}
        remote = {"T-001": remote_task}

        conflicts = resolver.detect_conflicts(local, remote)

        assert len(conflicts) == 1
        assert conflicts[0].entity_id == "T-001"
        assert conflicts[0].local_version == 2
        assert conflicts[0].remote_version == 3
        assert "title" in conflicts[0].conflict_fields

    def test_detect_conflicts_ignores_metadata(self):
        """Test conflict detection ignores metadata fields."""
        resolver = ConflictResolver()

        # Same content, different metadata
        local_task = Task(
            id="T-001",
            title="Same",
            version=1,
            modified_at="2025-01-01T00:00:00Z"
        )
        remote_task = Task(
            id="T-001",
            title="Same",
            version=2,  # Different version triggers check
            modified_at="2025-01-02T00:00:00Z"
        )

        local = {"T-001": local_task}
        remote = {"T-001": remote_task}

        conflicts = resolver.detect_conflicts(local, remote)

        # Should detect version difference but not metadata as conflict field
        if conflicts:  # Only if versions differ
            for conflict in conflicts:
                assert "modified_at" not in conflict.conflict_fields
                assert "created_at" not in conflict.conflict_fields

    def test_ours_strategy_keeps_local(self):
        """Test OURS strategy returns local entity."""
        resolver = ConflictResolver(strategy=ConflictStrategy.OURS)

        local = Task(id="T-001", title="Local", version=2)
        remote = Task(id="T-001", title="Remote", version=3)

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title"]
        )

        result = resolver.resolve(conflict, local, remote)
        assert result is local
        assert result.title == "Local"

    def test_theirs_strategy_takes_remote(self):
        """Test THEIRS strategy returns remote entity."""
        resolver = ConflictResolver(strategy=ConflictStrategy.THEIRS)

        local = Task(id="T-001", title="Local", version=2)
        remote = Task(id="T-001", title="Remote", version=3)

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title"]
        )

        result = resolver.resolve(conflict, local, remote)
        assert result is remote
        assert result.title == "Remote"

    def test_merge_strategy_fails_on_same_field_conflict(self):
        """Test MERGE strategy raises error when same field conflicts."""
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)

        local = Task(id="T-001", title="Local", version=2)
        remote = Task(id="T-001", title="Remote", version=3)

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title"]  # Same field differs
        )

        with pytest.raises(ConflictError) as exc_info:
            resolver.resolve(conflict, local, remote)

        assert "Cannot auto-merge" in str(exc_info.value)
        assert "T-001" in str(exc_info.value)

    def test_merge_strategy_prefers_higher_version(self):
        """Test MERGE strategy prefers higher version when no field conflicts."""
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)

        local = Task(id="T-001", title="Same", version=2)
        remote = Task(id="T-001", title="Same", version=3)

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=[]  # No field conflicts
        )

        result = resolver.resolve(conflict, local, remote)
        assert result is remote  # Higher version
        assert result.version == 3

    def test_resolve_with_override_strategy(self):
        """Test resolve can override default strategy."""
        resolver = ConflictResolver(strategy=ConflictStrategy.OURS)

        local = Task(id="T-001", title="Local", version=2)
        remote = Task(id="T-001", title="Remote", version=3)

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title"]
        )

        # Override default OURS with THEIRS
        result = resolver.resolve(
            conflict, local, remote,
            strategy=ConflictStrategy.THEIRS
        )

        assert result is remote
        assert result.title == "Remote"

    def test_resolve_all_applies_strategy(self):
        """Test resolve_all applies strategy to all conflicts."""
        resolver = ConflictResolver(strategy=ConflictStrategy.OURS)

        local_t1 = Task(id="T-001", title="Local 1", version=2)
        local_t2 = Task(id="T-002", title="Local 2", version=3)

        remote_t1 = Task(id="T-001", title="Remote 1", version=3)
        remote_t2 = Task(id="T-002", title="Remote 2", version=4)

        local = {"T-001": local_t1, "T-002": local_t2}
        remote = {"T-001": remote_t1, "T-002": remote_t2}

        conflicts = [
            SyncConflict("T-001", "task", 2, 3, ["title"]),
            SyncConflict("T-002", "task", 3, 4, ["title"])
        ]

        resolved = resolver.resolve_all(conflicts, local, remote)

        assert len(resolved) == 2
        assert resolved["T-001"].title == "Local 1"  # OURS strategy
        assert resolved["T-002"].title == "Local 2"  # OURS strategy

    def test_resolve_all_with_different_entity_types(self):
        """Test resolve_all works with mixed entity types."""
        resolver = ConflictResolver(strategy=ConflictStrategy.THEIRS)

        task = Task(id="T-001", title="Task", version=1)
        decision = Decision(id="D-001", title="Decision", version=1)

        task_remote = Task(id="T-001", title="Remote Task", version=2)
        decision_remote = Decision(id="D-001", title="Remote Decision", version=2)

        local = {"T-001": task, "D-001": decision}
        remote = {"T-001": task_remote, "D-001": decision_remote}

        conflicts = [
            SyncConflict("T-001", "task", 1, 2, ["title"]),
            SyncConflict("D-001", "decision", 1, 2, ["title"])
        ]

        resolved = resolver.resolve_all(conflicts, local, remote)

        assert len(resolved) == 2
        assert resolved["T-001"].title == "Remote Task"
        assert resolved["D-001"].title == "Remote Decision"
