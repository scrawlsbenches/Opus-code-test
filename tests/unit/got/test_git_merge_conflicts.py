"""
Tests for git merge conflict handling in GoT sync layer.

These tests verify that the sync layer properly detects and handles
conflicts when multiple users/processes modify the same entities concurrently.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from cortical.got.conflict import (
    ConflictResolver,
    ConflictStrategy,
    SyncConflict,
)
from cortical.got.sync import SyncManager, SyncResult
from cortical.got.types import Task, Decision, Edge
from cortical.got.errors import ConflictError


class TestGitMergeConflicts:
    """Test git merge conflict detection and handling."""

    @pytest.fixture
    def resolver(self):
        """Create a conflict resolver for testing."""
        return ConflictResolver()

    @pytest.fixture
    def local_entities(self):
        """Create local entity state."""
        return {
            "T-001": Task(
                id="T-001",
                title="Local Task 1",
                status="in_progress",
                version=2,
                created_at="2025-12-21T10:00:00Z",
                modified_at="2025-12-21T11:00:00Z"
            ),
            "T-002": Task(
                id="T-002",
                title="Local Task 2",
                status="pending",
                version=1,
                created_at="2025-12-21T10:00:00Z",
                modified_at="2025-12-21T10:00:00Z"
            ),
        }

    @pytest.fixture
    def remote_entities(self):
        """Create remote entity state (with concurrent edits)."""
        return {
            "T-001": Task(
                id="T-001",
                title="Remote Task 1",  # Different title (conflict!)
                status="completed",     # Different status (conflict!)
                version=3,              # Higher version
                created_at="2025-12-21T10:00:00Z",
                modified_at="2025-12-21T11:30:00Z"
            ),
            "T-003": Task(
                id="T-003",
                title="Remote Task 3",  # New entity, no conflict
                status="pending",
                version=1,
                created_at="2025-12-21T11:00:00Z",
                modified_at="2025-12-21T11:00:00Z"
            ),
        }

    def test_conflict_detected_on_concurrent_edit(self, resolver, local_entities, remote_entities):
        """
        Test that conflicts are detected when two processes edit the same entity.

        Scenario:
        1. Both local and remote have entity T-001
        2. Local has version 2, remote has version 3
        3. Content differs (title and status)
        4. Conflict should be detected
        """
        conflicts = resolver.detect_conflicts(local_entities, remote_entities)

        # Should detect one conflict (T-001)
        assert len(conflicts) == 1

        conflict = conflicts[0]
        assert conflict.entity_id == "T-001"
        assert conflict.entity_type == "task"
        assert conflict.local_version == 2
        assert conflict.remote_version == 3

        # Should identify the conflicting fields
        assert "title" in conflict.conflict_fields
        assert "status" in conflict.conflict_fields

        # Should NOT include metadata in conflict fields
        assert "modified_at" not in conflict.conflict_fields
        assert "created_at" not in conflict.conflict_fields
        assert "version" not in conflict.conflict_fields

    def test_conflict_resolution_keeps_both_versions(self, resolver):
        """
        Test that conflict resolution preserves both local and remote versions.

        With OURS strategy: keep local
        With THEIRS strategy: keep remote
        """
        local = Task(
            id="T-001",
            title="Local Title",
            status="in_progress",
            version=2
        )
        remote = Task(
            id="T-001",
            title="Remote Title",
            status="completed",
            version=3
        )

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title", "status"]
        )

        # Test OURS strategy (keep local)
        resolver_ours = ConflictResolver(strategy=ConflictStrategy.OURS)
        result_ours = resolver_ours.resolve(conflict, local, remote)

        assert result_ours is local
        assert result_ours.title == "Local Title"
        assert result_ours.status == "in_progress"
        assert result_ours.version == 2

        # Test THEIRS strategy (keep remote)
        resolver_theirs = ConflictResolver(strategy=ConflictStrategy.THEIRS)
        result_theirs = resolver_theirs.resolve(conflict, local, remote)

        assert result_theirs is remote
        assert result_theirs.title == "Remote Title"
        assert result_theirs.status == "completed"
        assert result_theirs.version == 3

        # Both versions are preserved (not merged/lost)
        assert local.title == "Local Title"
        assert remote.title == "Remote Title"

    def test_conflict_message_includes_details(self):
        """
        Test that ConflictError includes entity ID, versions, and fields.

        The error should provide enough information to manually resolve.
        """
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)

        local = Task(
            id="T-001",
            title="Local Title",
            status="in_progress",
            version=2,
            modified_at="2025-12-21T11:00:00Z"
        )
        remote = Task(
            id="T-001",
            title="Remote Title",
            status="completed",
            version=3,
            modified_at="2025-12-21T11:30:00Z"
        )

        conflict = SyncConflict(
            entity_id="T-001",
            entity_type="task",
            local_version=2,
            remote_version=3,
            conflict_fields=["title", "status"]
        )

        # MERGE strategy should fail with detailed error
        with pytest.raises(ConflictError) as exc_info:
            resolver.resolve(conflict, local, remote)

        error = exc_info.value

        # Verify error message contains details
        assert "T-001" in str(error)
        assert "Cannot auto-merge" in str(error)

        # Verify error context includes conflict fields
        assert error.context.get("entity_id") == "T-001"
        assert "title" in error.context.get("conflict_fields", [])
        assert "status" in error.context.get("conflict_fields", [])

    def test_no_conflict_on_independent_entities(self, resolver):
        """
        Test that editing different entities concurrently does NOT cause conflict.

        Scenario:
        1. Local edits T-001
        2. Remote edits T-002
        3. No overlap = no conflict
        """
        local_entities = {
            "T-001": Task(
                id="T-001",
                title="Local Task 1",
                status="in_progress",
                version=2
            ),
        }

        remote_entities = {
            "T-002": Task(
                id="T-002",
                title="Remote Task 2",
                status="completed",
                version=2
            ),
        }

        conflicts = resolver.detect_conflicts(local_entities, remote_entities)

        # No common entities = no conflicts
        assert len(conflicts) == 0

    def test_no_conflict_when_only_metadata_differs(self, resolver):
        """
        Test that differing metadata alone doesn't cause conflict.

        Only content field changes should be considered conflicts.
        """
        local = Task(
            id="T-001",
            title="Same Title",
            status="pending",
            version=1,
            created_at="2025-12-21T10:00:00Z",
            modified_at="2025-12-21T11:00:00Z"
        )

        remote = Task(
            id="T-001",
            title="Same Title",
            status="pending",
            version=2,  # Different version (triggers check)
            created_at="2025-12-21T10:00:00Z",
            modified_at="2025-12-21T12:00:00Z"  # Different timestamp
        )

        local_entities = {"T-001": local}
        remote_entities = {"T-001": remote}

        conflicts = resolver.detect_conflicts(local_entities, remote_entities)

        # Version differs but content is same = no conflict
        assert len(conflicts) == 0

    def test_conflict_with_different_entity_types(self, resolver):
        """
        Test conflict detection works for Tasks, Decisions, and Edges.
        """
        # Test with Decision
        local_decision = Decision(
            id="D-001",
            title="Local Decision",
            rationale="Local reasoning",
            version=1
        )
        remote_decision = Decision(
            id="D-001",
            title="Remote Decision",
            rationale="Remote reasoning",
            version=2
        )

        conflicts = resolver.detect_conflicts(
            {"D-001": local_decision},
            {"D-001": remote_decision}
        )

        assert len(conflicts) == 1
        assert conflicts[0].entity_type == "decision"
        assert "title" in conflicts[0].conflict_fields
        assert "rationale" in conflicts[0].conflict_fields

        # Test with Edge
        local_edge = Edge(
            id="E-001",
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            weight=0.8,
            version=1
        )
        remote_edge = Edge(
            id="E-001",
            source_id="T-001",
            target_id="T-002",
            edge_type="BLOCKS",  # Different type
            weight=0.9,          # Different weight
            version=2
        )

        conflicts = resolver.detect_conflicts(
            {"E-001": local_edge},
            {"E-001": remote_edge}
        )

        assert len(conflicts) == 1
        assert conflicts[0].entity_type == "edge"
        assert "edge_type" in conflicts[0].conflict_fields
        assert "weight" in conflicts[0].conflict_fields

    def test_multiple_concurrent_conflicts(self, resolver):
        """
        Test detection of multiple conflicts in a single sync.
        """
        local_entities = {
            "T-001": Task(id="T-001", title="Local 1", version=2),
            "T-002": Task(id="T-002", title="Local 2", version=3),
            "T-003": Task(id="T-003", title="Local 3", version=1),
        }

        remote_entities = {
            "T-001": Task(id="T-001", title="Remote 1", version=3),  # Conflict
            "T-002": Task(id="T-002", title="Remote 2", version=4),  # Conflict
            "T-003": Task(id="T-003", title="Local 3", version=1),   # No conflict (same)
        }

        conflicts = resolver.detect_conflicts(local_entities, remote_entities)

        # Should detect 2 conflicts (T-001 and T-002)
        assert len(conflicts) == 2

        conflict_ids = {c.entity_id for c in conflicts}
        assert "T-001" in conflict_ids
        assert "T-002" in conflict_ids
        assert "T-003" not in conflict_ids

    def test_sync_pull_detects_git_merge_conflict(self, tmp_path):
        """
        Test that SyncManager.pull() detects git merge conflicts.

        Simulates git pull --rebase failing with merge conflict.
        """
        # Create minimal directory structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        got_dir = repo_dir / ".got"
        got_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        # Create SyncManager
        manager = SyncManager(got_dir, git_dir=repo_dir)

        # Mock can_sync to return True (no active transactions)
        with patch.object(manager, 'can_sync', return_value=True):
            # Mock get_status to show we're behind
            mock_status = Mock(ahead=0, behind=1, dirty=False, has_active_tx=False)
            with patch.object(manager, 'get_status', return_value=mock_status):
                # Mock _run_git to simulate conflict
                from cortical.got.errors import SyncError

                def mock_run_git(args, timeout=30):
                    if args == ["fetch"]:
                        return ""
                    elif args == ["pull", "--rebase"]:
                        # Simulate merge conflict
                        raise SyncError(
                            "Git command failed: CONFLICT (content): Merge conflict in .got/tasks.json",
                            command="pull --rebase"
                        )
                    return ""

                with patch.object(manager, '_run_git', side_effect=mock_run_git):
                    result = manager.pull()

        # Verify conflict was detected
        assert result.success is False
        assert result.action == "pull"
        assert "conflict" in result.error.lower()

    def test_sync_push_succeeds_without_conflict(self, tmp_path):
        """
        Test that SyncManager.push() succeeds when no conflicts exist.
        """
        # Create minimal directory structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        got_dir = repo_dir / ".got"
        got_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        # Create SyncManager
        manager = SyncManager(got_dir, git_dir=repo_dir)

        # Mock can_sync to return True
        with patch.object(manager, 'can_sync', return_value=True):
            # Mock get_status to show we're ahead
            mock_status = Mock(ahead=1, behind=0, dirty=False, has_active_tx=False)
            with patch.object(manager, 'get_status', return_value=mock_status):
                # Mock _get_current_commit
                with patch.object(manager, '_get_current_commit', return_value="abc123"):
                    # Mock _run_git to succeed
                    def mock_run_git(args, timeout=30):
                        return ""  # All git commands succeed

                    with patch.object(manager, '_run_git', side_effect=mock_run_git):
                        result = manager.push()

        # Verify push succeeded
        assert result.success is True
        assert result.action == "push"
        assert result.version == "abc123"
        assert result.error is None

    def test_resolve_all_preserves_entity_context(self, resolver):
        """
        Test that resolve_all maintains entity relationships and context.
        """
        # Create interconnected entities
        local_task = Task(id="T-001", title="Local Task", version=2)
        local_decision = Decision(
            id="D-001",
            title="Local Decision",
            affects=["T-001"],  # References the task
            version=2
        )

        remote_task = Task(id="T-001", title="Remote Task", version=3)
        remote_decision = Decision(
            id="D-001",
            title="Remote Decision",
            affects=["T-001"],  # Same reference
            version=3
        )

        local = {"T-001": local_task, "D-001": local_decision}
        remote = {"T-001": remote_task, "D-001": remote_decision}

        conflicts = [
            SyncConflict("T-001", "task", 2, 3, ["title"]),
            SyncConflict("D-001", "decision", 2, 3, ["title"])
        ]

        # Resolve with THEIRS strategy
        resolver_theirs = ConflictResolver(strategy=ConflictStrategy.THEIRS)
        resolved = resolver_theirs.resolve_all(conflicts, local, remote)

        # Verify all entities resolved
        assert len(resolved) == 2

        # Verify decision still references the task
        assert "T-001" in resolved["D-001"].affects

        # Verify both are remote versions
        assert resolved["T-001"].title == "Remote Task"
        assert resolved["D-001"].title == "Remote Decision"

    def test_conflict_field_detection_precision(self, resolver):
        """
        Test that _find_conflicting_fields accurately identifies differences.
        """
        # Create tasks with specific field differences
        local = Task(
            id="T-001",
            title="Same Title",
            status="in_progress",  # Different
            priority="high",       # Different
            description="Same description",
            version=2
        )

        remote = Task(
            id="T-001",
            title="Same Title",
            status="completed",    # Different
            priority="medium",     # Different
            description="Same description",
            version=3
        )

        conflicting_fields = resolver._find_conflicting_fields(local, remote)

        # Should detect exactly the differing fields
        assert "status" in conflicting_fields
        assert "priority" in conflicting_fields

        # Should NOT detect same fields
        assert "title" not in conflicting_fields
        assert "description" not in conflicting_fields

        # Should ignore metadata
        assert "version" not in conflicting_fields
        assert "modified_at" not in conflicting_fields
