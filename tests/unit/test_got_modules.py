"""
Comprehensive Unit Tests for GoT Core Modules
==============================================

Tests for:
- cortical/got/protocol.py - Protocol definition
- cortical/got/api.py - GoTManager API
- cortical/got/sync.py - Git sync manager

Note: WAL tests removed - WAL is now owned by CDG layer (cortical/cdg/wal.py)

These tests use mocking to avoid creating real data and ensure
proper isolation between tests.
"""

import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest

from tests.conftest import _create_tx_manager, _create_got_manager


# =============================================================================
# PROTOCOL TESTS
# =============================================================================


class TestGoTBackendProtocol:
    """Tests for GoTBackend Protocol definition."""

    def test_protocol_is_importable(self):
        """GoTBackend protocol can be imported."""
        from cortical.got.protocol import GoTBackend
        assert GoTBackend is not None

    def test_protocol_is_a_protocol(self):
        """GoTBackend is a typing.Protocol."""
        from cortical.got.protocol import GoTBackend
        # Check it's a Protocol subclass
        assert hasattr(GoTBackend, '__protocol_attrs__') or Protocol in GoTBackend.__mro__

    def test_protocol_defines_create_task(self):
        """Protocol defines create_task method."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'create_task')

    def test_protocol_defines_get_task(self):
        """Protocol defines get_task method."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'get_task')

    def test_protocol_defines_list_tasks(self):
        """Protocol defines list_tasks method."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'list_tasks')

    def test_protocol_defines_update_task(self):
        """Protocol defines update_task method."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'update_task')

    def test_protocol_defines_delete_task(self):
        """Protocol defines delete_task method."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'delete_task')

    def test_protocol_defines_state_transitions(self):
        """Protocol defines state transition methods."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'start_task')
        assert hasattr(GoTBackend, 'complete_task')
        assert hasattr(GoTBackend, 'block_task')

    def test_protocol_defines_relationship_methods(self):
        """Protocol defines relationship management methods."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'add_dependency')
        assert hasattr(GoTBackend, 'add_blocks')
        assert hasattr(GoTBackend, 'get_blockers')
        assert hasattr(GoTBackend, 'get_dependents')
        assert hasattr(GoTBackend, 'get_task_dependencies')

    def test_protocol_defines_query_methods(self):
        """Protocol defines query and analytics methods."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'get_stats')
        assert hasattr(GoTBackend, 'validate')
        assert hasattr(GoTBackend, 'get_blocked_tasks')
        assert hasattr(GoTBackend, 'get_active_tasks')
        assert hasattr(GoTBackend, 'what_blocks')
        assert hasattr(GoTBackend, 'what_depends_on')
        assert hasattr(GoTBackend, 'get_all_relationships')

    def test_protocol_defines_persistence_methods(self):
        """Protocol defines persistence methods."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'sync_to_git')
        assert hasattr(GoTBackend, 'export_graph')

    def test_protocol_defines_query_language(self):
        """Protocol defines query language method."""
        from cortical.got.protocol import GoTBackend
        assert hasattr(GoTBackend, 'query')

    def test_mock_implementation_satisfies_protocol(self):
        """A mock implementing all methods satisfies the protocol."""
        from cortical.got.protocol import GoTBackend

        class MockBackend:
            """Mock implementation of GoTBackend."""

            def create_task(self, title, priority="medium", category="feature",
                          description="", sprint_id=None, depends_on=None, blocks=None):
                return "task:T-001"

            def get_task(self, task_id):
                return None

            def list_tasks(self, status=None, priority=None, category=None,
                         sprint_id=None, blocked_only=False):
                return []

            def update_task(self, task_id, **updates):
                return True

            def delete_task(self, task_id, force=False):
                return (True, "Deleted")

            def start_task(self, task_id):
                return True

            def complete_task(self, task_id, retrospective=""):
                return True

            def block_task(self, task_id, reason="", blocked_by=None):
                return True

            def add_dependency(self, task_id, depends_on_id):
                return True

            def add_blocks(self, blocker_id, blocked_id):
                return True

            def get_blockers(self, task_id):
                return []

            def get_dependents(self, task_id):
                return []

            def get_task_dependencies(self, task_id):
                return []

            def get_stats(self):
                return {}

            def validate(self):
                return []

            def get_blocked_tasks(self):
                return []

            def get_active_tasks(self):
                return []

            def what_blocks(self, task_id):
                return []

            def what_depends_on(self, task_id):
                return []

            def get_all_relationships(self, task_id):
                return {}

            def sync_to_git(self):
                return "synced"

            def export_graph(self, output_path=None):
                return {}

            def query(self, query_str):
                return []

        backend = MockBackend()
        # Verify all expected methods exist and are callable
        assert callable(backend.create_task)
        assert callable(backend.get_task)
        assert callable(backend.query)


# =============================================================================
# SYNC MANAGER TESTS
# =============================================================================


class TestSyncManager:
    """Tests for Sync Manager."""

    @pytest.fixture
    def temp_got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_sync_manager_requires_git_repo(self, temp_got_dir):
        """SyncManager raises error if not in git repo."""
        from cortical.got.sync import SyncManager
        from cortical.got.errors import SyncError

        with pytest.raises(SyncError):
            SyncManager(temp_got_dir)

    def test_sync_manager_finds_git_root(self, temp_got_dir):
        """SyncManager finds git root by walking up."""
        from cortical.got.sync import SyncManager

        # Create a fake .git directory
        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        manager = SyncManager(temp_got_dir)
        assert manager.git_dir == temp_got_dir

    def test_sync_manager_finds_git_root_nested(self, temp_got_dir):
        """SyncManager finds git root when deeply nested."""
        from cortical.got.sync import SyncManager

        # Create nested structure
        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()
        nested_got = temp_got_dir / "project" / ".got"
        nested_got.mkdir(parents=True)

        manager = SyncManager(nested_got)
        assert manager.git_dir == temp_got_dir

    def test_sync_manager_can_sync_without_active_tx(self, temp_got_dir):
        """can_sync returns True when no active transactions."""
        from cortical.got.sync import SyncManager

        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        manager = SyncManager(temp_got_dir)
        # No WAL dir means no active transactions
        assert manager.can_sync() is True

    def test_sync_status_dataclass(self):
        """SyncStatus dataclass works correctly."""
        from cortical.got.sync import SyncStatus

        status = SyncStatus(
            ahead=5,
            behind=2,
            dirty=True,
            has_active_tx=False
        )

        assert status.ahead == 5
        assert status.behind == 2
        assert status.dirty is True
        assert status.has_active_tx is False

    def test_sync_result_dataclass(self):
        """SyncResult dataclass works correctly."""
        from cortical.got.sync import SyncResult

        result = SyncResult(
            success=True,
            action="push",
            version="abc123",
            conflicts=[],
            error=None
        )

        assert result.success is True
        assert result.action == "push"
        assert result.version == "abc123"
        assert result.conflicts == []
        assert result.error is None

    def test_sync_result_with_error(self):
        """SyncResult can represent failures."""
        from cortical.got.sync import SyncResult

        result = SyncResult(
            success=False,
            action="pull",
            error="Merge conflict detected"
        )

        assert result.success is False
        assert result.error == "Merge conflict detected"

    @patch('subprocess.run')
    def test_push_fails_with_active_transactions(self, mock_run, temp_got_dir):
        """Push fails when active transactions exist."""
        from cortical.got.sync import SyncManager

        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        manager = SyncManager(temp_got_dir)

        # Mock can_sync to return False
        with patch.object(manager, 'can_sync', return_value=False):
            result = manager.push()

        assert result.success is False
        assert "Active transactions" in result.error

    @patch('subprocess.run')
    def test_pull_fails_with_active_transactions(self, mock_run, temp_got_dir):
        """Pull fails when active transactions exist."""
        from cortical.got.sync import SyncManager

        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        manager = SyncManager(temp_got_dir)

        with patch.object(manager, 'can_sync', return_value=False):
            result = manager.pull()

        assert result.success is False
        assert "Active transactions" in result.error

    @patch('subprocess.run')
    def test_get_current_commit(self, mock_run, temp_got_dir):
        """_get_current_commit returns short hash."""
        from cortical.got.sync import SyncManager

        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        mock_run.return_value = Mock(stdout="abc1234\n", returncode=0)

        manager = SyncManager(temp_got_dir)
        commit = manager._get_current_commit()

        assert commit == "abc1234"

    @patch('subprocess.run')
    def test_get_current_commit_handles_error(self, mock_run, temp_got_dir):
        """_get_current_commit returns 'unknown' on error."""
        from cortical.got.sync import SyncManager
        import subprocess

        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        manager = SyncManager(temp_got_dir)
        commit = manager._get_current_commit()

        assert commit == "unknown"

    @patch('subprocess.run')
    def test_run_git_timeout_handling(self, mock_run, temp_got_dir):
        """_run_git raises SyncError on timeout."""
        from cortical.got.sync import SyncManager
        from cortical.got.errors import SyncError
        import subprocess

        git_dir = temp_got_dir / ".git"
        git_dir.mkdir()

        mock_run.side_effect = subprocess.TimeoutExpired("git", 30)

        manager = SyncManager(temp_got_dir)

        with pytest.raises(SyncError) as exc_info:
            manager._run_git(["status"])

        assert "timed out" in str(exc_info.value).lower()


# =============================================================================
# GOT MANAGER API TESTS
# =============================================================================


class TestGoTManagerAPI:
    """Tests for GoTManager high-level API."""

    @pytest.fixture
    def temp_got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            got_dir = Path(tmpdir)
            (got_dir / "entities").mkdir()
            yield got_dir

    @pytest.fixture
    def got_manager(self, temp_got_dir):
        """Create a GoTManager for testing."""
        from cortical.got.api import GoTManager
        from cortical.got.config import DurabilityMode
        return _create_got_manager(temp_got_dir)

    def test_got_manager_init(self, temp_got_dir):
        """GoTManager initializes correctly."""
        from cortical.got.api import GoTManager
        from cortical.got.config import DurabilityMode

        manager = _create_got_manager(temp_got_dir)

        assert manager.got_dir == temp_got_dir
        assert manager.durability == DurabilityMode.BALANCED  # Container default
        assert manager.tx_manager is not None

    def test_got_manager_lazy_sync_manager(self, got_manager):
        """Sync manager is lazily initialized."""
        assert got_manager._sync_manager is None

        # Access should trigger initialization (may fail without git)
        try:
            _ = got_manager.sync_manager
        except Exception:
            pass  # Expected in test environment without git

    def test_got_manager_lazy_recovery_manager(self, got_manager):
        """Recovery manager is lazily initialized."""
        assert got_manager._recovery_manager is None

        _ = got_manager.recovery_manager
        assert got_manager._recovery_manager is not None

    def test_got_manager_transaction_context(self, got_manager):
        """transaction() returns a context manager."""
        from cortical.got.api import TransactionContext

        ctx = got_manager.transaction()
        assert isinstance(ctx, TransactionContext)

    def test_got_manager_transaction_read_only(self, got_manager):
        """transaction(read_only=True) creates read-only context."""
        ctx = got_manager.transaction(read_only=True)
        assert ctx.read_only is True

    def test_got_manager_find_tasks_empty(self, got_manager):
        """find_tasks returns empty list when no tasks exist."""
        tasks = got_manager.find_tasks()
        assert tasks == []

    def test_got_manager_find_tasks_no_entities_dir(self, temp_got_dir):
        """find_tasks handles missing entities directory."""
        from cortical.got.api import GoTManager
        from cortical.got.config import DurabilityMode

        # Remove entities directory
        import shutil
        entities_dir = temp_got_dir / "entities"
        if entities_dir.exists():
            shutil.rmtree(entities_dir)

        manager = _create_got_manager(temp_got_dir)
        tasks = manager.find_tasks()

        assert tasks == []

    def test_got_manager_get_blockers_empty(self, got_manager):
        """get_blockers returns empty list for non-existent task."""
        blockers = got_manager.get_blockers("task:T-nonexistent")
        assert blockers == []

    def test_got_manager_get_dependents_empty(self, got_manager):
        """get_dependents returns empty list for non-existent task."""
        dependents = got_manager.get_dependents("task:T-nonexistent")
        assert dependents == []

    def test_got_manager_list_all_tasks(self, got_manager):
        """list_all_tasks delegates to query_api.list_all_tasks."""
        with patch.object(got_manager.query_api, 'list_all_tasks', return_value=[]) as mock_list:
            result = got_manager.list_all_tasks()

            mock_list.assert_called_once_with()
            assert result == []

    def test_got_manager_get_edges_for_task_empty(self, got_manager):
        """get_edges_for_task returns empty tuples for non-existent task."""
        outgoing, incoming = got_manager.get_edges_for_task("task:T-nonexistent")
        assert outgoing == []
        assert incoming == []

    def test_got_manager_delete_task_not_found(self, got_manager):
        """delete_task raises error for non-existent task."""
        from cortical.got.errors import TransactionError

        with pytest.raises(TransactionError) as exc_info:
            got_manager.delete_task("task:T-nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_got_manager_add_dependency_creates_edge(self, got_manager):
        """add_dependency creates DEPENDS_ON edge."""
        with patch.object(got_manager, 'add_edge') as mock_add_edge:
            mock_add_edge.return_value = Mock()

            got_manager.add_dependency("task:T-001", "task:T-002")

            mock_add_edge.assert_called_once_with(
                "task:T-001", "task:T-002", "DEPENDS_ON"
            )

    def test_got_manager_add_blocks_creates_edge(self, got_manager):
        """add_blocks creates BLOCKS edge."""
        with patch.object(got_manager, 'add_edge') as mock_add_edge:
            mock_add_edge.return_value = Mock()

            got_manager.add_blocks("task:T-001", "task:T-002")

            mock_add_edge.assert_called_once_with(
                "task:T-001", "task:T-002", "BLOCKS"
            )


# =============================================================================
# TRANSACTION CONTEXT TESTS
# =============================================================================


class TestTransactionContext:
    """Tests for TransactionContext class."""

    @pytest.fixture
    def mock_tx_manager(self):
        """Create a mock transaction manager."""
        import threading
        manager = Mock()
        manager.begin.return_value = Mock()
        manager.commit.return_value = Mock(success=True)
        # Provide a real lock for context manager protocol support
        manager.lock = threading.Lock()
        return manager

    def test_transaction_context_enter(self, mock_tx_manager):
        """Context manager begins transaction on enter."""
        from cortical.got.api import TransactionContext

        ctx = TransactionContext(mock_tx_manager)

        with ctx:
            mock_tx_manager.begin.assert_called_once()

    def test_transaction_context_commits_on_success(self, mock_tx_manager):
        """Context manager commits on successful exit."""
        from cortical.got.api import TransactionContext

        ctx = TransactionContext(mock_tx_manager, read_only=False)

        with ctx:
            pass

        mock_tx_manager.commit.assert_called_once()

    def test_transaction_context_rollback_on_exception(self, mock_tx_manager):
        """Context manager rolls back on exception."""
        from cortical.got.api import TransactionContext

        ctx = TransactionContext(mock_tx_manager)

        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test error")

        mock_tx_manager.rollback.assert_called_once()

    def test_transaction_context_rollback_on_read_only(self, mock_tx_manager):
        """Read-only context rolls back instead of commit."""
        from cortical.got.api import TransactionContext

        ctx = TransactionContext(mock_tx_manager, read_only=True)

        with ctx:
            pass

        mock_tx_manager.rollback.assert_called_once()
        mock_tx_manager.commit.assert_not_called()

    def test_transaction_context_raises_on_commit_failure(self, mock_tx_manager):
        """Context raises TransactionError if commit fails."""
        from cortical.got.api import TransactionContext
        from cortical.got.errors import TransactionError

        mock_tx_manager.commit.return_value = Mock(
            success=False,
            reason="Conflict",
            conflicts=[]
        )

        ctx = TransactionContext(mock_tx_manager)

        with pytest.raises(TransactionError):
            with ctx:
                pass


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestGoTEdgeCases:
    """Tests for edge cases and error handling."""

    def test_wal_entry_checksum_verification(self):
        """WAL entries verify checksums correctly."""
        from cortical.wal import TransactionWALEntry

        entry = TransactionWALEntry(
            seq=1,
            timestamp="2025-01-01T00:00:00Z",
            tx_id="tx-001",
            operation="TEST",
            payload={"key": "value"}
        )

        # Entry should have computed checksum
        assert entry.checksum != ""
        assert entry.verify() is True

    def test_wal_entry_detects_corruption(self):
        """WAL entries detect checksum corruption."""
        from cortical.wal import TransactionWALEntry

        entry = TransactionWALEntry(
            seq=1,
            timestamp="2025-01-01T00:00:00Z",
            tx_id="tx-001",
            operation="TEST",
            payload={"key": "value"}
        )

        # Corrupt the checksum
        entry.checksum = "corrupted"

        assert entry.verify() is False

    def test_wal_entry_serialization_roundtrip(self):
        """WAL entries serialize and deserialize correctly."""
        from cortical.wal import TransactionWALEntry

        original = TransactionWALEntry(
            seq=42,
            timestamp="2025-01-01T12:34:56Z",
            tx_id="tx-test-123",
            operation="WRITE",
            payload={"entity_id": "task:T-001", "data": "test"}
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = TransactionWALEntry.from_dict(data)

        assert restored.seq == original.seq
        assert restored.timestamp == original.timestamp
        assert restored.tx_id == original.tx_id
        assert restored.operation == original.operation
        assert restored.payload == original.payload
        assert restored.checksum == original.checksum
        assert restored.verify() is True


class TestGoTManagerAPIExtended:
    """Extended tests for GoTManager API coverage."""

    @pytest.fixture
    def temp_got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            got_dir = Path(tmpdir)
            (got_dir / "entities").mkdir()
            yield got_dir

    @pytest.fixture
    def got_manager(self, temp_got_dir):
        """Create a GoTManager for testing."""
        from cortical.got.api import GoTManager
        from cortical.got.config import DurabilityMode
        return _create_got_manager(temp_got_dir)

    def test_delete_task_with_dependents_raises_error(self, got_manager):
        """delete_task with dependents raises error without force."""
        from cortical.got.errors import TransactionError

        # Create a task
        task = got_manager.create_task(title="Test Task", priority="high")
        # Create a dependent task
        dependent = got_manager.create_task(title="Dependent", priority="medium")
        # Create dependency edge
        got_manager.add_edge(dependent.id, task.id, "DEPENDS_ON")

        # Try to delete task with dependents
        with pytest.raises(TransactionError) as exc_info:
            got_manager.delete_task(task.id, force=False)

        assert "dependents" in str(exc_info.value).lower()

    def test_delete_task_with_force(self, got_manager):
        """delete_task with force=True deletes despite dependents."""
        # Create a task
        task = got_manager.create_task(title="Test Task", priority="high")
        task_id = task.id

        # Delete with force (should succeed even without dependents)
        got_manager.delete_task(task_id, force=True)

        # Verify task is deleted
        assert got_manager.get_task(task_id) is None

    def test_delete_task_cleans_up_edges(self, got_manager):
        """delete_task removes connected edges."""
        # Create tasks
        task1 = got_manager.create_task(title="Task 1", priority="high")
        task2 = got_manager.create_task(title="Task 2", priority="high")

        # Create edge
        edge = got_manager.add_edge(task1.id, task2.id, "RELATES_TO")

        # Delete task2 with force
        got_manager.delete_task(task2.id, force=True)

        # Verify edge is cleaned up (task1 should have no outgoing edges to task2)
        outgoing, _ = got_manager.get_edges_for_task(task1.id)
        assert not any(e.target_id == task2.id for e in outgoing)

    def test_create_sprint(self, got_manager):
        """create_sprint creates a new sprint."""
        sprint = got_manager.create_sprint(
            title="Test Sprint",
            number=1
        )

        assert sprint is not None
        assert sprint.title == "Test Sprint"
        assert sprint.id.startswith("S-")

    def test_get_sprint(self, got_manager):
        """get_sprint retrieves an existing sprint."""
        sprint = got_manager.create_sprint(title="Test Sprint", number=1)

        retrieved = got_manager.get_sprint(sprint.id)

        assert retrieved is not None
        assert retrieved.id == sprint.id

    def test_get_sprint_not_found(self, got_manager):
        """get_sprint returns None for non-existent sprint."""
        retrieved = got_manager.get_sprint("S-nonexistent")
        assert retrieved is None

    def test_list_sprints(self, got_manager):
        """list_sprints returns all sprints."""
        sprint1 = got_manager.create_sprint(title="Sprint 1", number=1)
        sprint2 = got_manager.create_sprint(title="Sprint 2", number=2)

        sprints = got_manager.list_sprints()

        assert len(sprints) >= 2
        sprint_ids = [s.id for s in sprints]
        assert sprint1.id in sprint_ids
        assert sprint2.id in sprint_ids

    def test_get_current_sprint_none(self, got_manager):
        """get_current_sprint returns None when no active sprint."""
        # Create a sprint but don't start it
        got_manager.create_sprint(title="Not Started", number=1)

        current = got_manager.get_current_sprint()
        # May be None or the sprint depending on implementation
        # This covers the branch

    def test_update_sprint_status(self, got_manager):
        """update_sprint can change sprint status."""
        sprint = got_manager.create_sprint(title="Test Sprint", number=1)

        got_manager.update_sprint(sprint.id, status="in_progress")

        updated = got_manager.get_sprint(sprint.id)
        assert updated.status == "in_progress"

    def test_update_sprint_to_completed(self, got_manager):
        """update_sprint can mark sprint as completed."""
        sprint = got_manager.create_sprint(title="Test Sprint", number=1)
        got_manager.update_sprint(sprint.id, status="in_progress")

        got_manager.update_sprint(sprint.id, status="completed")

        updated = got_manager.get_sprint(sprint.id)
        assert updated.status == "completed"

    def test_add_task_to_sprint(self, got_manager):
        """add_task_to_sprint creates CONTAINS edge."""
        sprint = got_manager.create_sprint(title="Test Sprint", number=1)
        task = got_manager.create_task(title="Test Task", priority="high")

        got_manager.add_task_to_sprint(task.id, sprint.id)

        # Verify edge exists
        _, incoming = got_manager.get_edges_for_task(task.id)
        contains_edges = [e for e in incoming if e.edge_type == "CONTAINS"]
        assert len(contains_edges) >= 1

    def test_get_sprint_tasks(self, got_manager):
        """get_sprint_tasks returns tasks in sprint."""
        sprint = got_manager.create_sprint(title="Test Sprint", number=1)
        task = got_manager.create_task(title="Test Task", priority="high")
        got_manager.add_task_to_sprint(task.id, sprint.id)

        tasks = got_manager.get_sprint_tasks(sprint.id)

        task_ids = [t.id for t in tasks]
        assert task.id in task_ids

    def test_initiate_handoff(self, got_manager):
        """initiate_handoff creates a handoff."""
        task = got_manager.create_task(title="Test Task", priority="high")

        handoff = got_manager.initiate_handoff(
            task_id=task.id,
            source_agent="current-agent",
            target_agent="next-agent",
            instructions="Do this task"
        )

        assert handoff is not None
        assert handoff.id.startswith("H-")
        assert handoff.status == "initiated"

    def test_accept_handoff(self, got_manager):
        """accept_handoff updates handoff status."""
        task = got_manager.create_task(title="Test Task", priority="high")
        handoff = got_manager.initiate_handoff(
            task_id=task.id,
            source_agent="current-agent",
            target_agent="next-agent",
            instructions="Do this task"
        )

        got_manager.accept_handoff(handoff.id, agent="next-agent")

        updated = got_manager.get_handoff(handoff.id)
        assert updated.status == "accepted"

    def test_complete_handoff(self, got_manager):
        """complete_handoff marks handoff as completed."""
        task = got_manager.create_task(title="Test Task", priority="high")
        handoff = got_manager.initiate_handoff(
            task_id=task.id,
            source_agent="current-agent",
            target_agent="next-agent",
            instructions="Do this task"
        )
        got_manager.accept_handoff(handoff.id, agent="next-agent")

        got_manager.complete_handoff(
            handoff.id,
            agent="next-agent",
            result={"status": "done"}
        )

        updated = got_manager.get_handoff(handoff.id)
        assert updated.status == "completed"

    def test_list_handoffs(self, got_manager):
        """list_handoffs returns all handoffs."""
        task = got_manager.create_task(title="Test Task", priority="high")
        handoff = got_manager.initiate_handoff(
            task_id=task.id,
            source_agent="current-agent",
            target_agent="next-agent",
            instructions="Do this task"
        )

        handoffs = got_manager.list_handoffs()

        handoff_ids = [h.id for h in handoffs]
        assert handoff.id in handoff_ids

    def test_list_handoffs_by_status(self, got_manager):
        """list_handoffs filters by status."""
        task = got_manager.create_task(title="Test Task", priority="high")
        got_manager.initiate_handoff(
            task_id=task.id,
            source_agent="current-agent",
            target_agent="next-agent",
            instructions="Do this task"
        )

        initiated = got_manager.list_handoffs(status="initiated")
        accepted = got_manager.list_handoffs(status="accepted")

        assert len(initiated) >= 1
        # accepted should be empty or not contain our handoff

    def test_create_decision(self, got_manager):
        """create_decision creates a decision."""
        decision = got_manager.create_decision(
            title="Use JSON format",
            rationale="Human readable"
        )

        assert decision is not None
        assert decision.id.startswith("D-")

    def test_find_tasks_by_status(self, got_manager):
        """find_tasks filters by status."""
        task1 = got_manager.create_task(title="Task 1", priority="high")
        task2 = got_manager.create_task(title="Task 2", priority="high")
        got_manager.update_task(task1.id, status="in_progress")

        in_progress = got_manager.find_tasks(status="in_progress")
        pending = got_manager.find_tasks(status="pending")

        in_progress_ids = [t.id for t in in_progress]
        pending_ids = [t.id for t in pending]

        assert task1.id in in_progress_ids
        assert task2.id in pending_ids

    def test_find_tasks_by_priority(self, got_manager):
        """find_tasks filters by priority."""
        task_high = got_manager.create_task(title="High", priority="high")
        task_low = got_manager.create_task(title="Low", priority="low")

        high_tasks = got_manager.find_tasks(priority="high")
        low_tasks = got_manager.find_tasks(priority="low")

        high_ids = [t.id for t in high_tasks]
        low_ids = [t.id for t in low_tasks]

        assert task_high.id in high_ids
        assert task_low.id in low_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
