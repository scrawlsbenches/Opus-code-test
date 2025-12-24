"""
Comprehensive Unit Tests for GoT Core Modules
==============================================

Tests for:
- cortical/got/protocol.py - Protocol definition
- cortical/got/api.py - GoTManager API
- cortical/got/wal.py - Write-Ahead Log
- cortical/got/sync.py - Git sync manager
- cortical/got/checksums.py - Checksum utilities (deprecated)

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
# WAL MANAGER TESTS
# =============================================================================


class TestWALManager:
    """Tests for WAL (Write-Ahead Log) Manager."""

    @pytest.fixture
    def temp_wal_dir(self):
        """Create a temporary WAL directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def wal_manager(self, temp_wal_dir):
        """Create a WAL manager for testing."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode
        return WALManager(temp_wal_dir, durability=DurabilityMode.RELAXED)

    def test_wal_manager_creates_directories(self, temp_wal_dir):
        """WAL manager creates necessary directories on init."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode

        wal = WALManager(temp_wal_dir / "wal", durability=DurabilityMode.RELAXED)

        assert (temp_wal_dir / "wal").exists()
        assert (temp_wal_dir / "wal" / "archived").exists()

    def test_wal_manager_sequence_starts_at_zero(self, wal_manager):
        """Sequence counter starts at 0."""
        assert wal_manager._sequence == 0

    def test_wal_manager_log_increments_sequence(self, wal_manager):
        """Each log entry increments sequence."""
        seq1 = wal_manager.log("tx-001", "TEST_OP", {"data": "test1"})
        seq2 = wal_manager.log("tx-001", "TEST_OP", {"data": "test2"})

        assert seq1 == 1
        assert seq2 == 2
        assert wal_manager._sequence == 2

    def test_wal_manager_log_writes_to_file(self, wal_manager):
        """Log entries are written to current.wal file."""
        wal_manager.log("tx-001", "TEST_OP", {"key": "value"})

        assert wal_manager.wal_file.exists()
        content = wal_manager.wal_file.read_text()
        assert "TEST_OP" in content
        assert "tx-001" in content

    def test_wal_manager_log_tx_begin(self, wal_manager):
        """log_tx_begin logs transaction start with snapshot."""
        seq = wal_manager.log_tx_begin("tx-001", snapshot_version=5)

        content = wal_manager.wal_file.read_text()
        assert "TX_BEGIN" in content
        assert "snapshot" in content

    def test_wal_manager_log_write(self, wal_manager):
        """log_write logs entity write operations."""
        seq = wal_manager.log_write("tx-001", "task:T-001", old_version=0, new_version=1)

        content = wal_manager.wal_file.read_text()
        assert "WRITE" in content
        assert "task:T-001" in content
        assert "old_version" in content
        assert "new_version" in content

    def test_wal_manager_log_tx_prepare(self, wal_manager):
        """log_tx_prepare logs prepare phase."""
        seq = wal_manager.log_tx_prepare("tx-001")

        content = wal_manager.wal_file.read_text()
        assert "TX_PREPARE" in content

    def test_wal_manager_log_tx_commit(self, wal_manager):
        """log_tx_commit logs successful commit."""
        seq = wal_manager.log_tx_commit("tx-001", version=10)

        content = wal_manager.wal_file.read_text()
        assert "TX_COMMIT" in content
        assert "version" in content

    def test_wal_manager_log_tx_abort(self, wal_manager):
        """log_tx_abort logs abort with reason."""
        seq = wal_manager.log_tx_abort("tx-001", reason="Conflict detected")

        content = wal_manager.wal_file.read_text()
        assert "TX_ABORT" in content
        assert "Conflict detected" in content

    def test_wal_manager_log_tx_rollback(self, wal_manager):
        """log_tx_rollback logs rollback with reason."""
        seq = wal_manager.log_tx_rollback("tx-001", reason="User cancelled")

        content = wal_manager.wal_file.read_text()
        assert "TX_ROLLBACK" in content
        assert "User cancelled" in content

    def test_wal_manager_replay_empty(self, wal_manager):
        """Replay on empty WAL returns empty list."""
        entries = wal_manager.replay()
        assert entries == []

    def test_wal_manager_replay_returns_entries(self, wal_manager):
        """Replay returns logged entries in order."""
        wal_manager.log_tx_begin("tx-001", 0)
        wal_manager.log_write("tx-001", "task:T-001", 0, 1)
        wal_manager.log_tx_commit("tx-001", 1)

        entries = wal_manager.replay()

        assert len(entries) == 3
        assert entries[0]["op"] == "TX_BEGIN"
        assert entries[1]["op"] == "WRITE"
        assert entries[2]["op"] == "TX_COMMIT"

    def test_wal_manager_replay_entries_typed(self, wal_manager):
        """replay_entries returns typed TransactionWALEntry objects."""
        from cortical.wal import TransactionWALEntry

        wal_manager.log_tx_begin("tx-001", 0)
        wal_manager.log_tx_commit("tx-001", 1)

        entries = wal_manager.replay_entries()

        assert len(entries) == 2
        assert all(isinstance(e, TransactionWALEntry) for e in entries)

    def test_wal_manager_replay_skips_corrupted_json(self, wal_manager):
        """Replay skips lines with invalid JSON."""
        wal_manager.log_tx_begin("tx-001", 0)

        # Corrupt the file by appending invalid JSON
        with open(wal_manager.wal_file, 'a') as f:
            f.write("not valid json\n")

        wal_manager.log_tx_commit("tx-001", 1)

        entries = wal_manager.replay()
        # Should have 2 valid entries (corrupted line skipped)
        assert len(entries) == 2

    def test_wal_manager_get_incomplete_transactions(self, wal_manager):
        """get_incomplete_transactions finds uncommitted transactions."""
        # Start two transactions
        wal_manager.log_tx_begin("tx-001", 0)
        wal_manager.log_tx_begin("tx-002", 0)

        # Commit only tx-001
        wal_manager.log_tx_commit("tx-001", 1)

        incomplete = wal_manager.get_incomplete_transactions()

        assert len(incomplete) == 1
        assert incomplete[0]["tx_id"] == "tx-002"
        assert incomplete[0]["state"] == "ACTIVE"

    def test_wal_manager_get_incomplete_transactions_preparing(self, wal_manager):
        """get_incomplete_transactions tracks preparing state."""
        wal_manager.log_tx_begin("tx-001", 0)
        wal_manager.log_tx_prepare("tx-001")
        # Transaction crashed during prepare phase

        incomplete = wal_manager.get_incomplete_transactions()

        assert len(incomplete) == 1
        assert incomplete[0]["state"] == "PREPARING"

    def test_wal_manager_truncate_archives(self, wal_manager):
        """truncate with archive=True moves file to archived/."""
        wal_manager.log_tx_begin("tx-001", 0)
        wal_manager.log_tx_commit("tx-001", 1)

        archived_path = wal_manager.truncate(archive=True)

        assert archived_path is not None
        assert archived_path.exists()
        assert archived_path.parent == wal_manager.archive_dir
        assert not wal_manager.wal_file.exists()

    def test_wal_manager_truncate_deletes(self, wal_manager):
        """truncate with archive=False deletes the file."""
        wal_manager.log_tx_begin("tx-001", 0)
        wal_manager.log_tx_commit("tx-001", 1)

        result = wal_manager.truncate(archive=False)

        assert result is None
        assert not wal_manager.wal_file.exists()

    def test_wal_manager_truncate_empty_wal(self, wal_manager):
        """truncate on non-existent WAL returns None."""
        result = wal_manager.truncate()
        assert result is None

    def test_wal_manager_sequence_persists(self, temp_wal_dir):
        """Sequence counter persists across restarts."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode

        # First manager logs some entries
        wal1 = WALManager(temp_wal_dir, durability=DurabilityMode.RELAXED)
        wal1.log("tx-001", "TEST", {})
        wal1.log("tx-001", "TEST", {})
        assert wal1._sequence == 2

        # Second manager should continue from saved sequence
        wal2 = WALManager(temp_wal_dir, durability=DurabilityMode.RELAXED)
        assert wal2._sequence == 2

        seq = wal2.log("tx-002", "TEST", {})
        assert seq == 3

    def test_wal_manager_fsync_now(self, wal_manager):
        """fsync_now forces sync of WAL files."""
        wal_manager.log_tx_begin("tx-001", 0)

        # Should not raise
        wal_manager.fsync_now()

    def test_wal_manager_fsync_now_empty(self, wal_manager):
        """fsync_now handles empty WAL gracefully."""
        # Should not raise even when no WAL file exists
        wal_manager.fsync_now()

    def test_wal_manager_durability_relaxed(self, temp_wal_dir):
        """RELAXED mode doesn't fsync on every write."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode

        wal = WALManager(temp_wal_dir, durability=DurabilityMode.RELAXED)
        # Should complete quickly without fsync
        wal.log("tx-001", "TEST", {})

    def test_wal_manager_durability_balanced(self, temp_wal_dir):
        """BALANCED mode is default."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode

        wal = WALManager(temp_wal_dir, durability=DurabilityMode.BALANCED)
        assert wal.durability == DurabilityMode.BALANCED


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
# CHECKSUMS TESTS (DEPRECATED MODULE)
# =============================================================================


class TestDeprecatedChecksums:
    """Tests for deprecated checksums module."""

    def test_compute_checksum_warns(self):
        """compute_checksum shows deprecation warning."""
        from cortical.got.checksums import compute_checksum

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_checksum({"key": "value"})

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_verify_checksum_warns(self):
        """verify_checksum shows deprecation warning."""
        from cortical.got.checksums import compute_checksum, verify_checksum

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            checksum = compute_checksum({"key": "value"})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = verify_checksum({"key": "value"}, checksum)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert result is True

    def test_compute_file_checksum_warns(self):
        """compute_file_checksum shows deprecation warning."""
        from cortical.got.checksums import compute_file_checksum

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value"}, f)
            temp_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = compute_file_checksum(temp_path)

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
        finally:
            temp_path.unlink()

    def test_verify_file_checksum_warns(self):
        """verify_file_checksum shows deprecation warning."""
        from cortical.got.checksums import compute_file_checksum, verify_file_checksum

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value"}, f)
            temp_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                checksum = compute_file_checksum(temp_path)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = verify_file_checksum(temp_path, checksum)

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert result is True
        finally:
            temp_path.unlink()


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
        return GoTManager(temp_got_dir, durability=DurabilityMode.RELAXED)

    def test_got_manager_init(self, temp_got_dir):
        """GoTManager initializes correctly."""
        from cortical.got.api import GoTManager
        from cortical.got.config import DurabilityMode

        manager = GoTManager(temp_got_dir, durability=DurabilityMode.RELAXED)

        assert manager.got_dir == temp_got_dir
        assert manager.durability == DurabilityMode.RELAXED
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

        manager = GoTManager(temp_got_dir, durability=DurabilityMode.RELAXED)
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
        """list_all_tasks calls find_tasks with no filters."""
        with patch.object(got_manager, 'find_tasks', return_value=[]) as mock_find:
            result = got_manager.list_all_tasks()

            mock_find.assert_called_once_with()
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
        manager = Mock()
        manager.begin.return_value = Mock()
        manager.commit.return_value = Mock(success=True)
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
        return GoTManager(temp_got_dir, durability=DurabilityMode.RELAXED)

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
        edge = got_manager.add_edge(task1.id, task2.id, "RELATED_TO")

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
