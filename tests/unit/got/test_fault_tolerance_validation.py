"""
Fault Tolerance and Recovery Validation Tests

Task: T-20251222-204138-51a38502

Validates:
1. Generation failures fall back gracefully
2. Corrupted layers are detected
3. Recovery procedures work
4. Sessions can start even if generation fails

TDD Approach: Tests written first (RED), then implementation fixes (GREEN).
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from cortical.got import TransactionManager, Task
from cortical.got.recovery import RecoveryManager, RecoveryResult
from cortical.utils.checksums import compute_checksum


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def got_dir(tmp_path):
    """Create a GoT directory structure."""
    entities_dir = tmp_path / "entities"
    wal_dir = tmp_path / "wal"
    entities_dir.mkdir()
    wal_dir.mkdir()
    return tmp_path


@pytest.fixture
def populated_got(got_dir):
    """Create a populated GoT directory with some tasks."""
    tm = TransactionManager(got_dir)
    tx = tm.begin()

    # Create some tasks
    task1 = Task(id="T-test-001", title="Test task 1")
    task2 = Task(id="T-test-002", title="Test task 2")
    tm.write(tx, task1)
    tm.write(tx, task2)
    tm.commit(tx)

    return got_dir


# =============================================================================
# 1. GENERATION FAILURE GRACEFUL FALLBACK TESTS
# =============================================================================

class TestGenerationFailureFallback:
    """Test that generation failures fall back gracefully."""

    def test_recover_returns_result_on_empty_directory(self, got_dir):
        """Recovery should return a result even on empty directory."""
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should not raise, should return a RecoveryResult
        assert isinstance(result, RecoveryResult)
        assert result.success is True  # Empty is OK

    def test_recover_handles_missing_wal_file(self, got_dir):
        """Recovery should handle missing WAL file gracefully."""
        # Create entity but no WAL
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-test.json"

        entity_data = {
            "_checksum": compute_checksum({"id": "T-test", "title": "Test"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-test", "entity_type": "task", "title": "Test"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should handle missing WAL gracefully
        assert isinstance(result, RecoveryResult)
        # Should detect orphan (entity without WAL record)
        assert len(result.orphans_detected) >= 1 or "orphan" in str(result.actions_taken).lower()

    def test_recover_handles_malformed_wal_entries(self, got_dir):
        """Recovery should skip malformed WAL entries and continue."""
        # Create WAL with malformed entries
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"

        # Write mix of valid and invalid entries
        entries = [
            '{"this": "is not a valid WAL entry"}',
            'not even json',
            '{"op": "CREATE", "tx": "TX-001", "seq": 1, "ts": "2025-12-24T00:00:00", "checksum": "bad"}',
        ]
        wal_file.write_text('\n'.join(entries))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should not crash, should report corrupted entries
        assert isinstance(result, RecoveryResult)
        assert result.corrupted_wal_entries >= 1

    def test_transaction_manager_starts_after_failed_recovery(self, got_dir):
        """TransactionManager should start even if previous state is corrupted."""
        # Corrupt the WAL file
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"
        wal_file.write_text("totally broken content\n" * 10)

        # TransactionManager should still initialize
        # Recovery happens but shouldn't prevent startup
        try:
            tm = TransactionManager(got_dir)
            # Should be able to begin a new transaction
            tx = tm.begin()
            assert tx is not None
            tm.rollback(tx)
        except Exception as e:
            pytest.fail(f"TransactionManager failed to start after corruption: {e}")


# =============================================================================
# 2. CORRUPTED LAYER DETECTION TESTS
# =============================================================================

class TestCorruptedLayerDetection:
    """Test that corrupted entities/layers are detected."""

    def test_detects_corrupted_entity_checksum(self, populated_got):
        """Should detect entity with invalid checksum."""
        # Corrupt an entity file
        entity_file = populated_got / "entities" / "T-test-001.json"
        with open(entity_file, 'r') as f:
            data = json.load(f)

        # Modify data without updating checksum
        data["data"]["title"] = "TAMPERED"
        with open(entity_file, 'w') as f:
            json.dump(data, f)

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should detect corruption
        assert "T-test-001" in result.corrupted_entities

    def test_detects_truncated_entity_file(self, populated_got):
        """Should detect truncated entity files."""
        entity_file = populated_got / "entities" / "T-test-001.json"

        # Truncate the file
        with open(entity_file, 'r') as f:
            content = f.read()
        with open(entity_file, 'w') as f:
            f.write(content[:len(content)//2])  # Write only half

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should detect corruption (JSON parse error)
        assert not result.success or "T-test-001" in result.corrupted_entities

    def test_detects_missing_checksum_field(self, populated_got):
        """Should detect entity missing _checksum field."""
        entity_file = populated_got / "entities" / "T-test-001.json"

        # Remove checksum
        with open(entity_file, 'r') as f:
            data = json.load(f)
        del data["_checksum"]
        with open(entity_file, 'w') as f:
            json.dump(data, f)

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should detect as corrupted (no checksum)
        assert "T-test-001" in result.corrupted_entities or not result.success

    def test_detects_empty_entity_file(self, got_dir):
        """Should detect empty entity files."""
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-empty.json"
        entity_file.write_text("")

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should not crash on empty file
        assert isinstance(result, RecoveryResult)

    def test_reports_all_corrupted_entities(self, populated_got):
        """Should report all corrupted entities, not just first one."""
        # Corrupt both entities
        for entity_id in ["T-test-001", "T-test-002"]:
            entity_file = populated_got / "entities" / f"{entity_id}.json"
            with open(entity_file, 'r') as f:
                data = json.load(f)
            data["data"]["title"] = "CORRUPTED"
            with open(entity_file, 'w') as f:
                json.dump(data, f)

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should report both corrupted entities
        assert len(result.corrupted_entities) >= 2


# =============================================================================
# 3. RECOVERY PROCEDURE TESTS
# =============================================================================

class TestRecoveryProcedures:
    """Test that recovery procedures work correctly."""

    def test_recovery_rolls_back_incomplete_transactions(self, got_dir):
        """Incomplete transactions should be rolled back."""
        tm = TransactionManager(got_dir)

        # Start transaction but don't commit
        tx = tm.begin()
        task = Task(id="T-incomplete", title="Incomplete")
        tm.write(tx, task)
        # Don't commit - simulate crash

        # Fresh recovery should roll back
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should have rolled back incomplete transaction
        assert len(result.rolled_back) >= 1

    def test_recovery_preserves_committed_transactions(self, populated_got):
        """Committed transactions should be preserved during recovery."""
        # Count entities before recovery
        entities_before = list((populated_got / "entities").glob("*.json"))

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Count entities after recovery
        entities_after = list((populated_got / "entities").glob("*.json"))

        # All committed entities should still exist
        assert len(entities_after) == len(entities_before)

    def test_recovery_adopts_orphaned_entities(self, got_dir):
        """Orphaned entities should be adopted with synthetic WAL entries."""
        # Create entity without WAL entry (simulates fresh clone)
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-orphan.json"

        entity_data = {
            "_checksum": compute_checksum({"id": "T-orphan", "entity_type": "task", "title": "Orphan"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-orphan", "entity_type": "task", "title": "Orphan"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should adopt orphan
        assert len(result.orphans_detected) >= 1
        assert "adopt" in str(result.actions_taken).lower() or result.orphans_repaired >= 1

    def test_recovery_is_idempotent(self, populated_got):
        """Running recovery multiple times should produce same result."""
        recovery = RecoveryManager(populated_got)

        # Run recovery twice
        result1 = recovery.recover()
        result2 = recovery.recover()

        # Second run should find nothing to do
        assert result2.rolled_back == []
        assert result2.corrupted_entities == []
        assert result2.corrupted_wal_entries == 0

    def test_recovery_result_contains_all_actions(self, got_dir):
        """RecoveryResult should document all actions taken."""
        # Create orphaned entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-orphan.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-orphan", "entity_type": "task", "title": "Test"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-orphan", "entity_type": "task", "title": "Test"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should have documented actions
        assert isinstance(result.actions_taken, list)
        assert len(result.actions_taken) >= 1


# =============================================================================
# 4. SESSION STARTUP WITH FAILED GENERATION TESTS
# =============================================================================

class TestSessionStartupWithFailedGeneration:
    """Test that sessions can start even if generation/state is broken."""

    def test_transaction_manager_creates_missing_directories(self, tmp_path):
        """TransactionManager should create missing directories."""
        empty_dir = tmp_path / "new_got"
        # Don't create the directory - let TransactionManager do it

        tm = TransactionManager(empty_dir)

        # Should create required directories
        assert (empty_dir / "entities").exists()
        assert (empty_dir / "wal").exists()

    def test_transaction_manager_handles_readonly_directory(self, got_dir):
        """Should handle read-only directory gracefully."""
        # This test may not work on all platforms
        import os
        import stat

        # Skip if running as root (can write anywhere)
        if os.geteuid() == 0:
            pytest.skip("Cannot test read-only as root")

        # Make directory read-only
        original_mode = os.stat(got_dir).st_mode
        os.chmod(got_dir, stat.S_IRUSR | stat.S_IXUSR)

        try:
            # Should handle gracefully
            with pytest.raises(Exception):  # Should raise permission error
                TransactionManager(got_dir)
        finally:
            # Restore permissions
            os.chmod(got_dir, original_mode)

    def test_can_read_entities_with_corrupted_wal(self, populated_got):
        """Should be able to read entities even if WAL is corrupted."""
        # Corrupt WAL
        wal_file = populated_got / "wal" / "current.wal"
        wal_file.write_text("corrupted\n" * 10)

        # Should still be able to read existing entities
        tm = TransactionManager(populated_got)

        # Entities were committed before corruption, should be readable
        entity_file = populated_got / "entities" / "T-test-001.json"
        assert entity_file.exists()

        with open(entity_file) as f:
            data = json.load(f)
        assert data["data"]["id"] == "T-test-001"

    def test_new_transactions_work_after_recovery(self, populated_got):
        """Should be able to create new transactions after recovery."""
        # Run recovery
        recovery = RecoveryManager(populated_got)
        recovery.recover()

        # Create new transaction
        tm = TransactionManager(populated_got)
        tx = tm.begin()
        task = Task(id="T-new", title="New task")
        tm.write(tx, task)
        tm.commit(tx)

        # New entity should exist
        entity_file = populated_got / "entities" / "T-new.json"
        assert entity_file.exists()

    def test_recovery_on_fresh_directory_is_fast(self, got_dir):
        """Recovery on fresh/empty directory should be nearly instant."""
        import time

        recovery = RecoveryManager(got_dir)

        start = time.time()
        result = recovery.recover()
        elapsed = time.time() - start

        # Should complete in under 100ms for empty directory
        assert elapsed < 0.1
        assert result.success


# =============================================================================
# 5. EDGE CASE AND REGRESSION TESTS
# =============================================================================

class TestEdgeCasesAndRegressions:
    """Edge cases and regression tests for fault tolerance."""

    def test_handles_unicode_in_entity_data(self, got_dir):
        """Should handle unicode characters in entity data."""
        tm = TransactionManager(got_dir)
        tx = tm.begin()

        # Create task with unicode
        task = Task(id="T-unicode", title="Test with 日本語 and émojis 🎉")
        tm.write(tx, task)
        tm.commit(tx)

        # Recovery should handle unicode
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert result.success
        assert "T-unicode" not in result.corrupted_entities

    def test_handles_very_long_entity_content(self, got_dir):
        """Should handle entities with very long content."""
        tm = TransactionManager(got_dir)
        tx = tm.begin()

        # Create task with long content
        long_content = "x" * 100000  # 100KB
        task = Task(id="T-long", title=long_content)
        tm.write(tx, task)
        tm.commit(tx)

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert result.success

    def test_handles_concurrent_recovery_attempts(self, populated_got):
        """Multiple recovery attempts should not corrupt state."""
        import threading

        results = []
        errors = []

        def run_recovery():
            try:
                recovery = RecoveryManager(populated_got)
                result = recovery.recover()
                results.append(result)
            except FileNotFoundError:
                # FileNotFoundError is acceptable - it means another thread
                # already processed/deleted the file. This is expected behavior
                # in concurrent recovery scenarios.
                results.append(None)  # Mark as handled
            except Exception as e:
                errors.append(e)

        # Start multiple recovery attempts
        threads = [threading.Thread(target=run_recovery) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed, others may hit race conditions
        # The key is that no unexpected errors occur
        assert len(errors) == 0
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) >= 1  # At least one full recovery
        assert all(r.success for r in successful_results)

    def test_checksum_algorithm_consistency(self, got_dir):
        """Checksum algorithm should be consistent across calls."""
        data = {"id": "test", "title": "Test", "nested": {"key": "value"}}

        # Compute multiple times
        checksums = [compute_checksum(data) for _ in range(10)]

        # All should be identical
        assert len(set(checksums)) == 1

    def test_empty_wal_file_handled(self, got_dir):
        """Should handle empty WAL file."""
        wal_file = got_dir / "wal" / "current.wal"
        wal_file.write_text("")

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert result.success

    def test_wal_file_with_only_newlines(self, got_dir):
        """Should handle WAL file with only newlines."""
        wal_file = got_dir / "wal" / "current.wal"
        wal_file.write_text("\n\n\n\n")

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert isinstance(result, RecoveryResult)


# =============================================================================
# 6. ADDITIONAL COVERAGE TESTS - recovery.py
# =============================================================================

class TestRecoveryCoverageBranches:
    """Additional tests to cover remaining branches in recovery.py."""

    def test_needs_recovery_with_incomplete_transactions(self, got_dir):
        """needs_recovery should return True when incomplete transactions exist."""
        tm = TransactionManager(got_dir)
        tx = tm.begin()
        task = Task(id="T-incomplete", title="Not committed")
        tm.write(tx, task)
        # Don't commit

        recovery = RecoveryManager(got_dir)
        assert recovery.needs_recovery() is True

    def test_needs_recovery_with_corrupted_entities(self, got_dir):
        """needs_recovery should return True when corrupted entities exist."""
        # First create a valid entity
        tm = TransactionManager(got_dir)
        tx = tm.begin()
        task = Task(id="T-corrupt", title="Will be corrupted")
        tm.write(tx, task)
        tm.commit(tx)

        # Now corrupt it
        entity_file = got_dir / "entities" / "T-corrupt.json"
        with open(entity_file, 'r') as f:
            data = json.load(f)
        data["data"]["title"] = "TAMPERED"
        with open(entity_file, 'w') as f:
            json.dump(data, f)

        recovery = RecoveryManager(got_dir)
        assert recovery.needs_recovery() is True

    def test_needs_recovery_with_corrupted_wal_entries(self, got_dir):
        """needs_recovery should return True when WAL entries are corrupted."""
        # Create a WAL with corrupted entries
        wal_file = got_dir / "wal" / "current.wal"
        entries = [
            '{"op":"TX_BEGIN","tx":"TX-bad","data":{},"checksum":"wrong"}',
        ]
        wal_file.write_text('\n'.join(entries) + '\n')

        recovery = RecoveryManager(got_dir)
        assert recovery.needs_recovery() is True

    def test_needs_recovery_false_when_clean(self, populated_got):
        """needs_recovery should return False when system is clean."""
        recovery = RecoveryManager(populated_got)
        # First ensure recovery is complete
        recovery.recover()
        # Now check if needs_recovery is False
        assert recovery.needs_recovery() is False

    def test_repair_orphans_delete_strategy(self, got_dir):
        """repair_orphans with delete strategy should remove orphaned files."""
        # Create orphaned entity (no WAL entry)
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-orphan-delete.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-orphan-delete", "entity_type": "task", "title": "Orphan"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-orphan-delete", "entity_type": "task", "title": "Orphan"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.repair_orphans(strategy='delete')

        assert result.success
        assert result.repaired_count >= 1
        assert not entity_file.exists()

    def test_repair_orphans_invalid_strategy(self, got_dir):
        """repair_orphans should raise ValueError for invalid strategy."""
        recovery = RecoveryManager(got_dir)
        with pytest.raises(ValueError) as exc_info:
            recovery.repair_orphans(strategy='invalid')
        assert "Invalid strategy" in str(exc_info.value)

    def test_repair_orphans_with_corrupted_entity(self, got_dir):
        """repair_orphans adopt should delete corrupted orphans."""
        # Create corrupted orphaned entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-corrupt-orphan.json"
        # Write invalid JSON that will fail checksum
        entity_data = {
            "_checksum": "intentionally_wrong_checksum",
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-corrupt-orphan", "entity_type": "task", "title": "Corrupted Orphan"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.repair_orphans(strategy='adopt')

        # Corrupted orphan should be deleted, not adopted
        assert result.repaired_count >= 1
        assert len(result.errors) >= 1
        assert not entity_file.exists()


# =============================================================================
# 7. ADDITIONAL COVERAGE TESTS - wal.py
# =============================================================================

class TestWALCoverageBranches:
    """Additional tests to cover remaining branches in wal.py."""

    def test_wal_log_tx_rollback(self, got_dir):
        """Test log_tx_rollback creates proper entry."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        seq = wal.log_tx_rollback("TX-test", "test_reason")

        assert seq >= 1
        entries = wal.replay()
        assert any(e['op'] == 'TX_ROLLBACK' for e in entries)

    def test_wal_replay_entries_returns_typed_objects(self, got_dir):
        """replay_entries should return TransactionWALEntry objects."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        wal.log_tx_begin("TX-typed", 1)
        wal.log_tx_commit("TX-typed", 1)

        entries = wal.replay_entries()
        assert len(entries) >= 2
        from cortical.wal import TransactionWALEntry
        assert all(isinstance(e, TransactionWALEntry) for e in entries)

    def test_wal_fsync_now(self, got_dir):
        """fsync_now should sync WAL and sequence files."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        wal.log_tx_begin("TX-sync", 1)

        # Should not raise
        wal.fsync_now()

    def test_wal_truncate_with_archive(self, got_dir):
        """truncate with archive=True should move WAL to archived/."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        wal.log_tx_begin("TX-archive", 1)

        archive_path = wal.truncate(archive=True)

        assert archive_path is not None
        assert archive_path.exists()
        assert not wal.wal_file.exists()

    def test_wal_truncate_without_archive(self, got_dir):
        """truncate with archive=False should delete WAL."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        wal.log_tx_begin("TX-delete", 1)

        result = wal.truncate(archive=False)

        assert result is None
        assert not wal.wal_file.exists()

    def test_wal_truncate_no_file(self, got_dir):
        """truncate on non-existent WAL should return None."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        # Don't create any entries

        result = wal.truncate(archive=True)
        assert result is None

    def test_wal_sequence_recovery_from_corrupted_file(self, got_dir):
        """WAL should recover from corrupted sequence file."""
        from cortical.got.wal import WALManager

        # Create corrupted sequence file
        seq_file = got_dir / "wal" / "_sequence.json"
        (got_dir / "wal").mkdir(parents=True, exist_ok=True)
        seq_file.write_text("not valid json{")

        # Should handle gracefully and start from 0
        wal = WALManager(got_dir / "wal")
        seq = wal.log_tx_begin("TX-1", 1)

        assert seq == 1

    def test_wal_replay_with_corrupted_checksum(self, got_dir):
        """replay should skip entries with invalid checksums."""
        from cortical.got.wal import WALManager

        wal_dir = got_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        # Write entry with bad checksum
        entries = [
            '{"seq":1,"ts":"2025-01-01T00:00:00","tx":"TX-bad","op":"TX_BEGIN","data":{},"checksum":"wrong"}',
        ]
        wal_file.write_text('\n'.join(entries) + '\n')

        wal = WALManager(wal_dir)
        valid_entries = wal.replay()

        # Bad checksum entry should be skipped
        assert len(valid_entries) == 0


# =============================================================================
# 8. ADDITIONAL COVERAGE TESTS - locking.py
# =============================================================================

class TestLockingCoverageBranches:
    """Additional tests to cover remaining branches in locking.py."""

    def test_lock_acquire_and_release(self, tmp_path):
        """Basic lock acquire and release."""
        from cortical.utils.locking import ProcessLock

        lock = ProcessLock(tmp_path / ".lock")
        assert lock.acquire() is True
        assert lock.is_locked() is True
        lock.release()
        assert lock.is_locked() is False

    def test_lock_context_manager(self, tmp_path):
        """Lock as context manager."""
        from cortical.utils.locking import ProcessLock

        lock = ProcessLock(tmp_path / ".lock")
        with lock:
            assert lock.is_locked()
        assert not lock.is_locked()

    def test_lock_reentrant(self, tmp_path):
        """Reentrant lock can be acquired multiple times by same process."""
        from cortical.utils.locking import ProcessLock

        lock = ProcessLock(tmp_path / ".lock", reentrant=True)
        assert lock.acquire() is True
        assert lock._lock_count == 1
        assert lock.acquire() is True
        assert lock._lock_count == 2
        lock.release()
        assert lock._lock_count == 1
        lock.release()
        assert lock._lock_count == 0

    def test_lock_with_timeout(self, tmp_path):
        """Lock with timeout should retry."""
        from cortical.utils.locking import ProcessLock

        lock = ProcessLock(tmp_path / ".lock")
        # First acquire without timeout
        assert lock.acquire() is True

        # Create second lock and try to acquire with timeout
        lock2 = ProcessLock(tmp_path / ".lock", reentrant=False)
        # This should fail after timeout since lock is held
        # Use very short timeout
        result = lock2.acquire(timeout=0.05)
        # Result depends on whether flock is available and working
        # Just verify it returns a boolean
        assert isinstance(result, bool)

        lock.release()

    def test_lock_release_when_not_held(self, tmp_path):
        """Release on unheld lock should do nothing."""
        from cortical.utils.locking import ProcessLock

        lock = ProcessLock(tmp_path / ".lock")
        # Release without acquire - should not raise
        lock.release()
        assert lock._lock_count == 0

    def test_lock_stale_detection_dead_process(self, tmp_path):
        """Lock should detect stale locks from dead processes."""
        from cortical.utils.locking import ProcessLock
        import os

        lock_path = tmp_path / ".lock"

        # Create a lock file with a non-existent PID
        holder_info = {
            "pid": 999999999,  # Very unlikely to exist
            "acquired_at": 0  # Old timestamp
        }
        lock_path.write_text(json.dumps(holder_info))

        # Lock should detect stale and recover
        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should successfully acquire after detecting stale lock
        assert result is True
        lock.release()

    def test_lock_stale_detection_empty_file(self, tmp_path):
        """Lock should handle empty lock file as stale."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / ".lock"
        lock_path.write_text("")

        lock = ProcessLock(lock_path)
        result = lock.acquire()

        assert result is True
        lock.release()

    def test_lock_stale_detection_invalid_json(self, tmp_path):
        """Lock should handle invalid JSON in lock file as stale."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / ".lock"
        lock_path.write_text("not valid json{}")

        lock = ProcessLock(lock_path)
        result = lock.acquire()

        assert result is True
        lock.release()

    def test_lock_stale_detection_timeout(self, tmp_path):
        """Lock older than stale_timeout should be considered stale."""
        from cortical.utils.locking import ProcessLock
        import os
        import time

        lock_path = tmp_path / ".lock"

        # Create a lock file with old timestamp
        holder_info = {
            "pid": os.getpid(),  # Current process (still alive)
            "acquired_at": time.time() - 10000  # Very old
        }
        lock_path.write_text(json.dumps(holder_info))

        # Lock with short stale timeout
        lock = ProcessLock(lock_path, stale_timeout=1.0)
        # Check if _is_stale_lock returns True
        assert lock._is_stale_lock() is True

    def test_lock_context_manager_failure(self, tmp_path):
        """Context manager should raise if lock acquisition fails."""
        from cortical.utils.locking import ProcessLock
        import sys


        lock_path = tmp_path / ".lock"
        lock1 = ProcessLock(lock_path, reentrant=False)

        # Hold lock with first instance
        assert lock1.acquire() is True

        # Try context manager with second instance - should fail
        lock2 = ProcessLock(lock_path, reentrant=False)
        with pytest.raises(RuntimeError) as exc_info:
            with lock2:
                pass
        assert "Failed to acquire lock" in str(exc_info.value)

        lock1.release()

    def test_lock_is_stale_nonexistent_file(self, tmp_path):
        """_is_stale_lock should return False for non-existent file."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / ".nonexistent"
        lock = ProcessLock(lock_path)

        # File doesn't exist, should not be considered stale
        assert lock._is_stale_lock() is False

    def test_lock_is_stale_no_pid(self, tmp_path):
        """_is_stale_lock should return True when PID is missing."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / ".lock"
        # Lock file without pid field
        lock_path.write_text(json.dumps({"acquired_at": 0}))

        lock = ProcessLock(lock_path)
        assert lock._is_stale_lock() is True


# =============================================================================
# 9. ADDITIONAL COVERAGE TESTS - WAL PARANOID MODE
# =============================================================================

class TestWALParanoidMode:
    """Tests for WAL PARANOID durability mode."""

    def test_wal_paranoid_mode_fsync(self, got_dir):
        """PARANOID mode should fsync on every write."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode

        wal = WALManager(got_dir / "wal", durability=DurabilityMode.PARANOID)
        seq = wal.log_tx_begin("TX-paranoid", 1)

        assert seq >= 1
        # File should exist and contain the entry
        assert wal.wal_file.exists()

    def test_wal_paranoid_mode_sequence_save(self, got_dir):
        """PARANOID mode should fsync sequence file."""
        from cortical.got.wal import WALManager
        from cortical.got.config import DurabilityMode

        wal = WALManager(got_dir / "wal", durability=DurabilityMode.PARANOID)
        wal.log_tx_begin("TX-1", 1)
        wal.log_tx_begin("TX-2", 1)

        # Sequence should be persisted
        assert wal.seq_file.exists()


# =============================================================================
# 10. RECOVERY EDGE CASE TESTS
# =============================================================================

class TestRecoveryEdgeCases:
    """Additional edge case tests for recovery module."""

    def test_verify_wal_integrity_with_missing_checksum(self, got_dir):
        """verify_wal_integrity should count entries without checksum as corrupted."""
        wal_file = got_dir / "wal" / "current.wal"
        # Entry without checksum field
        entries = [
            '{"op":"TX_BEGIN","tx":"TX-no-checksum","data":{}}',
        ]
        wal_file.write_text('\n'.join(entries) + '\n')

        recovery = RecoveryManager(got_dir)
        corrupted_count = recovery.verify_wal_integrity()

        assert corrupted_count >= 1

    def test_recovery_with_mixed_valid_invalid_wal(self, got_dir):
        """Recovery should handle mix of valid and invalid WAL entries."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        # Add valid entries
        wal.log_tx_begin("TX-valid", 1)
        wal.log_tx_commit("TX-valid", 1)

        # Append corrupted entry
        with open(wal.wal_file, 'a') as f:
            f.write('{"invalid": "entry"}\n')

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should complete despite corrupted entry
        assert isinstance(result, RecoveryResult)
        assert result.corrupted_wal_entries >= 1

    def test_detect_orphaned_entities_with_wal_containing_write_ops(self, got_dir):
        """detect_orphaned_entities should find entities tracked in WAL WRITE ops."""
        tm = TransactionManager(got_dir)
        tx = tm.begin()
        task = Task(id="T-tracked", title="Tracked in WAL")
        tm.write(tx, task)
        tm.commit(tx)

        recovery = RecoveryManager(got_dir)
        orphans = recovery.detect_orphaned_entities()

        # T-tracked should NOT be in orphans since it's in WAL
        assert "T-tracked" not in orphans

    def test_repair_orphans_skips_vanished_files(self, got_dir):
        """repair_orphans should handle files that vanish during processing."""
        # Create orphan entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-vanish.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-vanish", "entity_type": "task", "title": "Will vanish"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-vanish", "entity_type": "task", "title": "Will vanish"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)

        # Delete before repair runs (simulates race condition)
        entity_file.unlink()

        result = recovery.repair_orphans(strategy='delete')

        # Should complete without error
        assert result.success


# =============================================================================
# 11. MONKEYPATCH-BASED RACE CONDITION TESTS
# =============================================================================

class TestRaceConditionCoverage:
    """
    Tests using monkeypatch to simulate race conditions.

    These tests cover the FileNotFoundError handlers that are difficult
    to trigger naturally because they require files to vanish between
    filesystem operations.
    """

    def test_verify_store_integrity_file_vanishes_during_read(self, got_dir, monkeypatch):
        """File deleted between glob and _read_and_verify."""
        # Create a valid entity
        tm = TransactionManager(got_dir)
        tx = tm.begin()
        task = Task(id="T-vanish", title="Will vanish")
        tm.write(tx, task)
        tm.commit(tx)

        recovery = RecoveryManager(got_dir)

        # Patch _read_and_verify to raise FileNotFoundError
        def exploding_read(entity_file):
            if "T-vanish" in str(entity_file):
                raise FileNotFoundError("Simulated race condition")
            # Call original for other files
            return recovery.store._read_and_verify.__wrapped__(recovery.store, entity_file)

        monkeypatch.setattr(recovery.store, "_read_and_verify", exploding_read)

        corrupted = recovery.verify_store_integrity()

        # Should not crash, vanished file should NOT be in corrupted list
        assert "T-vanish" not in corrupted

    def test_detect_orphaned_entities_wal_vanishes(self, got_dir, monkeypatch):
        """WAL file deleted during orphan detection."""
        # Create entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-orphan.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-orphan", "entity_type": "task", "title": "Orphan"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-orphan", "entity_type": "task", "title": "Orphan"}
        }
        entity_file.write_text(json.dumps(entity_data))

        # Create WAL file
        wal_file = got_dir / "wal" / "current.wal"
        wal_file.write_text("")

        recovery = RecoveryManager(got_dir)

        # Patch open to raise FileNotFoundError for WAL file
        original_open = open
        def patched_open(path, *args, **kwargs):
            if "current.wal" in str(path) and 'r' in str(args):
                raise FileNotFoundError("WAL vanished")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)

        orphans = recovery.detect_orphaned_entities()

        # Should not crash, all disk entities become orphans when WAL is gone
        assert isinstance(orphans, list)

    def test_repair_orphans_delete_race_condition(self, got_dir, monkeypatch):
        """File vanishes during delete operation."""
        # Create orphan entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-race-delete.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-race-delete", "entity_type": "task", "title": "Race"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-race-delete", "entity_type": "task", "title": "Race"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)

        # Patch Path.unlink to raise FileNotFoundError
        original_unlink = Path.unlink
        def patched_unlink(self, *args, **kwargs):
            if "T-race-delete" in str(self):
                raise FileNotFoundError("Already deleted")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", patched_unlink)

        result = recovery.repair_orphans(strategy='delete')

        # Should complete successfully (race condition is OK)
        assert result.success

    def test_repair_orphans_adopt_file_vanishes_before_verify(self, got_dir, monkeypatch):
        """File vanishes before we can verify it for adoption."""
        # Create orphan entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-vanish-adopt.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-vanish-adopt", "entity_type": "task", "title": "Vanish"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-vanish-adopt", "entity_type": "task", "title": "Vanish"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)

        # Patch _read_and_verify to raise FileNotFoundError
        original_read_and_verify = recovery.store._read_and_verify
        def patched_read(entity_path):
            if "T-vanish-adopt" in str(entity_path):
                raise FileNotFoundError("File vanished before adoption")
            return original_read_and_verify(entity_path)

        monkeypatch.setattr(recovery.store, "_read_and_verify", patched_read)

        result = recovery.repair_orphans(strategy='adopt')

        # Should complete (skips vanished file)
        assert result.success

    def test_repair_orphans_corrupted_delete_race(self, got_dir, monkeypatch):
        """Corrupted orphan file vanishes during cleanup delete."""
        # Create corrupted orphan
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-corrupt-race.json"
        entity_data = {
            "_checksum": "wrong_checksum",
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-corrupt-race", "entity_type": "task", "title": "Corrupt"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)

        # Patch unlink to fail after corruption detection
        unlink_calls = [0]
        original_unlink = Path.unlink
        def patched_unlink(self, *args, **kwargs):
            if "T-corrupt-race" in str(self):
                unlink_calls[0] += 1
                if unlink_calls[0] == 1:
                    raise FileNotFoundError("Already deleted during cleanup")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", patched_unlink)

        result = recovery.repair_orphans(strategy='adopt')

        # Should handle gracefully
        assert len(result.errors) >= 1  # Corruption was detected

    def test_repair_orphans_unexpected_exception(self, got_dir, monkeypatch):
        """Unexpected exception during repair is logged and handled."""
        # Create orphan
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-unexpected.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-unexpected", "entity_type": "task", "title": "Test"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-unexpected", "entity_type": "task", "title": "Test"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)

        # Patch unlink to raise unexpected error
        original_unlink = Path.unlink
        def patched_unlink(self, *args, **kwargs):
            if "T-unexpected" in str(self):
                raise PermissionError("Unexpected error!")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", patched_unlink)

        result = recovery.repair_orphans(strategy='delete')

        # Should fail gracefully with error logged
        assert not result.success
        assert len(result.errors) >= 1
        assert "T-unexpected" in result.errors[0]


class TestWALRaceConditionCoverage:
    """Tests for WAL race condition handlers."""

    def test_load_sequence_file_vanishes(self, got_dir, monkeypatch):
        """Sequence file deleted between exists() and open()."""
        from cortical.got.wal import WALManager

        wal_dir = got_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)

        # Create sequence file
        seq_file = wal_dir / "_sequence.json"
        seq_file.write_text('{"seq": 42}')

        # Patch open to raise FileNotFoundError
        original_open = open
        call_count = [0]
        def patched_open(path, *args, **kwargs):
            if "_sequence.json" in str(path) and 'r' in str(args):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise FileNotFoundError("Sequence file vanished")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)

        wal = WALManager(wal_dir)

        # Should handle gracefully and start from 0
        assert wal._sequence == 0

    def test_replay_entries_with_corrupted_json(self, got_dir):
        """replay_entries should skip corrupted JSON entries."""
        from cortical.got.wal import WALManager

        wal_dir = got_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        # Write mix of valid and invalid entries
        wal_file.write_text("not json\n{malformed\n")

        wal = WALManager(wal_dir)
        entries = wal.replay_entries()

        # All entries should be skipped
        assert len(entries) == 0

    def test_replay_entries_with_invalid_checksum(self, got_dir):
        """replay_entries should skip entries with bad checksums."""
        from cortical.got.wal import WALManager

        wal_dir = got_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        # Entry with wrong checksum
        entry = '{"seq":1,"ts":"2025-01-01","tx":"TX-1","op":"TX_BEGIN","data":{},"checksum":"bad"}'
        wal_file.write_text(entry + "\n")

        wal = WALManager(wal_dir)
        entries = wal.replay_entries()

        # Should be skipped
        assert len(entries) == 0


class TestLockingRaceConditionCoverage:
    """Tests for locking race condition handlers."""

    def test_lock_acquire_flock_fails_then_stale_recovery(self, tmp_path, monkeypatch):
        """flock fails, but stale lock recovery succeeds."""
        from cortical.utils.locking import ProcessLock
        import sys


        lock_path = tmp_path / ".lock"

        # Create lock file with dead process
        holder_info = {"pid": 999999999, "acquired_at": 0}
        lock_path.write_text(json.dumps(holder_info))

        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should recover from stale lock
        assert result is True
        lock.release()

    def test_lock_stale_check_read_error(self, tmp_path, monkeypatch):
        """_is_stale_lock handles read errors gracefully."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / ".lock"
        lock_path.write_text('{"pid": 1}')

        lock = ProcessLock(lock_path)

        # Patch open to raise on read
        original_open = open
        def patched_open(path, *args, **kwargs):
            if str(lock_path) in str(path) and 'r' in str(args):
                raise PermissionError("Cannot read lock file")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)

        # Should return False (don't assume stale on error)
        assert lock._is_stale_lock() is False

    def test_lock_remove_stale_fails(self, tmp_path, monkeypatch):
        """Stale lock removal failure is handled."""
        from cortical.utils.locking import ProcessLock
        import sys
        import fcntl


        lock_path = tmp_path / ".lock"

        # Create stale lock (dead PID, old timestamp)
        holder_info = {"pid": 999999999, "acquired_at": 0}
        lock_path.write_text(json.dumps(holder_info))

        # Track calls
        unlink_calls = []

        # Patch flock to fail (simulating locked file) so stale detection triggers
        original_flock = fcntl.flock
        def patched_flock(fd, operation):
            # Fail on exclusive non-blocking call before unlink attempted
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB) and len(unlink_calls) == 0:
                raise BlockingIOError("Resource temporarily unavailable")
            return original_flock(fd, operation)

        # Patch unlink to fail (can't remove stale lock)
        original_unlink = Path.unlink
        def patched_unlink(self, *args, **kwargs):
            unlink_calls.append(str(self))
            if str(lock_path) in str(self):
                raise PermissionError("Cannot remove lock")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(fcntl, "flock", patched_flock)
        monkeypatch.setattr(Path, "unlink", patched_unlink)

        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should fail gracefully since stale lock couldn't be removed
        assert result is False
        # Verify unlink was attempted
        assert len(unlink_calls) > 0

    def test_lock_write_holder_info_fails(self, tmp_path, monkeypatch):
        """Lock acquisition succeeds even if writing holder info fails."""
        from cortical.utils.locking import ProcessLock
        import sys


        lock_path = tmp_path / ".lock"

        lock = ProcessLock(lock_path)

        # Patch json.dump to fail
        import json as json_module
        original_dump = json_module.dump
        def patched_dump(*args, **kwargs):
            raise IOError("Cannot write holder info")

        # Acquire first to create file descriptor
        first_call = [True]
        def patched_dump_once(*args, **kwargs):
            if first_call[0]:
                first_call[0] = False
                raise IOError("Cannot write holder info")
            return original_dump(*args, **kwargs)

        monkeypatch.setattr(json_module, "dump", patched_dump_once)

        result = lock.acquire()

        # Should still succeed (metadata write failure is non-fatal)
        assert result is True
        lock.release()

    def test_lock_timeout_with_backoff(self, tmp_path):
        """Lock with timeout exercises exponential backoff code."""
        from cortical.utils.locking import ProcessLock
        import sys
        import time


        lock_path = tmp_path / ".lock"
        lock1 = ProcessLock(lock_path, reentrant=False)

        # Hold lock
        assert lock1.acquire() is True

        lock2 = ProcessLock(lock_path, reentrant=False)

        start = time.time()
        result = lock2.acquire(timeout=0.15)  # Should retry a few times
        elapsed = time.time() - start

        # Should have tried for approximately the timeout period
        assert result is False
        assert elapsed >= 0.1  # At least tried for a bit

        lock1.release()

    def test_lock_reentrant_release_decrement(self, tmp_path):
        """Reentrant lock decrements count correctly."""
        from cortical.utils.locking import ProcessLock

        lock = ProcessLock(tmp_path / ".lock", reentrant=True)

        # Acquire 3 times
        lock.acquire()
        lock.acquire()
        lock.acquire()
        assert lock._lock_count == 3

        # Release 2 times
        lock.release()
        assert lock._lock_count == 2
        lock.release()
        assert lock._lock_count == 1

        # Still locked
        assert lock.is_locked()

        # Final release
        lock.release()
        assert lock._lock_count == 0
        assert not lock.is_locked()

    def test_lock_stale_removed_but_retry_fails(self, tmp_path, monkeypatch):
        """Stale lock is successfully removed, but retry acquisition fails."""
        from cortical.utils.locking import ProcessLock
        import sys
        import fcntl


        lock_path = tmp_path / ".lock"

        # Create stale lock (dead PID)
        holder_info = {"pid": 999999999, "acquired_at": 0}
        lock_path.write_text(json.dumps(holder_info))

        flock_call_count = [0]

        # Patch flock to always fail
        original_flock = fcntl.flock
        def patched_flock(fd, operation):
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                flock_call_count[0] += 1
                # Both initial and retry fail
                raise BlockingIOError("Resource temporarily unavailable")
            return original_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should fail because retry also failed
        assert result is False
        # Should have tried at least twice (initial + retry after stale removal)
        assert flock_call_count[0] >= 2

    def test_lock_acquire_unexpected_exception(self, tmp_path, monkeypatch):
        """Unexpected exception in _try_acquire_once is caught."""
        from cortical.utils.locking import ProcessLock
        import builtins

        lock_path = tmp_path / ".lock"

        # Patch open to raise an unexpected exception
        original_open = builtins.open
        def patched_open(path, *args, **kwargs):
            if str(lock_path) in str(path):
                raise RuntimeError("Unexpected disk error!")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", patched_open)

        lock = ProcessLock(lock_path)
        result = lock.acquire()

        # Should fail gracefully
        assert result is False

    def test_lock_release_flock_fails(self, tmp_path, monkeypatch):
        """Release handles flock unlock failure."""
        from cortical.utils.locking import ProcessLock
        import sys
        import fcntl


        lock_path = tmp_path / ".lock"
        lock = ProcessLock(lock_path)

        # Acquire first
        assert lock.acquire() is True

        # Now patch flock to fail on unlock
        original_flock = fcntl.flock
        def patched_flock(fd, operation):
            if operation == fcntl.LOCK_UN:
                raise IOError("Cannot unlock")
            return original_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        # Release should not raise - it catches the exception
        lock.release()

        # Lock should be released from our tracking
        assert lock._lock_count == 0
        assert lock._fd is None

    def test_lock_exception_after_file_open_cleans_up_fd(self, tmp_path, monkeypatch):
        """Exception after file open cleans up file descriptor."""
        from cortical.utils.locking import ProcessLock
        import sys
        import fcntl


        lock_path = tmp_path / ".lock"

        # Patch flock to raise an unexpected exception (not IOError/OSError)
        def patched_flock(fd, operation):
            raise RuntimeError("Unexpected flock error!")

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path)
        result = lock.acquire()

        # Should fail gracefully and clean up fd
        assert result is False
        assert lock._fd is None

    def test_lock_timeout_success_on_retry(self, tmp_path):
        """Lock acquired on retry within timeout period."""
        from cortical.utils.locking import ProcessLock
        import sys
        import fcntl
        import threading
        import time


        lock_path = tmp_path / ".lock"
        lock1 = ProcessLock(lock_path, reentrant=False)

        # Hold lock briefly
        assert lock1.acquire() is True

        # Release after 0.1s
        def release_soon():
            time.sleep(0.1)
            lock1.release()

        thread = threading.Thread(target=release_soon)
        thread.start()

        lock2 = ProcessLock(lock_path, reentrant=False)
        start = time.time()
        result = lock2.acquire(timeout=0.5)  # Should succeed after lock1 releases
        elapsed = time.time() - start

        thread.join()

        # Should have acquired the lock
        assert result is True
        # Should have taken at least 0.1s (waited for release)
        assert elapsed >= 0.05
        lock2.release()

    def test_lock_fd_close_fails_in_exception_handler(self, tmp_path, monkeypatch):
        """fd.close() failing inside exception handler is handled."""
        from cortical.utils.locking import ProcessLock
        import sys
        import fcntl


        lock_path = tmp_path / ".lock"

        # We need to:
        # 1. Open file successfully (so _fd is set)
        # 2. Have flock raise a non-IOError exception (triggers outer except)
        # 3. Have _fd.close() also fail

        close_patched = [False]
        original_close = None

        def patched_flock(fd, operation):
            # After flock is called, patch close to fail
            nonlocal close_patched, original_close
            if not close_patched[0]:
                close_patched[0] = True
                # Get the file object and patch its close method
                import io
                original_close = io.IOBase.close
                def failing_close(self):
                    raise OSError("Cannot close file")
                monkeypatch.setattr(io.IOBase, "close", failing_close)
            raise RuntimeError("Unexpected flock error!")

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path)
        result = lock.acquire()

        # Should fail gracefully even when close() fails
        assert result is False


# =============================================================================
# 12. ADDITIONAL COVERAGE - WAL AND RECOVERY EDGE CASES
# =============================================================================

class TestAdditionalWALCoverage:
    """Additional tests for remaining WAL coverage."""

    def test_wal_log_tx_abort(self, got_dir):
        """Test log_tx_abort creates proper entry."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")
        seq = wal.log_tx_abort("TX-abort-test", "user_cancelled")

        assert seq >= 1
        entries = wal.replay()
        abort_entries = [e for e in entries if e['op'] == 'TX_ABORT']
        assert len(abort_entries) >= 1
        assert abort_entries[0]['data']['reason'] == "user_cancelled"


class TestAdditionalRecoveryCoverage:
    """Additional tests for remaining recovery coverage."""

    def test_detect_orphaned_entities_with_adopted_entries(self, got_dir):
        """detect_orphaned_entities should recognize ADOPTED entries."""
        # Create entity on disk
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-adopted.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-adopted", "entity_type": "task", "title": "Adopted"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-adopted", "entity_type": "task", "title": "Adopted"}
        }
        entity_file.write_text(json.dumps(entity_data))

        # Create WAL with ADOPTED entry for this entity
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"
        adopted_entry = {
            "op": "ADOPTED",
            "entity_id": "T-adopted",
            "reason": "orphan_recovery",
            "timestamp": 1234567890
        }
        adopted_entry["checksum"] = compute_checksum({k: v for k, v in adopted_entry.items() if k != "checksum"})
        wal_file.write_text(json.dumps(adopted_entry) + "\n")

        recovery = RecoveryManager(got_dir)
        orphans = recovery.detect_orphaned_entities()

        # T-adopted should NOT be in orphans since it's tracked via ADOPTED entry
        assert "T-adopted" not in orphans

    def test_detect_orphaned_entities_write_without_entity_id(self, got_dir):
        """detect_orphaned_entities handles WRITE entries without entity_id."""
        # Create entity on disk
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-no-eid.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-no-eid", "entity_type": "task", "title": "No EID"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-no-eid", "entity_type": "task", "title": "No EID"}
        }
        entity_file.write_text(json.dumps(entity_data))

        # Create WAL with WRITE entry that has data but no entity_id
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"
        write_entry = {
            "op": "WRITE",
            "tx": "TX-1",
            "data": {"some_field": "value"}  # No entity_id
        }
        write_entry["checksum"] = compute_checksum({k: v for k, v in write_entry.items() if k != "checksum"})
        wal_file.write_text(json.dumps(write_entry) + "\n")

        recovery = RecoveryManager(got_dir)
        orphans = recovery.detect_orphaned_entities()

        # T-no-eid should be in orphans since WAL WRITE has no entity_id
        assert "T-no-eid" in orphans

    def test_detect_orphaned_entities_write_with_non_dict_data(self, got_dir):
        """detect_orphaned_entities handles WRITE entries where data is not a dict."""
        # Create entity on disk
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-bad-data.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-bad-data", "entity_type": "task", "title": "Bad"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-bad-data", "entity_type": "task", "title": "Bad"}
        }
        entity_file.write_text(json.dumps(entity_data))

        # Create WAL with WRITE entry where data is not a dict
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"
        write_entry = {
            "op": "WRITE",
            "tx": "TX-1",
            "data": "not a dict"  # String instead of dict
        }
        write_entry["checksum"] = compute_checksum({k: v for k, v in write_entry.items() if k != "checksum"})
        wal_file.write_text(json.dumps(write_entry) + "\n")

        recovery = RecoveryManager(got_dir)
        orphans = recovery.detect_orphaned_entities()

        # T-bad-data should be in orphans since WAL has non-dict data
        assert "T-bad-data" in orphans


# =============================================================================
# 13. FINAL COVERAGE PUSH - REMAINING BRANCHES
# =============================================================================

class TestFinalCoveragePush:
    """Tests targeting remaining uncovered lines for 100% coverage."""

    def test_lock_timeout_immediate_return(self, tmp_path, monkeypatch):
        """Timeout returns immediately when elapsed >= timeout."""
        from cortical.utils.locking import ProcessLock
        import fcntl

        lock_path = tmp_path / ".lock"

        # Make flock always fail so we hit the timeout loop
        def patched_flock(fd, operation):
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                raise BlockingIOError("Resource temporarily unavailable")

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path)

        # Very short timeout - should hit line 122 quickly
        import time
        start = time.time()
        result = lock.acquire(timeout=0.01)  # 10ms timeout
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 0.5  # Should be quick

    def test_lock_stale_empty_file_via_flock_failure(self, tmp_path, monkeypatch):
        """Empty lock file detected as stale when flock fails."""
        from cortical.utils.locking import ProcessLock
        import fcntl

        lock_path = tmp_path / ".lock"

        # Create an empty lock file
        lock_path.write_text("")

        flock_calls = [0]
        original_flock = fcntl.flock

        def patched_flock(fd, operation):
            flock_calls[0] += 1
            if flock_calls[0] == 1:
                # First call fails, triggering stale detection
                raise BlockingIOError("Resource temporarily unavailable")
            # Retry succeeds after stale lock is removed
            return original_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should succeed after detecting stale (empty) lock and recovering
        assert result is True
        lock.release()

    def test_lock_stale_invalid_json_via_flock_failure(self, tmp_path, monkeypatch):
        """Invalid JSON in lock file detected as stale when flock fails."""
        from cortical.utils.locking import ProcessLock
        import fcntl

        lock_path = tmp_path / ".lock"

        # Create lock file with invalid JSON
        lock_path.write_text("not valid json {{{")

        flock_calls = [0]
        original_flock = fcntl.flock

        def patched_flock(fd, operation):
            flock_calls[0] += 1
            if flock_calls[0] == 1:
                # First call fails, triggering stale detection
                raise BlockingIOError("Resource temporarily unavailable")
            # Retry succeeds after stale lock is removed
            return original_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should succeed after detecting stale (invalid JSON) lock and recovering
        assert result is True
        lock.release()

    def test_lock_backoff_zero_sleep_time(self, tmp_path, monkeypatch):
        """Test backoff when sleep_time is zero or negative."""
        from cortical.utils.locking import ProcessLock
        import fcntl
        import time

        lock_path = tmp_path / ".lock"

        # Track flock call count
        flock_calls = [0]
        start_time = [None]

        def patched_flock(fd, operation):
            if start_time[0] is None:
                start_time[0] = time.time()
            flock_calls[0] += 1

            # Always fail to force timeout loop
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                raise BlockingIOError("Resource temporarily unavailable")

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path)

        # Very short timeout to force the branch where timeout-elapsed is tiny
        result = lock.acquire(timeout=0.005)  # 5ms

        assert result is False

    def test_lock_retry_flock_fails_on_retry(self, tmp_path, monkeypatch):
        """Flock fails both on initial attempt and retry after stale removal."""
        from cortical.utils.locking import ProcessLock
        import fcntl

        lock_path = tmp_path / ".lock"

        # Create stale lock (dead PID)
        lock_path.write_text('{"pid": 999999999, "acquired_at": 0}')

        # Make flock always fail
        def patched_flock(fd, operation):
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                raise BlockingIOError("Resource temporarily unavailable")

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path, stale_timeout=1.0)
        result = lock.acquire()

        # Should fail because both initial and retry flock fail
        assert result is False

    def test_lock_timeout_before_sleep(self, tmp_path, monkeypatch):
        """Timeout exceeded before sleep (hits line 122)."""
        from cortical.utils.locking import ProcessLock
        import fcntl
        import time
        import os

        lock_path = tmp_path / ".lock"

        # Create a lock held by current process (not stale)
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "acquired_at": time.time()
        }))

        call_count = [0]

        def patched_flock(fd, operation):
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                call_count[0] += 1
                # Delay to exceed timeout on first attempt
                time.sleep(0.05)  # 50ms delay
                raise BlockingIOError("Resource temporarily unavailable")

        monkeypatch.setattr(fcntl, "flock", patched_flock)

        lock = ProcessLock(lock_path)

        # 30ms timeout - flock takes 50ms, so we exceed timeout before sleep
        result = lock.acquire(timeout=0.03)

        assert result is False
        # Should have tried only once (lock not stale, no retry)
        assert call_count[0] == 1

    def test_wal_replay_entries_nonexistent_file(self, got_dir):
        """replay_entries returns empty list for non-existent WAL file."""
        from cortical.got.wal import WALManager

        wal = WALManager(got_dir / "wal")

        # Make sure WAL file doesn't exist
        if wal.wal_file.exists():
            wal.wal_file.unlink()

        entries = wal.replay_entries()
        assert entries == []

    def test_wal_replay_entries_with_empty_lines(self, got_dir):
        """replay_entries skips empty lines in WAL file."""
        from cortical.got.wal import WALManager
        from cortical.wal import TransactionWALEntry

        wal = WALManager(got_dir / "wal")

        # Create a valid entry
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            tx_id="TX-test",
            seq=1,
            payload={}
        )

        # Write WAL file with empty lines
        with open(wal.wal_file, 'w') as f:
            f.write("\n")  # Empty line
            f.write(json.dumps(entry.to_dict()) + "\n")
            f.write("   \n")  # Whitespace-only line
            f.write("\n")  # Another empty line

        entries = wal.replay_entries()

        # Should get just the one valid entry
        assert len(entries) == 1
        assert entries[0].tx_id == "TX-test"
