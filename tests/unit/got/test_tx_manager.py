"""
Tests for TransactionManager with ACID guarantees.

Covers:
- Transaction lifecycle (begin/commit/rollback)
- Conflict detection
- Crash recovery
- Lock management
"""

import pytest
from pathlib import Path

from cortical.got.tx_manager import TransactionManager, CommitResult, Conflict
from cortical.got.transaction import TransactionState
from cortical.got.types import Task
from cortical.got.errors import TransactionError


class TestTransactionManager:
    """Test suite for TransactionManager."""

    @pytest.fixture
    def tmp_got_dir(self, tmp_path):
        """Create temporary GoT directory for tests."""
        got_dir = tmp_path / "got"
        got_dir.mkdir()
        return got_dir

    @pytest.fixture
    def manager(self, tmp_got_dir):
        """Create TransactionManager instance."""
        return TransactionManager(tmp_got_dir)

    def test_begin_creates_transaction(self, manager):
        """Test that begin() returns active transaction."""
        tx = manager.begin()

        assert tx is not None
        assert tx.id.startswith("TX-")
        assert tx.state == TransactionState.ACTIVE
        assert tx.snapshot_version == 0  # Initial version
        assert len(tx.write_set) == 0
        assert len(tx.read_set) == 0

    def test_commit_applies_writes(self, manager):
        """Test that writes are visible after commit."""
        # Create transaction
        tx = manager.begin()

        # Create task
        task = Task(
            id="T-001",
            title="Test task",
            status="pending",
            priority="medium"
        )

        # Write to transaction
        manager.write(tx, task)

        # Verify not visible yet
        assert not manager.store.exists("T-001")

        # Commit
        result = manager.commit(tx)

        # Verify success
        assert result.success is True
        assert result.version == 1
        assert len(result.conflicts) == 0

        # Verify write is visible
        assert manager.store.exists("T-001")
        loaded = manager.store.read("T-001")
        assert loaded.id == "T-001"
        assert loaded.title == "Test task"

    def test_rollback_discards_writes(self, manager):
        """Test that writes are not visible after rollback."""
        # Create transaction
        tx = manager.begin()

        # Create task
        task = Task(
            id="T-002",
            title="Test task",
            status="pending",
            priority="medium"
        )

        # Write to transaction
        manager.write(tx, task)

        # Rollback
        manager.rollback(tx, reason="test_rollback")

        # Verify state
        assert tx.state == TransactionState.ROLLED_BACK
        assert len(tx.write_set) == 0

        # Verify write is not visible
        assert not manager.store.exists("T-002")

    def test_conflict_detected_on_version_mismatch(self, manager):
        """Test that concurrent modifications are detected."""
        # Create initial task
        tx1 = manager.begin()
        task = Task(
            id="T-003",
            title="Original",
            status="pending",
            priority="medium"
        )
        manager.write(tx1, task)
        manager.commit(tx1)

        # TX2: Read task
        tx2 = manager.begin()
        task_read = manager.read(tx2, "T-003")
        assert task_read is not None

        # TX3: Update and commit (creates conflict)
        tx3 = manager.begin()
        task_read_tx3 = manager.read(tx3, "T-003")
        task_read_tx3.title = "Updated by TX3"
        manager.write(tx3, task_read_tx3)
        result3 = manager.commit(tx3)
        assert result3.success is True

        # TX2: Try to update (should conflict)
        task_read.title = "Updated by TX2"
        manager.write(tx2, task_read)
        result2 = manager.commit(tx2)

        # Verify conflict
        assert result2.success is False
        assert len(result2.conflicts) > 0
        assert result2.conflicts[0].entity_id == "T-003"
        assert result2.conflicts[0].conflict_type == "version_mismatch"

    def test_crash_recovery_rolls_back_incomplete(self, manager):
        """Test that recovery rolls back incomplete transactions."""
        # Create incomplete transaction
        tx = manager.begin()
        task = Task(
            id="T-004",
            title="Incomplete",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)

        # Don't commit - simulate crash

        # Create new manager (simulates restart)
        recovery_manager = TransactionManager(manager.got_dir)

        # Check recovery result (happens in __init__)
        # The recovery was already run during init, so we verify state
        assert not recovery_manager.store.exists("T-004")

    def test_lock_acquired_during_commit(self, manager):
        """Test that lock is held during commit."""
        # This is hard to test directly without threading,
        # but we can verify lock operations work
        tx = manager.begin()
        task = Task(
            id="T-005",
            title="Test",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)

        # Commit should succeed (lock acquired and released)
        result = manager.commit(tx)
        assert result.success is True

        # Verify lock was released (can acquire again)
        assert manager.lock.acquire()
        manager.lock.release()

    def test_read_returns_none_for_nonexistent(self, manager):
        """Test that reading non-existent entity returns None."""
        tx = manager.begin()
        entity = manager.read(tx, "NONEXISTENT")
        assert entity is None

    def test_read_sees_own_writes(self, manager):
        """Test that transaction sees its own writes."""
        tx = manager.begin()

        # Write task
        task = Task(
            id="T-006",
            title="Own write",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)

        # Read should see the write
        read_task = manager.read(tx, "T-006")
        assert read_task is not None
        assert read_task.id == "T-006"
        assert read_task.title == "Own write"

    def test_write_buffers_entity(self, manager):
        """Test that write buffers entity in write_set."""
        tx = manager.begin()

        task = Task(
            id="T-007",
            title="Buffered",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)

        # Check write set
        assert "T-007" in tx.write_set
        assert tx.write_set["T-007"].title == "Buffered"

    def test_commit_increments_version(self, manager):
        """Test that commit increments global version."""
        initial_version = manager.store.current_version()

        tx = manager.begin()
        task = Task(
            id="T-008",
            title="Version test",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)
        result = manager.commit(tx)

        assert result.success is True
        assert result.version == initial_version + 1
        assert manager.store.current_version() == initial_version + 1

    def test_cannot_commit_rolled_back_tx(self, manager):
        """Test that rolled back transaction cannot be committed."""
        tx = manager.begin()
        task = Task(
            id="T-009",
            title="Test",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)

        # Rollback
        manager.rollback(tx)

        # Try to commit
        result = manager.commit(tx)
        assert result.success is False
        assert "cannot commit" in result.reason.lower()

    def test_cannot_rollback_committed_tx(self, manager):
        """Test that committed transaction cannot be rolled back."""
        tx = manager.begin()
        task = Task(
            id="T-010",
            title="Test",
            status="pending",
            priority="medium"
        )
        manager.write(tx, task)

        # Commit
        manager.commit(tx)

        # Try to rollback
        with pytest.raises(TransactionError, match="cannot rollback"):
            manager.rollback(tx)

    def test_write_to_inactive_transaction_raises(self, manager):
        """Test that writing to inactive transaction raises error."""
        tx = manager.begin()

        # Commit to make inactive
        manager.commit(tx)

        # Try to write
        task = Task(
            id="T-011",
            title="Test",
            status="pending",
            priority="medium"
        )

        with pytest.raises(TransactionError, match="not active"):
            manager.write(tx, task)

    def test_multiple_writes_in_transaction(self, manager):
        """Test multiple writes in single transaction."""
        tx = manager.begin()

        # Write multiple tasks
        for i in range(3):
            task = Task(
                id=f"T-multi-{i}",
                title=f"Task {i}",
                status="pending",
                priority="medium"
            )
            manager.write(tx, task)

        # Commit
        result = manager.commit(tx)
        assert result.success is True

        # Verify all visible
        for i in range(3):
            assert manager.store.exists(f"T-multi-{i}")

    def test_read_at_snapshot_version(self, manager):
        """Test that reads use snapshot version."""
        # Create task in TX1
        tx1 = manager.begin()
        task = Task(
            id="T-snapshot",
            title="Version 1",
            status="pending",
            priority="medium"
        )
        manager.write(tx1, task)
        manager.commit(tx1)

        # Start TX2 (takes snapshot)
        tx2 = manager.begin()
        snapshot_v = tx2.snapshot_version

        # Modify in TX3
        tx3 = manager.begin()
        task_v3 = manager.read(tx3, "T-snapshot")
        task_v3.title = "Version 2"
        manager.write(tx3, task_v3)
        manager.commit(tx3)

        # TX2 should still see old version
        task_tx2 = manager.read(tx2, "T-snapshot")
        assert task_tx2.title == "Version 1"  # Sees snapshot

    def test_recovery_result_structure(self, manager):
        """Test recovery result has correct structure."""
        # Create incomplete transaction
        tx = manager.begin()
        manager.write(tx, Task(id="T-rec", title="Test", status="pending", priority="medium"))

        # Create new manager (triggers recovery)
        new_manager = TransactionManager(manager.got_dir)

        # Recovery ran in __init__, verify it worked
        assert not new_manager.store.exists("T-rec")


class TestProcessLock:
    """Test suite for ProcessLock."""

    def test_lock_acquire_release(self, tmp_path):
        """Test basic lock acquire and release."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        # Acquire
        assert lock.acquire() is True
        assert lock._lock_count == 1

        # Release
        lock.release()
        assert lock._lock_count == 0

    def test_lock_reentrant(self, tmp_path):
        """Test reentrant lock allows multiple acquires."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path, reentrant=True)

        # Multiple acquires
        assert lock.acquire() is True
        assert lock.acquire() is True
        assert lock._lock_count == 2

        # Must release same number of times
        lock.release()
        assert lock._lock_count == 1
        lock.release()
        assert lock._lock_count == 0

    def test_lock_context_manager(self, tmp_path):
        """Test lock as context manager."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        with lock:
            assert lock._lock_count == 1

        assert lock._lock_count == 0

    def test_lock_timeout_success(self, tmp_path):
        """Test lock acquired within timeout."""
        import threading
        import time
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "timeout_test.lock"
        lock1 = ProcessLock(lock_path, reentrant=False)
        lock2 = ProcessLock(lock_path, reentrant=False)

        # Lock1 acquires
        assert lock1.acquire() is True

        # Function to release lock1 after 0.2 seconds
        def release_after_delay():
            time.sleep(0.2)
            lock1.release()

        thread = threading.Thread(target=release_after_delay)
        thread.start()

        # Lock2 should acquire within 1 second timeout
        start = time.time()
        assert lock2.acquire(timeout=1.0) is True
        elapsed = time.time() - start

        # Should have waited approximately 0.2 seconds
        assert 0.15 < elapsed < 0.4

        lock2.release()
        thread.join()

    def test_lock_timeout_expired(self, tmp_path):
        """Test lock not acquired when timeout expires."""
        import time
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "timeout_expired.lock"
        lock1 = ProcessLock(lock_path, reentrant=False)
        lock2 = ProcessLock(lock_path, reentrant=False)

        # Lock1 acquires and holds
        assert lock1.acquire() is True

        # Lock2 tries with short timeout (should fail)
        start = time.time()
        assert lock2.acquire(timeout=0.1) is False
        elapsed = time.time() - start

        # Should have waited approximately the timeout duration
        assert 0.08 < elapsed < 0.2

        lock1.release()

    def test_lock_stale_recovery(self, tmp_path):
        """Test stale lock from dead PID is recovered."""
        import json
        import os
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "stale.lock"

        # Create a lock file with a fake (likely dead) PID
        fake_pid = 999999  # Very unlikely to exist
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, 'w') as f:
            json.dump({"pid": fake_pid, "acquired_at": 0.0}, f)

        # Try to acquire - should detect stale lock and steal it
        lock = ProcessLock(lock_path, reentrant=False)
        assert lock.acquire() is True

        # Verify holder info was updated with current PID
        with open(lock_path, 'r') as f:
            holder_info = json.load(f)
            assert holder_info["pid"] == os.getpid()

        lock.release()

    def test_lock_backoff_pattern(self, tmp_path):
        """Test exponential backoff timing pattern."""
        import time
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "backoff.lock"
        lock1 = ProcessLock(lock_path, reentrant=False)
        lock2 = ProcessLock(lock_path, reentrant=False)

        # Lock1 acquires and holds
        assert lock1.acquire() is True

        # Lock2 tries with timeout - measure time
        start = time.time()
        result = lock2.acquire(timeout=0.3)
        elapsed = time.time() - start

        # Should fail (lock1 still holds)
        assert result is False

        # Should have retried multiple times with backoff
        # Expected backoff: 10ms, 20ms, 40ms, 80ms, 160ms...
        # In 0.3 seconds, should have several attempts
        # Total time should be close to timeout
        assert 0.25 < elapsed < 0.4

        lock1.release()

    def test_lock_writes_holder_info(self, tmp_path):
        """Test that lock file contains PID and timestamp."""
        import json
        import os
        import time
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "holder_info.lock"
        lock = ProcessLock(lock_path)

        before_time = time.time()
        assert lock.acquire() is True
        after_time = time.time()

        # Read lock file
        assert lock_path.exists()
        with open(lock_path, 'r') as f:
            holder_info = json.load(f)

        # Verify structure
        assert "pid" in holder_info
        assert "acquired_at" in holder_info

        # Verify values
        assert holder_info["pid"] == os.getpid()
        assert before_time <= holder_info["acquired_at"] <= after_time

        lock.release()

    def test_lock_no_timeout_backward_compatible(self, tmp_path):
        """Test that timeout=None preserves original behavior."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "no_timeout.lock"
        lock1 = ProcessLock(lock_path, reentrant=False)
        lock2 = ProcessLock(lock_path, reentrant=False)

        # Lock1 acquires
        assert lock1.acquire() is True

        # Lock2 with no timeout should fail immediately (non-blocking)
        import time
        start = time.time()
        assert lock2.acquire(timeout=None) is False
        elapsed = time.time() - start

        # Should be nearly instant (< 10ms)
        assert elapsed < 0.01

        lock1.release()

    def test_lock_handles_empty_lock_file(self, tmp_path):
        """Test that empty lock file is considered stale."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "empty.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty lock file
        lock_path.touch()

        # Should be able to acquire (empty file = stale)
        lock = ProcessLock(lock_path)
        assert lock.acquire() is True

        lock.release()

    def test_lock_handles_invalid_json(self, tmp_path):
        """Test that invalid JSON in lock file is considered stale."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "invalid.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Create lock file with invalid JSON
        with open(lock_path, 'w') as f:
            f.write("not valid json {{{")

        # Should be able to acquire (invalid JSON = stale)
        lock = ProcessLock(lock_path)
        assert lock.acquire() is True

        lock.release()

    def test_lock_reentrant_with_timeout(self, tmp_path):
        """Test that reentrant lock works with timeout parameter."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "reentrant_timeout.lock"
        lock = ProcessLock(lock_path, reentrant=True)

        # First acquire
        assert lock.acquire(timeout=1.0) is True
        assert lock._lock_count == 1

        # Second acquire (reentrant) - should succeed immediately
        import time
        start = time.time()
        assert lock.acquire(timeout=1.0) is True
        elapsed = time.time() - start

        # Should be instant (no waiting for timeout)
        assert elapsed < 0.01
        assert lock._lock_count == 2

        lock.release()
        lock.release()
