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
from cortical.core.bootstrap import create_container


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
        """Create TransactionManager instance via container."""
        container = create_container(got_dir=tmp_got_dir)
        return container.resolve(TransactionManager)

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

        thread = None
        try:
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
            # Use generous upper bound for CI variance (can be 2-3x slower)
            assert 0.15 < elapsed < 1.0

            lock2.release()
        finally:
            # Ensure thread completes and locks are cleaned up
            if thread is not None:
                thread.join(timeout=2.0)
            # Force release any held locks to avoid resource leaks
            if lock1._lock_count > 0:
                lock1.release()
            if lock2._lock_count > 0:
                lock2.release()

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


class TestTransactionManagerKnowledgeTransfer:
    """Test suite for TransactionManager KnowledgeTransfer methods."""

    @pytest.fixture
    def tmp_got_dir(self, tmp_path):
        """Create temporary GoT directory for tests."""
        got_dir = tmp_path / "got"
        got_dir.mkdir()
        return got_dir

    @pytest.fixture
    def manager(self, tmp_got_dir):
        """Create TransactionManager instance via container."""
        container = create_container(got_dir=tmp_got_dir)
        return container.resolve(TransactionManager)

    def test_create_knowledge_transfer(self, manager):
        """Test creating a knowledge transfer."""
        from cortical.got.types import KnowledgeTransfer

        kt = manager.create_knowledge_transfer(
            title="Test KT",
            summary="Test summary",
            session_id="sess123",
        )

        assert kt is not None
        assert kt.id.startswith("KT-")
        assert kt.title == "Test KT"
        assert kt.summary == "Test summary"
        assert kt.session_id == "sess123"
        assert isinstance(kt, KnowledgeTransfer)

    def test_create_knowledge_transfer_with_custom_id(self, manager):
        """Test creating a knowledge transfer with custom ID."""
        kt = manager.create_knowledge_transfer(
            title="Custom ID KT",
            kt_id="KT-CUSTOM-123",
        )

        assert kt.id == "KT-CUSTOM-123"

    def test_create_knowledge_transfer_with_sections(self, manager):
        """Test creating a knowledge transfer with sections."""
        kt = manager.create_knowledge_transfer(
            title="Sections KT",
            sections={"Technical": "Tech content", "Notes": "Note content"},
        )

        assert "Technical" in kt.sections
        assert kt.sections["Technical"] == "Tech content"
        assert "Notes" in kt.sections

    def test_create_knowledge_transfer_with_tags(self, manager):
        """Test creating a knowledge transfer with tags."""
        kt = manager.create_knowledge_transfer(
            title="Tagged KT",
            tags=["architecture", "testing"],
        )

        assert "architecture" in kt.tags
        assert "testing" in kt.tags

    def test_get_knowledge_transfer(self, manager):
        """Test getting an existing knowledge transfer."""
        # Create a KT
        kt = manager.create_knowledge_transfer(
            title="Get Test KT",
            kt_id="KT-GET-TEST",
        )

        # Get it back
        retrieved = manager.get_knowledge_transfer("KT-GET-TEST")

        assert retrieved is not None
        assert retrieved.id == "KT-GET-TEST"
        assert retrieved.title == "Get Test KT"

    def test_get_knowledge_transfer_not_found(self, manager):
        """Test getting a non-existent knowledge transfer."""
        result = manager.get_knowledge_transfer("KT-NONEXISTENT")
        assert result is None

    def test_list_knowledge_transfers_empty(self, manager):
        """Test listing knowledge transfers when none exist."""
        transfers = manager.list_knowledge_transfers()
        assert transfers == []

    def test_list_knowledge_transfers(self, manager):
        """Test listing knowledge transfers."""
        # Create multiple KTs
        manager.create_knowledge_transfer(title="KT 1", kt_id="KT-LIST-1")
        manager.create_knowledge_transfer(title="KT 2", kt_id="KT-LIST-2")

        transfers = manager.list_knowledge_transfers()

        assert len(transfers) == 2
        ids = [kt.id for kt in transfers]
        assert "KT-LIST-1" in ids
        assert "KT-LIST-2" in ids

    def test_list_knowledge_transfers_filter_by_status(self, manager):
        """Test listing knowledge transfers filtered by status."""
        # Create a KT - default status is 'published' (per cortical/got/types.py)
        kt1 = manager.create_knowledge_transfer(title="Published KT", kt_id="KT-STATUS-1")

        # List with status filter
        drafts = manager.list_knowledge_transfers(status="draft")
        published = manager.list_knowledge_transfers(status="published")

        assert len(drafts) == 0  # No drafts - default is published
        assert len(published) == 1

    def test_list_knowledge_transfers_filter_by_tags(self, manager):
        """Test listing knowledge transfers filtered by tags."""
        manager.create_knowledge_transfer(
            title="Tagged KT",
            kt_id="KT-TAGS-1",
            tags=["architecture", "testing"]
        )
        manager.create_knowledge_transfer(
            title="Untagged KT",
            kt_id="KT-TAGS-2",
            tags=["performance"]
        )

        # Filter by tag
        arch_kts = manager.list_knowledge_transfers(tags=["architecture"])

        assert len(arch_kts) == 1
        assert arch_kts[0].id == "KT-TAGS-1"

    def test_append_to_knowledge_transfer_new_section(self, manager):
        """Test appending to a new section."""
        kt = manager.create_knowledge_transfer(
            title="Append Test",
            kt_id="KT-APPEND-1",
        )

        # Append to new section
        updated = manager.append_to_knowledge_transfer(
            "KT-APPEND-1",
            "New Section",
            "New content"
        )

        assert "New Section" in updated.sections
        assert updated.sections["New Section"] == "New content"

    def test_append_to_knowledge_transfer_existing_section(self, manager):
        """Test appending to an existing section."""
        kt = manager.create_knowledge_transfer(
            title="Append Existing Test",
            kt_id="KT-APPEND-2",
            sections={"Existing": "Initial content"}
        )

        # Append to existing section
        updated = manager.append_to_knowledge_transfer(
            "KT-APPEND-2",
            "Existing",
            "Additional content"
        )

        assert "Initial content" in updated.sections["Existing"]
        assert "Additional content" in updated.sections["Existing"]

    def test_append_to_knowledge_transfer_not_found(self, manager):
        """Test appending to non-existent KT raises error."""
        with pytest.raises(TransactionError):
            manager.append_to_knowledge_transfer(
                "KT-NONEXISTENT",
                "Section",
                "Content"
            )


class TestTransactionManagerDependencyInjection:
    """Test suite for TransactionManager dependency injection."""

    @pytest.fixture
    def tmp_got_dir(self, tmp_path):
        """Create temporary GoT directory for tests."""
        got_dir = tmp_path / "got"
        got_dir.mkdir()
        return got_dir

    def test_default_components_created_when_not_injected(self, tmp_got_dir):
        """Test that default components are created when none injected."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.utils.locking import ProcessLock

        manager = TransactionManager(tmp_got_dir)

        # Verify default components were created
        assert isinstance(manager.store, CDGStore)
        assert isinstance(manager.wal, CDGWALManager)
        assert isinstance(manager.lock, ProcessLock)

    def test_injected_store_is_used(self, tmp_got_dir):
        """Test that injected store is used instead of creating default."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.config import CDGConfig

        # Create custom store
        custom_store = CDGStore(
            tmp_got_dir / "custom_entities",
            config=CDGConfig.for_got()
        )

        manager = TransactionManager(tmp_got_dir, store=custom_store)

        # Verify injected store is used
        assert manager.store is custom_store

    def test_injected_wal_is_used(self, tmp_got_dir):
        """Test that injected WAL is used instead of creating default."""
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.config import CDGConfig

        # Create custom WAL
        custom_wal = CDGWALManager(
            tmp_got_dir / "custom_wal",
            CDGConfig.for_got()
        )

        manager = TransactionManager(tmp_got_dir, wal=custom_wal)

        # Verify injected WAL is used
        assert manager.wal is custom_wal

    def test_injected_lock_is_used(self, tmp_got_dir):
        """Test that injected lock is used instead of creating default."""
        from cortical.utils.locking import ProcessLock

        # Create custom lock
        custom_lock = ProcessLock(tmp_got_dir / "custom.lock", reentrant=True)

        manager = TransactionManager(tmp_got_dir, lock=custom_lock)

        # Verify injected lock is used
        assert manager.lock is custom_lock

    def test_all_components_can_be_injected(self, tmp_got_dir):
        """Test that all three components can be injected together."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.config import CDGConfig
        from cortical.utils.locking import ProcessLock

        config = CDGConfig.for_got()

        custom_store = CDGStore(tmp_got_dir / "inj_entities", config=config)
        custom_wal = CDGWALManager(tmp_got_dir / "inj_wal", config)
        custom_lock = ProcessLock(tmp_got_dir / "inj.lock", reentrant=True)

        manager = TransactionManager(
            tmp_got_dir,
            store=custom_store,
            wal=custom_wal,
            lock=custom_lock
        )

        assert manager.store is custom_store
        assert manager.wal is custom_wal
        assert manager.lock is custom_lock

    def test_invalid_store_type_raises_typeerror(self, tmp_got_dir):
        """Test that passing wrong type for store raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TransactionManager(tmp_got_dir, store="not a store")

        assert "store must be CDGStore instance" in str(exc_info.value)
        assert "got str" in str(exc_info.value)

    def test_invalid_wal_type_raises_typeerror(self, tmp_got_dir):
        """Test that passing wrong type for wal raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TransactionManager(tmp_got_dir, wal="not a wal")

        assert "wal must be CDGWALManager instance" in str(exc_info.value)
        assert "got str" in str(exc_info.value)

    def test_invalid_lock_type_raises_typeerror(self, tmp_got_dir):
        """Test that passing wrong type for lock raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TransactionManager(tmp_got_dir, lock="not a lock")

        assert "lock must be ProcessLock instance" in str(exc_info.value)
        assert "got str" in str(exc_info.value)

    def test_injected_store_works_for_transactions(self, tmp_got_dir):
        """Test that transactions work correctly with injected store."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.config import CDGConfig
        from cortical.got.versioned_store import _got_entity_factory

        # Create store with GoT entity factory
        custom_store = CDGStore(
            tmp_got_dir / "tx_entities",
            config=CDGConfig.for_got(),
            entity_factory=_got_entity_factory
        )

        manager = TransactionManager(tmp_got_dir, store=custom_store)

        # Perform a transaction
        tx = manager.begin()
        task = Task(id="T-INJ-1", title="Injected Test", status="pending", priority="high")
        manager.write(tx, task)
        result = manager.commit(tx)

        assert result.success

        # Read back
        tx2 = manager.begin()
        read_task = manager.read(tx2, "T-INJ-1")
        manager.rollback(tx2, "read_only")

        assert read_task is not None
        assert read_task.id == "T-INJ-1"
        assert read_task.title == "Injected Test"

    def test_partial_injection_uses_defaults_for_others(self, tmp_got_dir):
        """Test that partial injection creates defaults for non-injected components."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.config import CDGConfig
        from cortical.utils.locking import ProcessLock

        # Only inject store
        custom_store = CDGStore(
            tmp_got_dir / "partial_entities",
            config=CDGConfig.for_got()
        )

        manager = TransactionManager(tmp_got_dir, store=custom_store)

        # Store should be injected
        assert manager.store is custom_store

        # WAL and lock should be defaults (not None, proper instances)
        assert isinstance(manager.wal, CDGWALManager)
        assert isinstance(manager.lock, ProcessLock)

        # And they should NOT be the custom store's values
        assert manager.wal is not custom_store
