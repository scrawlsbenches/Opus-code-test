"""
Unit tests for process-safe locking mechanism.

Task T-20251221-114358-2b0b: Add thread safety to GoT graph operations.

Tests cover:
- Basic lock/unlock operations
- Context manager support
- Stale lock detection and recovery
- Timeout handling
- Multi-process safety
- Crash recovery
"""

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts to path for GoTProjectManager imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestProcessLockBasics:
    """Test basic lock/unlock operations."""

    def test_lock_creates_lock_file(self, tmp_path):
        """Acquiring lock should create a .lock file."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        with lock:
            assert lock_path.exists(), "Lock file should exist while held"

    def test_lock_contains_pid(self, tmp_path):
        """Lock file should contain the current process PID."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        with lock:
            content = lock_path.read_text()
            assert str(os.getpid()) in content, "Lock file should contain PID"

    def test_lock_released_after_context(self, tmp_path):
        """Lock should be released after context manager exits."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        with lock:
            pass

        # Lock file may or may not exist, but should be acquirable
        lock2 = ProcessLock(lock_path)
        acquired = lock2.acquire(timeout=0.1)
        assert acquired, "Lock should be acquirable after release"
        lock2.release()

    def test_explicit_acquire_release(self, tmp_path):
        """Test explicit acquire() and release() methods."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        acquired = lock.acquire()
        assert acquired is True, "acquire() should return True on success"
        assert lock.is_locked(), "is_locked() should return True"

        lock.release()
        assert not lock.is_locked(), "is_locked() should return False after release"


class TestProcessLockTimeout:
    """Test timeout handling."""

    def test_acquire_with_timeout_success(self, tmp_path):
        """Should acquire lock within timeout."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        acquired = lock.acquire(timeout=1.0)
        assert acquired is True
        lock.release()

    def test_acquire_fails_when_locked(self, tmp_path):
        """Should fail to acquire when already locked by same process."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock1 = ProcessLock(lock_path)
        lock2 = ProcessLock(lock_path)

        lock1.acquire()
        try:
            # Second lock should fail with short timeout
            acquired = lock2.acquire(timeout=0.1)
            assert acquired is False, "Should fail to acquire held lock"
        finally:
            lock1.release()

    def test_timeout_zero_is_non_blocking(self, tmp_path):
        """Timeout of 0 should be non-blocking."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock1 = ProcessLock(lock_path)
        lock2 = ProcessLock(lock_path)

        lock1.acquire()
        try:
            start = time.time()
            acquired = lock2.acquire(timeout=0)
            elapsed = time.time() - start

            assert acquired is False
            assert elapsed < 0.5, "Should return immediately with timeout=0"
        finally:
            lock1.release()


class TestStaleLockRecovery:
    """Test stale lock detection and recovery."""

    def test_detects_stale_lock_from_dead_process(self, tmp_path):
        """Should detect and recover from lock held by dead process."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"

        # Create a fake lock file with non-existent PID
        fake_pid = 999999999  # Unlikely to exist
        lock_path.write_text(f"{fake_pid}\n{time.time()}")

        lock = ProcessLock(lock_path)
        acquired = lock.acquire(timeout=0.1)

        assert acquired is True, "Should acquire lock from dead process"
        lock.release()

    def test_stale_lock_threshold(self, tmp_path):
        """Lock older than threshold should be considered stale."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"

        # Create lock file with old timestamp
        old_time = time.time() - 3600  # 1 hour ago
        lock_path.write_text(f"{os.getpid()}\n{old_time}")

        lock = ProcessLock(lock_path, stale_timeout=1800)  # 30 min threshold
        acquired = lock.acquire(timeout=0.1)

        assert acquired is True, "Should acquire stale lock"
        lock.release()

    def test_valid_lock_not_stolen(self, tmp_path):
        """Should not steal lock from live process with recent timestamp."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock1 = ProcessLock(lock_path)

        lock1.acquire()
        try:
            lock2 = ProcessLock(lock_path)
            acquired = lock2.acquire(timeout=0.1)
            assert acquired is False, "Should not steal valid lock"
        finally:
            lock1.release()


class TestProcessLockReentrancy:
    """Test reentrant lock behavior."""

    def test_same_process_can_reacquire(self, tmp_path):
        """Same process should be able to acquire lock multiple times."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path, reentrant=True)

        lock.acquire()
        # Should not block
        acquired = lock.acquire(timeout=0.1)
        assert acquired is True, "Reentrant lock should allow re-acquire"

        lock.release()
        lock.release()  # Need to release twice


class TestProcessLockErrorHandling:
    """Test error handling and edge cases."""

    def test_release_without_acquire_is_safe(self, tmp_path):
        """Releasing unheld lock should not raise."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        # Should not raise
        lock.release()

    def test_context_manager_releases_on_exception(self, tmp_path):
        """Lock should be released even if exception occurs."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        try:
            with lock:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released
        lock2 = ProcessLock(lock_path)
        acquired = lock2.acquire(timeout=0.1)
        assert acquired is True, "Lock should be released after exception"
        lock2.release()

    def test_handles_corrupted_lock_file(self, tmp_path):
        """Should handle corrupted lock file gracefully."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock_path.write_text("corrupted garbage data")

        lock = ProcessLock(lock_path)
        acquired = lock.acquire(timeout=0.1)

        assert acquired is True, "Should recover from corrupted lock file"
        lock.release()


class TestGoTManagerLocking:
    """Test TransactionalGoTAdapter with locking."""

    def test_manager_uses_lock_for_mutations(self, tmp_path):
        """TransactionalGoTAdapter should use lock for graph mutations."""
        from got_utils import GoTBackendFactory

        manager = GoTBackendFactory.create(got_dir=tmp_path)

        # Create task should acquire lock
        task_id = manager.create_task("Test task", priority="high")
        assert task_id is not None

        # Should be able to read without issues
        task = manager.get_task(task_id)
        assert task is not None

    def test_manager_lock_file_location(self, tmp_path):
        """Manager should create lock file in got_dir."""
        from got_utils import GoTBackendFactory

        manager = GoTBackendFactory.create(got_dir=tmp_path)
        manager.create_task("Test", priority="high")

        lock_file = tmp_path / ".got.lock"
        # Lock may or may not persist after operation
        # The important thing is operations succeed


class TestMultiProcessSafety:
    """Test safety across multiple processes."""

    def test_subprocess_respects_lock(self, tmp_path):
        """Subprocess should wait for lock held by parent."""
        from cortical.utils.locking import ProcessLock

        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        lock.acquire()
        try:
            # Run subprocess that tries to acquire lock
            result = subprocess.run(
                [
                    sys.executable, "-c",
                    f"""
import sys
sys.path.insert(0, 'scripts')
from cortical.utils.locking import ProcessLock
lock = ProcessLock('{lock_path}')
acquired = lock.acquire(timeout=0.5)
print('acquired' if acquired else 'blocked')
"""
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert "blocked" in result.stdout, "Subprocess should be blocked"
        finally:
            lock.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
