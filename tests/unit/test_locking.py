"""
Unit tests for cortical/utils/locking.py

Tests ProcessLock for:
- Basic locking/unlocking
- Context manager usage
- Reentrant locking
- Stale lock detection
- Timeout behavior
"""

import json
import os
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from cortical.utils.locking import ProcessLock


class TestProcessLockBasic:
    """Tests for basic ProcessLock functionality."""

    def test_init(self, tmp_path):
        """Initialize lock with defaults."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        assert lock.lock_path == lock_path
        assert lock.reentrant is True
        assert lock.stale_timeout == 3600.0
        assert lock._fd is None
        assert lock._lock_count == 0

    def test_init_custom_params(self, tmp_path):
        """Initialize with custom parameters."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path, reentrant=False, stale_timeout=60.0)

        assert lock.reentrant is False
        assert lock.stale_timeout == 60.0

    def test_acquire_release(self, tmp_path):
        """Acquire and release lock."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        assert lock.acquire()
        assert lock.is_locked()
        assert lock._lock_count == 1

        lock.release()
        assert not lock.is_locked()
        assert lock._lock_count == 0

    def test_context_manager(self, tmp_path):
        """Use lock as context manager."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        with lock:
            assert lock.is_locked()

        assert not lock.is_locked()

    def test_lock_file_created(self, tmp_path):
        """Lock file is created on acquire."""
        lock_path = tmp_path / "subdir" / "test.lock"
        lock = ProcessLock(lock_path)

        assert not lock_path.exists()

        with lock:
            assert lock_path.exists()

    def test_lock_file_contains_pid(self, tmp_path):
        """Lock file contains holder PID."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        with lock:
            content = lock_path.read_text()
            holder_info = json.loads(content)
            assert holder_info["pid"] == os.getpid()
            assert "acquired_at" in holder_info

    def test_release_when_not_locked(self, tmp_path):
        """Release when not locked is a no-op."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        # Should not raise
        lock.release()
        assert lock._lock_count == 0


class TestReentrantLocking:
    """Tests for reentrant lock behavior."""

    def test_reentrant_acquire(self, tmp_path):
        """Same process can acquire lock multiple times when reentrant."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path, reentrant=True)

        try:
            assert lock.acquire()
            assert lock._lock_count == 1

            assert lock.acquire()
            assert lock._lock_count == 2

            assert lock.acquire()
            assert lock._lock_count == 3
        finally:
            # Clean up all acquires
            while lock._lock_count > 0:
                lock.release()

    def test_reentrant_release(self, tmp_path):
        """Multiple releases needed for multiple acquires."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path, reentrant=True)

        lock.acquire()
        lock.acquire()
        lock.acquire()

        lock.release()
        assert lock._lock_count == 2
        assert lock.is_locked()

        lock.release()
        assert lock._lock_count == 1
        assert lock.is_locked()

        lock.release()
        assert lock._lock_count == 0
        assert not lock.is_locked()

    def test_non_reentrant_behavior(self, tmp_path):
        """Non-reentrant lock behavior."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path, reentrant=False)

        try:
            assert lock.acquire()
            # Non-reentrant doesn't allow increment of lock count
            # from the reentrant path, so second acquire goes through
            # flock path which may vary by platform
        finally:
            lock.release()


class TestStaleLockDetection:
    """Tests for stale lock detection and recovery."""

    def test_is_stale_lock_empty_file(self, tmp_path):
        """Empty lock file is considered stale."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("")
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is True

    def test_is_stale_lock_no_pid(self, tmp_path):
        """Lock file without PID is considered stale."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text(json.dumps({"acquired_at": time.time()}))
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is True

    def test_is_stale_lock_dead_process(self, tmp_path):
        """Lock held by dead process is stale."""
        lock_path = tmp_path / "test.lock"
        # Use a PID that definitely doesn't exist
        lock_path.write_text(json.dumps({
            "pid": 999999999,
            "acquired_at": time.time()
        }))
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is True

    def test_is_stale_lock_alive_process(self, tmp_path):
        """Lock held by current process is not stale."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "acquired_at": time.time()
        }))
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is False

    def test_is_stale_lock_old_timestamp(self, tmp_path):
        """Lock older than stale_timeout is stale."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "acquired_at": time.time() - 4000  # Older than default 3600s
        }))
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is True

    def test_is_stale_lock_invalid_json(self, tmp_path):
        """Invalid JSON in lock file is considered stale."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("not valid json {{{")
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is True

    def test_is_stale_lock_file_not_exists(self, tmp_path):
        """Non-existent lock file is not stale."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        assert lock._is_stale_lock() is False


class TestTimeoutBehavior:
    """Tests for timeout-based lock acquisition."""

    def test_acquire_with_zero_timeout(self, tmp_path):
        """Zero timeout means immediate return."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        # Should acquire immediately
        start = time.time()
        result = lock.acquire(timeout=0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 0.1
        lock.release()

    def test_acquire_with_timeout_success(self, tmp_path):
        """Lock acquired within timeout."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        result = lock.acquire(timeout=1.0)
        assert result is True
        lock.release()

    def test_acquire_none_timeout(self, tmp_path):
        """None timeout means single non-blocking attempt."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        result = lock.acquire(timeout=None)
        assert result is True
        lock.release()


class TestErrorHandling:
    """Tests for error handling in ProcessLock."""

    def test_context_manager_acquire_failure(self, tmp_path):
        """Context manager raises on acquire failure."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        # Mock _try_acquire_once to always fail
        with patch.object(lock, '_try_acquire_once', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to acquire lock"):
                with lock:
                    pass

    def test_acquire_with_exception(self, tmp_path):
        """Handle exceptions during acquire."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        # Mock open to raise an exception
        with patch('builtins.open', side_effect=PermissionError("No permission")):
            result = lock.acquire()
            assert result is False
            assert lock._lock_count == 0


class TestIsLocked:
    """Tests for is_locked method."""

    def test_is_locked_false_initially(self, tmp_path):
        """Lock is not held initially."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        assert lock.is_locked() is False

    def test_is_locked_true_after_acquire(self, tmp_path):
        """Lock is held after acquire."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        lock.acquire()
        assert lock.is_locked() is True
        lock.release()

    def test_is_locked_false_after_release(self, tmp_path):
        """Lock is not held after release."""
        lock_path = tmp_path / "test.lock"
        lock = ProcessLock(lock_path)

        lock.acquire()
        lock.release()
        assert lock.is_locked() is False
