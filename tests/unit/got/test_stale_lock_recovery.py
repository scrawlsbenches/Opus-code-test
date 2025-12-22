"""
Tests for stale lock recovery in ProcessLock.

Covers:
- Detection of stale locks from dead processes
- Automatic removal and recovery of stale locks
- Validation that active locks are not removed
- Lock file format and PID tracking
- Warning logs during recovery
"""

import json
import os
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from cortical.utils.locking import ProcessLock


class TestStaleLockRecovery:
    """Test suite for stale lock detection and recovery."""

    @pytest.fixture
    def lock_dir(self, tmp_path):
        """Create temporary directory for lock files."""
        lock_path = tmp_path / "locks"
        lock_path.mkdir()
        return lock_path

    @pytest.fixture
    def lock_file(self, lock_dir):
        """Create path for test lock file."""
        return lock_dir / "test.lock"

    def test_stale_lock_detected_and_removed(self, lock_file):
        """Test that stale lock from dead process is detected and removed."""
        # Create lock file with non-existent PID (99999999 is extremely unlikely to exist)
        fake_pid = 99999999
        holder_info = {
            "pid": fake_pid,
            "acquired_at": time.time()
        }
        lock_file.write_text(json.dumps(holder_info))

        # Verify lock file exists
        assert lock_file.exists()

        # Create ProcessLock instance
        lock = ProcessLock(lock_file)

        # Acquire lock - should detect stale lock and remove it
        acquired = lock.acquire(timeout=1.0)

        # Verify lock was acquired
        assert acquired is True
        assert lock._fd is not None
        assert lock._lock_count == 1

        # Verify lock file now contains current process PID
        with open(lock_file, 'r') as f:
            new_holder_info = json.load(f)
            assert new_holder_info["pid"] == os.getpid()
            assert "acquired_at" in new_holder_info

        # Clean up
        lock.release()

    def test_valid_lock_not_removed(self, lock_file):
        """Test that valid lock with live process PID is not removed."""
        # Create lock with current process PID
        current_pid = os.getpid()
        holder_info = {
            "pid": current_pid,
            "acquired_at": time.time()
        }

        # First acquire the lock normally
        lock1 = ProcessLock(lock_file)
        acquired1 = lock1.acquire(timeout=1.0)
        assert acquired1 is True

        # Try to acquire from another ProcessLock instance (simulating another thread/operation)
        lock2 = ProcessLock(lock_file, reentrant=False)
        acquired2 = lock2.acquire(timeout=0.5)

        # Should fail because lock is held by active process
        assert acquired2 is False

        # Original lock should still be valid
        assert lock1._fd is not None
        assert lock1._lock_count == 1

        # Clean up
        lock1.release()

    def test_reentrant_lock_allows_same_process(self, lock_file):
        """Test that reentrant lock allows same process to acquire multiple times."""
        # Create reentrant lock (default)
        lock = ProcessLock(lock_file, reentrant=True)

        # Acquire first time
        acquired1 = lock.acquire(timeout=1.0)
        assert acquired1 is True
        assert lock._lock_count == 1

        # Acquire second time (same process)
        acquired2 = lock.acquire(timeout=1.0)
        assert acquired2 is True
        assert lock._lock_count == 2

        # Release once
        lock.release()
        assert lock._lock_count == 1

        # Release again
        lock.release()
        assert lock._lock_count == 0

    def test_is_stale_lock_detects_dead_process(self, lock_file):
        """Test that _is_stale_lock correctly identifies dead process."""
        # Create lock file with non-existent PID
        fake_pid = 99999999
        holder_info = {
            "pid": fake_pid,
            "acquired_at": time.time()
        }
        lock_file.write_text(json.dumps(holder_info))

        # Create ProcessLock and check if it's stale
        lock = ProcessLock(lock_file)
        is_stale = lock._is_stale_lock()

        # Should be detected as stale
        assert is_stale is True, "Lock with non-existent PID should be stale"

    def test_is_stale_lock_respects_live_process(self, lock_file):
        """Test that _is_stale_lock respects live process."""
        # Create lock file with current process PID
        current_pid = os.getpid()
        holder_info = {
            "pid": current_pid,
            "acquired_at": time.time()
        }
        lock_file.write_text(json.dumps(holder_info))

        # Create ProcessLock and check if it's stale
        lock = ProcessLock(lock_file)
        is_stale = lock._is_stale_lock()

        # Should NOT be detected as stale (current process is alive)
        assert is_stale is False, "Lock with live PID should not be stale"

    def test_lock_file_contains_pid(self, lock_file):
        """Test that lock file contains current process PID after acquisition."""
        # Acquire lock
        lock = ProcessLock(lock_file)
        acquired = lock.acquire(timeout=1.0)
        assert acquired is True

        # Read lock file
        with open(lock_file, 'r') as f:
            holder_info = json.load(f)

        # Verify contents
        assert "pid" in holder_info
        assert holder_info["pid"] == os.getpid()
        assert "acquired_at" in holder_info
        assert isinstance(holder_info["acquired_at"], (int, float))

        # Clean up
        lock.release()

    def test_empty_lock_file_considered_stale(self, lock_file):
        """Test that empty lock file is considered stale and recovered."""
        # Create empty lock file
        lock_file.write_text("")
        assert lock_file.exists()

        # Acquire lock - should detect stale lock
        lock = ProcessLock(lock_file)

        try:
            acquired = lock.acquire(timeout=1.0)

            # Verify acquisition succeeded
            assert acquired is True

            # Verify lock file now has valid content
            with open(lock_file, 'r') as f:
                content = f.read().strip()
                assert content  # Not empty

            with open(lock_file, 'r') as f:
                holder_info = json.load(f)
                assert holder_info["pid"] == os.getpid()
        finally:
            # Clean up
            if lock._fd is not None:
                lock.release()

    def test_invalid_json_lock_file_considered_stale(self, lock_file):
        """Test that lock file with invalid JSON is considered stale."""
        # Create lock file with invalid JSON
        lock_file.write_text("not valid json{{{")
        assert lock_file.exists()

        # Acquire lock - should detect stale lock
        lock = ProcessLock(lock_file)
        acquired = lock.acquire(timeout=1.0)

        # Verify acquisition succeeded
        assert acquired is True

        # Verify lock file now has valid JSON
        with open(lock_file, 'r') as f:
            holder_info = json.load(f)
            assert holder_info["pid"] == os.getpid()

        # Clean up
        lock.release()

    def test_lock_file_missing_pid_considered_stale(self, lock_file):
        """Test that lock file without PID field is considered stale."""
        # Create lock file with JSON but no PID
        holder_info = {
            "acquired_at": time.time(),
            "hostname": "test-host"
        }
        lock_file.write_text(json.dumps(holder_info))

        # Acquire lock - should detect stale lock
        lock = ProcessLock(lock_file)
        acquired = lock.acquire(timeout=1.0)

        # Verify acquisition succeeded
        assert acquired is True

        # Verify lock file now has valid PID
        with open(lock_file, 'r') as f:
            new_holder_info = json.load(f)
            assert new_holder_info["pid"] == os.getpid()

        # Clean up
        lock.release()

    def test_is_stale_lock_empty_file(self, lock_file):
        """Test that empty lock file is considered stale."""
        # Create empty lock file
        lock_file.write_text("")

        # Check if it's stale
        lock = ProcessLock(lock_file)
        is_stale = lock._is_stale_lock()

        # Should be stale
        assert is_stale is True, "Empty lock file should be stale"

    def test_is_stale_lock_invalid_json(self, lock_file):
        """Test that lock file with invalid JSON is considered stale."""
        # Create lock file with invalid JSON
        lock_file.write_text("not valid json{{{")

        # Check if it's stale
        lock = ProcessLock(lock_file)
        is_stale = lock._is_stale_lock()

        # Should be stale
        assert is_stale is True, "Lock file with invalid JSON should be stale"

    def test_is_stale_lock_missing_pid(self, lock_file):
        """Test that lock file without PID is considered stale."""
        # Create lock file without PID
        holder_info = {
            "acquired_at": time.time(),
            "hostname": "test-host"
        }
        lock_file.write_text(json.dumps(holder_info))

        # Check if it's stale
        lock = ProcessLock(lock_file)
        is_stale = lock._is_stale_lock()

        # Should be stale
        assert is_stale is True, "Lock file without PID should be stale"

    def test_lock_context_manager(self, lock_file):
        """Test that ProcessLock works as context manager."""
        lock = ProcessLock(lock_file)

        # Use as context manager
        with lock:
            # Lock should be acquired
            assert lock._lock_count > 0
            assert lock._fd is not None

            # Lock file should exist and contain PID
            assert lock_file.exists()
            with open(lock_file, 'r') as f:
                holder_info = json.load(f)
                assert holder_info["pid"] == os.getpid()

        # After context, lock should be released
        assert lock._lock_count == 0
        assert lock._fd is None

    def test_multiple_locks_different_files(self, lock_dir):
        """Test that different lock files can be held simultaneously."""
        lock_file1 = lock_dir / "lock1.lock"
        lock_file2 = lock_dir / "lock2.lock"

        lock1 = ProcessLock(lock_file1)
        lock2 = ProcessLock(lock_file2)

        # Acquire both locks
        acquired1 = lock1.acquire(timeout=1.0)
        acquired2 = lock2.acquire(timeout=1.0)

        assert acquired1 is True
        assert acquired2 is True

        # Both should be held
        assert lock1._lock_count == 1
        assert lock2._lock_count == 1

        # Clean up
        lock1.release()
        lock2.release()

    def test_os_kill_check_for_process_existence(self, lock_file):
        """Test that os.kill(pid, 0) is used to check process existence."""
        # First acquire a lock to make the file actually locked
        holder_lock = ProcessLock(lock_file, reentrant=False)
        holder_lock.acquire(timeout=1.0)

        # Write fake PID to the lock file
        fake_pid = 88888888
        holder_info = {
            "pid": fake_pid,
            "acquired_at": time.time()
        }
        with open(lock_file, 'w') as f:
            json.dump(holder_info, f)

        try:
            # Mock os.kill in the tx_manager module
            with patch('cortical.got.tx_manager.os.kill') as mock_kill:
                # Make os.kill raise OSError (process doesn't exist)
                mock_kill.side_effect = OSError("No such process")

                # Try to acquire from another lock instance
                lock = ProcessLock(lock_file, reentrant=False)
                acquired = lock.acquire(timeout=1.0)

                # Should succeed after detecting stale lock and removing it
                assert acquired is True

                # Verify os.kill was called with signal 0 to check existence
                mock_kill.assert_called_with(fake_pid, 0)

                # Clean up new lock
                if lock._fd is not None:
                    lock.release()
        finally:
            # Clean up holder lock
            if holder_lock._fd is not None:
                try:
                    holder_lock.release()
                except:
                    pass

    def test_timeout_behavior_with_stale_lock(self, lock_file):
        """Test that stale lock recovery happens within timeout period."""
        # Create stale lock
        fake_pid = 99999999
        holder_info = {
            "pid": fake_pid,
            "acquired_at": time.time()
        }
        lock_file.write_text(json.dumps(holder_info))

        # Acquire with short timeout
        start_time = time.time()
        lock = ProcessLock(lock_file)
        acquired = lock.acquire(timeout=2.0)
        elapsed = time.time() - start_time

        # Should succeed quickly (stale lock detected and removed immediately)
        assert acquired is True
        assert elapsed < 1.0  # Should be nearly instant, not waiting for timeout

        # Clean up
        lock.release()
