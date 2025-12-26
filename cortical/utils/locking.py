"""
Process-safe file-based locking utilities.

Platform Support:
    This module requires POSIX systems (Linux, macOS). Windows is not supported.

Provides ProcessLock for cross-process synchronization with:
- Stale lock detection and recovery
- Timeout support with exponential backoff
- Reentrant locking option
- Thread-safe implementation

Logging:
    This module uses Python's standard logging. Configure via:

        import logging
        logging.getLogger('cortical.utils.locking').setLevel(logging.DEBUG)

    Log levels:
    - DEBUG: Lock acquisition attempts, stale lock checks
    - INFO: Lock acquired/released
    - WARNING: Stale lock recovery
    - ERROR: Lock failures
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

# Module-level logger - configure via logging.getLogger('cortical.utils.locking')
logger = logging.getLogger(__name__)


class ProcessLock:
    """
    Simple file-based lock for process safety.

    Uses fcntl.flock() on POSIX systems (Linux, macOS).
    Provides context manager interface for safe lock management.

    Features:
    - Cross-process locking via fcntl.flock
    - Stale lock detection (detects dead processes)
    - Timeout support with exponential backoff
    - Reentrant locking (same process can re-acquire)
    - Thread-safe implementation

    Usage:
        lock = ProcessLock(Path("/path/to/.lock"))
        with lock:
            # Critical section protected across processes
            pass

        # Or with explicit timeout:
        if lock.acquire(timeout=5.0):
            try:
                # Critical section
            finally:
                lock.release()
    """

    def __init__(self, lock_path: Path, reentrant: bool = True, stale_timeout: float = 3600.0):
        """
        Initialize process lock.

        Args:
            lock_path: Path to lock file
            reentrant: If True, same process can acquire multiple times
            stale_timeout: Seconds after which a lock is considered stale (default: 1 hour)
        """
        self.lock_path = Path(lock_path)
        self.reentrant = reentrant
        self.stale_timeout = stale_timeout
        self._fd = None
        self._lock_count = 0
        self._thread_lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock with timeout and stale lock recovery.

        Args:
            timeout: Timeout in seconds. If None, single non-blocking attempt.
                    If provided, retry with exponential backoff until timeout.

        Returns:
            True if lock acquired, False otherwise
        """
        with self._thread_lock:
            # Reentrant check
            if self.reentrant and self._lock_count > 0:
                self._lock_count += 1
                return True

            # Create lock file if needed
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)

            # Timeout=None: single attempt (backward compatible)
            if timeout is None:
                return self._try_acquire_once()

            # Timeout specified: retry with exponential backoff
            start_time = time.time()
            backoff_ms = 10  # Start with 10ms
            max_backoff_ms = 500  # Cap at 500ms

            while True:
                # Try to acquire
                if self._try_acquire_once():
                    return True

                # Check if timeout exceeded
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

                # Exponential backoff, capped at max_backoff
                sleep_time = min(backoff_ms / 1000.0, timeout - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Double backoff for next iteration, cap at max
                backoff_ms = min(backoff_ms * 2, max_backoff_ms)

                # Check again if we've exceeded timeout after sleep
                if time.time() - start_time >= timeout:
                    return False

    def _try_acquire_once(self) -> bool:
        """
        Try to acquire lock once (non-blocking).

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            # Open file for locking
            self._fd = open(self.lock_path, 'r+' if self.lock_path.exists() else 'w+')

            # Try to acquire exclusive lock (non-blocking)
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                # Lock held by another process - check if stale
                self._fd.close()
                self._fd = None

                if self._is_stale_lock():
                    logger.warning(
                        f"Detected stale lock at {self.lock_path}, recovering..."
                    )
                    # Remove stale lock file and retry
                    try:
                        self.lock_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.error(f"Failed to remove stale lock: {e}")
                        return False

                    # Retry acquisition after removing stale lock
                    try:
                        self._fd = open(self.lock_path, 'w+')
                        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (IOError, OSError):
                        if self._fd:
                            self._fd.close()
                            self._fd = None
                        return False
                else:
                    return False

            # Successfully acquired - write holder info
            # Note: We don't fsync here because holder info is just metadata
            # for stale detection, not critical data. This avoids unnecessary
            # syscalls and respects RELAXED durability modes.
            try:
                holder_info = {
                    "pid": os.getpid(),
                    "acquired_at": time.time()
                }
                self._fd.seek(0)
                self._fd.truncate()
                json.dump(holder_info, self._fd)
                self._fd.flush()
            except Exception as e:
                logger.error(f"Failed to write lock holder info: {e}")
                # Don't fail the lock acquisition if we can't write metadata

            self._lock_count = 1
            return True

        except Exception as e:
            # Catch all other exceptions and return False
            logger.error(f"Unexpected error in lock acquisition: {e}")
            if self._fd:
                try:
                    self._fd.close()
                except Exception:
                    pass
                self._fd = None
            return False

    def _is_stale_lock(self) -> bool:
        """
        Check if lock file is stale (held by dead process or too old).

        Returns:
            True if lock is stale and can be stolen, False otherwise
        """
        try:
            # Read lock file to get holder PID
            if not self.lock_path.exists():
                return False

            with open(self.lock_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    # Empty lock file - consider stale
                    return True

                holder_info = json.loads(content)
                holder_pid = holder_info.get("pid")
                acquired_at = holder_info.get("acquired_at")

                if holder_pid is None:
                    # No PID in lock file - consider stale
                    return True

                # Check if lock is too old
                if acquired_at and (time.time() - acquired_at > self.stale_timeout):
                    return True

                # Check if process is still alive
                try:
                    # os.kill(pid, 0) doesn't send signal, just checks if process exists
                    os.kill(holder_pid, 0)
                    # Process exists - lock is not stale
                    return False
                except OSError:
                    # Process doesn't exist - lock is stale
                    return True

        except json.JSONDecodeError:
            # Invalid JSON - consider stale
            logger.warning(f"Lock file {self.lock_path} has invalid JSON")
            return True
        except Exception as e:
            # Any other error - don't assume stale
            logger.error(f"Error checking stale lock: {e}")
            return False

    def release(self) -> None:
        """Release the lock."""
        with self._thread_lock:
            if self._lock_count == 0:
                return

            if self.reentrant and self._lock_count > 1:
                self._lock_count -= 1
                return

            if self._fd:
                try:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                except (IOError, OSError):
                    pass
                self._fd.close()
                self._fd = None

            self._lock_count = 0

    def is_locked(self) -> bool:
        """
        Check if lock is currently held by this instance.

        Returns:
            True if lock is held, False otherwise
        """
        return self._lock_count > 0

    def __enter__(self) -> ProcessLock:
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError(f"Failed to acquire lock: {self.lock_path}")
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.release()
