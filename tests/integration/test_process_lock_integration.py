"""
Integration tests for process-safe locking with actual subprocesses.

Moved from tests/unit/test_process_lock.py::TestMultiProcessSafety
because it spawns actual subprocesses.
"""

import sys
import subprocess
import pytest


class TestMultiProcessSafety:
    """Test safety across multiple processes.

    These tests spawn actual subprocesses to verify cross-process
    locking behavior, making them integration tests.
    """

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
