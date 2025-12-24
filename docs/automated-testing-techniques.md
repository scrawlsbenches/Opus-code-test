# Automated Testing Techniques Guide

A comprehensive guide to automated testing strategies, techniques, and workarounds developed through practical experience achieving 98%+ coverage on fault-tolerant systems.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Organization](#test-organization)
3. [Fixture Patterns](#fixture-patterns)
4. [Monkeypatching Techniques](#monkeypatching-techniques)
5. [Testing Race Conditions](#testing-race-conditions)
6. [Testing Exception Handlers](#testing-exception-handlers)
7. [Testing Timing-Dependent Code](#testing-timing-dependent-code)
8. [Coverage Strategies](#coverage-strategies)
9. [Common Patterns and Recipes](#common-patterns-and-recipes)
10. [What's Impractical to Test](#whats-impractical-to-test)

---

## Testing Philosophy

### The TDD Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TDD WORKFLOW                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. RED: Write failing tests first                                   │
│     └── Capture expected behavior before implementation              │
│                                                                       │
│  2. GREEN: Implement minimal code to pass tests                      │
│     └── Focus on making tests pass, not perfection                   │
│                                                                       │
│  3. REFACTOR: Clean up, convert behavioral → unit tests              │
│     └── Delete temp behavioral tests, keep permanent unit tests      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Two Types of Tests

| Type | Purpose | Data Creation | Lifecycle |
|------|---------|---------------|-----------|
| **Behavioral** | Explore, debug, capture "what should happen" | May create real data | Temporary - delete after feature works |
| **Unit** | Permanent protection, document API contract | Mocks only | Keep forever |

### Key Principles

1. **Test behavior, not implementation** - Tests should verify what code does, not how
2. **One assertion per concept** - Each test should verify one logical thing
3. **Tests are documentation** - Test names and structure should explain the code
4. **Fast tests run often** - Slow tests get skipped; keep unit tests under 100ms each

---

## Test Organization

### Directory Structure

```
tests/
├── smoke/                   # Quick sanity checks (<30s total)
├── unit/                    # Fast isolated tests (<1s each)
│   └── got/                 # Grouped by module
│       └── test_*.py
├── integration/             # Component interaction tests
├── performance/             # Timing tests (use small synthetic data)
├── regression/              # Bug-specific tests
├── behavioral/              # User workflow quality tests
└── fixtures/                # Shared test data
```

### Test Class Organization

Group related tests into classes with clear naming:

```python
class TestFeatureBasicFunctionality:
    """Tests for normal operation."""

class TestFeatureEdgeCases:
    """Tests for boundary conditions."""

class TestFeatureErrorHandling:
    """Tests for exception paths."""

class TestFeatureRaceConditions:
    """Tests for concurrent access scenarios."""
```

### Naming Conventions

```python
# Pattern: test_<what>_<condition>_<expected_result>

def test_acquire_lock_when_available_returns_true(self):
    """Lock acquisition succeeds when no contention."""

def test_acquire_lock_when_held_returns_false(self):
    """Lock acquisition fails when another process holds lock."""

def test_acquire_lock_with_stale_lock_recovers_successfully(self):
    """Stale lock is detected and removed before acquisition."""
```

---

## Fixture Patterns

### Session-Scoped Fixtures (Expensive Setup)

```python
@pytest.fixture(scope="session")
def shared_processor():
    """Pre-computed processor for read-only tests."""
    processor = CorticalTextProcessor()
    # Expensive setup done once
    for doc_id, content in load_corpus():
        processor.process_document(doc_id, content)
    processor.compute_all()
    return processor
```

### Function-Scoped Fixtures (Fresh State)

```python
@pytest.fixture
def fresh_processor():
    """Empty processor for isolated tests."""
    return CorticalTextProcessor()

@pytest.fixture
def got_dir(tmp_path):
    """Fresh GoT directory structure."""
    entities_dir = tmp_path / "entities"
    wal_dir = tmp_path / "wal"
    entities_dir.mkdir(parents=True)
    wal_dir.mkdir(parents=True)
    return tmp_path
```

### Parameterized Fixtures

```python
@pytest.fixture(params=["adopt", "delete", "skip"])
def repair_strategy(request):
    """Test all repair strategies."""
    return request.param

def test_repair_orphans_with_strategy(self, got_dir, repair_strategy):
    """Verify each repair strategy works correctly."""
    recovery = RecoveryManager(got_dir)
    result = recovery.repair_orphans(strategy=repair_strategy)
    assert result.success
```

---

## Monkeypatching Techniques

Monkeypatching is the most powerful technique for testing hard-to-reach code paths. It allows you to replace functions, methods, and attributes at runtime.

### Basic Monkeypatch Usage

```python
def test_function_handles_error(self, monkeypatch):
    """Verify error handling when dependency fails."""

    def failing_function(*args, **kwargs):
        raise IOError("Simulated failure")

    monkeypatch.setattr(module, "function_name", failing_function)

    result = code_under_test()
    assert result.handled_error
```

### Patching Built-in Functions

```python
def test_handles_file_open_failure(self, tmp_path, monkeypatch):
    """Test when open() fails."""
    import builtins

    original_open = builtins.open

    def patched_open(path, *args, **kwargs):
        if "target_file" in str(path):
            raise PermissionError("Access denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", patched_open)

    result = function_that_opens_files()
    assert result.failed_gracefully
```

### Patching Instance Methods

```python
def test_method_failure(self, monkeypatch):
    """Test when an object's method fails."""
    from pathlib import Path

    original_unlink = Path.unlink

    def patched_unlink(self, *args, **kwargs):
        if ".lock" in str(self):
            raise PermissionError("Cannot delete")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", patched_unlink)
```

### Conditional Patching (Call Counting)

```python
def test_retry_after_initial_failure(self, monkeypatch):
    """First call fails, retry succeeds."""
    import fcntl

    call_count = [0]
    original_flock = fcntl.flock

    def patched_flock(fd, operation):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call fails
            raise BlockingIOError("Resource busy")
        # Subsequent calls succeed
        return original_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", patched_flock)

    result = acquire_with_retry()
    assert result.success
    assert call_count[0] >= 2
```

### Patching with State Tracking

```python
def test_cleanup_called_on_failure(self, monkeypatch):
    """Verify cleanup happens even when operation fails."""

    cleanup_called = [False]

    def patched_cleanup():
        cleanup_called[0] = True

    monkeypatch.setattr(module, "cleanup", patched_cleanup)

    with pytest.raises(SomeError):
        operation_that_fails()

    assert cleanup_called[0], "Cleanup was not called"
```

---

## Testing Race Conditions

Race conditions are notoriously hard to test because they depend on timing. Here are techniques to simulate them deterministically.

### Simulating File Vanishing

```python
def test_file_vanishes_during_read(self, got_dir, monkeypatch):
    """File deleted between existence check and read."""

    # Create file
    target_file = got_dir / "entities" / "task.json"
    target_file.write_text('{"id": "T-1"}')

    # Patch read to simulate file vanishing
    def patched_read_and_verify(path):
        raise FileNotFoundError(f"File vanished: {path}")

    monkeypatch.setattr(
        recovery_module,
        "_read_and_verify",
        patched_read_and_verify
    )

    recovery = RecoveryManager(got_dir)
    result = recovery.verify_store_integrity()

    # Should handle gracefully, not crash
    assert result is not None
```

### Simulating Lock Contention

```python
def test_lock_held_by_another_process(self, tmp_path, monkeypatch):
    """Lock acquisition when another process holds the lock."""
    import fcntl

    def always_busy(fd, operation):
        if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
            raise BlockingIOError("Resource temporarily unavailable")

    monkeypatch.setattr(fcntl, "flock", always_busy)

    lock = ProcessLock(tmp_path / ".lock")
    result = lock.acquire(timeout=0.1)

    assert result is False
```

### Simulating Stale Lock Recovery

```python
def test_stale_lock_recovery(self, tmp_path, monkeypatch):
    """Detect and recover from stale lock."""
    import fcntl

    lock_path = tmp_path / ".lock"

    # Create stale lock (dead process, old timestamp)
    lock_path.write_text('{"pid": 999999999, "acquired_at": 0}')

    flock_calls = [0]
    original_flock = fcntl.flock

    def patched_flock(fd, operation):
        flock_calls[0] += 1
        if flock_calls[0] == 1:
            # First attempt fails (simulates held lock)
            raise BlockingIOError("Resource busy")
        # After stale removal, succeeds
        return original_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", patched_flock)

    lock = ProcessLock(lock_path, stale_timeout=1.0)
    result = lock.acquire()

    assert result is True
    lock.release()
```

### Testing Concurrent Operations

```python
def test_concurrent_lock_acquisition(self, tmp_path):
    """Two processes competing for lock."""
    import threading
    import time

    lock_path = tmp_path / ".lock"
    lock1 = ProcessLock(lock_path)
    lock2 = ProcessLock(lock_path)

    # Lock 1 holds the lock
    assert lock1.acquire() is True

    # Release after delay
    def release_soon():
        time.sleep(0.1)
        lock1.release()

    thread = threading.Thread(target=release_soon)
    thread.start()

    # Lock 2 waits and eventually acquires
    result = lock2.acquire(timeout=0.5)
    thread.join()

    assert result is True
    lock2.release()
```

---

## Testing Exception Handlers

Exception handlers are often the hardest code to reach. Here are patterns for testing them.

### Testing Simple Exception Handlers

```python
def test_handles_json_decode_error(self, got_dir):
    """Corrupted JSON is handled gracefully."""

    wal_file = got_dir / "wal" / "current.wal"
    wal_file.write_text("not valid json {{{")

    recovery = RecoveryManager(got_dir)
    result = recovery.needs_recovery()

    # Should detect corruption, not crash
    assert result is True
```

### Testing Nested Exception Handlers

Nested exception handlers (exception inside exception handler) require two things to fail:

```python
def test_nested_exception_handling(self, tmp_path, monkeypatch):
    """Exception handler itself fails."""
    import fcntl
    import io

    # Step 1: Make the main operation fail
    def patched_flock(fd, operation):
        # After this, we'll also make close() fail
        monkeypatch.setattr(
            io.IOBase,
            "close",
            lambda self: (_ for _ in ()).throw(OSError("Cannot close"))
        )
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(fcntl, "flock", patched_flock)

    lock = ProcessLock(tmp_path / ".lock")
    result = lock.acquire()

    # Should handle both failures gracefully
    assert result is False
```

### Testing Exception Propagation

```python
def test_critical_exceptions_propagate(self, got_dir):
    """Some exceptions should not be swallowed."""

    recovery = RecoveryManager(got_dir)

    # KeyboardInterrupt should propagate
    with pytest.raises(KeyboardInterrupt):
        with patch.object(recovery, 'verify', side_effect=KeyboardInterrupt):
            recovery.recover()
```

---

## Testing Timing-Dependent Code

### Testing Timeouts

```python
def test_timeout_exceeded(self, tmp_path, monkeypatch):
    """Operation times out correctly."""
    import fcntl
    import time

    def slow_flock(fd, operation):
        if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
            time.sleep(0.1)  # Delay exceeds timeout
            raise BlockingIOError("Busy")

    monkeypatch.setattr(fcntl, "flock", slow_flock)

    lock = ProcessLock(tmp_path / ".lock")

    start = time.time()
    result = lock.acquire(timeout=0.05)
    elapsed = time.time() - start

    assert result is False
    assert elapsed < 0.2  # Should not hang
```

### Testing Exponential Backoff

```python
def test_exponential_backoff(self, tmp_path, monkeypatch):
    """Backoff increases between retries."""
    import fcntl
    import time

    attempt_times = []

    def tracking_flock(fd, operation):
        attempt_times.append(time.time())
        if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
            raise BlockingIOError("Busy")

    monkeypatch.setattr(fcntl, "flock", tracking_flock)

    lock = ProcessLock(tmp_path / ".lock")
    lock.acquire(timeout=0.5)

    # Verify backoff is increasing
    if len(attempt_times) >= 3:
        gap1 = attempt_times[1] - attempt_times[0]
        gap2 = attempt_times[2] - attempt_times[1]
        assert gap2 > gap1, "Backoff should increase"
```

### Testing with Controlled Time

```python
def test_stale_timeout_detection(self, tmp_path, monkeypatch):
    """Lock older than stale_timeout is detected."""
    import time

    lock_path = tmp_path / ".lock"

    # Create lock with old timestamp
    old_time = time.time() - 3600  # 1 hour ago
    lock_path.write_text(f'{{"pid": {os.getpid()}, "acquired_at": {old_time}}}')

    lock = ProcessLock(lock_path, stale_timeout=60)  # 60 second timeout

    assert lock._is_stale_lock() is True
```

---

## Coverage Strategies

### Targeting Uncovered Lines

1. **Run coverage with `--show-missing`**:
   ```bash
   coverage run -m pytest tests/ && coverage report --show-missing
   ```

2. **Understand the notation**:
   - `42` = Line 42 not executed
   - `42->45` = Branch from line 42 to 45 not taken

3. **Read the source to understand what triggers each path**

### Common Coverage Gaps and Solutions

| Gap Type | Example | Solution |
|----------|---------|----------|
| Early return | `if not file.exists(): return []` | Test with missing file |
| Exception handler | `except IOError: log(e)` | Monkeypatch to raise IOError |
| Empty collection | `if not items: continue` | Test with empty input |
| Platform branch | `if sys.platform == 'win32':` | Remove or mark as no-cover |
| Fallback value | `value = cache.get(k) or compute()` | Test with cache miss |

### Practical Coverage Limits

Some code is impractical to cover:

```python
# Platform-specific (can't test Windows on Linux)
if sys.platform == 'win32':  # pragma: no cover
    import msvcrt

# Defensive code that "can't happen"
if self._fd is None:  # pragma: no cover
    raise RuntimeError("Invalid state")

# Exception inside exception handler
try:
    cleanup()
except Exception:  # pragma: no cover
    pass  # Best effort cleanup
```

---

## Common Patterns and Recipes

### Testing File Operations

```python
def test_atomic_write(self, tmp_path):
    """Atomic write creates file correctly."""
    target = tmp_path / "output.json"

    atomic_write(target, {"key": "value"})

    assert target.exists()
    assert json.loads(target.read_text()) == {"key": "value"}

def test_atomic_write_rollback_on_failure(self, tmp_path, monkeypatch):
    """Failed write doesn't corrupt existing file."""
    target = tmp_path / "output.json"
    target.write_text('{"original": true}')

    # Make rename fail
    def failing_rename(src, dst):
        raise OSError("Disk full")

    monkeypatch.setattr(os, "rename", failing_rename)

    with pytest.raises(OSError):
        atomic_write(target, {"new": "data"})

    # Original file unchanged
    assert json.loads(target.read_text()) == {"original": True}
```

### Testing Logging Output

```python
def test_error_is_logged(self, caplog):
    """Errors are logged with appropriate level."""
    import logging

    with caplog.at_level(logging.ERROR):
        function_that_logs_error()

    assert "error message" in caplog.text
    assert any(r.levelno == logging.ERROR for r in caplog.records)
```

### Testing Configuration

```python
@pytest.fixture
def configured_service(monkeypatch):
    """Service with test configuration."""
    monkeypatch.setenv("SERVICE_URL", "http://test.local")
    monkeypatch.setenv("SERVICE_TIMEOUT", "5")
    return Service()

def test_uses_configured_timeout(self, configured_service):
    assert configured_service.timeout == 5
```

### Testing Cleanup/Teardown

```python
def test_cleanup_on_exception(self, tmp_path):
    """Resources are cleaned up even on failure."""
    lock_file = tmp_path / ".lock"

    try:
        with ProcessLock(lock_file) as lock:
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Lock should be released
    assert not lock.is_locked()
```

---

## What's Impractical to Test

### Accept These Coverage Gaps

1. **Platform-specific code you can't run**:
   ```python
   if sys.platform == 'win32':  # Can't test on Linux
   ```

2. **Defensive assertions that "can't happen"**:
   ```python
   if self._state is None:
       raise RuntimeError("Should never happen")
   ```

3. **Nested exception handlers** (exception inside exception):
   ```python
   except Exception:
       try:
           cleanup()
       except Exception:  # Very hard to trigger
           pass
   ```

4. **Race condition edge cases** that depend on exact timing:
   ```python
   # Sleep time becomes negative due to timing
   sleep_time = timeout - elapsed
   if sleep_time > 0:
       time.sleep(sleep_time)
   ```

5. **Third-party library internals**

### Use `# pragma: no cover` Sparingly

```python
def handle_signal(signum, frame):  # pragma: no cover
    """Signal handlers are hard to test."""
    cleanup_and_exit()

if __name__ == "__main__":  # pragma: no cover
    main()
```

---

## Quick Reference

### Pytest Commands

```bash
# Run with coverage
coverage run -m pytest tests/ && coverage report --show-missing

# Run specific test
pytest tests/unit/test_file.py::TestClass::test_method -v

# Run with output
pytest -v --tb=short

# Parallel execution
pytest -n 4

# Stop on first failure
pytest -x
```

### Coverage Targets

| Module Type | Target | Rationale |
|-------------|--------|-----------|
| Core logic | 95%+ | Critical paths must be tested |
| Error handling | 90%+ | Most exception paths covered |
| Utilities | 85%+ | Good coverage, some edge cases ok |
| CLI/Entry points | 70%+ | Hard to unit test, integration tests help |

### Monkeypatch Cheatsheet

```python
# Patch module attribute
monkeypatch.setattr(module, "attr", value)

# Patch class method
monkeypatch.setattr(ClassName, "method", new_method)

# Patch environment variable
monkeypatch.setenv("VAR", "value")

# Patch dict item
monkeypatch.setitem(some_dict, "key", "value")

# Patch built-in
import builtins
monkeypatch.setattr(builtins, "open", patched_open)
```

---

## Summary

1. **Organize tests by behavior**, not implementation
2. **Use fixtures** for setup, monkeypatch for simulation
3. **Race conditions** can be tested with call counting and conditional failures
4. **Exception handlers** need the exception to be raised
5. **Timing code** can be tested with delays in mocks
6. **Accept 95-98% as excellent** - the last 2% is often impractical
7. **Document why** certain code isn't covered with `# pragma: no cover`

The goal isn't 100% coverage - it's **confidence that the code works correctly**.
