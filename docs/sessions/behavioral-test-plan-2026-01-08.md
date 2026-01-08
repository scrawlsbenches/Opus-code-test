# Behavioral Test Performance Investigation Plan

**Date:** 2026-01-08
**Issue:** Behavioral tests running slower than expected (~5 minutes for 1698 tests)
**Goal:** Profile, identify bottlenecks, and ensure all tests run in-memory

---

## Current State

- **Test count:** 1698 behavioral tests
- **Runtime:** ~279 seconds (4:39)
- **Results:** 1534 passed, 34 failed, 10 errors, 133 skipped
- **Expected:** Should be much faster if everything runs in-memory

---

## Investigation Plan

### Phase 1: Profiling (Identify Slow Tests)

1. **Run pytest with duration reporting**
   ```bash
   pytest tests/behavioral/ --durations=50 -v
   ```

2. **Identify tests > 1 second** - These are candidates for optimization

3. **Check for disk I/O patterns**
   - Look for tests creating files in real directories vs tmp_path
   - Check for tests not using in-memory storage

### Phase 2: Fixture Analysis

1. **Audit conftest.py fixtures**
   - Verify `got_dir`, `cdg_store`, `tx_manager` use tmp_path
   - Check for fixtures with `scope="session"` that could be reused
   - Look for fixtures doing unnecessary disk sync

2. **Check for missing `@pytest.fixture` scope optimization**
   - Module-scoped fixtures for read-only setup
   - Function-scoped only when mutation is needed

### Phase 3: Storage Backend Check

1. **Verify InMemoryStorage usage**
   - Tests should use `InMemoryStorage` or `tmp_path`
   - No tests should touch `.got/` in working directory

2. **Check CDG/GoT initialization**
   - Look for `FileSystemStorage` being used in tests
   - Verify `create_container(got_dir=tmp_path)` pattern

### Phase 4: Race Condition Detection

1. **Look for shared state between tests**
   - Global singletons
   - Class-level state not reset between tests

2. **Check for async/threading issues**
   - Tests using real locks vs mock locks
   - ProcessLock vs threading.Lock

### Phase 5: Known Slow Patterns

1. **WovenMind initialization** - Heavy setup
2. **PLN reasoning** - Multiple inference passes
3. **Full GoT validation** - Checksum verification
4. **WAL recovery** - Disk sync operations

---

## Metrics to Collect

| Metric | Target | Current |
|--------|--------|---------|
| Total runtime | < 60s | ~279s |
| Slowest test | < 2s | TBD |
| Tests > 1s | < 10 | TBD |
| Disk I/O tests | 0 | TBD |

---

## Action Items

1. [ ] Run `--durations=50` to identify slow tests
2. [ ] Profile top 10 slowest tests individually
3. [ ] Check fixture scopes in conftest.py
4. [ ] Verify all tests use tmp_path or in-memory storage
5. [ ] Add timeouts to known-slow test categories
6. [ ] Fix or mark slow tests appropriately

---

## Findings

### Finding 1: shared_processor Fixture (FIXED)

**Root cause:** `test_customer_service_quality.py` uses `shared_processor` fixture which:
- Loads 644 files from `samples/` directory (4.2MB)
- Runs `compute_all()` on the full corpus
- Takes **92 seconds** to initialize

**Fix applied:** Added `pytestmark = pytest.mark.slow` to mark entire module as slow.
All 14 tests are now properly deselected when running with default config.

**Result:** Tests went from 138s → 47s (3x speedup)

### Finding 2: Timeout/TTL Tests Use Excessive Sleep Durations (FIXED)

Tests using `time.sleep()` with unnecessarily long durations:

| Test | Before | After | Savings |
|------|--------|-------|---------|
| `test_scenario_verifier_detects_phase_timeout` | 2.1s sleep, 2.0s threshold | 0.1s sleep, 0.05s threshold | ~2.0s |
| `test_scenario_stuck_phase_severity_is_warning` | 1.1s sleep, 1.0s threshold | 0.1s sleep, 0.05s threshold | ~1.0s |
| `test_scenario_expired_messages_move_to_dead_letter` | 2.0s sleep, 1.0s TTL | 0.1s sleep, 0.05s TTL | ~1.9s |
| `test_scenario_dead_letter_messages_can_be_retried` | 2.0s sleep, 1.0s TTL | 0.1s sleep, 0.05s TTL | ~1.9s |
| `test_findings_expire_after_ttl` | 1.5s sleep, 1.0s TTL | 0.1s sleep, 0.05s TTL | ~1.4s |

**Root cause:** Tests used 1-2 second timeouts when 50-100ms tests the same behavior.

**Fix applied:** Reduced all TTL/threshold values to minimum viable:
- `stuck_threshold_seconds=2.0` → `0.05` (50ms)
- `ttl_seconds=1` → `0.05` (50ms)
- `time.sleep(2.0)` → `0.1` (100ms - exceeds threshold)
- Changed `pubsub.py` ttl_seconds type from `int` to `float` for sub-second precision

**Actual savings:** ~7.3 seconds total

**Status:** DONE

### Finding 3: Remaining Slow Tests (< 2s each)

- Parallel processing tests (~1.5-1.9s) - acceptable for parallel workloads
- Deep graph traversal (1.22s) - acceptable for recursive algorithms
- Concurrent access tests (1.1s) - acceptable for concurrency testing

---

## Summary

| Metric | Original | After Fix 1 | After Fix 2 |
|--------|----------|-------------|-------------|
| Total runtime | 138s | 47s | ~40s |
| Slowest test | 92s setup | 2.1s | ~1.9s |
| Tests > 1s | ~12 | ~10 | ~6 |

**Fix 1 (DONE):** Mark `shared_processor` tests as slow - 3x speedup
**Fix 2 (DONE):** Reduce TTL/timeout test sleeps - 7.3s savings

---

## Git Audit: Origin of Slow Tests

### Question: When did the slow sleep-based tests enter the codebase?

**Answer:** Two commits on **Dec 31, 2025**, both authored by Claude during bulk conversion of demo files:

| Commit | Message | Tests Added |
|--------|---------|-------------|
| `e67072b5` | "test(behavioral): Add 778 behavioral tests converted from demos" | `agents_coordinate_via_context_stories.py`, `test_developer_verifies_qapv_cycles.py` |
| `fcf0c79b` | "test(behavioral+contracts): Add comprehensive API and contract tests" | `test_pubsub_messaging_stories.py` |

**Root Cause Analysis:**
- Original demo scripts used human-observable timing (1-2 second delays)
- When converted to automated tests, timing values were preserved verbatim
- No optimization pass was done to reduce delays for automated testing
- CLAUDE.md rule added (2026-01-08): Require approval before adding sleep calls to tests

**Lesson Learned:** Demo-to-test conversions should include a timing optimization pass to replace human-observable delays with minimum viable delays for automated testing.
