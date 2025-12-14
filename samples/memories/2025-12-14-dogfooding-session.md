# Memory Entry: 2025-12-14 Dog-Fooding Session

**Tags:** `security`, `testing`, `fuzzing`, `dog-fooding`
**Related:** [[../decisions/adr-microseconds-task-id.md]], [[2025-12-13-security-review.md]]

---

## Context

Resumed dog-fooding the Cortical Text Processor. Goal was to use the system to test itself and complete pending security tasks.

## What I Learned

### 1. Hypothesis Fuzzing Finds Real Bugs

Property-based testing with Hypothesis discovered that `CorticalConfig` accepted `NaN` and `infinity` for `louvain_resolution`. The validation check:

```python
if self.louvain_resolution <= 0:  # BUG: NaN comparisons are always False!
```

**Fix:** Added explicit checks:
```python
if math.isnan(self.louvain_resolution) or math.isinf(self.louvain_resolution):
    raise ValueError(...)
```

**Lesson:** Fuzzing with extreme values (NaN, inf, empty strings, unicode) catches edge cases that manual tests miss.

### 2. Timestamp Precision Matters

Task ID generation was using seconds precision:
```
T-20251214-163052-a1b2  # Only unique per second
```

When generating 100 IDs in a tight loop, collisions occurred (~7% probability via birthday paradox with 65,536 possible suffixes).

**Fix:** Added microseconds:
```
T-20251214-163052123456-a1b2  # Unique per microsecond
```

### 3. Semantic Search Has Blind Spots

Searching for "security test fuzzing" returned staleness tests instead of actual security code. The search seems to over-weight common terms like "test".

**Created task:** T-20251214-171301-6aa8-001 to investigate.

## Connections Made

- **Fuzzing → Validation**: Property-based testing is essential for numeric validation
- **Timestamps → Uniqueness**: Sub-second precision needed for concurrent operations
- **Search → Relevance**: Domain-specific term weighting improves results

## Emotional State

Satisfying session. Finding a real bug through fuzzing validated the investment in security testing. The birthday paradox collision was a nice teachable moment about probability.

## Future Exploration

- [ ] Apply NaN/inf checks to other float config parameters
- [ ] Investigate TF-IDF weighting for common programming terms
- [ ] Consider security-specific synonym expansion

## Artifacts Created

- `tests/security/test_security.py` (22 tests)
- `tests/security/test_fuzzing.py` (17 Hypothesis tests)
- Task: T-20251214-171301-6aa8-001

---

*Committed to memory at: 2025-12-14T17:15:00Z*
