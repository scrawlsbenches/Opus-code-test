# Experiment: exp-20260107-201100-count-min-sketch

## Algorithm
**Name:** Count-Min Sketch for Comment Pattern Frequency
**Expected complexity:** O(d) update and query where d = depth (number of hash functions)
**Required operations:**
- `add(pattern: str, count: int)` - Add count for pattern
- `query(pattern: str) -> int` - Estimate count for pattern (may overestimate)
- `merge(other) -> CountMinSketch` - Merge two sketches

## Codebase Application

**Problem:** We want to track frequency of comment patterns across the codebase without storing all patterns:
- How often does "will be" appear in comments?
- What's the frequency of "FUTURE:" markers?
- Which patterns are "heavy hitters" (appear very frequently)?

Count-Min Sketch provides sub-linear space frequency estimation:
- Never underestimates (query >= actual)
- May slightly overestimate due to hash collisions
- Can merge sketches from different code modules

**Use Case:** Stream through all comments in `cortical/`, counting pattern occurrences efficiently.

**Reference:** This complements the Trie (which stores exact patterns) by providing approximate counts in bounded space.

## Hypothesis
**I expect:** The agent will implement a basic Count-Min Sketch correctly
**Because:** The algorithm is conceptually simple (d hash functions, w-wide arrays, take minimum). The challenge is understanding the error bounds.

## Task Prompt (Given to Agent)

```
Implement a Count-Min Sketch for tracking comment pattern frequencies in the Cortical codebase.

Context: We're streaming through thousands of comments looking for patterns:
- "will be" - speculation marker
- "FUTURE:" - future promise
- "TODO:" - actionable item
- "See:" - reference

We need sub-linear space to track frequencies without storing every pattern.

Count-Min Sketch properties:
- Never underestimates (query >= actual count)
- May overestimate due to hash collisions
- Uses O(d * w) space regardless of number of distinct patterns

Requirements:
1. NO external libraries except typing, math, and hashlib
2. Must use multiple hash functions (depth d)
3. Must use double-hashing technique for deterministic hash generation
4. Must handle these operations:

from typing import List
import hashlib
import math

class PatternFrequencySketch:
    def __init__(self, width: int, depth: int):
        """
        Initialize Count-Min Sketch for pattern frequency.
        width: Number of counters per row (w) - more = less collision
        depth: Number of hash functions/rows (d) - more = higher accuracy

        Error bounds:
        - Overestimate by at most N/w with probability 1 - (1/2)^d
        - where N = total count of all items

        Hash function generation:
        Use double hashing: h_i(x) = (hash1(x) + i * hash2(x)) % width
        where:
        - hash1(x) = int(md5(x).hexdigest()[:8], 16)
        - hash2(x) = int(md5(x).hexdigest()[8:16], 16)
        """
        self._width = width
        self._depth = depth
        self._counters: List[List[int]] = [[0] * width for _ in range(depth)]
        self._total_count = 0

    def _hash(self, item: str, row: int) -> int:
        """
        Generate hash for item at given row using double hashing.
        h_row(item) = (hash1(item) + row * hash2(item)) % width

        Use MD5 for consistent cross-platform behavior:
        - hash1 = first 8 hex digits of md5
        - hash2 = next 8 hex digits of md5
        """
        pass

    def add(self, pattern: str, count: int = 1) -> None:
        """
        Add count for pattern. Updates all d rows.
        For each row i:
            counters[i][hash_i(pattern)] += count
        """
        pass

    def query(self, pattern: str) -> int:
        """
        Estimate count for pattern.
        Returns minimum across all d rows (never underestimates).
        min(counters[i][hash_i(pattern)] for i in range(depth))
        """
        pass

    def merge(self, other: 'PatternFrequencySketch') -> 'PatternFrequencySketch':
        """
        Merge two sketches with same dimensions.
        Returns new sketch with combined counts.
        Raises ValueError if dimensions don't match.
        """
        pass

    @property
    def total_count(self) -> int:
        """Return total number of items added (sum of all counts)."""
        return self._total_count

    def heavy_hitters(self, threshold_fraction: float = 0.01) -> List[str]:
        """
        Note: CMS cannot enumerate heavy hitters by itself.
        This method is a placeholder - in practice you'd track
        candidates separately and verify with query().

        For this experiment, we'll skip this method.
        """
        raise NotImplementedError("CMS cannot enumerate - track candidates separately")

Test cases using REAL comment patterns:

# Test 1: Basic add and query with audit patterns
cms = PatternFrequencySketch(width=1000, depth=5)

# Real patterns from our audit
cms.add("FUTURE:", 10)  # 10 misleading FUTURE comments
cms.add("TODO:", 5)      # 5 accurate TODO comments
cms.add("See:", 8)       # 8 reference comments
cms.add("will be", 15)   # 15 speculation patterns

assert cms.query("FUTURE:") >= 10, "Never underestimates"
assert cms.query("TODO:") >= 5
assert cms.query("See:") >= 8
assert cms.query("will be") >= 15
assert cms.query("missing") >= 0  # Unknown returns 0 or small overestimate

# Test 2: Multiple adds accumulate
cms = PatternFrequencySketch(width=1000, depth=5)
cms.add("will be", 5)
cms.add("will be", 3)
cms.add("will be", 2)
assert cms.query("will be") >= 10

# Test 3: Estimates are reasonably accurate with large width
cms = PatternFrequencySketch(width=10000, depth=7)
patterns = {
    "FUTURE:": 100,
    "TODO:": 50,
    "FIXME:": 25,
    "NOTE:": 10,
}
for pattern, count in patterns.items():
    cms.add(pattern, count)

# With large width, estimates should be close to actual
for pattern, actual in patterns.items():
    estimate = cms.query(pattern)
    assert estimate >= actual, f"Underestimate for {pattern}: {estimate} < {actual}"
    assert estimate <= actual * 1.5, f"Overestimate for {pattern}: {estimate} > {actual * 1.5}"

# Test 4: Total count tracking
cms = PatternFrequencySketch(width=100, depth=3)
cms.add("a", 10)
cms.add("b", 20)
cms.add("c", 30)
assert cms.total_count == 60

# Test 5: Merge sketches from different modules
cms1 = PatternFrequencySketch(width=100, depth=3)
cms2 = PatternFrequencySketch(width=100, depth=3)

# Module 1 has these patterns
cms1.add("FUTURE:", 5)
cms1.add("TODO:", 3)

# Module 2 has these patterns
cms2.add("FUTURE:", 3)
cms2.add("See:", 4)

merged = cms1.merge(cms2)
assert merged.query("FUTURE:") >= 8  # 5 + 3
assert merged.query("TODO:") >= 3
assert merged.query("See:") >= 4
assert merged.total_count == cms1.total_count + cms2.total_count

# Test 6: Merge dimension mismatch
cms1 = PatternFrequencySketch(width=100, depth=3)
cms2 = PatternFrequencySketch(width=200, depth=3)  # Different width
try:
    cms1.merge(cms2)
    assert False, "Should raise ValueError for dimension mismatch"
except ValueError:
    pass  # Expected

# Test 7: High collision scenario (small width)
cms = PatternFrequencySketch(width=10, depth=3)
# Add many different patterns - will have collisions
for i in range(100):
    cms.add(f"pattern_{i}", 1)

# Estimates will be inflated due to collisions
# But minimum over depths helps reduce this
estimate = cms.query("pattern_0")
assert estimate >= 1, "Never underestimates"
# With small width (10) and 100 items, expect overestimate
# But depth=3 should help keep it reasonable

# Test 8: Real audit scenario - streaming comment analysis
cms = PatternFrequencySketch(width=1000, depth=5)

# Simulate streaming through all comments in cortical/
comment_patterns = [
    ("FUTURE:", 1),
    ("will be", 1),
    ("FUTURE:", 1),
    ("TODO:", 1),
    ("will be", 1),
    ("See:", 1),
    ("FUTURE:", 1),
    ("will be", 1),
    ("FIXME:", 1),
    ("will be", 1),
    ("FUTURE:", 1),
    ("See:", 1),
]

for pattern, count in comment_patterns:
    cms.add(pattern, count)

# Query frequencies for audit report
assert cms.query("FUTURE:") >= 4  # Should be at least 4
assert cms.query("will be") >= 4   # Should be at least 4
assert cms.query("TODO:") >= 1
assert cms.query("FIXME:") >= 1

# "will be" is the heavy hitter (speculation pattern)
speculation_count = cms.query("will be")
todo_count = cms.query("TODO:")
assert speculation_count > todo_count, "Speculation pattern should be more frequent"

# Test 9: Deterministic behavior
cms = PatternFrequencySketch(width=100, depth=3)
cms.add("test", 5)
result1 = cms.query("test")
result2 = cms.query("test")
assert result1 == result2, "Query should be deterministic"

Write the complete implementation with comments.
Include explanation of:
- Why we take minimum across rows (reduces collision impact)
- Error bounds formula
- Double hashing technique for hash independence
- When merging is useful (distributed counting)
```

## Success Criteria
- [ ] All 9 test cases pass
- [ ] Never underestimates (query >= actual)
- [ ] Multiple hash functions used (depth d)
- [ ] Double hashing implemented correctly
- [ ] Merge works correctly
- [ ] Deterministic behavior

## Failure Criteria
- [ ] Underestimates (query < actual)
- [ ] Only one hash function (no depth)
- [ ] Takes maximum instead of minimum
- [ ] Merge dimensions mismatch not handled
- [ ] Uses external libraries
- [ ] Non-deterministic results

## Prediction
Before running: **PASS**
Confidence: **HIGH**
Reasoning: Count-Min Sketch is straightforward: d arrays, hash to each, increment, query takes min. The algorithm is simpler than Bloom filters. The double-hashing specification makes implementation clear.

## Actual Result
Status: PASS
Operations implemented: 4/4 (add, query, merge, total_count property)
Tests passed: 9/9 (all required tests + 10/10 bonus edge cases)
Notes: Never underestimates (query >= actual always). Double-hashing h_i(x)=(hash1+i×hash2)%width generates d independent hash functions from one MD5. Minimum across rows gives best estimate (least collision). With width=10000, depth=7: estimates are exactly accurate.

## Agent Output
```python
# Complete implementation at: /home/user/Opus-code-test/pattern_frequency_sketch.py
# Key achievements:
# - Double hashing: hash1=md5[:8], hash2=md5[8:16]
# - Never underestimates: every add increments ALL rows, min preserves this
# - Merge: Linear sketch property allows element-wise addition
# - Error bound: overestimate <= N/w with probability 1-(1/2)^d
# - Real-world: Track pattern frequencies in sub-linear space
```

## Test Results
```
All 9/9 required tests PASSED + 10/10 bonus edge cases:
✅ Test 1: Basic add and query (FUTURE:=10, TODO:=5, See:=8, will be=15)
✅ Test 2: Multiple adds accumulate correctly
✅ Test 3: Estimates accurate with large width (exact with width=10000)
✅ Test 4: Total count tracking (60 total)
✅ Test 5: Merge sketches from different modules
✅ Test 6: Merge dimension mismatch raises ValueError
✅ Test 7: High collision scenario (small width=10, 100 patterns)
✅ Test 8: Real audit scenario (speculation patterns detected)
✅ Test 9: Deterministic behavior (same query = same result)
Bonus: Empty strings, Unicode, zero counts, extreme collision, etc. all handled
```

## Analysis
**Discrepancy:** None - prediction matched outcome (PASS)
**Root cause:** N/A
**Learning:** Taking MINIMUM across rows (not maximum) gives best estimate because it selects the row with least collision. Why never underestimates: Every add operation increments counters in ALL rows, so actual count always contributes. Even with collisions, counter[i][hash_i(P)] >= actual_count(P) for every row i. Minimum preserves this guarantee. Mergeable property: CMS is linear sketch, so CMS(stream1+stream2) = CMS(stream1).merge(CMS(stream2)).

## Integration Plan

After successful implementation:
1. Stream through all comments in `cortical/` directory
2. Track pattern frequencies in sub-linear space
3. Merge sketches from different modules
4. Report heavy hitters (most frequent patterns) for audit focus
