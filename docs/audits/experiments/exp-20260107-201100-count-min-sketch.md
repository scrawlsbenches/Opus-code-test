# Experiment: exp-20260107-201100-count-min-sketch

## Algorithm
**Name:** Count-Min Sketch
**Expected complexity:** O(d) update and query where d = depth (number of hash functions)
**Required operations:**
- `add(item: str, count: int)` - Add count for item
- `query(item: str) -> int` - Estimate count for item (may overestimate)
- `heavy_hitters(threshold: int) -> List[str]` - Find items above threshold

## Hypothesis
**I expect:** The agent will implement a basic Count-Min Sketch correctly
**Because:** The algorithm is conceptually simple (d hash functions, w-wide arrays, take minimum). The challenge is understanding the error bounds.

## Task Prompt (Given to Agent)

```
Implement a Count-Min Sketch from scratch in Python.

A Count-Min Sketch is a probabilistic data structure for frequency estimation.
It uses sub-linear space and provides estimates that may overestimate but never underestimate.

Requirements:
1. NO external libraries except typing, math, and hashlib
2. Must use multiple hash functions (depth d)
3. Must handle these operations:

class CountMinSketch:
    def __init__(self, width: int, depth: int):
        """
        Initialize Count-Min Sketch.
        width: Number of counters per row (w)
        depth: Number of hash functions/rows (d)

        Error bounds:
        - Overestimate by at most N/w with probability 1 - (1/2)^d
        - where N = total count of all items
        """
        pass

    def add(self, item: str, count: int = 1) -> None:
        """Add count for item. Updates all d rows."""
        pass

    def query(self, item: str) -> int:
        """
        Estimate count for item.
        Returns minimum across all d rows (never underestimates).
        """
        pass

    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """
        Merge two sketches (must have same dimensions).
        Returns new sketch with combined counts.
        """
        pass

    @property
    def total_count(self) -> int:
        """Return total number of items added (sum of all counts)."""
        pass

Hash function approach:
    Use double hashing: h_i(x) = (hash1(x) + i * hash2(x)) % width
    This generates d different hash functions from 2 base hashes.

Test cases that MUST pass:

# Test 1: Basic add and query
cms = CountMinSketch(width=1000, depth=5)
cms.add("hello", 10)
cms.add("world", 5)
assert cms.query("hello") >= 10  # Never underestimates
assert cms.query("world") >= 5
assert cms.query("missing") >= 0  # Unknown items return 0 or small overestimate

# Test 2: Multiple adds accumulate
cms = CountMinSketch(width=1000, depth=5)
cms.add("item", 5)
cms.add("item", 3)
cms.add("item", 2)
assert cms.query("item") >= 10

# Test 3: Estimates are reasonably accurate with low collision
cms = CountMinSketch(width=10000, depth=7)
items = {"apple": 100, "banana": 50, "cherry": 25}
for item, count in items.items():
    cms.add(item, count)
# With large width, estimates should be close to actual
for item, actual in items.items():
    estimate = cms.query(item)
    assert estimate >= actual  # Never underestimate
    assert estimate <= actual * 1.5  # Shouldn't overestimate by more than 50%

# Test 4: Total count tracking
cms = CountMinSketch(width=100, depth=3)
cms.add("a", 10)
cms.add("b", 20)
cms.add("c", 30)
assert cms.total_count == 60

# Test 5: Merge sketches
cms1 = CountMinSketch(width=100, depth=3)
cms2 = CountMinSketch(width=100, depth=3)
cms1.add("item", 5)
cms2.add("item", 3)
merged = cms1.merge(cms2)
assert merged.query("item") >= 8

# Test 6: High collision scenario (small width)
cms = CountMinSketch(width=10, depth=3)
# Add many different items - will have collisions
for i in range(100):
    cms.add(f"item_{i}", 1)
# Estimates will be inflated due to collisions
# But minimum over depths helps reduce this
estimate = cms.query("item_0")
assert estimate >= 1  # Never underestimates
# With small width (10) and 100 items, expect significant overestimate

# Test 7: Deterministic behavior (same hash = same result)
cms = CountMinSketch(width=100, depth=3)
cms.add("test", 5)
result1 = cms.query("test")
result2 = cms.query("test")
assert result1 == result2

Write the complete implementation.
Include comments explaining the error bounds and why we take minimum.
```

## Success Criteria
- [ ] All operations implemented
- [ ] Never underestimates (query >= actual)
- [ ] Multiple hash functions used (depth d)
- [ ] Merge works correctly
- [ ] All 7 test cases pass

## Failure Criteria
- [ ] Underestimates (query < actual)
- [ ] Only one hash function (no depth)
- [ ] Takes maximum instead of minimum
- [ ] Merge dimensions mismatch not handled
- [ ] Uses external libraries

## Prediction
Before running: **PASS**
Confidence: **HIGH**
Reasoning: Count-Min Sketch is straightforward: d arrays, hash to each, increment, query takes min. The algorithm is simpler than Bloom filters.

## Actual Result
Status: [NOT YET RUN]
Operations implemented: [X/4]
Tests passed: [X/7]
Notes:

## Agent Output
```python
[Agent's code will be pasted here after running]
```

## Test Results
```
[Test execution output will be pasted here]
```

## Analysis
**Discrepancy:**
**Root cause:**
**Learning:**

## Recommendations

