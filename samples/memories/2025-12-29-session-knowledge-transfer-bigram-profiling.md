# Knowledge Transfer: Bigram Connections Profiling Analysis

**Date:** 2025-12-29
**Session:** Profiling bigram_connections bottleneck
**Task:** T-20251229-131355-f4b7e3a7
**Branch:** `claude/accept-got-handoff-ZM39G`

---

## Summary

Profiled `compute_bigram_connections()` which takes ~65% of `compute_all()` time. Identified 3 concrete optimization opportunities with estimated savings of ~2.5s on 100-doc corpus.

---

## Profiling Results

**Test corpus:** 100 documents from samples/, 34,950 bigrams

**Timing:**
| Function | Time | % of Total |
|----------|------|------------|
| `compute_all()` | 8,207ms | 100% |
| `compute_bigram_connections()` | 6,599ms | 80% |

**Function hotspots within compute_bigram_connections:**
| Function | Calls | Cumulative Time | % |
|----------|-------|-----------------|---|
| `add_connection` (nested fn) | 2,028,350 | 3,530ms | 54% |
| `add_lateral_connections_batch` | 34,781 | 1,577ms | 24% |
| `sorted()` | 2,028,350 | 332ms | 5% |
| Edge `__init__` | 1,486,718 | 373ms | 6% |

**Connection statistics:**
| Metric | Count |
|--------|-------|
| Total bigrams | 34,950 |
| Connections created | 743,359 |
| Connection attempts | 2,028,350 |
| Skipped (max_connections limit) | 1,270,926 (62.7%) |
| Component connections | 509,422 |
| Chain connections | 122,625 |
| Co-occurrence connections | 111,312 |

---

## Key Finding

**62.7% of pair processing is wasted** - bigrams that have hit their `max_connections_per_bigram` limit (50) are still checked in inner loops before being rejected in `add_connection`.

---

## Optimization Opportunities

### 1. Cheaper Pair Canonicalization (Est. ~300ms savings)

**Current (line 313):**
```python
pair = tuple(sorted([b1.id, b2.id]))
```

**Optimized:**
```python
pair = (b1.id, b2.id) if b1.id < b2.id else (b2.id, b1.id)
```

**Measured speedup:** 1.39x on 2M iterations

### 2. Early Bailout for Maxed Bigrams (Est. ~2,200ms savings)

**Problem:** Inner loops iterate all pairs, even when one bigram has hit its connection limit.

**Solution:** In outer loops (lines 346-379), add early skip:
```python
for i, b1 in enumerate(bigram_list):
    if connection_counts[b1.id] >= max_connections_per_bigram:
        continue  # Skip bigram that can't take more connections
    for b2 in bigram_list[i+1:]:
        if connection_counts[b2.id] >= max_connections_per_bigram:
            continue
        # ... rest of logic
```

This avoids 62% of loop iterations and associated `add_connection` calls.

### 3. Edge Object Pooling (Est. ~200ms savings)

**Problem:** 1.5M Edge objects created during batch apply.

**Solution:** Use `__slots__` on Edge class (if not already), or use a flyweight pattern for common edge configurations.

---

## Implementation Priority

| Optimization | Complexity | Savings | Priority |
|--------------|-----------|---------|----------|
| Early bailout | Low | ~2.2s | **HIGH** |
| Pair canonicalization | Trivial | ~0.3s | Medium |
| Edge pooling | Medium | ~0.2s | Low |

---

## Related Files

- `cortical/analysis/connections.py:219-449` - compute_bigram_connections
- `cortical/minicolumn.py:212-241` - add_lateral_connections_batch
- `cortical/minicolumn.py:152-175` - Edge class and add_lateral_connection

---

## Test Commands

```bash
# Profile with samples corpus
python -c "
from cortical import CorticalTextProcessor
from pathlib import Path
import time

processor = CorticalTextProcessor()
for f in list(Path('samples').rglob('*.md'))[:100]:
    processor.process_document(str(f), f.read_text())

start = time.perf_counter()
processor.compute_all()
print(f'compute_all: {(time.perf_counter()-start)*1000:.0f}ms')
"

# Profile with real corpus (if available)
python scripts/profile_full_analysis.py
```

---

## Next Steps

1. Implement early bailout optimization in `compute_bigram_connections`
2. Run benchmarks to validate savings
3. Update bigram connection limits documentation if needed

---

*Generated: 2025-12-29*
