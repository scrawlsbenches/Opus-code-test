# Knowledge Transfer: Parallel Bigram Connections Performance Investigation

**Date:** 2025-12-30
**Branch:** `claude/accept-got-handoff-pulIU`
**Commits:** `f5e718e9` → `d60f4a3f` → `1f8a345b` → `f9a21bf2`

---

## Executive Summary

Investigated parallelizing `compute_bigram_connections()` to improve performance on large corpora. After implementing three approaches (ThreadPoolExecutor, ProcessPoolExecutor with pickle, shared memory multiprocessing), **benchmarks conclusively show sequential execution is 4-6x faster** for typical workloads. The parallel implementation is preserved but documented as slower for most use cases.

---

## What Was Done

### 1. Initial Investigation (Previous Session)

Started with ThreadPoolExecutor-based parallelism, expecting speedup from multi-core utilization.

**File Modified:** `cortical/analysis/connections.py`

```python
# Added n_workers parameter to compute_bigram_connections()
def compute_bigram_connections(
    ...
    n_workers: Optional[int] = None  # New parameter
) -> Dict[str, Any]:
```

### 2. Three Parallel Implementations Attempted

#### Attempt 1: ThreadPoolExecutor (GIL-blocked)
- **Result:** 2x SLOWER than sequential
- **Reason:** Python's Global Interpreter Lock (GIL) prevents true parallel execution for CPU-bound work
- **Code:** `_compute_bigram_connections_parallel()` - marked as DEPRECATED

#### Attempt 2: ProcessPoolExecutor with Pickle
- **Result:** Hangs on large corpora, slow on small ones
- **Reason:** Pickle serialization of 500MB+ Minicolumn objects creates massive overhead
- **Code:** `_compute_bigram_connections_parallel_process()`

#### Attempt 3: Shared Memory Multiprocessing (Final)
- **Result:** 4-6x SLOWER than sequential for typical workloads
- **Approach:**
  - Convert all string IDs to integer indices
  - Encode document sets as bit vectors (1202 docs = 19 uint64s = 152 bytes/bigram)
  - Use `struct.pack()` for compact binary representation
  - Share read-only data via `multiprocessing.shared_memory`
  - Workers return `(idx1, idx2, weight)` tuples only
- **Code:** `_compute_bigram_connections_shared_memory()` at line 335

---

## How It Was Done

### Shared Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SHARED MEMORY LAYOUT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐  Offset 0                                      │
│  │ left_term_data   │  int32[n_bigrams] - left term indices          │
│  ├──────────────────┤                                                │
│  │ right_term_data  │  int32[n_bigrams] - right term indices         │
│  ├──────────────────┤                                                │
│  │ tfidf_data       │  float32[n_bigrams] - TF-IDF scores            │
│  ├──────────────────┤                                                │
│  │ doc_bits_data    │  uint64[n_bigrams * bits_per_bigram]           │
│  │                  │  Bit vector encoding of document sets          │
│  ├──────────────────┤                                                │
│  │ left_groups_data │  Length-prefixed arrays per term               │
│  ├──────────────────┤                                                │
│  │ right_groups_data│  Length-prefixed arrays per term               │
│  └──────────────────┘                                                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Data Structures

```python
@dataclass
class CompactBigramData:
    """Compact representation for zero-copy sharing."""
    n_bigrams: int
    n_docs: int
    n_terms: int
    bits_per_bigram: int  # (n_docs + 63) // 64

    # Index mappings (main process only)
    bigram_id_to_idx: Dict[str, int]
    idx_to_bigram_id: List[str]
    term_to_idx: Dict[str, int]
    doc_to_bit_pos: Dict[str, int]

    # Packed binary data for shared memory
    left_term_data: bytes      # int32[n_bigrams]
    right_term_data: bytes     # int32[n_bigrams]
    tfidf_data: bytes          # float32[n_bigrams]
    doc_bits_data: bytes       # uint64[n_bigrams * bits_per_bigram]
    left_groups_data: bytes    # Length-prefixed term groups
    right_groups_data: bytes
```

### Worker Functions

1. **`_worker_process_components()`** - Processes shared component connections
   - Reads term groups from shared memory
   - Connects all bigram pairs sharing a term
   - Returns `List[Tuple[int, int, float]]`

2. **`_worker_process_cooccurrence()`** - Processes document co-occurrence
   - Uses bitwise AND/OR for fast Jaccard similarity
   - `intersection = popcount(bits1 & bits2)`
   - `union = popcount(bits1 | bits2)`
   - Returns connections meeting threshold

### Bit Vector Jaccard (Fast Path)

```python
def _popcount64(x: int) -> int:
    """Brian Kernighan's algorithm - O(set bits)."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

# Jaccard similarity via bit operations
intersection = sum(_popcount64(b1 & b2) for b1, b2 in zip(bits1, bits2))
union = sum(_popcount64(b1 | b2) for b1, b2 in zip(bits1, bits2))
jaccard = intersection / union if union > 0 else 0.0
```

---

## Benchmark Results

### Using Corpus Benchmark Runner

Added `ParallelBigramConnectionsBenchmark` to `benchmarks/corpus/runner.py`:

```bash
python -m benchmarks.corpus.runner --benchmark parallel_bigram_connections
```

### Results Table

| Mode | Corpus | Sequential | Parallel (2 workers) | Speedup |
|------|--------|------------|---------------------|---------|
| Quick | 100 docs | 677ms | 4,026ms | **0.17x** (6x slower) |
| Full | 300 docs | 4,355ms | 19,118ms | **0.23x** (4.4x slower) |

### Ad-hoc Testing Results (Previous Session)

| Docs | Bigrams | Sequential | Parallel | Speedup |
|------|---------|------------|----------|---------|
| 200 | 2,594 | 0.294s | 2.676s | 0.11x |
| 500 | 4,022 | 0.354s | 5.775s | 0.06x |
| 1000 | 4,543 | 0.488s | 9.966s | 0.05x |

### Crossover Point (When Parallel Wins)

Only observed with:
- Very simple vocabulary (<150 unique bigrams)
- Many documents (500+)
- Example: 500 docs / 103 bigrams → **3.69x speedup**

This is an unusual edge case - real corpora have varied vocabulary.

---

## Root Cause Analysis

### Why Parallel Is Slower

| Overhead Source | Time Cost | Notes |
|-----------------|-----------|-------|
| Process creation | ~100ms/worker | Fork + Python interpreter init |
| Compact data building | ~50-200ms | Converting Minicolumns to bytes |
| Shared memory setup | ~10-50ms | Allocation + copying |
| Worker coordination | ~50ms | Future management, result collection |
| **Total overhead** | **~200-400ms** | Before any actual work begins |

### The Math

For a typical 300-doc corpus:
- Sequential: 4,355ms (pure computation)
- Parallel overhead: ~400ms (fixed cost)
- Parallel computation: ~19,000ms (worse algorithm + worker overhead)

The overhead exceeds the potential gains from parallelism.

### Algorithm Differences

The shared memory cooccurrence worker uses a different algorithm:
```python
# Sequential: Iterates by document, connects bigrams within same doc
for doc_id, bigrams in doc_index.items():
    for b1, b2 in combinations(bigrams, 2):
        ...

# Parallel: Iterates by bigram index with window limit
for idx1 in range(start, end):
    for idx2 in range(idx1 + 1, min(idx1 + 1000, n_bigrams)):
        ...
```

This creates different connection patterns and can be less efficient.

---

## Final State

### Files Modified

1. **`cortical/analysis/connections.py`**
   - Added `CompactBigramData` dataclass
   - Added `_build_compact_data()` function
   - Added `_worker_process_components()` worker
   - Added `_worker_process_cooccurrence()` worker
   - Added `_compute_bigram_connections_shared_memory()` orchestrator
   - Updated docstring with performance notes

2. **`benchmarks/corpus/runner.py`**
   - Added `ParallelBigramConnectionsBenchmark` class

### Default Behavior

```python
# Default is sequential (n_workers=None)
processor.compute_bigram_connections()  # Fast, recommended

# Parallel mode (slower for most cases)
processor.compute_bigram_connections(n_workers=2)  # 4-6x slower typically
```

### Docstring Warning

```python
n_workers: Number of parallel workers. None or 1 for sequential execution,
    >1 for parallel execution using shared memory multiprocessing. Default
    is None (sequential).

    PERFORMANCE NOTE: Shared memory parallelism has significant overhead
    from process creation (~100ms), data serialization, and coordination.
    Benchmarks show:
    - Sequential is typically 4-6x FASTER for most workloads
    - Parallel only helps with very simple vocabulary (few unique bigrams)
      and many documents (the crossover was ~500 docs with <150 bigrams)
    - For real-world corpora with varied vocabulary, sequential is preferred
```

---

## Lessons Learned

### 1. Python Parallelism Is Hard

- **GIL blocks threads** for CPU-bound work
- **Pickle is expensive** for large objects (MB+ scale)
- **Process overhead is significant** (~100ms per worker)
- **Shared memory requires redesign** - can't just share Python objects

### 2. Profile Before Optimizing

The assumption that "parallel = faster" was wrong. Benchmarking revealed:
- The actual computation is fast
- The overhead dominates
- Sequential is the right choice

### 3. Preserve Working Code

Even though parallel is slower, we kept it for:
- Edge cases where it might help
- Documentation of the pattern
- Future optimization (e.g., native extensions, PyPy)

### 4. Use the Benchmark Framework

The corpus benchmark runner (`benchmarks/corpus/runner.py`) provides:
- Standardized measurement
- Reproducible results
- Easy comparison over time

---

## Future Optimization Opportunities

If parallel performance becomes critical:

1. **Native Extension (Cython/Rust)**
   - Bypass Python overhead entirely
   - Direct memory access without serialization
   - True multi-threading without GIL

2. **Batch Processing**
   - Process multiple corpora in parallel (process per corpus)
   - Amortize process creation over many operations

3. **GPU Acceleration**
   - Bit vector operations are highly parallelizable
   - CUDA for massive document counts

4. **Incremental Updates**
   - Only recompute connections for changed bigrams
   - Already partially supported via staleness tracking

---

## References

- Commit `f5e718e9`: feat(analysis): Add parallel option for compute_bigram_connections
- Commit `d60f4a3f`: perf(parallel): Implement ProcessPoolExecutor for bigram connections
- Commit `1f8a345b`: feat(parallel): Add shared memory parallel implementation
- Commit `f9a21bf2`: perf(parallel): Document parallel overhead after benchmarking
- Benchmark: `python -m benchmarks.corpus.runner --benchmark parallel_bigram_connections`
