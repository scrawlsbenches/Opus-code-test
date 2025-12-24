# GoT Query API Performance Analysis

## Profiling Results (50 tasks, 10 decisions, 88 edges)

### Summary Table

| Component | Avg Time | Ops/sec | Bottleneck |
|-----------|----------|---------|------------|
| **Query API** | 7ms | 140 | File I/O |
| **GraphWalker** | 20ms | 50 | File I/O + Graph traversal |
| **PathFinder** | 10-20ms | 50-100 | File I/O + BFS |
| **PatternMatcher** | 23ms | 42 | File I/O + Backtracking |

### Detailed Results

#### Query API (Fastest)
```
Query: All tasks                   7.4ms  (135 ops/s)
Query: Filter by status            7.5ms  (134 ops/s)
Query: Filter by priority          7.2ms  (139 ops/s)
Query: OR conditions               7.0ms  (142 ops/s)
Query: Order by priority           7.0ms  (144 ops/s)
Query: Pagination (limit 10)       7.0ms  (143 ops/s)
Query: Count                       6.9ms  (144 ops/s)
Query: Group by + count            7.0ms  (144 ops/s)
Query: Complex chain               7.7ms  (130 ops/s)
```

#### GraphWalker
```
Walker: BFS traversal              22ms   (44 ops/s)
Walker: DFS traversal              22ms   (46 ops/s)
Walker: Max depth 2                22ms   (45 ops/s)
Walker: Count with filter          22ms   (46 ops/s)
Walker: Collect titles             22ms   (46 ops/s)
```

#### PathFinder
```
PathFinder: Shortest path          10ms   (100 ops/s)
PathFinder: Connected components   20ms   (50 ops/s)
PathFinder: Reachable from         10ms   (100 ops/s)
PathFinder: All paths              SKIPPED (exponential)
```

#### PatternMatcher
```
Pattern: 2-node dependency         24ms   (42 ops/s)
Pattern: With constraints          23ms   (43 ops/s)
Pattern: 3-node chain              25ms   (39 ops/s)
Pattern: Count matches             23ms   (43 ops/s)
```

## Root Cause Analysis

### cProfile Results
```
ncalls  tottime  cumtime  filename:lineno(function)
    50   0.004    0.004   {built-in method io.open}
    50   0.001    0.002   __init__.py:274(load)  # json.load
    50   0.001    0.001   {method 'read' of '_io.TextIOWrapper'}
```

**The bottleneck is file I/O, not computation.**

Every query reads 50+ JSON files from disk:
- Opening files: 4ms
- Reading content: 1ms
- JSON parsing: <1ms

The GoTManager is designed for durability (always read from disk), not performance.

## Optimization Recommendations

### 1. Entity Caching (High Impact)
Add an in-memory cache to GoTManager:

```python
class GoTManager:
    def __init__(self, got_dir, cache_enabled=True):
        self._cache = {} if cache_enabled else None
        self._cache_valid = {}

    def _read_task_file(self, path):
        if self._cache and path in self._cache:
            return self._cache[path]
        task = self._read_from_disk(path)
        if self._cache:
            self._cache[path] = task
        return task
```

**Expected improvement:** 10-50x faster for repeated queries

### 2. Query-Level Caching (Medium Impact)
Cache entities during a single query execution:

```python
class Query:
    def execute(self):
        # Load all entities once
        entities = list(self._manager.list_all_tasks())
        # Filter in memory
        return self._filter(entities)
```

**Expected improvement:** 2-3x faster for complex queries

### 3. Batch Loading Mode (High Impact)
Pre-load all entities at startup for read-heavy workloads:

```python
class GoTManager:
    def load_all(self):
        """Load all entities into memory for fast queries."""
        self._tasks = list(self.list_all_tasks())
        self._edges = list(self.list_edges())
```

**Expected improvement:** Sub-millisecond queries after initial load

### 4. Index Files (Medium Impact)
Maintain index files for common query patterns:

```
.got/
  indices/
    by_status.json    # {"pending": ["T-001", "T-002"], ...}
    by_priority.json  # {"high": ["T-003"], ...}
```

**Expected improvement:** 5-10x faster for indexed queries

## Known Expensive Operations

### PathFinder.all_paths() - O(2^n)
This operation enumerates ALL paths between two nodes. On a connected graph, this is exponential:

- 10 nodes, 5% edges: ~100 paths (fast)
- 50 nodes, 5% edges: potentially millions of paths (hangs)

**Recommendation:** Add max_paths limit or max_depth parameter:
```python
def all_paths(self, from_id, to_id, max_paths=100, max_depth=10):
    """Find paths with limits to prevent explosion."""
```

### PatternMatcher.find() - O(n^k)
Pattern matching is polynomial in the number of pattern nodes (k):

- 2-node patterns: O(n²) - 24ms
- 3-node patterns: O(n³) - 25ms
- 4-node patterns: O(n⁴) - ~100ms estimated

**Recommendation:** Use `.limit()` for large patterns:
```python
PatternMatcher(manager).find(pattern).limit(10)
```

## Performance Thresholds

Based on user experience research:

| Response Time | Perception |
|---------------|------------|
| < 100ms | Instant |
| 100-300ms | Fast |
| 300-1000ms | Noticeable delay |
| > 1000ms | Slow |

Current GoT Query API performance:
- **Query**: 7ms - Instant
- **Walker**: 20ms - Instant
- **PathFinder**: 10-20ms - Instant
- **PatternMatcher**: 23ms - Instant

All operations are within acceptable range for interactive use.

## Profiling Script

Run the profiler:
```bash
python scripts/profile_got_query.py              # Quick (50 tasks)
python scripts/profile_got_query.py --full       # Full (50-200 tasks)
python scripts/profile_got_query.py --detailed   # With cProfile output
python scripts/profile_got_query.py --component query  # Specific component
```

---
*Generated: 2025-12-24*
*Graph: 50 tasks, 10 decisions, ~90 edges*
