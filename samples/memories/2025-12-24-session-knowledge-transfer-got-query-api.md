# Knowledge Transfer: GoT Query API Implementation

**Session Date:** 2025-12-24
**Branch:** `claude/got-codebase-review-OEigg`
**Focus:** GoT Query API - Fluent query builder, graph traversal, path finding, pattern matching

---

## What Was Built

### 1. Query Builder (`cortical/got/query_builder.py` ~800 lines)

SQL-like fluent interface for querying GoT entities:

```python
from cortical.got.query_builder import Query

# Filter, sort, paginate
results = (Query(manager)
    .tasks()
    .where(status="pending", priority="high")
    .or_where(priority="critical")
    .order_by("created_at", desc=True)
    .limit(10)
    .execute())

# Aggregations
counts = Query(manager).tasks().group_by("status").count().execute()
# {"pending": 10, "completed": 57}
```

**Key implementation details:**
- `_matches_filters()` uses `(all WHERE match) OR (any OR group matches)` logic
- `_count_mode` flag enables `.count().execute()` chaining with group_by
- Generator-based `_execute_query()` for lazy evaluation

### 2. GraphWalker (`cortical/got/graph_walker.py` ~380 lines)

Visitor pattern graph traversal:

```python
from cortical.got.graph_walker import GraphWalker

def collect_ids(node, acc):
    acc.append(node.id)
    return acc

ids = (GraphWalker(manager)
    .starting_from(task_id)
    .bfs()  # or .dfs()
    .max_depth(3)
    .visit(collect_ids, initial=[])
    .run())
```

**Key implementation details:**
- `_bidirectional = True` by default (treats edges as undirected)
- Use `.directed()` or `.reverse()` to change edge direction
- Visitor receives `(node, accumulator)` and returns updated accumulator

### 3. PathFinder (`cortical/got/path_finder.py` ~190 lines)

Graph path algorithms:

```python
from cortical.got.path_finder import PathFinder

finder = PathFinder(manager)
path = finder.shortest_path(from_id, to_id)  # BFS
reachable = finder.reachable_from(node_id)   # All connected nodes
components = finder.connected_components()    # Disconnected subgraphs
```

**WARNING:** `all_paths()` is O(2^n) on connected graphs - can hang!

### 4. PatternMatcher (`cortical/got/pattern_matcher.py` ~250 lines)

Subgraph isomorphism:

```python
from cortical.got.pattern_matcher import Pattern, PatternMatcher

pattern = (Pattern()
    .node("a", type="task", status="pending")
    .edge("DEPENDS_ON", direction="outgoing")
    .node("b", type="task", status="completed"))

matches = PatternMatcher(manager).find(pattern)
```

**Key implementation details:**
- Edges are chained (not named): `.node().edge().node()`
- Direction is relative to previous node: "outgoing" = prev -> current
- O(n^k) complexity where k = pattern nodes

### 5. CLI Integration (`cortical/got/cli/analyze.py`)

```bash
python scripts/got_utils.py analyze summary
python scripts/got_utils.py analyze dependencies TASK_ID
python scripts/got_utils.py analyze patterns
python scripts/got_utils.py analyze orphans
```

---

## Performance Findings

**Root cause of slowness: File I/O (no caching)**

| Component | Avg Time | Bottleneck |
|-----------|----------|------------|
| Query API | 7ms | 50 file reads per query |
| GraphWalker | 20ms | File I/O + traversal |
| PathFinder | 10-20ms | File I/O + BFS |
| PatternMatcher | 23ms | File I/O + backtracking |

**cProfile revealed:**
```
50 calls to io.open = 4ms
50 calls to read() = 1ms
JSON parsing < 1ms
```

See `docs/got-performance-analysis.md` for optimization recommendations.

---

## Tests

- **Location:** `tests/unit/got/test_query_builder.py`
- **Count:** 41 tests (75 total with query language tests)
- **Coverage:** All components tested

Key test patterns:
```python
# Tests use mock_manager fixture for isolation
def test_where_single_condition(self, mock_manager, mock_args):
    mock_manager.list_all_tasks.return_value = [mock_task]
    result = list(Query(mock_manager).tasks().where(status="pending").execute())
```

---

## Files Created/Modified

### New Files
- `cortical/got/query_builder.py` - Query builder with aggregations
- `cortical/got/graph_walker.py` - Visitor pattern traversal
- `cortical/got/path_finder.py` - BFS/DFS path algorithms
- `cortical/got/pattern_matcher.py` - Subgraph matching
- `cortical/got/cli/analyze.py` - CLI commands
- `tests/unit/got/test_query_builder.py` - 41 tests
- `docs/got-query-api.md` - API documentation
- `docs/got-performance-analysis.md` - Performance analysis
- `scripts/profile_got_query.py` - Profiling script

### Modified Files
- `cortical/got/__init__.py` - Added exports
- `scripts/got_utils.py` - Added analyze command
- `CLAUDE.md` - Added Query API to architecture and quick reference
- `README.md` - Added GoT Query API section

---

## Pending Tasks (Sprint 18: Performance & Observability)

| Priority | Task |
|----------|------|
| **High** | Add entity caching layer to GoTManager |
| **High** | Define performance KPI targets |
| **High** | Add max_paths limit to all_paths() |
| Medium | Add query-level caching |
| Medium | Add batch loading mode |
| Medium | Add query metrics |
| Medium | Add cache invalidation with TTL |
| Medium | Add streaming for large results |
| Low | Add query explain/plan |
| Low | Add index files |
| Low | Add query logging |
| Low | Add syntax validation |

---

## Key Decisions Made

1. **Bidirectional by default** - GraphWalker treats edges as undirected because most use cases want "find all connected" not "follow edge direction"

2. **Visitor pattern with accumulator** - Returns final value instead of yielding nodes, enables stateful traversal

3. **Skipped all_paths() in profiling** - O(2^n) complexity makes it unsuitable for benchmarking

4. **No caching in GoTManager** - Current design prioritizes durability (always read from disk)

---

## Gotchas for Future Development

1. **Pattern.edge() syntax**: Use `.edge("TYPE", direction="outgoing")`, NOT `.edge("a", "b", "TYPE")`

2. **Query.order_by()**: Use `desc=True`, NOT `descending=True`

3. **GraphWalker**: Use `.starting_from()`, NOT `.start_from()`

4. **Task attributes**: Access via `task.status`, NOT `task.properties.get("status")`

5. **all_paths() hangs**: On connected graphs with 50+ nodes and 5% edge density, this is exponential

---

## Quick Start for Next Session

```bash
# Check current state
python scripts/got_utils.py analyze summary
python scripts/got_utils.py task list --status pending

# Run tests
python -m pytest tests/unit/got/test_query_builder.py -v

# Profile performance
python scripts/profile_got_query.py

# Read the docs
cat docs/got-query-api.md
cat docs/got-performance-analysis.md
```

---

**Tags:** `got`, `query-api`, `performance`, `profiling`, `graph-algorithms`
