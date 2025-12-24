# GoT Query API

A powerful, fluent query system for the Graph of Thought. Built with clean design patterns and comprehensive test coverage.

## Overview

The GoT Query API provides four complementary tools for graph analysis:

| Component | Purpose | Pattern Used |
|-----------|---------|--------------|
| **Query** | SQL-like entity queries | Builder Pattern |
| **GraphWalker** | Graph traversal | Visitor Pattern |
| **PathFinder** | Path algorithms | BFS/DFS |
| **PatternMatcher** | Subgraph matching | Backtracking |

## Quick Start

```python
from cortical.got import Query, GraphWalker, PathFinder, Pattern, PatternMatcher

# Get all high-priority pending tasks
urgent = (
    Query(manager)
    .tasks()
    .where(status="pending", priority="high")
    .order_by("created_at", desc=True)
    .limit(10)
    .execute()
)

# Count tasks by status
counts = Query(manager).tasks().group_by("status").count().execute()
# {"pending": 5, "completed": 12, "in_progress": 3}

# Walk dependency graph
GraphWalker(manager).starting_from(task_id).bfs().visit(collector).run()

# Find shortest path
path = PathFinder(manager).shortest_path(task_a, task_b)

# Find patterns
pattern = Pattern().node("a", type="task").edge("BLOCKS").node("b", type="task")
matches = PatternMatcher(manager).find(pattern)
```

## Components

### Query Builder

Fluent, chainable API for filtering, sorting, and aggregating entities.

**Source:** [`cortical/got/query_builder.py`](../cortical/got/query_builder.py)

```python
# Complex query with multiple conditions
results = (
    Query(manager)
    .tasks()
    .where(status="pending")           # AND condition
    .where(priority="high")            # AND condition
    .or_where(priority="critical")     # OR alternative
    .connected_to(sprint_id)           # Must be in sprint
    .order_by("created_at", desc=True)
    .limit(10)
    .execute()
)
```

**Key Methods:**
- `.tasks()`, `.sprints()`, `.decisions()`, `.edges()` - Select entity type
- `.where(**conditions)` - Filter (AND logic)
- `.or_where(**conditions)` - Alternative filter (OR logic)
- `.connected_to(id, via=edge_type)` - Connection filter
- `.order_by(field, desc=False)` - Sort results
- `.limit(n)`, `.offset(n)` - Pagination
- `.group_by(*fields)` - Grouping for aggregation
- `.aggregate(name=Function())` - Aggregation functions
- `.execute()` - Run and return results
- `.iter()` - Lazy iteration (memory efficient)
- `.count()`, `.exists()`, `.first()` - Convenience methods
- `.explain()` - Query plan without execution

**Aggregation Functions:**
- `Count()` - Count items
- `Collect(field)` - Collect field values into list
- `Sum(field)`, `Avg(field)` - Numeric aggregations
- `Min(field)`, `Max(field)` - Find extremes

### GraphWalker

Traverse the graph with the Visitor pattern.

**Source:** [`cortical/got/graph_walker.py`](../cortical/got/graph_walker.py)

```python
# Count tasks by status across connected nodes
def count_by_status(node, acc):
    acc[node.status] = acc.get(node.status, 0) + 1
    return acc

result = (
    GraphWalker(manager)
    .starting_from(root_task_id)
    .follow("DEPENDS_ON")      # Only follow these edges
    .max_depth(3)              # Limit depth
    .filter(lambda n: n.priority == "high")
    .bfs()                     # Or .dfs()
    .visit(count_by_status, initial={})
    .run()
)
```

**Key Methods:**
- `.starting_from(node_id)` - Set start node
- `.bfs()`, `.dfs()` - Traversal strategy
- `.follow(*edge_types)` - Filter edge types
- `.max_depth(n)` - Limit traversal depth
- `.filter(predicate)` - Filter nodes
- `.reverse()` - Follow edges backwards
- `.directed()` - Only source→target direction
- `.visit(fn, initial=None)` - Set visitor function
- `.run()` - Execute traversal
- `.iter()` - Lazy iteration

### PathFinder

Find paths between nodes.

**Source:** [`cortical/got/path_finder.py`](../cortical/got/path_finder.py)

```python
# Shortest path
path = PathFinder(manager).shortest_path(start_id, end_id)

# All paths with length limit
paths = PathFinder(manager).max_length(5).all_paths(start_id, end_id)

# Check connectivity
if PathFinder(manager).path_exists(a, b):
    print("Connected!")

# Find isolated clusters
components = PathFinder(manager).connected_components()
```

**Key Methods:**
- `.shortest_path(from_id, to_id)` - BFS shortest path
- `.all_paths(from_id, to_id)` - All paths (use `max_length`!)
- `.path_exists(from_id, to_id)` - Boolean check
- `.reachable_from(node_id)` - All reachable nodes
- `.connected_components()` - Find isolated clusters
- `.via_edges(*types)` - Filter edge types
- `.max_length(n)` - Limit path length
- `.directed()` - Only source→target

### PatternMatcher

Find subgraph patterns.

**Source:** [`cortical/got/pattern_matcher.py`](../cortical/got/pattern_matcher.py)

```python
# Find blocking chains: A blocks B blocks C
pattern = (
    Pattern()
    .node("a", type="task")
    .edge("BLOCKS", direction="outgoing")
    .node("b", type="task")
    .edge("BLOCKS", direction="outgoing")
    .node("c", type="task", priority="high")
)

for match in PatternMatcher(manager).find(pattern):
    print(f"{match['a'].title} -> {match['b'].title} -> {match['c'].title}")
```

**Pattern Building:**
- `.node(name, type=None, **constraints)` - Add node with constraints
- `.edge(type, direction="outgoing")` - Add edge between nodes

**Matcher Methods:**
- `.find(pattern)` - Find all matches
- `.find_first(pattern)` - First match only
- `.count(pattern)` - Count matches
- `.limit(n)` - Cap number of results

## CLI Commands

The Query API powers the `got analyze` CLI commands:

```bash
# Summary with task counts by status/priority
python scripts/got_utils.py analyze summary

# Dependency analysis for a task
python scripts/got_utils.py analyze dependencies T-xxx

# Find graph patterns
python scripts/got_utils.py analyze patterns

# Find disconnected clusters
python scripts/got_utils.py analyze orphans
```

**Source:** [`cortical/got/cli/analyze.py`](../cortical/got/cli/analyze.py)

## Design Patterns

### Builder Pattern (Query)
Method chaining constructs complex queries incrementally:
```python
Query(manager).tasks().where(x=1).where(y=2).limit(10).execute()
```

### Visitor Pattern (GraphWalker)
Accumulator-based traversal without external mutable state:
```python
.visit(lambda node, acc: acc + [node.id], initial=[])
```

### Strategy Pattern (Aggregations)
Pluggable aggregation functions with three-phase lifecycle:
```python
class Sum(AggregateFunction):
    def initial(self): return 0
    def accumulate(self, acc, entity): return acc + entity.value
    def finalize(self, acc): return acc
```

## Performance Notes

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `Query.execute()` | O(n) | Scans all entities of type |
| `Query.iter()` | O(1) per item | Memory efficient streaming |
| `Query.exists()` | O(1) best | Stops at first match |
| `PathFinder.shortest_path()` | O(V+E) | BFS, safe for large graphs |
| `PathFinder.all_paths()` | O(V!) worst | **Use max_length!** |
| `PatternMatcher.find()` | O(n^k) | n=nodes, k=pattern size |

**Tips:**
- Use `.limit()` to avoid loading everything
- Use `.iter()` for memory-efficient streaming
- Use `.exists()` when you only need boolean
- Use `.first()` when you only need one result
- Use `.explain()` to see query plan

## Test Coverage

75 tests covering all components:
- `tests/unit/got/test_query_builder.py` - 41 tests
- `tests/unit/got/test_query_language.py` - 34 tests

```bash
python -m pytest tests/unit/got/test_query_builder.py -v
```

## Architecture

```
cortical/got/
├── query_builder.py     # Query, aggregations (790 lines)
├── graph_walker.py      # GraphWalker (380 lines)
├── path_finder.py       # PathFinder (190 lines)
├── pattern_matcher.py   # Pattern, PatternMatcher (250 lines)
└── cli/
    └── analyze.py       # CLI commands using Query API
```

All modules are extensively documented with:
- Module docstrings explaining purpose and usage
- Inline comments for complex logic
- Type hints throughout
- Usage examples in docstrings

## See Also

- [Graph of Thought Overview](graph-of-thought.md) - Core GoT concepts
- [GoT CLI Specification](got-cli-spec.md) - Full CLI reference
- Source code docstrings for detailed API documentation
