# Graph of Thought Query API Guide

This guide covers the four query tools for exploring and analyzing the GoT graph.
Each tool has a specific purpose, but they share consistent patterns for ease of use.

## Quick Reference

| Class | Purpose | Best For |
|-------|---------|----------|
| `Query` | SQL-like filtering & aggregation | Finding tasks by status, counting, grouping |
| `GraphWalker` | Traversal with visitor pattern | Collecting stats, impact analysis |
| `PathFinder` | Path-finding algorithms | Dependency chains, blocked-by analysis |
| `PatternMatcher` | Subgraph pattern matching | Finding structural patterns, anti-patterns |

## Shared Patterns

All four classes share these design patterns:

### 1. Fluent Builder API

Chain methods to build your query:

```python
# All classes use method chaining
Query(manager).tasks().where(status="pending").limit(10).execute()
GraphWalker(manager).starting_from(id).bfs().max_depth(3).run()
PathFinder(manager).via_edges("DEPENDS_ON").max_paths(50).all_paths(a, b)
PatternMatcher(manager).limit(10).find(pattern)
```

### 2. `explain()` for Introspection

See what a query will do without executing:

```python
# Query
plan = Query(manager).tasks().where(status="pending").explain()
print(plan)  # Shows: scan, filter, sort steps

# GraphWalker
plan = GraphWalker(manager).starting_from(id).bfs().explain()
print(plan)  # Shows: strategy, depth, edge types

# PathFinder
plan = PathFinder(manager).via_edges("DEPENDS_ON").explain()
print(plan)  # Shows: algorithm, limits, complexity

# PatternMatcher
plan = PatternMatcher(manager).limit(10).explain(pattern)
print(plan)  # Shows: pattern structure, constraints
```

### 3. Consistent Direction Methods

Control edge traversal direction with the same names:

```python
# PatternMatcher
Pattern().node("A").outgoing("DEPENDS_ON").node("B")  # A -> B
Pattern().node("A").incoming("BLOCKS").node("B")       # B -> A
Pattern().node("A").both("CONTAINS").node("B")         # Either

# GraphWalker
walker.follow("DEPENDS_ON").outgoing()  # Forward edges only
walker.follow("DEPENDS_ON").incoming()  # Reverse edges only
walker.follow("DEPENDS_ON").both()      # Both directions (default)
```

### 4. Transparent Truncation

When limits are hit, you're told explicitly:

```python
# PathFinder returns PathSearchResult
result = PathFinder(manager).max_paths(10).all_paths(a, b)
if result.truncated:
    print(f"Found {result.paths_found} paths, stopped at {result.limit_value}")
for path in result:  # Still works like a list
    print(path)

# PatternMatcher returns PatternSearchResult
result = PatternMatcher(manager).limit(10).find(pattern)
if result.truncated:
    print(f"More matches exist beyond limit={result.limit_value}")
for match in result:
    print(match)
```

---

## Query: SQL-like Filtering

Best for: Finding entities by properties, counting, grouping.

### Basic Usage

```python
from cortical.got import Query

# Find pending high-priority tasks
tasks = (
    Query(manager)
    .tasks()
    .where(status="pending", priority="high")
    .order_by("created_at", desc=True)
    .limit(10)
    .execute()
)

# Count tasks by status
counts = (
    Query(manager)
    .tasks()
    .group_by("status")
    .count()
    .execute()
)
# Returns: {"pending": 5, "completed": 3, "in_progress": 2}
```

### Aggregation Functions

```python
from cortical.got.query_builder import Count, Collect, Avg, Min, Max

# Multiple aggregations
result = (
    Query(manager)
    .tasks()
    .group_by("priority")
    .aggregate(
        total=Count(),
        ids=Collect("id"),
        avg_age=Avg("age_days")
    )
    .execute()
)
```

### Lazy Iteration

```python
# Memory-efficient for large result sets
for task in Query(manager).tasks().iter():
    if should_stop(task):
        break  # Stops early, no wasted work
```

---

## GraphWalker: Traversal with Visitors

Best for: Collecting statistics, finding all connected nodes, impact analysis.

### Basic Usage

```python
from cortical.got import GraphWalker

# Count connected tasks by status
def count_by_status(node, acc):
    status = getattr(node, 'status', 'unknown')
    acc[status] = acc.get(status, 0) + 1
    return acc

result = (
    GraphWalker(manager)
    .starting_from(task_id)
    .bfs()
    .visit(count_by_status, initial={})
    .run()
)
# Returns: {"pending": 3, "completed": 2}
```

### Following Specific Edges

```python
# Only follow DEPENDS_ON edges, max 3 levels deep
ids = (
    GraphWalker(manager)
    .starting_from(task_id)
    .follow("DEPENDS_ON")
    .max_depth(3)
    .bfs()
    .visit(lambda n, acc: acc + [n.id], initial=[])
    .run()
)
```

### Direction Control

```python
# What depends on this task? (incoming DEPENDS_ON edges)
dependents = (
    GraphWalker(manager)
    .starting_from(task_id)
    .follow("DEPENDS_ON")
    .incoming()  # Reverse direction
    .bfs()
    .visit(lambda n, acc: acc + [n], initial=[])
    .run()
)
```

---

## PathFinder: Path Algorithms

Best for: Finding dependency chains, checking connectivity, blocked-by analysis.

### Basic Usage

```python
from cortical.got import PathFinder

# Shortest path between two tasks
path = (
    PathFinder(manager)
    .via_edges("DEPENDS_ON")
    .shortest_path(task_a, task_b)
)
# Returns: ["T-001", "T-002", "T-003"] or None

# Check if path exists (faster than shortest_path)
if PathFinder(manager).path_exists(task_a, task_b):
    print("Connected!")
```

### Finding All Paths

```python
# Find all paths (with safety limits)
result = PathFinder(manager).all_paths(task_a, task_b)

# Default limits: max_paths=100, max_length=10
# Override with .max_paths(N) and .max_length(N)

if result.truncated:
    print(f"Warning: stopped at {result.truncation_reason}={result.limit_value}")

for path in result:
    print(" -> ".join(path))
```

### Reachability

```python
# Find all tasks reachable from this one
reachable = (
    PathFinder(manager)
    .via_edges("DEPENDS_ON", "BLOCKS")
    .reachable_from(task_id)
)
# Returns: {"T-001", "T-002", "T-003"}
```

---

## PatternMatcher: Subgraph Matching

Best for: Finding structural patterns, detecting anti-patterns, graph queries.

### Basic Usage

```python
from cortical.got import PatternMatcher, Pattern

# Find tasks blocking high-priority work
pattern = (
    Pattern()
    .node("blocker", type="task")
    .outgoing("BLOCKS")
    .node("blocked", type="task", priority="high")
)

result = PatternMatcher(manager).find(pattern)
for match in result:
    print(f"{match['blocker'].title} blocks {match['blocked'].title}")
```

### Dependency Chains

```python
# Find 3-node dependency chains
pattern = (
    Pattern()
    .node("a", type="task")
    .incoming("DEPENDS_ON")
    .node("b", type="task")
    .incoming("DEPENDS_ON")
    .node("c", type="task")
)

for match in PatternMatcher(manager).find(pattern):
    print(f"{match['c'].title} -> {match['b'].title} -> {match['a'].title}")
```

### Pattern with Limit

```python
# Find first 10 matches only
result = PatternMatcher(manager).limit(10).find(pattern)

if result.truncated:
    print(f"More matches may exist (stopped at {result.limit_value})")
```

---

## Choosing the Right Tool

| Scenario | Best Tool |
|----------|-----------|
| "Find all pending tasks" | `Query` |
| "Count tasks by status" | `Query` with `group_by().count()` |
| "What tasks are connected to X?" | `GraphWalker` with BFS |
| "What's the dependency chain?" | `PathFinder.shortest_path()` |
| "Is task A blocked by task B?" | `PathFinder.path_exists()` |
| "Find all circular dependencies" | `PatternMatcher` with cycle pattern |
| "What tasks block high-priority work?" | `PatternMatcher` with constraint |
| "Collect stats on connected nodes" | `GraphWalker` with visitor |

---

## Performance Tips

1. **Use `.limit()`** - Avoid loading all results when you need few
2. **Use `.iter()`** (Query) - Memory-efficient streaming for large sets
3. **Use `.first()`** - When you only need one result
4. **Use `.exists()`** (Query) or `.path_exists()` (PathFinder) - Boolean checks are faster
5. **Use `.explain()`** - Understand query cost before executing
6. **Filter early** - Apply `.where()` before expensive operations

---

## Complete Example

```python
from cortical.got import GoTManager, Query, GraphWalker, PathFinder, PatternMatcher, Pattern

# Initialize
manager = GoTManager()

# 1. Query: Find blocked tasks
blocked = Query(manager).tasks().where(status="blocked").execute()

# 2. GraphWalker: For each, find impact radius
for task in blocked:
    impacted = (
        GraphWalker(manager)
        .starting_from(task.id)
        .follow("BLOCKS")
        .outgoing()
        .max_depth(5)
        .bfs()
        .visit(lambda n, acc: acc + [n.id], initial=[])
        .run()
    )
    print(f"{task.id} blocks {len(impacted)} downstream tasks")

# 3. PathFinder: Find shortest resolution path
path = PathFinder(manager).via_edges("DEPENDS_ON").shortest_path(blocked[0].id, "root-task")
if path:
    print(f"Resolution path: {' -> '.join(path)}")

# 4. PatternMatcher: Find high-priority tasks blocked by low-priority
antipattern = (
    Pattern()
    .node("blocker", type="task", priority="low")
    .outgoing("BLOCKS")
    .node("blocked", type="task", priority="high")
)
for match in PatternMatcher(manager).find(antipattern):
    print(f"Priority inversion: {match['blocker'].id} -> {match['blocked'].id}")
```
