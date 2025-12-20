# Graph of Thought Query Language

The GoT query language provides a simple way to traverse and query the project management graph. It supports relationship queries, path finding, and status-based filtering.

## Quick Reference

```bash
# Relationship queries
python scripts/got_utils.py query "what blocks task:T-..."
python scripts/got_utils.py query "what depends on task:T-..."
python scripts/got_utils.py query "relationships task:T-..."

# Path finding
python scripts/got_utils.py query "path from task:T-1 to task:T-2"

# Status queries
python scripts/got_utils.py query "active tasks"
python scripts/got_utils.py query "pending tasks"
python scripts/got_utils.py query "blocked tasks"
```

## Query Types

### 1. What Blocks

Find all tasks that are blocking a given task.

```bash
python scripts/got_utils.py query "what blocks task:T-20251220-123456-abcd"
```

**Returns:** List of tasks with BLOCKS edges pointing to the target task.

**Example output:**
```
Query: what blocks task:T-20251220-123456-abcd

Results (2):

  task:T-20251220-111111-aaaa: Fix database connection
      Status: in_progress

  task:T-20251220-222222-bbbb: Update API schema
      Status: pending
```

### 2. What Depends On

Find all tasks that depend on a given task (i.e., tasks that have this task as a dependency).

```bash
python scripts/got_utils.py query "what depends on task:T-20251220-123456-abcd"
```

**Returns:** List of tasks with DEPENDS_ON edges pointing to the target task.

### 3. Relationships

Get all relationships for a task in one query.

```bash
python scripts/got_utils.py query "relationships task:T-20251220-123456-abcd"
```

**Returns:** All edges connected to the task, categorized by type:
- `blocks` - Tasks this task blocks
- `blocked_by` - Tasks blocking this task
- `depends_on` - Tasks this task depends on
- `depended_by` - Tasks depending on this task
- `in_sprint` - Sprint containing this task

### 4. Path From...To

Find a path between two nodes using BFS (shortest path).

```bash
python scripts/got_utils.py query "path from task:T-1 to task:T-2"
```

**Returns:** Ordered list of nodes forming the path, or empty if no path exists.

**Example output:**
```
Query: path from task:T-1 to task:T-5

Results (3):

  [0] task:T-1: Initial setup
  [1] task:T-3: Core implementation
  [2] task:T-5: Final integration
```

### 5. Status Queries

#### Active Tasks
Tasks currently in progress.

```bash
python scripts/got_utils.py query "active tasks"
```

#### Pending Tasks
Tasks waiting to be started.

```bash
python scripts/got_utils.py query "pending tasks"
```

#### Blocked Tasks
Tasks that are blocked, with their blocking reasons.

```bash
python scripts/got_utils.py query "blocked tasks"
```

## Programmatic API

The query language is also available programmatically:

```python
from scripts.got_utils import GoTProjectManager

manager = GoTProjectManager()

# Using the query string interface
results = manager.query("what blocks task:T-123")

# Or using direct methods
blockers = manager.what_blocks("task:T-123")
dependents = manager.what_depends_on("task:T-123")
path = manager.find_path("task:T-1", "task:T-5")
relationships = manager.get_all_relationships("task:T-123")
```

### Method Reference

| Method | Description | Returns |
|--------|-------------|---------|
| `what_blocks(task_id)` | Tasks blocking this task | `List[ThoughtNode]` |
| `what_depends_on(task_id)` | Tasks depending on this task | `List[ThoughtNode]` |
| `find_path(from_id, to_id)` | Shortest path between nodes | `Optional[List[ThoughtNode]]` |
| `get_all_relationships(task_id)` | All edges for a task | `Dict[str, List[ThoughtNode]]` |
| `query(query_str)` | Parse and execute query | `List[Dict[str, Any]]` |

## Edge Types

The query language understands these edge types:

| Edge Type | Meaning | Example |
|-----------|---------|---------|
| `BLOCKS` | Source blocks target from starting | Task A blocks Task B |
| `DEPENDS_ON` | Source requires target to complete first | Task A depends on Task B |
| `CONTAINS` | Container relationship | Sprint contains Task |
| `RELATES_TO` | General relationship | Task relates to Epic |

## Creating Relationships

To create edges that can be queried:

```bash
# Add dependency (Task A depends on Task B)
python scripts/got_utils.py task create "New feature" --depends task:T-existing

# Block a task (creates BLOCKS edge)
python scripts/got_utils.py task block task:T-123 --reason "Waiting for API" --blocker task:T-456
```

## Tips

1. **Task IDs are case-sensitive** - Use the full ID including the `task:` prefix
2. **Queries are case-insensitive** - "What Blocks" and "what blocks" are equivalent
3. **Path finding uses BFS** - Returns shortest path by edge count, not weight
4. **Empty results are valid** - A task with no blockers returns an empty list

## See Also

- [GoT Event Sourcing](got-event-sourcing.md) - How events are stored
- [Graph of Thought](graph-of-thought.md) - Core reasoning framework
- [CLAUDE.md Quick Reference](../CLAUDE.md) - All CLI commands
