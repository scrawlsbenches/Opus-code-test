# Merge-Friendly Task Management

This document describes the merge-friendly task ID system for parallel agent workflows.

## The Problem

When multiple Claude agents run in parallel on the same repository:
- Both might create task `#239` (next sequential ID)
- Both modify `TASK_LIST.md` simultaneously
- Git merge conflicts are guaranteed

## The Solution

Use **timestamp-based, session-scoped task IDs** that can't collide:

```
T-20251213-143052-a1b2
│ │        │      │
│ │        │      └── 4-char session suffix (unique per agent session)
│ │        └── Time created (HHMMSS)
│ └── Date created (YYYYMMDD)
└── Task prefix
```

Combined with **per-session task files**:

```
tasks/
├── 2025-12-13_14-30-52_a1b2.json    # Agent A's session
├── 2025-12-13_14-31-05_c3d4.json    # Agent B's session
└── ...
```

## How It Works

### 1. Each Agent Creates Its Own Session

```python
from scripts.task_utils import TaskSession

# Start a session (gets unique suffix like "a1b2")
session = TaskSession()

# Create tasks (all get same suffix)
task1 = session.create_task(
    title="Implement feature X",
    priority="high",
    category="arch",
    description="...",
    effort="medium"
)

task2 = session.create_task(
    title="Add tests for feature X",
    priority="medium",
    category="test",
    depends_on=[task1.id]
)

# Save to tasks/2025-12-13_14-30-52_a1b2.json
session.save()
```

### 2. No Merge Conflicts

Each agent writes to a **unique filename**:
- Agent A: `tasks/2025-12-13_14-30-52_a1b2.json`
- Agent B: `tasks/2025-12-13_14-31-05_c3d4.json`

Files never conflict because:
1. Timestamps are different (even by milliseconds)
2. Session IDs are randomly generated
3. Each agent only writes to its own file

### 3. Consolidation (Like `git gc`)

Periodically consolidate task files:

```bash
# Show summary of all tasks
python scripts/consolidate_tasks.py --summary

# Auto-merge duplicates and consolidate
python scripts/consolidate_tasks.py --update --auto-merge

# Archive old session files after consolidation
python scripts/consolidate_tasks.py --update --archive
```

## Task ID Formats

### Full Format (Default)
```
T-20251213-143052-a1b2
```
- Sortable by creation time
- Self-documenting (when it was created)
- Session-traceable (which agent created it)

### Short Format
```
T-a1b2c3d4
```
- More compact (8 hex chars)
- Still practically unique
- Good for quick references

```python
from scripts.task_utils import generate_short_task_id
task_id = generate_short_task_id()  # T-a1b2c3d4
```

## CLI Commands

### Generate Task ID
```bash
# Full format
python scripts/task_utils.py generate
# Output: T-20251213-143052-a1b2

# Short format
python scripts/task_utils.py generate --short
# Output: T-a1b2c3d4
```

### List All Tasks
```bash
# List all tasks
python scripts/task_utils.py list

# Filter by status
python scripts/task_utils.py list --status pending
```

### Consolidate Tasks
```bash
# Dry run (see what would happen)
python scripts/consolidate_tasks.py --dry-run

# Consolidate with summary
python scripts/consolidate_tasks.py --update

# Auto-merge duplicates
python scripts/consolidate_tasks.py --update --auto-merge
```

## Comparison with Legacy System

| Aspect | Legacy (`#133`) | New (`T-a1b2c3d4`) |
|--------|-----------------|---------------------|
| Collision risk | High (parallel agents) | ~Zero |
| Human readable | Very easy | Moderate |
| Git-friendly | Conflicts guaranteed | No conflicts |
| Sorting | Natural numeric | Chronological |
| Traceability | None | Session + timestamp |

## Best Practices

### For Parallel Agents

1. **Always create a session** at the start of your work
2. **Save the session** before your work is committed
3. **Reference tasks by full ID** in commits and comments

### For Consolidation

1. **Run consolidation weekly** (or after parallel agent runs)
2. **Use `--auto-merge`** to deduplicate similar tasks
3. **Archive old files** to keep the directory clean

### For Migration

The new system can coexist with legacy `TASK_LIST.md`:
- Legacy tasks keep their `#123` format
- New tasks use `T-...` format
- Both can be referenced and tracked

## Architecture

```
tasks/
├── 2025-12-13_14-30-52_a1b2.json    # Agent sessions (append-only)
├── 2025-12-13_14-31-05_c3d4.json
├── consolidated_2025-12-13.json     # Periodic consolidation
└── archive/                          # Archived old files
    └── ...

TASK_LIST.md                          # Optional: human-readable summary
```

This mirrors the `chunk_index.py` architecture for corpus indexing.

## Task File Format

```json
{
  "version": 1,
  "session_id": "a1b2",
  "started_at": "2025-12-13T14:30:52",
  "saved_at": "2025-12-13T14:35:00",
  "tasks": [
    {
      "id": "T-20251213-143052-a1b2",
      "title": "Implement feature X",
      "status": "pending",
      "priority": "high",
      "category": "arch",
      "description": "Detailed description...",
      "depends_on": [],
      "effort": "medium",
      "created_at": "2025-12-13T14:30:52",
      "updated_at": null,
      "completed_at": null,
      "context": {
        "files": ["cortical/processor.py"],
        "methods": ["compute_all()"]
      }
    }
  ]
}
```

## Future Enhancements

1. **Real-time sync**: Watch for file changes and auto-consolidate
2. **Web UI**: Visual task board from consolidated data
3. **GitHub Issues sync**: Two-way sync with GitHub Issues
4. **Task dependencies**: Topological sorting for execution order

---

*This system follows the same principles as `cortical/chunk_index.py` - append-only, git-friendly, merge-conflict-free.*
