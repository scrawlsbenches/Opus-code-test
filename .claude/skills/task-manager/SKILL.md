---
name: task-manager
description: Manage tasks with merge-friendly IDs for parallel agent workflows. Use when creating, updating, or querying tasks during development. Prevents conflicts when multiple agents run in parallel.
allowed-tools: Read, Bash, Write
---
# Task Manager Skill

This skill enables **merge-friendly task management** for parallel agent workflows. It uses timestamp+session IDs that can't conflict when multiple agents work simultaneously.

## Key Capabilities

- **Conflict-free task creation**: Each agent writes to its own session file
- **Unique task IDs**: `T-YYYYMMDD-HHMMSS-XXXX` format
- **Session tracking**: All tasks from one session share a suffix
- **Consolidation**: Merge task files like `git gc`
- **TASK_LIST.md compatible**: Can generate markdown summaries

## When to Use

- Starting a new piece of work that needs tracking
- Creating tasks from parallel agent workflows
- Consolidating tasks after multi-agent runs
- Querying task status across sessions

## Quick Start

### Create Tasks

```python
# In Python
from scripts.task_utils import TaskSession

session = TaskSession()
task = session.create_task(
    title="Implement feature X",
    priority="high",
    category="arch",
    description="Add new capability to processor"
)
print(f"Created: {task.id}")  # T-20251213-143052-a1b2
session.save()  # → tasks/2025-12-13_14-30-52_a1b2.json
```

### Generate Task ID (CLI)

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
python scripts/task_utils.py list
python scripts/task_utils.py list --status pending
```

### Consolidate Tasks

```bash
# See summary
python scripts/consolidate_tasks.py --summary

# Consolidate and deduplicate
python scripts/consolidate_tasks.py --update --auto-merge

# Archive old session files
python scripts/consolidate_tasks.py --update --archive
```

## Task Structure

```json
{
  "id": "T-20251213-143052-a1b2",
  "title": "Implement feature X",
  "status": "pending",
  "priority": "high",
  "category": "arch",
  "description": "...",
  "depends_on": ["T-20251213-143000-c3d4"],
  "effort": "medium",
  "context": {
    "files": ["cortical/processor.py"],
    "methods": ["compute_all()"]
  }
}
```

## Integration with TASK_LIST.md

The new system can coexist with the legacy `TASK_LIST.md`:
- Legacy tasks keep `#123` format
- New parallel work uses `T-...` format
- Both can be referenced and tracked

## Tips

1. **Create session at workflow start** - all tasks share session suffix
2. **Save before commit** - persist tasks to disk
3. **Consolidate weekly** - merge sessions, resolve duplicates
4. **Use context field** - add file/method references for quick navigation
