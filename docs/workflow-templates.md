# Workflow Templates

Workflow templates enable creating multiple linked tasks from a single command. This is useful for standardizing common development patterns like bug fixes, features, and refactors.

## Quick Start

```bash
# List available workflows
python scripts/workflow.py list

# Run the bugfix workflow
python scripts/workflow.py run bugfix --bug_title "Login fails with special characters"

# Dry run (see tasks without creating)
python scripts/workflow.py run feature --feature_name "Dark mode" --dry-run
```

## Available Workflows

### Bug Fix (`bugfix`)

Creates 4 tasks with dependencies:

```
investigate → fix → test
                 → document
```

**Variables:**
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `bug_title` | Yes | - | Brief description of the bug |
| `priority` | No | high | Bug priority (high/medium/low) |
| `affected_file` | No | - | Primary file affected |

**Example:**
```bash
python scripts/workflow.py run bugfix \
    --bug_title "Search returns stale results" \
    --priority high
```

### Feature (`feature`)

Creates 5 tasks with dependencies:

```
design → implement → unit_tests → documentation
                  → integration_tests ↗
```

**Variables:**
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `feature_name` | Yes | - | Name of the feature |
| `priority` | No | medium | Feature priority |
| `effort` | No | large | Overall effort (small/medium/large) |

**Example:**
```bash
python scripts/workflow.py run feature \
    --feature_name "Semantic search" \
    --priority high \
    --effort large
```

### Refactor (`refactor`)

Creates 4 tasks with dependencies:

```
analyze → plan → execute → verify
```

**Variables:**
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `refactor_target` | Yes | - | What to refactor |
| `priority` | No | medium | Refactor priority |
| `scope` | No | module | Scope (function/module/system) |

**Example:**
```bash
python scripts/workflow.py run refactor \
    --refactor_target "Query expansion logic" \
    --scope module
```

## Creating Custom Workflows

Workflows are YAML files in `.claude/workflows/`. Here's the structure:

```yaml
# .claude/workflows/my_workflow.yaml
name: "My Workflow"
description: "Brief description of what this workflow does"
category: "general"

variables:
  - name: task_name
    description: "The name of the task"
    required: true
  - name: priority
    description: "Task priority"
    default: "medium"
    choices: ["high", "medium", "low"]

tasks:
  - id: first_task
    title: "First: {task_name}"
    category: "planning"
    priority: "{priority}"
    effort: "small"
    description: |
      Description with {task_name} substitution.

  - id: second_task
    title: "Second: {task_name}"
    depends_on: [first_task]
    description: |
      This task depends on first_task completing.
```

### Variable Types

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Variable name (used in `{name}` placeholders) |
| `description` | No | Help text shown to users |
| `required` | No | If true, must be provided (default: true) |
| `default` | No | Default value if not provided |
| `choices` | No | List of valid values |

### Task Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `id` | Yes | - | Unique ID within workflow (used for depends_on) |
| `title` | Yes | - | Task title (supports `{variable}` substitution) |
| `category` | No | general | Task category |
| `priority` | No | medium | Task priority (supports substitution) |
| `effort` | No | medium | Effort estimate (supports substitution) |
| `description` | No | "" | Detailed description |
| `depends_on` | No | [] | List of task IDs this depends on |

## API Usage

You can also use workflows programmatically:

```python
from scripts.workflow import Workflow, run_workflow

# Load a workflow
workflow = Workflow.load(Path(".claude/workflows/bugfix.yaml"))

# Run with variables
variables = {"bug_title": "Login bug", "priority": "high"}
tasks = run_workflow(workflow, variables, tasks_dir="tasks", dry_run=False)

# Access created tasks
for task in tasks:
    print(f"{task.id}: {task.title}")
```

## Task Output

Tasks are saved as JSON files in the `tasks/` directory:

```
tasks/
├── 2025-12-13_14-30-52_a1b2.json  # Session file with tasks
└── 2025-12-13_15-00-00_c3d4.json  # Another session
```

Each session file contains:
```json
{
  "version": 1,
  "session_id": "a1b2",
  "started_at": "2025-12-13T14:30:52",
  "saved_at": "2025-12-13T14:30:53",
  "tasks": [
    {
      "id": "T-20251213-143052-a1b2-001",
      "title": "Investigate: Login bug",
      "status": "pending",
      "priority": "high",
      "depends_on": []
    },
    {
      "id": "T-20251213-143052-a1b2-002",
      "title": "Fix: Login bug",
      "depends_on": ["T-20251213-143052-a1b2-001"]
    }
  ]
}
```

## Best Practices

1. **Use dry-run first**: Always preview with `--dry-run` before creating tasks
2. **Keep workflows focused**: Each workflow should handle one type of work
3. **Use dependencies**: Link tasks to show the correct order
4. **Descriptive titles**: Include the variable in titles for context
5. **Add descriptions**: Include checklists and acceptance criteria

## Integration with CI

The CI pipeline shows pending tasks using `scripts/ci_task_report.py`:

```bash
# GitHub Actions format
python scripts/ci_task_report.py --github

# Console format
python scripts/ci_task_report.py

# Fail if high-priority tasks exist
python scripts/ci_task_report.py --fail-on-high
```
