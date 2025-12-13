# Task Management Dog-Fooding Guide

This guide explains how to use the task management system to track its own development - eating our own dog food.

## Overview

The task management system is designed for parallel Claude agent workflows. We use it to:
1. Track our own development tasks
2. Coordinate between multiple agent sessions
3. Ensure nothing falls through the cracks

## Quick Reference

```bash
# List all tasks
python scripts/new_task.py --list

# Create a task
python scripts/new_task.py "Fix bug X" --priority high --category bugfix

# Complete a task
python scripts/new_task.py --complete T-XXXXX

# View task summary
python scripts/new_task.py --summary

# Use workflow templates
python scripts/workflow.py run bugfix --bug_title "Description"

# CI-friendly report
python scripts/ci_task_report.py --github
```

## Dog-Fooding Workflow

### 1. Start of Session

At the beginning of each development session:

```bash
# Check pending tasks
python scripts/new_task.py --list

# Identify high-priority items
python scripts/ci_task_report.py --quiet
# Output: Tasks: 5 pending (🔴2 🟡3 🟢0)
```

**Key questions:**
- Are there high-priority tasks I should address first?
- Are there tasks from previous sessions that are stale?
- What was I working on last time?

### 2. Creating Tasks

When you discover work to do:

```bash
# Quick task
python scripts/new_task.py "Investigate slow search" --priority high

# With workflow template (creates linked tasks)
python scripts/workflow.py run bugfix --bug_title "Search returns wrong results"

# Dry run first to preview
python scripts/workflow.py run feature --feature_name "New feature" --dry-run
```

**Best practices:**
- Create tasks immediately when you discover them
- Use workflow templates for standard patterns
- Set priority based on impact, not urgency
- Include category for filtering

### 3. Working on Tasks

When starting work:

```bash
# Check what's pending
python scripts/new_task.py --list --status pending

# Pick a task and start working
# (The system doesn't track "in_progress" automatically - you manage it)
```

### 4. Completing Tasks

After finishing work:

```bash
# Mark task as done
python scripts/new_task.py --complete T-20251213-143052-a1b2-001

# Verify completion
python scripts/new_task.py --list --status completed
```

**When to mark complete:**
- Code is written and tested
- Changes are committed
- Related documentation updated
- No blocking issues remain

### 5. End of Session

Before ending your session:

```bash
# Verify no orphaned tasks
python scripts/new_task.py --list

# Create continuation tasks for unfinished work
python scripts/new_task.py "Continue: Feature X implementation" --description "Left off at..."

# Commit task files
git add tasks/*.json
git commit -m "Update task status"
git push
```

## Session Continuity

The system maintains session state across CLI invocations:

```
tasks/
├── .current_session.json       # Current session metadata
├── 2025-12-13_14-30-52_a1b2.json  # Session 1 tasks
└── 2025-12-13_16-00-00_b2c3.json  # Session 2 tasks
```

### Starting a New Session

```bash
# Start fresh session (clears .current_session.json)
python scripts/new_task.py --new-session

# Tasks in new session get new IDs
python scripts/new_task.py "First task in new session"
# Creates: T-20251213-160000-c3d4-001
```

### Resuming Previous Session

Sessions persist until you explicitly start a new one. Task IDs continue incrementing.

## Parallel Agent Workflows

### Multiple Agents Working Simultaneously

Each agent creates tasks with unique session IDs:

```
Agent A: T-20251213-143052-a1b2-001, T-20251213-143052-a1b2-002
Agent B: T-20251213-143055-c3d4-001, T-20251213-143055-c3d4-002
```

**No merge conflicts:** Different session IDs mean different filenames.

### Consolidating Work

After parallel work completes:

```bash
# View all tasks from all sessions
python scripts/new_task.py --list

# Generate consolidated report
python scripts/consolidate_tasks.py --output CONSOLIDATED.md
```

## CI Integration

The CI pipeline automatically shows pending tasks:

```yaml
# In .github/workflows/ci.yml
- name: Report Pending Tasks
  run: python scripts/ci_task_report.py --github
```

### CI Output Formats

```bash
# GitHub Actions (markdown tables)
python scripts/ci_task_report.py --github

# Console (readable)
python scripts/ci_task_report.py

# Minimal (one-liner)
python scripts/ci_task_report.py --quiet

# Fail if high-priority tasks exist
python scripts/ci_task_report.py --fail-on-high
```

## Workflow Templates

Pre-defined task chains for common patterns:

| Workflow | Tasks | Use When |
|----------|-------|----------|
| `bugfix` | investigate → fix → test → document | Fixing bugs |
| `feature` | design → implement → unit_tests → integration_tests → docs | Adding features |
| `refactor` | analyze → plan → execute → verify | Restructuring code |

### Creating Custom Workflows

```yaml
# .claude/workflows/my_workflow.yaml
name: "My Workflow"
description: "Custom task chain"
variables:
  - name: target
    required: true
tasks:
  - id: step1
    title: "First: {target}"
  - id: step2
    title: "Second: {target}"
    depends_on: [step1]
```

## Testing the Task System

When making changes to the task system itself:

### 1. Run Unit Tests

```bash
python -m pytest tests/unit/test_task_utils.py tests/unit/test_workflow.py -v
```

### 2. Run Integration Tests

```bash
python -m pytest tests/integration/test_task_integration.py tests/integration/test_workflow_integration.py -v
```

### 3. Manual Dog-Fooding

```bash
# Create a test task
python scripts/new_task.py "Test task" --priority low

# Verify it appears
python scripts/new_task.py --list

# Complete it
python scripts/new_task.py --complete T-XXXXX

# Verify completion
python scripts/new_task.py --list --status completed

# Clean up (optional)
# Delete the session file from tasks/
```

## Common Patterns

### Bug Discovery During Feature Work

```bash
# You're working on feature X, discover bug Y
python scripts/new_task.py "Bug: Y found during X" --priority high --category bugfix
# Continue with feature X, bug Y is tracked
```

### Splitting Large Tasks

```bash
# Original task too big
python scripts/workflow.py run feature --feature_name "Large Feature"
# Creates 5 linked tasks automatically

# Or manually create subtasks
python scripts/new_task.py "Part 1: Setup" --priority high
python scripts/new_task.py "Part 2: Core logic" --priority high
python scripts/new_task.py "Part 3: Tests" --priority high
```

### Handling Blocked Tasks

If a task is blocked:
1. Don't mark it complete
2. Create a new task for the blocker
3. Add description noting the block

```bash
python scripts/new_task.py "Blocked: Need API access for X" --priority high
```

## Metrics and Reporting

Track progress over time:

```bash
# Summary counts
python scripts/new_task.py --summary

# Output:
# pending: 5
# in_progress: 0
# completed: 12
# deferred: 1

# CI report with priority breakdown
python scripts/ci_task_report.py
```

## Files and Locations

| Path | Purpose |
|------|---------|
| `tasks/*.json` | Task session files |
| `tasks/.current_session.json` | Active session state |
| `.claude/workflows/*.yaml` | Workflow templates |
| `.claude/skills/task-manager/` | Claude Code skill definition |
| `scripts/task_utils.py` | Core task utilities |
| `scripts/new_task.py` | CLI for task management |
| `scripts/workflow.py` | Workflow template engine |
| `scripts/ci_task_report.py` | CI-friendly task reporter |
| `scripts/consolidate_tasks.py` | Task consolidation |

## Troubleshooting

### Tasks Not Showing Up

```bash
# Check tasks directory exists
ls -la tasks/

# Check for valid JSON
cat tasks/*.json | python -m json.tool
```

### Duplicate Task IDs

This shouldn't happen due to timestamp + session + counter format. If it does:

```bash
# Check session file
cat tasks/.current_session.json

# Start new session
python scripts/new_task.py --new-session
```

### CI Report Empty

```bash
# Ensure tasks directory exists
mkdir -p tasks

# Check if any tasks exist
python scripts/task_utils.py list --dir tasks
```

---

*Remember: We build this system for ourselves. If something is painful, fix it and add a task for improving it.*
