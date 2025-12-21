# GoT Resume - Session Continuity Command

Resume work from Graph of Thought state. Use this at the start of any new session to restore context.

---

## Instructions

You are resuming work on the Cortical Text Processor project. The Graph of Thought (GoT) system persists your state across sessions, branches, and environments.

### Step 1: Load Current State

Run this to see what you were working on:

```bash
python scripts/got_utils.py stats && python scripts/got_utils.py task list --status in_progress
```

### Step 2: Check Recent Events

Look at the most recent events to understand what happened last session:

```bash
tail -20 .got/events/*.jsonl 2>/dev/null | python -c "import sys,json; [print(f\"{json.loads(l).get('event')}: {json.loads(l).get('id','')[:40]}\") for l in sys.stdin if l.strip()]"
```

### Step 3: Check Sprint Status

```bash
python scripts/got_utils.py sprint status
```

### Step 4: Review Pending Tasks

```bash
python scripts/got_utils.py task list --status pending | head -20
```

---

## Context Recovery

If you need more context:

1. **Recent commits**: `git log --oneline -10`
2. **Current branch**: `git branch --show-current`
3. **Recent memories**: `ls -t samples/memories/*.md | head -5`
4. **Sprint goals**: `cat tasks/CURRENT_SPRINT.md | head -50`

---

## Key Files

- **GoT Events**: `.got/events/*.jsonl` (source of truth)
- **Sprint Status**: `tasks/CURRENT_SPRINT.md`
- **Architecture**: `docs/got-event-sourcing.md`
- **CLI**: `scripts/got_utils.py`

---

## Common Actions

### Start a new task
```bash
python scripts/got_utils.py task create "Task title" --priority high
```

### Complete current work
```bash
python scripts/got_utils.py task complete <task_id>
```

### Sync before committing
```bash
python scripts/got_utils.py sync -m "description"
```

---

## If Environment Was Lost

If you're on a fresh clone:

1. GoT events are git-tracked, so state survives
2. Run `python scripts/got_utils.py stats` to verify
3. If empty, check `git log` for context

---

Now proceed to help the user with their current task, using the GoT state as your persistent memory.
