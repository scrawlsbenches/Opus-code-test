# Knowledge Transfer: GoT Reliability & Process Improvements

**Date:** 2025-12-21
**Branch:** `claude/test-task-workflow-gRXq7`
**Session Focus:** GoT bug fixes, process improvements, testing

---

## Executive Summary

This session discovered and fixed a critical bug in GoT edge persistence that caused 99% edge loss. We also improved CLAUDE.md with work priority rules, added regression tests, implemented a `got validate` command, and created tasks for future improvements.

---

## Critical Bug Fixed

### Edge Rebuild Failure (T-20251221-014654-d4b7)

**Problem:** `rebuild_graph_from_events()` in `scripts/got_utils.py` was silently failing to create edges.

**Root Cause:** Two issues:
1. Wrong parameter names: `source_id`/`target_id` instead of `from_id`/`to_id`
2. EdgeType lookup used `hasattr()` which doesn't work correctly with enums

**Fix Location:** `scripts/got_utils.py:448-457`

```python
# BEFORE (broken)
graph.add_edge(
    source_id=event["src"],  # WRONG
    target_id=event["tgt"],  # WRONG
    edge_type=edge_type,
)

# AFTER (fixed)
graph.add_edge(
    from_id=event["src"],   # CORRECT
    to_id=event["tgt"],     # CORRECT
    edge_type=edge_type,
)
```

**Impact:** Edges increased from 3 → 325, orphan nodes reduced from 285 → 144.

---

## New Features Added

### 1. `got validate` Command

**Usage:** `python scripts/got_utils.py validate`

**Purpose:** Health check that catches silent failures like the edge rebuild bug.

**Checks:**
- Edge loss rate (compares graph vs event log)
- Orphan node rate
- Edge density
- Event log statistics

**Exit Codes:**
- 0 = healthy
- 1 = critical issues detected

### 2. Work Priority Rules (CLAUDE.md)

Added to `CLAUDE.md` under "Development Workflow":

| Priority | Type | Rule |
|----------|------|------|
| 1 | Security | Fix BEFORE any code work |
| 2 | Bugs | Fix BEFORE features |
| 3 | Features | After security/bugs |
| 4 | Documentation | Update AS you work |

### 3. Architecture Diagram

Created `docs/architecture-diagram.md` with Mermaid class diagrams.

### 4. Regression Tests

Created `tests/regression/test_got_edge_rebuild.py` with 14 tests covering:
- All EdgeType values rebuild correctly
- Unknown types fall back gracefully
- Missing nodes are skipped
- Weight preservation
- Parameter validation

---

## Key Discoveries

### 1. Event Subscription Systems Exist

Two pub/sub systems are available in `cortical/reasoning/`:

| System | File | Purpose |
|--------|------|---------|
| `PubSubBroker` | `pubsub.py` | Inter-agent messaging |
| `ContextPool` | `context_pool.py` | Discovery sharing |

**Usage:**
```python
from cortical.reasoning import PubSubBroker, ContextPool

# PubSubBroker
broker = PubSubBroker()
broker.subscribe("agent.task.*", "worker1")
broker.publish("agent.task.done", {"result": "success"}, "orchestrator")

# ContextPool
pool = ContextPool()
pool.subscribe("findings", lambda f: print(f.content))
pool.publish("findings", "Found bug", "agent_a", confidence=0.9)
```

### 2. Decision Command Usage

Correct format (requires `--rationale`):
```bash
python scripts/got_utils.py decision log "Your decision" --rationale "Why this choice"
```

Exit code 2 = missing required argument (not a bug).

### 3. Priority Executor Bug (Also Fixed)

`scripts/got_priority_executor.py` was marking all tasks as BLOCKED due to parsing the query echo line. Fixed by skipping first line and excluding self-references.

---

## Current GoT State

| Metric | Value |
|--------|-------|
| Total Tasks | 269 |
| Completed | 168 (62.5%) |
| Pending | 101 |
| Edges | 325 |
| Edge Density | 0.81/node |
| Orphan Rate | 35.8% (warning level) |

---

## High Priority Tasks Remaining

| ID | Title | Priority |
|----|-------|----------|
| `T-20251221-020047-ecf6` | Add rebuild validation with telemetry | **Critical** |
| `T-20251221-020047-843a` | Implement atomic save wrapper | High |
| `T-20251221-020101-afc4` | Add auto-task creation hook | High |
| `T-20251220-194436-f39a` | Implement NestedLoopExecutor | High |
| `T-20251220-194436-d053` | Implement ClaudeCodeSpawner | High |

---

## Process Improvements Needed

1. **Auto-task creation:** Currently must manually create GoT tasks - easy to forget
2. **Commit-task linking:** Commits don't reference task IDs
3. **Rebuild telemetry:** Log success/failure counts during event replay
4. **Atomic saves:** graph + event log can get out of sync

---

## Files Modified This Session

| File | Change |
|------|--------|
| `scripts/got_utils.py` | Fixed edge rebuild, added validate command |
| `scripts/got_priority_executor.py` | Fixed blocker detection |
| `CLAUDE.md` | Added priority rules, updated coverage baseline to 89%, removed MCP refs |
| `tests/regression/test_got_edge_rebuild.py` | New - 14 regression tests |
| `docs/architecture-diagram.md` | New - Mermaid class diagrams |

---

## Commands to Resume Work

```bash
# Check current state
python scripts/got_utils.py validate
python scripts/got_utils.py dashboard

# See high priority tasks
python scripts/got_utils.py task list --status pending --priority high

# Start a task
python scripts/got_utils.py task start T-XXXXXX

# Run tests
python -m pytest tests/regression/test_got_edge_rebuild.py -v
```

---

## Lessons Learned

1. **Silent failures are dangerous** - The edge rebuild bug lost 99% of data with no errors
2. **Verify after operations** - The new `got validate` command catches these issues
3. **Test parameter names** - TypeErrors can be silently swallowed
4. **hasattr() doesn't work for enums** - Use try/except with bracket access instead
5. **Track work in GoT** - I initially forgot to create tasks for my bug fixes

---

*Generated: 2025-12-21T02:22Z*
*Branch: claude/test-task-workflow-gRXq7*
