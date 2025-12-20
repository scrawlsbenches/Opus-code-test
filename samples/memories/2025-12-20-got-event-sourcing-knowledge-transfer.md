# Knowledge Transfer: Graph of Thought Event-Sourced Architecture

**Date:** 2025-12-20
**Session:** claude/code-review-history-ckKBt
**Tags:** `got`, `event-sourcing`, `persistence`, `cross-branch`, `architecture`

---

## Executive Summary

This session implemented a fundamental architectural change to how the Graph of Thought (GoT) persists state. We transitioned from binary snapshots (which couldn't survive git merges or fresh clones) to an **event-sourced architecture** that enables true cross-branch, cross-environment coordination.

**Key Outcome:** GoT state now survives fresh `git clone` and enables parallel agent work without merge conflicts.

---

## Problem Statement

### Before This Session

```
.got/
├── snapshots/     # Binary blobs - git-tracked but NOT merge-friendly
└── wal/           # Local crash recovery
```

**Issues:**
1. Fresh clone = lost GoT state (snapshots were gitignored initially)
2. Two branches modifying GoT = merge conflict
3. Parallel agents couldn't coordinate (would overwrite each other)
4. No audit trail of changes

### The Insight

The original `tasks/*.json` system was merge-friendly because each session wrote to a unique file. We needed to apply the same pattern to GoT.

---

## Solution: Event-Sourced Persistence

### New Architecture

```
.got/
├── events/                         # GIT-TRACKED - Source of truth
│   ├── 20251220-193726-a1b2.jsonl  # Session 1 events
│   ├── 20251220-201119-c3d4.jsonl  # Session 2 events
│   └── migration-20251220.jsonl    # Migrated from snapshot
├── snapshots/                      # GIT-TRACKED - Cache only
│   └── snap_*.json.gz
├── wal/                            # LOCAL ONLY - Crash recovery
└── tracked/                        # GIT-TRACKED - Exports
```

### Event Format

Each action logs an event to a session-unique file:

```json
{"ts": "2025-12-20T19:37:26Z", "event": "node.create", "id": "task:T-...", "type": "TASK", "data": {...}, "meta": {"branch": "main", "session": "abc"}}
{"ts": "2025-12-20T19:38:00Z", "event": "node.update", "id": "task:T-...", "changes": {"status": "completed"}}
```

### Recovery Order

```python
def _load_state(self):
    # 1. Events (git-tracked, source of truth)
    # 2. WAL snapshots (local cache)
    # 3. Git-tracked snapshots (backup)
    # 4. WAL recovery (crash recovery)
    # 5. Empty graph (fresh start)
```

### Merge Strategy

```
Branch A: events/session-a.jsonl  (10 events)
Branch B: events/session-b.jsonl  (15 events)
                    ↓
            Git merge (no conflict!)
                    ↓
Merged: Both files present
                    ↓
Load: Read all events, sort by timestamp, replay
                    ↓
Result: Unified graph with 25 events
```

---

## Implementation Details

### Files Changed

| File | Changes |
|------|---------|
| `scripts/got_utils.py` | Added `EventLog` class, updated all CRUD operations to log events |
| `.gitignore` | Track `.got/events/`, `.got/snapshots/`, ignore `.got/wal/` |
| `docs/got-event-sourcing.md` | Architecture documentation |
| `.claude/commands/got-resume.md` | Session continuity command |

### Key Classes

```python
class EventLog:
    """Append-only event log for merge-friendly persistence."""

    def log_node_create(self, node_id, node_type, data): ...
    def log_node_update(self, node_id, changes): ...
    def log_edge_create(self, src, tgt, edge_type): ...

    @classmethod
    def load_all_events(cls, events_dir) -> List[Dict]: ...

    @classmethod
    def rebuild_graph_from_events(cls, events) -> ThoughtGraph: ...
```

### Commands Added

```bash
# Convert existing snapshot to events (one-time migration)
python scripts/got_utils.py migrate-events

# Resume work in new session
/got-resume

# Sync to git-tracked location
python scripts/got_utils.py sync
```

---

## Migration Path

### From Snapshot to Events

```bash
# 1. Existing state is in WAL snapshots
# 2. Run migration
python scripts/got_utils.py migrate-events

# 3. Verify
python scripts/got_utils.py stats
# Shows: 214 tasks, 1 sprint

# 4. Commit events
git add .got/events/ && git commit -m "got: Migrate to event-sourcing"
```

### Current State

- **215 events** in `.got/events/migration-20251220-202641.jsonl`
- **214 tasks**, 1 sprint preserved
- Events are the source of truth

---

## Session Continuity

### The `/got-resume` Command

Created `.claude/commands/got-resume.md` for resuming work:

```bash
# In new session/environment:
/got-resume

# Or manually:
python scripts/got_utils.py stats
python scripts/got_utils.py task list --status in_progress
```

### What It Does

1. Loads current GoT state from events
2. Shows in-progress tasks
3. Displays recent events for context
4. Provides common action commands

---

## Remaining Work

| Task | Priority | Notes |
|------|----------|-------|
| Agent handoff primitives | Medium | Events like `handoff.initiate`, `handoff.accept` |
| Event compaction | Low | Consolidate old events when log grows large |
| Cross-branch visualization | Low | Show graph state across branches |

---

## Lessons Learned

### 1. Understand Before Optimizing

Initially, I added `.got/` to `.gitignore` thinking "local state shouldn't pollute the repo." This broke environment resilience. The user caught this and asked "does this survive fresh clone?"

**Lesson:** Understand what resilience means before optimizing for aesthetics.

### 2. Event Sourcing > Snapshots for Coordination

Binary snapshots are convenient but:
- Can't merge
- No audit trail
- Overwrite on conflict

Event logs are slightly more complex but:
- Merge-friendly (unique files per session)
- Full audit trail
- Conflict-free (timestamp ordering)

### 3. The Original System Was Right

The `tasks/*.json` system we migrated FROM was actually well-designed for distributed work. Each session wrote to a unique file. We should have preserved this pattern in GoT from the start.

---

## Quick Reference

### Check GoT State
```bash
python scripts/got_utils.py stats
python scripts/got_utils.py task list --status pending
```

### Create/Complete Tasks
```bash
python scripts/got_utils.py task create "Title" --priority high
python scripts/got_utils.py task complete <task_id>
```

### Sync Before Commit
```bash
python scripts/got_utils.py sync
```

### View Events
```bash
cat .got/events/*.jsonl | tail -20
```

---

## Related Documents

- `docs/got-event-sourcing.md` - Detailed architecture
- `docs/got-project-management-schema.md` - Node/edge types
- `docs/got-cli-spec.md` - Full CLI reference
- `.claude/commands/got-resume.md` - Session continuity

---

*This knowledge transfer enables future sessions to understand and extend the event-sourced GoT architecture.*
