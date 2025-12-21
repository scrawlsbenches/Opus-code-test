# Graph of Thought: Event-Sourced Persistence

## The Problem

Binary snapshots (`.got/snapshots/*.json.gz`) are NOT merge-friendly:

```
Branch A: snapshot with 215 tasks
Branch B: snapshot with 220 tasks (some overlap)
Merge: CONFLICT - which snapshot wins?
```

This breaks cross-branch coordination. When I (Claude) work on branch A and another instance works on branch B, our work cannot be merged without data loss.

## The Solution: Event Sourcing

Instead of snapshots as source of truth, use **append-only event logs**:

```
.got/
├── events/                         # GIT-TRACKED (source of truth)
│   ├── 20251220-193726-a1b2.jsonl  # Session 1 events
│   ├── 20251220-201119-c3d4.jsonl  # Session 2 events
│   └── ...                         # Each session = unique file
├── snapshots/                      # GIT-TRACKED (cache only)
│   └── latest.json.gz              # Rebuilt from events on startup
├── wal/                            # LOCAL ONLY (crash recovery)
│   └── ...
└── tracked/                        # GIT-TRACKED (metadata)
    └── ...
```

## Event Format

Each event is a single JSON line:

```json
{"ts": "2025-12-20T19:37:26.123Z", "event": "node.create", "id": "task:T-20251220-193726-a1b2", "type": "TASK", "data": {"title": "Implement feature", "status": "pending", "priority": "high"}, "meta": {"branch": "main", "session": "abc123"}}
```

### Event Types

| Event | Description | Fields |
|-------|-------------|--------|
| `node.create` | Create a new node | `id`, `type`, `data`, `meta` |
| `node.update` | Update node properties | `id`, `changes` |
| `node.delete` | Delete a node | `id` |
| `edge.create` | Create an edge | `src`, `tgt`, `type`, `weight` |
| `edge.delete` | Delete an edge | `src`, `tgt`, `type` |
| `graph.merge` | Merge another branch | `branch`, `commit` |

### Metadata Fields

Every event includes:
- `ts`: ISO 8601 timestamp with milliseconds
- `branch`: Git branch name
- `session`: Session ID (unique per Claude instance)
- `agent`: Agent identifier (for parallel agents)

## Merge Strategy

### Step 1: Collect All Events
```bash
# On merge, all event files from both branches are present
.got/events/
├── 20251220-193726-a1b2.jsonl  # From branch A
├── 20251220-201119-c3d4.jsonl  # From branch A
├── 20251220-200000-e5f6.jsonl  # From branch B
└── 20251220-203000-g7h8.jsonl  # From branch B
```

### Step 2: Rebuild Graph
1. Read all event files
2. Sort events by timestamp (globally)
3. Apply events in order
4. Last-write-wins for property conflicts
5. Save new snapshot as cache

### Step 3: Conflict Resolution

**No conflicts possible** because:
- Node IDs include timestamps (unique)
- Edge IDs are derived from source+target+type (deterministic)
- Property updates are timestamped (later wins)

**What about semantic conflicts?**
- Two agents complete the same task → both completions recorded, graph shows completed
- Two agents make different decisions → both decisions recorded as nodes, linked by CONFLICTS_WITH edge
- Two agents work on same file → both recorded, human reviews

## Recovery Order

When loading the graph:

```
1. Try snapshot cache (.got/snapshots/latest.json.gz)
   - Check if cache is valid (hash matches events)
   - If valid, load directly (fast)

2. If cache invalid, rebuild from events
   - Read all .got/events/*.jsonl
   - Sort by timestamp
   - Apply in order
   - Save new snapshot cache

3. If no events, try WAL recovery (local crash recovery)

4. If all else fails, start fresh
```

## Implementation Changes

### GoTProjectManager Changes

```python
class GoTProjectManager:
    def __init__(self, got_dir: Path = GOT_DIR):
        self.events_dir = got_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Session-specific event file
        self.session_id = generate_session_id()
        self.event_file = self.events_dir / f"{timestamp()}-{self.session_id}.jsonl"

        # Load graph from events (or cache)
        self._load_from_events()

    def _log_event(self, event_type: str, **data):
        """Append event to session file."""
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            "meta": {
                "branch": get_current_branch(),
                "session": self.session_id,
            },
            **data
        }
        with open(self.event_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def create_task(self, title: str, **kwargs) -> str:
        task_id = generate_task_id()
        # ... create node in graph ...

        # Log the event
        self._log_event("node.create",
            id=task_id,
            type="TASK",
            data={"title": title, **kwargs}
        )
        return task_id
```

### Sync Command Changes

```python
def sync_to_git(self) -> str:
    """Sync current state to git-tracked storage.

    Events are already in git-tracked directory.
    This just updates the snapshot cache.
    """
    # Rebuild snapshot from all events
    self._rebuild_snapshot_cache()
    return self.event_file.name
```

## Benefits

1. **True merge-friendliness**: No binary conflicts
2. **Full audit trail**: Every action is recorded
3. **Branch awareness**: Know who did what where
4. **Parallel agent safe**: Each agent writes to unique file
5. **Time travel**: Can replay to any point in time
6. **Debugging**: Can see exactly what happened

## Migration from Snapshots

```python
def migrate_to_events():
    """One-time migration from snapshot to events."""
    # Load current snapshot
    snapshot = load_snapshot(".got/snapshots/latest.json.gz")

    # Generate synthetic creation events for all nodes
    for node_id, node in snapshot.nodes.items():
        log_event("node.create",
            id=node_id,
            type=node.type,
            data=node.properties,
            meta={"branch": "migration", "session": "migration"}
        )

    # Generate synthetic creation events for all edges
    for edge in snapshot.edges.values():
        log_event("edge.create",
            src=edge.source_id,
            tgt=edge.target_id,
            type=edge.type,
            weight=edge.weight
        )
```

## Cross-Agent Coordination

With event sourcing, parallel agents can coordinate:

```
Agent A (Task tool spawn):
  └── Writes to: .got/events/20251220-201500-agent-a.jsonl

Agent B (Task tool spawn):
  └── Writes to: .got/events/20251220-201500-agent-b.jsonl

Main Agent:
  └── Reads both files after agents complete
  └── Sees unified view of all work
```

### Handoff Protocol

```json
{"ts": "...", "event": "handoff.initiate", "from": "agent-a", "to": "agent-b", "task": "task:T-...", "context": "..."}
{"ts": "...", "event": "handoff.accept", "from": "agent-b", "task": "task:T-..."}
{"ts": "...", "event": "handoff.complete", "from": "agent-b", "task": "task:T-...", "result": "..."}
```

## Decision Record

This design was chosen because:

1. **The original tasks/*.json system was merge-friendly** - we should preserve that property
2. **Snapshots are convenient but not sufficient** - they work locally but break on merge
3. **Event sourcing is a proven pattern** - used in databases, CQRS, blockchain
4. **Git is the transport layer** - we must work WITH git, not against it
5. **Future agents need history** - not just current state, but how we got here

---

*This is how I (Claude) want to work with myself across branches and time.*
