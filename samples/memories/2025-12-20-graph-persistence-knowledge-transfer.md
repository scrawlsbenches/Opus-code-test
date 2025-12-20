# Knowledge Transfer: Graph Persistence Layer

**Date:** 2025-12-20
**Tags:** `graph-persistence`, `wal`, `recovery`, `git-integration`, `architecture`
**Related:** [[graph-of-thought.md]], [[graph-recovery-procedures.md]]

## Summary

Implemented a robust, git-integrated, multi-level recovery system for ThoughtGraph data. The architecture follows a "defense in depth" philosophy where each layer provides an additional safety net.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ThoughtGraph (In-Memory)                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────────┐
        │                         ▼                             │
        │  ┌─────────────────────────────────────────────────┐  │
        │  │              GraphWAL (Write-Ahead Log)          │  │
        │  │  - Append-only operations log                    │  │
        │  │  - Per-entry checksums for integrity             │  │
        │  │  - Supports all graph operations                 │  │
        │  └─────────────────────────┬───────────────────────┘  │
        │                            │                          │
        │                            ▼                          │
        │  ┌─────────────────────────────────────────────────┐  │
        │  │           GraphSnapshot (Point-in-Time)          │  │
        │  │  - Full graph state capture                      │  │
        │  │  - GZIP compression                              │  │
        │  │  - WAL reference for replay                      │  │
        │  └─────────────────────────┬───────────────────────┘  │
        │                            │                          │
        │                            ▼                          │
        │  ┌─────────────────────────────────────────────────┐  │
        │  │         GitAutoCommitter (Version Control)        │  │
        │  │  - Immediate/Debounced/Manual modes              │  │
        │  │  - Protected branch safety                       │  │
        │  │  - Pre-commit validation                         │  │
        │  └─────────────────────────────────────────────────┘  │
        │                                                       │
        │                     GraphRecovery                     │
        │                 (4-Level Cascade Recovery)            │
        └───────────────────────────────────────────────────────┘
```

## Key Components

### 1. GraphWAL (Write-Ahead Log)

**Purpose:** Durably log all graph operations before they're considered committed.

```python
from cortical.reasoning.graph_persistence import GraphWAL

wal = GraphWAL(wal_dir="/path/to/wal")

# Log operations
wal.log_add_node("node1", "concept", "Neural networks")
wal.log_add_edge("node1", "node2", "relates_to", weight=0.8)

# Create snapshot
snapshot_id = wal.create_snapshot(graph, compress=True)

# Replay for recovery
graph = wal.load_snapshot(snapshot_id)
```

**Key decisions:**
- Each entry has SHA256 checksum for tamper detection
- Append-only design prevents accidental overwrites
- WAL files are rotated automatically

### 2. GitAutoCommitter

**Purpose:** Automatically version graph state in git with safety guards.

```python
from cortical.reasoning.graph_persistence import GitAutoCommitter

committer = GitAutoCommitter(
    repo_path="/path/to/repo",
    mode='immediate',           # or 'debounced', 'manual'
    auto_push=False,            # Safety: don't auto-push
    protected_branches=['main', 'master']  # Never force push
)

# Auto-commit with validation
committer.auto_commit(
    message="Graph update: added reasoning nodes",
    files=["wal/", "snapshots/"],
    validate_graph=graph        # Pre-commit integrity check
)
```

**Key decisions:**
- Three commit modes: immediate, debounced (5s default), manual
- Protected branch detection prevents accidental pushes
- Pre-commit validation catches corrupt graphs before commit

### 3. GraphRecovery (4-Level Cascade)

**Purpose:** Recover graph state from any failure scenario.

| Level | Name | Method | When Used |
|-------|------|--------|-----------|
| 1 | WAL Replay | Replay operations from latest WAL | Default recovery |
| 2 | Snapshot Rollback | Roll back to previous snapshot | WAL corrupted |
| 3 | Git Recovery | Extract from git history | Snapshots corrupted |
| 4 | Chunk Reconstruction | Rebuild from chunk files | All else fails |

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(
    wal_dir="/path/to/wal",
    snapshots_dir="/path/to/snapshots",
    chunks_dir="/path/to/chunks"
)

# Check if recovery needed
if recovery.needs_recovery():
    result = recovery.recover()
    if result.success:
        graph = result.graph
        print(f"Recovered {result.nodes_recovered} nodes via {result.recovery_method}")
```

**Key decisions:**
- Cascade down levels only on failure
- Each level verifies graph integrity before accepting
- Recovery preserves all node properties and edge weights

## Integration Patterns

### Pattern 1: Standard Save/Load

```python
# Save with git versioning
wal = GraphWAL("/path/to/wal")
committer = GitAutoCommitter("/path/to/repo")

wal.log_add_node(...)
snapshot_id = wal.create_snapshot(graph)
committer.commit_on_save(snapshot_id, graph)
```

### Pattern 2: Crash Recovery

```python
# On startup, check for recovery
recovery = GraphRecovery(wal_dir, snapshots_dir)
if recovery.needs_recovery():
    result = recovery.recover()
    if result.success:
        graph = result.graph
    else:
        # All recovery levels failed
        log_errors(result.errors)
```

### Pattern 3: Safe Push with Validation

```python
committer = GitAutoCommitter(repo_path, mode='manual')

# Validate before commit
issues = committer.validate_before_commit(graph)
if not issues:
    committer.auto_commit("Graph update", files)
    committer.push_if_safe()  # Won't push to main/master
```

## Test Coverage

| Test Category | Count | Passing | Coverage |
|--------------|-------|---------|----------|
| Unit Tests | 147 | 147 | 83% |
| Behavioral Tests | 31 | 31 | - |
| Integration Tests | 29 | 23 | - |
| **Total** | **207** | **201** | **83%** |

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `cortical/reasoning/graph_persistence.py` | 2,012 | Core implementation |
| `tests/unit/test_graph_persistence.py` | 1,771 | Unit tests |
| `tests/behavioral/test_graph_persistence_behavioral.py` | 979 | Behavioral tests |
| `tests/integration/test_graph_persistence_integration.py` | 929 | Integration tests |
| `docs/graph-recovery-procedures.md` | 1,488 | Recovery documentation |

## Key Design Decisions

1. **Defense in depth** - Four recovery levels ensure data is never truly lost
2. **Git as source of truth** - Level 3 recovery uses git history, making graphs recoverable even after disk failures
3. **Protected branches** - Safety guards prevent accidental force pushes
4. **Checksums everywhere** - WAL entries and snapshots have integrity verification
5. **Compression by default** - Snapshots use GZIP for efficient storage

## Connections

- Builds on existing `cortical/wal.py` (WALWriter, WALEntry, SnapshotManager)
- Integrates with `cortical/chunk_index.py` for Level 4 reconstruction
- Extends the reasoning framework in `cortical/reasoning/`
- Uses ThoughtGraph from `cortical/reasoning/thought_graph.py`
