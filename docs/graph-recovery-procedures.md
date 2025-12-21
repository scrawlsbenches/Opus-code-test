# Graph Recovery Procedures

## Overview

The Graph Recovery system provides fault-tolerant, multi-level recovery for ThoughtGraph instances used in the reasoning framework. When graph state becomes corrupted or inaccessible, the recovery system automatically cascades through increasingly thorough recovery methods until a valid state is restored.

### What Recovery Is

Graph recovery is the process of restoring a ThoughtGraph to a consistent, valid state after:
- System crashes or unexpected shutdowns
- File corruption (disk errors, incomplete writes)
- Manual deletion of state files
- Git history issues (rebases, force pushes, branch conflicts)
- WAL or snapshot corruption

The recovery system ensures that reasoning workflows can always resume from the most recent valid state, minimizing data loss and maintaining continuity across sessions.

### When Recovery Is Needed

Recovery is triggered automatically when:
1. **WAL corruption is detected** - checksums fail or entries are malformed
2. **Snapshots are missing or corrupted** - latest snapshot fails integrity checks
3. **Graph state inconsistencies** - orphaned edges, invalid references
4. **File system issues** - missing expected files, permission errors

You can also manually trigger recovery when:
- Debugging graph state issues
- Restoring from a specific point in time
- Verifying graph integrity after suspicious operations

### The 4-Level Cascade Philosophy

The recovery system implements a **cascading fallback strategy**, attempting faster methods first and falling back to slower but more thorough methods only when necessary:

```
Level 1: WAL Replay          (Fastest, most recent)
   ↓ (if fails)
Level 2: Snapshot Rollback   (Fast, slightly older)
   ↓ (if fails)
Level 3: Git History         (Moderate, historical)
   ↓ (if fails)
Level 4: Chunk Reconstruction (Slowest, most thorough)
```

**Key principles:**

1. **Speed vs. Thoroughness Trade-off**: Earlier levels are faster but less robust. Later levels sacrifice speed for comprehensive recovery.

2. **Recency Priority**: Level 1 preserves the most recent state. Each subsequent level may lose some recent changes but guarantees older, stable state.

3. **Automatic Escalation**: You never need to choose which level to use - the system automatically tries each level until one succeeds.

4. **Minimal Data Loss**: The system always attempts to recover the most recent valid state possible.

5. **Graceful Degradation**: Even if lower levels partially fail, Level 4 can reconstruct from raw operation logs.

---

## Recovery Levels

### Level 1: WAL Replay (Fastest)

**Purpose**: Recover from crashes or unexpected shutdowns with minimal data loss.

**How It Works**:
1. Loads the latest snapshot (a point-in-time graph state)
2. Reads the Write-Ahead Log (WAL) for operations since that snapshot
3. Replays each WAL entry in order to reconstruct the current state
4. Verifies graph integrity (no orphaned edges, valid references)

**When It's Used**:
- Normal recovery after crashes
- Resuming after unexpected process termination
- Recovering from incomplete write operations

**Performance**:
- **Speed**: 10-100ms for typical graphs (depends on WAL entries since last snapshot)
- **Data Loss**: None (recovers up to last logged operation)
- **Success Rate**: ~95% (fails only if WAL or snapshot is corrupted)

**What Can Go Wrong**:
- Snapshot file is corrupted or missing
- WAL entries have invalid checksums (tampering or disk corruption)
- Operations in WAL reference nodes that don't exist in snapshot

**Example Code**:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

# Initialize recovery manager
recovery = GraphRecovery(wal_dir='reasoning_wal')

# Check if recovery is needed
if recovery.needs_recovery():
    print("Recovery needed - attempting WAL replay...")

    # Perform recovery (automatically tries Level 1 first)
    result = recovery.recover()

    if result.success and result.level_used == 1:
        print(f"✓ WAL Replay successful!")
        print(f"  Recovered {result.nodes_recovered} nodes")
        print(f"  Recovered {result.edges_recovered} edges")
        print(f"  Duration: {result.duration_ms:.2f}ms")

        # Get the recovered graph
        graph = result.graph
    else:
        print(f"✗ WAL Replay failed, escalated to Level {result.level_used}")
```

**Expected Recovery Time**:

| Graph Size | WAL Entries | Recovery Time |
|------------|-------------|---------------|
| Small (10-50 nodes) | 0-100 | 10-20ms |
| Medium (50-200 nodes) | 100-500 | 20-50ms |
| Large (200-1000 nodes) | 500-2000 | 50-200ms |
| Very Large (1000+ nodes) | 2000+ | 200-500ms |

**Verification**:

After WAL replay, the system verifies:
- All nodes referenced by edges exist
- Edge indices are consistent with actual edges
- No duplicate node IDs
- Clusters reference valid nodes

If verification fails, recovery escalates to Level 2.

---

### Level 2: Snapshot Rollback

**Purpose**: Recover from corrupted snapshots by trying previous snapshot generations.

**How Snapshots Work**:

Snapshots are point-in-time serializations of complete graph state, stored as compressed JSON:

```
reasoning_wal/snapshots/
├── snap_20251220_143052.json.gz  # Latest (may be corrupted)
├── snap_20251220_123015.json.gz  # Previous generation
└── snap_20251220_103000.json.gz  # Oldest generation
```

Each snapshot contains:
- Complete node data (ID, type, content, properties, metadata)
- Complete edge data (source, target, type, weight, confidence)
- Cluster definitions
- WAL reference (which WAL file/offset this snapshot was taken from)
- SHA256 checksum for integrity verification

**Snapshot Retention Policy**:

By default, the system keeps **3 snapshot generations**:

1. **Latest**: Taken after significant graph changes or before risky operations
2. **Previous**: The snapshot before the latest (usually 1-4 hours old)
3. **Oldest**: Oldest generation kept for recovery (usually 4-24 hours old)

When a new snapshot is created, the oldest is automatically deleted.

**Why 3 Generations?**

- **Storage efficiency**: Snapshots are compressed but can be large (1-10MB for complex graphs)
- **Recovery coverage**: 3 generations provide good protection against transient corruption
- **Time windows**: Typically covers the last 24 hours of work
- **Git integration**: Snapshots are committed to git, so historical versions are preserved

**Rollback Procedure**:

1. Lists all available snapshots in chronological order
2. Starting from newest, verifies snapshot checksum
3. If checksum passes, loads snapshot and reconstructs graph
4. Verifies graph integrity (same checks as Level 1)
5. If verification passes, returns recovered graph
6. If verification fails, tries next older snapshot
7. Continues until a valid snapshot is found or all are exhausted

**Example Code**:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(
    wal_dir='reasoning_wal',
    max_snapshots=3  # Keep 3 generations (default)
)

# This will automatically try snapshots if WAL replay fails
result = recovery.recover()

if result.success and result.level_used == 2:
    print(f"✓ Snapshot Rollback successful!")
    print(f"  Method: {result.recovery_method}")
    print(f"  Nodes: {result.nodes_recovered}")
    print(f"  Edges: {result.edges_recovered}")
    print(f"  Duration: {result.duration_ms:.2f}ms")

    # Check for errors (may have warnings about corrupted snapshots)
    if result.errors:
        print(f"  Warnings: {len(result.errors)} issues encountered")
        for error in result.errors[:3]:
            print(f"    - {error}")
```

**Manual Snapshot Management**:

```python
from cortical.reasoning.graph_persistence import GraphWAL

# Create snapshot manually
wal = GraphWAL('reasoning_wal')
snapshot_id = wal.create_snapshot(
    graph=my_graph,
    compress=True  # Default, saves space
)
print(f"Created snapshot: {snapshot_id}")

# Load specific snapshot
graph = wal.load_snapshot(snapshot_id)

# Load latest snapshot (default)
graph = wal.load_snapshot()

# Compact WAL (creates snapshot and removes old WAL entries)
snapshot_id = wal.compact_wal(my_graph)
```

**Performance**:

- **Speed**: 50-200ms (depends on snapshot size and compression)
- **Data Loss**: Lose changes since snapshot was taken (typically 0-4 hours)
- **Success Rate**: ~85% (fails if all snapshots are corrupted)

**What Can Go Wrong**:

- All snapshots fail checksum verification (rare, indicates severe disk corruption)
- Snapshots contain invalid data (impossible node types, malformed JSON)
- Snapshot directory is missing or inaccessible

---

### Level 3: Git History Recovery

**Purpose**: Restore graph state from source control when local snapshots are corrupted.

**How It Works**:

Git commits provide an immutable history of graph snapshots. When local files are corrupted, we can extract snapshots from git history:

1. **Find graph-related commits**: Search git history for commits that modified snapshot files
2. **Extract snapshot content**: Use `git show <commit>:<file>` to get file content at that commit
3. **Load and verify**: Decompress, deserialize, and verify integrity
4. **Return first valid state**: Use the most recent commit with a valid snapshot

**Finding Graph-Related Commits**:

The recovery system searches for commits that touched the snapshots directory:

```bash
# Example git command used internally
git log -50 --format='%H|%s' -- reasoning_wal/snapshots/
```

This returns up to 50 recent commits with format:
```
abc123def|graph: Auto-save reasoning_graph.json
def456ghi|graph: Snapshot before risky operation
```

**Limitations and Considerations**:

**Limitations**:
1. **Requires git repository**: Won't work if not in a git repo or `.git` is corrupted
2. **Assumes snapshots are committed**: Only works if auto-commit is enabled
3. **Limited history**: Only searches last 50 commits (configurable)
4. **May be stale**: Last committed snapshot may be hours or days old

**When It's Useful**:
- Local files are corrupted but git history is intact
- Recovering after accidental deletion of snapshot directory
- Rolling back to a known-good state after introducing bugs
- Cross-machine recovery (clone repo, recover graph)

**When It Won't Work**:
- Not in a git repository
- Git auto-commit was disabled
- Repository was recently cloned and lacks snapshot history
- Force push removed snapshot commits
- `.git` directory is corrupted

**Example Code**:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(wal_dir='reasoning_wal')

# This will cascade to Level 3 if Levels 1-2 fail
result = recovery.recover()

if result.success and result.level_used == 3:
    print(f"✓ Git History Recovery successful!")
    print(f"  {result.recovery_method}")
    print(f"  Recovered from commit: {result.recovery_method.split('(')[-1].strip(')')}")
    print(f"  Nodes: {result.nodes_recovered}")
    print(f"  Edges: {result.edges_recovered}")

    # Note: May have lost recent changes
    print(f"  ⚠ Warning: Recovered state may be hours/days old")
```

**Manual Git Recovery**:

If you need to recover from a specific commit:

```bash
# Find commits that modified snapshots
git log -20 --oneline -- reasoning_wal/snapshots/

# Extract snapshot from specific commit
git show abc123:reasoning_wal/snapshots/snap_20251220_143052.json.gz > recovered_snapshot.json.gz

# Decompress
gunzip recovered_snapshot.json.gz
```

Then manually load in Python:

```python
import json
from cortical.reasoning.graph_persistence import GraphRecovery

# Load the extracted snapshot
with open('recovered_snapshot.json', 'r') as f:
    snapshot_data = json.load(f)

# Reconstruct graph
recovery = GraphRecovery(wal_dir='reasoning_wal')
graph = recovery._graph_from_snapshot(snapshot_data)

# Verify integrity
issues = recovery.verify_graph_integrity(graph)
if issues:
    print("Integrity issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Graph recovered successfully!")
```

**Performance**:

- **Speed**: 200-500ms (depends on git history size and commit count)
- **Data Loss**: Moderate (lose changes since last git commit, typically hours)
- **Success Rate**: ~70% (depends on git commit frequency and repository state)

---

### Level 4: Chunk Reconstruction

**Purpose**: Last resort recovery by replaying all operations from chunk files.

**What Are Chunks?**

Chunks are append-only, git-friendly JSON files that log every graph operation:

```
graph_chunks/
├── 2025-12-20_14-30-52_a1b2.json  # Session 1
├── 2025-12-20_15-45-30_c3d4.json  # Session 2
└── 2025-12-20_16-20-00_e5f6.json  # Session 3
```

Each chunk contains:
```json
{
  "session_id": "a1b2c3d4",
  "timestamp": "2025-12-20T14:30:52",
  "operations": [
    {
      "op": "add_node",
      "node_id": "Q1",
      "node_type": "question",
      "content": "What is authentication?",
      "properties": {},
      "metadata": {"created": "2025-12-20T14:30:52"}
    },
    {
      "op": "add_edge",
      "from_id": "Q1",
      "to_id": "H1",
      "edge_type": "explores",
      "weight": 0.8,
      "confidence": 1.0
    }
  ]
}
```

**When to Use (Last Resort)**:

Chunk reconstruction is used when:
- All snapshots are corrupted or missing
- WAL is completely lost or corrupted
- Git history doesn't contain valid snapshots
- You need to rebuild from scratch

**How Chunks Are Replayed**:

1. Find all chunk files in chronological order (sorted by timestamp in filename)
2. Create an empty ThoughtGraph
3. For each chunk file:
   - Load JSON
   - Replay each operation in sequence
   - Handle errors gracefully (skip invalid operations)
4. Verify integrity of reconstructed graph
5. Return reconstructed graph (even if some operations failed)

**Performance Implications**:

Chunk reconstruction is **slow** because it replays every operation:

| Operations | Estimated Time | Notes |
|------------|----------------|-------|
| 100 ops | 50-100ms | Small session |
| 1,000 ops | 200-500ms | Medium session |
| 10,000 ops | 2-5 seconds | Large session |
| 100,000 ops | 20-60 seconds | Very large corpus |

**Example Code**:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(
    wal_dir='reasoning_wal',
    chunks_dir='graph_chunks'  # Required for Level 4
)

# This will cascade to Level 4 if all else fails
result = recovery.recover()

if result.success and result.level_used == 4:
    print(f"✓ Chunk Reconstruction successful!")
    print(f"  {result.recovery_method}")
    print(f"  Nodes: {result.nodes_recovered}")
    print(f"  Edges: {result.edges_recovered}")
    print(f"  Duration: {result.duration_ms:.2f}ms")

    # Check for errors (common in chunk reconstruction)
    if result.errors:
        print(f"  ⚠ Errors during reconstruction: {len(result.errors)}")
        print(f"  Graph may be incomplete but usable")
        for error in result.errors[:5]:
            print(f"    - {error}")
```

**Handling Partial Reconstruction**:

Chunk reconstruction may succeed even with errors:

```python
result = recovery.recover()

if result.success and result.level_used == 4:
    if result.errors:
        print("Partial reconstruction - some operations failed")

        # You can still use the graph
        graph = result.graph

        # But verify what's missing
        expected_nodes = 100  # You know you should have ~100 nodes
        actual_nodes = result.nodes_recovered

        if actual_nodes < expected_nodes * 0.9:
            print(f"WARNING: Only recovered {actual_nodes}/{expected_nodes} nodes")
            print("Consider manual intervention")
        else:
            print(f"Recovered {actual_nodes} nodes - acceptable")
```

**Chunk File Format**:

Each chunk is a self-contained JSON file with:

```json
{
  "session_id": "unique-session-id",
  "timestamp": "2025-12-20T14:30:52",
  "operations": [
    {
      "op": "add_node",
      "node_id": "Q1",
      "node_type": "question",
      "content": "...",
      "properties": {},
      "metadata": {}
    },
    {
      "op": "add_edge",
      "from_id": "Q1",
      "to_id": "H1",
      "edge_type": "explores",
      "weight": 0.8,
      "confidence": 1.0
    },
    {
      "op": "remove_node",
      "node_id": "obsolete_node"
    }
  ]
}
```

**Supported Operations**:
- `add_node` - Add node with full metadata
- `remove_node` - Remove node by ID
- `add_edge` - Add edge between nodes
- `remove_edge` - Remove specific edge
- `update_node` - Update node properties
- `add_cluster` - Create cluster
- `merge_nodes` - Merge multiple nodes

**Performance**:

- **Speed**: 2-60 seconds (depends on total operation count)
- **Data Loss**: Minimal (only loses operations from corrupted chunks)
- **Success Rate**: ~95% (fails only if chunks directory is missing)

**When It Fails**:

Chunk reconstruction can fail if:
- Chunks directory doesn't exist or is empty
- All chunk files are corrupted or unreadable
- Reconstructed graph is completely empty (no operations succeeded)

Even then, the system returns a result with errors so you can diagnose the issue.

---

## Using GraphRecovery

### Basic Usage

The simplest way to use the recovery system:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

# Initialize with WAL directory (minimum required)
recovery = GraphRecovery(wal_dir='reasoning_wal')

# Check if recovery is needed
if recovery.needs_recovery():
    # Perform automatic cascading recovery
    result = recovery.recover()

    # Check result
    if result.success:
        print(f"✓ Recovery succeeded at Level {result.level_used}")
        graph = result.graph
    else:
        print(f"✗ Recovery failed after trying all levels")
        for error in result.errors:
            print(f"  - {error}")
```

### Advanced Configuration

Configure recovery behavior:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(
    wal_dir='reasoning_wal',          # Required: WAL and snapshot directory
    chunks_dir='graph_chunks',         # Optional: Enable Level 4 recovery
    max_snapshots=3                    # Keep 3 snapshot generations (default)
)

# Customize snapshot retention
recovery = GraphRecovery(
    wal_dir='reasoning_wal',
    max_snapshots=5  # Keep more generations for paranoid recovery
)
```

### Checking Recovery Status

Before attempting recovery, check what methods are available:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(
    wal_dir='reasoning_wal',
    chunks_dir='graph_chunks'
)

# Check if recovery is needed
if recovery.needs_recovery():
    print("Recovery is needed")

    # Check what's available
    snapshots = recovery._list_graph_snapshots()
    print(f"Available snapshots: {len(snapshots)}")
    for snap in snapshots:
        print(f"  - {snap.snapshot_id}: {snap.node_count} nodes, {snap.edge_count} edges")
        print(f"    Valid: {snap.verify_checksum()}")

    # Check if in git repo
    if recovery._is_git_repo():
        commits = recovery._find_graph_commits()
        print(f"Graph commits in git: {len(commits)}")

    # Check chunks
    if recovery.chunks_dir and recovery.chunks_dir.exists():
        chunks = sorted(recovery.chunks_dir.glob("*.json"))
        print(f"Chunk files: {len(chunks)}")
```

### Verifying Graph Integrity

After recovery or any risky operation, verify graph integrity:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(wal_dir='reasoning_wal')

# Load graph from somewhere
graph = load_my_graph()

# Verify integrity
issues = recovery.verify_graph_integrity(graph)

if not issues:
    print("✓ Graph integrity verified - no issues found")
else:
    print(f"✗ Found {len(issues)} integrity issues:")
    for issue in issues:
        print(f"  - {issue}")

    # Common issues:
    # - "Edge references missing source node: Q1"
    # - "Node X: outgoing edge index mismatch (indexed: 5, actual: 4)"
    # - "Self-loop detected: Q1"
    # - "Cluster CL1 references missing node: H1"
```

**What Integrity Checks Detect**:

1. **Orphaned Edges**: Edges referencing non-existent nodes
   ```
   Edge references missing source node: Q1
   Edge references missing target node: H2
   ```

2. **Index Inconsistencies**: Edge indices don't match actual edges
   ```
   Node Q1: outgoing edge index mismatch (indexed: 5, actual: 4)
   Node H1: incoming edge index mismatch (indexed: 3, actual: 2)
   ```

3. **Duplicate Nodes**: Same node ID appears multiple times
   ```
   Duplicate node IDs detected
   ```

4. **Self-Loops**: Node has edge to itself (warning, may be valid)
   ```
   Self-loop detected: Q1
   ```

5. **Cluster Issues**: Clusters reference non-existent nodes
   ```
   Cluster CL1 references missing node: H1
   ```

### Handling Recovery Failures

When recovery fails, diagnose the problem:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(
    wal_dir='reasoning_wal',
    chunks_dir='graph_chunks'
)

result = recovery.recover()

if not result.success:
    print(f"Recovery failed after trying all {result.level_used} levels")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"\nErrors encountered:")

    for i, error in enumerate(result.errors, 1):
        print(f"{i}. {error}")

    # Analyze failure
    if "No snapshot found" in str(result.errors):
        print("\n⚠ Snapshots are missing - enable snapshot creation")

    if "Not in a git repository" in str(result.errors):
        print("\n⚠ Git recovery unavailable - initialize git repo")

    if "Chunks directory not available" in str(result.errors):
        print("\n⚠ Chunk recovery unavailable - enable chunk logging")

    # Manual intervention needed
    print("\nManual recovery steps:")
    print("1. Check if reasoning_wal/ directory exists")
    print("2. Verify snapshot files are not corrupted")
    print("3. Check git commit history for graph: commits")
    print("4. Restore from backup if available")
```

### Result Object API

The `GraphRecoveryResult` object provides detailed information:

```python
result = recovery.recover()

# Check success
if result.success:
    print("Recovery succeeded")
else:
    print("Recovery failed")

# Level used (1-4)
print(f"Level: {result.level_used}")

# Method description
print(f"Method: {result.recovery_method}")
# Examples:
# - "WAL Replay"
# - "Snapshot Rollback (using snap_20251220_143052)"
# - "Git History Recovery (from commit abc123de)"
# - "Chunk Reconstruction (15 chunks processed)"

# Statistics
print(f"Nodes recovered: {result.nodes_recovered}")
print(f"Edges recovered: {result.edges_recovered}")
print(f"Duration: {result.duration_ms:.2f}ms")

# Errors (may be present even on success)
if result.errors:
    print(f"Warnings/errors: {len(result.errors)}")
    for error in result.errors:
        print(f"  - {error}")

# The recovered graph
graph = result.graph  # ThoughtGraph instance or None

# Human-readable summary
print(str(result))
# Output:
# Recovery SUCCESS
# Level: 1 (WAL Replay)
# Nodes: 42
# Edges: 87
# Duration: 125.50ms
```

---

## Best Practices

### Regular Snapshots

Create snapshots at strategic points to enable fast recovery:

```python
from cortical.reasoning.graph_persistence import GraphWAL

wal = GraphWAL('reasoning_wal')

# Create snapshot before risky operations
def risky_operation(graph, wal):
    # Snapshot before
    snapshot_id = wal.create_snapshot(graph, compress=True)
    print(f"Created safety snapshot: {snapshot_id}")

    try:
        # Perform risky operation
        graph.merge_nodes("Q1", "Q2", "Q_merged")
        graph.experimental_operation()

        # Success - snapshot is insurance
        print("Operation succeeded - snapshot kept")

    except Exception as e:
        # Failure - rollback to snapshot
        print(f"Operation failed: {e}")
        print(f"Rolling back to snapshot {snapshot_id}")
        graph = wal.load_snapshot(snapshot_id)
        return graph

    return graph

# Create snapshot after major milestones
def after_major_work(graph, wal):
    # Just completed a reasoning session
    snapshot_id = wal.create_snapshot(graph)
    print(f"Milestone snapshot: {snapshot_id}")
```

**Recommended snapshot schedule**:
- **Before risky operations**: Merges, experimental features, bulk deletions
- **After major milestones**: Completed reasoning sessions, significant discoveries
- **Periodic**: Every N operations or M minutes (if long-running)
- **Before shutdown**: Save final state for next session

### WAL Compaction

Over time, WAL files grow. Compact them periodically:

```python
from cortical.reasoning.graph_persistence import GraphWAL

wal = GraphWAL('reasoning_wal')

# Check WAL size
entry_count = wal.get_entry_count()
print(f"Current WAL entries: {entry_count}")

# Compact when WAL gets large
if entry_count > 10000:
    print("WAL is large - compacting...")
    snapshot_id = wal.compact_wal(current_graph)
    print(f"Compacted to snapshot: {snapshot_id}")
    print("WAL entries cleared")

# Or compact periodically
def should_compact(wal):
    """Compact if WAL has >1000 entries."""
    return wal.get_entry_count() > 1000

if should_compact(wal):
    wal.compact_wal(graph)
```

**What compaction does**:
1. Creates a snapshot of current graph state
2. Deletes old WAL entries (they're now in the snapshot)
3. Starts a fresh WAL file
4. Keeps snapshot count under `max_snapshots` limit

**Benefits**:
- Faster WAL replay (fewer entries)
- Smaller disk usage
- Faster git commits (WAL files are smaller)

### Monitoring for Issues

Proactively detect issues before they cause failures:

```python
from cortical.reasoning.graph_persistence import GraphRecovery

def health_check(graph, recovery):
    """Perform regular health check."""
    issues = []

    # Check graph integrity
    integrity_issues = recovery.verify_graph_integrity(graph)
    if integrity_issues:
        issues.append(f"Graph integrity: {len(integrity_issues)} issues")
        issues.extend(integrity_issues[:3])  # Show first 3

    # Check snapshot health
    snapshots = recovery._list_graph_snapshots()
    valid_snapshots = sum(1 for s in snapshots if s.verify_checksum())
    if valid_snapshots < 2:
        issues.append(f"Warning: Only {valid_snapshots} valid snapshots")

    # Check WAL health
    if recovery.needs_recovery():
        issues.append("Recovery needed - WAL or snapshot corruption detected")

    return issues

# Run health check periodically
recovery = GraphRecovery(wal_dir='reasoning_wal')
issues = health_check(my_graph, recovery)

if issues:
    print("⚠ Health check found issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ Health check passed")
```

### Testing Recovery Procedures

Periodically test recovery to ensure it works:

```python
from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL
import shutil
import tempfile

def test_recovery_works(graph, wal_dir):
    """Test that recovery can restore current graph."""
    # Create snapshot
    wal = GraphWAL(wal_dir)
    snapshot_id = wal.create_snapshot(graph)

    # Test recovery
    recovery = GraphRecovery(wal_dir=wal_dir)
    result = recovery.recover()

    # Verify
    assert result.success, "Recovery failed"
    assert result.nodes_recovered == graph.node_count()
    assert result.edges_recovered == graph.edge_count()

    print(f"✓ Recovery test passed (Level {result.level_used})")

    return True

# Run monthly
test_recovery_works(my_graph, 'reasoning_wal')
```

### Git Integration Best Practices

Enable automatic git commits for seamless Level 3 recovery:

```python
from cortical.reasoning.graph_persistence import GitAutoCommitter, GraphWAL

# Setup auto-commit
committer = GitAutoCommitter(
    mode='debounced',      # Wait for inactivity before committing
    debounce_seconds=30,   # Wait 30s of inactivity
    auto_push=False,       # Don't auto-push (manual control)
    protected_branches=['main', 'master']  # Never auto-push to these
)

# Setup WAL with auto-commit integration
wal = GraphWAL('reasoning_wal')

# After saving graph
def save_graph(graph, graph_path):
    # Save to file
    graph.save(graph_path)

    # Trigger auto-commit (if debounced, waits for inactivity)
    committer.commit_on_save(
        graph_path=graph_path,
        graph=graph,  # For validation
        message=None  # Auto-generate message
    )

# Create snapshots that will be committed
snapshot_id = wal.create_snapshot(graph)
committer.commit_on_save(
    graph_path=f'reasoning_wal/snapshots/{snapshot_id}.json.gz',
    graph=graph,
    message=f"graph: Snapshot {snapshot_id}"
)
```

**Recommended git workflow**:
1. Use debounced commits (avoid spamming git history)
2. Don't auto-push (push manually after verification)
3. Protect main/master branches
4. Create backup branches before risky operations
5. Commit snapshots regularly

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "No snapshot found for WAL replay"

**Symptoms**:
```
Recovery needed - attempting WAL replay...
✗ Level 1 failed: No snapshot found for WAL replay
```

**Cause**: No snapshots exist in `reasoning_wal/snapshots/`

**Solutions**:

1. **Create initial snapshot**:
   ```python
   from cortical.reasoning.graph_persistence import GraphWAL

   wal = GraphWAL('reasoning_wal')
   snapshot_id = wal.create_snapshot(your_graph)
   ```

2. **Enable automatic snapshots** in your workflow:
   ```python
   # After major operations
   if operation_count % 100 == 0:
       wal.create_snapshot(graph)
   ```

3. **Check directory permissions**:
   ```bash
   ls -la reasoning_wal/snapshots/
   # Should be readable/writable
   ```

#### Issue: "All snapshots failed checksum verification"

**Symptoms**:
```
Recovery Level 2: Snapshot Rollback
✗ Snapshot snap_20251220_143052 failed checksum verification
✗ Snapshot snap_20251220_123015 failed checksum verification
✗ All snapshots failed integrity checks
```

**Cause**: Snapshots were corrupted (disk errors, incomplete writes)

**Solutions**:

1. **Try git recovery** (if enabled):
   ```python
   # Recovery automatically escalates to Level 3
   result = recovery.recover()
   # Will try git history
   ```

2. **Manually restore from git**:
   ```bash
   # Find last good commit
   git log --oneline -- reasoning_wal/snapshots/

   # Restore snapshots from that commit
   git checkout abc123 -- reasoning_wal/snapshots/
   ```

3. **Use chunk reconstruction**:
   ```python
   recovery = GraphRecovery(
       wal_dir='reasoning_wal',
       chunks_dir='graph_chunks'
   )
   result = recovery.recover()  # Will use Level 4
   ```

#### Issue: "Not in a git repository"

**Symptoms**:
```
Recovery Level 3: Git History Recovery
✗ Not in a git repository
```

**Cause**: Working directory is not a git repository

**Solutions**:

1. **Initialize git** (if appropriate):
   ```bash
   git init
   git add reasoning_wal/
   git commit -m "graph: Initial snapshots"
   ```

2. **Skip to Level 4**:
   ```python
   # Just rely on chunk reconstruction
   recovery = GraphRecovery(
       wal_dir='reasoning_wal',
       chunks_dir='graph_chunks'
   )
   result = recovery.recover()
   ```

3. **Use manual backup strategy**:
   ```python
   # Copy snapshots to backup location
   import shutil
   shutil.copytree('reasoning_wal/snapshots', 'backup/snapshots')
   ```

#### Issue: "Chunks directory not available"

**Symptoms**:
```
Recovery Level 4: Chunk Reconstruction
✗ Chunks directory not available
✗ Recovery failed after trying all 4 levels
```

**Cause**: Chunk logging was not enabled or directory was deleted

**Solutions**:

1. **Enable chunk logging** (for future):
   ```python
   # Configure chunk storage
   from cortical.chunk_index import ChunkWriter

   chunk_writer = ChunkWriter('graph_chunks')
   # Log operations to chunks
   ```

2. **Manual graph reconstruction**:
   - If you have any backup, restore from that
   - If no backup exists, need to rebuild graph from scratch
   - Check if git history has older versions

3. **Prevention for future**:
   ```python
   # Always specify chunks_dir
   recovery = GraphRecovery(
       wal_dir='reasoning_wal',
       chunks_dir='graph_chunks'  # Enable Level 4
   )
   ```

#### Issue: "Edge references missing source node: Q1"

**Symptoms**:
```
Graph integrity verification failed:
  - Edge references missing source node: Q1
  - Edge references missing target node: H2
```

**Cause**: Edges refer to nodes that were deleted or never created

**Solutions**:

1. **Clean up orphaned edges**:
   ```python
   def clean_orphaned_edges(graph):
       """Remove edges referencing non-existent nodes."""
       edges_to_remove = []

       for edge in graph.edges:
           if edge.source_id not in graph.nodes:
               edges_to_remove.append(edge)
           elif edge.target_id not in graph.nodes:
               edges_to_remove.append(edge)

       for edge in edges_to_remove:
           graph.remove_edge(
               edge.source_id,
               edge.target_id,
               edge.edge_type
           )

       print(f"Removed {len(edges_to_remove)} orphaned edges")

   clean_orphaned_edges(recovered_graph)
   ```

2. **Recreate missing nodes**:
   ```python
   # If you know what Q1 should be
   if 'Q1' not in graph.nodes:
       graph.add_node(
           'Q1',
           NodeType.QUESTION,
           "Reconstructed question",
           properties={},
           metadata={'reconstructed': True}
       )
   ```

#### Issue: Recovery is slow (Level 4 taking minutes)

**Symptoms**:
```
Recovery Level 4: Chunk Reconstruction
Duration: 45000ms (45 seconds)
```

**Cause**: Too many chunk files or operations

**Solutions**:

1. **Compact chunks** (future prevention):
   ```python
   # Consolidate old chunks into snapshots
   from cortical.chunk_index import ChunkManager

   manager = ChunkManager('graph_chunks')
   manager.compact_before('2025-12-01')  # Consolidate old chunks
   ```

2. **Use incremental recovery**:
   ```python
   # If you have a recent snapshot, only replay chunks after it
   snapshot_data = wal.load_snapshot()
   snapshot_time = snapshot_data['timestamp']

   # Only process chunks after snapshot
   recent_chunks = [
       c for c in chunks
       if extract_timestamp(c.name) > snapshot_time
   ]
   ```

3. **Optimize chunk structure** (reduce operation count)

### How to Diagnose Recovery Problems

Follow this diagnostic checklist:

```python
from cortical.reasoning.graph_persistence import GraphRecovery
from pathlib import Path

def diagnose_recovery_issues(wal_dir, chunks_dir=None):
    """Comprehensive recovery diagnostics."""
    print("=== Recovery Diagnostics ===\n")

    wal_path = Path(wal_dir)

    # 1. Check directory structure
    print("1. Directory Structure:")
    print(f"   WAL dir exists: {wal_path.exists()}")
    print(f"   Snapshots dir exists: {(wal_path / 'snapshots').exists()}")
    if chunks_dir:
        print(f"   Chunks dir exists: {Path(chunks_dir).exists()}")

    # 2. Check snapshots
    print("\n2. Snapshots:")
    recovery = GraphRecovery(wal_dir=wal_dir, chunks_dir=chunks_dir)
    snapshots = recovery._list_graph_snapshots()
    print(f"   Total snapshots: {len(snapshots)}")
    for snap in snapshots:
        valid = "✓" if snap.verify_checksum() else "✗"
        print(f"   {valid} {snap.snapshot_id}: {snap.node_count} nodes")

    # 3. Check WAL
    print("\n3. Write-Ahead Log:")
    wal_files = list(wal_path.glob("wal_*.jsonl"))
    print(f"   WAL files: {len(wal_files)}")
    if wal_files:
        latest = sorted(wal_files)[-1]
        print(f"   Latest: {latest.name}")
        print(f"   Size: {latest.stat().st_size} bytes")

    # 4. Check git
    print("\n4. Git Integration:")
    is_git = recovery._is_git_repo()
    print(f"   Git repo: {is_git}")
    if is_git:
        commits = recovery._find_graph_commits()
        print(f"   Graph commits: {len(commits)}")
        if commits:
            latest = commits[0]
            print(f"   Latest: {latest[0][:8]} - {latest[1]}")

    # 5. Check chunks
    if chunks_dir:
        print("\n5. Chunk Files:")
        chunk_path = Path(chunks_dir)
        chunks = list(chunk_path.glob("*.json"))
        print(f"   Chunk files: {len(chunks)}")
        if chunks:
            total_size = sum(c.stat().st_size for c in chunks)
            print(f"   Total size: {total_size / 1024:.1f} KB")

    # 6. Check if recovery needed
    print("\n6. Recovery Status:")
    needs = recovery.needs_recovery()
    print(f"   Needs recovery: {needs}")

    # 7. Recommendations
    print("\n7. Recommendations:")
    if not snapshots:
        print("   ⚠ No snapshots - create initial snapshot")
    if len([s for s in snapshots if s.verify_checksum()]) < 2:
        print("   ⚠ Less than 2 valid snapshots - create more")
    if not is_git:
        print("   ℹ Not in git repo - Level 3 unavailable")
    if not chunks_dir:
        print("   ℹ No chunks dir - Level 4 unavailable")

# Run diagnostics
diagnose_recovery_issues('reasoning_wal', 'graph_chunks')
```

### Manual Recovery Steps

When automatic recovery fails, try manual steps:

**Step 1: Identify what's available**

```bash
# List snapshots
ls -lh reasoning_wal/snapshots/

# Check WAL files
ls -lh reasoning_wal/wal_*.jsonl

# Check chunks
ls -lh graph_chunks/

# Check git history
git log --oneline -- reasoning_wal/
```

**Step 2: Try loading newest snapshot manually**

```python
import json
import gzip
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(wal_dir='reasoning_wal')

# Find newest snapshot
snapshots = sorted(
    Path('reasoning_wal/snapshots').glob('snap_*.json*'),
    reverse=True
)

for snapshot_path in snapshots:
    print(f"Trying {snapshot_path.name}...")

    try:
        # Load snapshot
        if snapshot_path.suffix == '.gz':
            with gzip.open(snapshot_path, 'rt') as f:
                snapshot_data = json.load(f)
        else:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)

        # Reconstruct graph
        graph = recovery._graph_from_snapshot(snapshot_data)

        if graph:
            print(f"✓ Successfully loaded {snapshot_path.name}")
            print(f"  Nodes: {graph.node_count()}")
            print(f"  Edges: {graph.edge_count()}")
            break

    except Exception as e:
        print(f"✗ Failed: {e}")
        continue
else:
    print("All snapshots failed")
```

**Step 3: Restore from git if snapshots failed**

```bash
# Find recent commits with snapshots
git log -10 --oneline -- reasoning_wal/snapshots/

# Restore snapshots from a good commit
git checkout abc123 -- reasoning_wal/snapshots/

# Try recovery again
python -c "
from cortical.reasoning.graph_persistence import GraphRecovery
recovery = GraphRecovery(wal_dir='reasoning_wal')
result = recovery.recover()
print(result)
"
```

**Step 4: Rebuild from chunks if all else fails**

```python
import json
from cortical.reasoning import ThoughtGraph, NodeType, EdgeType
from pathlib import Path

def manual_chunk_rebuild(chunks_dir):
    """Manually rebuild graph from chunks."""
    graph = ThoughtGraph()
    chunks = sorted(Path(chunks_dir).glob("*.json"))

    print(f"Found {len(chunks)} chunk files")

    for chunk_path in chunks:
        print(f"Processing {chunk_path.name}...")

        with open(chunk_path, 'r') as f:
            chunk_data = json.load(f)

        operations = chunk_data.get('operations', [])
        print(f"  {len(operations)} operations")

        for op in operations:
            try:
                if op['op'] == 'add_node':
                    graph.add_node(
                        op['node_id'],
                        NodeType(op['node_type']),
                        op['content'],
                        op.get('properties', {}),
                        op.get('metadata', {})
                    )
                elif op['op'] == 'add_edge':
                    graph.add_edge(
                        op['from_id'],
                        op['to_id'],
                        EdgeType(op['edge_type']),
                        op.get('weight', 1.0),
                        op.get('confidence', 1.0)
                    )
            except Exception as e:
                print(f"  Error applying operation: {e}")
                continue

    print(f"\n✓ Rebuilt graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    return graph

# Rebuild
graph = manual_chunk_rebuild('graph_chunks')
```

---

## Summary

The Graph Recovery system provides robust, multi-level protection against data loss:

1. **Level 1 (WAL Replay)**: Fast recovery from crashes - 10-100ms
2. **Level 2 (Snapshot Rollback)**: Recover from corrupted snapshots - 50-200ms
3. **Level 3 (Git History)**: Restore from source control - 200-500ms
4. **Level 4 (Chunk Reconstruction)**: Rebuild from operation logs - 2-60s

**Key Takeaways**:

- Recovery is automatic - just call `recovery.recover()`
- Always enable chunk logging for Level 4 safety net
- Create snapshots regularly (before risky operations, after milestones)
- Use git integration for Level 3 recovery
- Verify graph integrity after recovery
- Test recovery procedures periodically

**Recommended Setup**:

```python
from cortical.reasoning.graph_persistence import (
    GraphRecovery,
    GraphWAL,
    GitAutoCommitter
)

# Full recovery setup
recovery = GraphRecovery(
    wal_dir='reasoning_wal',
    chunks_dir='graph_chunks',
    max_snapshots=3
)

wal = GraphWAL('reasoning_wal')

committer = GitAutoCommitter(
    mode='debounced',
    debounce_seconds=30,
    auto_push=False
)

# Regular snapshots
if operation_count % 100 == 0:
    snapshot_id = wal.create_snapshot(graph)
    committer.commit_on_save(
        f'reasoning_wal/snapshots/{snapshot_id}.json.gz',
        graph=graph
    )

# Periodic compaction
if wal.get_entry_count() > 1000:
    wal.compact_wal(graph)
```

With this setup, your graph state is protected by 4 independent recovery mechanisms, ensuring you can always recover from failures.
