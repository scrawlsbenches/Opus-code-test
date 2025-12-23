# Storage Patterns for Git-Based Database Systems

## Executive Summary

This document provides battle-tested patterns specifically for Python systems using **git as slow storage** and **local files/memory as fast storage**. Covers real patterns from your codebase (Cortical Text Processor, Graph of Thought) and extends them for production use.

---

## Pattern 1: Multi-Tier Triage System

### Problem
Every change doesn't need git (expensive). But critical changes must be durably backed up. How to decide?

### Solution: Classification-Based Routing

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

class DataCriticality(Enum):
    """Criticality levels determine storage routing."""
    LOW = 'low'           # Cache, temporary → HOT only
    MEDIUM = 'medium'     # Session state → HOT + WARM
    HIGH = 'high'         # User data, graphs → HOT + WARM + COLD
    CRITICAL = 'critical' # Transaction log, tasks → HOT + WARM + COLD + REMOTE

@dataclass
class DataClassification:
    """Metadata for classifying data."""
    criticality: DataCriticality
    evictable: bool = True  # Can it be evicted from hot?
    compressible: bool = False  # Should it be compressed before git?
    ttl_seconds: int = 86400 * 30  # Default 30 days before archival

class TriageRouter:
    """Route data to appropriate storage tier based on criticality."""

    def __init__(self, storage_stack):
        self.storage = storage_stack
        self.classifications = {}

    def classify(self, key: str, criticality: DataCriticality,
                 **metadata):
        """Register how to handle this data."""
        self.classifications[key] = DataClassification(
            criticality=criticality,
            **metadata
        )

    def route_write(self, key: str, value: Any):
        """Write with routing based on criticality."""
        classification = self.classifications.get(key)

        if not classification:
            # Default: medium criticality
            classification = DataClassification(
                criticality=DataCriticality.MEDIUM
            )

        # ALL data goes to hot (fast write)
        self.storage.promote_to_hot(key, value)

        # Depending on criticality, also write to slower tiers
        if classification.criticality in (DataCriticality.MEDIUM,
                                         DataCriticality.HIGH,
                                         DataCriticality.CRITICAL):
            # Buffer for eventual write to warm
            self.storage.write_buffered(key, value)

        if classification.criticality in (DataCriticality.HIGH,
                                         DataCriticality.CRITICAL):
            # Schedule commit to git
            self.storage.schedule_git_commit(key, value)

        if classification.criticality == DataCriticality.CRITICAL:
            # Push to remote backup
            self.storage.schedule_remote_backup(key, value)

    def route_read(self, key: str) -> Any:
        """Read with appropriate consistency guarantees."""
        classification = self.classifications.get(key)

        if classification and classification.criticality == DataCriticality.LOW:
            # Best-effort read (may be inconsistent)
            return self.storage.read_from_hot_only(key)
        else:
            # Consistent read (check all tiers)
            return self.storage.read(key)

# Usage
router = TriageRouter(storage)

# Register classifications
router.classify('session:abc', DataCriticality.LOW)
router.classify('task:T-20251222-001', DataCriticality.CRITICAL)
router.classify('index:documents', DataCriticality.MEDIUM)

# Routes automatically based on criticality
router.route_write('session:abc', session_data)  # Hot only
router.route_write('task:T-20251222-001', task_data)  # HOT + WARM + GIT + REMOTE
```

### Real-World Use in Cortical

```python
# Tasks are CRITICAL (must never be lost)
router.classify('got:task:*', DataCriticality.CRITICAL)

# Reasoning graphs are HIGH (important but can rebuild)
router.classify('reasoning:graph:*', DataCriticality.HIGH)

# Query cache is LOW (can discard)
router.classify('cache:query:*', DataCriticality.LOW)
```

---

## Pattern 2: Debounced Git Commits

### Problem
Every operation commits to git (too slow). Batch commits (too laggy). How to balance?

### Solution: Debounced Commits with Time and Size Bounds

```python
from threading import Timer, Lock
from dataclasses import dataclass
from typing import Set, Tuple
import time

@dataclass
class CommitBatch:
    """Batch of changes to commit together."""
    changes: Set[str]  # Keys changed
    timestamp: float   # When batch started
    size_bytes: int = 0

    def age_seconds(self) -> float:
        """Age of oldest change in batch."""
        return time.time() - self.timestamp

    def is_ready(self, max_age: float, max_size: int) -> bool:
        """Check if batch should commit."""
        return (self.age_seconds() >= max_age or
                self.size_bytes >= max_size)

class DebouncedGitCommitter:
    """
    Commit to git with intelligent batching:
    - Debounce: Wait for quiet period (e.g., 5s)
    - Max age: Force commit after 60s even if quiet
    - Max size: Force commit if batch >10MB
    """

    def __init__(self, repo_path: str = ".",
                 debounce_sec: float = 5.0,
                 max_age_sec: float = 60.0,
                 max_size_bytes: int = 10 * 1024 * 1024):
        self.repo_path = repo_path
        self.debounce_sec = debounce_sec
        self.max_age_sec = max_age_sec
        self.max_size_bytes = max_size_bytes

        self.batch = CommitBatch(changes=set(), timestamp=time.time())
        self.timer: Optional[Timer] = None
        self.lock = Lock()

    def mark_changed(self, key: str, size_bytes: int = 0):
        """Mark a key as changed."""
        with self.lock:
            self.batch.changes.add(key)
            self.batch.size_bytes += size_bytes

            # Check if should commit immediately
            if self.batch.is_ready(self.max_age_sec, self.max_size_bytes):
                self._commit_batch()
            else:
                self._reschedule_commit()

    def _reschedule_commit(self):
        """Cancel pending timer and schedule new one."""
        if self.timer:
            self.timer.cancel()

        self.timer = Timer(self.debounce_sec, self._commit_batch)
        self.timer.start()

    def _commit_batch(self):
        """Commit the batch."""
        with self.lock:
            if not self.batch.changes:
                return

            keys = list(self.batch.changes)
            print(f"[GIT] Committing {len(keys)} changes")

            # In real implementation: write files, git add, git commit
            # For now: simulate
            self._git_commit(keys)

            # Reset batch
            self.batch = CommitBatch(changes=set(), timestamp=time.time())
            self.timer = None

    def _git_commit(self, keys: list):
        """Execute git commit."""
        import subprocess

        # Add changed files
        files_to_commit = [self._key_to_path(k) for k in keys]

        try:
            subprocess.run(
                ['git', 'add'] + files_to_commit,
                cwd=self.repo_path,
                check=True,
                timeout=30
            )

            subprocess.run(
                ['git', 'commit', '-m', f'batch: {len(keys)} changes'],
                cwd=self.repo_path,
                check=True,
                timeout=30
            )
        except subprocess.CalledProcessError as e:
            print(f"[GIT] Commit failed: {e}")

    def _key_to_path(self, key: str) -> str:
        """Convert storage key to file path."""
        return f"data/{key}.json"

# Usage
committer = DebouncedGitCommitter(
    debounce_sec=5,      # Wait for 5s quiet period
    max_age_sec=60,      # Force commit after 60s
    max_size_bytes=10*1024*1024  # Force commit if >10MB
)

# Each write marks as changed
def write(key: str, value):
    hot_cache[key] = value
    size = len(json.dumps(value))
    committer.mark_changed(key, size_bytes=size)

# Natural batching: rapid writes batch together
for i in range(100):
    write(f'doc:{i}', document_data)
# Only 1 git commit for 100 writes!
```

### Metrics and Tuning

```python
class CommitMetrics:
    """Track commit behavior for optimization."""

    def __init__(self):
        self.total_commits = 0
        self.total_changes = 0
        self.debounce_triggered = 0
        self.age_limit_triggered = 0
        self.size_limit_triggered = 0

    def record_commit(self, num_changes: int, reason: str):
        """Record a commit."""
        self.total_commits += 1
        self.total_changes += num_changes

        if reason == 'debounce':
            self.debounce_triggered += 1
        elif reason == 'age':
            self.age_limit_triggered += 1
        elif reason == 'size':
            self.size_limit_triggered += 1

    def avg_batch_size(self) -> float:
        """Average changes per commit."""
        return self.total_changes / max(1, self.total_commits)

    def recommendations(self) -> list:
        """Suggest tuning parameters."""
        suggestions = []

        if self.avg_batch_size() < 10:
            suggestions.append("Increase debounce_sec (too many small commits)")
        if self.avg_batch_size() > 1000:
            suggestions.append("Decrease max_age_sec or max_size_bytes")
        if self.age_limit_triggered > self.total_commits * 0.5:
            suggestions.append("Increase max_age_sec (many age-triggered)")

        return suggestions
```

---

## Pattern 3: Chunk-Based Git Storage

### Problem
Git is bad with large files. Storing 100MB processor state = bloated history.

### Solution: Append-Only Chunks (from your codebase)

```python
"""
Storage model that's git-friendly:

corpus_chunks/
├── 2025-12-10_21-53-45_a1b2.json  # Session 1: +50 docs
├── 2025-12-10_22-15-30_c3d4.json  # Session 2: +30 docs, -5 docs
└── 2025-12-10_23-00-00_e5f6.json  # Session 3: +10 docs

Total git history is small (3 x ~20KB chunks)
vs single corpus_dev.json (100MB, grows every commit)
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

@dataclass
class ChunkOperation:
    """Single operation in a chunk."""
    op: str  # 'add', 'modify', 'delete'
    doc_id: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Chunk:
    """Append-only chunk of operations."""
    version: int = 1
    timestamp: str = ""
    session_id: str = ""
    branch: str = ""
    operations: List[ChunkOperation] = None

    def to_json(self) -> str:
        """Serialize for storage."""
        ops = []
        for op in self.operations or []:
            ops.append({
                'op': op.op,
                'doc_id': op.doc_id,
                'content': op.content,
                'metadata': op.metadata,
            })

        return json.dumps({
            'version': self.version,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'branch': self.branch,
            'operations': ops
        })

class ChunkedStorage:
    """
    Git-friendly append-only chunk storage.

    Each session creates one timestamped chunk file.
    No merge conflicts, small git history.
    """

    def __init__(self, chunks_dir: str = "corpus_chunks"):
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(exist_ok=True)

        self.current_chunk = None
        self.current_operations = []

    def add_operation(self, op: str, doc_id: str,
                      content: str = None,
                      metadata: Dict = None):
        """Queue operation in current chunk."""
        self.current_operations.append(
            ChunkOperation(op=op, doc_id=doc_id,
                          content=content, metadata=metadata)
        )

    def flush_chunk(self) -> Path:
        """Write current chunk to disk and commit."""
        if not self.current_operations:
            return None

        # Generate chunk filename
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        session_id = "a1b2c3d4"  # In practice: UUID
        filename = f"{timestamp}_{session_id}.json"

        # Create chunk
        chunk = Chunk(
            timestamp=now.isoformat(),
            session_id=session_id,
            branch="main",  # In practice: git branch
            operations=self.current_operations
        )

        # Write to disk
        chunk_path = self.chunks_dir / filename
        with open(chunk_path, 'w') as f:
            f.write(chunk.to_json())

        # Commit to git
        import subprocess
        subprocess.run(['git', 'add', str(chunk_path)], check=True)
        subprocess.run(
            ['git', 'commit', '-m', f'chunk: {len(self.current_operations)} ops'],
            check=True
        )

        # Reset
        self.current_operations = []
        return chunk_path

    def rebuild_corpus(self) -> Dict[str, str]:
        """
        Rebuild full corpus by replaying chunks.

        Example of recovering state after crash:
        - Read all chunks in order
        - Replay operations
        - Rebuild in-memory state
        """
        corpus = {}

        # Replay all chunks in order
        for chunk_file in sorted(self.chunks_dir.glob("*.json")):
            with open(chunk_file) as f:
                chunk_data = json.load(f)

            for op in chunk_data['operations']:
                if op['op'] == 'add':
                    corpus[op['doc_id']] = op['content']
                elif op['op'] == 'modify':
                    corpus[op['doc_id']] = op['content']
                elif op['op'] == 'delete':
                    del corpus[op['doc_id']]

        return corpus
```

### Chunk Compaction (Periodic Maintenance)

```python
class ChunkCompactor:
    """
    Periodically compact old chunks.

    Similar to: `git gc` or `git rebase --root`
    Combines many small chunks into one large one.
    """

    def __init__(self, chunks_dir: str, keep_chunks: int = 10):
        self.chunks_dir = Path(chunks_dir)
        self.keep_chunks = keep_chunks

    def compact(self) -> bool:
        """
        Compact old chunks if too many accumulate.

        Keep only N newest chunks, compact rest into consolidated file.
        """
        chunk_files = sorted(self.chunks_dir.glob("*.json"))

        if len(chunk_files) <= self.keep_chunks:
            return False  # No need to compact

        # Which chunks to compact?
        to_compact = chunk_files[:-self.keep_chunks]
        to_keep = chunk_files[-self.keep_chunks:]

        # Replay and consolidate old chunks
        consolidated_ops = []
        for chunk_file in to_compact:
            with open(chunk_file) as f:
                chunk_data = json.load(f)
            consolidated_ops.extend(chunk_data['operations'])

        # Write consolidated chunk
        consolidated = Chunk(
            timestamp=datetime.now().isoformat(),
            session_id='consolidated',
            operations=consolidated_ops
        )

        consolidated_path = self.chunks_dir / "consolidated.json"
        with open(consolidated_path, 'w') as f:
            f.write(consolidated.to_json())

        # Remove old chunks from git
        for chunk_file in to_compact:
            chunk_file.unlink()

        # Commit cleanup
        import subprocess
        subprocess.run(['git', 'add', '-A'], check=True)
        subprocess.run(
            ['git', 'commit', '-m', f'chore: compact {len(to_compact)} chunks'],
            check=True
        )

        return True
```

---

## Pattern 4: Snapshot + WAL for Crash Recovery

### Problem
Reading all chunks is slow (N chunks = N file reads). But saving full state every time is expensive (WAL bloat).

### Solution: Periodic Snapshots + Incremental WAL

```python
"""
Architecture:
- Snapshot: Full state every 1000 operations (fast recovery)
- WAL: Incremental changes since last snapshot (safe)
- Recovery: Load snapshot + replay WAL
"""

from datetime import datetime
import gzip
import json
from pathlib import Path

class SnapshotManager:
    """Manage periodic snapshots for fast recovery."""

    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)

    def save_snapshot(self, data: Dict, snapshot_id: str) -> Path:
        """Save compressed snapshot."""
        path = self.snapshot_dir / f"snapshot_{snapshot_id}.json.gz"

        with gzip.open(path, 'wt') as f:
            json.dump(data, f)

        return path

    def load_snapshot(self, snapshot_id: str) -> Dict:
        """Load snapshot (fast)."""
        path = self.snapshot_dir / f"snapshot_{snapshot_id}.json.gz"

        with gzip.open(path, 'rt') as f:
            return json.load(f)

class RecoveryManager:
    """Recover state after crash."""

    def __init__(self, snapshot_dir: str, wal_dir: str, chunks_dir: str):
        self.snapshots = SnapshotManager(snapshot_dir)
        self.wal_dir = Path(wal_dir)
        self.chunks_dir = Path(chunks_dir)

    def recover(self) -> Dict:
        """
        Recovery in priority order:

        1. Latest snapshot + WAL (fast, ~0.1s)
        2. Rebuild from chunks (slow, ~1s)
        3. Manual reconstruction (very slow)
        """
        # Try snapshot + WAL (fast path)
        latest_snapshot = self._find_latest_snapshot()
        if latest_snapshot:
            print(f"[RECOVERY] Loading snapshot {latest_snapshot}")
            state = self.snapshots.load_snapshot(latest_snapshot)

            # Replay WAL since snapshot
            wal_entries = self._load_wal_since_snapshot(latest_snapshot)
            for entry in wal_entries:
                self._apply_wal_entry(state, entry)

            print(f"[RECOVERY] Recovered {len(state)} items")
            return state

        # Fallback: rebuild from chunks (slow path)
        print("[RECOVERY] Rebuilding from chunks...")
        state = self._rebuild_from_chunks()

        # Create new snapshot for next time
        snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshots.save_snapshot(state, snapshot_id)

        return state

    def _find_latest_snapshot(self) -> str:
        """Find most recent snapshot."""
        snapshots = list(self.snapshots.snapshot_dir.glob("*.gz"))
        if snapshots:
            return snapshots[-1].stem.replace("snapshot_", "").replace(".json", "")
        return None

    def _load_wal_since_snapshot(self, snapshot_id: str) -> list:
        """Load WAL entries after snapshot timestamp."""
        # Simplified: load all WAL entries
        # In practice: filter by timestamp
        entries = []
        for wal_file in sorted(self.wal_dir.glob("*.json")):
            with open(wal_file) as f:
                entries.append(json.load(f))
        return entries

    def _apply_wal_entry(self, state: Dict, entry: Dict):
        """Apply single WAL entry to state."""
        if entry['op'] == 'set':
            state[entry['key']] = entry['value']
        elif entry['op'] == 'delete':
            state.pop(entry['key'], None)

    def _rebuild_from_chunks(self) -> Dict:
        """Rebuild full state from chunks."""
        # (Implementation from ChunkedStorage.rebuild_corpus)
        pass
```

---

## Pattern 5: Safe Push to Remote with Verification

### Problem
Network push might fail. How to handle safely?

### Solution: Verify Before Push, Retry on Failure

```python
from typing import Optional
import subprocess
import time

class RemoteBackupManager:
    """Safe backup to remote with verification."""

    def __init__(self, repo_path: str = ".", remote: str = "origin"):
        self.repo_path = repo_path
        self.remote = remote
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def push_with_retry(self) -> bool:
        """Push to remote with automatic retry."""
        for attempt in range(self.max_retries):
            try:
                print(f"[PUSH] Attempt {attempt + 1}/{self.max_retries}")

                # Verify repo is clean
                if not self._is_clean():
                    print("[PUSH] Repository not clean, cannot push")
                    return False

                # Do the push
                subprocess.run(
                    ['git', 'push', self.remote, 'HEAD'],
                    cwd=self.repo_path,
                    check=True,
                    timeout=30
                )

                # Verify push succeeded
                if self._verify_push():
                    print("[PUSH] Verified on remote")
                    return True

                print("[PUSH] Verification failed, retrying...")

            except subprocess.CalledProcessError as e:
                print(f"[PUSH] Push failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        print("[PUSH] All retries failed")
        return False

    def _is_clean(self) -> bool:
        """Check if working directory is clean."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return not result.stdout.strip()
        except Exception:
            return False

    def _verify_push(self) -> bool:
        """Verify that remote matches local."""
        try:
            # Get local HEAD
            local_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            local_hash = local_result.stdout.strip()

            # Get remote HEAD
            remote_result = subprocess.run(
                ['git', 'rev-parse', f'{self.remote}/HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            remote_hash = remote_result.stdout.strip()

            return local_hash == remote_hash

        except subprocess.CalledProcessError:
            # Can't verify, assume failure
            return False

# Usage
backup = RemoteBackupManager(remote="origin")

# Safe push with retry
success = backup.push_with_retry()

if success:
    print("Data safely backed up to remote")
else:
    print("Warning: Remote backup failed, data is local-only")
```

---

## Pattern 6: Consistency Guarantees and Transactions

### Problem
Concurrent writes might corrupt state. Multiple related fields might be updated. How to maintain consistency?

### Solution: Transaction-Based Updates

```python
from threading import RLock
from typing import Callable, Any
from enum import Enum

class TransactionIsolation(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = 1   # Fastest, least safe
    READ_COMMITTED = 2
    REPEATABLE_READ = 3
    SERIALIZABLE = 4       # Slowest, most safe

class Transaction:
    """
    Atomic transaction with consistency guarantees.

    Ensures:
    - All-or-nothing: Either all writes apply or none
    - Atomicity: No partial states visible
    - Isolation: Transactions don't see each other's changes
    - Durability: Committed data survives crashes
    """

    def __init__(self, storage, isolation: TransactionIsolation = None):
        self.storage = storage
        self.isolation = isolation or TransactionIsolation.SERIALIZABLE
        self.writes = {}
        self.snapshot = None  # For isolation

    def read(self, key: str) -> Any:
        """Read within transaction."""
        if self.isolation == TransactionIsolation.SERIALIZABLE:
            # If not yet snapshotted, take snapshot
            if self.snapshot is None:
                self.snapshot = self.storage.get_state_snapshot()
            return self.snapshot.get(key)
        else:
            return self.storage.read(key)

    def write(self, key: str, value: Any):
        """Write within transaction (buffered)."""
        self.writes[key] = value

    def commit(self) -> bool:
        """Commit transaction atomically."""
        if not self.writes:
            return True  # No-op

        try:
            # Apply all writes atomically
            with self.storage.lock:
                # For SERIALIZABLE: verify no conflicts
                if self.isolation == TransactionIsolation.SERIALIZABLE:
                    if not self._check_no_conflicts():
                        raise ConflictError("Write conflict detected")

                # Apply all writes
                for key, value in self.writes.items():
                    self.storage.write(key, value)

            return True

        except ConflictError as e:
            print(f"[TX] Conflict: {e}")
            return False

    def _check_no_conflicts(self) -> bool:
        """Check if any keys we read were modified by others."""
        current_state = self.storage.get_state_snapshot()

        for key in self.snapshot.keys():
            if self.snapshot[key] != current_state.get(key):
                return False  # Key was modified!

        return True

# Usage
def transfer_nodes(graph_storage, from_id: str, to_id: str, node_ids: list):
    """
    Example: Move nodes between clusters atomically.

    If crash mid-way: either all nodes moved or none (no partial state).
    """
    tx = Transaction(graph_storage)

    try:
        # Read current state
        from_cluster = tx.read(f'cluster:{from_id}')
        to_cluster = tx.read(f'cluster:{to_id}')

        # Modify
        for node_id in node_ids:
            from_cluster['nodes'].remove(node_id)
            to_cluster['nodes'].append(node_id)

        # Write atomically
        tx.write(f'cluster:{from_id}', from_cluster)
        tx.write(f'cluster:{to_id}', to_cluster)

        # Commit
        if tx.commit():
            print(f"Successfully moved {len(node_ids)} nodes")
        else:
            print("Move failed due to conflict, retrying...")

    except Exception as e:
        print(f"Error: {e}")
```

---

## Real-World Integration: Complete Example

```python
"""
Full integration of all patterns with Cortical's GoT system.
"""

from cortical.got.manager import GoTManager
from cortical.reasoning.graph_persistence import GraphWAL, GitAutoCommitter
from cortical.wal import SnapshotManager

class ProductionStorageStack:
    """Production-grade storage combining all patterns."""

    def __init__(self):
        # Pattern 1: Triage router
        self.router = TriageRouter(self)

        # Pattern 2: Debounced commits
        self.committer = DebouncedGitCommitter(
            debounce_sec=5,
            max_age_sec=60,
            max_size_bytes=10*1024*1024
        )

        # Pattern 3: Chunk storage
        self.chunks = ChunkedStorage()
        self.compactor = ChunkCompactor(keep_chunks=10)

        # Pattern 4: Snapshots + WAL
        self.snapshots = SnapshotManager()
        self.recovery = RecoveryManager()

        # Pattern 5: Remote backup
        self.backup = RemoteBackupManager()

        # Pattern 6: Transactions
        self.lock = RLock()

    def startup(self):
        """Startup with crash recovery."""
        print("[STARTUP] Checking for crash recovery...")

        if Path("crash.lock").exists():
            print("[STARTUP] Detected unclean shutdown, recovering...")
            state = self.recovery.recover()
            print(f"[STARTUP] Recovered {len(state)} items")
            Path("crash.lock").unlink()
        else:
            print("[STARTUP] Clean startup")

    def create_task(self, title: str, **kwargs):
        """Create task with full durability stack."""
        task_id = None

        # Use transaction for atomicity
        tx = Transaction(self)

        try:
            # Create task
            task_id = f"T-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            task_data = {'title': title, **kwargs}

            # Write with triage routing (CRITICAL data)
            self.router.classify(f'task:{task_id}',
                                DataCriticality.CRITICAL)
            tx.write(f'task:{task_id}', task_data)

            # Commit transaction
            if not tx.commit():
                raise ValueError("Transaction commit failed")

            # Mark for git commit
            self.committer.mark_changed(f'task:{task_id}', len(json.dumps(task_data)))

            # Add to chunk for git
            self.chunks.add_operation(
                op='add',
                doc_id=task_id,
                content=json.dumps(task_data)
            )

            # Auto-backup to remote
            # (Would be called periodically in production)

            return task_id

        except Exception as e:
            print(f"[TASK] Create failed: {e}")
            return None

    def shutdown(self, backup_to_remote: bool = True):
        """Graceful shutdown with final backup."""
        print("[SHUTDOWN] Flushing data...")

        # Flush any pending writes
        self.chunks.flush_chunk()
        self.committer._commit_batch()

        # Verify local state
        print("[SHUTDOWN] Verifying local state...")

        # Backup to remote
        if backup_to_remote:
            print("[SHUTDOWN] Backing up to remote...")
            if self.backup.push_with_retry():
                print("[SHUTDOWN] Remote backup successful")
            else:
                print("[SHUTDOWN] WARNING: Remote backup failed!")

        # Create snapshot for next startup
        print("[SHUTDOWN] Creating snapshot for next startup...")
        # (Implementation omitted)

        print("[SHUTDOWN] Done")

# Usage
storage = ProductionStorageStack()
storage.startup()

try:
    task_id = storage.create_task("Important task")
    print(f"Created task: {task_id}")

finally:
    storage.shutdown(backup_to_remote=True)
```

---

## Summary Table: Patterns and When to Use

| Pattern | Problem | Solution | Trade-off |
|---------|---------|----------|-----------|
| **Triage** | Need different durability for different data | Route based on criticality | Extra classification overhead |
| **Debounced Commits** | Too many git commits (bloat), too batchy | Debounce + time bounds | Slight durability lag (5-60s) |
| **Chunked Storage** | Large monolithic files bloat git | Append-only timestamps chunks | Must replay chunks to recover |
| **Snapshot + WAL** | Replaying all chunks is slow | Snapshots every N ops, WAL for incremental | Extra disk space for snapshots |
| **Safe Remote Push** | Network push failures | Verify before push, retry | Network dependent (can fail) |
| **Transactions** | Concurrent writes corrupt state | Atomic all-or-nothing updates | Performance cost for locking |

---

## Recommended Configuration

```python
# For most Python systems with git as slow storage

# HOT tier
HOT_CAPACITY_MB = 256           # Adjust per system
HOT_MAX_ITEMS = 1000

# WARM tier
WARM_PATH = "corpus_warm/"
WARM_CAPACITY_MB = 2048

# Write-behind buffering
WRITE_BUFFER_SIZE = 100         # Items before flush
WRITE_BUFFER_TIME_SEC = 5       # Max debounce

# Git commits
COMMIT_DEBOUNCE_SEC = 5
COMMIT_MAX_AGE_SEC = 60
COMMIT_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Chunks
KEEP_CHUNKS = 10                # Compact after N chunks
CHUNK_WARNING_SIZE_KB = 1024    # Warn if >1MB

# Snapshots
SNAPSHOT_EVERY_N_OPS = 1000     # Snapshot frequency
SNAPSHOT_COMPRESS = True        # gzip

# Remote backup
PUSH_AFTER_COMMITS = 10         # Push every N commits
PUSH_RETRIES = 3
PUSH_RETRY_DELAY_SEC = 2

# Transactions
DEFAULT_ISOLATION = TransactionIsolation.SERIALIZABLE
```

---

*Last Updated: December 2025*
*Based on Cortical Text Processor and Graph of Thought implementations*
