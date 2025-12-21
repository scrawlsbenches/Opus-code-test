# GoT Transactional Architecture Design

**Status:** Draft
**Author:** Claude Agent
**Date:** 2025-12-21
**Task:** Design ACID-compliant transaction layer for Graph of Thought

---

## Executive Summary

This document describes a transactional architecture for the Graph of Thought (GoT) system that provides:

- **Atomicity**: All-or-nothing operations with rollback on failure
- **Consistency**: Invariants are always maintained
- **Isolation**: Concurrent transactions don't see each other's uncommitted changes
- **Durability**: Committed data survives crashes

**Key Insight:** Git IS our database. We leverage git's existing guarantees (content-addressed storage, atomic commits, checksums, history) rather than reinventing them.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Architecture Overview](#architecture-overview)
3. [Transaction Lifecycle](#transaction-lifecycle)
4. [MVCC: Snapshot Isolation](#mvcc-snapshot-isolation)
5. [Optimistic Locking](#optimistic-locking)
6. [Write-Ahead Log (WAL)](#write-ahead-log-wal)
7. [Recovery Procedures](#recovery-procedures)
8. [Conflict Resolution](#conflict-resolution)
9. [API Design](#api-design)
10. [File Structure](#file-structure)
11. [Implementation Plan](#implementation-plan)
12. [Failure Scenarios](#failure-scenarios)

---

## Design Principles

### 1. Git as Durability Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  (GoTProjectManager, Tasks, Decisions, Edges)                   │
├─────────────────────────────────────────────────────────────────┤
│                    TRANSACTION LAYER                            │
│  (Begin, Read, Write, Commit, Rollback)                         │
├─────────────────────────────────────────────────────────────────┤
│                         WAL LAYER                               │
│  (Checksums, Fsync, Redo/Undo Log)                              │
├─────────────────────────────────────────────────────────────────┤
│                      GIT STORAGE LAYER                          │
│  (Atomic commits, Content-addressed, Immutable history)         │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Zero External Dependencies
- Pure Python implementation
- Uses only git (already required for the project)
- No SQLite, no Redis, no external services

### 3. Agent-First Design
- Claude Agents are primary users
- Operations must be idempotent where possible
- Clear error messages for conflict resolution
- Automatic recovery from common failures

### 4. Fail-Safe Defaults
- Uncommitted transactions are automatically rolled back on crash
- Corrupted data is detected and rejected (never silently accepted)
- Recovery always leaves system in consistent state

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Transaction │    │ Transaction │    │ Transaction │         │
│  │     T1      │    │     T2      │    │     T3      │         │
│  │ (Agent A)   │    │ (Agent B)   │    │ (Agent C)   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TRANSACTION MANAGER                         │   │
│  │  - Assigns transaction IDs                               │   │
│  │  - Tracks active transactions                            │   │
│  │  - Coordinates commits                                   │   │
│  │  - Detects conflicts                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    WAL MANAGER                           │   │
│  │  - Logs operations before execution                      │   │
│  │  - Checksums all entries                                 │   │
│  │  - Fsync for durability                                  │   │
│  │  - Supports redo/undo                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   GIT BACKEND                            │   │
│  │  - Atomic commits                                        │   │
│  │  - Content-addressed storage                             │   │
│  │  - Branch-based isolation                                │   │
│  │  - Merge-based conflict resolution                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### State Machine

```
                    ┌──────────┐
                    │  BEGIN   │
                    └────┬─────┘
                         │
                         ▼
              ┌──────────────────────┐
              │       ACTIVE         │◄─────────────┐
              │  (read/write ops)    │              │
              └──────────┬───────────┘              │
                         │                          │
            ┌────────────┼────────────┐             │
            │            │            │             │
            ▼            ▼            ▼             │
      ┌──────────┐ ┌──────────┐ ┌──────────┐       │
      │ PREPARE  │ │ ROLLBACK │ │  ABORT   │       │
      │ COMMIT   │ │(explicit)│ │ (crash)  │       │
      └────┬─────┘ └────┬─────┘ └────┬─────┘       │
           │            │            │              │
           ▼            │            │              │
    ┌─────────────┐     │            │              │
    │  VALIDATE   │     │            │              │
    │ (conflicts?)│     │            │              │
    └──────┬──────┘     │            │              │
           │            │            │              │
     ┌─────┴─────┐      │            │              │
     │           │      │            │              │
     ▼           ▼      ▼            ▼              │
┌─────────┐ ┌─────────────────────────────┐        │
│COMMITTED│ │         ROLLED BACK         │        │
│(durable)│ │  (changes discarded)        │        │
└─────────┘ └─────────────────────────────┘        │
                         │                          │
                         │      RETRY               │
                         └──────────────────────────┘
```

---

## Transaction Lifecycle

### 1. Begin Transaction

```python
def begin(self, isolation_level: str = "snapshot") -> Transaction:
    """
    Start a new transaction.

    Args:
        isolation_level: "snapshot" (default) or "serializable"

    Returns:
        Transaction object with unique ID and base snapshot
    """
    tx_id = generate_transaction_id()  # TX-YYYYMMDD-HHMMSS-XXXX

    # Record the snapshot point (current git HEAD)
    base_commit = git_get_head()

    # Create transaction record
    tx = Transaction(
        id=tx_id,
        base_commit=base_commit,
        isolation_level=isolation_level,
        state=TransactionState.ACTIVE,
        started_at=datetime.now(),
        operations=[],  # Buffered operations
        read_set={},    # Keys read (for conflict detection)
        write_set={},   # Keys written (pending changes)
    )

    # Write transaction record to WAL (survives crash)
    self.wal.log_tx_begin(tx)

    # Register as active transaction
    self.active_transactions[tx_id] = tx

    return tx
```

### 2. Read Operations (Snapshot Isolation)

```python
def read(self, tx: Transaction, key: str) -> Optional[Any]:
    """
    Read a value within a transaction.

    Provides snapshot isolation: reads see the database state
    as of transaction start, plus any writes made within this transaction.
    """
    # First check our own pending writes
    if key in tx.write_set:
        return tx.write_set[key]

    # Read from our snapshot (base_commit)
    value = self.read_at_commit(key, tx.base_commit)

    # Track what we read (for serializable isolation conflict detection)
    tx.read_set[key] = hash(value) if value else None

    return value
```

### 3. Write Operations (Buffered)

```python
def write(self, tx: Transaction, key: str, value: Any) -> None:
    """
    Write a value within a transaction.

    Writes are buffered until commit. Other transactions
    cannot see these changes until commit succeeds.
    """
    if tx.state != TransactionState.ACTIVE:
        raise TransactionError(f"Transaction {tx.id} is not active")

    # Create undo record (for rollback)
    old_value = self.read(tx, key)

    # Log to WAL before buffering (crash recovery)
    self.wal.log_write(tx.id, key, old_value, value)

    # Buffer the write
    tx.write_set[key] = value
    tx.operations.append(WriteOp(key=key, old=old_value, new=value))
```

### 4. Commit (Optimistic Validation)

```python
def commit(self, tx: Transaction) -> CommitResult:
    """
    Attempt to commit a transaction.

    Uses optimistic concurrency control:
    1. Validate no conflicts with concurrent commits
    2. Apply changes atomically via git commit
    3. Return success or conflict details
    """
    if tx.state != TransactionState.ACTIVE:
        raise TransactionError(f"Transaction {tx.id} is not active")

    # Phase 1: PREPARE
    tx.state = TransactionState.PREPARING
    self.wal.log_tx_prepare(tx.id)

    # Phase 2: VALIDATE (optimistic lock check)
    current_head = git_get_head()

    if current_head != tx.base_commit:
        # Someone else committed. Check for actual conflicts.
        conflicts = self.detect_conflicts(tx, current_head)

        if conflicts:
            # Cannot auto-merge, return conflict info
            tx.state = TransactionState.ABORTED
            self.wal.log_tx_abort(tx.id, reason="conflict")
            return CommitResult(
                success=False,
                reason="conflict",
                conflicts=conflicts,
                suggestion=self.suggest_resolution(conflicts)
            )

        # No conflicts, can fast-forward or merge
        tx.base_commit = current_head

    # Phase 3: APPLY (atomic git commit)
    try:
        # Write all changes to files
        for key, value in tx.write_set.items():
            self.write_to_file(key, value)

        # Atomic git commit
        commit_sha = self.git_commit(
            message=f"Transaction {tx.id}",
            files=list(tx.write_set.keys())
        )

        # Fsync the commit
        self.fsync_git_objects()

        # Phase 4: FINALIZE
        tx.state = TransactionState.COMMITTED
        tx.commit_sha = commit_sha
        self.wal.log_tx_commit(tx.id, commit_sha)

        # Cleanup
        del self.active_transactions[tx.id]

        return CommitResult(
            success=True,
            commit_sha=commit_sha,
            version=self.increment_version()
        )

    except Exception as e:
        # Commit failed, rollback
        tx.state = TransactionState.ABORTED
        self.wal.log_tx_abort(tx.id, reason=str(e))
        self.rollback_files(tx)
        raise
```

### 5. Rollback

```python
def rollback(self, tx: Transaction, reason: str = "explicit") -> None:
    """
    Abort a transaction and discard all changes.
    """
    if tx.state == TransactionState.COMMITTED:
        raise TransactionError("Cannot rollback committed transaction")

    # Log the rollback
    self.wal.log_tx_rollback(tx.id, reason)

    # Discard buffered writes (they were never applied)
    tx.write_set.clear()
    tx.operations.clear()

    # Mark as rolled back
    tx.state = TransactionState.ROLLED_BACK

    # Cleanup
    if tx.id in self.active_transactions:
        del self.active_transactions[tx.id]
```

---

## MVCC: Snapshot Isolation

### Concept

Each transaction sees a consistent snapshot of the database as of its start time. Concurrent transactions don't see each other's uncommitted changes.

```
Timeline:
─────────────────────────────────────────────────────────────────►

T1: begin()──read(A)──────────────write(A)──commit()
              │                      │
              ▼                      ▼
         sees A=1               writes A=2

T2:      begin()───read(A)────────────────────read(A)──commit()
                     │                          │
                     ▼                          ▼
                sees A=1                   sees A=1 (snapshot!)
                (T1 not committed)         (T1 committed but after T2 started)
```

### Implementation Using Git

```python
class SnapshotReader:
    """Read from a specific git commit (snapshot)."""

    def __init__(self, commit_sha: str):
        self.commit_sha = commit_sha
        self._cache = {}

    def read(self, path: str) -> Optional[bytes]:
        """Read file contents at this snapshot."""
        if path in self._cache:
            return self._cache[path]

        try:
            # Use git show to read file at specific commit
            result = subprocess.run(
                ["git", "show", f"{self.commit_sha}:{path}"],
                capture_output=True,
                check=True
            )
            content = result.stdout
            self._cache[path] = content
            return content
        except subprocess.CalledProcessError:
            # File doesn't exist at this commit
            return None

    def list_files(self, pattern: str = "*") -> List[str]:
        """List files matching pattern at this snapshot."""
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", self.commit_sha],
            capture_output=True,
            text=True
        )
        files = result.stdout.strip().split('\n')
        return [f for f in files if fnmatch(f, pattern)]
```

### Snapshot Registry

```python
class SnapshotRegistry:
    """Track active snapshots to prevent garbage collection."""

    def __init__(self, got_dir: Path):
        self.snapshots_file = got_dir / "active_snapshots.json"
        self._snapshots: Dict[str, SnapshotInfo] = {}

    def register(self, tx_id: str, commit_sha: str) -> None:
        """Register a snapshot as in-use."""
        self._snapshots[tx_id] = SnapshotInfo(
            commit_sha=commit_sha,
            registered_at=datetime.now().isoformat(),
            tx_id=tx_id
        )
        self._persist()

    def release(self, tx_id: str) -> None:
        """Release a snapshot (transaction completed)."""
        if tx_id in self._snapshots:
            del self._snapshots[tx_id]
            self._persist()

    def get_protected_commits(self) -> Set[str]:
        """Get commits that should not be garbage collected."""
        return {s.commit_sha for s in self._snapshots.values()}
```

---

## Optimistic Locking

### Version Numbers

Each entity (task, decision, edge) carries a version number that increments on every modification.

```python
@dataclass
class VersionedEntity:
    """Base class for all versioned entities."""
    id: str
    version: int  # Monotonically increasing
    data: Dict[str, Any]
    checksum: str  # SHA256 of serialized data

    def increment_version(self) -> 'VersionedEntity':
        return VersionedEntity(
            id=self.id,
            version=self.version + 1,
            data=self.data,
            checksum=self.compute_checksum()
        )

    def compute_checksum(self) -> str:
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Conflict Detection

```python
def detect_conflicts(
    self,
    tx: Transaction,
    current_head: str
) -> List[Conflict]:
    """
    Detect conflicts between transaction and commits since base.

    A conflict exists when:
    1. Transaction wrote to a key that was modified by another commit
    2. Transaction read a key that was modified (serializable only)
    """
    conflicts = []

    # Get all commits between our base and current head
    commits_since = git_log(f"{tx.base_commit}..{current_head}")

    for commit in commits_since:
        # Get files modified in this commit
        modified_files = git_diff_files(commit.parent, commit.sha)

        for path in modified_files:
            # Check write-write conflict
            if path in tx.write_set:
                conflicts.append(Conflict(
                    type=ConflictType.WRITE_WRITE,
                    key=path,
                    our_value=tx.write_set[path],
                    their_value=self.read_at_commit(path, current_head),
                    their_commit=commit.sha,
                    their_author=commit.author
                ))

            # Check read-write conflict (serializable isolation)
            if tx.isolation_level == "serializable" and path in tx.read_set:
                current_hash = hash(self.read_at_commit(path, current_head))
                if current_hash != tx.read_set[path]:
                    conflicts.append(Conflict(
                        type=ConflictType.READ_WRITE,
                        key=path,
                        read_version=tx.read_set[path],
                        current_version=current_hash,
                        their_commit=commit.sha
                    ))

    return conflicts
```

### Compare-and-Swap Semantics

```python
def cas_update(
    self,
    tx: Transaction,
    key: str,
    expected_version: int,
    new_value: Any
) -> bool:
    """
    Compare-and-swap update within a transaction.

    Only succeeds if current version matches expected.
    """
    current = self.read(tx, key)

    if current is None:
        if expected_version != 0:
            return False  # Expected existing, got nothing
    elif current.version != expected_version:
        return False  # Version mismatch

    # Version matches, apply update
    new_entity = VersionedEntity(
        id=key,
        version=expected_version + 1,
        data=new_value,
        checksum=compute_checksum(new_value)
    )

    self.write(tx, key, new_entity)
    return True
```

---

## Write-Ahead Log (WAL)

### WAL Entry Format

```python
@dataclass
class WALEntry:
    """Single entry in the write-ahead log."""
    sequence: int           # Monotonic sequence number
    timestamp: str          # ISO format timestamp
    tx_id: str              # Transaction ID
    operation: str          # Operation type
    data: Dict[str, Any]    # Operation-specific data
    checksum: str           # SHA256 of above fields

    def serialize(self) -> bytes:
        """Serialize to bytes with length prefix."""
        content = json.dumps({
            'seq': self.sequence,
            'ts': self.timestamp,
            'tx': self.tx_id,
            'op': self.operation,
            'data': self.data,
            'checksum': self.checksum
        }, separators=(',', ':'))
        encoded = content.encode('utf-8')
        # Length-prefixed format: [4-byte length][content]
        return struct.pack('>I', len(encoded)) + encoded

    @classmethod
    def deserialize(cls, data: bytes) -> 'WALEntry':
        """Deserialize and verify checksum."""
        length = struct.unpack('>I', data[:4])[0]
        content = json.loads(data[4:4+length].decode('utf-8'))

        # Verify checksum
        expected = cls.compute_checksum(
            content['seq'], content['ts'], content['tx'],
            content['op'], content['data']
        )
        if content['checksum'] != expected:
            raise CorruptedWALError(f"Checksum mismatch at seq {content['seq']}")

        return cls(**content)

    @staticmethod
    def compute_checksum(seq, ts, tx, op, data) -> str:
        content = f"{seq}|{ts}|{tx}|{op}|{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### WAL Operations

```python
class WALManager:
    """Manages the write-ahead log with fsync guarantees."""

    def __init__(self, wal_dir: Path):
        self.wal_dir = wal_dir
        self.current_file = wal_dir / "current.wal"
        self.sequence = self._load_sequence()
        self._fd = None

    def log(self, tx_id: str, operation: str, data: Dict) -> int:
        """
        Append entry to WAL with fsync.

        Returns sequence number.
        """
        self.sequence += 1

        entry = WALEntry(
            sequence=self.sequence,
            timestamp=datetime.now().isoformat(),
            tx_id=tx_id,
            operation=operation,
            data=data,
            checksum=WALEntry.compute_checksum(
                self.sequence,
                datetime.now().isoformat(),
                tx_id, operation, data
            )
        )

        # Write with fsync
        serialized = entry.serialize()

        with open(self.current_file, 'ab') as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())  # CRITICAL: ensure on disk

        return self.sequence

    # Convenience methods for specific operations
    def log_tx_begin(self, tx: Transaction) -> int:
        return self.log(tx.id, "TX_BEGIN", {
            'base_commit': tx.base_commit,
            'isolation': tx.isolation_level
        })

    def log_write(self, tx_id: str, key: str, old: Any, new: Any) -> int:
        return self.log(tx_id, "WRITE", {
            'key': key,
            'old': self._serialize_value(old),
            'new': self._serialize_value(new)
        })

    def log_tx_prepare(self, tx_id: str) -> int:
        return self.log(tx_id, "TX_PREPARE", {})

    def log_tx_commit(self, tx_id: str, commit_sha: str) -> int:
        return self.log(tx_id, "TX_COMMIT", {'commit_sha': commit_sha})

    def log_tx_abort(self, tx_id: str, reason: str) -> int:
        return self.log(tx_id, "TX_ABORT", {'reason': reason})

    def log_tx_rollback(self, tx_id: str, reason: str) -> int:
        return self.log(tx_id, "TX_ROLLBACK", {'reason': reason})
```

### WAL Recovery

```python
class WALRecovery:
    """Recover from WAL after crash."""

    def __init__(self, wal_manager: WALManager):
        self.wal = wal_manager

    def recover(self) -> RecoveryResult:
        """
        Recover system state from WAL.

        Algorithm:
        1. Read all WAL entries
        2. Group by transaction
        3. For each transaction:
           - If COMMITTED: verify git commit exists (already durable)
           - If PREPARED but not COMMITTED: rollback (crash during commit)
           - If ACTIVE: rollback (crash during transaction)
        """
        entries = self.wal.read_all()
        transactions = self._group_by_transaction(entries)

        result = RecoveryResult()

        for tx_id, tx_entries in transactions.items():
            final_state = self._determine_final_state(tx_entries)

            if final_state == "COMMITTED":
                # Verify the commit exists in git
                commit_sha = self._get_commit_sha(tx_entries)
                if self._verify_git_commit(commit_sha):
                    result.committed.append(tx_id)
                else:
                    # Commit claimed but not in git - corruption!
                    result.corrupted.append(tx_id)

            elif final_state == "PREPARED":
                # Crash during commit - rollback
                self._undo_transaction(tx_entries)
                result.rolled_back.append(tx_id)

            elif final_state == "ACTIVE":
                # Crash during transaction - rollback
                # Writes were only buffered, nothing to undo in git
                result.rolled_back.append(tx_id)

            elif final_state in ("ABORTED", "ROLLED_BACK"):
                # Already handled
                result.already_handled.append(tx_id)

        # Truncate WAL after successful recovery
        self.wal.truncate_before(result.min_safe_sequence)

        return result

    def _undo_transaction(self, entries: List[WALEntry]) -> None:
        """Undo writes from a failed transaction."""
        # Process in reverse order
        writes = [e for e in entries if e.operation == "WRITE"]

        for entry in reversed(writes):
            key = entry.data['key']
            old_value = entry.data['old']

            # Restore old value
            if old_value is None:
                self._delete_file(key)
            else:
                self._write_file(key, old_value)
```

---

## Recovery Procedures

### Startup Recovery

```python
def startup_recovery(got_dir: Path) -> None:
    """
    Run on every startup to ensure consistent state.

    This is idempotent - safe to run multiple times.
    """
    # 1. Check for incomplete transactions in WAL
    wal = WALManager(got_dir / "wal")
    recovery = WALRecovery(wal)
    result = recovery.recover()

    if result.corrupted:
        raise CorruptionError(
            f"Corrupted transactions found: {result.corrupted}. "
            "Manual intervention required."
        )

    logger.info(
        f"Recovery complete: "
        f"{len(result.committed)} committed, "
        f"{len(result.rolled_back)} rolled back"
    )

    # 2. Verify git state matches expected
    verify_git_integrity(got_dir)

    # 3. Rebuild in-memory graph from git state
    rebuild_graph_from_git(got_dir)

    # 4. Clear stale transaction files
    cleanup_stale_transactions(got_dir)
```

### Corruption Detection

```python
def verify_integrity(got_dir: Path) -> IntegrityReport:
    """
    Comprehensive integrity check.

    Checks:
    1. All files have valid checksums
    2. All referenced commits exist
    3. Version numbers are monotonic
    4. No orphaned transactions
    """
    report = IntegrityReport()

    # Check each entity file
    for entity_file in (got_dir / "entities").glob("*.json"):
        try:
            entity = load_entity(entity_file)

            # Verify checksum
            expected = entity.compute_checksum()
            if entity.checksum != expected:
                report.add_error(
                    entity_file,
                    f"Checksum mismatch: expected {expected}, got {entity.checksum}"
                )

            # Verify version is positive
            if entity.version < 1:
                report.add_error(entity_file, f"Invalid version: {entity.version}")

        except json.JSONDecodeError as e:
            report.add_error(entity_file, f"JSON parse error: {e}")
        except Exception as e:
            report.add_error(entity_file, f"Unexpected error: {e}")

    # Check WAL integrity
    wal = WALManager(got_dir / "wal")
    for entry in wal.read_all():
        try:
            entry.verify_checksum()
        except ChecksumError as e:
            report.add_error("wal", f"WAL entry {entry.sequence}: {e}")

    return report
```

---

## Conflict Resolution

### Automatic Resolution Strategies

```python
class ConflictResolver:
    """Resolve conflicts between concurrent transactions."""

    def resolve(self, conflict: Conflict) -> Resolution:
        """
        Attempt automatic resolution.

        Strategies:
        1. Last-write-wins (for independent fields)
        2. Merge (for additive changes)
        3. Manual (for true conflicts)
        """
        if conflict.type == ConflictType.WRITE_WRITE:
            return self._resolve_write_write(conflict)
        elif conflict.type == ConflictType.READ_WRITE:
            return self._resolve_read_write(conflict)
        else:
            return Resolution(strategy="manual", reason="Unknown conflict type")

    def _resolve_write_write(self, conflict: Conflict) -> Resolution:
        """Resolve write-write conflict."""
        our = conflict.our_value
        their = conflict.their_value
        base = conflict.base_value

        # If we're adding a field they didn't touch, merge
        if isinstance(our, dict) and isinstance(their, dict):
            our_keys = set(our.keys()) - set(base.keys())
            their_keys = set(their.keys()) - set(base.keys())

            if not our_keys & their_keys:
                # No overlapping new keys, can merge
                merged = {**base, **their, **our}
                return Resolution(
                    strategy="merge",
                    result=merged,
                    description="Merged non-overlapping changes"
                )

        # Check if changes are identical (no real conflict)
        if our == their:
            return Resolution(
                strategy="no_conflict",
                result=our,
                description="Both made identical changes"
            )

        # True conflict - needs manual resolution
        return Resolution(
            strategy="manual",
            reason="Conflicting changes to same field",
            our_value=our,
            their_value=their,
            suggestion=self._suggest_resolution(conflict)
        )

    def _suggest_resolution(self, conflict: Conflict) -> str:
        """Generate helpful suggestion for manual resolution."""
        return (
            f"Conflict on {conflict.key}:\n"
            f"  Your change: {json.dumps(conflict.our_value, indent=2)}\n"
            f"  Their change: {json.dumps(conflict.their_value, indent=2)}\n"
            f"  (by {conflict.their_author} in {conflict.their_commit[:8]})\n"
            f"\n"
            f"Options:\n"
            f"  1. Retry with --force to overwrite their changes\n"
            f"  2. Abort and merge manually\n"
            f"  3. Use 'got conflict resolve {conflict.key}' for interactive merge"
        )
```

### Conflict UI for Agents

```python
def cmd_conflict_resolve(key: str, strategy: str = "interactive") -> None:
    """
    Resolve a conflict.

    Strategies:
    - interactive: Show diff and ask for resolution
    - ours: Keep our changes
    - theirs: Keep their changes
    - merge: Attempt automatic merge
    """
    conflict = load_pending_conflict(key)

    if strategy == "interactive":
        print(f"Conflict on {key}:")
        print(f"\n=== BASE ===")
        print(json.dumps(conflict.base_value, indent=2))
        print(f"\n=== OURS ===")
        print(json.dumps(conflict.our_value, indent=2))
        print(f"\n=== THEIRS ({conflict.their_author}) ===")
        print(json.dumps(conflict.their_value, indent=2))
        print(f"\nChoose resolution: [o]urs, [t]heirs, [m]erge, [a]bort")
        # ... handle input

    elif strategy == "ours":
        apply_resolution(key, conflict.our_value)

    elif strategy == "theirs":
        apply_resolution(key, conflict.their_value)

    elif strategy == "merge":
        merged = attempt_merge(conflict)
        if merged.success:
            apply_resolution(key, merged.result)
        else:
            print(f"Automatic merge failed: {merged.reason}")
            print("Use 'got conflict resolve --interactive' for manual merge")
```

---

## API Design

### High-Level Transaction API

```python
class GoTTransactionalManager:
    """
    Main API for transactional GoT operations.

    Example usage:

        manager = GoTTransactionalManager(got_dir=".got")

        # Explicit transaction
        with manager.transaction() as tx:
            task = tx.create_task("Implement feature X")
            tx.add_dependency(task.id, other_task.id)
            # Commit happens automatically at end of 'with' block
            # Rollback happens automatically if exception raised

        # Auto-transaction (each operation is its own transaction)
        manager.auto.create_task("Quick task")
    """

    def __init__(self, got_dir: Path):
        self.got_dir = Path(got_dir)
        self.tx_manager = TransactionManager(got_dir)
        self.wal = WALManager(got_dir / "wal")
        self.auto = AutoTransactionProxy(self)

        # Run startup recovery
        startup_recovery(got_dir)

    @contextmanager
    def transaction(
        self,
        isolation: str = "snapshot",
        retry_on_conflict: int = 3
    ) -> Generator[TransactionContext, None, None]:
        """
        Start a transaction with automatic commit/rollback.

        Args:
            isolation: "snapshot" or "serializable"
            retry_on_conflict: Number of automatic retries on conflict
        """
        attempts = 0

        while attempts <= retry_on_conflict:
            tx = self.tx_manager.begin(isolation)
            ctx = TransactionContext(self, tx)

            try:
                yield ctx

                # Attempt commit
                result = self.tx_manager.commit(tx)

                if result.success:
                    return  # Success!

                elif result.reason == "conflict":
                    attempts += 1
                    if attempts <= retry_on_conflict:
                        logger.info(
                            f"Conflict detected, retrying ({attempts}/{retry_on_conflict})"
                        )
                        continue
                    else:
                        raise ConflictError(result.conflicts)
                else:
                    raise CommitError(result.reason)

            except Exception as e:
                self.tx_manager.rollback(tx, reason=str(e))
                raise

    def read_snapshot(self, commit: Optional[str] = None) -> SnapshotReader:
        """
        Get a read-only snapshot of the database.

        Args:
            commit: Specific commit SHA, or None for current HEAD
        """
        if commit is None:
            commit = git_get_head()
        return SnapshotReader(commit)
```

### Transaction Context

```python
class TransactionContext:
    """
    Context object passed to transaction block.

    Provides task/decision/edge operations within the transaction.
    """

    def __init__(self, manager: GoTTransactionalManager, tx: Transaction):
        self._manager = manager
        self._tx = tx

    # Task operations
    def create_task(
        self,
        title: str,
        priority: str = "medium",
        category: str = "feature",
        **kwargs
    ) -> Task:
        """Create a task within this transaction."""
        task_id = generate_task_id()
        task = Task(
            id=task_id,
            version=1,
            title=title,
            priority=priority,
            category=category,
            status="pending",
            created_at=datetime.now().isoformat(),
            **kwargs
        )

        self._manager.tx_manager.write(
            self._tx,
            f"tasks/{task_id}.json",
            task.to_versioned_entity()
        )

        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Read a task within this transaction (snapshot isolation)."""
        entity = self._manager.tx_manager.read(
            self._tx,
            f"tasks/{task_id}.json"
        )
        return Task.from_versioned_entity(entity) if entity else None

    def update_task(
        self,
        task_id: str,
        expected_version: int,
        **changes
    ) -> bool:
        """
        Update a task with optimistic locking.

        Returns False if version mismatch (someone else modified).
        """
        task = self.get_task(task_id)
        if task is None:
            return False

        if task.version != expected_version:
            return False  # Optimistic lock failed

        # Apply changes
        updated = task.copy(
            version=expected_version + 1,
            **changes
        )

        self._manager.tx_manager.write(
            self._tx,
            f"tasks/{task_id}.json",
            updated.to_versioned_entity()
        )

        return True

    # Decision operations
    def log_decision(
        self,
        decision: str,
        rationale: str,
        affects: List[str] = None,
        **kwargs
    ) -> Decision:
        """Log a decision within this transaction."""
        decision_id = generate_decision_id()
        dec = Decision(
            id=decision_id,
            version=1,
            decision=decision,
            rationale=rationale,
            affects=affects or [],
            created_at=datetime.now().isoformat(),
            **kwargs
        )

        self._manager.tx_manager.write(
            self._tx,
            f"decisions/{decision_id}.json",
            dec.to_versioned_entity()
        )

        # Create edges to affected entities
        for affected_id in (affects or []):
            self.add_edge(decision_id, affected_id, "MOTIVATES")

        return dec

    # Edge operations
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        **properties
    ) -> Edge:
        """Add an edge within this transaction."""
        edge_id = f"{source_id}--{edge_type}-->{target_id}"
        edge = Edge(
            id=edge_id,
            version=1,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            created_at=datetime.now().isoformat(),
            **properties
        )

        self._manager.tx_manager.write(
            self._tx,
            f"edges/{edge_id}.json",
            edge.to_versioned_entity()
        )

        return edge
```

---

## File Structure

```
.got/
├── config.json                 # GoT configuration
├── HEAD                        # Current version pointer (git commit SHA)
│
├── wal/                        # Write-Ahead Log
│   ├── current.wal             # Active WAL file (binary, checksummed)
│   ├── archive/                # Archived WAL segments
│   │   ├── 0001.wal
│   │   └── 0002.wal
│   └── sequence               # Current sequence number
│
├── transactions/               # Active transaction state
│   └── TX-20251221-120000-a1b2.json
│
├── snapshots/                  # Named snapshots for recovery
│   ├── registry.json           # Active snapshot registry
│   └── refs/                   # Snapshot references
│       └── pre-migration-v2    # Named snapshot
│
├── entities/                   # Current state (git-tracked)
│   ├── tasks/
│   │   ├── T-20251221-001.json
│   │   └── T-20251221-002.json
│   ├── decisions/
│   │   └── D-20251221-001.json
│   ├── edges/
│   │   └── T-001--DEPENDS_ON-->T-002.json
│   └── sprints/
│       └── S-2025-W51.json
│
├── indices/                    # Derived indices (rebuilt from entities)
│   ├── by-status.json          # Tasks grouped by status
│   ├── by-priority.json        # Tasks grouped by priority
│   └── graph.json              # Full graph structure
│
└── events/                     # Append-only event log (legacy compatibility)
    └── 20251221-session.jsonl
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1)

1. **WAL Implementation**
   - [ ] WALEntry with checksums
   - [ ] WALManager with fsync
   - [ ] WAL recovery on startup
   - [ ] Unit tests for WAL

2. **Version Numbers**
   - [ ] VersionedEntity base class
   - [ ] Checksum computation
   - [ ] Version increment logic

### Phase 2: Transactions (Week 2)

3. **Transaction Manager**
   - [ ] Begin/commit/rollback
   - [ ] Transaction state machine
   - [ ] Active transaction tracking

4. **Snapshot Isolation**
   - [ ] SnapshotReader using git
   - [ ] Read-your-writes within transaction
   - [ ] Snapshot registry

### Phase 3: Conflict Handling (Week 3)

5. **Optimistic Locking**
   - [ ] Conflict detection
   - [ ] Version comparison
   - [ ] CAS operations

6. **Conflict Resolution**
   - [ ] Automatic merge strategies
   - [ ] Conflict UI for agents
   - [ ] Retry logic

### Phase 4: Integration (Week 4)

7. **High-Level API**
   - [ ] GoTTransactionalManager
   - [ ] TransactionContext
   - [ ] Auto-transaction proxy

8. **Migration**
   - [ ] Migrate existing entities to versioned format
   - [ ] Backward compatibility layer
   - [ ] Documentation

---

## Failure Scenarios

### Scenario 1: Crash During Commit

```
Timeline:
1. Transaction T1 calls commit()
2. WAL logs TX_PREPARE
3. Files written to disk
4. CRASH before git commit

Recovery:
1. Startup reads WAL
2. Finds T1 in PREPARED state
3. No TX_COMMIT entry
4. Rolls back T1 (undoes file changes)
5. Logs TX_ROLLBACK to WAL
```

### Scenario 2: Concurrent Modification

```
Timeline:
1. Agent A: begin() → base_commit = abc123
2. Agent B: begin() → base_commit = abc123
3. Agent B: update_task(T1) → writes to buffer
4. Agent B: commit() → SUCCESS, HEAD = def456
5. Agent A: update_task(T1) → writes to buffer
6. Agent A: commit() → CONFLICT (HEAD != base_commit)

Resolution:
- Agent A sees conflict details
- Can retry (auto-rebase on new HEAD)
- Or abort and let user decide
```

### Scenario 3: Corrupted WAL

```
Timeline:
1. Transaction T1 writes to WAL
2. Power failure corrupts WAL mid-write
3. System restarts

Recovery:
1. WAL reader finds checksum mismatch
2. Truncates WAL at last valid entry
3. Any transaction after truncation point is lost
4. System starts in consistent state
```

### Scenario 4: Git Repository Corruption

```
Timeline:
1. Git objects corrupted (disk error)
2. WAL references commit that doesn't exist

Recovery:
1. Integrity check detects missing commit
2. Falls back to last known good snapshot
3. Replays WAL entries after snapshot
4. Reports data loss (if any)
```

---

## Open Questions

1. **WAL Compaction**: When to compact the WAL? After N transactions? After M megabytes?

2. **Snapshot Retention**: How long to keep old snapshots? Git will garbage collect unreferenced commits.

3. **Lock Timeout**: What's a reasonable timeout for optimistic lock retries?

4. **Serializable vs Snapshot**: Do we need serializable isolation, or is snapshot sufficient for our use cases?

5. **Event Log Migration**: How to migrate existing event log to new versioned entity format?

---

## Conclusion

This architecture provides:

- **Atomicity**: Transactions commit or rollback completely
- **Consistency**: Version numbers and checksums ensure data integrity
- **Isolation**: Snapshot isolation via git commits
- **Durability**: WAL with fsync + git commits

By leveraging git as our storage layer, we get content-addressed storage, atomic commits, and history tracking without external dependencies. The transaction layer adds the safety guarantees needed for multi-agent concurrent access.

The design prioritizes **correctness over performance** - we'd rather be slow and correct than fast and corrupt. For our use case (Claude Agents with ~seconds between operations), this tradeoff is appropriate.
