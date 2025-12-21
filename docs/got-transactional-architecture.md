# GoT Transactional Architecture Design

**Status:** Draft
**Date:** 2025-12-21
**Task:** Design ACID-compliant transaction layer for Graph of Thought

---

## Executive Summary

This document describes a transactional architecture for the Graph of Thought (GoT) system that provides ACID guarantees for multi-agent concurrent access.

**Key Insight:** Git is for **synchronization between agents**, not for transaction isolation. The transaction layer must be self-contained with its own durability guarantees. Git becomes the "transport layer" for sharing state between agents working in different sessions or repositories.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   AGENT A (local)          AGENT B (local)         AGENT C     │
│   ┌─────────────┐          ┌─────────────┐      ┌─────────────┐│
│   │ Transaction │          │ Transaction │      │ Transaction ││
│   │   Layer     │          │   Layer     │      │   Layer     ││
│   └──────┬──────┘          └──────┬──────┘      └──────┬──────┘│
│          │                        │                    │        │
│          ▼                        ▼                    ▼        │
│   ┌─────────────┐          ┌─────────────┐      ┌─────────────┐│
│   │ Local State │          │ Local State │      │ Local State ││
│   │ (.got/)     │          │ (.got/)     │      │ (.got/)     ││
│   └──────┬──────┘          └──────┬──────┘      └──────┬──────┘│
│          │                        │                    │        │
│          └────────────┬───────────┴────────────────────┘        │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │   GIT SYNC      │  ← Periodic sync, NOT in       │
│              │   (push/pull)   │    transaction path            │
│              └─────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Architecture Layers](#architecture-layers)
3. [Transaction Layer](#transaction-layer)
4. [Storage Layer](#storage-layer)
5. [Sync Layer (Git)](#sync-layer-git)
6. [Conflict Resolution](#conflict-resolution)
7. [Recovery Procedures](#recovery-procedures)
8. [API Design](#api-design)
9. [User Stories](#user-stories)
10. [Edge Cases](#edge-cases)
11. [Implementation Plan](#implementation-plan)

---

## Design Principles

### 1. Separation of Concerns

| Layer | Responsibility | Git Involvement |
|-------|---------------|-----------------|
| Transaction | ACID guarantees, isolation, atomicity | **None** |
| Storage | Durability, checksums, versioning | **None** |
| Sync | Collaboration, sharing, history | **Yes** |

### 2. Local-First Architecture

Each agent works on **local state** with full ACID guarantees. Synchronization with other agents is a **separate, explicit operation** that happens outside the transaction path.

### 3. Explicit Over Implicit

- No automatic git commits during transactions
- No git commands in hot path
- Sync is user/agent-initiated
- Conflicts are surfaced, not hidden

### 4. Fail-Safe Defaults

- Uncommitted transactions are rolled back on crash
- Corrupted data is detected and rejected
- Sync conflicts block until resolved (no silent data loss)

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  GoTProjectManager, Tasks, Decisions, Edges                      │
│  - Business logic                                                │
│  - No knowledge of transactions or sync                          │
├─────────────────────────────────────────────────────────────────┤
│                    TRANSACTION LAYER                             │
│  TransactionManager, Transaction, WAL                            │
│  - Begin/Commit/Rollback                                         │
│  - ProcessLock for mutual exclusion                              │
│  - Snapshot isolation via version files                          │
│  - NO GIT COMMANDS                                               │
├─────────────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                                 │
│  VersionedStore, EntityFile, WALManager                          │
│  - Atomic file writes                                            │
│  - Checksums on all data                                         │
│  - Fsync for durability                                          │
│  - NO GIT COMMANDS                                               │
├─────────────────────────────────────────────────────────────────┤
│                    SYNC LAYER (SEPARATE PROCESS)                 │
│  SyncManager                                                     │
│  - git add/commit/push/pull                                      │
│  - Merge conflict detection                                      │
│  - Called explicitly, never automatically                        │
│  - Runs OUTSIDE transactions                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Transaction Layer

### ProcessLock (Already Implemented)

We already have `ProcessLock` for mutual exclusion. This prevents concurrent writes from multiple processes on the same machine.

```python
class ProcessLock:
    """File-based lock for process-safe operations."""
    # Already implemented in scripts/got_utils.py
    # Uses fcntl.flock() with PID tracking and stale detection
```

### Transaction Object

```python
@dataclass
class Transaction:
    id: str                          # TX-YYYYMMDD-HHMMSS-XXXX
    state: TransactionState          # ACTIVE, PREPARING, COMMITTED, ABORTED
    started_at: datetime
    snapshot_version: int            # Version at transaction start
    operations: List[Operation]      # Buffered operations
    write_set: Dict[str, Entity]     # Pending writes
    read_set: Dict[str, int]         # Keys read → versions (for conflict detection)

class TransactionState(Enum):
    ACTIVE = "active"
    PREPARING = "preparing"
    COMMITTED = "committed"
    ABORTED = "aborted"
    ROLLED_BACK = "rolled_back"
```

### Transaction Manager

```python
class TransactionManager:
    """
    Manages transactions with ACID guarantees.

    Thread-safety: Uses ProcessLock for all mutations.
    No git commands are executed by this class.
    """

    def __init__(self, got_dir: Path):
        self.got_dir = Path(got_dir)
        self.store = VersionedStore(got_dir / "entities")
        self.wal = WALManager(got_dir / "wal")
        self.lock = ProcessLock(got_dir / ".got.lock", reentrant=True)
        self.active_tx: Dict[str, Transaction] = {}

        # Run recovery on init
        self._recover_on_startup()

    def begin(self) -> Transaction:
        """Start a new transaction."""
        with self.lock:
            tx_id = generate_transaction_id()
            snapshot_version = self.store.current_version()

            tx = Transaction(
                id=tx_id,
                state=TransactionState.ACTIVE,
                started_at=datetime.now(),
                snapshot_version=snapshot_version,
                operations=[],
                write_set={},
                read_set={}
            )

            # Log to WAL (survives crash)
            self.wal.log_tx_begin(tx_id, snapshot_version)

            self.active_tx[tx_id] = tx
            return tx

    def read(self, tx: Transaction, entity_id: str) -> Optional[Entity]:
        """
        Read an entity within a transaction.

        Provides snapshot isolation:
        - Reads see state as of transaction start
        - Plus any writes made within this transaction
        """
        # Check our pending writes first
        if entity_id in tx.write_set:
            return tx.write_set[entity_id]

        # Read from snapshot version
        entity = self.store.read_at_version(entity_id, tx.snapshot_version)

        # Track what we read (for conflict detection)
        if entity:
            tx.read_set[entity_id] = entity.version

        return entity

    def write(self, tx: Transaction, entity: Entity) -> None:
        """
        Write an entity within a transaction.

        Writes are buffered until commit.
        Other transactions cannot see these changes.
        """
        if tx.state != TransactionState.ACTIVE:
            raise TransactionError(f"Transaction {tx.id} is not active")

        # Get old value for undo
        old_entity = self.read(tx, entity.id)

        # Log to WAL before buffering
        self.wal.log_write(tx.id, entity.id, old_entity, entity)

        # Buffer the write
        tx.write_set[entity.id] = entity
        tx.operations.append(WriteOp(
            entity_id=entity.id,
            old_version=old_entity.version if old_entity else 0,
            new_version=entity.version
        ))

    def commit(self, tx: Transaction) -> CommitResult:
        """
        Commit a transaction.

        Uses optimistic locking:
        1. Acquire lock
        2. Check for conflicts (version mismatches)
        3. Apply all writes atomically
        4. Release lock
        """
        with self.lock:
            if tx.state != TransactionState.ACTIVE:
                raise TransactionError(f"Transaction {tx.id} is not active")

            # Phase 1: PREPARE
            tx.state = TransactionState.PREPARING
            self.wal.log_tx_prepare(tx.id)

            # Phase 2: VALIDATE (optimistic lock check)
            conflicts = self._detect_conflicts(tx)
            if conflicts:
                tx.state = TransactionState.ABORTED
                self.wal.log_tx_abort(tx.id, "conflict")
                del self.active_tx[tx.id]
                return CommitResult(
                    success=False,
                    reason="conflict",
                    conflicts=conflicts
                )

            # Phase 3: APPLY (atomic write)
            try:
                new_version = self.store.apply_writes(tx.write_set)

                # Phase 4: FINALIZE
                tx.state = TransactionState.COMMITTED
                self.wal.log_tx_commit(tx.id, new_version)

                del self.active_tx[tx.id]

                return CommitResult(
                    success=True,
                    version=new_version
                )

            except Exception as e:
                tx.state = TransactionState.ABORTED
                self.wal.log_tx_abort(tx.id, str(e))
                raise

    def rollback(self, tx: Transaction, reason: str = "explicit") -> None:
        """Abort a transaction and discard all changes."""
        with self.lock:
            if tx.state == TransactionState.COMMITTED:
                raise TransactionError("Cannot rollback committed transaction")

            self.wal.log_tx_rollback(tx.id, reason)
            tx.write_set.clear()
            tx.operations.clear()
            tx.state = TransactionState.ROLLED_BACK

            if tx.id in self.active_tx:
                del self.active_tx[tx.id]

    def _detect_conflicts(self, tx: Transaction) -> List[Conflict]:
        """Detect conflicts between transaction and current state."""
        conflicts = []

        for entity_id, entity in tx.write_set.items():
            current = self.store.read(entity_id)

            if current is None:
                # Creating new entity, check it doesn't exist now
                if self.store.exists(entity_id):
                    conflicts.append(Conflict(
                        type=ConflictType.CREATE_EXISTS,
                        entity_id=entity_id,
                        message=f"Entity {entity_id} was created by another transaction"
                    ))
            else:
                # Updating existing, check version hasn't changed
                expected_version = tx.read_set.get(entity_id, 0)
                if current.version != expected_version:
                    conflicts.append(Conflict(
                        type=ConflictType.VERSION_MISMATCH,
                        entity_id=entity_id,
                        expected_version=expected_version,
                        actual_version=current.version,
                        message=f"Entity {entity_id} was modified by another transaction"
                    ))

        return conflicts

    def _recover_on_startup(self) -> None:
        """Recover from crash by replaying/rolling back incomplete transactions."""
        incomplete = self.wal.get_incomplete_transactions()

        for tx_record in incomplete:
            if tx_record.state == "PREPARING":
                # Crash during commit - rollback
                self.wal.log_tx_rollback(tx_record.id, "crash_recovery")
                logger.info(f"Rolled back incomplete transaction: {tx_record.id}")
            elif tx_record.state == "ACTIVE":
                # Crash during transaction - discard (writes were buffered)
                self.wal.log_tx_rollback(tx_record.id, "crash_recovery")
                logger.info(f"Discarded active transaction: {tx_record.id}")
```

---

## Storage Layer

### Versioned Store

```python
class VersionedStore:
    """
    File-based storage with versioning and checksums.

    Each entity is stored as a JSON file with:
    - Version number (monotonic)
    - Checksum (SHA256)
    - Timestamp

    The store maintains a global version counter that increments
    on every successful commit.
    """

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.store_dir / "_version.json"
        self._current_version = self._load_version()

    def current_version(self) -> int:
        """Get current global version."""
        return self._current_version

    def read(self, entity_id: str) -> Optional[Entity]:
        """Read current version of an entity."""
        path = self._entity_path(entity_id)
        if not path.exists():
            return None

        data = self._read_and_verify(path)
        return Entity.from_dict(data)

    def read_at_version(self, entity_id: str, version: int) -> Optional[Entity]:
        """
        Read entity as it was at a specific global version.

        Uses version history stored with each entity.
        """
        entity = self.read(entity_id)
        if entity is None:
            return None

        # If entity was created after the snapshot, it doesn't exist
        if entity.created_at_version > version:
            return None

        # If entity hasn't changed since snapshot, return current
        if entity.version <= version:
            return entity

        # Need to find historical version
        history_path = self._history_path(entity_id)
        if history_path.exists():
            history = json.loads(history_path.read_text())
            for entry in reversed(history):
                if entry['version'] <= version:
                    return Entity.from_dict(entry['data'])

        return None

    def apply_writes(self, write_set: Dict[str, Entity]) -> int:
        """
        Atomically apply a set of writes.

        Uses atomic file operations:
        1. Write to temp files
        2. Fsync all temp files
        3. Rename temp files to final (atomic on POSIX)
        4. Update version counter
        5. Fsync version file
        """
        if not write_set:
            return self._current_version

        new_version = self._current_version + 1
        temp_files = []

        try:
            # Phase 1: Write to temp files
            for entity_id, entity in write_set.items():
                entity.version = new_version
                entity.updated_at = datetime.now().isoformat()
                entity.checksum = entity.compute_checksum()

                # Save current to history before overwriting
                self._save_to_history(entity_id)

                temp_path = self._entity_path(entity_id).with_suffix('.tmp')
                self._write_with_checksum(temp_path, entity.to_dict())
                temp_files.append((temp_path, self._entity_path(entity_id)))

            # Phase 2: Fsync all temp files
            for temp_path, _ in temp_files:
                self._fsync_file(temp_path)

            # Phase 3: Atomic rename
            for temp_path, final_path in temp_files:
                temp_path.rename(final_path)

            # Phase 4: Update version
            self._current_version = new_version
            self._save_version()

            return new_version

        except Exception:
            # Cleanup temp files on failure
            for temp_path, _ in temp_files:
                if temp_path.exists():
                    temp_path.unlink()
            raise

    def _write_with_checksum(self, path: Path, data: dict) -> None:
        """Write JSON with embedded checksum."""
        content = json.dumps(data, indent=2, sort_keys=True)
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

        wrapper = {
            '_checksum': checksum,
            '_written_at': datetime.now().isoformat(),
            'data': data
        }

        path.write_text(json.dumps(wrapper, indent=2))

    def _read_and_verify(self, path: Path) -> dict:
        """Read JSON and verify checksum."""
        wrapper = json.loads(path.read_text())

        content = json.dumps(wrapper['data'], indent=2, sort_keys=True)
        expected = hashlib.sha256(content.encode()).hexdigest()[:16]

        if wrapper['_checksum'] != expected:
            raise CorruptionError(f"Checksum mismatch in {path}")

        return wrapper['data']

    def _fsync_file(self, path: Path) -> None:
        """Ensure file is durably written to disk."""
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
```

### WAL Manager

```python
class WALManager:
    """
    Write-Ahead Log for crash recovery.

    All operations are logged BEFORE they are applied.
    On crash, incomplete transactions can be rolled back.
    """

    def __init__(self, wal_dir: Path):
        self.wal_dir = Path(wal_dir)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self.wal_dir / "current.wal"
        self.sequence = self._load_sequence()

    def log(self, tx_id: str, operation: str, data: Dict) -> int:
        """Append entry to WAL with fsync."""
        self.sequence += 1

        entry = {
            'seq': self.sequence,
            'ts': datetime.now().isoformat(),
            'tx': tx_id,
            'op': operation,
            'data': data
        }

        # Add checksum
        content = json.dumps(entry, sort_keys=True)
        entry['checksum'] = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Append with fsync
        line = json.dumps(entry) + '\n'

        with open(self.current_file, 'a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        return self.sequence

    def log_tx_begin(self, tx_id: str, snapshot_version: int) -> int:
        return self.log(tx_id, "TX_BEGIN", {'snapshot': snapshot_version})

    def log_write(self, tx_id: str, entity_id: str, old: Entity, new: Entity) -> int:
        return self.log(tx_id, "WRITE", {
            'entity_id': entity_id,
            'old_version': old.version if old else 0,
            'new_version': new.version
        })

    def log_tx_prepare(self, tx_id: str) -> int:
        return self.log(tx_id, "TX_PREPARE", {})

    def log_tx_commit(self, tx_id: str, version: int) -> int:
        return self.log(tx_id, "TX_COMMIT", {'version': version})

    def log_tx_abort(self, tx_id: str, reason: str) -> int:
        return self.log(tx_id, "TX_ABORT", {'reason': reason})

    def log_tx_rollback(self, tx_id: str, reason: str) -> int:
        return self.log(tx_id, "TX_ROLLBACK", {'reason': reason})

    def get_incomplete_transactions(self) -> List[WALTransaction]:
        """Find transactions that didn't complete (for recovery)."""
        transactions = {}

        for entry in self._read_all():
            tx_id = entry['tx']
            op = entry['op']

            if tx_id not in transactions:
                transactions[tx_id] = WALTransaction(tx_id)

            tx = transactions[tx_id]

            if op == 'TX_BEGIN':
                tx.state = 'ACTIVE'
            elif op == 'TX_PREPARE':
                tx.state = 'PREPARING'
            elif op in ('TX_COMMIT', 'TX_ABORT', 'TX_ROLLBACK'):
                tx.state = 'COMPLETE'

        # Return incomplete ones
        return [tx for tx in transactions.values() if tx.state != 'COMPLETE']

    def truncate(self) -> None:
        """Truncate WAL after successful checkpoint."""
        # Archive current WAL
        if self.current_file.exists():
            archive_name = f"wal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            self.current_file.rename(self.wal_dir / "archive" / archive_name)

        # Reset sequence
        self.sequence = 0
        self._save_sequence()
```

---

## Sync Layer (Git)

### Separation from Transactions

**Critical:** The sync layer is completely separate from the transaction layer. Git operations NEVER occur during a transaction.

```python
class SyncManager:
    """
    Handles synchronization with git.

    This class is ONLY called explicitly by agents when they want to
    sync with others. It is NEVER called automatically during transactions.

    Workflow:
    1. Agent completes local work (multiple transactions)
    2. Agent explicitly calls sync.push() to share
    3. Agent explicitly calls sync.pull() to get others' changes
    4. Conflicts are detected and must be resolved before continuing
    """

    def __init__(self, repo_dir: Path, got_dir: Path):
        self.repo_dir = Path(repo_dir)
        self.got_dir = Path(got_dir)
        self.tx_manager = TransactionManager(got_dir)

    def push(self, message: str = None) -> PushResult:
        """
        Push local changes to git.

        Prerequisites:
        - No active transactions
        - All local changes committed (to transaction layer)

        Steps:
        1. Verify no active transactions
        2. Stage .got/ directory
        3. Git commit
        4. Git push
        """
        # Verify no active transactions
        if self.tx_manager.active_tx:
            raise SyncError("Cannot push with active transactions")

        # Stage and commit
        subprocess.run(
            ["git", "add", str(self.got_dir)],
            cwd=self.repo_dir,
            check=True
        )

        if message is None:
            message = f"GoT sync: {datetime.now().isoformat()}"

        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            raise SyncError(f"Git commit failed: {result.stderr}")

        # Push
        result = subprocess.run(
            ["git", "push"],
            cwd=self.repo_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if "rejected" in result.stderr:
                return PushResult(
                    success=False,
                    reason="rejected",
                    message="Remote has changes. Pull first."
                )
            raise SyncError(f"Git push failed: {result.stderr}")

        return PushResult(success=True)

    def pull(self) -> PullResult:
        """
        Pull remote changes.

        Prerequisites:
        - No active transactions

        Steps:
        1. Verify no active transactions
        2. Git fetch
        3. Check for conflicts
        4. If no conflicts: git merge
        5. If conflicts: return conflict info (don't auto-resolve)
        """
        # Verify no active transactions
        if self.tx_manager.active_tx:
            raise SyncError("Cannot pull with active transactions")

        # Fetch
        subprocess.run(
            ["git", "fetch"],
            cwd=self.repo_dir,
            check=True
        )

        # Check for conflicts
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "FETCH_HEAD"],
            cwd=self.repo_dir,
            capture_output=True,
            text=True
        )

        changed_files = [f for f in result.stdout.strip().split('\n') if f]
        got_conflicts = [f for f in changed_files if f.startswith('.got/')]

        if not got_conflicts:
            # No conflicts in .got/, safe to merge
            subprocess.run(
                ["git", "merge", "FETCH_HEAD"],
                cwd=self.repo_dir,
                check=True
            )
            return PullResult(success=True, merged_files=changed_files)

        # Check for actual content conflicts vs just different versions
        conflicts = self._detect_content_conflicts(got_conflicts)

        if conflicts:
            return PullResult(
                success=False,
                reason="conflict",
                conflicts=conflicts,
                message="Conflicts detected. Resolve before continuing."
            )

        # No real conflicts, can merge
        subprocess.run(
            ["git", "merge", "FETCH_HEAD"],
            cwd=self.repo_dir,
            check=True
        )

        # Reload transaction manager to pick up new state
        self.tx_manager._recover_on_startup()

        return PullResult(success=True, merged_files=changed_files)

    def _detect_content_conflicts(self, files: List[str]) -> List[SyncConflict]:
        """Detect actual content conflicts in entity files."""
        conflicts = []

        for file in files:
            if not file.endswith('.json'):
                continue

            # Get our version and their version
            ours = self._read_at_ref(file, "HEAD")
            theirs = self._read_at_ref(file, "FETCH_HEAD")

            if ours is None and theirs is None:
                continue

            if ours is None:
                # They created, we don't have - no conflict
                continue

            if theirs is None:
                # We have, they deleted - potential conflict
                conflicts.append(SyncConflict(
                    file=file,
                    type=SyncConflictType.DELETE_MODIFY,
                    ours=ours,
                    theirs=None
                ))
                continue

            # Both have versions - check if they're the same
            if ours == theirs:
                continue  # Identical, no conflict

            # Check if one is strictly newer
            ours_data = json.loads(ours)
            theirs_data = json.loads(theirs)

            ours_version = ours_data.get('data', {}).get('version', 0)
            theirs_version = theirs_data.get('data', {}).get('version', 0)

            if ours_version > theirs_version:
                # Ours is newer, keep ours (their change is stale)
                continue
            elif theirs_version > ours_version:
                # Theirs is newer, take theirs
                continue
            else:
                # Same version but different content - true conflict
                conflicts.append(SyncConflict(
                    file=file,
                    type=SyncConflictType.MODIFY_MODIFY,
                    ours=ours,
                    theirs=theirs
                ))

        return conflicts
```

### Conflict Resolution

```python
class ConflictResolver:
    """Resolve sync conflicts between agents."""

    def resolve(self, conflict: SyncConflict, strategy: str) -> None:
        """
        Resolve a conflict with the specified strategy.

        Strategies:
        - "ours": Keep our version
        - "theirs": Keep their version
        - "merge": Attempt automatic merge
        - "manual": Write conflict markers for manual resolution
        """
        if strategy == "ours":
            # Keep our file as-is
            pass

        elif strategy == "theirs":
            # Replace our file with theirs
            content = self.sync._read_at_ref(conflict.file, "FETCH_HEAD")
            Path(conflict.file).write_text(content)

        elif strategy == "merge":
            merged = self._attempt_merge(conflict)
            if merged.success:
                Path(conflict.file).write_text(merged.content)
            else:
                raise ConflictError(f"Cannot auto-merge {conflict.file}: {merged.reason}")

        elif strategy == "manual":
            self._write_conflict_markers(conflict)

    def _attempt_merge(self, conflict: SyncConflict) -> MergeResult:
        """Attempt to merge two versions of an entity."""
        ours = json.loads(conflict.ours)['data']
        theirs = json.loads(conflict.theirs)['data']

        # Find common base (if available from git)
        base = self._get_common_base(conflict.file)

        if base:
            base_data = json.loads(base)['data']
            return self._three_way_merge(base_data, ours, theirs)
        else:
            return self._two_way_merge(ours, theirs)

    def _three_way_merge(self, base: dict, ours: dict, theirs: dict) -> MergeResult:
        """Three-way merge using common ancestor."""
        result = {}
        all_keys = set(base.keys()) | set(ours.keys()) | set(theirs.keys())

        for key in all_keys:
            base_val = base.get(key)
            ours_val = ours.get(key)
            theirs_val = theirs.get(key)

            if ours_val == theirs_val:
                # Both made same change (or no change)
                result[key] = ours_val
            elif ours_val == base_val:
                # We didn't change, they did - take theirs
                result[key] = theirs_val
            elif theirs_val == base_val:
                # They didn't change, we did - take ours
                result[key] = ours_val
            else:
                # Both changed differently - conflict
                return MergeResult(
                    success=False,
                    reason=f"Conflicting changes to field '{key}'"
                )

        return MergeResult(success=True, data=result)
```

---

## Recovery Procedures

### Startup Recovery

```python
def startup_recovery(got_dir: Path) -> RecoveryReport:
    """
    Run on every startup to ensure consistent state.

    This is idempotent - safe to run multiple times.
    """
    report = RecoveryReport()

    # 1. Verify storage integrity
    store = VersionedStore(got_dir / "entities")
    corruption = store.verify_integrity()
    if corruption:
        report.add_error("Storage corruption detected", corruption)
        raise CorruptionError(corruption)

    # 2. Recover incomplete transactions from WAL
    wal = WALManager(got_dir / "wal")
    incomplete = wal.get_incomplete_transactions()

    for tx in incomplete:
        if tx.state in ('ACTIVE', 'PREPARING'):
            # Roll back incomplete transaction
            wal.log_tx_rollback(tx.id, "crash_recovery")
            report.add_rolled_back(tx.id)

    # 3. Truncate WAL if all transactions complete
    if not incomplete:
        wal.truncate()

    return report
```

### Corruption Recovery

```python
def recover_from_corruption(got_dir: Path) -> None:
    """
    Attempt to recover from corruption.

    Strategy:
    1. Try to recover from WAL
    2. If WAL is corrupted, try to recover from git history
    3. If git is corrupted, restore from last known good backup
    """
    store = VersionedStore(got_dir / "entities")
    wal = WALManager(got_dir / "wal")

    # Try WAL recovery
    try:
        wal.verify_integrity()
        wal.replay_to_store(store)
        return
    except CorruptionError:
        logger.warning("WAL corrupted, trying git recovery")

    # Try git recovery
    try:
        # Find last good commit
        result = subprocess.run(
            ["git", "log", "--oneline", "-20", "--", str(got_dir)],
            capture_output=True,
            text=True
        )

        for line in result.stdout.strip().split('\n'):
            commit_sha = line.split()[0]
            if verify_commit_integrity(commit_sha, got_dir):
                restore_from_commit(commit_sha, got_dir)
                return

        raise CorruptionError("No valid commit found in recent history")

    except Exception as e:
        logger.error(f"Git recovery failed: {e}")
        raise CorruptionError("Cannot recover. Restore from backup.")
```

---

## API Design

### High-Level API

```python
class GoTTransactionalManager:
    """
    Main API for transactional GoT operations.

    Usage:
        manager = GoTTransactionalManager(".got")

        # Explicit transaction
        with manager.transaction() as tx:
            task = tx.create_task("Implement feature X")
            tx.add_dependency(task.id, other_task_id)
            # Auto-commit on success, auto-rollback on exception

        # Sync with others (separate from transactions)
        manager.sync.pull()
        manager.sync.push("Completed feature X")
    """

    def __init__(self, got_dir: str = ".got"):
        self.got_dir = Path(got_dir)
        self.tx_manager = TransactionManager(self.got_dir)
        self.sync = SyncManager(Path.cwd(), self.got_dir)

    @contextmanager
    def transaction(self) -> Generator[TransactionContext, None, None]:
        """Start a transaction with auto-commit/rollback."""
        tx = self.tx_manager.begin()
        ctx = TransactionContext(self.tx_manager, tx)

        try:
            yield ctx
            result = self.tx_manager.commit(tx)

            if not result.success:
                raise ConflictError(result.conflicts)

        except Exception:
            self.tx_manager.rollback(tx)
            raise

    def read_only(self) -> ReadOnlyContext:
        """Get read-only access to current state."""
        return ReadOnlyContext(self.tx_manager.store)
```

### Transaction Context

```python
class TransactionContext:
    """Operations available within a transaction."""

    def __init__(self, tx_manager: TransactionManager, tx: Transaction):
        self._tx_manager = tx_manager
        self._tx = tx

    def create_task(self, title: str, **kwargs) -> Task:
        task_id = generate_task_id()
        task = Task(
            id=task_id,
            version=1,
            title=title,
            status="pending",
            created_at=datetime.now().isoformat(),
            **kwargs
        )
        self._tx_manager.write(self._tx, task.to_entity())
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        entity = self._tx_manager.read(self._tx, f"tasks/{task_id}")
        return Task.from_entity(entity) if entity else None

    def update_task(self, task_id: str, **changes) -> Task:
        task = self.get_task(task_id)
        if task is None:
            raise NotFoundError(f"Task {task_id} not found")

        updated = task.copy(
            version=task.version + 1,
            **changes
        )
        self._tx_manager.write(self._tx, updated.to_entity())
        return updated

    def delete_task(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            raise NotFoundError(f"Task {task_id} not found")

        # Mark as deleted (soft delete)
        deleted = task.copy(
            version=task.version + 1,
            deleted=True,
            deleted_at=datetime.now().isoformat()
        )
        self._tx_manager.write(self._tx, deleted.to_entity())

    # Similar methods for decisions, edges, etc.
```

---

## User Stories

### Story 1: Single Agent Creates Tasks

```
AS an agent
I WANT to create multiple tasks atomically
SO THAT either all are created or none are

GIVEN I start a transaction
WHEN I create Task A, Task B, and Task C
AND I commit the transaction
THEN all three tasks exist
AND they all have the same version number

GIVEN I start a transaction
WHEN I create Task A, Task B
AND an error occurs creating Task C
THEN no tasks are created
AND the system state is unchanged
```

### Story 2: Two Agents Work Concurrently

```
AS two agents working on the same repository
I WANT my changes to not corrupt the other's work
SO THAT we can work in parallel safely

GIVEN Agent A and Agent B both have local copies
WHEN Agent A creates Task X and commits
AND Agent B creates Task Y and commits
AND Agent A pushes
AND Agent B pushes
THEN Agent B gets a "pull first" error
WHEN Agent B pulls
THEN Agent B sees both Task X and Task Y
WHEN Agent B pushes
THEN both agents can pull and see both tasks
```

### Story 3: Conflicting Updates

```
AS two agents updating the same task
I WANT conflicts to be detected and surfaced
SO THAT I can resolve them explicitly

GIVEN Task T exists with version 1
WHEN Agent A updates T.status to "in_progress"
AND Agent B updates T.status to "completed"
AND Agent A pushes first
THEN Agent B's push fails with "pull first"
WHEN Agent B pulls
THEN Agent B sees a conflict on Task T
AND Agent B must resolve (choose ours, theirs, or merge)
AND Agent B can then push
```

### Story 4: Crash Recovery

```
AS an agent that crashed mid-transaction
I WANT my incomplete work to be rolled back
SO THAT the system is not left in an inconsistent state

GIVEN I start a transaction
AND I create Task A
AND I update Task B
WHEN my process crashes before commit
AND I restart
THEN Task A does not exist
AND Task B is unchanged
AND I can start a new transaction
```

### Story 5: Sub-Agent Coordination

```
AS a director agent coordinating sub-agents
I WANT sub-agents to work independently
SO THAT they don't block each other

GIVEN I spawn 3 sub-agents to work on different features
WHEN each sub-agent creates tasks in their own transactions
THEN sub-agents don't wait for each other
WHEN I collect their results
AND push all changes
THEN all tasks from all sub-agents are persisted
```

---

## Edge Cases

### Edge Case Matrix

| Scenario | Expected Behavior | Handling |
|----------|-------------------|----------|
| Two agents create same task ID | Second commit fails with conflict | Use UUIDs to prevent |
| Power loss during WAL write | Partial entry detected by checksum | Truncate at last valid entry |
| Power loss during commit | PREPARING state in WAL | Rollback on recovery |
| Corrupted entity file | Checksum mismatch detected | Recover from history |
| Corrupted WAL | Checksum mismatch detected | Recover from git |
| Git push rejected | Return error, don't retry | User must pull first |
| Git merge conflict | Don't auto-resolve | Surface to user |
| Process killed with lock held | PID no longer exists | Steal lock after timeout |
| Same process double-acquires lock | Reentrant lock allows | Already implemented |
| Transaction timeout | No implicit timeout | Agent must manage |
| Very large transaction | Memory pressure | Limit write set size |
| Read-only agent | No transactions needed | Use read_only() API |

### Handling Each Edge Case

```python
# 1. UUID collision prevention
def generate_task_id() -> str:
    """Generate collision-resistant task ID."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    random_part = secrets.token_hex(4)  # 8 chars, 32 bits of entropy
    return f"T-{timestamp}-{random_part}"

# 2. WAL truncation on corruption
def find_last_valid_wal_entry(wal_path: Path) -> int:
    """Find byte offset of last valid entry."""
    valid_offset = 0

    with open(wal_path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break

            try:
                entry = json.loads(line)
                verify_checksum(entry)
                valid_offset = f.tell()
            except (json.JSONDecodeError, ChecksumError):
                break

    return valid_offset

# 3. Commit atomicity
def atomic_commit(store: VersionedStore, writes: Dict) -> int:
    """Atomic commit using temp files and rename."""
    # Write to .tmp files
    # Fsync .tmp files
    # Rename .tmp to final (atomic on POSIX)
    # Fsync directory
    # Update version file
    # Fsync version file
    pass

# 4. Entity recovery from git
def recover_entity_from_git(entity_id: str, got_dir: Path) -> Entity:
    """Recover entity from git history."""
    # Binary search through git history for last valid version
    commits = get_git_commits_for_file(f"{got_dir}/entities/{entity_id}.json")

    for commit in commits:
        try:
            content = git_show(commit, f"{got_dir}/entities/{entity_id}.json")
            entity = parse_and_verify(content)
            return entity
        except (CorruptionError, json.JSONDecodeError):
            continue

    raise CorruptionError(f"Cannot recover {entity_id}")

# 5. Conflict detection
def detect_write_write_conflict(ours: Entity, theirs: Entity, base: Entity) -> bool:
    """Detect if there's a true conflict."""
    if ours.version == theirs.version and ours.checksum != theirs.checksum:
        return True  # Same version, different content = conflict
    return False
```

### Testing Edge Cases

```python
class TestEdgeCases:
    """Tests for edge cases. All must pass."""

    def test_concurrent_creates_different_ids(self):
        """Two agents creating tasks get different IDs."""
        id1 = generate_task_id()
        id2 = generate_task_id()
        assert id1 != id2

    def test_wal_corruption_detected(self):
        """Corrupted WAL entry is detected."""
        wal = WALManager(tmp_path)
        wal.log("tx1", "OP", {"key": "value"})

        # Corrupt the file
        with open(wal.current_file, 'r+') as f:
            f.seek(10)
            f.write("GARBAGE")

        with pytest.raises(CorruptionError):
            wal.verify_integrity()

    def test_crash_during_commit_rolls_back(self):
        """Incomplete commit is rolled back on restart."""
        tx = manager.begin()
        manager.write(tx, task.to_entity())

        # Simulate crash by writing PREPARE to WAL but not COMMIT
        manager.wal.log_tx_prepare(tx.id)

        # "Restart"
        new_manager = TransactionManager(got_dir)

        # Task should not exist
        assert new_manager.store.read(task.id) is None

    def test_checksum_mismatch_rejected(self):
        """File with wrong checksum is rejected."""
        store = VersionedStore(tmp_path)

        # Write valid entity
        store.write(task.to_entity())

        # Corrupt the file
        path = store._entity_path(task.id)
        data = json.loads(path.read_text())
        data['_checksum'] = 'invalid'
        path.write_text(json.dumps(data))

        with pytest.raises(CorruptionError):
            store.read(task.id)

    def test_stale_lock_recovered(self):
        """Lock from dead process is recovered."""
        lock = ProcessLock(tmp_path / "test.lock")

        # Write lock file with non-existent PID
        lock.lock_path.write_text("99999999\n1234567890")

        # Should be able to acquire
        assert lock.acquire(timeout=0.1) is True
```

---

## Implementation Plan

### Phase 1: Storage Layer (3-4 days)
- [ ] VersionedStore with checksums
- [ ] Atomic file operations (write-tmp-fsync-rename)
- [ ] Version history per entity
- [ ] Integrity verification
- [ ] Unit tests for all operations

### Phase 2: WAL Layer (2-3 days)
- [ ] WALManager with checksummed entries
- [ ] Fsync on every write
- [ ] Recovery: find incomplete transactions
- [ ] Truncation and archiving
- [ ] Unit tests including corruption scenarios

### Phase 3: Transaction Layer (3-4 days)
- [ ] Transaction object and state machine
- [ ] TransactionManager: begin/commit/rollback
- [ ] Snapshot isolation (read at version)
- [ ] Conflict detection
- [ ] Integration with ProcessLock
- [ ] Unit tests for all transaction scenarios

### Phase 4: High-Level API (2-3 days)
- [ ] GoTTransactionalManager
- [ ] TransactionContext with typed methods
- [ ] ReadOnlyContext
- [ ] Context manager support
- [ ] Integration tests

### Phase 5: Sync Layer (2-3 days)
- [ ] SyncManager: push/pull
- [ ] Conflict detection
- [ ] ConflictResolver strategies
- [ ] Integration tests with git

### Phase 6: Migration (1-2 days)
- [ ] Migrate existing .got/ data to new format
- [ ] Backward compatibility (if needed)
- [ ] Verification that old data is preserved

---

## Appendix: File Format

### Entity File Format

```json
{
  "_checksum": "a1b2c3d4e5f6",
  "_written_at": "2025-12-21T12:00:00",
  "data": {
    "id": "T-20251221-120000-a1b2",
    "version": 3,
    "type": "task",
    "title": "Implement feature X",
    "status": "in_progress",
    "priority": "high",
    "created_at": "2025-12-21T10:00:00",
    "updated_at": "2025-12-21T12:00:00"
  }
}
```

### WAL Entry Format

```json
{"seq":1,"ts":"2025-12-21T12:00:00","tx":"TX-20251221-120000-a1b2","op":"TX_BEGIN","data":{"snapshot":42},"checksum":"f1e2d3c4"}
{"seq":2,"ts":"2025-12-21T12:00:01","tx":"TX-20251221-120000-a1b2","op":"WRITE","data":{"entity_id":"T-123","old_version":0,"new_version":1},"checksum":"b5a4c3d2"}
{"seq":3,"ts":"2025-12-21T12:00:02","tx":"TX-20251221-120000-a1b2","op":"TX_PREPARE","data":{},"checksum":"e1f2a3b4"}
{"seq":4,"ts":"2025-12-21T12:00:03","tx":"TX-20251221-120000-a1b2","op":"TX_COMMIT","data":{"version":43},"checksum":"c5d6e7f8"}
```

### Version File Format

```json
{
  "current_version": 43,
  "last_commit_at": "2025-12-21T12:00:03",
  "checksum": "a1b2c3d4"
}
```
