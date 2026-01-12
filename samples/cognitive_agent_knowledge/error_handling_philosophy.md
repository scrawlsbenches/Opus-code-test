# Error Handling Philosophy

This document teaches AI agents about effective error handling strategies. These patterns are battle-tested in this codebase and represent lessons learned through real production issues.

## Fail Fast vs Fail Safe: When to Use Each

### Fail Fast: Detect Problems Early

**When to use**: During development, in constructors, during validation, when corruption would be worse than stopping.

**Principle**: If something is wrong, stop immediately with a clear error. Don't let bad state propagate.

**Example from this codebase** (`cortical/cdg/errors.py`):

```python
class ValidationError(CDGError):
    """
    Raised when entity data fails validation.

    This includes schema validation failures, constraint violations,
    and invalid field values.
    """

    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        errors: Optional[List[str]] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.entity_type = entity_type
        self.errors = errors or []
```

**When validation fails, we fail fast**:
- Constructor receives invalid data? Raise `ValidationError` immediately.
- Transaction state is wrong? Raise `TransactionError` before proceeding.
- Checksum mismatch? Raise `CorruptionError` and refuse to use the data.

### Fail Safe: Keep Running When Possible

**When to use**: In recovery code, during graceful degradation, when partial success is better than total failure.

**Principle**: Contain failures. Let what can work continue working.

**Example from this codebase** (`cortical/cdg/recovery.py`):

```python
def verify_store_integrity(self) -> List[str]:
    """Verify all entities have valid checksums."""
    corrupted = []

    for entity_file in entity_files:
        try:
            self.store._read_and_verify(entity_file)
        except FileNotFoundError:
            # File was deleted between glob and read (race condition)
            # This is fine - another process may have cleaned it up
            logger.debug(
                "Entity file %s vanished during integrity check (race condition)",
                entity_file.name
            )
        except (CorruptionError, json.JSONDecodeError, KeyError) as e:
            # Record but don't stop - keep checking other entities
            corrupted.append(entity_id)
            logger.debug(
                "Corrupted entity detected: %s - %s: %s",
                entity_id, type(e).__name__, e
            )

    return corrupted  # Report what's broken, don't crash
```

### Decision Matrix

| Situation | Strategy | Reason |
|-----------|----------|--------|
| Invalid input to public API | Fail Fast | Bad data should not enter system |
| Recovery checking file integrity | Fail Safe | Report all issues, don't stop at first |
| Transaction commit | Fail Fast | Partial commits cause corruption |
| Reading optional config | Fail Safe | Use defaults if config missing |
| Concurrent access race | Fail Safe | Log and continue if state is valid |
| Security violation detected | Fail Fast | Never allow security bypass |

## Error Recovery Strategies

### Strategy 1: Retry with Backoff

**When to use**: Transient failures (network, temporary locks, rate limits).

**Pattern**:
```python
def read_with_retry(path, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return read_file(path)
        except IOError as e:
            if attempt == max_attempts - 1:
                raise  # Final attempt failed
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
```

### Strategy 2: Fallback Values

**When to use**: Optional features, non-critical data, graceful degradation.

**Example from this codebase** (`cortical/cdg/wal.py`):

```python
def _load_sequence(self) -> int:
    """Load sequence counter from disk."""
    try:
        if self.seq_file.exists():
            with open(self.seq_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('seq', 0)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # File was deleted or corrupted between exists() and read
        # This is fine - start from 0
        logger.debug(
            "Sequence file unavailable, starting from 0: %s: %s",
            type(e).__name__, e
        )
    return 0  # Fallback: start fresh
```

### Strategy 3: Graceful Degradation

**When to use**: When partial functionality is better than none.

**Example from this codebase** (`cortical/cdg/recovery.py`):

```python
def needs_index_recovery(self) -> bool:
    """Check if indexes need to be rebuilt."""
    if self._index_manager is None:
        return False  # No index manager = no index recovery needed

    try:
        return self._index_manager.needs_rebuild()
    except Exception as e:
        logger.warning(
            "Index manager needs_rebuild check failed: %s: %s",
            type(e).__name__, e
        )
        return True  # Assume rebuild needed on error (safer)
```

### Strategy 4: Compensating Transaction

**When to use**: Multi-step operations that partially succeeded.

**Example**: If step 3 of 5 fails, undo steps 1-2 before returning error.

```python
def multi_step_operation():
    completed_steps = []
    try:
        step1_result = do_step1()
        completed_steps.append(('step1', step1_result))

        step2_result = do_step2()
        completed_steps.append(('step2', step2_result))

        step3_result = do_step3()  # This might fail
        completed_steps.append(('step3', step3_result))

    except OperationError as e:
        # Undo in reverse order
        for step_name, result in reversed(completed_steps):
            undo_step(step_name, result)
        raise
```

## Logging Errors: Debugging vs Alerting

### Log Levels and Their Purpose

| Level | Purpose | When to Use |
|-------|---------|-------------|
| `DEBUG` | Development troubleshooting | Race conditions, skipped items, detailed flow |
| `INFO` | Normal operations | Recovery actions, successful repairs |
| `WARNING` | Potential problems | Corrupted data found, integrity issues |
| `ERROR` | Failures requiring attention | Recovery failures, critical operations failed |

### Real Examples from This Codebase

**DEBUG - Expected race conditions** (`cortical/cdg/recovery.py`):
```python
except FileNotFoundError:
    # File was deleted between glob and read (race condition)
    # This is fine - another process may have cleaned it up
    logger.debug(
        "Entity file %s vanished during integrity check (race condition)",
        entity_file.name
    )
```

**INFO - Successful recovery actions**:
```python
logger.info("Adopted orphaned entity: %s", entity_id)
logger.info(
    "Reconstructed entity %s from WAL (TX %s)",
    entity_id, tx_id
)
```

**WARNING - Data issues that don't stop execution** (`cortical/cdg/wal.py`):
```python
logger.warning(
    "Skipping corrupted WAL entry at line %d: %s",
    line_num, e
)
```

**ERROR - Failures that need investigation**:
```python
logger.error(
    "Write failed after WAL commit for TX %s: %s. "
    "Recovery will need to redo writes from WAL.",
    tx.id, e
)
```

### Key Principle: Log for the Audience

- **DEBUG**: For developers stepping through code
- **INFO**: For operators monitoring normal flow
- **WARNING**: For operators who need to check something
- **ERROR**: For on-call engineers who need to act

## User-Facing vs Internal Errors

### Internal Errors: Rich Context

Internal errors should carry debugging context:

```python
class CDGError(Exception):
    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.message = message
        self.context = context  # entity_id, tx_id, versions, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to JSON-serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context
        }
```

**Usage**:
```python
raise ConflictError(
    "Entity modified by concurrent transaction",
    tx_id="TX-20251231-120000-abc123",
    entity_id="E-001",
    read_version=5,
    current_version=6
)
```

### User-Facing Errors: Clear Messages

User-facing errors should be:
- **Clear**: What went wrong in plain language
- **Actionable**: What the user can do about it
- **Safe**: No internal details that could be security risks

**Example transformation**:
```python
# Internal error (rich context)
raise ValidationError(
    "Invalid edge_type 'INVALID'",
    edge_type="INVALID",
    valid_types=list(VALID_EDGE_TYPES)
)

# User-facing message (clear, actionable)
"Error: 'INVALID' is not a valid edge type.
Valid types are: BLOCKS, DEPENDS_ON, RELATES_TO"
```

## Transaction Safety and Rollback

### The Transaction Contract

From `cortical/cdg/transaction_manager.py`:

```python
class CDGTransactionManager:
    """
    Manages transactions with ACID guarantees for CDG.

    - Atomicity: All writes in a TX succeed or all fail
    - Consistency: Checksums verify data integrity
    - Isolation: Snapshot isolation via versioning
    - Durability: WAL + fsync before commit (when enabled)
    """
```

### Rollback Protocol

```python
def rollback(self, tx: Transaction, reason: str = "explicit") -> None:
    """
    Rollback transaction.
    Discards write_set, sets state to ROLLED_BACK, logs to WAL.
    """
    if not tx.can_rollback():
        raise TransactionError(
            f"Transaction {tx.id} cannot rollback (state: {tx.state.value})"
        )

    # 1. Discard writes (never applied to store)
    tx.write_set.clear()

    # 2. Update state
    tx.state = TransactionState.ROLLED_BACK

    # 3. Log for audit trail
    if self.wal:
        self.wal.log_tx_rollback(tx.id, reason)

    # 4. Remove from active transactions
    self._active_tx.pop(tx.id, None)
```

### Conflict Detection

```python
def _detect_conflicts(self, tx: Transaction) -> List[Conflict]:
    """Detect version conflicts using optimistic locking."""
    conflicts = []

    for entity_id in tx.write_set:
        if entity_id in tx.read_set:
            expected_version = tx.read_set[entity_id]
            current_entity = self.store.read(entity_id)
            actual_version = current_entity.version if current_entity else 0

            if expected_version != actual_version:
                conflicts.append(Conflict(
                    entity_id=entity_id,
                    expected_version=expected_version,
                    actual_version=actual_version,
                    conflict_type="version_mismatch",
                    message=f"Expected version {expected_version}, got {actual_version}"
                ))

    return conflicts
```

## The WAL Pattern for Crash Recovery

### Write-Ahead Log Principle

**Core idea**: Log all operations BEFORE they are applied. On crash, replay the log to recover state.

From `cortical/cdg/wal.py`:

```python
class CDGWALManager:
    """
    Write-Ahead Log for crash recovery.

    All operations are logged BEFORE they are applied.
    On crash, incomplete transactions can be rolled back.
    Uses JSONL format (one JSON object per line).
    """
```

### WAL-First Commit Protocol

From `cortical/cdg/transaction_manager.py`:

```python
def commit(self, tx: Transaction) -> CommitResult:
    """
    Commit transaction with WAL-first durability.

    WAL-First Protocol (ACID-compliant):
    1. Acquire lock
    2. Set state to PREPARING, log TX_PREPARE
    3. Detect conflicts (version mismatch)
    4. If conflict: abort, return failure
    5. Log TX_COMMIT to WAL (commit decision is now durable)
    6. Fsync WAL (ensures commit survives crash)
    7. Apply writes to entity files (can be redone from WAL on crash)
    8. Set state to COMMITTED
    9. Release lock

    Key insight: Once TX_COMMIT is in WAL and fsynced, the transaction
    IS committed. Entity files are a materialized view that can be
    reconstructed from WAL on recovery.
    """
```

### Recovery from WAL

```python
def reconstruct_entities_from_wal(self) -> List[str]:
    """
    Reconstruct missing/corrupted entities from WAL for committed transactions.

    This handles the case where a crash occurred after TX_COMMIT was written
    to WAL (transaction is committed) but before entity files were written
    to disk.
    """
```

### Why WAL Works

1. **Durability**: WAL is fsynced before entity changes
2. **Atomicity**: Either TX_COMMIT is in WAL (committed) or not (rollback)
3. **Idempotency**: Replay can be run multiple times safely
4. **Auditability**: Complete history of all operations

## Error Handling Anti-Patterns

### Anti-Pattern 1: Swallowing Exceptions

**Bad**:
```python
try:
    process_data(data)
except Exception:
    pass  # Silently ignore ALL errors
```

**Why it's bad**: Hides bugs, makes debugging impossible, data may be silently corrupted.

**Good**:
```python
try:
    process_data(data)
except ValueError as e:
    logger.warning("Invalid data format, skipping: %s", e)
    return None  # Explicit handling with logging
```

### Anti-Pattern 2: Catch-All Handlers

**Bad**:
```python
try:
    complex_operation()
except Exception as e:
    print(f"Error: {e}")
    return False
```

**Why it's bad**: Catches system errors (KeyboardInterrupt, MemoryError) that shouldn't be caught.

**Good**:
```python
try:
    complex_operation()
except (ValueError, IOError, OperationError) as e:
    logger.error("Operation failed: %s", e)
    return OperationResult(success=False, error=str(e))
```

### Anti-Pattern 3: Bare Raise Without Context

**Bad**:
```python
try:
    do_something()
except Exception:
    raise  # No context added
```

**Good** (when you need to add context):
```python
try:
    do_something()
except ValueError as e:
    raise ValidationError(
        f"Failed processing entity {entity_id}: {e}",
        entity_id=entity_id,
        original_error=str(e)
    ) from e
```

### Anti-Pattern 4: Error Handling in Wrong Layer

**Bad**: Low-level code makes UI decisions:
```python
def save_entity(entity):
    try:
        write_to_disk(entity)
    except IOError:
        print("Save failed! Please try again.")  # Wrong layer!
```

**Good**: Propagate errors to appropriate layer:
```python
def save_entity(entity):
    try:
        write_to_disk(entity)
    except IOError as e:
        raise StorageError(
            "Failed to write entity file",
            path=str(entity_path),
            operation="write"
        ) from e
```

### Anti-Pattern 5: Ignoring Return Values

**Bad**:
```python
commit(transaction)  # Ignoring CommitResult
proceed_with_success_assumption()
```

**Good**:
```python
result = commit(transaction)
if not result.success:
    if result.conflicts:
        handle_conflicts(result.conflicts)
    else:
        raise TransactionError(result.reason)
```

## Error Hierarchy Design

### Build a Clear Hierarchy

From this codebase (`cortical/cdg/errors.py`):

```
CDGError (base)
    ValidationError - Schema/constraint violations
    CorruptionError - Data integrity failures
    TransactionError - Transaction lifecycle errors
        ConflictError - Optimistic locking conflicts
    PartitionError - Partition routing/management errors
    StorageError - Low-level storage failures
```

### Why Hierarchy Matters

```python
# Catch all CDG errors (broad)
try:
    store.write(entity)
except CDGError as e:
    logger.error(f"CDG operation failed: {e}")

# Catch specific transaction conflicts (narrow)
try:
    manager.commit(tx)
except ConflictError as e:
    # Handle optimistic locking conflict specifically
    retry_with_fresh_read(e.entity_id)
except TransactionError as e:
    # Handle other transaction errors
    logger.error(f"Transaction failed: {e}")
```

## Summary: The Error Handling Checklist

1. **Classify the error**: Is this fail-fast or fail-safe territory?
2. **Choose recovery strategy**: Retry? Fallback? Degrade? Compensate?
3. **Log appropriately**: DEBUG for developers, ERROR for operators
4. **Preserve context**: Include entity IDs, transaction IDs, versions
5. **Handle at right layer**: Low-level code raises, high-level code decides
6. **Test error paths**: Error handling code needs tests too
7. **Document error contracts**: What errors can this function raise?
