# Research: Foreign Key Patterns for Eventually-Consistent and Multi-Tier Transactional Systems

**Date:** 2025-12-23
**Purpose:** Practical patterns for file-based Python systems without external dependencies
**Audience:** Systems like Cortical Text Processor with GoT (Graph of Thought) and entity relationships

---

## Executive Summary

Foreign key management spans three distinct worlds:
1. **Hard References (Strong Consistency)**: Traditional CASCADE DELETE, immediate integrity
2. **Soft References (Eventual Consistency)**: Orphan-tolerant, async cleanup, distributed-friendly
3. **Hybrid Approaches**: Multi-tier systems with consistency boundaries between layers

This document explores practical patterns implementable in pure Python with no external dependencies, focusing on real-world trade-offs.

---

## Part 1: Understanding Reference Lifecycles

### The Four Phases of Entity Lifecycle

Every entity goes through these phases:

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐
│ RESERVED/    │ -> │ MATERIALIZED     │ -> │ REFERENCED      │ -> │ RETIRED/    │
│ PLACEHOLDER  │    │ (fully created)  │    │ (in use by      │    │ TOMBSTONE   │
│              │    │                  │    │  other entities)│    │             │
└──────────────┘    └──────────────────┘    └─────────────────┘    └─────────────┘

Patterns:        Patterns:                  Patterns:              Patterns:
- ID generation  - Indexing                 - Edge creation        - Soft delete
- Sequence       - Finalization             - Reference counting   - Tombstones
- UUID           - Data validation          - FK constraints       - Delayed cleanup
```

**Key insight:** Most FK problems occur when assuming entities jump from creation to deletion without considering intermediate states.

---

## Part 2: Placeholder/Reservation Patterns

### Pattern 1: Sequential IDs with Reservation Table

**Use case:** Task/decision creation in GoT system where ID must exist before entity fully materializes.

**How it works:**
```python
# File: cortical/got/reservation.py
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

class IDReservation:
    """Reserve IDs before entity creation for distributed coordination."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.reservations_file = self.storage_dir / "reservations.json"
        self.counter_file = self.storage_dir / "id_counter.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def reserve_id(self, entity_type: str, agent_id: str) -> str:
        """
        Reserve an ID without creating the entity.

        Returns:
            ID in format: T-20251223-093045-a1b2c3d4
        """
        # Atomic increment counter
        counter = self._load_counter()
        next_seq = counter.get(entity_type, 0) + 1
        counter[entity_type] = next_seq
        self._save_counter(counter)

        # Create reservation entry
        reserved_id = self._generate_id(entity_type, next_seq)
        reservation = {
            'id': reserved_id,
            'entity_type': entity_type,
            'reserved_by': agent_id,
            'reserved_at': datetime.now(timezone.utc).isoformat(),
            'status': 'reserved',  # 'reserved' -> 'materialized' -> 'retired'
            'expiry': (datetime.now(timezone.utc).timestamp() + 3600)  # 1 hour TTL
        }

        reservations = self._load_reservations()
        reservations[reserved_id] = reservation
        self._save_reservations(reservations)

        return reserved_id

    def materialize(self, reserved_id: str, entity_data: Dict[str, Any]) -> bool:
        """
        Convert reservation to materialized entity.

        Returns:
            True if successful, False if reservation expired
        """
        reservations = self._load_reservations()
        if reserved_id not in reservations:
            raise ValueError(f"Reservation {reserved_id} not found")

        reservation = reservations[reserved_id]

        # Check expiry
        if datetime.now(timezone.utc).timestamp() > reservation['expiry']:
            reservation['status'] = 'expired'
            self._save_reservations(reservations)
            return False

        # Mark as materialized
        reservation['status'] = 'materialized'
        reservation['materialized_at'] = datetime.now(timezone.utc).isoformat()
        reservation['data_hash'] = self._hash(entity_data)
        self._save_reservations(reservations)

        return True

    def get_status(self, reserved_id: str) -> Optional[str]:
        """Get current status of reservation: 'reserved', 'materialized', 'expired', 'retired'"""
        reservations = self._load_reservations()
        if reserved_id in reservations:
            return reservations[reserved_id]['status']
        return None

    def _generate_id(self, entity_type: str, seq: int) -> str:
        """Generate ID: T-YYYYMMDD-HHMMSS-{seq:08x}"""
        now = datetime.now(timezone.utc)
        date_part = now.strftime('%Y%m%d')
        time_part = now.strftime('%H%M%S')
        hex_part = f"{seq:08x}"
        prefix = entity_type[0].upper()  # T for Task, D for Decision, etc.
        return f"{prefix}-{date_part}-{time_part}-{hex_part}"

    def _load_reservations(self) -> Dict[str, Any]:
        if self.reservations_file.exists():
            return json.loads(self.reservations_file.read_text())
        return {}

    def _save_reservations(self, data: Dict[str, Any]) -> None:
        self.reservations_file.write_text(json.dumps(data, indent=2))

    def _load_counter(self) -> Dict[str, int]:
        if self.counter_file.exists():
            return json.loads(self.counter_file.read_text())
        return {}

    def _save_counter(self, data: Dict[str, int]) -> None:
        self.counter_file.write_text(json.dumps(data, indent=2))

    def _hash(self, data: Dict[str, Any]) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
```

**Advantages:**
- Decentralized: No central sequence server needed
- Distributed-safe: Each agent can reserve multiple IDs in one op
- Audit trail: Can trace which agent created which IDs
- Recoverable: Expired reservations can be cleaned up

**Disadvantages:**
- Requires cleanup of expired reservations
- ID gaps if materialization fails
- Storage grows with reservation count

**Use in GoT:**
```python
# When task is requested but not yet full created
reservations = IDReservation(Path(".got/reservations"))
task_id = reservations.reserve_id("task", agent_id="main-agent")

# Doing expensive work...
# ... fetch all dependencies, validate ...

# Only after validation succeeds:
if reservations.materialize(task_id, {"title": "...", "status": "pending"}):
    # Now safe to create references to task_id from other entities
    sprint.tasks.append(task_id)
else:
    # Reservation expired, need new ID
    task_id = reservations.reserve_id("task", agent_id="main-agent")
```

---

### Pattern 2: UUID with Deterministic Generation

**Use case:** When sequential IDs aren't needed, but deterministic reproducibility is required.

**How it works:**
```python
# File: cortical/utils/deterministic_uuid.py
import hashlib
from uuid import UUID
from typing import Any, Dict

class DeterministicUUID:
    """
    Generate UUIDs deterministically based on content.

    Useful for:
    - Deduplication (same content = same UUID)
    - Reproducible testing
    - Cross-system synchronization
    """

    NAMESPACE_CORTICAL = UUID('550e8400-e29b-41d4-a716-446655440000')

    @staticmethod
    def from_content(entity_type: str, content: Dict[str, Any]) -> str:
        """
        Generate deterministic UUID based on entity type and content.

        Examples:
            uuid = DeterministicUUID.from_content(
                "task",
                {"title": "Fix authentication", "priority": "high"}
            )
            # Same input always produces same UUID
        """
        # Canonical JSON representation for deterministic hashing
        content_json = json.dumps(content, sort_keys=True, separators=(',', ':'))

        # Create deterministic namespace UUID
        data = f"{entity_type}:{content_json}".encode()
        hash_obj = hashlib.sha256(data)

        # Convert to UUID using hash
        uuid_int = int.from_bytes(hash_obj.digest()[:16], 'big')
        uuid_obj = UUID(int=uuid_int)
        return str(uuid_obj)

    @staticmethod
    def from_parent(parent_id: str, child_type: str, child_index: int) -> str:
        """
        Generate child UUID from parent ID.

        Enables derived IDs without central coordination.

        Example:
            sprint_id = "S-001"
            task_1_id = DeterministicUUID.from_parent(sprint_id, "task", 0)
            task_2_id = DeterministicUUID.from_parent(sprint_id, "task", 1)
        """
        data = f"{parent_id}:{child_type}:{child_index}".encode()
        hash_obj = hashlib.sha256(data)
        uuid_int = int.from_bytes(hash_obj.digest()[:16], 'big')
        uuid_obj = UUID(int=uuid_int)
        return str(uuid_obj)
```

**Advantages:**
- No central counter needed
- IDs are reproducible: same input = same ID
- Good for deduplication
- Works across systems

**Disadvantages:**
- Not human-readable
- ID doesn't encode timestamp
- Can't extract creation time
- Hash collision risk (though negligible)

---

### Pattern 3: Composite Key with Status Tracking

**Use case:** Tracking entity lifecycle through immutable creation with mutable status.

**How it works:**
```python
# In cortical/got/types.py

@dataclass
class Task(Entity):
    """Task with explicit lifecycle states."""

    # Identity (immutable after creation)
    id: str
    entity_type: str = "task"

    # Lifecycle states (mutable)
    status: str = "pending"  # pending -> in_progress -> completed/blocked/cancelled

    # Reservation tracking
    reservation_id: str = ""  # ID of original reservation if applicable
    materialized_at: str = ""  # ISO timestamp when entity became materialized

    def is_materialized(self) -> bool:
        """Entity is fully created and usable for references."""
        return self.status != "reserved" and self.materialized_at != ""

    def can_be_referenced(self) -> bool:
        """Entity exists and is safe to reference."""
        return self.is_materialized() and self.status != "cancelled"
```

**Benefits:**
- Explicit lifecycle: code is self-documenting
- Single source of truth: status field
- Validation: `assert task.can_be_referenced()` before creating edges

---

## Part 3: Soft References vs Hard References

### Pattern 1: Hard References (CASCADE DELETE)

**When to use:** Entity hierarchy where child cannot exist without parent.

**Example:** Sprint contains Tasks → if sprint deleted, tasks become orphaned.

**Implementation:**
```python
# cortical/got/hard_references.py
from typing import Set
from dataclasses import dataclass

@dataclass
class CascadeDeleteConfig:
    """Configuration for cascade deletion behavior."""

    # If parent deleted:
    orphan_handling: str = "delete"  # "delete" | "reparent" | "error"

    # Before actually deleting child:
    validate_no_grandchildren: bool = True
    preserve_audit_trail: bool = True

class HardReferenceManager:
    """Manage references where deletion must be coordinated."""

    def delete_entity_cascade(
        self,
        entity_id: str,
        got_manager,
        config: CascadeDeleteConfig = CascadeDeleteConfig()
    ) -> Dict[str, int]:
        """
        Delete entity and all dependent entities (CASCADE).

        Returns:
            {'deleted_entities': 5, 'deleted_edges': 12}
        """
        deleted_count = 0
        deleted_edges = 0

        # Step 1: Find all entities that reference this one
        dependents = got_manager.find_entities_by_edge_type(
            edge_type="DEPENDS_ON",
            target_id=entity_id
        )

        # Step 2: Delete dependents first (depth-first)
        for dependent in dependents:
            result = self.delete_entity_cascade(
                dependent.id,
                got_manager,
                config
            )
            deleted_count += result['deleted_entities']
            deleted_edges += result['deleted_edges']

        # Step 3: Delete all edges pointing to/from entity
        edges = got_manager.get_edges_for_entity(entity_id)
        deleted_edges += len(edges)
        for edge in edges:
            got_manager.delete_edge(edge.id)

        # Step 4: Delete entity itself
        got_manager.delete_entity(entity_id)
        deleted_count += 1

        return {
            'deleted_entities': deleted_count,
            'deleted_edges': deleted_edges
        }

    def check_can_delete_safe(
        self,
        entity_id: str,
        got_manager
    ) -> tuple[bool, str]:
        """
        Check if entity can be safely deleted without cascade.

        Returns:
            (can_delete, reason)
        """
        # Find entities that MUST exist if this entity exists
        critical_dependents = got_manager.find_entities_by_edge_type(
            edge_type="REQUIRES",  # Strong requirement
            target_id=entity_id
        )

        if critical_dependents:
            names = [e.title for e in critical_dependents[:3]]
            return False, f"Required by: {', '.join(names)}"

        return True, "Safe to delete"
```

**Advantages:**
- Simple: deletion is automatic
- Consistent: no orphaned references
- Predictable: developers know what gets deleted

**Disadvantages:**
- Expensive: deep cascades are slow
- Fragile: one mistake cascades everywhere
- Dangerous: irreversible

**Safe usage pattern:**
```python
# Always check before cascade delete
can_delete, reason = manager.check_can_delete_safe(task_id, got_mgr)
if not can_delete:
    print(f"Cannot delete: {reason}")
    return

# Then delete with logging
result = manager.delete_entity_cascade(task_id, got_mgr)
print(f"Deleted {result['deleted_entities']} entities, {result['deleted_edges']} edges")
```

---

### Pattern 2: Soft References (Orphan-Tolerant)

**When to use:** Distributed systems, eventually consistent stores, references across consistency boundaries.

**Example:** Task references Decision → decision can be deleted without breaking task.

**Implementation:**
```python
# cortical/got/soft_references.py
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

@dataclass
class BrokenReference:
    """Record of a broken reference for auditing."""

    source_id: str
    target_id: str
    edge_type: str
    discovered_at: str  # ISO timestamp
    resolution_status: str = "pending"  # pending | resolved | ignored

class SoftReferenceManager:
    """Manage references that can tolerate missing targets."""

    def __init__(self, orphan_dir: Path):
        self.orphan_dir = Path(orphan_dir)
        self.orphan_dir.mkdir(parents=True, exist_ok=True)
        self.broken_refs_file = self.orphan_dir / "broken_references.jsonl"

    def dereference(
        self,
        entity_id: str,
        got_manager,
        strict: bool = False
    ) -> Optional[dict]:
        """
        Get entity by ID, handling missing entities gracefully.

        Args:
            entity_id: ID to look up
            got_manager: Manager with entities
            strict: Raise error if not found (vs return None)

        Returns:
            Entity dict or None if missing and not strict

        Raises:
            EntityNotFoundError if strict=True and not found
        """
        entity = got_manager.get_entity(entity_id)

        if entity is None:
            if strict:
                raise EntityNotFoundError(f"Entity {entity_id} not found")
            return None

        return entity

    def validate_references(self, entity_id: str, got_manager) -> List[BrokenReference]:
        """
        Check if all references from entity point to valid targets.

        Returns:
            List of broken references found
        """
        broken = []
        entity = got_manager.get_entity(entity_id)

        if not entity:
            return broken

        # Get all edges from this entity
        edges = got_manager.get_edges_from_entity(entity_id)

        for edge in edges:
            # Check if target exists
            target = got_manager.get_entity(edge.target_id)

            if target is None:
                broken_ref = BrokenReference(
                    source_id=entity_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    discovered_at=datetime.now(timezone.utc).isoformat()
                )
                broken.append(broken_ref)
                self._log_broken_reference(broken_ref)

        return broken

    def auto_cleanup_edges_to_deleted_entities(
        self,
        got_manager,
        older_than_days: int = 7
    ) -> int:
        """
        Clean up edges pointing to deleted entities (async cleanup).

        Returns:
            Number of edges cleaned up
        """
        cutoff_time = (datetime.now(timezone.utc).timestamp()
                      - (older_than_days * 86400))

        edges_to_delete = []

        # Scan all edges
        for entity in got_manager.list_entities():
            for edge in got_manager.get_edges_from_entity(entity.id):
                # Check if target exists
                if got_manager.get_entity(edge.target_id) is None:
                    # Check edge is old enough
                    if edge.created_at < cutoff_time:
                        edges_to_delete.append(edge.id)

        # Delete old edges to missing entities
        for edge_id in edges_to_delete:
            got_manager.delete_edge(edge_id)

        return len(edges_to_delete)

    def _log_broken_reference(self, broken_ref: BrokenReference) -> None:
        """Append broken reference to log."""
        with open(self.broken_refs_file, 'a') as f:
            f.write(json.dumps({
                'source_id': broken_ref.source_id,
                'target_id': broken_ref.target_id,
                'edge_type': broken_ref.edge_type,
                'discovered_at': broken_ref.discovered_at
            }) + '\n')

class EntityNotFoundError(Exception):
    """Raised when entity referenced by soft reference is missing."""
    pass
```

**Usage pattern:**
```python
# Soft reference: safe if target is deleted
soft_refs = SoftReferenceManager(Path(".got/orphans"))

# Check if reference is valid (doesn't error if missing)
decision = soft_refs.dereference(decision_id, got_mgr, strict=False)
if decision is None:
    print(f"Decision {decision_id} was deleted, using fallback")
    decision = FALLBACK_DECISION

# Periodic cleanup of dead edges
cleaned = soft_refs.auto_cleanup_edges_to_deleted_entities(
    got_mgr,
    older_than_days=7
)
print(f"Cleaned up {cleaned} edges to deleted entities")
```

**Advantages:**
- Resilient to deletions
- No cascading failures
- Decoupled entities
- Eventually consistent

**Disadvantages:**
- Requires fallbacks for missing data
- Orphaned edges accumulate
- Debugging is harder
- Need periodic cleanup

---

### Pattern 3: Hybrid Model (Selective Soft References)

**When to use:** Mixed consistency requirements within one system.

**Example:**
- Task REQUIRES Sprint (hard) → if sprint deleted, task must be cleaned
- Task MENTIONS Decision (soft) → if decision deleted, task still valid

**Implementation:**
```python
# cortical/got/reference_strength.py
from enum import Enum
from typing import List

class ReferenceStrength(Enum):
    """Semantic strength of a reference."""

    HARD = "hard"      # Edge.edge_type in [REQUIRES, DEPENDS_ON, PART_OF]
    SOFT = "soft"      # Edge.edge_type in [MENTIONS, INFLUENCES, REFERENCES]
    CONDITIONAL = "conditional"  # Depends on business rules

class ReferencePolicy:
    """Policy for how to handle references during deletion."""

    @staticmethod
    def get_strength(edge_type: str) -> ReferenceStrength:
        """Determine reference strength from edge type."""

        hard_types = {
            "REQUIRES", "DEPENDS_ON", "PART_OF", "CONTAINS"
        }

        soft_types = {
            "MENTIONS", "INFLUENCES", "REFERENCES", "SUGGESTS"
        }

        if edge_type in hard_types:
            return ReferenceStrength.HARD
        elif edge_type in soft_types:
            return ReferenceStrength.SOFT
        else:
            return ReferenceStrength.CONDITIONAL

    @staticmethod
    def can_delete_with_references(
        entity_id: str,
        got_manager
    ) -> tuple[bool, List[str]]:
        """
        Check if entity can be deleted given current references.

        Returns:
            (can_delete, blocking_reasons)
        """
        reasons = []

        edges = got_manager.get_edges_from_entity(entity_id)

        # Find hard references TO this entity
        hard_references = [
            e for e in edges
            if e.direction == "incoming"
            and ReferencePolicy.get_strength(e.edge_type) == ReferenceStrength.HARD
        ]

        if hard_references:
            reasons.append(
                f"Has {len(hard_references)} hard incoming references "
                "(cannot delete)"
            )
            return False, reasons

        # Soft references are OK
        soft_references = [
            e for e in edges
            if e.direction == "incoming"
            and ReferencePolicy.get_strength(e.edge_type) == ReferenceStrength.SOFT
        ]

        if soft_references:
            reasons.append(
                f"Has {len(soft_references)} soft incoming references "
                "(will become orphaned)"
            )

        return True, reasons
```

**Benefits:**
- Clear semantics: code shows reference strength
- Scalable: soft refs don't cascade
- Safe: hard refs prevent bad deletions
- Auditable: policy is explicit

---

## Part 4: Two-Phase Commit Patterns

**Use case:** References across consistency boundaries (e.g., cross-module transactions).

### Pattern 1: Explicit Two-Phase Commit

**How it works:**
```python
# cortical/got/two_phase_commit.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

class TxPhase(Enum):
    """Two-phase commit phases."""
    INIT = "init"
    PREPARE = "prepare"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    ABORTED = "aborted"

@dataclass
class TxParticipant:
    """One participant in a distributed transaction."""

    participant_id: str
    prepare_status: Optional[str] = None  # None | 'yes' | 'no'
    prepare_error: str = ""
    commit_status: Optional[str] = None  # None | 'ok' | 'error'
    commit_error: str = ""

@dataclass
class CrossDomainTransaction:
    """
    Transaction coordinating references across domains.

    Example:
        - Domain 1: Create Task with ID T-123
        - Domain 2: Create Sprint and add reference to T-123

    Need both to succeed or both to fail (atomically).
    """

    tx_id: str
    phase: TxPhase = TxPhase.INIT

    # Domains involved (e.g., 'tasks', 'sprints', 'decisions')
    participants: Dict[str, TxParticipant] = field(default_factory=dict)

    # Operations to perform in each domain
    operations: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    prepared_at: str = ""
    committed_at: str = ""
    rolled_back_at: str = ""

class TwoPhaseCommitCoordinator:
    """Coordinate cross-domain references with 2PC."""

    def __init__(self, tx_log_dir: Path):
        self.tx_log_dir = Path(tx_log_dir)
        self.tx_log_dir.mkdir(parents=True, exist_ok=True)

    def begin_transaction(self, tx_id: str, participants: List[str]) -> CrossDomainTransaction:
        """Begin a new cross-domain transaction."""

        tx = CrossDomainTransaction(
            tx_id=tx_id,
            participants={pid: TxParticipant(pid) for pid in participants}
        )

        self._save_tx(tx)
        return tx

    def add_operation(
        self,
        tx_id: str,
        participant: str,
        operation: Dict[str, Any]
    ) -> None:
        """Add operation for a participant to perform."""

        tx = self._load_tx(tx_id)
        if participant not in tx.operations:
            tx.operations[participant] = []

        tx.operations[participant].append(operation)
        self._save_tx(tx)

    def prepare(self, tx_id: str, participant_results: Dict[str, bool]) -> bool:
        """
        Prepare phase: ask all participants if they can commit.

        Returns:
            True if all participants can commit
        """
        tx = self._load_tx(tx_id)
        tx.phase = TxPhase.PREPARE

        can_commit_all = True
        for participant, can_commit in participant_results.items():
            if participant not in tx.participants:
                raise ValueError(f"Unknown participant: {participant}")

            tx.participants[participant].prepare_status = 'yes' if can_commit else 'no'

            if not can_commit:
                can_commit_all = False

        tx.prepared_at = datetime.now(timezone.utc).isoformat()
        self._save_tx(tx)

        return can_commit_all

    def commit(self, tx_id: str, commit_results: Dict[str, bool]) -> bool:
        """
        Commit phase: perform all operations.

        All participants should have returned success in prepare().

        Returns:
            True if all participants successfully committed
        """
        tx = self._load_tx(tx_id)
        tx.phase = TxPhase.COMMIT

        success_all = True
        for participant, success in commit_results.items():
            if participant not in tx.participants:
                raise ValueError(f"Unknown participant: {participant}")

            tx.participants[participant].commit_status = 'ok' if success else 'error'

            if not success:
                success_all = False
                # Mark for rollback
                tx.phase = TxPhase.ROLLBACK

        tx.committed_at = datetime.now(timezone.utc).isoformat()
        self._save_tx(tx)

        return success_all

    def abort(self, tx_id: str, reason: str = "") -> None:
        """Abort transaction and trigger rollbacks."""

        tx = self._load_tx(tx_id)
        tx.phase = TxPhase.ABORTED
        tx.rolled_back_at = datetime.now(timezone.utc).isoformat()

        # Log abort reason in operation context
        if reason:
            for ops in tx.operations.values():
                for op in ops:
                    op['abort_reason'] = reason

        self._save_tx(tx)

    def get_unfinished_transactions(self) -> List[CrossDomainTransaction]:
        """Get all unfinished transactions (for recovery)."""

        unfinished = []
        for tx_file in self.tx_log_dir.glob("tx_*.json"):
            tx = self._load_tx_from_file(tx_file)
            if tx.phase not in [TxPhase.COMMIT]:  # Not successfully committed
                unfinished.append(tx)

        return unfinished

    def recover_transaction(self, tx: CrossDomainTransaction) -> Dict[str, Any]:
        """
        Recover a transaction that was interrupted.

        Returns:
            {'status': 'recovered' | 'rolled_back' | 'needs_manual_intervention'}
        """

        if tx.phase == TxPhase.PREPARE:
            # Prepare phase interrupted -> safe to abort
            return {'status': 'rolled_back', 'reason': 'Interrupted in prepare'}

        elif tx.phase == TxPhase.COMMIT:
            # Commit phase was executing
            # Check which participants actually committed
            # (requires querying them)
            return {
                'status': 'needs_manual_intervention',
                'reason': 'Commit phase interrupted, need to check participants'
            }

        return {'status': 'safe'}

    def _load_tx(self, tx_id: str) -> CrossDomainTransaction:
        """Load transaction from disk."""
        tx_file = self.tx_log_dir / f"tx_{tx_id}.json"
        return self._load_tx_from_file(tx_file)

    def _load_tx_from_file(self, path: Path) -> CrossDomainTransaction:
        """Load transaction from file."""
        data = json.loads(path.read_text())

        # Reconstruct transaction
        participants = {
            pid: TxParticipant(
                participant_id=pid,
                prepare_status=p.get('prepare_status'),
                prepare_error=p.get('prepare_error', ''),
                commit_status=p.get('commit_status'),
                commit_error=p.get('commit_error', '')
            )
            for pid, p in data.get('participants', {}).items()
        }

        return CrossDomainTransaction(
            tx_id=data['tx_id'],
            phase=TxPhase(data['phase']),
            participants=participants,
            operations=data.get('operations', {}),
            created_at=data.get('created_at', ''),
            prepared_at=data.get('prepared_at', ''),
            committed_at=data.get('committed_at', ''),
            rolled_back_at=data.get('rolled_back_at', '')
        )

    def _save_tx(self, tx: CrossDomainTransaction) -> None:
        """Save transaction to disk."""

        tx_file = self.tx_log_dir / f"tx_{tx.tx_id}.json"

        data = {
            'tx_id': tx.tx_id,
            'phase': tx.phase.value,
            'participants': {
                pid: {
                    'participant_id': p.participant_id,
                    'prepare_status': p.prepare_status,
                    'prepare_error': p.prepare_error,
                    'commit_status': p.commit_status,
                    'commit_error': p.commit_error
                }
                for pid, p in tx.participants.items()
            },
            'operations': tx.operations,
            'created_at': tx.created_at,
            'prepared_at': tx.prepared_at,
            'committed_at': tx.committed_at,
            'rolled_back_at': tx.rolled_back_at
        }

        tx_file.write_text(json.dumps(data, indent=2))
```

**Usage pattern:**
```python
coordinator = TwoPhaseCommitCoordinator(Path(".got/transactions"))

# Begin transaction: create task AND add to sprint atomically
tx = coordinator.begin_transaction(
    tx_id="TX-20251223-001",
    participants=['tasks', 'sprints']
)

# Prepare phase: check if operations can succeed
coordinator.add_operation(
    "TX-20251223-001",
    'tasks',
    {'op': 'create', 'title': 'Fix bug', 'priority': 'high'}
)

coordinator.add_operation(
    "TX-20251223-001",
    'sprints',
    {'op': 'add_task', 'sprint_id': 'S-001', 'task_id': 'T-NEW'}
)

# Ask participants: can you commit?
can_commit = coordinator.prepare(
    "TX-20251223-001",
    {
        'tasks': True,      # Task manager says yes
        'sprints': True     # Sprint manager says yes
    }
)

if not can_commit:
    coordinator.abort("TX-20251223-001", "Prepare phase failed")
else:
    # Commit: actually perform operations
    success = coordinator.commit(
        "TX-20251223-001",
        {
            'tasks': True,      # Actually created task
            'sprints': True     # Actually added reference
        }
    )

# Recovery: after crash, find unfinished transactions
unfinished = coordinator.get_unfinished_transactions()
for tx in unfinished:
    result = coordinator.recover_transaction(tx)
```

**Advantages:**
- Atomic cross-domain updates
- Recoverable after crash
- Explicit phases aid debugging
- All-or-nothing semantics

**Disadvantages:**
- Complex coordination
- Requires all participants online
- Blocking: coordinator waits
- Slower than optimistic approaches

---

## Part 5: Eventual Consistency FK Patterns (NoSQL Style)

**Use case:** DynamoDB/Cassandra-style systems where ACID transactions aren't available.

### Pattern 1: Denormalization with Version Vectors

**How it works:**
```python
# cortical/patterns/denormalization.py
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class VersionVector:
    """
    Clock for tracking causality in distributed systems.

    [replica1_clock: 5, replica2_clock: 3, replica3_clock: 7]

    Allows determining: which write happened first?
    """

    clocks: Dict[str, int] = field(default_factory=dict)

    def increment(self, replica_id: str) -> None:
        """Increment clock for this replica."""
        self.clocks[replica_id] = self.clocks.get(replica_id, 0) + 1

    def merge(self, other: "VersionVector") -> "VersionVector":
        """Merge two version vectors (for concurrent updates)."""
        merged = VersionVector()

        # Take max for each replica
        all_replicas = set(self.clocks.keys()) | set(other.clocks.keys())
        for replica in all_replicas:
            merged.clocks[replica] = max(
                self.clocks.get(replica, 0),
                other.clocks.get(replica, 0)
            )

        return merged

    def is_concurrent(self, other: "VersionVector") -> bool:
        """Check if two updates are concurrent (neither happened before)."""

        self_less = False
        other_less = False

        for replica in set(self.clocks.keys()) | set(other.clocks.keys()):
            if self.clocks.get(replica, 0) < other.clocks.get(replica, 0):
                self_less = True
            if self.clocks.get(replica, 0) > other.clocks.get(replica, 0):
                other_less = True

        return self_less and other_less


@dataclass
class DenormalizedEdge:
    """
    Edge with denormalized target data for availability.

    Pattern:
        Instead of:  Task -> {sprint_id: "S-1"}
        Store:       Task -> {sprint_id: "S-1", sprint_title: "Sprint 1", sprint_status: "active"}

    When target (Sprint) is eventually unavailable:
    - We can still display the denormalized data
    - We detect staleness and request fresh data
    - No broken references
    """

    from_id: str
    to_id: str
    edge_type: str

    # Denormalized target data
    target_snapshot: Dict[str, Any] = field(default_factory=dict)
    target_snapshot_version: int = 1
    snapshot_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def is_stale(self, max_age_days: int = 7) -> bool:
        """Check if denormalized data is old."""
        from datetime import timedelta, timezone, datetime

        snapshot_time = datetime.fromisoformat(
            self.snapshot_timestamp.replace('Z', '+00:00')
        )
        age_days = (datetime.now(timezone.utc) - snapshot_time).days

        return age_days > max_age_days

    def refresh_snapshot(self, current_target: Dict[str, Any]) -> None:
        """Update denormalized data from current target."""
        self.target_snapshot = {
            k: v for k, v in current_target.items()
            if k in ['id', 'title', 'status', 'priority']  # Only mutable fields
        }
        self.snapshot_timestamp = datetime.now(timezone.utc).isoformat()
        self.target_snapshot_version += 1


class DenormalizationStrategy:
    """Strategy for deciding what to denormalize."""

    @staticmethod
    def should_denormalize(edge_type: str) -> bool:
        """Edges that should have denormalized target data."""

        # High-value targets worth denormalizing
        important_edges = {
            "PART_OF",      # Task part of Sprint
            "DEPENDS_ON",   # Task depends on Decision
            "CONTAINS",     # Sprint contains Tasks
            "IMPLEMENTS"    # Code implements Decision
        }

        return edge_type in important_edges

    @staticmethod
    def get_denormalization_fields(entity_type: str) -> List[str]:
        """Fields to include in denormalized snapshot."""

        fields = {
            'task': ['id', 'title', 'status', 'priority', 'assigned_to'],
            'sprint': ['id', 'title', 'status', 'number', 'epic_id'],
            'decision': ['id', 'title', 'rationale', 'status'],
            'epic': ['id', 'title', 'status', 'phase']
        }

        return fields.get(entity_type, ['id', 'title', 'status'])
```

**Advantages:**
- High availability: works even if target deleted
- Reads don't require joins
- Detectable staleness
- Refreshable when available

**Disadvantages:**
- Data duplication
- Staleness requires monitoring
- Need refresh mechanism
- Consensus on refresh timing

---

### Pattern 2: Tombstone Markers with TTL

**How it works:**
```python
# cortical/patterns/tombstones.py
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

@dataclass
class Tombstone:
    """
    Marker indicating entity was deleted.

    Pattern:
        1. Entity is marked for deletion (status="tombstone")
        2. References have grace period to clean up
        3. After TTL, entity is permanently removed
    """

    entity_id: str
    entity_type: str
    deleted_at: str  # ISO timestamp
    ttl_seconds: int = 86400  # 24 hours default

    reason: str = ""  # Why was it deleted
    deleted_by: str = ""  # Who deleted it

    def is_expired(self) -> bool:
        """Check if TTL has passed."""
        deleted_time = datetime.fromisoformat(
            self.deleted_at.replace('Z', '+00:00')
        )
        expiry_time = deleted_time + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry_time

    def time_until_expiry(self) -> int:
        """Seconds until permanent deletion."""
        deleted_time = datetime.fromisoformat(
            self.deleted_at.replace('Z', '+00:00')
        )
        expiry_time = deleted_time + timedelta(seconds=self.ttl_seconds)
        remaining = (expiry_time - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))


class TombstoneManager:
    """Manage soft deletes with eventual permanent removal."""

    def __init__(self, tombstone_dir: Path):
        self.tombstone_dir = Path(tombstone_dir)
        self.tombstone_dir.mkdir(parents=True, exist_ok=True)
        self.tombstones_file = self.tombstone_dir / "tombstones.jsonl"

    def soft_delete_entity(
        self,
        entity_id: str,
        entity_type: str,
        ttl_seconds: int = 86400,
        reason: str = "",
        deleted_by: str = ""
    ) -> Tombstone:
        """
        Soft delete: mark entity as deleted but keep reference-ability.

        During TTL period:
        - References still see the entity (but marked as deleted)
        - Systems can clean up references
        - Entity can be un-deleted
        """

        tombstone = Tombstone(
            entity_id=entity_id,
            entity_type=entity_type,
            deleted_at=datetime.now(timezone.utc).isoformat(),
            ttl_seconds=ttl_seconds,
            reason=reason,
            deleted_by=deleted_by
        )

        self._append_tombstone(tombstone)
        return tombstone

    def get_tombstone(self, entity_id: str) -> dict | None:
        """Get tombstone if entity is deleted."""

        for line in self.tombstones_file.read_text().splitlines():
            data = json.loads(line)
            if data['entity_id'] == entity_id:
                tombstone = Tombstone(**data)

                if tombstone.is_expired():
                    return None  # TTL passed, entity is gone

                return {
                    'entity_id': entity_id,
                    'deleted_at': tombstone.deleted_at,
                    'reason': tombstone.reason,
                    'time_until_expiry': tombstone.time_until_expiry()
                }

        return None

    def is_deleted(self, entity_id: str) -> bool:
        """Check if entity is soft-deleted."""
        return self.get_tombstone(entity_id) is not None

    def undelete_entity(self, entity_id: str) -> bool:
        """Undelete entity during TTL period."""

        tombstone = self.get_tombstone(entity_id)
        if not tombstone:
            return False

        # Remove from tombstone log
        lines = self.tombstones_file.read_text().splitlines()
        lines = [
            line for line in lines
            if json.loads(line).get('entity_id') != entity_id
        ]
        self.tombstones_file.write_text('\n'.join(lines))

        return True

    def purge_expired_tombstones(self) -> int:
        """Permanently delete entities whose TTL expired."""

        lines = self.tombstones_file.read_text().splitlines()
        unexpired = []
        expired_count = 0

        for line in lines:
            data = json.loads(line)
            tombstone = Tombstone(**data)

            if not tombstone.is_expired():
                unexpired.append(line)
            else:
                expired_count += 1

        self.tombstones_file.write_text('\n'.join(unexpired))
        return expired_count

    def _append_tombstone(self, tombstone: Tombstone) -> None:
        """Append tombstone to log."""
        data = {
            'entity_id': tombstone.entity_id,
            'entity_type': tombstone.entity_type,
            'deleted_at': tombstone.deleted_at,
            'ttl_seconds': tombstone.ttl_seconds,
            'reason': tombstone.reason,
            'deleted_by': tombstone.deleted_by
        }

        with open(self.tombstones_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
```

**Usage pattern:**
```python
tombstone_mgr = TombstoneManager(Path(".got/tombstones"))

# User requests deletion
tombstone_mgr.soft_delete_entity(
    entity_id="T-001",
    entity_type="task",
    ttl_seconds=86400 * 7,  # 7 days grace period
    reason="Duplicate task",
    deleted_by="user-123"
)

# References can see the entity is deleted
if tombstone_mgr.is_deleted("T-001"):
    print("Task is deleted but recoverable for 7 days")

# After 7 days, permanent removal
tombstone_mgr.purge_expired_tombstones()

# Can undelete within TTL
tombstone_mgr.undelete_entity("T-001")
```

**Advantages:**
- Deletions are reversible
- References have grace period
- Clear timeline
- Auditable

**Disadvantages:**
- Storage accumulates
- Complex cleanup
- Zombies during TTL

---

## Part 6: Choosing the Right Pattern

### Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Pattern Selection Based on Requirements                                  │
├─────────────────┬──────────────────┬──────────────┬─────────────────────┤
│ Requirement     │ Use Pattern      │ Consistency  │ Implementation Cost │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ HARD FK:        │ CASCADE DELETE    │ Strong       │ Medium (recursive)  │
│ Strict parent-  │ (Phase 1)        │              │                     │
│ child relation  │                  │              │                     │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ SOFT FK:        │ Orphan-tolerant   │ Eventual     │ Low (async cleanup) │
│ Cross-domain    │ + Async cleanup   │              │                     │
│ or distributed  │ (Phase 2)        │              │                     │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ MIXED FK:       │ Hybrid model      │ Selective    │ High (policy-based) │
│ Some hard,      │ + Reference       │              │                     │
│ some soft refs  │ strength enum     │              │                     │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ ID RESERVATION: │ Sequential IDs    │ N/A          │ Medium (TTL mgmt)   │
│ Multi-agent     │ + Materialization │              │                     │
│ entity creation │                  │              │                     │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ CROSS-DOMAIN:   │ Two-phase commit  │ Strong       │ High (complex)      │
│ Atomic updates  │ (Phase 4)        │              │                     │
│ across domains  │                  │              │                     │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ HIGH AVAIL:     │ Denormalization   │ Eventual     │ Medium (refresh)    │
│ No joins,       │ + Version vectors │              │                     │
│ redundancy OK   │                  │              │                     │
├─────────────────┼──────────────────┼──────────────┼─────────────────────┤
│ REVERSIBLE      │ Tombstones + TTL  │ Eventual     │ Low (append-only)   │
│ DELETION:       │ (Phase 5)        │              │                     │
│ Recovery needed │                  │              │                     │
└─────────────────┴──────────────────┴──────────────┴─────────────────────┘
```

### For Cortical GoT System

**Recommended combination:**

```python
# cortical/got/reference_configuration.py

# Hard references: REQUIRES, DEPENDS_ON, PART_OF
# -> Use CASCADE DELETE
# -> Check can_delete_safe() before deletion
# Example: Task REQUIRES Sprint

# Soft references: MENTIONS, SUGGESTS, INFLUENCES
# -> Use Orphan-tolerant + Async cleanup
# -> Reference resolution with fallbacks
# Example: Task MENTIONS Decision (decision can disappear)

# ID reservations: For multi-agent task creation
# -> Use Sequential IDs + TTL
# -> Await materialization before referencing
# Example: Agent 1 creates task, Agent 2 adds to sprint

# Cross-domain: Task created AND added to Sprint atomically
# -> Use Two-phase commit
# -> Recover unfinished transactions on startup
# Example: New task + sprint update

REFERENCE_CONFIG = {
    'hard': {
        'edge_types': ['REQUIRES', 'DEPENDS_ON', 'PART_OF', 'CONTAINS'],
        'strategy': 'cascade_delete',
        'validation': 'check_can_delete_safe'
    },
    'soft': {
        'edge_types': ['MENTIONS', 'SUGGESTS', 'INFLUENCES', 'REFERENCES'],
        'strategy': 'orphan_tolerant',
        'cleanup': 'async_after_7_days'
    },
    'denormalization': {
        'enabled': True,
        'stale_after_days': 7,
        'fields': ['id', 'title', 'status']
    },
    'tombstone_ttl_seconds': 604800  # 7 days
}
```

---

## Part 7: Implementation Checklist

### For New FK System in GoT

- [ ] **ID Reservation system** (IDReservation class)
  - [ ] Reserve IDs without materialization
  - [ ] Expiry and cleanup
  - [ ] Test materialization workflow

- [ ] **Hard Reference Management** (HardReferenceManager)
  - [ ] CASCADE DELETE with depth-first traversal
  - [ ] can_delete_safe() validation
  - [ ] Audit trail of deletions

- [ ] **Soft Reference Management** (SoftReferenceManager)
  - [ ] Orphan-tolerant dereference
  - [ ] Broken reference logging
  - [ ] Auto-cleanup of dead edges

- [ ] **Hybrid Model** (ReferencePolicy)
  - [ ] Edge type → strength classification
  - [ ] can_delete_with_references() check
  - [ ] Mixed deletion semantics

- [ ] **Two-Phase Commit** (TwoPhaseCommitCoordinator)
  - [ ] Begin → prepare → commit flow
  - [ ] Abort and rollback
  - [ ] Unfinished transaction recovery

- [ ] **Denormalization** (DenormalizedEdge)
  - [ ] Snapshot denormalized fields
  - [ ] Staleness detection
  - [ ] Refresh mechanism

- [ ] **Tombstones** (TombstoneManager)
  - [ ] Soft delete with TTL
  - [ ] Undelete within grace period
  - [ ] Purge expired tombstones

- [ ] **Testing**
  - [ ] Unit tests for each pattern
  - [ ] Integration tests: CASCADE → Soft → Hybrid
  - [ ] Recovery scenarios after crash
  - [ ] Concurrent entity creation

---

## Summary: Pattern Selection Quick Ref

**Use CASCADE DELETE when:**
- Entity hierarchy is fixed (parent owns children)
- Deletion should be immediate
- Small cascade depth (< 5 levels)
- Example: Sprint → Tasks (delete sprint = delete tasks)

**Use Orphan-Tolerant when:**
- Cross-domain references
- Systems eventually consistent
- Deletion is independent
- Example: Task mentions Decision

**Use 2PC when:**
- Multiple domains must update atomically
- Failure detection is critical
- Consistency boundary matters
- Example: Task creation + sprint assignment

**Use Tombstones when:**
- Deletions must be reversible
- Audit trail is important
- Grace period for cleanup needed
- Example: Task marked for deletion, 7-day recovery window

**Use Denormalization when:**
- Target frequently unavailable
- Join cost is high
- Stale data is acceptable
- Example: Task stores sprint title + status

---

## References

**Recommended papers:**
- "Designing Data-Intensive Applications" (Kleppmann) - Chapters 5-7 on consistency
- "Dynamo: Amazon's Highly Available Key-value Store" (DeCandia et al.) - Eventual consistency patterns
- "Saga Pattern" documentation - Distributed transactions without 2PC

**Real-world implementations:**
- **CockroachDB**: Hard FKs with distributed consistency
- **Cassandra**: Soft references, denormalization, tombstones
- **DynamoDB**: Denormalization, global secondary indexes, eventual consistency
- **Postgres**: Traditional hard FKs with CASCADE, soft references via triggers

