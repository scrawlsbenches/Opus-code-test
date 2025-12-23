# Foreign Key Patterns: Code Examples

**Quick reference for common FK scenarios**

---

## Scenario 1: Creating an Entity with Reserved ID (Multi-Agent)

### Problem
Agent A needs to tell Agent B about a new task ID before the task is fully created.

### Solution: ID Reservation Pattern

```python
# cortical/got/examples/example_reservation.py

from pathlib import Path
from cortical.got.reservation import IDReservation

# Initialize reservation system
reservations = IDReservation(Path(".got/reservations"))

# Agent A: Reserve an ID (fast, no entity creation yet)
task_id = reservations.reserve_id("task", agent_id="agent-A")
print(f"Reserved: {task_id}")  # T-20251223-093045-a1b2c3d4

# Agent A: Do expensive work...
# - Fetch dependencies
# - Validate requirements
# - Create related entities
task_data = {
    'title': 'Implement authentication',
    'priority': 'high',
    'sprint_id': 'S-001'  # Can reference other entities
}

# Agent A: Materialize the reservation
success = reservations.materialize(task_id, task_data)

if success:
    # Now safe for Agent B to reference this task
    print(f"Task {task_id} is ready")

    # Agent B can add it to sprint
    sprint.add_task(task_id)
else:
    # Reservation expired, try again
    task_id = reservations.reserve_id("task", agent_id="agent-A")
```

**Key pattern:**
```
┌──────────────────┐
│ Agent A reserves │ → T-001 (can tell Agent B)
│ ID (no entity)   │
└──────────────────┘
         ↓
┌──────────────────┐
│ Agent A creates  │ → validates, does work
│ entity data      │
└──────────────────┘
         ↓
┌──────────────────┐
│ Agent A          │ → NOW it's safe for Agent B
│ materializes     │   to create references
└──────────────────┘
```

---

## Scenario 2: Deleting a Task (Check Safe First)

### Problem
User clicks "Delete Task" - what gets deleted?

### Solution: Deletion Safety Check Pattern

```python
# Example usage in CLI or API

from cortical.got.safe_deletion import SafeEntityDeleter, DeletionMode
from cortical.got.deletion_safety import DeletionValidator

def delete_task_interactive(task_id: str, got_manager):
    """Interactive deletion with safety checks."""

    # Step 1: Check if safe to delete
    validator = DeletionValidator(got_manager)
    can_delete, reasons = validator.can_delete_entity(task_id, allow_cascade=False)

    if can_delete:
        # Simple case: no strong references
        print(f"✅ Safe to delete {task_id}")
        result = got_manager.delete_entity_safe(task_id, mode="safe")
        print(f"Deleted {result.entities_deleted} entities")
        return True

    # Step 2: Show what's blocking deletion
    print(f"\n❌ Cannot delete {task_id} in safe mode:")
    for reason in reasons:
        print(f"   {reason}")

    # Step 3: Offer cascade delete option
    scope = validator.get_cascade_scope(task_id)

    print(f"\n⚠️  CASCADE DELETE would delete {scope['total_count']} items:")
    print(f"   Entities: {scope['entities_to_delete']}")
    print(f"   Edges: {scope['deleted_edges']}")

    # Step 4: Ask user
    response = input("\nProceed with cascade delete? (yes/no): ")

    if response.lower() == 'yes':
        result = got_manager.delete_entity_safe(task_id, mode="cascade", reason="User confirmed")
        print(f"\n✅ Deleted {result.entities_deleted} entities")
        return True
    else:
        print("\n❌ Deletion cancelled")
        return False
```

---

## Scenario 3: Task References Deleted Decision (Graceful Degradation)

### Problem
Task displays decision it depends on. Decision gets deleted. What happens?

### Solution: Soft Reference Pattern

```python
# In task display logic

from cortical.got.soft_references import SoftReferenceHandler

def render_task_details(task_id: str, got_manager):
    """Render task with graceful handling of deleted references."""

    handler = SoftReferenceHandler(got_manager)
    task = got_manager.get_entity(task_id)

    print(f"Task: {task.title}")
    print(f"Status: {task.status}")

    # Handle decision reference gracefully
    decision = handler.resolve_reference(
        source_id=task_id,
        target_id=task.related_decision_id,
        fallback={
            'title': '(Decision deleted)',
            'status': 'unknown',
            'rationale': 'This decision was removed from the system'
        },
        strict=False  # Don't error, use fallback
    )

    if decision:
        print(f"\nDecision: {decision['title']}")
        print(f"Rationale: {decision.get('rationale', 'N/A')}")

        if decision.get('title') == '(Decision deleted)':
            print("⚠️  Decision was deleted but task is still valid")
    else:
        print("\n⚠️  Related decision is no longer available")


# Later: cleanup operation
def cleanup_broken_references(got_manager, older_than_days=7):
    """Periodically clean up references to deleted entities."""

    handler = SoftReferenceHandler(got_manager)

    # Find broken references
    broken = handler.validate_all_references()
    print(f"Found {len(broken)} broken references")

    # Clean up old ones (> 7 days old)
    cleaned = handler.cleanup_orphaned_edges(
        older_than_days=older_than_days,
        dry_run=False
    )
    print(f"Cleaned {cleaned} orphaned edges")
```

---

## Scenario 4: Strong References (CASCADE DELETE)

### Problem
Sprint is deleted. All tasks in that sprint should be deleted too.

### Solution: Strong Reference Classification

```python
# In cortical/got/reference_strength.py configuration

# Define which edges cause CASCADE DELETE
STRONG_EDGES = {
    "PART_OF",    # Task is PART_OF Sprint → cascade delete
    "REQUIRES",   # Task REQUIRES Decision → cascade delete if needed
    "CONTAINS"    # Container CONTAINS items → cascade items
}

WEAK_EDGES = {
    "MENTIONS",   # Task MENTIONS Decision → decision can disappear
    "REFERENCES", # Task REFERENCES another Task → can be orphaned
}

# Usage:

from cortical.got.safe_deletion import SafeEntityDeleter, DeletionMode

deleter = SafeEntityDeleter(got_manager)

# User deletes sprint
result = deleter.delete_entity(
    "S-001",
    mode=DeletionMode.CASCADE,  # Explicit cascade mode
    user_id="admin@company.com",
    reason="Closing old sprint"
)

print(f"Deleted {result.entities_deleted} entities:")
print(f"  - Sprint: 1")
print(f"  - Tasks: {result.entities_deleted - 1}")
```

---

## Scenario 5: Two-Phase Commit (Cross-Domain Atomicity)

### Problem
Create task AND add it to sprint atomically. If sprint addition fails, don't create task.

### Solution: Two-Phase Commit

```python
# Example: atomic task creation in a sprint

from cortical.got.two_phase_commit import TwoPhaseCommitCoordinator, TxPhase

def create_task_in_sprint_atomic(
    task_title: str,
    sprint_id: str,
    got_manager
) -> bool:
    """Create task and add to sprint atomically."""

    coordinator = TwoPhaseCommitCoordinator(Path(".got/transactions"))

    # Begin transaction with two domains
    tx = coordinator.begin_transaction(
        tx_id=f"TX-{sprint_id}-{task_title[:10]}",
        participants=['tasks', 'sprints']
    )

    # Add operations
    coordinator.add_operation(
        tx.tx_id,
        'tasks',
        {
            'op': 'create_task',
            'title': task_title,
            'priority': 'medium',
            'status': 'pending'
        }
    )

    coordinator.add_operation(
        tx.tx_id,
        'sprints',
        {
            'op': 'add_task',
            'sprint_id': sprint_id,
            'task_title': task_title
        }
    )

    # PREPARE PHASE: Ask participants if they can commit
    print("Phase 1: PREPARE")

    task_can_commit = try_create_task(task_title)
    sprint_can_commit = try_add_to_sprint(sprint_id, task_title)

    can_commit = coordinator.prepare(
        tx.tx_id,
        {
            'tasks': task_can_commit,
            'sprints': sprint_can_commit
        }
    )

    if not can_commit:
        # At least one said "no" - abort everything
        print("Prepare failed, aborting transaction")
        coordinator.abort(tx.tx_id, reason="Prepare phase failed")
        return False

    # COMMIT PHASE: Actually do it
    print("Phase 2: COMMIT")

    task_committed = commit_create_task(task_title)
    sprint_committed = commit_add_to_sprint(sprint_id, task_title)

    success = coordinator.commit(
        tx.tx_id,
        {
            'tasks': task_committed,
            'sprints': sprint_committed
        }
    )

    if success:
        print(f"✅ Transaction {tx.tx_id} committed")
        return True
    else:
        print(f"❌ Transaction {tx.tx_id} partially failed")
        # Manual intervention may be needed
        return False


# Recovery after crash
def recover_interrupted_transactions(got_manager):
    """Recover unfinished 2PC transactions."""

    coordinator = TwoPhaseCommitCoordinator(Path(".got/transactions"))

    unfinished = coordinator.get_unfinished_transactions()

    for tx in unfinished:
        print(f"Recovering {tx.tx_id} (phase: {tx.phase.value})")

        recovery_result = coordinator.recover_transaction(tx)

        if recovery_result['status'] == 'rolled_back':
            print(f"  → Safely rolled back")

        elif recovery_result['status'] == 'recovered':
            print(f"  → Recovered and committed")

        else:
            print(f"  ⚠️  Needs manual intervention:")
            print(f"    {recovery_result['reason']}")
```

---

## Scenario 6: Denormalization with Version Vectors

### Problem
Need to display task with sprint title/status but sprint might be deleted later. No joins allowed.

### Solution: Denormalized Edge with Version Vector

```python
# Task stores denormalized sprint data

from cortical.got.patterns.denormalization import DenormalizedEdge, DenormalizationStrategy

def create_task_with_denormalized_sprint(
    task_title: str,
    sprint_id: str,
    got_manager,
    current_replica_id: str = "replica-1"
):
    """Create task with denormalized sprint data."""

    # Get current sprint
    sprint = got_manager.get_entity(sprint_id)

    # Create denormalized edge
    denorm_edge = DenormalizedEdge(
        from_id=task_id,
        to_id=sprint_id,
        edge_type="PART_OF",
        target_snapshot={
            'id': sprint.id,
            'title': sprint.title,
            'status': sprint.status,
            'number': sprint.number
        }
    )

    # Store edge
    got_manager.add_edge(
        from_id=task_id,
        to_id=sprint_id,
        edge_type="PART_OF",
        metadata={'snapshot': denorm_edge.target_snapshot}
    )

    # Later: display task
    task = got_manager.get_entity(task_id)
    edge = got_manager.get_edge(from_id=task_id, to_id=sprint_id)

    # Check if denormalized data is stale
    if edge['metadata']['snapshot_age_days'] > 7:
        # Try to refresh
        fresh_sprint = got_manager.get_entity(sprint_id)
        if fresh_sprint:
            edge['metadata']['snapshot'] = {
                'id': fresh_sprint.id,
                'title': fresh_sprint.title,
                'status': fresh_sprint.status
            }
        else:
            # Sprint was deleted, but we have denormalized data
            print(f"Sprint {sprint_id} was deleted, using cached data")
```

---

## Scenario 7: Tombstones with TTL (Reversible Deletion)

### Problem
User deletes task by mistake. Need to recover within 24 hours.

### Solution: Tombstone with TTL

```python
# Task deletion with recovery window

from cortical.got.patterns.tombstones import TombstoneManager
from datetime import timedelta

tombstone_mgr = TombstoneManager(Path(".got/tombstones"))

def soft_delete_task(task_id: str, got_manager, reason: str = ""):
    """Delete task but allow recovery for 24 hours."""

    # Create tombstone
    tombstone = tombstone_mgr.soft_delete_entity(
        entity_id=task_id,
        entity_type="task",
        ttl_seconds=86400,  # 24 hours
        reason=reason,
        deleted_by="user-123"
    )

    # User sees confirmation
    print(f"✅ Task {task_id} deleted")
    print(f"   Recovery available until: {tombstone.get_expiry_time()}")

    # References to this task still work (but show as deleted)
    # They have 24 hours to clean up


# User wants to recover
def recover_deleted_task(task_id: str):
    """Undelete task within recovery window."""

    if not tombstone_mgr.is_deleted(task_id):
        print(f"Task {task_id} is not deleted")
        return False

    tombstone = tombstone_mgr.get_tombstone(task_id)
    remaining_secs = tombstone['time_until_expiry']

    if remaining_secs <= 0:
        print(f"Recovery window expired, cannot undelete")
        return False

    # Undelete
    success = tombstone_mgr.undelete_entity(task_id)

    if success:
        print(f"✅ Task {task_id} recovered")
        print(f"   TTL reset to 24 hours")
        return True


# Nightly cleanup
def purge_expired_tombstones():
    """Permanently delete entities that exceeded TTL."""

    expired_count = tombstone_mgr.purge_expired_tombstones()
    print(f"Purged {expired_count} expired tombstones")
```

---

## Scenario 8: Finding Broken References

### Problem
System has orphaned edges. How do we find and fix them?

### Solution: Validation and Cleanup

```python
# Audit script to find broken references

from cortical.got.soft_references import SoftReferenceHandler

def audit_references(got_manager):
    """Find and report broken references."""

    handler = SoftReferenceHandler(got_manager)

    # Find all broken references
    broken = handler.validate_all_references()

    if not broken:
        print("✅ All references are valid")
        return

    print(f"⚠️  Found {len(broken)} broken references:\n")

    # Group by source entity
    by_source = {}
    for ref in broken:
        if ref.source_id not in by_source:
            by_source[ref.source_id] = []
        by_source[ref.source_id].append(ref)

    for source_id, refs in by_source.items():
        entity = got_manager.get_entity(source_id)
        print(f"{source_id} ({entity.entity_type})")

        for ref in refs:
            print(f"  → {ref.target_id} (via {ref.edge_type})")
            print(f"    Broken since: {ref.discovered_at}")

    # Option 1: Clean up automatically
    print(f"\nCleaning up edges older than 7 days...")
    cleaned = handler.cleanup_orphaned_edges(
        older_than_days=7,
        dry_run=False
    )
    print(f"Cleaned {cleaned} edges")

    # Option 2: Show manual fix (for recent orphans)
    recent = [r for r in broken if r.attempts_to_resolve == 0]
    if recent:
        print(f"\n{len(recent)} recent orphans need manual review:")
        for ref in recent[:5]:
            print(f"  - {ref.source_id} → {ref.target_id}")
```

---

## Scenario 9: Policy-Based Deletion (Hybrid Model)

### Problem
Some edges should block deletion, others shouldn't. Need consistent policy.

### Solution: Reference Strength Policy

```python
# Configuration file: cortical/got/reference_policy.py

from enum import Enum

class ReferencePolicy(Enum):
    """Business rules for reference deletion semantics."""

    # By domain
    TASK_SPRINT = "strong"      # Task.PART_OF Sprint → cascade
    TASK_DECISION = "weak"      # Task.MENTIONS Decision → orphan OK

    # By role
    TASK_TASK = "weak"          # Task.DEPENDS_ON Task → can be resolved
    SPRINT_EPIC = "strong"      # Sprint.PART_OF Epic → cascade
    DECISION_TASK = "weak"      # Decision.IMPLEMENTS Task → orphan OK


def determine_deletion_policy(source_id: str, target_id: str, edge_type: str) -> str:
    """
    Determine if deletion should cascade based on policy.

    Returns: 'strong' | 'weak' | 'ask_user'
    """

    source_type = classify_entity_type(source_id)
    target_type = classify_entity_type(target_id)

    # Policy rules
    if source_type == 'task' and target_type == 'sprint' and edge_type == 'PART_OF':
        return 'strong'  # Task depends on sprint

    if source_type == 'task' and target_type == 'decision' and edge_type == 'MENTIONS':
        return 'weak'  # Task doesn't depend on decision existing

    if source_type == 'sprint' and target_type == 'epic' and edge_type == 'PART_OF':
        return 'strong'  # Sprint depends on epic

    # Default: ask user
    return 'ask_user'


# Usage in deletion logic
def delete_with_policy(entity_id: str, got_manager):
    """Delete entity respecting policy."""

    incoming_edges = got_manager.find_edges_by_target(entity_id)

    strong_count = 0
    weak_count = 0
    ask_count = 0

    for edge in incoming_edges:
        policy = determine_deletion_policy(edge.source_id, entity_id, edge.edge_type)

        if policy == 'strong':
            strong_count += 1
        elif policy == 'weak':
            weak_count += 1
        else:
            ask_count += 1

    # Show summary
    print(f"Deletion policy for {entity_id}:")
    print(f"  {strong_count} STRONG refs (will cascade)")
    print(f"  {weak_count} WEAK refs (will orphan)")
    print(f"  {ask_count} AMBIGUOUS refs (need input)")

    if strong_count > 0:
        response = input("\nProceed with cascade delete? (yes/no): ")
        if response.lower() == 'yes':
            got_manager.delete_entity_safe(entity_id, mode="cascade")
    else:
        got_manager.delete_entity_safe(entity_id, mode="safe")
```

---

## Quick Reference: When to Use What

| Scenario | Pattern | Code |
|----------|---------|------|
| **Multi-agent creation** | ID Reservation | `IDReservation.reserve_id()` + `materialize()` |
| **Safe deletion check** | Deletion Validator | `DeletionValidator.can_delete_entity()` |
| **Show cascade scope** | Deletion Scope | `DeletionValidator.get_cascade_scope()` |
| **Delete with cascade** | Safe Deleter | `SafeEntityDeleter.delete_entity(mode=CASCADE)` |
| **Reference to deleted** | Soft Reference | `SoftReferenceHandler.resolve_reference()` |
| **Find all broken refs** | Validation | `SoftReferenceHandler.validate_all_references()` |
| **Cleanup dead edges** | Cleanup | `SoftReferenceHandler.cleanup_orphaned_edges()` |
| **Cross-domain atomic** | Two-Phase Commit | `TwoPhaseCommitCoordinator` |
| **Reversible deletion** | Tombstone | `TombstoneManager.soft_delete_entity()` |
| **Denormalized data** | Denormalization | `DenormalizedEdge` + snapshot refresh |
| **Policy-based** | Reference Strength | `EdgeTypeClassification.get_strength()` |

---

## Testing Pattern

```python
# tests/unit/test_fk_scenario.py

def test_safe_deletion_blocks_cascade():
    """Safe mode blocks deletion if cascade needed."""
    got = setup_got_with_task_in_sprint()

    deleter = SafeEntityDeleter(got)

    # Try to delete sprint (which has tasks)
    result = deleter.delete_entity(
        "S-001",
        mode=DeletionMode.SAFE
    )

    # Should fail
    assert not result.success
    assert any("CASCADE" in e for e in result.errors)

    # Sprint should still exist
    assert got.get_entity("S-001") is not None


def test_cascade_deletion_succeeds():
    """Cascade mode deletes everything."""
    got = setup_got_with_task_in_sprint()

    deleter = SafeEntityDeleter(got)

    # Delete with cascade
    result = deleter.delete_entity(
        "S-001",
        mode=DeletionMode.CASCADE
    )

    # Should succeed
    assert result.success
    assert result.entities_deleted >= 2  # sprint + tasks

    # Sprint should be gone
    assert got.get_entity("S-001") is None
```

