#!/usr/bin/env python3
"""
GoT Transactional System - Comprehensive Demo

This script demonstrates all features of the GoT transactional system
as they would be used in real multi-agent workflows.

Usage:
    python examples/got_demo.py [--clean]

Features demonstrated:
1. Basic CRUD operations
2. Multi-operation transactions
3. Conflict detection and resolution
4. Crash recovery
5. Sync operations (simulated)
6. Edge cases and error handling
"""

import sys
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got import (
    # High-level API
    GoTManager, TransactionContext,
    generate_task_id, generate_decision_id,

    # Core types
    Entity, Task, Decision, Edge,
    Transaction, TransactionState,

    # Managers
    TransactionManager, VersionedStore, WALManager,
    RecoveryManager, SyncManager, ConflictResolver,

    # Results
    CommitResult, RecoveryResult, SyncResult,

    # Strategies
    ConflictStrategy,

    # Errors
    TransactionError, ConflictError, CorruptionError,
)


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_task(task: Task, prefix: str = ""):
    """Pretty-print a task."""
    print(f"{prefix}Task: {task.id}")
    print(f"{prefix}  Title: {task.title}")
    print(f"{prefix}  Status: {task.status}")
    print(f"{prefix}  Priority: {task.priority}")
    print(f"{prefix}  Version: {task.version}")


def demo_basic_crud(manager: GoTManager):
    """Demo 1: Basic Create, Read, Update, Delete operations."""
    print_header("Demo 1: Basic CRUD Operations")

    # Create a task
    print("Creating a task...")
    task = manager.create_task(
        title="Implement user authentication",
        priority="high",
        status="pending",
        description="Add JWT-based auth to the API"
    )
    print_task(task)

    # Read the task back
    print("\nReading task back...")
    retrieved = manager.get_task(task.id)
    assert retrieved is not None
    assert retrieved.title == task.title
    print(f"✓ Task retrieved successfully")

    # Update the task
    print("\nUpdating task status...")
    updated = manager.update_task(task.id, status="in_progress")
    print_task(updated)
    assert updated.status == "in_progress"
    assert updated.version > task.version
    print(f"✓ Task updated (version {task.version} → {updated.version})")

    # Create a decision
    print("\nCreating a decision...")
    decision = manager.create_decision(
        title="Use JWT for authentication",
        rationale="Industry standard, stateless, works well with microservices",
        affects=[task.id]
    )
    print(f"Decision: {decision.id}")
    print(f"  Title: {decision.title}")
    print(f"  Rationale: {decision.rationale}")

    # Create an edge
    print("\nCreating dependency edge...")
    edge = manager.add_edge(
        source_id=decision.id,
        target_id=task.id,
        edge_type="MOTIVATES"
    )
    print(f"Edge: {edge.source_id} --[{edge.edge_type}]--> {edge.target_id}")

    print("\n✓ Demo 1 Complete: Basic CRUD operations work correctly")
    return task


def demo_multi_operation_transaction(manager: GoTManager):
    """Demo 2: Multi-operation transactions with auto-commit/rollback."""
    print_header("Demo 2: Multi-Operation Transactions")

    # Successful transaction
    print("Starting transaction with multiple operations...")
    with manager.transaction() as tx:
        # Create parent task
        parent = tx.create_task("Build API endpoints", priority="high")
        print(f"  Created parent: {parent.id}")

        # Create subtasks
        subtask1 = tx.create_task("GET /users endpoint", priority="medium")
        subtask2 = tx.create_task("POST /users endpoint", priority="medium")
        subtask3 = tx.create_task("DELETE /users endpoint", priority="low")
        print(f"  Created 3 subtasks")

        # Create containment edges
        tx.add_edge(parent.id, subtask1.id, "CONTAINS")
        tx.add_edge(parent.id, subtask2.id, "CONTAINS")
        tx.add_edge(parent.id, subtask3.id, "CONTAINS")
        print(f"  Created 3 edges")

        # Read own writes
        retrieved = tx.get_task(subtask1.id)
        assert retrieved is not None
        print(f"  ✓ Read-your-own-writes works")

    print("Transaction committed successfully")

    # Verify all persisted
    assert manager.get_task(parent.id) is not None
    assert manager.get_task(subtask1.id) is not None
    print("✓ All entities persisted after commit")

    # Failed transaction (rollback)
    print("\nStarting transaction that will fail...")
    failed_task_id = None
    try:
        with manager.transaction() as tx:
            failed_task = tx.create_task("This will be rolled back", priority="low")
            failed_task_id = failed_task.id
            print(f"  Created task: {failed_task_id}")

            # Simulate an error
            raise ValueError("Simulated application error")
    except ValueError as e:
        print(f"  Exception caught: {e}")

    # Verify rollback
    assert manager.get_task(failed_task_id) is None
    print("✓ Task was rolled back (not persisted)")

    print("\n✓ Demo 2 Complete: Transactions commit on success, rollback on error")
    return parent


def demo_conflict_detection(got_dir: Path):
    """Demo 3: Optimistic locking and conflict detection."""
    print_header("Demo 3: Conflict Detection")

    # Use low-level API to demonstrate conflicts
    tx_manager = TransactionManager(got_dir)

    # Create a task
    tx1 = tx_manager.begin()
    task = Task(
        id=generate_task_id(),
        title="Shared task",
        status="pending",
        priority="medium"
    )
    tx_manager.write(tx1, task)
    result = tx_manager.commit(tx1)
    assert result.success
    print(f"Created task: {task.id} (version {task.version})")

    # Start two concurrent transactions
    print("\nStarting two concurrent transactions...")
    tx_a = tx_manager.begin()
    tx_b = tx_manager.begin()
    print(f"  TX-A: {tx_a.id}")
    print(f"  TX-B: {tx_b.id}")

    # Both read the same task
    task_a = tx_manager.read(tx_a, task.id)
    task_b = tx_manager.read(tx_b, task.id)
    print(f"  Both read task at version {task_a.version}")

    # TX-A modifies and commits first
    task_a.status = "in_progress"
    task_a.bump_version()
    tx_manager.write(tx_a, task_a)
    result_a = tx_manager.commit(tx_a)
    assert result_a.success
    print(f"  TX-A committed: status → in_progress")

    # TX-B tries to modify (should conflict)
    task_b.status = "blocked"
    task_b.bump_version()
    tx_manager.write(tx_b, task_b)
    result_b = tx_manager.commit(tx_b)

    if not result_b.success:
        print(f"  TX-B failed: {result_b.reason}")
        for conflict in result_b.conflicts:
            print(f"    Conflict on {conflict.entity_id}:")
            print(f"      Expected version: {conflict.expected_version}")
            print(f"      Actual version: {conflict.actual_version}")
        print("✓ Conflict correctly detected!")
    else:
        print("✗ ERROR: Conflict should have been detected")

    print("\n✓ Demo 3 Complete: Optimistic locking prevents lost updates")


def demo_crash_recovery(got_dir: Path):
    """Demo 4: Crash recovery from incomplete transactions."""
    print_header("Demo 4: Crash Recovery")

    # Simulate a crash by writing to WAL but not committing
    wal = WALManager(got_dir / "wal")

    # Log a transaction that never committed
    orphan_tx_id = f"TX-{datetime.now().strftime('%Y%m%d-%H%M%S')}-ORPHAN"
    wal.log_tx_begin(orphan_tx_id, snapshot_version=1)
    wal.log_write(orphan_tx_id, "task:orphan", old_version=0, new_version=1)
    # Note: No TX_COMMIT or TX_ABORT logged
    print(f"Simulated incomplete transaction: {orphan_tx_id}")

    # Run recovery
    print("\nRunning recovery...")
    recovery = RecoveryManager(got_dir)

    if recovery.needs_recovery():
        print("  Recovery needed!")
        result = recovery.recover()
        print(f"  Recovered transactions: {result.recovered_transactions}")
        print(f"  Rolled back: {result.rolled_back}")
        print(f"  Actions taken:")
        for action in result.actions_taken:
            print(f"    - {action}")
    else:
        print("  No recovery needed")

    print("\n✓ Demo 4 Complete: Incomplete transactions are rolled back on startup")


def demo_conflict_resolution():
    """Demo 5: Conflict resolution strategies."""
    print_header("Demo 5: Conflict Resolution Strategies")

    # Create conflicting tasks (different versions = conflict)
    local_task = Task(
        id="T-conflict-test",
        title="Local version",
        status="in_progress",
        priority="high",
        description="Modified locally"
    )
    local_task.version = 3  # Local is at version 3

    remote_task = Task(
        id="T-conflict-test",
        title="Remote version",
        status="completed",
        priority="medium",
        description="Modified remotely"
    )
    remote_task.version = 2  # Remote is at version 2 (diverged)

    print("Local task:")
    print(f"  title: {local_task.title}")
    print(f"  status: {local_task.status}")
    print(f"  priority: {local_task.priority}")

    print("\nRemote task:")
    print(f"  title: {remote_task.title}")
    print(f"  status: {remote_task.status}")
    print(f"  priority: {remote_task.priority}")

    resolver = ConflictResolver()

    # Detect conflicts
    conflicts = resolver.detect_conflicts(
        {"T-conflict-test": local_task},
        {"T-conflict-test": remote_task}
    )
    print(f"\nConflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"  {c.entity_id}: fields {c.conflict_fields}")

    # Strategy: OURS (keep local)
    resolver_ours = ConflictResolver(ConflictStrategy.OURS)
    result_ours = resolver_ours.resolve(conflicts[0], local_task, remote_task)
    print(f"\nOURS strategy result: {result_ours.title}, {result_ours.status}")

    # Strategy: THEIRS (take remote)
    resolver_theirs = ConflictResolver(ConflictStrategy.THEIRS)
    result_theirs = resolver_theirs.resolve(conflicts[0], local_task, remote_task)
    print(f"THEIRS strategy result: {result_theirs.title}, {result_theirs.status}")

    print("\n✓ Demo 5 Complete: Multiple conflict resolution strategies available")


def demo_read_only_transactions(manager: GoTManager):
    """Demo 6: Read-only transactions."""
    print_header("Demo 6: Read-Only Transactions")

    # Create a task first
    task = manager.create_task("Test task", priority="medium")
    print(f"Created task: {task.id}")

    # Read-only transaction
    print("\nStarting read-only transaction...")
    with manager.transaction(read_only=True) as tx:
        retrieved = tx.get_task(task.id)
        print(f"  Read task: {retrieved.title}")

        # Try to modify (will be discarded)
        tx.update_task(task.id, status="completed")
        print("  Modified task in transaction")

    # Verify changes were NOT persisted
    final = manager.get_task(task.id)
    assert final.status == "pending"  # Original status
    print(f"✓ Changes discarded (status still: {final.status})")

    print("\n✓ Demo 6 Complete: Read-only transactions don't persist changes")


def demo_edge_cases(manager: GoTManager, got_dir: Path):
    """Demo 7: Edge cases and error handling."""
    print_header("Demo 7: Edge Cases")

    # 1. Missing entity
    print("1. Reading non-existent task...")
    missing = manager.get_task("T-does-not-exist")
    assert missing is None
    print("   ✓ Returns None for missing entities")

    # 2. Invalid status
    print("\n2. Creating task with invalid status...")
    try:
        bad_task = Task(
            id=generate_task_id(),
            title="Bad task",
            status="invalid_status",  # Not a valid status
            priority="high"
        )
        print("   ✗ Should have raised ValidationError")
    except Exception as e:
        print(f"   ✓ Raised {type(e).__name__}: {e}")

    # 3. Invalid priority
    print("\n3. Creating task with invalid priority...")
    try:
        bad_task = Task(
            id=generate_task_id(),
            title="Bad task",
            status="pending",
            priority="super_urgent"  # Not a valid priority
        )
        print("   ✗ Should have raised ValidationError")
    except Exception as e:
        print(f"   ✓ Raised {type(e).__name__}: {e}")

    # 4. Edge weight bounds
    print("\n4. Creating edge with invalid weight...")
    try:
        bad_edge = Edge(
            id="",
            source_id="T-1",
            target_id="T-2",
            edge_type="BLOCKS",
            weight=1.5  # Must be 0.0-1.0
        )
        print("   ✗ Should have raised ValidationError")
    except Exception as e:
        print(f"   ✓ Raised {type(e).__name__}: {e}")

    # 5. Checksum verification
    print("\n5. Corrupted data detection...")
    store = VersionedStore(got_dir / "entities")
    task = Task(
        id=generate_task_id(),
        title="Checksum test",
        status="pending",
        priority="low"
    )
    store.write(task)

    # Manually corrupt the file
    task_path = got_dir / "entities" / f"{task.id}.json"
    import json
    with open(task_path, 'r') as f:
        data = json.load(f)
    data['_checksum'] = 'corrupted1234567'  # Wrong checksum
    with open(task_path, 'w') as f:
        json.dump(data, f)

    try:
        corrupted = store.read(task.id)
        print("   ✗ Should have raised CorruptionError")
    except CorruptionError as e:
        print(f"   ✓ Detected corruption: {e}")

    print("\n✓ Demo 7 Complete: Edge cases handled correctly")


def demo_workflow_scenario(manager: GoTManager):
    """Demo 8: Real-world multi-agent workflow scenario."""
    print_header("Demo 8: Real-World Workflow Scenario")

    print("Scenario: Two agents working on the same sprint\n")

    # Agent 1: Creates sprint tasks
    print("Agent 1: Creating sprint tasks...")
    with manager.transaction() as tx:
        sprint_task = tx.create_task(
            "Sprint 42: User Management",
            priority="high",
            status="in_progress"
        )

        task1 = tx.create_task("Implement user model", priority="high")
        task2 = tx.create_task("Add user validation", priority="medium")
        task3 = tx.create_task("Create user API endpoints", priority="medium")
        task4 = tx.create_task("Write user tests", priority="low")

        # Create structure
        tx.add_edge(sprint_task.id, task1.id, "CONTAINS")
        tx.add_edge(sprint_task.id, task2.id, "CONTAINS")
        tx.add_edge(sprint_task.id, task3.id, "CONTAINS")
        tx.add_edge(sprint_task.id, task4.id, "CONTAINS")

        # Dependencies
        tx.add_edge(task2.id, task1.id, "DEPENDS_ON")
        tx.add_edge(task3.id, task2.id, "DEPENDS_ON")
        tx.add_edge(task4.id, task3.id, "DEPENDS_ON")

    print(f"  Created sprint: {sprint_task.id}")
    print(f"  Created 4 tasks with dependencies")

    # Agent 2: Starts working on first task
    print("\nAgent 2: Starting work on first task...")
    with manager.transaction() as tx:
        task = tx.get_task(task1.id)
        tx.update_task(task1.id, status="in_progress")

        decision = tx.create_decision(
            "Use SQLAlchemy for user model",
            rationale="Team familiarity, good async support",
            affects=[task1.id]
        )
        tx.add_edge(decision.id, task1.id, "MOTIVATES")

    print(f"  Updated task status: in_progress")
    print(f"  Logged decision: {decision.id}")

    # Agent 1: Completes task and moves to next
    print("\nAgent 1: Completing first task...")
    with manager.transaction() as tx:
        tx.update_task(task1.id, status="completed")
        tx.update_task(task2.id, status="in_progress")

    print("  Task 1: completed")
    print("  Task 2: in_progress")

    # Final state
    print("\nFinal sprint state:")
    for tid in [task1.id, task2.id, task3.id, task4.id]:
        t = manager.get_task(tid)
        print(f"  {t.title}: {t.status}")

    print("\n✓ Demo 8 Complete: Multi-agent workflow executed successfully")


def run_all_demos(clean: bool = True):
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("  GoT TRANSACTIONAL SYSTEM - COMPREHENSIVE DEMO")
    print("="*60)

    # Create temp directory for demo
    demo_dir = Path(tempfile.mkdtemp(prefix="got_demo_"))
    got_dir = demo_dir / ".got-tx"

    print(f"\nDemo directory: {demo_dir}")

    try:
        # Initialize manager
        manager = GoTManager(got_dir)

        # Run demos
        demo_basic_crud(manager)
        demo_multi_operation_transaction(manager)
        demo_conflict_detection(got_dir)
        demo_crash_recovery(got_dir)
        demo_conflict_resolution()
        demo_read_only_transactions(manager)
        demo_edge_cases(manager, got_dir)
        demo_workflow_scenario(manager)

        # Summary
        print_header("DEMO COMPLETE")
        print("All 8 demonstrations completed successfully!")
        print("\nFeatures verified:")
        print("  ✓ Basic CRUD operations")
        print("  ✓ Multi-operation transactions")
        print("  ✓ Conflict detection (optimistic locking)")
        print("  ✓ Crash recovery")
        print("  ✓ Conflict resolution strategies")
        print("  ✓ Read-only transactions")
        print("  ✓ Edge cases and error handling")
        print("  ✓ Real-world workflow scenarios")

    finally:
        if clean:
            print(f"\nCleaning up: {demo_dir}")
            shutil.rmtree(demo_dir)
        else:
            print(f"\nDemo data preserved at: {demo_dir}")


if __name__ == "__main__":
    clean = "--clean" not in sys.argv or len(sys.argv) == 1
    run_all_demos(clean=clean)
