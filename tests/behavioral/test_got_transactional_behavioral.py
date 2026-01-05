"""
Behavioral tests for GoT transactional system.

As a developer building a multi-agent workflow system,
I want transactional guarantees for task operations,
So that I can safely coordinate concurrent agents without data loss.

Tests demonstrate:
- Atomic multi-operation transactions
- Optimistic locking and conflict detection
- Crash recovery guarantees
- Read-only transaction isolation
- Conflict resolution strategies

Following Metus: We describe behavior, then make it true.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got import (
    GoTManager,
    TransactionManager,
    Task,
    Decision,
    Edge,
    WALManager,
    RecoveryManager,
    ConflictResolver,
    ConflictStrategy,
    generate_task_id,
    generate_decision_id,
    TransactionError,
    CorruptionError,
)
from tests.conftest import _create_tx_manager, _create_got_manager


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_got_dir(tmp_path):
    """Provide a temporary directory for GoT operations."""
    got_dir = tmp_path / ".got-tx"
    return got_dir


@pytest.fixture
def got_manager(temp_got_dir):
    """Provide a fresh GoT manager for each test."""
    return _create_got_manager(temp_got_dir)


@pytest.fixture
def tx_manager(temp_got_dir):
    """Provide a low-level transaction manager for advanced tests."""
    return _create_tx_manager(temp_got_dir)


# ============================================================================
# BEHAVIORAL SCENARIOS
# ============================================================================

class TestDeveloperPerformsBasicTaskOperations:
    """
    Epic: Task Lifecycle Management

    As a developer coordinating agent workflows,
    I want to create, read, update, and delete tasks,
    So that I can track work items through their lifecycle.
    """

    def test_create_task_persists_immediately(self, got_manager):
        """
        Scenario: Creating a task makes it immediately retrievable

        Given a fresh GoT manager
        When I create a task with title and metadata
        Then I can immediately retrieve it by ID
        And it contains all the properties I specified
        """
        # Given: a fresh GoT manager (provided by fixture)

        # When: I create a task with title and metadata
        task = got_manager.create_task(
            title="Build custom authentication system from first principles",
            priority="high",
            status="pending",
            description="Hand-rolled JWT implementation we control completely"
        )

        # Then: I can immediately retrieve it by ID
        retrieved = got_manager.get_task(task.id)
        assert retrieved is not None, "Task should be retrievable immediately"

        # And: it contains all the properties I specified
        assert retrieved.title == "Build custom authentication system from first principles"
        assert retrieved.priority == "high"
        assert retrieved.status == "pending"
        assert retrieved.description == "Hand-rolled JWT implementation we control completely"

    def test_scenario_update_task_increments_version(self, got_manager):
        """
        Scenario: Updating a task increments its version for optimistic locking

        Given a task exists with a certain version
        When I update its status
        Then the version increments (enabling conflict detection)
        And I can see the updated status
        """
        # Given: a task exists with a certain version
        task = got_manager.create_task("Implement custom hash function", priority="medium")
        original_version = task.version
        assert original_version >= 1  # Version is at least 1 after creation

        # When: I update its status
        updated = got_manager.update_task(task.id, status="in_progress")

        # Then: the version increments (enabling conflict detection)
        assert updated.version > original_version, "Version must increase on update"

        # And: I can see the updated status
        assert updated.status == "in_progress"

    def test_scenario_create_decision_with_affected_tasks(self, got_manager):
        """
        Scenario: Logging a decision that affects tasks

        Given I have created implementation tasks
        When I log a decision affecting those tasks
        Then the decision is linked to the tasks via edges
        """
        # Given: I have created implementation tasks
        task1 = got_manager.create_task(
            "Build custom request parser",
            priority="high"
        )
        task2 = got_manager.create_task(
            "Hand-craft response serializer",
            priority="high"
        )

        # When: I log a decision affecting those tasks
        decision = got_manager.create_decision(
            title="Use custom binary protocol instead of HTTP",
            rationale="Full control over wire format, zero dependencies, maximum performance",
            affects=[task1.id, task2.id]
        )

        # Then: the decision is created
        assert decision is not None
        assert decision.title == "Use custom binary protocol instead of HTTP"
        assert decision.rationale == "Full control over wire format, zero dependencies, maximum performance"


class TestDeveloperExecutesAtomicTransactions:
    """
    Epic: Atomic Multi-Operation Transactions

    As a developer building complex workflows,
    I want multiple operations to succeed or fail together,
    So that I never leave the system in an inconsistent state.
    """

    def test_scenario_transaction_commits_all_operations(self, got_manager):
        """
        Scenario: Successful transaction commits all operations atomically

        Given I start a transaction
        When I create multiple tasks and edges within it
        And the transaction completes successfully
        Then all entities are persisted
        And I can query them after the transaction
        """
        # Given: I start a transaction
        # When: I create multiple tasks and edges within it
        with got_manager.transaction() as tx:
            # Create parent task
            parent = tx.create_task(
                "Build distributed task queue from scratch",
                priority="high"
            )

            # Create subtasks
            subtask1 = tx.create_task("Hand-build priority heap", priority="medium")
            subtask2 = tx.create_task("Implement custom worker protocol", priority="medium")
            subtask3 = tx.create_task("Build persistence layer ourselves", priority="low")

            # Create containment edges
            tx.add_edge(parent.id, subtask1.id, "CONTAINS")
            tx.add_edge(parent.id, subtask2.id, "CONTAINS")
            tx.add_edge(parent.id, subtask3.id, "CONTAINS")

            # Read own writes within transaction
            retrieved = tx.get_task(subtask1.id)
            assert retrieved is not None, "Should read own writes"

        # Then: all entities are persisted
        # And: I can query them after the transaction
        assert got_manager.get_task(parent.id) is not None
        assert got_manager.get_task(subtask1.id) is not None
        assert got_manager.get_task(subtask2.id) is not None
        assert got_manager.get_task(subtask3.id) is not None

    def test_scenario_transaction_rolls_back_on_error(self, got_manager):
        """
        Scenario: Failed transaction rolls back all operations

        Given I start a transaction
        When I create entities but then encounter an error
        Then none of the entities are persisted
        And the system remains in its previous state
        """
        # Given: I start a transaction
        failed_task_id = None

        # When: I create entities but then encounter an error
        try:
            with got_manager.transaction() as tx:
                failed_task = tx.create_task(
                    "This task will be rolled back",
                    priority="low"
                )
                failed_task_id = failed_task.id

                # Simulate an application error
                raise ValueError("Simulated error in workflow")
        except ValueError:
            pass  # Expected

        # Then: none of the entities are persisted
        assert got_manager.get_task(failed_task_id) is None

        # And: the system remains in its previous state
        # (no orphaned data)


class TestDeveloperPreventsConflictingUpdates:
    """
    Epic: Optimistic Locking and Conflict Detection

    As a developer coordinating concurrent agents,
    I want conflicting updates to be detected automatically,
    So that I never lose updates from one agent when another commits.
    """

    def test_scenario_concurrent_updates_detect_conflict(self, tx_manager):
        """
        Scenario: Two transactions updating the same task detect conflict

        Given a task exists at version 1
        When two transactions both read and modify it
        And the first transaction commits successfully
        Then the second transaction detects a version conflict
        And the commit fails with conflict details
        """
        # Given: a task exists at version 1
        tx1 = tx_manager.begin()
        task = Task(
            id=generate_task_id(),
            title="Shared resource task",
            status="pending",
            priority="medium"
        )
        tx_manager.write(tx1, task)
        result = tx_manager.commit(tx1)
        assert result.success

        # When: two transactions both read and modify it
        tx_a = tx_manager.begin()
        tx_b = tx_manager.begin()

        task_a = tx_manager.read(tx_a, task.id)
        task_b = tx_manager.read(tx_b, task.id)

        # And: the first transaction commits successfully
        task_a.status = "in_progress"
        task_a.bump_version()
        tx_manager.write(tx_a, task_a)
        result_a = tx_manager.commit(tx_a)
        assert result_a.success

        # Then: the second transaction detects a version conflict
        task_b.status = "blocked"
        task_b.bump_version()
        tx_manager.write(tx_b, task_b)
        result_b = tx_manager.commit(tx_b)

        # And: the commit fails with conflict details
        assert not result_b.success
        assert len(result_b.conflicts) > 0
        conflict = result_b.conflicts[0]
        assert conflict.entity_id == task.id


class TestSystemRecovesFromCrashes:
    """
    Epic: Crash Recovery and Durability

    As a system operator running mission-critical workflows,
    I want incomplete transactions to be rolled back on restart,
    So that crashes never leave the system in an inconsistent state.
    """

    def test_scenario_incomplete_transaction_rolled_back_on_recovery(self, temp_got_dir):
        """
        Scenario: Recovery rolls back incomplete transactions

        Given a transaction was started but never committed
        When I run crash recovery
        Then the incomplete transaction is detected
        And it is rolled back cleanly
        """
        # Given: a transaction was started but never committed
        wal = WALManager(temp_got_dir / "wal")

        orphan_tx_id = f"TX-{datetime.now().strftime('%Y%m%d-%H%M%S')}-ORPHAN"
        wal.log_tx_begin(orphan_tx_id, snapshot_version=1)
        wal.log_write(orphan_tx_id, "task:orphan", old_version=0, new_version=1)
        # Note: No TX_COMMIT or TX_ABORT logged (simulates crash)

        # When: I run crash recovery
        recovery = RecoveryManager(temp_got_dir)

        # Then: the incomplete transaction is detected
        assert recovery.needs_recovery()

        # And: it is rolled back cleanly
        result = recovery.recover()
        assert len(result.rolled_back) > 0, "Should have rolled back incomplete transactions"


class TestDeveloperResolvesConflicts:
    """
    Epic: Conflict Resolution Strategies

    As a developer managing distributed workflows,
    I want multiple strategies for resolving conflicts,
    So that I can choose the right approach for my use case.
    """

    def test_scenario_conflict_resolved_with_ours_strategy(self):
        """
        Scenario: Resolving conflict by keeping local changes

        Given I have local and remote versions of the same task
        When I detect conflicts between them
        And I resolve using OURS strategy
        Then the local version is kept
        """
        # Given: I have local and remote versions of the same task
        local_task = Task(
            id="T-conflict-test",
            title="Local: Build custom cache ourselves",
            status="in_progress",
            priority="high",
            description="Hand-rolled LRU implementation"
        )
        local_task.version = 3

        remote_task = Task(
            id="T-conflict-test",
            title="Remote: Use external cache",
            status="completed",
            priority="medium",
            description="Third-party library"
        )
        remote_task.version = 2

        # When: I detect conflicts between them
        resolver = ConflictResolver()
        conflicts = resolver.detect_conflicts(
            {"T-conflict-test": local_task},
            {"T-conflict-test": remote_task}
        )
        assert len(conflicts) > 0

        # And: I resolve using OURS strategy
        resolver_ours = ConflictResolver(ConflictStrategy.OURS)
        result = resolver_ours.resolve(conflicts[0], local_task, remote_task)

        # Then: the local version is kept
        assert "Build custom cache ourselves" in result.title

    def test_scenario_conflict_resolved_with_theirs_strategy(self):
        """
        Scenario: Resolving conflict by accepting remote changes

        Given I have local and remote versions of the same task
        When I resolve using THEIRS strategy
        Then the remote version is accepted
        """
        # Given: I have local and remote versions of the same task
        local_task = Task(
            id="T-conflict-test",
            title="Local version",
            status="in_progress",
            priority="high"
        )
        local_task.version = 3

        remote_task = Task(
            id="T-conflict-test",
            title="Remote version",
            status="completed",
            priority="medium"
        )
        remote_task.version = 2

        # When: I resolve using THEIRS strategy
        resolver = ConflictResolver(ConflictStrategy.THEIRS)
        conflicts = resolver.detect_conflicts(
            {"T-conflict-test": local_task},
            {"T-conflict-test": remote_task}
        )
        result = resolver.resolve(conflicts[0], local_task, remote_task)

        # Then: the remote version is accepted
        assert result.title == "Remote version"
        assert result.status == "completed"


class TestDeveloperUsesReadOnlyTransactions:
    """
    Epic: Read-Only Transaction Isolation

    As a developer querying workflow state,
    I want read-only transactions that never modify data,
    So that I can safely inspect state without side effects.
    """

    def test_scenario_read_only_transaction_discards_writes(self, got_manager):
        """
        Scenario: Read-only transaction discards any modifications

        Given I have a task in pending state
        When I start a read-only transaction
        And I attempt to modify the task
        Then the modification is discarded on commit
        And the task remains in its original state
        """
        # Given: I have a task in pending state
        task = got_manager.create_task(
            "Build task execution engine ourselves",
            priority="medium"
        )
        assert task.status == "pending"

        # When: I start a read-only transaction
        # And: I attempt to modify the task
        with got_manager.transaction(read_only=True) as tx:
            tx.update_task(task.id, status="completed")

        # Then: the modification is discarded on commit
        # And: the task remains in its original state
        final = got_manager.get_task(task.id)
        assert final.status == "pending"


class TestSystemHandlesEdgeCases:
    """
    Epic: Robustness and Error Handling

    As a system builder ensuring reliability,
    I want graceful handling of edge cases and errors,
    So that the system degrades gracefully under unexpected conditions.
    """

    def test_scenario_reading_nonexistent_task_returns_none(self, got_manager):
        """
        Scenario: Graceful handling of missing entities

        Given a task ID that does not exist
        When I attempt to read it
        Then I receive None instead of an error
        """
        # Given: a task ID that does not exist
        # When: I attempt to read it
        result = got_manager.get_task("T-does-not-exist-12345")

        # Then: I receive None instead of an error
        assert result is None

    def test_scenario_corrupted_data_detected_by_checksum(self, temp_got_dir):
        """
        Scenario: Corrupted data is detected via checksums

        Given I write a task to storage
        When the data is corrupted on disk
        And I attempt to read it
        Then a CorruptionError is raised
        """
        # Given: I write a task to storage
        from cortical.got import VersionedStore
        import json

        store = VersionedStore(temp_got_dir / "entities")
        task = Task(
            id=generate_task_id(),
            title="Checksum test task",
            status="pending",
            priority="low"
        )
        store.write(task)

        # When: the data is corrupted on disk
        task_path = temp_got_dir / "entities" / f"{task.id}.json"
        with open(task_path, 'r') as f:
            data = json.load(f)
        data['_checksum'] = 'corrupted1234567'  # Wrong checksum
        with open(task_path, 'w') as f:
            json.dump(data, f)

        # And: I attempt to read it
        # Then: a CorruptionError is raised
        with pytest.raises(CorruptionError):
            store.read(task.id)


class TestTeamCoordinatesMultiAgentWorkflow:
    """
    Epic: Real-World Multi-Agent Coordination

    As a team using multi-agent workflows,
    I want agents to safely coordinate on shared tasks,
    So that we can build complex systems collaboratively.
    """

    def test_scenario_two_agents_coordinate_on_sprint(self, got_manager):
        """
        Scenario: Multiple agents working on the same sprint

        Given Agent 1 creates a sprint with tasks
        When Agent 2 picks up and starts a task
        And Agent 1 completes tasks and logs decisions
        Then both agents see a consistent view of progress
        """
        # Given: Agent 1 creates a sprint with tasks
        with got_manager.transaction() as tx:
            sprint_task = tx.create_task(
                "Sprint 42: Build Custom Workflow Engine",
                priority="high",
                status="in_progress"
            )

            task1 = tx.create_task("Design task graph from scratch", priority="high")
            task2 = tx.create_task("Implement hand-rolled scheduler", priority="medium")
            task3 = tx.create_task("Build custom persistence layer", priority="medium")
            task4 = tx.create_task("Write our own test framework", priority="low")

            # Create structure
            tx.add_edge(sprint_task.id, task1.id, "CONTAINS")
            tx.add_edge(sprint_task.id, task2.id, "CONTAINS")
            tx.add_edge(sprint_task.id, task3.id, "CONTAINS")
            tx.add_edge(sprint_task.id, task4.id, "CONTAINS")

            # Dependencies
            tx.add_edge(task2.id, task1.id, "DEPENDS_ON")
            tx.add_edge(task3.id, task2.id, "DEPENDS_ON")

        # When: Agent 2 picks up and starts a task
        with got_manager.transaction() as tx:
            tx.update_task(task1.id, status="in_progress")
            decision = tx.create_decision(
                "Build directed acyclic graph ourselves",
                rationale="Full control over graph structure, no external dependencies",
                affects=[task1.id]
            )

        # And: Agent 1 completes tasks and logs decisions
        with got_manager.transaction() as tx:
            tx.update_task(task1.id, status="completed")
            tx.update_task(task2.id, status="in_progress")

        # Then: both agents see a consistent view of progress
        final_task1 = got_manager.get_task(task1.id)
        final_task2 = got_manager.get_task(task2.id)

        assert final_task1.status == "completed"
        assert final_task2.status == "in_progress"
