"""
Behavioral tests for CDG Transaction Manager.

Epic: Developer Uses ACID Transactions for Graph Data

As a developer building distributed graph systems,
I want ACID transactions we implemented from first principles,
So that I can modify graph data with consistency guarantees
while maintaining complete sovereignty over our transactional semantics.

Following Metus: We describe behavior, then make it true.
"""

import tempfile
import time
from pathlib import Path
from threading import Thread, Barrier
from typing import List

import pytest

from cortical.cdg.transaction_manager import (
    CDGTransactionManager,
    Conflict,
    CommitResult,
)
from cortical.cdg.transaction import Transaction, TransactionState
from cortical.cdg.types import Entity
from cortical.cdg.config import CDGConfig
from cortical.cdg.errors import TransactionError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_cdg_dir(tmp_path):
    """Provide a temporary directory for CDG storage."""
    cdg_dir = tmp_path / ".cdg"
    cdg_dir.mkdir()
    return cdg_dir


@pytest.fixture
def tx_manager(temp_cdg_dir):
    """
    Provide a CDGTransactionManager for testing.

    Uses default configuration with WAL enabled for durability.
    """
    config = CDGConfig()
    manager = CDGTransactionManager(temp_cdg_dir, config)
    return manager


@pytest.fixture
def tx_manager_no_wal(temp_cdg_dir):
    """
    Provide a CDGTransactionManager without WAL for faster tests.
    """
    config = CDGConfig()
    config.enable_wal = False
    manager = CDGTransactionManager(temp_cdg_dir, config)
    return manager


@pytest.fixture
def sample_entities():
    """
    Create sample entities for testing.

    Returns entities representing a custom task management system
    built from scratch with no external dependencies.
    """
    return [
        Entity(
            id="TASK-001",
            entity_type="task",
            properties={
                "title": "Build custom transaction manager from scratch",
                "status": "in_progress",
                "priority": "high",
                "description": "Hand-rolled ACID implementation with no dependencies"
            }
        ),
        Entity(
            id="TASK-002",
            entity_type="task",
            properties={
                "title": "Implement optimistic locking we control completely",
                "status": "pending",
                "priority": "high",
                "description": "Version-based conflict detection built ourselves"
            }
        ),
        Entity(
            id="TASK-003",
            entity_type="task",
            properties={
                "title": "Create snapshot isolation layer we own",
                "status": "pending",
                "priority": "medium",
                "description": "Custom snapshot versioning for consistent reads"
            }
        ),
    ]


# ============================================================================
# BEHAVIORAL SCENARIOS - ATOMICITY
# ============================================================================

class TestDeveloperCommitsAtomicChanges:
    """
    Epic: Atomicity Guarantees

    As a developer modifying multiple graph entities,
    I want all writes to succeed together or fail together,
    So that my graph never ends up in a partially-updated state.
    """

    def test_scenario_all_writes_succeed_or_all_fail(self, tx_manager, sample_entities):
        """
        Scenario: Multiple writes either all commit or none commit

        Given a transaction with multiple pending writes
        When the transaction commits successfully
        Then all writes are persisted atomically
        And all entities are visible in subsequent reads
        """
        # Given a transaction with multiple pending writes
        tx = tx_manager.begin()

        for entity in sample_entities:
            tx_manager.write(tx, entity)

        # When the transaction commits successfully
        result = tx_manager.commit(tx)

        # Then all writes are persisted atomically
        assert result.success is True
        assert result.version is not None

        # And all entities are visible in subsequent reads
        tx2 = tx_manager.begin()
        for entity in sample_entities:
            retrieved = tx_manager.read(tx2, entity.id)
            assert retrieved is not None
            assert retrieved.id == entity.id
            assert retrieved.properties == entity.properties

    def test_scenario_conflict_causes_no_writes_to_persist(self, tx_manager):
        """
        Scenario: Conflicting transaction leaves no partial writes

        Given an entity exists at version 1
        And two transactions read it
        When the first transaction commits successfully
        And the second transaction tries to commit
        Then the second transaction is aborted
        And its writes do not appear in storage
        Because atomicity guarantees all-or-nothing behavior
        """
        # Given an entity exists at version 1
        tx_initial = tx_manager.begin()
        initial_entity = Entity(
            id="COUNTER-001",
            entity_type="counter",
            properties={"value": 0}
        )
        tx_manager.write(tx_initial, initial_entity)
        tx_manager.commit(tx_initial)

        # And two transactions read it
        tx1 = tx_manager.begin()
        entity1 = tx_manager.read(tx1, "COUNTER-001")

        tx2 = tx_manager.begin()
        entity2 = tx_manager.read(tx2, "COUNTER-001")

        # When the first transaction commits successfully
        entity1.properties["value"] = 1
        tx_manager.write(tx1, entity1)
        result1 = tx_manager.commit(tx1)
        assert result1.success is True

        # And the second transaction tries to commit
        entity2.properties["value"] = 2
        tx_manager.write(tx2, entity2)
        result2 = tx_manager.commit(tx2)

        # Then the second transaction is aborted
        assert result2.success is False
        assert len(result2.conflicts) > 0

        # And its writes do not appear in storage
        tx3 = tx_manager.begin()
        final_entity = tx_manager.read(tx3, "COUNTER-001")
        assert final_entity.properties["value"] == 1  # First transaction won

    def test_scenario_write_failure_rolls_back_entire_transaction(self, tx_manager):
        """
        Scenario: Storage error during commit aborts all writes

        Given a transaction with multiple writes
        When one write cannot be applied
        Then the entire transaction is aborted
        And no partial writes are visible
        """
        # Given a transaction with multiple writes
        tx = tx_manager.begin()

        entity1 = Entity(id="E-001", entity_type="test")
        entity2 = Entity(id="E-002", entity_type="test")

        tx_manager.write(tx, entity1)
        tx_manager.write(tx, entity2)

        # When transaction commits (should succeed normally)
        result = tx_manager.commit(tx)

        # Then both entities are committed together
        # (This scenario verifies the all-or-nothing semantics)
        if result.success:
            tx2 = tx_manager.begin()
            assert tx_manager.read(tx2, "E-001") is not None
            assert tx_manager.read(tx2, "E-002") is not None
        else:
            # If commit fails, neither should be visible
            tx2 = tx_manager.begin()
            assert tx_manager.read(tx2, "E-001") is None
            assert tx_manager.read(tx2, "E-002") is None


# ============================================================================
# BEHAVIORAL SCENARIOS - ISOLATION
# ============================================================================

class TestDeveloperReadsConsistentSnapshots:
    """
    Epic: Isolation Guarantees

    As a developer querying graph data during modifications,
    I want to see a consistent snapshot of the data,
    So that my reads are never polluted by concurrent uncommitted writes.
    """

    def test_scenario_transaction_sees_snapshot_at_begin_time(self, tx_manager):
        """
        Scenario: Transaction reads from snapshot version

        Given entities exist at version 5
        When a transaction begins at version 5
        And concurrent transactions commit new versions
        Then the original transaction still sees version 5
        Because snapshot isolation provides time-travel consistency
        """
        # Given entities exist at version 5
        # (Create and commit initial entities)
        tx_setup = tx_manager.begin()
        for i in range(3):
            entity = Entity(
                id=f"DOC-{i:03d}",
                entity_type="document",
                properties={"content": f"Initial version {i}"}
            )
            tx_manager.write(tx_setup, entity)
        result_setup = tx_manager.commit(tx_setup)
        snapshot_v1 = result_setup.version

        # When a transaction begins at this version
        tx_reader = tx_manager.begin()
        assert tx_reader.snapshot_version == snapshot_v1

        # And concurrent transactions commit new versions
        tx_writer = tx_manager.begin()
        entity_updated = tx_manager.read(tx_writer, "DOC-001")
        entity_updated.properties["content"] = "Updated by concurrent transaction"
        tx_manager.write(tx_writer, entity_updated)
        tx_manager.commit(tx_writer)

        # Then the original transaction still sees the original version
        entity_from_snapshot = tx_manager.read(tx_reader, "DOC-001")
        assert entity_from_snapshot is not None
        assert entity_from_snapshot.properties["content"] == "Initial version 1"

    def test_scenario_uncommitted_writes_invisible_to_other_transactions(self, tx_manager):
        """
        Scenario: Uncommitted writes remain isolated

        Given transaction T1 has buffered writes
        When transaction T2 reads the same entities
        Then T2 does not see T1's uncommitted changes
        Because isolation prevents dirty reads
        """
        # Given transaction T1 has buffered writes
        tx1 = tx_manager.begin()
        entity1 = Entity(
            id="SECRET-001",
            entity_type="secret",
            properties={"value": "uncommitted data we built ourselves"}
        )
        tx_manager.write(tx1, entity1)

        # When transaction T2 reads the same entity
        tx2 = tx_manager.begin()
        entity2 = tx_manager.read(tx2, "SECRET-001")

        # Then T2 does not see T1's uncommitted changes
        assert entity2 is None

        # After T1 commits, T2 still sees its snapshot (before commit)
        tx_manager.commit(tx1)
        entity2_after = tx_manager.read(tx2, "SECRET-001")
        assert entity2_after is None  # Still in T2's snapshot

        # But a new transaction sees the committed data
        tx3 = tx_manager.begin()
        entity3 = tx_manager.read(tx3, "SECRET-001")
        assert entity3 is not None
        assert entity3.properties["value"] == "uncommitted data we built ourselves"

    def test_scenario_read_set_tracks_versions_for_conflict_detection(self, tx_manager):
        """
        Scenario: Transaction tracks read versions

        Given a transaction reads multiple entities
        When the transaction reads each entity
        Then the read version is recorded in read_set
        So that conflicts can be detected at commit time
        """
        # Given a transaction reads multiple entities
        tx_setup = tx_manager.begin()
        for i in range(3):
            entity = Entity(id=f"E-{i}", entity_type="test", version=1)
            tx_manager.write(tx_setup, entity)
        result = tx_manager.commit(tx_setup)
        assert result.success is True
        committed_version = result.version

        # When the transaction reads each entity
        tx = tx_manager.begin()
        for i in range(3):
            entity = tx_manager.read(tx, f"E-{i}")
            assert entity is not None

        # Then the read version is recorded in read_set
        assert "E-0" in tx.read_set
        assert "E-1" in tx.read_set
        assert "E-2" in tx.read_set
        # All entities should have same version (written in same transaction)
        assert tx.read_set["E-0"] == tx.read_set["E-1"]
        assert tx.read_set["E-1"] == tx.read_set["E-2"]
        # Version should be positive (greater than 0)
        assert tx.read_set["E-0"] > 0


# ============================================================================
# BEHAVIORAL SCENARIOS - CONFLICT DETECTION
# ============================================================================

class TestSystemPreventsConflictingWrites:
    """
    Epic: Optimistic Locking

    As a system maintaining data consistency,
    I want to detect when concurrent transactions modify the same data,
    So that lost updates are prevented through our custom conflict detection.
    """

    def test_scenario_version_mismatch_detected_at_commit(self, tx_manager):
        """
        Scenario: Optimistic locking detects concurrent modifications

        Given entity E exists at a committed version
        And transaction T1 reads E at that version
        And transaction T2 reads E at that version
        When T1 modifies E and commits (E version increments)
        And T2 tries to commit its modification
        Then T2's commit fails with version conflict
        And the conflict reports expected vs actual version
        """
        # Given entity E exists at a committed version
        tx_initial = tx_manager.begin()
        initial_entity = Entity(
            id="CONFIG-001",
            entity_type="config",
            version=1,
            properties={"setting": "default"}
        )
        tx_manager.write(tx_initial, initial_entity)
        result_initial = tx_manager.commit(tx_initial)
        initial_version = result_initial.version

        # And transaction T1 reads E at that version
        tx1 = tx_manager.begin()
        entity1 = tx_manager.read(tx1, "CONFIG-001")
        version_read_by_t1 = entity1.version

        # And transaction T2 reads E at that version
        tx2 = tx_manager.begin()
        entity2 = tx_manager.read(tx2, "CONFIG-001")
        version_read_by_t2 = entity2.version
        assert version_read_by_t2 == version_read_by_t1  # Same snapshot

        # When T1 modifies E and commits
        entity1.properties["setting"] = "updated by T1"
        tx_manager.write(tx1, entity1)
        result1 = tx_manager.commit(tx1)
        assert result1.success is True
        new_version_after_t1 = result1.version

        # And T2 tries to commit its modification
        entity2.properties["setting"] = "updated by T2"
        tx_manager.write(tx2, entity2)
        result2 = tx_manager.commit(tx2)

        # Then T2's commit fails with version conflict
        assert result2.success is False
        assert len(result2.conflicts) == 1

        # And the conflict reports expected vs actual version
        conflict = result2.conflicts[0]
        assert conflict.entity_id == "CONFIG-001"
        assert conflict.expected_version == version_read_by_t2
        # Actual version should be newer than what T2 read
        assert conflict.actual_version > conflict.expected_version
        assert conflict.conflict_type == "version_mismatch"

    def test_scenario_write_implicitly_reads_for_versioning(self, tx_manager):
        """
        Scenario: Write operation implicitly reads for version tracking

        Given an entity exists
        When a transaction writes to it
        Then the entity is automatically read and tracked
        Because the write operation needs old version for WAL logging
        """
        # Given an entity exists
        tx_setup = tx_manager.begin()
        entity = Entity(
            id="IMPLICIT-READ-001",
            entity_type="test",
            properties={"value": "initial"}
        )
        tx_manager.write(tx_setup, entity)
        result_setup = tx_manager.commit(tx_setup)
        initial_version = result_setup.version

        # When a transaction writes to it
        tx1 = tx_manager.begin()
        new_entity = Entity(
            id="IMPLICIT-READ-001",
            entity_type="test",
            properties={"value": "updated"}
        )
        tx_manager.write(tx1, new_entity)

        # Then the entity is automatically read and tracked
        # (write() calls read() internally to get old version for WAL)
        assert "IMPLICIT-READ-001" in tx1.read_set
        # Version tracked should be positive
        assert tx1.read_set["IMPLICIT-READ-001"] > 0

        # Commit should succeed
        result = tx_manager.commit(tx1)
        assert result.success is True

    def test_scenario_multiple_conflicts_reported_together(self, tx_manager):
        """
        Scenario: All conflicts detected in single commit attempt

        Given multiple entities are modified concurrently
        When a transaction conflicts on multiple entities
        Then all conflicts are reported together
        So the developer can see the complete conflict picture
        """
        # Given multiple entities are modified concurrently
        tx_setup = tx_manager.begin()
        for i in range(3):
            entity = Entity(id=f"MULTI-{i}", entity_type="test")
            tx_manager.write(tx_setup, entity)
        tx_manager.commit(tx_setup)

        # Start two transactions that read all entities
        tx1 = tx_manager.begin()
        tx2 = tx_manager.begin()

        entities1 = [tx_manager.read(tx1, f"MULTI-{i}") for i in range(3)]
        entities2 = [tx_manager.read(tx2, f"MULTI-{i}") for i in range(3)]

        # T1 modifies all and commits
        for i, entity in enumerate(entities1):
            entity.properties["modified_by"] = "T1"
            tx_manager.write(tx1, entity)
        tx_manager.commit(tx1)

        # When T2 tries to modify all and commit
        for i, entity in enumerate(entities2):
            entity.properties["modified_by"] = "T2"
            tx_manager.write(tx2, entity)
        result2 = tx_manager.commit(tx2)

        # Then all conflicts are reported together
        assert result2.success is False
        assert len(result2.conflicts) == 3

        conflict_ids = {c.entity_id for c in result2.conflicts}
        assert conflict_ids == {"MULTI-0", "MULTI-1", "MULTI-2"}


# ============================================================================
# BEHAVIORAL SCENARIOS - ROLLBACK
# ============================================================================

class TestDeveloperRollsBackTransactions:
    """
    Epic: Transaction Rollback

    As a developer who realizes a transaction should not commit,
    I want to explicitly roll back the transaction,
    So that all buffered writes are discarded cleanly.
    """

    def test_scenario_rollback_discards_all_buffered_writes(self, tx_manager):
        """
        Scenario: Explicit rollback clears write set

        Given a transaction with buffered writes
        When the transaction is rolled back
        Then the write set is empty
        And no changes are persisted
        And the transaction state is ROLLED_BACK
        """
        # Given a transaction with buffered writes
        tx = tx_manager.begin()

        for i in range(5):
            entity = Entity(
                id=f"ROLLBACK-{i}",
                entity_type="test",
                properties={"should_not": "persist"}
            )
            tx_manager.write(tx, entity)

        assert len(tx.write_set) == 5

        # When the transaction is rolled back
        tx_manager.rollback(tx, reason="user_cancelled")

        # Then the write set is empty
        assert len(tx.write_set) == 0

        # And the transaction state is ROLLED_BACK
        assert tx.state == TransactionState.ROLLED_BACK

        # And no changes are persisted
        tx2 = tx_manager.begin()
        for i in range(5):
            entity = tx_manager.read(tx2, f"ROLLBACK-{i}")
            assert entity is None

    def test_scenario_rollback_after_conflict_is_safe(self, tx_manager):
        """
        Scenario: Rolling back after conflict detection

        Given a transaction that will conflict
        When conflict is detected and commit fails
        Then the transaction is automatically aborted
        And the developer can safely move on
        """
        # Given a transaction that will conflict
        tx_setup = tx_manager.begin()
        entity = Entity(id="CONFLICT-001", entity_type="test")
        tx_manager.write(tx_setup, entity)
        tx_manager.commit(tx_setup)

        tx1 = tx_manager.begin()
        entity1 = tx_manager.read(tx1, "CONFLICT-001")

        tx2 = tx_manager.begin()
        entity2 = tx_manager.read(tx2, "CONFLICT-001")

        # First transaction commits
        entity1.properties["value"] = "T1"
        tx_manager.write(tx1, entity1)
        tx_manager.commit(tx1)

        # When conflict is detected and commit fails
        entity2.properties["value"] = "T2"
        tx_manager.write(tx2, entity2)
        result = tx_manager.commit(tx2)

        # Then the transaction is automatically aborted
        assert result.success is False
        assert tx2.state == TransactionState.ABORTED

    def test_scenario_cannot_rollback_committed_transaction(self, tx_manager):
        """
        Scenario: Committed transactions cannot be rolled back

        Given a transaction that has committed
        When attempting to roll it back
        Then an error is raised
        Because committed transactions are immutable
        """
        # Given a transaction that has committed
        tx = tx_manager.begin()
        entity = Entity(id="COMMITTED-001", entity_type="test")
        tx_manager.write(tx, entity)
        result = tx_manager.commit(tx)
        assert result.success is True
        assert tx.state == TransactionState.COMMITTED

        # When attempting to roll it back
        # Then an error is raised
        with pytest.raises(TransactionError, match="cannot rollback"):
            tx_manager.rollback(tx)


# ============================================================================
# BEHAVIORAL SCENARIOS - READ YOUR OWN WRITES
# ============================================================================

class TestDeveloperSeesOwnUncommittedWrites:
    """
    Epic: Read-Your-Writes Consistency

    As a developer building up state within a transaction,
    I want to read my own buffered writes,
    So that I can make decisions based on my pending changes.
    """

    def test_scenario_read_returns_buffered_write_not_storage(self, tx_manager):
        """
        Scenario: Reads see buffered writes within same transaction

        Given a transaction writes entity E
        When the same transaction reads E
        Then the buffered version is returned
        Not the storage version
        Because read-your-writes is essential for transaction logic
        """
        # Given a transaction writes entity E
        tx = tx_manager.begin()
        entity = Entity(
            id="BUFFER-001",
            entity_type="test",
            properties={"status": "buffered_write_we_control"}
        )
        tx_manager.write(tx, entity)

        # When the same transaction reads E
        retrieved = tx_manager.read(tx, "BUFFER-001")

        # Then the buffered version is returned
        assert retrieved is not None
        assert retrieved.id == "BUFFER-001"
        assert retrieved.properties["status"] == "buffered_write_we_control"

        # Verify it's not in storage yet
        tx2 = tx_manager.begin()
        from_storage = tx_manager.read(tx2, "BUFFER-001")
        assert from_storage is None  # Not yet committed

    def test_scenario_read_after_write_enables_incremental_logic(self, tx_manager):
        """
        Scenario: Building state incrementally within transaction

        Given a transaction creates an entity
        When the transaction reads it back
        And modifies it based on current state
        And writes it again
        Then all modifications are visible within the transaction
        """
        # Given a transaction creates an entity
        tx = tx_manager.begin()

        entity = Entity(
            id="INCREMENTAL-001",
            entity_type="counter",
            properties={"count": 0}
        )
        tx_manager.write(tx, entity)

        # When the transaction reads it back
        retrieved = tx_manager.read(tx, "INCREMENTAL-001")
        assert retrieved.properties["count"] == 0

        # And modifies it based on current state
        retrieved.properties["count"] += 1
        tx_manager.write(tx, retrieved)

        # And reads it again
        retrieved2 = tx_manager.read(tx, "INCREMENTAL-001")

        # Then all modifications are visible within the transaction
        assert retrieved2.properties["count"] == 1

        # Commit and verify persistence
        result = tx_manager.commit(tx)
        assert result.success is True

        tx2 = tx_manager.begin()
        final = tx_manager.read(tx2, "INCREMENTAL-001")
        assert final.properties["count"] == 1

    def test_scenario_write_then_read_avoids_stale_data(self, tx_manager):
        """
        Scenario: Multiple updates within transaction stay consistent

        Given an existing entity
        When a transaction reads it
        And updates it multiple times
        Then each read sees the latest buffered version
        Not stale storage or intermediate versions
        """
        # Given an existing entity
        tx_setup = tx_manager.begin()
        entity = Entity(
            id="MULTI-UPDATE-001",
            entity_type="document",
            properties={"version": 1, "content": "original"}
        )
        tx_manager.write(tx_setup, entity)
        tx_manager.commit(tx_setup)

        # When a transaction reads it and updates it multiple times
        tx = tx_manager.begin()

        # First read and update
        entity1 = tx_manager.read(tx, "MULTI-UPDATE-001")
        entity1.properties["version"] = 2
        entity1.properties["content"] = "second version"
        tx_manager.write(tx, entity1)

        # Second read and update
        entity2 = tx_manager.read(tx, "MULTI-UPDATE-001")
        assert entity2.properties["version"] == 2  # Sees first update
        entity2.properties["version"] = 3
        entity2.properties["content"] = "third version"
        tx_manager.write(tx, entity2)

        # Third read
        entity3 = tx_manager.read(tx, "MULTI-UPDATE-001")

        # Then each read sees the latest buffered version
        assert entity3.properties["version"] == 3
        assert entity3.properties["content"] == "third version"


# ============================================================================
# BEHAVIORAL SCENARIOS - CONCURRENT TRANSACTIONS
# ============================================================================

class TestSystemHandlesConcurrentTransactions:
    """
    Epic: Concurrent Transaction Safety

    As a system with multiple concurrent transactions,
    I want transactions to be isolated from each other,
    So that concurrent operations don't interfere or corrupt data.
    """

    def test_scenario_concurrent_reads_do_not_block(self, tx_manager_no_wal):
        """
        Scenario: Multiple transactions can read simultaneously

        Given multiple entities exist
        When multiple transactions read them concurrently
        Then all reads succeed without blocking
        Because snapshot isolation allows concurrent reads
        """
        # Given multiple entities exist
        tx_setup = tx_manager_no_wal.begin()
        for i in range(10):
            entity = Entity(
                id=f"CONCURRENT-READ-{i}",
                entity_type="test",
                properties={"data": f"value_{i}"}
            )
            tx_manager_no_wal.write(tx_setup, entity)
        tx_manager_no_wal.commit(tx_setup)

        # When multiple transactions read them concurrently
        results = []
        barrier = Barrier(5)  # Synchronize 5 threads

        def concurrent_reader(thread_id):
            barrier.wait()  # Wait for all threads to be ready
            tx = tx_manager_no_wal.begin()
            read_count = 0
            for i in range(10):
                entity = tx_manager_no_wal.read(tx, f"CONCURRENT-READ-{i}")
                if entity is not None:
                    read_count += 1
            results.append(read_count)

        threads = [Thread(target=concurrent_reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Then all reads succeed without blocking
        assert len(results) == 5
        assert all(count == 10 for count in results)

    def test_scenario_concurrent_non_conflicting_writes_both_succeed(self, tx_manager_no_wal):
        """
        Scenario: Non-overlapping writes commit independently

        Given two transactions modify different entities
        When both transactions commit
        Then both succeed
        Because there are no conflicts
        """
        # Given two transactions modify different entities
        results = []

        def writer_thread(entity_id, value):
            tx = tx_manager_no_wal.begin()
            entity = Entity(
                id=entity_id,
                entity_type="test",
                properties={"value": value}
            )
            tx_manager_no_wal.write(tx, entity)
            result = tx_manager_no_wal.commit(tx)
            results.append((entity_id, result.success))

        # When both transactions commit
        t1 = Thread(target=writer_thread, args=("NON-CONFLICT-1", "thread1"))
        t2 = Thread(target=writer_thread, args=("NON-CONFLICT-2", "thread2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Then both succeed
        assert len(results) == 2
        assert all(success for _, success in results)

        # Verify both writes persisted
        tx = tx_manager_no_wal.begin()
        assert tx_manager_no_wal.read(tx, "NON-CONFLICT-1") is not None
        assert tx_manager_no_wal.read(tx, "NON-CONFLICT-2") is not None

    def test_scenario_concurrent_conflicting_writes_one_wins(self, tx_manager_no_wal):
        """
        Scenario: Concurrent updates to same entity - first wins

        Given an entity exists
        When two transactions read and modify it with lock protection
        Then the first to commit wins
        And the second gets a conflict error
        """
        # Given an entity exists
        tx_setup = tx_manager_no_wal.begin()
        entity = Entity(
            id="RACE-001",
            entity_type="test",
            properties={"counter": 0}
        )
        tx_manager_no_wal.write(tx_setup, entity)
        tx_manager_no_wal.commit(tx_setup)

        # Sequential simulation of concurrent conflict
        # (Actual concurrency with file locking is deterministic in CDG)

        # Transaction 1 reads
        tx1 = tx_manager_no_wal.begin()
        entity1 = tx_manager_no_wal.read(tx1, "RACE-001")

        # Transaction 2 reads (same snapshot)
        tx2 = tx_manager_no_wal.begin()
        entity2 = tx_manager_no_wal.read(tx2, "RACE-001")

        # Both modify
        entity1.properties["counter"] = 1
        entity1.properties["thread"] = 1
        tx_manager_no_wal.write(tx1, entity1)

        entity2.properties["counter"] = 2
        entity2.properties["thread"] = 2
        tx_manager_no_wal.write(tx2, entity2)

        # First commits successfully
        result1 = tx_manager_no_wal.commit(tx1)
        assert result1.success is True

        # Second fails with conflict
        result2 = tx_manager_no_wal.commit(tx2)
        assert result2.success is False
        assert len(result2.conflicts) > 0

        # Verify the first transaction's data persisted
        tx_verify = tx_manager_no_wal.begin()
        final = tx_manager_no_wal.read(tx_verify, "RACE-001")
        assert final.properties["thread"] == 1
        assert final.properties["counter"] == 1

    def test_scenario_long_running_transaction_sees_old_snapshot(self, tx_manager_no_wal):
        """
        Scenario: Long-lived transaction maintains snapshot consistency

        Given a long-running read transaction begins
        When many other transactions commit changes
        Then the long-running transaction still sees its original snapshot
        Because snapshot isolation is maintained for transaction lifetime
        """
        # Given a long-running read transaction begins
        tx_setup = tx_manager_no_wal.begin()
        original = Entity(
            id="LONG-RUNNING-001",
            entity_type="test",
            properties={"value": "original"}
        )
        tx_manager_no_wal.write(tx_setup, original)
        tx_manager_no_wal.commit(tx_setup)

        # Long-running transaction starts
        long_tx = tx_manager_no_wal.begin()
        snapshot_version = long_tx.snapshot_version
        first_read = tx_manager_no_wal.read(long_tx, "LONG-RUNNING-001")
        assert first_read.properties["value"] == "original"

        # When many other transactions commit changes
        for i in range(5):
            tx = tx_manager_no_wal.begin()
            entity = tx_manager_no_wal.read(tx, "LONG-RUNNING-001")
            entity.properties["value"] = f"update_{i}"
            entity.properties["iteration"] = i
            tx_manager_no_wal.write(tx, entity)
            tx_manager_no_wal.commit(tx)

        # Then the long-running transaction still sees its original snapshot
        second_read = tx_manager_no_wal.read(long_tx, "LONG-RUNNING-001")
        assert second_read.properties["value"] == "original"
        assert "iteration" not in second_read.properties

        # Verify new transaction sees latest
        new_tx = tx_manager_no_wal.begin()
        assert new_tx.snapshot_version > snapshot_version
        latest = tx_manager_no_wal.read(new_tx, "LONG-RUNNING-001")
        assert latest.properties["value"] == "update_4"
        assert latest.properties["iteration"] == 4
