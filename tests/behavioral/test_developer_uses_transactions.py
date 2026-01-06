"""
Behavioral tests for transactional operations in Graph of Thought.

Epic: Developer Uses Transactional System

As a developer ensuring data integrity,
I want to use transactions for atomic operations,
So that my custom-built graph database maintains consistency.
"""

import pytest
from cortical.got.api import GoTManager
from cortical.got.errors import TransactionError
from tests.conftest import _create_tx_manager, _create_got_manager


class TestDeveloperExecutesAtomicOperations:
    """
    As a developer performing multi-step operations,
    I want transactions that commit or rollback atomically,
    So that our hand-built database maintains consistency.
    """

    def test_scenario_successful_transaction_commits_changes(self, tmp_path):
        """
        Scenario: Committing a successful transaction

        Given a transaction context
        When I create multiple entities within it
        And the transaction completes successfully
        Then all changes are persisted
        """
        # Given a transaction context
        manager = _create_got_manager(tmp_path / ".got")

        # When I create multiple entities within it
        with manager.transaction() as tx:
            task1 = tx.create_task(title="Task 1")
            task2 = tx.create_task(title="Task 2")
            edge = tx.add_edge(task1.id, task2.id, "DEPENDS_ON")
            # And the transaction completes successfully (implicit commit on exit)

        # Then all changes are persisted
        assert manager.get_task(task1.id) is not None
        assert manager.get_task(task2.id) is not None

        outgoing, incoming = manager.get_edges_for_task(task1.id)
        assert len(outgoing) == 1
        assert outgoing[0].target_id == task2.id

    def test_scenario_exception_rolls_back_transaction(self, tmp_path):
        """
        Scenario: Rolling back on exception

        Given a transaction that encounters an error
        When an exception is raised within the transaction
        Then all changes are rolled back
        And nothing is persisted
        """
        # Given a transaction that encounters an error
        manager = _create_got_manager(tmp_path / ".got")
        task_id = None

        try:
            # When an exception is raised within the transaction
            with manager.transaction() as tx:
                task = tx.create_task(title="Will be rolled back")
                task_id = task.id
                # Simulate an error
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # Then all changes are rolled back
        # And nothing is persisted
        assert manager.get_task(task_id) is None

    def test_scenario_read_only_transaction_never_commits(self, tmp_path):
        """
        Scenario: Using read-only transactions for queries

        Given a read-only transaction context
        When I perform operations within it
        Then the operations succeed during the transaction
        But no changes are persisted
        """
        # Given a read-only transaction context
        manager = _create_got_manager(tmp_path / ".got")

        existing_task = manager.create_task(title="Existing task")
        task_id = None

        # When I perform operations within it
        with manager.transaction(read_only=True) as tx:
            # Can read
            task = tx.get_task(existing_task.id)
            assert task is not None

            # Can create (but won't be committed)
            new_task = tx.create_task(title="Not persisted")
            task_id = new_task.id

            # Then the operations succeed during the transaction
            assert tx.get_task(task_id) is not None  # Visible in transaction

        # But no changes are persisted
        assert manager.get_task(task_id) is None

    def test_scenario_update_task_within_transaction(self, tmp_path):
        """
        Scenario: Updating entities transactionally

        Given an existing task
        When I update it within a transaction
        Then the updates are committed atomically
        """
        # Given an existing task
        manager = _create_got_manager(tmp_path / ".got")
        task = manager.create_task(title="Original", status="pending")

        # When I update it within a transaction
        with manager.transaction() as tx:
            updated = tx.update_task(
                task.id,
                title="Updated title",
                status="in_progress",
                priority="high"
            )

        # Then the updates are committed atomically
        retrieved = manager.get_task(task.id)
        assert retrieved.title == "Updated title"
        assert retrieved.status == "in_progress"
        assert retrieved.priority == "high"


class TestDeveloperUsesCachingForPerformance:
    """
    As a developer optimizing query performance,
    I want entity caching for faster repeated queries,
    So that our custom database doesn't hit disk unnecessarily.
    """

    def test_scenario_cached_reads_faster_than_disk_reads(self, tmp_path):
        """
        Scenario: Benefiting from entity caching

        Given a task that's been read once
        When I read it again
        Then it comes from cache
        And cache hit statistics increase
        """
        # Given a task that's been read once
        manager = _create_got_manager(tmp_path / ".got")
        task = manager.create_task(title="Cached task")

        # First read populates cache
        manager.get_task(task.id)
        stats_before = manager.cache_stats()

        # When I read it again
        manager.get_task(task.id)
        stats_after = manager.cache_stats()

        # Then cache hit statistics increase
        assert stats_after['hits'] > stats_before['hits']

    def test_scenario_cache_invalidated_on_write(self, tmp_path):
        """
        Scenario: Cache invalidation maintains consistency

        Given a cached task
        When I update it
        Then the cache is invalidated
        And subsequent reads get fresh data
        """
        # Given a cached task
        manager = _create_got_manager(tmp_path / ".got")
        task = manager.create_task(title="Original")

        # Read to populate cache
        cached = manager.get_task(task.id)
        assert cached.title == "Original"

        # When I update it
        manager.update_task(task.id, title="Updated")

        # Then subsequent reads get fresh data
        fresh = manager.get_task(task.id)
        assert fresh.title == "Updated"

    @pytest.mark.skip(reason="cache_configure() moved to CDGStore - TTL/max_size not yet implemented")
    def test_scenario_configure_cache_with_ttl_and_size(self, tmp_path):
        """
        Scenario: Configuring cache behavior

        Given a GoT manager
        When I configure cache TTL and max size
        Then the cache respects those limits

        Note: This functionality requires TTL/max_size support in CDGStore,
        which is not yet implemented.
        """
        pass

    def test_scenario_cache_can_be_cleared_manually(self, tmp_path):
        """
        Scenario: Manual cache clearing

        Given a cache with entries
        When I clear the cache
        Then all entries are removed
        And statistics are reset
        """
        # Given a cache with entries
        manager = _create_got_manager(tmp_path / ".got")
        task = manager.create_task(title="Task")
        manager.get_task(task.id)  # Populate cache

        stats_before = manager.cache_stats()
        assert stats_before['size'] > 0

        # When I clear the cache
        manager.cache_clear()

        # Then all entries are removed
        # And statistics are reset
        stats_after = manager.cache_stats()
        assert stats_after['size'] == 0
        assert stats_after['hits'] == 0
        assert stats_after['misses'] == 0


class TestDeveloperPreloadsEntitiesForFastQueries:
    """
    As a developer optimizing batch operations,
    I want to pre-load all entities into memory,
    So that subsequent queries are sub-millisecond.
    """

    def test_scenario_load_all_entities_for_fast_access(self, tmp_path):
        """
        Scenario: Pre-loading entities for read-heavy workloads

        Given a graph with many entities
        When I call load_all()
        Then all entities are loaded into cache
        And I get counts of what was loaded
        """
        # Given a graph with many entities
        manager = _create_got_manager(tmp_path / ".got")
        task1 = manager.create_task(title="Task 1")
        task2 = manager.create_task(title="Task 2")
        sprint = manager.create_sprint(title="Sprint 1")
        edge = manager.add_task_to_sprint(task1.id, sprint.id)

        # When I call load_all()
        counts = manager.load_all()

        # Then all entities are loaded into cache
        assert counts['tasks'] == 2
        assert counts['sprints'] == 1
        assert counts['edges'] == 1

        # And subsequent queries hit cache
        stats_before = manager.cache_stats()
        manager.get_task(task1.id)
        stats_after = manager.cache_stats()
        assert stats_after['hits'] > stats_before['hits']


class TestDeveloperHandlesTransactionFailures:
    """
    As a developer dealing with errors,
    I want clear error messages on transaction failures,
    So that I can debug issues in our custom transaction system.
    """

    def test_scenario_updating_nonexistent_task_fails_clearly(self, tmp_path):
        """
        Scenario: Clear error on missing entity

        Given a transaction
        When I try to update a non-existent task
        Then I get a clear error message
        And the transaction rolls back
        """
        # Given a transaction
        manager = _create_got_manager(tmp_path / ".got")

        # When I try to update a non-existent task
        # Then I get a clear error message
        with pytest.raises(TransactionError, match="Task not found"):
            with manager.transaction() as tx:
                tx.update_task("T-nonexistent-id", status="completed")

    def test_scenario_transaction_sees_own_writes(self, tmp_path):
        """
        Scenario: Read-your-own-writes consistency

        Given an active transaction
        When I create an entity
        Then I can immediately read it within the transaction
        Even before commit
        """
        # Given an active transaction
        manager = _create_got_manager(tmp_path / ".got")

        with manager.transaction() as tx:
            # When I create an entity
            task = tx.create_task(title="New task")

            # Then I can immediately read it within the transaction
            read_back = tx.get_task(task.id)
            assert read_back is not None
            assert read_back.id == task.id
            assert read_back.title == "New task"


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
