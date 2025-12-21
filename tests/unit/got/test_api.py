"""
Tests for GoT high-level API (GoTManager and TransactionContext).

Tests context manager behavior, single-operation methods, and error handling.
"""

import tempfile
import pytest
from pathlib import Path

from cortical.got.api import GoTManager, TransactionContext, generate_task_id, generate_decision_id
from cortical.got.types import Task, Decision, Edge
from cortical.got.errors import TransactionError


class TestGoTManager:
    """Tests for GoTManager high-level API."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        """Provide temporary GoT directory."""
        return tmp_path / ".got"

    @pytest.fixture
    def manager(self, got_dir):
        """Provide GoTManager instance."""
        return GoTManager(got_dir)

    def test_context_manager_commits_on_success(self, manager):
        """Context manager commits transaction on successful exit."""
        # Create task in transaction
        with manager.transaction() as tx:
            task = tx.create_task("Test task", priority="high")
            task_id = task.id

        # Verify task persisted
        retrieved = manager.get_task(task_id)
        assert retrieved is not None
        assert retrieved.title == "Test task"
        assert retrieved.priority == "high"

    def test_context_manager_rolls_back_on_exception(self, manager):
        """Context manager rolls back transaction on exception."""
        task_id = None

        try:
            with manager.transaction() as tx:
                task = tx.create_task("Test task")
                task_id = task.id
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # Verify task was NOT persisted
        retrieved = manager.get_task(task_id)
        assert retrieved is None

    def test_create_task_in_transaction(self, manager):
        """Task creation within transaction works correctly."""
        with manager.transaction() as tx:
            task = tx.create_task(
                "Implement feature",
                priority="high",
                status="in_progress",
                description="Test description"
            )

            assert task.title == "Implement feature"
            assert task.priority == "high"
            assert task.status == "in_progress"
            assert task.description == "Test description"
            assert task.id.startswith("T-")

    def test_update_task_in_transaction(self, manager):
        """Task update within transaction works correctly."""
        # Create task first
        task = manager.create_task("Original title", status="pending")
        task_id = task.id
        original_version = task.version

        # Update in transaction
        with manager.transaction() as tx:
            updated = tx.update_task(
                task_id,
                title="Updated title",
                status="in_progress",
                priority="critical"
            )

            assert updated.title == "Updated title"
            assert updated.status == "in_progress"
            assert updated.priority == "critical"
            assert updated.version == original_version + 1

        # Verify persistence
        retrieved = manager.get_task(task_id)
        assert retrieved.title == "Updated title"
        assert retrieved.status == "in_progress"

    def test_read_only_context(self, manager):
        """Read-only context rolls back instead of committing."""
        task_id = None

        with manager.transaction(read_only=True) as tx:
            task = tx.create_task("Read-only task")
            task_id = task.id

        # Verify task was NOT persisted
        retrieved = manager.get_task(task_id)
        assert retrieved is None

    def test_get_task_returns_none_for_missing(self, manager):
        """get_task returns None for non-existent task."""
        result = manager.get_task("T-20251221-000000-abcd")
        assert result is None

    def test_single_operation_create_task(self, manager):
        """manager.create_task() single-operation method works."""
        task = manager.create_task(
            "Single-op task",
            priority="high",
            description="Created in one call"
        )

        assert task.title == "Single-op task"
        assert task.priority == "high"
        assert task.description == "Created in one call"

        # Verify persistence
        retrieved = manager.get_task(task.id)
        assert retrieved is not None
        assert retrieved.title == "Single-op task"

    def test_single_operation_update_task(self, manager):
        """manager.update_task() single-operation method works."""
        # Create task
        task = manager.create_task("Original", status="pending")

        # Update using single-operation method
        updated = manager.update_task(
            task.id,
            title="Modified",
            status="completed"
        )

        assert updated.title == "Modified"
        assert updated.status == "completed"

        # Verify persistence
        retrieved = manager.get_task(task.id)
        assert retrieved.title == "Modified"
        assert retrieved.status == "completed"

    def test_add_edge_creates_relationship(self, manager):
        """Edge creation between tasks works correctly."""
        # Create two tasks
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")

        # Create edge
        edge = manager.add_edge(
            task1.id,
            task2.id,
            "DEPENDS_ON",
            weight=0.8
        )

        assert edge.source_id == task1.id
        assert edge.target_id == task2.id
        assert edge.edge_type == "DEPENDS_ON"
        assert edge.weight == 0.8
        assert edge.id == f"E-{task1.id}-{task2.id}-DEPENDS_ON"

    def test_transaction_sees_own_writes(self, manager):
        """Reads within transaction see pending writes."""
        with manager.transaction() as tx:
            # Create task
            task = tx.create_task("Test task")
            task_id = task.id

            # Read it back within same transaction
            retrieved = tx.get_task(task_id)
            assert retrieved is not None
            assert retrieved.title == "Test task"

            # Update it
            tx.update_task(task_id, title="Modified")

            # Read again - should see update
            retrieved2 = tx.get_task(task_id)
            assert retrieved2.title == "Modified"


class TestHelperFunctions:
    """Tests for ID generation helper functions."""

    def test_generate_task_id_format(self):
        """Task ID has correct format T-YYYYMMDD-HHMMSS-XXXX."""
        task_id = generate_task_id()
        assert task_id.startswith("T-")
        parts = task_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 4  # XXXX hex

    def test_generate_decision_id_format(self):
        """Decision ID has correct format D-YYYYMMDD-HHMMSS-XXXX."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("D-")
        parts = decision_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 4  # XXXX hex

    def test_generate_task_id_uniqueness(self):
        """Generated task IDs are unique."""
        ids = [generate_task_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique


class TestDecisionOperations:
    """Tests for decision operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Provide GoTManager instance."""
        return GoTManager(tmp_path / ".got")

    def test_create_decision(self, manager):
        """Decision creation works correctly."""
        decision = manager.create_decision(
            "Use PostgreSQL",
            rationale="Better ACID guarantees",
            affects=["T-001", "T-002"]
        )

        assert decision.title == "Use PostgreSQL"
        assert decision.rationale == "Better ACID guarantees"
        assert decision.affects == ["T-001", "T-002"]
        assert decision.id.startswith("D-")

    def test_create_decision_in_transaction(self, manager):
        """Decision creation in transaction context."""
        with manager.transaction() as tx:
            decision = tx.create_decision(
                "Architectural choice",
                rationale="Simplifies implementation"
            )
            decision_id = decision.id

        # Verify persistence
        with manager.transaction(read_only=True) as tx:
            retrieved = tx.read(decision_id)
            assert retrieved is not None
            assert isinstance(retrieved, Decision)
            assert retrieved.title == "Architectural choice"


class TestErrorHandling:
    """Tests for error handling in API."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Provide GoTManager instance."""
        return GoTManager(tmp_path / ".got")

    def test_update_missing_task_raises_error(self, manager):
        """Updating non-existent task raises TransactionError."""
        with pytest.raises(TransactionError, match="Task not found"):
            with manager.transaction() as tx:
                tx.update_task("T-20251221-000000-abcd", title="New title")

    def test_exception_propagates_from_context(self, manager):
        """Exceptions from within context are propagated."""
        with pytest.raises(ValueError, match="Test error"):
            with manager.transaction() as tx:
                tx.create_task("Test")
                raise ValueError("Test error")
