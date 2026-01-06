"""
Behavioral tests for CDG IndexManager.

These tests define the expected behavior of the CDG indexing system,
which generalizes GoT's QueryIndexManager into a reusable CDG component.

Story: As a developer using CDG, I want to create indexes on entity fields
       so that I can efficiently query entities by field values.

Design Reference: docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md (Section 8)
Replaces: cortical/got/indexer.py (QueryIndexManager)
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List

# These imports will fail until we implement the module
# That's intentional - RED phase of TDD
from cortical.cdg.index import (
    IndexManager,
    IndexEntry,
    IndexType,
    IndexConfig,
)


class TestIndexManagerCreation:
    """Story: Creating and configuring an IndexManager."""

    def test_create_index_manager_with_store_dir(self, tmp_path: Path):
        """Given a store directory, I can create an IndexManager."""
        manager = IndexManager(store_dir=tmp_path)
        assert manager is not None
        assert manager.store_dir == tmp_path

    def test_index_manager_creates_index_directory(self, tmp_path: Path):
        """The IndexManager creates an indexes subdirectory."""
        manager = IndexManager(store_dir=tmp_path)
        assert (tmp_path / "indexes").exists()

    def test_index_manager_with_namespace(self, tmp_path: Path):
        """I can create a namespaced IndexManager for domain separation."""
        manager = IndexManager(store_dir=tmp_path, namespace="got")
        assert manager.namespace == "got"
        # Namespace creates its own subdirectory
        assert (tmp_path / "indexes" / "got").exists()


class TestIndexCreation:
    """Story: Creating indexes on entity fields."""

    def test_create_simple_index(self, tmp_path: Path):
        """I can create an index on a single field."""
        manager = IndexManager(store_dir=tmp_path)

        index = manager.create_index(
            name="status_idx",
            fields=["status"],
        )

        assert index is not None
        assert index.name == "status_idx"
        assert index.fields == ["status"]
        assert manager.has_index("status_idx")

    def test_create_index_with_type(self, tmp_path: Path):
        """I can specify the index type (default is HASH for equality lookups)."""
        manager = IndexManager(store_dir=tmp_path)

        index = manager.create_index(
            name="priority_idx",
            fields=["priority"],
            index_type=IndexType.HASH,
        )

        assert index.index_type == IndexType.HASH

    def test_create_bitmap_index_for_low_cardinality(self, tmp_path: Path):
        """For low-cardinality fields like status, BITMAP index is efficient."""
        manager = IndexManager(store_dir=tmp_path)

        index = manager.create_index(
            name="status_idx",
            fields=["status"],
            index_type=IndexType.BITMAP,
        )

        assert index.index_type == IndexType.BITMAP

    def test_create_composite_index(self, tmp_path: Path):
        """I can create a composite index on multiple fields."""
        manager = IndexManager(store_dir=tmp_path)

        index = manager.create_index(
            name="status_priority_idx",
            fields=["status", "priority"],
        )

        assert index.fields == ["status", "priority"]

    def test_list_indexes(self, tmp_path: Path):
        """I can list all indexes."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="idx1", fields=["field1"])
        manager.create_index(name="idx2", fields=["field2"])

        indexes = manager.list_indexes()

        assert len(indexes) == 2
        assert {idx.name for idx in indexes} == {"idx1", "idx2"}

    def test_drop_index(self, tmp_path: Path):
        """I can drop an index."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="temp_idx", fields=["temp"])

        manager.drop_index("temp_idx")

        assert not manager.has_index("temp_idx")


class TestIndexOperations:
    """Story: Adding, updating, and removing entities from indexes."""

    @pytest.fixture
    def indexed_manager(self, tmp_path: Path) -> IndexManager:
        """Create a manager with status and priority indexes."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])
        manager.create_index(name="priority_idx", fields=["priority"])
        return manager

    def test_add_entity_to_index(self, indexed_manager: IndexManager):
        """Adding an entity indexes it by its field values."""
        indexed_manager.index_entity(
            entity_id="T-001",
            fields={"status": "pending", "priority": "high"}
        )

        # Can look up by status
        result = indexed_manager.lookup("status_idx", "pending")
        assert "T-001" in result

        # Can look up by priority
        result = indexed_manager.lookup("priority_idx", "high")
        assert "T-001" in result

    def test_lookup_returns_empty_for_no_match(self, indexed_manager: IndexManager):
        """Lookup returns empty set when no entities match."""
        result = indexed_manager.lookup("status_idx", "nonexistent")
        assert result == set()

    def test_lookup_multiple_entities(self, indexed_manager: IndexManager):
        """Multiple entities with same field value are returned together."""
        indexed_manager.index_entity("T-001", {"status": "pending"})
        indexed_manager.index_entity("T-002", {"status": "pending"})
        indexed_manager.index_entity("T-003", {"status": "completed"})

        pending = indexed_manager.lookup("status_idx", "pending")

        assert pending == {"T-001", "T-002"}

    def test_update_entity_field(self, indexed_manager: IndexManager):
        """Updating an entity moves it between index buckets."""
        indexed_manager.index_entity("T-001", {"status": "pending"})

        # Update status from pending to completed
        indexed_manager.update_entity(
            entity_id="T-001",
            old_fields={"status": "pending"},
            new_fields={"status": "completed"}
        )

        # No longer in pending
        assert "T-001" not in indexed_manager.lookup("status_idx", "pending")
        # Now in completed
        assert "T-001" in indexed_manager.lookup("status_idx", "completed")

    def test_remove_entity(self, indexed_manager: IndexManager):
        """Removing an entity removes it from all indexes."""
        indexed_manager.index_entity(
            "T-001",
            {"status": "pending", "priority": "high"}
        )

        indexed_manager.remove_entity("T-001")

        assert "T-001" not in indexed_manager.lookup("status_idx", "pending")
        assert "T-001" not in indexed_manager.lookup("priority_idx", "high")

    def test_lookup_multi_values(self, indexed_manager: IndexManager):
        """I can look up entities matching any of multiple values."""
        indexed_manager.index_entity("T-001", {"status": "pending"})
        indexed_manager.index_entity("T-002", {"status": "in_progress"})
        indexed_manager.index_entity("T-003", {"status": "completed"})

        active = indexed_manager.lookup_multi(
            "status_idx",
            ["pending", "in_progress"]
        )

        assert active == {"T-001", "T-002"}


class TestIndexPersistence:
    """Story: Indexes persist across restarts."""

    def test_indexes_persist_to_disk(self, tmp_path: Path):
        """Index data is saved to disk."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])
        manager.index_entity("T-001", {"status": "pending"})
        manager.save()

        # Create new manager instance (simulates restart)
        manager2 = IndexManager(store_dir=tmp_path)

        # Index definition persists
        assert manager2.has_index("status_idx")
        # Index data persists
        assert "T-001" in manager2.lookup("status_idx", "pending")

    def test_dirty_tracking(self, tmp_path: Path):
        """Manager tracks when indexes need saving."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])

        assert not manager.is_dirty

        manager.index_entity("T-001", {"status": "pending"})

        assert manager.is_dirty

        manager.save()

        assert not manager.is_dirty

    def test_atomic_save(self, tmp_path: Path):
        """Save is atomic - partial failures don't corrupt index."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])
        manager.index_entity("T-001", {"status": "pending"})
        manager.save()

        # Verify file exists and is valid JSON
        index_file = tmp_path / "indexes" / "status_idx.json"
        assert index_file.exists()

        import json
        data = json.loads(index_file.read_text())
        assert "pending" in str(data)


class TestIndexRebuild:
    """Story: Rebuilding indexes from entity data."""

    def test_rebuild_index_from_entities(self, tmp_path: Path):
        """I can rebuild an index from a list of entities."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])

        entities = [
            {"id": "T-001", "status": "pending"},
            {"id": "T-002", "status": "pending"},
            {"id": "T-003", "status": "completed"},
        ]

        manager.rebuild_index("status_idx", entities)

        assert manager.lookup("status_idx", "pending") == {"T-001", "T-002"}
        assert manager.lookup("status_idx", "completed") == {"T-003"}

    def test_rebuild_all_indexes(self, tmp_path: Path):
        """I can rebuild all indexes at once."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])
        manager.create_index(name="priority_idx", fields=["priority"])

        entities = [
            {"id": "T-001", "status": "pending", "priority": "high"},
            {"id": "T-002", "status": "completed", "priority": "low"},
        ]

        manager.rebuild_all(entities)

        assert manager.lookup("status_idx", "pending") == {"T-001"}
        assert manager.lookup("priority_idx", "high") == {"T-001"}


class TestRelationshipIndex:
    """Story: Indexing relationships (like sprint membership)."""

    def test_relationship_index(self, tmp_path: Path):
        """I can index relationships between entities."""
        manager = IndexManager(store_dir=tmp_path)

        # Create a relationship index (sprint contains tasks)
        manager.create_index(
            name="sprint_tasks_idx",
            fields=["sprint_id"],
            index_type=IndexType.HASH,
        )

        # Index task membership in sprints
        manager.index_entity("T-001", {"sprint_id": "S-001"})
        manager.index_entity("T-002", {"sprint_id": "S-001"})
        manager.index_entity("T-003", {"sprint_id": "S-002"})

        # Look up tasks in sprint
        sprint1_tasks = manager.lookup("sprint_tasks_idx", "S-001")
        assert sprint1_tasks == {"T-001", "T-002"}


class TestIndexStats:
    """Story: Monitoring index performance."""

    def test_index_stats(self, tmp_path: Path):
        """I can get statistics about index usage."""
        manager = IndexManager(store_dir=tmp_path)
        manager.create_index(name="status_idx", fields=["status"])
        manager.index_entity("T-001", {"status": "pending"})

        # Perform some lookups
        manager.lookup("status_idx", "pending")  # hit
        manager.lookup("status_idx", "nonexistent")  # miss

        stats = manager.get_stats("status_idx")

        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert "hit_rate" in stats


class TestContainerIntegration:
    """Story: IndexManager works with DI container."""

    def test_index_manager_injectable(self, tmp_path: Path):
        """IndexManager can be resolved from container."""
        from cortical.core.bootstrap import create_container

        container = create_container(got_dir=tmp_path)
        manager = container.resolve(IndexManager)

        assert manager is not None
        assert isinstance(manager, IndexManager)


class TestGoTCompatibility:
    """Story: CDG IndexManager can replace GoT QueryIndexManager."""

    def test_got_task_indexing_pattern(self, tmp_path: Path):
        """
        The pattern GoT uses for task indexing works with CDG IndexManager.

        GoT does:
            index_manager.index_task(task_id, status=status, priority=priority)
            index_manager.lookup("status", "pending")

        CDG equivalent:
            index_manager.index_entity(task_id, {"status": status, "priority": priority})
            index_manager.lookup("status_idx", "pending")
        """
        manager = IndexManager(store_dir=tmp_path, namespace="got")

        # Create GoT-style indexes
        manager.create_index(name="status_idx", fields=["status"])
        manager.create_index(name="priority_idx", fields=["priority"])
        manager.create_index(name="sprint_idx", fields=["sprint_id"])

        # Index a task (GoT pattern)
        manager.index_entity(
            entity_id="T-001",
            fields={
                "status": "pending",
                "priority": "high",
                "sprint_id": "S-001"
            }
        )

        # Query patterns GoT uses
        pending_tasks = manager.lookup("status_idx", "pending")
        high_priority = manager.lookup("priority_idx", "high")
        sprint_tasks = manager.lookup("sprint_idx", "S-001")

        assert "T-001" in pending_tasks
        assert "T-001" in high_priority
        assert "T-001" in sprint_tasks

    def test_got_update_pattern(self, tmp_path: Path):
        """
        GoT's update pattern works with CDG IndexManager.

        GoT does:
            index_manager.update_task(task_id, old_status="pending", new_status="completed")

        CDG equivalent:
            index_manager.update_entity(task_id, {"status": "pending"}, {"status": "completed"})
        """
        manager = IndexManager(store_dir=tmp_path, namespace="got")
        manager.create_index(name="status_idx", fields=["status"])

        # Add task
        manager.index_entity("T-001", {"status": "pending"})

        # Update task status (GoT pattern)
        manager.update_entity(
            entity_id="T-001",
            old_fields={"status": "pending"},
            new_fields={"status": "completed"}
        )

        # Verify update worked
        assert "T-001" not in manager.lookup("status_idx", "pending")
        assert "T-001" in manager.lookup("status_idx", "completed")
