"""
Behavioral tests for CDG IndexManager.

Story: As a developer using CDG, I want to create indexes on entity fields
       so that I can efficiently query entities by field values.

Design Reference: docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md (Section 8)

Testing Strategy:
    - Unit/behavioral tests use direct instantiation with InMemoryFileSystem
      for isolation and speed (acceptable for testing)
    - Production code MUST use container.resolve(IndexManager) via DI
    - See TestContainerIntegration for container integration verification
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List

from cortical.common.filesystem import InMemoryFileSystem, FileSystem

# These imports will fail until we implement the module
# That's intentional - RED phase of TDD
from cortical.cdg.index import (
    IndexManager,
    IndexEntry,
    IndexType,
)


@pytest.fixture
def fs() -> InMemoryFileSystem:
    """Provide in-memory filesystem for testing."""
    return InMemoryFileSystem()


@pytest.fixture
def store_dir(fs: InMemoryFileSystem) -> Path:
    """Provide a virtual store directory."""
    path = Path("/test/store")
    fs.mkdir(path, parents=True, exist_ok=True)
    return path


class TestIndexManagerCreation:
    """Story: Creating and configuring an IndexManager."""

    def test_create_index_manager_with_filesystem(self, fs: InMemoryFileSystem, store_dir: Path):
        """Given a filesystem and store directory, I can create an IndexManager."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        assert manager is not None
        assert manager.store_dir == store_dir

    def test_index_manager_creates_index_directory(self, fs: InMemoryFileSystem, store_dir: Path):
        """The IndexManager creates an indexes subdirectory."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        assert fs.is_dir(store_dir / "indexes")

    def test_index_manager_with_namespace(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can create a namespaced IndexManager for domain separation."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs, namespace="tasks")
        assert manager.namespace == "tasks"
        # Namespace creates its own subdirectory
        assert fs.is_dir(store_dir / "indexes" / "tasks")


class TestIndexCreation:
    """Story: Creating indexes on entity fields."""

    def test_create_simple_index(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can create an index on a single field."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)

        index = manager.create_index(
            name="status_idx",
            fields=["status"],
        )

        assert index is not None
        assert index.name == "status_idx"
        assert index.fields == ["status"]
        assert manager.has_index("status_idx")

    def test_create_index_with_type(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can specify the index type (default is HASH for equality lookups)."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)

        index = manager.create_index(
            name="priority_idx",
            fields=["priority"],
            index_type=IndexType.HASH,
        )

        assert index.index_type == IndexType.HASH

    def test_create_bitmap_index_for_low_cardinality(self, fs: InMemoryFileSystem, store_dir: Path):
        """For low-cardinality fields like status, BITMAP index is efficient."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)

        index = manager.create_index(
            name="status_idx",
            fields=["status"],
            index_type=IndexType.BITMAP,
        )

        assert index.index_type == IndexType.BITMAP

    def test_create_composite_index(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can create a composite index on multiple fields."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)

        index = manager.create_index(
            name="status_priority_idx",
            fields=["status", "priority"],
        )

        assert index.fields == ["status", "priority"]

    def test_list_indexes(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can list all indexes."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        manager.create_index(name="idx1", fields=["field1"])
        manager.create_index(name="idx2", fields=["field2"])

        indexes = manager.list_indexes()

        assert len(indexes) == 2
        assert {idx.name for idx in indexes} == {"idx1", "idx2"}

    def test_drop_index(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can drop an index."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        manager.create_index(name="temp_idx", fields=["temp"])

        manager.drop_index("temp_idx")

        assert not manager.has_index("temp_idx")


class TestIndexOperations:
    """Story: Adding, updating, and removing entities from indexes."""

    @pytest.fixture
    def indexed_manager(self, fs: InMemoryFileSystem, store_dir: Path) -> IndexManager:
        """Create a manager with status and priority indexes."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
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
    """Story: Indexes persist across manager instances (simulated restart)."""

    def test_indexes_persist_via_save_load(self, fs: InMemoryFileSystem, store_dir: Path):
        """Index data persists after save and reload."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        manager.create_index(name="status_idx", fields=["status"])
        manager.index_entity("T-001", {"status": "pending"})
        manager.save()

        # Create new manager instance (simulates restart)
        manager2 = IndexManager(store_dir=store_dir, filesystem=fs)

        # Index definition persists
        assert manager2.has_index("status_idx")
        # Index data persists
        assert "T-001" in manager2.lookup("status_idx", "pending")

    def test_dirty_tracking(self, fs: InMemoryFileSystem, store_dir: Path):
        """Manager tracks when indexes need saving."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        manager.create_index(name="status_idx", fields=["status"])

        assert not manager.is_dirty

        manager.index_entity("T-001", {"status": "pending"})

        assert manager.is_dirty

        manager.save()

        assert not manager.is_dirty

    def test_save_writes_to_filesystem(self, fs: InMemoryFileSystem, store_dir: Path):
        """Save writes index data to the filesystem."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        manager.create_index(name="status_idx", fields=["status"])
        manager.index_entity("T-001", {"status": "pending"})
        manager.save()

        # Verify data written to filesystem
        index_file = store_dir / "indexes" / "status_idx.json"
        assert fs.exists(index_file)

        import json
        data = json.loads(fs.read_text(index_file))
        assert "pending" in str(data)


class TestIndexRebuild:
    """Story: Rebuilding indexes from entity data."""

    def test_rebuild_index_from_entities(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can rebuild an index from a list of entities."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
        manager.create_index(name="status_idx", fields=["status"])

        entities = [
            {"id": "T-001", "status": "pending"},
            {"id": "T-002", "status": "pending"},
            {"id": "T-003", "status": "completed"},
        ]

        manager.rebuild_index("status_idx", entities)

        assert manager.lookup("status_idx", "pending") == {"T-001", "T-002"}
        assert manager.lookup("status_idx", "completed") == {"T-003"}

    def test_rebuild_all_indexes(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can rebuild all indexes at once."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
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
    """Story: Indexing relationships (like container membership)."""

    def test_relationship_index(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can index relationships between entities."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)

        # Create a relationship index (container holds items)
        manager.create_index(
            name="container_items_idx",
            fields=["container_id"],
            index_type=IndexType.HASH,
        )

        # Index item membership in containers
        manager.index_entity("ITEM-001", {"container_id": "CONT-001"})
        manager.index_entity("ITEM-002", {"container_id": "CONT-001"})
        manager.index_entity("ITEM-003", {"container_id": "CONT-002"})

        # Look up items in container
        container1_items = manager.lookup("container_items_idx", "CONT-001")
        assert container1_items == {"ITEM-001", "ITEM-002"}


class TestIndexStats:
    """Story: Monitoring index performance."""

    def test_index_stats(self, fs: InMemoryFileSystem, store_dir: Path):
        """I can get statistics about index usage."""
        manager = IndexManager(store_dir=store_dir, filesystem=fs)
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

    def test_index_manager_injectable(self, fs: InMemoryFileSystem, store_dir: Path):
        """IndexManager can be resolved from container with filesystem injection."""
        from cortical.core.bootstrap import create_container

        # Container should wire up IndexManager with provided filesystem
        container = create_container(got_dir=store_dir, filesystem=fs)
        manager = container.resolve(IndexManager)

        assert manager is not None
        assert isinstance(manager, IndexManager)


class TestDomainEntityIndexing:
    """Story: IndexManager supports common domain entity indexing patterns."""

    def test_multi_field_entity_indexing(self, fs: InMemoryFileSystem, store_dir: Path):
        """
        Entities with multiple indexed fields can be queried by any field.

        Common pattern for task-like entities:
            index_manager.index_entity(entity_id, {"status": status, "priority": priority})
            index_manager.lookup("status_idx", "pending")
        """
        manager = IndexManager(store_dir=store_dir, filesystem=fs, namespace="domain")

        # Create indexes for common entity fields
        manager.create_index(name="status_idx", fields=["status"])
        manager.create_index(name="priority_idx", fields=["priority"])
        manager.create_index(name="container_idx", fields=["container_id"])

        # Index an entity with multiple fields
        manager.index_entity(
            entity_id="T-001",
            fields={
                "status": "pending",
                "priority": "high",
                "container_id": "C-001"
            }
        )

        # Query by any indexed field
        pending_entities = manager.lookup("status_idx", "pending")
        high_priority = manager.lookup("priority_idx", "high")
        container_entities = manager.lookup("container_idx", "C-001")

        assert "T-001" in pending_entities
        assert "T-001" in high_priority
        assert "T-001" in container_entities

    def test_entity_field_update_pattern(self, fs: InMemoryFileSystem, store_dir: Path):
        """
        Updating an entity's indexed field moves it between index buckets.

        Common pattern:
            index_manager.update_entity(entity_id, {"status": "pending"}, {"status": "completed"})
        """
        manager = IndexManager(store_dir=store_dir, filesystem=fs, namespace="domain")
        manager.create_index(name="status_idx", fields=["status"])

        # Add entity
        manager.index_entity("T-001", {"status": "pending"})

        # Update entity status
        manager.update_entity(
            entity_id="T-001",
            old_fields={"status": "pending"},
            new_fields={"status": "completed"}
        )

        # Verify update moved entity between buckets
        assert "T-001" not in manager.lookup("status_idx", "pending")
        assert "T-001" in manager.lookup("status_idx", "completed")
