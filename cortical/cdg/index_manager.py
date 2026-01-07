"""
CDG Index Manager - Schema-based index maintenance.

This module provides automatic index management for CDG entities based on
schema field annotations. Like SQL Server column indexes, fields marked
with `indexed=True` in the schema will have indexes maintained automatically.

Example schema:
    EntitySchema(
        name="task",
        fields=[
            Field("id", FieldType.STRING, required=True),
            Field("status", FieldType.STRING, indexed=True),  # Indexed!
            Field("priority", FieldType.STRING, indexed=True),  # Indexed!
            Field("title", FieldType.STRING),  # Not indexed
        ]
    )

Index types:
    - "hash": O(1) exact match lookups (default)
    - "btree": Range queries (future)
    - "fulltext": Text search (future)

Storage layout:
    {store_dir}/
        _indexes/
            {entity_type}/
                {field_name}.json  # Hash index: {value: [entity_ids]}
                _metadata.json     # Index metadata and rebuild status

Design:
    - Indexes are CDG's responsibility (not GoT)
    - Schema defines what's indexed (declarative)
    - CDGStore calls index_manager on write/delete
    - CDGRecoveryManager uses index_manager.needs_rebuild()/rebuild_all()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Set, Optional, Any, List, TYPE_CHECKING

from cortical.common.filesystem import FileSystem, RealFileSystem

if TYPE_CHECKING:
    from .schema import SchemaRegistry, EntitySchema
    from .config import CDGConfig

logger = logging.getLogger(__name__)


class CDGIndexManager:
    """
    Manages indexes for CDG entities based on schema configuration.

    Indexes are automatically maintained on entity writes and deletes.
    The schema defines which fields are indexed via Field(indexed=True).

    Example:
        # Initialize with schema registry
        index_manager = CDGIndexManager(store_dir, schema_registry)

        # Called by CDGStore on write
        index_manager.update_index(
            "task", "T-001",
            old_data={"status": "pending"},
            new_data={"status": "completed"}
        )

        # Called by CDGStore for queries
        task_ids = index_manager.lookup("task", "status", "completed")
        # Returns {"T-001", "T-002", ...}

    Attributes:
        store_dir: Base directory for entity storage
        index_dir: Directory for index files
        schema_registry: Registry containing entity schemas
    """

    def __init__(
        self,
        store_dir: Path,
        schema_registry: Optional["SchemaRegistry"] = None,
        config: Optional["CDGConfig"] = None,
        filesystem: Optional[FileSystem] = None,
    ):
        """
        Initialize the index manager.

        Args:
            store_dir: Base directory for entity storage
            schema_registry: Registry containing entity schemas
            config: CDG configuration (for durability settings)
            filesystem: FileSystem abstraction (defaults to RealFileSystem)
        """
        self._fs: FileSystem = filesystem or RealFileSystem()
        self.store_dir = Path(store_dir)
        self.index_dir = self.store_dir / "_indexes"

        self._schema_registry = schema_registry
        self._config = config

        # In-memory index cache: {entity_type: {field_name: {value: set(entity_ids)}}}
        self._indexes: Dict[str, Dict[str, Dict[Any, Set[str]]]] = {}

        # Track index staleness
        self._dirty: bool = False
        self._last_rebuild_time: Optional[float] = None

        # Ensure index directory exists
        if not self._fs.exists(self.index_dir):
            self._fs.mkdir(self.index_dir, parents=True, exist_ok=True)

        # Load existing indexes from disk
        self._load_indexes()

    @property
    def schema_registry(self) -> Optional["SchemaRegistry"]:
        """Get the schema registry."""
        return self._schema_registry

    @schema_registry.setter
    def schema_registry(self, registry: "SchemaRegistry") -> None:
        """Set the schema registry (allows late binding via DI container)."""
        self._schema_registry = registry

    def _get_indexed_fields(self, entity_type: str) -> List[tuple]:
        """
        Get list of indexed fields for an entity type.

        Returns:
            List of (field_name, index_type) tuples
        """
        if self._schema_registry is None:
            return []

        if not self._schema_registry.has_schema(entity_type):
            return []

        schema = self._schema_registry.get_schema(entity_type)
        indexed_fields = []

        for field in schema.fields:
            if getattr(field, 'indexed', False):
                index_type = getattr(field, 'index_type', 'hash')
                indexed_fields.append((field.name, index_type))

        return indexed_fields

    def _ensure_index_structure(self, entity_type: str) -> None:
        """Ensure index data structures exist for entity type."""
        if entity_type not in self._indexes:
            self._indexes[entity_type] = {}

        for field_name, _ in self._get_indexed_fields(entity_type):
            if field_name not in self._indexes[entity_type]:
                self._indexes[entity_type][field_name] = {}

    def update_index(
        self,
        entity_type: str,
        entity_id: str,
        old_data: Optional[Dict[str, Any]],
        new_data: Optional[Dict[str, Any]],
    ) -> None:
        """
        Update indexes when an entity changes.

        Called by CDGStore on write() and delete().

        Args:
            entity_type: Type of entity (e.g., "task")
            entity_id: Entity ID (e.g., "T-001")
            old_data: Previous entity data (None if new entity)
            new_data: New entity data (None if deleted)
        """
        indexed_fields = self._get_indexed_fields(entity_type)
        if not indexed_fields:
            return

        self._ensure_index_structure(entity_type)

        for field_name, index_type in indexed_fields:
            # Get old and new values
            old_value = old_data.get(field_name) if old_data else None
            new_value = new_data.get(field_name) if new_data else None

            # Skip if value unchanged
            if old_value == new_value:
                continue

            field_index = self._indexes[entity_type][field_name]

            # Remove from old value's index set
            if old_value is not None:
                old_value_key = self._normalize_value(old_value)
                if old_value_key in field_index:
                    field_index[old_value_key].discard(entity_id)
                    # Clean up empty sets
                    if not field_index[old_value_key]:
                        del field_index[old_value_key]

            # Add to new value's index set
            if new_value is not None:
                new_value_key = self._normalize_value(new_value)
                if new_value_key not in field_index:
                    field_index[new_value_key] = set()
                field_index[new_value_key].add(entity_id)

        self._dirty = True

    def remove_from_index(self, entity_type: str, entity_id: str) -> None:
        """
        Remove an entity from all indexes (for delete operations).

        This is a convenience method when you don't have the old data.
        It scans all indexes for the entity type and removes the ID.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID to remove
        """
        if entity_type not in self._indexes:
            return

        for field_name, field_index in self._indexes[entity_type].items():
            # Find and remove entity_id from all value sets
            for value, entity_ids in list(field_index.items()):
                entity_ids.discard(entity_id)
                if not entity_ids:
                    del field_index[value]

        self._dirty = True

    def lookup(
        self,
        entity_type: str,
        field_name: str,
        value: Any,
    ) -> Set[str]:
        """
        Look up entity IDs by indexed field value.

        Args:
            entity_type: Type of entity (e.g., "task")
            field_name: Indexed field name (e.g., "status")
            value: Value to search for (e.g., "completed")

        Returns:
            Set of entity IDs matching the value

        Example:
            # Find all completed tasks
            completed_ids = index_manager.lookup("task", "status", "completed")
        """
        if entity_type not in self._indexes:
            return set()

        if field_name not in self._indexes[entity_type]:
            return set()

        value_key = self._normalize_value(value)
        return self._indexes[entity_type][field_name].get(value_key, set()).copy()

    def lookup_multi(
        self,
        entity_type: str,
        field_name: str,
        values: List[Any],
    ) -> Set[str]:
        """
        Look up entity IDs matching any of the given values.

        Args:
            entity_type: Type of entity
            field_name: Indexed field name
            values: List of values to search for

        Returns:
            Set of entity IDs matching any value
        """
        result = set()
        for value in values:
            result.update(self.lookup(entity_type, field_name, value))
        return result

    def get_distinct_values(
        self,
        entity_type: str,
        field_name: str,
    ) -> Set[Any]:
        """
        Get all distinct values for an indexed field.

        Useful for building filter dropdowns or faceted search.

        Args:
            entity_type: Type of entity
            field_name: Indexed field name

        Returns:
            Set of distinct values
        """
        if entity_type not in self._indexes:
            return set()

        if field_name not in self._indexes[entity_type]:
            return set()

        return set(self._indexes[entity_type][field_name].keys())

    def _normalize_value(self, value: Any) -> str:
        """
        Normalize a value for index key storage.

        Values are stored as-is since we control the input through schema.
        Only type conversion to string is performed for consistent hashing.
        """
        if value is None:
            return "__NULL__"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value  # Store as-is, we control the input
        if isinstance(value, (list, dict)):
            return json.dumps(value, sort_keys=True)
        return str(value)

    def needs_rebuild(self) -> bool:
        """
        Check if indexes need rebuilding.

        Returns True if:
        - Indexes have never been built
        - Index metadata is missing or corrupted
        - Schema has changed (new indexed fields)

        Called by CDGRecoveryManager during startup.
        """
        metadata_path = self.index_dir / "_metadata.json"

        if not self._fs.exists(metadata_path):
            return True

        try:
            content = self._fs.read_text(metadata_path)
            metadata = json.loads(content)

            # Check for schema changes
            if self._schema_registry is not None:
                indexed_schemas = metadata.get("indexed_schemas", {})

                for entity_type in self._schema_registry.list_schemas():
                    current_fields = sorted(
                        [f[0] for f in self._get_indexed_fields(entity_type)]
                    )
                    stored_fields = sorted(indexed_schemas.get(entity_type, []))

                    if current_fields != stored_fields:
                        logger.info(
                            f"Schema change detected for {entity_type}: "
                            f"stored={stored_fields}, current={current_fields}"
                        )
                        return True

            return False

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read index metadata: {e}")
            return True

    def rebuild_all(self, entity_iterator: Optional[callable] = None) -> int:
        """
        Rebuild all indexes from scratch.

        Args:
            entity_iterator: Optional callable that yields (entity_type, entity_data) tuples.
                           If not provided, indexes will be cleared but not populated.

        Returns:
            Number of entities indexed
        """
        logger.info("Rebuilding all indexes...")
        start_time = time.time()

        # Clear existing indexes
        self._indexes.clear()

        # Clear index directory
        if self._fs.exists(self.index_dir):
            for index_file in self._fs.glob(self.index_dir, "**/*.json"):
                self._fs.unlink(index_file, missing_ok=True)

        # Ensure directory exists
        self._fs.mkdir(self.index_dir, parents=True, exist_ok=True)

        entity_count = 0

        if entity_iterator is not None:
            for entity_type, entity_data in entity_iterator():
                entity_id = entity_data.get("id")
                if entity_id:
                    self.update_index(entity_type, entity_id, None, entity_data)
                    entity_count += 1

        # Save indexes and metadata
        self._save_indexes()
        self._save_metadata()

        self._dirty = False
        self._last_rebuild_time = time.time()

        elapsed = time.time() - start_time
        logger.info(f"Index rebuild complete: {entity_count} entities in {elapsed:.2f}s")

        return entity_count

    def persist(self) -> None:
        """
        Persist indexes to disk if dirty.

        Called periodically or on shutdown to ensure indexes are saved.
        """
        if not self._dirty:
            return

        self._save_indexes()
        self._dirty = False

    def _load_indexes(self) -> None:
        """Load indexes from disk on startup."""
        if not self._fs.exists(self.index_dir):
            return

        for type_dir in self._fs.iterdir(self.index_dir):
            if not self._fs.is_dir(type_dir):
                continue

            entity_type = type_dir.name
            if entity_type.startswith("_"):
                continue

            self._indexes[entity_type] = {}

            for index_file in self._fs.glob(type_dir, "*.json"):
                if index_file.name.startswith("_"):
                    continue

                field_name = index_file.stem

                try:
                    content = self._fs.read_text(index_file)
                    data = json.loads(content)

                    # Convert lists back to sets
                    self._indexes[entity_type][field_name] = {
                        k: set(v) for k, v in data.items()
                    }
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load index {index_file}: {e}")

    def _save_indexes(self) -> None:
        """Save indexes to disk."""
        for entity_type, field_indexes in self._indexes.items():
            type_dir = self.index_dir / entity_type
            self._fs.mkdir(type_dir, parents=True, exist_ok=True)

            for field_name, value_index in field_indexes.items():
                index_file = type_dir / f"{field_name}.json"

                # Convert sets to lists for JSON serialization
                data = {k: sorted(v) for k, v in value_index.items()}

                content = json.dumps(data, indent=2, sort_keys=True)
                self._fs.write_text(index_file, content)

    def _save_metadata(self) -> None:
        """Save index metadata."""
        indexed_schemas = {}

        if self._schema_registry is not None:
            for entity_type in self._schema_registry.list_schemas():
                fields = [f[0] for f in self._get_indexed_fields(entity_type)]
                if fields:
                    indexed_schemas[entity_type] = sorted(fields)

        metadata = {
            "last_rebuild": self._last_rebuild_time,
            "indexed_schemas": indexed_schemas,
            "version": 1,
        }

        metadata_path = self.index_dir / "_metadata.json"
        content = json.dumps(metadata, indent=2, sort_keys=True)
        self._fs.write_text(metadata_path, content)

    def stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        total_entries = 0
        type_stats = {}

        for entity_type, field_indexes in self._indexes.items():
            field_stats = {}
            for field_name, value_index in field_indexes.items():
                entry_count = sum(len(ids) for ids in value_index.values())
                field_stats[field_name] = {
                    "distinct_values": len(value_index),
                    "total_entries": entry_count,
                }
                total_entries += entry_count
            type_stats[entity_type] = field_stats

        return {
            "total_entries": total_entries,
            "entity_types": len(self._indexes),
            "dirty": self._dirty,
            "last_rebuild": self._last_rebuild_time,
            "by_type": type_stats,
        }
