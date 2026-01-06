"""
CDG Index Manager - Generic entity field indexing.

Provides efficient lookup of entities by field values through various
index types (HASH, BITMAP, BTREE). Designed for dependency injection
with FileSystem abstraction for testability.

Design Reference: docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md (Section 8)

Example:
    from cortical.cdg.index import IndexManager, IndexType
    from cortical.common.filesystem import InMemoryFileSystem

    fs = InMemoryFileSystem()
    manager = IndexManager(store_dir=Path("/data"), filesystem=fs)

    # Create an index on status field
    manager.create_index(name="status_idx", fields=["status"], index_type=IndexType.HASH)

    # Index an entity
    manager.index_entity("E-001", {"status": "pending"})

    # Look up by status
    pending = manager.lookup("status_idx", "pending")  # {"E-001"}
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Union

from cortical.common.filesystem import FileSystem, RealFileSystem


class IndexType(Enum):
    """Index types supported by IndexManager."""
    HASH = auto()      # Point lookups, equality checks
    BITMAP = auto()    # Low-cardinality fields (status, type)
    BTREE = auto()     # Range queries, sorting (future)


@dataclass
class IndexEntry:
    """
    A single index definition with its data.

    Attributes:
        name: Index name (e.g., "status_idx")
        fields: List of field names this index covers
        index_type: Type of index (HASH, BITMAP, etc.)
        values: Mapping of field value -> set of entity IDs
        version: Incremented on each modification
    """
    name: str
    fields: List[str]
    index_type: IndexType = IndexType.HASH
    values: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    version: int = 0

    def add(self, entity_id: str, value: Any) -> None:
        """Add an entity ID to a value group."""
        str_value = str(value) if value is not None else "__null__"
        self.values[str_value].add(entity_id)
        self.version += 1

    def remove(self, entity_id: str, value: Optional[Any] = None) -> None:
        """Remove an entity ID from index."""
        if value is not None:
            str_value = str(value) if value is not None else "__null__"
            if str_value in self.values:
                self.values[str_value].discard(entity_id)
                if not self.values[str_value]:
                    del self.values[str_value]
        else:
            # Remove from all values (entity deleted)
            for val_set in self.values.values():
                val_set.discard(entity_id)
        self.version += 1

    def get(self, value: Any) -> Set[str]:
        """Get entity IDs for a value."""
        str_value = str(value) if value is not None else "__null__"
        return self.values.get(str_value, set()).copy()

    def clear(self) -> None:
        """Clear all indexed values."""
        self.values.clear()
        self.version = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "fields": self.fields,
            "index_type": self.index_type.name,
            "values": {k: list(v) for k, v in self.values.items()},
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexEntry":
        """Deserialize from dictionary."""
        entry = cls(
            name=data["name"],
            fields=data["fields"],
            index_type=IndexType[data.get("index_type", "HASH")],
            version=data.get("version", 0)
        )
        for key, ids in data.get("values", {}).items():
            entry.values[key] = set(ids)
        return entry


@dataclass
class IndexStats:
    """Statistics for an index."""
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class IndexManager:
    """
    Manages entity field indexes for efficient lookups.

    Designed for dependency injection - accepts FileSystem abstraction
    for testability (use InMemoryFileSystem in tests).

    Args:
        store_dir: Base directory for index storage
        filesystem: FileSystem implementation (real or in-memory)
        namespace: Optional namespace for domain separation

    Example:
        # Production
        manager = IndexManager(Path("./data"), RealFileSystem())

        # Testing
        fs = InMemoryFileSystem()
        manager = IndexManager(Path("/test"), fs)
    """

    def __init__(
        self,
        store_dir: Path,
        filesystem: Optional[FileSystem] = None,
        namespace: Optional[str] = None
    ):
        self._store_dir = Path(store_dir)
        self._fs = filesystem or RealFileSystem()
        self._namespace = namespace
        self._indexes: Dict[str, IndexEntry] = {}
        self._stats: Dict[str, IndexStats] = defaultdict(IndexStats)
        self._dirty = False

        # Determine index directory path
        self._index_dir = self._store_dir / "indexes"
        if namespace:
            self._index_dir = self._index_dir / namespace

        # Create directory structure
        self._fs.mkdir(self._index_dir, parents=True, exist_ok=True)

        # Load existing indexes
        self._load_indexes()

    @property
    def store_dir(self) -> Path:
        """Base storage directory."""
        return self._store_dir

    @property
    def namespace(self) -> Optional[str]:
        """Index namespace for domain separation."""
        return self._namespace

    @property
    def is_dirty(self) -> bool:
        """True if indexes have unsaved changes."""
        return self._dirty

    def _load_indexes(self) -> None:
        """Load existing indexes from filesystem."""
        try:
            index_files = self._fs.glob(self._index_dir, "*.json")
            for index_file in index_files:
                try:
                    content = self._fs.read_text(index_file)
                    data = json.loads(content)
                    entry = IndexEntry.from_dict(data)
                    self._indexes[entry.name] = entry
                except (json.JSONDecodeError, KeyError) as e:
                    # Skip corrupted index files
                    pass
        except Exception:
            # Directory may not exist yet or be empty
            pass

    def create_index(
        self,
        name: str,
        fields: List[str],
        index_type: IndexType = IndexType.HASH
    ) -> IndexEntry:
        """
        Create a new index on specified fields.

        Args:
            name: Unique index name
            fields: List of field names to index
            index_type: Type of index (HASH, BITMAP, BTREE)

        Returns:
            The created IndexEntry

        Raises:
            ValueError: If index with this name already exists
        """
        if name in self._indexes:
            raise ValueError(f"Index '{name}' already exists")

        entry = IndexEntry(
            name=name,
            fields=fields,
            index_type=index_type
        )
        self._indexes[name] = entry
        self._stats[name] = IndexStats()

        # Save immediately so index definition persists
        self._save_index(name)

        return entry

    def has_index(self, name: str) -> bool:
        """Check if an index exists."""
        return name in self._indexes

    def list_indexes(self) -> List[IndexEntry]:
        """List all indexes."""
        return list(self._indexes.values())

    def drop_index(self, name: str) -> None:
        """
        Drop an index.

        Args:
            name: Index name to drop

        Raises:
            KeyError: If index doesn't exist
        """
        if name not in self._indexes:
            raise KeyError(f"Index '{name}' not found")

        del self._indexes[name]
        if name in self._stats:
            del self._stats[name]

        # Remove from filesystem
        index_file = self._index_dir / f"{name}.json"
        self._fs.unlink(index_file, missing_ok=True)

    def index_entity(self, entity_id: str, fields: Dict[str, Any]) -> None:
        """
        Add an entity to all relevant indexes.

        Args:
            entity_id: Entity identifier
            fields: Dict of field name -> field value
        """
        for index_name, index in self._indexes.items():
            for field_name in index.fields:
                if field_name in fields:
                    index.add(entity_id, fields[field_name])
        self._dirty = True

    def lookup(self, index_name: str, value: Any) -> Set[str]:
        """
        Look up entity IDs by index value.

        Args:
            index_name: Name of the index to query
            value: Value to match

        Returns:
            Set of entity IDs matching the value
        """
        if index_name not in self._indexes:
            self._stats[index_name].misses += 1
            return set()

        result = self._indexes[index_name].get(value)

        if result:
            self._stats[index_name].hits += 1
        else:
            self._stats[index_name].misses += 1

        return result

    def lookup_multi(self, index_name: str, values: List[Any]) -> Set[str]:
        """
        Look up entity IDs matching any of the given values.

        Args:
            index_name: Name of the index to query
            values: List of values to match

        Returns:
            Set of entity IDs matching any value
        """
        result: Set[str] = set()
        for value in values:
            result |= self.lookup(index_name, value)
        return result

    def update_entity(
        self,
        entity_id: str,
        old_fields: Dict[str, Any],
        new_fields: Dict[str, Any]
    ) -> None:
        """
        Update an entity's indexed fields.

        Removes entity from old value buckets and adds to new ones.

        Args:
            entity_id: Entity identifier
            old_fields: Previous field values
            new_fields: New field values
        """
        for index_name, index in self._indexes.items():
            for field_name in index.fields:
                old_value = old_fields.get(field_name)
                new_value = new_fields.get(field_name)

                if old_value != new_value:
                    if old_value is not None:
                        index.remove(entity_id, old_value)
                    if new_value is not None:
                        index.add(entity_id, new_value)

        self._dirty = True

    def remove_entity(self, entity_id: str) -> None:
        """
        Remove an entity from all indexes.

        Args:
            entity_id: Entity identifier to remove
        """
        for index in self._indexes.values():
            index.remove(entity_id)
        self._dirty = True

    def save(self) -> None:
        """Persist all dirty indexes to filesystem."""
        if not self._dirty:
            return

        for name in self._indexes:
            self._save_index(name)

        self._dirty = False

    def _save_index(self, name: str) -> None:
        """Save a single index to filesystem."""
        if name not in self._indexes:
            return

        index = self._indexes[name]
        index_file = self._index_dir / f"{name}.json"

        content = json.dumps(index.to_dict(), indent=2)
        self._fs.write_text(index_file, content)

    def rebuild_index(self, index_name: str, entities: List[Dict[str, Any]]) -> None:
        """
        Rebuild a single index from entity data.

        Args:
            index_name: Index to rebuild
            entities: List of entity dicts with 'id' field

        Raises:
            KeyError: If index doesn't exist
        """
        if index_name not in self._indexes:
            raise KeyError(f"Index '{index_name}' not found")

        index = self._indexes[index_name]
        index.clear()

        for entity in entities:
            entity_id = entity.get("id")
            if entity_id:
                for field_name in index.fields:
                    if field_name in entity:
                        index.add(entity_id, entity[field_name])

        self._dirty = True

    def rebuild_all(self, entities: List[Dict[str, Any]]) -> None:
        """
        Rebuild all indexes from entity data.

        Args:
            entities: List of entity dicts with 'id' field
        """
        # Clear all indexes
        for index in self._indexes.values():
            index.clear()

        # Rebuild from entities
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id:
                self.index_entity(entity_id, entity)

        self._dirty = True
        self.save()

    def get_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for an index.

        Args:
            index_name: Index to get stats for

        Returns:
            Dict with hits, misses, hit_rate
        """
        stats = self._stats.get(index_name, IndexStats())
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
        }
