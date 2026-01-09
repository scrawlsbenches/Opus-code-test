"""
CDG Index Manager - Schema-based and runtime index maintenance.

This module provides automatic index management for CDG entities with two modes:

1. **Schema-driven indexes** (declarative):
   - Fields marked with `indexed=True` in schema get automatic indexes
   - Schema-level `indexes` list supports composite indexes

2. **Runtime indexes** (imperative):
   - Create indexes dynamically via `create_index()` API
   - Useful for workload-specific or environment-specific indexes

Example schema with indexes:
    class DocumentSchema(BaseSchema):
        fields = {
            "id": Field("id", FieldType.STRING, required=True),
            "category": Field("category", FieldType.STRING, indexed=True),  # Hash index
            "created_at": Field("created_at", FieldType.DATETIME, indexed=True, index_type="btree"),
        }
        indexes = [
            'category',                         # Single field (redundant with Field.indexed)
            ('author', 'created_at'),           # Composite btree index
        ]

Example runtime index creation:
    index_manager.create_index(
        name="document_author_idx",
        entity_type="document",
        fields=["author"],
        index_type="hash",
        options=IndexConfig(async_build=True)
    )

Index types:
    - "hash": O(1) exact match lookups (default)
    - "btree": Range queries, ordering
    - "fulltext": Text search (future - TODO)
    - "bitmap": Low-cardinality fields (future - TODO)

Storage layout:
    {store_dir}/
        _indexes/
            {entity_type}/
                {field_name}.json       # Hash index: {value: [entity_ids]}
                {field_name}.btree.json # BTree index: {keys: [...], entries: {...}}
                _metadata.json          # Index metadata, runtime indexes, rebuild status

See: docs/design/cdg-query-language.md
See: docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md (Section 7: Index Structures)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Dict, Set, Optional, Any, List, TYPE_CHECKING, Tuple

from cortical.common.filesystem import FileSystem, RealFileSystem
from .btree import BTreeIndex

if TYPE_CHECKING:
    from .schema import SchemaRegistry, EntitySchema
    from .config import CDGConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Index Configuration Types
# =============================================================================

class IndexType(Enum):
    """Supported index types."""
    HASH = "hash"      # O(1) equality lookups
    BTREE = "btree"    # Range queries, ordering
    # TODO(cdg-index): Add FULLTEXT index type for text search
    # FULLTEXT = "fulltext"
    # TODO(cdg-index): Add BITMAP index type for low-cardinality fields (status, type)
    # BITMAP = "bitmap"


@dataclass
class IndexConfig:
    """
    Configuration options for index creation.

    Used with create_index() for runtime index configuration.

    Example:
        config = IndexConfig(
            async_build=True,
            storage_budget_mb=100,
            description="Index for fast category lookups"
        )
        index_manager.create_index("category_idx", "document", ["category"], "hash", config)
    """
    # Build behavior
    async_build: bool = False
    """If True, build index in background without blocking writes."""
    # TODO(cdg-index): Implement async index building with thread pool
    # parallelism: int = 4  # Number of concurrent threads for building

    # Storage limits (future)
    # TODO(cdg-index): Implement storage budget enforcement
    # storage_budget_mb: Optional[int] = None
    # """Maximum storage size for this index. None = unlimited."""

    # Filtering (future)
    # TODO(cdg-index): Implement partition filtering for multi-tenant indexes
    # partitions: Optional[List[str]] = None
    # """Only index entities in these partitions/namespaces."""

    # TODO(cdg-index): Implement conditional indexing
    # filter_predicate: Optional[str] = None
    # """Only index entities matching this predicate (e.g., "status != 'archived'")."""

    # Metadata
    description: str = ""
    """Human-readable description of the index purpose."""

    # Internal tracking
    created_at: Optional[str] = None
    """ISO timestamp when index was created."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "async_build": self.async_build,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexConfig":
        """Deserialize from dictionary."""
        return cls(
            async_build=data.get("async_build", False),
            description=data.get("description", ""),
            created_at=data.get("created_at"),
        )


@dataclass
class IndexDefinition:
    """
    Complete definition of an index (schema-based or runtime).

    This is the internal representation used by CDGIndexManager.
    """
    name: str
    """Unique index name (e.g., 'category_idx', 'priority_created_at_idx')."""

    entity_type: str
    """Entity type this index applies to (e.g., 'document')."""

    fields: List[str]
    """List of field names (single for simple index, multiple for composite)."""

    index_type: str = "hash"
    """Index type: 'hash' or 'btree'."""

    source: str = "schema"
    """Where this index came from: 'schema' (Field.indexed), 'schema_list' (BaseSchema.indexes), or 'runtime'."""

    config: IndexConfig = dataclass_field(default_factory=IndexConfig)
    """Additional configuration options."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "fields": self.fields,
            "index_type": self.index_type,
            "source": self.source,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexDefinition":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            entity_type=data["entity_type"],
            fields=data["fields"],
            index_type=data.get("index_type", "hash"),
            source=data.get("source", "runtime"),
            config=IndexConfig.from_dict(data.get("config", {})),
        )


class CDGIndexManager:
    """
    Manages indexes for CDG entities based on schema and runtime configuration.

    Supports two modes of index definition:
    1. Schema-driven: Fields with indexed=True, or BaseSchema.indexes list
    2. Runtime: Via create_index() API for dynamic index creation

    Example:
        # Initialize with schema registry
        index_manager = CDGIndexManager(store_dir, schema_registry)

        # Called by CDGStore on write (automatic)
        index_manager.update_index(
            "document", "E-001",
            old_data={"category": "draft"},
            new_data={"category": "published"}
        )

        # Lookup by index
        entity_ids = index_manager.lookup("document", "category", "published")

        # Create runtime index
        index_manager.create_index(
            name="document_author_idx",
            entity_type="document",
            fields=["author"],
            index_type="hash"
        )

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

        # Thread safety lock for index modifications (RLock allows reentry for rebuild_all)
        self._lock = threading.RLock()

        # In-memory hash indexes: {entity_type: {field_key: {value: set(entity_ids)}}}
        # field_key is field name for single-field, or "field1__field2" for composite
        self._indexes: Dict[str, Dict[str, Dict[Any, Set[str]]]] = {}

        # In-memory btree indexes: {entity_type: {field_key: BTreeIndex}}
        self._btree_indexes: Dict[str, Dict[str, BTreeIndex]] = {}

        # Runtime-created index definitions: {index_name: IndexDefinition}
        self._runtime_indexes: Dict[str, IndexDefinition] = {}

        # Track index staleness
        self._dirty: bool = False
        self._last_rebuild_time: Optional[float] = None

        # Ensure index directory exists
        if not self._fs.exists(self.index_dir):
            self._fs.mkdir(self.index_dir, parents=True, exist_ok=True)

        # Load existing indexes and runtime definitions from disk
        self._load_indexes()
        self._load_runtime_indexes()

    @property
    def schema_registry(self) -> Optional["SchemaRegistry"]:
        """Get the schema registry."""
        return self._schema_registry

    @schema_registry.setter
    def schema_registry(self, registry: "SchemaRegistry") -> None:
        """Set the schema registry (allows late binding via DI container)."""
        self._schema_registry = registry

    def _get_indexed_fields(self, entity_type: str) -> List[Tuple[str, str]]:
        """
        Get list of indexed fields for an entity type (single-field indexes only).

        This method returns simple (field_name, index_type) tuples for backward
        compatibility. For composite indexes, use get_all_index_definitions().

        Returns:
            List of (field_name, index_type) tuples for single-field indexes
        """
        result = []
        for idx_def in self.get_all_index_definitions(entity_type):
            # Only include single-field indexes for backward compatibility
            if len(idx_def.fields) == 1:
                result.append((idx_def.fields[0], idx_def.index_type))
        return result

    def get_all_index_definitions(self, entity_type: str) -> List[IndexDefinition]:
        """
        Get all index definitions for an entity type.

        Collects indexes from three sources:
        1. Field-level: Fields with indexed=True in schema
        2. Schema-level: BaseSchema.indexes list (supports composite indexes)
        3. Runtime: Indexes created via create_index() API

        Args:
            entity_type: The entity type to get indexes for

        Returns:
            List of IndexDefinition objects

        Example:
            definitions = index_manager.get_all_index_definitions("document")
            for idx_def in definitions:
                print(f"{idx_def.name}: {idx_def.fields} ({idx_def.index_type})")
        """
        definitions: Dict[str, IndexDefinition] = {}  # Dedupe by name

        # 1. Field-level indexes from schema (Field.indexed=True)
        if self._schema_registry is not None:
            schema = self._schema_registry.get_schema(entity_type)
            if schema is not None:
                # Check each field for indexed=True
                for field_name, field_def in schema.fields.items():
                    if getattr(field_def, 'indexed', False):
                        idx_name = f"{field_name}_idx"
                        if idx_name not in definitions:
                            definitions[idx_name] = IndexDefinition(
                                name=idx_name,
                                entity_type=entity_type,
                                fields=[field_name],
                                index_type=getattr(field_def, 'index_type', 'hash'),
                                source="schema",
                            )

                # 2. Schema-level indexes list (BaseSchema.indexes)
                # This supports composite indexes like ('priority', 'created_at')
                for idx_name, fields in schema.get_indexes():
                    if idx_name not in definitions:
                        # Composite indexes default to btree (range queries on multiple fields)
                        # Single-field schema indexes default to hash
                        index_type = "btree" if len(fields) > 1 else "hash"
                        definitions[idx_name] = IndexDefinition(
                            name=idx_name,
                            entity_type=entity_type,
                            fields=fields,
                            index_type=index_type,
                            source="schema_list",
                        )

        # 3. Runtime-created indexes
        for idx_name, idx_def in self._runtime_indexes.items():
            if idx_def.entity_type == entity_type:
                definitions[idx_name] = idx_def

        return list(definitions.values())

    def _ensure_index_structure(self, entity_type: str) -> None:
        """Ensure index data structures exist for entity type."""
        if entity_type not in self._indexes:
            self._indexes[entity_type] = {}
        if entity_type not in self._btree_indexes:
            self._btree_indexes[entity_type] = {}

        for field_name, index_type in self._get_indexed_fields(entity_type):
            if index_type == "btree":
                if field_name not in self._btree_indexes[entity_type]:
                    self._btree_indexes[entity_type][field_name] = BTreeIndex()
            else:  # hash (default)
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
            entity_type: Type of entity (e.g., "document")
            entity_id: Entity ID (e.g., "E-001")
            old_data: Previous entity data (None if new entity)
            new_data: New entity data (None if deleted)
        """
        indexed_fields = self._get_indexed_fields(entity_type)
        if not indexed_fields:
            return

        with self._lock:
            self._ensure_index_structure(entity_type)

            for field_name, index_type in indexed_fields:
                # Get old and new values
                old_value = old_data.get(field_name) if old_data else None
                new_value = new_data.get(field_name) if new_data else None

                # Skip if value unchanged
                if old_value == new_value:
                    continue

                if index_type == "btree":
                    # Use btree index
                    btree = self._btree_indexes[entity_type][field_name]

                    # Remove from old value
                    if old_value is not None:
                        btree.remove(old_value, entity_id)

                    # Add to new value
                    if new_value is not None:
                        btree.insert(new_value, entity_id)
                else:
                    # Use hash index (default)
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
        with self._lock:
            # Remove from hash indexes
            if entity_type in self._indexes:
                for field_name, field_index in self._indexes[entity_type].items():
                    # Find and remove entity_id from all value sets
                    for value, entity_ids in list(field_index.items()):
                        entity_ids.discard(entity_id)
                        if not entity_ids:
                            del field_index[value]

            # Remove from btree indexes
            if entity_type in self._btree_indexes:
                for field_name, btree in self._btree_indexes[entity_type].items():
                    # Remove from all keys that contain this entity_id
                    for key in btree.get_distinct_keys():
                        btree.remove(key, entity_id)

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
            entity_type: Type of entity (e.g., "document")
            field_name: Indexed field name (e.g., "category")
            value: Value to search for (e.g., "published")

        Returns:
            Set of entity IDs matching the value

        Example:
            # Find all published documents
            entity_ids = index_manager.lookup("document", "category", "published")
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

    def get_index_type(
        self,
        entity_type: str,
        field_name: str,
    ) -> Optional[str]:
        """
        Get the index type for a field.

        Args:
            entity_type: Type of entity
            field_name: Field name

        Returns:
            Index type ("hash" or "btree") or None if not indexed
        """
        for fname, itype in self._get_indexed_fields(entity_type):
            if fname == field_name:
                return itype
        return None

    def is_btree_indexed(
        self,
        entity_type: str,
        field_name: str,
    ) -> bool:
        """
        Check if a field has a btree index.

        Args:
            entity_type: Type of entity
            field_name: Field name

        Returns:
            True if field has btree index, False otherwise
        """
        return self.get_index_type(entity_type, field_name) == "btree"

    def lookup_range(
        self,
        entity_type: str,
        field_name: str,
        start_value: Optional[Any] = None,
        end_value: Optional[Any] = None,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
    ) -> Set[str]:
        """
        Look up entity IDs within a value range (btree indexes only).

        For hash indexes, raises ValueError. Use lookup() for equality.

        Args:
            entity_type: Type of entity (e.g., "document")
            field_name: Indexed field name (must have btree index)
            start_value: Lower bound (None = no lower bound)
            end_value: Upper bound (None = no upper bound)
            start_inclusive: Include start_value in results
            end_inclusive: Include end_value in results

        Returns:
            Set of entity IDs within the range

        Raises:
            ValueError: If field does not have a btree index

        Example:
            # Find entities created after 2026-01-01
            ids = index_manager.lookup_range(
                "document", "created_at",
                start_value="2026-01-01",
                start_inclusive=False
            )
        """
        if entity_type not in self._btree_indexes:
            raise ValueError(
                f"No btree index for entity type '{entity_type}'. "
                f"Field '{field_name}' may use hash index instead."
            )

        if field_name not in self._btree_indexes[entity_type]:
            raise ValueError(
                f"Field '{field_name}' does not have a btree index. "
                f"Check schema definition or use lookup() for hash indexes."
            )

        btree = self._btree_indexes[entity_type][field_name]
        return btree.lookup_range(
            start_key=start_value,
            end_key=end_value,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive
        )

    def lookup_gt(
        self,
        entity_type: str,
        field_name: str,
        value: Any,
    ) -> Set[str]:
        """
        Look up entity IDs with field value > given value (btree only).

        Args:
            entity_type: Type of entity
            field_name: Indexed field name (must have btree index)
            value: The lower bound (exclusive)

        Returns:
            Set of entity IDs

        Raises:
            ValueError: If field does not have a btree index
        """
        return self.lookup_range(
            entity_type, field_name,
            start_value=value,
            start_inclusive=False
        )

    def lookup_gte(
        self,
        entity_type: str,
        field_name: str,
        value: Any,
    ) -> Set[str]:
        """
        Look up entity IDs with field value >= given value (btree only).

        Args:
            entity_type: Type of entity
            field_name: Indexed field name (must have btree index)
            value: The lower bound (inclusive)

        Returns:
            Set of entity IDs

        Raises:
            ValueError: If field does not have a btree index
        """
        return self.lookup_range(
            entity_type, field_name,
            start_value=value,
            start_inclusive=True
        )

    def lookup_lt(
        self,
        entity_type: str,
        field_name: str,
        value: Any,
    ) -> Set[str]:
        """
        Look up entity IDs with field value < given value (btree only).

        Args:
            entity_type: Type of entity
            field_name: Indexed field name (must have btree index)
            value: The upper bound (exclusive)

        Returns:
            Set of entity IDs

        Raises:
            ValueError: If field does not have a btree index
        """
        return self.lookup_range(
            entity_type, field_name,
            end_value=value,
            end_inclusive=False
        )

    def lookup_lte(
        self,
        entity_type: str,
        field_name: str,
        value: Any,
    ) -> Set[str]:
        """
        Look up entity IDs with field value <= given value (btree only).

        Args:
            entity_type: Type of entity
            field_name: Indexed field name (must have btree index)
            value: The upper bound (inclusive)

        Returns:
            Set of entity IDs

        Raises:
            ValueError: If field does not have a btree index
        """
        return self.lookup_range(
            entity_type, field_name,
            end_value=value,
            end_inclusive=True
        )

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

    # =========================================================================
    # Runtime Index API
    # =========================================================================

    def create_index(
        self,
        name: str,
        entity_type: str,
        fields: List[str],
        index_type: str = "hash",
        options: Optional[IndexConfig] = None,
    ) -> IndexDefinition:
        """
        Create a new index at runtime.

        This allows dynamic index creation without modifying schemas.
        Useful for workload-specific or environment-specific indexes.

        Args:
            name: Unique index name (e.g., "document_author_idx")
            entity_type: Entity type to index (e.g., "document")
            fields: List of field names to index (single or composite)
            index_type: "hash" or "btree" (default: "hash")
            options: Additional configuration options

        Returns:
            IndexDefinition for the created index

        Raises:
            ValueError: If index name already exists or invalid parameters

        Example:
            # Create single-field hash index
            index_manager.create_index(
                name="document_author_idx",
                entity_type="document",
                fields=["author"]
            )

            # Create composite btree index
            index_manager.create_index(
                name="document_priority_date_idx",
                entity_type="document",
                fields=["priority", "created_at"],
                index_type="btree"
            )
        """
        with self._lock:
            # Validate index name is unique
            if name in self._runtime_indexes:
                raise ValueError(f"Index '{name}' already exists")

            # Check for existing schema-defined index with same name
            existing_defs = self.get_all_index_definitions(entity_type)
            for idx_def in existing_defs:
                if idx_def.name == name:
                    raise ValueError(
                        f"Index '{name}' already exists (source: {idx_def.source})"
                    )

            # Validate index type
            if index_type not in ("hash", "btree"):
                raise ValueError(
                    f"Invalid index_type '{index_type}'. Must be 'hash' or 'btree'."
                )

            # Validate fields
            if not fields:
                raise ValueError("At least one field is required")

            # TODO(cdg-index): Validate fields exist in schema
            # TODO(cdg-index): Support nested paths like "properties.status"

            # Create index definition
            config = options or IndexConfig()
            config.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            idx_def = IndexDefinition(
                name=name,
                entity_type=entity_type,
                fields=fields,
                index_type=index_type,
                source="runtime",
                config=config,
            )

            # Store definition
            self._runtime_indexes[name] = idx_def

            # Create index structure
            self._ensure_index_structure(entity_type)

            # TODO(cdg-index): Implement async index building
            # For now, indexes start empty and are populated on subsequent writes
            # To backfill, use rebuild_all() with an entity iterator

            self._dirty = True
            logger.info(f"Created runtime index '{name}' on {entity_type}.{fields}")

            return idx_def

    def drop_index(self, name: str, force: bool = False) -> bool:
        """
        Drop a runtime-created index.

        Schema-defined indexes cannot be dropped (modify the schema instead).

        Args:
            name: Index name to drop
            force: If True, drop even if index may be in use (future)

        Returns:
            True if index was dropped, False if not found

        Raises:
            ValueError: If attempting to drop a schema-defined index

        Example:
            index_manager.drop_index("document_author_idx")
        """
        with self._lock:
            # Check if it's a runtime index
            if name not in self._runtime_indexes:
                # Check if it's a schema-defined index
                for entity_type in (self._schema_registry.list_schemas()
                                    if self._schema_registry else []):
                    for idx_def in self.get_all_index_definitions(entity_type):
                        if idx_def.name == name and idx_def.source != "runtime":
                            raise ValueError(
                                f"Cannot drop schema-defined index '{name}'. "
                                f"Modify the schema instead."
                            )
                return False

            idx_def = self._runtime_indexes[name]
            entity_type = idx_def.entity_type
            field_key = self._get_field_key(idx_def.fields)

            # Remove from index structures
            if idx_def.index_type == "btree":
                if entity_type in self._btree_indexes:
                    self._btree_indexes[entity_type].pop(field_key, None)
            else:
                if entity_type in self._indexes:
                    self._indexes[entity_type].pop(field_key, None)

            # Remove definition
            del self._runtime_indexes[name]

            self._dirty = True
            logger.info(f"Dropped runtime index '{name}'")

            return True

    def list_indexes(
        self,
        entity_type: Optional[str] = None
    ) -> List[IndexDefinition]:
        """
        List all indexes (schema-defined and runtime).

        Args:
            entity_type: Filter by entity type (None = all types)

        Returns:
            List of IndexDefinition objects

        Example:
            # List all indexes
            for idx in index_manager.list_indexes():
                print(f"{idx.name}: {idx.entity_type}.{idx.fields}")

            # List indexes for a specific entity type
            doc_indexes = index_manager.list_indexes("document")
        """
        if entity_type is not None:
            return self.get_all_index_definitions(entity_type)

        # Collect from all known entity types
        result = []
        entity_types: Set[str] = set()

        # From schema registry
        if self._schema_registry is not None:
            entity_types.update(self._schema_registry.list_schemas().keys())

        # From runtime indexes
        for idx_def in self._runtime_indexes.values():
            entity_types.add(idx_def.entity_type)

        # Get definitions for each type
        for etype in entity_types:
            result.extend(self.get_all_index_definitions(etype))

        return result

    def _get_field_key(self, fields: List[str]) -> str:
        """
        Get the key used to store index data for a field or field combination.

        Args:
            fields: List of field names

        Returns:
            String key for index storage (e.g., "category" or "priority__created_at")
        """
        return "__".join(fields)

    def _load_runtime_indexes(self) -> None:
        """Load runtime index definitions from metadata file."""
        metadata_path = self.index_dir / "_metadata.json"
        if not self._fs.exists(metadata_path):
            return

        try:
            content = self._fs.read_text(metadata_path)
            metadata = json.loads(content)

            runtime_indexes = metadata.get("runtime_indexes", {})
            for name, idx_data in runtime_indexes.items():
                self._runtime_indexes[name] = IndexDefinition.from_dict(idx_data)

            if runtime_indexes:
                logger.debug(f"Loaded {len(runtime_indexes)} runtime index definitions")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load runtime indexes: {e}")

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

        with self._lock:
            # Clear existing indexes (both hash and btree)
            self._indexes.clear()
            self._btree_indexes.clear()

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
        with self._lock:
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
            self._btree_indexes[entity_type] = {}

            for index_file in self._fs.glob(type_dir, "*.json"):
                if index_file.name.startswith("_"):
                    continue

                # Check if this is a btree index file
                if index_file.name.endswith(".btree.json"):
                    # BTree index: field_name.btree.json
                    field_name = index_file.stem.replace(".btree", "")
                    try:
                        content = self._fs.read_text(index_file)
                        data = json.loads(content)
                        self._btree_indexes[entity_type][field_name] = BTreeIndex.from_dict(data)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Failed to load btree index {index_file}: {e}")
                else:
                    # Hash index: field_name.json
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
        """Save indexes to disk (both hash and btree)."""
        # Save hash indexes
        for entity_type, field_indexes in self._indexes.items():
            type_dir = self.index_dir / entity_type
            self._fs.mkdir(type_dir, parents=True, exist_ok=True)

            for field_name, value_index in field_indexes.items():
                index_file = type_dir / f"{field_name}.json"

                # Convert sets to lists for JSON serialization
                data = {k: sorted(v) for k, v in value_index.items()}

                content = json.dumps(data, indent=2, sort_keys=True)
                self._fs.write_text(index_file, content)

        # Save btree indexes
        for entity_type, btree_field_indexes in self._btree_indexes.items():
            type_dir = self.index_dir / entity_type
            self._fs.mkdir(type_dir, parents=True, exist_ok=True)

            for field_name, btree in btree_field_indexes.items():
                index_file = type_dir / f"{field_name}.btree.json"

                data = btree.to_dict()
                content = json.dumps(data, indent=2, sort_keys=True)
                self._fs.write_text(index_file, content)

    def _save_metadata(self) -> None:
        """Save index metadata including runtime index definitions."""
        indexed_schemas = {}

        if self._schema_registry is not None:
            for entity_type in self._schema_registry.list_schemas():
                fields = [f[0] for f in self._get_indexed_fields(entity_type)]
                if fields:
                    indexed_schemas[entity_type] = sorted(fields)

        # Serialize runtime indexes
        runtime_indexes = {
            name: idx_def.to_dict()
            for name, idx_def in self._runtime_indexes.items()
        }

        metadata = {
            "last_rebuild": self._last_rebuild_time,
            "indexed_schemas": indexed_schemas,
            "runtime_indexes": runtime_indexes,
            "version": 2,  # Bumped version for runtime index support
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

        # Hash index stats
        for entity_type, field_indexes in self._indexes.items():
            if entity_type not in type_stats:
                type_stats[entity_type] = {}
            for field_name, value_index in field_indexes.items():
                entry_count = sum(len(ids) for ids in value_index.values())
                type_stats[entity_type][field_name] = {
                    "index_type": "hash",
                    "distinct_values": len(value_index),
                    "total_entries": entry_count,
                }
                total_entries += entry_count

        # BTree index stats
        for entity_type, btree_field_indexes in self._btree_indexes.items():
            if entity_type not in type_stats:
                type_stats[entity_type] = {}
            for field_name, btree in btree_field_indexes.items():
                btree_stats = btree.stats()
                type_stats[entity_type][field_name] = {
                    "index_type": "btree",
                    "distinct_keys": btree_stats["distinct_keys"],
                    "total_entries": btree_stats["total_entries"],
                    "min_key": btree_stats["min_key"],
                    "max_key": btree_stats["max_key"],
                }
                total_entries += btree_stats["total_entries"]

        # Count all entity types (union of hash and btree)
        all_entity_types = set(self._indexes.keys()) | set(self._btree_indexes.keys())

        return {
            "total_entries": total_entries,
            "entity_types": len(all_entity_types),
            "dirty": self._dirty,
            "last_rebuild": self._last_rebuild_time,
            "by_type": type_stats,
        }
