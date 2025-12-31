"""
File-based storage with versioning and checksums for GoT transactional system.

Provides ACID-compliant storage using atomic file operations, checksums for
integrity verification, and append-only history for snapshot isolation.

This module now delegates to CDGStore (Cortical Distributed Graph) for
core storage operations, providing GoT-specific entity type handling
on top of the generic CDG infrastructure.

Migration Note (2025-12-31):
    VersionedStore now wraps CDGStore. All storage operations are delegated
    to CDG, with GoT providing:
    - Entity type factory (Task, Decision, Sprint, etc.)
    - Schema validation via GoT's registry

    This enables GoT to benefit from CDG improvements while maintaining
    backward compatibility with existing code and data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

from .types import (
    Entity, Task, Decision, Edge, Sprint, Epic, Handoff,
    ClaudeMdLayer, ClaudeMdVersion, PersonaProfile, Team, Document,
    VALID_ENTITY_TYPES,
)
from .errors import ValidationError
from .config import DurabilityMode
from .schema import get_registry

# Import CDGStore for core storage operations
from cortical.cdg.storage import CDGStore
from cortical.cdg.config import DurabilityMode as CDGDurabilityMode


def _got_entity_factory(data: Dict[str, Any]) -> Entity:
    """
    GoT entity factory - creates correct entity subclass based on entity_type.

    This is the GoT-specific dispatch logic that maps entity_type strings
    to Task, Decision, Sprint, etc. classes.

    Args:
        data: Entity data dictionary

    Returns:
        Appropriate Entity subclass instance

    Raises:
        ValueError: If entity_type is missing or invalid
    """
    entity_type = data.get("entity_type")

    # Validate entity_type is present
    if not entity_type:
        entity_id = data.get("id", "<unknown>")
        raise ValueError(
            f"Missing entity_type in entity data for {entity_id}. "
            f"Valid types: {sorted(VALID_ENTITY_TYPES)}"
        )

    # Validate entity_type is known
    if entity_type not in VALID_ENTITY_TYPES:
        entity_id = data.get("id", "<unknown>")
        raise ValueError(
            f"Unknown entity_type '{entity_type}' for entity {entity_id}. "
            f"Valid types: {sorted(VALID_ENTITY_TYPES)}"
        )

    # Dispatch to appropriate factory
    if entity_type == "task":
        return Task.from_dict(data)
    elif entity_type == "decision":
        return Decision.from_dict(data)
    elif entity_type == "edge":
        return Edge.from_dict(data)
    elif entity_type == "sprint":
        return Sprint.from_dict(data)
    elif entity_type == "epic":
        return Epic.from_dict(data)
    elif entity_type == "handoff":
        return Handoff.from_dict(data)
    elif entity_type == "claudemd_layer":
        return ClaudeMdLayer.from_dict(data)
    elif entity_type == "claudemd_version":
        return ClaudeMdVersion.from_dict(data)
    elif entity_type == "persona_profile":
        return PersonaProfile.from_dict(data)
    elif entity_type == "team":
        return Team.from_dict(data)
    elif entity_type == "document":
        return Document.from_dict(data)
    else:
        # Fallback to base Entity
        return Entity.from_dict(data)


def _convert_durability(durability: DurabilityMode) -> CDGDurabilityMode:
    """Convert GoT DurabilityMode to CDG DurabilityMode."""
    mapping = {
        DurabilityMode.RELAXED: CDGDurabilityMode.FAST,
        DurabilityMode.BALANCED: CDGDurabilityMode.BALANCED,
        DurabilityMode.PARANOID: CDGDurabilityMode.PARANOID,
    }
    return mapping.get(durability, CDGDurabilityMode.BALANCED)


class VersionedStore:
    """
    File-based storage with versioning and checksums.

    Each entity is stored as a JSON file with:
    - Version number (monotonic)
    - Checksum (SHA256)
    - Timestamp

    The store maintains a global version counter that increments
    on every successful commit. History is maintained in append-only
    JSONL files for snapshot isolation support.

    Implementation Note:
        This class now wraps CDGStore, delegating storage operations
        while providing GoT-specific entity type handling and schema
        validation.

    Storage layout:
        {store_dir}/
            {entity_id}.json          # Current entity state
            _version.json             # Global version counter
            _history/
                {entity_id}.jsonl     # Historical snapshots
    """

    def __init__(
        self,
        store_dir: Path,
        durability: DurabilityMode = DurabilityMode.BALANCED,
        validate_on_save: bool = True
    ):
        """
        Initialize store, creating directory structure if needed.

        Args:
            store_dir: Directory path for storing entities
            durability: Durability mode controlling fsync behavior
            validate_on_save: If True, validate entities against schemas before saving
                             (default: True for data integrity)
        """
        self.store_dir = Path(store_dir)
        self.durability = durability
        self.validate_on_save = validate_on_save

        # Create CDGStore with GoT entity factory
        self._cdg_store = CDGStore(
            store_dir=self.store_dir,
            durability=_convert_durability(durability),
            validate_on_save=validate_on_save,
            entity_factory=_got_entity_factory,
        )

        # Expose history_dir for backward compatibility
        self.history_dir = self._cdg_store.history_dir

    def current_version(self) -> int:
        """
        Get current global version.

        Returns:
            Current global version number
        """
        return self._cdg_store.current_version()

    def read(self, entity_id: str) -> Optional[Entity]:
        """
        Read current version of an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity instance or None if not found

        Raises:
            CorruptionError: If checksum verification fails
        """
        return self._cdg_store.read(entity_id)

    def read_at_version(self, entity_id: str, version: int) -> Optional[Entity]:
        """
        Read entity as it was at a specific global version (for snapshot isolation).

        Args:
            entity_id: Entity identifier
            version: Global version to read at

        Returns:
            Entity state at that version, or None if didn't exist

        Note:
            For entities that were never modified (no history file),
            we assume they existed since version 1.
        """
        return self._cdg_store.read_at_version(entity_id, version)

    def write(self, entity: Entity) -> None:
        """
        Write an entity (used for single writes, increments entity.version).

        Args:
            entity: Entity to write

        Raises:
            CorruptionError: If checksum operations fail
            ValidationError: If entity fails schema validation (when validate_on_save=True)
        """
        # GoT-specific schema validation
        self._validate_entity(entity)

        # Delegate to CDGStore
        self._cdg_store.write(entity)

    def apply_writes(self, write_set: Dict[str, Entity]) -> int:
        """
        Atomically apply a set of writes.

        Args:
            write_set: Dictionary mapping entity_id to Entity

        Returns:
            New global version after writes

        Raises:
            CorruptionError: If checksum operations fail
            ValidationError: If any entity fails schema validation
        """
        # GoT-specific schema validation for all entities
        for entity in write_set.values():
            self._validate_entity(entity)

        # Delegate to CDGStore
        return self._cdg_store.apply_writes(write_set)

    def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity file exists, False otherwise
        """
        return self._cdg_store.exists(entity_id)

    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            True if deleted, False if not found
        """
        return self._cdg_store.delete(entity_id)

    def fsync_all(self) -> None:
        """
        Force fsync of all entity files and version file.

        Used by BALANCED mode to sync on transaction commit.
        """
        self._cdg_store.fsync_all()

    def _validate_entity(self, entity: Entity) -> None:
        """
        Validate entity against its schema using GoT's schema registry.

        Args:
            entity: Entity to validate

        Raises:
            ValidationError: If entity fails schema validation
        """
        if not self.validate_on_save:
            return

        # Get entity type from the entity
        entity_type = getattr(entity, 'entity_type', '')
        if not entity_type:
            return  # Skip validation for unknown types

        # Validate against schema using GoT's registry
        registry = get_registry()
        if not registry.has_schema(entity_type):
            return  # Skip validation for types without schemas

        result = registry.validate(entity_type, entity.to_dict())
        if not result.valid:
            raise ValidationError(
                f"Entity {entity.id} failed schema validation",
                entity_type=entity_type,
                errors=result.errors
            )

    # Backward compatibility: expose internal version
    @property
    def _version(self) -> int:
        """Internal version for backward compatibility."""
        return self._cdg_store._version

    # Backward compatibility: expose _entity_path for tests that use it
    def _entity_path(self, entity_id: str):
        """Get path for entity JSON file (for backward compatibility)."""
        return self._cdg_store._entity_path(entity_id)

    # Backward compatibility: expose _history_path for tests that use it
    def _history_path(self, entity_id: str):
        """Get path for entity history file (for backward compatibility)."""
        return self._cdg_store._history_path(entity_id)

    # Backward compatibility: expose _read_and_verify for advanced usage
    def _read_and_verify(self, path):
        """Read JSON and verify checksum (for backward compatibility)."""
        return self._cdg_store._read_and_verify(path)
