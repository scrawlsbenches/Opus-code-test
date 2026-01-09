"""
CDG Storage Layer - File-based graph storage with versioning and checksums.

This is the core storage engine for CDG, providing:
- ACID-compliant storage using atomic file operations
- Checksums for integrity verification
- Append-only history for snapshot isolation (MVCC)
- Pluggable entity factory for domain-specific types

Design: Lifted from GoT's VersionedStore with CDG extensions:
- Pluggable entity factory (no hardcoded entity types)
- Future partition support hooks (partition_count=1 for now)
- CDG configuration integration

Storage layout:
    {store_dir}/
        {entity_id}.json          # Current entity state
        _version.json             # Global version counter
        _history/
            {entity_id}.jsonl     # Historical snapshots
"""

from __future__ import annotations

import copy
import logging
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Callable, Any, Type, TYPE_CHECKING

from cortical.utils.checksums import compute_checksum
from cortical.utils.locking import ProcessLock
from cortical.common.filesystem import FileSystem, RealFileSystem, InMemoryFileSystem

from .types import Entity
from .errors import CorruptionError, ValidationError, StorageError
from .config import CDGConfig, DurabilityMode

if TYPE_CHECKING:
    from .schema import SchemaRegistry
    from .index_manager import CDGIndexManager


class NoOpLock:
    """
    No-operation lock for in-memory filesystems.

    Process locking is unnecessary when using InMemoryFileSystem since
    the data only exists within a single process. This avoids trying to
    create lock files on paths that don't exist on the real filesystem.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# Type alias for entity factory functions
EntityFactory = Callable[[Dict[str, Any]], Entity]


def default_entity_factory(data: Dict[str, Any]) -> Entity:
    """
    Default entity factory - creates base Entity instances.

    For domain-specific types (Task, Decision, etc.), provide
    a custom factory that dispatches based on entity_type.

    Args:
        data: Entity data dictionary

    Returns:
        Entity instance
    """
    return Entity.from_dict(data)


class CDGStore:
    """
    File-based storage with versioning and checksums.

    Each entity is stored as a JSON file with:
    - Version number (monotonic)
    - Checksum (SHA256)
    - Timestamp

    The store maintains a global version counter that increments
    on every successful commit. History is maintained in append-only
    JSONL files for snapshot isolation support.

    Example:
        # Basic usage with default entity factory
        fs = RealFileSystem(Path("./data"))
        store = CDGStore(filesystem=fs)

        entity = Entity(id="E-001", entity_type="document")
        store.write(entity)

        loaded = store.read("E-001")

    Example with custom entity factory (for GoT compatibility):
        def got_entity_factory(data: dict) -> Entity:
            entity_type = data.get("entity_type")
            if entity_type == "task":
                return Task.from_dict(data)
            elif entity_type == "decision":
                return Decision.from_dict(data)
            else:
                return Entity.from_dict(data)

        fs = RealFileSystem(Path("./data"))
        store = CDGStore(
            filesystem=fs,
            entity_factory=got_entity_factory
        )

    Attributes:
        store_dir: Directory path for storing entities
        config: CDG configuration
        entity_factory: Function to create entities from dicts
    """

    def __init__(
        self,
        filesystem: FileSystem,
        config: Optional[CDGConfig] = None,
        entity_factory: Optional[EntityFactory] = None,
        # Caching (enabled by default for performance)
        cache_enabled: bool = True,
        # Schema registry for validation (injected via Container)
        schema_registry: Optional["SchemaRegistry"] = None,
        # Index manager for schema-based indexes (injected via Container)
        index_manager: Optional["CDGIndexManager"] = None,
    ):
        """
        Initialize store, creating directory structure if needed.

        Args:
            filesystem: FileSystem implementation (required). The filesystem's
                       base_dir determines where entities are stored.
            config: CDG configuration (optional, creates default if not provided)
            entity_factory: Function to create Entity from dict (optional)
            cache_enabled: Enable entity caching for read performance
            schema_registry: SchemaRegistry for entity validation (optional,
                           injected via Container for schema-aware validation)
            index_manager: CDGIndexManager for schema-based indexes (optional,
                          injected via Container for indexed field maintenance)
        """
        # FileSystem abstraction - required
        self._fs: FileSystem = filesystem

        self.store_dir = self._fs.base_dir
        self._fs.mkdir(self.store_dir, parents=True, exist_ok=True)

        # Configuration - use provided config or create default
        self.config = config or CDGConfig()

        # Convenience aliases (used throughout storage code)
        self.durability = self.config.durability
        self.validate_on_save = self.config.validate_on_write

        # Entity factory for creating entities from dicts
        self.entity_factory = entity_factory or default_entity_factory

        # Schema registry for validation (injected via Container)
        # When set, enables schema-based validation on write
        self._schema_registry: Optional["SchemaRegistry"] = schema_registry

        # Index manager for schema-based indexes (injected via Container)
        # When set, maintains indexes on write/delete based on schema configuration
        self._index_manager: Optional["CDGIndexManager"] = index_manager

        # History directory for MVCC snapshots
        self.history_dir = self.store_dir / "_history"
        self._fs.mkdir(self.history_dir, exist_ok=True)

        # Pending history directory for crash-safe history writes
        self._pending_history_dir = self.history_dir / "_pending"
        self._fs.mkdir(self._pending_history_dir, exist_ok=True)

        # Determine if we need process locks (not needed for in-memory filesystems)
        use_process_locks = not isinstance(self._fs, InMemoryFileSystem)

        # Process lock for concurrent history file access protection
        self._history_lock = ProcessLock(self.history_dir / ".history.lock") if use_process_locks else NoOpLock()

        # Thread lock for concurrent version file access protection (within same process)
        self._version_thread_lock = threading.Lock()

        # Process lock for cross-process version file protection
        self._version_lock = ProcessLock(self.store_dir / ".version.lock", reentrant=False) if use_process_locks else NoOpLock()

        # Write lock for thread-safe write operations (covers entire write transaction)
        self._write_lock = threading.RLock()

        # Process lock for cross-process write protection (covers entire write transaction)
        self._write_process_lock = ProcessLock(self.store_dir / ".write.lock", reentrant=True) if use_process_locks else NoOpLock()

        # Load current version
        self._version = self._load_version()

        # Entity cache for read performance (must be initialized before recovery!)
        self._cache_enabled = cache_enabled
        self._cache: Dict[str, Entity] = {}
        self._cache_timestamps: Dict[str, float] = {}  # entity_id -> last_access_time
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_ttl: Optional[float] = None  # TTL in seconds (None = no expiration)
        self._cache_max_size: Optional[int] = None  # Max entries (None = unlimited)

        # Recover any pending history entries from interrupted writes
        # Note: This must be AFTER cache initialization since recovery reads entities
        self._recover_pending_history()

    def current_version(self) -> int:
        """
        Get current global version.

        Returns:
            Current global version number
        """
        return self._version

    # ==================== Cache Methods ====================

    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size, enabled status, ttl, and max_size
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'enabled': self._cache_enabled,
            'ttl': self._cache_ttl,
            'max_size': self._cache_max_size,
        }

    def cache_configure(self, ttl: Optional[float] = None, max_size: Optional[int] = None) -> None:
        """
        Configure cache behavior.

        Args:
            ttl: Time-to-live in seconds for cached entries. None disables TTL.
            max_size: Maximum number of entries. Oldest entries are evicted when exceeded.
                     None means unlimited.
        """
        self._cache_ttl = ttl
        self._cache_max_size = max_size
        # Immediately apply max_size if set and cache exceeds it
        if max_size is not None and len(self._cache) > max_size:
            self._evict_lru_entries(len(self._cache) - max_size)

    def cache_clear(self) -> None:
        """Clear all cached entities and reset statistics."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _evict_lru_entries(self, count: int) -> None:
        """Evict the oldest entries from cache based on access time."""
        if count <= 0 or not self._cache:
            return
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(self._cache_timestamps.items(), key=lambda x: x[1])
        # Evict the oldest entries
        for entity_id, _ in sorted_entries[:count]:
            self._cache.pop(entity_id, None)
            self._cache_timestamps.pop(entity_id, None)

    def _cache_get(self, entity_id: str) -> Optional[Entity]:
        """Get entity from cache, updating hit/miss stats."""
        if not self._cache_enabled:
            return None
        entity = self._cache.get(entity_id)
        if entity is not None:
            # Check TTL expiration
            if self._cache_ttl is not None:
                timestamp = self._cache_timestamps.get(entity_id, 0)
                if time.time() - timestamp > self._cache_ttl:
                    # Entry has expired, remove it
                    self._cache.pop(entity_id, None)
                    self._cache_timestamps.pop(entity_id, None)
                    return None
            # Update access time and record hit
            self._cache_timestamps[entity_id] = time.time()
            self._cache_hits += 1
        return entity

    def _cache_set(self, entity_id: str, entity: Entity) -> None:
        """Add entity to cache, updating miss stats."""
        if not self._cache_enabled:
            return
        # Enforce max_size before adding
        if self._cache_max_size is not None:
            # If already at max size and this is a new entry, evict one
            if entity_id not in self._cache and len(self._cache) >= self._cache_max_size:
                self._evict_lru_entries(1)
        self._cache[entity_id] = entity
        self._cache_timestamps[entity_id] = time.time()
        self._cache_misses += 1

    def _cache_invalidate(self, entity_id: str) -> None:
        """Remove entity from cache."""
        self._cache.pop(entity_id, None)
        self._cache_timestamps.pop(entity_id, None)

    # ==================== Read Methods ====================

    def read(self, entity_id: str) -> Optional[Entity]:
        """
        Read current version of an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity instance or None if not found

        Raises:
            CorruptionError: If checksum verification fails

        Note:
            Handles TOCTOU race gracefully: if file is deleted between
            exists() check and read, returns None instead of raising.
            This is expected during concurrent delete + read operations.

            FUTURE: When CDG index is implemented per the distributed graph
            specification (docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md),
            this race condition will be eliminated at the storage layer since
            index lookups won't return IDs for deleted entities.
        """
        # Check cache first - return copy to ensure isolation
        cached = self._cache_get(entity_id)
        if cached is not None:
            return copy.deepcopy(cached)

        path = self._entity_path(entity_id)
        if not self._fs.exists(path):
            return None

        try:
            wrapper = self._read_and_verify(path)
            entity = self.entity_factory(wrapper["data"])
            # Populate cache on miss, return copy to ensure isolation
            self._cache_set(entity_id, entity)
            return copy.deepcopy(entity)
        except FileNotFoundError:
            # File was deleted between exists() check and read - treat as not found.
            # This is expected during concurrent delete + read operations.
            return None

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
        # If reading at or after current version, return current entity
        if version >= self._version:
            return self.read(entity_id)

        # Check history for earlier versions
        history_path = self._history_path(entity_id)

        if self._fs.exists(history_path):
            # Find entry with highest global_version <= version
            matching_entry = None
            content = self._fs.read_text(history_path)
            for line in content.splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                gv = entry.get("global_version", 0)
                if gv <= version:
                    matching_entry = entry
                else:
                    break  # History is sorted, can stop early

            if matching_entry:
                return self.entity_factory(matching_entry["data"])

            # No matching entry - entity didn't exist at that version
            return None
        else:
            # No history file - entity never modified since creation
            # Assume entity existed since version 1
            if version >= 1:
                return self.read(entity_id)
            else:
                return None

    def write(self, entity: Entity) -> None:
        """
        Write an entity (used for single writes, increments entity.version).

        Thread-safe and process-safe via write locks.

        Args:
            entity: Entity to write

        Raises:
            CorruptionError: If checksum operations fail
            ValidationError: If entity fails validation (when validate_on_save=True)
        """
        # Acquire both thread and process locks for full safety
        with self._write_lock:
            with self._write_process_lock:
                # Validate entity before writing
                self._validate_entity(entity)

                # Calculate expected entity version after write
                current_entity = self.read(entity.id)
                expected_entity_version = (current_entity.version + 1) if current_entity else 1

                # Phase 1: Capture and write pending history (crash-safe)
                # This happens BEFORE entity write so we can recover on crash
                pending_path = None
                if current_entity is not None:
                    history_entry = self._capture_history_entry(
                        entity.id, self._version, expected_entity_version
                    )
                    if history_entry is not None:
                        pending_path = self._write_pending_history(entity.id, history_entry)

                # Phase 2: Write entity
                entity.bump_version()
                path = self._entity_path(entity.id)
                self._write_with_checksum(path, entity.to_dict())

                # Phase 3: Finalize history (move from pending to main)
                # If crash after phase 2 but before phase 3, recovery will finalize
                if pending_path is not None:
                    self._finalize_pending_history(entity.id, pending_path)

                # Increment global version
                self._version += 1
                self._save_version()

                # Invalidate cache for written entity
                self._cache_invalidate(entity.id)

                # Update indexes (failure logged but doesn't fail the write)
                if self._index_manager is not None:
                    try:
                        old_data = current_entity.to_dict() if current_entity else None
                        entity_type = getattr(entity, 'entity_type', None)
                        if entity_type:
                            self._index_manager.update_index(
                                entity_type, entity.id, old_data, entity.to_dict()
                            )
                    except Exception as e:
                        logger.warning(f"Index update failed for {entity.id}: {e}")

    def apply_writes(self, write_set: Dict[str, Entity]) -> int:
        """
        Atomically apply a set of writes.

        Thread-safe and process-safe via write locks.

        Uses atomic file operations:
        1. Validate all entities (if validate_on_save=True)
        2. Write to temp files
        3. Fsync all temp files
        4. Rename temp files to final (atomic on POSIX)
        5. Update version counter
        6. Fsync version file

        If any operation fails, all successfully renamed files are rolled back.

        Args:
            write_set: Dictionary mapping entity_id to Entity

        Returns:
            New global version after writes

        Raises:
            CorruptionError: If checksum operations fail
            ValidationError: If any entity fails validation
        """
        # Acquire both thread and process locks for full safety
        with self._write_lock:
            with self._write_process_lock:
                # Step 0: Validate all entities before any writes
                for entity in write_set.values():
                    self._validate_entity(entity)

                temp_files = []
                renamed_files = []  # Track successful renames for rollback
                pending_history = []  # (entity_id, pending_path) for crash-safe history
                old_entity_data = {}  # Capture old data for index updates

                try:
                    # Step 1: Capture history and write to pending files (crash-safe)
                    # Pending files are written BEFORE entity writes so crash recovery works
                    for entity_id, entity in write_set.items():
                        current_entity = self.read(entity_id)
                        # Capture old data for index updates
                        if current_entity is not None:
                            old_entity_data[entity_id] = current_entity.to_dict()
                        if current_entity is not None:
                            expected_version = current_entity.version + 1
                            entry = self._capture_history_entry(
                                entity_id, self._version, expected_version
                            )
                            if entry is not None:
                                pending_path = self._write_pending_history(entity_id, entry)
                                pending_history.append((entity_id, pending_path))

                    # Step 2: Write new states to temp files
                    for entity_id, entity in write_set.items():
                        # Increment entity version
                        entity.bump_version()

                        # Write to temp file
                        temp_path = self._entity_path(entity_id).with_suffix('.tmp')
                        self._write_with_checksum(temp_path, entity.to_dict())
                        temp_files.append((temp_path, self._entity_path(entity_id)))

                    # Step 3: Fsync all temp files (respects durability mode)
                    for temp_path, _ in temp_files:
                        self._fsync_file(temp_path)

                    # Step 4: Rename all temp files to final (atomic on POSIX)
                    for temp_path, final_path in temp_files:
                        self._fs.rename(temp_path, final_path)
                        renamed_files.append(final_path)

                    # Step 5: Finalize pending history files
                    # If crash after step 4 but before step 5, recovery will finalize
                    for entity_id, pending_path in pending_history:
                        self._finalize_pending_history(entity_id, pending_path)

                    # Step 6: Update global version
                    self._version += 1
                    self._save_version()

                    # Step 7: Update indexes and invalidate cache
                    # Index failures are logged but don't fail the transaction
                    for entity_id, entity in write_set.items():
                        self._cache_invalidate(entity_id)
                        if self._index_manager is not None:
                            try:
                                entity_type = getattr(entity, 'entity_type', None)
                                if entity_type:
                                    self._index_manager.update_index(
                                        entity_type, entity_id,
                                        old_entity_data.get(entity_id),
                                        entity.to_dict()
                                    )
                            except Exception as e:
                                logger.warning(f"Index update failed for {entity_id}: {e}")

                    return self._version

                except Exception:
                    # Rollback: Delete successfully renamed files
                    for final_path in renamed_files:
                        self._fs.unlink(final_path, missing_ok=True)

                    # Clean up remaining temp files
                    for temp_path, _ in temp_files:
                        self._fs.unlink(temp_path, missing_ok=True)

                    # Clean up pending history files (writes didn't complete)
                    for _, pending_path in pending_history:
                        self._fs.unlink(pending_path, missing_ok=True)

                    raise

    def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity file exists, False otherwise
        """
        return self._fs.exists(self._entity_path(entity_id))

    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity with crash-safe history persistence.

        Thread-safe and process-safe via write locks.
        Uses pending file pattern for crash-safe history (same as write()).

        Args:
            entity_id: Entity identifier

        Returns:
            True if deleted, False if not found
        """
        # Acquire both thread and process locks for full safety
        with self._write_lock:
            with self._write_process_lock:
                path = self._entity_path(entity_id)
                if not self._fs.exists(path):
                    return False

                # Capture entity before deletion for index updates
                entity = self.read(entity_id)
                old_data = entity.to_dict() if entity else None
                entity_type = getattr(entity, 'entity_type', None) if entity else None

                # Phase 1: Capture and write pending history (crash-safe)
                # Use version 0 as expected_entity_version to indicate deletion
                pending_path = None
                history_entry = self._capture_history_entry(
                    entity_id, self._version, expected_entity_version=0
                )
                if history_entry is not None:
                    pending_path = self._write_pending_history(entity_id, history_entry)

                # Phase 2: Delete file
                self._fs.unlink(path)

                # Phase 3: Finalize history (move from pending to main)
                if pending_path is not None:
                    self._finalize_pending_history(entity_id, pending_path)

                # Increment global version
                self._version += 1
                self._save_version()

                # Invalidate cache for deleted entity
                self._cache_invalidate(entity_id)

                # Update indexes (failure logged but doesn't fail the delete)
                if self._index_manager is not None and entity_type:
                    try:
                        self._index_manager.update_index(
                            entity_type, entity_id, old_data, None
                        )
                    except Exception as e:
                        logger.warning(f"Index update failed for deleted {entity_id}: {e}")

                return True

    def apply_deletes(self, delete_set: set) -> int:
        """
        Delete multiple entities atomically with crash-safe history.

        Thread-safe and process-safe via write locks.
        Either all deletes succeed or none do (rollback on failure).
        Uses pending file pattern for crash-safe history (same as apply_writes()).

        Args:
            delete_set: Set of entity IDs to delete

        Returns:
            New global version after deletes

        Raises:
            Exception: If any delete fails (with rollback attempted)
        """
        if not delete_set:
            return self._version

        # Acquire both thread and process locks for full safety
        with self._write_lock:
            with self._write_process_lock:
                deleted_files = []  # Track for rollback: (entity_id, entity_data, path)
                pending_history = []  # Track pending history: (entity_id, pending_path)
                deleted_entities = []  # Track for index updates: (entity_id, entity_type, old_data)

                try:
                    # Step 1: Capture history and write to pending files (crash-safe)
                    # Pending files are written BEFORE deletes so crash recovery works
                    for entity_id in delete_set:
                        path = self._entity_path(entity_id)
                        if self._fs.exists(path):
                            # Capture entity for index updates before deletion
                            entity = self.read(entity_id)
                            if entity:
                                entity_type = getattr(entity, 'entity_type', None)
                                deleted_entities.append((entity_id, entity_type, entity.to_dict()))

                            # Capture history entry with version 0 indicating deletion
                            entry = self._capture_history_entry(
                                entity_id, self._version, expected_entity_version=0
                            )
                            if entry is not None:
                                pending_path = self._write_pending_history(entity_id, entry)
                                pending_history.append((entity_id, pending_path))

                    # Step 2: Delete all files (capture data for rollback)
                    for entity_id in delete_set:
                        path = self._entity_path(entity_id)
                        if self._fs.exists(path):
                            # Read entity data for potential rollback
                            entity_data = self._fs.read_text(path)
                            # Delete file
                            self._fs.unlink(path)
                            deleted_files.append((entity_id, entity_data, path))

                    # Step 3: Finalize pending history files
                    # If crash after step 2 but before step 3, recovery will finalize
                    for entity_id, pending_path in pending_history:
                        self._finalize_pending_history(entity_id, pending_path)

                    # Step 4: Update global version (only if we deleted something)
                    if deleted_files:
                        self._version += 1
                        self._save_version()

                    # Step 5: Invalidate cache and update indexes for all deleted entities
                    # Index failures are logged but don't fail the transaction
                    for entity_id, _, _ in deleted_files:
                        self._cache_invalidate(entity_id)

                    if self._index_manager is not None:
                        for entity_id, entity_type, old_data in deleted_entities:
                            if entity_type:
                                try:
                                    self._index_manager.update_index(
                                        entity_type, entity_id, old_data, None
                                    )
                                except Exception as e:
                                    logger.warning(f"Index update failed for deleted {entity_id}: {e}")

                    return self._version

                except Exception:
                    # Rollback: Restore deleted files
                    for entity_id, entity_data, path in deleted_files:
                        try:
                            self._fs.write_text(path, entity_data)
                        except Exception:
                            pass  # Best effort rollback

                    # Clean up pending history files (deletes didn't complete)
                    for _, pending_path in pending_history:
                        self._fs.unlink(pending_path, missing_ok=True)

                    raise

    def _entity_path(self, entity_id: str) -> Path:
        """Get path for entity JSON file."""
        return self.store_dir / f"{entity_id}.json"

    def _history_path(self, entity_id: str) -> Path:
        """Get path for entity history file (JSONL format)."""
        return self.history_dir / f"{entity_id}.jsonl"

    def _validate_entity(self, entity: Entity) -> None:
        """
        Validate entity before writing.

        Performs two levels of validation:
        1. Basic validation (entity must have an ID)
        2. Schema validation (if SchemaRegistry is configured)

        Args:
            entity: Entity to validate

        Raises:
            ValidationError: If entity fails validation
        """
        if not self.validate_on_save:
            return

        # Basic validation - entity must have an ID
        if not entity.id:
            raise ValidationError(
                "Entity must have an ID",
                entity_type=getattr(entity, 'entity_type', 'unknown')
            )

        # Schema validation (when registry is available)
        if self._schema_registry is not None:
            entity_type = getattr(entity, 'entity_type', None)
            if entity_type and self._schema_registry.has_schema(entity_type):
                # Convert entity to dict for validation
                entity_data = entity.to_dict()
                result = self._schema_registry.validate(entity_type, entity_data)
                if not result.valid:
                    raise ValidationError(
                        f"Schema validation failed: {'; '.join(result.errors)}",
                        entity_id=entity.id,
                        entity_type=entity_type
                    )

    def _write_with_checksum(
        self, path: Path, data: dict, max_retries: int = 3
    ) -> None:
        """
        Write JSON with embedded checksum wrapper and verify after write.

        Uses write-then-verify pattern to detect corruption.

        Args:
            path: File path to write to
            data: Entity data dictionary
            max_retries: Maximum retry attempts on verification failure
        """
        checksum = compute_checksum(data)
        wrapper = {
            "_checksum": checksum,
            "_written_at": datetime.now(timezone.utc).isoformat(),
            "data": data
        }

        content = json.dumps(wrapper, indent=2, sort_keys=True)

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                # Write the file
                self._fs.write_text(path, content)

                # Fsync if PARANOID mode (immediate durability per write)
                # BALANCED mode syncs at commit time via fsync_all() instead
                if self.durability == DurabilityMode.PARANOID:
                    self._fs.fsync(path)

                # Verify by reading back and checking checksum
                read_back_content = self._fs.read_text(path)
                read_back = json.loads(read_back_content)

                if read_back.get("_checksum") != checksum:
                    raise CorruptionError(
                        f"Write verification failed for {path.name}: checksum mismatch",
                        expected_checksum=checksum,
                        actual_checksum=read_back.get("_checksum"),
                        path=str(path)
                    )

                # Verify the data portion
                actual_checksum = compute_checksum(read_back.get("data", {}))
                if actual_checksum != checksum:
                    raise CorruptionError(
                        f"Write verification failed for {path.name}: data checksum mismatch",
                        expected_checksum=checksum,
                        actual_checksum=actual_checksum,
                        path=str(path)
                    )

                # Success - write verified
                return

            except (json.JSONDecodeError, CorruptionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.01s, 0.02s, 0.04s
                    time.sleep(0.01 * (2 ** attempt))
                    continue
                raise

        if last_error:
            raise last_error

    def _read_and_verify(self, path: Path) -> dict:
        """
        Read JSON and verify checksum.

        Args:
            path: File path to read from

        Returns:
            Wrapper dictionary with verified data

        Raises:
            CorruptionError: If checksum verification fails
        """
        content = self._fs.read_text(path)
        wrapper = json.loads(content)

        expected_checksum = wrapper.get("_checksum")
        data = wrapper.get("data", {})

        actual_checksum = compute_checksum(data)
        if actual_checksum != expected_checksum:
            raise CorruptionError(
                f"Checksum mismatch for {path.name}",
                expected_checksum=expected_checksum,
                actual_checksum=actual_checksum,
                path=str(path)
            )

        return wrapper

    def _fsync_file(self, path: Path) -> None:
        """
        Ensure file is durably written to disk.

        Args:
            path: File path to sync
        """
        # Skip fsync if RELAXED mode (no fsync at all)
        # BALANCED and PARANOID both use this for batch fsync at commit
        if self.durability == DurabilityMode.RELAXED:
            return

        self._fs.fsync(path)

    def fsync_all(self) -> None:
        """
        Force fsync of all entity files and version file.

        Used by BALANCED mode to sync on transaction commit.
        """
        # Fsync all entity files
        for entity_file in self._fs.glob(self.store_dir, "*.json"):
            if entity_file.name != "_version.json":
                self._fsync_file(entity_file)

        # Fsync version file
        version_path = self.store_dir / "_version.json"
        if self._fs.exists(version_path):
            self._fsync_file(version_path)

    def _save_to_history(self, entity_id: str, global_version: int) -> None:
        """
        Append current entity version to history file before overwriting.

        Args:
            entity_id: Entity identifier
            global_version: Global version to associate with this snapshot
        """
        path = self._entity_path(entity_id)
        if not self._fs.exists(path):
            return

        wrapper = self._read_and_verify(path)
        data = wrapper["data"]

        # Append to history file (JSONL format)
        history_path = self._history_path(entity_id)
        history_entry = {
            "global_version": global_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }

        with self._history_lock:
            content = json.dumps(history_entry, sort_keys=True) + '\n'
            self._fs.append_text(history_path, content)

    def _capture_history_entry(
        self, entity_id: str, global_version: int, expected_entity_version: int
    ) -> Optional[dict]:
        """
        Capture current entity state for history WITHOUT persisting.

        This allows us to capture the "before" state and use atomic
        pending file pattern for crash-safe history persistence.

        Args:
            entity_id: Entity identifier
            global_version: Global version to associate with this snapshot
            expected_entity_version: Entity version AFTER the write completes
                                    (used for crash recovery validation)

        Returns:
            History entry dict, or None if entity doesn't exist
        """
        path = self._entity_path(entity_id)
        if not self._fs.exists(path):
            return None

        wrapper = self._read_and_verify(path)
        data = wrapper["data"]

        return {
            "global_version": global_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "expected_entity_version": expected_entity_version,  # For crash recovery
        }

    def _write_pending_history(self, entity_id: str, history_entry: dict) -> Path:
        """
        Write history entry to pending file (crash-safe first phase).

        The pending file is written and fsynced BEFORE entity write.
        On crash recovery, pending files are validated and finalized.

        Args:
            entity_id: Entity identifier
            history_entry: History entry dict from _capture_history_entry

        Returns:
            Path to the pending file
        """
        pending_path = self._pending_history_dir / f"{entity_id}.pending"

        content = json.dumps(history_entry, sort_keys=True) + '\n'
        self._fs.write_text(pending_path, content)
        # Only fsync for PARANOID (BALANCED syncs at commit)
        if self.durability == DurabilityMode.PARANOID:
            self._fs.fsync(pending_path)

        return pending_path

    def _finalize_pending_history(self, entity_id: str, pending_path: Path) -> None:
        """
        Finalize pending history by appending to main history file.

        Called AFTER entity write succeeds. Removes the pending file after
        successful append.

        Args:
            entity_id: Entity identifier
            pending_path: Path to the pending file
        """
        if not self._fs.exists(pending_path):
            return

        history_path = self._history_path(entity_id)

        # Read pending entry
        entry_line = self._fs.read_text(pending_path)

        # Parse to remove the recovery metadata before storing
        entry = json.loads(entry_line.strip())
        entry.pop("expected_entity_version", None)  # Remove recovery metadata

        # Append to main history file
        with self._history_lock:
            content = json.dumps(entry, sort_keys=True) + '\n'
            self._fs.append_text(history_path, content)
            # Only fsync for PARANOID (BALANCED syncs at commit)
            if self.durability == DurabilityMode.PARANOID:
                self._fs.fsync(history_path)

        # Remove pending file
        self._fs.unlink(pending_path)

    def _recover_pending_history(self) -> None:
        """
        Recover any pending history entries from interrupted writes/deletes.

        Called on startup to handle crash recovery. For each pending file:
        - If expected_entity_version=0 (delete) and entity doesn't exist: finalize
        - If expected_entity_version>0 (write) and entity.version matches: finalize
        - Otherwise: delete pending (operation did not complete)

        This ensures history is never lost if crash happens after entity
        write/delete but before history finalization.
        """
        if not self._fs.exists(self._pending_history_dir):
            return

        for pending_path in self._fs.glob(self._pending_history_dir, "*.pending"):
            entity_id = pending_path.stem

            try:
                content = self._fs.read_text(pending_path)
                entry = json.loads(content.strip())

                expected_version = entry.get("expected_entity_version")
                if expected_version is None:
                    # Old format pending file, delete it
                    self._fs.unlink(pending_path)
                    continue

                entity = self.read(entity_id)

                # Check if operation completed successfully
                if expected_version == 0:
                    # This was a delete operation
                    # If entity doesn't exist, delete succeeded → finalize history
                    if entity is None:
                        self._finalize_pending_history(entity_id, pending_path)
                    else:
                        # Entity still exists, delete didn't complete
                        self._fs.unlink(pending_path)
                else:
                    # This was a write operation
                    # If entity version matches expected, write succeeded → finalize
                    if entity is not None and entity.version == expected_version:
                        self._finalize_pending_history(entity_id, pending_path)
                    else:
                        # Version mismatch, write didn't complete
                        self._fs.unlink(pending_path)

            except (json.JSONDecodeError, OSError):
                # Corrupted pending file, delete it
                self._fs.unlink(pending_path, missing_ok=True)

    def _load_version(self) -> int:
        """
        Compute global version from entities and history.

        The version is computed as the maximum of:
        1. Stored value in _version.json (if exists)
        2. Count of entity files
        3. Max global_version from all history entries

        This makes the version self-healing and merge-conflict-free.

        Returns:
            Current version
        """
        # Start with stored value (if exists)
        stored_version = 0
        version_path = self.store_dir / "_version.json"
        if self._fs.exists(version_path):
            try:
                content = self._fs.read_text(version_path)
                data = json.loads(content)
                stored_version = data.get("version", 0)
            except (json.JSONDecodeError, OSError):
                pass

        # Count entity files
        entity_count = 0
        for entity_file in self._fs.glob(self.store_dir, "*.json"):
            if entity_file.name != "_version.json":
                entity_count += 1

        # Find max global_version from history files
        max_history_version = 0
        if self._fs.exists(self.history_dir):
            for history_file in self._fs.glob(self.history_dir, "*.jsonl"):
                try:
                    content = self._fs.read_text(history_file)
                    for line in content.splitlines():
                        if line.strip():
                            entry = json.loads(line)
                            gv = entry.get("global_version", 0)
                            if gv > max_history_version:
                                max_history_version = gv
                except (json.JSONDecodeError, OSError):
                    continue

        # Use max of all sources
        computed_version = max(stored_version, entity_count, max_history_version)

        # If computed is higher than stored, update the file
        if computed_version > stored_version:
            self._version = computed_version
            self._save_version()

        return computed_version

    def _save_version(self) -> None:
        """
        Save global version to _version.json.

        Uses both threading.Lock and ProcessLock for safety.
        """
        version_path = self.store_dir / "_version.json"
        data = {"version": self._version}

        with self._version_thread_lock:
            with self._version_lock:
                # Write to temp first
                temp_path = version_path.with_suffix('.tmp')
                content = json.dumps(data, indent=2, sort_keys=True)
                self._fs.write_text(temp_path, content)

                # Fsync (respects durability mode)
                self._fsync_file(temp_path)

                # Rename (atomic on POSIX)
                self._fs.rename(temp_path, version_path)

    def list_by_prefix(self, prefix: str) -> List[str]:
        """
        List entity IDs matching a prefix by scanning the directory.

        Args:
            prefix: ID prefix to match (e.g., "T-" for tasks)

        Returns:
            List of matching entity IDs
        """
        if not self._fs.exists(self.store_dir):
            return []
        return [
            f.stem for f in self._fs.glob(self.store_dir, f"{prefix}*.json")
            if not f.name.startswith("_")  # Skip _version.json etc
        ]

    def iter_entities(self, prefix: Optional[str] = None) -> List[Entity]:
        """
        Iterate over all entities, optionally filtered by prefix.

        Args:
            prefix: Optional ID prefix to filter (e.g., "T-" for tasks)

        Returns:
            List of Entity objects
        """
        if not self._fs.exists(self.store_dir):
            return []

        pattern = f"{prefix}*.json" if prefix else "*.json"
        entities = []
        for entity_file in self._fs.glob(self.store_dir, pattern):
            # Skip internal files
            if entity_file.name.startswith("_"):
                continue
            try:
                entity = self.read(entity_file.stem)
                if entity is not None:
                    entities.append(entity)
            except (CorruptionError, json.JSONDecodeError) as e:
                # Skip corrupted entities during iteration (graceful degradation)
                # Use debug level to avoid spamming output on every list operation
                logger.debug(f"Skipping corrupted entity {entity_file.stem}: {e}")
                continue
        return entities


# Alias for backward compatibility with VersionedStore API
VersionedStore = CDGStore


