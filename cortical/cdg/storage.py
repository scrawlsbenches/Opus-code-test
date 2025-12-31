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

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any, Type

from cortical.utils.checksums import compute_checksum
from cortical.utils.locking import ProcessLock

from .types import Entity
from .errors import CorruptionError, ValidationError, StorageError
from .config import CDGConfig, DurabilityMode


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
        store = CDGStore(Path("./data"))

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

        store = CDGStore(
            Path("./data"),
            entity_factory=got_entity_factory
        )

    Attributes:
        store_dir: Directory path for storing entities
        config: CDG configuration
        entity_factory: Function to create entities from dicts
    """

    def __init__(
        self,
        store_dir: Path,
        config: Optional[CDGConfig] = None,
        entity_factory: Optional[EntityFactory] = None,
        # Legacy parameters for VersionedStore compatibility
        durability: Optional[DurabilityMode] = None,
        validate_on_save: bool = True,
    ):
        """
        Initialize store, creating directory structure if needed.

        Args:
            store_dir: Directory path for storing entities
            config: CDG configuration (optional, creates default if not provided)
            entity_factory: Function to create Entity from dict (optional)
            durability: Legacy parameter for VersionedStore compatibility
            validate_on_save: Legacy parameter for VersionedStore compatibility
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Handle configuration - support both new CDGConfig and legacy parameters
        if config is not None:
            self.config = config
        else:
            # Create config from legacy parameters or defaults
            self.config = CDGConfig(
                durability=durability or DurabilityMode.BALANCED,
                validate_on_write=validate_on_save,
            )

        # For legacy compatibility
        self.durability = self.config.durability
        self.validate_on_save = self.config.validate_on_write

        # Entity factory for creating entities from dicts
        self.entity_factory = entity_factory or default_entity_factory

        # History directory for MVCC snapshots
        self.history_dir = self.store_dir / "_history"
        self.history_dir.mkdir(exist_ok=True)

        # Process lock for concurrent history file access protection
        self._history_lock = ProcessLock(self.history_dir / ".history.lock")

        # Thread lock for concurrent version file access protection (within same process)
        self._version_thread_lock = threading.Lock()

        # Process lock for cross-process version file protection
        self._version_lock = ProcessLock(self.store_dir / ".version.lock", reentrant=False)

        # Load current version
        self._version = self._load_version()

    def current_version(self) -> int:
        """
        Get current global version.

        Returns:
            Current global version number
        """
        return self._version

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
        path = self._entity_path(entity_id)
        if not path.exists():
            return None

        wrapper = self._read_and_verify(path)
        return self.entity_factory(wrapper["data"])

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

        if history_path.exists():
            # Find entry with highest global_version <= version
            matching_entry = None
            with open(history_path, 'r', encoding='utf-8') as f:
                for line in f:
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

        Args:
            entity: Entity to write

        Raises:
            CorruptionError: If checksum operations fail
            ValidationError: If entity fails validation (when validate_on_save=True)
        """
        # Validate entity before writing
        self._validate_entity(entity)

        # Save current state to history before overwriting
        if self.exists(entity.id):
            self._save_to_history(entity.id, self._version)

        # Increment entity version
        entity.bump_version()

        # Write entity to file
        path = self._entity_path(entity.id)
        self._write_with_checksum(path, entity.to_dict())

        # Increment global version
        self._version += 1
        self._save_version()

    def apply_writes(self, write_set: Dict[str, Entity]) -> int:
        """
        Atomically apply a set of writes.

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
        # Step 0: Validate all entities before any writes
        for entity in write_set.values():
            self._validate_entity(entity)

        temp_files = []
        renamed_files = []  # Track successful renames for rollback

        try:
            # Step 1: Save old states to history and write new states to temp files
            for entity_id, entity in write_set.items():
                # Save current state to history if entity exists
                if self.exists(entity_id):
                    self._save_to_history(entity_id, self._version)

                # Increment entity version
                entity.bump_version()

                # Write to temp file
                temp_path = self._entity_path(entity_id).with_suffix('.tmp')
                self._write_with_checksum(temp_path, entity.to_dict())
                temp_files.append((temp_path, self._entity_path(entity_id)))

            # Step 2: Fsync all temp files (respects durability mode)
            for temp_path, _ in temp_files:
                self._fsync_file(temp_path)

            # Step 3: Rename all temp files to final (atomic on POSIX)
            for temp_path, final_path in temp_files:
                temp_path.rename(final_path)
                renamed_files.append(final_path)

            # Step 4: Update global version
            self._version += 1
            self._save_version()

            return self._version

        except Exception:
            # Rollback: Delete successfully renamed files
            for final_path in renamed_files:
                if final_path.exists():
                    final_path.unlink()

            # Clean up remaining temp files
            for temp_path, _ in temp_files:
                if temp_path.exists():
                    temp_path.unlink()
            raise

    def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity file exists, False otherwise
        """
        return self._entity_path(entity_id).exists()

    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            True if deleted, False if not found
        """
        path = self._entity_path(entity_id)
        if not path.exists():
            return False

        # Save to history before deleting
        self._save_to_history(entity_id, self._version)

        # Delete file
        path.unlink()

        # Increment global version
        self._version += 1
        self._save_version()

        return True

    def _entity_path(self, entity_id: str) -> Path:
        """Get path for entity JSON file."""
        return self.store_dir / f"{entity_id}.json"

    def _history_path(self, entity_id: str) -> Path:
        """Get path for entity history file (JSONL format)."""
        return self.history_dir / f"{entity_id}.jsonl"

    def _validate_entity(self, entity: Entity) -> None:
        """
        Validate entity before writing.

        Override this method for custom validation logic.

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

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                # Write the file
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(wrapper, f, indent=2, sort_keys=True)
                    f.flush()
                    # Only fsync if durability mode requires it
                    if self.durability != DurabilityMode.FAST:
                        os.fsync(f.fileno())

                # Verify by reading back and checking checksum
                with open(path, 'r', encoding='utf-8') as f:
                    read_back = json.load(f)

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
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

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
        # Skip fsync if FAST mode
        if self.durability == DurabilityMode.FAST:
            return

        with open(path, 'r+', encoding='utf-8') as f:
            os.fsync(f.fileno())

    def fsync_all(self) -> None:
        """
        Force fsync of all entity files and version file.

        Used by BALANCED mode to sync on transaction commit.
        """
        # Fsync all entity files
        for entity_file in self.store_dir.glob("*.json"):
            if entity_file.name != "_version.json":
                self._fsync_file(entity_file)

        # Fsync version file
        version_path = self.store_dir / "_version.json"
        if version_path.exists():
            self._fsync_file(version_path)

    def _save_to_history(self, entity_id: str, global_version: int) -> None:
        """
        Append current entity version to history file before overwriting.

        Args:
            entity_id: Entity identifier
            global_version: Global version to associate with this snapshot
        """
        path = self._entity_path(entity_id)
        if not path.exists():
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
            with open(history_path, 'a', encoding='utf-8') as f:
                json.dump(history_entry, f, sort_keys=True)
                f.write('\n')

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
        if version_path.exists():
            try:
                with open(version_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                stored_version = data.get("version", 0)
            except (json.JSONDecodeError, OSError):
                pass

        # Count entity files
        entity_count = 0
        for entity_file in self.store_dir.glob("*.json"):
            if entity_file.name != "_version.json":
                entity_count += 1

        # Find max global_version from history files
        max_history_version = 0
        if self.history_dir.exists():
            for history_file in self.history_dir.glob("*.jsonl"):
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        for line in f:
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
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, sort_keys=True)

                # Fsync (respects durability mode)
                self._fsync_file(temp_path)

                # Rename (atomic on POSIX)
                temp_path.rename(version_path)


# Alias for backward compatibility with VersionedStore API
VersionedStore = CDGStore
