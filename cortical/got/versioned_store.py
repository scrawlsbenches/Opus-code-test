"""
File-based storage with versioning and checksums for GoT transactional system.

Provides ACID-compliant storage using atomic file operations, checksums for
integrity verification, and append-only history for snapshot isolation.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

from .types import Entity, Task, Decision, Edge, Sprint, Epic, Handoff, ClaudeMdLayer, ClaudeMdVersion, PersonaProfile, Team
from .errors import CorruptionError
from cortical.utils.checksums import compute_checksum
from .config import DurabilityMode


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

    Storage layout:
        {store_dir}/
            {entity_id}.json          # Current entity state
            _version.json             # Global version counter
            _history/
                {entity_id}.jsonl     # Historical snapshots
    """

    def __init__(self, store_dir: Path, durability: DurabilityMode = DurabilityMode.BALANCED):
        """
        Initialize store, creating directory structure if needed.

        Args:
            store_dir: Directory path for storing entities
            durability: Durability mode controlling fsync behavior
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.durability = durability
        self.history_dir = self.store_dir / "_history"
        self.history_dir.mkdir(exist_ok=True)
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
        return self._entity_from_dict(wrapper["data"])

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
            we assume they existed since version 1. This is a known
            limitation for the MVP implementation.
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
                return self._entity_from_dict(matching_entry["data"])

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
        """
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
        1. Write to temp files
        2. Fsync all temp files
        3. Rename temp files to final (atomic on POSIX)
        4. Update version counter
        5. Fsync version file

        If any operation fails, all successfully renamed files are rolled back
        to ensure no partial state persists.

        Args:
            write_set: Dictionary mapping entity_id to Entity

        Returns:
            New global version after writes

        Raises:
            CorruptionError: If checksum operations fail
            Exception: Any error during write (all changes rolled back)
        """
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
                renamed_files.append(final_path)  # Track for rollback

            # Step 4: Update global version
            self._version += 1
            self._save_version()

            return self._version

        except Exception:
            # Rollback: Delete successfully renamed files to avoid partial state
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

    def _write_with_checksum(self, path: Path, data: dict) -> None:
        """
        Write JSON with embedded checksum wrapper.

        Args:
            path: File path to write to
            data: Entity data dictionary
        """
        checksum = compute_checksum(data)
        wrapper = {
            "_checksum": checksum,
            "_written_at": datetime.now(timezone.utc).isoformat(),
            "data": data
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(wrapper, f, indent=2, sort_keys=True)

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
                expected=expected_checksum,
                actual=actual_checksum,
                path=str(path)
            )

        return wrapper

    def _fsync_file(self, path: Path) -> None:
        """
        Ensure file is durably written to disk using os.fsync.

        Args:
            path: File path to sync
        """
        # Skip fsync if RELAXED mode
        if self.durability == DurabilityMode.RELAXED:
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
        # Read current entity file
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

        with open(history_path, 'a', encoding='utf-8') as f:
            json.dump(history_entry, f, sort_keys=True)
            f.write('\n')

    def _load_version(self) -> int:
        """
        Load global version from _version.json.

        Returns:
            Current version, or 0 if not found
        """
        version_path = self.store_dir / "_version.json"
        if not version_path.exists():
            return 0

        with open(version_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("version", 0)

    def _save_version(self) -> None:
        """Save global version to _version.json."""
        version_path = self.store_dir / "_version.json"
        data = {"version": self._version}

        # Write to temp first
        temp_path = version_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)

        # Fsync (respects durability mode)
        self._fsync_file(temp_path)

        # Rename (atomic on POSIX)
        temp_path.rename(version_path)

    def _entity_from_dict(self, data: dict) -> Entity:
        """
        Factory method to create correct entity subclass based on entity_type.

        Args:
            data: Entity data dictionary

        Returns:
            Appropriate Entity subclass instance
        """
        entity_type = data.get("entity_type", "")

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
        else:
            return Entity.from_dict(data)
