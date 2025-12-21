"""
Conflict resolution for GoT sync operations.

Provides strategies for resolving conflicts between local and remote entity versions
during git-based synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .types import Entity
from .errors import ConflictError


class ConflictStrategy(Enum):
    """Strategy for resolving sync conflicts."""

    OURS = "ours"      # Keep local version
    THEIRS = "theirs"  # Take remote version
    MERGE = "merge"    # Attempt field-level merge


@dataclass
class SyncConflict:
    """
    Represents a conflict between local and remote entity versions.

    Attributes:
        entity_id: Unique identifier of conflicting entity
        entity_type: Type of entity (task, decision, edge)
        local_version: Version number in local state
        remote_version: Version number in remote state
        conflict_fields: List of fields that differ between versions
    """

    entity_id: str
    entity_type: str
    local_version: int
    remote_version: int
    conflict_fields: List[str]


class ConflictResolver:
    """
    Resolves sync conflicts between local and remote state.

    Supports three strategies:
    - OURS: Keep local version (reject remote changes)
    - THEIRS: Take remote version (overwrite local changes)
    - MERGE: Combine non-overlapping changes, fail on same-field conflicts
    """

    def __init__(self, strategy: ConflictStrategy = ConflictStrategy.OURS):
        """
        Initialize conflict resolver with default strategy.

        Args:
            strategy: Default strategy for conflict resolution
        """
        self.strategy = strategy

    def detect_conflicts(
        self,
        local_entities: Dict[str, Entity],
        remote_entities: Dict[str, Entity]
    ) -> List[SyncConflict]:
        """
        Detect conflicts between local and remote entities.

        A conflict exists when:
        1. Both local and remote have the same entity ID
        2. Their version numbers differ
        3. Their content differs (not just metadata timestamps)

        Args:
            local_entities: Dictionary of local entities by ID
            remote_entities: Dictionary of remote entities by ID

        Returns:
            List of detected conflicts (empty if none)
        """
        conflicts = []

        # Check entities present in both local and remote
        common_ids = set(local_entities.keys()) & set(remote_entities.keys())

        for entity_id in common_ids:
            local = local_entities[entity_id]
            remote = remote_entities[entity_id]

            # Version mismatch indicates potential conflict
            if local.version != remote.version:
                # Find which fields differ
                conflict_fields = self._find_conflicting_fields(local, remote)

                if conflict_fields:
                    conflicts.append(SyncConflict(
                        entity_id=entity_id,
                        entity_type=local.entity_type,
                        local_version=local.version,
                        remote_version=remote.version,
                        conflict_fields=conflict_fields
                    ))

        return conflicts

    def resolve(
        self,
        conflict: SyncConflict,
        local: Entity,
        remote: Entity,
        strategy: Optional[ConflictStrategy] = None
    ) -> Entity:
        """
        Resolve a single conflict using specified strategy.

        Args:
            conflict: Conflict to resolve
            local: Local entity version
            remote: Remote entity version
            strategy: Strategy to use (defaults to self.strategy)

        Returns:
            Resolved entity

        Raises:
            ConflictError: If MERGE strategy encounters same-field conflict
        """
        used_strategy = strategy or self.strategy

        if used_strategy == ConflictStrategy.OURS:
            return local

        elif used_strategy == ConflictStrategy.THEIRS:
            return remote

        elif used_strategy == ConflictStrategy.MERGE:
            return self._merge_entities(conflict, local, remote)

        else:
            raise ValueError(f"Unknown strategy: {used_strategy}")

    def resolve_all(
        self,
        conflicts: List[SyncConflict],
        local_entities: Dict[str, Entity],
        remote_entities: Dict[str, Entity]
    ) -> Dict[str, Entity]:
        """
        Resolve all conflicts using default strategy.

        Args:
            conflicts: List of conflicts to resolve
            local_entities: Dictionary of local entities
            remote_entities: Dictionary of remote entities

        Returns:
            Dictionary of resolved entities

        Raises:
            ConflictError: If any conflict cannot be resolved
        """
        resolved = {}

        for conflict in conflicts:
            local = local_entities[conflict.entity_id]
            remote = remote_entities[conflict.entity_id]

            resolved[conflict.entity_id] = self.resolve(conflict, local, remote)

        return resolved

    def _find_conflicting_fields(self, local: Entity, remote: Entity) -> List[str]:
        """
        Find fields that differ between local and remote entities.

        Ignores metadata fields (created_at, modified_at) as these are expected
        to differ. Focuses on content fields that represent actual conflicts.

        Args:
            local: Local entity
            remote: Remote entity

        Returns:
            List of field names that conflict
        """
        # Get dictionaries for comparison
        local_dict = local.to_dict()
        remote_dict = remote.to_dict()

        # Ignore metadata fields
        ignore_fields = {'created_at', 'modified_at', 'version'}

        conflicting = []
        all_fields = set(local_dict.keys()) | set(remote_dict.keys())

        for field in all_fields:
            if field in ignore_fields:
                continue

            local_value = local_dict.get(field)
            remote_value = remote_dict.get(field)

            if local_value != remote_value:
                conflicting.append(field)

        return conflicting

    def _merge_entities(
        self,
        conflict: SyncConflict,
        local: Entity,
        remote: Entity
    ) -> Entity:
        """
        Merge local and remote entities by combining non-overlapping changes.

        Args:
            conflict: Conflict metadata
            local: Local entity
            remote: Remote entity

        Returns:
            Merged entity

        Raises:
            ConflictError: If same field differs in both versions
        """
        # For MERGE strategy, we can't resolve if the same content field
        # differs - this requires manual intervention
        if conflict.conflict_fields:
            raise ConflictError(
                f"Cannot auto-merge entity {conflict.entity_id}: "
                f"conflicting fields {conflict.conflict_fields}",
                entity_id=conflict.entity_id,
                conflict_fields=conflict.conflict_fields
            )

        # If no content conflicts (only version/metadata differs),
        # prefer higher version number
        if local.version >= remote.version:
            return local
        else:
            return remote
