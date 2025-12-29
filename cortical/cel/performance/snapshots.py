"""
Snapshot-based recovery for fast CEL startup.

The problem: Loading 100K events at startup takes seconds.
The solution: Periodic snapshots + incremental replay.

Snapshot Strategy:
    1. Full snapshot every N events (configurable)
    2. Delta snapshots between full snapshots
    3. Materialized entity cache in snapshots
    4. Compressed storage for efficiency

Recovery Process:
    1. Load most recent snapshot (fast - single file)
    2. Replay events since snapshot (incremental)
    3. Result: Full state in O(events_since_snapshot)
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
)

from ..core.events import CognitiveEvent
from ..core.references import EventHorizon


@dataclass
class SnapshotConfig:
    """Configuration for snapshot behavior."""

    # Create full snapshot every N events
    full_interval: int = 1000

    # Create delta snapshot every N events (between fulls)
    delta_interval: int = 100

    # Keep this many full snapshots
    retention_count: int = 5

    # Compress snapshots
    compress: bool = True

    # Include materialized entities in snapshot
    include_materializations: bool = True


@dataclass
class SnapshotMetadata:
    """Metadata about a snapshot."""

    snapshot_id: str
    snapshot_type: str  # 'full' or 'delta'
    event_horizon: str  # Event ID at snapshot time
    event_count: int  # Total events at snapshot time
    created_at: str  # ISO timestamp
    parent_snapshot_id: Optional[str] = None  # For delta snapshots
    compressed: bool = True
    size_bytes: int = 0
    entity_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotMetadata':
        return cls(**data)


@dataclass
class Snapshot:
    """
    A point-in-time snapshot of CEL state.

    Contains:
    - Metadata (horizon, counts, timestamps)
    - Entity index state
    - Materialized entities (optional)
    - Event IDs (for verification)
    """

    metadata: SnapshotMetadata
    entity_index: Dict[str, List[str]]  # entity_id → event_ids
    materialized_entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event_ids: List[str] = field(default_factory=list)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get materialized entity from snapshot."""
        return self.materialized_entities.get(entity_id)

    def has_entity(self, entity_id: str) -> bool:
        """Check if entity exists in snapshot."""
        return entity_id in self.entity_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': self.metadata.to_dict(),
            'entity_index': self.entity_index,
            'materialized_entities': self.materialized_entities,
            'event_ids': self.event_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Snapshot':
        return cls(
            metadata=SnapshotMetadata.from_dict(data['metadata']),
            entity_index=data.get('entity_index', {}),
            materialized_entities=data.get('materialized_entities', {}),
            event_ids=data.get('event_ids', []),
        )


class SnapshotManager:
    """
    Manages CEL snapshots for fast recovery.

    Responsibilities:
    1. Create snapshots at configured intervals
    2. Store and load snapshots efficiently
    3. Manage snapshot retention
    4. Provide recovery entry point

    Usage:
        manager = SnapshotManager(Path(".cel/snapshots"))

        # Create snapshot when needed
        if manager.should_snapshot(event_count):
            manager.create_snapshot(
                horizon=current_horizon,
                entity_index=index,
                materializer=materializer,
            )

        # Recover from snapshot
        snapshot = manager.load_latest()
        # Replay events since snapshot.metadata.event_horizon
    """

    def __init__(
        self,
        base_path: Path,
        config: Optional[SnapshotConfig] = None,
    ):
        """
        Initialize snapshot manager.

        Args:
            base_path: Directory for snapshot storage
            config: Snapshot configuration
        """
        self._base_path = Path(base_path)
        self._config = config or SnapshotConfig()
        self._lock = threading.RLock()

        # Ensure directories exist
        self._base_path.mkdir(parents=True, exist_ok=True)
        (self._base_path / "full").mkdir(exist_ok=True)
        (self._base_path / "delta").mkdir(exist_ok=True)

        # Cache of snapshot metadata
        self._metadata_cache: Dict[str, SnapshotMetadata] = {}
        self._last_full_snapshot_id: Optional[str] = None
        self._events_since_snapshot = 0

    def should_snapshot(self, event_count: int) -> str:
        """
        Check if a snapshot should be created.

        Args:
            event_count: Total events in store

        Returns:
            'full', 'delta', or 'none'
        """
        if event_count == 0:
            return 'none'

        # Check if we need a full snapshot
        if event_count % self._config.full_interval == 0:
            return 'full'

        # Check if we need a delta snapshot
        if (
            self._last_full_snapshot_id is not None and
            event_count % self._config.delta_interval == 0
        ):
            return 'delta'

        return 'none'

    def create_snapshot(
        self,
        horizon: EventHorizon,
        event_count: int,
        entity_index: Dict[str, List[str]],
        materialized_entities: Optional[Dict[str, Dict[str, Any]]] = None,
        event_ids: Optional[List[str]] = None,
        snapshot_type: str = 'full',
    ) -> SnapshotMetadata:
        """
        Create a new snapshot.

        Args:
            horizon: Current event horizon
            event_count: Total event count
            entity_index: Entity → event IDs mapping
            materialized_entities: Pre-materialized entities (optional)
            event_ids: List of all event IDs (optional, for verification)
            snapshot_type: 'full' or 'delta'

        Returns:
            Metadata for created snapshot
        """
        with self._lock:
            # Generate snapshot ID
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            snapshot_id = f"{snapshot_type}-{timestamp}-{horizon.event_id[:8]}"

            # Build metadata
            metadata = SnapshotMetadata(
                snapshot_id=snapshot_id,
                snapshot_type=snapshot_type,
                event_horizon=horizon.event_id,
                event_count=event_count,
                created_at=datetime.now().isoformat(),
                parent_snapshot_id=self._last_full_snapshot_id if snapshot_type == 'delta' else None,
                compressed=self._config.compress,
                entity_count=len(entity_index),
            )

            # Build snapshot
            snapshot = Snapshot(
                metadata=metadata,
                entity_index=entity_index,
                materialized_entities=materialized_entities or {},
                event_ids=event_ids or [],
            )

            # Write to disk
            subdir = "full" if snapshot_type == "full" else "delta"
            file_path = self._base_path / subdir / f"{snapshot_id}.json"

            self._write_snapshot(file_path, snapshot)

            # Update metadata
            metadata.size_bytes = file_path.stat().st_size
            self._metadata_cache[snapshot_id] = metadata

            # Track for delta snapshots
            if snapshot_type == 'full':
                self._last_full_snapshot_id = snapshot_id
                self._events_since_snapshot = 0
                self._cleanup_old_snapshots()

            return metadata

    def _write_snapshot(self, path: Path, snapshot: Snapshot) -> None:
        """Write snapshot to disk, optionally compressed."""
        data = json.dumps(snapshot.to_dict(), separators=(',', ':'))

        if self._config.compress:
            path = path.with_suffix('.json.gz')
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(data)
        else:
            with open(path, 'w') as f:
                f.write(data)

    def _read_snapshot(self, path: Path) -> Snapshot:
        """Read snapshot from disk."""
        if path.suffix == '.gz' or path.with_suffix('.json.gz').exists():
            actual_path = path if path.suffix == '.gz' else path.with_suffix('.json.gz')
            with gzip.open(actual_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path) as f:
                data = json.load(f)

        return Snapshot.from_dict(data)

    def load_latest(self) -> Optional[Snapshot]:
        """
        Load the most recent snapshot.

        Prefers full snapshots, falls back to delta if needed.

        Returns:
            Most recent snapshot, or None if none exist
        """
        with self._lock:
            # Find latest full snapshot
            full_dir = self._base_path / "full"
            full_files = sorted(full_dir.glob("*.json*"), reverse=True)

            if full_files:
                return self._read_snapshot(full_files[0])

            return None

    def load_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """Load a specific snapshot by ID."""
        with self._lock:
            # Check both directories
            for subdir in ["full", "delta"]:
                for ext in [".json", ".json.gz"]:
                    path = self._base_path / subdir / f"{snapshot_id}{ext}"
                    if path.exists():
                        return self._read_snapshot(path)

            return None

    def list_snapshots(self) -> List[SnapshotMetadata]:
        """List all available snapshots."""
        with self._lock:
            snapshots = []

            for subdir in ["full", "delta"]:
                dir_path = self._base_path / subdir
                for file_path in dir_path.glob("*.json*"):
                    try:
                        snapshot = self._read_snapshot(file_path)
                        snapshots.append(snapshot.metadata)
                    except Exception:
                        continue

            return sorted(snapshots, key=lambda s: s.created_at, reverse=True)

    def find_snapshot_before(self, horizon: EventHorizon) -> Optional[Snapshot]:
        """
        Find the most recent snapshot before a given horizon.

        Useful for temporal queries - load snapshot, then replay
        events up to the target horizon.

        Args:
            horizon: Target event horizon

        Returns:
            Most recent snapshot before horizon, or None
        """
        # This is a simplified implementation
        # In production, would use binary search on sorted snapshots
        snapshots = self.list_snapshots()

        for metadata in snapshots:
            if metadata.event_horizon <= horizon.event_id:
                return self.load_snapshot(metadata.snapshot_id)

        return None

    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond retention limit."""
        full_dir = self._base_path / "full"
        full_files = sorted(full_dir.glob("*.json*"), reverse=True)

        # Keep only retention_count full snapshots
        for old_file in full_files[self._config.retention_count:]:
            old_file.unlink()

        # Remove delta snapshots older than oldest kept full
        if len(full_files) >= self._config.retention_count:
            oldest_kept = full_files[self._config.retention_count - 1]
            oldest_time = oldest_kept.stat().st_mtime

            delta_dir = self._base_path / "delta"
            for delta_file in delta_dir.glob("*.json*"):
                if delta_file.stat().st_mtime < oldest_time:
                    delta_file.unlink()

    @property
    def snapshot_count(self) -> int:
        """Total number of snapshots."""
        count = 0
        for subdir in ["full", "delta"]:
            count += len(list((self._base_path / subdir).glob("*.json*")))
        return count

    @property
    def total_size_bytes(self) -> int:
        """Total disk usage of all snapshots."""
        total = 0
        for subdir in ["full", "delta"]:
            for f in (self._base_path / subdir).glob("*.json*"):
                total += f.stat().st_size
        return total


class SnapshotRecovery:
    """
    Recovery coordinator that uses snapshots for fast startup.

    Usage:
        recovery = SnapshotRecovery(snapshot_manager, event_store)

        # Fast recovery
        state = recovery.recover()
        # state.entity_index is ready
        # state.events_to_replay are the events since snapshot
    """

    @dataclass
    class RecoveryState:
        """Result of snapshot recovery."""

        snapshot: Optional[Snapshot]
        events_to_replay: List[str]  # Event IDs to replay
        recovered_entities: int
        replay_count: int

    def __init__(
        self,
        snapshot_manager: SnapshotManager,
        event_iterator: Callable[[Optional[str]], Iterator[CognitiveEvent]],
    ):
        """
        Initialize recovery coordinator.

        Args:
            snapshot_manager: Snapshot manager
            event_iterator: Function that returns events after a given ID
        """
        self._snapshots = snapshot_manager
        self._event_iterator = event_iterator

    def recover(self) -> 'SnapshotRecovery.RecoveryState':
        """
        Perform recovery from latest snapshot.

        Returns:
            RecoveryState with snapshot and events to replay
        """
        # Load latest snapshot
        snapshot = self._snapshots.load_latest()

        if snapshot is None:
            # No snapshot - full replay needed
            return self.RecoveryState(
                snapshot=None,
                events_to_replay=[],
                recovered_entities=0,
                replay_count=0,
            )

        # Find events since snapshot
        events_to_replay = []
        for event in self._event_iterator(snapshot.metadata.event_horizon):
            events_to_replay.append(event.id)

        return self.RecoveryState(
            snapshot=snapshot,
            events_to_replay=events_to_replay,
            recovered_entities=snapshot.metadata.entity_count,
            replay_count=len(events_to_replay),
        )

    def recover_to_horizon(
        self,
        target: EventHorizon,
    ) -> 'SnapshotRecovery.RecoveryState':
        """
        Recover to a specific point in time.

        Finds the best snapshot before the target and returns
        events needed to replay to reach the target.
        """
        snapshot = self._snapshots.find_snapshot_before(target)

        if snapshot is None:
            return self.RecoveryState(
                snapshot=None,
                events_to_replay=[],
                recovered_entities=0,
                replay_count=0,
            )

        # Collect events between snapshot and target
        events_to_replay = []
        for event in self._event_iterator(snapshot.metadata.event_horizon):
            events_to_replay.append(event.id)
            if event.id == target.event_id:
                break

        return self.RecoveryState(
            snapshot=snapshot,
            events_to_replay=events_to_replay,
            recovered_entities=snapshot.metadata.entity_count,
            replay_count=len(events_to_replay),
        )
