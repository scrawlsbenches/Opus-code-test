"""
Write-Ahead Logging (WAL) for Cortical Text Processor.

Provides fault-tolerant persistence through:
- Append-only operation log (WAL)
- Periodic snapshots for fast recovery
- Crash recovery by replaying WAL from last snapshot

Usage:
    from cortical.wal import WALWriter, SnapshotManager, WALRecovery

    # Writing operations
    writer = WALWriter("corpus_wal")
    writer.append(WALEntry(operation="add_document", doc_id="doc1", payload={...}))

    # Creating snapshots
    mgr = SnapshotManager("corpus_wal")
    mgr.create_snapshot(processor_state)

    # Recovering after crash
    recovery = WALRecovery("corpus_wal")
    if recovery.needs_recovery():
        state = recovery.recover()

Author: Cortical Text Processor Team
"""

import hashlib
import json
import gzip
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import os
import tempfile


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class WALEntry:
    """A single entry in the Write-Ahead Log."""

    operation: str  # add_document, remove_document, compute_phase, mark_stale, mark_fresh
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    doc_id: Optional[str] = None
    phase: Optional[str] = None
    reason: Optional[str] = None
    affected_computations: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        """Compute checksum if not provided."""
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of entry content."""
        content = json.dumps({
            'operation': self.operation,
            'timestamp': self.timestamp,
            'doc_id': self.doc_id,
            'phase': self.phase,
            'reason': self.reason,
            'affected_computations': self.affected_computations,
            'payload': self.payload,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'WALEntry':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def verify(self) -> bool:
        """Verify checksum matches content."""
        expected = self._compute_checksum()
        return self.checksum == expected


@dataclass
class SnapshotInfo:
    """Metadata about a snapshot."""

    snapshot_id: str
    timestamp: str
    document_count: int
    size_bytes: int
    operations_since_last: int
    wal_file: str
    wal_offset: int
    path: Path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'document_count': self.document_count,
            'size_bytes': self.size_bytes,
            'operations_since_last': self.operations_since_last,
            'wal_file': self.wal_file,
            'wal_offset': self.wal_offset,
            'path': str(self.path),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotInfo':
        """Create from dictionary."""
        data['path'] = Path(data['path'])
        return cls(**data)


@dataclass
class WALIndex:
    """Index tracking latest snapshot and WAL state."""

    latest_snapshot_id: Optional[str] = None
    current_wal_file: str = ""
    wal_entry_count: int = 0
    last_compaction: Optional[str] = None
    snapshots: List[str] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WALIndex':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RecoveryResult:
    """Result of crash recovery."""

    success: bool
    snapshot_id: Optional[str] = None
    wal_entries_replayed: int = 0
    documents_recovered: int = 0
    errors: List[str] = field(default_factory=list)
    state: Optional[Dict[str, Any]] = None


# ==============================================================================
# WAL WRITER
# ==============================================================================

class WALWriter:
    """Append-only Write-Ahead Log writer."""

    def __init__(self, wal_dir: str):
        """
        Initialize WAL writer.

        Args:
            wal_dir: Directory for WAL files
        """
        self.wal_dir = Path(wal_dir)
        self.logs_dir = self.wal_dir / "logs"
        self.index_path = self.wal_dir / "wal_index.json"

        # Create directories
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Load or create index
        self.index = self._load_index()

        # Ensure we have a current WAL file
        if not self.index.current_wal_file:
            self._rotate()

        self._current_file: Optional[Path] = None

    def _load_index(self) -> WALIndex:
        """Load WAL index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return WALIndex.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        return WALIndex()

    def _save_index(self) -> None:
        """Save WAL index to disk atomically, preserving fields set by other components."""
        # Reload from disk to get latest state (SnapshotManager may have updated it)
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    disk_index = WALIndex.from_dict(json.load(f))
                # Preserve snapshot-related fields from disk
                self.index.latest_snapshot_id = disk_index.latest_snapshot_id
                self.index.snapshots = disk_index.snapshots
                self.index.last_compaction = disk_index.last_compaction
            except (json.JSONDecodeError, KeyError):
                pass

        temp_path = self.index_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.index.to_dict(), f, indent=2)
        temp_path.replace(self.index_path)

    def _rotate(self) -> None:
        """Create a new WAL file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        seq = len(list(self.logs_dir.glob("wal_*.jsonl"))) + 1
        filename = f"wal_{timestamp}_{seq:03d}.jsonl"

        self.index.current_wal_file = filename
        self.index.wal_entry_count = 0
        self._save_index()

    def append(self, entry: WALEntry) -> None:
        """
        Append an entry to the WAL.

        Args:
            entry: WALEntry to append
        """
        wal_path = self.logs_dir / self.index.current_wal_file

        # Write entry atomically
        json_line = entry.to_json() + "\n"
        with open(wal_path, 'a') as f:
            f.write(json_line)
            f.flush()
            os.fsync(f.fileno())

        self.index.wal_entry_count += 1

        # Rotate if file gets too large (10MB)
        if wal_path.exists() and wal_path.stat().st_size > 10 * 1024 * 1024:
            self._rotate()

        self._save_index()

    def get_current_wal_path(self) -> Path:
        """Get path to current WAL file."""
        return self.logs_dir / self.index.current_wal_file

    def get_entries_since(
        self,
        wal_file: str,
        offset: int = 0
    ) -> Iterator[WALEntry]:
        """
        Get WAL entries since a specific point.

        Args:
            wal_file: WAL file to start from
            offset: Line offset in that file

        Yields:
            WALEntry objects
        """
        wal_files = sorted(self.logs_dir.glob("wal_*.jsonl"))

        # Find starting file index
        start_idx = 0
        for i, f in enumerate(wal_files):
            if f.name == wal_file:
                start_idx = i
                break

        # Iterate through files
        for i, wal_path in enumerate(wal_files[start_idx:]):
            line_offset = offset if i == 0 else 0

            if not wal_path.exists():
                continue

            with open(wal_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num < line_offset:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = WALEntry.from_json(line)
                        if entry.verify():
                            yield entry
                    except (json.JSONDecodeError, TypeError):
                        continue

    def get_entry_count(self) -> int:
        """Get total number of entries in current WAL."""
        return self.index.wal_entry_count


# ==============================================================================
# SNAPSHOT MANAGER
# ==============================================================================

class SnapshotManager:
    """Manages processor state snapshots."""

    def __init__(self, wal_dir: str, max_snapshots: int = 3):
        """
        Initialize snapshot manager.

        Args:
            wal_dir: Directory for WAL and snapshots
            max_snapshots: Maximum number of snapshots to keep
        """
        self.wal_dir = Path(wal_dir)
        self.snapshots_dir = self.wal_dir / "snapshots"
        self.index_path = self.wal_dir / "wal_index.json"
        self.max_snapshots = max_snapshots

        # Create directories
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> WALIndex:
        """Load WAL index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return WALIndex.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        return WALIndex()

    def _save_index(self, index: WALIndex) -> None:
        """Save WAL index."""
        temp_path = self.index_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(index.to_dict(), f, indent=2)
        temp_path.replace(self.index_path)

    def create_snapshot(
        self,
        state: Dict[str, Any],
        wal_file: str = "",
        wal_offset: int = 0,
        compress: bool = True
    ) -> str:
        """
        Create a new snapshot.

        Args:
            state: Full processor state to snapshot
            wal_file: Current WAL file name
            wal_offset: Current position in WAL
            compress: Whether to gzip the snapshot

        Returns:
            Snapshot ID
        """
        timestamp = datetime.now()
        # Include microseconds to ensure unique IDs even in rapid succession
        snapshot_id = f"snap_{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond:06d}"

        # Build snapshot data
        snapshot_data = {
            'snapshot_id': snapshot_id,
            'timestamp': timestamp.isoformat(),
            'version': '2.3',
            'state': state,
            'wal_reference': {
                'wal_file': wal_file,
                'wal_offset': wal_offset,
            },
            'metadata': {
                'document_count': len(state.get('documents', {})),
            }
        }

        # Write snapshot
        if compress:
            snapshot_path = self.snapshots_dir / f"{snapshot_id}.json.gz"
            with gzip.open(snapshot_path, 'wt', encoding='utf-8') as f:
                json.dump(snapshot_data, f)
        else:
            snapshot_path = self.snapshots_dir / f"{snapshot_id}.json"
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)

        # Update index
        index = self._load_index()
        index.latest_snapshot_id = snapshot_id
        if snapshot_id not in index.snapshots:
            index.snapshots.append(snapshot_id)
        self._save_index(index)

        # Prune old snapshots
        self._prune_snapshots()

        return snapshot_id

    def load_snapshot(self, snapshot_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load a snapshot.

        Args:
            snapshot_id: Snapshot to load (latest if None)

        Returns:
            Snapshot data dict, or None if not found
        """
        if snapshot_id is None:
            index = self._load_index()
            snapshot_id = index.latest_snapshot_id

        if not snapshot_id:
            return None

        # Try compressed first
        gz_path = self.snapshots_dir / f"{snapshot_id}.json.gz"
        if gz_path.exists():
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                return json.load(f)

        # Try uncompressed
        json_path = self.snapshots_dir / f"{snapshot_id}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)

        return None

    def list_snapshots(self) -> List[SnapshotInfo]:
        """List all available snapshots."""
        snapshots = []

        for path in sorted(self.snapshots_dir.glob("snap_*.json*")):
            try:
                # Load snapshot to get metadata
                if path.suffix == '.gz':
                    with gzip.open(path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(path, 'r') as f:
                        data = json.load(f)

                wal_ref = data.get('wal_reference', {})
                metadata = data.get('metadata', {})

                info = SnapshotInfo(
                    snapshot_id=data['snapshot_id'],
                    timestamp=data['timestamp'],
                    document_count=metadata.get('document_count', 0),
                    size_bytes=path.stat().st_size,
                    operations_since_last=0,
                    wal_file=wal_ref.get('wal_file', ''),
                    wal_offset=wal_ref.get('wal_offset', 0),
                    path=path,
                )
                snapshots.append(info)
            except (json.JSONDecodeError, KeyError):
                continue

        return snapshots

    def _prune_snapshots(self) -> None:
        """Remove old snapshots beyond max_snapshots."""
        snapshots = self.list_snapshots()

        if len(snapshots) <= self.max_snapshots:
            return

        # Sort by timestamp (oldest first)
        snapshots.sort(key=lambda s: s.timestamp)

        # Remove oldest
        to_remove = snapshots[:-self.max_snapshots]
        for snap in to_remove:
            if snap.path.exists():
                snap.path.unlink()

        # Update index
        index = self._load_index()
        kept_ids = {s.snapshot_id for s in snapshots[-self.max_snapshots:]}
        index.snapshots = [sid for sid in index.snapshots if sid in kept_ids]
        self._save_index(index)

    def compact_wal(self, state: Dict[str, Any]) -> str:
        """
        Compact WAL by creating a new snapshot and removing old WAL files.

        Args:
            state: Current processor state

        Returns:
            New snapshot ID
        """
        # Create new snapshot
        snapshot_id = self.create_snapshot(state)

        # Remove old WAL files
        logs_dir = self.wal_dir / "logs"
        if logs_dir.exists():
            for wal_file in logs_dir.glob("wal_*.jsonl"):
                wal_file.unlink()

        # Update index
        index = self._load_index()
        index.wal_entry_count = 0
        index.last_compaction = datetime.now().isoformat()
        self._save_index(index)

        return snapshot_id


# ==============================================================================
# WAL RECOVERY
# ==============================================================================

class WALRecovery:
    """Handles crash recovery using WAL and snapshots."""

    def __init__(self, wal_dir: str):
        """
        Initialize WAL recovery.

        Args:
            wal_dir: Directory containing WAL and snapshots
        """
        self.wal_dir = Path(wal_dir)
        self.index_path = self.wal_dir / "wal_index.json"
        self.snapshot_mgr = SnapshotManager(wal_dir)

    def _load_index(self) -> WALIndex:
        """Load WAL index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return WALIndex.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        return WALIndex()

    def needs_recovery(self) -> bool:
        """
        Check if recovery is needed.

        Returns:
            True if there are unapplied WAL entries
        """
        index = self._load_index()

        # No WAL = no recovery needed
        if not index.current_wal_file:
            return False

        # Check for unapplied entries
        if index.wal_entry_count > 0:
            return True

        return False

    def get_recovery_info(self) -> Dict[str, Any]:
        """
        Get information about what recovery would do.

        Returns:
            Dict with recovery plan details
        """
        index = self._load_index()
        snapshots = self.snapshot_mgr.list_snapshots()

        return {
            'has_wal': bool(index.current_wal_file),
            'wal_entries': index.wal_entry_count,
            'latest_snapshot': index.latest_snapshot_id,
            'snapshot_count': len(snapshots),
            'needs_recovery': self.needs_recovery(),
        }

    def recover(self) -> RecoveryResult:
        """
        Perform crash recovery.

        Returns:
            RecoveryResult with recovery status and state
        """
        result = RecoveryResult(success=False)
        errors = []

        # Load latest snapshot
        snapshot_data = self.snapshot_mgr.load_snapshot()
        if not snapshot_data:
            result.errors.append("No snapshot found for recovery")
            return result

        result.snapshot_id = snapshot_data.get('snapshot_id')
        state = snapshot_data.get('state', {})
        wal_ref = snapshot_data.get('wal_reference', {})

        # Get WAL entries to replay
        wal_file = wal_ref.get('wal_file', '')
        wal_offset = wal_ref.get('wal_offset', 0)

        if wal_file:
            wal_writer = WALWriter(str(self.wal_dir))
            entries = list(wal_writer.get_entries_since(wal_file, wal_offset))

            # Replay entries
            for entry in entries:
                try:
                    self._apply_entry(entry, state)
                    result.wal_entries_replayed += 1
                except (KeyError, ValueError, AttributeError, TypeError) as e:
                    errors.append(f"Error replaying entry {entry.sequence_number if hasattr(entry, 'sequence_number') else 'unknown'}: {type(e).__name__}: {e}")

        result.state = state
        result.documents_recovered = len(state.get('documents', {}))
        result.errors = errors
        result.success = True

        return result

    def _apply_entry(self, entry: WALEntry, state: Dict[str, Any]) -> None:
        """
        Apply a single WAL entry to state.

        Args:
            entry: WALEntry to apply
            state: State dict to modify
        """
        if entry.operation == 'add_document':
            doc_id = entry.doc_id
            payload = entry.payload
            if doc_id and 'content' in payload:
                if 'documents' not in state:
                    state['documents'] = {}
                state['documents'][doc_id] = payload['content']

        elif entry.operation == 'remove_document':
            doc_id = entry.doc_id
            if doc_id and 'documents' in state:
                state['documents'].pop(doc_id, None)

        elif entry.operation == 'mark_stale':
            if 'stale_computations' not in state:
                state['stale_computations'] = []
            for comp in entry.affected_computations:
                if comp not in state['stale_computations']:
                    state['stale_computations'].append(comp)

        elif entry.operation == 'mark_fresh':
            if 'stale_computations' in state:
                for comp in entry.affected_computations:
                    if comp in state['stale_computations']:
                        state['stale_computations'].remove(comp)

        elif entry.operation == 'compute_phase':
            # Record that phase completed
            if 'completed_phases' not in state:
                state['completed_phases'] = []
            phase = entry.phase
            if phase and phase not in state['completed_phases']:
                state['completed_phases'].append(phase)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def log_add_document(wal_writer: WALWriter, doc_id: str, content: str) -> None:
    """Log a document addition to WAL."""
    entry = WALEntry(
        operation='add_document',
        doc_id=doc_id,
        payload={'content': content[:10000]},  # Truncate for space
        reason='user_added',
    )
    wal_writer.append(entry)


def log_remove_document(wal_writer: WALWriter, doc_id: str) -> None:
    """Log a document removal to WAL."""
    entry = WALEntry(
        operation='remove_document',
        doc_id=doc_id,
    )
    wal_writer.append(entry)


def log_compute_phase(
    wal_writer: WALWriter,
    phase: str,
    duration_ms: Optional[int] = None
) -> None:
    """Log a compute phase completion to WAL."""
    entry = WALEntry(
        operation='compute_phase',
        phase=phase,
        payload={'duration_ms': duration_ms} if duration_ms else {},
    )
    wal_writer.append(entry)


def log_staleness_change(
    wal_writer: WALWriter,
    mark_fresh: bool,
    computations: List[str]
) -> None:
    """Log staleness changes to WAL."""
    entry = WALEntry(
        operation='mark_fresh' if mark_fresh else 'mark_stale',
        affected_computations=computations,
    )
    wal_writer.append(entry)
