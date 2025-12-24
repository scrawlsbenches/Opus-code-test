"""
Write-Ahead Log for GoT transactional system.

Provides crash recovery by logging all operations BEFORE they are applied.
Uses JSONL format with checksums and fsync for durability.

This module uses the shared TransactionWALEntry from cortical.wal for entry
representation, ensuring consistent checksum computation and serialization
across all WAL implementations in the system.

Logging:
    This module uses Python's standard logging. Configure via:

        import logging
        logging.getLogger('cortical.got.wal').setLevel(logging.DEBUG)

    Log levels:
    - DEBUG: Race conditions, sequence file recovery
    - WARNING: Corrupted WAL entries skipped during replay
    - ERROR: WAL operation failures

See also:
    - cortical.wal: Base WAL infrastructure (BaseWALEntry, TransactionWALEntry)
    - cortical.utils.checksums: Shared checksum utilities
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path

# Module-level logger - configure via logging.getLogger('cortical.got.wal')
logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from cortical.wal import TransactionWALEntry
from cortical.utils.checksums import compute_checksum
from .errors import CorruptionError
from .config import DurabilityMode


# Re-export for convenience
__all__ = ['WALManager', 'TransactionWALEntry']


def _is_legacy_entry(data: Dict[str, Any]) -> bool:
    """
    Detect legacy WAL entry format from orphan_recovery migration.

    Legacy format: {"op": "ADOPTED", "entity_id": "...", "timestamp": float, "checksum": "..."}
    Current format: {"seq": N, "ts": "ISO-8601", "tx": "TX-xxx", "op": "...", "data": {...}}

    Legacy entries have already been replayed into entities and can be safely skipped
    during WAL replay. They remain in the WAL for historical audit purposes.
    """
    # Legacy entries have entity_id (not in new format)
    if 'entity_id' in data:
        return True
    # Legacy entries have numeric timestamp (new format uses 'ts' as ISO string)
    if 'timestamp' in data and isinstance(data.get('timestamp'), (int, float)):
        return True
    # Legacy entries lack the 'seq' field
    if 'seq' not in data and 'tx' not in data:
        return True
    return False


class WALManager:
    """
    Write-Ahead Log for crash recovery.

    All operations are logged BEFORE they are applied.
    On crash, incomplete transactions can be rolled back.
    Uses JSONL format (one JSON object per line).

    WAL layout:
        {wal_dir}/
            current.wal           # Active WAL file (JSONL)
            _sequence.json        # Sequence counter
            archived/             # Archived WAL files
                TIMESTAMP.wal
    """

    def __init__(self, wal_dir: Path, durability: DurabilityMode = DurabilityMode.BALANCED):
        """
        Initialize WAL, creating directory if needed.

        Args:
            wal_dir: Directory path for WAL files
            durability: Durability mode controlling fsync behavior
        """
        self.wal_dir = Path(wal_dir)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.durability = durability

        # Create archived directory
        self.archive_dir = self.wal_dir / "archived"
        self.archive_dir.mkdir(exist_ok=True)

        # WAL file paths
        self.wal_file = self.wal_dir / "current.wal"
        self.seq_file = self.wal_dir / "_sequence.json"

        # Load or initialize sequence counter
        self._sequence = self._load_sequence()

    def _load_sequence(self) -> int:
        """Load sequence counter from disk."""
        try:
            if self.seq_file.exists():
                with open(self.seq_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('seq', 0)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # File was deleted or corrupted between exists() and read
            # This is fine - start from 0
            logger.debug(
                "Sequence file unavailable, starting from 0: %s: %s",
                type(e).__name__, e
            )
        return 0

    def _save_sequence(self) -> None:
        """Persist sequence counter to disk."""
        with open(self.seq_file, 'w', encoding='utf-8') as f:
            json.dump({'seq': self._sequence}, f)
            f.flush()
            # Only fsync if PARANOID mode
            if self.durability == DurabilityMode.PARANOID:
                os.fsync(f.fileno())

    def _next_seq(self) -> int:
        """Get next sequence number and persist it."""
        self._sequence += 1
        self._save_sequence()
        return self._sequence

    def log(self, tx_id: str, operation: str, data: Dict[str, Any]) -> int:
        """
        Append entry to WAL with fsync.

        Args:
            tx_id: Transaction ID
            operation: Operation type (TX_BEGIN, WRITE, TX_COMMIT, etc.)
            data: Operation-specific data

        Returns:
            Sequence number of the entry
        """
        seq = self._next_seq()

        # Create entry using shared TransactionWALEntry
        entry = TransactionWALEntry(
            seq=seq,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tx_id=tx_id,
            operation=operation,
            payload=data,
        )

        # Append to WAL file using entry's serialization
        with open(self.wal_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict(), separators=(',', ':')) + '\n')
            f.flush()
            # Only fsync if PARANOID mode
            if self.durability == DurabilityMode.PARANOID:
                os.fsync(f.fileno())

        return seq

    def log_tx_begin(self, tx_id: str, snapshot_version: int) -> int:
        """
        Log transaction start with snapshot version.

        Args:
            tx_id: Transaction ID
            snapshot_version: Snapshot version at transaction start

        Returns:
            Sequence number of the entry
        """
        return self.log(tx_id, 'TX_BEGIN', {'snapshot': snapshot_version})

    def log_write(self, tx_id: str, entity_id: str, old_version: int, new_version: int) -> int:
        """
        Log a write operation (entity_id, old version → new version).

        Args:
            tx_id: Transaction ID
            entity_id: Entity being written
            old_version: Version before write
            new_version: Version after write

        Returns:
            Sequence number of the entry
        """
        return self.log(tx_id, 'WRITE', {
            'entity_id': entity_id,
            'old_version': old_version,
            'new_version': new_version
        })

    def log_tx_prepare(self, tx_id: str) -> int:
        """
        Log that transaction is entering prepare phase.

        Args:
            tx_id: Transaction ID

        Returns:
            Sequence number of the entry
        """
        return self.log(tx_id, 'TX_PREPARE', {})

    def log_tx_commit(self, tx_id: str, version: int) -> int:
        """
        Log successful transaction commit with final version.

        Args:
            tx_id: Transaction ID
            version: Global version after commit

        Returns:
            Sequence number of the entry
        """
        return self.log(tx_id, 'TX_COMMIT', {'version': version})

    def log_tx_abort(self, tx_id: str, reason: str) -> int:
        """
        Log transaction abort with reason.

        Args:
            tx_id: Transaction ID
            reason: Reason for abort

        Returns:
            Sequence number of the entry
        """
        return self.log(tx_id, 'TX_ABORT', {'reason': reason})

    def log_tx_rollback(self, tx_id: str, reason: str) -> int:
        """
        Log transaction rollback with reason.

        Args:
            tx_id: Transaction ID
            reason: Reason for rollback

        Returns:
            Sequence number of the entry
        """
        return self.log(tx_id, 'TX_ROLLBACK', {'reason': reason})

    def replay(self) -> List[Dict[str, Any]]:
        """
        Read all WAL entries in order.

        Skips entries with corrupted checksums.

        Returns:
            List of valid entries in sequence order (as dictionaries for
            backward compatibility; use replay_entries() for typed entries)
        """
        if not self.wal_file.exists():
            return []

        entries = []
        legacy_count = 0
        invalid_checksum_count = 0

        with open(self.wal_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip corrupted JSON
                    logger.warning(
                        "Skipping corrupted WAL entry at line %d: %s",
                        line_num, e
                    )
                    continue

                # Skip legacy format entries (already replayed into entities)
                if _is_legacy_entry(data):
                    legacy_count += 1
                    continue

                # Parse into TransactionWALEntry for verification
                entry = TransactionWALEntry.from_dict(data)

                if not entry.verify():
                    # Count but don't log per-line to avoid spam
                    invalid_checksum_count += 1
                    continue

                # Return as dictionary for backward compatibility
                entries.append(entry.to_dict())

        # Consolidated warnings
        if legacy_count > 0:
            logger.debug(
                "Skipped %d legacy WAL entries (already applied to entities)",
                legacy_count
            )
        if invalid_checksum_count > 0:
            logger.warning(
                "Skipped %d WAL entries with invalid checksums",
                invalid_checksum_count
            )

        return entries

    def replay_entries(self) -> List[TransactionWALEntry]:
        """
        Read all WAL entries as typed TransactionWALEntry objects.

        Skips entries with corrupted checksums.

        Returns:
            List of valid TransactionWALEntry objects in sequence order
        """
        if not self.wal_file.exists():
            return []

        entries = []
        legacy_count = 0
        invalid_checksum_count = 0

        with open(self.wal_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip corrupted JSON
                    logger.warning(
                        "Skipping corrupted WAL entry at line %d: %s",
                        line_num, e
                    )
                    continue

                # Skip legacy format entries (already replayed into entities)
                if _is_legacy_entry(data):
                    legacy_count += 1
                    continue

                # Parse into TransactionWALEntry for verification
                entry = TransactionWALEntry.from_dict(data)

                if not entry.verify():
                    # Count but don't log per-line to avoid spam
                    invalid_checksum_count += 1
                    continue

                entries.append(entry)

        # Consolidated warnings
        if legacy_count > 0:
            logger.debug(
                "Skipped %d legacy WAL entries (already applied to entities)",
                legacy_count
            )
        if invalid_checksum_count > 0:
            logger.warning(
                "Skipped %d WAL entries with invalid checksums",
                invalid_checksum_count
            )

        return entries

    def get_incomplete_transactions(self) -> List[Dict[str, Any]]:
        """
        Find transactions that started but didn't commit/abort.

        Returns:
            List of transaction records with:
            - tx_id: Transaction ID
            - state: Last known state ("ACTIVE", "PREPARING")
            - snapshot: Snapshot version at TX_BEGIN
        """
        entries = self.replay()

        # Track transaction states
        transactions = {}

        for entry in entries:
            tx_id = entry['tx']
            op = entry['op']

            if op == 'TX_BEGIN':
                transactions[tx_id] = {
                    'tx_id': tx_id,
                    'state': 'ACTIVE',
                    'snapshot': entry['data'].get('snapshot', 0)
                }
            elif op == 'TX_PREPARE':
                if tx_id in transactions:
                    transactions[tx_id]['state'] = 'PREPARING'
            elif op in ('TX_COMMIT', 'TX_ABORT', 'TX_ROLLBACK'):
                # Transaction completed, remove from tracking
                transactions.pop(tx_id, None)

        # Return transactions still in ACTIVE or PREPARING state
        return list(transactions.values())

    def fsync_now(self) -> None:
        """
        Force fsync of WAL file and sequence file.

        Used by BALANCED mode to sync on transaction commit.
        """
        # Fsync WAL file if it exists
        if self.wal_file.exists():
            with open(self.wal_file, 'r+', encoding='utf-8') as f:
                os.fsync(f.fileno())

        # Fsync sequence file
        if self.seq_file.exists():
            with open(self.seq_file, 'r+', encoding='utf-8') as f:
                os.fsync(f.fileno())

    def truncate(self, archive: bool = True) -> Optional[Path]:
        """
        Truncate WAL after successful checkpoint.

        Args:
            archive: If True, move current.wal to archived/TIMESTAMP.wal
                    If False, delete current.wal

        Returns:
            Path to archived file (or None if deleted)
        """
        if not self.wal_file.exists():
            return None

        if archive:
            # Generate timestamp-based archive name
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            archive_path = self.archive_dir / f"{timestamp}.wal"

            # Move current WAL to archive
            self.wal_file.rename(archive_path)
            return archive_path
        else:
            # Delete current WAL
            self.wal_file.unlink()
            return None
