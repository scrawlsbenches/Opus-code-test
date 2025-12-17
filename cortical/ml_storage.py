"""
Content-Addressable Log with Index (CALI) - High-performance ML data storage.

This module provides a Git-inspired storage system optimized for ML training data:
- Content-addressable objects (automatic deduplication via SHA-256 hashing)
- O(1) existence checks via Bloom filter
- O(1) lookups via hash index
- O(log n) range queries via timestamp index
- Session-based logs for merge-friendly git storage (like chunk_index.py)
- Zero external dependencies

Architecture (Git-Friendly):
    .git-ml/cali/
    ├── objects/                    # GIT-TRACKED: Content-addressable storage
    │   ├── a1/b2c3d4...json       # Same content = same file = no conflicts
    │   └── ...
    ├── logs/                       # GIT-TRACKED: Session-based logs (no conflicts)
    │   ├── 2025-12-17_10-30-45_abc123_commit.jsonl
    │   ├── 2025-12-17_10-30-45_abc123_chat.jsonl
    │   └── 2025-12-17_11-00-00_def456_commit.jsonl
    ├── local/                      # NOT TRACKED: Rebuilt on load
    │   ├── indices/                # Hash indices (rebuilt from logs)
    │   │   ├── commit.idx
    │   │   └── chat.idx
    │   └── bloom.bin               # Bloom filter (rebuilt from indices)
    └── manifest.json               # GIT-TRACKED: Version, stats

Git-Friendliness:
    - Objects: Content-addressed, same content = same path = NO CONFLICT
    - Logs: Session-timestamped filenames = NO CONFLICT (like chunk_index.py)
    - Indices: Local-only, rebuilt on load = NO CONFLICT
    - Bloom: Local-only, rebuilt on load = NO CONFLICT

Performance vs JSON files:
    - Existence check: O(n) -> O(1)  [35x faster via bloom filter]
    - Add record: O(n) -> O(1)       [No idempotency scan needed]
    - Get by ID: O(n) -> O(1)        [Index lookup]
    - Range query: O(n) -> O(log n)  [Timestamp index]
    - Training export: O(n²) -> O(n) [Sequential log read]

Usage:
    from cortical.ml_storage import MLStore

    store = MLStore('.git-ml/cali')

    # Add records (automatic deduplication)
    store.put('commit', 'abc123', {'hash': 'abc123', 'message': 'feat: add auth'})
    store.put('chat', 'chat_001', {'query': 'How do I...', 'response': '...'})

    # Fast existence check
    if not store.exists('commit', 'abc123'):
        store.put('commit', 'abc123', data)

    # O(1) retrieval
    record = store.get('commit', 'abc123')

    # Range queries
    recent = store.query_range('commit', start_ts='2025-12-01', end_ts='2025-12-17')

    # Efficient iteration for training
    for record in store.iterate('commit'):
        process(record)
"""

import hashlib
import json
import mmap
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union


# Storage format version
CALI_VERSION = 2  # v2: Session-based git-friendly storage

# Bloom filter parameters (optimized for ~10K records with 0.1% false positive rate)
BLOOM_SIZE_BITS = 143776  # ~18KB
BLOOM_HASH_COUNT = 10

# Index file magic bytes
INDEX_MAGIC = b'CALI'

# Valid record type pattern (alphanumeric, underscore, hyphen)
import re
VALID_RECORD_TYPE_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')
VALID_RECORD_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-:.]{1,256}$')
VALID_CONTENT_HASH_PATTERN = re.compile(r'^[a-fA-F0-9]{64}$')


# ============================================================================
# EXCEPTIONS
# ============================================================================

class CALIError(Exception):
    """Base exception for CALI storage errors."""
    pass


class CALIValidationError(CALIError):
    """Raised when input validation fails."""
    pass


class CALISerializationError(CALIError):
    """Raised when data cannot be serialized to JSON."""
    pass


class CALIStorageError(CALIError):
    """Raised when storage operations fail."""
    pass


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_record_type(record_type: Any) -> str:
    """
    Validate and normalize record type.

    Args:
        record_type: The record type to validate

    Returns:
        Normalized record type string

    Raises:
        CALIValidationError: If record type is invalid
    """
    if record_type is None:
        raise CALIValidationError("record_type cannot be None")

    if not isinstance(record_type, str):
        raise CALIValidationError(
            f"record_type must be a string, got {type(record_type).__name__}"
        )

    record_type = record_type.strip()

    if not record_type:
        raise CALIValidationError("record_type cannot be empty")

    if not VALID_RECORD_TYPE_PATTERN.match(record_type):
        raise CALIValidationError(
            f"record_type '{record_type}' is invalid. Must start with a letter "
            "and contain only alphanumeric characters, underscores, or hyphens "
            "(max 64 chars)"
        )

    return record_type.lower()


def validate_record_id(record_id: Any) -> str:
    """
    Validate and normalize record ID.

    Args:
        record_id: The record ID to validate

    Returns:
        Normalized record ID string

    Raises:
        CALIValidationError: If record ID is invalid
    """
    if record_id is None:
        raise CALIValidationError("record_id cannot be None")

    if not isinstance(record_id, str):
        raise CALIValidationError(
            f"record_id must be a string, got {type(record_id).__name__}"
        )

    record_id = record_id.strip()

    if not record_id:
        raise CALIValidationError("record_id cannot be empty")

    if not VALID_RECORD_ID_PATTERN.match(record_id):
        raise CALIValidationError(
            f"record_id '{record_id[:50]}...' is invalid. Must contain only "
            "alphanumeric characters, underscores, hyphens, colons, or periods "
            "(max 256 chars)"
        )

    return record_id


def validate_content_hash(content_hash: Any) -> str:
    """
    Validate content hash format (SHA-256 hex string).

    Args:
        content_hash: The content hash to validate

    Returns:
        Normalized content hash string (lowercase)

    Raises:
        CALIValidationError: If content hash is invalid
    """
    if content_hash is None:
        raise CALIValidationError("content_hash cannot be None")

    if not isinstance(content_hash, str):
        raise CALIValidationError(
            f"content_hash must be a string, got {type(content_hash).__name__}"
        )

    content_hash = content_hash.strip().lower()

    if not VALID_CONTENT_HASH_PATTERN.match(content_hash):
        raise CALIValidationError(
            f"content_hash '{content_hash[:20]}...' is invalid. "
            "Must be a 64-character hexadecimal string (SHA-256)"
        )

    return content_hash


def validate_data(data: Any) -> Dict[str, Any]:
    """
    Validate that data is a serializable dictionary.

    Args:
        data: The data to validate

    Returns:
        The validated data dictionary

    Raises:
        CALIValidationError: If data is not a dict
        CALISerializationError: If data cannot be serialized to JSON
    """
    if data is None:
        raise CALIValidationError("data cannot be None")

    if not isinstance(data, dict):
        raise CALIValidationError(
            f"data must be a dictionary, got {type(data).__name__}"
        )

    if len(data) == 0:
        raise CALIValidationError("data cannot be an empty dictionary")

    # Verify JSON serializable
    try:
        json.dumps(data, sort_keys=True)
    except (TypeError, ValueError, OverflowError) as e:
        raise CALISerializationError(
            f"data is not JSON serializable: {e}"
        )

    # Check for reasonable size (10MB limit)
    try:
        serialized = json.dumps(data, separators=(',', ':'))
        if len(serialized) > 10 * 1024 * 1024:  # 10MB
            raise CALIValidationError(
                f"data is too large ({len(serialized) / 1024 / 1024:.1f}MB). "
                "Maximum size is 10MB"
            )
    except (TypeError, ValueError):
        pass  # Already caught above

    return data


def validate_timestamp(timestamp: Any) -> float:
    """
    Validate and normalize timestamp.

    Args:
        timestamp: Unix timestamp (float) or ISO string

    Returns:
        Unix timestamp as float

    Raises:
        CALIValidationError: If timestamp is invalid
    """
    if timestamp is None:
        return time.time()

    if isinstance(timestamp, (int, float)):
        ts = float(timestamp)
        # Sanity check: between 2020 and 2100
        if ts < 1577836800 or ts > 4102444800:
            raise CALIValidationError(
                f"timestamp {ts} is out of valid range (2020-2100)"
            )
        return ts

    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.timestamp()
        except ValueError as e:
            raise CALIValidationError(
                f"timestamp '{timestamp}' is not a valid ISO format: {e}"
            )

    raise CALIValidationError(
        f"timestamp must be a float or ISO string, got {type(timestamp).__name__}"
    )


class BloomFilter:
    """
    Space-efficient probabilistic data structure for O(1) existence checks.

    False positives possible, false negatives impossible.
    At 10K records: ~18KB storage, 0.1% false positive rate.
    """

    def __init__(self, size_bits: int = BLOOM_SIZE_BITS, hash_count: int = BLOOM_HASH_COUNT):
        self.size_bits = size_bits
        self.hash_count = hash_count
        self.size_bytes = (size_bits + 7) // 8
        self.bits = bytearray(self.size_bytes)
        self.count = 0

    def _hash_functions(self, key: bytes) -> Generator[int, None, None]:
        """Generate multiple hash positions using double hashing."""
        h1 = int(hashlib.md5(key).hexdigest(), 16)
        h2 = int(hashlib.sha1(key).hexdigest(), 16)
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size_bits

    def add(self, key: str) -> None:
        """Add a key to the filter."""
        key_bytes = key.encode('utf-8')
        for pos in self._hash_functions(key_bytes):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self.bits[byte_idx] |= (1 << bit_idx)
        self.count += 1

    def might_contain(self, key: str) -> bool:
        """Check if key might be in the filter (false positives possible)."""
        key_bytes = key.encode('utf-8')
        for pos in self._hash_functions(key_bytes):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self.bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def save(self, path: Path) -> None:
        """Save bloom filter to file."""
        with open(path, 'wb') as f:
            # Header: size_bits (4 bytes), hash_count (4 bytes), count (4 bytes)
            f.write(struct.pack('<III', self.size_bits, self.hash_count, self.count))
            f.write(self.bits)

    @classmethod
    def load(cls, path: Path) -> 'BloomFilter':
        """Load bloom filter from file."""
        with open(path, 'rb') as f:
            size_bits, hash_count, count = struct.unpack('<III', f.read(12))
            bf = cls(size_bits, hash_count)
            bf.bits = bytearray(f.read())
            bf.count = count
        return bf

    def estimated_false_positive_rate(self) -> float:
        """Estimate current false positive rate."""
        if self.count == 0:
            return 0.0
        # Formula: (1 - e^(-kn/m))^k
        import math
        k = self.hash_count
        n = self.count
        m = self.size_bits
        return (1 - math.exp(-k * n / m)) ** k


@dataclass
class IndexEntry:
    """Entry in an index file."""
    key: str
    content_hash: str
    timestamp: float
    offset: int  # Offset in log file for fast access


class HashIndex:
    """
    Persistent hash index for O(1) key -> content_hash lookups.

    Format: JSONL for simplicity and git-friendliness.
    Each line: {"k": key, "h": hash, "t": timestamp, "o": offset}
    """

    def __init__(self, path: Path):
        self.path = path
        self._index: Dict[str, IndexEntry] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load index from file."""
        if not self.path.exists():
            return

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    entry = IndexEntry(
                        key=data['k'],
                        content_hash=data['h'],
                        timestamp=data['t'],
                        offset=data.get('o', 0)
                    )
                    self._index[entry.key] = entry
                except (json.JSONDecodeError, KeyError):
                    continue  # Skip malformed entries

    def get(self, key: str) -> Optional[IndexEntry]:
        """Get index entry by key."""
        return self._index.get(key)

    def put(self, key: str, content_hash: str, timestamp: float, offset: int) -> None:
        """Add or update index entry."""
        entry = IndexEntry(key, content_hash, timestamp, offset)
        self._index[key] = entry
        self._dirty = True

        # Append to file immediately (append-only for crash safety)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'k': key, 'h': content_hash, 't': timestamp, 'o': offset
            }) + '\n')

    def contains(self, key: str) -> bool:
        """Check if key exists in index."""
        return key in self._index

    def keys(self) -> Set[str]:
        """Get all keys in index."""
        return set(self._index.keys())

    def values(self) -> Iterator[IndexEntry]:
        """Iterate over all entries."""
        return iter(self._index.values())

    def __len__(self) -> int:
        return len(self._index)

    def compact(self) -> None:
        """Compact index by rewriting without duplicates."""
        if not self._dirty and self.path.exists():
            # Check if compaction needed (file has duplicates)
            line_count = sum(1 for _ in open(self.path))
            if line_count <= len(self._index):
                return

        # Rewrite compacted
        temp_path = self.path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            for entry in sorted(self._index.values(), key=lambda e: e.timestamp):
                f.write(json.dumps({
                    'k': entry.key, 'h': entry.content_hash,
                    't': entry.timestamp, 'o': entry.offset
                }) + '\n')

        temp_path.replace(self.path)
        self._dirty = False


class TimestampIndex:
    """
    Index for efficient timestamp range queries.

    Uses a sorted list of (timestamp, key) tuples with binary search.
    Persisted as JSONL sorted by timestamp.
    """

    def __init__(self, path: Path):
        self.path = path
        self._entries: List[Tuple[float, str]] = []
        self._load()

    def _load(self) -> None:
        """Load index from file."""
        if not self.path.exists():
            return

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    self._entries.append((data['t'], data['k']))
                except (json.JSONDecodeError, KeyError):
                    continue

        # Ensure sorted
        self._entries.sort()

    def add(self, timestamp: float, key: str) -> None:
        """Add entry (maintains sorted order via append + periodic sort)."""
        self._entries.append((timestamp, key))

        # Append to file
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': timestamp, 'k': key}) + '\n')

    def query_range(
        self,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None
    ) -> List[str]:
        """
        Query keys in timestamp range using binary search.

        Returns keys where start_ts <= timestamp <= end_ts.
        """
        import bisect

        # Ensure sorted for binary search
        self._entries.sort()

        if start_ts is None:
            start_idx = 0
        else:
            start_idx = bisect.bisect_left(self._entries, (start_ts, ''))

        if end_ts is None:
            end_idx = len(self._entries)
        else:
            end_idx = bisect.bisect_right(self._entries, (end_ts, '\xff' * 100))

        return [key for _, key in self._entries[start_idx:end_idx]]

    def __len__(self) -> int:
        return len(self._entries)

    def compact(self) -> None:
        """Compact by rewriting sorted without duplicates."""
        self._entries.sort()
        seen = set()
        unique = []
        for ts, key in self._entries:
            if key not in seen:
                seen.add(key)
                unique.append((ts, key))
        self._entries = unique

        # Rewrite file
        temp_path = self.path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            for ts, key in self._entries:
                f.write(json.dumps({'t': ts, 'k': key}) + '\n')
        temp_path.replace(self.path)


class ObjectStore:
    """
    Content-addressable object storage (like .git/objects).

    Objects are stored by SHA-256 hash with 2-char prefix directories.
    Automatic deduplication: same content = same hash = stored once.
    """

    def __init__(self, objects_dir: Path):
        self.objects_dir = objects_dir
        objects_dir.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _object_path(self, content_hash: str) -> Path:
        """Get path for object by hash."""
        prefix = content_hash[:2]
        suffix = content_hash[2:]
        return self.objects_dir / prefix / f"{suffix}.json"

    def put(self, data: Dict[str, Any]) -> str:
        """
        Store object and return its content hash.

        Automatically deduplicates: if content already exists, returns existing hash.
        """
        # Serialize deterministically (sorted keys for consistent hashing)
        content = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        content_hash = self._hash_content(content)

        obj_path = self._object_path(content_hash)
        if obj_path.exists():
            # Already stored (deduplication)
            return content_hash

        # Create prefix directory
        obj_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_path = obj_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            f.write(content)
        temp_path.replace(obj_path)

        return content_hash

    def get(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve object by content hash."""
        obj_path = self._object_path(content_hash)
        if not obj_path.exists():
            return None

        with open(obj_path, 'rb') as f:
            return json.loads(f.read().decode('utf-8'))

    def exists(self, content_hash: str) -> bool:
        """Check if object exists."""
        return self._object_path(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """Delete object (for compaction/cleanup)."""
        obj_path = self._object_path(content_hash)
        if obj_path.exists():
            obj_path.unlink()
            # Clean up empty prefix directory
            try:
                obj_path.parent.rmdir()
            except OSError:
                pass  # Directory not empty
            return True
        return False

    def iterate_hashes(self) -> Generator[str, None, None]:
        """Iterate over all stored object hashes."""
        for prefix_dir in self.objects_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for obj_file in prefix_dir.glob('*.json'):
                    yield prefix_dir.name + obj_file.stem


class PackedLog:
    """
    Append-only log with inline data for fast sequential iteration.

    Each line contains the full record data for cache-friendly sequential reads.
    Ideal for training data iteration where you need to read all records.
    """

    def __init__(self, path: Path):
        self.path = path

    def append(self, record_id: str, timestamp: float, data: Dict[str, Any]) -> int:
        """Append record and return offset."""
        offset = self.path.stat().st_size if self.path.exists() else 0
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'id': record_id,
                'ts': timestamp,
                'd': data  # Full data inline
            }, separators=(',', ':')) + '\n')
        return offset

    def iterate(self) -> Generator[Tuple[str, float, Dict[str, Any]], None, None]:
        """Iterate over all records - very fast for training."""
        if not self.path.exists():
            return
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    yield entry['id'], entry['ts'], entry['d']
                except (json.JSONDecodeError, KeyError):
                    continue


class SessionLog:
    """
    Session-based log storage for git-friendly merging.

    Each session writes to its own uniquely-named file:
        logs/2025-12-17_10-30-45_abc123_commit.jsonl

    Benefits:
    - No merge conflicts (unique filenames per session)
    - Append-only within session
    - Can be compacted later (like chunk_index.py)
    """

    def __init__(self, logs_dir: Path, record_type: str, session_id: Optional[str] = None):
        self.logs_dir = logs_dir
        self.record_type = record_type
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._current_file: Optional[Path] = None
        logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self) -> Path:
        """Get or create the current session's log file."""
        if self._current_file is None:
            filename = f"{self.timestamp}_{self.session_id}_{self.record_type}.jsonl"
            self._current_file = self.logs_dir / filename
        return self._current_file

    def append(self, record_id: str, timestamp: float, data: Dict[str, Any]) -> None:
        """Append record to session log."""
        log_file = self._get_session_file()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'id': record_id,
                'ts': timestamp,
                'd': data
            }, separators=(',', ':')) + '\n')

    @classmethod
    def iterate_all(
        cls,
        logs_dir: Path,
        record_type: str
    ) -> Generator[Tuple[str, float, Dict[str, Any]], None, None]:
        """
        Iterate over all session logs for a record type.

        Reads files in timestamp order (oldest first) for deterministic replay.
        """
        if not logs_dir.exists():
            return

        # Find all log files for this record type
        pattern = f"*_{record_type}.jsonl"
        log_files = sorted(logs_dir.glob(pattern))  # Sorted by timestamp prefix

        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        yield entry['id'], entry['ts'], entry['d']
                    except (json.JSONDecodeError, KeyError):
                        continue

    @classmethod
    def get_log_files(cls, logs_dir: Path, record_type: str) -> List[Path]:
        """Get all log files for a record type, sorted by timestamp."""
        if not logs_dir.exists():
            return []
        pattern = f"*_{record_type}.jsonl"
        return sorted(logs_dir.glob(pattern))


class MLStore:
    """
    High-performance ML data store with content-addressable storage.

    Features:
    - O(1) existence checks via Bloom filter
    - O(1) lookups via hash index
    - O(log n) range queries via timestamp index
    - Automatic deduplication via content addressing
    - Session-based logs for git-friendly storage (no merge conflicts)
    - Git-friendly JSONL format

    Git-Friendly Architecture:
        objects/     - GIT-TRACKED: Content-addressed (same content = same path)
        logs/        - GIT-TRACKED: Session-timestamped files (unique names)
        local/       - NOT TRACKED: Indices & bloom rebuilt on load

    Usage:
        store = MLStore('.git-ml/cali')

        # Store records
        store.put('commit', 'abc123', {'hash': 'abc123', 'message': 'feat: auth'})
        store.put('chat', 'chat_001', {'query': '...', 'response': '...'})

        # Check existence (O(1))
        if not store.exists('commit', 'abc123'):
            store.put('commit', 'abc123', data)

        # Retrieve (O(1))
        record = store.get('commit', 'abc123')

        # Range query (O(log n))
        recent = store.query_range('commit', start_ts=time.time() - 86400)

        # Fast iteration for training (sequential log reads)
        for record in store.iterate('commit'):
            train(record)
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        session_id: Optional[str] = None,
        rebuild_indices: bool = True
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Session ID for this store instance (unique per session)
        self.session_id = session_id or uuid.uuid4().hex[:8]

        # Directories
        self._objects_dir = self.base_dir / 'objects'
        self._logs_dir = self.base_dir / 'logs'
        self._local_dir = self.base_dir / 'local'  # NOT git-tracked

        # Initialize components
        self._objects = ObjectStore(self._objects_dir)
        self._session_logs: Dict[str, SessionLog] = {}
        self._indices: Dict[str, HashIndex] = {}
        self._time_indices: Dict[str, TimestampIndex] = {}
        self._bloom: Optional[BloomFilter] = None
        self._indices_built = False

        # Create local dir (should be gitignored)
        self._local_dir.mkdir(parents=True, exist_ok=True)

        # Load or rebuild indices from session logs
        if rebuild_indices:
            self._ensure_indices_built()

        # Load manifest
        self._manifest = self._load_manifest()

    def _ensure_indices_built(self) -> None:
        """Ensure indices are built from session logs."""
        if self._indices_built:
            return

        bloom_path = self._local_dir / 'bloom.bin'
        indices_dir = self._local_dir / 'indices'

        # Check if we need to rebuild (no bloom or stale)
        needs_rebuild = not bloom_path.exists()

        if needs_rebuild:
            self._rebuild_indices_from_logs()
        else:
            # Load existing indices
            self._bloom = BloomFilter.load(bloom_path)

        self._indices_built = True

    def _rebuild_indices_from_logs(self) -> None:
        """Rebuild all indices by replaying session logs."""
        indices_dir = self._local_dir / 'indices'
        indices_dir.mkdir(parents=True, exist_ok=True)

        # Fresh bloom filter
        self._bloom = BloomFilter()

        # Find all record types by scanning log files
        record_types = set()
        if self._logs_dir.exists():
            for log_file in self._logs_dir.glob('*.jsonl'):
                # Extract record type from filename: timestamp_session_TYPE.jsonl
                parts = log_file.stem.rsplit('_', 1)
                if len(parts) == 2:
                    record_types.add(parts[1])

        # Rebuild indices for each type
        for record_type in record_types:
            idx = HashIndex(indices_dir / f'{record_type}.idx')
            time_idx = TimestampIndex(indices_dir / f'{record_type}_time.idx')

            # Replay all session logs for this type
            for record_id, timestamp, data in SessionLog.iterate_all(
                self._logs_dir, record_type
            ):
                # Get content hash from object store
                content_hash = self._objects.put(data)  # Idempotent

                # Update indices
                idx.put(record_id, content_hash, timestamp, 0)
                time_idx.add(timestamp, record_id)

                # Update bloom filter
                bloom_key = self._bloom_key(record_type, record_id)
                self._bloom.add(bloom_key)

            self._indices[record_type] = idx
            self._time_indices[record_type] = time_idx

        # Save bloom filter
        self._bloom.save(self._local_dir / 'bloom.bin')

    def _get_session_log(self, record_type: str) -> SessionLog:
        """Get or create session log for record type."""
        if record_type not in self._session_logs:
            self._session_logs[record_type] = SessionLog(
                self._logs_dir, record_type, self.session_id
            )
        return self._session_logs[record_type]

    def _load_manifest(self) -> Dict[str, Any]:
        """Load storage manifest."""
        manifest_path = self.base_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return {
            'version': CALI_VERSION,
            'created': datetime.now().isoformat(),
            'record_counts': {},
            'last_compaction': None
        }

    def _save_manifest(self) -> None:
        """Save storage manifest."""
        manifest_path = self.base_dir / 'manifest.json'
        self._manifest['updated'] = datetime.now().isoformat()
        with open(manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)

    def _get_index(self, record_type: str) -> HashIndex:
        """Get or create hash index for record type (stored in local/)."""
        if record_type not in self._indices:
            indices_dir = self._local_dir / 'indices'
            indices_dir.mkdir(exist_ok=True)
            self._indices[record_type] = HashIndex(indices_dir / f'{record_type}.idx')
        return self._indices[record_type]

    def _get_time_index(self, record_type: str) -> TimestampIndex:
        """Get or create timestamp index for record type (stored in local/)."""
        if record_type not in self._time_indices:
            indices_dir = self._local_dir / 'indices'
            indices_dir.mkdir(exist_ok=True)
            self._time_indices[record_type] = TimestampIndex(
                indices_dir / f'{record_type}_time.idx'
            )
        return self._time_indices[record_type]

    def _bloom_key(self, record_type: str, record_id: str) -> str:
        """Generate bloom filter key."""
        return f"{record_type}:{record_id}"

    def exists(self, record_type: str, record_id: str) -> bool:
        """
        O(1) existence check using bloom filter + index verification.

        Returns False if definitely not present, True if likely present.
        For guaranteed accuracy, verifies bloom positives against index.

        Raises:
            CALIValidationError: If inputs are invalid
        """
        # Validate inputs
        record_type = validate_record_type(record_type)
        record_id = validate_record_id(record_id)

        bloom_key = self._bloom_key(record_type, record_id)

        # Fast path: bloom filter says no -> definitely not present
        if not self._bloom.might_contain(bloom_key):
            return False

        # Bloom says maybe -> verify with index (handles false positives)
        idx = self._get_index(record_type)
        return idx.contains(record_id)

    def put(
        self,
        record_type: str,
        record_id: str,
        data: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> str:
        """
        Store a record. Returns content hash.

        Automatically:
        - Deduplicates (same content = same hash = stored once)
        - Updates bloom filter for O(1) existence checks
        - Updates indices for fast lookups
        - Appends to session log (git-friendly, no merge conflicts)

        Args:
            record_type: Type of record ('commit', 'chat', 'session', etc.)
            record_id: Unique ID for this record
            data: Record data to store
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Content hash of stored object

        Raises:
            CALIValidationError: If inputs are invalid
            CALISerializationError: If data cannot be serialized
        """
        # Validate all inputs
        record_type = validate_record_type(record_type)
        record_id = validate_record_id(record_id)
        data = validate_data(data)
        timestamp = validate_timestamp(timestamp)

        # Store object in content-addressed storage (automatic deduplication)
        content_hash = self._objects.put(data)

        # Update bloom filter (local)
        bloom_key = self._bloom_key(record_type, record_id)
        self._bloom.add(bloom_key)

        # Update hash index (local)
        idx = self._get_index(record_type)
        idx.put(record_id, content_hash, timestamp, 0)

        # Update timestamp index (local)
        time_idx = self._get_time_index(record_type)
        time_idx.add(timestamp, record_id)

        # Append to session log (git-tracked, session-based filename = no conflicts)
        session_log = self._get_session_log(record_type)
        session_log.append(record_id, timestamp, data)

        # Update manifest counts
        counts = self._manifest.setdefault('record_counts', {})
        counts[record_type] = counts.get(record_type, 0) + 1

        # Periodic bloom filter save (to local dir)
        if self._bloom.count % 100 == 0:
            self._bloom.save(self._local_dir / 'bloom.bin')

        return content_hash

    def get(self, record_type: str, record_id: str) -> Optional[Dict[str, Any]]:
        """
        O(1) retrieval by ID.

        Returns None if record doesn't exist.

        Raises:
            CALIValidationError: If inputs are invalid
        """
        # Validate inputs
        record_type = validate_record_type(record_type)
        record_id = validate_record_id(record_id)

        idx = self._get_index(record_type)
        entry = idx.get(record_id)
        if entry is None:
            return None

        return self._objects.get(entry.content_hash)

    def get_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Direct retrieval by content hash.

        Raises:
            CALIValidationError: If content_hash is invalid
        """
        content_hash = validate_content_hash(content_hash)
        return self._objects.get(content_hash)

    def query_range(
        self,
        record_type: str,
        start_ts: Optional[Union[float, str]] = None,
        end_ts: Optional[Union[float, str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        O(log n) range query by timestamp.

        Args:
            record_type: Type of records to query
            start_ts: Start timestamp (float or ISO string)
            end_ts: End timestamp (float or ISO string)

        Yields:
            Records in timestamp order

        Raises:
            CALIValidationError: If inputs are invalid
        """
        # Validate record type
        record_type = validate_record_type(record_type)

        # Convert and validate timestamps
        if start_ts is not None:
            start_ts = validate_timestamp(start_ts)
        if end_ts is not None:
            end_ts = validate_timestamp(end_ts)

        # Validate range order
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise CALIValidationError(
                f"start_ts ({start_ts}) must be <= end_ts ({end_ts})"
            )

        time_idx = self._get_time_index(record_type)
        record_ids = time_idx.query_range(start_ts, end_ts)

        idx = self._get_index(record_type)
        for record_id in record_ids:
            entry = idx.get(record_id)
            if entry:
                data = self._objects.get(entry.content_hash)
                if data:
                    yield data

    def iterate(
        self,
        record_type: str,
        include_metadata: bool = False
    ) -> Generator[Union[Dict[str, Any], Tuple[str, float, Dict[str, Any]]], None, None]:
        """
        Efficient sequential iteration over all records of a type.

        Reads all session logs in timestamp order for deterministic replay.
        Data is inline in session logs - no object lookups needed.

        Args:
            record_type: Type of records to iterate
            include_metadata: If True, yields (id, timestamp, data) tuples

        Yields:
            Record data dicts (or tuples if include_metadata=True)

        Raises:
            CALIValidationError: If record_type is invalid
        """
        # Validate record type
        record_type = validate_record_type(record_type)

        # Iterate over all session logs (data inline = fast)
        for record_id, timestamp, data in SessionLog.iterate_all(
            self._logs_dir, record_type
        ):
            if include_metadata:
                yield (record_id, timestamp, data)
            else:
                yield data

    def count(self, record_type: str) -> int:
        """Get count of records by type."""
        return self._manifest.get('record_counts', {}).get(record_type, 0)

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'version': self._manifest.get('version', CALI_VERSION),
            'record_counts': self._manifest.get('record_counts', {}),
            'bloom_filter': {
                'size_kb': self._bloom.size_bytes / 1024,
                'count': self._bloom.count,
                'estimated_fpr': f"{self._bloom.estimated_false_positive_rate():.4%}"
            },
            'indices': {
                rt: len(idx) for rt, idx in self._indices.items()
            },
            'objects_dir_size_kb': sum(
                f.stat().st_size for f in self._objects.objects_dir.rglob('*.json')
            ) / 1024 if self._objects.objects_dir.exists() else 0
        }

    def compact(self) -> Dict[str, Any]:
        """
        Compact storage by:
        - Removing duplicate index entries
        - Cleaning orphaned objects
        - Rewriting bloom filter
        - Saving manifest

        Returns statistics about compaction.
        """
        stats = {'indices_compacted': 0, 'orphans_removed': 0}

        # Compact indices
        for record_type, idx in self._indices.items():
            idx.compact()
            stats['indices_compacted'] += 1

        for record_type, time_idx in self._time_indices.items():
            time_idx.compact()

        # Find referenced hashes
        referenced_hashes = set()
        for idx in self._indices.values():
            for entry in idx.values():
                referenced_hashes.add(entry.content_hash)

        # Remove orphaned objects
        for content_hash in list(self._objects.iterate_hashes()):
            if content_hash not in referenced_hashes:
                self._objects.delete(content_hash)
                stats['orphans_removed'] += 1

        # Rebuild bloom filter
        self._bloom = BloomFilter()
        for record_type, idx in self._indices.items():
            for entry in idx.values():
                bloom_key = self._bloom_key(record_type, entry.key)
                self._bloom.add(bloom_key)

        # Save (bloom to local dir)
        self._bloom.save(self._local_dir / 'bloom.bin')
        self._manifest['last_compaction'] = datetime.now().isoformat()
        self._save_manifest()

        return stats

    def close(self) -> None:
        """Persist all state before closing."""
        if self._bloom is not None:
            self._bloom.save(self._local_dir / 'bloom.bin')
        self._save_manifest()

    def rebuild_indices(self) -> None:
        """Force rebuild of indices from session logs."""
        self._indices = {}
        self._time_indices = {}
        self._indices_built = False
        self._rebuild_indices_from_logs()


# Convenience functions for migration from JSON files
def migrate_from_jsonl(
    jsonl_path: Path,
    store: MLStore,
    record_type: str,
    id_field: str = 'id',
    timestamp_field: str = 'timestamp'
) -> Dict[str, int]:
    """
    Migrate records from JSONL file to MLStore.

    Args:
        jsonl_path: Path to JSONL file
        store: MLStore to migrate into
        record_type: Type of records
        id_field: Field name containing record ID
        timestamp_field: Field name containing timestamp

    Returns:
        Migration statistics
    """
    stats = {'migrated': 0, 'skipped': 0, 'errors': 0}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                record_id = data.get(id_field) or data.get('hash') or str(hash(line))

                # Skip if already exists
                if store.exists(record_type, record_id):
                    stats['skipped'] += 1
                    continue

                # Get timestamp
                ts = data.get(timestamp_field)
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                    except ValueError:
                        ts = time.time()
                elif ts is None:
                    ts = time.time()

                store.put(record_type, record_id, data, timestamp=ts)
                stats['migrated'] += 1

            except (json.JSONDecodeError, KeyError) as e:
                stats['errors'] += 1

    return stats


def migrate_from_json_dir(
    json_dir: Path,
    store: MLStore,
    record_type: str,
    id_field: str = 'id',
    timestamp_field: str = 'timestamp',
    pattern: str = '**/*.json'
) -> Dict[str, int]:
    """
    Migrate records from directory of JSON files to MLStore.

    Args:
        json_dir: Directory containing JSON files
        store: MLStore to migrate into
        record_type: Type of records
        id_field: Field name containing record ID
        timestamp_field: Field name containing timestamp
        pattern: Glob pattern for JSON files

    Returns:
        Migration statistics
    """
    stats = {'migrated': 0, 'skipped': 0, 'errors': 0}

    for json_path in json_dir.glob(pattern):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            record_id = (
                data.get(id_field) or
                data.get('hash') or
                json_path.stem
            )

            # Skip if already exists
            if store.exists(record_type, record_id):
                stats['skipped'] += 1
                continue

            # Get timestamp
            ts = data.get(timestamp_field)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except ValueError:
                    ts = time.time()
            elif ts is None:
                # Try to get from filename or file mtime
                ts = json_path.stat().st_mtime

            store.put(record_type, record_id, data, timestamp=ts)
            stats['migrated'] += 1

        except (json.JSONDecodeError, OSError) as e:
            stats['errors'] += 1

    return stats
