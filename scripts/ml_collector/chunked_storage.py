"""
Chunked storage for large ML data files.

Provides git-friendly storage for chat transcripts and commit data by:
1. Compressing large content with zlib
2. Splitting into manageable chunks
3. Storing as JSONL for easy git diffs
4. Supporting reconstruction from chunks

Usage:
    # Store a large chat
    store_chunked_chat(chat_data)

    # Reconstruct from chunks
    chat_data = reconstruct_chat(chat_id)

    # Compact old chunks
    compact_chunks()
"""

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ML_DATA_DIR, TRACKED_DIR


# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory for chunked data (git-tracked)
CHUNKED_DIR = TRACKED_DIR / "chunked"

# Size thresholds
COMPRESSION_THRESHOLD = 1024 * 5      # 5KB - compress content larger than this
CHUNK_SIZE_LIMIT = 1024 * 50          # 50KB - max size per chunk file
WARN_TOTAL_SIZE = 1024 * 1024 * 10    # 10MB - warn if total chunked data exceeds this

# Chunk file pattern: chats-YYYYMMDD-HHMMSS-SESSION.jsonl
CHUNK_FILE_PATTERN = "{type}-{timestamp}-{session}.jsonl"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChunkRecord:
    """A single record in a chunk file."""
    record_type: str          # 'chat', 'commit', 'session'
    record_id: str            # Unique ID of the record
    timestamp: str            # ISO timestamp
    sequence: int             # Sequence number (0 for single-part, 1+ for multi-part)
    total_parts: int          # Total parts (1 for single-part)
    compressed: bool          # Whether content is compressed
    content_hash: str         # SHA256 of original content (for deduplication)
    data: Dict[str, Any]      # The actual record data (or chunk of it)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ChunkRecord':
        return cls(**d)


# ============================================================================
# COMPRESSION UTILITIES
# ============================================================================

def compress_content(content: str) -> Tuple[str, bool]:
    """
    Compress content if it exceeds threshold.

    Returns:
        Tuple of (content_or_compressed, was_compressed)
    """
    if len(content) < COMPRESSION_THRESHOLD:
        return content, False

    # Compress with zlib
    compressed = zlib.compress(content.encode('utf-8'), level=6)

    # Only use compression if it actually reduces size
    if len(compressed) >= len(content.encode('utf-8')):
        return content, False

    # Base64 encode for JSON storage
    encoded = base64.b64encode(compressed).decode('ascii')
    return encoded, True


def decompress_content(content: str, compressed: bool) -> str:
    """Decompress content if it was compressed."""
    if not compressed:
        return content

    # Decode base64 and decompress
    decoded = base64.b64decode(content.encode('ascii'))
    decompressed = zlib.decompress(decoded)
    return decompressed.decode('utf-8')


def content_hash(content: str) -> str:
    """Generate SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


# ============================================================================
# CHUNKED STORAGE
# ============================================================================

def ensure_chunked_dir():
    """Ensure chunked storage directory exists."""
    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)


def get_chunk_filename(record_type: str, session_id: str) -> str:
    """Generate chunk filename for current session."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return CHUNK_FILE_PATTERN.format(
        type=record_type,
        timestamp=timestamp,
        session=session_id[:8]
    )


def store_chunked_record(
    record_type: str,
    record_id: str,
    data: Dict[str, Any],
    session_id: str,
    large_fields: List[str] = None
) -> str:
    """
    Store a record with chunking support for large fields.

    Args:
        record_type: Type of record ('chat', 'commit', 'session')
        record_id: Unique ID for the record
        data: The record data
        session_id: Current session ID
        large_fields: Fields that may be large and should be compressed
                     (default: ['response', 'query', 'hunks'])

    Returns:
        Path to the chunk file used
    """
    ensure_chunked_dir()

    if large_fields is None:
        large_fields = ['response', 'query', 'hunks', 'content']

    timestamp = datetime.now().isoformat()

    # Process large fields
    processed_data = data.copy()
    any_compressed = False

    for field in large_fields:
        if field in processed_data and isinstance(processed_data[field], str):
            content = processed_data[field]
            compressed_content, was_compressed = compress_content(content)
            if was_compressed:
                processed_data[field] = compressed_content
                processed_data[f'_{field}_compressed'] = True
                any_compressed = True

    # Handle list fields (like hunks)
    for field in large_fields:
        if field in processed_data and isinstance(processed_data[field], list):
            # Serialize list to JSON string for compression
            list_content = json.dumps(processed_data[field])
            if len(list_content) > COMPRESSION_THRESHOLD:
                compressed_content, was_compressed = compress_content(list_content)
                if was_compressed:
                    processed_data[field] = compressed_content
                    processed_data[f'_{field}_compressed'] = True
                    processed_data[f'_{field}_was_list'] = True
                    any_compressed = True

    # Create chunk record
    record = ChunkRecord(
        record_type=record_type,
        record_id=record_id,
        timestamp=timestamp,
        sequence=0,
        total_parts=1,
        compressed=any_compressed,
        content_hash=content_hash(json.dumps(data)),
        data=processed_data
    )

    # Find or create chunk file for this session
    chunk_file = CHUNKED_DIR / f"{record_type}s-{session_id[:8]}.jsonl"

    # Append to chunk file
    with open(chunk_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record.to_dict()) + '\n')

    return str(chunk_file)


def store_chunked_chat(chat_data: Dict[str, Any], session_id: str) -> str:
    """Store a chat record with compression for large responses."""
    return store_chunked_record(
        record_type='chat',
        record_id=chat_data.get('id', 'unknown'),
        data=chat_data,
        session_id=session_id,
        large_fields=['response', 'query']
    )


def store_chunked_commit(commit_data: Dict[str, Any], session_id: str) -> str:
    """Store a commit record with compression for large diffs."""
    return store_chunked_record(
        record_type='commit',
        record_id=commit_data.get('hash', 'unknown'),
        data=commit_data,
        session_id=session_id,
        large_fields=['hunks', 'message']
    )


# ============================================================================
# RECONSTRUCTION
# ============================================================================

def load_all_chunks() -> Dict[str, List[ChunkRecord]]:
    """
    Load all chunk files and organize by record ID.

    Returns:
        Dict mapping record_id -> list of ChunkRecords
    """
    if not CHUNKED_DIR.exists():
        return {}

    records_by_id: Dict[str, List[ChunkRecord]] = {}

    for chunk_file in CHUNKED_DIR.glob("*.jsonl"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = ChunkRecord.from_dict(json.loads(line))
                    if record.record_id not in records_by_id:
                        records_by_id[record.record_id] = []
                    records_by_id[record.record_id].append(record)
                except (json.JSONDecodeError, KeyError):
                    continue

    return records_by_id


def reconstruct_record(record_id: str) -> Optional[Dict[str, Any]]:
    """
    Reconstruct a record from its chunks.

    Args:
        record_id: The ID of the record to reconstruct

    Returns:
        The reconstructed record data, or None if not found
    """
    records_by_id = load_all_chunks()

    if record_id not in records_by_id:
        return None

    chunks = sorted(records_by_id[record_id], key=lambda r: r.sequence)

    if len(chunks) == 1 and chunks[0].total_parts == 1:
        # Single-part record
        return decompress_record(chunks[0].data)

    # Multi-part record - combine chunks
    combined_data = {}
    for chunk in chunks:
        chunk_data = decompress_record(chunk.data)
        combined_data.update(chunk_data)

    return combined_data


def decompress_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """Decompress any compressed fields in a record."""
    result = {}

    for key, value in data.items():
        # Skip metadata fields
        if key.startswith('_') and key.endswith('_compressed'):
            continue
        if key.startswith('_') and key.endswith('_was_list'):
            continue

        # Check if this field was compressed
        compressed_key = f'_{key}_compressed'
        was_list_key = f'_{key}_was_list'

        if data.get(compressed_key):
            decompressed = decompress_content(value, True)
            # Check if it was originally a list
            if data.get(was_list_key):
                result[key] = json.loads(decompressed)
            else:
                result[key] = decompressed
        else:
            result[key] = value

    return result


def reconstruct_all(record_type: str = None) -> List[Dict[str, Any]]:
    """
    Reconstruct all records, optionally filtered by type.

    Args:
        record_type: Optional filter ('chat', 'commit', 'session')

    Returns:
        List of reconstructed records
    """
    records_by_id = load_all_chunks()
    results = []

    for record_id, chunks in records_by_id.items():
        if record_type and chunks[0].record_type != record_type:
            continue

        reconstructed = reconstruct_record(record_id)
        if reconstructed:
            results.append(reconstructed)

    return results


# ============================================================================
# COMPACTION
# ============================================================================

def compact_chunks(keep_days: int = 30) -> Dict[str, int]:
    """
    Compact old chunk files into consolidated files.

    Args:
        keep_days: Keep separate chunks for this many days

    Returns:
        Stats about compaction
    """
    from datetime import timedelta

    if not CHUNKED_DIR.exists():
        return {'files_before': 0, 'files_after': 0, 'bytes_saved': 0}

    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff.strftime("%Y%m%d")

    old_files = []
    recent_files = []

    for chunk_file in CHUNKED_DIR.glob("*.jsonl"):
        # Extract date from filename (format: type-YYYYMMDD-HHMMSS-session.jsonl)
        try:
            parts = chunk_file.stem.split('-')
            file_date = parts[1]  # YYYYMMDD
            if file_date < cutoff_str:
                old_files.append(chunk_file)
            else:
                recent_files.append(chunk_file)
        except (IndexError, ValueError):
            recent_files.append(chunk_file)

    if not old_files:
        return {
            'files_before': len(recent_files),
            'files_after': len(recent_files),
            'bytes_saved': 0
        }

    # Consolidate old files by type
    bytes_before = sum(f.stat().st_size for f in old_files)
    consolidated: Dict[str, List[str]] = {}

    for old_file in old_files:
        record_type = old_file.stem.split('-')[0]
        if record_type not in consolidated:
            consolidated[record_type] = []

        with open(old_file, 'r', encoding='utf-8') as f:
            consolidated[record_type].extend(f.readlines())

    # Write consolidated files
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_files = []

    for record_type, lines in consolidated.items():
        if not lines:
            continue

        # Deduplicate by content hash
        seen_hashes = set()
        unique_lines = []
        for line in lines:
            try:
                record = json.loads(line)
                hash_val = record.get('content_hash', '')
                if hash_val not in seen_hashes:
                    seen_hashes.add(hash_val)
                    unique_lines.append(line)
            except json.JSONDecodeError:
                unique_lines.append(line)

        consolidated_file = CHUNKED_DIR / f"{record_type}-consolidated-{timestamp}.jsonl"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        new_files.append(consolidated_file)

    # Remove old files
    for old_file in old_files:
        old_file.unlink()

    bytes_after = sum(f.stat().st_size for f in new_files) if new_files else 0

    return {
        'files_before': len(old_files) + len(recent_files),
        'files_after': len(new_files) + len(recent_files),
        'bytes_saved': bytes_before - bytes_after,
        'records_deduplicated': sum(len(lines) for lines in consolidated.values()) - sum(1 for f in new_files for _ in open(f))
    }


# ============================================================================
# UTILITIES
# ============================================================================

def get_chunked_stats() -> Dict[str, Any]:
    """Get statistics about chunked storage."""
    if not CHUNKED_DIR.exists():
        return {
            'total_files': 0,
            'total_bytes': 0,
            'total_records': 0,
            'by_type': {}
        }

    stats = {
        'total_files': 0,
        'total_bytes': 0,
        'total_records': 0,
        'by_type': {}
    }

    for chunk_file in CHUNKED_DIR.glob("*.jsonl"):
        stats['total_files'] += 1
        stats['total_bytes'] += chunk_file.stat().st_size

        record_type = chunk_file.stem.split('-')[0]
        if record_type not in stats['by_type']:
            stats['by_type'][record_type] = {'files': 0, 'records': 0, 'bytes': 0}

        stats['by_type'][record_type]['files'] += 1
        stats['by_type'][record_type]['bytes'] += chunk_file.stat().st_size

        with open(chunk_file, 'r', encoding='utf-8') as f:
            record_count = sum(1 for line in f if line.strip())
            stats['total_records'] += record_count
            stats['by_type'][record_type]['records'] += record_count

    return stats


def migrate_to_chunked(source_dir: Path, record_type: str, session_id: str) -> int:
    """
    Migrate existing JSON files to chunked storage.

    Args:
        source_dir: Directory containing JSON files to migrate
        record_type: Type of records ('chat', 'commit')
        session_id: Session ID for the migration

    Returns:
        Number of files migrated
    """
    if not source_dir.exists():
        return 0

    migrated = 0

    for json_file in source_dir.glob("**/*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if record_type == 'chat':
                store_chunked_chat(data, session_id)
            elif record_type == 'commit':
                store_chunked_commit(data, session_id)
            else:
                store_chunked_record(record_type, data.get('id', json_file.stem), data, session_id)

            migrated += 1
        except (json.JSONDecodeError, IOError):
            continue

    return migrated
