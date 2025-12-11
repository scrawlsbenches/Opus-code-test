"""
Chunk-based indexing for git-compatible corpus storage.

This module provides append-only, time-stamped JSON chunks that can be
safely committed to git without merge conflicts. Each indexing session
creates a uniquely named chunk file containing document operations.

Architecture:
    corpus_chunks/                        # Tracked in git
    ├── 2025-12-10_21-53-45_a1b2.json    # Session 1 changes
    ├── 2025-12-10_22-15-30_c3d4.json    # Session 2 changes
    └── 2025-12-10_23-00-00_e5f6.json    # Session 3 changes

    corpus_dev.pkl                        # NOT tracked (local cache)

Chunk Format:
    {
        "version": 1,
        "timestamp": "2025-12-10T21:53:45",
        "session_id": "a1b2c3d4",
        "branch": "main",
        "operations": [
            {"op": "add", "doc_id": "...", "content": "...", "mtime": 123},
            {"op": "modify", "doc_id": "...", "content": "...", "mtime": 124},
            {"op": "delete", "doc_id": "..."}
        ]
    }
"""

import hashlib
import json
import os
import subprocess
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Chunk format version
CHUNK_VERSION = 1

# Default size threshold for chunk size warnings (in KB)
# Chunks larger than this may bloat git history
DEFAULT_WARN_SIZE_KB = 1024  # 1MB


@dataclass
class ChunkOperation:
    """A single operation in a chunk (add, modify, or delete)."""
    op: str  # 'add', 'modify', 'delete'
    doc_id: str
    content: Optional[str] = None  # None for delete operations
    mtime: Optional[float] = None  # Modification time
    metadata: Optional[Dict[str, Any]] = None  # Document metadata (doc_type, headings, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {'op': self.op, 'doc_id': self.doc_id}
        if self.content is not None:
            d['content'] = self.content
        if self.mtime is not None:
            d['mtime'] = self.mtime
        if self.metadata is not None:
            d['metadata'] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ChunkOperation':
        """Create from dictionary."""
        return cls(
            op=d['op'],
            doc_id=d['doc_id'],
            content=d.get('content'),
            mtime=d.get('mtime'),
            metadata=d.get('metadata')
        )


@dataclass
class Chunk:
    """A chunk containing operations from a single indexing session."""
    version: int
    timestamp: str
    session_id: str
    branch: str
    operations: List[ChunkOperation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'branch': self.branch,
            'operations': [op.to_dict() for op in self.operations]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Chunk':
        """Create from dictionary."""
        return cls(
            version=d.get('version', 1),
            timestamp=d['timestamp'],
            session_id=d['session_id'],
            branch=d.get('branch', 'unknown'),
            operations=[ChunkOperation.from_dict(op) for op in d['operations']]
        )

    def get_filename(self) -> str:
        """Generate filename for this chunk."""
        # Format: YYYY-MM-DD_HH-MM-SS_sessionid.json
        ts = self.timestamp.replace(':', '-').replace('T', '_')
        short_id = self.session_id[:8]
        return f"{ts}_{short_id}.json"


class ChunkWriter:
    """
    Writes indexing session changes to timestamped JSON chunks.

    Usage:
        writer = ChunkWriter(chunks_dir='corpus_chunks')
        writer.add_document('doc1', 'content here', mtime=1234567890)
        writer.modify_document('doc2', 'new content', mtime=1234567891)
        writer.delete_document('doc3')
        chunk_path = writer.save()
    """

    def __init__(self, chunks_dir: str = 'corpus_chunks'):
        self.chunks_dir = Path(chunks_dir)
        self.session_id = uuid.uuid4().hex[:16]
        self.timestamp = datetime.now().isoformat(timespec='seconds')
        self.branch = self._get_git_branch()
        self.operations: List[ChunkOperation] = []

    def _get_git_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return 'unknown'

    def add_document(
        self,
        doc_id: str,
        content: str,
        mtime: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record an add operation."""
        self.operations.append(ChunkOperation(
            op='add',
            doc_id=doc_id,
            content=content,
            mtime=mtime,
            metadata=metadata
        ))

    def modify_document(
        self,
        doc_id: str,
        content: str,
        mtime: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a modify operation."""
        self.operations.append(ChunkOperation(
            op='modify',
            doc_id=doc_id,
            content=content,
            mtime=mtime,
            metadata=metadata
        ))

    def delete_document(self, doc_id: str):
        """Record a delete operation."""
        self.operations.append(ChunkOperation(
            op='delete',
            doc_id=doc_id
        ))

    def has_operations(self) -> bool:
        """Check if any operations were recorded."""
        return len(self.operations) > 0

    def save(self, warn_size_kb: int = DEFAULT_WARN_SIZE_KB) -> Optional[Path]:
        """
        Save chunk to file.

        Args:
            warn_size_kb: Emit a warning if the saved chunk exceeds this size
                in kilobytes. Set to 0 to disable warning. Default is 1024 KB (1MB).

        Returns:
            Path to saved chunk file, or None if no operations.
        """
        if not self.operations:
            return None

        # Create chunks directory if needed
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create chunk
        chunk = Chunk(
            version=CHUNK_VERSION,
            timestamp=self.timestamp,
            session_id=self.session_id,
            branch=self.branch,
            operations=self.operations
        )

        # Write to file
        filepath = self.chunks_dir / chunk.get_filename()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False)

        # Check file size and warn if too large
        if warn_size_kb > 0:
            file_size_bytes = filepath.stat().st_size
            file_size_kb = file_size_bytes / 1024
            if file_size_kb > warn_size_kb:
                warnings.warn(
                    f"Chunk file '{filepath.name}' is {file_size_kb:.1f}KB "
                    f"(exceeds {warn_size_kb}KB threshold). "
                    f"Large chunks may bloat git history. "
                    f"Consider running --compact to consolidate old chunks.",
                    UserWarning
                )

        return filepath


class ChunkLoader:
    """
    Loads and combines chunks to rebuild document state.

    Usage:
        loader = ChunkLoader(chunks_dir='corpus_chunks')
        documents = loader.load_all()  # Returns {doc_id: content}
        metadata = loader.get_metadata()  # Returns {doc_id: metadata_dict}

        # Check if cache is valid
        if loader.is_cache_valid('corpus_dev.pkl'):
            # Load from pkl
        else:
            # Rebuild from documents
    """

    def __init__(self, chunks_dir: str = 'corpus_chunks'):
        self.chunks_dir = Path(chunks_dir)
        self._chunks: List[Chunk] = []
        self._documents: Dict[str, str] = {}
        self._mtimes: Dict[str, float] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def get_chunk_files(self) -> List[Path]:
        """Get all chunk files sorted by timestamp."""
        if not self.chunks_dir.exists():
            return []

        files = list(self.chunks_dir.glob('*.json'))
        # Sort by filename (which starts with timestamp)
        return sorted(files, key=lambda p: p.name)

    def load_chunk(self, filepath: Path) -> Chunk:
        """Load a single chunk file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Chunk.from_dict(data)

    def load_all(self) -> Dict[str, str]:
        """
        Load all chunks and replay operations to get current document state.

        Returns:
            Dictionary mapping doc_id to content.
        """
        if self._loaded:
            return self._documents

        self._chunks = []
        self._documents = {}
        self._mtimes = {}
        self._metadata = {}

        for filepath in self.get_chunk_files():
            chunk = self.load_chunk(filepath)
            self._chunks.append(chunk)

            # Replay operations
            for op in chunk.operations:
                if op.op == 'add':
                    self._documents[op.doc_id] = op.content
                    if op.mtime:
                        self._mtimes[op.doc_id] = op.mtime
                    if op.metadata:
                        self._metadata[op.doc_id] = op.metadata
                elif op.op == 'modify':
                    self._documents[op.doc_id] = op.content
                    if op.mtime:
                        self._mtimes[op.doc_id] = op.mtime
                    if op.metadata:
                        self._metadata[op.doc_id] = op.metadata
                elif op.op == 'delete':
                    self._documents.pop(op.doc_id, None)
                    self._mtimes.pop(op.doc_id, None)
                    self._metadata.pop(op.doc_id, None)

        self._loaded = True
        return self._documents

    def get_documents(self) -> Dict[str, str]:
        """Get loaded documents (calls load_all if needed)."""
        if not self._loaded:
            self.load_all()
        return self._documents

    def get_mtimes(self) -> Dict[str, float]:
        """Get document modification times."""
        if not self._loaded:
            self.load_all()
        return self._mtimes

    def get_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get document metadata (doc_type, headings, etc.)."""
        if not self._loaded:
            self.load_all()
        return self._metadata

    def get_chunks(self) -> List[Chunk]:
        """Get loaded chunks."""
        if not self._loaded:
            self.load_all()
        return self._chunks

    def compute_hash(self) -> str:
        """
        Compute hash of current document state.

        Used to check if pkl cache is still valid.
        """
        if not self._loaded:
            self.load_all()

        # Hash based on sorted (doc_id, content) pairs
        hasher = hashlib.sha256()
        for doc_id in sorted(self._documents.keys()):
            hasher.update(doc_id.encode('utf-8'))
            hasher.update(self._documents[doc_id].encode('utf-8'))

        return hasher.hexdigest()[:16]

    def is_cache_valid(self, cache_path: str, cache_hash_path: Optional[str] = None) -> bool:
        """
        Check if pkl cache is valid for current chunk state.

        Args:
            cache_path: Path to pkl cache file
            cache_hash_path: Path to hash file (defaults to cache_path + '.hash')

        Returns:
            True if cache exists and hash matches
        """
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return False

        hash_file = Path(cache_hash_path or f"{cache_path}.hash")
        if not hash_file.exists():
            return False

        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()

            current_hash = self.compute_hash()
            return stored_hash == current_hash
        except (IOError, OSError):
            return False

    def save_cache_hash(self, cache_path: str, cache_hash_path: Optional[str] = None):
        """Save current document hash for cache validation."""
        hash_file = Path(cache_hash_path or f"{cache_path}.hash")
        current_hash = self.compute_hash()

        with open(hash_file, 'w') as f:
            f.write(current_hash)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded chunks."""
        if not self._loaded:
            self.load_all()

        total_ops = sum(len(c.operations) for c in self._chunks)
        add_ops = sum(
            1 for c in self._chunks
            for op in c.operations if op.op == 'add'
        )
        modify_ops = sum(
            1 for c in self._chunks
            for op in c.operations if op.op == 'modify'
        )
        delete_ops = sum(
            1 for c in self._chunks
            for op in c.operations if op.op == 'delete'
        )

        return {
            'chunk_count': len(self._chunks),
            'document_count': len(self._documents),
            'total_operations': total_ops,
            'add_operations': add_ops,
            'modify_operations': modify_ops,
            'delete_operations': delete_ops,
            'hash': self.compute_hash()
        }


class ChunkCompactor:
    """
    Compacts multiple chunk files into a single file.

    Usage:
        compactor = ChunkCompactor(chunks_dir='corpus_chunks')
        compactor.compact(before='2025-12-01')  # Compact old chunks
        compactor.compact()  # Compact all chunks into one
    """

    def __init__(self, chunks_dir: str = 'corpus_chunks'):
        self.chunks_dir = Path(chunks_dir)

    def compact(
        self,
        before: Optional[str] = None,
        keep_recent: int = 0,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Compact chunks into a single chunk.

        Args:
            before: Only compact chunks before this date (YYYY-MM-DD)
            keep_recent: Keep this many recent chunks uncompacted
            dry_run: If True, don't actually compact, just report what would happen

        Returns:
            Statistics about the compaction
        """
        loader = ChunkLoader(str(self.chunks_dir))
        chunk_files = loader.get_chunk_files()

        if not chunk_files:
            return {'status': 'no_chunks', 'compacted': 0}

        # Filter chunks to compact
        to_compact = []
        to_keep = []

        for filepath in chunk_files:
            filename = filepath.name
            # Extract date from filename (YYYY-MM-DD_HH-MM-SS_...)
            file_date = filename[:10]

            should_compact = True

            if before:
                should_compact = file_date < before

            if should_compact:
                to_compact.append(filepath)
            else:
                to_keep.append(filepath)

        # Keep recent chunks if requested
        if keep_recent > 0 and len(to_compact) > keep_recent:
            # Move some from to_compact to to_keep
            to_keep = to_compact[-keep_recent:] + to_keep
            to_compact = to_compact[:-keep_recent]

        if not to_compact:
            return {'status': 'nothing_to_compact', 'compacted': 0}

        if dry_run:
            return {
                'status': 'dry_run',
                'would_compact': len(to_compact),
                'would_keep': len(to_keep),
                'files_to_compact': [str(f) for f in to_compact]
            }

        # Load and merge chunks to compact
        documents = {}
        mtimes = {}
        metadata = {}

        for filepath in to_compact:
            chunk = loader.load_chunk(filepath)
            for op in chunk.operations:
                if op.op in ('add', 'modify'):
                    documents[op.doc_id] = op.content
                    if op.mtime:
                        mtimes[op.doc_id] = op.mtime
                    if op.metadata:
                        metadata[op.doc_id] = op.metadata
                elif op.op == 'delete':
                    documents.pop(op.doc_id, None)
                    mtimes.pop(op.doc_id, None)
                    metadata.pop(op.doc_id, None)

        # Create compacted chunk with all remaining documents as 'add' operations
        writer = ChunkWriter(str(self.chunks_dir))
        writer.timestamp = datetime.now().isoformat(timespec='seconds')
        writer.session_id = 'compacted_' + uuid.uuid4().hex[:8]

        for doc_id, content in sorted(documents.items()):
            writer.add_document(doc_id, content, mtimes.get(doc_id), metadata.get(doc_id))

        # Save compacted chunk
        compacted_path = None
        if writer.has_operations():
            compacted_path = writer.save()

        # Delete old chunk files
        for filepath in to_compact:
            filepath.unlink()

        return {
            'status': 'compacted',
            'compacted': len(to_compact),
            'kept': len(to_keep),
            'documents': len(documents),
            'compacted_file': str(compacted_path) if compacted_path else None
        }


def get_changes_from_manifest(
    current_files: Dict[str, float],
    manifest: Dict[str, float]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare current files to manifest to find changes.

    Args:
        current_files: Dict mapping file paths to modification times
        manifest: Dict mapping file paths to last indexed modification times

    Returns:
        Tuple of (added, modified, deleted) file lists
    """
    current_set = set(current_files.keys())
    manifest_set = set(manifest.keys())

    added = list(current_set - manifest_set)
    deleted = list(manifest_set - current_set)

    # Check for modified files
    modified = []
    for filepath in current_set & manifest_set:
        if current_files[filepath] > manifest[filepath]:
            modified.append(filepath)

    return added, modified, deleted
