"""
Incremental Training System for CognitiveAgent.

This module provides infrastructure for training a CognitiveAgent on text
documents, with intelligent tracking to only process new or modified files.

Design Philosophy:
    - Track what's been trained using content hashes (not just filenames)
    - Detect new files AND modified files automatically
    - Make training idempotent: running twice does nothing if no changes
    - Support incremental vocabulary learning
    - Provide clear statistics for observability

Usage:
    >>> from cortical.cognitive.training import IncrementalTrainer
    >>> from cortical.cognitive.graph import CognitiveAgent
    >>>
    >>> agent = CognitiveAgent()
    >>> trainer = IncrementalTrainer(agent, model_dir="models/cognitive")
    >>>
    >>> # Train on all txt files in samples/
    >>> stats = trainer.train_directory("samples/")
    >>> print(f"Trained on {stats['new_documents']} new documents")
    >>>
    >>> # Later, after adding new samples:
    >>> stats = trainer.train_directory("samples/")  # Only trains new files

Architecture:
    TrainingManifest
        - Tracks trained files with content hashes
        - Detects new, modified, and deleted files
        - Persisted as JSON

    IncrementalTrainer
        - Orchestrates training with manifest
        - Uses TextToAtomsBridge for text processing
        - Provides statistics and reporting
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cortical.cognitive.graph import CognitiveAgent

from cortical.common.filesystem import FileSystem, RealFileSystem, InMemoryFileSystem
from cortical.cognitive.text_bridge import (
    TextToAtomsBridge,
    ProgressReporter,
)
from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
from cortical.cognitive.graph import Atom, AtomType, TruthValue


# =============================================================================
# Training Manifest
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training behavior."""
    staleness_warning_threshold: float = 0.2  # 20% growth triggers warning


@dataclass
class TrainedDocument:
    """Record of a trained document."""

    path: str  # Relative path from samples directory
    content_hash: str  # SHA256 of content
    trained_at: str  # ISO timestamp
    word_count: int  # Number of tokens extracted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "trained_at": self.trained_at,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainedDocument':
        return cls(
            path=data["path"],
            content_hash=data["content_hash"],
            trained_at=data["trained_at"],
            word_count=data.get("word_count", 0),
        )


@dataclass
class TrainingManifest:
    """
    Tracks which documents have been trained on.

    Uses content hashes to detect:
    - New files (path not in manifest)
    - Modified files (path exists but hash differs)
    - Deleted files (path in manifest but file gone)

    Attributes:
        documents: Dict mapping relative path -> TrainedDocument
        last_training: ISO timestamp of last training run
        total_documents: Total documents ever trained
        vocabulary_size: Size of vocabulary at last save
    """

    documents: Dict[str, TrainedDocument] = field(default_factory=dict)
    last_training: Optional[str] = None
    total_documents: int = 0
    vocabulary_size: int = 0
    model_version: str = "1.0"
    last_reindex_doc_count: int = 0  # corpus size at last IDF reindex
    idf_epoch: int = 0  # increments each time IDF is recalculated

    def add_document(
        self,
        path: str,
        content_hash: str,
        word_count: int = 0,
    ) -> None:
        """Record a newly trained document."""
        self.documents[path] = TrainedDocument(
            path=path,
            content_hash=content_hash,
            trained_at=datetime.now().isoformat(),
            word_count=word_count,
        )
        self.total_documents = len(self.documents)

    def is_trained(self, path: str, content_hash: str) -> bool:
        """Check if a document with this exact content has been trained."""
        if path not in self.documents:
            return False
        return self.documents[path].content_hash == content_hash

    def get_untrained(
        self,
        files: List[Tuple[str, str, str]],  # (path, content, hash)
    ) -> List[Tuple[str, str]]:
        """
        Filter to only untrained or modified documents.

        Args:
            files: List of (relative_path, content, content_hash) tuples

        Returns:
            List of (path, content) tuples for documents needing training
        """
        untrained = []
        for path, content, content_hash in files:
            if not self.is_trained(path, content_hash):
                untrained.append((path, content))
        return untrained

    def save(self, path: Path, filesystem: FileSystem) -> None:
        """Save manifest to JSON file."""
        data = {
            "model_version": self.model_version,
            "last_training": self.last_training,
            "total_documents": self.total_documents,
            "vocabulary_size": self.vocabulary_size,
            "last_reindex_doc_count": self.last_reindex_doc_count,
            "idf_epoch": self.idf_epoch,
            "documents": {
                k: v.to_dict() for k, v in self.documents.items()
            },
        }
        content = json.dumps(data, indent=2)
        filesystem.write_text(path, content)

    @classmethod
    def load(cls, path: Path, filesystem: FileSystem) -> 'TrainingManifest':
        """Load manifest from JSON file."""
        if not filesystem.exists(path):
            return cls()
        data = json.loads(filesystem.read_text(path))

        manifest = cls(
            last_training=data.get("last_training"),
            total_documents=data.get("total_documents", 0),
            vocabulary_size=data.get("vocabulary_size", 0),
            model_version=data.get("model_version", "1.0"),
            last_reindex_doc_count=data.get("last_reindex_doc_count", 0),
            idf_epoch=data.get("idf_epoch", 0),
        )

        for path_key, doc_data in data.get("documents", {}).items():
            manifest.documents[path_key] = TrainedDocument.from_dict(doc_data)

        return manifest

    def get_staleness(self) -> float:
        """Calculate IDF staleness as fraction of corpus growth since last reindex."""
        if self.last_reindex_doc_count == 0:
            return 0.0
        return (self.total_documents - self.last_reindex_doc_count) / self.last_reindex_doc_count


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


# =============================================================================
# Incremental Trainer
# =============================================================================


@dataclass
class TrainingStats:
    """Statistics from a training run."""

    total_files_scanned: int = 0
    new_documents: int = 0
    modified_documents: int = 0
    skipped_documents: int = 0
    atoms_created: int = 0
    links_created: int = 0
    vocabulary_size: int = 0
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_scanned": self.total_files_scanned,
            "new_documents": self.new_documents,
            "modified_documents": self.modified_documents,
            "skipped_documents": self.skipped_documents,
            "atoms_created": self.atoms_created,
            "links_created": self.links_created,
            "vocabulary_size": self.vocabulary_size,
            "training_time_seconds": round(self.training_time_seconds, 2),
        }

    def __str__(self) -> str:
        if self.new_documents == 0 and self.modified_documents == 0:
            return "No new or modified documents to train on."

        return (
            f"Training complete:\n"
            f"  Scanned: {self.total_files_scanned} files\n"
            f"  New: {self.new_documents}, Modified: {self.modified_documents}, "
            f"Skipped: {self.skipped_documents}\n"
            f"  Atoms created: {self.atoms_created}\n"
            f"  Links created: {self.links_created}\n"
            f"  Vocabulary: {self.vocabulary_size} words\n"
            f"  Time: {self.training_time_seconds:.1f}s"
        )


class IncrementalTrainer:
    """
    Orchestrates incremental training of a CognitiveAgent.

    Tracks what's been trained and only processes new or modified files.
    Persists both the trained model and the manifest for future runs.

    Attributes:
        agent: The CognitiveAgent to train
        model_dir: Directory for persisting model and manifest
        bridge: TextToAtomsBridge for text processing
        manifest: TrainingManifest tracking trained documents
        filesystem: FileSystem abstraction for I/O (enables in-memory testing)

    Usage:
        >>> trainer = IncrementalTrainer(agent, "models/cognitive")
        >>> stats = trainer.train_directory("samples/")
        >>> print(stats)
        Training complete:
          Scanned: 100 files
          New: 10, Modified: 2, Skipped: 88
          ...

        # For testing with in-memory filesystem:
        >>> fs = InMemoryFileSystem(Path("/test"))
        >>> trainer = IncrementalTrainer(agent, "/test/model", filesystem=fs)
    """

    def __init__(
        self,
        agent: 'CognitiveAgent',
        model_dir: str | Path,
        filesystem: FileSystem,
        checkpoint_interval: int = 50,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            agent: CognitiveAgent to train
            model_dir: Directory for model persistence
            filesystem: FileSystem for I/O operations
            checkpoint_interval: Save progress every N documents (default 50)
            config: Training configuration (default: TrainingConfig())
        """
        self.agent = agent
        self.config = config or TrainingConfig()
        self.model_dir = Path(model_dir)
        self.filesystem = filesystem
        self.checkpoint_interval = checkpoint_interval

        # Create model directory
        self.filesystem.mkdir(self.model_dir, parents=True, exist_ok=True)

        # Load or create manifest
        self.manifest_path = self.model_dir / "training_manifest.json"
        self.manifest = TrainingManifest.load(self.manifest_path, self.filesystem)

        # Initialize bridge
        self.bridge = TextToAtomsBridge(agent.graph)

        # Initialize tokenizer storage
        self.tokenizer_storage = ShardedTokenizerStorage(self.filesystem)

        # Load existing tokenizer if available (from sharded directory)
        tokenizer_dir = self.model_dir / "tokenizer"
        if self.filesystem.exists(tokenizer_dir / "meta.json"):
            self.bridge.tokenizer = self.tokenizer_storage.load(tokenizer_dir)

        # Load existing graph state if available (CRITICAL for session recovery)
        bridge_dir = self.model_dir / "bridge"
        if bridge_dir.exists():
            self._load_graph_state(bridge_dir)

    def _load_graph_state(self, bridge_dir: Path) -> None:
        """
        Load graph state from sharded files or legacy graph.json.

        This is CRITICAL for session recovery - without this, all learned
        atoms and links are lost when resuming training in a new session.

        PERFORMANCE NOTE (2026-01-11):
        This method uses direct Atom instantiation instead of graph.node()
        and graph.link() to achieve O(n) loading instead of O(n²).

        The bottleneck was graph.link() calling find_by_type() for each link,
        which is O(n) per call. With 23,653 links, this caused ~560 million
        comparisons and 32+ second load times.

        Direct Atom creation: 0.15s (200x faster)

        Args:
            bridge_dir: Directory containing graph shards or graph.json
        """
        from cortical.cognitive.graph_storage import ShardedGraphStorage

        storage_handler = ShardedGraphStorage()
        atoms_data = storage_handler.load(bridge_dir)

        if not atoms_data:
            return

        storage = self.agent.graph._storage

        # OPTIMIZED: Direct Atom instantiation bypasses O(n²) find_by_type()
        # We load all atoms in a single pass since we have the complete data
        id_map = {}  # old_id -> new_atom

        for atom_data in atoms_data:
            # Handle both old format (tv_strength) and new format (tv.strength)
            if "tv" in atom_data:
                tv = TruthValue(atom_data["tv"]["strength"], atom_data["tv"]["confidence"])
            else:
                tv = TruthValue(atom_data.get("tv_strength", 0.5), atom_data.get("tv_confidence", 0.5))

            atom = Atom(
                id=atom_data["id"],
                atom_type=AtomType[atom_data["atom_type"]],
                name=atom_data.get("name", ""),
                outgoing=atom_data.get("outgoing", []),
                tv=tv,
                sti=atom_data.get("sti", 0.0),
                lti=atom_data.get("lti", 0.0),
                metadata=atom_data.get("metadata", {}),  # Restore IDF weights
            )
            id_map[atom_data["id"]] = atom
            storage.save(atom)

        # Mark storage as clean after loading (enables incremental saves)
        if hasattr(storage, 'mark_all_clean_after_load'):
            storage.mark_all_clean_after_load()

        # Restore bridge stats from meta.json if available
        meta_path = bridge_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            stats = meta.get("bridge_stats", {})
        else:
            stats = {}

        self.bridge._documents_fed = stats.get("documents_fed", 0)
        self.bridge._atoms_created = stats.get("atoms_created", 0)
        self.bridge._links_created = stats.get("links_created", 0)

    def _check_staleness_warning(self) -> None:
        """Emit warning to stderr if IDF weights are stale."""
        staleness = self.manifest.get_staleness()
        if staleness > self.config.staleness_warning_threshold:
            import sys
            print(
                f"Warning: IDF weights are {staleness:.0%} stale. "
                f"Consider running --reindex to update link weights.",
                file=sys.stderr,
            )

    def scan_directory(
        self,
        directory: str | Path,
        pattern: str = "*.txt",
        recursive: bool = True,
    ) -> Iterator[Tuple[str, str, str]]:
        """
        Scan directory for text files.

        Yields:
            (relative_path, content, content_hash) tuples
        """
        directory = Path(directory)

        # Use filesystem glob for pattern matching
        glob_pattern = f"**/{pattern}" if recursive else pattern
        files = self.filesystem.glob(directory, glob_pattern)

        for file_path in sorted(files):
            if self.filesystem.is_dir(file_path):
                continue

            try:
                content = self._read_text_file(file_path)
                relative_path = str(file_path.relative_to(directory))
                content_hash = compute_content_hash(content)
                yield relative_path, content, content_hash
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

    def _read_text_file(self, path: Path) -> str:
        """Read text file with encoding fallback."""
        try:
            return self.filesystem.read_text(path)
        except UnicodeDecodeError:
            # For real filesystem, try latin-1 fallback
            if isinstance(self.filesystem, RealFileSystem):
                return path.read_text(encoding='latin-1')
            raise

    def train_directory(
        self,
        directory: str | Path,
        pattern: str = "*.txt",
        recursive: bool = True,
        show_progress: bool = True,
        force_retrain: bool = False,
        checkpoint_interval: Optional[int] = None,
    ) -> TrainingStats:
        """
        Train on documents in a directory (incrementally).

        Only processes documents that are new or modified since last training.
        Checkpoints progress periodically for crash recovery.

        Args:
            directory: Directory containing text files
            pattern: Glob pattern for files (default: *.txt)
            recursive: Search subdirectories (default: True)
            show_progress: Show progress bar (default: True)
            force_retrain: Ignore manifest and retrain all (default: False)
            checkpoint_interval: Override checkpoint interval for this run

        Returns:
            TrainingStats with details of what was trained
        """
        import time
        start_time = time.time()

        stats = TrainingStats()
        directory = Path(directory)

        # Check for stale IDF weights before training
        self._check_staleness_warning()

        # Scan all files
        all_files = list(self.scan_directory(directory, pattern, recursive))
        stats.total_files_scanned = len(all_files)

        if not all_files:
            print(f"No files matching '{pattern}' found in {directory}")
            return stats

        # Filter to untrained documents
        if force_retrain:
            to_train = [(path, content) for path, content, _ in all_files]
            stats.new_documents = len(to_train)
        else:
            to_train = self.manifest.get_untrained(all_files)
            stats.skipped_documents = stats.total_files_scanned - len(to_train)

            # Count new vs modified
            for path, _ in to_train:
                if path in self.manifest.documents:
                    stats.modified_documents += 1
                else:
                    stats.new_documents += 1

        if not to_train:
            stats.training_time_seconds = time.time() - start_time
            stats.vocabulary_size = len(self.bridge.tokenizer.vocab)
            if show_progress:
                print("All documents already trained. Nothing to do.")
            return stats

        # Extract texts and paths
        paths = [p for p, _ in to_train]
        texts = [t for _, t in to_train]

        # Phase 1: Learn vocabulary incrementally
        if show_progress:
            print(f"Learning vocabulary from {len(texts)} documents...", file=sys.stderr)

        self.bridge.learn_vocabulary(texts, incremental=True)

        if show_progress:
            print(f"  Vocabulary: {len(self.bridge.tokenizer.vocab)} words", file=sys.stderr)

        # Phase 2: Feed documents to create atoms
        atoms_before = self.bridge._atoms_created
        links_before = self.bridge._links_created

        # Determine checkpoint interval for this run
        interval = checkpoint_interval if checkpoint_interval is not None else self.checkpoint_interval
        docs_since_checkpoint = 0

        if show_progress:
            with ProgressReporter(len(texts), desc="Training") as progress:
                for path, text in zip(paths, texts):
                    self.bridge.feed_text(text, doc_id=path)

                    # Update manifest
                    content_hash = compute_content_hash(text)
                    word_count = len(self.bridge.tokenizer.tokenize(text))
                    self.manifest.add_document(path, content_hash, word_count)

                    docs_since_checkpoint += 1
                    progress.update(1)

                    # Checkpoint periodically for crash recovery
                    if interval > 0 and docs_since_checkpoint >= interval:
                        self.manifest.last_training = datetime.now().isoformat()
                        self.manifest.vocabulary_size = len(self.bridge.tokenizer.vocab)
                        self.save()
                        docs_since_checkpoint = 0
        else:
            for path, text in zip(paths, texts):
                self.bridge.feed_text(text, doc_id=path)
                content_hash = compute_content_hash(text)
                word_count = len(self.bridge.tokenizer.tokenize(text))
                self.manifest.add_document(path, content_hash, word_count)

                docs_since_checkpoint += 1

                # Checkpoint periodically for crash recovery
                if interval > 0 and docs_since_checkpoint >= interval:
                    self.manifest.last_training = datetime.now().isoformat()
                    self.manifest.vocabulary_size = len(self.bridge.tokenizer.vocab)
                    self.save()
                    docs_since_checkpoint = 0

        # Update stats
        stats.atoms_created = self.bridge._atoms_created - atoms_before
        stats.links_created = self.bridge._links_created - links_before
        stats.vocabulary_size = len(self.bridge.tokenizer.vocab)
        stats.training_time_seconds = time.time() - start_time

        # Update manifest metadata
        self.manifest.last_training = datetime.now().isoformat()
        self.manifest.vocabulary_size = stats.vocabulary_size

        # Save everything
        self.save()

        if show_progress:
            print(f"\n{stats}")

        return stats

    def train_files(
        self,
        file_paths: List[str | Path],
        base_dir: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> TrainingStats:
        """
        Train on specific files only.

        Args:
            file_paths: List of file paths to train on
            base_dir: Base directory for relative paths (default: common parent)
            show_progress: Show progress bar

        Returns:
            TrainingStats
        """
        import time
        start_time = time.time()

        stats = TrainingStats()

        # Check for stale IDF weights before training
        self._check_staleness_warning()

        # Resolve paths and load content
        files_data = []
        for fp in file_paths:
            fp = Path(fp)
            if not self.filesystem.exists(fp):
                print(f"Warning: File not found: {fp}", file=sys.stderr)
                continue

            try:
                content = self._read_text_file(fp)
                content_hash = compute_content_hash(content)

                # Compute relative path
                if base_dir:
                    rel_path = str(fp.relative_to(base_dir))
                else:
                    rel_path = fp.name

                files_data.append((rel_path, content, content_hash))
            except Exception as e:
                print(f"Warning: Could not read {fp}: {e}", file=sys.stderr)

        stats.total_files_scanned = len(files_data)

        # Filter to untrained
        to_train = self.manifest.get_untrained(files_data)
        stats.skipped_documents = stats.total_files_scanned - len(to_train)
        stats.new_documents = len(to_train)

        if not to_train:
            stats.training_time_seconds = time.time() - start_time
            if show_progress:
                print("All specified files already trained.")
            return stats

        # Train
        paths = [p for p, _ in to_train]
        texts = [t for _, t in to_train]

        self.bridge.learn_vocabulary(texts, incremental=True)

        atoms_before = self.bridge._atoms_created
        links_before = self.bridge._links_created

        for path, text in zip(paths, texts):
            self.bridge.feed_text(text, doc_id=path)
            content_hash = compute_content_hash(text)
            word_count = len(self.bridge.tokenizer.tokenize(text))
            self.manifest.add_document(path, content_hash, word_count)

        stats.atoms_created = self.bridge._atoms_created - atoms_before
        stats.links_created = self.bridge._links_created - links_before
        stats.vocabulary_size = len(self.bridge.tokenizer.vocab)
        stats.training_time_seconds = time.time() - start_time

        self.manifest.last_training = datetime.now().isoformat()
        self.manifest.vocabulary_size = stats.vocabulary_size
        self.save()

        if show_progress:
            print(f"\n{stats}")

        return stats

    def save(self) -> None:
        """Save model, tokenizer, and manifest."""
        from cortical.cognitive.graph_storage import ShardedGraphStorage

        # Initialize staleness tracking baseline on first save
        # This enables proper staleness calculation after initial training
        if self.manifest.last_reindex_doc_count == 0 and self.manifest.total_documents > 0:
            self.manifest.last_reindex_doc_count = self.manifest.total_documents
            self.manifest.idf_epoch = 1  # Mark as having initial IDF values

        # Save tokenizer to sharded directory (merge-conflict-free)
        tokenizer_dir = self.model_dir / "tokenizer"
        self.tokenizer_storage.save_incremental(self.bridge.tokenizer, tokenizer_dir)

        # Save bridge graph data using sharded storage (git-friendly)
        bridge_dir = self.model_dir / "bridge"
        self.filesystem.mkdir(bridge_dir, parents=True, exist_ok=True)

        storage = ShardedGraphStorage()
        result = storage.save(self.bridge.graph, bridge_dir)

        # Save stats in meta.json
        meta_path = bridge_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        meta["bridge_stats"] = self.bridge.get_statistics()
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Save manifest
        self.manifest.save(self.manifest_path, self.filesystem)

    def reindex(self, show_progress: bool = True) -> Dict[str, Any]:
        """
        Recalculate IDF weights for all similarity links.

        This should be called after incremental training to update stale
        link weights. Updates manifest with new reindex stats.

        Args:
            show_progress: Print progress info to stderr

        Returns:
            Dict with reindex statistics
        """
        import sys

        if show_progress:
            n_links = len(self.bridge.get_similarity_links())
            print(f"Reindexing {n_links} links...", file=sys.stderr)

        result = self.bridge.reindex_idf()

        # Update manifest
        self.manifest.last_reindex_doc_count = self.manifest.total_documents
        self.manifest.idf_epoch = result['new_epoch']

        # Save updated state
        self.save()

        if show_progress:
            print(f"Reindex complete: {result['links_updated']} links updated in {result['time_ms']}ms", file=sys.stderr)
            print(f"IDF epoch: {result['new_epoch']}", file=sys.stderr)

        return result

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        filesystem: FileSystem,
        agent: 'CognitiveAgent',
    ) -> 'IncrementalTrainer':
        """
        Load a previously trained model.

        Args:
            model_dir: Directory containing saved model
            filesystem: FileSystem for I/O operations
            agent: CognitiveAgent to load into

        Returns:
            IncrementalTrainer with loaded state
        """
        model_dir = Path(model_dir)

        trainer = cls(agent, model_dir, filesystem)

        # Load bridge with graph if exists
        bridge_dir = model_dir / "bridge"
        if trainer.filesystem.exists(bridge_dir):
            graph_path = bridge_dir / "graph.json"
            if trainer.filesystem.exists(graph_path):
                trainer.bridge = TextToAtomsBridge.load(bridge_dir, agent.graph, filesystem)

        return trainer

    def status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "model_dir": str(self.model_dir),
            "total_documents_trained": self.manifest.total_documents,
            "vocabulary_size": self.manifest.vocabulary_size,
            "last_training": self.manifest.last_training,
            "model_version": self.manifest.model_version,
        }

    def list_trained(self) -> List[str]:
        """List all trained document paths."""
        return sorted(self.manifest.documents.keys())


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI for incremental training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Incremental training for CognitiveAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on all .txt files in samples/
  python -m cortical.cognitive.training samples/

  # Train next 5 documents only (safe batch)
  python -m cortical.cognitive.training --batch-size 5

  # Train with custom checkpoint interval
  python -m cortical.cognitive.training --batch-size 25 --checkpoint 10

  # Train on specific files only
  python -m cortical.cognitive.training --files samples/doc1.txt samples/doc2.txt

  # Force retrain everything
  python -m cortical.cognitive.training samples/ --force

  # Check training status
  python -m cortical.cognitive.training --status

  # List trained documents
  python -m cortical.cognitive.training --list
        """,
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default="samples",
        help="Directory containing training documents (default: samples/)",
    )
    parser.add_argument(
        "--model-dir",
        default="models/cognitive_agent",
        help="Directory for model storage (default: models/cognitive_agent)",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for files (default: *.txt)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Train on specific files instead of directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain all documents",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show training status and exit",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List trained documents and exit",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Recalculate IDF weights for all similarity links",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--batch-size", "-n",
        type=int,
        default=None,
        help="Limit training to N documents (for controlled batch training)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=int,
        default=None,
        help="Checkpoint interval (save every N documents)",
    )

    args = parser.parse_args()
    run_cli(args)


def run_cli(args, container: 'Optional[Container]' = None) -> None:
    """
    Execute CLI command with given args.

    This function is separated from main() to allow testing with DI.
    Tests can pass a pre-configured container with InMemoryFileSystem.

    Args:
        args: Parsed command line arguments
        container: Optional DI container. If None, creates one with RealFileSystem.
    """
    from cortical.common import Container
    from cortical.core.modules import CognitiveModule
    # Import from canonical module to avoid __main__ class identity issues with -m
    import cortical.cognitive.training as training_module

    # Use provided container or create one
    if container is None:
        model_dir = Path(args.model_dir)
        container = Container()
        container.apply_module(CognitiveModule(model_dir=model_dir, use_memory=False))

    trainer = container.resolve(training_module.IncrementalTrainer)

    if args.status:
        status = trainer.status()
        print(json.dumps(status, indent=2))
        return

    if args.list:
        trained = trainer.list_trained()
        if trained:
            print(f"Trained documents ({len(trained)}):")
            for doc in trained:
                print(f"  {doc}")
        else:
            print("No documents trained yet.")
        return

    if args.reindex:
        result = trainer.reindex(show_progress=not args.quiet)
        if not args.quiet:
            staleness = trainer.manifest.get_staleness()
            print(f"Staleness after reindex: {staleness:.1%}")
        return

    if args.files:
        stats = trainer.train_files(
            args.files,
            base_dir=args.directory,
            show_progress=not args.quiet,
        )
    elif args.batch_size is not None:
        # Batch mode: train only N untrained documents
        all_files = list(trainer.scan_directory(
            args.directory, args.pattern, recursive=True
        ))
        untrained = trainer.manifest.get_untrained(all_files)

        if not untrained:
            print("All documents are already trained. Nothing to do.")
            return

        # Select batch
        batch = untrained[:args.batch_size]
        paths = [path for path, _ in batch]

        if not args.quiet:
            print(f"Batch training: {len(batch)} of {len(untrained)} remaining documents")
            for i, path in enumerate(paths, 1):
                print(f"  {i}. {path}")
            print()

        # Build full paths and train
        base_dir = Path(args.directory)
        full_paths = [base_dir / path for path in paths]

        stats = trainer.train_files(
            file_paths=full_paths,
            base_dir=base_dir,
            show_progress=not args.quiet,
        )
    else:
        stats = trainer.train_directory(
            args.directory,
            pattern=args.pattern,
            show_progress=not args.quiet,
            force_retrain=args.force,
            checkpoint_interval=args.checkpoint,
        )

    # Print final status
    if not args.quiet:
        print(f"\nModel saved to: {trainer.model_dir}")


def run_cli_command(command: str, args, container: 'Optional[Container]' = None) -> int:
    """
    Execute a CLI command from __main__.py.

    This is the proper entry point for CLI execution, called from
    cortical/cognitive/__main__.py. It avoids class identity issues
    by being called after all imports are complete.

    Args:
        command: Command name ('train', 'status', 'list', 'reindex')
        args: Parsed command line arguments
        container: Optional DI container for testing

    Returns:
        Exit code (0 for success)
    """
    import json
    from cortical.common import Container
    from cortical.core.modules import CognitiveModule

    # Create container if not provided
    if container is None:
        model_dir = Path(args.model_dir)
        container = Container()
        container.apply_module(CognitiveModule(model_dir=model_dir, use_memory=False))

    trainer = container.resolve(IncrementalTrainer)

    if command == "status":
        status = trainer.status()
        print(json.dumps(status, indent=2))
        return 0

    if command == "list":
        trained = trainer.list_trained()
        if trained:
            print(f"Trained documents ({len(trained)}):")
            for doc in trained:
                print(f"  {doc}")
        else:
            print("No documents trained yet.")
        return 0

    if command == "reindex":
        trainer.reindex(show_progress=not args.quiet)
        if not args.quiet:
            staleness = trainer.manifest.get_staleness()
            print(f"Staleness after reindex: {staleness:.1%}")
        return 0

    if command == "train":
        if args.files:
            trainer.train_files(
                args.files,
                base_dir=args.directory,
                show_progress=not args.quiet,
            )
        elif args.batch_size is not None:
            # Batch mode: train only N untrained documents
            all_files = list(trainer.scan_directory(
                args.directory, args.pattern, recursive=True
            ))
            untrained = trainer.manifest.get_untrained(all_files)

            if not untrained:
                print("All documents are already trained. Nothing to do.")
                return 0

            batch = untrained[:args.batch_size]
            paths = [path for path, _ in batch]

            if not args.quiet:
                print(f"Batch training: {len(batch)} of {len(untrained)} remaining documents")
                for i, path in enumerate(paths, 1):
                    print(f"  {i}. {path}")
                print()

            base_dir = Path(args.directory)
            full_paths = [base_dir / path for path in paths]

            trainer.train_files(
                file_paths=full_paths,
                base_dir=base_dir,
                show_progress=not args.quiet,
            )
        else:
            trainer.train_directory(
                args.directory,
                pattern=args.pattern,
                show_progress=not args.quiet,
                force_retrain=args.force,
                checkpoint_interval=args.checkpoint,
            )

        if not args.quiet:
            print(f"\nModel saved to: {trainer.model_dir}")
        return 0

    if command == "rebuild-df":
        # Rebuild document frequency from trained documents
        trained_docs = trainer.list_trained()
        if not trained_docs:
            print("No trained documents found.")
            return 0

        if not args.quiet:
            print(f"Rebuilding document frequency from {len(trained_docs)} documents...")

        # Reset doc frequency
        tok = trainer.bridge.tokenizer
        tok._doc_frequency = {}
        tok._total_docs = 0

        # Re-read each document and count word frequencies
        base_dir = Path("samples/")  # Default training directory
        processed = 0
        for doc_path in trained_docs:
            full_path = base_dir / doc_path
            if not full_path.exists():
                if not args.quiet:
                    print(f"  Skipping (not found): {doc_path}", file=sys.stderr)
                continue

            try:
                text = full_path.read_text(encoding='utf-8')
                tokens = tok.tokenize(text)
                unique_words = set(tokens)
                for word in unique_words:
                    tok._doc_frequency[word] = tok._doc_frequency.get(word, 0) + 1
                tok._total_docs += 1
                processed += 1
            except Exception as e:
                if not args.quiet:
                    print(f"  Error reading {doc_path}: {e}", file=sys.stderr)

        # Save updated tokenizer
        trainer.save()

        if not args.quiet:
            print(f"Rebuilt document frequency:")
            print(f"  Documents processed: {processed}")
            print(f"  Unique words with DF: {len(tok._doc_frequency)}")
            print(f"  Sample IDF(data): {tok.get_idf('data'):.4f}")

        # Also reindex IDF weights on links
        if not args.quiet:
            print("Reindexing IDF weights on links...")
        trainer.reindex(show_progress=not args.quiet)

        return 0

    if command == "query":
        # Query requires the agent, not just the trainer
        agent = trainer.agent
        word = args.word.lower()

        associations = agent.get_associations(
            word,
            weight_type=args.weight_type,
            top_k=args.top_k,
        )

        if not associations:
            print(f"No associations found for '{word}'")
            print("(Word may not exist in vocabulary or have no similarity links)")
            return 0

        if args.json:
            import json
            result = {
                "word": word,
                "weight_type": args.weight_type,
                "associations": [
                    {"word": a.word, "weight": a.weight}
                    for a in associations
                ]
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Associations for '{word}' ({args.weight_type} weights):")
            print("-" * 40)
            for i, assoc in enumerate(associations, 1):
                print(f"  {i:2}. {assoc.word:<20} {assoc.weight:.4f}")

        return 0

    if command == "demo":
        _run_demo(trainer)
        return 0

    if command == "generate":
        _run_generate(trainer, args)
        return 0

    if command == "index-code":
        _run_index_code(trainer, args)
        return 0

    if command == "ask":
        _run_ask(trainer, args)
        return 0

    print(f"Unknown command: {command}")
    return 1


def _run_generate(trainer: 'IncrementalTrainer', args) -> None:
    """
    Generate text using FOLLOWS links and predict_next().

    Uses directional word transitions learned from training data
    to generate text token by token.

    Args:
        trainer: The IncrementalTrainer with loaded model
        args: CLI arguments with prompt, max_tokens, temperature, etc.
    """
    import random
    import json as json_module

    agent = trainer.agent
    graph = agent.graph

    # Get starting word(s)
    if args.prompt:
        # Tokenize the prompt
        tokens = trainer.bridge.tokenizer.tokenize(args.prompt.lower())
        if not tokens:
            print(f"Could not tokenize prompt: {args.prompt}")
            return
        current_word = tokens[-1]  # Start prediction from last word
        generated = list(tokens)
    else:
        # Pick a random word from vocabulary
        word_atoms = [a for a in graph._storage.all_atoms()
                      if a.atom_type.name == 'WORD' and a.name]
        if not word_atoms:
            print("No words in vocabulary. Train the model first.")
            return
        current_word = random.choice(word_atoms).name
        generated = [current_word]

    # Track predictions for JSON output
    predictions = []

    # Generate tokens
    for i in range(args.max_tokens):
        pred = agent.predict_next(current_word)

        # Record prediction details
        pred_record = {
            "step": i + 1,
            "from_word": current_word,
            "is_unknown": pred.is_unknown,
            "is_boundary": pred.is_boundary,
            "confidence": pred.confidence,
            "candidates": pred.candidates[:5],
        }

        if pred.is_unknown:
            pred_record["result"] = "[UNKNOWN]"
            predictions.append(pred_record)
            break

        if pred.is_boundary:
            pred_record["result"] = "[BOUNDARY]"
            predictions.append(pred_record)
            break

        if args.min_confidence > 0 and pred.confidence < args.min_confidence:
            pred_record["result"] = "[LOW_CONFIDENCE]"
            predictions.append(pred_record)
            break

        # Select next word
        if args.temperature == 0 or len(pred.candidates) == 1:
            # Greedy: pick top candidate
            next_word = pred.top
        else:
            # Temperature-based sampling
            import math
            candidates = pred.candidates
            # Apply temperature
            if args.temperature != 1.0:
                # Adjust probabilities with temperature
                adjusted = []
                for word, prob in candidates:
                    # Temperature scaling in log space
                    adjusted_prob = math.pow(prob, 1.0 / args.temperature)
                    adjusted.append((word, adjusted_prob))
                # Renormalize
                total = sum(p for _, p in adjusted)
                candidates = [(w, p / total) for w, p in adjusted]

            # Weighted random selection
            r = random.random()
            cumulative = 0.0
            next_word = candidates[0][0]  # fallback
            for word, prob in candidates:
                cumulative += prob
                if r <= cumulative:
                    next_word = word
                    break

        pred_record["selected"] = next_word
        predictions.append(pred_record)

        generated.append(next_word)
        current_word = next_word

    # Output results
    if args.json:
        result = {
            "prompt": args.prompt,
            "generated_text": " ".join(generated),
            "token_count": len(generated),
            "temperature": args.temperature,
            "predictions": predictions,
        }
        print(json_module.dumps(result, indent=2))
    elif args.show_confidence:
        # Show text with confidence annotations
        print(" ".join(generated))
        print()
        print("Prediction details:")
        print("-" * 50)
        for pred in predictions:
            conf_str = f"{pred['confidence']:.2f}" if not pred.get('is_unknown') else "N/A"
            result = pred.get('selected') or pred.get('result', '?')
            print(f"  {pred['from_word']:15} -> {result:15} (conf={conf_str})")
    else:
        # Simple text output
        print(" ".join(generated))


def _run_demo(trainer: IncrementalTrainer) -> None:
    """
    Interactive demo showcasing CognitiveAgent capabilities.

    Demonstrates:
    - Model statistics
    - Word associations with IDF weighting
    - Comparison of IDF vs raw weights
    - Semantic discovery examples
    """
    agent = trainer.agent
    tok = trainer.bridge.tokenizer

    # Header
    print()
    print("=" * 70)
    print("        COGNITIVE AGENT DEMO - Semantic Knowledge Graph")
    print("=" * 70)
    print()

    # Model statistics
    print("MODEL STATISTICS")
    print("-" * 40)
    status = trainer.status()
    print(f"  Documents trained:  {status['total_documents_trained']}")
    print(f"  Vocabulary size:    {status['vocabulary_size']}")
    print(f"  Total docs (IDF):   {tok._total_docs}")
    print(f"  Last training:      {status['last_training'][:10] if status['last_training'] else 'Never'}")
    print()

    # Demo queries
    demo_words = ["neural", "machine", "data", "algorithm", "learning"]

    print("WORD ASSOCIATIONS (IDF-weighted)")
    print("-" * 40)
    print("IDF (Inverse Document Frequency) down-weights common words,")
    print("highlighting semantically meaningful associations.")
    print()

    for word in demo_words:
        associations = agent.get_associations(word, weight_type="idf", top_k=5)
        if associations:
            top_assocs = ", ".join(f"{a.word}({a.weight:.2f})" for a in associations[:3])
            print(f"  {word:<12} -> {top_assocs}")
        else:
            print(f"  {word:<12} -> (not in vocabulary)")
    print()

    # IDF vs Raw comparison
    print("IDF vs RAW WEIGHT COMPARISON")
    print("-" * 40)
    print("Raw weights reflect co-occurrence frequency.")
    print("IDF weights penalize common terms, surfacing rare connections.")
    print()

    comparison_word = "neural"
    idf_assocs = agent.get_associations(comparison_word, weight_type="idf", top_k=5)
    raw_assocs = agent.get_associations(comparison_word, weight_type="raw", top_k=5)

    if idf_assocs and raw_assocs:
        print(f"  Associations for '{comparison_word}':")
        print()
        print(f"  {'IDF-weighted':<25} {'Raw co-occurrence':<25}")
        print(f"  {'-'*23:<25} {'-'*23:<25}")
        for i in range(min(5, len(idf_assocs), len(raw_assocs))):
            idf_item = f"{idf_assocs[i].word} ({idf_assocs[i].weight:.3f})"
            raw_item = f"{raw_assocs[i].word} ({raw_assocs[i].weight:.3f})"
            print(f"  {idf_item:<25} {raw_item:<25}")
    print()

    # Interesting discoveries
    print("SEMANTIC DISCOVERIES")
    print("-" * 40)
    print("Words that bridge different domains (high connectivity):")
    print()

    # Find words with many strong associations
    bridge_words = []
    for word in list(tok.vocab)[:500]:  # Sample vocabulary
        assocs = agent.get_associations(word, weight_type="idf", top_k=10)
        if assocs:
            avg_weight = sum(a.weight for a in assocs) / len(assocs)
            if avg_weight > 0.3 and len(assocs) >= 5:
                bridge_words.append((word, len(assocs), avg_weight))

    bridge_words.sort(key=lambda x: -x[2])
    for word, num_assocs, avg_weight in bridge_words[:8]:
        print(f"  {word:<15} {num_assocs:>3} associations, avg weight: {avg_weight:.3f}")

    print()
    print("=" * 70)
    print("Try: python -m cortical.cognitive query <word> --top-k 10")
    print("=" * 70)
    print()


def _run_index_code(trainer: 'IncrementalTrainer', args) -> None:
    """
    Index Python code structure into the cognitive graph.

    Creates atoms for FILE, CLASS, FUNCTION entities and links for
    DEFINES, CONTAINS, CALLS, INHERITANCE relationships.

    Args:
        trainer: The IncrementalTrainer with loaded model
        args: CLI arguments with path, exclude, quiet, json options
    """
    import json as json_module

    from cortical.cognitive.code_bridge import CodeBridge

    # Use the trainer's agent's graph
    graph = trainer.agent.graph
    bridge = CodeBridge(graph)

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    if not args.quiet:
        print(f"Indexing Python code in: {path}")
        if args.exclude:
            print(f"Excluding: {', '.join(args.exclude)}")
        print()

    # Progress callback
    def progress(current: int, total: int) -> None:
        if not args.quiet and total > 0:
            pct = current * 100 // total
            if current % 50 == 0 or current == total:
                print(f"  Indexed {current}/{total} files ({pct}%)")

    # Index
    if path.is_file():
        stats = bridge.index_file(path)
    else:
        stats = bridge.index_directory(
            path,
            exclude=args.exclude,
            progress_callback=progress
        )

    # Create REFERS_TO links if requested
    refers_to_stats = None
    if getattr(args, 'link_text', False):
        if not args.quiet:
            print("\nCreating REFERS_TO links (bridging code to text)...")
        refers_to_stats = bridge.create_refers_to_links()
        stats.refers_to_links = refers_to_stats.refers_to_links

    # Save the updated graph
    trainer.save()

    # Output results
    if args.json:
        result = {
            "path": str(path),
            "files": stats.files,
            "classes": stats.classes,
            "functions": stats.functions,
            "calls_links": stats.calls_links,
            "inheritance_links": stats.inheritance_links,
            "defines_links": stats.defines_links,
            "contains_links": stats.contains_links,
            "refers_to_links": stats.refers_to_links,
            "parse_errors": stats.parse_errors,
            "elapsed_seconds": round(stats.elapsed_seconds, 2),
        }
        print(json_module.dumps(result, indent=2))
    else:
        if not args.quiet:
            print()
        print("Code indexing complete:")
        print(f"  Files indexed:    {stats.files}")
        print(f"  Classes:          {stats.classes}")
        print(f"  Functions:        {stats.functions}")
        print(f"  CALLS links:      {stats.calls_links}")
        print(f"  INHERITANCE:      {stats.inheritance_links}")
        if stats.refers_to_links > 0:
            print(f"  REFERS_TO links:  {stats.refers_to_links}")
        print(f"  Parse errors:     {stats.parse_errors}")
        print(f"  Elapsed time:     {stats.elapsed_seconds:.2f}s")


def _run_ask(trainer: 'IncrementalTrainer', args) -> None:
    """
    Ask a natural language question about the codebase.

    Uses NLQuery to parse the question, gather knowledge from
    trained vocabulary and indexed code, and generate a response.

    Args:
        trainer: The IncrementalTrainer with loaded model
        args: CLI arguments with question, verbose options
    """
    from cortical.cognitive.nl_query import NLQuery

    # Create NLQuery with the trainer's agent
    nl = NLQuery(trainer.agent)

    # Get the answer
    response = nl.ask(args.question)

    # Show verbose info if requested
    if getattr(args, 'verbose', False):
        intent = nl.parse_intent(args.question)
        print(f"Question type: {intent.question_type}")
        print(f"Concepts: {', '.join(intent.concepts)}")
        print(f"Strategy: {', '.join(intent.query_strategy)}")
        print("-" * 40)

    # Print the response
    print(response)


# =============================================================================
# IMPORTANT: Do NOT add `if __name__ == "__main__"` here!
#
# This module should be run via: python -m cortical.cognitive
# which uses cortical/cognitive/__main__.py as the entry point.
#
# Why? When Python runs `python -m cortical.cognitive.training`:
#   1. It imports cortical.cognitive package first (__init__.py)
#   2. If __init__.py imports from this module, classes are created
#   3. Then this module runs as __main__, creating DIFFERENT class objects
#   4. DI containers use class objects as dict keys, so lookup fails
#
# The __main__.py pattern avoids this by being a separate file that:
#   - Is never imported by __init__.py
#   - Imports this module's classes only when needed for CLI
#   - Preserves class identity throughout the application
#
# For backward compatibility, main() and run_cli() are still available
# for direct programmatic use, but CLI should use __main__.py.
# =============================================================================
