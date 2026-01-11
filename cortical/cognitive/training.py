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


# =============================================================================
# Training Manifest
# =============================================================================


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

    def save(self, path: Path, filesystem: Optional[FileSystem] = None) -> None:
        """Save manifest to JSON file."""
        data = {
            "model_version": self.model_version,
            "last_training": self.last_training,
            "total_documents": self.total_documents,
            "vocabulary_size": self.vocabulary_size,
            "documents": {
                k: v.to_dict() for k, v in self.documents.items()
            },
        }
        content = json.dumps(data, indent=2)
        if filesystem:
            filesystem.write_text(path, content)
        else:
            path.write_text(content)

    @classmethod
    def load(cls, path: Path, filesystem: Optional[FileSystem] = None) -> 'TrainingManifest':
        """Load manifest from JSON file."""
        if filesystem:
            if not filesystem.exists(path):
                return cls()
            data = json.loads(filesystem.read_text(path))
        else:
            if not path.exists():
                return cls()
            data = json.loads(path.read_text())

        manifest = cls(
            last_training=data.get("last_training"),
            total_documents=data.get("total_documents", 0),
            vocabulary_size=data.get("vocabulary_size", 0),
            model_version=data.get("model_version", "1.0"),
        )

        for path_key, doc_data in data.get("documents", {}).items():
            manifest.documents[path_key] = TrainedDocument.from_dict(doc_data)

        return manifest


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
        model_dir: str | Path = "models/cognitive_agent",
        filesystem: Optional[FileSystem] = None,
    ):
        """
        Initialize trainer.

        Args:
            agent: CognitiveAgent to train
            model_dir: Directory for model persistence
            filesystem: Optional FileSystem for I/O (defaults to RealFileSystem)
        """
        self.agent = agent
        self.model_dir = Path(model_dir)

        # Use provided filesystem or create real one
        if filesystem is None:
            self.filesystem: FileSystem = RealFileSystem(self.model_dir)
        else:
            self.filesystem = filesystem

        # Create model directory
        self.filesystem.mkdir(self.model_dir, parents=True, exist_ok=True)

        # Load or create manifest
        self.manifest_path = self.model_dir / "training_manifest.json"
        self.manifest = TrainingManifest.load(self.manifest_path, self.filesystem)

        # Initialize bridge
        self.bridge = TextToAtomsBridge(agent.graph)

        # Load existing tokenizer if available
        tokenizer_path = self.model_dir / "tokenizer.json"
        if self.filesystem.exists(tokenizer_path):
            from cortical.cognitive.text_bridge import BPETokenizer
            content = self.filesystem.read_text(tokenizer_path)
            self.bridge.tokenizer = BPETokenizer.from_dict(json.loads(content))

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
    ) -> TrainingStats:
        """
        Train on documents in a directory (incrementally).

        Only processes documents that are new or modified since last training.

        Args:
            directory: Directory containing text files
            pattern: Glob pattern for files (default: *.txt)
            recursive: Search subdirectories (default: True)
            show_progress: Show progress bar (default: True)
            force_retrain: Ignore manifest and retrain all (default: False)

        Returns:
            TrainingStats with details of what was trained
        """
        import time
        start_time = time.time()

        stats = TrainingStats()
        directory = Path(directory)

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

        if show_progress:
            with ProgressReporter(len(texts), desc="Training") as progress:
                for path, text in zip(paths, texts):
                    self.bridge.feed_text(text, doc_id=path)

                    # Update manifest
                    content_hash = compute_content_hash(text)
                    word_count = len(self.bridge.tokenizer.tokenize(text))
                    self.manifest.add_document(path, content_hash, word_count)

                    progress.update(1)
        else:
            for path, text in zip(paths, texts):
                self.bridge.feed_text(text, doc_id=path)
                content_hash = compute_content_hash(text)
                word_count = len(self.bridge.tokenizer.tokenize(text))
                self.manifest.add_document(path, content_hash, word_count)

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
        # Save tokenizer using filesystem
        tokenizer_path = self.model_dir / "tokenizer.json"
        tokenizer_content = json.dumps(self.bridge.tokenizer.to_dict(), indent=2)
        self.filesystem.write_text(tokenizer_path, tokenizer_content)

        # Save bridge graph data
        bridge_dir = self.model_dir / "bridge"
        self.filesystem.mkdir(bridge_dir, parents=True, exist_ok=True)

        # Serialize graph to bridge directory
        graph_data = {
            "atoms": [],
            "stats": self.bridge.get_statistics(),
        }
        for atom in self.bridge.graph._storage.all_atoms():
            atom_data = {
                "id": atom.id,
                "name": atom.name,
                "atom_type": atom.atom_type.name,
                "tv_strength": atom.tv.strength,
                "tv_confidence": atom.tv.confidence,
                "sti": atom.sti,
                "lti": atom.lti,
                "outgoing": atom.outgoing,
            }
            graph_data["atoms"].append(atom_data)

        self.filesystem.write_text(
            bridge_dir / "graph.json",
            json.dumps(graph_data, indent=2)
        )

        # Save manifest
        self.manifest.save(self.manifest_path, self.filesystem)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        agent: Optional['CognitiveAgent'] = None,
        filesystem: Optional[FileSystem] = None,
    ) -> 'IncrementalTrainer':
        """
        Load a previously trained model.

        Args:
            model_dir: Directory containing saved model
            agent: CognitiveAgent to load into (creates new if None)
            filesystem: Optional FileSystem for I/O

        Returns:
            IncrementalTrainer with loaded state
        """
        from cortical.cognitive.graph import CognitiveAgent

        model_dir = Path(model_dir)

        if agent is None:
            agent = CognitiveAgent()

        trainer = cls(agent, model_dir, filesystem=filesystem)

        # Load bridge with graph if exists
        bridge_dir = model_dir / "bridge"
        if trainer.filesystem.exists(bridge_dir):
            graph_path = bridge_dir / "graph.json"
            if trainer.filesystem.exists(graph_path):
                trainer.bridge = TextToAtomsBridge.load(bridge_dir, agent.graph)

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
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Import here to avoid circular imports at module level
    from cortical.cognitive.graph import CognitiveAgent

    agent = CognitiveAgent()
    trainer = IncrementalTrainer(agent, args.model_dir)

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

    if args.files:
        stats = trainer.train_files(
            args.files,
            base_dir=args.directory,
            show_progress=not args.quiet,
        )
    else:
        stats = trainer.train_directory(
            args.directory,
            pattern=args.pattern,
            show_progress=not args.quiet,
            force_retrain=args.force,
        )

    # Print final status
    if not args.quiet:
        print(f"\nModel saved to: {trainer.model_dir}")


if __name__ == "__main__":
    main()
