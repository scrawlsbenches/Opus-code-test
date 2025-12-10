#!/usr/bin/env python3
"""
Index the Cortical Text Processor codebase for dog-fooding.

This script indexes all Python files and documentation to enable
semantic search over the codebase using the Cortical Text Processor itself.

Supports incremental indexing to only re-index changed files.

Usage:
    python scripts/index_codebase.py [--output corpus_dev.pkl]
    python scripts/index_codebase.py --incremental  # Only index changes
    python scripts/index_codebase.py --status       # Show what would change
    python scripts/index_codebase.py --force        # Force full rebuild
    python scripts/index_codebase.py --log indexer.log  # Log to file
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor


# Manifest file version for compatibility checking
MANIFEST_VERSION = "1.0"

# Default timeout in seconds (0 = no timeout)
DEFAULT_TIMEOUT = 300  # 5 minutes


# =============================================================================
# Progress Tracking System
# =============================================================================

@dataclass
class PhaseStats:
    """Statistics for a single phase of indexing."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    items_total: int = 0
    items_processed: int = 0
    status: str = "pending"  # pending, running, completed, failed

    @property
    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    @property
    def progress_pct(self) -> float:
        if self.items_total == 0:
            return 0.0
        return (self.items_processed / self.items_total) * 100


class ProgressTracker:
    """
    Tracks progress through indexing phases with timing and logging.

    Provides:
    - Per-phase timing
    - Per-file progress within phases
    - Log file output
    - Console progress updates
    - Summary statistics
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        verbose: bool = False,
        quiet: bool = False
    ):
        self.start_time = time.time()
        self.phases: Dict[str, PhaseStats] = {}
        self.current_phase: Optional[str] = None
        self.verbose = verbose
        self.quiet = quiet
        self.warnings: List[str] = []
        self.errors: List[str] = []

        # Set up logging
        self.logger = logging.getLogger("indexer")
        self.logger.setLevel(logging.DEBUG)

        # Console handler (INFO level unless verbose)
        if not quiet:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
            console_format = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # File handler (DEBUG level - captures everything)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None

    def log(self, message: str, level: str = "info"):
        """Log a message at the specified level."""
        getattr(self.logger, level)(message)

    def start_phase(self, name: str, total_items: int = 0):
        """Start a new phase of indexing."""
        self.current_phase = name
        self.phases[name] = PhaseStats(
            name=name,
            start_time=time.time(),
            items_total=total_items,
            status="running"
        )
        self.log(f"\n[PHASE] {name}", "info")
        if total_items > 0:
            self.log(f"  Items to process: {total_items}", "debug")

    def end_phase(self, name: Optional[str] = None, status: str = "completed"):
        """End a phase and record timing."""
        phase_name = name or self.current_phase
        if phase_name and phase_name in self.phases:
            phase = self.phases[phase_name]
            phase.end_time = time.time()
            phase.status = status
            duration = phase.duration

            status_symbol = "✓" if status == "completed" else "✗"
            self.log(f"  {status_symbol} {phase_name} completed in {duration:.2f}s", "info")

            if self.current_phase == phase_name:
                self.current_phase = None

    def update_progress(self, items_processed: int, item_name: Optional[str] = None):
        """Update progress within the current phase."""
        if self.current_phase and self.current_phase in self.phases:
            phase = self.phases[self.current_phase]
            phase.items_processed = items_processed

            if phase.items_total > 0:
                pct = phase.progress_pct
                if item_name:
                    self.log(
                        f"  [{items_processed}/{phase.items_total}] {pct:.0f}% - {item_name}",
                        "debug"
                    )
                # Show progress at 25%, 50%, 75% milestones
                if items_processed in [
                    phase.items_total // 4,
                    phase.items_total // 2,
                    (phase.items_total * 3) // 4
                ]:
                    self.log(f"  Progress: {pct:.0f}% ({items_processed}/{phase.items_total})", "info")

    def warn(self, message: str):
        """Log a warning."""
        self.warnings.append(message)
        self.log(f"  WARNING: {message}", "warning")

    def error(self, message: str):
        """Log an error."""
        self.errors.append(message)
        self.log(f"  ERROR: {message}", "error")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all phases."""
        total_duration = time.time() - self.start_time
        return {
            "total_duration": total_duration,
            "phases": {
                name: {
                    "duration": phase.duration,
                    "items_processed": phase.items_processed,
                    "items_total": phase.items_total,
                    "status": phase.status
                }
                for name, phase in self.phases.items()
            },
            "warnings": len(self.warnings),
            "errors": len(self.errors)
        }

    def print_summary(self):
        """Print a summary of the indexing run."""
        total_duration = time.time() - self.start_time

        self.log("\n" + "=" * 50, "info")
        self.log("INDEXING SUMMARY", "info")
        self.log("=" * 50, "info")

        self.log(f"\nTotal time: {total_duration:.2f}s", "info")

        self.log("\nPhase breakdown:", "info")
        for name, phase in self.phases.items():
            status_symbol = "✓" if phase.status == "completed" else "✗"
            items_str = ""
            if phase.items_total > 0:
                items_str = f" ({phase.items_processed}/{phase.items_total} items)"
            self.log(f"  {status_symbol} {name}: {phase.duration:.2f}s{items_str}", "info")

        if self.warnings:
            self.log(f"\nWarnings: {len(self.warnings)}", "warning")
            for w in self.warnings[:5]:
                self.log(f"  - {w}", "warning")
            if len(self.warnings) > 5:
                self.log(f"  ... and {len(self.warnings) - 5} more", "warning")

        if self.errors:
            self.log(f"\nErrors: {len(self.errors)}", "error")
            for e in self.errors[:5]:
                self.log(f"  - {e}", "error")

        if self.log_file:
            self.log(f"\nFull log written to: {self.log_file}", "info")


# =============================================================================
# Timeout Handler
# =============================================================================

class TimeoutError(Exception):
    """Raised when indexing exceeds the timeout."""
    pass


@contextmanager
def timeout_handler(seconds: int, tracker: Optional[ProgressTracker] = None):
    """
    Context manager for timeout handling.

    Args:
        seconds: Timeout in seconds (0 = no timeout)
        tracker: Optional progress tracker for logging
    """
    if seconds <= 0:
        yield
        return

    def handler(signum, frame):
        msg = f"Indexing timed out after {seconds} seconds"
        if tracker:
            tracker.error(msg)
            tracker.print_summary()
        raise TimeoutError(msg)

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# Manifest Operations
# =============================================================================

def get_manifest_path(corpus_path: Path) -> Path:
    """Get the manifest file path based on corpus path."""
    return corpus_path.with_suffix('.manifest.json')


def load_manifest(
    manifest_path: Path,
    tracker: Optional[ProgressTracker] = None
) -> Optional[Dict[str, Any]]:
    """
    Load the manifest file if it exists.

    Args:
        manifest_path: Path to the manifest file
        tracker: Optional progress tracker for logging

    Returns:
        Manifest dict if found and valid, None otherwise
    """
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Check version compatibility
        if manifest.get('version') != MANIFEST_VERSION:
            if tracker:
                tracker.warn(f"Manifest version mismatch (expected {MANIFEST_VERSION})")
            return None

        return manifest
    except (json.JSONDecodeError, IOError) as e:
        if tracker:
            tracker.warn(f"Could not load manifest: {e}")
        return None


def save_manifest(
    manifest_path: Path,
    files: Dict[str, float],
    corpus_path: str,
    stats: Dict[str, Any],
    tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Save the manifest file with current file state.

    Args:
        manifest_path: Path to save the manifest
        files: Dict mapping file paths to modification times
        corpus_path: Path to the corpus file
        stats: Statistics about the indexed corpus
        tracker: Optional progress tracker for logging
    """
    manifest = {
        'version': MANIFEST_VERSION,
        'corpus_path': str(corpus_path),
        'indexed_at': datetime.now().isoformat(),
        'files': files,
        'stats': stats,
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    if tracker:
        tracker.log(f"  Manifest saved to {manifest_path.name}", "debug")


# =============================================================================
# File Operations
# =============================================================================

def get_file_mtime(file_path: Path) -> float:
    """Get the modification time of a file."""
    return file_path.stat().st_mtime


def get_file_changes(
    manifest: Dict[str, Any],
    current_files: List[Path],
    base_path: Path
) -> Tuple[List[Path], List[Path], List[str]]:
    """
    Compare current files to manifest and detect changes.

    Args:
        manifest: Previously saved manifest
        current_files: List of current file paths
        base_path: Base path for relative path calculation

    Returns:
        Tuple of (added_files, modified_files, deleted_doc_ids)
    """
    old_files = manifest.get('files', {})

    added = []
    modified = []
    deleted_ids = []

    # Build set of current relative paths
    current_rel_paths = {}
    for file_path in current_files:
        rel_path = str(file_path.relative_to(base_path))
        current_rel_paths[rel_path] = file_path

    # Check for added and modified files
    for rel_path, file_path in current_rel_paths.items():
        if rel_path not in old_files:
            added.append(file_path)
        else:
            old_mtime = old_files[rel_path]
            current_mtime = get_file_mtime(file_path)
            if current_mtime > old_mtime:
                modified.append(file_path)

    # Check for deleted files
    for rel_path in old_files:
        if rel_path not in current_rel_paths:
            deleted_ids.append(rel_path)  # doc_id is the relative path

    return added, modified, deleted_ids


def get_python_files(base_path: Path) -> list:
    """Get all Python files in cortical/ and tests/ directories."""
    files = []
    for directory in ['cortical', 'tests']:
        dir_path = base_path / directory
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                if not py_file.name.startswith('__'):
                    files.append(py_file)
    return sorted(files)


def get_doc_files(base_path: Path) -> list:
    """Get documentation files from root and docs/ directory."""
    # Root documentation files
    root_docs = ['CLAUDE.md', 'TASK_LIST.md', 'README.md', 'KNOWLEDGE_TRANSFER.md']
    files = []
    for doc in root_docs:
        doc_path = base_path / doc
        if doc_path.exists():
            files.append(doc_path)

    # Intelligence documentation in docs/
    docs_dir = base_path / 'docs'
    if docs_dir.exists():
        for md_file in docs_dir.glob('*.md'):
            files.append(md_file)

    return files


def create_doc_id(file_path: Path, base_path: Path) -> str:
    """Create a document ID from file path."""
    rel_path = file_path.relative_to(base_path)
    return str(rel_path)


# =============================================================================
# Indexing Operations
# =============================================================================

def index_file(
    processor: CorticalTextProcessor,
    file_path: Path,
    base_path: Path,
    tracker: Optional[ProgressTracker] = None
) -> Optional[dict]:
    """Index a single file with line number metadata."""
    doc_id = create_doc_id(file_path, base_path)

    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        if tracker:
            tracker.warn(f"Could not read {doc_id}: {e}")
        return None

    # Create metadata with file info
    metadata = {
        'file_path': str(file_path),
        'relative_path': doc_id,
        'file_type': file_path.suffix,
        'line_count': content.count('\n') + 1,
        'mtime': get_file_mtime(file_path),
    }

    # For Python files, extract additional metadata
    if file_path.suffix == '.py':
        metadata['language'] = 'python'
        # Count functions and classes
        metadata['function_count'] = content.count('\ndef ')
        metadata['class_count'] = content.count('\nclass ')

    processor.process_document(doc_id, content, metadata=metadata)
    return metadata


def show_status(
    added: List[Path],
    modified: List[Path],
    deleted: List[str],
    base_path: Path,
    tracker: ProgressTracker
) -> None:
    """Display what would change without actually indexing."""
    tracker.log("\n" + "=" * 50)
    tracker.log("STATUS: Changes detected (no indexing performed)")
    tracker.log("=" * 50)

    if not added and not modified and not deleted:
        tracker.log("\nNo changes detected. Corpus is up to date.")
        return

    if added:
        tracker.log(f"\n  Added ({len(added)} files):")
        for f in added[:10]:
            tracker.log(f"    + {create_doc_id(f, base_path)}")
        if len(added) > 10:
            tracker.log(f"    ... and {len(added) - 10} more")

    if modified:
        tracker.log(f"\n  Modified ({len(modified)} files):")
        for f in modified[:10]:
            tracker.log(f"    ~ {create_doc_id(f, base_path)}")
        if len(modified) > 10:
            tracker.log(f"    ... and {len(modified) - 10} more")

    if deleted:
        tracker.log(f"\n  Deleted ({len(deleted)} files):")
        for doc_id in deleted[:10]:
            tracker.log(f"    - {doc_id}")
        if len(deleted) > 10:
            tracker.log(f"    ... and {len(deleted) - 10} more")

    total = len(added) + len(modified) + len(deleted)
    tracker.log(f"\nTotal: {total} files would be updated.")
    tracker.log("Run with --incremental to apply changes.")


def full_index(
    processor: CorticalTextProcessor,
    all_files: List[Path],
    base_path: Path,
    tracker: ProgressTracker
) -> Tuple[int, int, Dict[str, float]]:
    """
    Perform a full index of all files.

    Returns:
        Tuple of (indexed_count, total_lines, file_mtimes)
    """
    tracker.start_phase("Indexing files", len(all_files))

    indexed = 0
    total_lines = 0
    file_mtimes = {}

    for i, file_path in enumerate(all_files, 1):
        doc_id = create_doc_id(file_path, base_path)
        tracker.update_progress(i, doc_id)

        metadata = index_file(processor, file_path, base_path, tracker)
        if metadata:
            indexed += 1
            total_lines += metadata.get('line_count', 0)
            file_mtimes[doc_id] = metadata.get('mtime', 0)

    tracker.end_phase("Indexing files")
    tracker.log(f"  Indexed {indexed} files ({total_lines:,} total lines)")

    return indexed, total_lines, file_mtimes


def incremental_index(
    processor: CorticalTextProcessor,
    added: List[Path],
    modified: List[Path],
    deleted: List[str],
    base_path: Path,
    tracker: ProgressTracker
) -> Tuple[int, int, int, int]:
    """
    Perform an incremental index updating only changed files.

    Returns:
        Tuple of (added_count, modified_count, deleted_count, total_lines_updated)
    """
    total_items = len(added) + len(modified) + len(deleted)
    tracker.start_phase("Incremental update", total_items)

    added_count = 0
    modified_count = 0
    deleted_count = 0
    total_lines = 0
    processed = 0

    # Remove deleted documents
    if deleted:
        tracker.log(f"  Removing {len(deleted)} deleted files...")
        for doc_id in deleted:
            result = processor.remove_document(doc_id, verbose=False)
            if result['found']:
                deleted_count += 1
            processed += 1
            tracker.update_progress(processed, f"Deleted: {doc_id}")

    # Update modified documents (remove old, add new)
    if modified:
        tracker.log(f"  Updating {len(modified)} modified files...")
        for file_path in modified:
            doc_id = create_doc_id(file_path, base_path)
            # Remove old version
            processor.remove_document(doc_id, verbose=False)
            # Add new version
            metadata = index_file(processor, file_path, base_path, tracker)
            if metadata:
                modified_count += 1
                total_lines += metadata.get('line_count', 0)
            processed += 1
            tracker.update_progress(processed, f"Modified: {doc_id}")

    # Add new documents
    if added:
        tracker.log(f"  Indexing {len(added)} new files...")
        for file_path in added:
            doc_id = create_doc_id(file_path, base_path)
            metadata = index_file(processor, file_path, base_path, tracker)
            if metadata:
                added_count += 1
                total_lines += metadata.get('line_count', 0)
            processed += 1
            tracker.update_progress(processed, f"Added: {doc_id}")

    tracker.end_phase("Incremental update")
    tracker.log(f"  Added: {added_count}, Modified: {modified_count}, Deleted: {deleted_count}")
    tracker.log(f"  Lines processed: {total_lines:,}")

    return added_count, modified_count, deleted_count, total_lines


def compute_analysis(
    processor: CorticalTextProcessor,
    tracker: ProgressTracker,
    fast_mode: bool = True
) -> None:
    """
    Run all analysis computations with progress tracking.

    Args:
        processor: The text processor
        tracker: Progress tracker for logging
        fast_mode: If True, use faster but simpler analysis (skips slow bigram connections).
                   If False, use full semantic PageRank and hybrid connections.
    """
    if fast_mode:
        # Fast mode: Skip expensive operations
        # - Use standard PageRank (not semantic/hierarchical)
        # - Skip bigram connections (O(n²) on large corpora)
        # - Skip concept cluster connections
        # Completes in seconds for any size corpus
        tracker.start_phase("Computing analysis (fast mode)")

        # Manual fast computation - skip compute_all() to avoid bigram connections
        tracker.log("  Propagating activation...", "debug")
        processor.propagate_activation(verbose=False)

        tracker.log("  Computing PageRank...", "debug")
        processor.compute_importance(verbose=False)

        tracker.log("  Computing TF-IDF...", "debug")
        processor.compute_tfidf(verbose=False)

        tracker.log("  Computing document connections...", "debug")
        processor.compute_document_connections(verbose=False)

        # Skip bigram connections (too slow with large corpora)
        # Skip concept clusters (not needed for basic search)

        tracker.end_phase("Computing analysis (fast mode)")
    else:
        # Full mode: semantic PageRank, hybrid connections
        # More accurate but can take minutes for large codebases
        tracker.start_phase("Computing analysis (full mode - may take several minutes)")
        processor.compute_all(
            build_concepts=True,
            pagerank_method='semantic',
            connection_strategy='hybrid',
            verbose=False
        )
        tracker.end_phase("Computing analysis (full mode - may take several minutes)")

        tracker.start_phase("Extracting semantic relations")
        processor.extract_corpus_semantics(
            use_pattern_extraction=True,
            verbose=False
        )
        tracker.end_phase("Extracting semantic relations")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Index the codebase for semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_codebase.py                   # Full rebuild
  python scripts/index_codebase.py --incremental    # Update changed files only
  python scripts/index_codebase.py --status         # Show what would change
  python scripts/index_codebase.py --force          # Force full rebuild
  python scripts/index_codebase.py --log index.log  # Log to file
  python scripts/index_codebase.py --timeout 60     # Timeout after 60s
        """
    )
    parser.add_argument('--output', '-o', default='corpus_dev.pkl',
                        help='Output file path (default: corpus_dev.pkl)')
    parser.add_argument('--incremental', '-i', action='store_true',
                        help='Only index changed files (requires existing corpus)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force full rebuild even if manifest exists')
    parser.add_argument('--status', '-s', action='store_true',
                        help='Show what would change without indexing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output (per-file progress)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress console output (still writes to log)')
    parser.add_argument('--log', '-l', type=str, default=None,
                        help='Log file path (writes detailed log)')
    parser.add_argument('--timeout', '-t', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout in seconds (0=none, default={DEFAULT_TIMEOUT})')
    parser.add_argument('--full-analysis', action='store_true',
                        help='Use full semantic analysis (slower but more accurate)')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    output_path = base_path / args.output
    manifest_path = get_manifest_path(output_path)

    # Set up log file path
    log_path = None
    if args.log:
        log_path = args.log if os.path.isabs(args.log) else str(base_path / args.log)

    # Initialize progress tracker
    tracker = ProgressTracker(
        log_file=log_path,
        verbose=args.verbose,
        quiet=args.quiet
    )

    tracker.log("Cortical Text Processor - Codebase Indexer")
    tracker.log("=" * 50)
    tracker.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.timeout > 0:
        tracker.log(f"Timeout: {args.timeout}s")

    try:
        with timeout_handler(args.timeout, tracker):
            run_indexer(args, base_path, output_path, manifest_path, tracker)
    except TimeoutError:
        tracker.log("\nIndexing was terminated due to timeout.", "error")
        sys.exit(1)
    except KeyboardInterrupt:
        tracker.log("\nIndexing was interrupted by user.", "warning")
        tracker.print_summary()
        sys.exit(1)
    except Exception as e:
        tracker.error(f"Unexpected error: {e}")
        tracker.print_summary()
        raise


def run_indexer(
    args,
    base_path: Path,
    output_path: Path,
    manifest_path: Path,
    tracker: ProgressTracker
) -> None:
    """Main indexing logic."""

    # Get files to index
    tracker.start_phase("Discovering files")
    python_files = get_python_files(base_path)
    doc_files = get_doc_files(base_path)
    all_files = python_files + doc_files
    tracker.end_phase("Discovering files")

    tracker.log(f"\nFound {len(python_files)} Python files and {len(doc_files)} documentation files")

    # Load existing manifest if doing incremental update
    manifest = None
    if (args.incremental or args.status) and not args.force:
        if manifest_path.exists():
            tracker.log(f"\nLoading manifest from {manifest_path.name}...")
            manifest = load_manifest(manifest_path, tracker)
        else:
            tracker.log("\nNo manifest found - will perform full index")

    # Detect changes if we have a manifest
    added, modified, deleted = [], [], []
    if manifest:
        added, modified, deleted = get_file_changes(manifest, all_files, base_path)
        tracker.log(f"\nChanges detected:")
        tracker.log(f"  Added: {len(added)}, Modified: {len(modified)}, Deleted: {len(deleted)}")

    # Status mode - just show what would change
    if args.status:
        show_status(added, modified, deleted, base_path, tracker)
        return

    # Determine if we need to do work
    if manifest and not (added or modified or deleted):
        tracker.log("\nNo changes detected. Corpus is up to date.")
        tracker.log("Use --force to rebuild anyway.")
        return

    # Initialize or load processor
    if args.incremental and manifest and output_path.exists():
        tracker.start_phase("Loading existing corpus")
        try:
            processor = CorticalTextProcessor.load(str(output_path))
            tracker.log(f"  Loaded {len(processor.documents)} documents")
            tracker.end_phase("Loading existing corpus")
        except Exception as e:
            tracker.warn(f"Error loading corpus: {e}")
            tracker.log("  Falling back to full rebuild...")
            tracker.end_phase("Loading existing corpus", status="failed")
            processor = CorticalTextProcessor()
            added, modified, deleted = all_files, [], []
    else:
        processor = CorticalTextProcessor()
        # Full index - treat all files as "added"
        added = all_files
        modified = []
        deleted = []

    # Perform indexing
    if args.incremental and manifest:
        incremental_index(processor, added, modified, deleted, base_path, tracker)
    else:
        full_index(processor, all_files, base_path, tracker)

    # Compute analysis
    fast_mode = not args.full_analysis
    compute_analysis(processor, tracker, fast_mode=fast_mode)

    # Print corpus statistics
    tracker.log("\nCorpus Statistics:")
    tracker.log(f"  Documents: {len(processor.documents)}")
    tracker.log(f"  Tokens (Layer 0): {processor.layers[0].column_count()}")
    tracker.log(f"  Bigrams (Layer 1): {processor.layers[1].column_count()}")
    tracker.log(f"  Concepts (Layer 2): {processor.layers[2].column_count()}")
    tracker.log(f"  Semantic relations: {len(processor.semantic_relations)}")

    # Save corpus
    tracker.start_phase("Saving corpus")
    processor.save(str(output_path))
    file_size = output_path.stat().st_size / 1024
    tracker.log(f"  Saved to {output_path.name} ({file_size:.1f} KB)")
    tracker.end_phase("Saving corpus")

    # Build file_mtimes for manifest
    file_mtimes = {}
    for file_path in all_files:
        rel_path = create_doc_id(file_path, base_path)
        if rel_path in processor.documents:
            file_mtimes[rel_path] = get_file_mtime(file_path)

    # Save manifest
    stats = {
        'documents': len(processor.documents),
        'tokens': processor.layers[0].column_count(),
        'bigrams': processor.layers[1].column_count(),
        'concepts': processor.layers[2].column_count(),
        'semantic_relations': len(processor.semantic_relations),
    }
    save_manifest(manifest_path, file_mtimes, str(output_path), stats, tracker)

    # Print summary
    tracker.print_summary()

    tracker.log("\nDone! Use search_codebase.py to query the indexed corpus.")


if __name__ == '__main__':
    main()
