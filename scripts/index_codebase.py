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

Background Full Analysis (for long-running environments):
    python scripts/index_codebase.py --full-analysis  # Start in background
    python scripts/index_codebase.py --full-analysis  # Check progress (run again)
    python scripts/index_codebase.py --full-analysis --foreground  # Run synchronously

Incremental Full Analysis (resumable, for short-lived processes):
    python scripts/index_codebase.py --full-analysis --batch  # Process one batch
    python scripts/index_codebase.py --full-analysis --batch  # Continue (run again)
    python scripts/index_codebase.py --full-analysis --batch --batch-size 10  # Smaller batches
    python scripts/index_codebase.py --full-analysis --batch --status  # Check progress

The --full-analysis flag runs semantic PageRank and hybrid connections, which
can take several minutes. Two modes are available:

1. Background mode (default): Spawns a background process. Best for environments
   that support long-running processes. Progress file: .index_progress.json

2. Batch mode (--batch): Processes files in batches, saving after each batch.
   Run the command multiple times to complete. Best for environments with
   process timeouts. Progress file: .index_incremental_progress.json
"""

import argparse
import json
import logging
import os
import platform
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
from cortical.chunk_index import (
    ChunkWriter, ChunkLoader, ChunkCompactor,
    get_changes_from_manifest as get_chunk_changes
)


def create_code_processor() -> CorticalTextProcessor:
    """
    Create a CorticalTextProcessor configured for code indexing.

    Enables split_identifiers so that:
    - getUserCredentials → ['getusercredentials', 'get', 'user', 'credentials']
    - auth_service → ['auth_service', 'auth', 'service']

    This dramatically improves code search - "auth" will find AuthService,
    authenticate, user_auth, etc.
    """
    tokenizer = Tokenizer(split_identifiers=True)
    return CorticalTextProcessor(tokenizer=tokenizer)


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

class IndexingTimeoutError(Exception):
    """Raised when indexing exceeds the timeout."""
    pass


# Check platform for timeout implementation
_IS_WINDOWS = platform.system() == 'Windows'


@contextmanager
def timeout_handler(seconds: int, tracker: Optional[ProgressTracker] = None):
    """
    Context manager for timeout handling.

    Uses signal.SIGALRM on Unix systems and threading.Timer on Windows.
    Note: Windows implementation cannot interrupt blocking I/O operations.

    Args:
        seconds: Timeout in seconds (0 = no timeout)
        tracker: Optional progress tracker for logging
    """
    if seconds <= 0:
        yield
        return

    if _IS_WINDOWS:
        # Windows implementation using threading.Timer
        # Note: This cannot interrupt blocking operations like file I/O
        timed_out = threading.Event()

        def timeout_callback():
            timed_out.set()
            msg = f"Indexing timed out after {seconds} seconds"
            if tracker:
                tracker.error(msg)
                tracker.error("Note: Windows timeout cannot interrupt blocking I/O")

        timer = threading.Timer(seconds, timeout_callback)
        timer.daemon = True  # Don't prevent program exit
        timer.start()

        try:
            yield
            # Check if timeout occurred (for non-blocking operations)
            if timed_out.is_set():
                if tracker:
                    tracker.print_summary()
                raise IndexingTimeoutError(f"Indexing timed out after {seconds} seconds")
        finally:
            timer.cancel()
    else:
        # Unix implementation using signal.SIGALRM
        def handler(signum, frame):
            msg = f"Indexing timed out after {seconds} seconds"
            if tracker:
                tracker.error(msg)
                tracker.print_summary()
            raise IndexingTimeoutError(msg)

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
# Background Analysis Progress Tracking
# =============================================================================

# Default progress file path
DEFAULT_PROGRESS_FILE = '.index_progress.json'


@dataclass
class BackgroundProgress:
    """Progress state for background full-analysis runs."""
    status: str = "not_started"  # not_started, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pid: Optional[int] = None
    current_phase: str = ""
    progress_percent: float = 0.0
    phases_completed: List[str] = field(default_factory=list)
    phases_pending: List[str] = field(default_factory=list)
    error: Optional[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'pid': self.pid,
            'current_phase': self.current_phase,
            'progress_percent': self.progress_percent,
            'phases_completed': self.phases_completed,
            'phases_pending': self.phases_pending,
            'error': self.error,
            'output_path': self.output_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackgroundProgress':
        return cls(
            status=data.get('status', 'not_started'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            pid=data.get('pid'),
            current_phase=data.get('current_phase', ''),
            progress_percent=data.get('progress_percent', 0.0),
            phases_completed=data.get('phases_completed', []),
            phases_pending=data.get('phases_pending', []),
            error=data.get('error'),
            output_path=data.get('output_path'),
        )


def get_progress_file_path(base_path: Path) -> Path:
    """Get the progress file path."""
    return base_path / DEFAULT_PROGRESS_FILE


def load_background_progress(progress_path: Path) -> Optional[BackgroundProgress]:
    """Load background progress from file."""
    if not progress_path.exists():
        return None
    try:
        with open(progress_path, 'r') as f:
            data = json.load(f)
        return BackgroundProgress.from_dict(data)
    except (json.JSONDecodeError, IOError):
        return None


def save_background_progress(progress_path: Path, progress: BackgroundProgress) -> None:
    """Save background progress to file."""
    with open(progress_path, 'w') as f:
        json.dump(progress.to_dict(), f, indent=2)


def is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # Signal 0 checks existence without killing
        return True
    except (OSError, ProcessLookupError):
        return False


def display_progress(progress: BackgroundProgress, progress_path: Path) -> None:
    """Display current progress to the user."""
    print("\n" + "=" * 50)
    print("BACKGROUND FULL-ANALYSIS STATUS")
    print("=" * 50)

    print(f"\nStatus: {progress.status.upper()}")

    if progress.started_at:
        print(f"Started: {progress.started_at}")

    if progress.status == "running":
        # Check if process is actually alive
        if progress.pid and not is_process_alive(progress.pid):
            print("\n⚠️  WARNING: Background process appears to have died unexpectedly!")
            print("   The analysis may have crashed. Check logs for details.")
            print(f"   Progress file: {progress_path}")
        else:
            print(f"Process ID: {progress.pid}")
            print(f"\nCurrent phase: {progress.current_phase}")
            print(f"Progress: {progress.progress_percent:.1f}%")

            # Progress bar
            bar_width = 40
            filled = int(bar_width * progress.progress_percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"[{bar}]")

            if progress.phases_completed:
                print(f"\nCompleted phases: {', '.join(progress.phases_completed)}")
            if progress.phases_pending:
                print(f"Remaining phases: {', '.join(progress.phases_pending)}")

    elif progress.status == "completed":
        print(f"Completed: {progress.completed_at}")
        print(f"\n✓ Full analysis completed successfully!")
        if progress.output_path:
            print(f"Output saved to: {progress.output_path}")

        if progress.phases_completed:
            print(f"\nPhases completed: {', '.join(progress.phases_completed)}")

    elif progress.status == "failed":
        print(f"\n✗ Full analysis failed!")
        if progress.error:
            print(f"Error: {progress.error}")

    print(f"\nProgress file: {progress_path}")
    print("Run the command again to check for updates.\n")


class BackgroundProgressTracker(ProgressTracker):
    """
    Extended ProgressTracker that writes progress to a file for background monitoring.
    """

    def __init__(
        self,
        progress_path: Path,
        output_path: Path,
        log_file: Optional[str] = None,
        verbose: bool = False,
        quiet: bool = True,  # Default quiet for background
        total_phases: int = 6
    ):
        super().__init__(log_file=log_file, verbose=verbose, quiet=quiet)
        self.progress_path = progress_path
        self.output_path = output_path
        self.total_phases = total_phases
        self.completed_phase_count = 0

        # Define the phases for full analysis
        self.all_phases = [
            "Discovering files",
            "Indexing files",
            "Computing analysis (full mode)",
            "Extracting semantic relations",
            "Saving corpus",
            "Saving manifest"
        ]

        # Initialize progress
        self.bg_progress = BackgroundProgress(
            status="running",
            started_at=datetime.now().isoformat(),
            pid=os.getpid(),
            output_path=str(output_path),
            phases_pending=self.all_phases.copy(),
        )
        self._save_progress()

    def _save_progress(self):
        """Save current progress to file."""
        try:
            save_background_progress(self.progress_path, self.bg_progress)
        except Exception:
            pass  # Don't let progress save failures stop the analysis

    def start_phase(self, name: str, total_items: int = 0):
        """Start a new phase and update progress file."""
        super().start_phase(name, total_items)
        self.bg_progress.current_phase = name

        # Calculate progress based on phase
        base_progress = (self.completed_phase_count / self.total_phases) * 100
        self.bg_progress.progress_percent = base_progress
        self._save_progress()

    def end_phase(self, name: Optional[str] = None, status: str = "completed"):
        """End a phase and update progress file."""
        super().end_phase(name, status)
        phase_name = name or self.current_phase

        if status == "completed" and phase_name:
            self.completed_phase_count += 1
            if phase_name not in self.bg_progress.phases_completed:
                self.bg_progress.phases_completed.append(phase_name)
            if phase_name in self.bg_progress.phases_pending:
                self.bg_progress.phases_pending.remove(phase_name)

            # Update progress
            self.bg_progress.progress_percent = (
                self.completed_phase_count / self.total_phases
            ) * 100
            self._save_progress()

    def update_progress(self, items_processed: int, item_name: Optional[str] = None):
        """Update progress within current phase."""
        super().update_progress(items_processed, item_name)

        if self.current_phase and self.current_phase in self.phases:
            phase = self.phases[self.current_phase]
            if phase.items_total > 0:
                # Calculate intra-phase progress
                phase_progress = items_processed / phase.items_total
                base_progress = (self.completed_phase_count / self.total_phases) * 100
                phase_contribution = (1 / self.total_phases) * 100 * phase_progress
                self.bg_progress.progress_percent = base_progress + phase_contribution
                self._save_progress()

    def mark_completed(self):
        """Mark the analysis as completed."""
        self.bg_progress.status = "completed"
        self.bg_progress.completed_at = datetime.now().isoformat()
        self.bg_progress.progress_percent = 100.0
        self.bg_progress.current_phase = ""
        self._save_progress()

    def mark_failed(self, error: str):
        """Mark the analysis as failed."""
        self.bg_progress.status = "failed"
        self.bg_progress.completed_at = datetime.now().isoformat()
        self.bg_progress.error = error
        self._save_progress()


def run_background_analysis(
    base_path: Path,
    output_path: Path,
    progress_path: Path,
    use_chunks: bool,
    chunks_dir: str,
    timeout: int,
    verbose: bool,
    log_file: Optional[str]
) -> None:
    """
    Run full analysis in a background-compatible way.

    This function is designed to be called from a background thread/process.
    It writes progress updates to a file that can be monitored.
    """
    # Initialize background progress tracker
    tracker = BackgroundProgressTracker(
        progress_path=progress_path,
        output_path=output_path,
        log_file=log_file,
        verbose=verbose,
        quiet=True,  # Background mode is quiet to console
    )

    try:
        tracker.log("Cortical Text Processor - Background Full Analysis")
        tracker.log("=" * 50)
        tracker.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        tracker.log(f"Progress file: {progress_path}")

        # Get files to index
        tracker.start_phase("Discovering files")
        python_files = get_python_files(base_path)
        doc_files = get_doc_files(base_path)
        all_files = python_files + doc_files
        tracker.end_phase("Discovering files")

        tracker.log(f"\nFound {len(python_files)} Python files and {len(doc_files)} documentation files")

        # Create processor
        processor = create_code_processor()

        # Index all files
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

        # Run FULL analysis (the slow part)
        tracker.start_phase("Computing analysis (full mode)")
        processor.compute_all(
            build_concepts=True,
            pagerank_method='semantic',
            connection_strategy='hybrid',
            verbose=False
        )
        tracker.end_phase("Computing analysis (full mode)")

        # Extract semantic relations
        tracker.start_phase("Extracting semantic relations")
        processor.extract_corpus_semantics(
            use_pattern_extraction=True,
            verbose=False
        )
        tracker.end_phase("Extracting semantic relations")

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

        # Save manifest
        manifest_path = get_manifest_path(output_path)
        tracker.start_phase("Saving manifest")
        stats = {
            'documents': len(processor.documents),
            'tokens': processor.layers[0].column_count(),
            'bigrams': processor.layers[1].column_count(),
            'concepts': processor.layers[2].column_count(),
            'semantic_relations': len(processor.semantic_relations),
            'full_analysis': True,
        }
        save_manifest(manifest_path, file_mtimes, str(output_path), stats, tracker)
        tracker.end_phase("Saving manifest")

        # Mark as completed
        tracker.mark_completed()
        tracker.print_summary()
        tracker.log("\n✓ Background full analysis completed successfully!")
        tracker.log(f"Output saved to: {output_path}")

    except Exception as e:
        tracker.mark_failed(str(e))
        tracker.error(f"Background analysis failed: {e}")
        tracker.print_summary()
        raise


def start_background_analysis(
    base_path: Path,
    output_path: Path,
    progress_path: Path,
    use_chunks: bool = False,
    chunks_dir: str = 'corpus_chunks',
    timeout: int = 0,
    verbose: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Start full analysis in a background thread.

    The analysis runs in a daemon thread that will continue running
    even after the main process returns. Progress can be monitored
    via the progress file.
    """
    # Use a separate process for true background execution
    # This ensures the analysis continues even if the main script exits
    process = multiprocessing.Process(
        target=run_background_analysis,
        args=(
            base_path,
            output_path,
            progress_path,
            use_chunks,
            chunks_dir,
            timeout,
            verbose,
            log_file,
        ),
        daemon=False,  # Non-daemon so it survives parent exit
    )
    process.start()

    # Give it a moment to start and create progress file
    time.sleep(0.5)

    print("\n" + "=" * 50)
    print("FULL ANALYSIS STARTED IN BACKGROUND")
    print("=" * 50)
    print(f"\nProcess ID: {process.pid}")
    print(f"Progress file: {progress_path}")
    print(f"Output will be saved to: {output_path}")
    print("\nThe analysis is running in the background.")
    print("You can safely close this terminal.")
    print("\nTo check progress, run:")
    print(f"  python scripts/index_codebase.py --full-analysis")
    print("\nOr monitor the progress file directly:")
    print(f"  cat {progress_path}")
    print("=" * 50 + "\n")


# =============================================================================
# Incremental Full Analysis (Resumable Batch Mode)
# =============================================================================

DEFAULT_BATCH_SIZE = 20

# Analysis phases for incremental mode
PHASE_INDEXING = "indexing"
PHASE_FAST_ANALYSIS = "fast_analysis"
PHASE_FULL_ANALYSIS = "full_analysis"
PHASE_SEMANTIC_RELATIONS = "semantic_relations"
PHASE_COMPLETED = "completed"


@dataclass
class IncrementalProgress:
    """Progress state for incremental full-analysis runs."""
    status: str = "not_started"  # not_started, in_progress, completed
    started_at: Optional[str] = None
    last_updated: Optional[str] = None
    completed_at: Optional[str] = None

    # File tracking
    total_files: int = 0
    files_indexed: List[str] = field(default_factory=list)
    files_pending: List[str] = field(default_factory=list)

    # Phase tracking
    current_phase: str = PHASE_INDEXING
    phases_completed: List[str] = field(default_factory=list)

    # Stats
    batch_count: int = 0
    total_lines: int = 0
    error: Optional[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'started_at': self.started_at,
            'last_updated': self.last_updated,
            'completed_at': self.completed_at,
            'total_files': self.total_files,
            'files_indexed': self.files_indexed,
            'files_pending': self.files_pending,
            'current_phase': self.current_phase,
            'phases_completed': self.phases_completed,
            'batch_count': self.batch_count,
            'total_lines': self.total_lines,
            'error': self.error,
            'output_path': self.output_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IncrementalProgress':
        return cls(
            status=data.get('status', 'not_started'),
            started_at=data.get('started_at'),
            last_updated=data.get('last_updated'),
            completed_at=data.get('completed_at'),
            total_files=data.get('total_files', 0),
            files_indexed=data.get('files_indexed', []),
            files_pending=data.get('files_pending', []),
            current_phase=data.get('current_phase', PHASE_INDEXING),
            phases_completed=data.get('phases_completed', []),
            batch_count=data.get('batch_count', 0),
            total_lines=data.get('total_lines', 0),
            error=data.get('error'),
            output_path=data.get('output_path'),
        )

    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        # Phases: indexing (60%), fast_analysis (10%), full_analysis (20%), semantic (10%)
        phase_weights = {
            PHASE_INDEXING: 0.6,
            PHASE_FAST_ANALYSIS: 0.1,
            PHASE_FULL_ANALYSIS: 0.2,
            PHASE_SEMANTIC_RELATIONS: 0.1,
        }

        progress = 0.0

        # Add completed phases
        for phase in self.phases_completed:
            progress += phase_weights.get(phase, 0) * 100

        # Add current phase progress
        if self.current_phase == PHASE_INDEXING and self.total_files > 0:
            file_progress = len(self.files_indexed) / self.total_files
            progress += phase_weights[PHASE_INDEXING] * 100 * file_progress

        return min(progress, 100.0)


def get_incremental_progress_path(base_path: Path) -> Path:
    """Get the incremental progress file path."""
    return base_path / '.index_incremental_progress.json'


def load_incremental_progress(progress_path: Path) -> Optional[IncrementalProgress]:
    """Load incremental progress from file."""
    if not progress_path.exists():
        return None
    try:
        with open(progress_path, 'r') as f:
            data = json.load(f)
        return IncrementalProgress.from_dict(data)
    except (json.JSONDecodeError, IOError):
        return None


def save_incremental_progress(progress_path: Path, progress: IncrementalProgress) -> None:
    """Save incremental progress to file."""
    progress.last_updated = datetime.now().isoformat()
    with open(progress_path, 'w') as f:
        json.dump(progress.to_dict(), f, indent=2)


def display_incremental_progress(progress: IncrementalProgress, progress_path: Path) -> None:
    """Display current incremental progress to the user."""
    print("\n" + "=" * 50)
    print("INCREMENTAL FULL-ANALYSIS STATUS")
    print("=" * 50)

    print(f"\nStatus: {progress.status.upper()}")
    print(f"Current phase: {progress.current_phase}")

    if progress.started_at:
        print(f"Started: {progress.started_at}")
    if progress.last_updated:
        print(f"Last updated: {progress.last_updated}")

    # Progress bar
    pct = progress.progress_percent
    bar_width = 40
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\nProgress: {pct:.1f}%")
    print(f"[{bar}]")

    # File stats
    indexed = len(progress.files_indexed)
    pending = len(progress.files_pending)
    total = progress.total_files
    print(f"\nFiles: {indexed}/{total} indexed, {pending} pending")
    print(f"Batches completed: {progress.batch_count}")
    print(f"Total lines: {progress.total_lines:,}")

    # Phase status
    if progress.phases_completed:
        print(f"\nPhases completed: {', '.join(progress.phases_completed)}")

    if progress.status == "completed":
        print(f"\n✓ Full analysis completed successfully!")
        if progress.output_path:
            print(f"Output saved to: {progress.output_path}")
    elif progress.status == "in_progress":
        print(f"\nRun the command again to continue processing.")

    if progress.error:
        print(f"\nLast error: {progress.error}")

    print(f"\nProgress file: {progress_path}")
    print("")


def run_incremental_full_analysis(
    base_path: Path,
    output_path: Path,
    progress_path: Path,
    batch_size: int,
    tracker: ProgressTracker
) -> bool:
    """
    Run one batch of the incremental full analysis.

    Returns True if there's more work to do, False if complete.
    """
    # Load or create progress
    progress = load_incremental_progress(progress_path)

    if progress is None:
        # First run - discover files and initialize
        tracker.log("\nInitializing incremental full analysis...")
        tracker.start_phase("Discovering files")

        python_files = get_python_files(base_path)
        doc_files = get_doc_files(base_path)
        all_files = python_files + doc_files

        # Create list of doc_ids
        all_doc_ids = [create_doc_id(f, base_path) for f in all_files]

        progress = IncrementalProgress(
            status="in_progress",
            started_at=datetime.now().isoformat(),
            total_files=len(all_files),
            files_pending=all_doc_ids,
            current_phase=PHASE_INDEXING,
            output_path=str(output_path),
        )
        save_incremental_progress(progress_path, progress)
        tracker.end_phase("Discovering files")
        tracker.log(f"  Found {len(all_files)} files to index")

    # Check if already complete
    if progress.status == "completed":
        tracker.log("\n✓ Incremental full analysis already complete!")
        display_incremental_progress(progress, progress_path)
        return False

    # Load or create processor
    if output_path.exists() and len(progress.files_indexed) > 0:
        tracker.log(f"\nLoading existing corpus ({len(progress.files_indexed)} files indexed)...")
        try:
            processor = CorticalTextProcessor.load(str(output_path))
        except Exception as e:
            tracker.warn(f"Could not load corpus: {e}. Starting fresh.")
            processor = create_code_processor()
            progress.files_indexed = []
            progress.files_pending = [create_doc_id(f, base_path)
                                      for f in get_python_files(base_path) + get_doc_files(base_path)]
    else:
        processor = create_code_processor()

    # Phase: Indexing files
    if progress.current_phase == PHASE_INDEXING:
        if progress.files_pending:
            # Process next batch
            batch = progress.files_pending[:batch_size]
            tracker.start_phase(f"Indexing batch {progress.batch_count + 1}", len(batch))

            batch_lines = 0
            for i, doc_id in enumerate(batch, 1):
                file_path = base_path / doc_id
                tracker.update_progress(i, doc_id)

                if file_path.exists():
                    metadata = index_file(processor, file_path, base_path, tracker)
                    if metadata:
                        batch_lines += metadata.get('line_count', 0)
                        progress.files_indexed.append(doc_id)
                else:
                    tracker.warn(f"File not found: {doc_id}")

                # Remove from pending
                if doc_id in progress.files_pending:
                    progress.files_pending.remove(doc_id)

            progress.batch_count += 1
            progress.total_lines += batch_lines
            tracker.end_phase(f"Indexing batch {progress.batch_count}")
            tracker.log(f"  Indexed {len(batch)} files ({batch_lines:,} lines)")

            # Run fast analysis after each batch so corpus is usable
            tracker.start_phase("Fast analysis (batch)")
            processor.propagate_activation(verbose=False)
            processor.compute_importance(verbose=False)
            processor.compute_tfidf(verbose=False)
            tracker.end_phase("Fast analysis (batch)")

            # Save corpus and progress
            tracker.start_phase("Saving checkpoint")
            processor.save(str(output_path))
            save_incremental_progress(progress_path, progress)
            tracker.end_phase("Saving checkpoint")

            # Show status
            display_incremental_progress(progress, progress_path)

            if progress.files_pending:
                tracker.log(f"\n→ {len(progress.files_pending)} files remaining. Run again to continue.")
                return True

        # All files indexed - move to next phase
        progress.phases_completed.append(PHASE_INDEXING)
        progress.current_phase = PHASE_FAST_ANALYSIS
        save_incremental_progress(progress_path, progress)
        tracker.log("\n✓ All files indexed! Moving to analysis phase...")

    # Phase: Fast analysis (full corpus)
    if progress.current_phase == PHASE_FAST_ANALYSIS:
        tracker.start_phase("Fast analysis (full corpus)")
        processor.propagate_activation(verbose=False)
        processor.compute_importance(verbose=False)
        processor.compute_tfidf(verbose=False)
        processor.compute_document_connections(verbose=False)
        tracker.end_phase("Fast analysis (full corpus)")

        progress.phases_completed.append(PHASE_FAST_ANALYSIS)
        progress.current_phase = PHASE_FULL_ANALYSIS

        # Save checkpoint
        processor.save(str(output_path))
        save_incremental_progress(progress_path, progress)

        tracker.log("\n→ Fast analysis complete. Run again for full semantic analysis.")
        display_incremental_progress(progress, progress_path)
        return True

    # Phase: Full analysis (expensive)
    if progress.current_phase == PHASE_FULL_ANALYSIS:
        tracker.start_phase("Full semantic analysis")
        tracker.log("  This may take a few minutes...")

        try:
            processor.compute_all(
                build_concepts=True,
                pagerank_method='semantic',
                connection_strategy='hybrid',
                verbose=False
            )
            tracker.end_phase("Full semantic analysis")

            progress.phases_completed.append(PHASE_FULL_ANALYSIS)
            progress.current_phase = PHASE_SEMANTIC_RELATIONS

            # Save checkpoint
            processor.save(str(output_path))
            save_incremental_progress(progress_path, progress)

            tracker.log("\n→ Full analysis complete. Run again for semantic relations.")
            display_incremental_progress(progress, progress_path)
            return True

        except Exception as e:
            progress.error = str(e)
            save_incremental_progress(progress_path, progress)
            tracker.error(f"Full analysis failed: {e}")
            tracker.log("  You can retry by running the command again.")
            return True

    # Phase: Semantic relations
    if progress.current_phase == PHASE_SEMANTIC_RELATIONS:
        tracker.start_phase("Extracting semantic relations")

        try:
            processor.extract_corpus_semantics(
                use_pattern_extraction=True,
                verbose=False
            )
            tracker.end_phase("Extracting semantic relations")

            progress.phases_completed.append(PHASE_SEMANTIC_RELATIONS)
            progress.current_phase = PHASE_COMPLETED
            progress.status = "completed"
            progress.completed_at = datetime.now().isoformat()
            progress.error = None

            # Final save
            processor.save(str(output_path))

            # Save manifest
            manifest_path = get_manifest_path(output_path)
            file_mtimes = {}
            for doc_id in progress.files_indexed:
                file_path = base_path / doc_id
                if file_path.exists():
                    file_mtimes[doc_id] = get_file_mtime(file_path)

            stats = {
                'documents': len(processor.documents),
                'tokens': processor.layers[0].column_count(),
                'bigrams': processor.layers[1].column_count(),
                'concepts': processor.layers[2].column_count(),
                'semantic_relations': len(processor.semantic_relations),
                'full_analysis': True,
                'incremental': True,
            }
            save_manifest(manifest_path, file_mtimes, str(output_path), stats, tracker)
            save_incremental_progress(progress_path, progress)

            tracker.log("\n" + "=" * 50)
            tracker.log("✓ INCREMENTAL FULL ANALYSIS COMPLETE!")
            tracker.log("=" * 50)
            tracker.log(f"\nCorpus Statistics:")
            tracker.log(f"  Documents: {len(processor.documents)}")
            tracker.log(f"  Tokens: {processor.layers[0].column_count()}")
            tracker.log(f"  Bigrams: {processor.layers[1].column_count()}")
            tracker.log(f"  Concepts: {processor.layers[2].column_count()}")
            tracker.log(f"  Semantic relations: {len(processor.semantic_relations)}")
            tracker.log(f"\nOutput saved to: {output_path}")

            return False

        except Exception as e:
            progress.error = str(e)
            save_incremental_progress(progress_path, progress)
            tracker.error(f"Semantic extraction failed: {e}")
            tracker.log("  You can retry by running the command again.")
            return True

    return False


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


def extract_markdown_headings(content: str) -> List[str]:
    """Extract section headings from markdown content."""
    import re
    # Match ## and ### headings (skip # as it's usually the title)
    headings = re.findall(r'^##+ (.+)$', content, re.MULTILINE)
    return headings


def get_doc_type(doc_id: str) -> str:
    """
    Determine document type from document ID.

    Returns:
        One of: 'code', 'test', 'docs', 'root_docs'
    """
    if doc_id.startswith('tests/'):
        return 'test'
    elif doc_id.startswith('docs/'):
        return 'docs'
    elif doc_id.endswith('.md'):
        return 'root_docs'
    else:
        return 'code'


def _extract_file_metadata(
    doc_id: str,
    file_path: Path,
    content: str,
    mtime: float
) -> Dict[str, Any]:
    """
    Extract metadata from a file for chunk storage.

    Args:
        doc_id: Document ID (relative path)
        file_path: Full file path
        content: File content
        mtime: File modification time

    Returns:
        Metadata dictionary with doc_type, headings (for md), etc.
    """
    metadata = {
        'relative_path': doc_id,
        'file_type': file_path.suffix,
        'line_count': content.count('\n') + 1,
        'mtime': mtime,
        'doc_type': get_doc_type(doc_id),
    }

    # For Python files, extract additional metadata
    if file_path.suffix == '.py':
        metadata['language'] = 'python'
        metadata['function_count'] = content.count('\ndef ')
        metadata['class_count'] = content.count('\nclass ')

    # For Markdown files, extract headings
    if file_path.suffix == '.md':
        metadata['language'] = 'markdown'
        metadata['headings'] = extract_markdown_headings(content)

    return metadata


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
        'doc_type': get_doc_type(doc_id),
    }

    # For Python files, extract additional metadata
    if file_path.suffix == '.py':
        metadata['language'] = 'python'
        # Count functions and classes
        metadata['function_count'] = content.count('\ndef ')
        metadata['class_count'] = content.count('\nclass ')

    # For Markdown files, extract headings
    if file_path.suffix == '.md':
        metadata['language'] = 'markdown'
        metadata['headings'] = extract_markdown_headings(content)

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
# Chunk-Based Indexing
# =============================================================================

def run_compaction(args, base_path: Path, tracker: ProgressTracker) -> None:
    """Run chunk compaction."""
    chunks_dir = base_path / args.chunks_dir

    tracker.log(f"\nCompacting chunks in {chunks_dir}")

    compactor = ChunkCompactor(str(chunks_dir))

    # First do a dry run to show what would happen
    dry_result = compactor.compact(
        before=args.compact_before,
        keep_recent=args.compact_keep,
        dry_run=True
    )

    if dry_result['status'] == 'no_chunks':
        tracker.log("No chunks found to compact.")
        return

    if dry_result['status'] == 'nothing_to_compact':
        tracker.log("No chunks match compaction criteria.")
        return

    tracker.log(f"  Would compact {dry_result['would_compact']} chunks")
    tracker.log(f"  Would keep {dry_result['would_keep']} chunks")

    # Actually compact
    tracker.start_phase("Compacting chunks")
    result = compactor.compact(
        before=args.compact_before,
        keep_recent=args.compact_keep,
        dry_run=False
    )
    tracker.end_phase("Compacting chunks")

    tracker.log(f"\nCompaction complete:")
    tracker.log(f"  Chunks compacted: {result['compacted']}")
    tracker.log(f"  Chunks kept: {result['kept']}")
    tracker.log(f"  Documents in compacted chunk: {result['documents']}")
    if result.get('compacted_file'):
        tracker.log(f"  Output file: {result['compacted_file']}")

    tracker.print_summary()


def index_with_chunks(
    args,
    base_path: Path,
    output_path: Path,
    tracker: ProgressTracker
) -> None:
    """
    Index using chunk-based storage.

    This creates timestamped JSON chunks that can be committed to git.
    """
    chunks_dir = base_path / args.chunks_dir

    # Initialize chunk loader to get current state from existing chunks
    loader = ChunkLoader(str(chunks_dir))
    existing_docs = loader.get_documents()
    existing_mtimes = loader.get_mtimes()

    tracker.log(f"\nChunk-based indexing mode")
    tracker.log(f"  Chunks directory: {chunks_dir}")
    tracker.log(f"  Existing documents from chunks: {len(existing_docs)}")

    # Get files to index
    tracker.start_phase("Discovering files")
    python_files = get_python_files(base_path)
    doc_files = get_doc_files(base_path)
    all_files = python_files + doc_files
    tracker.end_phase("Discovering files")

    tracker.log(f"\nFound {len(python_files)} Python files and {len(doc_files)} documentation files")

    # Build current file state
    current_files = {}
    for file_path in all_files:
        doc_id = create_doc_id(file_path, base_path)
        current_files[doc_id] = get_file_mtime(file_path)

    # Detect changes against chunk state
    added, modified, deleted = get_chunk_changes(current_files, existing_mtimes)

    tracker.log(f"\nChanges detected:")
    tracker.log(f"  Added: {len(added)}, Modified: {len(modified)}, Deleted: {len(deleted)}")

    # Status mode
    if args.status:
        show_chunk_status(added, modified, deleted, loader, tracker)
        return

    # No changes
    if not (added or modified or deleted):
        tracker.log("\nNo changes detected. Corpus is up to date.")
        return

    # Create chunk writer for this session
    writer = ChunkWriter(str(chunks_dir))

    # Record operations
    total_ops = len(added) + len(modified) + len(deleted)
    tracker.start_phase("Recording chunk operations", total_items=total_ops)
    processed = 0

    # Process added files
    for doc_id in added:
        file_path = base_path / doc_id
        try:
            content = file_path.read_text(encoding='utf-8')
            mtime = get_file_mtime(file_path)
            metadata = _extract_file_metadata(doc_id, file_path, content, mtime)
            writer.add_document(doc_id, content, mtime, metadata)
            processed += 1
            tracker.update_progress(processed, f"Added: {doc_id}" if args.verbose else None)
        except Exception as e:
            tracker.warn(f"Error reading {doc_id}: {e}")

    # Process modified files
    for doc_id in modified:
        file_path = base_path / doc_id
        try:
            content = file_path.read_text(encoding='utf-8')
            mtime = get_file_mtime(file_path)
            metadata = _extract_file_metadata(doc_id, file_path, content, mtime)
            writer.modify_document(doc_id, content, mtime, metadata)
            processed += 1
            tracker.update_progress(processed, f"Modified: {doc_id}" if args.verbose else None)
        except Exception as e:
            tracker.warn(f"Error reading {doc_id}: {e}")

    # Record deletions
    for doc_id in deleted:
        writer.delete_document(doc_id)
        processed += 1
        tracker.update_progress(processed, f"Deleted: {doc_id}" if args.verbose else None)

    tracker.end_phase("Recording chunk operations")

    # Save chunk
    tracker.start_phase("Saving chunk")
    chunk_path = writer.save()
    if chunk_path:
        tracker.log(f"  Saved chunk: {chunk_path.name}")
    tracker.end_phase("Saving chunk")

    # Now rebuild processor from all chunks
    tracker.start_phase("Loading documents from chunks")
    loader = ChunkLoader(str(chunks_dir))  # Reload to include new chunk
    all_docs = loader.get_documents()
    all_metadata = loader.get_metadata()
    tracker.log(f"  Total documents: {len(all_docs)}")
    tracker.log(f"  Documents with metadata: {len(all_metadata)}")
    tracker.end_phase("Loading documents from chunks")

    # Check if we can use cached pkl
    cache_valid = loader.is_cache_valid(str(output_path))
    if cache_valid and not (added or modified or deleted):
        tracker.log("\nCache is valid, loading from pkl...")
        processor = CorticalTextProcessor.load(str(output_path))
    else:
        # Build processor from documents (with metadata)
        tracker.start_phase("Building processor from chunks")
        processor = create_code_processor()
        documents = [
            (doc_id, content, all_metadata.get(doc_id))
            for doc_id, content in all_docs.items()
        ]
        processor.add_documents_batch(documents, recompute='none', verbose=False)
        tracker.log(f"  Added {len(documents)} documents")
        tracker.end_phase("Building processor from chunks")

        # Compute analysis
        fast_mode = not args.full_analysis
        compute_analysis(processor, tracker, fast_mode=fast_mode)

    # Print corpus statistics
    tracker.log("\nCorpus Statistics:")
    tracker.log(f"  Documents: {len(processor.documents)}")
    tracker.log(f"  Tokens (Layer 0): {processor.layers[0].column_count()}")
    tracker.log(f"  Bigrams (Layer 1): {processor.layers[1].column_count()}")

    # Save processor
    tracker.start_phase("Saving corpus")
    processor.save(str(output_path))
    file_size = output_path.stat().st_size / 1024
    tracker.log(f"  Saved to {output_path.name} ({file_size:.1f} KB)")
    tracker.end_phase("Saving corpus")

    # Save cache hash
    loader.save_cache_hash(str(output_path))
    tracker.log(f"  Cache hash saved")

    # Print chunk stats
    stats = loader.get_stats()
    tracker.log(f"\nChunk Statistics:")
    tracker.log(f"  Total chunks: {stats['chunk_count']}")
    tracker.log(f"  Total operations: {stats['total_operations']}")
    tracker.log(f"  Content hash: {stats['hash']}")

    tracker.print_summary()
    tracker.log("\nDone! Use search_codebase.py to query the indexed corpus.")


def show_chunk_status(
    added: List[str],
    modified: List[str],
    deleted: List[str],
    loader: ChunkLoader,
    tracker: ProgressTracker
) -> None:
    """Show chunk status without indexing."""
    stats = loader.get_stats()

    tracker.log(f"\n=== Chunk Status ===")
    tracker.log(f"Chunks: {stats['chunk_count']}")
    tracker.log(f"Documents: {stats['document_count']}")
    tracker.log(f"Content hash: {stats['hash']}")

    if added:
        tracker.log(f"\nFiles to add ({len(added)}):")
        for f in added[:10]:
            tracker.log(f"  + {f}")
        if len(added) > 10:
            tracker.log(f"  ... and {len(added) - 10} more")

    if modified:
        tracker.log(f"\nFiles to modify ({len(modified)}):")
        for f in modified[:10]:
            tracker.log(f"  ~ {f}")
        if len(modified) > 10:
            tracker.log(f"  ... and {len(modified) - 10} more")

    if deleted:
        tracker.log(f"\nFiles to delete ({len(deleted)}):")
        for f in deleted[:10]:
            tracker.log(f"  - {f}")
        if len(deleted) > 10:
            tracker.log(f"  ... and {len(deleted) - 10} more")

    if not (added or modified or deleted):
        tracker.log("\nNo changes detected.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Index the codebase for semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_codebase.py                   # Full rebuild (fast mode)
  python scripts/index_codebase.py --incremental    # Update changed files only
  python scripts/index_codebase.py --status         # Show what would change
  python scripts/index_codebase.py --force          # Force full rebuild
  python scripts/index_codebase.py --log index.log  # Log to file
  python scripts/index_codebase.py --timeout 60     # Timeout after 60s

Background Full Analysis (for long-running environments):
  python scripts/index_codebase.py --full-analysis              # Start in background
  python scripts/index_codebase.py --full-analysis              # Check progress
  python scripts/index_codebase.py --full-analysis --foreground # Run synchronously

Incremental Full Analysis (resumable, for short-lived processes):
  python scripts/index_codebase.py --full-analysis --batch      # Process one batch
  python scripts/index_codebase.py --full-analysis --batch      # Continue (run again)
  python scripts/index_codebase.py --full-analysis --batch --batch-size 10  # Smaller batches
  python scripts/index_codebase.py --full-analysis --batch --status  # Check progress
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
                        help='Use full semantic analysis (runs in background by default)')
    parser.add_argument('--foreground', action='store_true',
                        help='Run full analysis in foreground (blocking) instead of background')
    parser.add_argument('--progress-file', type=str, default=None,
                        help='Custom path for progress file (default: .index_progress.json)')

    # Incremental batch mode options
    parser.add_argument('--batch', action='store_true',
                        help='Use incremental batch mode (resumable, for short-lived processes)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Number of files per batch (default: {DEFAULT_BATCH_SIZE})')

    # Chunk-based indexing options
    parser.add_argument('--use-chunks', action='store_true',
                        help='Use git-compatible chunk-based indexing')
    parser.add_argument('--chunks-dir', default='corpus_chunks',
                        help='Directory for chunk files (default: corpus_chunks)')
    parser.add_argument('--compact', action='store_true',
                        help='Compact old chunks into a single file')
    parser.add_argument('--compact-before', type=str, default=None,
                        help='Only compact chunks before this date (YYYY-MM-DD)')
    parser.add_argument('--compact-keep', type=int, default=0,
                        help='Keep this many recent chunks when compacting')

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    output_path = base_path / args.output
    manifest_path = get_manifest_path(output_path)

    # Set up progress file path
    if args.progress_file:
        progress_path = Path(args.progress_file)
        if not progress_path.is_absolute():
            progress_path = base_path / args.progress_file
    else:
        progress_path = get_progress_file_path(base_path)

    # Set up log file path
    log_path = None
    if args.log:
        log_path = args.log if os.path.isabs(args.log) else str(base_path / args.log)

    # Handle incremental batch mode for full-analysis
    if args.full_analysis and args.batch:
        incremental_progress_path = get_incremental_progress_path(base_path)

        # Status check only
        if args.status:
            existing = load_incremental_progress(incremental_progress_path)
            if existing:
                display_incremental_progress(existing, incremental_progress_path)
            else:
                print("\nNo incremental full-analysis in progress.")
                print(f"Run with --full-analysis --batch to start.\n")
            return

        # Force restart
        if args.force:
            if incremental_progress_path.exists():
                incremental_progress_path.unlink()
                print("Cleared previous incremental progress.\n")

        # Initialize tracker for batch mode
        tracker = ProgressTracker(
            log_file=log_path,
            verbose=args.verbose,
            quiet=args.quiet
        )

        tracker.log("Cortical Text Processor - Incremental Full Analysis")
        tracker.log("=" * 50)
        tracker.log(f"Batch size: {args.batch_size} files")

        # Run one batch
        more_work = run_incremental_full_analysis(
            base_path=base_path,
            output_path=output_path,
            progress_path=incremental_progress_path,
            batch_size=args.batch_size,
            tracker=tracker
        )

        tracker.print_summary()

        if more_work:
            print("\nRun the command again to continue processing.")
        return

    # Handle background full-analysis mode
    if args.full_analysis and not args.foreground and not args.batch:
        # Check for existing progress
        existing_progress = load_background_progress(progress_path)

        if existing_progress:
            if existing_progress.status == "running":
                # Check if the process is actually alive
                if existing_progress.pid and is_process_alive(existing_progress.pid):
                    # Show progress and exit
                    display_progress(existing_progress, progress_path)
                    return
                else:
                    # Process died, show warning and offer to restart
                    print("\n" + "=" * 50)
                    print("PREVIOUS BACKGROUND ANALYSIS")
                    print("=" * 50)
                    print(f"\nStatus: STALE (process {existing_progress.pid} no longer running)")
                    print(f"Started: {existing_progress.started_at}")
                    print(f"Last phase: {existing_progress.current_phase}")
                    print(f"Progress: {existing_progress.progress_percent:.1f}%")
                    print("\nThe previous background process appears to have stopped.")
                    print("Starting a new background analysis...\n")

            elif existing_progress.status == "completed":
                # Show completion and ask if they want to re-run
                display_progress(existing_progress, progress_path)
                print("To start a fresh analysis, delete the progress file first:")
                print(f"  rm {progress_path}")
                print("Or run with --force to overwrite.\n")
                if not args.force:
                    return

            elif existing_progress.status == "failed":
                # Show failure and start new
                print("\n" + "=" * 50)
                print("PREVIOUS ANALYSIS FAILED")
                print("=" * 50)
                print(f"\nError: {existing_progress.error}")
                print("\nStarting a new background analysis...\n")

        # Start new background analysis
        start_background_analysis(
            base_path=base_path,
            output_path=output_path,
            progress_path=progress_path,
            use_chunks=args.use_chunks,
            chunks_dir=args.chunks_dir,
            timeout=args.timeout,
            verbose=args.verbose,
            log_file=log_path,
        )
        return

    # Initialize progress tracker for foreground operation
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

    # Handle compaction mode
    if args.compact:
        run_compaction(args, base_path, tracker)
        return

    try:
        with timeout_handler(args.timeout, tracker):
            if args.use_chunks:
                index_with_chunks(args, base_path, output_path, tracker)
            else:
                run_indexer(args, base_path, output_path, manifest_path, tracker)
    except IndexingTimeoutError:
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
            processor = create_code_processor()
            added, modified, deleted = all_files, [], []
    else:
        processor = create_code_processor()
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
