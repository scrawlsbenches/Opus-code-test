#!/usr/bin/env python3
"""
Commit-Task Linker

Links git commits to GoT tasks using multiple strategies:
1. Explicit references (T-XXXXXXXX-XXXXXXXX pattern in commit messages)
2. Semantic similarity between commit messages and task descriptions
3. File overlap analysis (files changed in commit vs files related to task)

Usage:
    python scripts/commit_task_linker.py link              # Create all links
    python scripts/commit_task_linker.py show T-XXXX       # Show commits for task
    python scripts/commit_task_linker.py show <commit_hash> # Show tasks for commit
    python scripts/commit_task_linker.py export            # Export training data
    python scripts/commit_task_linker.py stats             # Show statistics
"""

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator

# Import from ml_file_prediction for semantic similarity
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ml_file_prediction import compute_semantic_similarity, load_commit_data

# ============================================================================
# CONFIGURATION
# ============================================================================

# GoT entities directory
GOT_ENTITIES_DIR = Path(__file__).parent.parent / ".got" / "entities"

# Commit links storage
COMMIT_LINKS_FILE = Path(__file__).parent.parent / ".got" / "commit_links.json"

# Linking thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for semantic linking
FILE_OVERLAP_THRESHOLD = 0.2  # Minimum Jaccard similarity for file-based linking

# Performance and resource limits
BATCH_SIZE = 100  # Process commits in batches for semantic similarity
MAX_CANDIDATES = 50  # Only consider top candidates per commit
MAX_COMMITS_TO_PROCESS = 5000  # Maximum commits to process
MAX_TASKS_TO_MATCH = 1000  # Maximum tasks to match against
MAX_JSON_SIZE_MB = 50  # Maximum JSON file size to load (MB)
SIMILARITY_TIMEOUT_SECONDS = 30  # Timeout per batch of similarity computations

# Task ID pattern in commit messages
TASK_ID_PATTERN = re.compile(r'T-\d{8}-\d{6}-[0-9a-f]{8}')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TaskInfo:
    """Information about a GoT task."""
    id: str
    title: str
    description: str
    status: str
    category: str
    created_at: str
    completed_at: Optional[str] = None

    def get_text_for_similarity(self) -> str:
        """Get combined text for semantic similarity matching."""
        return f"{self.title} {self.description}"


@dataclass
class CommitInfo:
    """Information about a git commit."""
    hash: str
    message: str
    timestamp: str
    files_changed: List[str]
    author: str
    insertions: int
    deletions: int


@dataclass
class CommitTaskLink:
    """A link between a commit and a task."""
    commit_hash: str
    task_id: str
    link_type: str  # 'explicit', 'semantic', 'file_overlap'
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CommitTaskLink':
        return cls(**d)


@dataclass
class LinkageStats:
    """Statistics about commit-task linkage."""
    total_commits: int = 0
    total_tasks: int = 0
    total_links: int = 0
    explicit_links: int = 0
    semantic_links: int = 0
    file_overlap_links: int = 0
    commits_with_links: int = 0
    tasks_with_links: int = 0
    avg_links_per_commit: float = 0.0
    avg_links_per_task: float = 0.0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_output_path(path: str) -> str:
    """
    Validate output path to prevent directory traversal attacks.

    Args:
        path: The requested output path

    Returns:
        Validated absolute path

    Raises:
        ValueError: If path is invalid or outside allowed directories
    """
    resolved = os.path.abspath(path)
    cwd = os.getcwd()

    # Ensure within allowed output directories
    # Allow paths within project directory or temporary directories
    allowed_prefixes = [
        cwd,
        os.path.join(cwd, '.got'),
        '/tmp',
        '/var/tmp'
    ]

    # Check for directory traversal attempts
    if '..' in path or (path.startswith('/') and not any(resolved.startswith(prefix) for prefix in allowed_prefixes)):
        if not any(resolved.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"Invalid output path: {path} (resolved to {resolved}). Path must be within project directory.")

    return resolved


def _validate_json_size(file_path: Path, max_size_mb: int = MAX_JSON_SIZE_MB) -> None:
    """
    Validate JSON file size before loading to prevent memory exhaustion.

    Args:
        file_path: Path to JSON file
        max_size_mb: Maximum allowed size in MB

    Raises:
        ValueError: If file is too large
    """
    if not file_path.exists():
        return

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"JSON file too large: {file_size_mb:.2f}MB exceeds maximum of {max_size_mb}MB. "
            f"File: {file_path}"
        )


# ============================================================================
# COMMIT-TASK LINKER
# ============================================================================

class CommitTaskLinker:
    """Links git commits to GoT tasks using multiple strategies."""

    def __init__(self, links_file: Path = None, max_commits: int = None, max_tasks: int = None):
        """
        Initialize the commit-task linker.

        Args:
            links_file: Path to store/load links (default: COMMIT_LINKS_FILE)
            max_commits: Maximum number of commits to process (default: MAX_COMMITS_TO_PROCESS)
            max_tasks: Maximum number of tasks to match (default: MAX_TASKS_TO_MATCH)
        """
        self.links_file = links_file or COMMIT_LINKS_FILE
        self.max_commits = max_commits or MAX_COMMITS_TO_PROCESS
        self.max_tasks = max_tasks or MAX_TASKS_TO_MATCH
        self.links: List[CommitTaskLink] = []
        self.tasks: Dict[str, TaskInfo] = {}
        self.commits: Dict[str, CommitInfo] = {}

        # Index structures for fast lookup
        self._commit_to_tasks: Dict[str, Set[str]] = defaultdict(set)
        self._task_to_commits: Dict[str, Set[str]] = defaultdict(set)

        # Load existing links if available
        if self.links_file.exists():
            self._load_links()

    def load_tasks(self) -> None:
        """Load tasks from GoT entities directory (up to max_tasks limit)."""
        logger.info(f"Loading tasks from {GOT_ENTITIES_DIR} (max: {self.max_tasks})")

        if not GOT_ENTITIES_DIR.exists():
            logger.warning(f"GoT entities directory not found: {GOT_ENTITIES_DIR}")
            return

        task_count = 0
        for task_file in GOT_ENTITIES_DIR.glob("T-*.json"):
            # Respect max_tasks limit
            if task_count >= self.max_tasks:
                logger.warning(f"Reached max_tasks limit of {self.max_tasks}, stopping task loading")
                break

            # Skip edge files
            if task_file.stem.startswith("E-"):
                continue

            try:
                # Validate JSON size before loading
                _validate_json_size(task_file)

                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                # Extract task information
                data = task_data.get('data', {})
                if data.get('entity_type') != 'task':
                    continue

                task_info = TaskInfo(
                    id=data.get('id', ''),
                    title=data.get('title', ''),
                    description=data.get('description', ''),
                    status=data.get('status', 'unknown'),
                    category=data.get('properties', {}).get('category', 'unknown'),
                    created_at=data.get('created_at', ''),
                    completed_at=data.get('metadata', {}).get('completed_at')
                )

                self.tasks[task_info.id] = task_info
                task_count += 1

            except (json.JSONDecodeError, KeyError, IOError, ValueError) as e:
                logger.warning(f"Failed to load task from {task_file}: {e}")
                continue

        logger.info(f"Loaded {task_count} tasks")

    def load_commits(self) -> None:
        """Load commits from ML data collector (up to max_commits limit)."""
        logger.info(f"Loading commits from ML data collector (max: {self.max_commits})")

        # Use the existing load_commit_data function
        try:
            commit_examples = load_commit_data(filter_deleted=False, use_cali=True)

            # Process commits in limited batches to avoid memory issues
            commit_count = 0
            for example in commit_examples:
                # Respect max_commits limit
                if commit_count >= self.max_commits:
                    logger.warning(f"Reached max_commits limit of {self.max_commits}, stopping commit loading")
                    break

                commit_info = CommitInfo(
                    hash=example.commit_hash,
                    message=example.message,
                    timestamp=example.timestamp,
                    files_changed=example.files_changed,
                    author='',  # Not available in TrainingExample
                    insertions=example.insertions,
                    deletions=example.deletions
                )

                self.commits[commit_info.hash] = commit_info
                commit_count += 1

            logger.info(f"Loaded {len(self.commits)} commits")

        except Exception as e:
            logger.error(f"Failed to load commits: {e}")
            raise

    def _load_links(self) -> None:
        """Load existing links from file."""
        try:
            # Validate JSON size before loading
            _validate_json_size(self.links_file)

            with open(self.links_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.links = [CommitTaskLink.from_dict(link) for link in data.get('links', [])]

            # Rebuild indices
            self._rebuild_indices()

            logger.info(f"Loaded {len(self.links)} existing links from {self.links_file}")

        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Failed to load existing links: {e}")
            self.links = []

    def _save_links(self) -> None:
        """Save links to file."""
        self.links_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'total_links': len(self.links),
            'links': [link.to_dict() for link in self.links]
        }

        with open(self.links_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.links)} links to {self.links_file}")

    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices from links."""
        self._commit_to_tasks.clear()
        self._task_to_commits.clear()

        for link in self.links:
            self._commit_to_tasks[link.commit_hash].add(link.task_id)
            self._task_to_commits[link.task_id].add(link.commit_hash)

    def _add_link(self, commit_hash: str, task_id: str, link_type: str,
                  confidence: float, metadata: Dict[str, Any] = None) -> None:
        """
        Add a link between a commit and a task.

        Args:
            commit_hash: Commit hash
            task_id: Task ID
            link_type: Type of link ('explicit', 'semantic', 'file_overlap')
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata about the link
        """
        # Check if link already exists
        if task_id in self._commit_to_tasks[commit_hash]:
            # Update existing link if new confidence is higher
            for link in self.links:
                if link.commit_hash == commit_hash and link.task_id == task_id:
                    if confidence > link.confidence:
                        link.confidence = confidence
                        link.link_type = link_type
                        link.metadata = metadata or {}
                    return

        # Create new link
        link = CommitTaskLink(
            commit_hash=commit_hash,
            task_id=task_id,
            link_type=link_type,
            confidence=confidence,
            metadata=metadata or {}
        )

        self.links.append(link)
        self._commit_to_tasks[commit_hash].add(task_id)
        self._task_to_commits[task_id].add(commit_hash)

    def link_explicit_references(self) -> int:
        """
        Link commits that explicitly reference task IDs in their messages.

        Returns:
            Number of explicit links found
        """
        logger.info("Finding explicit task references in commit messages...")

        links_found = 0
        for commit_hash, commit in self.commits.items():
            # Find all task ID references in commit message
            task_ids = TASK_ID_PATTERN.findall(commit.message)

            for task_id in task_ids:
                # Verify task exists
                if task_id in self.tasks:
                    self._add_link(
                        commit_hash=commit_hash,
                        task_id=task_id,
                        link_type='explicit',
                        confidence=1.0,  # Explicit references have 100% confidence
                        metadata={'pattern_matched': task_id}
                    )
                    links_found += 1

        logger.info(f"Found {links_found} explicit task references")
        return links_found

    def link_semantic_similarity(self, threshold: float = None) -> int:
        """
        Link commits to tasks based on semantic similarity.

        Uses batching and progress logging to handle large datasets efficiently.
        Pre-filters tasks by time/category to reduce comparison space.

        Args:
            threshold: Minimum similarity threshold (default: SEMANTIC_SIMILARITY_THRESHOLD)

        Returns:
            Number of semantic links found
        """
        threshold = threshold or SEMANTIC_SIMILARITY_THRESHOLD
        logger.info(f"Finding semantic similarities (threshold={threshold})...")
        logger.info(f"Processing {len(self.commits)} commits against {len(self.tasks)} tasks")
        logger.info(f"Batch size: {BATCH_SIZE}, Max candidates per commit: {MAX_CANDIDATES}")

        links_found = 0
        total_comparisons = 0

        # Convert commits to list for batching
        commit_items = list(self.commits.items())
        total_commits = len(commit_items)

        # Process commits in batches
        for batch_start in range(0, total_commits, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_commits)
            batch = commit_items[batch_start:batch_end]

            logger.info(f"Processing commits {batch_start + 1}-{batch_end} of {total_commits}...")

            batch_links = 0
            for commit_hash, commit in batch:
                # Pre-filter tasks - only compare against potentially relevant tasks
                # For now, compare against all tasks but limit top candidates
                candidates: List[Tuple[str, TaskInfo, float]] = []

                for task_id, task in self.tasks.items():
                    # Skip if already explicitly linked
                    if task_id in self._commit_to_tasks[commit_hash]:
                        continue

                    # Compute semantic similarity
                    similarity = compute_semantic_similarity(
                        commit.message,
                        task.get_text_for_similarity()
                    )
                    total_comparisons += 1

                    # Only consider above threshold
                    if similarity >= threshold:
                        candidates.append((task_id, task, similarity))

                # Sort candidates by similarity (highest first) and take top MAX_CANDIDATES
                candidates.sort(key=lambda x: x[2], reverse=True)
                top_candidates = candidates[:MAX_CANDIDATES]

                # Add links for top candidates
                for task_id, task, similarity in top_candidates:
                    self._add_link(
                        commit_hash=commit_hash,
                        task_id=task_id,
                        link_type='semantic',
                        confidence=similarity,
                        metadata={
                            'similarity_score': similarity,
                            'commit_message': commit.message[:100],
                            'task_title': task.title[:100]
                        }
                    )
                    batch_links += 1

            links_found += batch_links
            logger.info(f"  Batch found {batch_links} links (total so far: {links_found})")

        logger.info(f"Found {links_found} semantic similarity links from {total_comparisons} comparisons")
        return links_found

    def link_file_overlap(self, threshold: float = None) -> int:
        """
        Link commits to tasks based on file overlap.

        This uses a simple heuristic: if a commit mentions a task ID explicitly
        and changes certain files, we can infer that those files are related
        to that task. Then, other commits that change those same files might
        also be related to the task.

        Args:
            threshold: Minimum Jaccard similarity threshold

        Returns:
            Number of file overlap links found
        """
        threshold = threshold or FILE_OVERLAP_THRESHOLD
        logger.info(f"Finding file overlap patterns (threshold={threshold})...")

        # First, build a map of task -> files from explicit links
        task_files: Dict[str, Set[str]] = defaultdict(set)

        for link in self.links:
            if link.link_type == 'explicit':
                commit = self.commits.get(link.commit_hash)
                if commit:
                    task_files[link.task_id].update(commit.files_changed)

        if not task_files:
            logger.warning("No explicit links found; skipping file overlap linking")
            return 0

        # Now find commits with file overlap
        links_found = 0
        for commit_hash, commit in self.commits.items():
            commit_files = set(commit.files_changed)
            if not commit_files:
                continue

            for task_id, task_file_set in task_files.items():
                # Skip if already linked
                if task_id in self._commit_to_tasks[commit_hash]:
                    continue

                # Compute Jaccard similarity
                intersection = commit_files & task_file_set
                union = commit_files | task_file_set

                if not union:
                    continue

                jaccard = len(intersection) / len(union)

                if jaccard >= threshold:
                    self._add_link(
                        commit_hash=commit_hash,
                        task_id=task_id,
                        link_type='file_overlap',
                        confidence=jaccard,
                        metadata={
                            'jaccard_similarity': jaccard,
                            'overlapping_files': list(intersection)[:5],
                            'total_overlap': len(intersection)
                        }
                    )
                    links_found += 1

        logger.info(f"Found {links_found} file overlap links")
        return links_found

    def link_all(self, semantic_threshold: float = None,
                 file_threshold: float = None) -> Dict[str, int]:
        """
        Run all linking strategies.

        Args:
            semantic_threshold: Semantic similarity threshold
            file_threshold: File overlap threshold

        Returns:
            Dict with counts for each link type
        """
        logger.info("Starting comprehensive commit-task linking...")

        # Load data if not already loaded
        if not self.tasks:
            self.load_tasks()
        if not self.commits:
            self.load_commits()

        # Clear existing links
        self.links.clear()
        self._commit_to_tasks.clear()
        self._task_to_commits.clear()

        # Run linking strategies in order
        explicit_count = self.link_explicit_references()
        semantic_count = self.link_semantic_similarity(semantic_threshold)
        file_count = self.link_file_overlap(file_threshold)

        # Save links
        self._save_links()

        return {
            'explicit': explicit_count,
            'semantic': semantic_count,
            'file_overlap': file_count,
            'total': len(self.links)
        }

    def get_links_for_task(self, task_id: str) -> List[Tuple[CommitInfo, CommitTaskLink]]:
        """
        Get all commits linked to a task.

        Args:
            task_id: Task ID

        Returns:
            List of (CommitInfo, CommitTaskLink) tuples
        """
        result = []
        commit_hashes = self._task_to_commits.get(task_id, set())

        for commit_hash in commit_hashes:
            commit = self.commits.get(commit_hash)
            if not commit:
                continue

            # Find the link
            link = next(
                (l for l in self.links
                 if l.commit_hash == commit_hash and l.task_id == task_id),
                None
            )

            if link:
                result.append((commit, link))

        # Sort by timestamp (most recent first)
        result.sort(key=lambda x: x[0].timestamp, reverse=True)
        return result

    def get_links_for_commit(self, commit_hash: str) -> List[Tuple[TaskInfo, CommitTaskLink]]:
        """
        Get all tasks linked to a commit.

        Args:
            commit_hash: Commit hash (can be partial)

        Returns:
            List of (TaskInfo, CommitTaskLink) tuples
        """
        # Support partial commit hashes
        matching_hashes = [h for h in self.commits.keys() if h.startswith(commit_hash)]

        if not matching_hashes:
            return []

        # Use the first matching hash
        full_hash = matching_hashes[0]

        result = []
        task_ids = self._commit_to_tasks.get(full_hash, set())

        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if not task:
                continue

            # Find the link
            link = next(
                (l for l in self.links
                 if l.commit_hash == full_hash and l.task_id == task_id),
                None
            )

            if link:
                result.append((task, link))

        # Sort by confidence (highest first)
        result.sort(key=lambda x: x[1].confidence, reverse=True)
        return result

    def get_stats(self) -> LinkageStats:
        """Get statistics about linkage."""
        # Count unique commits and tasks with links
        linked_commits = set(link.commit_hash for link in self.links)
        linked_tasks = set(link.task_id for link in self.links)

        # Count by type
        explicit = sum(1 for link in self.links if link.link_type == 'explicit')
        semantic = sum(1 for link in self.links if link.link_type == 'semantic')
        file_overlap = sum(1 for link in self.links if link.link_type == 'file_overlap')

        # Calculate averages
        avg_links_per_commit = len(self.links) / len(linked_commits) if linked_commits else 0
        avg_links_per_task = len(self.links) / len(linked_tasks) if linked_tasks else 0

        return LinkageStats(
            total_commits=len(self.commits),
            total_tasks=len(self.tasks),
            total_links=len(self.links),
            explicit_links=explicit,
            semantic_links=semantic,
            file_overlap_links=file_overlap,
            commits_with_links=len(linked_commits),
            tasks_with_links=len(linked_tasks),
            avg_links_per_commit=avg_links_per_commit,
            avg_links_per_task=avg_links_per_task
        )

    def export_training_data(self, output_file: Path = None) -> str:
        """
        Export links in format suitable for ML training.

        The format includes:
        - Positive examples: (commit_message, task_description, 1)
        - Can be used to train a classifier for commit-task matching

        Args:
            output_file: Output file path

        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.links_file.parent / "commit_task_training_data.jsonl"
        else:
            # Validate output path to prevent directory traversal
            validated_path = _validate_output_path(str(output_file))
            output_file = Path(validated_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        training_examples = []

        for link in self.links:
            commit = self.commits.get(link.commit_hash)
            task = self.tasks.get(link.task_id)

            if not commit or not task:
                continue

            example = {
                'commit_hash': commit.hash,
                'commit_message': commit.message,
                'task_id': task.id,
                'task_title': task.title,
                'task_description': task.description,
                'link_type': link.link_type,
                'confidence': link.confidence,
                'timestamp': commit.timestamp,
                'files_changed': commit.files_changed,
                'metadata': link.metadata
            }

            training_examples.append(example)

        # Write as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Exported {len(training_examples)} training examples to {output_file}")
        return str(output_file)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Commit-Task Linker - Link git commits to GoT tasks'
    )

    # Global options
    parser.add_argument('--max-commits', type=int,
                       default=MAX_COMMITS_TO_PROCESS,
                       help=f'Maximum number of commits to process (default: {MAX_COMMITS_TO_PROCESS})')
    parser.add_argument('--max-tasks', type=int,
                       default=MAX_TASKS_TO_MATCH,
                       help=f'Maximum number of tasks to match (default: {MAX_TASKS_TO_MATCH})')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving results')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Link command
    link_parser = subparsers.add_parser('link', help='Create all links')
    link_parser.add_argument('--semantic-threshold', type=float,
                            default=SEMANTIC_SIMILARITY_THRESHOLD,
                            help=f'Semantic similarity threshold (default: {SEMANTIC_SIMILARITY_THRESHOLD})')
    link_parser.add_argument('--file-threshold', type=float,
                            default=FILE_OVERLAP_THRESHOLD,
                            help=f'File overlap threshold (default: {FILE_OVERLAP_THRESHOLD})')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show links for task or commit')
    show_parser.add_argument('identifier', type=str,
                            help='Task ID (T-XXXX) or commit hash')
    show_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed information')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export training data')
    export_parser.add_argument('--output', '-o', type=str,
                              help='Output file path')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')

    args = parser.parse_args()

    # Create linker instance with resource limits
    linker = CommitTaskLinker(
        max_commits=args.max_commits,
        max_tasks=args.max_tasks
    )

    if args.command == 'link':
        # Load data and create links
        linker.load_tasks()
        linker.load_commits()

        # Run linking but modify link_all to support dry-run
        if args.dry_run:
            logger.info("DRY RUN MODE - Results will not be saved")

        # Clear existing links
        linker.links.clear()
        linker._commit_to_tasks.clear()
        linker._task_to_commits.clear()

        # Run linking strategies
        explicit_count = linker.link_explicit_references()
        semantic_count = linker.link_semantic_similarity(args.semantic_threshold)
        file_count = linker.link_file_overlap(args.file_threshold)

        counts = {
            'explicit': explicit_count,
            'semantic': semantic_count,
            'file_overlap': file_count,
            'total': len(linker.links)
        }

        # Save links unless dry-run
        if not args.dry_run:
            linker._save_links()

        print(f"\nLinking complete!")
        print(f"  Explicit references:  {counts['explicit']}")
        print(f"  Semantic similarity:  {counts['semantic']}")
        print(f"  File overlap:         {counts['file_overlap']}")
        print(f"  Total links:          {counts['total']}")

        if args.dry_run:
            print(f"\nDRY RUN - Results not saved")
        else:
            print(f"\nLinks saved to: {linker.links_file}")

    elif args.command == 'show':
        identifier = args.identifier

        # Load data
        linker.load_tasks()
        linker.load_commits()

        # Determine if it's a task ID or commit hash
        if identifier.startswith('T-'):
            # Show commits for task
            task = linker.tasks.get(identifier)
            if not task:
                print(f"Task not found: {identifier}")
                return 1

            print(f"\nTask: {identifier}")
            print(f"Title: {task.title}")
            print(f"Status: {task.status}")
            print(f"\nLinked commits:")
            print("-" * 80)

            links = linker.get_links_for_task(identifier)
            if not links:
                print("  No linked commits found")
            else:
                for commit, link in links:
                    print(f"\n  {commit.hash[:12]} - {link.link_type} (confidence: {link.confidence:.2f})")
                    print(f"  {commit.message[:70]}")
                    if args.verbose:
                        print(f"    Files changed: {len(commit.files_changed)}")
                        print(f"    Timestamp: {commit.timestamp}")
                        if link.metadata:
                            print(f"    Metadata: {json.dumps(link.metadata, indent=6)}")

        else:
            # Show tasks for commit
            links = linker.get_links_for_commit(identifier)
            if not links:
                print(f"Commit not found or no linked tasks: {identifier}")
                return 1

            # Get full commit info
            matching_hashes = [h for h in linker.commits.keys() if h.startswith(identifier)]
            commit = linker.commits.get(matching_hashes[0]) if matching_hashes else None

            if commit:
                print(f"\nCommit: {commit.hash[:12]}")
                print(f"Message: {commit.message[:70]}")
                print(f"\nLinked tasks:")
                print("-" * 80)

            for task, link in links:
                print(f"\n  {task.id} - {link.link_type} (confidence: {link.confidence:.2f})")
                print(f"  {task.title}")
                if args.verbose:
                    print(f"    Status: {task.status}")
                    print(f"    Category: {task.category}")
                    if link.metadata:
                        print(f"    Metadata: {json.dumps(link.metadata, indent=6)}")

    elif args.command == 'export':
        # Load data
        linker.load_tasks()
        linker.load_commits()

        output_path = Path(args.output) if args.output else None
        result_path = linker.export_training_data(output_path)

        print(f"\nTraining data exported to: {result_path}")
        print(f"Total examples: {len(linker.links)}")

    elif args.command == 'stats':
        # Load data
        linker.load_tasks()
        linker.load_commits()

        stats = linker.get_stats()

        print("\n" + "=" * 80)
        print("COMMIT-TASK LINKAGE STATISTICS")
        print("=" * 80)
        print(f"\nData:")
        print(f"  Total commits:        {stats.total_commits}")
        print(f"  Total tasks:          {stats.total_tasks}")
        print(f"\nLinks:")
        print(f"  Total links:          {stats.total_links}")
        print(f"  Explicit references:  {stats.explicit_links}")
        print(f"  Semantic similarity:  {stats.semantic_links}")
        print(f"  File overlap:         {stats.file_overlap_links}")
        print(f"\nCoverage:")
        print(f"  Commits with links:   {stats.commits_with_links} ({100*stats.commits_with_links/stats.total_commits:.1f}%)")
        print(f"  Tasks with links:     {stats.tasks_with_links} ({100*stats.tasks_with_links/stats.total_tasks:.1f}%)")
        print(f"\nAverages:")
        print(f"  Links per commit:     {stats.avg_links_per_commit:.2f}")
        print(f"  Links per task:       {stats.avg_links_per_task:.2f}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
