#!/usr/bin/env python3
"""
Auto Learning Capture Script

Scans recently completed/blocked tasks and automatically captures
learning experiences that are missing.

This script:
- Scans for tasks completed in the last 24 hours (configurable)
- Checks which tasks have learning experiences already captured
- Auto-captures missing experiences with available metadata
- Reports statistics on learning capture rate

Usage:
    # Scan and show what would be captured (dry-run)
    python scripts/auto_learning_capture.py scan

    # Actually capture missing experiences
    python scripts/auto_learning_capture.py capture

    # Show learning statistics
    python scripts/auto_learning_capture.py stats

    # Scan last 7 days instead of 24 hours
    python scripts/auto_learning_capture.py scan --days 7

    # Auto-extract patterns after capture
    python scripts/auto_learning_capture.py capture --extract-patterns
"""

import json
import logging
import sys
import fcntl
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got.learning_integration import GoTLearningBridge

# Define GOT_DIR
GOT_DIR = PROJECT_ROOT / ".got"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoLearningCapture:
    """Automatic learning capture from completed tasks."""

    # Task ID validation pattern
    TASK_ID_PATTERN = re.compile(r'^T-\d{8}-\d{6}-[a-f0-9]{8}$')

    # Max values for safety
    MAX_BATCH_SIZE = 100
    MAX_DAYS = 365

    def __init__(self, got_dir: Path = GOT_DIR):
        """Initialize the auto-capture system."""
        self.got_dir = Path(got_dir).resolve()  # Resolve to absolute path
        self._validate_path(self.got_dir, "got_dir")

        self.entities_dir = self.got_dir / "entities"
        self.learning_dir = self.got_dir / "learning"
        self.locks_dir = self.got_dir / "locks"
        self.bridge = GoTLearningBridge(self.got_dir)

        # Ensure directories exist
        self.entities_dir.mkdir(parents=True, exist_ok=True)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"AutoLearningCapture initialized with {self.got_dir}")

    def _validate_path(self, path: Path, name: str) -> None:
        """
        Validate that path is within expected directories (prevent traversal).

        Args:
            path: Path to validate
            name: Name of the path (for error messages)

        Raises:
            ValueError: If path is invalid or outside allowed directories
        """
        resolved = path.resolve()

        # Must be within PROJECT_ROOT
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            raise ValueError(f"{name} must be within project root: {resolved}")

    def _validate_task_id(self, task_id: str) -> None:
        """
        Validate task ID format.

        Args:
            task_id: Task ID to validate

        Raises:
            ValueError: If task ID format is invalid
        """
        if not task_id:
            raise ValueError("task_id cannot be empty")

        if not self.TASK_ID_PATTERN.match(task_id):
            raise ValueError(
                f"Invalid task_id format: {task_id}. "
                f"Expected format: T-YYYYMMDD-HHMMSS-xxxxxxxx"
            )

    def _validate_days(self, days: int) -> None:
        """
        Validate days parameter.

        Args:
            days: Number of days to validate

        Raises:
            ValueError: If days is invalid
        """
        if days < 0:
            raise ValueError("days must be non-negative")
        if days > self.MAX_DAYS:
            raise ValueError(f"days cannot exceed {self.MAX_DAYS}")

    def _with_lock(self, task_id: str):
        """
        Get exclusive lock for a task to prevent race conditions.

        Args:
            task_id: Task ID to lock

        Returns:
            File handle with exclusive lock
        """
        self._validate_task_id(task_id)

        lock_path = self.locks_dir / f"{task_id}.lock"
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return lock_file

    def _get_files_changed(self, task_id: str) -> List[str]:
        """
        Try to get files changed for a task from git history.

        Args:
            task_id: Task ID to get files for

        Returns:
            List of file paths changed, or empty list if not available
        """
        try:
            # Try to find commits mentioning this task
            result = subprocess.run(
                ['git', 'log', '--all', '--oneline', '--grep', task_id],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0 or not result.stdout.strip():
                return []

            # Get the first commit hash
            lines = result.stdout.strip().split('\n')
            if not lines:
                return []

            commit_hash = lines[0].split()[0]

            # Get files changed in that commit
            result = subprocess.run(
                ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                return files[:50]  # Limit to 50 files

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Could not get files changed for {task_id}: {e}")

        return []

    def get_recent_tasks(
        self,
        days: int = 1,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tasks from recent time period.

        Args:
            days: Number of days to look back
            status_filter: Only include tasks with these statuses (e.g., ['completed', 'blocked'])

        Returns:
            List of task dictionaries
        """
        # Validate input
        self._validate_days(days)

        if not self.entities_dir.exists():
            logger.warning(f"Entities directory not found: {self.entities_dir}")
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent_tasks = []

        # Scan task files (T-*.json)
        for task_file in self.entities_dir.glob("T-*.json"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                # GoT transactional format: data.metadata.updated_at or data.modified_at
                data_section = task_data.get('data', {})

                # Try metadata.updated_at first (completion time)
                updated_at_str = data_section.get('metadata', {}).get('updated_at')
                if not updated_at_str:
                    # Fall back to modified_at
                    updated_at_str = data_section.get('modified_at')
                if not updated_at_str:
                    # Fall back to created_at
                    updated_at_str = data_section.get('created_at')

                if not updated_at_str:
                    continue

                # Parse timestamp
                try:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    continue

                # Check if recent enough
                if updated_at < cutoff_time:
                    continue

                # Check status filter
                if status_filter:
                    status = data_section.get('status', '')
                    if status not in status_filter:
                        continue

                # Extract task ID from filename
                task_id = task_file.stem

                # Build task summary
                task_summary = {
                    'id': task_id,
                    'title': data_section.get('title', ''),
                    'status': data_section.get('status', 'unknown'),
                    'category': data_section.get('properties', {}).get('category', 'general'),
                    'priority': data_section.get('priority', 'medium'),
                    'retrospective': data_section.get('properties', {}).get('retrospective', ''),
                    'updated_at': updated_at,
                    'file_path': str(task_file),
                }

                # Calculate duration if both timestamps available
                created_at_str = data_section.get('created_at')
                completed_at_str = data_section.get('metadata', {}).get('completed_at')

                if created_at_str and completed_at_str:
                    try:
                        created = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        completed = datetime.fromisoformat(completed_at_str.replace('Z', '+00:00'))
                        duration = (completed - created).total_seconds()
                        task_summary['duration_seconds'] = duration
                    except (ValueError, AttributeError):
                        pass

                recent_tasks.append(task_summary)

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read task file {task_file}: {e}")
                continue

        # Sort by update time (newest first)
        recent_tasks.sort(key=lambda t: t['updated_at'], reverse=True)

        return recent_tasks

    def get_captured_task_ids(self) -> Set[str]:
        """
        Get set of task IDs that already have learning experiences.

        Returns:
            Set of task IDs (e.g., {'T-123', 'T-456'})
        """
        captured_ids = set()

        experiences_dir = self.learning_dir / "experiences"
        if not experiences_dir.exists():
            return captured_ids

        # Scan experience files
        for exp_file in experiences_dir.glob("*.json"):
            try:
                with open(exp_file, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)

                # Look for task:T-XXX tag
                tags = exp_data.get('tags', [])
                for tag in tags:
                    if tag.startswith('task:'):
                        task_id = tag.split(':', 1)[1]
                        captured_ids.add(task_id)

            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Failed to read experience file {exp_file}: {e}")
                continue

        return captured_ids

    def scan_missing_captures(
        self,
        days: int = 1,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan for tasks that are missing learning captures.

        Args:
            days: Number of days to look back
            status_filter: Only check tasks with these statuses

        Returns:
            List of tasks that need learning capture
        """
        # Validate input
        self._validate_days(days)

        logger.info(f"Scanning for tasks in last {days} day(s)...")

        # Get recent tasks (this will validate days again, but that's ok)
        recent_tasks = self.get_recent_tasks(days=days, status_filter=status_filter)
        logger.info(f"Found {len(recent_tasks)} recent tasks")

        # Get already captured task IDs
        captured_ids = self.get_captured_task_ids()
        logger.info(f"Already captured: {len(captured_ids)} tasks")

        # Find missing
        missing = []
        for task in recent_tasks:
            if task['id'] not in captured_ids:
                missing.append(task)

        logger.info(f"Missing captures: {len(missing)} tasks")
        return missing

    def capture_missing(
        self,
        tasks: List[Dict[str, Any]],
        dry_run: bool = False,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Capture learning from missing tasks.

        Args:
            tasks: List of tasks to capture
            dry_run: If True, don't actually capture (just report)
            batch_size: Number of tasks to process in one batch

        Returns:
            Dictionary with counts and errors: {'captured': N, 'failed': M, 'errors': [...]}
        """
        # Validate batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(f"batch_size cannot exceed {self.MAX_BATCH_SIZE}")

        results = {
            'captured': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }

        # Process tasks in batches
        for i, task in enumerate(tasks):
            if i >= batch_size:
                logger.info(f"Reached batch limit ({batch_size}), stopping")
                break

            task_id = task.get('id', '')
            status = task.get('status', '')

            # Validate task_id
            try:
                self._validate_task_id(task_id)
            except ValueError as e:
                logger.error(f"Invalid task ID '{task_id}': {e}")
                results['failed'] += 1
                results['errors'].append({
                    'task_id': task_id,
                    'error': str(e),
                    'type': 'validation'
                })
                continue

            lock_file = None
            try:
                if dry_run:
                    logger.info(f"[DRY-RUN] Would capture {task_id} ({status})")
                    results['captured'] += 1
                    continue

                # Acquire lock to prevent race condition
                lock_file = self._with_lock(task_id)

                # Double-check if already captured (race condition check)
                captured_ids = self.get_captured_task_ids()
                if task_id in captured_ids:
                    logger.info(f"Task {task_id} already captured (skipping)")
                    results['skipped'] += 1
                    continue

                # Get files changed from git
                files_changed = self._get_files_changed(task_id)
                if files_changed:
                    logger.debug(f"Found {len(files_changed)} files changed for {task_id}")

                # Capture based on status
                if status == 'completed':
                    # Capture success
                    experience = self.bridge.capture_task_completion(
                        task_id=task_id,
                        retrospective=task.get('retrospective', ''),
                        files_changed=files_changed,
                        task_title=task.get('title', ''),
                        task_category=task.get('category', 'general'),
                        task_priority=task.get('priority', 'medium'),
                        duration_seconds=task.get('duration_seconds'),
                    )
                    logger.info(f"Captured completion: {task_id} -> {experience.id}")
                    results['captured'] += 1

                elif status == 'blocked':
                    # Capture failure
                    # Try to infer error from retrospective
                    error_msg = task.get('retrospective', 'Task blocked')
                    experience = self.bridge.capture_task_failure(
                        task_id=task_id,
                        error_message=error_msg,
                        task_title=task.get('title', ''),
                        task_category=task.get('category', 'general'),
                        task_priority=task.get('priority', 'medium'),
                    )
                    logger.warning(f"Captured failure: {task_id} -> {experience.id}")
                    results['captured'] += 1

                else:
                    # Unknown status, skip
                    logger.debug(f"Skipping {task_id} with status '{status}'")
                    results['skipped'] += 1

            except Exception as e:
                logger.error(f"Failed to capture {task_id}: {e}", exc_info=True)
                results['failed'] += 1
                results['errors'].append({
                    'task_id': task_id,
                    'error': str(e),
                    'type': type(e).__name__
                })
                # Continue processing other tasks

            finally:
                # Release lock
                if lock_file:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        lock_file.close()
                    except Exception as e:
                        logger.warning(f"Failed to release lock for {task_id}: {e}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self.bridge.get_learning_stats()

    def extract_patterns(self) -> Dict[str, int]:
        """Run pattern extraction."""
        return self.bridge.extract_patterns_and_lessons()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto Learning Capture Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Scan command
    scan_parser = subparsers.add_parser(
        'scan',
        help='Scan for tasks missing learning captures (dry-run)'
    )
    scan_parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to look back (default: 1)'
    )
    scan_parser.add_argument(
        '--status',
        choices=['completed', 'blocked', 'in_progress', 'pending'],
        nargs='*',
        default=['completed', 'blocked'],
        help='Task statuses to scan (default: completed blocked)'
    )

    # Capture command
    capture_parser = subparsers.add_parser(
        'capture',
        help='Capture learning from missing tasks'
    )
    capture_parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to look back (default: 1)'
    )
    capture_parser.add_argument(
        '--status',
        choices=['completed', 'blocked', 'in_progress', 'pending'],
        nargs='*',
        default=['completed', 'blocked'],
        help='Task statuses to capture (default: completed blocked)'
    )
    capture_parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Maximum tasks to process in one run (default: 50, max: 100)'
    )
    capture_parser.add_argument(
        '--extract-patterns',
        action='store_true',
        help='Run pattern extraction after capture'
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Show learning statistics'
    )
    stats_parser.add_argument(
        '--extract-patterns',
        action='store_true',
        help='Also run pattern extraction'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize auto-capture
    auto_capture = AutoLearningCapture()

    if args.command == 'scan':
        # Validate arguments
        try:
            auto_capture._validate_days(args.days)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        # Scan for missing captures
        missing = auto_capture.scan_missing_captures(
            days=args.days,
            status_filter=args.status
        )

        if missing:
            print(f"\n{'='*70}")
            print(f"MISSING LEARNING CAPTURES ({len(missing)} tasks)")
            print(f"{'='*70}")
            for task in missing:
                duration_str = ""
                if 'duration_seconds' in task:
                    hours = task['duration_seconds'] / 3600
                    duration_str = f" ({hours:.1f}h)"

                print(f"  {task['id']} ({task['status']}) - {task['title'][:50]}{duration_str}")
            print(f"{'='*70}")
            print(f"\nRun 'python scripts/auto_learning_capture.py capture' to capture these.")
        else:
            print("✅ All recent tasks have learning captures!")

        return 0

    elif args.command == 'capture':
        # Validate arguments
        try:
            auto_capture._validate_days(args.days)
            if args.batch_size <= 0:
                print(f"Error: batch-size must be positive")
                return 1
            if args.batch_size > AutoLearningCapture.MAX_BATCH_SIZE:
                print(f"Error: batch-size cannot exceed {AutoLearningCapture.MAX_BATCH_SIZE}")
                return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        # Scan and capture
        missing = auto_capture.scan_missing_captures(
            days=args.days,
            status_filter=args.status
        )

        if not missing:
            print("✅ All recent tasks already have learning captures!")
            return 0

        print(f"Capturing learning from {len(missing)} tasks...")
        results = auto_capture.capture_missing(
            missing,
            dry_run=False,
            batch_size=args.batch_size
        )

        print(f"\n{'='*70}")
        print("CAPTURE RESULTS")
        print(f"{'='*70}")
        print(f"  Captured:  {results['captured']}")
        print(f"  Skipped:   {results['skipped']}")
        print(f"  Failed:    {results['failed']}")
        print(f"{'='*70}")

        # Show errors if any
        if results['errors']:
            print(f"\nERRORS ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10
                print(f"  {error['task_id']}: {error['error'][:80]}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")

        # Extract patterns if requested
        if args.extract_patterns and results['captured'] > 0:
            print("\nExtracting patterns and distilling lessons...")
            pattern_results = auto_capture.extract_patterns()
            print(f"  Lessons:           {pattern_results.get('lessons', 0)}")
            print(f"  Sequence patterns: {pattern_results.get('sequence_patterns', 0)}")
            print(f"  Strategy patterns: {pattern_results.get('strategy_patterns', 0)}")
            print(f"  Anti-patterns:     {pattern_results.get('antipatterns', 0)}")

        return 0 if results['failed'] == 0 else 1

    elif args.command == 'stats':
        # Show statistics
        stats = auto_capture.get_stats()

        if stats:
            print(f"\n{'='*70}")
            print("LEARNING STATISTICS")
            print(f"{'='*70}")
            print(f"Experiences:       {stats.get('total_experiences', 0)}")
            print(f"  Successes:       {stats.get('successes', 0)}")
            print(f"  Failures:        {stats.get('failures', 0)}")
            print(f"  Partial:         {stats.get('partial_successes', 0)}")
            print(f"\nPatterns:          {stats.get('total_patterns', 0)}")
            print(f"  Sequence:        {stats.get('sequence_patterns', 0)}")
            print(f"  Strategy:        {stats.get('strategy_patterns', 0)}")
            print(f"  Anti-patterns:   {stats.get('antipatterns', 0)}")
            print(f"\nLessons:           {stats.get('total_lessons', 0)}")

            # Show recent activity
            recent_tasks = auto_capture.get_recent_tasks(days=7, status_filter=['completed', 'blocked'])
            captured_ids = auto_capture.get_captured_task_ids()
            capture_rate = len(captured_ids) / max(len(recent_tasks), 1) * 100

            print(f"\nCapture Rate (7d): {capture_rate:.1f}%")
            print(f"  Recent tasks:    {len(recent_tasks)}")
            print(f"  Captured:        {len([t for t in recent_tasks if t['id'] in captured_ids])}")

            print(f"{'='*70}")
        else:
            print("No learning statistics available")

        # Extract patterns if requested
        if args.extract_patterns:
            print("\nExtracting patterns and distilling lessons...")
            pattern_results = auto_capture.extract_patterns()
            if pattern_results:
                print(f"  Extracted {pattern_results.get('lessons', 0)} lessons")
                print(f"  Found {pattern_results.get('sequence_patterns', 0)} sequence patterns")

        return 0

    return 1


if __name__ == '__main__':
    sys.exit(main())
