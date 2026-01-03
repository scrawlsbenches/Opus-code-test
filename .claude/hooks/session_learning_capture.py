#!/usr/bin/env python3
"""
Session Learning Capture Hook

Automatically captures learning data during Claude Code sessions.
Called when tasks are completed, blocked, or abandoned to feed data
into the GoTLearningBridge.

This hook bridges the gap between task management (GoT) and learning
systems (LearningCycle) to continuously improve AI agent performance.

Usage:
    # Auto-capture on task completion
    python .claude/hooks/session_learning_capture.py complete T-123 \
        --retrospective "TDD approach worked well" \
        --files cortical/api.py tests/test_api.py

    # Capture failure when task blocked
    python .claude/hooks/session_learning_capture.py failure T-123 \
        --error "Missing test fixtures" \
        --blockers "Need test data setup"

    # Generate periodic stats
    python .claude/hooks/session_learning_capture.py stats
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got.learning_integration import GoTLearningBridge

# Define GOT_DIR
GOT_DIR = PROJECT_ROOT / ".got"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SessionLearningCapture:
    """Handles automatic learning capture during sessions."""

    def __init__(self, got_dir: Path = GOT_DIR):
        """Initialize the capture system."""
        self.got_dir = Path(got_dir)
        self.bridge = GoTLearningBridge(self.got_dir)
        logger.info(f"SessionLearningCapture initialized with {self.got_dir}")

    def capture_completion(
        self,
        task_id: str,
        retrospective: str = "",
        files_changed: Optional[List[str]] = None,
        approach: Optional[str] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Capture task completion as learning experience.

        Args:
            task_id: Task identifier
            retrospective: Completion notes/lessons learned
            files_changed: Files modified during task
            approach: Strategy used (e.g., "test-first")
            task_metadata: Additional task info (title, category, priority, duration)

        Returns:
            True if capture succeeded, False otherwise
        """
        try:
            metadata = task_metadata or {}

            # Extract task metadata
            task_title = metadata.get('title', '')
            task_category = metadata.get('category', 'general')
            task_priority = metadata.get('priority', 'medium')
            duration_seconds = metadata.get('duration_seconds')

            # Capture via bridge
            experience = self.bridge.capture_task_completion(
                task_id=task_id,
                retrospective=retrospective,
                files_changed=files_changed or [],
                approach=approach,
                task_title=task_title,
                task_category=task_category,
                task_priority=task_priority,
                duration_seconds=duration_seconds,
            )

            logger.info(
                f"Captured task completion: {task_id} -> {experience.id} "
                f"({len(files_changed or [])} files)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to capture task completion {task_id}: {e}")
            return False

    def capture_failure(
        self,
        task_id: str,
        error_message: str,
        attempted_approach: Optional[str] = None,
        files_attempted: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Capture task failure as learning experience.

        Args:
            task_id: Task identifier
            error_message: What went wrong
            attempted_approach: Strategy that was tried
            files_attempted: Files that were attempted to be modified
            blockers: Blocking issues
            task_metadata: Additional task info

        Returns:
            True if capture succeeded, False otherwise
        """
        try:
            metadata = task_metadata or {}

            task_title = metadata.get('title', '')
            task_category = metadata.get('category', 'general')
            task_priority = metadata.get('priority', 'medium')

            experience = self.bridge.capture_task_failure(
                task_id=task_id,
                error_message=error_message,
                attempted_approach=attempted_approach,
                task_title=task_title,
                task_category=task_category,
                task_priority=task_priority,
                files_attempted=files_attempted or [],
                blockers=blockers or [],
            )

            logger.warning(
                f"Captured task failure: {task_id} -> {experience.id} "
                f"(blockers: {len(blockers or [])})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to capture task failure {task_id}: {e}")
            return False

    def generate_stats(self) -> Dict[str, Any]:
        """
        Generate learning statistics.

        Returns:
            Dictionary with experience counts, pattern counts, etc.
        """
        try:
            stats = self.bridge.get_learning_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to generate stats: {e}")
            return {}

    def extract_patterns(self) -> Dict[str, int]:
        """
        Run pattern extraction and lesson distillation.

        Should be called periodically (e.g., after every 10 completions).

        Returns:
            Dictionary with counts of patterns and lessons extracted
        """
        try:
            results = self.bridge.extract_patterns_and_lessons()
            logger.info(
                f"Pattern extraction complete: "
                f"{results.get('lessons', 0)} lessons, "
                f"{results.get('sequence_patterns', 0)} sequence patterns"
            )
            return results
        except Exception as e:
            logger.error(f"Failed to extract patterns: {e}")
            return {}


def load_task_metadata(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Load task metadata from GoT storage.

    Args:
        task_id: Task identifier

    Returns:
        Task metadata dictionary or None if not found
    """
    try:
        entities_dir = GOT_DIR / "entities"
        task_file = entities_dir / f"{task_id}.json"

        if not task_file.exists():
            logger.warning(f"Task file not found: {task_file}")
            return None

        with open(task_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        # GoT transactional format: data section
        data_section = task_data.get('data', {})

        # Extract relevant metadata
        metadata = {
            'title': data_section.get('title', ''),
            'category': data_section.get('properties', {}).get('category', 'general'),
            'priority': data_section.get('priority', 'medium'),
        }

        # Calculate duration if timestamps available
        created_at = data_section.get('created_at')
        completed_at = data_section.get('metadata', {}).get('completed_at')

        if created_at and completed_at:
            try:
                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                completed = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                duration = (completed - created).total_seconds()
                metadata['duration_seconds'] = duration
            except (ValueError, AttributeError):
                pass

        return metadata

    except Exception as e:
        logger.error(f"Failed to load task metadata for {task_id}: {e}")
        return None


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Session Learning Capture Hook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Capture task completion')
    complete_parser.add_argument('task_id', help='Task ID (e.g., T-123)')
    complete_parser.add_argument('--retrospective', default='',
                                 help='Completion notes/lessons learned')
    complete_parser.add_argument('--files', nargs='*', default=[],
                                 help='Files modified during task')
    complete_parser.add_argument('--approach', help='Strategy used (e.g., test-first)')

    # Failure command
    failure_parser = subparsers.add_parser('failure', help='Capture task failure')
    failure_parser.add_argument('task_id', help='Task ID (e.g., T-123)')
    failure_parser.add_argument('--error', required=True,
                                help='Error message describing what went wrong')
    failure_parser.add_argument('--approach', help='Strategy that was attempted')
    failure_parser.add_argument('--files', nargs='*', default=[],
                                help='Files that were attempted to be modified')
    failure_parser.add_argument('--blockers', nargs='*', default=[],
                                help='Blocking issues encountered')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show learning statistics')
    stats_parser.add_argument('--extract-patterns', action='store_true',
                             help='Also run pattern extraction')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize capture system
    capture = SessionLearningCapture()

    if args.command == 'complete':
        # Load task metadata
        metadata = load_task_metadata(args.task_id)

        # Capture completion
        success = capture.capture_completion(
            task_id=args.task_id,
            retrospective=args.retrospective,
            files_changed=args.files,
            approach=args.approach,
            task_metadata=metadata,
        )

        if success:
            print(f"✅ Captured learning from task completion: {args.task_id}")
            return 0
        else:
            print(f"❌ Failed to capture learning from {args.task_id}")
            return 1

    elif args.command == 'failure':
        # Load task metadata
        metadata = load_task_metadata(args.task_id)

        # Capture failure
        success = capture.capture_failure(
            task_id=args.task_id,
            error_message=args.error,
            attempted_approach=args.approach,
            files_attempted=args.files,
            blockers=args.blockers,
            task_metadata=metadata,
        )

        if success:
            print(f"⚠️  Captured learning from task failure: {args.task_id}")
            return 0
        else:
            print(f"❌ Failed to capture failure from {args.task_id}")
            return 1

    elif args.command == 'stats':
        # Generate stats
        stats = capture.generate_stats()

        if stats:
            print("\n" + "="*70)
            print("LEARNING STATISTICS")
            print("="*70)
            print(f"Experiences:       {stats.get('total_experiences', 0)}")
            print(f"  Successes:       {stats.get('successes', 0)}")
            print(f"  Failures:        {stats.get('failures', 0)}")
            print(f"  Partial:         {stats.get('partial_successes', 0)}")
            print(f"\nPatterns:          {stats.get('total_patterns', 0)}")
            print(f"  Sequence:        {stats.get('sequence_patterns', 0)}")
            print(f"  Strategy:        {stats.get('strategy_patterns', 0)}")
            print(f"  Anti-patterns:   {stats.get('antipatterns', 0)}")
            print(f"\nLessons:           {stats.get('total_lessons', 0)}")
            print("="*70)
        else:
            print("No learning statistics available")

        # Extract patterns if requested
        if args.extract_patterns:
            print("\nExtracting patterns and distilling lessons...")
            results = capture.extract_patterns()
            if results:
                print(f"  Extracted {results.get('lessons', 0)} lessons")
                print(f"  Found {results.get('sequence_patterns', 0)} sequence patterns")
                print(f"  Found {results.get('strategy_patterns', 0)} strategy patterns")

        return 0

    return 1


if __name__ == '__main__':
    sys.exit(main())
