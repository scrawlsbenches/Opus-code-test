#!/usr/bin/env python3
"""
Generate session context summaries for new agent sessions.

This script provides a "catch-up" summary from task files and git history,
reducing cold-start time for new agents joining the repository.

Usage:
    # Print context to stdout
    python scripts/session_context.py

    # Output JSON format
    python scripts/session_context.py --json

    # Custom time range
    python scripts/session_context.py --days 14

    # Save to file
    python scripts/session_context.py --output context.md

Example Output:
    # Session Context (generated 2025-12-14 01:30:00)

    ## Recent Work (last 5 sessions)
    - [e233] completed: 2 tasks, pending: 1 tasks
    - [2d89] completed: 1 tasks, pending: 0 tasks

    ## Pending Tasks (by priority)
    ### High
    - T-xxxx-001: Task title

    ## Recent Changes (last 7 days)
    - cortical/: 5 files modified
    - scripts/: 3 files added

    ## Recent Commits
    - abc1234: Fix validation bug (2 hours ago)
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from task_utils import (
    Task, TaskSession, load_all_tasks, DEFAULT_TASKS_DIR
)


class SessionContextGenerator:
    """Generate catch-up summaries from task files and git history."""

    def __init__(self, repo_path: str = '.'):
        """
        Initialize the context generator.

        Args:
            repo_path: Path to git repository root
        """
        self.repo_path = Path(repo_path)
        self.tasks_dir = self.repo_path / DEFAULT_TASKS_DIR

    def get_recent_sessions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get N most recent task sessions from tasks/*.json files.

        Args:
            n: Number of recent sessions to retrieve

        Returns:
            List of session summaries with counts by status
        """
        if not self.tasks_dir.exists():
            return []

        sessions = []
        # Sort by filename (newest first). Filenames include the started_at timestamp
        # (format: YYYY-MM-DD_HH-MM-SS_session_id.json), making this deterministic.
        # This is more reliable than mtime which can be non-deterministic in fast loops.
        session_files = sorted(
            self.tasks_dir.glob("*.json"),
            key=lambda p: p.name,
            reverse=True
        )

        for filepath in session_files[:n]:
            try:
                session = TaskSession.load(filepath)

                # Count tasks by status
                status_counts = defaultdict(int)
                for task in session.tasks:
                    status_counts[task.status] += 1

                sessions.append({
                    'session_id': session.session_id,
                    'started_at': session.started_at,
                    'filename': filepath.name,
                    'total': len(session.tasks),
                    'completed': status_counts.get('completed', 0),
                    'in_progress': status_counts.get('in_progress', 0),
                    'pending': status_counts.get('pending', 0),
                    'deferred': status_counts.get('deferred', 0),
                })
            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted files
                continue

        return sessions

    def get_pending_tasks(self) -> Dict[str, List[Task]]:
        """
        Get all pending tasks grouped by priority.

        Returns:
            Dict of {priority: [tasks]} sorted within each priority
        """
        all_tasks = load_all_tasks(str(self.tasks_dir))

        # Group pending and in_progress tasks by priority
        by_priority = {
            'high': [],
            'medium': [],
            'low': []
        }

        for task in all_tasks:
            if task.status in ('pending', 'in_progress'):
                priority = task.priority if task.priority in by_priority else 'medium'
                by_priority[priority].append(task)

        # Sort each priority group by creation time
        for priority in by_priority:
            by_priority[priority].sort(key=lambda t: t.created_at)

        return by_priority

    def get_recent_commits(self, n: int = 10) -> List[Dict[str, str]]:
        """
        Get N recent commits with metadata.

        Args:
            n: Number of commits to retrieve

        Returns:
            List of commit dicts with hash, time_ago, subject, files_changed
        """
        try:
            # Get commit info with relative time
            result = subprocess.run(
                ['git', 'log', f'--max-count={n}', '--format=%h|%ar|%s', '--no-merges'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split('|', 2)
                if len(parts) == 3:
                    commit_hash, time_ago, subject = parts

                    # Get files changed for this commit
                    files_result = subprocess.run(
                        ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    files = [f for f in files_result.stdout.strip().split('\n') if f]

                    commits.append({
                        'hash': commit_hash,
                        'time_ago': time_ago,
                        'subject': subject,
                        'files_changed': len(files),
                        'files': files
                    })

            return commits

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git not available or not a git repo
            return []

    def get_recent_file_changes(self, days: int = 7) -> Dict[str, List[str]]:
        """
        Get files changed in last N days grouped by directory.

        Args:
            days: Number of days to look back

        Returns:
            Dict of {directory: [filenames]} for files changed
        """
        try:
            since_date = datetime.now() - timedelta(days=days)
            since_str = since_date.strftime('%Y-%m-%d')

            result = subprocess.run(
                ['git', 'log', f'--since={since_str}', '--name-only', '--format=', '--no-merges'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            # Group by directory
            by_dir = defaultdict(set)
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                path = Path(line)
                # Use first directory component or 'root' if in root
                if len(path.parts) > 1:
                    directory = path.parts[0]
                else:
                    directory = 'root'

                by_dir[directory].add(line)

            # Convert sets to sorted lists
            return {dir: sorted(files) for dir, files in by_dir.items()}

        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}

    def _count_file_changes(self, changes: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Count files by type (added/modified/deleted) per directory.

        This requires comparing with previous state, which is complex.
        For now, we just count total files changed per directory.

        Args:
            changes: Dict from get_recent_file_changes()

        Returns:
            Dict of {directory: {'total': count}}
        """
        return {
            directory: {'total': len(files)}
            for directory, files in changes.items()
        }

    def generate_context(
        self,
        sessions: int = 5,
        commits: int = 10,
        days: int = 7
    ) -> str:
        """
        Generate full markdown context summary.

        Args:
            sessions: Number of recent sessions to include
            commits: Number of recent commits to include
            days: Number of days for file change analysis

        Returns:
            Markdown-formatted context string
        """
        lines = []
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines.append(f"# Session Context (generated {now})")
        lines.append("")

        # Recent sessions
        recent_sessions = self.get_recent_sessions(sessions)
        if recent_sessions:
            lines.append(f"## Recent Work (last {len(recent_sessions)} sessions)")
            lines.append("")
            for session in recent_sessions:
                started = datetime.fromisoformat(session['started_at']).strftime('%Y-%m-%d %H:%M')
                lines.append(
                    f"- [{session['session_id']}] {started}: "
                    f"{session['completed']} completed, "
                    f"{session['in_progress']} in progress, "
                    f"{session['pending']} pending"
                )
            lines.append("")
        else:
            lines.append("## Recent Work")
            lines.append("")
            lines.append("No task sessions found.")
            lines.append("")

        # Pending tasks by priority
        pending_by_priority = self.get_pending_tasks()
        total_pending = sum(len(tasks) for tasks in pending_by_priority.values())

        if total_pending > 0:
            lines.append(f"## Pending Tasks ({total_pending} total)")
            lines.append("")

            for priority in ['high', 'medium', 'low']:
                tasks = pending_by_priority.get(priority, [])
                if tasks:
                    lines.append(f"### {priority.capitalize()} Priority")
                    lines.append("")
                    for task in tasks[:10]:  # Limit to 10 per priority
                        status_marker = "🔄" if task.status == 'in_progress' else "📋"
                        lines.append(f"- {status_marker} **{task.id}**: {task.title}")
                        if task.description:
                            # Show first 100 chars of description
                            desc = task.description[:100]
                            if len(task.description) > 100:
                                desc += "..."
                            lines.append(f"  {desc}")
                    lines.append("")
        else:
            lines.append("## Pending Tasks")
            lines.append("")
            lines.append("No pending tasks found.")
            lines.append("")

        # Recent file changes
        file_changes = self.get_recent_file_changes(days)
        if file_changes:
            lines.append(f"## Recent Changes (last {days} days)")
            lines.append("")

            # Sort directories by number of changes
            sorted_dirs = sorted(
                file_changes.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )

            for directory, files in sorted_dirs[:10]:  # Top 10 directories
                lines.append(f"- **{directory}**/: {len(files)} files changed")
            lines.append("")

        # Recent commits
        recent_commits = self.get_recent_commits(commits)
        if recent_commits:
            lines.append(f"## Recent Commits (last {len(recent_commits)})")
            lines.append("")
            for commit in recent_commits:
                lines.append(
                    f"- `{commit['hash']}`: {commit['subject']} "
                    f"({commit['time_ago']}, {commit['files_changed']} files)"
                )
            lines.append("")

        # Summary stats
        lines.append("## Quick Stats")
        lines.append("")
        lines.append(f"- Task sessions: {len(recent_sessions)}")
        lines.append(f"- Pending tasks: {total_pending}")
        lines.append(f"- Recent commits: {len(recent_commits)}")
        lines.append(f"- Directories changed: {len(file_changes)}")
        lines.append("")

        return '\n'.join(lines)

    def generate_json(
        self,
        sessions: int = 5,
        commits: int = 10,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate context as JSON structure.

        Args:
            sessions: Number of recent sessions to include
            commits: Number of recent commits to include
            days: Number of days for file change analysis

        Returns:
            Dict with all context data
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'recent_sessions': self.get_recent_sessions(sessions),
            'pending_tasks': {
                priority: [t.to_dict() for t in tasks]
                for priority, tasks in self.get_pending_tasks().items()
            },
            'recent_commits': self.get_recent_commits(commits),
            'recent_file_changes': self.get_recent_file_changes(days),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate session context for new agent sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print context to stdout
  python scripts/session_context.py

  # Output as JSON
  python scripts/session_context.py --json

  # Analyze last 14 days
  python scripts/session_context.py --days 14

  # Save to file
  python scripts/session_context.py --output context.md
        """
    )

    parser.add_argument(
        '--json', action='store_true',
        help='Output as JSON instead of markdown'
    )
    parser.add_argument(
        '--days', type=int, default=7,
        help='Number of days for file change analysis (default: 7)'
    )
    parser.add_argument(
        '--sessions', type=int, default=5,
        help='Number of recent sessions to include (default: 5)'
    )
    parser.add_argument(
        '--commits', type=int, default=10,
        help='Number of recent commits to include (default: 10)'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--repo-path', default='.',
        help='Path to git repository (default: current directory)'
    )

    args = parser.parse_args()

    # Generate context
    generator = SessionContextGenerator(args.repo_path)

    if args.json:
        context = generator.generate_json(
            sessions=args.sessions,
            commits=args.commits,
            days=args.days
        )
        output = json.dumps(context, indent=2)
    else:
        output = generator.generate_context(
            sessions=args.sessions,
            commits=args.commits,
            days=args.days
        )

    # Write to file or stdout
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output)
        print(f"Context written to: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
