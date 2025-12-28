#!/usr/bin/env python3
"""
Generate session handoff documents for knowledge transfer.

Creates automatic handoff documents when ending a coding session, capturing:
- Git status and branch information
- Recently completed tasks (from GoT)
- Uncommitted changes
- Suggested next steps

Usage:
    # Generate handoff for current session
    python scripts/session_handoff.py

    # Preview without creating
    python scripts/session_handoff.py --dry-run

    # Custom output location
    python scripts/session_handoff.py --output samples/memories/handoff.md

Example:
    $ python scripts/session_handoff.py
    Created session handoff: samples/memories/session-handoff-2025-12-14_14-30-52_a1b2.md
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.utils.id_generation import generate_session_id
from cortical.got.api import GoTManager
from cortical.got.types import Task

# Default directories
MEMORIES_DIR = Path("samples/memories")
GOT_DIR = PROJECT_ROOT / ".got"


def gather_session_context() -> Dict[str, Any]:
    """
    Gather current session context from git and system.

    Returns:
        Dictionary with:
        - branch: Current git branch name
        - status_summary: Git status summary
        - uncommitted_files: List of modified/staged files
        - recent_commits: List of (hash, message) tuples for last 5 commits
        - background_processes: Optional list of running processes
    """
    context = {
        'branch': None,
        'status_summary': '',
        'uncommitted_files': [],
        'recent_commits': [],
        'background_processes': []
    }

    # Get current branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            context['branch'] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Get git status
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            context['uncommitted_files'] = [l for l in lines if l.strip()]

            # Generate summary
            if not context['uncommitted_files']:
                context['status_summary'] = "clean"
            else:
                staged = sum(1 for l in lines if l and l[0] != ' ' and l[0] != '?')
                modified = sum(1 for l in lines if l and len(l) > 1 and l[1] == 'M')
                untracked = sum(1 for l in lines if l.startswith('??'))
                parts = []
                if staged:
                    parts.append(f"{staged} staged")
                if modified:
                    parts.append(f"{modified} modified")
                if untracked:
                    parts.append(f"{untracked} untracked")
                context['status_summary'] = ", ".join(parts) if parts else "changes pending"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        context['status_summary'] = "unknown"

    # Get recent commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        context['recent_commits'].append((parts[0], parts[1]))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return context


def gather_completed_tasks(got_dir: Path = GOT_DIR) -> List[Task]:
    """
    Gather tasks completed today from GoT.

    Args:
        got_dir: GoT directory

    Returns:
        List of Task objects completed today, sorted by completion time
    """
    try:
        manager = GoTManager(got_dir)
        all_tasks = manager.list_tasks(status="completed")
    except Exception as e:
        print(f"Warning: Could not load GoT tasks: {e}", file=sys.stderr)
        return []

    # Get today's date range
    today = datetime.now().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    # Filter to completed tasks from today
    completed_today = []
    for task in all_tasks:
        if task.completed_at:
            try:
                # Handle both ISO format and other formats
                completed_str = task.completed_at
                if isinstance(completed_str, str):
                    # Remove timezone info if present for comparison
                    if completed_str.endswith('Z'):
                        completed_str = completed_str[:-1]
                    elif '+' in completed_str:
                        completed_str = completed_str.split('+')[0]
                    completed_time = datetime.fromisoformat(completed_str)
                    if today_start <= completed_time <= today_end:
                        completed_today.append(task)
            except (ValueError, TypeError):
                # Skip tasks with invalid completion dates
                continue

    # Sort by completion time
    completed_today.sort(key=lambda t: t.completed_at or '')

    return completed_today


def generate_handoff_document(
    context: Dict[str, Any],
    completed_tasks: List[Task],
    title: str = "Session Handoff"
) -> str:
    """
    Generate a handoff document from session context and completed tasks.

    Args:
        context: Session context from gather_session_context()
        completed_tasks: List of completed tasks from gather_completed_tasks()
        title: Document title (default: "Session Handoff")

    Returns:
        Markdown formatted handoff document
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# {title}: {date_str}",
        "",
        f"**Date:** {date_str}",
        f"**Time:** {timestamp}",
        f"**Branch:** {context['branch'] or 'unknown'}",
        "",
        "---",
        "",
        "## Summary",
        ""
    ]

    # Generate summary
    num_tasks = len(completed_tasks)
    if num_tasks > 0:
        lines.append(
            f"Completed {num_tasks} task{'s' if num_tasks != 1 else ''} this session. "
            f"Repository state: {context['status_summary']}."
        )
    else:
        lines.append(
            f"Session focused on exploration and investigation. "
            f"Repository state: {context['status_summary']}."
        )

    lines.extend([
        "",
        "## Completed This Session",
        ""
    ])

    if completed_tasks:
        for task in completed_tasks:
            lines.append(f"### {task.id}: {task.title}")
            if task.description:
                lines.append(f"{task.description}")

            # Add notes if available
            if task.notes:
                lines.append(f"**Notes:** {task.notes}")
            lines.append("")
    else:
        lines.append("*No tasks completed this session*")
        lines.append("")

    lines.extend([
        "## Current State",
        "",
        f"**Git Status:** {context['status_summary']}",
        ""
    ])

    if context['uncommitted_files']:
        lines.append("**Uncommitted Changes:**")
        for file_status in context['uncommitted_files']:
            lines.append(f"- `{file_status}`")
        lines.append("")

    if context['recent_commits']:
        lines.extend([
            "**Recent Commits:**",
            ""
        ])
        for commit_hash, message in context['recent_commits']:
            lines.append(f"- `{commit_hash}` {message}")
        lines.append("")

    lines.extend([
        "## Suggested Next Steps",
        ""
    ])

    # Generate suggested next steps based on context
    next_steps = []

    # Check for uncommitted changes
    if context['uncommitted_files']:
        modified_count = sum(1 for f in context['uncommitted_files'] if f.startswith('M'))
        if modified_count > 0:
            next_steps.append("Review and commit uncommitted changes")

    # Check for pending tasks from GoT
    try:
        manager = GoTManager(GOT_DIR)
        pending = manager.list_tasks(status="pending")
        in_progress = manager.list_tasks(status="in_progress")

        if in_progress:
            for task in in_progress[:3]:  # Show first 3
                next_steps.append(f"Continue: {task.title} ({task.id})")

        if pending:
            high_priority = [t for t in pending if t.priority in ('critical', 'high')]
            if high_priority:
                for task in high_priority[:2]:  # Show first 2 high priority
                    next_steps.append(f"Start: {task.title} ({task.id})")
            elif pending:
                next_steps.append(f"Start next pending task ({len(pending)} available)")
    except Exception:
        pass  # Skip task suggestions if GoT unavailable

    # Add test and documentation reminders
    if completed_tasks:
        next_steps.append("Run full test suite to verify changes")
        next_steps.append("Update documentation if needed")

    if next_steps:
        for i, step in enumerate(next_steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("*Check GoT for pending tasks: `python scripts/got_utils.py task list`*")

    lines.extend([
        "",
        "## Files Modified",
        ""
    ])

    # Collect all modified files from git status
    all_files = set()

    # Parse git status files
    for file_status in context['uncommitted_files']:
        # Format: "XX filename" where XX is status code
        parts = file_status.split(maxsplit=1)
        if len(parts) == 2:
            all_files.add(parts[1])

    if all_files:
        for file in sorted(all_files):
            lines.append(f"- `{file}`")
    else:
        lines.append("*No uncommitted file changes*")

    lines.extend([
        "",
        "---",
        "",
        "*Generated automatically by session_handoff.py*"
    ])

    return '\n'.join(lines)


def create_handoff_file(content: str, output_path: Optional[Path] = None) -> Path:
    """
    Create handoff file at specified path or generate unique filename.

    Args:
        content: Markdown content to write
        output_path: Optional specific path. If None, generates timestamped filename.

    Returns:
        Path to created file
    """
    if output_path is None:
        # Generate unique filename
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        session_id = generate_session_id()

        filename = f"session-handoff-{date_str}_{time_str}_{session_id}.md"
        output_path = MEMORIES_DIR / filename

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write content
    with open(output_path, 'w') as f:
        f.write(content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate session handoff documents for knowledge transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for handoff document"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview document without creating file"
    )
    parser.add_argument(
        "--title", "-t",
        default="Session Handoff",
        help="Document title (default: 'Session Handoff')"
    )
    parser.add_argument(
        "--got-dir",
        type=Path,
        default=GOT_DIR,
        help=f"GoT directory (default: {GOT_DIR})"
    )

    args = parser.parse_args()

    # Gather context
    print("Gathering session context...")
    context = gather_session_context()

    print("Loading completed tasks from GoT...")
    completed_tasks = gather_completed_tasks(args.got_dir)

    # Generate document
    print(f"Found {len(completed_tasks)} tasks completed today")
    content = generate_handoff_document(context, completed_tasks, args.title)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Document preview:")
        print("=" * 60 + "\n")
        print(content)
    else:
        output_path = create_handoff_file(content, args.output)
        print(f"Created session handoff: {output_path}")


if __name__ == "__main__":
    main()
