#!/usr/bin/env python3
"""
Generate session handoff documents for knowledge transfer.

Creates automatic handoff documents when ending a coding session, capturing:
- Git status and branch information
- Recently completed tasks
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

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from task_utils import load_all_tasks, Task, generate_session_id


# Default output directory
MEMORIES_DIR = Path("samples/memories")


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
            timeout=2
        )
        if result.returncode == 0:
            status_lines = result.stdout.strip().split('\n')
            context['uncommitted_files'] = [
                line.strip() for line in status_lines if line.strip()
            ]

            # Create summary
            if context['uncommitted_files']:
                modified = sum(1 for line in context['uncommitted_files'] if line.startswith('M'))
                added = sum(1 for line in context['uncommitted_files'] if line.startswith('A'))
                deleted = sum(1 for line in context['uncommitted_files'] if line.startswith('D'))
                untracked = sum(1 for line in context['uncommitted_files'] if line.startswith('??'))

                parts = []
                if modified:
                    parts.append(f"{modified} modified")
                if added:
                    parts.append(f"{added} added")
                if deleted:
                    parts.append(f"{deleted} deleted")
                if untracked:
                    parts.append(f"{untracked} untracked")

                context['status_summary'] = ', '.join(parts) if parts else 'clean'
            else:
                context['status_summary'] = 'clean'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        context['status_summary'] = 'unknown'

    # Get recent commits (last 5)
    try:
        result = subprocess.run(
            ["git", "log", "--pretty=format:%h|%s", "-n", "5"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    commit_hash, message = line.split('|', 1)
                    context['recent_commits'].append((commit_hash, message))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return context


def gather_completed_tasks(tasks_dir: str = "tasks") -> List[Task]:
    """
    Gather tasks completed today from task session files.

    Args:
        tasks_dir: Directory containing task session files

    Returns:
        List of Task objects completed today, sorted by completion time
    """
    all_tasks = load_all_tasks(tasks_dir)

    # Get today's date range
    today = datetime.now().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    # Filter to completed tasks from today
    completed_today = []
    for task in all_tasks:
        if task.status == 'completed' and task.completed_at:
            try:
                completed_time = datetime.fromisoformat(task.completed_at)
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

            # Handle retrospective (must be a dict)
            if task.retrospective and isinstance(task.retrospective, dict):
                if task.retrospective.get('notes'):
                    lines.append(f"**Notes:** {task.retrospective['notes']}")
                if task.retrospective.get('files_touched'):
                    files = task.retrospective['files_touched']
                    if files:
                        lines.append("**Files modified:**")
                        for file in files:
                            lines.append(f"- `{file}`")
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

    # Check for pending tasks
    all_tasks = load_all_tasks("tasks")
    pending = [t for t in all_tasks if t.status == 'pending']
    in_progress = [t for t in all_tasks if t.status == 'in_progress']

    if in_progress:
        for task in in_progress[:3]:  # Show first 3
            next_steps.append(f"Continue: {task.title} ({task.id})")

    if pending:
        high_priority = [t for t in pending if t.priority == 'high']
        if high_priority:
            for task in high_priority[:2]:  # Show first 2 high priority
                next_steps.append(f"Start: {task.title} ({task.id})")
        elif pending:
            next_steps.append(f"Start next pending task ({len(pending)} available)")

    # Add test and documentation reminders
    if completed_tasks:
        next_steps.append("Run full test suite to verify changes")
        next_steps.append("Update documentation if needed")

    if next_steps:
        for i, step in enumerate(next_steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("*Review pending tasks in `tasks/` directory*")

    lines.extend([
        "",
        "## Files Modified",
        ""
    ])

    # Collect all modified files from tasks and git status
    all_files = set()

    for task in completed_tasks:
        if task.retrospective and isinstance(task.retrospective, dict):
            if task.retrospective.get('files_touched'):
                all_files.update(task.retrospective['files_touched'])

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
        lines.append("*No files modified this session*")

    lines.extend([
        "",
        "---",
        "",
        f"*Session handoff generated at: {timestamp}*"
    ])

    return '\n'.join(lines)


def generate_handoff_filename() -> str:
    """
    Generate merge-safe handoff filename.

    Returns:
        Filename in format: session-handoff-YYYY-MM-DD_HH-MM-SS_XXXX.md
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    session_id = generate_session_id()

    return f"session-handoff-{date_str}_{time_str}_{session_id}.md"


def main():
    parser = argparse.ArgumentParser(
        description="Generate session handoff documents for knowledge transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview handoff document without creating file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output file path (default: auto-generated in samples/memories/)"
    )
    parser.add_argument(
        "--title",
        default="Session Handoff",
        help="Document title (default: 'Session Handoff')"
    )
    parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task files (default: tasks/)"
    )

    args = parser.parse_args()

    # Gather session information
    print("Gathering session context...")
    context = gather_session_context()

    print("Loading completed tasks...")
    completed_tasks = gather_completed_tasks(args.tasks_dir)

    # Generate document
    print("Generating handoff document...")
    document = generate_handoff_document(context, completed_tasks, args.title)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
        filename = generate_handoff_filename()
        output_path = MEMORIES_DIR / filename

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would create: {output_path}")
        print(f"\nDocument preview:\n")
        print(document)
        return

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(document)

    print(f"\nCreated session handoff:")
    print(f"  {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review: $EDITOR {output_path}")
    print(f"  2. Commit: git add {output_path} && git commit -m 'memory: session handoff'")
    print(f"  3. Re-index: python scripts/index_codebase.py --incremental")


if __name__ == "__main__":
    main()
