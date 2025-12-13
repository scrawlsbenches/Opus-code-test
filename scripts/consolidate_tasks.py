#!/usr/bin/env python3
"""
Consolidate task files from parallel agent sessions.

This script merges task files created by parallel agents into a unified
view, resolving any conflicts and generating an updated TASK_LIST.md.

Similar to `git gc` for chunk files, this consolidates distributed task
state into a coherent whole.

Usage:
    # Show what would be consolidated (dry run)
    python scripts/consolidate_tasks.py --dry-run

    # Consolidate and update TASK_LIST.md
    python scripts/consolidate_tasks.py --update

    # Just generate a summary without modifying anything
    python scripts/consolidate_tasks.py --summary

    # Consolidate tasks from a specific directory
    python scripts/consolidate_tasks.py --dir tasks/ --update

Architecture:
    tasks/
    ├── 2025-12-13_14-30-52_a1b2.json    # Agent A's session
    ├── 2025-12-13_14-31-05_c3d4.json    # Agent B's session
    └── ...

    After consolidation:
    ├── consolidated_2025-12-13_15-00-00.json  # Merged state
    └── (old files can be archived or removed)
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from task_utils import (
    Task, TaskSession, load_all_tasks, consolidate_tasks,
    DEFAULT_TASKS_DIR, generate_session_id
)


def find_conflicts(tasks: List[Task]) -> Dict[str, List[Task]]:
    """
    Find tasks that might be duplicates or conflicts.

    Returns dict of potential duplicate groups (same title, different IDs).
    """
    by_title = defaultdict(list)
    for task in tasks:
        # Normalize title for comparison
        normalized = task.title.lower().strip()
        by_title[normalized].append(task)

    # Return only groups with potential conflicts
    return {title: tasks for title, tasks in by_title.items() if len(tasks) > 1}


def merge_duplicate_tasks(tasks: List[Task]) -> Task:
    """
    Merge potentially duplicate tasks into one.

    Strategy:
    - Keep the earliest creation time
    - Use the most complete description
    - Prefer higher priority
    - Prefer "in_progress" or "completed" status over "pending"
    """
    if len(tasks) == 1:
        return tasks[0]

    # Sort by creation time (keep earliest ID)
    sorted_tasks = sorted(tasks, key=lambda t: t.created_at)
    merged = Task(
        id=sorted_tasks[0].id,
        title=sorted_tasks[0].title,
        created_at=sorted_tasks[0].created_at
    )

    # Merge fields from all tasks
    priority_order = {"high": 0, "medium": 1, "low": 2}
    status_order = {"completed": 0, "in_progress": 1, "pending": 2, "deferred": 3}

    best_priority = min(tasks, key=lambda t: priority_order.get(t.priority, 1))
    best_status = min(tasks, key=lambda t: status_order.get(t.status, 2))

    merged.priority = best_priority.priority
    merged.status = best_status.status
    merged.category = sorted_tasks[0].category

    # Use longest description
    merged.description = max(tasks, key=lambda t: len(t.description)).description

    # Merge dependencies
    all_deps = set()
    for task in tasks:
        all_deps.update(task.depends_on)
    merged.depends_on = list(all_deps)

    # Merge context
    merged.context = {}
    for task in tasks:
        merged.context.update(task.context)

    # Track completion
    completed = [t for t in tasks if t.completed_at]
    if completed:
        merged.completed_at = min(t.completed_at for t in completed)

    return merged


def consolidate_and_dedupe(
    tasks_dir: str = DEFAULT_TASKS_DIR,
    auto_merge: bool = False
) -> Tuple[List[Task], Dict[str, List[Task]]]:
    """
    Load all tasks and identify/resolve duplicates.

    Args:
        tasks_dir: Directory containing task session files
        auto_merge: If True, automatically merge duplicates

    Returns:
        Tuple of (final task list, conflicts dict)
    """
    all_tasks = load_all_tasks(tasks_dir)
    conflicts = find_conflicts(all_tasks)

    if not auto_merge or not conflicts:
        return all_tasks, conflicts

    # Auto-merge duplicates
    merged_ids = set()
    final_tasks = []

    for title, conflict_group in conflicts.items():
        merged = merge_duplicate_tasks(conflict_group)
        final_tasks.append(merged)
        merged_ids.update(t.id for t in conflict_group)

    # Add non-conflicting tasks
    for task in all_tasks:
        if task.id not in merged_ids:
            final_tasks.append(task)

    return final_tasks, conflicts


def generate_markdown_section(
    tasks: List[Task],
    status_filter: str,
    priority_filter: Optional[str] = None
) -> List[str]:
    """Generate markdown table rows for tasks matching filters."""
    filtered = [t for t in tasks if t.status == status_filter]
    if priority_filter:
        filtered = [t for t in filtered if t.priority == priority_filter]

    if not filtered:
        return []

    # Sort by priority then creation time
    priority_order = {"high": 0, "medium": 1, "low": 2}
    filtered.sort(key=lambda t: (priority_order.get(t.priority, 1), t.created_at))

    lines = []
    for task in filtered:
        deps = ", ".join(task.depends_on) if task.depends_on else "-"
        lines.append(
            f"| {task.id} | {task.title} | {task.category} | {deps} | {task.effort} |"
        )

    return lines


def write_consolidated_file(
    tasks: List[Task],
    output_dir: str,
    session_id: Optional[str] = None
) -> Path:
    """Write consolidated tasks to a single JSON file."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    sid = session_id or generate_session_id()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"consolidated_{timestamp}_{sid}.json"

    filepath = dir_path / filename

    data = {
        "version": 1,
        "type": "consolidated",
        "session_id": sid,
        "created_at": datetime.now().isoformat(),
        "task_count": len(tasks),
        "tasks": [t.to_dict() for t in tasks]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath


def archive_old_session_files(
    tasks_dir: str,
    archive_dir: Optional[str] = None,
    keep_consolidated: bool = True
) -> List[Path]:
    """
    Move old session files to archive after consolidation.

    Args:
        tasks_dir: Directory containing task files
        archive_dir: Where to move old files (default: tasks/archive/)
        keep_consolidated: Don't archive consolidated_*.json files

    Returns:
        List of archived file paths
    """
    dir_path = Path(tasks_dir)
    archive_path = Path(archive_dir or (dir_path / "archive"))
    archive_path.mkdir(parents=True, exist_ok=True)

    archived = []
    for filepath in dir_path.glob("*.json"):
        if keep_consolidated and filepath.name.startswith("consolidated_"):
            continue

        dest = archive_path / filepath.name
        shutil.move(str(filepath), str(dest))
        archived.append(dest)

    return archived


def print_summary(tasks: List[Task], conflicts: Dict[str, List[Task]]) -> None:
    """Print a summary of task state."""
    by_status = defaultdict(list)
    for task in tasks:
        by_status[task.status].append(task)

    print("\n=== Task Summary ===\n")
    print(f"Total tasks: {len(tasks)}")
    print(f"  In Progress: {len(by_status['in_progress'])}")
    print(f"  Pending:     {len(by_status['pending'])}")
    print(f"  Completed:   {len(by_status['completed'])}")
    print(f"  Deferred:    {len(by_status['deferred'])}")

    if conflicts:
        print(f"\n⚠️  Found {len(conflicts)} potential duplicate groups:")
        for title, group in conflicts.items():
            print(f"  - \"{title[:50]}...\" ({len(group)} tasks)")
            for task in group:
                print(f"      {task.id} [{task.status}]")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate task files from parallel agent sessions"
    )
    parser.add_argument(
        "--dir", default=DEFAULT_TASKS_DIR,
        help="Tasks directory (default: tasks/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show summary only"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Write consolidated file and archive old files"
    )
    parser.add_argument(
        "--auto-merge", action="store_true",
        help="Automatically merge duplicate tasks"
    )
    parser.add_argument(
        "--output", help="Output file for consolidated JSON"
    )
    parser.add_argument(
        "--archive", action="store_true",
        help="Archive old session files after consolidation"
    )

    args = parser.parse_args()

    # Check if tasks directory exists
    if not Path(args.dir).exists():
        print(f"Tasks directory '{args.dir}' does not exist.")
        print("No tasks to consolidate. Use task_utils.py to create tasks first.")
        return

    # Load and analyze tasks
    tasks, conflicts = consolidate_and_dedupe(args.dir, args.auto_merge)

    if not tasks:
        print("No tasks found.")
        return

    # Always show summary
    print_summary(tasks, conflicts)

    if args.summary or args.dry_run:
        if args.dry_run and args.update:
            print("\n[Dry run] Would consolidate to:")
            print(f"  {args.dir}/consolidated_TIMESTAMP_XXXX.json")
            if args.archive:
                print(f"  Would archive {len(list(Path(args.dir).glob('*.json')))} files")
        return

    if args.update:
        # Write consolidated file
        output_path = write_consolidated_file(tasks, args.dir)
        print(f"\n✅ Consolidated to: {output_path}")

        if args.archive:
            archived = archive_old_session_files(args.dir)
            print(f"📦 Archived {len(archived)} session files")

    if conflicts and not args.auto_merge:
        print("\n💡 Tip: Use --auto-merge to automatically resolve duplicates")


if __name__ == "__main__":
    main()
