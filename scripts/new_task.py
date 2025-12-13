#!/usr/bin/env python3
"""
Quick task creation from command line.

Usage:
    # Create a task (interactive prompts)
    python scripts/new_task.py

    # Create with title only
    python scripts/new_task.py "Fix the login bug"

    # Create with options
    python scripts/new_task.py "Fix login bug" --priority high --category bugfix

    # Show current session tasks
    python scripts/new_task.py --list

    # Complete a task
    python scripts/new_task.py --complete T-20251213-123456-a1b2-01

Examples:
    $ python scripts/new_task.py "Add dark mode" --priority medium --category feature
    Created: T-20251213-143052-a1b2-01 - Add dark mode
    Saved to: tasks/2025-12-13_14-30-52_a1b2.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from task_utils import (
    TaskSession,
    Task,
    load_all_tasks,
    consolidate_tasks,
    get_task_by_id,
    DEFAULT_TASKS_DIR,
)


# Session file to persist between calls
SESSION_FILE = Path(DEFAULT_TASKS_DIR) / ".current_session.json"


def get_or_create_session() -> TaskSession:
    """Get existing session or create new one."""
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE) as f:
                data = json.load(f)
            session = TaskSession(
                session_id=data["session_id"],
                started_at=data["started_at"]
            )
            session._task_counter = data.get("task_counter", 0)
            # Load existing tasks
            if Path(DEFAULT_TASKS_DIR).exists():
                for filepath in Path(DEFAULT_TASKS_DIR).glob(f"*_{data['session_id']}.json"):
                    loaded = TaskSession.load(filepath)
                    session.tasks = loaded.tasks
                    break
            return session
        except (json.JSONDecodeError, KeyError):
            pass

    # Create new session
    session = TaskSession()
    save_session_state(session)
    return session


def save_session_state(session: TaskSession) -> None:
    """Save session state for persistence."""
    Path(DEFAULT_TASKS_DIR).mkdir(parents=True, exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({
            "session_id": session.session_id,
            "started_at": session.started_at,
            "task_counter": session._task_counter
        }, f)


def create_task(
    title: str,
    priority: str = "medium",
    category: str = "general",
    description: str = "",
    effort: str = "medium"
) -> Task:
    """Create a task in the current session."""
    session = get_or_create_session()

    task = session.create_task(
        title=title,
        priority=priority,
        category=category,
        description=description,
        effort=effort
    )

    filepath = session.save()
    save_session_state(session)

    return task, filepath


def list_tasks(status_filter: str = None) -> None:
    """List all tasks."""
    tasks = load_all_tasks()

    if status_filter:
        tasks = [t for t in tasks if t.status == status_filter]

    if not tasks:
        print("No tasks found.")
        return

    # Group by status
    by_status = {}
    for task in tasks:
        by_status.setdefault(task.status, []).append(task)

    status_emoji = {
        "in_progress": "🔄",
        "pending": "📋",
        "completed": "✅",
        "deferred": "⏸️"
    }

    for status in ["in_progress", "pending", "completed", "deferred"]:
        if status not in by_status:
            continue
        print(f"\n{status_emoji.get(status, '•')} {status.upper()}")
        for task in by_status[status]:
            priority_marker = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "")
            print(f"  {priority_marker} {task.id}: {task.title}")


def complete_task(task_id: str) -> bool:
    """Mark a task as completed."""
    # Find the task
    task = get_task_by_id(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return False

    # Load the session file containing this task
    for filepath in Path(DEFAULT_TASKS_DIR).glob("*.json"):
        if filepath.name.startswith("."):
            continue
        try:
            session = TaskSession.load(filepath)
            for t in session.tasks:
                if t.id == task_id:
                    t.mark_complete()
                    session.save(DEFAULT_TASKS_DIR)
                    print(f"✅ Completed: {task_id} - {t.title}")
                    return True
        except:
            continue

    print(f"Could not update task: {task_id}")
    return False


def new_session() -> None:
    """Start a new session."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
    session = get_or_create_session()
    print(f"Started new session: {session.session_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick task creation for parallel agent workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("title", nargs="?", help="Task title")
    parser.add_argument("-p", "--priority", choices=["high", "medium", "low"],
                        default="medium", help="Task priority")
    parser.add_argument("-c", "--category", default="general", help="Task category")
    parser.add_argument("-d", "--description", default="", help="Task description")
    parser.add_argument("-e", "--effort", choices=["small", "medium", "large"],
                        default="medium", help="Effort estimate")
    parser.add_argument("-l", "--list", action="store_true", help="List all tasks")
    parser.add_argument("-s", "--status", help="Filter by status when listing")
    parser.add_argument("--complete", metavar="TASK_ID", help="Mark task as completed")
    parser.add_argument("--new-session", action="store_true", help="Start a new session")
    parser.add_argument("--summary", action="store_true", help="Show task summary")

    args = parser.parse_args()

    # Ensure tasks directory exists
    Path(DEFAULT_TASKS_DIR).mkdir(parents=True, exist_ok=True)

    if args.new_session:
        new_session()
    elif args.list:
        list_tasks(args.status)
    elif args.complete:
        complete_task(args.complete)
    elif args.summary:
        grouped = consolidate_tasks()
        print("\n=== Task Summary ===")
        for status, tasks in grouped.items():
            if tasks:
                print(f"{status}: {len(tasks)}")
    elif args.title:
        task, filepath = create_task(
            title=args.title,
            priority=args.priority,
            category=args.category,
            description=args.description,
            effort=args.effort
        )
        print(f"Created: {task.id} - {task.title}")
        print(f"Saved to: {filepath}")
    else:
        # Interactive mode
        print("Create a new task (Ctrl+C to cancel)")
        title = input("Title: ").strip()
        if not title:
            print("Title is required")
            return

        priority = input("Priority [high/medium/low] (medium): ").strip() or "medium"
        category = input("Category (general): ").strip() or "general"
        description = input("Description (optional): ").strip()

        task, filepath = create_task(
            title=title,
            priority=priority,
            category=category,
            description=description
        )
        print(f"\nCreated: {task.id} - {task.title}")
        print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()
