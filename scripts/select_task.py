#!/usr/bin/env python3
"""
Task Selection Assistant

Interactive script to help select the best task to work on based on priority,
effort, dependencies, and category preferences.

Uses the merge-friendly task system (tasks/*.json).

Usage:
    python scripts/select_task.py              # Interactive mode
    python scripts/select_task.py --auto       # Auto-select best task
    python scripts/select_task.py --category arch  # Filter by category
    python scripts/select_task.py --effort small   # Prefer small tasks

Author: Cortical Text Processor Team
"""

import argparse
from typing import List, Optional, Set

# Import from the new task system
from task_utils import load_all_tasks, Task


def get_available_tasks(
    tasks: List[Task],
    exclude_completed: bool = True,
    exclude_deferred: bool = True
) -> List[Task]:
    """
    Get tasks that are available to work on.

    Args:
        tasks: List of all tasks
        exclude_completed: Exclude completed tasks
        exclude_deferred: Exclude deferred tasks

    Returns:
        List of available tasks
    """
    available = []
    for task in tasks:
        if exclude_completed and task.status == 'completed':
            continue
        if exclude_deferred and task.status == 'deferred':
            continue
        available.append(task)
    return available


def get_ready_tasks(tasks: List[Task]) -> List[Task]:
    """
    Get tasks whose dependencies are all completed.

    Args:
        tasks: List of tasks to check

    Returns:
        List of tasks ready to work on
    """
    all_tasks = {t.id: t for t in tasks}
    completed_ids = {t.id for t in tasks if t.status == 'completed'}

    ready = []
    for task in tasks:
        if task.status in ('completed', 'deferred'):
            continue

        # Check if all dependencies are completed
        deps = task.depends_on or []
        deps_met = all(
            dep in completed_ids or dep not in all_tasks
            for dep in deps
        )

        if deps_met:
            ready.append(task)

    return ready


def score_task(task: Task, prefer_effort: Optional[str] = None) -> float:
    """
    Score a task for auto-selection.

    Higher score = better candidate to work on.

    Args:
        task: Task to score
        prefer_effort: Preferred effort level ('small', 'medium', 'large')

    Returns:
        Numeric score
    """
    score = 0.0

    # Priority weight (high = 30, medium = 20, low = 10)
    priority_scores = {'high': 30, 'medium': 20, 'low': 10}
    score += priority_scores.get(task.priority.lower(), 15)

    # In-progress tasks get a boost (continuity)
    if task.status == 'in_progress':
        score += 25

    # Effort preference
    if prefer_effort:
        if task.effort.lower() == prefer_effort.lower():
            score += 15
        elif prefer_effort == 'small' and task.effort.lower() == 'medium':
            score += 5  # Medium is acceptable when small preferred

    # Tasks with no dependencies are easier to start
    if not task.depends_on:
        score += 5

    return score


def select_best_task(
    tasks: List[Task],
    category: Optional[str] = None,
    prefer_effort: Optional[str] = None
) -> Optional[Task]:
    """
    Auto-select the best task to work on.

    Args:
        tasks: List of all tasks
        category: Filter to specific category
        prefer_effort: Preferred effort level

    Returns:
        Best task to work on, or None if no tasks available
    """
    # Get ready tasks (dependencies met)
    ready = get_ready_tasks(tasks)

    if not ready:
        return None

    # Filter by category if specified
    if category:
        ready = [t for t in ready if t.category.lower() == category.lower()]

    if not ready:
        return None

    # Score and sort
    scored = [(score_task(t, prefer_effort), t) for t in ready]
    scored.sort(key=lambda x: -x[0])  # Descending by score

    return scored[0][1] if scored else None


def display_task(task: Task, show_deps: bool = True) -> None:
    """Display a task with formatting."""
    status_icons = {
        'pending': '○',
        'in_progress': '◐',
        'completed': '●',
        'deferred': '◌'
    }
    priority_icons = {
        'high': '🔴',
        'medium': '🟡',
        'low': '🟢'
    }

    icon = status_icons.get(task.status, '?')
    priority = priority_icons.get(task.priority.lower(), '⚪')

    print(f"\n{icon} {priority} {task.id}")
    print(f"   Title: {task.title}")
    print(f"   Category: {task.category} | Effort: {task.effort}")

    if task.description:
        desc = task.description[:200]
        if len(task.description) > 200:
            desc += "..."
        print(f"   Description: {desc}")

    if show_deps and task.depends_on:
        print(f"   Depends on: {', '.join(task.depends_on)}")


def interactive_mode(tasks: List[Task]) -> None:
    """Run interactive task selection."""
    print("=" * 60)
    print("TASK SELECTION ASSISTANT")
    print("=" * 60)

    # Show summary
    available = get_available_tasks(tasks)
    ready = get_ready_tasks(tasks)

    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Available (not completed/deferred): {len(available)}")
    print(f"Ready (dependencies met): {len(ready)}")

    if not ready:
        print("\n⚠️  No tasks are ready to work on!")
        print("   All pending tasks have unmet dependencies.")
        return

    # Show in-progress first
    in_progress = [t for t in ready if t.status == 'in_progress']
    if in_progress:
        print("\n📍 CURRENTLY IN PROGRESS:")
        for task in in_progress:
            display_task(task)

    # Show top recommendations
    print("\n🎯 TOP RECOMMENDATIONS:")
    scored = [(score_task(t), t) for t in ready if t.status != 'in_progress']
    scored.sort(key=lambda x: -x[0])

    for i, (score, task) in enumerate(scored[:5], 1):
        print(f"\n--- Option {i} (score: {score:.0f}) ---")
        display_task(task)

    # Interactive selection
    print("\n" + "-" * 60)
    print("Commands:")
    print("  1-5    : Select recommended task")
    print("  l      : List all ready tasks")
    print("  c CAT  : Filter by category")
    print("  q      : Quit")

    while True:
        try:
            choice = input("\nSelect task: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice == 'q':
            break
        elif choice == 'l':
            print("\n📋 ALL READY TASKS:")
            for task in ready:
                display_task(task, show_deps=False)
        elif choice.startswith('c '):
            cat = choice[2:].strip()
            filtered = [t for t in ready if cat in t.category.lower()]
            print(f"\n📋 TASKS IN CATEGORY '{cat}':")
            for task in filtered:
                display_task(task, show_deps=False)
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scored):
                selected = scored[idx][1]
                print(f"\n✅ Selected: {selected.id}")
                print(f"   {selected.title}")
                print("\n   To start working, run:")
                print(f"   python scripts/task_utils.py complete {selected.id}")
                break
            else:
                print("Invalid selection.")
        else:
            print("Unknown command. Try 1-5, l, c CAT, or q.")


def main():
    parser = argparse.ArgumentParser(
        description="Task selection assistant using merge-friendly task system"
    )
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Auto-select best task without interaction'
    )
    parser.add_argument(
        '--category', '-c',
        help='Filter by category (e.g., arch, devex, bugfix)'
    )
    parser.add_argument(
        '--effort', '-e',
        choices=['small', 'medium', 'large'],
        help='Prefer tasks of this effort level'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='Just list ready tasks, no interaction'
    )

    args = parser.parse_args()

    # Load all tasks
    all_tasks = load_all_tasks()

    if args.list:
        ready = get_ready_tasks(all_tasks)
        if args.category:
            ready = [t for t in ready if args.category.lower() in t.category.lower()]

        print(f"Ready tasks: {len(ready)}")
        for task in ready:
            display_task(task, show_deps=False)
        return

    if args.auto:
        best = select_best_task(all_tasks, args.category, args.effort)
        if best:
            print("🎯 RECOMMENDED TASK:")
            display_task(best)
            print(f"\nScore: {score_task(best, args.effort):.0f}")
        else:
            print("No tasks available to work on.")
        return

    # Interactive mode
    interactive_mode(all_tasks)


if __name__ == '__main__':
    main()
