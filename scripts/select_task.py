#!/usr/bin/env python3
"""
Task Selection Assistant

Interactive script to help select the best task to work on based on priority,
effort, dependencies, and category preferences.

Usage:
    python scripts/select_task.py              # Interactive mode
    python scripts/select_task.py --auto       # Auto-select best task
    python scripts/select_task.py --category DevEx  # Filter by category
    python scripts/select_task.py --effort Small    # Prefer small tasks

Author: Cortical Text Processor Team
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class Task:
    """Represents a task with its metadata."""
    id: int
    name: str
    category: str
    depends: List[int]
    effort: str
    priority: str

    def __repr__(self):
        return f"Task(#{self.id}, {self.priority}, {self.effort})"


def parse_task_list(file_path: Path) -> Dict[int, Task]:
    """
    Parse TASK_LIST.md and extract all tasks.

    Args:
        file_path: Path to TASK_LIST.md

    Returns:
        Dictionary mapping task ID to Task object
    """
    tasks = {}
    current_priority = None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect priority sections (skip Deferred and Future)
        if '### 🟠 High' in line:
            current_priority = 'High'
        elif '### 🟡 Medium' in line:
            current_priority = 'Medium'
        elif '### 🟢 Low' in line:
            current_priority = 'Low'
        elif '### ⏸️ Deferred' in line or '### 🔮 Future' in line:
            current_priority = None  # Skip deferred/future tasks

        # Parse table rows
        if current_priority and line.startswith('|') and not line.startswith('|---|') and not line.startswith('| #'):
            parts = [p.strip() for p in line.split('|')[1:-1]]

            if len(parts) >= 4 and parts[0].isdigit():
                task_id = int(parts[0])
                name = parts[1]
                category = parts[2]
                depends_str = parts[3]
                effort = parts[4] if len(parts) > 4 else 'Unknown'

                # Parse dependencies
                depends = []
                if depends_str and depends_str != '-':
                    for dep in depends_str.split(','):
                        dep = dep.strip()
                        if dep.isdigit():
                            depends.append(int(dep))

                tasks[task_id] = Task(
                    id=task_id,
                    name=name,
                    category=category,
                    depends=depends,
                    effort=effort,
                    priority=current_priority
                )

        i += 1

    return tasks


def get_available_tasks(tasks: Dict[int, Task], completed_tasks: Set[int]) -> List[Task]:
    """
    Get tasks that have all dependencies met.

    Args:
        tasks: All tasks
        completed_tasks: Set of completed task IDs

    Returns:
        List of tasks ready to be worked on
    """
    available = []

    for task in tasks.values():
        # Check if all dependencies are met
        if all(dep in completed_tasks or dep not in tasks for dep in task.depends):
            available.append(task)

    return available


def score_task(task: Task, effort_preference: Optional[str] = None) -> float:
    """
    Score a task based on priority, effort, and preferences.

    Higher score = better task to work on.

    Args:
        task: The task to score
        effort_preference: Preferred effort level (Small, Medium, Large)

    Returns:
        Score for the task
    """
    # Priority scores
    priority_scores = {
        'High': 10.0,
        'Medium': 5.0,
        'Low': 2.0
    }

    # Effort scores (smaller = better for learning/quick wins)
    effort_scores = {
        'Small': 3.0,
        'Medium': 2.0,
        'Large': 1.0,
        'Unknown': 1.5
    }

    # Base score
    score = priority_scores.get(task.priority, 1.0) * effort_scores.get(task.effort, 1.0)

    # Boost if matches effort preference
    if effort_preference and task.effort == effort_preference:
        score *= 1.5

    return score


def get_blocked_tasks(tasks: Dict[int, Task], available_task_ids: Set[int]) -> Dict[int, List[int]]:
    """
    Find tasks that are blocked by missing dependencies.

    Args:
        tasks: All tasks
        available_task_ids: Set of available task IDs

    Returns:
        Dictionary mapping task ID to list of missing dependency IDs
    """
    blocked = {}

    for task in tasks.values():
        if task.id not in available_task_ids and task.depends:
            missing_deps = [dep for dep in task.depends if dep not in tasks]
            if missing_deps:
                # Only track if dependency is missing (completed task)
                # Tasks with dependencies on active tasks are just "not ready"
                pass
            else:
                # Check if blocked by active task
                blocking = [dep for dep in task.depends if dep in tasks]
                if blocking:
                    blocked[task.id] = blocking

    return blocked


def print_task_recommendation(task: Task, tasks: Dict[int, Task], score: float):
    """Print a detailed recommendation for a task."""
    print("\n" + "=" * 70)
    print(f"RECOMMENDED TASK: #{task.id}")
    print("=" * 70)
    print(f"Title:     {task.name}")
    print(f"Priority:  {task.priority}")
    print(f"Effort:    {task.effort}")
    print(f"Category:  {task.category}")
    print(f"Score:     {score:.2f}")

    if task.depends:
        deps_str = ", ".join(f"#{dep}" for dep in task.depends)
        print(f"Depends:   {deps_str} (all dependencies met)")
    else:
        print(f"Depends:   None (independent task)")

    print("\nWhy this task?")

    reasons = []
    if task.priority == 'High':
        reasons.append("• High priority - needs to be done this week")
    elif task.priority == 'Medium':
        reasons.append("• Medium priority - scheduled for this month")

    if task.effort == 'Small':
        reasons.append("• Small effort - quick win, can finish in <1 hour")
    elif task.effort == 'Medium':
        reasons.append("• Medium effort - manageable scope (1-4 hours)")

    if not task.depends:
        reasons.append("• No dependencies - can start immediately")
    else:
        reasons.append("• All dependencies met - ready to start")

    if task.category in ['Testing', 'CodeQual', 'TaskMgmt']:
        reasons.append(f"• {task.category} - improves project infrastructure")

    for reason in reasons:
        print(reason)

    print("\n" + "=" * 70)


def interactive_mode(tasks: Dict[int, Task], effort_preference: Optional[str] = None,
                     category_filter: Optional[str] = None):
    """
    Run interactive task selection.

    Args:
        tasks: All available tasks
        effort_preference: Optional effort preference
        category_filter: Optional category filter
    """
    # Assume all missing dependencies are completed
    all_task_ids = set(tasks.keys())
    all_mentioned_deps = set()
    for task in tasks.values():
        all_mentioned_deps.update(task.depends)

    # Completed tasks are those mentioned in dependencies but not in active list
    completed_tasks = all_mentioned_deps - all_task_ids

    # Get available tasks
    available = get_available_tasks(tasks, completed_tasks)

    # Apply category filter
    if category_filter:
        available = [t for t in available if t.category == category_filter]
        if not available:
            print(f"\nNo available tasks in category '{category_filter}'")
            return

    # Sort by score
    scored_tasks = [(task, score_task(task, effort_preference)) for task in available]
    scored_tasks.sort(key=lambda x: (-x[1], x[0].id))

    # Show top recommendation
    if scored_tasks:
        best_task, best_score = scored_tasks[0]
        print_task_recommendation(best_task, tasks, best_score)

        # Show other options
        if len(scored_tasks) > 1:
            print("\nOther available tasks (sorted by score):")
            print("\n{:<5} {:<8} {:<8} {:<8} {:<50}".format(
                "ID", "Priority", "Effort", "Score", "Task"))
            print("-" * 90)

            for task, score in scored_tasks[1:11]:  # Show top 10
                print("{:<5} {:<8} {:<8} {:<8.2f} {:<50}".format(
                    f"#{task.id}", task.priority, task.effort, score,
                    task.name[:47] + "..." if len(task.name) > 50 else task.name))

    else:
        print("\nNo available tasks found!")
        return

    # Show blocked tasks
    available_ids = {t.id for t in available}
    blocked = get_blocked_tasks(tasks, available_ids)

    if blocked:
        print(f"\n\nBlocked tasks (waiting for dependencies):")
        for task_id, deps in sorted(blocked.items()):
            task = tasks[task_id]
            deps_str = ", ".join(f"#{d}" for d in deps)
            print(f"  #{task_id}: {task.name[:50]} (waiting for: {deps_str})")

    # Show summary
    print(f"\n\nSummary:")
    print(f"  Total active tasks: {len(tasks)}")
    print(f"  Available now: {len(available)}")
    print(f"  Blocked: {len(blocked)}")
    print(f"  Completed (estimated): {len(completed_tasks)}")

    # Category breakdown
    categories = {}
    for task in available:
        categories[task.category] = categories.get(task.category, 0) + 1

    if categories:
        print(f"\n  Available by category:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive task selection assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/select_task.py                    # Show recommended task
    python scripts/select_task.py --category DevEx   # Filter by category
    python scripts/select_task.py --effort Small     # Prefer small tasks
    python scripts/select_task.py --effort Medium --category Testing
        """
    )
    parser.add_argument(
        '--task-list',
        type=Path,
        default=Path('TASK_LIST.md'),
        help='Path to TASK_LIST.md (default: TASK_LIST.md)'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Filter tasks by category (e.g., DevEx, Testing, Arch)'
    )
    parser.add_argument(
        '--effort',
        choices=['Small', 'Medium', 'Large'],
        help='Prefer tasks with this effort level'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-select best task without interaction'
    )

    args = parser.parse_args()

    # Parse task list
    if not args.task_list.exists():
        print(f"Error: {args.task_list} not found!")
        print("\nNote: TASK_LIST.md has been replaced with a new task management system.")
        print("To view and select tasks, use:")
        print("  python scripts/task_utils.py list")
        print("  python scripts/task_utils.py list --status pending")
        print("\nFor more information, see docs/merge-friendly-tasks.md")
        return 1

    tasks = parse_task_list(args.task_list)

    if not tasks:
        print("No tasks found in TASK_LIST.md!")
        return 1

    # Run interactive mode
    interactive_mode(tasks, args.effort, args.category)

    return 0


if __name__ == '__main__':
    exit(main())
