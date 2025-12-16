#!/usr/bin/env python3
"""
Task Dependency Graph Generator

Parses TASK_LIST.md and generates a visual dependency graph showing which tasks
depend on others. Useful for understanding task ordering and planning work.

Usage:
    python scripts/task_graph.py                 # ASCII art output
    python scripts/task_graph.py --format ascii  # ASCII art (default)
    python scripts/task_graph.py --format mermaid # Mermaid diagram
    python scripts/task_graph.py --verbose       # Include task names

Author: Cortical Text Processor Team
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class Task:
    """Represents a task with its metadata."""

    def __init__(self, task_id: int, name: str, category: str,
                 depends: List[int], effort: str, priority: str):
        self.id = task_id
        self.name = name
        self.category = category
        self.depends = depends  # List of task IDs this task depends on
        self.effort = effort
        self.priority = priority

    def __repr__(self):
        deps = f", depends={self.depends}" if self.depends else ""
        return f"Task({self.id}: {self.name[:30]}...{deps})"


def parse_task_list(file_path: Path) -> Dict[int, Task]:
    """
    Parse TASK_LIST.md and extract all tasks with their dependencies.

    Args:
        file_path: Path to TASK_LIST.md

    Returns:
        Dictionary mapping task ID to Task object
    """
    tasks = {}
    current_priority = None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find priority sections
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect priority sections
        if '### 🟠 High' in line:
            current_priority = 'High'
        elif '### 🟡 Medium' in line:
            current_priority = 'Medium'
        elif '### 🟢 Low' in line:
            current_priority = 'Low'
        elif '### ⏸️ Deferred' in line:
            current_priority = 'Deferred'

        # Parse table rows (skip header and separator)
        if line.startswith('|') and not line.startswith('|---|') and not line.startswith('| #'):
            parts = [p.strip() for p in line.split('|')[1:-1]]  # Remove empty first/last

            if len(parts) >= 4 and parts[0].isdigit():
                task_id = int(parts[0])
                name = parts[1]
                category = parts[2]
                depends_str = parts[3]
                effort = parts[4] if len(parts) > 4 else 'Unknown'

                # Parse dependencies
                depends = []
                if depends_str and depends_str != '-':
                    # Handle comma-separated dependencies: "132, 133" or single: "132"
                    for dep in depends_str.split(','):
                        dep = dep.strip()
                        if dep.isdigit():
                            depends.append(int(dep))

                tasks[task_id] = Task(
                    task_id=task_id,
                    name=name,
                    category=category,
                    depends=depends,
                    effort=effort,
                    priority=current_priority or 'Unknown'
                )

        i += 1

    return tasks


def build_dependency_graph(tasks: Dict[int, Task]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build forward and reverse dependency graphs.

    Args:
        tasks: Dictionary of tasks

    Returns:
        Tuple of (forward_deps, reverse_deps)
        - forward_deps[task_id] = list of tasks that depend on this task
        - reverse_deps[task_id] = list of tasks this task depends on
    """
    forward_deps = defaultdict(list)  # task_id -> tasks that depend on it
    reverse_deps = defaultdict(list)  # task_id -> tasks it depends on

    for task_id, task in tasks.items():
        reverse_deps[task_id] = task.depends

        for dep_id in task.depends:
            forward_deps[dep_id].append(task_id)

    return dict(forward_deps), dict(reverse_deps)


def topological_sort(tasks: Dict[int, Task], reverse_deps: Dict[int, List[int]]) -> List[List[int]]:
    """
    Perform topological sort to find task ordering levels.

    Returns:
        List of levels, where each level is a list of task IDs that can be done in parallel
    """
    # Count dependencies for each task
    in_degree = {task_id: len(deps) for task_id, deps in reverse_deps.items()}

    # Add tasks with no dependencies
    for task_id in tasks:
        if task_id not in in_degree:
            in_degree[task_id] = 0

    levels = []
    remaining = set(tasks.keys())

    while remaining:
        # Find all tasks with no remaining dependencies
        current_level = [tid for tid in remaining if in_degree[tid] == 0]

        if not current_level:
            # Circular dependency detected
            break

        levels.append(current_level)

        # Remove this level from remaining
        for tid in current_level:
            remaining.remove(tid)

            # Reduce in-degree for tasks that depend on this one
            for dep_task_id in tasks.keys():
                if tid in reverse_deps.get(dep_task_id, []):
                    in_degree[dep_task_id] -= 1

    return levels


def generate_ascii_graph(tasks: Dict[int, Task], forward_deps: Dict[int, List[int]],
                         reverse_deps: Dict[int, List[int]], verbose: bool = False) -> str:
    """
    Generate ASCII art dependency graph.

    Args:
        tasks: Dictionary of tasks
        forward_deps: Forward dependency mapping
        reverse_deps: Reverse dependency mapping
        verbose: Include task names

    Returns:
        ASCII art string
    """
    lines = []
    lines.append("Task Dependency Graph")
    lines.append("=" * 60)
    lines.append("")

    # Get topological sort
    levels = topological_sort(tasks, reverse_deps)

    if not levels:
        lines.append("No tasks or circular dependencies detected!")
        return '\n'.join(lines)

    lines.append(f"Found {len(levels)} dependency levels:")
    lines.append("")

    for level_num, level in enumerate(levels, 1):
        lines.append(f"Level {level_num} (can be done in parallel):")
        for task_id in sorted(level):
            task = tasks[task_id]
            deps_str = ""
            if task.depends:
                deps_str = f" [depends on: {', '.join(map(str, task.depends))}]"

            blocked_str = ""
            if task_id in forward_deps:
                blocked_by = forward_deps[task_id]
                blocked_str = f" [blocks: {', '.join(map(str, blocked_by))}]"

            if verbose:
                lines.append(f"  #{task_id}: {task.name[:50]}{deps_str}{blocked_str}")
            else:
                lines.append(f"  #{task_id} ({task.priority}, {task.effort}){deps_str}{blocked_str}")
        lines.append("")

    # Show tasks with dependencies
    lines.append("Dependency Chains:")
    lines.append("-" * 60)

    tasks_with_deps = [(tid, t) for tid, t in tasks.items() if t.depends]
    tasks_with_deps.sort(key=lambda x: x[0])

    if not tasks_with_deps:
        lines.append("No dependencies found - all tasks are independent!")
    else:
        for task_id, task in tasks_with_deps:
            dep_chain = " -> ".join(f"#{d}" for d in task.depends)
            lines.append(f"  {dep_chain} -> #{task_id}")
            if verbose:
                lines.append(f"    ({task.name[:60]})")

    lines.append("")
    lines.append(f"Total tasks: {len(tasks)}")
    lines.append(f"Tasks with dependencies: {len(tasks_with_deps)}")
    lines.append(f"Independent tasks: {len(tasks) - len(tasks_with_deps)}")

    return '\n'.join(lines)


def generate_mermaid_graph(tasks: Dict[int, Task], forward_deps: Dict[int, List[int]],
                           reverse_deps: Dict[int, List[int]]) -> str:
    """
    Generate Mermaid diagram code.

    Args:
        tasks: Dictionary of tasks
        forward_deps: Forward dependency mapping
        reverse_deps: Reverse dependency mapping

    Returns:
        Mermaid diagram code
    """
    lines = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("")

    # Define nodes with styling based on priority
    for task_id, task in sorted(tasks.items()):
        label = f"#{task_id}: {task.name[:40]}"

        # Style based on priority
        if task.priority == 'High':
            style = ":::high"
        elif task.priority == 'Medium':
            style = ":::medium"
        elif task.priority == 'Low':
            style = ":::low"
        else:
            style = ""

        lines.append(f"    T{task_id}[\"{label}\"]{style}")

    lines.append("")

    # Add edges
    for task_id, task in sorted(tasks.items()):
        for dep_id in task.depends:
            lines.append(f"    T{dep_id} --> T{task_id}")

    lines.append("")

    # Add styling classes
    lines.append("    classDef high fill:#ff9999,stroke:#cc0000,stroke-width:2px")
    lines.append("    classDef medium fill:#ffcc99,stroke:#ff9900,stroke-width:2px")
    lines.append("    classDef low fill:#99ff99,stroke:#00cc00,stroke-width:2px")

    lines.append("```")
    lines.append("")
    lines.append("<!-- Copy the above code to a Mermaid live editor: https://mermaid.live -->")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate task dependency graph from TASK_LIST.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/task_graph.py                    # ASCII art
    python scripts/task_graph.py --format mermaid   # Mermaid diagram
    python scripts/task_graph.py --verbose          # Include task names
        """
    )
    parser.add_argument(
        '--format',
        choices=['ascii', 'mermaid'],
        default='ascii',
        help='Output format (default: ascii)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Include task names in output'
    )
    parser.add_argument(
        '--task-list',
        type=Path,
        default=Path('TASK_LIST.md'),
        help='Path to TASK_LIST.md (default: TASK_LIST.md)'
    )

    args = parser.parse_args()

    # Parse task list
    if not args.task_list.exists():
        print(f"Error: {args.task_list} not found!")
        print("\nNote: TASK_LIST.md has been replaced with a new task management system.")
        print("To view tasks, use:")
        print("  python scripts/task_utils.py list")
        print("\nFor more information, see docs/merge-friendly-tasks.md")
        return 1

    tasks = parse_task_list(args.task_list)

    if not tasks:
        print("No tasks found in TASK_LIST.md!")
        return 1

    # Build dependency graph
    forward_deps, reverse_deps = build_dependency_graph(tasks)

    # Generate output
    if args.format == 'ascii':
        output = generate_ascii_graph(tasks, forward_deps, reverse_deps, args.verbose)
    else:  # mermaid
        output = generate_mermaid_graph(tasks, forward_deps, reverse_deps)

    print(output)
    return 0


if __name__ == '__main__':
    exit(main())
