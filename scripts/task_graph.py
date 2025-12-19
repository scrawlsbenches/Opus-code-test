#!/usr/bin/env python3
"""
Task Dependency Graph Generator

Parses tasks from the merge-friendly task system (tasks/*.json) and generates
a visual dependency graph showing which tasks depend on others.

Usage:
    python scripts/task_graph.py                 # ASCII art output
    python scripts/task_graph.py --format ascii  # ASCII art (default)
    python scripts/task_graph.py --format mermaid # Mermaid diagram
    python scripts/task_graph.py --verbose       # Include task names
    python scripts/task_graph.py --status pending # Only pending tasks

Author: Cortical Text Processor Team
"""

import argparse
from typing import Dict, List, Set
from collections import defaultdict

# Import from the new task system
from task_utils import load_all_tasks, Task


def build_dependency_graph(tasks: List[Task]) -> Dict[str, List[str]]:
    """
    Build a dependency graph from tasks.

    Args:
        tasks: List of Task objects

    Returns:
        Dictionary mapping task ID to list of task IDs it depends on
    """
    graph = {}
    for task in tasks:
        graph[task.id] = task.depends_on if task.depends_on else []
    return graph


def find_roots(graph: Dict[str, List[str]], task_ids: Set[str]) -> List[str]:
    """Find tasks with no dependencies (roots of the graph)."""
    roots = []
    for task_id in task_ids:
        deps = graph.get(task_id, [])
        # Filter deps to only include tasks in our set
        valid_deps = [d for d in deps if d in task_ids]
        if not valid_deps:
            roots.append(task_id)
    return sorted(roots)


def topological_sort(graph: Dict[str, List[str]], task_ids: Set[str]) -> List[str]:
    """
    Perform topological sort on the dependency graph.

    Returns tasks in order where dependencies come before dependents.
    """
    visited = set()
    result = []

    def visit(task_id: str):
        if task_id in visited or task_id not in task_ids:
            return
        visited.add(task_id)
        for dep in graph.get(task_id, []):
            if dep in task_ids:
                visit(dep)
        result.append(task_id)

    for task_id in sorted(task_ids):
        visit(task_id)

    return result


def generate_ascii_graph(
    tasks: List[Task],
    verbose: bool = False
) -> str:
    """
    Generate ASCII art representation of the dependency graph.

    Args:
        tasks: List of Task objects
        verbose: Include task names in output

    Returns:
        ASCII art string
    """
    if not tasks:
        return "No tasks found."

    task_map = {t.id: t for t in tasks}
    task_ids = set(task_map.keys())
    graph = build_dependency_graph(tasks)

    # Find which tasks depend on each task (reverse graph)
    dependents = defaultdict(list)
    for task_id, deps in graph.items():
        for dep in deps:
            if dep in task_ids:
                dependents[dep].append(task_id)

    lines = []
    lines.append("=" * 60)
    lines.append("TASK DEPENDENCY GRAPH")
    lines.append("=" * 60)
    lines.append("")

    # Group by priority
    by_priority = defaultdict(list)
    for task in tasks:
        by_priority[task.priority.lower()].append(task)

    for priority in ['high', 'medium', 'low']:
        priority_tasks = by_priority.get(priority, [])
        if not priority_tasks:
            continue

        icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
        lines.append(f"\n{icon} {priority.upper()} PRIORITY ({len(priority_tasks)} tasks)")
        lines.append("-" * 40)

        for task in sorted(priority_tasks, key=lambda t: t.id):
            # Build task line
            status_icon = {
                'pending': '○',
                'in_progress': '◐',
                'completed': '●',
                'deferred': '◌'
            }.get(task.status, '?')

            task_line = f"  {status_icon} {task.id}"
            if verbose:
                task_line += f": {task.title[:40]}"

            # Show dependencies
            deps = [d for d in graph.get(task.id, []) if d in task_ids]
            if deps:
                task_line += f" ← depends on: {', '.join(deps)}"

            # Show what depends on this
            children = dependents.get(task.id, [])
            if children:
                task_line += f" → blocks: {', '.join(children)}"

            lines.append(task_line)

    # Summary
    lines.append("")
    lines.append("=" * 60)
    lines.append("LEGEND")
    lines.append("  ○ pending  ◐ in_progress  ● completed  ◌ deferred")
    lines.append("  ← depends on (must complete first)")
    lines.append("  → blocks (waiting on this)")

    # Stats
    pending = sum(1 for t in tasks if t.status == 'pending')
    in_progress = sum(1 for t in tasks if t.status == 'in_progress')
    completed = sum(1 for t in tasks if t.status == 'completed')

    lines.append("")
    lines.append(f"STATS: {pending} pending, {in_progress} in progress, {completed} completed")

    return "\n".join(lines)


def generate_mermaid_graph(
    tasks: List[Task],
    verbose: bool = False
) -> str:
    """
    Generate Mermaid diagram of the dependency graph.

    Args:
        tasks: List of Task objects
        verbose: Include task names in nodes

    Returns:
        Mermaid diagram string
    """
    if not tasks:
        return "graph TD\n    empty[No tasks found]"

    task_map = {t.id: t for t in tasks}
    task_ids = set(task_map.keys())
    graph = build_dependency_graph(tasks)

    lines = ["graph TD"]

    # Define node styles by status
    lines.append("    %% Status styles")
    lines.append("    classDef pending fill:#fff,stroke:#333")
    lines.append("    classDef in_progress fill:#ffd700,stroke:#333")
    lines.append("    classDef completed fill:#90EE90,stroke:#333")
    lines.append("    classDef deferred fill:#ddd,stroke:#999")
    lines.append("")

    # Add nodes
    for task in tasks:
        # Sanitize ID for Mermaid (replace special chars)
        node_id = task.id.replace('-', '_').replace('.', '_')

        if verbose:
            label = f"{task.id}<br/>{task.title[:30]}"
        else:
            label = task.id

        # Shape by priority
        if task.priority == 'high':
            lines.append(f"    {node_id}[/{label}\\]")  # Trapezoid
        elif task.priority == 'low':
            lines.append(f"    {node_id}({label})")  # Rounded
        else:
            lines.append(f"    {node_id}[{label}]")  # Rectangle

        # Apply status class
        lines.append(f"    class {node_id} {task.status}")

    lines.append("")

    # Add edges
    for task in tasks:
        node_id = task.id.replace('-', '_').replace('.', '_')
        for dep in graph.get(task.id, []):
            if dep in task_ids:
                dep_id = dep.replace('-', '_').replace('.', '_')
                lines.append(f"    {dep_id} --> {node_id}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate task dependency graph from merge-friendly task system"
    )
    parser.add_argument(
        '--format', '-f',
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
        '--status', '-s',
        choices=['pending', 'in_progress', 'completed', 'deferred', 'all'],
        default='all',
        help='Filter by status (default: all)'
    )
    parser.add_argument(
        '--priority', '-p',
        choices=['high', 'medium', 'low', 'all'],
        default='all',
        help='Filter by priority (default: all)'
    )

    args = parser.parse_args()

    # Load tasks from new system
    all_tasks = load_all_tasks()

    # Filter by status
    if args.status != 'all':
        all_tasks = [t for t in all_tasks if t.status == args.status]

    # Filter by priority
    if args.priority != 'all':
        all_tasks = [t for t in all_tasks if t.priority.lower() == args.priority]

    # Generate graph
    if args.format == 'mermaid':
        print(generate_mermaid_graph(all_tasks, args.verbose))
    else:
        print(generate_ascii_graph(all_tasks, args.verbose))


if __name__ == '__main__':
    main()
