#!/usr/bin/env python3
"""
GoT Priority Task Executor

Implements prioritized task execution based on:
1. Success potential (can it be completed?)
2. Performance impact (will it speed up work?)
3. Importance (how critical is it?)

Usage:
    python scripts/got_priority_executor.py list          # Show prioritized tasks
    python scripts/got_priority_executor.py next          # Get next task to work on
    python scripts/got_priority_executor.py plan          # Generate execution plan
    python scripts/got_priority_executor.py breakdown ID  # Break task into subtasks
    python scripts/got_priority_executor.py status        # Show progress dashboard
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Priority weights based on user preference: Success > Performance > Importance
PRIORITY_WEIGHTS = {
    'success': 0.5,      # Highest weight - can it be completed?
    'performance': 0.3,  # Medium weight - will it speed things up?
    'importance': 0.2,   # Lower weight - how critical?
}

# Priority level scores
PRIORITY_SCORES = {
    'critical': 100,
    'high': 75,
    'medium': 50,
    'low': 25,
}

# Category success likelihood (based on typical completion rates)
CATEGORY_SUCCESS = {
    'bugfix': 0.9,       # High success - well-defined scope
    'feature': 0.7,      # Medium - may have unknowns
    'refactor': 0.8,     # Good success - clear patterns
    'test': 0.95,        # Very high - mechanical work
    'docs': 0.95,        # Very high - straightforward
    'arch': 0.5,         # Lower - complex decisions
    'research': 0.6,     # Medium - may not find answers
}

# Performance impact by category
CATEGORY_PERFORMANCE = {
    'bugfix': 0.8,       # Fixes bottlenecks
    'feature': 0.5,      # New capability
    'refactor': 0.9,     # Code quality
    'test': 0.7,         # Confidence
    'docs': 0.4,         # Understanding
    'arch': 0.95,        # Foundation
    'research': 0.3,     # Knowledge
}


@dataclass
class PrioritizedTask:
    """Task with computed priority score."""
    id: str
    title: str
    priority: str
    category: str
    status: str
    score: float = 0.0
    success_score: float = 0.0
    performance_score: float = 0.0
    importance_score: float = 0.0
    blockers: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)


def get_tasks() -> List[Dict]:
    """Get all tasks from GoT."""
    try:
        result = subprocess.run(
            ['python', 'scripts/got_utils.py', 'task', 'list', '--format', 'json'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Fallback: parse table output
    result = subprocess.run(
        ['python', 'scripts/got_utils.py', 'task', 'list', '--status', 'pending'],
        capture_output=True, text=True, timeout=30
    )

    tasks = []
    for line in result.stdout.split('\n'):
        if line.startswith('│') and 'T-' in line:
            parts = [p.strip() for p in line.split('│')[1:-1]]
            if len(parts) >= 4:
                tasks.append({
                    'id': parts[0],
                    'title': parts[1],
                    'status': parts[2],
                    'priority': parts[3].lower() if len(parts) > 3 else 'medium',
                    'category': 'feature',  # Default
                })
    return tasks


def get_blockers(task_id: str) -> List[str]:
    """Get tasks blocking this task."""
    try:
        result = subprocess.run(
            ['python', 'scripts/got_utils.py', 'query', f'what blocks {task_id}'],
            capture_output=True, text=True, timeout=10
        )
        # Parse blocker IDs from output
        # Skip first line (query echo) and exclude the task's own ID
        blockers = []
        lines = result.stdout.split('\n')
        for line in lines[1:]:  # Skip query echo line
            if 'T-' in line:
                # Extract task ID
                import re
                match = re.search(r'T-\d{8}-\d{6}-[a-f0-9]{4}', line)
                if match and match.group() != task_id:  # Exclude self
                    blockers.append(match.group())
        return blockers
    except subprocess.TimeoutExpired:
        return []


def infer_category(title: str) -> str:
    """Infer task category from title."""
    title_lower = title.lower()

    if any(w in title_lower for w in ['fix', 'bug', 'error', 'issue']):
        return 'bugfix'
    if any(w in title_lower for w in ['test', 'coverage', 'verify']):
        return 'test'
    if any(w in title_lower for w in ['doc', 'readme', 'comment']):
        return 'docs'
    if any(w in title_lower for w in ['refactor', 'clean', 'reorganize']):
        return 'refactor'
    if any(w in title_lower for w in ['architect', 'design', 'structure']):
        return 'arch'
    if any(w in title_lower for w in ['research', 'investigate', 'explore']):
        return 'research'

    return 'feature'


def calculate_priority(task: Dict) -> PrioritizedTask:
    """Calculate composite priority score for a task."""
    task_id = task.get('id', '')
    title = task.get('title', '')
    priority = task.get('priority', 'medium').lower()
    category = task.get('category', infer_category(title))
    status = task.get('status', 'pending')

    # Base importance from priority level
    importance = PRIORITY_SCORES.get(priority, 50) / 100.0

    # Success likelihood from category
    success = CATEGORY_SUCCESS.get(category, 0.7)

    # Performance impact from category
    performance = CATEGORY_PERFORMANCE.get(category, 0.5)

    # Check blockers (blocked tasks have reduced score)
    blockers = get_blockers(task_id) if task_id else []
    if blockers:
        success *= 0.3  # Significantly reduce score if blocked

    # Composite score using user's weights
    score = (
        success * PRIORITY_WEIGHTS['success'] +
        performance * PRIORITY_WEIGHTS['performance'] +
        importance * PRIORITY_WEIGHTS['importance']
    )

    return PrioritizedTask(
        id=task_id,
        title=title,
        priority=priority,
        category=category,
        status=status,
        score=score,
        success_score=success,
        performance_score=performance,
        importance_score=importance,
        blockers=blockers,
    )


def prioritize_tasks(tasks: List[Dict]) -> List[PrioritizedTask]:
    """Prioritize all tasks and return sorted list."""
    prioritized = [calculate_priority(t) for t in tasks]
    return sorted(prioritized, key=lambda t: t.score, reverse=True)


def suggest_breakdown(task: PrioritizedTask) -> List[str]:
    """Suggest subtask breakdown for complex tasks."""
    title = task.title.lower()
    subtasks = []

    # Generic breakdown patterns
    if 'implement' in title:
        subtasks = [
            f"Research existing patterns for: {task.title}",
            f"Design API/interface for: {task.title}",
            f"Implement core functionality for: {task.title}",
            f"Add tests for: {task.title}",
            f"Update documentation for: {task.title}",
        ]
    elif 'add' in title:
        subtasks = [
            f"Define requirements for: {task.title}",
            f"Implement: {task.title}",
            f"Test: {task.title}",
        ]
    elif 'fix' in title:
        subtasks = [
            f"Reproduce issue: {task.title}",
            f"Identify root cause: {task.title}",
            f"Implement fix: {task.title}",
            f"Add regression test: {task.title}",
        ]
    else:
        subtasks = [
            f"Analyze scope: {task.title}",
            f"Implement: {task.title}",
            f"Verify: {task.title}",
        ]

    return subtasks


def print_status_dashboard():
    """Print progress status dashboard."""
    try:
        result = subprocess.run(
            ['python', 'scripts/got_utils.py', 'stats'],
            capture_output=True, text=True, timeout=10
        )
        print(result.stdout)
    except subprocess.TimeoutExpired:
        print("Could not get stats")

    print("\n" + "=" * 60)
    print("PRIORITIZED NEXT ACTIONS")
    print("=" * 60)

    tasks = get_tasks()
    pending = [t for t in tasks if t.get('status') == 'pending']
    prioritized = prioritize_tasks(pending[:20])  # Top 20

    for i, task in enumerate(prioritized[:5], 1):
        blocked = " [BLOCKED]" if task.blockers else ""
        print(f"\n{i}. [{task.category.upper()}] {task.title[:50]}{blocked}")
        print(f"   Score: {task.score:.2f} (S:{task.success_score:.2f} P:{task.performance_score:.2f} I:{task.importance_score:.2f})")
        print(f"   ID: {task.id}")


def print_execution_plan():
    """Generate and print execution plan."""
    tasks = get_tasks()
    pending = [t for t in tasks if t.get('status') == 'pending']
    prioritized = prioritize_tasks(pending)

    # Group by category
    by_category = {}
    for task in prioritized:
        cat = task.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(task)

    print("=" * 60)
    print("EXECUTION PLAN")
    print("=" * 60)
    print(f"\nTotal pending: {len(pending)} tasks")
    print(f"Categories: {', '.join(by_category.keys())}")

    # Phase 1: Quick wins (high success, any priority)
    quick_wins = [t for t in prioritized if t.success_score >= 0.9 and not t.blockers][:5]
    print(f"\n📌 PHASE 1: Quick Wins ({len(quick_wins)} tasks)")
    for t in quick_wins:
        print(f"   • {t.title[:50]} [{t.category}]")

    # Phase 2: High impact (high performance score)
    high_impact = [t for t in prioritized if t.performance_score >= 0.8 and t not in quick_wins and not t.blockers][:5]
    print(f"\n🚀 PHASE 2: High Impact ({len(high_impact)} tasks)")
    for t in high_impact:
        print(f"   • {t.title[:50]} [{t.category}]")

    # Phase 3: Critical path (critical priority)
    critical = [t for t in prioritized if t.priority == 'critical' and t not in quick_wins and t not in high_impact][:5]
    print(f"\n🔴 PHASE 3: Critical ({len(critical)} tasks)")
    for t in critical:
        blocked = " [BLOCKED]" if t.blockers else ""
        print(f"   • {t.title[:50]}{blocked}")

    # Blocked tasks
    blocked_tasks = [t for t in prioritized if t.blockers]
    if blocked_tasks:
        print(f"\n⛔ BLOCKED ({len(blocked_tasks)} tasks)")
        for t in blocked_tasks[:3]:
            print(f"   • {t.title[:40]} - blocked by {len(t.blockers)} task(s)")


def main():
    parser = argparse.ArgumentParser(description='GoT Priority Task Executor')
    parser.add_argument('command', choices=['list', 'next', 'plan', 'breakdown', 'status'],
                        help='Command to execute')
    parser.add_argument('task_id', nargs='?', help='Task ID for breakdown command')

    args = parser.parse_args()

    if args.command == 'list':
        tasks = get_tasks()
        pending = [t for t in tasks if t.get('status') == 'pending']
        prioritized = prioritize_tasks(pending)

        print(f"{'Score':<8} {'Category':<10} {'Priority':<10} {'Title':<40}")
        print("-" * 70)
        for task in prioritized[:20]:
            blocked = "⛔" if task.blockers else ""
            print(f"{task.score:.3f}   {task.category:<10} {task.priority:<10} {task.title[:38]}{blocked}")

    elif args.command == 'next':
        tasks = get_tasks()
        pending = [t for t in tasks if t.get('status') == 'pending']
        prioritized = prioritize_tasks(pending)

        # Find first unblocked task
        for task in prioritized:
            if not task.blockers:
                print(f"NEXT TASK: {task.title}")
                print(f"ID: {task.id}")
                print(f"Category: {task.category}")
                print(f"Priority: {task.priority}")
                print(f"Score: {task.score:.3f}")
                print(f"\nSuggested breakdown:")
                for i, sub in enumerate(suggest_breakdown(task), 1):
                    print(f"  {i}. {sub}")
                break
        else:
            print("No unblocked tasks available")

    elif args.command == 'plan':
        print_execution_plan()

    elif args.command == 'breakdown':
        if not args.task_id:
            print("Error: task_id required for breakdown command")
            sys.exit(1)

        tasks = get_tasks()
        task = next((t for t in tasks if t.get('id') == args.task_id), None)
        if task:
            pt = calculate_priority(task)
            print(f"Breakdown for: {pt.title}")
            print("-" * 50)
            for i, sub in enumerate(suggest_breakdown(pt), 1):
                print(f"  {i}. {sub}")
        else:
            print(f"Task not found: {args.task_id}")

    elif args.command == 'status':
        print_status_dashboard()


if __name__ == '__main__':
    main()
