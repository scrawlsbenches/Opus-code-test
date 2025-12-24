"""
Backlog CLI commands for GoT system.

The backlog is the collection of tasks that aren't assigned to any sprint.
This module provides a user-friendly interface for managing the backlog,
which is essentially a wrapper around orphan detection functionality.

Commands:
    got backlog list    - List backlog items sorted by priority
    got backlog promote - Promote task from backlog to a sprint
    got backlog review  - Interactive backlog review with suggestions
    got backlog stats   - Backlog statistics

The backlog serves as a holding area for tasks before they're planned into
sprints. Tasks in the backlog are "orphans" - they have no sprint assignment.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter

from ..orphan import OrphanDetector


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_backlog_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backlog list' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)
    orphans = detector.find_orphan_tasks()

    # Collect task data
    backlog_items = []
    for task_id in orphans:
        task = got_manager.get_task(task_id)
        if task:
            # Filter by status if specified
            if hasattr(args, 'status') and args.status and task.status != args.status:
                continue
            backlog_items.append({
                "id": task_id,
                "title": task.title,
                "status": task.status,
                "priority": task.priority,
                "created_at": getattr(task, 'created_at', ''),
            })

    # Sort by priority (default) or creation date
    sort_by = getattr(args, 'sort', 'priority')
    if sort_by == 'priority':
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        backlog_items.sort(key=lambda t: priority_order.get(t['priority'], 99))
    elif sort_by == 'created':
        backlog_items.sort(key=lambda t: t.get('created_at', ''), reverse=True)

    if getattr(args, 'json', False):
        print(json.dumps(backlog_items, indent=2))
    else:
        if not backlog_items:
            print("📋 Backlog is empty! All tasks are assigned to sprints.")
            return 0

        print(f"\n📋 BACKLOG ({len(backlog_items)} items)")
        print("=" * 90)
        print(f"{'#':<3} {'Priority':<10} {'Status':<12} {'ID':<30} {'Title':<32}")
        print("-" * 90)

        for idx, t in enumerate(backlog_items, 1):
            priority_markers = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            marker = priority_markers.get(t['priority'], '⚪')
            title = t['title'][:30] + '..' if len(t['title']) > 32 else t['title']
            print(f"{idx:<3} {marker} {t['priority']:<8} {t['status']:<12} {t['id']:<30} {title}")

        print("=" * 90)
        print("\nCommands:")
        print("  got backlog promote TASK_ID --sprint S-XXX  # Move to sprint")
        print("  got backlog review TASK_ID                  # Get suggestions")

    return 0


def cmd_backlog_promote(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backlog promote' command - moves task to a sprint."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    task_id = args.task_id
    sprint_id = getattr(args, 'sprint', None)

    # If no sprint specified, use current sprint
    if sprint_id is None:
        current = got_manager.get_current_sprint()
        if current is None:
            print("Error: No current sprint. Specify --sprint SPRINT_ID")
            return 1
        sprint_id = current.id

    # Check if task exists
    task = got_manager.get_task(task_id)
    if task is None:
        print(f"Error: Task not found: {task_id}")
        return 1

    # Check if sprint exists
    sprint = got_manager.get_sprint(sprint_id)
    if sprint is None:
        print(f"Error: Sprint not found: {sprint_id}")
        return 1

    # Add task to sprint
    try:
        got_manager.add_task_to_sprint(task_id, sprint_id)
        print(f"✅ Promoted: {task_id}")
        print(f"   → Sprint: {sprint_id} ({sprint.title})")
    except Exception as e:
        print(f"Error promoting task: {e}")
        return 1

    return 0


def cmd_backlog_review(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backlog review' command - review a task with suggestions."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    task_id = args.task_id
    task = got_manager.get_task(task_id)
    if task is None:
        print(f"Error: Task not found: {task_id}")
        return 1

    detector = OrphanDetector(got_manager)

    print(f"\n📋 BACKLOG REVIEW: {task_id}")
    print("=" * 60)
    print(f"Title:    {task.title}")
    print(f"Status:   {task.status}")
    print(f"Priority: {task.priority}")
    if task.description:
        print(f"Description: {task.description[:100]}...")
    print()

    # Sprint suggestions
    print("📅 SPRINT SUGGESTIONS:")
    sprint_suggestions = detector.suggest_sprint(task_id)
    if sprint_suggestions:
        for i, s in enumerate(sprint_suggestions[:3], 1):
            current = " [CURRENT]" if s.is_current else ""
            print(f"  {i}. {s.sprint_id}: {s.sprint_title}{current}")
            print(f"     Confidence: {s.confidence:.0%} - {s.reason}")
    else:
        print("  No sprint suggestions available")
    print()

    # Connection suggestions
    print("🔗 RELATED TASKS:")
    connection_suggestions = detector.suggest_connections(task_id)
    if connection_suggestions:
        for c in connection_suggestions[:5]:
            other = got_manager.get_task(c.target_id)
            title = other.title[:40] if other else "Unknown"
            print(f"  → {c.target_id}: {title}")
            print(f"    {c.edge_type} (confidence: {c.confidence:.0%})")
    else:
        print("  No related tasks found")

    print("\n" + "=" * 60)
    print("Actions:")
    print(f"  got backlog promote {task_id} --sprint <SPRINT_ID>")
    print(f"  got task start {task_id}")

    return 0


def cmd_backlog_stats(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backlog stats' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)
    report = detector.generate_orphan_report()

    # Count by priority
    priority_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    status_counts = {'pending': 0, 'in_progress': 0, 'completed': 0, 'blocked': 0}

    for task_id in report.orphan_tasks:
        task = got_manager.get_task(task_id)
        if task:
            priority_counts[task.priority] = priority_counts.get(task.priority, 0) + 1
            status_counts[task.status] = status_counts.get(task.status, 0) + 1

    if getattr(args, 'json', False):
        print(json.dumps({
            'total_backlog': len(report.orphan_tasks),
            'total_tasks': report.total_tasks,
            'backlog_rate': report.orphan_rate,
            'by_priority': priority_counts,
            'by_status': status_counts,
        }, indent=2))
    else:
        print("\n📊 BACKLOG STATISTICS")
        print("=" * 40)
        print(f"Total in backlog: {len(report.orphan_tasks)}")
        print(f"Total tasks:      {report.total_tasks}")
        print(f"Backlog rate:     {report.orphan_rate:.1f}%")
        print()
        print("By Priority:")
        for p in ['critical', 'high', 'medium', 'low']:
            count = priority_counts.get(p, 0)
            if count > 0:
                bar = '█' * min(count, 30)
                print(f"  {p:<8}: {bar} {count}")
        print()
        print("By Status:")
        for s in ['pending', 'in_progress', 'completed', 'blocked']:
            count = status_counts.get(s, 0)
            if count > 0:
                bar = '█' * min(count, 30)
                print(f"  {s:<12}: {bar} {count}")
        print("=" * 40)

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_backlog_parser(subparsers) -> None:
    """
    Set up argparse subparsers for backlog commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    backlog_parser = subparsers.add_parser(
        "backlog",
        help="Manage task backlog (unassigned tasks)"
    )
    backlog_subparsers = backlog_parser.add_subparsers(
        dest="backlog_command",
        help="Backlog subcommands"
    )

    # backlog list
    list_parser = backlog_subparsers.add_parser(
        "list",
        help="List backlog items sorted by priority"
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    list_parser.add_argument(
        "--sort",
        choices=["priority", "created"],
        default="priority",
        help="Sort by priority (default) or creation date"
    )
    list_parser.add_argument(
        "--status",
        choices=["pending", "in_progress", "completed", "blocked"],
        help="Filter by status"
    )

    # backlog promote TASK_ID
    promote_parser = backlog_subparsers.add_parser(
        "promote",
        help="Promote task from backlog to a sprint"
    )
    promote_parser.add_argument("task_id", help="Task ID to promote")
    promote_parser.add_argument(
        "--sprint",
        help="Target sprint ID (default: current sprint)"
    )

    # backlog review TASK_ID
    review_parser = backlog_subparsers.add_parser(
        "review",
        help="Review a backlog item with suggestions"
    )
    review_parser.add_argument("task_id", help="Task ID to review")

    # backlog stats
    stats_parser = backlog_subparsers.add_parser(
        "stats",
        help="Show backlog statistics"
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )


def handle_backlog_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route backlog subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: TransactionalGoTAdapter instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'backlog_command') or args.backlog_command is None:
        print("Error: No backlog subcommand specified. Use 'got backlog --help' for usage.")
        return 1

    command_handlers = {
        "list": cmd_backlog_list,
        "promote": cmd_backlog_promote,
        "review": cmd_backlog_review,
        "stats": cmd_backlog_stats,
    }

    handler = command_handlers.get(args.backlog_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown backlog subcommand: {args.backlog_command}")
    return 1
