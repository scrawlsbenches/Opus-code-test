"""
Orphan detection CLI commands for GoT system.

Provides commands for detecting and managing orphan entities:
- Report orphan status
- Suggest sprint assignments
- Auto-link orphan tasks
- Check new tasks for orphan status

This module integrates with got_utils.py CLI.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter

from ..orphan import OrphanDetector, OrphanReport


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_orphan_report(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got orphan report' command."""
    # Get the underlying GoTManager from adapter
    # Try multiple attribute names for compatibility
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)

    if args.json:
        report = detector.generate_orphan_report()
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(detector.get_orphan_summary())

    return 0


def cmd_orphan_check(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got orphan check TASK_ID' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)

    # Check if task is orphan
    is_orphan = detector.is_orphan(args.task_id)

    if args.json:
        result = detector.check_on_create(args.task_id)
        result['is_orphan'] = is_orphan
        print(json.dumps(result, indent=2))
    else:
        task = got_manager.get_task(args.task_id)
        if task is None:
            print(f"Error: Task not found: {args.task_id}")
            return 1

        print(f"Task: {args.task_id}")
        print(f"Title: {task.title}")
        print(f"Orphan: {'Yes' if is_orphan else 'No'}")

        if is_orphan:
            print("\nSprint Suggestions:")
            suggestions = detector.suggest_sprint(args.task_id)
            if suggestions:
                for s in suggestions[:3]:
                    current = " (current)" if s.is_current else ""
                    print(f"  - {s.sprint_id}: {s.sprint_title}{current}")
                    print(f"    Confidence: {s.confidence:.0%}, Reason: {s.reason}")
            else:
                print("  No sprint suggestions available")

            print("\nConnection Suggestions:")
            connections = detector.suggest_connections(args.task_id)
            if connections:
                for c in connections[:5]:
                    other_task = got_manager.get_task(c.target_id)
                    title = other_task.title if other_task else "Unknown"
                    print(f"  - {c.target_id}: {title[:40]}")
                    print(f"    Type: {c.edge_type}, Confidence: {c.confidence:.0%}")
            else:
                print("  No connection suggestions available")

    return 0


def cmd_orphan_suggest_sprint(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got orphan suggest-sprint TASK_ID' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)
    suggestions = detector.suggest_sprint(args.task_id)

    if args.json:
        print(json.dumps([s.to_dict() for s in suggestions], indent=2))
    else:
        if not suggestions:
            print(f"No sprint suggestions for {args.task_id}")
            return 0

        print(f"Sprint suggestions for {args.task_id}:\n")
        for s in suggestions:
            current = " [CURRENT]" if s.is_current else ""
            print(f"  {s.sprint_id}: {s.sprint_title}{current}")
            print(f"    Confidence: {s.confidence:.0%}")
            print(f"    Reason: {s.reason}")
            print()

    return 0


def cmd_orphan_suggest_links(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got orphan suggest-links TASK_ID' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)
    suggestions = detector.suggest_connections(args.task_id)

    if args.json:
        print(json.dumps([c.to_dict() for c in suggestions], indent=2))
    else:
        if not suggestions:
            print(f"No connection suggestions for {args.task_id}")
            return 0

        print(f"Connection suggestions for {args.task_id}:\n")
        for c in suggestions:
            other_task = got_manager.get_task(c.target_id)
            title = other_task.title if other_task else "Unknown"
            print(f"  → {c.target_id}")
            print(f"    Title: {title[:50]}")
            print(f"    Edge Type: {c.edge_type}")
            print(f"    Confidence: {c.confidence:.0%}")
            print(f"    Reason: {c.reason}")
            print()

    return 0


def cmd_orphan_auto_link(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got orphan auto-link' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)

    # Get orphan tasks if not specified
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = detector.find_orphan_tasks()

    if not task_ids:
        print("No orphan tasks to link")
        return 0

    if args.dry_run:
        print(f"Would link {len(task_ids)} orphan tasks to sprint {args.sprint or 'current'}")
        for task_id in task_ids[:10]:
            task = got_manager.get_task(task_id)
            title = task.title if task else "Unknown"
            print(f"  - {task_id}: {title[:50]}")
        if len(task_ids) > 10:
            print(f"  ... and {len(task_ids) - 10} more")
        return 0

    linked = detector.auto_link_to_sprint(task_ids, args.sprint)

    if args.json:
        print(json.dumps([{"task_id": t, "sprint_id": s} for t, s in linked], indent=2))
    else:
        print(f"Linked {len(linked)} tasks to sprint:")
        for task_id, sprint_id in linked:
            print(f"  {task_id} → {sprint_id}")

    return 0


def cmd_orphan_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got orphan list' command."""
    got_manager = getattr(manager, '_manager', None) or getattr(manager, '_got_manager', None)
    if got_manager is None:
        print("Error: Cannot access GoTManager from adapter")
        return 1

    detector = OrphanDetector(got_manager)
    orphans = detector.find_orphan_tasks()

    if args.json:
        orphan_data = []
        for task_id in orphans:
            task = got_manager.get_task(task_id)
            if task:
                orphan_data.append({
                    "id": task_id,
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                })
        print(json.dumps(orphan_data, indent=2))
    else:
        if not orphans:
            print("No orphan tasks found!")
            return 0

        print(f"Orphan Tasks ({len(orphans)}):\n")
        for task_id in orphans:
            task = got_manager.get_task(task_id)
            if task:
                print(f"  {task_id}")
                print(f"    Title: {task.title[:50]}")
                print(f"    Status: {task.status}, Priority: {task.priority}")
                print()

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_orphan_parser(subparsers) -> None:
    """
    Set up argparse subparsers for orphan commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create orphan subparser
    orphan_parser = subparsers.add_parser(
        "orphan",
        help="Detect and manage orphan entities"
    )
    orphan_subparsers = orphan_parser.add_subparsers(
        dest="orphan_command",
        help="Orphan subcommands"
    )

    # orphan report
    report_parser = orphan_subparsers.add_parser(
        "report",
        help="Generate orphan report"
    )
    report_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    # orphan list
    list_parser = orphan_subparsers.add_parser(
        "list",
        help="List all orphan tasks"
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    # orphan check TASK_ID
    check_parser = orphan_subparsers.add_parser(
        "check",
        help="Check if a task is orphan and get suggestions"
    )
    check_parser.add_argument("task_id", help="Task ID to check")
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    # orphan suggest-sprint TASK_ID
    suggest_sprint_parser = orphan_subparsers.add_parser(
        "suggest-sprint",
        help="Suggest sprints for a task"
    )
    suggest_sprint_parser.add_argument("task_id", help="Task ID")
    suggest_sprint_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    # orphan suggest-links TASK_ID
    suggest_links_parser = orphan_subparsers.add_parser(
        "suggest-links",
        help="Suggest connections for a task"
    )
    suggest_links_parser.add_argument("task_id", help="Task ID")
    suggest_links_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    # orphan auto-link
    auto_link_parser = orphan_subparsers.add_parser(
        "auto-link",
        help="Auto-link orphan tasks to sprint"
    )
    auto_link_parser.add_argument(
        "--task-ids",
        nargs="+",
        help="Specific task IDs to link (default: all orphans)"
    )
    auto_link_parser.add_argument(
        "--sprint",
        help="Target sprint ID (default: current sprint)"
    )
    auto_link_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be linked without making changes"
    )
    auto_link_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )


def handle_orphan_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route orphan subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'orphan_command') or args.orphan_command is None:
        print("Error: No orphan subcommand specified. Use 'got orphan --help' for usage.")
        return 1

    command_handlers = {
        "report": cmd_orphan_report,
        "list": cmd_orphan_list,
        "check": cmd_orphan_check,
        "suggest-sprint": cmd_orphan_suggest_sprint,
        "suggest-links": cmd_orphan_suggest_links,
        "auto-link": cmd_orphan_auto_link,
    }

    handler = command_handlers.get(args.orphan_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown orphan subcommand: {args.orphan_command}")
    return 1
