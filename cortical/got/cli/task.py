"""
Task CLI commands for GoT system.

Provides commands for:
- Creating tasks
- Listing tasks
- Showing task details
- Starting/completing/blocking tasks
- Managing task dependencies

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
from typing import TYPE_CHECKING

from .shared import (
    VALID_STATUSES,
    VALID_PRIORITIES,
    VALID_CATEGORIES,
    PRIORITY_MEDIUM,
    STATUS_IN_PROGRESS,
    format_task_table,
    format_task_details,
)

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_task_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task create' command."""
    task_id = manager.create_task(
        title=args.title,
        priority=getattr(args, 'priority', PRIORITY_MEDIUM),
        category=getattr(args, 'category', 'feature'),
        description=getattr(args, 'description', ''),
        sprint_id=getattr(args, 'sprint', None),
        depends_on=getattr(args, 'depends', None),
        blocks=getattr(args, 'blocks', None),
    )

    manager.save()
    print(f"Created: {task_id}")
    return 0


def cmd_task_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task list' command."""
    tasks = manager.list_tasks(
        status=getattr(args, 'status', None),
        priority=getattr(args, 'priority', None),
        category=getattr(args, 'category', None),
        sprint_id=getattr(args, 'sprint', None),
        blocked_only=getattr(args, 'blocked', False),
    )

    # Apply limit if specified
    limit = getattr(args, 'limit', None)
    if limit is not None and limit > 0:
        tasks = tasks[:limit]

    if getattr(args, 'json', False):
        data = [{"id": t.id, "title": t.content, **t.properties} for t in tasks]
        print(json.dumps(data, indent=2))
    else:
        print(format_task_table(tasks))

    return 0


def cmd_task_next(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task next' command."""
    result = manager.get_next_task()

    if result is None:
        print("No pending tasks available.")
        return 0

    # Format output
    print(f"Next task: {result['id']}")
    print(f"  Title:    {result['title']}")
    print(f"  Priority: {result['priority']}")
    print(f"  Category: {result['category']}")

    # If --start flag, also start the task
    if getattr(args, 'start', False):
        task_id = result['id']
        if task_id.startswith("task:"):
            task_id = task_id[5:]
        success = manager.start_task(task_id)
        if success:
            print(f"\nStarted: {result['id']}")

    return 0


def cmd_task_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task show' command."""
    task_id = args.task_id

    # Try to get task (with ID normalization)
    task = manager.get_task(task_id)

    # If not found, try with/without task: prefix
    if task is None:
        if task_id.startswith("task:"):
            task = manager.get_task(task_id[5:])
        else:
            task = manager.get_task(f"task:{task_id}")

    if task is None:
        print(f"Task not found: {task_id}")
        return 1

    # Display task details
    print(format_task_details(task))

    # Show dependencies
    deps = manager.get_task_dependencies(task.id)
    if deps:
        print(f"\nDepends On ({len(deps)}):")
        for dep in deps:
            print(f"  - {dep.id}: {dep.content}")

    # Show what depends on this task
    dependents = manager.what_depends_on(task.id)
    if dependents:
        print(f"\nBlocks ({len(dependents)}):")
        for dep in dependents:
            print(f"  - {dep.id}: {dep.content}")

    return 0


def cmd_task_start(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task start' command."""
    if manager.start_task(args.task_id):
        manager.save()
        print(f"Started: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task complete' command."""
    if manager.complete_task(args.task_id, getattr(args, 'retrospective', None)):
        manager.save()
        print(f"Completed: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_block(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task block' command."""
    if manager.block_task(args.task_id, args.reason, getattr(args, 'blocker', None)):
        manager.save()
        print(f"Blocked: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_depends(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task depends' command."""
    try:
        # Use add_dependency method
        if manager.add_dependency(args.task_id, args.depends_on_id):
            manager.save()
            print(f"Created dependency: {args.task_id} depends on {args.depends_on_id}")
            return 0
        else:
            print("Failed to create dependency - check that both task IDs exist")
            return 1
    except Exception as e:
        print(f"Error creating dependency: {e}")
        return 1


def cmd_task_update(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task update' command.

    Updates task properties. Only specified fields are updated.
    """
    task_id = args.task_id

    # Get existing task
    task = manager.get_task(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return 1

    # Build updates dict from provided arguments
    updates = {}

    if getattr(args, 'title', None):
        updates['title'] = args.title
    if getattr(args, 'priority', None):
        updates['priority'] = args.priority
    if getattr(args, 'category', None):
        updates['category'] = args.category
    if getattr(args, 'description', None):
        updates['description'] = args.description
    if getattr(args, 'retrospective', None):
        updates['retrospective'] = args.retrospective

    if not updates:
        print("No updates specified. Use --title, --priority, --category, --description, or --retrospective")
        return 1

    # Apply updates
    if manager.update_task(task_id, **updates):
        manager.save()
        print(f"Updated: {task_id}")
        for key, value in updates.items():
            # Truncate long values for display
            display_value = value if len(str(value)) < 60 else str(value)[:57] + "..."
            print(f"  {key}: {display_value}")
        return 0
    else:
        print(f"Failed to update: {task_id}")
        return 1


def cmd_task_delete(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task delete' command.

    TRANSACTIONAL: Verifies pre-conditions before deletion.
    - Task must exist
    - Without --force: fails if task has dependents, blocks others, or is in progress
    - With --force: removes edges and deletes the task
    """
    task_id = args.task_id
    force = getattr(args, 'force', False)

    # Get task info before deletion for display
    task = manager.get_task(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return 1

    # Show what we're about to do
    task_title = task.content
    task_status = task.properties.get("status", "unknown")

    if not force:
        # Show warnings about what might block deletion
        dependents = manager.what_depends_on(
            task_id if task_id.startswith("task:") else f"task:{task_id}"
        )
        if dependents:
            print(f"⚠️  Cannot delete: {len(dependents)} task(s) depend on this task:")
            for d in dependents[:5]:
                print(f"    - {d.id}: {d.content}")
            if len(dependents) > 5:
                print(f"    ... and {len(dependents) - 5} more")
            print("\nUse --force to delete anyway (will orphan dependent tasks)")
            return 1

        if task_status == STATUS_IN_PROGRESS:
            print("⚠️  Cannot delete: task is in progress")
            print("Use --force to delete anyway")
            return 1

    # Attempt deletion
    if manager.delete_task(task_id, force=force):
        manager.save()
        print(f"🗑️  Deleted: {task_id}")
        print(f"   Title: {task_title}")
        if force:
            print("   (forced deletion)")
        return 0
    else:
        print(f"Failed to delete: {task_id}")
        return 1


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_task_parser(subparsers) -> None:
    """
    Set up argparse subparsers for task commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create task subparser
    task_parser = subparsers.add_parser("task", help="Task operations")
    task_subparsers = task_parser.add_subparsers(
        dest="task_command",
        help="Task subcommands"
    )

    # task create
    create_parser = task_subparsers.add_parser("create", help="Create a task")
    create_parser.add_argument("title", help="Task title")
    create_parser.add_argument(
        "--priority", "-p",
        choices=VALID_PRIORITIES,
        default=PRIORITY_MEDIUM
    )
    create_parser.add_argument(
        "--category", "-c",
        choices=VALID_CATEGORIES,
        default="feature"
    )
    create_parser.add_argument("--description", "--notes", "-d", default="")
    create_parser.add_argument("--sprint", "-s", help="Sprint ID")
    create_parser.add_argument(
        "--depends-on", "--depends",
        nargs="+",
        dest="depends",
        help="Task IDs this task depends on"
    )
    create_parser.add_argument(
        "--blocks",
        nargs="+",
        help="Task IDs this task blocks"
    )

    # task list
    list_parser = task_subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=VALID_STATUSES)
    list_parser.add_argument("--priority", choices=VALID_PRIORITIES)
    list_parser.add_argument("--category", choices=VALID_CATEGORIES)
    list_parser.add_argument("--sprint", help="Filter by sprint")
    list_parser.add_argument("--blocked", action="store_true", help="Show only blocked")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of results"
    )

    # task show
    show_parser = task_subparsers.add_parser("show", help="Show task details")
    show_parser.add_argument("task_id", help="Task ID to display")

    # task next
    next_parser = task_subparsers.add_parser("next", help="Get the next task to work on")
    next_parser.add_argument(
        "--start", "-s",
        action="store_true",
        help="Also start the task after selecting it"
    )

    # task start
    start_parser = task_subparsers.add_parser("start", help="Start a task")
    start_parser.add_argument("task_id", help="Task ID")

    # task complete
    complete_parser = task_subparsers.add_parser("complete", help="Complete a task")
    complete_parser.add_argument("task_id", help="Task ID")
    complete_parser.add_argument(
        "--retrospective", "--notes", "-r", "-n",
        dest="retrospective",
        help="Retrospective notes"
    )

    # task block
    block_parser = task_subparsers.add_parser("block", help="Block a task")
    block_parser.add_argument("task_id", help="Task ID")
    block_parser.add_argument("--reason", "-r", required=True, help="Block reason")
    block_parser.add_argument("--blocker", "-b", help="Blocking task ID")

    # task update
    update_parser = task_subparsers.add_parser("update", help="Update a task's properties")
    update_parser.add_argument("task_id", help="Task ID to update")
    update_parser.add_argument("--title", "-t", help="New title")
    update_parser.add_argument(
        "--priority", "-p",
        choices=VALID_PRIORITIES,
        help="New priority"
    )
    update_parser.add_argument(
        "--category", "-c",
        choices=VALID_CATEGORIES,
        help="New category"
    )
    update_parser.add_argument("--description", "-d", help="New description")
    update_parser.add_argument(
        "--retrospective", "--notes", "-r", "-n",
        dest="retrospective",
        help="Retrospective notes"
    )

    # task delete
    delete_parser = task_subparsers.add_parser("delete", help="Delete a task (transactional)")
    delete_parser.add_argument("task_id", help="Task ID to delete")
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force delete even if task has dependencies or is in progress"
    )

    # task depends
    depends_parser = task_subparsers.add_parser("depends", help="Create task dependency")
    depends_parser.add_argument("task_id", help="Task that depends on another")
    depends_parser.add_argument(
        "--on",
        dest="depends_on_id",
        required=True,
        help="Task ID to depend on"
    )


def handle_task_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route task subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'task_command') or args.task_command is None:
        print("Error: No task subcommand specified. Use 'got task --help' for usage.")
        return 1

    command_handlers = {
        "create": cmd_task_create,
        "list": cmd_task_list,
        "show": cmd_task_show,
        "next": cmd_task_next,
        "start": cmd_task_start,
        "complete": cmd_task_complete,
        "block": cmd_task_block,
        "update": cmd_task_update,
        "delete": cmd_task_delete,
        "depends": cmd_task_depends,
    }

    handler = command_handlers.get(args.task_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown task subcommand: {args.task_command}")
    return 1
