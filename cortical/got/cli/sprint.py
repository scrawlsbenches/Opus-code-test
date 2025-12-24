"""
Sprint and Epic CLI commands for GoT system.

Provides commands for:
- Creating and managing sprints
- Sprint goals management
- Sprint task linking
- Sprint suggestions
- Epic management

This module can be integrated into got_utils.py CLI or used standalone.
"""

from typing import TYPE_CHECKING

from .shared import format_sprint_status, PRIORITY_SCORES

if TYPE_CHECKING:
    from scripts.got_utils import GoTProjectManager


# =============================================================================
# SPRINT CLI COMMAND HANDLERS
# =============================================================================

def cmd_sprint_create(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint create' command."""
    sprint_id = manager.create_sprint(
        name=args.name,
        number=getattr(args, 'number', None),
        epic_id=getattr(args, 'epic', None),
    )

    manager.save()
    print(f"Created: {sprint_id}")
    return 0


def cmd_sprint_list(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint list' command."""
    sprints = manager.list_sprints(
        status=getattr(args, 'status', None),
    )

    if not sprints:
        print("No sprints found.")
        return 0

    for sprint in sprints:
        progress = manager.get_sprint_progress(sprint.id)
        status = sprint.properties.get("status", "?")
        claimed_by = sprint.properties.get("claimed_by", "")

        # Build status line
        status_line = f"{sprint.id}: {sprint.content} [{status}] - {progress['progress_percent']:.0f}% complete"
        if claimed_by:
            status_line += f" (claimed by {claimed_by})"

        print(status_line)

    return 0


def cmd_sprint_status(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint status' command."""
    sprint_id = getattr(args, 'sprint_id', None)

    if sprint_id:
        sprint = manager.get_sprint(sprint_id)
        if not sprint:
            print(f"Sprint not found: {sprint_id}")
            return 1
        sprints = [sprint]
    else:
        # Show all active sprints
        sprints = manager.list_sprints(status="in_progress")
        if not sprints:
            sprints = manager.list_sprints(status="available")

    for sprint in sprints:
        progress = manager.get_sprint_progress(sprint.id)
        print(format_sprint_status(sprint, progress))
        print()

    return 0


def cmd_sprint_start(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint start' command."""
    sprint = manager.update_sprint(args.sprint_id, status="in_progress")
    manager.save()
    print(f"Started: {sprint.id}")
    print(f"  Title: {sprint.content}")
    return 0


def cmd_sprint_complete(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint complete' command."""
    sprint = manager.update_sprint(args.sprint_id, status="completed")
    manager.save()
    print(f"Completed: {sprint.id}")
    print(f"  Title: {sprint.content}")
    return 0


def cmd_sprint_claim(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint claim' command."""
    try:
        sprint = manager.claim_sprint(args.sprint_id, args.agent)
        manager.save()
        print(f"Claimed: {sprint.id}")
        print(f"  Agent: {args.agent}")
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_sprint_release(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint release' command."""
    try:
        sprint = manager.release_sprint(args.sprint_id, args.agent)
        manager.save()
        print(f"Released: {sprint.id}")
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_sprint_goal_add(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint goal add' command."""
    if manager.add_sprint_goal(args.sprint_id, args.description):
        manager.save()
        print(f"Added goal to {args.sprint_id}: {args.description}")
        return 0
    else:
        print(f"Sprint not found: {args.sprint_id}")
        return 1


def cmd_sprint_goal_list(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint goal list' command."""
    goals = manager.list_sprint_goals(args.sprint_id)
    if not goals:
        print(f"No goals for sprint {args.sprint_id}")
        return 0
    print(f"Goals for {args.sprint_id}:")
    for i, goal in enumerate(goals):
        status = "✓" if goal.get("completed") else " "
        print(f"  [{i}] [{status}] {goal.get('description', '')}")
    return 0


def cmd_sprint_goal_complete(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint goal complete' command."""
    if manager.complete_sprint_goal(args.sprint_id, args.index):
        manager.save()
        print(f"Completed goal {args.index} in {args.sprint_id}")
        return 0
    else:
        print("Failed - check sprint ID and goal index")
        return 1


def cmd_sprint_link(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint link' command."""
    if manager.link_task_to_sprint(args.sprint_id, args.task_id):
        manager.save()
        print(f"Linked task {args.task_id} to sprint {args.sprint_id}")
        return 0
    else:
        print("Failed to link - check that both IDs exist")
        return 1


def cmd_sprint_unlink(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint unlink' command."""
    if manager.unlink_task_from_sprint(args.sprint_id, args.task_id):
        manager.save()
        print(f"Unlinked task {args.task_id} from sprint {args.sprint_id}")
        return 0
    else:
        print(f"No link found between {args.sprint_id} and {args.task_id}")
        return 1


def cmd_sprint_tasks(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint tasks' command."""
    tasks = manager.get_sprint_tasks(args.sprint_id)
    if not tasks:
        print(f"No tasks in sprint {args.sprint_id}")
        return 0
    print(f"Tasks in {args.sprint_id}:")
    for task in tasks:
        status = task.properties.get("status", "unknown")
        priority = task.properties.get("priority", "medium")
        print(f"  {task.id}: {task.content} [status={status}, priority={priority}]")
    return 0


def cmd_sprint_suggest(args, manager: "GoTProjectManager") -> int:
    """Handle 'got sprint suggest' command."""
    try:
        # Get pending tasks
        if hasattr(manager, 'list_tasks'):
            pending_tasks = manager.list_tasks(status="pending")
        else:
            pending_tasks = [
                t for t in manager.tasks.values()
                if t.properties.get("status") == "pending"
            ]

        if not pending_tasks:
            print("No pending tasks to suggest.")
            return 0

        # Score and sort tasks
        scored_tasks = []
        for task in pending_tasks:
            priority = task.properties.get("priority", "medium")
            score = PRIORITY_SCORES.get(priority, 50)

            # Check if blocked
            if hasattr(manager, 'what_blocks'):
                blockers = manager.what_blocks(task.id)
                if blockers:
                    score -= 30  # Penalty for blocked tasks

            scored_tasks.append((score, task))

        # Sort by score descending
        scored_tasks.sort(key=lambda x: -x[0])

        # Limit results
        limit = getattr(args, 'limit', 10)
        suggestions = scored_tasks[:limit]

        # Display suggestions
        print(f"\n{'='*60}")
        print(f"SPRINT SUGGESTIONS ({len(suggestions)} tasks)")
        print(f"{'='*60}\n")

        for i, (score, task) in enumerate(suggestions, 1):
            priority = task.properties.get("priority", "medium")
            category = task.properties.get("category", "feature")
            title = task.content[:50] + "..." if len(task.content) > 50 else task.content
            print(f"{i:2}. [{priority.upper():8}] {task.id}")
            print(f"    {title}")
            print(f"    Category: {category}, Score: {score}")
            print()

        return 0
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        import traceback
        traceback.print_exc()
        return 1


# =============================================================================
# EPIC CLI COMMAND HANDLERS
# =============================================================================

def cmd_epic_create(args, manager: "GoTProjectManager") -> int:
    """Handle 'got epic create' command."""
    epic_id = manager.create_epic(
        name=args.name,
        epic_id=getattr(args, 'epic_id', None),
    )

    manager.save()
    print(f"Created: {epic_id}")
    return 0


def cmd_epic_list(args, manager: "GoTProjectManager") -> int:
    """Handle 'got epic list' command."""
    epics = manager.list_epics(
        status=getattr(args, 'status', None),
    )

    if not epics:
        print("No epics found.")
        return 0

    for epic in epics:
        status = epic.properties.get("status", "?")
        phase = epic.properties.get("phase", "?")
        print(f"{epic.id}: {epic.content} [{status}] - Phase: {phase}")

    return 0


def cmd_epic_show(args, manager: "GoTProjectManager") -> int:
    """Handle 'got epic show' command."""
    epic = manager.get_epic(args.epic_id)

    if not epic:
        print(f"Epic not found: {args.epic_id}")
        return 1

    print(f"Epic: {epic.id}")
    print(f"  Name: {epic.content}")
    print(f"  Status: {epic.properties.get('status', '?')}")
    print(f"  Phase: {epic.properties.get('phase', '?')}")

    # Show associated sprints
    sprints = manager.list_sprints(epic_id=epic.id)
    if sprints:
        print(f"  Sprints ({len(sprints)}):")
        for sprint in sprints:
            print(f"    - {sprint.id}: {sprint.content}")

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_sprint_parser(subparsers) -> None:
    """
    Set up argparse subparsers for sprint commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create sprint subparser
    sprint_parser = subparsers.add_parser("sprint", help="Sprint operations")
    sprint_subparsers = sprint_parser.add_subparsers(
        dest="sprint_command",
        help="Sprint subcommands"
    )

    # sprint create
    sprint_create = sprint_subparsers.add_parser("create", help="Create a sprint")
    sprint_create.add_argument("name", help="Sprint name")
    sprint_create.add_argument("--number", "-n", type=int, help="Sprint number")
    sprint_create.add_argument("--epic", "-e", help="Epic ID")

    # sprint list
    sprint_list = sprint_subparsers.add_parser("list", help="List sprints")
    sprint_list.add_argument("--status", help="Filter by status")

    # sprint status
    sprint_status = sprint_subparsers.add_parser("status", help="Show sprint status")
    sprint_status.add_argument("sprint_id", nargs="?", help="Sprint ID (optional)")

    # sprint start
    sprint_start = sprint_subparsers.add_parser("start", help="Start a sprint")
    sprint_start.add_argument("sprint_id", help="Sprint ID to start")

    # sprint complete
    sprint_complete = sprint_subparsers.add_parser("complete", help="Complete a sprint")
    sprint_complete.add_argument("sprint_id", help="Sprint ID to complete")

    # sprint claim
    sprint_claim = sprint_subparsers.add_parser("claim", help="Claim sprint for an agent")
    sprint_claim.add_argument("sprint_id", help="Sprint ID to claim")
    sprint_claim.add_argument("--agent", required=True, help="Agent name")

    # sprint release
    sprint_release = sprint_subparsers.add_parser("release", help="Release sprint claim")
    sprint_release.add_argument("sprint_id", help="Sprint ID to release")
    sprint_release.add_argument("--agent", required=True, help="Agent name")

    # sprint goal
    goal_parser = sprint_subparsers.add_parser("goal", help="Manage sprint goals")
    goal_subparsers = goal_parser.add_subparsers(dest="goal_action")

    # goal add
    goal_add = goal_subparsers.add_parser("add", help="Add a goal")
    goal_add.add_argument("sprint_id", help="Sprint ID")
    goal_add.add_argument("description", help="Goal description")

    # goal list
    goal_list = goal_subparsers.add_parser("list", help="List goals")
    goal_list.add_argument("sprint_id", help="Sprint ID")

    # goal complete
    goal_complete = goal_subparsers.add_parser("complete", help="Mark goal complete")
    goal_complete.add_argument("sprint_id", help="Sprint ID")
    goal_complete.add_argument("index", type=int, help="Goal index (0-based)")

    # sprint link
    sprint_link = sprint_subparsers.add_parser("link", help="Link a task to sprint")
    sprint_link.add_argument("sprint_id", help="Sprint ID")
    sprint_link.add_argument("task_id", help="Task ID to link")

    # sprint unlink
    sprint_unlink = sprint_subparsers.add_parser("unlink", help="Unlink task from sprint")
    sprint_unlink.add_argument("sprint_id", help="Sprint ID")
    sprint_unlink.add_argument("task_id", help="Task ID to unlink")

    # sprint tasks
    sprint_tasks = sprint_subparsers.add_parser("tasks", help="List tasks in sprint")
    sprint_tasks.add_argument("sprint_id", help="Sprint ID")

    # sprint suggest
    sprint_suggest = sprint_subparsers.add_parser("suggest", help="Suggest tasks for next sprint")
    sprint_suggest.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of suggestions"
    )
    sprint_suggest.add_argument(
        "--strategy",
        choices=["balanced", "quick-wins", "impact"],
        default="balanced",
        help="Selection strategy"
    )


def setup_epic_parser(subparsers) -> None:
    """
    Set up argparse subparsers for epic commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create epic subparser
    epic_parser = subparsers.add_parser("epic", help="Epic operations")
    epic_subparsers = epic_parser.add_subparsers(
        dest="epic_command",
        help="Epic subcommands"
    )

    # epic create
    epic_create = epic_subparsers.add_parser("create", help="Create an epic")
    epic_create.add_argument("name", help="Epic name")
    epic_create.add_argument("--id", dest="epic_id", help="Custom epic ID")

    # epic list
    epic_list = epic_subparsers.add_parser("list", help="List epics")
    epic_list.add_argument("--status", help="Filter by status")

    # epic show
    epic_show = epic_subparsers.add_parser("show", help="Show epic details")
    epic_show.add_argument("epic_id", help="Epic ID to display")


def handle_sprint_command(args, manager: "GoTProjectManager") -> int:
    """
    Route sprint subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'sprint_command') or args.sprint_command is None:
        print("Error: No sprint subcommand specified. Use 'got sprint --help' for usage.")
        return 1

    # Handle goal subcommands specially
    if args.sprint_command == "goal":
        goal_handlers = {
            "add": cmd_sprint_goal_add,
            "list": cmd_sprint_goal_list,
            "complete": cmd_sprint_goal_complete,
        }
        goal_action = getattr(args, 'goal_action', None)
        if goal_action and goal_action in goal_handlers:
            return goal_handlers[goal_action](args, manager)
        print("Error: No goal subcommand specified. Use 'got sprint goal --help' for usage.")
        return 1

    command_handlers = {
        "create": cmd_sprint_create,
        "list": cmd_sprint_list,
        "status": cmd_sprint_status,
        "start": cmd_sprint_start,
        "complete": cmd_sprint_complete,
        "claim": cmd_sprint_claim,
        "release": cmd_sprint_release,
        "link": cmd_sprint_link,
        "unlink": cmd_sprint_unlink,
        "tasks": cmd_sprint_tasks,
        "suggest": cmd_sprint_suggest,
    }

    handler = command_handlers.get(args.sprint_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown sprint subcommand: {args.sprint_command}")
    return 1


def handle_epic_command(args, manager: "GoTProjectManager") -> int:
    """
    Route epic subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'epic_command') or args.epic_command is None:
        print("Error: No epic subcommand specified. Use 'got epic --help' for usage.")
        return 1

    command_handlers = {
        "create": cmd_epic_create,
        "list": cmd_epic_list,
        "show": cmd_epic_show,
    }

    handler = command_handlers.get(args.epic_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown epic subcommand: {args.epic_command}")
    return 1
