"""
Handoff CLI commands for GoT system.

Provides commands for agent-to-agent work transfers:
- Initiating handoffs
- Accepting handoffs
- Completing handoffs
- Listing handoff status

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_handoff_initiate(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got handoff initiate' command."""
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

    # Use manager's handoff method (works with TX backend)
    handoff_id = manager.initiate_handoff(
        source_agent=args.source,
        target_agent=args.target,
        task_id=args.task_id,
        context={
            "task_title": task.content,
            "task_status": task.properties.get("status"),
            "task_priority": task.properties.get("priority"),
        },
        instructions=args.instructions,
    )

    print(f"Handoff initiated: {handoff_id}")
    print(f"  Task: {task.content}")
    print(f"  From: {args.source} → To: {args.target}")
    if args.instructions:
        print(f"  Instructions: {args.instructions}")
    return 0


def cmd_handoff_accept(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got handoff accept' command."""
    # Use manager's handoff method (works with TX backend)
    success = manager.accept_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        acknowledgment=args.message,
    )

    if not success:
        print(f"Failed to accept handoff: {args.handoff_id}")
        return 1

    print(f"Handoff accepted: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    return 0


def cmd_handoff_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got handoff complete' command."""
    try:
        result = json.loads(args.result)
    except json.JSONDecodeError:
        result = {"message": args.result}

    # Use manager's handoff method (works with TX backend)
    success = manager.complete_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        result=result,
        artifacts=args.artifacts or [],
    )

    if not success:
        print(f"Failed to complete handoff: {args.handoff_id}")
        return 1

    print(f"Handoff completed: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    print(f"  Result: {json.dumps(result, indent=2)}")
    return 0


def cmd_handoff_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got handoff show' command."""
    handoff_id = args.handoff_id

    # Get all handoffs and find the one we want
    handoffs = manager.list_handoffs(status=None)
    handoff = None
    for h in handoffs:
        if h.get("id") == handoff_id:
            handoff = h
            break

    if not handoff:
        print(f"Handoff not found: {handoff_id}")
        return 1

    # Display full handoff details
    status = handoff.get("status", "?")
    status_icon = {
        "initiated": "→",
        "accepted": "✓",
        "completed": "✓✓",
        "rejected": "✗",
    }.get(status, "?")

    print("=" * 60)
    print(f"HANDOFF: {handoff_id}")
    print("=" * 60)
    print(f"Status:      {status_icon} {status}")
    print(f"From:        {handoff.get('source_agent', '?')}")
    print(f"To:          {handoff.get('target_agent', '?')}")
    print(f"Task:        {handoff.get('task_id', '?')}")

    if handoff.get("created_at"):
        print(f"Created:     {handoff['created_at']}")
    if handoff.get("accepted_at"):
        print(f"Accepted:    {handoff['accepted_at']}")
    if handoff.get("completed_at"):
        print(f"Completed:   {handoff['completed_at']}")

    # Show context
    context = handoff.get("context", {})
    if context:
        print(f"\nContext:")
        for key, value in context.items():
            print(f"  {key}: {value}")

    # Show full instructions (not truncated)
    if handoff.get("instructions"):
        print(f"\nInstructions:")
        print("-" * 40)
        print(handoff["instructions"])
        print("-" * 40)

    # Show result if completed
    if handoff.get("result"):
        print(f"\nResult:")
        print(json.dumps(handoff["result"], indent=2))

    # Show artifacts if any
    if handoff.get("artifacts"):
        print(f"\nArtifacts:")
        for artifact in handoff["artifacts"]:
            print(f"  - {artifact}")

    print("=" * 60)
    return 0


def cmd_handoff_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got handoff list' command."""
    # Normalize status alias: in_progress -> accepted (matches task terminology)
    status = args.status
    if status == "in_progress":
        status = "accepted"

    # Use manager's handoff method (works with TX backend)
    handoffs = manager.list_handoffs(status=status)

    if not handoffs:
        print("No handoffs found.")
        return 0

    # Apply limit if specified
    limit = getattr(args, 'limit', None)
    if limit is not None and limit > 0:
        handoffs = handoffs[:limit]

    print(f"Handoffs ({len(handoffs)}):\n")
    for h in handoffs:
        status = h.get("status", "?")
        status_icon = {
            "initiated": "→",
            "accepted": "✓",
            "completed": "✓✓",
            "rejected": "✗",
        }.get(status, "?")

        print(f"  {status_icon} {h['id']}")
        print(f"      {h.get('source_agent', '?')} → {h.get('target_agent', '?')}")
        print(f"      Task: {h.get('task_id', '?')}")
        print(f"      Status: {status}")
        if h.get("instructions"):
            print(f"      Instructions: {h['instructions'][:50]}...")
        print()

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_handoff_parser(subparsers) -> None:
    """
    Set up argparse subparsers for handoff commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create handoff subparser
    handoff_parser = subparsers.add_parser("handoff", help="Agent handoff operations")
    handoff_subparsers = handoff_parser.add_subparsers(
        dest="handoff_command",
        help="Handoff subcommands"
    )

    # handoff initiate
    handoff_init = handoff_subparsers.add_parser(
        "initiate",
        help="Initiate a handoff to another agent"
    )
    handoff_init.add_argument("task_id", help="Task to hand off")
    handoff_init.add_argument(
        "--target", "-t",
        required=True,
        help="Target agent (e.g., 'sub-agent-1')"
    )
    handoff_init.add_argument(
        "--source", "-s",
        default="main",
        help="Source agent (default: main)"
    )
    handoff_init.add_argument(
        "--instructions", "-i",
        default="",
        help="Instructions for target agent"
    )

    # handoff accept
    handoff_accept = handoff_subparsers.add_parser("accept", help="Accept a handoff")
    handoff_accept.add_argument("handoff_id", help="Handoff ID to accept")
    handoff_accept.add_argument("--agent", "-a", required=True, help="Agent accepting")
    handoff_accept.add_argument("--message", "-m", default="", help="Acknowledgment message")

    # handoff complete
    handoff_complete = handoff_subparsers.add_parser("complete", help="Complete a handoff")
    handoff_complete.add_argument("handoff_id", help="Handoff ID to complete")
    handoff_complete.add_argument("--agent", "-a", required=True, help="Agent completing")
    handoff_complete.add_argument("--result", "-r", default="{}", help="Result as JSON")
    handoff_complete.add_argument(
        "--artifacts",
        nargs="*",
        help="Artifacts created (files, commits)"
    )

    # handoff show
    handoff_show = handoff_subparsers.add_parser("show", help="Show handoff details")
    handoff_show.add_argument("handoff_id", help="Handoff ID to display")

    # handoff list
    handoff_list = handoff_subparsers.add_parser("list", help="List handoffs")
    handoff_list.add_argument(
        "--status",
        # Note: in_progress is an alias for accepted (matches task terminology)
        choices=["initiated", "accepted", "in_progress", "completed", "rejected"]
    )
    handoff_list.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of results"
    )


def handle_handoff_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route handoff subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'handoff_command') or args.handoff_command is None:
        print("Error: No handoff subcommand specified. Use 'got handoff --help' for usage.")
        return 1

    command_handlers = {
        "initiate": cmd_handoff_initiate,
        "accept": cmd_handoff_accept,
        "complete": cmd_handoff_complete,
        "show": cmd_handoff_show,
        "list": cmd_handoff_list,
    }

    handler = command_handlers.get(args.handoff_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown handoff subcommand: {args.handoff_command}")
    return 1
