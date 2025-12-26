"""
Decision CLI commands for GoT system.

Provides commands for logging decisions with rationale:
- Logging decisions
- Listing decisions
- Querying why tasks were created

This module can be integrated into got_utils.py CLI or used standalone.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_decision_log(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got decision log' command."""
    context = {}
    if args.file:
        context["file"] = args.file

    decision_id = manager.log_decision(
        decision=args.decision,
        rationale=args.rationale,
        affects=args.affects,
        alternatives=args.alternatives,
        context=context if context else None,
    )

    print(f"Decision logged: {decision_id}")
    print(f"  Decision: {args.decision}")
    print(f"  Rationale: {args.rationale}")
    if args.affects:
        print(f"  Affects: {', '.join(args.affects)}")
    if args.alternatives:
        print(f"  Alternatives considered: {', '.join(args.alternatives)}")
    return 0


def cmd_decision_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got decision list' command."""
    # Use list_decisions for transactional backend compatibility
    if hasattr(manager, 'list_decisions'):
        decisions = manager.list_decisions()
    else:
        decisions = manager.get_decisions()

    if not decisions:
        print("No decisions logged yet.")
        return 0

    # Apply limit if specified
    limit = getattr(args, 'limit', None)
    if limit is not None and limit > 0:
        decisions = decisions[:limit]

    print(f"Decisions ({len(decisions)}):\n")
    for d in decisions:
        print(f"  {d.id}")
        print(f"    Decision: {d.content}")
        print(f"    Rationale: {d.properties.get('rationale', 'N/A')}")
        if d.properties.get("alternatives"):
            print(f"    Alternatives: {', '.join(d.properties['alternatives'])}")
        print()

    return 0


def cmd_decision_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got decision show' command."""
    decision_id = args.decision_id

    # Get all decisions and find the one we want
    if hasattr(manager, 'list_decisions'):
        decisions = manager.list_decisions()
    else:
        decisions = manager.get_decisions()

    decision = None
    for d in decisions:
        if d.id == decision_id:
            decision = d
            break

    if not decision:
        print(f"Decision not found: {decision_id}")
        return 1

    # Display full decision details
    print("=" * 60)
    print(f"DECISION: {decision_id}")
    print("=" * 60)
    print(f"Decision:    {decision.content}")
    print(f"Rationale:   {decision.properties.get('rationale', 'N/A')}")

    if decision.properties.get("created_at"):
        print(f"Created:     {decision.properties['created_at']}")

    # Show alternatives
    alternatives = decision.properties.get("alternatives", [])
    if alternatives:
        print(f"\nAlternatives Considered:")
        for alt in alternatives:
            print(f"  - {alt}")

    # Show affected tasks
    affects = decision.properties.get("affects", [])
    if affects:
        print(f"\nAffects:")
        for task_id in affects:
            # Try to get task details
            task = manager.get_task(task_id)
            if task:
                print(f"  - {task_id}: {task.content}")
            else:
                print(f"  - {task_id}")

    # Show context
    context = decision.properties.get("context", {})
    if context:
        print(f"\nContext:")
        for key, value in context.items():
            print(f"  {key}: {value}")

    print("=" * 60)
    return 0


def cmd_decision_why(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got decision why' command."""
    reasons = manager.why(args.task_id)

    if not reasons:
        print(f"No decisions found affecting {args.task_id}")
        return 0

    print(f"Why {args.task_id}?\n")
    for r in reasons:
        print(f"  {r['decision_id']}")
        print(f"    Decision: {r['decision']}")
        print(f"    Rationale: {r['rationale']}")
        if r["alternatives"]:
            print(f"    Alternatives: {', '.join(r['alternatives'])}")
        print()

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_decision_parser(subparsers) -> None:
    """
    Set up argparse subparsers for decision commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create decision subparser
    decision_parser = subparsers.add_parser(
        "decision",
        help="Log decisions with rationale"
    )
    decision_subparsers = decision_parser.add_subparsers(
        dest="decision_command",
        help="Decision subcommands"
    )

    # decision log
    decision_log = decision_subparsers.add_parser("log", help="Log a decision")
    decision_log.add_argument("decision", help="What was decided")
    decision_log.add_argument(
        "--rationale", "-r",
        required=True,
        help="Why this choice was made"
    )
    decision_log.add_argument(
        "--affects", "-a",
        nargs="+",
        help="Task IDs affected by this decision"
    )
    decision_log.add_argument(
        "--alternatives",
        nargs="+",
        help="Alternatives considered"
    )
    decision_log.add_argument(
        "--file", "-f",
        help="File this decision relates to"
    )

    # decision list
    decision_list = decision_subparsers.add_parser("list", help="List all decisions")
    decision_list.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of results"
    )

    # decision show
    decision_show = decision_subparsers.add_parser("show", help="Show decision details")
    decision_show.add_argument("decision_id", help="Decision ID to display")

    # decision why
    decision_why = decision_subparsers.add_parser("why", help="Ask why a task exists")
    decision_why.add_argument("task_id", help="Task ID to query")


def handle_decision_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route decision subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'decision_command') or args.decision_command is None:
        print("Error: No decision subcommand specified. Use 'got decision --help' for usage.")
        return 1

    command_handlers = {
        "log": cmd_decision_log,
        "list": cmd_decision_list,
        "show": cmd_decision_show,
        "why": cmd_decision_why,
    }

    handler = command_handlers.get(args.decision_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown decision subcommand: {args.decision_command}")
    return 1
