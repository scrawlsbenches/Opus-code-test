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
    from cortical.got.api import GoTManager

from ..types import EdgeTypes


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_decision_log(args, manager: "GoTManager") -> int:
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

    # Prompt for task linkage (optional)
    _prompt_task_linkage(decision_id, manager)

    return 0


def _prompt_task_linkage(decision_id: str, manager: "GoTManager") -> None:
    """Prompt user to optionally link decision to a task.

    Args:
        decision_id: The decision ID to link
        manager: GoT manager instance
    """
    # Get recent in-progress tasks
    try:
        tasks = manager.list_tasks(status="in_progress")
        if not tasks:
            # No in-progress tasks, skip prompt
            return

        # Limit to 5 most recent
        tasks = tasks[:5]

        print("\nLink to a task? Recent in-progress tasks:")
        for i, task in enumerate(tasks, 1):
            # Get task content/title
            content = getattr(task, 'content', getattr(task, 'title', 'Untitled'))
            task_id = getattr(task, 'id', str(task))
            # Truncate long titles
            if len(content) > 60:
                content = content[:57] + "..."
            print(f"  {i}. {task_id}: {content}")

        # Get user input
        try:
            response = input("(Enter number, task ID, or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle non-interactive mode or Ctrl+C
            print()
            return

        if not response:
            # User pressed Enter to skip
            return

        # Parse response (number or task ID)
        selected_task_id = None
        if response.isdigit():
            # User entered a number
            idx = int(response) - 1
            if 0 <= idx < len(tasks):
                selected_task_id = getattr(tasks[idx], 'id', None)
            else:
                print(f"Invalid selection: {response}")
                return
        else:
            # User entered a task ID directly
            selected_task_id = response

        if selected_task_id:
            # Create JUSTIFIES edge: decision -> task
            try:
                manager.add_edge(
                    source_id=decision_id,
                    target_id=selected_task_id,
                    edge_type=EdgeTypes.JUSTIFIES,
                )
                print(f"✓ Linked {decision_id} -> {selected_task_id} (JUSTIFIES)")
            except Exception as e:
                print(f"✗ Failed to create edge: {e}")

    except Exception as e:
        # Fail gracefully if anything goes wrong
        import sys
        print(f"Warning: Could not prompt for task linkage: {e}", file=sys.stderr)


def cmd_decision_list(args, manager: "GoTManager") -> int:
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


def cmd_decision_show(args, manager: "GoTManager") -> int:
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


def cmd_decision_why(args, manager: "GoTManager") -> int:
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


def cmd_decision_trace(args, manager: "GoTManager") -> int:
    """Handle 'got decision trace' command.

    Shows the full context of a decision:
    - The decision itself
    - What it affects (outbound: JUSTIFIES, MOTIVATES)
    - What it superseded (SUPERSEDES)
    - What supersedes it (inbound: SUPERSEDES)
    """
    decision_id = args.decision_id

    # Get the decision
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

    # Display decision header
    print("=" * 70)
    print(f"DECISION TRACE: {decision_id}")
    print("=" * 70)
    print(f"\nDecision:  {decision.content}")
    print(f"Rationale: {decision.properties.get('rationale', 'N/A')}")

    if decision.properties.get("created_at"):
        print(f"Created:   {decision.properties['created_at']}")

    # Get all edges to find relationships
    edges = manager.list_edges()

    # Find outbound edges (what this decision affects)
    justifies = []
    motivates = []
    supersedes = []

    # Find inbound edges (what affects this decision)
    superseded_by = []

    for edge in edges:
        if edge.source_id == decision_id:
            if edge.edge_type == EdgeTypes.JUSTIFIES:
                justifies.append(edge.target_id)
            elif edge.edge_type == EdgeTypes.MOTIVATES:
                motivates.append(edge.target_id)
            elif edge.edge_type == EdgeTypes.SUPERSEDES:
                supersedes.append(edge.target_id)
        elif edge.target_id == decision_id:
            if edge.edge_type == EdgeTypes.SUPERSEDES:
                superseded_by.append(edge.source_id)

    # Display what this decision justifies
    if justifies:
        print(f"\n┌─ JUSTIFIES ({len(justifies)} items)")
        for target_id in justifies:
            # Try to get entity details
            entity = _get_entity_summary(manager, target_id)
            print(f"│  → {target_id}: {entity}")
        print("└─")

    # Display what this decision motivates
    if motivates:
        print(f"\n┌─ MOTIVATES ({len(motivates)} items)")
        for target_id in motivates:
            entity = _get_entity_summary(manager, target_id)
            print(f"│  → {target_id}: {entity}")
        print("└─")

    # Display what this decision supersedes
    if supersedes:
        print(f"\n┌─ SUPERSEDES ({len(supersedes)} decisions)")
        for target_id in supersedes:
            entity = _get_entity_summary(manager, target_id)
            print(f"│  → {target_id}: {entity}")
        print("└─")

    # Display what supersedes this decision
    if superseded_by:
        print(f"\n┌─ SUPERSEDED BY ({len(superseded_by)} decisions)")
        for source_id in superseded_by:
            entity = _get_entity_summary(manager, source_id)
            print(f"│  ← {source_id}: {entity}")
        print("└─")

    # Summary
    if not any([justifies, motivates, supersedes, superseded_by]):
        print("\n(No edges connected to this decision)")

    print("\n" + "=" * 70)
    return 0


def _get_entity_summary(manager: "GoTManager", entity_id: str) -> str:
    """Get a brief summary of an entity by ID."""
    # Try task first
    try:
        task = manager.get_task(entity_id)
        if task:
            content = getattr(task, 'content', getattr(task, 'title', ''))
            return content[:50] + "..." if len(content) > 50 else content
    except Exception:
        pass

    # Try decision
    try:
        if hasattr(manager, 'get_decision'):
            decision = manager.get_decision(entity_id)
            if decision:
                content = getattr(decision, 'content', '')
                return content[:50] + "..." if len(content) > 50 else content
    except Exception:
        pass

    # Try sprint
    try:
        if hasattr(manager, 'get_sprint'):
            sprint = manager.get_sprint(entity_id)
            if sprint:
                return getattr(sprint, 'name', str(sprint))
    except Exception:
        pass

    return "(details unavailable)"


def cmd_decision_delete(args, manager: "GoTManager") -> int:
    """Handle 'got decision delete' command."""
    decision_id = args.decision_id
    force = getattr(args, 'force', False)

    # Verify decision exists first
    decision = manager.get_decision(decision_id)
    if not decision:
        print(f"Decision not found: {decision_id}")
        return 1

    try:
        # Get the content/title - supports both ThoughtNode (content) and Decision (title)
        title = getattr(decision, 'content', getattr(decision, 'title', 'Unknown'))
        manager.delete_decision(decision_id, force=force)
        print(f"Deleted decision: {decision_id}")
        print(f"  Title: {title[:60]}..." if len(title) > 60 else f"  Title: {title}")
        return 0
    except Exception as e:
        print(f"Failed to delete decision: {e}")
        if not force:
            print("Hint: Use --force to delete despite connected edges")
        return 1


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
        required=False,
        default="Quick log - rationale pending",
        help="Why this choice was made (optional, defaults to placeholder)"
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

    # decision trace
    decision_trace = decision_subparsers.add_parser(
        "trace",
        help="Trace decision context (what it affects and what affects it)"
    )
    decision_trace.add_argument("decision_id", help="Decision ID to trace")

    # decision delete
    decision_delete = decision_subparsers.add_parser("delete", help="Delete a decision")
    decision_delete.add_argument("decision_id", help="Decision ID to delete")
    decision_delete.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force delete even if decision has connected edges"
    )


def handle_decision_command(args, manager: "GoTManager") -> int:
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
        "trace": cmd_decision_trace,
        "delete": cmd_decision_delete,
    }

    handler = command_handlers.get(args.decision_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown decision subcommand: {args.decision_command}")
    return 1
