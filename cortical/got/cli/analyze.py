"""
Graph Analysis CLI commands using the fluent Query API.

This module provides powerful analysis commands that demonstrate the
Query builder, GraphWalker, PathFinder, and PatternMatcher APIs.

Commands:
    got analyze summary          - Quick status summary with counts
    got analyze blockers         - Find all blocking chains
    got analyze orphans          - Find disconnected tasks
    got analyze dependencies ID  - Analyze dependency chain for a task
    got analyze patterns         - Find common graph patterns

These commands showcase the fluent query API for maintainability.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# ANALYSIS COMMANDS USING FLUENT QUERY API
# =============================================================================


def cmd_analyze_summary(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Generate a summary using the Query builder.

    This demonstrates using Query.group_by().count().execute() for
    efficient aggregation without loading all entities.
    """
    from cortical.got import Query, Count, Collect

    # Get the underlying GoTManager from the adapter
    got_manager = manager._manager

    print("=" * 60)
    print("GoT ANALYSIS SUMMARY")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Task counts by status (using fluent Query API)
    # -------------------------------------------------------------------------
    print("TASKS BY STATUS:")
    print("-" * 40)

    # Use Query builder for grouped counting
    status_counts = (
        Query(got_manager)
        .tasks()
        .group_by("status")
        .count()
        .execute()
    )

    # Display counts with visual bars
    total = sum(status_counts.values()) if status_counts else 0
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 30)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {status:15} {count:3} ({pct:5.1f}%) {bar}")
    print(f"  {'TOTAL':15} {total:3}")
    print()

    # -------------------------------------------------------------------------
    # Tasks by priority (another Query example)
    # -------------------------------------------------------------------------
    print("TASKS BY PRIORITY:")
    print("-" * 40)

    priority_counts = (
        Query(got_manager)
        .tasks()
        .group_by("priority")
        .count()
        .execute()
    )

    # Priority order for display
    priority_order = ["critical", "high", "medium", "low"]
    for priority in priority_order:
        count = priority_counts.get(priority, 0)
        bar = "#" * min(count, 30)
        print(f"  {priority:15} {count:3} {bar}")
    print()

    # -------------------------------------------------------------------------
    # High-priority pending tasks (WHERE + ORDER BY)
    # -------------------------------------------------------------------------
    print("HIGH PRIORITY PENDING (action needed):")
    print("-" * 40)

    urgent = (
        Query(got_manager)
        .tasks()
        .where(status="pending", priority="high")
        .or_where(priority="critical")  # Also include critical
        .order_by("created_at", desc=True)
        .limit(5)
        .execute()
    )

    if urgent:
        for task in urgent:
            age = _task_age_str(task.created_at)
            print(f"  [{task.priority:8}] {task.title[:40]:<40} ({age})")
    else:
        print("  (none)")
    print()

    # -------------------------------------------------------------------------
    # Sprint overview
    # -------------------------------------------------------------------------
    print("SPRINTS:")
    print("-" * 40)

    sprints = Query(got_manager).sprints().execute()
    active = [s for s in sprints if s.status == "in_progress"]

    for sprint in active:
        # Count tasks in this sprint using connected_to
        sprint_tasks = (
            Query(got_manager)
            .tasks()
            .connected_to(sprint.id, via="CONTAINS")
            .execute()
        )
        completed = len([t for t in sprint_tasks if t.status == "completed"])
        total_sprint = len(sprint_tasks)
        pct = (completed / total_sprint * 100) if total_sprint > 0 else 0
        print(f"  {sprint.title}: {completed}/{total_sprint} ({pct:.0f}%)")

    if not active:
        print("  (no active sprints)")
    print()

    print("=" * 60)
    print(f"Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    return 0


def cmd_analyze_dependencies(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Analyze the dependency chain for a specific task.

    Uses GraphWalker to traverse dependencies and PathFinder to show paths.
    """
    from cortical.got import GraphWalker, PathFinder

    got_manager = manager._manager
    task_id = args.task_id

    # Validate task exists
    task = got_manager.get_task(task_id)
    if not task:
        print(f"Error: Task {task_id} not found")
        return 1

    print(f"DEPENDENCY ANALYSIS: {task.title}")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Find what this task depends on (upstream)
    # -------------------------------------------------------------------------
    print("DEPENDS ON (upstream, must complete first):")
    print("-" * 40)

    upstream_ids = []

    def collect_upstream(node, acc):
        if node.id != task_id:
            acc.append(node.id)
        return acc

    # Follow DEPENDS_ON edges in reverse (target->source)
    upstream = (
        GraphWalker(got_manager)
        .starting_from(task_id)
        .follow("DEPENDS_ON")
        .reverse()  # Go from dependent to dependency
        .bfs()
        .max_depth(5)
        .visit(collect_upstream, initial=[])
        .run()
    )

    if upstream:
        for dep_id in upstream:
            dep = got_manager.get_task(dep_id)
            if dep:
                status_icon = _status_icon(dep.status)
                print(f"  {status_icon} {dep.title[:50]}")
    else:
        print("  (no dependencies)")
    print()

    # -------------------------------------------------------------------------
    # Find what depends on this task (downstream)
    # -------------------------------------------------------------------------
    print("DEPENDENTS (downstream, waiting for this):")
    print("-" * 40)

    def collect_downstream(node, acc):
        if node.id != task_id:
            acc.append(node.id)
        return acc

    # Follow DEPENDS_ON edges normally (source->target means source depends on target)
    # Actually we want "what has this task as a dependency"
    # So we look for edges where target_id = task_id
    downstream = (
        GraphWalker(got_manager)
        .starting_from(task_id)
        .follow("DEPENDS_ON")
        .directed()  # Only source->target
        .bfs()
        .max_depth(5)
        .visit(collect_downstream, initial=[])
        .run()
    )

    if downstream:
        for dep_id in downstream:
            dep = got_manager.get_task(dep_id)
            if dep:
                status_icon = _status_icon(dep.status)
                print(f"  {status_icon} {dep.title[:50]}")
    else:
        print("  (nothing depends on this)")
    print()

    # -------------------------------------------------------------------------
    # Detect potential issues
    # -------------------------------------------------------------------------
    print("ISSUES:")
    print("-" * 40)

    issues = []

    # Check if blocked by incomplete dependencies
    for dep_id in upstream:
        dep = got_manager.get_task(dep_id)
        if dep and dep.status != "completed":
            issues.append(f"Blocked by incomplete: {dep.title[:40]}")

    # Check for circular dependencies
    path = PathFinder(got_manager).via_edges("DEPENDS_ON").shortest_path(task_id, task_id)
    if path and len(path) > 1:
        issues.append(f"CIRCULAR DEPENDENCY detected! Path length: {len(path)}")

    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("  (no issues detected)")
    print()

    return 0


def cmd_analyze_patterns(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Find common patterns in the graph using PatternMatcher.

    Demonstrates subgraph pattern matching.
    """
    from cortical.got import Pattern, PatternMatcher

    got_manager = manager._manager

    print("GRAPH PATTERN ANALYSIS")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Find blocking chains (A blocks B blocks C)
    # -------------------------------------------------------------------------
    print("BLOCKING CHAINS (A blocks B blocks C):")
    print("-" * 40)

    blocking_pattern = (
        Pattern()
        .node("a", type="task")
        .edge("BLOCKS", direction="outgoing")
        .node("b", type="task")
        .edge("BLOCKS", direction="outgoing")
        .node("c", type="task")
    )

    blocking_chains = PatternMatcher(got_manager).limit(5).find(blocking_pattern)

    if blocking_chains:
        for match in blocking_chains:
            a, b, c = match["a"], match["b"], match["c"]
            print(f"  {a.title[:20]} -> {b.title[:20]} -> {c.title[:20]}")
    else:
        print("  (no blocking chains found)")
    print()

    # -------------------------------------------------------------------------
    # Find long dependency chains
    # -------------------------------------------------------------------------
    print("DEPENDENCY CHAINS (length 3+):")
    print("-" * 40)

    dep_pattern = (
        Pattern()
        .node("a", type="task")
        .edge("DEPENDS_ON", direction="incoming")
        .node("b", type="task")
        .edge("DEPENDS_ON", direction="incoming")
        .node("c", type="task")
    )

    dep_chains = PatternMatcher(got_manager).limit(5).find(dep_pattern)

    if dep_chains:
        for match in dep_chains:
            a, b, c = match["a"], match["b"], match["c"]
            # Chain is: c depends on b depends on a
            print(f"  {c.title[:20]} <- {b.title[:20]} <- {a.title[:20]}")
    else:
        print("  (no long chains found)")
    print()

    # -------------------------------------------------------------------------
    # Find tasks connected to decisions (good documentation)
    # -------------------------------------------------------------------------
    print("DOCUMENTED DECISIONS:")
    print("-" * 40)

    doc_pattern = (
        Pattern()
        .node("task", type="task")
        .edge("JUSTIFIED_BY", direction="any")
        .node("decision", type="decision")
    )

    documented = PatternMatcher(got_manager).limit(10).find(doc_pattern)

    if documented:
        for match in documented:
            task = match["task"]
            decision = match["decision"]
            print(f"  {task.title[:30]}")
            print(f"    -> {decision.title[:40]}")
    else:
        print("  (no task-decision links found)")
    print()

    return 0


def cmd_analyze_orphans(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Find orphan tasks using PathFinder's connected_components.

    Shows tasks that are completely disconnected from the main graph.
    """
    from cortical.got import PathFinder, Query

    got_manager = manager._manager

    print("ORPHAN ANALYSIS")
    print("=" * 60)
    print()

    # Find connected components
    components = PathFinder(got_manager).connected_components()

    # Find the largest component (main graph)
    components.sort(key=len, reverse=True)
    main_component = components[0] if components else set()

    print(f"Found {len(components)} connected components:")
    print(f"  Main component: {len(main_component)} nodes")
    print()

    if len(components) > 1:
        print("ORPHAN CLUSTERS (disconnected from main graph):")
        print("-" * 40)

        orphan_count = 0
        for i, component in enumerate(components[1:], 1):
            print(f"\n  Cluster {i} ({len(component)} nodes):")
            for node_id in list(component)[:5]:  # Show first 5
                task = got_manager.get_task(node_id)
                if task:
                    print(f"    - {task.title[:50]}")
                    orphan_count += 1
            if len(component) > 5:
                print(f"    ... and {len(component) - 5} more")

        print(f"\nTotal orphan tasks: {orphan_count}")
    else:
        print("No orphan clusters found - all tasks are connected!")
    print()

    return 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _status_icon(status: str) -> str:
    """Return emoji/icon for status."""
    icons = {
        "completed": "[x]",
        "in_progress": "[~]",
        "pending": "[ ]",
        "blocked": "[!]",
    }
    return icons.get(status, "[?]")


def _task_age_str(created_at: str) -> str:
    """Return human-readable age string."""
    try:
        if isinstance(created_at, str):
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            created = created_at

        now = datetime.now(timezone.utc)
        delta = now - created

        if delta.days > 30:
            return f"{delta.days // 30}mo ago"
        elif delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        else:
            return "recent"
    except Exception:
        return "unknown"


# =============================================================================
# CLI PARSER SETUP
# =============================================================================


def setup_analyze_parser(subparsers):
    """Set up 'analyze' subcommand parser."""
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Graph analysis using fluent Query API",
        description="Powerful analysis commands using Query builder, "
                    "GraphWalker, PathFinder, and PatternMatcher."
    )

    analyze_subparsers = analyze_parser.add_subparsers(
        dest="analyze_command",
        help="Analysis commands"
    )

    # Summary command
    analyze_subparsers.add_parser(
        "summary",
        help="Quick status summary with task counts"
    )

    # Dependencies command
    deps_parser = analyze_subparsers.add_parser(
        "dependencies",
        help="Analyze dependency chain for a task"
    )
    deps_parser.add_argument(
        "task_id",
        help="Task ID to analyze"
    )

    # Patterns command
    analyze_subparsers.add_parser(
        "patterns",
        help="Find common graph patterns"
    )

    # Orphans command
    analyze_subparsers.add_parser(
        "orphans",
        help="Find disconnected task clusters"
    )


def handle_analyze_command(args, manager: "TransactionalGoTAdapter") -> int:
    """Route 'analyze' subcommand to appropriate handler."""
    if not hasattr(args, 'analyze_command') or not args.analyze_command:
        # Default to summary
        return cmd_analyze_summary(args, manager)

    if args.analyze_command == "summary":
        return cmd_analyze_summary(args, manager)
    elif args.analyze_command == "dependencies":
        return cmd_analyze_dependencies(args, manager)
    elif args.analyze_command == "patterns":
        return cmd_analyze_patterns(args, manager)
    elif args.analyze_command == "orphans":
        return cmd_analyze_orphans(args, manager)

    return 1
