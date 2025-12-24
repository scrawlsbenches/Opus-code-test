"""
Query and validation CLI commands for GoT system.

Provides commands for:
- Querying the graph
- Showing blocked/active/stats
- Validating graph health
- Inferring edges from git
- Event compaction

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING

from .shared import format_task_table

if TYPE_CHECKING:
    from scripts.got_utils import GoTProjectManager


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_query(args, manager: "GoTProjectManager") -> int:
    """Handle 'got query' command."""
    query_str = " ".join(args.query_string)

    print(f"Query: {query_str}\n")

    results = manager.query(query_str)

    if not results:
        print("No results found.")
        return 0

    print(f"Results ({len(results)}):\n")
    for r in results:
        if "step" in r:
            # Path query
            print(f"  [{r['step']}] {r['id']}: {r['title']}")
        elif "relation" in r:
            # Relationship query
            print(f"  {r['relation']}: {r['id']}")
            if r.get('title'):
                print(f"      {r['title']}")
        elif "reason" in r:
            # Blocked tasks
            print(f"  {r['id']}: {r['title']}")
            print(f"      Reason: {r['reason']}")
        else:
            # Generic result
            print(f"  {r['id']}: {r.get('title', '')}")
            if r.get('priority'):
                print(f"      Priority: {r['priority']}")
            if r.get('status'):
                print(f"      Status: {r['status']}")
        print()

    return 0


def cmd_blocked(args, manager: "GoTProjectManager") -> int:
    """Handle 'got blocked' command."""
    blocked = manager.get_blocked_tasks()

    if not blocked:
        print("No blocked tasks.")
        return 0

    print(f"Blocked Tasks ({len(blocked)}):")
    print()

    for task, reason in blocked:
        print(f"  {task.id}")
        print(f"    Title: {task.content}")
        print(f"    Reason: {reason}")
        print()

    return 0


def cmd_active(args, manager: "GoTProjectManager") -> int:
    """Handle 'got active' command."""
    active = manager.get_active_tasks()
    print(format_task_table(active))
    return 0


def cmd_stats(args, manager: "GoTProjectManager") -> int:
    """Handle 'got stats' command."""
    stats = manager.get_stats()

    print("GoT Project Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Total sprints: {stats['total_sprints']}")
    print(f"  Total epics: {stats['total_epics']}")
    print(f"  Total edges: {stats['total_edges']}")
    print()
    print("Tasks by status:")
    for status, count in stats.get("tasks_by_status", {}).items():
        print(f"  {status}: {count}")

    return 0


def cmd_dashboard(args, manager: "GoTProjectManager") -> int:
    """Handle 'got dashboard' command."""
    # Import dashboard module
    try:
        from scripts.got_dashboard import render_dashboard
        dashboard = render_dashboard(manager)
        print(dashboard)
        return 0
    except ImportError as e:
        print(f"Error: Could not import dashboard module: {e}")
        return 1
    except Exception as e:
        print(f"Error rendering dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_validate(args, manager: "GoTProjectManager") -> int:
    """Handle 'got validate' command."""
    from cortical.reasoning.graph_of_thought import NodeType

    print("=" * 60)
    print("GoT VALIDATION REPORT")
    print("=" * 60)

    issues = []
    warnings = []

    # Count nodes and edges from TX backend entities
    total_nodes = len(manager.graph.nodes)
    total_edges = len(manager.graph.edges)

    # Count tasks by status
    tasks = [n for n in manager.graph.nodes.values() if n.node_type == NodeType.TASK]
    task_count = len(tasks)

    # Count by status
    by_status = {}
    for task in tasks:
        status = str(task.properties.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1

    # Check for orphan nodes (no edges)
    nodes_with_edges = set()
    for edge in manager.graph.edges:
        nodes_with_edges.add(edge.source_id)
        nodes_with_edges.add(edge.target_id)

    orphan_count = total_nodes - len(nodes_with_edges)
    orphan_rate = orphan_count / max(total_nodes, 1) * 100

    # Check orphan rate (warning if high, but not critical)
    if orphan_rate > 50:
        warnings.append(f"High orphan rate: {orphan_rate:.1f}% of nodes have no edges")
    elif orphan_rate > 25:
        warnings.append(f"Moderate orphan rate: {orphan_rate:.1f}%")

    # Check edge density
    edge_density = total_edges / max(total_nodes, 1)
    if edge_density < 0.1 and total_nodes > 10:
        warnings.append(f"Low edge density: {edge_density:.2f} edges/node")

    # Count entity files for accurate statistics
    entities_dir = manager.got_dir / "entities"
    task_files = len(list(entities_dir.glob("T-*.json"))) if entities_dir.exists() else 0
    edge_files = len(list(entities_dir.glob("E-*.json"))) if entities_dir.exists() else 0
    decision_files = len(list(entities_dir.glob("D-*.json"))) if entities_dir.exists() else 0
    handoff_files = len(list(entities_dir.glob("H-*.json"))) if entities_dir.exists() else 0

    # Print stats
    print(f"\n📊 STATISTICS")
    print(f"   Tasks: {task_count}")
    print(f"   Edges: {total_edges}")
    print(f"   Edge density: {edge_density:.2f} edges/node")
    print(f"   Orphan nodes: {orphan_count} ({orphan_rate:.1f}%)")

    print(f"\n📁 ENTITY FILES")
    print(f"   Task files: {task_files}")
    print(f"   Edge files: {edge_files}")
    print(f"   Decision files: {decision_files}")
    print(f"   Handoff files: {handoff_files}")

    print(f"\n📈 TASKS BY STATUS")
    for status, count in sorted(by_status.items()):
        print(f"   {status}: {count}")

    # Print issues
    if issues:
        print(f"\n❌ ISSUES ({len(issues)})")
        for issue in issues:
            print(f"   • {issue}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)})")
        for warning in warnings:
            print(f"   • {warning}")

    if not issues and not warnings:
        print("\n✅ HEALTHY - No issues detected")

    print()

    # Return non-zero if critical issues
    return 1 if issues else 0


def cmd_infer(args, manager: "GoTProjectManager") -> int:
    """Handle 'got infer' command."""
    if args.message:
        # Analyze a specific message
        edges = manager.infer_edges_from_commit(args.message)
        print(f"Analyzing message: {args.message[:50]}...")
    else:
        # Analyze recent commits
        edges = manager.infer_edges_from_recent_commits(args.commits)
        print(f"Analyzed last {args.commits} commits")

    if not edges:
        print("\nNo task references found in commits.")
        return 0

    print(f"\nEdges inferred ({len(edges)}):\n")
    for edge in edges:
        if "commit_hash" in edge:
            print(
                f"  [{edge['commit_hash']}] {edge['type']}: "
                f"{edge.get('from', edge.get('commit', ''))} → "
                f"{edge.get('to', edge.get('task', ''))}"
            )
        else:
            print(
                f"  {edge['type']}: {edge.get('from', '')} → {edge.get('to', '')}"
            )

    return 0


def cmd_compact(args, manager: "GoTProjectManager") -> int:
    """Handle 'got compact' command.

    DEPRECATED: This command is for the legacy event-sourced backend.
    The TX backend uses entity files in .got/entities/ which don't need compaction.
    """
    print("The 'compact' command is deprecated.")
    print("The TX backend stores entities directly in .got/entities/ and doesn't use event logs.")
    print("No compaction is needed.")
    return 0


def cmd_export(args, manager: "GoTProjectManager") -> int:
    """Handle 'got export' command."""
    from pathlib import Path

    output = getattr(args, 'output', None)
    if output:
        output = Path(output)

    data = manager.export_graph(output)

    if output:
        print(f"Exported to: {output}")
    else:
        print(json.dumps(data, indent=2))

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_query_parser(subparsers) -> None:
    """
    Set up argparse subparsers for query commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the graph")
    query_parser.add_argument(
        "query_string",
        nargs="+",
        help="Query (e.g., 'what blocks task:T-...')"
    )

    # Simple query shortcuts
    subparsers.add_parser("blocked", help="Show blocked tasks")
    subparsers.add_parser("active", help="Show active tasks")
    subparsers.add_parser("stats", help="Show statistics")
    subparsers.add_parser("dashboard", help="Show comprehensive metrics dashboard")

    # Validation command
    subparsers.add_parser("validate", help="Validate graph health")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Infer edges from git history")
    infer_parser.add_argument(
        "--commits", "-n",
        type=int,
        default=10,
        help="Number of recent commits to analyze"
    )
    infer_parser.add_argument(
        "--message", "-m",
        help="Analyze a specific commit message"
    )

    # Compact command
    compact_parser = subparsers.add_parser(
        "compact",
        help="Compact old events into consolidated file"
    )
    compact_parser.add_argument(
        "--preserve-days", "-d",
        type=int,
        default=7,
        help="Preserve events from last N days (default: 7)"
    )
    compact_parser.add_argument(
        "--no-preserve-handoffs",
        action="store_true",
        help="Don't preserve handoff events"
    )
    compact_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be compacted"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export graph")
    export_parser.add_argument("--output", "-o", help="Output file")


def handle_query_commands(args, manager: "GoTProjectManager") -> int:
    """
    Route query-related commands to appropriate handlers.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error), or None if not handled
    """
    command = args.command

    handlers = {
        "query": cmd_query,
        "blocked": cmd_blocked,
        "active": cmd_active,
        "stats": cmd_stats,
        "dashboard": cmd_dashboard,
        "validate": cmd_validate,
        "infer": cmd_infer,
        "compact": cmd_compact,
        "export": cmd_export,
    }

    handler = handlers.get(command)
    if handler:
        return handler(args, manager)

    return None  # Not handled by this module
