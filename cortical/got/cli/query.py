# MERGE_CONFLICT_RESOLVED: From branch claude/engineering-session-T73QD on 20260108-215836
"""
Query and validation CLI commands for GoT system.

Provides commands for:
- Querying the graph (legacy natural language and new expression-based)
- Showing blocked/active/stats
- Validating graph health
- Inferring edges from git

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, List

from .shared import format_task_table

if TYPE_CHECKING:
    from cortical.got.adapter import TransactionalGoTAdapter


# 
# CLI COMMAND HANDLERS
# 

def cmd_query(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_blocked(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_active(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got active' command."""
    active = manager.get_active_tasks()
    print(format_task_table(active))
    return 0


def cmd_stats(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_dashboard(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_validate(args, manager: "TransactionalGoTAdapter") -> int:
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

    # Count tasks by status - use list_all_tasks for compatibility with both adapters
    tasks = manager.list_all_tasks()
    task_count = len(tasks)

    # Count by status - support both Task objects (.status) and ThoughtNode (.properties["status"])
    by_status = {}
    for task in tasks:
        status = getattr(task, 'status', None) or task.properties.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    # Check for orphan nodes (no edges)

    # Build comprehensive set of ALL entity IDs (not just graph.nodes which only has TASK/DECISION)
    all_node_ids = set(manager.graph.nodes.keys())

    # Add Sprint IDs - stored separately from graph.nodes
    try:
        for sprint in manager.list_sprints():
            all_node_ids.add(sprint.id if hasattr(sprint, 'id') else sprint.get('id'))
    except Exception:
        pass  # Sprint listing may fail, continue with what we have

    # Add Epic IDs - stored separately from graph.nodes
    try:
        for epic in manager.list_epics():
            all_node_ids.add(epic.id if hasattr(epic, 'id') else epic.get('id'))
    except Exception:
        pass

    # Add Handoff IDs - stored separately from graph.nodes
    try:
        for handoff in manager.list_handoffs():
            all_node_ids.add(handoff.get('id') if isinstance(handoff, dict) else handoff.id)
    except Exception:
        pass

    # Add KnowledgeTransfer IDs - stored separately from graph.nodes
    try:
        for kt in manager.list_knowledge_transfers():
            all_node_ids.add(kt.get('id') if isinstance(kt, dict) else kt.id)
    except Exception:
        pass

    nodes_with_edges = set()
    orphan_edges = []  # Edges pointing to non-existent entities

    # CRITICAL: Use list_edges() to get ALL edges from disk, not graph.edges which may be incomplete
    # Bug fix: graph.edges only contains edges loaded in memory, missing ~50 edges
    all_edges = manager.list_edges()
    for edge in all_edges:
        # Handle both Edge objects and dicts
        source_id = edge.source_id if hasattr(edge, 'source_id') else edge.get('source_id')
        target_id = edge.target_id if hasattr(edge, 'target_id') else edge.get('target_id')

        # Create edge identifier from source->target
        edge_repr = f"{source_id}->{target_id}"
        if source_id in all_node_ids:
            nodes_with_edges.add(source_id)
        else:
            orphan_edges.append((edge_repr, "source", source_id))
        if target_id in all_node_ids:
            nodes_with_edges.add(target_id)
        else:
            orphan_edges.append((edge_repr, "target", target_id))

    orphan_count = len(all_node_ids - nodes_with_edges)
    # Bug fix: Use all_node_ids count as denominator, not just graph.nodes (which only has Tasks+Decisions)
    total_all_entities = len(all_node_ids)
    orphan_rate = orphan_count / max(total_all_entities, 1) * 100

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

    # Check for broken edge references if --check-refs is passed
    if getattr(args, 'check_refs', False):
        if orphan_edges:
            issues.append(f"Found {len(orphan_edges)} broken edge reference(s)")
            print(f"\n🔗 BROKEN EDGE REFERENCES ({len(orphan_edges)})")
            for edge_id, ref_type, missing_id in orphan_edges[:10]:  # Show first 10
                print(f"   • Edge {edge_id}: {ref_type} '{missing_id}' not found")
            if len(orphan_edges) > 10:
                print(f"   ... and {len(orphan_edges) - 10} more")
        else:
            print("\n🔗 EDGE REFERENCES: All valid")

    if not issues and not warnings:
        print("\n✅ HEALTHY - No issues detected")

    print()

    # Return non-zero if critical issues
    return 1 if issues else 0


def cmd_infer(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_export(args, manager: "TransactionalGoTAdapter") -> int:
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


# 
# EXPRESSION QUERY COMMANDS
# 

def cmd_expr(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got expr' command - expression-based queries.

    Uses the new expression parser to execute structured queries like:
        got expr "status = 'pending' AND priority = 'high'"
        got expr "NOT status = 'completed'"
        got expr "recent(days=7)"
    """
    from cortical.got.expression import parse, execute, validate
    from cortical.got.expression.errors import QueryError

    query_str = " ".join(args.expression)

    # Handle special flags first
    if getattr(args, 'list_fields', False):
        return _cmd_list_fields(args)

    if getattr(args, 'list_functions', False):
        return _cmd_list_functions()

    if getattr(args, 'explain', False):
        return _cmd_explain(query_str)

    if not query_str:
        print("Error: No expression provided.")
        print("Usage: got expr \"status = 'pending'\"")
        return 1

    try:
        # Parse the expression
        query = parse(query_str)

        # Validate if entity type is specified
        entity_type = getattr(args, 'type', 'task')
        if entity_type:
            validate(query, entity_type=entity_type)

        # Execute against the GoT manager
        # The execute function expects a GoTManager. TransactionalGoTAdapter
        # stores the actual manager in _manager
        got_manager = getattr(manager, '_manager', manager)
        results = execute(got_manager, query)

        # Format and display results
        _display_results(results, query_str, args)

        return 0

    except QueryError as e:
        print(f"Query Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if getattr(args, 'debug', False):
            import traceback
            traceback.print_exc()
        return 1


def _display_results(results: Any, query_str: str, args) -> None:
    """Display query results in a formatted way."""
    output_format = getattr(args, 'format', 'table')
    show_count = getattr(args, 'count', False)

    if show_count:
        print(len(results) if hasattr(results, '__len__') else 1)
        return

    if not results:
        print("No results found.")
        return

    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    print(f"Results ({len(results)}):\n")

    if output_format == 'json':
        # JSON output
        output = []
        for r in results:
            if hasattr(r, 'to_dict'):
                output.append(r.to_dict())
            elif hasattr(r, '__dict__'):
                output.append({k: v for k, v in r.__dict__.items()
                             if not k.startswith('_')})
            else:
                output.append(str(r))
        print(json.dumps(output, indent=2, default=str))
    elif output_format == 'ids':
        # IDs only
        for r in results:
            entity_id = getattr(r, 'id', None) or getattr(r, 'entity_id', str(r))
            print(entity_id)
    else:
        # Table format (default)
        for r in results:
            _print_result_item(r)


def _print_result_item(item: Any) -> None:
    """Print a single result item."""
    # Get entity ID
    entity_id = getattr(item, 'id', None) or getattr(item, 'entity_id', None)

    # Get title/content
    title = (getattr(item, 'title', None) or
             getattr(item, 'content', None) or
             getattr(item, 'name', None) or
             str(item))

    if entity_id:
        print(f"  {entity_id}")
        if title and str(title) != str(entity_id):
            print(f"    Title: {title}")
    else:
        print(f"  {title}")

    # Show status and priority if available
    status = getattr(item, 'status', None)
    if status is None and hasattr(item, 'properties'):
        status = item.properties.get('status')
    if status:
        print(f"    Status: {status}")

    priority = getattr(item, 'priority', None)
    if priority is None and hasattr(item, 'properties'):
        priority = item.properties.get('priority')
    if priority:
        print(f"    Priority: {priority}")

    print()


def _cmd_list_fields(args) -> int:
    """List available fields for an entity type."""
    from cortical.core.bootstrap import get_container
    from cortical.cdg.schema import SchemaRegistry

    container = get_container()
    registry = container.resolve(SchemaRegistry)

    entity_type = getattr(args, 'type', 'task')

    schema = registry.get_schema(entity_type)
    if schema is not None:
        print(f"Fields for '{entity_type}':\n")
        for field_name, field_def in schema.fields.items():
            field_type = field_def.field_type.name if hasattr(field_def.field_type, 'name') else str(field_def.field_type)
            required = "required" if field_def.required else "optional"
            print(f"  {field_name}: {field_type.lower()} ({required})")
            if field_def.description:
                print(f"      {field_def.description}")
        return 0
    else:
        print(f"No schema registered for '{entity_type}'.")
        # Fall back to common fields
        from cortical.got.expression.validator import COMMON_FIELDS
        print(f"\nCommon fields (available for all entity types):")
        for field in sorted(COMMON_FIELDS):
            print(f"  {field}")
        return 0


def _cmd_list_functions() -> int:
    """List available query functions."""
    from cortical.got.expression.registry import FunctionRegistry

    # Ensure functions are registered
    from cortical.got.expression.functions import graph, filters  # noqa: F401

    registry = FunctionRegistry.instance()
    functions = registry.list_functions()

    if not functions:
        print("No functions registered.")
        return 0

    print("Available query functions:\n")

    # Group functions - FunctionSignature doesn't have category, so just list all
    for sig in sorted(functions, key=lambda s: s.name):
        # Format parameters from required_args and optional_args
        params = []
        # Required args first
        for arg in sig.required_args:
            params.append(arg)
        # Optional args with defaults
        for arg, default in sig.optional_args.items():
            params.append(f"{arg}={default}")
        params_str = ", ".join(params)

        print(f"  {sig.name}({params_str})")
        if sig.description:
            print(f"      {sig.description}")
        if sig.returns:
            print(f"      Returns: {sig.returns}")
        print()

    return 0


def _cmd_explain(query_str: str) -> int:
    """Explain how a query will be executed."""
    from cortical.got.expression import parse
    from cortical.got.expression.errors import QueryError

    if not query_str:
        print("Error: No expression to explain.")
        return 1

    try:
        query = parse(query_str)

        print(f"Expression: {query_str}\n")
        print("Parsed AST:")
        _print_ast(query.expression, indent=2)

        print(f"\nEntity type: {query.entity_type or 'task (default)'}")
        if query.order_by:
            field, desc = query.order_by
            print(f"Order by: {field} {'DESC' if desc else 'ASC'}")
        if query.limit:
            print(f"Limit: {query.limit}")
        if query.offset:
            print(f"Offset: {query.offset}")

        return 0

    except QueryError as e:
        print(f"Parse Error: {e}")
        return 1


def _print_ast(node, indent=0) -> None:
    """Print AST node for debugging."""
    from cortical.got.expression.ast import (
        Comparison, AndExpr, OrExpr, NotExpr, FunctionCall, Literal, Field
    )

    prefix = " " * indent

    if node is None:
        print(f"{prefix}(empty)")
    elif isinstance(node, Comparison):
        print(f"{prefix}Comparison:")
        print(f"{prefix}  field: {node.field.name if hasattr(node.field, 'name') else node.field}")
        print(f"{prefix}  op: {node.op.name}")
        print(f"{prefix}  value: {node.value.value if hasattr(node.value, 'value') else node.value}")
    elif isinstance(node, AndExpr):
        print(f"{prefix}AND:")
        for child in node.children:
            _print_ast(child, indent + 2)
    elif isinstance(node, OrExpr):
        print(f"{prefix}OR:")
        for child in node.children:
            _print_ast(child, indent + 2)
    elif isinstance(node, NotExpr):
        print(f"{prefix}NOT:")
        _print_ast(node.child, indent + 2)
    elif isinstance(node, FunctionCall):
        args_str = ", ".join(str(a) for a in node.args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in node.kwargs)
        print(f"{prefix}Function: {node.name}({args_str}{', ' + kwargs_str if kwargs_str else ''})")
    else:
        print(f"{prefix}{type(node).__name__}: {node}")


# 
# CLI INTEGRATION
# 

def setup_query_parser(subparsers) -> None:
    """
    Set up argparse subparsers for query commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Query command (legacy natural language)
    query_parser = subparsers.add_parser("query", help="Query the graph (natural language)")
    query_parser.add_argument(
        "query_string",
        nargs="+",
        help="Query (e.g., 'what blocks task:T-...')"
    )

    # Expression query command (new structured queries)
    expr_parser = subparsers.add_parser(
        "expr",
        help="Query using expression syntax (e.g., \"status = 'pending'\")",
        description="""
Execute structured queries using the expression DSL.

Examples:
  got expr "status = 'pending'"
  got expr "status = 'pending' AND priority = 'high'"
  got expr "NOT status = 'completed'"
  got expr "priority IN ['high', 'critical']"
  got expr --type decision "status = 'draft'"
  got expr --list-functions
  got expr --list-fields --type task
  got expr --explain "status = 'pending' AND priority = 'high'"
        """,
    )
    expr_parser.add_argument(
        "expression",
        nargs="*",
        help="Expression to query (e.g., \"status = 'pending'\")"
    )
    expr_parser.add_argument(
        "--type", "-t",
        default="task",
        help="Entity type to query (default: task)"
    )
    expr_parser.add_argument(
        "--format", "-f",
        choices=["table", "json", "ids"],
        default="table",
        help="Output format (default: table)"
    )
    expr_parser.add_argument(
        "--count", "-c",
        action="store_true",
        help="Only show the count of results"
    )
    expr_parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available fields for the entity type"
    )
    expr_parser.add_argument(
        "--list-functions",
        action="store_true",
        help="List available query functions"
    )
    expr_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show how the query will be parsed and executed"
    )
    expr_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information on errors"
    )

    # Simple query shortcuts
    subparsers.add_parser("blocked", help="Show blocked tasks")
    subparsers.add_parser("active", help="Show active tasks")
    subparsers.add_parser("stats", help="Show statistics")
    subparsers.add_parser("dashboard", help="Show comprehensive metrics dashboard")

    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate graph health")
    validate_parser.add_argument(
        "--check-refs",
        action="store_true",
        help="Check for broken edge references (edges pointing to non-existent entities)"
    )

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

    # Export command
    export_parser = subparsers.add_parser("export", help="Export graph")
    export_parser.add_argument("--output", "-o", help="Output file")


def handle_query_commands(args, manager: "TransactionalGoTAdapter") -> int:
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
        "expr": cmd_expr,
        "blocked": cmd_blocked,
        "active": cmd_active,
        "stats": cmd_stats,
        "dashboard": cmd_dashboard,
        "validate": cmd_validate,
        "infer": cmd_infer,
        "export": cmd_export,
    }

    handler = handlers.get(command)
    if handler:
        return handler(args, manager)

    return None  # Not handled by this module
