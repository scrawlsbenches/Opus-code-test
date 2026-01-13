"""
Edge CLI commands for GoT system.

Provides commands for:
- Adding edges (relationships) between entities
- Listing edges
- Removing edges
- Showing edge types

This module addresses the CLI gap where users couldn't directly manage edges.
"""

from typing import TYPE_CHECKING, List

from cortical.got.types import VALID_EDGE_TYPES

if TYPE_CHECKING:
    from cortical.got.api import GoTManager


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_edge_add(args, manager: "GoTManager") -> int:
    """Handle 'got edge add' command."""
    edge_type = args.edge_type.upper()

    if edge_type not in VALID_EDGE_TYPES:
        print(f"Error: Invalid edge type '{edge_type}'")
        print(f"Valid types: {', '.join(sorted(VALID_EDGE_TYPES))}")
        return 1

    reason = getattr(args, 'reason', None) or ""

    try:
        edge = manager.add_edge(
            source_id=args.source_id,
            target_id=args.target_id,
            edge_type=edge_type,
            weight=getattr(args, 'weight', 1.0),
            reason=reason,
        )

        if edge:
            manager.save()
            print(f"Created edge: {args.source_id} --[{edge_type}]--> {args.target_id}")
            if reason:
                print(f"  Reason: {reason}")
            return 0
        else:
            print("Failed to create edge - check that both entity IDs exist")
            return 1
    except Exception as e:
        print(f"Error creating edge: {e}")
        return 1


def cmd_edge_list(args, manager: "GoTManager") -> int:
    """Handle 'got edge list' command."""
    try:
        edges = manager.list_edges()

        if not edges:
            print("No edges found")
            return 0

        # Filter by type if specified
        if hasattr(args, 'type') and args.type:
            filter_type = args.type.upper()
            edges = [e for e in edges if e.edge_type.upper() == filter_type]

        # Filter by source if specified
        if hasattr(args, 'source') and args.source:
            edges = [e for e in edges if e.source_id == args.source or
                     e.source_id.endswith(args.source)]

        # Filter by target if specified
        if hasattr(args, 'target') and args.target:
            edges = [e for e in edges if e.target_id == args.target or
                     e.target_id.endswith(args.target)]

        if not edges:
            print("No edges match the filter criteria")
            return 0

        # Format output
        print(f"\n{'Source':<40} {'Type':<15} {'Target':<40}")
        print("-" * 100)

        for edge in edges:
            source = edge.source_id if hasattr(edge, 'source_id') else edge.from_id
            target = edge.target_id if hasattr(edge, 'target_id') else edge.to_id
            edge_type = edge.edge_type

            # Truncate long IDs
            source_display = source[:37] + "..." if len(source) > 40 else source
            target_display = target[:37] + "..." if len(target) > 40 else target

            print(f"{source_display:<40} {edge_type:<15} {target_display:<40}")

        print(f"\nTotal: {len(edges)} edge(s)")
        return 0

    except Exception as e:
        print(f"Error listing edges: {e}")
        return 1


def cmd_edge_types(args, manager: "GoTManager") -> int:
    """Handle 'got edge types' command - show available edge types.

    Displays the 22 authoritative edge types from VALID_EDGE_TYPES.
    These were consolidated in OI-002 resolution.
    """
    print("\nAvailable Edge Types (22 authoritative types):")
    print("=" * 60)

    # Categories based on the authoritative VALID_EDGE_TYPES from cortical/got/types.py
    categories = {
        "Core Relationships": [
            ("DEPENDS_ON", "Task A depends on Task B"),
            ("BLOCKS", "Task A blocks Task B"),
            ("CONTAINS", "Sprint contains Task, Epic contains Sprint"),
            ("RELATES_TO", "General relationship"),
            ("REQUIRES", "Hard requirement"),
            ("IMPLEMENTS", "Task implements Decision"),
            ("SUPERSEDES", "Entity replaces another"),
            ("DERIVED_FROM", "Entity derived from another"),
        ],
        "Hierarchical Relationships": [
            ("PARENT_OF", "Hierarchical parent"),
            ("CHILD_OF", "Hierarchical child"),
            ("PART_OF", "Component of larger entity"),
        ],
        "Reference & Semantic Relationships": [
            ("REFERENCES", "Soft reference"),
            ("CONTRADICTS", "Conflicting entities"),
            ("JUSTIFIES", "Decision justifies Task"),
            ("MOTIVATES", "Entity motivates another"),
            ("CAUSED_BY", "Entity was caused by another"),
        ],
        "Workflow Relationships": [
            ("TRANSFERS", "Task transfers to Handoff"),
            ("PRODUCES", "Task produces Document/Artifact"),
            ("DOCUMENTED_BY", "Task/Decision documented by Document"),
        ],
        "Knowledge Transfer Relationships": [
            ("DOCUMENTS", "KnowledgeTransfer documents Task/Decision"),
            ("CONTINUES", "KnowledgeTransfer continues from Handoff"),
        ],
        "Failure Tracking": [
            ("FAILED_ATTEMPT", "Failure record for a failed approach"),
        ],
    }

    for category, types in categories.items():
        print(f"\n{category}:")
        print("-" * 50)
        for type_name, description in types:
            # Verify this type is actually in VALID_EDGE_TYPES
            marker = "  " if type_name in VALID_EDGE_TYPES else "? "
            print(f"{marker}{type_name:<15} {description}")

    print()
    return 0


def cmd_edge_for(args, manager: "GoTManager") -> int:
    """Handle 'got edge for' command - show edges for a specific entity."""
    entity_id = args.entity_id

    try:
        # Try to get edges for task (most common case)
        try:
            outgoing, incoming = manager.get_edges_for_task(entity_id)
        except Exception:
            # Fall back to listing all and filtering
            all_edges = manager.list_edges()
            outgoing = [e for e in all_edges if
                       (hasattr(e, 'source_id') and e.source_id == entity_id) or
                       (hasattr(e, 'from_id') and e.from_id == entity_id)]
            incoming = [e for e in all_edges if
                       (hasattr(e, 'target_id') and e.target_id == entity_id) or
                       (hasattr(e, 'to_id') and e.to_id == entity_id)]

        if not outgoing and not incoming:
            print(f"No edges found for entity: {entity_id}")
            return 0

        print(f"\nEdges for: {entity_id}")
        print("=" * 70)

        if outgoing:
            print(f"\nOutgoing ({len(outgoing)}):")
            print("-" * 50)
            for edge in outgoing:
                target = edge.target_id if hasattr(edge, 'target_id') else edge.to_id
                print(f"  --[{edge.edge_type}]--> {target}")

        if incoming:
            print(f"\nIncoming ({len(incoming)}):")
            print("-" * 50)
            for edge in incoming:
                source = edge.source_id if hasattr(edge, 'source_id') else edge.from_id
                print(f"  <--[{edge.edge_type}]-- {source}")

        print()
        return 0

    except Exception as e:
        print(f"Error getting edges: {e}")
        return 1


# =============================================================================
# PARSER SETUP
# =============================================================================

def setup_edge_parser(subparsers) -> None:
    """Set up the edge subparser and its subcommands."""
    edge_parser = subparsers.add_parser(
        "edge",
        help="Edge (relationship) operations",
        description="Manage edges between entities in the graph"
    )

    edge_subparsers = edge_parser.add_subparsers(
        dest="edge_command",
        help="Edge subcommands"
    )

    # edge add
    add_parser = edge_subparsers.add_parser(
        "add",
        help="Add an edge between two entities"
    )
    add_parser.add_argument(
        "source_id",
        help="Source entity ID"
    )
    add_parser.add_argument(
        "target_id",
        help="Target entity ID"
    )
    add_parser.add_argument(
        "edge_type",
        help="Type of relationship (e.g., DEPENDS_ON, BLOCKS, CAUSED_BY)"
    )
    add_parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Edge weight (default: 1.0)"
    )
    add_parser.add_argument(
        "--reason", "-r",
        help="Why this relationship exists (context capture)"
    )

    # edge list
    list_parser = edge_subparsers.add_parser(
        "list",
        help="List all edges"
    )
    list_parser.add_argument(
        "--type",
        help="Filter by edge type"
    )
    list_parser.add_argument(
        "--source",
        help="Filter by source entity ID (partial match)"
    )
    list_parser.add_argument(
        "--target",
        help="Filter by target entity ID (partial match)"
    )

    # edge types
    edge_subparsers.add_parser(
        "types",
        help="Show available edge types with descriptions"
    )

    # edge for
    for_parser = edge_subparsers.add_parser(
        "for",
        help="Show all edges for a specific entity"
    )
    for_parser.add_argument(
        "entity_id",
        help="Entity ID to show edges for"
    )


def handle_edge_command(args, manager: "GoTManager") -> int:
    """
    Route edge subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoT manager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'edge_command') or not args.edge_command:
        print("Error: No edge subcommand specified")
        print("Use 'got edge --help' for available commands")
        return 1

    command_handlers = {
        "add": cmd_edge_add,
        "list": cmd_edge_list,
        "types": cmd_edge_types,
        "for": cmd_edge_for,
    }

    handler = command_handlers.get(args.edge_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown edge subcommand: {args.edge_command}")
    return 1
