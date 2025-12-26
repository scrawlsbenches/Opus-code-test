"""
Backup and sync CLI commands for GoT system.

Provides commands for:
- Creating snapshots
- Listing snapshots
- Verifying snapshot integrity
- Restoring from snapshots
- Syncing to git

This module can be integrated into got_utils.py CLI or used standalone.
"""

import gzip
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_backup_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backup create' command."""
    compress = getattr(args, 'compress', True)

    try:
        snapshot_id = manager.wal.create_snapshot(manager.graph, compress=compress)
        print(f"Snapshot created: {snapshot_id}")

        # Show snapshot info
        snapshots_dir = manager.got_dir / "wal" / "snapshots"
        snapshot_files = list(snapshots_dir.glob(f"*{snapshot_id}*"))
        if snapshot_files:
            size = snapshot_files[0].stat().st_size
            print(f"  Size: {size / 1024:.1f} KB")
            print(f"  Compressed: {compress}")
        return 0
    except Exception as e:
        print(f"Error creating snapshot: {e}")
        return 1


def cmd_backup_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backup list' command."""
    limit = getattr(args, 'limit', 10)

    snapshots_dir = manager.got_dir / "wal" / "snapshots"
    if not snapshots_dir.exists():
        print("No snapshots found.")
        return 0

    # Find all snapshot files
    snapshots = []
    for snap_file in sorted(snapshots_dir.glob("snap_*.json*"), reverse=True):
        try:
            size = snap_file.stat().st_size
            is_compressed = snap_file.suffix == ".gz"

            # Extract timestamp from filename
            name = snap_file.stem
            if name.endswith(".json"):
                name = name[:-5]
            parts = name.split("_")
            if len(parts) >= 3:
                timestamp = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]} "
                timestamp += f"{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}"
            else:
                timestamp = "unknown"

            # Try to get node count
            node_count = "?"
            try:
                if is_compressed:
                    with gzip.open(snap_file, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(snap_file) as f:
                        data = json.load(f)
                state = data.get("state", data)
                nodes = state.get("nodes", {})
                node_count = len(nodes)
            except Exception:
                pass

            snapshots.append({
                "file": snap_file.name,
                "timestamp": timestamp,
                "size": size,
                "compressed": is_compressed,
                "nodes": node_count
            })
        except Exception:
            continue

    if not snapshots:
        print("No snapshots found.")
        return 0

    print(f"Available Snapshots ({len(snapshots)} total):\n")
    print(f"{'Timestamp':<20} {'Nodes':<8} {'Size':<10} {'File'}")
    print("-" * 70)

    for snap in snapshots[:limit]:
        size_str = f"{snap['size'] / 1024:.1f} KB"
        print(
            f"{snap['timestamp']:<20} {str(snap['nodes']):<8} "
            f"{size_str:<10} {snap['file']}"
        )

    if len(snapshots) > limit:
        print(f"\n... and {len(snapshots) - limit} more")

    return 0


def cmd_backup_verify(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backup verify' command."""
    snapshot_id = getattr(args, 'snapshot_id', None)

    snapshots_dir = manager.got_dir / "wal" / "snapshots"
    if not snapshots_dir.exists():
        print("No snapshots found.")
        return 1

    # Find the snapshot to verify
    if snapshot_id:
        files = list(snapshots_dir.glob(f"*{snapshot_id}*"))
    else:
        files = sorted(snapshots_dir.glob("snap_*.json*"), reverse=True)

    if not files:
        print(f"Snapshot not found: {snapshot_id or '(latest)'}")
        return 1

    snap_file = files[0]
    print(f"Verifying: {snap_file.name}")

    try:
        # Load and parse
        if snap_file.suffix == ".gz":
            with gzip.open(snap_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(snap_file) as f:
                data = json.load(f)

        # Check required fields
        required = ["snapshot_id", "timestamp", "state"]
        missing = [r for r in required if r not in data]
        if missing:
            print(f"  ✗ Missing fields: {missing}")
            return 1

        # Check state structure
        state = data.get("state", {})
        nodes = state.get("nodes", {})
        edges = state.get("edges", {})

        print("  ✓ Valid JSON structure")
        print(f"  ✓ Snapshot ID: {data.get('snapshot_id', 'missing')}")
        print(f"  ✓ Timestamp: {data.get('timestamp', 'missing')}")
        print(f"  ✓ Nodes: {len(nodes)}")
        print(f"  ✓ Edges: {len(edges)}")

        # Verify node structure
        invalid_nodes = 0
        for node_id, node in nodes.items():
            if not isinstance(node, dict) or "node_type" not in node:
                invalid_nodes += 1
        if invalid_nodes:
            print(f"  ⚠ Invalid nodes: {invalid_nodes}")
        else:
            print("  ✓ All nodes valid")

        print("\nSnapshot verification: PASSED")
        return 0

    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1


def cmd_backup_restore(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got backup restore' command."""
    from cortical.reasoning.thought_graph import ThoughtGraph
    from cortical.reasoning.graph_of_thought import NodeType, EdgeType

    snapshot_id = args.snapshot_id
    force = getattr(args, 'force', False)

    snapshots_dir = manager.got_dir / "wal" / "snapshots"
    if not snapshots_dir.exists():
        print("No snapshots found.")
        return 1

    # Find the snapshot
    files = list(snapshots_dir.glob(f"*{snapshot_id}*"))
    if not files:
        print(f"Snapshot not found: {snapshot_id}")
        return 1

    snap_file = files[0]

    # Confirm unless forced
    if not force:
        print(f"About to restore from: {snap_file.name}")
        print("This will overwrite the current graph state.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Restore cancelled.")
            return 0

    try:
        # Load snapshot
        if snap_file.suffix == ".gz":
            with gzip.open(snap_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(snap_file) as f:
                data = json.load(f)

        state = data.get("state", {})
        nodes = state.get("nodes", {})

        # Rebuild graph
        manager.graph = ThoughtGraph()
        for node_id, node in nodes.items():
            manager.graph.add_node(
                node_id=node_id,
                node_type=NodeType[node.get("node_type", "TASK").upper()],
                content=node.get("content", ""),
                properties=node.get("properties", {}),
                metadata=node.get("metadata", {})
            )

        # Restore edges
        edges = state.get("edges", {})
        for edge_id, edge in edges.items():
            try:
                manager.graph.add_edge(
                    source_id=edge.get("source_id"),
                    target_id=edge.get("target_id"),
                    edge_type=EdgeType[edge.get("edge_type", "RELATES_TO").upper()],
                    weight=edge.get("weight", 1.0),
                    metadata=edge.get("metadata", {})
                )
            except Exception:
                pass  # Skip invalid edges

        # Save the restored state
        manager._save_state()

        print(f"Restored from: {snap_file.name}")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        return 0

    except Exception as e:
        print(f"Error restoring: {e}")
        return 1


def cmd_sync(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got sync' command.

    This is CRITICAL for environment resilience:
    - Ensures state survives fresh git clone
    - Enables cross-branch/cross-agent coordination
    - Should be run before committing
    """
    try:
        # Sync to git-tracked location
        snapshot_name = manager.sync_to_git()
        print(f"Synced to git-tracked snapshot: {snapshot_name}")

        # Show stats
        stats = manager.get_stats()
        print(f"  Tasks: {stats['total_tasks']}")
        print(f"  Sprints: {stats['total_sprints']}")

        # Auto-commit if message provided
        message = getattr(args, 'message', None)
        if message:
            snapshot_path = manager.snapshots_dir / snapshot_name
            try:
                subprocess.run(
                    ["git", "add", str(snapshot_path)],
                    check=True, capture_output=True
                )
                subprocess.run(
                    ["git", "commit", "-m", f"got: {message}"],
                    check=True, capture_output=True
                )
                print(f"  Committed: got: {message}")
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Git commit failed: {e}")

        print("\nTo persist across environments, commit .got/snapshots/")
        return 0

    except Exception as e:
        print(f"Error syncing: {e}")
        return 1


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_backup_parser(subparsers) -> None:
    """
    Set up argparse subparsers for backup commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Backup commands
    backup_parser = subparsers.add_parser("backup", help="Backup and recovery")
    backup_subparsers = backup_parser.add_subparsers(
        dest="backup_command",
        help="Backup subcommands"
    )

    # backup create
    backup_create = backup_subparsers.add_parser("create", help="Create a snapshot")
    backup_create.add_argument(
        "--compress", "-c",
        action="store_true",
        default=True,
        help="Compress snapshot (default: true)"
    )

    # backup list
    backup_list = backup_subparsers.add_parser("list", help="List available snapshots")
    backup_list.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number to show"
    )

    # backup verify
    backup_verify = backup_subparsers.add_parser("verify", help="Verify snapshot integrity")
    backup_verify.add_argument(
        "snapshot_id",
        nargs="?",
        help="Snapshot ID (default: latest)"
    )

    # backup restore
    backup_restore = backup_subparsers.add_parser("restore", help="Restore from snapshot")
    backup_restore.add_argument("snapshot_id", help="Snapshot ID to restore")
    backup_restore.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force restore without confirmation"
    )

    # Sync command (critical for environment resilience)
    sync_parser = subparsers.add_parser("sync", help="Sync state to git-tracked snapshot")
    sync_parser.add_argument(
        "--message", "-m",
        help="Commit message (auto-commits if provided)"
    )


def handle_backup_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route backup subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'backup_command') or args.backup_command is None:
        print("Error: No backup subcommand specified. Use 'got backup --help' for usage.")
        return 1

    command_handlers = {
        "create": cmd_backup_create,
        "list": cmd_backup_list,
        "verify": cmd_backup_verify,
        "restore": cmd_backup_restore,
    }

    handler = command_handlers.get(args.backup_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown backup subcommand: {args.backup_command}")
    return 1


def handle_sync_migrate_commands(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route sync/migrate commands to appropriate handlers.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error), or None if not handled
    """
    command = args.command

    handlers = {
        "sync": cmd_sync,
    }

    handler = handlers.get(command)
    if handler:
        return handler(args, manager)

    return None  # Not handled by this module
