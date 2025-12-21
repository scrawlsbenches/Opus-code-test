#!/usr/bin/env python3
"""
Graph Persistence Demo

Demonstrates the graph persistence layer for ThoughtGraph:
- GraphWAL: Write-Ahead Logging for crash recovery
- GitAutoCommitter: Automatic git commits on save
- GraphRecovery: Multi-level cascading recovery
- Snapshots: Periodic state checkpointing

Usage:
    python examples/graph_persistence_demo.py
"""

import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from cortical.reasoning import (
    ThoughtGraph,
    NodeType,
    EdgeType,
    GraphWAL,
    GraphWALEntry,
    GitAutoCommitter,
    GraphRecovery,
    GraphRecoveryResult,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(step: str) -> None:
    """Print a step indicator."""
    print(f"  → {step}")


def demo_wal_logging():
    """Demonstrate Write-Ahead Logging for graph operations."""
    print_header("1. WRITE-AHEAD LOGGING (WAL)")

    # Create temporary directory for WAL
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "reasoning_wal"

        # Initialize GraphWAL
        print_step("Initializing GraphWAL")
        graph_wal = GraphWAL(str(wal_dir))
        print(f"  WAL directory: {wal_dir}")

        # Create initial graph
        print_step("Creating ThoughtGraph with reasoning nodes")
        graph = ThoughtGraph()

        # Add nodes with WAL logging
        print_step("Adding nodes (with WAL logging)")

        # Question node
        graph_wal.log_add_node(
            "Q1", NodeType.QUESTION,
            "What is the best authentication approach?",
            properties={'priority': 'high', 'urgency': 'immediate'},
            metadata={'created_by': 'agent', 'timestamp': '2025-01-15T10:00:00'}
        )
        graph.add_node("Q1", NodeType.QUESTION, "What is the best authentication approach?")
        print("  Added Q1: QUESTION node")

        # Hypothesis nodes
        graph_wal.log_add_node(
            "H1", NodeType.HYPOTHESIS,
            "Use JWT for stateless authentication",
            properties={'confidence': 0.8}
        )
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT for stateless authentication")
        print("  Added H1: HYPOTHESIS node")

        graph_wal.log_add_node(
            "H2", NodeType.HYPOTHESIS,
            "Use session-based authentication",
            properties={'confidence': 0.6}
        )
        graph.add_node("H2", NodeType.HYPOTHESIS, "Use session-based authentication")
        print("  Added H2: HYPOTHESIS node")

        # Add edges with WAL logging
        print_step("Adding edges (with WAL logging)")

        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9, confidence=0.8)
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9, confidence=0.8)
        print("  Added edge: Q1 --EXPLORES--> H1")

        graph_wal.log_add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.7, confidence=0.6)
        graph.add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.7, confidence=0.6)
        print("  Added edge: Q1 --EXPLORES--> H2")

        # Check WAL entry count
        entry_count = graph_wal.get_entry_count()
        print(f"\n  Total WAL entries: {entry_count}")
        print(f"  Current WAL path: {graph_wal.get_current_wal_path()}")


def demo_snapshot_creation():
    """Demonstrate snapshot creation and loading."""
    print_header("2. SNAPSHOT CREATION & LOADING")

    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "reasoning_wal"

        # Create graph with WAL
        print_step("Creating graph with multiple nodes")
        graph_wal = GraphWAL(str(wal_dir))
        graph = ThoughtGraph()

        # Add reasoning graph
        for i in range(5):
            node_id = f"N{i}"
            graph_wal.log_add_node(
                node_id, NodeType.CONCEPT,
                f"Concept {i}: Important reasoning point",
                properties={'iteration': i}
            )
            graph.add_node(node_id, NodeType.CONCEPT, f"Concept {i}: Important reasoning point")

        # Add some edges
        graph_wal.log_add_edge("N0", "N1", EdgeType.SUPPORTS)
        graph.add_edge("N0", "N1", EdgeType.SUPPORTS)

        graph_wal.log_add_edge("N1", "N2", EdgeType.CONTRADICTS)
        graph.add_edge("N1", "N2", EdgeType.CONTRADICTS)

        print(f"  Created graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

        # Create snapshot
        print_step("Creating snapshot")
        snapshot_id = graph_wal.create_snapshot(graph, compress=True)
        print(f"  Snapshot ID: {snapshot_id}")

        # Load snapshot back
        print_step("Loading snapshot")
        loaded_graph = graph_wal.load_snapshot(snapshot_id)

        if loaded_graph:
            print(f"  Loaded: {loaded_graph.node_count()} nodes, {loaded_graph.edge_count()} edges")
            print("  ✓ Snapshot loaded successfully!")
        else:
            print("  ✗ Failed to load snapshot")


def demo_crash_recovery():
    """Demonstrate crash recovery using WAL replay."""
    print_header("3. CRASH RECOVERY SIMULATION")

    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "reasoning_wal"

        # Phase 1: Normal operation
        print_step("Phase 1: Normal operation - building graph")
        graph_wal = GraphWAL(str(wal_dir))
        graph = ThoughtGraph()

        # Add initial state
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Main question")
        graph.add_node("Q1", NodeType.QUESTION, "Main question")

        # Create snapshot
        snapshot_id = graph_wal.create_snapshot(graph)
        print(f"  Created snapshot: {snapshot_id}")

        # Continue adding nodes (these will be in WAL only)
        print_step("Adding more nodes after snapshot...")
        for i in range(3):
            node_id = f"A{i}"
            graph_wal.log_add_node(node_id, NodeType.ACTION, f"Action {i}")
            graph.add_node(node_id, NodeType.ACTION, f"Action {i}")

        print(f"  Graph before crash: {graph.node_count()} nodes")

        # Phase 2: Simulate crash - discard in-memory graph
        print_step("Phase 2: CRASH! In-memory graph lost...")
        graph = None  # Simulate crash

        # Phase 3: Recovery
        print_step("Phase 3: Recovering from WAL")
        recovery = GraphRecovery(str(wal_dir))

        if recovery.needs_recovery():
            print("  Recovery needed - attempting...")
            result = recovery.recover()

            if result.success:
                print(f"  ✓ Recovery successful!")
                print(f"    Level used: {result.level_used} ({result.recovery_method})")
                print(f"    Nodes recovered: {result.nodes_recovered}")
                print(f"    Edges recovered: {result.edges_recovered}")
                print(f"    Duration: {result.duration_ms:.2f}ms")

                if result.graph:
                    print(f"\n  Recovered graph: {result.graph.node_count()} nodes")
            else:
                print(f"  ✗ Recovery failed: {result.errors}")


def demo_git_auto_commit():
    """Demonstrate automatic git commits."""
    print_header("4. GIT AUTO-COMMIT")

    print_step("Creating GitAutoCommitter in manual mode")
    committer = GitAutoCommitter(mode='manual', auto_push=False)

    print(f"  Mode: {committer.mode}")
    print(f"  Auto-push: {committer.auto_push}")
    print(f"  Protected branches: {committer.protected_branches}")

    # Check current branch
    branch = committer.get_current_branch()
    if branch:
        print(f"  Current branch: {branch}")
        is_protected = committer.is_protected_branch(branch)
        print(f"  Is protected: {is_protected}")
    else:
        print("  Not in a git repository or detached HEAD")

    # In real usage with 'immediate' or 'debounced' mode:
    print_step("Usage pattern (manual mode - no actual commit)")
    print("  committer = GitAutoCommitter(mode='debounced', debounce_seconds=5)")
    print("  committer.commit_on_save('/path/to/graph.json', graph=graph)")
    print("  # Waits 5 seconds, then commits if no more saves")

    print_step("Safety features")
    print("  ✓ Never force pushes")
    print("  ✓ Protected branch detection (main/master)")
    print("  ✓ Pre-commit graph validation")
    print("  ✓ Backup branch creation for risky operations")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  GRAPH PERSISTENCE DEMO")
    print("  Demonstrating WAL, Snapshots, Recovery, and Git Integration")
    print("="*70)

    try:
        # Run demos
        demo_wal_logging()
        time.sleep(1)

        demo_snapshot_creation()
        time.sleep(1)

        demo_crash_recovery()
        time.sleep(1)

        demo_git_auto_commit()

        # Summary
        print_header("SUMMARY")
        print("✓ GraphWAL: Write-ahead logging for durability")
        print("✓ Snapshots: Fast recovery checkpoints")
        print("✓ Recovery: Multi-level cascading recovery (WAL → Snapshot → Git → Chunks)")
        print("✓ GitAutoCommitter: Automatic version control integration")

        print("\nFor production usage:")
        print("  1. Initialize GraphWAL with your WAL directory")
        print("  2. Log all graph operations (add_node, add_edge, etc.)")
        print("  3. Create snapshots periodically (every N operations)")
        print("  4. Use GraphRecovery.recover() to restore after crashes")
        print("  5. Optionally use GitAutoCommitter for automatic commits")

        print("\n" + "="*70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
