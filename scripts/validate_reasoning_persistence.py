#!/usr/bin/env python3
"""
Validation script for reasoning framework and graph persistence integration.

Tests that ThoughtGraph and GraphWAL work correctly together:
1. Create a ThoughtGraph with nodes and edges
2. Log operations to GraphWAL
3. Create a snapshot
4. Load the snapshot and verify graph is identical
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_persistence import GraphWAL
from cortical.reasoning.graph_of_thought import NodeType, EdgeType


def verify_graphs_equal(graph1: ThoughtGraph, graph2: ThoughtGraph) -> tuple[bool, list[str]]:
    """
    Verify that two graphs have identical structure and content.

    Returns:
        Tuple of (is_equal, differences)
    """
    differences = []

    # Check node counts
    if graph1.node_count() != graph2.node_count():
        differences.append(
            f"Node count mismatch: {graph1.node_count()} vs {graph2.node_count()}"
        )

    # Check edge counts
    if graph1.edge_count() != graph2.edge_count():
        differences.append(
            f"Edge count mismatch: {graph1.edge_count()} vs {graph2.edge_count()}"
        )

    # Check cluster counts
    if graph1.cluster_count() != graph2.cluster_count():
        differences.append(
            f"Cluster count mismatch: {graph1.cluster_count()} vs {graph2.cluster_count()}"
        )

    # Check individual nodes
    for node_id in graph1.nodes:
        if node_id not in graph2.nodes:
            differences.append(f"Node {node_id} missing in graph2")
            continue

        node1 = graph1.nodes[node_id]
        node2 = graph2.nodes[node_id]

        if node1.node_type != node2.node_type:
            differences.append(
                f"Node {node_id} type mismatch: {node1.node_type} vs {node2.node_type}"
            )

        if node1.content != node2.content:
            differences.append(
                f"Node {node_id} content mismatch: '{node1.content}' vs '{node2.content}'"
            )

    # Check for nodes in graph2 not in graph1
    for node_id in graph2.nodes:
        if node_id not in graph1.nodes:
            differences.append(f"Extra node {node_id} in graph2")

    # Check edges (convert to comparable tuples)
    edges1 = {
        (e.source_id, e.target_id, e.edge_type.value, e.weight)
        for e in graph1.edges
    }
    edges2 = {
        (e.source_id, e.target_id, e.edge_type.value, e.weight)
        for e in graph2.edges
    }

    missing_edges = edges1 - edges2
    extra_edges = edges2 - edges1

    if missing_edges:
        for edge in missing_edges:
            differences.append(f"Missing edge: {edge}")

    if extra_edges:
        for edge in extra_edges:
            differences.append(f"Extra edge: {edge}")

    return len(differences) == 0, differences


def test_basic_persistence():
    """Test basic GraphWAL persistence with a simple graph."""
    print("=" * 70)
    print("TEST 1: Basic Persistence")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "test_wal"

        # Create a simple graph
        print("\n1. Creating ThoughtGraph with nodes and edges...")
        graph = ThoughtGraph()

        graph.add_node("Q1", NodeType.QUESTION, "What is the best approach?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Try approach A")
        graph.add_node("H2", NodeType.HYPOTHESIS, "Try approach B")
        graph.add_node("E1", NodeType.EVIDENCE, "Team has experience with A")

        graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)
        graph.add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.6)
        graph.add_edge("E1", "H1", EdgeType.SUPPORTS, weight=0.9)

        print(f"   Created graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

        # Create GraphWAL and log operations
        print("\n2. Creating GraphWAL and logging operations...")
        graph_wal = GraphWAL(str(wal_dir))

        # Log all nodes
        for node_id, node in graph.nodes.items():
            graph_wal.log_add_node(
                node_id,
                node.node_type,
                node.content,
                node.properties,
                node.metadata,
            )

        # Log all edges
        for edge in graph.edges:
            graph_wal.log_add_edge(
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                edge.weight,
                edge.confidence,
                edge.bidirectional,
            )

        entry_count = graph_wal.get_entry_count()
        print(f"   Logged {entry_count} WAL entries")

        # Create a snapshot
        print("\n3. Creating snapshot...")
        snapshot_id = graph_wal.create_snapshot(graph, compress=False)
        print(f"   Created snapshot: {snapshot_id}")

        # Load the snapshot
        print("\n4. Loading snapshot...")
        loaded_graph = graph_wal.load_snapshot(snapshot_id)

        if loaded_graph is None:
            print("   ❌ FAILED: Could not load snapshot")
            return False

        print(f"   Loaded graph: {loaded_graph.node_count()} nodes, {loaded_graph.edge_count()} edges")

        # Verify graphs are identical
        print("\n5. Verifying graphs are identical...")
        is_equal, differences = verify_graphs_equal(graph, loaded_graph)

        if is_equal:
            print("   ✅ PASSED: Graphs are identical")
            return True
        else:
            print("   ❌ FAILED: Graphs differ:")
            for diff in differences:
                print(f"      - {diff}")
            return False


def test_wal_replay():
    """Test WAL replay from scratch."""
    print("\n" + "=" * 70)
    print("TEST 2: WAL Replay (Recovery)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "test_wal"

        # Create and populate a graph via WAL
        print("\n1. Creating graph via WAL operations...")
        graph_wal = GraphWAL(str(wal_dir))

        graph_wal.log_add_node("Q1", NodeType.QUESTION, "How to implement feature X?")
        graph_wal.log_add_node("D1", NodeType.DECISION, "Use microservices architecture")
        graph_wal.log_add_node("T1", NodeType.TASK, "Design API endpoints")

        graph_wal.log_add_edge("Q1", "D1", EdgeType.ANSWERS, weight=1.0)
        graph_wal.log_add_edge("D1", "T1", EdgeType.REQUIRES, weight=1.0)

        print(f"   Logged {graph_wal.get_entry_count()} operations")

        # Replay WAL to reconstruct graph
        print("\n2. Replaying WAL to reconstruct graph...")
        replayed_graph = ThoughtGraph()

        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, replayed_graph)

        print(f"   Replayed graph: {replayed_graph.node_count()} nodes, {replayed_graph.edge_count()} edges")

        # Verify structure
        print("\n3. Verifying graph structure...")

        if replayed_graph.node_count() != 3:
            print(f"   ❌ FAILED: Expected 3 nodes, got {replayed_graph.node_count()}")
            return False

        if replayed_graph.edge_count() != 2:
            print(f"   ❌ FAILED: Expected 2 edges, got {replayed_graph.edge_count()}")
            return False

        if "Q1" not in replayed_graph.nodes:
            print("   ❌ FAILED: Missing node Q1")
            return False

        if "D1" not in replayed_graph.nodes:
            print("   ❌ FAILED: Missing node D1")
            return False

        if "T1" not in replayed_graph.nodes:
            print("   ❌ FAILED: Missing node T1")
            return False

        # Verify edge relationships
        q1_edges = replayed_graph.get_edges_from("Q1")
        if len(q1_edges) != 1 or q1_edges[0].target_id != "D1":
            print("   ❌ FAILED: Incorrect edges from Q1")
            return False

        print("   ✅ PASSED: Graph structure is correct")
        return True


def test_incremental_updates():
    """Test incremental graph updates with WAL."""
    print("\n" + "=" * 70)
    print("TEST 3: Incremental Updates")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "test_wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Create initial graph
        print("\n1. Creating initial graph...")
        graph = ThoughtGraph()
        graph.add_node("N1", NodeType.CONCEPT, "Concept 1")
        graph.add_node("N2", NodeType.CONCEPT, "Concept 2")

        # Log initial state
        graph_wal.log_add_node("N1", NodeType.CONCEPT, "Concept 1")
        graph_wal.log_add_node("N2", NodeType.CONCEPT, "Concept 2")

        # Create snapshot
        snap1 = graph_wal.create_snapshot(graph, compress=False)
        print(f"   Created snapshot: {snap1}")

        # Make incremental updates
        print("\n2. Making incremental updates...")
        graph.add_node("N3", NodeType.CONCEPT, "Concept 3")
        graph.add_edge("N1", "N3", EdgeType.SIMILAR)

        graph_wal.log_add_node("N3", NodeType.CONCEPT, "Concept 3")
        graph_wal.log_add_edge("N1", "N3", EdgeType.SIMILAR)

        # Create second snapshot
        snap2 = graph_wal.create_snapshot(graph, compress=False)
        print(f"   Created snapshot: {snap2}")

        # Load latest snapshot
        print("\n3. Loading latest snapshot...")
        loaded = graph_wal.load_snapshot()  # Should load snap2

        if loaded is None:
            print("   ❌ FAILED: Could not load snapshot")
            return False

        # Verify it has all updates
        print("\n4. Verifying updates are present...")
        if loaded.node_count() != 3:
            print(f"   ❌ FAILED: Expected 3 nodes, got {loaded.node_count()}")
            return False

        if "N3" not in loaded.nodes:
            print("   ❌ FAILED: Missing node N3")
            return False

        n1_edges = loaded.get_edges_from("N1")
        if len(n1_edges) != 1:
            print(f"   ❌ FAILED: Expected 1 edge from N1, got {len(n1_edges)}")
            return False

        print("   ✅ PASSED: Incremental updates preserved correctly")
        return True


def test_api_compatibility():
    """Test that ThoughtGraph and GraphWAL APIs are compatible."""
    print("\n" + "=" * 70)
    print("TEST 4: API Compatibility")
    print("=" * 70)

    issues = []

    # Check that ThoughtGraph has expected methods
    print("\n1. Checking ThoughtGraph API...")
    graph = ThoughtGraph()

    required_methods = [
        'add_node', 'remove_node', 'add_edge', 'remove_edge',
        'add_cluster', 'nodes', 'edges', 'clusters',
        'node_count', 'edge_count', 'get_edges_from', 'get_edges_to'
    ]

    for method_name in required_methods:
        if not hasattr(graph, method_name):
            issues.append(f"ThoughtGraph missing method: {method_name}")

    # Check that GraphWAL has expected methods
    print("2. Checking GraphWAL API...")
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_wal = GraphWAL(str(tmpdir))

        required_wal_methods = [
            'log_add_node', 'log_remove_node', 'log_add_edge', 'log_remove_edge',
            'log_add_cluster', 'apply_entry', 'get_all_entries',
            'create_snapshot', 'load_snapshot'
        ]

        for method_name in required_wal_methods:
            if not hasattr(graph_wal, method_name):
                issues.append(f"GraphWAL missing method: {method_name}")

    # Check parameter compatibility
    print("3. Checking parameter compatibility...")

    # ThoughtGraph.add_edge signature: (from_id, to_id, edge_type, ...)
    # GraphWAL.log_add_edge signature: (source_id, target_id, edge_type, ...)
    # GraphWAL.apply_entry must map source_id -> from_id, target_id -> to_id

    # This is tested implicitly in other tests, but we can add a specific check
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_wal = GraphWAL(str(tmpdir))
        test_graph = ThoughtGraph()

        # Add nodes via graph
        test_graph.add_node("A", NodeType.CONCEPT, "A")
        test_graph.add_node("B", NodeType.CONCEPT, "B")

        # Log edge via WAL
        graph_wal.log_add_edge("A", "B", EdgeType.SIMILAR)

        # Apply to graph
        entries = list(graph_wal.get_all_entries())
        if len(entries) != 1:
            issues.append(f"Expected 1 WAL entry, got {len(entries)}")
        else:
            try:
                graph_wal.apply_entry(entries[0], test_graph)

                # Check edge was added
                edges = test_graph.get_edges_from("A")
                if len(edges) != 1:
                    issues.append(f"Expected 1 edge from A, got {len(edges)}")
                elif edges[0].target_id != "B":
                    issues.append(f"Expected edge to B, got {edges[0].target_id}")
            except Exception as e:
                issues.append(f"apply_entry failed: {e}")

    if issues:
        print("\n   ❌ FAILED: API compatibility issues:")
        for issue in issues:
            print(f"      - {issue}")
        return False
    else:
        print("   ✅ PASSED: APIs are compatible")
        return True


def main():
    """Run all validation tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "REASONING PERSISTENCE VALIDATION" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        ("Basic Persistence", test_basic_persistence),
        ("WAL Replay", test_wal_replay),
        ("Incremental Updates", test_incremental_updates),
        ("API Compatibility", test_api_compatibility),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n   ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    for test_name, passed, error in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error}")

    print()
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! ThoughtGraph and GraphWAL are compatible.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
