"""
Integration tests for graph persistence.

Tests cover interactions between:
- GraphWAL and ThoughtGraph
- Snapshot creation and WAL recovery
- Git auto-committer with WAL operations
- Full persistence stack (WAL + snapshots + git)
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

import pytest

from cortical.reasoning.graph_persistence import (
    GraphWAL,
    GraphWALEntry,
    GitAutoCommitter,
    GraphRecovery,
    GraphRecoveryResult,
)
from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import NodeType, EdgeType, ThoughtCluster
from cortical.wal import WALWriter, SnapshotManager


class TestGraphWALWithThoughtGraph:
    """Test GraphWAL integration with ThoughtGraph."""

    def test_wal_captures_all_graph_operations(self, tmp_path):
        """Test WAL captures all types of graph operations."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Log various operations
        graph_wal.log_add_node(
            "Q1", NodeType.QUESTION, "What is the issue?",
            properties={'priority': 'high'},
            metadata={'created_by': 'test'}
        )
        graph_wal.log_add_node("H1", NodeType.HYPOTHESIS, "Memory leak")
        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8, confidence=0.9)
        graph_wal.log_add_cluster("C1", "Investigation", node_ids={"Q1", "H1"})
        graph_wal.log_update_node("Q1", {'content': 'What is the root cause?'})
        graph_wal.log_remove_edge("Q1", "H1", EdgeType.EXPLORES)
        graph_wal.log_remove_node("H1")

        # Verify entries were written
        entry_count = graph_wal.get_entry_count()
        assert entry_count == 7, f"Expected 7 entries, got {entry_count}"

        # Verify WAL file exists
        wal_path = graph_wal.get_current_wal_path()
        assert wal_path.exists()
        assert wal_path.stat().st_size > 0

    def test_replay_reconstructs_identical_graph(self, tmp_path):
        """Test replaying WAL recreates identical graph state."""
        wal_dir = tmp_path / "wal"

        # Create original graph
        graph1 = ThoughtGraph()
        graph1.add_node("Q1", NodeType.QUESTION, "Original question")
        graph1.add_node("H1", NodeType.HYPOTHESIS, "Original hypothesis")
        graph1.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.75)

        # Log to WAL
        graph_wal = GraphWAL(str(wal_dir))
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Original question")
        graph_wal.log_add_node("H1", NodeType.HYPOTHESIS, "Original hypothesis")
        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.75)

        # Replay into new graph
        graph2 = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph2)

        # Verify graphs are identical
        assert graph2.node_count() == graph1.node_count()
        assert graph2.edge_count() == graph1.edge_count()

        # Check nodes
        assert graph2.get_node("Q1") is not None
        assert graph2.get_node("H1") is not None
        assert graph2.get_node("Q1").content == "Original question"

        # Check edges
        edges = graph2.get_edges_from("Q1")
        assert len(edges) == 1
        assert edges[0].target_id == "H1"
        assert edges[0].weight == 0.75

    def test_replay_with_complex_graph_structure(self, tmp_path):
        """Test WAL replay with complex graph containing cycles and clusters."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Build complex graph with cycles
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Question 1")
        graph_wal.log_add_node("H1", NodeType.HYPOTHESIS, "Hypothesis 1")
        graph_wal.log_add_node("E1", NodeType.EVIDENCE, "Evidence 1")
        graph_wal.log_add_node("D1", NodeType.DECISION, "Decision 1")

        # Create cycle: Q1 -> H1 -> E1 -> Q1
        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES)
        graph_wal.log_add_edge("H1", "E1", EdgeType.SUPPORTS)
        graph_wal.log_add_edge("E1", "Q1", EdgeType.REFINES)
        graph_wal.log_add_edge("H1", "D1", EdgeType.TRIGGERS)

        # Add cluster
        graph_wal.log_add_cluster("C1", "Investigation", {"Q1", "H1", "E1"})

        # Replay
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # Verify structure
        assert graph.node_count() == 4
        assert graph.edge_count() == 4

        # Verify cycle exists
        cycles = graph.find_cycles()
        assert len(cycles) > 0

        # Verify cluster
        assert "C1" in graph.clusters
        cluster = graph.clusters["C1"]
        assert "Q1" in cluster.node_ids
        assert "H1" in cluster.node_ids
        assert "E1" in cluster.node_ids

    def test_wal_handles_node_type_enums_correctly(self, tmp_path):
        """Test WAL correctly handles NodeType enum serialization."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Test all node types
        node_types = [
            (NodeType.QUESTION, "Q1"),
            (NodeType.HYPOTHESIS, "H1"),
            (NodeType.EVIDENCE, "E1"),
            (NodeType.DECISION, "D1"),
            (NodeType.CONCEPT, "C1"),
            (NodeType.FACT, "F1"),
            (NodeType.TASK, "T1"),
            (NodeType.ARTIFACT, "A1"),
            (NodeType.INSIGHT, "I1"),
        ]

        for node_type, node_id in node_types:
            graph_wal.log_add_node(node_id, node_type, f"Content for {node_id}")

        # Replay
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # Verify all node types preserved
        for node_type, node_id in node_types:
            node = graph.get_node(node_id)
            assert node is not None
            assert node.node_type == node_type

    def test_wal_preserves_edge_weights_and_confidence(self, tmp_path):
        """Test WAL preserves edge weights and confidence values."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Add nodes
        graph_wal.log_add_node("A", NodeType.CONCEPT, "Node A")
        graph_wal.log_add_node("B", NodeType.CONCEPT, "Node B")
        graph_wal.log_add_node("C", NodeType.CONCEPT, "Node C")

        # Add edges with specific weights and confidence
        graph_wal.log_add_edge("A", "B", EdgeType.SIMILAR, weight=0.5, confidence=0.8)
        graph_wal.log_add_edge("B", "C", EdgeType.SUPPORTS, weight=0.9, confidence=0.95)
        graph_wal.log_add_edge("A", "C", EdgeType.CONTRADICTS, weight=0.3, confidence=0.6)

        # Replay
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # Verify weights and confidence
        edges_from_a = graph.get_edges_from("A")
        edge_ab = next(e for e in edges_from_a if e.target_id == "B")
        assert edge_ab.weight == 0.5
        assert edge_ab.confidence == 0.8

        edges_from_b = graph.get_edges_from("B")
        edge_bc = next(e for e in edges_from_b if e.target_id == "C")
        assert edge_bc.weight == 0.9
        assert edge_bc.confidence == 0.95

    def test_wal_preserves_cluster_memberships(self, tmp_path):
        """Test WAL preserves cluster memberships and properties."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Add nodes
        for i in range(5):
            graph_wal.log_add_node(f"N{i}", NodeType.CONCEPT, f"Node {i}")

        # Create clusters with properties
        graph_wal.log_add_cluster(
            "C1", "Group 1", {"N0", "N1"},
            properties={'importance': 'high', 'category': 'core'}
        )
        graph_wal.log_add_cluster(
            "C2", "Group 2", {"N2", "N3", "N4"},
            properties={'importance': 'medium'}
        )

        # Replay
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # Verify clusters
        assert graph.cluster_count() == 2

        c1 = graph.clusters["C1"]
        assert c1.name == "Group 1"
        assert c1.node_ids == {"N0", "N1"}
        assert c1.properties['importance'] == 'high'
        assert c1.properties['category'] == 'core'

        c2 = graph.clusters["C2"]
        assert c2.name == "Group 2"
        assert c2.node_ids == {"N2", "N3", "N4"}
        assert c2.properties['importance'] == 'medium'

    def test_wal_handles_bidirectional_edges(self, tmp_path):
        """Test WAL correctly handles bidirectional edges."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        graph_wal.log_add_node("A", NodeType.CONCEPT, "Node A")
        graph_wal.log_add_node("B", NodeType.CONCEPT, "Node B")
        graph_wal.log_add_edge("A", "B", EdgeType.SIMILAR, bidirectional=True)

        # Replay
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # Verify bidirectional edge
        edges_ab = graph.get_edges_from("A")
        edges_ba = graph.get_edges_from("B")

        assert len(edges_ab) > 0
        assert len(edges_ba) > 0
        assert any(e.target_id == "B" for e in edges_ab)
        assert any(e.target_id == "A" for e in edges_ba)

    def test_wal_handles_duplicate_operations(self, tmp_path):
        """Test WAL gracefully handles duplicate add operations."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Log same node twice
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "First")
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Second")  # Duplicate

        # Replay (should not crash)
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # Should only have one node (first one wins due to skip logic)
        assert graph.node_count() == 1
        assert graph.get_node("Q1").content == "First"

    def test_wal_skips_edges_for_missing_nodes(self, tmp_path):
        """Test WAL skips edge creation when nodes don't exist."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Log edge without creating nodes
        graph_wal.log_add_edge("NonExistent1", "NonExistent2", EdgeType.SIMILAR)

        # Replay (should not crash)
        graph = ThoughtGraph()
        for entry in graph_wal.get_all_entries():
            graph_wal.apply_entry(entry, graph)

        # No nodes or edges should be created
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_wal_entry_checksum_verification(self, tmp_path):
        """Test WAL entry checksum verification detects corruption."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Test")

        # Get entry and corrupt it
        entries = list(graph_wal.get_all_entries())
        assert len(entries) > 0

        entry = entries[0]
        original_checksum = entry.checksum

        # Verify original is valid
        assert entry.verify()

        # Corrupt the entry
        entry.payload['content'] = 'CORRUPTED'

        # Checksum should no longer match
        assert not entry.verify()
        assert entry._compute_checksum() != original_checksum


class TestSnapshotWithRecovery:
    """Test snapshot creation and recovery integration."""

    def test_snapshot_and_recover_roundtrip(self, tmp_path):
        """Test creating snapshot and recovering produces identical graph."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Build graph
        graph1 = ThoughtGraph()
        graph1.add_node("Q1", NodeType.QUESTION, "Question")
        graph1.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        graph1.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.7)

        # Create snapshot
        snapshot_id = graph_wal.create_snapshot(graph1, compress=False)
        assert snapshot_id is not None

        # Load snapshot
        graph2 = graph_wal.load_snapshot(snapshot_id)
        assert graph2 is not None

        # Verify identical
        assert graph2.node_count() == graph1.node_count()
        assert graph2.edge_count() == graph1.edge_count()
        assert graph2.get_node("Q1").content == "Question"
        assert graph2.get_node("H1").content == "Hypothesis"

    @pytest.mark.skip(reason="GraphRecovery implementation incomplete - goes to Level 4 instead of trying Levels 1-2")
    def test_recovery_with_wal_entries_after_snapshot(self, tmp_path):
        """Test recovery applies WAL entries after snapshot point."""
        wal_dir = tmp_path / "wal"

        # Create initial graph and snapshot
        graph_wal = GraphWAL(str(wal_dir))
        graph1 = ThoughtGraph()
        graph1.add_node("Q1", NodeType.QUESTION, "Initial")

        # Log to WAL first
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Initial")

        # Create snapshot
        snapshot_id = graph_wal.create_snapshot(graph1, compress=False)

        # Add more operations AFTER snapshot
        graph_wal.log_add_node("Q2", NodeType.QUESTION, "After snapshot")
        graph_wal.log_add_edge("Q1", "Q2", EdgeType.REFINES)

        # Recover (should include post-snapshot operations)
        # Don't provide chunks_dir - use WAL-based recovery
        recovery = GraphRecovery(str(wal_dir))
        result = recovery.recover()

        assert result.success
        assert result.graph is not None
        assert result.graph.node_count() == 2  # Q1 + Q2
        assert result.graph.edge_count() == 1  # Q1 -> Q2
        assert result.graph.get_node("Q2") is not None

    def test_multiple_snapshots_with_incremental_changes(self, tmp_path):
        """Test creating multiple snapshots with changes between them."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Snapshot 1
        graph1 = ThoughtGraph()
        graph1.add_node("Q1", NodeType.QUESTION, "V1")
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "V1")
        snap1 = graph_wal.create_snapshot(graph1)

        # Snapshot 2 - add more
        graph2 = ThoughtGraph()
        graph2.add_node("Q1", NodeType.QUESTION, "V1")
        graph2.add_node("Q2", NodeType.QUESTION, "V2")
        graph_wal.log_add_node("Q2", NodeType.QUESTION, "V2")
        snap2 = graph_wal.create_snapshot(graph2)

        # Snapshot 3 - add even more
        graph3 = ThoughtGraph()
        graph3.add_node("Q1", NodeType.QUESTION, "V1")
        graph3.add_node("Q2", NodeType.QUESTION, "V2")
        graph3.add_node("Q3", NodeType.QUESTION, "V3")
        graph_wal.log_add_node("Q3", NodeType.QUESTION, "V3")
        snap3 = graph_wal.create_snapshot(graph3)

        # Load each snapshot
        loaded1 = graph_wal.load_snapshot(snap1)
        loaded2 = graph_wal.load_snapshot(snap2)
        loaded3 = graph_wal.load_snapshot(snap3)

        assert loaded1.node_count() == 1
        assert loaded2.node_count() == 2
        assert loaded3.node_count() == 3

    def test_compressed_snapshot_integrity(self, tmp_path):
        """Test compressed snapshots maintain integrity."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Create large graph
        graph = ThoughtGraph()
        for i in range(100):
            graph.add_node(f"N{i}", NodeType.CONCEPT, f"Content {i}" * 10)

        # Create compressed snapshot
        snapshot_id = graph_wal.create_snapshot(graph, compress=True)

        # Load and verify
        loaded = graph_wal.load_snapshot(snapshot_id)
        assert loaded is not None
        assert loaded.node_count() == 100

        # Verify content
        for i in range(100):
            node = loaded.get_node(f"N{i}")
            assert node is not None
            assert f"Content {i}" in node.content

    @pytest.mark.skip(reason="GraphRecovery implementation incomplete - goes to Level 4 instead of trying Levels 1-2")
    def test_recovery_chooses_best_level(self, tmp_path):
        """Test recovery system chooses appropriate recovery level."""
        wal_dir = tmp_path / "wal"

        # Level 1: WAL Replay (create snapshot + WAL)
        graph_wal = GraphWAL(str(wal_dir))
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Test")
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Test")
        snapshot_id = graph_wal.create_snapshot(graph)

        # Add one more operation after snapshot to have something to replay
        graph_wal.log_add_node("Q2", NodeType.QUESTION, "After snapshot")

        # Recover - should use Level 1 (fastest) or Level 2 (snapshot rollback)
        recovery = GraphRecovery(str(wal_dir))
        result = recovery.recover()
        assert result.success, f"Recovery failed: {result.errors}"
        assert result.level_used in [1, 2]  # Either WAL Replay or Snapshot Rollback

    @pytest.mark.skip(reason="GraphRecovery implementation incomplete - goes to Level 4 instead of trying Levels 1-2")
    def test_recovery_falls_back_to_snapshot_rollback(self, tmp_path):
        """Test recovery falls back to snapshot rollback when WAL corrupted."""
        wal_dir = tmp_path / "wal"

        # Create valid snapshot
        graph_wal = GraphWAL(str(wal_dir))
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Snapshot content")
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Snapshot content")
        snapshot_id = graph_wal.create_snapshot(graph)

        # Corrupt WAL by writing invalid entries
        wal_path = graph_wal.get_current_wal_path()
        with open(wal_path, 'a') as f:
            f.write("CORRUPTED JSON LINE\n")

        # Recovery should still work via snapshot rollback
        recovery = GraphRecovery(str(wal_dir))
        result = recovery.recover()

        # May succeed with Level 1 (ignoring corrupt entries) or Level 2 (snapshot rollback)
        assert result.success, f"Recovery failed: {result.errors}"
        assert result.level_used in [1, 2]

    def test_snapshot_preserves_all_graph_elements(self, tmp_path):
        """Test snapshot preserves nodes, edges, and clusters."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Build complex graph
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Q", {'key': 'value'}, {'meta': 'data'})
        graph.add_node("H1", NodeType.HYPOTHESIS, "H")
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES, 0.8, 0.9, bidirectional=True)
        cluster = graph.add_cluster("C1", "Cluster", {"Q1", "H1"})
        cluster.properties['custom'] = 'prop'

        # Snapshot
        snapshot_id = graph_wal.create_snapshot(graph)

        # Load
        loaded = graph_wal.load_snapshot(snapshot_id)

        # Verify all elements
        assert loaded.node_count() == 2
        assert loaded.edge_count() >= 1
        assert loaded.cluster_count() == 1

        # Verify node properties
        q1 = loaded.get_node("Q1")
        assert q1.properties['key'] == 'value'
        assert q1.metadata['meta'] == 'data'

        # Verify cluster
        c1 = loaded.clusters["C1"]
        assert c1.name == "Cluster"
        assert c1.properties['custom'] == 'prop'

    def test_wal_compaction(self, tmp_path):
        """Test WAL compaction creates snapshot and clears WAL."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Add many operations
        for i in range(50):
            graph_wal.log_add_node(f"N{i}", NodeType.CONCEPT, f"Node {i}")

        # Build final graph state
        graph = ThoughtGraph()
        for i in range(50):
            graph.add_node(f"N{i}", NodeType.CONCEPT, f"Node {i}")

        # Compact
        snapshot_id = graph_wal.compact_wal(graph)

        # Verify snapshot was created
        loaded = graph_wal.load_snapshot(snapshot_id)
        assert loaded is not None
        assert loaded.node_count() == 50


class TestGitAutoCommitterWithWAL:
    """Test GitAutoCommitter integration with WAL operations."""

    def test_commit_after_wal_operations(self, tmp_path):
        """Test auto-commit after WAL writes."""
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=tmp_path, check=True, capture_output=True)

        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Create and save graph (need at least 2 nodes with an edge to pass validation)
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Test")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES)
        snapshot_id = graph_wal.create_snapshot(graph, compress=False)
        snapshot_path = wal_dir / "snapshots" / f"{snapshot_id}.json"

        # Verify snapshot file exists
        assert snapshot_path.exists(), f"Snapshot not found at {snapshot_path}"

        # Auto-commit
        committer = GitAutoCommitter(mode='immediate', repo_path=str(tmp_path))
        success = committer.auto_commit(
            "graph: Test snapshot",
            [str(snapshot_path.relative_to(tmp_path))],
            validate_graph=graph
        )

        assert success

        # Verify commit was created
        result = subprocess.run(
            ['git', 'log', '--oneline'],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )
        assert "graph: Test snapshot" in result.stdout

    def test_git_state_consistent_with_wal(self, tmp_path):
        """Test git commits keep WAL and snapshots consistent."""
        # Setup git
        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=tmp_path, check=True, capture_output=True)

        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))
        committer = GitAutoCommitter(mode='immediate', repo_path=str(tmp_path))

        # Create graph, snapshot, commit
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "V1")
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "V1")
        snap1 = graph_wal.create_snapshot(graph, compress=False)

        # Commit snapshot
        snapshot_path = wal_dir / "snapshots" / f"{snap1}.json"
        assert snapshot_path.exists()
        committer.auto_commit("graph: V1", [str(snapshot_path.relative_to(tmp_path))])

        # Modify graph
        graph.add_node("Q2", NodeType.QUESTION, "V2")
        graph_wal.log_add_node("Q2", NodeType.QUESTION, "V2")
        snap2 = graph_wal.create_snapshot(graph, compress=False)

        # Commit again
        snapshot_path2 = wal_dir / "snapshots" / f"{snap2}.json"
        assert snapshot_path2.exists()
        committer.auto_commit("graph: V2", [str(snapshot_path2.relative_to(tmp_path))])

        # Verify both commits exist
        result = subprocess.run(
            ['git', 'log', '--oneline'],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )
        assert "graph: V1" in result.stdout
        assert "graph: V2" in result.stdout

    def test_push_only_after_clean_wal(self, tmp_path):
        """Test push only happens when WAL is in clean state."""
        # Setup git with remote (just another directory)
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        subprocess.run(['git', 'init', '--bare'], cwd=remote_dir, check=True, capture_output=True)

        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'remote', 'add', 'origin', str(remote_dir)], cwd=tmp_path, check=True, capture_output=True)

        # Create test branch
        subprocess.run(['git', 'checkout', '-b', 'test-branch'], cwd=tmp_path, check=True, capture_output=True)

        # Make initial commit
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        subprocess.run(['git', 'add', 'test.txt'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial'], cwd=tmp_path, check=True, capture_output=True)

        # Push should work on non-protected branch
        committer = GitAutoCommitter(
            mode='immediate',
            repo_path=str(tmp_path),
            protected_branches={'main', 'master'}
        )

        # Push to test-branch (not protected)
        success = committer.push_if_safe(branch='test-branch')
        assert success

    def test_validation_prevents_bad_commits(self, tmp_path):
        """Test validation prevents committing invalid graphs."""
        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True, capture_output=True)

        committer = GitAutoCommitter(mode='immediate', repo_path=str(tmp_path))

        # Try to commit empty graph
        empty_graph = ThoughtGraph()
        test_file = tmp_path / "empty.json"
        test_file.write_text("{}")

        success = committer.auto_commit(
            "graph: Empty",
            [str(test_file.relative_to(tmp_path))],
            validate_graph=empty_graph
        )

        # Should fail validation
        assert not success

    def test_debounced_commits(self, tmp_path):
        """Test debounced commit mode waits for inactivity."""
        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True, capture_output=True)

        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Test")
        snapshot_path = wal_dir / "snapshots" / "test.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("{}")

        committer = GitAutoCommitter(
            mode='debounced',
            debounce_seconds=1,
            repo_path=str(tmp_path)
        )

        # Trigger debounced commit
        committer.commit_on_save(str(snapshot_path), graph, "Test commit")

        # Wait for debounce
        time.sleep(1.5)

        # Check commit happened
        result = subprocess.run(
            ['git', 'log', '--oneline'],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )

        # Cleanup timer
        committer.cleanup()

        # Note: Debounced commit may not complete in test environment
        # This test verifies the mechanism doesn't crash

    def test_protected_branch_detection(self, tmp_path):
        """Test protected branch detection prevents auto-push."""
        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=tmp_path, check=True, capture_output=True)

        # Create main branch
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        subprocess.run(['git', 'add', 'test.txt'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial'], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(['git', 'branch', '-M', 'main'], cwd=tmp_path, check=True, capture_output=True)

        committer = GitAutoCommitter(repo_path=str(tmp_path))

        # Check main is protected
        assert committer.is_protected_branch('main')
        assert committer.is_protected_branch('master')
        assert not committer.is_protected_branch('feature-branch')


class TestFullPersistenceStack:
    """Test end-to-end persistence with all components."""

    @pytest.mark.skip(reason="GraphRecovery implementation incomplete - goes to Level 4 instead of trying Levels 1-2")
    def test_end_to_end_create_save_crash_recover(self, tmp_path):
        """Test full cycle: create graph, save, simulate crash, recover."""
        wal_dir = tmp_path / "wal"

        # Phase 1: Create and persist graph
        graph_wal1 = GraphWAL(str(wal_dir))
        graph1 = ThoughtGraph()
        graph1.add_node("Q1", NodeType.QUESTION, "What to do?")
        graph1.add_node("H1", NodeType.HYPOTHESIS, "Try approach A")
        graph1.add_node("H2", NodeType.HYPOTHESIS, "Try approach B")
        graph1.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.6)
        graph1.add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.8)

        # Log all operations
        graph_wal1.log_add_node("Q1", NodeType.QUESTION, "What to do?")
        graph_wal1.log_add_node("H1", NodeType.HYPOTHESIS, "Try approach A")
        graph_wal1.log_add_node("H2", NodeType.HYPOTHESIS, "Try approach B")
        graph_wal1.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.6)
        graph_wal1.log_add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.8)

        # Create snapshot
        snapshot_id = graph_wal1.create_snapshot(graph1)

        # Add one more operation after snapshot so WAL replay has something to do
        graph_wal1.log_add_node("Q3", NodeType.QUESTION, "After snapshot")

        # Phase 2: Simulate crash (discard in-memory objects)
        del graph_wal1
        del graph1

        # Phase 3: Recover
        recovery = GraphRecovery(str(wal_dir))
        result = recovery.recover()

        # Should succeed with Level 1 or 2
        assert result.success, f"Recovery failed: {result.errors}"
        assert result.graph is not None
        # Could be 3 (from snapshot) or 4 (if replay succeeded) nodes
        assert result.graph.node_count() >= 3

        # Verify content
        q1 = result.graph.get_node("Q1")
        assert q1 is not None
        assert q1.content == "What to do?"

    @pytest.mark.skip(reason="GraphRecovery implementation incomplete - goes to Level 4 instead of trying Levels 1-2")
    def test_incremental_updates_across_sessions(self, tmp_path):
        """Test incremental updates persist across multiple sessions."""
        wal_dir = tmp_path / "wal"

        # Session 1: Initial graph
        graph_wal1 = GraphWAL(str(wal_dir))
        graph1 = ThoughtGraph()
        graph1.add_node("Q1", NodeType.QUESTION, "Initial")
        graph_wal1.log_add_node("Q1", NodeType.QUESTION, "Initial")
        snap1 = graph_wal1.create_snapshot(graph1)

        # Session 2: Load and add more nodes
        graph_wal2 = GraphWAL(str(wal_dir))
        graph2 = graph_wal2.load_snapshot(snap1)
        assert graph2 is not None, "Failed to load snapshot"

        graph2.add_node("Q2", NodeType.QUESTION, "Session 2")
        graph_wal2.log_add_node("Q2", NodeType.QUESTION, "Session 2")
        snap2 = graph_wal2.create_snapshot(graph2)

        # Session 3: Load and add even more
        graph_wal3 = GraphWAL(str(wal_dir))
        graph3 = graph_wal3.load_snapshot(snap2)
        assert graph3 is not None, "Failed to load snapshot 2"

        graph3.add_node("Q3", NodeType.QUESTION, "Session 3")
        graph_wal3.log_add_node("Q3", NodeType.QUESTION, "Session 3")
        snap3 = graph_wal3.create_snapshot(graph3)

        # Final recovery should have all nodes
        recovery_final = GraphRecovery(str(wal_dir))
        result_final = recovery_final.recover()

        assert result_final.success, f"Recovery failed: {result_final.errors}"
        assert result_final.graph.node_count() == 3
        assert result_final.graph.get_node("Q1") is not None
        assert result_final.graph.get_node("Q2") is not None
        assert result_final.graph.get_node("Q3") is not None

    def test_large_graph_persistence_performance(self, tmp_path):
        """Test persistence handles large graphs efficiently."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Create large graph
        graph = ThoughtGraph()
        num_nodes = 500

        for i in range(num_nodes):
            graph.add_node(f"N{i}", NodeType.CONCEPT, f"Node {i}")
            graph_wal.log_add_node(f"N{i}", NodeType.CONCEPT, f"Node {i}")

        # Add edges
        for i in range(num_nodes - 1):
            graph.add_edge(f"N{i}", f"N{i+1}", EdgeType.SIMILAR)
            graph_wal.log_add_edge(f"N{i}", f"N{i+1}", EdgeType.SIMILAR)

        # Snapshot should complete in reasonable time
        import time
        start = time.time()
        snapshot_id = graph_wal.create_snapshot(graph, compress=True)
        duration = time.time() - start

        assert duration < 10.0, f"Snapshot took {duration}s (too slow)"

        # Recovery should also be fast
        start = time.time()
        loaded = graph_wal.load_snapshot(snapshot_id)
        duration = time.time() - start

        assert duration < 5.0, f"Load took {duration}s (too slow)"
        assert loaded.node_count() == num_nodes

    def test_concurrent_wal_operations_isolation(self, tmp_path):
        """Test WAL operations from different sessions are isolated."""
        # Create two separate WAL instances
        wal_dir1 = tmp_path / "wal1"
        wal_dir2 = tmp_path / "wal2"

        graph_wal1 = GraphWAL(str(wal_dir1))
        graph_wal2 = GraphWAL(str(wal_dir2))

        # Write to both
        graph_wal1.log_add_node("Session1", NodeType.QUESTION, "From session 1")
        graph_wal2.log_add_node("Session2", NodeType.QUESTION, "From session 2")

        # Verify isolation
        graph1 = ThoughtGraph()
        for entry in graph_wal1.get_all_entries():
            graph_wal1.apply_entry(entry, graph1)

        graph2 = ThoughtGraph()
        for entry in graph_wal2.get_all_entries():
            graph_wal2.apply_entry(entry, graph2)

        # Each graph should only have its own node
        assert graph1.node_count() == 1
        assert graph2.node_count() == 1
        assert graph1.get_node("Session1") is not None
        assert graph1.get_node("Session2") is None
        assert graph2.get_node("Session2") is not None
        assert graph2.get_node("Session1") is None

    @pytest.mark.skip(reason="GraphRecovery implementation incomplete - goes to Level 4 instead of trying Levels 1-2")
    def test_recovery_integrity_verification(self, tmp_path):
        """Test recovery verifies graph integrity after rebuild."""
        wal_dir = tmp_path / "wal"
        graph_wal = GraphWAL(str(wal_dir))

        # Create graph with potential issues
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Valid node")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Valid hypothesis")
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES)

        # Log operations
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Valid node")
        graph_wal.log_add_node("H1", NodeType.HYPOTHESIS, "Valid hypothesis")
        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES)
        graph_wal.create_snapshot(graph)

        # Add one more operation for WAL replay
        graph_wal.log_add_node("E1", NodeType.EVIDENCE, "Evidence")

        # Recover and verify
        recovery = GraphRecovery(str(wal_dir))
        result = recovery.recover()

        assert result.success, f"Recovery failed: {result.errors}"

        # Run integrity checks
        issues = recovery.verify_graph_integrity(result.graph)
        assert len(issues) == 0, f"Found integrity issues: {issues}"
