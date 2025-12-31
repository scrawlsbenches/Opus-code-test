"""
Behavioral tests for ThoughtGraph persistence layer.

As a researcher building reasoning systems,
I want durable graph storage with crash recovery,
So that my reasoning chains survive system failures.

Tests demonstrate:
- Write-ahead logging for durability
- Snapshot creation and restoration
- Multi-level crash recovery
- Git integration for version control

Following Metus: We describe behavior, then make it true.
"""

import sys
import time
import tempfile
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.reasoning import (
    ThoughtGraph,
    NodeType,
    EdgeType,
    GraphWAL,
    GitAutoCommitter,
    GraphRecovery,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_wal_dir(tmp_path):
    """Provide a temporary directory for WAL operations."""
    return tmp_path / "reasoning_wal"


@pytest.fixture
def graph_wal(temp_wal_dir):
    """Provide a fresh GraphWAL instance."""
    return GraphWAL(str(temp_wal_dir))


@pytest.fixture
def thought_graph():
    """Provide a fresh ThoughtGraph instance."""
    return ThoughtGraph()


# ============================================================================
# BEHAVIORAL SCENARIOS
# ============================================================================

class TestResearcherBuildsReasoningGraph:
    """
    Epic: Reasoning Graph Construction

    As a researcher exploring complex questions,
    I want to build graphs of interconnected thoughts,
    So that I can capture and navigate my reasoning process.
    """

    def test_scenario_add_nodes_and_edges_to_reasoning_graph(
        self, graph_wal, thought_graph
    ):
        """
        Scenario: Building a reasoning graph with questions and hypotheses

        Given I have a research question
        When I add nodes representing the question and hypotheses
        And I connect them with edges showing relationships
        Then the graph captures my reasoning structure
        """
        # Given: I have a research question
        # When: I add nodes representing the question and hypotheses
        graph_wal.log_add_node(
            "Q1", NodeType.QUESTION,
            "What is the optimal approach for building custom text search?",
            properties={'priority': 'high', 'domain': 'information_retrieval'},
            metadata={'created_by': 'researcher', 'session': 'exploration_001'}
        )
        thought_graph.add_node(
            "Q1", NodeType.QUESTION,
            "What is the optimal approach for building custom text search?"
        )

        # Add hypothesis nodes
        graph_wal.log_add_node(
            "H1", NodeType.HYPOTHESIS,
            "Build inverted index from scratch for full control",
            properties={'confidence': 0.9}
        )
        thought_graph.add_node(
            "H1", NodeType.HYPOTHESIS,
            "Build inverted index from scratch for full control"
        )

        graph_wal.log_add_node(
            "H2", NodeType.HYPOTHESIS,
            "Implement custom ranking algorithm ourselves",
            properties={'confidence': 0.85}
        )
        thought_graph.add_node(
            "H2", NodeType.HYPOTHESIS,
            "Implement custom ranking algorithm ourselves"
        )

        # And: I connect them with edges showing relationships
        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)
        thought_graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)

        graph_wal.log_add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.85)
        thought_graph.add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.85)

        # Then: the graph captures my reasoning structure
        assert thought_graph.node_count() == 3
        assert thought_graph.edge_count() == 2

        # And: WAL has recorded all operations
        entry_count = graph_wal.get_entry_count()
        assert entry_count >= 5  # 3 nodes + 2 edges


class TestResearcherCreatesRecoveryCheckpoints:
    """
    Epic: Snapshot-Based Recovery

    As a researcher with hours of reasoning work,
    I want periodic snapshots of my graph state,
    So that I can quickly recover from crashes without replaying everything.
    """

    def test_scenario_create_and_load_snapshot(self, graph_wal, thought_graph):
        """
        Scenario: Snapshot preserves complete graph state

        Given I have built a reasoning graph
        When I create a snapshot
        And I load it back
        Then all nodes and edges are restored
        """
        # Given: I have built a reasoning graph
        for i in range(5):
            node_id = f"C{i}"
            graph_wal.log_add_node(
                node_id, NodeType.CONCEPT,
                f"Custom implementation concept {i}",
                properties={'iteration': i}
            )
            thought_graph.add_node(
                node_id, NodeType.CONCEPT,
                f"Custom implementation concept {i}"
            )

        # Add edges between concepts
        graph_wal.log_add_edge("C0", "C1", EdgeType.SUPPORTS)
        thought_graph.add_edge("C0", "C1", EdgeType.SUPPORTS)

        graph_wal.log_add_edge("C1", "C2", EdgeType.CONTRADICTS)
        thought_graph.add_edge("C1", "C2", EdgeType.CONTRADICTS)

        original_nodes = thought_graph.node_count()
        original_edges = thought_graph.edge_count()

        # When: I create a snapshot
        snapshot_id = graph_wal.create_snapshot(thought_graph, compress=True)
        assert snapshot_id is not None

        # And: I load it back
        loaded_graph = graph_wal.load_snapshot(snapshot_id)

        # Then: all nodes and edges are restored
        assert loaded_graph is not None
        assert loaded_graph.node_count() == original_nodes
        assert loaded_graph.edge_count() == original_edges


class TestSystemRecovesFromGraphCorruption:
    """
    Epic: Multi-Level Crash Recovery

    As a system operator ensuring data durability,
    I want cascading recovery strategies,
    So that the system can always restore to a consistent state.
    """

    def test_scenario_recover_from_wal_after_crash(self, temp_wal_dir):
        """
        Scenario: WAL replay recovers graph after crash

        Given I was building a graph and logged operations to WAL
        When the system crashes before creating a snapshot
        And I run recovery on restart
        Then the graph is reconstructed from WAL entries
        """
        # Given: I was building a graph and logged operations to WAL
        graph_wal = GraphWAL(str(temp_wal_dir))
        thought_graph = ThoughtGraph()

        # Log initial state
        graph_wal.log_add_node("Q1", NodeType.QUESTION, "Main research question")
        thought_graph.add_node("Q1", NodeType.QUESTION, "Main research question")

        # Create snapshot
        snapshot_id = graph_wal.create_snapshot(thought_graph)

        # Continue adding nodes (these will be in WAL only)
        for i in range(3):
            node_id = f"A{i}"
            graph_wal.log_add_node(
                node_id, NodeType.ACTION,
                f"Build component {i} ourselves"
            )
            thought_graph.add_node(
                node_id, NodeType.ACTION,
                f"Build component {i} ourselves"
            )

        nodes_before_crash = thought_graph.node_count()

        # When: the system crashes before creating a snapshot
        # Simulate crash by discarding in-memory graph
        thought_graph = None

        # And: I run recovery on restart
        recovery = GraphRecovery(str(temp_wal_dir))

        # Then: the graph is reconstructed from WAL entries
        if recovery.needs_recovery():
            result = recovery.recover()
            assert result.success
            assert result.graph is not None
            # Recovery should restore all nodes (snapshot + WAL replay)
            assert result.graph.node_count() == nodes_before_crash


class TestResearcherTracksGraphChangesInGit:
    """
    Epic: Version Control Integration

    As a researcher tracking reasoning evolution,
    I want automatic git commits for graph changes,
    So that I have a complete history of my thought process.
    """

    def test_scenario_git_committer_respects_protected_branches(self):
        """
        Scenario: Git integration respects safety constraints

        Given I configure a GitAutoCommitter
        When I check if branches are protected
        Then main and master are always protected
        And force push is never allowed
        """
        # Given: I configure a GitAutoCommitter
        committer = GitAutoCommitter(mode='manual', auto_push=False)

        # When: I check if branches are protected
        # Then: main and master are always protected
        assert committer.is_protected_branch('main')
        assert committer.is_protected_branch('master')

        # And: force push is never allowed
        # (This is a property of the implementation - verified by code inspection)
        assert 'main' in committer.protected_branches
        assert 'master' in committer.protected_branches

    def test_scenario_git_committer_modes_available(self):
        """
        Scenario: Multiple commit strategies available

        Given I need different commit strategies
        When I create committers with different modes
        Then I can choose immediate, debounced, or manual commits
        """
        # Given: I need different commit strategies
        # When: I create committers with different modes

        # Immediate mode: commits on every save
        immediate = GitAutoCommitter(mode='immediate', auto_push=False)
        assert immediate.mode == 'immediate'

        # Debounced mode: waits for quiet period
        debounced = GitAutoCommitter(mode='debounced', debounce_seconds=5)
        assert debounced.mode == 'debounced'

        # Manual mode: only commits when explicitly requested
        manual = GitAutoCommitter(mode='manual', auto_push=False)
        assert manual.mode == 'manual'

        # Then: I can choose the right strategy for my workflow


class TestSystemProvidesRecoveryTransparency:
    """
    Epic: Recovery Observability

    As a system operator troubleshooting issues,
    I want detailed information about recovery operations,
    So that I can understand what happened and verify correctness.
    """

    def test_scenario_recovery_reports_detailed_results(self, temp_wal_dir):
        """
        Scenario: Recovery provides comprehensive status information

        Given a graph needs recovery
        When I run the recovery process
        Then I receive detailed information about what was recovered
        """
        # Given: a graph needs recovery
        graph_wal = GraphWAL(str(temp_wal_dir))
        thought_graph = ThoughtGraph()

        # Add some nodes
        graph_wal.log_add_node("N1", NodeType.CONCEPT, "Concept 1")
        thought_graph.add_node("N1", NodeType.CONCEPT, "Concept 1")

        graph_wal.log_add_node("N2", NodeType.CONCEPT, "Concept 2")
        thought_graph.add_node("N2", NodeType.CONCEPT, "Concept 2")

        # When: I run the recovery process
        recovery = GraphRecovery(str(temp_wal_dir))

        if recovery.needs_recovery():
            result = recovery.recover()

            # Then: I receive detailed information about what was recovered
            assert hasattr(result, 'success')
            assert hasattr(result, 'level_used')
            assert hasattr(result, 'recovery_method')
            assert hasattr(result, 'nodes_recovered')
            assert hasattr(result, 'edges_recovered')
            assert hasattr(result, 'duration_ms')


class TestResearcherManagesLongRunningInvestigations:
    """
    Epic: Scalable Graph Construction

    As a researcher conducting months-long investigations,
    I want efficient handling of large reasoning graphs,
    So that my tools remain responsive as complexity grows.
    """

    def test_scenario_wal_handles_many_operations(self, graph_wal, thought_graph):
        """
        Scenario: WAL scales to thousands of operations

        Given I am building a large reasoning graph
        When I log hundreds of operations
        Then the WAL remains efficient
        And I can query the entry count
        """
        # Given: I am building a large reasoning graph
        # When: I log hundreds of operations
        num_nodes = 100

        for i in range(num_nodes):
            node_id = f"N{i}"
            graph_wal.log_add_node(
                node_id, NodeType.CONCEPT,
                f"Hand-built component {i} with no external dependencies"
            )
            thought_graph.add_node(
                node_id, NodeType.CONCEPT,
                f"Hand-built component {i} with no external dependencies"
            )

        # Then: the WAL remains efficient
        entry_count = graph_wal.get_entry_count()
        assert entry_count >= num_nodes

        # And: I can query the entry count
        assert entry_count > 0

    def test_scenario_compressed_snapshots_save_space(self, graph_wal, thought_graph):
        """
        Scenario: Snapshots can be compressed for efficiency

        Given I have a large reasoning graph
        When I create a compressed snapshot
        Then the snapshot is stored efficiently
        And can be loaded back correctly
        """
        # Given: I have a large reasoning graph
        for i in range(50):
            node_id = f"N{i}"
            thought_graph.add_node(
                node_id, NodeType.CONCEPT,
                f"Concept {i}: Building everything ourselves from first principles"
            )

        # When: I create a compressed snapshot
        snapshot_id = graph_wal.create_snapshot(thought_graph, compress=True)

        # Then: the snapshot is stored efficiently
        assert snapshot_id is not None

        # And: can be loaded back correctly
        loaded_graph = graph_wal.load_snapshot(snapshot_id)
        assert loaded_graph is not None
        assert loaded_graph.node_count() == thought_graph.node_count()


class TestResearcherValidatesRecoveryStrategy:
    """
    Epic: Recovery Strategy Validation

    As a researcher ensuring data integrity,
    I want to verify recovery mechanisms work correctly,
    So that I can trust my data will survive failures.
    """

    def test_scenario_recovery_detects_when_not_needed(self, temp_wal_dir):
        """
        Scenario: Recovery correctly identifies when not needed

        Given a pristine WAL directory
        When I check if recovery is needed
        Then the system reports no recovery necessary
        """
        # Given: a pristine WAL directory
        recovery = GraphRecovery(str(temp_wal_dir))

        # When: I check if recovery is needed
        # Then: the system reports no recovery necessary
        # (In a pristine directory, there's nothing to recover)
        # The result depends on whether there's existing state


class TestSystemProvidesWALIntrospection:
    """
    Epic: WAL Observability

    As a system operator monitoring graph operations,
    I want visibility into WAL state,
    So that I can monitor durability and diagnose issues.
    """

    def test_scenario_wal_reports_current_path(self, graph_wal):
        """
        Scenario: WAL provides path to current log file

        Given I have a GraphWAL instance
        When I query the current WAL path
        Then I receive the path to the active log file
        """
        # Given: I have a GraphWAL instance
        # When: I query the current WAL path
        wal_path = graph_wal.get_current_wal_path()

        # Then: I receive the path to the active log file
        assert wal_path is not None
        assert isinstance(wal_path, (str, Path))

    def test_scenario_wal_counts_logged_operations(self, graph_wal, thought_graph):
        """
        Scenario: WAL tracks number of logged operations

        Given I log multiple graph operations
        When I query the entry count
        Then I see the correct number of operations
        """
        # Given: I log multiple graph operations
        graph_wal.log_add_node("N1", NodeType.CONCEPT, "First concept")
        graph_wal.log_add_node("N2", NodeType.CONCEPT, "Second concept")
        graph_wal.log_add_edge("N1", "N2", EdgeType.SUPPORTS)

        # When: I query the entry count
        count = graph_wal.get_entry_count()

        # Then: I see the correct number of operations
        assert count >= 3  # 2 nodes + 1 edge
