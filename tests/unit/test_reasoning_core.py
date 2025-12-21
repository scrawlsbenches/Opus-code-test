"""
Unit tests for the reasoning framework core components.

Tests cover:
- NodeType and EdgeType enums
- ThoughtNode, ThoughtEdge, ThoughtCluster dataclasses
- ThoughtGraph operations
- Pattern factory functions
"""

import pytest
from cortical.reasoning import (
    # Core types
    NodeType,
    EdgeType,
    ThoughtNode,
    ThoughtEdge,
    ThoughtCluster,
    ThoughtGraph,
    # Pattern factories
    create_investigation_graph,
    create_decision_graph,
    create_debug_graph,
    create_feature_graph,
    create_requirements_graph,
    create_analysis_graph,
    create_pattern_graph,
    PATTERN_REGISTRY,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_core_types_exist(self):
        """Verify core node types are defined."""
        assert NodeType.CONCEPT.value == "concept"
        assert NodeType.QUESTION.value == "question"
        assert NodeType.DECISION.value == "decision"
        assert NodeType.FACT.value == "fact"
        assert NodeType.TASK.value == "task"
        assert NodeType.ARTIFACT.value == "artifact"
        assert NodeType.INSIGHT.value == "insight"

    def test_extended_types_exist(self):
        """Verify extended node types are defined."""
        assert NodeType.HYPOTHESIS.value == "hypothesis"
        assert NodeType.OPTION.value == "option"
        assert NodeType.EVIDENCE.value == "evidence"
        assert NodeType.OBSERVATION.value == "observation"
        assert NodeType.GOAL.value == "goal"
        assert NodeType.CONTEXT.value == "context"
        assert NodeType.CONSTRAINT.value == "constraint"
        assert NodeType.ACTION.value == "action"

    def test_repr(self):
        """Test string representation."""
        assert repr(NodeType.CONCEPT) == "NodeType.CONCEPT"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_semantic_edges_exist(self):
        """Verify semantic edge types are defined."""
        assert EdgeType.REQUIRES.value == "requires"
        assert EdgeType.ENABLES.value == "enables"
        assert EdgeType.CONFLICTS.value == "conflicts"
        assert EdgeType.SUPPORTS.value == "supports"
        assert EdgeType.REFUTES.value == "refutes"
        assert EdgeType.SIMILAR.value == "similar"
        assert EdgeType.CONTAINS.value == "contains"
        assert EdgeType.CONTRADICTS.value == "contradicts"

    def test_temporal_edges_exist(self):
        """Verify temporal edge types are defined."""
        assert EdgeType.PRECEDES.value == "precedes"
        assert EdgeType.TRIGGERS.value == "triggers"
        assert EdgeType.BLOCKS.value == "blocks"

    def test_epistemic_edges_exist(self):
        """Verify epistemic edge types are defined."""
        assert EdgeType.ANSWERS.value == "answers"
        assert EdgeType.RAISES.value == "raises"
        assert EdgeType.EXPLORES.value == "explores"
        assert EdgeType.OBSERVES.value == "observes"
        assert EdgeType.SUGGESTS.value == "suggests"

    def test_practical_edges_exist(self):
        """Verify practical edge types are defined."""
        assert EdgeType.IMPLEMENTS.value == "implements"
        assert EdgeType.TESTS.value == "tests"
        assert EdgeType.DEPENDS_ON.value == "depends_on"
        assert EdgeType.REFINES.value == "refines"
        assert EdgeType.MOTIVATES.value == "motivates"

    def test_structural_edges_exist(self):
        """Verify structural edge types are defined."""
        assert EdgeType.HAS_OPTION.value == "has_option"
        assert EdgeType.HAS_ASPECT.value == "has_aspect"


class TestThoughtNode:
    """Tests for ThoughtNode dataclass."""

    def test_create_node(self):
        """Test creating a basic node."""
        node = ThoughtNode(
            id="n1",
            node_type=NodeType.CONCEPT,
            content="Test concept",
        )
        assert node.id == "n1"
        assert node.node_type == NodeType.CONCEPT
        assert node.content == "Test concept"
        assert node.properties == {}
        assert node.metadata == {}

    def test_create_node_with_properties(self):
        """Test creating node with properties."""
        node = ThoughtNode(
            id="n1",
            node_type=NodeType.DECISION,
            content="Choose auth method",
            properties={"options": ["OAuth", "JWT"]},
            metadata={"created_by": "test"},
        )
        assert node.properties == {"options": ["OAuth", "JWT"]}
        assert node.metadata == {"created_by": "test"}

    def test_node_hash(self):
        """Test node hashing is based on ID."""
        node1 = ThoughtNode(id="n1", node_type=NodeType.CONCEPT, content="A")
        node2 = ThoughtNode(id="n1", node_type=NodeType.CONCEPT, content="B")
        assert hash(node1) == hash(node2)

    def test_node_equality(self):
        """Test node equality is based on ID."""
        node1 = ThoughtNode(id="n1", node_type=NodeType.CONCEPT, content="A")
        node2 = ThoughtNode(id="n1", node_type=NodeType.CONCEPT, content="B")
        node3 = ThoughtNode(id="n2", node_type=NodeType.CONCEPT, content="A")
        assert node1 == node2
        assert node1 != node3

    def test_node_repr(self):
        """Test node string representation."""
        node = ThoughtNode(id="n1", node_type=NodeType.CONCEPT, content="Test")
        repr_str = repr(node)
        assert "n1" in repr_str
        assert "CONCEPT" in repr_str


class TestThoughtEdge:
    """Tests for ThoughtEdge dataclass."""

    def test_create_edge(self):
        """Test creating a basic edge."""
        edge = ThoughtEdge(
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.REQUIRES,
        )
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"
        assert edge.edge_type == EdgeType.REQUIRES
        assert edge.weight == 1.0
        assert edge.confidence == 1.0
        assert edge.bidirectional is False

    def test_edge_with_weight(self):
        """Test edge with custom weight."""
        edge = ThoughtEdge(
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.SUPPORTS,
            weight=0.8,
            confidence=0.9,
        )
        assert edge.weight == 0.8
        assert edge.confidence == 0.9

    def test_edge_weight_validation(self):
        """Test edge weight must be in [0, 1]."""
        with pytest.raises(ValueError):
            ThoughtEdge(
                source_id="n1",
                target_id="n2",
                edge_type=EdgeType.REQUIRES,
                weight=1.5,
            )

    def test_edge_confidence_validation(self):
        """Test edge confidence must be in [0, 1]."""
        with pytest.raises(ValueError):
            ThoughtEdge(
                source_id="n1",
                target_id="n2",
                edge_type=EdgeType.REQUIRES,
                confidence=-0.1,
            )

    def test_bidirectional_edge(self):
        """Test bidirectional edge."""
        edge = ThoughtEdge(
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.SIMILAR,
            bidirectional=True,
        )
        assert edge.bidirectional is True

    def test_edge_repr(self):
        """Test edge string representation."""
        edge = ThoughtEdge(
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.REQUIRES,
        )
        repr_str = repr(edge)
        assert "n1" in repr_str
        assert "n2" in repr_str
        assert "REQUIRES" in repr_str


class TestThoughtCluster:
    """Tests for ThoughtCluster dataclass."""

    def test_create_cluster(self):
        """Test creating a basic cluster."""
        cluster = ThoughtCluster(
            id="c1",
            name="Auth Cluster",
        )
        assert cluster.id == "c1"
        assert cluster.name == "Auth Cluster"
        assert cluster.node_ids == set()
        assert cluster.size == 0

    def test_cluster_with_nodes(self):
        """Test cluster with initial nodes."""
        cluster = ThoughtCluster(
            id="c1",
            name="Auth Cluster",
            node_ids={"n1", "n2", "n3"},
        )
        assert cluster.size == 3

    def test_add_node(self):
        """Test adding node to cluster."""
        cluster = ThoughtCluster(id="c1", name="Test")
        cluster.add_node("n1")
        assert "n1" in cluster.node_ids
        assert cluster.size == 1

    def test_remove_node(self):
        """Test removing node from cluster."""
        cluster = ThoughtCluster(id="c1", name="Test", node_ids={"n1", "n2"})
        cluster.remove_node("n1")
        assert "n1" not in cluster.node_ids
        assert cluster.size == 1

    def test_remove_nonexistent_node(self):
        """Test removing non-existent node raises KeyError."""
        cluster = ThoughtCluster(id="c1", name="Test")
        with pytest.raises(KeyError):
            cluster.remove_node("n1")

    def test_contains_node(self):
        """Test checking if node is in cluster."""
        cluster = ThoughtCluster(id="c1", name="Test", node_ids={"n1"})
        assert cluster.contains_node("n1") is True
        assert cluster.contains_node("n2") is False


class TestThoughtGraph:
    """Tests for ThoughtGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = ThoughtGraph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.cluster_count() == 0

    def test_add_node(self):
        """Test adding a node."""
        graph = ThoughtGraph()
        node = graph.add_node("n1", NodeType.CONCEPT, "Test concept")
        assert graph.node_count() == 1
        assert node.id == "n1"
        assert node.content == "Test concept"

    def test_add_duplicate_node_raises(self):
        """Test adding duplicate node ID raises ValueError."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "Test")
        with pytest.raises(ValueError):
            graph.add_node("n1", NodeType.CONCEPT, "Another")

    def test_add_edge(self):
        """Test adding an edge."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        edge = graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        assert graph.edge_count() == 1
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"

    def test_add_edge_nonexistent_source_raises(self):
        """Test adding edge with non-existent source raises ValueError."""
        graph = ThoughtGraph()
        graph.add_node("n2", NodeType.CONCEPT, "B")
        with pytest.raises(ValueError):
            graph.add_edge("n1", "n2", EdgeType.REQUIRES)

    def test_add_edge_nonexistent_target_raises(self):
        """Test adding edge with non-existent target raises ValueError."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        with pytest.raises(ValueError):
            graph.add_edge("n1", "n2", EdgeType.REQUIRES)

    def test_add_bidirectional_edge(self):
        """Test adding bidirectional edge creates two edges."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_edge("n1", "n2", EdgeType.SIMILAR, bidirectional=True)
        assert graph.edge_count() == 2

    def test_remove_node(self):
        """Test removing a node."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)

        removed = graph.remove_node("n1")
        assert removed.id == "n1"
        assert graph.node_count() == 1
        assert graph.edge_count() == 0

    def test_remove_nonexistent_node_raises(self):
        """Test removing non-existent node raises ValueError."""
        graph = ThoughtGraph()
        with pytest.raises(ValueError):
            graph.remove_node("n1")

    def test_get_node(self):
        """Test getting a node by ID."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "Test")
        node = graph.get_node("n1")
        assert node is not None
        assert node.id == "n1"

    def test_get_nonexistent_node(self):
        """Test getting non-existent node returns None."""
        graph = ThoughtGraph()
        assert graph.get_node("n1") is None

    def test_get_edges_from(self):
        """Test getting edges from a node."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        graph.add_edge("n1", "n3", EdgeType.ENABLES)

        edges = graph.get_edges_from("n1")
        assert len(edges) == 2

    def test_get_edges_to(self):
        """Test getting edges to a node."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n3", EdgeType.REQUIRES)
        graph.add_edge("n2", "n3", EdgeType.REQUIRES)

        edges = graph.get_edges_to("n3")
        assert len(edges) == 2

    def test_get_neighbors(self):
        """Test getting neighboring nodes."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        graph.add_edge("n1", "n3", EdgeType.ENABLES)

        neighbors = graph.get_neighbors("n1")
        assert set(neighbors) == {"n2", "n3"}

    def test_nodes_of_type(self):
        """Test filtering nodes by type."""
        graph = ThoughtGraph()
        graph.add_node("q1", NodeType.QUESTION, "Q1")
        graph.add_node("q2", NodeType.QUESTION, "Q2")
        graph.add_node("d1", NodeType.DECISION, "D1")

        questions = graph.nodes_of_type(NodeType.QUESTION)
        assert len(questions) == 2


class TestThoughtGraphTraversal:
    """Tests for ThoughtGraph traversal operations."""

    def test_dfs(self):
        """Test depth-first search."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        graph.add_edge("n2", "n3", EdgeType.REQUIRES)

        visited = graph.dfs("n1")
        assert visited == ["n1", "n2", "n3"]

    def test_bfs(self):
        """Test breadth-first search."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        graph.add_edge("n1", "n3", EdgeType.REQUIRES)

        visited = graph.bfs("n1")
        assert visited[0] == "n1"
        assert set(visited[1:]) == {"n2", "n3"}

    def test_shortest_path(self):
        """Test finding shortest path."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        graph.add_edge("n2", "n3", EdgeType.REQUIRES)

        path = graph.shortest_path("n1", "n3")
        assert path == ["n1", "n2", "n3"]

    def test_shortest_path_same_node(self):
        """Test shortest path to same node."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")

        path = graph.shortest_path("n1", "n1")
        assert path == ["n1"]

    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")

        path = graph.shortest_path("n1", "n2")
        assert path is None


class TestThoughtGraphAnalysis:
    """Tests for ThoughtGraph analysis operations."""

    def test_find_orphans(self):
        """Test finding orphan nodes."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)

        orphans = graph.find_orphans()
        assert orphans == ["n3"]

    def test_find_hubs(self):
        """Test finding hub nodes."""
        graph = ThoughtGraph()
        graph.add_node("hub", NodeType.CONCEPT, "Hub")
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("hub", "n1", EdgeType.REQUIRES)
        graph.add_edge("hub", "n2", EdgeType.REQUIRES)
        graph.add_edge("hub", "n3", EdgeType.REQUIRES)

        hubs = graph.find_hubs(top_n=1)
        assert hubs[0][0] == "hub"
        assert hubs[0][1] == 3

    def test_find_cycles(self):
        """Test finding cycles."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.CONCEPT, "A")
        graph.add_node("n2", NodeType.CONCEPT, "B")
        graph.add_node("n3", NodeType.CONCEPT, "C")
        graph.add_edge("n1", "n2", EdgeType.REQUIRES)
        graph.add_edge("n2", "n3", EdgeType.REQUIRES)
        graph.add_edge("n3", "n1", EdgeType.REQUIRES)

        cycles = graph.find_cycles()
        assert len(cycles) > 0


class TestPatternFactories:
    """Tests for pattern factory functions."""

    def test_investigation_graph(self):
        """Test creating investigation graph."""
        graph = create_investigation_graph("Why is the API slow?")
        assert graph.node_count() == 4  # question + 3 hypotheses
        assert graph.edge_count() == 3

    def test_investigation_graph_with_hypotheses(self):
        """Test investigation graph with custom hypotheses."""
        graph = create_investigation_graph(
            "Why crash?",
            initial_hypotheses=["Memory leak", "Race condition"]
        )
        assert graph.node_count() == 3  # question + 2 hypotheses
        assert graph.edge_count() == 2

    def test_decision_graph(self):
        """Test creating decision graph."""
        graph = create_decision_graph(
            "Choose auth method",
            ["OAuth", "JWT", "Session"]
        )
        # decision + 3 options + 6 evidence (2 per option)
        assert graph.node_count() == 10
        # 3 has_option + 6 supports/contradicts
        assert graph.edge_count() == 9

    def test_debug_graph(self):
        """Test creating debug graph."""
        graph = create_debug_graph("Server returns 500")
        # symptom + 3 observations + 3 causes
        assert graph.node_count() == 7
        # 3 observes + 3 suggests
        assert graph.edge_count() == 6

    def test_feature_graph(self):
        """Test creating feature graph."""
        graph = create_feature_graph(
            "Add authentication",
            "As a user, I want to log in"
        )
        # goal + story + 3 tasks
        assert graph.node_count() == 5
        # 1 motivates + 3 requires
        assert graph.edge_count() == 4

    def test_requirements_graph(self):
        """Test creating requirements graph."""
        graph = create_requirements_graph("Users need password reset")
        # need + 3 requirements + 3 specs + 3 designs
        assert graph.node_count() == 10
        # 3 requires + 3 refines + 3 implements
        assert graph.edge_count() == 9

    def test_analysis_graph(self):
        """Test creating analysis graph."""
        graph = create_analysis_graph("Performance issues")
        # topic + 3 aspects + 6 findings
        assert graph.node_count() == 10
        # 3 has_aspect + 6 observes
        assert graph.edge_count() == 9

    def test_analysis_graph_with_aspects(self):
        """Test analysis graph with custom aspects."""
        graph = create_analysis_graph(
            "Security review",
            aspects=["Authentication", "Authorization"]
        )
        # topic + 2 aspects + 4 findings
        assert graph.node_count() == 7

    def test_pattern_registry(self):
        """Test pattern registry contains all patterns."""
        assert "investigation" in PATTERN_REGISTRY
        assert "decision" in PATTERN_REGISTRY
        assert "debug" in PATTERN_REGISTRY
        assert "feature" in PATTERN_REGISTRY
        assert "requirements" in PATTERN_REGISTRY
        assert "analysis" in PATTERN_REGISTRY

    def test_create_pattern_graph(self):
        """Test creating graph via registry."""
        graph = create_pattern_graph(
            "investigation",
            question="Why is this failing?"
        )
        assert graph.node_count() == 4

    def test_create_pattern_graph_invalid_pattern(self):
        """Test invalid pattern name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_pattern_graph("invalid_pattern")
        assert "Unknown pattern" in str(exc_info.value)
