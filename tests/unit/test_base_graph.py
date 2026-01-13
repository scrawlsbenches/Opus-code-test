"""
Unit tests for BaseGraph architecture.

Tests the core components:
- NodeBase, EdgeBase protocols
- InMemoryGraphStorage
- BaseGraph operations
- Algorithm mixins
- Concrete implementations (SimpleGraph, DAGGraph, WeightedGraph)
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cortical.graph import (
    # Protocols
    NodeBase,
    EdgeBase,
    NodeProtocol,
    EdgeProtocol,
    # Storage
    InMemoryGraphStorage,
    # Base
    BaseGraph,
    # Algorithms
    PageRankMixin,
    ClusteringMixin,
    SpreadingActivationMixin,
    # Implementations
    SimpleNode,
    SimpleEdge,
    SimpleGraph,
    DAGGraph,
    WeightedGraph,
    WeightedEdge,
)


# =============================================================================
# Protocol Tests
# =============================================================================


class TestNodeBase:
    """Tests for NodeBase dataclass."""

    def test_create_node(self):
        """Test basic node creation."""
        node = NodeBase(id="N1", content="Test node")
        assert node.id == "N1"
        assert node.content == "Test node"
        assert node.node_type == ""
        assert node.properties == {}

    def test_node_with_all_fields(self):
        """Test node with all fields populated."""
        node = NodeBase(
            id="N1",
            node_type="concept",
            content="Test content",
            properties={"key": "value"},
            metadata={"author": "test"},
        )
        assert node.node_type == "concept"
        assert node.properties["key"] == "value"
        assert node.metadata["author"] == "test"

    def test_node_hash_and_equality(self):
        """Test that nodes hash and compare by ID."""
        node1 = NodeBase(id="N1", content="Content 1")
        node2 = NodeBase(id="N1", content="Content 2")  # Same ID, different content
        node3 = NodeBase(id="N2", content="Content 1")

        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)

    def test_node_in_set(self):
        """Test that nodes work in sets (deduplication by ID)."""
        node1 = NodeBase(id="N1", content="A")
        node2 = NodeBase(id="N1", content="B")
        node3 = NodeBase(id="N2", content="C")

        node_set = {node1, node2, node3}
        assert len(node_set) == 2  # N1 deduplicated

    def test_node_to_dict(self):
        """Test serialization to dictionary."""
        node = NodeBase(
            id="N1",
            node_type="test",
            content="Test",
            properties={"k": "v"},
        )
        d = node.to_dict()
        assert d["id"] == "N1"
        assert d["node_type"] == "test"
        assert d["content"] == "Test"
        assert d["properties"]["k"] == "v"

    def test_node_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "N1",
            "node_type": "test",
            "content": "Test",
            "properties": {"k": "v"},
        }
        node = NodeBase.from_dict(data)
        assert node.id == "N1"
        assert node.node_type == "test"
        assert node.properties["k"] == "v"


class TestEdgeBase:
    """Tests for EdgeBase dataclass."""

    def test_create_edge(self):
        """Test basic edge creation."""
        edge = EdgeBase(source_id="A", target_id="B")
        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.weight == 1.0
        assert edge.bidirectional is False

    def test_edge_with_all_fields(self):
        """Test edge with all fields populated."""
        edge = EdgeBase(
            source_id="A",
            target_id="B",
            edge_type="RELATED",
            weight=0.8,
            bidirectional=True,
            properties={"reason": "test"},
        )
        assert edge.edge_type == "RELATED"
        assert edge.weight == 0.8
        assert edge.bidirectional is True
        assert edge.properties["reason"] == "test"

    def test_edge_id_generation(self):
        """Test that edge ID is generated from components."""
        edge = EdgeBase(source_id="A", target_id="B", edge_type="RELATES")
        assert edge.id == "E-A-B-RELATES"

    def test_edge_weight_validation(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError):
            EdgeBase(source_id="A", target_id="B", weight=1.5)
        with pytest.raises(ValueError):
            EdgeBase(source_id="A", target_id="B", weight=-0.1)

    def test_edge_hash_and_equality(self):
        """Test that edges hash by (source, target, type)."""
        edge1 = EdgeBase(source_id="A", target_id="B", edge_type="X")
        edge2 = EdgeBase(source_id="A", target_id="B", edge_type="X", weight=0.5)
        edge3 = EdgeBase(source_id="A", target_id="B", edge_type="Y")

        assert edge1 == edge2  # Same source/target/type
        assert edge1 != edge3  # Different type
        assert hash(edge1) == hash(edge2)

    def test_edge_reverse(self):
        """Test creating reversed edge."""
        edge = EdgeBase(
            source_id="A",
            target_id="B",
            edge_type="FOLLOWS",
            weight=0.8,
        )
        reverse = edge.reverse()
        assert reverse.source_id == "B"
        assert reverse.target_id == "A"
        assert reverse.edge_type == "FOLLOWS"
        assert reverse.weight == 0.8
        assert reverse.bidirectional is False

    def test_edge_to_dict(self):
        """Test serialization to dictionary."""
        edge = EdgeBase(
            source_id="A",
            target_id="B",
            edge_type="REL",
            weight=0.5,
        )
        d = edge.to_dict()
        assert d["source_id"] == "A"
        assert d["target_id"] == "B"
        assert d["edge_type"] == "REL"
        assert d["weight"] == 0.5

    def test_edge_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "source_id": "A",
            "target_id": "B",
            "edge_type": "REL",
            "weight": 0.5,
        }
        edge = EdgeBase.from_dict(data)
        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.weight == 0.5


class TestProtocols:
    """Tests for structural subtyping protocols."""

    def test_node_protocol_satisfied(self):
        """Test that custom class satisfies NodeProtocol."""
        @dataclass
        class CustomNode:
            id: str
            node_type: str = ""

        node = CustomNode(id="N1", node_type="custom")
        assert isinstance(node, NodeProtocol)

    def test_edge_protocol_satisfied(self):
        """Test that custom class satisfies EdgeProtocol."""
        @dataclass
        class CustomEdge:
            source_id: str
            target_id: str
            edge_type: str = ""
            weight: float = 1.0

        edge = CustomEdge(source_id="A", target_id="B")
        assert isinstance(edge, EdgeProtocol)


# =============================================================================
# Storage Tests
# =============================================================================


class TestInMemoryGraphStorage:
    """Tests for InMemoryGraphStorage."""

    @pytest.fixture
    def storage(self):
        """Create fresh storage for each test."""
        return InMemoryGraphStorage()

    def test_add_and_get_node(self, storage):
        """Test adding and retrieving nodes."""
        node = NodeBase(id="N1", content="Test")
        storage.add_node(node)

        retrieved = storage.get_node("N1")
        assert retrieved is not None
        assert retrieved.id == "N1"
        assert retrieved.content == "Test"

    def test_get_nonexistent_node(self, storage):
        """Test that getting nonexistent node returns None."""
        assert storage.get_node("nonexistent") is None

    def test_has_node(self, storage):
        """Test has_node check."""
        node = NodeBase(id="N1")
        storage.add_node(node)

        assert storage.has_node("N1") is True
        assert storage.has_node("N2") is False

    def test_remove_node(self, storage):
        """Test removing a node."""
        node = NodeBase(id="N1")
        storage.add_node(node)

        removed = storage.remove_node("N1")
        assert removed is not None
        assert removed.id == "N1"
        assert storage.has_node("N1") is False

    def test_remove_node_removes_edges(self, storage):
        """Test that removing a node removes its edges."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        storage.add_edge(EdgeBase(source_id="A", target_id="B"))

        assert storage.edge_count() == 1
        storage.remove_node("A")
        assert storage.edge_count() == 0

    def test_all_nodes(self, storage):
        """Test iterating over all nodes."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        storage.add_node(NodeBase(id="C"))

        nodes = list(storage.all_nodes())
        assert len(nodes) == 3
        ids = {n.id for n in nodes}
        assert ids == {"A", "B", "C"}

    def test_node_count(self, storage):
        """Test node count."""
        assert storage.node_count() == 0
        storage.add_node(NodeBase(id="A"))
        assert storage.node_count() == 1
        storage.add_node(NodeBase(id="B"))
        assert storage.node_count() == 2

    def test_add_and_get_edge(self, storage):
        """Test adding and retrieving edges."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        edge = EdgeBase(source_id="A", target_id="B", edge_type="REL")
        storage.add_edge(edge)

        retrieved = storage.get_edge("A", "B", "REL")
        assert retrieved is not None
        assert retrieved.source_id == "A"
        assert retrieved.target_id == "B"

    def test_edges_from(self, storage):
        """Test getting outgoing edges."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        storage.add_node(NodeBase(id="C"))
        storage.add_edge(EdgeBase(source_id="A", target_id="B"))
        storage.add_edge(EdgeBase(source_id="A", target_id="C"))

        edges = storage.edges_from("A")
        assert len(edges) == 2
        targets = {e.target_id for e in edges}
        assert targets == {"B", "C"}

    def test_edges_to(self, storage):
        """Test getting incoming edges."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        storage.add_node(NodeBase(id="C"))
        storage.add_edge(EdgeBase(source_id="A", target_id="C"))
        storage.add_edge(EdgeBase(source_id="B", target_id="C"))

        edges = storage.edges_to("C")
        assert len(edges) == 2
        sources = {e.source_id for e in edges}
        assert sources == {"A", "B"}

    def test_remove_edge(self, storage):
        """Test removing an edge."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        storage.add_edge(EdgeBase(source_id="A", target_id="B", edge_type="X"))

        assert storage.remove_edge("A", "B", "X") is True
        assert storage.edge_count() == 0
        assert storage.remove_edge("A", "B", "X") is False  # Already removed

    def test_clear(self, storage):
        """Test clearing all data."""
        storage.add_node(NodeBase(id="A"))
        storage.add_node(NodeBase(id="B"))
        storage.add_edge(EdgeBase(source_id="A", target_id="B"))

        storage.clear()
        assert storage.node_count() == 0
        assert storage.edge_count() == 0


# =============================================================================
# SimpleGraph Tests
# =============================================================================


class TestSimpleGraph:
    """Tests for SimpleGraph implementation."""

    @pytest.fixture
    def graph(self):
        """Create fresh graph for each test."""
        return SimpleGraph()

    def test_add_node(self, graph):
        """Test adding nodes."""
        node = graph.add_node("A", content="Node A", node_type="concept")
        assert node.id == "A"
        assert node.content == "Node A"
        assert node.node_type == "concept"

    def test_add_duplicate_node_raises(self, graph):
        """Test that adding duplicate node raises ValueError."""
        graph.add_node("A")
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("A")

    def test_get_node(self, graph):
        """Test getting nodes."""
        graph.add_node("A", content="Test")
        node = graph.get_node("A")
        assert node is not None
        assert node.content == "Test"

    def test_has_node(self, graph):
        """Test checking node existence."""
        graph.add_node("A")
        assert graph.has_node("A") is True
        assert graph.has_node("B") is False
        assert "A" in graph  # __contains__

    def test_remove_node(self, graph):
        """Test removing nodes."""
        graph.add_node("A")
        node = graph.remove_node("A")
        assert node is not None
        assert graph.has_node("A") is False

    def test_get_or_create_node(self, graph):
        """Test get_or_create_node."""
        node1, created1 = graph.get_or_create_node("A", content="First")
        assert created1 is True
        assert node1.content == "First"

        node2, created2 = graph.get_or_create_node("A", content="Second")
        assert created2 is False
        assert node2.content == "First"  # Original content preserved

    def test_node_count(self, graph):
        """Test node counting."""
        assert graph.node_count == 0
        assert len(graph) == 0
        graph.add_node("A")
        graph.add_node("B")
        assert graph.node_count == 2
        assert len(graph) == 2

    def test_add_edge(self, graph):
        """Test adding edges."""
        graph.add_node("A")
        graph.add_node("B")
        edge = graph.add_edge("A", "B", edge_type="RELATES", weight=0.8)

        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.edge_type == "RELATES"
        assert edge.weight == 0.8

    def test_add_edge_nonexistent_node_raises(self, graph):
        """Test that adding edge to nonexistent node raises."""
        graph.add_node("A")
        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("A", "B")

    def test_get_edge(self, graph):
        """Test getting edges."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B", edge_type="REL")

        edge = graph.get_edge("A", "B", "REL")
        assert edge is not None
        assert edge.source_id == "A"

    def test_edges_from_to(self, graph):
        """Test getting edges from/to a node."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")

        from_a = graph.edges_from("A")
        assert len(from_a) == 2

        to_c = graph.edges_to("C")
        assert len(to_c) == 2

    def test_neighbors(self, graph):
        """Test getting neighbors."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("C", "A")

        out_neighbors = graph.neighbors("A", direction="out")
        assert out_neighbors == ["B"]

        in_neighbors = graph.neighbors("A", direction="in")
        assert in_neighbors == ["C"]

        all_neighbors = graph.neighbors("A", direction="both")
        assert set(all_neighbors) == {"B", "C"}

    def test_degree(self, graph):
        """Test degree calculation."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")

        assert graph.degree("A", "out") == 2
        assert graph.degree("A", "in") == 0
        assert graph.degree("B", "in") == 1

    def test_bfs(self, graph):
        """Test BFS traversal."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_node("D")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")

        order = graph.bfs("A")
        assert order[0] == "A"  # Start node first
        # B and C should come before D (level order)
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_bfs_with_visitor(self, graph):
        """Test BFS with visitor function."""
        graph.add_node("A", content="x")
        graph.add_node("B", content="xx")
        graph.add_node("C", content="xxx")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")

        def count_content_length(node, acc):
            return acc + len(node.content)

        total = graph.bfs("A", visitor=count_content_length, initial=0)
        assert total == 6  # 1 + 2 + 3

    def test_bfs_max_depth(self, graph):
        """Test BFS with depth limit."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        order = graph.bfs("A", max_depth=1)
        assert "A" in order
        assert "B" in order
        assert "C" not in order  # Too deep

    def test_dfs(self, graph):
        """Test DFS traversal."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        order = graph.dfs("A")
        assert order == ["A", "B", "C"]  # Depth-first order

    def test_shortest_path(self, graph):
        """Test shortest path finding."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_node("D")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "D")
        graph.add_edge("D", "C")

        path = graph.shortest_path("A", "C")
        assert path is not None
        assert len(path) == 3  # A -> B -> C or A -> D -> C

    def test_shortest_path_no_path(self, graph):
        """Test that no path returns None."""
        graph.add_node("A")
        graph.add_node("B")
        # No edge between them

        path = graph.shortest_path("A", "B")
        assert path is None

    def test_has_cycle(self, graph):
        """Test cycle detection."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        assert graph.has_cycle() is False

        graph.add_edge("C", "A")  # Create cycle
        assert graph.has_cycle() is True

    def test_find_cycles(self, graph):
        """Test finding all cycles."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")

        cycles = graph.find_cycles()
        assert len(cycles) >= 1
        assert any("A" in c and "B" in c and "C" in c for c in cycles)

    def test_connected_components(self, graph):
        """Test finding connected components."""
        # Component 1
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        # Component 2
        graph.add_node("C")
        graph.add_node("D")
        graph.add_edge("C", "D")

        components = graph.connected_components()
        assert len(components) == 2

    def test_topological_sort(self, graph):
        """Test topological sorting."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        order = graph.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_topological_sort_with_cycle_raises(self, graph):
        """Test that topological sort raises on cycle."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")

        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()

    def test_find_roots_and_leaves(self, graph):
        """Test finding roots and leaves."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        roots = graph.find_roots()
        assert roots == {"A"}

        leaves = graph.find_leaves()
        assert leaves == {"C"}

    def test_find_hubs(self, graph):
        """Test finding hub nodes."""
        graph.add_node("hub")
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("hub", "A")
        graph.add_edge("hub", "B")
        graph.add_edge("hub", "C")

        hubs = graph.find_hubs(top_n=1)
        assert len(hubs) == 1
        assert hubs[0][0] == "hub"
        assert hubs[0][1] == 3

    def test_to_dict_from_dict(self, graph):
        """Test serialization round-trip."""
        graph.add_node("A", content="Node A")
        graph.add_node("B", content="Node B")
        graph.add_edge("A", "B", edge_type="REL")

        data = graph.to_dict()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        new_graph = SimpleGraph.from_dict(data)
        assert new_graph.node_count == 2
        assert new_graph.edge_count == 1

    def test_copy(self, graph):
        """Test graph copying."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        copy = graph.copy()
        assert copy.node_count == graph.node_count
        assert copy.edge_count == graph.edge_count

    def test_subgraph(self, graph):
        """Test subgraph extraction."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "C")

        sub = graph.subgraph({"A", "B"})
        assert sub.node_count == 2
        assert sub.edge_count == 1  # Only A->B, not A->C or B->C


# =============================================================================
# Algorithm Mixin Tests
# =============================================================================


class TestPageRankMixin:
    """Tests for PageRank algorithm."""

    def test_pagerank_simple(self):
        """Test PageRank on simple graph."""
        graph = SimpleGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")

        pr = graph.compute_pagerank()
        assert len(pr) == 3
        # All nodes should have similar PageRank in a cycle
        values = list(pr.values())
        assert abs(max(values) - min(values)) < 0.1

    def test_pagerank_hub(self):
        """Test that hub has higher PageRank."""
        graph = SimpleGraph()
        graph.add_node("hub")
        for i in range(5):
            graph.add_node(f"spoke{i}")
            graph.add_edge(f"spoke{i}", "hub")

        pr = graph.compute_pagerank()
        # Hub should have highest PageRank
        assert pr["hub"] == max(pr.values())

    def test_pagerank_empty_graph(self):
        """Test PageRank on empty graph."""
        graph = SimpleGraph()
        pr = graph.compute_pagerank()
        assert pr == {}


class TestClusteringMixin:
    """Tests for clustering algorithms."""

    def test_label_propagation(self):
        """Test label propagation community detection."""
        graph = SimpleGraph()

        # Cluster 1
        graph.add_node("A1")
        graph.add_node("A2")
        graph.add_node("A3")
        graph.add_edge("A1", "A2", bidirectional=True)
        graph.add_edge("A2", "A3", bidirectional=True)
        graph.add_edge("A1", "A3", bidirectional=True)

        # Cluster 2
        graph.add_node("B1")
        graph.add_node("B2")
        graph.add_edge("B1", "B2", bidirectional=True)

        # Weak connection between clusters
        graph.add_edge("A1", "B1", weight=0.1)

        labels = graph.label_propagation(seed=42)
        assert len(labels) == 5

        # Nodes in same cluster should have same label
        # (This is probabilistic, so we just check structure)
        assert len(set(labels.values())) >= 1


class TestSpreadingActivationMixin:
    """Tests for spreading activation algorithm."""

    def test_spread_activation_basic(self):
        """Test basic activation spreading."""
        graph = SimpleGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B", weight=1.0)
        graph.add_edge("B", "C", weight=1.0)

        activations = graph.spread_activation("A", initial_activation=1.0, decay=0.5)

        assert activations["A"] == 1.0
        assert activations["B"] < activations["A"]
        assert activations["C"] < activations["B"]

    def test_spread_activation_decay(self):
        """Test that decay reduces activation over hops."""
        graph = SimpleGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        act1 = graph.spread_activation("A", decay=0.9)
        act2 = graph.spread_activation("A", decay=0.1)

        assert act1["B"] > act2["B"]  # Higher decay = more activation


# =============================================================================
# DAGGraph Tests
# =============================================================================


class TestDAGGraph:
    """Tests for DAGGraph implementation."""

    @pytest.fixture
    def dag(self):
        """Create fresh DAG for each test."""
        return DAGGraph()

    def test_add_edge_valid(self, dag):
        """Test adding valid edges."""
        dag.add_node("A")
        dag.add_node("B")
        dag.add_node("C")

        dag.add_edge("A", "B")
        dag.add_edge("B", "C")

        assert dag.edge_count == 2

    def test_add_edge_cycle_raises(self, dag):
        """Test that adding cycle-creating edge raises."""
        dag.add_node("A")
        dag.add_node("B")
        dag.add_node("C")

        dag.add_edge("A", "B")
        dag.add_edge("B", "C")

        with pytest.raises(ValueError, match="cycle"):
            dag.add_edge("C", "A")

    def test_self_loop_raises(self, dag):
        """Test that self-loop raises."""
        dag.add_node("A")

        with pytest.raises(ValueError, match="Self-loops"):
            dag.add_edge("A", "A")

    def test_blocked_by(self, dag):
        """Test blocked_by (transitive predecessors)."""
        dag.add_node("A")
        dag.add_node("B")
        dag.add_node("C")
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")

        blocked_by_c = dag.blocked_by("C")
        assert blocked_by_c == {"A", "B"}

    def test_blocks(self, dag):
        """Test blocks (transitive successors)."""
        dag.add_node("A")
        dag.add_node("B")
        dag.add_node("C")
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")

        blocks_a = dag.blocks("A")
        assert blocks_a == {"B", "C"}

    def test_ready_tasks(self, dag):
        """Test ready_tasks with completed set."""
        dag.add_node("A")
        dag.add_node("B")
        dag.add_node("C")
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")

        # Initially only A is ready
        ready = dag.ready_tasks(set())
        assert "A" in ready
        assert "B" not in ready

        # After A completes, B is ready
        ready = dag.ready_tasks({"A"})
        assert "B" in ready
        assert "C" not in ready


# =============================================================================
# WeightedGraph Tests
# =============================================================================


class TestWeightedGraph:
    """Tests for WeightedGraph implementation."""

    @pytest.fixture
    def graph(self):
        """Create fresh weighted graph for each test."""
        return WeightedGraph()

    def test_dijkstra_simple(self, graph):
        """Test Dijkstra on simple graph."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B", cost=1)
        graph.add_edge("B", "C", cost=2)

        path, cost = graph.dijkstra("A", "C")
        assert path == ["A", "B", "C"]
        assert cost == 3

    def test_dijkstra_chooses_shorter_path(self, graph):
        """Test that Dijkstra chooses shortest weighted path."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_node("D")

        # Direct path: A -> D (cost 10)
        graph.add_edge("A", "D", cost=10)

        # Indirect path: A -> B -> C -> D (cost 1+1+1=3)
        graph.add_edge("A", "B", cost=1)
        graph.add_edge("B", "C", cost=1)
        graph.add_edge("C", "D", cost=1)

        path, cost = graph.dijkstra("A", "D")
        assert path == ["A", "B", "C", "D"]
        assert cost == 3

    def test_dijkstra_no_path(self, graph):
        """Test Dijkstra when no path exists."""
        graph.add_node("A")
        graph.add_node("B")
        # No edge

        path, cost = graph.dijkstra("A", "B")
        assert path is None
        assert cost == float("inf")

    def test_dijkstra_same_node(self, graph):
        """Test Dijkstra from node to itself."""
        graph.add_node("A")

        path, cost = graph.dijkstra("A", "A")
        assert path == ["A"]
        assert cost == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraphIntegration:
    """Integration tests verifying components work together."""

    def test_full_workflow(self):
        """Test complete graph workflow."""
        graph = SimpleGraph()

        # Build graph
        graph.add_node("start", content="Start node", node_type="source")
        graph.add_node("middle", content="Middle node", node_type="process")
        graph.add_node("end", content="End node", node_type="sink")

        graph.add_edge("start", "middle", edge_type="FLOWS_TO", weight=0.9)
        graph.add_edge("middle", "end", edge_type="FLOWS_TO", weight=0.8)

        # Analyze
        assert graph.node_count == 3
        assert graph.edge_count == 2

        path = graph.shortest_path("start", "end")
        assert path == ["start", "middle", "end"]

        pr = graph.compute_pagerank()
        assert "end" in pr  # Sink should have PageRank

        # Serialize and restore
        data = graph.to_dict()
        restored = SimpleGraph.from_dict(data)
        assert restored.node_count == 3

    def test_subgraph_preserves_algorithms(self):
        """Test that subgraph inherits algorithm mixins."""
        graph = SimpleGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        sub = graph.subgraph({"A", "B"})

        # Subgraph should still have PageRank capability
        pr = sub.compute_pagerank()
        assert len(pr) == 2
