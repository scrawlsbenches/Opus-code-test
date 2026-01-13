"""
Behavioral tests for developers using the BaseGraph architecture.

Epic: Graph-Based Knowledge Modeling

As a developer building knowledge systems,
I want a unified graph abstraction with pluggable algorithms,
So that I can model domain-specific relationships without reinventing graph primitives.

This test suite validates the BaseGraph architecture from PR #283.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from cortical.graph import (
    # Core types
    BaseGraph,
    NodeBase,
    EdgeBase,
    NodeProtocol,
    EdgeProtocol,
    # Storage
    InMemoryGraphStorage,
    # Implementations
    SimpleGraph,
    DAGGraph,
    WeightedGraph,
    SimpleNode,
    SimpleEdge,
    WeightedEdge,
    # Algorithm mixins
    PageRankMixin,
    ClusteringMixin,
    SpreadingActivationMixin,
    CentralityMixin,
)


class TestDeveloperBuildsKnowledgeGraphs:
    """
    Epic: Graph-Based Knowledge Modeling

    As a developer building a knowledge system,
    I want a flexible graph API with domain-specific node types,
    So that I can model complex relationships while leveraging common algorithms.
    """

    def test_scenario_developer_creates_domain_specific_graph(self):
        """
        Scenario: Creating a custom graph for a specific domain

        Given a need to model research papers and their citations
        When I define custom node and edge types
        And I inherit from BaseGraph with factory methods
        Then I can create a strongly-typed citation graph
        And all standard graph operations work out of the box
        Because domain modeling shouldn't require reimplementing traversals.
        """
        # GIVEN a need to model research papers and their citations
        @dataclass
        class PaperNode(NodeBase):
            """A research paper with citation metadata."""
            author: str = ""
            year: int = 0
            citations: int = 0

        @dataclass
        class CitationEdge(EdgeBase):
            """A citation link between papers."""
            citation_context: str = ""

        class CitationGraph(BaseGraph[PaperNode, CitationEdge]):
            """Domain-specific graph for paper citations."""

            def _create_node(self, id: str, **kwargs: Any) -> PaperNode:
                return PaperNode(
                    id=id,
                    node_type=kwargs.get("node_type", "paper"),
                    content=kwargs.get("content", ""),
                    author=kwargs.get("author", ""),
                    year=kwargs.get("year", 0),
                    citations=kwargs.get("citations", 0),
                )

            def _create_edge(
                self, source_id: str, target_id: str, edge_type: str = "", **kwargs
            ) -> CitationEdge:
                return CitationEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type or "CITES",
                    weight=kwargs.get("weight", 1.0),
                    citation_context=kwargs.get("citation_context", ""),
                )

        # WHEN I define custom node and edge types
        graph = CitationGraph()

        # AND I create nodes with domain-specific attributes
        graph.add_node(
            "paper_1",
            content="Attention Is All You Need",
            author="Vaswani et al.",
            year=2017,
            citations=50000,
        )
        graph.add_node(
            "paper_2",
            content="BERT: Pre-training of Deep Bidirectional Transformers",
            author="Devlin et al.",
            year=2018,
            citations=30000,
        )
        graph.add_edge(
            "paper_2", "paper_1",
            citation_context="Building on the transformer architecture...",
        )

        # THEN I can create a strongly-typed citation graph
        paper1 = graph.get_node("paper_1")
        assert paper1 is not None
        assert isinstance(paper1, PaperNode)
        assert paper1.author == "Vaswani et al."
        assert paper1.citations == 50000

        # AND all standard graph operations work out of the box
        path = graph.shortest_path("paper_2", "paper_1")
        assert path == ["paper_2", "paper_1"]

        neighbors = graph.neighbors("paper_2", direction="out")
        assert neighbors == ["paper_1"]

    def test_scenario_developer_composes_algorithms_selectively(self):
        """
        Scenario: Adding only needed algorithms to a graph

        Given a lightweight graph that doesn't need all algorithms
        When I create a graph with only the PageRank mixin
        Then I have PageRank capability without clustering overhead
        And my graph class remains focused and minimal
        Because not every graph needs community detection.
        """
        # GIVEN a lightweight graph that doesn't need all algorithms
        class PageRankOnlyGraph(BaseGraph[SimpleNode, SimpleEdge], PageRankMixin):
            """Graph with only PageRank algorithm."""

            def _create_node(self, id: str, **kwargs: Any) -> SimpleNode:
                return SimpleNode(id=id, node_type=kwargs.get("node_type", ""))

            def _create_edge(
                self, source_id: str, target_id: str, edge_type: str = "", **kwargs
            ) -> SimpleEdge:
                return SimpleEdge(
                    source_id=source_id, target_id=target_id, edge_type=edge_type
                )

        # WHEN I create a graph with only the PageRank mixin
        graph = PageRankOnlyGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")

        # THEN I have PageRank capability
        pr = graph.compute_pagerank()
        assert len(pr) == 3
        assert all(0 <= v <= 1 for v in pr.values())

        # AND my graph class remains focused (no clustering method)
        assert hasattr(graph, "compute_pagerank")
        assert not hasattr(graph, "label_propagation")

    def test_scenario_developer_uses_dag_for_task_dependencies(self):
        """
        Scenario: Modeling task dependencies with a DAG

        Given a project with tasks that have dependencies
        When I model tasks in a DAGGraph
        And I try to add a circular dependency
        Then the graph rejects the cyclic edge
        And I can get the correct task execution order
        Because circular dependencies would cause deadlocks.
        """
        # GIVEN a project with tasks that have dependencies
        dag = DAGGraph()
        dag.add_node("design", content="Design the API")
        dag.add_node("implement", content="Implement the feature")
        dag.add_node("test", content="Write tests")
        dag.add_node("deploy", content="Deploy to production")

        # WHEN I model tasks in a DAGGraph
        dag.add_edge("design", "implement")
        dag.add_edge("implement", "test")
        dag.add_edge("test", "deploy")

        # AND I try to add a circular dependency
        with pytest.raises(ValueError, match="cycle"):
            dag.add_edge("deploy", "design")

        # THEN the graph rejects the cyclic edge (caught above)
        assert dag.edge_count == 3

        # AND I can get the correct task execution order
        order = dag.topological_sort()
        assert order.index("design") < order.index("implement")
        assert order.index("implement") < order.index("test")
        assert order.index("test") < order.index("deploy")

        # AND I can see what tasks are blocked
        blocked = dag.blocked_by("deploy")
        assert blocked == {"design", "implement", "test"}

        # AND I can see which tasks are ready to start
        ready = dag.ready_tasks(completed={"design", "implement"})
        assert "test" in ready
        assert "deploy" not in ready

    def test_scenario_developer_finds_shortest_weighted_path(self):
        """
        Scenario: Finding optimal routes in a weighted graph

        Given a network with weighted connections
        When I add paths with different costs
        And I query for the shortest path
        Then Dijkstra finds the lowest-cost route
        Even if it requires more hops
        Because cheaper paths are preferable to shorter ones.
        """
        # GIVEN a network with weighted connections
        graph = WeightedGraph()
        graph.add_node("A", content="Start")
        graph.add_node("B", content="Via B")
        graph.add_node("C", content="Via C")
        graph.add_node("D", content="End")

        # WHEN I add paths with different costs
        # Direct path A -> D (expensive: cost 10)
        graph.add_edge("A", "D", cost=10)

        # Indirect path A -> B -> C -> D (cheap: cost 1+1+1=3)
        graph.add_edge("A", "B", cost=1)
        graph.add_edge("B", "C", cost=1)
        graph.add_edge("C", "D", cost=1)

        # AND I query for the shortest weighted path
        path, total_cost = graph.dijkstra("A", "D")

        # THEN Dijkstra finds the lowest-cost route
        assert path == ["A", "B", "C", "D"]
        assert total_cost == 3

    def test_scenario_developer_discovers_communities(self):
        """
        Scenario: Finding communities in a social network

        Given a social network with distinct groups
        When I run community detection
        Then closely connected users cluster together
        And inter-group connections are identified
        Because understanding community structure aids recommendation.
        """
        # GIVEN a social network with distinct groups
        graph = SimpleGraph()

        # Group 1: Engineering team
        for user in ["alice", "bob", "charlie"]:
            graph.add_node(user, node_type="engineer")
        graph.add_edge("alice", "bob", bidirectional=True)
        graph.add_edge("bob", "charlie", bidirectional=True)
        graph.add_edge("alice", "charlie", bidirectional=True)

        # Group 2: Marketing team
        for user in ["diana", "eve"]:
            graph.add_node(user, node_type="marketing")
        graph.add_edge("diana", "eve", bidirectional=True)

        # Weak inter-group connection
        graph.add_edge("charlie", "diana", weight=0.1)

        # WHEN I run community detection
        labels = graph.label_propagation(seed=42)
        clusters = graph.get_clusters(seed=42)

        # THEN closely connected users cluster together
        assert len(labels) == 5
        assert len(clusters) >= 1  # At least one cluster identified

        # Members of same dense group tend to share labels
        # (Probabilistic, so we just verify structure)
        assert labels["alice"] == labels["bob"] or labels["bob"] == labels["charlie"]

    def test_scenario_developer_spreads_activation_for_recommendation(self):
        """
        Scenario: Using spreading activation for content recommendation

        Given a content graph with user interests
        When a user engages with one topic
        And I spread activation through related concepts
        Then related content receives activation scores
        And distant content gets progressively lower scores
        Because related recommendations should be weighted by relevance.
        """
        # GIVEN a content graph with user interests
        graph = SimpleGraph()

        # Create a content taxonomy
        graph.add_node("machine_learning", node_type="topic")
        graph.add_node("neural_networks", node_type="topic")
        graph.add_node("deep_learning", node_type="topic")
        graph.add_node("nlp", node_type="topic")
        graph.add_node("computer_vision", node_type="topic")

        # Connect related topics
        graph.add_edge("machine_learning", "neural_networks", weight=0.9)
        graph.add_edge("neural_networks", "deep_learning", weight=0.95)
        graph.add_edge("deep_learning", "nlp", weight=0.7)
        graph.add_edge("deep_learning", "computer_vision", weight=0.7)

        # WHEN a user engages with one topic
        # AND I spread activation through related concepts
        activations = graph.spread_activation(
            source_id="machine_learning",
            initial_activation=1.0,
            decay=0.5,
            max_hops=3,
        )

        # THEN related content receives activation scores
        assert "machine_learning" in activations
        assert "neural_networks" in activations
        assert "deep_learning" in activations

        # AND distant content gets progressively lower scores
        assert activations["machine_learning"] > activations["neural_networks"]
        assert activations["neural_networks"] > activations["deep_learning"]

    def test_scenario_developer_serializes_graph_for_persistence(self):
        """
        Scenario: Saving and loading a graph state

        Given a graph with nodes and edges
        When I serialize it to a dictionary
        And I create a new graph from that dictionary
        Then all nodes and edges are restored
        And I can continue working with the graph
        Because graphs need to persist across sessions.
        """
        # GIVEN a graph with nodes and edges
        original = SimpleGraph()
        original.add_node("concept_a", content="Neural Networks", node_type="concept")
        original.add_node("concept_b", content="Machine Learning", node_type="concept")
        original.add_edge("concept_a", "concept_b", edge_type="IS_A", weight=0.95)

        # WHEN I serialize it to a dictionary
        data = original.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        # AND I create a new graph from that dictionary
        restored = SimpleGraph.from_dict(data)

        # THEN all nodes and edges are restored
        assert restored.node_count == 2
        assert restored.edge_count == 1

        node_a = restored.get_node("concept_a")
        assert node_a is not None
        assert node_a.content == "Neural Networks"

        edge = restored.get_edge("concept_a", "concept_b", "IS_A")
        assert edge is not None
        assert edge.weight == 0.95

        # AND I can continue working with the graph
        restored.add_node("concept_c", content="Deep Learning")
        restored.add_edge("concept_b", "concept_c", edge_type="CONTAINS")
        assert restored.node_count == 3
        assert restored.edge_count == 2

    def test_scenario_developer_extracts_subgraph_for_analysis(self):
        """
        Scenario: Extracting a focused subgraph for detailed analysis

        Given a large graph with multiple topics
        When I identify nodes of interest
        And I extract a subgraph containing only those nodes
        Then the subgraph contains only relevant nodes and edges
        And I can run algorithms on the focused subset
        Because analyzing subsets is faster than the full graph.
        """
        # GIVEN a large graph with multiple topics
        graph = SimpleGraph()

        # Topic 1: AI
        graph.add_node("ai", node_type="topic")
        graph.add_node("ml", node_type="subtopic")
        graph.add_node("dl", node_type="subtopic")
        graph.add_edge("ai", "ml")
        graph.add_edge("ml", "dl")

        # Topic 2: Databases (separate)
        graph.add_node("databases", node_type="topic")
        graph.add_node("sql", node_type="subtopic")
        graph.add_edge("databases", "sql")

        assert graph.node_count == 5
        assert graph.edge_count == 3

        # WHEN I identify nodes of interest
        ai_nodes = {"ai", "ml", "dl"}

        # AND I extract a subgraph containing only those nodes
        ai_subgraph = graph.subgraph(ai_nodes)

        # THEN the subgraph contains only relevant nodes and edges
        assert ai_subgraph.node_count == 3
        assert ai_subgraph.edge_count == 2
        assert not ai_subgraph.has_node("databases")
        assert not ai_subgraph.has_node("sql")

        # AND I can run algorithms on the focused subset
        pr = ai_subgraph.compute_pagerank()
        assert len(pr) == 3
        assert "ai" in pr
        assert "databases" not in pr


class TestDeveloperUsesProtocolsForCompatibility:
    """
    Epic: Gradual Migration to Unified Graph

    As a developer with existing graph code,
    I want to use protocols for structural typing,
    So that I can integrate legacy code without rewrites.
    """

    def test_scenario_developer_uses_existing_class_as_node(self):
        """
        Scenario: Using a legacy dataclass as a graph node

        Given an existing dataclass that has an 'id' field
        When I check if it satisfies NodeProtocol
        Then the protocol check passes without inheritance
        Because structural typing enables gradual migration.
        """
        # GIVEN an existing dataclass that has an 'id' field
        @dataclass
        class LegacyDocument:
            id: str
            title: str
            node_type: str = "document"
            content: str = ""

        # WHEN I check if it satisfies NodeProtocol
        doc = LegacyDocument(id="doc_1", title="My Paper")

        # THEN the protocol check passes without inheritance
        assert isinstance(doc, NodeProtocol)
        assert doc.id == "doc_1"
        assert doc.node_type == "document"

    def test_scenario_developer_uses_existing_class_as_edge(self):
        """
        Scenario: Using a legacy dataclass as a graph edge

        Given an existing relationship dataclass
        When I check if it satisfies EdgeProtocol
        Then the protocol check passes
        And I can use it in graph algorithms expecting edges
        Because protocols enable duck typing for edges.
        """
        # GIVEN an existing relationship dataclass
        @dataclass
        class LegacyRelation:
            source_id: str
            target_id: str
            edge_type: str = "RELATES"
            weight: float = 1.0
            confidence: float = 0.9

        # WHEN I check if it satisfies EdgeProtocol
        rel = LegacyRelation(source_id="A", target_id="B")

        # THEN the protocol check passes
        assert isinstance(rel, EdgeProtocol)
        assert rel.source_id == "A"
        assert rel.target_id == "B"
        assert rel.weight == 1.0


class TestDeveloperUsesStorageBackends:
    """
    Epic: Pluggable Graph Storage

    As a developer deploying to different environments,
    I want pluggable storage backends,
    So that I can use in-memory for tests and persistent storage for production.
    """

    def test_scenario_developer_injects_custom_storage(self):
        """
        Scenario: Injecting a custom storage backend

        Given a custom storage backend implementation
        When I inject it into a BaseGraph
        Then all operations use the custom storage
        And I can verify the storage was used correctly
        Because dependency injection enables testability.
        """
        # GIVEN a custom storage backend implementation
        class TrackedStorage(InMemoryGraphStorage):
            """Storage that tracks all operations."""

            def __init__(self):
                super().__init__()
                self.operations = []

            def add_node(self, node):
                self.operations.append(("add_node", node.id))
                super().add_node(node)

            def add_edge(self, edge):
                self.operations.append(("add_edge", edge.source_id, edge.target_id))
                super().add_edge(edge)

        # WHEN I inject it into a SimpleGraph
        storage = TrackedStorage()
        graph = SimpleGraph(storage=storage)

        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        # THEN all operations use the custom storage
        assert ("add_node", "A") in storage.operations
        assert ("add_node", "B") in storage.operations
        assert ("add_edge", "A", "B") in storage.operations

        # AND I can verify the storage was used correctly
        assert len(storage.operations) == 3


class TestDeveloperAnalyzesGraphStructure:
    """
    Epic: Graph Analysis and Metrics

    As a data scientist analyzing graph structure,
    I want built-in analysis methods,
    So that I can understand connectivity patterns quickly.
    """

    def test_scenario_analyst_finds_hub_nodes(self):
        """
        Scenario: Identifying the most connected nodes

        Given a graph with varying node connectivity
        When I search for hub nodes
        Then I get nodes ranked by connection count
        And the most connected node appears first
        Because hubs are often influential in networks.
        """
        # GIVEN a graph with varying node connectivity
        graph = SimpleGraph()

        # Create a hub-and-spoke pattern
        graph.add_node("hub", node_type="central")
        for i in range(10):
            spoke = f"spoke_{i}"
            graph.add_node(spoke, node_type="peripheral")
            graph.add_edge("hub", spoke)

        # Add some inter-spoke connections
        graph.add_edge("spoke_0", "spoke_1")
        graph.add_edge("spoke_1", "spoke_2")

        # WHEN I search for hub nodes
        hubs = graph.find_hubs(top_n=3)

        # THEN I get nodes ranked by connection count
        assert len(hubs) == 3
        assert hubs[0][0] == "hub"
        assert hubs[0][1] == 10  # 10 outgoing edges

        # AND the most connected node appears first
        assert hubs[0][1] >= hubs[1][1] >= hubs[2][1]

    def test_scenario_analyst_identifies_connected_components(self):
        """
        Scenario: Finding disconnected subgraphs

        Given a graph with multiple disconnected components
        When I identify connected components
        Then I get separate sets of connected nodes
        And isolated nodes form their own components
        Because understanding connectivity is essential for analysis.
        """
        # GIVEN a graph with multiple disconnected components
        graph = SimpleGraph()

        # Component 1: Triangle
        graph.add_node("A1")
        graph.add_node("A2")
        graph.add_node("A3")
        graph.add_edge("A1", "A2")
        graph.add_edge("A2", "A3")
        graph.add_edge("A3", "A1")

        # Component 2: Line
        graph.add_node("B1")
        graph.add_node("B2")
        graph.add_edge("B1", "B2")

        # Component 3: Isolated node
        graph.add_node("C1")

        # WHEN I identify connected components
        components = graph.connected_components()

        # THEN I get separate sets of connected nodes
        assert len(components) == 3

        # AND isolated nodes form their own components
        component_sizes = sorted([len(c) for c in components])
        assert component_sizes == [1, 2, 3]

        # Verify each component's contents
        found_triangle = any({"A1", "A2", "A3"} == c for c in components)
        found_line = any({"B1", "B2"} == c for c in components)
        found_isolated = any({"C1"} == c for c in components)

        assert found_triangle
        assert found_line
        assert found_isolated

    def test_scenario_analyst_calculates_centrality_measures(self):
        """
        Scenario: Computing multiple centrality metrics

        Given a graph representing information flow
        When I compute different centrality measures
        Then I get complementary views of node importance
        And I can identify different types of important nodes
        Because different measures reveal different structural roles.
        """
        # GIVEN a graph representing information flow
        graph = SimpleGraph()

        # Create a bow-tie structure:
        # A -> B -> C -> D -> E
        #      ^    |
        #      +----+  (B-C bidirectional)
        for node in ["A", "B", "C", "D", "E"]:
            graph.add_node(node)

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "B")  # Bidirectional with B
        graph.add_edge("C", "D")
        graph.add_edge("D", "E")

        # WHEN I compute different centrality measures
        degree_centrality = graph.degree_centrality(direction="both")
        closeness_centrality = graph.closeness_centrality()

        # THEN I get complementary views of node importance
        assert len(degree_centrality) == 5
        assert len(closeness_centrality) == 5

        # B and C should have high degree (2 connections each to each other)
        # C should have high closeness (center of the structure)
        assert degree_centrality["B"] > degree_centrality["A"]
        assert degree_centrality["C"] > degree_centrality["E"]
