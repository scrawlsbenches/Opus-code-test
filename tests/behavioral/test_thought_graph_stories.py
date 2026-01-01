"""
Behavioral Tests for ThoughtGraph: Network-Based Reasoning.

This module tests the graph-based representation of reasoning processes
we built ourselves for managing complex thought relationships.

Epic: Researcher maps complex reasoning networks
Story: As a researcher analyzing reasoning patterns,
       I want graph-based thought representation we built from scratch,
       So that I can visualize and analyze reasoning networks we control.
"""

import pytest
from cortical.reasoning.graph_of_thought import NodeType, EdgeType
from cortical.reasoning.thought_graph import ThoughtGraph


class TestResearcherMapsComplexReasoningNetworks:
    """
    Epic: Researcher Maps Complex Reasoning Networks

    As a researcher analyzing custom reasoning patterns,
    I want graph structures we implemented ourselves,
    So that I can represent thought relationships we control.
    """

    def test_scenario_graph_captures_thought_relationships(self):
        """
        Scenario: Graph represents interconnected thoughts

        Given a reasoning problem we're analyzing
        When I build a graph of related thoughts
        Then nodes and edges capture the structure
        Because we implemented graph representation ourselves
        """
        # Given reasoning problem
        graph = ThoughtGraph()

        # When building graph
        q1 = graph.add_node("Q1", NodeType.QUESTION, "What indexing strategy to build ourselves?")
        h1 = graph.add_node("H1", NodeType.HYPOTHESIS, "Inverted index we implement")
        h2 = graph.add_node("H2", NodeType.HYPOTHESIS, "B-tree structure we build")

        graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)
        graph.add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.6)

        # Then structure is captured
        assert graph.node_count() == 3
        assert graph.edge_count() == 2
        assert len(graph.get_edges_from("Q1")) == 2

    def test_scenario_graph_traversal_explores_reasoning_paths(self):
        """
        Scenario: Traversal algorithms explore reasoning paths

        Given a thought graph we built
        When I traverse from a starting point
        Then paths through reasoning are discovered
        Because we implemented traversal ourselves
        """
        # Given thought graph
        graph = ThoughtGraph()
        graph.add_node("start", NodeType.QUESTION, "How to optimize custom search?")
        graph.add_node("approach1", NodeType.HYPOTHESIS, "Build caching layer ourselves")
        graph.add_node("approach2", NodeType.HYPOTHESIS, "Implement parallel processing")
        graph.add_node("detail1", NodeType.CONCEPT, "LRU cache we build")

        graph.add_edge("start", "approach1", EdgeType.EXPLORES)
        graph.add_edge("start", "approach2", EdgeType.EXPLORES)
        graph.add_edge("approach1", "detail1", EdgeType.CONTAINS)

        # When traversing
        bfs_order = graph.bfs("start")
        dfs_order = graph.dfs("start")

        # Then paths are discovered
        assert "start" in bfs_order
        assert "approach1" in bfs_order
        assert len(bfs_order) >= 3
        assert len(dfs_order) >= 3

    def test_scenario_graph_analysis_identifies_key_thoughts(self):
        """
        Scenario: Analysis identifies central concepts

        Given a complex reasoning graph we built
        When I analyze node importance
        Then hub nodes reveal key concepts
        Because we implement graph analysis ourselves
        """
        # Given complex graph
        graph = ThoughtGraph()

        # Create central concept with many connections
        graph.add_node("central", NodeType.CONCEPT, "Custom search architecture")
        for i in range(5):
            node_id = f"related{i}"
            graph.add_node(node_id, NodeType.CONCEPT, f"Aspect {i} we built")
            graph.add_edge("central", node_id, EdgeType.CONTAINS)

        # When analyzing importance
        hubs = graph.find_hubs(top_n=3)

        # Then central nodes identified
        assert len(hubs) > 0
        hub_ids = [hub_id for hub_id, degree in hubs]
        assert "central" in hub_ids

    def test_scenario_graph_detects_circular_reasoning(self):
        """
        Scenario: System detects circular reasoning patterns

        Given a thought graph with potential cycles
        When I check for circular dependencies
        Then cycles are detected
        Because we built cycle detection ourselves
        """
        # Given graph with cycle
        graph = ThoughtGraph()
        graph.add_node("A", NodeType.HYPOTHESIS, "Approach A we designed")
        graph.add_node("B", NodeType.HYPOTHESIS, "Approach B we built")
        graph.add_node("C", NodeType.HYPOTHESIS, "Approach C")

        graph.add_edge("A", "B", EdgeType.ENABLES)
        graph.add_edge("B", "C", EdgeType.ENABLES)
        graph.add_edge("C", "A", EdgeType.REQUIRES)  # Creates cycle

        # When checking for cycles
        cycles = graph.find_cycles()

        # Then cycle is detected
        assert len(cycles) > 0
        # Cycle should include all three nodes
        cycle_nodes = set(cycles[0])
        assert len(cycle_nodes & {"A", "B", "C"}) >= 2

    def test_scenario_graph_identifies_isolated_thoughts(self):
        """
        Scenario: Orphaned thoughts are identified for review

        Given a graph with disconnected thoughts
        When I search for orphans
        Then unconnected nodes are found
        Because we track graph connectivity ourselves
        """
        # Given graph with orphans
        graph = ThoughtGraph()
        graph.add_node("connected1", NodeType.CONCEPT, "Part of our architecture")
        graph.add_node("connected2", NodeType.CONCEPT, "Related concept")
        graph.add_node("orphan", NodeType.CONCEPT, "Isolated thought")

        graph.add_edge("connected1", "connected2", EdgeType.SIMILAR)

        # When finding orphans
        orphans = graph.find_orphans()

        # Then isolated nodes found
        assert "orphan" in orphans
        assert "connected1" not in orphans

    def test_scenario_graph_supports_clustering_related_thoughts(self):
        """
        Scenario: Related thoughts can be clustered

        Given thoughts with thematic groupings we identified
        When I create clusters
        Then related thoughts are organized
        Because we built clustering ourselves
        """
        # Given thematically grouped thoughts
        graph = ThoughtGraph()

        # Indexing cluster
        idx1 = graph.add_node("idx1", NodeType.CONCEPT, "Custom inverted index")
        idx2 = graph.add_node("idx2", NodeType.CONCEPT, "Index updates we handle")

        # Ranking cluster
        rank1 = graph.add_node("rank1", NodeType.CONCEPT, "TF-IDF we implemented")
        rank2 = graph.add_node("rank2", NodeType.CONCEPT, "PageRank we built")

        # When creating clusters
        indexing = graph.add_cluster("C1", "Indexing", {"idx1", "idx2"})
        ranking = graph.add_cluster("C2", "Ranking", {"rank1", "rank2"})

        # Then clusters organize thoughts
        assert graph.cluster_count() == 2
        assert indexing.contains_node("idx1")
        assert not indexing.contains_node("rank1")


class TestDeveloperBuildsReasoningVisualization:
    """
    Epic: Developer Builds Reasoning Visualization

    As a developer building visualization tools,
    I want export formats we implemented,
    So that I can render reasoning graphs we control.
    """

    def test_scenario_graph_exports_to_mermaid_diagram(self):
        """
        Scenario: Graph exports to Mermaid for documentation

        Given a reasoning graph we need to document
        When I export to Mermaid format
        Then diagram syntax is generated
        Because we built Mermaid export ourselves
        """
        # Given reasoning graph
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "What to build?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Custom indexer")
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES)

        # When exporting to Mermaid
        mermaid = graph.to_mermaid()

        # Then valid Mermaid syntax generated
        assert "graph TD" in mermaid
        assert "Q1" in mermaid
        assert "H1" in mermaid
        assert "explores" in mermaid.lower()

    def test_scenario_graph_exports_to_graphviz_format(self):
        """
        Scenario: Graph exports to Graphviz for rendering

        Given a complex graph needing visualization
        When I export to DOT format
        Then Graphviz can render it
        Because we implemented DOT export ourselves
        """
        # Given complex graph
        graph = ThoughtGraph()
        graph.add_node("C1", NodeType.CONCEPT, "Custom architecture")
        graph.add_node("C2", NodeType.CONCEPT, "Implementation details")
        graph.add_edge("C1", "C2", EdgeType.CONTAINS)

        # When exporting to DOT
        dot = graph.to_dot()

        # Then valid DOT syntax generated
        assert "digraph ThoughtGraph" in dot
        assert "C1" in dot
        assert "C2" in dot
        assert "->" in dot

    def test_scenario_graph_generates_ascii_tree_for_terminal(self):
        """
        Scenario: ASCII tree enables terminal inspection

        Given a graph we're debugging
        When I request ASCII representation
        Then terminal-friendly tree is generated
        Because we built text visualization ourselves
        """
        # Given graph
        graph = ThoughtGraph()
        graph.add_node("root", NodeType.TASK, "Build custom system")
        graph.add_node("sub1", NodeType.TASK, "Implement indexing")
        graph.add_node("sub2", NodeType.TASK, "Build query parser")
        graph.add_edge("root", "sub1", EdgeType.CONTAINS)
        graph.add_edge("root", "sub2", EdgeType.CONTAINS)

        # When generating ASCII
        ascii_tree = graph.to_ascii("root")

        # Then readable tree generated
        assert "[TASK]" in ascii_tree
        assert "Build custom system" in ascii_tree
        # Should show tree structure
        assert "└──" in ascii_tree or "├──" in ascii_tree


class TestReasonerManipulatesThoughtStructures:
    """
    Epic: Reasoner Manipulates Thought Structures

    As a system managing evolving reasoning,
    I want graph manipulation operations,
    So that I can refine thought structures we control.
    """

    def test_scenario_merging_similar_concepts_reduces_redundancy(self):
        """
        Scenario: Similar thoughts can be merged

        Given duplicate or similar concepts we identified
        When I merge them into one
        Then redundancy is eliminated
        Because we built graph manipulation ourselves
        """
        # Given duplicate concepts
        graph = ThoughtGraph()
        graph.add_node("C1", NodeType.CONCEPT, "Custom indexing")
        graph.add_node("C2", NodeType.CONCEPT, "Index implementation")
        graph.add_node("related", NodeType.CONCEPT, "Related topic")

        graph.add_edge("C1", "related", EdgeType.SIMILAR)
        graph.add_edge("C2", "related", EdgeType.SIMILAR)

        # When merging
        merged = graph.merge_nodes("C1", "C2", "merged")

        # Then redundancy eliminated
        assert merged.id == "merged"
        assert "C1" not in graph.nodes or "C2" not in graph.nodes
        # Edges should be redirected
        assert any(e.source_id == "merged" or e.target_id == "merged"
                  for e in graph.edges)

    def test_scenario_splitting_complex_thoughts_improves_clarity(self):
        """
        Scenario: Complex thoughts can be decomposed

        Given an overly complex thought node
        When I split it into components
        Then clarity improves
        Because we built decomposition ourselves
        """
        # Given complex node
        graph = ThoughtGraph()
        graph.add_node("complex", NodeType.CONCEPT,
                      "Custom indexing and ranking and query processing")

        # When splitting
        node1, node2 = graph.split_node(
            "complex",
            "part1", "part2",
            "Custom indexing we built",
            "Ranking algorithm we designed"
        )

        # Then decomposed
        assert node1.id == "part1"
        assert node2.id == "part2"
        assert "complex" not in graph.nodes

    def test_scenario_finding_shortest_reasoning_path(self):
        """
        Scenario: Shortest path reveals direct reasoning

        Given alternative reasoning paths we built
        When I find the shortest path
        Then most direct reasoning is revealed
        Because we implemented pathfinding ourselves
        """
        # Given alternative paths
        graph = ThoughtGraph()
        graph.add_node("start", NodeType.QUESTION, "Problem to solve")
        graph.add_node("mid1", NodeType.HYPOTHESIS, "Indirect approach")
        graph.add_node("mid2", NodeType.HYPOTHESIS, "Another step")
        graph.add_node("end", NodeType.DECISION, "Final decision")

        # Direct path
        graph.add_edge("start", "end", EdgeType.ANSWERS)

        # Indirect path
        graph.add_edge("start", "mid1", EdgeType.EXPLORES)
        graph.add_edge("mid1", "mid2", EdgeType.ENABLES)
        graph.add_edge("mid2", "end", EdgeType.SUGGESTS)

        # When finding shortest path
        path = graph.shortest_path("start", "end")

        # Then direct path found
        assert path is not None
        assert len(path) == 2  # start -> end
        assert path[0] == "start"
        assert path[1] == "end"
