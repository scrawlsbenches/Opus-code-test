"""
Tests for ThoughtGraph visualization methods.

Tests the three visualization methods:
- to_mermaid() - Mermaid diagram format
- to_dot() - Graphviz DOT format
- to_ascii() - ASCII tree format
"""

import pytest

from cortical.reasoning.graph_of_thought import EdgeType, NodeType
from cortical.reasoning.thought_graph import ThoughtGraph


class TestThoughtGraphVisualization:
    """Test visualization methods for ThoughtGraph."""

    def setup_method(self):
        """Set up a basic graph for testing."""
        self.graph = ThoughtGraph()

    def test_to_mermaid_empty_graph(self):
        """Test Mermaid output for empty graph."""
        result = self.graph.to_mermaid()
        assert "graph TD" in result
        assert "empty[Empty Graph]" in result

    def test_to_mermaid_single_node(self):
        """Test Mermaid output for single node."""
        self.graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        result = self.graph.to_mermaid()

        assert "graph TD" in result
        assert "Q1[What auth method?]" in result

    def test_to_mermaid_node_shapes(self):
        """Test that different node types have different shapes."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_node("D1", NodeType.DECISION, "Decision")
        self.graph.add_node("E1", NodeType.EVIDENCE, "Evidence")
        self.graph.add_node("C1", NodeType.CONCEPT, "Concept")
        self.graph.add_node("F1", NodeType.FACT, "Fact")
        self.graph.add_node("T1", NodeType.TASK, "Task")
        self.graph.add_node("A1", NodeType.ARTIFACT, "Artifact")
        self.graph.add_node("I1", NodeType.INSIGHT, "Insight")

        result = self.graph.to_mermaid()

        # Check different shapes
        assert "Q1[Question]" in result  # Box
        assert "H1((Hypothesis))" in result  # Double circle
        assert "D1{Decision}" in result  # Diamond
        assert "E1>Evidence]" in result  # Asymmetric shape
        assert "C1([Concept])" in result  # Stadium
        assert "F1[[Fact]]" in result  # Subroutine
        assert "T1[/Task/]" in result  # Parallelogram
        assert "A1[(Artifact)]" in result  # Cylindrical
        assert "I1[\\Insight\\]" in result  # Trapezoid

    def test_to_mermaid_with_edges(self):
        """Test Mermaid output with edges."""
        self.graph.add_node("Q1", NodeType.QUESTION, "What auth?")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT")
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES)

        result = self.graph.to_mermaid()

        assert "Q1 -->|explores| H1" in result

    def test_to_mermaid_edge_weights(self):
        """Test that edge weights are shown when not 1.0."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.75)

        result = self.graph.to_mermaid()

        assert "explores (0.75)" in result

    def test_to_mermaid_bidirectional_edges(self):
        """Test bidirectional edges in Mermaid."""
        self.graph.add_node("C1", NodeType.CONCEPT, "Concept A")
        self.graph.add_node("C2", NodeType.CONCEPT, "Concept B")
        self.graph.add_edge("C1", "C2", EdgeType.SIMILAR, bidirectional=True)

        result = self.graph.to_mermaid()

        assert "C1 <-->|similar| C2" in result
        # Should not show reverse edge
        assert result.count("C1") == 2  # Once in node def, once in edge

    def test_to_mermaid_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "This is a very long content that should be truncated to prevent the diagram from being too wide"
        self.graph.add_node("Q1", NodeType.QUESTION, long_content)

        result = self.graph.to_mermaid()

        assert "..." in result
        assert len(long_content) > 50  # Original is long
        # Check that the content in mermaid is shorter
        lines = result.split("\n")
        node_line = [l for l in lines if "Q1[" in l][0]
        assert len(node_line) < len(long_content) + 20  # Should be truncated

    def test_to_mermaid_escapes_special_chars(self):
        """Test that special characters are escaped."""
        self.graph.add_node("Q1", NodeType.QUESTION, 'Question with "quotes"')

        result = self.graph.to_mermaid()

        # Quotes should be replaced with single quotes
        assert "Question with 'quotes'" in result

    def test_to_dot_empty_graph(self):
        """Test DOT output for empty graph."""
        result = self.graph.to_dot()

        assert "digraph ThoughtGraph" in result
        assert "empty [label=\"Empty Graph\" shape=plaintext]" in result

    def test_to_dot_single_node(self):
        """Test DOT output for single node."""
        self.graph.add_node("Q1", NodeType.QUESTION, "What auth?")

        result = self.graph.to_dot()

        assert "digraph ThoughtGraph" in result
        assert 'Q1 [label="Question: What auth?" shape=box fillcolor=lightblue style=filled]' in result

    def test_to_dot_node_shapes_and_colors(self):
        """Test that different node types have different shapes and colors."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_node("D1", NodeType.DECISION, "Decision")
        self.graph.add_node("E1", NodeType.EVIDENCE, "Evidence")
        self.graph.add_node("C1", NodeType.CONCEPT, "Concept")
        self.graph.add_node("F1", NodeType.FACT, "Fact")

        result = self.graph.to_dot()

        # Check shapes and colors
        assert "shape=box fillcolor=lightblue" in result  # Question
        assert "shape=ellipse fillcolor=lightgreen" in result  # Hypothesis
        assert "shape=diamond fillcolor=lightyellow" in result  # Decision
        assert "shape=note fillcolor=lightcyan" in result  # Evidence
        assert "shape=hexagon fillcolor=lightpink" in result  # Concept
        assert "shape=rectangle fillcolor=lightgray" in result  # Fact

    def test_to_dot_with_edges(self):
        """Test DOT output with edges."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)

        result = self.graph.to_dot()

        assert 'Q1 -> H1 [label="explores\\n(0.80)"]' in result

    def test_to_dot_bidirectional_edges(self):
        """Test bidirectional edges in DOT."""
        self.graph.add_node("C1", NodeType.CONCEPT, "Concept A")
        self.graph.add_node("C2", NodeType.CONCEPT, "Concept B")
        self.graph.add_edge("C1", "C2", EdgeType.SIMILAR, bidirectional=True)

        result = self.graph.to_dot()

        assert "dir=both" in result

    def test_to_dot_low_confidence_edges(self):
        """Test that low confidence edges are shown as dashed."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_edge("Q1", "H1", EdgeType.SUGGESTS, confidence=0.5)

        result = self.graph.to_dot()

        assert "style=dashed" in result

    def test_to_dot_escapes_special_chars(self):
        """Test that special characters are escaped in DOT."""
        self.graph.add_node("Q1", NodeType.QUESTION, 'Question with "quotes"')

        result = self.graph.to_dot()

        assert '\\"' in result  # Escaped quotes

    def test_to_ascii_empty_graph(self):
        """Test ASCII output for empty graph."""
        result = self.graph.to_ascii()

        assert result == "[Empty Graph]"

    def test_to_ascii_single_node(self):
        """Test ASCII output for single node."""
        self.graph.add_node("Q1", NodeType.QUESTION, "What auth?")

        result = self.graph.to_ascii()

        assert "[QUESTION] What auth?" in result

    def test_to_ascii_tree_structure(self):
        """Test ASCII tree structure with parent-child relationships."""
        self.graph.add_node("Q1", NodeType.QUESTION, "What auth?")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT")
        self.graph.add_node("E1", NodeType.EVIDENCE, "Team has experience")
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES)
        self.graph.add_edge("H1", "E1", EdgeType.SUPPORTS)

        result = self.graph.to_ascii()

        # Check tree structure
        assert "[QUESTION] What auth?" in result
        assert "[HYPOTHESIS] Use JWT" in result
        assert "[EVIDENCE] Team has experience" in result
        # Check tree characters
        assert "└──" in result or "├──" in result

    def test_to_ascii_with_explicit_root(self):
        """Test ASCII with explicit root node."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question 1")
        self.graph.add_node("Q2", NodeType.QUESTION, "Question 2")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_edge("Q2", "H1", EdgeType.EXPLORES)

        # Use Q2 as root explicitly
        result = self.graph.to_ascii(root_id="Q2")

        assert result.startswith("[QUESTION] Question 2")

    def test_to_ascii_invalid_root(self):
        """Test that invalid root raises ValueError."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")

        with pytest.raises(ValueError, match="Root node.*not found"):
            self.graph.to_ascii(root_id="INVALID")

    def test_to_ascii_max_depth(self):
        """Test that max_depth parameter limits traversal."""
        # Create a deep chain
        self.graph.add_node("N1", NodeType.CONCEPT, "Node 1")
        self.graph.add_node("N2", NodeType.CONCEPT, "Node 2")
        self.graph.add_node("N3", NodeType.CONCEPT, "Node 3")
        self.graph.add_node("N4", NodeType.CONCEPT, "Node 4")
        self.graph.add_edge("N1", "N2", EdgeType.REQUIRES)
        self.graph.add_edge("N2", "N3", EdgeType.REQUIRES)
        self.graph.add_edge("N3", "N4", EdgeType.REQUIRES)

        # Limit depth to 2
        result = self.graph.to_ascii(root_id="N1", max_depth=2)

        # N4 should not appear (depth 3)
        lines = result.split("\n")
        concept_lines = [l for l in lines if "[CONCEPT]" in l]
        # Should have N1, N2, N3 but not N4 due to depth limit
        assert len(concept_lines) <= 3

    def test_to_ascii_shows_edge_types(self):
        """Test that edge types are shown in ASCII output."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES)

        result = self.graph.to_ascii()

        assert "(explores)" in result

    def test_to_ascii_shows_edge_weights(self):
        """Test that edge weights are shown when not 1.0."""
        self.graph.add_node("Q1", NodeType.QUESTION, "Question")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.75)

        result = self.graph.to_ascii()

        assert "[0.75]" in result

    def test_to_ascii_truncates_long_content(self):
        """Test that long content is truncated in ASCII."""
        long_content = "This is a very long content that should be truncated to prevent lines from being too wide in the terminal"
        self.graph.add_node("Q1", NodeType.QUESTION, long_content)

        result = self.graph.to_ascii()

        assert "..." in result

    def test_to_ascii_handles_cycles(self):
        """Test that cycles don't cause infinite loops."""
        self.graph.add_node("A", NodeType.CONCEPT, "Node A")
        self.graph.add_node("B", NodeType.CONCEPT, "Node B")
        self.graph.add_node("C", NodeType.CONCEPT, "Node C")
        self.graph.add_edge("A", "B", EdgeType.REQUIRES)
        self.graph.add_edge("B", "C", EdgeType.REQUIRES)
        self.graph.add_edge("C", "A", EdgeType.REQUIRES)  # Creates cycle

        # Should not hang
        result = self.graph.to_ascii(root_id="A")

        assert "[CONCEPT] Node A" in result

    def test_to_ascii_shows_unvisited_nodes(self):
        """Test that unvisited nodes are noted."""
        # Create two disconnected components
        self.graph.add_node("A", NodeType.CONCEPT, "Node A")
        self.graph.add_node("B", NodeType.CONCEPT, "Node B")
        self.graph.add_node("C", NodeType.CONCEPT, "Node C")
        self.graph.add_edge("A", "B", EdgeType.REQUIRES)
        # C is disconnected

        result = self.graph.to_ascii(root_id="A")

        # Should mention unvisited nodes
        assert "node(s) not shown" in result.lower() or "C" not in result

    def test_complex_graph_all_formats(self):
        """Test a complex graph with all three visualization formats."""
        # Build a realistic reasoning graph
        self.graph.add_node("Q1", NodeType.QUESTION, "How to handle auth?")
        self.graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        self.graph.add_node("H2", NodeType.HYPOTHESIS, "Use session cookies")
        self.graph.add_node("E1", NodeType.EVIDENCE, "Team has JWT experience")
        self.graph.add_node("E2", NodeType.EVIDENCE, "Stateless is preferred")
        self.graph.add_node("D1", NodeType.DECISION, "Choose JWT")
        self.graph.add_node("T1", NodeType.TASK, "Implement JWT auth")

        # Connect the graph
        self.graph.add_edge("Q1", "H1", EdgeType.EXPLORES)
        self.graph.add_edge("Q1", "H2", EdgeType.EXPLORES)
        self.graph.add_edge("E1", "H1", EdgeType.SUPPORTS, weight=0.9)
        self.graph.add_edge("E2", "H1", EdgeType.SUPPORTS, weight=0.8)
        self.graph.add_edge("H1", "D1", EdgeType.MOTIVATES)
        self.graph.add_edge("D1", "T1", EdgeType.IMPLEMENTS)

        # Test all three formats produce valid output
        mermaid = self.graph.to_mermaid()
        dot = self.graph.to_dot()
        ascii_tree = self.graph.to_ascii()

        # Mermaid checks
        assert "graph TD" in mermaid
        assert "Q1[How to handle auth?]" in mermaid
        assert "H1((Use JWT tokens))" in mermaid
        assert "D1{Choose JWT}" in mermaid

        # DOT checks
        assert "digraph ThoughtGraph" in dot
        assert "shape=box" in dot  # Question
        assert "shape=ellipse" in dot  # Hypothesis
        assert "shape=diamond" in dot  # Decision

        # ASCII checks
        assert "[QUESTION] How to handle auth?" in ascii_tree
        assert "[HYPOTHESIS]" in ascii_tree
        assert "[DECISION]" in ascii_tree

    def test_visualization_consistency(self):
        """Test that all formats represent the same graph structure."""
        self.graph.add_node("A", NodeType.CONCEPT, "Node A")
        self.graph.add_node("B", NodeType.CONCEPT, "Node B")
        self.graph.add_edge("A", "B", EdgeType.REQUIRES)

        mermaid = self.graph.to_mermaid()
        dot = self.graph.to_dot()
        ascii_tree = self.graph.to_ascii()

        # All should mention both nodes
        for output in [mermaid, dot, ascii_tree]:
            assert "Node A" in output
            assert "Node B" in output

        # Mermaid and DOT should show the edge
        for output in [mermaid, dot]:
            assert "requires" in output.lower()


class TestThoughtGraphVisualizationEdgeCases:
    """Test edge cases for visualization methods."""

    def test_single_node_all_formats(self):
        """Test that a single node renders in all formats."""
        graph = ThoughtGraph()
        graph.add_node("ONLY", NodeType.FACT, "Single fact")

        assert "Single fact" in graph.to_mermaid()
        assert "Single fact" in graph.to_dot()
        assert "Single fact" in graph.to_ascii()

    def test_multiple_disconnected_components(self):
        """Test graph with multiple disconnected components."""
        graph = ThoughtGraph()
        graph.add_node("A1", NodeType.CONCEPT, "Component A")
        graph.add_node("A2", NodeType.CONCEPT, "Connected to A")
        graph.add_node("B1", NodeType.CONCEPT, "Component B")
        graph.add_node("B2", NodeType.CONCEPT, "Connected to B")

        graph.add_edge("A1", "A2", EdgeType.REQUIRES)
        graph.add_edge("B1", "B2", EdgeType.REQUIRES)

        # All formats should work
        mermaid = graph.to_mermaid()
        dot = graph.to_dot()
        ascii_tree = graph.to_ascii()

        # All nodes should be present in Mermaid and DOT
        for output in [mermaid, dot]:
            assert "Component A" in output
            assert "Component B" in output

    def test_node_with_special_markdown_chars(self):
        """Test nodes with markdown special characters."""
        graph = ThoughtGraph()
        graph.add_node("N1", NodeType.CONCEPT, "Code: `foo()` and **bold**")

        # Should not break the output
        mermaid = graph.to_mermaid()
        dot = graph.to_dot()
        ascii_tree = graph.to_ascii()

        assert mermaid  # Just check it doesn't crash
        assert dot
        assert ascii_tree

    def test_all_node_types_covered(self):
        """Test that all NodeType enum values are handled."""
        graph = ThoughtGraph()

        # Add one node of each type
        node_types = [
            NodeType.CONCEPT,
            NodeType.QUESTION,
            NodeType.DECISION,
            NodeType.FACT,
            NodeType.TASK,
            NodeType.ARTIFACT,
            NodeType.INSIGHT,
            NodeType.HYPOTHESIS,
            NodeType.OPTION,
            NodeType.EVIDENCE,
            NodeType.OBSERVATION,
            NodeType.GOAL,
            NodeType.CONTEXT,
            NodeType.CONSTRAINT,
            NodeType.ACTION,
        ]

        for i, node_type in enumerate(node_types):
            graph.add_node(f"N{i}", node_type, f"Node {i}")

        # All formats should work
        mermaid = graph.to_mermaid()
        dot = graph.to_dot()
        ascii_tree = graph.to_ascii()

        # Check outputs are non-empty and contain some nodes
        assert len(mermaid) > 100
        assert len(dot) > 100
        assert len(ascii_tree) > 50
