"""
Behavioral Tests for Unified Query Interface.

Specification:
    CognitiveAgent.query() provides a single entry point for all query types:
    - Text queries (similar_to, predict_next)
    - Code queries (callers_of, subclasses_of, methods_of)
    - Bridge queries (code_for_word, words_for_code)
"""

import pytest
import tempfile
from pathlib import Path

from cortical.cognitive.graph import CognitiveGraph, CognitiveAgent, AtomType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent():
    """Fresh CognitiveAgent for each test."""
    return CognitiveAgent()


@pytest.fixture
def agent_with_text(agent):
    """Agent with text vocabulary (WORD atoms with SIMILARITY links)."""
    graph = agent.graph

    # Create WORD atoms
    neural = graph.node("neural", atom_type=AtomType.WORD)
    network = graph.node("network", atom_type=AtomType.WORD)
    learning = graph.node("learning", atom_type=AtomType.WORD)
    deep = graph.node("deep", atom_type=AtomType.WORD)

    # Create SIMILARITY links
    graph.link(AtomType.SIMILARITY, [neural, network])
    graph.link(AtomType.SIMILARITY, [neural, learning])
    graph.link(AtomType.SIMILARITY, [deep, learning])

    return agent


@pytest.fixture
def agent_with_code(agent):
    """Agent with code structure (CLASS, FUNCTION atoms with links)."""
    graph = agent.graph

    # Create FILE atom
    file_atom = graph.node("test.py", atom_type=AtomType.FILE)

    # Create CLASS atoms
    base_class = graph.node("BaseClass", atom_type=AtomType.CLASS)
    child_class = graph.node("ChildClass", atom_type=AtomType.CLASS)

    # Create FUNCTION atoms
    helper = graph.node("helper", atom_type=AtomType.FUNCTION)
    caller = graph.node("caller", atom_type=AtomType.FUNCTION)
    method = graph.node("BaseClass.process", atom_type=AtomType.FUNCTION)

    # Create links
    graph.link(AtomType.DEFINES, [file_atom, base_class])
    graph.link(AtomType.DEFINES, [file_atom, helper])
    graph.link(AtomType.INHERITANCE, [child_class, base_class])
    graph.link(AtomType.CONTAINS, [base_class, method])
    graph.link(AtomType.CALLS, [caller, helper])

    return agent


@pytest.fixture
def agent_with_bridge(agent_with_text, agent_with_code):
    """Agent with both text and code, plus REFERS_TO links."""
    # This fixture combines text and code, then adds bridge links
    # For simplicity, we'll create a fresh agent with everything
    agent = CognitiveAgent()
    graph = agent.graph

    # Text vocabulary
    neural = graph.node("neural", atom_type=AtomType.WORD)
    process = graph.node("process", atom_type=AtomType.WORD)
    base = graph.node("base", atom_type=AtomType.WORD)

    # Code structure
    base_class = graph.node("BaseClass", atom_type=AtomType.CLASS)
    process_func = graph.node("process_data", atom_type=AtomType.FUNCTION)

    # REFERS_TO links (bridge)
    graph.link(AtomType.REFERS_TO, [process, process_func])
    graph.link(AtomType.REFERS_TO, [base, base_class])

    return agent


# =============================================================================
# Test: Query Method Exists
# =============================================================================


class TestQueryMethodExists:
    """Verify the query method exists and has correct signature."""

    def test_agent_has_query_method(self, agent):
        """CognitiveAgent should have a query() method."""
        assert hasattr(agent, 'query')
        assert callable(agent.query)

    def test_query_returns_list(self, agent):
        """query() should return a list (possibly empty)."""
        result = agent.query("similar_to", "nonexistent")
        assert isinstance(result, list)

    def test_unknown_query_type_raises(self, agent):
        """Unknown query type should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            agent.query("unknown_type", "target")
        assert "unknown_type" in str(exc_info.value).lower()


# =============================================================================
# Test: Text Queries
# =============================================================================


class TestTextQueries:
    """Verify text-related queries work through unified interface."""

    def test_similar_to_returns_associated_words(self, agent_with_text):
        """query('similar_to', word) should return associated words."""
        results = agent_with_text.query("similar_to", "neural")

        result_names = [r.name for r in results]
        # neural has SIMILARITY links to network and learning
        assert "network" in result_names or "learning" in result_names

    def test_similar_to_nonexistent_returns_empty(self, agent_with_text):
        """query('similar_to', nonexistent) should return empty list."""
        results = agent_with_text.query("similar_to", "nonexistent_word")
        assert results == []


# =============================================================================
# Test: Code Queries
# =============================================================================


class TestCodeQueries:
    """Verify code-related queries work through unified interface."""

    def test_callers_of_returns_calling_functions(self, agent_with_code):
        """query('callers_of', func) should return functions that call it."""
        results = agent_with_code.query("callers_of", "helper")

        result_names = [r.name for r in results]
        assert "caller" in result_names

    def test_subclasses_of_returns_child_classes(self, agent_with_code):
        """query('subclasses_of', class) should return inheriting classes."""
        results = agent_with_code.query("subclasses_of", "BaseClass")

        result_names = [r.name for r in results]
        assert "ChildClass" in result_names

    def test_methods_of_returns_class_methods(self, agent_with_code):
        """query('methods_of', class) should return methods."""
        results = agent_with_code.query("methods_of", "BaseClass")

        result_names = [r.name for r in results]
        assert any("process" in name for name in result_names)

    def test_defined_in_returns_file_contents(self, agent_with_code):
        """query('defined_in', file) should return defined entities."""
        results = agent_with_code.query("defined_in", "test.py")

        result_names = [r.name for r in results]
        assert "BaseClass" in result_names or "helper" in result_names


# =============================================================================
# Test: Bridge Queries
# =============================================================================


class TestBridgeQueries:
    """Verify bridge queries (text <-> code) work through unified interface."""

    def test_code_for_word_returns_code_entities(self, agent_with_bridge):
        """query('code_for_word', word) should return code entities."""
        results = agent_with_bridge.query("code_for_word", "process")

        result_names = [r.name for r in results]
        assert any("process" in name.lower() for name in result_names)

    def test_words_for_code_returns_vocabulary(self, agent_with_bridge):
        """query('words_for_code', code) should return related words."""
        results = agent_with_bridge.query("words_for_code", "BaseClass")

        result_names = [r.name for r in results]
        assert "base" in result_names


# =============================================================================
# Test: Query Options
# =============================================================================


class TestQueryOptions:
    """Verify query options (top_k, etc.) work."""

    def test_top_k_limits_results(self, agent_with_text):
        """top_k option should limit number of results."""
        # Add more similarity links
        graph = agent_with_text.graph
        neural = graph.get_node("neural")
        for i in range(10):
            word = graph.node(f"word{i}", atom_type=AtomType.WORD)
            graph.link(AtomType.SIMILARITY, [neural, word])

        results = agent_with_text.query("similar_to", "neural", top_k=3)
        assert len(results) <= 3
