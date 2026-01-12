"""
Behavioral tests for Natural Language Query Interface.

These tests define the expected behavior of the `ask` command,
which takes natural language questions and generates complete answers.

TDD: Write these tests FIRST, then implement to make them pass.
"""

import pytest
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tool_registry():
    """Create a fresh tool registry."""
    from cortical.cognitive.tool_registry import ToolRegistry
    return ToolRegistry()


@pytest.fixture
def trained_agent(tmp_path):
    """Create a CognitiveAgent with some trained knowledge."""
    from cortical.cognitive.graph import CognitiveAgent, CognitiveGraph, AtomType, TruthValue

    graph = CognitiveGraph()
    agent = CognitiveAgent(graph=graph)

    # Add some vocabulary with associations (name first, then type)
    word_code = graph.node("codebridge", AtomType.WORD)
    word_index = graph.node("indexing", AtomType.WORD)
    word_ast = graph.node("ast", AtomType.WORD)
    word_file = graph.node("file", AtomType.WORD)

    # Create similarity links
    graph.link(AtomType.SIMILARITY, [word_code, word_index], TruthValue(0.8, 0.9))
    graph.link(AtomType.SIMILARITY, [word_code, word_ast], TruthValue(0.7, 0.8))
    graph.link(AtomType.SIMILARITY, [word_index, word_file], TruthValue(0.6, 0.7))

    # Add code entities (name first, then type)
    file_atom = graph.node("cortical/cognitive/code_bridge.py", AtomType.FILE)
    file_atom.metadata["file_path"] = "cortical/cognitive/code_bridge.py"

    class_atom = graph.node("CodeBridge", AtomType.CLASS)
    class_atom.metadata["file_path"] = "cortical/cognitive/code_bridge.py"
    class_atom.metadata["lineno"] = 45

    func_atom = graph.node("index_file", AtomType.FUNCTION)
    func_atom.metadata["file_path"] = "cortical/cognitive/code_bridge.py"
    func_atom.metadata["lineno"] = 89

    # Create REFERS_TO links (word -> code)
    graph.link(AtomType.REFERS_TO, [word_code, class_atom], TruthValue(0.9, 0.9))
    graph.link(AtomType.REFERS_TO, [word_index, func_atom], TruthValue(0.8, 0.8))

    # Create DEFINES link (file -> class)
    graph.link(AtomType.DEFINES, [file_atom, class_atom], TruthValue(1.0, 1.0))
    graph.link(AtomType.DEFINES, [file_atom, func_atom], TruthValue(1.0, 1.0))

    return agent


@pytest.fixture
def nl_query(trained_agent):
    """Create NLQuery instance with trained agent."""
    from cortical.cognitive.nl_query import NLQuery
    return NLQuery(trained_agent)


# =============================================================================
# Tool Registry Tests
# =============================================================================

class TestToolRegistry:
    """Tool registry allows extensible capabilities."""

    def test_register_tool(self, tool_registry):
        """Can register a tool with name, handler, description."""
        def my_handler(target: str):
            return [f"result for {target}"]

        tool_registry.register("my_tool", my_handler, "Does something useful")

        assert tool_registry.has("my_tool")

    def test_get_registered_tool(self, tool_registry):
        """Can retrieve and call a registered tool."""
        def my_handler(target: str):
            return [f"found: {target}"]

        tool_registry.register("finder", my_handler, "Finds things")
        handler = tool_registry.get("finder")

        result = handler("test")
        assert result == ["found: test"]

    def test_get_unknown_tool_returns_none(self, tool_registry):
        """Getting unknown tool returns None, not exception."""
        assert tool_registry.get("nonexistent") is None

    def test_register_with_category(self, tool_registry):
        """Tools can be registered with categories."""
        tool_registry.register("tool1", lambda x: x, "desc", category="cognitive")
        tool_registry.register("tool2", lambda x: x, "desc", category="got")

        cognitive_tools = tool_registry.find_by_category("cognitive")
        assert len(cognitive_tools) == 1
        assert cognitive_tools[0].name == "tool1"

    def test_list_all_tools(self, tool_registry):
        """Can list all registered tools."""
        tool_registry.register("a", lambda x: x, "tool a")
        tool_registry.register("b", lambda x: x, "tool b")

        all_tools = tool_registry.list_all()
        names = [t.name for t in all_tools]

        assert "a" in names
        assert "b" in names


# =============================================================================
# Intent Parser Tests
# =============================================================================

class TestIntentParser:
    """Intent parser extracts meaning from questions."""

    def test_parse_how_question(self, nl_query):
        """'How does X work?' extracts concept and question type."""
        intent = nl_query.parse_intent("How does code indexing work?")

        assert intent.question_type == "how"
        assert "indexing" in intent.concepts or "code" in intent.concepts

    def test_parse_where_question(self, nl_query):
        """'Where is X?' identifies location query."""
        intent = nl_query.parse_intent("Where is the CodeBridge class?")

        assert intent.question_type == "where"
        assert "codebridge" in [c.lower() for c in intent.concepts]

    def test_parse_what_calls_question(self, nl_query):
        """'What calls X?' identifies caller query."""
        intent = nl_query.parse_intent("What calls index_file?")

        assert intent.question_type == "what"
        assert "calls" in intent.query_strategy or "callers_of" in intent.query_strategy

    def test_parse_extracts_multiple_concepts(self, nl_query):
        """Parser extracts multiple relevant concepts."""
        intent = nl_query.parse_intent("How does CodeBridge index Python files?")

        # Should extract meaningful concepts, not stop words
        assert len(intent.concepts) >= 2
        assert "how" not in intent.concepts
        assert "does" not in intent.concepts


# =============================================================================
# Knowledge Gatherer Tests
# =============================================================================

class TestKnowledgeGatherer:
    """Knowledge gatherer collects information from multiple sources."""

    def test_gather_finds_associations(self, nl_query, trained_agent):
        """Gatherer finds word associations."""
        intent = nl_query.parse_intent("How does codebridge work?")
        knowledge = nl_query.gather_knowledge(intent)

        # Should find associations for "codebridge"
        assert len(knowledge.associations) > 0

    def test_gather_finds_code_entities(self, nl_query):
        """Gatherer finds code entities via REFERS_TO."""
        intent = nl_query.parse_intent("Where is codebridge?")
        knowledge = nl_query.gather_knowledge(intent)

        # Should find CodeBridge class
        assert len(knowledge.code_entities) > 0
        entity_names = [e.name for e in knowledge.code_entities]
        assert "CodeBridge" in entity_names

    def test_gather_includes_file_paths(self, nl_query):
        """Gatherer includes file paths from code entities."""
        intent = nl_query.parse_intent("Where is codebridge?")
        knowledge = nl_query.gather_knowledge(intent)

        assert len(knowledge.related_files) > 0
        assert any("code_bridge.py" in f for f in knowledge.related_files)

    def test_gather_empty_for_unknown_concept(self, nl_query):
        """Gatherer returns empty knowledge for unknown concepts."""
        intent = nl_query.parse_intent("How does xyzfoobar work?")
        knowledge = nl_query.gather_knowledge(intent)

        assert len(knowledge.associations) == 0
        assert len(knowledge.code_entities) == 0


# =============================================================================
# Response Generator Tests
# =============================================================================

class TestResponseGenerator:
    """Response generator formats knowledge into natural language."""

    def test_generate_includes_summary(self, nl_query):
        """Response includes a summary sentence."""
        response = nl_query.ask("How does codebridge work?")

        # Should have some content
        assert len(response) > 50
        assert "codebridge" in response.lower() or "CodeBridge" in response

    def test_generate_includes_file_location(self, nl_query):
        """Response includes file:line references."""
        response = nl_query.ask("Where is CodeBridge?")

        assert "code_bridge.py" in response
        # Should include line number
        assert "line" in response.lower() or ":" in response

    def test_generate_unknown_says_so(self, nl_query):
        """Response honestly says 'I don't know' for unknown concepts."""
        response = nl_query.ask("How does xyzfoobar123 work?")

        assert "don't" in response.lower() or "unknown" in response.lower() or "no information" in response.lower()

    def test_generate_includes_related_concepts(self, nl_query):
        """Response includes related concepts."""
        response = nl_query.ask("How does codebridge work?")

        # Should mention related concepts from associations
        # (indexing, ast are associated with codebridge in fixture)
        has_related = "indexing" in response.lower() or "ast" in response.lower()
        assert has_related or "related" in response.lower()


# =============================================================================
# End-to-End Ask Command Tests
# =============================================================================

class TestAskCommand:
    """End-to-end tests for the ask command."""

    def test_how_question_returns_mechanism(self, nl_query):
        """'How does X work?' returns explanation with code refs."""
        response = nl_query.ask("How does code indexing work?")

        # Should mention the relevant class/function
        assert "CodeBridge" in response or "index" in response.lower()
        # Should include file reference
        assert ".py" in response

    def test_where_question_returns_location(self, nl_query):
        """'Where is X?' returns file:line location."""
        response = nl_query.ask("Where is CodeBridge?")

        assert "code_bridge.py" in response
        assert "45" in response or "line" in response.lower()

    def test_response_is_actionable(self, nl_query):
        """Response includes actionable information."""
        response = nl_query.ask("How does codebridge work?")

        # Should include file path user can navigate to
        assert "/" in response or ".py" in response

    def test_ask_returns_string(self, nl_query):
        """ask() always returns a string, never raises."""
        response = nl_query.ask("Some random question?")

        assert isinstance(response, str)
        assert len(response) > 0


# =============================================================================
# Tool Integration Tests
# =============================================================================

class TestToolIntegration:
    """Tests for tool registry integration with NLQuery."""

    def test_nl_query_uses_registered_tools(self, trained_agent):
        """NLQuery uses tools from registry."""
        from cortical.cognitive.nl_query import NLQuery
        from cortical.cognitive.tool_registry import ToolRegistry

        registry = ToolRegistry()

        # Register a custom tool
        custom_results = []
        def custom_tool(target):
            custom_results.append(target)
            return [f"custom result for {target}"]

        registry.register("custom_finder", custom_tool, "Custom finder", category="custom")

        nl = NLQuery(trained_agent, registry=registry)
        # The tool should be available
        assert nl.registry.has("custom_finder")

    def test_default_cognitive_tools_registered(self, trained_agent):
        """Default cognitive tools are auto-registered."""
        from cortical.cognitive.nl_query import NLQuery

        nl = NLQuery(trained_agent)

        # Should have default cognitive tools
        assert nl.registry.has("similar_to")
        assert nl.registry.has("code_for_word")
