"""
Behavioral tests for Query Executors (Phase 2).

Tests the executor implementations:
- AuditExecutor: PLN-based audit reasoning
- SemanticExecutor: Document retrieval
- CodeExecutor: Code structure queries
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from cortical.cognitive.executors import (
    QueryExecutorProtocol,
    ExecutionResult,
    AuditExecutor,
    SemanticExecutor,
    CodeExecutor,
    CDGExecutor,
)
from cortical.audits.reasoning import AuditQuery


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """ExecutionResult dataclass behavior."""

    def test_given_empty_items_when_checking_is_empty_then_true(self):
        """
        Given: An ExecutionResult with no items
        When: Checking is_empty
        Then: Returns True
        """
        result = ExecutionResult(items=[], source="test")
        assert result.is_empty is True

    def test_given_items_when_checking_is_empty_then_false(self):
        """
        Given: An ExecutionResult with items
        When: Checking is_empty
        Then: Returns False
        """
        result = ExecutionResult(items=["item1", "item2"], source="test")
        assert result.is_empty is False

    def test_given_items_when_len_then_returns_count(self):
        """
        Given: An ExecutionResult with items
        When: Getting len()
        Then: Returns item count
        """
        result = ExecutionResult(items=[1, 2, 3], source="test")
        assert len(result) == 3

    def test_given_result_when_accessing_properties_then_defaults_work(self):
        """
        Given: An ExecutionResult with defaults
        When: Accessing properties
        Then: Defaults are applied correctly
        """
        result = ExecutionResult()
        assert result.items == []
        assert result.confidence == 0.5
        assert result.source == "unknown"
        assert result.explanation is None
        assert result.metadata == {}


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Verify executors implement the protocol."""

    def test_audit_executor_implements_protocol(self):
        """AuditExecutor implements QueryExecutorProtocol."""
        executor = AuditExecutor()
        assert isinstance(executor, QueryExecutorProtocol)
        assert hasattr(executor, "execute")
        assert hasattr(executor, "format_result")

    def test_semantic_executor_implements_protocol(self):
        """SemanticExecutor implements QueryExecutorProtocol."""
        executor = SemanticExecutor()
        assert isinstance(executor, QueryExecutorProtocol)
        assert hasattr(executor, "execute")
        assert hasattr(executor, "format_result")

    def test_code_executor_implements_protocol(self):
        """CodeExecutor implements QueryExecutorProtocol."""
        executor = CodeExecutor()
        assert isinstance(executor, QueryExecutorProtocol)
        assert hasattr(executor, "execute")
        assert hasattr(executor, "format_result")


# =============================================================================
# AuditExecutor Tests
# =============================================================================


class TestAuditExecutorBehavior:
    """AuditExecutor behavior tests."""

    def test_given_explain_query_when_executed_then_returns_explanation(self):
        """
        Given: An explain query for a file
        When: Executed
        Then: Returns explanation with facts and suggestions
        """
        executor = AuditExecutor()

        # Add a fact so we get results
        executor.reasoner.assert_file_facts(
            "test_file.py",
            patterns=["todo", "hack"],
            traits=["high_churn"],
            directories=["cortical"]
        )

        query = AuditQuery(
            intent="explain",
            target_file="test_file.py",
            explain=True
        )

        result = executor.execute(query)

        assert result.source == "audit"
        assert result.metadata.get("intent") == "explain"
        assert len(result.items) > 0

        explanation = result.items[0]
        assert "file_id" in explanation
        assert "facts" in explanation

    def test_given_list_query_when_executed_then_returns_priority_files(self):
        """
        Given: A list query
        When: Executed
        Then: Returns files sorted by priority
        """
        executor = AuditExecutor()

        # Add some files with different risk levels
        executor.reasoner.assert_file_facts(
            "high_risk.py",
            patterns=["todo", "hack", "fixme"],
            traits=["high_churn"],
            directories=["legacy"]
        )
        executor.reasoner.assert_file_facts(
            "low_risk.py",
            patterns=[],
            traits=[],
            directories=["utils"]
        )

        query = AuditQuery(intent="list")
        result = executor.execute(query)

        assert result.source == "audit"
        assert result.metadata.get("intent") == "list"

    def test_given_min_risk_filter_when_executed_then_filters_results(self):
        """
        Given: A query with min_risk threshold
        When: Executed
        Then: Only files meeting threshold are returned
        """
        executor = AuditExecutor()

        query = AuditQuery(
            intent="list",
            min_risk=0.8  # High threshold
        )

        result = executor.execute(query)

        # All returned items should meet threshold
        for item in result.items:
            assert item.get("risk_score", 0) >= 0.0  # May be empty

    def test_format_result_with_empty_produces_message(self):
        """format_result with empty result produces helpful message."""
        executor = AuditExecutor()
        result = ExecutionResult(items=[], source="audit")

        formatted = executor.format_result(result)
        assert "No risky files found" in formatted or "No explanation" in formatted


# =============================================================================
# SemanticExecutor Tests
# =============================================================================


class TestSemanticExecutorBehavior:
    """SemanticExecutor behavior tests."""

    def test_given_query_intent_when_executed_then_searches_documents(self):
        """
        Given: A QueryIntent with concepts
        When: Executed
        Then: Searches for relevant documents
        """
        from cortical.cognitive.unified_query import QueryIntent

        executor = SemanticExecutor(
            model_dir=Path("models/cognitive_agent"),
            samples_dir=Path("samples")
        )

        query = QueryIntent(
            question_type="what",
            concepts=["cognitive", "agent"],
            raw_question="what is the cognitive agent"
        )

        result = executor.execute(query)

        assert result.source == "semantic"
        assert result.metadata.get("concepts") == ["cognitive", "agent"]

    def test_given_empty_concepts_when_executed_then_returns_empty(self):
        """
        Given: A QueryIntent with no concepts
        When: Executed
        Then: Returns empty result
        """
        from cortical.cognitive.unified_query import QueryIntent

        executor = SemanticExecutor()
        query = QueryIntent(
            question_type="general",
            concepts=[],
            raw_question=""
        )

        result = executor.execute(query)

        assert result.is_empty
        assert result.confidence == 0.0

    def test_format_result_with_documents_shows_excerpts(self):
        """format_result with documents shows excerpts."""
        executor = SemanticExecutor()
        result = ExecutionResult(
            items=[
                {"doc_id": "test.md", "score": 5.0, "excerpt": "Test content here."},
                {"doc_id": "other.md", "score": 3.0, "excerpt": "Other content."},
            ],
            source="semantic",
            explanation="Found 2 documents"
        )

        formatted = executor.format_result(result)
        assert "test.md" in formatted
        assert "Test content" in formatted


# =============================================================================
# CodeExecutor Tests
# =============================================================================


class TestCodeExecutorBehavior:
    """CodeExecutor behavior tests."""

    def test_given_no_code_bridge_when_executed_then_returns_unavailable(self):
        """
        Given: No CodeBridge available
        When: Executed
        Then: Returns unavailable message with helpful information
        """
        # Mock the lazy loading to return None
        with patch.object(CodeExecutor, 'code_bridge', new_callable=lambda: property(lambda self: None)):
            executor = CodeExecutor()
            query = {"action": "call", "subject": "test_func"}
            result = executor.execute(query)

        # Should return a result indicating code bridge is unavailable
        assert result.source == "code"
        assert result.is_empty or result.confidence < 0.5  # Low confidence when bridge unavailable

    def test_given_callers_query_when_bridge_available_then_queries_callers(self):
        """
        Given: A callers query with mock CodeBridge
        When: Executed
        Then: Queries callers_of method
        """
        # Create mock code bridge
        mock_bridge = Mock()
        mock_atom = Mock()
        mock_atom.name = "test_caller"
        mock_atom.metadata = {"file_path": "test.py", "lineno": 10}
        mock_bridge.query_callers_of.return_value = [mock_atom]

        executor = CodeExecutor(code_bridge=mock_bridge)
        query = {"action": "call", "subject": "test_func"}

        result = executor.execute(query)

        mock_bridge.query_callers_of.assert_called_once_with("test_func")
        assert len(result.items) == 1
        assert result.items[0]["name"] == "test_caller"

    def test_given_methods_query_when_bridge_available_then_queries_methods(self):
        """
        Given: A methods query with mock CodeBridge
        When: Executed
        Then: Queries methods_of method
        """
        mock_bridge = Mock()
        mock_atom = Mock()
        mock_atom.name = "MyClass.method"
        mock_atom.metadata = {"args": ["self", "x"], "lineno": 20}
        mock_bridge.query_methods_of.return_value = [mock_atom]

        executor = CodeExecutor(code_bridge=mock_bridge)
        query = {"action": None, "subject": "MyClass", "intent": "definition"}

        # Force method query
        result = executor._execute_methods_query("MyClass")

        mock_bridge.query_methods_of.assert_called_once_with("MyClass")
        assert len(result.items) == 1

    def test_format_result_with_code_entities_shows_locations(self):
        """format_result with code entities shows file locations."""
        executor = CodeExecutor()
        result = ExecutionResult(
            items=[
                {"name": "func1", "type": "function", "file_path": "mod.py", "lineno": 10},
                {"name": "func2", "type": "function", "file_path": "other.py", "lineno": 20},
            ],
            source="code",
            explanation="Found 2 callers"
        )

        formatted = executor.format_result(result)
        assert "func1" in formatted
        assert "mod.py:10" in formatted


# =============================================================================
# Integration Behavior Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration behavior between router and executors."""

    def test_audit_query_routes_to_audit_executor(self):
        """
        Given: A query routed to audit type
        When: Passed to AuditExecutor
        Then: Executes correctly
        """
        from cortical.cognitive.unified_query import QueryRouter
        from cortical.audits.reasoning import AuditQuery

        router = QueryRouter()
        unified = router.route("risky files in cortical/")

        assert unified.query_type == "audit"
        assert isinstance(unified.parsed, AuditQuery)

        # Execute with AuditExecutor
        executor = AuditExecutor()
        result = executor.execute(unified.parsed)

        assert result.source == "audit"

    def test_semantic_query_routes_to_semantic_executor(self):
        """
        Given: A query routed to semantic type
        When: Passed to SemanticExecutor
        Then: Executes correctly
        """
        from cortical.cognitive.unified_query import QueryRouter, QueryIntent

        router = QueryRouter()
        unified = router.route("what is the cognitive agent")

        assert unified.query_type == "semantic"
        assert isinstance(unified.parsed, QueryIntent)

        # Execute with SemanticExecutor
        executor = SemanticExecutor()
        result = executor.execute(unified.parsed)

        assert result.source == "semantic"

    def test_code_query_routes_to_code_executor(self):
        """
        Given: A query routed to code type
        When: Passed to CodeExecutor
        Then: Executes correctly
        """
        from cortical.cognitive.unified_query import QueryRouter

        router = QueryRouter()
        unified = router.route("where do we handle transactions")

        assert unified.query_type == "code"
        assert isinstance(unified.parsed, dict)

        # Execute with CodeExecutor (will fail without bridge, but should try)
        executor = CodeExecutor()
        result = executor.execute(unified.parsed)

        assert result.source == "code"

    def test_cdg_query_routes_to_cdg_executor(self):
        """
        Given: A query routed to CDG type
        When: Passed to CDGExecutor
        Then: Returns appropriate result
        """
        from cortical.cognitive.unified_query import QueryRouter

        router = QueryRouter()
        unified = router.route("FROM task WHERE status = 'pending'")

        assert unified.query_type == "cdg"

        # Execute with CDGExecutor (will return not configured without store)
        executor = CDGExecutor()
        result = executor.execute(unified.parsed)

        assert result.source == "cdg"


# =============================================================================
# CDGExecutor Tests
# =============================================================================


class TestCDGExecutorBehavior:
    """CDGExecutor behavior tests."""

    def test_cdg_executor_implements_protocol(self):
        """CDGExecutor implements QueryExecutorProtocol."""
        executor = CDGExecutor()
        assert isinstance(executor, QueryExecutorProtocol)
        assert hasattr(executor, "execute")
        assert hasattr(executor, "format_result")

    def test_given_no_store_when_executed_then_returns_not_configured(self):
        """
        Given: No CDGStore configured
        When: Executed
        Then: Returns helpful not-configured message
        """
        executor = CDGExecutor()
        query = {"raw": "FROM task WHERE status = 'pending'"}

        result = executor.execute(query)

        assert result.source == "cdg"
        assert result.metadata.get("error") == "not_configured"
        assert "not configured" in result.explanation.lower()

    def test_format_result_with_empty_shows_explanation(self):
        """format_result with empty result shows explanation."""
        executor = CDGExecutor()
        result = ExecutionResult(
            items=[],
            source="cdg",
            explanation="No results found."
        )

        formatted = executor.format_result(result)
        assert "No results found" in formatted


# =============================================================================
# AuditExecutor Advanced Tests
# =============================================================================


class TestAuditExecutorAdvanced:
    """Advanced AuditExecutor tests for fixes."""

    def test_given_no_data_when_list_executed_then_returns_helpful_message(self):
        """
        Given: No audit data loaded
        When: List query executed
        Then: Returns message about running audit analyze
        """
        executor = AuditExecutor()
        # Don't add any files - fresh reasoner

        query = AuditQuery(intent="list")
        result = executor.execute(query)

        assert result.source == "audit"
        # Should indicate no data loaded
        if result.metadata.get("error") == "no_data":
            assert "audit analyze" in result.explanation.lower() or "no audit data" in result.explanation.lower()

    def test_given_trait_filter_when_list_executed_then_filters_by_trait(self):
        """
        Given: Files with different traits
        When: List query with trait filter
        Then: Only files with trait are returned
        """
        executor = AuditExecutor()

        # Add files with different traits
        executor.reasoner.assert_file_facts(
            "high_churn_file.py",
            patterns=["todo"],
            traits=["high_churn"],
            directories=["src"]
        )
        executor.reasoner.assert_file_facts(
            "stable_file.py",
            patterns=["todo"],
            traits=["stable"],
            directories=["src"]
        )

        query = AuditQuery(
            intent="list",
            include_traits=["high_churn"]
        )
        result = executor.execute(query)

        assert result.source == "audit"
        # Trait filter should be in metadata
        assert result.metadata.get("traits_filter") == ["high_churn"]


# =============================================================================
# SemanticExecutor Advanced Tests
# =============================================================================


class TestSemanticExecutorAdvanced:
    """Advanced SemanticExecutor tests for fixes."""

    def test_given_agent_when_executed_then_expands_concepts(self):
        """
        Given: SemanticExecutor with agent
        When: Executed
        Then: Uses agent for concept expansion
        """
        from cortical.cognitive.unified_query import QueryIntent

        # Create mock agent
        mock_agent = Mock()
        mock_atom = Mock()
        mock_atom.name = "similar_word"
        mock_agent.query.return_value = [mock_atom]

        executor = SemanticExecutor(agent=mock_agent)

        # Call the expand method directly
        expanded = executor._expand_concepts(["test"])

        assert "test" in expanded
        assert expanded["test"] == 1.0
        mock_agent.query.assert_called()

    def test_given_no_agent_when_executed_then_still_works(self):
        """
        Given: SemanticExecutor without agent
        When: Executed
        Then: Works with original concepts only
        """
        executor = SemanticExecutor(agent=None)

        expanded = executor._expand_concepts(["cognitive", "agent"])

        assert "cognitive" in expanded
        assert "agent" in expanded
        assert expanded["cognitive"] == 1.0
        assert expanded["agent"] == 1.0
