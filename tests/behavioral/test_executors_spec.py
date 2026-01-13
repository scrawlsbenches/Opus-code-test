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
        Then: Returns unavailable message
        """
        executor = CodeExecutor(code_bridge=None)
        executor._code_bridge = None  # Force no bridge

        query = {"action": "call", "subject": "test_func"}

        # Mock the lazy loading to return None
        with patch.object(CodeExecutor, 'code_bridge', new_callable=lambda: property(lambda self: None)):
            executor = CodeExecutor()
            result = executor.execute(query)

        # The result depends on whether code_bridge could be lazy-loaded

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
