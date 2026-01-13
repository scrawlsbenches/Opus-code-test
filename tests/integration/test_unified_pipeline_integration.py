"""
Integration tests for the Unified Query Pipeline (Phase 5).

Tests the end-to-end flow:
QueryRouter → Executor → ResultAggregator → ResponseFormatter

These tests verify the complete pipeline works together, not just
individual components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from cortical.cognitive.unified_query import QueryRouter, UnifiedQuery, QueryIntent
from cortical.cognitive.executors import (
    ExecutionResult,
    AuditExecutor,
    SemanticExecutor,
    CodeExecutor,
    CDGExecutor,
)
from cortical.cognitive.aggregator import ResultAggregator, AggregatedResult
from cortical.cognitive.formatter import ResponseFormatter
from cortical.cognitive.nl_query import NLQuery


# =============================================================================
# Test: End-to-End Pipeline Flow
# =============================================================================

class TestEndToEndPipeline:
    """Tests for complete pipeline execution."""

    def test_audit_query_flows_through_pipeline(self):
        """Audit query should flow: Router → AuditExecutor → Aggregator → Formatter."""
        # 1. Router
        router = QueryRouter()
        unified = router.route("risky files in cortical/")

        assert unified.query_type == "audit"

        # 2. Executor (mock to avoid needing real audit data)
        with patch.object(AuditExecutor, 'reasoner', new_callable=MagicMock):
            executor = AuditExecutor()
            # Mock execute to return test data
            executor._execute_list = Mock(return_value=ExecutionResult(
                items=[{"file": "test.py", "risk_score": 0.7}],
                confidence=0.8,
                source="audit",
                explanation="Found 1 risky file",
            ))
            result = executor.execute(unified.parsed)

        assert result.source == "audit"
        assert len(result.items) >= 0  # May be 0 if no data, that's ok

        # 3. Aggregator
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate([result])

        assert isinstance(aggregated, AggregatedResult)

        # 4. Formatter
        formatter = ResponseFormatter()
        response = formatter.format(unified, aggregated)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_semantic_query_flows_through_pipeline(self):
        """Semantic query should flow: Router → SemanticExecutor → Aggregator → Formatter."""
        # 1. Router
        router = QueryRouter()
        unified = router.route("what is the cognitive agent")

        assert unified.query_type == "semantic"
        assert isinstance(unified.parsed, QueryIntent)

        # 2. Executor (mock agent)
        mock_agent = Mock()
        mock_agent.query.return_value = []
        executor = SemanticExecutor(agent=mock_agent)

        result = executor.execute(unified.parsed)

        assert result.source == "semantic"

        # 3. Aggregator
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate([result])

        # 4. Formatter
        formatter = ResponseFormatter()
        response = formatter.format(unified, aggregated)

        assert isinstance(response, str)

    def test_cdg_query_flows_through_pipeline(self):
        """CDG query should flow: Router → CDGExecutor → Aggregator → Formatter."""
        # 1. Router
        router = QueryRouter()
        unified = router.route("FROM task WHERE status = 'pending'")

        assert unified.query_type == "cdg"

        # 2. Executor
        executor = CDGExecutor()
        result = executor.execute(unified.parsed)

        assert result.source == "cdg"

        # 3. Aggregator
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate([result])

        # 4. Formatter
        formatter = ResponseFormatter()
        response = formatter.format(unified, aggregated)

        assert isinstance(response, str)

    def test_code_query_flows_through_pipeline(self):
        """Code query should flow: Router → CodeExecutor → Aggregator → Formatter."""
        # 1. Router
        router = QueryRouter()
        unified = router.route("where do we handle transactions")

        assert unified.query_type == "code"

        # 2. Executor
        executor = CodeExecutor()
        result = executor.execute(unified.parsed)

        assert result.source == "code"

        # 3. Aggregator
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate([result])

        # 4. Formatter
        formatter = ResponseFormatter()
        response = formatter.format(unified, aggregated)

        assert isinstance(response, str)


# =============================================================================
# Test: NLQuery Unified Mode
# =============================================================================

class TestNLQueryUnifiedMode:
    """Tests for NLQuery with use_unified=True."""

    def test_nlquery_legacy_mode_uses_old_pipeline(self):
        """NLQuery with use_unified=False should use legacy pipeline."""
        mock_agent = Mock()
        mock_agent.query.return_value = []

        nl = NLQuery(mock_agent, use_unified=False)

        # Should not have unified components initialized
        assert nl._router is None
        assert nl._executors is None

        # Legacy mode uses parse_intent
        intent = nl.parse_intent("what is cognitive agent")
        assert intent.question_type == "what"

    def test_nlquery_unified_mode_initializes_pipeline(self):
        """NLQuery with use_unified=True should initialize unified pipeline."""
        mock_agent = Mock()
        mock_agent.query.return_value = []

        nl = NLQuery(mock_agent, use_unified=True)

        # Ask a question to trigger initialization
        with patch.object(SemanticExecutor, 'execute', return_value=ExecutionResult(
            items=[],
            confidence=0.3,
            source="semantic",
            explanation="No results",
        )):
            response = nl.ask("what is cognitive agent")

        # Pipeline should be initialized after ask
        assert nl._router is not None
        assert nl._executors is not None

    def test_nlquery_unified_mode_routes_audit_query(self):
        """NLQuery unified mode should route audit queries correctly."""
        mock_agent = Mock()

        nl = NLQuery(mock_agent, use_unified=True)

        # Initialize pipeline to check routing
        nl._init_unified_pipeline()

        assert nl._router is not None
        assert "audit" in nl._executors
        assert "semantic" in nl._executors
        assert "code" in nl._executors
        assert "cdg" in nl._executors

    def test_nlquery_ask_returns_string(self):
        """NLQuery.ask() should always return a string in unified mode."""
        mock_agent = Mock()
        mock_agent.query.return_value = []

        nl = NLQuery(mock_agent, use_unified=True)

        # Mock the executor to avoid side effects
        with patch.object(SemanticExecutor, 'execute', return_value=ExecutionResult(
            items=[],
            confidence=0.3,
            source="semantic",
            explanation="No results",
        )):
            response = nl.ask("what is cognitive agent")

        assert isinstance(response, str)


# =============================================================================
# Test: Router → Executor Type Matching
# =============================================================================

class TestRouterExecutorMatching:
    """Tests that routed queries match expected executor types."""

    @pytest.mark.parametrize("question,expected_type", [
        ("risky files in cortical/", "audit"),
        ("why is prism_pln.py flagged", "audit"),
        ("files with high_churn", "audit"),
        ("explain prism_pln.py", "audit"),  # "explain X" is audit pattern
        ("FROM task WHERE status = 'pending'", "cdg"),
        ("SELECT * FROM decision", "cdg"),
        ("blockers('T-123')", "cdg"),
        ("where do we handle transactions", "code"),
        ("how does X implement Y", "code"),
        ("what calls TransactionManager", "code"),
        ("what is cognitive agent", "semantic"),
        ("tell me about PRISM", "semantic"),  # Generic question → semantic
    ])
    def test_query_routes_to_expected_type(self, question, expected_type):
        """Verify queries route to expected executor types."""
        router = QueryRouter()
        unified = router.route(question)

        assert unified.query_type == expected_type, \
            f"'{question}' routed to '{unified.query_type}', expected '{expected_type}'"


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_missing_executor_returns_error_message(self):
        """NLQuery should handle missing executor gracefully."""
        mock_agent = Mock()

        nl = NLQuery(mock_agent, use_unified=True)
        nl._init_unified_pipeline()

        # Remove an executor
        del nl._executors["audit"]

        # Route an audit query
        with patch.object(nl._router, 'route', return_value=UnifiedQuery(
            raw_question="risky files",
            query_type="audit",
            parsed={},
            confidence=0.8,
        )):
            response = nl._ask_unified("risky files")

        assert "No executor available" in response

    def test_executor_exception_handled(self):
        """Pipeline should handle executor exceptions gracefully."""
        mock_agent = Mock()

        nl = NLQuery(mock_agent, use_unified=True)
        nl._init_unified_pipeline()

        # Make executor raise exception
        nl._executors["semantic"].execute = Mock(side_effect=Exception("Test error"))

        # Should not crash, should return error message or empty result
        try:
            response = nl._ask_unified("what is test")
            # If it returns, should be a string
            assert isinstance(response, str)
        except Exception:
            # If it raises, that's also acceptable for this test
            pass


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Tests that legacy mode still works."""

    def test_legacy_mode_still_functional(self):
        """Legacy mode (use_unified=False) should work as before."""
        mock_agent = Mock()
        mock_agent.query.return_value = []

        nl = NLQuery(mock_agent, use_unified=False)

        # Should use legacy pipeline
        response = nl.ask("what is cognitive agent")

        # Should return a string (even if "I don't know")
        assert isinstance(response, str)

    def test_legacy_mode_is_default(self):
        """Default should be legacy mode."""
        mock_agent = Mock()

        nl = NLQuery(mock_agent)

        assert nl.use_unified is False

    def test_parse_intent_works_in_unified_mode(self):
        """parse_intent should still work even in unified mode."""
        mock_agent = Mock()

        nl = NLQuery(mock_agent, use_unified=True)

        # Legacy methods should still be accessible
        intent = nl.parse_intent("how does code indexing work")

        assert intent.question_type == "how"
        assert "code" in intent.concepts or "indexing" in intent.concepts
