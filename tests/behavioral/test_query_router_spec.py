"""
Behavioral tests for Query Router.

Tests the routing logic that determines which backend handles a question.
"""

import pytest
from cortical.cognitive.unified_query import (
    QueryRouter,
    UnifiedQuery,
    QueryIntent,
    route_question,
    get_query_type,
)
from cortical.audits.reasoning import AuditQuery


# =============================================================================
# Query Router Specs
# =============================================================================

class TestQueryRouterCDGRouting:
    """CDG query routing behavior."""

    def test_given_from_clause_when_routed_then_cdg_type(self):
        """
        Given: A query with FROM clause
        When: Routed
        Then: Returns CDG query type
        """
        router = QueryRouter()
        result = router.route("FROM task WHERE status = 'pending'")

        assert result.query_type == "cdg"
        assert result.confidence == 0.9

    def test_given_blockers_function_when_routed_then_cdg_type(self):
        """
        Given: A query with blockers() function
        When: Routed
        Then: Returns CDG query type
        """
        router = QueryRouter()
        result = router.route("blockers('T-123')")

        assert result.query_type == "cdg"

    def test_given_where_clause_when_routed_then_cdg_type(self):
        """
        Given: A query with WHERE clause
        When: Routed
        Then: Returns CDG query type
        """
        router = QueryRouter()
        result = router.route("WHERE status = 'completed'")

        assert result.query_type == "cdg"


class TestQueryRouterAuditRouting:
    """Audit query routing behavior."""

    def test_given_risky_files_when_routed_then_audit_type(self):
        """
        Given: A query about risky files
        When: Routed
        Then: Returns audit query type with AuditQuery parsed
        """
        router = QueryRouter()
        result = router.route("risky files in cortical/")

        assert result.query_type == "audit"
        assert result.confidence == 0.8
        assert isinstance(result.parsed, AuditQuery)
        assert result.parsed.directory == "cortical/"

    def test_given_why_flagged_when_routed_then_audit_type(self):
        """
        Given: A 'why is X flagged' query
        When: Routed
        Then: Returns audit query type with explain intent
        """
        router = QueryRouter()
        result = router.route("why is prism_pln.py flagged")

        assert result.query_type == "audit"
        assert isinstance(result.parsed, AuditQuery)
        assert result.parsed.intent == "explain"

    def test_given_high_churn_when_routed_then_audit_type(self):
        """
        Given: A query about high_churn trait
        When: Routed
        Then: Returns audit query type
        """
        router = QueryRouter()
        result = router.route("files with high_churn in reasoning/")

        assert result.query_type == "audit"
        assert "high_churn" in result.parsed.include_traits


class TestQueryRouterCodeRouting:
    """Code intent routing behavior."""

    def test_given_where_handle_when_routed_then_code_type(self):
        """
        Given: A 'where do we handle X' query
        When: Routed
        Then: Returns code query type
        """
        router = QueryRouter()
        result = router.route("where do we handle transactions")

        assert result.query_type == "code"
        assert result.confidence == 0.7
        assert result.parsed["action"] == "handle"
        assert result.parsed["subject"] == "transactions"

    def test_given_how_implement_when_routed_then_code_type(self):
        """
        Given: A 'how does X implement Y' query
        When: Routed
        Then: Returns code query type
        """
        router = QueryRouter()
        result = router.route("how does the system implement caching")

        assert result.query_type == "code"
        assert result.parsed["action"] == "implement"

    def test_given_who_calls_when_routed_then_code_type(self):
        """
        Given: A 'who calls X' query
        When: Routed
        Then: Returns code query type
        """
        router = QueryRouter()
        result = router.route("who calls the validate function")

        assert result.query_type == "code"


class TestQueryRouterSemanticRouting:
    """Semantic (fallback) routing behavior."""

    def test_given_what_is_when_routed_then_semantic_type(self):
        """
        Given: A 'what is X' query (no specific pattern)
        When: Routed
        Then: Returns semantic query type
        """
        router = QueryRouter()
        result = router.route("what is the cognitive agent")

        assert result.query_type == "semantic"
        assert result.confidence == 0.5
        assert isinstance(result.parsed, QueryIntent)
        assert result.parsed.question_type == "what"

    def test_given_simple_topic_when_routed_then_semantic_type(self):
        """
        Given: A simple topic query
        When: Routed
        Then: Returns semantic query type with concepts extracted
        """
        router = QueryRouter()
        result = router.route("tell me about PRISM reasoning")

        assert result.query_type == "semantic"
        assert "prism" in result.parsed.concepts
        assert "reasoning" in result.parsed.concepts

    def test_given_how_does_work_when_routed_then_semantic_type(self):
        """
        Given: A 'how does X work' query (general, not code-specific)
        When: Routed
        Then: Returns semantic query type
        """
        router = QueryRouter()
        result = router.route("how does WAL work")

        # General "how does X work" without handle/implement/process
        assert result.query_type == "semantic"
        assert result.parsed.question_type == "how"


class TestQueryRouterConceptExtraction:
    """Concept extraction behavior."""

    def test_given_question_when_parsed_then_stop_words_removed(self):
        """
        Given: A question with stop words
        When: Parsed for concepts
        Then: Stop words are removed
        """
        router = QueryRouter()
        result = router.route("what is the purpose of the cognitive agent")

        # "what", "is", "the", "of" should be removed
        assert "what" not in result.parsed.concepts
        assert "the" not in result.parsed.concepts
        assert "purpose" in result.parsed.concepts
        assert "cognitive" in result.parsed.concepts
        assert "agent" in result.parsed.concepts

    def test_given_question_when_parsed_then_short_words_removed(self):
        """
        Given: A question with short words
        When: Parsed for concepts
        Then: Words <= 2 chars are removed
        """
        router = QueryRouter()
        result = router.route("is it a good API")

        # "is", "it", "a" should be removed (stop words or short)
        assert "good" in result.parsed.concepts
        assert "api" in result.parsed.concepts


class TestConvenienceFunctions:
    """Convenience function behavior."""

    def test_route_question_returns_unified_query(self):
        """route_question() returns UnifiedQuery."""
        result = route_question("what is PRISM")
        assert isinstance(result, UnifiedQuery)

    def test_get_query_type_returns_string(self):
        """get_query_type() returns type string."""
        result = get_query_type("risky files in cortical/")
        assert result == "audit"

        result = get_query_type("FROM task WHERE status = 'pending'")
        assert result == "cdg"

        result = get_query_type("what is the cognitive agent")
        assert result == "semantic"
