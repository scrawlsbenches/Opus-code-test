"""
Behavioral tests for ResponseFormatter (Phase 4).

Tests the formatting of aggregated results into natural language responses
across all query types: audit, cdg, code, semantic.
"""

import pytest

from cortical.cognitive.formatter import (
    ResponseFormatter,
    FormatterConfig,
    format_response,
)
from cortical.cognitive.unified_query import UnifiedQuery, QueryIntent
from cortical.cognitive.aggregator import AggregatedResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def formatter():
    """Default formatter instance."""
    return ResponseFormatter()


@pytest.fixture
def verbose_formatter():
    """Formatter with verbose output enabled."""
    config = FormatterConfig(verbose=True, show_scores=True, show_confidence=True)
    return ResponseFormatter(config)


@pytest.fixture
def audit_query():
    """Sample audit query."""
    return UnifiedQuery(
        raw_question="risky files in cortical/",
        query_type="audit",
        parsed={"intent": "list", "directory": "cortical/"},
        confidence=0.8,
    )


@pytest.fixture
def cdg_query():
    """Sample CDG query."""
    return UnifiedQuery(
        raw_question="FROM task WHERE status = 'pending'",
        query_type="cdg",
        parsed={"raw": "FROM task WHERE status = 'pending'"},
        confidence=0.9,
    )


@pytest.fixture
def code_query():
    """Sample code intent query."""
    return UnifiedQuery(
        raw_question="where do we handle transactions",
        query_type="code",
        parsed={"action": "handle", "subject": "transactions"},
        confidence=0.7,
    )


@pytest.fixture
def semantic_query():
    """Sample semantic query."""
    return UnifiedQuery(
        raw_question="what is cognitive agent",
        query_type="semantic",
        parsed=QueryIntent(
            question_type="what",
            concepts=["cognitive", "agent"],
            raw_question="what is cognitive agent",
        ),
        confidence=0.5,
    )


@pytest.fixture
def why_query():
    """Sample 'why' question."""
    return UnifiedQuery(
        raw_question="why is prism_pln.py flagged",
        query_type="audit",
        parsed={"intent": "explain", "target_file": "prism_pln.py"},
        confidence=0.8,
        metadata={"intent": "explain"},
    )


@pytest.fixture
def audit_result():
    """Sample audit aggregated result."""
    return AggregatedResult(
        items=[
            {"file": "cortical/reasoning/prism_pln.py", "risk_score": 0.75},
            {"file": "cortical/got/api.py", "risk_score": 0.62},
            {"file": "cortical/cdg/storage.py", "risk_score": 0.45},
        ],
        sources=["audit"],
        total_confidence=0.8,
        explanation="Found 3 files with average risk score 0.61",
    )


@pytest.fixture
def cdg_result():
    """Sample CDG aggregated result."""
    return AggregatedResult(
        items=[
            {"id": "T-001", "type": "task", "status": "pending", "title": "Fix bug"},
            {"id": "T-002", "type": "task", "status": "pending", "title": "Add feature"},
        ],
        sources=["cdg"],
        total_confidence=0.9,
    )


@pytest.fixture
def code_result():
    """Sample code aggregated result."""
    return AggregatedResult(
        items=[
            {"file_path": "cortical/cdg/transaction_manager.py", "name": "commit", "line_number": 293},
            {"file_path": "cortical/got/versioned_store.py", "name": "with_transaction", "line_number": 138},
        ],
        sources=["code"],
        total_confidence=0.7,
    )


@pytest.fixture
def semantic_result():
    """Sample semantic aggregated result."""
    return AggregatedResult(
        items=[
            {"doc_id": "what_is_cognitive_agent.md", "similarity": 0.92},
            {"doc_id": "cognitive_bridge.py", "similarity": 0.78},
        ],
        sources=["semantic"],
        total_confidence=0.5,
    )


@pytest.fixture
def semantic_result_with_excerpts():
    """Sample semantic result with excerpts from document content."""
    return AggregatedResult(
        items=[
            {
                "doc_id": "what_is_cognitive_agent.md",
                "score": 0.92,
                "excerpt": "The Cognitive Agent is your long-term memory.\nIt learns from documents and code to help you recover context.",
            },
            {
                "doc_id": "cognitive_bridge.py",
                "score": 0.78,
                "excerpt": "class CognitiveBridge:\n    '''Bridge between code indexer and cognitive graph.'''",
            },
        ],
        sources=["semantic"],
        total_confidence=0.7,
    )


@pytest.fixture
def pln_explanation_result():
    """Sample PLN explanation result for 'why' questions."""
    return AggregatedResult(
        items=[{
            "file_id": "prism_pln_py",
            "facts": [
                {"atom": "has_trait(prism_pln_py, complexity)", "strength": 0.8, "confidence": 0.9},
                {"atom": "has_pattern(prism_pln_py, todo)", "strength": 0.6, "confidence": 0.8},
            ],
            "risk_level": {"mean": 0.72, "strength": 0.8, "confidence": 0.85},
            "inferences": ["complexity + todos => risk"],
            "suggestions": ["Review and address TODO comments"],
            "summary": "File flagged due to complexity and pending TODOs",
        }],
        sources=["audit"],
        total_confidence=0.8,
    )


# =============================================================================
# Test: Empty Results
# =============================================================================

class TestEmptyResults:
    """Tests for empty result formatting."""

    def test_given_empty_audit_result_when_formatted_then_shows_hint(self, formatter, audit_query):
        """Empty audit results should suggest running 'audit analyze'."""
        empty_result = AggregatedResult()

        response = formatter.format(audit_query, empty_result)

        assert "No audit results found" in response
        assert "audit analyze" in response

    def test_given_empty_cdg_result_when_formatted_then_shows_no_entities(self, formatter, cdg_query):
        """Empty CDG results should mention no entities found."""
        empty_result = AggregatedResult()

        response = formatter.format(cdg_query, empty_result)

        assert "No matching entities found" in response

    def test_given_empty_code_result_when_formatted_then_shows_no_locations(self, formatter, code_query):
        """Empty code results should mention no code locations."""
        empty_result = AggregatedResult()

        response = formatter.format(code_query, empty_result)

        assert "No code locations found" in response

    def test_given_empty_semantic_result_when_formatted_then_shows_no_results(self, formatter, semantic_query):
        """Empty semantic results should show generic no results message."""
        empty_result = AggregatedResult()

        response = formatter.format(semantic_query, empty_result)

        assert "No results found" in response


# =============================================================================
# Test: Audit Response Formatting
# =============================================================================

class TestAuditFormatting:
    """Tests for audit query response formatting."""

    def test_given_audit_results_when_formatted_then_shows_file_count(self, formatter, audit_query, audit_result):
        """Audit response should show number of files found."""
        response = formatter.format(audit_query, audit_result)

        assert "Found 3 files:" in response

    def test_given_audit_results_when_formatted_then_lists_files(self, formatter, audit_query, audit_result):
        """Audit response should list file names."""
        response = formatter.format(audit_query, audit_result)

        assert "prism_pln.py" in response
        assert "api.py" in response
        assert "storage.py" in response

    def test_given_audit_results_when_formatted_then_shows_risk_category(self, formatter, audit_query, audit_result):
        """Audit response should categorize risk levels."""
        response = formatter.format(audit_query, audit_result)

        # 0.75 = HIGH, 0.62 = HIGH, 0.45 = MODERATE
        assert "HIGH" in response
        assert "MODERATE" in response

    def test_given_show_scores_config_when_formatted_then_shows_numeric_scores(self, verbose_formatter, audit_query, audit_result):
        """With show_scores=True, should display numeric risk scores."""
        response = verbose_formatter.format(audit_query, audit_result)

        assert "0.75" in response
        assert "0.62" in response

    def test_given_single_file_when_formatted_then_uses_singular(self, formatter, audit_query):
        """Single file should use 'file' not 'files'."""
        single_result = AggregatedResult(
            items=[{"file": "test.py", "risk_score": 0.5}],
            sources=["audit"],
            total_confidence=0.8,
        )

        response = formatter.format(audit_query, single_result)

        assert "Found 1 file:" in response

    def test_given_many_results_when_formatted_then_truncates_with_notice(self, formatter, audit_query):
        """More than max_items should show truncation notice."""
        many_items = [{"file": f"file{i}.py", "risk_score": 0.5} for i in range(15)]
        large_result = AggregatedResult(items=many_items, sources=["audit"], total_confidence=0.8)

        response = formatter.format(audit_query, large_result)

        assert "... and 5 more files" in response


# =============================================================================
# Test: CDG Response Formatting
# =============================================================================

class TestCDGFormatting:
    """Tests for CDG query response formatting."""

    def test_given_cdg_results_when_formatted_then_shows_query(self, formatter, cdg_query, cdg_result):
        """CDG response should echo the original query."""
        response = formatter.format(cdg_query, cdg_result)

        assert "Query:" in response
        assert "FROM task WHERE status" in response

    def test_given_cdg_results_when_formatted_then_shows_entity_count(self, formatter, cdg_query, cdg_result):
        """CDG response should show entity count."""
        response = formatter.format(cdg_query, cdg_result)

        assert "Found 2 entities:" in response

    def test_given_cdg_results_when_formatted_then_shows_entity_type(self, formatter, cdg_query, cdg_result):
        """CDG response should show entity type in brackets."""
        response = formatter.format(cdg_query, cdg_result)

        assert "[task]" in response
        assert "T-001" in response
        assert "T-002" in response

    def test_given_verbose_config_when_formatted_then_shows_properties(self, verbose_formatter, cdg_query, cdg_result):
        """Verbose mode should show entity properties."""
        response = verbose_formatter.format(cdg_query, cdg_result)

        assert "status: pending" in response


# =============================================================================
# Test: Code Response Formatting
# =============================================================================

class TestCodeFormatting:
    """Tests for code intent response formatting."""

    def test_given_code_results_when_formatted_then_shows_match_count(self, formatter, code_query, code_result):
        """Code response should show match count."""
        response = formatter.format(code_query, code_result)

        assert "Found 2 matches:" in response

    def test_given_code_results_when_formatted_then_shows_locations(self, formatter, code_query, code_result):
        """Code response should show file:line locations."""
        response = formatter.format(code_query, code_result)

        assert "transaction_manager.py:293" in response
        assert "versioned_store.py:138" in response

    def test_given_code_results_when_formatted_then_shows_function_names(self, formatter, code_query, code_result):
        """Code response should show function names."""
        response = formatter.format(code_query, code_result)

        assert "commit" in response
        assert "with_transaction" in response

    def test_given_code_result_without_line_number_when_formatted_then_shows_file_only(self, formatter, code_query):
        """Code results without line numbers should still display."""
        result = AggregatedResult(
            items=[{"file_path": "some/file.py", "name": "function"}],
            sources=["code"],
            total_confidence=0.7,
        )

        response = formatter.format(code_query, result)

        assert "function at some/file.py" in response
        # Should NOT have colon without line number
        assert "file.py:" not in response


# =============================================================================
# Test: Semantic Response Formatting
# =============================================================================

class TestSemanticFormatting:
    """Tests for semantic query response formatting."""

    def test_given_semantic_results_when_formatted_then_shows_question(self, formatter, semantic_query, semantic_result):
        """Semantic response should show original question."""
        response = formatter.format(semantic_query, semantic_result)

        assert "what is cognitive agent" in response

    def test_given_semantic_results_when_formatted_then_shows_related_count(self, formatter, semantic_query, semantic_result):
        """Semantic response should show count of related documents."""
        response = formatter.format(semantic_query, semantic_result)

        assert "Found 2 related documents:" in response

    def test_given_semantic_results_when_formatted_then_shows_doc_ids(self, formatter, semantic_query, semantic_result):
        """Semantic response should show document IDs."""
        response = formatter.format(semantic_query, semantic_result)

        assert "what_is_cognitive_agent.md" in response
        assert "cognitive_bridge.py" in response

    def test_given_show_scores_when_formatted_then_shows_relevance(self, verbose_formatter, semantic_query, semantic_result):
        """With show_scores, should show relevance scores."""
        response = verbose_formatter.format(semantic_query, semantic_result)

        assert "relevance: 0.92" in response
        assert "relevance: 0.78" in response

    def test_given_zero_similarity_when_formatted_then_still_shows_score(self, verbose_formatter, semantic_query):
        """Zero similarity should still be displayed (not skipped as falsy)."""
        result = AggregatedResult(
            items=[{"doc_id": "test.md", "similarity": 0.0}],
            sources=["semantic"],
            total_confidence=0.5,
        )

        response = verbose_formatter.format(semantic_query, result)

        assert "relevance: 0.00" in response

    def test_given_semantic_results_with_excerpts_when_formatted_then_shows_excerpts(
        self, formatter, semantic_query, semantic_result_with_excerpts
    ):
        """Semantic results should display document excerpts - the useful content."""
        response = formatter.format(semantic_query, semantic_result_with_excerpts)

        # Excerpt content should be visible, not just file names
        assert "long-term memory" in response
        assert "CognitiveBridge" in response

    def test_given_multiline_excerpt_when_formatted_then_shows_multiple_lines(
        self, formatter, semantic_query, semantic_result_with_excerpts
    ):
        """Multiline excerpts should show multiple lines."""
        response = formatter.format(semantic_query, semantic_result_with_excerpts)

        # Both lines of the first excerpt should be present
        assert "Cognitive Agent is your long-term memory" in response
        assert "documents and code" in response


# =============================================================================
# Test: Why Question Formatting
# =============================================================================

class TestWhyQuestionFormatting:
    """Tests for 'why' question explanation formatting."""

    def test_given_why_question_when_formatted_then_detected_as_why(self, formatter, why_query, pln_explanation_result):
        """'Why' questions should be detected and use explanation format."""
        response = formatter.format(why_query, pln_explanation_result)

        assert "Risk Analysis:" in response

    def test_given_pln_explanation_when_formatted_then_shows_risk_level(self, formatter, why_query, pln_explanation_result):
        """PLN explanation should show overall risk level."""
        response = formatter.format(why_query, pln_explanation_result)

        # 0.72 = HIGH
        assert "HIGH" in response
        assert "0.72" in response

    def test_given_pln_explanation_when_formatted_then_shows_evidence(self, formatter, why_query, pln_explanation_result):
        """PLN explanation should show evidence facts."""
        response = formatter.format(why_query, pln_explanation_result)

        assert "Evidence" in response
        assert "facts" in response.lower()

    def test_given_pln_explanation_when_formatted_then_shows_suggestions(self, formatter, why_query, pln_explanation_result):
        """PLN explanation should show recommendations."""
        response = formatter.format(why_query, pln_explanation_result)

        assert "Recommendations:" in response
        assert "TODO" in response

    def test_given_pln_explanation_when_formatted_then_shows_summary(self, formatter, why_query, pln_explanation_result):
        """PLN explanation should show summary at end."""
        response = formatter.format(why_query, pln_explanation_result)

        assert "Summary:" in response
        assert "complexity" in response.lower()

    def test_given_question_with_why_in_middle_when_formatted_then_detected(self, formatter, audit_result):
        """Questions with 'why' in middle should also be detected."""
        query = UnifiedQuery(
            raw_question="explain why this file is risky",
            query_type="audit",
            parsed={},
            confidence=0.8,
        )
        # Use non-PLN result to test generic why handling
        generic_result = AggregatedResult(
            items=[{"reason": "File has high complexity", "file": "test.py"}],
            sources=["audit"],
            total_confidence=0.8,
        )

        response = formatter.format(query, generic_result)

        assert "Explanation for:" in response

    def test_given_explain_intent_in_metadata_when_formatted_then_uses_why_format(self, formatter):
        """Metadata intent='explain' should trigger why formatting."""
        query = UnifiedQuery(
            raw_question="tell me about file.py",
            query_type="audit",
            parsed={},
            confidence=0.8,
            metadata={"intent": "explain"},
        )
        result = AggregatedResult(
            items=[{"summary": "This file is critical infrastructure"}],
            sources=["audit"],
            total_confidence=0.8,
        )

        response = formatter.format(query, result)

        assert "Explanation for:" in response


# =============================================================================
# Test: PLN Fact Readability
# =============================================================================

class TestPLNFactReadability:
    """Tests for converting PLN atoms to readable text."""

    def test_has_trait_atom_converted_to_readable(self, formatter):
        """has_trait(file, trait) should become readable."""
        atom = "has_trait(prism_pln_py, complexity)"

        readable = formatter._make_fact_readable(atom)

        assert readable == "prism_pln_py has trait: complexity"

    def test_has_pattern_atom_converted_to_readable(self, formatter):
        """has_pattern(file, pattern) should become readable."""
        atom = "has_pattern(test_py, todo)"

        readable = formatter._make_fact_readable(atom)

        assert readable == "test_py contains: todo"

    def test_is_risky_atom_converted_to_readable(self, formatter):
        """is_risky(file) should become readable."""
        atom = "is_risky(critical_py)"

        readable = formatter._make_fact_readable(atom)

        assert readable == "critical_py is marked as risky"

    def test_unknown_atom_returned_unchanged(self, formatter):
        """Unknown atom patterns should be returned as-is."""
        atom = "some_other_predicate(arg1, arg2)"

        readable = formatter._make_fact_readable(atom)

        assert readable == atom

    def test_malformed_atom_returned_unchanged(self, formatter):
        """Malformed atoms should be returned as-is."""
        atom = "has_trait(incomplete"

        readable = formatter._make_fact_readable(atom)

        assert readable == atom


# =============================================================================
# Test: Risk Categorization
# =============================================================================

class TestRiskCategorization:
    """Tests for risk score to category conversion."""

    def test_critical_risk_at_0_8(self, formatter):
        """Score >= 0.8 should be CRITICAL."""
        assert formatter._categorize_risk(0.8) == "CRITICAL"
        assert formatter._categorize_risk(0.95) == "CRITICAL"
        assert formatter._categorize_risk(1.0) == "CRITICAL"

    def test_high_risk_at_0_6(self, formatter):
        """Score >= 0.6 and < 0.8 should be HIGH."""
        assert formatter._categorize_risk(0.6) == "HIGH"
        assert formatter._categorize_risk(0.79) == "HIGH"

    def test_moderate_risk_at_0_4(self, formatter):
        """Score >= 0.4 and < 0.6 should be MODERATE."""
        assert formatter._categorize_risk(0.4) == "MODERATE"
        assert formatter._categorize_risk(0.59) == "MODERATE"

    def test_low_risk_at_0_2(self, formatter):
        """Score >= 0.2 and < 0.4 should be LOW."""
        assert formatter._categorize_risk(0.2) == "LOW"
        assert formatter._categorize_risk(0.39) == "LOW"

    def test_minimal_risk_below_0_2(self, formatter):
        """Score < 0.2 should be MINIMAL."""
        assert formatter._categorize_risk(0.19) == "MINIMAL"
        assert formatter._categorize_risk(0.0) == "MINIMAL"


# =============================================================================
# Test: Configuration
# =============================================================================

class TestFormatterConfiguration:
    """Tests for formatter configuration options."""

    def test_default_config_values(self):
        """Default config should have sensible defaults."""
        config = FormatterConfig()

        assert config.max_items_shown == 10
        assert config.show_confidence is False
        assert config.show_sources is True
        assert config.show_scores is False
        assert config.verbose is False

    def test_custom_max_items_limits_output(self, audit_query):
        """Custom max_items should limit displayed items."""
        config = FormatterConfig(max_items_shown=2)
        formatter = ResponseFormatter(config)
        result = AggregatedResult(
            items=[{"file": f"file{i}.py", "risk_score": 0.5} for i in range(5)],
            sources=["audit"],
            total_confidence=0.8,
        )

        response = formatter.format(audit_query, result)

        assert "file0.py" in response
        assert "file1.py" in response
        assert "file2.py" not in response
        assert "... and 3 more files" in response

    def test_show_confidence_adds_confidence_line(self, audit_query, audit_result):
        """show_confidence=True should add confidence to output."""
        config = FormatterConfig(show_confidence=True)
        formatter = ResponseFormatter(config)

        response = formatter.format(audit_query, audit_result)

        assert "Confidence: 0.80" in response


# =============================================================================
# Test: Convenience Function
# =============================================================================

class TestConvenienceFunction:
    """Tests for format_response convenience function."""

    def test_format_response_works_without_config(self, audit_query, audit_result):
        """format_response should work with default config."""
        response = format_response(audit_query, audit_result)

        assert "Found 3 files:" in response

    def test_format_response_accepts_config(self, audit_query, audit_result):
        """format_response should accept custom config."""
        config = FormatterConfig(show_scores=True)

        response = format_response(audit_query, audit_result, config)

        assert "0.75" in response


# =============================================================================
# Test: Generic Fallback
# =============================================================================

class TestGenericFallback:
    """Tests for generic/unknown query type formatting."""

    def test_unknown_query_type_uses_generic_format(self, formatter):
        """Unknown query types should use generic formatting."""
        query = UnifiedQuery(
            raw_question="some custom query",
            query_type="unknown_type",
            parsed={},
            confidence=0.5,
        )
        result = AggregatedResult(
            items=[{"name": "item1"}, {"name": "item2"}],
            sources=["unknown"],
            total_confidence=0.5,
        )

        response = formatter.format(query, result)

        assert "Results for: some custom query" in response
        assert "Found 2 items:" in response
        assert "item1" in response
        assert "item2" in response

    def test_generic_format_shows_sources(self, formatter):
        """Generic format should show sources when enabled."""
        query = UnifiedQuery(
            raw_question="query",
            query_type="custom",
            parsed={},
            confidence=0.5,
        )
        result = AggregatedResult(
            items=[{"id": "1"}],
            sources=["source1", "source2"],
            total_confidence=0.5,
        )

        response = formatter.format(query, result)

        assert "Sources: source1, source2" in response
