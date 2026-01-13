"""
Behavioral tests for Result Aggregator (Phase 3).

Tests the aggregation of results from multiple executors:
- Deduplication across sources
- Confidence-based ranking
- Multiple aggregation strategies
"""

import pytest
from typing import Dict, Any, List

from cortical.cognitive.aggregator import (
    AggregatedResult,
    ResultAggregator,
    aggregate_results,
)
from cortical.cognitive.executors.protocol import ExecutionResult


# =============================================================================
# AggregatedResult Tests
# =============================================================================


class TestAggregatedResult:
    """AggregatedResult dataclass behavior."""

    def test_given_empty_items_when_checking_is_empty_then_true(self):
        """
        Given: An AggregatedResult with no items
        When: Checking is_empty
        Then: Returns True
        """
        result = AggregatedResult()
        assert result.is_empty is True

    def test_given_items_when_checking_is_empty_then_false(self):
        """
        Given: An AggregatedResult with items
        When: Checking is_empty
        Then: Returns False
        """
        result = AggregatedResult(items=[{"id": "1"}, {"id": "2"}])
        assert result.is_empty is False

    def test_given_items_when_len_then_returns_count(self):
        """
        Given: An AggregatedResult with items
        When: Getting len()
        Then: Returns item count
        """
        result = AggregatedResult(items=[{"a": 1}, {"b": 2}, {"c": 3}])
        assert len(result) == 3

    def test_given_defaults_when_created_then_has_correct_defaults(self):
        """
        Given: An AggregatedResult created with defaults
        When: Checking properties
        Then: Has expected default values
        """
        result = AggregatedResult()
        assert result.items == []
        assert result.sources == []
        assert result.total_confidence == 0.0
        assert result.explanation == ""
        assert result.source_results == {}


# =============================================================================
# ResultAggregator Merge Strategy Tests
# =============================================================================


class TestMergeStrategy:
    """Tests for the default 'merge' aggregation strategy."""

    def test_given_single_result_when_merged_then_returns_items(self):
        """
        Given: A single ExecutionResult
        When: Aggregated with merge strategy
        Then: Returns all items from that result
        """
        exec_result = ExecutionResult(
            items=[{"id": "1", "score": 0.9}, {"id": "2", "score": 0.7}],
            confidence=0.8,
            source="audit",
            explanation="Found 2 risky files",
        )

        aggregator = ResultAggregator(strategy="merge")
        result = aggregator.aggregate([exec_result])

        assert len(result.items) == 2
        assert result.sources == ["audit"]
        assert result.total_confidence == 0.8

    def test_given_multiple_results_when_merged_then_combines_all(self):
        """
        Given: Multiple ExecutionResults from different sources
        When: Aggregated with merge strategy
        Then: Combines items from all sources
        """
        audit_result = ExecutionResult(
            items=[{"file": "a.py", "score": 0.9}],
            confidence=0.8,
            source="audit",
        )
        semantic_result = ExecutionResult(
            items=[{"doc_id": "b.md", "score": 5.0}],
            confidence=0.7,
            source="semantic",
        )

        aggregator = ResultAggregator(strategy="merge")
        result = aggregator.aggregate([audit_result, semantic_result])

        assert len(result.items) == 2
        assert "audit" in result.sources
        assert "semantic" in result.sources

    def test_given_duplicate_items_when_merged_then_deduplicates(self):
        """
        Given: Results with duplicate items (same key)
        When: Aggregated with merge strategy
        Then: Deduplicates by key, keeping highest-scored
        """
        result1 = ExecutionResult(
            items=[{"file": "same.py", "score": 0.9}],
            confidence=0.8,
            source="audit",
        )
        result2 = ExecutionResult(
            items=[{"file": "same.py", "score": 0.5}],
            confidence=0.6,
            source="audit",
        )

        aggregator = ResultAggregator(strategy="merge")
        result = aggregator.aggregate([result1, result2])

        # Should have only one item (deduplicated)
        assert len(result.items) == 1
        assert result.items[0]["file"] == "same.py"
        # Should keep highest scored: 0.9 * 0.8 = 0.72 > 0.5 * 0.6 = 0.30
        assert result.items[0]["_score"] == pytest.approx(0.72, rel=0.01)

    def test_given_results_when_merged_then_sorts_by_score(self):
        """
        Given: Results with different scores
        When: Aggregated with merge strategy
        Then: Items are sorted by score descending
        """
        result = ExecutionResult(
            items=[
                {"id": "low", "score": 0.2},
                {"id": "high", "score": 0.9},
                {"id": "mid", "score": 0.5},
            ],
            confidence=0.8,
            source="test",
        )

        aggregator = ResultAggregator(strategy="merge")
        aggregated = aggregator.aggregate([result])

        # Check order
        assert aggregated.items[0]["id"] == "high"
        assert aggregated.items[1]["id"] == "mid"
        assert aggregated.items[2]["id"] == "low"

    def test_given_low_confidence_results_when_merged_then_filtered(self):
        """
        Given: Results below min_confidence threshold
        When: Aggregated
        Then: Low-confidence results are filtered out
        """
        high_conf = ExecutionResult(
            items=[{"id": "1"}],
            confidence=0.8,
            source="audit",
        )
        low_conf = ExecutionResult(
            items=[{"id": "2"}],
            confidence=0.1,
            source="semantic",
        )

        aggregator = ResultAggregator(strategy="merge", min_confidence=0.2)
        result = aggregator.aggregate([high_conf, low_conf])

        # Only high confidence result should be included
        assert len(result.sources) == 1
        assert "audit" in result.sources


# =============================================================================
# ResultAggregator Best Strategy Tests
# =============================================================================


class TestBestStrategy:
    """Tests for the 'best' aggregation strategy."""

    def test_given_multiple_results_when_best_then_takes_highest_confidence(self):
        """
        Given: Multiple results with different confidence
        When: Aggregated with best strategy
        Then: Only takes results from highest-confidence source
        """
        low_conf = ExecutionResult(
            items=[{"id": "1"}, {"id": "2"}],
            confidence=0.5,
            source="semantic",
        )
        high_conf = ExecutionResult(
            items=[{"id": "3"}],
            confidence=0.9,
            source="audit",
        )

        aggregator = ResultAggregator(strategy="best")
        result = aggregator.aggregate([low_conf, high_conf])

        assert len(result.items) == 1
        assert result.items[0]["id"] == "3"
        assert result.sources == ["audit"]
        assert result.total_confidence == 0.9

    def test_given_single_result_when_best_then_returns_that_result(self):
        """
        Given: A single result
        When: Aggregated with best strategy
        Then: Returns that result's items
        """
        single = ExecutionResult(
            items=[{"id": "1"}],
            confidence=0.7,
            source="code",
        )

        aggregator = ResultAggregator(strategy="best")
        result = aggregator.aggregate([single])

        assert len(result.items) == 1
        assert result.sources == ["code"]


# =============================================================================
# ResultAggregator Weighted Strategy Tests
# =============================================================================


class TestWeightedStrategy:
    """Tests for the 'weighted' aggregation strategy."""

    def test_given_results_when_weighted_then_uses_confidence_squared(self):
        """
        Given: Results with different confidence levels
        When: Aggregated with weighted strategy
        Then: High-confidence results are weighted more heavily
        """
        high_conf = ExecutionResult(
            items=[{"id": "high", "score": 1.0}],
            confidence=0.9,
            source="audit",
        )
        low_conf = ExecutionResult(
            items=[{"id": "low", "score": 1.0}],
            confidence=0.3,
            source="semantic",
        )

        aggregator = ResultAggregator(strategy="weighted")
        result = aggregator.aggregate([high_conf, low_conf])

        # High confidence item should rank higher due to confidence^2 weighting
        assert result.items[0]["id"] == "high"

    def test_given_weighted_results_when_aggregated_then_confidence_weighted_avg(self):
        """
        Given: Multiple results
        When: Aggregated with weighted strategy
        Then: Total confidence is weighted average
        """
        r1 = ExecutionResult(items=[{"id": "1"}], confidence=0.8, source="a")
        r2 = ExecutionResult(items=[{"id": "2"}], confidence=0.4, source="b")

        aggregator = ResultAggregator(strategy="weighted")
        result = aggregator.aggregate([r1, r2])

        # Weighted average: (0.8^2 + 0.4^2) / (0.8 + 0.4) = (0.64 + 0.16) / 1.2 ≈ 0.667
        expected = (0.8**2 + 0.4**2) / (0.8 + 0.4)
        assert result.total_confidence == pytest.approx(expected, rel=0.01)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestAggregateResultsFunction:
    """Tests for the aggregate_results convenience function."""

    def test_aggregate_results_uses_default_merge(self):
        """
        Given: Results passed to aggregate_results()
        When: No strategy specified
        Then: Uses merge strategy by default
        """
        results = [
            ExecutionResult(items=[{"id": "1"}], confidence=0.8, source="a"),
            ExecutionResult(items=[{"id": "2"}], confidence=0.7, source="b"),
        ]

        aggregated = aggregate_results(results)

        assert len(aggregated.items) == 2
        assert len(aggregated.sources) == 2

    def test_aggregate_results_respects_max_items(self):
        """
        Given: Many items
        When: max_items is specified
        Then: Limits output to max_items
        """
        many_items = [{"id": str(i), "score": 1.0 / (i + 1)} for i in range(50)]
        results = [
            ExecutionResult(items=many_items, confidence=0.8, source="test"),
        ]

        aggregated = aggregate_results(results, max_items=5)

        assert len(aggregated.items) == 5


# =============================================================================
# Deduplication Key Extraction Tests
# =============================================================================


class TestKeyExtraction:
    """Tests for source-specific key extraction."""

    def test_audit_key_extracts_file(self):
        """Audit items use 'file' as key."""
        aggregator = ResultAggregator()
        item = {"file": "test.py", "score": 0.5}
        key = aggregator._get_item_key(item, "audit")
        assert key == "test.py"

    def test_semantic_key_extracts_doc_id(self):
        """Semantic items use 'doc_id' as key."""
        aggregator = ResultAggregator()
        item = {"doc_id": "readme.md", "score": 3.0}
        key = aggregator._get_item_key(item, "semantic")
        assert key == "readme.md"

    def test_code_key_extracts_file_and_name(self):
        """Code items use 'file_path:name' as key."""
        aggregator = ResultAggregator()
        item = {"name": "my_func", "file_path": "module.py"}
        key = aggregator._get_item_key(item, "code")
        assert key == "module.py:my_func"

    def test_cdg_key_extracts_id(self):
        """CDG items use 'id' as key."""
        aggregator = ResultAggregator()
        item = {"id": "T-001", "status": "pending"}
        key = aggregator._get_item_key(item, "cdg")
        assert key == "T-001"

    def test_unknown_source_uses_fallback(self):
        """Unknown sources use fallback key extraction."""
        aggregator = ResultAggregator()
        item = {"id": "fallback-id", "data": "test"}
        key = aggregator._get_item_key(item, "unknown_source")
        assert key == "fallback-id"


# =============================================================================
# Item Normalization Tests
# =============================================================================


class TestItemNormalization:
    """Tests for item normalization."""

    def test_dict_items_returned_as_is(self):
        """Dictionary items are returned unchanged."""
        aggregator = ResultAggregator()
        item = {"key": "value", "score": 1.0}
        normalized = aggregator._normalize_item(item, "test")
        assert normalized == item

    def test_object_items_converted_to_dict(self):
        """Objects with attributes are converted to dicts."""
        aggregator = ResultAggregator()

        class MockItem:
            name = "test_name"
            file_path = "test.py"

        item = MockItem()
        normalized = aggregator._normalize_item(item, "test")
        assert normalized["name"] == "test_name"
        assert normalized["file_path"] == "test.py"


# =============================================================================
# Explanation Building Tests
# =============================================================================


class TestExplanationBuilding:
    """Tests for combined explanation generation."""

    def test_explanation_includes_item_count(self):
        """Explanation mentions number of results."""
        results = [
            ExecutionResult(
                items=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
                confidence=0.8,
                source="test",
            )
        ]

        aggregated = aggregate_results(results)

        assert "3 results" in aggregated.explanation

    def test_explanation_includes_source_count(self):
        """Explanation mentions number of sources."""
        results = [
            ExecutionResult(items=[{"id": "1"}], confidence=0.8, source="audit"),
            ExecutionResult(items=[{"id": "2"}], confidence=0.7, source="semantic"),
        ]

        aggregated = aggregate_results(results)

        assert "2 source" in aggregated.explanation

    def test_empty_results_explanation(self):
        """Empty results produce helpful explanation."""
        results = [
            ExecutionResult(items=[], confidence=0.1, source="test"),
        ]

        aggregated = aggregate_results(results, min_confidence=0.5)

        assert "No results" in aggregated.explanation or "threshold" in aggregated.explanation


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestStrategyValidation:
    """Tests for strategy validation."""

    def test_invalid_strategy_raises_value_error(self):
        """Invalid strategy raises ValueError with helpful message."""
        import pytest
        with pytest.raises(ValueError) as exc_info:
            ResultAggregator(strategy="invalid_strategy")
        assert "invalid_strategy" in str(exc_info.value).lower()
        assert "merge" in str(exc_info.value)  # Shows valid options

    def test_valid_strategies_accepted(self):
        """All valid strategies are accepted."""
        for strategy in ["merge", "best", "weighted"]:
            aggregator = ResultAggregator(strategy=strategy)
            assert aggregator.strategy == strategy


class TestDeduplicationFixes:
    """Tests for deduplication behavior fixes."""

    def test_deduplication_keeps_highest_scored_item(self):
        """When duplicates exist, highest scored is kept."""
        # Create results with same item at different scores
        result1 = ExecutionResult(
            items=[{"file": "same.py", "score": 0.3}],
            confidence=0.5,
            source="audit",
        )
        result2 = ExecutionResult(
            items=[{"file": "same.py", "score": 0.9}],
            confidence=0.9,
            source="audit",
        )

        aggregator = ResultAggregator(strategy="merge")
        aggregated = aggregator.aggregate([result1, result2])

        # Should have only one item with higher score
        assert len(aggregated.items) == 1
        # The item should be from the higher confidence/score source: 0.9 * 0.9 = 0.81
        # NOT from lower: 0.3 * 0.5 = 0.15
        assert aggregated.items[0]["_score"] == pytest.approx(0.81, rel=0.01)

    def test_cross_source_deduplication_by_file_path(self):
        """Same file from different sources is deduplicated."""
        # Audit uses 'file', code uses 'file_path'
        audit_result = ExecutionResult(
            items=[{"file": "module.py", "score": 0.5}],
            confidence=0.8,
            source="audit",
        )
        code_result = ExecutionResult(
            items=[{"file_path": "module.py", "name": "func", "score": 0.3}],
            confidence=0.6,
            source="code",
        )

        aggregator = ResultAggregator(strategy="merge")
        aggregated = aggregator.aggregate([audit_result, code_result])

        # Should deduplicate based on file path
        # Higher scored item should be kept: audit 0.5*0.8=0.40 > code 0.3*0.6=0.18
        assert len(aggregated.items) == 1
        assert aggregated.items[0]["_source"] == "audit"
        assert aggregated.items[0]["_score"] == pytest.approx(0.40, rel=0.01)

    def test_best_strategy_includes_score_field(self):
        """Best strategy items include _score for consistency."""
        result = ExecutionResult(
            items=[{"id": "1", "score": 0.8}],
            confidence=0.9,
            source="audit",
        )

        aggregator = ResultAggregator(strategy="best")
        aggregated = aggregator.aggregate([result])

        assert len(aggregated.items) == 1
        assert "_score" in aggregated.items[0]
        assert "_source" in aggregated.items[0]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input_returns_empty_result(self):
        """Empty input list returns empty aggregated result."""
        aggregator = ResultAggregator()
        result = aggregator.aggregate([])
        assert result.is_empty
        assert "threshold" in result.explanation.lower()

    def test_all_filtered_returns_helpful_message(self):
        """When all results filtered by confidence, returns helpful message."""
        low_conf = ExecutionResult(
            items=[{"id": "1"}],
            confidence=0.1,
            source="test",
        )

        aggregator = ResultAggregator(min_confidence=0.5)
        result = aggregator.aggregate([low_conf])

        assert result.is_empty
        assert "threshold" in result.explanation.lower()

    def test_items_without_keys_are_included(self):
        """Items that can't be keyed are still included."""
        results = [
            ExecutionResult(
                items=[{"random_field": "value"}],  # No standard key field
                confidence=0.8,
                source="test",
            )
        ]

        aggregated = aggregate_results(results)

        assert len(aggregated.items) == 1

    def test_max_items_zero_returns_empty(self):
        """max_items=0 returns no items."""
        results = [
            ExecutionResult(items=[{"id": "1"}], confidence=0.8, source="test"),
        ]

        aggregated = aggregate_results(results, max_items=0)

        assert len(aggregated.items) == 0


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipelineIntegration:
    """Integration tests for the complete query pipeline."""

    def test_router_to_executor_to_aggregator_audit_query(self):
        """
        Given: An audit query
        When: Routed, executed, and aggregated
        Then: Returns properly aggregated result
        """
        from cortical.cognitive.unified_query import QueryRouter
        from cortical.cognitive.executors import AuditExecutor

        # Route the query
        router = QueryRouter()
        unified = router.route("risky files in cortical/")

        assert unified.query_type == "audit"

        # Execute
        executor = AuditExecutor()
        exec_result = executor.execute(unified.parsed)

        # Aggregate (single source)
        aggregated = aggregate_results([exec_result])

        # Verify aggregation worked correctly
        assert aggregated.sources == ["audit"]  # Exactly one source
        assert exec_result.confidence == aggregated.total_confidence  # Single source = same confidence
        assert aggregated.source_results.get("audit") is exec_result  # Original result preserved

    def test_router_to_executor_to_aggregator_semantic_query(self):
        """
        Given: A semantic query
        When: Routed, executed, and aggregated
        Then: Returns properly aggregated result
        """
        from cortical.cognitive.unified_query import QueryRouter
        from cortical.cognitive.executors import SemanticExecutor

        # Route the query
        router = QueryRouter()
        unified = router.route("what is the cognitive agent")

        assert unified.query_type == "semantic"

        # Execute
        executor = SemanticExecutor()
        exec_result = executor.execute(unified.parsed)

        # Aggregate
        aggregated = aggregate_results([exec_result])

        # Verify aggregation worked correctly
        assert aggregated.sources == ["semantic"]  # Exactly one source
        assert exec_result.confidence == aggregated.total_confidence
        assert aggregated.source_results.get("semantic") is exec_result

    def test_multi_executor_aggregation(self):
        """
        Given: A query that could match multiple backends
        When: Executed by multiple executors and aggregated
        Then: Results from all sources are combined
        """
        from cortical.cognitive.executors import (
            AuditExecutor,
            SemanticExecutor,
            CodeExecutor,
        )
        from cortical.cognitive.unified_query import QueryIntent
        from cortical.audits.reasoning import AuditQuery

        # Create queries for each executor type
        audit_query = AuditQuery(intent="list")
        semantic_query = QueryIntent(
            question_type="what",
            concepts=["test"],
            raw_question="test query"
        )
        code_query = {"action": "call", "subject": "test_func"}

        # Execute all
        audit_result = AuditExecutor().execute(audit_query)
        semantic_result = SemanticExecutor().execute(semantic_query)
        code_result = CodeExecutor().execute(code_query)

        # Aggregate all results
        aggregated = aggregate_results(
            [audit_result, semantic_result, code_result],
            strategy="merge"
        )

        # All three sources should be represented (even if some have no items)
        # Sources are only included if they have confidence >= min_confidence (0.2)
        valid_results = [r for r in [audit_result, semantic_result, code_result]
                        if r.confidence >= 0.2]
        assert len(aggregated.sources) == len(valid_results)

        # Confidence should be average of included sources
        if valid_results:
            expected_conf = sum(r.confidence for r in valid_results) / len(valid_results)
            assert aggregated.total_confidence == pytest.approx(expected_conf, rel=0.01)

        # All source results should be preserved
        for source in aggregated.sources:
            assert source in aggregated.source_results

    def test_aggregation_strategies_produce_different_results(self):
        """
        Given: Results with different confidence levels
        When: Aggregated with different strategies
        Then: Results differ based on strategy
        """
        high_conf = ExecutionResult(
            items=[{"id": "high", "score": 1.0}],
            confidence=0.9,
            source="audit",
        )
        low_conf = ExecutionResult(
            items=[{"id": "low", "score": 1.0}],
            confidence=0.4,
            source="semantic",
        )

        # Merge: includes both
        merged = aggregate_results([high_conf, low_conf], strategy="merge")

        # Best: only highest confidence
        best = aggregate_results([high_conf, low_conf], strategy="best")

        # Weighted: includes both but ranked differently
        weighted = aggregate_results([high_conf, low_conf], strategy="weighted")

        # Merge should have both sources
        assert len(merged.sources) == 2

        # Best should only have audit
        assert best.sources == ["audit"]
        assert len(best.items) == 1

        # Weighted should have both but high confidence first
        assert len(weighted.sources) == 2
        assert weighted.items[0]["id"] == "high"
