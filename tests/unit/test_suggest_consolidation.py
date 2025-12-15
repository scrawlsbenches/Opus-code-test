"""
Unit tests for suggest_consolidation.py script.

Tests the memory consolidation suggestion functionality.
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from suggest_consolidation import (
    parse_memory_date,
    get_memory_age_days,
    is_concept_doc,
    suggest_consolidations,
    format_suggestions_text
)

from cortical.processor import CorticalTextProcessor


class TestMemoryDateParsing(unittest.TestCase):
    """Test memory date parsing functions."""

    def test_parse_memory_date_basic(self):
        """Test parsing basic date format YYYY-MM-DD."""
        doc_id = "samples/memories/2025-12-14-topic.md"
        date = parse_memory_date(doc_id)
        self.assertEqual(date.year, 2025)
        self.assertEqual(date.month, 12)
        self.assertEqual(date.day, 14)

    def test_parse_memory_date_with_timestamp(self):
        """Test parsing date with timestamp format."""
        doc_id = "samples/memories/2025-12-14_20-54-35_3b3a-topic.md"
        date = parse_memory_date(doc_id)
        self.assertEqual(date.year, 2025)
        self.assertEqual(date.month, 12)
        self.assertEqual(date.day, 14)

    def test_parse_memory_date_concept(self):
        """Test parsing concept document (should return old date)."""
        doc_id = "samples/memories/concept-something.md"
        date = parse_memory_date(doc_id)
        self.assertEqual(date.year, 2000)

    def test_parse_memory_date_invalid(self):
        """Test parsing invalid date format (should return default)."""
        doc_id = "samples/memories/invalid-format.md"
        date = parse_memory_date(doc_id)
        self.assertEqual(date.year, 2000)

    def test_get_memory_age_days(self):
        """Test calculating memory age in days."""
        # Use today's date for a new memory
        today = datetime.now()
        doc_id = f"samples/memories/{today.strftime('%Y-%m-%d')}-topic.md"
        age = get_memory_age_days(doc_id)
        self.assertEqual(age, 0)

        # Old memory
        old_date = today - timedelta(days=30)
        doc_id = f"samples/memories/{old_date.strftime('%Y-%m-%d')}-topic.md"
        age = get_memory_age_days(doc_id)
        self.assertGreaterEqual(age, 29)  # Allow for rounding

    def test_is_concept_doc(self):
        """Test concept document detection."""
        self.assertTrue(is_concept_doc("samples/memories/concept-foo.md"))
        self.assertFalse(is_concept_doc("samples/memories/2025-12-14-topic.md"))
        self.assertFalse(is_concept_doc("samples/decisions/adr-001.md"))


class TestSuggestConsolidation(unittest.TestCase):
    """Test consolidation suggestion functions."""

    def setUp(self):
        """Create a test processor with sample memories."""
        self.processor = CorticalTextProcessor()

        # Add some sample memory documents
        self.processor.process_document(
            "samples/memories/2025-12-01-security-testing.md",
            """
            # Security Testing

            We discovered bugs using fuzzing and hypothesis testing.
            The validation checks were failing for NaN values.
            """
        )

        self.processor.process_document(
            "samples/memories/2025-12-02-fuzzing-results.md",
            """
            # Fuzzing Results

            More fuzzing tests revealed edge cases with NaN and infinity.
            Security testing found validation bugs.
            """
        )

        self.processor.process_document(
            "samples/memories/2025-11-01-architecture-refactor.md",
            """
            # Architecture Refactor

            Refactored the processor into separate mixins.
            Improved code organization and maintainability.
            """
        )

        self.processor.process_document(
            "samples/memories/concept-testing.md",
            """
            # Concept: Testing

            Overview of testing approaches including fuzzing and unit tests.
            """
        )

        # Compute features
        self.processor.compute_all()

    def test_suggest_consolidations_basic(self):
        """Test basic consolidation suggestion."""
        suggestions = suggest_consolidations(
            self.processor,
            min_overlap=0.3,
            min_cluster_size=2,
            min_age_days=7,
            verbose=False
        )

        self.assertIn('clusters', suggestions)
        self.assertIn('similar_pairs', suggestions)
        self.assertIn('old_memories', suggestions)
        self.assertIn('stats', suggestions)

        # Check stats
        stats = suggestions['stats']
        self.assertEqual(stats['total_memories'], 3)
        self.assertEqual(stats['total_concepts'], 1)

    def test_suggest_consolidations_high_threshold(self):
        """Test with high similarity threshold."""
        suggestions = suggest_consolidations(
            self.processor,
            min_overlap=0.9,
            min_cluster_size=2,
            verbose=False
        )

        # With high threshold, should find fewer pairs
        self.assertIsInstance(suggestions['similar_pairs'], list)

    def test_suggest_consolidations_old_memories(self):
        """Test old memory detection."""
        suggestions = suggest_consolidations(
            self.processor,
            min_age_days=7,
            verbose=False
        )

        # Should detect the November memory as old
        old_memories = suggestions['old_memories']
        self.assertIsInstance(old_memories, list)

        # Check if we found the old memory
        old_doc_ids = [m['doc_id'] for m in old_memories]
        self.assertTrue(
            any('2025-11-01' in doc_id for doc_id in old_doc_ids),
            "Should detect November memory as old"
        )

    def test_format_suggestions_text(self):
        """Test text formatting of suggestions."""
        suggestions = suggest_consolidations(
            self.processor,
            min_overlap=0.3,
            verbose=False
        )

        text = format_suggestions_text(suggestions, verbose=False)

        # Check that output contains expected sections
        self.assertIn('MEMORY CONSOLIDATION SUGGESTIONS', text)
        self.assertIn('Analyzed', text)
        self.assertIn('RECOMMENDATIONS', text)

    def test_format_suggestions_text_verbose(self):
        """Test verbose text formatting."""
        suggestions = suggest_consolidations(
            self.processor,
            min_overlap=0.3,
            verbose=False
        )

        text = format_suggestions_text(suggestions, verbose=True)

        # Verbose output should include more details
        self.assertIn('MEMORY CONSOLIDATION SUGGESTIONS', text)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_corpus(self):
        """Test with empty corpus."""
        processor = CorticalTextProcessor()
        processor.compute_all()

        suggestions = suggest_consolidations(processor, verbose=False)

        self.assertEqual(suggestions['stats']['total_memories'], 0)
        self.assertEqual(len(suggestions['clusters']), 0)
        self.assertEqual(len(suggestions['similar_pairs']), 0)

    def test_single_memory(self):
        """Test with single memory document."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "samples/memories/2025-12-14-single.md",
            "Single memory entry."
        )
        processor.compute_all()

        suggestions = suggest_consolidations(
            processor,
            min_cluster_size=2,
            verbose=False
        )

        # Should not crash, but won't find clusters
        self.assertEqual(suggestions['stats']['total_memories'], 1)
        self.assertEqual(len(suggestions['clusters']), 0)

    def test_invalid_threshold(self):
        """Test that invalid thresholds are handled."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "samples/memories/2025-12-14-test.md",
            "Test content"
        )
        processor.compute_all()

        # Should still work, even with edge case thresholds
        suggestions = suggest_consolidations(
            processor,
            min_overlap=0.0,
            verbose=False
        )
        self.assertIsNotNone(suggestions)

        suggestions = suggest_consolidations(
            processor,
            min_overlap=1.0,
            verbose=False
        )
        self.assertIsNotNone(suggestions)


if __name__ == '__main__':
    unittest.main()
