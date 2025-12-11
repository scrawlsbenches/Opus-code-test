"""
Tests for scripts/search_codebase.py - search functions and utilities.
"""

import unittest
import sys
from pathlib import Path

# Add parent and scripts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from cortical.processor import CorticalTextProcessor
from search_codebase import (
    find_line_number,
    format_passage,
    get_doc_type_label,
    search_codebase,
    find_similar_code
)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_find_line_number_start(self):
        """Test line number at start of document."""
        content = "line1\nline2\nline3"
        self.assertEqual(find_line_number(content, 0), 1)

    def test_find_line_number_second_line(self):
        """Test line number for second line."""
        content = "line1\nline2\nline3"
        self.assertEqual(find_line_number(content, 6), 2)

    def test_find_line_number_third_line(self):
        """Test line number for third line."""
        content = "line1\nline2\nline3"
        self.assertEqual(find_line_number(content, 12), 3)

    def test_format_passage_short(self):
        """Test formatting a short passage."""
        passage = "Line 1\nLine 2\nLine 3"
        result = format_passage(passage)
        self.assertIn("Line 1", result)
        self.assertIn("Line 2", result)

    def test_format_passage_truncates_long_lines(self):
        """Test that long lines are truncated."""
        long_line = "x" * 100
        passage = long_line
        result = format_passage(passage, max_width=50)
        self.assertIn("...", result)
        self.assertLessEqual(len(result), 50)

    def test_format_passage_limits_lines(self):
        """Test that many lines are limited."""
        passage = "\n".join([f"Line {i}" for i in range(20)])
        result = format_passage(passage)
        self.assertIn("more lines", result)

    def test_get_doc_type_label_docs_markdown(self):
        """Test label for docs/ markdown files."""
        self.assertEqual(get_doc_type_label("docs/guide.md"), "DOCS")

    def test_get_doc_type_label_markdown(self):
        """Test label for other markdown files."""
        self.assertEqual(get_doc_type_label("README.md"), "DOC")

    def test_get_doc_type_label_test(self):
        """Test label for test files."""
        self.assertEqual(get_doc_type_label("tests/test_processor.py"), "TEST")

    def test_get_doc_type_label_code(self):
        """Test label for code files."""
        self.assertEqual(get_doc_type_label("cortical/processor.py"), "CODE")


class TestSearchCodebase(unittest.TestCase):
    """Tests for the search_codebase function."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with test documents."""
        cls.processor = CorticalTextProcessor()

        # Add test documents
        cls.processor.process_document(
            "processor.py",
            """
            The CorticalTextProcessor is the main API for text analysis.
            It uses PageRank for term importance and TF-IDF for relevance.
            Query expansion adds related terms to improve recall.
            """
        )
        cls.processor.process_document(
            "docs/guide.md",
            """
            # User Guide

            This guide explains how PageRank works in the system.
            PageRank measures the importance of terms based on connections.
            """
        )
        cls.processor.process_document(
            "tests/test_processor.py",
            """
            import unittest

            class TestProcessor(unittest.TestCase):
                def test_pagerank(self):
                    processor = CorticalTextProcessor()
                    processor.compute_all()
            """
        )

        cls.processor.compute_all()

    def test_search_returns_results(self):
        """Test that search returns results."""
        results = search_codebase(self.processor, "PageRank", top_n=3)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_search_result_structure(self):
        """Test result dict structure."""
        results = search_codebase(self.processor, "PageRank", top_n=1)

        if results:
            result = results[0]
            self.assertIn('file', result)
            self.assertIn('line', result)
            self.assertIn('passage', result)
            self.assertIn('score', result)
            self.assertIn('reference', result)
            self.assertIn('doc_type', result)

    def test_search_fast_mode(self):
        """Test fast search mode."""
        results = search_codebase(
            self.processor, "PageRank", top_n=3, fast=True
        )

        self.assertIsInstance(results, list)
        # Fast mode always returns line 1
        for result in results:
            self.assertEqual(result['line'], 1)

    def test_search_no_boost_mode(self):
        """Test search with boosting disabled."""
        results = search_codebase(
            self.processor, "PageRank", top_n=3, no_boost=True
        )

        self.assertIsInstance(results, list)

    def test_search_prefer_docs(self):
        """Test search with prefer_docs flag."""
        results = search_codebase(
            self.processor, "PageRank", top_n=3, prefer_docs=True
        )

        self.assertIsInstance(results, list)

    def test_search_empty_query(self):
        """Test search with empty query."""
        results = search_codebase(self.processor, "", top_n=3)
        self.assertIsInstance(results, list)


class TestFindSimilarCode(unittest.TestCase):
    """Tests for the find_similar_code function."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with test documents."""
        cls.processor = CorticalTextProcessor()

        cls.processor.process_document(
            "module_a.py",
            """
            def calculate_score(items, weights):
                total = 0
                for item, weight in zip(items, weights):
                    total += item * weight
                return total / len(items) if items else 0
            """
        )
        cls.processor.process_document(
            "module_b.py",
            """
            def compute_weighted_average(values, factors):
                result = 0
                for value, factor in zip(values, factors):
                    result += value * factor
                return result / len(values) if values else 0
            """
        )
        cls.processor.process_document(
            "unrelated.py",
            """
            class UserAuthentication:
                def verify_password(self, password, hash):
                    return bcrypt.check(password, hash)
            """
        )

        cls.processor.compute_all()

    def test_find_similar_with_text(self):
        """Test finding similar code with raw text."""
        target_code = "def compute_score(data, weights): return sum()"
        results = find_similar_code(
            self.processor, target_code, top_n=3
        )

        self.assertIsInstance(results, list)

    def test_find_similar_result_structure(self):
        """Test that results have expected structure."""
        results = find_similar_code(
            self.processor, "def calculate total weighted", top_n=1
        )

        if results:
            result = results[0]
            self.assertIn('file', result)
            self.assertIn('line', result)
            self.assertIn('passage', result)
            self.assertIn('score', result)
            self.assertIn('reference', result)
            self.assertIn('doc_type', result)

    def test_find_similar_with_file_reference(self):
        """Test finding similar code with file:line reference."""
        results = find_similar_code(
            self.processor, "module_a.py:1", top_n=3
        )

        self.assertIsInstance(results, list)
        # Should not include the source file itself
        for result in results:
            self.assertNotIn('module_a.py', result['reference'])

    def test_find_similar_empty_text(self):
        """Test with empty target text."""
        results = find_similar_code(self.processor, "", top_n=3)
        self.assertEqual(results, [])

    def test_find_similar_nonexistent_file(self):
        """Test with nonexistent file reference."""
        results = find_similar_code(
            self.processor, "nonexistent.py:100", top_n=3
        )
        self.assertEqual(results, [])


class TestSearchCodebaseEmpty(unittest.TestCase):
    """Tests with empty processor."""

    def test_search_empty_processor(self):
        """Test search with empty processor."""
        processor = CorticalTextProcessor()
        results = search_codebase(processor, "anything", top_n=3)
        self.assertEqual(results, [])

    def test_find_similar_empty_processor(self):
        """Test find_similar with empty processor."""
        processor = CorticalTextProcessor()
        results = find_similar_code(processor, "some code", top_n=3)
        self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()
