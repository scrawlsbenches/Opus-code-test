"""
Tests for scripts/ask_codebase.py - CodebaseQA class and utilities.
"""

import unittest
import sys
from pathlib import Path

# Add parent and scripts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from cortical.processor import CorticalTextProcessor
from ask_codebase import (
    CodebaseQA,
    find_line_number,
    format_reference,
    get_doc_type_emoji
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

    def test_format_reference(self):
        """Test reference formatting."""
        ref = format_reference("cortical/processor.py", 42)
        self.assertEqual(ref, "cortical/processor.py:42")

    def test_get_doc_type_emoji_markdown(self):
        """Test emoji for markdown files."""
        self.assertEqual(get_doc_type_emoji("README.md"), "📖")
        self.assertEqual(get_doc_type_emoji("docs/guide.md"), "📖")

    def test_get_doc_type_emoji_test(self):
        """Test emoji for test files."""
        self.assertEqual(get_doc_type_emoji("tests/test_processor.py"), "🧪")
        self.assertEqual(get_doc_type_emoji("tests/test_analysis.py"), "🧪")

    def test_get_doc_type_emoji_code(self):
        """Test emoji for code files."""
        self.assertEqual(get_doc_type_emoji("cortical/processor.py"), "💻")
        self.assertEqual(get_doc_type_emoji("scripts/index.py"), "💻")


class TestCodebaseQA(unittest.TestCase):
    """Tests for the CodebaseQA class."""

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
            "README.md",
            """
            # Cortical Text Processor

            A hierarchical text analysis library inspired by visual cortex.

            ## Features
            - PageRank for centrality
            - TF-IDF for relevance
            - Query expansion
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
        cls.qa = CodebaseQA(cls.processor)

    def test_find_relevant_passages(self):
        """Test finding relevant passages."""
        passages = self.qa.find_relevant_passages("PageRank algorithm", top_n=2)

        self.assertIsInstance(passages, list)
        self.assertGreater(len(passages), 0)

        # Each passage should be (text, reference, line_num, score)
        passage = passages[0]
        self.assertEqual(len(passage), 4)
        self.assertIsInstance(passage[0], str)  # text
        self.assertIsInstance(passage[1], str)  # reference
        self.assertIsInstance(passage[2], int)  # line_num
        self.assertIsInstance(passage[3], float)  # score

    def test_find_relevant_passages_empty_query(self):
        """Test with empty query."""
        passages = self.qa.find_relevant_passages("", top_n=2)
        self.assertIsInstance(passages, list)

    def test_format_answer_with_passages(self):
        """Test answer formatting with passages."""
        passages = self.qa.find_relevant_passages("PageRank", top_n=2)
        answer = self.qa.format_answer("What is PageRank?", passages)

        self.assertIsInstance(answer, str)
        self.assertIn("PageRank", answer)
        self.assertIn("Relevant Context", answer)

    def test_format_answer_empty_passages(self):
        """Test answer formatting with no passages."""
        answer = self.qa.format_answer("What is XYZ123?", [])

        self.assertIn("No relevant passages found", answer)

    def test_format_answer_shows_sources(self):
        """Test that sources are shown."""
        passages = self.qa.find_relevant_passages("PageRank", top_n=2)
        answer = self.qa.format_answer("What is PageRank?", passages, show_sources=True)

        self.assertIn("Sources", answer)

    def test_format_answer_hides_sources(self):
        """Test that sources can be hidden."""
        passages = self.qa.find_relevant_passages("PageRank", top_n=2)
        answer = self.qa.format_answer("What is PageRank?", passages, show_sources=False)

        self.assertNotIn("Sources:", answer)

    def test_answer_method(self):
        """Test the main answer method."""
        answer = self.qa.answer("How does query expansion work?")

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_answer_detects_conceptual_query(self):
        """Test that conceptual queries are detected."""
        answer = self.qa.answer("What is PageRank?")
        self.assertIn("conceptual", answer.lower())

    def test_answer_detects_implementation_query(self):
        """Test that implementation queries are detected."""
        answer = self.qa.answer("compute pagerank damping factor")
        self.assertIn("implementation", answer.lower())


class TestCodebaseQAEmpty(unittest.TestCase):
    """Tests for CodebaseQA with empty processor."""

    def test_empty_processor(self):
        """Test with empty processor."""
        processor = CorticalTextProcessor()
        qa = CodebaseQA(processor)

        passages = qa.find_relevant_passages("anything", top_n=2)
        self.assertEqual(passages, [])

    def test_empty_processor_answer(self):
        """Test answer with empty processor."""
        processor = CorticalTextProcessor()
        qa = CodebaseQA(processor)

        answer = qa.answer("How does X work?")
        self.assertIn("No relevant passages", answer)


if __name__ == '__main__':
    unittest.main()
