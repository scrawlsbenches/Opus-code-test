"""
Tests for the result dataclasses.

Ensures DocumentMatch, PassageMatch, and QueryResult work correctly.
"""

import unittest

from cortical.results import (
    DocumentMatch,
    PassageMatch,
    QueryResult,
    convert_document_matches,
    convert_passage_matches,
)


class TestDocumentMatch(unittest.TestCase):
    """Test DocumentMatch dataclass."""

    def test_creation_minimal(self):
        """Test creating with required fields only."""
        match = DocumentMatch(doc_id="doc1", score=0.95)
        self.assertEqual(match.doc_id, "doc1")
        self.assertEqual(match.score, 0.95)
        self.assertIsNone(match.metadata)

    def test_creation_with_metadata(self):
        """Test creating with metadata."""
        match = DocumentMatch(
            doc_id="doc1",
            score=0.95,
            metadata={"author": "test"}
        )
        self.assertEqual(match.metadata["author"], "test")

    def test_immutable(self):
        """Test that dataclass is frozen."""
        match = DocumentMatch(doc_id="doc1", score=0.95)
        with self.assertRaises(AttributeError):
            match.score = 0.5

    def test_repr(self):
        """Test string representation."""
        match = DocumentMatch(doc_id="doc1", score=0.95)
        repr_str = repr(match)
        self.assertIn("doc1", repr_str)
        self.assertIn("0.95", repr_str)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        match = DocumentMatch(doc_id="doc1", score=0.95)
        d = match.to_dict()
        self.assertEqual(d["doc_id"], "doc1")
        self.assertEqual(d["score"], 0.95)

    def test_to_tuple(self):
        """Test conversion to tuple."""
        match = DocumentMatch(doc_id="doc1", score=0.95)
        t = match.to_tuple()
        self.assertEqual(t, ("doc1", 0.95))

    def test_from_tuple(self):
        """Test creation from tuple arguments."""
        match = DocumentMatch.from_tuple("doc1", 0.95)
        self.assertEqual(match.doc_id, "doc1")
        self.assertEqual(match.score, 0.95)

    def test_from_tuple_with_metadata(self):
        """Test creation from tuple with metadata."""
        match = DocumentMatch.from_tuple("doc1", 0.95, {"key": "value"})
        self.assertEqual(match.metadata["key"], "value")

    def test_from_dict(self):
        """Test creation from dictionary."""
        match = DocumentMatch.from_dict({"doc_id": "doc1", "score": 0.95})
        self.assertEqual(match.doc_id, "doc1")
        self.assertEqual(match.score, 0.95)

    def test_roundtrip_dict(self):
        """Test dict roundtrip preserves data."""
        original = DocumentMatch(doc_id="doc1", score=0.95, metadata={"key": "value"})
        restored = DocumentMatch.from_dict(original.to_dict())
        self.assertEqual(original.doc_id, restored.doc_id)
        self.assertEqual(original.score, restored.score)
        self.assertEqual(original.metadata, restored.metadata)


class TestPassageMatch(unittest.TestCase):
    """Test PassageMatch dataclass."""

    def test_creation_minimal(self):
        """Test creating with required fields."""
        match = PassageMatch(
            doc_id="doc1",
            text="Sample text",
            score=0.85,
            start=0,
            end=11
        )
        self.assertEqual(match.doc_id, "doc1")
        self.assertEqual(match.text, "Sample text")
        self.assertEqual(match.score, 0.85)
        self.assertEqual(match.start, 0)
        self.assertEqual(match.end, 11)

    def test_creation_with_metadata(self):
        """Test creating with metadata."""
        match = PassageMatch(
            doc_id="doc1",
            text="Sample text",
            score=0.85,
            start=0,
            end=11,
            metadata={"highlight": True}
        )
        self.assertTrue(match.metadata["highlight"])

    def test_immutable(self):
        """Test that dataclass is frozen."""
        match = PassageMatch(doc_id="doc1", text="Text", score=0.5, start=0, end=4)
        with self.assertRaises(AttributeError):
            match.text = "Changed"

    def test_location_property(self):
        """Test location property for citations."""
        match = PassageMatch(
            doc_id="file.py",
            text="Sample",
            score=0.5,
            start=100,
            end=150
        )
        self.assertEqual(match.location, "file.py:100-150")

    def test_length_property(self):
        """Test length property."""
        match = PassageMatch(doc_id="doc1", text="Test", score=0.5, start=10, end=60)
        self.assertEqual(match.length, 50)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        match = PassageMatch(doc_id="doc1", text="Text", score=0.5, start=0, end=4)
        d = match.to_dict()
        self.assertEqual(d["doc_id"], "doc1")
        self.assertEqual(d["text"], "Text")
        self.assertEqual(d["score"], 0.5)
        self.assertEqual(d["start"], 0)
        self.assertEqual(d["end"], 4)

    def test_to_tuple(self):
        """Test conversion to tuple (doc_id, text, start, end, score)."""
        match = PassageMatch(doc_id="doc1", text="Text", score=0.5, start=0, end=4)
        t = match.to_tuple()
        # Order is: doc_id, text, start, end, score
        self.assertEqual(t, ("doc1", "Text", 0, 4, 0.5))

    def test_from_tuple(self):
        """Test creation from tuple arguments."""
        # Order is: doc_id, text, start, end, score
        match = PassageMatch.from_tuple("doc1", "Text", 0, 4, 0.5)
        self.assertEqual(match.doc_id, "doc1")
        self.assertEqual(match.text, "Text")
        self.assertEqual(match.start, 0)
        self.assertEqual(match.end, 4)
        self.assertEqual(match.score, 0.5)

    def test_from_dict(self):
        """Test creation from dictionary."""
        match = PassageMatch.from_dict({
            "doc_id": "doc1",
            "text": "Text",
            "score": 0.5,
            "start": 0,
            "end": 4
        })
        self.assertEqual(match.doc_id, "doc1")

    def test_repr_truncates_long_text(self):
        """Test repr truncates long text."""
        long_text = "A" * 100
        match = PassageMatch(doc_id="doc1", text=long_text, score=0.5, start=0, end=100)
        repr_str = repr(match)
        self.assertLess(len(repr_str), len(long_text) + 100)


class TestQueryResult(unittest.TestCase):
    """Test QueryResult wrapper."""

    def test_creation_with_document_matches(self):
        """Test creating with document matches."""
        matches = [
            DocumentMatch(doc_id="doc1", score=0.9),
            DocumentMatch(doc_id="doc2", score=0.7)
        ]
        result = QueryResult(query="test", matches=matches)
        self.assertEqual(result.query, "test")
        self.assertEqual(len(result.matches), 2)

    def test_creation_with_passage_matches(self):
        """Test creating with passage matches."""
        matches = [
            PassageMatch(doc_id="doc1", text="Text", score=0.9, start=0, end=4)
        ]
        result = QueryResult(query="test", matches=matches)
        self.assertEqual(len(result.matches), 1)

    def test_creation_with_all_fields(self):
        """Test creating with all optional fields."""
        result = QueryResult(
            query="test",
            matches=[DocumentMatch(doc_id="doc1", score=0.9)],
            expansion_terms={"test": 1.0, "testing": 0.8},
            timing_ms=15.5,
            metadata={"source": "api"}
        )
        self.assertEqual(result.expansion_terms["testing"], 0.8)
        self.assertEqual(result.timing_ms, 15.5)

    def test_top_match_property(self):
        """Test top_match returns highest scoring."""
        matches = [
            DocumentMatch(doc_id="doc1", score=0.5),
            DocumentMatch(doc_id="doc2", score=0.9),
            DocumentMatch(doc_id="doc3", score=0.7)
        ]
        result = QueryResult(query="test", matches=matches)
        self.assertEqual(result.top_match.doc_id, "doc2")

    def test_top_match_empty(self):
        """Test top_match with no matches."""
        result = QueryResult(query="test", matches=[])
        self.assertIsNone(result.top_match)

    def test_match_count_property(self):
        """Test match_count property."""
        matches = [
            DocumentMatch(doc_id="doc1", score=0.5),
            DocumentMatch(doc_id="doc2", score=0.9)
        ]
        result = QueryResult(query="test", matches=matches)
        self.assertEqual(result.match_count, 2)

    def test_average_score_property(self):
        """Test average_score calculation."""
        matches = [
            DocumentMatch(doc_id="doc1", score=0.4),
            DocumentMatch(doc_id="doc2", score=0.6)
        ]
        result = QueryResult(query="test", matches=matches)
        self.assertEqual(result.average_score, 0.5)

    def test_average_score_empty(self):
        """Test average_score with no matches."""
        result = QueryResult(query="test", matches=[])
        self.assertEqual(result.average_score, 0.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QueryResult(
            query="test",
            matches=[DocumentMatch(doc_id="doc1", score=0.9)]
        )
        d = result.to_dict()
        self.assertEqual(d["query"], "test")
        self.assertEqual(len(d["matches"]), 1)

    def test_from_dict_document_matches(self):
        """Test creation from dict with document matches."""
        d = {
            "query": "test",
            "matches": [{"doc_id": "doc1", "score": 0.9, "metadata": None}]
        }
        result = QueryResult.from_dict(d)
        self.assertEqual(result.query, "test")
        self.assertIsInstance(result.matches[0], DocumentMatch)

    def test_from_dict_passage_matches(self):
        """Test creation from dict with passage matches."""
        d = {
            "query": "test",
            "matches": [{"doc_id": "doc1", "text": "Text", "score": 0.9, "start": 0, "end": 4, "metadata": None}]
        }
        result = QueryResult.from_dict(d)
        self.assertIsInstance(result.matches[0], PassageMatch)


class TestHelperFunctions(unittest.TestCase):
    """Test conversion helper functions."""

    def test_convert_document_matches_basic(self):
        """Test converting list of tuples."""
        raw = [("doc1", 0.9), ("doc2", 0.7)]
        matches = convert_document_matches(raw)
        self.assertEqual(len(matches), 2)
        self.assertIsInstance(matches[0], DocumentMatch)
        self.assertEqual(matches[0].doc_id, "doc1")

    def test_convert_document_matches_empty(self):
        """Test converting empty list."""
        matches = convert_document_matches([])
        self.assertEqual(matches, [])

    def test_convert_document_matches_with_metadata(self):
        """Test converting with metadata dict."""
        raw = [("doc1", 0.9), ("doc2", 0.7)]
        metadata = {"doc1": {"author": "test"}}
        matches = convert_document_matches(raw, metadata)
        self.assertEqual(matches[0].metadata["author"], "test")
        self.assertIsNone(matches[1].metadata)

    def test_convert_passage_matches_basic(self):
        """Test converting passage tuples (doc_id, text, start, end, score)."""
        raw = [("doc1", "Sample", 0, 6, 0.9)]
        matches = convert_passage_matches(raw)
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], PassageMatch)
        self.assertEqual(matches[0].text, "Sample")

    def test_convert_passage_matches_empty(self):
        """Test converting empty list."""
        matches = convert_passage_matches([])
        self.assertEqual(matches, [])


class TestIntegration(unittest.TestCase):
    """Test integration with CorticalTextProcessor."""

    def test_document_search_workflow(self):
        """Test converting actual search results."""
        from cortical import CorticalTextProcessor

        proc = CorticalTextProcessor()
        proc.process_document("doc1", "Neural networks process data")
        proc.process_document("doc2", "Machine learning is powerful")
        proc.compute_all()

        raw_results = proc.find_documents_for_query("neural", top_n=2)
        matches = convert_document_matches(raw_results)

        self.assertGreater(len(matches), 0)
        self.assertIsInstance(matches[0], DocumentMatch)
        self.assertIsInstance(matches[0].score, float)


if __name__ == '__main__':
    unittest.main()
