"""
Unit Tests for Results Module
==============================

Task #185: Create result dataclasses for query results.

Tests the DocumentMatch, PassageMatch, and QueryResult dataclasses that
provide strongly-typed containers for search results with IDE support.

Coverage goal: 95%
Test count goal: 40+
"""

import pytest

from cortical.results import (
    DocumentMatch,
    PassageMatch,
    QueryResult,
    convert_document_matches,
    convert_passage_matches
)


# =============================================================================
# DOCUMENTMATCH CLASS TESTS
# =============================================================================


class TestDocumentMatchClass:
    """Tests for the DocumentMatch dataclass."""

    def test_creation_minimal(self):
        """DocumentMatch created with minimal parameters."""
        match = DocumentMatch("doc1", 0.85)
        assert match.doc_id == "doc1"
        assert match.score == 0.85
        assert match.metadata is None

    def test_creation_with_metadata(self):
        """DocumentMatch created with metadata."""
        metadata = {"doc_type": "markdown", "size": 1024}
        match = DocumentMatch("doc1.md", 0.92, metadata)
        assert match.doc_id == "doc1.md"
        assert match.score == 0.92
        assert match.metadata == metadata
        assert match.metadata["doc_type"] == "markdown"

    def test_immutable(self):
        """DocumentMatch is immutable (frozen)."""
        match = DocumentMatch("doc1", 0.8)
        with pytest.raises(AttributeError):
            match.score = 0.9

    def test_repr_without_metadata(self):
        """String representation without metadata."""
        match = DocumentMatch("doc1", 0.8521)
        repr_str = repr(match)
        assert "DocumentMatch" in repr_str
        assert "doc1" in repr_str
        assert "0.8521" in repr_str

    def test_repr_with_metadata(self):
        """String representation with metadata."""
        match = DocumentMatch("doc1", 0.8, {"type": "test"})
        repr_str = repr(match)
        assert "metadata=" in repr_str

    def test_to_dict(self):
        """Convert to dictionary."""
        match = DocumentMatch("doc1", 0.85)
        d = match.to_dict()
        assert d == {"doc_id": "doc1", "score": 0.85, "metadata": None}

    def test_to_dict_with_metadata(self):
        """Convert to dictionary with metadata."""
        metadata = {"key": "value"}
        match = DocumentMatch("doc1", 0.85, metadata)
        d = match.to_dict()
        assert d["metadata"] == metadata

    def test_to_tuple(self):
        """Convert to tuple format."""
        match = DocumentMatch("doc1", 0.85)
        t = match.to_tuple()
        assert t == ("doc1", 0.85)

    def test_from_tuple_minimal(self):
        """Create from tuple with minimal args."""
        match = DocumentMatch.from_tuple("doc1", 0.85)
        assert match.doc_id == "doc1"
        assert match.score == 0.85
        assert match.metadata is None

    def test_from_tuple_with_metadata(self):
        """Create from tuple with metadata."""
        metadata = {"type": "test"}
        match = DocumentMatch.from_tuple("doc1", 0.85, metadata)
        assert match.metadata == metadata

    def test_from_dict_minimal(self):
        """Create from dictionary with minimal fields."""
        data = {"doc_id": "doc1", "score": 0.85}
        match = DocumentMatch.from_dict(data)
        assert match.doc_id == "doc1"
        assert match.score == 0.85
        assert match.metadata is None

    def test_from_dict_with_metadata(self):
        """Create from dictionary with metadata."""
        data = {"doc_id": "doc1", "score": 0.85, "metadata": {"type": "test"}}
        match = DocumentMatch.from_dict(data)
        assert match.metadata == {"type": "test"}

    def test_roundtrip_dict(self):
        """Roundtrip through dictionary preserves data."""
        original = DocumentMatch("doc1", 0.85, {"key": "value"})
        d = original.to_dict()
        restored = DocumentMatch.from_dict(d)
        assert restored.doc_id == original.doc_id
        assert restored.score == original.score
        assert restored.metadata == original.metadata

    def test_roundtrip_tuple(self):
        """Roundtrip through tuple preserves data (without metadata)."""
        original = DocumentMatch("doc1", 0.85)
        t = original.to_tuple()
        restored = DocumentMatch.from_tuple(*t)
        assert restored.doc_id == original.doc_id
        assert restored.score == original.score


# =============================================================================
# PASSAGEMATCH CLASS TESTS
# =============================================================================


class TestPassageMatchClass:
    """Tests for the PassageMatch dataclass."""

    def test_creation_minimal(self):
        """PassageMatch created with minimal parameters."""
        match = PassageMatch("doc1.py", "def foo():\n    pass", 0.9, 100, 120)
        assert match.doc_id == "doc1.py"
        assert match.text == "def foo():\n    pass"
        assert match.score == 0.9
        assert match.start == 100
        assert match.end == 120
        assert match.metadata is None

    def test_creation_with_metadata(self):
        """PassageMatch created with metadata."""
        metadata = {"function": "foo", "line": 10}
        match = PassageMatch("doc1.py", "code here", 0.85, 0, 9, metadata)
        assert match.metadata == metadata

    def test_immutable(self):
        """PassageMatch is immutable (frozen)."""
        match = PassageMatch("doc1", "text", 0.8, 0, 4)
        with pytest.raises(AttributeError):
            match.score = 0.9

    def test_repr_truncates_long_text(self):
        """String representation truncates long text."""
        long_text = "a" * 100
        match = PassageMatch("doc1", long_text, 0.8, 0, 100)
        repr_str = repr(match)
        assert "..." in repr_str
        assert len(repr_str) < 200  # Should be shorter than full text

    def test_repr_escapes_newlines(self):
        """String representation escapes newlines."""
        match = PassageMatch("doc1", "line1\nline2", 0.8, 0, 11)
        repr_str = repr(match)
        assert "\\n" in repr_str

    def test_to_dict(self):
        """Convert to dictionary."""
        match = PassageMatch("doc1", "text", 0.85, 0, 4)
        d = match.to_dict()
        assert d["doc_id"] == "doc1"
        assert d["text"] == "text"
        assert d["score"] == 0.85
        assert d["start"] == 0
        assert d["end"] == 4
        assert d["metadata"] is None

    def test_to_tuple(self):
        """Convert to tuple format."""
        match = PassageMatch("doc1", "text", 0.85, 10, 14)
        t = match.to_tuple()
        assert t == ("doc1", "text", 10, 14, 0.85)

    def test_location_property(self):
        """Location property returns citation-style string."""
        match = PassageMatch("doc1.py", "text", 0.8, 100, 150)
        assert match.location == "doc1.py:100-150"

    def test_length_property(self):
        """Length property returns character count."""
        match = PassageMatch("doc1", "hello", 0.8, 0, 5)
        assert match.length == 5

    def test_length_property_larger_range(self):
        """Length property for larger range."""
        match = PassageMatch("doc1", "text", 0.8, 100, 250)
        assert match.length == 150

    def test_from_tuple_minimal(self):
        """Create from tuple with minimal args."""
        match = PassageMatch.from_tuple("doc1", "text", 0, 4, 0.9)
        assert match.doc_id == "doc1"
        assert match.text == "text"
        assert match.start == 0
        assert match.end == 4
        assert match.score == 0.9
        assert match.metadata is None

    def test_from_tuple_with_metadata(self):
        """Create from tuple with metadata."""
        metadata = {"line": 5}
        match = PassageMatch.from_tuple("doc1", "text", 0, 4, 0.9, metadata)
        assert match.metadata == metadata

    def test_from_dict_minimal(self):
        """Create from dictionary with minimal fields."""
        data = {
            "doc_id": "doc1",
            "text": "hello",
            "score": 0.8,
            "start": 0,
            "end": 5
        }
        match = PassageMatch.from_dict(data)
        assert match.text == "hello"
        assert match.length == 5

    def test_from_dict_with_metadata(self):
        """Create from dictionary with metadata."""
        data = {
            "doc_id": "doc1",
            "text": "hello",
            "score": 0.8,
            "start": 0,
            "end": 5,
            "metadata": {"type": "definition"}
        }
        match = PassageMatch.from_dict(data)
        assert match.metadata == {"type": "definition"}

    def test_roundtrip_dict(self):
        """Roundtrip through dictionary preserves data."""
        original = PassageMatch("doc1", "text", 0.8, 10, 14, {"key": "value"})
        d = original.to_dict()
        restored = PassageMatch.from_dict(d)
        assert restored.doc_id == original.doc_id
        assert restored.text == original.text
        assert restored.score == original.score
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.metadata == original.metadata

    def test_roundtrip_tuple(self):
        """Roundtrip through tuple preserves data (without metadata)."""
        original = PassageMatch("doc1", "text", 0.8, 10, 14)
        t = original.to_tuple()
        restored = PassageMatch.from_tuple(*t)
        assert restored.doc_id == original.doc_id
        assert restored.text == original.text
        assert restored.score == original.score


# =============================================================================
# QUERYRESULT CLASS TESTS
# =============================================================================


class TestQueryResultClass:
    """Tests for the QueryResult wrapper class."""

    def test_creation_with_document_matches(self):
        """QueryResult created with DocumentMatch list."""
        matches = [DocumentMatch("doc1", 0.9), DocumentMatch("doc2", 0.7)]
        result = QueryResult("neural networks", matches)
        assert result.query == "neural networks"
        assert len(result.matches) == 2
        assert result.expansion_terms is None
        assert result.timing_ms is None
        assert result.metadata is None

    def test_creation_with_passage_matches(self):
        """QueryResult created with PassageMatch list."""
        matches = [PassageMatch("doc1", "text1", 0.9, 0, 5)]
        result = QueryResult("test query", matches)
        assert len(result.matches) == 1
        assert isinstance(result.matches[0], PassageMatch)

    def test_creation_with_all_fields(self):
        """QueryResult created with all optional fields."""
        matches = [DocumentMatch("doc1", 0.9)]
        expansion = {"neural": 1.0, "network": 0.8}
        metadata = {"source": "test"}
        result = QueryResult(
            "neural",
            matches,
            expansion_terms=expansion,
            timing_ms=15.3,
            metadata=metadata
        )
        assert result.expansion_terms == expansion
        assert result.timing_ms == 15.3
        assert result.metadata == metadata

    def test_immutable(self):
        """QueryResult is immutable (frozen)."""
        matches = [DocumentMatch("doc1", 0.9)]
        result = QueryResult("test", matches)
        with pytest.raises(AttributeError):
            result.query = "new query"

    def test_repr(self):
        """String representation shows key info."""
        matches = [DocumentMatch("doc1", 0.9), DocumentMatch("doc2", 0.7)]
        result = QueryResult("test", matches, expansion_terms={"a": 1.0})
        repr_str = repr(result)
        assert "QueryResult" in repr_str
        assert "test" in repr_str
        assert "2 x DocumentMatch" in repr_str

    def test_to_dict(self):
        """Convert to dictionary with nested match dicts."""
        matches = [DocumentMatch("doc1", 0.9)]
        result = QueryResult("test", matches, timing_ms=10.0)
        d = result.to_dict()
        assert d["query"] == "test"
        assert len(d["matches"]) == 1
        assert d["matches"][0]["doc_id"] == "doc1"
        assert d["timing_ms"] == 10.0

    def test_top_match_property(self):
        """Top match property returns highest scoring match."""
        matches = [
            DocumentMatch("doc1", 0.5),
            DocumentMatch("doc2", 0.9),
            DocumentMatch("doc3", 0.7)
        ]
        result = QueryResult("test", matches)
        assert result.top_match.doc_id == "doc2"
        assert result.top_match.score == 0.9

    def test_top_match_empty_matches(self):
        """Top match returns None when no matches."""
        result = QueryResult("test", [])
        assert result.top_match is None

    def test_match_count_property(self):
        """Match count property returns number of matches."""
        matches = [DocumentMatch("doc1", 0.9), DocumentMatch("doc2", 0.7)]
        result = QueryResult("test", matches)
        assert result.match_count == 2

    def test_match_count_empty(self):
        """Match count returns 0 for empty matches."""
        result = QueryResult("test", [])
        assert result.match_count == 0

    def test_average_score_property(self):
        """Average score property calculates correctly."""
        matches = [
            DocumentMatch("doc1", 0.8),
            DocumentMatch("doc2", 0.6)
        ]
        result = QueryResult("test", matches)
        assert result.average_score == 0.7

    def test_average_score_empty_matches(self):
        """Average score returns 0.0 for empty matches."""
        result = QueryResult("test", [])
        assert result.average_score == 0.0

    def test_from_dict_document_matches(self):
        """Create from dictionary with DocumentMatch results."""
        data = {
            "query": "test",
            "matches": [
                {"doc_id": "doc1", "score": 0.9, "metadata": None},
                {"doc_id": "doc2", "score": 0.7, "metadata": None}
            ],
            "expansion_terms": {"test": 1.0},
            "timing_ms": 10.0
        }
        result = QueryResult.from_dict(data)
        assert result.query == "test"
        assert len(result.matches) == 2
        assert isinstance(result.matches[0], DocumentMatch)
        assert result.expansion_terms == {"test": 1.0}

    def test_from_dict_passage_matches(self):
        """Create from dictionary with PassageMatch results."""
        data = {
            "query": "test",
            "matches": [
                {
                    "doc_id": "doc1",
                    "text": "hello",
                    "score": 0.9,
                    "start": 0,
                    "end": 5,
                    "metadata": None
                }
            ]
        }
        result = QueryResult.from_dict(data)
        assert len(result.matches) == 1
        assert isinstance(result.matches[0], PassageMatch)
        assert result.matches[0].text == "hello"

    def test_from_dict_empty_matches(self):
        """Create from dictionary with empty matches."""
        data = {"query": "test", "matches": []}
        result = QueryResult.from_dict(data)
        assert result.match_count == 0

    def test_roundtrip_dict(self):
        """Roundtrip through dictionary preserves data."""
        matches = [DocumentMatch("doc1", 0.9, {"type": "test"})]
        original = QueryResult(
            "test query",
            matches,
            expansion_terms={"test": 1.0},
            timing_ms=15.0
        )
        d = original.to_dict()
        restored = QueryResult.from_dict(d)
        assert restored.query == original.query
        assert restored.match_count == original.match_count
        assert restored.expansion_terms == original.expansion_terms
        assert restored.timing_ms == original.timing_ms


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for batch conversion helper functions."""

    def test_convert_document_matches_basic(self):
        """Convert list of tuples to DocumentMatch objects."""
        results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        matches = convert_document_matches(results)
        assert len(matches) == 3
        assert all(isinstance(m, DocumentMatch) for m in matches)
        assert matches[0].doc_id == "doc1"
        assert matches[0].score == 0.9

    def test_convert_document_matches_with_metadata(self):
        """Convert with per-document metadata."""
        results = [("doc1", 0.9), ("doc2", 0.7)]
        metadata = {
            "doc1": {"type": "markdown"},
            "doc2": {"type": "python"}
        }
        matches = convert_document_matches(results, metadata)
        assert matches[0].metadata == {"type": "markdown"}
        assert matches[1].metadata == {"type": "python"}

    def test_convert_document_matches_partial_metadata(self):
        """Convert with metadata for some documents."""
        results = [("doc1", 0.9), ("doc2", 0.7)]
        metadata = {"doc1": {"type": "markdown"}}
        matches = convert_document_matches(results, metadata)
        assert matches[0].metadata == {"type": "markdown"}
        assert matches[1].metadata is None

    def test_convert_document_matches_empty(self):
        """Convert empty list."""
        matches = convert_document_matches([])
        assert matches == []

    def test_convert_passage_matches_basic(self):
        """Convert list of tuples to PassageMatch objects."""
        results = [
            ("doc1", "text1", 0, 5, 0.9),
            ("doc2", "text2", 10, 15, 0.7)
        ]
        matches = convert_passage_matches(results)
        assert len(matches) == 2
        assert all(isinstance(m, PassageMatch) for m in matches)
        assert matches[0].text == "text1"
        assert matches[0].start == 0
        assert matches[0].end == 5

    def test_convert_passage_matches_with_metadata(self):
        """Convert with per-document metadata."""
        results = [("doc1", "text1", 0, 5, 0.9)]
        metadata = {"doc1": {"line": 1}}
        matches = convert_passage_matches(results, metadata)
        assert matches[0].metadata == {"line": 1}

    def test_convert_passage_matches_empty(self):
        """Convert empty list."""
        matches = convert_passage_matches([])
        assert matches == []


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for realistic usage patterns."""

    def test_workflow_document_search(self):
        """Realistic workflow for document search results."""
        # Simulate search results
        raw_results = [
            ("neural_networks.md", 0.95),
            ("deep_learning.py", 0.87),
            ("ai_overview.md", 0.72)
        ]

        # Convert to dataclasses
        matches = convert_document_matches(raw_results)

        # Access with IDE autocomplete
        for match in matches:
            assert hasattr(match, 'doc_id')
            assert hasattr(match, 'score')

        # Get top result
        top = matches[0]
        assert top.doc_id == "neural_networks.md"

        # Convert back to tuple for legacy code
        tuples = [m.to_tuple() for m in matches]
        assert tuples[0] == ("neural_networks.md", 0.95)

    def test_workflow_passage_retrieval(self):
        """Realistic workflow for passage retrieval."""
        # Simulate passage results
        raw_results = [
            ("processor.py", "def compute_pagerank():\n    ...", 100, 150, 0.92),
            ("README.md", "PageRank is an algorithm...", 500, 600, 0.85)
        ]

        # Convert to dataclasses
        matches = convert_passage_matches(raw_results)

        # Access properties
        for match in matches:
            location = match.location
            length = match.length
            assert isinstance(location, str)
            assert isinstance(length, int)

        # Get citation info
        citation = f"[{matches[0].location}]"
        assert citation == "[processor.py:100-150]"

    def test_workflow_with_query_result(self):
        """Complete workflow with QueryResult wrapper."""
        # Search results
        matches = [
            DocumentMatch("doc1", 0.9),
            DocumentMatch("doc2", 0.7)
        ]

        # Wrap in QueryResult
        result = QueryResult(
            query="neural networks",
            matches=matches,
            expansion_terms={"neural": 1.0, "network": 0.8, "deep": 0.5},
            timing_ms=12.5
        )

        # Analyze results
        assert result.match_count == 2
        assert result.top_match.score == 0.9
        assert result.average_score == 0.8

        # Export for logging/storage
        result_dict = result.to_dict()
        assert "query" in result_dict
        assert "matches" in result_dict

        # Restore from storage
        restored = QueryResult.from_dict(result_dict)
        assert restored.query == result.query
        assert restored.match_count == result.match_count
