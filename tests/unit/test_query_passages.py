"""
Unit Tests for Query Passages Module
=====================================

Task #172: Unit tests for cortical/query/passages.py

Tests passage retrieval, chunking, and scoring functions for RAG systems.
Covers both passages.py and chunking.py modules.

Test Categories:
- Chunking: create_chunks, create_code_aware_chunks, code boundaries
- Code detection: is_code_file, find_code_boundaries
- Chunk scoring: score_chunk, score_chunk_fast, precompute_term_cols
- Passage retrieval: find_passages_for_query with various options
- Batch operations: find_documents_batch, find_passages_batch
"""

import pytest
from typing import Dict, List
from unittest.mock import Mock, patch

from cortical.query.passages import (
    find_passages_for_query,
    find_documents_batch,
    find_passages_batch,
)
from cortical.query.chunking import (
    create_chunks,
    create_code_aware_chunks,
    find_code_boundaries,
    is_code_file,
    precompute_term_cols,
    score_chunk_fast,
    score_chunk,
    CODE_BOUNDARY_PATTERN,
)
from cortical.tokenizer import Tokenizer
from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
)


# =============================================================================
# CHUNKING TESTS
# =============================================================================


class TestCreateChunks:
    """Tests for create_chunks() fixed-size chunking."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        result = create_chunks("", chunk_size=100, overlap=20)
        assert result == []

    def test_text_smaller_than_chunk_size(self):
        """Text smaller than chunk_size returns single chunk."""
        text = "Short text."
        result = create_chunks(text, chunk_size=100, overlap=20)
        assert len(result) == 1
        assert result[0] == (text, 0, len(text))

    def test_text_exactly_chunk_size(self):
        """Text exactly chunk_size returns single chunk."""
        text = "a" * 100
        result = create_chunks(text, chunk_size=100, overlap=20)
        assert len(result) == 1
        assert result[0] == (text, 0, 100)

    def test_two_chunks_no_overlap(self):
        """Text creating two chunks with no overlap."""
        text = "a" * 200
        result = create_chunks(text, chunk_size=100, overlap=0)
        assert len(result) == 2
        assert result[0] == ("a" * 100, 0, 100)
        assert result[1] == ("a" * 100, 100, 200)

    def test_two_chunks_with_overlap(self):
        """Text creating two chunks with overlap."""
        text = "a" * 150
        result = create_chunks(text, chunk_size=100, overlap=20)
        # stride = 100 - 20 = 80
        # Chunk 1: [0:100]
        # Chunk 2: [80:150]
        assert len(result) == 2
        assert result[0] == ("a" * 100, 0, 100)
        assert result[1] == ("a" * 70, 80, 150)

    def test_multiple_chunks(self):
        """Text creating multiple chunks."""
        text = "a" * 300
        result = create_chunks(text, chunk_size=100, overlap=10)
        # stride = 90, so chunks at: 0, 90, 180, 270 (exceeds)
        assert len(result) == 4
        assert result[0][1] == 0  # start position
        assert result[1][1] == 90
        assert result[2][1] == 180
        assert result[-1][2] == 300  # last end position

    def test_overlap_creates_redundancy(self):
        """Overlap causes text to appear in multiple chunks."""
        text = "abcdefghij"
        result = create_chunks(text, chunk_size=6, overlap=2)
        # stride = 4, chunks: [0:6], [4:10]
        assert len(result) == 2
        # "ef" appears in both chunks
        assert result[0][0] == "abcdef"
        assert result[1][0] == "efghij"

    def test_chunk_positions_correct(self):
        """Chunk positions match text content."""
        text = "Hello World Python"
        result = create_chunks(text, chunk_size=10, overlap=3)
        for chunk_text, start, end in result:
            assert text[start:end] == chunk_text

    def test_invalid_chunk_size_zero(self):
        """Zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_chunks("text", chunk_size=0, overlap=0)

    def test_invalid_chunk_size_negative(self):
        """Negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_chunks("text", chunk_size=-10, overlap=0)

    def test_invalid_overlap_negative(self):
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            create_chunks("text", chunk_size=100, overlap=-5)

    def test_invalid_overlap_equals_chunk_size(self):
        """overlap == chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            create_chunks("text", chunk_size=100, overlap=100)

    def test_invalid_overlap_exceeds_chunk_size(self):
        """overlap > chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            create_chunks("text", chunk_size=100, overlap=150)

    def test_very_small_chunks(self):
        """Very small chunk_size works correctly."""
        text = "abcdefgh"
        result = create_chunks(text, chunk_size=3, overlap=1)
        # stride = 2, chunks: [0:3], [2:5], [4:7], [6:8]
        assert len(result) == 4
        assert result[0][0] == "abc"
        assert result[1][0] == "cde"

    def test_very_large_overlap(self):
        """Large overlap (but < chunk_size) works correctly."""
        text = "a" * 200
        result = create_chunks(text, chunk_size=100, overlap=99)
        # stride = 1, so 101 chunks needed to cover 200 chars
        assert len(result) > 100


# =============================================================================
# CODE BOUNDARIES TESTS
# =============================================================================


class TestFindCodeBoundaries:
    """Tests for find_code_boundaries() semantic boundary detection."""

    def test_empty_text(self):
        """Empty text returns just [0]."""
        result = find_code_boundaries("")
        assert result == [0]

    def test_no_boundaries(self):
        """Text without code patterns returns [0] and blank lines."""
        text = "This is plain text\n\nwith blank lines."
        result = find_code_boundaries(text)
        assert 0 in result
        # Blank line at position after "text\n\n"
        assert len(result) >= 1

    def test_class_definition(self):
        """Class definition creates boundary."""
        text = "class Foo:\n    pass"
        result = find_code_boundaries(text)
        assert 0 in result
        assert len(result) >= 1

    def test_function_definition(self):
        """Function definition creates boundary."""
        text = "def bar():\n    return 42"
        result = find_code_boundaries(text)
        assert 0 in result

    def test_async_function_definition(self):
        """Async function definition creates boundary."""
        text = "async def fetch():\n    await something()"
        result = find_code_boundaries(text)
        assert 0 in result

    def test_decorator(self):
        """Decorator creates boundary."""
        text = "@property\ndef value(self):\n    return self._value"
        result = find_code_boundaries(text)
        assert 0 in result
        # Should have boundary at decorator line

    def test_multiple_functions(self):
        """Multiple functions create multiple boundaries."""
        text = "def foo():\n    pass\n\ndef bar():\n    pass"
        result = find_code_boundaries(text)
        assert len(result) >= 2  # At least start + one function

    def test_comment_separator(self):
        """Comment separator creates boundary."""
        text = "# ---\nSection 1\n# ===\nSection 2"
        result = find_code_boundaries(text)
        assert len(result) >= 2  # Multiple separator boundaries

    def test_blank_lines_create_boundaries(self):
        """Blank line sequences create boundaries."""
        text = "line1\n\n\nline2"
        result = find_code_boundaries(text)
        # Boundary after blank lines
        assert len(result) >= 2

    def test_boundaries_sorted(self):
        """Boundaries are returned in sorted order."""
        text = "def c():\n    pass\n\ndef a():\n    pass\n\ndef b():\n    pass"
        result = find_code_boundaries(text)
        assert result == sorted(result)

    def test_boundaries_unique(self):
        """No duplicate boundaries."""
        text = "class A:\n    pass\n\nclass B:\n    pass"
        result = find_code_boundaries(text)
        assert len(result) == len(set(result))

    def test_complex_code_structure(self):
        """Complex code with mixed patterns."""
        text = '''
class MyClass:
    """Docstring"""

    @property
    def value(self):
        return self._value

    def method(self):
        pass

def standalone():
    pass
'''
        result = find_code_boundaries(text)
        # Multiple boundaries for class, decorator, methods
        assert len(result) >= 4


# =============================================================================
# CODE-AWARE CHUNKING TESTS
# =============================================================================


class TestCreateCodeAwareChunks:
    """Tests for create_code_aware_chunks() semantic chunking."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        result = create_code_aware_chunks("")
        assert result == []

    def test_small_text(self):
        """Text smaller than target_size returns single chunk."""
        text = "class Foo:\n    pass"
        result = create_code_aware_chunks(text, target_size=100)
        assert len(result) == 1
        assert result[0] == (text, 0, len(text))

    def test_respects_target_size(self):
        """Chunks are created near target_size."""
        text = "def func():\n    pass\n\n" * 20  # ~360 chars
        result = create_code_aware_chunks(text, target_size=100, max_size=200)
        # Should create multiple chunks
        assert len(result) >= 2
        # Each chunk should be <= max_size
        for chunk_text, start, end in result:
            assert end - start <= 200

    def test_aligns_to_function_boundaries(self):
        """Chunks align to function definitions when possible."""
        text = "def foo():\n    pass\n\ndef bar():\n    pass\n\ndef baz():\n    pass"
        result = create_code_aware_chunks(text, target_size=15, max_size=50)
        # Should have chunks starting at function definitions
        chunk_texts = [chunk[0] for chunk in result]
        # At least one chunk should start with "def"
        assert any(chunk.strip().startswith("def") for chunk in chunk_texts)

    def test_respects_min_size(self):
        """Won't create chunks smaller than min_size."""
        text = "a\n\nb\n\nc\n\nd\n\ne"
        result = create_code_aware_chunks(text, target_size=10, min_size=5)
        for chunk_text, start, end in result:
            if chunk_text.strip():  # Ignore whitespace-only
                assert end - start >= 5 or end == len(text)

    def test_respects_max_size(self):
        """Forces split at max_size even if no boundary."""
        text = "x" * 500  # No boundaries
        result = create_code_aware_chunks(text, target_size=100, max_size=150)
        for chunk_text, start, end in result:
            assert end - start <= 150

    def test_prefers_blank_lines(self):
        """Prefers splitting at blank lines."""
        text = "Section 1\nContent\n\nSection 2\nContent\n\nSection 3\nContent"
        result = create_code_aware_chunks(text, target_size=20, max_size=40)
        # Should have multiple chunks split at blank lines
        assert len(result) >= 2

    def test_class_definitions_kept_together(self):
        """Tries to keep class definitions in same chunk when possible."""
        text = '''class Small:
    def method(self):
        pass

class Another:
    pass
'''
        result = create_code_aware_chunks(text, target_size=50, max_size=100)
        # Classes should ideally be in separate chunks or together if small
        assert len(result) >= 1

    def test_no_empty_chunks(self):
        """Doesn't create empty or whitespace-only chunks."""
        text = "def foo():\n    pass\n\n\n\ndef bar():\n    pass"
        result = create_code_aware_chunks(text, target_size=10, max_size=30)
        for chunk_text, start, end in result:
            assert chunk_text.strip() != ""


# =============================================================================
# CODE FILE DETECTION TESTS
# =============================================================================


class TestIsCodeFile:
    """Tests for is_code_file() extension detection."""

    def test_python_file(self):
        """Python files detected."""
        assert is_code_file("script.py")
        assert is_code_file("path/to/module.py")
        assert is_code_file("/absolute/path/file.py")

    def test_javascript_files(self):
        """JavaScript files detected."""
        assert is_code_file("app.js")
        assert is_code_file("component.jsx")
        assert is_code_file("module.ts")
        assert is_code_file("component.tsx")

    def test_common_languages(self):
        """Common programming languages detected."""
        assert is_code_file("Main.java")
        assert is_code_file("program.c")
        assert is_code_file("program.cpp")
        assert is_code_file("header.h")
        assert is_code_file("main.go")
        assert is_code_file("lib.rs")
        assert is_code_file("script.rb")
        assert is_code_file("index.php")

    def test_other_languages(self):
        """Other languages detected."""
        assert is_code_file("App.swift")
        assert is_code_file("MainActivity.kt")
        assert is_code_file("Program.scala")
        assert is_code_file("Program.cs")

    def test_text_files_not_code(self):
        """Text files not detected as code."""
        assert not is_code_file("README.md")
        assert not is_code_file("notes.txt")
        assert not is_code_file("data.json")
        assert not is_code_file("config.yaml")
        assert not is_code_file("style.css")
        assert not is_code_file("page.html")

    def test_no_extension(self):
        """Files without extension not detected as code."""
        assert not is_code_file("README")
        assert not is_code_file("Makefile")

    def test_case_sensitive(self):
        """Extension check is case sensitive."""
        assert is_code_file("script.py")
        assert not is_code_file("script.PY")  # Capital extension


# =============================================================================
# PRECOMPUTE TERM COLS TESTS
# =============================================================================


class TestPrecomputeTermCols:
    """Tests for precompute_term_cols() optimization helper."""

    def test_empty_query_terms(self):
        """Empty query terms returns empty dict."""
        layer = MockHierarchicalLayer([])
        result = precompute_term_cols({}, layer)
        assert result == {}

    def test_single_term_exists(self):
        """Single term that exists returns mapping."""
        col = MockMinicolumn(content="neural", layer=0)
        layer = MockHierarchicalLayer([col])
        result = precompute_term_cols({"neural": 1.0}, layer)
        assert "neural" in result
        assert result["neural"] is col

    def test_single_term_missing(self):
        """Single term that doesn't exist returns empty dict."""
        layer = MockHierarchicalLayer([])
        result = precompute_term_cols({"missing": 1.0}, layer)
        assert result == {}

    def test_multiple_terms_all_exist(self):
        """Multiple terms all exist."""
        cols = [
            MockMinicolumn(content="neural", layer=0),
            MockMinicolumn(content="networks", layer=0),
        ]
        layer = MockHierarchicalLayer(cols)
        query_terms = {"neural": 1.0, "networks": 0.8}
        result = precompute_term_cols(query_terms, layer)
        assert len(result) == 2
        assert "neural" in result
        assert "networks" in result

    def test_multiple_terms_some_missing(self):
        """Multiple terms with some missing."""
        col = MockMinicolumn(content="neural", layer=0)
        layer = MockHierarchicalLayer([col])
        query_terms = {"neural": 1.0, "missing": 0.5}
        result = precompute_term_cols(query_terms, layer)
        assert len(result) == 1
        assert "neural" in result
        assert "missing" not in result

    def test_ignores_term_weights(self):
        """Term weights don't affect lookup, only presence."""
        col = MockMinicolumn(content="test", layer=0)
        layer = MockHierarchicalLayer([col])
        result = precompute_term_cols({"test": 999.0}, layer)
        assert "test" in result


# =============================================================================
# CHUNK SCORING TESTS
# =============================================================================


class TestScoreChunkFast:
    """Tests for score_chunk_fast() with pre-computed lookups."""

    def test_empty_chunk_tokens(self):
        """Empty chunk returns zero score."""
        result = score_chunk_fast([], {}, {})
        assert result == 0.0

    def test_no_query_terms_match(self):
        """No matching terms returns zero score."""
        chunk_tokens = ["foo", "bar"]
        query_terms = {"baz": 1.0}
        term_cols = {}
        result = score_chunk_fast(chunk_tokens, query_terms, term_cols)
        assert result == 0.0

    def test_single_term_match(self):
        """Single matching term returns positive score."""
        col = MockMinicolumn(content="neural", tfidf=2.5)
        chunk_tokens = ["neural", "networks"]
        query_terms = {"neural": 1.0}
        term_cols = {"neural": col}
        result = score_chunk_fast(chunk_tokens, query_terms, term_cols)
        # score = tfidf * count * weight / len
        # score = 2.5 * 1 * 1.0 / 2 = 1.25
        assert result == pytest.approx(1.25)

    def test_multiple_term_matches(self):
        """Multiple matching terms accumulate score."""
        col1 = MockMinicolumn(content="neural", tfidf=2.0)
        col2 = MockMinicolumn(content="networks", tfidf=1.5)
        chunk_tokens = ["neural", "networks", "processing"]
        query_terms = {"neural": 1.0, "networks": 1.0}
        term_cols = {"neural": col1, "networks": col2}
        result = score_chunk_fast(chunk_tokens, query_terms, term_cols)
        # score = (2.0 * 1 * 1.0 + 1.5 * 1 * 1.0) / 3 = 3.5 / 3
        assert result == pytest.approx(3.5 / 3)

    def test_term_appears_multiple_times(self):
        """Term appearing multiple times in chunk increases score."""
        col = MockMinicolumn(content="neural", tfidf=2.0)
        chunk_tokens = ["neural", "networks", "neural"]
        query_terms = {"neural": 1.0}
        term_cols = {"neural": col}
        result = score_chunk_fast(chunk_tokens, query_terms, term_cols)
        # score = 2.0 * 2 * 1.0 / 3 = 4.0 / 3
        assert result == pytest.approx(4.0 / 3)

    def test_query_term_weight_affects_score(self):
        """Higher query term weight increases score."""
        col = MockMinicolumn(content="neural", tfidf=2.0)
        chunk_tokens = ["neural"]

        query_low = {"neural": 0.5}
        term_cols = {"neural": col}
        score_low = score_chunk_fast(chunk_tokens, query_low, term_cols)

        query_high = {"neural": 2.0}
        score_high = score_chunk_fast(chunk_tokens, query_high, term_cols)

        # Higher weight should give higher score
        assert score_high > score_low

    def test_per_doc_tfidf(self):
        """Uses per-document TF-IDF when doc_id provided."""
        col = MockMinicolumn(
            content="neural",
            tfidf=1.0,
            tfidf_per_doc={"doc1": 3.0, "doc2": 0.5}
        )
        chunk_tokens = ["neural"]
        query_terms = {"neural": 1.0}
        term_cols = {"neural": col}

        # Without doc_id, uses global tfidf
        score_global = score_chunk_fast(chunk_tokens, query_terms, term_cols)
        assert score_global == pytest.approx(1.0)

        # With doc_id, uses per-doc tfidf
        score_doc1 = score_chunk_fast(chunk_tokens, query_terms, term_cols, doc_id="doc1")
        assert score_doc1 == pytest.approx(3.0)

        score_doc2 = score_chunk_fast(chunk_tokens, query_terms, term_cols, doc_id="doc2")
        assert score_doc2 == pytest.approx(0.5)

    def test_normalizes_by_chunk_length(self):
        """Score normalized by chunk length to avoid length bias."""
        col = MockMinicolumn(content="neural", tfidf=2.0)
        query_terms = {"neural": 1.0}
        term_cols = {"neural": col}

        # Short chunk
        short_tokens = ["neural", "networks"]
        score_short = score_chunk_fast(short_tokens, query_terms, term_cols)

        # Long chunk with same term
        long_tokens = ["neural", "networks", "process", "data", "learning"]
        score_long = score_chunk_fast(long_tokens, query_terms, term_cols)

        # Short chunk should score higher (same match, less noise)
        assert score_short > score_long


class TestScoreChunk:
    """Tests for score_chunk() standard scoring function."""

    def test_empty_chunk_text(self):
        """Empty chunk returns zero score."""
        layer = MockHierarchicalLayer([])
        tokenizer = Tokenizer()
        result = score_chunk("", {}, layer, tokenizer)
        assert result == 0.0

    def test_no_matches(self):
        """Chunk with no matching terms returns zero."""
        col = MockMinicolumn(content="neural", tfidf=2.0)
        layer = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        result = score_chunk("foo bar baz", {"neural": 1.0}, layer, tokenizer)
        assert result == 0.0

    def test_single_match(self):
        """Chunk with matching term returns positive score."""
        col = MockMinicolumn(content="neural", tfidf=2.0)
        layer = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        result = score_chunk("neural networks", {"neural": 1.0}, layer, tokenizer)
        assert result > 0.0

    def test_equivalent_to_fast_version(self):
        """score_chunk should give same result as score_chunk_fast."""
        col = MockMinicolumn(content="neural", tfidf=2.5)
        layer = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        text = "neural networks process data"
        query_terms = {"neural": 1.0}

        # Standard version
        score_std = score_chunk(text, query_terms, layer, tokenizer)

        # Fast version
        tokens = tokenizer.tokenize(text)
        term_cols = precompute_term_cols(query_terms, layer)
        score_fast = score_chunk_fast(tokens, query_terms, term_cols)

        assert score_std == pytest.approx(score_fast)


# =============================================================================
# FIND PASSAGES FOR QUERY TESTS
# =============================================================================


class TestFindPassagesForQuery:
    """Tests for find_passages_for_query() main passage retrieval."""

    def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()
        result = find_passages_for_query(
            "", layers, tokenizer, {}, use_expansion=False, use_definition_search=False
        )
        assert result == []

    def test_empty_documents(self):
        """Empty documents returns empty results."""
        layers = MockLayers.single_term("neural")
        tokenizer = Tokenizer()
        result = find_passages_for_query(
            "neural", layers, tokenizer, {}, use_expansion=False, use_definition_search=False
        )
        assert result == []

    def test_query_term_not_in_corpus(self):
        """Query term not in corpus returns empty results."""
        layers = MockLayers.single_term("neural")
        tokenizer = Tokenizer()
        documents = {"doc1": "unrelated content here"}
        result = find_passages_for_query(
            "missing", layers, tokenizer, documents, use_expansion=False, use_definition_search=False
        )
        assert result == []

    def test_single_document_single_match(self):
        """Single document with matching term returns passage."""
        col = MockMinicolumn(
            content="neural",
            tfidf=2.0,
            document_ids={"doc1"}
        )
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "This text contains neural networks."}

        result = find_passages_for_query(
            "neural", layers, tokenizer, documents,
            top_n=1, use_expansion=False, use_definition_search=False
        )

        assert len(result) == 1
        passage_text, doc_id, start, end, score = result[0]
        assert doc_id == "doc1"
        assert "neural" in passage_text
        assert score > 0

    def test_top_n_limits_results(self):
        """top_n parameter limits number of results."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1", "doc2", "doc3"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {
            "doc1": "test " * 100,
            "doc2": "test " * 100,
            "doc3": "test " * 100,
        }

        result = find_passages_for_query(
            "test", layers, tokenizer, documents,
            top_n=2, chunk_size=50, overlap=10, use_expansion=False, use_definition_search=False
        )

        assert len(result) == 2

    def test_chunk_size_affects_passage_length(self):
        """chunk_size parameter affects passage length."""
        col = MockMinicolumn(content="word", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "word " * 1000}

        result = find_passages_for_query(
            "word", layers, tokenizer, documents,
            chunk_size=100, overlap=0, use_expansion=False, use_definition_search=False
        )

        # Passages should be approximately chunk_size
        for passage_text, _, start, end, _ in result:
            assert end - start <= 100

    def test_overlap_creates_redundant_passages(self):
        """Overlap causes text to appear in multiple passages."""
        col = MockMinicolumn(content="word", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "word " * 100}

        result = find_passages_for_query(
            "word", layers, tokenizer, documents,
            top_n=10, chunk_size=50, overlap=25, use_expansion=False, use_definition_search=False
        )

        # Should have overlapping passages
        assert len(result) > 2

    def test_doc_filter_restricts_search(self):
        """doc_filter parameter restricts search to specific documents."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1", "doc2", "doc3"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {
            "doc1": "test content",
            "doc2": "test content",
            "doc3": "test content",
        }

        result = find_passages_for_query(
            "test", layers, tokenizer, documents,
            doc_filter=["doc2"], use_expansion=False, use_definition_search=False
        )

        # Should only return passages from doc2
        for _, doc_id, _, _, _ in result:
            assert doc_id == "doc2"

    def test_passages_sorted_by_score(self):
        """Results sorted by relevance score descending."""
        col1 = MockMinicolumn(content="rare", tfidf=5.0, document_ids={"doc1"})
        col2 = MockMinicolumn(content="common", tfidf=0.5, document_ids={"doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col1, col2])
        tokenizer = Tokenizer()
        documents = {
            "doc1": "This document has the rare term.",
            "doc2": "This document has the common term.",
        }

        result = find_passages_for_query(
            "rare common", layers, tokenizer, documents,
            use_expansion=False, use_definition_search=False
        )

        # Scores should be descending
        scores = [score for _, _, _, _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_doc_id_not_in_documents(self):
        """Handles case where doc_id from scoring doesn't exist in documents."""
        # Create a mock where layer has doc_id but it's not in documents dict
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1", "doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        # Only provide doc1, not doc2
        documents = {"doc1": "test content"}

        result = find_passages_for_query(
            "test", layers, tokenizer, documents,
            use_expansion=False, use_definition_search=False
        )

        # Should only return passages from doc1
        for _, doc_id, _, _, _ in result:
            assert doc_id == "doc1"

    def test_passage_positions_valid(self):
        """Passage positions are valid slices of document text."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "test content here with test words"}

        result = find_passages_for_query(
            "test", layers, tokenizer, documents,
            use_expansion=False, use_definition_search=False
        )

        for passage_text, doc_id, start, end, _ in result:
            assert documents[doc_id][start:end] == passage_text

    def test_use_code_aware_chunks_for_code_files(self):
        """Code files use semantic chunking when use_code_aware_chunks=True."""
        col = MockMinicolumn(content="def", tfidf=1.0, document_ids={"test.py"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"test.py": "def foo():\n    pass\n\ndef bar():\n    pass"}

        result = find_passages_for_query(
            "def", layers, tokenizer, documents,
            use_code_aware_chunks=True, use_expansion=False, use_definition_search=False
        )

        # Should have results from code file
        assert len(result) > 0

    @patch('cortical.query.passages.find_definition_passages')
    def test_definition_search_with_doc_filter(self, mock_def_search):
        """Definition search respects doc_filter."""
        # Mock definition search to return empty
        mock_def_search.return_value = []

        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1", "doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "test content", "doc2": "test content"}

        result = find_passages_for_query(
            "class Foo", layers, tokenizer, documents,
            doc_filter=["doc1"], use_definition_search=True, use_expansion=False
        )

        # Verify definition search was called with filtered docs
        assert mock_def_search.called
        call_args = mock_def_search.call_args
        docs_searched = call_args[0][1]  # Second arg is documents dict
        assert "doc1" in docs_searched
        assert "doc2" not in docs_searched

    @patch('cortical.query.passages.find_definition_passages')
    def test_definition_only_results_with_boosting(self, mock_def_search):
        """When only definition results exist, they can be boosted."""
        # Mock definition search to return a result
        mock_def_search.return_value = [
            ("def foo():\n    pass", "doc1.py", 0, 20, 5.0)
        ]

        # Empty layers so no query terms found
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()
        documents = {"doc1.py": "def foo():\n    pass"}

        result = find_passages_for_query(
            "def foo", layers, tokenizer, documents,
            use_definition_search=True, use_expansion=False,
            apply_doc_boost=True, prefer_docs=True
        )

        # Should return the definition passage
        assert len(result) > 0

    @patch('cortical.query.passages.find_definition_passages')
    def test_definition_passages_avoid_duplicates(self, mock_def_search):
        """Definition passages don't duplicate regular chunking."""
        # Mock definition search to return a passage at position [0, 50]
        mock_def_search.return_value = [
            ("This is a definition passage", "doc1", 0, 50, 10.0)
        ]

        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "This is a definition passage that contains test"}

        result = find_passages_for_query(
            "test", layers, tokenizer, documents,
            chunk_size=50, overlap=0,
            use_definition_search=True, use_expansion=False
        )

        # Should have results, but no duplicate at the exact same position
        assert len(result) > 0


# =============================================================================
# BATCH OPERATIONS TESTS
# =============================================================================


class TestFindDocumentsBatch:
    """Tests for find_documents_batch() batch document retrieval."""

    def test_empty_queries(self):
        """Empty query list returns empty results."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()
        result = find_documents_batch([], layers, tokenizer)
        assert result == []

    def test_query_terms_not_found(self):
        """Query with terms not in corpus returns empty results."""
        layers = MockLayers.single_term("test")
        tokenizer = Tokenizer()
        result = find_documents_batch(["nonexistent"], layers, tokenizer, use_expansion=False)
        assert len(result) == 1
        assert result[0] == []

    def test_single_query(self):
        """Single query returns single result list."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()

        result = find_documents_batch(["test"], layers, tokenizer, use_expansion=False)

        assert len(result) == 1
        assert len(result[0]) >= 1  # Has results for query

    def test_multiple_queries(self):
        """Multiple queries return multiple result lists."""
        col1 = MockMinicolumn(content="neural", tfidf=1.0, document_ids={"doc1"})
        col2 = MockMinicolumn(content="data", tfidf=1.0, document_ids={"doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col1, col2])
        tokenizer = Tokenizer()

        result = find_documents_batch(
            ["neural", "data"], layers, tokenizer, use_expansion=False
        )

        assert len(result) == 2

    def test_query_with_no_results(self):
        """Query with no results returns empty list."""
        layers = MockLayers.single_term("test")
        tokenizer = Tokenizer()

        result = find_documents_batch(
            ["missing"], layers, tokenizer, use_expansion=False
        )

        assert len(result) == 1
        assert result[0] == []

    def test_top_n_limits_each_query(self):
        """top_n applies to each query independently."""
        col = MockMinicolumn(
            content="test",
            tfidf=1.0,
            document_ids={"doc1", "doc2", "doc3", "doc4", "doc5"}
        )
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()

        result = find_documents_batch(
            ["test", "test"], layers, tokenizer, top_n=3, use_expansion=False
        )

        assert len(result) == 2
        assert len(result[0]) <= 3
        assert len(result[1]) <= 3


class TestFindPassagesBatch:
    """Tests for find_passages_batch() batch passage retrieval."""

    def test_empty_queries(self):
        """Empty query list returns empty results."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()
        result = find_passages_batch([], layers, tokenizer, {})
        assert result == []

    def test_single_query(self):
        """Single query returns single result list."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "test content here"}

        result = find_passages_batch(
            ["test"], layers, tokenizer, documents, use_expansion=False
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_multiple_queries(self):
        """Multiple queries return multiple result lists."""
        col1 = MockMinicolumn(content="neural", tfidf=1.0, document_ids={"doc1"})
        col2 = MockMinicolumn(content="data", tfidf=1.0, document_ids={"doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col1, col2])
        tokenizer = Tokenizer()
        documents = {
            "doc1": "neural networks content",
            "doc2": "data processing content",
        }

        result = find_passages_batch(
            ["neural", "data"], layers, tokenizer, documents, use_expansion=False
        )

        assert len(result) == 2

    def test_query_with_no_matches(self):
        """Query with no matches returns empty list."""
        layers = MockLayers.single_term("test")
        tokenizer = Tokenizer()
        documents = {"doc1": "unrelated content"}

        result = find_passages_batch(
            ["missing"], layers, tokenizer, documents, use_expansion=False
        )

        assert len(result) == 1
        assert result[0] == []

    def test_empty_query_terms(self):
        """Query that tokenizes to nothing returns empty results."""
        layers = MockLayers.single_term("test")
        tokenizer = Tokenizer()
        documents = {"doc1": "test content"}

        # Query with only stop words or empty string
        result = find_passages_batch(
            [""], layers, tokenizer, documents, use_expansion=False
        )

        assert len(result) == 1
        assert result[0] == []

    def test_top_n_limits_each_query(self):
        """top_n applies to each query independently."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "test " * 1000}

        result = find_passages_batch(
            ["test", "test"], layers, tokenizer, documents,
            top_n=3, chunk_size=50, overlap=10, use_expansion=False
        )

        assert len(result) == 2
        assert len(result[0]) <= 3
        assert len(result[1]) <= 3

    def test_doc_filter_applies_to_all_queries(self):
        """doc_filter applies to all queries in batch."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1", "doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {
            "doc1": "test content",
            "doc2": "test content",
        }

        result = find_passages_batch(
            ["test", "test"], layers, tokenizer, documents,
            doc_filter=["doc1"], use_expansion=False
        )

        # All results should be from doc1
        for query_results in result:
            for _, doc_id, _, _, _ in query_results:
                assert doc_id == "doc1"

    def test_doc_filter_excludes_documents_from_chunking(self):
        """doc_filter prevents documents from being chunked."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1", "doc2", "doc3"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {
            "doc1": "test content",
            "doc2": "test content",
            "doc3": "test content",
        }

        # Filter to only doc1, so doc2 and doc3 won't be chunked
        result = find_passages_batch(
            ["test"], layers, tokenizer, documents,
            doc_filter=["doc1"], use_expansion=False
        )

        # Should only have results from doc1
        assert len(result) == 1
        for _, doc_id, _, _, _ in result[0]:
            assert doc_id in ["doc1"]

    def test_chunk_caching_efficiency(self):
        """Chunks are cached and reused across queries (performance optimization)."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "test " * 1000}

        # Multiple queries should use cached chunks
        result = find_passages_batch(
            ["test"] * 5, layers, tokenizer, documents,
            chunk_size=100, overlap=20, use_expansion=False
        )

        # All queries should return results (chunks cached internally)
        assert len(result) == 5
        for query_result in result:
            assert len(query_result) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPassageRetrievalIntegration:
    """Integration tests combining multiple components."""

    def test_full_passage_retrieval_pipeline(self):
        """Complete pipeline from query to ranked passages."""
        # Setup corpus
        col_neural = MockMinicolumn(content="neural", tfidf=2.0, document_ids={"doc1"})
        col_data = MockMinicolumn(content="data", tfidf=1.5, document_ids={"doc1", "doc2"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col_neural, col_data])

        tokenizer = Tokenizer()
        documents = {
            "doc1": "Neural networks process data efficiently. " * 10,
            "doc2": "Data processing systems handle information. " * 10,
        }

        # Run query
        result = find_passages_for_query(
            "neural data", layers, tokenizer, documents,
            top_n=5, chunk_size=100, overlap=20, use_expansion=False, use_definition_search=False
        )

        # Validate results
        assert len(result) > 0
        assert len(result) <= 5

        for passage_text, doc_id, start, end, score in result:
            assert doc_id in documents
            assert documents[doc_id][start:end] == passage_text
            assert score > 0

    def test_code_file_semantic_chunking(self):
        """Code files get semantic chunking aligned to boundaries."""
        col = MockMinicolumn(content="def", tfidf=1.0, document_ids={"code.py"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        code_content = """
def function_one():
    pass

def function_two():
    pass

class MyClass:
    def method(self):
        pass
"""
        documents = {"code.py": code_content}

        result = find_passages_for_query(
            "def", layers, tokenizer, documents,
            chunk_size=50, use_code_aware_chunks=True, use_expansion=False, use_definition_search=False
        )

        # Should have passages aligned to code boundaries
        assert len(result) > 0

    def test_batch_operations_consistency(self):
        """Batch operations give consistent results with single calls."""
        col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {"doc1": "test content here"}

        # Single call
        single = find_passages_for_query(
            "test", layers, tokenizer, documents,
            use_expansion=False, use_definition_search=False
        )

        # Batch call
        batch = find_passages_batch(
            ["test"], layers, tokenizer, documents, use_expansion=False
        )

        # Results should match
        assert len(batch) == 1
        assert len(batch[0]) == len(single)

    def test_doc_type_boosting_changes_scores(self):
        """Doc-type boosting changes scores for test files.

        Task #180: Verify that apply_doc_boost parameter actually affects scores.
        Test files (with 'test' in name) should get lower scores when boosted.
        """
        # Create two documents: regular code and test file
        col = MockMinicolumn(
            content="filter",
            tfidf=2.0,
            document_ids={"data_processor.py", "test_data_processor.py"}
        )
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()
        documents = {
            "data_processor.py": "filter data records efficiently",
            "test_data_processor.py": "filter data records in tests",
        }

        # Without boosting
        results_no_boost = find_passages_for_query(
            "filter data", layers, tokenizer, documents,
            apply_doc_boost=False, use_expansion=False, use_definition_search=False
        )

        # With boosting (prefer_docs=True to enable boosting)
        results_with_boost = find_passages_for_query(
            "filter data", layers, tokenizer, documents,
            apply_doc_boost=True, prefer_docs=True, use_expansion=False, use_definition_search=False
        )

        # Both should return results
        assert len(results_no_boost) > 0
        assert len(results_with_boost) > 0

        # Extract scores for each document
        def get_scores_by_doc(results):
            scores = {}
            for _, doc_id, _, _, score in results:
                if doc_id not in scores or score > scores[doc_id]:
                    scores[doc_id] = score
            return scores

        scores_no_boost = get_scores_by_doc(results_no_boost)
        scores_with_boost = get_scores_by_doc(results_with_boost)

        # Without boosting, both files should have similar scores
        # (may differ slightly due to document length normalization)

        # With boosting, test file should have lower score than regular file
        if "data_processor.py" in scores_with_boost and "test_data_processor.py" in scores_with_boost:
            # Test file should be penalized (0.8x boost vs 1.0x)
            assert scores_with_boost["test_data_processor.py"] < scores_with_boost["data_processor.py"], \
                f"Test file should have lower score with boosting. " \
                f"Got test={scores_with_boost['test_data_processor.py']:.3f}, " \
                f"regular={scores_with_boost['data_processor.py']:.3f}"

        # Verify scores actually changed between boosted and non-boosted
        # At least one document's score should be different
        changed = False
        for doc_id in set(scores_no_boost.keys()) & set(scores_with_boost.keys()):
            if abs(scores_no_boost[doc_id] - scores_with_boost[doc_id]) > 0.001:
                changed = True
                break

        assert changed, "Boosting should change at least one document's score"

    def test_definition_only_with_conceptual_query_boosting(self):
        """
        Definition-only results with conceptual query trigger doc-type boosting.

        Task #172: Cover lines 138-144 in passages.py where query_terms is empty
        but definition_passages exist and should_boost=True.
        """
        # Use a query that will find definitions but won't tokenize to anything
        # (e.g., all stop words or special characters)
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([])  # No terms in layer
        tokenizer = Tokenizer()

        # Documents with actual class definitions
        documents = {
            "docs/guide.md": "class MyClass:\n    '''Documentation class'''\n    pass",
            "test_file.py": "class MyClass:\n    '''Test class'''\n    pass"
        }

        # Use conceptual query with definition pattern
        # The query "class MyClass" will be found by definition search
        # but won't produce query_terms if we filter code keywords
        result = find_passages_for_query(
            "class MyClass", layers, tokenizer, documents,
            use_definition_search=True, use_expansion=False,
            filter_code_stop_words=True,  # Filters "class"
            apply_doc_boost=True, auto_detect_intent=True,
            doc_metadata={
                "docs/guide.md": {"doc_type": "documentation"},
                "test_file.py": {"doc_type": "test"}
            }
        )

        # Should return definition results even without query terms
        assert len(result) > 0

    def test_chunk_overlaps_with_definition_passage_skipped(self):
        """
        Chunks that overlap with definition passages are skipped to avoid duplicates.

        Task #172: Cover line 195 in passages.py where chunks overlapping
        definition passages are skipped.
        """
        col = MockMinicolumn(content="function", tfidf=1.0, document_ids={"doc1.py"})
        layers = MockLayers.empty()
        layers[0] = MockHierarchicalLayer([col])
        tokenizer = Tokenizer()

        # Document with a function definition at the start
        # The definition search will find it, and regular chunking will also create
        # a chunk at the same position [0, chunk_size]
        text = "def my_function():\n    '''Function docstring'''\n    pass\n" + ("x = 1\n" * 100)
        documents = {"doc1.py": text}

        result = find_passages_for_query(
            "def my_function", layers, tokenizer, documents,
            chunk_size=100, overlap=0,
            use_definition_search=True, use_expansion=False
        )

        # Should have results including the definition
        assert len(result) > 0
        # Count how many passages start at position 0
        # Should only be 1 (the definition), not 2 (definition + duplicate chunk)
        passages_at_start = sum(1 for _, _, start, _, _ in result if start == 0)
        # Due to deduplication, should only have one passage at position 0
        assert passages_at_start == 1

    def test_batch_passages_with_doc_not_in_cache(self):
        """
        Batch passage search skips documents not in chunk cache.

        Task #172: Cover line 393 in passages.py where doc_id not in
        doc_chunks_cache is skipped.
        """
        # Mock find_documents_for_query to return doc2 which won't be in cache
        with patch('cortical.query.passages.find_documents_for_query') as mock_find_docs:
            # Mock returns both doc1 and doc2, but only doc1 is in documents dict
            mock_find_docs.return_value = [("doc1", 1.0), ("doc2", 0.8)]

            col = MockMinicolumn(content="test", tfidf=1.0, document_ids={"doc1"})
            layers = MockLayers.empty()
            layers[0] = MockHierarchicalLayer([col])
            tokenizer = Tokenizer()

            # Only provide doc1 in documents - doc2 won't be chunked
            documents = {"doc1": "test content"}

            result = find_passages_batch(
                ["test"], layers, tokenizer, documents,
                use_expansion=False
            )

            # Should successfully return results for doc1 and skip doc2
            assert len(result) == 1
            assert len(result[0]) > 0
            # All results should be from doc1 (doc2 was skipped)
            for _, doc_id, _, _, _ in result[0]:
                assert doc_id == "doc1"
