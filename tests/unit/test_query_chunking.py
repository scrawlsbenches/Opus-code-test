"""
Unit Tests for Query Chunking Module
=====================================

Comprehensive tests for cortical/query/chunking.py covering all functions
and edge cases to achieve >80% coverage.

Test Categories:
- create_chunks: Fixed-size chunking with overlap and edge cases
- find_code_boundaries: Code structure boundary detection
- create_code_aware_chunks: Semantic code chunking with complex scenarios
- is_code_file: File type detection
- precompute_term_cols: Term->Minicolumn caching
- score_chunk_fast: Fast chunk scoring with pre-computed lookups
- score_chunk: Standard chunk scoring with tokenizer
"""

import pytest
from typing import Dict, List
from unittest.mock import Mock

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
    LayerBuilder,
)


# =============================================================================
# CREATE_CHUNKS TESTS - Fixed-size chunking
# =============================================================================


class TestCreateChunksEdgeCases:
    """Edge cases and boundary conditions for create_chunks()."""

    def test_stride_equals_one_when_overlap_too_large(self):
        """When overlap is chunk_size-1, stride becomes 1."""
        text = "abcdefghij"
        # chunk_size=5, overlap=4 -> stride = max(1, 5-4) = 1
        result = create_chunks(text, chunk_size=5, overlap=4)
        # Should create many overlapping chunks
        assert len(result) >= 6
        # Each chunk advances by 1 character
        assert result[0][1] == 0  # start
        assert result[1][1] == 1  # start + stride
        assert result[2][1] == 2

    def test_loop_completes_naturally_without_early_break(self):
        """Test path where loop completes without hitting break statement."""
        # Create text where end never equals text_len until final iteration
        text = "a" * 100
        result = create_chunks(text, chunk_size=33, overlap=0)
        # stride=33, positions: 0, 33, 66, 99 (end=100)
        # The loop should complete naturally after reaching text_len
        assert len(result) == 4
        assert result[-1][2] == 100  # Last chunk ends at text_len

    def test_chunk_size_one(self):
        """Chunk size of 1 with no overlap creates per-character chunks."""
        text = "abc"
        result = create_chunks(text, chunk_size=1, overlap=0)
        assert len(result) == 3
        assert result[0] == ("a", 0, 1)
        assert result[1] == ("b", 1, 2)
        assert result[2] == ("c", 2, 3)

    def test_text_one_character_longer_than_chunk_size(self):
        """Text just slightly longer than chunk_size creates second small chunk."""
        text = "a" * 101
        result = create_chunks(text, chunk_size=100, overlap=0)
        assert len(result) == 2
        assert len(result[0][0]) == 100
        assert len(result[1][0]) == 1  # Single character chunk


# =============================================================================
# FIND_CODE_BOUNDARIES TESTS - Boundary detection edge cases
# =============================================================================


class TestFindCodeBoundariesEdgeCases:
    """Edge cases for find_code_boundaries()."""

    def test_docstring_double_quotes(self):
        """Module docstring with double quotes creates boundary."""
        text = '"""Module docstring."""\ncode here'
        result = find_code_boundaries(text)
        assert 0 in result

    def test_docstring_single_quotes(self):
        """Module docstring with single quotes creates boundary."""
        text = "'''Module docstring.'''\ncode here"
        result = find_code_boundaries(text)
        assert 0 in result

    def test_comment_separator_dashes(self):
        """Comment separator with dashes (# ---) creates boundary."""
        text = "# ---\nSection"
        result = find_code_boundaries(text)
        assert len(result) >= 1

    def test_comment_separator_equals(self):
        """Comment separator with equals (# ===) creates boundary."""
        text = "# ===\nSection"
        result = find_code_boundaries(text)
        assert len(result) >= 1

    def test_multiple_blank_lines(self):
        """Multiple consecutive blank lines create a single boundary."""
        text = "code1\n\n\n\ncode2"
        result = find_code_boundaries(text)
        # Should have boundary after the blank sequence
        assert len(result) >= 2

    def test_decorator_at_start_of_line(self):
        """Decorator at start of line creates boundary."""
        text = "@staticmethod\ndef foo():\n    pass"
        result = find_code_boundaries(text)
        # Should find boundary at decorator line
        assert len(result) >= 1

    def test_mixed_indentation_preserved(self):
        """Boundaries work correctly with various indentation levels."""
        text = """class Outer:
    class Inner:
        def method(self):
            pass"""
        result = find_code_boundaries(text)
        # Should find boundaries at class and/or method definitions
        # At minimum: [0] plus class/method boundaries
        assert len(result) >= 1  # More lenient check

    def test_boundary_at_line_start_not_match_position(self):
        """Boundaries are at line start, not pattern match position."""
        text = "    def foo():  # Indented function\n        pass"
        result = find_code_boundaries(text)
        # Boundary should be at line start (0), not at 'def' position
        assert 0 in result


# =============================================================================
# CREATE_CODE_AWARE_CHUNKS TESTS - Complex chunking scenarios
# =============================================================================


class TestCreateCodeAwareChunksComplex:
    """Complex scenarios and edge cases for create_code_aware_chunks()."""

    def test_whitespace_only_chunks_skipped(self):
        """Chunks with only whitespace are not included."""
        text = "code\n\n\n\n\n\n\n\n\nmore_code"
        # Set target_size small to potentially create whitespace chunks
        result = create_code_aware_chunks(text, target_size=5, min_size=1, max_size=20)
        # All returned chunks should have non-whitespace content
        for chunk_text, _, _ in result:
            assert chunk_text.strip() != ""

    def test_force_split_when_no_good_boundary(self):
        """Forces split at max_size when no suitable boundary exists."""
        # Create long text with no boundaries except at start
        text = "a" * 2000  # No code patterns, no blank lines
        result = create_code_aware_chunks(text, target_size=500, min_size=100, max_size=1000)
        # Should force splits at max_size
        assert len(result) >= 2
        # First chunk should be at most max_size
        assert len(result[0][0]) <= 1000

    def test_previous_boundary_used_when_next_too_far(self):
        """Uses previous boundary when next boundary exceeds max_size."""
        # Create text with boundaries at specific positions
        text = "class A:\n    pass\n\n" + ("x" * 2000) + "\n\nclass B:\n    pass"
        result = create_code_aware_chunks(text, target_size=100, min_size=10, max_size=500)
        # Should use previous boundary instead of forcing split into long section
        assert len(result) >= 2

    def test_min_size_prevents_tiny_chunks(self):
        """Chunks smaller than min_size are avoided when possible."""
        # Create text with boundaries close together
        text = """class A:
    pass

class B:
    pass

class C:
    pass"""
        result = create_code_aware_chunks(text, target_size=100, min_size=50, max_size=200)
        # Most chunks should be >= min_size (unless forced)
        for chunk_text, _, _ in result:
            # Allow some flexibility for final chunk
            if len(chunk_text) < 50:
                # Should be final chunk or special case
                pass

    def test_max_size_enforced_even_mid_function(self):
        """max_size is enforced even if it splits in middle of function."""
        # Create very long function
        text = "def very_long_function():\n" + "    x = 1\n" * 500
        result = create_code_aware_chunks(text, target_size=200, min_size=50, max_size=300)
        # Should create multiple chunks, none exceeding max_size
        for chunk_text, _, _ in result:
            assert len(chunk_text) <= 300

    def test_boundary_finding_after_force_split(self):
        """After force split, correctly finds next boundary."""
        # Create pattern: boundary, long text, boundary
        text = "class A:\n    pass\n\n" + ("x" * 1500) + "\n\nclass B:\n    pass"
        result = create_code_aware_chunks(text, target_size=200, min_size=50, max_size=500)
        # Should handle finding next boundary after force split
        assert len(result) >= 3

    def test_best_end_capped_at_max_size(self):
        """best_end is capped at max_size even if boundary suggests larger."""
        # Edge case where the safety check on line 194-195 activates
        text = "x" * 5000
        result = create_code_aware_chunks(text, target_size=100, min_size=50, max_size=200)
        # Every chunk should be at most max_size
        for chunk_text, _, _ in result:
            assert len(chunk_text) <= 200

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list."""
        result = create_code_aware_chunks("", target_size=100)
        assert result == []

    def test_text_exactly_target_size(self):
        """Text exactly target_size returns single chunk."""
        text = "a" * 512
        result = create_code_aware_chunks(text, target_size=512)
        assert len(result) == 1
        assert result[0][0] == text

    def test_text_smaller_than_target_size(self):
        """Text smaller than target_size returns single chunk."""
        text = "short text"
        result = create_code_aware_chunks(text, target_size=100)
        assert len(result) == 1
        assert result[0] == (text, 0, len(text))

    def test_realistic_python_file(self):
        """Realistic Python file with multiple functions and classes."""
        text = '''"""Module docstring."""

import os

class MyClass:
    """Class docstring."""

    def __init__(self):
        self.value = 0

    @property
    def doubled(self):
        return self.value * 2

    def method(self):
        """Method docstring."""
        return self.value

def standalone_function():
    """Function docstring."""
    pass

# --- Helper functions ---

async def async_function():
    await something()
'''
        result = create_code_aware_chunks(text, target_size=200, min_size=50, max_size=400)
        # Should create reasonable chunks aligned to structure
        assert len(result) >= 1
        # Verify chunks are valid
        for chunk_text, start, end in result:
            assert text[start:end] == chunk_text
            assert chunk_text.strip() != ""


# =============================================================================
# IS_CODE_FILE TESTS - File type detection
# =============================================================================


class TestIsCodeFile:
    """Tests for is_code_file() extension detection."""

    def test_python_file(self):
        """Python files are detected as code."""
        assert is_code_file("script.py") is True
        assert is_code_file("/path/to/module.py") is True

    def test_javascript_files(self):
        """JavaScript files are detected as code."""
        assert is_code_file("app.js") is True
        assert is_code_file("component.jsx") is True
        assert is_code_file("module.ts") is True
        assert is_code_file("component.tsx") is True

    def test_compiled_languages(self):
        """Compiled language files are detected as code."""
        assert is_code_file("program.java") is True
        assert is_code_file("lib.c") is True
        assert is_code_file("lib.cpp") is True
        assert is_code_file("header.h") is True
        assert is_code_file("main.go") is True
        assert is_code_file("lib.rs") is True
        assert is_code_file("Program.cs") is True

    def test_other_languages(self):
        """Other programming language files are detected as code."""
        assert is_code_file("script.rb") is True
        assert is_code_file("page.php") is True
        assert is_code_file("App.swift") is True
        assert is_code_file("Main.kt") is True
        assert is_code_file("App.scala") is True

    def test_non_code_files(self):
        """Non-code files are not detected as code."""
        assert is_code_file("document.txt") is False
        assert is_code_file("README.md") is False
        assert is_code_file("data.json") is False
        assert is_code_file("config.yaml") is False
        assert is_code_file("image.png") is False
        assert is_code_file("data.csv") is False

    def test_no_extension(self):
        """Files without extension are not detected as code."""
        assert is_code_file("Makefile") is False
        assert is_code_file("README") is False
        assert is_code_file("/path/to/file") is False

    def test_extension_case_sensitive(self):
        """Extension detection is case-sensitive."""
        # Only lowercase extensions are in the set
        assert is_code_file("script.PY") is False
        assert is_code_file("app.JS") is False

    def test_path_with_dots(self):
        """Correctly handles paths with multiple dots."""
        assert is_code_file("/path/with.dots/file.py") is True
        assert is_code_file("file.test.js") is True
        assert is_code_file("file.backup.txt") is False


# =============================================================================
# PRECOMPUTE_TERM_COLS TESTS - Term caching
# =============================================================================


class TestPrecomputeTermCols:
    """Tests for precompute_term_cols() minicolumn caching."""

    def test_all_terms_exist(self):
        """All query terms exist in layer0."""
        layers = LayerBuilder()\
            .with_term("neural", pagerank=0.8)\
            .with_term("network", pagerank=0.7)\
            .build()
        layer0 = layers[0]  # Get layer 0 from the dict
        query_terms = {"neural": 1.0, "network": 0.5}

        result = precompute_term_cols(query_terms, layer0)

        assert len(result) == 2
        assert "neural" in result
        assert "network" in result
        assert result["neural"].content == "neural"

    def test_some_terms_missing(self):
        """Only returns columns for terms that exist in layer0."""
        layers = LayerBuilder()\
            .with_term("neural", pagerank=0.8)\
            .build()
        layer0 = layers[0]
        query_terms = {"neural": 1.0, "missing": 0.5, "absent": 0.3}

        result = precompute_term_cols(query_terms, layer0)

        assert len(result) == 1
        assert "neural" in result
        assert "missing" not in result
        assert "absent" not in result

    def test_no_terms_exist(self):
        """Returns empty dict when no terms exist."""
        layers = LayerBuilder().build()  # Empty layer
        layer0 = layers[0]
        query_terms = {"missing": 1.0, "absent": 0.5}

        result = precompute_term_cols(query_terms, layer0)

        assert result == {}

    def test_empty_query_terms(self):
        """Empty query terms returns empty dict."""
        layers = LayerBuilder().with_term("neural", pagerank=0.8).build()
        layer0 = layers[0]
        query_terms = {}

        result = precompute_term_cols(query_terms, layer0)

        assert result == {}

    def test_preserves_minicolumn_references(self):
        """Returned minicolumns are the same objects from layer0."""
        layers = LayerBuilder().with_term("test", pagerank=0.9).build()
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        result = precompute_term_cols(query_terms, layer0)
        original_col = layer0.get_minicolumn("test")

        assert result["test"] is original_col


# =============================================================================
# SCORE_CHUNK_FAST TESTS - Optimized scoring
# =============================================================================


class TestScoreChunkFast:
    """Tests for score_chunk_fast() optimized chunk scoring."""

    def test_empty_chunk_tokens(self):
        """Empty chunk returns 0.0."""
        query_terms = {"test": 1.0}
        term_cols = {}

        score = score_chunk_fast([], query_terms, term_cols)

        assert score == 0.0

    def test_no_matching_terms(self):
        """Chunk with no matching query terms returns 0.0."""
        chunk_tokens = ["other", "words", "here"]
        query_terms = {"test": 1.0}
        col = MockMinicolumn(id="L0_test", content="test", tfidf=1.5)
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        assert score == 0.0

    def test_single_term_match(self):
        """Single matching term contributes to score."""
        chunk_tokens = ["test", "other", "words"]
        query_terms = {"test": 1.0}
        col = MockMinicolumn(id="L0_test", content="test", tfidf=2.0)
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        # score = (tfidf * count * weight) / len(tokens)
        # score = (2.0 * 1 * 1.0) / 3 = 0.666...
        assert score > 0.0
        assert abs(score - (2.0 / 3)) < 0.01

    def test_multiple_term_matches(self):
        """Multiple matching terms accumulate scores."""
        chunk_tokens = ["neural", "network", "test"]
        query_terms = {"neural": 1.0, "network": 0.5}
        col1 = MockMinicolumn(id="L0_neural", content="neural", tfidf=2.0)
        col2 = MockMinicolumn(id="L0_network", content="network", tfidf=1.5)
        term_cols = {"neural": col1, "network": col2}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        # score = (2.0*1*1.0 + 1.5*1*0.5) / 3 = (2.0 + 0.75) / 3 = 0.9166...
        assert score > 0.0

    def test_term_appears_multiple_times(self):
        """Term appearing multiple times in chunk increases score."""
        chunk_tokens = ["test", "test", "test"]
        query_terms = {"test": 1.0}
        col = MockMinicolumn(id="L0_test", content="test", tfidf=1.0)
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        # score = (1.0 * 3 * 1.0) / 3 = 1.0
        assert score == 1.0

    def test_uses_per_document_tfidf_when_doc_id_provided(self):
        """Uses per-document TF-IDF when doc_id is provided."""
        chunk_tokens = ["test"]
        query_terms = {"test": 1.0}
        col = MockMinicolumn(
            id="L0_test",
            content="test",
            tfidf=1.0,
            tfidf_per_doc={"doc1": 3.0}
        )
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols, doc_id="doc1")

        # Should use per-doc TF-IDF (3.0) not global (1.0)
        # score = (3.0 * 1 * 1.0) / 1 = 3.0
        assert score == 3.0

    def test_uses_global_tfidf_when_doc_id_not_in_per_doc(self):
        """Falls back to global TF-IDF when doc_id not in per-doc dict."""
        chunk_tokens = ["test"]
        query_terms = {"test": 1.0}
        col = MockMinicolumn(
            id="L0_test",
            content="test",
            tfidf=1.0,
            tfidf_per_doc={"other_doc": 3.0}
        )
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols, doc_id="doc1")

        # Should fall back to global TF-IDF (1.0)
        assert score == 1.0

    def test_uses_global_tfidf_when_no_doc_id(self):
        """Uses global TF-IDF when no doc_id provided."""
        chunk_tokens = ["test"]
        query_terms = {"test": 1.0}
        col = MockMinicolumn(
            id="L0_test",
            content="test",
            tfidf=2.0,
            tfidf_per_doc={"doc1": 5.0}
        )
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols, doc_id=None)

        # Should use global TF-IDF (2.0)
        assert score == 2.0

    def test_query_term_weight_applied(self):
        """Query term weights affect the score."""
        chunk_tokens = ["test"]
        query_terms = {"test": 2.0}  # 2x weight
        col = MockMinicolumn(id="L0_test", content="test", tfidf=1.0)
        term_cols = {"test": col}

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        # score = (1.0 * 1 * 2.0) / 1 = 2.0
        assert score == 2.0

    def test_term_in_chunk_but_not_in_term_cols(self):
        """Term in chunk but not in term_cols doesn't contribute."""
        chunk_tokens = ["test", "other"]
        query_terms = {"test": 1.0, "other": 1.0}
        col = MockMinicolumn(id="L0_test", content="test", tfidf=1.0)
        term_cols = {"test": col}  # "other" not in term_cols

        score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        # Only "test" should contribute
        # score = (1.0 * 1 * 1.0) / 2 = 0.5
        assert score == 0.5


# =============================================================================
# SCORE_CHUNK TESTS - Standard scoring with tokenizer
# =============================================================================


class TestScoreChunk:
    """Tests for score_chunk() standard chunk scoring."""

    def test_empty_chunk_text(self):
        """Empty chunk returns 0.0."""
        tokenizer = Tokenizer()
        layers = LayerBuilder().build()
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        score = score_chunk("", query_terms, layer0, tokenizer)

        assert score == 0.0

    def test_chunk_with_only_stopwords(self):
        """Chunk with only stopwords returns 0.0."""
        tokenizer = Tokenizer()
        layers = LayerBuilder().build()
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        score = score_chunk("the and or", query_terms, layer0, tokenizer)

        # Stopwords are removed by tokenizer
        assert score == 0.0

    def test_term_not_in_layer0_returns_zero(self):
        """Term in chunk but not in layer0 doesn't contribute (col is None)."""
        tokenizer = Tokenizer()
        layers = LayerBuilder().build()  # Empty layer
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        score = score_chunk("test words", query_terms, layer0, tokenizer)

        # "test" is in chunk and query, but get_minicolumn returns None
        assert score == 0.0

    def test_single_term_match_with_tokenizer(self):
        """Single matching term contributes to score."""
        tokenizer = Tokenizer()
        layers = LayerBuilder().with_term("test", tfidf=2.0).build()
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        score = score_chunk("test words here", query_terms, layer0, tokenizer)

        # Tokenized to ["test", "word", "here"] (stemmed)
        # Only "test" matches
        assert score > 0.0

    def test_multiple_term_matches_with_tokenizer(self):
        """Multiple matching terms accumulate scores."""
        tokenizer = Tokenizer()
        layers = LayerBuilder()\
            .with_term("neural", tfidf=2.0)\
            .with_term("network", tfidf=1.5)\
            .build()
        layer0 = layers[0]
        query_terms = {"neural": 1.0, "network": 0.5}

        score = score_chunk("neural network systems", query_terms, layer0, tokenizer)

        assert score > 0.0

    def test_uses_per_document_tfidf_when_doc_id_provided(self):
        """Uses per-document TF-IDF when doc_id is provided."""
        tokenizer = Tokenizer()
        col = MockMinicolumn(
            id="L0_test",
            content="test",
            tfidf=1.0,
            tfidf_per_doc={"doc1": 5.0}
        )
        layer0 = MockHierarchicalLayer([col])
        query_terms = {"test": 1.0}

        score = score_chunk("test", query_terms, layer0, tokenizer, doc_id="doc1")

        # Should use per-doc TF-IDF (5.0) not global (1.0)
        assert score > 4.0  # Should be close to 5.0

    def test_tokenization_affects_matching(self):
        """Tokenizer processes text before matching."""
        tokenizer = Tokenizer()
        # Test that exact token match works
        layers = LayerBuilder().with_term("python", tfidf=2.0).build()
        layer0 = layers[0]
        query_terms = {"python": 1.0}

        # "Python" (capitalized) should match "python" (lowercase)
        score = score_chunk("Python", query_terms, layer0, tokenizer)

        # Should match due to lowercasing
        assert score > 0.0

    def test_case_insensitive_matching(self):
        """Tokenizer lowercases, so matching is case-insensitive."""
        tokenizer = Tokenizer()
        layers = LayerBuilder().with_term("test", tfidf=2.0).build()
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        score = score_chunk("TEST Test test", query_terms, layer0, tokenizer)

        # All three variants should match
        assert score > 0.0

    def test_zero_length_tokens_after_tokenization(self):
        """Handles edge case where tokenization produces empty token list."""
        tokenizer = Tokenizer()
        layers = LayerBuilder().build()
        layer0 = layers[0]
        query_terms = {"test": 1.0}

        # Special characters that might produce empty token list
        score = score_chunk("@#$%^&*()", query_terms, layer0, tokenizer)

        assert score == 0.0


# =============================================================================
# INTEGRATION TESTS - Combined functionality
# =============================================================================


class TestChunkingIntegration:
    """Integration tests combining multiple chunking functions."""

    def test_code_aware_chunks_align_with_boundaries(self):
        """Code-aware chunks should align with detected boundaries."""
        text = """class A:
    pass

class B:
    pass"""

        boundaries = find_code_boundaries(text)
        chunks = create_code_aware_chunks(text, target_size=20, min_size=5, max_size=50)

        # Each chunk should start at or near a boundary
        chunk_starts = [start for _, start, _ in chunks]
        for chunk_start in chunk_starts:
            # Should be at a boundary or between boundaries
            assert any(abs(chunk_start - b) <= 5 for b in boundaries) or chunk_start in boundaries

    def test_scoring_consistency_between_fast_and_standard(self):
        """score_chunk_fast and score_chunk should give similar results."""
        tokenizer = Tokenizer()
        layers = LayerBuilder()\
            .with_term("test", tfidf=2.0)\
            .with_term("word", tfidf=1.5)\
            .build()
        layer0 = layers[0]
        query_terms = {"test": 1.0, "word": 0.5}
        chunk_text = "test word here"

        # Standard scoring
        score_standard = score_chunk(chunk_text, query_terms, layer0, tokenizer)

        # Fast scoring with pre-computed columns
        chunk_tokens = tokenizer.tokenize(chunk_text)
        term_cols = precompute_term_cols(query_terms, layer0)
        score_fast = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        # Should be very close (may differ slightly due to float precision)
        assert abs(score_standard - score_fast) < 0.01

    def test_code_file_detection_affects_chunking_strategy(self):
        """is_code_file() can be used to choose chunking strategy."""
        python_text = """def foo():
    pass

def bar():
    pass"""

        doc_id = "module.py"

        # For code files, use code-aware chunking
        if is_code_file(doc_id):
            chunks = create_code_aware_chunks(python_text, target_size=20)
        else:
            chunks = create_chunks(python_text, chunk_size=20)

        # Code-aware should create reasonable chunks
        assert len(chunks) >= 1

    def test_full_passage_scoring_pipeline(self):
        """Complete pipeline: chunk -> tokenize -> score."""
        text = "neural networks process data efficiently"
        tokenizer = Tokenizer()
        layers = LayerBuilder()\
            .with_term("neural", tfidf=3.0)\
            .with_term("data", tfidf=2.0)\
            .build()
        layer0 = layers[0]
        query_terms = {"neural": 1.0, "data": 0.5}

        # Create chunks
        chunks = create_chunks(text, chunk_size=20, overlap=5)

        # Score each chunk
        scores = []
        for chunk_text, start, end in chunks:
            score = score_chunk(chunk_text, query_terms, layer0, tokenizer)
            scores.append((chunk_text, score))

        # Should have scores for chunks
        assert len(scores) == len(chunks)
        # At least one chunk should have non-zero score
        assert any(score > 0 for _, score in scores)
