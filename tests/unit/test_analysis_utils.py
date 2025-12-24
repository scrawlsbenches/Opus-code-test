"""
Unit tests for cortical/analysis/utils.py

Focuses on edge cases and error paths to improve coverage.
"""

import pytest
from cortical.analysis.utils import (
    SparseMatrix,
    cosine_similarity,
    _doc_similarity,
    _vector_similarity
)


class TestSparseMatrix:
    """Test SparseMatrix class."""

    def test_set_and_get_basic(self):
        """Test basic set and get operations."""
        matrix = SparseMatrix(10, 10)
        matrix.set(0, 0, 5.0)
        assert matrix.get(0, 0) == 5.0
        assert matrix.get(1, 1) == 0.0

    def test_set_zero_removes_entry(self):
        """Test that setting 0 removes existing entry (line 48)."""
        matrix = SparseMatrix(10, 10)

        # Set a non-zero value
        matrix.set(2, 3, 7.5)
        assert (2, 3) in matrix.data
        assert matrix.get(2, 3) == 7.5

        # Set it to zero - should remove it
        matrix.set(2, 3, 0.0)
        assert (2, 3) not in matrix.data
        assert matrix.get(2, 3) == 0.0

    def test_set_zero_nonexistent_entry(self):
        """Test setting zero for entry that doesn't exist."""
        matrix = SparseMatrix(10, 10)

        # Set zero for entry that was never set
        matrix.set(5, 5, 0.0)
        assert (5, 5) not in matrix.data

    def test_multiply_transpose_simple(self):
        """Test multiply_transpose with simple matrix."""
        matrix = SparseMatrix(rows=3, cols=3)

        # Create a simple document-term matrix
        # Doc 0: has terms 0 and 1
        # Doc 1: has terms 1 and 2
        # Doc 2: has term 0
        matrix.set(0, 0, 1.0)
        matrix.set(0, 1, 1.0)
        matrix.set(1, 1, 1.0)
        matrix.set(1, 2, 1.0)
        matrix.set(2, 0, 1.0)

        result = matrix.multiply_transpose()

        # Check diagonal elements (self-similarity)
        assert result.get(0, 0) == 2.0  # Term 0 appears in 2 docs
        assert result.get(1, 1) == 2.0  # Term 1 appears in 2 docs
        assert result.get(2, 2) == 1.0  # Term 2 appears in 1 doc

        # Check off-diagonal elements (co-occurrence)
        # Terms 0 and 1 co-occur in doc 0
        assert result.get(0, 1) == 1.0
        assert result.get(1, 0) == 1.0

        # Terms 1 and 2 co-occur in doc 1
        assert result.get(1, 2) == 1.0
        assert result.get(2, 1) == 1.0

        # Terms 0 and 2 don't co-occur
        assert result.get(0, 2) == 0.0
        assert result.get(2, 0) == 0.0

    def test_multiply_transpose_weighted(self):
        """Test multiply_transpose with weighted values (lines 85-97)."""
        matrix = SparseMatrix(rows=2, cols=3)

        # Weighted document-term matrix
        matrix.set(0, 0, 2.0)
        matrix.set(0, 1, 3.0)
        matrix.set(1, 1, 4.0)
        matrix.set(1, 2, 5.0)

        result = matrix.multiply_transpose()

        # Diagonal elements
        assert result.get(0, 0) == 4.0  # 2.0^2
        assert result.get(1, 1) == 25.0  # 3.0^2 + 4.0^2
        assert result.get(2, 2) == 25.0  # 5.0^2

        # Off-diagonal elements
        assert result.get(0, 1) == 6.0  # 2.0 * 3.0
        assert result.get(1, 0) == 6.0  # Symmetric

        assert result.get(1, 2) == 20.0  # 4.0 * 5.0
        assert result.get(2, 1) == 20.0  # Symmetric

        # No co-occurrence
        assert result.get(0, 2) == 0.0
        assert result.get(2, 0) == 0.0

    def test_multiply_transpose_larger_matrix(self):
        """Test multiply_transpose with larger matrix to ensure all paths."""
        matrix = SparseMatrix(rows=5, cols=5)

        # Create a more complex pattern
        matrix.set(0, 0, 1.0)
        matrix.set(0, 1, 2.0)
        matrix.set(1, 1, 1.0)
        matrix.set(1, 2, 3.0)
        matrix.set(2, 2, 1.0)
        matrix.set(2, 3, 2.0)
        matrix.set(3, 3, 1.0)
        matrix.set(3, 4, 4.0)

        result = matrix.multiply_transpose()

        # Verify symmetry
        for i in range(5):
            for j in range(5):
                assert result.get(i, j) == result.get(j, i)

        # Verify diagonals are positive
        for i in range(5):
            assert result.get(i, i) >= 0

    def test_get_nonzero(self):
        """Test get_nonzero returns all entries (line 108)."""
        matrix = SparseMatrix(5, 5)

        matrix.set(0, 1, 1.5)
        matrix.set(2, 3, 2.5)
        matrix.set(4, 4, 3.5)

        nonzero = matrix.get_nonzero()

        assert len(nonzero) == 3
        assert (0, 1, 1.5) in nonzero
        assert (2, 3, 2.5) in nonzero
        assert (4, 4, 3.5) in nonzero

    def test_get_nonzero_empty(self):
        """Test get_nonzero on empty matrix."""
        matrix = SparseMatrix(10, 10)
        nonzero = matrix.get_nonzero()
        assert nonzero == []

    def test_multiply_transpose_no_cooccurrence(self):
        """Test multiply_transpose when columns don't share rows."""
        matrix = SparseMatrix(rows=3, cols=3)

        # Each column appears in different rows (no co-occurrence)
        matrix.set(0, 0, 1.0)
        matrix.set(1, 1, 1.0)
        matrix.set(2, 2, 1.0)

        result = matrix.multiply_transpose()

        # Only diagonal should be non-zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert result.get(i, j) == 1.0
                else:
                    assert result.get(i, j) == 0.0


class TestCosineSimilarity:
    """Test cosine_similarity function."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_cosine_similarity_no_common_keys(self):
        """Test cosine similarity with no common keys."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 3.0, "d": 4.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_cosine_similarity_zero_magnitude(self):
        """Test cosine similarity when one vector has zero magnitude (line 141)."""
        # Test case where vectors share keys but one has all zero values
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"a": 0.0, "b": 0.0}
        # vec2 has magnitude 0, should return 0.0
        assert cosine_similarity(vec1, vec2) == 0.0

        # Test reverse case
        vec1_zero = {"a": 0.0, "b": 0.0}
        vec2_normal = {"a": 1.0, "b": 2.0}
        assert cosine_similarity(vec1_zero, vec2_normal) == 0.0


class TestDocSimilarity:
    """Test _doc_similarity function."""

    def test_doc_similarity_identical(self):
        """Test document similarity with identical sets."""
        docs = frozenset(["doc1", "doc2", "doc3"])
        assert _doc_similarity(docs, docs) == 1.0

    def test_doc_similarity_disjoint(self):
        """Test document similarity with disjoint sets."""
        docs1 = frozenset(["doc1", "doc2"])
        docs2 = frozenset(["doc3", "doc4"])
        assert _doc_similarity(docs1, docs2) == 0.0

    def test_doc_similarity_partial_overlap(self):
        """Test document similarity with partial overlap."""
        docs1 = frozenset(["doc1", "doc2", "doc3"])
        docs2 = frozenset(["doc2", "doc3", "doc4"])
        # Intersection: {doc2, doc3} = 2
        # Union: {doc1, doc2, doc3, doc4} = 4
        # Similarity: 2/4 = 0.5
        assert _doc_similarity(docs1, docs2) == 0.5

    def test_doc_similarity_empty_set(self):
        """Test document similarity with empty sets."""
        docs1 = frozenset(["doc1", "doc2"])
        docs2 = frozenset()
        assert _doc_similarity(docs1, docs2) == 0.0


class TestVectorSimilarity:
    """Test _vector_similarity function."""

    def test_vector_similarity_identical(self):
        """Test vector similarity with identical vectors."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert _vector_similarity(vec, vec) == 1.0

    def test_vector_similarity_disjoint(self):
        """Test vector similarity with no common keys."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 3.0, "d": 4.0}
        # min_sum = 0, max_sum = 1+2+3+4 = 10
        # Should be 0
        assert _vector_similarity(vec1, vec2) == 0.0

    def test_vector_similarity_partial_overlap(self):
        """Test vector similarity with partial overlap."""
        vec1 = {"a": 2.0, "b": 4.0}
        vec2 = {"b": 3.0, "c": 5.0}
        # Common key: b
        # Union: {a, b, c}
        # min_sum: min(2,0) + min(4,3) + min(0,5) = 0 + 3 + 0 = 3
        # max_sum: max(2,0) + max(4,3) + max(0,5) = 2 + 4 + 5 = 11
        expected = 3.0 / 11.0
        assert _vector_similarity(vec1, vec2) == pytest.approx(expected)

    def test_vector_similarity_empty_vectors(self):
        """Test vector similarity with empty vectors (line 182)."""
        vec1 = {}
        vec2 = {"a": 1.0}
        assert _vector_similarity(vec1, vec2) == 0.0

        vec1 = {"a": 1.0}
        vec2 = {}
        assert _vector_similarity(vec1, vec2) == 0.0

    def test_vector_similarity_both_empty(self):
        """Test vector similarity with both vectors empty (line 182)."""
        vec1 = {}
        vec2 = {}
        # Both empty -> no union -> should return 0.0
        assert _vector_similarity(vec1, vec2) == 0.0

    def test_vector_similarity_zero_max_sum(self):
        """Test edge case where max_sum could be zero (defensive check)."""
        # This shouldn't happen in practice, but test the safety check
        vec1 = {"a": 0.0}
        vec2 = {"a": 0.0}
        # Both have key "a" with value 0
        # Union = {a}, min_sum = 0, max_sum = 0
        # Should return 0.0 to avoid division by zero
        assert _vector_similarity(vec1, vec2) == 0.0
