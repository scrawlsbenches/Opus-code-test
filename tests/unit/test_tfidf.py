"""
Unit Tests for TF-IDF and Cosine Similarity
============================================

Tests for TF-IDF calculation and cosine similarity algorithms:
- _tfidf_core: Core TF-IDF calculation algorithm
- compute_tfidf: TF-IDF computation on layers
- cosine_similarity: Cosine similarity between sparse vectors

Extracted from test_analysis.py for better organization (Task #T-20251215-213424-8400-004).
"""

import pytest
import math

from cortical.analysis import (
    _tfidf_core,
    compute_tfidf,
    cosine_similarity,
)


# =============================================================================
# TF-IDF CORE ALGORITHM TESTS
# =============================================================================


class TestTfidfCore:
    """Tests for _tfidf_core pure algorithm."""

    def test_empty_corpus(self):
        """Empty corpus returns empty dict."""
        result = _tfidf_core({}, num_docs=0)
        assert result == {}

    def test_single_term_single_doc(self):
        """Single term in single doc has IDF of 0."""
        stats = {
            "term": (5, 1, {"doc1": 5})
        }
        result = _tfidf_core(stats, num_docs=1)
        # IDF = log(1/1) = 0, so TF-IDF = 0
        assert result["term"][0] == pytest.approx(0.0)

    def test_rare_term_high_tfidf(self):
        """Rare term (in 1 of 10 docs) has high TF-IDF."""
        stats = {
            "rare": (5, 1, {"doc1": 5}),
            "common": (50, 10, {"doc1": 5, "doc2": 5, "doc3": 5, "doc4": 5, "doc5": 5,
                                "doc6": 5, "doc7": 5, "doc8": 5, "doc9": 5, "doc10": 5})
        }
        result = _tfidf_core(stats, num_docs=10)
        # Rare term should have higher TF-IDF
        assert result["rare"][0] > result["common"][0]

    def test_frequent_term_higher_tf(self):
        """Term with higher frequency has higher TF component."""
        stats = {
            "frequent": (100, 5, {"doc1": 100}),
            "infrequent": (10, 5, {"doc1": 10})
        }
        result = _tfidf_core(stats, num_docs=10)
        # Same IDF, but frequent has higher TF
        assert result["frequent"][0] > result["infrequent"][0]

    def test_per_doc_tfidf(self):
        """Per-document TF-IDF calculated correctly."""
        stats = {
            "term": (15, 2, {"doc1": 10, "doc2": 5})
        }
        result = _tfidf_core(stats, num_docs=10)
        global_tfidf, per_doc = result["term"]
        # doc1 has higher count, so higher per-doc TF-IDF
        assert per_doc["doc1"] > per_doc["doc2"]

    def test_zero_doc_frequency(self):
        """Term with zero doc frequency returns zero TF-IDF."""
        stats = {
            "ghost": (0, 0, {})
        }
        result = _tfidf_core(stats, num_docs=10)
        assert result["ghost"] == (0.0, {})

    def test_idf_formula(self):
        """Verify IDF formula: log(N/df)."""
        stats = {
            "term": (10, 5, {"doc1": 10})
        }
        result = _tfidf_core(stats, num_docs=10)
        expected_idf = math.log(10 / 5)  # log(2)
        expected_tf = math.log1p(10)
        expected_tfidf = expected_tf * expected_idf
        assert result["term"][0] == pytest.approx(expected_tfidf)


# =============================================================================
# LAYER TF-IDF TESTS
# =============================================================================


class TestComputeTfidf:
    """Tests for compute_tfidf() wrapper function."""

    def test_empty_corpus(self):
        """Empty corpus with no documents."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        compute_tfidf(layers, {})
        # Should not crash, no columns to update

    def test_single_term_single_doc(self):
        """Single term in single doc has zero TF-IDF."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer0.get_or_create_minicolumn("test")
        col.document_ids.add("doc1")
        col.occurrence_count = 5
        col.doc_occurrence_counts["doc1"] = 5

        layers = {CorticalLayer.TOKENS: layer0}
        documents = {"doc1": "test test test test test"}

        compute_tfidf(layers, documents)
        # IDF = log(1/1) = 0, so TF-IDF should be 0
        assert col.tfidf == pytest.approx(0.0)

    def test_rare_term_high_tfidf(self):
        """Rare term in 1 of 10 docs has high TF-IDF."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer0.get_or_create_minicolumn("rare")
        col.document_ids.add("doc1")
        col.occurrence_count = 5
        col.doc_occurrence_counts["doc1"] = 5

        layers = {CorticalLayer.TOKENS: layer0}
        documents = {f"doc{i}": "text" for i in range(10)}

        compute_tfidf(layers, documents)
        # IDF = log(10/1), TF = log1p(5)
        expected_idf = math.log(10)
        expected_tf = math.log1p(5)
        expected_tfidf = expected_tf * expected_idf
        assert col.tfidf == pytest.approx(expected_tfidf)

    def test_per_doc_tfidf(self):
        """Per-document TF-IDF calculated correctly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer0.get_or_create_minicolumn("term")
        col.document_ids.add("doc1")
        col.document_ids.add("doc2")
        col.occurrence_count = 15
        col.doc_occurrence_counts["doc1"] = 10
        col.doc_occurrence_counts["doc2"] = 5

        layers = {CorticalLayer.TOKENS: layer0}
        # Need more than 2 docs for non-zero IDF when term appears in 2
        documents = {"doc1": "text", "doc2": "text", "doc3": "other"}

        compute_tfidf(layers, documents)
        # doc1 has higher count, so higher per-doc TF-IDF
        # IDF = log(3/2) > 0, so TF-IDF will be non-zero
        assert col.tfidf_per_doc["doc1"] > col.tfidf_per_doc["doc2"]


# =============================================================================
# COSINE SIMILARITY TESTS
# =============================================================================


class TestCosineSimilarity:
    """Tests for cosine_similarity() utility function."""

    def test_empty_vectors(self):
        """Empty vectors return 0."""
        assert cosine_similarity({}, {}) == 0.0

    def test_no_common_keys(self):
        """Vectors with no common keys return 0."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 3.0, "d": 4.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_identical_vectors(self):
        """Identical vectors return 1.0."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_sparse(self):
        """Sparse orthogonal vectors return 0."""
        vec1 = {"a": 1.0}
        vec2 = {"b": 1.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_opposite_vectors(self):
        """Vectors with opposite values."""
        vec1 = {"a": 1.0, "b": 1.0}
        vec2 = {"a": -1.0, "b": -1.0}
        # Cosine of opposite vectors is -1
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_partial_overlap(self):
        """Vectors with partial overlap."""
        vec1 = {"a": 1.0, "b": 2.0, "c": 0.0}
        vec2 = {"a": 1.0, "b": 0.0, "c": 3.0}
        # Only "a" is common
        result = cosine_similarity(vec1, vec2)
        assert 0 < result < 1


class TestCosineSimilarityZeroMagnitude:
    """Test cosine_similarity zero magnitude case (line 2012)."""

    def test_zero_magnitude_vector(self):
        """Test cosine similarity with zero magnitude vectors."""
        vec1 = {"a": 0.0, "b": 0.0}
        vec2 = {"a": 1.0, "b": 1.0}

        result = cosine_similarity(vec1, vec2)

        # Should return 0.0 (line 2012)
        assert result == 0.0
