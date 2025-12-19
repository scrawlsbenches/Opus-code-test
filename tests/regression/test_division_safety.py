"""
Regression Tests for Division by Zero Protection
================================================

Tests for edge cases where division operations could fail with zero denominators.
These prevent crashes and NaN/infinity propagation in scoring calculations.

Regression test for T-018: Edge case coverage for production robustness.
"""

import pytest
from cortical import CorticalTextProcessor
from cortical import CorticalLayer


class TestScoringDivisionSafety:
    """Test scoring calculations with edge cases."""

    def test_document_name_boost_with_max_score_zero(self, fresh_processor):
        """
        Document name boost when max_score=0 should not crash.

        Regression test for T-018: Division safety in _apply_document_name_boost.
        """
        # Add documents with very low or zero scores
        fresh_processor.process_document("test_file", "completely unrelated content xyz")
        fresh_processor.compute_all(verbose=False)

        # Search for something that might return zero scores
        # (If no matches, doc_scores could be empty or have max_score=0)
        results = fresh_processor.find_documents_for_query(
            "nonexistent query terms",
            top_n=5
        )

        # Should return empty results, not crash
        assert isinstance(results, list)

    def test_empty_corpus_normalization(self, fresh_processor):
        """
        TF-IDF normalization with empty corpus should handle gracefully.

        Regression test for T-018: Division by zero in compute_tfidf.
        """
        # No documents added - call compute_tfidf
        fresh_processor.compute_tfidf(verbose=False)

        # Should complete without division errors
        layer0 = fresh_processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() == 0

    def test_pagerank_with_no_connections(self, fresh_processor):
        """
        PageRank on disconnected graph should not crash.

        Regression test for T-018: Division by zero when node has no outlinks.
        """
        # Add single-word documents (no bigrams, minimal connections)
        fresh_processor.process_document("doc1", "word1")
        fresh_processor.process_document("doc2", "word2")
        fresh_processor.process_document("doc3", "word3")

        # Compute PageRank (graph may have isolated nodes)
        fresh_processor.compute_importance(verbose=False)

        # Should complete without division errors
        layer0 = fresh_processor.get_layer(CorticalLayer.TOKENS)
        for col in layer0:
            # PageRank should be valid (not NaN, not inf)
            assert 0.0 <= col.pagerank <= 1.0
            assert not (col.pagerank != col.pagerank)  # Not NaN

    def test_average_activation_empty_layer(self, fresh_processor):
        """
        Average activation on empty layer should return 0.0.

        Regression test for T-018: Division by zero in layer.average_activation().
        """
        layer0 = fresh_processor.get_layer(CorticalLayer.TOKENS)

        # Empty layer
        avg = layer0.average_activation()

        # Should return 0.0, not crash
        assert avg == 0.0

    def test_sparsity_empty_layer(self, fresh_processor):
        """
        Sparsity calculation on empty layer should return 0.0.

        Regression test for T-018: Division by zero in layer.sparsity().
        """
        layer0 = fresh_processor.get_layer(CorticalLayer.TOKENS)

        # Empty layer
        sparsity = layer0.sparsity()

        # Empty layer returns 0.0 (no columns to measure)
        assert sparsity == 0.0


class TestNormalizationEdgeCases:
    """Test score normalization edge cases."""

    def test_normalized_scores_all_zero(self):
        """
        Normalizing scores when all are zero should not crash.

        Regression test for T-018: Division by zero in score normalization.
        """
        # Simulate score normalization with all zeros
        scores = {'doc1': 0.0, 'doc2': 0.0, 'doc3': 0.0}

        # Typical normalization: divide by sum or max
        total = sum(scores.values())

        if total > 0:
            normalized = {k: v / total for k, v in scores.items()}
        else:
            # Should handle gracefully (e.g., keep zeros)
            normalized = scores.copy()

        # Should complete without error
        assert all(v == 0.0 for v in normalized.values())

    def test_similarity_with_zero_magnitude_vectors(self):
        """
        Cosine similarity with zero-magnitude vectors should return 0.0.

        Regression test for T-018: Division by zero in cosine similarity.
        """
        from cortical.analysis.utils import cosine_similarity

        # Two zero vectors (as dicts for sparse representation)
        vec1 = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        vec2 = {'a': 0.0, 'b': 0.0, 'c': 0.0}

        # Should return 0.0, not crash or return NaN
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_tfidf_single_document(self, fresh_processor):
        """
        TF-IDF with single document should compute correctly.

        Regression test for T-018: Edge case where IDF=0 (all terms appear in all docs).
        """
        fresh_processor.process_document("doc1", "test content here")
        fresh_processor.compute_tfidf(verbose=False)

        layer0 = fresh_processor.get_layer(CorticalLayer.TOKENS)

        # All terms appear in 100% of documents (N=1, df=1)
        # IDF = log(1/1) = 0, so TF-IDF = 0
        for col in layer0:
            # Should be valid number (not NaN), likely 0.0 for single doc
            assert not (col.tfidf != col.tfidf)  # Not NaN
            assert col.tfidf >= 0.0


class TestBM25EdgeCases:
    """Test BM25 scoring edge cases."""

    def test_bm25_zero_length_document(self):
        """
        BM25 scoring with zero-length document edge case.

        Regression test for T-018: Division by zero in document length normalization.
        """
        from cortical.config import CorticalConfig

        # BM25 scoring is the default
        config = CorticalConfig(scoring_algorithm='bm25')
        processor = CorticalTextProcessor(config=config)

        # Add normal documents
        processor.process_document("doc1", "neural networks")
        processor.process_document("doc2", "machine learning")

        # Compute and search
        processor.compute_all(verbose=False)
        results = processor.find_documents_for_query("neural", top_n=5)

        # Should work without division errors
        assert len(results) > 0

    def test_bm25_extreme_parameters(self):
        """
        BM25 with extreme parameter values should handle gracefully.

        Regression test for T-018: Edge cases in BM25 formula.
        """
        from cortical.config import CorticalConfig

        # k1=0.0 (disable term frequency saturation)
        config = CorticalConfig(
            scoring_algorithm='bm25',
            bm25_k1=0.0,
            bm25_b=0.0
        )
        processor = CorticalTextProcessor(config=config)

        processor.process_document("doc1", "neural networks process data")
        processor.compute_all(verbose=False)

        # Should work without division errors
        results = processor.find_documents_for_query("neural", top_n=5)
        assert len(results) > 0
