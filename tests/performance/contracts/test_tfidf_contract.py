"""
╔══════════════════════════════════════════════════════════════════════╗
║                      TF-IDF PERFORMANCE CONTRACT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • TF-IDF computation < 200ms for ≤ 1,000 terms                     ║
║  • BM25 computation < 300ms for ≤ 1,000 terms                       ║
║  • Rare terms score higher than common terms                        ║
║  • Scores are non-negative                                          ║
║  • Per-document scores are computed correctly                       ║
║  • Algorithm handles edge cases (empty docs, single term)           ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import math
import pytest


@pytest.mark.contract
class TestTFIDFPerformanceContract:
    """
    TF-IDF Performance Contract

    As a developer building search systems,
    I expect TF-IDF computation to be fast,
    So that indexing completes in reasonable time.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_LATENCY_MS = 200

    def test_tfidf_latency_honored(self, small_processor):
        """
        CONTRACT: TF-IDF computes in < 200ms for ≤ 1,000 terms.

        Fast term weighting is essential for responsive indexing.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.tfidf import compute_tfidf

        layer = small_processor.layers[CorticalLayer.TOKENS]

        # Verify we're within contract bounds
        assert layer.column_count() < 1000

        start = time.perf_counter()
        compute_tfidf(small_processor.layers, small_processor.documents)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: TF-IDF took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_tfidf_scores_non_negative(self, small_processor):
        """
        CONTRACT: TF-IDF scores are always non-negative.

        Negative scores are mathematically invalid.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.tfidf import compute_tfidf

        compute_tfidf(small_processor.layers, small_processor.documents)
        layer = small_processor.layers[CorticalLayer.TOKENS]

        for col in layer.minicolumns.values():
            assert col.tfidf >= 0, (
                f"CONTRACT VIOLATION: Term '{col.content}' has negative TF-IDF: {col.tfidf}"
            )

            # Per-document scores also non-negative
            for doc_id, score in col.tfidf_per_doc.items():
                assert score >= 0, (
                    f"CONTRACT VIOLATION: Term '{col.content}' has negative "
                    f"per-doc TF-IDF for {doc_id}: {score}"
                )


@pytest.mark.contract
class TestBM25PerformanceContract:
    """
    BM25 Performance Contract

    As a developer using modern ranking functions,
    I expect BM25 to compute efficiently,
    So that advanced ranking is practical.
    """

    MAX_LATENCY_MS = 300  # BM25 is more complex than TF-IDF

    def test_bm25_latency_honored(self, small_processor):
        """
        CONTRACT: BM25 computes in < 300ms for ≤ 1,000 terms.

        BM25 is more complex but must remain practical.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.tfidf import compute_bm25

        layer = small_processor.layers[CorticalLayer.TOKENS]
        assert layer.column_count() < 1000

        # Compute document lengths (required for BM25)
        doc_lengths = {}
        avg_length = 0
        for doc_id, content in small_processor.documents.items():
            doc_lengths[doc_id] = len(content.split())
            avg_length += doc_lengths[doc_id]
        avg_length /= len(small_processor.documents) if small_processor.documents else 1

        start = time.perf_counter()
        compute_bm25(
            small_processor.layers,
            small_processor.documents,
            doc_lengths,
            avg_length,
            k1=1.2,
            b=0.75
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: BM25 took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_bm25_scores_non_negative(self, small_processor):
        """
        CONTRACT: BM25 scores are always non-negative.

        The BM25 IDF variant uses +1 to ensure non-negative scores.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.tfidf import compute_bm25

        # Compute doc lengths
        doc_lengths = {doc_id: len(content.split())
                      for doc_id, content in small_processor.documents.items()}
        avg_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1

        compute_bm25(
            small_processor.layers,
            small_processor.documents,
            doc_lengths,
            avg_length
        )

        layer = small_processor.layers[CorticalLayer.TOKENS]

        for col in layer.minicolumns.values():
            assert col.tfidf >= 0, (
                f"CONTRACT VIOLATION: Term '{col.content}' has negative BM25: {col.tfidf}"
            )


@pytest.mark.contract
class TestTFIDFCorrectnessContract:
    """
    TF-IDF Correctness Contract

    As a developer relying on term weighting,
    I expect correct information retrieval properties,
    So that rare terms are properly emphasized.
    """

    def test_rare_terms_score_higher_than_common(self):
        """
        CONTRACT: Rare terms have higher TF-IDF than common terms.

        This is the fundamental property of TF-IDF.
        """
        from cortical.analysis.tfidf import _tfidf_core

        # "rare" appears in 1 doc, "common" in all 10 docs
        term_stats = {
            "rare": (5, 1, {"doc1": 5}),
            "common": (100, 10, {f"doc{i}": 10 for i in range(1, 11)})
        }

        results = _tfidf_core(term_stats, num_docs=10)

        rare_tfidf = results["rare"][0]
        common_tfidf = results["common"][0]

        assert rare_tfidf > common_tfidf, (
            f"CONTRACT VIOLATION: Rare term ({rare_tfidf:.3f}) should score "
            f"higher than common term ({common_tfidf:.3f})"
        )

    def test_tfidf_handles_empty_corpus(self):
        """
        CONTRACT: TF-IDF handles empty corpus gracefully.

        Edge cases should not crash the system.
        """
        from cortical.analysis.tfidf import _tfidf_core

        results = _tfidf_core({}, num_docs=0)
        assert results == {}

    def test_tfidf_core_per_document_scores(self):
        """
        CONTRACT: Per-document TF-IDF scores are computed correctly.

        Documents with higher term frequency should have higher scores.
        """
        from cortical.analysis.tfidf import _tfidf_core

        term_stats = {
            "test": (15, 2, {"doc1": 10, "doc2": 5})  # 10 occurrences in doc1, 5 in doc2
        }

        results = _tfidf_core(term_stats, num_docs=10)
        global_tfidf, per_doc_tfidf = results["test"]

        # doc1 has more occurrences, should have higher per-doc TF-IDF
        assert per_doc_tfidf["doc1"] > per_doc_tfidf["doc2"], (
            "CONTRACT VIOLATION: Document with more term occurrences should score higher"
        )

    def test_tfidf_idf_formula_correct(self):
        """
        CONTRACT: IDF formula is log(N / df).

        This is the standard IDF formula from information retrieval.
        """
        from cortical.analysis.tfidf import _tfidf_core

        # Term appears in 5 out of 10 documents
        term_stats = {
            "term": (50, 5, {f"doc{i}": 10 for i in range(5)})
        }

        results = _tfidf_core(term_stats, num_docs=10)

        # IDF should be log(10/5) = log(2) ≈ 0.693
        expected_idf = math.log(10 / 5)

        # TF-IDF = TF * IDF where TF = log1p(50) ≈ 3.93
        global_tfidf = results["term"][0]
        expected_tf = math.log1p(50)
        expected_tfidf = expected_tf * expected_idf

        # Allow small floating point tolerance
        assert abs(global_tfidf - expected_tfidf) < 0.01, (
            f"CONTRACT VIOLATION: TF-IDF formula incorrect. "
            f"Got {global_tfidf:.3f}, expected {expected_tfidf:.3f}"
        )


@pytest.mark.contract
class TestBM25CorrectnessContract:
    """
    BM25 Correctness Contract

    As a developer using modern ranking,
    I expect correct BM25 properties,
    So that length normalization works properly.
    """

    def test_bm25_rare_terms_score_higher(self):
        """
        CONTRACT: BM25 rare terms score higher than common terms.

        Same fundamental property as TF-IDF.
        """
        from cortical.analysis.tfidf import _bm25_core

        term_stats = {
            "rare": (5, 1, {"doc1": 5}),
            "common": (100, 10, {f"doc{i}": 10 for i in range(1, 11)})
        }

        doc_lengths = {f"doc{i}": 100 for i in range(1, 11)}
        avg_length = 100.0

        results = _bm25_core(term_stats, num_docs=10, doc_lengths=doc_lengths,
                            avg_doc_length=avg_length, k1=1.2, b=0.75)

        rare_bm25 = results["rare"][0]
        common_bm25 = results["common"][0]

        assert rare_bm25 > common_bm25, (
            f"CONTRACT VIOLATION: Rare term BM25 ({rare_bm25:.3f}) should be "
            f"higher than common term ({common_bm25:.3f})"
        )

    def test_bm25_length_normalization(self):
        """
        CONTRACT: BM25 normalizes by document length.

        Same term frequency in longer document should score lower.
        """
        from cortical.analysis.tfidf import _bm25_core

        # Same term frequency (5) in short doc (50 words) vs long doc (200 words)
        term_stats = {
            "term": (10, 2, {"short": 5, "long": 5})
        }

        doc_lengths = {"short": 50, "long": 200}
        avg_length = 125.0

        results = _bm25_core(term_stats, num_docs=2, doc_lengths=doc_lengths,
                            avg_doc_length=avg_length, k1=1.2, b=0.75)

        _, per_doc_bm25 = results["term"]

        # Short document should score higher (length normalization)
        # because same TF in shorter doc is more significant
        assert per_doc_bm25["short"] > per_doc_bm25["long"], (
            "CONTRACT VIOLATION: BM25 length normalization not working. "
            f"Short doc: {per_doc_bm25['short']:.3f}, Long doc: {per_doc_bm25['long']:.3f}"
        )

    def test_bm25_term_frequency_saturation(self):
        """
        CONTRACT: BM25 exhibits term frequency saturation.

        Doubling TF should not double the score (diminishing returns).
        """
        from cortical.analysis.tfidf import _bm25_core

        term_stats = {
            "term": (15, 2, {"doc1": 5, "doc2": 10})  # doc2 has double the TF
        }

        doc_lengths = {"doc1": 100, "doc2": 100}
        avg_length = 100.0

        results = _bm25_core(term_stats, num_docs=2, doc_lengths=doc_lengths,
                            avg_doc_length=avg_length, k1=1.2, b=0.75)

        _, per_doc_bm25 = results["term"]

        score_5 = per_doc_bm25["doc1"]
        score_10 = per_doc_bm25["doc2"]

        # Score for TF=10 should be less than double the score for TF=5
        # (saturation effect)
        assert score_10 < 2 * score_5, (
            f"CONTRACT VIOLATION: BM25 should exhibit saturation. "
            f"TF=5: {score_5:.3f}, TF=10: {score_10:.3f}, ratio: {score_10/score_5:.2f}"
        )

        # But should still be higher
        assert score_10 > score_5, (
            "Higher TF should still produce higher score"
        )
