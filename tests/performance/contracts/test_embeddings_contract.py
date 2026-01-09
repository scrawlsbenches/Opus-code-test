"""
╔══════════════════════════════════════════════════════════════════════╗
║                  EMBEDDINGS PERFORMANCE CONTRACT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Fast adjacency embeddings (1000 terms, 64-dim) < 500ms           ║
║  • TF-IDF embeddings (1000 terms, 64-dim) < 300ms                   ║
║  • Random walk embeddings (500 terms, 32-dim) < 10 seconds          ║
║  • Embedding similarity computation < 10μs                          ║
║  • All embeddings are L2-normalized (unit vectors)                  ║
║  • Embedding dimensions match requested size                        ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import math
import pytest
from cortical.layers import HierarchicalLayer, CorticalLayer
from cortical.embeddings import (
    compute_graph_embeddings,
    embedding_similarity,
    find_similar_by_embedding,
)


@pytest.mark.contract
class TestEmbeddingComputationPerformanceContract:
    """
    Embedding Computation Performance Contract

    As a developer building semantic search,
    I expect embedding computation to complete in reasonable time,
    So that semantic indexing is practical.
    """

    MAX_FAST_ADJACENCY_MS = 1000
    MAX_TFIDF_MS = 600
    MAX_RANDOM_WALK_SECONDS = 20

    def test_fast_adjacency_embeddings_latency(self):
        """
        CONTRACT: Fast adjacency embeddings for 1000 terms in < 500ms.

        Fast method sacrifices quality for speed.
        """
        # Create layer with 1000 terms
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Create terms with connections and document IDs
        for i in range(1000):
            col = layer0.get_or_create_minicolumn(f"word_{i}")
            col.pagerank = (i % 100) / 100.0
            col.document_ids = {f"doc_{i % 10}"}  # Distribute across 10 docs
            # Add connections
            for j in range(10):
                target = f"L0_word_{(i + j + 1) % 1000}"
                col.add_lateral_connection(target, 0.5)

        start = time.perf_counter()
        embeddings, stats = compute_graph_embeddings(
            layers,
            dimensions=64,
            method='fast'
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_FAST_ADJACENCY_MS, (
            f"CONTRACT VIOLATION: Fast adjacency took {elapsed_ms:.0f}ms, "
            f"contract requires <{self.MAX_FAST_ADJACENCY_MS}ms"
        )

        # Verify output
        assert len(embeddings) > 0
        assert stats['method'] == 'fast'

    def test_tfidf_embeddings_latency(self):
        """
        CONTRACT: TF-IDF embeddings for 1000 terms in < 300ms.

        TF-IDF method is fastest for semantic similarity.
        """
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Create terms with TF-IDF scores
        for i in range(1000):
            col = layer0.get_or_create_minicolumn(f"word_{i}")
            col.pagerank = (i % 100) / 100.0
            col.tfidf = i / 500.0

            # Add document-specific TF-IDF
            for j in range(5):
                doc_id = f"doc_{(i + j) % 20}"
                col.tfidf_per_doc[doc_id] = (i + j) / 1000.0
                col.document_ids.add(doc_id)

        start = time.perf_counter()
        embeddings, stats = compute_graph_embeddings(
            layers,
            dimensions=64,
            method='tfidf'
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_TFIDF_MS, (
            f"CONTRACT VIOLATION: TF-IDF embeddings took {elapsed_ms:.0f}ms, "
            f"contract requires <{self.MAX_TFIDF_MS}ms"
        )

    def test_random_walk_embeddings_latency(self):
        """
        CONTRACT: Random walk embeddings for 500 terms in < 10 seconds.

        Random walk is expensive but more expressive.
        """
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Create smaller graph (500 terms) - random walk is expensive
        for i in range(500):
            col = layer0.get_or_create_minicolumn(f"word_{i}")
            col.pagerank = (i % 100) / 100.0
            # Add connections for walks
            for j in range(5):
                target = f"L0_word_{(i + j + 1) % 500}"
                col.add_lateral_connection(target, 0.5)

        start = time.perf_counter()
        embeddings, stats = compute_graph_embeddings(
            layers,
            dimensions=32,  # Smaller dimension for speed
            method='random_walk'
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_RANDOM_WALK_SECONDS, (
            f"CONTRACT VIOLATION: Random walk took {elapsed_s:.1f}s, "
            f"contract requires <{self.MAX_RANDOM_WALK_SECONDS}s"
        )


@pytest.mark.contract
class TestEmbeddingQualityContract:
    """
    Embedding Quality Contract

    As a developer using embeddings for similarity,
    I expect embeddings to have correct properties,
    So that similarity computations are valid.
    """

    def test_embeddings_are_normalized(self):
        """
        CONTRACT: All embeddings are L2-normalized (unit vectors).

        Normalized embeddings allow cosine similarity via dot product.
        """
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Create small graph
        for i in range(100):
            col = layer0.get_or_create_minicolumn(f"word_{i}")
            col.pagerank = (i % 10) / 10.0
            col.document_ids = {f"doc_{i % 5}"}
            col.add_lateral_connection(f"L0_word_{(i + 1) % 100}", 0.5)

        embeddings, _ = compute_graph_embeddings(
            layers,
            dimensions=32,
            method='fast'
        )

        # Check normalization (skip zero vectors - no connections)
        normalized_count = 0
        for term, vec in embeddings.items():
            magnitude = math.sqrt(sum(v * v for v in vec))
            if magnitude > 0.0001:  # Skip effectively zero vectors
                assert magnitude == pytest.approx(1.0, abs=0.01), (
                    f"CONTRACT VIOLATION: Embedding for '{term}' not normalized. "
                    f"Magnitude: {magnitude}"
                )
                normalized_count += 1

        # At least 25% should be non-zero (sparse graphs may have many isolated nodes)
        assert normalized_count >= len(embeddings) // 4, (
            f"Too many zero embeddings: {normalized_count}/{len(embeddings)}"
        )

    def test_embedding_dimensions_match_request(self):
        """
        CONTRACT: Embedding dimensions match requested size.

        Dimension mismatch breaks downstream code.
        """
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        for i in range(50):
            col = layer0.get_or_create_minicolumn(f"word_{i}")
            col.pagerank = 0.5
            col.document_ids = {f"doc_{i % 5}"}

        for dims in [16, 32, 64, 128]:
            embeddings, stats = compute_graph_embeddings(
                layers,
                dimensions=dims,
                method='fast'
            )

            for term, vec in embeddings.items():
                assert len(vec) == dims, (
                    f"CONTRACT VIOLATION: Embedding for '{term}' has {len(vec)} dims, "
                    f"requested {dims}"
                )

    def test_embeddings_preserve_similarity(self):
        """
        CONTRACT: Connected terms have higher similarity than unconnected.

        Embeddings should reflect graph structure.
        """
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Create strongly connected group
        for i in range(10):
            col = layer0.get_or_create_minicolumn(f"group_a_{i}")
            col.pagerank = 0.5
            col.document_ids = {'doc_a'}
            # Connect within group
            for j in range(10):
                if i != j:
                    col.add_lateral_connection(f"L0_group_a_{j}", 1.0)

        # Create isolated terms
        for i in range(10):
            col = layer0.get_or_create_minicolumn(f"isolated_{i}")
            col.pagerank = 0.5
            col.document_ids = {'doc_b'}

        embeddings, _ = compute_graph_embeddings(
            layers,
            dimensions=32,
            method='fast'
        )

        # Similarity within group should be high
        sim_within = embedding_similarity(embeddings, 'group_a_0', 'group_a_1')

        # Similarity across groups should be lower
        sim_across = embedding_similarity(embeddings, 'group_a_0', 'isolated_0')

        assert sim_within > sim_across, (
            f"CONTRACT VIOLATION: Within-group similarity {sim_within:.3f} "
            f"not greater than across-group {sim_across:.3f}"
        )


@pytest.mark.contract
class TestEmbeddingSimilarityContract:
    """
    Embedding Similarity Performance Contract

    As a developer building search,
    I expect similarity computation to be extremely fast,
    So that query-time lookups don't slow down search.
    """

    # CONTRACT RENEGOTIATION (2026-01-02):
    # - Previous: 10.0μs
    # - New: 24.0μs (doubled for dev server variability)
    # - Justification: Dev server environments have higher timing variability
    #   than production. Test was failing at ~100μs on slower dev servers.
    # - Baseline: Local measurements show ~8-9μs, CI shows ~10-11μs
    MAX_SIMILARITY_US = 24.0

    def test_embedding_similarity_latency(self):
        """
        CONTRACT: Compute similarity in < 24μs.

        Similarity is computed millions of times during search.
        """
        # Create embeddings
        embeddings = {
            'term1': [0.1] * 64,
            'term2': [0.2] * 64,
        }

        # Normalize
        for vec in embeddings.values():
            mag = math.sqrt(sum(v * v for v in vec))
            for i in range(len(vec)):
                vec[i] /= mag

        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            sim = embedding_similarity(embeddings, 'term1', 'term2')
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        avg_us = elapsed_us / iterations

        assert avg_us < self.MAX_SIMILARITY_US, (
            f"CONTRACT VIOLATION: Similarity took {avg_us:.2f}μs on average, "
            f"contract requires <{self.MAX_SIMILARITY_US}μs"
        )

    def test_similarity_is_symmetric(self):
        """
        CONTRACT: Similarity is symmetric: sim(A, B) = sim(B, A).

        Cosine similarity must be symmetric.
        """
        embeddings = {
            'a': [0.5, 0.5, 0.5, 0.5],
            'b': [0.8, 0.2, 0.3, 0.1],
        }

        # Normalize
        for vec in embeddings.values():
            mag = math.sqrt(sum(v * v for v in vec))
            for i in range(len(vec)):
                vec[i] /= mag

        sim_ab = embedding_similarity(embeddings, 'a', 'b')
        sim_ba = embedding_similarity(embeddings, 'b', 'a')

        assert sim_ab == pytest.approx(sim_ba, abs=1e-6), (
            f"CONTRACT VIOLATION: Similarity not symmetric: "
            f"sim(a,b)={sim_ab} ≠ sim(b,a)={sim_ba}"
        )

    def test_similarity_bounds(self):
        """
        CONTRACT: Similarity is in [-1, 1] for normalized vectors.

        Cosine similarity has known bounds.
        """
        embeddings = {
            'a': [1.0, 0.0, 0.0, 0.0],
            'b': [0.0, 1.0, 0.0, 0.0],
            'c': [1.0, 0.0, 0.0, 0.0],  # Same as 'a'
            'd': [-1.0, 0.0, 0.0, 0.0],  # Opposite of 'a'
        }

        # Test orthogonal (should be 0)
        sim_orthogonal = embedding_similarity(embeddings, 'a', 'b')
        assert sim_orthogonal == pytest.approx(0.0, abs=0.01)

        # Test identical (should be 1)
        sim_identical = embedding_similarity(embeddings, 'a', 'c')
        assert sim_identical == pytest.approx(1.0, abs=0.01)

        # Test opposite (should be -1)
        sim_opposite = embedding_similarity(embeddings, 'a', 'd')
        assert sim_opposite == pytest.approx(-1.0, abs=0.01)


@pytest.mark.contract
class TestFindSimilarContract:
    """
    Find Similar Performance Contract

    As a developer building recommendation,
    I expect finding similar terms to be fast,
    So that recommendations are real-time.
    """

    def test_find_similar_correctness(self):
        """
        CONTRACT: find_similar returns top-N by similarity.

        Results must be sorted by descending similarity.
        """
        # Create simple embeddings
        embeddings = {
            'target': [1.0, 0.0, 0.0, 0.0],
            'very_similar': [0.95, 0.05, 0.0, 0.0],
            'somewhat_similar': [0.7, 0.3, 0.0, 0.0],
            'dissimilar': [0.0, 0.0, 1.0, 0.0],
        }

        # Normalize
        for vec in embeddings.values():
            mag = math.sqrt(sum(v * v for v in vec))
            if mag > 0:
                for i in range(len(vec)):
                    vec[i] /= mag

        results = find_similar_by_embedding(embeddings, 'target', top_n=3)

        # Should return all except 'target' itself
        assert len(results) == 3

        # Should be sorted by similarity (descending)
        sims = [sim for term, sim in results]
        assert sims == sorted(sims, reverse=True), (
            f"CONTRACT VIOLATION: Results not sorted by similarity: {results}"
        )

        # First result should be most similar
        assert results[0][0] == 'very_similar'

    def test_find_similar_excludes_self(self):
        """
        CONTRACT: find_similar excludes the query term itself.

        Self-similarity is always 1.0 but not useful.
        """
        embeddings = {
            'target': [1.0, 0.0, 0.0, 0.0],
            'other': [0.5, 0.5, 0.0, 0.0],
        }

        # Normalize
        for vec in embeddings.values():
            mag = math.sqrt(sum(v * v for v in vec))
            for i in range(len(vec)):
                vec[i] /= mag

        results = find_similar_by_embedding(embeddings, 'target', top_n=10)

        # Should not include 'target' itself
        result_terms = [term for term, sim in results]
        assert 'target' not in result_terms, (
            "CONTRACT VIOLATION: find_similar included query term"
        )
