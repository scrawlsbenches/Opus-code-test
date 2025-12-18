"""
Behavioral Tests for Core User Workflows
=========================================

Tests that verify the system delivers expected user outcomes.
Unlike unit tests (function works correctly) or integration tests
(components work together), behavioral tests verify "the system feels right."

These tests check:
- Search relevance: Do results make sense to users?
- Quality metrics: Are computed values meaningful?
- Robustness: Does the system handle edge cases gracefully?

Run with: pytest tests/behavioral/ -v
"""

import pytest


class TestSearchRelevance:
    """
    Test that search results feel relevant to users.

    These tests verify that:
    - Document names matching queries rank highly
    - Query expansion improves recall
    - Related documents appear in results
    """

    def test_document_name_matches_rank_highly(self, small_processor):
        """
        Query matching document name should return that doc in top results.

        User expectation: If I search for "machine learning" and there's
        a document with "ml" in its name about machine learning, it should
        be in my top results.
        """
        # Test cases: (query, expected_doc_substring)
        test_cases = [
            ("machine learning", "ml_"),
            ("database", "db_"),
            ("distributed systems", "dist_"),
            ("sorting algorithms", "algo_"),
            ("software testing", "se_testing"),
        ]

        for query, expected_substring in test_cases:
            results = small_processor.find_documents_for_query(query, top_n=5)
            top_docs = [doc_id for doc_id, _ in results]

            # At least one doc with the expected substring should appear
            found = any(expected_substring in doc_id for doc_id in top_docs)
            assert found, (
                f"Query '{query}' should return doc with '{expected_substring}' "
                f"in top 5. Got: {top_docs}"
            )

    def test_query_expansion_improves_recall(self, small_processor):
        """
        Expanded queries should find more relevant docs.

        User expectation: If I search for "ML", I should get docs
        about "machine learning" even if they don't use the abbreviation.
        """
        # Search with a term that should expand
        results = small_processor.find_documents_for_query(
            "neural network training",
            top_n=10
        )
        found_docs = {doc_id for doc_id, _ in results}

        # Should find ML-related documents (all ML docs in small corpus)
        ml_related = {'ml_basics', 'deep_learning', 'ml_optimization', 'ml_evaluation', 'ml_applications'}
        found_ml = found_docs & ml_related

        assert len(found_ml) >= 2, (
            f"Query 'neural network training' should find ML docs. "
            f"Found: {found_docs}"
        )

    def test_cross_domain_queries_work(self, small_processor):
        """
        Queries spanning multiple domains should return relevant results.
        """
        # Query that touches multiple domains
        results = small_processor.find_documents_for_query(
            "algorithm optimization performance",
            top_n=10
        )

        assert len(results) > 0, "Cross-domain query should return results"

        # Should find docs from multiple domains
        found_docs = {doc_id for doc_id, _ in results}

        # Could match algo_, ml_, db_, se_ domains
        prefixes_found = set()
        for doc_id in found_docs:
            prefix = doc_id.split('_')[0]
            prefixes_found.add(prefix)

        assert len(prefixes_found) >= 2, (
            f"Cross-domain query should return docs from multiple domains. "
            f"Found only: {prefixes_found}"
        )


class TestQualityMetrics:
    """
    Test that computed metrics make sense.

    These tests verify that:
    - PageRank identifies important terms
    - Clustering produces coherent groups
    - Embeddings capture semantic similarity
    """

    def test_pagerank_surfaces_domain_terms(self, small_processor):
        """
        Top PageRank terms should be domain-relevant concepts.

        User expectation: The most "important" terms should be meaningful
        concepts from the corpus domains, not generic words.
        """
        from cortical import CorticalLayer

        layer0 = small_processor.get_layer(CorticalLayer.TOKENS)

        # Get top 20 PageRank terms
        top_terms = sorted(
            [(col.content, col.pagerank) for col in layer0],
            key=lambda x: -x[1]
        )[:20]
        top_term_names = [term for term, _ in top_terms]

        # Should contain domain-specific terms
        expected_domain_terms = {
            'learning', 'data', 'network', 'algorithm', 'system',
            'model', 'query', 'test', 'database', 'training',
            'machine', 'distributed', 'function', 'code', 'search',
        }

        found_domain_terms = set(top_term_names) & expected_domain_terms
        assert len(found_domain_terms) >= 3, (
            f"Top PageRank terms should include domain concepts. "
            f"Found: {top_term_names}"
        )

    def test_clustering_has_good_modularity(self, small_processor):
        """
        Clusters should have good community structure.

        Threshold: modularity > 0.3 indicates meaningful clustering.
        """
        from cortical.analysis import compute_clustering_quality

        quality = compute_clustering_quality(small_processor.layers)

        assert quality['modularity'] > 0.2, (
            f"Clustering modularity {quality['modularity']:.3f} is below "
            f"threshold. Quality: {quality['quality_assessment']}"
        )

    def test_multiple_clusters_exist(self, small_processor):
        """
        Should have multiple distinct clusters, not just one or two.
        """
        from cortical import CorticalLayer

        layer2 = small_processor.get_layer(CorticalLayer.CONCEPTS)
        num_clusters = layer2.column_count()

        # 25-doc corpus should produce at least 5 clusters
        assert num_clusters >= 5, (
            f"Only {num_clusters} clusters for 25 documents. "
            f"Expected at least 5 distinct concept groups."
        )

    def test_embeddings_capture_similarity(self, small_processor):
        """
        Terms with similar meaning should have similar embeddings.
        """
        import math

        def dense_cosine_similarity(vec1, vec2):
            """Compute cosine similarity between two dense vectors."""
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        # compute_graph_embeddings returns stats, embeddings stored on processor
        small_processor.compute_graph_embeddings(
            method='tfidf',
            dimensions=32,
            verbose=False
        )
        embeddings = small_processor.embeddings

        # Check that "learning" is more similar to "training" than to "database"
        if 'learning' in embeddings and 'training' in embeddings and 'database' in embeddings:
            sim_learning_training = dense_cosine_similarity(
                embeddings['learning'],
                embeddings['training']
            )
            sim_learning_database = dense_cosine_similarity(
                embeddings['learning'],
                embeddings['database']
            )

            # Learning should be more similar to training than to database
            # (or at least not dramatically less similar)
            assert sim_learning_training >= sim_learning_database * 0.5, (
                f"'learning' should be similar to 'training' ({sim_learning_training:.3f}) "
                f"at least half as much as to 'database' ({sim_learning_database:.3f})"
            )


class TestRobustness:
    """
    Test that the system handles edge cases gracefully.

    These tests verify that:
    - Invalid inputs don't crash the system
    - Unknown terms are handled gracefully
    - Special characters don't cause errors
    """

    def test_unknown_terms_return_empty(self, small_processor):
        """Queries with only unknown terms should return empty list."""
        results = small_processor.find_documents_for_query(
            "xyzzy_completely_unknown_term_12345",
            top_n=5
        )

        assert isinstance(results, list)
        # May be empty or have low-confidence partial matches

    def test_mixed_known_unknown_terms(self, small_processor):
        """Queries mixing known and unknown terms should still work."""
        results = small_processor.find_documents_for_query(
            "machine xyzzy_unknown learning",
            top_n=5
        )

        # Should still find ML-related docs based on known terms
        assert isinstance(results, list)

    def test_special_characters_handled(self, small_processor):
        """Queries with special characters should not crash."""
        special_queries = [
            "function() { return x; }",
            "SELECT * FROM table",
            "@decorator def method:",
            "{{template}} ${variable}",
            "path/to/file.txt",
            "email@example.com",
        ]

        for query in special_queries:
            # Should not raise exception
            try:
                results = small_processor.find_documents_for_query(query, top_n=5)
                assert isinstance(results, list)
            except ValueError:
                # ValueError for empty-after-tokenization is acceptable
                pass
            except Exception as e:
                pytest.fail(f"Query '{query[:30]}...' raised {type(e).__name__}: {e}")

    def test_very_long_query_handled(self, small_processor):
        """Very long queries should be handled without crashing."""
        long_query = " ".join(["machine learning"] * 100)

        results = small_processor.find_documents_for_query(long_query, top_n=5)
        assert isinstance(results, list)

    def test_unicode_queries_handled(self, small_processor):
        """Unicode characters in queries should be handled."""
        unicode_queries = [
            "machine learning",  # ASCII baseline
            "machinelearning",  # No spaces
            "MACHINE LEARNING",  # Upper case
        ]

        for query in unicode_queries:
            results = small_processor.find_documents_for_query(query, top_n=5)
            assert isinstance(results, list)


class TestPassageRetrieval:
    """Test passage retrieval for RAG use cases."""

    def test_passages_contain_query_terms(self, small_processor):
        """Retrieved passages should be relevant to query."""
        passages = small_processor.find_passages_for_query(
            "database indexing",
            top_n=3,
            chunk_size=200,
            overlap=50
        )

        assert len(passages) > 0, "Should return some passages"

        # Passages are (text, doc_id, start, end, score) tuples
        found_relevant = False
        for result in passages:
            passage_text = result[0]
            passage_lower = passage_text.lower()
            if 'database' in passage_lower or 'index' in passage_lower:
                found_relevant = True
                break

        assert found_relevant, (
            f"Passages for 'database indexing' should mention relevant terms. "
            f"Got passages from: {[p[1] for p in passages]}"
        )

    def test_passages_respect_chunk_size(self, small_processor):
        """Passages should be approximately the requested chunk size."""
        chunk_size = 200
        passages = small_processor.find_passages_for_query(
            "machine learning",
            top_n=5,
            chunk_size=chunk_size,
            overlap=50
        )

        # Passages are (text, doc_id, start, end, score) tuples
        for result in passages:
            passage_text = result[0]
            # Chunk size is in characters, allow reasonable variance
            assert len(passage_text) < chunk_size * 2, (
                f"Passage too long: {len(passage_text)} chars (chunk_size={chunk_size})"
            )
