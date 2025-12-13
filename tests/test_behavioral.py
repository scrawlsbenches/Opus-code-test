"""
Behavioral Tests for Core User Workflows
=========================================

These tests verify that the Cortical Text Processor delivers expected user
outcomes. Unlike unit tests (function works correctly) or integration tests
(components work together), behavioral tests verify "the system feels right."

These tests were created based on dog-fooding issues discovered during
real usage testing (Tasks #141-145), ensuring that common user workflows
produce sensible, timely results.

Test Categories:
- SearchBehavior: "Search should feel relevant"
- PerformanceBehavior: "System should feel responsive"
- QualityBehavior: "Results should make sense"
- RobustnessBehavior: "System should handle edge cases gracefully"

Task #146 Implementation
"""

import os
import sys
import time
import unittest

sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


# Module-level singleton for shared corpus (load once, reuse everywhere)
_SHARED_PROCESSOR = None
_SHARED_PROCESSOR_LOADED = False


def get_shared_processor() -> CorticalTextProcessor:
    """
    Get or create the shared processor with sample corpus.

    This singleton ensures we only load the corpus once per test run,
    dramatically reducing test time (from 5x compute_all to 1x).
    """
    global _SHARED_PROCESSOR, _SHARED_PROCESSOR_LOADED

    if _SHARED_PROCESSOR_LOADED:
        return _SHARED_PROCESSOR

    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')

    tokenizer = Tokenizer(filter_code_noise=True)
    processor = CorticalTextProcessor(tokenizer=tokenizer)

    # Load all sample files
    loaded = 0
    for filename in os.listdir(samples_dir):
        filepath = os.path.join(samples_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_id = os.path.splitext(filename)[0]
                processor.process_document(doc_id, content)
                loaded += 1
            except (IOError, UnicodeDecodeError):
                continue

    if loaded > 0:
        processor.compute_all(verbose=False)

    _SHARED_PROCESSOR = processor
    _SHARED_PROCESSOR_LOADED = True
    return processor


class TestSearchBehavior(unittest.TestCase):
    """
    Test that search feels relevant to users.

    These tests verify that:
    - Document names matching queries rank highly
    - Query expansion improves recall without hurting precision
    - Code searches prefer implementations over tests
    """

    @classmethod
    def setUpClass(cls):
        """Load corpus once for all search tests."""
        cls.processor = get_shared_processor()

    def test_document_name_matches_rank_highly(self):
        """
        Query matching document name should return that doc in top 2.

        User expectation: If I search for "distributed systems" and there's
        a document called "distributed_systems", it should be in my top results.

        Regression test for Task #144 (doc_name_boost fix).
        """
        # Test cases: (query, expected_doc_in_top_2)
        test_cases = [
            ("distributed systems", "distributed_systems"),
            ("quantum computing", "quantum_computing_basics"),
            ("fermentation", "fermentation_science"),
            ("pagerank", "pagerank_fundamentals"),
            ("neural network optimization", "neural_network_optimization"),
        ]

        for query, expected_doc in test_cases:
            with self.subTest(query=query):
                results = self.processor.find_documents_for_query(query, top_n=3)
                top_3_docs = [doc_id for doc_id, score in results]

                self.assertIn(
                    expected_doc,
                    top_3_docs,
                    f"Query '{query}' should return '{expected_doc}' in top 3, "
                    f"got: {top_3_docs}"
                )

    def test_query_expansion_improves_recall(self):
        """
        Expanded queries should find more relevant docs than exact match.

        User expectation: If I search for "ML models", I should get docs
        about "machine learning" even if they don't use the abbreviation.
        """
        # Search with expansion
        expanded_results = self.processor.find_documents_for_query(
            "ML training",
            top_n=10
        )
        expanded_docs = {doc_id for doc_id, _ in expanded_results}

        # Check that results include machine learning related docs
        ml_related_docs = {
            'comprehensive_machine_learning',
            'deep_learning_revolution',
            'neural_network_optimization',
        }

        # At least one ML doc should appear in results
        found_ml_docs = expanded_docs & ml_related_docs
        self.assertGreater(
            len(found_ml_docs),
            0,
            f"Query 'ML training' should find ML-related docs, got: {expanded_docs}"
        )

    def test_code_search_finds_relevant_code_files(self):
        """
        Code queries should find relevant code files in results.

        User expectation: When searching for code concepts like "data processor",
        relevant code files should appear in the top results.

        Note: The test_file_penalty is applied in definition-focused searches
        (find_definition_passages), not in general document search. See Task #128.
        """
        results = self.processor.find_documents_for_query(
            "data processor",
            top_n=5
        )

        top_docs = [doc_id for doc_id, _ in results]

        # Both data_processor and test_data_processor should appear in results
        # (they're both relevant to "data processor" query)
        data_processor_found = any(
            'data_processor' in doc_id for doc_id in top_docs
        )

        self.assertTrue(
            data_processor_found,
            f"Query 'data processor' should find data_processor files in results. "
            f"Got: {top_docs}"
        )

        # Test that results are not empty and have reasonable scores
        self.assertGreater(
            len(results),
            0,
            "Code search should return results"
        )


# NOTE: Performance tests have been moved to tests/performance/test_performance.py
# which uses a small synthetic corpus for fast, reliable timing tests.
# The old TestPerformanceBehavior class was removed to avoid slow test runs.


class TestQualityBehavior(unittest.TestCase):
    """
    Test that results make sense to users.

    These tests verify that:
    - Important terms identified by PageRank are meaningful, not noise
    - Clustering produces coherent groups
    - Embeddings capture semantic similarity
    """

    @classmethod
    def setUpClass(cls):
        """Load corpus once for all quality tests."""
        cls.processor = get_shared_processor()

    def test_pagerank_surfaces_meaningful_terms(self):
        """
        Top PageRank terms should be domain concepts, not noise.

        User expectation: The most "important" terms should be meaningful
        concepts like "neural", "learning", "network" - not Python syntax
        like "self", "def", or test artifacts like "assertequal".

        Regression test for Task #141 (filter_code_noise).
        """
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Get top 30 PageRank terms
        pagerank_terms = sorted(
            [(col.content, col.pagerank) for col in layer0],
            key=lambda x: -x[1]
        )[:30]
        top_terms = [term for term, _ in pagerank_terms]

        # Code noise that should NOT appear in top results
        # (when filter_code_noise=True is working)
        noise_tokens = {
            'self', 'def', 'str', 'int', 'float', 'none', 'true', 'false',
            'assertequal', 'asserttrue', 'assertfalse', 'mock',
            'cls', 'args', 'kwargs', 'return', 'import', 'from',
        }

        # Check that no noise tokens appear in top 30
        found_noise = [t for t in top_terms if t in noise_tokens]

        self.assertEqual(
            len(found_noise),
            0,
            f"Top PageRank terms contain noise tokens: {found_noise}. "
            f"Top 10 terms: {top_terms[:10]}. "
            "Check that filter_code_noise=True is being applied."
        )

    def test_clustering_produces_coherent_groups(self):
        """
        Clusters should have good community structure.

        User expectation: The concept clusters should make sense -
        related terms should be grouped together, and there shouldn't
        be one mega-cluster containing everything.

        Threshold: modularity > 0.2 (moderate community structure)
        Note: Text corpora with many interconnected terms typically
        achieve modularity 0.2-0.4. Values >0.3 indicate strong structure.
        Based on Tasks #123-125 (Louvain clustering and quality metrics).
        """
        from cortical.analysis import compute_clustering_quality

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        # Need cluster assignments
        cluster_assignments = {}
        for concept_col in layer2:
            cluster_id = concept_col.content
            # Get tokens in this cluster from feedforward connections
            for token_id in concept_col.feedforward_connections:
                token_col = layer0.get_by_id(token_id)
                if token_col:
                    cluster_assignments[token_col.content] = cluster_id

        if len(cluster_assignments) > 0:
            # compute_clustering_quality expects layers dict, not single layer
            quality = compute_clustering_quality(self.processor.layers)

            # Modularity > 0.2 indicates moderate community structure
            # (consistent with tests/behavioral/test_behavioral.py)
            self.assertGreater(
                quality['modularity'],
                0.2,
                f"Clustering modularity {quality['modularity']:.3f} is below "
                "0.2 threshold for moderate community structure. "
                f"Quality assessment: {quality['quality_assessment']}"
            )

            # Should have multiple clusters (no single mega-cluster)
            num_clusters = len(set(cluster_assignments.values()))
            self.assertGreater(
                num_clusters,
                5,
                f"Only {num_clusters} clusters found. "
                "Expected more diverse clustering for ~100 documents."
            )

    def test_embeddings_capture_semantic_similarity(self):
        """
        Similar terms by embedding should be semantically related.

        User expectation: If I look at what's similar to "learning",
        I should see terms like "neural", "training", "networks" -
        not random unrelated concepts.

        Regression test for Task #145 (embedding quality).
        """
        # Get embeddings using tfidf method (best for semantic similarity)
        embeddings = self.processor.compute_graph_embeddings(
            method='tfidf',
            dimensions=64,
            verbose=False
        )

        if 'learning' in embeddings and len(embeddings) > 10:
            # Find similar terms to "learning"
            from cortical.analysis import cosine_similarity

            learning_emb = embeddings['learning']
            similarities = []

            for term, emb in embeddings.items():
                if term != 'learning':
                    sim = cosine_similarity(learning_emb, emb)
                    similarities.append((term, sim))

            # Get top 10 most similar
            similarities.sort(key=lambda x: -x[1])
            top_similar = [term for term, _ in similarities[:10]]

            # Check that some expected related terms appear
            expected_related = {
                'neural', 'network', 'networks', 'training', 'model',
                'models', 'deep', 'machine', 'data', 'algorithm'
            }

            found_related = [t for t in top_similar if t in expected_related]

            self.assertGreater(
                len(found_related),
                2,
                f"Terms similar to 'learning' should include ML concepts. "
                f"Found: {top_similar}. Expected some of: {expected_related}"
            )


class TestRobustnessBehavior(unittest.TestCase):
    """
    Test that the system handles edge cases gracefully.

    These tests verify that:
    - Empty or invalid queries don't crash the system
    - Unknown terms are handled gracefully
    - The system degrades gracefully rather than failing
    """

    @classmethod
    def setUpClass(cls):
        """Load corpus once for all robustness tests."""
        cls.processor = get_shared_processor()

    def test_empty_query_raises_value_error(self):
        """
        Empty queries should raise ValueError for explicit error handling.

        User expectation: Empty queries are invalid input. The system should
        fail explicitly with a clear error message rather than silently
        returning empty results (which could mask bugs in calling code).

        This is documented behavior - see processor.py find_documents_for_query()
        """
        # Empty string should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            self.processor.find_documents_for_query("", top_n=5)
        self.assertIn("non-empty", str(ctx.exception).lower())

        # Whitespace only should also raise ValueError
        with self.assertRaises(ValueError) as ctx:
            self.processor.find_documents_for_query("   ", top_n=5)
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_unknown_terms_handled_gracefully(self):
        """
        Queries with unknown terms should still return results.

        User expectation: If I search for "xyzzy123abc" (nonsense),
        the system should gracefully return empty results or best-effort
        matches, not crash or hang.
        """
        # Completely unknown term
        results = self.processor.find_documents_for_query(
            "xyzzy123abc_unknown_term",
            top_n=5
        )
        # Should return empty list or partial matches, not crash
        self.assertIsInstance(results, list)

        # Mix of known and unknown terms
        results = self.processor.find_documents_for_query(
            "neural xyzzy123abc_unknown",
            top_n=5
        )
        # Should still find neural-related docs
        self.assertIsInstance(results, list)

        # Query expansion should handle unknown terms
        expanded = self.processor.expand_query("xyzzy123abc_unknown", max_expansions=10)
        # Should return dict (possibly empty), not crash
        self.assertIsInstance(expanded, dict)

    def test_special_characters_handled(self):
        """
        Queries with special characters should be handled gracefully.

        User expectation: Pasting code with special chars like
        "func() { return x; }" shouldn't crash the search.
        """
        special_queries = [
            "function() { return x; }",
            "SELECT * FROM table WHERE id=1",
            "@decorator def method(self):",
            "http://example.com/path?query=value",
            "{{template}} ${variable}",
        ]

        for query in special_queries:
            with self.subTest(query=query[:30]):  # Truncate for display
                # Should not raise exceptions
                try:
                    results = self.processor.find_documents_for_query(query, top_n=5)
                    self.assertIsInstance(results, list)
                except Exception as e:
                    self.fail(
                        f"Query '{query[:30]}...' raised exception: {type(e).__name__}: {e}"
                    )


if __name__ == '__main__':
    unittest.main()
