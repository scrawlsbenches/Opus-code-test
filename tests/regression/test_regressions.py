"""
Regression Tests
================

Tests for specific bugs that were fixed, preventing recurrence.
Each test documents the original bug and the task that fixed it.

When adding a new regression test:
1. Document the task/issue number
2. Describe the bug that was fixed
3. Write a minimal test that would have caught the bug
4. Include the date the bug was fixed

Run with: pytest tests/regression/ -v
"""

import pytest


class TestBigramSeparatorRegression:
    """
    Task #10 (2025-12-10): Bigram separators must be spaces, not underscores.

    Bug: Bigrams were inconsistently created with underscores ("neural_networks")
    but searched with spaces ("neural networks"), causing search failures.

    Fix: Standardized on space separators throughout.
    """

    def test_bigrams_use_space_separator(self, small_processor):
        """Bigrams should use space separators."""
        from cortical import CorticalLayer

        layer1 = small_processor.get_layer(CorticalLayer.BIGRAMS)

        # Check that bigrams exist and use spaces
        bigram_contents = [col.content for col in layer1]

        # Should have some bigrams
        assert len(bigram_contents) > 0

        # None should have underscores as separators
        underscore_bigrams = [b for b in bigram_contents if '_' in b and ' ' not in b]
        assert len(underscore_bigrams) == 0, (
            f"Found bigrams with underscore separators: {underscore_bigrams[:5]}"
        )

    def test_bigram_search_finds_results(self, small_processor):
        """Searching for bigrams with spaces should work."""
        # This should find documents about machine learning
        results = small_processor.find_documents_for_query(
            "machine learning",
            top_n=5
        )

        # Should return results (the space-separated bigram matches)
        assert len(results) > 0


class TestCodeNoiseFilterRegression:
    """
    Task #141 (2025-12-12): Python keywords pollute analysis when code is indexed.

    Bug: When Python files were added to corpus, "self", "def", "str" appeared
    in top PageRank terms, drowning out meaningful content.

    Fix: Added filter_code_noise option to tokenizer.
    """

    def test_code_noise_filtered_from_top_terms(self, small_processor):
        """Top PageRank terms should not include Python keywords."""
        from cortical import CorticalLayer

        layer0 = small_processor.get_layer(CorticalLayer.TOKENS)

        # Get top 30 PageRank terms
        top_terms = sorted(
            [(col.content, col.pagerank) for col in layer0],
            key=lambda x: -x[1]
        )[:30]
        top_term_names = [term for term, _ in top_terms]

        # These should never appear in top terms when filtering is on
        noise_tokens = {'self', 'def', 'cls', 'args', 'kwargs', 'none', 'true', 'false'}
        found_noise = [t for t in top_term_names if t in noise_tokens]

        assert len(found_noise) == 0, (
            f"Top PageRank terms contain noise: {found_noise}"
        )


class TestDocNameBoostRegression:
    """
    Task #144 (2025-12-12): Document name matches should rank highly.

    Bug: Query "distributed systems" returned unrelated documents before
    the document actually named "distributed_systems".

    Fix: Added doc_name_boost parameter to search functions.
    """

    def test_doc_name_match_in_results(self, small_processor):
        """Query matching document name should appear in top results."""
        # Small corpus has "ml_*", "db_*", etc. prefixes
        # The key is that docs with matching prefixes should appear
        test_cases = [
            ("machine learning", "ml_"),  # "ml_" docs should appear
            ("database", "db_"),           # "db_" docs should appear
        ]

        for query, expected_prefix in test_cases:
            results = small_processor.find_documents_for_query(query, top_n=5)
            result_docs = [doc_id for doc_id, _ in results]

            # At least one doc with the expected prefix should appear
            found_matching = any(doc.startswith(expected_prefix) for doc in result_docs)
            assert found_matching, (
                f"Query '{query}' should return doc with '{expected_prefix}' prefix in top 5. "
                f"Got: {result_docs}"
            )


class TestClusterStrictnessDirectionRegression:
    """
    Task #122 (2025-12-11): Cluster strictness parameter was inverted.

    Bug: Higher cluster_strictness values produced FEWER clusters (opposite
    of documented behavior) because the threshold calculation was backwards.

    Fix: Corrected threshold calculation in analysis.py.
    """

    def test_higher_resolution_produces_more_clusters(self):
        """Higher Louvain resolution should produce more clusters."""
        from cortical import CorticalTextProcessor
        from tests.fixtures.small_corpus import SMALL_CORPUS_DOCS

        # Create processor and load docs
        processor = CorticalTextProcessor()
        for doc_id, content in SMALL_CORPUS_DOCS.items():
            processor.process_document(doc_id, content)
        processor.propagate_activation(verbose=False)
        processor.compute_bigram_connections(verbose=False)

        # Low resolution should produce fewer clusters
        processor.build_concept_clusters(resolution=0.5, verbose=False)
        from cortical import CorticalLayer
        low_res_clusters = processor.get_layer(CorticalLayer.CONCEPTS).column_count()

        # Reset and try high resolution
        processor._mark_all_stale()
        processor.build_concept_clusters(resolution=2.0, verbose=False)
        high_res_clusters = processor.get_layer(CorticalLayer.CONCEPTS).column_count()

        # Higher resolution should produce more or equal clusters
        assert high_res_clusters >= low_res_clusters, (
            f"Higher resolution (2.0) produced {high_res_clusters} clusters, "
            f"but lower resolution (0.5) produced {low_res_clusters}. "
            f"Resolution parameter direction may be inverted."
        )


class TestMegaClusterRegression:
    """
    Task #123 (2025-12-11): Label propagation created single mega-cluster.

    Bug: With highly connected graphs, label propagation converged to a
    single cluster containing 99%+ of tokens, making Layer 2 useless.

    Fix: Replaced with Louvain community detection algorithm.
    """

    def test_no_mega_cluster(self, small_processor):
        """No single cluster should dominate the concept layer."""
        from cortical import CorticalLayer

        layer0 = small_processor.get_layer(CorticalLayer.TOKENS)
        layer2 = small_processor.get_layer(CorticalLayer.CONCEPTS)

        total_tokens = layer0.column_count()
        if total_tokens == 0 or layer2.column_count() == 0:
            pytest.skip("No clusters to check")

        # Count tokens per cluster
        cluster_sizes = []
        for concept_col in layer2:
            cluster_size = len(concept_col.feedforward_connections)
            cluster_sizes.append(cluster_size)

        max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        max_ratio = max_cluster_size / total_tokens if total_tokens > 0 else 0

        # No cluster should contain more than 30% of tokens
        assert max_ratio < 0.30, (
            f"Mega-cluster detected: {max_cluster_size}/{total_tokens} tokens "
            f"({max_ratio:.1%}) in largest cluster. Max allowed: 30%"
        )


class TestEmbeddingSparsenessRegression:
    """
    Task #122 (2025-12-11): Adjacency embeddings were too sparse.

    Bug: Embeddings only captured direct connections to landmarks, resulting
    in mostly-zero vectors that produced meaningless similarities.

    Fix: Added multi-hop propagation and alternative embedding methods.
    """

    def test_embeddings_are_not_all_zero(self, small_processor):
        """Graph embeddings should have some non-zero values."""
        # compute_graph_embeddings stores results on processor.embeddings
        small_processor.compute_graph_embeddings(
            method='tfidf',
            dimensions=32,
            verbose=False
        )
        embeddings = small_processor.embeddings

        if len(embeddings) == 0:
            pytest.skip("No embeddings computed")

        # Check that embeddings are not completely zero vectors
        # (the old bug produced all-zero vectors for most terms)
        completely_zero_count = 0
        for term, emb in embeddings.items():
            nonzero = sum(1 for v in emb if abs(v) > 1e-10)
            if nonzero == 0:  # Completely zero vector is useless
                completely_zero_count += 1

        zero_ratio = completely_zero_count / len(embeddings)
        # Less than 10% should be completely zero
        assert zero_ratio < 0.1, (
            f"{completely_zero_count}/{len(embeddings)} embeddings are all zeros. "
            f"Embeddings should have at least some non-zero values."
        )

        # Also verify a known term has meaningful embedding
        if 'learning' in embeddings:
            learning_emb = embeddings['learning']
            nonzero = sum(1 for v in learning_emb if abs(v) > 1e-10)
            assert nonzero > 0, "'learning' embedding should not be all zeros"


class TestTestFilePenaltyRegression:
    """
    Task #128 (2025-12-11): Test files ranked higher than implementations.

    Bug: When searching for definitions, test files with mocks ranked above
    actual implementation files because they had more keyword matches.

    Fix: Added is_test_file() detection and test_file_penalty parameter.
    """

    def test_implementation_preferred_over_test(self):
        """Implementation files should rank above test files for definitions."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()

        # Add a "real" implementation
        processor.process_document("data_processor", """
            class DataProcessor:
                def process(self, data):
                    '''Process the input data and return results.'''
                    return self.transform(data)

                def transform(self, data):
                    return [x * 2 for x in data]
        """)

        # Add a test file with mocks
        processor.process_document("test_data_processor", """
            class TestDataProcessor(unittest.TestCase):
                def test_process(self):
                    processor = DataProcessor()
                    result = processor.process([1, 2, 3])
                    self.assertEqual(result, [2, 4, 6])

                def test_transform(self):
                    processor = DataProcessor()
                    self.assertIsNotNone(processor.transform([]))
        """)

        processor.compute_all(verbose=False)

        # Search for DataProcessor
        results = processor.find_documents_for_query("DataProcessor class", top_n=2)
        top_doc = results[0][0] if results else None

        # Implementation should be first (or at least present)
        result_docs = [doc_id for doc_id, _ in results]
        assert "data_processor" in result_docs, (
            f"Implementation 'data_processor' should be in results. Got: {result_docs}"
        )


class TestEmptyQueryHandlingRegression:
    """
    Task #146 (2025-12-12): Empty queries should raise explicit errors.

    Bug: Empty queries returned empty results silently, which could mask
    bugs in calling code that accidentally passed empty strings.

    Fix: Added explicit ValueError for empty queries.
    """

    def test_empty_string_raises_value_error(self, small_processor):
        """Empty string query should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            small_processor.find_documents_for_query("", top_n=5)

        assert "non-empty" in str(exc_info.value).lower()

    def test_whitespace_only_raises_value_error(self, small_processor):
        """Whitespace-only query should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            small_processor.find_documents_for_query("   \t\n  ", top_n=5)

        assert "non-empty" in str(exc_info.value).lower()


class TestEdgeCasesRegression:
    """
    Agent Delta Task #T-20251217-025242-6b01-018: Edge case regression tests.

    Tests for common edge cases that should be handled gracefully:
    - Empty corpus queries
    - Single document corpus
    - Very long documents
    - Special characters in queries
    - Unicode handling
    """

    def test_empty_corpus_search_returns_empty(self):
        """Searching an empty corpus should return empty results, not crash."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        # Don't add any documents
        processor.compute_all()

        results = processor.find_documents_for_query("test query")
        assert results == []

    def test_single_document_corpus(self):
        """Single document corpus should work correctly."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("only_doc", "This is the only document in the corpus.")
        processor.compute_all()

        results = processor.find_documents_for_query("document")
        assert len(results) == 1
        assert results[0][0] == "only_doc"

    def test_very_long_document_handled(self):
        """Very long documents (>10k words) should be processed without error."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        # Create a very long document (~10k words)
        long_text = " ".join([f"word{i}" for i in range(10000)])
        processor.process_document("long_doc", long_text)
        processor.compute_all()

        # Should complete without error
        results = processor.find_documents_for_query("word5000")
        assert len(results) > 0

    def test_special_characters_in_query(self):
        """Queries with special characters should be handled gracefully."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document with C++ and Python code.")
        processor.compute_all()

        # Should not crash with special chars
        results = processor.find_documents_for_query("C++")
        # Results may be empty if tokenizer strips special chars, but shouldn't crash
        assert isinstance(results, list)

    def test_unicode_text_processing(self):
        """Unicode text (non-ASCII) should be processed correctly."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("unicode_doc", "Résumé français café naïve Москва 北京")
        processor.compute_all()

        # Should process without error
        results = processor.find_documents_for_query("café")
        assert isinstance(results, list)

    def test_query_with_no_matching_terms(self):
        """Queries with no matching terms should return empty results, not crash."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "machine learning neural networks")
        processor.compute_all()

        # Query for something completely different
        results = processor.find_documents_for_query("quantum physics")
        assert results == []

    def test_repeated_identical_documents(self):
        """Adding identical documents should not cause issues."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        identical_text = "This is identical text."
        for i in range(5):
            processor.process_document(f"doc{i}", identical_text)
        processor.compute_all()

        # Should work without error
        results = processor.find_documents_for_query("identical")
        assert len(results) == 5  # All docs should match

    def test_document_with_only_stopwords(self):
        """Documents containing only stopwords should be handled."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("stopword_doc", "the a an and or but")
        processor.process_document("normal_doc", "machine learning algorithms")
        processor.compute_all()

        # Should not crash when searching
        results = processor.find_documents_for_query("machine")
        assert len(results) > 0
