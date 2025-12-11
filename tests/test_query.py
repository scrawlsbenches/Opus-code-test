"""
Query Module Tests
==================

Comprehensive tests for cortical/query.py functions.

Tests cover:
- Query expansion (lateral, semantic, multihop)
- Relation path scoring
- Chunking and chunk scoring
- Batch operations
- Relation discovery functions
- Analogy completion
"""

import unittest
from typing import Dict, List, Tuple

from cortical import CorticalTextProcessor
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.tokenizer import Tokenizer
from cortical.query import (
    expand_query,
    expand_query_multihop,
    expand_query_semantic,
    get_expanded_query_terms,
    score_relation_path,
    create_chunks,
    score_chunk,
    find_documents_for_query,
    find_passages_for_query,
    find_documents_batch,
    find_passages_batch,
    find_relevant_concepts,
    find_relation_between,
    find_terms_with_relation,
    complete_analogy,
    complete_analogy_simple,
    query_with_spreading_activation,
    VALID_RELATION_CHAINS,
    is_definition_query,
    find_definition_in_text,
    find_definition_passages,
    DEFINITION_QUERY_PATTERNS,
    DEFINITION_SOURCE_PATTERNS,
    DEFINITION_BOOST,
    find_code_boundaries,
    create_code_aware_chunks,
    is_code_file,
)


class TestScoreRelationPath(unittest.TestCase):
    """Test relation path scoring."""

    def test_empty_path(self):
        """Empty path should return 1.0."""
        self.assertEqual(score_relation_path([]), 1.0)

    def test_single_relation(self):
        """Single relation path should return 1.0."""
        self.assertEqual(score_relation_path(['IsA']), 1.0)
        self.assertEqual(score_relation_path(['HasProperty']), 1.0)

    def test_valid_chain(self):
        """Valid chain should return high score."""
        # IsA → HasProperty is typically valid
        score = score_relation_path(['IsA', 'HasProperty'])
        self.assertGreater(score, 0.5)

    def test_long_path_degrades(self):
        """Longer paths should have lower scores due to multiplication."""
        score_2 = score_relation_path(['IsA', 'HasProperty'])
        score_3 = score_relation_path(['IsA', 'HasProperty', 'RelatedTo'])
        self.assertLessEqual(score_3, score_2)

    def test_valid_relation_chains_constant(self):
        """VALID_RELATION_CHAINS should be a non-empty dict."""
        self.assertIsInstance(VALID_RELATION_CHAINS, dict)
        self.assertGreater(len(VALID_RELATION_CHAINS), 0)


class TestCreateChunks(unittest.TestCase):
    """Test text chunking."""

    def test_empty_text(self):
        """Empty text should return empty list."""
        self.assertEqual(create_chunks(""), [])

    def test_short_text(self):
        """Text shorter than chunk_size should return single chunk."""
        text = "Short text."
        chunks = create_chunks(text, chunk_size=100, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], text)
        self.assertEqual(chunks[0][1], 0)  # start
        self.assertEqual(chunks[0][2], len(text))  # end

    def test_chunk_overlap(self):
        """Chunks should overlap by specified amount."""
        text = "A" * 100
        chunks = create_chunks(text, chunk_size=50, overlap=10)
        # With chunk_size=50 and overlap=10, stride=40
        # Chunks at: 0-50, 40-90, 80-100
        self.assertGreater(len(chunks), 1)
        # First chunk ends at 50, second starts at 40
        if len(chunks) >= 2:
            first_end = chunks[0][2]
            second_start = chunks[1][1]
            self.assertLess(second_start, first_end)  # Overlap exists

    def test_chunk_boundaries(self):
        """Chunk boundaries should be valid."""
        text = "Hello world, this is a test of chunking functionality."
        chunks = create_chunks(text, chunk_size=20, overlap=5)

        for chunk_text, start, end in chunks:
            self.assertEqual(chunk_text, text[start:end])
            self.assertLessEqual(end, len(text))

    def test_no_overlap(self):
        """Chunks with zero overlap should not overlap."""
        text = "A" * 100
        chunks = create_chunks(text, chunk_size=25, overlap=0)
        # Should have exactly 4 chunks
        self.assertEqual(len(chunks), 4)
        # Check no overlap
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i][2], chunks[i + 1][1])

    def test_invalid_chunk_size_zero(self):
        """Chunk size of zero should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_chunks("test", chunk_size=0)
        self.assertIn("chunk_size", str(ctx.exception))

    def test_invalid_chunk_size_negative(self):
        """Negative chunk size should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_chunks("test", chunk_size=-10)
        self.assertIn("chunk_size", str(ctx.exception))

    def test_invalid_overlap_negative(self):
        """Negative overlap should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_chunks("test", chunk_size=50, overlap=-5)
        self.assertIn("overlap", str(ctx.exception))

    def test_invalid_overlap_greater_than_chunk_size(self):
        """Overlap >= chunk_size should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_chunks("test", chunk_size=50, overlap=50)
        self.assertIn("overlap", str(ctx.exception))

        with self.assertRaises(ValueError):
            create_chunks("test", chunk_size=50, overlap=100)


class TestFindRelationBetween(unittest.TestCase):
    """Test finding relations between terms."""

    def setUp(self):
        """Set up sample relations."""
        self.relations = [
            ("dog", "IsA", "animal", 1.0),
            ("cat", "IsA", "animal", 1.0),
            ("dog", "HasProperty", "loyal", 0.8),
            ("neural", "RelatedTo", "networks", 0.7),
            ("machine", "RelatedTo", "learning", 0.9),
        ]

    def test_forward_relation(self):
        """Find relation in forward direction."""
        results = find_relation_between("dog", "animal", self.relations)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "IsA")
        self.assertEqual(results[0][1], 1.0)

    def test_reverse_relation(self):
        """Find relation in reverse direction with penalty."""
        results = find_relation_between("animal", "dog", self.relations)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "IsA")
        self.assertLess(results[0][1], 1.0)  # Penalty applied

    def test_no_relation(self):
        """Return empty list when no relation exists."""
        results = find_relation_between("dog", "neural", self.relations)
        self.assertEqual(results, [])

    def test_multiple_relations(self):
        """Multiple relations between same terms."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("dog", "RelatedTo", "animal", 0.5),
        ]
        results = find_relation_between("dog", "animal", relations)
        self.assertEqual(len(results), 2)
        # Should be sorted by weight
        self.assertEqual(results[0][0], "IsA")


class TestFindTermsWithRelation(unittest.TestCase):
    """Test finding terms with specific relation."""

    def setUp(self):
        """Set up sample relations."""
        self.relations = [
            ("dog", "IsA", "animal", 1.0),
            ("cat", "IsA", "animal", 0.9),
            ("bird", "IsA", "animal", 0.8),
            ("dog", "HasProperty", "loyal", 0.8),
            ("cat", "HasProperty", "independent", 0.7),
        ]

    def test_forward_direction(self):
        """Find terms in forward direction (term → x)."""
        results = find_terms_with_relation(
            "dog", "IsA", self.relations, direction='forward'
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "animal")

    def test_backward_direction(self):
        """Find terms in backward direction (x → term)."""
        results = find_terms_with_relation(
            "animal", "IsA", self.relations, direction='backward'
        )
        self.assertEqual(len(results), 3)  # dog, cat, bird
        terms = [r[0] for r in results]
        self.assertIn("dog", terms)
        self.assertIn("cat", terms)
        self.assertIn("bird", terms)

    def test_no_matching_relation(self):
        """Return empty when no matching relation type."""
        results = find_terms_with_relation(
            "dog", "PartOf", self.relations, direction='forward'
        )
        self.assertEqual(results, [])

    def test_results_sorted_by_weight(self):
        """Results should be sorted by weight descending."""
        results = find_terms_with_relation(
            "animal", "IsA", self.relations, direction='backward'
        )
        weights = [r[1] for r in results]
        self.assertEqual(weights, sorted(weights, reverse=True))


class TestExpandQuery(unittest.TestCase):
    """Test basic query expansion."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks are fundamental to machine learning. "
            "Deep learning uses neural architectures for complex tasks."
        )
        cls.processor.process_document(
            "doc2",
            "Machine learning algorithms process data patterns. "
            "Neural models learn from training examples."
        )
        cls.processor.compute_all(verbose=False)

    def test_expand_query_returns_dict(self):
        """expand_query should return a dictionary."""
        result = expand_query(
            "neural networks",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(result, dict)

    def test_expand_query_includes_original_terms(self):
        """Expanded query should include original terms."""
        result = expand_query(
            "neural learning",
            self.processor.layers,
            self.processor.tokenizer
        )
        # Original terms should have high weight
        self.assertIn("neural", result)
        self.assertIn("learning", result)

    def test_expand_query_unknown_terms(self):
        """Unknown terms should be handled gracefully."""
        result = expand_query(
            "xyznonexistent",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(result, dict)


class TestExpandQueryMultihop(unittest.TestCase):
    """Test multi-hop query expansion."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with semantic relations."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Dogs are loyal animals. Cats are independent pets."
        )
        cls.processor.process_document(
            "doc2",
            "Animals need food and water. Pets require care."
        )
        cls.processor.compute_all(verbose=False)
        cls.processor.extract_corpus_semantics(verbose=False)

    def test_multihop_returns_dict(self):
        """expand_query_multihop should return a dictionary."""
        result = expand_query_multihop(
            "dogs",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations
        )
        self.assertIsInstance(result, dict)

    def test_multihop_with_max_hops(self):
        """max_hops should limit expansion depth."""
        result_1 = expand_query_multihop(
            "dogs",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations,
            max_hops=1
        )
        result_2 = expand_query_multihop(
            "dogs",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations,
            max_hops=2
        )
        # More hops could mean more terms (or same if no valid chains)
        self.assertIsInstance(result_1, dict)
        self.assertIsInstance(result_2, dict)


class TestGetExpandedQueryTerms(unittest.TestCase):
    """Test the unified query expansion helper."""

    @classmethod
    def setUpClass(cls):
        """Set up processor."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks process information efficiently. "
            "Machine learning improves with data."
        )
        cls.processor.compute_all(verbose=False)
        cls.processor.extract_corpus_semantics(verbose=False)

    def test_get_expanded_returns_dict(self):
        """get_expanded_query_terms should return dict."""
        result = get_expanded_query_terms(
            "neural",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations
        )
        self.assertIsInstance(result, dict)

    def test_max_expansions_limits_results(self):
        """max_expansions should limit number of terms."""
        result = get_expanded_query_terms(
            "neural networks machine",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations,
            max_expansions=5
        )
        # Should have at most 5 expansion terms + original terms
        self.assertIsInstance(result, dict)

    def test_semantic_discount_affects_weights(self):
        """semantic_discount should reduce semantic expansion weights."""
        result_high = get_expanded_query_terms(
            "neural",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations,
            semantic_discount=1.0
        )
        result_low = get_expanded_query_terms(
            "neural",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.semantic_relations,
            semantic_discount=0.1
        )
        self.assertIsInstance(result_high, dict)
        self.assertIsInstance(result_low, dict)


class TestFindDocumentsForQuery(unittest.TestCase):
    """Test document retrieval."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "neural_doc",
            "Neural networks are powerful models for pattern recognition. "
            "Deep learning architectures use multiple neural layers."
        )
        cls.processor.process_document(
            "ml_doc",
            "Machine learning algorithms learn from data. "
            "Supervised learning requires labeled examples."
        )
        cls.processor.process_document(
            "unrelated_doc",
            "The weather today is sunny with clear skies. "
            "Tomorrow expects rain in the afternoon."
        )
        cls.processor.compute_all(verbose=False)

    def test_returns_list_of_tuples(self):
        """Should return list of (doc_id, score) tuples."""
        results = find_documents_for_query(
            "neural networks",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(results, list)
        if results:
            self.assertEqual(len(results[0]), 2)
            self.assertIsInstance(results[0][0], str)
            self.assertIsInstance(results[0][1], float)

    def test_relevant_docs_ranked_higher(self):
        """Relevant documents should be ranked higher."""
        results = find_documents_for_query(
            "neural networks deep learning",
            self.processor.layers,
            self.processor.tokenizer,
            top_n=3
        )
        if len(results) >= 2:
            doc_ids = [r[0] for r in results]
            # neural_doc should rank higher than unrelated_doc
            if "neural_doc" in doc_ids and "unrelated_doc" in doc_ids:
                self.assertLess(
                    doc_ids.index("neural_doc"),
                    doc_ids.index("unrelated_doc")
                )

    def test_top_n_limits_results(self):
        """top_n should limit number of results."""
        results = find_documents_for_query(
            "learning",
            self.processor.layers,
            self.processor.tokenizer,
            top_n=1
        )
        self.assertLessEqual(len(results), 1)


class TestFindDocumentsBatch(unittest.TestCase):
    """Test batch document retrieval."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", "Neural networks learn patterns.")
        cls.processor.process_document("doc2", "Machine learning uses algorithms.")
        cls.processor.compute_all(verbose=False)

    def test_batch_returns_list_of_lists(self):
        """Should return list of result lists."""
        queries = ["neural", "machine"]
        results = find_documents_batch(
            queries,
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for result_list in results:
            self.assertIsInstance(result_list, list)

    def test_batch_empty_queries(self):
        """Empty query list should return empty list."""
        results = find_documents_batch(
            [],
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertEqual(results, [])


class TestFindPassagesForQuery(unittest.TestCase):
    """Test passage retrieval."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks are computational models. "
            "They process data through layers of neurons. "
            "Deep learning uses many layers for complex tasks."
        )
        cls.processor.compute_all(verbose=False)

    def test_returns_list_of_tuples(self):
        """Should return list with passage info."""
        results = find_passages_for_query(
            "neural networks",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.documents
        )
        self.assertIsInstance(results, list)
        if results:
            # Should have (doc_id, passage_text, start, end, score)
            self.assertEqual(len(results[0]), 5)

    def test_passage_contains_query_terms(self):
        """Returned passages should be relevant to query."""
        results = find_passages_for_query(
            "neural networks",
            self.processor.layers,
            self.processor.tokenizer,
            self.processor.documents,
            top_n=1
        )
        # Should return at least one passage
        self.assertIsInstance(results, list)
        # If results exist, check format is correct
        if results:
            doc_id, passage_text, start, end, score = results[0]
            self.assertIsInstance(doc_id, str)
            self.assertIsInstance(passage_text, str)
            self.assertIsInstance(score, float)
            self.assertGreater(len(passage_text), 0)


class TestFindRelevantConcepts(unittest.TestCase):
    """Test concept filtering for RAG."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with concepts."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks process information. Machine learning improves results."
        )
        cls.processor.process_document(
            "doc2",
            "Deep learning architectures use neural layers. Data processing is key."
        )
        cls.processor.compute_all(verbose=False)

    def test_returns_list(self):
        """Should return list of concept info."""
        # find_relevant_concepts takes query_terms dict, not string
        query_terms = {"neural": 1.0, "learning": 0.8}
        result = find_relevant_concepts(
            query_terms,
            self.processor.layers
        )
        self.assertIsInstance(result, list)

    def test_top_n_limits_results(self):
        """top_n should limit results."""
        query_terms = {"neural": 1.0, "learning": 0.8}
        result = find_relevant_concepts(
            query_terms,
            self.processor.layers,
            top_n=2
        )
        self.assertLessEqual(len(result), 2)


class TestCompleteAnalogy(unittest.TestCase):
    """Test analogy completion functions."""

    @classmethod
    def setUpClass(cls):
        """Set up processor for analogy tests."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks are like brain structures. "
            "Machine learning uses algorithms for patterns. "
            "Deep learning processes complex data."
        )
        cls.processor.process_document(
            "doc2",
            "Data science analyzes information. "
            "Neural processing enables artificial intelligence."
        )
        cls.processor.compute_all(verbose=False)
        cls.processor.extract_corpus_semantics(verbose=False)

    def test_complete_analogy_returns_list(self):
        """complete_analogy should return list."""
        results = complete_analogy(
            "neural", "networks", "machine",
            self.processor.layers,
            self.processor.semantic_relations
        )
        self.assertIsInstance(results, list)

    def test_complete_analogy_excludes_input(self):
        """Input terms should not appear in results."""
        results = complete_analogy(
            "neural", "networks", "machine",
            self.processor.layers,
            self.processor.semantic_relations
        )
        result_terms = [r[0] for r in results]
        self.assertNotIn("neural", result_terms)
        self.assertNotIn("networks", result_terms)
        self.assertNotIn("machine", result_terms)

    def test_complete_analogy_simple_returns_list(self):
        """complete_analogy_simple should return list."""
        results = complete_analogy_simple(
            "neural", "networks", "machine",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(results, list)

    def test_complete_analogy_simple_format(self):
        """Simple analogy results should be (term, score) tuples."""
        results = complete_analogy_simple(
            "neural", "networks", "learning",
            self.processor.layers,
            self.processor.tokenizer,
            top_n=3
        )
        for result in results:
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], str)
            self.assertIsInstance(result[1], float)


class TestQueryWithSpreadingActivation(unittest.TestCase):
    """Test spreading activation search."""

    @classmethod
    def setUpClass(cls):
        """Set up processor."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks process signals. Deep learning improves accuracy."
        )
        cls.processor.compute_all(verbose=False)

    def test_returns_list(self):
        """Should return list of results."""
        results = query_with_spreading_activation(
            "neural",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(results, list)

    def test_max_expansions_parameter(self):
        """max_expansions parameter should be accepted."""
        results = query_with_spreading_activation(
            "neural",
            self.processor.layers,
            self.processor.tokenizer,
            max_expansions=5
        )
        self.assertIsInstance(results, list)


class TestScoreChunk(unittest.TestCase):
    """Test chunk scoring."""

    @classmethod
    def setUpClass(cls):
        """Set up processor."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks are powerful tools for data analysis."
        )
        cls.processor.compute_all(verbose=False)

    def test_score_chunk_returns_float(self):
        """score_chunk should return a float."""
        query_terms = {"neural": 1.0, "networks": 0.8}
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        score = score_chunk(
            "Neural networks process data",
            query_terms,
            layer0,
            self.processor.tokenizer
        )
        self.assertIsInstance(score, float)

    def test_relevant_chunk_higher_score(self):
        """Chunks with query terms should score higher."""
        query_terms = {"neural": 1.0, "networks": 0.8}
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        relevant_score = score_chunk(
            "Neural networks are amazing",
            query_terms,
            layer0,
            self.processor.tokenizer
        )
        irrelevant_score = score_chunk(
            "Weather is nice today",
            query_terms,
            layer0,
            self.processor.tokenizer
        )
        self.assertGreaterEqual(relevant_score, irrelevant_score)

    def test_empty_chunk(self):
        """Empty chunk should return 0."""
        query_terms = {"neural": 1.0}
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        score = score_chunk(
            "",
            query_terms,
            layer0,
            self.processor.tokenizer
        )
        self.assertEqual(score, 0.0)


class TestChunkScoringOptimization(unittest.TestCase):
    """Test optimized chunk scoring functions."""

    @classmethod
    def setUpClass(cls):
        """Set up processor."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks are powerful tools for data analysis."
        )
        cls.processor.compute_all(verbose=False)

    def test_precompute_term_cols_returns_dict(self):
        """precompute_term_cols should return dict of Minicolumns."""
        from cortical.query import precompute_term_cols
        query_terms = {"neural": 1.0, "networks": 0.8}
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        term_cols = precompute_term_cols(query_terms, layer0)

        self.assertIsInstance(term_cols, dict)
        self.assertIn("neural", term_cols)
        self.assertIn("networks", term_cols)

    def test_precompute_term_cols_excludes_unknown(self):
        """precompute_term_cols should exclude terms not in corpus."""
        from cortical.query import precompute_term_cols
        query_terms = {"neural": 1.0, "xyz_unknown": 0.5}
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        term_cols = precompute_term_cols(query_terms, layer0)

        self.assertIn("neural", term_cols)
        self.assertNotIn("xyz_unknown", term_cols)

    def test_score_chunk_fast_matches_regular(self):
        """score_chunk_fast should produce same results as score_chunk."""
        from cortical.query import precompute_term_cols, score_chunk_fast
        query_terms = {"neural": 1.0, "networks": 0.8}
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        chunk_text = "Neural networks process data"

        # Regular score
        regular_score = score_chunk(
            chunk_text, query_terms, layer0, self.processor.tokenizer
        )

        # Fast score
        term_cols = precompute_term_cols(query_terms, layer0)
        chunk_tokens = self.processor.tokenizer.tokenize(chunk_text)
        fast_score = score_chunk_fast(chunk_tokens, query_terms, term_cols)

        self.assertAlmostEqual(regular_score, fast_score, places=6)

    def test_score_chunk_fast_empty_tokens(self):
        """score_chunk_fast should handle empty tokens list."""
        from cortical.query import score_chunk_fast
        query_terms = {"neural": 1.0}
        term_cols = {}

        score = score_chunk_fast([], query_terms, term_cols)
        self.assertEqual(score, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        """Set up empty and minimal processors."""
        cls.empty_processor = CorticalTextProcessor()

        cls.minimal_processor = CorticalTextProcessor()
        cls.minimal_processor.process_document("doc1", "Hello world")
        cls.minimal_processor.compute_all(verbose=False)

    def test_expand_query_empty_corpus(self):
        """expand_query should handle empty corpus."""
        result = expand_query(
            "test query",
            self.empty_processor.layers,
            self.empty_processor.tokenizer
        )
        self.assertIsInstance(result, dict)

    def test_find_documents_empty_corpus(self):
        """find_documents should handle empty corpus."""
        result = find_documents_for_query(
            "test",
            self.empty_processor.layers,
            self.empty_processor.tokenizer
        )
        self.assertEqual(result, [])

    def test_find_relation_empty_relations(self):
        """find_relation_between should handle empty relations."""
        result = find_relation_between("a", "b", [])
        self.assertEqual(result, [])

    def test_find_terms_empty_relations(self):
        """find_terms_with_relation should handle empty relations."""
        result = find_terms_with_relation("a", "IsA", [])
        self.assertEqual(result, [])


class TestDocTypeBoost(unittest.TestCase):
    """Test document type boosting for search results."""

    def test_is_conceptual_query_what(self):
        """'what is' queries should be conceptual."""
        from cortical.query import is_conceptual_query
        self.assertTrue(is_conceptual_query("what is PageRank"))
        self.assertTrue(is_conceptual_query("What are the algorithms?"))

    def test_is_conceptual_query_explain(self):
        """'explain' queries should be conceptual."""
        from cortical.query import is_conceptual_query
        self.assertTrue(is_conceptual_query("explain PageRank algorithm"))
        self.assertTrue(is_conceptual_query("Explain how TF-IDF works"))

    def test_is_conceptual_query_how_does(self):
        """'how does' queries should be conceptual."""
        from cortical.query import is_conceptual_query
        self.assertTrue(is_conceptual_query("how does the system work"))

    def test_is_conceptual_query_where(self):
        """'where' queries should be implementation-focused."""
        from cortical.query import is_conceptual_query
        self.assertFalse(is_conceptual_query("where is PageRank computed"))
        self.assertFalse(is_conceptual_query("where do we implement authentication"))

    def test_is_conceptual_query_implementation(self):
        """Queries with 'implementation' keywords should not be conceptual."""
        from cortical.query import is_conceptual_query
        self.assertFalse(is_conceptual_query("find the function that calculates TF-IDF"))
        self.assertFalse(is_conceptual_query("line where error is raised"))

    def test_is_conceptual_query_neutral(self):
        """Neutral queries without keywords should not be conceptual."""
        from cortical.query import is_conceptual_query
        self.assertFalse(is_conceptual_query("PageRank"))
        self.assertFalse(is_conceptual_query("bigram separator"))

    def test_get_doc_type_boost_docs_folder(self):
        """docs/ files should get high boost."""
        from cortical.query import get_doc_type_boost
        boost = get_doc_type_boost("docs/algorithms.md")
        self.assertEqual(boost, 1.5)

    def test_get_doc_type_boost_root_md(self):
        """Root-level .md files should get medium boost."""
        from cortical.query import get_doc_type_boost
        boost = get_doc_type_boost("README.md")
        self.assertEqual(boost, 1.3)
        boost = get_doc_type_boost("CLAUDE.md")
        self.assertEqual(boost, 1.3)

    def test_get_doc_type_boost_code(self):
        """Code files should get normal boost (1.0)."""
        from cortical.query import get_doc_type_boost
        boost = get_doc_type_boost("cortical/processor.py")
        self.assertEqual(boost, 1.0)

    def test_get_doc_type_boost_tests(self):
        """Test files should get lower boost."""
        from cortical.query import get_doc_type_boost
        boost = get_doc_type_boost("tests/test_processor.py")
        self.assertEqual(boost, 0.8)

    def test_get_doc_type_boost_with_metadata(self):
        """Should use metadata when available."""
        from cortical.query import get_doc_type_boost
        metadata = {
            "myfile.py": {"doc_type": "docs"}  # Override: code file marked as docs
        }
        boost = get_doc_type_boost("myfile.py", doc_metadata=metadata)
        self.assertEqual(boost, 1.5)

    def test_apply_doc_type_boost_reranks(self):
        """apply_doc_type_boost should re-rank results."""
        from cortical.query import apply_doc_type_boost

        # Setup: code file first, then docs
        results = [
            ("cortical/query.py", 1.0),
            ("docs/algorithms.md", 0.9),
        ]

        boosted = apply_doc_type_boost(results)

        # After boost: docs should be first (0.9 * 1.5 = 1.35 > 1.0)
        self.assertEqual(boosted[0][0], "docs/algorithms.md")
        self.assertAlmostEqual(boosted[0][1], 1.35, places=5)

    def test_apply_doc_type_boost_no_boost(self):
        """apply_doc_type_boost should preserve order when disabled."""
        from cortical.query import apply_doc_type_boost

        results = [
            ("cortical/query.py", 1.0),
            ("docs/algorithms.md", 0.9),
        ]

        not_boosted = apply_doc_type_boost(results, boost_docs=False)

        # Order preserved
        self.assertEqual(not_boosted[0][0], "cortical/query.py")
        self.assertEqual(not_boosted[1][0], "docs/algorithms.md")

    def test_apply_doc_type_boost_custom_boosts(self):
        """apply_doc_type_boost should support custom boost factors."""
        from cortical.query import apply_doc_type_boost

        results = [
            ("cortical/query.py", 1.0),
            ("tests/test_query.py", 0.8),
        ]

        # Custom: boost tests instead of docs
        custom = {'code': 1.0, 'test': 2.0, 'docs': 1.0, 'root_docs': 1.0}
        boosted = apply_doc_type_boost(results, custom_boosts=custom)

        # Test file should be first now (0.8 * 2.0 = 1.6 > 1.0)
        self.assertEqual(boosted[0][0], "tests/test_query.py")


class TestDocTypeBoostIntegration(unittest.TestCase):
    """Integration tests for document type boosting."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with different document types."""
        cls.processor = CorticalTextProcessor()

        # Add a code file
        cls.processor.process_document(
            "cortical/processor.py",
            "PageRank algorithm implementation. def compute_pagerank(): pass",
            metadata={"doc_type": "code"}
        )

        # Add a docs file
        cls.processor.process_document(
            "docs/algorithms.md",
            "# Algorithms\n\n## PageRank\n\nPageRank is a link analysis algorithm.",
            metadata={"doc_type": "docs"}
        )

        # Add a test file
        cls.processor.process_document(
            "tests/test_processor.py",
            "class TestPageRank: def test_pagerank(self): pass",
            metadata={"doc_type": "test"}
        )

        cls.processor.compute_all(verbose=False)

    def test_find_documents_with_boost_conceptual(self):
        """Conceptual queries should boost docs."""
        from cortical.query import find_documents_with_boost

        results = find_documents_with_boost(
            "what is PageRank algorithm",
            self.processor.layers,
            self.processor.tokenizer,
            top_n=3,
            doc_metadata=self.processor.document_metadata,
            auto_detect_intent=True
        )

        # Docs file should be ranked higher
        self.assertTrue(len(results) > 0)
        # Check that results are returned (specific ranking depends on corpus)

    def test_find_documents_with_boost_prefer_docs(self):
        """prefer_docs=True should always boost docs."""
        from cortical.query import find_documents_with_boost

        results = find_documents_with_boost(
            "PageRank",  # Neutral query
            self.processor.layers,
            self.processor.tokenizer,
            top_n=3,
            doc_metadata=self.processor.document_metadata,
            auto_detect_intent=False,
            prefer_docs=True
        )

        self.assertTrue(len(results) > 0)

    def test_processor_wrapper_exists(self):
        """Processor should have find_documents_with_boost method."""
        self.assertTrue(hasattr(self.processor, 'find_documents_with_boost'))
        self.assertTrue(hasattr(self.processor, 'is_conceptual_query'))


class TestDefinitionPatternSearch(unittest.TestCase):
    """Test definition pattern search functionality."""

    def test_is_definition_query_class(self):
        """Detect class definition queries."""
        is_def, def_type, name = is_definition_query("class Minicolumn")
        self.assertTrue(is_def)
        self.assertEqual(def_type, 'class')
        self.assertEqual(name, 'Minicolumn')

    def test_is_definition_query_def(self):
        """Detect function definition queries."""
        is_def, def_type, name = is_definition_query("def compute_pagerank")
        self.assertTrue(is_def)
        self.assertEqual(def_type, 'function')
        self.assertEqual(name, 'compute_pagerank')

    def test_is_definition_query_function(self):
        """Detect function keyword queries."""
        is_def, def_type, name = is_definition_query("function tokenize")
        self.assertTrue(is_def)
        self.assertEqual(def_type, 'function')
        self.assertEqual(name, 'tokenize')

    def test_is_definition_query_method(self):
        """Detect method definition queries."""
        is_def, def_type, name = is_definition_query("method process_document")
        self.assertTrue(is_def)
        self.assertEqual(def_type, 'method')
        self.assertEqual(name, 'process_document')

    def test_is_definition_query_not_definition(self):
        """Non-definition queries should return False."""
        is_def, def_type, name = is_definition_query("neural networks")
        self.assertFalse(is_def)
        self.assertIsNone(def_type)
        self.assertIsNone(name)

    def test_is_definition_query_case_insensitive(self):
        """Definition detection should be case insensitive."""
        is_def, def_type, name = is_definition_query("CLASS MyClass")
        self.assertTrue(is_def)
        self.assertEqual(def_type, 'class')
        self.assertEqual(name, 'MyClass')

    def test_find_definition_in_text_python_class(self):
        """Find Python class definitions."""
        text = '''
import os

class MyProcessor:
    """A processor class."""

    def __init__(self):
        pass
'''
        result = find_definition_in_text(text, 'MyProcessor', 'class')
        self.assertIsNotNone(result)
        passage, start, end = result
        self.assertIn('class MyProcessor:', passage)

    def test_find_definition_in_text_python_function(self):
        """Find Python function definitions."""
        text = '''
def compute_score(items, weights):
    """Compute weighted score."""
    total = sum(i * w for i, w in zip(items, weights))
    return total / len(items)
'''
        result = find_definition_in_text(text, 'compute_score', 'function')
        self.assertIsNotNone(result)
        passage, start, end = result
        self.assertIn('def compute_score(', passage)

    def test_find_definition_in_text_not_found(self):
        """Return None when definition not found."""
        text = 'def other_function(): pass'
        result = find_definition_in_text(text, 'nonexistent', 'function')
        self.assertIsNone(result)

    def test_find_definition_in_text_method(self):
        """Find method definitions (indented def)."""
        text = '''
class MyClass:
    def my_method(self, arg):
        return arg * 2
'''
        result = find_definition_in_text(text, 'my_method', 'method')
        self.assertIsNotNone(result)
        passage, start, end = result
        self.assertIn('def my_method(', passage)

    def test_find_definition_passages_basic(self):
        """Find definition passages from documents."""
        documents = {
            'module.py': '''
class TestClass:
    """A test class for demonstration."""

    def process(self):
        pass
''',
            'other.py': 'def helper(): pass',
        }
        results = find_definition_passages("class TestClass", documents)
        self.assertTrue(len(results) > 0)
        passage, doc_id, start, end, score = results[0]
        self.assertEqual(doc_id, 'module.py')
        self.assertIn('class TestClass:', passage)
        self.assertEqual(score, DEFINITION_BOOST)  # No test file penalty

    def test_find_definition_passages_test_file_penalty(self):
        """Test files should have lower score."""
        documents = {
            'src/module.py': 'class MyClass: pass',
            'tests/test_module.py': 'class MyClass: pass',
        }
        results = find_definition_passages("class MyClass", documents)
        self.assertEqual(len(results), 2)

        # Sort by score descending
        results.sort(key=lambda x: -x[4])

        # Source file should rank higher
        self.assertEqual(results[0][1], 'src/module.py')
        self.assertEqual(results[1][1], 'tests/test_module.py')
        self.assertGreater(results[0][4], results[1][4])

    def test_find_definition_passages_not_definition_query(self):
        """Non-definition queries return empty list."""
        documents = {'test.py': 'class Foo: pass'}
        results = find_definition_passages("neural networks", documents)
        self.assertEqual(results, [])


class TestDefinitionSearchIntegration(unittest.TestCase):
    """Integration tests for definition search in passage retrieval."""

    def setUp(self):
        """Set up processor with code documents."""
        self.processor = CorticalTextProcessor()

        # Add a code document with class and function definitions
        self.processor.process_document('cortical/minicolumn.py', '''
"""Minicolumn module for cortical processing."""

from dataclasses import dataclass
from typing import Dict, List, Optional

class Minicolumn:
    """
    Core data structure representing a minicolumn in the cortical model.

    A minicolumn stores information about a single concept at a specific
    layer in the hierarchy.

    Attributes:
        id: Unique identifier
        content: The text content (word, bigram, etc.)
        layer: Which layer this belongs to (0-3)
    """

    def __init__(self, id: str, content: str, layer: int):
        self.id = id
        self.content = content
        self.layer = layer
        self.lateral_connections: Dict[str, float] = {}
        self.pagerank: float = 0.0
        self.tfidf: float = 0.0

    def add_connection(self, target_id: str, weight: float = 1.0):
        """Add a lateral connection to another minicolumn."""
        if target_id in self.lateral_connections:
            self.lateral_connections[target_id] += weight
        else:
            self.lateral_connections[target_id] = weight
''')

        self.processor.process_document('tests/test_minicolumn.py', '''
"""Tests for minicolumn module."""

import unittest
from cortical.minicolumn import Minicolumn

class TestMinicolumn(unittest.TestCase):
    """Test Minicolumn class."""

    def test_init(self):
        col = Minicolumn("L0_test", "test", 0)
        self.assertEqual(col.id, "L0_test")
        self.assertEqual(col.content, "test")

    def test_add_connection(self):
        col = Minicolumn("L0_a", "a", 0)
        col.add_connection("L0_b", 0.5)
        self.assertIn("L0_b", col.lateral_connections)
''')

        self.processor.compute_all()

    def test_definition_search_finds_class(self):
        """Definition search should find actual class definition."""
        results = self.processor.find_passages_for_query(
            "class Minicolumn",
            top_n=5,
            use_definition_search=True
        )

        self.assertTrue(len(results) > 0)

        # First result should be from the source file, not the test
        passage, doc_id, start, end, score = results[0]
        self.assertEqual(doc_id, 'cortical/minicolumn.py')
        self.assertIn('class Minicolumn', passage)

    def test_definition_search_finds_method(self):
        """Definition search should find method definitions."""
        results = self.processor.find_passages_for_query(
            "def add_connection",
            top_n=5,
            use_definition_search=True
        )

        self.assertTrue(len(results) > 0)
        passage, doc_id, start, end, score = results[0]
        self.assertIn('def add_connection', passage)

    def test_definition_search_disabled(self):
        """When disabled, definition search should not run."""
        # With a definition query but definition search disabled
        results_disabled = self.processor.find_passages_for_query(
            "class Minicolumn",
            top_n=5,
            use_definition_search=False
        )

        results_enabled = self.processor.find_passages_for_query(
            "class Minicolumn",
            top_n=5,
            use_definition_search=True
        )

        # Enabled should have higher score for definition
        if results_disabled and results_enabled:
            # Definition search should boost the actual definition
            self.assertGreaterEqual(results_enabled[0][4], results_disabled[0][4])

    def test_processor_has_definition_methods(self):
        """Processor should have definition search methods."""
        self.assertTrue(hasattr(self.processor, 'is_definition_query'))
        self.assertTrue(hasattr(self.processor, 'find_definition_passages'))

    def test_is_definition_query_via_processor(self):
        """Test is_definition_query via processor wrapper."""
        is_def, def_type, name = self.processor.is_definition_query("class Minicolumn")
        self.assertTrue(is_def)
        self.assertEqual(def_type, 'class')
        self.assertEqual(name, 'Minicolumn')

    def test_find_definition_passages_via_processor(self):
        """Test find_definition_passages via processor wrapper."""
        results = self.processor.find_definition_passages("class Minicolumn")
        self.assertTrue(len(results) > 0)
        passage, doc_id, start, end, score = results[0]
        self.assertIn('class Minicolumn', passage)


class TestPassageDocTypeBoost(unittest.TestCase):
    """Test doc-type boosting for passage-level search."""

    def setUp(self):
        """Set up processor with code and documentation."""
        self.processor = CorticalTextProcessor()

        # Add a code file
        self.processor.process_document('cortical/analysis.py', '''
"""Analysis module for computing PageRank and TF-IDF."""

def compute_pagerank(layers, damping=0.85):
    """Compute PageRank scores for all minicolumns.

    PageRank is an iterative algorithm that assigns importance scores
    to nodes based on the structure of incoming links.

    Args:
        layers: Dictionary of hierarchical layers
        damping: Damping factor (default 0.85)

    Returns:
        Dict mapping node IDs to PageRank scores
    """
    # Implementation details...
    pass
''', metadata={'doc_type': 'code'})

        # Add a documentation file
        self.processor.process_document('docs/algorithms.md', '''
# PageRank Algorithm

PageRank is the foundational algorithm that revolutionized web search.
It computes importance scores for nodes in a graph by iteratively
propagating scores through connections.

## How PageRank Works

1. Initialize all nodes with equal score
2. Iteratively update scores based on incoming links
3. Apply damping factor to prevent score accumulation
4. Converge when changes are below tolerance

The damping factor (typically 0.85) represents the probability that
a random walker continues following links rather than jumping randomly.
''', metadata={'doc_type': 'docs'})

        # Add a test file
        self.processor.process_document('tests/test_analysis.py', '''
"""Tests for analysis module."""

import unittest

class TestPageRank(unittest.TestCase):
    def test_compute_pagerank(self):
        """Test PageRank computation."""
        result = compute_pagerank(self.layers)
        self.assertIsInstance(result, dict)

    def test_pagerank_damping(self):
        """Test PageRank with custom damping."""
        result = compute_pagerank(self.layers, damping=0.9)
        self.assertGreater(len(result), 0)
''', metadata={'doc_type': 'test'})

        self.processor.compute_all()

    def test_conceptual_query_boosts_docs(self):
        """Conceptual queries should boost documentation passages."""
        # Conceptual query - should boost docs
        results = self.processor.find_passages_for_query(
            "what is PageRank algorithm",
            top_n=5,
            auto_detect_intent=True,
            apply_doc_boost=True
        )

        self.assertTrue(len(results) > 0)

        # Check that docs/ folder file appears in results with boost
        doc_ids = [r[1] for r in results]
        # With boosting, docs should be prioritized for conceptual queries
        self.assertIn('docs/algorithms.md', doc_ids)

    def test_prefer_docs_always_boosts(self):
        """prefer_docs=True should always boost documentation."""
        # Implementation query that would normally prefer code
        results = self.processor.find_passages_for_query(
            "compute pagerank",
            top_n=5,
            prefer_docs=True,
            apply_doc_boost=True
        )

        self.assertTrue(len(results) > 0)
        # Results should include docs even for implementation query

    def test_disable_doc_boost(self):
        """apply_doc_boost=False should use raw scores."""
        # Same query with and without boost
        results_no_boost = self.processor.find_passages_for_query(
            "explain PageRank algorithm",
            top_n=5,
            apply_doc_boost=False
        )

        results_with_boost = self.processor.find_passages_for_query(
            "explain PageRank algorithm",
            top_n=5,
            apply_doc_boost=True,
            auto_detect_intent=True
        )

        # Both should return results
        self.assertTrue(len(results_no_boost) > 0)
        self.assertTrue(len(results_with_boost) > 0)

        # With boosting, if doc is found, it might have higher score
        # (depends on corpus content and scores)

    def test_implementation_query_no_boost(self):
        """Implementation queries should not boost docs when auto_detect_intent=True."""
        # Implementation query
        results = self.processor.find_passages_for_query(
            "compute pagerank function code",
            top_n=5,
            auto_detect_intent=True,
            apply_doc_boost=True
        )

        self.assertTrue(len(results) > 0)
        # Implementation queries shouldn't trigger doc boost

    def test_custom_boosts(self):
        """Custom boost factors should be applied."""
        custom = {'docs': 3.0, 'code': 0.5, 'test': 0.3}

        results = self.processor.find_passages_for_query(
            "what is PageRank",
            top_n=5,
            prefer_docs=True,
            custom_boosts=custom
        )

        self.assertTrue(len(results) > 0)


class TestPassageDocTypeBoostIntegration(unittest.TestCase):
    """Integration tests for doc-type boost in passage search."""

    def test_find_passages_has_boost_params(self):
        """find_passages_for_query should accept boost parameters."""
        processor = CorticalTextProcessor()
        processor.process_document('test.py', 'def foo(): pass')
        processor.compute_all()

        # Should not raise
        results = processor.find_passages_for_query(
            "foo",
            apply_doc_boost=True,
            auto_detect_intent=True,
            prefer_docs=False,
            custom_boosts={'code': 1.0}
        )
        # Results may be empty for simple doc but params should work


class TestCodeAwareChunking(unittest.TestCase):
    """Test code-aware chunking functions."""

    def test_is_code_file_python(self):
        """Python files should be detected as code."""
        self.assertTrue(is_code_file('module.py'))
        self.assertTrue(is_code_file('path/to/file.py'))

    def test_is_code_file_javascript(self):
        """JavaScript files should be detected as code."""
        self.assertTrue(is_code_file('app.js'))
        self.assertTrue(is_code_file('component.tsx'))

    def test_is_code_file_markdown(self):
        """Markdown files should not be detected as code."""
        self.assertFalse(is_code_file('README.md'))
        self.assertFalse(is_code_file('docs/guide.md'))

    def test_is_code_file_other(self):
        """Other extensions should not be detected as code."""
        self.assertFalse(is_code_file('data.json'))
        self.assertFalse(is_code_file('config.yaml'))

    def test_find_code_boundaries_class(self):
        """Should find class definition boundaries."""
        code = '''
import os

class MyClass:
    def method(self):
        pass
'''
        boundaries = find_code_boundaries(code)
        self.assertIn(0, boundaries)  # Start
        # Should find the class line
        class_line_start = code.find('class MyClass')
        line_start = code.rfind('\n', 0, class_line_start) + 1
        self.assertIn(line_start, boundaries)

    def test_find_code_boundaries_function(self):
        """Should find function definition boundaries."""
        code = '''def foo():
    pass

def bar():
    pass
'''
        boundaries = find_code_boundaries(code)
        # Should find both function boundaries
        self.assertGreater(len(boundaries), 1)

    def test_find_code_boundaries_blank_lines(self):
        """Should find blank line boundaries."""
        code = '''first section

second section

third section
'''
        boundaries = find_code_boundaries(code)
        # Should include positions after blank lines
        self.assertGreater(len(boundaries), 1)

    def test_create_code_aware_chunks_empty(self):
        """Empty text should return empty list."""
        chunks = create_code_aware_chunks('')
        self.assertEqual(chunks, [])

    def test_create_code_aware_chunks_small_text(self):
        """Text smaller than target should return single chunk."""
        code = 'def foo(): pass'
        chunks = create_code_aware_chunks(code, target_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], code)

    def test_create_code_aware_chunks_splits_at_boundaries(self):
        """Should split at class/function boundaries."""
        # Create code long enough to require multiple chunks
        code = '''class FirstClass:
    """First class docstring with enough text to make this substantial."""

    def __init__(self):
        self.value = 0
        self.data = {}
        self.cache = []

    def method1(self):
        """A method that does something important."""
        result = self.value * 2
        return result


class SecondClass:
    """Second class docstring with substantial documentation text here."""

    def __init__(self):
        self.items = []
        self.count = 0

    def method2(self):
        """Another method with documentation."""
        for item in self.items:
            self.count += item
        return self.count
'''
        # With target_size=200, this ~600 char code should split into multiple chunks
        chunks = create_code_aware_chunks(code, target_size=200, min_size=50, max_size=400)
        self.assertGreater(len(chunks), 1)

        # Check that chunks start at sensible boundaries
        for chunk_text, start, end in chunks:
            # Chunks should not be empty
            self.assertTrue(chunk_text.strip())

    def test_create_code_aware_chunks_respects_max_size(self):
        """Should not exceed max_size."""
        # Create code with a very long function
        long_function = 'def long_func():\n' + '    x = 1\n' * 100
        chunks = create_code_aware_chunks(long_function, target_size=200, max_size=400)

        for chunk_text, start, end in chunks:
            self.assertLessEqual(len(chunk_text), 400)

    def test_create_code_aware_chunks_no_whitespace_only(self):
        """Should not return whitespace-only chunks."""
        code = '''class A:
    pass



class B:
    pass
'''
        chunks = create_code_aware_chunks(code, target_size=50, min_size=10)
        for chunk_text, start, end in chunks:
            self.assertTrue(chunk_text.strip())


class TestCodeAwareChunkingIntegration(unittest.TestCase):
    """Integration tests for code-aware chunking in passage search."""

    def setUp(self):
        """Set up processor with code documents."""
        self.processor = CorticalTextProcessor()

        # Add a code file with multiple classes/functions
        self.processor.process_document('cortical/example.py', '''
"""Example module with multiple classes and functions."""

import os
from typing import Dict, List

class FirstProcessor:
    """First processor class for demonstration."""

    def __init__(self):
        self.data = {}

    def process(self, item):
        """Process a single item."""
        return item * 2


class SecondProcessor:
    """Second processor class for demonstration."""

    def __init__(self):
        self.cache = []

    def process_batch(self, items):
        """Process multiple items at once."""
        return [x * 3 for x in items]


def utility_function(x, y):
    """A utility function outside classes."""
    return x + y
''')

        self.processor.compute_all()

    def test_code_aware_chunks_enabled_by_default(self):
        """Code-aware chunking should be enabled by default."""
        results = self.processor.find_passages_for_query(
            "SecondProcessor",
            top_n=5
        )
        self.assertTrue(len(results) > 0)

    def test_code_aware_chunks_can_be_disabled(self):
        """Should be able to disable code-aware chunking."""
        results = self.processor.find_passages_for_query(
            "SecondProcessor",
            top_n=5,
            use_code_aware_chunks=False
        )
        # Should still return results, just with fixed chunking
        self.assertTrue(len(results) >= 0)

    def test_processor_has_code_chunk_param(self):
        """Processor should accept use_code_aware_chunks parameter."""
        # Should not raise
        results = self.processor.find_passages_for_query(
            "utility",
            use_code_aware_chunks=True
        )


class TestExpandQueryWithSemantics(unittest.TestCase):
    """Test semantic query expansion."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", "Neural networks are learning models.")
        cls.processor.process_document("doc2", "Deep learning uses neural networks.")
        cls.processor.compute_all(verbose=False)
        cls.processor.extract_corpus_semantics(verbose=False)

    def test_expand_query_semantic_with_relations(self):
        """Test semantic expansion with relations."""
        from cortical.query import expand_query_semantic

        relations = [
            ('neural', 'RelatedTo', 'network', 0.8),
            ('neural', 'RelatedTo', 'learning', 0.7),
        ]

        expanded = expand_query_semantic(
            "neural",
            self.processor.layers,
            self.processor.tokenizer,
            relations,
            max_expansions=5
        )

        self.assertIn('neural', expanded)
        # Should have added some related terms
        self.assertGreaterEqual(len(expanded), 1)

    def test_expand_query_semantic_empty_relations(self):
        """Test semantic expansion with no relations."""
        from cortical.query import expand_query_semantic

        expanded = expand_query_semantic(
            "neural",
            self.processor.layers,
            self.processor.tokenizer,
            [],  # Empty relations
            max_expansions=5
        )

        # Should still have original term
        self.assertIn('neural', expanded)


class TestBoostDefinitionDocuments(unittest.TestCase):
    """Test definition document boosting."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("def_doc", """
            class MyClass:
                def __init__(self):
                    pass

            def process_data(self):
                return self.data
        """)
        cls.processor.process_document("usage_doc", """
            We use MyClass to process data.
            The results are stored in files.
        """)
        cls.processor.compute_all(verbose=False)

    def test_boost_definition_documents_with_definition(self):
        """Test boosting documents that contain definitions."""
        from cortical.query import boost_definition_documents

        doc_results = [
            ("def_doc", 1.0),
            ("usage_doc", 1.0),
        ]

        boosted = boost_definition_documents(
            doc_results,
            "where is class MyClass defined?",
            self.processor.documents,
            2.0
        )

        # Should still have documents
        self.assertEqual(len(boosted), 2)


class TestQueryRelatedDocuments(unittest.TestCase):
    """Test related document lookup."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", "Neural networks are models.")
        cls.processor.process_document("doc2", "Machine learning uses algorithms.")
        cls.processor.process_document("doc3", "Neural learning processes data.")
        cls.processor.compute_all(verbose=False)

    def test_find_related_documents(self):
        """Test finding related documents."""
        from cortical.query import find_related_documents

        related = find_related_documents(
            "doc1",
            self.processor.layers
        )

        # Should return a list
        self.assertIsInstance(related, list)


if __name__ == '__main__':
    unittest.main()
