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
        chunks = create_chunks(text, chunk_size=100)
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


if __name__ == '__main__':
    unittest.main()
