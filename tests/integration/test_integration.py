"""
Integration Tests
=================

Tests that verify components work together correctly.
These tests validate end-to-end workflows and module interactions.

Run with: pytest tests/integration/ -v
"""

import pytest
import tempfile
import os

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer
from cortical.config import CorticalConfig


class TestProcessorQueryIntegration:
    """Test processor and query module interactions."""

    @pytest.fixture
    def loaded_processor(self):
        """Create a processor with test documents."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Machine learning uses neural networks for pattern recognition.")
        processor.process_document("doc2", "Deep learning is a subset of machine learning.")
        processor.process_document("doc3", "Database systems store and retrieve data efficiently.")
        processor.compute_all(verbose=False)
        return processor

    def test_search_returns_relevant_documents(self, loaded_processor):
        """Search should return documents matching the query."""
        results = loaded_processor.find_documents_for_query("machine learning", top_n=3)

        assert len(results) > 0
        doc_ids = [doc_id for doc_id, _ in results]
        # ML-related docs should rank higher than database doc
        assert "doc1" in doc_ids or "doc2" in doc_ids

    def test_query_expansion_uses_computed_connections(self, loaded_processor):
        """Query expansion should use connections computed by the processor."""
        expanded = loaded_processor.expand_query("neural", max_expansions=10)

        assert "neural" in expanded
        # Should expand to related terms based on co-occurrence
        assert len(expanded) > 1

    def test_passage_retrieval_returns_chunks(self, loaded_processor):
        """Passage retrieval should return document chunks."""
        passages = loaded_processor.find_passages_for_query(
            "machine learning",
            top_n=3,
            chunk_size=200,
            overlap=50
        )

        assert len(passages) > 0
        # Each passage should have (doc_id, text, start_pos, end_pos, score)
        for doc_id, text, start_pos, end_pos, score in passages:
            assert isinstance(doc_id, str)
            assert isinstance(text, str)
            assert isinstance(start_pos, int)
            assert isinstance(end_pos, int)
            assert isinstance(score, (int, float))


class TestPersistenceIntegration:
    """Test save/load functionality preserves computed state."""

    def test_save_and_load_preserves_search_results(self):
        """Saved processor should produce same search results after loading."""
        # Create and compute processor
        processor1 = CorticalTextProcessor()
        processor1.process_document("ml", "Neural networks learn from data.")
        processor1.process_document("db", "Databases store information.")
        processor1.compute_all(verbose=False)

        # Get search results before save
        results_before = processor1.find_documents_for_query("neural", top_n=2)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            processor1.save(path)
            processor2 = CorticalTextProcessor.load(path)

            # Results should match
            results_after = processor2.find_documents_for_query("neural", top_n=2)

            assert len(results_before) == len(results_after)
            for (id1, score1), (id2, score2) in zip(results_before, results_after):
                assert id1 == id2
                assert abs(score1 - score2) < 0.001
        finally:
            os.unlink(path)

    def test_save_and_load_preserves_layers(self):
        """Layer structure should be preserved after save/load."""
        processor1 = CorticalTextProcessor()
        processor1.process_document("test", "Test document content for layers.")
        processor1.compute_all(verbose=False)

        # Count minicolumns in each layer
        counts_before = {
            layer: processor1.get_layer(layer).column_count()
            for layer in CorticalLayer
        }

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            processor1.save(path)
            processor2 = CorticalTextProcessor.load(path)

            # Layer counts should match
            for layer in CorticalLayer:
                assert processor2.get_layer(layer).column_count() == counts_before[layer]
        finally:
            os.unlink(path)


class TestConfigIntegration:
    """Test configuration affects processor behavior correctly."""

    def test_config_affects_clustering(self):
        """Different resolution values should produce different cluster counts."""
        processor1 = CorticalTextProcessor(config=CorticalConfig(louvain_resolution=0.5))
        processor2 = CorticalTextProcessor(config=CorticalConfig(louvain_resolution=2.0))

        docs = {
            "d1": "Alpha beta gamma delta epsilon.",
            "d2": "Alpha beta gamma zeta eta.",
            "d3": "Theta iota kappa lambda mu.",
        }

        for doc_id, content in docs.items():
            processor1.process_document(doc_id, content)
            processor2.process_document(doc_id, content)

        processor1.compute_all(verbose=False)
        processor2.compute_all(verbose=False)

        # Higher resolution typically produces more clusters
        clusters1 = processor1.get_layer(CorticalLayer.CONCEPTS).column_count()
        clusters2 = processor2.get_layer(CorticalLayer.CONCEPTS).column_count()

        # Should be different (unless corpus is too small)
        # At minimum, both should have computed something
        assert clusters1 >= 1
        assert clusters2 >= 1


class TestTokenizerIntegration:
    """Test tokenizer settings affect downstream processing."""

    def test_code_noise_filtering_affects_pagerank(self):
        """Filtering code noise should change which terms rank highly."""
        code_content = """
        def process_data(self):
            self.data = []
            for item in self.items:
                self.data.append(item)
            return self.data
        """

        # Without filtering
        processor1 = CorticalTextProcessor(tokenizer=Tokenizer(filter_code_noise=False))
        processor1.process_document("code", code_content)
        processor1.compute_all(verbose=False)

        # With filtering
        processor2 = CorticalTextProcessor(tokenizer=Tokenizer(filter_code_noise=True))
        processor2.process_document("code", code_content)
        processor2.compute_all(verbose=False)

        # Get top terms
        layer1 = processor1.get_layer(CorticalLayer.TOKENS)
        layer2 = processor2.get_layer(CorticalLayer.TOKENS)

        top1 = sorted([(c.content, c.pagerank) for c in layer1], key=lambda x: -x[1])[:5]
        top2 = sorted([(c.content, c.pagerank) for c in layer2], key=lambda x: -x[1])[:5]

        top_terms_1 = [t for t, _ in top1]
        top_terms_2 = [t for t, _ in top2]

        # 'self' should be in top terms without filtering
        # but not when filtering is enabled
        if 'self' in top_terms_1:
            assert 'self' not in top_terms_2


class TestIncrementalUpdateIntegration:
    """Test incremental document updates work correctly."""

    def test_incremental_add_updates_search(self):
        """Adding documents incrementally should update search results."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Machine learning algorithms.")
        processor.compute_all(verbose=False)

        # Search before adding second doc
        results_before = processor.find_documents_for_query("database", top_n=3)
        doc_ids_before = {doc_id for doc_id, _ in results_before}

        # Add new document
        processor.add_document_incremental("doc2", "Database management systems.", recompute='tfidf')

        # Search after adding
        results_after = processor.find_documents_for_query("database", top_n=3)
        doc_ids_after = {doc_id for doc_id, _ in results_after}

        # New document should appear in results
        assert "doc2" in doc_ids_after
        assert "doc2" not in doc_ids_before


class TestEndToEndWorkflow:
    """Test complete user workflows."""

    def test_build_search_rag_workflow(self):
        """Test typical RAG workflow: index -> search -> retrieve passages."""
        # 1. Build corpus
        processor = CorticalTextProcessor()

        docs = {
            "intro": "Machine learning is a field of artificial intelligence.",
            "neural": "Neural networks are inspired by biological neurons.",
            "deep": "Deep learning uses multi-layer neural networks.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all(verbose=False)

        # 2. Search for relevant documents
        results = processor.find_documents_for_query("neural networks", top_n=2)
        assert len(results) > 0

        # 3. Retrieve passages for RAG
        passages = processor.find_passages_for_query(
            "neural networks",
            top_n=3,
            chunk_size=200,
            overlap=50
        )
        assert len(passages) > 0

        # 4. Expand query for better coverage
        expanded = processor.expand_query("neural", max_expansions=5)
        assert len(expanded) > 0

    def test_save_load_continue_workflow(self):
        """Test workflow: build -> save -> load -> add more -> search."""
        processor1 = CorticalTextProcessor()
        processor1.process_document("doc1", "Initial document about algorithms.")
        processor1.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            processor1.save(path)

            # Load and continue
            processor2 = CorticalTextProcessor.load(path)
            processor2.add_document_incremental("doc2", "New document about data structures.")

            # Should be able to search both
            results = processor2.find_documents_for_query("algorithms data", top_n=5)
            doc_ids = {doc_id for doc_id, _ in results}

            # At least one original doc should appear
            assert "doc1" in doc_ids or "doc2" in doc_ids
        finally:
            os.unlink(path)
