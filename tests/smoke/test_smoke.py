"""
Smoke Tests - Quick Sanity Checks
=================================

These tests verify that the system fundamentally works.
They should complete in < 10 seconds total and catch critical breakage early.

If smoke tests fail, there's likely a critical issue that will affect everything.
Fix smoke test failures before investigating other test failures.

Run with: pytest tests/smoke/ -v
"""

import pytest


class TestCoreImports:
    """Verify core modules can be imported."""

    def test_import_cortical_package(self):
        """Main package imports successfully."""
        import cortical
        assert hasattr(cortical, 'CorticalTextProcessor')
        assert hasattr(cortical, 'CorticalLayer')

    def test_import_processor(self):
        """Processor module imports."""
        from cortical import CorticalTextProcessor
        assert CorticalTextProcessor is not None

    def test_import_analysis(self):
        """Analysis module imports."""
        from cortical import analysis
        assert hasattr(analysis, 'compute_pagerank')
        assert hasattr(analysis, 'compute_tfidf')

    def test_import_query(self):
        """Query module imports."""
        from cortical import query
        assert hasattr(query, 'find_documents_for_query')

    def test_import_tokenizer(self):
        """Tokenizer module imports."""
        from cortical.tokenizer import Tokenizer
        assert Tokenizer is not None


class TestProcessorCreation:
    """Verify processor can be created and used."""

    def test_create_empty_processor(self):
        """Empty processor can be instantiated."""
        from cortical import CorticalTextProcessor
        processor = CorticalTextProcessor()
        assert processor is not None
        assert len(processor.documents) == 0

    def test_create_with_config(self):
        """Processor accepts configuration."""
        from cortical import CorticalTextProcessor
        from cortical.config import CorticalConfig

        config = CorticalConfig(pagerank_damping=0.9)
        processor = CorticalTextProcessor(config=config)
        assert processor.config.pagerank_damping == 0.9

    def test_create_with_tokenizer(self):
        """Processor accepts custom tokenizer."""
        from cortical import CorticalTextProcessor
        from cortical.tokenizer import Tokenizer

        tokenizer = Tokenizer(filter_code_noise=True)
        processor = CorticalTextProcessor(tokenizer=tokenizer)
        assert processor is not None


class TestBasicWorkflow:
    """Verify the basic processing workflow works."""

    def test_process_single_document(self):
        """Single document can be processed."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        stats = processor.process_document("test", "Hello world test document.")

        assert stats['tokens'] > 0
        assert "test" in processor.documents

    def test_process_multiple_documents(self):
        """Multiple documents can be processed."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "First document content.")
        processor.process_document("doc2", "Second document content.")

        assert len(processor.documents) == 2

    def test_compute_all_completes(self):
        """compute_all() completes without error."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("test", "Test document for computation.")
        processor.compute_all(verbose=False)

        # Verify some computation happened
        from cortical import CorticalLayer
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() > 0


class TestBasicSearch:
    """Verify search functionality works."""

    def test_search_returns_results(self, small_processor):
        """Search returns results from corpus."""
        results = small_processor.find_documents_for_query("machine learning", top_n=5)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_empty_query_raises(self, small_processor):
        """Empty query raises ValueError."""
        with pytest.raises(ValueError):
            small_processor.find_documents_for_query("", top_n=5)

    def test_query_expansion_works(self, small_processor):
        """Query expansion returns related terms."""
        expanded = small_processor.expand_query("database", max_expansions=10)

        assert isinstance(expanded, dict)
        assert "database" in expanded or len(expanded) > 0


class TestBasicPersistence:
    """Verify save/load functionality works."""

    def test_save_and_load(self, tmp_path, small_processor):
        """Processor can be saved and loaded."""
        from cortical import CorticalTextProcessor

        save_path = tmp_path / "test_corpus.pkl"

        # Save
        small_processor.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded = CorticalTextProcessor.load(str(save_path))
        assert len(loaded.documents) == len(small_processor.documents)


class TestLayerAccess:
    """Verify layer access works correctly."""

    def test_get_all_layers(self, small_processor):
        """All four layers are accessible."""
        from cortical import CorticalLayer

        for layer_type in CorticalLayer:
            layer = small_processor.get_layer(layer_type)
            assert layer is not None

    def test_token_layer_has_content(self, small_processor):
        """Token layer contains minicolumns."""
        from cortical import CorticalLayer

        layer0 = small_processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() > 0

    def test_document_layer_has_content(self, small_processor):
        """Document layer contains all documents."""
        from cortical import CorticalLayer

        layer3 = small_processor.get_layer(CorticalLayer.DOCUMENTS)
        assert layer3.column_count() == len(small_processor.documents)
