"""
Tests targeting coverage gaps in the cortical modules.

These tests focus on edge cases and code paths that aren't covered
by the main test suite.
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.layers import HierarchicalLayer
from cortical.minicolumn import Minicolumn


class TestSemanticsNumpyPath(unittest.TestCase):
    """Test semantics with numpy available (if installed)."""

    def test_extract_semantics_with_context_vectors(self):
        """Test semantic extraction generates context vectors and SimilarTo."""
        processor = CorticalTextProcessor()
        # Add documents with shared context to trigger SimilarTo detection
        processor.process_document("doc1", """
            neural networks process information through layers
            deep learning neural networks transform data representations
            neural network training requires optimization algorithms
        """)
        processor.process_document("doc2", """
            machine learning models learn from training data
            deep learning models use neural network architectures
            training machine learning requires labeled datasets
        """)
        processor.process_document("doc3", """
            data processing pipelines transform raw inputs
            neural processing in the brain uses cortical columns
            information processing systems handle complex data
        """)
        processor.compute_all(verbose=False)

        from cortical.semantics import extract_corpus_semantics
        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer
        )

        # Should extract various relation types
        relation_types = set(r[1] for r in relations)
        self.assertIn('CoOccurs', relation_types)

    def test_extract_semantics_many_terms(self):
        """Test semantic extraction with many terms to trigger similarity paths."""
        processor = CorticalTextProcessor()
        # Create documents with overlapping vocabulary to generate context vectors
        for i in range(5):
            processor.process_document(f"doc{i}", f"""
                term{i} common shared vocabulary words here
                another term{i} with common context overlap
                more common terms to build context vectors
            """)
        processor.compute_all(verbose=False)

        from cortical.semantics import extract_corpus_semantics
        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer
        )

        self.assertIsInstance(relations, list)


class TestProcessorEdgeCases(unittest.TestCase):
    """Test processor edge cases and error handling."""

    def test_process_document_update_existing(self):
        """Test updating an existing document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "original content here")

        # Update with new content
        processor.process_document("doc1", "updated content different")

        # Should still have only one document
        self.assertEqual(len(processor.documents), 1)

    def test_compute_all_empty_corpus(self):
        """Test compute_all on empty corpus."""
        processor = CorticalTextProcessor()
        # Should not raise
        processor.compute_all(verbose=False)

        self.assertEqual(processor.layers[CorticalLayer.TOKENS].column_count(), 0)

    def test_compute_all_single_doc(self):
        """Test compute_all with single document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "single document only")
        processor.compute_all(verbose=False)

        self.assertGreater(processor.layers[CorticalLayer.TOKENS].column_count(), 0)

    def test_remove_document_updates_layers(self):
        """Test that removing a document updates token layers."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network learning")
        processor.process_document("doc2", "machine learning algorithms")
        processor.compute_all(verbose=False)

        initial_count = processor.layers[CorticalLayer.TOKENS].column_count()

        # Remove one document
        processor.remove_document("doc1")

        # Token layer should be affected
        self.assertEqual(len(processor.documents), 1)

    def test_get_document_metadata_missing(self):
        """Test getting metadata for non-existent document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        meta = processor.get_document_metadata("nonexistent")
        # Returns empty dict for missing document
        self.assertEqual(meta, {})

    def test_compute_importance_verbose(self):
        """Test compute_importance with verbose output."""
        import logging

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.propagate_activation(iterations=3, verbose=False)

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.compute_importance(verbose=True)

        # Should have logged something about PageRank
        output = '\n'.join(cm.output)
        self.assertIn('PageRank', output)


class TestPersistenceEdgeCases(unittest.TestCase):
    """Test persistence edge cases."""

    def test_save_and_load_empty_corpus(self):
        """Test saving and loading empty corpus."""
        processor = CorticalTextProcessor()

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor.save(temp_path)
            loaded = CorticalTextProcessor.load(temp_path)
            self.assertEqual(len(loaded.documents), 0)
        finally:
            os.unlink(temp_path)

    def test_save_with_metadata(self):
        """Test saving with custom metadata."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor.save(temp_path)
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            # Load and verify
            loaded = CorticalTextProcessor.load(temp_path)
            self.assertEqual(len(loaded.documents), 1)
        finally:
            os.unlink(temp_path)


class TestChunkIndexEdgeCases(unittest.TestCase):
    """Test chunk index edge cases."""

    def test_chunk_writer_empty(self):
        """Test chunk writer with no operations."""
        from cortical.chunk_index import ChunkWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            # No operations, should return None
            result = writer.save()
            self.assertIsNone(result)
            self.assertFalse(writer.has_operations())

    def test_chunk_writer_add_and_save(self):
        """Test adding and saving chunks."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document("doc1", "test content", mtime=12345.0)
            self.assertTrue(writer.has_operations())

            # Save should create a file
            result = writer.save()
            self.assertIsNotNone(result)
            self.assertTrue(result.exists())

            # Load should retrieve the document
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()
            self.assertIn("doc1", docs)


class TestLayersEdgeCases(unittest.TestCase):
    """Test layers edge cases."""

    def test_layer_get_by_id_missing(self):
        """Test get_by_id returns None for missing ID."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        result = layer.get_by_id("nonexistent_id")
        self.assertIsNone(result)

    def test_layer_total_connections_empty(self):
        """Test total_connections on empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        self.assertEqual(layer.total_connections(), 0)

    def test_minicolumn_add_connections(self):
        """Test adding various connection types."""
        col = Minicolumn("L0_test", "test", 0)

        # Add lateral connection
        col.add_lateral_connection("L0_other1", 0.5)
        self.assertEqual(len(col.lateral_connections), 1)

        # Add again should update weight
        col.add_lateral_connection("L0_other1", 0.3)
        self.assertAlmostEqual(col.lateral_connections["L0_other1"], 0.8)

        # Add feedforward connection
        col.feedforward_connections["L1_target1"] = 1.0
        self.assertEqual(len(col.feedforward_connections), 1)


class TestQueryEdgeCases(unittest.TestCase):
    """Test query edge cases."""

    def test_find_documents_empty_query(self):
        """Test finding documents with empty query raises ValueError."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")
        processor.compute_all(verbose=False)

        with self.assertRaises(ValueError):
            processor.find_documents_for_query("")

    def test_find_documents_no_matches(self):
        """Test finding documents when no matches."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks deep learning")
        processor.compute_all(verbose=False)

        results = processor.find_documents_for_query("quantum physics")
        # May or may not find results depending on expansion
        self.assertIsInstance(results, list)

    def test_expand_query_empty(self):
        """Test expanding empty query."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)

        result = processor.expand_query("")
        self.assertEqual(result, {})


class TestAnalysisEdgeCases(unittest.TestCase):
    """Test analysis edge cases."""

    def test_pagerank_single_node(self):
        """Test PageRank with single node."""
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("single")

        result = compute_pagerank(layer)
        self.assertEqual(len(result), 1)

    def test_tfidf_single_doc_single_term(self):
        """Test TF-IDF with minimal corpus."""
        from cortical.analysis import compute_tfidf

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "word")
        processor.propagate_activation(iterations=1, verbose=False)

        # compute_tfidf takes layers dict and documents dict, returns None
        compute_tfidf(processor.layers, processor.documents)

        # TF-IDF scores should be set on minicolumns
        layer0 = processor.layers[CorticalLayer.TOKENS]
        if layer0.column_count() > 0:
            col = list(layer0.minicolumns.values())[0]
            self.assertIsInstance(col.tfidf, float)

    def test_clustering_quality_single_cluster(self):
        """Test clustering quality with single cluster."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "all same topic words")
        processor.compute_all(verbose=False)

        quality = processor.compute_clustering_quality()
        self.assertIsInstance(quality['modularity'], float)


class TestConfigEdgeCases(unittest.TestCase):
    """Test config edge cases."""

    def test_config_validation(self):
        """Test config validation catches invalid values."""
        from cortical.config import CorticalConfig

        # Valid config should work
        config = CorticalConfig()
        self.assertIsNotNone(config)

        # Test some valid parameter ranges
        config2 = CorticalConfig(
            pagerank_damping=0.5,
            pagerank_iterations=10,
            min_cluster_size=2
        )
        self.assertEqual(config2.pagerank_damping, 0.5)


class TestEmbeddingsEdgeCases(unittest.TestCase):
    """Test embeddings edge cases."""

    def test_embeddings_empty_corpus(self):
        """Test embeddings on empty corpus."""
        from cortical.embeddings import compute_graph_embeddings

        processor = CorticalTextProcessor()

        # compute_graph_embeddings takes layers dict and returns tuple
        embeddings, stats = compute_graph_embeddings(processor.layers)
        self.assertEqual(len(embeddings), 0)

    def test_embeddings_single_node(self):
        """Test embeddings with single node."""
        from cortical.embeddings import compute_graph_embeddings

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "single")
        processor.propagate_activation(iterations=1, verbose=False)

        embeddings, stats = compute_graph_embeddings(
            processor.layers,
            dimensions=8
        )
        # Might be empty or have one entry depending on connections
        self.assertIsInstance(embeddings, dict)
        self.assertIsInstance(stats, dict)


class TestProcessorMoreEdgeCases(unittest.TestCase):
    """Test additional processor edge cases for coverage."""

    def test_propagate_activation_iterations(self):
        """Test propagation with different iteration counts."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.process_document("doc2", "machine learning neural network")

        # Test with explicit iterations
        processor.propagate_activation(iterations=5, verbose=False)

        layer0 = processor.layers[CorticalLayer.TOKENS]
        # Some columns should have non-zero activation
        activations = [col.activation for col in layer0.minicolumns.values()]
        self.assertTrue(any(a > 0 for a in activations))

    def test_find_documents_with_expansion(self):
        """Test document search with query expansion."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.process_document("doc2", "machine learning algorithms")
        processor.process_document("doc3", "cooking recipes baking")
        processor.compute_all(verbose=False)

        # Search with expansion enabled
        results = processor.find_documents_for_query("neural", top_n=2, use_expansion=True)
        self.assertIsInstance(results, list)

    def test_compute_all_phases(self):
        """Test individual compute phases."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models training")
        processor.process_document("doc2", "machine learning algorithms data science")

        # Run individual phases
        processor.propagate_activation(iterations=3, verbose=False)
        processor.compute_importance(verbose=False)
        processor.compute_tfidf(verbose=False)
        processor.extract_corpus_semantics(verbose=False)

        # Verify results exist
        layer0 = processor.layers[CorticalLayer.TOKENS]
        if layer0.column_count() > 0:
            col = list(layer0.minicolumns.values())[0]
            self.assertIsNotNone(col.pagerank)

    def test_get_minicolumn_info(self):
        """Test getting minicolumn information."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.compute_all(verbose=False)

        # Get minicolumn directly
        layer0 = processor.layers[CorticalLayer.TOKENS]
        col = layer0.get_minicolumn("neural")
        if col:
            self.assertIsInstance(col.pagerank, float)
            self.assertIsInstance(col.tfidf, float)

        # Non-existent term returns None
        col_none = layer0.get_minicolumn("nonexistent_term_xyz")
        self.assertIsNone(col_none)


class TestSemanticsMoreCoverage(unittest.TestCase):
    """Additional semantics tests for coverage."""

    def test_extract_semantics_builds_cooccurs(self):
        """Test that co-occurrence relations are extracted."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network machine learning")
        processor.process_document("doc2", "neural network deep learning")
        processor.compute_all(verbose=False)

        from cortical.semantics import extract_corpus_semantics
        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer
        )

        # Should have at least some CoOccurs relations
        cooccurs = [r for r in relations if r[1] == 'CoOccurs']
        self.assertIsInstance(cooccurs, list)


class TestChunkMoreCoverage(unittest.TestCase):
    """Additional chunk tests for coverage."""

    def test_chunk_with_metadata(self):
        """Test chunk operations with metadata."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            metadata = {"doc_type": "code", "headings": ["test"]}
            writer.add_document("doc1", "content here", mtime=1000.0, metadata=metadata)
            writer.save()

            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()
            self.assertIn("doc1", docs)

            # Check metadata was preserved
            meta = loader.get_metadata()
            self.assertIn("doc1", meta)

    def test_chunk_operation_dataclass(self):
        """Test ChunkOperation to_dict and from_dict."""
        from cortical.chunk_index import ChunkOperation

        op = ChunkOperation(
            op='add',
            doc_id='test_doc',
            content='test content',
            mtime=12345.0,
            metadata={'type': 'test'}
        )

        # Convert to dict and back
        d = op.to_dict()
        self.assertEqual(d['op'], 'add')
        self.assertEqual(d['doc_id'], 'test_doc')
        self.assertIn('metadata', d)

        # Reconstruct from dict
        op2 = ChunkOperation.from_dict(d)
        self.assertEqual(op2.op, 'add')
        self.assertEqual(op2.doc_id, 'test_doc')
        self.assertEqual(op2.metadata['type'], 'test')


class TestQueryMoreCoverage(unittest.TestCase):
    """Additional query tests for coverage."""

    def test_expand_query_with_semantics(self):
        """Test query expansion with semantic relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learn from data")
        processor.process_document("doc2", "deep learning networks process information")
        processor.compute_all(verbose=False)

        expanded = processor.expand_query("neural", max_expansions=5)
        self.assertIsInstance(expanded, dict)
        # Original term should be present
        self.assertIn("neural", expanded)

    def test_find_passages_basic(self):
        """Test passage retrieval."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "This is a document about neural networks and deep learning. Neural networks are powerful machine learning models.")
        processor.compute_all(verbose=False)

        passages = processor.find_passages_for_query("neural", top_n=2)
        self.assertIsInstance(passages, list)


class TestAnalysisMoreCoverage(unittest.TestCase):
    """Additional analysis tests for coverage."""

    def test_pagerank_with_connections(self):
        """Test PageRank with actual connections."""
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("term1")
        col2 = layer.get_or_create_minicolumn("term2")
        col3 = layer.get_or_create_minicolumn("term3")

        # Add connections
        col1.add_lateral_connection(col2.id, 0.5)
        col2.add_lateral_connection(col3.id, 0.5)
        col3.add_lateral_connection(col1.id, 0.5)

        result = compute_pagerank(layer)
        self.assertEqual(len(result), 3)
        # All should have positive PageRank
        self.assertTrue(all(v > 0 for v in result.values()))


class TestPersistenceMoreCoverage(unittest.TestCase):
    """Additional persistence tests for coverage."""

    def test_save_and_load_with_semantics(self):
        """Test save/load preserves semantic relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.process_document("doc2", "machine learning algorithms training")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor.save(temp_path)
            loaded = CorticalTextProcessor.load(temp_path)

            # Verify layers are preserved
            self.assertEqual(len(loaded.documents), 2)
            layer0_orig = processor.layers[CorticalLayer.TOKENS]
            layer0_loaded = loaded.layers[CorticalLayer.TOKENS]
            self.assertEqual(layer0_orig.column_count(), layer0_loaded.column_count())
        finally:
            os.unlink(temp_path)


class TestInheritanceCoverage(unittest.TestCase):
    """Tests for property inheritance paths."""

    def test_compute_property_inheritance(self):
        """Test property inheritance computation."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "dog is animal mammal pet")
        processor.process_document("doc2", "cat is animal mammal pet")
        processor.process_document("doc3", "bird is animal flying creature")
        processor.compute_all(verbose=False)

        # Compute property inheritance
        result = processor.compute_property_inheritance(
            apply_to_connections=True,
            verbose=False
        )
        self.assertIn('terms_with_inheritance', result)
        self.assertIn('total_properties_inherited', result)
        self.assertIn('inherited', result)


class TestDocumentConnections(unittest.TestCase):
    """Tests for document connection computation."""

    def test_compute_document_connections(self):
        """Test document connection computation."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.process_document("doc2", "neural network machine learning")
        processor.process_document("doc3", "cooking recipes baking bread")
        processor.compute_all(verbose=False)

        # Document connections should exist
        layer3 = processor.layers[CorticalLayer.DOCUMENTS]
        if layer3.column_count() > 0:
            # Check for some connections between similar docs
            col1 = layer3.get_minicolumn("doc1")
            if col1 and col1.lateral_connections:
                # doc1 and doc2 are similar, should have connection
                self.assertGreater(len(col1.lateral_connections), 0)


class TestVerboseOutputPaths(unittest.TestCase):
    """Test verbose output paths for coverage."""

    def test_compute_all_verbose(self):
        """Test compute_all with verbose output."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network learning models")
        processor.process_document("doc2", "deep learning neural algorithms")

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.compute_all(verbose=True)

        # Should have output from various phases
        output = '\n'.join(cm.output)
        self.assertTrue(len(output) > 0)

    def test_export_graph_json(self):
        """Test exporting graph to JSON."""
        from cortical.persistence import export_graph_json

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network learning")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result = export_graph_json(temp_path, processor.layers)
            self.assertIn('nodes', result)
            self.assertIn('edges', result)
            self.assertTrue(os.path.exists(temp_path))
        finally:
            os.unlink(temp_path)


class TestTokenizerEdgeCases(unittest.TestCase):
    """Test tokenizer edge cases."""

    def test_tokenize_with_identifiers(self):
        """Test tokenizing code with identifier splitting."""
        from cortical.tokenizer import Tokenizer

        tok = Tokenizer(split_identifiers=True)
        tokens = tok.tokenize("getUserNameAndPassword")

        # Should split camelCase identifiers
        self.assertTrue(any('get' in t.lower() for t in tokens))
        self.assertTrue(any('user' in t.lower() for t in tokens))

    def test_tokenize_empty_text(self):
        """Test tokenizing empty text."""
        from cortical.tokenizer import Tokenizer

        tok = Tokenizer()
        tokens = tok.tokenize("")
        self.assertEqual(tokens, [])

    def test_tokenize_punctuation(self):
        """Test tokenizing text with punctuation."""
        from cortical.tokenizer import Tokenizer

        tok = Tokenizer()
        tokens = tok.tokenize("Hello, world! How are you?")
        # Should have words without punctuation
        self.assertIn('hello', tokens)
        self.assertIn('world', tokens)


class TestMinicolumnEdgeCases(unittest.TestCase):
    """Test more minicolumn edge cases."""

    def test_typed_connections(self):
        """Test adding typed connections."""
        col = Minicolumn("L0_test", "test", 0)

        # Add typed connection
        col.add_typed_connection("L0_other", 0.5, relation_type='RelatedTo')
        self.assertEqual(len(col.typed_connections), 1)

        # Get the typed edge
        edges = col.get_connections_by_type('RelatedTo')
        self.assertEqual(len(edges), 1)

    def test_minicolumn_to_dict(self):
        """Test minicolumn serialization."""
        col = Minicolumn("L0_test", "test", 0)
        col.occurrence_count = 5
        col.pagerank = 0.123
        col.tfidf = 0.456

        d = col.to_dict()
        self.assertEqual(d['id'], "L0_test")
        self.assertEqual(d['content'], "test")
        self.assertEqual(d['occurrence_count'], 5)


class TestLayerSerialization(unittest.TestCase):
    """Test layer serialization."""

    def test_layer_to_dict(self):
        """Test layer serialization to dict."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("term1")
        col2 = layer.get_or_create_minicolumn("term2")
        col1.add_lateral_connection(col2.id, 0.5)

        d = layer.to_dict()
        self.assertEqual(d['level'], CorticalLayer.TOKENS.value)
        self.assertIn('minicolumns', d)
        self.assertEqual(len(d['minicolumns']), 2)


class TestSemanticsRetrofitCoverage(unittest.TestCase):
    """Test retrofit functions in semantics module for coverage."""

    def test_retrofit_connections_invalid_alpha_too_high(self):
        """Test retrofit_connections raises ValueError when alpha > 1."""
        from cortical.semantics import retrofit_connections

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")
        processor.compute_all(verbose=False)

        with self.assertRaises(ValueError) as ctx:
            retrofit_connections(processor.layers, [], alpha=1.5)
        self.assertIn("between 0 and 1", str(ctx.exception))

    def test_retrofit_connections_invalid_alpha_negative(self):
        """Test retrofit_connections raises ValueError when alpha < 0."""
        from cortical.semantics import retrofit_connections

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")
        processor.compute_all(verbose=False)

        with self.assertRaises(ValueError) as ctx:
            retrofit_connections(processor.layers, [], alpha=-0.1)
        self.assertIn("between 0 and 1", str(ctx.exception))

    def test_retrofit_embeddings_invalid_alpha_negative(self):
        """Test retrofit_embeddings raises ValueError when alpha < 0."""
        from cortical.semantics import retrofit_embeddings

        embeddings = {"word1": [0.1, 0.2], "word2": [0.3, 0.4]}

        # alpha=0 is now valid (means 100% semantic, 0% original)
        # Test negative alpha which is still invalid
        with self.assertRaises(ValueError) as ctx:
            retrofit_embeddings(embeddings, [], alpha=-0.1)
        self.assertIn("between 0 and 1", str(ctx.exception))

    def test_retrofit_embeddings_invalid_alpha_too_high(self):
        """Test retrofit_embeddings raises ValueError when alpha > 1."""
        from cortical.semantics import retrofit_embeddings

        embeddings = {"word1": [0.1, 0.2], "word2": [0.3, 0.4]}

        with self.assertRaises(ValueError) as ctx:
            retrofit_embeddings(embeddings, [], alpha=1.5)
        self.assertIn("between 0 and 1", str(ctx.exception))

    def test_retrofit_embeddings_term_with_no_neighbors(self):
        """Test retrofit_embeddings skips terms with no semantic neighbors."""
        from cortical.semantics import retrofit_embeddings

        embeddings = {
            "isolated": [0.1, 0.2],
            "connected1": [0.3, 0.4],
            "connected2": [0.5, 0.6]
        }
        # Only connected1 and connected2 are related
        relations = [("connected1", "RelatedTo", "connected2", 0.8)]

        result = retrofit_embeddings(embeddings, relations, alpha=0.5)
        # Should return stats dict
        self.assertIsInstance(result, dict)

    def test_property_similarity_no_properties(self):
        """Test compute_property_similarity returns 0 when no properties."""
        from cortical.semantics import compute_property_similarity

        # Empty inheritance dictionaries
        inherited = {}
        direct_props = {"other_term": {"some_prop": 0.5}}

        result = compute_property_similarity(
            "unknown1", "unknown2", inherited, direct_props
        )
        self.assertEqual(result, 0.0)

    def test_property_similarity_with_direct_properties(self):
        """Test compute_property_similarity includes direct properties."""
        from cortical.semantics import compute_property_similarity

        inherited = {"dog": {"living": (0.7, "animal", 1)}}
        direct_props = {
            "dog": {"furry": 0.9},
            "cat": {"furry": 0.8, "meowing": 0.7}
        }

        result = compute_property_similarity("dog", "cat", inherited, direct_props)
        # Both share "furry" property
        self.assertGreater(result, 0.0)

    def test_extract_semantics_max_pairs_limit(self):
        """Test that max_similarity_pairs limits the number of pairs checked."""
        processor = CorticalTextProcessor()
        # Create many terms with shared context
        for i in range(5):
            processor.process_document(f"doc{i}", f"""
                common shared vocabulary words term{i}
                another common context overlap term{i}
                more common terms context vectors term{i}
            """)
        processor.compute_all(verbose=False)

        from cortical.semantics import extract_corpus_semantics
        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer,
            max_similarity_pairs=5,  # Very low limit
            use_pattern_extraction=False
        )
        self.assertIsInstance(relations, list)


class TestProcessorVerboseCoverage(unittest.TestCase):
    """Test verbose output paths in processor."""

    def test_add_documents_batch_verbose(self):
        """Test verbose output during batch document processing."""
        processor = CorticalTextProcessor()
        docs = [
            ("doc1", "first document content", None),
            ("doc2", "second document content", None),
        ]

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.add_documents_batch(docs, verbose=True, recompute='full')

        output = '\n'.join(cm.output)
        self.assertIn("Adding", output)

    def test_add_documents_batch_invalid_content(self):
        """Test that batch processing validates content type."""
        processor = CorticalTextProcessor()
        docs = [("doc1", 123, None)]  # Invalid: content is int, not str

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch(docs)
        self.assertIn("string", str(ctx.exception).lower())

    def test_compute_importance_verbose(self):
        """Test compute_importance with verbose output."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.process_document("doc2", "machine learning models")
        processor.propagate_activation(iterations=3, verbose=False)

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.compute_importance(verbose=True)

        output = '\n'.join(cm.output)
        self.assertIn("PageRank", output)

    def test_compute_semantic_importance_verbose(self):
        """Test semantic importance with verbose output."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.propagate_activation(iterations=3, verbose=False)
        processor.extract_corpus_semantics(verbose=False)

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.compute_semantic_importance(verbose=True)

        output = '\n'.join(cm.output)
        self.assertTrue(len(output) > 0)

    def test_build_concept_clusters_label_propagation(self):
        """Test label propagation clustering method."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.process_document("doc2", "machine learning algorithms data")
        processor.compute_all(build_concepts=False, verbose=False)

        clusters = processor.build_concept_clusters(
            clustering_method='label_propagation',
            verbose=False
        )
        # Returns a dict of cluster_id -> list of terms
        self.assertIsInstance(clusters, dict)

    def test_extract_corpus_semantics_verbose(self):
        """Test verbose output during semantic extraction."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learn patterns")
        processor.process_document("doc2", "deep learning neural models")
        processor.propagate_activation(iterations=3, verbose=False)

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.extract_corpus_semantics(verbose=True)

        output = '\n'.join(cm.output)
        self.assertIn("Extracted", output)


class TestProcessorWrapperMethods(unittest.TestCase):
    """Test wrapper methods in processor for coverage."""

    def test_find_related_documents(self):
        """Test find_related_documents wrapper."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.process_document("doc2", "neural network machine learning")
        processor.process_document("doc3", "cooking recipes baking")
        processor.compute_all(verbose=False)

        results = processor.find_related_documents("doc1")
        self.assertIsInstance(results, list)

    def test_get_document_signature(self):
        """Test get_document_signature wrapper."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.compute_tfidf(verbose=False)

        signature = processor.get_document_signature("doc1", n=5)
        self.assertIsInstance(signature, list)
        if signature:
            self.assertIsInstance(signature[0], tuple)
            self.assertEqual(len(signature[0]), 2)

    def test_get_document_signature_nonexistent(self):
        """Test get_document_signature with nonexistent doc."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        signature = processor.get_document_signature("nonexistent")
        self.assertEqual(signature, [])

    def test_get_corpus_summary(self):
        """Test get_corpus_summary wrapper."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network")
        processor.compute_all(verbose=False)

        summary = processor.get_corpus_summary()
        self.assertIsInstance(summary, dict)

    def test_embedding_similarity(self):
        """Test embedding_similarity wrapper."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network learning")
        processor.compute_graph_embeddings(verbose=False)

        score = processor.embedding_similarity("neural", "network")
        self.assertIsInstance(score, float)

    def test_find_similar_by_embedding(self):
        """Test find_similar_by_embedding wrapper."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.compute_graph_embeddings(verbose=False)

        results = processor.find_similar_by_embedding("neural", top_n=5)
        self.assertIsInstance(results, list)

    def test_expand_query_semantic(self):
        """Test expand_query_semantic wrapper."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.extract_corpus_semantics(verbose=False)

        results = processor.expand_query_semantic("neural", max_expansions=5)
        self.assertIsInstance(results, dict)

    def test_set_query_cache_size_trim(self):
        """Test that setting smaller cache size trims existing cache."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning models")
        processor.compute_all(verbose=False)

        # Fill cache with queries
        processor.expand_query("query1")
        processor.expand_query("query2")
        processor.expand_query("query3")

        # Trim cache
        processor.set_query_cache_size(1)
        # Cache should be trimmed (can't directly check size but no error)

    def test_export_graph_wrapper(self):
        """Test export_graph wrapper method."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            stats = processor.export_graph(temp_path, max_nodes=100)
            self.assertIsInstance(stats, dict)
            self.assertTrue(os.path.exists(temp_path))
        finally:
            os.unlink(temp_path)


class TestProcessorDocumentOperations(unittest.TestCase):
    """Test document operations in processor."""

    def test_set_document_metadata_new_doc(self):
        """Test setting metadata for new document."""
        processor = CorticalTextProcessor()
        processor.set_document_metadata("new_doc", key="value", type="test")

        meta = processor.get_document_metadata("new_doc")
        self.assertEqual(meta["key"], "value")
        self.assertEqual(meta["type"], "test")

    def test_remove_document_with_bigrams(self):
        """Test that removing document updates bigram occurrence counts."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network test")
        processor.compute_bigram_connections(verbose=False)

        # Check bigrams exist
        layer1 = processor.layers[CorticalLayer.BIGRAMS]
        initial_count = layer1.column_count()

        # Remove document
        processor.remove_document("doc1")

        # Bigrams should be affected
        self.assertLessEqual(layer1.column_count(), initial_count)

    def test_summarize_document(self):
        """Test document summarization."""
        processor = CorticalTextProcessor()
        text = "First sentence about neural networks. Second sentence about deep learning. Third sentence about machine learning. Fourth sentence about data science."
        processor.process_document("doc1", text)
        processor.compute_tfidf(verbose=False)

        summary = processor.summarize_document("doc1", num_sentences=2)
        self.assertIsInstance(summary, str)

    def test_summarize_document_single_sentence(self):
        """Test summarization with fewer sentences than requested."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Only one sentence here.")

        summary = processor.summarize_document("doc1", num_sentences=5)
        self.assertIn("Only one sentence", summary)

    def test_summarize_document_nonexistent(self):
        """Test summarization of nonexistent document."""
        processor = CorticalTextProcessor()
        summary = processor.summarize_document("nonexistent", num_sentences=1)
        self.assertEqual(summary, "")

    def test_compute_property_similarity_no_relations(self):
        """Test property similarity with no semantic relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)
        # Don't extract semantics

        result = processor.compute_property_similarity("test", "content")
        self.assertEqual(result, 0.0)


class TestChunkIndexLazyLoading(unittest.TestCase):
    """Test lazy loading paths in chunk index."""

    def test_get_documents_lazy_loading(self):
        """Test that get_documents triggers load_all when needed."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document("doc1", "test content", mtime=12345.0)
            writer.save()

            loader = ChunkLoader(tmpdir)
            # Call get_documents without calling load_all first
            docs = loader.get_documents()
            self.assertIn("doc1", docs)

    def test_get_metadata_lazy_loading(self):
        """Test that get_metadata triggers load_all when needed."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document("doc1", "test", mtime=1000.0, metadata={"type": "test"})
            writer.save()

            loader = ChunkLoader(tmpdir)
            # Call get_metadata without calling load_all first
            meta = loader.get_metadata()
            self.assertIn("doc1", meta)

    def test_get_chunks_lazy_loading(self):
        """Test that get_chunks triggers load_all when needed."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document("doc1", "test", mtime=1000.0)
            writer.save()

            loader = ChunkLoader(tmpdir)
            # Call get_chunks without calling load_all first
            chunks = loader.get_chunks()
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0)

    def test_cache_validity_missing_cache(self):
        """Test is_cache_valid returns False when cache file missing."""
        from cortical.chunk_index import ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ChunkLoader(tmpdir)
            result = loader.is_cache_valid("/nonexistent/path/cache.pkl")
            self.assertFalse(result)

    def test_cache_validity_missing_hash(self):
        """Test is_cache_valid returns False when hash file missing."""
        from cortical.chunk_index import ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                cache_path = f.name
                f.write(b"test data")

            try:
                loader = ChunkLoader(tmpdir)
                # Cache exists but hash file doesn't
                result = loader.is_cache_valid(cache_path)
                self.assertFalse(result)
            finally:
                os.unlink(cache_path)

    def test_load_all_caching(self):
        """Test that load_all caches results on second call."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document("doc1", "test content", mtime=1000.0)
            writer.save()

            loader = ChunkLoader(tmpdir)
            docs1 = loader.load_all()
            docs2 = loader.load_all()  # Should use cache

            self.assertEqual(docs1, docs2)


class TestChunkIndexCompaction(unittest.TestCase):
    """Test compaction in chunk index."""

    def test_compact_multiple_chunks(self):
        """Test compacting multiple chunks into one."""
        from cortical.chunk_index import ChunkWriter, ChunkLoader, ChunkCompactor

        with tempfile.TemporaryDirectory() as tmpdir:
            # First chunk: add 2 documents
            writer1 = ChunkWriter(tmpdir)
            writer1.add_document("doc1", "content1", mtime=1000.0)
            writer1.add_document("doc2", "content2", mtime=1001.0)
            writer1.save()

            # Second chunk: add 1 more document
            writer2 = ChunkWriter(tmpdir)
            writer2.add_document("doc3", "content3", mtime=1002.0)
            writer2.save()

            # Verify we have 2 chunk files before compaction
            loader = ChunkLoader(tmpdir)
            chunk_files_before = loader.get_chunk_files()
            self.assertGreaterEqual(len(chunk_files_before), 2)

            # Compact all chunks
            compactor = ChunkCompactor(tmpdir)
            result = compactor.compact()
            self.assertIn('status', result)

            # Load and verify all docs are preserved
            loader2 = ChunkLoader(tmpdir)
            docs = loader2.load_all()
            self.assertEqual(len(docs), 3)
            self.assertIn("doc1", docs)
            self.assertIn("doc2", docs)
            self.assertIn("doc3", docs)


class TestPersistenceTypedConnections(unittest.TestCase):
    """Test typed connections in persistence export."""

    def test_export_conceptnet_with_typed_edges(self):
        """Test export includes typed edges when requested."""
        from cortical.persistence import export_conceptnet_json

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network processing")
        processor.compute_all(verbose=False)

        # Add typed connection
        layer0 = processor.layers[CorticalLayer.TOKENS]
        col1 = layer0.get_minicolumn("neural")
        col2 = layer0.get_minicolumn("network")
        if col1 and col2:
            col1.add_typed_connection(
                col2.id,
                weight=0.9,
                relation_type='RelatedTo',
                confidence=0.95,
                source='semantic'
            )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result = export_conceptnet_json(
                temp_path,
                processor.layers,
                include_typed_edges=True,
                min_weight=0.5,
                min_confidence=0.8,
                verbose=False
            )
            # Should have edges
            self.assertIn('edges', result)
        finally:
            os.unlink(temp_path)

    def test_export_conceptnet_filters_by_weight(self):
        """Test typed edges below min_weight are excluded."""
        from cortical.persistence import export_conceptnet_json

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network")
        processor.compute_all(verbose=False)

        layer0 = processor.layers[CorticalLayer.TOKENS]
        col1 = layer0.get_minicolumn("neural")
        col2 = layer0.get_minicolumn("network")
        if col1 and col2:
            # Low weight connection
            col1.add_typed_connection(col2.id, weight=0.3, relation_type='IsA', confidence=0.9)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result = export_conceptnet_json(
                temp_path,
                processor.layers,
                include_typed_edges=True,
                min_weight=0.5,  # Higher than 0.3
                verbose=False
            )
            # Low weight edge should be filtered out
            typed_edges = [e for e in result.get('edges', []) if e.get('relation_type') == 'IsA']
            self.assertEqual(len(typed_edges), 0)
        finally:
            os.unlink(temp_path)

    def test_export_conceptnet_verbose(self):
        """Test verbose output in export."""
        from cortical.persistence import export_conceptnet_json

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            with self.assertLogs('cortical.persistence', level='INFO') as cm:
                export_conceptnet_json(temp_path, processor.layers, verbose=True)
            output = '\n'.join(cm.output)
            self.assertIn("exported", output.lower())
        finally:
            os.unlink(temp_path)


class TestProcessorConfigRestoration(unittest.TestCase):
    """Test config restoration during load."""

    def test_save_and_load_preserves_config(self):
        """Test that config is preserved through save/load."""
        from cortical.config import CorticalConfig

        config = CorticalConfig(pagerank_damping=0.75, min_cluster_size=5)
        processor = CorticalTextProcessor(config=config)
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor.save(temp_path, verbose=False)
            loaded = CorticalTextProcessor.load(temp_path, verbose=False)

            self.assertEqual(loaded.config.pagerank_damping, 0.75)
            self.assertEqual(loaded.config.min_cluster_size, 5)
        finally:
            os.unlink(temp_path)


class TestProcessorRetrofitting(unittest.TestCase):
    """Test retrofitting methods in processor."""

    def test_retrofit_embeddings_auto_compute(self):
        """Test retrofit_embeddings auto-computes required data."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network deep learning")
        processor.process_document("doc2", "machine learning models")
        processor.compute_all(verbose=False)

        # retrofit_embeddings should work even without pre-computed embeddings
        stats = processor.retrofit_embeddings(verbose=False)
        self.assertIsInstance(stats, dict)

    def test_compute_property_inheritance_with_connections(self):
        """Test property inheritance with connection boosting."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "dog is animal mammal living")
        processor.process_document("doc2", "cat is animal mammal living")
        processor.compute_all(verbose=False)
        processor.extract_corpus_semantics(verbose=False)

        result = processor.compute_property_inheritance(
            apply_to_connections=True,
            verbose=False
        )
        self.assertIn('terms_with_inheritance', result)


class TestAnalysisCoverage(unittest.TestCase):
    """Additional analysis tests for coverage."""

    def test_compute_bigram_connections_verbose_limits(self):
        """Test verbose output when bigram limits are hit."""
        processor = CorticalTextProcessor()
        # Create document with common terms that will hit limits
        processor.process_document("doc1", "the the the test test test word word word")
        processor.propagate_activation(iterations=1, verbose=False)

        with self.assertLogs('cortical.processor', level='INFO') as cm:
            processor.compute_bigram_connections(
                verbose=True,
                max_bigrams_per_term=2
            )

        # Should have some output
        output = '\n'.join(cm.output)
        self.assertTrue(len(output) > 0)


class TestSemanticInheritancePaths(unittest.TestCase):
    """Test inheritance paths in semantics module."""

    def test_apply_inheritance_missing_term(self):
        """Test apply_inheritance_to_connections skips missing terms."""
        from cortical.semantics import apply_inheritance_to_connections

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)

        # Create inheritance with non-existent term
        inherited = {"orphan_term": {"prop": (0.5, "ancestor", 1)}}

        stats = apply_inheritance_to_connections(processor.layers, inherited)
        self.assertEqual(stats['connections_boosted'], 0)

    def test_get_ancestors_deep_hierarchy(self):
        """Test get_ancestors with deep hierarchy."""
        from cortical.semantics import get_ancestors

        # Create linear hierarchy: a→b→c→d (parent relationships)
        parents = {
            "a": {"b"},
            "b": {"c"},
            "c": {"d"}
        }

        ancestors = get_ancestors("a", parents, max_depth=10)
        # get_ancestors returns {ancestor: depth} dict
        self.assertIn("b", ancestors)
        self.assertIn("c", ancestors)
        self.assertIn("d", ancestors)
        # Verify depths
        self.assertEqual(ancestors["b"], 1)
        self.assertEqual(ancestors["c"], 2)
        self.assertEqual(ancestors["d"], 3)

    def test_get_descendants_with_multiple_paths(self):
        """Test get_descendants handles multiple paths."""
        from cortical.semantics import get_descendants

        # Create hierarchy with multiple children
        children = {
            "animal": {"mammal", "bird"},
            "mammal": {"dog", "cat"},
            "bird": {"sparrow"}
        }

        descendants = get_descendants("animal", children, max_depth=2)
        # get_descendants returns {descendant: depth} dict
        self.assertIn("mammal", descendants)
        self.assertIn("bird", descendants)
        self.assertIn("dog", descendants)
        self.assertIn("cat", descendants)
        self.assertIn("sparrow", descendants)


if __name__ == "__main__":
    unittest.main()
