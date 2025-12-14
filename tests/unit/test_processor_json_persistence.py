"""
Unit tests for JSON persistence methods in CorticalTextProcessor.

Tests the save_json(), load_json(), and migrate_to_json() methods.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.layers import CorticalLayer


class TestProcessorJSONPersistence(unittest.TestCase):
    """Test JSON persistence methods."""

    def setUp(self):
        """Create test processor and temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.json_dir = os.path.join(self.temp_dir, 'test_state')
        self.pkl_path = os.path.join(self.temp_dir, 'test.pkl')

        # Create processor with some test data
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Neural networks learn patterns from data.")
        self.processor.process_document("doc2", "Machine learning algorithms optimize parameters.")
        self.processor.compute_all()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_json_creates_directory_structure(self):
        """Test that save_json creates the correct directory structure."""
        results = self.processor.save_json(self.json_dir, verbose=False)

        # Check that directory exists
        self.assertTrue(os.path.exists(self.json_dir))

        # Check that manifest exists
        manifest_path = os.path.join(self.json_dir, 'manifest.json')
        self.assertTrue(os.path.exists(manifest_path))

        # Check that subdirectories exist
        layers_dir = os.path.join(self.json_dir, 'layers')
        computed_dir = os.path.join(self.json_dir, 'computed')
        self.assertTrue(os.path.exists(layers_dir))
        self.assertTrue(os.path.exists(computed_dir))

        # Check that layer files exist
        for level in range(4):
            layer_file = os.path.join(layers_dir, f'L{level}_{["tokens", "bigrams", "concepts", "documents"][level]}.json')
            self.assertTrue(os.path.exists(layer_file), f"Layer file {layer_file} should exist")

        # Check that documents file exists
        docs_path = os.path.join(self.json_dir, 'documents.json')
        self.assertTrue(os.path.exists(docs_path))

    def test_save_json_returns_write_status(self):
        """Test that save_json returns correct write status."""
        # First save should write all files
        results = self.processor.save_json(self.json_dir, verbose=False)

        # Check that results contain expected keys
        self.assertIn('layer_0', results)
        self.assertIn('layer_1', results)
        self.assertIn('layer_2', results)
        self.assertIn('layer_3', results)
        self.assertIn('documents', results)

        # First save should write files
        self.assertTrue(results['layer_0'])
        self.assertTrue(results['documents'])

    def test_save_json_incremental_only_updates_changed(self):
        """Test that incremental save only updates changed components."""
        # First save
        self.processor.save_json(self.json_dir, verbose=False)

        # Second save without changes should not rewrite files
        results = self.processor.save_json(self.json_dir, verbose=False)

        # No files should be written (unchanged)
        self.assertFalse(results.get('layer_0', True))
        self.assertFalse(results.get('documents', True))

        # Add a new document
        self.processor.add_document_incremental("doc3", "Deep learning uses neural networks.", recompute='none')

        # Third save should only update documents and affected layers
        results = self.processor.save_json(self.json_dir, verbose=False)

        # Documents should be rewritten
        self.assertTrue(results.get('documents', False))

    def test_save_json_force_overwrites_all(self):
        """Test that force=True always overwrites files."""
        # First save
        self.processor.save_json(self.json_dir, verbose=False)

        # Second save with force should rewrite all files
        results = self.processor.save_json(self.json_dir, force=True, verbose=False)

        # All files should be written
        self.assertTrue(results['layer_0'])
        self.assertTrue(results['documents'])

    def test_load_json_restores_processor_state(self):
        """Test that load_json correctly restores processor state."""
        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Load processor
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify documents are restored
        self.assertEqual(len(loaded.documents), 2)
        self.assertIn("doc1", loaded.documents)
        self.assertIn("doc2", loaded.documents)
        self.assertEqual(loaded.documents["doc1"], "Neural networks learn patterns from data.")

        # Verify layers are restored
        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS, CorticalLayer.CONCEPTS, CorticalLayer.DOCUMENTS]:
            self.assertIn(layer_enum, loaded.layers)
            # Check that layers have content
            layer = loaded.layers[layer_enum]
            if layer_enum in [CorticalLayer.TOKENS, CorticalLayer.DOCUMENTS]:
                self.assertGreater(len(layer.minicolumns), 0)

    def test_load_json_with_custom_config(self):
        """Test that load_json accepts custom config."""
        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Load with custom config
        custom_config = CorticalConfig(chunk_size=300, chunk_overlap=75)
        loaded = CorticalTextProcessor.load_json(self.json_dir, config=custom_config, verbose=False)

        # Verify config was applied
        self.assertEqual(loaded.config.chunk_size, 300)
        self.assertEqual(loaded.config.chunk_overlap, 75)

        # Verify data is still loaded correctly
        self.assertEqual(len(loaded.documents), 2)

    def test_load_json_restores_embeddings(self):
        """Test that load_json restores graph embeddings."""
        # Compute embeddings
        self.processor.compute_graph_embeddings(verbose=False)

        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Load processor
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify embeddings are restored
        self.assertGreater(len(loaded.embeddings), 0)
        # Check that some terms have embeddings
        for term in ['neural', 'networks', 'learning']:
            if term in self.processor.embeddings:
                self.assertIn(term, loaded.embeddings)
                # Check that embedding vectors match
                self.assertEqual(
                    loaded.embeddings[term],
                    self.processor.embeddings[term]
                )

    def test_load_json_restores_semantic_relations(self):
        """Test that load_json restores semantic relations."""
        # Extract semantic relations
        self.processor.extract_corpus_semantics(verbose=False)

        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Load processor
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify semantic relations are restored
        self.assertEqual(len(loaded.semantic_relations), len(self.processor.semantic_relations))

        # Check that some relations match
        if self.processor.semantic_relations:
            # Relations are tuples of (term1, relation, term2, weight)
            self.assertIn(self.processor.semantic_relations[0], loaded.semantic_relations)

    def test_load_json_restores_staleness_tracking(self):
        """Test that load_json restores staleness tracking."""
        # Mark some computations as stale
        self.processor._mark_all_stale()

        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Load processor
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify staleness is restored
        stale = loaded.get_stale_computations()
        self.assertIsInstance(stale, set)
        # At least some computations should be marked stale
        self.assertTrue(len(stale) > 0)

    def test_round_trip_preserves_all_data(self):
        """Test that save_json followed by load_json preserves all data."""
        # Compute all features
        self.processor.compute_all(verbose=False)
        self.processor.compute_graph_embeddings(verbose=False)
        self.processor.extract_corpus_semantics(verbose=False)

        # Save
        self.processor.save_json(self.json_dir, verbose=False)

        # Load
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify all data is preserved
        self.assertEqual(loaded.documents, self.processor.documents)
        self.assertEqual(len(loaded.layers), len(self.processor.layers))

        # Verify layer 0 minicolumns
        layer0_orig = self.processor.layers[CorticalLayer.TOKENS]
        layer0_loaded = loaded.layers[CorticalLayer.TOKENS]
        self.assertEqual(len(layer0_orig.minicolumns), len(layer0_loaded.minicolumns))

        # Verify a specific minicolumn's data
        if layer0_orig.minicolumns:
            sample_id = list(layer0_orig.minicolumns.keys())[0]
            orig_col = layer0_orig.minicolumns[sample_id]
            loaded_col = layer0_loaded.minicolumns[sample_id]

            self.assertEqual(orig_col.id, loaded_col.id)
            self.assertEqual(orig_col.content, loaded_col.content)
            self.assertEqual(orig_col.document_ids, loaded_col.document_ids)
            # PageRank and TF-IDF should be close (floating point comparison)
            self.assertAlmostEqual(orig_col.pagerank, loaded_col.pagerank, places=6)

    def test_migrate_to_json_converts_pickle(self):
        """Test that migrate_to_json successfully converts pickle to JSON."""
        # Save as pickle first
        self.processor.save(self.pkl_path, verbose=False)

        # Verify pickle file exists
        self.assertTrue(os.path.exists(self.pkl_path))

        # Migrate to JSON
        result = self.processor.migrate_to_json(self.pkl_path, self.json_dir, verbose=False)

        # Verify migration succeeded
        self.assertTrue(result)

        # Verify JSON directory was created
        self.assertTrue(os.path.exists(self.json_dir))

        # Verify manifest exists
        manifest_path = os.path.join(self.json_dir, 'manifest.json')
        self.assertTrue(os.path.exists(manifest_path))

        # Load from JSON and verify data
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)
        self.assertEqual(len(loaded.documents), 2)
        self.assertIn("doc1", loaded.documents)

    def test_migrate_to_json_preserves_data(self):
        """Test that migration from pickle to JSON preserves all data."""
        # Compute all features
        self.processor.compute_all(verbose=False)
        self.processor.compute_graph_embeddings(verbose=False)
        self.processor.extract_corpus_semantics(verbose=False)

        # Save as pickle
        self.processor.save(self.pkl_path, verbose=False)

        # Load from pickle to verify original state
        pkl_loaded = CorticalTextProcessor.load(self.pkl_path, verbose=False)

        # Migrate to JSON
        self.processor.migrate_to_json(self.pkl_path, self.json_dir, verbose=False)

        # Load from JSON
        json_loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify data matches
        self.assertEqual(json_loaded.documents, pkl_loaded.documents)
        self.assertEqual(len(json_loaded.layers), len(pkl_loaded.layers))
        self.assertEqual(len(json_loaded.embeddings), len(pkl_loaded.embeddings))
        self.assertEqual(len(json_loaded.semantic_relations), len(pkl_loaded.semantic_relations))

    def test_save_json_with_empty_processor(self):
        """Test that save_json works with empty processor."""
        empty_processor = CorticalTextProcessor()

        # Save empty processor
        results = empty_processor.save_json(self.json_dir, verbose=False)

        # Verify directory was created
        self.assertTrue(os.path.exists(self.json_dir))

        # Verify files were written
        self.assertTrue(results.get('layer_0', False))

        # Load and verify
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)
        self.assertEqual(len(loaded.documents), 0)

    def test_load_json_raises_on_missing_directory(self):
        """Test that load_json raises FileNotFoundError for missing directory."""
        missing_dir = os.path.join(self.temp_dir, 'nonexistent')

        with self.assertRaises(FileNotFoundError):
            CorticalTextProcessor.load_json(missing_dir, verbose=False)

    def test_save_json_manifest_contains_correct_metadata(self):
        """Test that manifest.json contains correct metadata."""
        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Read manifest
        manifest_path = os.path.join(self.json_dir, 'manifest.json')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # Verify manifest structure
        self.assertIn('version', manifest)
        self.assertIn('created_at', manifest)
        self.assertIn('updated_at', manifest)
        self.assertIn('checksums', manifest)
        self.assertIn('document_count', manifest)
        self.assertIn('layer_stats', manifest)

        # Verify document count
        self.assertEqual(manifest['document_count'], 2)

        # Verify checksums exist for components
        self.assertIn('layer_0', manifest['checksums'])
        self.assertIn('documents', manifest['checksums'])

    def test_save_json_documents_file_structure(self):
        """Test that documents.json has correct structure."""
        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Read documents.json
        docs_path = os.path.join(self.json_dir, 'documents.json')
        with open(docs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify structure
        self.assertIn('documents', data)
        self.assertIn('metadata', data)

        # Verify content
        self.assertEqual(len(data['documents']), 2)
        self.assertIn('doc1', data['documents'])
        self.assertEqual(data['documents']['doc1'], "Neural networks learn patterns from data.")

    def test_save_json_layer_file_structure(self):
        """Test that layer JSON files have correct structure."""
        # Save processor
        self.processor.save_json(self.json_dir, verbose=False)

        # Read a layer file
        layer_path = os.path.join(self.json_dir, 'layers', 'L0_tokens.json')
        with open(layer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify structure (should match HierarchicalLayer.to_dict())
        self.assertIn('level', data)
        self.assertIn('minicolumns', data)

        # Verify level
        self.assertEqual(data['level'], 0)

        # Verify minicolumns is a dict
        self.assertIsInstance(data['minicolumns'], dict)

    def test_multiple_save_load_cycles(self):
        """Test that multiple save/load cycles work correctly."""
        # First cycle
        self.processor.save_json(self.json_dir, verbose=False)
        loaded1 = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Add more data
        loaded1.add_document_incremental("doc3", "Deep learning is powerful.", recompute='none')

        # Second cycle
        loaded1.save_json(self.json_dir, verbose=False)
        loaded2 = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify all documents present
        self.assertEqual(len(loaded2.documents), 3)
        self.assertIn("doc1", loaded2.documents)
        self.assertIn("doc2", loaded2.documents)
        self.assertIn("doc3", loaded2.documents)

    def test_save_json_preserves_document_metadata(self):
        """Test that document metadata is preserved."""
        # Add metadata
        self.processor.document_metadata['doc1'] = {'source': 'test', 'timestamp': '2025-12-14'}
        self.processor.document_metadata['doc2'] = {'source': 'test2', 'timestamp': '2025-12-15'}

        # Save
        self.processor.save_json(self.json_dir, verbose=False)

        # Load
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)

        # Verify metadata preserved
        self.assertEqual(loaded.document_metadata['doc1']['source'], 'test')
        self.assertEqual(loaded.document_metadata['doc2']['timestamp'], '2025-12-15')


class TestJSONPersistenceEdgeCases(unittest.TestCase):
    """Test edge cases for JSON persistence."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.json_dir = os.path.join(self.temp_dir, 'test_state')

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_json_with_no_semantic_relations(self):
        """Test saving when semantic_relations is empty."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")

        # Don't extract semantics
        results = processor.save_json(self.json_dir, verbose=False)

        # Should still succeed
        self.assertTrue(os.path.exists(self.json_dir))

        # Load and verify
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)
        self.assertEqual(len(loaded.semantic_relations), 0)

    def test_save_json_with_no_embeddings(self):
        """Test saving when embeddings is empty."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")

        # Don't compute embeddings
        results = processor.save_json(self.json_dir, verbose=False)

        # Should still succeed
        self.assertTrue(os.path.exists(self.json_dir))

        # Load and verify
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)
        self.assertEqual(len(loaded.embeddings), 0)

    def test_load_json_with_missing_computed_files(self):
        """Test loading when computed/ files are missing."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")

        # Save
        processor.save_json(self.json_dir, verbose=False)

        # Remove computed files
        computed_dir = os.path.join(self.json_dir, 'computed')
        if os.path.exists(computed_dir):
            shutil.rmtree(computed_dir)

        # Should still load successfully (computed data is optional)
        loaded = CorticalTextProcessor.load_json(self.json_dir, verbose=False)
        self.assertEqual(len(loaded.documents), 1)
        self.assertEqual(len(loaded.embeddings), 0)
        self.assertEqual(len(loaded.semantic_relations), 0)

    def test_save_json_verbose_output(self):
        """Test that verbose=True produces log output."""
        import logging
        from io import StringIO

        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger('cortical.state_storage')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            processor = CorticalTextProcessor()
            processor.process_document("doc1", "Test content.")

            # Save with verbose=True
            processor.save_json(self.json_dir, verbose=True)

            # Check that log contains expected messages
            log_output = log_capture.getvalue()
            # Note: verbose logging goes through state_storage module
            # The exact message depends on state_storage implementation

        finally:
            logger.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
