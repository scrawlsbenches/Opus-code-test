"""
Unit tests for cortical/state_storage.py

Tests the git-friendly JSON state storage system that replaces pickle-based persistence.
"""

import json
import os
import tempfile
import shutil
import unittest
from pathlib import Path

from cortical.state_storage import (
    StateManifest,
    StateWriter,
    StateLoader,
    migrate_pkl_to_json,
    STATE_VERSION,
    LAYER_FILENAMES,
)
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn, Edge


class TestStateManifest(unittest.TestCase):
    """Tests for StateManifest dataclass."""

    def test_default_values(self):
        """Test manifest creates with sensible defaults."""
        manifest = StateManifest()
        self.assertEqual(manifest.version, STATE_VERSION)
        self.assertIsNotNone(manifest.created_at)
        self.assertIsNotNone(manifest.updated_at)
        self.assertEqual(manifest.checksums, {})
        self.assertEqual(manifest.stale_computations, [])
        self.assertEqual(manifest.document_count, 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Test manifest serialization roundtrip."""
        manifest = StateManifest(
            version=1,
            checksums={'layer_0': 'abc123'},
            stale_computations=['pagerank', 'tfidf'],
            document_count=42
        )
        data = manifest.to_dict()
        restored = StateManifest.from_dict(data)

        self.assertEqual(restored.version, manifest.version)
        self.assertEqual(restored.checksums, manifest.checksums)
        self.assertEqual(restored.stale_computations, manifest.stale_computations)
        self.assertEqual(restored.document_count, manifest.document_count)

    def test_update_checksum_detects_change(self):
        """Test checksum update returns True for new/changed content."""
        manifest = StateManifest()

        # First update should return True (new)
        changed = manifest.update_checksum('test', 'content1')
        self.assertTrue(changed)

        # Same content should return False
        changed = manifest.update_checksum('test', 'content1')
        self.assertFalse(changed)

        # Different content should return True
        changed = manifest.update_checksum('test', 'content2')
        self.assertTrue(changed)

    def test_update_checksum_updates_timestamp(self):
        """Test that updating checksum updates the timestamp."""
        manifest = StateManifest()
        original_time = manifest.updated_at

        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)

        manifest.update_checksum('test', 'content')
        self.assertNotEqual(manifest.updated_at, original_time)


class TestStateWriter(unittest.TestCase):
    """Tests for StateWriter class."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = os.path.join(self.temp_dir, 'corpus_state')

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_test_layer(self, level: int, num_cols: int = 3) -> HierarchicalLayer:
        """Create a test layer with sample minicolumns."""
        layer = HierarchicalLayer(CorticalLayer(level))
        for i in range(num_cols):
            col = Minicolumn(f"L{level}_term{i}", f"term{i}", level)
            col.occurrence_count = i + 1
            col.tfidf = 0.5 + i * 0.1
            col.pagerank = 0.1 + i * 0.05
            layer.minicolumns[f"term{i}"] = col
            layer._id_index[col.id] = f"term{i}"
        return layer

    def test_creates_directory_structure(self):
        """Test that writer creates required directories."""
        writer = StateWriter(self.state_dir)
        layer = self._create_test_layer(0)
        writer.save_layer(layer)

        self.assertTrue(os.path.exists(self.state_dir))
        self.assertTrue(os.path.exists(os.path.join(self.state_dir, 'layers')))
        self.assertTrue(os.path.exists(os.path.join(self.state_dir, 'computed')))

    def test_save_layer(self):
        """Test saving a single layer."""
        writer = StateWriter(self.state_dir)
        layer = self._create_test_layer(0, num_cols=5)

        result = writer.save_layer(layer)

        self.assertTrue(result)  # File was written
        filepath = os.path.join(self.state_dir, 'layers', 'L0_tokens.json')
        self.assertTrue(os.path.exists(filepath))

        # Verify content
        with open(filepath) as f:
            data = json.load(f)
        self.assertEqual(data['level'], 0)
        self.assertEqual(len(data['minicolumns']), 5)

    def test_save_layer_skips_unchanged(self):
        """Test that saving same layer twice skips second write."""
        writer = StateWriter(self.state_dir)
        layer = self._create_test_layer(0)

        first_result = writer.save_layer(layer)
        second_result = writer.save_layer(layer)

        self.assertTrue(first_result)
        self.assertFalse(second_result)  # Skipped, unchanged

    def test_save_layer_force_writes_unchanged(self):
        """Test that force=True writes even if unchanged."""
        writer = StateWriter(self.state_dir)
        layer = self._create_test_layer(0)

        writer.save_layer(layer)
        result = writer.save_layer(layer, force=True)

        self.assertTrue(result)  # Force write

    def test_save_documents(self):
        """Test saving documents and metadata."""
        writer = StateWriter(self.state_dir)
        documents = {'doc1': 'Content one', 'doc2': 'Content two'}
        metadata = {'doc1': {'source': 'test'}}

        result = writer.save_documents(documents, metadata)

        self.assertTrue(result)
        filepath = os.path.join(self.state_dir, 'documents.json')
        self.assertTrue(os.path.exists(filepath))

        with open(filepath) as f:
            data = json.load(f)
        self.assertEqual(data['documents'], documents)
        self.assertEqual(data['metadata'], metadata)

    def test_save_semantic_relations(self):
        """Test saving semantic relations."""
        writer = StateWriter(self.state_dir)
        relations = [
            ('neural', 'RelatedTo', 'network', 0.8),
            ('machine', 'PartOf', 'learning', 0.9),
        ]

        result = writer.save_semantic_relations(relations)

        self.assertTrue(result)
        filepath = os.path.join(self.state_dir, 'computed', 'semantic_relations.json')
        self.assertTrue(os.path.exists(filepath))

        with open(filepath) as f:
            data = json.load(f)
        self.assertEqual(len(data['relations']), 2)
        self.assertEqual(data['count'], 2)

    def test_save_embeddings(self):
        """Test saving embeddings."""
        writer = StateWriter(self.state_dir)
        embeddings = {
            'neural': [0.1, 0.2, 0.3],
            'network': [0.4, 0.5, 0.6],
        }

        result = writer.save_embeddings(embeddings)

        self.assertTrue(result)
        filepath = os.path.join(self.state_dir, 'computed', 'embeddings.json')
        self.assertTrue(os.path.exists(filepath))

        with open(filepath) as f:
            data = json.load(f)
        self.assertEqual(data['embeddings'], embeddings)
        self.assertEqual(data['dimensions'], 3)
        self.assertEqual(data['count'], 2)

    def test_save_all(self):
        """Test saving complete processor state."""
        writer = StateWriter(self.state_dir)

        layers = {
            CorticalLayer(i): self._create_test_layer(i)
            for i in range(4)
        }
        documents = {'doc1': 'Test content'}
        metadata = {'doc1': {'source': 'unit_test'}}
        embeddings = {'term': [0.1, 0.2]}
        relations = [('a', 'rel', 'b', 0.5)]

        results = writer.save_all(
            layers=layers,
            documents=documents,
            document_metadata=metadata,
            embeddings=embeddings,
            semantic_relations=relations,
            stale_computations={'pagerank'},
            verbose=False
        )

        # All components should be written
        self.assertTrue(results['layer_0'])
        self.assertTrue(results['layer_1'])
        self.assertTrue(results['layer_2'])
        self.assertTrue(results['layer_3'])
        self.assertTrue(results['documents'])
        self.assertTrue(results['embeddings'])
        self.assertTrue(results['semantic_relations'])

        # Manifest should exist
        manifest_path = os.path.join(self.state_dir, 'manifest.json')
        self.assertTrue(os.path.exists(manifest_path))

    def test_atomic_write_creates_valid_file(self):
        """Test atomic write produces valid JSON."""
        writer = StateWriter(self.state_dir)
        layer = self._create_test_layer(0)
        writer.save_layer(layer)

        filepath = os.path.join(self.state_dir, 'layers', 'L0_tokens.json')

        # Should be valid JSON
        with open(filepath) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

        # No temp files should remain
        temp_path = filepath + '.tmp'
        self.assertFalse(os.path.exists(temp_path))


class TestStateLoader(unittest.TestCase):
    """Tests for StateLoader class."""

    def setUp(self):
        """Create temp directory and write test state."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = os.path.join(self.temp_dir, 'corpus_state')
        self._write_test_state()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_test_layer(self, level: int, num_cols: int = 3) -> HierarchicalLayer:
        """Create a test layer with sample minicolumns."""
        layer = HierarchicalLayer(CorticalLayer(level))
        for i in range(num_cols):
            col = Minicolumn(f"L{level}_term{i}", f"term{i}", level)
            col.occurrence_count = i + 1
            col.tfidf = 0.5 + i * 0.1
            col.pagerank = 0.1 + i * 0.05
            layer.minicolumns[f"term{i}"] = col
            layer._id_index[col.id] = f"term{i}"
        return layer

    def _write_test_state(self):
        """Write test state files."""
        writer = StateWriter(self.state_dir)
        layers = {
            CorticalLayer(i): self._create_test_layer(i)
            for i in range(4)
        }
        documents = {'doc1': 'Test content one', 'doc2': 'Test content two'}
        metadata = {'doc1': {'source': 'test'}}
        embeddings = {'term0': [0.1, 0.2, 0.3]}
        relations = [('neural', 'RelatedTo', 'network', 0.8)]

        writer.save_all(
            layers=layers,
            documents=documents,
            document_metadata=metadata,
            embeddings=embeddings,
            semantic_relations=relations,
            stale_computations={'pagerank'},
            verbose=False
        )

    def test_exists(self):
        """Test exists() returns True for valid state."""
        loader = StateLoader(self.state_dir)
        self.assertTrue(loader.exists())

    def test_exists_false_for_missing(self):
        """Test exists() returns False for missing state."""
        loader = StateLoader('/nonexistent/path')
        self.assertFalse(loader.exists())

    def test_load_manifest(self):
        """Test loading manifest file."""
        loader = StateLoader(self.state_dir)
        manifest = loader.load_manifest()

        self.assertEqual(manifest.version, STATE_VERSION)
        self.assertEqual(manifest.document_count, 2)
        self.assertIn('pagerank', manifest.stale_computations)

    def test_load_layer(self):
        """Test loading a single layer."""
        loader = StateLoader(self.state_dir)
        layer = loader.load_layer(0)

        self.assertEqual(layer.level, 0)
        self.assertEqual(len(layer.minicolumns), 3)
        self.assertIn('term0', layer.minicolumns)

    def test_load_layer_invalid_level(self):
        """Test loading invalid layer level raises error."""
        loader = StateLoader(self.state_dir)

        with self.assertRaises(ValueError):
            loader.load_layer(5)

    def test_load_documents(self):
        """Test loading documents and metadata."""
        loader = StateLoader(self.state_dir)
        documents, metadata = loader.load_documents()

        self.assertEqual(len(documents), 2)
        self.assertIn('doc1', documents)
        self.assertEqual(metadata['doc1']['source'], 'test')

    def test_load_semantic_relations(self):
        """Test loading semantic relations."""
        loader = StateLoader(self.state_dir)
        relations = loader.load_semantic_relations()

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0], ('neural', 'RelatedTo', 'network', 0.8))

    def test_load_semantic_relations_empty(self):
        """Test loading when no relations file exists."""
        # Remove the relations file
        relations_path = os.path.join(self.state_dir, 'computed', 'semantic_relations.json')
        os.remove(relations_path)

        loader = StateLoader(self.state_dir)
        relations = loader.load_semantic_relations()

        self.assertEqual(relations, [])

    def test_load_embeddings(self):
        """Test loading embeddings."""
        loader = StateLoader(self.state_dir)
        embeddings = loader.load_embeddings()

        self.assertIn('term0', embeddings)
        self.assertEqual(embeddings['term0'], [0.1, 0.2, 0.3])

    def test_load_all(self):
        """Test loading complete state."""
        loader = StateLoader(self.state_dir)
        layers, documents, metadata, embeddings, relations, manifest_data = loader.load_all(verbose=False)

        # Check layers
        self.assertEqual(len(layers), 4)
        for level in range(4):
            self.assertIn(CorticalLayer(level), layers)
            self.assertEqual(len(layers[CorticalLayer(level)].minicolumns), 3)

        # Check documents
        self.assertEqual(len(documents), 2)
        self.assertIn('doc1', documents)

        # Check computed values
        self.assertEqual(len(relations), 1)
        self.assertIn('term0', embeddings)

        # Check manifest data
        self.assertEqual(manifest_data['version'], STATE_VERSION)
        self.assertIn('pagerank', manifest_data['stale_computations'])

    def test_get_stats(self):
        """Test getting state statistics."""
        loader = StateLoader(self.state_dir)
        stats = loader.get_stats()

        self.assertTrue(stats['exists'])
        self.assertEqual(stats['version'], STATE_VERSION)
        self.assertEqual(stats['document_count'], 2)
        self.assertIn('layer_0', stats['components'])

    def test_get_stats_missing_state(self):
        """Test getting stats for missing state."""
        loader = StateLoader('/nonexistent')
        stats = loader.get_stats()

        self.assertFalse(stats['exists'])


class TestStateRoundtrip(unittest.TestCase):
    """Tests for complete save/load roundtrip."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = os.path.join(self.temp_dir, 'corpus_state')

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_full_roundtrip(self):
        """Test complete save and load cycle preserves data."""
        # Create test data
        layers = {}
        for level in range(4):
            layer = HierarchicalLayer(CorticalLayer(level))
            for i in range(5):
                col = Minicolumn(f"L{level}_term{i}", f"term{i}", level)
                col.occurrence_count = i * 10
                col.tfidf = 0.1 * (i + 1)
                col.pagerank = 0.05 * (i + 1)
                col.document_ids = {f"doc{j}" for j in range(i + 1)}
                layer.minicolumns[f"term{i}"] = col
                layer._id_index[col.id] = f"term{i}"
            layers[CorticalLayer(level)] = layer

        documents = {f"doc{i}": f"Content for document {i}" for i in range(10)}
        metadata = {f"doc{i}": {'index': i, 'source': 'test'} for i in range(10)}
        embeddings = {f"term{i}": [0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)}
        relations = [
            ('term0', 'RelatedTo', 'term1', 0.8),
            ('term1', 'PartOf', 'term2', 0.6),
        ]

        # Save
        writer = StateWriter(self.state_dir)
        writer.save_all(
            layers=layers,
            documents=documents,
            document_metadata=metadata,
            embeddings=embeddings,
            semantic_relations=relations,
            stale_computations={'pagerank', 'concepts'},
            verbose=False
        )

        # Load
        loader = StateLoader(self.state_dir)
        (loaded_layers, loaded_docs, loaded_meta,
         loaded_embed, loaded_rels, manifest_data) = loader.load_all(verbose=False)

        # Verify layers
        self.assertEqual(len(loaded_layers), 4)
        for level in range(4):
            original = layers[CorticalLayer(level)]
            loaded = loaded_layers[CorticalLayer(level)]
            self.assertEqual(len(loaded.minicolumns), len(original.minicolumns))

            for content, orig_col in original.minicolumns.items():
                loaded_col = loaded.minicolumns[content]
                self.assertEqual(loaded_col.id, orig_col.id)
                self.assertEqual(loaded_col.occurrence_count, orig_col.occurrence_count)
                self.assertAlmostEqual(loaded_col.tfidf, orig_col.tfidf, places=5)
                self.assertAlmostEqual(loaded_col.pagerank, orig_col.pagerank, places=5)

        # Verify documents
        self.assertEqual(loaded_docs, documents)
        self.assertEqual(loaded_meta, metadata)

        # Verify computed values
        self.assertEqual(loaded_embed, embeddings)
        self.assertEqual(loaded_rels, relations)

        # Verify staleness
        self.assertEqual(manifest_data['stale_computations'], {'pagerank', 'concepts'})


class TestMigration(unittest.TestCase):
    """Tests for pkl to JSON migration."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_migrate_requires_pkl_file(self):
        """Test migration fails for missing pkl file."""
        with self.assertRaises(FileNotFoundError):
            migrate_pkl_to_json(
                '/nonexistent/file.pkl',
                os.path.join(self.temp_dir, 'output'),
                verbose=False
            )


class TestIncrementalSave(unittest.TestCase):
    """Tests for incremental saving behavior."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = os.path.join(self.temp_dir, 'corpus_state')

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_only_changed_layers_written(self):
        """Test that only modified layers are written on second save."""
        # Initial save
        writer = StateWriter(self.state_dir)

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col0 = Minicolumn("L0_test", "test", 0)
        layer0.minicolumns["test"] = col0
        layer0._id_index[col0.id] = "test"

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        col1 = Minicolumn("L1_test bigram", "test bigram", 1)
        layer1.minicolumns["test bigram"] = col1
        layer1._id_index[col1.id] = "test bigram"

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1,
        }

        writer.save_layer(layer0)
        writer.save_layer(layer1)

        # Get initial modification times
        l0_path = os.path.join(self.state_dir, 'layers', 'L0_tokens.json')
        l1_path = os.path.join(self.state_dir, 'layers', 'L1_bigrams.json')

        # Modify only layer0
        col_new = Minicolumn("L0_new", "new", 0)
        layer0.minicolumns["new"] = col_new
        layer0._id_index[col_new.id] = "new"

        # Save both again
        result0 = writer.save_layer(layer0)
        result1 = writer.save_layer(layer1)

        # Only layer0 should be written
        self.assertTrue(result0)
        self.assertFalse(result1)


if __name__ == '__main__':
    unittest.main()
