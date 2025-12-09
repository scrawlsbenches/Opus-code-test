"""Tests for the persistence module."""

import unittest
import tempfile
import os
import json
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.persistence import (
    save_processor,
    load_processor,
    export_graph_json,
    export_embeddings_json,
    get_state_summary
)
from cortical.embeddings import compute_graph_embeddings


class TestSaveLoad(unittest.TestCase):
    """Test save and load functionality."""

    def test_save_and_load(self):
        """Test saving and loading processor state."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")
        processor.process_document("doc2", "Machine learning algorithms learn.")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(filepath, processor.layers, processor.documents, verbose=False)

            layers, documents, metadata = load_processor(filepath, verbose=False)

            self.assertEqual(len(documents), 2)
            self.assertIn("doc1", documents)
            self.assertIn("doc2", documents)

            # Check layers were restored
            layer0 = layers[CorticalLayer.TOKENS]
            self.assertGreater(len(layer0.minicolumns), 0)

    def test_save_load_preserves_id_index(self):
        """Test that save/load preserves the ID index."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks deep learning")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(filepath, processor.layers, processor.documents, verbose=False)

            layers, documents, _ = load_processor(filepath, verbose=False)

            layer0 = layers[CorticalLayer.TOKENS]
            neural = layer0.get_minicolumn("neural")

            # get_by_id should work after load
            retrieved = layer0.get_by_id(neural.id)
            self.assertEqual(retrieved.content, "neural")

    def test_save_load_preserves_doc_occurrence_counts(self):
        """Test that save/load preserves doc_occurrence_counts."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural neural neural")  # 3 times
        processor.process_document("doc2", "neural")  # 1 time
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(filepath, processor.layers, processor.documents, verbose=False)

            layers, documents, _ = load_processor(filepath, verbose=False)

            layer0 = layers[CorticalLayer.TOKENS]
            neural = layer0.get_minicolumn("neural")

            self.assertEqual(neural.doc_occurrence_counts.get("doc1"), 3)
            self.assertEqual(neural.doc_occurrence_counts.get("doc2"), 1)

    def test_save_load_empty_processor(self):
        """Test saving and loading empty processor."""
        processor = CorticalTextProcessor()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(filepath, processor.layers, processor.documents, verbose=False)

            layers, documents, metadata = load_processor(filepath, verbose=False)

            self.assertEqual(len(documents), 0)


class TestExportGraphJSON(unittest.TestCase):
    """Test graph JSON export."""

    def test_export_graph_json(self):
        """Test exporting graph to JSON."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning")
        processor.process_document("doc2", "machine learning algorithms")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            result = export_graph_json(filepath, processor.layers, verbose=False)

            # Check file was created
            self.assertTrue(os.path.exists(filepath))

            # Check result structure
            self.assertIn('nodes', result)
            self.assertIn('edges', result)
            self.assertIn('metadata', result)

            # Verify file contents
            with open(filepath) as f:
                data = json.load(f)
            self.assertEqual(data['metadata']['node_count'], len(data['nodes']))

    def test_export_graph_json_layer_filter(self):
        """Test exporting specific layer."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            result = export_graph_json(
                filepath,
                processor.layers,
                layer_filter=CorticalLayer.TOKENS,
                verbose=False
            )

            # All nodes should be from layer 0
            for node in result['nodes']:
                self.assertEqual(node['layer'], 0)

    def test_export_graph_json_min_weight(self):
        """Test filtering edges by minimum weight."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            result = export_graph_json(
                filepath,
                processor.layers,
                min_weight=0.5,
                verbose=False
            )

            # All edges should have weight >= 0.5
            for edge in result['edges']:
                self.assertGreaterEqual(edge['weight'], 0.5)

    def test_export_graph_json_max_nodes(self):
        """Test limiting number of nodes."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "word1 word2 word3 word4 word5 word6 word7 word8")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            result = export_graph_json(
                filepath,
                processor.layers,
                max_nodes=3,
                verbose=False
            )

            self.assertLessEqual(len(result['nodes']), 3)

    def test_export_graph_json_verbose_false(self):
        """Test that verbose=False suppresses output."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            # This should not print anything
            export_graph_json(filepath, processor.layers, verbose=False)


class TestExportEmbeddingsJSON(unittest.TestCase):
    """Test embeddings JSON export."""

    def test_export_embeddings_json(self):
        """Test exporting embeddings to JSON."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning")
        processor.compute_all(verbose=False)

        embeddings, _ = compute_graph_embeddings(
            processor.layers,
            dimensions=16,
            method='adjacency'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "embeddings.json")
            export_embeddings_json(filepath, embeddings)

            # Check file was created
            self.assertTrue(os.path.exists(filepath))

            # Check file contents
            with open(filepath) as f:
                data = json.load(f)
            self.assertIn('embeddings', data)
            self.assertIn('metadata', data)

    def test_export_embeddings_json_with_metadata(self):
        """Test exporting embeddings with custom metadata."""
        embeddings = {'term1': [1.0, 2.0], 'term2': [3.0, 4.0]}
        metadata = {'custom_key': 'custom_value'}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "embeddings.json")
            export_embeddings_json(filepath, embeddings, metadata)

            with open(filepath) as f:
                data = json.load(f)
            self.assertIn('custom_key', data['metadata'])


class TestGetStateSummary(unittest.TestCase):
    """Test state summary functionality."""

    def test_get_state_summary(self):
        """Test getting state summary."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning")
        processor.process_document("doc2", "machine learning algorithms")
        processor.compute_all(verbose=False)

        summary = get_state_summary(processor.layers, processor.documents)

        # Check expected keys (actual keys from get_state_summary)
        self.assertIn('documents', summary)
        self.assertIn('layers', summary)
        self.assertIn('total_connections', summary)
        self.assertIn('total_columns', summary)

        self.assertEqual(summary['documents'], 2)

    def test_get_state_summary_empty(self):
        """Test summary for empty processor."""
        processor = CorticalTextProcessor()

        summary = get_state_summary(processor.layers, processor.documents)

        self.assertEqual(summary['documents'], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
