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
    load_embeddings_json,
    export_semantic_relations_json,
    load_semantic_relations_json,
    get_state_summary,
    export_conceptnet_json,
    LAYER_COLORS,
    LAYER_NAMES
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
            save_processor(
                filepath, processor.layers, processor.documents,
                processor.document_metadata, processor.embeddings,
                processor.semantic_relations, verbose=False
            )

            result = load_processor(filepath, verbose=False)
            layers, documents, document_metadata, embeddings, semantic_relations, metadata = result

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
            save_processor(
                filepath, processor.layers, processor.documents,
                processor.document_metadata, processor.embeddings,
                processor.semantic_relations, verbose=False
            )

            result = load_processor(filepath, verbose=False)
            layers, documents, document_metadata, embeddings, semantic_relations, metadata = result

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
            save_processor(
                filepath, processor.layers, processor.documents,
                processor.document_metadata, processor.embeddings,
                processor.semantic_relations, verbose=False
            )

            result = load_processor(filepath, verbose=False)
            layers, documents, document_metadata, embeddings, semantic_relations, metadata = result

            layer0 = layers[CorticalLayer.TOKENS]
            neural = layer0.get_minicolumn("neural")

            self.assertEqual(neural.doc_occurrence_counts.get("doc1"), 3)
            self.assertEqual(neural.doc_occurrence_counts.get("doc2"), 1)

    def test_save_load_empty_processor(self):
        """Test saving and loading empty processor."""
        processor = CorticalTextProcessor()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(
                filepath, processor.layers, processor.documents,
                processor.document_metadata, processor.embeddings,
                processor.semantic_relations, verbose=False
            )

            result = load_processor(filepath, verbose=False)
            layers, documents, document_metadata, embeddings, semantic_relations, metadata = result

            self.assertEqual(len(documents), 0)

    def test_save_load_preserves_document_metadata(self):
        """Test that save/load preserves document metadata."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1", "Neural networks process information.",
            metadata={"source": "https://example.com", "author": "Test"}
        )
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(
                filepath, processor.layers, processor.documents,
                processor.document_metadata, processor.embeddings,
                processor.semantic_relations, verbose=False
            )

            result = load_processor(filepath, verbose=False)
            layers, documents, document_metadata, embeddings, semantic_relations, metadata = result

            self.assertEqual(document_metadata["doc1"]["source"], "https://example.com")
            self.assertEqual(document_metadata["doc1"]["author"], "Test")

    def test_save_load_preserves_embeddings(self):
        """Test that save/load preserves graph embeddings."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")
        processor.compute_all(verbose=False)
        processor.compute_graph_embeddings(dimensions=16, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            processor.save(filepath, verbose=False)

            loaded = CorticalTextProcessor.load(filepath, verbose=False)

            self.assertEqual(len(loaded.embeddings), len(processor.embeddings))
            # Check a specific embedding is preserved
            for term in processor.embeddings:
                self.assertIn(term, loaded.embeddings)
                self.assertEqual(processor.embeddings[term], loaded.embeddings[term])

    def test_save_load_preserves_semantic_relations(self):
        """Test that save/load preserves semantic relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks are computational models.")
        processor.process_document("doc2", "Deep learning uses neural networks.")
        processor.compute_all(verbose=False)
        processor.extract_corpus_semantics(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            processor.save(filepath, verbose=False)

            loaded = CorticalTextProcessor.load(filepath, verbose=False)

            self.assertEqual(len(loaded.semantic_relations), len(processor.semantic_relations))

    def test_save_verbose_with_embeddings_and_relations(self):
        """Test save with verbose=True when embeddings and relations exist."""
        import io
        import sys

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks are computational models.")
        processor.process_document("doc2", "Deep learning uses neural networks for analysis.")
        processor.compute_all(verbose=False)
        processor.compute_graph_embeddings(dimensions=8, verbose=False)
        processor.extract_corpus_semantics(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            # Capture stdout
            captured = io.StringIO()
            sys.stdout = captured
            try:
                save_processor(
                    filepath, processor.layers, processor.documents,
                    processor.document_metadata, processor.embeddings,
                    processor.semantic_relations, verbose=True
                )
            finally:
                sys.stdout = sys.__stdout__

            output = captured.getvalue()
            # Check verbose output mentions embeddings and relations
            self.assertIn("Saved processor", output)
            self.assertIn("embeddings", output)
            self.assertIn("semantic relations", output)

    def test_load_verbose_with_embeddings_and_relations(self):
        """Test load with verbose=True when embeddings and relations exist."""
        import io
        import sys

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks are computational models.")
        processor.process_document("doc2", "Deep learning uses neural networks for analysis.")
        processor.compute_all(verbose=False)
        processor.compute_graph_embeddings(dimensions=8, verbose=False)
        processor.extract_corpus_semantics(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            save_processor(
                filepath, processor.layers, processor.documents,
                processor.document_metadata, processor.embeddings,
                processor.semantic_relations, verbose=False
            )

            # Capture stdout
            captured = io.StringIO()
            sys.stdout = captured
            try:
                load_processor(filepath, verbose=True)
            finally:
                sys.stdout = sys.__stdout__

            output = captured.getvalue()
            # Check verbose output mentions embeddings and relations
            self.assertIn("Loaded processor", output)
            self.assertIn("embeddings", output)
            self.assertIn("semantic relations", output)


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

    def test_load_embeddings_json(self):
        """Test loading embeddings from JSON."""
        embeddings = {'term1': [1.0, 2.0, 3.0], 'term2': [4.0, 5.0, 6.0]}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "embeddings.json")
            export_embeddings_json(filepath, embeddings)

            loaded = load_embeddings_json(filepath)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded['term1'], [1.0, 2.0, 3.0])
            self.assertEqual(loaded['term2'], [4.0, 5.0, 6.0])


class TestSemanticRelationsJSON(unittest.TestCase):
    """Test semantic relations JSON export/import."""

    def test_export_semantic_relations_json(self):
        """Test exporting semantic relations to JSON."""
        relations = [
            ('neural', 'RelatedTo', 'network', 0.8),
            ('machine', 'IsA', 'learning', 0.9),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "relations.json")
            export_semantic_relations_json(filepath, relations)

            self.assertTrue(os.path.exists(filepath))

            with open(filepath) as f:
                data = json.load(f)
            self.assertIn('relations', data)
            self.assertEqual(data['count'], 2)

    def test_load_semantic_relations_json(self):
        """Test loading semantic relations from JSON."""
        relations = [
            ('neural', 'RelatedTo', 'network', 0.8),
            ('deep', 'RelatedTo', 'learning', 0.7),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "relations.json")
            export_semantic_relations_json(filepath, relations)

            loaded = load_semantic_relations_json(filepath)
            self.assertEqual(len(loaded), 2)


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


class TestExportConceptNetJSON(unittest.TestCase):
    """Test ConceptNet-style graph export."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with sample data."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", """
            Neural networks are a type of machine learning model.
            Deep learning uses neural networks for pattern recognition.
        """)
        cls.processor.process_document("doc2", """
            Machine learning algorithms process data efficiently.
            Pattern recognition is used for image classification.
        """)
        cls.processor.compute_all(verbose=False)
        cls.processor.extract_corpus_semantics(verbose=False)

    def test_export_conceptnet_json_creates_file(self):
        """Test that export creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            export_conceptnet_json(filepath, self.processor.layers, verbose=False)
            self.assertTrue(os.path.exists(filepath))

    def test_export_conceptnet_json_structure(self):
        """Test exported JSON structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(filepath, self.processor.layers, verbose=False)

            self.assertIn('nodes', result)
            self.assertIn('edges', result)
            self.assertIn('metadata', result)

            # Check metadata
            self.assertIn('node_count', result['metadata'])
            self.assertIn('edge_count', result['metadata'])
            self.assertIn('layers', result['metadata'])
            self.assertIn('edge_types', result['metadata'])
            self.assertIn('relation_types', result['metadata'])

    def test_export_conceptnet_json_node_structure(self):
        """Test node structure in exported JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(filepath, self.processor.layers, verbose=False)

            for node in result['nodes']:
                self.assertIn('id', node)
                self.assertIn('label', node)
                self.assertIn('layer', node)
                self.assertIn('layer_name', node)
                self.assertIn('color', node)
                self.assertIn('pagerank', node)
                # Color should be valid hex
                self.assertTrue(node['color'].startswith('#'))

    def test_export_conceptnet_json_edge_structure(self):
        """Test edge structure in exported JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(filepath, self.processor.layers, verbose=False)

            for edge in result['edges']:
                self.assertIn('source', edge)
                self.assertIn('target', edge)
                self.assertIn('weight', edge)
                self.assertIn('relation_type', edge)
                self.assertIn('edge_type', edge)
                self.assertIn('color', edge)

    def test_export_conceptnet_json_layer_colors(self):
        """Test that nodes have correct layer colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(filepath, self.processor.layers, verbose=False)

            for node in result['nodes']:
                layer = CorticalLayer(node['layer'])
                expected_color = LAYER_COLORS.get(layer, '#808080')
                self.assertEqual(node['color'], expected_color)

    def test_export_conceptnet_json_with_semantic_relations(self):
        """Test export with semantic relations included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(
                filepath,
                self.processor.layers,
                semantic_relations=self.processor.semantic_relations,
                verbose=False
            )

            # Should have edges
            self.assertGreater(len(result['edges']), 0)

    def test_export_conceptnet_json_cross_layer_edges(self):
        """Test that cross-layer edges are included when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(
                filepath,
                self.processor.layers,
                include_cross_layer=True,
                verbose=False
            )

            edge_types = result['metadata'].get('edge_types', {})
            # May have cross_layer edges if there are feedforward/feedback connections
            self.assertIsInstance(edge_types, dict)

    def test_export_conceptnet_json_no_cross_layer(self):
        """Test export without cross-layer edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(
                filepath,
                self.processor.layers,
                include_cross_layer=False,
                verbose=False
            )

            # No cross_layer edges should be present
            cross_layer_count = result['metadata'].get('edge_types', {}).get('cross_layer', 0)
            self.assertEqual(cross_layer_count, 0)

    def test_export_conceptnet_json_max_nodes(self):
        """Test limiting nodes per layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(
                filepath,
                self.processor.layers,
                max_nodes_per_layer=5,
                verbose=False
            )

            # Count nodes per layer
            layer_counts = {}
            for node in result['nodes']:
                layer = node['layer']
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

            # Each layer should have at most 5 nodes
            for layer, count in layer_counts.items():
                self.assertLessEqual(count, 5)

    def test_export_conceptnet_json_min_weight(self):
        """Test filtering edges by minimum weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = export_conceptnet_json(
                filepath,
                self.processor.layers,
                min_weight=0.5,
                verbose=False
            )

            for edge in result['edges']:
                self.assertGreaterEqual(edge['weight'], 0.5)

    def test_layer_colors_constant(self):
        """Test that LAYER_COLORS constant is defined."""
        self.assertIn(CorticalLayer.TOKENS, LAYER_COLORS)
        self.assertIn(CorticalLayer.BIGRAMS, LAYER_COLORS)
        self.assertIn(CorticalLayer.CONCEPTS, LAYER_COLORS)
        self.assertIn(CorticalLayer.DOCUMENTS, LAYER_COLORS)

    def test_layer_names_constant(self):
        """Test that LAYER_NAMES constant is defined."""
        self.assertIn(CorticalLayer.TOKENS, LAYER_NAMES)
        self.assertEqual(LAYER_NAMES[CorticalLayer.TOKENS], 'Tokens')
        self.assertEqual(LAYER_NAMES[CorticalLayer.BIGRAMS], 'Bigrams')

    def test_processor_export_conceptnet_json(self):
        """Test processor-level export method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "conceptnet.json")
            result = self.processor.export_conceptnet_json(filepath, verbose=False)

            self.assertIn('nodes', result)
            self.assertIn('edges', result)
            self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main(verbosity=2)
