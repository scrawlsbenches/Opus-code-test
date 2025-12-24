"""
Unit tests for cortical/processor/persistence_api.py

Focuses on edge cases and error paths to improve coverage.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.layers import CorticalLayer, HierarchicalLayer


class TestPersistenceAPI:
    """Test persistence API methods."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test basic save/load roundtrip."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.compute_all()

        save_path = str(tmp_path / "test_state")
        processor.save(save_path)

        loaded = CorticalTextProcessor.load(save_path)
        assert loaded is not None
        assert "doc1" in loaded.documents
        assert loaded.documents["doc1"] == "neural networks process data"

    def test_load_with_invalid_config(self, tmp_path):
        """Test loading with corrupted config metadata (lines 94-95)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all()

        save_path = str(tmp_path / "test_state")
        processor.save(save_path)

        # Corrupt the config in manifest
        manifest_path = Path(save_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Make config invalid (not a dict)
        manifest['config'] = "invalid_config_string"

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Should still load but with default config
        loaded = CorticalTextProcessor.load(save_path)
        assert loaded is not None
        assert loaded.config is not None

    def test_load_missing_config_key(self, tmp_path):
        """Test loading with missing config key in metadata (line 94)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all()

        save_path = str(tmp_path / "test_state")
        processor.save(save_path)

        # Corrupt the config to cause KeyError
        manifest_path = Path(save_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Make config dict incomplete (missing required keys)
        manifest['config'] = {}

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Should still load but with default config
        loaded = CorticalTextProcessor.load(save_path)
        assert loaded is not None

    def test_load_without_doc_lengths_backward_compat(self, tmp_path):
        """Test backward compatibility when doc_lengths not in metadata (lines 111-117)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks machine learning")
        processor.process_document("doc2", "deep learning neural models")
        processor.compute_all()

        save_path = str(tmp_path / "test_state")
        processor.save(save_path)

        # Manually modify the saved state to remove BM25 doc_lengths from manifest
        # The manifest uses 'bm25_doc_lengths' which gets mapped to 'doc_lengths'
        # in metadata by persistence.load_processor()
        manifest_file = Path(save_path) / "manifest.json"

        # Read and modify manifest
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        # Remove BM25 doc length metadata to simulate old format
        if 'bm25_doc_lengths' in manifest:
            del manifest['bm25_doc_lengths']
        if 'avg_doc_length' in manifest:
            del manifest['avg_doc_length']

        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Now load - should trigger backward compat path (lines 111-117)
        loaded = CorticalTextProcessor.load(save_path)
        assert loaded is not None

        # The backward compat code should have recomputed doc_lengths
        assert len(loaded.documents) == 2
        # Check that doc_lengths were recomputed
        assert len(loaded.doc_lengths) == 2
        assert "doc1" in loaded.doc_lengths
        assert "doc2" in loaded.doc_lengths
        assert loaded.doc_lengths["doc1"] > 0
        assert loaded.doc_lengths["doc2"] > 0
        # Check that avg was computed
        assert loaded.avg_doc_length > 0

    def test_load_empty_processor_no_doc_lengths(self, tmp_path):
        """Test loading empty processor doesn't crash when computing doc_lengths (line 110)."""
        processor = CorticalTextProcessor()
        # Don't add any documents

        save_path = str(tmp_path / "test_state")
        processor.save(save_path)

        # Remove doc_lengths from manifest
        manifest_path = Path(save_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if 'doc_lengths' in manifest:
            del manifest['doc_lengths']

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Should handle empty docs gracefully
        loaded = CorticalTextProcessor.load(save_path)
        assert loaded is not None
        assert len(loaded.doc_lengths) == 0

    def test_load_json_with_stale_computations(self, tmp_path):
        """Test loading preserves stale computations (line 208-209)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all()

        # Mark some computations as stale (accessing private method for testing)
        processor._stale_computations.add(processor.COMP_PAGERANK)
        processor._stale_computations.add(processor.COMP_TFIDF)

        save_path = str(tmp_path / "test_state")
        processor.save_json(save_path)

        loaded = CorticalTextProcessor.load_json(save_path)
        assert loaded is not None
        # Stale computations should be restored
        assert processor.COMP_PAGERANK in loaded._stale_computations or \
               processor.COMP_TFIDF in loaded._stale_computations

    def test_export_conceptnet_json_verbose(self, tmp_path):
        """Test export_conceptnet_json with verbose output (line 250)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.compute_all()

        export_path = str(tmp_path / "conceptnet.json")

        # Test with verbose=True
        result = processor.export_conceptnet_json(
            export_path,
            include_cross_layer=True,
            include_typed_edges=True,
            verbose=True
        )

        assert result is not None
        assert 'nodes' in result
        assert 'edges' in result

    def test_export_conceptnet_with_filters(self, tmp_path):
        """Test export_conceptnet_json with various filter parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.process_document("doc2", "deep learning models")
        processor.compute_all()
        processor.extract_corpus_semantics()

        export_path = str(tmp_path / "conceptnet_filtered.json")

        result = processor.export_conceptnet_json(
            export_path,
            include_cross_layer=False,
            include_typed_edges=True,
            min_weight=0.5,
            min_confidence=0.3,
            max_nodes_per_layer=50,
            verbose=False
        )

        assert result is not None

    def test_save_json_force_flag(self, tmp_path):
        """Test save_json with force flag."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all()

        save_path = str(tmp_path / "test_state")

        # Save once
        result1 = processor.save_json(save_path, force=False, verbose=False)
        assert result1 is not None

        # Save again with force=True (should write even if unchanged)
        result2 = processor.save_json(save_path, force=True, verbose=True)
        assert result2 is not None

    def test_load_json_with_custom_config(self, tmp_path):
        """Test load_json with custom config override."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all()

        save_path = str(tmp_path / "test_state")
        processor.save_json(save_path)

        # Load with custom config
        custom_config = CorticalConfig(pagerank_damping=0.9)
        loaded = CorticalTextProcessor.load_json(save_path, config=custom_config)

        assert loaded is not None
        assert loaded.config.pagerank_damping == 0.9

    def test_export_graph_with_layer_filter(self, tmp_path):
        """Test export_graph with specific layer."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.compute_all()

        from cortical.layers import CorticalLayer

        export_path = str(tmp_path / "graph.json")
        result = processor.export_graph(
            export_path,
            layer=CorticalLayer.TOKENS,
            max_nodes=100
        )

        assert result is not None

    def test_save_with_no_stale_computations_attr(self, tmp_path):
        """Test save when _stale_computations attribute doesn't exist (line 53)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        # Remove _stale_computations if it exists
        if hasattr(processor, '_stale_computations'):
            delattr(processor, '_stale_computations')

        save_path = str(tmp_path / "test_state")
        # Should not crash
        processor.save(save_path)

        loaded = CorticalTextProcessor.load(save_path)
        assert loaded is not None

    def test_save_json_with_no_stale_computations_attr(self, tmp_path):
        """Test save_json when _stale_computations attribute doesn't exist (line 141)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        # Remove _stale_computations if it exists
        if hasattr(processor, '_stale_computations'):
            delattr(processor, '_stale_computations')

        save_path = str(tmp_path / "test_state")
        # Should not crash
        result = processor.save_json(save_path)
        assert result is not None

    def test_backward_compat_doc_lengths_recompute(self):
        """Test backward compatibility: recompute doc_lengths when missing (lines 111-117)."""
        # Mock persistence.load_processor to return metadata without doc_lengths
        mock_layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS),
        }
        mock_documents = {
            "doc1": "neural networks process data",
            "doc2": "deep learning models"
        }
        mock_metadata_no_lengths = {
            'config': CorticalConfig().to_dict()
            # Intentionally missing 'doc_lengths' and 'avg_doc_length'
        }

        with patch('cortical.persistence.load_processor') as mock_load:
            mock_load.return_value = (
                mock_layers,
                mock_documents,
                {},  # document_metadata
                {},  # embeddings
                [],  # semantic_relations
                mock_metadata_no_lengths
            )

            # This should trigger the backward compat code path (lines 111-117)
            loaded = CorticalTextProcessor.load("fake_path")

            # Verify doc_lengths were recomputed
            assert len(loaded.doc_lengths) == 2
            assert "doc1" in loaded.doc_lengths
            assert "doc2" in loaded.doc_lengths
            assert loaded.doc_lengths["doc1"] > 0
            assert loaded.doc_lengths["doc2"] > 0
            # Verify avg_doc_length was computed
            assert loaded.avg_doc_length > 0
            expected_avg = sum(loaded.doc_lengths.values()) / len(loaded.doc_lengths)
            assert loaded.avg_doc_length == pytest.approx(expected_avg)

    def test_backward_compat_empty_tokens_edge_case(self):
        """Test backward compat when documents have no valid tokens (line 116)."""
        # Edge case: documents exist but yield no tokens
        mock_layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS),
        }
        # Documents with only stop words / punctuation that tokenize to empty
        mock_documents = {
            "doc1": "... ... ...",
            "doc2": ". . ."
        }
        mock_metadata_no_lengths = {}

        with patch('cortical.persistence.load_processor') as mock_load:
            mock_load.return_value = (
                mock_layers,
                mock_documents,
                {},
                {},
                [],
                mock_metadata_no_lengths
            )

            loaded = CorticalTextProcessor.load("fake_path")

            # Documents exist but may have no valid tokens
            # The backward compat code should handle this gracefully
            assert loaded is not None
            assert len(loaded.documents) == 2
