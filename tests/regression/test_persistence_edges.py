"""
Regression Tests for Persistence Edge Cases
===========================================

Tests for file I/O edge cases, format validation, and data integrity.
These prevent silent data corruption and ensure clear error messages.

Regression test for T-018: Edge case coverage for production robustness.
"""

import pytest
import os
import json
import tempfile
import pickle
from cortical import CorticalTextProcessor


class TestFileFormatValidation:
    """Test file format detection and validation."""

    def test_load_nonexistent_file(self):
        """
        Loading nonexistent file should raise FileNotFoundError.

        Regression test for T-018: Clear error for missing files.
        """
        with pytest.raises(FileNotFoundError):
            CorticalTextProcessor.load("/nonexistent/path/to/file.pkl")

    def test_load_invalid_pickle_file(self):
        """
        Loading corrupted pickle file should raise appropriate error.

        Regression test for T-018: Graceful handling of corrupted data.
        """
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Write invalid pickle data
            f.write(b"This is not valid pickle data!")
            temp_path = f.name

        try:
            with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
                CorticalTextProcessor.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_empty_file(self):
        """
        Loading empty file should raise appropriate error.

        Regression test for T-018: Edge case for zero-byte files.
        """
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Write nothing (empty file)
            temp_path = f.name

        try:
            with pytest.raises((EOFError, pickle.UnpicklingError, ValueError)):
                CorticalTextProcessor.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestSaveToReadOnlyLocation:
    """Test saving to inaccessible locations."""

    def test_save_to_readonly_directory(self, small_processor):
        """
        Saving to read-only directory should raise PermissionError or OSError.

        Regression test for T-018: Clear error for permission issues.
        """
        # Try to save to /dev/null directory (or other readonly location)
        readonly_path = "/dev/null/corpus.pkl"

        with pytest.raises((PermissionError, OSError, IOError)):
            small_processor.save(readonly_path)

    def test_save_to_invalid_path(self, small_processor):
        """
        Saving to invalid path should raise appropriate error.

        Regression test for T-018: Path validation.

        Note: This test is intentionally lenient as save() may create
        intermediate directories. The key is that it doesn't silently fail.
        """
        # Use a path that truly cannot be created (permission denied)
        # /proc is read-only on Linux
        if os.path.exists('/proc'):
            invalid_path = "/proc/cannot_write_here/corpus.pkl"

            with pytest.raises((PermissionError, OSError)):
                small_processor.save(invalid_path, format='pickle')
        else:
            # Skip on systems without /proc
            pytest.skip("No read-only filesystem available for testing")


class TestSaveLoadRoundTrip:
    """Test save/load round-trip edge cases."""

    def test_save_empty_processor(self, fresh_processor):
        """
        Saving empty processor should work and reload correctly.

        Regression test for T-018: Edge case with no documents.
        """
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Save empty processor (specify pickle format)
            fresh_processor.save(temp_path, verbose=False, format='pickle')

            # Load it back
            loaded = CorticalTextProcessor.load(temp_path, verbose=False)

            # Should have no documents
            assert len(loaded.documents) == 0
            from cortical import CorticalLayer
            assert loaded.get_layer(CorticalLayer.TOKENS).column_count() == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_overwrite_existing(self, small_processor):
        """
        Saving to existing file should overwrite correctly.

        Regression test for T-018: Overwrite behavior.
        """
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Save once (specify pickle format)
            small_processor.save(temp_path, verbose=False, format='pickle')
            first_size = os.path.getsize(temp_path)

            # Save again (overwrite)
            small_processor.save(temp_path, verbose=False, format='pickle')
            second_size = os.path.getsize(temp_path)

            # Sizes should be similar (same data)
            assert abs(first_size - second_size) < 1000  # Within 1KB
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_preserves_staleness_state(self):
        """
        Loading processor should preserve staleness tracking.

        Regression test for T-018: State preservation across save/load.
        """
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Create processor and compute
            processor = CorticalTextProcessor()
            processor.process_document("doc1", "test content")
            processor.compute_all(verbose=False)

            # All computations should be fresh
            assert not processor.is_stale(processor.COMP_TFIDF)
            assert not processor.is_stale(processor.COMP_PAGERANK)

            # Save (specify pickle format)
            processor.save(temp_path, verbose=False, format='pickle')

            # Load
            loaded = CorticalTextProcessor.load(temp_path, verbose=False)

            # Staleness state should be preserved (all fresh)
            assert not loaded.is_stale(loaded.COMP_TFIDF)
            assert not loaded.is_stale(loaded.COMP_PAGERANK)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestJSONExportEdgeCases:
    """Test JSON export edge cases."""

    def test_export_graph_empty_processor(self, fresh_processor):
        """
        Exporting graph from empty processor should work.

        Regression test for T-018: Edge case with no data.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Export graph (nodes and edges)
            fresh_processor.export_graph(temp_path)

            # Load and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)

            # Should have structure but empty
            assert 'nodes' in data
            assert 'edges' in data
            assert len(data['nodes']) == 0
            assert len(data['edges']) == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_embeddings_empty(self, fresh_processor):
        """
        Exporting embeddings when none exist should create valid JSON.

        Regression test for T-018: Edge case with no embeddings computed.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Export embeddings (none computed)
            from cortical.persistence import export_embeddings_json
            export_embeddings_json(temp_path, fresh_processor.embeddings)

            # Load and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)

            # Should have empty embeddings dict
            assert 'embeddings' in data
            assert data['embeddings'] == {}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_semantic_relations_empty(self, fresh_processor):
        """
        Exporting semantic relations when none exist should create valid JSON.

        Regression test for T-018: Edge case with no relations.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Export relations (none extracted)
            from cortical.persistence import export_semantic_relations_json
            export_semantic_relations_json(temp_path, fresh_processor.semantic_relations)

            # Load and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)

            # Should have empty relations list
            assert 'relations' in data
            assert data['relations'] == []
            assert data['count'] == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMetadataEdgeCases:
    """Test document metadata edge cases."""

    def test_get_metadata_nonexistent_document(self, small_processor):
        """
        Getting metadata for nonexistent document should return empty dict.

        Regression test for T-018: Graceful handling of missing documents.
        """
        metadata = small_processor.get_document_metadata("nonexistent_doc_id")

        # Should return empty dict, not crash
        assert metadata == {}

    def test_set_metadata_unicode_values(self, fresh_processor):
        """
        Setting metadata with unicode values should work.

        Regression test for T-018: UTF-8 support in metadata.
        """
        fresh_processor.process_document("doc1", "test content")

        # Set metadata with unicode (using **kwargs)
        fresh_processor.set_document_metadata("doc1",
            title='Données françaises',
            author='李明',
            tags=['データ', '🔬']
        )

        # Should retrieve correctly
        metadata = fresh_processor.get_document_metadata("doc1")
        assert metadata['title'] == 'Données françaises'
        assert metadata['author'] == '李明'
        assert '🔬' in metadata['tags']

    def test_metadata_preserved_across_save_load(self):
        """
        Document metadata should be preserved across save/load.

        Regression test for T-018: Metadata persistence.
        """
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor = CorticalTextProcessor()
            processor.process_document("doc1", "test content")
            # Set metadata using **kwargs
            processor.set_document_metadata("doc1",
                custom_field='custom_value',
                numeric=42
            )

            # Save (specify pickle format)
            processor.save(temp_path, verbose=False, format='pickle')

            # Load
            loaded = CorticalTextProcessor.load(temp_path, verbose=False)

            # Metadata should be preserved
            metadata = loaded.get_document_metadata("doc1")
            assert metadata['custom_field'] == 'custom_value'
            assert metadata['numeric'] == 42
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
