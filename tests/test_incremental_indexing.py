"""
Tests for incremental indexing functionality.

Tests cover:
- remove_document() method in processor
- remove_minicolumn() method in layers
- Manifest file operations
- File change detection
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn


class TestRemoveDocument(unittest.TestCase):
    """Tests for CorticalTextProcessor.remove_document()"""

    def setUp(self):
        """Set up a processor with test documents."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Neural networks process information efficiently.")
        self.processor.process_document("doc2", "Machine learning algorithms learn patterns.")
        self.processor.process_document("doc3", "Neural machine translation uses deep learning.")
        self.processor.compute_all(verbose=False)

    def test_remove_document_basic(self):
        """Test basic document removal."""
        self.assertEqual(len(self.processor.documents), 3)

        result = self.processor.remove_document("doc1")

        self.assertTrue(result['found'])
        self.assertEqual(len(self.processor.documents), 2)
        self.assertNotIn("doc1", self.processor.documents)

    def test_remove_document_not_found(self):
        """Test removing a non-existent document."""
        result = self.processor.remove_document("nonexistent")

        self.assertFalse(result['found'])
        self.assertEqual(result['tokens_affected'], 0)
        self.assertEqual(result['bigrams_affected'], 0)

    def test_remove_document_cleans_token_document_ids(self):
        """Test that document ID is removed from token document_ids sets."""
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        # neural appears in doc1 and doc3
        neural_col = layer0.get_minicolumn("neural")
        self.assertIn("doc1", neural_col.document_ids)

        self.processor.remove_document("doc1")

        # neural should no longer reference doc1
        self.assertNotIn("doc1", neural_col.document_ids)
        # But should still reference doc3
        self.assertIn("doc3", neural_col.document_ids)

    def test_remove_document_cleans_bigram_document_ids(self):
        """Test that document ID is removed from bigram document_ids sets."""
        layer1 = self.processor.layers[CorticalLayer.BIGRAMS]

        # Find a bigram from doc1
        bigram_col = layer1.get_minicolumn("neural networks")
        if bigram_col:
            self.assertIn("doc1", bigram_col.document_ids)
            self.processor.remove_document("doc1")
            self.assertNotIn("doc1", bigram_col.document_ids)

    def test_remove_document_removes_layer3_minicolumn(self):
        """Test that the document minicolumn is removed from Layer 3."""
        layer3 = self.processor.layers[CorticalLayer.DOCUMENTS]

        self.assertIn("doc1", layer3.minicolumns)
        self.processor.remove_document("doc1")
        self.assertNotIn("doc1", layer3.minicolumns)

    def test_remove_document_removes_metadata(self):
        """Test that document metadata is removed."""
        self.processor.set_document_metadata("doc1", source="test")
        self.assertEqual(self.processor.get_document_metadata("doc1"), {"source": "test"})

        self.processor.remove_document("doc1")

        self.assertEqual(self.processor.get_document_metadata("doc1"), {})

    def test_remove_document_marks_stale(self):
        """Test that removal marks computations as stale."""
        # After compute_all, computations should not be stale
        self.assertFalse(self.processor.is_stale(self.processor.COMP_TFIDF))

        self.processor.remove_document("doc1")

        # After removal, computations should be stale
        self.assertTrue(self.processor.is_stale(self.processor.COMP_TFIDF))

    def test_remove_document_returns_affected_counts(self):
        """Test that removal returns correct affected counts."""
        result = self.processor.remove_document("doc1")

        self.assertTrue(result['found'])
        self.assertGreater(result['tokens_affected'], 0)
        self.assertGreater(result['bigrams_affected'], 0)

    def test_remove_document_verbose(self):
        """Test verbose mode prints output."""
        with patch('builtins.print') as mock_print:
            self.processor.remove_document("doc1", verbose=True)
            mock_print.assert_called()


class TestRemoveDocumentsBatch(unittest.TestCase):
    """Tests for CorticalTextProcessor.remove_documents_batch()"""

    def setUp(self):
        """Set up a processor with test documents."""
        self.processor = CorticalTextProcessor()
        for i in range(5):
            self.processor.process_document(f"doc{i}", f"Document {i} content here.")
        self.processor.compute_all(verbose=False)

    def test_remove_documents_batch_basic(self):
        """Test removing multiple documents."""
        result = self.processor.remove_documents_batch(["doc0", "doc1", "doc2"])

        self.assertEqual(result['documents_removed'], 3)
        self.assertEqual(result['documents_not_found'], 0)
        self.assertEqual(len(self.processor.documents), 2)

    def test_remove_documents_batch_with_missing(self):
        """Test removing documents when some don't exist."""
        result = self.processor.remove_documents_batch(["doc0", "nonexistent", "doc1"])

        self.assertEqual(result['documents_removed'], 2)
        self.assertEqual(result['documents_not_found'], 1)

    def test_remove_documents_batch_with_recompute_tfidf(self):
        """Test batch removal with TF-IDF recomputation."""
        result = self.processor.remove_documents_batch(["doc0"], recompute='tfidf')

        self.assertEqual(result['recomputation'], 'tfidf')
        self.assertFalse(self.processor.is_stale(self.processor.COMP_TFIDF))

    def test_remove_documents_batch_with_recompute_full(self):
        """Test batch removal with full recomputation."""
        result = self.processor.remove_documents_batch(["doc0"], recompute='full')

        self.assertEqual(result['recomputation'], 'full')
        self.assertEqual(len(self.processor.get_stale_computations()), 0)


class TestRemoveMinicolumn(unittest.TestCase):
    """Tests for HierarchicalLayer.remove_minicolumn()"""

    def setUp(self):
        """Set up a test layer with minicolumns."""
        self.layer = HierarchicalLayer(CorticalLayer.TOKENS)
        self.layer.get_or_create_minicolumn("test")
        self.layer.get_or_create_minicolumn("neural")
        self.layer.get_or_create_minicolumn("network")

    def test_remove_minicolumn_basic(self):
        """Test basic minicolumn removal."""
        self.assertEqual(self.layer.column_count(), 3)

        result = self.layer.remove_minicolumn("test")

        self.assertTrue(result)
        self.assertEqual(self.layer.column_count(), 2)
        self.assertNotIn("test", self.layer.minicolumns)

    def test_remove_minicolumn_not_found(self):
        """Test removing non-existent minicolumn."""
        result = self.layer.remove_minicolumn("nonexistent")

        self.assertFalse(result)
        self.assertEqual(self.layer.column_count(), 3)

    def test_remove_minicolumn_removes_from_id_index(self):
        """Test that removal updates the ID index."""
        col = self.layer.get_minicolumn("test")
        col_id = col.id

        self.assertIsNotNone(self.layer.get_by_id(col_id))

        self.layer.remove_minicolumn("test")

        self.assertIsNone(self.layer.get_by_id(col_id))


class TestManifestOperations(unittest.TestCase):
    """Tests for manifest file operations in index_codebase.py"""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_path = Path(self.temp_dir) / "test.manifest.json"

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_manifest(self):
        """Test saving a manifest file."""
        # Import the functions from the script
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import save_manifest, load_manifest

        files = {
            "cortical/processor.py": 1234567890.0,
            "tests/test_processor.py": 1234567891.0,
        }
        stats = {"documents": 2, "tokens": 100}

        save_manifest(self.manifest_path, files, "corpus.pkl", stats)

        self.assertTrue(self.manifest_path.exists())

        # Verify content
        with open(self.manifest_path) as f:
            data = json.load(f)

        self.assertEqual(data['version'], "1.0")
        self.assertEqual(data['corpus_path'], "corpus.pkl")
        self.assertEqual(len(data['files']), 2)
        self.assertEqual(data['stats']['documents'], 2)

    def test_load_manifest_valid(self):
        """Test loading a valid manifest file."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import save_manifest, load_manifest

        files = {"test.py": 1234567890.0}
        save_manifest(self.manifest_path, files, "corpus.pkl", {})

        manifest = load_manifest(self.manifest_path)

        self.assertIsNotNone(manifest)
        self.assertEqual(manifest['files'], files)

    def test_load_manifest_not_exists(self):
        """Test loading a non-existent manifest file."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import load_manifest

        manifest = load_manifest(Path(self.temp_dir) / "nonexistent.json")

        self.assertIsNone(manifest)

    def test_load_manifest_invalid_version(self):
        """Test loading a manifest with wrong version."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import load_manifest

        # Write manifest with wrong version
        with open(self.manifest_path, 'w') as f:
            json.dump({"version": "0.1", "files": {}}, f)

        manifest = load_manifest(self.manifest_path)

        self.assertIsNone(manifest)


class TestFileChangeDetection(unittest.TestCase):
    """Tests for file change detection."""

    def setUp(self):
        """Set up temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

        # Create some test files
        (self.base_path / "file1.py").write_text("content1")
        (self.base_path / "file2.py").write_text("content2")
        (self.base_path / "file3.py").write_text("content3")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_file_changes_no_changes(self):
        """Test detecting no changes."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_file_changes, get_file_mtime

        current_files = list(self.base_path.glob("*.py"))
        manifest = {
            'files': {
                str(f.relative_to(self.base_path)): get_file_mtime(f)
                for f in current_files
            }
        }

        added, modified, deleted = get_file_changes(manifest, current_files, self.base_path)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_get_file_changes_added_file(self):
        """Test detecting added files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_file_changes, get_file_mtime

        # Create manifest without file3.py
        manifest = {
            'files': {
                "file1.py": get_file_mtime(self.base_path / "file1.py"),
                "file2.py": get_file_mtime(self.base_path / "file2.py"),
            }
        }

        current_files = list(self.base_path.glob("*.py"))
        added, modified, deleted = get_file_changes(manifest, current_files, self.base_path)

        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].name, "file3.py")
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_get_file_changes_deleted_file(self):
        """Test detecting deleted files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_file_changes, get_file_mtime

        # Create manifest with an extra file that doesn't exist
        manifest = {
            'files': {
                "file1.py": get_file_mtime(self.base_path / "file1.py"),
                "file2.py": get_file_mtime(self.base_path / "file2.py"),
                "file3.py": get_file_mtime(self.base_path / "file3.py"),
                "deleted.py": 1234567890.0,  # This file doesn't exist
            }
        }

        current_files = list(self.base_path.glob("*.py"))
        added, modified, deleted = get_file_changes(manifest, current_files, self.base_path)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 1)
        self.assertIn("deleted.py", deleted)

    def test_get_file_changes_modified_file(self):
        """Test detecting modified files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_file_changes, get_file_mtime
        import time

        # Create manifest with old mtime
        manifest = {
            'files': {
                "file1.py": 0.0,  # Very old mtime
                "file2.py": get_file_mtime(self.base_path / "file2.py"),
                "file3.py": get_file_mtime(self.base_path / "file3.py"),
            }
        }

        current_files = list(self.base_path.glob("*.py"))
        added, modified, deleted = get_file_changes(manifest, current_files, self.base_path)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 1)
        self.assertEqual(modified[0].name, "file1.py")
        self.assertEqual(len(deleted), 0)


class TestIncrementalIndexingIntegration(unittest.TestCase):
    """Integration tests for incremental indexing workflow."""

    def test_add_remove_reindex_workflow(self):
        """Test the full workflow of add, remove, and reindex."""
        processor = CorticalTextProcessor()

        # Initial indexing
        processor.process_document("doc1", "Neural networks are powerful.")
        processor.process_document("doc2", "Machine learning is useful.")
        processor.compute_all(verbose=False)

        initial_doc_count = len(processor.documents)
        self.assertEqual(initial_doc_count, 2)

        # Remove a document
        result = processor.remove_document("doc1")
        self.assertTrue(result['found'])
        self.assertEqual(len(processor.documents), 1)

        # Add a new document
        processor.process_document("doc3", "Deep learning advances rapidly.")

        # Recompute
        processor.compute_all(verbose=False)

        # Verify final state
        self.assertEqual(len(processor.documents), 2)
        self.assertNotIn("doc1", processor.documents)
        self.assertIn("doc2", processor.documents)
        self.assertIn("doc3", processor.documents)

    def test_incremental_preserves_other_documents(self):
        """Test that incremental updates don't affect unchanged documents."""
        processor = CorticalTextProcessor()

        processor.process_document("doc1", "The quick brown fox.")
        processor.process_document("doc2", "Jumps over the lazy dog.")
        processor.compute_all(verbose=False)

        # Store original state of doc2
        layer0 = processor.layers[CorticalLayer.TOKENS]
        original_quick_docs = layer0.get_minicolumn("quick").document_ids.copy()

        # Remove doc1
        processor.remove_document("doc1")

        # doc2 tokens should still reference doc2
        lazy_col = layer0.get_minicolumn("lazy")
        self.assertIn("doc2", lazy_col.document_ids)


if __name__ == '__main__':
    unittest.main()
