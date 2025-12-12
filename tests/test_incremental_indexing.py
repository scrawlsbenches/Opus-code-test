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
# unittest.mock removed - using assertLogs for verbose tests

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
        """Test verbose mode logs output."""
        with self.assertLogs('cortical.processor', level='INFO') as cm:
            self.processor.remove_document("doc1", verbose=True)
        # Should have logged something about removing
        output = '\n'.join(cm.output)
        self.assertIn('Removing', output)


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


class TestProgressTracker(unittest.TestCase):
    """Tests for ProgressTracker class."""

    def setUp(self):
        """Set up temporary directory for log files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_progress_tracker_init(self):
        """Test ProgressTracker initialization."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import ProgressTracker

        tracker = ProgressTracker(quiet=True)
        self.assertIsNotNone(tracker.start_time)
        self.assertEqual(tracker.phases, {})
        self.assertIsNone(tracker.current_phase)

    def test_progress_tracker_with_log_file(self):
        """Test ProgressTracker with log file output."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import ProgressTracker

        log_path = os.path.join(self.temp_dir, "test.log")
        tracker = ProgressTracker(log_file=log_path, quiet=True)
        tracker.log("Test message")

        # Flush handlers
        for handler in tracker.logger.handlers:
            handler.flush()

        self.assertTrue(os.path.exists(log_path))
        with open(log_path) as f:
            content = f.read()
        self.assertIn("Test message", content)

    def test_start_and_end_phase(self):
        """Test phase tracking."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import ProgressTracker

        tracker = ProgressTracker(quiet=True)

        tracker.start_phase("Test Phase", total_items=10)
        self.assertEqual(tracker.current_phase, "Test Phase")
        self.assertIn("Test Phase", tracker.phases)
        self.assertEqual(tracker.phases["Test Phase"].status, "running")

        tracker.end_phase("Test Phase")
        self.assertEqual(tracker.phases["Test Phase"].status, "completed")
        self.assertGreater(tracker.phases["Test Phase"].duration, 0)

    def test_update_progress(self):
        """Test progress updates within a phase."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import ProgressTracker

        tracker = ProgressTracker(quiet=True)
        tracker.start_phase("Processing", total_items=100)

        tracker.update_progress(25, "item_25")
        self.assertEqual(tracker.phases["Processing"].items_processed, 25)
        self.assertEqual(tracker.phases["Processing"].progress_pct, 25.0)

        tracker.update_progress(50, "item_50")
        self.assertEqual(tracker.phases["Processing"].items_processed, 50)
        self.assertEqual(tracker.phases["Processing"].progress_pct, 50.0)

    def test_warn_and_error(self):
        """Test warning and error tracking."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import ProgressTracker

        tracker = ProgressTracker(quiet=True)

        tracker.warn("Test warning")
        tracker.error("Test error")

        self.assertEqual(len(tracker.warnings), 1)
        self.assertEqual(len(tracker.errors), 1)
        self.assertIn("Test warning", tracker.warnings)
        self.assertIn("Test error", tracker.errors)

    def test_get_summary(self):
        """Test summary generation."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import ProgressTracker

        tracker = ProgressTracker(quiet=True)
        tracker.start_phase("Phase 1", total_items=5)
        tracker.update_progress(5)
        tracker.end_phase("Phase 1")
        tracker.warn("A warning")

        summary = tracker.get_summary()

        self.assertIn("total_duration", summary)
        self.assertIn("phases", summary)
        self.assertIn("Phase 1", summary["phases"])
        self.assertEqual(summary["warnings"], 1)
        self.assertEqual(summary["errors"], 0)


class TestPhaseStats(unittest.TestCase):
    """Tests for PhaseStats dataclass."""

    def test_phase_stats_duration(self):
        """Test duration calculation."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import PhaseStats
        import time

        phase = PhaseStats(name="test", start_time=time.time())
        time.sleep(0.01)
        phase.end_time = time.time()

        self.assertGreater(phase.duration, 0)
        self.assertLess(phase.duration, 1)

    def test_phase_stats_progress_pct(self):
        """Test progress percentage calculation."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import PhaseStats

        phase = PhaseStats(name="test", items_total=100, items_processed=25)
        self.assertEqual(phase.progress_pct, 25.0)

        phase.items_processed = 50
        self.assertEqual(phase.progress_pct, 50.0)

        # Edge case: zero total
        phase.items_total = 0
        self.assertEqual(phase.progress_pct, 0.0)


class TestTimeoutHandler(unittest.TestCase):
    """Tests for timeout handling."""

    def test_timeout_handler_no_timeout(self):
        """Test that timeout=0 means no timeout."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import timeout_handler

        # Should complete without issue
        with timeout_handler(0):
            result = 1 + 1
        self.assertEqual(result, 2)

    def test_timeout_handler_completes_in_time(self):
        """Test that operations completing in time succeed."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import timeout_handler

        with timeout_handler(5):
            result = sum(range(100))
        self.assertEqual(result, 4950)


class TestIndexingFunctions(unittest.TestCase):
    """Tests for indexing helper functions."""

    def setUp(self):
        """Set up temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

        # Create test file structure
        (self.base_path / "cortical").mkdir()
        (self.base_path / "tests").mkdir()
        (self.base_path / "cortical" / "test.py").write_text("# Test file\nprint('hello')")
        (self.base_path / "tests" / "test_test.py").write_text("# Test\nimport unittest")
        (self.base_path / "CLAUDE.md").write_text("# Documentation")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_python_files(self):
        """Test Python file discovery."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_python_files

        files = get_python_files(self.base_path)
        file_names = [f.name for f in files]

        self.assertIn("test.py", file_names)
        self.assertIn("test_test.py", file_names)

    def test_get_doc_files(self):
        """Test documentation file discovery."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_doc_files

        files = get_doc_files(self.base_path)
        file_names = [f.name for f in files]

        self.assertIn("CLAUDE.md", file_names)

    def test_create_doc_id(self):
        """Test document ID creation."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import create_doc_id

        file_path = self.base_path / "cortical" / "test.py"
        doc_id = create_doc_id(file_path, self.base_path)

        self.assertEqual(doc_id, "cortical/test.py")

    def test_get_file_mtime(self):
        """Test file modification time retrieval."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import get_file_mtime

        file_path = self.base_path / "CLAUDE.md"
        mtime = get_file_mtime(file_path)

        self.assertIsInstance(mtime, float)
        self.assertGreater(mtime, 0)

    def test_index_file(self):
        """Test single file indexing."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import index_file

        processor = CorticalTextProcessor()
        file_path = self.base_path / "cortical" / "test.py"

        metadata = index_file(processor, file_path, self.base_path)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['relative_path'], "cortical/test.py")
        self.assertEqual(metadata['file_type'], ".py")
        self.assertEqual(metadata['language'], "python")
        self.assertIn("cortical/test.py", processor.documents)

    def test_index_file_with_read_error(self):
        """Test handling of unreadable files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import index_file, ProgressTracker

        processor = CorticalTextProcessor()
        tracker = ProgressTracker(quiet=True)
        nonexistent = self.base_path / "nonexistent.py"

        metadata = index_file(processor, nonexistent, self.base_path, tracker)

        self.assertIsNone(metadata)
        self.assertEqual(len(tracker.warnings), 1)


class TestFullIndexFunction(unittest.TestCase):
    """Tests for full_index function."""

    def setUp(self):
        """Set up temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

        (self.base_path / "file1.py").write_text("# File 1\nprint('a')")
        (self.base_path / "file2.py").write_text("# File 2\nprint('b')")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_full_index(self):
        """Test full indexing of files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import full_index, ProgressTracker

        processor = CorticalTextProcessor()
        tracker = ProgressTracker(quiet=True)
        all_files = list(self.base_path.glob("*.py"))

        indexed, total_lines, file_mtimes = full_index(
            processor, all_files, self.base_path, tracker
        )

        self.assertEqual(indexed, 2)
        self.assertGreater(total_lines, 0)
        self.assertEqual(len(file_mtimes), 2)
        self.assertIn("Indexing files", tracker.phases)


class TestIncrementalIndexFunction(unittest.TestCase):
    """Tests for incremental_index function."""

    def setUp(self):
        """Set up temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

        (self.base_path / "existing.py").write_text("# Existing\nprint('x')")
        (self.base_path / "new.py").write_text("# New\nprint('y')")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_incremental_index_added_files(self):
        """Test incremental indexing of added files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import incremental_index, ProgressTracker

        processor = CorticalTextProcessor()
        tracker = ProgressTracker(quiet=True)

        added = [self.base_path / "new.py"]
        modified = []
        deleted = []

        added_count, modified_count, deleted_count, total_lines = incremental_index(
            processor, added, modified, deleted, self.base_path, tracker
        )

        self.assertEqual(added_count, 1)
        self.assertEqual(modified_count, 0)
        self.assertEqual(deleted_count, 0)
        self.assertIn("new.py", processor.documents)

    def test_incremental_index_modified_files(self):
        """Test incremental indexing of modified files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import incremental_index, index_file, ProgressTracker

        processor = CorticalTextProcessor()
        tracker = ProgressTracker(quiet=True)

        # First, index the existing file
        index_file(processor, self.base_path / "existing.py", self.base_path)

        # Now modify it (in our test, just re-index as modified)
        added = []
        modified = [self.base_path / "existing.py"]
        deleted = []

        added_count, modified_count, deleted_count, total_lines = incremental_index(
            processor, added, modified, deleted, self.base_path, tracker
        )

        self.assertEqual(modified_count, 1)

    def test_incremental_index_deleted_files(self):
        """Test incremental indexing handles deleted files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import incremental_index, index_file, ProgressTracker

        processor = CorticalTextProcessor()
        tracker = ProgressTracker(quiet=True)

        # First, index a file
        index_file(processor, self.base_path / "existing.py", self.base_path)
        self.assertIn("existing.py", processor.documents)

        # Now mark it as deleted
        added = []
        modified = []
        deleted = ["existing.py"]

        added_count, modified_count, deleted_count, total_lines = incremental_index(
            processor, added, modified, deleted, self.base_path, tracker
        )

        self.assertEqual(deleted_count, 1)
        self.assertNotIn("existing.py", processor.documents)


class TestComputeAnalysis(unittest.TestCase):
    """Tests for compute_analysis function."""

    def test_compute_analysis_fast_mode(self):
        """Test fast mode analysis."""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from index_codebase import compute_analysis, ProgressTracker

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks are powerful.")
        processor.process_document("doc2", "Machine learning algorithms.")

        tracker = ProgressTracker(quiet=True)
        compute_analysis(processor, tracker, fast_mode=True)

        self.assertIn("Computing analysis (fast mode)", tracker.phases)
        # TF-IDF should be computed
        layer0 = processor.layers[CorticalLayer.TOKENS]
        neural_col = layer0.get_minicolumn("neural")
        self.assertIsNotNone(neural_col)
        self.assertGreater(neural_col.tfidf, 0)


if __name__ == '__main__':
    unittest.main()
