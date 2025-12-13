"""
Tests for the FluentProcessor API.

Ensures the fluent/chainable interface works correctly.
"""

import os
import tempfile
import unittest

from cortical import CorticalTextProcessor, CorticalConfig
from cortical.fluent import FluentProcessor


class TestFluentProcessorInit(unittest.TestCase):
    """Test FluentProcessor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        fp = FluentProcessor()
        self.assertIsNotNone(fp)
        self.assertIsInstance(fp.processor, CorticalTextProcessor)
        self.assertFalse(fp.is_built)

    def test_init_with_config(self):
        """Test initialization with config."""
        config = CorticalConfig(pagerank_damping=0.9)
        fp = FluentProcessor(config=config)
        self.assertEqual(fp.processor.config.pagerank_damping, 0.9)

    def test_from_existing(self):
        """Test wrapping existing processor."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "Test content")
        fp = FluentProcessor.from_existing(proc)
        self.assertEqual(fp.processor, proc)

    def test_repr(self):
        """Test string representation."""
        fp = FluentProcessor()
        self.assertIn("FluentProcessor", repr(fp))


class TestFluentProcessorChaining(unittest.TestCase):
    """Test method chaining."""

    def test_add_document_returns_self(self):
        """add_document returns self for chaining."""
        fp = FluentProcessor()
        result = fp.add_document("doc1", "Content")
        self.assertIs(result, fp)

    def test_add_documents_returns_self(self):
        """add_documents returns self for chaining."""
        fp = FluentProcessor()
        result = fp.add_documents({"doc1": "Content"})
        self.assertIs(result, fp)

    def test_build_returns_self(self):
        """build returns self for chaining."""
        fp = FluentProcessor()
        fp.add_document("doc1", "Content")
        result = fp.build(verbose=False)
        self.assertIs(result, fp)

    def test_full_chain(self):
        """Test complete method chain."""
        results = (FluentProcessor()
            .add_document("doc1", "Neural networks process data")
            .add_document("doc2", "Machine learning is powerful")
            .build(verbose=False)
            .search("neural", top_n=2))
        self.assertIsInstance(results, list)


class TestFluentProcessorDocuments(unittest.TestCase):
    """Test document handling."""

    def test_add_single_document(self):
        """Test adding a single document."""
        fp = FluentProcessor()
        fp.add_document("doc1", "Test content here")
        self.assertIn("doc1", fp.processor.documents)

    def test_add_document_with_metadata(self):
        """Test adding document with metadata."""
        fp = FluentProcessor()
        fp.add_document("doc1", "Content", metadata={"author": "test"})
        self.assertIn("doc1", fp.processor.documents)

    def test_add_documents_dict(self):
        """Test adding multiple documents from dict."""
        fp = FluentProcessor()
        fp.add_documents({
            "doc1": "Content one",
            "doc2": "Content two"
        })
        self.assertIn("doc1", fp.processor.documents)
        self.assertIn("doc2", fp.processor.documents)

    def test_add_documents_tuples(self):
        """Test adding documents from list of tuples."""
        fp = FluentProcessor()
        fp.add_documents([
            ("doc1", "Content one"),
            ("doc2", "Content two")
        ])
        self.assertIn("doc1", fp.processor.documents)
        self.assertIn("doc2", fp.processor.documents)

    def test_add_documents_invalid_type(self):
        """Test error on invalid input type."""
        fp = FluentProcessor()
        with self.assertRaises(TypeError):
            fp.add_documents("invalid")


class TestFluentProcessorBuild(unittest.TestCase):
    """Test build functionality."""

    def test_build_marks_built(self):
        """Test that build marks processor as built."""
        fp = FluentProcessor()
        fp.add_document("doc1", "Content")
        self.assertFalse(fp.is_built)
        fp.build(verbose=False)
        self.assertTrue(fp.is_built)

    def test_add_after_build_marks_unbuilt(self):
        """Test adding document after build marks as unbuilt."""
        fp = FluentProcessor()
        fp.add_document("doc1", "Content")
        fp.build(verbose=False)
        self.assertTrue(fp.is_built)
        fp.add_document("doc2", "More content")
        self.assertFalse(fp.is_built)


class TestFluentProcessorSearch(unittest.TestCase):
    """Test search functionality."""

    def setUp(self):
        """Set up a built processor."""
        self.fp = (FluentProcessor()
            .add_document("neural", "Neural networks are computational models")
            .add_document("ml", "Machine learning algorithms learn from data")
            .add_document("deep", "Deep learning uses neural network layers")
            .build(verbose=False))

    def test_search_returns_results(self):
        """Test basic search returns results."""
        results = self.fp.search("neural", top_n=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_search_result_structure(self):
        """Test search result tuple structure."""
        results = self.fp.search("neural", top_n=1)
        self.assertEqual(len(results[0]), 2)  # (doc_id, score)

    def test_fast_search(self):
        """Test fast search method."""
        results = self.fp.fast_search("neural", top_n=3)
        self.assertIsInstance(results, list)

    def test_search_passages(self):
        """Test passage search."""
        results = self.fp.search_passages("neural", top_n=2)
        self.assertIsInstance(results, list)

    def test_expand_query(self):
        """Test query expansion."""
        expanded = self.fp.expand("neural", max_expansions=5)
        self.assertIsInstance(expanded, dict)


class TestFluentProcessorPersistence(unittest.TestCase):
    """Test save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading processor."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            # Save
            (FluentProcessor()
                .add_document("doc1", "Test content")
                .build(verbose=False)
                .save(path))

            # Load
            fp = FluentProcessor.load(path)
            self.assertTrue(fp.is_built)
            self.assertIn("doc1", fp.processor.documents)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestFluentProcessorFiles(unittest.TestCase):
    """Test file loading functionality."""

    def test_from_files(self):
        """Test loading from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "test1.txt")
            file2 = os.path.join(tmpdir, "test2.txt")
            with open(file1, 'w') as f:
                f.write("Content of file one")
            with open(file2, 'w') as f:
                f.write("Content of file two")

            fp = FluentProcessor.from_files([file1, file2])
            self.assertEqual(len(fp.processor.documents), 2)

    def test_from_directory(self):
        """Test loading from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                path = os.path.join(tmpdir, f"test{i}.txt")
                with open(path, 'w') as f:
                    f.write(f"Content of file {i}")

            fp = FluentProcessor.from_directory(tmpdir, pattern="*.txt")
            self.assertEqual(len(fp.processor.documents), 3)


if __name__ == '__main__':
    unittest.main()
