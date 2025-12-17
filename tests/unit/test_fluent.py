"""
Unit tests for the FluentProcessor API.

Tests the fluent/chainable interface for CorticalTextProcessor.
"""

import os
import pytest
import tempfile
from pathlib import Path

from cortical import CorticalTextProcessor, CorticalConfig, Tokenizer
from cortical.fluent import FluentProcessor


class TestFluentProcessorBasics:
    """Test basic FluentProcessor initialization and properties."""

    def test_init_default(self):
        """Test default initialization."""
        processor = FluentProcessor()
        assert processor is not None
        assert isinstance(processor.processor, CorticalTextProcessor)
        assert not processor.is_built

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = CorticalConfig(pagerank_damping=0.9)
        processor = FluentProcessor(config=config)
        assert processor.processor.config.pagerank_damping == 0.9

    def test_init_with_tokenizer(self):
        """Test initialization with custom tokenizer."""
        tokenizer = Tokenizer(split_identifiers=True)
        processor = FluentProcessor(tokenizer=tokenizer)
        assert processor.processor.tokenizer.split_identifiers

    def test_from_existing(self):
        """Test creating FluentProcessor from existing processor."""
        raw = CorticalTextProcessor()
        raw.process_document("doc1", "test content")

        fluent = FluentProcessor.from_existing(raw)
        assert fluent.processor is raw
        assert len(fluent.processor.documents) == 1

    def test_repr(self):
        """Test string representation."""
        processor = FluentProcessor()
        assert "documents=0" in repr(processor)
        assert "not built" in repr(processor)

        processor.add_document("doc1", "test")
        assert "documents=1" in repr(processor)


class TestFluentProcessorChaining:
    """Test method chaining functionality."""

    def test_add_document_returns_self(self):
        """Test that add_document returns self for chaining."""
        processor = FluentProcessor()
        result = processor.add_document("doc1", "content")
        assert result is processor

    def test_add_documents_dict_returns_self(self):
        """Test that add_documents returns self for chaining."""
        processor = FluentProcessor()
        result = processor.add_documents({"doc1": "content1", "doc2": "content2"})
        assert result is processor

    def test_build_returns_self(self):
        """Test that build returns self for chaining."""
        processor = FluentProcessor()
        processor.add_document("doc1", "neural networks process information")
        result = processor.build(verbose=False)
        assert result is processor
        assert processor.is_built

    def test_save_returns_self(self):
        """Test that save returns self for chaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "corpus_state")

            processor = FluentProcessor()
            processor.add_document("doc1", "test content")
            processor.build(verbose=False)
            result = processor.save(state_path)
            assert result is processor

    def test_with_config_returns_self(self):
        """Test that with_config returns self for chaining."""
        processor = FluentProcessor()
        config = CorticalConfig(pagerank_iterations=30)
        result = processor.with_config(config)
        assert result is processor

    def test_with_tokenizer_returns_self(self):
        """Test that with_tokenizer returns self for chaining."""
        processor = FluentProcessor()
        tokenizer = Tokenizer(split_identifiers=True)
        result = processor.with_tokenizer(tokenizer)
        assert result is processor

    def test_full_chain(self):
        """Test a complete method chain."""
        result = (FluentProcessor()
            .add_document("doc1", "neural networks process information efficiently")
            .add_document("doc2", "deep learning uses neural network architectures")
            .build(verbose=False)
            .search("neural processing"))

        assert isinstance(result, list)
        assert len(result) > 0


class TestFluentProcessorDocuments:
    """Test document addition methods."""

    def test_add_document_single(self):
        """Test adding a single document."""
        processor = FluentProcessor()
        processor.add_document("doc1", "test content")
        assert len(processor.processor.documents) == 1
        assert processor.processor.documents["doc1"] == "test content"
        assert not processor.is_built

    def test_add_document_with_metadata(self):
        """Test adding document with metadata."""
        processor = FluentProcessor()
        metadata = {"author": "Alice", "date": "2025-01-01"}
        processor.add_document("doc1", "test content", metadata=metadata)
        assert processor.processor.document_metadata["doc1"]["author"] == "Alice"

    def test_add_documents_from_dict(self):
        """Test adding multiple documents from dict."""
        processor = FluentProcessor()
        docs = {
            "doc1": "content 1",
            "doc2": "content 2",
            "doc3": "content 3"
        }
        processor.add_documents(docs)
        assert len(processor.processor.documents) == 3
        assert processor.processor.documents["doc2"] == "content 2"

    def test_add_documents_from_tuples(self):
        """Test adding documents from list of tuples."""
        processor = FluentProcessor()
        docs = [
            ("doc1", "content 1"),
            ("doc2", "content 2")
        ]
        processor.add_documents(docs)
        assert len(processor.processor.documents) == 2

    def test_add_documents_from_tuples_with_metadata(self):
        """Test adding documents from tuples with metadata."""
        processor = FluentProcessor()
        docs = [
            ("doc1", "content 1", {"author": "Alice"}),
            ("doc2", "content 2", {"author": "Bob"})
        ]
        processor.add_documents(docs)
        assert processor.processor.document_metadata["doc1"]["author"] == "Alice"
        assert processor.processor.document_metadata["doc2"]["author"] == "Bob"

    def test_add_documents_invalid_type(self):
        """Test that invalid document type raises error."""
        processor = FluentProcessor()
        with pytest.raises(TypeError):
            processor.add_documents("invalid")

    def test_add_documents_invalid_tuple_length(self):
        """Test that invalid tuple length raises error."""
        processor = FluentProcessor()
        with pytest.raises(ValueError, match="Invalid document tuple"):
            processor.add_documents([("doc1",)])  # Too short


class TestFluentProcessorFiles:
    """Test file and directory loading methods."""

    def test_from_files(self):
        """Test loading from file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "doc1.txt"
            file2 = Path(tmpdir) / "doc2.txt"
            file1.write_text("content 1")
            file2.write_text("content 2")

            processor = FluentProcessor.from_files([file1, file2])
            assert len(processor.processor.documents) == 2
            assert "doc1" in processor.processor.documents
            assert "doc2" in processor.processor.documents
            assert processor.processor.documents["doc1"] == "content 1"

    def test_from_files_missing_file(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            FluentProcessor.from_files(["/nonexistent/file.txt"])

    def test_from_files_not_a_file(self):
        """Test that directory path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Not a file"):
                FluentProcessor.from_files([tmpdir])

    def test_from_directory_default_pattern(self):
        """Test loading from directory with default pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("content 1")
            (Path(tmpdir) / "doc2.txt").write_text("content 2")
            (Path(tmpdir) / "readme.md").write_text("readme content")

            processor = FluentProcessor.from_directory(tmpdir)
            assert len(processor.processor.documents) == 2  # Only .txt files
            assert "doc1" in processor.processor.documents

    def test_from_directory_custom_pattern(self):
        """Test loading from directory with custom pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc1.txt").write_text("content 1")
            (Path(tmpdir) / "readme.md").write_text("readme content")

            processor = FluentProcessor.from_directory(tmpdir, pattern="*.md")
            assert len(processor.processor.documents) == 1
            assert "readme" in processor.processor.documents

    def test_from_directory_recursive(self):
        """Test recursive directory loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("content 1")
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "doc2.txt").write_text("content 2")

            processor = FluentProcessor.from_directory(tmpdir, recursive=True)
            assert len(processor.processor.documents) == 2

    def test_from_directory_not_recursive(self):
        """Test non-recursive directory loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("content 1")
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "doc2.txt").write_text("content 2")

            processor = FluentProcessor.from_directory(tmpdir, recursive=False)
            assert len(processor.processor.documents) == 1  # Only top-level

    def test_from_directory_missing(self):
        """Test that missing directory raises error."""
        with pytest.raises(FileNotFoundError):
            FluentProcessor.from_directory("/nonexistent/directory")

    def test_from_directory_not_a_directory(self):
        """Test that file path raises error."""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Not a directory"):
                FluentProcessor.from_directory(f.name)

    def test_from_directory_no_matches(self):
        """Test that no matching files raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No files matching pattern"):
                FluentProcessor.from_directory(tmpdir, pattern="*.xyz")


class TestFluentProcessorBuild:
    """Test build functionality."""

    def test_build_marks_as_built(self):
        """Test that build marks processor as built."""
        processor = FluentProcessor()
        processor.add_document("doc1", "neural networks process information")
        assert not processor.is_built

        processor.build(verbose=False)
        assert processor.is_built

    def test_build_with_options(self):
        """Test build with various options."""
        processor = FluentProcessor()
        processor.add_document("doc1", "neural networks and deep learning")
        processor.add_document("doc2", "machine learning algorithms")

        processor.build(
            verbose=False,
            build_concepts=True,
            cluster_strictness=0.8,
            bridge_weight=0.1
        )
        assert processor.is_built

    def test_add_document_after_build_marks_stale(self):
        """Test that adding document after build marks as not built."""
        processor = FluentProcessor()
        processor.add_document("doc1", "content")
        processor.build(verbose=False)
        assert processor.is_built

        processor.add_document("doc2", "more content")
        assert not processor.is_built


class TestFluentProcessorSearch:
    """Test search methods."""

    def test_search_basic(self):
        """Test basic search."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks process information efficiently")
            .add_document("doc2", "deep learning uses neural architectures")
            .build(verbose=False))

        results = processor.search("neural")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_with_options(self):
        """Test search with custom options."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks")
            .add_document("doc2", "machine learning")
            .build(verbose=False))

        results = processor.search("neural", top_n=1, use_expansion=False)
        assert len(results) <= 1

    def test_fast_search(self):
        """Test fast search."""
        processor = (FluentProcessor()
            .add_document("doc1", "authentication and authorization systems")
            .add_document("doc2", "database query optimization")
            .build(verbose=False))

        results = processor.fast_search("authentication", top_n=5)
        assert isinstance(results, list)

    def test_search_passages(self):
        """Test passage search."""
        processor = (FluentProcessor()
            .add_document("doc1", "Neural networks are computational models. They process information efficiently. Deep learning uses these architectures.")
            .build(verbose=False))

        results = processor.search_passages("neural networks", top_n=2)
        assert isinstance(results, list)
        if results:  # May be empty for short documents
            assert all(isinstance(r, tuple) and len(r) == 5 for r in results)
            # Verify structure: (doc_id, passage_text, start_pos, end_pos, score)
            for doc_id, passage_text, start_pos, end_pos, score in results:
                assert isinstance(doc_id, str)
                assert isinstance(passage_text, str)
                assert isinstance(start_pos, int)
                assert isinstance(end_pos, int)
                assert isinstance(score, float)

    def test_expand_query(self):
        """Test query expansion."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks and deep learning systems")
            .add_document("doc2", "machine learning algorithms and models")
            .build(verbose=False))

        expansions = processor.expand("neural", max_expansions=5)
        assert isinstance(expansions, dict)
        assert "neural" in expansions


class TestFluentProcessorPersistence:
    """Test save and load functionality."""

    def test_save_and_load(self):
        """Test saving and loading processor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "corpus_state")

            # Create and save
            (FluentProcessor()
                .add_document("doc1", "test content here")
                .build(verbose=False)
                .save(state_path))

            # Load
            loaded = FluentProcessor.load(state_path)
            assert loaded.is_built
            assert len(loaded.processor.documents) == 1
            assert "doc1" in loaded.processor.documents

            # Can search immediately
            results = loaded.search("test")
            assert isinstance(results, list)

    def test_load_marks_as_built(self):
        """Test that loading marks processor as built."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "corpus_state")
            FluentProcessor().add_document("doc1", "content").build(verbose=False).save(state_path)
            loaded = FluentProcessor.load(state_path)
            assert loaded.is_built


class TestFluentProcessorConfiguration:
    """Test configuration methods."""

    def test_with_config(self):
        """Test setting configuration."""
        config = CorticalConfig(pagerank_damping=0.9, pagerank_iterations=30)
        processor = (FluentProcessor()
            .with_config(config)
            .add_document("doc1", "test content"))

        assert processor.processor.config.pagerank_damping == 0.9
        assert processor.processor.config.pagerank_iterations == 30

    def test_with_tokenizer(self):
        """Test setting custom tokenizer."""
        tokenizer = Tokenizer(split_identifiers=True)
        processor = (FluentProcessor()
            .with_tokenizer(tokenizer)
            .add_document("doc1", "getUserCredentials"))

        assert processor.processor.tokenizer.split_identifiers


class TestFluentProcessorExamples:
    """Test example usage patterns from documentation."""

    def test_readme_example(self):
        """Test the example from README."""
        results = (FluentProcessor()
            .add_document("doc1", "Neural networks process information")
            .add_document("doc2", "Deep learning uses neural architectures")
            .build(verbose=False)
            .search("neural processing", top_n=5))

        assert isinstance(results, list)
        assert len(results) > 0

    def test_chained_operations(self):
        """Test complex chained operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "corpus_state")

            processor = (FluentProcessor()
                .add_documents({
                    "doc1": "neural networks and deep learning",
                    "doc2": "machine learning algorithms",
                    "doc3": "artificial intelligence systems"
                })
                .build(verbose=False)
                .save(state_path))

            # Search on built processor
            results = processor.search("neural", top_n=2)
            assert len(results) <= 2

            # Expand query
            expanded = processor.expand("learning")
            assert isinstance(expanded, dict)

    def test_from_files_workflow(self):
        """Test complete workflow with file loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("Neural networks are powerful")
            (Path(tmpdir) / "doc2.txt").write_text("Deep learning is effective")

            results = (FluentProcessor
                .from_directory(tmpdir)
                .build(verbose=False)
                .search("neural"))

            assert isinstance(results, list)
            assert len(results) > 0
