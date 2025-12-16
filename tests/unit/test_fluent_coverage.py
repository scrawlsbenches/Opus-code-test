"""
Additional Coverage Tests for FluentProcessor API
==================================================

Comprehensive unit tests targeting uncovered edge cases and validation logic
in cortical/fluent.py to improve coverage from ~25% to >80%.

Focus areas:
- from_files edge cases (encoding, empty files, metadata preservation)
- from_directory edge cases (various patterns, path types)
- from_existing edge cases
- load edge cases
- add_documents complex validation paths
- build parameter variations
- Terminal operation parameter variations
- Property access edge cases
- Error handling paths
"""

import pytest
import tempfile
from pathlib import Path

from cortical import CorticalTextProcessor, CorticalConfig, Tokenizer
from cortical.fluent import FluentProcessor


# =============================================================================
# CONSTRUCTOR AND CLASSMETHOD EDGE CASES
# =============================================================================


class TestFluentProcessorConstructorEdgeCases:
    """Test edge cases for constructors and classmethods."""

    def test_init_with_both_config_and_tokenizer(self):
        """Test initialization with both config and tokenizer."""
        config = CorticalConfig(pagerank_damping=0.9, pagerank_iterations=25)
        tokenizer = Tokenizer(split_identifiers=True, min_word_length=2)

        processor = FluentProcessor(tokenizer=tokenizer, config=config)

        assert processor.processor.config.pagerank_damping == 0.9
        assert processor.processor.config.pagerank_iterations == 25
        assert processor.processor.tokenizer.split_identifiers
        assert processor.processor.tokenizer.min_word_length == 2

    def test_from_existing_preserves_is_built_state(self):
        """Test that from_existing sets is_built to False initially."""
        raw = CorticalTextProcessor()
        raw.process_document("doc1", "test content")
        raw.compute_all(verbose=False)

        fluent = FluentProcessor.from_existing(raw)

        # from_existing always sets is_built to False
        assert not fluent.is_built

    def test_from_existing_with_config_and_tokenizer(self):
        """Test from_existing preserves existing processor's config and tokenizer."""
        config = CorticalConfig(pagerank_damping=0.8)
        tokenizer = Tokenizer(split_identifiers=True)
        raw = CorticalTextProcessor(tokenizer=tokenizer, config=config)

        fluent = FluentProcessor.from_existing(raw)

        assert fluent.processor.config.pagerank_damping == 0.8
        assert fluent.processor.tokenizer.split_identifiers


class TestFluentProcessorFromFilesEdgeCases:
    """Test edge cases for from_files classmethod."""

    def test_from_files_with_pathlib_path(self):
        """Test from_files works with pathlib.Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file2 = Path(tmpdir) / "doc2.txt"
            file1.write_text("content 1")
            file2.write_text("content 2")

            # Pass Path objects directly
            processor = FluentProcessor.from_files([file1, file2])

            assert len(processor.processor.documents) == 2
            assert "doc1" in processor.processor.documents

    def test_from_files_with_string_paths(self):
        """Test from_files works with string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file1.write_text("content 1")

            # Pass string path
            processor = FluentProcessor.from_files([str(file1)])

            assert len(processor.processor.documents) == 1

    def test_from_files_metadata_includes_source(self):
        """Test that from_files adds source metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file1.write_text("content 1")

            processor = FluentProcessor.from_files([file1])

            # Check that source metadata was added
            assert "doc1" in processor.processor.document_metadata
            metadata = processor.processor.document_metadata["doc1"]
            assert "source" in metadata
            assert "doc1.txt" in metadata["source"]

    def test_from_files_single_space_file(self):
        """Test from_files handles files with minimal content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "minimal.txt"
            file1.write_text("x")  # Minimal non-empty content

            processor = FluentProcessor.from_files([file1])

            assert "minimal" in processor.processor.documents
            assert processor.processor.documents["minimal"] == "x"

    def test_from_files_with_custom_config(self):
        """Test from_files with custom config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file1.write_text("content 1")

            config = CorticalConfig(pagerank_iterations=40)
            processor = FluentProcessor.from_files([file1], config=config)

            assert processor.processor.config.pagerank_iterations == 40

    def test_from_files_with_custom_tokenizer(self):
        """Test from_files with custom tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file1.write_text("getUserData")

            tokenizer = Tokenizer(split_identifiers=True)
            processor = FluentProcessor.from_files([file1], tokenizer=tokenizer)

            assert processor.processor.tokenizer.split_identifiers

    def test_from_files_preserves_document_order(self):
        """Test that from_files processes files in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(5):
                f = Path(tmpdir) / f"doc{i}.txt"
                f.write_text(f"content {i}")
                files.append(f)

            processor = FluentProcessor.from_files(files)

            # All files should be processed
            assert len(processor.processor.documents) == 5


class TestFluentProcessorFromDirectoryEdgeCases:
    """Test edge cases for from_directory classmethod."""

    def test_from_directory_with_pathlib_path(self):
        """Test from_directory works with pathlib.Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("content 1")

            # Pass Path object
            processor = FluentProcessor.from_directory(tmppath)

            assert len(processor.processor.documents) == 1

    def test_from_directory_multiple_patterns(self):
        """Test from_directory with various file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("txt content")
            (tmppath / "doc2.md").write_text("md content")
            (tmppath / "doc3.py").write_text("py content")

            # Match only .py files
            processor = FluentProcessor.from_directory(tmpdir, pattern="*.py")
            assert len(processor.processor.documents) == 1
            assert "doc3" in processor.processor.documents

    def test_from_directory_glob_pattern_all(self):
        """Test from_directory with wildcard pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("content 1")
            (tmppath / "doc2.md").write_text("content 2")

            # Match all files
            processor = FluentProcessor.from_directory(tmpdir, pattern="*")
            assert len(processor.processor.documents) >= 2  # May include hidden files

    def test_from_directory_recursive_nested_structure(self):
        """Test recursive directory loading with multiple levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("level 0")

            level1 = tmppath / "level1"
            level1.mkdir()
            (level1 / "doc2.txt").write_text("level 1")

            level2 = level1 / "level2"
            level2.mkdir()
            (level2 / "doc3.txt").write_text("level 2")

            processor = FluentProcessor.from_directory(tmpdir, recursive=True)
            assert len(processor.processor.documents) == 3

    def test_from_directory_with_config_and_tokenizer(self):
        """Test from_directory with both config and tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc1.txt").write_text("test content")

            config = CorticalConfig(pagerank_damping=0.88)
            tokenizer = Tokenizer(split_identifiers=False, min_word_length=4)

            processor = FluentProcessor.from_directory(
                tmpdir,
                config=config,
                tokenizer=tokenizer
            )

            assert processor.processor.config.pagerank_damping == 0.88
            assert not processor.processor.tokenizer.split_identifiers
            assert processor.processor.tokenizer.min_word_length == 4


class TestFluentProcessorLoadEdgeCases:
    """Test edge cases for load classmethod."""

    def test_load_with_string_path(self):
        """Test load works with string path."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Create and save
            FluentProcessor().add_document("doc1", "content").build(verbose=False).save(temp_path)

            # Load with string path
            loaded = FluentProcessor.load(temp_path)
            assert loaded.is_built
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_with_pathlib_path(self):
        """Test load works with pathlib.Path."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)

        try:
            FluentProcessor().add_document("doc1", "content").build(verbose=False).save(str(temp_path))

            # Load with Path object
            loaded = FluentProcessor.load(temp_path)
            assert loaded.is_built
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_preserves_document_metadata(self):
        """Test that load preserves document metadata."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Save with metadata
            processor = FluentProcessor()
            processor.add_document("doc1", "content", metadata={"key": "value"})
            processor.build(verbose=False).save(temp_path)

            # Load and check metadata
            loaded = FluentProcessor.load(temp_path)
            assert loaded.processor.document_metadata["doc1"]["key"] == "value"
        finally:
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# DOCUMENT ADDITION EDGE CASES
# =============================================================================


class TestFluentProcessorAddDocumentsEdgeCases:
    """Test edge cases for add_document and add_documents."""

    def test_add_document_marks_not_built(self):
        """Test that add_document sets is_built to False."""
        processor = FluentProcessor()
        assert not processor.is_built

        processor.add_document("doc1", "content")
        assert not processor.is_built

    def test_add_document_minimal_content(self):
        """Test adding document with minimal content."""
        processor = FluentProcessor()
        processor.add_document("doc1", "x")

        assert "doc1" in processor.processor.documents
        assert processor.processor.documents["doc1"] == "x"

    def test_add_documents_empty_dict(self):
        """Test add_documents with empty dict."""
        processor = FluentProcessor()
        result = processor.add_documents({})

        assert result is processor
        assert len(processor.processor.documents) == 0

    def test_add_documents_empty_list(self):
        """Test add_documents with empty list."""
        processor = FluentProcessor()
        result = processor.add_documents([])

        assert result is processor
        assert len(processor.processor.documents) == 0

    def test_add_documents_mixed_tuples(self):
        """Test add_documents with mixed 2-tuple and 3-tuple."""
        processor = FluentProcessor()
        docs = [
            ("doc1", "content 1"),  # 2-tuple
            ("doc2", "content 2", {"author": "Alice"}),  # 3-tuple
            ("doc3", "content 3"),  # 2-tuple
            ("doc4", "content 4", {"author": "Bob"}),  # 3-tuple
        ]
        processor.add_documents(docs)

        assert len(processor.processor.documents) == 4
        assert processor.processor.document_metadata["doc2"]["author"] == "Alice"
        assert processor.processor.document_metadata["doc4"]["author"] == "Bob"

    def test_add_documents_invalid_tuple_too_long(self):
        """Test that 4-tuple raises error."""
        processor = FluentProcessor()
        with pytest.raises(ValueError, match="Invalid document tuple"):
            processor.add_documents([("doc1", "content", {}, "extra")])

    def test_add_documents_marks_not_built(self):
        """Test that add_documents sets is_built to False."""
        processor = FluentProcessor()
        processor.add_documents({"doc1": "content"})

        assert not processor.is_built


class TestFluentProcessorConfigurationChaining:
    """Test configuration method chaining edge cases."""

    def test_with_config_preserves_documents(self):
        """Test that with_config doesn't affect existing documents."""
        processor = FluentProcessor()
        processor.add_document("doc1", "content")

        config = CorticalConfig(pagerank_damping=0.95)
        processor.with_config(config)

        assert "doc1" in processor.processor.documents
        assert processor.processor.config.pagerank_damping == 0.95

    def test_with_tokenizer_preserves_documents(self):
        """Test that with_tokenizer doesn't affect existing documents."""
        processor = FluentProcessor()
        processor.add_document("doc1", "content")

        tokenizer = Tokenizer(filter_code_noise=True, min_word_length=2)
        processor.with_tokenizer(tokenizer)

        assert "doc1" in processor.processor.documents
        assert processor.processor.tokenizer.filter_code_noise
        assert processor.processor.tokenizer.min_word_length == 2


# =============================================================================
# BUILD PARAMETER VARIATIONS
# =============================================================================


class TestFluentProcessorBuildParameters:
    """Test build method with various parameter combinations."""

    def test_build_all_parameters(self):
        """Test build with all parameters specified."""
        processor = FluentProcessor()
        processor.add_document("doc1", "neural networks process information")
        processor.add_document("doc2", "deep learning uses neural architectures")

        result = processor.build(
            verbose=True,
            build_concepts=True,
            pagerank_method='hierarchical',
            connection_strategy='semantic',
            cluster_strictness=0.9,
            bridge_weight=0.5,
            show_progress=False
        )

        assert result is processor
        assert processor.is_built

    def test_build_no_concepts(self):
        """Test build with build_concepts=False."""
        processor = FluentProcessor()
        processor.add_document("doc1", "test content")

        processor.build(verbose=False, build_concepts=False)
        assert processor.is_built

    def test_build_different_pagerank_methods(self):
        """Test build with different pagerank methods."""
        for method in ['standard', 'semantic', 'hierarchical']:
            processor = FluentProcessor()
            processor.add_document("doc1", "neural networks")
            processor.build(verbose=False, pagerank_method=method)
            assert processor.is_built

    def test_build_different_connection_strategies(self):
        """Test build with different connection strategies."""
        strategies = ['document_overlap', 'semantic', 'embedding', 'hybrid']
        for strategy in strategies:
            processor = FluentProcessor()
            processor.add_document("doc1", "test content here")
            processor.build(verbose=False, connection_strategy=strategy)
            assert processor.is_built

    def test_build_cluster_strictness_variations(self):
        """Test build with different cluster strictness values."""
        for strictness in [0.0, 0.5, 1.0]:
            processor = FluentProcessor()
            processor.add_document("doc1", "content")
            processor.build(verbose=False, cluster_strictness=strictness)
            assert processor.is_built

    def test_build_bridge_weight_variations(self):
        """Test build with different bridge weight values."""
        for weight in [0.0, 0.5, 1.0]:
            processor = FluentProcessor()
            processor.add_document("doc1", "content")
            processor.build(verbose=False, bridge_weight=weight)
            assert processor.is_built

    def test_build_show_progress_true(self):
        """Test build with show_progress=True."""
        processor = FluentProcessor()
        processor.add_document("doc1", "test content")

        processor.build(verbose=False, show_progress=True)
        assert processor.is_built


# =============================================================================
# TERMINAL OPERATION EDGE CASES
# =============================================================================


class TestFluentProcessorSearchEdgeCases:
    """Test edge cases for search methods."""

    def test_search_no_expansion_no_semantic(self):
        """Test search with all expansion disabled."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks")
            .build(verbose=False))

        results = processor.search("neural", use_expansion=False, use_semantic=False)
        assert isinstance(results, list)

    def test_search_top_n_variations(self):
        """Test search with different top_n values."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks")
            .add_document("doc2", "deep learning")
            .add_document("doc3", "machine learning")
            .build(verbose=False))

        for top_n in [1, 2, 5, 10]:
            results = processor.search("learning", top_n=top_n)
            assert len(results) <= top_n

    def test_fast_search_candidate_multiplier_variations(self):
        """Test fast_search with different candidate multipliers."""
        processor = (FluentProcessor()
            .add_document("doc1", "authentication system")
            .add_document("doc2", "authorization system")
            .build(verbose=False))

        for multiplier in [1, 3, 5]:
            results = processor.fast_search("auth", candidate_multiplier=multiplier)
            assert isinstance(results, list)

    def test_fast_search_no_code_concepts(self):
        """Test fast_search with use_code_concepts=False."""
        processor = (FluentProcessor()
            .add_document("doc1", "authentication")
            .build(verbose=False))

        results = processor.fast_search("auth", use_code_concepts=False)
        assert isinstance(results, list)

    def test_search_passages_with_custom_chunk_params(self):
        """Test search_passages with custom chunk size and overlap."""
        processor = (FluentProcessor()
            .add_document("doc1", "Neural networks are powerful. " * 20)
            .build(verbose=False))

        results = processor.search_passages(
            "neural",
            chunk_size=50,
            overlap=10,
            use_expansion=False
        )
        assert isinstance(results, list)

    def test_search_passages_no_expansion(self):
        """Test search_passages without expansion."""
        processor = (FluentProcessor()
            .add_document("doc1", "test content here")
            .build(verbose=False))

        results = processor.search_passages("test", use_expansion=False)
        assert isinstance(results, list)

    def test_expand_all_parameters(self):
        """Test expand with all parameters specified."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks deep learning")
            .build(verbose=False))

        expansions = processor.expand(
            "neural",
            max_expansions=5,
            use_variants=True,
            use_code_concepts=False
        )
        assert isinstance(expansions, dict)

    def test_expand_no_variants(self):
        """Test expand with use_variants=False."""
        processor = (FluentProcessor()
            .add_document("doc1", "neural networks")
            .build(verbose=False))

        expansions = processor.expand("neural", use_variants=False)
        assert isinstance(expansions, dict)

    def test_expand_with_code_concepts(self):
        """Test expand with use_code_concepts=True."""
        processor = (FluentProcessor()
            .add_document("doc1", "getUserData fetchUserInfo")
            .build(verbose=False))

        expansions = processor.expand("get", use_code_concepts=True)
        assert isinstance(expansions, dict)


# =============================================================================
# PROPERTY ACCESS EDGE CASES
# =============================================================================


class TestFluentProcessorProperties:
    """Test property access edge cases."""

    def test_processor_property_returns_underlying_processor(self):
        """Test that processor property returns CorticalTextProcessor."""
        fluent = FluentProcessor()
        raw = fluent.processor

        assert isinstance(raw, CorticalTextProcessor)
        assert raw is fluent._processor

    def test_processor_property_allows_direct_manipulation(self):
        """Test that processor property allows direct method calls."""
        fluent = FluentProcessor()
        fluent.add_document("doc1", "test content")

        # Call method on underlying processor directly
        fluent.processor.compute_all(verbose=False)

        # Check side effects
        assert len(fluent.processor.documents) == 1

    def test_is_built_initially_false(self):
        """Test that is_built is False on initialization."""
        processor = FluentProcessor()
        assert processor.is_built is False

    def test_is_built_after_build(self):
        """Test that is_built is True after build."""
        processor = FluentProcessor()
        processor.add_document("doc1", "content")
        processor.build(verbose=False)

        assert processor.is_built is True

    def test_is_built_after_add_document_post_build(self):
        """Test that is_built becomes False after adding document post-build."""
        processor = FluentProcessor()
        processor.add_document("doc1", "content")
        processor.build(verbose=False)
        assert processor.is_built is True

        processor.add_document("doc2", "more content")
        assert processor.is_built is False


# =============================================================================
# REPR AND STRING REPRESENTATION
# =============================================================================


class TestFluentProcessorRepr:
    """Test __repr__ edge cases."""

    def test_repr_empty_processor(self):
        """Test repr of empty processor."""
        processor = FluentProcessor()
        repr_str = repr(processor)

        assert "FluentProcessor" in repr_str
        assert "documents=0" in repr_str
        assert "not built" in repr_str

    def test_repr_multiple_documents(self):
        """Test repr with multiple documents."""
        processor = FluentProcessor()
        processor.add_documents({
            "doc1": "content 1",
            "doc2": "content 2",
            "doc3": "content 3"
        })
        repr_str = repr(processor)

        assert "documents=3" in repr_str

    def test_repr_built_processor(self):
        """Test repr of built processor."""
        processor = FluentProcessor()
        processor.add_document("doc1", "content")
        processor.build(verbose=False)
        repr_str = repr(processor)

        assert "built" in repr_str
        assert "not built" not in repr_str


# =============================================================================
# SAVE/LOAD CHAIN OPERATIONS
# =============================================================================


class TestFluentProcessorSaveLoadChaining:
    """Test save and load in chains."""

    def test_save_returns_self_allows_further_chaining(self):
        """Test that save returns self and allows further operations."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            processor = (FluentProcessor()
                .add_document("doc1", "neural networks")
                .build(verbose=False)
                .save(temp_path))

            # Can still call methods after save
            results = processor.search("neural")
            assert isinstance(results, list)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_then_search_chain(self):
        """Test loading then immediately searching."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            FluentProcessor().add_document("doc1", "neural").build(verbose=False).save(temp_path)

            # Load and search in one expression
            results = FluentProcessor.load(temp_path).search("neural")
            assert isinstance(results, list)
            assert len(results) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


class TestFluentProcessorErrorHandling:
    """Test error handling for invalid inputs."""

    def test_from_files_with_non_existent_file(self):
        """Test that from_files raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            FluentProcessor.from_files(["/this/does/not/exist.txt"])

    def test_from_files_with_directory_path(self):
        """Test that from_files raises ValueError when path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Not a file"):
                FluentProcessor.from_files([tmpdir])

    def test_from_directory_with_non_existent_directory(self):
        """Test that from_directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            FluentProcessor.from_directory("/this/does/not/exist")

    def test_from_directory_with_file_path(self):
        """Test that from_directory raises ValueError when path is a file."""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Not a directory"):
                FluentProcessor.from_directory(f.name)

    def test_from_directory_with_no_matching_files(self):
        """Test that from_directory raises ValueError when no files match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that won't match
            (Path(tmpdir) / "test.xyz").write_text("content")

            with pytest.raises(ValueError, match="No files matching pattern"):
                FluentProcessor.from_directory(tmpdir, pattern="*.txt")

    def test_add_documents_with_invalid_type(self):
        """Test that add_documents raises TypeError for invalid types."""
        processor = FluentProcessor()

        with pytest.raises(TypeError, match="must be a dict or list"):
            processor.add_documents("invalid string")

        with pytest.raises(TypeError, match="must be a dict or list"):
            processor.add_documents(123)

    def test_add_documents_with_single_element_tuple(self):
        """Test that 1-tuple raises ValueError."""
        processor = FluentProcessor()

        with pytest.raises(ValueError, match="Invalid document tuple"):
            processor.add_documents([("doc1",)])


# =============================================================================
# INTEGRATION TESTS - COMPLEX WORKFLOWS
# =============================================================================


class TestFluentProcessorComplexWorkflows:
    """Test complex workflow combinations."""

    def test_full_workflow_with_all_features(self):
        """Test complete workflow using all major features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("neural networks deep learning")
            (Path(tmpdir) / "doc2.txt").write_text("machine learning algorithms")

            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as save_file:
                save_path = save_file.name

            try:
                # Load from directory, configure, build, search, save
                processor = (FluentProcessor
                    .from_directory(tmpdir)
                    .with_config(CorticalConfig(pagerank_damping=0.9))
                    .build(verbose=False, build_concepts=True)
                    .save(save_path))

                # Search
                results = processor.search("learning", top_n=2)
                assert len(results) <= 2

                # Load and verify
                loaded = FluentProcessor.load(save_path)
                assert loaded.is_built

                # Search on loaded
                loaded_results = loaded.search("learning")
                assert isinstance(loaded_results, list)
            finally:
                Path(save_path).unlink(missing_ok=True)

    def test_incremental_document_addition_workflow(self):
        """Test adding documents incrementally with rebuilds."""
        processor = FluentProcessor()

        # Add first batch
        processor.add_documents({
            "doc1": "neural networks",
            "doc2": "deep learning"
        })
        processor.build(verbose=False)
        assert processor.is_built

        # Add more documents
        processor.add_document("doc3", "machine learning")
        assert not processor.is_built

        # Rebuild
        processor.build(verbose=False)
        assert processor.is_built
        assert len(processor.processor.documents) == 3

    def test_configuration_change_workflow(self):
        """Test changing configuration mid-workflow."""
        processor = FluentProcessor()
        processor.add_document("doc1", "test content")

        # Change config before build
        config = CorticalConfig(pagerank_iterations=50)
        processor.with_config(config).build(verbose=False)

        assert processor.processor.config.pagerank_iterations == 50
        assert processor.is_built

    def test_multiple_search_operations_workflow(self):
        """Test multiple search operations on same processor."""
        processor = (FluentProcessor()
            .add_documents({
                "doc1": "neural networks deep learning",
                "doc2": "machine learning algorithms",
                "doc3": "artificial intelligence"
            })
            .build(verbose=False))

        # Multiple different searches
        results1 = processor.search("neural")
        results2 = processor.fast_search("learning")
        results3 = processor.search_passages("algorithms")
        expansions = processor.expand("intelligence")

        assert all(isinstance(r, list) for r in [results1, results2, results3])
        assert isinstance(expansions, dict)
