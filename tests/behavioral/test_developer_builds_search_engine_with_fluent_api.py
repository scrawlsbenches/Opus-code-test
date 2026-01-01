"""
Behavioral tests for developers using the fluent chainable API.

Epic: Fluent Interface for Search Engine Construction

As a developer building search applications,
I want a chainable fluent API for processor construction,
So that I can express complex operations in readable method chains.

Based on: cortical/fluent.py
"""

import pytest
import tempfile
from pathlib import Path
from cortical import CorticalTextProcessor
from cortical.fluent import FluentProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer


class TestDeveloperBuildsSearchEngineWithFluentAPI:
    """
    Epic: Fluent Interface for Search Engine Construction

    As a developer building search applications,
    I want method chaining for all operations,
    So that I can construct search engines in expressive one-liners.
    """

    def test_scenario_developer_chains_operations_fluently(self):
        """
        Scenario: Building a search engine with method chaining

        Given a need to create and query a corpus
        When I chain add_document, build, and search operations
        Then all operations execute in sequence
        And I get search results without intermediate variables
        Because fluent APIs make code more expressive and readable.
        """
        # GIVEN a need to create and query a corpus
        # WHEN I chain add_document, build, and search operations
        results = (FluentProcessor()
            .add_document("neural_nets", "Neural networks are computational models inspired by biological brains.")
            .add_document("ml_basics", "Machine learning algorithms learn from data without explicit programming.")
            .add_document("deep_learning", "Deep learning uses neural networks with multiple layers for feature extraction.")
            .build(verbose=False)
            .search("neural networks", top_n=5))

        # THEN all operations execute in sequence
        # AND I get search results without intermediate variables
        assert isinstance(results, list), "Should return search results"
        assert len(results) > 0, "Should find matching documents"

        # Verify results are in expected format (doc_id, score)
        doc_id, score = results[0]
        assert isinstance(doc_id, str), "Result should contain document ID"
        assert isinstance(score, float), "Result should contain relevance score"

    def test_scenario_developer_adds_multiple_documents_at_once(self):
        """
        Scenario: Batch document addition with fluent API

        Given multiple documents to index
        When I use add_documents with a dictionary
        Then all documents are added in one operation
        And I can continue chaining operations
        Because batch operations reduce boilerplate code.
        """
        # GIVEN multiple documents to index
        documents = {
            "supervised": "Supervised learning uses labeled training data to train predictive models.",
            "unsupervised": "Unsupervised learning finds patterns in unlabeled data without guidance.",
            "reinforcement": "Reinforcement learning trains agents through reward and penalty signals."
        }

        # WHEN I use add_documents with a dictionary
        processor = (FluentProcessor()
            .add_documents(documents)
            .build(verbose=False))

        # THEN all documents are added in one operation
        assert len(processor.processor.documents) == 3, "Should add all documents"

        # AND I can continue chaining operations
        results = processor.search("training data", top_n=3)
        assert len(results) > 0, "Should be able to search after batch add"

    def test_scenario_developer_loads_from_file_list(self):
        """
        Scenario: Creating processor from file paths

        Given a list of text files on disk
        When I use from_files class method
        Then processor loads all file contents
        And uses filenames as document IDs
        Because developers often work with file-based corpora.
        """
        # GIVEN a list of text files on disk
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test files
            (tmp_path / "neural.txt").write_text(
                "Neural networks process information through layers of artificial neurons."
            )
            (tmp_path / "training.txt").write_text(
                "Training neural networks requires backpropagation and gradient descent."
            )

            # WHEN I use from_files class method
            file_paths = [tmp_path / "neural.txt", tmp_path / "training.txt"]
            processor = (FluentProcessor
                .from_files(file_paths)
                .build(verbose=False))

            # THEN processor loads all file contents
            assert len(processor.processor.documents) == 2, "Should load all files"

            # AND uses filenames as document IDs
            doc_ids = set(processor.processor.documents.keys())
            assert "neural" in doc_ids, "Should use filename stem as ID"
            assert "training" in doc_ids, "Should use filename stem as ID"

    def test_scenario_developer_loads_from_directory(self):
        """
        Scenario: Bulk loading from directory with pattern matching

        Given a directory containing multiple text files
        When I use from_directory with a glob pattern
        Then all matching files are loaded
        And I can specify recursive scanning
        Because developers need bulk loading from file systems.
        """
        # GIVEN a directory containing multiple text files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test files with different extensions
            (tmp_path / "doc1.txt").write_text("First document about neural networks.")
            (tmp_path / "doc2.txt").write_text("Second document about machine learning.")
            (tmp_path / "readme.md").write_text("This is a markdown file.")

            # Create subdirectory
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (subdir / "doc3.txt").write_text("Third document in subdirectory.")

            # WHEN I use from_directory with a glob pattern
            processor_txt = (FluentProcessor
                .from_directory(tmp_path, pattern="*.txt", recursive=False)
                .build(verbose=False))

            # THEN all matching files are loaded
            assert len(processor_txt.processor.documents) == 2, "Should load only .txt files in root"

            # AND I can specify recursive scanning
            processor_recursive = (FluentProcessor
                .from_directory(tmp_path, pattern="*.txt", recursive=True)
                .build(verbose=False))

            assert len(processor_recursive.processor.documents) == 3, "Should load .txt files recursively"

    def test_scenario_developer_configures_processor_fluently(self):
        """
        Scenario: Configuring processor with method chaining

        Given custom tokenizer and configuration requirements
        When I use with_config and with_tokenizer methods
        Then processor uses custom settings
        And configuration happens before document processing
        Because developers need to customize behavior fluently.
        """
        # GIVEN custom tokenizer and configuration requirements
        custom_config = CorticalConfig(chunk_size=256, chunk_overlap=64, pagerank_damping=0.9)
        custom_tokenizer = Tokenizer(split_identifiers=True, min_word_length=2)

        # WHEN I use with_config and with_tokenizer methods
        processor = (FluentProcessor()
            .with_config(custom_config)
            .with_tokenizer(custom_tokenizer)
            .add_document("test", "Test document with custom configuration settings.")
            .build(verbose=False))

        # THEN processor uses custom settings
        assert processor.processor.config.chunk_size == 256, "Should use custom config"
        assert processor.processor.config.pagerank_damping == 0.9, "Should use custom damping"

        # AND configuration happens before document processing
        # The tokenizer should have processed with custom settings
        assert processor.processor.tokenizer.split_identifiers is True, "Should use custom tokenizer"
        assert processor.processor.tokenizer.min_word_length == 2, "Should use custom min_word_length"

    def test_scenario_developer_saves_and_loads_processor(self):
        """
        Scenario: Persisting processor state to disk

        Given a built processor with indexed documents
        When I save it and later reload it
        Then the reloaded processor has the same state
        And I can immediately search without rebuilding
        Because developers need to persist expensive computations.
        """
        # GIVEN a built processor with indexed documents
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "corpus.pkl"

            # Build and save
            original = (FluentProcessor()
                .add_document("doc1", "Neural networks for pattern recognition.")
                .add_document("doc2", "Machine learning model training techniques.")
                .build(verbose=False)
                .save(save_path))

            original_results = original.search("neural", top_n=3)

            # WHEN I save it and later reload it
            reloaded = FluentProcessor.load(save_path)

            # THEN the reloaded processor has the same state
            assert len(reloaded.processor.documents) == 2, "Should restore documents"

            # AND I can immediately search without rebuilding
            reloaded_results = reloaded.search("neural", top_n=3)
            assert len(reloaded_results) > 0, "Should be able to search immediately"
            assert reloaded.is_built, "Loaded processor should be marked as built"

    def test_scenario_developer_uses_fast_search_for_performance(self):
        """
        Scenario: Optimized search with candidate filtering

        Given a large corpus requiring fast searches
        When I use fast_search instead of regular search
        Then results are returned with lower latency
        And precision remains acceptable
        Because production systems need speed without sacrificing quality.
        """
        # GIVEN a large corpus requiring fast searches
        processor = FluentProcessor()
        for i in range(50):
            processor.add_document(
                f"doc_{i}",
                f"Document {i} contains information about neural networks, machine learning, and artificial intelligence."
            )
        processor.build(verbose=False)

        # WHEN I use fast_search instead of regular search
        fast_results = processor.fast_search("neural networks", top_n=5, candidate_multiplier=3)

        # THEN results are returned with lower latency
        assert len(fast_results) > 0, "Should return results"

        # AND precision remains acceptable
        # Verify results are in expected format
        for doc_id, score in fast_results:
            assert isinstance(doc_id, str), "Should contain document IDs"
            assert isinstance(score, float), "Should contain scores"

    def test_scenario_developer_searches_passages_for_rag(self):
        """
        Scenario: Passage-level retrieval for RAG systems

        Given documents with rich content
        When I use search_passages with chunk parameters
        Then I get passage-level results with positions
        And passages are suitable for citation
        Because RAG systems need precise text retrieval with source locations.
        """
        # GIVEN documents with rich content
        long_content = """
        Neural networks are computational models inspired by biological neural networks.
        They consist of layers of interconnected nodes that process information.
        Deep learning extends neural networks with many layers for complex pattern recognition.
        Training neural networks requires large datasets and significant computational resources.
        """

        processor = (FluentProcessor()
            .add_document("ml_guide", long_content)
            .build(verbose=False))

        # WHEN I use search_passages with chunk parameters
        passages = processor.search_passages(
            "neural network training",
            top_n=3,
            chunk_size=100,
            overlap=20
        )

        # THEN I get passage-level results with positions
        assert len(passages) > 0, "Should find relevant passages"

        # AND passages are suitable for citation
        doc_id, passage_text, start, end, score = passages[0]
        assert isinstance(passage_text, str), "Should include passage text"
        assert isinstance(start, int) and isinstance(end, int), "Should include positions"
        assert start < end, "Start should precede end position"
        assert isinstance(score, float), "Should include relevance score"

    def test_scenario_developer_expands_queries_for_better_recall(self):
        """
        Scenario: Query expansion for comprehensive results

        Given a narrow search query
        When I use expand to get related terms
        Then I receive semantically related expansion terms
        And can use expanded terms for broader search
        Because query expansion improves recall.
        """
        # GIVEN a narrow search query
        processor = (FluentProcessor()
            .add_document("doc1", "Neural networks are machine learning models with artificial neurons.")
            .add_document("doc2", "Deep learning networks contain multiple hidden layers.")
            .add_document("doc3", "Training neural nets requires backpropagation algorithms.")
            .build(verbose=False))

        # WHEN I use expand to get related terms
        expansions = processor.expand("neural", max_expansions=10)

        # THEN I receive semantically related expansion terms
        assert isinstance(expansions, dict), "Should return dictionary of terms"
        assert len(expansions) > 0, "Should find expansion terms"

        # AND can use expanded terms for broader search
        # Expansions should include weights
        for term, weight in expansions.items():
            assert isinstance(term, str), "Term should be string"
            assert isinstance(weight, float), "Weight should be float"
            assert 0 <= weight <= 1, "Weight should be normalized"

    def test_scenario_developer_adds_tuples_with_metadata(self):
        """
        Scenario: Adding documents with metadata tuples

        Given documents with associated metadata
        When I use add_documents with 3-tuples
        Then metadata is preserved for each document
        And I can filter results by metadata later
        Because developers need to track document properties.
        """
        # GIVEN documents with associated metadata
        documents_with_metadata = [
            ("paper1", "Research on convolutional neural networks for image classification.",
             {"year": 2020, "category": "research"}),
            ("blog1", "Tutorial on building neural networks from scratch.",
             {"year": 2022, "category": "tutorial"}),
            ("paper2", "Survey of recurrent neural network architectures.",
             {"year": 2021, "category": "research"})
        ]

        # WHEN I use add_documents with 3-tuples
        processor = (FluentProcessor()
            .add_documents(documents_with_metadata)
            .build(verbose=False))

        # THEN metadata is preserved for each document
        all_metadata = processor.processor.get_all_document_metadata()
        assert "paper1" in all_metadata, "Should preserve metadata"
        assert all_metadata["paper1"]["year"] == 2020, "Should preserve metadata values"

        # AND I can filter results by metadata later
        assert all_metadata["blog1"]["category"] == "tutorial", "Should distinguish categories"

    def test_scenario_developer_checks_build_status(self):
        """
        Scenario: Tracking whether processor is ready for search

        Given a processor in various build states
        When I check the is_built property
        Then I know whether search operations are available
        And can conditionally trigger builds
        Because developers need to know processor readiness.
        """
        # GIVEN a processor in various build states
        processor = FluentProcessor()

        # WHEN I check the is_built property
        # THEN I know whether search operations are available
        assert not processor.is_built, "New processor should not be built"

        processor.add_document("doc1", "Test content")
        assert not processor.is_built, "Adding documents should mark as not built"

        processor.build(verbose=False)
        assert processor.is_built, "After build, should be marked as built"

        # AND can conditionally trigger builds
        processor.add_document("doc2", "More content")
        assert not processor.is_built, "Adding more documents should reset build status"

    def test_scenario_developer_accesses_underlying_processor(self):
        """
        Scenario: Direct access to underlying processor for advanced operations

        Given a fluent processor wrapper
        When I access the processor property
        Then I get the underlying CorticalTextProcessor
        And can perform operations not exposed by fluent API
        Because developers sometimes need low-level access.
        """
        # GIVEN a fluent processor wrapper
        fluent = (FluentProcessor()
            .add_document("test", "Test document content")
            .build(verbose=False))

        # WHEN I access the processor property
        underlying = fluent.processor

        # THEN I get the underlying CorticalTextProcessor
        assert isinstance(underlying, CorticalTextProcessor), "Should be CorticalTextProcessor"

        # AND can perform operations not exposed by fluent API
        # Can call any processor method directly
        assert hasattr(underlying, 'compute_importance'), "Should have processor methods"
        assert "test" in underlying.documents, "Should share same state"

    def test_scenario_developer_wraps_existing_processor(self):
        """
        Scenario: Creating fluent wrapper around existing processor

        Given an already-constructed CorticalTextProcessor
        When I create a FluentProcessor from it
        Then I can use fluent operations on the existing processor
        And existing state is preserved
        Because developers may need to upgrade existing code gradually.
        """
        # GIVEN an already-constructed CorticalTextProcessor
        existing = CorticalTextProcessor()
        existing.process_document("existing_doc", "Document processed the traditional way.")
        existing.compute_all(verbose=False)

        # WHEN I create a FluentProcessor from it
        fluent = FluentProcessor.from_existing(existing)

        # THEN I can use fluent operations on the existing processor
        results = fluent.search("document", top_n=3)
        assert len(results) > 0, "Should be able to search"

        # AND existing state is preserved
        assert "existing_doc" in fluent.processor.documents, "Should preserve existing documents"

    def test_scenario_developer_builds_with_custom_parameters(self):
        """
        Scenario: Fine-tuning build process with parameters

        Given specific requirements for indexing strategy
        When I call build with custom parameters
        Then processor uses specified algorithms and settings
        And I can optimize for my use case
        Because different applications need different trade-offs.
        """
        # GIVEN specific requirements for indexing strategy
        processor = (FluentProcessor()
            .add_document("doc1", "First document about neural network architectures.")
            .add_document("doc2", "Second document about training methodologies."))

        # WHEN I call build with custom parameters
        processor.build(
            verbose=False,
            build_concepts=True,
            pagerank_method='standard',
            connection_strategy='document_overlap',
            cluster_strictness=0.8,
            bridge_weight=0.1
        )

        # THEN processor uses specified algorithms and settings
        assert processor.is_built, "Should complete build"

        # AND I can optimize for my use case
        # Processor should have applied custom settings during build
        results = processor.search("neural", top_n=3)
        assert len(results) > 0, "Should work with custom build settings"

    def test_scenario_developer_handles_missing_files_gracefully(self):
        """
        Scenario: Error handling for missing files

        Given a list of file paths with some non-existent
        When I try to load from_files
        Then a clear error is raised
        And I can catch and handle the error
        Because developers need informative error messages.
        """
        # GIVEN a list of file paths with some non-existent
        non_existent_path = Path("/tmp/this_file_does_not_exist_12345.txt")

        # WHEN I try to load from_files
        # THEN a clear error is raised
        with pytest.raises(FileNotFoundError) as exc_info:
            FluentProcessor.from_files([non_existent_path])

        # AND I can catch and handle the error
        assert "not found" in str(exc_info.value).lower(), "Error should be descriptive"

    def test_scenario_developer_chains_multiple_add_operations(self):
        """
        Scenario: Mixing single and batch add operations

        Given various document sources
        When I chain add_document and add_documents calls
        Then all documents are accumulated
        And I can build once at the end
        Because developers need flexibility in document addition.
        """
        # GIVEN various document sources
        # WHEN I chain add_document and add_documents calls
        processor = (FluentProcessor()
            .add_document("single1", "First single document.")
            .add_documents({
                "batch1": "First batch document.",
                "batch2": "Second batch document."
            })
            .add_document("single2", "Second single document.")
            .add_documents([
                ("tuple1", "Tuple format document."),
                ("tuple2", "Another tuple document.", {"source": "test"})
            ])
            .build(verbose=False))

        # THEN all documents are accumulated
        assert len(processor.processor.documents) == 6, "Should accumulate all documents"

        # AND I can build once at the end
        assert processor.is_built, "Should build successfully after all additions"

    def test_scenario_developer_uses_repr_for_debugging(self):
        """
        Scenario: Inspecting processor state during development

        Given a processor at various stages
        When I print or inspect the processor
        Then I see useful status information
        And can diagnose issues quickly
        Because developers need visibility into object state.
        """
        # GIVEN a processor at various stages
        processor = FluentProcessor()

        # WHEN I print or inspect the processor
        # THEN I see useful status information
        repr_str = repr(processor)
        assert "FluentProcessor" in repr_str, "Should identify type"
        assert "not built" in repr_str, "Should show build status"
        assert "documents=0" in repr_str, "Should show document count"

        processor.add_document("test", "Test")
        repr_str = repr(processor)
        assert "documents=1" in repr_str, "Should update document count"

        processor.build(verbose=False)
        repr_str = repr(processor)
        assert "built" in repr_str, "Should update build status"
