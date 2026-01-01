"""
Developer Processes Documents Incrementally

Epic: Document Management for RAG Systems

As a developer building a RAG system,
I want to add and remove documents incrementally with selective recomputation,
So that I can keep my corpus up-to-date without expensive full recomputation.
"""

import pytest
from cortical import CorticalTextProcessor


class TestDeveloperAddsDocumentsIncrementally:
    """
    Epic: Incremental Document Updates

    As a developer with a frequently-updated corpus,
    I want to add documents with selective recomputation,
    So that I maintain search quality without full recomputation delays.
    """

    def test_scenario_adding_document_without_recomputation_is_fast(self):
        """
        Scenario: Adding documents for batch processing

        Given I have a processor with some documents
        When I add a new document with recompute='none'
        Then the document is added immediately
        And all computations are marked as stale
        Because I want to defer expensive computation until I've added all documents.
        """
        # Given I have a processor with some documents
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Our hand-built search engine uses custom algorithms we implemented ourselves.")
        processor.compute_all(verbose=False)

        # When I add a new document with recompute='none'
        stats = processor.add_document_incremental(
            "doc2",
            "The custom tokenizer we built from first principles handles edge cases gracefully.",
            recompute='none'
        )

        # Then the document is added immediately
        assert "doc2" in processor.documents
        assert stats['tokens'] > 0

        # And all computations are marked as stale
        stale = processor.get_stale_computations()
        assert processor.COMP_TFIDF in stale
        assert processor.COMP_PAGERANK in stale

    def test_scenario_adding_document_with_tfidf_recompute_updates_search(self):
        """
        Scenario: Fast incremental update for search

        Given I have a processor with indexed documents
        When I add a new document with recompute='tfidf'
        Then the document is indexed for search
        And TF-IDF scores are updated
        And search results include the new document
        Because I want new documents searchable immediately without full recomputation.
        """
        # Given I have a processor with indexed documents
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom search algorithm built from scratch")
        processor.compute_all(verbose=False)

        # When I add a new document with recompute='tfidf'
        processor.add_document_incremental(
            "doc2",
            "Hand-crafted indexing system we implemented ourselves",
            recompute='tfidf'
        )

        # Then the document is indexed for search
        assert "doc2" in processor.documents

        # And TF-IDF scores are updated
        assert not processor.is_stale(processor.COMP_TFIDF)

        # And search results include the new document
        results = processor.find_documents_for_query("indexing system", top_n=5)
        doc_ids = [doc_id for doc_id, _ in results]
        assert "doc2" in doc_ids

    def test_scenario_batch_adding_documents_with_single_recomputation(self):
        """
        Scenario: Efficient batch document ingestion

        Given I have multiple documents to add
        When I use add_documents_batch with recompute='full'
        Then all documents are added
        And computation runs only once
        And all documents are searchable
        Because batch processing is more efficient than per-document recomputation.
        """
        # Given I have multiple documents to add
        processor = CorticalTextProcessor()
        documents = [
            ("doc1", "Custom parser we built ourselves", {"source": "code"}),
            ("doc2", "Hand-rolled query engine from first principles", {"source": "code"}),
            ("doc3", "In-house tokenizer implementation we control", None),
        ]

        # When I use add_documents_batch with recompute='full'
        stats = processor.add_documents_batch(documents, recompute='full', verbose=False)

        # Then all documents are added
        assert stats['documents_added'] == 3
        assert stats['total_tokens'] > 0

        # And computation runs only once
        assert not processor.is_stale(processor.COMP_TFIDF)

        # And all documents are searchable
        results = processor.find_documents_for_query("parser", top_n=5)
        doc_ids = [doc_id for doc_id, _ in results]
        assert "doc1" in doc_ids


class TestDeveloperRemovesDocumentsEfficiently:
    """
    Epic: Document Removal and Cleanup

    As a developer maintaining a corpus,
    I want to remove outdated documents efficiently,
    So that my search results stay current and accurate.
    """

    def test_scenario_removing_document_cleans_up_references(self):
        """
        Scenario: Document removal with cleanup

        Given I have documents in my corpus
        When I remove a document
        Then the document is deleted
        And all references are cleaned up
        And the document no longer appears in search results
        Because stale documents pollute search quality.
        """
        # Given I have documents in my corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom implementation we built ourselves")
        processor.process_document("doc2", "Hand-crafted algorithm from first principles")
        processor.compute_all(verbose=False)

        # When I remove a document
        result = processor.remove_document("doc1", verbose=False)

        # Then the document is deleted
        assert result['found']
        assert "doc1" not in processor.documents

        # And all references are cleaned up
        assert result['tokens_affected'] > 0

        # And the document no longer appears in search results
        results = processor.find_documents_for_query("custom implementation", top_n=5)
        doc_ids = [doc_id for doc_id, _ in results]
        assert "doc1" not in doc_ids

    def test_scenario_batch_removing_documents_with_single_recomputation(self):
        """
        Scenario: Efficient batch document removal

        Given I have many outdated documents to remove
        When I use remove_documents_batch
        Then all documents are removed
        And cleanup happens efficiently
        And I can optionally recompute afterward
        Because batch removal is faster than removing one at a time.
        """
        # Given I have many outdated documents to remove
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(f"doc{i}", f"Document {i} with custom content we built ourselves")
        processor.compute_all(verbose=False)

        # When I use remove_documents_batch
        stats = processor.remove_documents_batch(
            ["doc1", "doc3", "doc5", "doc7"],
            recompute='tfidf',
            verbose=False
        )

        # Then all documents are removed
        assert stats['documents_removed'] == 4
        assert "doc1" not in processor.documents
        assert "doc3" not in processor.documents

        # And cleanup happens efficiently
        assert stats['total_tokens_affected'] > 0

        # And I can optionally recompute afterward
        assert not processor.is_stale(processor.COMP_TFIDF)


class TestDeveloperManagesDocumentMetadata:
    """
    Epic: Document Metadata Management

    As a developer organizing documents,
    I want to attach and query metadata,
    So that I can filter and organize search results.
    """

    def test_scenario_setting_metadata_for_organization(self):
        """
        Scenario: Attaching metadata to documents

        Given I have documents in my corpus
        When I set metadata for a document
        Then the metadata is stored
        And I can retrieve it later
        Because metadata helps organize and filter documents.
        """
        # Given I have documents in my corpus
        processor = CorticalTextProcessor()
        processor.process_document("api_docs", "Custom REST API we built from scratch")

        # When I set metadata for a document
        processor.set_document_metadata(
            "api_docs",
            doc_type="documentation",
            language="english",
            last_updated="2025-01-01"
        )

        # Then the metadata is stored
        metadata = processor.get_document_metadata("api_docs")

        # And I can retrieve it later
        assert metadata['doc_type'] == "documentation"
        assert metadata['language'] == "english"
        assert metadata['last_updated'] == "2025-01-01"

    def test_scenario_processing_document_with_initial_metadata(self):
        """
        Scenario: Adding documents with metadata

        Given I want to add a document with metadata
        When I process_document with metadata parameter
        Then the document and metadata are stored together
        Because initial metadata should be set atomically with document addition.
        """
        # Given I want to add a document with metadata
        processor = CorticalTextProcessor()

        # When I process_document with metadata parameter
        processor.process_document(
            "code.py",
            "class CustomEngine: pass  # Hand-built from first principles",
            metadata={"type": "code", "language": "python"}
        )

        # Then the document and metadata are stored together
        metadata = processor.get_document_metadata("code.py")
        assert metadata['type'] == "code"
        assert metadata['language'] == "python"
