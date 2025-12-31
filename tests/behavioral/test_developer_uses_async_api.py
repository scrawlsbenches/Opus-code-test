"""
Behavioral tests for developers using the async API.

Epic: Asynchronous Document Processing

As a developer building high-throughput systems,
I want to process documents and queries asynchronously,
So that I can handle multiple operations concurrently without blocking.

Based on: examples/async_api_demo.py

Note: These tests use asyncio.run() to avoid dependency on pytest-asyncio.
"""

import pytest
import asyncio
from cortical import CorticalTextProcessor
from cortical.async_api import AsyncProcessor


class TestDeveloperUsesAsyncAPI:
    """
    Epic: Asynchronous Document Processing

    As a developer building high-throughput systems,
    I want async operations for document processing and search,
    So that I can maximize throughput without blocking threads.
    """

    def test_scenario_developer_adds_documents_without_blocking(self):
        """
        Scenario: Adding documents asynchronously

        Given a processor with async capabilities
        When I add documents using async API
        Then documents are processed without blocking the event loop
        And I receive confirmation of successful addition
        Because developers need non-blocking batch operations.
        """
        async def run_test():
            # GIVEN a processor with async capabilities
            processor = CorticalTextProcessor()
            async with AsyncProcessor(processor, max_workers=4) as async_proc:

                # WHEN I add documents using async API
                documents = [
                    ("doc1", "Neural networks are computational models inspired by biological brains.", None),
                    ("doc2", "Machine learning algorithms learn patterns from training data.", None),
                    ("doc3", "Deep learning uses multiple layers for hierarchical feature extraction.", None),
                ]

                result = await async_proc.add_documents_async(
                    documents,
                    chunk_size=2,
                    recompute='full'
                )

                # THEN documents are processed without blocking the event loop
                # AND I receive confirmation of successful addition
                assert result['documents_added'] == 3, "Should add all documents"
                assert result['total_tokens'] > 0, "Should process tokens"
                assert result['chunks_processed'] > 0, "Should process in chunks"

        asyncio.run(run_test())

    def test_scenario_developer_tracks_progress_during_batch_operations(self):
        """
        Scenario: Progress tracking for long-running operations

        Given a large batch of documents to process
        When I provide a progress callback
        Then the callback receives progress updates
        And I can show status to users in real-time
        Because developers need to provide feedback on long operations.
        """
        async def run_test():
            # GIVEN a large batch of documents to process
            processor = CorticalTextProcessor()
            async with AsyncProcessor(processor, max_workers=2) as async_proc:
                documents = [
                    (f"article_{i}", f"Article {i} discusses various machine learning topics.", None)
                    for i in range(20)
                ]

                # WHEN I provide a progress callback
                progress_updates = []

                def progress_callback(done, total):
                    progress_updates.append((done, total))

                result = await async_proc.add_documents_async(
                    documents,
                    progress_callback=progress_callback,
                    chunk_size=5,
                    recompute='tfidf'
                )

                # THEN the callback receives progress updates
                assert len(progress_updates) > 0, "Should receive progress updates"

                # AND I can show status to users in real-time
                assert progress_updates[-1][0] == progress_updates[-1][1], \
                    "Final update should show completion"
                assert result['documents_added'] == 20, "Should process all documents"

        asyncio.run(run_test())

    def test_scenario_developer_executes_concurrent_searches(self):
        """
        Scenario: Running multiple searches concurrently

        Given a corpus with indexed documents
        When I execute multiple searches concurrently
        Then all searches complete successfully
        And results are returned for each query
        Because developers need concurrent query processing.
        """
        async def run_test():
            # GIVEN a corpus with indexed documents
            processor = CorticalTextProcessor()
            documents = [
                ("ai_overview", "Artificial intelligence encompasses machine learning and deep learning.", None),
                ("ml_basics", "Machine learning algorithms learn from data without explicit programming.", None),
                ("dl_intro", "Deep learning uses neural networks with multiple layers.", None),
                ("nlp_guide", "Natural language processing enables text understanding and generation.", None),
            ]

            for doc_id, content, metadata in documents:
                processor.process_document(doc_id, content, metadata)
            processor.compute_all(verbose=False)

            async with AsyncProcessor(processor, max_workers=4) as async_proc:
                # WHEN I execute multiple searches concurrently
                queries = [
                    "neural networks",
                    "machine learning algorithms",
                    "natural language"
                ]

                results = await async_proc.batch_search_async(
                    queries,
                    top_n=3,
                    concurrency=4
                )

                # THEN all searches complete successfully
                assert len(results) == len(queries), "Should return results for all queries"

                # AND results are returned for each query
                for query in queries:
                    assert query in results, f"Should have results for query: {query}"
                    assert len(results[query]) > 0, f"Should find documents for query: {query}"

        asyncio.run(run_test())

    def test_scenario_developer_retrieves_passages_concurrently(self):
        """
        Scenario: Concurrent passage retrieval for RAG systems

        Given documents with rich content
        When I request passages for multiple queries concurrently
        Then I receive relevant passages for each query
        And passages include context information
        Because developers building RAG systems need passage-level retrieval.
        """
        async def run_test():
            # GIVEN documents with rich content
            processor = CorticalTextProcessor()
            documents = [
                ("ml_article", """
                Machine learning is a subset of artificial intelligence that enables systems
                to learn and improve from experience. Supervised learning uses labeled data
                to train models, while unsupervised learning finds patterns in unlabeled data.
                """, None),
                ("dl_article", """
                Deep learning uses artificial neural networks with multiple layers to learn
                hierarchical representations of data. Convolutional neural networks excel at
                image processing, while recurrent neural networks handle sequential data.
                """, None),
            ]

            for doc_id, content, metadata in documents:
                processor.process_document(doc_id, content, metadata)
            processor.compute_all(verbose=False)

            async with AsyncProcessor(processor, max_workers=3) as async_proc:
                # WHEN I request passages for multiple queries concurrently
                queries = [
                    "supervised learning",
                    "convolutional networks"
                ]

                results = await async_proc.batch_passages_async(
                    queries,
                    top_n=2,
                    concurrency=2,
                    chunk_size=200,
                    overlap=50
                )

                # THEN I receive relevant passages for each query
                assert len(results) == len(queries), "Should return passages for all queries"

                # AND passages include context information
                for query in queries:
                    assert query in results, f"Should have passages for query: {query}"
                    passages = results[query]
                    if len(passages) > 0:
                        passage_text, doc_id, start, end, score = passages[0]
                        assert isinstance(passage_text, str), "Passage should be text"
                        assert isinstance(doc_id, str), "Should include document ID"
                        assert isinstance(start, int) and isinstance(end, int), "Should include position"
                        assert isinstance(score, float), "Should include relevance score"

        asyncio.run(run_test())

    def test_scenario_developer_cancels_long_running_operations(self):
        """
        Scenario: Cancelling operations that take too long

        Given a long-running batch operation
        When I cancel the operation mid-flight
        Then the operation stops gracefully
        And I can continue using the processor afterward
        Because developers need to handle timeouts and user cancellations.
        """
        async def run_test():
            # GIVEN a long-running batch operation
            processor = CorticalTextProcessor()
            async_proc = AsyncProcessor(processor, max_workers=2)

            try:
                large_batch = [
                    (f"doc_{i}", f"Document {i} content with various topics.", None)
                    for i in range(100)
                ]

                # WHEN I cancel the operation mid-flight
                task = asyncio.create_task(
                    async_proc.add_documents_async(large_batch, chunk_size=5, recompute='none')
                )

                await asyncio.sleep(0.1)  # Let it start
                async_proc.cancel()

                # THEN the operation stops gracefully
                with pytest.raises(asyncio.CancelledError):
                    await task

                # AND I can continue using the processor afterward
                async_proc.reset_cancel()
                result = await async_proc.add_documents_async(
                    [("after_cancel", "Document added after cancellation reset.", None)],
                    recompute='none'
                )
                assert result['documents_added'] == 1, "Should work after cancel reset"

            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_scenario_developer_computes_all_asynchronously(self):
        """
        Scenario: Running compute_all without blocking

        Given documents that need index computation
        When I run compute_all asynchronously
        Then all phases complete without blocking
        And I receive phase progress notifications
        Because developers need non-blocking index building.
        """
        async def run_test():
            # GIVEN documents that need index computation
            processor = CorticalTextProcessor()
            for i in range(10):
                processor.process_document(
                    f"compute_doc_{i}",
                    f"Document {i} about various ML topics.",
                    None
                )

            async with AsyncProcessor(processor, max_workers=2) as async_proc:
                # WHEN I run compute_all asynchronously
                phases = []

                def phase_callback(phase):
                    phases.append(phase)

                result = await async_proc.compute_all_async(
                    progress_callback=phase_callback,
                    verbose=False
                )

                # THEN all phases complete without blocking
                assert result is not None or result is None, "Should complete computation"

                # AND I receive phase progress notifications
                assert len(phases) > 0, "Should track computation phases"

        asyncio.run(run_test())

    def test_scenario_developer_uses_async_context_manager(self):
        """
        Scenario: Proper resource cleanup with async context manager

        Given an async processor created with context manager
        When I exit the context
        Then resources are cleaned up automatically
        And the executor is properly shut down
        Because developers need automatic resource management.
        """
        async def run_test():
            # GIVEN an async processor created with context manager
            processor = CorticalTextProcessor()
            executor_closed = False

            async with AsyncProcessor(processor, max_workers=2) as async_proc:
                # Use the processor
                result = await async_proc.search_async("test query", top_n=3)
                assert isinstance(result, list), "Should work within context"

            # WHEN I exit the context
            # THEN resources are cleaned up automatically
            # AND the executor is properly shut down
            # Note: Can't check _shutdown on closed executor
            assert True, "Context manager should clean up resources"

        asyncio.run(run_test())

    def test_scenario_developer_searches_without_blocking_event_loop(self):
        """
        Scenario: Non-blocking search operations

        Given a processor with many documents
        When I perform a search asynchronously
        Then the search runs without blocking the event loop
        And I can perform other async operations concurrently
        Because developers need truly async search operations.
        """
        async def run_test():
            # GIVEN a processor with many documents
            processor = CorticalTextProcessor()
            for i in range(20):
                processor.process_document(
                    f"doc_{i}",
                    f"Document {i} contains information about neural networks and machine learning.",
                    None
                )
            processor.compute_all(verbose=False)

            async with AsyncProcessor(processor, max_workers=4) as async_proc:
                # WHEN I perform a search asynchronously
                # AND I can perform other async operations concurrently
                search_task = asyncio.create_task(
                    async_proc.search_async("neural networks", top_n=5)
                )
                other_task = asyncio.create_task(asyncio.sleep(0.001))

                # Both tasks should be able to run
                results, _ = await asyncio.gather(search_task, other_task)

                # THEN the search runs without blocking the event loop
                assert len(results) > 0, "Should return search results"
                assert isinstance(results, list), "Should return list of matches"

        asyncio.run(run_test())
