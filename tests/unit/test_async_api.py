"""
Tests for async API (cortical/async_api.py).

These tests verify the async wrapper functionality for batch operations,
including progress callbacks, cancellation, concurrent execution, and error handling.
"""

import unittest
import asyncio
from cortical import CorticalTextProcessor
from cortical.async_api import AsyncProcessor


class TestAsyncProcessor(unittest.TestCase):
    """Tests for AsyncProcessor async wrapper."""

    def setUp(self):
        """Set up test processor."""
        self.processor = CorticalTextProcessor()
        # Add some initial documents for search tests
        self.processor.process_document("doc1", "Neural networks process information efficiently.")
        self.processor.process_document("doc2", "Machine learning models require training data.")
        self.processor.process_document("doc3", "Deep learning uses multiple layers for feature extraction.")
        self.processor.compute_all(verbose=False)

    def tearDown(self):
        """Clean up after tests."""
        # Nothing to tear down for now
        pass

    def test_initialization(self):
        """Test AsyncProcessor initialization."""
        async_proc = AsyncProcessor(self.processor, max_workers=4)
        self.assertIsNotNone(async_proc)
        self.assertEqual(async_proc.processor, self.processor)

    def test_initialization_invalid_workers(self):
        """Test AsyncProcessor initialization with invalid worker count."""
        with self.assertRaises(ValueError) as ctx:
            AsyncProcessor(self.processor, max_workers=0)
        self.assertIn("max_workers must be at least 1", str(ctx.exception))

    def test_add_documents_async(self):
        """Test adding documents asynchronously."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=2)
            try:
                documents = [
                    ("async_doc1", "First async document content.", None),
                    ("async_doc2", "Second async document content.", {"source": "test"}),
                    ("async_doc3", "Third async document content.", None),
                ]

                result = await async_proc.add_documents_async(
                    documents,
                    chunk_size=2,
                    recompute='tfidf'
                )

                self.assertEqual(result['documents_added'], 3)
                self.assertGreater(result['total_tokens'], 0)
                self.assertEqual(result['chunks_processed'], 2)  # 3 docs / chunk_size 2 = 2 chunks
                self.assertEqual(result['recomputation'], 'tfidf')

                # Verify documents were added
                self.assertIn("async_doc1", self.processor.documents)
                self.assertIn("async_doc2", self.processor.documents)
                self.assertIn("async_doc3", self.processor.documents)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_add_documents_async_with_progress(self):
        """Test adding documents with progress callback."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=2)
            try:
                progress_calls = []

                def progress_callback(done, total):
                    progress_calls.append((done, total))

                documents = [
                    (f"progress_doc{i}", f"Document {i} content.", None)
                    for i in range(5)
                ]

                result = await async_proc.add_documents_async(
                    documents,
                    progress_callback=progress_callback,
                    chunk_size=2,
                    recompute='none'
                )

                # Should have progress calls for each chunk
                self.assertGreater(len(progress_calls), 0)
                # Last call should be (5, 5)
                self.assertEqual(progress_calls[-1], (5, 5))
                self.assertEqual(result['documents_added'], 5)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_add_documents_async_validation(self):
        """Test validation in add_documents_async."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor)
            try:
                # Empty documents list
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.add_documents_async([])
                self.assertIn("must not be empty", str(ctx.exception))

                # Invalid chunk size
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.add_documents_async(
                        [("doc", "content", None)],
                        chunk_size=0
                    )
                self.assertIn("chunk_size must be at least 1", str(ctx.exception))
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_search_async(self):
        """Test async search."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=2)
            try:
                results = await async_proc.search_async("neural networks", top_n=3)

                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 0)
                # Should be tuples of (doc_id, score)
                for doc_id, score in results:
                    self.assertIsInstance(doc_id, str)
                    self.assertIsInstance(score, (int, float))
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_search_async_validation(self):
        """Test validation in search_async."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor)
            try:
                # Empty query
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.search_async("")
                self.assertIn("must be a non-empty string", str(ctx.exception))

                # Invalid top_n
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.search_async("query", top_n=0)
                self.assertIn("top_n must be at least 1", str(ctx.exception))
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_batch_search_async(self):
        """Test concurrent batch search."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=4)
            try:
                queries = [
                    "neural networks",
                    "machine learning",
                    "deep learning"
                ]

                results = await async_proc.batch_search_async(
                    queries,
                    top_n=2,
                    concurrency=3
                )

                # Should have results for all queries
                self.assertEqual(len(results), 3)
                for query in queries:
                    self.assertIn(query, results)
                    self.assertIsInstance(results[query], list)
                    # Each query should have results
                    self.assertGreater(len(results[query]), 0)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_batch_search_async_validation(self):
        """Test validation in batch_search_async."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor)
            try:
                # Empty queries list
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.batch_search_async([])
                self.assertIn("must not be empty", str(ctx.exception))

                # Invalid concurrency
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.batch_search_async(["query"], concurrency=0)
                self.assertIn("concurrency must be at least 1", str(ctx.exception))
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_compute_all_async(self):
        """Test async compute_all."""
        async def run_test():
            # Create fresh processor for compute_all test
            processor = CorticalTextProcessor()
            processor.process_document("test1", "Test document one.")
            processor.process_document("test2", "Test document two.")

            async_proc = AsyncProcessor(processor, max_workers=2)
            try:
                phase_calls = []

                def progress_callback(phase):
                    phase_calls.append(phase)

                result = await async_proc.compute_all_async(
                    progress_callback=progress_callback,
                    verbose=False
                )

                # Should have completion stats
                self.assertIsInstance(result, dict)
                # Progress callback should have been called
                self.assertGreater(len(phase_calls), 0)
                self.assertIn("Starting compute_all", phase_calls)
                self.assertIn("compute_all completed", phase_calls)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_remove_documents_async(self):
        """Test async document removal."""
        async def run_test():
            # Add documents to remove
            self.processor.process_document("remove1", "Document to remove 1.")
            self.processor.process_document("remove2", "Document to remove 2.")
            self.processor.process_document("remove3", "Document to remove 3.")

            async_proc = AsyncProcessor(self.processor, max_workers=2)
            try:
                doc_ids = ["remove1", "remove2", "remove3", "nonexistent"]

                result = await async_proc.remove_documents_async(
                    doc_ids,
                    chunk_size=2,
                    recompute='none'
                )

                self.assertEqual(result['documents_removed'], 3)
                self.assertEqual(result['documents_not_found'], 1)
                self.assertEqual(result['chunks_processed'], 2)  # 4 docs / chunk_size 2 = 2 chunks

                # Verify documents were removed
                self.assertNotIn("remove1", self.processor.documents)
                self.assertNotIn("remove2", self.processor.documents)
                self.assertNotIn("remove3", self.processor.documents)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_remove_documents_async_with_progress(self):
        """Test async removal with progress callback."""
        async def run_test():
            # Add documents to remove
            for i in range(5):
                self.processor.process_document(f"remove_progress{i}", f"Document {i}.")

            async_proc = AsyncProcessor(self.processor, max_workers=2)
            try:
                progress_calls = []

                def progress_callback(done, total):
                    progress_calls.append((done, total))

                doc_ids = [f"remove_progress{i}" for i in range(5)]

                result = await async_proc.remove_documents_async(
                    doc_ids,
                    progress_callback=progress_callback,
                    chunk_size=2
                )

                # Should have progress calls
                self.assertGreater(len(progress_calls), 0)
                # Last call should be (5, 5)
                self.assertEqual(progress_calls[-1], (5, 5))
                self.assertEqual(result['documents_removed'], 5)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_batch_passages_async(self):
        """Test concurrent passage retrieval."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=3)
            try:
                queries = ["neural networks", "machine learning"]

                results = await async_proc.batch_passages_async(
                    queries,
                    top_n=2,
                    concurrency=2,
                    chunk_size=100,
                    overlap=20
                )

                # Should have results for all queries
                self.assertEqual(len(results), 2)
                for query in queries:
                    self.assertIn(query, results)
                    self.assertIsInstance(results[query], list)
                    # Each result is (passage_text, doc_id, start, end, score)
                    if len(results[query]) > 0:
                        passage = results[query][0]
                        self.assertEqual(len(passage), 5)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_batch_passages_async_validation(self):
        """Test validation in batch_passages_async."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor)
            try:
                # Empty queries
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.batch_passages_async([])
                self.assertIn("must not be empty", str(ctx.exception))

                # Invalid concurrency
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.batch_passages_async(["query"], concurrency=0)
                self.assertIn("concurrency must be at least 1", str(ctx.exception))

                # Invalid top_n
                with self.assertRaises(ValueError) as ctx:
                    await async_proc.batch_passages_async(["query"], top_n=0)
                self.assertIn("top_n must be at least 1", str(ctx.exception))
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_cancellation(self):
        """Test operation cancellation."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=2)
            try:
                # Cancel before operation
                async_proc.cancel()

                # Operation should raise CancelledError
                with self.assertRaises(asyncio.CancelledError):
                    await async_proc.add_documents_async(
                        [("doc", "content", None)],
                        chunk_size=1
                    )

                # Reset and verify we can run again
                async_proc.reset_cancel()

                result = await async_proc.add_documents_async(
                    [("doc_after_reset", "content after reset", None)],
                    chunk_size=1,
                    recompute='none'
                )
                self.assertEqual(result['documents_added'], 1)
            finally:
                await async_proc.close()

        asyncio.run(run_test())

    def test_async_context_manager(self):
        """Test async context manager usage."""
        async def run_test():
            async with AsyncProcessor(self.processor, max_workers=2) as async_proc:
                results = await async_proc.search_async("neural networks", top_n=2)
                self.assertGreater(len(results), 0)

            # Processor should be closed after context exit
            # (we can't easily verify executor shutdown, but it should have happened)

        asyncio.run(run_test())

    def test_close(self):
        """Test explicit close."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=2)

            # Use the processor
            results = await async_proc.search_async("test", top_n=1)
            self.assertIsInstance(results, list)

            # Close it
            await async_proc.close()

            # After close, operations should be cancelled
            with self.assertRaises(asyncio.CancelledError):
                await async_proc.search_async("test", top_n=1)

        asyncio.run(run_test())

    def test_concurrent_operations(self):
        """Test running multiple async operations concurrently."""
        async def run_test():
            async_proc = AsyncProcessor(self.processor, max_workers=4)
            try:
                # Run multiple operations concurrently
                search_task = async_proc.search_async("neural networks", top_n=2)
                batch_task = async_proc.batch_search_async(
                    ["machine learning", "deep learning"],
                    top_n=2,
                    concurrency=2
                )

                # Wait for both
                search_results, batch_results = await asyncio.gather(search_task, batch_task)

                # Verify both completed successfully
                self.assertGreater(len(search_results), 0)
                self.assertEqual(len(batch_results), 2)
            finally:
                await async_proc.close()

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
