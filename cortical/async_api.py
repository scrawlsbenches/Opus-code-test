"""
Async API for batch operations in CorticalTextProcessor.

This module provides async wrappers for CPU-intensive batch operations,
allowing non-blocking document processing and concurrent search queries.

Example:
    >>> from cortical import CorticalTextProcessor
    >>> from cortical.async_api import AsyncProcessor
    >>> import asyncio
    >>>
    >>> processor = CorticalTextProcessor()
    >>> async_proc = AsyncProcessor(processor, max_workers=4)
    >>>
    >>> # Add documents asynchronously with progress callback
    >>> async def add_docs():
    ...     documents = [
    ...         ("doc1", "First document content", None),
    ...         ("doc2", "Second document content", {"source": "web"}),
    ...     ]
    ...     result = await async_proc.add_documents_async(
    ...         documents,
    ...         progress_callback=lambda done, total: print(f"{done}/{total}"),
    ...         chunk_size=10
    ...     )
    ...     print(f"Added {result['documents_added']} documents")
    >>>
    >>> # Run concurrent searches
    >>> async def search_multiple():
    ...     results = await async_proc.batch_search_async(
    ...         ["neural networks", "machine learning", "deep learning"],
    ...         top_n=5,
    ...         concurrency=3
    ...     )
    ...     return results
    >>>
    >>> # Run the async operations
    >>> asyncio.run(add_docs())
    >>> results = asyncio.run(search_multiple())
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Callable, Any
from threading import Event

logger = logging.getLogger(__name__)


class AsyncProcessor:
    """
    Async wrapper for CorticalTextProcessor batch operations.

    Provides non-blocking access to CPU-intensive operations by running them
    in a thread pool executor. Supports progress callbacks, cancellation,
    and concurrent query execution.

    Attributes:
        processor: The underlying CorticalTextProcessor instance
        max_workers: Maximum number of worker threads (default: 4)
    """

    def __init__(self, processor, max_workers: int = 4):
        """
        Initialize async processor wrapper.

        Args:
            processor: CorticalTextProcessor instance to wrap
            max_workers: Maximum number of worker threads (default: 4)

        Raises:
            ValueError: If max_workers < 1
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self.processor = processor
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cancel_event = Event()
        self._running_tasks = set()

    async def add_documents_async(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 10,
        recompute: str = 'full'
    ) -> Dict[str, Any]:
        """
        Add documents asynchronously in batches.

        Processes documents in chunks, yielding control to the event loop between
        chunks to avoid blocking. Optionally reports progress via callback.

        Args:
            documents: List of (doc_id, content, metadata) tuples
            progress_callback: Optional function(done, total) called after each chunk
            chunk_size: Number of documents to process per chunk (default: 10)
            recompute: Recomputation level ('none', 'tfidf', 'full')

        Returns:
            Dict with statistics:
                - documents_added: Number of documents added
                - total_tokens: Total tokens processed
                - chunks_processed: Number of chunks processed
                - recomputation: Type of recomputation performed

        Raises:
            ValueError: If documents list is invalid or chunk_size < 1
            asyncio.CancelledError: If operation is cancelled

        Example:
            >>> async def add_with_progress():
            ...     docs = [("doc1", "content1", None), ("doc2", "content2", None)]
            ...     result = await async_proc.add_documents_async(
            ...         docs,
            ...         progress_callback=lambda done, total: print(f"{done}/{total}"),
            ...         chunk_size=5
            ...     )
            ...     return result
        """
        if not documents:
            raise ValueError("documents list must not be empty")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        loop = asyncio.get_event_loop()
        total = len(documents)
        processed = 0
        chunks_count = 0

        total_tokens = 0
        total_bigrams = 0

        # Process documents in chunks without recomputation
        for i in range(0, total, chunk_size):
            if self._cancel_event.is_set():
                raise asyncio.CancelledError("Operation cancelled")

            chunk = documents[i:i + chunk_size]

            # Run chunk processing in executor
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.processor.add_documents_batch(
                    chunk,
                    recompute='none',  # Defer recomputation until the end
                    verbose=False
                )
            )

            total_tokens += result.get('total_tokens', 0)
            total_bigrams += result.get('total_bigrams', 0)
            processed += len(chunk)
            chunks_count += 1

            if progress_callback:
                progress_callback(processed, total)

            # Yield control to event loop
            await asyncio.sleep(0)

        # Perform single recomputation at the end if requested
        if recompute != 'none':
            await loop.run_in_executor(
                self._executor,
                lambda: self.processor.recompute(level=recompute, verbose=False)
            )

        return {
            'documents_added': total,
            'total_tokens': total_tokens,
            'total_bigrams': total_bigrams,
            'chunks_processed': chunks_count,
            'recomputation': recompute
        }

    async def compute_all_async(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run compute_all() in executor thread.

        Args:
            progress_callback: Optional function(phase_name) called before each phase
            **kwargs: Arguments to pass to compute_all()

        Returns:
            Dict with computation statistics

        Raises:
            asyncio.CancelledError: If operation is cancelled

        Example:
            >>> async def compute():
            ...     result = await async_proc.compute_all_async(
            ...         progress_callback=lambda phase: print(f"Running: {phase}")
            ...     )
            ...     return result
        """
        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Operation cancelled")

        loop = asyncio.get_event_loop()

        # Note: progress_callback is for phase reporting, not compute_all's progress
        # compute_all has its own show_progress parameter for internal progress
        if progress_callback:
            progress_callback("Starting compute_all")

        result = await loop.run_in_executor(
            self._executor,
            lambda: self.processor.compute_all(**kwargs)
        )

        if progress_callback:
            progress_callback("compute_all completed")

        return result

    async def search_async(
        self,
        query: str,
        top_n: int = 5,
        use_expansion: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search documents asynchronously.

        Args:
            query: Search query string
            top_n: Number of results to return
            use_expansion: Whether to use query expansion

        Returns:
            List of (doc_id, score) tuples

        Raises:
            ValueError: If query is empty or top_n < 1
            asyncio.CancelledError: If operation is cancelled

        Example:
            >>> async def search():
            ...     results = await async_proc.search_async("neural networks", top_n=10)
            ...     return results
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Operation cancelled")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self.processor.find_documents_for_query(
                query,
                top_n=top_n,
                use_expansion=use_expansion
            )
        )
        return result

    async def batch_search_async(
        self,
        queries: List[str],
        top_n: int = 5,
        concurrency: int = 4
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Run multiple searches concurrently.

        Executes multiple search queries in parallel, up to the specified
        concurrency limit. Returns a mapping of query to results.

        Args:
            queries: List of search query strings
            top_n: Number of results per query
            concurrency: Maximum concurrent searches (default: 4)

        Returns:
            Dict mapping query to list of (doc_id, score) tuples

        Raises:
            ValueError: If queries is empty or concurrency < 1
            asyncio.CancelledError: If operation is cancelled

        Example:
            >>> async def search_many():
            ...     queries = ["neural networks", "machine learning", "deep learning"]
            ...     results = await async_proc.batch_search_async(
            ...         queries,
            ...         top_n=5,
            ...         concurrency=3
            ...     )
            ...     for query, matches in results.items():
            ...         print(f"{query}: {len(matches)} results")
        """
        if not queries:
            raise ValueError("queries list must not be empty")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Operation cancelled")

        # Create semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)

        async def search_with_semaphore(query: str):
            async with sem:
                return await self.search_async(query, top_n=top_n)

        # Create tasks for all queries
        tasks = [search_with_semaphore(query) for query in queries]

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # Map queries to results
        return {query: result for query, result in zip(queries, results)}

    async def remove_documents_async(
        self,
        doc_ids: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 10,
        recompute: str = 'none'
    ) -> Dict[str, Any]:
        """
        Remove documents asynchronously in batches.

        Args:
            doc_ids: List of document IDs to remove
            progress_callback: Optional function(done, total) called after each chunk
            chunk_size: Number of documents to process per chunk (default: 10)
            recompute: Recomputation level ('none', 'tfidf', 'full')

        Returns:
            Dict with statistics:
                - documents_removed: Number of documents actually removed
                - documents_not_found: Number of doc_ids that didn't exist
                - chunks_processed: Number of chunks processed
                - recomputation: Type of recomputation performed

        Raises:
            ValueError: If doc_ids is empty or chunk_size < 1
            asyncio.CancelledError: If operation is cancelled

        Example:
            >>> async def remove_old_docs():
            ...     result = await async_proc.remove_documents_async(
            ...         ["old1", "old2", "old3"],
            ...         progress_callback=lambda done, total: print(f"{done}/{total}")
            ...     )
            ...     return result
        """
        if not doc_ids:
            raise ValueError("doc_ids list must not be empty")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        loop = asyncio.get_event_loop()
        total = len(doc_ids)
        processed = 0
        chunks_count = 0

        total_removed = 0
        total_not_found = 0

        # Process removals in chunks without recomputation
        for i in range(0, total, chunk_size):
            if self._cancel_event.is_set():
                raise asyncio.CancelledError("Operation cancelled")

            chunk = doc_ids[i:i + chunk_size]

            # Run chunk removal in executor
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.processor.remove_documents_batch(
                    chunk,
                    recompute='none',  # Defer recomputation until the end
                    verbose=False
                )
            )

            total_removed += result.get('documents_removed', 0)
            total_not_found += result.get('documents_not_found', 0)
            processed += len(chunk)
            chunks_count += 1

            if progress_callback:
                progress_callback(processed, total)

            # Yield control to event loop
            await asyncio.sleep(0)

        # Perform single recomputation at the end if requested
        if recompute != 'none':
            await loop.run_in_executor(
                self._executor,
                lambda: self.processor.recompute(level=recompute, verbose=False)
            )

        return {
            'documents_removed': total_removed,
            'documents_not_found': total_not_found,
            'chunks_processed': chunks_count,
            'recomputation': recompute
        }

    async def batch_passages_async(
        self,
        queries: List[str],
        top_n: int = 5,
        concurrency: int = 4,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> Dict[str, List[Tuple[str, str, int, int, float]]]:
        """
        Find passages for multiple queries concurrently.

        Args:
            queries: List of search query strings
            top_n: Number of passages per query
            concurrency: Maximum concurrent searches
            chunk_size: Size of each passage chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            Dict mapping query to list of (passage_text, doc_id, start, end, score) tuples

        Raises:
            ValueError: If queries is empty or parameters are invalid
            asyncio.CancelledError: If operation is cancelled

        Example:
            >>> async def find_passages():
            ...     results = await async_proc.batch_passages_async(
            ...         ["what is PageRank", "how does BM25 work"],
            ...         top_n=3,
            ...         concurrency=2
            ...     )
            ...     return results
        """
        if not queries:
            raise ValueError("queries list must not be empty")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Operation cancelled")

        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(concurrency)

        async def find_passages_with_semaphore(query: str):
            async with sem:
                return await loop.run_in_executor(
                    self._executor,
                    lambda: self.processor.find_passages_for_query(
                        query,
                        top_n=top_n,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                )

        # Create tasks for all queries
        tasks = [find_passages_with_semaphore(query) for query in queries]

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # Map queries to results
        return {query: result for query, result in zip(queries, results)}

    def cancel(self):
        """
        Cancel running operations.

        Sets a cancellation flag that will cause async operations to raise
        asyncio.CancelledError. Already running executor tasks will complete.

        Example:
            >>> async def process_with_timeout():
            ...     try:
            ...         await asyncio.wait_for(
            ...             async_proc.add_documents_async(large_doc_list),
            ...             timeout=10.0
            ...         )
            ...     except asyncio.TimeoutError:
            ...         async_proc.cancel()
            ...         raise
        """
        self._cancel_event.set()
        logger.info("Async operations cancelled")

    def reset_cancel(self):
        """
        Reset the cancellation flag.

        Allows new operations to run after a cancellation.

        Example:
            >>> async_proc.cancel()
            >>> # ... handle cancellation ...
            >>> async_proc.reset_cancel()
            >>> # Can now run new operations
        """
        self._cancel_event.clear()

    async def close(self):
        """
        Clean up resources and shutdown executor.

        Waits for all running tasks to complete, then shuts down the thread pool.
        Should be called when done with the async processor.

        Example:
            >>> async def main():
            ...     async_proc = AsyncProcessor(processor)
            ...     try:
            ...         # ... use async_proc ...
            ...         pass
            ...     finally:
            ...         await async_proc.close()
        """
        # Cancel any pending operations
        self.cancel()

        # Wait a bit for tasks to notice cancellation
        await asyncio.sleep(0.1)

        # Shutdown executor (wait for running tasks)
        self._executor.shutdown(wait=True)
        logger.info("Async processor closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False
