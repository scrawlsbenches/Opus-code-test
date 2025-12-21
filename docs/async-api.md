# Async API for Batch Operations

The `AsyncProcessor` class provides async wrappers for CPU-intensive batch operations in the Cortical Text Processor, enabling non-blocking document processing and concurrent search queries.

## Overview

The async API solves several challenges when working with large document corpora:

1. **Non-blocking processing**: Add/remove thousands of documents without blocking the event loop
2. **Concurrent searches**: Execute multiple search queries in parallel
3. **Progress tracking**: Monitor long-running operations with callbacks
4. **Cancellation support**: Gracefully cancel operations when needed
5. **Resource management**: Clean shutdown with async context managers

## Installation

The async API is included in the core package:

```python
from cortical import CorticalTextProcessor, AsyncProcessor
```

## Basic Usage

### Creating an Async Processor

```python
import asyncio
from cortical import CorticalTextProcessor, AsyncProcessor

# Create standard processor
processor = CorticalTextProcessor()

# Wrap with async API
async_proc = AsyncProcessor(processor, max_workers=4)
```

### Async Context Manager (Recommended)

```python
async def process_documents():
    processor = CorticalTextProcessor()

    async with AsyncProcessor(processor, max_workers=4) as async_proc:
        # Use async_proc here
        results = await async_proc.search_async("query")
        return results
    # Automatically cleaned up after context exit
```

## Core Operations

### 1. Add Documents Asynchronously

Add multiple documents in batches without blocking:

```python
async def add_docs():
    documents = [
        ("doc1", "First document content.", None),
        ("doc2", "Second document content.", {"source": "web"}),
        ("doc3", "Third document content.", None),
    ]

    result = await async_proc.add_documents_async(
        documents,
        chunk_size=10,        # Process 10 docs at a time
        recompute='full'      # Recompute all after adding
    )

    print(f"Added {result['documents_added']} documents")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Chunks: {result['chunks_processed']}")
```

**Parameters:**
- `documents`: List of `(doc_id, content, metadata)` tuples
- `chunk_size`: Documents per chunk (default: 10)
- `recompute`: Recomputation level (`'none'`, `'tfidf'`, `'full'`)
- `progress_callback`: Optional `function(done, total)` for progress tracking

**Returns:**
- `documents_added`: Number of documents added
- `total_tokens`: Total tokens processed
- `chunks_processed`: Number of chunks processed
- `recomputation`: Type of recomputation performed

### 2. Progress Callbacks

Track progress of long-running operations:

```python
async def add_with_progress():
    def progress(done, total):
        percent = (done / total) * 100
        print(f"Progress: {done}/{total} ({percent:.1f}%)")

    documents = [(f"doc{i}", f"Content {i}", None) for i in range(100)]

    result = await async_proc.add_documents_async(
        documents,
        progress_callback=progress,
        chunk_size=10
    )
```

### 3. Async Search

Search documents asynchronously:

```python
async def search():
    results = await async_proc.search_async(
        "neural networks",
        top_n=5,
        use_expansion=True
    )

    for doc_id, score in results:
        print(f"{doc_id}: {score:.4f}")
```

### 4. Batch Search (Concurrent)

Run multiple searches concurrently:

```python
async def batch_search():
    queries = [
        "neural networks",
        "machine learning",
        "deep learning"
    ]

    # Run up to 4 searches concurrently
    results = await async_proc.batch_search_async(
        queries,
        top_n=5,
        concurrency=4
    )

    # Results is a dict mapping query -> [(doc_id, score), ...]
    for query, matches in results.items():
        print(f"\n{query}:")
        for doc_id, score in matches:
            print(f"  {doc_id}: {score:.4f}")
```

**Benefits:**
- ~2-3x faster than sequential searches
- Configurable concurrency limit
- Automatic result mapping

### 5. Batch Passage Retrieval

Find passages for multiple queries concurrently:

```python
async def find_passages():
    queries = ["what is PageRank", "how does BM25 work"]

    results = await async_proc.batch_passages_async(
        queries,
        top_n=3,
        concurrency=2,
        chunk_size=512,
        overlap=128
    )

    for query, passages in results.items():
        for text, doc_id, start, end, score in passages:
            print(f"{query} -> {doc_id}[{start}:{end}]: {score:.4f}")
```

### 6. Remove Documents Asynchronously

Remove multiple documents in batches:

```python
async def remove_docs():
    doc_ids = ["old1", "old2", "old3"]

    result = await async_proc.remove_documents_async(
        doc_ids,
        progress_callback=lambda done, total: print(f"{done}/{total}"),
        chunk_size=10,
        recompute='tfidf'
    )

    print(f"Removed: {result['documents_removed']}")
    print(f"Not found: {result['documents_not_found']}")
```

### 7. Async Compute All

Run full corpus analysis asynchronously:

```python
async def compute():
    def phase_callback(phase):
        print(f"Running: {phase}")

    result = await async_proc.compute_all_async(
        progress_callback=phase_callback,
        verbose=False
    )

    return result
```

## Advanced Features

### Cancellation

Cancel long-running operations:

```python
async def process_with_timeout():
    # Create large batch
    documents = [(f"doc{i}", f"Content {i}", None) for i in range(1000)]

    # Start processing
    task = asyncio.create_task(
        async_proc.add_documents_async(documents, chunk_size=10)
    )

    try:
        # Wait with timeout
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        # Cancel if timeout exceeded
        async_proc.cancel()
        print("Operation cancelled due to timeout")

    # Reset to allow new operations
    async_proc.reset_cancel()
```

### Concurrent Operations

Run multiple async operations in parallel:

```python
async def concurrent_ops():
    async with AsyncProcessor(processor, max_workers=4) as async_proc:
        # Create multiple tasks
        add_task = async_proc.add_documents_async(documents)
        search_task = async_proc.search_async("query")
        batch_task = async_proc.batch_search_async(queries)

        # Wait for all to complete
        add_result, search_result, batch_result = await asyncio.gather(
            add_task,
            search_task,
            batch_task
        )

        return add_result, search_result, batch_result
```

### Resource Cleanup

Proper cleanup with async context managers:

```python
async def main():
    processor = CorticalTextProcessor()

    # Automatic cleanup on exit
    async with AsyncProcessor(processor) as async_proc:
        # Do work here
        results = await async_proc.search_async("query")
        return results
    # async_proc.close() called automatically
```

Or manual cleanup:

```python
async def main():
    processor = CorticalTextProcessor()
    async_proc = AsyncProcessor(processor)

    try:
        results = await async_proc.search_async("query")
        return results
    finally:
        await async_proc.close()
```

## Configuration

### Worker Pool Size

Control the number of worker threads:

```python
# More workers = more concurrency (but higher memory usage)
async_proc = AsyncProcessor(processor, max_workers=8)

# Fewer workers = less concurrency (but lower memory usage)
async_proc = AsyncProcessor(processor, max_workers=2)

# Default: 4 workers
async_proc = AsyncProcessor(processor)
```

**Guidelines:**
- CPU-bound tasks: `max_workers = CPU count`
- I/O-bound tasks: `max_workers = 2-4x CPU count`
- Memory-constrained: `max_workers = 2-4`

### Chunk Size

Control batch processing granularity:

```python
# Large chunks = fewer context switches (faster for large batches)
await async_proc.add_documents_async(
    documents,
    chunk_size=100  # Process 100 docs at a time
)

# Small chunks = more responsive progress (better for UI feedback)
await async_proc.add_documents_async(
    documents,
    chunk_size=10  # Process 10 docs at a time
)
```

**Guidelines:**
- Large batches (1000+ docs): `chunk_size = 50-100`
- Medium batches (100-1000 docs): `chunk_size = 10-50`
- Small batches (<100 docs): `chunk_size = 5-10`
- With progress callback: Use smaller chunks for responsive feedback

### Concurrency Limits

Control concurrent search operations:

```python
# High concurrency (fast, but high CPU usage)
results = await async_proc.batch_search_async(
    queries,
    concurrency=8
)

# Low concurrency (slower, but lower CPU usage)
results = await async_proc.batch_search_async(
    queries,
    concurrency=2
)
```

## Performance Tips

### 1. Batch Operations

Always prefer batch operations over sequential:

```python
# ❌ Slow: Sequential processing
for doc_id, content in documents:
    await async_proc.add_documents_async([(doc_id, content, None)])

# ✅ Fast: Batch processing
await async_proc.add_documents_async(documents)
```

### 2. Recomputation Strategy

Choose appropriate recomputation level:

```python
# Fast: No recomputation (defer until later)
await async_proc.add_documents_async(docs, recompute='none')

# Medium: Update TF-IDF only
await async_proc.add_documents_async(docs, recompute='tfidf')

# Slow: Full recomputation
await async_proc.add_documents_async(docs, recompute='full')
```

**Strategy:**
- Adding many documents: Use `recompute='none'` for each batch, then call `compute_all_async()` once at the end
- Adding few documents: Use `recompute='tfidf'` for fast search updates
- Final processing: Use `recompute='full'` for complete analysis

### 3. Concurrency Tuning

Match concurrency to your workload:

```python
# CPU count
import os
cpu_count = os.cpu_count()

# For search-heavy workloads
async_proc = AsyncProcessor(processor, max_workers=cpu_count)
results = await async_proc.batch_search_async(queries, concurrency=cpu_count)

# For memory-constrained environments
async_proc = AsyncProcessor(processor, max_workers=2)
results = await async_proc.batch_search_async(queries, concurrency=2)
```

## Common Patterns

### Pattern 1: Bulk Document Import

```python
async def bulk_import(file_paths):
    processor = CorticalTextProcessor()

    async with AsyncProcessor(processor, max_workers=4) as async_proc:
        documents = []

        # Read files (this part could also be async with aiofiles)
        for path in file_paths:
            with open(path) as f:
                content = f.read()
            documents.append((path, content, {"source": path}))

        # Add all documents with progress
        result = await async_proc.add_documents_async(
            documents,
            progress_callback=lambda done, total: print(f"{done}/{total}"),
            chunk_size=20,
            recompute='none'  # Defer recomputation
        )

        # Single recomputation at the end
        await async_proc.compute_all_async(
            progress_callback=lambda phase: print(f"Computing: {phase}")
        )

        return processor
```

### Pattern 2: Interactive Search

```python
async def interactive_search_service(processor):
    async with AsyncProcessor(processor, max_workers=8) as async_proc:
        while True:
            # Get query from user (e.g., from web API)
            query = await get_user_query()

            if query == "exit":
                break

            # Non-blocking search
            results = await async_proc.search_async(query, top_n=10)

            # Send results back
            await send_results(results)
```

### Pattern 3: Incremental Updates

```python
async def incremental_update_loop(processor):
    async with AsyncProcessor(processor, max_workers=2) as async_proc:
        while True:
            # Get new documents from queue
            new_docs = await get_new_documents_from_queue()

            if not new_docs:
                await asyncio.sleep(1)
                continue

            # Add incrementally with TF-IDF update
            await async_proc.add_documents_async(
                new_docs,
                chunk_size=10,
                recompute='tfidf'  # Fast incremental update
            )

            # Every 100 documents, do full recomputation
            if processor.document_count() % 100 == 0:
                await async_proc.compute_all_async()
```

## Error Handling

```python
async def robust_processing():
    async with AsyncProcessor(processor) as async_proc:
        try:
            # Operation that might fail
            result = await async_proc.add_documents_async(documents)
        except ValueError as e:
            # Handle validation errors
            print(f"Invalid input: {e}")
        except asyncio.CancelledError:
            # Handle cancellation
            print("Operation cancelled")
            async_proc.reset_cancel()
        except Exception as e:
            # Handle other errors
            print(f"Unexpected error: {e}")
```

## API Reference

### AsyncProcessor

```python
class AsyncProcessor:
    def __init__(self, processor, max_workers: int = 4)

    async def add_documents_async(
        self,
        documents: List[Tuple[str, str, Optional[Dict]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 10,
        recompute: str = 'full'
    ) -> Dict[str, Any]

    async def search_async(
        self,
        query: str,
        top_n: int = 5,
        use_expansion: bool = True
    ) -> List[Tuple[str, float]]

    async def batch_search_async(
        self,
        queries: List[str],
        top_n: int = 5,
        concurrency: int = 4
    ) -> Dict[str, List[Tuple[str, float]]]

    async def compute_all_async(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Dict[str, Any]

    async def remove_documents_async(
        self,
        doc_ids: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 10,
        recompute: str = 'none'
    ) -> Dict[str, Any]

    async def batch_passages_async(
        self,
        queries: List[str],
        top_n: int = 5,
        concurrency: int = 4,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> Dict[str, List[Tuple[str, str, int, int, float]]]

    def cancel(self) -> None
    def reset_cancel(self) -> None
    async def close(self) -> None

    async def __aenter__(self) -> 'AsyncProcessor'
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool
```

## Examples

See `examples/async_api_demo.py` for comprehensive demonstrations of all async API features.

## Testing

The async API includes comprehensive tests in `tests/unit/test_async_api.py` covering:
- Async document addition
- Concurrent searches
- Progress callbacks
- Cancellation
- Error handling
- Resource cleanup
- Async context managers

Run tests with:
```bash
python -m pytest tests/unit/test_async_api.py -v
```

## See Also

- [Batch Operations](batch-operations.md) - Sync batch processing
- [Performance Guide](performance.md) - Optimization strategies
- [RAG Integration](rag-integration.md) - Using async API with RAG systems
