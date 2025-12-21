#!/usr/bin/env python3
"""
Demonstration of async API for batch operations.

This example shows how to use AsyncProcessor for non-blocking document
processing and concurrent search operations.

Run with:
    python examples/async_api_demo.py
"""

import asyncio
import time
from cortical import CorticalTextProcessor
from cortical.async_api import AsyncProcessor


async def demo_basic_usage():
    """Basic async operations."""
    print("=" * 70)
    print("Demo 1: Basic Async Operations")
    print("=" * 70)

    # Create processor and async wrapper
    processor = CorticalTextProcessor()
    async with AsyncProcessor(processor, max_workers=4) as async_proc:
        # Add documents asynchronously
        documents = [
            ("doc1", "Neural networks are computational models inspired by biological brains.", None),
            ("doc2", "Machine learning algorithms learn patterns from training data.", None),
            ("doc3", "Deep learning uses multiple layers for hierarchical feature extraction.", None),
            ("doc4", "Natural language processing enables computers to understand human language.", None),
        ]

        print("\nAdding documents asynchronously...")
        result = await async_proc.add_documents_async(
            documents,
            chunk_size=2,
            recompute='full'
        )
        print(f"✓ Added {result['documents_added']} documents")
        print(f"  Total tokens: {result['total_tokens']}")
        print(f"  Chunks processed: {result['chunks_processed']}")

        # Search asynchronously
        print("\nSearching asynchronously...")
        results = await async_proc.search_async("neural networks", top_n=3)
        print("✓ Search results:")
        for doc_id, score in results:
            print(f"  - {doc_id}: {score:.4f}")


async def demo_progress_callbacks():
    """Demonstrate progress tracking."""
    print("\n" + "=" * 70)
    print("Demo 2: Progress Callbacks")
    print("=" * 70)

    processor = CorticalTextProcessor()
    async with AsyncProcessor(processor, max_workers=2) as async_proc:
        # Create larger document set
        documents = [
            (f"article_{i}", f"Article {i} discusses various machine learning topics.", None)
            for i in range(20)
        ]

        print("\nAdding 20 documents with progress tracking...")

        def progress_callback(done, total):
            percent = (done / total) * 100
            bar_length = 40
            filled = int(bar_length * done / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r  Progress: [{bar}] {done}/{total} ({percent:.1f}%)", end="", flush=True)

        result = await async_proc.add_documents_async(
            documents,
            progress_callback=progress_callback,
            chunk_size=5,
            recompute='tfidf'
        )
        print()  # New line after progress bar
        print(f"✓ Completed: {result['documents_added']} documents added")


async def demo_concurrent_searches():
    """Demonstrate concurrent search operations."""
    print("\n" + "=" * 70)
    print("Demo 3: Concurrent Batch Search")
    print("=" * 70)

    # Set up processor with sample documents
    processor = CorticalTextProcessor()
    documents = [
        ("ai_overview", "Artificial intelligence encompasses machine learning and deep learning.", None),
        ("ml_basics", "Machine learning algorithms learn from data without explicit programming.", None),
        ("dl_intro", "Deep learning uses neural networks with multiple layers.", None),
        ("nlp_guide", "Natural language processing enables text understanding and generation.", None),
        ("cv_intro", "Computer vision allows machines to interpret visual information.", None),
    ]

    # Add documents synchronously first
    for doc_id, content, metadata in documents:
        processor.process_document(doc_id, content, metadata)
    processor.compute_all(verbose=False)

    async with AsyncProcessor(processor, max_workers=4) as async_proc:
        queries = [
            "neural networks",
            "machine learning algorithms",
            "natural language",
            "computer vision"
        ]

        print(f"\nRunning {len(queries)} concurrent searches...")
        start_time = time.time()

        results = await async_proc.batch_search_async(
            queries,
            top_n=3,
            concurrency=4
        )

        elapsed = time.time() - start_time

        print(f"✓ Completed {len(results)} searches in {elapsed:.3f}s")
        print("\nResults:")
        for query, matches in results.items():
            print(f"\n  Query: '{query}'")
            for doc_id, score in matches[:2]:  # Show top 2
                print(f"    - {doc_id}: {score:.4f}")


async def demo_passage_retrieval():
    """Demonstrate concurrent passage retrieval."""
    print("\n" + "=" * 70)
    print("Demo 4: Concurrent Passage Retrieval")
    print("=" * 70)

    processor = CorticalTextProcessor()

    # Add longer documents for passage retrieval
    documents = [
        ("ml_article", """
        Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience. Supervised learning uses labeled data
        to train models, while unsupervised learning finds patterns in unlabeled data.
        Reinforcement learning teaches agents through trial and error with rewards.
        """, None),
        ("dl_article", """
        Deep learning uses artificial neural networks with multiple layers to learn
        hierarchical representations of data. Convolutional neural networks excel at
        image processing, while recurrent neural networks handle sequential data.
        Transformers have revolutionized natural language processing tasks.
        """, None),
    ]

    for doc_id, content, metadata in documents:
        processor.process_document(doc_id, content, metadata)
    processor.compute_all(verbose=False)

    async with AsyncProcessor(processor, max_workers=3) as async_proc:
        queries = [
            "supervised learning",
            "convolutional networks"
        ]

        print(f"\nFinding passages for {len(queries)} queries...")
        results = await async_proc.batch_passages_async(
            queries,
            top_n=2,
            concurrency=2,
            chunk_size=200,
            overlap=50
        )

        print("✓ Passage retrieval completed")
        for query, passages in results.items():
            print(f"\n  Query: '{query}'")
            for passage_text, doc_id, start, end, score in passages[:1]:  # Show top 1
                snippet = passage_text[:100].replace('\n', ' ').strip()
                print(f"    - {doc_id}[{start}:{end}]: {snippet}... (score: {score:.4f})")


async def demo_cancellation():
    """Demonstrate operation cancellation."""
    print("\n" + "=" * 70)
    print("Demo 5: Operation Cancellation")
    print("=" * 70)

    processor = CorticalTextProcessor()
    async_proc = AsyncProcessor(processor, max_workers=2)

    try:
        # Create a large batch that would take time
        large_batch = [
            (f"doc_{i}", f"Document {i} content with various topics.", None)
            for i in range(100)
        ]

        print("\nStarting large batch processing...")
        print("(Will be cancelled after 0.5 seconds)")

        # Start the operation
        task = asyncio.create_task(
            async_proc.add_documents_async(large_batch, chunk_size=5, recompute='none')
        )

        # Wait a bit then cancel
        await asyncio.sleep(0.5)
        async_proc.cancel()

        try:
            await task
            print("✗ Task should have been cancelled")
        except asyncio.CancelledError:
            print("✓ Task cancelled successfully")

        # Reset and verify we can continue
        async_proc.reset_cancel()
        print("\nReset cancellation flag, adding single document...")

        result = await async_proc.add_documents_async(
            [("after_cancel", "Document added after cancellation reset.", None)],
            recompute='none'
        )
        print(f"✓ Document added after reset: {result['documents_added']}")

    finally:
        await async_proc.close()


async def demo_compute_all_async():
    """Demonstrate async compute_all with progress."""
    print("\n" + "=" * 70)
    print("Demo 6: Async Compute All")
    print("=" * 70)

    processor = CorticalTextProcessor()

    # Add documents without computing
    for i in range(10):
        processor.process_document(f"compute_doc_{i}", f"Document {i} about various ML topics.", None)

    async with AsyncProcessor(processor, max_workers=2) as async_proc:
        phases = []

        def phase_callback(phase):
            phases.append(phase)
            print(f"  {phase}")

        print("\nRunning compute_all asynchronously...")
        result = await async_proc.compute_all_async(
            progress_callback=phase_callback,
            verbose=False
        )

        print(f"✓ Compute all completed")
        print(f"  Phases tracked: {len(phases)}")


async def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "Async API Demonstration" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    await demo_basic_usage()
    await demo_progress_callbacks()
    await demo_concurrent_searches()
    await demo_passage_retrieval()
    await demo_cancellation()
    await demo_compute_all_async()

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
