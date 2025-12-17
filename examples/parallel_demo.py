"""
Parallel Processing Demo

This example demonstrates the parallel processing capabilities
for TF-IDF and BM25 computation on large corpora.

For small corpora (< 2000 terms), parallel processing automatically
falls back to sequential to avoid multiprocessing overhead.

For large corpora (5000+ terms), parallel processing can provide
2-3x speedup by utilizing multiple CPU cores.
"""

import time
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig


def create_large_corpus(num_docs=200, terms_per_doc=50):
    """Create a synthetic large corpus for testing."""
    docs = []
    for i in range(num_docs):
        # Generate unique terms for each document
        terms = [f"term_{j}" for j in range(i*terms_per_doc, (i+1)*terms_per_doc)]
        # Add some common terms for connectivity
        terms.extend(["neural", "network", "data", "processing", "algorithm"])
        docs.append(" ".join(terms))
    return docs


def demo_sequential_vs_parallel():
    """Compare sequential vs parallel processing."""
    print("=" * 70)
    print("Parallel Processing Demo")
    print("=" * 70)

    # Create large corpus
    print("\nCreating large corpus (200 docs, ~10k unique terms)...")
    docs = create_large_corpus(num_docs=200, terms_per_doc=50)

    # Test with TF-IDF
    print("\n" + "=" * 70)
    print("TF-IDF Comparison")
    print("=" * 70)

    # Sequential
    processor_seq = CorticalTextProcessor()
    for i, doc in enumerate(docs):
        processor_seq.process_document(f"doc_{i}", doc)

    print(f"\nProcessed {len(processor_seq.documents)} documents")
    print(f"Total unique terms: {processor_seq.layers[0].column_count()}")

    print("\n1. Sequential TF-IDF...")
    start = time.time()
    processor_seq.compute_tfidf(verbose=False)
    seq_time = time.time() - start
    print(f"   Time: {seq_time:.3f}s")

    # Parallel
    processor_par = CorticalTextProcessor()
    for i, doc in enumerate(docs):
        processor_par.process_document(f"doc_{i}", doc)

    print("\n2. Parallel TF-IDF (4 workers)...")
    start = time.time()
    stats = processor_par.compute_tfidf_parallel(
        num_workers=4,
        chunk_size=1000,
        verbose=False
    )
    par_time = time.time() - start
    print(f"   Time: {par_time:.3f}s")
    print(f"   Method used: {stats['method']}")
    print(f"   Terms processed: {stats['terms_processed']}")

    if par_time < seq_time:
        speedup = seq_time / par_time
        print(f"\n   ✅ Speedup: {speedup:.2f}x faster!")
    else:
        print(f"\n   ℹ️  Parallel was not faster (corpus may be too small)")

    # Test with BM25
    print("\n" + "=" * 70)
    print("BM25 Comparison")
    print("=" * 70)

    # Sequential
    processor_seq2 = CorticalTextProcessor(
        config=CorticalConfig(scoring_algorithm='bm25')
    )
    for i, doc in enumerate(docs):
        processor_seq2.process_document(f"doc_{i}", doc)

    print("\n1. Sequential BM25...")
    start = time.time()
    processor_seq2.compute_bm25(verbose=False)
    seq_time = time.time() - start
    print(f"   Time: {seq_time:.3f}s")

    # Parallel
    processor_par2 = CorticalTextProcessor(
        config=CorticalConfig(scoring_algorithm='bm25')
    )
    for i, doc in enumerate(docs):
        processor_par2.process_document(f"doc_{i}", doc)

    print("\n2. Parallel BM25 (4 workers)...")
    start = time.time()
    stats = processor_par2.compute_bm25_parallel(
        num_workers=4,
        chunk_size=1000,
        verbose=False
    )
    par_time = time.time() - start
    print(f"   Time: {par_time:.3f}s")
    print(f"   Method used: {stats['method']}")
    print(f"   Terms processed: {stats['terms_processed']}")

    if par_time < seq_time:
        speedup = seq_time / par_time
        print(f"\n   ✅ Speedup: {speedup:.2f}x faster!")
    else:
        print(f"\n   ℹ️  Parallel was not faster (corpus may be too small)")


def demo_compute_all_parallel():
    """Demo using parallel processing with compute_all()."""
    print("\n" + "=" * 70)
    print("compute_all() with Parallel Processing")
    print("=" * 70)

    # Create corpus
    print("\nCreating corpus...")
    docs = create_large_corpus(num_docs=100, terms_per_doc=40)

    processor = CorticalTextProcessor()
    for i, doc in enumerate(docs):
        processor.process_document(f"doc_{i}", doc)

    print(f"Processed {len(processor.documents)} documents")
    print(f"Total unique terms: {processor.layers[0].column_count()}")

    # Run compute_all with parallel flag
    print("\nRunning compute_all(parallel=True, verbose=False)...")
    start = time.time()
    stats = processor.compute_all(
        parallel=True,
        parallel_num_workers=4,
        parallel_chunk_size=1000,
        verbose=False,
        build_concepts=False  # Skip concepts for faster demo
    )
    elapsed = time.time() - start

    print(f"\n✅ Complete in {elapsed:.2f}s")
    print(f"   All phases computed with parallel TF-IDF")


def demo_small_corpus_fallback():
    """Demo automatic fallback to sequential for small corpora."""
    print("\n" + "=" * 70)
    print("Automatic Fallback for Small Corpus")
    print("=" * 70)

    # Create small corpus
    processor = CorticalTextProcessor()
    processor.process_document("doc1", "neural networks process data efficiently")
    processor.process_document("doc2", "machine learning algorithms analyze patterns")
    processor.process_document("doc3", "deep learning models require large datasets")

    print(f"\nSmall corpus: {len(processor.documents)} documents")
    print(f"Total unique terms: {processor.layers[0].column_count()}")

    # Try parallel (will automatically fall back to sequential)
    print("\nCalling compute_tfidf_parallel()...")
    stats = processor.compute_tfidf_parallel(verbose=False)

    print(f"✅ Method used: {stats['method']}")
    print("   (Automatically fell back to sequential due to small corpus size)")


if __name__ == "__main__":
    # Run demos
    demo_sequential_vs_parallel()
    demo_compute_all_parallel()
    demo_small_corpus_fallback()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Parallel processing provides 2-3x speedup on large corpora")
    print("  • Automatically falls back to sequential for small corpora")
    print("  • Works with both TF-IDF and BM25 scoring")
    print("  • Integrated into compute_all() with parallel=True flag")
    print("  • Zero external dependencies (uses stdlib only)")
