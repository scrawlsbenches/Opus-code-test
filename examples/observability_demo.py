#!/usr/bin/env python3
"""
Observability Demo
==================

Demonstrates the observability features of the Cortical Text Processor:
- Timing metrics for operations
- Cache hit/miss tracking
- Custom metric recording
- Metrics summary generation

Run this script to see how metrics collection works.
"""

from cortical import CorticalTextProcessor


def main():
    print("=" * 80)
    print("Cortical Text Processor - Observability Demo")
    print("=" * 80)

    # Create processor with metrics enabled
    print("\n1. Creating processor with metrics enabled...")
    processor = CorticalTextProcessor(enable_metrics=True)

    # Process some documents
    print("\n2. Processing documents...")
    processor.process_document(
        "neural_nets",
        "Neural networks are computational models inspired by biological neural networks. "
        "They consist of layers of interconnected nodes that process information."
    )
    processor.process_document(
        "machine_learning",
        "Machine learning is a branch of artificial intelligence that focuses on building "
        "systems that can learn from data. It includes supervised and unsupervised learning."
    )
    processor.process_document(
        "deep_learning",
        "Deep learning uses neural networks with multiple layers to learn hierarchical "
        "representations of data. It has achieved remarkable success in image and speech recognition."
    )

    # Compute analysis
    print("\n3. Running compute_all()...")
    processor.compute_all(verbose=False)

    # Perform some queries
    print("\n4. Performing queries...")
    processor.find_documents_for_query("neural networks")
    processor.find_documents_for_query("machine learning algorithms")

    # Use cached queries to demonstrate cache metrics
    print("\n5. Testing query cache (first call = miss, second = hit)...")
    processor.expand_query_cached("neural")  # Cache miss
    processor.expand_query_cached("neural")  # Cache hit
    processor.expand_query_cached("learning")  # Cache miss
    processor.expand_query_cached("learning")  # Cache hit

    # Record custom metrics
    print("\n6. Recording custom metrics...")
    processor.record_metric("api_calls", 10)
    processor.record_metric("api_calls", 5)
    processor.record_metric("users_active", 3)

    # Display metrics summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(processor.get_metrics_summary())

    # Get detailed metrics programmatically
    print("\n" + "=" * 80)
    print("DETAILED METRICS (Programmatic Access)")
    print("=" * 80)
    metrics = processor.get_metrics()

    if "compute_all" in metrics:
        stats = metrics["compute_all"]
        print(f"\ncompute_all:")
        print(f"  Executed: {stats['count']} time(s)")
        print(f"  Average: {stats['avg_ms']:.2f}ms")
        print(f"  Min: {stats['min_ms']:.2f}ms")
        print(f"  Max: {stats['max_ms']:.2f}ms")

    if "find_documents_for_query" in metrics:
        stats = metrics["find_documents_for_query"]
        print(f"\nfind_documents_for_query:")
        print(f"  Executed: {stats['count']} time(s)")
        print(f"  Average: {stats['avg_ms']:.2f}ms")

    if "query_cache_hits" in metrics:
        hits = metrics["query_cache_hits"]["count"]
        misses = metrics["query_cache_misses"]["count"]
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        print(f"\nQuery Cache Performance:")
        print(f"  Hits: {hits}")
        print(f"  Misses: {misses}")
        print(f"  Hit Rate: {hit_rate:.1f}%")

    # Demonstrate disabling metrics
    print("\n" + "=" * 80)
    print("DISABLING METRICS")
    print("=" * 80)
    processor.disable_metrics()
    print("Metrics disabled. Processing more documents (not timed)...")
    processor.process_document("new_doc", "This won't be timed.")

    # Re-enable and show metrics haven't changed
    processor.enable_metrics()
    metrics_after = processor.get_metrics()
    print(f"Operations still in metrics: {len(metrics_after)}")
    print("(Metrics from before disable were preserved)")

    # Demonstrate reset
    print("\n" + "=" * 80)
    print("RESETTING METRICS")
    print("=" * 80)
    processor.reset_metrics()
    metrics_after_reset = processor.get_metrics()
    print(f"Operations after reset: {len(metrics_after_reset)}")
    print("(All metrics cleared)")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
