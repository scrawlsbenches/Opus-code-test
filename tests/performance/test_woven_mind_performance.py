"""
Performance benchmarks for WovenMind cognitive architecture.

Measures throughput, latency, and memory characteristics.

Part of Sprint 6: Integration & Polish (T6.2)
"""

import pytest
import time
import sys
from typing import Dict, List, Any


class TestTrainingPerformance:
    """Benchmark training throughput."""

    def test_training_throughput_small_corpus(self):
        """Measure training speed on small corpus."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        corpus = [f"document {i} contains words about topic {i % 10}" for i in range(100)]

        start = time.perf_counter()
        for doc in corpus:
            mind.train(doc)
        elapsed = time.perf_counter() - start

        docs_per_second = len(corpus) / elapsed
        print(f"\n  Training throughput: {docs_per_second:.1f} docs/sec ({elapsed*1000:.1f}ms total)")

        # Should train at least 100 docs/sec
        assert docs_per_second >= 100, f"Training too slow: {docs_per_second:.1f} docs/sec"

    def test_training_throughput_medium_corpus(self):
        """Measure training speed on medium corpus."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        corpus = [f"document {i} with longer content about neural networks and machine learning patterns"
                  for i in range(500)]

        start = time.perf_counter()
        for doc in corpus:
            mind.train(doc)
        elapsed = time.perf_counter() - start

        docs_per_second = len(corpus) / elapsed
        print(f"\n  Training throughput: {docs_per_second:.1f} docs/sec ({elapsed*1000:.1f}ms total)")

        # Should train at least 50 docs/sec for medium corpus
        assert docs_per_second >= 50, f"Training too slow: {docs_per_second:.1f} docs/sec"

    def test_incremental_training_overhead(self):
        """Measure overhead of incremental training."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Warm up
        for i in range(100):
            mind.train(f"warmup document {i}")

        # Measure incremental
        times = []
        for i in range(50):
            start = time.perf_counter()
            mind.train(f"incremental document {i}")
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        print(f"\n  Incremental training: avg={avg_time:.2f}ms, max={max_time:.2f}ms")

        # Each incremental train should be fast (< 50ms)
        assert avg_time < 50, f"Incremental training too slow: {avg_time:.2f}ms"


class TestProcessingPerformance:
    """Benchmark processing latency."""

    def test_processing_latency_cold(self):
        """Measure processing latency on cold start."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("basic training content")

        # Cold processing
        start = time.perf_counter()
        result = mind.process(["test", "input"])
        cold_latency = (time.perf_counter() - start) * 1000

        print(f"\n  Cold processing latency: {cold_latency:.2f}ms")

        # Cold latency should be under 10ms
        assert cold_latency < 10, f"Cold latency too high: {cold_latency:.2f}ms"

    def test_processing_latency_warm(self):
        """Measure processing latency after warm-up."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Train and warm up
        for i in range(50):
            mind.train(f"training document {i} with content")
            mind.process([f"doc{i}", "content"])

        # Measure warm latency
        times = []
        for i in range(100):
            start = time.perf_counter()
            mind.process([f"input{i}", "test"])
            times.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(times) / len(times)
        p50 = sorted(times)[50]
        p95 = sorted(times)[95]
        p99 = sorted(times)[99]

        print(f"\n  Warm processing latency: avg={avg_latency:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")

        # Warm latency should be under 5ms average
        assert avg_latency < 5, f"Average latency too high: {avg_latency:.2f}ms"

    def test_processing_throughput(self):
        """Measure processing throughput (calls/sec)."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("neural networks process data")

        num_calls = 1000
        start = time.perf_counter()
        for i in range(num_calls):
            mind.process([f"token{i}"])
        elapsed = time.perf_counter() - start

        calls_per_second = num_calls / elapsed
        print(f"\n  Processing throughput: {calls_per_second:.1f} calls/sec")

        # Should handle at least 500 calls/sec
        assert calls_per_second >= 500, f"Throughput too low: {calls_per_second:.1f} calls/sec"


class TestConsolidationPerformance:
    """Benchmark consolidation timing."""

    def test_consolidation_latency_empty(self):
        """Measure consolidation on empty state."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        start = time.perf_counter()
        result = mind.consolidate()
        latency = (time.perf_counter() - start) * 1000

        print(f"\n  Empty consolidation: {latency:.2f}ms (reported: {result.cycle_duration_ms:.2f}ms)")

        # Empty consolidation should be very fast (< 5ms)
        assert latency < 5, f"Empty consolidation too slow: {latency:.2f}ms"

    def test_consolidation_latency_with_patterns(self):
        """Measure consolidation with recorded patterns."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Build up patterns
        for i in range(100):
            mind.train(f"pattern document {i}")
            mind.process([f"pattern{i}", "content"])
            mind.consolidation.record_pattern({f"pattern{i}", "content"})

        start = time.perf_counter()
        result = mind.consolidate()
        latency = (time.perf_counter() - start) * 1000

        print(f"\n  Consolidation with 100 patterns: {latency:.2f}ms (reported: {result.cycle_duration_ms:.2f}ms)")

        # Consolidation should be under 100ms for 100 patterns
        assert latency < 100, f"Consolidation too slow: {latency:.2f}ms"

    def test_multiple_consolidation_cycles(self):
        """Measure latency consistency across cycles."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Train
        for i in range(50):
            mind.train(f"training {i}")

        # Multiple consolidation cycles
        times = []
        for _ in range(10):
            start = time.perf_counter()
            mind.consolidate()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"\n  Multi-cycle consolidation: avg={avg_time:.2f}ms, max={max_time:.2f}ms")

        # Should be consistent - but for very fast operations (sub-millisecond),
        # timing jitter on build servers can cause high variance ratios even when
        # the absolute variance is negligible. Use a minimum floor to avoid
        # failing on operations that are already extremely fast.
        MIN_TIME_FLOOR_MS = 0.1  # Below this, absolute time is negligible
        effective_avg = max(avg_time, MIN_TIME_FLOOR_MS)
        assert max_time < effective_avg * 3, f"Consolidation latency too variable: max={max_time:.2f}ms, avg={avg_time:.2f}ms"


class TestMemoryPerformance:
    """Benchmark memory usage."""

    def test_memory_growth_during_training(self):
        """Measure memory growth as training progresses."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        sizes = []
        for batch in range(10):
            for i in range(100):
                mind.train(f"batch {batch} document {i}")

            data = mind.to_dict()
            size = sys.getsizeof(str(data))
            sizes.append(size)

        # Print growth pattern
        print(f"\n  Memory growth: {[s//1024 for s in sizes]} KB")

        # Growth should be sublinear or linear, not exponential
        if len(sizes) >= 3:
            early_growth = sizes[2] / max(sizes[0], 1)
            late_growth = sizes[-1] / max(sizes[-3], 1)
            # Late growth rate should not be much higher than early
            assert late_growth <= early_growth * 2, f"Memory growth accelerating: early={early_growth:.2f}x, late={late_growth:.2f}x"

    def test_memory_after_consolidation(self):
        """Measure memory impact of consolidation."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Train
        for i in range(200):
            mind.train(f"document {i} with content")
            mind.consolidation.record_pattern({f"doc{i}"})

        before_data = mind.to_dict()
        before_size = sys.getsizeof(str(before_data))

        # Consolidate
        mind.consolidate()

        after_data = mind.to_dict()
        after_size = sys.getsizeof(str(after_data))

        print(f"\n  Memory: before={before_size//1024}KB, after={after_size//1024}KB")

        # Consolidation should not significantly increase memory
        growth = after_size / max(before_size, 1)
        assert growth < 1.5, f"Consolidation increased memory too much: {growth:.2f}x"

    def test_memory_per_pattern(self):
        """Estimate memory per recorded pattern."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        base_data = mind.to_dict()
        base_size = sys.getsizeof(str(base_data))

        # Record many patterns
        num_patterns = 1000
        for i in range(num_patterns):
            mind.consolidation.record_pattern({f"pattern{i}", "content"})

        after_data = mind.to_dict()
        after_size = sys.getsizeof(str(after_data))

        bytes_per_pattern = (after_size - base_size) / num_patterns
        print(f"\n  Memory per pattern: {bytes_per_pattern:.1f} bytes")

        # Should be reasonable (< 500 bytes per pattern)
        assert bytes_per_pattern < 500, f"Too much memory per pattern: {bytes_per_pattern:.1f} bytes"


class TestSerializationPerformance:
    """Benchmark serialization performance."""

    def test_serialization_latency_small(self):
        """Measure serialization latency for small state."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        for i in range(10):
            mind.train(f"small corpus {i}")

        start = time.perf_counter()
        data = mind.to_dict()
        serialize_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        restored = WovenMind.from_dict(data)
        deserialize_time = (time.perf_counter() - start) * 1000

        print(f"\n  Small state: serialize={serialize_time:.2f}ms, deserialize={deserialize_time:.2f}ms")

        # Both should be fast for small state
        assert serialize_time < 50, f"Serialization too slow: {serialize_time:.2f}ms"
        assert deserialize_time < 50, f"Deserialization too slow: {deserialize_time:.2f}ms"

    def test_serialization_latency_medium(self):
        """Measure serialization latency for medium state."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        for i in range(100):
            mind.train(f"medium corpus document {i} with more content")
            mind.consolidation.record_pattern({f"doc{i}", "content"})

        start = time.perf_counter()
        data = mind.to_dict()
        serialize_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        restored = WovenMind.from_dict(data)
        deserialize_time = (time.perf_counter() - start) * 1000

        print(f"\n  Medium state: serialize={serialize_time:.2f}ms, deserialize={deserialize_time:.2f}ms")

        # Should still be reasonably fast
        assert serialize_time < 200, f"Serialization too slow: {serialize_time:.2f}ms"
        assert deserialize_time < 200, f"Deserialization too slow: {deserialize_time:.2f}ms"

    def test_serialization_roundtrip_integrity(self):
        """Verify serialization preserves state."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        for i in range(50):
            mind.train(f"integrity test {i}")
            mind.consolidation.record_pattern({f"test{i}"})
        mind.consolidate()

        original_stats = mind.get_stats()

        # Roundtrip
        data = mind.to_dict()
        restored = WovenMind.from_dict(data)
        restored_stats = restored.get_stats()

        # Key metrics should match
        assert original_stats["mode"] == restored_stats["mode"]


class TestScalability:
    """Test scalability characteristics."""

    def test_training_scales_linearly(self):
        """Verify training time scales linearly with corpus size."""
        from cortical.reasoning.woven_mind import WovenMind

        times = []
        sizes = [100, 200, 400]

        for size in sizes:
            mind = WovenMind()
            corpus = [f"document {i}" for i in range(size)]

            start = time.perf_counter()
            for doc in corpus:
                mind.train(doc)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\n  Training scaling: sizes={sizes}, times={[f'{t*1000:.1f}ms' for t in times]}")

        # Check for roughly linear scaling
        # Doubling corpus should roughly double time (allow 3x due to overhead)
        if times[0] > 0.001:  # Avoid division by zero
            ratio_1 = times[1] / times[0]
            ratio_2 = times[2] / times[1]

            # Ratios should be roughly 2x (allow 1x to 4x)
            assert 1.0 <= ratio_1 <= 4.0, f"Non-linear scaling: ratio={ratio_1:.2f}"
            assert 1.0 <= ratio_2 <= 4.0, f"Non-linear scaling: ratio={ratio_2:.2f}"

    def test_processing_scales_with_context_length(self):
        """Verify processing time with different context lengths."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("neural networks process data efficiently")

        lengths = [1, 5, 10, 20]
        times = []

        for length in lengths:
            context = [f"token{i}" for i in range(length)]

            # Average over multiple runs
            run_times = []
            for _ in range(10):
                start = time.perf_counter()
                mind.process(context)
                run_times.append((time.perf_counter() - start) * 1000)
            times.append(sum(run_times) / len(run_times))

        print(f"\n  Context length scaling: lengths={lengths}, times={[f'{t:.2f}ms' for t in times]}")

        # All should be fast
        for t in times:
            assert t < 10, f"Processing too slow: {t:.2f}ms"


class TestRealWorldScenarios:
    """Performance in realistic usage patterns."""

    def test_interactive_session_simulation(self):
        """Simulate an interactive coding session."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Simulate session: train, query, train, consolidate pattern
        total_time = 0
        operations = []

        for cycle in range(10):
            # Train on new content
            start = time.perf_counter()
            for i in range(5):
                mind.train(f"cycle {cycle} code snippet {i}")
            train_time = (time.perf_counter() - start) * 1000

            # Process queries
            start = time.perf_counter()
            for i in range(10):
                mind.process([f"query{i}", "token"])
            query_time = (time.perf_counter() - start) * 1000

            # Periodic consolidation
            start = time.perf_counter()
            if cycle % 3 == 0:
                mind.consolidate()
            consolidate_time = (time.perf_counter() - start) * 1000

            total_time += train_time + query_time + consolidate_time
            operations.append({
                "cycle": cycle,
                "train_ms": train_time,
                "query_ms": query_time,
                "consolidate_ms": consolidate_time,
            })

        print(f"\n  Interactive session: {total_time:.1f}ms total for 10 cycles")
        print(f"    Avg train: {sum(o['train_ms'] for o in operations)/len(operations):.2f}ms")
        print(f"    Avg query: {sum(o['query_ms'] for o in operations)/len(operations):.2f}ms")

        # Entire session should be fast
        assert total_time < 1000, f"Session too slow: {total_time:.1f}ms"

    def test_batch_processing_simulation(self):
        """Simulate batch document processing."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        # Large batch of documents
        num_docs = 500
        docs = [f"document {i} with content about topic {i % 20}" for i in range(num_docs)]

        start = time.perf_counter()
        for doc in docs:
            mind.train(doc)
            mind.process(doc.split()[:3])
        total_time = (time.perf_counter() - start) * 1000

        docs_per_second = num_docs / (total_time / 1000)
        print(f"\n  Batch processing: {docs_per_second:.1f} docs/sec ({total_time:.1f}ms for {num_docs} docs)")

        # Should handle at least 50 docs/sec including processing
        assert docs_per_second >= 50, f"Batch processing too slow: {docs_per_second:.1f} docs/sec"
