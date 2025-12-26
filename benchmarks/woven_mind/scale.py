"""
Scalability benchmarks for Woven Mind.

These benchmarks test performance characteristics as corpus size grows:
- Processing time vs corpus size
- Memory usage patterns
- Cold start performance
"""

from typing import Any, Dict, List, Optional, Tuple
import statistics
import time

from .base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
    generate_synthetic_corpus,
    measure_time,
)


class ScalabilityBenchmark(BaseBenchmark):
    """
    Test performance scaling with corpus size.

    Hypothesis: Processing time should scale sub-quadratically with corpus size.

    Measures:
    - Training time vs corpus size
    - Query time vs corpus size
    - Consolidation time vs knowledge base size
    """

    name = "scalability"
    description = "Test performance scaling with corpus size"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.corpus_sizes = config.get("corpus_sizes", [100, 500, 1000, 2000]) if config else [100, 500, 1000, 2000]
        self.corpora: Dict[int, List[str]] = {}

    def setup(self) -> None:
        """Generate corpora of various sizes."""
        max_size = max(self.corpus_sizes)
        full_corpus = generate_synthetic_corpus(
            n_docs=max_size,
            doc_length=50,
            vocab_size=2000,
            pattern_frequency=0.3,
        )

        for size in self.corpus_sizes:
            self.corpora[size] = full_corpus[:size]

    def run(self) -> BenchmarkResult:
        """Run scalability analysis."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        try:
            from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
        except ImportError as e:
            result.status = BenchmarkStatus.ERROR
            result.error_message = f"Could not import Woven Mind: {e}"
            return result

        training_times: List[Tuple[int, float]] = []
        query_times: List[Tuple[int, float]] = []
        consolidation_times: List[Tuple[int, float]] = []

        for size in self.corpus_sizes:
            corpus = self.corpora[size]
            config = WovenMindConfig()
            mind = WovenMind(config=config)

            # Measure training time
            start = time.perf_counter()
            for doc in corpus:
                mind.train(doc)
            train_time_ms = (time.perf_counter() - start) * 1000
            training_times.append((size, train_time_ms))

            # Measure query time (average over multiple queries)
            test_queries = [
                ["pattern", "alpha"],
                ["word1", "word2"],
                ["test", "input"],
            ]

            query_times_for_size = []
            for query in test_queries:
                start = time.perf_counter()
                mind.process(query)
                query_times_for_size.append((time.perf_counter() - start) * 1000)

            avg_query_time = statistics.mean(query_times_for_size)
            query_times.append((size, avg_query_time))

            # Measure consolidation time
            start = time.perf_counter()
            if hasattr(mind, 'consolidate'):
                mind.consolidate()
            consolidation_time_ms = (time.perf_counter() - start) * 1000
            consolidation_times.append((size, consolidation_time_ms))

        # Analyze scaling behavior
        # Fit power law: time = a * size^b
        # If b < 2, we have sub-quadratic scaling

        def estimate_exponent(data: List[Tuple[int, float]]) -> float:
            """Estimate power law exponent using log-log regression."""
            if len(data) < 2:
                return 1.0

            import math
            log_sizes = [math.log(s) for s, _ in data]
            log_times = [math.log(t) if t > 0 else 0 for _, t in data]

            n = len(data)
            sum_x = sum(log_sizes)
            sum_y = sum(log_times)
            sum_xy = sum(x * y for x, y in zip(log_sizes, log_times))
            sum_x2 = sum(x * x for x in log_sizes)

            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 1.0

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope

        training_exponent = estimate_exponent(training_times)
        query_exponent = estimate_exponent(query_times)
        consolidation_exponent = estimate_exponent(consolidation_times)

        result.add_metric(
            name="training_scaling_exponent",
            value=training_exponent,
            unit="power",
            threshold_max=2.0,  # Should be sub-quadratic
        )

        result.add_metric(
            name="query_scaling_exponent",
            value=query_exponent,
            unit="power",
            threshold_max=1.5,  # Queries should be nearly linear
        )

        result.add_metric(
            name="consolidation_scaling_exponent",
            value=consolidation_exponent,
            unit="power",
            threshold_max=2.0,
        )

        # Absolute timing for largest corpus
        largest_size = max(self.corpus_sizes)
        largest_train_time = next(t for s, t in training_times if s == largest_size)
        largest_query_time = next(t for s, t in query_times if s == largest_size)

        result.add_metric(
            name=f"training_time_{largest_size}_docs",
            value=largest_train_time,
            unit="ms",
            threshold_max=10000,  # Should complete in < 10s
        )

        result.add_metric(
            name=f"query_time_{largest_size}_docs",
            value=largest_query_time,
            unit="ms",
            threshold_max=100,  # Queries should be < 100ms
        )

        result.metadata = {
            "corpus_sizes": self.corpus_sizes,
            "training_times": training_times,
            "query_times": query_times,
            "consolidation_times": consolidation_times,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.corpora = {}


class ColdStartBenchmark(BaseBenchmark):
    """
    Test cold start performance.

    Hypothesis: System should reach acceptable performance with minimal training.

    Measures:
    - Performance vs training documents
    - Time to reach 80% of asymptotic performance
    """

    name = "cold_start"
    description = "Test cold start and learning curve"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.training_corpus: List[str] = []
        self.test_queries: List[Tuple[List[str], Set[str]]] = []

    def setup(self) -> None:
        """Generate training corpus and test queries."""
        self.training_corpus = generate_synthetic_corpus(
            n_docs=200,
            doc_length=50,
            pattern_frequency=0.4,
        )

        # Test queries with known patterns from synthetic corpus
        self.test_queries = [
            (["pattern", "alpha", "test"], {"pattern", "alpha", "test"}),
            (["pattern", "beta", "verify"], {"pattern", "beta", "verify"}),
            (["concept", "neural", "network"], {"concept", "neural", "network"}),
        ]

    def run(self) -> BenchmarkResult:
        """Run cold start analysis."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        try:
            from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
            from cortical.reasoning.loom import ThinkingMode
        except ImportError as e:
            result.status = BenchmarkStatus.ERROR
            result.error_message = f"Could not import Woven Mind: {e}"
            return result

        config = WovenMindConfig()
        mind = WovenMind(config=config)

        # Track performance as training progresses
        checkpoints = [0, 5, 10, 20, 50, 100, 150, 200]
        performance_curve: List[Tuple[int, float]] = []
        mode_stability_curve: List[Tuple[int, float]] = []

        doc_idx = 0
        for checkpoint in checkpoints:
            # Train up to checkpoint
            while doc_idx < checkpoint and doc_idx < len(self.training_corpus):
                mind.train(self.training_corpus[doc_idx])
                doc_idx += 1

            # Measure activation quality
            recall_scores = []
            mode_stabilities = []

            for query_tokens, expected in self.test_queries:
                # Run query multiple times to measure stability
                modes = []
                activations_list = []

                for _ in range(5):
                    res = mind.process(query_tokens)
                    modes.append(res.mode)
                    activations_list.append(res.activations)

                # Calculate recall
                if activations_list:
                    best_activations = max(activations_list, key=len)
                    recall = len(best_activations & expected) / len(expected) if expected else 0
                    recall_scores.append(recall)

                # Calculate mode stability (fraction with same mode)
                if modes:
                    most_common = max(set(modes), key=modes.count)
                    stability = modes.count(most_common) / len(modes)
                    mode_stabilities.append(stability)

            avg_recall = statistics.mean(recall_scores) if recall_scores else 0
            avg_stability = statistics.mean(mode_stabilities) if mode_stabilities else 0

            performance_curve.append((checkpoint, avg_recall))
            mode_stability_curve.append((checkpoint, avg_stability))

        # Find time to reach 80% of final performance
        final_performance = performance_curve[-1][1] if performance_curve else 0
        target_performance = 0.8 * final_performance

        time_to_80_percent = len(self.training_corpus)  # Default: never
        for checkpoint, perf in performance_curve:
            if perf >= target_performance:
                time_to_80_percent = checkpoint
                break

        result.add_metric(
            name="time_to_80_percent_performance",
            value=time_to_80_percent,
            unit="documents",
            threshold_max=50,  # Should reach 80% within 50 docs
        )

        # Initial performance (after 10 docs)
        initial_perf = next((p for c, p in performance_curve if c == 10), 0)
        result.add_metric(
            name="initial_performance_10_docs",
            value=initial_perf,
            unit="recall",
            threshold_min=0.2,  # Should have some performance after 10 docs
        )

        # Final performance
        result.add_metric(
            name="final_performance",
            value=final_performance,
            unit="recall",
            threshold_min=0.5,
        )

        # Mode stability after minimal training
        initial_stability = next((s for c, s in mode_stability_curve if c == 10), 0)
        result.add_metric(
            name="initial_mode_stability",
            value=initial_stability,
            unit="ratio",
            threshold_min=0.6,
        )

        result.metadata = {
            "checkpoints": checkpoints,
            "performance_curve": performance_curve,
            "mode_stability_curve": mode_stability_curve,
            "target_performance": target_performance,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.training_corpus = []
        self.test_queries = []


class MemoryUsageBenchmark(BaseBenchmark):
    """
    Test memory usage patterns.

    Measures:
    - Memory growth with corpus size
    - Memory after consolidation
    - Memory leaks during extended operation
    """

    name = "memory_usage"
    description = "Test memory usage patterns"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.corpus: List[str] = []

    def setup(self) -> None:
        """Generate test corpus."""
        self.corpus = generate_synthetic_corpus(
            n_docs=500,
            doc_length=100,
        )

    def run(self) -> BenchmarkResult:
        """Run memory usage analysis."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        try:
            import sys
            from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
        except ImportError as e:
            result.status = BenchmarkStatus.ERROR
            result.error_message = f"Could not import required modules: {e}"
            return result

        def get_size(obj, seen=None) -> int:
            """Recursively calculate object size."""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

            if isinstance(obj, dict):
                size += sum(get_size(v, seen) for v in obj.values())
                size += sum(get_size(k, seen) for k in obj.keys())
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum(get_size(i, seen) for i in obj)

            return size

        config = WovenMindConfig()
        mind = WovenMind(config=config)

        # Measure initial memory
        initial_size = get_size(mind)

        # Train incrementally and track memory
        memory_checkpoints: List[Tuple[int, int]] = [(0, initial_size)]

        for i, doc in enumerate(self.corpus):
            mind.train(doc)
            tokens = doc.split()[:5]
            mind.process(tokens)

            if (i + 1) % 100 == 0:
                current_size = get_size(mind)
                memory_checkpoints.append((i + 1, current_size))

        # Final size before consolidation
        pre_consolidation_size = get_size(mind)

        # Run consolidation
        if hasattr(mind, 'consolidate'):
            mind.consolidate()

        post_consolidation_size = get_size(mind)

        # Calculate memory efficiency
        memory_per_doc = (pre_consolidation_size - initial_size) / len(self.corpus) if self.corpus else 0

        result.add_metric(
            name="memory_per_document",
            value=memory_per_doc,
            unit="bytes",
            threshold_max=10000,  # Should be < 10KB per doc
        )

        # Check for consolidation memory reduction
        consolidation_reduction = (pre_consolidation_size - post_consolidation_size) / pre_consolidation_size if pre_consolidation_size > 0 else 0

        result.add_metric(
            name="consolidation_memory_reduction",
            value=consolidation_reduction,
            unit="ratio",
            threshold_min=-0.1,  # Should not significantly increase memory
        )

        # Check for linear scaling (not exponential)
        if len(memory_checkpoints) >= 2:
            sizes = [s for _, s in memory_checkpoints]
            # Calculate ratio of last to first
            growth_ratio = sizes[-1] / sizes[0] if sizes[0] > 0 else float('inf')
            doc_ratio = memory_checkpoints[-1][0] / memory_checkpoints[0][0] if memory_checkpoints[0][0] > 0 else len(self.corpus)

            # Should grow roughly linearly
            growth_exponent = growth_ratio / doc_ratio if doc_ratio > 0 else growth_ratio

            result.add_metric(
                name="memory_growth_linearity",
                value=growth_exponent,
                unit="ratio",
                threshold_max=2.0,  # Should not grow faster than 2x linear
            )

        result.metadata = {
            "corpus_size": len(self.corpus),
            "initial_size_bytes": initial_size,
            "pre_consolidation_bytes": pre_consolidation_size,
            "post_consolidation_bytes": post_consolidation_size,
            "memory_checkpoints": memory_checkpoints,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.corpus = []
