"""
Integration Benchmarks for PRISM-SLM with Woven Mind.

Tests spreading activation, lateral inhibition, and sparse coding efficiency.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import statistics
import time

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkStatus,
)
from cortical.reasoning import PRISMLanguageModel


class SpreadingActivationBenchmark(BaseBenchmark):
    """
    Benchmark for spreading activation performance.

    Tests that:
    - Activation spreads to connected tokens
    - Spread decays with distance
    - Performance scales reasonably
    """

    name = "spreading_activation"
    description = "Verifies spreading activation reaches connected concepts"
    category = BenchmarkCategory.COGNITIVE

    def __init__(self, quick: bool = False):
        super().__init__(quick)

    def setup(self) -> None:
        """Initialize and train model."""
        self.model = PRISMLanguageModel(context_size=3)

        # Create a known graph structure
        training = [
            "neural networks learn patterns",
            "deep neural networks process data",
            "machine learning uses neural models",
            "patterns emerge from data",
            "learning algorithms improve performance",
        ]

        for text in training:
            for _ in range(5):
                self.model.train(text)

    def run(self) -> BenchmarkResult:
        """Run spreading activation benchmark."""
        start_time = time.time()

        # Test spreading from "neural"
        activations = self.model.graph.spreading_activation(
            seed_tokens={"neural": 1.0},
            spread_factor=0.5,
            decay_per_step=0.7,
            max_steps=3,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Analyze results
        seed_activation = activations.get("neural", 0.0)
        direct_neighbors = ["networks", "models"]
        indirect_neighbors = ["patterns", "data", "learn"]

        # Check direct neighbors got activation
        direct_activations = [
            activations.get(n, 0.0) for n in direct_neighbors
        ]
        avg_direct = statistics.mean(direct_activations) if direct_activations else 0

        # Check indirect neighbors got (lower) activation
        indirect_activations = [
            activations.get(n, 0.0) for n in indirect_neighbors
        ]
        avg_indirect = statistics.mean(indirect_activations) if indirect_activations else 0

        # Verify decay with distance
        distance_decay_correct = avg_direct > avg_indirect or avg_indirect == 0

        # Check total spread
        total_activated = len([v for v in activations.values() if v > 0.01])

        metrics = [
            BenchmarkMetric(
                name="seed_activation",
                value=seed_activation,
                unit="activation",
                threshold_min=0.99,  # Seed should stay at 1.0
            ),
            BenchmarkMetric(
                name="avg_direct_neighbor_activation",
                value=avg_direct,
                unit="activation",
                threshold_min=0.1,  # Should have some activation
            ),
            BenchmarkMetric(
                name="avg_indirect_neighbor_activation",
                value=avg_indirect,
                unit="activation",
            ),
            BenchmarkMetric(
                name="distance_decay_correct",
                value=1.0 if distance_decay_correct else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="total_tokens_activated",
                value=total_activated,
                unit="count",
                threshold_min=3,  # At least seed + some neighbors
            ),
            BenchmarkMetric(
                name="spread_duration_ms",
                value=duration_ms,
                unit="ms",
                threshold_max=1000,  # Should be fast
            ),
        ]

        all_passing = all(m.check_thresholds() for m in metrics)

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.PASSED if all_passing else BenchmarkStatus.FAILED,
            metrics=metrics,
            duration_ms=duration_ms,
            metadata={
                "activations": {k: v for k, v in sorted(
                    activations.items(),
                    key=lambda x: -x[1]
                )[:10]},
            },
        )

    def teardown(self) -> None:
        self.model = None


class LateralInhibitionBenchmark(BaseBenchmark):
    """
    Benchmark for lateral inhibition effectiveness.

    Tests that:
    - Strong activations inhibit weak ones
    - Sparsity is achieved
    - Strongest activation survives
    """

    name = "lateral_inhibition"
    description = "Verifies lateral inhibition produces sparse activations"
    category = BenchmarkCategory.COGNITIVE

    def __init__(self, quick: bool = False):
        super().__init__(quick)

    def setup(self) -> None:
        """Initialize model."""
        self.model = PRISMLanguageModel(context_size=2)

    def run(self) -> BenchmarkResult:
        """Run lateral inhibition benchmark."""
        start_time = time.time()

        # Create test activation pattern
        raw_activations = {
            "strong": 0.9,
            "medium_high": 0.7,
            "medium": 0.5,
            "weak": 0.3,
            "very_weak": 0.1,
        }

        # Apply lateral inhibition
        inhibited = self.model.graph.lateral_inhibition(
            raw_activations,
            inhibition_radius=2,
            inhibition_strength=0.5,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Analyze results
        raw_active = sum(1 for v in raw_activations.values() if v > 0.1)
        inhibited_active = sum(1 for v in inhibited.values() if v > 0.1)

        # Sparsity improvement
        sparsity_before = 1 - (raw_active / len(raw_activations))
        sparsity_after = 1 - (inhibited_active / len(inhibited))
        sparsity_improvement = sparsity_after - sparsity_before

        # Check that strongest survived
        strongest_before = max(raw_activations.items(), key=lambda x: x[1])
        strongest_after = max(inhibited.items(), key=lambda x: x[1])
        strongest_preserved = strongest_before[0] == strongest_after[0]

        # Check that weak got weaker
        weak_got_weaker = inhibited.get("weak", 0) < raw_activations.get("weak", 0)

        metrics = [
            BenchmarkMetric(
                name="active_before_inhibition",
                value=raw_active,
                unit="count",
            ),
            BenchmarkMetric(
                name="active_after_inhibition",
                value=inhibited_active,
                unit="count",
            ),
            BenchmarkMetric(
                name="sparsity_improvement",
                value=sparsity_improvement,
                unit="ratio",
                threshold_min=0.0,  # Should improve or stay same
            ),
            BenchmarkMetric(
                name="strongest_preserved",
                value=1.0 if strongest_preserved else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="weak_reduced",
                value=1.0 if weak_got_weaker else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
        ]

        all_passing = all(m.check_thresholds() for m in metrics)

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.PASSED if all_passing else BenchmarkStatus.FAILED,
            metrics=metrics,
            duration_ms=duration_ms,
            metadata={
                "raw": raw_activations,
                "inhibited": inhibited,
            },
        )

    def teardown(self) -> None:
        self.model = None


class SparsecodingEfficiencyBenchmark(BaseBenchmark):
    """
    Benchmark for sparse coding efficiency.

    Tests k-winners-take-all:
    - Exactly k winners survive
    - Winners are the strongest k
    - Losers are set to zero
    """

    name = "sparsecoding_efficiency"
    description = "Verifies k-winners-take-all produces correct sparsity"
    category = BenchmarkCategory.COGNITIVE

    def __init__(self, quick: bool = False):
        super().__init__(quick)

    def setup(self) -> None:
        """Initialize model."""
        self.model = PRISMLanguageModel(context_size=2)

    def run(self) -> BenchmarkResult:
        """Run sparse coding benchmark."""
        start_time = time.time()

        # Test with various k values
        activations = {
            f"token_{i}": 1.0 - (i * 0.1)
            for i in range(10)
        }  # token_0=1.0, token_1=0.9, ... token_9=0.1

        results_by_k = {}
        for k in [1, 3, 5]:
            winners = self.model.graph.k_winners_take_all(
                activations,
                k=k,
                min_activation=0.05,
            )
            active = [t for t, v in winners.items() if v > 0]
            results_by_k[k] = {
                "count": len(active),
                "winners": active,
            }

        duration_ms = (time.time() - start_time) * 1000

        # Verify k=1 gives exactly 1 winner (the strongest)
        k1_correct = (
            results_by_k[1]["count"] == 1 and
            "token_0" in results_by_k[1]["winners"]
        )

        # Verify k=3 gives exactly 3 winners (top 3)
        k3_correct = (
            results_by_k[3]["count"] == 3 and
            all(f"token_{i}" in results_by_k[3]["winners"] for i in range(3))
        )

        # Verify k=5 gives exactly 5 winners
        k5_correct = results_by_k[5]["count"] == 5

        metrics = [
            BenchmarkMetric(
                name="k1_winner_count",
                value=results_by_k[1]["count"],
                unit="count",
                threshold_min=1,
                threshold_max=1,
            ),
            BenchmarkMetric(
                name="k3_winner_count",
                value=results_by_k[3]["count"],
                unit="count",
                threshold_min=3,
                threshold_max=3,
            ),
            BenchmarkMetric(
                name="k5_winner_count",
                value=results_by_k[5]["count"],
                unit="count",
                threshold_min=5,
                threshold_max=5,
            ),
            BenchmarkMetric(
                name="k1_correct_winner",
                value=1.0 if k1_correct else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="k3_correct_winners",
                value=1.0 if k3_correct else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
        ]

        all_passing = all(m.check_thresholds() for m in metrics)

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.PASSED if all_passing else BenchmarkStatus.FAILED,
            metrics=metrics,
            duration_ms=duration_ms,
            metadata=results_by_k,
        )

    def teardown(self) -> None:
        self.model = None
