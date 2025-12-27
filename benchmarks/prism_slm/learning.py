"""
Hebbian Learning Benchmarks for PRISM-SLM.

Tests synaptic learning dynamics, decay stability, and reward learning.
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


class HebbianStrengtheningBenchmark(BaseBenchmark):
    """
    Benchmark for Hebbian learning dynamics.

    Tests that:
    - Repeated observations strengthen transitions
    - Weight growth is proportional to frequency
    - Saturation doesn't occur prematurely
    """

    name = "hebbian_strengthening"
    description = "Verifies Hebbian learning strengthens transitions correctly"
    category = BenchmarkCategory.STABILITY

    def __init__(self, quick: bool = False):
        super().__init__(quick)
        self.repetitions = 10 if quick else 50

    def setup(self) -> None:
        """Initialize fresh model."""
        self.model = PRISMLanguageModel(context_size=2)

    def run(self) -> BenchmarkResult:
        """Run Hebbian strengthening benchmark."""
        start_time = time.time()

        # Track weight growth over repetitions
        phrase = "the cat sat"
        weights_over_time = []

        for i in range(self.repetitions):
            self.model.train(phrase)

            # Check weight of "the" -> "cat" transition
            transitions = self.model.graph.get_transitions(("the",))
            cat_weight = next(
                (t.weight for t in transitions if t.to_token == "cat"),
                0.0
            )
            weights_over_time.append(cat_weight)

        duration_ms = (time.time() - start_time) * 1000

        # Analyze growth pattern
        initial_weight = weights_over_time[0]
        final_weight = weights_over_time[-1]
        growth_factor = final_weight / initial_weight if initial_weight > 0 else 0

        # Check monotonicity (weights should only increase)
        is_monotonic = all(
            weights_over_time[i] <= weights_over_time[i+1]
            for i in range(len(weights_over_time) - 1)
        )

        # Check for reasonable growth rate
        expected_growth = self.repetitions * 0.1  # ~0.1 per observation
        growth_efficiency = (final_weight - initial_weight) / expected_growth

        metrics = [
            BenchmarkMetric(
                name="initial_weight",
                value=initial_weight,
                unit="weight",
            ),
            BenchmarkMetric(
                name="final_weight",
                value=final_weight,
                unit="weight",
            ),
            BenchmarkMetric(
                name="growth_factor",
                value=growth_factor,
                unit="ratio",
                threshold_min=2.0,  # Should at least double
            ),
            BenchmarkMetric(
                name="is_monotonic",
                value=1.0 if is_monotonic else 0.0,
                unit="bool",
                threshold_min=1.0,  # Must be monotonic
            ),
            BenchmarkMetric(
                name="growth_efficiency",
                value=growth_efficiency,
                unit="ratio",
                threshold_min=0.5,  # At least 50% of expected growth
                threshold_max=2.0,  # Not more than 200%
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
                "repetitions": self.repetitions,
                "phrase": phrase,
            },
        )

    def teardown(self) -> None:
        self.model = None


class DecayStabilityBenchmark(BaseBenchmark):
    """
    Benchmark for synaptic decay stability.

    Tests that:
    - Decay reduces weights proportionally
    - Multiple decay cycles don't cause numerical issues
    - Strong connections survive longer than weak ones
    """

    name = "decay_stability"
    description = "Verifies synaptic decay is stable and proportional"
    category = BenchmarkCategory.STABILITY

    def __init__(self, quick: bool = False):
        super().__init__(quick)
        self.decay_cycles = 10 if quick else 50

    def setup(self) -> None:
        """Initialize and train model."""
        self.model = PRISMLanguageModel(context_size=2)

        # Create transitions with different strengths
        for _ in range(10):
            self.model.train("strong connection here")
        for _ in range(3):
            self.model.train("weak link exists")

    def run(self) -> BenchmarkResult:
        """Run decay stability benchmark."""
        start_time = time.time()

        decay_factor = 0.9

        # Get initial weights
        strong_trans = self.model.graph.get_transitions(("strong",))
        weak_trans = self.model.graph.get_transitions(("weak",))

        initial_strong = next(
            (t.weight for t in strong_trans if t.to_token == "connection"),
            0.0
        )
        initial_weak = next(
            (t.weight for t in weak_trans if t.to_token == "link"),
            0.0
        )

        # Apply decay cycles
        weights_strong = [initial_strong]
        weights_weak = [initial_weak]

        for _ in range(self.decay_cycles):
            self.model.apply_decay(factor=decay_factor)

            strong_trans = self.model.graph.get_transitions(("strong",))
            weak_trans = self.model.graph.get_transitions(("weak",))

            current_strong = next(
                (t.weight for t in strong_trans if t.to_token == "connection"),
                0.0
            )
            current_weak = next(
                (t.weight for t in weak_trans if t.to_token == "link"),
                0.0
            )

            weights_strong.append(current_strong)
            weights_weak.append(current_weak)

        duration_ms = (time.time() - start_time) * 1000

        final_strong = weights_strong[-1]
        final_weak = weights_weak[-1]

        # Calculate expected final weights
        expected_strong = initial_strong * (decay_factor ** self.decay_cycles)
        expected_weak = initial_weak * (decay_factor ** self.decay_cycles)

        # Check accuracy of decay
        strong_accuracy = (
            final_strong / expected_strong
            if expected_strong > 0 else 1.0
        )
        weak_accuracy = (
            final_weak / expected_weak
            if expected_weak > 0 else 1.0
        )

        # Check that strong > weak is preserved
        ratio_preserved = (
            (final_strong > final_weak)
            if (initial_strong > initial_weak) else True
        )

        # Check for numerical stability (no NaN, Inf, or negative)
        all_positive = all(w >= 0 for w in weights_strong + weights_weak)

        metrics = [
            BenchmarkMetric(
                name="strong_decay_accuracy",
                value=strong_accuracy,
                unit="ratio",
                threshold_min=0.95,  # Within 5% of expected
                threshold_max=1.05,
            ),
            BenchmarkMetric(
                name="weak_decay_accuracy",
                value=weak_accuracy,
                unit="ratio",
                threshold_min=0.95,
                threshold_max=1.05,
            ),
            BenchmarkMetric(
                name="ratio_preserved",
                value=1.0 if ratio_preserved else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="numerical_stability",
                value=1.0 if all_positive else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="final_strong_weight",
                value=final_strong,
                unit="weight",
            ),
            BenchmarkMetric(
                name="final_weak_weight",
                value=final_weak,
                unit="weight",
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
                "decay_factor": decay_factor,
                "decay_cycles": self.decay_cycles,
            },
        )

    def teardown(self) -> None:
        self.model = None


class RewardLearningBenchmark(BaseBenchmark):
    """
    Benchmark for reward-based learning.

    Tests that:
    - Positive rewards strengthen paths
    - Negative rewards weaken paths (but don't go negative)
    - Reward effects are proportional
    """

    name = "reward_learning"
    description = "Verifies reward-based path strengthening works correctly"
    category = BenchmarkCategory.QUALITY

    def __init__(self, quick: bool = False):
        super().__init__(quick)

    def setup(self) -> None:
        """Initialize and train model."""
        self.model = PRISMLanguageModel(context_size=2)

        # Train baseline
        for _ in range(5):
            self.model.train("good path leads here")
            self.model.train("bad path leads nowhere")

    def run(self) -> BenchmarkResult:
        """Run reward learning benchmark."""
        start_time = time.time()

        # Get initial weights
        good_trans = self.model.graph.get_transitions(("good",))
        bad_trans = self.model.graph.get_transitions(("bad",))

        initial_good = next(
            (t.weight for t in good_trans if t.to_token == "path"),
            0.0
        )
        initial_bad = next(
            (t.weight for t in bad_trans if t.to_token == "path"),
            0.0
        )

        # Apply positive reward to good path
        self.model.reward_path(["good", "path", "leads", "here"], reward=5.0)

        # Apply negative reward to bad path
        self.model.reward_path(["bad", "path", "leads", "nowhere"], reward=-2.0)

        # Get final weights
        good_trans = self.model.graph.get_transitions(("good",))
        bad_trans = self.model.graph.get_transitions(("bad",))

        final_good = next(
            (t.weight for t in good_trans if t.to_token == "path"),
            0.0
        )
        final_bad = next(
            (t.weight for t in bad_trans if t.to_token == "path"),
            0.0
        )

        duration_ms = (time.time() - start_time) * 1000

        # Calculate changes
        good_change = final_good - initial_good
        bad_change = final_bad - initial_bad

        # Verify positive reward increased weight
        positive_worked = good_change > 0

        # Verify negative reward decreased weight (or clamped to minimum)
        negative_worked = bad_change <= 0

        # Verify weights stay non-negative
        weights_positive = final_good >= 0 and final_bad >= 0

        metrics = [
            BenchmarkMetric(
                name="initial_good_weight",
                value=initial_good,
                unit="weight",
            ),
            BenchmarkMetric(
                name="final_good_weight",
                value=final_good,
                unit="weight",
            ),
            BenchmarkMetric(
                name="good_weight_change",
                value=good_change,
                unit="delta",
                threshold_min=0.1,  # Should increase
            ),
            BenchmarkMetric(
                name="bad_weight_change",
                value=bad_change,
                unit="delta",
                threshold_max=0.0,  # Should decrease or stay same
            ),
            BenchmarkMetric(
                name="positive_reward_effective",
                value=1.0 if positive_worked else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="negative_reward_effective",
                value=1.0 if negative_worked else 0.0,
                unit="bool",
                threshold_min=1.0,
            ),
            BenchmarkMetric(
                name="weights_non_negative",
                value=1.0 if weights_positive else 0.0,
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
        )

    def teardown(self) -> None:
        self.model = None
