"""
Stability benchmarks for Woven Mind.

These benchmarks test system stability under parameter variations and
extended operation, addressing concerns about:
- Parameter sensitivity
- Baseline drift
- Homeostasis stability
"""

from typing import Any, Dict, List, Optional, Tuple
import statistics

from .base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
    generate_synthetic_corpus,
    measure_stability,
)


class ParameterSensitivityBenchmark(BaseBenchmark):
    """
    Test sensitivity to parameter changes.

    Hypothesis: Small parameter changes should not cause disproportionately
    large behavioral changes.

    Measures:
    - Mode switching stability under surprise_threshold variations
    - Abstraction count stability under min_frequency variations
    - Activation stability under k_winners variations
    """

    name = "parameter_sensitivity"
    description = "Test behavioral stability across parameter ranges"
    category = BenchmarkCategory.STABILITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.corpus: List[str] = []
        self.test_inputs: List[List[str]] = []

    def setup(self) -> None:
        """Generate test corpus and inputs."""
        self.corpus = generate_synthetic_corpus(
            n_docs=self.config.get("n_docs", 100),
            doc_length=self.config.get("doc_length", 50),
            pattern_frequency=0.3,
        )

        # Create test inputs (mix of familiar and novel)
        self.test_inputs = [
            ["pattern", "alpha", "test"],      # Should be familiar
            ["pattern", "beta", "verify"],     # Should be familiar
            ["completely", "novel", "input"],  # Should be surprising
            ["word1", "word2", "word3"],       # Random from vocab
        ]

    def run(self) -> BenchmarkResult:
        """Run parameter sensitivity analysis."""
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

        # Test 1: Surprise threshold sensitivity
        surprise_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        mode_ratios = []

        for threshold in surprise_thresholds:
            config = WovenMindConfig(surprise_threshold=threshold)
            mind = WovenMind(config=config)

            # Train on corpus
            for doc in self.corpus[:50]:
                mind.train(doc)

            # Count mode selections
            slow_count = 0
            total = 0
            for tokens in self.test_inputs * 5:  # Multiple runs
                res = mind.process(tokens)
                if res.mode == ThinkingMode.SLOW:
                    slow_count += 1
                total += 1

            mode_ratios.append(slow_count / total if total > 0 else 0)

        # Calculate sensitivity (how much does output change per unit input change)
        # Lower is more stable
        if len(mode_ratios) > 1:
            deltas = [abs(mode_ratios[i+1] - mode_ratios[i]) for i in range(len(mode_ratios)-1)]
            input_deltas = [surprise_thresholds[i+1] - surprise_thresholds[i] for i in range(len(surprise_thresholds)-1)]
            sensitivities = [d/id if id > 0 else 0 for d, id in zip(deltas, input_deltas)]
            avg_sensitivity = statistics.mean(sensitivities)
        else:
            avg_sensitivity = 0.0

        result.add_metric(
            name="surprise_threshold_sensitivity",
            value=avg_sensitivity,
            unit="ratio_change/threshold_change",
            threshold_max=5.0,  # Sensitivity > 5 is concerning
        )

        # Test 2: k_winners sensitivity
        k_values = [3, 5, 7, 10, 15]
        activation_counts = []

        for k in k_values:
            config = WovenMindConfig(k_winners=k)
            mind = WovenMind(config=config)

            for doc in self.corpus[:50]:
                mind.train(doc)

            counts = []
            for tokens in self.test_inputs * 3:
                res = mind.process(tokens)
                counts.append(len(res.activations))

            activation_counts.append(statistics.mean(counts) if counts else 0)

        # Check monotonicity (more k_winners should mean more activations)
        monotonic_violations = sum(
            1 for i in range(len(activation_counts) - 1)
            if activation_counts[i+1] < activation_counts[i]
        )

        result.add_metric(
            name="k_winners_monotonicity_violations",
            value=monotonic_violations,
            unit="count",
            threshold_max=1,  # At most 1 violation due to noise
        )

        # Test 3: min_frequency sensitivity
        min_freq_values = [2, 3, 4, 5]
        abstraction_counts = []

        for min_freq in min_freq_values:
            config = WovenMindConfig(min_frequency=min_freq)
            mind = WovenMind(config=config)

            for doc in self.corpus:
                mind.train(doc)
                tokens = doc.split()[:5]
                mind.observe_pattern(tokens)

            # Count abstractions
            abs_count = len(mind.cortex.get_abstractions()) if hasattr(mind.cortex, 'get_abstractions') else 0
            abstraction_counts.append(abs_count)

        # Higher min_frequency should mean fewer abstractions
        anti_monotonic_violations = sum(
            1 for i in range(len(abstraction_counts) - 1)
            if abstraction_counts[i+1] > abstraction_counts[i]
        )

        result.add_metric(
            name="min_frequency_anti_monotonicity_violations",
            value=anti_monotonic_violations,
            unit="count",
            threshold_max=1,
        )

        # Overall stability score (0-1, higher is more stable)
        stability_score = 1.0
        if avg_sensitivity > 5.0:
            stability_score -= 0.3
        if monotonic_violations > 1:
            stability_score -= 0.3
        if anti_monotonic_violations > 1:
            stability_score -= 0.3

        result.add_metric(
            name="overall_stability_score",
            value=max(0, stability_score),
            unit="score",
            threshold_min=0.7,
        )

        result.metadata = {
            "surprise_thresholds_tested": surprise_thresholds,
            "mode_ratios": mode_ratios,
            "k_values_tested": k_values,
            "activation_counts": activation_counts,
            "min_freq_tested": min_freq_values,
            "abstraction_counts": abstraction_counts,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.corpus = []
        self.test_inputs = []


class BaselineDriftBenchmark(BaseBenchmark):
    """
    Test for surprise baseline drift under various conditions.

    Hypothesis: The adaptive baseline should stabilize in consistent
    environments and adapt appropriately to domain shifts.

    Measures:
    - Baseline convergence rate
    - Baseline stability in steady state
    - Baseline response to domain shift
    """

    name = "baseline_drift"
    description = "Test surprise baseline behavior over extended operation"
    category = BenchmarkCategory.STABILITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.domain_a_corpus: List[str] = []
        self.domain_b_corpus: List[str] = []

    def setup(self) -> None:
        """Generate domain-specific corpora."""
        # Domain A: Technical vocabulary
        self.domain_a_corpus = [
            "neural network architecture deep learning",
            "gradient descent optimization algorithm",
            "convolutional neural network image recognition",
            "recurrent neural network sequence modeling",
            "transformer attention mechanism nlp",
        ] * 20

        # Domain B: Different vocabulary
        self.domain_b_corpus = [
            "financial market stock trading investment",
            "portfolio risk management hedge fund",
            "bond yield interest rate economics",
            "derivatives futures options trading",
            "quantitative analysis financial modeling",
        ] * 20

    def run(self) -> BenchmarkResult:
        """Run baseline drift analysis."""
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

        config = WovenMindConfig()
        mind = WovenMind(config=config)

        baseline_history = []

        # Phase 1: Train on Domain A and track baseline convergence
        for doc in self.domain_a_corpus:
            mind.train(doc)
            tokens = doc.split()
            mind.process(tokens)

            baseline = mind.get_surprise_baseline() if hasattr(mind, 'get_surprise_baseline') else 0.0
            baseline_history.append(("domain_a", baseline))

        # Calculate convergence rate (how quickly baseline stabilizes)
        domain_a_baselines = [b for domain, b in baseline_history if domain == "domain_a"]
        if len(domain_a_baselines) > 10:
            early_variance = statistics.variance(domain_a_baselines[:10])
            late_variance = statistics.variance(domain_a_baselines[-10:])
            convergence_ratio = late_variance / early_variance if early_variance > 0 else 1.0
        else:
            convergence_ratio = 1.0

        result.add_metric(
            name="convergence_ratio",
            value=convergence_ratio,
            unit="late_var/early_var",
            threshold_max=0.5,  # Late variance should be < 50% of early
        )

        # Phase 2: Inject Domain B (domain shift)
        pre_shift_baseline = domain_a_baselines[-1] if domain_a_baselines else 0.0

        for doc in self.domain_b_corpus[:20]:
            mind.train(doc)
            tokens = doc.split()
            mind.process(tokens)

            baseline = mind.get_surprise_baseline() if hasattr(mind, 'get_surprise_baseline') else 0.0
            baseline_history.append(("domain_b", baseline))

        domain_b_baselines = [b for domain, b in baseline_history if domain == "domain_b"]

        # Calculate adaptation: baseline should rise then stabilize
        if domain_b_baselines:
            peak_baseline = max(domain_b_baselines)
            final_baseline = domain_b_baselines[-1]

            # Peak should be higher than pre-shift (novelty detected)
            novelty_detection = peak_baseline - pre_shift_baseline

            # Final should be lower than peak (adaptation occurred)
            adaptation_amount = peak_baseline - final_baseline if peak_baseline > final_baseline else 0

            result.add_metric(
                name="novelty_detection_magnitude",
                value=novelty_detection,
                unit="baseline_delta",
                threshold_min=0.05,  # Should detect some novelty
            )

            result.add_metric(
                name="adaptation_amount",
                value=adaptation_amount,
                unit="baseline_delta",
                threshold_min=0.0,  # Should show some adaptation
            )

        # Phase 3: Return to Domain A (should re-stabilize faster)
        for doc in self.domain_a_corpus[:10]:
            mind.train(doc)
            tokens = doc.split()
            mind.process(tokens)

            baseline = mind.get_surprise_baseline() if hasattr(mind, 'get_surprise_baseline') else 0.0
            baseline_history.append(("domain_a_return", baseline))

        return_baselines = [b for domain, b in baseline_history if domain == "domain_a_return"]

        if return_baselines and domain_a_baselines:
            # Should return toward original domain baseline
            original_baseline = statistics.mean(domain_a_baselines[-5:])
            return_baseline = statistics.mean(return_baselines[-3:]) if len(return_baselines) >= 3 else return_baselines[-1]
            return_error = abs(return_baseline - original_baseline)

            result.add_metric(
                name="return_to_original_error",
                value=return_error,
                unit="baseline_delta",
                threshold_max=0.3,  # Should be close to original
            )

        # Check for runaway drift (baseline never stabilizes)
        all_baselines = [b for _, b in baseline_history]
        if all_baselines:
            overall_trend = all_baselines[-1] - all_baselines[0]
            result.add_metric(
                name="overall_drift",
                value=abs(overall_trend),
                unit="baseline_delta",
                threshold_max=1.0,  # Should not drift too far
            )

        result.metadata = {
            "total_observations": len(baseline_history),
            "domain_a_count": len(domain_a_baselines),
            "domain_b_count": len(domain_b_baselines),
            "baseline_history_sample": baseline_history[::10],  # Every 10th
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.domain_a_corpus = []
        self.domain_b_corpus = []


class HomeostasisStabilityBenchmark(BaseBenchmark):
    """
    Test homeostatic regulation stability.

    Hypothesis: Homeostasis should maintain stable activation levels
    without oscillation or drift.

    Measures:
    - Activation variance over time
    - Convergence to target activation
    - Recovery from perturbation
    """

    name = "homeostasis_stability"
    description = "Test homeostatic regulation stability"
    category = BenchmarkCategory.STABILITY

    def setup(self) -> None:
        """Generate test data."""
        self.normal_inputs = generate_synthetic_corpus(
            n_docs=50,
            doc_length=20,
        )

        # High-activation inputs (many overlapping terms)
        self.high_activation_inputs = [
            "word1 word2 word3 word4 word5",
            "word1 word2 word3 word6 word7",
            "word1 word2 word8 word9 word10",
        ] * 20

        # Low-activation inputs (sparse, unique terms)
        self.low_activation_inputs = [
            f"unique{i}a unique{i}b unique{i}c"
            for i in range(50)
        ]

    def run(self) -> BenchmarkResult:
        """Run homeostasis stability analysis."""
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

        config = WovenMindConfig()
        mind = WovenMind(config=config)

        activation_history = []

        # Phase 1: Normal operation
        for doc in self.normal_inputs:
            mind.train(doc)
            tokens = doc.split()[:5]
            res = mind.process(tokens)
            activation_history.append(("normal", len(res.activations)))

        normal_activations = [a for phase, a in activation_history if phase == "normal"]
        normal_mean = statistics.mean(normal_activations) if normal_activations else 0
        normal_std = statistics.stdev(normal_activations) if len(normal_activations) > 1 else 0

        result.add_metric(
            name="normal_activation_mean",
            value=normal_mean,
            unit="count",
        )

        result.add_metric(
            name="normal_activation_cv",
            value=normal_std / normal_mean if normal_mean > 0 else 0,
            unit="coefficient_of_variation",
            threshold_max=0.5,  # CV should be < 50%
        )

        # Phase 2: High activation stress
        for doc in self.high_activation_inputs[:20]:
            mind.train(doc)
            tokens = doc.split()
            res = mind.process(tokens)
            activation_history.append(("high_stress", len(res.activations)))

        high_activations = [a for phase, a in activation_history if phase == "high_stress"]

        # Homeostasis should prevent runaway activation
        if high_activations:
            high_mean = statistics.mean(high_activations)
            activation_ratio = high_mean / normal_mean if normal_mean > 0 else float('inf')

            result.add_metric(
                name="high_stress_activation_ratio",
                value=activation_ratio,
                unit="high/normal",
                threshold_max=3.0,  # Should not exceed 3x normal
            )

        # Phase 3: Recovery to normal
        for doc in self.normal_inputs[:20]:
            tokens = doc.split()[:5]
            res = mind.process(tokens)
            activation_history.append(("recovery", len(res.activations)))

        recovery_activations = [a for phase, a in activation_history if phase == "recovery"]

        if recovery_activations and normal_mean > 0:
            recovery_mean = statistics.mean(recovery_activations[-5:])
            recovery_error = abs(recovery_mean - normal_mean) / normal_mean

            result.add_metric(
                name="recovery_error",
                value=recovery_error,
                unit="relative_error",
                threshold_max=0.3,  # Should recover to within 30%
            )

        result.metadata = {
            "phases": ["normal", "high_stress", "recovery"],
            "total_observations": len(activation_history),
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.normal_inputs = []
        self.high_activation_inputs = []
        self.low_activation_inputs = []
