"""
Cognitive benchmarks for Woven Mind.

These benchmarks validate the cognitive architecture's core behaviors:
- Surprise calibration
- Homeostasis-surprise interaction
- Dual-process coherence
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import statistics
import random

from .base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
    generate_synthetic_corpus,
)


class SurpriseCalibrationBenchmark(BaseBenchmark):
    """
    Test calibration of surprise detection.

    Hypothesis: Surprise magnitude should correlate with actual novelty.
    High surprise should indicate genuinely novel inputs.

    Measures:
    - Surprise-novelty correlation
    - Calibration error
    - False positive/negative rates
    """

    name = "surprise_calibration"
    description = "Test calibration of surprise detection mechanism"
    category = BenchmarkCategory.COGNITIVE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.training_corpus: List[str] = []
        self.test_cases: List[Tuple[List[str], float]] = []  # (tokens, expected_novelty)

    def setup(self) -> None:
        """Generate training corpus and labeled test cases."""
        # Training corpus establishes baseline patterns
        self.training_corpus = [
            "neural network deep learning model",
            "machine learning algorithm training",
            "gradient descent optimization loss",
            "convolutional layer feature extraction",
            "recurrent network sequence processing",
        ] * 20  # Repeat to establish strong patterns

        # Test cases with known novelty levels
        self.test_cases = [
            # Very familiar (novelty = 0.0)
            (["neural", "network", "deep"], 0.0),
            (["machine", "learning", "algorithm"], 0.0),
            (["gradient", "descent", "optimization"], 0.0),

            # Somewhat familiar (novelty = 0.3)
            (["neural", "network", "architecture"], 0.3),
            (["learning", "rate", "schedule"], 0.3),

            # Mixed (novelty = 0.5)
            (["neural", "quantum", "computing"], 0.5),
            (["learning", "blockchain", "data"], 0.5),

            # Mostly novel (novelty = 0.7)
            (["quantum", "entanglement", "network"], 0.7),
            (["cryptocurrency", "market", "learning"], 0.7),

            # Completely novel (novelty = 1.0)
            (["xylophone", "giraffe", "umbrella"], 1.0),
            (["quantum", "cryptocurrency", "genomics"], 1.0),
        ]

    def run(self) -> BenchmarkResult:
        """Run surprise calibration analysis."""
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

        # Train on corpus
        for doc in self.training_corpus:
            mind.train(doc)
            tokens = doc.split()
            mind.process(tokens)

        # Test each case and collect surprise values
        results_data: List[Tuple[float, float]] = []  # (expected_novelty, actual_surprise)

        for tokens, expected_novelty in self.test_cases:
            res = mind.process(tokens)
            actual_surprise = res.surprise.magnitude if res.surprise else 0.0
            results_data.append((expected_novelty, actual_surprise))

        # Calculate Pearson correlation
        expected = [e for e, _ in results_data]
        actual = [a for _, a in results_data]

        n = len(results_data)
        if n > 1:
            mean_e = statistics.mean(expected)
            mean_a = statistics.mean(actual)

            numerator = sum((e - mean_e) * (a - mean_a) for e, a in results_data)
            denom_e = sum((e - mean_e) ** 2 for e in expected) ** 0.5
            denom_a = sum((a - mean_a) ** 2 for a in actual) ** 0.5

            if denom_e * denom_a > 0:
                correlation = numerator / (denom_e * denom_a)
            else:
                correlation = 0.0
        else:
            correlation = 0.0

        result.add_metric(
            name="surprise_novelty_correlation",
            value=correlation,
            unit="pearson_r",
            threshold_min=0.5,  # Should have positive correlation
        )

        # Calculate calibration error (average |expected - actual|)
        calibration_errors = [abs(e - a) for e, a in results_data]
        mean_calibration_error = statistics.mean(calibration_errors) if calibration_errors else 0.0

        result.add_metric(
            name="mean_calibration_error",
            value=mean_calibration_error,
            unit="error",
            threshold_max=0.4,  # Should be well-calibrated
        )

        # Calculate discrimination: can it tell familiar from novel?
        familiar_surprises = [a for e, a in results_data if e <= 0.3]
        novel_surprises = [a for e, a in results_data if e >= 0.7]

        if familiar_surprises and novel_surprises:
            familiar_mean = statistics.mean(familiar_surprises)
            novel_mean = statistics.mean(novel_surprises)
            discrimination = novel_mean - familiar_mean
        else:
            discrimination = 0.0

        result.add_metric(
            name="novel_familiar_discrimination",
            value=discrimination,
            unit="surprise_delta",
            threshold_min=0.2,  # Novel should be at least 0.2 higher
        )

        # Calculate AUC-like metric: probability that novel > familiar
        correct_orderings = 0
        total_comparisons = 0
        for f_surprise in familiar_surprises:
            for n_surprise in novel_surprises:
                total_comparisons += 1
                if n_surprise > f_surprise:
                    correct_orderings += 1
                elif n_surprise == f_surprise:
                    correct_orderings += 0.5

        auc = correct_orderings / total_comparisons if total_comparisons > 0 else 0.5

        result.add_metric(
            name="discrimination_auc",
            value=auc,
            unit="probability",
            threshold_min=0.7,  # Should correctly order 70%+ of pairs
        )

        result.metadata = {
            "test_cases": len(self.test_cases),
            "results": results_data,
            "familiar_mean": statistics.mean(familiar_surprises) if familiar_surprises else None,
            "novel_mean": statistics.mean(novel_surprises) if novel_surprises else None,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.training_corpus = []
        self.test_cases = []


class HomeostasisInteractionBenchmark(BaseBenchmark):
    """
    Test interaction between homeostasis and surprise detection.

    Hypothesis: Homeostatic regulation should not interfere with
    surprise detection. Novelty should still be detected even when
    homeostasis is actively regulating.

    Measures:
    - Surprise detection accuracy with/without homeostasis
    - Mode switching accuracy under homeostatic regulation
    - Activation stability vs surprise sensitivity tradeoff
    """

    name = "homeostasis_interaction"
    description = "Test homeostasis-surprise interaction"
    category = BenchmarkCategory.COGNITIVE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.training_corpus: List[str] = []
        self.novel_inputs: List[List[str]] = []
        self.familiar_inputs: List[List[str]] = []

    def setup(self) -> None:
        """Generate test data."""
        self.training_corpus = [
            "machine learning neural network deep",
            "algorithm optimization gradient descent",
            "training model feature extraction",
        ] * 30

        self.familiar_inputs = [
            ["machine", "learning", "neural"],
            ["algorithm", "optimization", "gradient"],
            ["training", "model", "feature"],
        ]

        self.novel_inputs = [
            ["quantum", "computing", "qubit"],
            ["blockchain", "cryptocurrency", "ledger"],
            ["genomics", "protein", "folding"],
        ]

    def run(self) -> BenchmarkResult:
        """Run homeostasis interaction analysis."""
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

        # Test with homeostasis enabled
        config_with = WovenMindConfig()
        mind_with = WovenMind(config=config_with)

        for doc in self.training_corpus:
            mind_with.train(doc)
            tokens = doc.split()
            mind_with.process(tokens)

        # Collect surprise values and modes
        familiar_surprises_with = []
        novel_surprises_with = []
        familiar_modes_with = []
        novel_modes_with = []

        for tokens in self.familiar_inputs:
            res = mind_with.process(tokens)
            familiar_surprises_with.append(res.surprise.magnitude if res.surprise else 0.0)
            familiar_modes_with.append(res.mode)

        for tokens in self.novel_inputs:
            res = mind_with.process(tokens)
            novel_surprises_with.append(res.surprise.magnitude if res.surprise else 0.0)
            novel_modes_with.append(res.mode)

        # Calculate metrics
        familiar_mean = statistics.mean(familiar_surprises_with) if familiar_surprises_with else 0
        novel_mean = statistics.mean(novel_surprises_with) if novel_surprises_with else 0

        # Discrimination should be preserved
        discrimination = novel_mean - familiar_mean

        result.add_metric(
            name="surprise_discrimination_with_homeostasis",
            value=discrimination,
            unit="surprise_delta",
            threshold_min=0.1,  # Should still discriminate
        )

        # Mode switching should still work
        novel_slow_count = sum(1 for m in novel_modes_with if m == ThinkingMode.SLOW)
        familiar_fast_count = sum(1 for m in familiar_modes_with if m == ThinkingMode.FAST)

        mode_accuracy = (novel_slow_count + familiar_fast_count) / (len(self.novel_inputs) + len(self.familiar_inputs))

        result.add_metric(
            name="mode_accuracy_with_homeostasis",
            value=mode_accuracy,
            unit="ratio",
            threshold_min=0.5,  # Should be better than random
        )

        # Test activation stability during stress
        # Apply high-activation stress and check if surprise still works
        stress_inputs = [["word1", "word2", "word3", "word4", "word5"]] * 20

        for tokens in stress_inputs:
            mind_with.process(tokens)

        # After stress, novel inputs should still be detected
        post_stress_novel_surprises = []
        for tokens in self.novel_inputs:
            res = mind_with.process(tokens)
            post_stress_novel_surprises.append(res.surprise.magnitude if res.surprise else 0.0)

        post_stress_novel_mean = statistics.mean(post_stress_novel_surprises) if post_stress_novel_surprises else 0

        # Should still detect novelty after stress
        result.add_metric(
            name="post_stress_novel_surprise",
            value=post_stress_novel_mean,
            unit="magnitude",
            threshold_min=0.1,
        )

        # Compare pre and post stress
        surprise_retention = post_stress_novel_mean / novel_mean if novel_mean > 0 else 0

        result.add_metric(
            name="surprise_retention_after_stress",
            value=surprise_retention,
            unit="ratio",
            threshold_min=0.5,  # Should retain at least 50% sensitivity
        )

        result.metadata = {
            "familiar_surprises": familiar_surprises_with,
            "novel_surprises": novel_surprises_with,
            "post_stress_surprises": post_stress_novel_surprises,
            "stress_inputs_count": len(stress_inputs),
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.training_corpus = []
        self.novel_inputs = []
        self.familiar_inputs = []


class DualProcessCoherenceBenchmark(BaseBenchmark):
    """
    Test coherence between FAST and SLOW processing paths.

    Hypothesis: When both paths process the same input, their outputs
    should be complementary, not contradictory.

    Measures:
    - Agreement rate between paths
    - Complementarity (SLOW finds things FAST misses)
    - Consistency over repeated queries
    """

    name = "dual_process_coherence"
    description = "Test coherence between FAST and SLOW paths"
    category = BenchmarkCategory.COGNITIVE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.training_corpus: List[str] = []
        self.test_queries: List[List[str]] = []

    def setup(self) -> None:
        """Generate test data."""
        self.training_corpus = generate_synthetic_corpus(
            n_docs=100,
            doc_length=50,
            pattern_frequency=0.4,
        )

        self.test_queries = [
            ["pattern", "alpha", "test"],
            ["concept", "neural", "network"],
            ["random", "word1", "word2"],
            ["completely", "novel", "tokens"],
        ]

    def run(self) -> BenchmarkResult:
        """Run dual process coherence analysis."""
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

        # Train
        for doc in self.training_corpus:
            mind.train(doc)
            tokens = doc.split()
            mind.observe_pattern(tokens[:5])

        # Test dual routing if available
        agreement_scores = []
        complementarity_scores = []

        for query in self.test_queries:
            # Force FAST
            fast_result = mind.process(query, mode=ThinkingMode.FAST)
            fast_activations = fast_result.activations

            # Force SLOW
            slow_result = mind.process(query, mode=ThinkingMode.SLOW)
            slow_activations = slow_result.activations

            # Calculate agreement (Jaccard similarity)
            if fast_activations or slow_activations:
                intersection = len(fast_activations & slow_activations)
                union = len(fast_activations | slow_activations)
                jaccard = intersection / union if union > 0 else 0
                agreement_scores.append(jaccard)

            # Calculate complementarity (SLOW finds things FAST misses)
            slow_unique = slow_activations - fast_activations
            complementarity = len(slow_unique) / len(slow_activations) if slow_activations else 0
            complementarity_scores.append(complementarity)

        # Average agreement
        avg_agreement = statistics.mean(agreement_scores) if agreement_scores else 0

        result.add_metric(
            name="fast_slow_agreement",
            value=avg_agreement,
            unit="jaccard",
            threshold_min=0.1,  # Should have some agreement
            threshold_max=0.9,  # But not complete overlap (would be redundant)
        )

        # Average complementarity
        avg_complementarity = statistics.mean(complementarity_scores) if complementarity_scores else 0

        result.add_metric(
            name="slow_complementarity",
            value=avg_complementarity,
            unit="ratio",
            threshold_min=0.1,  # SLOW should find some unique things
        )

        # Test consistency: same query should give similar results
        consistency_query = ["pattern", "alpha", "test"]
        fast_results = []
        slow_results = []

        for _ in range(5):
            fast_res = mind.process(consistency_query, mode=ThinkingMode.FAST)
            slow_res = mind.process(consistency_query, mode=ThinkingMode.SLOW)
            fast_results.append(fast_res.activations)
            slow_results.append(slow_res.activations)

        # Calculate consistency (pairwise Jaccard average)
        def pairwise_consistency(results: List[Set[str]]) -> float:
            if len(results) < 2:
                return 1.0
            similarities = []
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    union = len(results[i] | results[j])
                    if union > 0:
                        similarities.append(len(results[i] & results[j]) / union)
            return statistics.mean(similarities) if similarities else 1.0

        fast_consistency = pairwise_consistency(fast_results)
        slow_consistency = pairwise_consistency(slow_results)

        result.add_metric(
            name="fast_path_consistency",
            value=fast_consistency,
            unit="jaccard",
            threshold_min=0.7,  # Should be consistent
        )

        result.add_metric(
            name="slow_path_consistency",
            value=slow_consistency,
            unit="jaccard",
            threshold_min=0.7,
        )

        result.metadata = {
            "test_queries": len(self.test_queries),
            "agreement_scores": agreement_scores,
            "complementarity_scores": complementarity_scores,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.training_corpus = []
        self.test_queries = []
