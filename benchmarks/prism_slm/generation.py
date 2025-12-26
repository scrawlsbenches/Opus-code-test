"""
Generation Quality Benchmarks for PRISM-SLM.

Tests text generation coherence, perplexity calibration, and temperature effects.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics
import time

# Add project root
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


class GenerationCoherenceBenchmark(BaseBenchmark):
    """
    Benchmark for generation coherence.

    Measures:
    - Repetition rate (lower is better)
    - Unique token ratio (higher is better)
    - Average generation length before stopping
    """

    name = "generation_coherence"
    description = "Measures text generation coherence and diversity"
    category = BenchmarkCategory.QUALITY

    def __init__(self, quick: bool = False):
        super().__init__(quick)
        self.num_generations = 20 if quick else 100
        self.max_tokens = 20 if quick else 50

    def setup(self) -> None:
        """Initialize model and train on corpus."""
        self.model = PRISMLanguageModel(context_size=3)

        # Training corpus
        training_texts = [
            "Neural networks learn patterns from data through training.",
            "Deep learning uses multiple layers of neural networks.",
            "Machine learning algorithms improve through experience.",
            "The brain processes information through neural connections.",
            "Synaptic plasticity enables learning and memory formation.",
            "Pattern recognition is a fundamental cognitive ability.",
            "Graph structures represent relationships between entities.",
            "Memory consolidation occurs during sleep cycles.",
            "Attention mechanisms focus on relevant information.",
            "Language models predict the next word in a sequence.",
        ]

        for text in training_texts:
            for _ in range(3):  # Repeat for stronger learning
                self.model.train(text)

    def run(self) -> BenchmarkResult:
        """Run generation coherence benchmark."""
        start_time = time.time()

        prompts = [
            "The neural",
            "Machine learning",
            "Pattern recognition",
            "Deep learning",
            "The brain",
        ]

        repetition_rates = []
        unique_ratios = []
        lengths = []

        for _ in range(self.num_generations // len(prompts)):
            for prompt in prompts:
                result = self.model.generate(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=1.0,
                    return_path=True,
                )

                tokens = result["path"]

                # Calculate repetition rate
                if len(tokens) > 1:
                    repetitions = sum(
                        1 for i in range(1, len(tokens))
                        if tokens[i] == tokens[i-1]
                    )
                    rep_rate = repetitions / (len(tokens) - 1)
                else:
                    rep_rate = 0.0

                repetition_rates.append(rep_rate)

                # Calculate unique token ratio
                unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
                unique_ratios.append(unique_ratio)

                # Track length
                lengths.append(len(tokens))

        duration_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        avg_repetition = statistics.mean(repetition_rates)
        avg_unique = statistics.mean(unique_ratios)
        avg_length = statistics.mean(lengths)

        metrics = [
            BenchmarkMetric(
                name="repetition_rate",
                value=avg_repetition,
                unit="ratio",
                threshold_max=0.3,  # Less than 30% repetition
            ),
            BenchmarkMetric(
                name="unique_token_ratio",
                value=avg_unique,
                unit="ratio",
                threshold_min=0.5,  # At least 50% unique
            ),
            BenchmarkMetric(
                name="avg_generation_length",
                value=avg_length,
                unit="tokens",
                threshold_min=5.0,  # At least 5 tokens average
            ),
            BenchmarkMetric(
                name="generations_per_second",
                value=self.num_generations / (duration_ms / 1000),
                unit="gen/s",
            ),
        ]

        # Check all thresholds
        all_passing = all(m.check_thresholds() for m in metrics)

        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.PASSED if all_passing else BenchmarkStatus.FAILED,
            metrics=metrics,
            duration_ms=duration_ms,
            metadata={
                "num_generations": self.num_generations,
                "max_tokens": self.max_tokens,
            },
        )

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None


class PerplexityCalibrationBenchmark(BaseBenchmark):
    """
    Benchmark for perplexity calibration.

    Tests that:
    - Training data has low perplexity
    - Random text has high perplexity
    - The ratio between them is significant
    """

    name = "perplexity_calibration"
    description = "Verifies perplexity correctly distinguishes in-domain vs out-of-domain text"
    category = BenchmarkCategory.QUALITY

    def __init__(self, quick: bool = False):
        super().__init__(quick)

    def setup(self) -> None:
        """Initialize and train model."""
        self.model = PRISMLanguageModel(context_size=3)

        # Training corpus - domain-specific text
        self.training_texts = [
            "Neural networks learn patterns from data.",
            "Deep learning uses backpropagation for training.",
            "Machine learning models require labeled data.",
            "The neural network architecture determines capacity.",
            "Training deep networks requires careful initialization.",
        ]

        for text in self.training_texts:
            for _ in range(5):
                self.model.train(text)

        # In-domain test sentences
        self.in_domain = [
            "Neural networks process data efficiently.",
            "Deep learning models learn representations.",
            "Machine learning uses training data.",
        ]

        # Out-of-domain sentences
        self.out_domain = [
            "The cat sat on the warm windowsill.",
            "Cooking recipes require fresh ingredients.",
            "The sunset painted the sky orange and pink.",
        ]

        # Nonsense
        self.nonsense = [
            "Xyzzy foobar blorp flonk wibble.",
            "Glorp snarf quux baz frobnicator.",
        ]

    def run(self) -> BenchmarkResult:
        """Run perplexity calibration benchmark."""
        start_time = time.time()

        # Measure perplexity for each category
        in_domain_ppl = [self.model.perplexity(s) for s in self.in_domain]
        out_domain_ppl = [self.model.perplexity(s) for s in self.out_domain]
        nonsense_ppl = [self.model.perplexity(s) for s in self.nonsense]

        avg_in = statistics.mean(in_domain_ppl)
        avg_out = statistics.mean(out_domain_ppl)
        avg_nonsense = statistics.mean(nonsense_ppl)

        # Calculate separation ratios
        out_in_ratio = avg_out / avg_in if avg_in > 0 else float('inf')
        nonsense_in_ratio = avg_nonsense / avg_in if avg_in > 0 else float('inf')

        duration_ms = (time.time() - start_time) * 1000

        metrics = [
            BenchmarkMetric(
                name="in_domain_perplexity",
                value=avg_in,
                unit="ppl",
                threshold_max=1000,  # Should be relatively low
            ),
            BenchmarkMetric(
                name="out_domain_perplexity",
                value=avg_out,
                unit="ppl",
            ),
            BenchmarkMetric(
                name="nonsense_perplexity",
                value=min(avg_nonsense, 1e10),  # Cap for display
                unit="ppl",
            ),
            BenchmarkMetric(
                name="out_in_separation_ratio",
                value=out_in_ratio,
                unit="ratio",
                threshold_min=2.0,  # Out-domain should be 2x+ higher
            ),
            BenchmarkMetric(
                name="nonsense_in_separation_ratio",
                value=min(nonsense_in_ratio, 1e6),  # Cap for display
                unit="ratio",
                threshold_min=10.0,  # Nonsense should be 10x+ higher
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


class TemperatureDiversityBenchmark(BaseBenchmark):
    """
    Benchmark for temperature effects on generation diversity.

    Tests that:
    - Low temperature produces more deterministic output
    - High temperature produces more diverse output
    - The relationship is monotonic
    """

    name = "temperature_diversity"
    description = "Verifies temperature parameter affects generation diversity correctly"
    category = BenchmarkCategory.STABILITY

    def __init__(self, quick: bool = False):
        super().__init__(quick)
        self.num_samples = 10 if quick else 50

    def setup(self) -> None:
        """Initialize and train model."""
        self.model = PRISMLanguageModel(context_size=3)

        training_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A quick brown fox jumped over a lazy dog.",
            "Quick brown foxes jump over lazy dogs.",
            "The fast brown fox leaps over the sleepy dog.",
            "A swift brown fox hops over a drowsy dog.",
        ]

        for text in training_texts:
            for _ in range(5):
                self.model.train(text)

    def run(self) -> BenchmarkResult:
        """Run temperature diversity benchmark."""
        start_time = time.time()

        temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]
        prompt = "The quick"

        diversity_by_temp = {}

        for temp in temperatures:
            generations = []
            for _ in range(self.num_samples):
                result = self.model.generate(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=temp,
                    return_path=True,
                )
                generations.append(tuple(result["path"]))

            # Measure diversity as unique generations / total
            unique = len(set(generations))
            diversity = unique / len(generations)
            diversity_by_temp[temp] = diversity

        duration_ms = (time.time() - start_time) * 1000

        # Check monotonicity (diversity should increase with temperature)
        diversities = list(diversity_by_temp.values())
        is_monotonic = all(
            diversities[i] <= diversities[i+1] * 1.1  # Allow 10% tolerance
            for i in range(len(diversities) - 1)
        )

        # Calculate diversity range
        diversity_range = max(diversities) - min(diversities)

        metrics = [
            BenchmarkMetric(
                name="diversity_at_temp_0.3",
                value=diversity_by_temp[0.3],
                unit="ratio",
            ),
            BenchmarkMetric(
                name="diversity_at_temp_1.0",
                value=diversity_by_temp[1.0],
                unit="ratio",
            ),
            BenchmarkMetric(
                name="diversity_at_temp_2.0",
                value=diversity_by_temp[2.0],
                unit="ratio",
            ),
            BenchmarkMetric(
                name="diversity_range",
                value=diversity_range,
                unit="ratio",
                threshold_min=0.1,  # At least 10% range
            ),
            BenchmarkMetric(
                name="monotonicity_score",
                value=1.0 if is_monotonic else 0.0,
                unit="bool",
                threshold_min=0.5,  # Should be monotonic
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
                "temperatures": temperatures,
                "num_samples": self.num_samples,
            },
        )

    def teardown(self) -> None:
        self.model = None
