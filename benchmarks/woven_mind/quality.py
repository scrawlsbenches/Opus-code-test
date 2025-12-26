"""
Quality benchmarks for Woven Mind.

These benchmarks test the quality of cognitive outputs:
- Abstraction formation quality
- Mode switching accuracy
- Retrieval relevance
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import statistics

from .base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
    generate_synthetic_corpus,
)


class AbstractionQualityBenchmark(BaseBenchmark):
    """
    Test quality of abstraction formation.

    Measures:
    - Precision: What fraction of formed abstractions are meaningful?
    - Recall: What fraction of known concepts are discovered?
    - Hierarchy coherence: Do levels make sense?
    """

    name = "abstraction_quality"
    description = "Test quality of Cortex abstraction formation"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Ground truth: patterns we know should become abstractions
        self.known_patterns: List[Tuple[Set[str], str]] = [
            ({"neural", "network"}, "neural_network_concept"),
            ({"deep", "learning"}, "deep_learning_concept"),
            ({"machine", "learning"}, "machine_learning_concept"),
            ({"natural", "language"}, "nlp_concept"),
            ({"computer", "vision"}, "cv_concept"),
        ]
        self.corpus: List[str] = []

    def setup(self) -> None:
        """Generate corpus with known patterns."""
        # Create documents containing known patterns
        pattern_docs = []
        for pattern_set, _ in self.known_patterns:
            pattern_text = " ".join(pattern_set)
            # Repeat each pattern enough times to exceed min_frequency
            for _ in range(5):
                doc = f"This discusses {pattern_text} and related topics about {pattern_text}."
                pattern_docs.append(doc)

        # Add noise documents
        noise_docs = [
            "Random words that dont form patterns unique789",
            "Another document with different vocabulary xyz123",
            "Some more text without the target patterns abc456",
        ] * 10

        self.corpus = pattern_docs + noise_docs

    def run(self) -> BenchmarkResult:
        """Run abstraction quality analysis."""
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

        config = WovenMindConfig(min_frequency=3)
        mind = WovenMind(config=config)

        # Train and observe patterns
        for doc in self.corpus:
            mind.train(doc)
            tokens = doc.lower().split()
            mind.observe_pattern(tokens[:5])

        # Run consolidation to form abstractions
        if hasattr(mind, 'consolidate'):
            mind.consolidate()

        # Get formed abstractions
        if hasattr(mind.cortex, 'get_abstractions'):
            abstractions = mind.cortex.get_abstractions()
        else:
            abstractions = []

        # Calculate precision: meaningful / total
        # An abstraction is meaningful if it overlaps with known patterns
        meaningful_count = 0
        for abs in abstractions:
            source_nodes = getattr(abs, 'source_nodes', set())
            if isinstance(source_nodes, frozenset):
                source_nodes = set(source_nodes)

            for known_pattern, _ in self.known_patterns:
                overlap = len(source_nodes & known_pattern)
                if overlap >= 2:  # At least 2 terms match
                    meaningful_count += 1
                    break

        precision = meaningful_count / len(abstractions) if abstractions else 0.0

        result.add_metric(
            name="abstraction_precision",
            value=precision,
            unit="ratio",
            threshold_min=0.5,  # At least 50% should be meaningful
        )

        # Calculate recall: discovered / known
        discovered_patterns = 0
        for known_pattern, name in self.known_patterns:
            for abs in abstractions:
                source_nodes = getattr(abs, 'source_nodes', set())
                if isinstance(source_nodes, frozenset):
                    source_nodes = set(source_nodes)

                overlap = len(source_nodes & known_pattern)
                if overlap >= 2:
                    discovered_patterns += 1
                    break

        recall = discovered_patterns / len(self.known_patterns) if self.known_patterns else 0.0

        result.add_metric(
            name="abstraction_recall",
            value=recall,
            unit="ratio",
            threshold_min=0.6,  # Should discover at least 60% of patterns
        )

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result.add_metric(
            name="abstraction_f1",
            value=f1,
            unit="score",
            threshold_min=0.5,
        )

        # Hierarchy coherence: check that higher-level abstractions
        # are composed of lower-level ones
        level_counts = {}
        for abs in abstractions:
            level = getattr(abs, 'level', 1)
            level_counts[level] = level_counts.get(level, 0) + 1

        # Should have more level 1 than level 2 (pyramid shape)
        if level_counts.get(2, 0) > level_counts.get(1, 0) and level_counts.get(1, 0) > 0:
            hierarchy_coherent = 0.0
        else:
            hierarchy_coherent = 1.0

        result.add_metric(
            name="hierarchy_coherence",
            value=hierarchy_coherent,
            unit="boolean",
            threshold_min=0.5,
        )

        result.metadata = {
            "total_abstractions": len(abstractions),
            "meaningful_abstractions": meaningful_count,
            "discovered_patterns": discovered_patterns,
            "known_patterns": len(self.known_patterns),
            "level_distribution": level_counts,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.corpus = []


class ModeSwitchingBenchmark(BaseBenchmark):
    """
    Test accuracy of mode switching decisions.

    Hypothesis: System should switch to SLOW mode for genuinely novel
    inputs and stay in FAST mode for familiar patterns.

    Measures:
    - True positive rate: SLOW mode for novel inputs
    - True negative rate: FAST mode for familiar inputs
    - Mode switching latency
    """

    name = "mode_switching_accuracy"
    description = "Test accuracy of FAST/SLOW mode selection"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.training_patterns: List[List[str]] = []
        self.novel_patterns: List[List[str]] = []
        self.familiar_patterns: List[List[str]] = []

    def setup(self) -> None:
        """Generate labeled test cases."""
        # Patterns that will be trained (should be familiar after training)
        self.training_patterns = [
            ["neural", "network", "training"],
            ["deep", "learning", "model"],
            ["gradient", "descent", "optimization"],
            ["convolutional", "layer", "filters"],
            ["attention", "mechanism", "transformer"],
        ]

        # Patterns that are similar to training (should be FAST)
        self.familiar_patterns = [
            ["neural", "network", "inference"],
            ["deep", "learning", "architecture"],
            ["gradient", "descent", "algorithm"],
        ]

        # Patterns that are completely different (should trigger SLOW)
        self.novel_patterns = [
            ["quantum", "computing", "qubit"],
            ["blockchain", "cryptocurrency", "ledger"],
            ["genomics", "dna", "sequencing"],
        ]

    def run(self) -> BenchmarkResult:
        """Run mode switching accuracy analysis."""
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

        config = WovenMindConfig(
            surprise_threshold=0.3,
            auto_switch=True,
        )
        mind = WovenMind(config=config)

        # Training phase
        for pattern in self.training_patterns:
            text = " ".join(pattern)
            mind.train(text)
            # Process multiple times to establish baseline
            for _ in range(5):
                mind.process(pattern)

        # Test familiar patterns (should be FAST)
        familiar_fast_count = 0
        for pattern in self.familiar_patterns:
            res = mind.process(pattern)
            if res.mode == ThinkingMode.FAST:
                familiar_fast_count += 1

        true_negative_rate = familiar_fast_count / len(self.familiar_patterns) if self.familiar_patterns else 0.0

        result.add_metric(
            name="familiar_fast_rate",
            value=true_negative_rate,
            unit="ratio",
            threshold_min=0.6,  # Should be FAST at least 60% of time
        )

        # Test novel patterns (should be SLOW)
        novel_slow_count = 0
        for pattern in self.novel_patterns:
            res = mind.process(pattern)
            if res.mode == ThinkingMode.SLOW:
                novel_slow_count += 1

        true_positive_rate = novel_slow_count / len(self.novel_patterns) if self.novel_patterns else 0.0

        result.add_metric(
            name="novel_slow_rate",
            value=true_positive_rate,
            unit="ratio",
            threshold_min=0.6,  # Should be SLOW at least 60% of time
        )

        # Overall accuracy
        total_correct = familiar_fast_count + novel_slow_count
        total_tests = len(self.familiar_patterns) + len(self.novel_patterns)
        accuracy = total_correct / total_tests if total_tests > 0 else 0.0

        result.add_metric(
            name="mode_switching_accuracy",
            value=accuracy,
            unit="ratio",
            threshold_min=0.6,
        )

        # Test mode switching latency (how quickly does surprise stabilize?)
        # Introduce novel pattern and measure steps until SLOW triggers
        test_novel = ["completely", "new", "terminology", "xyz"]
        mode_history = []

        for i in range(20):
            res = mind.process(test_novel)
            mode_history.append(res.mode)

        # Find first SLOW if any
        first_slow_idx = next(
            (i for i, m in enumerate(mode_history) if m == ThinkingMode.SLOW),
            len(mode_history)
        )

        result.add_metric(
            name="novel_detection_latency",
            value=first_slow_idx,
            unit="steps",
            threshold_max=5,  # Should detect novelty within 5 steps
        )

        result.metadata = {
            "training_patterns": len(self.training_patterns),
            "familiar_tests": len(self.familiar_patterns),
            "novel_tests": len(self.novel_patterns),
            "familiar_fast": familiar_fast_count,
            "novel_slow": novel_slow_count,
            "mode_history_sample": [m.name for m in mode_history[:10]],
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.training_patterns = []
        self.novel_patterns = []
        self.familiar_patterns = []


class RetrievalRelevanceBenchmark(BaseBenchmark):
    """
    Test relevance of retrieved activations.

    Measures:
    - Activation relevance to query
    - Ranking quality (MRR, NDCG)
    - Coverage of expected concepts
    """

    name = "retrieval_relevance"
    description = "Test relevance of Hive/Cortex activations"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.training_corpus: List[str] = []
        self.test_queries: List[Tuple[List[str], Set[str]]] = []

    def setup(self) -> None:
        """Generate corpus and queries with expected results."""
        self.training_corpus = [
            "neural networks are used for deep learning tasks",
            "machine learning includes neural network models",
            "deep learning uses multiple neural network layers",
            "convolutional networks process images efficiently",
            "recurrent networks handle sequential data",
            "transformers use attention mechanisms for nlp",
            "natural language processing benefits from transformers",
        ]

        # Queries with expected activations
        self.test_queries = [
            (
                ["neural", "network"],
                {"neural", "network", "deep", "learning", "machine"}
            ),
            (
                ["transformer", "attention"],
                {"transformer", "attention", "nlp", "language"}
            ),
            (
                ["image", "processing"],
                {"convolutional", "image", "network"}
            ),
        ]

    def run(self) -> BenchmarkResult:
        """Run retrieval relevance analysis."""
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

        # Test queries
        precision_scores = []
        recall_scores = []

        for query_tokens, expected_activations in self.test_queries:
            res = mind.process(query_tokens)
            actual = res.activations

            # Calculate precision and recall
            if actual:
                true_positives = len(actual & expected_activations)
                precision = true_positives / len(actual)
                recall = true_positives / len(expected_activations) if expected_activations else 0

                precision_scores.append(precision)
                recall_scores.append(recall)

        avg_precision = statistics.mean(precision_scores) if precision_scores else 0.0
        avg_recall = statistics.mean(recall_scores) if recall_scores else 0.0

        result.add_metric(
            name="activation_precision",
            value=avg_precision,
            unit="ratio",
            threshold_min=0.3,
        )

        result.add_metric(
            name="activation_recall",
            value=avg_recall,
            unit="ratio",
            threshold_min=0.3,
        )

        # F1
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

        result.add_metric(
            name="activation_f1",
            value=f1,
            unit="score",
            threshold_min=0.3,
        )

        result.metadata = {
            "corpus_size": len(self.training_corpus),
            "query_count": len(self.test_queries),
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
        }

        return result

    def teardown(self) -> None:
        """Clean up."""
        self.training_corpus = []
        self.test_queries = []
