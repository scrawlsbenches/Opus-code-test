"""
Quality Evaluation: Measuring SparkSLM prediction and search quality.

This module provides tools to evaluate:
1. Prediction accuracy (accuracy@k, perplexity)
2. Search quality (precision, recall, with/without spark)
3. Alignment acceleration (disambiguation reduction)

Used to validate Phase 2 hypotheses:
- Predictions provide useful signal
- Spark-enhanced expansion improves search
- Alignment context reduces disambiguation
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from .ngram import NGramModel


@dataclass
class PredictionMetrics:
    """Metrics for prediction quality."""

    accuracy_at_1: float  # Top-1 prediction accuracy
    accuracy_at_5: float  # Top-5 prediction accuracy
    accuracy_at_10: float  # Top-10 prediction accuracy
    mean_reciprocal_rank: float  # MRR
    perplexity: float  # Average perplexity
    coverage: float  # % of test cases with predictions
    total_tests: int
    successful_predictions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy_at_1': self.accuracy_at_1,
            'accuracy_at_5': self.accuracy_at_5,
            'accuracy_at_10': self.accuracy_at_10,
            'mean_reciprocal_rank': self.mean_reciprocal_rank,
            'perplexity': self.perplexity,
            'coverage': self.coverage,
            'total_tests': self.total_tests,
            'successful_predictions': self.successful_predictions,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Prediction Quality:\n"
            f"  Accuracy@1: {self.accuracy_at_1:.1%}\n"
            f"  Accuracy@5: {self.accuracy_at_5:.1%}\n"
            f"  Accuracy@10: {self.accuracy_at_10:.1%}\n"
            f"  MRR: {self.mean_reciprocal_rank:.3f}\n"
            f"  Perplexity: {self.perplexity:.1f}\n"
            f"  Coverage: {self.coverage:.1%}\n"
            f"  Tests: {self.total_tests} ({self.successful_predictions} with predictions)"
        )


@dataclass
class SearchMetrics:
    """Metrics for search quality."""

    precision_at_5: float  # Precision@5
    recall_at_5: float  # Recall@5
    ndcg_at_5: float  # NDCG@5
    mrr: float  # Mean Reciprocal Rank
    queries_tested: int
    avg_results: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'precision_at_5': self.precision_at_5,
            'recall_at_5': self.recall_at_5,
            'ndcg_at_5': self.ndcg_at_5,
            'mrr': self.mrr,
            'queries_tested': self.queries_tested,
            'avg_results': self.avg_results,
        }


@dataclass
class SearchComparison:
    """Comparison of search with vs without spark."""

    baseline: SearchMetrics
    with_spark: SearchMetrics
    precision_improvement: float  # % improvement in precision
    recall_improvement: float  # % improvement in recall
    mrr_improvement: float  # % improvement in MRR
    queries_improved: int  # Number of queries that improved
    queries_same: int  # Number unchanged
    queries_regressed: int  # Number that got worse

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'baseline': self.baseline.to_dict(),
            'with_spark': self.with_spark.to_dict(),
            'precision_improvement': self.precision_improvement,
            'recall_improvement': self.recall_improvement,
            'mrr_improvement': self.mrr_improvement,
            'queries_improved': self.queries_improved,
            'queries_same': self.queries_same,
            'queries_regressed': self.queries_regressed,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Search Quality Comparison:\n"
            f"  Baseline Precision@5: {self.baseline.precision_at_5:.1%}\n"
            f"  With Spark Precision@5: {self.with_spark.precision_at_5:.1%}\n"
            f"  Precision Improvement: {self.precision_improvement:+.1%}\n"
            f"  \n"
            f"  Baseline Recall@5: {self.baseline.recall_at_5:.1%}\n"
            f"  With Spark Recall@5: {self.with_spark.recall_at_5:.1%}\n"
            f"  Recall Improvement: {self.recall_improvement:+.1%}\n"
            f"  \n"
            f"  Queries: {self.queries_improved} improved, "
            f"{self.queries_same} same, {self.queries_regressed} regressed"
        )


@dataclass
class AlignmentMetrics:
    """Metrics for alignment acceleration."""

    baseline_rounds: float  # Average rounds without alignment
    with_alignment_rounds: float  # Average rounds with alignment
    reduction_percent: float  # % reduction in rounds
    sessions_tested: int
    sessions_improved: int
    sessions_same: int
    avg_alignment_hits: float  # Average alignment lookups that hit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'baseline_rounds': self.baseline_rounds,
            'with_alignment_rounds': self.with_alignment_rounds,
            'reduction_percent': self.reduction_percent,
            'sessions_tested': self.sessions_tested,
            'sessions_improved': self.sessions_improved,
            'sessions_same': self.sessions_same,
            'avg_alignment_hits': self.avg_alignment_hits,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Alignment Acceleration:\n"
            f"  Baseline Rounds: {self.baseline_rounds:.1f}\n"
            f"  With Alignment: {self.with_alignment_rounds:.1f}\n"
            f"  Reduction: {self.reduction_percent:.1%}\n"
            f"  Sessions: {self.sessions_tested} tested, "
            f"{self.sessions_improved} improved\n"
            f"  Avg Alignment Hits: {self.avg_alignment_hits:.1f}"
        )


class QualityEvaluator:
    """
    Evaluate SparkSLM prediction and search quality.

    Example:
        >>> evaluator = QualityEvaluator(ngram_model)
        >>> metrics = evaluator.evaluate_predictions(test_data)
        >>> print(metrics.summary())
    """

    def __init__(self, ngram_model: NGramModel):
        """
        Initialize evaluator.

        Args:
            ngram_model: Trained n-gram model to evaluate.
        """
        self.model = ngram_model

    def evaluate_predictions(
        self,
        test_texts: List[str],
        context_size: int = 2
    ) -> PredictionMetrics:
        """
        Evaluate prediction quality on test texts.

        For each text, uses context_size words to predict the next word.

        Args:
            test_texts: List of test texts.
            context_size: Number of context words to use.

        Returns:
            PredictionMetrics with accuracy and perplexity.
        """
        correct_at_1 = 0
        correct_at_5 = 0
        correct_at_10 = 0
        reciprocal_ranks = []
        perplexities = []
        total_tests = 0
        predictions_made = 0

        for text in test_texts:
            tokens = self.model._tokenize(text)
            if len(tokens) < context_size + 1:
                continue

            # Test each position
            for i in range(context_size, len(tokens)):
                context = tokens[i - context_size:i]
                actual = tokens[i]
                total_tests += 1

                # Get predictions
                predictions = self.model.predict(context, top_k=10)

                if not predictions:
                    continue

                predictions_made += 1
                predicted_words = [p[0] for p in predictions]

                # Check accuracy
                if actual == predicted_words[0]:
                    correct_at_1 += 1
                if actual in predicted_words[:5]:
                    correct_at_5 += 1
                if actual in predicted_words[:10]:
                    correct_at_10 += 1

                # Calculate reciprocal rank
                if actual in predicted_words:
                    rank = predicted_words.index(actual) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            # Calculate perplexity for the text
            perplexity = self.model.perplexity(text)
            if perplexity < float('inf'):
                perplexities.append(perplexity)

        # Calculate averages
        if total_tests == 0:
            return PredictionMetrics(
                accuracy_at_1=0.0,
                accuracy_at_5=0.0,
                accuracy_at_10=0.0,
                mean_reciprocal_rank=0.0,
                perplexity=float('inf'),
                coverage=0.0,
                total_tests=0,
                successful_predictions=0,
            )

        return PredictionMetrics(
            accuracy_at_1=correct_at_1 / total_tests,
            accuracy_at_5=correct_at_5 / total_tests,
            accuracy_at_10=correct_at_10 / total_tests,
            mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
            perplexity=sum(perplexities) / len(perplexities) if perplexities else float('inf'),
            coverage=predictions_made / total_tests,
            total_tests=total_tests,
            successful_predictions=predictions_made,
        )

    def create_held_out_split(
        self,
        texts: List[str],
        test_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Split texts into train and test sets.

        Args:
            texts: All texts to split.
            test_ratio: Fraction for test set.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_texts, test_texts).
        """
        random.seed(seed)
        shuffled = texts.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]

    def cross_validate_predictions(
        self,
        texts: List[str],
        folds: int = 5
    ) -> List[PredictionMetrics]:
        """
        Cross-validate prediction quality.

        Args:
            texts: All texts to use.
            folds: Number of cross-validation folds.

        Returns:
            List of PredictionMetrics for each fold.
        """
        random.shuffle(texts)
        fold_size = len(texts) // folds
        results = []

        for i in range(folds):
            test_start = i * fold_size
            test_end = test_start + fold_size

            test_texts = texts[test_start:test_end]
            train_texts = texts[:test_start] + texts[test_end:]

            # Train new model on training data
            fold_model = NGramModel(n=self.model.n)
            fold_model.train(train_texts)

            # Evaluate on test data
            fold_evaluator = QualityEvaluator(fold_model)
            metrics = fold_evaluator.evaluate_predictions(test_texts)
            results.append(metrics)

        return results

    def measure_perplexity_stability(
        self,
        texts: List[str],
        runs: int = 5
    ) -> Dict[str, float]:
        """
        Measure perplexity stability across runs.

        Args:
            texts: Texts to measure perplexity on.
            runs: Number of runs to average.

        Returns:
            Dict with mean, std, min, max perplexity.
        """
        perplexities = []

        for _ in range(runs):
            total_perplexity = 0
            count = 0
            for text in texts:
                p = self.model.perplexity(text)
                if p < float('inf'):
                    total_perplexity += p
                    count += 1
            if count > 0:
                perplexities.append(total_perplexity / count)

        if not perplexities:
            return {
                'mean': float('inf'),
                'std': 0.0,
                'min': float('inf'),
                'max': float('inf'),
                'is_stable': False,
            }

        mean = sum(perplexities) / len(perplexities)
        variance = sum((p - mean) ** 2 for p in perplexities) / len(perplexities)
        std = variance ** 0.5

        return {
            'mean': mean,
            'std': std,
            'min': min(perplexities),
            'max': max(perplexities),
            'is_stable': std / mean < 0.1 if mean > 0 else True,  # CV < 10%
        }


class SearchQualityEvaluator:
    """
    Evaluate search quality with and without spark.

    Requires a search function and relevance judgments.
    """

    def __init__(
        self,
        search_fn: Callable[[str], List[Tuple[str, float]]],
        search_with_spark_fn: Callable[[str], List[Tuple[str, float]]]
    ):
        """
        Initialize evaluator.

        Args:
            search_fn: Function that returns (doc_id, score) pairs for a query.
            search_with_spark_fn: Same but with spark enhancement.
        """
        self.search_fn = search_fn
        self.search_with_spark_fn = search_with_spark_fn

    def evaluate_search(
        self,
        queries: List[str],
        relevance: Dict[str, Set[str]],  # query -> relevant doc_ids
        k: int = 5
    ) -> SearchMetrics:
        """
        Evaluate search quality on a set of queries.

        Args:
            queries: List of queries to test.
            relevance: Dict mapping queries to relevant doc IDs.
            k: Number of results to evaluate.

        Returns:
            SearchMetrics with precision, recall, etc.
        """
        precisions = []
        recalls = []
        mrrs = []
        result_counts = []

        for query in queries:
            results = self.search_fn(query)[:k]
            result_ids = [r[0] for r in results]
            relevant = relevance.get(query, set())

            result_counts.append(len(results))

            if not relevant:
                continue

            # Calculate precision
            hits = len(set(result_ids) & relevant)
            precision = hits / k if k > 0 else 0
            precisions.append(precision)

            # Calculate recall
            recall = hits / len(relevant) if relevant else 0
            recalls.append(recall)

            # Calculate MRR
            mrr = 0.0
            for i, doc_id in enumerate(result_ids):
                if doc_id in relevant:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)

        return SearchMetrics(
            precision_at_5=sum(precisions) / len(precisions) if precisions else 0,
            recall_at_5=sum(recalls) / len(recalls) if recalls else 0,
            ndcg_at_5=0.0,  # TODO: implement if needed
            mrr=sum(mrrs) / len(mrrs) if mrrs else 0,
            queries_tested=len(queries),
            avg_results=sum(result_counts) / len(result_counts) if result_counts else 0,
        )

    def compare_search(
        self,
        queries: List[str],
        relevance: Dict[str, Set[str]],
        k: int = 5
    ) -> SearchComparison:
        """
        Compare search with and without spark.

        Args:
            queries: Queries to test.
            relevance: Relevance judgments.
            k: Results to evaluate.

        Returns:
            SearchComparison with baseline and spark metrics.
        """
        # Evaluate baseline
        baseline = self.evaluate_search(queries, relevance, k)

        # Evaluate with spark
        original_fn = self.search_fn
        self.search_fn = self.search_with_spark_fn
        with_spark = self.evaluate_search(queries, relevance, k)
        self.search_fn = original_fn

        # Calculate improvements
        precision_imp = (
            (with_spark.precision_at_5 - baseline.precision_at_5) / baseline.precision_at_5
            if baseline.precision_at_5 > 0 else 0
        )
        recall_imp = (
            (with_spark.recall_at_5 - baseline.recall_at_5) / baseline.recall_at_5
            if baseline.recall_at_5 > 0 else 0
        )
        mrr_imp = (
            (with_spark.mrr - baseline.mrr) / baseline.mrr
            if baseline.mrr > 0 else 0
        )

        # Count improved/same/regressed queries
        improved = 0
        same = 0
        regressed = 0

        for query in queries:
            base_results = set(r[0] for r in self.search_fn(query)[:k])
            spark_results = set(r[0] for r in self.search_with_spark_fn(query)[:k])
            relevant = relevance.get(query, set())

            if not relevant:
                same += 1
                continue

            base_hits = len(base_results & relevant)
            spark_hits = len(spark_results & relevant)

            if spark_hits > base_hits:
                improved += 1
            elif spark_hits < base_hits:
                regressed += 1
            else:
                same += 1

        return SearchComparison(
            baseline=baseline,
            with_spark=with_spark,
            precision_improvement=precision_imp,
            recall_improvement=recall_imp,
            mrr_improvement=mrr_imp,
            queries_improved=improved,
            queries_same=same,
            queries_regressed=regressed,
        )


class AlignmentEvaluator:
    """
    Evaluate alignment context acceleration.

    Simulates user sessions to measure disambiguation reduction.
    """

    def __init__(
        self,
        alignment_lookup_fn: Callable[[str], List[Dict]],
        disambiguation_needed_fn: Callable[[str], bool]
    ):
        """
        Initialize evaluator.

        Args:
            alignment_lookup_fn: Function to look up alignment context.
            disambiguation_needed_fn: Function that returns True if query
                                      needs disambiguation.
        """
        self.alignment_lookup = alignment_lookup_fn
        self.needs_disambiguation = disambiguation_needed_fn

    def simulate_session(
        self,
        queries: List[str],
        use_alignment: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate a user session.

        Args:
            queries: Queries in the session.
            use_alignment: Whether to use alignment context.

        Returns:
            Dict with session statistics.
        """
        rounds = 0
        alignment_hits = 0

        for query in queries:
            # Check if disambiguation needed
            if self.needs_disambiguation(query):
                if use_alignment:
                    # Check if alignment context helps
                    context = self.alignment_lookup(query)
                    if context:
                        alignment_hits += 1
                        # Alignment helped, no extra round needed
                        continue

                # Need an extra disambiguation round
                rounds += 1

        return {
            'total_queries': len(queries),
            'disambiguation_rounds': rounds,
            'alignment_hits': alignment_hits,
        }

    def evaluate_acceleration(
        self,
        sessions: List[List[str]],  # Each session is a list of queries
    ) -> AlignmentMetrics:
        """
        Evaluate alignment acceleration across sessions.

        Args:
            sessions: List of query sessions.

        Returns:
            AlignmentMetrics with acceleration statistics.
        """
        baseline_rounds = []
        alignment_rounds = []
        alignment_hits = []
        improved = 0
        same = 0

        for session in sessions:
            # Simulate without alignment
            base = self.simulate_session(session, use_alignment=False)
            baseline_rounds.append(base['disambiguation_rounds'])

            # Simulate with alignment
            with_align = self.simulate_session(session, use_alignment=True)
            alignment_rounds.append(with_align['disambiguation_rounds'])
            alignment_hits.append(with_align['alignment_hits'])

            if with_align['disambiguation_rounds'] < base['disambiguation_rounds']:
                improved += 1
            else:
                same += 1

        avg_baseline = sum(baseline_rounds) / len(baseline_rounds) if baseline_rounds else 0
        avg_alignment = sum(alignment_rounds) / len(alignment_rounds) if alignment_rounds else 0
        avg_hits = sum(alignment_hits) / len(alignment_hits) if alignment_hits else 0

        reduction = (
            (avg_baseline - avg_alignment) / avg_baseline
            if avg_baseline > 0 else 0
        )

        return AlignmentMetrics(
            baseline_rounds=avg_baseline,
            with_alignment_rounds=avg_alignment,
            reduction_percent=reduction,
            sessions_tested=len(sessions),
            sessions_improved=improved,
            sessions_same=same,
            avg_alignment_hits=avg_hits,
        )


def generate_test_queries(
    vocabulary: Set[str],
    count: int = 20,
    min_words: int = 2,
    max_words: int = 5,
    seed: int = 42
) -> List[str]:
    """
    Generate synthetic test queries from vocabulary.

    Args:
        vocabulary: Set of words to sample from.
        count: Number of queries to generate.
        min_words: Minimum words per query.
        max_words: Maximum words per query.
        seed: Random seed.

    Returns:
        List of generated queries.
    """
    random.seed(seed)
    vocab_list = list(vocabulary)
    queries = []

    for _ in range(count):
        length = random.randint(min_words, max_words)
        words = random.sample(vocab_list, min(length, len(vocab_list)))
        queries.append(' '.join(words))

    return queries


def generate_relevance_judgments(
    queries: List[str],
    doc_ids: List[str],
    avg_relevant: int = 3,
    seed: int = 42
) -> Dict[str, Set[str]]:
    """
    Generate synthetic relevance judgments.

    For testing only - not for real evaluation.

    Args:
        queries: List of queries.
        doc_ids: List of document IDs.
        avg_relevant: Average relevant docs per query.
        seed: Random seed.

    Returns:
        Dict mapping queries to relevant doc IDs.
    """
    random.seed(seed)
    relevance = {}

    for query in queries:
        n_relevant = max(1, random.randint(1, avg_relevant * 2))
        relevant = set(random.sample(doc_ids, min(n_relevant, len(doc_ids))))
        relevance[query] = relevant

    return relevance
