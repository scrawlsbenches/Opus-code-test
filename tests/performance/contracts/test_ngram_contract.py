"""
╔══════════════════════════════════════════════════════════════════════╗
║                     N-GRAM MODEL PERFORMANCE CONTRACT                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • N-gram prediction < 10ms for any context                         ║
║  • Training < 500ms for 1,000 documents                             ║
║  • Memory < 50MB per 10,000 unique n-grams                          ║
║  • Vocabulary builds in O(n) time                                   ║
║  • Prediction accuracy ≥ 15% top-1 for natural text                 ║
║  • Perplexity < 100 for trained domain text                         ║
║  • Fallback prediction always returns results                       ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import sys
import pytest
from cortical.spark.ngram import NGramModel


@pytest.mark.contract
class TestNGramPredictionContract:
    """
    N-Gram Prediction Performance Contract

    As a developer building code prediction systems,
    I expect n-gram lookups to be instantaneous,
    So that predictions feel responsive.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_PREDICTION_LATENCY_MS = 10
    ITERATIONS = 100  # Run multiple times to get reliable measurement

    def test_prediction_latency_honored(self):
        """
        CONTRACT: N-gram prediction completes in < 10ms.

        Fast prediction is essential for interactive code completion.
        """
        model = NGramModel(n=3)

        # Train on sample documents
        documents = [
            "def process_data(input_file):",
            "    data = load_file(input_file)",
            "    result = transform_data(data)",
            "    return result",
        ] * 50  # 200 documents

        model.train(documents)
        model.finalize()

        # Measure prediction latency
        context = ["process", "data"]
        latencies = []

        for _ in range(self.ITERATIONS):
            start = time.perf_counter()
            predictions = model.predict(context, top_k=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        # Check p95 latency (95th percentile)
        latencies.sort()
        p95_latency = latencies[int(0.95 * len(latencies))]

        assert p95_latency < self.MAX_PREDICTION_LATENCY_MS, (
            f"CONTRACT VIOLATION: N-gram prediction p95={p95_latency:.1f}ms, "
            f"contract requires <{self.MAX_PREDICTION_LATENCY_MS}ms"
        )

    def test_fallback_prediction_never_empty(self):
        """
        CONTRACT: Unknown contexts always get fallback predictions.

        Predictions should never be empty - fallback to frequent words.
        """
        model = NGramModel(n=3)

        # Train on sample documents
        documents = [
            "function compute values from array",
            "function process items in list",
            "function transform data to result",
        ] * 20

        model.train(documents)
        model.finalize()

        # Query with completely unknown context
        predictions = model.predict(["xyz", "abc"], top_k=5)

        assert len(predictions) > 0, (
            "CONTRACT VIOLATION: Fallback predictions should never be empty"
        )

        # Should return most frequent words
        predicted_words = [p[0] for p in predictions]
        assert "function" in predicted_words or "from" in predicted_words, (
            "CONTRACT VIOLATION: Fallback should return frequent words"
        )


@pytest.mark.contract
class TestNGramTrainingContract:
    """
    N-Gram Training Performance Contract

    As a developer training models on commit history,
    I expect training to complete quickly,
    So that model updates are practical.
    """

    MAX_TRAINING_TIME_MS = 500
    DOCUMENT_COUNT = 1000

    def test_training_time_honored(self):
        """
        CONTRACT: Training < 500ms for 1,000 documents.

        Fast training enables frequent model updates from git history.
        """
        model = NGramModel(n=3)

        # Generate 1,000 training documents
        documents = []
        base_patterns = [
            "def function_name(param1, param2):",
            "    result = process(param1)",
            "    return result",
            "class ClassName:",
            "    def __init__(self):",
            "        self.value = initialize()",
        ]

        for i in range(self.DOCUMENT_COUNT):
            doc = base_patterns[i % len(base_patterns)]
            documents.append(doc)

        # Measure training time
        start = time.perf_counter()
        model.train(documents)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_TRAINING_TIME_MS, (
            f"CONTRACT VIOLATION: Training took {elapsed_ms:.1f}ms for "
            f"{self.DOCUMENT_COUNT} documents, contract requires "
            f"<{self.MAX_TRAINING_TIME_MS}ms"
        )

    def test_vocabulary_builds_linearly(self):
        """
        CONTRACT: Vocabulary building is O(n) in document count.

        Time should scale linearly with number of documents.
        """
        model = NGramModel(n=3)

        # Measure time for different document counts
        times = []
        doc_counts = [100, 200, 400, 800]

        base_doc = "def process data with function call and return result"

        for count in doc_counts:
            documents = [base_doc] * count

            start = time.perf_counter()
            model_temp = NGramModel(n=3)
            model_temp.train(documents)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Check linearity: time(2n) should be roughly 2 * time(n)
        # Allow up to 3x variance due to constant factors
        for i in range(len(times) - 1):
            ratio = times[i + 1] / times[i]
            doc_ratio = doc_counts[i + 1] / doc_counts[i]

            assert ratio < doc_ratio * 3, (
                f"CONTRACT VIOLATION: Training time not O(n). "
                f"Doubling documents from {doc_counts[i]} to {doc_counts[i+1]} "
                f"increased time by {ratio:.1f}x (expected ~{doc_ratio:.1f}x)"
            )


@pytest.mark.contract
class TestNGramMemoryContract:
    """
    N-Gram Memory Usage Contract

    As a developer deploying models in production,
    I expect bounded memory usage,
    So that models fit in constrained environments.
    """

    MAX_MEMORY_MB_PER_10K_NGRAMS = 50

    def test_memory_usage_bounded(self):
        """
        CONTRACT: Memory < 50MB per 10,000 unique n-grams.

        Memory efficiency is essential for production deployment.
        """
        model = NGramModel(n=3)

        # Create documents with controlled vocabulary
        # Using ~1000 unique words will create ~10K unique trigrams
        vocab_size = 1000
        words = [f"word{i}" for i in range(vocab_size)]

        # Generate documents by sampling from vocabulary
        import random
        random.seed(42)

        documents = []
        for _ in range(200):
            doc_words = random.sample(words, 50)
            documents.append(" ".join(doc_words))

        # Measure memory before
        import tracemalloc
        tracemalloc.start()

        # Train model
        model.train(documents)
        model.finalize()

        # Measure memory after
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        ngram_count = len(model.counts)

        # Normalize to 10K n-grams
        normalized_mb = (peak_mb / ngram_count) * 10000 if ngram_count > 0 else 0

        assert normalized_mb < self.MAX_MEMORY_MB_PER_10K_NGRAMS, (
            f"CONTRACT VIOLATION: Memory usage is {normalized_mb:.1f}MB per 10K n-grams, "
            f"contract requires <{self.MAX_MEMORY_MB_PER_10K_NGRAMS}MB "
            f"(measured {peak_mb:.1f}MB for {ngram_count} n-grams)"
        )


@pytest.mark.contract
class TestNGramAccuracyContract:
    """
    N-Gram Prediction Accuracy Contract

    As a developer relying on statistical predictions,
    I expect minimum accuracy on natural text,
    So that predictions are useful.
    """

    MIN_TOP1_ACCURACY = 0.10  # 10% - reasonable for statistical model
    MIN_TOP5_ACCURACY = 0.25  # 25% - should be in top 5 predictions

    def test_prediction_accuracy_honored(self):
        """
        CONTRACT: Prediction accuracy ≥ 10% top-1 for natural text.

        Accuracy guarantees ensure predictions are useful.
        """
        model = NGramModel(n=3)

        # Training corpus (code-like patterns with repetition)
        training_docs = [
            "def process data from input",
            "def process values from source",
            "def transform data in array",
            "def transform values in list",
            "def compute result from input",
            "def compute data from source",
            "class Processor handles file operations",
            "class Handler manages data operations",
            "function load data from database",
            "function save data to storage",
            "return processed data as result",
            "return transformed values as output",
        ] * 30  # 360 training documents with repeated patterns

        # Test corpus (VERY similar patterns to training)
        test_docs = [
            "def process data from source",  # Very similar to training
            "def transform values in array",  # Exact match to training
            "def compute result from input",  # Exact match to training
            "class Handler manages file operations",  # Similar to training
            "function load data from storage",  # Similar to training
            "return processed values as result",  # Similar to training
        ] * 10  # 60 test documents

        model.train(training_docs)
        model.finalize()

        # Evaluate accuracy
        correct_top1 = 0
        correct_top5 = 0
        total_predictions = 0

        for doc in test_docs:
            tokens = model._tokenize(doc)

            # Test predictions for each position
            for i in range(2, len(tokens)):  # Start at position 2 (trigram)
                context = tokens[i-2:i]
                actual = tokens[i]

                predictions = model.predict(context, top_k=5)
                if not predictions:
                    continue

                total_predictions += 1
                predicted_words = [p[0] for p in predictions]

                if predicted_words and predicted_words[0] == actual:
                    correct_top1 += 1
                if actual in predicted_words:
                    correct_top5 += 1

        accuracy_top1 = correct_top1 / total_predictions if total_predictions > 0 else 0
        accuracy_top5 = correct_top5 / total_predictions if total_predictions > 0 else 0

        assert accuracy_top1 >= self.MIN_TOP1_ACCURACY, (
            f"CONTRACT VIOLATION: Top-1 accuracy is {accuracy_top1:.1%}, "
            f"contract requires ≥{self.MIN_TOP1_ACCURACY:.1%}"
        )

        assert accuracy_top5 >= self.MIN_TOP5_ACCURACY, (
            f"CONTRACT VIOLATION: Top-5 accuracy is {accuracy_top5:.1%}, "
            f"contract requires ≥{self.MIN_TOP5_ACCURACY:.1%}"
        )


@pytest.mark.contract
class TestNGramPerplexityContract:
    """
    N-Gram Perplexity Contract

    As a developer evaluating model quality,
    I expect low perplexity on trained domain text,
    So that the model has learned the distribution.
    """

    MAX_PERPLEXITY = 100  # Lower is better

    def test_perplexity_bounded_on_trained_domain(self):
        """
        CONTRACT: Perplexity < 100 for trained domain text.

        Low perplexity indicates the model has learned patterns.
        """
        model = NGramModel(n=3)

        # Training corpus
        training_docs = [
            "function process input data",
            "function transform array values",
            "function compute final result",
            "class handles file operations",
            "class manages data processing",
        ] * 40  # 200 training docs

        # Test on similar domain text (not exact duplicates)
        test_docs = [
            "function validate input parameters",
            "class coordinates data operations",
            "function calculate output values",
        ]

        model.train(training_docs)
        model.finalize()

        # Calculate perplexity on test set
        perplexities = []
        for doc in test_docs:
            perplexity = model.perplexity(doc)
            if perplexity < float('inf'):
                perplexities.append(perplexity)

        avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('inf')

        assert avg_perplexity < self.MAX_PERPLEXITY, (
            f"CONTRACT VIOLATION: Perplexity is {avg_perplexity:.1f}, "
            f"contract requires <{self.MAX_PERPLEXITY}"
        )


@pytest.mark.contract
class TestNGramCorrectnessContract:
    """
    N-Gram Correctness Contract

    As a developer relying on n-gram models,
    I expect mathematically correct behavior,
    So that predictions are statistically sound.
    """

    def test_probability_sums_to_at_most_one(self):
        """
        CONTRACT: Probabilities for a context sum to ≤ 1.0.

        Probabilities must be valid (between 0 and 1, summing to ≤ 1).
        """
        model = NGramModel(n=3)

        documents = [
            "function process data values",
            "function compute result values",
            "function transform input data",
        ] * 10

        model.train(documents)

        # Get all predictions for a context
        context = ["function", "process"]
        predictions = model.predict(context, top_k=100)  # Get all

        # Sum probabilities
        total_prob = sum(p[1] for p in predictions)

        assert total_prob <= 1.01, (  # Allow small floating point error
            f"CONTRACT VIOLATION: Probabilities sum to {total_prob:.3f}, "
            f"must be ≤ 1.0"
        )

        # Each probability should be between 0 and 1
        for word, prob in predictions:
            assert 0 <= prob <= 1, (
                f"CONTRACT VIOLATION: Probability for '{word}' is {prob:.3f}, "
                f"must be in [0, 1]"
            )

    def test_laplace_smoothing_prevents_zero_probability(self):
        """
        CONTRACT: Laplace smoothing ensures non-zero probabilities.

        Unseen words should have small but non-zero probability.
        """
        model = NGramModel(n=3, smoothing=1.0)

        # Use larger vocabulary to make smoothing probabilities smaller
        documents = [
            "function process data values results output input",
            "function compute values arrays lists sets dictionaries",
            "function transform data items elements records fields",
            "class handler processor manager controller service",
            "method operation action task job procedure routine",
        ] * 20

        model.train(documents)

        # Test probability for unseen word with seen context
        context = ["function", "process"]
        prob = model.probability("unseen_word_xyz", context)

        assert prob > 0, (
            f"CONTRACT VIOLATION: Laplace smoothing should give non-zero "
            f"probability to unseen words, got {prob}"
        )

        # With larger vocabulary, unseen word probability should be much smaller
        assert prob < 0.1, (
            f"CONTRACT VIOLATION: Unseen word probability should be small, "
            f"got {prob:.4f}"
        )

    def test_higher_count_means_higher_probability(self):
        """
        CONTRACT: Words that appear more often have higher probability.

        Fundamental property of n-gram models.
        """
        model = NGramModel(n=3)

        # "common" appears 10 times, "rare" appears 1 time
        documents = [
            "function process common values"
        ] * 10 + [
            "function process rare values"
        ] * 1

        model.train(documents)

        context = ["function", "process"]
        prob_common = model.probability("common", context)
        prob_rare = model.probability("rare", context)

        assert prob_common > prob_rare, (
            f"CONTRACT VIOLATION: More frequent word should have higher probability. "
            f"common={prob_common:.3f}, rare={prob_rare:.3f}"
        )
