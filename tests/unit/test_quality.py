"""
Unit tests for quality evaluation module.

Tests prediction accuracy, search quality comparison, and alignment acceleration.
"""

import unittest
from typing import Dict, List, Set, Tuple

from cortical.spark.ngram import NGramModel
from cortical.spark.quality import (
    QualityEvaluator,
    SearchQualityEvaluator,
    AlignmentEvaluator,
    PredictionMetrics,
    SearchMetrics,
    SearchComparison,
    AlignmentMetrics,
    generate_test_queries,
    generate_relevance_judgments,
)


class TestPredictionMetrics(unittest.TestCase):
    """Test PredictionMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PredictionMetrics(
            accuracy_at_1=0.2,
            accuracy_at_5=0.5,
            accuracy_at_10=0.7,
            mean_reciprocal_rank=0.4,
            perplexity=100.0,
            coverage=0.9,
            total_tests=100,
            successful_predictions=90,
        )

        d = metrics.to_dict()
        self.assertEqual(d['accuracy_at_1'], 0.2)
        self.assertEqual(d['accuracy_at_5'], 0.5)
        self.assertEqual(d['accuracy_at_10'], 0.7)
        self.assertEqual(d['mean_reciprocal_rank'], 0.4)
        self.assertEqual(d['perplexity'], 100.0)
        self.assertEqual(d['coverage'], 0.9)
        self.assertEqual(d['total_tests'], 100)
        self.assertEqual(d['successful_predictions'], 90)

    def test_summary(self):
        """Test human-readable summary."""
        metrics = PredictionMetrics(
            accuracy_at_1=0.2,
            accuracy_at_5=0.5,
            accuracy_at_10=0.7,
            mean_reciprocal_rank=0.4,
            perplexity=100.0,
            coverage=0.9,
            total_tests=100,
            successful_predictions=90,
        )

        summary = metrics.summary()
        self.assertIn('Accuracy@1', summary)
        self.assertIn('Accuracy@5', summary)
        self.assertIn('MRR', summary)
        self.assertIn('Perplexity', summary)


class TestSearchMetrics(unittest.TestCase):
    """Test SearchMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = SearchMetrics(
            precision_at_5=0.6,
            recall_at_5=0.4,
            ndcg_at_5=0.5,
            mrr=0.7,
            queries_tested=20,
            avg_results=4.5,
        )

        d = metrics.to_dict()
        self.assertEqual(d['precision_at_5'], 0.6)
        self.assertEqual(d['recall_at_5'], 0.4)
        self.assertEqual(d['mrr'], 0.7)


class TestSearchComparison(unittest.TestCase):
    """Test SearchComparison dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        baseline = SearchMetrics(
            precision_at_5=0.5,
            recall_at_5=0.3,
            ndcg_at_5=0.4,
            mrr=0.6,
            queries_tested=20,
            avg_results=4.0,
        )

        with_spark = SearchMetrics(
            precision_at_5=0.6,
            recall_at_5=0.4,
            ndcg_at_5=0.5,
            mrr=0.7,
            queries_tested=20,
            avg_results=4.5,
        )

        comparison = SearchComparison(
            baseline=baseline,
            with_spark=with_spark,
            precision_improvement=0.2,
            recall_improvement=0.33,
            mrr_improvement=0.17,
            queries_improved=10,
            queries_same=8,
            queries_regressed=2,
        )

        d = comparison.to_dict()
        self.assertIn('baseline', d)
        self.assertIn('with_spark', d)
        self.assertEqual(d['precision_improvement'], 0.2)
        self.assertEqual(d['queries_improved'], 10)

    def test_summary(self):
        """Test human-readable summary."""
        baseline = SearchMetrics(
            precision_at_5=0.5,
            recall_at_5=0.3,
            ndcg_at_5=0.4,
            mrr=0.6,
            queries_tested=20,
            avg_results=4.0,
        )

        with_spark = SearchMetrics(
            precision_at_5=0.6,
            recall_at_5=0.4,
            ndcg_at_5=0.5,
            mrr=0.7,
            queries_tested=20,
            avg_results=4.5,
        )

        comparison = SearchComparison(
            baseline=baseline,
            with_spark=with_spark,
            precision_improvement=0.2,
            recall_improvement=0.33,
            mrr_improvement=0.17,
            queries_improved=10,
            queries_same=8,
            queries_regressed=2,
        )

        summary = comparison.summary()
        self.assertIn('Baseline Precision', summary)
        self.assertIn('With Spark Precision', summary)
        self.assertIn('improved', summary)


class TestAlignmentMetrics(unittest.TestCase):
    """Test AlignmentMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AlignmentMetrics(
            baseline_rounds=3.5,
            with_alignment_rounds=2.0,
            reduction_percent=0.43,
            sessions_tested=10,
            sessions_improved=7,
            sessions_same=3,
            avg_alignment_hits=1.5,
        )

        d = metrics.to_dict()
        self.assertEqual(d['baseline_rounds'], 3.5)
        self.assertEqual(d['reduction_percent'], 0.43)
        self.assertEqual(d['sessions_improved'], 7)

    def test_summary(self):
        """Test human-readable summary."""
        metrics = AlignmentMetrics(
            baseline_rounds=3.5,
            with_alignment_rounds=2.0,
            reduction_percent=0.43,
            sessions_tested=10,
            sessions_improved=7,
            sessions_same=3,
            avg_alignment_hits=1.5,
        )

        summary = metrics.summary()
        self.assertIn('Baseline Rounds', summary)
        self.assertIn('With Alignment', summary)
        self.assertIn('Reduction', summary)


class TestQualityEvaluator(unittest.TestCase):
    """Test QualityEvaluator class."""

    def setUp(self):
        self.model = NGramModel(n=3)
        # Train on predictable content
        for _ in range(10):
            self.model.train(["the quick brown fox jumps over the lazy dog"])
            self.model.train(["the quick brown fox runs through the forest"])
            self.model.train(["a quick brown cat chases the mouse"])

        self.evaluator = QualityEvaluator(self.model)

    def test_init(self):
        """Test initialization."""
        evaluator = QualityEvaluator(self.model)
        self.assertIs(evaluator.model, self.model)

    def test_evaluate_predictions_returns_metrics(self):
        """Test evaluate_predictions returns PredictionMetrics."""
        test_texts = ["the quick brown fox"]
        metrics = self.evaluator.evaluate_predictions(test_texts)
        self.assertIsInstance(metrics, PredictionMetrics)

    def test_evaluate_predictions_accuracy_at_1(self):
        """Test accuracy@1 calculation."""
        # Use text that the model has seen
        test_texts = ["the quick brown fox jumps"]
        metrics = self.evaluator.evaluate_predictions(test_texts)

        # Should have some accuracy
        self.assertGreaterEqual(metrics.accuracy_at_1, 0.0)
        self.assertLessEqual(metrics.accuracy_at_1, 1.0)

    def test_evaluate_predictions_accuracy_at_5(self):
        """Test accuracy@5 is at least as good as accuracy@1."""
        test_texts = ["the quick brown fox jumps"]
        metrics = self.evaluator.evaluate_predictions(test_texts)

        self.assertGreaterEqual(metrics.accuracy_at_5, metrics.accuracy_at_1)

    def test_evaluate_predictions_accuracy_at_10(self):
        """Test accuracy@10 is at least as good as accuracy@5."""
        test_texts = ["the quick brown fox jumps"]
        metrics = self.evaluator.evaluate_predictions(test_texts)

        self.assertGreaterEqual(metrics.accuracy_at_10, metrics.accuracy_at_5)

    def test_evaluate_predictions_mrr(self):
        """Test MRR is in valid range."""
        test_texts = ["the quick brown fox jumps"]
        metrics = self.evaluator.evaluate_predictions(test_texts)

        self.assertGreaterEqual(metrics.mean_reciprocal_rank, 0.0)
        self.assertLessEqual(metrics.mean_reciprocal_rank, 1.0)

    def test_evaluate_predictions_perplexity(self):
        """Test perplexity is calculated."""
        test_texts = ["the quick brown fox jumps"]
        metrics = self.evaluator.evaluate_predictions(test_texts)

        self.assertGreater(metrics.perplexity, 0.0)

    def test_evaluate_predictions_coverage(self):
        """Test coverage is in valid range."""
        test_texts = ["the quick brown fox jumps"]
        metrics = self.evaluator.evaluate_predictions(test_texts)

        self.assertGreaterEqual(metrics.coverage, 0.0)
        self.assertLessEqual(metrics.coverage, 1.0)

    def test_evaluate_predictions_empty_input(self):
        """Test with empty input."""
        metrics = self.evaluator.evaluate_predictions([])

        self.assertEqual(metrics.total_tests, 0)
        self.assertEqual(metrics.accuracy_at_1, 0.0)

    def test_evaluate_predictions_short_text(self):
        """Test with text too short for context."""
        metrics = self.evaluator.evaluate_predictions(["a"])

        # Should handle gracefully
        self.assertEqual(metrics.total_tests, 0)

    def test_create_held_out_split(self):
        """Test train/test split."""
        texts = [f"text {i}" for i in range(100)]
        train, test = self.evaluator.create_held_out_split(texts, test_ratio=0.2)

        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_create_held_out_split_reproducible(self):
        """Test split is reproducible with seed."""
        texts = [f"text {i}" for i in range(100)]

        train1, test1 = self.evaluator.create_held_out_split(texts, seed=42)
        train2, test2 = self.evaluator.create_held_out_split(texts, seed=42)

        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)

    def test_cross_validate_predictions(self):
        """Test cross-validation."""
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown cat chases the mouse",
            "the dog runs after the cat",
            "quick foxes are brown animals",
            "lazy dogs sleep all day",
        ]

        results = self.evaluator.cross_validate_predictions(texts, folds=2)

        self.assertEqual(len(results), 2)
        for metrics in results:
            self.assertIsInstance(metrics, PredictionMetrics)

    def test_measure_perplexity_stability(self):
        """Test perplexity stability measurement."""
        texts = ["the quick brown fox jumps over the lazy dog"]

        result = self.evaluator.measure_perplexity_stability(texts, runs=3)

        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('min', result)
        self.assertIn('max', result)
        self.assertIn('is_stable', result)

    def test_measure_perplexity_stability_is_stable(self):
        """Test perplexity is stable across runs."""
        texts = ["the quick brown fox jumps over the lazy dog"]

        result = self.evaluator.measure_perplexity_stability(texts, runs=5)

        # Same input should give same perplexity
        self.assertTrue(result['is_stable'])


class TestSearchQualityEvaluator(unittest.TestCase):
    """Test SearchQualityEvaluator class."""

    def setUp(self):
        # Create simple mock search functions
        self.docs = {
            'doc1': 'authentication login password',
            'doc2': 'user management profiles',
            'doc3': 'authentication tokens jwt',
            'doc4': 'database queries sql',
            'doc5': 'api endpoints rest',
        }

        def search_fn(query: str) -> List[Tuple[str, float]]:
            """Simple keyword search."""
            query_words = set(query.lower().split())
            results = []
            for doc_id, content in self.docs.items():
                doc_words = set(content.lower().split())
                score = len(query_words & doc_words)
                if score > 0:
                    results.append((doc_id, float(score)))
            return sorted(results, key=lambda x: -x[1])

        def search_with_spark_fn(query: str) -> List[Tuple[str, float]]:
            """Enhanced search with bonus scores."""
            results = search_fn(query)
            # Spark boosts auth-related docs
            boosted = []
            for doc_id, score in results:
                if 'auth' in self.docs[doc_id]:
                    score += 0.5
                boosted.append((doc_id, score))
            return sorted(boosted, key=lambda x: -x[1])

        self.search_fn = search_fn
        self.search_with_spark_fn = search_with_spark_fn
        self.evaluator = SearchQualityEvaluator(search_fn, search_with_spark_fn)

    def test_init(self):
        """Test initialization."""
        evaluator = SearchQualityEvaluator(
            self.search_fn,
            self.search_with_spark_fn
        )
        self.assertIsNotNone(evaluator.search_fn)
        self.assertIsNotNone(evaluator.search_with_spark_fn)

    def test_evaluate_search_returns_metrics(self):
        """Test evaluate_search returns SearchMetrics."""
        queries = ['authentication login']
        relevance = {'authentication login': {'doc1', 'doc3'}}

        metrics = self.evaluator.evaluate_search(queries, relevance)

        self.assertIsInstance(metrics, SearchMetrics)

    def test_evaluate_search_precision(self):
        """Test precision calculation."""
        queries = ['authentication']
        relevance = {'authentication': {'doc1', 'doc3'}}

        metrics = self.evaluator.evaluate_search(queries, relevance, k=5)

        self.assertGreaterEqual(metrics.precision_at_5, 0.0)
        self.assertLessEqual(metrics.precision_at_5, 1.0)

    def test_evaluate_search_recall(self):
        """Test recall calculation."""
        queries = ['authentication']
        relevance = {'authentication': {'doc1', 'doc3'}}

        metrics = self.evaluator.evaluate_search(queries, relevance, k=5)

        self.assertGreaterEqual(metrics.recall_at_5, 0.0)
        self.assertLessEqual(metrics.recall_at_5, 1.0)

    def test_evaluate_search_mrr(self):
        """Test MRR calculation."""
        queries = ['authentication']
        relevance = {'authentication': {'doc1', 'doc3'}}

        metrics = self.evaluator.evaluate_search(queries, relevance, k=5)

        self.assertGreaterEqual(metrics.mrr, 0.0)
        self.assertLessEqual(metrics.mrr, 1.0)

    def test_evaluate_search_no_relevance(self):
        """Test with no relevance judgments."""
        queries = ['unknown query']
        relevance = {}

        metrics = self.evaluator.evaluate_search(queries, relevance, k=5)

        # Should handle gracefully
        self.assertEqual(metrics.queries_tested, 1)

    def test_compare_search_returns_comparison(self):
        """Test compare_search returns SearchComparison."""
        queries = ['authentication login']
        relevance = {'authentication login': {'doc1', 'doc3'}}

        comparison = self.evaluator.compare_search(queries, relevance)

        self.assertIsInstance(comparison, SearchComparison)

    def test_compare_search_has_baseline(self):
        """Test comparison includes baseline metrics."""
        queries = ['authentication']
        relevance = {'authentication': {'doc1', 'doc3'}}

        comparison = self.evaluator.compare_search(queries, relevance)

        self.assertIsInstance(comparison.baseline, SearchMetrics)
        self.assertIsInstance(comparison.with_spark, SearchMetrics)

    def test_compare_search_improvements(self):
        """Test improvement calculations."""
        queries = ['authentication']
        relevance = {'authentication': {'doc1', 'doc3'}}

        comparison = self.evaluator.compare_search(queries, relevance)

        # Just check they're calculated
        self.assertIsInstance(comparison.precision_improvement, float)
        self.assertIsInstance(comparison.recall_improvement, float)
        self.assertIsInstance(comparison.mrr_improvement, float)

    def test_compare_search_counts(self):
        """Test improved/same/regressed counts."""
        queries = ['authentication', 'database']
        relevance = {
            'authentication': {'doc1', 'doc3'},
            'database': {'doc4'},
        }

        comparison = self.evaluator.compare_search(queries, relevance)

        total = comparison.queries_improved + comparison.queries_same + comparison.queries_regressed
        self.assertEqual(total, 2)


class TestAlignmentEvaluator(unittest.TestCase):
    """Test AlignmentEvaluator class."""

    def setUp(self):
        # Mock alignment lookup
        self.alignment_data = {
            'auth': [{'term': 'authentication', 'definition': 'Login process'}],
            'jwt': [{'term': 'jwt', 'definition': 'JSON Web Token'}],
        }

        def alignment_lookup(query: str) -> List[Dict]:
            """Look up alignment context."""
            for key in self.alignment_data:
                if key in query.lower():
                    return self.alignment_data[key]
            return []

        def needs_disambiguation(query: str) -> bool:
            """Check if query needs clarification."""
            # Single-word queries need disambiguation
            return len(query.split()) <= 2

        self.alignment_lookup = alignment_lookup
        self.needs_disambiguation = needs_disambiguation
        self.evaluator = AlignmentEvaluator(alignment_lookup, needs_disambiguation)

    def test_init(self):
        """Test initialization."""
        evaluator = AlignmentEvaluator(
            self.alignment_lookup,
            self.needs_disambiguation
        )
        self.assertIsNotNone(evaluator.alignment_lookup)
        self.assertIsNotNone(evaluator.needs_disambiguation)

    def test_simulate_session_returns_dict(self):
        """Test simulate_session returns dict."""
        queries = ['auth', 'user login']

        result = self.evaluator.simulate_session(queries)

        self.assertIsInstance(result, dict)
        self.assertIn('total_queries', result)
        self.assertIn('disambiguation_rounds', result)
        self.assertIn('alignment_hits', result)

    def test_simulate_session_with_alignment(self):
        """Test session with alignment enabled."""
        queries = ['auth', 'jwt token']

        result = self.evaluator.simulate_session(queries, use_alignment=True)

        # 'auth' should get alignment hit
        self.assertGreater(result['alignment_hits'], 0)

    def test_simulate_session_without_alignment(self):
        """Test session without alignment."""
        queries = ['auth', 'jwt token']

        result = self.evaluator.simulate_session(queries, use_alignment=False)

        # No alignment hits without alignment
        self.assertEqual(result['alignment_hits'], 0)

    def test_simulate_session_disambiguation_reduction(self):
        """Test alignment reduces disambiguation rounds."""
        queries = ['auth', 'jwt']

        with_align = self.evaluator.simulate_session(queries, use_alignment=True)
        without_align = self.evaluator.simulate_session(queries, use_alignment=False)

        # With alignment should have fewer or equal rounds
        self.assertLessEqual(
            with_align['disambiguation_rounds'],
            without_align['disambiguation_rounds']
        )

    def test_evaluate_acceleration_returns_metrics(self):
        """Test evaluate_acceleration returns AlignmentMetrics."""
        sessions = [
            ['auth', 'login user'],
            ['jwt token', 'verify'],
        ]

        metrics = self.evaluator.evaluate_acceleration(sessions)

        self.assertIsInstance(metrics, AlignmentMetrics)

    def test_evaluate_acceleration_reduction(self):
        """Test acceleration shows reduction."""
        sessions = [
            ['auth', 'login'],  # 'auth' should get alignment help
            ['jwt', 'verify'],  # 'jwt' should get alignment help
        ]

        metrics = self.evaluator.evaluate_acceleration(sessions)

        # Should show some reduction
        self.assertGreaterEqual(metrics.reduction_percent, 0.0)

    def test_evaluate_acceleration_session_counts(self):
        """Test session counts are correct."""
        sessions = [
            ['auth', 'login'],
            ['jwt', 'verify'],
            ['database query', 'select'],
        ]

        metrics = self.evaluator.evaluate_acceleration(sessions)

        self.assertEqual(metrics.sessions_tested, 3)
        total = metrics.sessions_improved + metrics.sessions_same
        self.assertEqual(total, 3)


class TestGenerateTestQueries(unittest.TestCase):
    """Test generate_test_queries function."""

    def test_generates_queries(self):
        """Test query generation."""
        vocab = {'word1', 'word2', 'word3', 'word4', 'word5'}

        queries = generate_test_queries(vocab, count=10)

        self.assertEqual(len(queries), 10)

    def test_queries_use_vocabulary(self):
        """Test queries only use vocabulary words."""
        vocab = {'alpha', 'beta', 'gamma', 'delta', 'epsilon'}

        queries = generate_test_queries(vocab, count=5)

        for query in queries:
            words = query.split()
            for word in words:
                self.assertIn(word, vocab)

    def test_queries_respect_length(self):
        """Test queries respect min/max words."""
        vocab = {'word1', 'word2', 'word3', 'word4', 'word5', 'word6'}

        queries = generate_test_queries(vocab, count=20, min_words=2, max_words=4)

        for query in queries:
            word_count = len(query.split())
            self.assertGreaterEqual(word_count, 2)
            self.assertLessEqual(word_count, 4)

    def test_queries_reproducible(self):
        """Test queries are reproducible with seed."""
        vocab = {'a', 'b', 'c', 'd', 'e'}

        queries1 = generate_test_queries(vocab, count=5, seed=42)
        queries2 = generate_test_queries(vocab, count=5, seed=42)

        self.assertEqual(queries1, queries2)


class TestGenerateRelevanceJudgments(unittest.TestCase):
    """Test generate_relevance_judgments function."""

    def test_generates_relevance(self):
        """Test relevance generation."""
        queries = ['query1', 'query2', 'query3']
        doc_ids = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']

        relevance = generate_relevance_judgments(queries, doc_ids)

        self.assertEqual(len(relevance), 3)

    def test_relevance_uses_doc_ids(self):
        """Test relevance only uses provided doc IDs."""
        queries = ['query1', 'query2']
        doc_ids = ['doc1', 'doc2', 'doc3']

        relevance = generate_relevance_judgments(queries, doc_ids)

        for query, relevant_docs in relevance.items():
            for doc_id in relevant_docs:
                self.assertIn(doc_id, doc_ids)

    def test_relevance_is_set(self):
        """Test relevance values are sets."""
        queries = ['query1']
        doc_ids = ['doc1', 'doc2', 'doc3']

        relevance = generate_relevance_judgments(queries, doc_ids)

        for relevant_docs in relevance.values():
            self.assertIsInstance(relevant_docs, set)

    def test_relevance_reproducible(self):
        """Test relevance is reproducible with seed."""
        queries = ['query1', 'query2']
        doc_ids = ['doc1', 'doc2', 'doc3', 'doc4']

        rel1 = generate_relevance_judgments(queries, doc_ids, seed=42)
        rel2 = generate_relevance_judgments(queries, doc_ids, seed=42)

        self.assertEqual(rel1, rel2)


class TestQualityEvaluatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_model(self):
        """Test with untrained model."""
        model = NGramModel(n=3)
        evaluator = QualityEvaluator(model)

        metrics = evaluator.evaluate_predictions(["test text here"])

        # Should handle gracefully
        self.assertIsInstance(metrics, PredictionMetrics)

    def test_single_text(self):
        """Test with single text."""
        model = NGramModel(n=3)
        model.train(["the quick brown fox jumps over the lazy dog"])

        evaluator = QualityEvaluator(model)
        metrics = evaluator.evaluate_predictions(["the quick brown fox"])

        self.assertIsInstance(metrics, PredictionMetrics)

    def test_cross_validate_single_fold(self):
        """Test cross-validation with 1 fold (edge case)."""
        model = NGramModel(n=3)
        model.train(["the quick brown fox"])

        evaluator = QualityEvaluator(model)
        results = evaluator.cross_validate_predictions(
            ["text one", "text two", "text three"],
            folds=1
        )

        self.assertEqual(len(results), 1)


class TestImports(unittest.TestCase):
    """Test that all quality exports are importable."""

    def test_import_from_spark(self):
        """Test imports from spark package."""
        from cortical.spark import (
            QualityEvaluator,
            SearchQualityEvaluator,
            AlignmentEvaluator,
            PredictionMetrics,
            SearchMetrics,
            SearchComparison,
            AlignmentMetrics,
            generate_test_queries,
            generate_relevance_judgments,
        )

        # All imports successful
        self.assertIsNotNone(QualityEvaluator)
        self.assertIsNotNone(SearchQualityEvaluator)
        self.assertIsNotNone(AlignmentEvaluator)
        self.assertIsNotNone(PredictionMetrics)
        self.assertIsNotNone(SearchMetrics)
        self.assertIsNotNone(SearchComparison)
        self.assertIsNotNone(AlignmentMetrics)
        self.assertIsNotNone(generate_test_queries)
        self.assertIsNotNone(generate_relevance_judgments)


if __name__ == '__main__':
    unittest.main()
