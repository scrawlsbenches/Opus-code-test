#!/usr/bin/env python3
"""
Unit tests for MoE Foundation classes.

Tests the base classes for the Mixture of Experts system:
- MicroExpert base class
- ExpertRouter
- VotingAggregator
- FileExpert
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import sys
# Add scripts/hubris to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "hubris"))

from micro_expert import (
    MicroExpert, ExpertMetrics, ExpertPrediction, AggregatedPrediction
)
from expert_router import ExpertRouter, RoutingDecision
from voting_aggregator import VotingAggregator, AggregationConfig
from experts.file_expert import FileExpert


# Test implementation of MicroExpert for testing abstract base class
class DummyExpert(MicroExpert):
    """Concrete test implementation of MicroExpert."""

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """Simple test prediction."""
        query = context.get('query', '')
        items = [
            (f'item_{i}', 1.0 - i * 0.1)
            for i in range(context.get('num_items', 5))
        ]
        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=items,
            metadata={'query': query}
        )


class TestExpertMetrics(unittest.TestCase):
    """Test ExpertMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = ExpertMetrics(
            mrr=0.5,
            recall_at_k={1: 0.3, 5: 0.6, 10: 0.8},
            precision_at_k={1: 0.4, 5: 0.5},
            calibration_error=0.05,
            test_examples=100,
            last_evaluated="2025-12-17T12:00:00"
        )

        self.assertEqual(metrics.mrr, 0.5)
        self.assertEqual(metrics.recall_at_k[5], 0.6)
        self.assertEqual(metrics.test_examples, 100)

    def test_to_dict(self):
        """Test conversion to dict."""
        metrics = ExpertMetrics(mrr=0.5, test_examples=100)
        d = metrics.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d['mrr'], 0.5)
        self.assertEqual(d['test_examples'], 100)

    def test_from_dict(self):
        """Test loading from dict."""
        data = {
            'mrr': 0.5,
            'recall_at_k': {1: 0.3},
            'precision_at_k': {1: 0.4},
            'calibration_error': 0.05,
            'test_examples': 100,
            'last_evaluated': '2025-12-17'
        }

        metrics = ExpertMetrics.from_dict(data)
        self.assertEqual(metrics.mrr, 0.5)
        self.assertEqual(metrics.recall_at_k[1], 0.3)


class TestExpertPrediction(unittest.TestCase):
    """Test ExpertPrediction dataclass."""

    def test_create_prediction(self):
        """Test creating prediction."""
        pred = ExpertPrediction(
            expert_id='test_expert',
            expert_type='test',
            items=[('file1.py', 0.9), ('file2.py', 0.7)],
            metadata={'query': 'test query'}
        )

        self.assertEqual(pred.expert_id, 'test_expert')
        self.assertEqual(len(pred.items), 2)
        self.assertEqual(pred.items[0], ('file1.py', 0.9))

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        pred = ExpertPrediction(
            expert_id='test',
            expert_type='file',
            items=[('a.py', 0.8)],
            metadata={'key': 'value'}
        )

        d = pred.to_dict()
        loaded = ExpertPrediction.from_dict(d)

        self.assertEqual(loaded.expert_id, pred.expert_id)
        self.assertEqual(loaded.items, pred.items)
        self.assertEqual(loaded.metadata, pred.metadata)


class TestMicroExpert(unittest.TestCase):
    """Test MicroExpert base class."""

    def test_create_expert(self):
        """Test creating expert instance."""
        expert = DummyExpert(
            expert_id='test_123',
            expert_type='test',
            version='1.0.0',
            trained_on_commits=100
        )

        self.assertEqual(expert.expert_id, 'test_123')
        self.assertEqual(expert.expert_type, 'test')
        self.assertEqual(expert.trained_on_commits, 100)

    def test_predict_abstract(self):
        """Test predict method works in concrete implementation."""
        expert = DummyExpert(
            expert_id='test',
            expert_type='test'
        )

        pred = expert.predict({'query': 'test', 'num_items': 3})

        self.assertIsInstance(pred, ExpertPrediction)
        self.assertEqual(len(pred.items), 3)
        self.assertEqual(pred.items[0][0], 'item_0')

    def test_save_load(self):
        """Test saving and loading expert."""
        expert = DummyExpert(
            expert_id='test',
            expert_type='test',
            version='1.0.0',
            model_data={'key': 'value'}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'expert.json'
            expert.save(path)

            # Verify file exists and contains JSON
            self.assertTrue(path.exists())
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data['expert_id'], 'test')
            self.assertEqual(data['model_data']['key'], 'value')

            # Load and verify
            loaded = DummyExpert.load(path)
            self.assertEqual(loaded.expert_id, expert.expert_id)
            self.assertEqual(loaded.model_data, expert.model_data)

    def test_update_metrics(self):
        """Test updating expert metrics."""
        expert = DummyExpert(expert_id='test', expert_type='test')
        self.assertIsNone(expert.metrics)

        metrics = ExpertMetrics(mrr=0.7, test_examples=50)
        expert.update_metrics(metrics)

        self.assertIsNotNone(expert.metrics)
        self.assertEqual(expert.metrics.mrr, 0.7)

    def test_calibration(self):
        """Test confidence calibration."""
        expert = DummyExpert(
            expert_id='test',
            expert_type='test',
            calibration_curve=[(0.0, 0.0), (0.5, 0.4), (1.0, 0.9)]
        )

        # Test interpolation
        calibrated = expert.get_confidence_calibration(0.5)
        self.assertAlmostEqual(calibrated, 0.4)

        # Test edge cases
        self.assertEqual(expert.get_confidence_calibration(0.0), 0.0)
        self.assertEqual(expert.get_confidence_calibration(1.0), 0.9)

        # Test without calibration curve
        expert_uncalibrated = DummyExpert(expert_id='test', expert_type='test')
        self.assertEqual(expert_uncalibrated.get_confidence_calibration(0.7), 0.7)


class TestExpertRouter(unittest.TestCase):
    """Test ExpertRouter."""

    def test_create_router(self):
        """Test creating router."""
        router = ExpertRouter()
        self.assertIsNotNone(router)
        self.assertEqual(router.expert_weights, {})

    def test_classify_intent_debug(self):
        """Test intent classification for debugging."""
        router = ExpertRouter()

        # Debug error intent
        intent = router.classify_intent("Fix the error in authentication")
        self.assertEqual(intent, 'debug_error')

        intent = router.classify_intent("Why is this failing?")
        self.assertEqual(intent, 'debug_error')

    def test_classify_intent_feature(self):
        """Test intent classification for features."""
        router = ExpertRouter()

        intent = router.classify_intent("Add new authentication feature")
        self.assertEqual(intent, 'implement_feature')

        intent = router.classify_intent("Implement user registration")
        self.assertEqual(intent, 'implement_feature')

    def test_classify_intent_bug_fix(self):
        """Test intent classification for bug fixes."""
        router = ExpertRouter()

        intent = router.classify_intent("Fix bug in login handler")
        self.assertEqual(intent, 'fix_bug')

    def test_classify_intent_tests(self):
        """Test intent classification for tests."""
        router = ExpertRouter()

        intent = router.classify_intent("Add tests for authentication")
        self.assertEqual(intent, 'add_tests')

        intent = router.classify_intent("Write unit tests")
        self.assertEqual(intent, 'add_tests')

    def test_classify_intent_refactor(self):
        """Test intent classification for refactoring."""
        router = ExpertRouter()

        intent = router.classify_intent("Refactor the database module")
        self.assertEqual(intent, 'refactor')

    def test_classify_intent_docs(self):
        """Test intent classification for documentation."""
        router = ExpertRouter()

        intent = router.classify_intent("Update the README")
        self.assertEqual(intent, 'update_docs')

    def test_get_experts_for_feature(self):
        """Test expert selection for feature implementation."""
        router = ExpertRouter()

        experts = router.get_experts("Add new feature")
        self.assertIn('file', experts)
        self.assertIn('test', experts)
        self.assertIn('doc', experts)

    def test_get_experts_for_bug(self):
        """Test expert selection for bug fixes."""
        router = ExpertRouter()

        experts = router.get_experts("Fix error in handler")
        self.assertIn('file', experts)
        self.assertIn('error', experts)

    def test_get_experts_with_context(self):
        """Test expert selection with context."""
        router = ExpertRouter()

        # Context with error message should route to error expert
        context = {'error_message': 'AttributeError: ...'}
        experts = router.get_experts("What's wrong?", context)
        self.assertIn('error', experts)

    def test_route_full(self):
        """Test full routing decision."""
        router = ExpertRouter()

        decision = router.route("Add authentication feature")

        self.assertIsInstance(decision, RoutingDecision)
        self.assertEqual(decision.intent, 'implement_feature')
        self.assertIn('file', decision.expert_types)
        self.assertGreater(decision.confidence, 0.5)

    def test_update_weights(self):
        """Test updating routing weights."""
        router = ExpertRouter()
        router.update_weights({'file': 0.9, 'test': 0.7})

        weights = router.get_weights()
        self.assertEqual(weights['file'], 0.9)
        self.assertEqual(weights['test'], 0.7)


class TestVotingAggregator(unittest.TestCase):
    """Test VotingAggregator."""

    def test_create_aggregator(self):
        """Test creating aggregator."""
        agg = VotingAggregator()
        self.assertIsNotNone(agg.config)

    def test_aggregate_empty(self):
        """Test aggregation with no predictions."""
        agg = VotingAggregator()
        result = agg.aggregate([])

        self.assertEqual(len(result.items), 0)
        self.assertEqual(result.confidence, 0.0)

    def test_aggregate_single(self):
        """Test aggregation with single expert."""
        agg = VotingAggregator()

        pred = ExpertPrediction(
            expert_id='e1',
            expert_type='file',
            items=[('file1.py', 0.9), ('file2.py', 0.7)]
        )

        result = agg.aggregate([pred])

        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0][0], 'file1.py')
        self.assertIn('e1', result.contributing_experts)

    def test_aggregate_multiple_consensus(self):
        """Test aggregation when experts agree."""
        agg = VotingAggregator()

        pred1 = ExpertPrediction(
            expert_id='e1',
            expert_type='file',
            items=[('file1.py', 0.9), ('file2.py', 0.5)]
        )
        pred2 = ExpertPrediction(
            expert_id='e2',
            expert_type='file',
            items=[('file1.py', 0.8), ('file2.py', 0.6)]
        )

        result = agg.aggregate([pred1, pred2])

        # file1.py should be top (both experts agree)
        self.assertEqual(result.items[0][0], 'file1.py')
        # Confidence should be relatively high
        self.assertGreater(result.confidence, 0.5)
        # Low disagreement
        self.assertLess(result.disagreement_score, 0.5)

    def test_aggregate_multiple_disagreement(self):
        """Test aggregation when experts disagree."""
        agg = VotingAggregator()

        pred1 = ExpertPrediction(
            expert_id='e1',
            expert_type='file',
            items=[('file1.py', 0.9)]
        )
        pred2 = ExpertPrediction(
            expert_id='e2',
            expert_type='file',
            items=[('file2.py', 0.9)]
        )

        result = agg.aggregate([pred1, pred2])

        # Should have both items
        self.assertEqual(len(result.items), 2)
        # Higher disagreement
        self.assertGreater(result.disagreement_score, 0.5)

    def test_aggregate_with_weights(self):
        """Test aggregation with expert weights."""
        config = AggregationConfig(
            expert_weights={'e1': 2.0, 'e2': 1.0}
        )
        agg = VotingAggregator(config)

        pred1 = ExpertPrediction(
            expert_id='e1',
            expert_type='file',
            items=[('file1.py', 0.5)]
        )
        pred2 = ExpertPrediction(
            expert_id='e2',
            expert_type='file',
            items=[('file2.py', 0.9)]
        )

        result = agg.aggregate([pred1, pred2])

        # e1's file1.py should win due to higher weight (2.0 * 0.5 = 1.0 > 1.0 * 0.9)
        self.assertEqual(result.items[0][0], 'file1.py')

    def test_merge_by_rank(self):
        """Test rank-based merging (Borda count)."""
        agg = VotingAggregator()

        pred1 = ExpertPrediction(
            expert_id='e1',
            expert_type='file',
            items=[('a.py', 0.9), ('b.py', 0.8), ('c.py', 0.7)]
        )
        pred2 = ExpertPrediction(
            expert_id='e2',
            expert_type='file',
            items=[('b.py', 0.9), ('a.py', 0.8), ('c.py', 0.7)]
        )

        items = agg.merge_by_rank([pred1, pred2], top_n=3)

        # Both should rank a.py and b.py highly
        self.assertIn('a.py', items)
        self.assertIn('b.py', items)

    def test_weighted_average_confidence(self):
        """Test weighted average confidence for an item."""
        agg = VotingAggregator()

        pred1 = ExpertPrediction(
            expert_id='e1',
            expert_type='file',
            items=[('file1.py', 0.8)]
        )
        pred2 = ExpertPrediction(
            expert_id='e2',
            expert_type='file',
            items=[('file1.py', 0.6)]
        )

        avg_conf = agg.weighted_average_confidence([pred1, pred2], 'file1.py')

        # Should be average of 0.8 and 0.6 = 0.7
        self.assertAlmostEqual(avg_conf, 0.7)


class TestFileExpert(unittest.TestCase):
    """Test FileExpert."""

    def test_create_file_expert(self):
        """Test creating FileExpert."""
        expert = FileExpert()

        self.assertEqual(expert.expert_type, 'file')
        self.assertIn('file_cooccurrence', expert.model_data)

    def test_predict_empty_query(self):
        """Test prediction with empty query."""
        expert = FileExpert()
        pred = expert.predict({'query': ''})

        self.assertEqual(len(pred.items), 0)
        self.assertIn('error', pred.metadata)

    def test_predict_with_model_data(self):
        """Test prediction with model data."""
        expert = FileExpert(
            model_data={
                'file_cooccurrence': {},
                'type_to_files': {
                    'feat': {'file1.py': 5, 'file2.py': 3}
                },
                'keyword_to_files': {
                    'auth': {'file1.py': 10}
                },
                'file_frequency': {'file1.py': 15, 'file2.py': 8},
                'file_to_commits': {},
                'total_commits': 100
            }
        )

        pred = expert.predict({
            'query': 'feat: Add authentication',
            'top_n': 5
        })

        self.assertGreater(len(pred.items), 0)
        # file1.py should score highest (has both feat type and auth keyword)
        self.assertEqual(pred.items[0][0], 'file1.py')

    def test_predict_with_seed_files(self):
        """Test prediction with seed files for co-occurrence boost."""
        expert = FileExpert(
            model_data={
                'file_cooccurrence': {
                    'seed.py': {'related.py': 10}
                },
                'type_to_files': {},
                'keyword_to_files': {},
                'file_frequency': {'seed.py': 15, 'related.py': 12},
                'file_to_commits': {},
                'total_commits': 100
            }
        )

        pred = expert.predict({
            'query': 'test',
            'seed_files': ['seed.py'],
            'top_n': 5
        })

        # related.py should be boosted by co-occurrence
        self.assertGreater(len(pred.items), 0)
        file_names = [f for f, _ in pred.items]
        self.assertIn('related.py', file_names)

    def test_save_load_roundtrip(self):
        """Test saving and loading FileExpert."""
        expert = FileExpert(
            expert_id='file_test',
            version='1.0.0',
            trained_on_commits=50,
            model_data={
                'file_cooccurrence': {'a.py': {'b.py': 5}},
                'type_to_files': {},
                'keyword_to_files': {},
                'file_frequency': {},
                'file_to_commits': {},
                'total_commits': 50
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'file_expert.json'
            expert.save(path)

            loaded = FileExpert.load(path)

            self.assertEqual(loaded.expert_id, expert.expert_id)
            self.assertEqual(loaded.trained_on_commits, 50)
            self.assertIn('a.py', loaded.model_data['file_cooccurrence'])


class TestTestExpert(unittest.TestCase):
    """Test TestExpert."""

    def test_create_test_expert(self):
        """Test creating TestExpert."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert()

        self.assertEqual(expert.expert_type, 'test')
        self.assertIn('source_to_tests', expert.model_data)
        self.assertIn('test_failure_patterns', expert.model_data)

    def test_predict_by_convention(self):
        """Test prediction based on naming conventions."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert(
            model_data={
                'source_to_tests': {
                    'cortical/query/search.py': {'tests/test_query.py': 10}
                },
                'test_to_sources': {},
                'test_failure_patterns': {},
                'module_to_tests': {},
                'test_cochange': {},
                'total_commits': 100
            }
        )

        pred = expert.predict({
            'changed_files': ['cortical/query/search.py']
        })

        # Should find tests for the changed file
        self.assertGreater(len(pred.items), 0)
        test_files = [t for t, _ in pred.items]
        self.assertIn('tests/test_query.py', test_files)

    def test_predict_with_query(self):
        """Test prediction with query text."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert(
            model_data={
                'source_to_tests': {
                    'cortical/analysis.py': {'tests/test_analysis.py': 5}
                },
                'test_to_sources': {},
                'test_failure_patterns': {},
                'module_to_tests': {},
                'test_cochange': {},
                'total_commits': 50
            }
        )

        pred = expert.predict({
            'changed_files': ['cortical/analysis.py'],
            'query': 'pagerank analysis'
        })

        self.assertEqual(pred.expert_type, 'test')
        self.assertIn('scoring_signals', pred.metadata)

    def test_train_expert(self):
        """Test training TestExpert on commits."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert()

        commits = [
            {
                'files': ['cortical/query/search.py', 'tests/test_query.py'],
                'message': 'feat: Add search feature'
            },
            {
                'files': ['cortical/analysis.py', 'tests/test_analysis.py'],
                'message': 'fix: PageRank bug'
            },
            {
                'files': ['cortical/query/search.py', 'tests/test_query.py', 'tests/integration/test_search.py'],
                'message': 'test: Add integration tests'
            }
        ]

        expert.train(commits)

        # Should have learned mappings
        self.assertEqual(expert.model_data['total_commits'], 3)
        self.assertIn('cortical/query/search.py', expert.model_data['source_to_tests'])

    def test_is_test_file(self):
        """Test test file detection."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert()

        self.assertTrue(expert._is_test_file('tests/test_foo.py'))
        self.assertTrue(expert._is_test_file('test_bar.py'))
        self.assertTrue(expert._is_test_file('foo_test.py'))
        self.assertFalse(expert._is_test_file('cortical/query.py'))
        self.assertFalse(expert._is_test_file('scripts/run.py'))

    def test_coverage_estimate(self):
        """Test coverage estimation."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert(
            model_data={
                'source_to_tests': {
                    'a.py': {'test_a.py': 5},
                    'b.py': {'test_b.py': 3}
                },
                'test_to_sources': {},
                'test_failure_patterns': {},
                'module_to_tests': {},
                'test_cochange': {},
                'total_commits': 10
            }
        )

        # All files covered
        coverage = expert.get_coverage_estimate(['a.py', 'b.py'])
        self.assertEqual(coverage, 1.0)

        # Half covered
        coverage = expert.get_coverage_estimate(['a.py', 'c.py'])
        self.assertEqual(coverage, 0.5)

        # None covered
        coverage = expert.get_coverage_estimate(['c.py', 'd.py'])
        self.assertEqual(coverage, 0.0)

    def test_save_load_roundtrip(self):
        """Test saving and loading TestExpert."""
        from scripts.hubris.experts.test_expert import TestExpert

        expert = TestExpert(
            expert_id='test_expert_v1',
            version='1.0.0',
            trained_on_commits=25,
            model_data={
                'source_to_tests': {'x.py': {'test_x.py': 3}},
                'test_to_sources': {'test_x.py': ['x.py']},
                'test_failure_patterns': {},
                'module_to_tests': {},
                'test_cochange': {},
                'total_commits': 25
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_expert.json'
            expert.save(path)

            loaded = TestExpert.load(path)

            self.assertEqual(loaded.expert_id, expert.expert_id)
            self.assertEqual(loaded.trained_on_commits, 25)
            self.assertIn('x.py', loaded.model_data['source_to_tests'])


if __name__ == '__main__':
    unittest.main()
