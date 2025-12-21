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
from credit_account import CreditAccount, CreditTransaction, CreditLedger
from value_signal import (
    ValueSignal, SignalType, ValueAttributor, SignalBuffer
)
from staking import Stake, StakePool, StakeStrategy, AutoStaker


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


class TestEpisodeExpert(unittest.TestCase):
    """Test EpisodeExpert."""

    def test_create_episode_expert(self):
        """Test creating EpisodeExpert."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert()

        self.assertEqual(expert.expert_type, 'episode')
        self.assertIn('action_sequences', expert.model_data)
        self.assertIn('context_to_actions', expert.model_data)
        self.assertIn('success_patterns', expert.model_data)

    def test_extract_keywords(self):
        """Test keyword extraction from context."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert()

        keywords = expert._extract_keywords("Fix the authentication bug in the login module")
        self.assertIn('fix', keywords)
        self.assertIn('authentication', keywords)
        self.assertIn('bug', keywords)
        self.assertIn('login', keywords)
        self.assertIn('module', keywords)
        # Stop words should be filtered
        self.assertNotIn('the', keywords)
        self.assertNotIn('in', keywords)

    def test_train_episodes(self):
        """Test training on episodes."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert()

        episodes = [
            {
                'context': 'Fix bug in auth',
                'actions': ['Read', 'Grep', 'Edit', 'Bash'],
                'outcome': 'success',
                'files': ['auth.py']
            },
            {
                'context': 'Add new feature',
                'actions': ['Write', 'Edit', 'Bash'],
                'outcome': 'success',
                'files': ['feature.py']
            }
        ]

        expert.train(episodes)

        # Check training results
        self.assertEqual(expert.model_data['total_episodes'], 2)
        self.assertGreater(len(expert.model_data['action_sequences']), 0)
        self.assertGreater(len(expert.model_data['success_patterns']), 0)

    def test_predict_by_sequence(self):
        """Test prediction based on action sequences."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert(
            model_data={
                'action_sequences': {
                    'Read': {'Grep': 10, 'Edit': 5},
                    'Grep': {'Edit': 8}
                },
                'context_to_actions': {},
                'success_patterns': [],
                'failure_patterns': [],
                'action_frequency': {},
                'total_episodes': 10
            }
        )

        # Predict after Read action
        pred = expert.predict({
            'last_actions': ['Read'],
            'top_n': 5
        })

        # Should suggest Grep and Edit
        actions = [action for action, _ in pred.items]
        self.assertIn('Grep', actions)

    def test_predict_by_context(self):
        """Test prediction based on context keywords."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert(
            model_data={
                'action_sequences': {},
                'context_to_actions': {
                    'auth': {'Read': 5, 'Edit': 3},
                    'bug': {'Grep': 4, 'Edit': 6}
                },
                'success_patterns': [],
                'failure_patterns': [],
                'action_frequency': {},
                'total_episodes': 10
            }
        )

        pred = expert.predict({
            'query': 'Fix authentication bug',
            'top_n': 5
        })

        # Should suggest Edit (appears in both auth and bug contexts)
        actions = [action for action, _ in pred.items]
        self.assertIn('Edit', actions)

    def test_extract_episodes_from_transcript(self):
        """Test extracting episodes from transcript exchanges."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        # Mock transcript exchanges
        exchanges = [
            {
                'query': 'Fix the bug',
                'tools_used': ['Read', 'Edit', 'Bash'],
                'timestamp': '2025-12-17T10:00:00',
                'tool_inputs': [
                    {'tool': 'Read', 'input': {'file_path': 'test.py'}},
                    {'tool': 'Edit', 'input': {'file_path': 'test.py'}}
                ]
            },
            {
                'query': 'Add feature',
                'tools_used': ['Write', 'Bash'],
                'timestamp': '2025-12-17T11:00:00',
                'tool_inputs': []
            }
        ]

        episodes = EpisodeExpert.extract_episodes(exchanges)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]['context'], 'Fix the bug')
        self.assertEqual(episodes[0]['actions'], ['Read', 'Edit', 'Bash'])
        self.assertIn('test.py', episodes[0]['files'])

    def test_get_action_stats(self):
        """Test getting action statistics."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert(
            model_data={
                'action_sequences': {
                    'Read': {'Edit': 10, 'Grep': 5}
                },
                'context_to_actions': {},
                'success_patterns': [{'context': 'test'}],
                'failure_patterns': [],
                'action_frequency': {'Read': 20, 'Edit': 15},
                'total_episodes': 25
            }
        )

        stats = expert.get_action_stats()

        self.assertEqual(stats['total_episodes'], 25)
        self.assertEqual(stats['unique_actions'], 2)
        self.assertEqual(stats['success_patterns'], 1)
        self.assertGreater(len(stats['top_sequences']), 0)

    def test_predict_with_multiple_signals(self):
        """Test prediction combining multiple signals."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert(
            model_data={
                'action_sequences': {
                    'Read': {'Edit': 5}
                },
                'context_to_actions': {
                    'bug': {'Grep': 3, 'Edit': 2}
                },
                'success_patterns': [
                    {
                        'context': 'Fix authentication bug',
                        'actions': ['Read', 'Edit', 'Bash'],
                        'files': [],
                        'timestamp': ''
                    }
                ],
                'failure_patterns': [],
                'action_frequency': {'Read': 10, 'Edit': 8, 'Grep': 5},
                'total_episodes': 15
            }
        )

        pred = expert.predict({
            'query': 'Fix authentication bug',
            'last_actions': ['Read'],
            'files_touched': ['auth.py'],
            'top_n': 5
        })

        # Should have multiple predictions
        self.assertGreater(len(pred.items), 0)
        # Edit should rank high (sequence + context + success pattern)
        actions = [action for action, _ in pred.items]
        self.assertIn('Edit', actions)

    def test_save_load_roundtrip(self):
        """Test saving and loading EpisodeExpert."""
        from scripts.hubris.experts.episode_expert import EpisodeExpert

        expert = EpisodeExpert(
            expert_id='episode_test',
            version='1.0.0',
            trained_on_sessions=10,
            model_data={
                'action_sequences': {'Read': {'Edit': 5}},
                'context_to_actions': {'bug': {'Edit': 3}},
                'success_patterns': [],
                'failure_patterns': [],
                'action_frequency': {'Read': 10},
                'total_episodes': 10
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'episode_expert.json'
            expert.save(path)

            loaded = EpisodeExpert.load(path)

            self.assertEqual(loaded.expert_id, expert.expert_id)
            self.assertEqual(loaded.trained_on_sessions, 10)
            self.assertIn('Read', loaded.model_data['action_sequences'])


class TestExpertConsolidator(unittest.TestCase):
    """Test ExpertConsolidator."""

    def test_create_consolidator(self):
        """Test creating consolidator."""
        from scripts.hubris.expert_consolidator import ExpertConsolidator

        consolidator = ExpertConsolidator()

        self.assertIsNotNone(consolidator)
        self.assertEqual(len(consolidator.experts), 0)
        self.assertIsNotNone(consolidator.aggregator)

    def test_create_all_experts(self):
        """Test creating fresh instances of all experts."""
        from scripts.hubris.expert_consolidator import ExpertConsolidator

        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

        # Should have all 5 expert types
        self.assertEqual(len(consolidator.experts), 5)
        self.assertIn('file', consolidator.experts)
        self.assertIn('test', consolidator.experts)
        self.assertIn('error', consolidator.experts)
        self.assertIn('episode', consolidator.experts)
        self.assertIn('refactor', consolidator.experts)

    def test_consolidate_training(self):
        """Test training all experts with appropriate data."""
        from scripts.hubris.expert_consolidator import ExpertConsolidator

        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

        # Mock data
        commits = [
            {
                'files': ['a.py', 'tests/test_a.py'],
                'message': 'feat: Add feature'
            }
        ]

        transcripts = [
            {
                'query': 'Fix bug',
                'tools_used': ['Read', 'Edit'],
                'timestamp': '2025-12-17T10:00:00',
                'tool_inputs': []
            }
        ]

        errors = [
            {
                'error_type': 'TypeError',
                'error_message': 'test error',
                'files_modified': ['a.py'],
                'resolution': 'Fixed'
            }
        ]

        # Train
        results = consolidator.consolidate_training(
            commits=commits,
            transcripts=transcripts,
            errors=errors
        )

        # Check that some experts were trained
        self.assertIn('test', results)
        self.assertIn('episode', results)
        self.assertIn('error', results)

    def test_save_load_all_experts(self):
        """Test saving and loading all experts atomically."""
        from scripts.hubris.expert_consolidator import ExpertConsolidator

        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

        # Train with minimal data
        consolidator.consolidate_training(
            commits=[{'files': ['a.py'], 'message': 'test'}],
            errors=[{'error_type': 'Test', 'error_message': 'test', 'files_modified': []}]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / 'models'

            # Save all
            consolidator.save_all_experts(model_dir)

            # Verify files exist
            self.assertTrue((model_dir / 'test_expert.json').exists())
            self.assertTrue((model_dir / 'error_expert.json').exists())

            # Load all
            consolidator2 = ExpertConsolidator(model_dir)

            # Should have loaded experts
            self.assertGreater(len(consolidator2.experts), 0)
            self.assertTrue(consolidator2.has_expert('test'))
            self.assertTrue(consolidator2.has_expert('error'))

    def test_get_ensemble_prediction(self):
        """Test getting ensemble prediction from multiple experts."""
        from scripts.hubris.expert_consolidator import ExpertConsolidator

        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

        # Train test expert with some data
        consolidator.consolidate_training(
            commits=[
                {
                    'files': ['cortical/query/search.py', 'tests/test_query.py'],
                    'message': 'feat: search'
                }
            ]
        )

        # Get ensemble prediction
        context = {
            'changed_files': ['cortical/query/search.py'],
            'query': 'update search'
        }

        pred = consolidator.get_ensemble_prediction(context, expert_types=['test'])

        # Should return an AggregatedPrediction
        self.assertIsNotNone(pred)
        # Note: contributing_experts contains expert_id, not expert_type
        self.assertEqual(len(pred.contributing_experts), 1)

    def test_get_training_stats(self):
        """Test getting training statistics."""
        from scripts.hubris.expert_consolidator import ExpertConsolidator

        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

        stats = consolidator.get_training_stats()

        # Should have stats for all experts
        self.assertEqual(len(stats), 5)
        self.assertIn('file', stats)
        self.assertIn('refactor', stats)
        self.assertIn('version', stats['file'])
        self.assertIn('created_at', stats['file'])


class TestErrorDiagnosisExpert(unittest.TestCase):
    """Test ErrorDiagnosisExpert."""

    def test_create_error_expert(self):
        """Test creating ErrorDiagnosisExpert."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        self.assertEqual(expert.expert_type, 'error')
        self.assertIn('error_to_files', expert.model_data)
        self.assertIn('error_categories', expert.model_data)

    def test_extract_error_type(self):
        """Test error type extraction from messages."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        error_type = expert._extract_error_type("TypeError: unsupported operand", "")
        self.assertEqual(error_type, 'TypeError')

        error_type = expert._extract_error_type("ImportError: No module named 'foo'", "")
        self.assertEqual(error_type, 'ImportError')

    def test_get_error_category(self):
        """Test error categorization."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        self.assertEqual(expert._get_error_category('TypeError'), 'type')
        self.assertEqual(expert._get_error_category('ImportError'), 'import')
        self.assertEqual(expert._get_error_category('FileNotFoundError'), 'io')
        self.assertEqual(expert._get_error_category('KeyError'), 'key')
        self.assertEqual(expert._get_error_category('UnknownError'), 'unknown')

    def test_predict_with_error_message(self):
        """Test prediction with error message."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        pred = expert.predict({
            'error_message': "TypeError: 'NoneType' object has no attribute 'foo'"
        })

        self.assertEqual(pred.expert_type, 'error')
        self.assertEqual(pred.metadata['error_type'], 'TypeError')
        self.assertEqual(pred.metadata['category'], 'type')
        self.assertGreater(len(pred.items), 0)

    def test_predict_with_stack_trace(self):
        """Test prediction with stack trace."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        stack_trace = '''
Traceback (most recent call last):
  File "cortical/query/search.py", line 42, in find_documents
    return results[0]
TypeError: 'NoneType' object is not subscriptable
'''
        pred = expert.predict({
            'error_message': "TypeError: 'NoneType' object is not subscriptable",
            'stack_trace': stack_trace
        })

        # Should extract file from stack trace
        self.assertIn('cortical/query/search.py', pred.metadata['suggested_files'])

    def test_diagnose_convenience_method(self):
        """Test the diagnose convenience method."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        result = expert.diagnose("KeyError: 'missing_key'")

        self.assertEqual(result['error_type'], 'KeyError')
        self.assertEqual(result['category'], 'key')
        self.assertIn('likely_causes', result)
        self.assertIn('suggested_fixes', result)

    def test_suggest_fixes_by_keywords(self):
        """Test fix suggestions based on keywords."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        fixes = expert._suggest_fixes_by_keywords("import failed module not found")

        self.assertGreater(len(fixes), 0)
        # Should suggest import-related fixes
        fix_names = list(fixes.keys())
        self.assertTrue(any('import' in f.lower() or 'module' in f.lower() or 'installed' in f.lower() for f in fix_names))

    def test_get_common_causes(self):
        """Test getting common causes for error types."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        causes = expert._get_common_causes('ImportError')
        self.assertIn('Module not installed', causes)

        causes = expert._get_common_causes('KeyError')
        self.assertIn('Key not in dictionary', causes)

    def test_train_expert(self):
        """Test training ErrorDiagnosisExpert on error records."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert()

        error_records = [
            {
                'error_type': 'TypeError',
                'error_message': 'unsupported operand type',
                'files_modified': ['cortical/analysis.py'],
                'resolution': 'Added type check before operation'
            },
            {
                'error_type': 'ImportError',
                'error_message': 'No module named foo',
                'files_modified': ['requirements.txt'],
                'resolution': 'Added missing dependency to requirements'
            }
        ]

        expert.train(error_records)

        self.assertEqual(expert.model_data['total_errors'], 2)
        self.assertIn('TypeError', expert.model_data['error_to_files'])
        self.assertGreater(len(expert.model_data['resolution_history']), 0)

    def test_save_load_roundtrip(self):
        """Test saving and loading ErrorDiagnosisExpert."""
        from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

        expert = ErrorDiagnosisExpert(
            expert_id='error_expert_v1',
            version='1.0.0',
            trained_on_commits=10,
            model_data={
                'error_to_files': {'TypeError': {'fix.py': 3}},
                'error_to_causes': {},
                'keyword_to_fixes': {},
                'stack_patterns': {},
                'error_categories': ErrorDiagnosisExpert.ERROR_CATEGORIES.copy(),
                'resolution_history': [],
                'total_errors': 10
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'error_expert.json'
            expert.save(path)

            loaded = ErrorDiagnosisExpert.load(path)

            self.assertEqual(loaded.expert_id, expert.expert_id)
            self.assertEqual(loaded.trained_on_commits, 10)
            self.assertIn('TypeError', loaded.model_data['error_to_files'])


class TestValueAttribution(unittest.TestCase):
    """Test ValueSignal and attribution system."""

    def test_value_signal_creation(self):
        """Test creating a value signal."""
        signal = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=0.8,
            timestamp=1234567890.0,
            source='test_result',
            expert_id='expert_1',
            prediction_id='pred_1',
            context={'test': 'data'}
        )

        self.assertEqual(signal.signal_type, 'positive')
        self.assertEqual(signal.magnitude, 0.8)
        self.assertEqual(signal.source, 'test_result')
        self.assertEqual(signal.expert_id, 'expert_1')
        self.assertEqual(signal.prediction_id, 'pred_1')
        self.assertEqual(signal.context['test'], 'data')

    def test_signal_types(self):
        """Test all signal types."""
        # Positive signal
        pos_signal = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='e1',
            prediction_id='p1'
        )
        self.assertEqual(pos_signal.signal_type, 'positive')

        # Negative signal
        neg_signal = ValueSignal(
            signal_type=SignalType.NEGATIVE.value,
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='e1',
            prediction_id='p1'
        )
        self.assertEqual(neg_signal.signal_type, 'negative')

        # Neutral signal
        neutral_signal = ValueSignal(
            signal_type=SignalType.NEUTRAL.value,
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='e1',
            prediction_id='p1'
        )
        self.assertEqual(neutral_signal.signal_type, 'neutral')

    def test_signal_validation(self):
        """Test signal validation."""
        # Invalid magnitude (too high)
        with self.assertRaises(ValueError):
            ValueSignal(
                signal_type='positive',
                magnitude=1.5,
                timestamp=0.0,
                source='test',
                expert_id='e1',
                prediction_id='p1'
            )

        # Invalid magnitude (negative)
        with self.assertRaises(ValueError):
            ValueSignal(
                signal_type='positive',
                magnitude=-0.5,
                timestamp=0.0,
                source='test',
                expert_id='e1',
                prediction_id='p1'
            )

        # Invalid signal type
        with self.assertRaises(ValueError):
            ValueSignal(
                signal_type='invalid',
                magnitude=0.5,
                timestamp=0.0,
                source='test',
                expert_id='e1',
                prediction_id='p1'
            )

    def test_signal_immutable(self):
        """Test that signals are immutable."""
        signal = ValueSignal(
            signal_type='positive',
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='e1',
            prediction_id='p1'
        )

        # Should not be able to modify
        with self.assertRaises(AttributeError):
            signal.magnitude = 0.8

    def test_signal_serialization(self):
        """Test signal to_dict and from_dict."""
        signal = ValueSignal(
            signal_type='positive',
            magnitude=0.7,
            timestamp=1234567890.0,
            source='commit_result',
            expert_id='expert_2',
            prediction_id='pred_2',
            context={'accuracy': 0.85}
        )

        # Convert to dict
        data = signal.to_dict()
        self.assertEqual(data['signal_type'], 'positive')
        self.assertEqual(data['magnitude'], 0.7)
        self.assertEqual(data['context']['accuracy'], 0.85)

        # Load from dict
        loaded = ValueSignal.from_dict(data)
        self.assertEqual(loaded.signal_type, signal.signal_type)
        self.assertEqual(loaded.magnitude, signal.magnitude)
        self.assertEqual(loaded.expert_id, signal.expert_id)
        self.assertEqual(loaded.context, signal.context)

    def test_attributor_positive_signal(self):
        """Test attributor with positive signal."""
        attributor = ValueAttributor(
            positive_multiplier=10.0,
            use_confidence_scaling=False
        )

        signal = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=0.8,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1'
        )

        amount = attributor.calculate_credit_amount(signal)
        # 0.8 * 10.0 = 8.0
        self.assertAlmostEqual(amount, 8.0)

    def test_attributor_negative_signal(self):
        """Test attributor with negative signal."""
        attributor = ValueAttributor(
            negative_multiplier=5.0,
            use_confidence_scaling=False
        )

        signal = ValueSignal(
            signal_type=SignalType.NEGATIVE.value,
            magnitude=0.6,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1'
        )

        amount = attributor.calculate_credit_amount(signal)
        # -0.6 * 5.0 = -3.0
        self.assertAlmostEqual(amount, -3.0)

    def test_attributor_neutral_signal(self):
        """Test attributor with neutral signal."""
        attributor = ValueAttributor()

        signal = ValueSignal(
            signal_type=SignalType.NEUTRAL.value,
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1'
        )

        amount = attributor.calculate_credit_amount(signal)
        self.assertAlmostEqual(amount, 0.0)

    def test_attributor_confidence_scaling(self):
        """Test confidence scaling in attribution."""
        attributor = ValueAttributor(
            positive_multiplier=10.0,
            use_confidence_scaling=True
        )

        signal = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=0.8,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1',
            context={'confidence': 0.5}
        )

        amount = attributor.calculate_credit_amount(signal)
        # 0.8 * 10.0 * 0.5 = 4.0
        self.assertAlmostEqual(amount, 4.0)

    def test_attributor_from_test_result(self):
        """Test creating signal from test result."""
        attributor = ValueAttributor()

        # Test passed
        signal = attributor.attribute_from_test_result(
            expert_id='expert_1',
            prediction_id='pred_1',
            test_passed=True,
            confidence=0.9
        )

        self.assertEqual(signal.signal_type, SignalType.POSITIVE.value)
        self.assertEqual(signal.magnitude, 0.9)
        self.assertEqual(signal.source, 'test_result')
        self.assertTrue(signal.context['test_passed'])

        # Test failed
        signal_fail = attributor.attribute_from_test_result(
            expert_id='expert_1',
            prediction_id='pred_2',
            test_passed=False,
            confidence=0.7
        )

        self.assertEqual(signal_fail.signal_type, SignalType.NEGATIVE.value)
        self.assertFalse(signal_fail.context['test_passed'])

    def test_attributor_from_commit_result(self):
        """Test creating signal from commit result."""
        attributor = ValueAttributor()

        # High accuracy (>50%)
        signal = attributor.attribute_from_commit_result(
            expert_id='expert_1',
            prediction_id='pred_1',
            files_correct=['a.py', 'b.py', 'c.py'],
            files_total=['a.py', 'b.py', 'c.py', 'd.py']
        )

        self.assertEqual(signal.signal_type, SignalType.POSITIVE.value)
        self.assertAlmostEqual(signal.magnitude, 0.75)  # 3/4
        self.assertEqual(signal.source, 'commit_result')
        self.assertEqual(signal.context['accuracy'], 0.75)

        # Low accuracy (<50%)
        signal_low = attributor.attribute_from_commit_result(
            expert_id='expert_1',
            prediction_id='pred_2',
            files_correct=['a.py'],
            files_total=['a.py', 'b.py', 'c.py', 'd.py']
        )

        self.assertEqual(signal_low.signal_type, SignalType.NEGATIVE.value)
        self.assertAlmostEqual(signal_low.magnitude, 0.25)  # 1/4

        # Exactly 50% (neutral)
        signal_neutral = attributor.attribute_from_commit_result(
            expert_id='expert_1',
            prediction_id='pred_3',
            files_correct=['a.py', 'b.py'],
            files_total=['a.py', 'b.py', 'c.py', 'd.py']
        )

        self.assertEqual(signal_neutral.signal_type, SignalType.NEUTRAL.value)

    def test_attributor_from_user_feedback(self):
        """Test creating signal from user feedback."""
        attributor = ValueAttributor()

        # Helpful feedback
        signal = attributor.attribute_from_user_feedback(
            expert_id='expert_1',
            prediction_id='pred_1',
            helpful=True,
            importance=0.8
        )

        self.assertEqual(signal.signal_type, SignalType.POSITIVE.value)
        self.assertEqual(signal.magnitude, 0.8)
        self.assertEqual(signal.source, 'user_feedback')
        self.assertTrue(signal.context['helpful'])

        # Not helpful
        signal_neg = attributor.attribute_from_user_feedback(
            expert_id='expert_1',
            prediction_id='pred_2',
            helpful=False,
            importance=0.6
        )

        self.assertEqual(signal_neg.signal_type, SignalType.NEGATIVE.value)
        self.assertFalse(signal_neg.context['helpful'])

    def test_process_signal_with_ledger(self):
        """Test processing signal with credit ledger."""
        attributor = ValueAttributor(positive_multiplier=10.0)
        ledger = CreditLedger()

        # Add initial credit
        account = ledger.get_or_create_account('expert_1')

        # Positive signal
        signal = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=0.5,
            timestamp=0.0,
            source='test_result',
            expert_id='expert_1',
            prediction_id='pred_1',
            context={'confidence': 1.0}
        )

        amount = attributor.process_signal(signal, ledger)

        # Should have credited 5.0 (0.5 * 10.0)
        self.assertAlmostEqual(amount, 5.0)
        account = ledger.get_or_create_account('expert_1')
        self.assertGreater(account.balance, 0.0)

    def test_signal_buffer_add_flush(self):
        """Test signal buffer add and flush."""
        buffer = SignalBuffer()
        ledger = CreditLedger()

        # Add signals
        signal1 = ValueSignal(
            signal_type='positive',
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1'
        )
        signal2 = ValueSignal(
            signal_type='positive',
            magnitude=0.7,
            timestamp=0.0,
            source='test',
            expert_id='expert_2',
            prediction_id='pred_2'
        )

        buffer.add(signal1)
        buffer.add(signal2)

        self.assertEqual(buffer.get_pending_count(), 2)

        # Flush
        totals = buffer.flush(ledger)

        # Buffer should be empty
        self.assertEqual(buffer.get_pending_count(), 0)

        # Should have processed both signals
        self.assertIn('expert_1', totals)
        self.assertIn('expert_2', totals)

    def test_signal_buffer_clear(self):
        """Test clearing buffer without processing."""
        buffer = SignalBuffer()

        signal = ValueSignal(
            signal_type='positive',
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1'
        )

        buffer.add(signal)
        self.assertEqual(buffer.get_pending_count(), 1)

        buffer.clear()
        self.assertEqual(buffer.get_pending_count(), 0)

    def test_signal_buffer_peek(self):
        """Test peeking at buffer without removing."""
        buffer = SignalBuffer()

        signal = ValueSignal(
            signal_type='positive',
            magnitude=0.5,
            timestamp=0.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1'
        )

        buffer.add(signal)

        # Peek should return signals without removing
        signals = buffer.peek()
        self.assertEqual(len(signals), 1)
        self.assertEqual(buffer.get_pending_count(), 1)

    def test_attribution_with_confidence_scaling(self):
        """Test full attribution flow with confidence scaling."""
        attributor = ValueAttributor(
            positive_multiplier=10.0,
            use_confidence_scaling=True
        )
        ledger = CreditLedger()
        buffer = SignalBuffer()

        # Create test result signal with high confidence
        signal1 = attributor.attribute_from_test_result(
            expert_id='expert_1',
            prediction_id='pred_1',
            test_passed=True,
            confidence=0.9
        )

        # Create commit result signal with medium confidence
        signal2 = attributor.attribute_from_commit_result(
            expert_id='expert_1',
            prediction_id='pred_2',
            files_correct=['a.py', 'b.py'],
            files_total=['a.py', 'b.py'],
            confidence=0.6
        )

        buffer.add(signal1)
        buffer.add(signal2)

        # Flush all signals
        totals = buffer.flush(ledger, attributor)

        # Expert should have received credit
        self.assertIn('expert_1', totals)
        self.assertGreater(totals['expert_1'], 0.0)

        # Check ledger balance
        account = ledger.get_or_create_account('expert_1')
        self.assertGreater(account.balance, 0.0)

    def test_time_decay_attribution(self):
        """Test time decay in attribution."""
        attributor = ValueAttributor(
            positive_multiplier=10.0,
            use_time_decay=True,
            decay_halflife_seconds=100.0,
            use_confidence_scaling=False
        )

        # Recent prediction (no decay)
        signal_recent = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=1.0,
            timestamp=100.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_1',
            context={'prediction_time': 100.0}
        )

        amount_recent = attributor.calculate_credit_amount(signal_recent)
        self.assertAlmostEqual(amount_recent, 10.0)

        # Old prediction (half-life elapsed, should be ~5.0)
        signal_old = ValueSignal(
            signal_type=SignalType.POSITIVE.value,
            magnitude=1.0,
            timestamp=200.0,
            source='test',
            expert_id='expert_1',
            prediction_id='pred_2',
            context={'prediction_time': 100.0}
        )

        amount_old = attributor.calculate_credit_amount(signal_old)
        self.assertAlmostEqual(amount_old, 5.0, places=1)


class TestCreditSystem(unittest.TestCase):
    """Test CreditAccount and CreditLedger for expert value tracking."""

    def test_credit_transaction_creation(self):
        """Test creating credit transactions."""
        tx = CreditTransaction(
            timestamp=1234567890.0,
            amount=10.0,
            expert_id='expert_1',
            reason='correct_prediction',
            context={'accuracy': 0.95},
            balance_after=110.0
        )

        self.assertEqual(tx.timestamp, 1234567890.0)
        self.assertEqual(tx.amount, 10.0)
        self.assertEqual(tx.expert_id, 'expert_1')
        self.assertEqual(tx.reason, 'correct_prediction')
        self.assertEqual(tx.balance_after, 110.0)

        # Test serialization
        d = tx.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['amount'], 10.0)

        # Test deserialization
        tx2 = CreditTransaction.from_dict(d)
        self.assertEqual(tx2.amount, tx.amount)
        self.assertEqual(tx2.expert_id, tx.expert_id)

    def test_account_credit_debit(self):
        """Test crediting and debiting accounts."""
        account = CreditAccount('expert_1', initial_balance=100.0)

        # Test initial state
        self.assertEqual(account.balance, 100.0)
        self.assertEqual(len(account.transactions), 0)

        # Test credit
        tx1 = account.credit(25.0, 'good_prediction', {'query': 'test'})
        self.assertEqual(account.balance, 125.0)
        self.assertEqual(tx1.amount, 25.0)
        self.assertEqual(tx1.balance_after, 125.0)
        self.assertEqual(len(account.transactions), 1)

        # Test debit
        tx2 = account.debit(15.0, 'wrong_prediction')
        self.assertEqual(account.balance, 110.0)
        self.assertEqual(tx2.amount, -15.0)
        self.assertEqual(tx2.balance_after, 110.0)
        self.assertEqual(len(account.transactions), 2)

    def test_account_balance_never_below_min(self):
        """Test that balance can go negative (track poorly performing experts)."""
        account = CreditAccount('expert_bad', initial_balance=100.0)

        # Debit more than balance
        account.debit(150.0, 'very_wrong')

        # Balance should be negative
        self.assertEqual(account.balance, -50.0)
        self.assertLess(account.balance, 0)

    def test_transaction_history(self):
        """Test transaction history queries."""
        account = CreditAccount('expert_1', initial_balance=100.0)

        # Create some transactions
        import time
        t1 = time.time()
        account.credit(10.0, 'reason1')
        time.sleep(0.01)
        t2 = time.time()
        account.debit(5.0, 'reason2')
        time.sleep(0.01)
        account.credit(15.0, 'reason3')

        # Test get_recent_transactions
        recent = account.get_recent_transactions(n=2)
        self.assertEqual(len(recent), 2)
        # Should be newest first
        self.assertEqual(recent[0].reason, 'reason3')
        self.assertEqual(recent[1].reason, 'reason2')

        # Test get_transactions_since
        since = account.get_transactions_since(t2)
        self.assertEqual(len(since), 2)  # reason2 and reason3
        # Should be oldest first
        self.assertEqual(since[0].reason, 'reason2')

        # Test get_balance_history
        history = account.get_balance_history()
        self.assertEqual(len(history), 4)  # Initial + 3 transactions
        # Check balance progression
        self.assertEqual(history[0][1], 100.0)  # Initial
        self.assertEqual(history[-1][1], 120.0)  # Final: 100 + 10 - 5 + 15

    def test_account_serialization(self):
        """Test account save/load roundtrip."""
        account = CreditAccount('expert_1', initial_balance=100.0)
        account.credit(25.0, 'test_credit')
        account.debit(10.0, 'test_debit')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'account.json'
            account.save(path)

            # Verify file exists
            self.assertTrue(path.exists())

            # Load and verify
            loaded = CreditAccount.load(path)
            self.assertEqual(loaded.expert_id, 'expert_1')
            self.assertEqual(loaded.balance, 115.0)
            self.assertEqual(len(loaded.transactions), 2)
            self.assertEqual(loaded.transactions[0].amount, 25.0)

    def test_ledger_get_or_create(self):
        """Test ledger account creation and retrieval."""
        ledger = CreditLedger()

        # Initially empty
        self.assertEqual(len(ledger.accounts), 0)

        # Create first account
        account1 = ledger.get_or_create_account('expert_1')
        self.assertEqual(account1.expert_id, 'expert_1')
        self.assertEqual(account1.balance, 100.0)
        self.assertEqual(len(ledger.accounts), 1)

        # Get same account again
        account1_again = ledger.get_or_create_account('expert_1')
        self.assertIs(account1_again, account1)  # Same object
        self.assertEqual(len(ledger.accounts), 1)  # No new account

        # Create second account with custom initial balance
        account2 = ledger.get_or_create_account('expert_2', initial_balance=200.0)
        self.assertEqual(account2.balance, 200.0)
        self.assertEqual(len(ledger.accounts), 2)

    def test_ledger_transfer(self):
        """Test credit transfers between experts."""
        ledger = CreditLedger()

        # Create accounts
        account1 = ledger.get_or_create_account('expert_1')
        account2 = ledger.get_or_create_account('expert_2')

        initial_balance1 = account1.balance
        initial_balance2 = account2.balance

        # Transfer
        debit_tx, credit_tx = ledger.transfer(
            'expert_1',
            'expert_2',
            30.0,
            'transfer_test',
            {'metadata': 'value'}
        )

        # Check balances
        self.assertEqual(account1.balance, initial_balance1 - 30.0)
        self.assertEqual(account2.balance, initial_balance2 + 30.0)

        # Check transactions
        self.assertEqual(debit_tx.amount, -30.0)
        self.assertEqual(credit_tx.amount, 30.0)
        self.assertIn('transfer_from', debit_tx.context)
        self.assertIn('transfer_to', credit_tx.context)

        # Test invalid transfer
        with self.assertRaises(ValueError):
            ledger.transfer('expert_1', 'expert_2', -10.0, 'invalid')

    def test_ledger_top_experts(self):
        """Test getting top experts by balance."""
        ledger = CreditLedger()

        # Create experts with different balances
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        ledger.get_or_create_account('expert_2', initial_balance=200.0)
        ledger.get_or_create_account('expert_3', initial_balance=150.0)
        ledger.get_or_create_account('expert_4', initial_balance=50.0)

        # Get top 3
        top3 = ledger.get_top_experts(n=3)

        self.assertEqual(len(top3), 3)
        # Should be sorted by balance descending
        self.assertEqual(top3[0][0], 'expert_2')  # 200.0
        self.assertEqual(top3[0][1], 200.0)
        self.assertEqual(top3[1][0], 'expert_3')  # 150.0
        self.assertEqual(top3[2][0], 'expert_1')  # 100.0

        # Get all
        all_experts = ledger.get_top_experts(n=10)
        self.assertEqual(len(all_experts), 4)

    def test_ledger_persistence(self):
        """Test ledger save/load roundtrip."""
        ledger = CreditLedger()

        # Create some accounts with transactions
        account1 = ledger.get_or_create_account('expert_1')
        account1.credit(50.0, 'test')

        account2 = ledger.get_or_create_account('expert_2')
        account2.debit(20.0, 'test')

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'ledger.json'
            ledger.save(path)

            # Verify file exists
            self.assertTrue(path.exists())

            # Load
            loaded = CreditLedger.load(path)

            # Verify accounts
            self.assertEqual(len(loaded.accounts), 2)
            self.assertIn('expert_1', loaded.accounts)
            self.assertIn('expert_2', loaded.accounts)

            # Verify balances
            self.assertEqual(loaded.accounts['expert_1'].balance, 150.0)
            self.assertEqual(loaded.accounts['expert_2'].balance, 80.0)

            # Verify transaction history preserved
            self.assertEqual(len(loaded.accounts['expert_1'].transactions), 1)
            self.assertEqual(loaded.accounts['expert_1'].transactions[0].amount, 50.0)

    def test_ledger_total_credits(self):
        """Test getting total credits across all accounts."""
        ledger = CreditLedger()

        # Empty ledger
        self.assertEqual(ledger.get_total_credits(), 0.0)

        # Add accounts
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        ledger.get_or_create_account('expert_2', initial_balance=150.0)
        ledger.get_or_create_account('expert_3', initial_balance=50.0)

        # Total should be sum
        self.assertEqual(ledger.get_total_credits(), 300.0)

        # After transaction
        ledger.accounts['expert_1'].credit(25.0, 'test')
        self.assertEqual(ledger.get_total_credits(), 325.0)

    def test_credit_debit_validation(self):
        """Test that credit/debit validate positive amounts."""
        account = CreditAccount('expert_1')

        # Test credit with invalid amounts
        with self.assertRaises(ValueError):
            account.credit(0.0, 'invalid')

        with self.assertRaises(ValueError):
            account.credit(-10.0, 'invalid')

        # Test debit with invalid amounts
        with self.assertRaises(ValueError):
            account.debit(0.0, 'invalid')

        with self.assertRaises(ValueError):
            account.debit(-10.0, 'invalid')


class TestStaking(unittest.TestCase):
    """Test Stake and StakePool for expert credit staking."""

    def test_stake_creation(self):
        """Test creating a stake."""
        stake = Stake(
            stake_id='stake_1',
            expert_id='expert_1',
            prediction_id='pred_1',
            amount=10.0,
            multiplier=1.5,
            timestamp=1234567890.0,
            status='pending',
            outcome_value=None
        )

        self.assertEqual(stake.stake_id, 'stake_1')
        self.assertEqual(stake.expert_id, 'expert_1')
        self.assertEqual(stake.amount, 10.0)
        self.assertEqual(stake.multiplier, 1.5)
        self.assertEqual(stake.status, 'pending')
        self.assertIsNone(stake.outcome_value)

        # Test serialization
        d = stake.to_dict()
        self.assertEqual(d['stake_id'], 'stake_1')
        self.assertEqual(d['amount'], 10.0)

        # Test deserialization
        stake2 = Stake.from_dict(d)
        self.assertEqual(stake2.stake_id, stake.stake_id)
        self.assertEqual(stake2.amount, stake.amount)

    def test_place_stake_success(self):
        """Test successfully placing a stake."""
        ledger = CreditLedger()
        pool = StakePool(ledger, max_stake_ratio=0.5, min_stake=5.0)

        # Create account with balance
        account = ledger.get_or_create_account('expert_1', initial_balance=100.0)
        initial_balance = account.balance

        # Place stake
        stake = pool.place_stake('expert_1', 'pred_1', amount=20.0, multiplier=1.5)

        # Check stake created
        self.assertIsNotNone(stake)
        self.assertEqual(stake.expert_id, 'expert_1')
        self.assertEqual(stake.prediction_id, 'pred_1')
        self.assertEqual(stake.amount, 20.0)
        self.assertEqual(stake.multiplier, 1.5)
        self.assertEqual(stake.status, 'pending')

        # Check balance deducted
        account = ledger.get_or_create_account('expert_1')
        self.assertEqual(account.balance, initial_balance - 20.0)

        # Check stake in pool
        self.assertIn(stake.stake_id, pool.stakes)

    def test_place_stake_insufficient_balance(self):
        """Test placing stake with insufficient balance."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create account with low balance
        ledger.get_or_create_account('expert_1', initial_balance=10.0)

        # Try to stake more than available
        with self.assertRaises(ValueError) as ctx:
            pool.place_stake('expert_1', 'pred_1', amount=20.0, multiplier=1.5)

        self.assertIn('Insufficient balance', str(ctx.exception))

    def test_place_stake_exceeds_ratio(self):
        """Test placing stake that exceeds max_stake_ratio."""
        ledger = CreditLedger()
        pool = StakePool(ledger, max_stake_ratio=0.5, min_stake=5.0)

        # Create account
        ledger.get_or_create_account('expert_1', initial_balance=100.0)

        # Try to stake more than 50% of balance
        with self.assertRaises(ValueError) as ctx:
            pool.place_stake('expert_1', 'pred_1', amount=60.0, multiplier=1.5)

        self.assertIn('exceeds max allowed', str(ctx.exception))

    def test_resolve_stake_win(self):
        """Test resolving a winning stake."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create account and place stake
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        stake = pool.place_stake('expert_1', 'pred_1', amount=20.0, multiplier=2.0)

        balance_after_stake = ledger.accounts['expert_1'].balance

        # Resolve as win
        net_gain = pool.resolve_stake(stake.stake_id, success=True)

        # Check net gain (profit only, not including original stake)
        # Payout = 20.0 * 2.0 = 40.0
        # Net gain = 40.0 - 20.0 = 20.0
        self.assertEqual(net_gain, 20.0)

        # Check stake status
        self.assertEqual(stake.status, 'won')
        self.assertEqual(stake.outcome_value, 20.0)

        # Check balance (original - stake + payout)
        # 100 - 20 + 40 = 120
        account = ledger.get_or_create_account('expert_1')
        self.assertEqual(account.balance, 120.0)

    def test_resolve_stake_loss(self):
        """Test resolving a losing stake."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create account and place stake
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        stake = pool.place_stake('expert_1', 'pred_1', amount=20.0, multiplier=2.0)

        # Resolve as loss
        net_gain = pool.resolve_stake(stake.stake_id, success=False)

        # Check net loss
        self.assertEqual(net_gain, -20.0)

        # Check stake status
        self.assertEqual(stake.status, 'lost')
        self.assertEqual(stake.outcome_value, -20.0)

        # Check balance (original - stake)
        # 100 - 20 = 80
        account = ledger.get_or_create_account('expert_1')
        self.assertEqual(account.balance, 80.0)

    def test_cancel_stake(self):
        """Test cancelling a pending stake."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create account and place stake
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        stake = pool.place_stake('expert_1', 'pred_1', amount=20.0, multiplier=1.5)

        balance_after_stake = ledger.accounts['expert_1'].balance

        # Cancel stake
        result = pool.cancel_stake(stake.stake_id)

        # Check cancellation succeeded
        self.assertTrue(result)
        self.assertEqual(stake.status, 'cancelled')
        self.assertEqual(stake.outcome_value, 0.0)

        # Check balance restored
        account = ledger.get_or_create_account('expert_1')
        self.assertEqual(account.balance, 100.0)

        # Try to cancel again (should fail)
        result = pool.cancel_stake(stake.stake_id)
        self.assertFalse(result)

    def test_get_active_stakes(self):
        """Test getting active stakes."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create accounts
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        ledger.get_or_create_account('expert_2', initial_balance=100.0)

        # Place stakes
        stake1 = pool.place_stake('expert_1', 'pred_1', 10.0, 1.5)
        stake2 = pool.place_stake('expert_1', 'pred_2', 15.0, 2.0)
        stake3 = pool.place_stake('expert_2', 'pred_3', 20.0, 1.5)

        # Resolve one stake
        pool.resolve_stake(stake1.stake_id, success=True)

        # Get all active stakes
        active = pool.get_active_stakes()
        self.assertEqual(len(active), 2)
        active_ids = {s.stake_id for s in active}
        self.assertIn(stake2.stake_id, active_ids)
        self.assertIn(stake3.stake_id, active_ids)
        self.assertNotIn(stake1.stake_id, active_ids)

        # Get active stakes for expert_1
        active_expert1 = pool.get_active_stakes('expert_1')
        self.assertEqual(len(active_expert1), 1)
        self.assertEqual(active_expert1[0].stake_id, stake2.stake_id)

        # Get active stakes for expert_2
        active_expert2 = pool.get_active_stakes('expert_2')
        self.assertEqual(len(active_expert2), 1)
        self.assertEqual(active_expert2[0].stake_id, stake3.stake_id)

    def test_stake_pool_persistence(self):
        """Test saving and loading stake pool."""
        ledger = CreditLedger()
        pool = StakePool(ledger, max_stake_ratio=0.6, min_stake=10.0)

        # Create account and place stakes
        ledger.get_or_create_account('expert_1', initial_balance=200.0)
        stake1 = pool.place_stake('expert_1', 'pred_1', 30.0, 1.5)
        stake2 = pool.place_stake('expert_1', 'pred_2', 40.0, 2.0)

        # Resolve one stake
        pool.resolve_stake(stake1.stake_id, success=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'stake_pool.json'
            pool.save(path)

            # Verify file exists
            self.assertTrue(path.exists())

            # Load
            loaded_pool = StakePool.load(path, ledger)

            # Verify configuration
            self.assertEqual(loaded_pool.max_stake_ratio, 0.6)
            self.assertEqual(loaded_pool.min_stake, 10.0)

            # Verify stakes
            self.assertEqual(len(loaded_pool.stakes), 2)
            self.assertIn(stake1.stake_id, loaded_pool.stakes)
            self.assertIn(stake2.stake_id, loaded_pool.stakes)

            # Verify stake details
            loaded_stake1 = loaded_pool.stakes[stake1.stake_id]
            self.assertEqual(loaded_stake1.status, 'won')
            self.assertEqual(loaded_stake1.amount, 30.0)

            loaded_stake2 = loaded_pool.stakes[stake2.stake_id]
            self.assertEqual(loaded_stake2.status, 'pending')
            self.assertEqual(loaded_stake2.amount, 40.0)

    def test_auto_staker_high_confidence(self):
        """Test auto-staker with high confidence prediction."""
        ledger = CreditLedger()
        pool = StakePool(ledger, max_stake_ratio=0.5, min_stake=5.0)
        auto_staker = AutoStaker(pool)

        # Create account
        ledger.get_or_create_account('expert_1', initial_balance=100.0)

        # Create high confidence prediction
        prediction = ExpertPrediction(
            expert_id='expert_1',
            expert_type='test',
            items=[('file1.py', 0.9), ('file2.py', 0.7)],
            metadata={}
        )

        # Decide stake with MODERATE strategy
        decision = auto_staker.decide_stake(
            'expert_1',
            prediction,
            StakeStrategy.MODERATE
        )

        # Should stake
        self.assertIsNotNone(decision)
        amount, multiplier = decision

        # Should stake up to max (50% of 100 = 50)
        self.assertEqual(amount, 50.0)
        # Should use full strategy multiplier
        self.assertEqual(multiplier, 1.5)

    def test_auto_staker_low_confidence(self):
        """Test auto-staker with low confidence prediction."""
        ledger = CreditLedger()
        pool = StakePool(ledger, max_stake_ratio=0.5, min_stake=5.0)
        auto_staker = AutoStaker(pool)

        # Create account
        ledger.get_or_create_account('expert_1', initial_balance=100.0)

        # Create low confidence prediction
        prediction = ExpertPrediction(
            expert_id='expert_1',
            expert_type='test',
            items=[('file1.py', 0.3), ('file2.py', 0.2)],
            metadata={}
        )

        # Decide stake
        decision = auto_staker.decide_stake(
            'expert_1',
            prediction,
            StakeStrategy.AGGRESSIVE
        )

        # Should not stake (confidence too low)
        self.assertIsNone(decision)

    def test_auto_staker_medium_confidence(self):
        """Test auto-staker with medium confidence prediction."""
        ledger = CreditLedger()
        pool = StakePool(ledger, max_stake_ratio=0.5, min_stake=5.0)
        auto_staker = AutoStaker(pool)

        # Create account
        ledger.get_or_create_account('expert_1', initial_balance=100.0)

        # Create medium confidence prediction (0.65)
        prediction = ExpertPrediction(
            expert_id='expert_1',
            expert_type='test',
            items=[('file1.py', 0.65), ('file2.py', 0.5)],
            metadata={}
        )

        # Decide stake with AGGRESSIVE strategy (2.0x)
        decision = auto_staker.decide_stake(
            'expert_1',
            prediction,
            StakeStrategy.AGGRESSIVE
        )

        # Should stake
        self.assertIsNotNone(decision)
        amount, multiplier = decision

        # Should stake half of max (0.5 * 50 = 25)
        self.assertEqual(amount, 25.0)

        # Multiplier should be adjusted based on confidence
        # At 0.65 confidence: confidence_factor = (0.65 - 0.5) / 0.3 = 0.5
        # adjusted = 1.0 + (2.0 - 1.0) * 0.5 = 1.5
        self.assertAlmostEqual(multiplier, 1.5)

    def test_stake_strategy_values(self):
        """Test StakeStrategy enum values."""
        self.assertEqual(StakeStrategy.CONSERVATIVE.value, 1.0)
        self.assertEqual(StakeStrategy.MODERATE.value, 1.5)
        self.assertEqual(StakeStrategy.AGGRESSIVE.value, 2.0)
        self.assertEqual(StakeStrategy.YOLO.value, 3.0)

    def test_get_total_staked(self):
        """Test getting total staked amount."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create accounts
        ledger.get_or_create_account('expert_1', initial_balance=100.0)
        ledger.get_or_create_account('expert_2', initial_balance=100.0)

        # Place stakes
        pool.place_stake('expert_1', 'pred_1', 10.0, 1.5)
        pool.place_stake('expert_1', 'pred_2', 15.0, 2.0)
        pool.place_stake('expert_2', 'pred_3', 20.0, 1.5)

        # Total staked (all experts)
        total = pool.get_total_staked()
        self.assertEqual(total, 45.0)

        # Total staked by expert_1
        total_expert1 = pool.get_total_staked('expert_1')
        self.assertEqual(total_expert1, 25.0)

        # Total staked by expert_2
        total_expert2 = pool.get_total_staked('expert_2')
        self.assertEqual(total_expert2, 20.0)

    def test_get_stake_history(self):
        """Test getting stake history for an expert."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=5.0)

        # Create account
        ledger.get_or_create_account('expert_1', initial_balance=100.0)

        # Place stakes (with small delays to ensure different timestamps)
        import time
        stake1 = pool.place_stake('expert_1', 'pred_1', 10.0, 1.5)
        time.sleep(0.01)
        stake2 = pool.place_stake('expert_1', 'pred_2', 15.0, 2.0)
        time.sleep(0.01)
        stake3 = pool.place_stake('expert_1', 'pred_3', 20.0, 1.5)

        # Resolve one stake
        pool.resolve_stake(stake1.stake_id, success=True)

        # Get history
        history = pool.get_stake_history('expert_1')

        # Should have all stakes
        self.assertEqual(len(history), 3)

        # Should be sorted newest first
        self.assertEqual(history[0].stake_id, stake3.stake_id)
        self.assertEqual(history[1].stake_id, stake2.stake_id)
        self.assertEqual(history[2].stake_id, stake1.stake_id)

        # Should include resolved stakes
        self.assertEqual(history[2].status, 'won')

    def test_stake_validation(self):
        """Test stake validation (amount, multiplier, etc.)."""
        ledger = CreditLedger()
        pool = StakePool(ledger, min_stake=10.0)

        ledger.get_or_create_account('expert_1', initial_balance=100.0)

        # Test below minimum stake
        with self.assertRaises(ValueError) as ctx:
            pool.place_stake('expert_1', 'pred_1', 5.0, 1.5)
        self.assertIn('below minimum', str(ctx.exception))

        # Test invalid multiplier (too low)
        with self.assertRaises(ValueError) as ctx:
            pool.place_stake('expert_1', 'pred_1', 15.0, 0.5)
        self.assertIn('Multiplier must be', str(ctx.exception))

        # Test invalid multiplier (too high)
        with self.assertRaises(ValueError) as ctx:
            pool.place_stake('expert_1', 'pred_1', 15.0, 4.0)
        self.assertIn('Multiplier must be', str(ctx.exception))

    def test_stake_pool_initialization_validation(self):
        """Test StakePool initialization parameter validation."""
        ledger = CreditLedger()

        # Invalid max_stake_ratio (too high)
        with self.assertRaises(ValueError):
            StakePool(ledger, max_stake_ratio=1.5)

        # Invalid max_stake_ratio (zero)
        with self.assertRaises(ValueError):
            StakePool(ledger, max_stake_ratio=0.0)

        # Invalid min_stake (negative)
        with self.assertRaises(ValueError):
            StakePool(ledger, min_stake=-5.0)

        # Valid parameters
        pool = StakePool(ledger, max_stake_ratio=1.0, min_stake=1.0)
        self.assertIsNotNone(pool)


class TestCreditRouter(unittest.TestCase):
    """Test CreditRouter for credit-weighted routing."""

    def test_create_router(self):
        """Test creating CreditRouter."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        router = CreditRouter(ledger)

        self.assertEqual(router.min_weight, 0.1)
        self.assertEqual(router.temperature, 1.0)
        self.assertIs(router.ledger, ledger)

    def test_router_validation(self):
        """Test router parameter validation."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()

        # Invalid min_weight
        with self.assertRaises(ValueError):
            CreditRouter(ledger, min_weight=-0.1)

        with self.assertRaises(ValueError):
            CreditRouter(ledger, min_weight=1.5)

        # Invalid temperature
        with self.assertRaises(ValueError):
            CreditRouter(ledger, temperature=0.0)

        with self.assertRaises(ValueError):
            CreditRouter(ledger, temperature=-1.0)

    def test_compute_weights_equal_balance(self):
        """Test weight computation with equal balances."""
        from scripts.hubris.credit_router import CreditRouter, ExpertWeight

        ledger = CreditLedger()
        # All start with 100.0 balance
        ledger.get_or_create_account('expert_1')
        ledger.get_or_create_account('expert_2')
        ledger.get_or_create_account('expert_3')

        router = CreditRouter(ledger)
        weights = router.compute_weights(['expert_1', 'expert_2', 'expert_3'])

        # All should have equal normalized weights (1/3 each)
        self.assertEqual(len(weights), 3)
        for exp_id in ['expert_1', 'expert_2', 'expert_3']:
            self.assertAlmostEqual(weights[exp_id].normalized_weight, 1/3, places=5)
            self.assertIsInstance(weights[exp_id], ExpertWeight)

    def test_compute_weights_varied_balance(self):
        """Test weight computation with varied balances."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')
        acc2 = ledger.get_or_create_account('expert_2')
        acc3 = ledger.get_or_create_account('expert_3')

        # Vary balances significantly
        acc1.credit(300.0, 'test')  # 400.0
        # acc2 stays at 100.0
        acc3.debit(50.0, 'test')    # 50.0

        router = CreditRouter(ledger)
        weights = router.compute_weights(['expert_1', 'expert_2', 'expert_3'])

        # expert_1 should have highest weight
        self.assertGreater(
            weights['expert_1'].normalized_weight,
            weights['expert_2'].normalized_weight
        )
        # expert_2 should have higher or equal weight to expert_3
        # (min_weight floor may make them equal)
        self.assertGreaterEqual(
            weights['expert_2'].normalized_weight,
            weights['expert_3'].normalized_weight
        )

        # Weights should sum to 1.0
        total = sum(w.normalized_weight for w in weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_min_weight_floor(self):
        """Test that min_weight floor is enforced."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')
        acc2 = ledger.get_or_create_account('expert_2')

        # Give expert_1 very high balance
        acc1.credit(900.0, 'test')  # 1000.0
        # expert_2 stays at 100.0

        router = CreditRouter(ledger, min_weight=0.1)
        weights = router.compute_weights(['expert_1', 'expert_2'])

        # expert_2 should have at least min_weight (before renormalization)
        # After renormalization, all weights should be >= min_weight if possible
        self.assertGreaterEqual(weights['expert_2'].normalized_weight, 0.05)

        # Weights still sum to 1.0
        total = sum(w.normalized_weight for w in weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_temperature_effect(self):
        """Test that temperature affects weight distribution."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')
        acc2 = ledger.get_or_create_account('expert_2')

        # Different balances
        acc1.credit(100.0, 'test')  # 200.0
        # acc2 stays at 100.0

        # Low temperature (sharper distribution)
        router_cold = CreditRouter(ledger, temperature=0.1)
        weights_cold = router_cold.compute_weights(['expert_1', 'expert_2'])

        # High temperature (smoother distribution)
        router_hot = CreditRouter(ledger, temperature=10.0)
        weights_hot = router_hot.compute_weights(['expert_1', 'expert_2'])

        # With low temp, expert_1 should have much higher relative weight
        ratio_cold = (
            weights_cold['expert_1'].normalized_weight /
            weights_cold['expert_2'].normalized_weight
        )

        ratio_hot = (
            weights_hot['expert_1'].normalized_weight /
            weights_hot['expert_2'].normalized_weight
        )

        # Cold should have sharper difference
        self.assertGreater(ratio_cold, ratio_hot)

    def test_aggregate_predictions(self):
        """Test aggregating predictions with credit weighting."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')
        acc2 = ledger.get_or_create_account('expert_2')

        # Give expert_1 higher balance
        acc1.credit(100.0, 'test')  # 200.0
        # expert_2 stays at 100.0

        router = CreditRouter(ledger)

        # Create predictions
        pred1 = ExpertPrediction(
            expert_id='expert_1',
            expert_type='file',
            items=[('file1.py', 0.9), ('file2.py', 0.5)]
        )
        pred2 = ExpertPrediction(
            expert_id='expert_2',
            expert_type='file',
            items=[('file2.py', 0.8), ('file3.py', 0.6)]
        )

        predictions = {'expert_1': pred1, 'expert_2': pred2}
        result = router.aggregate_predictions(predictions)

        # Should return AggregatedPrediction
        self.assertIsInstance(result, AggregatedPrediction)

        # Should have items
        self.assertGreater(len(result.items), 0)

        # Contributing experts
        self.assertIn('expert_1', result.contributing_experts)
        self.assertIn('expert_2', result.contributing_experts)

        # file1.py should rank high (expert_1 has high weight and confidence)
        top_file = result.items[0][0]
        # file1.py or file2.py should be top (both strong signals)
        self.assertIn(top_file, ['file1.py', 'file2.py'])

    def test_aggregate_predictions_empty(self):
        """Test aggregation with no predictions."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        router = CreditRouter(ledger)

        result = router.aggregate_predictions({})

        self.assertEqual(len(result.items), 0)
        self.assertEqual(len(result.contributing_experts), 0)
        self.assertEqual(result.confidence, 0.0)

    def test_select_expert(self):
        """Test expert selection by credit."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')
        acc2 = ledger.get_or_create_account('expert_2')
        acc3 = ledger.get_or_create_account('expert_3')

        # Give expert_2 highest balance
        acc2.credit(200.0, 'test')  # 300.0
        # Others stay at 100.0

        router = CreditRouter(ledger)
        selected = router.select_expert(
            context={'query': 'test'},
            available=['expert_1', 'expert_2', 'expert_3']
        )

        # Should select expert_2 (highest weight)
        self.assertEqual(selected, 'expert_2')

        # Should record in history
        self.assertEqual(len(router.routing_history), 1)
        self.assertEqual(router.routing_history[0]['selected'], 'expert_2')

    def test_select_expert_no_available(self):
        """Test expert selection with no available experts."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        router = CreditRouter(ledger)

        with self.assertRaises(ValueError):
            router.select_expert(context={}, available=[])

    def test_confidence_boost_high_credit(self):
        """Test confidence boost for high-credit experts."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')
        acc2 = ledger.get_or_create_account('expert_2')

        # Give expert_1 very high balance (> 150)
        acc1.credit(200.0, 'test')  # 300.0
        # expert_2 stays at 100.0

        router = CreditRouter(ledger)
        weights = router.compute_weights(['expert_1', 'expert_2'])

        # expert_1 should have confidence boost > 1.0
        self.assertGreater(weights['expert_1'].confidence_boost, 1.0)

        # expert_2 should have boost = 1.0 (not enough credit)
        self.assertEqual(weights['expert_2'].confidence_boost, 1.0)

    def test_confidence_boost_in_aggregation(self):
        """Test that confidence boost affects aggregation."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        acc1 = ledger.get_or_create_account('expert_1')

        # Give expert_1 high balance (> 150)
        acc1.credit(150.0, 'test')  # 250.0

        router = CreditRouter(ledger)

        # Single prediction with boost
        pred1 = ExpertPrediction(
            expert_id='expert_1',
            expert_type='file',
            items=[('file1.py', 0.5)]
        )

        result = router.aggregate_predictions({'expert_1': pred1})

        # Check metadata contains boost info
        self.assertIn('boosts', result.metadata)
        self.assertGreater(result.metadata['boosts']['expert_1'], 1.0)

    def test_routing_stats(self):
        """Test routing statistics tracking."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        ledger.get_or_create_account('expert_1')
        ledger.get_or_create_account('expert_2')

        router = CreditRouter(ledger)

        # Make several routing decisions
        router.select_expert({'query': 'test1'}, ['expert_1', 'expert_2'])
        router.select_expert({'query': 'test2'}, ['expert_1', 'expert_2'])
        router.select_expert({'query': 'test3'}, ['expert_1', 'expert_2'])

        stats = router.get_routing_stats()

        # Should have stats
        self.assertEqual(stats['total_routings'], 3)
        self.assertIn('expert_usage', stats)
        self.assertIn('average_weights', stats)
        self.assertIn('recent_decisions', stats)

        # Recent decisions should be limited to last 10
        self.assertLessEqual(len(stats['recent_decisions']), 10)

    def test_routing_stats_empty(self):
        """Test routing stats with no history."""
        from scripts.hubris.credit_router import CreditRouter

        ledger = CreditLedger()
        router = CreditRouter(ledger)

        stats = router.get_routing_stats()

        self.assertEqual(stats['total_routings'], 0)
        self.assertEqual(stats['expert_usage'], {})
        self.assertEqual(stats['average_weights'], {})
        self.assertEqual(stats['recent_decisions'], [])


class TestFeedbackCollector(unittest.TestCase):
    """Test feedback collection and expert credit updates."""

    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for predictions
        self.temp_dir = tempfile.mkdtemp()
        self.predictions_dir = Path(self.temp_dir) / 'predictions'

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_prediction_recorder_basic(self):
        """Test recording and retrieving predictions."""
        from scripts.hubris.feedback_collector import PredictionRecorder

        recorder = PredictionRecorder(self.predictions_dir)

        # Record a prediction
        pred = recorder.record_prediction(
            prediction_id='test_pred_1',
            expert_id='file_expert',
            predicted_files=['file1.py', 'file2.py'],
            confidence=0.8,
            context={'task': 'Add feature'}
        )

        self.assertEqual(pred.prediction_id, 'test_pred_1')
        self.assertEqual(pred.expert_id, 'file_expert')
        self.assertEqual(len(pred.predicted_files), 2)
        self.assertEqual(pred.confidence, 0.8)

        # Retrieve pending predictions
        pending = recorder.get_pending_predictions()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].prediction_id, 'test_pred_1')

        # Filter by expert_id
        expert_preds = recorder.get_pending_predictions(expert_id='file_expert')
        self.assertEqual(len(expert_preds), 1)

        other_preds = recorder.get_pending_predictions(expert_id='other_expert')
        self.assertEqual(len(other_preds), 0)

    def test_feedback_processor_commit_outcome(self):
        """Test processing commit outcomes."""
        from scripts.hubris.feedback_collector import (
            PredictionRecorder, FeedbackProcessor
        )

        recorder = PredictionRecorder(self.predictions_dir)
        ledger = CreditLedger()
        attributor = ValueAttributor()
        processor = FeedbackProcessor(ledger, attributor, recorder)

        # Record a prediction
        recorder.record_prediction(
            prediction_id='pred_1',
            expert_id='file_expert',
            predicted_files=['file1.py', 'file2.py', 'file3.py'],
            confidence=0.8,
            context={}
        )

        # Process commit outcome (2 out of 3 files correct)
        actual_files = ['file1.py', 'file2.py']
        credit_updates = processor.process_commit_outcome(
            commit_hash='abc123',
            actual_files=actual_files,
            prediction_id='pred_1'
        )

        # Should have credit update for file_expert
        self.assertIn('file_expert', credit_updates)

        # Accuracy is 2/3 = 0.67, which is > 0.5, so positive signal
        # With confidence 0.8, magnitude 0.67, should get positive credit
        self.assertGreater(credit_updates['file_expert'], 0)

        # Prediction should be resolved
        pending = recorder.get_pending_predictions()
        self.assertEqual(len(pending), 0)

    def test_feedback_processor_updates_credits(self):
        """Test that feedback processor updates credit ledger."""
        from scripts.hubris.feedback_collector import (
            PredictionRecorder, FeedbackProcessor
        )

        recorder = PredictionRecorder(self.predictions_dir)
        ledger = CreditLedger()
        attributor = ValueAttributor()
        processor = FeedbackProcessor(ledger, attributor, recorder)

        # Record prediction
        recorder.record_prediction(
            prediction_id='pred_1',
            expert_id='test_expert',
            predicted_files=['file1.py'],
            confidence=0.9,
            context={}
        )

        # Get initial balance
        account = ledger.get_or_create_account('test_expert')
        initial_balance = account.balance

        # Process successful commit (100% accuracy)
        processor.process_commit_outcome(
            commit_hash='xyz789',
            actual_files=['file1.py'],
            prediction_id='pred_1'
        )

        # Balance should have increased
        self.assertGreater(account.balance, initial_balance)

        # Should have a transaction
        self.assertGreater(len(account.transactions), 0)

        # Transaction should be from commit_result
        latest_tx = account.transactions[-1]
        self.assertEqual(latest_tx.reason, 'commit_result')

    def test_on_pre_commit_records_prediction(self):
        """Test on_pre_commit hook function."""
        from scripts.hubris.feedback_collector import (
            PredictionRecorder, on_pre_commit
        )

        recorder = PredictionRecorder(self.predictions_dir)

        # Call pre-commit hook
        prediction_id = on_pre_commit(
            task_description='Add authentication',
            expert_id='file_expert',
            predicted_files=['auth.py', 'test_auth.py'],
            confidence=0.75,
            recorder=recorder
        )

        # Should return a prediction ID
        self.assertIsNotNone(prediction_id)
        self.assertTrue(prediction_id.startswith('pred_'))

        # Prediction should be in pending
        pending = recorder.get_pending_predictions()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].prediction_id, prediction_id)
        self.assertEqual(pending[0].expert_id, 'file_expert')

    def test_on_post_commit_resolves_prediction(self):
        """Test on_post_commit hook function."""
        from scripts.hubris.feedback_collector import (
            PredictionRecorder, on_pre_commit, on_post_commit
        )

        recorder = PredictionRecorder(self.predictions_dir)
        ledger = CreditLedger()

        # Record prediction via pre-commit hook
        prediction_id = on_pre_commit(
            task_description='Fix bug',
            expert_id='file_expert',
            predicted_files=['bugfix.py', 'test_bugfix.py'],
            confidence=0.8,
            recorder=recorder
        )

        # Verify prediction is pending
        pending_before = recorder.get_pending_predictions()
        self.assertEqual(len(pending_before), 1)

        # Call post-commit hook
        actual_files = ['bugfix.py', 'test_bugfix.py', 'docs.md']
        credit_updates = on_post_commit(
            commit_hash='def456',
            actual_files=actual_files,
            prediction_id=prediction_id,
            ledger=ledger,
            recorder=recorder
        )

        # Should have credit update
        self.assertIn('file_expert', credit_updates)

        # Prediction should be resolved
        pending_after = recorder.get_pending_predictions()
        self.assertEqual(len(pending_after), 0)

        # Ledger should have been updated
        account = ledger.get_or_create_account('file_expert')
        self.assertGreater(len(account.transactions), 0)


if __name__ == '__main__':
    unittest.main()
