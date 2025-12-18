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


if __name__ == '__main__':
    unittest.main()
