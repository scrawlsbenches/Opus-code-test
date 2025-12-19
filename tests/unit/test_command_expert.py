#!/usr/bin/env python3
"""
Unit tests for CommandExpert.

Tests the command prediction micro-expert that learns from
Bash tool call history.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'hubris'))

from experts.command_expert import CommandExpert


class TestCommandExpertInit(unittest.TestCase):
    """Test CommandExpert initialization."""

    def test_default_init(self):
        """Test default initialization."""
        expert = CommandExpert()
        self.assertEqual(expert.expert_id, "command_expert")
        self.assertEqual(expert.expert_type, "command")
        self.assertEqual(expert.version, "1.0.0")
        self.assertIsNotNone(expert.model_data)

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        expert = CommandExpert(
            expert_id="my_command_expert",
            version="2.0.0"
        )
        self.assertEqual(expert.expert_id, "my_command_expert")
        self.assertEqual(expert.version, "2.0.0")

    def test_model_data_structure(self):
        """Test model_data has required keys."""
        expert = CommandExpert()
        required_keys = [
            'command_templates',
            'keyword_to_commands',
            'command_frequency',
            'context_patterns',
            'python_c_patterns',
            'total_commands'
        ]
        for key in required_keys:
            self.assertIn(key, expert.model_data)


class TestCommandExpertPredict(unittest.TestCase):
    """Test CommandExpert prediction."""

    def setUp(self):
        """Set up test expert with mock data."""
        self.expert = CommandExpert()
        self.expert.model_data = {
            'command_templates': {
                'git status': {'count': 10, 'examples': ['git status'], 'keywords': ['git', 'status']},
                'ls -la': {'count': 5, 'examples': ['ls -la'], 'keywords': ['list', 'files']},
                'python -m pytest': {'count': 8, 'examples': ['python -m pytest'], 'keywords': ['test', 'pytest']}
            },
            'keyword_to_commands': {
                'git': {'git status': 10, 'git diff': 3},
                'status': {'git status': 10},
                'list': {'ls -la': 5},
                'files': {'ls -la': 5},
                'test': {'python -m pytest': 8},
                'pytest': {'python -m pytest': 8}
            },
            'command_frequency': {
                'git status': 10,
                'ls -la': 5,
                'python -m pytest': 8,
                'git diff': 3
            },
            'context_patterns': {},
            'python_c_patterns': {},
            'total_commands': 26
        }

    def test_predict_empty_query(self):
        """Test prediction with empty query returns error."""
        result = self.expert.predict({'query': ''})
        self.assertEqual(result.items, [])
        self.assertIn('error', result.metadata)

    def test_predict_git_query(self):
        """Test prediction for git-related query."""
        result = self.expert.predict({'query': 'check git status', 'top_n': 3})
        self.assertGreater(len(result.items), 0)
        # git status should be top prediction
        top_cmd, top_conf = result.items[0]
        self.assertEqual(top_cmd, 'git status')
        self.assertGreater(top_conf, 0)

    def test_predict_test_query(self):
        """Test prediction for test-related query."""
        result = self.expert.predict({'query': 'run pytest tests', 'top_n': 3})
        self.assertGreater(len(result.items), 0)
        # Should include pytest command
        commands = [cmd for cmd, _ in result.items]
        self.assertIn('python -m pytest', commands)

    def test_predict_respects_top_n(self):
        """Test that top_n limits results."""
        result = self.expert.predict({'query': 'git status test files', 'top_n': 2})
        self.assertLessEqual(len(result.items), 2)

    def test_predict_normalizes_scores(self):
        """Test that scores are normalized to 0-1."""
        result = self.expert.predict({'query': 'git status', 'top_n': 5})
        for cmd, score in result.items:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_predict_metadata(self):
        """Test prediction metadata."""
        result = self.expert.predict({'query': 'check git status', 'top_n': 3})
        self.assertIn('query', result.metadata)
        self.assertIn('keywords', result.metadata)
        self.assertEqual(result.metadata['query'], 'check git status')


class TestCommandExpertKeywordExtraction(unittest.TestCase):
    """Test keyword extraction."""

    def setUp(self):
        self.expert = CommandExpert()

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        keywords = self.expert._extract_keywords("run the python tests")
        self.assertIn('run', keywords)
        self.assertIn('python', keywords)
        self.assertIn('tests', keywords)
        # Stop words should be excluded
        self.assertNotIn('the', keywords)

    def test_extract_keywords_filters_short(self):
        """Test that short words are filtered."""
        keywords = self.expert._extract_keywords("a to go do it")
        # All words <= 2 chars should be filtered
        self.assertEqual(len(keywords), 0)

    def test_extract_keywords_lowercase(self):
        """Test keywords are lowercased."""
        keywords = self.expert._extract_keywords("Run PYTHON Tests")
        self.assertIn('run', keywords)
        self.assertIn('python', keywords)
        self.assertIn('tests', keywords)


class TestCommandExpertNormalization(unittest.TestCase):
    """Test command normalization."""

    def setUp(self):
        self.expert = CommandExpert()

    def test_normalize_user_path(self):
        """Test user path normalization."""
        cmd = "/home/john/project/script.py"
        normalized = self.expert._normalize_command(cmd)
        self.assertIn('/home/USER/', normalized)

    def test_normalize_truncates_long(self):
        """Test long commands are truncated."""
        cmd = "echo " + "x" * 300
        normalized = self.expert._normalize_command(cmd)
        self.assertLessEqual(len(normalized), 210)  # 200 + "..."


class TestCommandExpertTrain(unittest.TestCase):
    """Test CommandExpert training."""

    def test_train_empty_directory(self):
        """Test training on non-existent directory."""
        expert = CommandExpert()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = expert.train(Path(tmpdir) / "nonexistent")
            self.assertIn('error', result)

    def test_train_with_actions(self):
        """Test training with mock action files."""
        expert = CommandExpert()

        with tempfile.TemporaryDirectory() as tmpdir:
            actions_dir = Path(tmpdir)
            date_dir = actions_dir / "2025-12-18"
            date_dir.mkdir(parents=True)

            # Create mock action files
            for i in range(5):
                action = {
                    'id': f'A-{i}',
                    'session_id': 'test-session',
                    'context': {
                        'tool': 'Bash',
                        'input': {
                            'input': {
                                'command': f'git status',
                                'description': 'Check git status'
                            }
                        }
                    }
                }
                with open(date_dir / f'A-2025-{i:04d}.json', 'w') as f:
                    json.dump(action, f)

            # Train
            stats = expert.train(actions_dir)

            self.assertEqual(stats['bash_actions'], 5)
            self.assertEqual(stats['sessions'], 1)
            self.assertGreater(expert.model_data['total_commands'], 0)


class TestCommandExpertSerialization(unittest.TestCase):
    """Test save/load functionality."""

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        expert = CommandExpert(
            expert_id="test_expert",
            version="1.0.0"
        )
        expert.model_data['command_frequency']['git status'] = 10
        expert.trained_on_commits = 100
        expert.trained_on_sessions = 5

        # Convert to dict
        data = expert.to_dict()

        # Recreate from dict
        loaded = CommandExpert.from_dict(data)

        self.assertEqual(loaded.expert_id, "test_expert")
        self.assertEqual(loaded.version, "1.0.0")
        self.assertEqual(loaded.trained_on_commits, 100)
        self.assertEqual(loaded.model_data['command_frequency']['git status'], 10)

    def test_save_load_file(self):
        """Test saving and loading from file."""
        expert = CommandExpert()
        expert.model_data['total_commands'] = 50

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            expert.save(path)
            loaded = CommandExpert.load(path)
            self.assertEqual(loaded.model_data['total_commands'], 50)
        finally:
            path.unlink()


class TestCommandExpertEvaluate(unittest.TestCase):
    """Test evaluation functionality."""

    def setUp(self):
        self.expert = CommandExpert()
        self.expert.model_data = {
            'command_templates': {},
            'keyword_to_commands': {
                'status': {'git status': 10},
                'git': {'git status': 10}
            },
            'command_frequency': {'git status': 10},
            'context_patterns': {},
            'python_c_patterns': {},
            'total_commands': 10
        }

    def test_evaluate_empty(self):
        """Test evaluation with empty test set."""
        metrics = self.expert.evaluate([])
        self.assertEqual(metrics.test_examples, 0)

    def test_evaluate_with_actions(self):
        """Test evaluation with test actions."""
        test_actions = [
            {'query': 'check git status', 'command': 'git status'},
            {'query': 'show git status', 'command': 'git status'}
        ]

        metrics = self.expert.evaluate(test_actions)

        self.assertEqual(metrics.test_examples, 2)
        self.assertGreaterEqual(metrics.mrr, 0.0)
        self.assertLessEqual(metrics.mrr, 1.0)


if __name__ == '__main__':
    unittest.main()
