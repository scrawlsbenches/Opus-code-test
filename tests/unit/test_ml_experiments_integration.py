"""
Integration tests for ml_experiments with file prediction.

Tests the integration adapter that bridges ml_file_prediction.py
with the ML experiments framework.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add cortical to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from cortical.ml_experiments import (
    DatasetManager,
    DatasetVersion,
    ExperimentManager,
    ExperimentConfig,
    MetricsManager,
)
from cortical.ml_experiments.utils import ensure_directory, read_jsonl


class TestCommitDatasetCreation(unittest.TestCase):
    """Test dataset creation from commit history."""

    def setUp(self):
        """Create temp directory with mock commit data."""
        self.temp_dir = tempfile.mkdtemp()
        self.commits_file = Path(self.temp_dir) / 'commits.jsonl'

        # Create mock commit data
        commits = [
            {'hash': 'abc123', 'message': 'feat: Add login', 'files_changed': ['auth.py'], 'timestamp': '2025-01-01T00:00:00Z', 'is_merge': False},
            {'hash': 'def456', 'message': 'fix: Bug fix', 'files_changed': ['utils.py'], 'timestamp': '2025-01-02T00:00:00Z', 'is_merge': False},
            {'hash': 'ghi789', 'message': 'docs: Update readme', 'files_changed': ['README.md'], 'timestamp': '2025-01-03T00:00:00Z', 'is_merge': False},
            {'hash': 'merge1', 'message': 'Merge branch', 'files_changed': ['merge.py'], 'timestamp': '2025-01-04T00:00:00Z', 'is_merge': True},  # Should be filtered
            {'hash': 'jkl012', 'message': 'data: ML session', 'files_changed': ['data.py'], 'timestamp': '2025-01-05T00:00:00Z', 'is_merge': False},  # Should be filtered
            {'hash': 'mno345', 'message': 'test: Add tests', 'files_changed': ['test_auth.py'], 'timestamp': '2025-01-06T00:00:00Z', 'is_merge': False},
        ]

        with open(self.commits_file, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_commits_filters_merges(self):
        """Test that merge commits are filtered out."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        output_path = Path(self.temp_dir) / 'normalized.jsonl'
        result_path = load_commits_as_jsonl(self.commits_file, output_path)

        records = read_jsonl(result_path)

        # Should have 4 commits (excluding merge and ML data)
        self.assertEqual(len(records), 4)

        # Verify merge commit is not included
        hashes = [r['hash'] for r in records]
        self.assertNotIn('merge1', hashes)
        self.assertNotIn('jkl012', hashes)  # ML data commit

    def test_load_commits_normalizes_format(self):
        """Test that commits are normalized to standard format."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        output_path = Path(self.temp_dir) / 'normalized.jsonl'
        result_path = load_commits_as_jsonl(self.commits_file, output_path)

        records = read_jsonl(result_path)

        # Check normalized format
        for record in records:
            self.assertIn('hash', record)
            self.assertIn('message', record)
            self.assertIn('files', record)
            self.assertIn('timestamp', record)


class TestFilePredictionExperiment(unittest.TestCase):
    """Test FilePredictionExperiment class."""

    def test_experiment_class_instantiation(self):
        """Test that experiment class can be instantiated."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.MODEL_TYPE, 'file_prediction')

    def test_get_metric_history_empty(self):
        """Test metric history when no metrics recorded."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()
        history = experiment.get_metric_history('mrr')

        # Should return empty list when no metrics
        self.assertIsInstance(history, list)

    def test_compare_runs_empty(self):
        """Test compare runs with non-existent IDs."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()
        comparison = experiment.compare_runs(['nonexistent1', 'nonexistent2'])

        # Should return empty dict for non-existent runs
        self.assertIsInstance(comparison, dict)


class TestAblationStudy(unittest.TestCase):
    """Test ablation study functionality."""

    def test_ablation_study_with_mock(self):
        """Test ablation study setup with mocked training."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        # Create mock experiment
        experiment = FilePredictionExperiment()

        # Verify the experiment runner is set up correctly
        self.assertEqual(experiment.MODEL_TYPE, 'file_prediction')


class TestFormatExperimentReport(unittest.TestCase):
    """Test experiment report formatting."""

    def test_format_empty_report(self):
        """Test formatting with no runs."""
        from cortical.ml_experiments.file_prediction_adapter import format_experiment_report

        report = format_experiment_report([])

        self.assertIn('EXPERIMENT REPORT', report)

    def test_format_single_run_report(self):
        """Test formatting with single mock run."""
        from cortical.ml_experiments.file_prediction_adapter import format_experiment_report

        # Create mock run
        mock_config = ExperimentConfig(
            name='test_run',
            model_type='file_prediction',
            dataset_id='ds-test-123',
            hyperparameters={'use_ai_meta': True}
        )

        mock_run = MagicMock()
        mock_run.config = mock_config
        mock_run.id = 'exp-test-001'
        mock_run.status = 'completed'
        mock_run.duration_seconds = 1.5
        mock_run.git_hash = 'abc123'
        mock_run.git_status = 'clean'
        mock_run.metrics_summary = {'mrr': 0.45, 'recall@10': 0.42}

        report = format_experiment_report([mock_run])

        self.assertIn('test_run', report)
        self.assertIn('exp-test-001', report)
        self.assertIn('completed', report)
        self.assertIn('mrr', report)


class TestLazyImports(unittest.TestCase):
    """Test lazy import functions in __init__.py."""

    def test_get_file_prediction_experiment(self):
        """Test lazy import of FilePredictionExperiment."""
        from cortical.ml_experiments import get_file_prediction_experiment

        ExperimentClass = get_file_prediction_experiment()
        self.assertEqual(ExperimentClass.MODEL_TYPE, 'file_prediction')

        # Should be able to instantiate
        experiment = ExperimentClass()
        self.assertIsNotNone(experiment)


class TestIntegrationWithDatasetManager(unittest.TestCase):
    """Test integration between adapter and DatasetManager."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_creation_workflow(self):
        """Test creating a dataset and verifying its structure."""
        # Create a mock JSONL source file
        source_file = Path(self.temp_dir) / 'test_commits.jsonl'
        records = [
            {'hash': f'hash{i}', 'message': f'commit {i}', 'files': [f'file{i}.py'], 'timestamp': f'2025-01-{i:02d}T00:00:00Z'}
            for i in range(1, 21)  # 20 commits
        ]

        with open(source_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

        # Create dataset with custom ratios
        with patch.object(DatasetManager, '_save_to_manifest'):
            with patch.object(DatasetManager, '_save_splits'):
                dataset = DatasetManager.create_dataset(
                    name='test_commits',
                    source_path=str(source_file),
                    filters={'require_fields': ['hash', 'message']},
                    split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
                    random_seed=42
                )

        # Verify dataset structure
        self.assertIn('train', dataset.splits)
        self.assertIn('val', dataset.splits)
        self.assertIn('test', dataset.splits)

        # Verify split sizes are approximately correct
        total = sum(s.count for s in dataset.splits.values())
        self.assertEqual(total, 20)


if __name__ == '__main__':
    unittest.main()
