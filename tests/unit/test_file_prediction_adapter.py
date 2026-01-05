"""
Tests for cortical.ml_experiments.file_prediction_adapter module.

This module provides comprehensive tests for the file prediction ML
experiment adapter, including:
- CommitExample dataclass
- load_commits_as_jsonl function
- create_commit_dataset function
- FilePredictionExperiment class
- run_ablation_study function
- format_experiment_report function
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from dataclasses import dataclass
from typing import Any, Dict, List

# Mark all tests in this module as slow (disk-heavy)
pytestmark = pytest.mark.slow


class TestCommitExample:
    """Tests for CommitExample dataclass."""

    def test_commit_example_creation(self):
        """Test basic CommitExample creation."""
        from cortical.ml_experiments.file_prediction_adapter import CommitExample

        example = CommitExample(
            commit_hash="abc123",
            message="feat: add feature",
            files_changed=["file1.py", "file2.py"],
            timestamp="2025-01-01T00:00:00Z",
            is_merge=False
        )

        assert example.commit_hash == "abc123"
        assert example.message == "feat: add feature"
        assert example.files_changed == ["file1.py", "file2.py"]
        assert example.timestamp == "2025-01-01T00:00:00Z"
        assert example.is_merge is False

    def test_commit_example_default_is_merge(self):
        """Test CommitExample with default is_merge value."""
        from cortical.ml_experiments.file_prediction_adapter import CommitExample

        example = CommitExample(
            commit_hash="abc123",
            message="fix: bug",
            files_changed=["file.py"],
            timestamp="2025-01-01T00:00:00Z"
        )

        assert example.is_merge is False

    def test_commit_example_to_dict(self):
        """Test CommitExample serialization to dictionary."""
        from cortical.ml_experiments.file_prediction_adapter import CommitExample

        example = CommitExample(
            commit_hash="abc123",
            message="feat: add feature",
            files_changed=["file1.py", "file2.py"],
            timestamp="2025-01-01T00:00:00Z",
            is_merge=True
        )

        result = example.to_dict()

        assert result == {
            'hash': 'abc123',
            'message': 'feat: add feature',
            'files': ['file1.py', 'file2.py'],
            'timestamp': '2025-01-01T00:00:00Z',
            'is_merge': True
        }

    def test_commit_example_to_dict_empty_files(self):
        """Test CommitExample with empty files list."""
        from cortical.ml_experiments.file_prediction_adapter import CommitExample

        example = CommitExample(
            commit_hash="abc123",
            message="empty commit",
            files_changed=[],
            timestamp="2025-01-01T00:00:00Z"
        )

        result = example.to_dict()
        assert result['files'] == []


class TestLoadCommitsFromCALI:
    """Tests for _load_commits_from_cali function."""

    def test_load_commits_cali_not_available(self):
        """Test loading commits when CALI is not available."""
        from cortical.ml_experiments import file_prediction_adapter

        # Temporarily set CALI_AVAILABLE to False
        original = file_prediction_adapter.CALI_AVAILABLE
        file_prediction_adapter.CALI_AVAILABLE = False

        try:
            result = file_prediction_adapter._load_commits_from_cali()
            assert result == []
        finally:
            file_prediction_adapter.CALI_AVAILABLE = original

    def test_load_commits_cali_dir_not_exists(self, tmp_path):
        """Test loading commits when CALI dir doesn't exist."""
        from cortical.ml_experiments import file_prediction_adapter

        original_available = file_prediction_adapter.CALI_AVAILABLE
        original_dir = file_prediction_adapter.CALI_DIR

        file_prediction_adapter.CALI_AVAILABLE = True
        file_prediction_adapter.CALI_DIR = tmp_path / "nonexistent"

        try:
            result = file_prediction_adapter._load_commits_from_cali()
            assert result == []
        finally:
            file_prediction_adapter.CALI_AVAILABLE = original_available
            file_prediction_adapter.CALI_DIR = original_dir


class TestLoadCommitsAsJsonl:
    """Tests for load_commits_as_jsonl function."""

    def test_load_commits_file_not_found(self, tmp_path):
        """Test load_commits_as_jsonl with non-existent source file."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        non_existent = tmp_path / "nonexistent.jsonl"
        output_path = tmp_path / "output.jsonl"

        with pytest.raises(FileNotFoundError):
            load_commits_as_jsonl(source_path=non_existent, output_path=output_path, use_cali=False)

    def test_load_commits_normalizes_records(self, tmp_path):
        """Test that load_commits_as_jsonl normalizes records correctly."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        # Create source file with sample commits
        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat: add feature", "files_changed": ["file1.py"], "timestamp": "2025-01-01T00:00:00Z", "is_merge": False},
            {"hash": "def456", "message": "fix: bug fix", "files_changed": ["file2.py"], "timestamp": "2025-01-02T00:00:00Z", "is_merge": False},
        ]

        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        output_path = tmp_path / "output.jsonl"

        result = load_commits_as_jsonl(source_path=source_path, output_path=output_path, use_cali=False)

        assert result == output_path
        assert output_path.exists()

        # Read and verify normalized output
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_load_commits_skips_merge_commits(self, tmp_path):
        """Test that merge commits are skipped."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat: add feature", "files_changed": ["file1.py"], "timestamp": "2025-01-01T00:00:00Z", "is_merge": False},
            {"hash": "merge123", "message": "Merge branch", "files_changed": ["file2.py"], "timestamp": "2025-01-02T00:00:00Z", "is_merge": True},
        ]

        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        output_path = tmp_path / "output.jsonl"

        load_commits_as_jsonl(source_path=source_path, output_path=output_path, use_cali=False)

        with open(output_path) as f:
            lines = f.readlines()

        # Only non-merge commit should be written
        assert len(lines) == 1

    def test_load_commits_skips_ml_data_commits(self, tmp_path):
        """Test that ML data commits are skipped."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat: add feature", "files_changed": ["file1.py"], "timestamp": "2025-01-01T00:00:00Z", "is_merge": False},
            {"hash": "ml123", "message": "data: ML training data update", "files_changed": ["data.jsonl"], "timestamp": "2025-01-02T00:00:00Z", "is_merge": False},
        ]

        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        output_path = tmp_path / "output.jsonl"

        load_commits_as_jsonl(source_path=source_path, output_path=output_path, use_cali=False)

        with open(output_path) as f:
            lines = f.readlines()

        # ML data commit should be skipped
        assert len(lines) == 1

    def test_load_commits_skips_empty_files(self, tmp_path):
        """Test that commits with no files are skipped."""
        from cortical.ml_experiments.file_prediction_adapter import load_commits_as_jsonl

        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat: add feature", "files_changed": ["file1.py"], "timestamp": "2025-01-01T00:00:00Z"},
            {"hash": "empty123", "message": "empty commit", "files_changed": [], "timestamp": "2025-01-02T00:00:00Z"},
        ]

        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        output_path = tmp_path / "output.jsonl"

        load_commits_as_jsonl(source_path=source_path, output_path=output_path, use_cali=False)

        with open(output_path) as f:
            lines = f.readlines()

        # Empty files commit should be skipped
        assert len(lines) == 1

    def test_load_commits_default_output_path(self, tmp_path):
        """Test that default output path is created."""
        from cortical.ml_experiments import file_prediction_adapter

        # Temporarily change cache dir
        original_cache = file_prediction_adapter.DATASET_CACHE
        file_prediction_adapter.DATASET_CACHE = tmp_path / "datasets"

        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat", "files_changed": ["file.py"], "timestamp": "2025-01-01T00:00:00Z"},
        ]

        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        try:
            result = file_prediction_adapter.load_commits_as_jsonl(
                source_path=source_path,
                output_path=None,
                use_cali=False
            )

            expected_path = tmp_path / "datasets" / "commits_normalized.jsonl"
            assert result == expected_path
            assert result.exists()
        finally:
            file_prediction_adapter.DATASET_CACHE = original_cache


class TestFilePredictionExperiment:
    """Tests for FilePredictionExperiment class."""

    def test_init(self):
        """Test FilePredictionExperiment initialization."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        assert experiment.MODEL_TYPE == 'file_prediction'
        assert experiment._ml_prediction_module is None

    def test_get_ml_module_import_error(self):
        """Test _get_ml_module raises ImportError when module not available."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch.dict('sys.modules', {'ml_file_prediction': None}):
            with pytest.raises(ImportError):
                experiment._get_ml_module()

    def test_records_to_examples_empty_list(self):
        """Test _records_to_examples with empty list."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        # Mock the ML module
        mock_ml = {
            'TrainingExample': MagicMock(return_value=MagicMock(files_changed=[]))
        }
        experiment._ml_prediction_module = mock_ml

        with patch('cortical.ml_experiments.file_prediction_adapter.FilePredictionExperiment._get_ml_module', return_value=mock_ml):
            result = experiment._records_to_examples([])

        assert result == []

    def test_get_metric_history(self):
        """Test get_metric_history delegates to MetricsManager."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch('cortical.ml_experiments.file_prediction_adapter.MetricsManager') as mock_manager:
            mock_manager.get_metric_history.return_value = [
                ("2025-01-01", "exp1", 0.5),
                ("2025-01-02", "exp2", 0.6),
            ]

            result = experiment.get_metric_history('mrr', 'val')

            mock_manager.get_metric_history.assert_called_once_with('mrr', 'val')
            assert len(result) == 2

    def test_compare_runs(self):
        """Test compare_runs delegates to ExperimentManager."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch('cortical.ml_experiments.file_prediction_adapter.ExperimentManager') as mock_manager:
            mock_manager.compare_runs.return_value = {
                'exp1': {'mrr': 0.5},
                'exp2': {'mrr': 0.6}
            }

            result = experiment.compare_runs(['exp1', 'exp2'])

            mock_manager.compare_runs.assert_called_once_with(['exp1', 'exp2'])
            assert 'exp1' in result
            assert 'exp2' in result

    def test_get_best_run(self):
        """Test get_best_run delegates to ExperimentManager."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        mock_run = MagicMock()
        mock_run.metrics_summary = {'mrr': 0.7}

        with patch('cortical.ml_experiments.file_prediction_adapter.ExperimentManager') as mock_manager:
            mock_manager.get_best_run.return_value = mock_run

            result = experiment.get_best_run('mrr', higher_is_better=True)

            mock_manager.get_best_run.assert_called_once_with(
                model_type='file_prediction',
                metric_name='mrr',
                higher_is_better=True
            )
            assert result == mock_run

    def test_get_best_run_none(self):
        """Test get_best_run returns None when no runs exist."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch('cortical.ml_experiments.file_prediction_adapter.ExperimentManager') as mock_manager:
            mock_manager.get_best_run.return_value = None

            result = experiment.get_best_run('mrr')

            assert result is None

    def test_detect_regression_no_regressions(self):
        """Test detect_regression when no regressions exist."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch('cortical.ml_experiments.file_prediction_adapter.MetricsManager') as mock_manager:
            mock_manager.detect_regression.return_value = None

            result = experiment.detect_regression({'mrr': 0.5, 'recall@10': 0.6})

            assert result == []
            assert mock_manager.detect_regression.call_count == 2

    def test_detect_regression_with_regressions(self):
        """Test detect_regression when regressions are detected."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        regression_data = {'metric': 'mrr', 'baseline': 0.6, 'current': 0.5, 'drop_pct': 16.67}

        with patch('cortical.ml_experiments.file_prediction_adapter.MetricsManager') as mock_manager:
            mock_manager.detect_regression.side_effect = [regression_data, None]

            result = experiment.detect_regression({'mrr': 0.5, 'recall@10': 0.6})

            assert len(result) == 1
            assert result[0] == regression_data

    def test_run_dataset_not_found(self):
        """Test run raises ValueError when dataset not found."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch('cortical.ml_experiments.file_prediction_adapter.DatasetManager') as mock_manager:
            mock_manager.load_dataset.return_value = None

            with pytest.raises(ValueError, match="Dataset not found"):
                experiment.run(
                    name="test_run",
                    dataset_id="nonexistent",
                    hyperparameters={}
                )

    def test_evaluate_on_test_run_not_found(self):
        """Test evaluate_on_test raises ValueError when run not found."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        with patch('cortical.ml_experiments.file_prediction_adapter.ExperimentManager') as mock_manager:
            mock_manager.load_run.return_value = None

            with pytest.raises(ValueError, match="Run not found"):
                experiment.evaluate_on_test("nonexistent_run")

    def test_evaluate_on_test_no_model_path(self):
        """Test evaluate_on_test raises ValueError when no model path."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        experiment = FilePredictionExperiment()

        mock_run = MagicMock()
        mock_run.model_path = None
        mock_run.config = MagicMock()
        mock_run.config.dataset_id = "dataset1"

        with patch('cortical.ml_experiments.file_prediction_adapter.ExperimentManager') as mock_manager:
            mock_manager.load_run.return_value = mock_run

            with pytest.raises(ValueError, match="No model path"):
                experiment.evaluate_on_test("run1", model_path=None)


class TestRunAblationStudy:
    """Tests for run_ablation_study function."""

    def test_ablation_study_structure(self):
        """Test run_ablation_study returns proper structure."""
        from cortical.ml_experiments.file_prediction_adapter import run_ablation_study

        # Mock FilePredictionExperiment
        with patch('cortical.ml_experiments.file_prediction_adapter.FilePredictionExperiment') as MockExperiment:
            mock_experiment = MagicMock()
            mock_run = MagicMock()
            mock_run.config = MagicMock()
            mock_run.config.name = "test_baseline"
            mock_experiment.run.return_value = mock_run
            MockExperiment.return_value = mock_experiment

            result = run_ablation_study(
                base_name="test",
                dataset_id="dataset1",
                feature_variants={
                    "variant1": {"param1": True},
                    "variant2": {"param2": False}
                },
                base_hyperparameters={"base_param": 1}
            )

            # Should have baseline + 2 variants = 3 runs
            assert len(result) == 3
            assert mock_experiment.run.call_count == 3

    def test_ablation_study_no_variants(self):
        """Test run_ablation_study with no variants (baseline only)."""
        from cortical.ml_experiments.file_prediction_adapter import run_ablation_study

        with patch('cortical.ml_experiments.file_prediction_adapter.FilePredictionExperiment') as MockExperiment:
            mock_experiment = MagicMock()
            mock_run = MagicMock()
            mock_experiment.run.return_value = mock_run
            MockExperiment.return_value = mock_experiment

            result = run_ablation_study(
                base_name="test",
                dataset_id="dataset1",
                feature_variants={},
                base_hyperparameters=None
            )

            # Should have only baseline
            assert len(result) == 1

    def test_ablation_study_tags(self):
        """Test that ablation study runs have correct tags."""
        from cortical.ml_experiments.file_prediction_adapter import run_ablation_study

        with patch('cortical.ml_experiments.file_prediction_adapter.FilePredictionExperiment') as MockExperiment:
            mock_experiment = MagicMock()
            mock_run = MagicMock()
            mock_experiment.run.return_value = mock_run
            MockExperiment.return_value = mock_experiment

            run_ablation_study(
                base_name="test",
                dataset_id="dataset1",
                feature_variants={"variant1": {"param": True}}
            )

            # Check baseline call
            baseline_call = mock_experiment.run.call_args_list[0]
            assert 'ablation' in baseline_call.kwargs.get('tags', [])
            assert 'baseline' in baseline_call.kwargs.get('tags', [])

            # Check variant call
            variant_call = mock_experiment.run.call_args_list[1]
            assert 'ablation' in variant_call.kwargs.get('tags', [])
            assert 'variant1' in variant_call.kwargs.get('tags', [])


class TestFormatExperimentReport:
    """Tests for format_experiment_report function."""

    def test_format_single_run(self):
        """Test formatting report for single run."""
        from cortical.ml_experiments.file_prediction_adapter import format_experiment_report

        mock_run = MagicMock()
        mock_run.id = "run1"
        mock_run.status = "completed"
        mock_run.duration_seconds = 10.5
        mock_run.git_hash = "abc123"
        mock_run.git_status = "clean"
        mock_run.metrics_summary = {'mrr': 0.5, 'recall@10': 0.6}
        mock_run.config = MagicMock()
        mock_run.config.name = "test_run"
        mock_run.config.hyperparameters = {'param1': True}

        report = format_experiment_report([mock_run])

        assert "FILE PREDICTION EXPERIMENT REPORT" in report
        assert "test_run" in report
        assert "run1" in report
        assert "completed" in report
        assert "mrr" in report
        assert "param1" in report

    def test_format_multiple_runs_comparison(self):
        """Test formatting report for multiple runs includes comparison."""
        from cortical.ml_experiments.file_prediction_adapter import format_experiment_report

        mock_run1 = MagicMock()
        mock_run1.id = "run1"
        mock_run1.status = "completed"
        mock_run1.duration_seconds = 10.5
        mock_run1.git_hash = "abc123"
        mock_run1.git_status = "clean"
        mock_run1.metrics_summary = {'mrr': 0.5}
        mock_run1.config = MagicMock()
        mock_run1.config.name = "run1"
        mock_run1.config.hyperparameters = {}

        mock_run2 = MagicMock()
        mock_run2.id = "run2"
        mock_run2.status = "completed"
        mock_run2.duration_seconds = 15.0
        mock_run2.git_hash = "def456"
        mock_run2.git_status = "clean"
        mock_run2.metrics_summary = {'mrr': 0.7}
        mock_run2.config = MagicMock()
        mock_run2.config.name = "run2"
        mock_run2.config.hyperparameters = {}

        with patch('cortical.ml_experiments.file_prediction_adapter.MetricsManager') as mock_metrics:
            mock_metrics.format_comparison_table.return_value = "| Metric | run1 | run2 |"

            report = format_experiment_report([mock_run1, mock_run2])

            assert "COMPARISON SUMMARY" in report
            assert "Best MRR" in report

    def test_format_run_no_metrics(self):
        """Test formatting run without metrics."""
        from cortical.ml_experiments.file_prediction_adapter import format_experiment_report

        mock_run = MagicMock()
        mock_run.id = "run1"
        mock_run.status = "failed"
        mock_run.duration_seconds = 5.0
        mock_run.git_hash = None
        mock_run.git_status = "unknown"
        mock_run.metrics_summary = None
        mock_run.config = MagicMock()
        mock_run.config.name = "failed_run"
        mock_run.config.hyperparameters = {}

        report = format_experiment_report([mock_run])

        assert "failed_run" in report
        assert "failed" in report

    def test_format_empty_runs(self):
        """Test formatting with empty runs list."""
        from cortical.ml_experiments.file_prediction_adapter import format_experiment_report

        report = format_experiment_report([])

        assert "FILE PREDICTION EXPERIMENT REPORT" in report


class TestCreateCommitDataset:
    """Tests for create_commit_dataset function."""

    def test_create_dataset_calls_dataset_manager(self, tmp_path):
        """Test that create_commit_dataset calls DatasetManager correctly."""
        from cortical.ml_experiments import file_prediction_adapter

        # Create a source file
        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat", "files_changed": ["file.py"], "timestamp": "2025-01-01T00:00:00Z"},
        ]
        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        # Set up temp dataset cache
        original_cache = file_prediction_adapter.DATASET_CACHE
        file_prediction_adapter.DATASET_CACHE = tmp_path / "datasets"

        try:
            mock_dataset = MagicMock()
            mock_dataset.id = "dataset1"

            with patch.object(file_prediction_adapter.DatasetManager, 'create_dataset', return_value=mock_dataset) as mock_create:
                result = file_prediction_adapter.create_commit_dataset(
                    name="test_dataset",
                    filters={'custom_filter': True},
                    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
                    random_seed=123,
                    source_path=source_path
                )

                assert result == mock_dataset
                mock_create.assert_called_once()

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs['name'] == "test_dataset"
                assert call_kwargs['random_seed'] == 123
        finally:
            file_prediction_adapter.DATASET_CACHE = original_cache

    def test_create_dataset_default_filters(self, tmp_path):
        """Test that default filters are applied when not specified."""
        from cortical.ml_experiments import file_prediction_adapter

        # Create a source file
        source_path = tmp_path / "commits.jsonl"
        commits = [
            {"hash": "abc123", "message": "feat", "files_changed": ["file.py"], "timestamp": "2025-01-01T00:00:00Z"},
        ]
        with open(source_path, 'w') as f:
            for commit in commits:
                f.write(json.dumps(commit) + '\n')

        # Set up temp dataset cache
        original_cache = file_prediction_adapter.DATASET_CACHE
        file_prediction_adapter.DATASET_CACHE = tmp_path / "datasets"

        try:
            mock_dataset = MagicMock()

            with patch.object(file_prediction_adapter.DatasetManager, 'create_dataset', return_value=mock_dataset) as mock_create:
                file_prediction_adapter.create_commit_dataset(
                    name="test_dataset",
                    filters=None,  # Use defaults
                    source_path=source_path
                )

                call_kwargs = mock_create.call_args.kwargs
                filters = call_kwargs['filters']
                assert 'exclude_empty_files' in filters
                assert 'require_fields' in filters
        finally:
            file_prediction_adapter.DATASET_CACHE = original_cache
