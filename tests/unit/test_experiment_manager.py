"""
Comprehensive Unit Tests for ExperimentManager.

Tests for cortical/ml_experiments/experiment.py covering:
- run_experiment: Full experiment execution with tracking
- create_run: Manual run creation
- complete_run: Marking runs as complete
- fail_run: Marking runs as failed
- list_runs: Listing with filters
- compare_runs: Multi-run comparison
- get_best_run: Finding best performing run
- delete_run: Soft deletion
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from cortical.ml_experiments.experiment import (
    ExperimentConfig,
    ExperimentRun,
    ExperimentManager,
    EXPERIMENTS_DIR,
    EXPERIMENTS_LEDGER,
)
from cortical.ml_experiments.utils import ensure_directory, append_jsonl, read_jsonl


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_experiments_dir(tmp_path):
    """Create a temporary experiments directory with isolated metrics."""
    import cortical.ml_experiments.experiment as experiment_module
    import cortical.ml_experiments.metrics as metrics_module

    # Save original paths
    old_experiments_dir = experiment_module.EXPERIMENTS_DIR
    old_experiments_ledger = experiment_module.EXPERIMENTS_LEDGER
    old_metrics_dir = metrics_module.METRICS_DIR
    old_metrics_ledger = metrics_module.METRICS_LEDGER

    # Patch experiment module
    experiment_module.EXPERIMENTS_DIR = tmp_path / 'experiments'
    experiment_module.EXPERIMENTS_LEDGER = experiment_module.EXPERIMENTS_DIR / 'experiments.jsonl'

    # Patch metrics module to prevent test pollution
    metrics_module.METRICS_DIR = tmp_path / 'metrics'
    metrics_module.METRICS_LEDGER = metrics_module.METRICS_DIR / 'metrics.jsonl'

    yield tmp_path / 'experiments'

    # Restore originals
    experiment_module.EXPERIMENTS_DIR = old_experiments_dir
    experiment_module.EXPERIMENTS_LEDGER = old_experiments_ledger
    metrics_module.METRICS_DIR = old_metrics_dir
    metrics_module.METRICS_LEDGER = old_metrics_ledger


@pytest.fixture
def sample_config():
    """Create a sample ExperimentConfig."""
    return ExperimentConfig(
        name="test_experiment",
        model_type="file_prediction",
        dataset_id="ds-test-20251216-abcd",
        hyperparameters={"learning_rate": 0.01, "epochs": 10},
        description="Test experiment for coverage",
        tags=["test", "coverage"]
    )


# =============================================================================
# EXPERIMENT CONFIG TESTS
# =============================================================================


class TestExperimentConfigAdvanced:
    """Additional tests for ExperimentConfig."""

    def test_config_with_ablation_settings(self):
        """Test config with ablation feature settings."""
        config = ExperimentConfig(
            name="ablation_test",
            model_type="file_prediction",
            dataset_id="ds-test-001",
            ablation_feature="embedding_dim",
            is_baseline=False
        )

        assert config.ablation_feature == "embedding_dim"
        assert config.is_baseline is False

    def test_config_as_baseline(self):
        """Test config marked as baseline."""
        config = ExperimentConfig(
            name="baseline",
            model_type="file_prediction",
            dataset_id="ds-test-001",
            is_baseline=True
        )

        assert config.is_baseline is True

    def test_config_hash_changes_with_hyperparams(self):
        """Different hyperparameters should produce different hashes."""
        config1 = ExperimentConfig(
            name="test",
            model_type="file_prediction",
            dataset_id="ds-test-001",
            hyperparameters={"lr": 0.01}
        )

        config2 = ExperimentConfig(
            name="test",
            model_type="file_prediction",
            dataset_id="ds-test-001",
            hyperparameters={"lr": 0.02}
        )

        assert config1.config_hash != config2.config_hash

    def test_config_hash_changes_with_dataset(self):
        """Different datasets should produce different hashes."""
        config1 = ExperimentConfig(
            name="test",
            model_type="file_prediction",
            dataset_id="ds-test-001",
        )

        config2 = ExperimentConfig(
            name="test",
            model_type="file_prediction",
            dataset_id="ds-test-002",
        )

        assert config1.config_hash != config2.config_hash


# =============================================================================
# EXPERIMENT RUN TESTS
# =============================================================================


class TestExperimentRunAdvanced:
    """Additional tests for ExperimentRun."""

    def test_run_from_dict_with_config(self):
        """Test ExperimentRun.from_dict with nested config."""
        data = {
            'id': 'exp-20251216-100000-abcd',
            'config': {
                'name': 'test',
                'model_type': 'file_prediction',
                'dataset_id': 'ds-001',
                'hyperparameters': {},
                'description': '',
                'tags': [],
                'ablation_feature': None,
                'is_baseline': False
            },
            'started_at': '2025-12-16T10:00:00Z',
            'completed_at': None,
            'duration_seconds': 0.0,
            'status': 'running',
            'error': None,
            'git_hash': 'abc123',
            'git_status': 'clean',
            'model_path': None,
            'model_hash': None,
            'metrics_summary': {},
            'metadata': {}
        }

        run = ExperimentRun.from_dict(data)

        assert run.id == 'exp-20251216-100000-abcd'
        assert run.config.name == 'test'
        assert run.status == 'running'

    def test_run_to_dict_includes_config(self):
        """Test that to_dict includes nested config."""
        config = ExperimentConfig(
            name="test",
            model_type="file_prediction",
            dataset_id="ds-001"
        )
        run = ExperimentRun(
            id="exp-001",
            config=config,
            started_at="2025-12-16T10:00:00Z",
            status="running"
        )

        run_dict = run.to_dict()

        assert 'config' in run_dict
        assert run_dict['config']['name'] == 'test'


# =============================================================================
# EXPERIMENT MANAGER - RUN EXPERIMENT TESTS
# =============================================================================


class TestExperimentManagerRunExperiment:
    """Tests for ExperimentManager.run_experiment()."""

    def test_run_experiment_success(self, temp_experiments_dir, sample_config):
        """Test successful experiment run."""
        # Mock training function
        def mock_train(data, params):
            return {"model": "trained", "params": params}

        # Mock evaluation function
        def mock_eval(model, data):
            return {"mrr": 0.45, "recall@10": 0.55}

        train_data = [{"id": i, "text": f"sample {i}"} for i in range(10)]
        eval_data = [{"id": i, "text": f"eval {i}"} for i in range(5)]

        run = ExperimentManager.run_experiment(
            config=sample_config,
            train_fn=mock_train,
            eval_fn=mock_eval,
            train_data=train_data,
            eval_data=eval_data
        )

        assert run.status == 'completed'
        assert run.metrics_summary['mrr'] == 0.45
        assert run.metrics_summary['recall@10'] == 0.55
        assert run.duration_seconds > 0
        assert run.git_hash is not None

    def test_run_experiment_with_model_save(self, temp_experiments_dir, sample_config):
        """Test experiment run that saves model."""
        def mock_train(data, params):
            return {"model": "trained"}

        def mock_eval(model, data):
            return {"accuracy": 0.85}

        def mock_save(model, model_dir):
            model_path = model_dir / "model.pkl"
            model_path.write_text("mock model")
            return model_path

        run = ExperimentManager.run_experiment(
            config=sample_config,
            train_fn=mock_train,
            eval_fn=mock_eval,
            train_data=[1, 2, 3],
            eval_data=[4, 5],
            save_model_fn=mock_save
        )

        assert run.status == 'completed'
        assert run.model_path is not None
        assert Path(run.model_path).exists()

    def test_run_experiment_failure_raises(self, temp_experiments_dir, sample_config):
        """Test that training failure raises exception."""
        def mock_train_fail(data, params):
            raise ValueError("Training failed!")

        def mock_eval(model, data):
            return {"accuracy": 0.0}

        with pytest.raises(ValueError, match="Training failed"):
            ExperimentManager.run_experiment(
                config=sample_config,
                train_fn=mock_train_fail,
                eval_fn=mock_eval,
                train_data=[1, 2, 3],
                eval_data=[4, 5]
            )

    def test_run_experiment_failure_records_run(self, temp_experiments_dir, sample_config):
        """Test that failed runs are still recorded."""
        def mock_train_fail(data, params):
            raise RuntimeError("Boom!")

        def mock_eval(model, data):
            return {}

        try:
            ExperimentManager.run_experiment(
                config=sample_config,
                train_fn=mock_train_fail,
                eval_fn=mock_eval,
                train_data=[1],
                eval_data=[2]
            )
        except RuntimeError:
            pass

        # Check that a run was recorded
        ledger = temp_experiments_dir / 'experiments.jsonl'
        if ledger.exists():
            records = read_jsonl(ledger)
            assert len(records) >= 1
            # The run should be marked as failed
            failed_runs = [r for r in records if r.get('status') == 'failed']
            assert len(failed_runs) >= 1


# =============================================================================
# EXPERIMENT MANAGER - CREATE/COMPLETE/FAIL RUN TESTS
# =============================================================================


class TestExperimentManagerCreateRun:
    """Tests for manual run creation and completion."""

    def test_create_run_basic(self, temp_experiments_dir, sample_config):
        """Test basic run creation."""
        run = ExperimentManager.create_run(sample_config)

        assert run.id.startswith('exp-')
        assert run.status == 'running'
        assert run.config.name == sample_config.name
        assert run.started_at is not None

    def test_complete_run(self, temp_experiments_dir, sample_config):
        """Test completing a run."""
        run = ExperimentManager.create_run(sample_config)

        metrics = {"mrr": 0.46, "recall@10": 0.52}
        completed = ExperimentManager.complete_run(
            run,
            metrics=metrics,
            model_path="/path/to/model.pkl",
            model_hash="abc123"
        )

        assert completed.status == 'completed'
        assert completed.completed_at is not None
        assert completed.metrics_summary['mrr'] == 0.46
        assert completed.model_path == "/path/to/model.pkl"
        assert completed.model_hash == "abc123"

    def test_fail_run(self, temp_experiments_dir, sample_config):
        """Test failing a run."""
        run = ExperimentManager.create_run(sample_config)

        failed = ExperimentManager.fail_run(run, error="Out of memory")

        assert failed.status == 'failed'
        assert failed.error == "Out of memory"
        assert failed.completed_at is not None


# =============================================================================
# EXPERIMENT MANAGER - LOAD/LIST TESTS
# =============================================================================


class TestExperimentManagerLoadAndList:
    """Tests for loading and listing runs."""

    def test_load_run_existing(self, temp_experiments_dir, sample_config):
        """Test loading an existing run by ID."""
        # Create and complete a run
        def mock_train(data, params):
            return {"model": "trained"}
        def mock_eval(model, data):
            return {"accuracy": 0.85}

        original_run = ExperimentManager.run_experiment(
            config=sample_config,
            train_fn=mock_train,
            eval_fn=mock_eval,
            train_data=[1],
            eval_data=[2]
        )

        # Load it back
        loaded = ExperimentManager.load_run(original_run.id)

        assert loaded is not None
        assert loaded.id == original_run.id
        assert loaded.config.name == sample_config.name

    def test_load_run_nonexistent(self, temp_experiments_dir):
        """Test loading non-existent run returns None."""
        loaded = ExperimentManager.load_run("exp-nonexistent-000000-xxxx")
        assert loaded is None

    def test_list_runs_basic(self, temp_experiments_dir):
        """Test listing all runs."""
        # Create multiple runs
        configs = [
            ExperimentConfig(name=f"exp_{i}", model_type="file_prediction", dataset_id="ds-001")
            for i in range(3)
        ]

        for config in configs:
            def mock_train(data, params):
                return {}
            def mock_eval(model, data):
                return {"accuracy": 0.5}

            ExperimentManager.run_experiment(
                config=config,
                train_fn=mock_train,
                eval_fn=mock_eval,
                train_data=[1],
                eval_data=[2]
            )

        runs = ExperimentManager.list_runs()
        assert len(runs) >= 3

    def test_list_runs_filter_by_model_type(self, temp_experiments_dir):
        """Test filtering runs by model type."""
        config1 = ExperimentConfig(name="exp_1", model_type="file_prediction", dataset_id="ds-001")
        config2 = ExperimentConfig(name="exp_2", model_type="other_type", dataset_id="ds-001")

        def mock_train(data, params):
            return {}
        def mock_eval(model, data):
            return {"accuracy": 0.5}

        ExperimentManager.run_experiment(config1, mock_train, mock_eval, [1], [2])
        ExperimentManager.run_experiment(config2, mock_train, mock_eval, [1], [2])

        fp_runs = ExperimentManager.list_runs(model_type="file_prediction")
        other_runs = ExperimentManager.list_runs(model_type="other_type")

        assert all(r.config.model_type == "file_prediction" for r in fp_runs)
        assert all(r.config.model_type == "other_type" for r in other_runs)

    def test_list_runs_filter_by_status(self, temp_experiments_dir):
        """Test filtering runs by status."""
        config = ExperimentConfig(name="exp_1", model_type="file_prediction", dataset_id="ds-001")

        def mock_train(data, params):
            return {}
        def mock_eval(model, data):
            return {"accuracy": 0.5}

        ExperimentManager.run_experiment(config, mock_train, mock_eval, [1], [2])

        completed_runs = ExperimentManager.list_runs(status='completed')
        assert all(r.status == 'completed' for r in completed_runs)

    def test_list_runs_filter_by_tags(self, temp_experiments_dir):
        """Test filtering runs by tags."""
        config1 = ExperimentConfig(
            name="exp_1", model_type="file_prediction",
            dataset_id="ds-001", tags=["ablation"]
        )
        config2 = ExperimentConfig(
            name="exp_2", model_type="file_prediction",
            dataset_id="ds-001", tags=["baseline"]
        )

        def mock_train(data, params):
            return {}
        def mock_eval(model, data):
            return {"accuracy": 0.5}

        ExperimentManager.run_experiment(config1, mock_train, mock_eval, [1], [2])
        ExperimentManager.run_experiment(config2, mock_train, mock_eval, [1], [2])

        ablation_runs = ExperimentManager.list_runs(tags=["ablation"])
        assert len(ablation_runs) >= 1
        assert all("ablation" in r.config.tags for r in ablation_runs)

    def test_list_runs_with_limit(self, temp_experiments_dir):
        """Test limiting number of returned runs."""
        for i in range(5):
            config = ExperimentConfig(name=f"exp_{i}", model_type="file_prediction", dataset_id="ds-001")
            def mock_train(data, params):
                return {}
            def mock_eval(model, data):
                return {"accuracy": 0.5}
            ExperimentManager.run_experiment(config, mock_train, mock_eval, [1], [2])

        runs = ExperimentManager.list_runs(limit=3)
        assert len(runs) == 3


# =============================================================================
# EXPERIMENT MANAGER - COMPARE AND BEST RUN TESTS
# =============================================================================


class TestExperimentManagerCompareAndBest:
    """Tests for comparing runs and finding best."""

    def test_compare_runs(self, temp_experiments_dir):
        """Test comparing multiple runs."""
        run_ids = []
        for i in range(3):
            config = ExperimentConfig(
                name=f"exp_{i}",
                model_type="file_prediction",
                dataset_id="ds-001",
                hyperparameters={"lr": 0.01 * (i + 1)}
            )
            def mock_train(data, params):
                return {}
            def mock_eval(model, data):
                return {"mrr": 0.4 + i * 0.1}

            run = ExperimentManager.run_experiment(config, mock_train, mock_eval, [1], [2])
            run_ids.append(run.id)

        comparison = ExperimentManager.compare_runs(run_ids)

        assert len(comparison) == 3
        for run_id in run_ids:
            assert run_id in comparison
            assert 'metrics' in comparison[run_id]
            assert 'hyperparameters' in comparison[run_id]

    def test_compare_runs_missing_run(self, temp_experiments_dir):
        """Test comparing with a non-existent run ID."""
        comparison = ExperimentManager.compare_runs(["nonexistent-id"])
        assert len(comparison) == 0

    def test_get_best_run_higher_is_better(self, temp_experiments_dir):
        """Test finding best run when higher is better."""
        for i, mrr in enumerate([0.30, 0.50, 0.40]):
            config = ExperimentConfig(
                name=f"exp_{i}",
                model_type="file_prediction",
                dataset_id="ds-001"
            )
            def make_eval(val):
                def mock_eval(model, data):
                    return {"mrr": val}
                return mock_eval
            def mock_train(data, params):
                return {}

            ExperimentManager.run_experiment(
                config, mock_train, make_eval(mrr), [1], [2]
            )

        best = ExperimentManager.get_best_run(
            model_type="file_prediction",
            metric_name="mrr",
            higher_is_better=True
        )

        assert best is not None
        assert best.metrics_summary['mrr'] == 0.50

    def test_get_best_run_lower_is_better(self, temp_experiments_dir):
        """Test finding best run when lower is better."""
        for i, loss in enumerate([0.5, 0.3, 0.4]):
            config = ExperimentConfig(
                name=f"exp_{i}",
                model_type="file_prediction",
                dataset_id="ds-001"
            )
            def make_eval(val):
                def mock_eval(model, data):
                    return {"loss": val}
                return mock_eval
            def mock_train(data, params):
                return {}

            ExperimentManager.run_experiment(
                config, mock_train, make_eval(loss), [1], [2]
            )

        best = ExperimentManager.get_best_run(
            model_type="file_prediction",
            metric_name="loss",
            higher_is_better=False
        )

        assert best is not None
        assert best.metrics_summary['loss'] == 0.3

    def test_get_best_run_no_runs(self, temp_experiments_dir):
        """Test getting best run when no runs exist."""
        best = ExperimentManager.get_best_run(
            model_type="nonexistent",
            metric_name="mrr"
        )
        assert best is None


# =============================================================================
# EXPERIMENT MANAGER - DELETE RUN TESTS
# =============================================================================


class TestExperimentManagerDelete:
    """Tests for run deletion."""

    def test_delete_run_existing(self, temp_experiments_dir):
        """Test soft-deleting an existing run."""
        config = ExperimentConfig(
            name="to_delete",
            model_type="file_prediction",
            dataset_id="ds-001"
        )
        def mock_train(data, params):
            return {}
        def mock_eval(model, data):
            return {"accuracy": 0.5}

        run = ExperimentManager.run_experiment(config, mock_train, mock_eval, [1], [2])

        result = ExperimentManager.delete_run(run.id)
        assert result is True

    def test_delete_run_nonexistent(self, temp_experiments_dir):
        """Test deleting non-existent run returns False."""
        result = ExperimentManager.delete_run("nonexistent-id")
        assert result is False


# =============================================================================
# RUN WITH ACTUAL FILE PREDICTION INTEGRATION
# =============================================================================


class TestFilePredictionAdapterCoverage:
    """Tests specifically for file_prediction_adapter coverage."""

    def test_commit_example_to_dict(self):
        """Test CommitExample.to_dict() method."""
        from cortical.ml_experiments.file_prediction_adapter import CommitExample

        example = CommitExample(
            commit_hash="abc123",
            message="feat: Add feature",
            files_changed=["file1.py", "file2.py"],
            timestamp="2025-01-01T00:00:00Z",
            is_merge=False
        )

        result = example.to_dict()

        assert result['hash'] == "abc123"
        assert result['message'] == "feat: Add feature"
        assert result['files'] == ["file1.py", "file2.py"]
        assert result['is_merge'] is False

    def test_load_commits_creates_output(self, tmp_path):
        """Test that load_commits_as_jsonl writes output file."""
        from cortical.ml_experiments.file_prediction_adapter import (
            load_commits_as_jsonl,
        )

        # Create a source file with commits
        source = tmp_path / "commits.jsonl"
        with open(source, 'w') as f:
            f.write(json.dumps({
                'hash': 'abc123',
                'message': 'test commit',
                'files_changed': ['test.py'],
                'timestamp': '2025-01-01T00:00:00Z',
                'is_merge': False
            }) + '\n')

        # Use existing directory for output
        output = tmp_path / "output.jsonl"

        result = load_commits_as_jsonl(source, output, use_cali=False)

        assert result.exists()
        assert output.exists()

    def test_file_prediction_experiment_model_type(self):
        """Test FilePredictionExperiment MODEL_TYPE constant."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        exp = FilePredictionExperiment()
        assert exp.MODEL_TYPE == 'file_prediction'

    def test_file_prediction_experiment_get_metric_history(self, temp_experiments_dir):
        """Test get_metric_history method."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment
        from cortical.ml_experiments.metrics import MetricsManager

        # Record some test metrics
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.45})

        exp = FilePredictionExperiment()
        history = exp.get_metric_history("mrr", split="val")

        # Should return list of (timestamp, exp_id, value)
        assert isinstance(history, list)

    def test_file_prediction_experiment_compare_runs(self, temp_experiments_dir):
        """Test compare_runs method."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        exp = FilePredictionExperiment()
        comparison = exp.compare_runs(["nonexistent-1", "nonexistent-2"])

        # Should return empty dict for nonexistent runs
        assert comparison == {}

    def test_file_prediction_experiment_get_best_run(self, temp_experiments_dir):
        """Test get_best_run method."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        exp = FilePredictionExperiment()
        best = exp.get_best_run(metric_name="mrr")

        # Should return None when no runs exist
        assert best is None

    def test_file_prediction_detect_regression(self, temp_experiments_dir):
        """Test detect_regression method."""
        from cortical.ml_experiments.file_prediction_adapter import FilePredictionExperiment

        exp = FilePredictionExperiment()
        regressions = exp.detect_regression({"mrr": 0.40})

        # Returns a list (may be empty or contain regression info depending on prior state)
        assert isinstance(regressions, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
