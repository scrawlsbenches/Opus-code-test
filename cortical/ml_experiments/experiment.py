"""
Experiment tracking and orchestration.

Provides:
- Experiment configuration and execution
- Run tracking with git state
- Reproducible experiment records
- Experiment comparison
"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .utils import (
    generate_experiment_id,
    now_iso,
    get_git_hash,
    get_git_status,
    compute_dict_hash,
    append_jsonl,
    read_jsonl,
    ensure_directory,
    save_json,
    load_json,
)


# Default paths
DEFAULT_ML_DIR = Path('.git-ml')
EXPERIMENTS_DIR = DEFAULT_ML_DIR / 'experiments'
EXPERIMENTS_LEDGER = EXPERIMENTS_DIR / 'experiments.jsonl'


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str                              # Human-readable name
    model_type: str                        # 'file_prediction', etc.
    dataset_id: str                        # Reference to DatasetVersion
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""                  # Optional description
    tags: List[str] = field(default_factory=list)

    # Ablation settings
    ablation_feature: Optional[str] = None  # Feature to ablate (remove)
    is_baseline: bool = False               # Is this a baseline run?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(**data)

    @property
    def config_hash(self) -> str:
        """Hash of configuration for comparison."""
        return compute_dict_hash({
            'model_type': self.model_type,
            'dataset_id': self.dataset_id,
            'hyperparameters': self.hyperparameters,
            'ablation_feature': self.ablation_feature
        })


@dataclass
class ExperimentRun:
    """
    Record of a completed experiment run.

    Immutable after creation - provides full audit trail.
    """
    id: str                                # Unique identifier
    config: ExperimentConfig               # Full configuration

    # Timing
    started_at: str                        # ISO timestamp
    completed_at: Optional[str] = None     # ISO timestamp
    duration_seconds: float = 0.0

    # Status
    status: str = 'running'                # 'running', 'completed', 'failed'
    error: Optional[str] = None            # Error message if failed

    # Git state (for reproducibility)
    git_hash: Optional[str] = None
    git_status: str = 'unknown'            # 'clean' or 'dirty'

    # Artifacts
    model_path: Optional[str] = None       # Path to saved model
    model_hash: Optional[str] = None       # Hash of model file

    # Metrics (summary - full metrics in metrics.jsonl)
    metrics_summary: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if hasattr(self.config, 'to_dict'):
            result['config'] = self.config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        config_data = data.pop('config', {})
        config = ExperimentConfig.from_dict(config_data) if config_data else None
        return cls(config=config, **data)


class ExperimentManager:
    """
    Manage ML experiments with full tracking.

    Example usage:
        # Define experiment
        config = ExperimentConfig(
            name='baseline_v1',
            model_type='file_prediction',
            dataset_id='ds-commits-20251216-abc1',
            hyperparameters={'freq_penalty': 0.3}
        )

        # Run with tracking
        run = ExperimentManager.run_experiment(
            config=config,
            train_fn=my_train_function,
            eval_fn=my_eval_function
        )

        # Compare experiments
        comparison = ExperimentManager.compare_runs(['exp-001', 'exp-002'])
    """

    @staticmethod
    def run_experiment(
        config: ExperimentConfig,
        train_fn: Callable[[List[Any], Dict[str, Any]], Any],
        eval_fn: Callable[[Any, List[Any]], Dict[str, float]],
        train_data: List[Any],
        eval_data: List[Any],
        save_model_fn: Optional[Callable[[Any, Path], str]] = None
    ) -> ExperimentRun:
        """
        Execute an experiment with full tracking.

        Args:
            config: Experiment configuration
            train_fn: Function(train_data, hyperparams) -> model
            eval_fn: Function(model, eval_data) -> metrics dict
            train_data: Training examples
            eval_data: Evaluation examples
            save_model_fn: Optional function to save model

        Returns:
            ExperimentRun with all tracking data
        """
        # Initialize run
        run = ExperimentRun(
            id=generate_experiment_id(),
            config=config,
            started_at=now_iso(),
            git_hash=get_git_hash(),
            git_status=get_git_status()
        )

        start_time = time.perf_counter()

        try:
            # Train model
            model = train_fn(train_data, config.hyperparameters)

            # Evaluate
            metrics = eval_fn(model, eval_data)
            run.metrics_summary = metrics

            # Save model if function provided
            if save_model_fn:
                model_dir = EXPERIMENTS_DIR / 'models' / run.id
                ensure_directory(model_dir)
                model_path = save_model_fn(model, model_dir)
                run.model_path = str(model_path)

            run.status = 'completed'

        except Exception as e:
            run.status = 'failed'
            run.error = str(e)
            raise

        finally:
            run.completed_at = now_iso()
            run.duration_seconds = time.perf_counter() - start_time

            # Save run record
            ExperimentManager._save_run(run)

        return run

    @staticmethod
    def create_run(config: ExperimentConfig) -> ExperimentRun:
        """
        Create a new experiment run without executing.

        Useful for manual tracking or custom training loops.

        Args:
            config: Experiment configuration

        Returns:
            New ExperimentRun in 'running' state
        """
        return ExperimentRun(
            id=generate_experiment_id(),
            config=config,
            started_at=now_iso(),
            git_hash=get_git_hash(),
            git_status=get_git_status(),
            status='running'
        )

    @staticmethod
    def complete_run(
        run: ExperimentRun,
        metrics: Dict[str, float],
        model_path: Optional[str] = None,
        model_hash: Optional[str] = None
    ) -> ExperimentRun:
        """
        Mark a run as completed and save.

        Args:
            run: The run to complete
            metrics: Evaluation metrics
            model_path: Optional path to saved model
            model_hash: Optional hash of model file

        Returns:
            Updated ExperimentRun
        """
        run.completed_at = now_iso()
        run.status = 'completed'
        run.metrics_summary = metrics
        run.model_path = model_path
        run.model_hash = model_hash

        # Calculate duration if started_at is set
        # (simplified - just mark completion time)

        ExperimentManager._save_run(run)
        return run

    @staticmethod
    def fail_run(run: ExperimentRun, error: str) -> ExperimentRun:
        """
        Mark a run as failed.

        Args:
            run: The run that failed
            error: Error message

        Returns:
            Updated ExperimentRun
        """
        run.completed_at = now_iso()
        run.status = 'failed'
        run.error = error

        ExperimentManager._save_run(run)
        return run

    @staticmethod
    def _save_run(run: ExperimentRun) -> None:
        """Append run to experiments ledger."""
        ensure_directory(EXPERIMENTS_DIR)
        append_jsonl(EXPERIMENTS_LEDGER, run.to_dict())

    @staticmethod
    def load_run(run_id: str) -> Optional[ExperimentRun]:
        """
        Load an experiment run by ID.

        Args:
            run_id: Experiment identifier

        Returns:
            ExperimentRun or None if not found
        """
        records = read_jsonl(EXPERIMENTS_LEDGER)

        # Search newest first (in case of updates)
        for record in reversed(records):
            if record.get('id') == run_id:
                return ExperimentRun.from_dict(record)

        return None

    @staticmethod
    def list_runs(
        model_type: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[ExperimentRun]:
        """
        List experiment runs with optional filtering.

        Args:
            model_type: Filter by model type
            dataset_id: Filter by dataset
            status: Filter by status ('completed', 'failed')
            tags: Filter by tags (any match)
            limit: Maximum results

        Returns:
            List of ExperimentRuns, newest first
        """
        records = read_jsonl(EXPERIMENTS_LEDGER)

        # Apply filters
        filtered = []
        for record in records:
            config = record.get('config', {})

            if model_type and config.get('model_type') != model_type:
                continue
            if dataset_id and config.get('dataset_id') != dataset_id:
                continue
            if status and record.get('status') != status:
                continue
            if tags:
                record_tags = config.get('tags', [])
                if not any(t in record_tags for t in tags):
                    continue

            filtered.append(record)

        # Sort by started_at descending
        filtered.sort(key=lambda r: r.get('started_at', ''), reverse=True)

        return [ExperimentRun.from_dict(r) for r in filtered[:limit]]

    @staticmethod
    def compare_runs(run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare metrics across multiple runs.

        Args:
            run_ids: List of experiment IDs to compare

        Returns:
            Dict mapping run_id to metrics and config summary
        """
        comparison = {}

        for run_id in run_ids:
            run = ExperimentManager.load_run(run_id)
            if run:
                comparison[run_id] = {
                    'name': run.config.name if run.config else 'unknown',
                    'status': run.status,
                    'metrics': run.metrics_summary,
                    'duration': run.duration_seconds,
                    'hyperparameters': run.config.hyperparameters if run.config else {},
                    'dataset_id': run.config.dataset_id if run.config else None
                }

        return comparison

    @staticmethod
    def get_best_run(
        model_type: str,
        metric_name: str,
        higher_is_better: bool = True
    ) -> Optional[ExperimentRun]:
        """
        Find the best performing run for a model type.

        Args:
            model_type: Model type to search
            metric_name: Metric to optimize
            higher_is_better: Whether higher metric values are better

        Returns:
            Best ExperimentRun or None
        """
        runs = ExperimentManager.list_runs(
            model_type=model_type,
            status='completed',
            limit=1000
        )

        if not runs:
            return None

        def get_metric(run: ExperimentRun) -> float:
            return run.metrics_summary.get(metric_name, float('-inf') if higher_is_better else float('inf'))

        return max(runs, key=get_metric) if higher_is_better else min(runs, key=get_metric)

    @staticmethod
    def delete_run(run_id: str) -> bool:
        """
        Mark a run as deleted (soft delete).

        Note: We don't actually remove from JSONL to maintain audit trail.
        Instead, we append a deletion marker.

        Args:
            run_id: Experiment ID to delete

        Returns:
            True if found and marked, False otherwise
        """
        run = ExperimentManager.load_run(run_id)
        if not run:
            return False

        # Append deletion marker
        append_jsonl(EXPERIMENTS_LEDGER, {
            'id': run_id,
            'deleted_at': now_iso(),
            'status': 'deleted'
        })

        return True
