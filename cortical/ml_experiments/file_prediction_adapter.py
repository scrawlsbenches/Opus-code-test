"""
Integration adapter for ml_file_prediction with ML experiments framework.

Provides:
- Reproducible dataset creation from commit history
- Experiment tracking for model training runs
- Historical metrics tracking for trend analysis
- Ablation study support

Example usage:
    from cortical.ml_experiments.file_prediction_adapter import (
        FilePredictionExperiment,
        create_commit_dataset,
        run_ablation_study
    )

    # Create versioned dataset
    dataset = create_commit_dataset(
        name='commits_v1',
        filters={'exclude_merge': True},
        split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}
    )

    # Run experiment with tracking
    experiment = FilePredictionExperiment()
    run = experiment.run(
        name='baseline_v1',
        dataset_id=dataset.id,
        hyperparameters={'use_ai_meta': True}
    )

    # View metrics history
    history = experiment.get_metric_history('mrr')
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add scripts to path for imports
scripts_dir = Path(__file__).parent.parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from .dataset import DatasetManager, DatasetVersion
from .experiment import ExperimentManager, ExperimentConfig, ExperimentRun
from .metrics import MetricsManager
from .utils import (
    now_iso,
    compute_file_hash,
    read_jsonl,
    append_jsonl,
    ensure_directory,
    save_json,
)

# CALI support (high-performance ML storage)
try:
    from cortical.ml_storage import MLStore
    CALI_AVAILABLE = True
except ImportError:
    CALI_AVAILABLE = False


# Default paths
COMMITS_SOURCE = Path('.git-ml/tracked/commits.jsonl')
CALI_DIR = Path('.git-ml/cali')
DATASET_CACHE = Path('.git-ml/datasets')


@dataclass
class CommitExample:
    """A training example from commit history."""
    commit_hash: str
    message: str
    files_changed: List[str]
    timestamp: str
    is_merge: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash': self.commit_hash,
            'message': self.message,
            'files': self.files_changed,
            'timestamp': self.timestamp,
            'is_merge': self.is_merge
        }


def _load_commits_from_cali() -> List[Dict[str, Any]]:
    """Load commits from CALI store (O(n) sequential iteration)."""
    if not CALI_AVAILABLE or not CALI_DIR.exists():
        return []

    try:
        store = MLStore(CALI_DIR, rebuild_indices=False)  # Don't need indices for iteration
        commits = list(store.iterate('commit'))
        store.close()
        return commits
    except Exception:
        return []


def load_commits_as_jsonl(
    source_path: Path = COMMITS_SOURCE,
    output_path: Optional[Path] = None,
    use_cali: bool = True
) -> Path:
    """
    Load commits and write as standardized JSONL for dataset creation.

    The ml_file_prediction.py has its own commit loading logic. This function
    creates a normalized JSONL file that can be used with DatasetManager.

    Args:
        source_path: Path to raw commits.jsonl
        output_path: Where to write normalized output (default: auto-generated)
        use_cali: If True, try CALI first, then fall back to JSONL.

    Returns:
        Path to normalized JSONL file
    """
    if output_path is None:
        ensure_directory(DATASET_CACHE)
        output_path = DATASET_CACHE / 'commits_normalized.jsonl'

    # Try CALI first (faster iteration, no index needed)
    records = []
    if use_cali:
        records = _load_commits_from_cali()

    # Fall back to JSONL
    if not records:
        if not source_path.exists():
            raise FileNotFoundError(f"Commits file not found: {source_path}")
        records = read_jsonl(source_path)

    # Write normalized format
    with open(output_path, 'w') as f:
        for record in records:
            # Skip merge commits and ML data commits
            if record.get('is_merge', False):
                continue
            if record.get('message', '').startswith('data: ML'):
                continue

            normalized = {
                'hash': record.get('hash', ''),
                'message': record.get('message', ''),
                'files': record.get('files_changed', []),
                'timestamp': record.get('timestamp', ''),
                'is_merge': record.get('is_merge', False),
            }

            # Only include commits with files
            if normalized['files']:
                f.write(json.dumps(normalized) + '\n')

    return output_path


def create_commit_dataset(
    name: str,
    filters: Optional[Dict[str, Any]] = None,
    split_ratios: Optional[Dict[str, float]] = None,
    random_seed: int = 42,
    source_path: Path = COMMITS_SOURCE
) -> DatasetVersion:
    """
    Create a versioned dataset from commit history.

    This wraps DatasetManager to work with commit data specifically.

    Args:
        name: Dataset name (e.g., 'commits_v1')
        filters: Filter configuration for commits
        split_ratios: Train/val/test split ratios
        random_seed: Seed for reproducible splits
        source_path: Path to commits JSONL

    Returns:
        DatasetVersion with train/val/test splits
    """
    # Normalize commits to standard format
    normalized_path = load_commits_as_jsonl(source_path)

    # Default filters for commit data
    if filters is None:
        filters = {
            'exclude_empty_files': True,
            'require_fields': ['hash', 'message', 'files'],
        }

    # Create dataset with reproducible splits
    return DatasetManager.create_dataset(
        name=name,
        source_path=str(normalized_path),
        filters=filters,
        split_ratios=split_ratios,
        split_strategy='random',
        random_seed=random_seed,
        metadata={'source': 'commits', 'original_path': str(source_path)}
    )


class FilePredictionExperiment:
    """
    Experiment runner for file prediction model.

    Integrates with:
    - DatasetManager for versioned datasets
    - ExperimentManager for run tracking
    - MetricsManager for historical metrics
    """

    MODEL_TYPE = 'file_prediction'

    def __init__(self):
        """Initialize experiment runner."""
        self._ml_prediction_module = None

    def _get_ml_module(self):
        """Lazy load ml_file_prediction module."""
        if self._ml_prediction_module is None:
            try:
                from ml_file_prediction import (
                    train_model,
                    evaluate_model,
                    save_model,
                    load_model,
                    FilePredictionModel,
                    TrainingExample,
                )
                self._ml_prediction_module = {
                    'train_model': train_model,
                    'evaluate_model': evaluate_model,
                    'save_model': save_model,
                    'load_model': load_model,
                    'FilePredictionModel': FilePredictionModel,
                    'TrainingExample': TrainingExample,
                }
            except ImportError as e:
                raise ImportError(
                    f"Cannot import ml_file_prediction: {e}\n"
                    "Make sure scripts/ is in your Python path."
                )
        return self._ml_prediction_module

    def _records_to_examples(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Any]:
        """Convert JSONL records to TrainingExample objects."""
        ml = self._get_ml_module()
        TrainingExample = ml['TrainingExample']

        examples = []
        for record in records:
            # Import extract functions from ml_file_prediction
            try:
                from ml_file_prediction import extract_commit_type, extract_keywords
            except ImportError:
                # Fallback - simple extraction
                def extract_commit_type(msg):
                    return None
                def extract_keywords(msg):
                    return []

            example = TrainingExample(
                commit_hash=record.get('hash', ''),
                message=record.get('message', ''),
                files_changed=record.get('files', []),
                commit_type=extract_commit_type(record.get('message', '')),
                keywords=extract_keywords(record.get('message', '')),
                timestamp=record.get('timestamp', ''),
                insertions=record.get('insertions', 0),
                deletions=record.get('deletions', 0),
            )
            if example.files_changed:
                examples.append(example)

        return examples

    def run(
        self,
        name: str,
        dataset_id: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        save_model_path: Optional[Path] = None
    ) -> ExperimentRun:
        """
        Run a file prediction experiment with full tracking.

        Args:
            name: Experiment name (e.g., 'baseline_v1')
            dataset_id: ID of versioned dataset
            hyperparameters: Model hyperparameters
            description: Experiment description
            tags: Tags for filtering
            save_model_path: Where to save the trained model

        Returns:
            ExperimentRun with metrics and tracking
        """
        ml = self._get_ml_module()

        # Load dataset
        dataset = DatasetManager.load_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Get train and validation data
        train_records = DatasetManager.get_split_data(dataset, 'train')
        val_records = DatasetManager.get_split_data(dataset, 'val')

        train_examples = self._records_to_examples(train_records)
        val_examples = self._records_to_examples(val_records)

        # Create experiment config
        config = ExperimentConfig(
            name=name,
            model_type=self.MODEL_TYPE,
            dataset_id=dataset_id,
            hyperparameters=hyperparameters or {},
            description=description,
            tags=tags or [],
        )

        # Define training function
        def train_fn(data, params):
            return ml['train_model'](data)

        # Define evaluation function
        def eval_fn(model, data):
            results = ml['evaluate_model'](model, data)
            return {
                'mrr': results.get('mrr', 0.0),
                'recall@1': results.get('recall@1', 0.0),
                'recall@5': results.get('recall@5', 0.0),
                'recall@10': results.get('recall@10', 0.0),
                'precision@1': results.get('precision@1', 0.0),
            }

        # Define model save function
        def save_model_fn(model, model_dir):
            if save_model_path:
                path = save_model_path
            else:
                path = model_dir / 'model.json'
            ml['save_model'](model, path)
            return path

        # Run experiment with tracking
        run = ExperimentManager.run_experiment(
            config=config,
            train_fn=train_fn,
            eval_fn=eval_fn,
            train_data=train_examples,
            eval_data=val_examples,
            save_model_fn=save_model_fn,
        )

        # Record metrics to historical ledger
        MetricsManager.record_metrics(
            experiment_id=run.id,
            split='val',
            metrics=run.metrics_summary,
            metadata={
                'dataset_id': dataset_id,
                'config_hash': config.config_hash,
            }
        )

        return run

    def evaluate_on_test(
        self,
        run_id: str,
        model_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on held-out test set.

        WARNING: Only use this for final evaluation, not hyperparameter tuning!

        Args:
            run_id: Experiment run ID
            model_path: Path to model (uses run's model_path if not specified)

        Returns:
            Test set metrics
        """
        ml = self._get_ml_module()

        # Load run
        run = ExperimentManager.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        # Load model
        if model_path is None:
            model_path = Path(run.model_path) if run.model_path else None

        if model_path is None:
            raise ValueError("No model path available for this run")

        model = ml['load_model'](model_path)
        if model is None:
            raise ValueError(f"Could not load model from {model_path}")

        # Load dataset and get test split
        dataset = DatasetManager.load_dataset(run.config.dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {run.config.dataset_id}")

        test_records = DatasetManager.get_split_data(dataset, 'test')
        test_examples = self._records_to_examples(test_records)

        # Evaluate
        results = ml['evaluate_model'](model, test_examples)

        # Record test metrics
        metrics = {
            'mrr': results.get('mrr', 0.0),
            'recall@1': results.get('recall@1', 0.0),
            'recall@5': results.get('recall@5', 0.0),
            'recall@10': results.get('recall@10', 0.0),
            'precision@1': results.get('precision@1', 0.0),
        }

        MetricsManager.record_metrics(
            experiment_id=run_id,
            split='test',
            metrics=metrics,
            metadata={'warning': 'Final evaluation - do not use for tuning'}
        )

        return metrics

    def get_metric_history(
        self,
        metric_name: str,
        split: str = 'val'
    ) -> List[Tuple[str, str, float]]:
        """
        Get historical values for a metric.

        Args:
            metric_name: Which metric ('mrr', 'recall@10', etc.)
            split: Which split to query

        Returns:
            List of (timestamp, experiment_id, value) tuples
        """
        return MetricsManager.get_metric_history(metric_name, split)

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare metrics across multiple experiment runs.

        Args:
            run_ids: List of experiment IDs to compare

        Returns:
            Comparison data
        """
        return ExperimentManager.compare_runs(run_ids)

    def get_best_run(
        self,
        metric_name: str = 'mrr',
        higher_is_better: bool = True
    ) -> Optional[ExperimentRun]:
        """
        Find the best performing run.

        Args:
            metric_name: Metric to optimize
            higher_is_better: Whether higher values are better

        Returns:
            Best ExperimentRun or None
        """
        return ExperimentManager.get_best_run(
            model_type=self.MODEL_TYPE,
            metric_name=metric_name,
            higher_is_better=higher_is_better
        )

    def detect_regression(
        self,
        current_metrics: Dict[str, float],
        threshold_pct: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Detect if current metrics represent a regression.

        Args:
            current_metrics: Dict of metric_name -> value
            threshold_pct: Percentage threshold for regression

        Returns:
            List of regression warnings (empty if none detected)
        """
        regressions = []

        for metric_name, value in current_metrics.items():
            regression = MetricsManager.detect_regression(
                metric_name=metric_name,
                current_value=value,
                split='val',
                threshold_pct=threshold_pct,
                higher_is_better=True
            )
            if regression:
                regressions.append(regression)

        return regressions


def run_ablation_study(
    base_name: str,
    dataset_id: str,
    feature_variants: Dict[str, Dict[str, Any]],
    base_hyperparameters: Optional[Dict[str, Any]] = None
) -> List[ExperimentRun]:
    """
    Run ablation study across feature variants.

    Args:
        base_name: Base experiment name
        dataset_id: Dataset to use
        feature_variants: Dict of variant_name -> hyperparameter overrides
        base_hyperparameters: Base hyperparameters to use

    Returns:
        List of ExperimentRuns for each variant
    """
    experiment = FilePredictionExperiment()
    runs = []

    # Run baseline
    base_config = base_hyperparameters or {}
    baseline_run = experiment.run(
        name=f"{base_name}_baseline",
        dataset_id=dataset_id,
        hyperparameters=base_config,
        tags=['ablation', 'baseline']
    )
    runs.append(baseline_run)

    # Run each variant
    for variant_name, variant_config in feature_variants.items():
        merged_config = {**base_config, **variant_config}

        variant_run = experiment.run(
            name=f"{base_name}_{variant_name}",
            dataset_id=dataset_id,
            hyperparameters=merged_config,
            tags=['ablation', variant_name]
        )
        runs.append(variant_run)

    return runs


def format_experiment_report(runs: List[ExperimentRun]) -> str:
    """
    Format experiment results as a readable report.

    Args:
        runs: List of experiment runs

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "FILE PREDICTION EXPERIMENT REPORT",
        "=" * 70,
        ""
    ]

    for run in runs:
        lines.append(f"Experiment: {run.config.name}")
        lines.append(f"  ID: {run.id}")
        lines.append(f"  Status: {run.status}")
        lines.append(f"  Duration: {run.duration_seconds:.2f}s")
        lines.append(f"  Git: {run.git_hash or 'unknown'} ({run.git_status})")

        if run.metrics_summary:
            lines.append("  Metrics:")
            for metric, value in sorted(run.metrics_summary.items()):
                lines.append(f"    {metric}: {value:.4f}")

        if run.config.hyperparameters:
            lines.append("  Hyperparameters:")
            for param, value in run.config.hyperparameters.items():
                lines.append(f"    {param}: {value}")

        lines.append("")

    # Summary comparison
    if len(runs) > 1:
        lines.append("-" * 70)
        lines.append("COMPARISON SUMMARY")
        lines.append("-" * 70)

        # Find best by MRR
        best_mrr_run = max(
            runs,
            key=lambda r: r.metrics_summary.get('mrr', 0) if r.metrics_summary else 0
        )
        lines.append(f"Best MRR: {best_mrr_run.config.name} "
                    f"({best_mrr_run.metrics_summary.get('mrr', 0):.4f})")

        # Comparison table
        lines.append("")
        lines.append(MetricsManager.format_comparison_table(
            {run.id: run.metrics_summary or {} for run in runs}
        ))

    return "\n".join(lines)
