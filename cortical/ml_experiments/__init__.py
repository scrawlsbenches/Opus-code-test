"""
ML Experiment Tracking Framework
================================

A lightweight, zero-dependency framework for reproducible ML experiments.
"""

from .dataset import (
    DatasetManager,
    DatasetVersion,
    SplitInfo,
    create_holdout_split,
)

from .experiment import (
    ExperimentManager,
    ExperimentConfig,
    ExperimentRun,
)

from .metrics import (
    MetricsManager,
    MetricEntry,
)

from .utils import (
    compute_file_hash,
    compute_content_hash,
    compute_dict_hash,
    generate_experiment_id,
    generate_dataset_id,
    now_iso,
    get_git_hash,
    get_git_status,
    set_random_seed,
    reproducible_shuffle,
    split_indices,
)

# File prediction integration (lazy import to avoid circular deps)
def get_file_prediction_experiment():
    """Get FilePredictionExperiment class (lazy import)."""
    from .file_prediction_adapter import FilePredictionExperiment
    return FilePredictionExperiment

def create_commit_dataset(*args, **kwargs):
    """Create versioned dataset from commit history."""
    from .file_prediction_adapter import create_commit_dataset as _create
    return _create(*args, **kwargs)

def run_ablation_study(*args, **kwargs):
    """Run ablation study across feature variants."""
    from .file_prediction_adapter import run_ablation_study as _run
    return _run(*args, **kwargs)

__all__ = [
    # Dataset management
    'DatasetManager',
    'DatasetVersion',
    'SplitInfo',
    'create_holdout_split',
    # Experiment tracking
    'ExperimentManager',
    'ExperimentConfig',
    'ExperimentRun',
    # Metrics
    'MetricsManager',
    'MetricEntry',
    # Utilities
    'compute_file_hash',
    'compute_content_hash',
    'compute_dict_hash',
    'generate_experiment_id',
    'generate_dataset_id',
    'now_iso',
    'get_git_hash',
    'get_git_status',
    'set_random_seed',
    'reproducible_shuffle',
    'split_indices',
    # File prediction integration
    'get_file_prediction_experiment',
    'create_commit_dataset',
    'run_ablation_study',
]

__version__ = '1.0.0'
