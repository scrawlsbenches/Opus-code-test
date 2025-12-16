"""
Historical metrics tracking for ML experiments.

Provides:
- Append-only metrics ledger for trend analysis
- Metric comparison across experiments
- Time-series queries for monitoring
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import (
    now_iso,
    append_jsonl,
    read_jsonl,
    ensure_directory,
)


# Default paths
DEFAULT_ML_DIR = Path('.git-ml')
METRICS_DIR = DEFAULT_ML_DIR / 'metrics'
METRICS_LEDGER = METRICS_DIR / 'metrics.jsonl'


@dataclass
class MetricEntry:
    """A single metric measurement."""
    timestamp: str                     # ISO timestamp
    experiment_id: str                 # Reference to ExperimentRun
    split: str                         # 'train', 'val', 'test'
    metric_name: str                   # 'mrr', 'recall@10', etc.
    value: float                       # Metric value
    metadata: Dict[str, Any] = None    # Optional extra info

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result['metadata'] is None:
            result['metadata'] = {}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricEntry':
        return cls(**data)


class MetricsManager:
    """
    Track and query metrics over time.

    Example usage:
        # Record metrics from an experiment
        MetricsManager.record_metrics(
            experiment_id='exp-20251216-abc1',
            split='val',
            metrics={'mrr': 0.46, 'recall@10': 0.42, 'precision@1': 0.37}
        )

        # Get metric history
        history = MetricsManager.get_metric_history('mrr', model_type='file_prediction')

        # Compare experiments
        comparison = MetricsManager.compare_experiments(['exp-001', 'exp-002'])
    """

    @staticmethod
    def record_metrics(
        experiment_id: str,
        split: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Record metrics from an experiment evaluation.

        Args:
            experiment_id: Reference to the experiment
            split: Which data split ('train', 'val', 'test')
            metrics: Dict of metric_name -> value
            metadata: Optional extra metadata

        Returns:
            List of metric entry IDs (timestamps)
        """
        ensure_directory(METRICS_DIR)

        timestamp = now_iso()
        entry_ids = []

        for metric_name, value in metrics.items():
            entry = MetricEntry(
                timestamp=timestamp,
                experiment_id=experiment_id,
                split=split,
                metric_name=metric_name,
                value=value,
                metadata=metadata
            )
            append_jsonl(METRICS_LEDGER, entry.to_dict())
            entry_ids.append(f"{timestamp}:{metric_name}")

        return entry_ids

    @staticmethod
    def get_metric_history(
        metric_name: str,
        split: str = 'val',
        experiment_ids: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Tuple[str, str, float]]:
        """
        Get historical values for a metric.

        Args:
            metric_name: Which metric ('mrr', 'recall@10', etc.)
            split: Which split to query
            experiment_ids: Optional filter to specific experiments
            limit: Maximum entries to return

        Returns:
            List of (timestamp, experiment_id, value) tuples, sorted by time
        """
        records = read_jsonl(METRICS_LEDGER)

        values = []
        for record in records:
            if record.get('metric_name') != metric_name:
                continue
            if record.get('split') != split:
                continue
            if experiment_ids and record.get('experiment_id') not in experiment_ids:
                continue

            values.append((
                record.get('timestamp', ''),
                record.get('experiment_id', ''),
                record.get('value', 0.0)
            ))

        # Sort by timestamp
        values.sort(key=lambda x: x[0])

        return values[-limit:]

    @staticmethod
    def get_experiment_metrics(
        experiment_id: str,
        split: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get all metrics for an experiment.

        Args:
            experiment_id: Experiment to query
            split: Optional split filter

        Returns:
            Dict of metric_name -> value (latest value if duplicates)
        """
        records = read_jsonl(METRICS_LEDGER)

        metrics = {}
        for record in records:
            if record.get('experiment_id') != experiment_id:
                continue
            if split and record.get('split') != split:
                continue

            metric_name = record.get('metric_name')
            metrics[metric_name] = record.get('value', 0.0)

        return metrics

    @staticmethod
    def compare_experiments(
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
        split: str = 'val'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics across multiple experiments.

        Args:
            experiment_ids: Experiments to compare
            metrics: Optional list of metrics to include
            split: Which split to compare

        Returns:
            Dict mapping experiment_id -> {metric_name: value}
        """
        comparison = {}

        for exp_id in experiment_ids:
            exp_metrics = MetricsManager.get_experiment_metrics(exp_id, split)

            if metrics:
                exp_metrics = {k: v for k, v in exp_metrics.items() if k in metrics}

            comparison[exp_id] = exp_metrics

        return comparison

    @staticmethod
    def get_best_value(
        metric_name: str,
        split: str = 'val',
        higher_is_better: bool = True
    ) -> Optional[Tuple[str, float]]:
        """
        Find the best recorded value for a metric.

        Args:
            metric_name: Metric to search
            split: Which split
            higher_is_better: Whether higher is better

        Returns:
            Tuple of (experiment_id, value) or None
        """
        records = read_jsonl(METRICS_LEDGER)

        best_exp = None
        best_value = float('-inf') if higher_is_better else float('inf')

        for record in records:
            if record.get('metric_name') != metric_name:
                continue
            if record.get('split') != split:
                continue

            value = record.get('value', 0.0)
            is_better = (value > best_value) if higher_is_better else (value < best_value)

            if is_better:
                best_value = value
                best_exp = record.get('experiment_id')

        if best_exp is None:
            return None

        return (best_exp, best_value)

    @staticmethod
    def get_metric_stats(
        metric_name: str,
        split: str = 'val'
    ) -> Dict[str, float]:
        """
        Get statistics for a metric across all experiments.

        Args:
            metric_name: Metric to analyze
            split: Which split

        Returns:
            Dict with min, max, mean, std, count
        """
        records = read_jsonl(METRICS_LEDGER)

        values = []
        for record in records:
            if record.get('metric_name') != metric_name:
                continue
            if record.get('split') != split:
                continue
            values.append(record.get('value', 0.0))

        if not values:
            return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'std': 0}

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
        std = variance ** 0.5

        return {
            'count': n,
            'min': min(values),
            'max': max(values),
            'mean': mean,
            'std': std
        }

    @staticmethod
    def detect_regression(
        metric_name: str,
        current_value: float,
        split: str = 'val',
        threshold_pct: float = 5.0,
        higher_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if current value represents a regression from best.

        Args:
            metric_name: Metric to check
            current_value: Current metric value
            split: Which split
            threshold_pct: Percentage threshold for regression
            higher_is_better: Whether higher is better

        Returns:
            Regression info dict if regression detected, None otherwise
        """
        best = MetricsManager.get_best_value(metric_name, split, higher_is_better)

        if best is None:
            return None

        best_exp, best_value = best

        if best_value == 0:
            return None

        if higher_is_better:
            pct_change = ((current_value - best_value) / best_value) * 100
            is_regression = pct_change < -threshold_pct
        else:
            pct_change = ((best_value - current_value) / best_value) * 100
            is_regression = pct_change < -threshold_pct

        if is_regression:
            return {
                'metric': metric_name,
                'current_value': current_value,
                'best_value': best_value,
                'best_experiment': best_exp,
                'pct_change': pct_change,
                'threshold_pct': threshold_pct
            }

        return None

    @staticmethod
    def format_comparison_table(
        comparison: Dict[str, Dict[str, float]],
        metric_order: Optional[List[str]] = None
    ) -> str:
        """
        Format a comparison as an ASCII table.

        Args:
            comparison: Output from compare_experiments
            metric_order: Optional ordering for metrics

        Returns:
            Formatted string table
        """
        if not comparison:
            return "No experiments to compare"

        # Get all metrics
        all_metrics = set()
        for metrics in comparison.values():
            all_metrics.update(metrics.keys())

        if metric_order:
            all_metrics = [m for m in metric_order if m in all_metrics]
        else:
            all_metrics = sorted(all_metrics)

        # Build table
        exp_ids = list(comparison.keys())

        # Header
        lines = []
        header = "| Metric |"
        for exp_id in exp_ids:
            short_id = exp_id[-12:] if len(exp_id) > 12 else exp_id
            header += f" {short_id:>12} |"
        lines.append(header)

        # Separator
        sep = "|--------|"
        for _ in exp_ids:
            sep += "--------------|"
        lines.append(sep)

        # Rows
        for metric in all_metrics:
            row = f"| {metric[:6]:6} |"
            for exp_id in exp_ids:
                value = comparison[exp_id].get(metric, 0.0)
                row += f" {value:12.4f} |"
            lines.append(row)

        return "\n".join(lines)
