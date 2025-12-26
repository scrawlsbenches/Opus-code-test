"""
Base classes for Woven Mind benchmarks.

Provides common infrastructure for benchmark definition, execution, and reporting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import statistics
import time


class BenchmarkStatus(Enum):
    """Benchmark execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class BenchmarkCategory(Enum):
    """Benchmark categories for organization."""
    STABILITY = "stability"
    QUALITY = "quality"
    SCALE = "scale"
    COGNITIVE = "cognitive"
    REGRESSION = "regression"


@dataclass
class BenchmarkMetric:
    """A single metric from a benchmark run."""
    name: str
    value: float
    unit: str = ""
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    is_passing: bool = True

    def check_thresholds(self) -> bool:
        """Check if value is within thresholds."""
        if self.threshold_min is not None and self.value < self.threshold_min:
            self.is_passing = False
            return False
        if self.threshold_max is not None and self.value > self.threshold_max:
            self.is_passing = False
            return False
        self.is_passing = True
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "is_passing": self.is_passing,
        }


@dataclass
class BenchmarkResult:
    """Complete result from a benchmark run."""
    benchmark_name: str
    category: BenchmarkCategory
    status: BenchmarkStatus
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_passing(self) -> bool:
        """Check if all metrics pass their thresholds."""
        if self.status not in (BenchmarkStatus.PASSED, BenchmarkStatus.RUNNING):
            return False
        return all(m.is_passing for m in self.metrics)

    def add_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> BenchmarkMetric:
        """Add a metric to the result."""
        metric = BenchmarkMetric(
            name=name,
            value=value,
            unit=unit,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
        metric.check_thresholds()
        self.metrics.append(metric)
        return metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "category": self.category.value,
            "status": self.status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "metadata": self.metadata,
            "is_passing": self.is_passing,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status_icon = {
            BenchmarkStatus.PASSED: "PASS",
            BenchmarkStatus.FAILED: "FAIL",
            BenchmarkStatus.ERROR: "ERR ",
            BenchmarkStatus.SKIPPED: "SKIP",
            BenchmarkStatus.PENDING: "PEND",
            BenchmarkStatus.RUNNING: "RUN ",
        }
        icon = status_icon.get(self.status, "????")
        metrics_summary = ", ".join(
            f"{m.name}={m.value:.3f}{m.unit}" for m in self.metrics[:3]
        )
        if len(self.metrics) > 3:
            metrics_summary += f" (+{len(self.metrics) - 3} more)"
        return f"[{icon}] {self.benchmark_name}: {metrics_summary} ({self.duration_ms:.0f}ms)"


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    name: str = "unnamed_benchmark"
    description: str = "No description provided"
    category: BenchmarkCategory = BenchmarkCategory.REGRESSION

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._result: Optional[BenchmarkResult] = None

    @abstractmethod
    def setup(self) -> None:
        """Prepare resources for the benchmark."""
        pass

    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results."""
        pass

    def teardown(self) -> None:
        """Clean up resources after the benchmark."""
        pass

    def execute(self) -> BenchmarkResult:
        """Full execution lifecycle: setup, run, teardown."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        start_time = time.perf_counter()

        try:
            self.setup()
            result = self.run()
            result.status = BenchmarkStatus.PASSED if result.is_passing else BenchmarkStatus.FAILED
        except Exception as e:
            result.status = BenchmarkStatus.ERROR
            result.error_message = str(e)
        finally:
            try:
                self.teardown()
            except Exception:
                pass  # Don't let teardown failures mask run failures

            result.duration_ms = (time.perf_counter() - start_time) * 1000

        self._result = result
        return result


class BenchmarkSuite:
    """Collection of benchmarks that can be run together."""

    def __init__(self, name: str = "woven_mind_benchmarks"):
        self.name = name
        self.benchmarks: List[BaseBenchmark] = []
        self.results: List[BenchmarkResult] = []

    def add(self, benchmark: BaseBenchmark) -> "BenchmarkSuite":
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
        return self

    def filter_by_category(self, category: BenchmarkCategory) -> List[BaseBenchmark]:
        """Get benchmarks in a specific category."""
        return [b for b in self.benchmarks if b.category == category]

    def run_all(
        self,
        categories: Optional[List[BenchmarkCategory]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[BenchmarkResult]:
        """Run all benchmarks (optionally filtered by category)."""
        self.results = []

        to_run = self.benchmarks
        if categories:
            to_run = [b for b in self.benchmarks if b.category in categories]

        total = len(to_run)
        for i, benchmark in enumerate(to_run):
            if progress_callback:
                progress_callback(benchmark.name, i + 1, total)

            result = benchmark.execute()
            self.results.append(result)

        return self.results

    def summary(self) -> str:
        """Generate a summary report of all results."""
        lines = [
            f"\n{'=' * 60}",
            f"BENCHMARK SUITE: {self.name}",
            f"{'=' * 60}",
        ]

        # Group by category
        by_category: Dict[BenchmarkCategory, List[BenchmarkResult]] = {}
        for result in self.results:
            by_category.setdefault(result.category, []).append(result)

        for category in BenchmarkCategory:
            if category not in by_category:
                continue

            results = by_category[category]
            passed = sum(1 for r in results if r.status == BenchmarkStatus.PASSED)
            failed = sum(1 for r in results if r.status == BenchmarkStatus.FAILED)
            errors = sum(1 for r in results if r.status == BenchmarkStatus.ERROR)

            lines.append(f"\n{category.value.upper()}: {passed} passed, {failed} failed, {errors} errors")
            lines.append("-" * 40)

            for result in results:
                lines.append(f"  {result.summary()}")

        # Overall summary
        total_passed = sum(1 for r in self.results if r.status == BenchmarkStatus.PASSED)
        total_failed = sum(1 for r in self.results if r.status == BenchmarkStatus.FAILED)
        total_errors = sum(1 for r in self.results if r.status == BenchmarkStatus.ERROR)
        total_duration = sum(r.duration_ms for r in self.results)

        lines.extend([
            f"\n{'=' * 60}",
            f"TOTAL: {total_passed} passed, {total_failed} failed, {total_errors} errors",
            f"Duration: {total_duration:.0f}ms",
            f"{'=' * 60}",
        ])

        return "\n".join(lines)

    def save_results(self, path: Path) -> None:
        """Save results to JSON file."""
        data = {
            "suite_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_results(cls, path: Path) -> "BenchmarkSuite":
        """Load results from JSON file (for comparison)."""
        data = json.loads(path.read_text())
        suite = cls(name=data["suite_name"])
        # Note: This only loads results, not runnable benchmarks
        return suite


# Utility functions for benchmark implementations

def measure_stability(
    func: Callable[[], float],
    n_runs: int = 10,
) -> Tuple[float, float, float]:
    """
    Measure stability of a function's output.

    Returns:
        Tuple of (mean, std_dev, coefficient_of_variation)
    """
    values = [func() for _ in range(n_runs)]
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = std_dev / mean if mean != 0 else float('inf')
    return mean, std_dev, cv


def measure_time(func: Callable[[], Any], n_runs: int = 5) -> Tuple[float, float]:
    """
    Measure execution time of a function.

    Returns:
        Tuple of (mean_ms, std_dev_ms)
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    mean = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std_dev


def generate_synthetic_corpus(
    n_docs: int,
    doc_length: int = 100,
    vocab_size: int = 1000,
    pattern_frequency: float = 0.3,
) -> List[str]:
    """
    Generate synthetic corpus for benchmarking.

    Args:
        n_docs: Number of documents
        doc_length: Average words per document
        vocab_size: Size of vocabulary
        pattern_frequency: How often to inject known patterns

    Returns:
        List of document strings
    """
    import random

    # Create vocabulary
    vocab = [f"word{i}" for i in range(vocab_size)]

    # Create patterns (for testing abstraction formation)
    patterns = [
        ["pattern", "alpha", "test"],
        ["pattern", "beta", "verify"],
        ["concept", "neural", "network"],
        ["concept", "deep", "learning"],
    ]

    docs = []
    for _ in range(n_docs):
        words = []
        remaining = doc_length

        # Maybe insert a pattern
        if random.random() < pattern_frequency:
            pattern = random.choice(patterns)
            words.extend(pattern)
            remaining -= len(pattern)

        # Fill with random words
        words.extend(random.choices(vocab, k=max(0, remaining)))
        random.shuffle(words)

        docs.append(" ".join(words))

    return docs
