"""
Woven Mind Benchmark Suite

Comprehensive benchmarks for validating the dual-process cognitive architecture.

Benchmark Categories:
- Stability: Parameter sensitivity, baseline drift
- Quality: Abstraction precision/recall, mode switching accuracy
- Scale: Performance vs corpus size
- Cognitive: Dual-process behavior validation

Usage:
    python -m benchmarks.woven_mind.runner --all
    python -m benchmarks.woven_mind.runner --category stability
    python -m benchmarks.woven_mind.runner --benchmark parameter_sensitivity
"""

from .base import BenchmarkResult, BenchmarkSuite
from .stability import ParameterSensitivityBenchmark, BaselineDriftBenchmark
from .quality import AbstractionQualityBenchmark, ModeSwitchingBenchmark
from .scale import ScalabilityBenchmark
from .cognitive import SurpriseCalibrationBenchmark, HomeostasisInteractionBenchmark

__all__ = [
    'BenchmarkResult',
    'BenchmarkSuite',
    'ParameterSensitivityBenchmark',
    'BaselineDriftBenchmark',
    'AbstractionQualityBenchmark',
    'ModeSwitchingBenchmark',
    'ScalabilityBenchmark',
    'SurpriseCalibrationBenchmark',
    'HomeostasisInteractionBenchmark',
]
