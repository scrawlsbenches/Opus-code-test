"""
Experiment infrastructure for training and evaluating graph neural networks.

This package provides:
- ExperimentKernel: Training harness for TrainableGraphProtocol implementations
- Profiler: Timing and memory tracking for training diagnostics
- Tokenizer: Simple text tokenization for language modeling tasks

Usage:
    from cortical.experiments import ExperimentKernel, Profiler
    from cortical.experiments.tokenizer import tokenize, build_vocab

See experiments/EXPERIMENT_KERNEL_PLAN.md for design documentation.
"""

from .profiler import Profiler, StepMetrics, ProfilingReport
from .kernel import ExperimentKernel, clip_gradients
from .tokenizer import tokenize, build_vocab, tokens_to_ids, ids_to_tokens

__all__ = [
    # Core
    "ExperimentKernel",
    "clip_gradients",
    # Profiling
    "Profiler",
    "StepMetrics",
    "ProfilingReport",
    # Tokenization
    "tokenize",
    "build_vocab",
    "tokens_to_ids",
    "ids_to_tokens",
]
