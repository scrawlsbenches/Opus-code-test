"""
Experiment infrastructure for training and evaluating graph neural networks.

This package provides:
- ExperimentKernel: Training harness for TrainableGraphProtocol implementations
- Profiler: Timing and memory tracking for training diagnostics
- Tokenizer: Simple text tokenization for language modeling tasks
- ExperimentConfig: Configuration management for experiments
- ExperimentLog: JSON-based experiment logging and comparison

Usage:
    from cortical.experiments import ExperimentKernel, Profiler, ExperimentConfig
    from cortical.experiments.tokenizer import tokenize, build_vocab

CLI usage:
    python -m cortical.experiments.cli run --input samples/text.txt --name my-experiment
    python -m cortical.experiments.cli compare exp1 exp2
    python -m cortical.experiments.cli list

See experiments/EXPERIMENT_KERNEL_PLAN.md and experiments/EXPERIMENT_CLI_PLAN.md
for design documentation.
"""

from .profiler import Profiler, StepMetrics, ProfilingReport
from .kernel import ExperimentKernel, clip_gradients
from .tokenizer import tokenize, build_vocab, tokens_to_ids, ids_to_tokens
from .config import ExperimentConfig
from .logging import ExperimentLog, ExperimentMetrics, list_experiments
from .position import LearnedPositionEncoding, create_position_encoding

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
    # Config and Logging
    "ExperimentConfig",
    "ExperimentLog",
    "ExperimentMetrics",
    "list_experiments",
    # Position Encoding
    "LearnedPositionEncoding",
    "create_position_encoding",
]
