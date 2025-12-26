"""
PRISM-SLM Benchmarks

Benchmarks for the Statistical Language Model with Synaptic Learning.

Categories:
- generation: Text generation quality and coherence
- learning: Hebbian learning dynamics and stability
- integration: Integration with Woven Mind

Usage:
    python -m benchmarks.prism_slm.runner --all
    python -m benchmarks.prism_slm.runner --category generation
    python -m benchmarks.prism_slm.runner --list
"""

from .generation import (
    GenerationCoherenceBenchmark,
    PerplexityCalibrationBenchmark,
    TemperatureDiversityBenchmark,
)
from .learning import (
    HebbianStrengtheningBenchmark,
    DecayStabilityBenchmark,
    RewardLearningBenchmark,
)
from .integration import (
    SpreadingActivationBenchmark,
    LateralInhibitionBenchmark,
    SparsecodingEfficiencyBenchmark,
)

__all__ = [
    # Generation
    "GenerationCoherenceBenchmark",
    "PerplexityCalibrationBenchmark",
    "TemperatureDiversityBenchmark",
    # Learning
    "HebbianStrengtheningBenchmark",
    "DecayStabilityBenchmark",
    "RewardLearningBenchmark",
    # Integration
    "SpreadingActivationBenchmark",
    "LateralInhibitionBenchmark",
    "SparsecodingEfficiencyBenchmark",
]
