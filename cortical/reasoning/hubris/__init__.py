"""
Hubris MoE: Mixture of Experts with Calibration and Commitment.

The Hubris MoE system provides a calibrated ensemble of micro-experts
that collaborate on complex tasks. The name "Hubris" reminds us that
overconfidence is the enemy of reliable systems—well-calibrated
experts know what they don't know.

Core Components:
    - MicroExpert: Specialized expert with narrow competence
    - MetaExpert: Coordinator that orchestrates sub-experts
    - CreditLedger: Performance tracking with calibration metrics
    - ValueSignal: Reward-based learning for expert behavior
    - StakingManager: Commitment mechanism for predictions
    - HubrisMoE: Main orchestrator

Key Metrics:
    - ECE (Expected Calibration Error): Measures confidence-accuracy gap
    - Credit Score: Accumulated reputation from prediction history
    - Value Signal: Learned utility of actions/approaches
    - Staking Balance: Resources available for commitment

Example:
    >>> from cortical.reasoning.hubris import HubrisMoE, MicroExpert
    >>>
    >>> # Create experts
    >>> parser = MicroExpert("parser", "nlp", ["parsing", "syntax"])
    >>> semantic = MicroExpert("semantic", "nlp", ["meaning", "intent"])
    >>>
    >>> # Create MoE system
    >>> moe = HubrisMoE()
    >>> moe.register_expert(parser)
    >>> moe.register_expert(semantic)
    >>>
    >>> # Query
    >>> result = moe.query("What is the structure of 'The cat sat on the mat'?")
    >>> print(f"Answer: {result.answer}")
    >>> print(f"Confidence: {result.confidence:.2f}")
    >>> print(f"Contributing experts: {result.contributing_experts}")

Philosophy:
    This system embodies the Metus principle of reverence—in this case,
    reverence for the truth. Overconfident systems are dangerous because
    they mislead users into trusting unreliable outputs. By tracking
    calibration (ECE), requiring commitment (staking), and learning from
    outcomes (value signals), we build systems that honestly represent
    their uncertainty.
"""

from .expert import (
    MicroExpert,
    MetaExpert,
    ExpertResponse,
    CombinedResponse,
    Competency,
    ExpertTrainer,
)

from .credit import (
    CreditLedger,
    CalibrationMetrics,
    PredictionRecord,
    CalibrationBin,
)

from .value import (
    ValueSignal,
    HierarchicalValueSignal,
    ValueRecord,
)

from .staking import (
    StakingManager,
    Stake,
    StakeStatus,
)

from .orchestrator import (
    HubrisMoE,
    QueryResult,
    CombinationStrategy,
)

__all__ = [
    # Expert
    'MicroExpert',
    'MetaExpert',
    'ExpertResponse',
    'CombinedResponse',
    'Competency',
    'ExpertTrainer',
    # Credit
    'CreditLedger',
    'CalibrationMetrics',
    'PredictionRecord',
    'CalibrationBin',
    # Value
    'ValueSignal',
    'HierarchicalValueSignal',
    'ValueRecord',
    # Staking
    'StakingManager',
    'Stake',
    'StakeStatus',
    # Orchestrator
    'HubrisMoE',
    'QueryResult',
    'CombinationStrategy',
]
