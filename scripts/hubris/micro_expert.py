#!/usr/bin/env python3
"""
Micro-Expert Base Classes

Foundation classes for the Mixture of Experts (MoE) system.
Each micro-expert is a specialized model trained on specific aspects
of coding tasks (file prediction, test selection, error diagnosis, etc.)

Inspired by Thousand Brains Theory where multiple specialized cortical
columns vote to reach consensus.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod


@dataclass
class ExpertMetrics:
    """
    Performance metrics for an expert.

    Tracks accuracy and calibration across evaluations.
    """
    mrr: float = 0.0                                    # Mean Reciprocal Rank
    recall_at_k: Dict[int, float] = field(default_factory=dict)  # Recall@K for K in [1,3,5,10]
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # Precision@K
    calibration_error: float = 0.0                      # Expected calibration error
    test_examples: int = 0                              # Number of test examples used
    last_evaluated: str = ""                            # ISO timestamp of last evaluation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertMetrics':
        """Load from dict."""
        return cls(**data)


@dataclass
class ExpertPrediction:
    """
    A single prediction from an expert.

    Represents the expert's ranked predictions with confidence scores.
    """
    expert_id: str                                      # Unique expert identifier
    expert_type: str                                    # Expert type (file, test, error, etc.)
    items: List[Tuple[str, float]]                      # Ranked (item, confidence) pairs
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'expert_id': self.expert_id,
            'expert_type': self.expert_type,
            'items': self.items,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertPrediction':
        """Load from dict."""
        # Convert items from lists back to tuples
        items = [tuple(item) if isinstance(item, list) else item
                 for item in data['items']]
        return cls(
            expert_id=data['expert_id'],
            expert_type=data['expert_type'],
            items=items,
            metadata=data.get('metadata', {})
        )


@dataclass
class AggregatedPrediction:
    """
    Final aggregated prediction from multiple experts.

    Combines predictions from multiple experts with consensus metrics.
    """
    items: List[Tuple[str, float]]                      # Final ranked (item, confidence)
    contributing_experts: List[str]                     # Expert IDs that contributed
    disagreement_score: float                           # How much experts disagreed (0-1)
    confidence: float                                   # Overall confidence (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'items': self.items,
            'contributing_experts': self.contributing_experts,
            'disagreement_score': self.disagreement_score,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregatedPrediction':
        """Load from dict."""
        items = [tuple(item) if isinstance(item, list) else item
                 for item in data['items']]
        return cls(
            items=items,
            contributing_experts=data['contributing_experts'],
            disagreement_score=data['disagreement_score'],
            confidence=data['confidence'],
            metadata=data.get('metadata', {})
        )


class MicroExpert(ABC):
    """
    Base class for all micro-experts.

    Each expert is a specialized model trained on specific aspects of
    coding tasks. Experts can be combined through voting to reach consensus.

    Attributes:
        expert_id: Unique identifier for this expert
        expert_type: Type of expert (file, test, error, doc, refactor, etc.)
        version: Semantic version of the expert
        created_at: ISO timestamp of creation
        trained_on_commits: Number of commits used for training
        trained_on_sessions: Number of sessions contributing to training
        git_hash: Git commit hash at training time
        model_data: Expert-specific model parameters and learned patterns
        metrics: Performance metrics (optional)
        calibration_curve: Confidence calibration data (optional)
    """

    def __init__(
        self,
        expert_id: str,
        expert_type: str,
        version: str = "1.0.0",
        created_at: Optional[str] = None,
        trained_on_commits: int = 0,
        trained_on_sessions: int = 0,
        git_hash: str = "",
        model_data: Optional[Dict[str, Any]] = None,
        metrics: Optional[ExpertMetrics] = None,
        calibration_curve: Optional[List[Tuple[float, float]]] = None
    ):
        self.expert_id = expert_id
        self.expert_type = expert_type
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.trained_on_commits = trained_on_commits
        self.trained_on_sessions = trained_on_sessions
        self.git_hash = git_hash
        self.model_data = model_data or {}
        self.metrics = metrics
        self.calibration_curve = calibration_curve

    @abstractmethod
    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Make a prediction given context.

        Args:
            context: Dictionary with prediction context (varies by expert type)

        Returns:
            ExpertPrediction with ranked items and confidence scores
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert expert to JSON-serializable dict.

        Returns:
            Dictionary representation of expert
        """
        return {
            'expert_id': self.expert_id,
            'expert_type': self.expert_type,
            'version': self.version,
            'created_at': self.created_at,
            'trained_on_commits': self.trained_on_commits,
            'trained_on_sessions': self.trained_on_sessions,
            'git_hash': self.git_hash,
            'model_data': self.model_data,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'calibration_curve': self.calibration_curve
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MicroExpert':
        """
        Load expert from dict.

        Note: This returns a generic MicroExpert. Subclasses should override
        to return the correct expert type.

        Args:
            data: Dictionary representation

        Returns:
            MicroExpert instance
        """
        metrics = ExpertMetrics.from_dict(data['metrics']) if data.get('metrics') else None

        # Create instance with proper subclass if available
        expert_type = data.get('expert_type', '')

        # This will be overridden by subclasses
        return cls(
            expert_id=data['expert_id'],
            expert_type=expert_type,
            version=data['version'],
            created_at=data['created_at'],
            trained_on_commits=data['trained_on_commits'],
            trained_on_sessions=data['trained_on_sessions'],
            git_hash=data['git_hash'],
            model_data=data['model_data'],
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )

    def save(self, path: Path) -> None:
        """
        Save expert to JSON file.

        Args:
            path: File path to save to (will be created if doesn't exist)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'MicroExpert':
        """
        Load expert from JSON file.

        Args:
            path: File path to load from

        Returns:
            MicroExpert instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_metrics(self, metrics: ExpertMetrics) -> None:
        """
        Update expert's performance metrics.

        Args:
            metrics: New metrics to store
        """
        self.metrics = metrics

    def get_confidence_calibration(self, predicted_confidence: float) -> float:
        """
        Apply calibration curve to predicted confidence.

        If no calibration curve exists, returns the raw confidence.

        Args:
            predicted_confidence: Raw predicted confidence (0-1)

        Returns:
            Calibrated confidence (0-1)
        """
        if not self.calibration_curve:
            return predicted_confidence

        # Find closest calibration points
        curve = sorted(self.calibration_curve)

        # Edge cases
        if predicted_confidence <= curve[0][0]:
            return curve[0][1]
        if predicted_confidence >= curve[-1][0]:
            return curve[-1][1]

        # Linear interpolation between calibration points
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]

            if x1 <= predicted_confidence <= x2:
                # Interpolate
                t = (predicted_confidence - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)

        # Fallback (shouldn't reach here)
        return predicted_confidence

    def __repr__(self) -> str:
        """String representation."""
        return (f"MicroExpert(id={self.expert_id}, type={self.expert_type}, "
                f"version={self.version}, commits={self.trained_on_commits})")
