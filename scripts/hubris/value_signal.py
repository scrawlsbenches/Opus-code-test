#!/usr/bin/env python3
"""
Value Signal and Attribution System

Tracks when expert predictions create value and attributes credit/debit
to experts based on the outcome. This enables the credit-based economy
where experts earn credit for helpful predictions and lose credit for
harmful ones.

Value signals come from multiple sources:
- Test results (did the prediction help tests pass?)
- Commit results (were the predicted files actually modified?)
- User feedback (was the suggestion helpful?)
- CI/CD outcomes (did the build succeed?)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class SignalType(Enum):
    """Types of value signals."""
    POSITIVE = "positive"  # Prediction helped
    NEGATIVE = "negative"  # Prediction hurt
    NEUTRAL = "neutral"    # No clear impact


@dataclass(frozen=True)
class ValueSignal:
    """
    Represents feedback that a prediction was helpful or harmful.

    Immutable record of value created or destroyed by an expert's prediction.

    Attributes:
        signal_type: "positive", "negative", or "neutral"
        magnitude: Strength of signal (0.0 to 1.0)
        timestamp: When signal was generated
        source: Where signal came from (e.g., "user_feedback", "test_result")
        expert_id: Which expert's prediction this evaluates
        prediction_id: Links to the original prediction
        context: Additional metadata about the signal
    """
    signal_type: str
    magnitude: float
    timestamp: float
    source: str
    expert_id: str
    prediction_id: str
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal on creation."""
        # Validate magnitude
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError(f"magnitude must be in [0.0, 1.0], got {self.magnitude}")

        # Validate signal_type
        valid_types = {st.value for st in SignalType}
        if self.signal_type not in valid_types:
            raise ValueError(
                f"signal_type must be one of {valid_types}, got {self.signal_type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'signal_type': self.signal_type,
            'magnitude': self.magnitude,
            'timestamp': self.timestamp,
            'source': self.source,
            'expert_id': self.expert_id,
            'prediction_id': self.prediction_id,
            'context': self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValueSignal':
        """Load from dict."""
        return cls(
            signal_type=data['signal_type'],
            magnitude=data['magnitude'],
            timestamp=data['timestamp'],
            source=data['source'],
            expert_id=data['expert_id'],
            prediction_id=data['prediction_id'],
            context=data.get('context', {})
        )


class ValueAttributor:
    """
    Processes value signals and attributes credit/debit to experts.

    Attribution rules:
    - Positive signals: base credit = magnitude * 10.0
    - Negative signals: base debit = magnitude * 5.0 (less punitive)
    - Confidence multiplier: scale by prediction confidence
    - Time decay: older predictions get reduced attribution (optional)
    """

    def __init__(
        self,
        positive_multiplier: float = 10.0,
        negative_multiplier: float = 5.0,
        use_confidence_scaling: bool = True,
        use_time_decay: bool = False,
        decay_halflife_seconds: float = 86400.0  # 1 day
    ):
        """
        Initialize attributor with configuration.

        Args:
            positive_multiplier: Credit amount for positive signals
            negative_multiplier: Debit amount for negative signals
            use_confidence_scaling: Scale by prediction confidence
            use_time_decay: Apply time-based decay to attribution
            decay_halflife_seconds: Half-life for exponential decay
        """
        self.positive_multiplier = positive_multiplier
        self.negative_multiplier = negative_multiplier
        self.use_confidence_scaling = use_confidence_scaling
        self.use_time_decay = use_time_decay
        self.decay_halflife_seconds = decay_halflife_seconds

    def calculate_credit_amount(self, signal: ValueSignal) -> float:
        """
        Determine credit/debit amount for a signal.

        Args:
            signal: The value signal to process

        Returns:
            Credit amount (positive for credit, negative for debit)
        """
        # Base credit/debit from signal type and magnitude
        if signal.signal_type == SignalType.POSITIVE.value:
            base_amount = signal.magnitude * self.positive_multiplier
        elif signal.signal_type == SignalType.NEGATIVE.value:
            base_amount = -signal.magnitude * self.negative_multiplier
        else:  # NEUTRAL
            base_amount = 0.0

        # Apply confidence scaling if enabled
        if self.use_confidence_scaling and 'confidence' in signal.context:
            confidence = signal.context['confidence']
            base_amount *= confidence

        # Apply time decay if enabled
        if self.use_time_decay and 'prediction_time' in signal.context:
            prediction_time = signal.context['prediction_time']
            time_diff = signal.timestamp - prediction_time

            # Exponential decay: amount = base * 0.5^(time_diff / halflife)
            import math
            decay_factor = math.pow(0.5, time_diff / self.decay_halflife_seconds)
            base_amount *= decay_factor

        return base_amount

    def process_signal(self, signal: ValueSignal, ledger: Any) -> float:
        """
        Process a signal and apply credit/debit to ledger.

        Args:
            signal: The value signal to process
            ledger: CreditLedger to update (duck-typed)

        Returns:
            Amount credited/debited
        """
        amount = self.calculate_credit_amount(signal)

        # Get or create account for the expert
        account = ledger.get_or_create_account(signal.expert_id)

        # Apply to account
        if amount > 0:
            account.credit(amount, signal.source, signal.context)
        elif amount < 0:
            account.debit(abs(amount), signal.source, signal.context)

        return amount

    def attribute_from_test_result(
        self,
        expert_id: str,
        prediction_id: str,
        test_passed: bool,
        confidence: float,
        prediction_time: Optional[float] = None
    ) -> ValueSignal:
        """
        Create signal from test result.

        Args:
            expert_id: Expert that made the prediction
            prediction_id: ID of the prediction being evaluated
            test_passed: Whether the test passed
            confidence: Prediction confidence (0-1)
            prediction_time: When prediction was made

        Returns:
            ValueSignal for the test result
        """
        signal_type = SignalType.POSITIVE.value if test_passed else SignalType.NEGATIVE.value
        magnitude = confidence  # Higher confidence predictions get more credit/blame

        context = {
            'confidence': confidence,
            'test_passed': test_passed
        }
        if prediction_time is not None:
            context['prediction_time'] = prediction_time

        return ValueSignal(
            signal_type=signal_type,
            magnitude=magnitude,
            timestamp=time.time(),
            source='test_result',
            expert_id=expert_id,
            prediction_id=prediction_id,
            context=context
        )

    def attribute_from_commit_result(
        self,
        expert_id: str,
        prediction_id: str,
        files_correct: List[str],
        files_total: List[str],
        confidence: float = 0.7,
        prediction_time: Optional[float] = None
    ) -> ValueSignal:
        """
        Create signal from commit result.

        Args:
            expert_id: Expert that made the prediction
            prediction_id: ID of the prediction being evaluated
            files_correct: Files that were predicted and actually modified
            files_total: All files that were predicted
            confidence: Prediction confidence (0-1)
            prediction_time: When prediction was made

        Returns:
            ValueSignal for the commit result
        """
        # Calculate accuracy as magnitude
        if not files_total:
            magnitude = 0.0
        else:
            magnitude = len(files_correct) / len(files_total)

        # Positive if >50% correct, negative if <50% correct
        signal_type = (
            SignalType.POSITIVE.value if magnitude > 0.5
            else SignalType.NEGATIVE.value if magnitude < 0.5
            else SignalType.NEUTRAL.value
        )

        context = {
            'confidence': confidence,
            'files_correct': files_correct,
            'files_total': files_total,
            'accuracy': magnitude
        }
        if prediction_time is not None:
            context['prediction_time'] = prediction_time

        return ValueSignal(
            signal_type=signal_type,
            magnitude=magnitude,
            timestamp=time.time(),
            source='commit_result',
            expert_id=expert_id,
            prediction_id=prediction_id,
            context=context
        )

    def attribute_from_user_feedback(
        self,
        expert_id: str,
        prediction_id: str,
        helpful: bool,
        importance: float = 0.5,
        prediction_time: Optional[float] = None
    ) -> ValueSignal:
        """
        Create signal from user feedback.

        Args:
            expert_id: Expert that made the prediction
            prediction_id: ID of the prediction being evaluated
            helpful: Whether user found prediction helpful
            importance: How important this feedback is (0-1)
            prediction_time: When prediction was made

        Returns:
            ValueSignal for the user feedback
        """
        signal_type = SignalType.POSITIVE.value if helpful else SignalType.NEGATIVE.value
        magnitude = importance

        context = {
            'helpful': helpful,
            'importance': importance
        }
        if prediction_time is not None:
            context['prediction_time'] = prediction_time

        return ValueSignal(
            signal_type=signal_type,
            magnitude=magnitude,
            timestamp=time.time(),
            source='user_feedback',
            expert_id=expert_id,
            prediction_id=prediction_id,
            context=context
        )


class SignalBuffer:
    """
    Batches value signals for efficient processing.

    Allows accumulating signals and processing them in bulk to reduce
    overhead and enable batch optimizations.
    """

    def __init__(self):
        """Initialize empty signal buffer."""
        self.signals: List[ValueSignal] = []

    def add(self, signal: ValueSignal) -> None:
        """
        Add signal to buffer.

        Args:
            signal: ValueSignal to add
        """
        self.signals.append(signal)

    def flush(self, ledger: Any, attributor: Optional[ValueAttributor] = None) -> Dict[str, float]:
        """
        Process all buffered signals and clear buffer.

        Args:
            ledger: CreditLedger to update
            attributor: ValueAttributor to use (creates default if None)

        Returns:
            Dictionary mapping expert_id to total credit/debit amount
        """
        if attributor is None:
            attributor = ValueAttributor()

        totals: Dict[str, float] = {}

        for signal in self.signals:
            amount = attributor.process_signal(signal, ledger)
            totals[signal.expert_id] = totals.get(signal.expert_id, 0.0) + amount

        # Clear buffer
        self.signals = []

        return totals

    def get_pending_count(self) -> int:
        """
        Get number of pending signals.

        Returns:
            Number of signals in buffer
        """
        return len(self.signals)

    def clear(self) -> None:
        """Clear all pending signals without processing."""
        self.signals = []

    def peek(self) -> List[ValueSignal]:
        """
        View pending signals without removing them.

        Returns:
            List of pending signals
        """
        return self.signals.copy()
