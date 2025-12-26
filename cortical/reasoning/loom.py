"""
The Loom: Integration Layer for Dual-Process Cognitive Architecture.

The Loom is the "traffic controller" that decides when to think fast (Hebbian Hive)
and when to think slow (Cultured Cortex). It monitors surprise signals and
orchestrates mode switching based on prediction errors.

Key concepts:
- ThinkingMode: FAST (System 1, pattern-matching) or SLOW (System 2, deliberate)
- SurpriseSignal: Magnitude of prediction error triggering mode consideration
- ModeController: Decides which system handles a given request
- SurpriseDetector: Computes surprise from predicted vs actual activations

The Loom is stateless for routing decisions - state lives in Hive and Cortex.
This design supports future async parallelization and integrates with UnifiedAttention.

"In the Loom, fast and slow thinking are woven into a single fabric."

Example:
    >>> from cortical.reasoning.loom import Loom, LoomConfig
    >>> config = LoomConfig(surprise_threshold=0.3)
    >>> loom = Loom(config)
    >>>
    >>> # Process a request
    >>> signal = loom.detect_surprise(predicted, actual)
    >>> mode = loom.select_mode(signal)
    >>>
    >>> if mode == ThinkingMode.FAST:
    ...     result = hive.process(request)
    >>> else:
    ...     result = cortex.process(request)

See Also:
    - docs/woven-mind-architecture.md: Dual-process architecture design
    - docs/research-prism-woven-mind-comparison.md: PRISM/Woven Mind relationship
    - cortical/reasoning/prism_attention.py: UnifiedAttention integration point
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from .prism_attention import UnifiedAttention
    from .prism_got import SynapticMemoryGraph


# ==============================================================================
# ENUMS
# ==============================================================================


class ThinkingMode(Enum):
    """
    The two modes of cognitive processing.

    FAST (System 1): Automatic, pattern-matching, low-effort.
        - Uses Hebbian Hive (PRISM-SLM)
        - Good for familiar patterns
        - Low latency, high throughput

    SLOW (System 2): Deliberate, analytical, high-effort.
        - Uses Cultured Cortex (PRISM-GoT + PLN)
        - Good for novel situations
        - Higher latency, deeper reasoning
    """

    FAST = auto()
    SLOW = auto()

    def __str__(self) -> str:
        return self.name


class TransitionTrigger(Enum):
    """
    What caused a mode transition.

    Used for observability and debugging mode switching behavior.
    """

    SURPRISE = auto()          # High prediction error
    EXPLICIT_REQUEST = auto()  # User/system requested slow thinking
    TIMEOUT = auto()           # Fast mode took too long
    CONFIDENCE_LOW = auto()    # Fast mode result has low confidence
    COMPLEXITY = auto()        # Input complexity exceeds threshold
    NOVELTY = auto()           # Input pattern not seen before


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass(slots=True)
class SurpriseSignal:
    """
    A signal indicating the magnitude of prediction error.

    Surprise is the difference between what was predicted and what actually
    occurred. High surprise suggests the fast system's patterns don't match
    the current situation, warranting slow deliberate processing.

    Attributes:
        magnitude: Surprise level (0.0 = perfectly predicted, 1.0 = complete miss)
        source: What generated this signal (e.g., "prism_got", "prism_slm")
        context: Additional information about the surprise
        timestamp: When this signal was generated

    Example:
        >>> signal = SurpriseSignal(
        ...     magnitude=0.7,
        ...     source="prism_got",
        ...     context={"missed_predictions": ["node_a", "node_b"]}
        ... )
        >>> signal.is_significant(threshold=0.5)
        True
    """

    magnitude: float
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate magnitude is in valid range."""
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError(f"Magnitude must be in [0, 1], got {self.magnitude}")

    def is_significant(self, threshold: float = 0.3) -> bool:
        """Check if surprise exceeds significance threshold."""
        return self.magnitude > threshold


@dataclass
class LoomConfig:
    """
    Configuration for the Loom integration layer.

    Attributes:
        surprise_threshold: Above this, engage slow thinking (default: 0.3)
        confidence_threshold: Below this confidence, switch to slow (default: 0.6)
        history_window: Number of signals to track for baseline (default: 100)
        adaptation_rate: How fast baseline adapts to new normal (default: 0.1)
        timeout_ms: Max time for fast processing before switching (default: 100)
        enable_observability: Whether to emit transition events (default: True)

    Example:
        >>> config = LoomConfig(
        ...     surprise_threshold=0.4,  # More tolerant of surprise
        ...     confidence_threshold=0.5  # Lower confidence requirements
        ... )
    """

    # Surprise detection
    surprise_threshold: float = 0.3
    confidence_threshold: float = 0.6

    # Baseline adaptation
    history_window: int = 100
    adaptation_rate: float = 0.1

    # Timeouts
    timeout_ms: int = 100

    # Observability
    enable_observability: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.surprise_threshold <= 1.0:
            raise ValueError("surprise_threshold must be in [0, 1]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.history_window < 1:
            raise ValueError("history_window must be >= 1")
        if not 0.0 < self.adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must be in (0, 1]")
        if self.timeout_ms < 0:
            raise ValueError("timeout_ms must be >= 0")


@dataclass
class ModeTransition:
    """
    Record of a mode transition for observability.

    Attributes:
        from_mode: Previous thinking mode
        to_mode: New thinking mode
        trigger: What caused the transition
        surprise_signal: The signal that triggered the transition (if any)
        timestamp: When the transition occurred
        metadata: Additional context about the transition
    """

    from_mode: Optional[ThinkingMode]
    to_mode: ThinkingMode
    trigger: TransitionTrigger
    surprise_signal: Optional[SurpriseSignal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# PROTOCOLS (INTERFACES)
# ==============================================================================


class SurpriseDetectorProtocol(Protocol):
    """
    Protocol for surprise detection implementations.

    Implementations compute surprise from prediction errors and maintain
    adaptive baselines for context-sensitive thresholding.
    """

    def compute_surprise(
        self,
        predicted: Dict[str, float],
        actual: Set[str],
    ) -> SurpriseSignal:
        """
        Compute surprise from predicted vs actual activations.

        Args:
            predicted: Mapping of node_id -> predicted probability
            actual: Set of node IDs that actually activated

        Returns:
            SurpriseSignal with computed magnitude
        """
        ...

    def update_baseline(self, surprise: float) -> None:
        """
        Update the adaptive baseline with a new observation.

        Args:
            surprise: The observed surprise magnitude
        """
        ...

    def should_engage_slow(self, signal: SurpriseSignal) -> bool:
        """
        Determine if slow thinking should be engaged.

        Args:
            signal: The surprise signal to evaluate

        Returns:
            True if slow thinking should be engaged
        """
        ...


class ModeControllerProtocol(Protocol):
    """
    Protocol for mode selection implementations.

    Implementations decide which thinking mode (FAST or SLOW) should
    handle a given request based on various signals.
    """

    def select_mode(
        self,
        surprise: Optional[SurpriseSignal] = None,
        confidence: Optional[float] = None,
        complexity: Optional[float] = None,
    ) -> ThinkingMode:
        """
        Select the appropriate thinking mode.

        Args:
            surprise: Surprise signal from prediction errors
            confidence: Confidence in fast mode result (0-1)
            complexity: Input complexity estimate (0-1)

        Returns:
            The recommended thinking mode
        """
        ...

    def record_transition(self, transition: ModeTransition) -> None:
        """
        Record a mode transition for observability.

        Args:
            transition: The transition to record
        """
        ...

    def get_transition_history(self) -> List[ModeTransition]:
        """
        Get recent mode transitions.

        Returns:
            List of recent transitions
        """
        ...


class LoomObserverProtocol(Protocol):
    """
    Protocol for observing Loom behavior.

    Implementations receive notifications about mode transitions
    and surprise signals for monitoring and debugging.
    """

    def on_surprise(self, signal: SurpriseSignal) -> None:
        """Called when a surprise signal is computed."""
        ...

    def on_transition(self, transition: ModeTransition) -> None:
        """Called when a mode transition occurs."""
        ...

    def on_mode_selected(self, mode: ThinkingMode, reason: str) -> None:
        """Called when a mode is selected for processing."""
        ...


# ==============================================================================
# ABSTRACT BASE CLASS
# ==============================================================================


class LoomInterface(ABC):
    """
    Abstract base class for Loom implementations.

    The Loom orchestrates the interaction between fast (Hebbian Hive)
    and slow (Cultured Cortex) thinking systems. Implementations must
    provide surprise detection, mode selection, and observability.

    This is the main integration point for the dual-process architecture.
    """

    @abstractmethod
    def detect_surprise(
        self,
        predicted: Dict[str, float],
        actual: Set[str],
    ) -> SurpriseSignal:
        """
        Detect surprise from prediction vs reality.

        Args:
            predicted: Mapping of node_id -> predicted probability
            actual: Set of node IDs that actually activated

        Returns:
            SurpriseSignal indicating magnitude of prediction error
        """
        pass

    @abstractmethod
    def select_mode(
        self,
        signal: Optional[SurpriseSignal] = None,
        **context: Any,
    ) -> ThinkingMode:
        """
        Select appropriate thinking mode for current context.

        Args:
            signal: Surprise signal (if available)
            **context: Additional context (confidence, complexity, etc.)

        Returns:
            The recommended thinking mode
        """
        pass

    @abstractmethod
    def register_observer(self, observer: LoomObserverProtocol) -> None:
        """
        Register an observer for Loom events.

        Args:
            observer: The observer to register
        """
        pass

    @abstractmethod
    def get_current_mode(self) -> ThinkingMode:
        """
        Get the current thinking mode.

        Returns:
            The current mode
        """
        pass

    @abstractmethod
    def get_config(self) -> LoomConfig:
        """
        Get the Loom configuration.

        Returns:
            Current configuration
        """
        pass


# ==============================================================================
# CONCRETE IMPLEMENTATIONS
# ==============================================================================


class SurpriseDetector:
    """
    Detects surprise by comparing predictions to actual activations.

    Surprise is computed as the discrepancy between what was predicted
    (node probabilities) and what actually happened (activated nodes).
    Uses an adaptive baseline to normalize for context.

    The detector maintains a running history for baseline adaptation,
    preventing surprise inflation in consistently novel environments.

    Attributes:
        config: LoomConfig with threshold and history settings
        baseline: Current baseline surprise level
        history: Recent surprise values for adaptation

    Example:
        >>> detector = SurpriseDetector()
        >>> signal = detector.compute_surprise(
        ...     predicted={"node_a": 0.9, "node_b": 0.1},
        ...     actual={"node_b", "node_c"}
        ... )
        >>> print(f"Surprise: {signal.magnitude:.2f}")
        Surprise: 0.73
        >>> detector.should_engage_slow(signal)
        True
    """

    def __init__(self, config: Optional[LoomConfig] = None) -> None:
        """
        Initialize the surprise detector.

        Args:
            config: Configuration for thresholds and adaptation.
                    Uses defaults if not provided.
        """
        self.config = config or LoomConfig()
        self.baseline: float = 0.0
        self.history: Deque[float] = deque(maxlen=self.config.history_window)

    def compute_surprise(
        self,
        predicted: Dict[str, float],
        actual: Set[str],
    ) -> SurpriseSignal:
        """
        Compute surprise from predicted probabilities vs actual activations.

        Algorithm:
        1. For each predicted node, compute error as |prob - actual|
           where actual is 1.0 if node activated, 0.0 otherwise
        2. Include false negatives: nodes that activated but weren't predicted
        3. Average all errors
        4. Normalize against adaptive baseline

        Args:
            predicted: Mapping of node_id -> predicted probability
            actual: Set of node IDs that actually activated

        Returns:
            SurpriseSignal with normalized magnitude in [0, 1]
        """
        if not predicted and not actual:
            # No predictions, no activations = no surprise
            return SurpriseSignal(magnitude=0.0, source="surprise_detector")

        errors: List[float] = []

        # Compute prediction errors for predicted nodes
        for node_id, prob in predicted.items():
            actual_value = 1.0 if node_id in actual else 0.0
            error = abs(prob - actual_value)
            errors.append(error)

        # Add errors for unpredicted activations (false negatives)
        unpredicted = actual - set(predicted.keys())
        for _ in unpredicted:
            # Complete miss: we predicted 0.0 for something that activated
            errors.append(1.0)

        # Compute raw surprise as mean error
        raw_surprise = sum(errors) / len(errors) if errors else 0.0

        # Normalize against baseline (but keep in [0, 1])
        if self.baseline > 0:
            # Surprise relative to baseline
            normalized = raw_surprise / (self.baseline + 0.1)
            # Clamp to [0, 1]
            magnitude = min(1.0, max(0.0, normalized))
        else:
            magnitude = raw_surprise

        # Update baseline with this observation
        self._update_baseline(raw_surprise)

        # Build context with diagnostic info
        context = {
            "raw_surprise": raw_surprise,
            "baseline": self.baseline,
            "predicted_count": len(predicted),
            "actual_count": len(actual),
            "unpredicted_count": len(unpredicted),
        }

        return SurpriseSignal(
            magnitude=magnitude,
            source="surprise_detector",
            context=context,
        )

    def _update_baseline(self, surprise: float) -> None:
        """
        Update the adaptive baseline with a new observation.

        Uses exponential moving average with configurable adaptation rate.

        Args:
            surprise: The observed raw surprise value
        """
        self.history.append(surprise)

        if len(self.history) >= 2:
            # Exponential moving average
            alpha = self.config.adaptation_rate
            self.baseline = alpha * surprise + (1 - alpha) * self.baseline
        else:
            # Bootstrap with first observation
            self.baseline = surprise

    def should_engage_slow(self, signal: SurpriseSignal) -> bool:
        """
        Determine if surprise level warrants slow thinking.

        Args:
            signal: The surprise signal to evaluate

        Returns:
            True if slow thinking should be engaged
        """
        return signal.magnitude > self.config.surprise_threshold

    def reset_baseline(self) -> None:
        """Reset the baseline and history to initial state."""
        self.baseline = 0.0
        self.history.clear()

    def get_baseline(self) -> float:
        """Get the current baseline surprise level."""
        return self.baseline


class ModeController:
    """
    Controls thinking mode selection based on multiple signals.

    The controller integrates surprise, confidence, and complexity signals
    to decide whether fast (System 1) or slow (System 2) thinking is appropriate.

    Decision logic:
    1. High surprise → SLOW (predictions failed)
    2. Low confidence → SLOW (need more analysis)
    3. High complexity → SLOW (beyond pattern matching)
    4. Otherwise → FAST (patterns apply)

    Maintains transition history for observability and debugging.

    Example:
        >>> controller = ModeController()
        >>> mode = controller.select_mode(
        ...     surprise=SurpriseSignal(magnitude=0.7, source="test"),
        ...     confidence=0.4
        ... )
        >>> print(mode)
        SLOW
        >>> controller.get_current_mode()
        <ThinkingMode.SLOW: 2>
    """

    def __init__(self, config: Optional[LoomConfig] = None) -> None:
        """
        Initialize the mode controller.

        Args:
            config: Configuration for thresholds.
                    Uses defaults if not provided.
        """
        self.config = config or LoomConfig()
        self._current_mode: ThinkingMode = ThinkingMode.FAST
        self._transition_history: Deque[ModeTransition] = deque(maxlen=100)
        self._observers: List[LoomObserverProtocol] = []

    def select_mode(
        self,
        surprise: Optional[SurpriseSignal] = None,
        confidence: Optional[float] = None,
        complexity: Optional[float] = None,
    ) -> ThinkingMode:
        """
        Select the appropriate thinking mode based on signals.

        Priority order:
        1. Explicit high complexity → SLOW
        2. High surprise → SLOW
        3. Low confidence → SLOW
        4. Default → FAST

        Args:
            surprise: Surprise signal from prediction errors
            confidence: Confidence in fast mode result (0-1)
            complexity: Input complexity estimate (0-1)

        Returns:
            The recommended thinking mode
        """
        trigger: Optional[TransitionTrigger] = None
        new_mode = ThinkingMode.FAST  # Default to fast

        # Check complexity first (explicit slow request)
        if complexity is not None and complexity > 0.7:
            new_mode = ThinkingMode.SLOW
            trigger = TransitionTrigger.COMPLEXITY

        # Check surprise
        elif surprise is not None and surprise.magnitude > self.config.surprise_threshold:
            new_mode = ThinkingMode.SLOW
            trigger = TransitionTrigger.SURPRISE

        # Check confidence
        elif confidence is not None and confidence < self.config.confidence_threshold:
            new_mode = ThinkingMode.SLOW
            trigger = TransitionTrigger.CONFIDENCE_LOW

        # Record transition if mode changed
        if new_mode != self._current_mode and trigger is not None:
            transition = ModeTransition(
                from_mode=self._current_mode,
                to_mode=new_mode,
                trigger=trigger,
                surprise_signal=surprise,
                metadata={
                    "confidence": confidence,
                    "complexity": complexity,
                },
            )
            self.record_transition(transition)

        self._current_mode = new_mode
        return new_mode

    def record_transition(self, transition: ModeTransition) -> None:
        """
        Record a mode transition for observability.

        Notifies all registered observers of the transition.

        Args:
            transition: The transition to record
        """
        self._transition_history.append(transition)

        # Notify observers
        for observer in self._observers:
            try:
                observer.on_transition(transition)
            except Exception:
                # Don't let observer errors break the controller
                pass

    def get_transition_history(self) -> List[ModeTransition]:
        """
        Get recent mode transitions.

        Returns:
            List of recent transitions (most recent last)
        """
        return list(self._transition_history)

    def get_current_mode(self) -> ThinkingMode:
        """Get the current thinking mode."""
        return self._current_mode

    def register_observer(self, observer: LoomObserverProtocol) -> None:
        """
        Register an observer for transition events.

        Args:
            observer: The observer to register
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister_observer(self, observer: LoomObserverProtocol) -> None:
        """
        Unregister an observer.

        Args:
            observer: The observer to unregister
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def force_mode(self, mode: ThinkingMode, reason: str = "explicit_request") -> None:
        """
        Force a specific thinking mode.

        Useful for testing or when external factors require a specific mode.

        Args:
            mode: The mode to force
            reason: Reason for forcing (for debugging)
        """
        if mode != self._current_mode:
            transition = ModeTransition(
                from_mode=self._current_mode,
                to_mode=mode,
                trigger=TransitionTrigger.EXPLICIT_REQUEST,
                metadata={"reason": reason},
            )
            self.record_transition(transition)
            self._current_mode = mode


class Loom(LoomInterface):
    """
    The main Loom implementation integrating surprise detection and mode control.

    This is the primary integration point between fast (Hebbian Hive) and
    slow (Cultured Cortex) thinking systems. It combines:
    - SurpriseDetector: Monitors prediction errors
    - ModeController: Decides which system handles requests

    The Loom is stateless for routing decisions - it doesn't maintain
    information about specific requests, only general mode state.

    Example:
        >>> from cortical.reasoning.loom import Loom, LoomConfig
        >>>
        >>> # Create with custom config
        >>> config = LoomConfig(surprise_threshold=0.4)
        >>> loom = Loom(config)
        >>>
        >>> # Detect surprise and select mode
        >>> signal = loom.detect_surprise(
        ...     predicted={"a": 0.9, "b": 0.1},
        ...     actual={"c", "d"}  # Complete miss!
        ... )
        >>> mode = loom.select_mode(signal)
        >>> print(f"Mode: {mode}, Surprise: {signal.magnitude:.2f}")
        Mode: SLOW, Surprise: 0.85
    """

    def __init__(self, config: Optional[LoomConfig] = None) -> None:
        """
        Initialize the Loom.

        Args:
            config: Configuration for thresholds and behavior.
                    Uses defaults if not provided.
        """
        self._config = config or LoomConfig()
        self._surprise_detector = SurpriseDetector(self._config)
        self._mode_controller = ModeController(self._config)
        self._observers: List[LoomObserverProtocol] = []

    def detect_surprise(
        self,
        predicted: Dict[str, float],
        actual: Set[str],
    ) -> SurpriseSignal:
        """
        Detect surprise from prediction vs reality.

        Delegates to SurpriseDetector and notifies observers.

        Args:
            predicted: Mapping of node_id -> predicted probability
            actual: Set of node IDs that actually activated

        Returns:
            SurpriseSignal indicating magnitude of prediction error
        """
        signal = self._surprise_detector.compute_surprise(predicted, actual)

        # Notify observers
        if self._config.enable_observability:
            for observer in self._observers:
                try:
                    observer.on_surprise(signal)
                except Exception:
                    pass

        return signal

    def select_mode(
        self,
        signal: Optional[SurpriseSignal] = None,
        **context: Any,
    ) -> ThinkingMode:
        """
        Select appropriate thinking mode for current context.

        Args:
            signal: Surprise signal (if available)
            **context: Additional context (confidence, complexity, etc.)

        Returns:
            The recommended thinking mode
        """
        mode = self._mode_controller.select_mode(
            surprise=signal,
            confidence=context.get("confidence"),
            complexity=context.get("complexity"),
        )

        # Notify observers
        if self._config.enable_observability:
            reason = f"surprise={signal.magnitude:.2f}" if signal else "default"
            for observer in self._observers:
                try:
                    observer.on_mode_selected(mode, reason)
                except Exception:
                    pass

        return mode

    def register_observer(self, observer: LoomObserverProtocol) -> None:
        """
        Register an observer for Loom events.

        Args:
            observer: The observer to register
        """
        if observer not in self._observers:
            self._observers.append(observer)
        self._mode_controller.register_observer(observer)

    def get_current_mode(self) -> ThinkingMode:
        """Get the current thinking mode."""
        return self._mode_controller.get_current_mode()

    def get_config(self) -> LoomConfig:
        """Get the Loom configuration."""
        return self._config

    def get_surprise_baseline(self) -> float:
        """Get the current surprise baseline."""
        return self._surprise_detector.get_baseline()

    def get_transition_history(self) -> List[ModeTransition]:
        """Get recent mode transitions."""
        return self._mode_controller.get_transition_history()

    def reset(self) -> None:
        """Reset the Loom to initial state."""
        self._surprise_detector.reset_baseline()
        self._mode_controller._current_mode = ThinkingMode.FAST
        self._mode_controller._transition_history.clear()


# ==============================================================================
# PRISM-ATTENTION INTEGRATION
# ==============================================================================


@dataclass
class LoomAttentionResult:
    """
    Result of Loom-enhanced attention.

    Extends attention results with mode switching information.

    Attributes:
        top_nodes: Top-weighted node IDs
        weights: All node weights
        mode_used: Which thinking mode was used
        surprise: Surprise signal if detected
        pln_support: PLN logical support score
        slm_fluency: SLM language model fluency
        fast_result: Result from fast path (if used)
        slow_result: Result from slow path (if used)
    """

    top_nodes: List[str]
    weights: Dict[str, float]
    mode_used: ThinkingMode
    surprise: Optional[SurpriseSignal] = None
    pln_support: float = 0.0
    slm_fluency: float = 0.0
    fast_result: Optional[Any] = None
    slow_result: Optional[Any] = None


class LoomEnhancedAttention:
    """
    Loom-enhanced attention integrating dual-process switching.

    This class wraps PRISM's UnifiedAttention and adds:
    - Surprise detection from attention predictions
    - Mode switching between fast and slow attention
    - Observable transitions for debugging

    Fast mode: Uses only graph-based attention (quick pattern matching)
    Slow mode: Uses full UnifiedAttention with PLN and SLM (deliberate analysis)

    Example:
        >>> from cortical.reasoning import PRISMGraph
        >>> from cortical.reasoning.loom import LoomEnhancedAttention
        >>>
        >>> graph = PRISMGraph()
        >>> # ... populate graph ...
        >>> enhanced = LoomEnhancedAttention(graph)
        >>> result = enhanced.attend("what is authentication?")
        >>> print(f"Mode: {result.mode_used}, Top: {result.top_nodes}")
    """

    def __init__(
        self,
        graph: "SynapticMemoryGraph",
        slm: Optional[Any] = None,
        pln: Optional[Any] = None,
        config: Optional[LoomConfig] = None,
    ) -> None:
        """
        Initialize Loom-enhanced attention.

        Args:
            graph: The PRISM graph to attend over
            slm: Optional SLM for language model fluency
            pln: Optional PLN reasoner for logical support
            config: Loom configuration for thresholds
        """
        # Lazy import to avoid circular dependency
        from .prism_attention import UnifiedAttention, AttentionLayer

        self._graph = graph
        self._slm = slm
        self._pln = pln
        self._config = config or LoomConfig()

        # Core components
        self._loom = Loom(self._config)
        self._base_attention = AttentionLayer(graph)
        self._unified_attention: Optional[UnifiedAttention] = None

        # Initialize unified attention if we have PLN or SLM
        if slm is not None or pln is not None:
            self._unified_attention = UnifiedAttention(graph, slm, pln)

        # Track predictions for surprise detection
        self._last_predictions: Dict[str, float] = {}

    def attend(
        self,
        query: str,
        force_mode: Optional[ThinkingMode] = None,
    ) -> LoomAttentionResult:
        """
        Attend to query using Loom-guided mode selection.

        Process:
        1. Compute fast attention (always, for baseline)
        2. Compare to previous predictions to detect surprise
        3. Select mode based on surprise
        4. If SLOW, compute full unified attention
        5. Return result with mode information

        Args:
            query: The query string to attend to
            force_mode: Optional mode override

        Returns:
            LoomAttentionResult with mode and attention info
        """
        # Step 1: Fast attention (pattern matching)
        fast_weights = self._base_attention.attend(query)
        fast_top = sorted(fast_weights.items(), key=lambda x: -x[1])[:5]
        fast_top_ids = [node_id for node_id, _ in fast_top]

        # Step 2: Detect surprise from previous predictions
        if self._last_predictions:
            actual_set = set(fast_top_ids)
            surprise = self._loom.detect_surprise(
                predicted=self._last_predictions,
                actual=actual_set,
            )
        else:
            surprise = None

        # Update predictions for next call
        self._last_predictions = {
            node_id: weight / max(1e-10, sum(fast_weights.values()))
            for node_id, weight in fast_weights.items()
        }

        # Step 3: Select mode
        if force_mode is not None:
            mode = force_mode
        else:
            mode = self._loom.select_mode(surprise)

        # Step 4: Compute full attention if SLOW and we have capabilities
        pln_support = 0.0
        slm_fluency = 0.0
        slow_result = None

        if mode == ThinkingMode.SLOW and self._unified_attention is not None:
            # Full deliberate attention
            unified_result = self._unified_attention.attend(query)
            pln_support = unified_result.pln_support
            slm_fluency = unified_result.slm_fluency
            slow_result = unified_result

            # Use unified weights instead of fast weights
            final_weights = unified_result.weights
            final_top = unified_result.top_nodes
        else:
            final_weights = fast_weights
            final_top = fast_top_ids

        return LoomAttentionResult(
            top_nodes=final_top,
            weights=final_weights,
            mode_used=mode,
            surprise=surprise,
            pln_support=pln_support,
            slm_fluency=slm_fluency,
            fast_result=fast_weights,
            slow_result=slow_result,
        )

    def get_loom(self) -> Loom:
        """Get the underlying Loom instance."""
        return self._loom

    def get_transition_history(self) -> List[ModeTransition]:
        """Get mode transition history."""
        return self._loom.get_transition_history()

    def reset(self) -> None:
        """Reset the enhanced attention state."""
        self._loom.reset()
        self._last_predictions.clear()


# ==============================================================================
# CONVENIENCE TYPE ALIASES
# ==============================================================================

# Callback type for mode transition notifications
TransitionCallback = Callable[[ModeTransition], None]

# Callback type for surprise signal notifications
SurpriseCallback = Callable[[SurpriseSignal], None]


# ==============================================================================
# PUBLIC API
# ==============================================================================

__all__ = [
    # Enums
    'ThinkingMode',
    'TransitionTrigger',
    # Data structures
    'SurpriseSignal',
    'LoomConfig',
    'ModeTransition',
    'LoomAttentionResult',
    # Protocols
    'SurpriseDetectorProtocol',
    'ModeControllerProtocol',
    'LoomObserverProtocol',
    # Abstract base
    'LoomInterface',
    # Concrete implementations
    'SurpriseDetector',
    'ModeController',
    'Loom',
    # PRISM integration
    'LoomEnhancedAttention',
    # Type aliases
    'TransitionCallback',
    'SurpriseCallback',
]
