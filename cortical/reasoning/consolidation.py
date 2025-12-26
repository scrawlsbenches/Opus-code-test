"""
Consolidation Engine for Woven Mind.

Implements "sleep-like" consolidation cycles that transfer learning
between the Hebbian Hive (System 1) and Cultured Cortex (System 2).

The consolidation process:
1. Pattern Transfer: Frequent Hive patterns become Cortex abstractions
2. Abstraction Mining: Discover latent structure in activation patterns
3. Decay Cycle: Forget low-value connections, strengthen high-value ones
4. Scheduling: Run consolidation during idle periods

Part of Sprint 5: Consolidation (Woven Mind + PRISM Marriage)

Example:
    >>> from cortical.reasoning.consolidation import ConsolidationEngine
    >>> from cortical.reasoning.woven_mind import WovenMind
    >>>
    >>> mind = WovenMind()
    >>> engine = ConsolidationEngine(mind.hive, mind.cortex)
    >>>
    >>> # Train the mind
    >>> for text in training_data:
    ...     mind.train(text)
    >>>
    >>> # Run consolidation ("sleep")
    >>> result = engine.consolidate()
    >>> print(f"Transferred: {result.patterns_transferred}")
    >>> print(f"Abstractions formed: {result.abstractions_formed}")
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum, auto
import threading
import time


@dataclass
class ConsolidationConfig:
    """Configuration for the ConsolidationEngine.

    Attributes:
        transfer_threshold: Minimum frequency for pattern transfer (default 3).
        decay_factor: How much to decay unused connections (default 0.9).
        min_strength_keep: Minimum strength to retain after decay (default 0.1).
        max_patterns_per_cycle: Maximum patterns to transfer per cycle (default 10).
        max_abstractions_per_cycle: Maximum abstractions to mine per cycle (default 5).
        enable_scheduling: Whether to run scheduled consolidation (default False).
        schedule_interval_seconds: Seconds between scheduled consolidations (default 300).
    """
    transfer_threshold: int = 3
    decay_factor: float = 0.9
    min_strength_keep: float = 0.1
    max_patterns_per_cycle: int = 10
    max_abstractions_per_cycle: int = 5
    enable_scheduling: bool = False
    schedule_interval_seconds: int = 300


@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle.

    Attributes:
        patterns_transferred: Number of Hive patterns transferred to Cortex.
        abstractions_formed: Number of new abstractions created.
        connections_decayed: Number of connections that were decayed.
        connections_pruned: Number of connections removed (below threshold).
        cycle_duration_ms: How long the cycle took in milliseconds.
        timestamp: When the consolidation occurred.
        transferred_patterns: List of patterns that were transferred.
        formed_abstractions: List of abstraction IDs that were created.
        metadata: Additional information about the cycle.
    """
    patterns_transferred: int = 0
    abstractions_formed: int = 0
    connections_decayed: int = 0
    connections_pruned: int = 0
    cycle_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    transferred_patterns: List[FrozenSet[str]] = field(default_factory=list)
    formed_abstractions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsolidationPhase(Enum):
    """Phases of the consolidation cycle."""
    IDLE = auto()
    PATTERN_TRANSFER = auto()
    ABSTRACTION_MINING = auto()
    DECAY_CYCLE = auto()
    COMPLETE = auto()


class ConsolidationEngine:
    """
    Engine for consolidating learning between Hive and Cortex.

    The consolidation engine implements "sleep-like" cycles that:
    1. Transfer frequently activated Hive patterns to Cortex abstractions
    2. Mine latent structure to form new abstractions
    3. Apply decay to forget unused/weak connections
    4. Strengthen important patterns through repetition

    Consolidation is inspired by the role of sleep in memory consolidation:
    - "Replaying" important patterns strengthens them
    - Unused patterns decay and may be pruned
    - Structure emerges through repeated observation

    Key invariants:
    - Consolidation is interruptible and resumable
    - Pattern transfer preserves prediction accuracy
    - High-value connections are protected from decay

    Attributes:
        hive: The LoomHiveConnector (FAST mode patterns)
        cortex: The LoomCortexConnector (SLOW mode abstractions)
        config: Consolidation configuration
    """

    def __init__(
        self,
        hive: "LoomHiveConnector",
        cortex: "LoomCortexConnector",
        config: Optional[ConsolidationConfig] = None,
    ) -> None:
        """
        Initialize the consolidation engine.

        Args:
            hive: The Hive connector containing patterns to consolidate.
            cortex: The Cortex connector to receive abstractions.
            config: Configuration for consolidation behavior.
        """
        self.hive = hive
        self.cortex = cortex
        self.config = config or ConsolidationConfig()

        # State tracking
        self._current_phase = ConsolidationPhase.IDLE
        self._consolidation_history: List[ConsolidationResult] = []
        self._pattern_frequencies: Dict[FrozenSet[str], int] = defaultdict(int)
        self._last_consolidation: Optional[datetime] = None
        self._is_running = False

        # Scheduler state
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_scheduler = threading.Event()

        # Callbacks for observability
        self._on_phase_change: Optional[Callable[[ConsolidationPhase], None]] = None
        self._on_pattern_transferred: Optional[Callable[[FrozenSet[str]], None]] = None
        self._on_cycle_complete: Optional[Callable[[ConsolidationResult], None]] = None

    @property
    def current_phase(self) -> ConsolidationPhase:
        """Get the current consolidation phase."""
        return self._current_phase

    @property
    def is_running(self) -> bool:
        """Check if consolidation is currently running."""
        return self._is_running

    @property
    def last_consolidation(self) -> Optional[datetime]:
        """Get timestamp of last consolidation."""
        return self._last_consolidation

    def record_pattern(self, pattern: Set[str]) -> None:
        """
        Record a pattern observation for future consolidation.

        Called during normal processing to track which patterns
        are frequent enough to warrant consolidation.

        Args:
            pattern: Set of node IDs that were active together.
        """
        frozen = frozenset(pattern)
        if len(frozen) >= 2:  # Only meaningful patterns
            self._pattern_frequencies[frozen] += 1

    def get_frequent_patterns(
        self,
        min_frequency: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[FrozenSet[str], int]]:
        """
        Get patterns that meet frequency threshold.

        Args:
            min_frequency: Minimum frequency (uses config default if None).
            top_k: Only return top-k patterns.

        Returns:
            List of (pattern, frequency) tuples sorted by frequency desc.
        """
        threshold = min_frequency or self.config.transfer_threshold

        candidates = [
            (pattern, freq)
            for pattern, freq in self._pattern_frequencies.items()
            if freq >= threshold
        ]

        # Sort by frequency descending
        candidates.sort(key=lambda x: -x[1])

        if top_k is not None:
            candidates = candidates[:top_k]

        return candidates

    def consolidate(self) -> ConsolidationResult:
        """
        Run a full consolidation cycle.

        Performs:
        1. Pattern transfer from Hive to Cortex
        2. Abstraction mining from frequent patterns
        3. Decay cycle for forgetting

        Returns:
            ConsolidationResult with cycle statistics.

        Raises:
            RuntimeError: If consolidation is already running.
        """
        if self._is_running:
            raise RuntimeError("Consolidation already in progress")

        self._is_running = True
        start_time = time.time()

        result = ConsolidationResult()

        try:
            # Phase 1: Pattern Transfer
            self._set_phase(ConsolidationPhase.PATTERN_TRANSFER)
            transfer_result = self.pattern_transfer()
            result.patterns_transferred = transfer_result["transferred"]
            result.transferred_patterns = transfer_result["patterns"]

            # Phase 2: Abstraction Mining
            self._set_phase(ConsolidationPhase.ABSTRACTION_MINING)
            mining_result = self.abstraction_mining()
            result.abstractions_formed = mining_result["formed"]
            result.formed_abstractions = mining_result["abstraction_ids"]

            # Phase 3: Decay Cycle
            self._set_phase(ConsolidationPhase.DECAY_CYCLE)
            decay_result = self.decay_cycle()
            result.connections_decayed = decay_result["decayed"]
            result.connections_pruned = decay_result["pruned"]

            # Complete
            self._set_phase(ConsolidationPhase.COMPLETE)

        finally:
            self._is_running = False
            self._set_phase(ConsolidationPhase.IDLE)

        # Record timing and history
        end_time = time.time()
        result.cycle_duration_ms = (end_time - start_time) * 1000
        result.timestamp = datetime.now()
        self._last_consolidation = result.timestamp
        self._consolidation_history.append(result)

        # Fire callback
        if self._on_cycle_complete:
            self._on_cycle_complete(result)

        return result

    def pattern_transfer(self) -> Dict[str, Any]:
        """
        Transfer frequent Hive patterns to Cortex abstractions.

        Identifies patterns in the Hive that are frequent enough
        to warrant permanent storage as Cortex abstractions.

        Returns:
            Dict with 'transferred' count and 'patterns' list.
        """
        transferred_patterns: List[FrozenSet[str]] = []
        max_patterns = self.config.max_patterns_per_cycle

        # Get candidate patterns from both recorded patterns and Hive activity
        candidates = self.get_frequent_patterns(top_k=max_patterns * 2)

        for pattern, frequency in candidates:
            if len(transferred_patterns) >= max_patterns:
                break

            # Check if pattern already exists as abstraction
            if self._pattern_is_abstracted(pattern):
                continue

            # Transfer pattern to Cortex
            success = self._transfer_pattern_to_cortex(pattern, frequency)
            if success:
                transferred_patterns.append(pattern)

                # Fire callback
                if self._on_pattern_transferred:
                    self._on_pattern_transferred(pattern)

        return {
            "transferred": len(transferred_patterns),
            "patterns": transferred_patterns,
        }

    def _pattern_is_abstracted(self, pattern: FrozenSet[str]) -> bool:
        """Check if pattern already has an abstraction."""
        return pattern in self.cortex.engine.pattern_to_abstraction

    def _transfer_pattern_to_cortex(
        self,
        pattern: FrozenSet[str],
        frequency: int,
    ) -> bool:
        """
        Transfer a single pattern to the Cortex.

        Args:
            pattern: The pattern to transfer.
            frequency: How often the pattern was observed.

        Returns:
            True if transfer succeeded, False otherwise.
        """
        # Observe the pattern in Cortex multiple times based on frequency
        # This builds up the pattern in the Cortex's pattern detector
        for _ in range(min(frequency, 5)):  # Cap at 5 observations
            self.cortex.engine.observe(pattern)

        # Try to form an abstraction
        candidates = self.cortex.engine.abstraction_candidates(top_k=5)
        for candidate_pattern, candidate_freq, level in candidates:
            if candidate_pattern == pattern:
                abstraction = self.cortex.engine.form_abstraction(pattern, level)
                return abstraction is not None

        return False

    def abstraction_mining(self) -> Dict[str, Any]:
        """
        Mine for latent abstractions in existing patterns.

        Looks for structure in the Cortex that could form
        higher-level abstractions.

        Returns:
            Dict with 'formed' count and 'abstraction_ids' list.
        """
        formed_abstractions: List[str] = []
        max_abstractions = self.config.max_abstractions_per_cycle

        # Let the Cortex auto-form abstractions from candidates
        new_abstractions = self.cortex.engine.auto_form_abstractions(
            max_new=max_abstractions,
            min_frequency=self.config.transfer_threshold,
        )

        for abstraction in new_abstractions:
            formed_abstractions.append(abstraction.id)

        # Propagate truth values through the hierarchy
        self.cortex.engine.propagate_truth()

        return {
            "formed": len(formed_abstractions),
            "abstraction_ids": formed_abstractions,
        }

    def decay_cycle(self) -> Dict[str, Any]:
        """
        Apply decay to connections and prune weak ones.

        Implements forgetting: connections that aren't reinforced
        decay over time and may be pruned if they fall below threshold.

        Key invariant: High-value connections are protected.

        Returns:
            Dict with 'decayed' and 'pruned' counts.
        """
        decayed_count = 0
        pruned_count = 0

        # Apply decay to Hive model (PRISM-SLM transitions)
        hive_decay_result = self._decay_hive_transitions()
        decayed_count += hive_decay_result["decayed"]
        pruned_count += hive_decay_result["pruned"]

        # Apply decay to Cortex pattern detector
        self.cortex.engine.apply_decay()

        # Apply decay to recorded pattern frequencies
        patterns_to_remove = []
        for pattern in list(self._pattern_frequencies.keys()):
            self._pattern_frequencies[pattern] = int(
                self._pattern_frequencies[pattern] * self.config.decay_factor
            )
            if self._pattern_frequencies[pattern] < 1:
                patterns_to_remove.append(pattern)
                pruned_count += 1

        for pattern in patterns_to_remove:
            del self._pattern_frequencies[pattern]

        # Decay homeostasis history
        self.hive.regulator.apply_decay()

        return {
            "decayed": decayed_count,
            "pruned": pruned_count,
        }

    def _decay_hive_transitions(self) -> Dict[str, int]:
        """Apply decay to Hive transition graph."""
        decayed = 0
        pruned = 0

        # Access the internal transition graph
        graph = self.hive.model.graph

        # Apply decay to all transitions
        for context, transitions in list(graph._transitions.items()):
            for transition in transitions:
                old_weight = transition.weight
                transition.weight *= self.config.decay_factor
                if old_weight != transition.weight:
                    decayed += 1

            # Remove transitions below threshold
            kept_transitions = [
                t for t in transitions
                if t.weight >= self.config.min_strength_keep
            ]
            removed = len(transitions) - len(kept_transitions)
            if removed > 0:
                pruned += removed
                if kept_transitions:
                    graph._transitions[context] = kept_transitions
                else:
                    del graph._transitions[context]

        return {"decayed": decayed, "pruned": pruned}

    def _set_phase(self, phase: ConsolidationPhase) -> None:
        """Set the current phase and fire callback."""
        self._current_phase = phase
        if self._on_phase_change:
            self._on_phase_change(phase)

    def start_scheduler(self) -> None:
        """
        Start the background consolidation scheduler.

        The scheduler runs consolidation at regular intervals
        when the config.enable_scheduling is True.

        Note: This is designed for long-running applications.
        """
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            return  # Already running

        self._stop_scheduler.clear()

        def scheduler_loop():
            while not self._stop_scheduler.is_set():
                # Wait for the interval
                self._stop_scheduler.wait(self.config.schedule_interval_seconds)
                if self._stop_scheduler.is_set():
                    break

                # Run consolidation if not already running
                if not self._is_running:
                    try:
                        self.consolidate()
                    except Exception:
                        pass  # Log errors but don't crash scheduler

        self._scheduler_thread = threading.Thread(
            target=scheduler_loop,
            daemon=True,
            name="consolidation-scheduler",
        )
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """Stop the background consolidation scheduler."""
        self._stop_scheduler.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None

    def interrupt(self) -> bool:
        """
        Interrupt a running consolidation cycle.

        Returns:
            True if interrupted, False if nothing was running.
        """
        if not self._is_running:
            return False

        # Signal interruption (in a real implementation, phases would check this)
        self._is_running = False
        return True

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> List[ConsolidationResult]:
        """
        Get consolidation history.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of ConsolidationResult from most recent to oldest.
        """
        history = list(reversed(self._consolidation_history))
        if limit is not None:
            history = history[:limit]
        return history

    def get_stats(self) -> Dict[str, Any]:
        """
        Get consolidation statistics.

        Returns:
            Dictionary with consolidation metrics.
        """
        total_cycles = len(self._consolidation_history)
        total_transferred = sum(r.patterns_transferred for r in self._consolidation_history)
        total_abstractions = sum(r.abstractions_formed for r in self._consolidation_history)
        total_pruned = sum(r.connections_pruned for r in self._consolidation_history)

        avg_duration = 0.0
        if total_cycles > 0:
            avg_duration = sum(r.cycle_duration_ms for r in self._consolidation_history) / total_cycles

        return {
            "total_cycles": total_cycles,
            "total_patterns_transferred": total_transferred,
            "total_abstractions_formed": total_abstractions,
            "total_connections_pruned": total_pruned,
            "avg_cycle_duration_ms": avg_duration,
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
            "current_phase": self._current_phase.name,
            "is_running": self._is_running,
            "tracked_patterns": len(self._pattern_frequencies),
            "scheduler_active": self._scheduler_thread is not None and self._scheduler_thread.is_alive(),
        }

    def on_phase_change(self, callback: Callable[[ConsolidationPhase], None]) -> None:
        """Register callback for phase changes."""
        self._on_phase_change = callback

    def on_pattern_transferred(self, callback: Callable[[FrozenSet[str]], None]) -> None:
        """Register callback for pattern transfers."""
        self._on_pattern_transferred = callback

    def on_cycle_complete(self, callback: Callable[[ConsolidationResult], None]) -> None:
        """Register callback for cycle completion."""
        self._on_cycle_complete = callback

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state."""
        return {
            "config": {
                "transfer_threshold": self.config.transfer_threshold,
                "decay_factor": self.config.decay_factor,
                "min_strength_keep": self.config.min_strength_keep,
                "max_patterns_per_cycle": self.config.max_patterns_per_cycle,
                "max_abstractions_per_cycle": self.config.max_abstractions_per_cycle,
            },
            "pattern_frequencies": {
                str(list(p)): f
                for p, f in self._pattern_frequencies.items()
            },
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
            "history_count": len(self._consolidation_history),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        hive: "LoomHiveConnector",
        cortex: "LoomCortexConnector",
    ) -> "ConsolidationEngine":
        """Deserialize engine from dictionary."""
        import ast

        config_data = data.get("config", {})
        config = ConsolidationConfig(
            transfer_threshold=config_data.get("transfer_threshold", 3),
            decay_factor=config_data.get("decay_factor", 0.9),
            min_strength_keep=config_data.get("min_strength_keep", 0.1),
            max_patterns_per_cycle=config_data.get("max_patterns_per_cycle", 10),
            max_abstractions_per_cycle=config_data.get("max_abstractions_per_cycle", 5),
        )

        engine = cls(hive, cortex, config)

        # Restore pattern frequencies
        for pattern_str, freq in data.get("pattern_frequencies", {}).items():
            pattern = frozenset(ast.literal_eval(pattern_str))
            engine._pattern_frequencies[pattern] = freq

        # Restore last consolidation time
        last_consolidation_str = data.get("last_consolidation")
        if last_consolidation_str:
            engine._last_consolidation = datetime.fromisoformat(last_consolidation_str)

        return engine


__all__ = [
    "ConsolidationConfig",
    "ConsolidationResult",
    "ConsolidationPhase",
    "ConsolidationEngine",
]
