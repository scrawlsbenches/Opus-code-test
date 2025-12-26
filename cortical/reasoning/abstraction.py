"""
Cortex Abstraction System for Woven Mind.

Implements hierarchical abstraction formation from repeated patterns.
Abstractions are higher-level concepts that emerge from observing
patterns of lower-level node co-activation.

Part of Sprint 3: Cortex Abstraction (Woven Mind + PRISM Marriage)

Key concepts:
- Abstraction: A higher-level concept formed from repeated patterns
- PatternDetector: Discovers repeated patterns in activation history
- AbstractionEngine: Manages abstraction formation and hierarchy

Example:
    >>> from cortical.reasoning.abstraction import AbstractionEngine
    >>> engine = AbstractionEngine()
    >>> engine.observe(frozenset(["neural", "network"]))
    >>> engine.observe(frozenset(["neural", "network"]))
    >>> engine.observe(frozenset(["neural", "network"]))
    >>> candidates = engine.abstraction_candidates()
    >>> # After 3 observations, pattern is eligible for abstraction
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any
from datetime import datetime
import uuid


@dataclass
class Abstraction:
    """
    A higher-level concept formed from repeated patterns.

    Abstractions emerge when patterns are observed frequently enough
    (minimum 3 times by default). They create hierarchical structure
    in the knowledge graph.

    Attributes:
        id: Unique identifier (format: A{level}-{hex})
        source_nodes: The pattern of nodes this abstracts
        level: Hierarchy level (0 = concrete, higher = more abstract)
        frequency: How often this pattern was observed
        formed_at: When the abstraction was created
        truth_value: Confidence/strength from PLN (0.0-1.0)
        strength: Combined measure of frequency and truth
    """
    id: str
    source_nodes: FrozenSet[str]
    level: int
    frequency: int
    formed_at: datetime = field(default_factory=datetime.now)
    truth_value: float = 0.5  # Default neutral truth value
    strength: float = 0.0

    def __post_init__(self):
        """Compute strength from frequency and truth."""
        self.strength = self._compute_strength()

    def _compute_strength(self) -> float:
        """Strength = frequency_factor * truth_value."""
        # Frequency factor: log-scaled, asymptotes toward 1.0
        import math
        freq_factor = 1.0 - (1.0 / (1.0 + math.log(1 + self.frequency)))
        return freq_factor * self.truth_value

    def update_truth(self, new_truth: float) -> None:
        """Update truth value and recompute strength."""
        self.truth_value = max(0.0, min(1.0, new_truth))
        self.strength = self._compute_strength()

    def observe(self) -> None:
        """Record another observation of this pattern."""
        self.frequency += 1
        self.strength = self._compute_strength()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_nodes": list(self.source_nodes),
            "level": self.level,
            "frequency": self.frequency,
            "formed_at": self.formed_at.isoformat(),
            "truth_value": self.truth_value,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Abstraction":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            source_nodes=frozenset(data["source_nodes"]),
            level=data["level"],
            frequency=data["frequency"],
            formed_at=datetime.fromisoformat(data["formed_at"]),
            truth_value=data.get("truth_value", 0.5),
        )


@dataclass
class PatternObservation:
    """Record of a pattern observation."""
    pattern: FrozenSet[str]
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


class PatternDetector:
    """
    Detects repeated patterns in activation history.

    Tracks which patterns occur together and how often, enabling
    the AbstractionEngine to identify candidates for abstraction.

    Attributes:
        min_pattern_size: Minimum nodes in a pattern (default 2)
        max_pattern_size: Maximum nodes in a pattern (default 10)
        min_frequency: Minimum observations to consider (default 3)
        decay_factor: How fast old observations decay (default 0.99)
    """

    def __init__(
        self,
        min_pattern_size: int = 2,
        max_pattern_size: int = 10,
        min_frequency: int = 3,
        decay_factor: float = 0.99,
    ):
        """Initialize the pattern detector.

        Args:
            min_pattern_size: Minimum nodes in a pattern.
            max_pattern_size: Maximum nodes in a pattern.
            min_frequency: How many times a pattern must be seen.
            decay_factor: Decay rate for pattern counts over time.
        """
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.min_frequency = min_frequency
        self.decay_factor = decay_factor

        # Pattern tracking
        self._pattern_counts: Dict[FrozenSet[str], float] = defaultdict(float)
        self._pattern_timestamps: Dict[FrozenSet[str], datetime] = {}
        self._observations: List[PatternObservation] = []

        # Statistics
        self._total_observations = 0

    def observe(
        self,
        active_nodes: FrozenSet[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[FrozenSet[str]]:
        """
        Observe a set of co-active nodes.

        Args:
            active_nodes: The nodes that are currently active together.
            context: Optional context for this observation.

        Returns:
            List of patterns that are now candidates (meet min_frequency).
        """
        self._total_observations += 1
        now = datetime.now()

        # Record observation
        self._observations.append(PatternObservation(
            pattern=active_nodes,
            timestamp=now,
            context=context,
        ))

        # Track the pattern if it meets size requirements
        candidates = []
        if self.min_pattern_size <= len(active_nodes) <= self.max_pattern_size:
            self._pattern_counts[active_nodes] += 1.0
            self._pattern_timestamps[active_nodes] = now

            # Check if it's now a candidate
            if self._pattern_counts[active_nodes] >= self.min_frequency:
                candidates.append(active_nodes)

        # Also track subsets (for finding smaller patterns within larger ones)
        if len(active_nodes) > self.min_pattern_size:
            for subset in self._generate_subsets(active_nodes):
                if self.min_pattern_size <= len(subset) <= self.max_pattern_size:
                    self._pattern_counts[subset] += 0.5  # Partial credit
                    self._pattern_timestamps[subset] = now

                    if self._pattern_counts[subset] >= self.min_frequency:
                        candidates.append(subset)

        return candidates

    def _generate_subsets(
        self,
        nodes: FrozenSet[str],
        min_size: Optional[int] = None,
    ) -> List[FrozenSet[str]]:
        """Generate subsets of a node set.

        Only generates subsets that are one size smaller to avoid
        combinatorial explosion.
        """
        if min_size is None:
            min_size = self.min_pattern_size

        result = []
        nodes_list = list(nodes)

        # Only remove one element at a time
        for i in range(len(nodes_list)):
            subset = frozenset(nodes_list[:i] + nodes_list[i+1:])
            if len(subset) >= min_size:
                result.append(subset)

        return result

    def get_candidates(
        self,
        min_frequency: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[FrozenSet[str], float]]:
        """
        Get patterns that are candidates for abstraction.

        Args:
            min_frequency: Override default minimum frequency.
            top_k: Only return top-k by frequency.

        Returns:
            List of (pattern, frequency) tuples, sorted by frequency desc.
        """
        threshold = min_frequency if min_frequency is not None else self.min_frequency

        candidates = [
            (pattern, count)
            for pattern, count in self._pattern_counts.items()
            if count >= threshold
        ]

        # Sort by frequency descending
        candidates.sort(key=lambda x: -x[1])

        if top_k is not None:
            candidates = candidates[:top_k]

        return candidates

    def apply_decay(self) -> None:
        """Apply temporal decay to all pattern counts."""
        for pattern in list(self._pattern_counts.keys()):
            self._pattern_counts[pattern] *= self.decay_factor
            # Remove patterns that have decayed below 1.0
            if self._pattern_counts[pattern] < 1.0:
                del self._pattern_counts[pattern]
                self._pattern_timestamps.pop(pattern, None)

    def get_pattern_frequency(self, pattern: FrozenSet[str]) -> float:
        """Get the current frequency count for a pattern."""
        return self._pattern_counts.get(pattern, 0.0)

    def clear(self) -> None:
        """Clear all tracked patterns."""
        self._pattern_counts.clear()
        self._pattern_timestamps.clear()
        self._observations.clear()
        self._total_observations = 0


class AbstractionEngine:
    """
    Manages abstraction formation and hierarchy.

    The engine observes patterns through its PatternDetector,
    identifies candidates for abstraction, and forms new
    Abstraction objects when patterns are stable enough.

    Attributes:
        detector: The PatternDetector for finding patterns.
        abstractions: Dictionary of formed abstractions.
        hierarchy_levels: List of abstraction IDs per level.
    """

    def __init__(
        self,
        min_pattern_size: int = 2,
        min_frequency: int = 3,
        max_levels: int = 5,
    ):
        """Initialize the abstraction engine.

        Args:
            min_pattern_size: Minimum nodes for abstraction (default 2).
            min_frequency: How many times pattern must be seen (default 3).
            max_levels: Maximum hierarchy depth (default 5).
        """
        self.detector = PatternDetector(
            min_pattern_size=min_pattern_size,
            min_frequency=min_frequency,
        )
        self.min_frequency = min_frequency
        self.max_levels = max_levels

        # Abstraction storage
        self.abstractions: Dict[str, Abstraction] = {}
        self.pattern_to_abstraction: Dict[FrozenSet[str], str] = {}
        self.hierarchy_levels: List[Set[str]] = [set() for _ in range(max_levels)]

        # Node level tracking (for hierarchical placement)
        self._node_levels: Dict[str, int] = defaultdict(int)

    def observe(
        self,
        active_nodes: FrozenSet[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Observe a set of co-active nodes.

        Args:
            active_nodes: The nodes that are currently active together.
            context: Optional context for this observation.

        Returns:
            List of abstraction IDs that were reinforced or created.
        """
        affected_abstractions = []

        # Check if this pattern already has an abstraction
        if active_nodes in self.pattern_to_abstraction:
            abstraction_id = self.pattern_to_abstraction[active_nodes]
            abstraction = self.abstractions[abstraction_id]
            abstraction.observe()
            affected_abstractions.append(abstraction_id)

        # Also observe in detector
        self.detector.observe(active_nodes, context)

        return affected_abstractions

    def abstraction_candidates(
        self,
        top_k: Optional[int] = 10,
    ) -> List[Tuple[FrozenSet[str], float, int]]:
        """
        Get patterns that are ready to become abstractions.

        Args:
            top_k: Maximum number of candidates to return.

        Returns:
            List of (pattern, frequency, suggested_level) tuples.
        """
        # Get candidate patterns from detector
        raw_candidates = self.detector.get_candidates(top_k=top_k)

        result = []
        for pattern, frequency in raw_candidates:
            # Skip if already abstracted
            if pattern in self.pattern_to_abstraction:
                continue

            # Determine appropriate level
            level = self._compute_level(pattern)

            result.append((pattern, frequency, level))

        return result

    def _compute_level(self, pattern: FrozenSet[str]) -> int:
        """Compute the appropriate hierarchy level for a pattern.

        Level is one above the highest-level component.
        """
        if not pattern:
            return 0

        max_component_level = 0
        for node in pattern:
            # Check if this node is itself an abstraction
            if node in self.abstractions:
                node_level = self.abstractions[node].level
            else:
                node_level = self._node_levels.get(node, 0)
            max_component_level = max(max_component_level, node_level)

        return min(max_component_level + 1, self.max_levels - 1)

    def form_abstraction(
        self,
        pattern: FrozenSet[str],
        level: Optional[int] = None,
        truth_value: float = 0.5,
    ) -> Optional[Abstraction]:
        """
        Form a new abstraction from a pattern.

        Args:
            pattern: The pattern to abstract.
            level: Optional explicit level (computed if not provided).
            truth_value: Initial PLN truth value.

        Returns:
            The new Abstraction, or None if already exists or invalid.
        """
        # Validate pattern size (must have ≥3 components as per guardrails)
        if len(pattern) < 2:
            return None

        # Check if already abstracted
        if pattern in self.pattern_to_abstraction:
            return None

        # Compute level if not provided
        if level is None:
            level = self._compute_level(pattern)

        # Generate ID
        abstraction_id = f"A{level}-{uuid.uuid4().hex[:8]}"

        # Get frequency from detector
        frequency = int(self.detector.get_pattern_frequency(pattern))

        # Create abstraction
        abstraction = Abstraction(
            id=abstraction_id,
            source_nodes=pattern,
            level=level,
            frequency=max(1, frequency),
            truth_value=truth_value,
        )

        # Register
        self.abstractions[abstraction_id] = abstraction
        self.pattern_to_abstraction[pattern] = abstraction_id
        self.hierarchy_levels[level].add(abstraction_id)

        # Update node levels
        self._node_levels[abstraction_id] = level

        return abstraction

    def auto_form_abstractions(
        self,
        max_new: int = 5,
        min_frequency: int = 3,
    ) -> List[Abstraction]:
        """
        Automatically form abstractions from top candidates.

        Args:
            max_new: Maximum new abstractions to form.
            min_frequency: Minimum frequency required.

        Returns:
            List of newly formed abstractions.
        """
        candidates = self.abstraction_candidates(top_k=max_new * 2)
        formed = []

        for pattern, frequency, level in candidates:
            if len(formed) >= max_new:
                break

            if frequency >= min_frequency:
                abstraction = self.form_abstraction(pattern, level)
                if abstraction:
                    formed.append(abstraction)

        return formed

    def get_abstraction(self, abstraction_id: str) -> Optional[Abstraction]:
        """Get an abstraction by ID."""
        return self.abstractions.get(abstraction_id)

    def get_abstractions_for_pattern(
        self,
        pattern: FrozenSet[str],
    ) -> List[Abstraction]:
        """Get all abstractions that contain this pattern as components."""
        result = []
        for abstraction in self.abstractions.values():
            if pattern.issubset(abstraction.source_nodes):
                result.append(abstraction)
        return result

    def get_level(self, level: int) -> List[Abstraction]:
        """Get all abstractions at a given hierarchy level."""
        if 0 <= level < len(self.hierarchy_levels):
            return [
                self.abstractions[aid]
                for aid in self.hierarchy_levels[level]
                if aid in self.abstractions
            ]
        return []

    def update_truth_values(
        self,
        truth_updates: Dict[str, float],
    ) -> None:
        """
        Update truth values for abstractions (PLN integration).

        Args:
            truth_updates: Dictionary mapping abstraction IDs to new truth values.
        """
        for abstraction_id, truth in truth_updates.items():
            if abstraction_id in self.abstractions:
                self.abstractions[abstraction_id].update_truth(truth)

    def propagate_truth(self) -> None:
        """
        Propagate truth values up the hierarchy.

        Lower-level abstractions' truth values influence higher-level ones.
        """
        # Process level by level, bottom-up
        for level in range(1, self.max_levels):
            for abstraction_id in self.hierarchy_levels[level]:
                if abstraction_id not in self.abstractions:
                    continue

                abstraction = self.abstractions[abstraction_id]

                # Find component abstractions
                component_truths = []
                for node_id in abstraction.source_nodes:
                    if node_id in self.abstractions:
                        component_truths.append(
                            self.abstractions[node_id].truth_value
                        )

                # If components have truth values, combine them
                if component_truths:
                    # Use minimum (conjunction-like) for conservative estimate
                    combined_truth = min(component_truths)
                    # Blend with existing (smooth update)
                    new_truth = 0.7 * abstraction.truth_value + 0.3 * combined_truth
                    abstraction.update_truth(new_truth)

    def apply_decay(self) -> None:
        """Apply temporal decay to pattern detector."""
        self.detector.apply_decay()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state to dictionary."""
        return {
            "abstractions": {
                aid: a.to_dict()
                for aid, a in self.abstractions.items()
            },
            "pattern_to_abstraction": {
                str(list(p)): aid
                for p, aid in self.pattern_to_abstraction.items()
            },
            "hierarchy_levels": [
                list(level) for level in self.hierarchy_levels
            ],
            "node_levels": dict(self._node_levels),
            "min_frequency": self.min_frequency,
            "max_levels": self.max_levels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AbstractionEngine":
        """Deserialize engine from dictionary."""
        engine = cls(
            min_frequency=data.get("min_frequency", 3),
            max_levels=data.get("max_levels", 5),
        )

        # Restore abstractions
        for aid, adata in data.get("abstractions", {}).items():
            abstraction = Abstraction.from_dict(adata)
            engine.abstractions[aid] = abstraction

        # Restore pattern mapping
        for pattern_str, aid in data.get("pattern_to_abstraction", {}).items():
            # Convert string back to frozenset
            import ast
            pattern = frozenset(ast.literal_eval(pattern_str))
            engine.pattern_to_abstraction[pattern] = aid

        # Restore hierarchy
        for level, ids in enumerate(data.get("hierarchy_levels", [])):
            if level < len(engine.hierarchy_levels):
                engine.hierarchy_levels[level] = set(ids)

        # Restore node levels
        engine._node_levels = defaultdict(int, data.get("node_levels", {}))

        return engine
