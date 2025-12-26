"""
Loom-Cortex Connector: Integrating the Loom with Enhanced Cortex.

Provides the connection between the Loom (mode switching) and the
enhanced Cortex (AbstractionEngine for SLOW mode deliberative processing).

Part of Sprint 4: The Loom Weaves (T4.2)
Part of the Woven Mind + PRISM Marriage project.

Key functionality:
- process_slow(): Process input through the Cortex for SLOW mode
- query_abstractions(): Query abstractions by source nodes
- get_abstraction_activations(): Get activation levels for Loom integration
- analyze(): Deep analysis using PLN when available

Example:
    >>> from cortical.reasoning.loom_cortex import LoomCortexConnector
    >>> from cortical.reasoning.loom import Loom
    >>>
    >>> connector = LoomCortexConnector(min_frequency=3)
    >>> # Build patterns through repeated observation
    >>> for _ in range(5):
    ...     connector.process_slow(["neural", "networks"])
    >>>
    >>> # Query abstractions
    >>> abstractions = connector.get_abstractions()
    >>> for a in abstractions:
    ...     print(f"{a.id}: {a.source_nodes}")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .abstraction import Abstraction, AbstractionEngine
from .abstraction_pln import AbstractionPLNBridge


@dataclass
class LoomCortexConfig:
    """Configuration for Loom-Cortex connector.

    Attributes:
        min_frequency: Minimum observations before forming abstraction.
        max_levels: Maximum abstraction hierarchy depth.
        auto_form: Whether to automatically form abstractions.
    """
    min_frequency: int = 3
    max_levels: int = 5
    auto_form: bool = True


class LoomCortexConnector:
    """
    Connects the Loom to the enhanced Cortex (AbstractionEngine).

    Wraps AbstractionEngine to provide:
    - SLOW mode processing through pattern detection and abstraction
    - Abstraction querying and retrieval
    - Optional PLN integration for deeper analysis
    - Deliberative processing capabilities

    Attributes:
        engine: The AbstractionEngine for pattern detection
        pln_bridge: Optional PLN bridge for probabilistic inference
        min_frequency: Minimum observations for abstraction formation
    """

    def __init__(
        self,
        engine: Optional[AbstractionEngine] = None,
        pln_bridge: Optional[AbstractionPLNBridge] = None,
        min_frequency: int = 3,
        config: Optional[LoomCortexConfig] = None,
    ):
        """
        Initialize the connector.

        Args:
            engine: AbstractionEngine instance. Creates default if not provided.
            pln_bridge: Optional PLN bridge for probabilistic inference.
            min_frequency: Minimum observations for abstraction (default 3).
            config: Configuration object (overrides min_frequency).
        """
        config = config or LoomCortexConfig(min_frequency=min_frequency)
        self.min_frequency = config.min_frequency
        self.config = config

        # Initialize engine
        self.engine = engine or AbstractionEngine(
            min_frequency=config.min_frequency,
            max_levels=config.max_levels,
        )

        # Optional PLN bridge
        self.pln_bridge = pln_bridge

        # Activation tracking for Loom integration
        self._abstraction_activations: Dict[str, float] = {}
        self._last_active: Set[str] = set()

    def process_slow(self, tokens: List[str]) -> Set[str]:
        """
        Process input through the Cortex (SLOW mode).

        Observes the pattern, potentially forms abstractions,
        and returns which abstractions became active.

        Args:
            tokens: List of tokens representing the current input.

        Returns:
            Set of abstraction IDs that are active.
        """
        # Convert to frozenset for pattern observation
        pattern = frozenset(t.lower() for t in tokens)

        # Observe the pattern
        affected = self.engine.observe(pattern)

        # Auto-form abstractions if configured
        if self.config.auto_form:
            new_abstractions = self.engine.auto_form_abstractions(
                max_new=3,
                min_frequency=self.min_frequency,
            )

            # Register new abstractions with PLN if available
            if self.pln_bridge:
                for abstraction in new_abstractions:
                    self.pln_bridge.register_abstraction(abstraction)

            affected.extend([a.id for a in new_abstractions])

        # Update activations
        active = set(affected)

        # Also activate any existing abstractions that match this pattern
        for abstraction_id, abstraction in self.engine.abstractions.items():
            if abstraction.source_nodes == pattern:
                active.add(abstraction_id)
                self._abstraction_activations[abstraction_id] = 1.0
            elif abstraction.source_nodes.issubset(pattern):
                active.add(abstraction_id)
                # Partial match gets lower activation
                overlap = len(abstraction.source_nodes) / len(pattern)
                self._abstraction_activations[abstraction_id] = overlap

        self._last_active = active
        return active

    def get_pattern_stats(self) -> Dict[str, Any]:
        """
        Get statistics about observed patterns.

        Returns:
            Dictionary with pattern observation statistics.
        """
        return {
            "total_observations": self.engine.detector._total_observations,
            "unique_patterns": len(self.engine.detector._pattern_counts),
            "total_abstractions": len(self.engine.abstractions),
        }

    def get_abstractions(self) -> List[Abstraction]:
        """
        Get all formed abstractions.

        Returns:
            List of all abstractions in the engine.
        """
        return list(self.engine.abstractions.values())

    def get_abstraction(self, abstraction_id: str) -> Optional[Abstraction]:
        """
        Get a specific abstraction by ID.

        Args:
            abstraction_id: The ID of the abstraction to retrieve.

        Returns:
            The Abstraction or None if not found.
        """
        return self.engine.get_abstraction(abstraction_id)

    def query_abstractions(self, tokens: List[str]) -> List[Abstraction]:
        """
        Query abstractions that involve any of the given tokens.

        Args:
            tokens: List of tokens to search for.

        Returns:
            List of abstractions containing any of the tokens.
        """
        token_set = frozenset(t.lower() for t in tokens)
        results = []

        for abstraction in self.engine.abstractions.values():
            if token_set & abstraction.source_nodes:  # Intersection
                results.append(abstraction)

        # Sort by relevance (overlap size)
        results.sort(
            key=lambda a: len(token_set & a.source_nodes),
            reverse=True
        )

        return results

    def get_related_abstractions(self, token: str) -> List[Abstraction]:
        """
        Find abstractions related to a specific token.

        Args:
            token: The token to find relations for.

        Returns:
            List of abstractions containing this token.
        """
        token_lower = token.lower()
        results = []

        for abstraction in self.engine.abstractions.values():
            if token_lower in abstraction.source_nodes:
                results.append(abstraction)

        # Sort by strength
        results.sort(key=lambda a: a.strength, reverse=True)

        return results

    def get_abstraction_activations(self) -> Dict[str, float]:
        """
        Get current activation levels for all abstractions.

        Returns:
            Dictionary mapping abstraction IDs to activation levels.
        """
        return dict(self._abstraction_activations)

    def analyze(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Perform deep analysis on input using PLN when available.

        Args:
            tokens: List of tokens to analyze.

        Returns:
            Dictionary with analysis results including abstractions and truth values.
        """
        result = {
            "abstractions": [],
            "truth_values": {},
            "pattern": frozenset(t.lower() for t in tokens),
        }

        # Find matching abstractions
        matching = self.query_abstractions(tokens)
        result["abstractions"] = [
            {
                "id": a.id,
                "source_nodes": list(a.source_nodes),
                "level": a.level,
                "strength": a.strength,
            }
            for a in matching
        ]

        # Get truth values from PLN if available
        if self.pln_bridge:
            for abstraction in matching:
                tv = self.pln_bridge.infer_abstraction_truth(abstraction.id)
                if tv:
                    result["truth_values"][abstraction.id] = {
                        "strength": tv.strength,
                        "confidence": tv.confidence,
                    }

        return result

    def explain_abstraction(self, abstraction_id: str) -> Dict[str, Any]:
        """
        Explain why an abstraction exists.

        Args:
            abstraction_id: The ID of the abstraction to explain.

        Returns:
            Dictionary with explanation details.
        """
        abstraction = self.engine.get_abstraction(abstraction_id)
        if not abstraction:
            return {"error": f"Abstraction {abstraction_id} not found"}

        explanation = {
            "id": abstraction_id,
            "source_nodes": list(abstraction.source_nodes),
            "frequency": abstraction.frequency,
            "level": abstraction.level,
            "truth_value": abstraction.truth_value,
            "strength": abstraction.strength,
            "formed_at": abstraction.formed_at.isoformat(),
        }

        # Add PLN truth if available
        if self.pln_bridge:
            tv = self.pln_bridge.infer_abstraction_truth(abstraction_id)
            if tv:
                explanation["pln_truth"] = {
                    "strength": tv.strength,
                    "confidence": tv.confidence,
                }

        return explanation

    def build_meta_abstractions(self) -> List[Abstraction]:
        """
        Build meta-abstractions from existing abstractions.

        Meta-abstractions are higher-level patterns formed from
        co-occurring lower-level abstractions.

        Returns:
            List of newly formed meta-abstractions.
        """
        # Get all level-1 abstractions
        level_1 = self.engine.get_level(1)

        # Look for co-occurring abstractions
        formed = []

        # Check which abstractions have overlapping source nodes
        for i, a1 in enumerate(level_1):
            for a2 in level_1[i + 1:]:
                # If they share a node, they might form a meta-abstraction
                if a1.source_nodes & a2.source_nodes:
                    meta_pattern = frozenset([a1.id, a2.id])
                    meta = self.engine.form_abstraction(meta_pattern, level=2)
                    if meta:
                        formed.append(meta)
                        if self.pln_bridge:
                            self.pln_bridge.register_abstraction(meta)

        return formed

    def get_abstraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about abstractions.

        Returns:
            Dictionary with abstraction statistics.
        """
        level_counts = {}
        for level in range(self.config.max_levels):
            abstractions = self.engine.get_level(level)
            if abstractions:
                level_counts[f"level_{level}"] = len(abstractions)

        return {
            "total_abstractions": len(self.engine.abstractions),
            "by_level": level_counts,
            "total_patterns_observed": self.engine.detector._total_observations,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize connector state.

        Returns:
            Dictionary representation.
        """
        return {
            "min_frequency": self.min_frequency,
            "config": {
                "min_frequency": self.config.min_frequency,
                "max_levels": self.config.max_levels,
                "auto_form": self.config.auto_form,
            },
            "engine_state": self.engine.to_dict(),
            "abstraction_activations": self._abstraction_activations,
            "last_active": list(self._last_active),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoomCortexConnector":
        """
        Deserialize connector from dictionary.

        Args:
            data: Serialized connector data.

        Returns:
            Reconstructed LoomCortexConnector.
        """
        # Reconstruct config
        config_data = data.get("config", {})
        config = LoomCortexConfig(
            min_frequency=config_data.get("min_frequency", 3),
            max_levels=config_data.get("max_levels", 5),
            auto_form=config_data.get("auto_form", True),
        )

        # Reconstruct engine
        engine = AbstractionEngine.from_dict(data.get("engine_state", {}))

        # Create connector
        connector = cls(
            engine=engine,
            config=config,
        )

        # Restore activations
        connector._abstraction_activations = data.get("abstraction_activations", {})
        connector._last_active = set(data.get("last_active", []))

        return connector
