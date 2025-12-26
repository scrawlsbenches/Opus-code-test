"""
Abstraction-PLN Bridge: Wiring abstractions to PRISM-PLN.

Connects the AbstractionEngine with PLNReasoner for:
1. Converting abstractions to PLN atoms
2. Inferring abstraction truth from component evidence
3. Synchronizing truth values bidirectionally
4. Creating logical rules from abstraction relationships

Part of Sprint 3: Cortex Abstraction (T3.5)
Part of the Woven Mind + PRISM Marriage project.

Example:
    >>> from cortical.reasoning.abstraction import AbstractionEngine
    >>> from cortical.reasoning.prism_pln import PLNReasoner
    >>> from cortical.reasoning.abstraction_pln import AbstractionPLNBridge
    >>>
    >>> engine = AbstractionEngine()
    >>> reasoner = PLNReasoner()
    >>> bridge = AbstractionPLNBridge(reasoner, engine=engine)
    >>>
    >>> # Form abstraction and register with PLN
    >>> abstraction = engine.form_abstraction(frozenset(["neural", "network"]))
    >>> bridge.register_abstraction(abstraction)
    >>>
    >>> # Use PLN inference for truth
    >>> inferred = bridge.infer_abstraction_truth(abstraction.id)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .abstraction import Abstraction, AbstractionEngine
from .prism_pln import PLNReasoner, TruthValue, pln_and, SynapticTruthValue


@dataclass
class CoactivationRecord:
    """Record of abstraction co-occurrences."""
    abstraction_ids: FrozenSet[str]
    count: int = 0
    last_strength: float = 0.0


class AbstractionPLNBridge:
    """
    Bridge between AbstractionEngine and PLNReasoner.

    Enables bidirectional flow of truth values between the abstraction
    hierarchy and PLN's probabilistic logic network.

    Attributes:
        reasoner: The PLNReasoner for probabilistic inference
        engine: Optional AbstractionEngine for synchronized updates
    """

    def __init__(
        self,
        reasoner: PLNReasoner,
        engine: Optional[AbstractionEngine] = None,
    ):
        """
        Initialize the bridge.

        Args:
            reasoner: PLNReasoner instance for inference
            engine: Optional AbstractionEngine for bidirectional sync
        """
        self.reasoner = reasoner
        self.engine = engine

        # Track registered abstractions
        self._registered: Set[str] = set()

        # Track abstraction -> PLN atom mapping
        self._abstraction_to_atom: Dict[str, str] = {}

        # Store abstraction source nodes for inference (when engine not available)
        self._abstraction_sources: Dict[str, FrozenSet[str]] = {}

        # Track coactivations for association rules
        self._coactivation_counts: Dict[FrozenSet[str], int] = defaultdict(int)

    def register_abstraction(self, abstraction: Abstraction) -> None:
        """
        Register an abstraction with PLN.

        Creates:
        1. PLN atom for the abstraction itself
        2. Implications from source nodes to the abstraction

        Args:
            abstraction: The abstraction to register
        """
        if abstraction.id in self._registered:
            return

        # Create PLN atom for the abstraction
        self.reasoner.graph.add_atom(
            abstraction.id,
            TruthValue(
                strength=abstraction.truth_value,
                confidence=self._frequency_to_confidence(abstraction.frequency),
            ),
        )

        # Create atoms for source nodes if they don't exist
        for node in abstraction.source_nodes:
            if self.reasoner.graph.get_atom(node) is None:
                # Default truth value for source nodes (observed)
                self.reasoner.graph.add_atom(
                    node,
                    TruthValue(strength=0.8, confidence=0.5),
                )

        # Create implication: (source_nodes conjunction) → abstraction
        # We represent this as a rule from a synthetic conjunction node
        conjunction_id = self._make_conjunction_id(abstraction.source_nodes)

        # Add conjunction atom
        self.reasoner.graph.add_atom(
            conjunction_id,
            TruthValue(strength=0.8, confidence=0.6),
        )

        # Add implication: conjunction → abstraction
        self.reasoner.graph.add_implication(
            conjunction_id,
            abstraction.id,
            TruthValue(strength=0.9, confidence=0.8),
        )

        self._registered.add(abstraction.id)
        self._abstraction_to_atom[abstraction.id] = abstraction.id
        self._abstraction_sources[abstraction.id] = abstraction.source_nodes

    def _make_conjunction_id(self, nodes: FrozenSet[str]) -> str:
        """Create a unique ID for a conjunction of nodes."""
        sorted_nodes = sorted(nodes)
        return f"AND({','.join(sorted_nodes)})"

    def _frequency_to_confidence(self, frequency: int) -> float:
        """Convert frequency to confidence (asymptotic toward 1.0)."""
        # More observations = more confidence
        return frequency / (frequency + 5.0)

    def infer_abstraction_truth(self, abstraction_id: str) -> Optional[TruthValue]:
        """
        Infer truth value for an abstraction using PLN.

        Combines evidence from source nodes using PLN inference rules.

        Args:
            abstraction_id: The abstraction ID to infer

        Returns:
            Inferred TruthValue or None if inference not possible
        """
        if abstraction_id not in self._registered:
            return None

        # Get source nodes - try engine first, then local cache
        source_nodes: Optional[FrozenSet[str]] = None
        if self.engine:
            abstraction = self.engine.get_abstraction(abstraction_id)
            if abstraction:
                source_nodes = abstraction.source_nodes

        if source_nodes is None:
            # Use locally stored source nodes
            source_nodes = self._abstraction_sources.get(abstraction_id)

        if source_nodes is None:
            # Fall back to PLN query
            return self.reasoner.query(abstraction_id)

        # Gather truth values for source nodes
        component_tvs: List[TruthValue] = []
        for node in source_nodes:
            # Try PLN first
            node_tv = self.reasoner.graph.get_truth_value(node)
            if node_tv is None:
                # Default for unknown nodes
                node_tv = TruthValue(strength=0.5, confidence=0.1)
            component_tvs.append(node_tv)

        if not component_tvs:
            return None

        # Combine using conjunction (AND) - weakest link principle
        result = component_tvs[0]
        for tv in component_tvs[1:]:
            result = pln_and(result, tv)

        return result

    def sync_truth_to_engine(self) -> None:
        """
        Synchronize truth values from PLN to AbstractionEngine.

        Updates abstraction truth_value fields based on PLN atoms.
        """
        if self.engine is None:
            return

        updates: Dict[str, float] = {}

        for abstraction_id in self._registered:
            atom = self.reasoner.graph.get_atom(abstraction_id)
            if atom:
                updates[abstraction_id] = atom.truth_value.strength

        if updates:
            self.engine.update_truth_values(updates)

    def sync_truth_from_engine(self) -> None:
        """
        Synchronize truth values from AbstractionEngine to PLN.

        Updates PLN atoms based on abstraction truth_value fields.
        """
        if self.engine is None:
            return

        for abstraction_id in self._registered:
            abstraction = self.engine.get_abstraction(abstraction_id)
            if abstraction:
                atom = self.reasoner.graph.get_atom(abstraction_id)
                if atom:
                    # Update atom's truth value
                    atom.truth_value = TruthValue(
                        strength=abstraction.truth_value,
                        confidence=self._frequency_to_confidence(abstraction.frequency),
                    )

    def propagate_hierarchical_truth(self) -> None:
        """
        Propagate truth values through the abstraction hierarchy.

        Lower-level abstraction truths influence higher-level ones.
        Processes levels in order to ensure correct propagation.
        """
        if self.engine is None:
            return

        # First sync engine to PLN
        self.sync_truth_from_engine()

        # Group abstractions by level for ordered processing
        by_level: Dict[int, List[str]] = defaultdict(list)
        for abstraction_id in self._registered:
            abstraction = self.engine.get_abstraction(abstraction_id)
            if abstraction:
                by_level[abstraction.level].append(abstraction_id)

        # Process from lowest to highest level
        # Only higher-level abstractions (level > 1) should be updated based on components
        # Level 1 abstractions have their truth set directly
        for level in sorted(by_level.keys()):
            if level < 2:
                # Skip level 1 - these have truth set from observations
                continue

            for abstraction_id in by_level[level]:
                abstraction = self.engine.get_abstraction(abstraction_id)
                if abstraction is None:
                    continue

                # For meta-abstractions, check if source nodes are abstractions
                component_truths: List[TruthValue] = []
                for node in abstraction.source_nodes:
                    if node in self._registered:
                        # Source is another abstraction - use its truth
                        node_tv = self.reasoner.graph.get_truth_value(node)
                        if node_tv:
                            component_truths.append(node_tv)

                if component_truths:
                    # Combine using minimum (conservative) instead of AND (multiplicative)
                    # This prevents steep truth decay through hierarchy
                    min_strength = min(tv.strength for tv in component_truths)
                    avg_confidence = sum(tv.confidence for tv in component_truths) / len(component_truths)

                    # Update abstraction's atom
                    atom = self.reasoner.graph.get_atom(abstraction_id)
                    if atom:
                        # Blend: 40% own + 60% from components
                        new_strength = 0.4 * atom.truth_value.strength + 0.6 * min_strength
                        atom.truth_value = TruthValue(
                            strength=new_strength,
                            confidence=max(atom.truth_value.confidence, avg_confidence),
                        )

        # Sync back to engine
        self.sync_truth_to_engine()

    def create_subsumption_rules(self) -> None:
        """
        Create PLN implication rules for subsumption relationships.

        If abstraction A's source nodes are a subset of B's,
        then B → A (having B implies having A).
        """
        if self.engine is None:
            return

        abstractions = list(self.engine.abstractions.values())

        for a in abstractions:
            for b in abstractions:
                if a.id == b.id:
                    continue

                # Check if a's sources are a subset of b's
                if a.source_nodes.issubset(b.source_nodes):
                    # b → a (if you have the superset, you have the subset)
                    existing = self.reasoner.graph.get_implication(b.id, a.id)
                    if existing is None:
                        # Calculate strength based on subset relationship
                        overlap = len(a.source_nodes) / len(b.source_nodes)
                        self.reasoner.graph.add_implication(
                            b.id,
                            a.id,
                            TruthValue(strength=0.9 * overlap, confidence=0.8),
                        )

    def observe_coactivation(self, abstraction_ids: List[str]) -> None:
        """
        Record co-activation of abstractions.

        Used to build association rules based on co-occurrence.

        Args:
            abstraction_ids: List of abstraction IDs that were active together
        """
        if len(abstraction_ids) < 2:
            return

        # Record pairwise coactivations
        for i, id1 in enumerate(abstraction_ids):
            for id2 in abstraction_ids[i + 1:]:
                pair = frozenset([id1, id2])
                self._coactivation_counts[pair] += 1

    def get_coactivation_count(self, id1: str, id2: str) -> int:
        """Get the coactivation count for a pair of abstractions."""
        pair = frozenset([id1, id2])
        return self._coactivation_counts.get(pair, 0)

    def create_association_rules(self, min_cooccurrence: int = 3) -> None:
        """
        Create PLN rules from frequent coactivations.

        Args:
            min_cooccurrence: Minimum coactivations to create a rule
        """
        for pair, count in self._coactivation_counts.items():
            if count < min_cooccurrence:
                continue

            ids = list(pair)
            id1, id2 = ids[0], ids[1]

            # Create bidirectional weak implications
            # Higher count = higher strength
            strength = min(0.9, 0.5 + (count / 20.0))
            confidence = min(0.8, 0.3 + (count / 15.0))

            # id1 → id2
            if self.reasoner.graph.get_implication(id1, id2) is None:
                self.reasoner.graph.add_implication(
                    id1,
                    id2,
                    TruthValue(strength=strength, confidence=confidence),
                )

            # id2 → id1
            if self.reasoner.graph.get_implication(id2, id1) is None:
                self.reasoner.graph.add_implication(
                    id2,
                    id1,
                    TruthValue(strength=strength, confidence=confidence),
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize bridge state to dictionary.

        Returns:
            Dictionary with registered abstractions and coactivation counts
        """
        return {
            "registered_abstractions": list(self._registered),
            "coactivation_counts": {
                ":".join(sorted(pair)): count
                for pair, count in self._coactivation_counts.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        reasoner: PLNReasoner,
        engine: Optional[AbstractionEngine] = None,
    ) -> "AbstractionPLNBridge":
        """
        Deserialize bridge from dictionary.

        Args:
            data: Serialized bridge data
            reasoner: PLNReasoner instance
            engine: Optional AbstractionEngine

        Returns:
            Reconstructed AbstractionPLNBridge
        """
        bridge = cls(reasoner, engine=engine)

        # Restore registered abstractions
        bridge._registered = set(data.get("registered_abstractions", []))

        # Restore coactivation counts
        for pair_str, count in data.get("coactivation_counts", {}).items():
            ids = pair_str.split(":")
            if len(ids) == 2:
                bridge._coactivation_counts[frozenset(ids)] = count

        return bridge
