"""
Tests for Abstraction-PLN integration.

TDD: Write tests first, then implement the integration.

The integration should:
1. Create PLN atoms from abstractions
2. Infer truth values for abstractions using PLN rules
3. Propagate truth values between abstraction hierarchy and PLN
"""

import pytest
from datetime import datetime
from typing import Set, FrozenSet

from cortical.reasoning.abstraction import (
    Abstraction,
    AbstractionEngine,
    PatternDetector,
)
from cortical.reasoning.prism_pln import (
    TruthValue,
    PLNReasoner,
    PLNGraph,
    pln_and,
    pln_or,
)


class TestAbstractionToPLNConversion:
    """Test converting abstractions to PLN atoms."""

    def test_abstraction_to_pln_atom(self):
        """An abstraction should create a corresponding PLN atom."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        # Create abstraction
        abstraction = Abstraction(
            id="A1-test123",
            source_nodes=frozenset(["neural", "network"]),
            level=1,
            frequency=5,
            truth_value=0.8,
        )

        # Create bridge
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)

        # Register abstraction
        bridge.register_abstraction(abstraction)

        # Verify PLN atom was created
        atom = reasoner.graph.get_atom("A1-test123")
        assert atom is not None
        assert atom.truth_value.strength == 0.8

    def test_abstraction_source_nodes_create_implications(self):
        """Source nodes should have implications TO the abstraction."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        abstraction = Abstraction(
            id="A1-concept",
            source_nodes=frozenset(["neural", "network"]),
            level=1,
            frequency=5,
            truth_value=0.8,
        )

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)
        bridge.register_abstraction(abstraction)

        # Verify implications exist: neural ∧ network → A1-concept
        # This is represented as an implication from the conjunction
        links = reasoner.graph.find_implications_to("A1-concept")
        assert len(links) >= 1

    def test_abstraction_engine_with_pln(self):
        """AbstractionEngine should integrate with PLN through bridge."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine(min_frequency=3)
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)

        # Observe patterns
        for _ in range(4):
            engine.observe(frozenset(["machine", "learning"]))

        # Form abstractions
        abstractions = engine.auto_form_abstractions(max_new=5)
        assert len(abstractions) >= 1

        # Register with PLN
        for a in abstractions:
            bridge.register_abstraction(a)

        # Verify in PLN
        assert reasoner.graph.atom_count >= len(abstractions)


class TestPLNInferenceForAbstractions:
    """Test using PLN inference to determine abstraction truth."""

    def test_infer_abstraction_from_components(self):
        """If source nodes are true, abstraction should be inferred true."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        abstraction = Abstraction(
            id="A1-concept",
            source_nodes=frozenset(["neural", "network"]),
            level=1,
            frequency=5,
            truth_value=0.5,  # Start uncertain
        )

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)

        # Assert source nodes as facts
        reasoner.assert_fact("neural", strength=0.95, confidence=0.9)
        reasoner.assert_fact("network", strength=0.90, confidence=0.85)

        # Register abstraction
        bridge.register_abstraction(abstraction)

        # Infer abstraction truth from components
        inferred_tv = bridge.infer_abstraction_truth(abstraction.id)

        # Should have high strength since components are true
        assert inferred_tv is not None
        assert inferred_tv.strength > 0.7

    def test_weak_component_weakens_abstraction(self):
        """Weak truth in one component should weaken the abstraction."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        abstraction = Abstraction(
            id="A1-concept",
            source_nodes=frozenset(["known", "unknown"]),
            level=1,
            frequency=5,
            truth_value=0.5,
        )

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)

        # One strong, one weak
        reasoner.assert_fact("known", strength=0.95, confidence=0.9)
        reasoner.assert_fact("unknown", strength=0.3, confidence=0.8)

        bridge.register_abstraction(abstraction)
        inferred_tv = bridge.infer_abstraction_truth(abstraction.id)

        # Weakest link should pull down the truth
        assert inferred_tv is not None
        assert inferred_tv.strength < 0.7


class TestTruthValueSynchronization:
    """Test synchronizing truth values between abstraction and PLN."""

    def test_update_abstraction_from_pln(self):
        """PLN truth updates should propagate to abstraction."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine()
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner, engine=engine)

        # Create abstraction
        abstraction = engine.form_abstraction(
            frozenset(["test", "pattern"]),
            truth_value=0.5,
        )
        assert abstraction is not None

        # Register with PLN
        bridge.register_abstraction(abstraction)

        # Update truth in PLN
        reasoner.assert_fact(abstraction.id, strength=0.85, confidence=0.9)

        # Sync back to abstraction engine
        bridge.sync_truth_to_engine()

        # Verify abstraction was updated
        updated = engine.get_abstraction(abstraction.id)
        assert updated is not None
        assert abs(updated.truth_value - 0.85) < 0.1

    def test_update_pln_from_abstraction(self):
        """Abstraction truth updates should propagate to PLN."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine()
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner, engine=engine)

        # Create and register abstraction
        abstraction = engine.form_abstraction(
            frozenset(["test", "pattern"]),
            truth_value=0.6,
        )
        bridge.register_abstraction(abstraction)

        # Update in engine
        engine.update_truth_values({abstraction.id: 0.9})

        # Sync to PLN
        bridge.sync_truth_from_engine()

        # Verify PLN was updated
        atom = reasoner.graph.get_atom(abstraction.id)
        assert atom is not None
        assert abs(atom.truth_value.strength - 0.9) < 0.1


class TestHierarchicalPropagation:
    """Test truth propagation through abstraction hierarchy."""

    def test_level1_updates_propagate_to_level2(self):
        """Level 1 truth changes should affect Level 2 meta-abstractions."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine()
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner, engine=engine)

        # Create Level 1 abstractions
        a1 = engine.form_abstraction(
            frozenset(["neural", "network"]),
            level=1,
            truth_value=0.8,
        )
        a2 = engine.form_abstraction(
            frozenset(["deep", "learning"]),
            level=1,
            truth_value=0.7,
        )

        # Create Level 2 meta-abstraction
        meta = engine.form_abstraction(
            frozenset([a1.id, a2.id]),
            level=2,
            truth_value=0.5,
        )

        # Register all with PLN
        for a in [a1, a2, meta]:
            bridge.register_abstraction(a)

        # Propagate truth through hierarchy
        bridge.propagate_hierarchical_truth()

        # Meta should reflect component truths
        inferred = bridge.infer_abstraction_truth(meta.id)
        assert inferred is not None
        # Meta truth should be influenced by components
        assert inferred.strength > 0.5


class TestPLNRulesForAbstractions:
    """Test creating PLN rules from abstraction relationships."""

    def test_subsumption_creates_implication(self):
        """If A subsumes B, create A → B implication."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine()
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner, engine=engine)

        # General concept
        general = engine.form_abstraction(
            frozenset(["machine", "learning"]),
            level=1,
            truth_value=0.9,
        )

        # More specific concept (superset of nodes)
        specific = engine.form_abstraction(
            frozenset(["machine", "learning", "deep"]),
            level=1,
            truth_value=0.85,
        )

        bridge.register_abstraction(general)
        bridge.register_abstraction(specific)

        # Create subsumption rules
        bridge.create_subsumption_rules()

        # The more specific should imply the more general
        # (if you have ML + deep, you have ML)
        links = reasoner.graph.find_implications_from(specific.id)
        target_ids = [link.consequent for link in links]
        assert general.id in target_ids

    def test_co_occurrence_creates_association(self):
        """Frequently co-occurring abstractions should create associations."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine()
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner, engine=engine)

        a1 = engine.form_abstraction(frozenset(["neural", "network"]), truth_value=0.8)
        a2 = engine.form_abstraction(frozenset(["back", "propagation"]), truth_value=0.8)

        bridge.register_abstraction(a1)
        bridge.register_abstraction(a2)

        # Record co-occurrences
        for _ in range(5):
            bridge.observe_coactivation([a1.id, a2.id])

        # Create association rules based on co-occurrence
        bridge.create_association_rules(min_cooccurrence=3)

        # Should have mutual implications
        links_from_a1 = reasoner.graph.find_implications_from(a1.id)
        links_from_a2 = reasoner.graph.find_implications_from(a2.id)

        assert any(link.consequent == a2.id for link in links_from_a1) or \
               any(link.consequent == a1.id for link in links_from_a2)


class TestAbstractionPLNBridgeAPI:
    """Test the complete AbstractionPLNBridge API."""

    def test_bridge_initialization(self):
        """Bridge should initialize with reasoner."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)

        assert bridge.reasoner is reasoner
        assert bridge.engine is None

    def test_bridge_with_engine(self):
        """Bridge should accept optional engine."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        engine = AbstractionEngine()
        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner, engine=engine)

        assert bridge.engine is engine

    def test_bridge_to_dict(self):
        """Bridge state should be serializable."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)

        abstraction = Abstraction(
            id="A1-test",
            source_nodes=frozenset(["a", "b"]),
            level=1,
            frequency=5,
            truth_value=0.8,
        )
        bridge.register_abstraction(abstraction)

        # Should be able to serialize
        data = bridge.to_dict()
        assert "registered_abstractions" in data
        assert "A1-test" in data["registered_abstractions"]

    def test_bridge_from_dict(self):
        """Bridge should be deserializable."""
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        data = {
            "registered_abstractions": ["A1-test", "A1-other"],
            "coactivation_counts": {"A1-test:A1-other": 5},
        }

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge.from_dict(data, reasoner)

        assert "A1-test" in bridge._registered
        assert bridge.get_coactivation_count("A1-test", "A1-other") == 5
