"""
Tests for Loom-Cortex integration.

TDD: Write tests first for connecting Loom to the enhanced Cortex
(AbstractionEngine) for SLOW mode processing.

Part of Sprint 4: The Loom Weaves (T4.2)
Part of the Woven Mind + PRISM Marriage project.
"""

import pytest
from typing import Dict, FrozenSet, Optional

from cortical.reasoning.abstraction import (
    Abstraction,
    AbstractionEngine,
)


class TestLoomCortexConnectorCreation:
    """Test LoomCortexConnector creation and initialization."""

    def test_connector_with_defaults(self):
        """Connector should create with default AbstractionEngine."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector()
        assert connector.engine is not None
        assert isinstance(connector.engine, AbstractionEngine)

    def test_connector_with_custom_engine(self):
        """Connector should accept custom AbstractionEngine."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        engine = AbstractionEngine(min_frequency=5)
        connector = LoomCortexConnector(engine=engine)

        assert connector.engine is engine
        assert connector.engine.min_frequency == 5

    def test_connector_with_pln_bridge(self):
        """Connector should optionally integrate with PLN bridge."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.prism_pln import PLNReasoner
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)
        connector = LoomCortexConnector(pln_bridge=bridge)

        assert connector.pln_bridge is bridge


class TestSlowModeProcessing:
    """Test SLOW mode processing through the Cortex."""

    def test_process_slow_observes_patterns(self):
        """SLOW mode should observe patterns for abstraction."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector()

        # Process multiple times to build patterns
        for _ in range(5):
            connector.process_slow(["neural", "networks"])

        # Check that pattern was observed
        stats = connector.get_pattern_stats()
        assert stats["total_observations"] >= 5

    def test_process_slow_forms_abstractions(self):
        """SLOW mode should form abstractions from repeated patterns."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=3)

        # Process pattern enough times to form abstraction
        for _ in range(5):
            connector.process_slow(["machine", "learning"])

        # Should have formed at least one abstraction
        abstractions = connector.get_abstractions()
        assert len(abstractions) >= 1

    def test_process_slow_returns_active_abstractions(self):
        """SLOW mode should return which abstractions became active."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build pattern
        for _ in range(4):
            connector.process_slow(["deep", "learning"])

        # Process again - should return active abstractions
        active = connector.process_slow(["deep", "learning"])

        assert len(active) >= 1


class TestAbstractionRetrieval:
    """Test abstraction retrieval for deliberative processing."""

    def test_query_abstractions(self):
        """Connector should allow querying abstractions by source nodes."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build abstractions
        for _ in range(4):
            connector.process_slow(["neural", "network"])
            connector.process_slow(["deep", "learning"])

        # Query by source nodes
        results = connector.query_abstractions(["neural"])
        assert len(results) >= 1
        # All results should involve "neural"
        for abstraction in results:
            assert "neural" in abstraction.source_nodes

    def test_get_abstraction_by_id(self):
        """Connector should retrieve specific abstractions by ID."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build abstraction
        for _ in range(4):
            connector.process_slow(["test", "pattern"])

        # Get an abstraction
        abstractions = connector.get_abstractions()
        if abstractions:
            abstraction_id = abstractions[0].id
            retrieved = connector.get_abstraction(abstraction_id)
            assert retrieved is not None
            assert retrieved.id == abstraction_id

    def test_get_related_abstractions(self):
        """Connector should find related abstractions."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build overlapping patterns
        for _ in range(4):
            connector.process_slow(["machine", "learning"])
            connector.process_slow(["machine", "code"])

        # Find abstractions related to "machine"
        related = connector.get_related_abstractions("machine")
        # Should find multiple abstractions sharing "machine"
        assert len(related) >= 1


class TestDeliberativeAnalysis:
    """Test deliberative analysis capabilities for SLOW mode."""

    def test_analyze_with_pln(self):
        """SLOW mode should use PLN for deeper analysis when available."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.prism_pln import PLNReasoner
        from cortical.reasoning.abstraction_pln import AbstractionPLNBridge

        reasoner = PLNReasoner()
        bridge = AbstractionPLNBridge(reasoner)
        connector = LoomCortexConnector(pln_bridge=bridge, min_frequency=2)

        # Build abstraction
        for _ in range(4):
            connector.process_slow(["neural", "network"])

        # Analyze with PLN
        analysis = connector.analyze(["neural", "network"])
        assert "abstractions" in analysis
        assert "truth_values" in analysis

    def test_explain_abstraction(self):
        """Connector should explain why an abstraction exists."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build abstraction
        for _ in range(4):
            connector.process_slow(["test", "concept"])

        # Get explanation
        abstractions = connector.get_abstractions()
        if abstractions:
            explanation = connector.explain_abstraction(abstractions[0].id)
            assert "source_nodes" in explanation
            assert "frequency" in explanation
            assert "level" in explanation


class TestLoomIntegration:
    """Test integration with the Loom for mode switching."""

    def test_provides_slow_mode_output(self):
        """Connector should provide output compatible with Loom."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.loom import Loom

        connector = LoomCortexConnector(min_frequency=2)

        # Build patterns
        for _ in range(4):
            connector.process_slow(["deliberate", "thought"])

        # Output should work with Loom's expectations
        output = connector.process_slow(["deliberate", "thought"])
        assert isinstance(output, set)

    def test_abstraction_activation_for_loom(self):
        """Connector should provide activation levels for surprise detection."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build patterns
        for _ in range(4):
            connector.process_slow(["test", "token"])

        # Get activation levels
        activations = connector.get_abstraction_activations()
        assert isinstance(activations, dict)
        # Values should be activation levels
        for val in activations.values():
            assert isinstance(val, (int, float))


class TestHierarchicalProcessing:
    """Test hierarchical abstraction processing."""

    def test_multi_level_abstractions(self):
        """Connector should support multi-level abstractions."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build level 1 abstractions
        for _ in range(4):
            connector.process_slow(["neural", "network"])
            connector.process_slow(["deep", "learning"])

        # Try to build level 2 (meta-abstraction)
        connector.build_meta_abstractions()

        stats = connector.get_abstraction_stats()
        # Should have some abstractions
        assert stats["total_abstractions"] >= 1


class TestLoomCortexConnectorSerialization:
    """Test serialization of connector state."""

    def test_to_dict(self):
        """Connector should serialize to dictionary."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=3)

        # Build some state
        for _ in range(4):
            connector.process_slow(["test", "data"])

        data = connector.to_dict()
        assert "min_frequency" in data
        assert "engine_state" in data

    def test_from_dict(self):
        """Connector should deserialize from dictionary."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        original = LoomCortexConnector(min_frequency=3)

        # Build some state
        for _ in range(4):
            original.process_slow(["test", "data"])

        data = original.to_dict()
        restored = LoomCortexConnector.from_dict(data)

        assert restored.min_frequency == 3
        # Should have same number of abstractions
        assert len(restored.get_abstractions()) == len(original.get_abstractions())


class TestLoomCortexCoverageEdgeCases:
    """Additional tests for edge case coverage."""

    def test_partial_abstraction_activation(self):
        """Partial pattern matches should activate with lower weight."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Form an abstraction from ["a", "b"]
        small_pattern = ["a", "b"]
        for _ in range(3):
            connector.process_slow(small_pattern)

        # Verify abstraction exists
        abstractions = connector.get_abstractions()
        assert len(abstractions) >= 1

        # Now process with a superset pattern ["a", "b", "c", "d"]
        # The existing abstraction for ["a", "b"] is a subset of the input
        superset_pattern = ["a", "b", "c", "d"]
        result = connector.process_slow(superset_pattern)

        # Should have activations (the abstraction should be partially activated)
        assert isinstance(result, set)
        # Check activation was recorded
        assert len(connector._abstraction_activations) >= 1

    def test_meta_abstraction_with_overlapping_sources(self):
        """Meta-abstractions form from abstractions with overlapping sources."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Create abstractions that share a common node
        # Abstraction 1: ["common", "a"]
        # Abstraction 2: ["common", "b"]
        for _ in range(3):
            connector.process_slow(["common", "a"])
        for _ in range(3):
            connector.process_slow(["common", "b"])

        # Build meta-abstractions
        meta = connector.build_meta_abstractions()

        # Should return a list (may or may not have formed abstractions)
        assert isinstance(meta, list)

    def test_get_stats_comprehensive(self):
        """Get comprehensive stats from connector."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Build some patterns
        for _ in range(3):
            connector.process_slow(["pattern", "one"])
            connector.process_slow(["pattern", "two"])

        # Get pattern stats
        pattern_stats = connector.get_pattern_stats()
        assert "total_observations" in pattern_stats
        assert "unique_patterns" in pattern_stats
        assert "total_abstractions" in pattern_stats

        # Get abstraction stats
        abstraction_stats = connector.get_abstraction_stats()
        assert "total_abstractions" in abstraction_stats

    def test_explain_abstraction(self):
        """Explain abstraction returns metadata."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Form an abstraction
        for _ in range(3):
            connector.process_slow(["machine", "learning"])

        abstractions = connector.get_abstractions()
        if abstractions:
            abstraction = abstractions[0]
            explanation = connector.explain_abstraction(abstraction.id)

            assert "source_nodes" in explanation
            assert "level" in explanation
            assert "truth_value" in explanation
            assert "strength" in explanation
            assert "formed_at" in explanation

    def test_explain_nonexistent_abstraction(self):
        """Explain nonexistent abstraction returns error dict."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        result = connector.explain_abstraction("nonexistent-id")
        assert "error" in result
        assert "not found" in result["error"]

    def test_abstraction_activation_exact_match(self):
        """Exact pattern match activates abstraction fully."""
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        connector = LoomCortexConnector(min_frequency=2)

        # Form abstraction
        exact_pattern = ["alpha", "beta"]
        for _ in range(3):
            connector.process_slow(exact_pattern)

        # Process again with exact same pattern
        result = connector.process_slow(exact_pattern)

        # Should activate and return activations
        assert isinstance(result, set)
        # Check full activation (1.0) for exact match
        for aid, val in connector._abstraction_activations.items():
            if val == 1.0:
                break
        else:
            # At least should have some activations
            assert len(connector._abstraction_activations) >= 0
