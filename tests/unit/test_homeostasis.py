"""
Unit tests for HomeostasisRegulator.

Tests cover:
- HomeostasisConfig validation
- NodeState tracking and statistics
- HomeostasisRegulator activation recording and excitability
- Regulation adjustments
- Health metrics
- Serialization/deserialization
- AdaptiveHomeostasisRegulator

Part of Sprint 2: Hebbian Hive Enhancement (Woven Mind + PRISM Marriage)
"""

import pytest
from collections import deque
from cortical.reasoning.homeostasis import (
    HomeostasisConfig,
    NodeState,
    HomeostasisRegulator,
    AdaptiveHomeostasisRegulator,
)


# ==============================================================================
# HOMEOSTASIS CONFIG TESTS
# ==============================================================================


class TestHomeostasisConfig:
    """Tests for HomeostasisConfig validation."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = HomeostasisConfig()
        assert config.target_activation == 0.05
        assert config.min_excitability == 0.1
        assert config.max_excitability == 10.0
        assert config.adjustment_rate == 0.01
        assert config.history_size == 100
        assert config.min_history_for_adjustment == 10

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = HomeostasisConfig(
            target_activation=0.10,
            min_excitability=0.2,
            max_excitability=5.0,
            adjustment_rate=0.05,
            history_size=50,
            min_history_for_adjustment=5,
        )
        assert config.target_activation == 0.10
        assert config.min_excitability == 0.2
        assert config.max_excitability == 5.0
        assert config.adjustment_rate == 0.05
        assert config.history_size == 50
        assert config.min_history_for_adjustment == 5

    def test_target_activation_validation(self):
        """target_activation must be in [0, 1]."""
        with pytest.raises(ValueError, match="target_activation"):
            HomeostasisConfig(target_activation=-0.1)
        with pytest.raises(ValueError, match="target_activation"):
            HomeostasisConfig(target_activation=1.5)

    def test_min_excitability_validation(self):
        """min_excitability must be positive."""
        with pytest.raises(ValueError, match="min_excitability"):
            HomeostasisConfig(min_excitability=0.0)
        with pytest.raises(ValueError, match="min_excitability"):
            HomeostasisConfig(min_excitability=-0.1)

    def test_excitability_range_validation(self):
        """max_excitability must be >= min_excitability."""
        with pytest.raises(ValueError, match="max_excitability"):
            HomeostasisConfig(min_excitability=5.0, max_excitability=2.0)

    def test_adjustment_rate_validation(self):
        """adjustment_rate must be in (0, 1]."""
        with pytest.raises(ValueError, match="adjustment_rate"):
            HomeostasisConfig(adjustment_rate=0.0)
        with pytest.raises(ValueError, match="adjustment_rate"):
            HomeostasisConfig(adjustment_rate=1.5)


# ==============================================================================
# NODE STATE TESTS
# ==============================================================================


class TestNodeState:
    """Tests for NodeState tracking."""

    def test_default_state(self):
        """Default state should be neutral."""
        state = NodeState()
        assert state.excitability == 1.0
        assert len(state.activation_history) == 0
        assert state.total_activations == 0

    def test_record_activation(self):
        """Recording should update history and count."""
        state = NodeState()
        state.record(0.5)
        assert len(state.activation_history) == 1
        assert state.activation_history[0] == 0.5
        assert state.total_activations == 1

    def test_record_zero_activation(self):
        """Zero activation should be recorded but not counted."""
        state = NodeState()
        state.record(0.0)
        assert len(state.activation_history) == 1
        assert state.total_activations == 0

    def test_history_limit(self):
        """History should be bounded by maxlen."""
        state = NodeState(activation_history=deque(maxlen=5))
        for i in range(10):
            state.record(float(i) / 10)
        assert len(state.activation_history) == 5
        # Should have last 5 values: 0.5, 0.6, 0.7, 0.8, 0.9
        assert list(state.activation_history) == [0.5, 0.6, 0.7, 0.8, 0.9]

    def test_average_activation_empty(self):
        """Empty history should return 0."""
        state = NodeState()
        assert state.average_activation() == 0.0

    def test_average_activation(self):
        """Average should be calculated correctly."""
        state = NodeState()
        state.record(0.2)
        state.record(0.4)
        state.record(0.6)
        assert state.average_activation() == pytest.approx(0.4)

    def test_variance_insufficient_data(self):
        """Variance needs at least 2 data points."""
        state = NodeState()
        assert state.recent_activation_variance() == 0.0
        state.record(0.5)
        assert state.recent_activation_variance() == 0.0

    def test_variance_calculation(self):
        """Variance should be calculated correctly."""
        state = NodeState()
        # All same values -> variance = 0
        for _ in range(5):
            state.record(0.5)
        assert state.recent_activation_variance() == 0.0

        # Different values
        state2 = NodeState()
        state2.record(0.0)
        state2.record(1.0)
        # avg = 0.5, variance = (0.25 + 0.25) / 2 = 0.25
        assert state2.recent_activation_variance() == pytest.approx(0.25)


# ==============================================================================
# HOMEOSTASIS REGULATOR TESTS
# ==============================================================================


class TestHomeostasisRegulator:
    """Tests for HomeostasisRegulator."""

    @pytest.fixture
    def regulator(self):
        """Create a default regulator."""
        return HomeostasisRegulator()

    @pytest.fixture
    def custom_regulator(self):
        """Create a regulator with custom config."""
        return HomeostasisRegulator(
            target_activation=0.10,
            adjustment_rate=0.05,
            min_history_for_adjustment=5,
        )

    def test_init_default(self, regulator):
        """Default initialization."""
        assert regulator.config.target_activation == 0.05
        assert len(regulator.nodes) == 0

    def test_init_with_kwargs(self, custom_regulator):
        """Init with kwargs overrides."""
        assert custom_regulator.config.target_activation == 0.10
        assert custom_regulator.config.adjustment_rate == 0.05

    def test_record_activation_creates_node(self, regulator):
        """Recording creates node state."""
        regulator.record_activation("node_a", 0.5)
        assert "node_a" in regulator.nodes
        assert regulator.nodes["node_a"].activation_history[-1] == 0.5

    def test_record_activations_batch(self, regulator):
        """Batch recording works."""
        regulator.record_activations({
            "node_a": 0.5,
            "node_b": 0.3,
            "node_c": 0.7,
        })
        assert len(regulator.nodes) == 3
        assert regulator.nodes["node_a"].activation_history[-1] == 0.5
        assert regulator.nodes["node_b"].activation_history[-1] == 0.3

    def test_get_excitability_unknown(self, regulator):
        """Unknown nodes have default excitability."""
        assert regulator.get_excitability("unknown") == 1.0

    def test_get_excitability_known(self, regulator):
        """Known nodes return their excitability."""
        regulator.record_activation("node_a", 0.5)
        regulator.nodes["node_a"].excitability = 0.8
        assert regulator.get_excitability("node_a") == 0.8

    def test_get_all_excitabilities(self, regulator):
        """All excitabilities returned."""
        regulator.record_activation("node_a", 0.5)
        regulator.record_activation("node_b", 0.3)
        regulator.nodes["node_a"].excitability = 0.8
        regulator.nodes["node_b"].excitability = 1.2

        all_exc = regulator.get_all_excitabilities()
        assert all_exc == {"node_a": 0.8, "node_b": 1.2}


class TestRegulation:
    """Tests for the regulate() method."""

    @pytest.fixture
    def regulator(self):
        """Create a regulator with fast adjustment for testing."""
        return HomeostasisRegulator(
            adjustment_rate=0.1,
            min_history_for_adjustment=3,
        )

    def test_regulate_no_nodes(self, regulator):
        """Empty regulator returns empty dict."""
        result = regulator.regulate()
        assert result == {}

    def test_regulate_insufficient_history(self, regulator):
        """Nodes without enough history aren't adjusted."""
        regulator.record_activation("node_a", 0.5)
        regulator.record_activation("node_a", 0.5)
        # Only 2 activations, need 3
        result = regulator.regulate()
        assert "node_a" not in result

    def test_regulate_overactive_node(self, regulator):
        """Overactive nodes get reduced excitability."""
        # High activation (above 0.05 target)
        for _ in range(5):
            regulator.record_activation("busy", 0.8)

        result = regulator.regulate()
        assert "busy" in result
        assert result["busy"] < 1.0  # Decreased

    def test_regulate_underactive_node(self, regulator):
        """Underactive nodes get increased excitability."""
        # Low activation (below 0.05 target)
        for _ in range(5):
            regulator.record_activation("quiet", 0.01)

        result = regulator.regulate()
        assert "quiet" in result
        assert result["quiet"] > 1.0  # Increased

    def test_regulate_respects_bounds(self):
        """Excitability stays within configured bounds."""
        regulator = HomeostasisRegulator(
            adjustment_rate=0.5,  # Fast adjustment
            min_excitability=0.5,
            max_excitability=2.0,
            min_history_for_adjustment=2,
        )

        # Drive excitability down
        for _ in range(100):
            regulator.record_activation("busy", 0.9)
            regulator.regulate()

        assert regulator.get_excitability("busy") >= 0.5

        # Drive excitability up
        for _ in range(100):
            regulator.record_activation("quiet", 0.001)
            regulator.regulate()

        assert regulator.get_excitability("quiet") <= 2.0


class TestApplyExcitability:
    """Tests for excitability application to activations."""

    def test_apply_to_unknown_nodes(self):
        """Unknown nodes get multiplied by 1.0."""
        regulator = HomeostasisRegulator()
        result = regulator.apply_excitability({"unknown": 0.5})
        assert result["unknown"] == 0.5

    def test_apply_modulates_activation(self):
        """Known excitabilities modulate activations."""
        regulator = HomeostasisRegulator()
        regulator.record_activation("node_a", 0.5)
        regulator.nodes["node_a"].excitability = 0.5

        result = regulator.apply_excitability({"node_a": 1.0})
        assert result["node_a"] == 0.5  # 1.0 * 0.5

    def test_apply_multiple_nodes(self):
        """Multiple nodes are all modulated."""
        regulator = HomeostasisRegulator()
        regulator.record_activation("a", 0.5)
        regulator.record_activation("b", 0.5)
        regulator.nodes["a"].excitability = 0.5
        regulator.nodes["b"].excitability = 2.0

        result = regulator.apply_excitability({"a": 0.4, "b": 0.3})
        assert result["a"] == pytest.approx(0.2)   # 0.4 * 0.5
        assert result["b"] == pytest.approx(0.6)   # 0.3 * 2.0


class TestNodeAnalysis:
    """Tests for underactive/overactive node detection."""

    @pytest.fixture
    def regulator_with_nodes(self):
        """Create regulator with various activity levels."""
        regulator = HomeostasisRegulator(
            target_activation=0.10,
            min_history_for_adjustment=3,
        )

        # Underactive (< 50% of 0.10 = < 0.05)
        for _ in range(5):
            regulator.record_activation("dead", 0.01)

        # Normal
        for _ in range(5):
            regulator.record_activation("normal", 0.10)

        # Overactive (> 200% of 0.10 = > 0.20)
        for _ in range(5):
            regulator.record_activation("hyperactive", 0.50)

        return regulator

    def test_get_underactive_nodes(self, regulator_with_nodes):
        """Underactive nodes are detected."""
        underactive = regulator_with_nodes.get_underactive_nodes()
        node_ids = [n[0] for n in underactive]
        assert "dead" in node_ids
        assert "normal" not in node_ids
        assert "hyperactive" not in node_ids

    def test_get_overactive_nodes(self, regulator_with_nodes):
        """Overactive nodes are detected."""
        overactive = regulator_with_nodes.get_overactive_nodes()
        node_ids = [n[0] for n in overactive]
        assert "hyperactive" in node_ids
        assert "normal" not in node_ids
        assert "dead" not in node_ids

    def test_nodes_sorted_by_activity(self, regulator_with_nodes):
        """Results are sorted appropriately."""
        # Add another underactive node
        for _ in range(5):
            regulator_with_nodes.record_activation("almost_dead", 0.02)

        underactive = regulator_with_nodes.get_underactive_nodes()
        # "dead" (0.01) should be before "almost_dead" (0.02)
        node_ids = [n[0] for n in underactive]
        assert node_ids.index("dead") < node_ids.index("almost_dead")


class TestHealthMetrics:
    """Tests for network health metrics."""

    def test_empty_network_metrics(self):
        """Empty network has default metrics."""
        regulator = HomeostasisRegulator()
        metrics = regulator.get_health_metrics()

        assert metrics["avg_excitability"] == 1.0
        assert metrics["excitability_std"] == 0.0
        assert metrics["node_count"] == 0

    def test_health_metrics_calculation(self):
        """Metrics are calculated correctly."""
        regulator = HomeostasisRegulator(min_history_for_adjustment=2)

        # Add nodes with different states
        regulator.record_activation("a", 0.5)
        regulator.record_activation("a", 0.5)
        regulator.record_activation("b", 0.5)
        regulator.record_activation("b", 0.5)

        regulator.nodes["a"].excitability = 0.5
        regulator.nodes["b"].excitability = 1.5

        metrics = regulator.get_health_metrics()

        assert metrics["node_count"] == 2
        assert metrics["avg_excitability"] == pytest.approx(1.0)
        assert metrics["excitability_std"] == pytest.approx(0.5)


class TestReset:
    """Tests for reset functionality."""

    def test_reset_node(self):
        """Reset single node."""
        regulator = HomeostasisRegulator()
        regulator.record_activation("a", 0.5)
        regulator.record_activation("b", 0.5)

        regulator.reset_node("a")
        assert "a" not in regulator.nodes
        assert "b" in regulator.nodes

    def test_reset_unknown_node(self):
        """Resetting unknown node is safe."""
        regulator = HomeostasisRegulator()
        regulator.reset_node("unknown")  # Should not raise

    def test_reset_all(self):
        """Reset all nodes."""
        regulator = HomeostasisRegulator()
        regulator.record_activation("a", 0.5)
        regulator.record_activation("b", 0.5)

        regulator.reset_all()
        assert len(regulator.nodes) == 0


class TestSerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict(self):
        """Serialize regulator state."""
        regulator = HomeostasisRegulator(
            target_activation=0.10,
            adjustment_rate=0.02,
        )
        regulator.record_activation("node_a", 0.5)
        regulator.record_activation("node_a", 0.6)
        regulator.nodes["node_a"].excitability = 0.8

        data = regulator.to_dict()

        assert data["config"]["target_activation"] == 0.10
        assert data["config"]["adjustment_rate"] == 0.02
        assert "node_a" in data["nodes"]
        assert data["nodes"]["node_a"]["excitability"] == 0.8
        assert data["nodes"]["node_a"]["activation_history"] == [0.5, 0.6]

    def test_from_dict(self):
        """Deserialize regulator state."""
        data = {
            "config": {
                "target_activation": 0.10,
                "adjustment_rate": 0.02,
            },
            "nodes": {
                "node_a": {
                    "excitability": 0.8,
                    "activation_history": [0.5, 0.6],
                    "total_activations": 2,
                }
            }
        }

        regulator = HomeostasisRegulator.from_dict(data)

        assert regulator.config.target_activation == 0.10
        assert regulator.get_excitability("node_a") == 0.8
        assert list(regulator.nodes["node_a"].activation_history) == [0.5, 0.6]

    def test_round_trip(self):
        """Serialize and deserialize preserves state."""
        original = HomeostasisRegulator(target_activation=0.15)
        original.record_activation("a", 0.3)
        original.record_activation("a", 0.4)
        original.nodes["a"].excitability = 0.75

        data = original.to_dict()
        restored = HomeostasisRegulator.from_dict(data)

        assert restored.config.target_activation == original.config.target_activation
        assert restored.get_excitability("a") == original.get_excitability("a")
        assert list(restored.nodes["a"].activation_history) == \
               list(original.nodes["a"].activation_history)


# ==============================================================================
# ADAPTIVE HOMEOSTASIS REGULATOR TESTS
# ==============================================================================


class TestAdaptiveHomeostasisRegulator:
    """Tests for AdaptiveHomeostasisRegulator."""

    @pytest.fixture
    def adaptive(self):
        """Create an adaptive regulator."""
        return AdaptiveHomeostasisRegulator(
            target_activation=0.10,
            min_target=0.01,
            max_target=0.30,
            target_adjustment_rate=0.1,
            min_history_for_adjustment=3,
        )

    def test_init(self, adaptive):
        """Initialization with adaptive parameters."""
        assert adaptive.min_target == 0.01
        assert adaptive.max_target == 0.30
        assert adaptive.target_adjustment_rate == 0.1

    def test_target_decreases_when_many_underactive(self, adaptive):
        """Target should decrease if many nodes are underactive."""
        # Create many underactive nodes
        for i in range(10):
            for _ in range(5):
                adaptive.record_activation(f"node_{i}", 0.01)

        initial_target = adaptive.config.target_activation
        adaptive.regulate()

        # Target should decrease (too many underactive)
        assert adaptive.config.target_activation < initial_target

    def test_target_increases_when_many_overactive(self, adaptive):
        """Target should increase if many nodes are overactive."""
        # Create many overactive nodes
        for i in range(10):
            for _ in range(5):
                adaptive.record_activation(f"node_{i}", 0.90)

        initial_target = adaptive.config.target_activation
        adaptive.regulate()

        # Target should increase (too many overactive)
        assert adaptive.config.target_activation > initial_target

    def test_target_respects_bounds(self, adaptive):
        """Target stays within min/max."""
        # Push toward min
        for i in range(100):
            for j in range(5):
                adaptive.record_activation(f"under_{i}_{j}", 0.001)
            adaptive.regulate()

        assert adaptive.config.target_activation >= adaptive.min_target

        # Push toward max
        for i in range(100):
            for j in range(5):
                adaptive.record_activation(f"over_{i}_{j}", 0.99)
            adaptive.regulate()

        assert adaptive.config.target_activation <= adaptive.max_target

    def test_get_target_history(self, adaptive):
        """Target history is tracked."""
        initial_target = adaptive.config.target_activation

        # Cause a target change
        for i in range(10):
            for _ in range(5):
                adaptive.record_activation(f"node_{i}", 0.90)
        adaptive.regulate()

        history = adaptive.get_target_history()
        if history:  # Only if target changed
            assert initial_target in history or adaptive.config.target_activation != initial_target


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests for homeostasis with PRISM-SLM."""

    def test_with_prism_slm_activations(self):
        """Test homeostasis with typical PRISM-SLM activation patterns."""
        from cortical.reasoning.prism_slm import TransitionGraph

        graph = TransitionGraph(context_size=2)
        graph.learn_sequence(["the", "quick", "brown", "fox"])
        graph.learn_sequence(["the", "lazy", "dog"])

        regulator = HomeostasisRegulator(
            target_activation=0.05,
            min_history_for_adjustment=3,
        )

        # Simulate multiple activation rounds
        for _ in range(10):
            activations = graph.sparse_activate(["the"], k=3)
            regulator.record_activations(activations)
            regulator.regulate()

        # Check that excitabilities have been adjusted
        metrics = regulator.get_health_metrics()
        assert metrics["node_count"] > 0

    def test_excitability_modulation_effect(self):
        """Excitability modulation affects activation values."""
        regulator = HomeostasisRegulator(min_history_for_adjustment=2)

        # Create known activations
        activations = {
            "neural": 0.9,
            "networks": 0.7,
            "learn": 0.5,
        }

        regulator.record_activations(activations)
        regulator.record_activations(activations)

        # Suppress "networks" node
        regulator.nodes["networks"].excitability = 0.5

        # Apply excitability modulation
        modulated = regulator.apply_excitability(activations)

        # "networks" should be suppressed by 50%
        assert modulated["networks"] == pytest.approx(0.35)  # 0.7 * 0.5
        # Other nodes should be unchanged (excitability = 1.0)
        assert modulated["neural"] == pytest.approx(0.9)
        assert modulated["learn"] == pytest.approx(0.5)
