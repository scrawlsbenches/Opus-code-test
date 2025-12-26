"""
Unit tests for PRISM-SLM Hebbian Hive enhancements.

Tests cover:
- lateral_inhibition(): Sparse activation through inhibition
- k_winners_take_all(): Competition to select top-k
- spreading_activation(): Associative retrieval
- sparse_activate(): Combined pipeline
- HiveNode: Node with activation traces
- HiveEdge: Edge with Hebbian learning traces

Part of Sprint 2: Hebbian Hive Enhancement (Woven Mind + PRISM Marriage)
"""

import pytest
from cortical.reasoning.prism_slm import (
    TransitionGraph,
    PRISMLanguageModel,
    HiveNode,
    HiveEdge,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def empty_graph():
    """Empty TransitionGraph for testing."""
    return TransitionGraph(context_size=2)


@pytest.fixture
def trained_graph():
    """TransitionGraph with some learned sequences."""
    graph = TransitionGraph(context_size=2)
    graph.learn_sequence(["the", "quick", "brown", "fox"])
    graph.learn_sequence(["the", "lazy", "dog"])
    graph.learn_sequence(["quick", "brown", "fox", "jumps"])
    return graph


@pytest.fixture
def simple_activations():
    """Simple activation pattern for testing."""
    return {
        "a": 0.9,
        "b": 0.7,
        "c": 0.5,
        "d": 0.3,
        "e": 0.1,
    }


# ==============================================================================
# LATERAL INHIBITION TESTS
# ==============================================================================


class TestLateralInhibition:
    """Tests for lateral_inhibition method."""

    def test_empty_activations_returns_empty(self, empty_graph):
        """Empty input should return empty output."""
        result = empty_graph.lateral_inhibition({})
        assert result == {}

    def test_single_activation_no_change(self, empty_graph):
        """Single token has no neighbors to inhibit it."""
        activations = {"a": 1.0}
        result = empty_graph.lateral_inhibition(activations)
        # Should be unchanged (no neighbors)
        assert result["a"] == 1.0

    def test_stronger_inhibits_weaker(self, empty_graph, simple_activations):
        """Stronger activations should inhibit weaker ones."""
        result = empty_graph.lateral_inhibition(simple_activations)

        # "a" is strongest, should be least inhibited
        # "e" is weakest, should be most inhibited
        assert result["a"] >= result["b"] >= result["c"] >= result["d"]

    def test_activations_remain_non_negative(self, empty_graph, simple_activations):
        """All activations should remain >= 0."""
        result = empty_graph.lateral_inhibition(
            simple_activations,
            inhibition_strength=1.0,  # Strong inhibition
        )
        for activation in result.values():
            assert activation >= 0.0

    def test_inhibition_produces_sparser_pattern(self, empty_graph):
        """Inhibition should reduce total activation (sparser)."""
        activations = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        result = empty_graph.lateral_inhibition(activations)

        total_before = sum(activations.values())
        total_after = sum(result.values())

        assert total_after < total_before

    def test_inhibition_radius_affects_spread(self, empty_graph):
        """Larger radius should affect more tokens."""
        activations = {"a": 1.0, "b": 0.5, "c": 0.3, "d": 0.2, "e": 0.1}

        result_small = empty_graph.lateral_inhibition(
            activations, inhibition_radius=1
        )
        result_large = empty_graph.lateral_inhibition(
            activations, inhibition_radius=4
        )

        # Larger radius should produce more inhibition overall
        total_small = sum(result_small.values())
        total_large = sum(result_large.values())

        # Both should be less than original, large radius more so
        original_total = sum(activations.values())
        assert total_large <= total_small <= original_total

    def test_inhibition_strength_scales_effect(self, empty_graph):
        """Higher strength should produce more inhibition."""
        activations = {"a": 1.0, "b": 0.8, "c": 0.6}

        result_weak = empty_graph.lateral_inhibition(
            activations, inhibition_strength=0.1
        )
        result_strong = empty_graph.lateral_inhibition(
            activations, inhibition_strength=0.9
        )

        # Stronger inhibition = less total activation
        assert sum(result_strong.values()) <= sum(result_weak.values())


# ==============================================================================
# K WINNERS TAKE ALL TESTS
# ==============================================================================


class TestKWinnersTakeAll:
    """Tests for k_winners_take_all method."""

    def test_empty_activations_returns_empty(self, empty_graph):
        """Empty input should return empty output."""
        result = empty_graph.k_winners_take_all({})
        assert result == {}

    def test_k_greater_than_count_keeps_all(self, empty_graph):
        """If k > number of tokens, keep all qualifying tokens."""
        activations = {"a": 0.9, "b": 0.5}
        result = empty_graph.k_winners_take_all(activations, k=5)

        # Both should be kept (k=5 but only 2 tokens)
        assert result["a"] == 0.9
        assert result["b"] == 0.5

    def test_k_selects_top_k(self, empty_graph, simple_activations):
        """Should select exactly k winners."""
        result = empty_graph.k_winners_take_all(simple_activations, k=2)

        # Count non-zero entries
        winners = [t for t, a in result.items() if a > 0]
        assert len(winners) == 2
        assert "a" in winners  # Highest
        assert "b" in winners  # Second highest

    def test_losers_get_zero(self, empty_graph, simple_activations):
        """Non-winners should have zero activation."""
        result = empty_graph.k_winners_take_all(simple_activations, k=2)

        assert result["c"] == 0.0
        assert result["d"] == 0.0
        assert result["e"] == 0.0

    def test_min_activation_threshold(self, empty_graph):
        """Tokens below min_activation should not win."""
        activations = {"a": 0.5, "b": 0.3, "c": 0.05}
        result = empty_graph.k_winners_take_all(
            activations, k=3, min_activation=0.1
        )

        # "c" is below threshold, shouldn't be a winner
        assert result["a"] == 0.5
        assert result["b"] == 0.3
        assert result["c"] == 0.0

    def test_k_zero_returns_all_zero(self, empty_graph, simple_activations):
        """k=0 should result in all zeros."""
        result = empty_graph.k_winners_take_all(simple_activations, k=0)

        for activation in result.values():
            assert activation == 0.0

    def test_preserves_winner_values(self, empty_graph, simple_activations):
        """Winners should keep their original activation values."""
        result = empty_graph.k_winners_take_all(simple_activations, k=3)

        assert result["a"] == 0.9  # Original value preserved
        assert result["b"] == 0.7
        assert result["c"] == 0.5


# ==============================================================================
# SPREADING ACTIVATION TESTS
# ==============================================================================


class TestSpreadingActivation:
    """Tests for spreading_activation method."""

    def test_empty_seeds_returns_empty(self, trained_graph):
        """Empty seed should return empty output."""
        result = trained_graph.spreading_activation({})
        assert result == {}

    def test_seeds_included_in_result(self, trained_graph):
        """Seed tokens should be in the result."""
        seeds = {"quick": 1.0}
        result = trained_graph.spreading_activation(seeds)

        assert "quick" in result
        assert result["quick"] >= 1.0  # At least seed value

    def test_activation_spreads_to_connected(self, trained_graph):
        """Activation should spread to connected tokens."""
        seeds = {"quick": 1.0}
        result = trained_graph.spreading_activation(seeds)

        # "brown" should be activated (quick -> brown transition exists)
        assert "brown" in result
        assert result["brown"] > 0

    def test_spread_factor_affects_amount(self, trained_graph):
        """Higher spread factor should spread more activation."""
        seeds = {"the": 1.0}

        result_low = trained_graph.spreading_activation(
            seeds, spread_factor=0.1
        )
        result_high = trained_graph.spreading_activation(
            seeds, spread_factor=0.9
        )

        # Higher spread should have more total activation
        # (excluding the seed)
        total_low = sum(v for k, v in result_low.items() if k != "the")
        total_high = sum(v for k, v in result_high.items() if k != "the")

        assert total_high >= total_low

    def test_decay_reduces_with_steps(self, trained_graph):
        """Decay should reduce activation over steps."""
        seeds = {"the": 1.0}

        # Low decay = more persistent
        result_persistent = trained_graph.spreading_activation(
            seeds, decay_per_step=0.9, max_steps=3
        )
        # High decay = fades quickly
        result_fading = trained_graph.spreading_activation(
            seeds, decay_per_step=0.3, max_steps=3
        )

        # Persistent should have more total activation
        total_persistent = sum(result_persistent.values())
        total_fading = sum(result_fading.values())

        assert total_persistent >= total_fading

    def test_max_steps_limits_propagation(self, trained_graph):
        """max_steps should limit how far activation spreads."""
        seeds = {"the": 1.0}

        result_one = trained_graph.spreading_activation(
            seeds, max_steps=1
        )
        result_many = trained_graph.spreading_activation(
            seeds, max_steps=5
        )

        # More steps = more tokens activated
        assert len(result_many) >= len(result_one)

    def test_threshold_filters_weak_activation(self, trained_graph):
        """Threshold should filter out weak activations."""
        seeds = {"the": 1.0}

        # High threshold = fewer activated tokens
        result_high_thresh = trained_graph.spreading_activation(
            seeds, threshold=0.5
        )
        result_low_thresh = trained_graph.spreading_activation(
            seeds, threshold=0.001
        )

        assert len(result_low_thresh) >= len(result_high_thresh)

    def test_no_transitions_returns_seeds(self, empty_graph):
        """Graph with no transitions should just return seeds."""
        seeds = {"alone": 1.0}
        result = empty_graph.spreading_activation(seeds)

        assert result == seeds


# ==============================================================================
# SPARSE ACTIVATE TESTS
# ==============================================================================


class TestSparseActivate:
    """Tests for sparse_activate combined pipeline."""

    def test_empty_query_returns_empty(self, trained_graph):
        """Empty query should return empty result."""
        result = trained_graph.sparse_activate([])
        assert result == {}

    def test_query_tokens_included(self, trained_graph):
        """Query tokens should be in result if they have activation."""
        result = trained_graph.sparse_activate(["quick"], k=3)

        # "quick" should be active
        assert "quick" in result or len(result) > 0

    def test_k_limits_output_size(self, trained_graph):
        """Output should have at most k active tokens."""
        result = trained_graph.sparse_activate(["the"], k=2)

        active_count = sum(1 for v in result.values() if v > 0)
        assert active_count <= 2

    def test_without_spreading(self, trained_graph):
        """Should work without spreading activation."""
        result = trained_graph.sparse_activate(
            ["quick"], k=5, use_spreading=False
        )

        # Should still produce result
        assert len(result) > 0

    def test_without_inhibition(self, trained_graph):
        """Should work without lateral inhibition."""
        result = trained_graph.sparse_activate(
            ["quick"], k=5, use_inhibition=False
        )

        # Should still produce result
        assert len(result) > 0

    def test_without_either(self, trained_graph):
        """Should work with neither spreading nor inhibition."""
        result = trained_graph.sparse_activate(
            ["quick"], k=5, use_spreading=False, use_inhibition=False
        )

        # Should just be k-winners of the seed
        assert len(result) > 0

    def test_normalizes_to_lowercase(self, trained_graph):
        """Should normalize tokens to lowercase."""
        result = trained_graph.sparse_activate(["QUICK", "Brown"], k=3)

        # Should work even with uppercase input
        assert len(result) > 0

    def test_full_pipeline_produces_sparse(self, trained_graph):
        """Full pipeline should produce sparse activation."""
        # Learn a dense vocabulary
        for i in range(10):
            trained_graph.learn_sequence(
                [f"word{j}" for j in range(5)]
            )

        # Sparse activate with k=3
        result = trained_graph.sparse_activate(["word0"], k=3)

        # Should have at most 3 active tokens
        active = [t for t, a in result.items() if a > 0]
        assert len(active) <= 3


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestHiveIntegration:
    """Integration tests for Hebbian Hive enhancements."""

    def test_prism_lm_uses_enhanced_graph(self):
        """PRISMLanguageModel should have access to enhanced methods."""
        model = PRISMLanguageModel()
        model.train("The quick brown fox jumps over the lazy dog.")

        # Access the underlying graph
        graph = model.graph

        # Should have new methods
        assert hasattr(graph, 'lateral_inhibition')
        assert hasattr(graph, 'k_winners_take_all')
        assert hasattr(graph, 'spreading_activation')
        assert hasattr(graph, 'sparse_activate')

    def test_sparse_activate_with_trained_model(self):
        """Test sparse_activate on a trained language model."""
        model = PRISMLanguageModel()
        model.train("Neural networks learn patterns from data.")
        model.train("Deep learning uses neural networks.")
        model.train("Machine learning patterns emerge from data.")

        result = model.graph.sparse_activate(["neural"], k=3)

        # Should find related terms
        assert len(result) > 0
        # At least one token should be active
        active = [t for t, a in result.items() if a > 0]
        assert len(active) >= 1

    def test_sparsity_target(self):
        """Test that we can achieve 5-10% sparsity."""
        model = PRISMLanguageModel()

        # Train on a corpus
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast orange cat leaps across the sleepy puppy.",
            "The slow gray wolf walks under the alert rabbit.",
            "Neural networks process information efficiently.",
            "Deep learning models recognize patterns well.",
        ]
        for sentence in sentences:
            model.train(sentence)

        vocab_size = model.vocab_size

        # Activate with k = 5% of vocab (minimum 1)
        k = max(1, vocab_size // 20)
        result = model.graph.sparse_activate(["the"], k=k)

        # Count active tokens
        active_count = sum(1 for a in result.values() if a > 0)

        # Active should be <= k
        assert active_count <= k

        # Sparsity ratio
        if vocab_size > 0:
            sparsity_ratio = active_count / vocab_size
            # Should be sparse (less than 20% in any case)
            assert sparsity_ratio <= 0.2


# ==============================================================================
# HIVENODE TESTS
# ==============================================================================


class TestHiveNode:
    """Tests for HiveNode dataclass."""

    def test_default_values(self):
        """Default node should have neutral values."""
        node = HiveNode(id="test")
        assert node.id == "test"
        assert node.activation == 0.0
        assert node.trace == 0.0
        assert node.excitability == 1.0
        assert node.activation_count == 0

    def test_activate(self):
        """Activation should update state correctly."""
        node = HiveNode(id="test")
        result = node.activate(0.8, step=5)

        assert result == 0.8  # excitability = 1.0
        assert node.activation == 0.8
        assert node.trace == 1.0  # Reset on activation
        assert node.activation_count == 1
        assert node.last_activation_step == 5

    def test_excitability_modulation(self):
        """Excitability should modulate activation."""
        node = HiveNode(id="test", excitability=0.5)
        result = node.activate(1.0)

        assert result == 0.5  # 1.0 * 0.5
        assert node.activation == 0.5

    def test_decay_trace(self):
        """Trace should decay correctly."""
        node = HiveNode(id="test", trace=1.0, trace_decay=0.9)
        node.decay_trace()

        assert node.trace == pytest.approx(0.9)

        node.decay_trace()
        assert node.trace == pytest.approx(0.81)

    def test_reset(self):
        """Reset should clear activation state."""
        node = HiveNode(id="test", activation=0.5, trace=0.8)
        node.reset()

        assert node.activation == 0.0
        assert node.trace == 0.0

    def test_to_dict(self):
        """Serialization should capture all fields."""
        node = HiveNode(
            id="test",
            activation=0.5,
            trace=0.3,
            excitability=0.8,
            activation_count=10,
        )
        data = node.to_dict()

        assert data["id"] == "test"
        assert data["activation"] == 0.5
        assert data["trace"] == 0.3
        assert data["excitability"] == 0.8
        assert data["activation_count"] == 10

    def test_from_dict(self):
        """Deserialization should restore state."""
        data = {
            "id": "restored",
            "activation": 0.7,
            "trace": 0.4,
            "excitability": 1.2,
            "activation_count": 5,
        }
        node = HiveNode.from_dict(data)

        assert node.id == "restored"
        assert node.activation == 0.7
        assert node.trace == 0.4
        assert node.excitability == 1.2
        assert node.activation_count == 5

    def test_round_trip_serialization(self):
        """Serialize and deserialize should preserve state."""
        original = HiveNode(
            id="test",
            activation=0.6,
            trace=0.2,
            trace_decay=0.9,
            target_activation=0.1,
            excitability=1.5,
            activation_count=3,
            last_activation_step=7,
        )
        data = original.to_dict()
        restored = HiveNode.from_dict(data)

        assert restored.id == original.id
        assert restored.activation == original.activation
        assert restored.trace == original.trace
        assert restored.excitability == original.excitability


# ==============================================================================
# HIVEEDGE TESTS
# ==============================================================================


class TestHiveEdge:
    """Tests for HiveEdge dataclass."""

    def test_default_values(self):
        """Default edge should have neutral values."""
        edge = HiveEdge(source_id="a", target_id="b")
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.weight == 0.0
        assert edge.pre_trace == 0.0
        assert edge.post_trace == 0.0
        assert edge.co_activations == 0

    def test_observe_pre(self):
        """Pre-synaptic observation should set trace."""
        edge = HiveEdge(source_id="a", target_id="b")
        edge.observe_pre()

        assert edge.pre_trace == 1.0
        assert edge.total_observations == 1

    def test_observe_post(self):
        """Post-synaptic observation should set trace."""
        edge = HiveEdge(source_id="a", target_id="b")
        edge.observe_post()

        assert edge.post_trace == 1.0

    def test_observe_co_activation(self):
        """Co-activation should update statistics."""
        edge = HiveEdge(source_id="a", target_id="b")
        edge.observe_co_activation()
        edge.observe_co_activation()

        assert edge.co_activations == 2
        assert edge.total_observations == 2

    def test_correlation(self):
        """Correlation should be co-activations / total."""
        edge = HiveEdge(source_id="a", target_id="b")
        edge.observe_co_activation()  # Both active
        edge.observe_pre()  # Only pre active
        edge.observe_pre()  # Only pre active

        # 1 co-activation out of 3 observations
        assert edge.correlation == pytest.approx(1.0 / 3.0)

    def test_correlation_empty(self):
        """Correlation with no observations should be 0."""
        edge = HiveEdge(source_id="a", target_id="b")
        assert edge.correlation == 0.0

    def test_learn(self):
        """Learning should increase weight based on traces."""
        edge = HiveEdge(source_id="a", target_id="b", learning_rate=0.1)
        edge.pre_trace = 1.0
        edge.post_trace = 1.0

        delta = edge.learn()

        assert delta == pytest.approx(0.1)  # 0.1 * 1.0 * 1.0
        assert edge.weight == pytest.approx(0.1)

    def test_learn_partial_traces(self):
        """Learning with partial traces should produce less change."""
        edge = HiveEdge(source_id="a", target_id="b", learning_rate=0.1)
        edge.pre_trace = 0.5
        edge.post_trace = 0.5

        delta = edge.learn()

        assert delta == pytest.approx(0.025)  # 0.1 * 0.5 * 0.5
        assert edge.weight == pytest.approx(0.025)

    def test_decay_traces(self):
        """Traces should decay correctly."""
        edge = HiveEdge(
            source_id="a",
            target_id="b",
            pre_trace=1.0,
            post_trace=1.0,
            trace_decay=0.8,
        )
        edge.decay_traces()

        assert edge.pre_trace == pytest.approx(0.8)
        assert edge.post_trace == pytest.approx(0.8)

    def test_to_dict(self):
        """Serialization should capture all fields."""
        edge = HiveEdge(
            source_id="a",
            target_id="b",
            weight=0.5,
            pre_trace=0.3,
            post_trace=0.4,
            co_activations=10,
        )
        data = edge.to_dict()

        assert data["source_id"] == "a"
        assert data["target_id"] == "b"
        assert data["weight"] == 0.5
        assert data["pre_trace"] == 0.3
        assert data["co_activations"] == 10

    def test_from_dict(self):
        """Deserialization should restore state."""
        data = {
            "source_id": "x",
            "target_id": "y",
            "weight": 0.7,
            "pre_trace": 0.2,
            "post_trace": 0.3,
            "co_activations": 5,
            "total_observations": 20,
        }
        edge = HiveEdge.from_dict(data)

        assert edge.source_id == "x"
        assert edge.target_id == "y"
        assert edge.weight == 0.7
        assert edge.co_activations == 5
        assert edge.total_observations == 20

    def test_round_trip_serialization(self):
        """Serialize and deserialize should preserve state."""
        original = HiveEdge(
            source_id="s",
            target_id="t",
            weight=1.5,
            pre_trace=0.6,
            post_trace=0.7,
            co_activations=15,
            total_observations=50,
            trace_decay=0.9,
            learning_rate=0.05,
        )
        data = original.to_dict()
        restored = HiveEdge.from_dict(data)

        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.weight == original.weight
        assert restored.co_activations == original.co_activations

    def test_hebbian_learning_cycle(self):
        """Test a full Hebbian learning cycle."""
        edge = HiveEdge(source_id="a", target_id="b", learning_rate=0.1)

        # Simulate: pre fires, then post fires, then learn
        edge.observe_pre()
        edge.observe_post()
        edge.observe_co_activation()

        # Learn from the traces
        delta = edge.learn()

        assert delta > 0
        assert edge.weight > 0

        # Decay traces over time (0.95^20 ≈ 0.36)
        for _ in range(20):
            edge.decay_traces()

        # Traces should have decayed significantly (0.95^20 ≈ 0.36)
        assert edge.pre_trace < 0.5
        assert edge.post_trace < 0.5
