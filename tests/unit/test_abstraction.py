"""
Unit tests for the Cortex Abstraction System.

Tests cover:
- Abstraction dataclass
- PatternDetector pattern discovery
- AbstractionEngine abstraction formation
- Hierarchical level computation
- Truth value propagation

Part of Sprint 3: Cortex Abstraction (Woven Mind + PRISM Marriage)
"""

import pytest
from datetime import datetime
from cortical.reasoning.abstraction import (
    Abstraction,
    PatternObservation,
    PatternDetector,
    AbstractionEngine,
)


# ==============================================================================
# ABSTRACTION DATACLASS TESTS
# ==============================================================================


class TestAbstraction:
    """Tests for Abstraction dataclass."""

    def test_default_values(self):
        """Default abstraction should have neutral values."""
        abstraction = Abstraction(
            id="A1-test",
            source_nodes=frozenset(["a", "b"]),
            level=1,
            frequency=5,
        )
        assert abstraction.id == "A1-test"
        assert abstraction.level == 1
        assert abstraction.frequency == 5
        assert abstraction.truth_value == 0.5

    def test_strength_computation(self):
        """Strength should increase with frequency."""
        low_freq = Abstraction(
            id="A1-low", source_nodes=frozenset(["a"]),
            level=1, frequency=1,
        )
        high_freq = Abstraction(
            id="A1-high", source_nodes=frozenset(["a"]),
            level=1, frequency=100,
        )

        assert high_freq.strength > low_freq.strength

    def test_update_truth(self):
        """Truth update should recompute strength."""
        abstraction = Abstraction(
            id="A1-test", source_nodes=frozenset(["a"]),
            level=1, frequency=10,
        )
        old_strength = abstraction.strength

        abstraction.update_truth(0.9)
        assert abstraction.truth_value == 0.9
        assert abstraction.strength > old_strength

    def test_update_truth_clamped(self):
        """Truth values should be clamped to [0, 1]."""
        abstraction = Abstraction(
            id="A1-test", source_nodes=frozenset(["a"]),
            level=1, frequency=1,
        )

        abstraction.update_truth(1.5)
        assert abstraction.truth_value == 1.0

        abstraction.update_truth(-0.5)
        assert abstraction.truth_value == 0.0

    def test_observe_increases_frequency(self):
        """Observe should increment frequency and update strength."""
        abstraction = Abstraction(
            id="A1-test", source_nodes=frozenset(["a"]),
            level=1, frequency=5,
        )
        old_freq = abstraction.frequency
        old_strength = abstraction.strength

        abstraction.observe()

        assert abstraction.frequency == old_freq + 1
        assert abstraction.strength > old_strength

    def test_serialization_roundtrip(self):
        """Serialize and deserialize should preserve state."""
        original = Abstraction(
            id="A2-test",
            source_nodes=frozenset(["x", "y", "z"]),
            level=2,
            frequency=15,
            truth_value=0.8,
        )

        data = original.to_dict()
        restored = Abstraction.from_dict(data)

        assert restored.id == original.id
        assert restored.source_nodes == original.source_nodes
        assert restored.level == original.level
        assert restored.frequency == original.frequency
        assert restored.truth_value == pytest.approx(original.truth_value)


# ==============================================================================
# PATTERN DETECTOR TESTS
# ==============================================================================


class TestPatternDetector:
    """Tests for PatternDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a default pattern detector."""
        return PatternDetector(min_frequency=3)

    def test_observe_tracks_pattern(self, detector):
        """Observing a pattern should track it."""
        pattern = frozenset(["a", "b", "c"])
        detector.observe(pattern)

        freq = detector.get_pattern_frequency(pattern)
        assert freq >= 1.0

    def test_observe_returns_candidates(self, detector):
        """After min_frequency observations, pattern is candidate."""
        pattern = frozenset(["a", "b", "c"])

        # First two observations - not yet a candidate
        candidates = detector.observe(pattern)
        assert pattern not in candidates
        candidates = detector.observe(pattern)
        assert pattern not in candidates

        # Third observation - now a candidate
        candidates = detector.observe(pattern)
        assert pattern in candidates

    def test_get_candidates_sorted(self, detector):
        """Candidates should be sorted by frequency."""
        pattern1 = frozenset(["a", "b"])
        pattern2 = frozenset(["x", "y"])

        # Observe pattern1 more than pattern2
        for _ in range(5):
            detector.observe(pattern1)
        for _ in range(3):
            detector.observe(pattern2)

        candidates = detector.get_candidates()
        # pattern1 should be first (higher frequency)
        assert candidates[0][0] == pattern1
        assert candidates[0][1] > candidates[1][1]

    def test_min_pattern_size(self):
        """Patterns smaller than min_size should be ignored."""
        detector = PatternDetector(min_pattern_size=3)

        small_pattern = frozenset(["a", "b"])  # Only 2 elements
        for _ in range(10):
            detector.observe(small_pattern)

        freq = detector.get_pattern_frequency(small_pattern)
        assert freq == 0

    def test_max_pattern_size(self):
        """Patterns larger than max_size should be ignored."""
        detector = PatternDetector(max_pattern_size=3)

        large_pattern = frozenset(["a", "b", "c", "d", "e"])  # 5 elements
        for _ in range(10):
            detector.observe(large_pattern)

        freq = detector.get_pattern_frequency(large_pattern)
        assert freq == 0

    def test_apply_decay(self, detector):
        """Decay should reduce pattern frequencies."""
        pattern = frozenset(["a", "b", "c"])

        for _ in range(10):
            detector.observe(pattern)

        freq_before = detector.get_pattern_frequency(pattern)
        detector.apply_decay()
        freq_after = detector.get_pattern_frequency(pattern)

        assert freq_after < freq_before

    def test_clear(self, detector):
        """Clear should remove all tracked patterns."""
        pattern = frozenset(["a", "b", "c"])
        detector.observe(pattern)

        detector.clear()

        assert detector.get_pattern_frequency(pattern) == 0
        assert len(detector.get_candidates()) == 0

    def test_subset_tracking(self, detector):
        """Subsets of larger patterns should get partial credit."""
        large_pattern = frozenset(["a", "b", "c", "d"])

        for _ in range(5):
            detector.observe(large_pattern)

        # Subsets should have some frequency
        subset = frozenset(["a", "b", "c"])
        freq = detector.get_pattern_frequency(subset)
        assert freq > 0


# ==============================================================================
# ABSTRACTION ENGINE TESTS
# ==============================================================================


class TestAbstractionEngine:
    """Tests for AbstractionEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a default abstraction engine."""
        return AbstractionEngine(min_frequency=3)

    def test_observe_creates_candidate(self, engine):
        """Repeated observation should create candidates."""
        pattern = frozenset(["a", "b", "c"])

        for _ in range(5):
            engine.observe(pattern)

        candidates = engine.abstraction_candidates()
        patterns = [c[0] for c in candidates]
        assert pattern in patterns

    def test_form_abstraction(self, engine):
        """form_abstraction should create an Abstraction."""
        pattern = frozenset(["a", "b", "c"])

        for _ in range(5):
            engine.observe(pattern)

        abstraction = engine.form_abstraction(pattern)

        assert abstraction is not None
        assert abstraction.source_nodes == pattern
        assert abstraction.id in engine.abstractions

    def test_form_abstraction_prevents_duplicates(self, engine):
        """Cannot form abstraction twice for same pattern."""
        pattern = frozenset(["a", "b", "c"])

        for _ in range(5):
            engine.observe(pattern)

        abstraction1 = engine.form_abstraction(pattern)
        abstraction2 = engine.form_abstraction(pattern)

        assert abstraction1 is not None
        assert abstraction2 is None  # Duplicate rejected

    def test_form_abstraction_min_size(self, engine):
        """Pattern must have at least 2 nodes."""
        small_pattern = frozenset(["a"])
        abstraction = engine.form_abstraction(small_pattern)
        assert abstraction is None

    def test_hierarchical_levels(self, engine):
        """Higher-level abstractions should be placed correctly."""
        # Create base-level abstraction
        pattern1 = frozenset(["a", "b"])
        for _ in range(5):
            engine.observe(pattern1)
        abs1 = engine.form_abstraction(pattern1)

        # Create abstraction that includes the first one
        pattern2 = frozenset([abs1.id, "c"])
        abs2 = engine.form_abstraction(pattern2)

        assert abs2 is not None
        assert abs2.level > abs1.level

    def test_auto_form_abstractions(self, engine):
        """auto_form_abstractions should create from top candidates."""
        patterns = [
            frozenset(["a", "b"]),
            frozenset(["x", "y"]),
            frozenset(["p", "q"]),
        ]

        for pattern in patterns:
            for _ in range(5):
                engine.observe(pattern)

        formed = engine.auto_form_abstractions(max_new=2)

        assert len(formed) == 2
        for abstraction in formed:
            assert abstraction.id in engine.abstractions

    def test_update_truth_values(self, engine):
        """Truth values should be updatable."""
        pattern = frozenset(["a", "b"])
        for _ in range(5):
            engine.observe(pattern)

        abstraction = engine.form_abstraction(pattern)

        engine.update_truth_values({abstraction.id: 0.9})

        assert engine.abstractions[abstraction.id].truth_value == 0.9

    def test_propagate_truth(self, engine):
        """Truth should propagate up hierarchy."""
        # Create base-level abstraction
        pattern1 = frozenset(["a", "b"])
        for _ in range(5):
            engine.observe(pattern1)
        abs1 = engine.form_abstraction(pattern1)
        engine.update_truth_values({abs1.id: 0.8})

        # Create higher-level abstraction
        pattern2 = frozenset([abs1.id, "c"])
        abs2 = engine.form_abstraction(pattern2)

        # Propagate truth
        engine.propagate_truth()

        # Higher abstraction should be influenced by component
        assert abs2.truth_value != 0.5  # Changed from default

    def test_get_level(self, engine):
        """Should retrieve all abstractions at a level."""
        pattern = frozenset(["a", "b"])
        for _ in range(5):
            engine.observe(pattern)

        abstraction = engine.form_abstraction(pattern)
        level = abstraction.level

        level_abstractions = engine.get_level(level)
        assert abstraction in level_abstractions

    def test_serialization_roundtrip(self, engine):
        """Serialize and deserialize should preserve state."""
        pattern = frozenset(["a", "b", "c"])
        for _ in range(5):
            engine.observe(pattern)
        engine.form_abstraction(pattern)

        data = engine.to_dict()
        restored = AbstractionEngine.from_dict(data)

        assert len(restored.abstractions) == len(engine.abstractions)
        assert restored.min_frequency == engine.min_frequency


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestAbstractionIntegration:
    """Integration tests for the abstraction system."""

    def test_full_abstraction_workflow(self):
        """Test complete workflow: observe → candidate → abstract."""
        engine = AbstractionEngine(min_frequency=3)

        # Simulate observing patterns
        patterns = [
            frozenset(["concept", "learn"]),
            frozenset(["concept", "understand"]),
            frozenset(["concept", "learn"]),  # Repeated
            frozenset(["concept", "learn"]),  # Repeated
            frozenset(["concept", "learn"]),  # Repeated again
        ]

        for pattern in patterns:
            engine.observe(pattern)

        # Check candidates
        candidates = engine.abstraction_candidates()
        assert len(candidates) > 0

        # Form abstraction
        top_pattern = candidates[0][0]
        abstraction = engine.form_abstraction(top_pattern)

        assert abstraction is not None
        assert abstraction.frequency >= 3

    def test_hierarchical_abstraction_building(self):
        """Test building multiple levels of abstraction."""
        engine = AbstractionEngine(min_frequency=2, max_levels=4)

        # Level 0: Create base abstractions
        base_patterns = [
            frozenset(["a", "b"]),
            frozenset(["c", "d"]),
        ]

        for pattern in base_patterns:
            for _ in range(3):
                engine.observe(pattern)

        abs1 = engine.form_abstraction(base_patterns[0])
        abs2 = engine.form_abstraction(base_patterns[1])

        # Level 1: Create meta-abstraction
        meta_pattern = frozenset([abs1.id, abs2.id])
        for _ in range(3):
            engine.observe(meta_pattern)

        meta_abs = engine.form_abstraction(meta_pattern)

        assert abs1.level < meta_abs.level
        assert abs2.level < meta_abs.level

    def test_observation_reinforces_existing(self):
        """Observing existing abstraction pattern should reinforce it."""
        engine = AbstractionEngine(min_frequency=3)

        pattern = frozenset(["neural", "network"])

        # Create abstraction
        for _ in range(5):
            engine.observe(pattern)
        abstraction = engine.form_abstraction(pattern)

        initial_freq = abstraction.frequency

        # Observe same pattern again
        engine.observe(pattern)

        assert abstraction.frequency > initial_freq
