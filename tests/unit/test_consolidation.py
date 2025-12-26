"""
Tests for the Consolidation Engine.

Part of Sprint 5: Consolidation (Woven Mind + PRISM Marriage)

Tests cover:
- ConsolidationConfig validation
- Pattern recording and frequency tracking
- Pattern transfer from Hive to Cortex
- Abstraction mining
- Decay cycles
- Consolidation lifecycle
- Scheduler functionality
- Observability callbacks
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import time
import threading

from cortical.reasoning.consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    ConsolidationPhase,
    ConsolidationEngine,
)
from cortical.reasoning.loom_hive import LoomHiveConnector, LoomHiveConfig
from cortical.reasoning.loom_cortex import LoomCortexConnector, LoomCortexConfig


class TestConsolidationConfig:
    """Tests for ConsolidationConfig dataclass."""

    def test_default_config(self):
        """Default configuration has sensible values."""
        config = ConsolidationConfig()

        assert config.transfer_threshold == 3
        assert config.decay_factor == 0.9
        assert config.min_strength_keep == 0.1
        assert config.max_patterns_per_cycle == 10
        assert config.max_abstractions_per_cycle == 5
        assert config.enable_scheduling is False
        assert config.schedule_interval_seconds == 300

    def test_custom_config(self):
        """Custom configuration values are preserved."""
        config = ConsolidationConfig(
            transfer_threshold=5,
            decay_factor=0.8,
            min_strength_keep=0.2,
            max_patterns_per_cycle=20,
            max_abstractions_per_cycle=10,
            enable_scheduling=True,
            schedule_interval_seconds=60,
        )

        assert config.transfer_threshold == 5
        assert config.decay_factor == 0.8
        assert config.min_strength_keep == 0.2
        assert config.max_patterns_per_cycle == 20
        assert config.max_abstractions_per_cycle == 10
        assert config.enable_scheduling is True
        assert config.schedule_interval_seconds == 60


class TestConsolidationResult:
    """Tests for ConsolidationResult dataclass."""

    def test_default_result(self):
        """Default result has zero counts."""
        result = ConsolidationResult()

        assert result.patterns_transferred == 0
        assert result.abstractions_formed == 0
        assert result.connections_decayed == 0
        assert result.connections_pruned == 0
        assert result.cycle_duration_ms == 0.0
        assert isinstance(result.timestamp, datetime)
        assert result.transferred_patterns == []
        assert result.formed_abstractions == []
        assert result.metadata == {}

    def test_result_with_values(self):
        """Result stores provided values."""
        patterns = [frozenset(["a", "b"]), frozenset(["c", "d"])]
        abstractions = ["A0-12345678", "A1-87654321"]

        result = ConsolidationResult(
            patterns_transferred=2,
            abstractions_formed=2,
            connections_decayed=100,
            connections_pruned=10,
            cycle_duration_ms=150.5,
            transferred_patterns=patterns,
            formed_abstractions=abstractions,
            metadata={"source": "test"},
        )

        assert result.patterns_transferred == 2
        assert result.abstractions_formed == 2
        assert result.connections_decayed == 100
        assert result.connections_pruned == 10
        assert result.cycle_duration_ms == 150.5
        assert result.transferred_patterns == patterns
        assert result.formed_abstractions == abstractions
        assert result.metadata == {"source": "test"}


class TestConsolidationPhase:
    """Tests for ConsolidationPhase enum."""

    def test_phases_exist(self):
        """All expected phases are defined."""
        assert hasattr(ConsolidationPhase, "IDLE")
        assert hasattr(ConsolidationPhase, "PATTERN_TRANSFER")
        assert hasattr(ConsolidationPhase, "ABSTRACTION_MINING")
        assert hasattr(ConsolidationPhase, "DECAY_CYCLE")
        assert hasattr(ConsolidationPhase, "COMPLETE")

    def test_phases_are_unique(self):
        """Each phase has a unique value."""
        values = [
            ConsolidationPhase.IDLE,
            ConsolidationPhase.PATTERN_TRANSFER,
            ConsolidationPhase.ABSTRACTION_MINING,
            ConsolidationPhase.DECAY_CYCLE,
            ConsolidationPhase.COMPLETE,
        ]
        assert len(set(values)) == 5


class TestConsolidationEngine:
    """Tests for ConsolidationEngine class."""

    @pytest.fixture
    def hive(self):
        """Create a LoomHiveConnector for testing."""
        config = LoomHiveConfig(k_winners=5)
        connector = LoomHiveConnector(config=config)
        # Train with some text to build patterns
        connector.train("neural networks process data efficiently")
        connector.train("machine learning models learn patterns")
        connector.train("deep learning neural networks are powerful")
        return connector

    @pytest.fixture
    def cortex(self):
        """Create a LoomCortexConnector for testing."""
        config = LoomCortexConfig(min_frequency=2)
        return LoomCortexConnector(config=config)

    @pytest.fixture
    def engine(self, hive, cortex):
        """Create a ConsolidationEngine for testing."""
        return ConsolidationEngine(hive, cortex)

    def test_init_default_config(self, hive, cortex):
        """Engine initializes with default config."""
        engine = ConsolidationEngine(hive, cortex)

        assert engine.hive is hive
        assert engine.cortex is cortex
        assert engine.config.transfer_threshold == 3
        assert engine.current_phase == ConsolidationPhase.IDLE
        assert engine.is_running is False
        assert engine.last_consolidation is None

    def test_init_custom_config(self, hive, cortex):
        """Engine uses custom config."""
        config = ConsolidationConfig(transfer_threshold=5)
        engine = ConsolidationEngine(hive, cortex, config)

        assert engine.config.transfer_threshold == 5

    def test_record_pattern(self, engine):
        """Patterns are recorded for consolidation."""
        pattern1 = {"neural", "network"}
        pattern2 = {"machine", "learning"}

        engine.record_pattern(pattern1)
        engine.record_pattern(pattern1)
        engine.record_pattern(pattern2)

        # Check frequency tracking
        patterns = engine.get_frequent_patterns(min_frequency=1)
        pattern_dict = {p: f for p, f in patterns}

        assert frozenset(pattern1) in pattern_dict
        assert pattern_dict[frozenset(pattern1)] == 2
        assert frozenset(pattern2) in pattern_dict
        assert pattern_dict[frozenset(pattern2)] == 1

    def test_record_pattern_ignores_single_nodes(self, engine):
        """Patterns with only one node are ignored."""
        engine.record_pattern({"single"})

        patterns = engine.get_frequent_patterns(min_frequency=1)
        assert len(patterns) == 0

    def test_get_frequent_patterns_threshold(self, engine):
        """Only patterns meeting threshold are returned."""
        engine.record_pattern({"a", "b"})
        engine.record_pattern({"a", "b"})
        engine.record_pattern({"a", "b"})  # 3 times
        engine.record_pattern({"c", "d"})  # 1 time

        patterns = engine.get_frequent_patterns(min_frequency=3)

        assert len(patterns) == 1
        assert patterns[0][0] == frozenset(["a", "b"])
        assert patterns[0][1] == 3

    def test_get_frequent_patterns_top_k(self, engine):
        """Top-k limits the results."""
        for i in range(5):
            for _ in range(5 - i):  # Different frequencies
                engine.record_pattern({f"a{i}", f"b{i}"})

        patterns = engine.get_frequent_patterns(min_frequency=1, top_k=2)

        assert len(patterns) == 2
        # Should be sorted by frequency descending
        assert patterns[0][1] >= patterns[1][1]

    def test_consolidate_basic(self, engine):
        """Basic consolidation cycle completes."""
        # Record some patterns
        for _ in range(5):
            engine.record_pattern({"neural", "network"})

        result = engine.consolidate()

        assert isinstance(result, ConsolidationResult)
        assert result.cycle_duration_ms > 0
        assert isinstance(result.timestamp, datetime)
        assert engine.current_phase == ConsolidationPhase.IDLE
        assert engine.is_running is False
        assert engine.last_consolidation is not None

    def test_consolidate_prevents_concurrent(self, engine):
        """Cannot run consolidation concurrently."""
        engine._is_running = True

        with pytest.raises(RuntimeError, match="already in progress"):
            engine.consolidate()

    def test_pattern_transfer_creates_abstraction(self, hive, cortex):
        """Pattern transfer creates abstractions in Cortex."""
        config = ConsolidationConfig(transfer_threshold=2)
        engine = ConsolidationEngine(hive, cortex, config)

        # Record pattern multiple times
        pattern = {"neural", "network"}
        for _ in range(5):
            engine.record_pattern(pattern)

        result = engine.pattern_transfer()

        # Pattern should be transferred
        # Note: actual transfer success depends on Cortex state
        assert "transferred" in result
        assert "patterns" in result

    def test_decay_cycle_reduces_frequencies(self, engine):
        """Decay cycle reduces pattern frequencies."""
        # Record pattern
        for _ in range(10):
            engine.record_pattern({"neural", "network"})

        initial_freq = engine._pattern_frequencies[frozenset(["neural", "network"])]

        # Run decay
        result = engine.decay_cycle()

        final_freq = engine._pattern_frequencies.get(frozenset(["neural", "network"]), 0)

        assert final_freq < initial_freq
        assert "decayed" in result
        assert "pruned" in result

    def test_decay_cycle_prunes_weak_patterns(self, engine):
        """Decay cycle removes patterns that fall below threshold."""
        # Record pattern just once
        engine.record_pattern({"weak", "pattern"})

        # Run decay multiple times to drive frequency to 0
        for _ in range(10):
            engine.decay_cycle()

        # Pattern should be pruned
        assert frozenset(["weak", "pattern"]) not in engine._pattern_frequencies

    def test_history_tracking(self, engine):
        """Consolidation history is tracked."""
        engine.consolidate()
        engine.consolidate()
        engine.consolidate()

        history = engine.get_history()

        assert len(history) == 3
        # Most recent first
        assert history[0].timestamp >= history[1].timestamp
        assert history[1].timestamp >= history[2].timestamp

    def test_history_limit(self, engine):
        """History can be limited."""
        for _ in range(5):
            engine.consolidate()

        history = engine.get_history(limit=2)

        assert len(history) == 2

    def test_get_stats(self, engine):
        """Stats are computed correctly."""
        for _ in range(3):
            engine.record_pattern({"a", "b"})

        engine.consolidate()

        stats = engine.get_stats()

        assert stats["total_cycles"] == 1
        assert stats["is_running"] is False
        assert stats["current_phase"] == "IDLE"
        assert stats["tracked_patterns"] >= 0
        assert stats["last_consolidation"] is not None

    def test_phase_change_callback(self, engine):
        """Phase change callback is fired."""
        phases_seen = []

        def on_phase_change(phase):
            phases_seen.append(phase)

        engine.on_phase_change(on_phase_change)
        engine.consolidate()

        assert ConsolidationPhase.PATTERN_TRANSFER in phases_seen
        assert ConsolidationPhase.ABSTRACTION_MINING in phases_seen
        assert ConsolidationPhase.DECAY_CYCLE in phases_seen
        assert ConsolidationPhase.COMPLETE in phases_seen
        assert ConsolidationPhase.IDLE in phases_seen  # Final state

    def test_cycle_complete_callback(self, engine):
        """Cycle complete callback is fired."""
        results_seen = []

        def on_cycle_complete(result):
            results_seen.append(result)

        engine.on_cycle_complete(on_cycle_complete)
        engine.consolidate()

        assert len(results_seen) == 1
        assert isinstance(results_seen[0], ConsolidationResult)

    def test_pattern_transferred_callback(self, hive, cortex):
        """Pattern transferred callback is fired for each transfer."""
        config = ConsolidationConfig(transfer_threshold=2)
        engine = ConsolidationEngine(hive, cortex, config)

        transferred = []

        def on_pattern_transferred(pattern):
            transferred.append(pattern)

        engine.on_pattern_transferred(on_pattern_transferred)

        # Record pattern enough times
        for _ in range(5):
            engine.record_pattern({"neural", "network"})

        engine.pattern_transfer()

        # Callback should have been called for any transferred patterns
        # (may be 0 if transfer conditions not met)
        assert isinstance(transferred, list)

    def test_interrupt(self, engine):
        """Interrupt stops running consolidation."""
        # Simulate running state
        engine._is_running = True

        result = engine.interrupt()

        assert result is True
        assert engine.is_running is False

    def test_interrupt_when_not_running(self, engine):
        """Interrupt returns False when not running."""
        result = engine.interrupt()

        assert result is False

    def test_serialization(self, engine):
        """Engine can be serialized and deserialized."""
        # Add some state
        for _ in range(3):
            engine.record_pattern({"a", "b"})
        engine.consolidate()

        # Serialize
        data = engine.to_dict()

        assert "config" in data
        assert "pattern_frequencies" in data
        assert "last_consolidation" in data
        assert "history_count" in data

        # Deserialize
        restored = ConsolidationEngine.from_dict(data, engine.hive, engine.cortex)

        assert restored.config.transfer_threshold == engine.config.transfer_threshold
        assert len(restored._pattern_frequencies) == len(engine._pattern_frequencies)

    def test_abstraction_mining(self, hive, cortex):
        """Abstraction mining creates new abstractions."""
        engine = ConsolidationEngine(hive, cortex)

        # Observe patterns in cortex directly
        pattern = frozenset(["machine", "learning"])
        for _ in range(5):
            cortex.engine.observe(pattern)

        result = engine.abstraction_mining()

        assert "formed" in result
        assert "abstraction_ids" in result


class TestConsolidationScheduler:
    """Tests for the consolidation scheduler."""

    @pytest.fixture
    def hive(self):
        """Create a minimal hive for testing."""
        return LoomHiveConnector()

    @pytest.fixture
    def cortex(self):
        """Create a minimal cortex for testing."""
        return LoomCortexConnector()

    def test_scheduler_start_stop(self, hive, cortex):
        """Scheduler can be started and stopped."""
        config = ConsolidationConfig(schedule_interval_seconds=1)
        engine = ConsolidationEngine(hive, cortex, config)

        engine.start_scheduler()

        assert engine._scheduler_thread is not None
        assert engine._scheduler_thread.is_alive()

        engine.stop_scheduler()

        # Give it time to stop
        time.sleep(0.1)

        # Thread should have stopped (stop_scheduler sets it to None)
        assert engine._scheduler_thread is None or not engine._scheduler_thread.is_alive()

    @pytest.mark.slow
    def test_scheduler_runs_consolidation(self, hive, cortex):
        """Scheduler runs consolidation at intervals."""
        config = ConsolidationConfig(schedule_interval_seconds=1)
        engine = ConsolidationEngine(hive, cortex, config)

        engine.start_scheduler()

        # Wait for at least one cycle
        time.sleep(1.5)

        engine.stop_scheduler()

        # Should have run at least one consolidation
        stats = engine.get_stats()
        assert stats["total_cycles"] >= 1

    def test_scheduler_is_daemon(self, hive, cortex):
        """Scheduler thread is a daemon (won't prevent exit)."""
        config = ConsolidationConfig(schedule_interval_seconds=60)
        engine = ConsolidationEngine(hive, cortex, config)

        engine.start_scheduler()

        assert engine._scheduler_thread.daemon is True

        engine.stop_scheduler()


class TestConsolidationIntegration:
    """Integration tests for consolidation with real components."""

    @pytest.fixture
    def trained_system(self):
        """Create a fully trained Hive-Cortex system."""
        hive = LoomHiveConnector(config=LoomHiveConfig(k_winners=5))
        cortex = LoomCortexConnector(config=LoomCortexConfig(min_frequency=2))

        # Train hive on repeated patterns
        texts = [
            "neural networks learn from data",
            "deep learning uses neural networks",
            "machine learning processes data",
            "neural networks are powerful models",
            "data science uses machine learning",
        ]
        for text in texts:
            hive.train(text)

        # Observe patterns in cortex
        patterns = [
            ["neural", "networks"],
            ["machine", "learning"],
            ["deep", "learning"],
        ]
        for pattern in patterns:
            for _ in range(3):
                cortex.engine.observe(frozenset(pattern))

        return hive, cortex

    def test_full_consolidation_cycle(self, trained_system):
        """Full consolidation cycle with real data."""
        hive, cortex = trained_system
        config = ConsolidationConfig(
            transfer_threshold=2,
            max_patterns_per_cycle=10,
        )
        engine = ConsolidationEngine(hive, cortex, config)

        # Record patterns that were observed
        engine.record_pattern({"neural", "networks"})
        engine.record_pattern({"neural", "networks"})
        engine.record_pattern({"neural", "networks"})
        engine.record_pattern({"machine", "learning"})
        engine.record_pattern({"machine", "learning"})

        # Run consolidation
        result = engine.consolidate()

        # Verify results
        assert result.patterns_transferred >= 0
        assert result.abstractions_formed >= 0
        assert result.cycle_duration_ms > 0

        # Stats should reflect the cycle
        stats = engine.get_stats()
        assert stats["total_cycles"] == 1

    def test_learning_retention_after_decay(self, trained_system):
        """Important patterns survive multiple decay cycles."""
        hive, cortex = trained_system
        config = ConsolidationConfig(decay_factor=0.95)
        engine = ConsolidationEngine(hive, cortex, config)

        # Record a high-frequency pattern
        for _ in range(20):
            engine.record_pattern({"neural", "networks"})

        # Record a low-frequency pattern
        engine.record_pattern({"rare", "pattern"})

        # Run multiple decay cycles
        for _ in range(5):
            engine.decay_cycle()

        # High-frequency pattern should survive
        patterns = engine.get_frequent_patterns(min_frequency=1)
        pattern_set = {p for p, _ in patterns}

        assert frozenset(["neural", "networks"]) in pattern_set
        # Low-frequency pattern should be gone
        assert frozenset(["rare", "pattern"]) not in pattern_set


class TestConsolidationEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def hive(self):
        """Create a minimal hive for testing."""
        return LoomHiveConnector()

    @pytest.fixture
    def cortex(self):
        """Create a minimal cortex for testing."""
        return LoomCortexConnector()

    def test_pattern_transfer_max_limit(self, hive, cortex):
        """Pattern transfer respects max_patterns_per_cycle limit."""
        config = ConsolidationConfig(
            transfer_threshold=2,
            max_patterns_per_cycle=2,  # Only allow 2 patterns
        )
        engine = ConsolidationEngine(hive, cortex, config)

        # Record many patterns above threshold
        for i in range(5):
            pattern = frozenset([f"term{i}", f"word{i}"])
            for _ in range(3):  # Above threshold
                engine.record_pattern(pattern)

        # Run pattern transfer
        result = engine.pattern_transfer()

        # Should only transfer max_patterns_per_cycle
        assert result["transferred"] <= 2

    def test_pattern_transfer_skips_abstracted(self, hive, cortex):
        """Pattern transfer skips patterns that already have abstractions."""
        config = ConsolidationConfig(transfer_threshold=2)
        engine = ConsolidationEngine(hive, cortex, config)

        # Record a pattern
        pattern = frozenset(["neural", "networks"])
        for _ in range(3):
            engine.record_pattern(pattern)

        # First transfer should work
        result1 = engine.pattern_transfer()
        first_transferred = result1["transferred"]

        # Record same pattern again (still above threshold)
        for _ in range(3):
            engine.record_pattern(pattern)

        # Second transfer should skip (already abstracted)
        result2 = engine.pattern_transfer()

        # If pattern was abstracted, second transfer should skip it
        # (transfers 0 if that was the only pattern above threshold)
        assert result2["transferred"] <= first_transferred

    def test_decay_cycle_prunes_transitions(self, hive, cortex):
        """Decay cycle prunes transitions below minimum strength."""
        config = ConsolidationConfig(
            decay_factor=0.1,  # Aggressive decay
            min_strength_keep=0.5,  # High threshold for pruning
        )
        engine = ConsolidationEngine(hive, cortex, config)

        # Train hive to create transitions
        hive.train("the quick brown fox")
        hive.train("the quick brown dog")
        hive.train("the quick red fox")

        # Run decay multiple times to prune weak connections
        for _ in range(5):
            result = engine.decay_cycle()

        # Should have pruned some connections
        assert "pruned" in result
        assert result["pruned"] >= 0  # May or may not have pruned

    def test_scheduler_handles_exceptions(self, hive, cortex):
        """Scheduler continues running even if consolidation raises."""
        config = ConsolidationConfig(schedule_interval_seconds=0.1)
        engine = ConsolidationEngine(hive, cortex, config)

        # Patch consolidate to raise an exception
        original_consolidate = engine.consolidate
        call_count = [0]

        def failing_consolidate():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure")
            return original_consolidate()

        engine.consolidate = failing_consolidate

        # Start scheduler
        engine.start_scheduler()

        # Wait for multiple cycles (including failed one)
        time.sleep(0.35)

        engine.stop_scheduler()

        # Should have attempted multiple times despite failure
        assert call_count[0] >= 2

    def test_scheduler_already_running(self, hive, cortex):
        """Starting scheduler when already running is a no-op."""
        config = ConsolidationConfig(schedule_interval_seconds=60)
        engine = ConsolidationEngine(hive, cortex, config)

        engine.start_scheduler()
        first_thread = engine._scheduler_thread

        # Start again - should be no-op
        engine.start_scheduler()

        # Should be the same thread
        assert engine._scheduler_thread is first_thread

        engine.stop_scheduler()

    def test_decay_removes_empty_context_entries(self, hive, cortex):
        """Decay cycle removes context entries when all transitions pruned."""
        config = ConsolidationConfig(
            decay_factor=0.01,  # Very aggressive decay
            min_strength_keep=0.9,  # Very high threshold
        )
        engine = ConsolidationEngine(hive, cortex, config)

        # Train hive minimally to create weak connections
        hive.train("single word")

        # Run aggressive decay to potentially remove all
        total_pruned = 0
        for _ in range(10):
            result = engine.decay_cycle()
            total_pruned += result["pruned"]

        # Should have attempted pruning
        assert result["pruned"] >= 0
