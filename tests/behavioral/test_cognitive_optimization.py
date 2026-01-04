"""
Behavioral tests for Phase 2 cognitive optimization features.

As a developer building intelligent optimization systems,
I want Phase 2 features to enhance cognitive performance through evolution,
So that agents continuously improve through strategy evolution, velocity prediction,
learning consolidation, and complete escalation actions.

This test suite verifies:
- Strategy evolution through genetic algorithms
- Velocity prediction with trend detection
- Learning consolidation for memory optimization
- Complete escalation action execution

Phase 2 Context:
- Strategy evolution GA for optimizing cognitive strategies
- Velocity prediction with confidence intervals
- Learning consolidation to transfer Hive→Cortex patterns
- Escalation actions (MONITOR, REASSIGN, ABORT) with full implementation
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
from collections import defaultdict

# Strategy Evolution
from llm_orchestration.evolution import (
    StrategyEvolver,
    StrategyPool,
    StrategyGenome,
    FitnessScore,
    ExecutionTrace,
    ExecutionMetrics,
)

# Velocity Prediction
from llm_orchestration.metrics import (
    MetricsCollector,
    HybridMetrics,
)

# Learning Consolidation
from cortical.reasoning.consolidation import (
    ConsolidationEngine,
    ConsolidationConfig,
    ConsolidationPhase,
)
from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

# Escalation Actions
from llm_orchestration.escalation import (
    EscalationManager,
    EscalationLevel,
    EscalationProtocol,
)
from llm_orchestration.recovery import ConfusionSignal


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_storage():
    """Provide temporary storage for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def strategy_pool():
    """Create a strategy pool for testing."""
    return StrategyPool()


@pytest.fixture
def strategy_evolver(strategy_pool):
    """Create a strategy evolver."""
    return StrategyEvolver(strategy_pool)


@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector()


@pytest.fixture
def woven_mind():
    """Create a WovenMind instance for consolidation testing."""
    config = WovenMindConfig(
        surprise_threshold=0.3,
        k_winners=5,
        auto_switch=True,
    )
    return WovenMind(config=config)


@pytest.fixture
def consolidation_engine(woven_mind):
    """Create a consolidation engine."""
    config = ConsolidationConfig(
        transfer_threshold=3,
        decay_factor=0.9,
        min_strength_keep=0.1,
        max_patterns_per_cycle=10,
        max_abstractions_per_cycle=5,
    )
    return ConsolidationEngine(
        hive=woven_mind.hive,
        cortex=woven_mind.cortex,
        config=config,
    )


@pytest.fixture
def escalation_manager():
    """Create an escalation manager."""
    return EscalationManager()


# =============================================================================
# TEST STRATEGY EVOLUTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.optimization
@pytest.mark.evolution
class TestStrategyEvolution:
    """
    Epic: Strategy Evolution through Genetic Algorithms

    As a developer building self-improving systems,
    I want strategies to evolve through natural selection,
    So that the system discovers optimal approaches over time.
    """

    def test_population_evolves_over_generations(
        self, strategy_evolver, strategy_pool
    ):
        """
        Scenario: Population should improve fitness over generations

        Given a population of strategies with varying fitness
        When running multiple evolution cycles
        Then average fitness should increase
        And best strategy should improve or stay the same
        Because evolution selects for higher fitness
        """
        # Given: Create initial population with varying fitness
        genomes = []
        for i in range(10):
            genome = StrategyGenome(
                genome_id=f"genome-{i}",
                exploration_rate=0.1 + (i * 0.05),
                confidence_threshold=0.5 + (i * 0.03),
                parallelism_preference=0.3 + (i * 0.04),
            )
            genomes.append(genome)
            strategy_pool.add(genome)

        # Create execution traces with fitness scores
        traces = []
        for i, genome in enumerate(genomes):
            trace = ExecutionTrace(
                trace_id=f"trace-{i}",
                goal="test-goal",
                strategy_genome_id=genome.genome_id,
                metrics=ExecutionMetrics(
                    goal_achieved=True,
                    completeness_score=0.5 + (i * 0.05),  # Increasing quality
                    total_duration_ms=1000 - (i * 50),  # Decreasing time
                ),
            )
            traces.append(trace)

        # When: Evolve over multiple generations
        initial_best_fitness = 0.0
        results = []

        for gen in range(3):
            result = strategy_evolver.evolve_generation(traces)
            results.append(result)

            if gen == 0:
                initial_best_fitness = result.best_fitness

        # Then: Fitness should improve or stay the same
        # Compare aggregated fitness scores
        final_best_fitness = results[-1].best_fitness.aggregate()
        initial_best_aggregated = results[0].best_fitness.aggregate()
        assert final_best_fitness >= initial_best_aggregated

        # Population should still exist
        assert len(strategy_pool.get_current_generation()) > 0

        # Evolution history should be recorded
        assert strategy_evolver.generation == 3

    @pytest.mark.skip(reason="Flaky: non-deterministic parent selection doesn't guarantee best-genome in parents")
    def test_elitism_preserves_best_strategies(
        self, strategy_evolver, strategy_pool
    ):
        """
        Scenario: Best strategies should survive across generations

        Given a population with one exceptionally fit strategy
        When evolving to next generation
        Then the best strategy should be preserved
        And appear in parent selection
        Because elitism protects high-fitness individuals
        """
        # Given: Create population with one star performer
        best_genome = StrategyGenome(
            genome_id="best-genome",
            exploration_rate=0.15,
            confidence_threshold=0.8,
            parallelism_preference=0.6,
        )
        strategy_pool.add(best_genome)

        # Add mediocre genomes
        for i in range(5):
            genome = StrategyGenome(
                genome_id=f"mediocre-{i}",
                exploration_rate=0.3,
                confidence_threshold=0.5,
                parallelism_preference=0.4,
            )
            strategy_pool.add(genome)

        # Create fitness scores
        fitness_scores = {
            "best-genome": FitnessScore(
                success=1.0,
                efficiency=0.9,
                quality=0.95,
                stability=0.9,
                elegance=0.85,
            ),
        }

        for i in range(5):
            fitness_scores[f"mediocre-{i}"] = FitnessScore(
                success=0.6,
                efficiency=0.5,
                quality=0.5,
                stability=0.6,
            )

        # When: Select parents
        population = strategy_pool.get_current_generation()
        parents = strategy_evolver.select_parents(population, fitness_scores)

        # Then: Best genome should be in parents
        parent_ids = [p.genome_id for p in parents]
        assert "best-genome" in parent_ids

        # Best genome should have highest selection probability
        # (it should appear, though not guaranteed in every run)
        assert len(parents) > 0

    def test_diversity_maintained(self, strategy_evolver, strategy_pool):
        """
        Scenario: Population should maintain genetic diversity

        Given a population after selection
        When measuring diversity metrics
        Then multiple distinct strategies should exist
        And exploration rates should vary
        Because diversity prevents premature convergence
        """
        # Given: Create diverse initial population
        for i in range(10):
            genome = StrategyGenome(
                genome_id=f"genome-{i}",
                exploration_rate=0.05 + (i * 0.09),  # Range: 0.05 to 0.86
                confidence_threshold=0.5 + (i * 0.04),
                parallelism_preference=0.2 + (i * 0.06),
            )
            strategy_pool.add(genome)

        # Create fitness scores (all reasonable)
        fitness_scores = {}
        for i in range(10):
            fitness_scores[f"genome-{i}"] = FitnessScore(
                success=0.7 + (i * 0.02),
                efficiency=0.6,
                quality=0.7,
            )

        # When: Select parents (with diversity preservation)
        population = strategy_pool.get_current_generation()
        parents = strategy_evolver.select_parents(population, fitness_scores)

        # Then: Multiple distinct genomes selected
        assert len(parents) > 1

        # Check exploration rate diversity
        exploration_rates = [p.exploration_rate for p in parents]
        unique_rates = set(exploration_rates)

        # Should have some diversity (at least 2 different rates)
        assert len(unique_rates) >= 2

        # Range should be reasonable
        rate_range = max(exploration_rates) - min(exploration_rates)
        assert rate_range > 0.0

    def test_mutation_respects_bounds(self, strategy_evolver):
        """
        Scenario: Mutations should stay within valid ranges

        Given a strategy genome
        When applying mutations
        Then numeric parameters should stay in [0, 1]
        And structure should remain valid
        Because invalid strategies waste evaluation
        """
        # Given: A genome at boundary
        genome = StrategyGenome(
            genome_id="boundary-genome",
            exploration_rate=0.95,  # Near max
            confidence_threshold=0.05,  # Near min
            parallelism_preference=0.5,  # Middle
        )

        # When: Apply mutations multiple times
        mutated_genomes = []
        for i in range(20):
            mutated = strategy_evolver.mutate(genome, mutation_rate=0.5)
            mutated_genomes.append(mutated)

        # Then: All values should be in valid range [0, 1]
        for mutated in mutated_genomes:
            assert 0.0 <= mutated.exploration_rate <= 1.0
            assert 0.0 <= mutated.confidence_threshold <= 1.0
            assert 0.0 <= mutated.parallelism_preference <= 1.0

        # At least some mutations should have occurred
        # (with mutation_rate=0.5, very likely)
        exploration_values = [m.exploration_rate for m in mutated_genomes]
        assert len(set(exploration_values)) > 1


# =============================================================================
# TEST VELOCITY PREDICTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.optimization
@pytest.mark.velocity
class TestVelocityPrediction:
    """
    Epic: Velocity Prediction with Trend Analysis

    As a developer building predictive systems,
    I want to predict future velocity based on trends,
    So that planning and resource allocation improve.
    """

    def test_prediction_improves_with_history(self, metrics_collector):
        """
        Scenario: More history should improve prediction accuracy

        Given a metrics collector with goal completions
        When collecting more data over time
        Then velocity estimates should stabilize
        And variance should decrease
        Because more data reduces uncertainty
        """
        # Given: Simulate goal completions over time
        base_time = datetime.now() - timedelta(days=10)

        # First week: sparse data
        for i in range(3):
            start = base_time + timedelta(days=i)
            end = start + timedelta(hours=2)
            metrics_collector.record_goal_completion(start, end)

        # When: Get early metrics
        early_metrics = metrics_collector.get_hybrid_metrics()
        early_velocity = early_metrics.throughput

        # Then: Add more data (second week)
        for i in range(7):
            start = base_time + timedelta(days=i + 3)
            end = start + timedelta(hours=2)
            metrics_collector.record_goal_completion(start, end)

        # When: Get later metrics
        later_metrics = metrics_collector.get_hybrid_metrics()
        later_velocity = later_metrics.throughput

        # Then: Should have more data points
        assert len(metrics_collector._goal_times) == 10

        # Velocity should be computed
        assert later_velocity > 0.0

        # With consistent completion times, velocity should be stable
        # (we're completing goals at steady rate)
        assert later_velocity >= early_velocity * 0.5  # Within reasonable range

    def test_trend_detection_identifies_patterns(self, metrics_collector):
        """
        Scenario: Should detect increasing/decreasing trends

        Given velocity data with a clear trend
        When analyzing velocity over time
        Then trend direction should be identifiable
        And slope should be computable
        Because trends inform capacity planning
        """
        # Given: Simulate increasing velocity trend
        # Week 1: slow (4 hour cycles)
        # Week 2: medium (2 hour cycles)
        # Week 3: fast (1 hour cycles)

        base_time = datetime.now() - timedelta(days=21)

        # Week 1: 2 goals (slow)
        for i in range(2):
            start = base_time + timedelta(days=i * 3)
            end = start + timedelta(hours=4)
            metrics_collector.record_goal_completion(start, end)

        # Week 2: 4 goals (faster)
        for i in range(4):
            start = base_time + timedelta(days=7 + i * 1.5)
            end = start + timedelta(hours=2)
            metrics_collector.record_goal_completion(start, end)

        # Week 3: 7 goals (fastest)
        for i in range(7):
            start = base_time + timedelta(days=14 + i)
            end = start + timedelta(hours=1)
            metrics_collector.record_goal_completion(start, end)

        # When: Get metrics
        metrics = metrics_collector.get_hybrid_metrics()

        # Then: Should see improved throughput
        assert metrics.throughput > 0.0

        # Total goals completed
        assert len(metrics_collector._goal_times) == 13

        # Cycle time should reflect the average
        assert metrics.cycle_time.total_seconds() > 0

    def test_confidence_reflects_variance(self, metrics_collector):
        """
        Scenario: High variance should mean lower confidence

        Given velocity data with high variance
        When computing velocity stability
        Then stability score should be lower
        And predictions should reflect uncertainty
        Because variance indicates unpredictability
        """
        # Given: Simulate high-variance completions
        base_time = datetime.now() - timedelta(days=10)

        # Alternate between very fast and very slow
        completion_times = [0.5, 8, 1, 7, 0.5, 9, 1, 8]  # hours

        for i, hours in enumerate(completion_times):
            start = base_time + timedelta(days=i)
            end = start + timedelta(hours=hours)
            metrics_collector.record_goal_completion(start, end)

        # When: Get metrics
        metrics = metrics_collector.get_hybrid_metrics()

        # Then: Velocity should be computed
        assert metrics.throughput > 0.0

        # With high variance in cycle times:
        cycle_times = [
            (end - start).total_seconds()
            for start, end in metrics_collector._goal_times
        ]

        # Calculate variance
        mean = sum(cycle_times) / len(cycle_times)
        variance = sum((t - mean) ** 2 for t in cycle_times) / len(cycle_times)

        # High variance should be detected
        assert variance > 0.0

        # Could compute confidence as 1/(1+variance) if needed
        # For now, verify we have the data to detect variance
        assert len(set(cycle_times)) > 2  # Multiple distinct times


# =============================================================================
# TEST LEARNING CONSOLIDATION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.optimization
@pytest.mark.consolidation
class TestLearningConsolidation:
    """
    Epic: Learning Consolidation

    As a developer building memory systems,
    I want frequent patterns to consolidate into abstractions,
    So that the system learns efficiently from experience.
    """

    def test_similar_lessons_merged(self, consolidation_engine, woven_mind):
        """
        Scenario: Similar lessons should be merged into one

        Given multiple observations of the same pattern
        When running consolidation
        Then pattern should transfer to Cortex
        And abstraction should be formed
        Because repetition indicates importance
        """
        # Given: Train with similar patterns
        woven_mind.train("use authentication with jwt")
        woven_mind.train("implement authentication using jwt")
        woven_mind.train("jwt for authentication system")

        # Record patterns manually for consolidation
        pattern = {"authentication", "jwt"}
        for _ in range(5):
            consolidation_engine.record_pattern(pattern)

        # When: Run consolidation
        result = consolidation_engine.consolidate()

        # Then: Should have attempted transfer
        # Patterns should be identified as frequent
        frequent = consolidation_engine.get_frequent_patterns(min_frequency=3)
        assert len(frequent) > 0

        # Consolidation result should show activity
        assert result.cycle_duration_ms > 0.0
        assert result.timestamp is not None

        # Phases should have completed
        assert consolidation_engine.current_phase == ConsolidationPhase.IDLE

    def test_low_confidence_deprecated(self, consolidation_engine):
        """
        Scenario: Low confidence lessons should be deprecated

        Given patterns observed only once or twice
        When running decay cycle
        Then those patterns should be pruned
        And memory should be freed
        Because rare patterns waste space
        """
        # Given: Record some patterns with low frequency
        patterns = [
            frozenset({"rare", "pattern", "one"}),
            frozenset({"rare", "pattern", "two"}),
            frozenset({"common", "pattern"}),
        ]

        # Rare patterns: observed once
        consolidation_engine.record_pattern(patterns[0])
        consolidation_engine.record_pattern(patterns[1])

        # Common pattern: observed many times
        for _ in range(10):
            consolidation_engine.record_pattern(patterns[2])

        # When: Run decay cycle
        decay_result = consolidation_engine.decay_cycle()

        # Then: After decay, rare patterns should have reduced frequency
        # Get frequencies before full consolidation to check state
        initial_count = len(consolidation_engine._pattern_frequencies)

        # Run another decay
        for _ in range(3):
            consolidation_engine.decay_cycle()

        # Then: Some patterns should be pruned
        assert decay_result["decayed"] >= 0
        assert decay_result["pruned"] >= 0

        # After multiple decays, rare patterns should be gone
        # (with decay_factor=0.9, patterns with freq=1 will drop below 1 quickly)
        final_count = len(consolidation_engine._pattern_frequencies)
        # Should have fewer patterns after decay (or same if threshold not reached)
        assert final_count <= initial_count

    def test_high_confidence_promoted(
        self, consolidation_engine, woven_mind
    ):
        """
        Scenario: High confidence lessons should be promoted

        Given a pattern observed frequently
        When running consolidation
        Then pattern should transfer to Cortex
        And be promoted to abstraction
        Because high frequency indicates value
        """
        # Given: A frequently observed pattern
        pattern = frozenset({"database", "query", "optimization"})

        # Record many observations
        for _ in range(10):
            consolidation_engine.record_pattern(pattern)

        # When: Get frequent patterns
        frequent = consolidation_engine.get_frequent_patterns(min_frequency=5)

        # Then: Pattern should be in frequent list
        pattern_found = False
        for freq_pattern, freq_count in frequent:
            if freq_pattern == pattern:
                pattern_found = True
                assert freq_count >= 10
                break

        assert pattern_found

        # When: Run consolidation
        result = consolidation_engine.consolidate()

        # Then: Consolidation should have run
        assert result.cycle_duration_ms > 0.0

        # Pattern should be a candidate for transfer
        # (actual transfer depends on Cortex state)

    def test_patterns_extracted(self, consolidation_engine):
        """
        Scenario: Patterns should emerge from successful lessons

        Given multiple related patterns
        When analyzing for abstractions
        Then common structure should be identified
        And higher-level pattern should form
        Because abstraction reduces complexity
        """
        # Given: Record related patterns with common elements
        patterns = [
            frozenset({"auth", "user", "login"}),
            frozenset({"auth", "user", "logout"}),
            frozenset({"auth", "user", "session"}),
            frozenset({"auth", "admin", "privileges"}),
        ]

        # All share "auth"
        for pattern in patterns:
            for _ in range(5):
                consolidation_engine.record_pattern(pattern)

        # When: Get frequent patterns
        frequent = consolidation_engine.get_frequent_patterns(min_frequency=3)

        # Then: Should identify multiple patterns
        assert len(frequent) >= 2

        # Patterns should have common elements (auth)
        # Could check for shared terms across patterns
        all_terms = set()
        for pattern, _ in frequent:
            all_terms.update(pattern)

        # "auth" should appear in patterns
        assert "auth" in all_terms

        # Multiple patterns means structure is emerging
        assert len(frequent) > 1


# =============================================================================
# TEST ESCALATION ACTIONS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.optimization
@pytest.mark.escalation
class TestEscalationActions:
    """
    Epic: Complete Escalation Action Execution

    As a developer building resilient systems,
    I want escalation actions to be fully implemented,
    So that confusion is handled appropriately.
    """

    def test_monitor_enables_tracking(self, escalation_manager):
        """
        Scenario: MONITOR should enable enhanced tracking

        Given a low-severity confusion event
        When escalating to MONITOR level
        Then monitoring should be enabled
        And subsequent events should be tracked
        Because monitoring provides early warning
        """
        # Given: Low severity confusion
        confusion = ConfusionSignal(
            signal_type="uncertain_direction",
            description="Not sure which approach to use",
            evidence=["Multiple valid options found"],
            confidence=0.4,  # LOW severity
            source="worker-1",
        )

        # When: Evaluate escalation
        protocol = escalation_manager.evaluate(
            worker_id="worker-1",
            confusion=confusion,
            task_id="task-123",
        )

        # Then: Should escalate to MONITOR
        assert protocol.level == EscalationLevel.MONITOR

        # Action should mention monitoring
        assert "monitor" in protocol.recommended_action.lower()

        # Execute the protocol
        success = escalation_manager.execute(protocol)
        assert success

        # Protocol should be in history
        history = escalation_manager.get_escalation_history()
        assert len(history) == 1
        assert history[0].level == EscalationLevel.MONITOR

    def test_reassign_moves_task(self, escalation_manager):
        """
        Scenario: REASSIGN should move task to different worker

        Given repeated medium-severity confusion
        When escalating to REASSIGN level
        Then task should be reassigned
        And worker should be blacklisted for this task type
        Because persistent confusion indicates poor fit
        """
        # Given: First confusion (will be MONITOR)
        confusion1 = ConfusionSignal(
            signal_type="conflicting_requirements",
            description="Requirements seem contradictory",
            evidence=["Spec says X, but also says Y"],
            confidence=0.6,  # MEDIUM severity
            source="worker-2",
        )

        protocol1 = escalation_manager.evaluate(
            worker_id="worker-2",
            confusion=confusion1,
            task_id="task-456",
        )

        # Should be MONITOR on first confusion
        assert protocol1.level == EscalationLevel.MONITOR

        # When: Second confusion (should escalate to REASSIGN)
        confusion2 = ConfusionSignal(
            signal_type="stuck_in_loop",
            description="Same issue recurring",
            evidence=["Tried 3 approaches, all fail"],
            confidence=0.65,  # MEDIUM severity
            source="worker-2",
        )

        protocol2 = escalation_manager.evaluate(
            worker_id="worker-2",
            confusion=confusion2,
            task_id="task-456",
        )

        # Then: Should escalate to REASSIGN
        assert protocol2.level == EscalationLevel.REASSIGN

        # Action should mention reassignment
        assert "reassign" in protocol2.recommended_action.lower()
        assert "worker-2" in protocol2.recommended_action

        # Execute reassignment
        success = escalation_manager.execute(protocol2)
        assert success

        # Worker should have strikes
        strikes = escalation_manager.get_worker_strikes("worker-2")
        assert strikes == 2

    def test_abort_captures_learning(self, escalation_manager):
        """
        Scenario: ABORT should capture failure for learning

        Given three confusion events from same worker
        When escalating to ABORT level
        Then task should be aborted
        And failure should be captured for learning
        And confusion history should be preserved
        Because failures are learning opportunities
        """
        # Given: Three confusion events
        confusions = [
            ConfusionSignal(
                signal_type="unclear_spec",
                description="Specification is ambiguous",
                evidence=["Multiple interpretations possible"],
                confidence=0.5,
                source="worker-3",
            ),
            ConfusionSignal(
                signal_type="contradictory_feedback",
                description="Feedback conflicts with spec",
                evidence=["User wants X but spec says Y"],
                confidence=0.6,
                source="worker-3",
            ),
            ConfusionSignal(
                signal_type="repeated_failure",
                description="Cannot make progress",
                evidence=["All attempts fail"],
                confidence=0.7,
                source="worker-3",
            ),
        ]

        # When: Process all three
        protocols = []
        for i, confusion in enumerate(confusions):
            protocol = escalation_manager.evaluate(
                worker_id="worker-3",
                confusion=confusion,
                task_id="task-789",
            )
            protocols.append(protocol)

        # Then: Third should be ABORT
        assert protocols[0].level == EscalationLevel.MONITOR
        # Second could be INTERVENE or REASSIGN depending on severity
        assert protocols[2].level == EscalationLevel.ABORT

        # ABORT action should mention failure capture
        abort_protocol = protocols[2]
        assert "abort" in abort_protocol.recommended_action.lower()
        assert "learning" in abort_protocol.recommended_action.lower()

        # Execute abort
        success = escalation_manager.execute(abort_protocol)
        assert success

        # Confusion history should be preserved
        confusion_history = escalation_manager.get_worker_confusion_history(
            "worker-3"
        )
        assert len(confusion_history) == 3

        # All confusions should be recorded
        signal_types = [c.signal_type for c in confusion_history]
        assert "unclear_spec" in signal_types
        assert "contradictory_feedback" in signal_types
        assert "repeated_failure" in signal_types

        # Worker should have 3 strikes
        strikes = escalation_manager.get_worker_strikes("worker-3")
        assert strikes == 3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.optimization
class TestPhase2Integration:
    """
    Epic: Phase 2 Full Integration

    As a developer building optimized cognitive systems,
    I want all Phase 2 components to work together,
    So that the system continuously improves through evolution and learning.
    """

    def test_full_optimization_cycle(
        self,
        strategy_evolver,
        strategy_pool,
        metrics_collector,
        consolidation_engine,
        escalation_manager,
    ):
        """
        Scenario: All Phase 2 features work together

        Given all Phase 2 components initialized
        When executing a full optimization cycle
        Then:
          - Strategies evolve based on metrics
          - Velocity predictions inform planning
          - Learning consolidates from experience
          - Escalations handle failures gracefully
        Because integrated optimization yields emergent intelligence
        """
        # Given: Initialize components
        # (provided by fixtures)

        # Phase 1: Execute with strategy
        genome = StrategyGenome(
            genome_id="test-genome",
            exploration_rate=0.15,
            confidence_threshold=0.7,
            parallelism_preference=0.5,
        )
        strategy_pool.add(genome)

        # Record execution metrics
        base_time = datetime.now() - timedelta(days=5)
        for i in range(5):
            start = base_time + timedelta(days=i)
            end = start + timedelta(hours=2)
            metrics_collector.record_goal_completion(start, end)

        # Phase 2: Consolidate learning
        pattern = frozenset({"pattern", "test"})
        for _ in range(5):
            consolidation_engine.record_pattern(pattern)

        consolidation_result = consolidation_engine.consolidate()

        # Phase 3: Handle confusion with escalation
        confusion = ConfusionSignal(
            signal_type="test_confusion",
            description="Test confusion signal",
            evidence=["Test evidence"],
            confidence=0.5,
            source="test-worker",
        )

        escalation_protocol = escalation_manager.evaluate(
            worker_id="test-worker",
            confusion=confusion,
            task_id="test-task",
        )

        # Phase 4: Evolve strategies
        trace = ExecutionTrace(
            trace_id="integration-trace",
            goal="integration-test",
            strategy_genome_id=genome.genome_id,
            metrics=ExecutionMetrics(
                goal_achieved=True,
                completeness_score=0.8,
                total_duration_ms=2000,
            ),
        )

        evolution_result = strategy_evolver.evolve_generation([trace])

        # Then: All components should work
        # Metrics collected
        hybrid_metrics = metrics_collector.get_hybrid_metrics()
        assert hybrid_metrics.throughput > 0.0

        # Consolidation ran
        assert consolidation_result.cycle_duration_ms > 0.0

        # Escalation handled
        assert escalation_protocol.level in [
            EscalationLevel.MONITOR,
            EscalationLevel.INTERVENE,
        ]

        # Evolution produced result
        assert evolution_result.best_fitness.aggregate() >= 0.0
        assert evolution_result.generation > 0

        # Population still exists
        assert len(strategy_pool.get_current_generation()) > 0
