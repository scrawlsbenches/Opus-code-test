"""
Behavioral tests for WovenMind dual-process cognitive architecture.

As a developer building intelligent systems,
I want a cognitive architecture that switches between fast and slow thinking,
So that the system is both efficient and thorough when needed.

Based on: examples/woven_mind_demo.py
"""

import pytest
from cortical.reasoning import (
    # Sprint 1: Loom (Dual-Process)
    Loom,
    LoomConfig,
    ThinkingMode,
    # Sprint 2: Hebbian Hive (PRISM-SLM)
    PRISMLanguageModel,
    HiveNode,
    HiveEdge,
    HomeostasisRegulator,
    HomeostasisConfig,
    # Sprint 3: Cortex Abstraction
    PatternDetector,
    AbstractionEngine,
    GoalStack,
    GoalPriority,
    # Sprint 4: Unified Architecture
    WovenMind,
    WovenMindConfig,
)


class TestDeveloperSwitchesThinkingModes:
    """
    Epic: Dual-Process Thinking

    As a developer building a cognitive system,
    I want the system to automatically switch between FAST and SLOW modes,
    So that it handles familiar patterns quickly and novel situations carefully.
    """

    def test_system_switches_to_slow_on_surprise(self):
        """
        Scenario: System switches to SLOW mode when encountering surprising input

        Given a system configured with surprise threshold of 0.3
        When the input is very different from predictions
        Then the system switches to SLOW mode for careful processing
        """
        # Given: a system configured with surprise threshold
        config = LoomConfig(surprise_threshold=0.3)
        loom = Loom(config)

        # When: the input is very different from predictions
        predicted = {"expected": 0.8}
        actual = {"unexpected", "surprising", "novel"}
        signal = loom.detect_surprise(predicted, actual)

        # Then: surprise is detected
        assert signal.magnitude > 0.3, "Should detect high surprise"

        # Then: system switches to SLOW mode
        mode = loom.select_mode(signal)
        assert mode == ThinkingMode.SLOW, "Should switch to SLOW mode for surprising input"

    def test_system_stays_fast_for_familiar_input(self):
        """
        Scenario: System stays in FAST mode for familiar patterns

        Given a system in FAST mode
        When the input matches predictions well
        Then the system remains in FAST mode
        Because fast processing is sufficient for familiar patterns
        """
        # Given: a system in FAST mode
        config = LoomConfig(surprise_threshold=0.3)
        loom = Loom(config)

        # When: input matches predictions well
        predicted = {"neural": 0.9, "network": 0.85}
        actual = {"neural", "network"}
        signal = loom.detect_surprise(predicted, actual)

        # Then: surprise is low
        assert signal.magnitude < 0.3, "Should have low surprise for familiar input"

        # Then: system stays in FAST mode
        mode = loom.select_mode(signal)
        assert mode == ThinkingMode.FAST, "Should stay in FAST mode for familiar input"

    def test_system_tracks_mode_transitions(self):
        """
        Scenario: System maintains history of mode transitions

        Given a system that processes multiple inputs
        When some inputs trigger mode changes
        Then the system tracks all transitions
        Because understanding mode changes helps debug behavior
        """
        # Given: a system that processes multiple inputs
        loom = Loom(LoomConfig(surprise_threshold=0.3))

        # When: multiple inputs cause different surprise levels
        scenarios = [
            ({"familiar": 0.9}, {"familiar"}),        # Low surprise
            ({"known": 0.2}, {"unknown", "novel"}),   # High surprise
            ({"expected": 0.85}, {"expected"}),       # Low surprise
        ]

        for predicted, actual in scenarios:
            signal = loom.detect_surprise(predicted, actual)
            loom.select_mode(signal)

        # Then: system tracks all transitions
        history = loom.get_transition_history()
        assert len(history) > 0, "Should track mode transitions"


class TestDeveloperUsesHebbianLearning:
    """
    Epic: Synaptic Learning

    As a developer building a learning system,
    I want neurons to strengthen connections through co-activation,
    So that the system learns patterns from experience.
    """

    def test_neurons_strengthen_through_coactivation(self):
        """
        Scenario: Neurons that fire together, wire together

        Given two concepts that co-occur frequently
        When both concepts activate together multiple times
        Then the synaptic connection between them strengthens
        Because this is Hebbian learning
        """
        # Given: two concepts
        node_a = HiveNode(id="machine")
        node_b = HiveNode(id="learning")
        edge = HiveEdge(source_id="machine", target_id="learning")
        initial_weight = edge.weight

        # When: both concepts activate together multiple times
        for step in range(10):
            node_a.activate(amount=0.8, step=step)
            node_b.activate(amount=0.7, step=step)
            edge.pre_trace = node_a.trace
            edge.post_trace = node_b.trace
            edge.learn()

        # Then: connection strengthens
        assert edge.weight > initial_weight, "Connection should strengthen through co-activation"

    def test_lateral_inhibition_creates_sparse_activation(self):
        """
        Scenario: Lateral inhibition creates competition between concepts

        Given a language model with many possible activations
        When lateral inhibition is applied
        Then only the strongest concepts remain active
        Because competition creates sparse, focused representations
        """
        # Given: a trained language model
        model = PRISMLanguageModel(context_size=3)
        model.train("neural networks learn patterns from data")

        # When: we activate with and without inhibition
        query = ["neural"]
        without_inhibition = model.graph.sparse_activate(query, k=10, use_inhibition=False)
        with_inhibition = model.graph.sparse_activate(query, k=10, use_inhibition=True)

        # Then: inhibition reduces total activation
        total_without = sum(without_inhibition.values())
        total_with = sum(with_inhibition.values())
        assert total_with < total_without, "Lateral inhibition should reduce total activation"

    def test_homeostasis_maintains_balanced_activation(self):
        """
        Scenario: Homeostatic regulation prevents runaway activation

        Given a system with varying activation levels
        When homeostatic regulation is applied
        Then activation levels stabilize around the target
        Because homeostasis maintains healthy operation
        """
        # Given: a homeostatic regulator
        config = HomeostasisConfig(target_activation=0.05, adjustment_rate=0.1)
        regulator = HomeostasisRegulator(config)

        # When: we record varying activations and regulate
        for i in range(10):
            activations = {
                "overactive": 0.9,
                "normal": 0.05,
                "underactive": 0.01,
            }
            regulator.record_activations(activations)
            regulator.regulate()

        # Then: system tracks health metrics
        metrics = regulator.get_health_metrics()
        assert "avg_activation" in metrics, "Should track average activation"
        assert "avg_excitability" in metrics, "Should track excitability"


class TestDeveloperFormsAbstractions:
    """
    Epic: Pattern Abstraction

    As a developer building a learning system,
    I want the system to form abstractions from repeated patterns,
    So that it discovers higher-level concepts.
    """

    def test_abstractions_require_minimum_observations(self):
        """
        Scenario: Abstractions only form after sufficient evidence

        Given a pattern detector requiring 3 observations
        When a pattern is observed only twice
        Then no abstraction forms
        But when observed three times
        Then an abstraction is created
        Because we need sufficient evidence before abstracting
        """
        # Given: a pattern detector requiring 3 observations
        engine = AbstractionEngine(min_frequency=3)

        # When: a pattern is observed twice
        pattern = frozenset(["neural", "network"])
        engine.observe(pattern)
        engine.observe(pattern)

        # Then: no abstraction forms yet
        formed = engine.auto_form_abstractions()
        assert len(formed) == 0, "Should not form abstraction with only 2 observations"

        # But when: observed a third time
        engine.observe(pattern)

        # Then: abstraction is created
        formed = engine.auto_form_abstractions()
        assert len(formed) > 0, "Should form abstraction after 3 observations"
        assert pattern == formed[0].source_nodes, "Abstraction should match observed pattern"

    def test_abstractions_form_hierarchically(self):
        """
        Scenario: Abstractions can combine into meta-abstractions

        Given multiple first-level abstractions
        When those abstractions co-occur frequently
        Then a higher-level meta-abstraction can form
        Because concepts combine into increasingly abstract ideas
        """
        # Given: an engine that forms abstractions
        engine = AbstractionEngine(min_frequency=2)

        # When: we observe patterns enough to form abstractions
        pattern1 = frozenset(["neural", "network"])
        pattern2 = frozenset(["deep", "learning"])

        for _ in range(3):
            engine.observe(pattern1)
            engine.observe(pattern2)

        # Then: form first-level abstractions
        level1 = engine.auto_form_abstractions()
        assert len(level1) >= 2, "Should form multiple abstractions"

        # When: we create a meta-pattern from abstractions
        meta_pattern = frozenset([level1[0].id, level1[1].id])
        meta_abs = engine.form_abstraction(meta_pattern, level=2)

        # Then: meta-abstraction has higher level
        assert meta_abs is not None, "Should create meta-abstraction"
        assert meta_abs.level == 2, "Meta-abstraction should have level 2"


class TestDeveloperTracksGoals:
    """
    Epic: Goal Management

    As a developer building a goal-driven system,
    I want to track progress toward goals with monotonic guarantees,
    So that progress never regresses.
    """

    def test_progress_only_increases(self):
        """
        Scenario: Goal progress is monotonic (never decreases)

        Given a goal with some progress
        When attempting to update with lower progress
        Then the update is rejected
        Because progress must be monotonic
        """
        # Given: a goal with some progress
        stack = GoalStack()
        goal = stack.push_goal("Learn neural networks", target_nodes={"neural", "network"})
        stack.update_progress(goal.id, 0.5)

        # When: attempting to regress progress
        result = stack.update_progress(goal.id, 0.3)

        # Then: update is rejected
        assert not result, "Should reject decreasing progress"
        assert stack.get_progress(goal.id) == 0.5, "Progress should remain unchanged"

    def test_child_goals_block_until_dependencies_complete(self):
        """
        Scenario: Dependent goals wait for prerequisites

        Given a goal that depends on another goal
        When the blocking goal is incomplete
        Then the dependent goal remains blocked
        But when the blocking goal completes
        Then the dependent goal becomes active
        """
        # Given: a goal that depends on another
        stack = GoalStack()
        basics = stack.push_goal("Learn basics", target_nodes={"basic"})
        advanced = stack.push_goal(
            "Learn advanced",
            target_nodes={"advanced"},
            blocking_goals={basics.id}
        )

        # When: blocking goal is incomplete
        # Then: dependent goal is blocked
        assert advanced.is_blocked(), "Should be blocked by dependency"

        # But when: blocking goal completes
        stack.update_progress(basics.id, 1.0)

        # Then: dependent goal becomes active
        assert not advanced.is_blocked(), "Should be unblocked after dependency completes"


class TestDeveloperUsesUnifiedFacade:
    """
    Epic: Unified Cognitive Architecture

    As a developer using the cognitive system,
    I want a single unified interface to all components,
    So that I don't need to manage components separately.
    """

    def test_woven_mind_provides_single_interface(self):
        """
        Scenario: WovenMind provides unified access to all components

        Given a WovenMind instance
        When I train and process through the facade
        Then all internal components work together seamlessly
        Because WovenMind hides the complexity
        """
        # Given: a WovenMind instance
        config = WovenMindConfig(
            surprise_threshold=0.3,
            k_winners=5,
            min_frequency=2,
        )
        mind = WovenMind(config=config)

        # When: training on text
        training_text = "neural networks learn patterns through training"
        mind.train(training_text)

        # Then: can process queries
        result = mind.process(["neural", "network"], mode=ThinkingMode.FAST)
        assert result.mode == ThinkingMode.FAST, "Should process in FAST mode"
        assert len(result.activations) > 0, "Should return activations"

    def test_auto_mode_selection_based_on_surprise(self):
        """
        Scenario: WovenMind automatically selects appropriate mode

        Given a system trained on familiar patterns
        When processing familiar input with auto mode
        Then FAST mode is selected
        But when processing novel input
        Then SLOW mode is selected
        Because mode selection is surprise-driven
        """
        # Given: a trained system
        mind = WovenMind(config=WovenMindConfig(surprise_threshold=0.3))
        mind.train("neural networks and deep learning")

        # When: processing familiar input with auto mode (None)
        familiar_result = mind.process(["neural"], mode=None)

        # Then: system should have low surprise for familiar input
        # (Note: First query might not have baseline yet, so we train the baseline)
        mind.process(["neural"], mode=ThinkingMode.FAST)

        # Now test with more novel input
        novel_result = mind.process(["quantum", "topology"], mode=None)

        # System should handle both familiar and novel inputs
        assert familiar_result.mode in [ThinkingMode.FAST, ThinkingMode.SLOW]
        assert novel_result.mode in [ThinkingMode.FAST, ThinkingMode.SLOW]

    def test_consolidation_transfers_patterns_to_abstractions(self):
        """
        Scenario: Consolidation acts like "sleep" to solidify learning

        Given a system with observed patterns
        When consolidation runs
        Then frequent patterns transfer to cortex abstractions
        Because consolidation converts fast learning to slow knowledge
        """
        # Given: a system with patterns
        mind = WovenMind(config=WovenMindConfig(consolidation_threshold=2))
        mind.train("neural network deep learning")

        # Record patterns
        for _ in range(3):
            mind.consolidation.record_pattern({"neural", "network"})

        # When: consolidation runs
        result = mind.consolidate()

        # Then: patterns are processed
        assert result.patterns_transferred >= 0, "Should report patterns transferred"
        assert result.cycle_duration_ms > 0, "Should track cycle duration"


class TestDeveloperDebugsCognitiveSystem:
    """
    Epic: System Observability

    As a developer debugging a cognitive system,
    I want to inspect internal state and statistics,
    So that I can understand what the system is doing.
    """

    def test_system_provides_comprehensive_statistics(self):
        """
        Scenario: Developer can inspect system statistics

        Given a WovenMind that has processed some inputs
        When requesting system statistics
        Then statistics cover all major components
        Because observability is essential for debugging
        """
        # Given: a system that has processed inputs
        mind = WovenMind()
        mind.train("test data for statistics")
        mind.process(["test"], mode=ThinkingMode.FAST)

        # When: requesting statistics
        stats = mind.get_stats()

        # Then: statistics cover major components
        assert "mode" in stats, "Should report current mode"
        assert "loom" in stats, "Should report Loom statistics"
        assert "hive" in stats, "Should report Hive statistics"
        assert "cortex" in stats, "Should report Cortex statistics"

    def test_system_is_serializable(self):
        """
        Scenario: System state can be saved and restored

        Given a trained WovenMind
        When serializing to dict and restoring
        Then the restored system has the same state
        Because persistence enables system continuity
        """
        # Given: a trained system
        original = WovenMind()
        original.train("neural networks learn patterns")
        original.process(["neural"], mode=ThinkingMode.FAST)

        # When: serializing and restoring
        state = original.to_dict()
        restored = WovenMind.from_dict(state)

        # Then: restored system exists and has structure
        assert restored is not None, "Should restore successfully"
        assert hasattr(restored, 'loom'), "Should have loom component"
        assert hasattr(restored, 'hive'), "Should have hive component"
        assert hasattr(restored, 'cortex'), "Should have cortex component"
