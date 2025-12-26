#!/usr/bin/env python3
"""
Woven Mind + PRISM Marriage Demo

This example demonstrates the complete cognitive architecture built across
Sprints 1-3 of the Woven Mind project:

Sprint 1 (Loom Foundation):
  - Dual-process thinking (FAST/SLOW modes)
  - Surprise detection for mode switching
  - Mode transitions based on uncertainty

Sprint 2 (Hebbian Hive):
  - PRISM-SLM for synaptic language modeling
  - Lateral inhibition for sparse activation
  - K-winners-take-all for competition
  - Homeostatic regulation for stability

Sprint 3 (Cortex Abstraction):
  - Pattern detection from activation history
  - Hierarchical abstraction formation
  - Goal tracking with monotonic progress

Sprint 4 (The Loom Weaves):
  - LoomHiveConnector: FAST mode with PRISM-SLM + homeostasis
  - LoomCortexConnector: SLOW mode with abstraction engine
  - AttentionRouter: Mode-based routing
  - WovenMind: Unified facade for the complete architecture

Sprint 5 (Consolidation Engine):
  - ConsolidationEngine: "Sleep-like" learning transfer
  - Pattern transfer from Hive to Cortex
  - Abstraction mining from frequent patterns
  - Decay cycles for forgetting unused connections

Usage:
    python examples/woven_mind_demo.py
    python examples/woven_mind_demo.py --verbose
    python examples/woven_mind_demo.py --corpus samples/
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.reasoning import (
    # Sprint 1: Loom (Dual-Process Integration)
    Loom,
    LoomConfig,
    ThinkingMode,
    SurpriseDetector,
    ModeController,
    # Sprint 2: Hebbian Hive (PRISM-SLM)
    PRISMLanguageModel,
    HiveNode,
    HiveEdge,
    # Sprint 2: Homeostasis
    HomeostasisRegulator,
    HomeostasisConfig,
    # Sprint 3: Abstraction
    PatternDetector,
    AbstractionEngine,
    Abstraction,
    # Sprint 3: Goal Stack
    GoalStack,
    Goal,
    GoalStatus,
    GoalPriority,
    # Sprint 4: The Loom Weaves (Unified Architecture)
    LoomHiveConnector,
    LoomHiveConfig,
    LoomCortexConnector,
    LoomCortexConfig,
    AttentionRouter,
    WovenMind,
    WovenMindConfig,
)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def load_corpus(corpus_path: str, max_docs: int = 10) -> list:
    """Load text documents from a directory."""
    documents = []
    path = Path(corpus_path)

    if not path.exists():
        print(f"  Warning: Corpus path '{corpus_path}' not found")
        return []

    for txt_file in sorted(path.glob("*.txt"))[:max_docs]:
        try:
            content = txt_file.read_text(encoding="utf-8")
            documents.append({
                "id": txt_file.stem,
                "content": content,
            })
        except Exception as e:
            print(f"  Warning: Could not read {txt_file}: {e}")

    return documents


def demo_loom_dual_process(verbose: bool = False):
    """
    Demonstrate Sprint 1: Loom Dual-Process Integration.

    Shows how the system switches between FAST (intuitive) and SLOW
    (deliberate) thinking based on surprise signals.
    """
    print_header("SPRINT 1: LOOM - Dual-Process Integration")

    # Configure the Loom with thresholds
    config = LoomConfig(
        surprise_threshold=0.3,      # Switch to SLOW when surprise > 0.3
        confidence_threshold=0.7,    # Switch to FAST when confidence > 0.7
    )

    loom = Loom(config)
    print(f"\n  Initial mode: {loom.get_current_mode().name}")
    print(f"  Surprise threshold: {config.surprise_threshold}")
    print(f"  Confidence threshold: {config.confidence_threshold}")

    # Simulate processing with varying surprise levels
    # Each scenario: (description, predicted tokens with probs, actual tokens)
    scenarios = [
        ("Familiar pattern", {"neural": 0.9, "network": 0.8}, {"neural", "network"}),
        ("Slightly novel", {"quantum": 0.4, "neural": 0.6}, {"quantum", "computing"}),
        ("Surprising", {"logic": 0.2, "proof": 0.3}, {"consciousness", "silicon"}),
        ("Very surprising", {"math": 0.1}, {"paradox", "infinite", "loop"}),
        ("Back to familiar", {"gradient": 0.85, "descent": 0.9}, {"gradient", "descent"}),
    ]

    print_subheader("Mode Transitions Based on Surprise")

    prev_mode = loom.get_current_mode()

    for description, predicted, actual in scenarios:
        # Detect surprise level
        signal = loom.detect_surprise(predicted, actual)

        # Select thinking mode based on surprise
        new_mode = loom.select_mode(signal)

        switched = " [SWITCHED]" if new_mode != prev_mode else ""
        prev_mode = new_mode

        print(f"\n  Input: {description}")
        print(f"    Predicted: {list(predicted.keys())}")
        print(f"    Actual: {actual}")
        print(f"    Surprise: {signal.magnitude:.2f}")
        print(f"    Mode: {new_mode.name}{switched}")

        if verbose:
            print(f"    Unexpected: {signal.unexpected_items}")

    # Show transition history
    print_subheader("Transition Summary")
    history = loom.get_transition_history()
    print(f"  Total transitions: {len(history)}")

    fast_count = sum(1 for t in history if t.to_mode == ThinkingMode.FAST)
    slow_count = sum(1 for t in history if t.to_mode == ThinkingMode.SLOW)
    total = fast_count + slow_count if (fast_count + slow_count) > 0 else 1
    print(f"  Transitions to FAST: {fast_count}")
    print(f"  Transitions to SLOW: {slow_count}")


def demo_hebbian_hive(corpus: list, verbose: bool = False):
    """
    Demonstrate Sprint 2: Hebbian Hive with PRISM-SLM.

    Shows synaptic learning, lateral inhibition, k-winners-take-all,
    and homeostatic regulation.
    """
    print_header("SPRINT 2: HEBBIAN HIVE - Synaptic Language Model")

    # Create PRISM Language Model
    model = PRISMLanguageModel(context_size=3)

    # Train on corpus
    print_subheader("Training PRISM-SLM")
    for doc in corpus[:5]:
        model.train(doc["content"])
        if verbose:
            print(f"  Trained on: {doc['id']}")

    print(f"\n  Vocabulary size: {model.vocab_size}")
    print(f"  Context window: {model.context_size}")

    # Demonstrate sparse activation
    print_subheader("Sparse Activation (k-Winners-Take-All)")

    test_queries = [
        ["neural"],
        ["machine", "learning"],
        ["algorithm", "optimization"],
    ]

    for query in test_queries:
        # Get sparse activation (top-k winners)
        k = max(1, int(model.vocab_size * 0.05))  # 5% sparsity target
        result = model.graph.sparse_activate(query, k=k)

        active_count = sum(1 for v in result.values() if v > 0)
        sparsity = active_count / model.vocab_size * 100 if model.vocab_size > 0 else 0

        print(f"\n  Query: {query}")
        print(f"    Active tokens: {active_count}/{model.vocab_size}")
        print(f"    Sparsity: {sparsity:.1f}%")

        if verbose:
            # Show top activations
            top_5 = sorted(result.items(), key=lambda x: -x[1])[:5]
            print(f"    Top 5: {[(t, f'{v:.3f}') for t, v in top_5]}")

    # Demonstrate lateral inhibition
    print_subheader("Lateral Inhibition Effect")

    query = ["neural"]
    result_no_inh = model.graph.sparse_activate(query, k=10, use_inhibition=False)
    result_with_inh = model.graph.sparse_activate(query, k=10, use_inhibition=True)

    total_no_inh = sum(result_no_inh.values())
    total_with_inh = sum(result_with_inh.values())
    reduction = (1 - total_with_inh / total_no_inh) * 100 if total_no_inh > 0 else 0

    print(f"\n  Without inhibition: total activation = {total_no_inh:.3f}")
    print(f"  With inhibition: total activation = {total_with_inh:.3f}")
    print(f"  Reduction: {reduction:.1f}%")

    # Demonstrate homeostasis
    print_subheader("Homeostatic Regulation")

    home_config = HomeostasisConfig(
        target_activation=0.05,  # 5% target
        adjustment_rate=0.1,
    )
    regulator = HomeostasisRegulator(home_config)

    # Simulate activations over time
    for i in range(5):
        # Create synthetic activations
        activations = {
            "neural": 0.8 - i * 0.1,  # Decreasing
            "network": 0.2 + i * 0.1,  # Increasing
            "learning": 0.5,          # Stable
        }
        regulator.record_activations(activations)
        regulator.regulate()

    metrics = regulator.get_health_metrics()
    print(f"\n  Target activation: {home_config.target_activation}")
    print(f"  Average activation: {metrics['avg_activation']:.3f}")
    print(f"  Average excitability: {metrics['avg_excitability']:.3f}")
    print(f"  Under-active nodes: {metrics['pct_underactive']:.1f}%")
    print(f"  Over-active nodes: {metrics['pct_overactive']:.1f}%")

    # Demonstrate HiveNode and HiveEdge
    print_subheader("Hebbian Learning (HiveNode/HiveEdge)")

    node_a = HiveNode(id="concept_A")
    node_b = HiveNode(id="concept_B")
    edge = HiveEdge(source_id="concept_A", target_id="concept_B")

    print(f"\n  Initial edge weight: {edge.weight:.3f}")

    # Simulate co-activation (Hebbian learning)
    for step in range(5):
        node_a.activate(amount=0.8, step=step)
        node_b.activate(amount=0.7, step=step)
        edge.pre_trace = node_a.trace
        edge.post_trace = node_b.trace
        delta = edge.learn()
        if verbose:
            print(f"    Step {step}: weight delta = {delta:.4f}")

    print(f"  Final edge weight: {edge.weight:.3f}")
    print(f"  Weight increase: {edge.weight:.3f} (from co-activation)")


def demo_cortex_abstraction(verbose: bool = False):
    """
    Demonstrate Sprint 3: Cortex Abstraction.

    Shows pattern detection, abstraction formation, and the
    requirement for minimum observations (≥3).
    """
    print_header("SPRINT 3: CORTEX ABSTRACTION - Pattern Detection")

    # Create pattern detector with min_frequency=3
    detector = PatternDetector(min_frequency=3, min_pattern_size=2)
    engine = AbstractionEngine(min_frequency=3)

    print(f"\n  Min frequency for abstraction: 3")
    print(f"  Min pattern size: 2 nodes")

    # Simulate activation patterns over time
    print_subheader("Observing Activation Patterns")

    patterns = [
        frozenset(["neural", "network"]),       # Will be observed 4x
        frozenset(["neural", "network"]),
        frozenset(["machine", "learning"]),     # Will be observed 3x
        frozenset(["neural", "network"]),
        frozenset(["machine", "learning"]),
        frozenset(["deep", "learning"]),        # Only 2x - won't become abstraction
        frozenset(["neural", "network"]),
        frozenset(["machine", "learning"]),
        frozenset(["deep", "learning"]),
        frozenset(["random", "pattern"]),       # Only 1x - ignored
    ]

    for i, pattern in enumerate(patterns):
        candidates = detector.observe(pattern)
        engine.observe(pattern)

        if verbose:
            print(f"  Observation {i+1}: {set(pattern)}")
            if candidates:
                print(f"    New candidates: {[set(c) for c in candidates]}")

    # Check which patterns became candidates
    print_subheader("Pattern Candidates (≥3 observations)")

    candidates = engine.abstraction_candidates()
    for pattern, freq, level in candidates:
        print(f"  {set(pattern)}: observed {freq}x")

    # Form abstractions
    print_subheader("Forming Abstractions")

    formed = engine.auto_form_abstractions(max_new=5)
    for abstraction in formed:
        print(f"\n  Abstraction: {abstraction.id}")
        print(f"    Source nodes: {set(abstraction.source_nodes)}")
        print(f"    Level: {abstraction.level}")
        print(f"    Frequency: {abstraction.frequency}")
        print(f"    Truth value: {abstraction.truth_value:.2f}")
        print(f"    Strength: {abstraction.strength:.3f}")

    # Demonstrate hierarchical abstraction
    if len(formed) >= 2:
        print_subheader("Hierarchical Abstraction (Meta-Level)")

        # Create a meta-abstraction from existing abstractions
        meta_pattern = frozenset([formed[0].id, formed[1].id])
        meta_abs = engine.form_abstraction(meta_pattern, level=2)

        if meta_abs:
            print(f"\n  Meta-abstraction: {meta_abs.id}")
            print(f"    Combines: {set(meta_abs.source_nodes)}")
            print(f"    Level: {meta_abs.level} (higher than components)")


def demo_goal_stack(verbose: bool = False):
    """
    Demonstrate Sprint 3: Goal Stack with Monotonic Progress.

    Shows goal tracking, progress that can only increase, and
    automatic unblocking when dependencies complete.
    """
    print_header("SPRINT 3: GOAL STACK - Monotonic Progress Tracking")

    stack = GoalStack(max_active_goals=5)

    # Create learning goals with dependencies
    print_subheader("Creating Goals with Dependencies")

    # Main goal: Master neural networks
    master_goal = stack.push_goal(
        "Master Neural Networks",
        target_nodes={"neural", "network", "backprop", "gradient", "loss"},
        priority=GoalPriority.HIGH,
    )
    print(f"\n  Created: {master_goal.name} [{master_goal.id}]")
    print(f"    Priority: {master_goal.priority.name}")
    print(f"    Target nodes: {set(master_goal.target_nodes)}")

    # Sub-goals
    basics = stack.push_goal(
        "Learn Basics",
        target_nodes={"neural", "network"},
        parent_id=master_goal.id,
    )

    backprop = stack.push_goal(
        "Understand Backpropagation",
        target_nodes={"backprop", "gradient"},
        parent_id=master_goal.id,
        blocking_goals={basics.id},  # Must learn basics first!
    )

    print(f"\n  Sub-goals created:")
    print(f"    - {basics.name} (active)")
    print(f"    - {backprop.name} (blocked by basics)")

    # Demonstrate monotonic progress
    print_subheader("Monotonic Progress Guarantee")

    print(f"\n  Initial progress: {basics.progress}")

    # Make progress
    result1 = stack.update_progress(basics.id, 0.5)
    print(f"  Update to 0.5: {'accepted' if result1 else 'REJECTED'}")
    print(f"    Current: {stack.get_progress(basics.id)}")

    # Try to regress (should fail!)
    result2 = stack.update_progress(basics.id, 0.3)
    print(f"  Update to 0.3: {'accepted' if result2 else 'REJECTED (regression prevented)'}")
    print(f"    Current: {stack.get_progress(basics.id)}")

    # Continue forward
    result3 = stack.update_progress(basics.id, 0.7)
    print(f"  Update to 0.7: {'accepted' if result3 else 'REJECTED'}")
    print(f"    Current: {stack.get_progress(basics.id)}")

    # Demonstrate achievement-based progress
    print_subheader("Node-Based Achievement Tracking")

    # Simulate learning nodes
    active_nodes = frozenset(["neural"])
    progress = stack.check_achievement(master_goal.id, active_nodes)
    print(f"\n  Active nodes: {set(active_nodes)}")
    print(f"  Progress: {progress:.0%}")

    active_nodes = frozenset(["neural", "network", "backprop"])
    progress = stack.check_achievement(master_goal.id, active_nodes)
    print(f"\n  Active nodes: {set(active_nodes)}")
    print(f"  Progress: {progress:.0%}")

    # Demonstrate unblocking
    print_subheader("Automatic Unblocking")

    print(f"\n  Backprop goal blocked: {backprop.is_blocked()}")
    print(f"  Blocking goals: {backprop.blocking_goals}")

    # Complete basics goal
    stack.update_progress(basics.id, 1.0)
    print(f"\n  Completed: {basics.name}")
    print(f"  Backprop goal blocked: {backprop.is_blocked()}")
    print(f"  Backprop goal status: {backprop.status.name}")

    # Show statistics
    print_subheader("Goal Stack Statistics")
    stats = stack.get_statistics()
    print(f"\n  Total goals: {stats['total_goals']}")
    print(f"  Active: {stats['active_goals']}")
    print(f"  Achieved: {stats['achieved_count']}")
    print(f"  Average progress: {stats['avg_progress']:.0%}")


def demo_woven_mind_facade(corpus: list, verbose: bool = False):
    """
    Demonstrate Sprint 4: WovenMind Unified Facade.

    Shows the complete dual-process architecture through the
    WovenMind facade, integrating:
    - LoomHiveConnector for FAST mode
    - LoomCortexConnector for SLOW mode
    - AttentionRouter for mode-based routing
    - Surprise detection for automatic mode switching
    """
    print_header("SPRINT 4: WOVEN MIND - Unified Dual-Process Architecture")

    # Create WovenMind with custom configuration
    config = WovenMindConfig(
        surprise_threshold=0.3,    # Switch to SLOW when surprise > 0.3
        k_winners=5,               # Lateral inhibition winners
        min_frequency=2,           # Abstractions form after 2 observations
    )
    mind = WovenMind(config=config)

    print(f"\n  Created WovenMind with:")
    print(f"    Surprise threshold: {config.surprise_threshold}")
    print(f"    K-winners: {config.k_winners}")
    print(f"    Min frequency: {config.min_frequency}")

    # Train on corpus
    print_subheader("Training the Hive (FAST Mode)")
    training_text = " ".join([doc["content"] for doc in corpus[:5]])
    mind.train(training_text)
    print(f"  Trained on {len(training_text.split())} words from corpus")

    # Demonstrate FAST mode processing
    print_subheader("FAST Mode Processing (Pattern Matching)")

    fast_queries = [
        ["neural", "network"],
        ["machine", "learning"],
        ["deep", "learning"],
    ]

    for query in fast_queries:
        result = mind.process(query, mode=ThinkingMode.FAST)
        print(f"\n  Query: {query}")
        print(f"    Mode: {result.mode.name}")
        print(f"    Source: {result.source}")
        print(f"    Activations: {sorted(list(result.activations))[:8]}")

    # Demonstrate SLOW mode processing
    print_subheader("SLOW Mode Processing (Abstraction)")

    # Observe patterns to build abstractions
    patterns = [
        ["neural", "network"],
        ["neural", "network"],
        ["neural", "network"],
        ["machine", "learning"],
        ["machine", "learning"],
    ]

    for pattern in patterns:
        result = mind.process(pattern, mode=ThinkingMode.SLOW)
        if verbose:
            print(f"  Observed: {pattern} -> {len(result.activations)} abstractions")

    # Check formed abstractions
    abstractions = mind.cortex.get_abstractions()
    print(f"\n  Formed {len(abstractions)} abstraction(s):")
    for abs in abstractions[:3]:
        print(f"    - {abs.id}: {set(abs.source_nodes)} (freq={abs.frequency})")

    # Demonstrate auto mode selection
    print_subheader("Auto Mode Selection (Surprise-Based)")

    # First, establish baseline with predictable input
    mind.process(["neural"])

    # Now process and let system decide
    auto_queries = [
        ["neural"],           # Expected - should stay FAST
        ["quantum", "computer"],  # Novel - might trigger SLOW
        ["neural"],           # Back to expected
    ]

    for query in auto_queries:
        result = mind.process(query, mode=None)  # Auto mode
        surprise_str = f", surprise={result.surprise.magnitude:.2f}" if result.surprise else ""
        print(f"\n  Query: {query}")
        print(f"    Auto-selected: {result.mode.name}{surprise_str}")

    # Demonstrate dual-path comparison
    print_subheader("Dual-Path Comparison")

    context = ["pattern", "recognition"]
    dual_result = mind.router.route_both(context)

    print(f"\n  Context: {context}")
    print(f"  FAST path: {sorted(list(dual_result.fast_result))[:5]}")
    print(f"  SLOW path: {sorted(list(dual_result.slow_result))[:5]}")
    if dual_result.surprise:
        print(f"  Path divergence (surprise): {dual_result.surprise.magnitude:.2f}")
    print(f"  Recommended mode: {dual_result.recommended_mode}")

    # Show system statistics
    print_subheader("System Statistics")
    stats = mind.get_stats()

    print(f"\n  Current Mode: {stats['mode']}")
    print(f"  Surprise Baseline: {stats['loom']['surprise_baseline']:.3f}")
    print(f"  Mode Transitions: {stats['loom']['transition_count']}")
    print(f"  Hive Nodes Tracked: {stats['hive']['total_nodes_tracked']}")
    print(f"  Cortex Observations: {stats['cortex']['total_observations']}")
    print(f"  Cortex Abstractions: {stats['cortex']['total_abstractions']}")

    # Demonstrate serialization
    if verbose:
        print_subheader("Serialization")
        state = mind.to_dict()
        print(f"  State keys: {list(state.keys())}")
        restored = WovenMind.from_dict(state)
        print(f"  Restored successfully: {restored is not None}")


def demo_integrated_workflow(corpus: list, verbose: bool = False):
    """
    Demonstrate the full integrated cognitive workflow.

    This shows how all components work together:
    1. Loom selects thinking mode
    2. PRISM-SLM provides activations
    3. Homeostasis regulates activity
    4. Patterns are detected
    5. Abstractions form
    6. Goals track progress
    """
    print_header("INTEGRATED WORKFLOW: Full Cognitive Architecture")

    # Initialize all components
    loom = Loom()
    model = PRISMLanguageModel(context_size=3)
    regulator = HomeostasisRegulator()
    engine = AbstractionEngine(min_frequency=2)  # Lower threshold for demo
    stack = GoalStack()

    # Train model
    for doc in corpus[:3]:
        model.train(doc["content"])

    # Create a learning goal
    goal = stack.push_goal(
        "Understand Document Concepts",
        target_nodes={"neural", "network", "learning", "algorithm"},
        priority=GoalPriority.HIGH,
    )

    print(f"\n  Goal: {goal.name}")
    print(f"  Target concepts: {set(goal.target_nodes)}")

    # Simulate processing several inputs
    print_subheader("Processing Inputs Through Architecture")

    inputs = [
        "What is a neural network?",
        "How does backpropagation work?",
        "Explain gradient descent optimization",
        "Neural networks learn patterns",
    ]

    all_activations = []

    for i, input_text in enumerate(inputs):
        print(f"\n  [{i+1}] Input: '{input_text}'")

        # 1. Loom decides thinking mode
        tokens = input_text.lower().split()
        predicted_token = model.generate_next(tokens[:3])

        # Create predicted/actual sets for surprise detection
        # If we predicted a token, use moderate confidence; otherwise low
        if predicted_token:
            predicted_probs = {predicted_token: 0.6}
        else:
            predicted_probs = {}
        actual_tokens = set(tokens[:2])

        signal = loom.detect_surprise(predicted_probs, actual_tokens)
        mode = loom.select_mode(signal)
        print(f"      Mode: {mode.name} (surprise={signal.magnitude:.2f})")

        # 2. Get sparse activations from PRISM
        activations = model.graph.sparse_activate(
            tokens[:2],
            k=max(5, int(model.vocab_size * 0.1)),
        )

        # 3. Apply homeostatic regulation
        regulator.record_activations(activations)
        regulated = regulator.apply_excitability(activations)

        active_tokens = [t for t, v in regulated.items() if v > 0.1][:5]
        print(f"      Active concepts: {active_tokens}")

        # 4. Feed to pattern detector
        if len(active_tokens) >= 2:
            pattern = frozenset(active_tokens[:3])
            engine.observe(pattern)
            all_activations.append(pattern)

        # 5. Check goal progress
        active_set = frozenset(active_tokens)
        stack.check_achievement(goal.id, active_set)
        print(f"      Goal progress: {goal.progress:.0%}")

    # Show what abstractions formed
    print_subheader("Abstractions Formed")

    formed = engine.auto_form_abstractions(max_new=3)
    if formed:
        for abs in formed:
            print(f"  {abs.id}: {set(abs.source_nodes)} (freq={abs.frequency})")
    else:
        print("  (No abstractions formed yet - need more observations)")

    # Final goal status
    print_subheader("Final Goal Status")
    print(f"  Goal: {goal.name}")
    print(f"  Status: {goal.status.name}")
    print(f"  Progress: {goal.progress:.0%}")

    # Architecture summary
    print_subheader("Architecture Summary")
    loom_history = loom.get_transition_history()
    home_metrics = regulator.get_health_metrics()
    goal_stats = stack.get_statistics()

    print(f"\n  Loom transitions: {len(loom_history)}")
    print(f"  PRISM vocabulary: {model.vocab_size} tokens")
    print(f"  Homeostasis avg excitability: {home_metrics['avg_excitability']:.2f}")
    print(f"  Abstractions formed: {len(engine.abstractions)}")
    print(f"  Goal completion: {goal_stats['avg_progress']:.0%}")


def demo_consolidation(corpus: list, verbose: bool = False):
    """
    Demonstrate Sprint 5: Consolidation Engine.

    Shows the "sleep-like" consolidation cycles that:
    - Transfer frequent Hive patterns to Cortex abstractions
    - Mine latent structure in patterns
    - Apply decay to unused connections
    """
    print_header("SPRINT 5: CONSOLIDATION ENGINE")

    from cortical.reasoning.consolidation import (
        ConsolidationEngine,
        ConsolidationConfig,
        ConsolidationResult,
    )

    # Create WovenMind for consolidation demo
    config = WovenMindConfig(
        consolidation_threshold=2,
        consolidation_decay_factor=0.9,
    )
    mind = WovenMind(config=config)

    # Train on corpus
    print_subheader("Training Phase")
    for doc in corpus[:5]:
        mind.train(doc["content"])
    print(f"  Trained on {min(5, len(corpus))} documents")

    # Process and record patterns
    print_subheader("Pattern Recording")
    patterns_recorded = 0
    for doc in corpus[:5]:
        tokens = doc["content"].lower().split()[:3]
        mind.process(tokens)
        if len(tokens) >= 2:
            mind.consolidation.record_pattern(set(tokens[:2]))
            patterns_recorded += 1
    print(f"  Recorded {patterns_recorded} patterns")

    # Simulate repeated exposure
    print_subheader("Repeated Exposure (Building Frequency)")
    for _ in range(5):
        for doc in corpus[:3]:
            tokens = doc["content"].lower().split()[:2]
            if len(tokens) >= 2:
                mind.consolidation.record_pattern(set(tokens))
    print("  Built up pattern frequencies")

    # Check frequent patterns
    frequent = mind.consolidation.get_frequent_patterns(min_frequency=3)
    print(f"  Frequent patterns (freq>=3): {len(frequent)}")

    # Run consolidation
    print_subheader("Consolidation Cycle ('Sleep')")
    result = mind.consolidate()

    print(f"  Patterns transferred: {result.patterns_transferred}")
    print(f"  Abstractions formed: {result.abstractions_formed}")
    print(f"  Connections decayed: {result.connections_decayed}")
    print(f"  Connections pruned: {result.connections_pruned}")
    print(f"  Cycle duration: {result.cycle_duration_ms:.2f}ms")

    # Run multiple cycles
    print_subheader("Multiple Consolidation Cycles")
    for i in range(3):
        r = mind.consolidate()
        print(f"  Cycle {i+2}: transferred={r.patterns_transferred}, "
              f"abstractions={r.abstractions_formed}, "
              f"duration={r.cycle_duration_ms:.2f}ms")

    # Final stats
    print_subheader("Consolidation Statistics")
    stats = mind.get_consolidation_stats()
    print(f"  Total cycles: {stats['total_cycles']}")
    print(f"  Total patterns transferred: {stats['total_patterns_transferred']}")
    print(f"  Total abstractions formed: {stats['total_abstractions_formed']}")
    print(f"  Avg cycle duration: {stats['avg_cycle_duration_ms']:.2f}ms")

    if verbose:
        print_subheader("Phase Breakdown")
        print(f"  Phase durations: {result.phase_durations_ms}")


def main():
    parser = argparse.ArgumentParser(
        description="Woven Mind + PRISM Marriage Demo"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--corpus", type=str, default="samples",
        help="Path to corpus directory (default: samples/)"
    )
    parser.add_argument(
        "--section", type=str, choices=["loom", "hive", "abstraction", "goals", "wovenmind", "consolidation", "all"],
        default="all",
        help="Which section to run (default: all)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  WOVEN MIND + PRISM MARRIAGE DEMO")
    print("  Cognitive Architecture Demonstration")
    print("=" * 70)

    # Load corpus
    corpus = load_corpus(args.corpus, max_docs=10)
    print(f"\n  Loaded {len(corpus)} documents from {args.corpus}/")

    if len(corpus) == 0:
        print("\n  Note: No corpus found. Using synthetic examples.\n")
        corpus = [
            {"id": "synthetic_1", "content": "Neural networks learn patterns from data through training."},
            {"id": "synthetic_2", "content": "Machine learning algorithms optimize parameters using gradient descent."},
            {"id": "synthetic_3", "content": "Deep learning models have multiple layers of neural networks."},
        ]

    # Run demos
    if args.section in ["loom", "all"]:
        demo_loom_dual_process(args.verbose)

    if args.section in ["hive", "all"]:
        demo_hebbian_hive(corpus, args.verbose)

    if args.section in ["abstraction", "all"]:
        demo_cortex_abstraction(args.verbose)

    if args.section in ["goals", "all"]:
        demo_goal_stack(args.verbose)

    if args.section in ["wovenmind", "all"]:
        demo_woven_mind_facade(corpus, args.verbose)

    if args.section in ["consolidation", "all"]:
        demo_consolidation(corpus, args.verbose)

    if args.section == "all":
        demo_integrated_workflow(corpus, args.verbose)

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
