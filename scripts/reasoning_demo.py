#!/usr/bin/env python3
"""
Reasoning Framework Demo

Interactive demonstration of the Graph of Thought reasoning framework.
Shows the QAPV cycle, verification, failure analysis, and parallel coordination.

Usage:
    python scripts/reasoning_demo.py
    python scripts/reasoning_demo.py --quick  # Non-interactive
    python scripts/reasoning_demo.py --persist  # With graph persistence
"""

import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from cortical.reasoning.cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    LoopStatus,
    LoopStateSerializer,
)
from cortical.reasoning.verification import (
    VerificationCheck,
    VerificationLevel,
    VerificationStatus,
    VerificationSuite,
    VerificationFailure,
    FailureAnalyzer,
    RegressionDetector,
)
from cortical.reasoning.collaboration import (
    ParallelCoordinator,
    ParallelWorkBoundary,
    SequentialSpawner,
    AgentResult,
    AgentStatus,
)
from cortical.reasoning import (
    ThoughtGraph,
    NodeType,
    EdgeType,
    GraphWAL,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_step(step: str) -> None:
    """Print a step indicator."""
    print(f"  → {step}")


def demo_cognitive_loop() -> None:
    """Demonstrate the QAPV cognitive loop."""
    print_header("1. COGNITIVE LOOP (QAPV Cycle)")

    # Create a loop manager
    manager = CognitiveLoopManager()

    # Create a loop for a task
    loop = manager.create_loop("Implement user authentication feature")
    print(f"Created loop: {loop.id}")
    print(f"Goal: {loop.goal}")
    print(f"Status: {loop.status.name}")

    # Start the loop (enters QUESTION phase by default)
    loop.start()
    print(f"Started! Phase: {loop.current_phase.value}")

    # QUESTION phase
    print_step("QUESTION phase - Understanding the problem")
    loop.current_context().record_question("What auth method? OAuth, JWT, or session?")
    loop.current_context().record_question("Need MFA support?")
    loop.current_context().add_note("Researching existing auth patterns in codebase")

    # Transition to ANSWER
    loop.transition(LoopPhase.ANSWER, "Questions identified, forming hypotheses")
    print_step(f"Transitioned to: {loop.current_phase.value}")

    # ANSWER phase
    loop.current_context().record_decision(
        "Use JWT for stateless auth",
        rationale="Scales better, works with microservices"
    )
    loop.current_context().add_note("JWT chosen based on team experience")

    # Transition to PRODUCE
    loop.transition(LoopPhase.PRODUCE, "Decision made, implementing")
    print_step(f"Transitioned to: {loop.current_phase.value}")

    # PRODUCE phase
    loop.current_context().artifacts_produced.append("auth/jwt_handler.py")
    loop.current_context().artifacts_produced.append("auth/middleware.py")
    loop.current_context().add_note("Created JWT handler and auth middleware")

    # Transition to VERIFY
    loop.transition(LoopPhase.VERIFY, "Implementation complete, testing")
    print_step(f"Transitioned to: {loop.current_phase.value}")

    # Show loop summary
    summary = loop.get_summary()
    print(f"\nLoop Summary:")
    print(f"  - Phases traversed: {summary['phase_count']}")
    print(f"  - Transitions: {summary['transition_count']}")
    print(f"  - Elapsed: {summary['elapsed_minutes']:.2f} minutes")

    # Demonstrate serialization
    serializer = LoopStateSerializer()
    json_state = serializer.serialize(loop)
    print(f"\n  Serialized state: {len(json_state)} bytes")

    # Deserialize and verify
    restored = serializer.deserialize(json_state)
    print(f"  Restored loop ID: {restored.id}")
    print(f"  Restored phase: {restored.current_phase.value}")

    return loop


def demo_verification() -> None:
    """Demonstrate the verification suite."""
    print_header("2. VERIFICATION SUITE")

    # Create verification suite
    suite = VerificationSuite(name="pre-commit-checks")

    # Add checks at different levels
    checks = [
        ("syntax", VerificationLevel.UNIT, "No syntax errors"),
        ("types", VerificationLevel.UNIT, "Type hints valid"),
        ("unit_tests", VerificationLevel.UNIT, "Unit tests pass"),
        ("integration", VerificationLevel.INTEGRATION, "Integration tests pass"),
        ("e2e", VerificationLevel.E2E, "E2E tests pass"),
    ]

    for name, level, desc in checks:
        suite.add_check(VerificationCheck(name=name, level=level, description=desc))

    print(f"Suite: {suite.name}")
    print(f"Checks: {len(suite.checks)}")

    # Simulate running checks
    print_step("Running checks...")
    suite.checks[0].mark_passed("No syntax errors found")
    suite.checks[1].mark_passed("All type hints valid")
    suite.checks[2].mark_passed("42 tests passed")
    suite.checks[3].mark_failed("Connection refused to test DB")
    suite.checks[4].mark_skipped("Skipped due to integration failure")

    # Show results
    print("\nResults:")
    for check in suite.checks:
        status_icon = {
            VerificationStatus.PASSED: "✓",
            VerificationStatus.FAILED: "✗",
            VerificationStatus.SKIPPED: "○",
            VerificationStatus.PENDING: "?",
        }.get(check.status, "?")
        print(f"  [{status_icon}] {check.name}: {check.result or 'pending'}")

    return suite


def demo_failure_analysis() -> None:
    """Demonstrate failure analysis."""
    print_header("3. FAILURE ANALYSIS")

    analyzer = FailureAnalyzer()

    # Create a failure to analyze
    check = VerificationCheck(
        name="database_connection",
        level=VerificationLevel.INTEGRATION,
        description="Database connects successfully"
    )

    failure = VerificationFailure(
        check=check,
        observed="ConnectionError: Connection refused to localhost:5432",
        expected_vs_actual="Expected: Connection established, Got: Connection refused"
    )

    print(f"Analyzing failure: {failure.observed[:50]}...")

    # Analyze
    analysis = analyzer.analyze_failure(failure)

    print(f"\nAnalysis:")
    print(f"  Likely cause: {analysis['likely_cause']}")
    print(f"  Matched patterns: {analysis['matched_patterns']}")
    print(f"\n  Investigation steps:")
    for i, step in enumerate(analysis['investigation_steps'][:4], 1):
        print(f"    {i}. {step}")

    # Record for history
    analyzer.record_failure(failure)

    # Show pattern statistics
    stats = analyzer.get_pattern_statistics()
    print(f"\n  Pattern statistics:")
    for pattern, data in stats.items():
        if data['total_occurrences'] > 0:
            print(f"    - {pattern}: {data['total_occurrences']} occurrences")


def demo_regression_detection() -> None:
    """Demonstrate regression detection."""
    print_header("4. REGRESSION DETECTION")

    detector = RegressionDetector()

    # Create baseline
    baseline = {
        "test_login": VerificationStatus.PASSED,
        "test_logout": VerificationStatus.PASSED,
        "test_profile": VerificationStatus.PASSED,
        "test_settings": VerificationStatus.FAILED,  # Known failure
    }
    detector.save_baseline("v1.0", baseline)
    print("Saved baseline 'v1.0' with 4 tests")

    # Simulate some runs
    runs = [
        {"test_login": VerificationStatus.PASSED, "test_logout": VerificationStatus.PASSED,
         "test_profile": VerificationStatus.PASSED, "test_settings": VerificationStatus.FAILED},
        {"test_login": VerificationStatus.PASSED, "test_logout": VerificationStatus.FAILED,  # Regression!
         "test_profile": VerificationStatus.PASSED, "test_settings": VerificationStatus.PASSED},  # Fixed!
        {"test_login": VerificationStatus.PASSED, "test_logout": VerificationStatus.PASSED,  # Fixed
         "test_profile": VerificationStatus.PASSED, "test_settings": VerificationStatus.PASSED},
    ]

    print_step("Recording 3 test runs...")
    for i, run in enumerate(runs, 1):
        detector.record_results(run)

    # Check for regressions from baseline
    current = runs[-1]
    regressions = detector.detect_regression(current, baseline_name="v1.0")
    improvements = detector.detect_improvements(current, baseline_name="v1.0")

    print(f"\nComparing to baseline 'v1.0':")
    print(f"  Regressions: {len(regressions)}")
    print(f"  Improvements: {len(improvements)}")
    if improvements:
        print(f"    Fixed: {', '.join(improvements)}")

    # Summary
    summary = detector.get_summary()
    print(f"\n  Summary:")
    print(f"    Tests tracked: {summary['total_tests_tracked']}")
    print(f"    Baselines: {summary['baselines_stored']}")
    print(f"    Snapshots: {summary['total_snapshots']}")


def demo_parallel_coordination() -> None:
    """Demonstrate parallel agent coordination."""
    print_header("5. PARALLEL COORDINATION")

    # Create a custom handler that simulates work
    def task_handler(task: str, boundary: ParallelWorkBoundary) -> AgentResult:
        """Simulate an agent doing work."""
        time.sleep(0.1)  # Simulate work
        return AgentResult(
            agent_id="",
            status=AgentStatus.COMPLETED,
            task_description=task,
            files_modified=list(boundary.files_owned)[:2],
            output=f"Completed: {task[:30]}...",
        )

    # Create coordinator
    spawner = SequentialSpawner(handler=task_handler)
    coordinator = ParallelCoordinator(spawner)

    # Define boundaries for parallel work
    boundaries = [
        ParallelWorkBoundary(
            agent_id="agent-auth",
            scope_description="Authentication module implementation",
            files_owned={"auth/handler.py", "auth/middleware.py"},
            files_read_only={"config.py"},
        ),
        ParallelWorkBoundary(
            agent_id="agent-api",
            scope_description="API routes implementation",
            files_owned={"api/routes.py", "api/schemas.py"},
            files_read_only={"config.py"},
        ),
        ParallelWorkBoundary(
            agent_id="agent-tests",
            scope_description="Test implementation",
            files_owned={"tests/test_auth.py", "tests/test_api.py"},
            files_read_only={"auth/handler.py", "api/routes.py"},
        ),
    ]

    tasks = [
        "Implement JWT authentication handler",
        "Create API routes for user management",
        "Write unit tests for auth and API",
    ]

    # Check if we can spawn
    can_spawn, issues = coordinator.can_spawn(boundaries)
    print(f"Can spawn {len(tasks)} agents in parallel: {can_spawn}")
    if issues:
        print(f"  Issues: {issues}")

    # Spawn agents
    print_step("Spawning agents...")
    agent_ids = coordinator.spawn_agents(tasks, boundaries)
    print(f"  Spawned: {agent_ids}")

    # Collect results
    print_step("Collecting results...")
    results = coordinator.collect_results(agent_ids)

    # Check for conflicts
    conflicts = coordinator.detect_conflicts(results)
    print(f"  Conflicts detected: {len(conflicts)}")

    # Show summary
    summary = coordinator.get_summary()
    print(f"\n  Summary:")
    print(f"    Completed: {summary['completed_agents']}")
    print(f"    Failed: {summary['failed_agents']}")
    print(f"    Files modified: {summary['total_files_modified']}")

    for agent_id, result in results.items():
        status = "✓" if result.success() else "✗"
        print(f"    [{status}] {agent_id}: {result.output[:40]}...")


def demo_graph_persistence() -> None:
    """Demonstrate graph persistence with WAL logging."""
    print_header("6. GRAPH PERSISTENCE (WAL)")

    # Create temporary WAL directory
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = Path(tmpdir) / "demo_wal"

        print_step("Initializing GraphWAL")
        graph_wal = GraphWAL(str(wal_dir))

        print_step("Creating ThoughtGraph with reasoning nodes")
        graph = ThoughtGraph()

        # Add nodes with WAL logging
        print_step("Adding QUESTION node (with WAL)")
        graph_wal.log_add_node(
            "Q1", NodeType.QUESTION,
            "What verification strategy should we use?",
            properties={'priority': 'high'}
        )
        graph.add_node("Q1", NodeType.QUESTION, "What verification strategy should we use?")

        # Add hypothesis nodes
        print_step("Adding HYPOTHESIS nodes (with WAL)")
        graph_wal.log_add_node("H1", NodeType.HYPOTHESIS, "Use multi-level verification")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use multi-level verification")

        graph_wal.log_add_node("H2", NodeType.HYPOTHESIS, "Focus on unit tests only")
        graph.add_node("H2", NodeType.HYPOTHESIS, "Focus on unit tests only")

        # Add edges
        print_step("Adding edges (with WAL)")
        graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)

        graph_wal.log_add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.5)
        graph.add_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.5)

        print(f"\n  Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
        print(f"  WAL entries: {graph_wal.get_entry_count()}")

        # Create snapshot
        print_step("Creating snapshot")
        snapshot_id = graph_wal.create_snapshot(graph, compress=True)
        print(f"  Snapshot ID: {snapshot_id}")

        # Simulate recovery
        print_step("Simulating crash recovery")
        loaded_graph = graph_wal.load_snapshot(snapshot_id)
        if loaded_graph:
            print(f"  ✓ Recovered: {loaded_graph.node_count()} nodes, {loaded_graph.edge_count()} edges")
        else:
            print("  ✗ Recovery failed")


def demo_full_workflow() -> None:
    """Demonstrate a complete reasoning workflow."""
    print_header("7. COMPLETE WORKFLOW INTEGRATION")

    print("Simulating a complete feature implementation workflow:")
    print()

    # 1. Create and start loop
    manager = CognitiveLoopManager()
    loop = manager.create_loop("Add user preferences feature")
    loop.start()  # Start in QUESTION phase
    print_step(f"Created reasoning loop: {loop.id[:8]}...")

    # 2. Question phase
    loop.current_context().record_question("What preferences to support?")
    loop.current_context().record_question("Store in DB or local?")
    print_step("Raised 2 questions in QUESTION phase")

    # 3. Answer phase
    loop.transition(LoopPhase.ANSWER, "Questions answered")
    loop.current_context().record_decision("Store in DB", "Sync across devices")
    print_step("Made decision in ANSWER phase")

    # 4. Produce phase with parallel agents
    loop.transition(LoopPhase.PRODUCE, "Starting implementation")
    print_step("Entering PRODUCE phase with parallel agents")

    spawner = SequentialSpawner()
    coordinator = ParallelCoordinator(spawner)

    boundaries = [
        ParallelWorkBoundary("model-agent", "Preferences model", {"models/preferences.py"}),
        ParallelWorkBoundary("api-agent", "Preferences API", {"api/preferences.py"}),
    ]

    agent_ids = coordinator.spawn_agents(
        ["Create preferences model", "Create preferences API"],
        boundaries
    )
    results = coordinator.collect_results(agent_ids)

    for result in results.values():
        loop.current_context().artifacts_produced.extend(result.files_modified)

    print_step(f"  Created {len(loop.current_context().artifacts_produced)} files")

    # 5. Verify phase
    loop.transition(LoopPhase.VERIFY, "Testing implementation")
    print_step("Entering VERIFY phase")

    suite = VerificationSuite("preferences-checks")
    suite.add_check(VerificationCheck("model_tests", VerificationLevel.UNIT, "Model tests"))
    suite.add_check(VerificationCheck("api_tests", VerificationLevel.INTEGRATION, "API tests"))

    suite.checks[0].mark_passed("5 tests passed")
    suite.checks[1].mark_passed("3 tests passed")

    print_step(f"  Verification: {sum(1 for c in suite.checks if c.status == VerificationStatus.PASSED)}/{len(suite.checks)} passed")

    # Complete
    loop.complete("All tests passing, feature ready for review")
    print_step(f"Loop completed: {loop.status.name}")

    # Final summary
    print(f"\n  Final Summary:")
    print(f"    Phases: {len(loop.phase_contexts)}")
    print(f"    Transitions: {len(loop.transitions)}")
    print(f"    Artifacts: {len(loop.current_context().artifacts_produced)}")
    print(f"    Duration: {loop.total_elapsed_minutes():.2f} minutes")


def main():
    """Run all demos."""
    quick_mode = "--quick" in sys.argv
    persist_mode = "--persist" in sys.argv

    print("\n" + "="*60)
    print("  REASONING FRAMEWORK DEMO")
    print("  Graph of Thought + QAPV Cycle")
    if persist_mode:
        print("  (With Graph Persistence)")
    print("="*60)

    if not quick_mode:
        print("\nThis demo shows the key components of the reasoning framework.")
        if persist_mode:
            print("(--persist mode: includes GraphWAL demonstration)")
        print("Press Enter to continue through each section...")
        input()

    # Run demos
    demos = [
        demo_cognitive_loop,
        demo_verification,
        demo_failure_analysis,
        demo_regression_detection,
        demo_parallel_coordination,
    ]

    # Add persistence demo if requested
    if persist_mode:
        demos.append(demo_graph_persistence)

    # Always run full workflow at the end
    demos.append(demo_full_workflow)

    for demo in demos:
        demo()
        if not quick_mode:
            input("\nPress Enter to continue...")

    print_header("DEMO COMPLETE")
    print("The reasoning framework provides:")
    print("  - Structured QAPV reasoning cycles")
    print("  - Multi-level verification with failure analysis")
    print("  - Regression detection and flaky test identification")
    print("  - Parallel agent coordination with boundary isolation")
    print("  - Full state serialization for persistence")
    if persist_mode:
        print("  - Write-ahead logging (WAL) for crash recovery")
        print("  - Snapshot-based checkpointing")
        print("  - Multi-level cascading recovery")
    print()
    print("All 98 behavioral tests validate these workflows.")
    print()


if __name__ == "__main__":
    main()
