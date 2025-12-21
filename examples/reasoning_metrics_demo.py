"""
Reasoning Metrics Demo

Demonstrates integration of ReasoningMetrics with:
- CognitiveLoop and CognitiveLoopManager
- VerificationManager (for verification tracking)
- CrisisManager (for crisis tracking)
- Integration with cortical.observability metrics

This shows how to use metrics for observability in reasoning workflows.
"""

import time
from cortical.reasoning import (
    # Core loop
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    TerminationReason,
    # Verification
    VerificationManager,
    VerificationLevel,
    VerificationPhase,
    create_drafting_checklist,
    # Crisis
    CrisisManager,
    CrisisLevel,
    RecoveryAction,
)
from cortical.reasoning.metrics import (
    ReasoningMetrics,
    create_loop_metrics_handler,
)


def demo_basic_metrics():
    """Demonstrate basic metrics collection."""
    print("=" * 80)
    print("DEMO 1: Basic Metrics Collection")
    print("=" * 80)

    metrics = ReasoningMetrics()

    # Simulate loop phases with timing
    print("\nSimulating QAPV loop phases...")

    with metrics.phase_timer(LoopPhase.QUESTION):
        print("  - QUESTION phase (50ms)")
        time.sleep(0.05)
        metrics.record_question("clarification")
        metrics.record_question("technical")

    with metrics.phase_timer(LoopPhase.ANSWER):
        print("  - ANSWER phase (100ms)")
        time.sleep(0.1)
        metrics.record_decision("architecture")
        metrics.record_decision("implementation")

    with metrics.phase_timer(LoopPhase.PRODUCE):
        print("  - PRODUCE phase (200ms)")
        time.sleep(0.2)
        metrics.record_production("code")
        metrics.record_production("test")

    with metrics.phase_timer(LoopPhase.VERIFY):
        print("  - VERIFY phase (75ms)")
        time.sleep(0.075)
        metrics.record_verification(passed=True)
        metrics.record_verification(passed=True)
        metrics.record_verification(passed=False)

    print("\n" + metrics.get_summary())


def demo_loop_integration():
    """Demonstrate automatic metrics via CognitiveLoopManager integration."""
    print("\n" + "=" * 80)
    print("DEMO 2: CognitiveLoopManager Integration")
    print("=" * 80)

    metrics = ReasoningMetrics()
    manager = CognitiveLoopManager()

    # Register handler for automatic tracking
    handler = create_loop_metrics_handler(metrics)
    manager.register_transition_handler(handler)

    print("\nRunning cognitive loop with automatic metrics...")

    # Create and execute loop
    loop = manager.create_loop("Implement authentication feature")
    metrics.record_loop_start()

    # QUESTION phase
    ctx = loop.start(LoopPhase.QUESTION)
    print("  - Started QUESTION phase")
    ctx.record_question("What auth method to use?")
    ctx.record_question("Where to store credentials?")
    metrics.record_question("requirements")
    metrics.record_question("security")
    time.sleep(0.03)

    # ANSWER phase
    ctx = loop.transition(LoopPhase.ANSWER, reason="Requirements clarified")
    print("  - Transitioned to ANSWER phase")
    ctx.record_decision("Use OAuth 2.0", rationale="Industry standard")
    ctx.record_decision("Store in encrypted database", rationale="Security")
    metrics.record_decision("architecture")
    metrics.record_decision("security")
    time.sleep(0.05)

    # PRODUCE phase
    ctx = loop.transition(LoopPhase.PRODUCE, reason="Design complete")
    print("  - Transitioned to PRODUCE phase")
    ctx.artifacts_produced.extend(["auth.py", "tests/test_auth.py"])
    metrics.record_production("code")
    metrics.record_production("test")
    time.sleep(0.08)

    # VERIFY phase
    ctx = loop.transition(LoopPhase.VERIFY, reason="Implementation complete")
    print("  - Transitioned to VERIFY phase")
    metrics.record_verification(passed=True, level="unit")
    metrics.record_verification(passed=True, level="integration")
    time.sleep(0.04)

    # Complete
    loop.complete(TerminationReason.SUCCESS)
    metrics.record_loop_complete(success=True)
    print("  - Loop completed successfully")

    print("\n" + metrics.get_summary())


def demo_verification_integration():
    """Demonstrate integration with VerificationManager."""
    print("\n" + "=" * 80)
    print("DEMO 3: VerificationManager Integration")
    print("=" * 80)

    metrics = ReasoningMetrics()
    vm = VerificationManager()

    print("\nRunning verification suite with metrics...")

    # Create verification suite
    suite = vm.create_suite("drafting", "Drafting phase checks")
    drafting_checks = create_drafting_checklist()
    for check in drafting_checks[:5]:  # Add first 5 checks
        suite.add_check(check)

    # Run checks and record metrics
    for check in suite.checks:  # Run all checks
        print(f"  - Running: {check.name}")
        time.sleep(0.01)  # Simulate check execution

        # Simulate pass/fail
        if check.name.startswith("Syntax"):
            check.mark_passed()
            metrics.record_verification(passed=True, level=check.level.name.lower())
        else:
            check.mark_failed("Simulated failure")
            metrics.record_verification(passed=False, level=check.level.name.lower())

    # Display results
    pass_rate = metrics.get_verification_pass_rate()
    print(f"\n  Verification pass rate: {pass_rate:.1f}%")
    print(f"  Passed: {metrics.verifications_passed}")
    print(f"  Failed: {metrics.verifications_failed}")


def demo_crisis_integration():
    """Demonstrate integration with CrisisManager."""
    print("\n" + "=" * 80)
    print("DEMO 4: CrisisManager Integration")
    print("=" * 80)

    metrics = ReasoningMetrics()
    cm = CrisisManager()

    print("\nSimulating crisis events with metrics...")

    # Simulate HICCUP (recovered)
    print("  - HICCUP: Test failure with obvious fix")
    crisis1 = cm.record_crisis(
        CrisisLevel.HICCUP,
        "Test failed due to missing import",
        context={'test': 'test_auth.py', 'error': 'ImportError'}
    )
    crisis1.resolve(RecoveryAction.CONTINUE, "Added missing import")
    metrics.record_crisis(recovered=True, level="hiccup")

    # Simulate OBSTACLE (recovered)
    print("  - OBSTACLE: Verification repeatedly failing")
    crisis2 = cm.record_crisis(
        CrisisLevel.OBSTACLE,
        "Integration tests failing after 3 attempts",
        context={'attempts': 3, 'test': 'test_integration.py'}
    )
    crisis2.resolve(RecoveryAction.ADAPT, "Changed approach to mocking")
    metrics.record_crisis(recovered=True, level="obstacle")

    # Simulate WALL (escalated, not recovered)
    print("  - WALL: Fundamental assumption proven false")
    crisis3 = cm.record_crisis(
        CrisisLevel.WALL,
        "API doesn't support required authentication method",
        context={'issue': 'OAuth2 not supported by third-party API'}
    )
    crisis3.resolve(RecoveryAction.ESCALATE, "Escalated to human for decision")
    metrics.record_crisis(recovered=False, level="wall")

    # Display results
    recovery_rate = metrics.get_crisis_recovery_rate()
    print(f"\n  Crisis recovery rate: {recovery_rate:.1f}%")
    print(f"  Detected: {metrics.crises_detected}")
    print(f"  Recovered: {metrics.crises_recovered}")


def demo_metrics_export():
    """Demonstrate metrics export in observability format."""
    print("\n" + "=" * 80)
    print("DEMO 5: Metrics Export (Observability Format)")
    print("=" * 80)

    metrics = ReasoningMetrics()

    # Add various metrics
    with metrics.phase_timer(LoopPhase.QUESTION):
        time.sleep(0.02)

    with metrics.phase_timer(LoopPhase.ANSWER):
        time.sleep(0.03)

    metrics.record_decision("architecture")
    metrics.record_decision("implementation")
    metrics.record_question("technical")
    metrics.record_verification(passed=True)
    metrics.record_verification(passed=True)
    metrics.record_verification(passed=False)
    metrics.record_crisis(recovered=True)
    metrics.record_loop_start()
    metrics.record_loop_complete(success=True)

    # Export in observability format
    print("\nMetrics in dictionary format (compatible with observability.py):")
    metrics_dict = metrics.get_metrics_dict()

    for key, value in sorted(metrics_dict.items()):
        if 'count' in value:
            print(f"  {key}: {value['count']} operations")
        elif 'avg_ms' in value:
            print(f"  {key}: {value['count']} entries, avg={value['avg_ms']:.2f}ms")
        elif 'value' in value:
            print(f"  {key}: {value['value']:.1f}%")


def demo_full_workflow():
    """Demonstrate complete workflow with all integrations."""
    print("\n" + "=" * 80)
    print("DEMO 6: Complete Workflow with Integrated Metrics")
    print("=" * 80)

    # Initialize all components
    metrics = ReasoningMetrics()
    manager = CognitiveLoopManager()
    handler = create_loop_metrics_handler(metrics)
    manager.register_transition_handler(handler)

    print("\nExecuting complete reasoning workflow...")

    # Start loop
    loop = manager.create_loop("Build new feature")
    metrics.record_loop_start()

    # QUESTION
    loop.start(LoopPhase.QUESTION)
    metrics.record_question("requirements")
    metrics.record_question("constraints")
    time.sleep(0.02)

    # ANSWER
    loop.transition(LoopPhase.ANSWER, reason="Requirements gathered")
    metrics.record_decision("architecture")
    metrics.record_decision("design")
    time.sleep(0.03)

    # PRODUCE (with minor crisis)
    loop.transition(LoopPhase.PRODUCE, reason="Design approved")
    metrics.record_production("code")
    metrics.record_crisis(recovered=True, level="hiccup")  # Minor issue, recovered
    metrics.record_production("test")
    time.sleep(0.05)

    # VERIFY
    loop.transition(LoopPhase.VERIFY, reason="Code complete")
    metrics.record_verification(passed=True)
    metrics.record_verification(passed=True)
    metrics.record_verification(passed=False)  # One failure
    time.sleep(0.02)

    # Back to PRODUCE to fix issue
    loop.transition(LoopPhase.PRODUCE, reason="Fix test failure")
    metrics.record_production("code")
    time.sleep(0.01)

    # VERIFY again
    loop.transition(LoopPhase.VERIFY, reason="Fix applied")
    metrics.record_verification(passed=True)
    time.sleep(0.01)

    # Complete
    loop.complete(TerminationReason.SUCCESS)
    metrics.record_loop_complete(success=True)

    print("\n" + metrics.get_summary())

    # Show completion rate
    print(f"\nLoop completion rate: {metrics.get_loop_completion_rate():.1f}%")
    print(f"Verification pass rate: {metrics.get_verification_pass_rate():.1f}%")
    print(f"Crisis recovery rate: {metrics.get_crisis_recovery_rate():.1f}%")


def main():
    """Run all demos."""
    demo_basic_metrics()
    demo_loop_integration()
    demo_verification_integration()
    demo_crisis_integration()
    demo_metrics_export()
    demo_full_workflow()

    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
