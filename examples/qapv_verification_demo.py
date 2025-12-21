#!/usr/bin/env python3
"""
QAPV Verification Demo - Behavioral Verification for Cognitive Loops.

This demo shows how to use QAPVVerifier to detect behavioral anomalies
in QAPV (Question→Answer→Produce→Verify) cognitive loops.

Run:
    python examples/qapv_verification_demo.py
"""

import time
from cortical.reasoning import (
    CognitiveLoop,
    LoopPhase,
    QAPVVerifier,
    QAPVAnomaly,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_healthy_cycle():
    """Demonstrate a healthy QAPV cycle."""
    print_section("Healthy QAPV Cycle")

    verifier = QAPVVerifier()
    loop = CognitiveLoop(goal="Implement authentication feature")

    # Perfect QAPV cycle
    print("Executing phases: QUESTION → ANSWER → PRODUCE → VERIFY → COMPLETE")
    loop.start(LoopPhase.QUESTION)
    loop.transition(LoopPhase.ANSWER, reason="Requirements clarified")
    loop.transition(LoopPhase.PRODUCE, reason="Solution designed")
    loop.transition(LoopPhase.VERIFY, reason="Implementation complete")

    # Record all transitions
    for transition in loop.transitions:
        verifier.record_transition(
            transition.from_phase.value if transition.from_phase else None,
            transition.to_phase.value
        )

    # Check health
    report = verifier.get_diagnostic_report()
    print(f"Health Status: {report['health_status'].upper()}")
    print(f"Total Transitions: {report['total_transitions']}")
    print(f"Complete Cycles: {report['cycle_count']}")
    print(f"Anomalies Detected: {report['total_anomalies']}")

    if report['anomalies']:
        print("\nAnomalies:")
        for anomaly in report['anomalies']:
            print(f"  [{anomaly.severity.upper()}] {anomaly.description}")
    else:
        print("\n✓ No anomalies detected - healthy cycle!")


def demo_invalid_transitions():
    """Demonstrate detection of invalid transitions."""
    print_section("Invalid Transition Detection")

    verifier = QAPVVerifier()

    # Simulate skipping ANSWER phase
    print("Attempting invalid transition: QUESTION → PRODUCE (skipping ANSWER)")
    verifier.record_transition(None, "question")
    verifier.record_transition("question", "produce")  # Invalid!

    anomalies = verifier.check_health()

    print(f"\nDetected {len(anomalies)} anomaly(ies):")
    for anomaly in anomalies:
        print(f"\n  Type: {anomaly.anomaly_type.value}")
        print(f"  Severity: {anomaly.severity.upper()}")
        print(f"  Description: {anomaly.description}")
        print(f"  Suggestions:")
        for suggestion in anomaly.suggestions:
            print(f"    - {suggestion}")


def demo_stuck_phase():
    """Demonstrate detection of stuck phase."""
    print_section("Stuck Phase Detection")

    verifier = QAPVVerifier(stuck_threshold_seconds=2.0)

    print("Entering QUESTION phase and staying too long...")
    verifier.record_transition(None, "question")

    print("Waiting 2.5 seconds (threshold: 2.0s)...")
    time.sleep(2.5)

    anomalies = verifier.check_health()

    stuck_anomalies = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.STUCK_PHASE]
    if stuck_anomalies:
        anomaly = stuck_anomalies[0]
        print(f"\n✓ Detected stuck phase!")
        print(f"  Severity: {anomaly.severity.upper()}")
        print(f"  Description: {anomaly.description}")


def demo_infinite_loop():
    """Demonstrate detection of infinite loops."""
    print_section("Infinite Loop Detection")

    verifier = QAPVVerifier(max_cycles_before_warning=3)

    print("Executing 3 complete QAPV cycles without finishing...")
    for i in range(3):
        print(f"  Cycle {i + 1}...")
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")
        verifier.record_transition("answer", "produce")
        verifier.record_transition("produce", "verify")
        verifier.record_transition("verify", "question")  # Start new cycle

    print(f"\nTotal cycles: {verifier.get_cycle_count()}")

    anomalies = verifier.check_health()
    loop_anomalies = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.INFINITE_LOOP]

    if loop_anomalies:
        anomaly = loop_anomalies[0]
        print(f"\n✓ Detected infinite loop!")
        print(f"  Severity: {anomaly.severity.upper()}")
        print(f"  Description: {anomaly.description}")
        print(f"  Suggestions:")
        for suggestion in anomaly.suggestions[:2]:  # First 2
            print(f"    - {suggestion}")


def demo_premature_exit():
    """Demonstrate detection of premature exit."""
    print_section("Premature Exit Detection")

    verifier = QAPVVerifier()

    print("Completing from PRODUCE phase without verification...")
    verifier.record_transition(None, "question")
    verifier.record_transition("question", "answer")
    verifier.record_transition("answer", "produce")
    verifier.record_transition("produce", "complete")  # Should verify first!

    anomalies = verifier.check_health()
    exit_anomalies = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.PREMATURE_EXIT]

    if exit_anomalies:
        anomaly = exit_anomalies[0]
        print(f"\n✓ Detected premature exit!")
        print(f"  Severity: {anomaly.severity.upper()}")
        print(f"  Description: {anomaly.description}")


def demo_missing_production():
    """Demonstrate detection of missing production."""
    print_section("Missing Production Detection")

    verifier = QAPVVerifier()

    print("Entering VERIFY phase without producing artifacts...")
    verifier.record_transition(None, "question")
    verifier.record_transition("question", "answer")
    verifier.record_transition("answer", "verify")  # Skipped produce!

    anomalies = verifier.check_health()
    prod_anomalies = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.MISSING_PRODUCTION]

    if prod_anomalies:
        anomaly = prod_anomalies[0]
        print(f"\n✓ Detected missing production!")
        print(f"  Severity: {anomaly.severity.upper()}")
        print(f"  Description: {anomaly.description}")


def demo_comprehensive_report():
    """Demonstrate comprehensive diagnostic reporting."""
    print_section("Comprehensive Diagnostic Report")

    verifier = QAPVVerifier()

    # Create a complex scenario with multiple issues
    print("Creating complex scenario with multiple transitions...")
    verifier.record_transition(None, "question")
    verifier.record_transition("question", "answer")
    verifier.record_transition("answer", "produce")
    verifier.record_transition("produce", "verify")
    verifier.record_transition("verify", "question")  # New cycle
    verifier.record_transition("question", "answer")
    verifier.record_transition("answer", "verify")  # Skip produce - anomaly!

    report = verifier.get_diagnostic_report()

    print(f"\n{'─' * 50}")
    print(f"Diagnostic Report")
    print(f"{'─' * 50}")
    print(f"Health Status:      {report['health_status'].upper()}")
    print(f"Current Phase:      {report['current_phase']}")
    print(f"Total Transitions:  {report['total_transitions']}")
    print(f"Complete Cycles:    {report['cycle_count']}")
    print(f"Total Anomalies:    {report['total_anomalies']}")

    if report['anomalies_by_type']:
        print(f"\nAnomalies by Type:")
        for anomaly_type, count in report['anomalies_by_type'].items():
            print(f"  {anomaly_type}: {count}")

    if report['anomalies']:
        print(f"\nDetailed Anomalies:")
        for i, anomaly in enumerate(report['anomalies'], 1):
            print(f"\n  {i}. [{anomaly.severity.upper()}] {anomaly.anomaly_type.value}")
            print(f"     {anomaly.description}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  QAPV Verification Demo")
    print("  Behavioral Verification for Cognitive Loops")
    print("=" * 70)

    try:
        # Run demos
        demo_healthy_cycle()
        demo_invalid_transitions()
        demo_stuck_phase()
        demo_infinite_loop()
        demo_premature_exit()
        demo_missing_production()
        demo_comprehensive_report()

        # Summary
        print_section("Summary")
        print("QAPVVerifier can detect:")
        print("  ✓ Invalid transitions (state machine violations)")
        print("  ✓ Stuck phases (spending too long in one phase)")
        print("  ✓ Infinite loops (too many cycles without completion)")
        print("  ✓ Premature exits (completing without verification)")
        print("  ✓ Missing production (verifying without producing)")
        print("\nIntegration:")
        print("  - Works with CognitiveLoop from cortical.reasoning")
        print("  - Provides actionable suggestions for each anomaly")
        print("  - Supports comprehensive diagnostic reporting")
        print("  - Helps ensure QAPV cycles follow correct patterns")
        print()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        raise


if __name__ == '__main__':
    main()
