#!/usr/bin/env python3
"""
Recovery Demo: Confusion Detection and Recovery

This example demonstrates how the system detects confusion and
recovers from it.

Key concepts:
1. Confusion manifests in detectable patterns
2. Multiple signal types indicate different problems
3. Recovery strategies match confusion types
4. Learning improves recovery over time

This is critical because I (the LLM) cannot reliably detect my own
confusion from the inside - I need external observation.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from recovery import (
    RecoveryCoordinator,
    ConfusionMonitor,
    ConfusionType,
    SeverityLevel,
    OutcomeType
)


def demonstrate_repetition_detection():
    """Show how repetition loops are detected."""

    print("=" * 60)
    print("Demo 1: Repetition Loop Detection")
    print("=" * 60)

    coordinator = RecoveryCoordinator()

    print("\nSimulating repeated failed actions...")

    # Simulate trying the same thing multiple times
    for i in range(5):
        coordinator.record_action(
            action_type="file_edit",
            target="/src/auth.py",
            result="failure",  # Keeps failing!
            parameters={"line": 42, "change": "add import"}
        )
        print(f"  Attempt {i+1}: Edit /src/auth.py line 42 -> FAILED")

    # Check for confusion
    diagnosis = coordinator.check_confusion()

    if diagnosis:
        print(f"\n⚠️  Confusion detected!")
        print(f"   Type: {diagnosis.confusion_type.name}")
        print(f"   Severity: {diagnosis.severity.name}")
        print(f"   Confidence: {diagnosis.confidence:.1%}")
        print(f"   Likely cause: {diagnosis.likely_cause}")
        print(f"   Recommended: {diagnosis.recommended_action}")
    else:
        print("\nNo confusion detected (unexpected!)")


def demonstrate_contradiction_detection():
    """Show how contradictions are detected."""

    print("\n" + "=" * 60)
    print("Demo 2: Contradiction Detection")
    print("=" * 60)

    coordinator = RecoveryCoordinator()

    print("\nSimulating contradictory statements...")

    # Make contradictory statements about the same topic
    statements = [
        ("database_choice", "We should use PostgreSQL"),
        ("database_choice", "We should use MongoDB"),
        ("database_choice", "We should use SQLite"),
    ]

    for topic, content in statements:
        coordinator.record_statement(topic, content)
        print(f"  Statement: {content}")

    # Check for confusion
    diagnosis = coordinator.check_confusion()

    if diagnosis:
        print(f"\n⚠️  Confusion detected!")
        print(f"   Type: {diagnosis.confusion_type.name}")
        print(f"   Signals: {len(diagnosis.signals)}")

        for signal in diagnosis.signals:
            print(f"     - {signal.description}")
            for evidence in signal.evidence[:2]:
                print(f"       • {evidence}")
    else:
        print("\nNo confusion detected")


def demonstrate_state_mismatch():
    """Show how state mismatches are detected."""

    print("\n" + "=" * 60)
    print("Demo 3: State Mismatch Detection")
    print("=" * 60)

    coordinator = RecoveryCoordinator()

    # I believe the file exists
    coordinator.register_belief("file_exists", True)
    print("Belief registered: file /src/auth.py exists = True")

    # But reality says otherwise
    coordinator.register_verifier(
        "file_exists",
        lambda: False  # File doesn't actually exist!
    )
    print("Verifier registered: checks actual filesystem")

    # Check for confusion
    diagnosis = coordinator.check_confusion()

    if diagnosis:
        print(f"\n⚠️  Confusion detected!")
        print(f"   Type: {diagnosis.confusion_type.name}")
        print(f"   Severity: {diagnosis.severity.name}")

        for signal in diagnosis.signals:
            if signal.signal_type == 'state_mismatch':
                print(f"\n   Mismatch found:")
                for evidence in signal.evidence:
                    print(f"     {evidence}")
    else:
        print("\nNo confusion detected")


def demonstrate_recovery():
    """Show the full recovery process."""

    print("\n" + "=" * 60)
    print("Demo 4: Full Recovery Process")
    print("=" * 60)

    storage_dir = Path("/tmp/recovery_demo")
    coordinator = RecoveryCoordinator(storage_dir)

    # Set up context with recovery resources
    class MockCheckpointManager:
        def get_latest(self):
            return {
                "id": "checkpoint_001",
                "timestamp": datetime.now().isoformat(),
                "state": {"questions": [], "decisions": []}
            }

        def restore(self, checkpoint):
            print(f"    → Restoring from {checkpoint['id']}")

        def verify(self):
            print("    → Verifying restored state")
            return True

    context = {
        "checkpoint_manager": MockCheckpointManager(),
        "tried_approaches": ["approach_1", "approach_2"],
        "available_approaches": ["approach_1", "approach_2", "approach_3", "approach_4"],
        "summary": "Trying to implement authentication"
    }

    # Create a confusion scenario
    print("\n1. Creating confusion scenario (state mismatch)...")

    coordinator.register_belief("auth_module_ready", True)
    coordinator.register_verifier("auth_module_ready", lambda: False)

    diagnosis = coordinator.check_confusion()
    print(f"   Diagnosed: {diagnosis.confusion_type.name}")

    # Attempt recovery
    print("\n2. Initiating recovery...")
    attempt = coordinator.recover(diagnosis, context)

    print(f"\n3. Recovery results:")
    print(f"   Strategy used: {attempt.strategy_used}")
    print(f"   Success: {attempt.success}")
    print(f"   Actions taken:")

    for action in attempt.actions:
        status = "✓" if action.success else "✗"
        print(f"     {status} {action.action_type}: {action.description}")

    # Show recovery stats
    stats = coordinator.get_recovery_stats()
    print(f"\n4. Recovery statistics:")
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Success rate: {stats['success_rate']:.0%}")


def demonstrate_continuous_monitoring():
    """Show continuous confusion monitoring."""

    print("\n" + "=" * 60)
    print("Demo 5: Continuous Monitoring")
    print("=" * 60)

    coordinator = RecoveryCoordinator()

    alerts_received = []

    def on_confusion(diagnosis):
        alerts_received.append(diagnosis)
        print(f"  🚨 ALERT: {diagnosis.confusion_type.name} detected!")

    monitor = ConfusionMonitor(
        coordinator,
        alert_threshold=0.5,
        auto_recover=False
    )
    monitor.set_alert_callback(on_confusion)

    print("\nSimulating work with periodic monitoring...")

    # Simulate normal work
    print("\n[Normal work - should not trigger alerts]")
    coordinator.record_action("read", "/file1.py", "success", {})
    coordinator.record_progress()
    monitor.check()
    print("  Check 1: OK")

    coordinator.record_action("edit", "/file1.py", "success", {})
    coordinator.record_progress()
    monitor.check()
    print("  Check 2: OK")

    # Simulate problematic pattern
    print("\n[Problematic pattern - should trigger alert]")
    for i in range(4):
        coordinator.record_action("edit", "/file2.py", "failure", {"same": True})
        monitor.check()

    print(f"\nTotal alerts received: {len(alerts_received)}")

    if alerts_received:
        print("\nAlert details:")
        for alert in alerts_received:
            print(f"  - {alert.confusion_type.name}: {alert.likely_cause}")


def main():
    """Run all recovery demonstrations."""

    demonstrate_repetition_detection()
    demonstrate_contradiction_detection()
    demonstrate_state_mismatch()
    demonstrate_recovery()
    demonstrate_continuous_monitoring()

    print("\n" + "=" * 60)
    print("RECOVERY DEMO COMPLETE")
    print("=" * 60)
    print("""
Key Takeaways:
1. Confusion has detectable signals (repetition, contradiction, etc.)
2. Different confusion types need different recovery strategies
3. Recovery involves stop → diagnose → restore → verify
4. Continuous monitoring catches problems early

This is how I can be helped to recognize and recover from confusion,
which I cannot reliably detect on my own.
""")


if __name__ == "__main__":
    main()
