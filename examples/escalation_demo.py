"""
Demonstration of the Escalation Protocol for Director-Worker coordination.

This script shows how the EscalationManager handles worker confusion signals
and determines appropriate escalation levels.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_orchestration.escalation import (
    EscalationLevel,
    EscalationManager,
)
from llm_orchestration.recovery import ConfusionSignal


def main():
    """Demonstrate escalation protocol in action."""

    print("=" * 70)
    print("ESCALATION PROTOCOL DEMONSTRATION")
    print("=" * 70)
    print()

    # Create escalation manager
    manager = EscalationManager()

    # Scenario 1: Low severity confusion (gradual escalation)
    print("Scenario 1: Gradual escalation with low severity confusions")
    print("-" * 70)

    worker_id = "worker-alpha"
    task_id = "task-001"

    # First confusion: LOW severity
    signal1 = ConfusionSignal(
        signal_type="repetition",
        description="Worker repeating same action",
        evidence=["action1", "action1", "action1"],
        confidence=0.4,  # Low confidence
        source="Director"
    )

    protocol1 = manager.evaluate(worker_id, signal1, task_id)
    print(f"1st Confusion (LOW): {protocol1.level.name}")
    print(f"   Action: {protocol1.recommended_action[:60]}...")
    print()

    # Second confusion: Still LOW
    signal2 = ConfusionSignal(
        signal_type="repetition",
        description="Worker still repeating",
        evidence=["action2", "action2"],
        confidence=0.3,
        source="Director"
    )

    protocol2 = manager.evaluate(worker_id, signal2, task_id)
    print(f"2nd Confusion (LOW): {protocol2.level.name}")
    print(f"   Action: {protocol2.recommended_action[:60]}...")
    print()

    # Third confusion: Triggers ABORT
    signal3 = ConfusionSignal(
        signal_type="context_loss",
        description="Worker lost context",
        evidence=["no_context"],
        confidence=0.5,
        source="Director"
    )

    protocol3 = manager.evaluate(worker_id, signal3, task_id)
    print(f"3rd Confusion (MEDIUM): {protocol3.level.name}")
    print(f"   Action: {protocol3.recommended_action[:60]}...")
    print(f"   Total strikes: {manager.get_worker_strikes(worker_id)}")
    print()

    # Scenario 2: High severity confusion (fast escalation)
    print()
    print("Scenario 2: Fast escalation with high severity confusions")
    print("-" * 70)

    manager2 = EscalationManager()
    worker_id2 = "worker-beta"
    task_id2 = "task-002"

    # First confusion: HIGH severity
    signal_high1 = ConfusionSignal(
        signal_type="critical_error",
        description="Worker encountered critical error",
        evidence=["error: critical failure"],
        confidence=0.85,  # High confidence
        source="Director"
    )

    protocol_high1 = manager2.evaluate(worker_id2, signal_high1, task_id2)
    print(f"1st Confusion (HIGH): {protocol_high1.level.name}")
    print(f"   Skipped MONITOR, went straight to {protocol_high1.level.name}")
    print(f"   Action: {protocol_high1.recommended_action[:60]}...")
    print()

    # Second confusion: HIGH severity -> ESCALATE
    signal_high2 = ConfusionSignal(
        signal_type="critical_error",
        description="Worker cannot recover",
        evidence=["error: unrecoverable"],
        confidence=0.9,  # Critical
        source="Director"
    )

    protocol_high2 = manager2.evaluate(worker_id2, signal_high2, task_id2)
    print(f"2nd Confusion (CRITICAL): {protocol_high2.level.name}")
    print(f"   Action: {protocol_high2.recommended_action[:60]}...")
    print()

    # Scenario 3: Different workers tracked independently
    print()
    print("Scenario 3: Independent tracking of multiple workers")
    print("-" * 70)

    manager3 = EscalationManager()

    # Worker 1 gets 2 confusions
    for i in range(2):
        sig = ConfusionSignal(
            signal_type="test",
            description="Test",
            evidence=["e"],
            confidence=0.5,
            source="Director"
        )
        manager3.evaluate("worker-1", sig, "task-1")

    # Worker 2 gets 1 confusion
    sig = ConfusionSignal(
        signal_type="test",
        description="Test",
        evidence=["e"],
        confidence=0.5,
        source="Director"
    )
    manager3.evaluate("worker-2", sig, "task-2")

    print(f"Worker-1 strikes: {manager3.get_worker_strikes('worker-1')}")
    print(f"Worker-2 strikes: {manager3.get_worker_strikes('worker-2')}")
    print(f"Workers tracked independently: worker-1 has 2 strikes, worker-2 has 1")
    print()

    # Show escalation history
    print()
    print("Escalation History Summary")
    print("-" * 70)

    all_protocols = [protocol1, protocol2, protocol3, protocol_high1, protocol_high2]

    for i, prot in enumerate(all_protocols, 1):
        print(f"{i}. Worker: {prot.worker_id} | Level: {prot.level.name} | "
              f"Confusions: {len(prot.confusion_history)}")

    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
