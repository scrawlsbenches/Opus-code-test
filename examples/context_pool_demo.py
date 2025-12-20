"""
Example: Multi-Agent Coordination with ContextPool

Demonstrates how agents share findings via a context pool.
"""

import time
from pathlib import Path
from cortical.reasoning.context_pool import (
    ContextPool,
    ContextFinding,
    ConflictResolutionStrategy
)


def agent_callback(agent_name: str):
    """Create a callback for an agent to receive findings."""
    def callback(finding: ContextFinding):
        print(f"\n[{agent_name}] 🔔 Received finding on '{finding.topic}':")
        print(f"  From: {finding.source_agent}")
        print(f"  Content: {finding.content}")
        print(f"  Confidence: {finding.confidence}")
    return callback


def example_basic_usage():
    """Basic publish/query example."""
    print("=" * 70)
    print("Example 1: Basic Publish and Query")
    print("=" * 70)

    # Create pool with 1-hour TTL
    pool = ContextPool(ttl_seconds=3600)

    # Agent A publishes a finding
    pool.publish(
        topic="file_structure",
        content="Authentication code is in cortical/auth.py",
        source_agent="agent_a",
        confidence=0.95,
        metadata={"task_id": "T-001", "file_path": "cortical/auth.py"}
    )

    # Agent B queries the finding
    findings = pool.query("file_structure")
    print(f"\nAgent B found {len(findings)} findings on 'file_structure':")
    for f in findings:
        print(f"  - {f.content} (from {f.source_agent}, confidence: {f.confidence})")

    print(f"\nTotal findings in pool: {pool.count()}")
    print(f"Topics: {pool.get_topics()}")


def example_subscriptions():
    """Example with subscriptions."""
    print("\n" + "=" * 70)
    print("Example 2: Subscriptions")
    print("=" * 70)

    pool = ContextPool()

    # Agent B subscribes to bug findings
    pool.subscribe("bug_analysis", agent_callback("Agent B"))
    pool.subscribe("bug_analysis", agent_callback("Agent C"))

    # Agent A publishes a bug finding
    print("\nAgent A publishing bug finding...")
    pool.publish(
        topic="bug_analysis",
        content="Found null pointer dereference in login.py:42",
        source_agent="agent_a",
        confidence=0.9,
        metadata={"severity": "high", "line": 42}
    )


def example_conflicts():
    """Example with conflict detection."""
    print("\n" + "=" * 70)
    print("Example 3: Conflict Detection")
    print("=" * 70)

    # Manual conflict resolution (director must resolve)
    pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

    # Agent A's finding
    pool.publish(
        topic="code_location",
        content="The validation logic is in validators.py",
        source_agent="agent_a",
        confidence=0.8
    )

    # Agent B's conflicting finding
    pool.publish(
        topic="code_location",
        content="The validation logic is in utils/validation.py",
        source_agent="agent_b",
        confidence=0.9
    )

    print(f"\nFindings on 'code_location': {pool.count('code_location')}")
    for f in pool.query("code_location"):
        print(f"  - {f.content} (agent: {f.source_agent}, conf: {f.confidence})")

    conflicts = pool.get_conflicts()
    print(f"\nDetected {len(conflicts)} conflicts:")
    for f1, f2 in conflicts:
        print(f"\n  Conflict:")
        print(f"    {f1.source_agent}: {f1.content} (conf: {f1.confidence})")
        print(f"    {f2.source_agent}: {f2.content} (conf: {f2.confidence})")


def example_conflict_resolution():
    """Example with automatic conflict resolution."""
    print("\n" + "=" * 70)
    print("Example 4: Automatic Conflict Resolution")
    print("=" * 70)

    # Highest confidence wins
    pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)

    pool.publish(
        topic="performance",
        content="Query takes 500ms on average",
        source_agent="agent_a",
        confidence=0.7
    )

    pool.publish(
        topic="performance",
        content="Query takes 450ms on average",
        source_agent="agent_b",
        confidence=0.9  # Higher confidence
    )

    findings = pool.query("performance")
    print(f"\nFindings after conflict resolution: {len(findings)}")
    for f in findings:
        print(f"  - {f.content} (agent: {f.source_agent}, conf: {f.confidence})")


def example_ttl_expiration():
    """Example with TTL expiration."""
    print("\n" + "=" * 70)
    print("Example 5: TTL Expiration")
    print("=" * 70)

    # 2-second TTL for demo
    pool = ContextPool(ttl_seconds=2)

    pool.publish(
        topic="temp_finding",
        content="This will expire soon",
        source_agent="agent_a"
    )

    print(f"\nFindings before expiration: {pool.count()}")

    print("Waiting 3 seconds...")
    time.sleep(3)

    print(f"Findings after expiration: {pool.count()}")


def example_persistence():
    """Example with persistence."""
    print("\n" + "=" * 70)
    print("Example 6: Persistence")
    print("=" * 70)

    storage_dir = Path("/tmp/context_pool_demo")

    # Create pool and publish findings
    pool = ContextPool(storage_dir=storage_dir)

    pool.publish(
        topic="architecture",
        content="System uses layered architecture",
        source_agent="agent_a",
        confidence=1.0
    )

    pool.publish(
        topic="architecture",
        content="Database is PostgreSQL",
        source_agent="agent_b",
        confidence=0.95
    )

    # Save pool state
    pool.save()
    print(f"\nSaved pool to {storage_dir / 'context_pool.json'}")
    print(f"Findings before save: {pool.count()}")

    # Load into new pool
    new_pool = ContextPool(storage_dir=storage_dir)
    new_pool.load()
    print(f"Findings after load: {new_pool.count()}")

    for f in new_pool.query_all():
        print(f"  - [{f.topic}] {f.content}")


def example_multi_agent_workflow():
    """Complete multi-agent workflow example."""
    print("\n" + "=" * 70)
    print("Example 7: Complete Multi-Agent Workflow")
    print("=" * 70)

    pool = ContextPool(ttl_seconds=3600)

    # Subscribe agents to relevant topics
    pool.subscribe("bug_analysis", agent_callback("QA Agent"))
    pool.subscribe("code_location", agent_callback("Implementer Agent"))

    print("\n--- Phase 1: Discovery ---")

    # Explorer agent finds issues
    pool.publish(
        topic="bug_analysis",
        content="Authentication fails when username contains special chars",
        source_agent="explorer_agent",
        confidence=0.95,
        metadata={"task_id": "T-100", "priority": "high"}
    )

    # Code analyzer locates relevant code
    pool.publish(
        topic="code_location",
        content="Auth validation in cortical/auth/validators.py:validate_username()",
        source_agent="analyzer_agent",
        confidence=0.9,
        metadata={"task_id": "T-100", "file": "cortical/auth/validators.py", "line": 42}
    )

    print("\n--- Phase 2: Implementation ---")

    # Implementer queries context
    bugs = pool.query("bug_analysis")
    locations = pool.query("code_location")

    print(f"\nImplementer retrieved:")
    print(f"  - {len(bugs)} bug reports")
    print(f"  - {len(locations)} code locations")

    # Implementer publishes fix status
    pool.publish(
        topic="fix_status",
        content="Added regex validation for special characters",
        source_agent="implementer_agent",
        confidence=1.0,
        metadata={"task_id": "T-100", "commit": "abc123"}
    )

    print("\n--- Phase 3: Summary ---")
    print(f"\nFinal pool state:")
    print(f"  Topics: {pool.get_topics()}")
    print(f"  Total findings: {pool.count()}")

    for topic in pool.get_topics():
        findings = pool.query(topic)
        print(f"\n  [{topic}]: {len(findings)} findings")
        for f in findings:
            print(f"    - {f.content[:60]}... (from {f.source_agent})")


def example_got_integration():
    """Example showing integration with GoT (Graph of Thought)."""
    print("\n" + "=" * 70)
    print("Example 8: GoT Integration")
    print("=" * 70)

    pool = ContextPool()

    # Findings linked to GoT tasks
    pool.publish(
        topic="task_progress",
        content="Task T-001: Authentication implementation 60% complete",
        source_agent="agent_a",
        metadata={
            "task_id": "T-001",
            "progress": 0.6,
            "blockers": ["waiting for DB schema approval"]
        }
    )

    pool.publish(
        topic="task_dependencies",
        content="Task T-002 depends on T-001 completion",
        source_agent="agent_b",
        metadata={
            "task_id": "T-002",
            "depends_on": ["T-001"],
            "edge_type": "DEPENDS_ON"
        }
    )

    # Query for GoT construction
    print("\nGoT-relevant findings:")
    for topic in ["task_progress", "task_dependencies"]:
        findings = pool.query(topic)
        for f in findings:
            print(f"\n  [{f.topic}]")
            print(f"    Content: {f.content}")
            print(f"    Metadata: {f.metadata}")


if __name__ == "__main__":
    example_basic_usage()
    example_subscriptions()
    example_conflicts()
    example_conflict_resolution()
    example_ttl_expiration()
    example_persistence()
    example_multi_agent_workflow()
    example_got_integration()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
