#!/usr/bin/env python3
"""
Agent Rejection Protocol Demo

Demonstrates the complete rejection protocol flow:
1. Agent attempts task and discovers blocker/scope creep
2. Agent creates structured rejection with evidence
3. Validation layer checks rejection legitimacy
4. Director decides how to handle rejection
5. Rejection logged to GoT for pattern learning

Run: python examples/rejection_protocol_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.reasoning.rejection_protocol import (
    RejectionReason,
    TaskRejection,
    RejectionValidator,
    DecisionType,
    RejectionDecision,
    log_rejection_to_got,
    analyze_rejection_patterns,
)
from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import NodeType, EdgeType


def demo_valid_scope_creep_rejection():
    """Demonstrate valid SCOPE_CREEP rejection."""
    print("=" * 80)
    print("SCENARIO 1: Valid SCOPE_CREEP Rejection")
    print("=" * 80)
    print()

    # Agent attempts task and discovers scope creep
    task_id = "task:T-20251220-153045-a1b2"
    handoff_id = "handoff:H-20251220-153100-c3d4"
    agent_id = "sub-agent-1"

    print("Task: 'Add OAuth login button to login page'")
    print(f"Agent: {agent_id}")
    print()

    # Agent creates structured rejection
    rejection = TaskRejection(
        task_id=task_id,
        handoff_id=handoff_id,
        agent_id=agent_id,
        reason_type=RejectionReason.SCOPE_CREEP,
        reason_summary="Add OAuth login requires auth system redesign",
        reason_detail="""
Original task: "Add OAuth login button to login page"

Investigation revealed this requires:
1. Refactoring entire auth module for plugin architecture
2. Updating 23 API endpoints to support multiple auth methods
3. Migrating existing user sessions to new auth system
4. Updating 89 test files

Original scope: Add UI button and OAuth flow (~2 hours)
Actual scope: Auth system redesign (~16-20 hours)

Scope growth factor: 8-10x
""",
        what_attempted=[
            "Analyzed auth/ module structure - 47 files, 12,000 LOC",
            "Reviewed existing tests - 89 test files to update",
            "Checked API documentation - 23 endpoints need changes",
            "Estimated effort - 16-20 hours for complete implementation",
        ],
        blocking_factor="Cannot add OAuth without refactoring auth module for extensibility",
        evidence=[
            {
                "type": "file_analysis",
                "data": {"files_requiring_changes": 47, "lines_of_code": 12000},
                "source": "static analysis of auth/ module",
            },
            {
                "type": "complexity_metric",
                "data": {"test_files_to_update": 89, "api_endpoints_affected": 23},
                "source": "impact analysis",
            },
            {
                "type": "time_estimate",
                "data": {"original_estimate_hours": 2, "revised_estimate_hours": 18},
                "source": "engineering judgment",
            },
        ],
        suggested_alternative="Decompose into 3 sequential tasks: refactor auth, add OAuth plugin, add UI",
        alternative_tasks=[
            {
                "title": "Refactor auth module for plugin architecture",
                "scope": "Extract auth provider interface, migrate existing auth",
                "dependencies": "none",
                "estimate": "8 hours",
            },
            {
                "title": "Implement OAuth provider plugin",
                "scope": "Add OAuth using new plugin interface",
                "dependencies": "Requires auth refactor task",
                "estimate": "4 hours",
            },
            {
                "title": "Add OAuth UI and user flows",
                "scope": "Add login button, redirect flows, documentation",
                "dependencies": "Requires OAuth plugin task",
                "estimate": "2 hours",
            },
        ],
        task_original_scope="Add OAuth login button to login page (~2 hours)",
        scope_growth_factor=9.0,
    )

    print("Agent's rejection:")
    print(f"  Reason: {rejection.reason_type.name}")
    print(f"  Summary: {rejection.reason_summary}")
    print(f"  Scope growth: {rejection.scope_growth_factor}x")
    print(f"  Attempts made: {len(rejection.what_attempted)}")
    print(f"  Evidence pieces: {len(rejection.evidence)}")
    print(f"  Alternative tasks: {len(rejection.alternative_tasks)}")
    print()

    # Validate rejection
    validator = RejectionValidator()
    task_context = {
        "title": "Add OAuth login",
        "scope": "Add OAuth login button to login page",
        "priority": "medium",
        "category": "feature",
    }

    is_valid, issues = validator.validate(rejection, task_context)

    print(f"Validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No issues found - rejection is legitimate")
    print()

    # Director creates decision
    decision = RejectionDecision(
        decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
        rejection=rejection,
        rationale="Valid scope creep. Decomposed into 3 sub-tasks.",
        created_tasks=[
            "task:T-20251220-160001-e5f6",
            "task:T-20251220-160002-g7h8",
            "task:T-20251220-160003-i9j0",
        ],
    )

    print(f"Director decision: {decision.decision_type.name}")
    print(f"Rationale: {decision.rationale}")
    print(f"Created tasks: {len(decision.created_tasks)}")
    for i, created_task_id in enumerate(decision.created_tasks, 1):
        print(f"  {i}. {created_task_id}")
    print()

    # Log to GoT
    graph = ThoughtGraph()

    # Create task node first (would normally exist from task creation)
    graph.add_node(
        task_id,
        NodeType.TASK,
        "Add OAuth login button to login page",
        properties=task_context,
    )

    # Create sub-task nodes
    for i, sub_task_id in enumerate(decision.created_tasks):
        graph.add_node(
            sub_task_id,
            NodeType.TASK,
            rejection.alternative_tasks[i]["title"],
            properties={"status": "pending"},
        )

    rejection_node_id = log_rejection_to_got(graph, rejection, decision)

    print(f"Logged to GoT: {rejection_node_id}")
    print(f"Graph size: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print()


def demo_lazy_rejection_override():
    """Demonstrate lazy rejection that gets overridden."""
    print("=" * 80)
    print("SCENARIO 2: Lazy Rejection (Gets Overridden)")
    print("=" * 80)
    print()

    task_id = "task:T-20251220-160000-x1y2"
    handoff_id = "handoff:H-20251220-160010-z3w4"
    agent_id = "sub-agent-2"

    print("Task: 'Implement user authentication'")
    print(f"Agent: {agent_id}")
    print()

    # Agent creates lazy rejection
    lazy_rejection = TaskRejection(
        task_id=task_id,
        handoff_id=handoff_id,
        agent_id=agent_id,
        reason_type=RejectionReason.BLOCKER,
        reason_summary="Too complex",
        reason_detail="This task is too complex for me to handle.",
        what_attempted=[
            "Looked at the code",
            "Tried to understand it",
        ],
        blocking_factor="It's confusing",
        evidence=[],
        suggested_alternative="Ask someone else to do it",
    )

    print("Agent's rejection:")
    print(f"  Reason: {lazy_rejection.reason_type.name}")
    print(f"  Summary: {lazy_rejection.reason_summary}")
    print(f"  Attempts: {lazy_rejection.what_attempted}")
    print(f"  Blocking factor: {lazy_rejection.blocking_factor}")
    print(f"  Alternative: {lazy_rejection.suggested_alternative}")
    print()

    # Validate rejection
    validator = RejectionValidator()
    task_context = {
        "title": "Implement user authentication",
        "scope": "Add JWT-based authentication to API",
        "priority": "high",
        "category": "feature",
    }

    is_valid, issues = validator.validate(lazy_rejection, task_context)

    print(f"Validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  ❌ {issue}")
    print()

    # Director overrides
    decision = RejectionDecision(
        decision_type=DecisionType.OVERRIDE,
        rejection=lazy_rejection,
        rationale=f"Rejection validation failed: {len(issues)} issues",
        override_message=f"""
Your rejection was not accepted due to the following issues:

{chr(10).join(f'- {issue}' for issue in issues)}

Please retry the task and ensure you:
1. Document at least 2 concrete attempts
2. Provide specific, measurable blocking factors
3. Suggest actionable alternatives
4. Include evidence supporting your rejection

Remember: Rejection must be earned through demonstrated effort.
""",
        reassign_to=agent_id,
    )

    print(f"Director decision: {decision.decision_type.name}")
    print(f"Rationale: {decision.rationale}")
    print("Override message sent to agent ✉️")
    print()


def demo_blocker_rejection():
    """Demonstrate valid BLOCKER rejection."""
    print("=" * 80)
    print("SCENARIO 3: Valid BLOCKER Rejection")
    print("=" * 80)
    print()

    task_id = "task:T-20251220-161000-m1n2"
    handoff_id = "handoff:H-20251220-161010-p3q4"
    agent_id = "sub-agent-3"

    print("Task: 'Deploy to production environment'")
    print(f"Agent: {agent_id}")
    print()

    # Agent discovers external blocker
    rejection = TaskRejection(
        task_id=task_id,
        handoff_id=handoff_id,
        agent_id=agent_id,
        reason_type=RejectionReason.BLOCKER,
        reason_summary="Production API credentials not available",
        reason_detail="""
Cannot deploy to production without valid API credentials.

Attempted:
1. Checked local environment variables - not set
2. Checked deployment config files - placeholder values only
3. Attempted to access credentials manager - access denied (403)
4. Contacted ops team - credentials not provisioned yet

This is an external blocker requiring ops team action.
""",
        what_attempted=[
            "Verified local environment variables - PROD_API_KEY not set",
            "Checked config/production.yml - contains placeholder '${PROD_API_KEY}'",
            "Attempted API call with test credentials - received 401 Unauthorized",
            "Contacted ops team via Slack - ticket opened, ETA 2 business days",
        ],
        blocking_factor="Production API credentials not provisioned by ops team",
        evidence=[
            {
                "type": "error_log",
                "data": "401 Unauthorized: Invalid API key",
                "source": "production deployment attempt",
            },
            {
                "type": "status_check",
                "data": {"credentials_provisioned": False, "ops_ticket": "OPS-1234"},
                "source": "ops team status",
            },
        ],
        suggested_alternative="Defer deployment until ops team provisions credentials (OPS-1234)",
    )

    print("Agent's rejection:")
    print(f"  Reason: {rejection.reason_type.name}")
    print(f"  Summary: {rejection.reason_summary}")
    print(f"  Attempts made: {len(rejection.what_attempted)}")
    print(f"  Evidence pieces: {len(rejection.evidence)}")
    print()

    # Validate rejection
    validator = RejectionValidator()
    task_context = {
        "title": "Deploy to production",
        "scope": "Deploy latest release to production environment",
        "priority": "high",
        "category": "deployment",
    }

    is_valid, issues = validator.validate(rejection, task_context)

    print(f"Validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No issues found - valid external blocker")
    print()

    # Director accepts and defers
    decision = RejectionDecision(
        decision_type=DecisionType.ACCEPT_AND_DEFER,
        rejection=rejection,
        rationale="Valid blocker. Created blocker resolution task.",
        created_tasks=["task:T-20251220-161500-r5s6"],  # Ops ticket resolution task
        deferred_task=task_id,
    )

    print(f"Director decision: {decision.decision_type.name}")
    print(f"Rationale: {decision.rationale}")
    print(f"Created blocker resolution task: {decision.created_tasks[0]}")
    print(f"Deferred original task until blocker resolved")
    print()


def demo_pattern_analysis():
    """Demonstrate rejection pattern analysis."""
    print("=" * 80)
    print("SCENARIO 4: Rejection Pattern Analysis")
    print("=" * 80)
    print()

    # Create graph with multiple rejections
    graph = ThoughtGraph()

    # Add some rejections from previous scenarios
    from datetime import datetime, timedelta

    base_time = datetime.now()
    rejections = [
        TaskRejection(
            task_id=f"task:T-example-{i}",
            handoff_id=f"handoff:H-example-{i}",
            agent_id=f"agent-{i % 3}",
            reason_type=RejectionReason.SCOPE_CREEP if i % 2 == 0 else RejectionReason.BLOCKER,
            reason_summary=f"Rejection {i}",
            reason_detail="Details...",
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="Some blocker",
            suggested_alternative="Some alternative",
            rejected_at=base_time + timedelta(seconds=i),  # Unique timestamps
        )
        for i in range(10)
    ]

    decisions = [
        RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DECOMPOSE if i % 2 == 0 else DecisionType.ACCEPT_AND_DEFER,
            rejection=rej,
            rationale=f"Decision for rejection {i}",
        )
        for i, rej in enumerate(rejections)
    ]

    # Create task nodes first
    for rej in rejections:
        graph.add_node(
            rej.task_id,
            NodeType.TASK,
            f"Task {rej.task_id}",
            properties={"status": "pending"},
        )

    # Log all to graph
    for rejection, decision in zip(rejections, decisions):
        log_rejection_to_got(graph, rejection, decision)

    # Analyze patterns
    patterns = analyze_rejection_patterns(graph)

    print("Pattern Analysis:")
    print(f"  Total rejections: {patterns['total_rejections']}")
    print()

    print("  By reason:")
    for reason, count in sorted(patterns['by_reason'].items()):
        print(f"    {reason}: {count}")
    print()

    print("  By agent:")
    for agent, count in sorted(patterns['by_agent'].items()):
        print(f"    {agent}: {count}")
    print()

    print("  Decision outcomes:")
    for decision, count in sorted(patterns['decision_outcomes'].items()):
        print(f"    {decision}: {count}")
    print()

    print(f"  Override rate: {patterns['override_rate']:.1%}")
    print()


def main():
    """Run all demo scenarios."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AGENT REJECTION PROTOCOL DEMO" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    demo_valid_scope_creep_rejection()
    demo_lazy_rejection_override()
    demo_blocker_rejection()
    demo_pattern_analysis()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  ✅ Valid rejections are accepted and handled productively")
    print("  ❌ Lazy rejections are caught and overridden")
    print("  📊 Patterns are logged to GoT for learning")
    print("  🔄 Director can decompose, defer, or reformulate tasks")
    print()


if __name__ == "__main__":
    main()
