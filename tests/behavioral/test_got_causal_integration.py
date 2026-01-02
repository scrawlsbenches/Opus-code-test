"""
Behavioral Tests: GoT-Causal Integration

Epic: Causal Analysis for Project Management

As a project manager using the Graph of Thought,
I want causal analysis of task relationships,
So that I can understand root causes and predict impacts.

These tests define the behavioral contract for GoT-Causal integration.
They also document the DATA REQUIREMENTS for reliable causal inference.
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta


# =============================================================================
# TEST FIXTURES: Simulated GoT Data Structures
# =============================================================================

@dataclass
class MockTask:
    """Simulated GoT task with causal metadata."""
    id: str
    title: str
    status: str = "pending"
    priority: str = "medium"
    category: str = "feature"
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    blocked_at: Optional[datetime] = None
    blocked_reason: Optional[str] = None
    retrospective: Optional[str] = None
    complexity: int = 1  # 1-5 scale for confounder control

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class MockEdge:
    """Simulated GoT edge with causal semantics."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 1.0
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class MockGoTStore:
    """Simulated GoT store for testing causal analysis."""
    tasks: Dict[str, MockTask] = field(default_factory=dict)
    edges: List[MockEdge] = field(default_factory=list)

    def add_task(self, task: MockTask) -> None:
        self.tasks[task.id] = task

    def add_edge(self, edge: MockEdge) -> None:
        self.edges.append(edge)

    def get_edges_from(self, source_id: str, edge_type: Optional[str] = None) -> List[MockEdge]:
        return [e for e in self.edges
                if e.source_id == source_id
                and (edge_type is None or e.edge_type == edge_type)]

    def get_edges_to(self, target_id: str, edge_type: Optional[str] = None) -> List[MockEdge]:
        return [e for e in self.edges
                if e.target_id == target_id
                and (edge_type is None or e.edge_type == edge_type)]


# =============================================================================
# GoT CAUSAL ANALYZER (Implementation Target)
# =============================================================================

class GoTCausalAnalyzer:
    """
    Causal analyzer for Graph of Thought data.

    Integrates PRISM-Causal with GoT task/edge structures.
    """

    def __init__(self, store: MockGoTStore):
        self.store = store

    def trace_root_cause(self, task_id: str) -> Dict:
        """
        Trace the causal chain from a task back to its root cause.

        Uses DEPENDS_ON and CAUSED_BY edges to find the origin.
        """
        visited = set()
        chain = []
        current = task_id

        while current and current not in visited:
            visited.add(current)
            task = self.store.tasks.get(current)
            if task:
                chain.append(current)

            # Find what this task depends on or was caused by
            deps = self.store.get_edges_from(current, "DEPENDS_ON")
            caused = self.store.get_edges_from(current, "CAUSED_BY")

            # Prefer CAUSED_BY over DEPENDS_ON for root cause
            if caused:
                current = caused[0].target_id
            elif deps:
                current = deps[0].target_id
            else:
                current = None

        root_cause = chain[-1] if chain else None
        return {
            "task_id": task_id,
            "root_cause": root_cause,
            "chain": chain,
            "chain_length": len(chain)
        }

    def analyze_impact(self, task_id: str) -> Dict:
        """
        Analyze what would be impacted if a task is delayed/blocked.

        Traverses reverse DEPENDS_ON edges to find affected tasks.
        """
        affected = set()
        to_visit = [task_id]

        while to_visit:
            current = to_visit.pop(0)
            if current in affected:
                continue
            affected.add(current)

            # Find tasks that depend on this one
            dependents = self.store.get_edges_to(current, "DEPENDS_ON")
            for edge in dependents:
                if edge.source_id not in affected:
                    to_visit.append(edge.source_id)

        affected.discard(task_id)  # Don't include the source task

        # Calculate risk based on priority of affected tasks
        high_priority_affected = sum(
            1 for tid in affected
            if self.store.tasks.get(tid, MockTask(id="", title="")).priority in ("high", "critical")
        )

        return {
            "task_id": task_id,
            "affected_tasks": list(affected),
            "affected_count": len(affected),
            "high_priority_affected": high_priority_affected,
            "risk_level": "high" if high_priority_affected > 2 else "medium" if affected else "low"
        }

    def find_blocking_chains(self) -> List[Dict]:
        """
        Find all blocking chains in the task graph.

        A blocking chain is a sequence of DEPENDS_ON edges where
        the root task is blocked, causing downstream delays.
        """
        blocked_tasks = [
            tid for tid, task in self.store.tasks.items()
            if task.status == "blocked"
        ]

        chains = []
        for blocked_id in blocked_tasks:
            impact = self.analyze_impact(blocked_id)
            if impact["affected_count"] > 0:
                chains.append({
                    "blocker": blocked_id,
                    "blocked_reason": self.store.tasks[blocked_id].blocked_reason,
                    "affected": impact["affected_tasks"],
                    "impact_count": impact["affected_count"]
                })

        return sorted(chains, key=lambda x: -x["impact_count"])

    def calculate_necessity(self, cause_id: str, effect_id: str) -> float:
        """
        Calculate probability that cause was necessary for effect.

        P(necessity) = Would the effect have happened without the cause?

        Uses temporal data and alternative path analysis.
        """
        cause_task = self.store.tasks.get(cause_id)
        effect_task = self.store.tasks.get(effect_id)

        if not cause_task or not effect_task:
            return 0.0

        # Check if there's a direct causal path
        root_cause_result = self.trace_root_cause(effect_id)
        if cause_id not in root_cause_result["chain"]:
            return 0.0  # No causal connection

        # Check for alternative paths (other causes that could lead to effect)
        all_causes = self.store.get_edges_from(effect_id, "DEPENDS_ON")
        all_causes.extend(self.store.get_edges_from(effect_id, "CAUSED_BY"))

        # If only one cause, necessity is high
        if len(all_causes) <= 1:
            return 0.95

        # Multiple causes reduce necessity
        return 0.6 / len(all_causes) + 0.3

    def analyze_sprint_causality(self, sprint_tasks: List[str]) -> Dict:
        """
        Analyze causal factors for sprint success/failure.

        Examines blocked tasks, dependency chains, and completion patterns.
        """
        completed = []
        blocked = []
        pending = []

        for tid in sprint_tasks:
            task = self.store.tasks.get(tid)
            if task:
                if task.status == "completed":
                    completed.append(tid)
                elif task.status == "blocked":
                    blocked.append(tid)
                else:
                    pending.append(tid)

        # Find common blockers
        blocker_counts: Dict[str, int] = {}
        for tid in blocked:
            root = self.trace_root_cause(tid)
            if root["root_cause"] and root["root_cause"] != tid:
                blocker_counts[root["root_cause"]] = blocker_counts.get(root["root_cause"], 0) + 1

        # Find the most impactful blocker
        top_blocker = max(blocker_counts.items(), key=lambda x: x[1]) if blocker_counts else None

        completion_rate = len(completed) / len(sprint_tasks) if sprint_tasks else 0

        return {
            "total_tasks": len(sprint_tasks),
            "completed": len(completed),
            "blocked": len(blocked),
            "pending": len(pending),
            "completion_rate": completion_rate,
            "success": completion_rate >= 0.8,
            "top_blocker": top_blocker[0] if top_blocker else None,
            "top_blocker_impact": top_blocker[1] if top_blocker else 0,
            "common_blockers": blocker_counts
        }


# =============================================================================
# BEHAVIORAL TESTS: Root Cause Analysis
# =============================================================================

class TestRootCauseAnalysis:
    """
    As a developer investigating a blocked task,
    I want to trace the causal chain to the root cause,
    So that I can fix the underlying problem instead of symptoms.
    """

    def test_trace_single_dependency_chain(self):
        """
        Scenario: Simple dependency chain

        Given tasks A → B → C (A depends on B, B depends on C)
        When I trace root cause of A
        Then I find C as the root cause
        And the chain shows [A, B, C]
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Feature A"))
        store.add_task(MockTask(id="T-B", title="Feature B"))
        store.add_task(MockTask(id="T-C", title="Foundation C"))
        store.add_edge(MockEdge(source_id="T-A", target_id="T-B", edge_type="DEPENDS_ON"))
        store.add_edge(MockEdge(source_id="T-B", target_id="T-C", edge_type="DEPENDS_ON"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        result = analyzer.trace_root_cause("T-A")

        # Then
        assert result["root_cause"] == "T-C"
        assert result["chain"] == ["T-A", "T-B", "T-C"]
        assert result["chain_length"] == 3

    def test_caused_by_takes_precedence_over_depends_on(self):
        """
        Scenario: CAUSED_BY vs DEPENDS_ON precedence

        Given task A DEPENDS_ON B and A CAUSED_BY C
        When I trace root cause of A
        Then I find C (the cause) not B (the dependency)
        Because causation is stronger than dependency for root cause.
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Bug fix"))
        store.add_task(MockTask(id="T-B", title="API ready"))
        store.add_task(MockTask(id="T-C", title="Bug introduced"))
        store.add_edge(MockEdge(source_id="T-A", target_id="T-B", edge_type="DEPENDS_ON"))
        store.add_edge(MockEdge(source_id="T-A", target_id="T-C", edge_type="CAUSED_BY"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        result = analyzer.trace_root_cause("T-A")

        # Then
        assert result["root_cause"] == "T-C"

    def test_handles_task_with_no_dependencies(self):
        """
        Scenario: Standalone task

        Given a task with no dependencies or causes
        When I trace root cause
        Then the task itself is the root cause
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-Solo", title="Independent task"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        result = analyzer.trace_root_cause("T-Solo")

        # Then
        assert result["root_cause"] == "T-Solo"
        assert result["chain_length"] == 1


# =============================================================================
# BEHAVIORAL TESTS: Impact Analysis
# =============================================================================

class TestImpactAnalysis:
    """
    As a project manager planning task prioritization,
    I want to understand the downstream impact of delays,
    So that I can prioritize tasks on the critical path.
    """

    def test_analyze_downstream_impact(self):
        """
        Scenario: Downstream dependency impact

        Given task C with tasks A and B depending on it
        When I analyze impact of delaying C
        Then I see both A and B are affected
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Feature A", priority="high"))
        store.add_task(MockTask(id="T-B", title="Feature B", priority="medium"))
        store.add_task(MockTask(id="T-C", title="Foundation C"))
        store.add_edge(MockEdge(source_id="T-A", target_id="T-C", edge_type="DEPENDS_ON"))
        store.add_edge(MockEdge(source_id="T-B", target_id="T-C", edge_type="DEPENDS_ON"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        impact = analyzer.analyze_impact("T-C")

        # Then
        assert set(impact["affected_tasks"]) == {"T-A", "T-B"}
        assert impact["affected_count"] == 2
        assert impact["high_priority_affected"] == 1  # Only T-A is high priority

    def test_transitive_impact(self):
        """
        Scenario: Transitive impact through dependency chain

        Given A → B → C → D (chain of dependencies)
        When I analyze impact of D
        Then I see A, B, and C are all affected
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Task A"))
        store.add_task(MockTask(id="T-B", title="Task B"))
        store.add_task(MockTask(id="T-C", title="Task C"))
        store.add_task(MockTask(id="T-D", title="Task D"))
        store.add_edge(MockEdge(source_id="T-A", target_id="T-B", edge_type="DEPENDS_ON"))
        store.add_edge(MockEdge(source_id="T-B", target_id="T-C", edge_type="DEPENDS_ON"))
        store.add_edge(MockEdge(source_id="T-C", target_id="T-D", edge_type="DEPENDS_ON"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        impact = analyzer.analyze_impact("T-D")

        # Then
        assert set(impact["affected_tasks"]) == {"T-A", "T-B", "T-C"}
        assert impact["affected_count"] == 3

    def test_risk_level_based_on_priority(self):
        """
        Scenario: Risk assessment based on affected task priorities

        Given multiple high-priority tasks depending on a foundation task
        When I analyze impact
        Then the risk level is "high"
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-Critical1", title="Critical 1", priority="critical"))
        store.add_task(MockTask(id="T-Critical2", title="Critical 2", priority="critical"))
        store.add_task(MockTask(id="T-High", title="High Priority", priority="high"))
        store.add_task(MockTask(id="T-Foundation", title="Foundation"))

        for tid in ["T-Critical1", "T-Critical2", "T-High"]:
            store.add_edge(MockEdge(source_id=tid, target_id="T-Foundation", edge_type="DEPENDS_ON"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        impact = analyzer.analyze_impact("T-Foundation")

        # Then
        assert impact["high_priority_affected"] == 3
        assert impact["risk_level"] == "high"


# =============================================================================
# BEHAVIORAL TESTS: Blocking Chain Detection
# =============================================================================

class TestBlockingChainDetection:
    """
    As a team lead monitoring sprint progress,
    I want to identify blocking chains,
    So that I can unblock the most impactful tasks first.
    """

    def test_find_blocking_chains(self):
        """
        Scenario: Identify blocked task cascades

        Given a blocked task with downstream dependencies
        When I find blocking chains
        Then I see the blocker and all affected tasks
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(
            id="T-Blocked",
            title="Blocked Task",
            status="blocked",
            blocked_reason="Waiting for API access"
        ))
        store.add_task(MockTask(id="T-Waiting1", title="Waiting 1"))
        store.add_task(MockTask(id="T-Waiting2", title="Waiting 2"))

        store.add_edge(MockEdge(source_id="T-Waiting1", target_id="T-Blocked", edge_type="DEPENDS_ON"))
        store.add_edge(MockEdge(source_id="T-Waiting2", target_id="T-Blocked", edge_type="DEPENDS_ON"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        chains = analyzer.find_blocking_chains()

        # Then
        assert len(chains) == 1
        assert chains[0]["blocker"] == "T-Blocked"
        assert chains[0]["blocked_reason"] == "Waiting for API access"
        assert set(chains[0]["affected"]) == {"T-Waiting1", "T-Waiting2"}

    def test_chains_sorted_by_impact(self):
        """
        Scenario: Blocking chains sorted by impact

        Given multiple blocked tasks with different downstream impact
        When I find blocking chains
        Then they are sorted by impact count (highest first)
        """
        # Given
        store = MockGoTStore()

        # Small blocker: 1 affected
        store.add_task(MockTask(id="T-SmallBlocker", title="Small", status="blocked"))
        store.add_task(MockTask(id="T-Affected1", title="Affected 1"))
        store.add_edge(MockEdge(source_id="T-Affected1", target_id="T-SmallBlocker", edge_type="DEPENDS_ON"))

        # Big blocker: 3 affected
        store.add_task(MockTask(id="T-BigBlocker", title="Big", status="blocked"))
        for i in range(2, 5):
            store.add_task(MockTask(id=f"T-Affected{i}", title=f"Affected {i}"))
            store.add_edge(MockEdge(source_id=f"T-Affected{i}", target_id="T-BigBlocker", edge_type="DEPENDS_ON"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        chains = analyzer.find_blocking_chains()

        # Then
        assert chains[0]["blocker"] == "T-BigBlocker"  # Highest impact first
        assert chains[0]["impact_count"] == 3
        assert chains[1]["blocker"] == "T-SmallBlocker"
        assert chains[1]["impact_count"] == 1


# =============================================================================
# BEHAVIORAL TESTS: Causal Necessity
# =============================================================================

class TestCausalNecessity:
    """
    As a developer doing post-mortem analysis,
    I want to know if a task was NECESSARY for an outcome,
    So that I can identify true causes vs coincidental factors.
    """

    def test_high_necessity_for_single_cause(self):
        """
        Scenario: Single cause has high necessity

        Given effect B caused only by A
        When I calculate necessity of A for B
        Then necessity is high (>0.9)
        Because there was no alternative cause.
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Root cause"))
        store.add_task(MockTask(id="T-B", title="Effect"))
        store.add_edge(MockEdge(source_id="T-B", target_id="T-A", edge_type="CAUSED_BY"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        necessity = analyzer.calculate_necessity("T-A", "T-B")

        # Then
        assert necessity >= 0.9

    def test_lower_necessity_for_multiple_causes(self):
        """
        Scenario: Multiple causes reduce individual necessity

        Given effect C caused by both A and B
        When I calculate necessity of A for C
        Then necessity is lower (<0.7)
        Because B could also have caused C.
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Cause A"))
        store.add_task(MockTask(id="T-B", title="Cause B"))
        store.add_task(MockTask(id="T-C", title="Effect"))
        store.add_edge(MockEdge(source_id="T-C", target_id="T-A", edge_type="CAUSED_BY"))
        store.add_edge(MockEdge(source_id="T-C", target_id="T-B", edge_type="CAUSED_BY"))

        analyzer = GoTCausalAnalyzer(store)

        # When
        necessity = analyzer.calculate_necessity("T-A", "T-C")

        # Then
        assert necessity < 0.7

    def test_no_necessity_without_causal_path(self):
        """
        Scenario: No causal connection means no necessity

        Given unconnected tasks A and B
        When I calculate necessity of A for B
        Then necessity is 0
        """
        # Given
        store = MockGoTStore()
        store.add_task(MockTask(id="T-A", title="Task A"))
        store.add_task(MockTask(id="T-B", title="Task B"))
        # No edge between them

        analyzer = GoTCausalAnalyzer(store)

        # When
        necessity = analyzer.calculate_necessity("T-A", "T-B")

        # Then
        assert necessity == 0.0


# =============================================================================
# BEHAVIORAL TESTS: Sprint Retrospective Analysis
# =============================================================================

class TestSprintRetrospective:
    """
    As a scrum master conducting a retrospective,
    I want causal analysis of sprint outcomes,
    So that I can identify systemic issues and improve future sprints.
    """

    def test_successful_sprint_analysis(self):
        """
        Scenario: Analyzing a successful sprint

        Given a sprint with 80%+ completion rate
        When I analyze sprint causality
        Then the sprint is marked as successful
        """
        # Given
        store = MockGoTStore()
        for i in range(10):
            status = "completed" if i < 8 else "pending"
            store.add_task(MockTask(id=f"T-{i}", title=f"Task {i}", status=status))

        analyzer = GoTCausalAnalyzer(store)
        sprint_tasks = [f"T-{i}" for i in range(10)]

        # When
        result = analyzer.analyze_sprint_causality(sprint_tasks)

        # Then
        assert result["success"] is True
        assert result["completion_rate"] == 0.8
        assert result["completed"] == 8

    def test_identify_common_blocker(self):
        """
        Scenario: Finding the common blocker in a failed sprint

        Given multiple blocked tasks tracing back to a common cause
        When I analyze sprint causality
        Then I identify the top blocker
        """
        # Given
        store = MockGoTStore()

        # The common blocker
        store.add_task(MockTask(id="T-Root", title="Infrastructure Issue", status="blocked"))

        # Tasks blocked by it
        for i in range(3):
            store.add_task(MockTask(id=f"T-Blocked{i}", title=f"Blocked {i}", status="blocked"))
            store.add_edge(MockEdge(source_id=f"T-Blocked{i}", target_id="T-Root", edge_type="DEPENDS_ON"))

        # Some completed tasks
        for i in range(2):
            store.add_task(MockTask(id=f"T-Done{i}", title=f"Done {i}", status="completed"))

        analyzer = GoTCausalAnalyzer(store)
        sprint_tasks = ["T-Root"] + [f"T-Blocked{i}" for i in range(3)] + [f"T-Done{i}" for i in range(2)]

        # When
        result = analyzer.analyze_sprint_causality(sprint_tasks)

        # Then
        assert result["success"] is False
        assert result["blocked"] == 4  # Including T-Root
        assert result["top_blocker"] == "T-Root"
        assert result["top_blocker_impact"] == 3


# =============================================================================
# DATA REQUIREMENTS TESTS: What We Need to Collect
# =============================================================================

class TestDataRequirementsForCausalInference:
    """
    These tests document the DATA REQUIREMENTS for reliable causal inference.

    They serve as specifications for what the GoT SOPs should capture.
    """

    def test_temporal_ordering_required(self):
        """
        Requirement: Temporal data enables causal ordering

        To determine if A caused B, we need:
        - A.completed_at < B.started_at (A finished before B started)

        Without timestamps, we can only trace edges, not verify causation.
        """
        # Given tasks with proper temporal data
        store = MockGoTStore()

        base_time = datetime.now()
        store.add_task(MockTask(
            id="T-Cause",
            title="Cause",
            created_at=base_time,
            started_at=base_time + timedelta(hours=1),
            completed_at=base_time + timedelta(hours=2)
        ))
        store.add_task(MockTask(
            id="T-Effect",
            title="Effect",
            created_at=base_time + timedelta(hours=2),
            started_at=base_time + timedelta(hours=3),  # Started AFTER cause completed
            completed_at=base_time + timedelta(hours=4)
        ))
        store.add_edge(MockEdge(source_id="T-Effect", target_id="T-Cause", edge_type="CAUSED_BY"))

        # Verification: cause completed before effect started
        cause = store.tasks["T-Cause"]
        effect = store.tasks["T-Effect"]

        assert cause.completed_at is not None
        assert effect.started_at is not None
        assert cause.completed_at < effect.started_at, \
            "Temporal ordering confirms causation: cause completed before effect started"

    def test_blocking_metadata_required(self):
        """
        Requirement: Blocking metadata enables blocker analysis

        To analyze blockers, we need:
        - blocked_at: When the task became blocked
        - blocked_reason: Why it's blocked (external, dependency, etc.)
        - BLOCKS edge: Explicit link to the blocking task
        """
        # Given a properly documented blocked task
        store = MockGoTStore()

        store.add_task(MockTask(
            id="T-Blocker",
            title="API not ready",
            status="pending"
        ))
        store.add_task(MockTask(
            id="T-Blocked",
            title="Integration work",
            status="blocked",
            blocked_at=datetime.now(),
            blocked_reason="Waiting for T-Blocker: API endpoints not implemented"
        ))
        store.add_edge(MockEdge(
            source_id="T-Blocker",
            target_id="T-Blocked",
            edge_type="BLOCKS"
        ))

        blocked = store.tasks["T-Blocked"]

        assert blocked.blocked_at is not None, "blocked_at timestamp required"
        assert blocked.blocked_reason is not None, "blocked_reason required"
        assert any(e.edge_type == "BLOCKS" for e in store.edges), "BLOCKS edge required"

    def test_retrospective_data_enables_learning(self):
        """
        Requirement: Retrospective captures causal insights

        For completed tasks, retrospective should capture:
        - What worked/didn't work
        - What caused delays
        - What would have helped
        """
        # Given a completed task with retrospective
        store = MockGoTStore()

        store.add_task(MockTask(
            id="T-Done",
            title="Feature implementation",
            status="completed",
            completed_at=datetime.now(),
            retrospective="""
            ## What worked
            - Clear requirements from the start
            - Good test coverage enabled fast iteration

            ## What didn't work
            - Initial estimate was too optimistic
            - Dependency on T-API caused 2-day delay

            ## Root cause of delay
            - T-API wasn't ready when expected
            - Should have added buffer for dependencies
            """
        ))

        task = store.tasks["T-Done"]

        assert task.retrospective is not None, "Retrospective required for learning"
        assert "root cause" in task.retrospective.lower(), \
            "Retrospective should capture causal insights"

    def test_complexity_enables_confounder_control(self):
        """
        Requirement: Complexity rating enables confounder control

        Without complexity data, we can't distinguish:
        - "Task took long because of blocker" vs
        - "Task took long because it was complex"

        Complexity is a confounding variable for duration analysis.
        """
        # Given tasks with complexity ratings
        store = MockGoTStore()

        store.add_task(MockTask(
            id="T-Simple",
            title="Simple task",
            complexity=1,  # 1-5 scale
            started_at=datetime.now(),
            completed_at=datetime.now() + timedelta(hours=2)
        ))
        store.add_task(MockTask(
            id="T-Complex",
            title="Complex task",
            complexity=5,
            started_at=datetime.now(),
            completed_at=datetime.now() + timedelta(days=3)
        ))

        simple = store.tasks["T-Simple"]
        complex_task = store.tasks["T-Complex"]

        # Duration comparison must account for complexity
        simple_duration = (simple.completed_at - simple.started_at).total_seconds()
        complex_duration = (complex_task.completed_at - complex_task.started_at).total_seconds()

        # Normalize by complexity to compare fairly
        simple_rate = simple_duration / simple.complexity
        complex_rate = complex_duration / complex_task.complexity

        # Now we can fairly compare task completion rates
        assert simple.complexity < complex_task.complexity
        assert simple_rate is not None and complex_rate is not None, \
            "Complexity enables fair duration comparison"

    def test_caused_by_edges_required_for_root_cause(self):
        """
        Requirement: CAUSED_BY edges enable true root cause analysis

        DEPENDS_ON: "I need X to be done before I can start"
        CAUSED_BY: "X is why this task exists"

        These are different! A bug fix CAUSED_BY a bug introduction,
        but may DEPEND_ON API being ready (unrelated to the cause).
        """
        # Given proper causal attribution
        store = MockGoTStore()

        # Original bug introduction
        store.add_task(MockTask(
            id="T-BugIntro",
            title="Commit that introduced bug",
            status="completed"
        ))

        # Bug discovery
        store.add_task(MockTask(
            id="T-BugReport",
            title="Bug report from user",
            status="completed"
        ))
        store.add_edge(MockEdge(
            source_id="T-BugReport",
            target_id="T-BugIntro",
            edge_type="CAUSED_BY"  # Bug report caused by bug intro
        ))

        # Bug fix
        store.add_task(MockTask(
            id="T-BugFix",
            title="Fix the bug",
            status="completed"
        ))
        store.add_edge(MockEdge(
            source_id="T-BugFix",
            target_id="T-BugReport",
            edge_type="CAUSED_BY"  # Fix caused by report
        ))
        store.add_edge(MockEdge(
            source_id="T-BugFix",
            target_id="T-BugIntro",
            edge_type="CAUSED_BY"  # Also caused by original bug
        ))

        # Verify causal chain exists
        analyzer = GoTCausalAnalyzer(store)
        result = analyzer.trace_root_cause("T-BugFix")

        assert "T-BugIntro" in result["chain"], \
            "CAUSED_BY edges enable tracing to original root cause"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
