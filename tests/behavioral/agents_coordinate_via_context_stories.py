"""
Behavioral tests for multi-agent coordination via ContextPool.

As a developer building multi-agent systems,
I want agents to share findings through a context pool,
So that agents coordinate without tight coupling.

Based on: examples/context_pool_demo.py
"""

import pytest
import time
from pathlib import Path
from cortical.reasoning.context_pool import (
    ContextPool,
    ContextFinding,
    ConflictResolutionStrategy,
)


class TestAgentsShareFindings:
    """
    Epic: Agent Coordination

    As a multi-agent system developer,
    I want agents to publish and query shared findings,
    So that knowledge flows between agents.
    """

    def test_agent_publishes_finding_for_others(self):
        """
        Scenario: Agent publishes finding that others can retrieve

        Given a context pool shared by multiple agents
        When one agent publishes a finding
        Then other agents can query and retrieve that finding
        Because the pool enables knowledge sharing
        """
        # Given: a context pool
        pool = ContextPool(ttl_seconds=3600)

        # When: agent A publishes a finding
        pool.publish(
            topic="code_location",
            content="Authentication logic is in cortical/auth.py",
            source_agent="agent_a",
            confidence=0.95,
            metadata={"file": "cortical/auth.py"}
        )

        # Then: agent B can query and retrieve it
        findings = pool.query("code_location")
        assert len(findings) == 1, "Should retrieve one finding"
        assert findings[0].content == "Authentication logic is in cortical/auth.py"
        assert findings[0].source_agent == "agent_a"
        assert findings[0].confidence == 0.95

    def test_pool_tracks_multiple_topics(self):
        """
        Scenario: Pool organizes findings by topic

        Given a pool with findings on different topics
        When querying for a specific topic
        Then only findings for that topic are returned
        Because topic-based organization enables focused queries
        """
        # Given: a pool with multiple topics
        pool = ContextPool()
        pool.publish(topic="bugs", content="Bug in login.py", source_agent="agent_a")
        pool.publish(topic="bugs", content="Bug in auth.py", source_agent="agent_b")
        pool.publish(topic="features", content="New API endpoint", source_agent="agent_c")

        # When: querying for specific topic
        bug_findings = pool.query("bugs")
        feature_findings = pool.query("features")

        # Then: only relevant findings returned
        assert len(bug_findings) == 2, "Should find both bug findings"
        assert len(feature_findings) == 1, "Should find feature finding"
        assert all(f.topic == "bugs" for f in bug_findings), "All should be bug findings"


class TestAgentsSubscribeToFindings:
    """
    Epic: Real-Time Notifications

    As a developer building reactive agents,
    I want agents to subscribe to topics of interest,
    So that they receive notifications when relevant findings appear.
    """

    def test_subscriber_receives_notifications(self):
        """
        Scenario: Subscribed agent receives notification on publish

        Given an agent subscribed to a topic
        When another agent publishes to that topic
        Then the subscriber receives a notification
        Because subscriptions enable push-based coordination
        """
        # Given: pool with subscription
        pool = ContextPool()
        received_findings = []

        def callback(finding: ContextFinding):
            received_findings.append(finding)

        pool.subscribe("security", callback)

        # When: publishing to that topic
        pool.publish(
            topic="security",
            content="Found SQL injection vulnerability",
            source_agent="scanner_agent",
            confidence=0.9
        )

        # Then: subscriber receives notification
        assert len(received_findings) == 1, "Should receive one notification"
        assert received_findings[0].content == "Found SQL injection vulnerability"

    def test_multiple_subscribers_all_notified(self):
        """
        Scenario: All subscribers to a topic receive notifications

        Given multiple agents subscribed to the same topic
        When a finding is published to that topic
        Then all subscribers receive the notification
        Because all interested agents should be informed
        """
        # Given: multiple subscribers
        pool = ContextPool()
        agent_b_findings = []
        agent_c_findings = []

        pool.subscribe("critical_bugs", lambda f: agent_b_findings.append(f))
        pool.subscribe("critical_bugs", lambda f: agent_c_findings.append(f))

        # When: publishing to that topic
        pool.publish(
            topic="critical_bugs",
            content="Crash on startup",
            source_agent="agent_a",
            confidence=1.0
        )

        # Then: all subscribers notified
        assert len(agent_b_findings) == 1, "Agent B should receive notification"
        assert len(agent_c_findings) == 1, "Agent C should receive notification"


class TestSystemDetectsConflicts:
    """
    Epic: Conflict Detection

    As a system coordinator,
    I want to detect when agents have conflicting findings,
    So that conflicts can be resolved.
    """

    def test_system_detects_conflicting_findings(self):
        """
        Scenario: Conflicting findings are detected automatically

        Given two agents with different findings on the same topic
        When both publish their findings
        Then the system detects a conflict
        Because differing information needs resolution
        """
        # Given: pool with manual conflict resolution
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

        # When: agents publish conflicting findings
        pool.publish(
            topic="performance",
            content="Query takes 500ms",
            source_agent="agent_a",
            confidence=0.8
        )
        pool.publish(
            topic="performance",
            content="Query takes 300ms",
            source_agent="agent_b",
            confidence=0.7
        )

        # Then: conflict is detected
        conflicts = pool.get_conflicts()
        assert len(conflicts) > 0, "Should detect conflicting findings"

    def test_highest_confidence_wins_automatically(self):
        """
        Scenario: Automatic conflict resolution keeps highest confidence

        Given a pool using highest-confidence resolution
        When conflicting findings are published
        Then only the highest-confidence finding remains
        Because confidence indicates reliability
        """
        # Given: pool with highest-confidence resolution
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)

        # When: publishing conflicting findings with different confidence
        pool.publish(
            topic="database",
            content="Database is MySQL",
            source_agent="agent_a",
            confidence=0.6
        )
        pool.publish(
            topic="database",
            content="Database is PostgreSQL",
            source_agent="agent_b",
            confidence=0.9  # Higher confidence
        )

        # Then: only highest-confidence finding remains
        findings = pool.query("database")
        assert len(findings) == 1, "Should keep only one finding"
        assert findings[0].content == "Database is PostgreSQL", "Should keep higher-confidence finding"
        assert findings[0].confidence == 0.9


class TestFindingsExpireAutomatically:
    """
    Epic: Temporal Relevance

    As a system designer,
    I want old findings to expire automatically,
    So that the pool doesn't accumulate stale information.
    """

    def test_findings_expire_after_ttl(self):
        """
        Scenario: Findings expire after time-to-live

        Given a pool with short TTL
        When a finding is published and time passes
        Then the finding expires and is no longer retrievable
        Because stale information should not persist
        """
        # Given: pool with 50ms TTL (minimum viable for testing)
        pool = ContextPool(ttl_seconds=0.05)

        # When: publishing and waiting
        pool.publish(
            topic="temp",
            content="Temporary finding",
            source_agent="agent_a"
        )
        assert pool.count() == 1, "Should have one finding initially"

        # Wait for expiration (100ms > 50ms TTL)
        time.sleep(0.1)

        # Then: finding expires
        assert pool.count() == 0, "Finding should expire after TTL"


class TestSystemPersistsState:
    """
    Epic: State Persistence

    As a system operator,
    I want to save and restore the context pool,
    So that findings survive system restarts.
    """

    def test_pool_state_saves_and_loads(self):
        """
        Scenario: Pool state can be persisted and restored

        Given a pool with findings
        When saving to disk and loading into new pool
        Then all findings are restored
        Because persistence enables system continuity
        """
        # Given: pool with findings
        storage_dir = Path("/tmp/context_pool_test")
        storage_dir.mkdir(exist_ok=True)

        original_pool = ContextPool(storage_dir=storage_dir)
        original_pool.publish(
            topic="architecture",
            content="System uses layered architecture",
            source_agent="architect_agent",
            confidence=1.0
        )
        original_pool.publish(
            topic="architecture",
            content="Custom implementation from scratch",
            source_agent="architect_agent",
            confidence=0.95
        )

        # When: saving and loading
        original_pool.save()
        new_pool = ContextPool(storage_dir=storage_dir)
        new_pool.load()

        # Then: findings are restored
        assert new_pool.count() == 2, "Should restore both findings"
        findings = new_pool.query("architecture")
        assert len(findings) == 2, "Should restore all architecture findings"

        # Cleanup
        import shutil
        shutil.rmtree(storage_dir, ignore_errors=True)


class TestAgentsCoordinateComplexWorkflow:
    """
    Epic: Multi-Phase Workflows

    As a workflow designer,
    I want agents to coordinate through discovery, implementation, and verification,
    So that complex tasks get completed systematically.
    """

    def test_agents_complete_bug_fix_workflow(self):
        """
        Scenario: Agents coordinate from bug discovery to fix

        Given multiple specialized agents
        When they work through discovery, implementation, and verification
        Then findings flow between phases
        Because agents build on each other's work
        """
        # Given: shared context pool
        pool = ContextPool(ttl_seconds=3600)

        # Phase 1: Discovery - Explorer finds bug
        pool.publish(
            topic="bug_analysis",
            content="Authentication fails with special characters",
            source_agent="explorer_agent",
            confidence=0.95,
            metadata={"priority": "high", "task_id": "T-100"}
        )

        # Phase 1: Analysis - Analyzer locates code
        pool.publish(
            topic="code_location",
            content="Auth validation in cortical/auth/validators.py:validate_username()",
            source_agent="analyzer_agent",
            confidence=0.9,
            metadata={"task_id": "T-100", "file": "cortical/auth/validators.py"}
        )

        # Phase 2: Implementation - Developer queries context
        bugs = pool.query("bug_analysis")
        locations = pool.query("code_location")
        assert len(bugs) == 1, "Should find bug report"
        assert len(locations) == 1, "Should find code location"

        # Phase 2: Implementation - Developer publishes fix
        pool.publish(
            topic="fix_status",
            content="Added regex validation for special characters",
            source_agent="developer_agent",
            confidence=1.0,
            metadata={"task_id": "T-100", "commit": "abc123"}
        )

        # Phase 3: Verification - Query full workflow
        all_topics = pool.get_topics()
        assert "bug_analysis" in all_topics, "Should have bug analysis"
        assert "code_location" in all_topics, "Should have code location"
        assert "fix_status" in all_topics, "Should have fix status"
        assert pool.count() == 3, "Should have findings from all phases"

    def test_findings_link_to_task_graph(self):
        """
        Scenario: Findings carry metadata for task graph integration

        Given findings with task metadata
        When querying findings
        Then metadata enables task graph construction
        Because findings and tasks are related
        """
        # Given: findings with task metadata
        pool = ContextPool()

        pool.publish(
            topic="task_progress",
            content="Authentication implementation 60% complete",
            source_agent="agent_a",
            metadata={
                "task_id": "T-001",
                "progress": 0.6,
                "blockers": ["waiting for schema approval"]
            }
        )

        pool.publish(
            topic="task_dependencies",
            content="Task T-002 depends on T-001",
            source_agent="agent_b",
            metadata={
                "task_id": "T-002",
                "depends_on": ["T-001"],
                "edge_type": "DEPENDS_ON"
            }
        )

        # When: querying findings
        progress_findings = pool.query("task_progress")
        dependency_findings = pool.query("task_dependencies")

        # Then: metadata enables graph construction
        assert progress_findings[0].metadata["task_id"] == "T-001"
        assert progress_findings[0].metadata["progress"] == 0.6
        assert dependency_findings[0].metadata["depends_on"] == ["T-001"]


class TestSystemProvidesObservability:
    """
    Epic: System Monitoring

    As a system operator,
    I want to monitor pool statistics,
    So that I can understand system behavior.
    """

    def test_pool_reports_statistics(self):
        """
        Scenario: Pool provides count and topic information

        Given a pool with multiple findings
        When requesting statistics
        Then pool reports counts and topics
        Because observability is essential
        """
        # Given: pool with findings
        pool = ContextPool()
        pool.publish(topic="topic_a", content="Finding 1", source_agent="agent_1")
        pool.publish(topic="topic_a", content="Finding 2", source_agent="agent_1")
        pool.publish(topic="topic_b", content="Finding 3", source_agent="agent_2")

        # When: requesting statistics
        total = pool.count()
        topic_a_count = pool.count("topic_a")
        topics = pool.get_topics()

        # Then: accurate statistics
        assert total == 3, "Should report total count"
        assert topic_a_count == 2, "Should count findings by topic"
        assert set(topics) == {"topic_a", "topic_b"}, "Should list all topics"

    def test_pool_supports_querying_all_findings(self):
        """
        Scenario: Can retrieve all findings regardless of topic

        Given a pool with findings on multiple topics
        When querying all findings
        Then all findings are returned
        Because sometimes we need the complete view
        """
        # Given: pool with multiple topics
        pool = ContextPool()
        pool.publish(topic="bugs", content="Bug 1", source_agent="agent_a")
        pool.publish(topic="features", content="Feature 1", source_agent="agent_b")
        pool.publish(topic="performance", content="Perf issue", source_agent="agent_c")

        # When: querying all
        all_findings = pool.query_all()

        # Then: all findings returned
        assert len(all_findings) == 3, "Should return all findings"
        topics = {f.topic for f in all_findings}
        assert topics == {"bugs", "features", "performance"}, "Should include all topics"
