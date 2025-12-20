"""
Behavioral tests for parallel agent coordination.

These tests verify realistic scenarios for parallel work coordination,
inspired by the complex-reasoning-workflow.md document Part 14
(Collaboration and Coordination).

Test scenarios:
- Sprint-style parallel implementation
- Boundary conflict detection and resolution
- Multi-agent result collection
- Conflict detection at merge time
- Recovery from agent failures
"""

import pytest
from datetime import datetime

from cortical.reasoning.collaboration import (
    AgentResult,
    AgentSpawner,
    AgentStatus,
    ConflictDetail,
    ConflictType,
    ParallelCoordinator,
    ParallelWorkBoundary,
    SequentialSpawner,
    CollaborationManager,
    CollaborationMode,
    BlockerType,
)


class TestSprintStyleParallelWork:
    """
    Test sprint-style parallel implementation scenarios.

    Inspired by: "The 8 sprint agents worked WITHOUT communication
    due to clear boundaries" - proven in practice during Sprints 1-3.
    """

    def test_three_agent_sprint_no_conflicts(self):
        """
        Scenario: Three agents work on separate modules in parallel.
        Expected: All complete successfully with no conflicts.
        """
        # Setup - each agent owns different files
        boundaries = [
            ParallelWorkBoundary(
                agent_id="frontend-agent",
                scope_description="Implement UI components",
                files_owned={"ui/Button.tsx", "ui/Modal.tsx", "ui/Form.tsx"}
            ),
            ParallelWorkBoundary(
                agent_id="backend-agent",
                scope_description="Implement API endpoints",
                files_owned={"api/users.py", "api/auth.py", "api/routes.py"}
            ),
            ParallelWorkBoundary(
                agent_id="test-agent",
                scope_description="Write comprehensive tests",
                files_owned={"tests/test_users.py", "tests/test_auth.py"}
            ),
        ]

        tasks = [
            "Implement Button, Modal, and Form components",
            "Create user and auth API endpoints",
            "Write tests for user and auth APIs"
        ]

        # Custom handler simulates work being done
        work_results = {
            0: {"files_modified": ["ui/Button.tsx", "ui/Modal.tsx", "ui/Form.tsx"]},
            1: {"files_modified": ["api/users.py", "api/auth.py"]},
            2: {"files_modified": ["tests/test_users.py", "tests/test_auth.py"]},
        }
        call_count = [0]

        def sprint_handler(task, boundary):
            idx = call_count[0]
            call_count[0] += 1
            result = work_results.get(idx, {"files_modified": []})
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=result["files_modified"],
                output=f"Completed: {task[:30]}..."
            )

        spawner = SequentialSpawner(handler=sprint_handler)
        coordinator = ParallelCoordinator(spawner)

        # Pre-check - should be able to spawn all
        can_spawn, issues = coordinator.can_spawn(boundaries)
        assert can_spawn, f"Should be able to spawn, but got issues: {issues}"

        # Execute parallel work
        agent_ids = coordinator.spawn_agents(tasks, boundaries)
        results = coordinator.collect_results(agent_ids)
        conflicts = coordinator.detect_conflicts(results)

        # Verify success
        assert len(agent_ids) == 3
        assert all(r.success() for r in results.values())
        assert len(conflicts) == 0

        summary = coordinator.get_summary()
        assert summary['completed_agents'] == 3
        assert summary['failed_agents'] == 0
        assert summary['conflicts_detected'] == 0

    def test_eight_agent_sprint_realistic(self):
        """
        Scenario: Eight agents implementing different stub classes.
        Based on actual Sprint 1-3 implementation pattern.
        """
        # Replicate the 8 parallel agents from Sprints 1-3
        boundaries = [
            # Sprint 1
            ParallelWorkBoundary(
                agent_id="loop-serializer",
                scope_description="Implement LoopStateSerializer",
                files_owned={"cortical/reasoning/cognitive_loop.py"}
            ),
            ParallelWorkBoundary(
                agent_id="question-batcher",
                scope_description="Implement QuestionBatcher",
                files_owned={"cortical/reasoning/collaboration.py"}
            ),
            ParallelWorkBoundary(
                agent_id="production-metrics",
                scope_description="Implement ProductionMetrics",
                files_owned={"cortical/reasoning/production_state.py"}
            ),
            # Sprint 2
            ParallelWorkBoundary(
                agent_id="chunk-planner",
                scope_description="Implement ChunkPlanner",
                files_owned={"cortical/reasoning/production_state.py"}  # Conflict!
            ),
            ParallelWorkBoundary(
                agent_id="comment-cleaner",
                scope_description="Implement CommentCleaner",
                files_owned={"cortical/reasoning/production_state.py"}  # Conflict!
            ),
            ParallelWorkBoundary(
                agent_id="regression-detector",
                scope_description="Implement RegressionDetector",
                files_owned={"cortical/reasoning/verification.py"}
            ),
            # Sprint 3
            ParallelWorkBoundary(
                agent_id="recovery-procedures",
                scope_description="Implement RecoveryProcedures",
                files_owned={"cortical/reasoning/crisis_manager.py"}
            ),
            ParallelWorkBoundary(
                agent_id="failure-analyzer",
                scope_description="Implement FailureAnalyzer",
                files_owned={"cortical/reasoning/verification.py"}  # Conflict!
            ),
        ]

        spawner = SequentialSpawner()
        coordinator = ParallelCoordinator(spawner)

        # Pre-check should detect file ownership conflicts
        can_spawn, issues = coordinator.can_spawn(boundaries)

        # We expect conflicts because multiple agents claim same files
        assert not can_spawn
        assert len(issues) > 0

        # Issues should mention the conflicting files
        issues_text = " ".join(issues)
        assert "production_state.py" in issues_text or "verification.py" in issues_text


class TestBoundaryConflictDetection:
    """Test boundary conflict detection before work begins."""

    def test_detect_ownership_conflict(self):
        """
        Scenario: Two agents claim ownership of same file.
        Expected: can_spawn returns False with clear issue description.
        """
        b1 = ParallelWorkBoundary(
            agent_id="agent-a",
            scope_description="Add feature X",
            files_owned={"config.py", "main.py", "shared.py"}
        )
        b2 = ParallelWorkBoundary(
            agent_id="agent-b",
            scope_description="Add feature Y",
            files_owned={"utils.py", "shared.py", "helpers.py"}
        )

        coordinator = ParallelCoordinator(SequentialSpawner())
        can_spawn, issues = coordinator.can_spawn([b1, b2])

        assert not can_spawn
        assert len(issues) == 1
        assert "shared.py" in issues[0]
        assert "agent-a" in issues[0]
        assert "agent-b" in issues[0]

    def test_detect_read_write_race(self):
        """
        Scenario: One agent reads a file another writes.
        Expected: Warning about potential race condition.
        """
        b1 = ParallelWorkBoundary(
            agent_id="writer",
            scope_description="Update config",
            files_owned={"config.py"}
        )
        b2 = ParallelWorkBoundary(
            agent_id="reader",
            scope_description="Generate docs from config",
            files_owned={"docs/config.md"},
            files_read_only={"config.py"}  # Reads what writer modifies
        )

        coordinator = ParallelCoordinator(SequentialSpawner())
        can_spawn, issues = coordinator.can_spawn([b1, b2])

        assert not can_spawn
        assert any("race" in issue.lower() for issue in issues)
        assert any("config.py" in issue for issue in issues)

    def test_no_conflict_disjoint_boundaries(self):
        """
        Scenario: Boundaries are completely disjoint.
        Expected: can_spawn returns True.
        """
        boundaries = [
            ParallelWorkBoundary(
                agent_id="module-a",
                scope_description="Work on A",
                files_owned={"src/a/file1.py", "src/a/file2.py"}
            ),
            ParallelWorkBoundary(
                agent_id="module-b",
                scope_description="Work on B",
                files_owned={"src/b/file1.py", "src/b/file2.py"}
            ),
            ParallelWorkBoundary(
                agent_id="module-c",
                scope_description="Work on C",
                files_owned={"src/c/file1.py", "src/c/file2.py"}
            ),
        ]

        coordinator = ParallelCoordinator(SequentialSpawner())
        can_spawn, issues = coordinator.can_spawn(boundaries)

        assert can_spawn
        assert issues == []


class TestMergeTimeConflictDetection:
    """Test conflict detection after agents complete."""

    def test_detect_overlapping_modifications(self):
        """
        Scenario: Two agents unexpectedly modify the same file.
        Expected: Conflict detected in results.
        """
        results = {
            "agent-a": AgentResult(
                agent_id="agent-a",
                status=AgentStatus.COMPLETED,
                task_description="Add feature A",
                files_modified=["shared.py", "a.py"],
                output="Done"
            ),
            "agent-b": AgentResult(
                agent_id="agent-b",
                status=AgentStatus.COMPLETED,
                task_description="Add feature B",
                files_modified=["shared.py", "b.py"],  # Overlap!
                output="Done"
            ),
        }

        coordinator = ParallelCoordinator(SequentialSpawner())
        conflicts = coordinator.detect_conflicts(results)

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.FILE_CONFLICT
        assert "shared.py" in conflicts[0].files_affected
        assert "agent-a" in conflicts[0].agents_involved
        assert "agent-b" in conflicts[0].agents_involved

    def test_detect_create_delete_conflict(self):
        """
        Scenario: One agent creates a file another deletes.
        Expected: Conflict detected.
        """
        results = {
            "creator": AgentResult(
                agent_id="creator",
                status=AgentStatus.COMPLETED,
                task_description="Add new module",
                files_created=["new_module.py"],
                output="Created module"
            ),
            "cleaner": AgentResult(
                agent_id="cleaner",
                status=AgentStatus.COMPLETED,
                task_description="Clean up unused files",
                files_deleted=["new_module.py"],  # Deleted what creator made!
                output="Cleaned up"
            ),
        }

        coordinator = ParallelCoordinator(SequentialSpawner())
        conflicts = coordinator.detect_conflicts(results)

        assert len(conflicts) == 1
        assert "new_module.py" in conflicts[0].files_affected

    def test_no_conflict_when_files_disjoint(self):
        """
        Scenario: Agents modify completely different files.
        Expected: No conflicts.
        """
        results = {
            "agent-a": AgentResult(
                agent_id="agent-a",
                status=AgentStatus.COMPLETED,
                task_description="Task A",
                files_modified=["a.py", "b.py"],
            ),
            "agent-b": AgentResult(
                agent_id="agent-b",
                status=AgentStatus.COMPLETED,
                task_description="Task B",
                files_modified=["c.py", "d.py"],
            ),
        }

        coordinator = ParallelCoordinator(SequentialSpawner())
        conflicts = coordinator.detect_conflicts(results)

        assert len(conflicts) == 0


class TestAgentFailureRecovery:
    """Test handling of agent failures."""

    def test_one_agent_fails_others_succeed(self):
        """
        Scenario: One agent in a group fails.
        Expected: Other agents complete, failure is tracked.
        """
        call_count = [0]

        def mixed_handler(task, boundary):
            idx = call_count[0]
            call_count[0] += 1

            if idx == 1:  # Second agent fails
                raise ValueError("Simulated failure")

            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=[f"file{idx}.py"],
            )

        spawner = SequentialSpawner(handler=mixed_handler)
        coordinator = ParallelCoordinator(spawner)

        boundaries = [
            ParallelWorkBoundary("a1", "Task 1", {"file0.py"}),
            ParallelWorkBoundary("a2", "Task 2", {"file1.py"}),
            ParallelWorkBoundary("a3", "Task 3", {"file2.py"}),
        ]

        agent_ids = coordinator.spawn_agents(
            ["Task 1", "Task 2", "Task 3"],
            boundaries
        )
        results = coordinator.collect_results(agent_ids)

        # Check results
        success_count = sum(1 for r in results.values() if r.success())
        fail_count = sum(1 for r in results.values() if not r.success())

        assert success_count == 2
        assert fail_count == 1

        summary = coordinator.get_summary()
        assert summary['completed_agents'] == 2
        assert summary['failed_agents'] == 1

    def test_boundary_violation_detected(self):
        """
        Scenario: Agent modifies file outside its boundary.
        Expected: Violation is recorded in result.
        """
        def violating_handler(task, boundary):
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=["outside_boundary.py"],  # Not in owned files
            )

        spawner = SequentialSpawner(handler=violating_handler)
        coordinator = ParallelCoordinator(spawner)

        boundary = ParallelWorkBoundary(
            agent_id="bounded-agent",
            scope_description="Work only in src/",
            files_owned={"src/main.py", "src/utils.py"}
        )

        agent_ids = coordinator.spawn_agents(["Do work"], [boundary])
        results = coordinator.collect_results(agent_ids)
        conflicts = coordinator.detect_conflicts(results)

        # Boundary violation should be detected
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.SCOPE_OVERLAP
        assert "Boundary violation" in conflicts[0].description


class TestCollaborationIntegration:
    """Test integration with CollaborationManager."""

    def test_coordinator_with_collaboration_manager(self):
        """
        Scenario: Use ParallelCoordinator with CollaborationManager boundaries.
        Expected: Boundaries work with both systems.
        """
        manager = CollaborationManager(mode=CollaborationMode.ASYNCHRONOUS)

        # Create boundaries through manager
        boundary1 = manager.create_boundary(
            agent_id="agent-alpha",
            scope="Implement feature A",
            files={"feature_a.py", "test_a.py"}
        )
        boundary2 = manager.create_boundary(
            agent_id="agent-beta",
            scope="Implement feature B",
            files={"feature_b.py", "test_b.py"}
        )

        # Check for conflicts in manager
        manager_conflicts = manager.check_conflicts()
        assert len(manager_conflicts) == 0

        # Use same boundaries in coordinator
        coordinator = ParallelCoordinator(SequentialSpawner())
        can_spawn, issues = coordinator.can_spawn([boundary1, boundary2])

        assert can_spawn
        assert len(issues) == 0

    def test_blocker_escalation_on_conflict(self):
        """
        Scenario: Coordinator detects conflict, manager escalates.
        Expected: Blocker raised for human resolution.
        """
        manager = CollaborationManager(mode=CollaborationMode.SEMI_SYNCHRONOUS)
        coordinator = ParallelCoordinator(SequentialSpawner())

        # Simulate conflict in results
        results = {
            "a1": AgentResult(
                agent_id="a1",
                status=AgentStatus.COMPLETED,
                task_description="T1",
                files_modified=["conflict.py"]
            ),
            "a2": AgentResult(
                agent_id="a2",
                status=AgentStatus.COMPLETED,
                task_description="T2",
                files_modified=["conflict.py"]
            ),
        }

        conflicts = coordinator.detect_conflicts(results)

        # Escalate to manager
        if conflicts:
            blocker = manager.raise_blocker(
                description=f"Merge conflict in {conflicts[0].files_affected}",
                blocker_type=BlockerType.HARD,
                resolution_needed="Human must resolve file conflicts",
                context={"conflicts": [c.files_affected for c in conflicts]}
            )

            assert blocker.blocker_type == BlockerType.HARD
            assert len(manager.get_hard_blockers()) == 1


class TestRealisticWorkflows:
    """Test complete realistic workflows."""

    def test_feature_branch_parallel_implementation(self):
        """
        Scenario: Feature branch with frontend, backend, and tests.
        Simulates typical feature development workflow.
        """
        # Define work breakdown
        work_items = [
            {
                "agent_id": "frontend",
                "task": "Add user profile page with edit form",
                "files": {"src/pages/Profile.tsx", "src/components/ProfileForm.tsx"},
                "output_files": ["src/pages/Profile.tsx", "src/components/ProfileForm.tsx"],
            },
            {
                "agent_id": "backend",
                "task": "Add profile API endpoints",
                "files": {"src/api/profile.py", "src/models/user.py"},
                "output_files": ["src/api/profile.py"],
            },
            {
                "agent_id": "tests",
                "task": "Write tests for profile feature",
                "files": {"tests/test_profile.py", "tests/test_profile_api.py"},
                "output_files": ["tests/test_profile.py", "tests/test_profile_api.py"],
            },
        ]

        call_count = [0]

        def feature_handler(task, boundary):
            idx = call_count[0]
            call_count[0] += 1
            item = work_items[idx]
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=item["output_files"],
                output=f"Implemented: {item['task']}"
            )

        spawner = SequentialSpawner(handler=feature_handler)
        coordinator = ParallelCoordinator(spawner)

        boundaries = [
            ParallelWorkBoundary(
                agent_id=item["agent_id"],
                scope_description=item["task"],
                files_owned=item["files"]
            )
            for item in work_items
        ]

        tasks = [item["task"] for item in work_items]

        # Pre-check
        can_spawn, issues = coordinator.can_spawn(boundaries)
        assert can_spawn, f"Expected no issues, got: {issues}"

        # Execute
        agent_ids = coordinator.spawn_agents(tasks, boundaries)
        results = coordinator.collect_results(agent_ids)
        conflicts = coordinator.detect_conflicts(results)

        # Verify
        assert len(conflicts) == 0
        assert all(r.success() for r in results.values())

        summary = coordinator.get_summary()
        assert summary['total_files_modified'] == 5
        assert summary['completed_agents'] == 3

    def test_coordinator_reset_for_new_sprint(self):
        """
        Scenario: Complete one sprint, reset, start another.
        Expected: State is cleanly reset between sprints.
        """
        spawner = SequentialSpawner()
        coordinator = ParallelCoordinator(spawner)

        # Sprint 1
        b1 = ParallelWorkBoundary("s1-agent", "Sprint 1 task", {"s1.py"})
        agent_ids_1 = coordinator.spawn_agents(["Sprint 1"], [b1])
        coordinator.collect_results(agent_ids_1)

        summary_1 = coordinator.get_summary()
        assert summary_1['completed_agents'] == 1

        # Reset for Sprint 2
        coordinator.reset()

        summary_reset = coordinator.get_summary()
        assert summary_reset['completed_agents'] == 0
        assert summary_reset['active_agents'] == 0
        assert summary_reset['conflicts_detected'] == 0

        # Sprint 2
        b2 = ParallelWorkBoundary("s2-agent", "Sprint 2 task", {"s2.py"})
        agent_ids_2 = coordinator.spawn_agents(["Sprint 2"], [b2])
        coordinator.collect_results(agent_ids_2)

        summary_2 = coordinator.get_summary()
        assert summary_2['completed_agents'] == 1
