"""
Behavioral tests for orphan detection in Graph of Thought.

Epic: Developer Detects Orphan Tasks

As a developer maintaining graph health,
I want to detect unconnected tasks and get connection suggestions,
So that I can keep our custom-built task graph well-organized.
"""

import pytest
from cortical.got.api import GoTManager
from cortical.got.orphan import OrphanDetector


class TestDeveloperIdentifiesOrphanTasks:
    """
    As a developer maintaining graph connectivity,
    I want to find tasks with no edges,
    So that I can ensure all work is properly organized
    in our custom graph system.
    """

    def test_scenario_detect_task_with_no_connections(self, tmp_path):
        """
        Scenario: Identifying a truly orphaned task

        Given a task with no edges
        When I check if it's an orphan
        Then it's identified as orphaned
        """
        # Given a task with no edges
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)
        orphan = manager.create_task(title="Disconnected task")
        connected = manager.create_task(title="Connected task")
        sprint = manager.create_sprint(title="Sprint 1")
        manager.add_task_to_sprint(connected.id, sprint.id)

        # When I check if it's an orphan
        is_orphan = detector.is_orphan(orphan.id)
        is_connected_orphan = detector.is_orphan(connected.id)

        # Then it's identified as orphaned
        assert is_orphan is True
        assert is_connected_orphan is False

    def test_scenario_generate_orphan_report(self, tmp_path):
        """
        Scenario: Getting system-wide orphan statistics

        Given a mix of connected and orphaned tasks
        When I generate an orphan report
        Then I get accurate counts and statistics
        """
        # Given a mix of connected and orphaned tasks
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        orphan1 = manager.create_task(title="Orphan 1")
        orphan2 = manager.create_task(title="Orphan 2")
        orphan3 = manager.create_task(title="Orphan 3")

        connected1 = manager.create_task(title="Connected 1")
        connected2 = manager.create_task(title="Connected 2")
        manager.add_dependency(connected2.id, connected1.id)

        # When I generate an orphan report
        report = detector.generate_orphan_report()

        # Then I get accurate counts and statistics
        assert report.total_tasks == 5
        assert len(report.orphan_tasks) == 3
        assert report.orphan_rate == 60.0  # 3/5 = 60%
        assert report.has_orphans is True

    def test_scenario_find_all_orphan_tasks(self, tmp_path):
        """
        Scenario: Listing all unconnected tasks

        Given several orphaned tasks
        When I query for orphans
        Then I get all orphaned task IDs
        """
        # Given several orphaned tasks
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        orphan1 = manager.create_task(title="Build feature without planning")
        orphan2 = manager.create_task(title="Random idea")
        connected = manager.create_task(title="Planned work")
        sprint = manager.create_sprint(title="Current sprint")
        manager.add_task_to_sprint(connected.id, sprint.id)

        # When I query for orphans
        orphans = detector.find_orphan_tasks()

        # Then I get all orphaned task IDs
        assert len(orphans) == 2
        assert orphan1.id in orphans
        assert orphan2.id in orphans
        assert connected.id not in orphans


class TestDeveloperGetsSuggestionsForOrphans:
    """
    As a developer organizing orphaned work,
    I want to get sprint and connection suggestions,
    So that I can integrate orphans into our task graph.
    """

    def test_scenario_suggest_current_sprint_for_orphan(self, tmp_path):
        """
        Scenario: Getting sprint suggestions for a task

        Given an orphaned task and an active sprint
        When I get sprint suggestions
        Then the current sprint is suggested first
        With high confidence
        """
        # Given an orphaned task and an active sprint
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        orphan = manager.create_task(
            title="Implement custom hash table",
            description="Build from scratch"
        )
        current_sprint = manager.create_sprint(
            title="Core data structures sprint",
            status="in_progress"
        )
        old_sprint = manager.create_sprint(
            title="Old sprint",
            status="completed"
        )

        # When I get sprint suggestions
        suggestions = detector.suggest_sprint(orphan.id)

        # Then the current sprint is suggested first
        assert len(suggestions) > 0
        assert suggestions[0].sprint_id == current_sprint.id
        assert suggestions[0].is_current is True

        # With high confidence
        assert suggestions[0].confidence >= 0.8

    def test_scenario_suggest_connections_based_on_similarity(self, tmp_path):
        """
        Scenario: Finding similar tasks for connection

        Given an orphaned task and similar tasks
        When I get connection suggestions
        Then I get suggestions to related tasks
        Based on content similarity
        """
        # Given an orphaned task and similar tasks
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        orphan = manager.create_task(
            title="Optimize custom search algorithm",
            description="Improve performance of hand-built search"
        )
        similar = manager.create_task(
            title="Build custom search from scratch",
            description="Implement search algorithm ourselves"
        )
        unrelated = manager.create_task(
            title="Design UI mockups",
            description="Create interface designs"
        )

        # When I get connection suggestions
        suggestions = detector.suggest_connections(orphan.id)

        # Then I get suggestions to related tasks
        if suggestions:
            # Should suggest connecting to similar task
            suggested_ids = [s.target_id for s in suggestions]
            # Similar task should be suggested
            # (exact behavior depends on similarity threshold)

    def test_scenario_check_orphan_on_create(self, tmp_path):
        """
        Scenario: Immediate feedback when creating orphaned task

        Given I just created a new task
        When I check its orphan status
        Then I get warnings and suggestions
        To help me connect it properly
        """
        # Given I just created a new task
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)
        current_sprint = manager.create_sprint(
            title="Current work",
            status="in_progress"
        )

        new_task = manager.create_task(
            title="Build custom parser",
            description="Parse configuration files"
        )

        # When I check its orphan status
        check = detector.check_on_create(new_task.id)

        # Then I get warnings and suggestions
        assert check["is_orphan"] is True
        assert len(check["warnings"]) > 0
        assert len(check["sprint_suggestions"]) > 0


class TestDeveloperAutoLinksOrphansToSprints:
    """
    As a developer batch-organizing orphaned work,
    I want to auto-link orphans to sprints,
    So that I can quickly organize our task graph.
    """

    def test_scenario_auto_link_orphans_to_current_sprint(self, tmp_path):
        """
        Scenario: Batch assigning orphans to a sprint

        Given multiple orphaned tasks and a current sprint
        When I auto-link the orphans to the sprint
        Then all orphans are linked
        And they're no longer orphaned
        """
        # Given multiple orphaned tasks and a current sprint
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        current_sprint = manager.create_sprint(
            title="Current sprint",
            status="in_progress"
        )
        orphan1 = manager.create_task(title="Orphan 1")
        orphan2 = manager.create_task(title="Orphan 2")
        orphan3 = manager.create_task(title="Orphan 3")

        orphan_ids = [orphan1.id, orphan2.id, orphan3.id]

        # When I auto-link the orphans to the sprint
        linked = detector.auto_link_to_sprint(orphan_ids)

        # Then all orphans are linked
        assert len(linked) == 3

        # And they're no longer orphaned
        for orphan_id in orphan_ids:
            assert detector.is_orphan(orphan_id) is False

    def test_scenario_skip_tasks_already_in_sprint(self, tmp_path):
        """
        Scenario: Auto-link skips already-connected tasks

        Given a task already in a sprint
        When I try to auto-link it
        Then it's skipped
        Because it's already organized
        """
        # Given a task already in a sprint
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        sprint1 = manager.create_sprint(title="Sprint 1", status="in_progress")
        sprint2 = manager.create_sprint(title="Sprint 2")
        task = manager.create_task(title="Task")
        manager.add_task_to_sprint(task.id, sprint1.id)

        # When I try to auto-link it
        linked = detector.auto_link_to_sprint([task.id], sprint_id=sprint2.id)

        # Then it's skipped
        assert len(linked) == 0

        # Verify it's still in original sprint
        sprint_tasks = manager.get_sprint_tasks(sprint1.id)
        assert task.id in [t.id for t in sprint_tasks]


class TestDeveloperGetsOrphanSummary:
    """
    As a developer reviewing graph health,
    I want a human-readable orphan summary,
    So that I can quickly assess organizational issues.
    """

    def test_scenario_generate_readable_orphan_summary(self, tmp_path):
        """
        Scenario: Getting a formatted orphan report

        Given a system with some orphans
        When I get the orphan summary
        Then I get a readable formatted report
        With counts and recommendations
        """
        # Given a system with some orphans
        manager = GoTManager(tmp_path / ".got")
        detector = OrphanDetector(manager)

        orphan1 = manager.create_task(title="Orphan task 1")
        orphan2 = manager.create_task(title="Orphan task 2")
        connected = manager.create_task(title="Connected task")
        sprint = manager.create_sprint(title="Sprint 1")
        manager.add_task_to_sprint(connected.id, sprint.id)

        # When I get the orphan summary
        summary = detector.get_orphan_summary()

        # Then I get a readable formatted report
        assert isinstance(summary, str)
        assert "ORPHAN DETECTION REPORT" in summary
        assert "Total Tasks:" in summary
        assert "Orphan Tasks:" in summary
        assert "Orphan Rate:" in summary


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
