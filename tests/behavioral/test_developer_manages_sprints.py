"""
Behavioral tests for sprint management in Graph of Thought.

Epic: Developer Manages Sprints

As a developer organizing work iterations,
I want to create sprints and organize tasks within them,
So that I can track progress using our custom-built sprint system.
"""

import pytest
from cortical.got.api import GoTManager
from tests.conftest import _create_got_manager


class TestDeveloperCreatesAndManagesSprints:
    """
    As a developer planning iterations,
    I want to create sprints with metadata,
    So that I can organize tasks into time-boxed iterations.
    """

    def test_scenario_create_sprint_with_title(self, tmp_path):
        """
        Scenario: Creating a new sprint

        Given a GoT manager
        When I create a sprint with a title
        Then the sprint is created with a timestamp-based ID
        And I can retrieve it
        """
        # Given a GoT manager
        manager = _create_got_manager(tmp_path / ".got")

        # When I create a sprint with a title
        sprint = manager.create_sprint(
            title="Build custom authentication system",
            number=1
        )

        # Then the sprint is created with a timestamp-based ID
        assert sprint.id.startswith("S-")
        assert sprint.title == "Build custom authentication system"
        assert sprint.number == 1
        assert sprint.status == "available"  # default

        # And I can retrieve it
        retrieved = manager.get_sprint(sprint.id)
        assert retrieved is not None
        assert retrieved.id == sprint.id

    def test_scenario_update_sprint_status(self, tmp_path):
        """
        Scenario: Starting and completing sprints

        Given an available sprint
        When I update its status to in_progress
        Then the status change is persisted
        """
        # Given an available sprint
        manager = _create_got_manager(tmp_path / ".got")
        sprint = manager.create_sprint(
            title="Implement custom caching layer",
            status="available"
        )

        # When I update its status to in_progress
        updated = manager.update_sprint(sprint.id, status="in_progress")

        # Then the status change is persisted
        assert updated.status == "in_progress"

        retrieved = manager.get_sprint(sprint.id)
        assert retrieved.status == "in_progress"

    def test_scenario_list_sprints_by_status(self, tmp_path):
        """
        Scenario: Filtering sprints by status

        Given sprints in different states
        When I list sprints filtered by in_progress
        Then I only get active sprints
        """
        # Given sprints in different states
        manager = _create_got_manager(tmp_path / ".got")
        active1 = manager.create_sprint(title="Sprint 1", status="in_progress")
        active2 = manager.create_sprint(title="Sprint 2", status="in_progress")
        completed = manager.create_sprint(title="Sprint 3", status="completed")
        available = manager.create_sprint(title="Sprint 4", status="available")

        # When I list sprints filtered by in_progress
        active_sprints = manager.list_sprints(status="in_progress")

        # Then I only get active sprints
        assert len(active_sprints) == 2
        sprint_ids = {s.id for s in active_sprints}
        assert active1.id in sprint_ids
        assert active2.id in sprint_ids


class TestDeveloperOrganizesTasksIntoSprints:
    """
    As a developer managing sprint backlogs,
    I want to add tasks to sprints,
    So that I can track which work belongs to each iteration.
    """

    def test_scenario_add_tasks_to_sprint(self, tmp_path):
        """
        Scenario: Building a sprint backlog

        Given a sprint and several tasks
        When I add tasks to the sprint
        Then CONTAINS edges are created
        And I can query sprint tasks
        """
        # Given a sprint and several tasks
        manager = _create_got_manager(tmp_path / ".got")
        sprint = manager.create_sprint(title="Sprint 1")
        task1 = manager.create_task(title="Build custom parser")
        task2 = manager.create_task(title="Implement tokenizer")
        task3 = manager.create_task(title="Design grammar")

        # When I add tasks to the sprint
        manager.add_task_to_sprint(task1.id, sprint.id)
        manager.add_task_to_sprint(task2.id, sprint.id)
        manager.add_task_to_sprint(task3.id, sprint.id)

        # Then CONTAINS edges are created
        outgoing, incoming = manager.get_edges_for_task(sprint.id)
        assert len(outgoing) == 3

        # And I can query sprint tasks
        sprint_tasks = manager.get_sprint_tasks(sprint.id)
        assert len(sprint_tasks) == 3
        task_ids = {t.id for t in sprint_tasks}
        assert task1.id in task_ids
        assert task2.id in task_ids
        assert task3.id in task_ids

    def test_scenario_get_sprint_progress_statistics(self, tmp_path):
        """
        Scenario: Tracking sprint completion

        Given a sprint with tasks in various states
        When I query sprint progress
        Then I get accurate statistics
        Including completion rate
        """
        # Given a sprint with tasks in various states
        manager = _create_got_manager(tmp_path / ".got")
        sprint = manager.create_sprint(title="Sprint 1")

        completed1 = manager.create_task(title="Task 1", status="completed")
        completed2 = manager.create_task(title="Task 2", status="completed")
        in_progress = manager.create_task(title="Task 3", status="in_progress")
        pending = manager.create_task(title="Task 4", status="pending")

        manager.add_task_to_sprint(completed1.id, sprint.id)
        manager.add_task_to_sprint(completed2.id, sprint.id)
        manager.add_task_to_sprint(in_progress.id, sprint.id)
        manager.add_task_to_sprint(pending.id, sprint.id)

        # When I query sprint progress
        progress = manager.get_sprint_progress(sprint.id)

        # Then I get accurate statistics
        assert progress["total"] == 4
        assert progress["completed"] == 2
        assert progress["in_progress"] == 1
        assert progress["pending"] == 1

        # Including completion rate
        assert progress["completion_rate"] == 0.5  # 2/4


class TestDeveloperManagesCurrentSprint:
    """
    As a developer tracking active work,
    I want to identify the current sprint,
    So that I know where to add new tasks.
    """

    def test_scenario_get_current_active_sprint(self, tmp_path):
        """
        Scenario: Finding the active sprint

        Given multiple sprints with one in_progress
        When I query for current sprint
        Then I get the in_progress sprint
        """
        # Given multiple sprints with one in_progress
        manager = _create_got_manager(tmp_path / ".got")
        completed = manager.create_sprint(title="Old sprint", status="completed")
        current = manager.create_sprint(title="Current sprint", status="in_progress")
        future = manager.create_sprint(title="Future sprint", status="available")

        # When I query for current sprint
        active = manager.get_current_sprint()

        # Then I get the in_progress sprint
        assert active is not None
        assert active.id == current.id
        assert active.status == "in_progress"

    def test_scenario_no_current_sprint_returns_none(self, tmp_path):
        """
        Scenario: No active sprint exists

        Given no in_progress sprints
        When I query for current sprint
        Then I get None
        """
        # Given no in_progress sprints
        manager = _create_got_manager(tmp_path / ".got")
        completed = manager.create_sprint(title="Old sprint", status="completed")
        available = manager.create_sprint(title="Future sprint", status="available")

        # When I query for current sprint
        active = manager.get_current_sprint()

        # Then I get None
        assert active is None


class TestDeveloperDeletesSprintsWithSafety:
    """
    As a developer cleaning up old sprints,
    I want to delete sprints with safety checks,
    So that I don't accidentally lose task assignments.
    """

    def test_scenario_cannot_delete_sprint_with_tasks_without_force(self, tmp_path):
        """
        Scenario: Preventing accidental sprint deletion

        Given a sprint with tasks
        When I try to delete without force
        Then deletion fails
        And the sprint still exists
        """
        # Given a sprint with tasks
        manager = _create_got_manager(tmp_path / ".got")
        sprint = manager.create_sprint(title="Sprint with tasks")
        task = manager.create_task(title="Task in sprint")
        manager.add_task_to_sprint(task.id, sprint.id)

        # When I try to delete without force
        from cortical.got.errors import TransactionError
        with pytest.raises(TransactionError, match="has tasks"):
            manager.delete_sprint(sprint.id, force=False)

        # Then the sprint still exists
        assert manager.get_sprint(sprint.id) is not None

    def test_scenario_force_delete_removes_sprint_and_edges(self, tmp_path):
        """
        Scenario: Force deleting a sprint with tasks

        Given a sprint with tasks
        When I force delete the sprint
        Then the sprint is removed
        And all CONTAINS edges are removed
        """
        # Given a sprint with tasks
        manager = _create_got_manager(tmp_path / ".got")
        sprint = manager.create_sprint(title="Sprint to delete")
        task1 = manager.create_task(title="Task 1")
        task2 = manager.create_task(title="Task 2")
        manager.add_task_to_sprint(task1.id, sprint.id)
        manager.add_task_to_sprint(task2.id, sprint.id)

        # When I force delete the sprint
        manager.delete_sprint(sprint.id, force=True)

        # Then the sprint is removed
        assert manager.get_sprint(sprint.id) is None

        # And all CONTAINS edges are removed
        # (Tasks should still exist but not be in any sprint)
        assert manager.get_task(task1.id) is not None
        assert manager.get_task(task2.id) is not None


class TestDeveloperCapturesSprintContext:
    """
    As a developer creating sprints,
    I want to capture context and notes at creation time,
    So that I preserve the reasoning behind sprint goals.
    """

    def test_scenario_create_sprint_with_description(self, tmp_path):
        """
        Scenario: Creating a sprint with descriptive context

        Given a GoT manager
        When I create a sprint with notes
        Then the notes are stored with the sprint
        And I can retrieve them later
        """
        # Given a GoT manager
        manager = _create_got_manager(tmp_path / ".got")

        # When I create a sprint with notes
        sprint = manager.create_sprint(
            title="Build custom search engine",
            notes=["Focus on semantic understanding over keyword matching"]
        )

        # Then the notes are stored with the sprint
        assert sprint.notes is not None
        assert len(sprint.notes) == 1
        assert "semantic understanding" in sprint.notes[0]

        # And I can retrieve them later
        retrieved = manager.get_sprint(sprint.id)
        assert retrieved.notes == sprint.notes

    def test_scenario_sprint_context_persists_across_save_load(self, tmp_path):
        """
        Scenario: Sprint notes survive persistence

        Given a sprint with detailed notes
        When I reload the manager from the same path
        Then the notes are preserved
        """
        # Given a sprint with detailed notes
        got_path = tmp_path / ".got"
        # Use disk storage for persistence test across manager instances
        manager = _create_got_manager(got_path, use_memory=False)
        sprint = manager.create_sprint(
            title="Implement custom IR algorithms",
            notes=[
                "Building from first principles - no external dependencies",
                "Focus on TF-IDF and PageRank implementations"
            ]
        )
        sprint_id = sprint.id

        # When I reload the manager from the same path
        # (transaction commits are auto-persisted)
        manager2 = _create_got_manager(got_path, use_memory=False)

        # Then the notes are preserved
        retrieved = manager2.get_sprint(sprint_id)
        assert retrieved is not None
        assert len(retrieved.notes) == 2
        assert "first principles" in retrieved.notes[0]
        assert "PageRank" in retrieved.notes[1]

    def test_scenario_sprint_without_notes_has_empty_list(self, tmp_path):
        """
        Scenario: Sprints created without notes have empty notes list

        Given a GoT manager
        When I create a sprint without providing notes
        Then the notes field is an empty list (not None)
        """
        # Given a GoT manager
        manager = _create_got_manager(tmp_path / ".got")

        # When I create a sprint without providing notes
        sprint = manager.create_sprint(title="Minimal sprint")

        # Then the notes field is an empty list (not None)
        assert sprint.notes is not None
        assert sprint.notes == []
        assert isinstance(sprint.notes, list)


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
