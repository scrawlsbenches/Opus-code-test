"""
Behavioral tests for generic entity accessor in Query builder.

Tests the Query(manager).entities('entity_type') method for:
- Querying any registered entity type by string name
- Case-insensitive entity type names
- Helpful error messages for unknown types
- Filtering and chaining with other query methods
"""

import pytest
from pathlib import Path
from cortical.got import GoTManager
from cortical.got.query_builder import Query, QueryValidationError
from cortical.got.types import Task, Sprint, Decision
from tests.conftest import _create_got_manager


@pytest.fixture
def manager(tmp_path):
    """Create a GoTManager with test data."""
    got_dir = tmp_path / ".got"
    manager = _create_got_manager(got_dir)

    # Create some test tasks
    with manager.transaction():
        manager.create_task(
            "Test Task 1",
            status="pending",
            priority="high"
        )
        manager.create_task(
            "Test Task 2",
            status="completed",
            priority="medium"
        )
        manager.create_task(
            "Test Task 3",
            status="in_progress",
            priority="low"
        )

    # Create some test sprints
    with manager.transaction():
        manager.create_sprint("Sprint 1", status="available")
        manager.create_sprint("Sprint 2", status="in_progress")

    # Create some test decisions
    with manager.transaction():
        manager.create_decision(
            "Decision 1",
            rationale="Because reasons"
        )
        manager.create_decision(
            "Decision 2",
            rationale="More reasons"
        )

    return manager


def test_query_tasks_via_entities(manager):
    """
    GIVEN a manager with tasks
    WHEN querying via entities('task')
    THEN should return all tasks
    """
    results = Query(manager).entities('task').execute()

    assert len(results) == 3
    assert all(isinstance(r, Task) for r in results)
    assert all(r.entity_type == 'task' for r in results)


def test_query_sprints_via_entities(manager):
    """
    GIVEN a manager with sprints
    WHEN querying via entities('sprint')
    THEN should return all sprints
    """
    results = Query(manager).entities('sprint').execute()

    assert len(results) == 2
    assert all(isinstance(r, Sprint) for r in results)
    assert all(r.entity_type == 'sprint' for r in results)


def test_query_decisions_via_entities(manager):
    """
    GIVEN a manager with decisions
    WHEN querying via entities('decision')
    THEN should return all decisions
    """
    results = Query(manager).entities('decision').execute()

    assert len(results) == 2
    assert all(isinstance(r, Decision) for r in results)
    assert all(r.entity_type == 'decision' for r in results)


def test_case_insensitive_entity_type(manager):
    """
    GIVEN a manager with tasks
    WHEN querying via entities('TASK') or entities('TaSk')
    THEN should work (case-insensitive)
    """
    results_upper = Query(manager).entities('TASK').execute()
    results_mixed = Query(manager).entities('TaSk').execute()
    results_lower = Query(manager).entities('task').execute()

    assert len(results_upper) == 3
    assert len(results_mixed) == 3
    assert len(results_lower) == 3


def test_unknown_entity_type_raises_error_with_suggestions(manager):
    """
    GIVEN a manager
    WHEN querying with unknown entity type
    THEN should raise QueryValidationError with suggestions
    """
    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('taask').execute()

    error_msg = str(exc_info.value)
    assert "Unknown entity type 'taask'" in error_msg
    assert "Did you mean:" in error_msg
    assert "task" in error_msg  # Should suggest 'task'
    assert "Available:" in error_msg


def test_unknown_entity_type_shows_available_types(manager):
    """
    GIVEN a manager
    WHEN querying with completely unknown entity type
    THEN should show available entity types
    """
    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('nonexistent').execute()

    error_msg = str(exc_info.value)
    assert "Unknown entity type 'nonexistent'" in error_msg
    assert "Available:" in error_msg
    # Should list all registered entity types
    assert "task" in error_msg
    assert "sprint" in error_msg
    assert "decision" in error_msg


def test_entities_with_where_filter(manager):
    """
    GIVEN a manager with tasks
    WHEN querying entities('task').where(status='pending')
    THEN should filter correctly
    """
    results = Query(manager).entities('task').where(status='pending').execute()

    assert len(results) == 1
    assert results[0].status == 'pending'


def test_entities_with_multiple_filters(manager):
    """
    GIVEN a manager with tasks
    WHEN querying with multiple where conditions
    THEN should apply all filters
    """
    results = (
        Query(manager)
        .entities('task')
        .where(status='pending')
        .where(priority='high')
        .execute()
    )

    assert len(results) == 1
    assert results[0].status == 'pending'
    assert results[0].priority == 'high'


def test_entities_with_limit(manager):
    """
    GIVEN a manager with tasks
    WHEN querying entities('task').limit(2)
    THEN should return limited results
    """
    results = Query(manager).entities('task').limit(2).execute()

    assert len(results) == 2


def test_entities_count(manager):
    """
    GIVEN a manager with tasks
    WHEN calling .count() on entities query
    THEN should return count
    """
    count = Query(manager).entities('task').count()

    assert count == 3


def test_entities_first(manager):
    """
    GIVEN a manager with tasks
    WHEN calling .first() on entities query
    THEN should return first entity
    """
    result = Query(manager).entities('task').first()

    assert result is not None
    assert isinstance(result, Task)


def test_entities_exists(manager):
    """
    GIVEN a manager with tasks
    WHEN calling .exists() on entities query
    THEN should return True
    """
    exists = Query(manager).entities('task').exists()

    assert exists is True


def test_entities_exists_with_filter_no_match(manager):
    """
    GIVEN a manager with tasks
    WHEN calling .exists() with filter that matches nothing
    THEN should return False
    """
    exists = Query(manager).entities('task').where(status='blocked').exists()

    assert exists is False


def test_entities_iter(manager):
    """
    GIVEN a manager with tasks
    WHEN iterating via .iter()
    THEN should yield entities
    """
    entities = list(Query(manager).entities('task').iter())

    assert len(entities) == 3
    assert all(isinstance(e, Task) for e in entities)


def test_query_epic_entity_type(manager):
    """
    GIVEN a manager
    WHEN querying entities('epic')
    THEN should work (even though epic not in EntityType enum)
    """
    # Create an epic
    with manager.transaction():
        manager.create_epic("Test Epic")

    results = Query(manager).entities('epic').execute()

    assert len(results) == 1
    assert results[0].entity_type == 'epic'


def test_query_handoff_entity_type(manager):
    """
    GIVEN a manager
    WHEN querying entities('handoff')
    THEN should work (handoff is in enum but also works via string)
    """
    # Create a handoff
    with manager.transaction():
        task = manager.create_task("Task for handoff")
        manager.initiate_handoff(
            source_agent="agent-1",
            target_agent="agent-2",
            task_id=task.id,
            instructions="Do the thing"
        )

    results = Query(manager).entities('handoff').execute()

    assert len(results) == 1
    assert results[0].entity_type == 'handoff'


def test_entities_backward_compatible_with_enum_methods(manager):
    """
    GIVEN a manager with tasks
    WHEN using both .tasks() and .entities('task')
    THEN should return same results
    """
    enum_results = Query(manager).tasks().execute()
    string_results = Query(manager).entities('task').execute()

    assert len(enum_results) == len(string_results)
    assert set(e.id for e in enum_results) == set(e.id for e in string_results)


def test_entities_explain(manager):
    """
    GIVEN a query with entities()
    WHEN calling .explain()
    THEN should show entity type in plan
    """
    plan = Query(manager).entities('task').where(status='pending').explain()

    # Check that the plan includes the entity type
    assert plan.steps[0]['entity_type'] == 'task'
    assert plan.steps[0]['type'] == 'scan'
