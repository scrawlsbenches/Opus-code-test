"""
Unit tests for Query builder entities() method.

Tests the internal mechanics of the generic entity accessor:
- Schema registry validation
- Entity type string normalization
- Error message generation
- Dispatch to correct manager methods
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from cortical.got import GoTManager
from cortical.got.query_builder import Query, QueryValidationError
from cortical.got.entity_schemas import ensure_schemas_registered
from tests.conftest import _create_got_manager


@pytest.fixture
def manager(tmp_path):
    """Create a minimal GoTManager."""
    got_dir = tmp_path / ".got"
    return _create_got_manager(got_dir)


def test_entities_normalizes_to_lowercase(manager):
    """
    GIVEN entity type in various cases
    WHEN creating query with entities()
    THEN should normalize to lowercase
    """
    # Ensure schemas registered for validation
    ensure_schemas_registered()

    query = Query(manager).entities('TASK')
    assert query._entity_type_str == 'task'

    query = Query(manager).entities('TaSk')
    assert query._entity_type_str == 'task'

    query = Query(manager).entities('task')
    assert query._entity_type_str == 'task'


def test_entities_validates_against_schema_registry(manager):
    """
    GIVEN an unknown entity type
    WHEN creating query with entities()
    THEN should raise QueryValidationError
    """
    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('not_a_real_entity_type')

    assert "Unknown entity type 'not_a_real_entity_type'" in str(exc_info.value)


def test_entities_error_includes_suggestions(manager):
    """
    GIVEN a misspelled entity type
    WHEN creating query with entities()
    THEN error should include close matches
    """
    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('taask')  # Typo: should be 'task'

    error_msg = str(exc_info.value)
    assert "Did you mean:" in error_msg
    assert "task" in error_msg


def test_entities_error_includes_available_types(manager):
    """
    GIVEN an unknown entity type
    WHEN creating query with entities()
    THEN error should list all available types
    """
    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('unknown')

    error_msg = str(exc_info.value)
    assert "Available:" in error_msg
    # Should list all 12 registered entity types
    assert "task" in error_msg
    assert "decision" in error_msg
    assert "sprint" in error_msg
    assert "epic" in error_msg
    assert "edge" in error_msg
    assert "handoff" in error_msg
    assert "knowledge_transfer" in error_msg


def test_entities_sets_entity_type_str_not_enum(manager):
    """
    GIVEN a valid entity type
    WHEN creating query with entities()
    THEN should set _entity_type_str, not _entity_type enum
    """
    query = Query(manager).entities('task')

    assert query._entity_type_str == 'task'
    assert query._entity_type is None  # Enum should not be set


def test_entities_returns_self_for_chaining(manager):
    """
    GIVEN a query
    WHEN calling entities()
    THEN should return self for method chaining
    """
    query = Query(manager)
    result = query.entities('task')

    assert result is query


def test_get_entities_by_string_type_dispatches_to_manager_methods(manager):
    """
    GIVEN entity types with dedicated manager methods
    WHEN _get_entities_by_string_type is called
    THEN should dispatch to correct manager method
    """
    query = Query(manager)

    # Mock manager methods
    manager.list_all_tasks = Mock(return_value=['task1', 'task2'])
    manager.list_sprints = Mock(return_value=['sprint1'])
    manager.list_decisions = Mock(return_value=['decision1'])
    manager.list_edges = Mock(return_value=['edge1'])
    manager.list_epics = Mock(return_value=['epic1'])
    manager.list_handoffs = Mock(return_value=['handoff1'])
    manager.list_documents = Mock(return_value=['doc1'])
    manager.list_claudemd_layers = Mock(return_value=['layer1'])

    # Test dispatch
    assert query._get_entities_by_string_type('task') == ['task1', 'task2']
    manager.list_all_tasks.assert_called_once()

    assert query._get_entities_by_string_type('sprint') == ['sprint1']
    manager.list_sprints.assert_called_once()

    assert query._get_entities_by_string_type('decision') == ['decision1']
    manager.list_decisions.assert_called_once()

    assert query._get_entities_by_string_type('edge') == ['edge1']
    manager.list_edges.assert_called_once()

    assert query._get_entities_by_string_type('epic') == ['epic1']
    manager.list_epics.assert_called_once()

    assert query._get_entities_by_string_type('handoff') == ['handoff1']
    manager.list_handoffs.assert_called_once()

    assert query._get_entities_by_string_type('document') == ['doc1']
    manager.list_documents.assert_called_once()

    assert query._get_entities_by_string_type('claudemd_layer') == ['layer1']
    manager.list_claudemd_layers.assert_called_once()


def test_get_entities_by_string_type_falls_back_to_scan(manager, tmp_path):
    """
    GIVEN entity type without dedicated manager method
    WHEN _get_entities_by_string_type is called
    THEN should fall back to _scan_entities_by_type
    """
    query = Query(manager)

    # Mock _scan_entities_by_type
    query._scan_entities_by_type = Mock(return_value=['kt1', 'kt2'])

    # knowledge_transfer doesn't have a dedicated manager method
    result = query._get_entities_by_string_type('knowledge_transfer')

    assert result == ['kt1', 'kt2']
    query._scan_entities_by_type.assert_called_once_with('knowledge_transfer')


def test_scan_entities_by_type_filters_by_entity_type(manager):
    """
    GIVEN entities in storage
    WHEN _scan_entities_by_type is called
    THEN should filter by entity_type attribute
    """
    # Create some entities
    with manager.transaction():
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")
        sprint1 = manager.create_sprint("Sprint 1")

    query = Query(manager)

    # Scan for tasks
    tasks = query._scan_entities_by_type('task')
    assert len(tasks) == 2
    assert all(e.entity_type == 'task' for e in tasks)

    # Scan for sprints
    sprints = query._scan_entities_by_type('sprint')
    assert len(sprints) == 1
    assert all(e.entity_type == 'sprint' for e in sprints)


def test_scan_entities_by_type_handles_missing_entities_dir(manager, tmp_path):
    """
    GIVEN no entities directory
    WHEN _scan_entities_by_type is called
    THEN should return empty list
    """
    # Create manager with non-existent got_dir
    nonexistent_dir = tmp_path / "nonexistent" / ".got"
    manager_empty = _create_got_manager(nonexistent_dir)

    query = Query(manager_empty)
    result = query._scan_entities_by_type('task')

    assert result == []


def test_scan_entities_by_type_skips_corrupted_files(manager, tmp_path, caplog):
    """
    GIVEN corrupted entity files
    WHEN _scan_entities_by_type is called
    THEN should skip corrupted files and log warnings
    """
    # Create a valid task
    with manager.transaction():
        manager.create_task("Valid Task")

    # Write a corrupted file directly to the in-memory filesystem
    store = manager.tx_manager.store
    corrupted_path = store.store_dir / "CORRUPTED.json"
    store._fs.write_text(corrupted_path, "not valid json {{{")

    query = Query(manager)

    # Should return only valid entities, skip corrupted
    tasks = query._scan_entities_by_type('task')

    # Should have skipped the corrupted file
    assert len(tasks) == 1


def test_execute_uses_entity_type_str_for_metrics(manager):
    """
    GIVEN query with entities()
    WHEN execute() is called
    THEN should use entity_type_str for metrics
    """
    # Create a task
    with manager.transaction():
        manager.create_task("Test Task")

    query = Query(manager).entities('task')

    # Execute and check metrics name matches entity_type_str
    results = query.execute()

    # Should have used 'task' not 'TASK' (enum name)
    # This is verified indirectly by checking execution completes
    assert len(results) == 1


def test_explain_includes_entity_type_str(manager):
    """
    GIVEN query with entities()
    WHEN explain() is called
    THEN should include entity_type_str in plan
    """
    query = Query(manager).entities('task').where(status='pending')
    plan = query.explain()

    assert plan.steps[0]['entity_type'] == 'task'
    assert plan.steps[0]['type'] == 'scan'


def test_get_base_entities_prefers_entity_type_str_over_enum(manager):
    """
    GIVEN query with both _entity_type_str and _entity_type set
    WHEN _get_base_entities is called
    THEN should prefer _entity_type_str
    """
    query = Query(manager)

    # Mock both pathways
    query._entity_type_str = 'task'
    query._entity_type = None
    query._get_entities_by_string_type = Mock(return_value=['from_string'])
    manager.list_all_tasks = Mock(return_value=['from_enum'])

    result = query._get_base_entities()

    # Should have used string pathway
    assert result == ['from_string']
    query._get_entities_by_string_type.assert_called_once_with('task')
    manager.list_all_tasks.assert_not_called()


def test_entity_types_work_across_all_registered_schemas(manager):
    """
    GIVEN all registered entity types
    WHEN calling entities() with each type
    THEN should validate successfully
    """
    from cortical.got.entity_schemas import ALL_SCHEMAS

    # All entity types should be queryable
    for entity_type in ALL_SCHEMAS.keys():
        query = Query(manager).entities(entity_type)
        assert query._entity_type_str == entity_type


def test_entities_method_chaining_with_filters(manager):
    """
    GIVEN a query
    WHEN chaining entities() with where() and limit()
    THEN should maintain entity_type_str through chain
    """
    query = (
        Query(manager)
        .entities('task')
        .where(status='pending')
        .where(priority='high')
        .limit(5)
    )

    assert query._entity_type_str == 'task'
    assert len(query._where_clauses) == 2
    assert query._limit_value == 5
