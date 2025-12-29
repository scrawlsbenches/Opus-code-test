"""
Behavioral tests for CEL Schema Migration.

User stories test the migration system from a schema maintainer's
perspective, focusing on evolving the event schema safely.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock

from cortical.cel.sanity.migration import (
    MigrationStatus,
    MigrationStep,
    MigrationPlan,
    SchemaMigrationEngine,
    rename_field,
    add_field,
    remove_field,
    transform_field,
    compose_transforms,
    by_event_type,
    by_content_field,
)
from cortical.cel.core.events import CognitiveEvent, EventType
from cortical.cel.core.references import MerkleRoot


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_event_store():
    """Create a mock event store."""
    store = MagicMock()
    store.iterate.return_value = iter([])
    store.append.return_value = MerkleRoot("new-event-id")
    return store


@pytest.fixture
def sample_v1_events():
    """Create events with v1 schema."""
    return [
        CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={
                '_schema_version': 'v1',
                'task_name': 'Old field name',  # Will be renamed to 'title'
                'task_priority': 'high',
            },
            concepts=('task',),
        ),
        CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={
                '_schema_version': 'v1',
                'task_name': 'Another task',
            },
            concepts=('task',),
        ),
    ]


@pytest.fixture
def sample_migration_step():
    """Create a sample migration step."""
    return MigrationStep(
        name='rename_task_name_to_title',
        description='Rename task_name field to title',
        event_filter=lambda e: 'task_name' in e.content,
        transform=rename_field('task_name', 'title'),
        version_from='v1',
        version_to='v2',
    )


# =============================================================================
# USER STORY: MigrationStatus
# =============================================================================

class TestMigrationStatusBehavior:
    """
    User Story: As a migration operator, I want clear status
    tracking, so I can understand migration progress.
    """

    def test_pending_is_initial_state(self):
        """PENDING indicates migration hasn't started."""
        assert MigrationStatus.PENDING.name == 'PENDING'

    def test_in_progress_indicates_running(self):
        """IN_PROGRESS indicates migration is running."""
        assert MigrationStatus.IN_PROGRESS.name == 'IN_PROGRESS'

    def test_completed_indicates_success(self):
        """COMPLETED indicates successful finish."""
        assert MigrationStatus.COMPLETED.name == 'COMPLETED'

    def test_failed_indicates_error(self):
        """FAILED indicates an error occurred."""
        assert MigrationStatus.FAILED.name == 'FAILED'

    def test_rolled_back_indicates_reversion(self):
        """ROLLED_BACK indicates migration was reverted."""
        assert MigrationStatus.ROLLED_BACK.name == 'ROLLED_BACK'


# =============================================================================
# USER STORY: MigrationStep
# =============================================================================

class TestMigrationStepBehavior:
    """
    User Story: As a schema designer, I want to define individual
    migration steps, so I can transform events incrementally.
    """

    def test_step_has_descriptive_name(self, sample_migration_step):
        """Step has a human-readable name."""
        assert sample_migration_step.name == 'rename_task_name_to_title'
        assert len(sample_migration_step.description) > 0

    def test_step_tracks_version_transition(self, sample_migration_step):
        """Step knows what versions it migrates between."""
        assert sample_migration_step.version_from == 'v1'
        assert sample_migration_step.version_to == 'v2'

    def test_applies_to_checks_filter(self, sample_migration_step, sample_v1_events):
        """applies_to() uses the event filter."""
        assert sample_migration_step.applies_to(sample_v1_events[0]) is True

        # Event without task_name
        other_event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'other_field': 'value'},
            concepts=(),
        )
        assert sample_migration_step.applies_to(other_event) is False

    def test_apply_transforms_event(self, sample_migration_step, sample_v1_events):
        """apply() creates new event with transformed content."""
        original = sample_v1_events[0]
        migrated = sample_migration_step.apply(original)

        # Should have 'title' instead of 'task_name'
        assert 'title' in migrated.content
        assert 'task_name' not in migrated.content
        assert migrated.content['title'] == 'Old field name'

    def test_apply_adds_migration_metadata(self, sample_migration_step, sample_v1_events):
        """apply() adds metadata about the migration."""
        original = sample_v1_events[0]
        migrated = sample_migration_step.apply(original)

        assert migrated.content['_migrated_from'] == original.id
        assert migrated.content['_migration_step'] == sample_migration_step.name
        assert migrated.content['_schema_version'] == 'v2'

    def test_apply_adds_causal_link(self, sample_migration_step, sample_v1_events):
        """apply() links new event to original."""
        original = sample_v1_events[0]
        migrated = sample_migration_step.apply(original)

        assert original.id in migrated.causal_parents


# =============================================================================
# USER STORY: MigrationPlan
# =============================================================================

class TestMigrationPlanBehavior:
    """
    User Story: As a release manager, I want to define complete
    migration plans with multiple steps, so I can evolve schemas
    in a coordinated way.
    """

    def test_plan_has_ordered_steps(self, sample_migration_step):
        """Plan contains steps in execution order."""
        plan = MigrationPlan(
            name='v1-to-v2',
            description='Migrate from v1 to v2 schema',
            steps=[sample_migration_step],
        )

        assert len(plan.steps) == 1
        assert plan.steps[0] == sample_migration_step

    def test_plan_starts_pending(self, sample_migration_step):
        """New plans start in PENDING status."""
        plan = MigrationPlan(
            name='test-plan',
            description='Test plan',
            steps=[sample_migration_step],
        )

        assert plan.status == MigrationStatus.PENDING

    def test_progress_tracks_completion(self, sample_migration_step):
        """Progress shows percentage of steps completed."""
        step2 = MigrationStep(
            name='step2',
            description='Second step',
            event_filter=lambda e: True,
            transform=lambda c: c,
            version_from='v2',
            version_to='v3',
        )

        plan = MigrationPlan(
            name='multi-step',
            description='Multi-step migration',
            steps=[sample_migration_step, step2],
        )

        assert plan.progress == 0.0  # No steps completed

        plan._current_step = 1
        assert plan.progress == 50.0  # 1/2 steps

        plan._current_step = 2
        assert plan.progress == 100.0  # All steps

    def test_dry_run_mode(self, sample_migration_step):
        """Plan can be configured for dry run."""
        plan = MigrationPlan(
            name='dry-run-test',
            description='Test dry run',
            steps=[sample_migration_step],
            dry_run=True,
        )

        assert plan.dry_run is True

    def test_plan_serialization(self, sample_migration_step):
        """Plan can be serialized for storage/logging."""
        plan = MigrationPlan(
            name='serialize-test',
            description='Test serialization',
            steps=[sample_migration_step],
        )

        data = plan.to_dict()

        assert data['name'] == 'serialize-test'
        assert data['status'] == 'PENDING'
        assert 'steps' in data
        assert 'progress' in data


# =============================================================================
# USER STORY: SchemaMigrationEngine
# =============================================================================

class TestSchemaMigrationEngineBehavior:
    """
    User Story: As a database administrator, I want an engine
    that executes migrations safely, so I can evolve schemas
    without data loss.
    """

    def test_engine_registers_plans(self, mock_event_store, sample_migration_step):
        """Engine accepts plan registration."""
        engine = SchemaMigrationEngine(mock_event_store)
        plan = MigrationPlan(
            name='test-plan',
            description='Test plan',
            steps=[sample_migration_step],
        )

        engine.register_plan(plan)

        # Should be able to migrate using this plan
        assert engine.can_migrate('v1', 'v2')

    def test_engine_executes_migration(
        self, mock_event_store, sample_migration_step, sample_v1_events
    ):
        """Engine executes registered migration plans."""
        mock_event_store.iterate.return_value = iter(sample_v1_events)

        engine = SchemaMigrationEngine(mock_event_store)
        plan = MigrationPlan(
            name='v1-to-v2',
            description='Migrate v1 to v2',
            steps=[sample_migration_step],
        )
        engine.register_plan(plan)

        result = engine.migrate('v1-to-v2')

        assert result.status == MigrationStatus.COMPLETED
        assert result._events_processed > 0

    def test_engine_handles_missing_plan(self, mock_event_store):
        """Engine raises error for unknown plan."""
        engine = SchemaMigrationEngine(mock_event_store)

        with pytest.raises(KeyError, match="Migration plan not found"):
            engine.migrate('nonexistent-plan')

    def test_engine_dry_run_mode(
        self, mock_event_store, sample_migration_step, sample_v1_events
    ):
        """Dry run counts events without persisting changes."""
        mock_event_store.iterate.return_value = iter(sample_v1_events)

        engine = SchemaMigrationEngine(mock_event_store)
        plan = MigrationPlan(
            name='dry-run',
            description='Dry run test',
            steps=[sample_migration_step],
            dry_run=True,
        )
        engine.register_plan(plan)

        result = engine.migrate('dry-run')

        # Events counted but not persisted
        assert result._events_processed > 0
        mock_event_store.append.assert_not_called()

    def test_engine_records_history(
        self, mock_event_store, sample_migration_step, sample_v1_events
    ):
        """Engine maintains migration history."""
        mock_event_store.iterate.return_value = iter(sample_v1_events)

        engine = SchemaMigrationEngine(mock_event_store)
        plan = MigrationPlan(
            name='history-test',
            description='History test',
            steps=[sample_migration_step],
        )
        engine.register_plan(plan)

        engine.migrate('history-test')

        assert len(engine.history) == 1
        assert engine.history[0]['name'] == 'history-test'


# =============================================================================
# USER STORY: Lazy Migration (Adapters)
# =============================================================================

class TestLazyMigrationBehavior:
    """
    User Story: As a performance-conscious operator, I want to
    migrate events lazily on read, so I don't have to migrate
    everything at once.
    """

    def test_adapter_registered_for_version(
        self, mock_event_store, sample_migration_step
    ):
        """Adapters are registered for specific schema versions."""
        engine = SchemaMigrationEngine(mock_event_store)
        engine.register_adapter('v1', sample_migration_step)

        # Adapter should be available
        assert 'v1' in engine._adapters

    def test_adapt_event_transforms_old_schema(
        self, mock_event_store, sample_migration_step, sample_v1_events
    ):
        """adapt_event() transforms events with old schema."""
        engine = SchemaMigrationEngine(mock_event_store)
        engine.register_adapter('v1', sample_migration_step)

        adapted = engine.adapt_event(sample_v1_events[0])

        assert 'title' in adapted.content
        assert 'task_name' not in adapted.content

    def test_adapt_event_returns_original_if_no_adapter(
        self, mock_event_store, sample_v1_events
    ):
        """adapt_event() returns original if no adapter matches."""
        engine = SchemaMigrationEngine(mock_event_store)

        # Create event with unknown version
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'_schema_version': 'v999'},
            concepts=(),
        )

        adapted = engine.adapt_event(event)

        assert adapted == event  # Unchanged


# =============================================================================
# USER STORY: Migration Path Finding
# =============================================================================

class TestMigrationPathFindingBehavior:
    """
    User Story: As a schema maintainer, I want to find migration
    paths between versions, so I can plan upgrades.
    """

    def test_can_migrate_detects_valid_path(self, mock_event_store):
        """can_migrate() returns True for valid paths."""
        engine = SchemaMigrationEngine(mock_event_store)

        step1 = MigrationStep(
            name='v1-to-v2',
            description='Step 1',
            event_filter=lambda e: True,
            transform=lambda c: c,
            version_from='v1',
            version_to='v2',
        )
        step2 = MigrationStep(
            name='v2-to-v3',
            description='Step 2',
            event_filter=lambda e: True,
            transform=lambda c: c,
            version_from='v2',
            version_to='v3',
        )

        plan = MigrationPlan(
            name='full-upgrade',
            description='Full upgrade path',
            steps=[step1, step2],
        )
        engine.register_plan(plan)

        assert engine.can_migrate('v1', 'v2') is True
        assert engine.can_migrate('v2', 'v3') is True
        assert engine.can_migrate('v1', 'v3') is True  # Transitive

    def test_can_migrate_returns_false_for_invalid_path(self, mock_event_store):
        """can_migrate() returns False when no path exists."""
        engine = SchemaMigrationEngine(mock_event_store)

        assert engine.can_migrate('v1', 'v99') is False

    def test_get_migration_path_returns_steps(self, mock_event_store):
        """get_migration_path() returns ordered steps."""
        engine = SchemaMigrationEngine(mock_event_store)

        step1 = MigrationStep(
            name='v1-to-v2',
            description='Step 1',
            event_filter=lambda e: True,
            transform=lambda c: c,
            version_from='v1',
            version_to='v2',
        )
        step2 = MigrationStep(
            name='v2-to-v3',
            description='Step 2',
            event_filter=lambda e: True,
            transform=lambda c: c,
            version_from='v2',
            version_to='v3',
        )

        plan = MigrationPlan(
            name='path-test',
            description='Path test',
            steps=[step1, step2],
        )
        engine.register_plan(plan)

        path = engine.get_migration_path('v1', 'v3')

        assert len(path) == 2
        assert path[0].name == 'v1-to-v2'
        assert path[1].name == 'v2-to-v3'

    def test_get_migration_path_raises_for_no_path(self, mock_event_store):
        """get_migration_path() raises ValueError when no path exists."""
        engine = SchemaMigrationEngine(mock_event_store)

        with pytest.raises(ValueError, match="No migration path"):
            engine.get_migration_path('v1', 'v99')


# =============================================================================
# USER STORY: Transform Helpers
# =============================================================================

class TestTransformHelpersBehavior:
    """
    User Story: As a migration author, I want helper functions
    for common transformations, so I don't have to write boilerplate.
    """

    def test_rename_field_moves_value(self):
        """rename_field() moves value to new key."""
        transform = rename_field('old_name', 'new_name')
        content = {'old_name': 'value', 'other': 'data'}

        result = transform(content)

        assert result['new_name'] == 'value'
        assert 'old_name' not in result
        assert result['other'] == 'data'

    def test_rename_field_handles_missing(self):
        """rename_field() handles missing field gracefully."""
        transform = rename_field('nonexistent', 'new_name')
        content = {'other': 'data'}

        result = transform(content)

        assert 'new_name' not in result
        assert result['other'] == 'data'

    def test_add_field_sets_default(self):
        """add_field() adds field with default value."""
        transform = add_field('new_field', 'default_value')
        content = {'existing': 'data'}

        result = transform(content)

        assert result['new_field'] == 'default_value'
        assert result['existing'] == 'data'

    def test_add_field_respects_existing(self):
        """add_field() doesn't overwrite existing values."""
        transform = add_field('field', 'default')
        content = {'field': 'existing_value'}

        result = transform(content)

        assert result['field'] == 'existing_value'

    def test_remove_field_deletes_key(self):
        """remove_field() removes the field."""
        transform = remove_field('to_remove')
        content = {'to_remove': 'value', 'keep': 'data'}

        result = transform(content)

        assert 'to_remove' not in result
        assert result['keep'] == 'data'

    def test_remove_field_handles_missing(self):
        """remove_field() handles missing field gracefully."""
        transform = remove_field('nonexistent')
        content = {'other': 'data'}

        result = transform(content)

        assert result == {'other': 'data'}

    def test_transform_field_modifies_value(self):
        """transform_field() applies function to value."""
        transform = transform_field('count', lambda x: x * 2)
        content = {'count': 5}

        result = transform(content)

        assert result['count'] == 10

    def test_compose_transforms_chains_operations(self):
        """compose_transforms() chains multiple transforms."""
        transform = compose_transforms(
            rename_field('old', 'new'),
            add_field('added', 'value'),
            remove_field('unused'),
        )
        content = {'old': 'data', 'unused': 'junk'}

        result = transform(content)

        assert result['new'] == 'data'
        assert result['added'] == 'value'
        assert 'old' not in result
        assert 'unused' not in result


# =============================================================================
# USER STORY: Event Filters
# =============================================================================

class TestEventFiltersBehavior:
    """
    User Story: As a migration author, I want helper functions
    for filtering events, so I can target specific events.
    """

    def test_by_event_type_filters_correctly(self):
        """by_event_type() filters by event type."""
        filter_fn = by_event_type(EventType.INTENTION, EventType.FULFILLMENT)

        intention = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={},
            concepts=(),
        )
        observation = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={},
            concepts=(),
        )

        assert filter_fn(intention) is True
        assert filter_fn(observation) is False

    def test_by_content_field_checks_existence(self):
        """by_content_field() checks if field exists."""
        filter_fn = by_content_field('required_field')

        with_field = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'required_field': 'value'},
            concepts=(),
        )
        without_field = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'other_field': 'value'},
            concepts=(),
        )

        assert filter_fn(with_field) is True
        assert filter_fn(without_field) is False

    def test_by_content_field_checks_value(self):
        """by_content_field() checks specific value."""
        filter_fn = by_content_field('status', value='active')

        active = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'status': 'active'},
            concepts=(),
        )
        inactive = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'status': 'inactive'},
            concepts=(),
        )

        assert filter_fn(active) is True
        assert filter_fn(inactive) is False

    def test_by_content_field_checks_absence(self):
        """by_content_field() can check field absence."""
        filter_fn = by_content_field('deprecated_field', exists=False)

        without = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'other': 'data'},
            concepts=(),
        )
        with_field = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'deprecated_field': 'old'},
            concepts=(),
        )

        assert filter_fn(without) is True
        assert filter_fn(with_field) is False
