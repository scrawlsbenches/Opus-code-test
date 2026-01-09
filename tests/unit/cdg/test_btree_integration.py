"""
Integration tests for BTree index support in CDGIndexManager.

Tests the complete flow:
- Schema-based btree index creation
- Index updates on entity writes
- Range queries via CDGIndexManager
- Query planner btree support
- Query executor btree range lookups

See: docs/design/cdg-query-language.md
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from cortical.cdg.index_manager import CDGIndexManager
from cortical.cdg.schema import SchemaRegistry, BaseSchema, Field, FieldType
from cortical.cdg.query.ast import CDGQuery, Comparison, Field as ASTField, Literal, Op
from cortical.cdg.query.planner import QueryPlanner
from cortical.common.filesystem import RealFileSystem


class TestEntitySchema(BaseSchema):
    """Test schema with both hash and btree indexed fields."""
    schema_version = 1
    entity_type = 'test_entity'
    id_prefix = 'TE-'

    fields = {
        'id': Field('id', FieldType.STRING, required=True),
        'entity_type': Field('entity_type', FieldType.STRING, required=True),
        'status': Field('status', FieldType.STRING, indexed=True, index_type='hash'),
        'created_at': Field('created_at', FieldType.STRING, indexed=True, index_type='btree'),
        'priority': Field('priority', FieldType.INTEGER, indexed=True, index_type='btree'),
        'title': Field('title', FieldType.STRING),
    }


@pytest.fixture
def schema_registry():
    """Create a schema registry with test schema."""
    registry = SchemaRegistry()
    registry.register('test_entity', TestEntitySchema)
    return registry


@pytest.fixture
def index_manager(schema_registry):
    """Create an index manager with btree support."""
    with TemporaryDirectory() as tmpdir:
        manager = CDGIndexManager(
            store_dir=Path(tmpdir),
            schema_registry=schema_registry,
            filesystem=RealFileSystem()
        )
        yield manager


class TestCDGIndexManagerBTreeSupport:
    """Test BTree index support in CDGIndexManager."""

    def test_btree_index_creation(self, index_manager):
        """BTree index is created for btree-indexed fields."""
        # Update index should create btree structure
        index_manager.update_index(
            'test_entity', 'TE-001',
            old_data=None,
            new_data={'created_at': '2026-01-01', 'priority': 5}
        )

        # Check that btree index exists
        assert index_manager.is_btree_indexed('test_entity', 'created_at')
        assert index_manager.is_btree_indexed('test_entity', 'priority')

        # Hash indexed field should not be btree
        assert not index_manager.is_btree_indexed('test_entity', 'status')

    def test_btree_index_update(self, index_manager):
        """BTree index is updated on entity changes."""
        # Create entity
        index_manager.update_index(
            'test_entity', 'TE-001',
            old_data=None,
            new_data={'created_at': '2026-01-01', 'priority': 5}
        )

        # Update entity
        index_manager.update_index(
            'test_entity', 'TE-001',
            old_data={'created_at': '2026-01-01', 'priority': 5},
            new_data={'created_at': '2026-01-05', 'priority': 10}
        )

        # Old value should not be found
        result = index_manager.lookup_range(
            'test_entity', 'created_at',
            start_value='2026-01-01',
            end_value='2026-01-01'
        )
        assert result == set()

        # New value should be found
        result = index_manager.lookup_range(
            'test_entity', 'created_at',
            start_value='2026-01-05',
            end_value='2026-01-05'
        )
        assert result == {'TE-001'}

    def test_btree_range_queries(self, index_manager):
        """Range queries work with btree indexes."""
        # Create multiple entities
        for i, date in enumerate(['2026-01-01', '2026-01-05', '2026-01-10', '2026-01-15']):
            index_manager.update_index(
                'test_entity', f'TE-00{i+1}',
                old_data=None,
                new_data={'created_at': date, 'priority': (i + 1) * 5}
            )

        # Test GT
        result = index_manager.lookup_gt('test_entity', 'created_at', '2026-01-05')
        assert result == {'TE-003', 'TE-004'}

        # Test GTE
        result = index_manager.lookup_gte('test_entity', 'created_at', '2026-01-05')
        assert result == {'TE-002', 'TE-003', 'TE-004'}

        # Test LT
        result = index_manager.lookup_lt('test_entity', 'created_at', '2026-01-10')
        assert result == {'TE-001', 'TE-002'}

        # Test LTE
        result = index_manager.lookup_lte('test_entity', 'created_at', '2026-01-10')
        assert result == {'TE-001', 'TE-002', 'TE-003'}

        # Test numeric range query
        result = index_manager.lookup_range(
            'test_entity', 'priority',
            start_value=5,
            end_value=15,
            start_inclusive=True,
            end_inclusive=True
        )
        assert result == {'TE-001', 'TE-002', 'TE-003'}

    def test_hash_index_still_works(self, index_manager):
        """Hash indexes still work alongside btree indexes."""
        index_manager.update_index(
            'test_entity', 'TE-001',
            old_data=None,
            new_data={'status': 'pending', 'created_at': '2026-01-01'}
        )
        index_manager.update_index(
            'test_entity', 'TE-002',
            old_data=None,
            new_data={'status': 'pending', 'created_at': '2026-01-05'}
        )
        index_manager.update_index(
            'test_entity', 'TE-003',
            old_data=None,
            new_data={'status': 'completed', 'created_at': '2026-01-10'}
        )

        # Hash index lookup
        result = index_manager.lookup('test_entity', 'status', 'pending')
        assert result == {'TE-001', 'TE-002'}

        # BTree range lookup
        result = index_manager.lookup_gt('test_entity', 'created_at', '2026-01-01')
        assert result == {'TE-002', 'TE-003'}

    def test_range_query_on_hash_index_raises(self, index_manager):
        """Range query on hash-indexed field raises ValueError."""
        index_manager.update_index(
            'test_entity', 'TE-001',
            old_data=None,
            new_data={'status': 'pending'}
        )

        with pytest.raises(ValueError, match="does not have a btree index"):
            index_manager.lookup_gt('test_entity', 'status', 'pending')

    def test_index_stats_includes_btree(self, index_manager):
        """Index stats include btree index information."""
        index_manager.update_index(
            'test_entity', 'TE-001',
            old_data=None,
            new_data={'status': 'pending', 'created_at': '2026-01-01', 'priority': 5}
        )
        index_manager.update_index(
            'test_entity', 'TE-002',
            old_data=None,
            new_data={'status': 'pending', 'created_at': '2026-01-05', 'priority': 10}
        )

        stats = index_manager.stats()

        # Check hash index stats
        assert stats['by_type']['test_entity']['status']['index_type'] == 'hash'
        assert stats['by_type']['test_entity']['status']['total_entries'] == 2

        # Check btree index stats
        assert stats['by_type']['test_entity']['created_at']['index_type'] == 'btree'
        assert stats['by_type']['test_entity']['created_at']['total_entries'] == 2


class TestQueryPlannerBTreeSupport:
    """Test QueryPlanner btree index support."""

    @pytest.fixture
    def planner(self, schema_registry):
        """Create a query planner with test schema."""
        return QueryPlanner(schema_registry)

    def test_planner_recognizes_btree_range_operators(self, planner):
        """Planner creates index lookup for range operators on btree fields."""
        # Query: FROM test_entity WHERE created_at > '2026-01-01'
        query = CDGQuery(
            entity_type='test_entity',
            expression=Comparison(
                field=ASTField(name='created_at'),
                op=Op.GT,
                value=Literal(value='2026-01-01')
            )
        )

        plan = planner.plan(query)

        # Should have an index lookup, not full scan
        assert len(plan.index_lookups) == 1
        assert plan.index_lookups[0].field == 'created_at'
        assert plan.index_lookups[0].op == Op.GT
        assert plan.index_lookups[0].value == '2026-01-01'

    def test_planner_rejects_range_on_hash_field(self, planner):
        """Planner doesn't create index lookup for range operators on hash fields."""
        # Query: FROM test_entity WHERE status > 'pending'
        query = CDGQuery(
            entity_type='test_entity',
            expression=Comparison(
                field=ASTField(name='status'),
                op=Op.GT,
                value=Literal(value='pending')
            )
        )

        plan = planner.plan(query)

        # Should not have index lookup (hash doesn't support GT)
        assert len(plan.index_lookups) == 0
        assert plan.post_filter is not None

    def test_planner_supports_all_range_operators(self, planner):
        """Planner supports all range operators for btree fields."""
        for op in [Op.GT, Op.GTE, Op.LT, Op.LTE]:
            query = CDGQuery(
                entity_type='test_entity',
                expression=Comparison(
                    field=ASTField(name='priority'),
                    op=op,
                    value=Literal(value=5)
                )
            )

            plan = planner.plan(query)
            assert len(plan.index_lookups) == 1, f"Failed for operator {op}"
            assert plan.index_lookups[0].op == op


class TestBTreeIndexPersistence:
    """Test BTree index persistence."""

    def test_btree_persistence_roundtrip(self, schema_registry):
        """BTree indexes are saved and loaded correctly."""
        with TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir)

            # Create and populate index
            manager1 = CDGIndexManager(
                store_dir=store_dir,
                schema_registry=schema_registry,
                filesystem=RealFileSystem()
            )

            manager1.update_index(
                'test_entity', 'TE-001',
                old_data=None,
                new_data={'created_at': '2026-01-01', 'priority': 5}
            )
            manager1.update_index(
                'test_entity', 'TE-002',
                old_data=None,
                new_data={'created_at': '2026-01-10', 'priority': 15}
            )

            # Persist
            manager1.persist()

            # Create new manager (simulates restart)
            manager2 = CDGIndexManager(
                store_dir=store_dir,
                schema_registry=schema_registry,
                filesystem=RealFileSystem()
            )

            # Verify btree indexes were loaded
            result = manager2.lookup_gt('test_entity', 'created_at', '2026-01-05')
            assert result == {'TE-002'}

            result = manager2.lookup_lte('test_entity', 'priority', 10)
            assert result == {'TE-001'}
