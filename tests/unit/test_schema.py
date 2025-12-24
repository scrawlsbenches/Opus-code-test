"""
Unit tests for GoT schema system.

Tests schema definitions, field validation, migrations, and registry.
"""

import pytest
from typing import Dict, Any

from cortical.got.schema import (
    BaseSchema,
    Field,
    FieldType,
    SchemaRegistry,
    ValidationResult,
    get_registry,
    register_schema,
    validate_entity,
    migrate_entity,
)


class TestField:
    """Tests for Field class."""

    def test_string_field_validation(self):
        """Test string field type validation."""
        field = Field('name', FieldType.STRING, required=True)

        valid, error = field.validate('hello')
        assert valid
        assert error is None

        valid, error = field.validate(123)
        assert not valid
        assert 'expected STRING' in error

    def test_integer_field_validation(self):
        """Test integer field type validation."""
        field = Field('count', FieldType.INTEGER, required=True)

        valid, error = field.validate(42)
        assert valid

        valid, error = field.validate('42')
        assert not valid
        assert 'expected INTEGER' in error

    def test_float_field_validation(self):
        """Test float field accepts int and float."""
        field = Field('score', FieldType.FLOAT, required=True)

        valid, _ = field.validate(3.14)
        assert valid

        valid, _ = field.validate(42)  # int should be accepted
        assert valid

    def test_boolean_field_validation(self):
        """Test boolean field type validation."""
        field = Field('active', FieldType.BOOLEAN, required=True)

        valid, _ = field.validate(True)
        assert valid

        valid, _ = field.validate(False)
        assert valid

        valid, error = field.validate('true')
        assert not valid

    def test_list_field_validation(self):
        """Test list field type validation."""
        field = Field('items', FieldType.LIST, required=True)

        valid, _ = field.validate([1, 2, 3])
        assert valid

        valid, error = field.validate('not a list')
        assert not valid

    def test_list_field_with_item_type(self):
        """Test list field with item type validation."""
        field = Field('names', FieldType.LIST, item_type=FieldType.STRING)

        valid, _ = field.validate(['a', 'b', 'c'])
        assert valid

        valid, error = field.validate(['a', 123, 'c'])
        assert not valid
        assert 'names[1]' in error

    def test_dict_field_validation(self):
        """Test dict field type validation."""
        field = Field('metadata', FieldType.DICT, required=True)

        valid, _ = field.validate({'key': 'value'})
        assert valid

        valid, error = field.validate([1, 2])
        assert not valid

    def test_enum_field_validation(self):
        """Test enum field with choices."""
        field = Field(
            'status',
            FieldType.ENUM,
            choices=['pending', 'done', 'cancelled']
        )

        valid, _ = field.validate('pending')
        assert valid

        valid, error = field.validate('invalid')
        assert not valid
        assert 'must be one of' in error

    def test_required_field_none(self):
        """Test required field fails on None."""
        field = Field('name', FieldType.STRING, required=True)

        valid, error = field.validate(None)
        assert not valid
        assert 'required' in error

    def test_optional_field_none(self):
        """Test optional field accepts None."""
        field = Field('name', FieldType.STRING, required=False)

        valid, error = field.validate(None)
        assert valid
        assert error is None

    def test_custom_validator(self):
        """Test custom validator function."""
        def is_positive(value):
            return value > 0

        field = Field(
            'count',
            FieldType.INTEGER,
            validator=is_positive
        )

        valid, _ = field.validate(5)
        assert valid

        valid, error = field.validate(-1)
        assert not valid
        assert 'failed custom validation' in error

    def test_apply_default(self):
        """Test applying default values."""
        field = Field('count', FieldType.INTEGER, required=False, default=0)

        data = {}
        field.apply_default(data)
        assert data['count'] == 0

        # Should not override existing value
        data = {'count': 5}
        field.apply_default(data)
        assert data['count'] == 5

    def test_default_mutable_copy(self):
        """Test that mutable defaults are deep copied."""
        field = Field('tags', FieldType.LIST, required=False, default=[])

        data1 = {}
        field.apply_default(data1)
        data1['tags'].append('test')

        data2 = {}
        field.apply_default(data2)

        assert data1['tags'] == ['test']
        assert data2['tags'] == []  # Should be independent


class TestBaseSchema:
    """Tests for BaseSchema class."""

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
                'name': Field('name', FieldType.STRING, required=True),
            }

        result = TestSchema.validate({'id': '123', 'name': 'test'})
        assert result.valid
        assert len(result.errors) == 0

    def test_validate_missing_required(self):
        """Test validation catches missing required fields."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
                'name': Field('name', FieldType.STRING, required=True),
            }

        result = TestSchema.validate({'id': '123'})
        assert not result.valid
        assert any('name' in e for e in result.errors)

    def test_validate_strict_mode(self):
        """Test strict mode warns about unknown fields."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
            }

        result = TestSchema.validate(
            {'id': '123', 'unknown': 'value'},
            strict=True
        )
        assert result.valid  # Unknown fields are warnings, not errors
        assert any('unknown' in w.lower() for w in result.warnings)

    def test_apply_defaults(self):
        """Test applying all defaults."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
                'count': Field('count', FieldType.INTEGER, default=0),
                'tags': Field('tags', FieldType.LIST, default=[]),
            }

        data = {'id': '123'}
        result = TestSchema.apply_defaults(data)

        assert result['id'] == '123'
        assert result['count'] == 0
        assert result['tags'] == []

    def test_migration_discovery(self):
        """Test migration methods are discovered."""
        class TestSchema(BaseSchema):
            schema_version = 3
            entity_type = 'test'
            fields = {}

            @classmethod
            def migrate_v1_to_v2(cls, data):
                data['v2_field'] = True
                return data

            @classmethod
            def migrate_v2_to_v3(cls, data):
                data['v3_field'] = True
                return data

        migrations = TestSchema.get_migrations()
        assert 1 in migrations
        assert 2 in migrations
        assert len(migrations) == 2

    def test_migrate_single_version(self):
        """Test migration from v1 to v2."""
        class TestSchema(BaseSchema):
            schema_version = 2
            entity_type = 'test'
            fields = {}

            @classmethod
            def migrate_v1_to_v2(cls, data):
                data['new_field'] = 'migrated'
                return data

        data = {'id': '123', '_schema_version': 1}
        migrated, result = TestSchema.migrate(data)

        assert result.valid
        assert result.migrated
        assert result.from_version == 1
        assert result.to_version == 2
        assert migrated['new_field'] == 'migrated'
        assert migrated['_schema_version'] == 2

    def test_migrate_multiple_versions(self):
        """Test migration chain across multiple versions."""
        class TestSchema(BaseSchema):
            schema_version = 3
            entity_type = 'test'
            fields = {}

            @classmethod
            def migrate_v1_to_v2(cls, data):
                data['v2'] = True
                return data

            @classmethod
            def migrate_v2_to_v3(cls, data):
                data['v3'] = True
                return data

        data = {'id': '123', '_schema_version': 1}
        migrated, result = TestSchema.migrate(data)

        assert result.valid
        assert result.migrated
        assert migrated['v2'] is True
        assert migrated['v3'] is True
        assert migrated['_schema_version'] == 3

    def test_migrate_no_migration_needed(self):
        """Test migration skips when already current."""
        class TestSchema(BaseSchema):
            schema_version = 2
            entity_type = 'test'
            fields = {}

        data = {'id': '123', '_schema_version': 2}
        migrated, result = TestSchema.migrate(data)

        assert result.valid
        assert not result.migrated
        assert migrated == data

    def test_migrate_auto_detect_version(self):
        """Test migration auto-detects version from data."""
        class TestSchema(BaseSchema):
            schema_version = 2
            entity_type = 'test'
            fields = {}

            @classmethod
            def migrate_v1_to_v2(cls, data):
                data['migrated'] = True
                return data

        # No version key defaults to v1
        data = {'id': '123'}
        migrated, result = TestSchema.migrate(data)

        assert result.migrated
        assert migrated['migrated'] is True

    def test_prepare_for_save(self):
        """Test prepare_for_save applies defaults and version."""
        class TestSchema(BaseSchema):
            schema_version = 2
            entity_type = 'test'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
                'count': Field('count', FieldType.INTEGER, default=0),
            }

        data = {'id': '123'}
        prepared = TestSchema.prepare_for_save(data)

        assert prepared['count'] == 0
        assert prepared['_schema_version'] == 2


class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        reg = SchemaRegistry()
        reg.clear()
        return reg

    def test_register_and_get(self, registry):
        """Test schema registration and retrieval."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {}

        registry.register('test', TestSchema)
        assert registry.has_schema('test')
        assert registry.get_schema('test') is TestSchema

    def test_get_unregistered(self, registry):
        """Test getting unregistered schema returns None."""
        assert registry.get_schema('unknown') is None
        assert not registry.has_schema('unknown')

    def test_validate_through_registry(self, registry):
        """Test validation through registry."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'name': Field('name', FieldType.STRING, required=True),
            }

        registry.register('test', TestSchema)
        result = registry.validate('test', {'name': 'hello'})
        assert result.valid

    def test_validate_unregistered_type(self, registry):
        """Test validation of unregistered type warns."""
        result = registry.validate('unknown', {'data': 'value'})
        assert result.valid  # No schema = no validation = valid
        assert any('no schema' in w.lower() for w in result.warnings)

    def test_migrate_through_registry(self, registry):
        """Test migration through registry."""
        class TestSchema(BaseSchema):
            schema_version = 2
            entity_type = 'test'
            fields = {}

            @classmethod
            def migrate_v1_to_v2(cls, data):
                data['migrated'] = True
                return data

        registry.register('test', TestSchema)
        data = {'id': '123', '_schema_version': 1}
        migrated, result = registry.migrate('test', data)

        assert result.migrated
        assert migrated['migrated'] is True

    def test_prepare_for_save_through_registry(self, registry):
        """Test prepare_for_save through registry."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'count': Field('count', FieldType.INTEGER, default=0),
            }

        registry.register('test', TestSchema)
        data = {'id': '123'}
        prepared = registry.prepare_for_save('test', data)

        assert prepared['count'] == 0
        assert prepared['_schema_version'] == 1

    def test_list_schemas(self, registry):
        """Test listing registered schemas."""
        class Schema1(BaseSchema):
            schema_version = 1

        class Schema2(BaseSchema):
            schema_version = 2

        registry.register('type1', Schema1)
        registry.register('type2', Schema2)

        schemas = registry.list_schemas()
        assert schemas == {'type1': 1, 'type2': 2}

    def test_singleton_pattern(self):
        """Test registry is singleton."""
        reg1 = SchemaRegistry()
        reg2 = SchemaRegistry()
        assert reg1 is reg2


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear global registry before each test."""
        get_registry().clear()
        yield
        get_registry().clear()

    def test_register_schema(self):
        """Test global register_schema function."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {}

        register_schema('test', TestSchema)
        assert get_registry().has_schema('test')

    def test_validate_entity(self):
        """Test global validate_entity function."""
        class TestSchema(BaseSchema):
            schema_version = 1
            entity_type = 'test'
            fields = {
                'name': Field('name', FieldType.STRING, required=True),
            }

        register_schema('test', TestSchema)
        result = validate_entity('test', {'name': 'hello'})
        assert result.valid

    def test_migrate_entity(self):
        """Test global migrate_entity function."""
        class TestSchema(BaseSchema):
            schema_version = 2
            entity_type = 'test'
            fields = {}

            @classmethod
            def migrate_v1_to_v2(cls, data):
                data['v2'] = True
                return data

        register_schema('test', TestSchema)
        data = {'id': '123', '_schema_version': 1}
        migrated, result = migrate_entity('test', data)

        assert result.migrated
        assert migrated['v2'] is True


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_add_error(self):
        """Test adding errors invalidates result."""
        result = ValidationResult(valid=True)
        result.add_error('Something wrong')

        assert not result.valid
        assert 'Something wrong' in result.errors

    def test_add_warning(self):
        """Test warnings don't invalidate result."""
        result = ValidationResult(valid=True)
        result.add_warning('Minor issue')

        assert result.valid
        assert 'Minor issue' in result.warnings

    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult(valid=True)
        result1.add_warning('Warning 1')

        result2 = ValidationResult(valid=True)
        result2.add_error('Error 1')
        result2.add_warning('Warning 2')

        result1.merge(result2)

        assert not result1.valid
        assert 'Error 1' in result1.errors
        assert 'Warning 1' in result1.warnings
        assert 'Warning 2' in result1.warnings
