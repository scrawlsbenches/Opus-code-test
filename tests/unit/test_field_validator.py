"""
Unit tests for field validation in query expressions.

Tests:
- Valid common fields pass validation
- Valid entity-specific fields pass (when entity_type specified)
- Invalid fields raise QueryValidationError with suggestions
- Nested expressions (AND/OR/NOT) validate all fields
- Function call arguments are validated correctly
"""

import pytest

from cortical.got.expression.ast import (
    Field,
    Literal,
    Comparison,
    AndExpr,
    OrExpr,
    NotExpr,
    FunctionCall,
    Op,
)
from cortical.got.expression.validator import FieldValidator, COMMON_FIELDS
from cortical.got.expression.errors import QueryValidationError
from cortical.cdg.schema import SchemaRegistry
from cortical.got.entity_schemas import register_all_schemas


@pytest.fixture
def schema_registry():
    """Create a properly-initialized SchemaRegistry for field validation tests."""
    registry = SchemaRegistry()
    register_all_schemas(registry)
    return registry


class TestCommonFieldValidation:
    """Test validation of common fields (valid for all entity types)."""

    def test_valid_common_fields_without_entity_type(self, schema_registry):
        """Common fields should validate when no entity_type specified."""
        validator = FieldValidator(schema_registry, entity_type=None)

        for field_name in COMMON_FIELDS:
            # Should not raise
            validator.validate_field(field_name)

    def test_valid_common_fields_with_entity_type(self, schema_registry):
        """Common fields should validate even with entity_type specified."""
        validator = FieldValidator(schema_registry, entity_type='task')

        for field_name in COMMON_FIELDS:
            # Should not raise
            validator.validate_field(field_name)

    def test_common_fields_in_comparisons(self, schema_registry):
        """Common fields in comparison expressions should validate."""
        validator = FieldValidator(schema_registry, entity_type=None)

        # status = 'pending'
        expr = Comparison(
            field=Field(name='status'),
            op=Op.EQ,
            value=Literal(value='pending')
        )

        # Should not raise
        validator.validate_expression(expr)

    def test_all_common_fields_are_valid(self, schema_registry):
        """Test each common field individually."""
        validator = FieldValidator(schema_registry, entity_type=None)

        # Test each common field
        for field_name in ['id', 'title', 'status', 'created_at', 'modified_at']:
            validator.validate_field(field_name)


class TestEntitySpecificFieldValidation:
    """Test validation of entity-specific fields."""

    def test_task_specific_fields(self, schema_registry):
        """Task-specific fields should validate with entity_type='task'."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # Task-specific fields
        task_fields = ['priority', 'description']
        for field_name in task_fields:
            # Should not raise
            validator.validate_field(field_name)

    def test_decision_specific_fields(self, schema_registry):
        """Decision-specific fields should validate with entity_type='decision'."""
        validator = FieldValidator(schema_registry, entity_type='decision')

        # Decision-specific fields
        validator.validate_field('rationale')
        validator.validate_field('affects')

    def test_knowledge_transfer_specific_fields(self, schema_registry):
        """KT-specific fields should validate with entity_type='knowledge_transfer'."""
        validator = FieldValidator(schema_registry, entity_type='knowledge_transfer')

        # KT-specific fields
        validator.validate_field('summary')
        validator.validate_field('session_id')
        validator.validate_field('session_date')

    def test_entity_specific_field_not_valid_without_entity_type(self, schema_registry):
        """Entity-specific fields should fail without entity_type."""
        validator = FieldValidator(schema_registry, entity_type=None)

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_field('priority')

        assert 'priority' in str(exc_info.value)
        assert 'Unknown field' in str(exc_info.value)

    def test_entity_specific_field_not_valid_for_wrong_entity_type(self, schema_registry):
        """Task fields shouldn't validate for decision entity type."""
        validator = FieldValidator(schema_registry, entity_type='decision')

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_field('priority')

        assert 'priority' in str(exc_info.value)


class TestInvalidFieldValidation:
    """Test that invalid fields raise appropriate errors."""

    def test_completely_invalid_field(self, schema_registry):
        """Completely invalid field should raise error."""
        validator = FieldValidator(schema_registry, entity_type='task')

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_field('nonexistent_field')

        error = exc_info.value
        assert 'Unknown field' in error.message
        assert 'nonexistent_field' in error.message
        assert error.field_name == 'nonexistent_field'
        assert error.valid_fields is not None

    def test_typo_field_gets_suggestion(self, schema_registry):
        """Typo in field name should suggest correct field."""
        validator = FieldValidator(schema_registry, entity_type='task')

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_field('stat')  # Typo for 'status'

        error = exc_info.value
        # Should auto-suggest 'status' based on prefix match
        assert len(error.suggestions) > 0
        assert 'status' in error.suggestions

    def test_partial_field_name_gets_suggestion(self, schema_registry):
        """Partial field name should suggest full field."""
        validator = FieldValidator(schema_registry, entity_type='task')

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_field('prior')  # Partial 'priority'

        error = exc_info.value
        # Should suggest 'priority' based on prefix match
        assert len(error.suggestions) > 0
        assert 'priority' in error.suggestions

    def test_error_includes_valid_fields_list(self, schema_registry):
        """Error should include list of valid fields."""
        validator = FieldValidator(schema_registry, entity_type='task')

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_field('invalid')

        error = exc_info.value
        assert error.valid_fields is not None
        # Should include common fields
        assert 'status' in error.valid_fields
        assert 'title' in error.valid_fields
        # Should include task-specific fields
        assert 'priority' in error.valid_fields


class TestNestedExpressionValidation:
    """Test validation of nested logical expressions."""

    def test_and_expression_validates_all_children(self, schema_registry):
        """AND expression should validate all children."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # status = 'pending' AND priority = 'high'
        expr = AndExpr(children=(
            Comparison(
                field=Field(name='status'),
                op=Op.EQ,
                value=Literal(value='pending')
            ),
            Comparison(
                field=Field(name='priority'),
                op=Op.EQ,
                value=Literal(value='high')
            ),
        ))

        # Should not raise
        validator.validate_expression(expr)

    def test_and_expression_fails_on_invalid_child(self, schema_registry):
        """AND expression should fail if any child has invalid field."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # status = 'pending' AND invalid_field = 'value'
        expr = AndExpr(children=(
            Comparison(
                field=Field(name='status'),
                op=Op.EQ,
                value=Literal(value='pending')
            ),
            Comparison(
                field=Field(name='invalid_field'),
                op=Op.EQ,
                value=Literal(value='value')
            ),
        ))

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_expression(expr)

        assert 'invalid_field' in str(exc_info.value)

    def test_or_expression_validates_all_children(self, schema_registry):
        """OR expression should validate all children."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # status = 'pending' OR status = 'blocked'
        expr = OrExpr(children=(
            Comparison(
                field=Field(name='status'),
                op=Op.EQ,
                value=Literal(value='pending')
            ),
            Comparison(
                field=Field(name='status'),
                op=Op.EQ,
                value=Literal(value='blocked')
            ),
        ))

        # Should not raise
        validator.validate_expression(expr)

    def test_or_expression_fails_on_invalid_child(self, schema_registry):
        """OR expression should fail if any child has invalid field."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # status = 'pending' OR bad_field = 'value'
        expr = OrExpr(children=(
            Comparison(
                field=Field(name='status'),
                op=Op.EQ,
                value=Literal(value='pending')
            ),
            Comparison(
                field=Field(name='bad_field'),
                op=Op.EQ,
                value=Literal(value='value')
            ),
        ))

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_expression(expr)

        assert 'bad_field' in str(exc_info.value)

    def test_not_expression_validates_child(self, schema_registry):
        """NOT expression should validate its child."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # NOT (status = 'completed')
        expr = NotExpr(
            child=Comparison(
                field=Field(name='status'),
                op=Op.EQ,
                value=Literal(value='completed')
            )
        )

        # Should not raise
        validator.validate_expression(expr)

    def test_not_expression_fails_on_invalid_child(self, schema_registry):
        """NOT expression should fail if child has invalid field."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # NOT (invalid = 'value')
        expr = NotExpr(
            child=Comparison(
                field=Field(name='invalid'),
                op=Op.EQ,
                value=Literal(value='value')
            )
        )

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_expression(expr)

        assert 'invalid' in str(exc_info.value)

    def test_deeply_nested_expression(self, schema_registry):
        """Deeply nested expression should validate all fields."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # (status = 'pending' AND priority = 'high') OR NOT (status = 'blocked')
        expr = OrExpr(children=(
            AndExpr(children=(
                Comparison(
                    field=Field(name='status'),
                    op=Op.EQ,
                    value=Literal(value='pending')
                ),
                Comparison(
                    field=Field(name='priority'),
                    op=Op.EQ,
                    value=Literal(value='high')
                ),
            )),
            NotExpr(
                child=Comparison(
                    field=Field(name='status'),
                    op=Op.EQ,
                    value=Literal(value='blocked')
                )
            ),
        ))

        # Should not raise
        validator.validate_expression(expr)

    def test_deeply_nested_expression_with_invalid_field(self, schema_registry):
        """Deeply nested expression should catch invalid field anywhere."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # (status = 'pending' AND invalid = 'high') OR NOT (status = 'blocked')
        expr = OrExpr(children=(
            AndExpr(children=(
                Comparison(
                    field=Field(name='status'),
                    op=Op.EQ,
                    value=Literal(value='pending')
                ),
                Comparison(
                    field=Field(name='invalid'),
                    op=Op.EQ,
                    value=Literal(value='high')
                ),
            )),
            NotExpr(
                child=Comparison(
                    field=Field(name='status'),
                    op=Op.EQ,
                    value=Literal(value='blocked')
                )
            ),
        ))

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_expression(expr)

        assert 'invalid' in str(exc_info.value)


class TestFunctionCallValidation:
    """Test that function calls are handled correctly."""

    def test_function_arguments_with_field_references(self, schema_registry):
        """Function arguments that reference fields should be validated."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # Some hypothetical function: days_since(created_at)
        expr = FunctionCall(
            name='days_since',
            args=(Field(name='created_at'),),
            kwargs=()
        )

        # Should not raise - created_at is a valid field
        validator.validate_expression(expr)

    def test_function_arguments_with_invalid_field(self, schema_registry):
        """Function arguments with invalid fields should fail validation."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # days_since(invalid_field)
        expr = FunctionCall(
            name='days_since',
            args=(Field(name='invalid_field'),),
            kwargs=()
        )

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_expression(expr)

        assert 'invalid_field' in str(exc_info.value)

    def test_function_kwargs_with_field_references(self, schema_registry):
        """Function kwargs that reference fields should be validated."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # Some function with kwargs: func(field=status)
        expr = FunctionCall(
            name='some_func',
            args=(),
            kwargs=(('field', Field(name='status')),)
        )

        # Should not raise
        validator.validate_expression(expr)

    def test_function_kwargs_with_invalid_field(self, schema_registry):
        """Function kwargs with invalid fields should fail validation."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # func(field=invalid_field)
        expr = FunctionCall(
            name='some_func',
            args=(),
            kwargs=(('field', Field(name='invalid_field')),)
        )

        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_expression(expr)

        assert 'invalid_field' in str(exc_info.value)

    def test_function_name_is_not_validated_as_field(self, schema_registry):
        """Function name itself should not be validated as a field."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # Even if function name looks like invalid field, it shouldn't be validated
        # unknown_function('literal_value')
        expr = FunctionCall(
            name='unknown_function',
            args=(Literal(value='test'),),
            kwargs=()
        )

        # Should not raise - function validation is separate
        validator.validate_expression(expr)


class TestFieldValidatorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_validator_caches_valid_fields(self, schema_registry):
        """Validator should cache valid fields for performance."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # First call populates cache
        fields1 = validator._get_valid_fields()

        # Second call should return same object (cached)
        fields2 = validator._get_valid_fields()

        assert fields1 is fields2

    def test_validator_without_entity_type_only_allows_common_fields(self, schema_registry):
        """Validator without entity_type should only validate common fields."""
        validator = FieldValidator(schema_registry, entity_type=None)

        valid_fields = validator._get_valid_fields()

        # Should only have common fields
        assert valid_fields == COMMON_FIELDS

    def test_literal_values_are_not_validated(self, schema_registry):
        """Literal values should not be validated as fields."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # status = 'some_random_string_that_looks_like_field'
        expr = Comparison(
            field=Field(name='status'),
            op=Op.EQ,
            value=Literal(value='some_random_string_that_looks_like_field')
        )

        # Should not raise - literal values aren't validated as fields
        validator.validate_expression(expr)

    def test_comparison_value_with_nested_expression(self, schema_registry):
        """Comparison value can be a complex expression."""
        validator = FieldValidator(schema_registry, entity_type='task')

        # This is a weird case but syntactically possible:
        # status = some_function(priority)
        expr = Comparison(
            field=Field(name='status'),
            op=Op.EQ,
            value=FunctionCall(
                name='some_func',
                args=(Field(name='priority'),),
                kwargs=()
            )
        )

        # Should validate the nested field reference
        validator.validate_expression(expr)
