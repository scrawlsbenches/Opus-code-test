"""
Integration tests for field validation with the query parser.

Tests the complete workflow: parse -> validate -> execute
"""

import pytest

from cortical.got.expression import parse, validate
from cortical.got.expression.errors import QueryValidationError


class TestValidateFunction:
    """Test the public validate() function."""

    def test_validate_valid_query(self):
        """Valid query should pass validation."""
        query = parse("status = 'pending' AND priority = 'high'")

        # Should not raise
        validate(query, entity_type='task')

    def test_validate_invalid_field_raises_error(self):
        """Invalid field should raise QueryValidationError."""
        query = parse("invalid_field = 'value'")

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        assert 'invalid_field' in str(exc_info.value)

    def test_validate_with_suggestions(self):
        """Validation error should include helpful suggestions."""
        query = parse("stat = 'pending'")  # Typo for 'status'

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        error = exc_info.value
        # Should suggest 'status'
        assert 'status' in error.suggestions

    def test_validate_without_entity_type(self):
        """Validation without entity_type should only allow common fields."""
        # Common field - should pass
        query1 = parse("status = 'pending'")
        validate(query1, entity_type=None)

        # Entity-specific field - should fail
        query2 = parse("priority = 'high'")
        with pytest.raises(QueryValidationError):
            validate(query2, entity_type=None)

    def test_validate_empty_query(self):
        """Empty query should not raise."""
        # This would be a query with no WHERE clause
        # For now, we just test that None doesn't crash
        from cortical.got.expression.ast import Query

        empty_query = Query(expression=None)
        validate(empty_query, entity_type='task')  # Should not raise


class TestEndToEndValidation:
    """Test complete parse -> validate -> error workflow."""

    def test_complex_valid_query(self):
        """Complex query with valid fields should validate."""
        query = parse(
            "status = 'pending' AND (priority = 'high' OR priority = 'critical') "
            "AND NOT (status = 'blocked')"
        )

        # Should not raise
        validate(query, entity_type='task')

    def test_nested_expression_with_invalid_field(self):
        """Nested expression with invalid field should be caught."""
        query = parse(
            "status = 'pending' AND (invalid = 'value' OR priority = 'high')"
        )

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        assert 'invalid' in str(exc_info.value)

    def test_deeply_nested_invalid_field(self):
        """Invalid field deep in expression tree should be caught."""
        query = parse(
            "((status = 'pending' AND priority = 'high') OR "
            "(status = 'blocked' AND bad_field = 'value'))"
        )

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        assert 'bad_field' in str(exc_info.value)


class TestValidationErrorMessages:
    """Test the quality of validation error messages."""

    def test_error_message_includes_field_name(self):
        """Error message should mention the invalid field."""
        query = parse("nonexistent = 'value'")

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        error_msg = str(exc_info.value)
        assert 'nonexistent' in error_msg
        assert 'Unknown field' in error_msg

    def test_error_message_includes_suggestions(self):
        """Error message should include suggestions."""
        query = parse("priorit = 'high'")  # Close to 'priority'

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        error_msg = str(exc_info.value)
        assert 'Did you mean' in error_msg
        assert 'priority' in error_msg

    def test_error_message_includes_valid_fields(self):
        """Error message should list valid fields."""
        query = parse("invalid = 'value'")

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        error_msg = str(exc_info.value)
        assert 'Valid fields:' in error_msg
        # Should mention some valid fields
        assert 'status' in error_msg or 'priority' in error_msg


class TestDifferentEntityTypes:
    """Test validation with different entity types."""

    def test_task_specific_validation(self):
        """Task-specific fields should validate for task entity type."""
        query = parse("priority = 'high'")

        # Should pass for task
        validate(query, entity_type='task')

        # Should fail for decision
        with pytest.raises(QueryValidationError):
            validate(query, entity_type='decision')

    def test_decision_specific_validation(self):
        """Decision-specific fields should validate for decision entity type."""
        query = parse("rationale LIKE '%performance%'")

        # Should pass for decision
        validate(query, entity_type='decision')

        # Should fail for task
        with pytest.raises(QueryValidationError):
            validate(query, entity_type='task')

    def test_common_fields_valid_for_all_types(self):
        """Common fields should validate for any entity type."""
        query = parse("status = 'pending' AND title LIKE '%test%'")

        # Should pass for all entity types
        validate(query, entity_type='task')
        validate(query, entity_type='decision')
        validate(query, entity_type='knowledge_transfer')
        validate(query, entity_type='sprint')


class TestValidationWithFunctions:
    """Test validation when expressions contain function calls."""

    def test_function_with_valid_field_argument(self):
        """Function with valid field argument should validate."""
        # Assuming we have a function that takes field references
        from cortical.got.expression.ast import FunctionCall, Field, Query

        func_call = FunctionCall(
            name='some_func',
            args=(Field(name='status'),),
            kwargs=()
        )

        query = Query(expression=func_call)

        # Should not raise - status is a valid field
        validate(query, entity_type='task')

    def test_function_with_invalid_field_argument(self):
        """Function with invalid field argument should fail validation."""
        from cortical.got.expression.ast import FunctionCall, Field, Query

        func_call = FunctionCall(
            name='some_func',
            args=(Field(name='invalid_field'),),
            kwargs=()
        )

        query = Query(expression=func_call)

        with pytest.raises(QueryValidationError) as exc_info:
            validate(query, entity_type='task')

        assert 'invalid_field' in str(exc_info.value)
