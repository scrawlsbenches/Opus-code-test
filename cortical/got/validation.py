"""
JSON Schema Validation for GoT Entity Files.

Provides simple validation functions for entity structure without external dependencies.
Uses built-in Python only - no jsonschema library.
"""

from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .types import VALID_ENTITY_TYPES, VALID_EDGE_TYPES


def validate_entity(data: dict, strict: bool = False) -> Tuple[bool, str]:
    """
    Validate entity data structure.

    Performs basic structural validation of GoT entity data without external libraries.

    Args:
        data: Entity data dictionary to validate
        strict: If True, perform strict validation (requires all optional fields)

    Returns:
        Tuple of (is_valid, error_message)
        - (True, "") if valid
        - (False, error_msg) if invalid

    Example:
        >>> data = {"id": "T-123", "entity_type": "task", "created_at": "2025-12-27T...", "checksum": "abc..."}
        >>> is_valid, error = validate_entity(data)
        >>> if not is_valid:
        ...     print(f"Validation failed: {error}")
    """
    # Check that data is a dictionary
    if not isinstance(data, dict):
        return False, f"Entity data must be a dictionary, got {type(data).__name__}"

    # Required fields for all entities
    required_fields = ['id', 'entity_type', 'created_at']

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate entity_type
    entity_type = data['entity_type']
    if entity_type not in VALID_ENTITY_TYPES:
        return False, f"Invalid entity_type: {entity_type}. Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"

    # Validate id is non-empty string
    if not isinstance(data['id'], str) or not data['id']:
        return False, "Field 'id' must be a non-empty string"

    # Validate created_at is ISO datetime string
    is_valid_date, date_error = _validate_iso_datetime(data['created_at'])
    if not is_valid_date:
        return False, f"Invalid created_at timestamp: {date_error}"

    # Validate modified_at if present
    if 'modified_at' in data:
        is_valid_date, date_error = _validate_iso_datetime(data['modified_at'])
        if not is_valid_date:
            return False, f"Invalid modified_at timestamp: {date_error}"

    # Validate version if present
    if 'version' in data:
        if not isinstance(data['version'], int) or data['version'] < 1:
            return False, "Field 'version' must be a positive integer"

    # Entity-specific validation
    is_valid, error = _validate_entity_specific(data, entity_type, strict)
    if not is_valid:
        return False, error

    return True, ""


def _validate_iso_datetime(timestamp: str) -> Tuple[bool, str]:
    """
    Validate that a string is a valid ISO 8601 datetime.

    Args:
        timestamp: String to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(timestamp, str):
        return False, f"Timestamp must be a string, got {type(timestamp).__name__}"

    # Try parsing with fromisoformat (Python 3.7+)
    try:
        # Handle 'Z' suffix (Python 3.11+ supports it natively)
        timestamp_normalized = timestamp.replace('Z', '+00:00')
        datetime.fromisoformat(timestamp_normalized)
        return True, ""
    except ValueError as e:
        return False, f"Not a valid ISO 8601 datetime: {e}"


def _validate_entity_specific(data: dict, entity_type: str, strict: bool) -> Tuple[bool, str]:
    """
    Validate entity-specific required fields.

    Args:
        data: Entity data dictionary
        entity_type: Type of entity
        strict: If True, require all fields

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Task-specific validation
    if entity_type == 'task':
        required = ['title', 'status', 'priority']
        for field in required:
            if field not in data:
                return False, f"Task missing required field: {field}"

        # Validate status
        valid_statuses = {'pending', 'in_progress', 'completed', 'blocked'}
        if data['status'] not in valid_statuses:
            return False, f"Invalid task status: {data['status']}. Must be one of: {', '.join(sorted(valid_statuses))}"

        # Validate priority
        valid_priorities = {'low', 'medium', 'high', 'critical'}
        if data['priority'] not in valid_priorities:
            return False, f"Invalid task priority: {data['priority']}. Must be one of: {', '.join(sorted(valid_priorities))}"

    # Decision-specific validation
    elif entity_type == 'decision':
        required = ['title', 'rationale']
        for field in required:
            if field not in data:
                return False, f"Decision missing required field: {field}"

    # Edge-specific validation
    elif entity_type == 'edge':
        required = ['source_id', 'target_id', 'edge_type']
        for field in required:
            if field not in data:
                return False, f"Edge missing required field: {field}"

        # Validate edge_type
        if data['edge_type'] not in VALID_EDGE_TYPES:
            return False, f"Invalid edge_type: {data['edge_type']}. Must be one of: {', '.join(sorted(VALID_EDGE_TYPES))}"

        # Validate weight if present
        if 'weight' in data:
            if not isinstance(data['weight'], (int, float)) or data['weight'] < 0:
                return False, "Edge weight must be a non-negative number"

    # Sprint-specific validation
    elif entity_type == 'sprint':
        required = ['title', 'status']
        for field in required:
            if field not in data:
                return False, f"Sprint missing required field: {field}"

        # Validate status
        valid_statuses = {'available', 'in_progress', 'completed', 'on_hold'}
        if data['status'] not in valid_statuses:
            return False, f"Invalid sprint status: {data['status']}. Must be one of: {', '.join(sorted(valid_statuses))}"

    # Epic-specific validation
    elif entity_type == 'epic':
        required = ['title', 'status']
        for field in required:
            if field not in data:
                return False, f"Epic missing required field: {field}"

        # Validate status
        valid_statuses = {'active', 'completed', 'on_hold', 'archived'}
        if data['status'] not in valid_statuses:
            return False, f"Invalid epic status: {data['status']}. Must be one of: {', '.join(sorted(valid_statuses))}"

    # Handoff-specific validation
    elif entity_type == 'handoff':
        required = ['source_agent', 'target_agent', 'task_id', 'status']
        for field in required:
            if field not in data:
                return False, f"Handoff missing required field: {field}"

        # Validate status
        valid_statuses = {'initiated', 'accepted', 'completed', 'rejected'}
        if data['status'] not in valid_statuses:
            return False, f"Invalid handoff status: {data['status']}. Must be one of: {', '.join(sorted(valid_statuses))}"

    # Document-specific validation
    elif entity_type == 'document':
        required = ['path', 'doc_type']
        for field in required:
            if field not in data:
                return False, f"Document missing required field: {field}"

    # ClaudeMd layer-specific validation
    elif entity_type == 'claudemd_layer':
        required = ['layer_type', 'section_id', 'title', 'content']
        for field in required:
            if field not in data:
                return False, f"ClaudeMd layer missing required field: {field}"

    return True, ""


def validate_entity_file(file_data: dict, strict: bool = False) -> Tuple[bool, str]:
    """
    Validate a complete entity file (with wrapper).

    GoT entity files have a wrapper structure:
    {
        "data": { ... entity fields ... },
        "checksum": "..."
    }

    Args:
        file_data: Complete file data dictionary
        strict: If True, perform strict validation

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check wrapper structure
    if 'data' not in file_data:
        return False, "Entity file missing 'data' wrapper"

    if 'checksum' not in file_data:
        return False, "Entity file missing 'checksum' field"

    # Validate checksum is a string
    if not isinstance(file_data['checksum'], str) or not file_data['checksum']:
        return False, "Field 'checksum' must be a non-empty string"

    # Validate the entity data
    return validate_entity(file_data['data'], strict=strict)


def validate_checksum(data: dict, expected_checksum: str) -> Tuple[bool, str]:
    """
    Validate that entity checksum matches expected value.

    Args:
        data: Entity data dictionary
        expected_checksum: Expected checksum value

    Returns:
        Tuple of (is_valid, error_message)

    Note:
        This function does NOT compute the checksum (would create circular dependency).
        It only validates the format. Use cortical.utils.checksums.compute_checksum
        to compute checksums.
    """
    if not isinstance(expected_checksum, str):
        return False, f"Checksum must be a string, got {type(expected_checksum).__name__}"

    if not expected_checksum:
        return False, "Checksum cannot be empty"

    # Checksum should be hex string (SHA256 truncated to 16 chars)
    if not all(c in '0123456789abcdef' for c in expected_checksum.lower()):
        return False, "Checksum must be a hexadecimal string"

    return True, ""
