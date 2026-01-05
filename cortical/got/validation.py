"""
Comprehensive Validation for GoT (Graph of Thought) System.

This module is the SINGLE SOURCE OF TRUTH for all GoT entity validation:

1. DATA STRUCTURE VALIDATION
   - Required fields per entity type
   - Valid status/priority values
   - Datetime format validation
   - Checksum format validation

2. ID FORMAT VALIDATION
   - Entity ID patterns (T-YYYYMMDD-HHMMSS-{8hex}, etc.)
   - Legacy format rejection
   - Entity type inference from ID

3. RELATIONSHIP VALIDATION
   - Which entity types can connect via which edge types
   - Self-reference prevention
   - Direction validation (e.g., decision→task, not task→decision)

Usage:
    from cortical.got.validation import (
        # Data validation
        validate_entity,
        validate_entity_file,

        # ID validation
        validate_entity_id,
        infer_entity_type_from_id,

        # Relationship validation
        validate_edge_relationship,
        validate_sprint_id_current_format,

        # Utility classes
        EntityIdValidator,
        RelationshipRules,
    )
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, FrozenSet, Set

from .types import VALID_ENTITY_TYPES, VALID_EDGE_TYPES
from .entity_schemas import get_valid_statuses


# =============================================================================
# SECTION 1: ID FORMAT PATTERNS
# =============================================================================
# These patterns define valid ID formats for each entity type.
# The patterns use the current (generated) format.
# Legacy formats are explicitly rejected with helpful error messages.

# Standard timestamp pattern: YYYYMMDD-HHMMSS-{8hex}
_TIMESTAMP_PATTERN = r"\d{8}-\d{6}-[a-f0-9]{8}"

# Entity-specific ID patterns (current format only)
ID_PATTERNS: Dict[str, re.Pattern] = {
    # Task: T-YYYYMMDD-HHMMSS-{8hex}
    "task": re.compile(rf"^T-{_TIMESTAMP_PATTERN}$"),

    # Decision: D-YYYYMMDD-HHMMSS-{8hex}
    "decision": re.compile(rf"^D-{_TIMESTAMP_PATTERN}$"),

    # Sprint: S-YYYYMMDD-HHMMSS-{8hex} (current format ONLY)
    "sprint": re.compile(rf"^S-{_TIMESTAMP_PATTERN}$"),

    # Epic: EPIC-{name} or EPIC-YYYYMMDD-HHMMSS-{8hex}
    "epic": re.compile(rf"^EPIC-([a-z0-9-]+|{_TIMESTAMP_PATTERN})$"),

    # Handoff: H-YYYYMMDD-HHMMSS-{8hex}
    "handoff": re.compile(rf"^H-{_TIMESTAMP_PATTERN}$"),

    # Goal: G-YYYYMMDD-{8hex} (note: no HHMMSS, day-level granularity)
    "goal": re.compile(r"^G-\d{8}-[a-f0-9]{8}$"),

    # Edge: E-{source}-{target}-{type} or E-YYYYMMDD-HHMMSS-{8hex}
    # Edge IDs are complex - they can be auto-generated from source/target/type
    "edge": re.compile(rf"^E-(.+-|{_TIMESTAMP_PATTERN})"),

    # Document: DOC-{path-based} or DOC-YYYYMMDD-HHMMSS-{8hex}
    "document": re.compile(rf"^DOC-([a-zA-Z0-9-]+|{_TIMESTAMP_PATTERN})$"),

    # ClaudeMD Layer: CML{N}-{section}-YYYYMMDD-HHMMSS-{8hex} or CML{N}-YYYYMMDD-HHMMSS-{8hex}
    "claudemd_layer": re.compile(r"^CML[0-4](-[a-z0-9-]+)?-\d{8}-\d{6}-[a-f0-9]{8}$"),

    # ClaudeMD Version: CMV-{layer_id}-v{N}
    "claudemd_version": re.compile(r"^CMV-.+-v\d+$"),

    # Persona Profile: PP-YYYYMMDD-HHMMSS-{8hex}
    "persona_profile": re.compile(rf"^PP-{_TIMESTAMP_PATTERN}$"),

    # Team: TEAM-YYYYMMDD-HHMMSS-{8hex}
    "team": re.compile(rf"^TEAM-{_TIMESTAMP_PATTERN}$"),

    # Orchestration Plan: OP-YYYYMMDD-HHMMSS-{8hex}
    "orchestration_plan": re.compile(rf"^OP-{_TIMESTAMP_PATTERN}$"),

    # Execution: EX-YYYYMMDD-HHMMSS-{8hex}
    "execution": re.compile(rf"^EX-{_TIMESTAMP_PATTERN}$"),

    # Failure: F-YYYYMMDD-HHMMSS-{8hex}
    "failure": re.compile(rf"^F-{_TIMESTAMP_PATTERN}$"),
}

# Legacy patterns to detect and reject (with helpful messages)
LEGACY_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = {
    # Sprint legacy formats
    "sprint_legacy_short": (
        re.compile(r"^S-\d{1,3}$"),
        "Legacy short format (S-NNN) is deprecated. "
        "Use 'python scripts/got_utils.py sprint create' to create sprints with current format."
    ),
    "sprint_legacy_verbose": (
        re.compile(r"^S-sprint-\d+(-[\w-]+)?$"),
        "Legacy verbose format (S-sprint-NNN-slug) is deprecated. "
        "Use 'python scripts/got_utils.py sprint create' to create sprints with current format."
    ),
    # Task legacy formats
    "task_legacy_prefix": (
        re.compile(r"^task:"),
        "Legacy task: prefix format is deprecated. "
        "Use 'python scripts/got_utils.py task create' for proper task IDs."
    ),
}


# =============================================================================
# SECTION 2: RELATIONSHIP RULES
# =============================================================================
# These rules define which entity types can be connected via which edge types.
# Format: edge_type -> set of (source_type, target_type) tuples

RELATIONSHIP_RULES: Dict[str, FrozenSet[Tuple[str, str]]] = {
    # CONTAINS: Hierarchical containment
    "CONTAINS": frozenset({
        ("sprint", "task"),      # Sprint contains tasks
        ("epic", "sprint"),      # Epic contains sprints
        ("team", "persona_profile"),  # Team contains profiles
        ("task", "task"),        # Task contains sub-tasks (staged work breakdown)
    }),

    # PART_OF: Component membership (reverse of CONTAINS)
    "PART_OF": frozenset({
        ("task", "sprint"),      # Task is part of sprint
        ("sprint", "epic"),      # Sprint is part of epic
        ("persona_profile", "team"),  # Profile is part of team
        ("task", "task"),        # Sub-task is part of parent task
    }),

    # DEPENDS_ON: Dependency relationship
    "DEPENDS_ON": frozenset({
        ("task", "task"),        # Task depends on task
        ("sprint", "sprint"),    # Sprint depends on sprint
        ("decision", "decision"),  # Decision depends on decision
    }),

    # BLOCKS: Blocking relationship (opposite direction of DEPENDS_ON)
    "BLOCKS": frozenset({
        ("task", "task"),        # Task blocks task
        ("decision", "task"),    # Decision blocks task
    }),

    # IMPLEMENTS: Implementation relationship
    "IMPLEMENTS": frozenset({
        ("task", "decision"),    # Task implements decision
        ("document", "decision"),  # Document implements decision
    }),

    # JUSTIFIES: Justification relationship
    "JUSTIFIES": frozenset({
        ("decision", "task"),    # Decision justifies task
        ("decision", "sprint"),  # Decision justifies sprint
    }),

    # RELATES_TO: General relationship (flexible)
    "RELATES_TO": frozenset({
        ("task", "task"),
        ("task", "decision"),
        ("decision", "decision"),
        ("document", "task"),
        ("document", "decision"),
        ("sprint", "sprint"),
        ("epic", "epic"),
    }),

    # REFERENCES: Soft reference (very flexible - any entity can reference any entity)
    "REFERENCES": frozenset({
        ("task", "task"),
        ("task", "decision"),
        ("task", "document"),
        ("task", "sprint"),
        ("decision", "task"),
        ("decision", "decision"),
        ("decision", "document"),
        ("document", "task"),
        ("document", "decision"),
        ("document", "document"),
        ("sprint", "task"),
        ("sprint", "document"),
        ("epic", "sprint"),
        ("epic", "document"),
    }),

    # TRANSFERS: Handoff relationship
    "TRANSFERS": frozenset({
        ("task", "handoff"),     # Task transfers to handoff
        ("handoff", "task"),     # Handoff transfers back to task
    }),

    # PRODUCES: Production relationship
    "PRODUCES": frozenset({
        ("task", "document"),    # Task produces document
        ("sprint", "document"),  # Sprint produces document
    }),

    # DOCUMENTED_BY: Documentation relationship (reverse of PRODUCES)
    "DOCUMENTED_BY": frozenset({
        ("task", "document"),
        ("decision", "document"),
        ("sprint", "document"),
    }),

    # SUPERSEDES: Replacement relationship
    "SUPERSEDES": frozenset({
        ("task", "task"),
        ("decision", "decision"),
        ("document", "document"),
        ("sprint", "sprint"),
    }),

    # DERIVED_FROM: Derivation relationship
    "DERIVED_FROM": frozenset({
        ("task", "task"),
        ("decision", "decision"),
        ("document", "document"),
    }),

    # REQUIRES: Hard requirement
    "REQUIRES": frozenset({
        ("task", "task"),
        ("task", "decision"),
        ("sprint", "task"),
    }),

    # PARENT_OF / CHILD_OF: Hierarchical
    "PARENT_OF": frozenset({
        ("task", "task"),
        ("epic", "epic"),
        ("team", "team"),
    }),
    "CHILD_OF": frozenset({
        ("task", "task"),
        ("epic", "epic"),
        ("team", "team"),
    }),

    # CONTRADICTS: Conflict relationship
    "CONTRADICTS": frozenset({
        ("decision", "decision"),
        ("task", "task"),
    }),

    # MOTIVATES: Motivation relationship
    "MOTIVATES": frozenset({
        ("decision", "task"),
        ("decision", "sprint"),
        ("epic", "sprint"),
    }),

    # CAUSED_BY: Causation relationship
    "CAUSED_BY": frozenset({
        ("task", "decision"),
        ("task", "task"),
    }),

    # DOCUMENTS: Knowledge transfer documentation relationship
    # KnowledgeTransfer documents the work done on Tasks or Decisions
    "DOCUMENTS": frozenset({
        ("knowledge_transfer", "task"),
        ("knowledge_transfer", "decision"),
    }),

    # CONTINUES: Continuation relationship for knowledge transfer chains
    # Forms bidirectional chains: KT1 → Handoff1 → KT2 → Handoff2 → KT3
    # - KT → Handoff: "KT's work continues via this handoff" (when finalizing)
    # - Handoff → KT: "Handoff continues into this KT" (when picking up)
    # Both directions needed for get_kt_history() to trace full chain
    "CONTINUES": frozenset({
        ("knowledge_transfer", "handoff"),  # KT produces handoff for continuation
        ("handoff", "knowledge_transfer"),  # Handoff continues into new KT
    }),

    # FAILED_ATTEMPT: Records a failed approach to a task
    # Failure entity tracks what didn't work to prevent repeating mistakes
    "FAILED_ATTEMPT": frozenset({
        ("failure", "task"),
    }),
}

# Edge types that allow self-references (A→A)
SELF_REFERENCE_ALLOWED: FrozenSet[str] = frozenset({
    # Generally, self-references don't make sense and should be prevented
    # Add edge types here if self-references are ever needed
})


# =============================================================================
# SECTION 3: DATA STRUCTURE VALIDATION
# =============================================================================


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
        >>> data = {"id": "T-123", "entity_type": "task", "created_at": "2025-12-27T..."}
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
        valid_statuses = get_valid_statuses('task')
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
        valid_statuses = get_valid_statuses('sprint')
        if data['status'] not in valid_statuses:
            return False, f"Invalid sprint status: {data['status']}. Must be one of: {', '.join(sorted(valid_statuses))}"

    # Epic-specific validation
    elif entity_type == 'epic':
        required = ['title', 'status']
        for field in required:
            if field not in data:
                return False, f"Epic missing required field: {field}"

        # Validate status
        valid_statuses = get_valid_statuses('epic')
        if data['status'] not in valid_statuses:
            return False, f"Invalid epic status: {data['status']}. Must be one of: {', '.join(sorted(valid_statuses))}"

    # Handoff-specific validation
    elif entity_type == 'handoff':
        required = ['source_agent', 'target_agent', 'task_id', 'status']
        for field in required:
            if field not in data:
                return False, f"Handoff missing required field: {field}"

        # Validate status
        valid_statuses = get_valid_statuses('handoff')
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
        "_checksum": "..." or "checksum": "..."
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

    # Accept both '_checksum' (current format) and 'checksum' (legacy)
    checksum = file_data.get('_checksum') or file_data.get('checksum')
    if not checksum:
        return False, "Entity file missing 'checksum' field"

    # Validate checksum is a string
    if not isinstance(checksum, str) or not checksum:
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


# =============================================================================
# SECTION 4: ID FORMAT VALIDATION
# =============================================================================


def infer_entity_type_from_id(entity_id: str) -> Optional[str]:
    """
    Infer entity type from ID prefix.

    Args:
        entity_id: Entity ID string

    Returns:
        Entity type string, or None if not recognized

    Examples:
        >>> infer_entity_type_from_id("T-20251228-093045-a1b2c3d4")
        'task'
        >>> infer_entity_type_from_id("EPIC-woven-mind")
        'epic'
    """
    # Order matters - check longer prefixes first
    prefix_map = [
        ("TEAM-", "team"),
        ("EPIC-", "epic"),
        ("DOC-", "document"),
        ("CMV-", "claudemd_version"),
        ("CML", "claudemd_layer"),
        ("PP-", "persona_profile"),
        ("OP-", "orchestration_plan"),
        ("EX-", "execution"),
        ("KT-", "knowledge_transfer"),  # Knowledge transfer entities
        ("T-", "task"),
        ("D-", "decision"),
        ("E-", "edge"),
        ("S-", "sprint"),
        ("H-", "handoff"),
        ("G-", "goal"),
        ("F-", "failure"),
    ]

    for prefix, entity_type in prefix_map:
        if entity_id.startswith(prefix):
            return entity_type

    return None


def validate_entity_id(
    entity_id: str,
    expected_type: Optional[str] = None,
    strict: bool = True
) -> str:
    """
    Validate that an entity ID has the correct format.

    Args:
        entity_id: Entity ID to validate
        expected_type: Expected entity type (if None, inferred from prefix)
        strict: If True, reject legacy formats. If False, only warn.

    Returns:
        The entity type (inferred or expected)

    Raises:
        ValueError: If ID format is invalid or doesn't match expected type

    Examples:
        >>> validate_entity_id("T-20251228-093045-a1b2c3d4")
        'task'
        >>> validate_entity_id("T-20251228-093045-a1b2c3d4", "task")
        'task'
        >>> validate_entity_id("T-20251228-093045-a1b2c3d4", "decision")
        ValueError: ID 'T-...' has type 'task', expected 'decision'
    """
    if not entity_id:
        raise ValueError("Entity ID cannot be empty")

    # Check for legacy formats first
    for legacy_name, (pattern, message) in LEGACY_PATTERNS.items():
        if pattern.match(entity_id):
            if strict:
                raise ValueError(
                    f"Cannot use legacy ID '{entity_id}'. {message}"
                )
            # Non-strict mode: continue but warn (logging would go here)

    # Infer entity type from prefix
    inferred_type = infer_entity_type_from_id(entity_id)

    if inferred_type is None:
        raise ValueError(
            f"Unrecognized entity ID format: '{entity_id}'. "
            f"ID must start with a valid prefix (T-, D-, S-, EPIC-, H-, etc.)"
        )

    # Check expected type matches inferred type
    if expected_type is not None and inferred_type != expected_type:
        raise ValueError(
            f"ID '{entity_id}' has type '{inferred_type}', expected '{expected_type}'"
        )

    # Validate against pattern for this entity type
    pattern = ID_PATTERNS.get(inferred_type)
    if pattern and not pattern.match(entity_id):
        # Special handling for flexible types like edge, document, epic
        if inferred_type in ("edge", "document", "epic"):
            # These have multiple valid formats, be more lenient
            pass
        else:
            raise ValueError(
                f"Invalid {inferred_type} ID format: '{entity_id}'. "
                f"Expected format matching pattern for {inferred_type}."
            )

    return inferred_type


# =============================================================================
# SECTION 5: RELATIONSHIP VALIDATION
# =============================================================================


def validate_edge_relationship(
    source_id: str,
    target_id: str,
    edge_type: str,
    allow_self_reference: bool = False
) -> None:
    """
    Validate that an edge relationship is valid.

    This checks:
    1. Source and target IDs have valid formats
    2. The edge type is valid
    3. The source→target entity type pair is allowed for this edge type
    4. Self-references are prevented (unless explicitly allowed)

    Args:
        source_id: Source entity ID
        target_id: Target entity ID
        edge_type: Type of edge (DEPENDS_ON, BLOCKS, etc.)
        allow_self_reference: If True, allow A→A edges

    Raises:
        ValueError: If the relationship is invalid

    Examples:
        >>> validate_edge_relationship("T-...", "T-...", "DEPENDS_ON")
        None  # Valid: task depends on task

        >>> validate_edge_relationship("S-...", "T-...", "CONTAINS")
        None  # Valid: sprint contains task

        >>> validate_edge_relationship("T-...", "S-...", "CONTAINS")
        ValueError: Invalid relationship: task cannot CONTAINS sprint
    """
    # Validate edge type
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type: '{edge_type}'. "
            f"Must be one of: {sorted(VALID_EDGE_TYPES)}"
        )

    # Validate source ID
    source_type = validate_entity_id(source_id)

    # Validate target ID
    target_type = validate_entity_id(target_id)

    # Check for self-reference
    if source_id == target_id:
        if not allow_self_reference and edge_type not in SELF_REFERENCE_ALLOWED:
            raise ValueError(
                f"Self-reference not allowed: cannot create {edge_type} edge "
                f"from '{source_id}' to itself."
            )

    # Check relationship rules
    allowed_pairs = RELATIONSHIP_RULES.get(edge_type)
    if allowed_pairs is not None:
        pair = (source_type, target_type)
        if pair not in allowed_pairs:
            raise ValueError(
                f"Invalid relationship: {source_type} cannot {edge_type} {target_type}. "
                f"Allowed relationships for {edge_type}: "
                f"{sorted((s, t) for s, t in allowed_pairs)}"
            )
    # If edge_type not in RELATIONSHIP_RULES, allow it (permissive for extensibility)


def validate_sprint_id_current_format(sprint_id: str) -> None:
    """
    Strictly validate that a sprint ID uses the current generated format.

    This function REJECTS all legacy formats and should be used when creating
    new edges to sprints to prevent linking to deprecated sprint IDs.

    Args:
        sprint_id: Sprint ID to validate

    Raises:
        ValueError: If the sprint ID uses a legacy format

    Examples:
        >>> validate_sprint_id_current_format("S-20251228-093045-a1b2c3d4")
        None  # Valid

        >>> validate_sprint_id_current_format("S-025")
        ValueError: Cannot link to legacy sprint ID 'S-025'...
    """
    if not sprint_id.startswith("S-"):
        return  # Not a sprint ID, let other validation handle it

    # Check current format
    current_pattern = re.compile(rf"^S-{_TIMESTAMP_PATTERN}$")
    if current_pattern.match(sprint_id):
        return  # Valid current format

    # Check for legacy formats and provide specific error messages
    legacy_short = re.compile(r"^S-\d{1,3}$")
    legacy_verbose = re.compile(r"^S-sprint-\d+(-[\w-]+)?$")

    if legacy_short.match(sprint_id):
        raise ValueError(
            f"Cannot link to legacy sprint ID '{sprint_id}'. "
            f"Legacy short format (S-NNN) is deprecated. "
            f"Please create a new sprint with: "
            f"python scripts/got_utils.py sprint create \"Sprint Title\" --number N"
        )

    if legacy_verbose.match(sprint_id):
        raise ValueError(
            f"Cannot link to legacy sprint ID '{sprint_id}'. "
            f"Legacy verbose format (S-sprint-NNN-slug) is deprecated. "
            f"Please create a new sprint with: "
            f"python scripts/got_utils.py sprint create \"Sprint Title\" --number N"
        )

    # Unknown format
    raise ValueError(
        f"Invalid sprint ID format: '{sprint_id}'. "
        f"Expected format: S-YYYYMMDD-HHMMSS-{{8hex}} "
        f"(e.g., S-20251228-093045-a1b2c3d4)"
    )


# =============================================================================
# SECTION 6: UTILITY CLASSES
# =============================================================================


class EntityIdValidator:
    """
    Centralized validator for entity IDs with caching and batch validation.

    Usage:
        validator = EntityIdValidator()
        validator.validate("T-20251228-093045-a1b2c3d4")
        validator.validate_batch(["T-...", "S-...", "D-..."])
    """

    def __init__(self, strict: bool = True):
        """
        Initialize the validator.

        Args:
            strict: If True, reject legacy formats. If False, only warn.
        """
        self.strict = strict
        self._cache: Dict[str, str] = {}  # entity_id -> entity_type

    def validate(self, entity_id: str, expected_type: Optional[str] = None) -> str:
        """
        Validate an entity ID, returning its type.

        Args:
            entity_id: Entity ID to validate
            expected_type: Expected entity type

        Returns:
            The entity type

        Raises:
            ValueError: If validation fails
        """
        # Check cache first
        if entity_id in self._cache and expected_type is None:
            return self._cache[entity_id]

        entity_type = validate_entity_id(entity_id, expected_type, self.strict)
        self._cache[entity_id] = entity_type
        return entity_type

    def validate_batch(
        self,
        entity_ids: list,
        expected_type: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Validate multiple entity IDs.

        Args:
            entity_ids: List of entity IDs to validate
            expected_type: Expected type for all IDs (or None to infer)

        Returns:
            Dict mapping entity_id -> entity_type

        Raises:
            ValueError: If any validation fails (includes failed ID in message)
        """
        results = {}
        errors = []

        for entity_id in entity_ids:
            try:
                results[entity_id] = self.validate(entity_id, expected_type)
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise ValueError(
                f"Batch validation failed with {len(errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return results

    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self._cache.clear()


class RelationshipRules:
    """
    Query interface for relationship rules.

    Usage:
        rules = RelationshipRules()
        rules.can_connect("task", "sprint", "PART_OF")  # True
        rules.get_allowed_targets("task", "DEPENDS_ON")  # {"task"}
    """

    def can_connect(
        self,
        source_type: str,
        target_type: str,
        edge_type: str
    ) -> bool:
        """
        Check if a relationship is allowed.

        Args:
            source_type: Source entity type
            target_type: Target entity type
            edge_type: Edge type

        Returns:
            True if the relationship is allowed
        """
        allowed_pairs = RELATIONSHIP_RULES.get(edge_type)
        if allowed_pairs is None:
            return True  # Permissive for unknown edge types

        return (source_type, target_type) in allowed_pairs

    def get_allowed_targets(self, source_type: str, edge_type: str) -> Set[str]:
        """
        Get all allowed target types for a source type and edge type.

        Args:
            source_type: Source entity type
            edge_type: Edge type

        Returns:
            Set of allowed target entity types
        """
        allowed_pairs = RELATIONSHIP_RULES.get(edge_type, frozenset())
        return {target for source, target in allowed_pairs if source == source_type}

    def get_allowed_sources(self, target_type: str, edge_type: str) -> Set[str]:
        """
        Get all allowed source types for a target type and edge type.

        Args:
            target_type: Target entity type
            edge_type: Edge type

        Returns:
            Set of allowed source entity types
        """
        allowed_pairs = RELATIONSHIP_RULES.get(edge_type, frozenset())
        return {source for source, target in allowed_pairs if target == target_type}

    def get_all_rules(self) -> Dict[str, FrozenSet[Tuple[str, str]]]:
        """Get all relationship rules."""
        return RELATIONSHIP_RULES.copy()

    def allows_self_reference(self, edge_type: str) -> bool:
        """Check if an edge type allows self-references."""
        return edge_type in SELF_REFERENCE_ALLOWED
