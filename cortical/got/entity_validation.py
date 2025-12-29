"""
Schema-driven entity validation for GoT (Graph of Thought) system.

This module provides comprehensive validation for:
1. ID format validation for ALL entity types
2. Relationship rules (which entity types can connect via which edge types)
3. Self-reference validation (preventing A→A edges)

This is the single source of truth for entity validation rules.
All validation functions are designed to raise ValueError with clear messages.

Usage:
    from cortical.got.entity_validation import (
        validate_entity_id,
        validate_edge_relationship,
        EntityIdValidator,
        RelationshipRules,
    )

    # Validate an entity ID
    validate_entity_id("T-20251228-093045-a1b2c3d4", "task")

    # Validate an edge relationship
    validate_edge_relationship("T-...", "S-...", "CONTAINS")
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Optional, Set, Tuple

from .types import VALID_EDGE_TYPES, VALID_ENTITY_TYPES


# =============================================================================
# ID FORMAT PATTERNS
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
# RELATIONSHIP RULES
# =============================================================================
# These rules define which entity types can be connected via which edge types.
# Format: edge_type -> set of (source_type, target_type) tuples

RELATIONSHIP_RULES: Dict[str, FrozenSet[Tuple[str, str]]] = {
    # CONTAINS: Hierarchical containment
    "CONTAINS": frozenset({
        ("sprint", "task"),      # Sprint contains tasks
        ("epic", "sprint"),      # Epic contains sprints
        ("team", "persona_profile"),  # Team contains profiles
    }),

    # PART_OF: Component membership (reverse of CONTAINS)
    "PART_OF": frozenset({
        ("task", "sprint"),      # Task is part of sprint
        ("sprint", "epic"),      # Sprint is part of epic
        ("persona_profile", "team"),  # Profile is part of team
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
}

# Edge types that allow self-references (A→A)
SELF_REFERENCE_ALLOWED: FrozenSet[str] = frozenset({
    # Generally, self-references don't make sense and should be prevented
    # Add edge types here if self-references are ever needed
})


# =============================================================================
# VALIDATION FUNCTIONS
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
        ("T-", "task"),
        ("D-", "decision"),
        ("E-", "edge"),
        ("S-", "sprint"),
        ("H-", "handoff"),
        ("G-", "goal"),
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
# VALIDATOR CLASS
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
