"""
Schema definitions for all GoT entity types.

Defines declarative schemas for validation and migration of:
- Task, Decision, Sprint, Epic, Edge
- Handoff, ClaudeMdLayer, ClaudeMdVersion
- Team, PersonaProfile, Document

All schemas are registered with the global registry on module import.

Usage:
    from cortical.got.entity_schemas import ensure_schemas_registered

    # Schemas are auto-registered on import, but can be explicitly ensured
    ensure_schemas_registered()

    # Then use registry
    from cortical.got.schema import validate_entity, migrate_entity
    result = validate_entity('task', data)
"""

from __future__ import annotations

from typing import Any, Dict

from .schema import (
    BaseSchema,
    Field,
    FieldType,
    register_schema,
    get_registry,
)


# =============================================================================
# Base Entity Fields (shared by all entities)
# =============================================================================

BASE_ENTITY_FIELDS = {
    'id': Field('id', FieldType.STRING, required=True,
                description="Unique entity identifier"),
    'entity_type': Field('entity_type', FieldType.STRING, required=True,
                        description="Type discriminator"),
    'version': Field('version', FieldType.INTEGER, required=False, default=1,
                    description="Optimistic locking version"),
    'created_at': Field('created_at', FieldType.DATETIME, required=False,
                       description="ISO 8601 creation timestamp"),
    'modified_at': Field('modified_at', FieldType.DATETIME, required=False,
                        description="ISO 8601 modification timestamp"),
}


# =============================================================================
# Task Schema
# =============================================================================

class TaskSchema(BaseSchema):
    """
    Schema for Task entities.

    Tasks represent work items with status tracking and priority.
    """
    schema_version = 1
    entity_type = 'task'

    fields = {
        **BASE_ENTITY_FIELDS,
        'title': Field('title', FieldType.STRING, required=True,
                      description="Task title"),
        'status': Field('status', FieldType.ENUM, required=True,
                       choices=['pending', 'in_progress', 'completed', 'blocked'],
                       default='pending',
                       description="Current task status"),
        'priority': Field('priority', FieldType.ENUM, required=False,
                         choices=['low', 'medium', 'high', 'critical'],
                         default='medium',
                         description="Task priority level"),
        'description': Field('description', FieldType.STRING, required=False,
                            default='',
                            description="Detailed task description"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Arbitrary key-value properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# Decision Schema
# =============================================================================

class DecisionSchema(BaseSchema):
    """
    Schema for Decision entities.

    Decisions capture choices with rationale and affected entities.
    """
    schema_version = 1
    entity_type = 'decision'

    fields = {
        **BASE_ENTITY_FIELDS,
        'title': Field('title', FieldType.STRING, required=True,
                      description="Decision title/summary"),
        'rationale': Field('rationale', FieldType.STRING, required=False,
                          default='',
                          description="Why this decision was made"),
        'affects': Field('affects', FieldType.LIST, required=False,
                        default=[],
                        item_type=FieldType.STRING,
                        description="List of affected entity IDs"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
    }


# =============================================================================
# Sprint Schema
# =============================================================================

class SprintSchema(BaseSchema):
    """
    Schema for Sprint entities.

    Sprints are time-boxed work periods with goals and isolation.
    """
    schema_version = 1
    entity_type = 'sprint'

    fields = {
        **BASE_ENTITY_FIELDS,
        'title': Field('title', FieldType.STRING, required=True,
                      description="Sprint title"),
        'status': Field('status', FieldType.ENUM, required=True,
                       choices=['available', 'in_progress', 'completed', 'blocked'],
                       default='available',
                       description="Sprint status"),
        'epic_id': Field('epic_id', FieldType.STRING, required=False,
                        default='',
                        description="Parent epic ID"),
        'number': Field('number', FieldType.INTEGER, required=False,
                       default=0,
                       description="Sprint number"),
        'session_id': Field('session_id', FieldType.STRING, required=False,
                           default='',
                           description="Associated session ID"),
        'isolation': Field('isolation', FieldType.LIST, required=False,
                          default=[],
                          item_type=FieldType.STRING,
                          description="Isolated entity prefixes"),
        'goals': Field('goals', FieldType.LIST, required=False,
                      default=[],
                      description="Sprint goals (list of goal objects)"),
        'notes': Field('notes', FieldType.LIST, required=False,
                      default=[],
                      item_type=FieldType.STRING,
                      description="Sprint notes"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# Epic Schema
# =============================================================================

class EpicSchema(BaseSchema):
    """
    Schema for Epic entities.

    Epics are large initiatives spanning multiple sprints.
    """
    schema_version = 1
    entity_type = 'epic'

    fields = {
        **BASE_ENTITY_FIELDS,
        'title': Field('title', FieldType.STRING, required=True,
                      description="Epic title"),
        'status': Field('status', FieldType.ENUM, required=True,
                       choices=['active', 'completed', 'on_hold'],
                       default='active',
                       description="Epic status"),
        'phase': Field('phase', FieldType.INTEGER, required=False,
                      default=1,
                      description="Current phase number"),
        'phases': Field('phases', FieldType.LIST, required=False,
                       default=[],
                       description="Phase definitions (list of phase objects)"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# Edge Schema
# =============================================================================

def _validate_weight(value: float) -> bool:
    """Validate edge weight is in [0.0, 1.0]."""
    return 0.0 <= value <= 1.0


def _validate_confidence(value: float) -> bool:
    """Validate edge confidence is in [0.0, 1.0]."""
    return 0.0 <= value <= 1.0


class EdgeSchema(BaseSchema):
    """
    Schema for Edge entities.

    Edges represent relationships between entities.
    """
    schema_version = 1
    entity_type = 'edge'

    # Valid edge types
    EDGE_TYPES = [
        'DEPENDS_ON', 'BLOCKS', 'CONTAINS', 'RELATES_TO',
        'REQUIRES', 'IMPLEMENTS', 'SUPERSEDES', 'DERIVED_FROM',
        'PARENT_OF', 'CHILD_OF', 'REFERENCES', 'CONTRADICTS',
    ]

    fields = {
        **BASE_ENTITY_FIELDS,
        'source_id': Field('source_id', FieldType.STRING, required=True,
                          description="Source entity ID"),
        'target_id': Field('target_id', FieldType.STRING, required=True,
                          description="Target entity ID"),
        'edge_type': Field('edge_type', FieldType.STRING, required=True,
                          description="Relationship type"),
        'weight': Field('weight', FieldType.FLOAT, required=False,
                       default=1.0,
                       validator=_validate_weight,
                       description="Edge weight [0.0, 1.0]"),
        'confidence': Field('confidence', FieldType.FLOAT, required=False,
                           default=1.0,
                           validator=_validate_confidence,
                           description="Confidence score [0.0, 1.0]"),
    }


# =============================================================================
# Handoff Schema
# =============================================================================

class HandoffSchema(BaseSchema):
    """
    Schema for Handoff entities.

    Handoffs track agent-to-agent work transfers.
    """
    schema_version = 1
    entity_type = 'handoff'

    fields = {
        **BASE_ENTITY_FIELDS,
        'source_agent': Field('source_agent', FieldType.STRING, required=True,
                             description="Initiating agent ID"),
        'target_agent': Field('target_agent', FieldType.STRING, required=True,
                             description="Receiving agent ID"),
        'task_id': Field('task_id', FieldType.STRING, required=False,
                        default='',
                        description="Associated task ID"),
        'status': Field('status', FieldType.ENUM, required=True,
                       choices=['initiated', 'accepted', 'completed', 'rejected'],
                       default='initiated',
                       description="Handoff status"),
        'instructions': Field('instructions', FieldType.STRING, required=False,
                             default='',
                             description="Instructions for receiving agent"),
        'context': Field('context', FieldType.DICT, required=False,
                        default={},
                        description="Context data for handoff"),
        'result': Field('result', FieldType.DICT, required=False,
                       default={},
                       description="Completion result"),
        'artifacts': Field('artifacts', FieldType.LIST, required=False,
                          default=[],
                          item_type=FieldType.STRING,
                          description="List of artifact paths"),
        'initiated_at': Field('initiated_at', FieldType.DATETIME, required=False,
                             description="When handoff was initiated"),
        'accepted_at': Field('accepted_at', FieldType.DATETIME, required=False,
                            default='',
                            description="When handoff was accepted"),
        'completed_at': Field('completed_at', FieldType.DATETIME, required=False,
                             default='',
                             description="When handoff was completed"),
        'rejected_at': Field('rejected_at', FieldType.DATETIME, required=False,
                            default='',
                            description="When handoff was rejected"),
        'reject_reason': Field('reject_reason', FieldType.STRING, required=False,
                              default='',
                              description="Reason for rejection"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
    }


# =============================================================================
# ClaudeMdLayer Schema
# =============================================================================

class ClaudeMdLayerSchema(BaseSchema):
    """
    Schema for ClaudeMdLayer entities.

    Layers store CLAUDE.md content sections with freshness tracking.
    """
    schema_version = 1
    entity_type = 'claudemd_layer'

    fields = {
        **BASE_ENTITY_FIELDS,
        'layer_type': Field('layer_type', FieldType.ENUM, required=False,
                           choices=['core', 'operational', 'contextual', 'persona', 'ephemeral', ''],
                           default='',
                           description="Layer type classification"),
        'layer_number': Field('layer_number', FieldType.INTEGER, required=False,
                             default=0,
                             description="Layer hierarchy number (0-4)"),
        'section_id': Field('section_id', FieldType.STRING, required=False,
                           default='',
                           description="Section identifier"),
        'title': Field('title', FieldType.STRING, required=False,
                      default='',
                      description="Human-readable title"),
        'content': Field('content', FieldType.STRING, required=False,
                        default='',
                        description="Markdown content"),
        'freshness_status': Field('freshness_status', FieldType.ENUM, required=False,
                                 choices=['fresh', 'stale', 'regenerating'],
                                 default='fresh',
                                 description="Content freshness status"),
        'freshness_decay_days': Field('freshness_decay_days', FieldType.INTEGER, required=False,
                                     default=0,
                                     description="Days until content becomes stale (0 = never)"),
        'last_regenerated': Field('last_regenerated', FieldType.DATETIME, required=False,
                                 default='',
                                 description="When content was last regenerated"),
        'regeneration_trigger': Field('regeneration_trigger', FieldType.STRING, required=False,
                                     default='',
                                     description="What triggered regeneration"),
        'inclusion_rule': Field('inclusion_rule', FieldType.ENUM, required=False,
                               choices=['always', 'context', 'user_pref'],
                               default='always',
                               description="When to include this layer"),
        'context_modules': Field('context_modules', FieldType.LIST, required=False,
                                default=[],
                                item_type=FieldType.STRING,
                                description="Modules for context-based inclusion"),
        'context_branches': Field('context_branches', FieldType.LIST, required=False,
                                 default=[],
                                 item_type=FieldType.STRING,
                                 description="Branch patterns for inclusion"),
        'content_hash': Field('content_hash', FieldType.STRING, required=False,
                             default='',
                             description="SHA256 hash of content (16 chars)"),
        'version_number': Field('version_number', FieldType.INTEGER, required=False,
                               default=1,
                               description="Content version number"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# ClaudeMdVersion Schema
# =============================================================================

class ClaudeMdVersionSchema(BaseSchema):
    """
    Schema for ClaudeMdVersion entities.

    Stores complete CLAUDE.md snapshots.
    """
    schema_version = 1
    entity_type = 'claudemd_version'

    fields = {
        **BASE_ENTITY_FIELDS,
        'content': Field('content', FieldType.STRING, required=True,
                        description="Complete CLAUDE.md content"),
        'content_hash': Field('content_hash', FieldType.STRING, required=False,
                             default='',
                             description="SHA256 hash of content"),
        'source': Field('source', FieldType.STRING, required=False,
                       default='',
                       description="Generation source (e.g., 'layer_merge')"),
        'layer_ids': Field('layer_ids', FieldType.LIST, required=False,
                          default=[],
                          item_type=FieldType.STRING,
                          description="Layer IDs included in this version"),
        'notes': Field('notes', FieldType.STRING, required=False,
                      default='',
                      description="Version notes"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# Team Schema
# =============================================================================

class TeamSchema(BaseSchema):
    """
    Schema for Team entities.

    Teams organize agents with roles and capabilities.
    """
    schema_version = 1
    entity_type = 'team'

    fields = {
        **BASE_ENTITY_FIELDS,
        'name': Field('name', FieldType.STRING, required=True,
                     description="Team name"),
        'description': Field('description', FieldType.STRING, required=False,
                            default='',
                            description="Team description"),
        'members': Field('members', FieldType.LIST, required=False,
                        default=[],
                        description="Team member definitions"),
        'capabilities': Field('capabilities', FieldType.LIST, required=False,
                             default=[],
                             item_type=FieldType.STRING,
                             description="Team capabilities"),
        'protocols': Field('protocols', FieldType.LIST, required=False,
                          default=[],
                          description="Communication protocols"),
        'active': Field('active', FieldType.BOOLEAN, required=False,
                       default=True,
                       description="Whether team is active"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# PersonaProfile Schema
# =============================================================================

class PersonaProfileSchema(BaseSchema):
    """
    Schema for PersonaProfile entities.

    Personas define agent characteristics and behavior.
    """
    schema_version = 1
    entity_type = 'persona_profile'

    fields = {
        **BASE_ENTITY_FIELDS,
        'name': Field('name', FieldType.STRING, required=True,
                     description="Persona name"),
        'role': Field('role', FieldType.STRING, required=False,
                     default='',
                     description="Primary role"),
        'description': Field('description', FieldType.STRING, required=False,
                            default='',
                            description="Persona description"),
        'expertise': Field('expertise', FieldType.LIST, required=False,
                          default=[],
                          item_type=FieldType.STRING,
                          description="Areas of expertise"),
        'communication_style': Field('communication_style', FieldType.STRING, required=False,
                                    default='',
                                    description="How the persona communicates"),
        'constraints': Field('constraints', FieldType.LIST, required=False,
                            default=[],
                            item_type=FieldType.STRING,
                            description="Behavioral constraints"),
        'preferences': Field('preferences', FieldType.DICT, required=False,
                            default={},
                            description="Persona preferences"),
        'active': Field('active', FieldType.BOOLEAN, required=False,
                       default=True,
                       description="Whether persona is active"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# Document Schema
# =============================================================================

class DocumentSchema(BaseSchema):
    """
    Schema for Document entities.

    Documents store text content with metadata.
    """
    schema_version = 1
    entity_type = 'document'

    fields = {
        **BASE_ENTITY_FIELDS,
        'title': Field('title', FieldType.STRING, required=True,
                      description="Document title"),
        'content': Field('content', FieldType.STRING, required=False,
                        default='',
                        description="Document content"),
        'doc_type': Field('doc_type', FieldType.STRING, required=False,
                         default='',
                         description="Document type (e.g., 'memory', 'reference')"),
        'path': Field('path', FieldType.STRING, required=False,
                     default='',
                     description="File path if applicable"),
        'content_hash': Field('content_hash', FieldType.STRING, required=False,
                             default='',
                             description="SHA256 hash of content"),
        'tags': Field('tags', FieldType.LIST, required=False,
                     default=[],
                     item_type=FieldType.STRING,
                     description="Document tags"),
        'references': Field('references', FieldType.LIST, required=False,
                           default=[],
                           item_type=FieldType.STRING,
                           description="Referenced entity IDs"),
        'properties': Field('properties', FieldType.DICT, required=False,
                           default={},
                           description="Additional properties"),
        'metadata': Field('metadata', FieldType.DICT, required=False,
                         default={},
                         description="System metadata"),
    }


# =============================================================================
# Schema Registration
# =============================================================================

# All schema classes
ALL_SCHEMAS = {
    'task': TaskSchema,
    'decision': DecisionSchema,
    'sprint': SprintSchema,
    'epic': EpicSchema,
    'edge': EdgeSchema,
    'handoff': HandoffSchema,
    'claudemd_layer': ClaudeMdLayerSchema,
    'claudemd_version': ClaudeMdVersionSchema,
    'team': TeamSchema,
    'persona_profile': PersonaProfileSchema,
    'document': DocumentSchema,
}

_schemas_registered = False


def ensure_schemas_registered() -> None:
    """
    Ensure all entity schemas are registered in the global registry.

    Safe to call multiple times - only registers once.
    """
    global _schemas_registered
    if _schemas_registered:
        return

    for entity_type, schema_class in ALL_SCHEMAS.items():
        register_schema(entity_type, schema_class)

    _schemas_registered = True


def get_schema_for_entity_type(entity_type: str) -> type:
    """
    Get schema class for an entity type.

    Args:
        entity_type: Entity type name

    Returns:
        Schema class or None if not found
    """
    return ALL_SCHEMAS.get(entity_type)


def list_entity_types() -> list:
    """
    List all registered entity types.

    Returns:
        List of entity type names
    """
    return list(ALL_SCHEMAS.keys())


# Auto-register schemas on module import
ensure_schemas_registered()
