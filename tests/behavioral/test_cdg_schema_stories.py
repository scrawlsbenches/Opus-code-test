"""
Behavioral tests for CDG Schema infrastructure.

These tests define the expected behavior for:
- SchemaRegistry lifecycle (Container-managed, not singleton)
- Schema validation during entity writes
- Referential integrity rules
- Migration support

The schema module lives in CDG (foundation layer) and provides
validation infrastructure that domain layers (GoT, CEL) build upon.
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from cortical.cdg.storage import CDGStore
from cortical.cdg.schema import (
    SchemaRegistry,
    BaseSchema,
    Field,
    FieldType,
    ValidationResult,
    OnDeleteAction,
    ReferenceRule,
    get_registry,
    set_registry,
)
from cortical.cdg.errors import ValidationError
from cortical.common.filesystem import InMemoryFileSystem


# ============================================================================
# Test Entities and Schemas
# ============================================================================

@dataclass
class SampleEntity:
    """Minimal entity for schema testing."""
    id: str
    name: str
    status: str
    entity_type: str = "sample_entity"
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "entity_type": self.entity_type,
            "version": self.version,
        }

    def bump_version(self) -> None:
        self.version += 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleEntity":
        return cls(
            id=data["id"],
            name=data["name"],
            status=data.get("status", "pending"),
            entity_type=data.get("entity_type", "sample_entity"),
            version=data.get("version", 1),
        )


class SampleEntitySchema(BaseSchema):
    """Schema for SampleEntity."""
    schema_version = 1
    entity_type = "sample_entity"
    id_prefix = "SE-"
    fields = {
        "id": Field("id", FieldType.STRING, required=True),
        "name": Field("name", FieldType.STRING, required=True),
        "status": Field(
            "status",
            FieldType.ENUM,
            required=True,
            choices=["pending", "active", "completed"]
        ),
    }


def sample_entity_factory(data: Dict[str, Any]) -> SampleEntity:
    """Factory function for CDGStore."""
    return SampleEntity.from_dict(data)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def registry():
    """Create a fresh SchemaRegistry for testing."""
    return SchemaRegistry()


@pytest.fixture
def store_with_schema(tmp_path):
    """Create CDGStore with schema validation enabled."""
    registry = SchemaRegistry()
    registry.register("sample_entity", SampleEntitySchema)

    fs = InMemoryFileSystem()
    store = CDGStore(
        tmp_path / "entities",
        entity_factory=sample_entity_factory,
        filesystem=fs,
        schema_registry=registry,
    )
    return store, registry


@pytest.fixture
def store_without_schema(tmp_path):
    """Create CDGStore without schema registry (validation disabled)."""
    fs = InMemoryFileSystem()
    return CDGStore(
        tmp_path / "entities",
        entity_factory=sample_entity_factory,
        filesystem=fs,
        schema_registry=None,  # No schema validation
    )


# ============================================================================
# Test: Schema Registry Lifecycle
# ============================================================================

class TestSchemaRegistryLifecycle:
    """
    Schema registry is now Container-managed, not a singleton.

    GIVEN multiple SchemaRegistry instantiations
    WHEN each creates its own instance
    THEN they should be independent registries
    """

    def test_multiple_registries_are_independent(self):
        """
        GIVEN two SchemaRegistry instances
        WHEN schemas are registered to one
        THEN the other should not see them
        """
        reg1 = SchemaRegistry()
        reg2 = SchemaRegistry()

        # They are different instances
        assert reg1 is not reg2

        # Register schema to reg1
        reg1.register("sample_entity", SampleEntitySchema)

        # reg1 has it, reg2 doesn't
        assert reg1.has_schema("sample_entity")
        assert not reg2.has_schema("sample_entity")

    def test_global_registry_can_be_replaced(self):
        """
        GIVEN a global registry with schemas
        WHEN a new registry is set via set_registry()
        THEN the new registry becomes the global
        """
        # Create and set a new registry
        new_registry = SchemaRegistry()
        new_registry.register("custom", SampleEntitySchema)
        set_registry(new_registry)

        # Global now points to new registry
        assert get_registry() is new_registry
        assert get_registry().has_schema("custom")


# ============================================================================
# Test: Schema Validation on Write
# ============================================================================

class TestSchemaValidationOnWrite:
    """
    When SchemaRegistry is injected into CDGStore, entities are
    validated against their schema before writing.
    """

    def test_valid_entity_writes_successfully(self, store_with_schema):
        """
        GIVEN an entity that passes schema validation
        WHEN writing to store with schema validation
        THEN write should succeed
        """
        store, registry = store_with_schema

        entity = SampleEntity(
            id="SE-001",
            name="Valid Entity",
            status="pending"
        )

        # Should not raise
        store.write(entity)

        # Verify it was written
        loaded = store.read("SE-001")
        assert loaded is not None
        assert loaded.name == "Valid Entity"

    def test_invalid_status_fails_validation(self, store_with_schema):
        """
        GIVEN an entity with invalid status enum value
        WHEN writing to store with schema validation
        THEN should raise ValidationError
        """
        store, registry = store_with_schema

        entity = SampleEntity(
            id="SE-002",
            name="Invalid Status",
            status="invalid_status"  # Not in allowed choices
        )

        with pytest.raises(ValidationError) as exc_info:
            store.write(entity)

        assert "Schema validation failed" in str(exc_info.value)
        assert "status" in str(exc_info.value)

    def test_missing_required_field_fails_validation(self, store_with_schema):
        """
        GIVEN an entity missing a required field
        WHEN writing to store with schema validation
        THEN should raise ValidationError
        """
        store, registry = store_with_schema

        # Create entity with missing name (hack via to_dict override)
        entity = SampleEntity(
            id="SE-003",
            name="",  # Empty name
            status="pending"
        )
        # Override to_dict to omit name entirely
        original_to_dict = entity.to_dict

        def to_dict_without_name():
            d = original_to_dict()
            del d["name"]
            return d

        entity.to_dict = to_dict_without_name

        with pytest.raises(ValidationError) as exc_info:
            store.write(entity)

        assert "Schema validation failed" in str(exc_info.value)
        assert "name" in str(exc_info.value)

    def test_validation_skipped_without_registry(self, store_without_schema):
        """
        GIVEN a store without SchemaRegistry
        WHEN writing an entity with invalid data
        THEN write should succeed (no schema validation)
        """
        store = store_without_schema

        # Invalid status would fail schema validation if registry was set
        entity = SampleEntity(
            id="SE-004",
            name="No Schema Check",
            status="whatever"  # Invalid but no schema to check
        )

        # Should not raise - no schema validation
        store.write(entity)

        loaded = store.read("SE-004")
        assert loaded.status == "whatever"

    def test_unknown_entity_type_skips_validation(self, store_with_schema):
        """
        GIVEN an entity type without a registered schema
        WHEN writing to store
        THEN write should succeed (no schema to validate against)
        """
        store, registry = store_with_schema

        # Use different entity type not in registry
        entity = SampleEntity(
            id="UNK-001",
            name="Unknown Type",
            status="invalid"
        )
        entity.entity_type = "unknown_type"

        # Should succeed - no schema for unknown_type
        store.write(entity)

        loaded = store.read("UNK-001")
        assert loaded is not None


# ============================================================================
# Test: Reference Rules
# ============================================================================

class TestReferenceRules:
    """
    Reference rules define referential integrity constraints.
    """

    def test_add_and_retrieve_reference_rule(self, registry):
        """
        GIVEN a registry
        WHEN adding a reference rule
        THEN it should be retrievable
        """
        rule = ReferenceRule(
            field="parent_id",
            must_exist=True,
            on_delete=OnDeleteAction.CASCADE
        )
        registry.add_reference_rule("child_entity", rule)

        rules = registry.get_reference_rules("child_entity")
        assert len(rules) == 1
        assert rules[0].field == "parent_id"
        assert rules[0].must_exist is True
        assert rules[0].on_delete == OnDeleteAction.CASCADE

    def test_multiple_rules_per_entity_type(self, registry):
        """
        GIVEN an entity type with multiple references
        WHEN adding multiple rules
        THEN all should be retrievable
        """
        rule1 = ReferenceRule(field="from_id", must_exist=True)
        rule2 = ReferenceRule(field="to_id", must_exist=True)

        registry.add_reference_rule("edge", rule1)
        registry.add_reference_rule("edge", rule2)

        rules = registry.get_reference_rules("edge")
        assert len(rules) == 2
        fields = {r.field for r in rules}
        assert fields == {"from_id", "to_id"}

    def test_get_referencing_rules(self, registry):
        """
        GIVEN rules that target specific entity types
        WHEN querying for rules that reference a type
        THEN should return all matching rules
        """
        # Edge references tasks
        edge_rule = ReferenceRule(
            field="from_id",
            must_exist=True,
            target_types=["task"]
        )
        registry.add_reference_rule("edge", edge_rule)

        # Sprint contains tasks
        sprint_rule = ReferenceRule(
            field="task_ids",
            is_collection=True,
            target_types=["task"],
            on_delete=OnDeleteAction.SET_NULL
        )
        registry.add_reference_rule("sprint", sprint_rule)

        # Get rules that reference tasks
        referencing = registry.get_referencing_rules("task")
        assert len(referencing) == 2

        entity_types = {et for et, rule in referencing}
        assert entity_types == {"edge", "sprint"}

    def test_get_on_delete_actions(self, registry):
        """
        GIVEN entities with different on_delete actions
        WHEN getting on_delete actions for a target type
        THEN should return grouped actions
        """
        registry.add_reference_rule("edge", ReferenceRule(
            field="from_id",
            on_delete=OnDeleteAction.CASCADE,
            target_types=["task"]
        ))
        registry.add_reference_rule("sprint", ReferenceRule(
            field="task_ids",
            on_delete=OnDeleteAction.SET_NULL,
            target_types=["task"]
        ))

        actions = registry.get_on_delete_actions("task")

        assert "edge" in actions
        assert ("from_id", OnDeleteAction.CASCADE) in actions["edge"]

        assert "sprint" in actions
        assert ("task_ids", OnDeleteAction.SET_NULL) in actions["sprint"]

    def test_rules_with_no_target_type_match_all(self, registry):
        """
        GIVEN a rule without target_types (matches any)
        WHEN querying for a specific type
        THEN the rule should match
        """
        # Generic rule that applies to any entity type
        generic_rule = ReferenceRule(
            field="ref_id",
            must_exist=True,
            target_types=None  # Matches any type
        )
        registry.add_reference_rule("ref_holder", generic_rule)

        # Should match when querying for any type
        referencing = registry.get_referencing_rules("task")
        assert len(referencing) == 1

        referencing = registry.get_referencing_rules("decision")
        assert len(referencing) == 1


# ============================================================================
# Test: Clear Behavior
# ============================================================================

class TestClearBehavior:
    """
    Registry clear should reset both schemas and rules.
    """

    def test_clear_removes_schemas_and_rules(self, registry):
        """
        GIVEN a registry with schemas and rules
        WHEN calling clear()
        THEN both should be removed
        """
        registry.register("test", SampleEntitySchema)
        registry.add_reference_rule("test", ReferenceRule(field="ref"))

        assert registry.has_schema("test")
        assert len(registry.get_reference_rules("test")) == 1

        registry.clear()

        assert not registry.has_schema("test")
        assert len(registry.get_reference_rules("test")) == 0


# ============================================================================
# Test: Schema Migration
# ============================================================================

class TestSchemaMigration:
    """
    Schema migration converts data from old versions to current version.
    """

    def test_migrate_adds_missing_fields(self, registry):
        """
        GIVEN a schema with migration support
        WHEN migrating old data
        THEN migration should add missing fields
        """

        class MigratableSchema(BaseSchema):
            schema_version = 2
            entity_type = "migratable"
            fields = {
                "id": Field("id", FieldType.STRING, required=True),
                "name": Field("name", FieldType.STRING, required=True),
                "new_field": Field("new_field", FieldType.STRING, required=False, default="default"),
            }

            @classmethod
            def migrate_v1_to_v2(cls, data: Dict[str, Any]) -> Dict[str, Any]:
                if "new_field" not in data:
                    data["new_field"] = "migrated_value"
                return data

        registry.register("migratable", MigratableSchema)

        # Old data (v1) missing new_field
        old_data = {"id": "M-001", "name": "Old", "_schema_version": 1}

        migrated, result = registry.migrate("migratable", old_data)

        assert result.valid
        assert result.migrated
        assert migrated.get("new_field") == "migrated_value"
        assert migrated.get("_schema_version") == 2

    def test_no_migration_for_current_version(self, registry):
        """
        GIVEN data at current schema version
        WHEN attempting migration
        THEN data should be unchanged
        """
        registry.register("sample_entity", SampleEntitySchema)

        data = {
            "id": "SE-001",
            "name": "Current",
            "status": "pending",
            "_schema_version": SampleEntitySchema.schema_version
        }

        migrated, result = registry.migrate("sample_entity", data)

        assert not result.migrated
        assert migrated == data


# ============================================================================
# Test: Integration with Container
# ============================================================================

class TestContainerIntegration:
    """
    SchemaRegistry should integrate with DI Container.
    """

    def test_container_provides_same_registry_to_all_services(self):
        """
        GIVEN a container with SchemaModule applied
        WHEN resolving SchemaRegistry from different services
        THEN should get the same instance
        """
        from cortical.core.bootstrap import create_container
        from cortical.cdg.schema import SchemaRegistry
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            container = create_container(
                got_dir=Path(tmp) / ".got",
                use_memory=True
            )

            reg1 = container.resolve(SchemaRegistry)
            reg2 = container.resolve(SchemaRegistry)

            # Same instance (singleton within container)
            assert reg1 is reg2

            # And it has GoT schemas registered
            assert reg1.has_schema("task")
            assert reg1.has_schema("decision")
            assert reg1.has_schema("edge")
