"""
Unit tests for PersonaProfile entity.

Tests PersonaProfile creation, validation, serialization, and layer preference logic.
"""

import pytest
from datetime import datetime, timezone

from cortical.got.types import PersonaProfile
from cortical.got.errors import ValidationError


class TestPersonaProfileInstantiation:
    """Test PersonaProfile entity instantiation with various parameters."""

    def test_minimal_instantiation(self):
        """Test creating PersonaProfile with only required fields."""
        profile = PersonaProfile(id="PP-test-001")

        assert profile.id == "PP-test-001"
        assert profile.entity_type == "persona_profile"
        assert profile.name == ""
        assert profile.role == ""
        assert profile.team_id == ""

    def test_full_instantiation(self):
        """Test creating PersonaProfile with all fields populated."""
        profile = PersonaProfile(
            id="PP-backend-dev",
            name="Senior Backend Developer",
            role="developer",
            team_id="TEAM-engineering",
            layer_preferences={"spark-module": False, "query-module": True},
            inherits_from="PP-base-dev",
            default_branch="dev",
            default_modules=["query", "analysis"],
            custom_layers=["custom-backend"],
            excluded_layers=["marketing"],
            properties={"skill_level": "senior"},
            metadata={"created_by": "admin"}
        )

        assert profile.id == "PP-backend-dev"
        assert profile.name == "Senior Backend Developer"
        assert profile.role == "developer"
        assert profile.team_id == "TEAM-engineering"
        assert profile.layer_preferences == {"spark-module": False, "query-module": True}
        assert profile.inherits_from == "PP-base-dev"
        assert profile.default_branch == "dev"
        assert profile.default_modules == ["query", "analysis"]
        assert profile.custom_layers == ["custom-backend"]
        assert profile.excluded_layers == ["marketing"]
        assert profile.properties == {"skill_level": "senior"}
        assert profile.metadata == {"created_by": "admin"}


class TestPersonaProfileDefaults:
    """Test default values for PersonaProfile fields."""

    def test_default_values(self):
        """Verify all fields have correct default values."""
        profile = PersonaProfile(id="PP-test")

        # String fields default to empty string
        assert profile.name == ""
        assert profile.role == ""
        assert profile.team_id == ""
        assert profile.inherits_from == ""
        assert profile.default_branch == ""

        # List fields default to empty list
        assert profile.default_modules == []
        assert profile.custom_layers == []
        assert profile.excluded_layers == []

        # Dict fields default to empty dict
        assert profile.layer_preferences == {}
        assert profile.properties == {}
        assert profile.metadata == {}

        # Entity base class defaults
        assert profile.entity_type == "persona_profile"
        assert profile.version == 1
        assert isinstance(profile.created_at, str)
        assert isinstance(profile.modified_at, str)

    def test_timestamp_defaults(self):
        """Verify created_at and modified_at are set to current time."""
        before = datetime.now(timezone.utc)
        profile = PersonaProfile(id="PP-test")
        after = datetime.now(timezone.utc)

        created = datetime.fromisoformat(profile.created_at.replace('Z', '+00:00'))
        modified = datetime.fromisoformat(profile.modified_at.replace('Z', '+00:00'))

        assert before <= created <= after
        assert before <= modified <= after


class TestPersonaProfileValidation:
    """Test PersonaProfile validation logic."""

    def test_valid_roles(self):
        """Test that all valid roles are accepted."""
        valid_roles = ["developer", "qa", "devops", "marketing", "manager", "analyst", "designer", ""]

        for role in valid_roles:
            profile = PersonaProfile(id=f"PP-{role or 'empty'}", role=role)
            assert profile.role == role

    def test_invalid_role_raises_error(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PersonaProfile(id="PP-test", role="invalid_role")

        assert "Invalid role 'invalid_role'" in str(exc_info.value)
        assert 'valid_roles' in exc_info.value.context
        assert isinstance(exc_info.value.context['valid_roles'], list)

    def test_entity_type_auto_set(self):
        """Test that entity_type is automatically set to 'persona_profile'."""
        profile = PersonaProfile(id="PP-test")
        assert profile.entity_type == "persona_profile"

        # Even if provided explicitly (in from_dict), it should be set
        profile2 = PersonaProfile(id="PP-test2", entity_type="wrong_type")
        assert profile2.entity_type == "persona_profile"


class TestPersonaProfileSerialization:
    """Test PersonaProfile to_dict() serialization."""

    def test_to_dict_minimal(self):
        """Test serialization of minimal PersonaProfile."""
        profile = PersonaProfile(id="PP-test")
        data = profile.to_dict()

        assert data["id"] == "PP-test"
        assert data["entity_type"] == "persona_profile"
        assert data["version"] == 1
        assert data["name"] == ""
        assert data["role"] == ""
        assert data["team_id"] == ""
        assert data["layer_preferences"] == {}
        assert data["inherits_from"] == ""
        assert data["default_branch"] == ""
        assert data["default_modules"] == []
        assert data["custom_layers"] == []
        assert data["excluded_layers"] == []
        assert data["properties"] == {}
        assert data["metadata"] == {}
        assert "created_at" in data
        assert "modified_at" in data

    def test_to_dict_full(self):
        """Test serialization of fully populated PersonaProfile."""
        profile = PersonaProfile(
            id="PP-backend-dev",
            name="Backend Developer",
            role="developer",
            team_id="TEAM-eng",
            layer_preferences={"module-a": True, "module-b": False},
            inherits_from="PP-base",
            default_branch="dev",
            default_modules=["query"],
            custom_layers=["custom"],
            excluded_layers=["excluded"],
            properties={"key": "value"},
            metadata={"meta": "data"}
        )
        data = profile.to_dict()

        assert data["id"] == "PP-backend-dev"
        assert data["name"] == "Backend Developer"
        assert data["role"] == "developer"
        assert data["team_id"] == "TEAM-eng"
        assert data["layer_preferences"] == {"module-a": True, "module-b": False}
        assert data["inherits_from"] == "PP-base"
        assert data["default_branch"] == "dev"
        assert data["default_modules"] == ["query"]
        assert data["custom_layers"] == ["custom"]
        assert data["excluded_layers"] == ["excluded"]
        assert data["properties"] == {"key": "value"}
        assert data["metadata"] == {"meta": "data"}

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all expected fields."""
        profile = PersonaProfile(id="PP-test")
        data = profile.to_dict()

        expected_fields = {
            "id", "entity_type", "version", "created_at", "modified_at",
            "name", "role", "team_id", "layer_preferences", "inherits_from",
            "default_branch", "default_modules", "custom_layers",
            "excluded_layers", "properties", "metadata"
        }

        assert set(data.keys()) == expected_fields


class TestPersonaProfileDeserialization:
    """Test PersonaProfile from_dict() deserialization."""

    def test_from_dict_minimal(self):
        """Test deserialization with minimal required fields."""
        data = {"id": "PP-test"}
        profile = PersonaProfile.from_dict(data)

        assert profile.id == "PP-test"
        assert profile.entity_type == "persona_profile"
        assert profile.version == 1
        assert profile.name == ""
        assert profile.role == ""

    def test_from_dict_full(self):
        """Test deserialization with all fields."""
        data = {
            "id": "PP-backend-dev",
            "entity_type": "persona_profile",
            "version": 2,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-02T00:00:00+00:00",
            "name": "Backend Developer",
            "role": "developer",
            "team_id": "TEAM-eng",
            "layer_preferences": {"module-a": True},
            "inherits_from": "PP-base",
            "default_branch": "dev",
            "default_modules": ["query"],
            "custom_layers": ["custom"],
            "excluded_layers": ["excluded"],
            "properties": {"key": "value"},
            "metadata": {"meta": "data"}
        }
        profile = PersonaProfile.from_dict(data)

        assert profile.id == "PP-backend-dev"
        assert profile.entity_type == "persona_profile"
        assert profile.version == 2
        assert profile.created_at == "2025-01-01T00:00:00+00:00"
        assert profile.modified_at == "2025-01-02T00:00:00+00:00"
        assert profile.name == "Backend Developer"
        assert profile.role == "developer"
        assert profile.team_id == "TEAM-eng"
        assert profile.layer_preferences == {"module-a": True}
        assert profile.inherits_from == "PP-base"
        assert profile.default_branch == "dev"
        assert profile.default_modules == ["query"]
        assert profile.custom_layers == ["custom"]
        assert profile.excluded_layers == ["excluded"]
        assert profile.properties == {"key": "value"}
        assert profile.metadata == {"meta": "data"}

    def test_from_dict_with_missing_optional_fields(self):
        """Test deserialization handles missing optional fields gracefully."""
        data = {
            "id": "PP-test",
            "name": "Test Profile"
        }
        profile = PersonaProfile.from_dict(data)

        assert profile.id == "PP-test"
        assert profile.name == "Test Profile"
        assert profile.role == ""
        assert profile.layer_preferences == {}
        assert profile.default_modules == []


class TestPersonaProfileRoundTrip:
    """Test round-trip serialization (to_dict -> from_dict)."""

    def test_roundtrip_minimal(self):
        """Test round-trip preserves minimal PersonaProfile data."""
        original = PersonaProfile(id="PP-test")
        data = original.to_dict()
        restored = PersonaProfile.from_dict(data)

        assert restored.id == original.id
        assert restored.entity_type == original.entity_type
        assert restored.version == original.version
        assert restored.name == original.name
        assert restored.role == original.role
        assert restored.layer_preferences == original.layer_preferences

    def test_roundtrip_full(self):
        """Test round-trip preserves all PersonaProfile data."""
        original = PersonaProfile(
            id="PP-backend-dev",
            name="Backend Developer",
            role="developer",
            team_id="TEAM-eng",
            layer_preferences={"module-a": True, "module-b": False},
            inherits_from="PP-base",
            default_branch="dev",
            default_modules=["query", "analysis"],
            custom_layers=["custom1", "custom2"],
            excluded_layers=["excluded1"],
            properties={"skill": "senior"},
            metadata={"created_by": "admin"}
        )

        data = original.to_dict()
        restored = PersonaProfile.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.role == original.role
        assert restored.team_id == original.team_id
        assert restored.layer_preferences == original.layer_preferences
        assert restored.inherits_from == original.inherits_from
        assert restored.default_branch == original.default_branch
        assert restored.default_modules == original.default_modules
        assert restored.custom_layers == original.custom_layers
        assert restored.excluded_layers == original.excluded_layers
        assert restored.properties == original.properties
        assert restored.metadata == original.metadata
        assert restored.created_at == original.created_at
        assert restored.modified_at == original.modified_at

    def test_roundtrip_complex_preferences(self):
        """Test round-trip preserves complex layer preferences."""
        original = PersonaProfile(
            id="PP-test",
            layer_preferences={
                "spark": False,
                "query": True,
                "analysis": True,
                "marketing": False,
                "custom-section": True
            }
        )

        data = original.to_dict()
        restored = PersonaProfile.from_dict(data)

        assert restored.layer_preferences == original.layer_preferences


class TestShouldIncludeLayer:
    """Test the should_include_layer() method."""

    def test_excluded_layers_take_precedence(self):
        """Test that excluded_layers override all other settings."""
        profile = PersonaProfile(
            id="PP-test",
            excluded_layers=["section-a"],
            custom_layers=["section-a"],  # Also in custom - exclusion wins
            layer_preferences={"section-a": True}  # Also in preferences - exclusion wins
        )

        assert profile.should_include_layer("section-a") is False

    def test_custom_layers_included(self):
        """Test that custom_layers are always included (unless excluded)."""
        profile = PersonaProfile(
            id="PP-test",
            custom_layers=["section-b"],
            layer_preferences={"section-b": False}  # Preference says no, but custom wins
        )

        assert profile.should_include_layer("section-b") is True

    def test_layer_preferences_respected(self):
        """Test that layer_preferences control inclusion."""
        profile = PersonaProfile(
            id="PP-test",
            layer_preferences={
                "section-include": True,
                "section-exclude": False
            }
        )

        assert profile.should_include_layer("section-include") is True
        assert profile.should_include_layer("section-exclude") is False

    def test_default_inclusion(self):
        """Test that unknown sections default to included."""
        profile = PersonaProfile(id="PP-test")

        assert profile.should_include_layer("unknown-section") is True

    def test_priority_order(self):
        """Test the full priority order: excluded > custom > preferences > default."""
        profile = PersonaProfile(
            id="PP-test",
            excluded_layers=["excluded-1"],
            custom_layers=["custom-1"],
            layer_preferences={"pref-1": False}
        )

        # Excluded - highest priority
        assert profile.should_include_layer("excluded-1") is False

        # Custom - second priority
        assert profile.should_include_layer("custom-1") is True

        # Preference - third priority
        assert profile.should_include_layer("pref-1") is False

        # Default - lowest priority
        assert profile.should_include_layer("unknown") is True

    def test_empty_section_id(self):
        """Test handling of empty section_id."""
        profile = PersonaProfile(id="PP-test")

        # Empty string should default to included
        assert profile.should_include_layer("") is True


class TestGetEffectivePreferences:
    """Test the get_effective_preferences() method."""

    def test_no_parent_returns_own_preferences(self):
        """Test that without parent, returns own layer_preferences."""
        profile = PersonaProfile(
            id="PP-test",
            layer_preferences={"section-a": True, "section-b": False}
        )

        effective = profile.get_effective_preferences()

        assert effective == {"section-a": True, "section-b": False}
        # Should be a copy, not the same object
        assert effective is not profile.layer_preferences

    def test_inheritance_from_parent(self):
        """Test that preferences inherit from parent."""
        parent = PersonaProfile(
            id="PP-parent",
            layer_preferences={
                "section-a": True,
                "section-b": False,
                "section-c": True
            }
        )

        child = PersonaProfile(
            id="PP-child",
            layer_preferences={
                "section-b": True,  # Override parent
                "section-d": False  # New preference
            }
        )

        effective = child.get_effective_preferences(parent)

        assert effective == {
            "section-a": True,   # From parent
            "section-b": True,   # Overridden by child
            "section-c": True,   # From parent
            "section-d": False   # From child
        }

    def test_child_overrides_parent(self):
        """Test that child preferences override parent preferences."""
        parent = PersonaProfile(
            id="PP-parent",
            layer_preferences={"section-a": False}
        )

        child = PersonaProfile(
            id="PP-child",
            layer_preferences={"section-a": True}
        )

        effective = child.get_effective_preferences(parent)

        assert effective["section-a"] is True

    def test_empty_preferences(self):
        """Test inheritance with empty preference sets."""
        parent = PersonaProfile(id="PP-parent")
        child = PersonaProfile(id="PP-child")

        effective = child.get_effective_preferences(parent)

        assert effective == {}

    def test_parent_only_preferences(self):
        """Test child with no preferences inherits all from parent."""
        parent = PersonaProfile(
            id="PP-parent",
            layer_preferences={"section-a": True, "section-b": False}
        )

        child = PersonaProfile(id="PP-child")

        effective = child.get_effective_preferences(parent)

        assert effective == {"section-a": True, "section-b": False}

    def test_complex_inheritance(self):
        """Test complex preference inheritance scenario."""
        parent = PersonaProfile(
            id="PP-parent",
            layer_preferences={
                "spark": False,
                "query": True,
                "analysis": True,
                "marketing": False,
                "docs": True
            }
        )

        child = PersonaProfile(
            id="PP-child",
            layer_preferences={
                "spark": True,      # Override to enable
                "marketing": True,  # Override to enable
                "custom": False     # Add new preference
            }
        )

        effective = child.get_effective_preferences(parent)

        assert effective == {
            "spark": True,       # Child override
            "query": True,       # From parent
            "analysis": True,    # From parent
            "marketing": True,   # Child override
            "docs": True,        # From parent
            "custom": False      # From child
        }


class TestPersonaProfileEntityType:
    """Test entity_type field behavior."""

    def test_entity_type_is_persona_profile(self):
        """Test that entity_type is always 'persona_profile'."""
        profile = PersonaProfile(id="PP-test")
        assert profile.entity_type == "persona_profile"

    def test_entity_type_in_dict(self):
        """Test that entity_type appears in serialized dict."""
        profile = PersonaProfile(id="PP-test")
        data = profile.to_dict()
        assert data["entity_type"] == "persona_profile"

    def test_entity_type_from_dict(self):
        """Test that entity_type is set correctly when deserializing."""
        data = {"id": "PP-test", "entity_type": "persona_profile"}
        profile = PersonaProfile.from_dict(data)
        assert profile.entity_type == "persona_profile"

    def test_entity_type_overridden_in_post_init(self):
        """Test that entity_type is set in __post_init__ even if wrong value provided."""
        # This tests that __post_init__ always sets it correctly
        profile = PersonaProfile(id="PP-test", entity_type="wrong_type")
        assert profile.entity_type == "persona_profile"
