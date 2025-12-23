"""
Unit tests for Team entity.

Tests cover instantiation, serialization, deserialization, and team-specific
methods like is_in_scope() and matches_branch().
"""

import pytest
from datetime import datetime, timezone

from cortical.got.types import Team
from cortical.got.errors import ValidationError


class TestTeamInstantiation:
    """Test creating Team instances with various parameters."""

    def test_minimal_team(self):
        """Test creating team with only required ID."""
        team = Team(id="TEAM-001")
        assert team.id == "TEAM-001"
        assert team.entity_type == "team"

    def test_full_team(self):
        """Test creating team with all parameters."""
        team = Team(
            id="TEAM-backend",
            name="Backend Engineering",
            description="Backend development team",
            parent_team_id="TEAM-engineering",
            branch_patterns=["feature/*", "dev"],
            module_scope=["query", "analysis"],
            layer_ids=["layer1", "layer2"],
            member_profiles=["profile1", "profile2"],
            settings={"foo": "bar"},
            properties={"custom": "value"},
            metadata={"meta": "data"}
        )
        assert team.id == "TEAM-backend"
        assert team.name == "Backend Engineering"
        assert team.description == "Backend development team"
        assert team.parent_team_id == "TEAM-engineering"
        assert team.branch_patterns == ["feature/*", "dev"]
        assert team.module_scope == ["query", "analysis"]
        assert team.layer_ids == ["layer1", "layer2"]
        assert team.member_profiles == ["profile1", "profile2"]
        assert team.settings == {"foo": "bar"}
        assert team.properties == {"custom": "value"}
        assert team.metadata == {"meta": "data"}

    def test_with_version_and_timestamps(self):
        """Test creating team with explicit version and timestamps."""
        created = "2025-01-01T00:00:00+00:00"
        modified = "2025-01-02T00:00:00+00:00"
        team = Team(
            id="TEAM-001",
            version=5,
            created_at=created,
            modified_at=modified
        )
        assert team.version == 5
        assert team.created_at == created
        assert team.modified_at == modified


class TestTeamDefaults:
    """Test default values for Team fields."""

    def test_default_values(self):
        """Test all fields have correct defaults."""
        team = Team(id="TEAM-001")

        # Entity base fields
        assert team.id == "TEAM-001"
        assert team.entity_type == "team"
        assert team.version == 1
        assert team.created_at != ""
        assert team.modified_at != ""

        # Team-specific fields
        assert team.name == ""
        assert team.description == ""
        assert team.parent_team_id == ""
        assert team.branch_patterns == []
        assert team.module_scope == []
        assert team.layer_ids == []
        assert team.member_profiles == []
        assert team.settings == {}
        assert team.properties == {}
        assert team.metadata == {}

    def test_default_timestamps_are_valid(self):
        """Test that default timestamps are valid ISO format."""
        team = Team(id="TEAM-001")

        # Should be parseable as datetime
        created = datetime.fromisoformat(team.created_at.replace('Z', '+00:00'))
        modified = datetime.fromisoformat(team.modified_at.replace('Z', '+00:00'))

        assert isinstance(created, datetime)
        assert isinstance(modified, datetime)

    def test_list_defaults_are_mutable(self):
        """Test that list defaults are independent instances."""
        team1 = Team(id="TEAM-001")
        team2 = Team(id="TEAM-002")

        team1.branch_patterns.append("test")
        assert len(team2.branch_patterns) == 0  # Should not affect team2


class TestTeamValidation:
    """
    Test validation behavior for Team entity.

    Note: Team currently has no validation in __post_init__.
    These tests verify current behavior (no validation) and document
    where validation tests would go if validation were implemented.
    """

    def test_empty_name_allowed(self):
        """Test that empty name is currently allowed (no validation)."""
        team = Team(id="TEAM-001", name="")
        assert team.name == ""
        # TODO: If validation is added, this should raise ValidationError

    def test_self_referential_parent_allowed(self):
        """Test that self-referential parent is currently allowed (no validation)."""
        team = Team(id="TEAM-001", parent_team_id="TEAM-001")
        assert team.parent_team_id == "TEAM-001"
        # TODO: If validation is added, this should raise ValidationError

    def test_invalid_parent_team_id_format_allowed(self):
        """Test that invalid parent_team_id format is currently allowed."""
        team = Team(id="TEAM-001", parent_team_id="invalid-format")
        assert team.parent_team_id == "invalid-format"
        # TODO: If validation is added, might want to enforce TEAM-* format


class TestTeamSerialization:
    """Test Team to_dict() serialization."""

    def test_minimal_serialization(self):
        """Test serializing minimal team."""
        team = Team(id="TEAM-001")
        data = team.to_dict()

        assert data["id"] == "TEAM-001"
        assert data["entity_type"] == "team"
        assert data["version"] == 1
        assert "created_at" in data
        assert "modified_at" in data
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["parent_team_id"] == ""
        assert data["branch_patterns"] == []
        assert data["module_scope"] == []
        assert data["layer_ids"] == []
        assert data["member_profiles"] == []
        assert data["settings"] == {}
        assert data["properties"] == {}
        assert data["metadata"] == {}

    def test_full_serialization(self):
        """Test serializing team with all fields populated."""
        team = Team(
            id="TEAM-backend",
            name="Backend Engineering",
            description="Backend team",
            parent_team_id="TEAM-engineering",
            branch_patterns=["feature/*", "dev"],
            module_scope=["query", "analysis"],
            layer_ids=["L1", "L2"],
            member_profiles=["P1", "P2"],
            settings={"key": "value"},
            properties={"prop": "val"},
            metadata={"meta": "info"}
        )
        data = team.to_dict()

        assert data["id"] == "TEAM-backend"
        assert data["name"] == "Backend Engineering"
        assert data["description"] == "Backend team"
        assert data["parent_team_id"] == "TEAM-engineering"
        assert data["branch_patterns"] == ["feature/*", "dev"]
        assert data["module_scope"] == ["query", "analysis"]
        assert data["layer_ids"] == ["L1", "L2"]
        assert data["member_profiles"] == ["P1", "P2"]
        assert data["settings"] == {"key": "value"}
        assert data["properties"] == {"prop": "val"}
        assert data["metadata"] == {"meta": "info"}


class TestTeamDeserialization:
    """Test Team from_dict() deserialization."""

    def test_minimal_deserialization(self):
        """Test deserializing minimal team data."""
        data = {"id": "TEAM-001"}
        team = Team.from_dict(data)

        assert team.id == "TEAM-001"
        assert team.entity_type == "team"
        assert team.version == 1
        assert team.name == ""
        assert team.description == ""
        assert team.parent_team_id == ""
        assert team.branch_patterns == []
        assert team.module_scope == []

    def test_full_deserialization(self):
        """Test deserializing team with all fields."""
        data = {
            "id": "TEAM-backend",
            "entity_type": "team",
            "version": 3,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-02T00:00:00+00:00",
            "name": "Backend Engineering",
            "description": "Backend team",
            "parent_team_id": "TEAM-engineering",
            "branch_patterns": ["feature/*", "dev"],
            "module_scope": ["query", "analysis"],
            "layer_ids": ["L1", "L2"],
            "member_profiles": ["P1", "P2"],
            "settings": {"key": "value"},
            "properties": {"prop": "val"},
            "metadata": {"meta": "info"}
        }
        team = Team.from_dict(data)

        assert team.id == "TEAM-backend"
        assert team.entity_type == "team"
        assert team.version == 3
        assert team.created_at == "2025-01-01T00:00:00+00:00"
        assert team.modified_at == "2025-01-02T00:00:00+00:00"
        assert team.name == "Backend Engineering"
        assert team.description == "Backend team"
        assert team.parent_team_id == "TEAM-engineering"
        assert team.branch_patterns == ["feature/*", "dev"]
        assert team.module_scope == ["query", "analysis"]
        assert team.layer_ids == ["L1", "L2"]
        assert team.member_profiles == ["P1", "P2"]
        assert team.settings == {"key": "value"}
        assert team.properties == {"prop": "val"}
        assert team.metadata == {"meta": "info"}

    def test_missing_optional_fields(self):
        """Test deserialization with missing optional fields uses defaults."""
        data = {
            "id": "TEAM-001",
            "name": "Test Team"
        }
        team = Team.from_dict(data)

        assert team.id == "TEAM-001"
        assert team.name == "Test Team"
        assert team.entity_type == "team"  # Default
        assert team.version == 1  # Default
        assert team.branch_patterns == []  # Default
        assert team.module_scope == []  # Default


class TestTeamRoundTrip:
    """Test to_dict() -> from_dict() preserves all data."""

    def test_minimal_roundtrip(self):
        """Test round-trip with minimal team."""
        original = Team(id="TEAM-001")
        data = original.to_dict()
        restored = Team.from_dict(data)

        assert restored.id == original.id
        assert restored.entity_type == original.entity_type
        assert restored.version == original.version
        assert restored.name == original.name
        assert restored.description == original.description

    def test_full_roundtrip(self):
        """Test round-trip with fully populated team."""
        original = Team(
            id="TEAM-backend",
            name="Backend Engineering",
            description="Backend team",
            parent_team_id="TEAM-engineering",
            branch_patterns=["feature/*", "dev", "hotfix/*"],
            module_scope=["query", "analysis", "persistence"],
            layer_ids=["L1", "L2", "L3"],
            member_profiles=["P1", "P2", "P3"],
            settings={"setting1": "value1", "setting2": 42},
            properties={"prop1": "val1"},
            metadata={"meta1": "info1"}
        )
        data = original.to_dict()
        restored = Team.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.parent_team_id == original.parent_team_id
        assert restored.branch_patterns == original.branch_patterns
        assert restored.module_scope == original.module_scope
        assert restored.layer_ids == original.layer_ids
        assert restored.member_profiles == original.member_profiles
        assert restored.settings == original.settings
        assert restored.properties == original.properties
        assert restored.metadata == original.metadata

    def test_nested_data_roundtrip(self):
        """Test round-trip preserves nested data structures."""
        original = Team(
            id="TEAM-001",
            settings={
                "nested": {
                    "deep": {
                        "value": 123
                    }
                },
                "list": [1, 2, 3]
            }
        )
        data = original.to_dict()
        restored = Team.from_dict(data)

        assert restored.settings == original.settings
        assert restored.settings["nested"]["deep"]["value"] == 123
        assert restored.settings["list"] == [1, 2, 3]


class TestTeamIsInScope:
    """Test Team.is_in_scope() method."""

    def test_empty_scope_matches_all(self):
        """Test that empty module_scope matches all modules."""
        team = Team(id="TEAM-001", module_scope=[])

        assert team.is_in_scope("query")
        assert team.is_in_scope("analysis")
        assert team.is_in_scope("persistence")
        assert team.is_in_scope("anything")

    def test_exact_match(self):
        """Test exact module name match."""
        team = Team(
            id="TEAM-001",
            module_scope=["query", "analysis"]
        )

        assert team.is_in_scope("query")
        assert team.is_in_scope("analysis")

    def test_no_match(self):
        """Test module not in scope."""
        team = Team(
            id="TEAM-001",
            module_scope=["query", "analysis"]
        )

        assert not team.is_in_scope("persistence")
        assert not team.is_in_scope("semantics")
        assert not team.is_in_scope("other")

    def test_single_module_scope(self):
        """Test scope with single module."""
        team = Team(id="TEAM-001", module_scope=["query"])

        assert team.is_in_scope("query")
        assert not team.is_in_scope("analysis")

    def test_case_sensitive_match(self):
        """Test that module matching is case-sensitive."""
        team = Team(id="TEAM-001", module_scope=["Query"])

        assert team.is_in_scope("Query")
        assert not team.is_in_scope("query")


class TestTeamMatchesBranch:
    """Test Team.matches_branch() method."""

    def test_empty_patterns_match_all(self):
        """Test that empty branch_patterns matches all branches."""
        team = Team(id="TEAM-001", branch_patterns=[])

        assert team.matches_branch("main")
        assert team.matches_branch("dev")
        assert team.matches_branch("feature/foo")
        assert team.matches_branch("anything")

    def test_exact_match(self):
        """Test exact branch name match."""
        team = Team(
            id="TEAM-001",
            branch_patterns=["dev", "main"]
        )

        assert team.matches_branch("dev")
        assert team.matches_branch("main")

    def test_wildcard_pattern(self):
        """Test wildcard pattern matching."""
        team = Team(
            id="TEAM-001",
            branch_patterns=["feature/*"]
        )

        assert team.matches_branch("feature/auth")
        assert team.matches_branch("feature/search")
        assert team.matches_branch("feature/anything")
        assert not team.matches_branch("bugfix/auth")
        assert not team.matches_branch("main")

    def test_multiple_patterns(self):
        """Test multiple branch patterns."""
        team = Team(
            id="TEAM-001",
            branch_patterns=["feature/*", "dev", "hotfix/*"]
        )

        assert team.matches_branch("feature/auth")
        assert team.matches_branch("dev")
        assert team.matches_branch("hotfix/critical")
        assert not team.matches_branch("main")
        assert not team.matches_branch("bugfix/test")

    def test_prefix_wildcard(self):
        """Test prefix wildcard matching."""
        team = Team(
            id="TEAM-001",
            branch_patterns=["feature/*", "bug*"]
        )

        assert team.matches_branch("feature/foo")
        assert team.matches_branch("bugfix")
        assert team.matches_branch("bug123")
        assert team.matches_branch("bug")

    def test_question_mark_wildcard(self):
        """Test single-character wildcard (?)."""
        team = Team(
            id="TEAM-001",
            branch_patterns=["v?.?"]
        )

        assert team.matches_branch("v1.0")
        assert team.matches_branch("v2.5")
        assert not team.matches_branch("v10.0")  # Two digits don't match single ?

    def test_case_sensitive_match(self):
        """Test that branch matching is case-sensitive."""
        team = Team(
            id="TEAM-001",
            branch_patterns=["Feature/*"]
        )

        assert team.matches_branch("Feature/auth")
        assert not team.matches_branch("feature/auth")


class TestTeamGetSetting:
    """Test Team.get_setting() method."""

    def test_get_existing_setting(self):
        """Test retrieving existing setting."""
        team = Team(
            id="TEAM-001",
            settings={"foo": "bar", "count": 42}
        )

        assert team.get_setting("foo") == "bar"
        assert team.get_setting("count") == 42

    def test_get_missing_setting_returns_none(self):
        """Test that missing setting returns None by default."""
        team = Team(id="TEAM-001", settings={})

        assert team.get_setting("missing") is None

    def test_get_missing_setting_with_default(self):
        """Test missing setting returns provided default."""
        team = Team(id="TEAM-001", settings={})

        assert team.get_setting("missing", "default") == "default"
        assert team.get_setting("missing", 0) == 0
        assert team.get_setting("missing", []) == []

    def test_get_setting_with_none_value(self):
        """Test retrieving setting that is explicitly None."""
        team = Team(id="TEAM-001", settings={"value": None})

        assert team.get_setting("value") is None
        assert team.get_setting("value", "default") is None  # Explicit None, not default

    def test_nested_setting_access(self):
        """Test accessing nested setting values."""
        team = Team(
            id="TEAM-001",
            settings={
                "nested": {
                    "deep": "value"
                }
            }
        )

        nested = team.get_setting("nested")
        assert nested == {"deep": "value"}
        assert nested["deep"] == "value"


class TestTeamEntityType:
    """Test Team entity_type field."""

    def test_entity_type_is_team(self):
        """Test that entity_type is always 'team'."""
        team = Team(id="TEAM-001")
        assert team.entity_type == "team"

    def test_entity_type_in_serialization(self):
        """Test entity_type is included in serialization."""
        team = Team(id="TEAM-001")
        data = team.to_dict()
        assert data["entity_type"] == "team"

    def test_entity_type_from_deserialization(self):
        """Test entity_type is set correctly from deserialization."""
        data = {"id": "TEAM-001", "entity_type": "team"}
        team = Team.from_dict(data)
        assert team.entity_type == "team"

    def test_entity_type_override_in_init(self):
        """Test that entity_type is forced to 'team' even if different value provided."""
        # Note: entity_type is set in __post_init__, so it overrides any passed value
        team = Team(id="TEAM-001", entity_type="wrong")
        assert team.entity_type == "team"


class TestTeamVersioning:
    """Test Team versioning and timestamp methods."""

    def test_bump_version(self):
        """Test bump_version increments version and updates modified_at."""
        team = Team(id="TEAM-001")
        original_version = team.version
        original_modified = team.modified_at

        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.001)

        team.bump_version()

        assert team.version == original_version + 1
        assert team.modified_at != original_modified

    def test_multiple_version_bumps(self):
        """Test multiple version bumps."""
        team = Team(id="TEAM-001")
        assert team.version == 1

        team.bump_version()
        assert team.version == 2

        team.bump_version()
        assert team.version == 3

    def test_compute_checksum(self):
        """Test compute_checksum returns consistent hash."""
        team = Team(id="TEAM-001", name="Test")
        checksum1 = team.compute_checksum()
        checksum2 = team.compute_checksum()

        # Should be consistent
        assert checksum1 == checksum2
        assert isinstance(checksum1, str)
        assert len(checksum1) == 16  # First 16 chars of SHA256

    def test_checksum_changes_with_content(self):
        """Test checksum changes when team data changes."""
        team = Team(id="TEAM-001", name="Test")
        checksum1 = team.compute_checksum()

        team.name = "Different"
        checksum2 = team.compute_checksum()

        assert checksum1 != checksum2
