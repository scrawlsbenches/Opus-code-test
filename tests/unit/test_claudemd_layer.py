"""
Unit tests for ClaudeMdLayer and ClaudeMdVersion entity types.

Tests cover:
- Entity instantiation and validation
- Serialization/deserialization
- Freshness tracking and staleness detection
- Content hashing
- Version snapshots
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from cortical.got.types import ClaudeMdLayer, ClaudeMdVersion
from cortical.got.errors import ValidationError


class TestClaudeMdLayerInstantiation(unittest.TestCase):
    """Tests for ClaudeMdLayer creation and validation."""

    def test_create_minimal_layer(self):
        """Test creating layer with minimal required fields."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test Layer",
            content="# Test"
        )
        self.assertEqual(layer.id, "CML0-test-123")
        self.assertEqual(layer.entity_type, "claudemd_layer")
        self.assertEqual(layer.layer_type, "core")
        self.assertEqual(layer.freshness_status, "fresh")

    def test_create_full_layer(self):
        """Test creating layer with all fields."""
        layer = ClaudeMdLayer(
            id="CML2-arch-456",
            layer_type="contextual",
            layer_number=2,
            section_id="architecture",
            title="Architecture Guide",
            content="# Architecture\n\nDetails here...",
            freshness_status="fresh",
            freshness_decay_days=7,
            inclusion_rule="context",
            context_modules=["query", "analysis"],
            context_branches=["main", "develop"],
            properties={"author": "system"},
            metadata={"importance": "high"},
        )
        self.assertEqual(layer.layer_number, 2)
        self.assertEqual(layer.freshness_decay_days, 7)
        self.assertEqual(layer.inclusion_rule, "context")
        self.assertIn("query", layer.context_modules)

    def test_valid_layer_types(self):
        """Test all valid layer types are accepted."""
        valid_types = ["core", "operational", "contextual", "persona", "ephemeral", ""]
        for lt in valid_types:
            layer = ClaudeMdLayer(
                id=f"CML0-{lt or 'empty'}-123",
                layer_type=lt,
                section_id="test",
                title="Test",
                content="Content"
            )
            self.assertEqual(layer.layer_type, lt)

    def test_invalid_layer_type_raises_error(self):
        """Test invalid layer type raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            ClaudeMdLayer(
                id="CML0-test-123",
                layer_type="invalid_type",
                section_id="test",
                title="Test",
                content="Content"
            )
        self.assertIn("invalid_type", str(ctx.exception))

    def test_valid_freshness_statuses(self):
        """Test all valid freshness statuses are accepted."""
        valid_statuses = ["fresh", "stale", "regenerating"]
        for status in valid_statuses:
            layer = ClaudeMdLayer(
                id=f"CML0-{status}-123",
                layer_type="core",
                freshness_status=status,
                section_id="test",
                title="Test",
                content="Content"
            )
            self.assertEqual(layer.freshness_status, status)

    def test_invalid_freshness_status_raises_error(self):
        """Test invalid freshness status raises ValidationError."""
        with self.assertRaises(ValidationError):
            ClaudeMdLayer(
                id="CML0-test-123",
                layer_type="core",
                freshness_status="invalid",
                section_id="test",
                title="Test",
                content="Content"
            )

    def test_valid_inclusion_rules(self):
        """Test all valid inclusion rules are accepted."""
        valid_rules = ["always", "context", "user_pref"]
        for rule in valid_rules:
            layer = ClaudeMdLayer(
                id=f"CML0-{rule}-123",
                layer_type="core",
                inclusion_rule=rule,
                section_id="test",
                title="Test",
                content="Content"
            )
            self.assertEqual(layer.inclusion_rule, rule)

    def test_invalid_inclusion_rule_raises_error(self):
        """Test invalid inclusion rule raises ValidationError."""
        with self.assertRaises(ValidationError):
            ClaudeMdLayer(
                id="CML0-test-123",
                layer_type="core",
                inclusion_rule="invalid_rule",
                section_id="test",
                title="Test",
                content="Content"
            )

    def test_valid_layer_numbers(self):
        """Test valid layer numbers 0-4 are accepted."""
        for num in range(5):
            layer = ClaudeMdLayer(
                id=f"CML{num}-test-123",
                layer_type="core",
                layer_number=num,
                section_id="test",
                title="Test",
                content="Content"
            )
            self.assertEqual(layer.layer_number, num)

    def test_invalid_layer_number_raises_error(self):
        """Test invalid layer numbers raise ValidationError."""
        for num in [-1, 5, 10, 100]:
            with self.assertRaises(ValidationError):
                ClaudeMdLayer(
                    id=f"CML{num}-test-123",
                    layer_type="core",
                    layer_number=num,
                    section_id="test",
                    title="Test",
                    content="Content"
                )


class TestClaudeMdLayerSerialization(unittest.TestCase):
    """Tests for ClaudeMdLayer serialization/deserialization."""

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all layer fields."""
        layer = ClaudeMdLayer(
            id="CML2-arch-123",
            layer_type="contextual",
            layer_number=2,
            section_id="architecture",
            title="Architecture",
            content="# Architecture",
            freshness_decay_days=7,
            inclusion_rule="context",
            context_modules=["query"],
        )
        d = layer.to_dict()

        self.assertEqual(d["id"], "CML2-arch-123")
        self.assertEqual(d["entity_type"], "claudemd_layer")
        self.assertEqual(d["layer_type"], "contextual")
        self.assertEqual(d["layer_number"], 2)
        self.assertEqual(d["section_id"], "architecture")
        self.assertEqual(d["freshness_decay_days"], 7)
        self.assertIn("query", d["context_modules"])

    def test_from_dict_roundtrip(self):
        """Test from_dict correctly deserializes a layer."""
        original = ClaudeMdLayer(
            id="CML1-ops-456",
            layer_type="operational",
            layer_number=1,
            section_id="commands",
            title="Commands",
            content="# Commands\n\nList of commands...",
            freshness_status="stale",
            freshness_decay_days=30,
            context_modules=["got", "reasoning"],
        )

        d = original.to_dict()
        restored = ClaudeMdLayer.from_dict(d)

        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.layer_type, original.layer_type)
        self.assertEqual(restored.content, original.content)
        self.assertEqual(restored.freshness_status, original.freshness_status)
        self.assertEqual(restored.context_modules, original.context_modules)

    def test_from_dict_with_defaults(self):
        """Test from_dict handles missing optional fields."""
        minimal = {
            "id": "CML0-min-789",
            "entity_type": "claudemd_layer",
        }
        layer = ClaudeMdLayer.from_dict(minimal)

        self.assertEqual(layer.id, "CML0-min-789")
        self.assertEqual(layer.layer_type, "")
        self.assertEqual(layer.freshness_status, "fresh")
        self.assertEqual(layer.inclusion_rule, "always")


class TestClaudeMdLayerFreshness(unittest.TestCase):
    """Tests for freshness tracking and staleness detection."""

    def test_is_stale_never_decays(self):
        """Test layer with decay_days=0 never becomes stale."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Content",
            freshness_decay_days=0,
            last_regenerated=(datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
        )
        self.assertFalse(layer.is_stale())

    def test_is_stale_without_last_regenerated(self):
        """Test layer without last_regenerated is stale if has decay."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Content",
            freshness_decay_days=7,
            last_regenerated="",
        )
        self.assertTrue(layer.is_stale())

    def test_is_stale_within_decay_period(self):
        """Test layer within decay period is not stale."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Content",
            freshness_decay_days=7,
            last_regenerated=datetime.now(timezone.utc).isoformat(),
        )
        self.assertFalse(layer.is_stale())

    def test_is_stale_past_decay_period(self):
        """Test layer past decay period is stale."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Content",
            freshness_decay_days=7,
            last_regenerated=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        )
        self.assertTrue(layer.is_stale())

    def test_mark_stale(self):
        """Test mark_stale updates status and reason."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Content",
        )
        original_version = layer.version

        layer.mark_stale("Manual invalidation")

        self.assertEqual(layer.freshness_status, "stale")
        self.assertEqual(layer.regeneration_trigger, "Manual invalidation")
        self.assertEqual(layer.version, original_version + 1)

    def test_mark_fresh(self):
        """Test mark_fresh updates status, timestamp, and hash."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Test content",
            freshness_status="stale",
        )

        layer.mark_fresh()

        self.assertEqual(layer.freshness_status, "fresh")
        self.assertNotEqual(layer.last_regenerated, "")
        self.assertNotEqual(layer.content_hash, "")


class TestClaudeMdLayerContentHash(unittest.TestCase):
    """Tests for content hashing."""

    def test_compute_content_hash(self):
        """Test content hash is computed correctly."""
        layer = ClaudeMdLayer(
            id="CML0-test-123",
            layer_type="core",
            section_id="test",
            title="Test",
            content="Test content for hashing",
        )

        hash1 = layer.compute_content_hash()
        self.assertEqual(len(hash1), 16)  # First 16 chars of SHA256

        # Same content = same hash
        hash2 = layer.compute_content_hash()
        self.assertEqual(hash1, hash2)

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        layer1 = ClaudeMdLayer(
            id="CML0-test-1",
            layer_type="core",
            section_id="test1",
            title="Test 1",
            content="Content A",
        )
        layer2 = ClaudeMdLayer(
            id="CML0-test-2",
            layer_type="core",
            section_id="test2",
            title="Test 2",
            content="Content B",
        )

        self.assertNotEqual(
            layer1.compute_content_hash(),
            layer2.compute_content_hash()
        )


class TestClaudeMdVersion(unittest.TestCase):
    """Tests for ClaudeMdVersion entity."""

    def test_create_version(self):
        """Test creating a version snapshot."""
        version = ClaudeMdVersion(
            id="CMV-CML3-persona-123-v1",
            layer_id="CML3-persona-123",
            version_number=1,
            content_snapshot="# Persona\n\nWorking philosophy...",
            change_rationale="Initial version",
            changed_by="system",
        )

        self.assertEqual(version.entity_type, "claudemd_version")
        self.assertEqual(version.layer_id, "CML3-persona-123")
        self.assertEqual(version.version_number, 1)

    def test_auto_generate_id(self):
        """Test auto-generation of version ID."""
        version = ClaudeMdVersion(
            id="",  # Empty, should auto-generate
            layer_id="CML3-persona-456",
            version_number=3,
            content_snapshot="Content",
        )

        self.assertEqual(version.id, "CMV-CML3-persona-456-v3")

    def test_version_serialization(self):
        """Test version serialization roundtrip."""
        original = ClaudeMdVersion(
            id="CMV-CML3-persona-789-v2",
            layer_id="CML3-persona-789",
            version_number=2,
            content_snapshot="# Updated Persona",
            change_rationale="Style update",
            changed_by="user",
            changed_sections=["intro", "principles"],
            additions=10,
            deletions=5,
        )

        d = original.to_dict()
        restored = ClaudeMdVersion.from_dict(d)

        self.assertEqual(restored.layer_id, original.layer_id)
        self.assertEqual(restored.version_number, original.version_number)
        self.assertEqual(restored.change_rationale, original.change_rationale)
        self.assertEqual(restored.additions, original.additions)
        self.assertEqual(restored.deletions, original.deletions)


if __name__ == "__main__":
    unittest.main()
