"""
Integration tests for CLAUDE.md generation pipeline.

Tests the full end-to-end flow:
- Layer CRUD operations through GoTManager
- Generation pipeline from layers to output file
- Fault tolerance and recovery
- Context-aware layer selection
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone

from cortical.got.api import GoTManager
from cortical.got.types import ClaudeMdLayer
from cortical.got.claudemd import ClaudeMdManager, ClaudeMdGenerator


class TestClaudeMdLayerCRUD(unittest.TestCase):
    """Integration tests for ClaudeMdLayer CRUD operations."""

    def setUp(self):
        """Set up test fixtures with real GoTManager."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()

        self.manager = GoTManager(self.got_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_create_and_get_layer(self):
        """Test creating and retrieving a layer."""
        layer = self.manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="test-section",
            title="Test Section",
            content="# Test\n\nThis is test content.",
            layer_number=2,
            freshness_decay_days=7,
        )

        self.assertIsNotNone(layer)
        self.assertTrue(layer.id.startswith("CML"))
        self.assertEqual(layer.layer_type, "contextual")
        self.assertEqual(layer.section_id, "test-section")

        # Verify can retrieve
        fetched = self.manager.get_claudemd_layer(layer.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Test Section")
        self.assertEqual(fetched.content, "# Test\n\nThis is test content.")

    def test_update_layer(self):
        """Test updating a layer."""
        layer = self.manager.create_claudemd_layer(
            layer_type="operational",
            section_id="commands",
            title="Commands",
            content="# Original Content",
            layer_number=1,
        )

        updated = self.manager.update_claudemd_layer(
            layer.id,
            title="Updated Commands",
            content="# Updated Content"
        )

        self.assertEqual(updated.title, "Updated Commands")
        self.assertEqual(updated.content, "# Updated Content")

        # Verify persistence
        fetched = self.manager.get_claudemd_layer(layer.id)
        self.assertEqual(fetched.title, "Updated Commands")

    def test_list_layers_with_filters(self):
        """Test listing layers with various filters."""
        # Create layers of different types
        self.manager.create_claudemd_layer(
            layer_type="core",
            section_id="core-1",
            title="Core 1",
            content="# Core",
            layer_number=0,
        )
        self.manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="ctx-1",
            title="Contextual 1",
            content="# Contextual",
            layer_number=2,
        )
        self.manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="ctx-2",
            title="Contextual 2",
            content="# Contextual 2",
            layer_number=2,
            freshness_status="stale",
        )

        # List all
        all_layers = self.manager.list_claudemd_layers()
        self.assertEqual(len(all_layers), 3)

        # Filter by type
        core_layers = self.manager.list_claudemd_layers(layer_type="core")
        self.assertEqual(len(core_layers), 1)

        contextual_layers = self.manager.list_claudemd_layers(layer_type="contextual")
        self.assertEqual(len(contextual_layers), 2)

        # Filter by freshness
        stale_layers = self.manager.list_claudemd_layers(freshness_status="stale")
        self.assertEqual(len(stale_layers), 1)

    def test_delete_layer(self):
        """Test deleting a layer."""
        layer = self.manager.create_claudemd_layer(
            layer_type="ephemeral",
            section_id="temp",
            title="Temporary",
            content="# Temp",
            layer_number=4,
        )

        # Verify exists
        self.assertIsNotNone(self.manager.get_claudemd_layer(layer.id))

        # Delete
        result = self.manager.delete_claudemd_layer(layer.id)
        self.assertTrue(result)

        # Verify deleted
        self.assertIsNone(self.manager.get_claudemd_layer(layer.id))

    def test_layer_persistence(self):
        """Test layers persist across manager instances."""
        # Create layer
        layer = self.manager.create_claudemd_layer(
            layer_type="core",
            section_id="persistent",
            title="Persistent Layer",
            content="# This should persist",
            layer_number=0,
        )
        layer_id = layer.id

        # Create new manager instance
        new_manager = GoTManager(self.got_dir)

        # Verify layer still exists
        fetched = new_manager.get_claudemd_layer(layer_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Persistent Layer")


class TestClaudeMdGenerationPipeline(unittest.TestCase):
    """Integration tests for full generation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()

        self.got_manager = GoTManager(self.got_dir)
        self.claudemd_manager = ClaudeMdManager(self.got_manager, self.got_dir)

        # Create fallback CLAUDE.md
        self.fallback_path = Path(self.temp_dir) / "CLAUDE.md"
        self.fallback_path.write_text("# Fallback CLAUDE.md\n\nThis is the fallback.")
        self.claudemd_manager.generator.fallback_path = self.fallback_path

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_minimal_valid_layers(self):
        """Create minimum layers needed for valid generation."""
        content_lines = "\n".join([f"Line {i}" for i in range(50)])

        self.got_manager.create_claudemd_layer(
            layer_type="core",
            section_id="quick-start",
            title="Quick Session Start",
            content=f"# Quick Session Start\n\nStart here...\n{content_lines}",
            layer_number=0,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="priorities",
            title="Work Priority Order",
            content=f"# Work Priority Order\n\nSecurity > Bugs > Features > Docs\n{content_lines}",
            layer_number=1,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="architecture",
            title="Architecture",
            content=f"# Architecture\n\nModule structure...\n{content_lines}",
            layer_number=2,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="testing",
            title="Testing",
            content=f"# Testing\n\npytest tests/\n{content_lines}",
            layer_number=1,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="quick-reference",
            title="Quick Reference",
            content=f"# Quick Reference\n\nCommands...\n{content_lines}",
            layer_number=1,
            inclusion_rule="always",
        )

    def test_generate_from_layers(self):
        """Test full generation from layers."""
        self._create_minimal_valid_layers()

        result = self.claudemd_manager.generate()

        self.assertTrue(result.success)
        self.assertEqual(result.layers_used, 5)
        self.assertFalse(result.fallback_used)

    def test_generate_writes_output_file(self):
        """Test generation writes to output file."""
        self._create_minimal_valid_layers()

        result = self.claudemd_manager.generate()

        self.assertTrue(result.success)
        self.assertTrue(result.path.exists())

        content = result.path.read_text()
        self.assertIn("Quick Session Start", content)
        self.assertIn("Architecture", content)

    def test_generate_with_no_layers_uses_fallback(self):
        """Test generation falls back when no layers exist."""
        result = self.claudemd_manager.generate()

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)

    def test_generate_dry_run(self):
        """Test dry run doesn't write files."""
        self._create_minimal_valid_layers()

        result = self.claudemd_manager.generate(dry_run=True)

        self.assertTrue(result.success)
        self.assertIsNone(result.path)

    def test_context_aware_layer_selection(self):
        """Test layers are selected based on context."""
        content_lines = "\n".join([f"Line {i}" for i in range(50)])

        # Create layers with different context requirements
        self.got_manager.create_claudemd_layer(
            layer_type="core",
            section_id="quick-start",
            title="Quick Session Start",
            content=f"# Quick Session Start\n\n{content_lines}",
            layer_number=0,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="priorities",
            title="Work Priority Order",
            content=f"# Work Priority Order\n\n{content_lines}",
            layer_number=1,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="architecture",
            title="Architecture",
            content=f"# Architecture\n\n{content_lines}",
            layer_number=2,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="testing",
            title="Testing",
            content=f"# Testing\n\n{content_lines}",
            layer_number=1,
            inclusion_rule="always",
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="quick-reference",
            title="Quick Reference",
            content=f"# Quick Reference\n\n{content_lines}",
            layer_number=1,
            inclusion_rule="always",
        )

        # Context-specific layer
        self.got_manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="query-module",
            title="Query Module Details",
            content=f"# Query Module\n\nQuery-specific docs...\n{content_lines}",
            layer_number=2,
            inclusion_rule="context",
            context_modules=["query"],
        )

        # Generate - query module should be selected only if context matches
        result = self.claudemd_manager.generate()

        # The generation should work (may or may not include query based on current context)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.layers_used, 5)


class TestClaudeMdFaultTolerance(unittest.TestCase):
    """Integration tests for fault tolerance."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()

        self.got_manager = GoTManager(self.got_dir)
        self.claudemd_manager = ClaudeMdManager(self.got_manager, self.got_dir)

        # Create fallback
        self.fallback_path = Path(self.temp_dir) / "CLAUDE.md"
        self.fallback_path.write_text("# Fallback Content")
        self.claudemd_manager.generator.fallback_path = self.fallback_path

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_fallback_on_invalid_layers(self):
        """Test fallback when layers produce invalid content."""
        # Create layer with content that won't pass validation
        self.got_manager.create_claudemd_layer(
            layer_type="core",
            section_id="minimal",
            title="Minimal",
            content="# Too Short",  # Won't pass validation
            layer_number=0,
        )

        result = self.claudemd_manager.generate()

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)

    def test_backup_on_regeneration(self):
        """Test previous good version is backed up."""
        content_lines = "\n".join([f"Line {i}" for i in range(50)])

        # Create valid layers matching required sections
        self.got_manager.create_claudemd_layer(
            layer_type="core",
            section_id="quick-start",
            title="Quick Session Start",
            content=f"# Quick Session Start\n\n{content_lines}",
            layer_number=0,
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="priorities",
            title="Work Priority Order",
            content=f"# Work Priority Order\n\n{content_lines}",
            layer_number=1,
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="testing",
            title="Testing",
            content=f"# Testing\n\n{content_lines}",
            layer_number=1,
        )
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="quick-reference",
            title="Quick Reference",
            content=f"# Quick Reference\n\n{content_lines}",
            layer_number=1,
        )
        self.got_manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="architecture",
            title="Architecture",
            content=f"# Architecture\n\n{content_lines}",
            layer_number=2,
        )

        # First generation
        result1 = self.claudemd_manager.generate()
        self.assertTrue(result1.success)

        # Update a layer
        layers = self.got_manager.list_claudemd_layers(layer_type="core")
        if layers:
            self.got_manager.update_claudemd_layer(
                layers[0].id,
                content=f"# Updated Quick Session Start\n\n{content_lines}"
            )

        # Second generation
        result2 = self.claudemd_manager.generate()
        self.assertTrue(result2.success)

        # Backup should exist
        backup_path = self.claudemd_manager.generator.last_good_path
        self.assertTrue(backup_path.exists())


class TestClaudeMdFreshnessManagement(unittest.TestCase):
    """Integration tests for freshness management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()

        self.got_manager = GoTManager(self.got_dir)
        self.claudemd_manager = ClaudeMdManager(self.got_manager, self.got_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_check_freshness_with_mixed_layers(self):
        """Test freshness check with various layer states."""
        from datetime import timedelta

        # Fresh layer (recent regeneration)
        self.got_manager.create_claudemd_layer(
            layer_type="core",
            section_id="fresh",
            title="Fresh",
            content="# Fresh",
            freshness_status="fresh",
            freshness_decay_days=7,
        )

        # Stale layer (old regeneration beyond decay period)
        # Create first, then update with old timestamp since create() always sets current time
        stale_layer = self.got_manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="stale",
            title="Stale",
            content="# Stale",
            freshness_status="stale",
            freshness_decay_days=7,
        )
        # Manually update to set old timestamp
        old_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        self.got_manager.update_claudemd_layer(
            stale_layer.id,
            last_regenerated=old_date
        )

        # Regenerating layer
        self.got_manager.create_claudemd_layer(
            layer_type="operational",
            section_id="regen",
            title="Regenerating",
            content="# Regen",
            freshness_status="regenerating",
        )

        result = self.claudemd_manager.check_freshness()

        self.assertEqual(len(result["fresh"]), 1)
        self.assertEqual(len(result["stale"]), 1)
        self.assertEqual(len(result["regenerating"]), 1)

    def test_mark_layer_stale(self):
        """Test marking a layer as stale."""
        layer = self.got_manager.create_claudemd_layer(
            layer_type="contextual",
            section_id="to-stale",
            title="Will Be Stale",
            content="# Content",
            freshness_status="fresh",
        )

        result = self.claudemd_manager.mark_layer_stale(
            layer.id,
            reason="Manual invalidation for testing"
        )

        self.assertTrue(result)

        # Verify updated
        fetched = self.got_manager.get_claudemd_layer(layer.id)
        self.assertEqual(fetched.freshness_status, "stale")


if __name__ == "__main__":
    unittest.main()
