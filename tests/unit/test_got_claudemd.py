"""
Additional unit tests for cortical/got/claudemd.py to improve coverage.

This file focuses on uncovered edge cases and error paths:
- Exception handling in subprocess calls
- File I/O error handling
- Edge cases in layer selection
- File writing and backup logic
- Manager wrapper methods
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from cortical.got.claudemd import (
    ContextAnalyzer, LayerSelector, ClaudeMdComposer, ClaudeMdValidator,
    ClaudeMdGenerator, ClaudeMdManager,
    GenerationContext, GenerationResult, ValidationResult
)
from cortical.got.types import ClaudeMdLayer


class TestContextAnalyzerErrorHandling(unittest.TestCase):
    """Tests for ContextAnalyzer error handling paths."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()
        self.analyzer = ContextAnalyzer(self.got_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('subprocess.run')
    def test_get_current_branch_timeout_exception(self, mock_run):
        """Test branch detection handles timeout exception."""
        mock_run.side_effect = Exception("Timeout")
        branch = self.analyzer._get_current_branch()
        self.assertEqual(branch, "unknown")

    def test_get_active_sprint_id_with_no_entities_dir(self):
        """Test active sprint detection when entities dir doesn't exist."""
        # Remove entities dir
        shutil.rmtree(self.got_dir / "entities")
        sprint_id = self.analyzer._get_active_sprint_id()
        self.assertEqual(sprint_id, "")

    def test_get_active_sprint_id_with_sprint_files(self):
        """Test active sprint detection with actual sprint files."""
        entities_dir = self.got_dir / "entities"

        # Create inactive sprint
        inactive_sprint = entities_dir / "S-20251201-000000-aaaa.json"
        inactive_sprint.write_text(json.dumps({
            "data": {
                "id": "S-20251201-000000-aaaa",
                "status": "completed"
            }
        }))

        # Create active sprint
        active_sprint = entities_dir / "S-20251202-000000-bbbb.json"
        active_sprint.write_text(json.dumps({
            "data": {
                "id": "S-20251202-000000-bbbb",
                "status": "in_progress"
            }
        }))

        sprint_id = self.analyzer._get_active_sprint_id()
        self.assertEqual(sprint_id, "S-20251202-000000-bbbb")

    def test_get_active_sprint_id_with_exception(self):
        """Test active sprint detection handles file read errors."""
        entities_dir = self.got_dir / "entities"

        # Create malformed JSON file
        bad_sprint = entities_dir / "S-20251201-000000-bad.json"
        bad_sprint.write_text("not valid json {")

        sprint_id = self.analyzer._get_active_sprint_id()
        self.assertEqual(sprint_id, "")

    @patch('subprocess.run')
    def test_get_recently_touched_files_on_error(self, mock_run):
        """Test touched files detection handles git errors."""
        mock_run.return_value = Mock(returncode=1, stdout="")
        files = self.analyzer._get_recently_touched_files()
        self.assertEqual(files, [])

    @patch('subprocess.run')
    def test_get_recently_touched_files_with_exception(self, mock_run):
        """Test touched files detection handles exceptions."""
        mock_run.side_effect = Exception("Git error")
        files = self.analyzer._get_recently_touched_files()
        self.assertEqual(files, [])

    def test_get_in_progress_tasks_no_entities_dir(self):
        """Test task detection when entities dir doesn't exist."""
        shutil.rmtree(self.got_dir / "entities")
        tasks = self.analyzer._get_in_progress_tasks()
        self.assertEqual(tasks, [])

    def test_get_in_progress_tasks_with_tasks(self):
        """Test task detection with actual task files."""
        entities_dir = self.got_dir / "entities"

        # Create completed task
        completed_task = entities_dir / "T-20251201-000000-aaaa.json"
        completed_task.write_text(json.dumps({
            "data": {
                "id": "T-20251201-000000-aaaa",
                "status": "completed"
            }
        }))

        # Create in-progress task
        active_task = entities_dir / "T-20251202-000000-bbbb.json"
        active_task.write_text(json.dumps({
            "data": {
                "id": "T-20251202-000000-bbbb",
                "status": "in_progress"
            }
        }))

        tasks = self.analyzer._get_in_progress_tasks()
        self.assertIn("T-20251202-000000-bbbb", tasks)
        self.assertNotIn("T-20251201-000000-aaaa", tasks)

    def test_get_in_progress_tasks_with_exception(self):
        """Test task detection handles file read errors."""
        entities_dir = self.got_dir / "entities"

        # Create malformed JSON file
        bad_task = entities_dir / "T-20251201-000000-bad.json"
        bad_task.write_text("not valid json {")

        tasks = self.analyzer._get_in_progress_tasks()
        self.assertEqual(tasks, [])

    def test_load_user_preferences_with_file(self):
        """Test loading user preferences from file."""
        prefs_dir = self.got_dir / "claude-md"
        prefs_dir.mkdir()
        prefs_file = prefs_dir / "preferences.json"
        prefs_file.write_text(json.dumps({
            "include_persona": False,
            "include_ml-collection": True
        }))

        prefs = self.analyzer._load_user_preferences()
        self.assertEqual(prefs["include_persona"], False)
        self.assertEqual(prefs["include_ml-collection"], True)

    def test_load_user_preferences_no_file(self):
        """Test loading user preferences when file doesn't exist."""
        prefs = self.analyzer._load_user_preferences()
        self.assertEqual(prefs, {})

    def test_load_user_preferences_with_exception(self):
        """Test user preferences handles file read errors."""
        prefs_dir = self.got_dir / "claude-md"
        prefs_dir.mkdir()
        prefs_file = prefs_dir / "preferences.json"
        prefs_file.write_text("not valid json {")

        prefs = self.analyzer._load_user_preferences()
        self.assertEqual(prefs, {})


class TestLayerSelectorEdgeCases(unittest.TestCase):
    """Tests for LayerSelector edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = LayerSelector()

    def test_should_include_context_layer_with_no_modules(self):
        """Test context layer with no specific modules is always included."""
        layer = ClaudeMdLayer(
            id="CML2-generic-1",
            layer_type="contextual",
            layer_number=2,
            section_id="generic",
            title="Generic",
            content="# Generic",
            inclusion_rule="context",
            context_modules=None,  # No specific modules
        )
        ctx = GenerationContext(detected_modules=[])

        result = self.selector._should_include(layer, ctx)
        self.assertTrue(result)

    def test_should_include_user_pref_defaults_to_true(self):
        """Test user_pref without matching preference defaults to include."""
        layer = ClaudeMdLayer(
            id="CML0-test-1",
            layer_type="core",
            layer_number=0,
            section_id="test",
            title="Test",
            content="# Test",
            inclusion_rule="user_pref",
        )
        # Context without user_preferences for this layer
        ctx = GenerationContext(user_preferences={})

        result = self.selector._should_include(layer, ctx)
        # Should default to True when preference not specified
        self.assertTrue(result)


class TestClaudeMdComposerEdgeCases(unittest.TestCase):
    """Tests for ClaudeMdComposer edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.composer = ClaudeMdComposer()

    def test_compose_with_empty_content_layer(self):
        """Test composing with layer that has empty content."""
        layers = [
            ClaudeMdLayer(
                id="CML0-empty-1",
                layer_type="core",
                layer_number=0,
                section_id="empty",
                title="Empty",
                content="   ",  # Whitespace only
            ),
            ClaudeMdLayer(
                id="CML0-content-1",
                layer_type="core",
                layer_number=0,
                section_id="content",
                title="Content",
                content="# Real Content",
            )
        ]
        ctx = GenerationContext()

        result = self.composer.compose(layers, ctx)

        # Empty layer should be skipped
        self.assertNotIn("Empty", result)
        self.assertIn("Real Content", result)

    def test_compose_with_content_needing_spacing(self):
        """Test compose adds proper spacing between layers."""
        layers = [
            ClaudeMdLayer(
                id="CML0-first-1",
                layer_type="core",
                layer_number=0,
                section_id="first",
                title="First",
                content="# First",  # No trailing newlines
            ),
            ClaudeMdLayer(
                id="CML0-second-1",
                layer_type="core",
                layer_number=0,
                section_id="second",
                title="Second",
                content="# Second",
            )
        ]
        ctx = GenerationContext()

        result = self.composer.compose(layers, ctx)

        # Should have proper spacing
        self.assertIn("# First\n\n# Second", result)


class TestClaudeMdValidatorEdgeCases(unittest.TestCase):
    """Tests for ClaudeMdValidator edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ClaudeMdValidator()

    def test_validate_content_with_few_headings(self):
        """Test validation warns about content with few headings."""
        # Content with required sections but few headings
        content = """
# Quick Session Start

## Work Priority Order

Architecture section here without heading.
Testing section here without heading.
Quick Reference here without heading.
""" + ("Line of content\n" * 250)  # Make it long enough

        result = self.validator.validate(content)

        # Should have warning about few headings
        self.assertTrue(any("headings" in w.lower() for w in result.warnings))

    def test_validate_content_too_long(self):
        """Test validation warns about very long content."""
        content = """# CLAUDE.md

## Quick Session Start
Start

## Work Priority Order
Order

## Architecture
Arch

## Testing
Test

## Quick Reference
Ref
""" + ("Line of content\n" * 15000)  # Exceed MAX_LINES

        result = self.validator.validate(content)

        # Should warn about length
        self.assertTrue(any("long" in w.lower() for w in result.warnings))


class TestClaudeMdGeneratorPaths(unittest.TestCase):
    """Tests for ClaudeMdGenerator uncovered paths."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()

        self.mock_got = Mock()
        self.generator = ClaudeMdGenerator(self.mock_got, self.got_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_generate_with_no_selected_layers(self):
        """Test generation when no layers are selected by selector."""
        # Create layers that won't be selected (user_pref with False)
        layers = [
            ClaudeMdLayer(
                id="CML3-persona-1",
                layer_type="persona",
                layer_number=3,
                section_id="persona",
                title="Persona",
                content="# Persona",
                inclusion_rule="user_pref",
            )
        ]
        self.mock_got.list_claudemd_layers.return_value = layers

        # Mock context analyzer to return preferences that exclude the layer
        with patch.object(
            self.generator.analyzer,
            'analyze',
            return_value=GenerationContext(
                user_preferences={"include_persona": False}
            )
        ):
            # Create fallback
            fallback = Path(self.temp_dir) / "CLAUDE.md"
            fallback.write_text("# Fallback")
            self.generator.fallback_path = fallback

            result = self.generator.generate()

            self.assertFalse(result.success)
            self.assertTrue(result.fallback_used)
            self.assertIn("No layers selected", result.error)

    def test_generate_with_validation_failure(self):
        """Test generation when validation fails."""
        # Create layers that will compose to invalid content
        layers = [
            ClaudeMdLayer(
                id="CML0-tiny-1",
                layer_type="core",
                layer_number=0,
                section_id="tiny",
                title="Tiny",
                content="# Too Short",  # Will fail validation
                inclusion_rule="always",
            )
        ]
        self.mock_got.list_claudemd_layers.return_value = layers

        # Create fallback
        fallback = Path(self.temp_dir) / "CLAUDE.md"
        fallback.write_text("# Fallback")
        self.generator.fallback_path = fallback

        result = self.generator.generate()

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertIn("Validation failed", result.error)

    def test_generate_writes_output_when_not_dry_run(self):
        """Test generation writes to disk when not dry run."""
        # Create valid layers
        layers = [
            ClaudeMdLayer(
                id="CML0-start-1",
                layer_type="core",
                layer_number=0,
                section_id="quick-start",
                title="Quick Session Start",
                content="# Quick Session Start\n\n" + ("Content\n" * 50),
                inclusion_rule="always",
            ),
            ClaudeMdLayer(
                id="CML1-workflow-1",
                layer_type="operational",
                layer_number=1,
                section_id="workflow",
                title="Work Priority Order",
                content="# Work Priority Order\n\n" + ("Content\n" * 50),
                inclusion_rule="always",
            ),
            ClaudeMdLayer(
                id="CML2-arch-1",
                layer_type="contextual",
                layer_number=2,
                section_id="architecture",
                title="Architecture",
                content="# Architecture\n\n" + ("Content\n" * 50),
                inclusion_rule="always",
            ),
            ClaudeMdLayer(
                id="CML1-test-1",
                layer_type="operational",
                layer_number=1,
                section_id="testing",
                title="Testing",
                content="# Testing\n\n" + ("Content\n" * 50),
                inclusion_rule="always",
            ),
            ClaudeMdLayer(
                id="CML1-ref-1",
                layer_type="operational",
                layer_number=1,
                section_id="quick-reference",
                title="Quick Reference",
                content="# Quick Reference\n\n" + ("Content\n" * 50),
                inclusion_rule="always",
            ),
        ]
        self.mock_got.list_claudemd_layers.return_value = layers

        result = self.generator.generate(dry_run=False)

        self.assertTrue(result.success)
        self.assertTrue(self.generator.output_path.exists())
        self.assertTrue(self.generator.hash_path.exists())

    def test_use_fallback_when_output_exists(self):
        """Test _use_fallback when output already exists."""
        # Create existing output
        self.generator.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator.output_path.write_text("# Existing")

        result = self.generator._use_fallback("Test reason")

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.path, self.generator.output_path)

    def test_write_with_backup_creates_backup(self):
        """Test _write_with_backup backs up existing valid content."""
        # Create existing valid output
        self.generator.output_dir.mkdir(parents=True, exist_ok=True)

        valid_content = """# CLAUDE.md

## Quick Session Start
Start

## Work Priority Order
Order

## Architecture
Arch

## Testing
Test

## Quick Reference
Ref
""" + ("Content\n" * 250)

        self.generator.output_path.write_text(valid_content)

        # Write new content
        new_content = valid_content + "\n# New Section"
        self.generator._write_with_backup(new_content)

        # Should have backed up old content
        self.assertTrue(self.generator.last_good_path.exists())
        backup_content = self.generator.last_good_path.read_text()
        self.assertEqual(backup_content, valid_content)

        # Should have new content
        current_content = self.generator.output_path.read_text()
        self.assertIn("# New Section", current_content)

    def test_copy_fallback(self):
        """Test _copy_fallback copies original CLAUDE.md."""
        # Create fallback file
        fallback = Path(self.temp_dir) / "CLAUDE.md"
        fallback.write_text("# Original CLAUDE.md")
        self.generator.fallback_path = fallback

        self.generator._copy_fallback()

        self.assertTrue(self.generator.output_path.exists())
        content = self.generator.output_path.read_text()
        self.assertEqual(content, "# Original CLAUDE.md")


class TestClaudeMdManagerWrapperMethods(unittest.TestCase):
    """Tests for ClaudeMdManager wrapper methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"
        self.got_dir.mkdir()
        (self.got_dir / "entities").mkdir()

        self.mock_got = Mock()
        self.mock_got.list_claudemd_layers = Mock(return_value=[])
        self.mock_got.create_claudemd_layer = Mock()
        self.mock_got.get_claudemd_layer = Mock()
        self.mock_got.update_claudemd_layer = Mock()
        self.mock_got.delete_claudemd_layer = Mock(return_value=True)

        self.manager = ClaudeMdManager(self.mock_got, self.got_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_update_layer_wrapper(self):
        """Test update_layer delegates to GoT manager."""
        self.manager.update_layer("CML0-test-1", content="New content")
        self.mock_got.update_claudemd_layer.assert_called_once_with(
            "CML0-test-1", content="New content"
        )

    def test_list_layers_with_filters(self):
        """Test list_layers passes filters to GoT manager."""
        self.manager.list_layers(
            layer_type="contextual",
            freshness_status="stale"
        )
        self.mock_got.list_claudemd_layers.assert_called_once_with(
            layer_type="contextual",
            freshness_status="stale"
        )

    def test_delete_layer_wrapper(self):
        """Test delete_layer delegates to GoT manager."""
        result = self.manager.delete_layer("CML0-test-1")
        self.mock_got.delete_claudemd_layer.assert_called_once_with("CML0-test-1")
        self.assertTrue(result)

    def test_mark_layer_stale(self):
        """Test mark_layer_stale updates layer status."""
        # Create mock layer
        mock_layer = ClaudeMdLayer(
            id="CML0-test-1",
            layer_type="core",
            layer_number=0,
            section_id="test",
            title="Test",
            content="Content",
        )
        self.mock_got.get_claudemd_layer.return_value = mock_layer

        result = self.manager.mark_layer_stale("CML0-test-1", reason="Test reason")

        self.assertTrue(result)
        self.mock_got.get_claudemd_layer.assert_called_once_with("CML0-test-1")
        self.mock_got.update_claudemd_layer.assert_called_once_with(
            "CML0-test-1",
            freshness_status="stale",
            regeneration_trigger="Test reason"
        )

    def test_mark_layer_stale_with_nonexistent_layer(self):
        """Test mark_layer_stale returns False for nonexistent layer."""
        self.mock_got.get_claudemd_layer.return_value = None

        result = self.manager.mark_layer_stale("CML0-nonexistent-1")

        self.assertFalse(result)

    def test_get_output_path(self):
        """Test get_output_path returns correct path."""
        path = self.manager.get_output_path()
        self.assertEqual(path, self.manager.generator.output_path)

    def test_get_fallback_path(self):
        """Test get_fallback_path returns correct path."""
        path = self.manager.get_fallback_path()
        self.assertEqual(path, self.manager.generator.fallback_path)

    def test_generate_wrapper(self):
        """Test generate delegates to generator."""
        # Create fallback for when no layers exist
        fallback = Path(self.temp_dir) / "CLAUDE.md"
        fallback.write_text("# Fallback")
        self.manager.generator.fallback_path = fallback

        result = self.manager.generate(dry_run=True)

        self.assertIsInstance(result, GenerationResult)
        self.mock_got.list_claudemd_layers.assert_called_once()


if __name__ == "__main__":
    unittest.main()
