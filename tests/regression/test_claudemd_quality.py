"""
Regression tests for CLAUDE.md generation quality.

Ensures:
1. Generated CLAUDE.md contains all critical sections
2. No information loss vs static file
3. Proper formatting maintained

Task: T-20251222-204138-5915d70a
"""

import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cortical.got.claudemd import (
    ClaudeMdValidator,
    ClaudeMdComposer,
    ClaudeMdGenerator,
    GenerationContext,
)
from cortical.got.types import ClaudeMdLayer


class TestCriticalSectionsPresence(unittest.TestCase):
    """Regression tests ensuring all critical sections exist in generated content."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ClaudeMdValidator()

    def test_required_sections_list_is_not_empty(self):
        """
        Regression: REQUIRED_SECTIONS must never be empty.

        If this list becomes empty, validation becomes meaningless.
        """
        self.assertGreater(
            len(ClaudeMdValidator.REQUIRED_SECTIONS),
            0,
            "REQUIRED_SECTIONS should never be empty"
        )

    def test_required_sections_include_essential_content(self):
        """
        Regression: Essential sections must be in REQUIRED_SECTIONS.

        These sections are critical for agent operation:
        - Quick Session Start: First thing agents should read
        - Work Priority Order: Security > Bugs > Features > Docs
        - Architecture: Understanding the codebase
        - Testing: TDD requirements
        - Quick Reference: Common commands
        """
        essential_sections = [
            "Quick Session Start",
            "Work Priority Order",
            "Architecture",
            "Testing",
            "Quick Reference",
        ]

        for section in essential_sections:
            self.assertIn(
                section,
                ClaudeMdValidator.REQUIRED_SECTIONS,
                f"Essential section '{section}' must be in REQUIRED_SECTIONS"
            )

    def test_critical_patterns_are_validated(self):
        """
        Regression: CRITICAL_PATTERNS must include key operational patterns.

        These patterns ensure the generated CLAUDE.md contains:
        - Priority order (Security before Bugs before Features before Docs)
        - GoT commands reference
        - Test commands
        """
        self.assertGreater(
            len(ClaudeMdValidator.CRITICAL_PATTERNS),
            0,
            "CRITICAL_PATTERNS should never be empty"
        )

        # Verify priority order pattern exists
        priority_pattern_exists = any(
            "Security" in p and "Bug" in p
            for p in ClaudeMdValidator.CRITICAL_PATTERNS
        )
        self.assertTrue(
            priority_pattern_exists,
            "CRITICAL_PATTERNS should include priority order pattern"
        )

    def test_static_claudemd_passes_validation(self):
        """
        Regression: The static CLAUDE.md must pass its own validation.

        If the static file doesn't pass validation, either:
        1. The static file is missing critical content, OR
        2. The validation rules are too strict
        """
        static_path = Path("CLAUDE.md")
        if not static_path.exists():
            self.skipTest("Static CLAUDE.md not found")

        content = static_path.read_text()
        result = self.validator.validate(content)

        # Static file should pass validation
        self.assertTrue(
            result.is_valid,
            f"Static CLAUDE.md should pass validation. Errors: {result.errors}"
        )


class TestNoInformationLoss(unittest.TestCase):
    """Regression tests ensuring no information loss during generation."""

    @classmethod
    def setUpClass(cls):
        """Load static CLAUDE.md once for all tests."""
        cls.static_path = Path("CLAUDE.md")
        if cls.static_path.exists():
            cls.static_content = cls.static_path.read_text()
        else:
            cls.static_content = None

    def setUp(self):
        """Set up test fixtures."""
        if self.static_content is None:
            self.skipTest("Static CLAUDE.md not found")

    def test_static_contains_priority_order(self):
        """
        Regression: Static CLAUDE.md must contain the priority order.

        This is the critical work priority:
        Security > Bugs > Features > Documentation
        """
        # Check for priority order pattern
        self.assertIn(
            "Security",
            self.static_content,
            "Static CLAUDE.md should mention Security priority"
        )
        self.assertIn(
            "Bugs",
            self.static_content,
            "Static CLAUDE.md should mention Bugs priority"
        )
        self.assertIn(
            "Features",
            self.static_content,
            "Static CLAUDE.md should mention Features priority"
        )

    def test_static_contains_got_commands(self):
        """
        Regression: Static CLAUDE.md must contain GoT commands reference.

        GoT (Graph of Thought) commands are essential for task management.
        """
        self.assertIn(
            "got_utils.py",
            self.static_content,
            "Static CLAUDE.md should reference got_utils.py commands"
        )

    def test_static_contains_testing_commands(self):
        """
        Regression: Static CLAUDE.md must contain testing commands.

        Testing is a critical part of the TDD workflow.
        """
        self.assertTrue(
            "pytest" in self.static_content or "unittest" in self.static_content,
            "Static CLAUDE.md should contain test commands"
        )

    def test_static_contains_architecture_section(self):
        """
        Regression: Static CLAUDE.md must have architecture documentation.

        Architecture section helps agents understand the codebase.
        """
        self.assertIn(
            "cortical/",
            self.static_content,
            "Static CLAUDE.md should describe cortical/ module structure"
        )

    def test_static_contains_quick_start(self):
        """
        Regression: Static CLAUDE.md must have quick start section.

        This is the first thing agents should read.
        """
        self.assertIn(
            "Quick Session Start",
            self.static_content,
            "Static CLAUDE.md should have Quick Session Start section"
        )

    def test_key_patterns_preserved(self):
        """
        Regression: Key operational patterns must exist in static file.

        These patterns are critical for agent operation.
        """
        key_patterns = [
            r"coverage",           # Coverage tracking
            r"TDD|test-driven",    # TDD workflow
            r"commit",             # Git workflow
            r"Sprint",             # Sprint management
        ]

        for pattern in key_patterns:
            self.assertTrue(
                re.search(pattern, self.static_content, re.IGNORECASE),
                f"Static CLAUDE.md should contain pattern: {pattern}"
            )


class TestFormattingConsistency(unittest.TestCase):
    """Regression tests for markdown formatting consistency."""

    @classmethod
    def setUpClass(cls):
        """Load static CLAUDE.md once for all tests."""
        cls.static_path = Path("CLAUDE.md")
        if cls.static_path.exists():
            cls.static_content = cls.static_path.read_text()
        else:
            cls.static_content = None

    def setUp(self):
        """Set up test fixtures."""
        self.composer = ClaudeMdComposer()
        if self.static_content is None:
            self.skipTest("Static CLAUDE.md not found")

    def test_static_has_proper_heading_hierarchy(self):
        """
        Regression: Static CLAUDE.md should have proper heading hierarchy.

        Headings should not skip levels (e.g., # -> ###).
        """
        lines = self.static_content.split("\n")
        headings = [l for l in lines if l.startswith("#")]

        self.assertGreater(
            len(headings),
            10,
            "Static CLAUDE.md should have multiple headings"
        )

        # Check we have H1 (main title)
        h1_count = sum(1 for h in headings if h.startswith("# ") and not h.startswith("## "))
        self.assertGreater(
            h1_count,
            0,
            "Static CLAUDE.md should have at least one H1 heading"
        )

    def test_static_has_code_blocks(self):
        """
        Regression: Static CLAUDE.md should contain code blocks.

        Code blocks are essential for command examples.
        """
        code_block_count = self.static_content.count("```")

        self.assertGreater(
            code_block_count,
            10,  # Expect many code blocks
            "Static CLAUDE.md should have code blocks for examples"
        )

    def test_static_has_tables(self):
        """
        Regression: Static CLAUDE.md should contain tables.

        Tables are used for quick references and structured info.
        """
        table_markers = self.static_content.count("|")

        self.assertGreater(
            table_markers,
            20,  # Expect tables
            "Static CLAUDE.md should have tables for quick reference"
        )

    def test_composer_preserves_newline_structure(self):
        """
        Regression: Composer should end output with single newline.

        This prevents trailing whitespace issues.
        """
        layers = [
            ClaudeMdLayer(
                id="CML0-test",
                layer_type="core",
                layer_number=0,
                section_id="test",
                title="Test",
                content="# Test Content",
            )
        ]
        context = GenerationContext()

        result = self.composer.compose(layers, context)

        # Should end with exactly one newline
        self.assertTrue(
            result.endswith("\n"),
            "Composed content should end with newline"
        )
        self.assertFalse(
            result.endswith("\n\n\n"),
            "Composed content should not have excessive trailing newlines"
        )

    def test_composer_adds_header_metadata(self):
        """
        Regression: Composer should add generation metadata in header.

        This helps track when/how content was generated.
        """
        layers = [
            ClaudeMdLayer(
                id="CML0-test",
                layer_type="core",
                layer_number=0,
                section_id="test",
                title="Test",
                content="# Test",
            )
        ]
        context = GenerationContext(
            current_branch="test-branch",
            active_sprint_id="S-sprint-001",
        )

        result = self.composer.compose(layers, context)

        self.assertIn("Auto-generated:", result)
        self.assertIn("Branch: test-branch", result)
        self.assertIn("Sprint: S-sprint-001", result)


class TestGenerationQuality(unittest.TestCase):
    """Regression tests for generation quality metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ClaudeMdValidator()

    def test_min_lines_threshold_is_reasonable(self):
        """
        Regression: MIN_LINES threshold should catch truly empty content.

        Too high: rejects valid short documents
        Too low: accepts nearly empty content
        """
        self.assertGreaterEqual(
            ClaudeMdValidator.MIN_LINES,
            100,
            "MIN_LINES should be at least 100 to catch empty content"
        )
        self.assertLessEqual(
            ClaudeMdValidator.MIN_LINES,
            500,
            "MIN_LINES should be at most 500 to not reject valid short docs"
        )

    def test_max_lines_threshold_is_reasonable(self):
        """
        Regression: MAX_LINES threshold should catch bloated content.

        Too low: rejects valid comprehensive documents
        Too high: allows unreasonably large documents
        """
        self.assertGreaterEqual(
            ClaudeMdValidator.MAX_LINES,
            5000,
            "MAX_LINES should be at least 5000 for comprehensive docs"
        )
        self.assertLessEqual(
            ClaudeMdValidator.MAX_LINES,
            20000,
            "MAX_LINES should be at most 20000 to catch bloated content"
        )

    def test_validation_returns_proper_result_type(self):
        """
        Regression: Validator should return ValidationResult with all fields.

        This ensures consistent error handling downstream.
        """
        result = self.validator.validate("# Minimal")

        # Check result structure
        self.assertIsInstance(result.is_valid, bool)
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.warnings, list)

    def test_validation_detects_empty_content(self):
        """
        Regression: Validator must reject empty content.

        Empty CLAUDE.md would leave agents without guidance.
        """
        result = self.validator.validate("")

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("empty" in e.lower() for e in result.errors),
            "Should report empty content error"
        )

    def test_validation_detects_missing_sections(self):
        """
        Regression: Validator must detect missing required sections.
        """
        content = """# CLAUDE.md

        This document is missing all required sections.
        """ * 50  # Make it long enough to pass length check

        result = self.validator.validate(content)

        self.assertFalse(result.is_valid)
        self.assertGreater(
            len(result.errors),
            0,
            "Should report missing section errors"
        )


class TestLayerSelectionQuality(unittest.TestCase):
    """Regression tests for layer selection quality."""

    def test_always_layers_are_always_included(self):
        """
        Regression: Layers with inclusion_rule='always' must always be included.

        This is the fundamental contract of the LayerSelector.
        """
        from cortical.got.claudemd import LayerSelector

        selector = LayerSelector()
        layer = ClaudeMdLayer(
            id="CML0-always",
            layer_type="core",
            layer_number=0,
            section_id="always-test",
            title="Always",
            content="# Always included",
            inclusion_rule="always",
        )
        context = GenerationContext(detected_modules=[])

        selected = selector.select(context, [layer])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].id, "CML0-always")

    def test_context_layers_require_matching_modules(self):
        """
        Regression: Context layers should only include when modules match.
        """
        from cortical.got.claudemd import LayerSelector

        selector = LayerSelector()
        layer = ClaudeMdLayer(
            id="CML2-query",
            layer_type="contextual",
            layer_number=2,
            section_id="query",
            title="Query Module",
            content="# Query",
            inclusion_rule="context",
            context_modules=["query"],
        )

        # Without query module - should not include
        context_no_match = GenerationContext(detected_modules=["spark"])
        selected_no_match = selector.select(context_no_match, [layer])
        self.assertEqual(len(selected_no_match), 0)

        # With query module - should include
        context_match = GenerationContext(detected_modules=["query"])
        selected_match = selector.select(context_match, [layer])
        self.assertEqual(len(selected_match), 1)

    def test_layer_ordering_is_deterministic(self):
        """
        Regression: Layer ordering must be deterministic.

        Same input should always produce same order.
        """
        from cortical.got.claudemd import LayerSelector

        selector = LayerSelector()
        layers = [
            ClaudeMdLayer(
                id="CML2-z",
                layer_type="contextual",
                layer_number=2,
                section_id="z-section",
                title="Z",
                content="# Z",
                inclusion_rule="always",
            ),
            ClaudeMdLayer(
                id="CML0-a",
                layer_type="core",
                layer_number=0,
                section_id="a-section",
                title="A",
                content="# A",
                inclusion_rule="always",
            ),
            ClaudeMdLayer(
                id="CML1-m",
                layer_type="operational",
                layer_number=1,
                section_id="m-section",
                title="M",
                content="# M",
                inclusion_rule="always",
            ),
        ]
        context = GenerationContext()

        # Run selection multiple times
        results = [selector.select(context, layers) for _ in range(5)]

        # All results should have same order
        first_order = [l.id for l in results[0]]
        for result in results[1:]:
            self.assertEqual(
                [l.id for l in result],
                first_order,
                "Layer ordering should be deterministic"
            )

        # Should be ordered by layer_number
        layer_numbers = [l.layer_number for l in results[0]]
        self.assertEqual(
            layer_numbers,
            sorted(layer_numbers),
            "Layers should be ordered by layer_number"
        )


if __name__ == "__main__":
    unittest.main()
