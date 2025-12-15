"""
Unit tests for scripts/new_memory.py.

Tests memory and decision record creation utilities including:
- Filename generation with merge-safe timestamps
- Slugification of titles
- Template generation for memories and decisions
- File creation and dry-run mode
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from new_memory import (
    generate_memory_filename,
    slugify,
    get_git_author,
    create_memory_template,
    create_decision_template,
    create_memory,
    MEMORIES_DIR,
    DECISIONS_DIR,
)


class TestSlugify(unittest.TestCase):
    """Tests for slugify function."""

    def test_converts_spaces_to_hyphens(self):
        """Spaces should be converted to hyphens."""
        self.assertEqual(slugify("hello world"), "hello-world")
        self.assertEqual(slugify("foo bar baz"), "foo-bar-baz")

    def test_removes_special_characters(self):
        """Special characters should be removed."""
        self.assertEqual(slugify("hello!world"), "helloworld")
        self.assertEqual(slugify("test@#$%case"), "testcase")
        self.assertEqual(slugify("a/b\\c*d?e"), "abcde")

    def test_lowercases_text(self):
        """Text should be lowercased."""
        self.assertEqual(slugify("HELLO"), "hello")
        self.assertEqual(slugify("CamelCase"), "camelcase")
        self.assertEqual(slugify("MiXeD CaSe"), "mixed-case")

    def test_handles_empty_string(self):
        """Empty string should return empty string."""
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify("   "), "")

    def test_removes_duplicate_hyphens(self):
        """Duplicate hyphens should be collapsed."""
        self.assertEqual(slugify("foo--bar"), "foo-bar")
        self.assertEqual(slugify("a   b"), "a-b")
        self.assertEqual(slugify("hello---world"), "hello-world")

    def test_truncates_long_strings(self):
        """Strings longer than 50 chars should be truncated."""
        long_text = "a" * 100
        result = slugify(long_text)
        self.assertEqual(len(result), 50)

    def test_handles_unicode(self):
        """Unicode characters should be handled gracefully."""
        # Python's isalnum() preserves unicode letters
        self.assertEqual(slugify("café"), "café")
        self.assertEqual(slugify("naïve"), "naïve")
        # Only non-alphanumeric symbols are removed
        self.assertEqual(slugify("hello!世界"), "hello世界")

    def test_strips_leading_trailing_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        self.assertEqual(slugify("  hello  "), "hello")
        self.assertEqual(slugify("\tfoo bar\n"), "foo-bar")

    def test_real_world_examples(self):
        """Test real-world memory title examples."""
        self.assertEqual(
            slugify("learned about NaN validation"),
            "learned-about-nan-validation"
        )
        self.assertEqual(
            slugify("add microseconds to timestamps"),
            "add-microseconds-to-timestamps"
        )
        self.assertEqual(
            slugify("Why we chose PostgreSQL"),
            "why-we-chose-postgresql"
        )


class TestGenerateMemoryFilename(unittest.TestCase):
    """Tests for generate_memory_filename function."""

    @patch('new_memory.generate_session_id')
    @patch('new_memory.datetime')
    def test_correct_format(self, mock_datetime, mock_session_id):
        """Filename should have format YYYY-MM-DD_HH-MM-SS_XXXX-topic.md."""
        # Mock datetime
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        # Mock session ID
        mock_session_id.return_value = "a1b2"

        result = generate_memory_filename("test topic")
        self.assertEqual(result, "2025-12-14_14-30-52_a1b2-test-topic.md")

    @patch('new_memory.generate_session_id')
    @patch('new_memory.datetime')
    def test_handles_empty_topic(self, mock_datetime, mock_session_id):
        """Empty topic should still generate valid filename."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now
        mock_session_id.return_value = "a1b2"

        result = generate_memory_filename("")
        self.assertEqual(result, "2025-12-14_14-30-52_a1b2-.md")

    @patch('new_memory.generate_session_id')
    @patch('new_memory.datetime')
    def test_handles_special_characters_in_topic(self, mock_datetime, mock_session_id):
        """Special characters in topic should be slugified."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now
        mock_session_id.return_value = "a1b2"

        result = generate_memory_filename("Test! @#$ Topic?")
        self.assertEqual(result, "2025-12-14_14-30-52_a1b2-test-topic.md")

    @patch('new_memory.generate_session_id')
    @patch('new_memory.datetime')
    def test_decision_flag_ignored_in_filename(self, mock_datetime, mock_session_id):
        """is_decision parameter doesn't affect filename format."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now
        mock_session_id.return_value = "a1b2"

        # Decision flag doesn't change filename format
        result1 = generate_memory_filename("test", is_decision=False)
        result2 = generate_memory_filename("test", is_decision=True)
        self.assertEqual(result1, result2)


class TestGetGitAuthor(unittest.TestCase):
    """Tests for get_git_author function."""

    @patch('new_memory.subprocess.run')
    def test_returns_author_on_success(self, mock_run):
        """Should return git user.name when command succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "John Doe\n"
        mock_run.return_value = mock_result

        result = get_git_author()
        self.assertEqual(result, "John Doe")
        mock_run.assert_called_once_with(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2
        )

    @patch('new_memory.subprocess.run')
    def test_returns_unknown_on_failure(self, mock_run):
        """Should return 'Unknown' when git command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = get_git_author()
        self.assertEqual(result, "Unknown")

    @patch('new_memory.subprocess.run')
    def test_handles_timeout(self, mock_run):
        """Should return 'Unknown' on timeout."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("git", 2)

        result = get_git_author()
        self.assertEqual(result, "Unknown")

    @patch('new_memory.subprocess.run')
    def test_handles_file_not_found(self, mock_run):
        """Should return 'Unknown' if git not installed."""
        mock_run.side_effect = FileNotFoundError()

        result = get_git_author()
        self.assertEqual(result, "Unknown")


class TestCreateMemoryTemplate(unittest.TestCase):
    """Tests for create_memory_template function."""

    @patch('new_memory.datetime')
    def test_contains_required_sections(self, mock_datetime):
        """Template should contain all required memory sections."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        template = create_memory_template("test topic")

        # Check for required sections
        self.assertIn("# Memory Entry:", template)
        self.assertIn("**Tags:**", template)
        self.assertIn("**Related:**", template)
        self.assertIn("## Context", template)
        self.assertIn("## What I Learned", template)
        self.assertIn("## Connections Made", template)
        self.assertIn("## Emotional State", template)
        self.assertIn("## Future Exploration", template)
        self.assertIn("## Artifacts Created", template)

    @patch('new_memory.datetime')
    def test_includes_title(self, mock_datetime):
        """Template should include the provided title."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        template = create_memory_template("learning about pagerank")
        self.assertIn("Learning About Pagerank", template)

    @patch('new_memory.datetime')
    def test_includes_tags_when_provided(self, mock_datetime):
        """Template should include formatted tags."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        template = create_memory_template("test", tags="testing,validation,bugfix")
        self.assertIn("`testing`", template)
        self.assertIn("`validation`", template)
        self.assertIn("`bugfix`", template)

    @patch('new_memory.datetime')
    def test_empty_tags_when_not_provided(self, mock_datetime):
        """Template should have empty tags section when no tags provided."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        template = create_memory_template("test", tags="")
        # Should have Tags: line but empty
        self.assertIn("**Tags:**", template)
        # No backtick-wrapped tags
        self.assertNotIn("`test`", template)

    @patch('new_memory.datetime')
    def test_includes_timestamp(self, mock_datetime):
        """Template should include commit timestamp."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        template = create_memory_template("test")
        self.assertIn("*Committed to memory at: 2025-12-14T14:30:52Z*", template)

    @patch('new_memory.datetime')
    def test_author_parameter_not_used(self, mock_datetime):
        """Author parameter exists but is not used in template."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        # Author parameter accepted but not used
        template = create_memory_template("test", author="John Doe")
        self.assertNotIn("John Doe", template)


class TestCreateDecisionTemplate(unittest.TestCase):
    """Tests for create_decision_template function."""

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_contains_adr_sections(self, mock_decisions_dir, mock_datetime):
        """Template should contain ADR sections."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now

        # Mock no existing ADR files
        mock_decisions_dir.glob.return_value = []

        template = create_decision_template("test decision")

        # Check for ADR sections
        self.assertIn("ADR-001:", template)
        self.assertIn("**Status:**", template)
        self.assertIn("**Date:**", template)
        self.assertIn("**Deciders:**", template)
        self.assertIn("## Context and Problem Statement", template)
        self.assertIn("## Decision Drivers", template)
        self.assertIn("## Considered Options", template)
        self.assertIn("## Decision Outcome", template)
        self.assertIn("## Implementation", template)
        self.assertIn("## Consequences", template)
        self.assertIn("## Validation", template)
        self.assertIn("## Related Decisions", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_adr_number_starts_at_001(self, mock_decisions_dir, mock_datetime):
        """First ADR should be numbered 001."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now

        # No existing files
        mock_decisions_dir.glob.return_value = []

        template = create_decision_template("test")
        self.assertIn("# ADR-001:", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_adr_number_increments(self, mock_decisions_dir, mock_datetime):
        """ADR number should increment based on existing files."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now

        # Mock existing ADR files
        mock_file1 = MagicMock()
        mock_file1.stem = "adr-001-first-decision"
        mock_file2 = MagicMock()
        mock_file2.stem = "adr-002-second-decision"
        mock_decisions_dir.glob.return_value = [mock_file1, mock_file2]

        template = create_decision_template("third decision")
        self.assertIn("# ADR-003:", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_adr_number_handles_gaps(self, mock_decisions_dir, mock_datetime):
        """ADR number should use max + 1 even with gaps."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now

        # Mock existing ADR files with gaps
        mock_file1 = MagicMock()
        mock_file1.stem = "adr-001-first"
        mock_file2 = MagicMock()
        mock_file2.stem = "adr-005-fifth"
        mock_decisions_dir.glob.return_value = [mock_file1, mock_file2]

        template = create_decision_template("next decision")
        self.assertIn("# ADR-006:", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_adr_number_ignores_invalid_filenames(self, mock_decisions_dir, mock_datetime):
        """Should ignore files that don't match adr-NNN pattern."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now

        # Mix of valid and invalid filenames
        mock_file1 = MagicMock()
        mock_file1.stem = "adr-001-valid"
        mock_file2 = MagicMock()
        mock_file2.stem = "not-an-adr"
        mock_file3 = MagicMock()
        mock_file3.stem = "adr-abc-invalid"
        mock_decisions_dir.glob.return_value = [mock_file1, mock_file2, mock_file3]

        template = create_decision_template("test")
        self.assertIn("# ADR-002:", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_includes_title(self, mock_decisions_dir, mock_datetime):
        """Template should include the decision title."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now
        mock_decisions_dir.glob.return_value = []

        template = create_decision_template("use postgresql for storage")
        self.assertIn("Use Postgresql For Storage", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_includes_tags_when_provided(self, mock_decisions_dir, mock_datetime):
        """Template should include formatted tags."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now
        mock_decisions_dir.glob.return_value = []

        template = create_decision_template("test", tags="architecture,database")
        self.assertIn("`architecture`", template)
        self.assertIn("`database`", template)

    @patch('new_memory.datetime')
    @patch('new_memory.DECISIONS_DIR')
    def test_handles_glob_exception(self, mock_decisions_dir, mock_datetime):
        """Should handle exceptions when reading existing files."""
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-12-14"
        mock_datetime.now.return_value = mock_now

        # Simulate exception
        mock_decisions_dir.glob.side_effect = Exception("Permission denied")

        # Should default to ADR-001
        template = create_decision_template("test")
        self.assertIn("# ADR-001:", template)


class TestCreateMemory(unittest.TestCase):
    """Tests for create_memory integration function."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_memories_dir = MEMORIES_DIR
        self.original_decisions_dir = DECISIONS_DIR

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('new_memory.DECISIONS_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    @patch('new_memory.MEMORIES_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    @patch('new_memory.get_git_author')
    def test_dry_run_does_not_create_file(self, mock_author, mock_mem_dir, mock_dec_dir):
        """Dry-run should not create any files."""
        mock_author.return_value = "Test User"

        # Dry-run
        filepath = create_memory("test topic", dry_run=True)

        # File should not exist
        self.assertFalse(filepath.exists())

        # Clean up mocked directories
        import shutil
        shutil.rmtree(mock_mem_dir, ignore_errors=True)
        shutil.rmtree(mock_dec_dir, ignore_errors=True)

    @patch('new_memory.get_git_author')
    def test_creates_memory_file(self, mock_author):
        """Should create memory file with correct name."""
        mock_author.return_value = "Test User"

        # Create in temp directory
        with patch('new_memory.MEMORIES_DIR', Path(self.temp_dir)):
            filepath = create_memory("test topic")

            # File should exist
            self.assertTrue(filepath.exists())

            # Should be in memories directory
            self.assertEqual(filepath.parent, Path(self.temp_dir))

            # Should be markdown
            self.assertTrue(filepath.name.endswith(".md"))

            # Should contain topic in filename
            self.assertIn("test-topic", filepath.name)

    @patch('new_memory.get_git_author')
    def test_creates_decision_file(self, mock_author):
        """Should create decision file when is_decision=True."""
        mock_author.return_value = "Test User"

        with patch('new_memory.DECISIONS_DIR', Path(self.temp_dir)):
            filepath = create_memory("test decision", is_decision=True)

            # File should exist
            self.assertTrue(filepath.exists())

            # Should be markdown
            self.assertTrue(filepath.name.endswith(".md"))

    @patch('new_memory.get_git_author')
    def test_file_contains_correct_content_memory(self, mock_author):
        """Memory file should contain correct content."""
        mock_author.return_value = "Test User"

        with patch('new_memory.MEMORIES_DIR', Path(self.temp_dir)):
            filepath = create_memory("test topic", tags="testing,validation")

            # Read file
            content = filepath.read_text()

            # Check content
            self.assertIn("# Memory Entry:", content)
            self.assertIn("`testing`", content)
            self.assertIn("`validation`", content)
            self.assertIn("## What I Learned", content)

    @patch('new_memory.get_git_author')
    def test_file_contains_correct_content_decision(self, mock_author):
        """Decision file should contain ADR content."""
        mock_author.return_value = "Test User"

        with patch('new_memory.DECISIONS_DIR', Path(self.temp_dir)):
            filepath = create_memory("test decision", is_decision=True)

            # Read file
            content = filepath.read_text()

            # Check content
            self.assertIn("# ADR-001:", content)
            self.assertIn("**Status:**", content)
            self.assertIn("## Decision Outcome", content)

    @patch('new_memory.get_git_author')
    def test_creates_directory_if_missing(self, mock_author):
        """Should create target directory if it doesn't exist."""
        mock_author.return_value = "Test User"

        # Use non-existent subdirectory
        new_dir = Path(self.temp_dir) / "new_memories"
        self.assertFalse(new_dir.exists())

        with patch('new_memory.MEMORIES_DIR', new_dir):
            filepath = create_memory("test topic")

            # Directory should now exist
            self.assertTrue(new_dir.exists())
            self.assertTrue(filepath.exists())

    @patch('new_memory.get_git_author')
    def test_handles_special_characters_in_title(self, mock_author):
        """Should handle special characters in title."""
        mock_author.return_value = "Test User"

        with patch('new_memory.MEMORIES_DIR', Path(self.temp_dir)):
            filepath = create_memory("Test! @#$ Topic?")

            # Should create valid filename
            self.assertTrue(filepath.exists())
            # Special chars should be removed
            self.assertIn("test-topic", filepath.name)

    @patch('new_memory.get_git_author')
    @patch('new_memory.datetime')
    def test_dry_run_shows_preview(self, mock_datetime, mock_author):
        """Dry-run should print preview without creating file."""
        import io
        from contextlib import redirect_stdout

        mock_author.return_value = "Test User"
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52",
            "%Y-%m-%dT%H:%M:%SZ": "2025-12-14T14:30:52Z"
        }.get(fmt, "2025-12-14")
        mock_datetime.now.return_value = mock_now

        with patch('new_memory.MEMORIES_DIR', Path(self.temp_dir)):
            # Capture output
            output = io.StringIO()
            with redirect_stdout(output):
                filepath = create_memory("test", dry_run=True)

            # Check output
            output_str = output.getvalue()
            self.assertIn("DRY RUN", output_str)
            self.assertIn("Would create:", output_str)

            # File should not exist
            self.assertFalse(filepath.exists())


class TestFilenameFormat(unittest.TestCase):
    """Integration tests for filename format consistency."""

    @patch('new_memory.generate_session_id')
    @patch('new_memory.datetime')
    def test_filename_uniqueness(self, mock_datetime, mock_session_id):
        """Multiple calls should generate unique filenames."""
        # First call
        mock_now1 = MagicMock()
        mock_now1.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now1
        mock_session_id.return_value = "a1b2"

        filename1 = generate_memory_filename("test")

        # Second call with different session ID
        mock_now2 = MagicMock()
        mock_now2.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now2
        mock_session_id.return_value = "c3d4"

        filename2 = generate_memory_filename("test")

        # Should be different due to session ID
        self.assertNotEqual(filename1, filename2)

    @patch('new_memory.generate_session_id')
    @patch('new_memory.datetime')
    def test_filename_parts_accessible(self, mock_datetime, mock_session_id):
        """Filename parts should be parseable."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2025-12-14",
            "%H-%M-%S": "14-30-52"
        }[fmt]
        mock_datetime.now.return_value = mock_now
        mock_session_id.return_value = "a1b2"

        filename = generate_memory_filename("test topic")

        # Remove .md extension
        base = filename[:-3]

        # Split parts
        parts = base.split("_")
        self.assertEqual(len(parts), 3)

        # Date part
        self.assertEqual(parts[0], "2025-12-14")

        # Time part
        self.assertEqual(parts[1], "14-30-52")

        # Session + topic part
        self.assertTrue(parts[2].startswith("a1b2-"))


if __name__ == "__main__":
    unittest.main()
