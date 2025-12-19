"""
Unit tests for CommentCleaner class.

Tests the comment marker detection, cleanup suggestion, and application logic.
"""

import os
import tempfile
import unittest
from pathlib import Path

from cortical.reasoning.production_state import CommentCleaner, CommentMarker


class TestParseMarker(unittest.TestCase):
    """Test _parse_marker method with various input patterns."""

    def setUp(self):
        self.cleaner = CommentCleaner()

    def test_parse_marker_with_colon(self):
        """Test parsing marker with colon separator."""
        line = "    # TODO: Fix this bug"
        marker = self.cleaner._parse_marker(line, 10)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.marker_type, 'TODO')
        self.assertEqual(marker.content, 'Fix this bug')
        self.assertEqual(marker.line_number, 10)

    def test_parse_marker_without_colon(self):
        """Test parsing marker without colon separator."""
        line = "# THINKING This is the reasoning"
        marker = self.cleaner._parse_marker(line, 5)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.marker_type, 'THINKING')
        self.assertEqual(marker.content, 'This is the reasoning')
        self.assertEqual(marker.line_number, 5)

    def test_parse_marker_lowercase(self):
        """Test parsing lowercase marker."""
        line = "# todo: handle edge case"
        marker = self.cleaner._parse_marker(line, 1)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.marker_type, 'TODO')
        self.assertEqual(marker.content, 'handle edge case')

    def test_parse_marker_mixed_case(self):
        """Test parsing mixed case marker."""
        line = "# QuEsTiOn: Is this correct?"
        marker = self.cleaner._parse_marker(line, 1)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.marker_type, 'QUESTION')
        self.assertEqual(marker.content, 'Is this correct?')

    def test_parse_marker_with_indentation(self):
        """Test parsing marker with various indentation levels."""
        line = "        # PERF: O(n²) complexity"
        marker = self.cleaner._parse_marker(line, 20)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.marker_type, 'PERF')
        self.assertEqual(marker.content, 'O(n²) complexity')

    def test_parse_all_marker_types(self):
        """Test parsing all supported marker types."""
        marker_types = ['THINKING', 'TODO', 'QUESTION', 'NOTE', 'PERF', 'HACK']

        for marker_type in marker_types:
            line = f"# {marker_type}: Test content"
            marker = self.cleaner._parse_marker(line, 1)

            self.assertIsNotNone(marker)
            self.assertEqual(marker.marker_type, marker_type)
            self.assertEqual(marker.content, 'Test content')

    def test_parse_non_marker_comment(self):
        """Test that regular comments are not parsed as markers."""
        line = "# This is just a regular comment"
        marker = self.cleaner._parse_marker(line, 1)

        self.assertIsNone(marker)

    def test_parse_empty_content(self):
        """Test parsing marker with empty content."""
        line = "# TODO:"
        marker = self.cleaner._parse_marker(line, 1)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.marker_type, 'TODO')
        self.assertEqual(marker.content, '')

    def test_parse_non_comment_line(self):
        """Test that non-comment lines return None."""
        line = "def some_function():"
        marker = self.cleaner._parse_marker(line, 1)

        self.assertIsNone(marker)


class TestSuggestCleanup(unittest.TestCase):
    """Test suggest_cleanup method with different marker types and content."""

    def setUp(self):
        self.cleaner = CommentCleaner()

    def test_thinking_with_keep_keyword(self):
        """Test THINKING marker with keyword that suggests keeping it."""
        marker = CommentMarker('THINKING', 'This is an edge case workaround', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'keep')
        self.assertIn('non-obvious', suggestion['reason'].lower())

    def test_thinking_without_keep_keyword(self):
        """Test THINKING marker without special keywords (should remove)."""
        marker = CommentMarker('THINKING', 'Just some random thought', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'remove')
        self.assertIn('reasoning', suggestion['reason'].lower())

    def test_todo_with_task_reference(self):
        """Test TODO with task reference (should keep)."""
        marker = CommentMarker('TODO', 'Fix this (task #123)', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'keep')
        self.assertIn('task', suggestion['reason'].lower())

    def test_todo_without_task_reference(self):
        """Test TODO without task reference (should escalate)."""
        marker = CommentMarker('TODO', 'Implement error handling', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'escalate')
        self.assertIn('task', suggestion['reason'].lower())

    def test_question_resolved(self):
        """Test QUESTION marker that appears resolved (should convert)."""
        marker = CommentMarker('QUESTION', 'Should we cache? Answered: yes, decided in review', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'convert')
        self.assertIn('documentation', suggestion['reason'].lower())

    def test_question_unresolved(self):
        """Test QUESTION marker that is unresolved (should escalate)."""
        marker = CommentMarker('QUESTION', 'What is the best approach here?', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'escalate')
        self.assertIn('attention', suggestion['reason'].lower())

    def test_note_with_cross_reference(self):
        """Test NOTE with cross-reference (should keep)."""
        marker = CommentMarker('NOTE', 'See also the authentication module', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'keep')
        self.assertIn('cross-reference', suggestion['reason'].lower())

    def test_note_with_remove_keyword(self):
        """Test NOTE with keyword suggesting removal."""
        marker = CommentMarker('NOTE', 'This is obvious', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'remove')
        self.assertIn('obvious', suggestion['reason'].lower())

    def test_note_default_keep(self):
        """Test NOTE without special keywords (should keep by default)."""
        marker = CommentMarker('NOTE', 'Some useful context', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'keep')
        self.assertIn('context', suggestion['reason'].lower())

    def test_perf_always_keep(self):
        """Test PERF marker (should always keep)."""
        marker = CommentMarker('PERF', 'O(n²) complexity, acceptable for n < 1000', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'keep')
        self.assertIn('performance', suggestion['reason'].lower())

    def test_hack_with_task_reference(self):
        """Test HACK with task reference (should keep)."""
        marker = CommentMarker('HACK', 'Workaround for issue #456', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'keep')
        self.assertIn('tracked', suggestion['reason'].lower())

    def test_hack_without_task_reference(self):
        """Test HACK without task reference (should escalate)."""
        marker = CommentMarker('HACK', 'Quick fix for demo', line_number=1)
        suggestion = self.cleaner.suggest_cleanup(marker)

        self.assertEqual(suggestion['action'], 'escalate')
        self.assertIn('task', suggestion['reason'].lower())


class TestScanFile(unittest.TestCase):
    """Test scan_file method with real files."""

    def setUp(self):
        self.cleaner = CommentCleaner()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.test_dir)

    def test_scan_file_with_markers(self):
        """Test scanning a file with multiple markers."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """# First line
# TODO: Fix this
def foo():
    # THINKING: Using dict for fast lookup
    # PERF: O(1) access
    return {}
"""
        with open(test_file, 'w') as f:
            f.write(content)

        markers = self.cleaner.scan_file(test_file)

        self.assertEqual(len(markers), 3)
        self.assertEqual(markers[0].marker_type, 'TODO')
        self.assertEqual(markers[0].line_number, 2)
        self.assertEqual(markers[1].marker_type, 'THINKING')
        self.assertEqual(markers[1].line_number, 4)
        self.assertEqual(markers[2].marker_type, 'PERF')
        self.assertEqual(markers[2].line_number, 5)

        # Check file_path was set
        for marker in markers:
            self.assertEqual(marker.file_path, test_file)

    def test_scan_file_no_markers(self):
        """Test scanning a file without markers."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """# Regular comment
def foo():
    # Another regular comment
    return 42
"""
        with open(test_file, 'w') as f:
            f.write(content)

        markers = self.cleaner.scan_file(test_file)

        self.assertEqual(len(markers), 0)

    def test_scan_nonexistent_file(self):
        """Test scanning a file that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.cleaner.scan_file('/nonexistent/file.py')

    def test_scan_file_fixture(self):
        """Test scanning the fixture file with markers."""
        # Find the fixture file relative to this test
        fixture_path = Path(__file__).parent.parent / 'fixtures' / 'sample_with_markers.py'

        if not fixture_path.exists():
            self.skipTest(f"Fixture file not found: {fixture_path}")

        markers = self.cleaner.scan_file(str(fixture_path))

        # Should find multiple markers in the fixture
        self.assertGreater(len(markers), 0)

        # Check marker types are present
        marker_types = {m.marker_type for m in markers}
        self.assertTrue(marker_types.intersection({'TODO', 'THINKING', 'QUESTION', 'NOTE', 'PERF', 'HACK'}))


class TestApplyCleanup(unittest.TestCase):
    """Test apply_cleanup method."""

    def setUp(self):
        self.cleaner = CommentCleaner()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_apply_cleanup_remove(self):
        """Test applying remove action."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """def foo():
    # TODO: Fix this
    return 42
"""
        with open(test_file, 'w') as f:
            f.write(content)

        actions = [
            {'line_number': 2, 'action': 'remove'}
        ]

        result = self.cleaner.apply_cleanup(test_file, actions)

        self.assertTrue(result['modified'])
        self.assertEqual(result['removed'], 1)
        self.assertEqual(result['converted'], 0)
        self.assertEqual(result['kept'], 0)

        # Verify file was modified
        with open(test_file, 'r') as f:
            new_content = f.read()

        self.assertNotIn('TODO', new_content)
        self.assertIn('def foo():', new_content)
        self.assertIn('return 42', new_content)

    def test_apply_cleanup_convert(self):
        """Test applying convert action."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """def foo():
    # QUESTION: Is this correct?
    return 42
"""
        with open(test_file, 'w') as f:
            f.write(content)

        actions = [
            {'line_number': 2, 'action': 'convert'}
        ]

        result = self.cleaner.apply_cleanup(test_file, actions)

        self.assertTrue(result['modified'])
        self.assertEqual(result['removed'], 0)
        self.assertEqual(result['converted'], 1)
        self.assertEqual(result['kept'], 0)

        # Verify marker prefix was removed
        with open(test_file, 'r') as f:
            new_content = f.read()

        self.assertNotIn('QUESTION:', new_content)
        self.assertIn('# Is this correct?', new_content)

    def test_apply_cleanup_convert_with_replacement(self):
        """Test applying convert action with custom replacement."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """def foo():
    # QUESTION: Is this correct?
    return 42
"""
        with open(test_file, 'w') as f:
            f.write(content)

        actions = [
            {
                'line_number': 2,
                'action': 'convert',
                'replacement': '    # This approach was validated in code review.\n'
            }
        ]

        result = self.cleaner.apply_cleanup(test_file, actions)

        self.assertTrue(result['modified'])

        # Verify custom replacement was used
        with open(test_file, 'r') as f:
            new_content = f.read()

        self.assertIn('validated in code review', new_content)

    def test_apply_cleanup_keep(self):
        """Test applying keep action (no change)."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """def foo():
    # PERF: O(1) access
    return 42
"""
        with open(test_file, 'w') as f:
            f.write(content)

        actions = [
            {'line_number': 2, 'action': 'keep'}
        ]

        result = self.cleaner.apply_cleanup(test_file, actions)

        # File not modified (keep doesn't change anything)
        self.assertFalse(result['modified'])
        self.assertEqual(result['kept'], 1)

    def test_apply_cleanup_multiple_actions(self):
        """Test applying multiple actions at once."""
        test_file = os.path.join(self.test_dir, 'test.py')
        content = """def foo():
    # TODO: Fix this
    # QUESTION: Is this right?
    # PERF: O(1) access
    return 42
"""
        with open(test_file, 'w') as f:
            f.write(content)

        actions = [
            {'line_number': 2, 'action': 'remove'},
            {'line_number': 3, 'action': 'convert'},
            {'line_number': 4, 'action': 'keep'}
        ]

        result = self.cleaner.apply_cleanup(test_file, actions)

        self.assertTrue(result['modified'])
        self.assertEqual(result['removed'], 1)
        self.assertEqual(result['converted'], 1)
        self.assertEqual(result['kept'], 1)

    def test_apply_cleanup_nonexistent_file(self):
        """Test applying cleanup to nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            self.cleaner.apply_cleanup('/nonexistent/file.py', [])


class TestScanDirectory(unittest.TestCase):
    """Test scan_directory method."""

    def setUp(self):
        self.cleaner = CommentCleaner()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_scan_directory_python_files(self):
        """Test scanning directory for Python files."""
        # Create test files
        file1 = os.path.join(self.test_dir, 'module1.py')
        file2 = os.path.join(self.test_dir, 'module2.py')
        file3 = os.path.join(self.test_dir, 'readme.txt')

        with open(file1, 'w') as f:
            f.write("# TODO: Fix bug\ndef foo(): pass\n")

        with open(file2, 'w') as f:
            f.write("# PERF: Optimize\ndef bar(): pass\n")

        with open(file3, 'w') as f:
            f.write("Not a Python file\n")

        results = self.cleaner.scan_directory(self.test_dir)

        # Should find 2 Python files with markers
        self.assertEqual(len(results), 2)
        self.assertIn(file1, results)
        self.assertIn(file2, results)
        self.assertNotIn(file3, results)

    def test_scan_directory_with_subdirectories(self):
        """Test scanning directory recursively."""
        # Create subdirectory
        subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(subdir)

        file1 = os.path.join(self.test_dir, 'top.py')
        file2 = os.path.join(subdir, 'nested.py')

        with open(file1, 'w') as f:
            f.write("# TODO: Top level\n")

        with open(file2, 'w') as f:
            f.write("# HACK: Nested\n")

        results = self.cleaner.scan_directory(self.test_dir)

        # Should find both files
        self.assertEqual(len(results), 2)
        self.assertIn(file1, results)
        self.assertIn(file2, results)

    def test_scan_directory_custom_extensions(self):
        """Test scanning with custom file extensions."""
        file1 = os.path.join(self.test_dir, 'script.js')
        file2 = os.path.join(self.test_dir, 'module.py')

        with open(file1, 'w') as f:
            f.write("// TODO: Fix in JS\n")

        with open(file2, 'w') as f:
            f.write("# TODO: Fix in Python\n")

        # Scan only .js files
        results = self.cleaner.scan_directory(self.test_dir, extensions=['.js'])

        # Note: CommentCleaner only looks for Python-style # comments,
        # so it won't find markers in .js files with // comments
        # But it should still scan .js files if we specify the extension
        self.assertEqual(len(results), 0)  # No Python-style markers in .js

    def test_scan_directory_nonexistent(self):
        """Test scanning nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            self.cleaner.scan_directory('/nonexistent/dir')

    def test_scan_directory_not_a_directory(self):
        """Test scanning a file instead of directory."""
        test_file = os.path.join(self.test_dir, 'test.py')
        with open(test_file, 'w') as f:
            f.write("# TODO: Test\n")

        with self.assertRaises(ValueError):
            self.cleaner.scan_directory(test_file)


class TestGenerateCleanupReport(unittest.TestCase):
    """Test generate_cleanup_report method."""

    def setUp(self):
        self.cleaner = CommentCleaner()

    def test_generate_report_empty(self):
        """Test generating report with no markers."""
        report = self.cleaner.generate_cleanup_report({})

        self.assertIn('No markers found', report)

    def test_generate_report_single_file(self):
        """Test generating report with markers from one file."""
        markers = {
            'test.py': [
                CommentMarker('TODO', 'Fix bug', file_path='test.py', line_number=5),
                CommentMarker('PERF', 'O(1) access', file_path='test.py', line_number=10)
            ]
        }

        report = self.cleaner.generate_cleanup_report(markers)

        self.assertIn('Comment Cleanup Report', report)
        self.assertIn('2', report)  # Check count is present
        self.assertIn('TODO', report)
        self.assertIn('PERF', report)
        self.assertIn('test.py', report)
        self.assertIn('Line 5', report)
        self.assertIn('Line 10', report)

    def test_generate_report_multiple_files(self):
        """Test generating report with markers from multiple files."""
        markers = {
            'file1.py': [
                CommentMarker('TODO', 'Task 1', file_path='file1.py', line_number=1)
            ],
            'file2.py': [
                CommentMarker('HACK', 'Workaround', file_path='file2.py', line_number=2),
                CommentMarker('QUESTION', 'Is this right?', file_path='file2.py', line_number=3)
            ]
        }

        report = self.cleaner.generate_cleanup_report(markers)

        self.assertIn('3', report)  # Check total count
        self.assertIn('file1.py', report)
        self.assertIn('file2.py', report)
        self.assertIn('TODO', report)
        self.assertIn('HACK', report)
        self.assertIn('QUESTION', report)

    def test_generate_report_includes_suggestions(self):
        """Test that report includes suggested actions and reasons."""
        markers = {
            'test.py': [
                CommentMarker('TODO', 'Fix this', file_path='test.py', line_number=1)
            ]
        }

        report = self.cleaner.generate_cleanup_report(markers)

        self.assertIn('Suggested action', report)
        self.assertIn('Reason', report)
        self.assertIn('escalate', report)


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflows."""

    def setUp(self):
        self.cleaner = CommentCleaner()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_full_workflow_scan_suggest_apply(self):
        """Test complete workflow: scan, suggest, apply."""
        # Create test file
        test_file = os.path.join(self.test_dir, 'module.py')
        content = """def process():
    # TODO: Add validation (task #123)
    # THINKING: Simple reasoning
    # QUESTION: Is this right? Decided: yes
    # PERF: This is O(n)
    return data
"""
        with open(test_file, 'w') as f:
            f.write(content)

        # Scan for markers
        markers = self.cleaner.scan_file(test_file)
        self.assertEqual(len(markers), 4)

        # Generate suggestions
        suggestions = {marker.line_number: self.cleaner.suggest_cleanup(marker) for marker in markers}

        # Build actions for demonstration (we could apply these)
        actions = []
        for marker in markers:
            suggestion = suggestions[marker.line_number]
            actions.append({
                'line_number': marker.line_number,
                'action': suggestion['action']
            })

        # Verify suggestions are correct
        # TODO with task reference should be kept
        todo_marker = [m for m in markers if m.marker_type == 'TODO'][0]
        self.assertEqual(suggestions[todo_marker.line_number]['action'], 'keep')

        # THINKING without special keywords should be removed
        thinking_marker = [m for m in markers if m.marker_type == 'THINKING'][0]
        self.assertEqual(suggestions[thinking_marker.line_number]['action'], 'remove')

        # QUESTION resolved should be converted
        question_marker = [m for m in markers if m.marker_type == 'QUESTION'][0]
        self.assertEqual(suggestions[question_marker.line_number]['action'], 'convert')

        # PERF should always be kept
        perf_marker = [m for m in markers if m.marker_type == 'PERF'][0]
        self.assertEqual(suggestions[perf_marker.line_number]['action'], 'keep')

    def test_scan_directory_and_generate_report(self):
        """Test scanning directory and generating comprehensive report."""
        # Create multiple files
        file1 = os.path.join(self.test_dir, 'auth.py')
        file2 = os.path.join(self.test_dir, 'utils.py')

        with open(file1, 'w') as f:
            f.write("# TODO: Implement OAuth\n# HACK: Quick fix\n")

        with open(file2, 'w') as f:
            f.write("# PERF: Cache results\n")

        # Scan directory
        markers = self.cleaner.scan_directory(self.test_dir)

        # Generate report
        report = self.cleaner.generate_cleanup_report(markers)

        self.assertIn('3', report)  # Check total count
        self.assertIn('auth.py', report)
        self.assertIn('utils.py', report)


if __name__ == '__main__':
    unittest.main()
