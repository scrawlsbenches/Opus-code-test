#!/usr/bin/env python3
"""
Unit tests for scripts/suggest_tasks.py

Tests the TaskSuggester class and its methods for analyzing
code changes and suggesting follow-up tasks.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
scripts_path = Path(__file__).parent.parent.parent / 'scripts'
sys.path.insert(0, str(scripts_path))

from suggest_tasks import TaskSuggester


class TestTaskSuggester(unittest.TestCase):
    """Test TaskSuggester class."""

    def setUp(self):
        """Create a temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    def test_check_git_repo_valid(self, mock_run):
        """Test _check_git_repo with valid git repository."""
        mock_run.return_value = Mock(returncode=0)
        suggester = TaskSuggester(self.temp_dir)
        self.assertTrue(suggester._is_git_repo)

    @patch('subprocess.run')
    def test_check_git_repo_invalid(self, mock_run):
        """Test _check_git_repo with non-git directory."""
        mock_run.return_value = Mock(returncode=1)
        suggester = TaskSuggester(self.temp_dir)
        self.assertFalse(suggester._is_git_repo)

    @patch('subprocess.run')
    def test_get_staged_changes(self, mock_run):
        """Test get_staged_changes parsing."""
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse check
            Mock(
                returncode=0,
                stdout='A\tscripts/new_file.py\nM\tcortical/processor.py\nD\told_file.py\n'
            )
        ]

        suggester = TaskSuggester(self.temp_dir)
        changes = suggester.get_staged_changes()

        self.assertEqual(changes['added'], ['scripts/new_file.py'])
        self.assertEqual(changes['modified'], ['cortical/processor.py'])
        self.assertEqual(changes['deleted'], ['old_file.py'])

    @patch('subprocess.run')
    def test_get_unstaged_changes(self, mock_run):
        """Test get_unstaged_changes parsing."""
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse check
            Mock(
                returncode=0,
                stdout='M\ttests/test_processor.py\n'
            )
        ]

        suggester = TaskSuggester(self.temp_dir)
        changes = suggester.get_unstaged_changes()

        self.assertEqual(changes['modified'], ['tests/test_processor.py'])
        self.assertEqual(changes['added'], [])
        self.assertEqual(changes['deleted'], [])

    @patch('subprocess.run')
    def test_get_staged_changes_not_git_repo(self, mock_run):
        """Test get_staged_changes raises error if not git repo."""
        mock_run.return_value = Mock(returncode=1)
        suggester = TaskSuggester(self.temp_dir)

        with self.assertRaises(RuntimeError):
            suggester.get_staged_changes()

    def test_parse_git_status(self):
        """Test _parse_git_status parsing logic."""
        suggester = TaskSuggester(self.temp_dir)

        output = """A\tnew_file.py
M\tmodified_file.py
D\tdeleted_file.py
A\tanother_new.py"""

        changes = suggester._parse_git_status(output)

        self.assertEqual(len(changes['added']), 2)
        self.assertEqual(len(changes['modified']), 1)
        self.assertEqual(len(changes['deleted']), 1)
        self.assertIn('new_file.py', changes['added'])
        self.assertIn('modified_file.py', changes['modified'])

    def test_parse_git_status_empty(self):
        """Test _parse_git_status with empty output."""
        suggester = TaskSuggester(self.temp_dir)
        changes = suggester._parse_git_status('')

        self.assertEqual(changes['added'], [])
        self.assertEqual(changes['modified'], [])
        self.assertEqual(changes['deleted'], [])

    def test_has_undocumented_methods_true(self):
        """Test _has_undocumented_methods detects missing docstrings."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def function_one():
    pass

def function_two():
    pass

def function_three():
    pass
"""
        self.assertTrue(suggester._has_undocumented_methods(content))

    def test_has_undocumented_methods_false(self):
        """Test _has_undocumented_methods with good documentation."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def function_one():
    \"\"\"Docstring here.\"\"\"
    pass

def function_two():
    \"\"\"Another docstring.\"\"\"
    pass
"""
        self.assertFalse(suggester._has_undocumented_methods(content))

    def test_is_new_test_file_true(self):
        """Test _is_new_test_file detects test files."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def test_something():
    assert True
"""
        self.assertTrue(suggester._is_new_test_file('test_feature.py', content))

    def test_is_new_test_file_false(self):
        """Test _is_new_test_file with non-test file."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def process_data():
    pass
"""
        self.assertFalse(suggester._is_new_test_file('processor.py', content))

    def test_has_performance_code_true(self):
        """Test _has_performance_code detects loops."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
# This is O(n^2)
for item in items:
    for other in others:
        process(item, other)
"""
        self.assertTrue(suggester._has_performance_code('cortical/analysis.py', content))

    def test_has_performance_code_false(self):
        """Test _has_performance_code with simple code."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def simple_function():
    return True
"""
        self.assertFalse(suggester._has_performance_code('cortical/analysis.py', content))

    def test_has_performance_code_non_cortical_false(self):
        """Test _has_performance_code ignores non-cortical files."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
for item in items:
    process(item)
"""
        # Should be False because not in cortical/
        self.assertFalse(suggester._has_performance_code('scripts/tool.py', content))

    def test_has_validation_logic_true(self):
        """Test _has_validation_logic detects validation code."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def validate(value):
    if value < 0:
        raise ValueError("Value must be positive")
    return value
"""
        self.assertTrue(suggester._has_validation_logic(content))

    def test_has_validation_logic_false(self):
        """Test _has_validation_logic with no validation."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def process(value):
    return value * 2
"""
        self.assertFalse(suggester._has_validation_logic(content))

    def test_missing_type_hints_true(self):
        """Test _missing_type_hints detects untyped functions."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def function_one(x):
    return x

def function_two(y):
    return y
"""
        self.assertTrue(suggester._missing_type_hints(content))

    def test_missing_type_hints_false(self):
        """Test _missing_type_hints with typed functions."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
def function_one(x: int) -> int:
    return x

def function_two(y: str) -> str:
    return y
"""
        self.assertFalse(suggester._missing_type_hints(content))

    def test_is_config_change_true(self):
        """Test _is_config_change detects config files."""
        suggester = TaskSuggester(self.temp_dir)

        self.assertTrue(suggester._is_config_change('cortical/config.py'))
        self.assertTrue(suggester._is_config_change('.github/workflows/ci.yml'))
        self.assertTrue(suggester._is_config_change('pyproject.toml'))

    def test_is_config_change_false(self):
        """Test _is_config_change with regular files."""
        suggester = TaskSuggester(self.temp_dir)

        self.assertFalse(suggester._is_config_change('cortical/processor.py'))
        self.assertFalse(suggester._is_config_change('tests/test_processor.py'))

    def test_has_new_public_api_true(self):
        """Test _has_new_public_api detects public methods."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
class Processor:
    def process_data(self, data):
        return data

    def compute_result(self):
        return 42
"""
        self.assertTrue(suggester._has_new_public_api('cortical/processor.py', content))

    def test_has_new_public_api_false_private(self):
        """Test _has_new_public_api ignores private methods."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
class Processor:
    def _internal_method(self):
        pass

    def __private_method(self):
        pass
"""
        self.assertFalse(suggester._has_new_public_api('cortical/processor.py', content))

    def test_has_new_public_api_false_non_cortical(self):
        """Test _has_new_public_api ignores non-cortical files."""
        suggester = TaskSuggester(self.temp_dir)

        content = """
class Tool:
    def run(self):
        pass
"""
        self.assertFalse(suggester._has_new_public_api('scripts/tool.py', content))

    @patch('subprocess.run')
    def test_analyze_file_undocumented(self, mock_run):
        """Test analyze_file detects undocumented methods."""
        mock_run.return_value = Mock(returncode=0)

        # Create test file
        test_file = self.temp_path / 'test.py'
        test_file.write_text("""
def function_one():
    pass

def function_two():
    pass
""")

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.analyze_file('test.py')

        # Should suggest adding docstrings
        self.assertTrue(any('docstring' in s['title'].lower() for s in suggestions))

    @patch('subprocess.run')
    def test_analyze_file_new_test(self, mock_run):
        """Test analyze_file detects new test files."""
        mock_run.return_value = Mock(returncode=0)

        # Create test file
        test_file = self.temp_path / 'test_feature.py'
        test_file.write_text("""
def test_something():
    assert True
""")

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.analyze_file('test_feature.py')

        # Should suggest running tests
        self.assertTrue(any('run tests' in s['title'].lower() for s in suggestions))

    @patch('subprocess.run')
    def test_analyze_file_validation(self, mock_run):
        """Test analyze_file detects validation logic."""
        mock_run.return_value = Mock(returncode=0)

        # Create file with validation
        test_file = self.temp_path / 'validator.py'
        test_file.write_text("""
def validate(value):
    if value < 0:
        raise ValueError("Invalid")
""")

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.analyze_file('validator.py')

        # Should suggest verifying validation tests
        self.assertTrue(any('validation' in s['title'].lower() for s in suggestions))

    @patch('subprocess.run')
    def test_analyze_file_nonexistent(self, mock_run):
        """Test analyze_file handles non-existent files."""
        mock_run.return_value = Mock(returncode=0)

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.analyze_file('nonexistent.py')

        # Should return empty list
        self.assertEqual(suggestions, [])

    @patch('subprocess.run')
    def test_analyze_file_non_python(self, mock_run):
        """Test analyze_file skips non-Python files."""
        mock_run.return_value = Mock(returncode=0)

        # Create non-Python file
        test_file = self.temp_path / 'README.md'
        test_file.write_text('# Documentation')

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.analyze_file('README.md')

        # Should return empty list
        self.assertEqual(suggestions, [])

    @patch('subprocess.run')
    def test_suggest_from_changes_deduplication(self, mock_run):
        """Test suggest_from_changes deduplicates suggestions."""
        # Mock git repo check and changes
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse check
            Mock(returncode=0, stdout='M\ttest1.py\nM\ttest2.py\n')  # staged changes
        ]

        # Create test files with similar issues
        test1 = self.temp_path / 'test1.py'
        test1.write_text('def f1(): pass\ndef f2(): pass')

        test2 = self.temp_path / 'test2.py'
        test2.write_text('def f3(): pass\ndef f4(): pass')

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.suggest_from_changes()

        # Check that duplicate titles are removed
        titles = [s['title'] for s in suggestions]
        self.assertEqual(len(titles), len(set(titles)))

    @patch('subprocess.run')
    def test_suggest_from_changes_priority_ordering(self, mock_run):
        """Test suggest_from_changes sorts by priority."""
        # Mock git repo check and changes
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse check
            Mock(returncode=0, stdout='M\ttest_high.py\n')
        ]

        # Create file that triggers high priority suggestion
        high_file = self.temp_path / 'test_high.py'
        high_file.write_text('def test_something(): pass')  # High priority: new test

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.suggest_from_changes()

        # Should have at least one suggestion
        self.assertGreater(len(suggestions), 0)

        # Find the test-related suggestion
        test_suggestions = [s for s in suggestions if 'test' in s['title'].lower()]
        if test_suggestions:
            # Test suggestions should be high priority
            self.assertEqual(test_suggestions[0]['priority'], 'high')

    @patch('subprocess.run')
    def test_suggest_from_changes_limit(self, mock_run):
        """Test suggest_from_changes limits to 10 suggestions."""
        # Mock git repo check and many changes
        files = '\n'.join([f'M\tfile{i}.py' for i in range(20)])
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse check
            Mock(returncode=0, stdout=files)
        ]

        # Create many files
        for i in range(20):
            f = self.temp_path / f'file{i}.py'
            f.write_text(f'def f{i}(): pass')

        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.suggest_from_changes()

        # Should be limited to 10
        self.assertLessEqual(len(suggestions), 10)

    def test_format_suggestions_empty(self):
        """Test format_suggestions with no suggestions."""
        suggester = TaskSuggester(self.temp_dir)
        output = suggester.format_suggestions([])

        self.assertIn('No suggestions', output)

    def test_format_suggestions_markdown(self):
        """Test format_suggestions produces markdown."""
        suggester = TaskSuggester(self.temp_dir)

        suggestions = [
            {
                'title': 'Add tests',
                'priority': 'high',
                'category': 'test',
                'reason': 'New code added',
                'files': ['test.py']
            }
        ]

        output = suggester.format_suggestions(suggestions)

        self.assertIn('# Suggested Follow-up Tasks', output)
        self.assertIn('Add tests', output)
        self.assertIn('high', output.lower())
        self.assertIn('test', output.lower())

    def test_format_json(self):
        """Test format_json produces valid JSON."""
        suggester = TaskSuggester(self.temp_dir)

        suggestions = [
            {
                'title': 'Add tests',
                'priority': 'high',
                'category': 'test',
                'reason': 'New code added',
                'files': ['test.py']
            }
        ]

        output = suggester.format_json(suggestions)

        # Should be valid JSON
        parsed = json.loads(output)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['title'], 'Add tests')

    @patch('suggest_tasks.TaskSession')
    def test_create_tasks(self, mock_session_class):
        """Test create_tasks creates tasks using TaskSession."""
        # Check if task_utils is actually available
        import suggest_tasks
        if not suggest_tasks.TASK_UTILS_AVAILABLE:
            self.skipTest("task_utils not available")

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        suggester = TaskSuggester(self.temp_dir)

        suggestions = [
            {
                'title': 'Add tests',
                'priority': 'high',
                'category': 'test',
                'reason': 'New code added',
                'files': ['test.py']
            },
            {
                'title': 'Add docs',
                'priority': 'low',
                'category': 'docs',
                'reason': 'Missing docstrings',
                'files': ['module.py']
            }
        ]

        count = suggester.create_tasks(suggestions)

        # Should create 2 tasks
        self.assertEqual(count, 2)
        self.assertEqual(mock_session.create_task.call_count, 2)
        mock_session.save.assert_called_once()

    def test_create_tasks_no_task_utils(self):
        """Test create_tasks raises error if task_utils not available."""
        import suggest_tasks
        original_value = suggest_tasks.TASK_UTILS_AVAILABLE
        try:
            # Temporarily disable task_utils
            suggest_tasks.TASK_UTILS_AVAILABLE = False

            suggester = TaskSuggester(self.temp_dir)

            with self.assertRaises(ImportError):
                suggester.create_tasks([])
        finally:
            # Restore original value
            suggest_tasks.TASK_UTILS_AVAILABLE = original_value


class TestTaskSuggesterIntegration(unittest.TestCase):
    """Integration tests using real git repository."""

    def setUp(self):
        """Create temporary git repository."""
        import shutil
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.temp_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'],
                      cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'],
                      cwd=self.temp_dir, check=True, capture_output=True)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_git_workflow(self):
        """Test with real git staging workflow."""
        # Create and stage a file
        test_file = self.temp_path / 'new_feature.py'
        test_file.write_text("""
def new_function():
    pass

def another_function():
    pass
""")

        subprocess.run(['git', 'add', 'new_feature.py'],
                      cwd=self.temp_dir, check=True, capture_output=True)

        # Analyze
        suggester = TaskSuggester(self.temp_dir)
        suggestions = suggester.suggest_from_changes()

        # Should have suggestions
        self.assertGreater(len(suggestions), 0)


if __name__ == '__main__':
    unittest.main()
