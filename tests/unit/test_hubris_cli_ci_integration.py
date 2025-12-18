#!/usr/bin/env python3
"""
Unit tests for CI integration in hubris_cli.py

Tests the transform_commit_for_test_expert function that converts
CI results into test_results format for TestExpert training.
"""

import unittest
import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent.parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from hubris_cli import transform_commit_for_test_expert, _is_test_file


class TestCIIntegration(unittest.TestCase):
    """Test CI result transformation for TestExpert."""

    def test_is_test_file(self):
        """Test test file detection."""
        self.assertTrue(_is_test_file('tests/test_foo.py'))
        self.assertTrue(_is_test_file('tests/unit/test_bar.py'))
        self.assertTrue(_is_test_file('foo_test.py'))
        self.assertFalse(_is_test_file('cortical/processor.py'))
        self.assertFalse(_is_test_file('scripts/foo.py'))

    def test_transform_with_no_ci_data(self):
        """Commits without CI data should pass through unchanged."""
        commit = {
            'hash': 'abc123',
            'message': 'Fix bug',
            'files': ['cortical/processor.py', 'tests/test_processor.py']
        }

        result = transform_commit_for_test_expert(commit)

        # Should not add test_results
        self.assertNotIn('test_results', result)

    def test_transform_with_ci_fail_and_test_files(self):
        """Failed CI with changed test files should map to failed tests."""
        commit = {
            'hash': 'abc123',
            'message': 'Add feature',
            'files': ['cortical/processor.py', 'tests/test_processor.py'],
            'ci_result': 'fail'
        }

        result = transform_commit_for_test_expert(commit)

        # Should add test_results with failed tests
        self.assertIn('test_results', result)
        self.assertEqual(result['test_results']['failed'], ['tests/test_processor.py'])
        self.assertEqual(result['test_results']['passed'], [])
        self.assertEqual(result['test_results']['source'], 'ci_heuristic')

    def test_transform_with_ci_pass_and_test_files(self):
        """Passed CI with changed test files should map to passed tests."""
        commit = {
            'hash': 'abc123',
            'message': 'Add tests',
            'files': ['cortical/processor.py', 'tests/test_processor.py'],
            'ci_result': 'pass'
        }

        result = transform_commit_for_test_expert(commit)

        # Should add test_results with passed tests
        self.assertIn('test_results', result)
        self.assertEqual(result['test_results']['passed'], ['tests/test_processor.py'])
        self.assertEqual(result['test_results']['failed'], [])
        self.assertEqual(result['test_results']['source'], 'ci_heuristic')

    def test_transform_with_ci_fail_no_test_files(self):
        """Failed CI without test file changes should not add test_results."""
        commit = {
            'hash': 'abc123',
            'message': 'Update docs',
            'files': ['README.md', 'docs/guide.md'],
            'ci_result': 'fail'
        }

        result = transform_commit_for_test_expert(commit)

        # Should not add test_results (can't determine which tests failed)
        self.assertNotIn('test_results', result)

    def test_transform_preserves_existing_test_results(self):
        """Commits with existing test_results should not be modified."""
        commit = {
            'hash': 'abc123',
            'message': 'Fix test',
            'files': ['tests/test_foo.py'],
            'ci_result': 'pass',
            'test_results': {
                'failed': ['tests/test_bar.py'],
                'passed': ['tests/test_foo.py']
            }
        }

        result = transform_commit_for_test_expert(commit)

        # Should preserve original test_results
        self.assertEqual(result['test_results']['failed'], ['tests/test_bar.py'])
        self.assertEqual(result['test_results']['passed'], ['tests/test_foo.py'])
        self.assertNotIn('source', result['test_results'])

    def test_transform_with_files_changed_key(self):
        """Should handle both 'files' and 'files_changed' keys."""
        commit = {
            'hash': 'abc123',
            'message': 'Update',
            'files_changed': ['tests/test_foo.py'],
            'ci_result': 'fail'
        }

        result = transform_commit_for_test_expert(commit)

        # Should add test_results using files_changed
        self.assertIn('test_results', result)
        self.assertEqual(result['test_results']['failed'], ['tests/test_foo.py'])

    def test_transform_immutability(self):
        """Transform should not mutate original commit dict."""
        original = {
            'hash': 'abc123',
            'message': 'Update',
            'files': ['tests/test_foo.py'],
            'ci_result': 'fail'
        }

        result = transform_commit_for_test_expert(original)

        # Original should not have test_results
        self.assertNotIn('test_results', original)

        # Result should have test_results
        self.assertIn('test_results', result)


if __name__ == '__main__':
    unittest.main()
