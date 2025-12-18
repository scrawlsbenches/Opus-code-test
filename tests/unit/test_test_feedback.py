#!/usr/bin/env python3
"""
Unit tests for test_feedback.py

Tests the pytest output parsing and TestExpert feedback integration.
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

# Add scripts/hubris to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'hubris'))

from test_feedback import (
    parse_pytest_output,
    parse_junit_xml,
    get_source_files,
    update_test_expert_failures,
    process_test_feedback
)
from experts.test_expert import TestExpert


class TestPytestOutputParsing(unittest.TestCase):
    """Test parsing of pytest verbose output."""

    def test_parse_verbose_output(self):
        """Test parsing pytest -v output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
tests/test_foo.py::test_basic PASSED                                     [ 10%]
tests/test_foo.py::test_advanced FAILED                                  [ 20%]
tests/test_bar.py::test_simple PASSED                                    [ 30%]
tests/unit/test_query.py::test_search PASSED                            [ 40%]
tests/unit/test_query.py::test_expansion FAILED                         [ 50%]
            """)
            f.flush()

            results = parse_pytest_output(f.name)

        self.assertEqual(len(results), 5)
        self.assertTrue(results['tests/test_foo.py::test_basic'])
        self.assertFalse(results['tests/test_foo.py::test_advanced'])
        self.assertTrue(results['tests/test_bar.py::test_simple'])
        self.assertTrue(results['tests/unit/test_query.py::test_search'])
        self.assertFalse(results['tests/unit/test_query.py::test_expansion'])

        # Cleanup
        Path(f.name).unlink()

    def test_parse_with_error_status(self):
        """Test parsing output with ERROR and SKIPPED status."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
tests/test_foo.py::test_basic PASSED
tests/test_foo.py::test_error ERROR
tests/test_foo.py::test_skipped SKIPPED
            """)
            f.flush()

            results = parse_pytest_output(f.name)

        self.assertTrue(results['tests/test_foo.py::test_basic'])
        self.assertFalse(results['tests/test_foo.py::test_error'])
        self.assertFalse(results['tests/test_foo.py::test_skipped'])

        Path(f.name).unlink()


class TestJUnitXMLParsing(unittest.TestCase):
    """Test parsing of JUnit XML output."""

    def test_parse_junit_xml(self):
        """Test parsing pytest --junitxml output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write("""<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" errors="0" failures="2" skipped="0" tests="5" time="1.234">
  <testcase classname="tests.test_foo" name="test_basic" time="0.001"/>
  <testcase classname="tests.test_foo" name="test_advanced" time="0.002">
    <failure message="assertion failed">AssertionError</failure>
  </testcase>
  <testcase classname="tests.test_bar" name="test_simple" time="0.001"/>
  <testcase classname="tests.unit.test_query" name="test_search" time="0.003"/>
  <testcase classname="tests.unit.test_query" name="test_expansion" time="0.002">
    <failure message="assertion failed">AssertionError</failure>
  </testcase>
</testsuite>
            """)
            f.flush()

            results = parse_junit_xml(f.name)

        self.assertEqual(len(results), 5)
        self.assertTrue(results['tests/test_foo.py::test_basic'])
        self.assertFalse(results['tests/test_foo.py::test_advanced'])
        self.assertTrue(results['tests/test_bar.py::test_simple'])
        self.assertTrue(results['tests/unit/test_query.py::test_search'])
        self.assertFalse(results['tests/unit/test_query.py::test_expansion'])

        Path(f.name).unlink()


class TestSourceFileFiltering(unittest.TestCase):
    """Test filtering source files from test files."""

    def test_get_source_files(self):
        """Test filtering out test files."""
        changed_files = [
            'cortical/query/search.py',
            'tests/test_query.py',
            'cortical/analysis.py',
            'tests/unit/test_analysis.py',
            'README.md',
            'scripts/ml_data_collector.py',
        ]

        source_files = get_source_files(changed_files)

        self.assertIn('cortical/query/search.py', source_files)
        self.assertIn('cortical/analysis.py', source_files)
        self.assertIn('scripts/ml_data_collector.py', source_files)
        self.assertNotIn('tests/test_query.py', source_files)
        self.assertNotIn('tests/unit/test_analysis.py', source_files)
        self.assertNotIn('README.md', source_files)  # Not a .py file


class TestExpertFailureUpdates(unittest.TestCase):
    """Test updating TestExpert failure patterns."""

    def test_update_failure_patterns(self):
        """Test adding failure patterns to TestExpert."""
        expert = TestExpert()

        test_results = {
            'tests/test_foo.py::test_basic': True,
            'tests/test_foo.py::test_advanced': False,
            'tests/test_bar.py::test_simple': True,
        }

        source_files = ['cortical/query.py', 'cortical/analysis.py']

        updates = update_test_expert_failures(expert, test_results, source_files)

        # Should add 1 failure * 2 source files = 2 updates
        self.assertEqual(updates, 2)

        # Check patterns were added
        patterns = expert.model_data['test_failure_patterns']
        self.assertIn('cortical/query.py', patterns)
        self.assertIn('cortical/analysis.py', patterns)
        self.assertEqual(patterns['cortical/query.py']['tests/test_foo.py::test_advanced'], 1)
        self.assertEqual(patterns['cortical/analysis.py']['tests/test_foo.py::test_advanced'], 1)

    def test_incremental_failure_patterns(self):
        """Test that failure patterns accumulate over time."""
        expert = TestExpert()

        test_results = {
            'tests/test_foo.py::test_basic': False,
        }
        source_files = ['cortical/query.py']

        # First update
        update_test_expert_failures(expert, test_results, source_files)
        patterns = expert.model_data['test_failure_patterns']
        self.assertEqual(patterns['cortical/query.py']['tests/test_foo.py::test_basic'], 1)

        # Second update (simulating same failure again)
        update_test_expert_failures(expert, test_results, source_files)
        patterns = expert.model_data['test_failure_patterns']
        self.assertEqual(patterns['cortical/query.py']['tests/test_foo.py::test_basic'], 2)

    def test_no_failures(self):
        """Test that no patterns are added when all tests pass."""
        expert = TestExpert()

        test_results = {
            'tests/test_foo.py::test_basic': True,
            'tests/test_bar.py::test_simple': True,
        }
        source_files = ['cortical/query.py']

        updates = update_test_expert_failures(expert, test_results, source_files)
        self.assertEqual(updates, 0)
        self.assertEqual(expert.model_data['test_failure_patterns'], {})


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end feedback processing."""

    def test_process_test_feedback_dry_run(self):
        """Test complete feedback processing in dry run mode."""
        test_results = {
            'tests/test_foo.py::test_basic': True,
            'tests/test_foo.py::test_advanced': False,
            'tests/test_bar.py::test_simple': True,
        }

        changed_files = ['cortical/query.py', 'tests/test_foo.py']

        # Process with dry run
        summary = process_test_feedback(
            test_results=test_results,
            changed_files=changed_files,
            dry_run=True,
            verbose=False
        )

        self.assertEqual(summary['total_tests'], 3)
        self.assertEqual(summary['passed'], 2)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['failure_patterns_added'], 1)  # 1 failure * 1 source file
        self.assertFalse(summary['expert_updated'])  # Dry run doesn't save

    def test_process_test_feedback_with_save(self):
        """Test feedback processing with actual save."""
        test_results = {
            'tests/test_foo.py::test_basic': False,
        }
        changed_files = ['cortical/query.py']

        # Create temp model path
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_expert.json'

            # Process and save
            summary = process_test_feedback(
                test_results=test_results,
                changed_files=changed_files,
                expert_model_path=model_path,
                dry_run=False,
                verbose=False
            )

            self.assertTrue(summary['expert_updated'])
            self.assertTrue(model_path.exists())

            # Load and verify
            expert = TestExpert.load(model_path)
            patterns = expert.model_data['test_failure_patterns']
            self.assertIn('cortical/query.py', patterns)
            self.assertEqual(patterns['cortical/query.py']['tests/test_foo.py::test_basic'], 1)


if __name__ == '__main__':
    unittest.main()
