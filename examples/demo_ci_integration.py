#!/usr/bin/env python3
"""
Demo: CI Integration for TestExpert Training

This demo shows how CI results are transformed into test_results
format that TestExpert can learn from.
"""

import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from hubris_cli import transform_commit_for_test_expert


def demo():
    """Demonstrate CI integration transformation."""

    print("=" * 70)
    print("Demo: CI Integration for TestExpert Training")
    print("=" * 70)
    print()

    # Example 1: CI Failure with test files changed
    print("Example 1: CI Failure with Test Files Changed")
    print("-" * 70)

    commit1 = {
        'hash': 'abc123',
        'message': 'feat: Add graph boosted search',
        'files': [
            'cortical/query/search.py',
            'tests/test_query.py',
            'tests/unit/test_query_search.py'
        ],
        'ci_result': 'fail'
    }

    print("Input commit:")
    print(f"  Hash:    {commit1['hash']}")
    print(f"  Message: {commit1['message']}")
    print(f"  Files:   {', '.join(commit1['files'])}")
    print(f"  CI:      {commit1['ci_result']}")
    print()

    transformed1 = transform_commit_for_test_expert(commit1)

    print("After transformation:")
    if 'test_results' in transformed1:
        tr = transformed1['test_results']
        print(f"  Failed:  {', '.join(tr['failed'])}")
        print(f"  Passed:  {', '.join(tr['passed']) if tr['passed'] else '(none)'}")
        print(f"  Source:  {tr['source']}")
    else:
        print("  (no test_results added)")
    print()
    print("✓ TestExpert will learn: When cortical/query/search.py changes,")
    print("  tests/test_query.py and tests/unit/test_query_search.py may fail")
    print()
    print()

    # Example 2: CI Pass with test files
    print("Example 2: CI Pass with Test Files Changed")
    print("-" * 70)

    commit2 = {
        'hash': 'def456',
        'message': 'fix: Fix PageRank convergence bug',
        'files': [
            'cortical/analysis.py',
            'tests/test_analysis.py'
        ],
        'ci_result': 'pass'
    }

    print("Input commit:")
    print(f"  Hash:    {commit2['hash']}")
    print(f"  Message: {commit2['message']}")
    print(f"  Files:   {', '.join(commit2['files'])}")
    print(f"  CI:      {commit2['ci_result']}")
    print()

    transformed2 = transform_commit_for_test_expert(commit2)

    print("After transformation:")
    if 'test_results' in transformed2:
        tr = transformed2['test_results']
        print(f"  Failed:  {', '.join(tr['failed']) if tr['failed'] else '(none)'}")
        print(f"  Passed:  {', '.join(tr['passed'])}")
        print(f"  Source:  {tr['source']}")
    else:
        print("  (no test_results added)")
    print()
    print("✓ TestExpert learns: When cortical/analysis.py changes,")
    print("  tests/test_analysis.py should be run (and passed)")
    print()
    print()

    # Example 3: CI Failure but no test files changed
    print("Example 3: CI Failure without Test Files Changed")
    print("-" * 70)

    commit3 = {
        'hash': 'ghi789',
        'message': 'docs: Update README',
        'files': [
            'README.md',
            'docs/quickstart.md'
        ],
        'ci_result': 'fail'
    }

    print("Input commit:")
    print(f"  Hash:    {commit3['hash']}")
    print(f"  Message: {commit3['message']}")
    print(f"  Files:   {', '.join(commit3['files'])}")
    print(f"  CI:      {commit3['ci_result']}")
    print()

    transformed3 = transform_commit_for_test_expert(commit3)

    print("After transformation:")
    if 'test_results' in transformed3:
        tr = transformed3['test_results']
        print(f"  Failed:  {', '.join(tr['failed'])}")
        print(f"  Passed:  {', '.join(tr['passed']) if tr['passed'] else '(none)'}")
        print(f"  Source:  {tr['source']}")
    else:
        print("  (no test_results added)")
    print()
    print("⚠  Cannot determine which tests failed - no test files changed")
    print("   Heuristic limitation: needs test files to map failures")
    print()
    print()

    # Example 4: No CI data
    print("Example 4: Commit without CI Data")
    print("-" * 70)

    commit4 = {
        'hash': 'jkl012',
        'message': 'refactor: Split processor module',
        'files': [
            'cortical/processor/core.py',
            'cortical/processor/compute.py',
            'tests/test_processor.py'
        ]
    }

    print("Input commit:")
    print(f"  Hash:    {commit4['hash']}")
    print(f"  Message: {commit4['message']}")
    print(f"  Files:   {', '.join(commit4['files'])}")
    print(f"  CI:      (no CI data)")
    print()

    transformed4 = transform_commit_for_test_expert(commit4)

    print("After transformation:")
    if 'test_results' in transformed4:
        tr = transformed4['test_results']
        print(f"  Failed:  {', '.join(tr['failed'])}")
        print(f"  Passed:  {', '.join(tr['passed']) if tr['passed'] else '(none)'}")
    else:
        print("  (no test_results added)")
    print()
    print("→  Passes through unchanged when no CI data available")
    print()
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The transform_commit_for_test_expert() function:")
    print()
    print("✓ Maps CI failures to test files (when test files changed)")
    print("✓ Maps CI passes to test files (tracks successful runs)")
    print("✓ Skips transformation when no test files changed")
    print("✓ Passes through unchanged when no CI data")
    print("✓ Preserves existing test_results if present")
    print("✓ Non-destructive (doesn't mutate original commit)")
    print()
    print("Usage:")
    print("  python scripts/hubris_cli.py train --commits 500 --include-ci")
    print()


if __name__ == '__main__':
    demo()
