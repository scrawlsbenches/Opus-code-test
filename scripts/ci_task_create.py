#!/usr/bin/env python3
"""
Auto-create tasks from CI test failures.

This script parses test output and creates tasks for failures automatically.
Designed to be run in CI after test failures to ensure nothing is forgotten.

Usage:
    # From pytest output file
    python scripts/ci_task_create.py --pytest output.txt

    # From pytest-json-report
    python scripts/ci_task_create.py --pytest-json report.json

    # Pipe from pytest directly
    pytest tests/ 2>&1 | python scripts/ci_task_create.py --pytest -

    # Dry run (show tasks without creating)
    pytest tests/ 2>&1 | python scripts/ci_task_create.py --pytest - --dry-run

Examples:
    # In CI workflow
    - name: Run tests
      run: pytest tests/ -v 2>&1 | tee test_output.txt || true

    - name: Create tasks for failures
      if: failure()
      run: python scripts/ci_task_create.py --pytest test_output.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from task_utils import TaskSession, DEFAULT_TASKS_DIR


def parse_pytest_output(content: str) -> List[Tuple[str, str, str]]:
    """
    Parse pytest output to extract test failures.

    Returns:
        List of (test_name, file_path, error_message) tuples
    """
    failures = []

    # Pattern for FAILED lines: FAILED tests/test_foo.py::TestClass::test_name
    failed_pattern = re.compile(r'FAILED\s+([^\s:]+)::(\S+)')

    # Pattern for error details in short format
    error_pattern = re.compile(
        r'([^\s]+\.py):(\d+):\s+(\w+(?:Error|Exception|Failed|Failure).*?)(?=\n[^\s]|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # Pattern for assertion errors
    assert_pattern = re.compile(
        r'>\s+assert\s+(.+?)\nE\s+(.+)',
        re.MULTILINE
    )

    # Find all FAILED lines
    for match in failed_pattern.finditer(content):
        file_path = match.group(1)
        test_name = match.group(2)

        # Try to extract error message
        error_msg = "Test failed"

        # Look for AssertionError nearby
        test_section_start = match.start()
        test_section_end = content.find('FAILED', test_section_start + 1)
        if test_section_end == -1:
            test_section_end = len(content)

        test_section = content[test_section_start:test_section_end]

        # Look for assertion details
        assert_match = assert_pattern.search(test_section)
        if assert_match:
            error_msg = f"Assertion: {assert_match.group(2).strip()}"
        else:
            # Look for any error
            error_match = error_pattern.search(test_section)
            if error_match:
                error_msg = error_match.group(3).strip()[:100]

        failures.append((test_name, file_path, error_msg))

    return failures


def parse_pytest_json(content: str) -> List[Tuple[str, str, str]]:
    """
    Parse pytest-json-report output.

    Returns:
        List of (test_name, file_path, error_message) tuples
    """
    failures = []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return failures

    tests = data.get('tests', [])
    for test in tests:
        if test.get('outcome') == 'failed':
            nodeid = test.get('nodeid', '')
            parts = nodeid.split('::')

            file_path = parts[0] if parts else 'unknown'
            test_name = parts[-1] if parts else nodeid

            # Extract error message from call phase
            call = test.get('call', {})
            longrepr = call.get('longrepr', '')
            if isinstance(longrepr, str):
                # Get first meaningful line
                error_msg = longrepr.split('\n')[0][:100]
            else:
                error_msg = "Test failed"

            failures.append((test_name, file_path, error_msg))

    return failures


def create_tasks_for_failures(
    failures: List[Tuple[str, str, str]],
    tasks_dir: str = DEFAULT_TASKS_DIR,
    dry_run: bool = False,
    ci_run_id: str = None
) -> List[str]:
    """
    Create tasks for test failures.

    Args:
        failures: List of (test_name, file_path, error_message) tuples
        tasks_dir: Directory to save tasks
        dry_run: If True, print tasks without creating
        ci_run_id: Optional CI run identifier for context

    Returns:
        List of created task IDs
    """
    if not failures:
        print("No test failures found.")
        return []

    session = TaskSession()
    created_ids = []

    for test_name, file_path, error_msg in failures:
        # Create descriptive title
        title = f"Fix failing test: {test_name}"

        # Create detailed description
        description = f"""Test failure detected in CI.

**Test:** {test_name}
**File:** {file_path}
**Error:** {error_msg}

**Steps to fix:**
1. Run the test locally to reproduce
2. Investigate the failure
3. Implement the fix
4. Verify the test passes
5. Ensure no regressions
"""

        if ci_run_id:
            description += f"\n**CI Run:** {ci_run_id}\n"

        context = {
            "source": "ci_auto_create",
            "test_file": file_path,
            "test_name": test_name,
            "error": error_msg[:200]
        }

        if ci_run_id:
            context["ci_run_id"] = ci_run_id

        task = session.create_task(
            title=title,
            priority="high",
            category="test",
            description=description,
            effort="small",
            context=context
        )
        created_ids.append(task.id)

        if dry_run:
            print(f"[DRY RUN] Would create: {task.id}")
            print(f"  Title: {title}")
            print(f"  File: {file_path}")
            print(f"  Error: {error_msg[:60]}...")
            print()

    if not dry_run and created_ids:
        filepath = session.save(tasks_dir)
        print(f"\n✅ Created {len(created_ids)} tasks for test failures")
        print(f"Saved to: {filepath}")
        print("\nCreated tasks:")
        for task_id in created_ids:
            print(f"  {task_id}")

    return created_ids


def main():
    parser = argparse.ArgumentParser(
        description="Auto-create tasks from CI test failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--pytest", metavar="FILE",
        help="Parse pytest output file (use '-' for stdin)"
    )
    parser.add_argument(
        "--pytest-json", metavar="FILE",
        help="Parse pytest-json-report output file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show tasks without creating"
    )
    parser.add_argument(
        "--tasks-dir", default=DEFAULT_TASKS_DIR,
        help=f"Tasks directory (default: {DEFAULT_TASKS_DIR})"
    )
    parser.add_argument(
        "--ci-run-id",
        help="CI run identifier for context"
    )

    args = parser.parse_args()

    failures = []

    if args.pytest:
        if args.pytest == '-':
            content = sys.stdin.read()
        else:
            with open(args.pytest) as f:
                content = f.read()
        failures = parse_pytest_output(content)

    elif args.pytest_json:
        with open(args.pytest_json) as f:
            content = f.read()
        failures = parse_pytest_json(content)

    else:
        parser.print_help()
        print("\nError: Must specify --pytest or --pytest-json")
        sys.exit(1)

    if failures:
        print(f"Found {len(failures)} test failure(s)\n")

    create_tasks_for_failures(
        failures,
        tasks_dir=args.tasks_dir,
        dry_run=args.dry_run,
        ci_run_id=args.ci_run_id
    )


if __name__ == "__main__":
    main()
