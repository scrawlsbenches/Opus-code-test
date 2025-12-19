#!/usr/bin/env python3
"""
Parse pytest output and update TestExpert feedback.

Parses pytest results (from output file or JUnit XML) and feeds
test pass/fail data to FeedbackProcessor and TestExpert.

This closes the feedback loop: TestExpert predicts tests -> tests run ->
results feed back into TestExpert's failure pattern learning.

Usage:
    # Parse pytest output file
    python test_feedback.py --parse-output .pytest-output.txt

    # Parse JUnit XML
    python test_feedback.py --parse-xml .pytest-results.xml

    # Auto-detect available output
    python test_feedback.py --auto

    # Dry run (show what would be updated)
    python test_feedback.py --parse-output .pytest-output.txt --dry-run

Integration:
    # Run after pytest
    pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
    python scripts/hubris/test_feedback.py --parse-output .pytest-output.txt

    # Or with JUnit XML
    pytest tests/ --junitxml=.pytest-results.xml
    python scripts/hubris/test_feedback.py --parse-xml .pytest-results.xml
"""

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from feedback_collector import FeedbackProcessor, PredictionRecorder
from credit_account import CreditLedger
from value_signal import ValueAttributor
from experts.test_expert import TestExpert


def parse_pytest_output(filepath: str) -> Dict[str, bool]:
    """
    Parse pytest verbose output for test pass/fail status.

    Parses lines like:
        tests/test_foo.py::test_bar PASSED [ 10%]
        tests/test_foo.py::test_baz FAILED [ 20%]

    Args:
        filepath: Path to pytest output file

    Returns:
        Dictionary mapping test_name -> passed (bool)
    """
    results = {}
    pattern = re.compile(r'^([\w/\.]+\.py::\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)')

    with open(filepath) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                test_name = match.group(1)
                status = match.group(2)
                # Consider PASSED as True, everything else as False
                results[test_name] = (status == 'PASSED')

    return results


def parse_junit_xml(filepath: str) -> Dict[str, bool]:
    """
    Parse pytest JUnit XML output for test pass/fail status.

    Args:
        filepath: Path to JUnit XML file

    Returns:
        Dictionary mapping test_name -> passed (bool)
    """
    results = {}

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for testcase in root.iter('testcase'):
            classname = testcase.get('classname', '')
            name = testcase.get('name', '')

            # Convert classname to file path
            # "tests.test_foo" -> "tests/test_foo.py"
            file_path = classname.replace('.', '/') + '.py'
            test_name = f"{file_path}::{name}"

            # Test passed if no failure/error/skipped elements
            has_failure = testcase.find('failure') is not None
            has_error = testcase.find('error') is not None
            has_skipped = testcase.find('skipped') is not None

            results[test_name] = not (has_failure or has_error or has_skipped)

    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML: {e}", file=sys.stderr)
        return {}

    return results


def get_changed_files(ref: str = 'HEAD') -> List[str]:
    """
    Get list of files changed in the most recent commit(s).

    Args:
        ref: Git ref to check (default: HEAD)

    Returns:
        List of changed file paths
    """
    try:
        # Get files changed in last commit
        result = subprocess.run(
            ['git', 'diff', '--name-only', f'{ref}~1..{ref}'],
            capture_output=True,
            text=True,
            check=True
        )
        files = [line.strip() for line in result.stdout.split('\n') if line.strip()]

        # If no files in last commit, check staged files
        if not files:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )
            files = [line.strip() for line in result.stdout.split('\n') if line.strip()]

        # If still no files, check unstaged files
        if not files:
            result = subprocess.run(
                ['git', 'diff', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )
            files = [line.strip() for line in result.stdout.split('\n') if line.strip()]

        return files

    except subprocess.CalledProcessError:
        # Not in a git repo or no changes
        return []


def get_source_files(changed_files: List[str]) -> List[str]:
    """
    Filter out test files from changed files list.

    Args:
        changed_files: All changed files

    Returns:
        List of source files (non-test files)
    """
    source_files = []
    for f in changed_files:
        f_lower = f.lower()
        is_test = (
            'test' in f_lower or
            f_lower.startswith('tests/') or
            f_lower.endswith('_test.py')
        )
        if not is_test and f.endswith('.py'):
            source_files.append(f)
    return source_files


def update_test_expert_failures(
    expert: TestExpert,
    test_results: Dict[str, bool],
    source_files: List[str]
) -> int:
    """
    Update TestExpert's failure patterns with new test results.

    Args:
        expert: TestExpert instance to update
        test_results: Test pass/fail results
        source_files: Source files that were changed

    Returns:
        Number of failure patterns added
    """
    # Get failed tests
    failed_tests = [test for test, passed in test_results.items() if not passed]

    if not failed_tests or not source_files:
        return 0

    # Update failure patterns in model_data
    failure_patterns = expert.model_data.get('test_failure_patterns', {})
    if not isinstance(failure_patterns, dict):
        failure_patterns = {}

    updates = 0
    for source in source_files:
        if source not in failure_patterns:
            failure_patterns[source] = {}

        for failed_test in failed_tests:
            # Increment failure count
            failure_patterns[source][failed_test] = \
                failure_patterns[source].get(failed_test, 0) + 1
            updates += 1

    # Update model_data
    expert.model_data['test_failure_patterns'] = failure_patterns

    return updates


def process_test_feedback(
    test_results: Dict[str, bool],
    changed_files: Optional[List[str]] = None,
    expert_model_path: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Process test results and update expert feedback.

    Args:
        test_results: Test pass/fail results
        changed_files: Files changed (auto-detect if None)
        expert_model_path: Path to TestExpert model
        dry_run: Show changes without saving
        verbose: Print detailed output

    Returns:
        Summary dictionary with stats
    """
    summary = {
        'total_tests': len(test_results),
        'passed': sum(1 for v in test_results.values() if v),
        'failed': sum(1 for v in test_results.values() if not v),
        'credit_updates': {},
        'failure_patterns_added': 0,
        'expert_updated': False
    }

    if verbose:
        print(f"Processing {summary['total_tests']} test results...")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")

    # Get changed files if not provided
    if changed_files is None:
        changed_files = get_changed_files()
        if verbose:
            print(f"  Changed files: {len(changed_files)}")

    source_files = get_source_files(changed_files)
    if verbose and source_files:
        print(f"  Source files: {len(source_files)}")
        for f in source_files[:5]:
            print(f"    - {f}")

    # Update FeedbackProcessor (for credit tracking)
    try:
        ledger = CreditLedger()
        attributor = ValueAttributor()
        recorder = PredictionRecorder()
        processor = FeedbackProcessor(ledger, attributor, recorder)

        if not dry_run:
            credit_updates = processor.process_test_outcome(test_results)
            summary['credit_updates'] = credit_updates

            if verbose and credit_updates:
                print(f"\nCredit updates:")
                for expert_id, amount in credit_updates.items():
                    print(f"  {expert_id}: {amount:+.2f}")

    except Exception as e:
        if verbose:
            print(f"Warning: Failed to update credits: {e}", file=sys.stderr)

    # Update TestExpert failure patterns
    if expert_model_path is None:
        # Default path
        git_ml_dir = Path(__file__).parent.parent.parent / '.git-ml'
        expert_model_path = git_ml_dir / 'models' / 'test_expert.json'

    try:
        # Load or create expert
        if expert_model_path.exists():
            expert = TestExpert.load(expert_model_path)
            if verbose:
                print(f"\nLoaded TestExpert from {expert_model_path}")
        else:
            expert = TestExpert()
            if verbose:
                print(f"\nCreated new TestExpert")

        # Update failure patterns
        updates = update_test_expert_failures(expert, test_results, source_files)
        summary['failure_patterns_added'] = updates

        if verbose and updates > 0:
            print(f"  Added {updates} failure pattern entries")

        # Save expert (unless dry run)
        if not dry_run and updates > 0:
            expert_model_path.parent.mkdir(parents=True, exist_ok=True)
            expert.save(expert_model_path)
            summary['expert_updated'] = True

            if verbose:
                print(f"  Saved updated expert to {expert_model_path}")

    except Exception as e:
        if verbose:
            print(f"Warning: Failed to update TestExpert: {e}", file=sys.stderr)

    return summary


def auto_detect_output() -> Optional[Tuple[str, str]]:
    """
    Auto-detect pytest output file.

    Returns:
        Tuple of (file_path, format) or None if not found
        Format is either 'text' or 'xml'
    """
    candidates = [
        ('.pytest-output.txt', 'text'),
        ('.pytest-results.xml', 'xml'),
        ('pytest-results.xml', 'xml'),
        ('test-results.xml', 'xml'),
    ]

    for filename, fmt in candidates:
        if Path(filename).exists():
            return (filename, fmt)

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Parse pytest output and update TestExpert feedback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse verbose pytest output
  python test_feedback.py --parse-output .pytest-output.txt

  # Parse JUnit XML
  python test_feedback.py --parse-xml .pytest-results.xml

  # Auto-detect output file
  python test_feedback.py --auto

  # Dry run (show changes without saving)
  python test_feedback.py --auto --dry-run --verbose

Integration:
  # Run tests and capture output
  pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
  python test_feedback.py --parse-output .pytest-output.txt

  # Or with JUnit XML
  pytest tests/ --junitxml=.pytest-results.xml
  python test_feedback.py --parse-xml .pytest-results.xml
        """
    )

    parser.add_argument(
        '--parse-output',
        help='Parse pytest verbose output file'
    )
    parser.add_argument(
        '--parse-xml',
        help='Parse pytest JUnit XML file'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-detect pytest output file'
    )
    parser.add_argument(
        '--changed-files',
        nargs='*',
        help='Explicitly specify changed files (auto-detect from git if not provided)'
    )
    parser.add_argument(
        '--expert-model',
        type=Path,
        help='Path to TestExpert model (default: .git-ml/models/test_expert.json)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show changes without saving'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    # Determine input source
    test_results = {}

    if args.parse_output:
        if not Path(args.parse_output).exists():
            print(f"Error: File not found: {args.parse_output}", file=sys.stderr)
            return 1

        test_results = parse_pytest_output(args.parse_output)
        if args.verbose:
            print(f"Parsed {len(test_results)} tests from {args.parse_output}")

    elif args.parse_xml:
        if not Path(args.parse_xml).exists():
            print(f"Error: File not found: {args.parse_xml}", file=sys.stderr)
            return 1

        test_results = parse_junit_xml(args.parse_xml)
        if args.verbose:
            print(f"Parsed {len(test_results)} tests from {args.parse_xml}")

    elif args.auto:
        detected = auto_detect_output()
        if detected is None:
            print("Error: No pytest output file found", file=sys.stderr)
            print("Looked for: .pytest-output.txt, .pytest-results.xml", file=sys.stderr)
            return 1

        filepath, fmt = detected
        if fmt == 'text':
            test_results = parse_pytest_output(filepath)
        else:
            test_results = parse_junit_xml(filepath)

        if args.verbose:
            print(f"Auto-detected {filepath} ({fmt})")
            print(f"Parsed {len(test_results)} tests")

    else:
        parser.print_help()
        return 1

    # Process results
    if not test_results:
        print("Warning: No test results found in output", file=sys.stderr)
        return 0

    summary = process_test_feedback(
        test_results=test_results,
        changed_files=args.changed_files,
        expert_model_path=args.expert_model,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # Print summary
    if args.verbose or args.dry_run:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Failure patterns added: {summary['failure_patterns_added']}")
        print(f"  Expert updated: {summary['expert_updated']}")
        if args.dry_run:
            print("\n  (Dry run - no changes saved)")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
