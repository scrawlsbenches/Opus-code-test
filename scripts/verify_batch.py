#!/usr/bin/env python3
"""
Batch Verification Script
=========================

Automated verification of batch agent work for quality assurance.

This script verifies that a batch of agent work completed successfully by
running comprehensive checks including tests, file conflict detection, and
git status validation.

Usage:
    python scripts/verify_batch.py                    # Run all checks (quick mode)
    python scripts/verify_batch.py --full             # Run all checks (full tests)
    python scripts/verify_batch.py --check tests      # Run only test verification
    python scripts/verify_batch.py --check conflicts  # Check file conflicts only
    python scripts/verify_batch.py --check git        # Check git status only
    python scripts/verify_batch.py --json             # Output as JSON
    python scripts/verify_batch.py --modified-files agent_work.json

Options:
    --quick             Run smoke tests only (default)
    --full              Run smoke + unit tests
    --check TYPE        Run specific check only (tests, conflicts, git)
    --json              Output results as JSON
    --modified-files    JSON file with agent file modifications
    --verbose, -v       Show detailed output
    --quiet, -q         Show minimal output

Modified Files Format:
    JSON file mapping agent IDs to lists of modified files:
    {
        "agent_1": ["file1.py", "file2.py"],
        "agent_2": ["file3.py"]
    }

Exit Codes:
    0: All verification checks passed
    1: One or more verification checks failed
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    """Result of a single verification check.

    Attributes:
        check_name: Name of the verification check
        passed: Whether the check passed
        message: Human-readable message describing the result
        details: Additional structured data about the result
    """
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BatchVerificationReport:
    """Comprehensive report of all verification checks.

    Attributes:
        timestamp: ISO timestamp of when verification was run
        overall_passed: Whether all checks passed
        results: List of individual verification results
        summary: Human-readable summary of the report
    """
    timestamp: str
    overall_passed: bool
    results: List[VerificationResult]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation with nested results
        """
        return {
            'timestamp': self.timestamp,
            'overall_passed': self.overall_passed,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)


def verify_tests(quick: bool = True, verbose: bool = False, quiet: bool = False) -> VerificationResult:
    """Run tests and return pass/fail result.

    Uses the run_tests.py script to execute either smoke tests (quick mode)
    or smoke + unit tests (full mode).

    Args:
        quick: If True, run only smoke tests. If False, run smoke + unit.
        verbose: Show verbose test output
        quiet: Show minimal output

    Returns:
        VerificationResult with test counts and timing in details
    """
    check_name = "tests"

    # Determine which test suite to run
    test_suite = "smoke" if quick else "quick"  # quick = smoke + unit

    # Build command
    script_path = Path(__file__).parent / "run_tests.py"
    if not script_path.exists():
        return VerificationResult(
            check_name=check_name,
            passed=False,
            message="Test runner script not found",
            details={'script_path': str(script_path)}
        )

    cmd = [sys.executable, str(script_path), test_suite]

    if verbose:
        cmd.append('-v')
    elif quiet:
        cmd.append('-q')

    # Run tests with timeout
    timeout = 60 if quick else 300  # 1 min for smoke, 5 min for quick

    try:
        if not quiet:
            print(f"\n🧪 Running {test_suite} tests (timeout: {timeout}s)...")

        start_time = datetime.now()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent  # Run from repo root
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        passed = result.returncode == 0

        # Try to parse test counts from output
        details = {
            'suite': test_suite,
            'elapsed_seconds': elapsed,
            'return_code': result.returncode,
        }

        # Parse pytest output for test counts
        stdout = result.stdout
        stderr = result.stderr

        # Look for patterns like "10 passed" or "5 failed"
        import re

        # Pytest output format: "10 passed in 1.23s"
        passed_match = re.search(r'(\d+) passed', stdout)
        failed_match = re.search(r'(\d+) failed', stdout)
        error_match = re.search(r'(\d+) error', stdout)

        if passed_match:
            details['tests_passed'] = int(passed_match.group(1))
        if failed_match:
            details['tests_failed'] = int(failed_match.group(1))
        if error_match:
            details['tests_error'] = int(error_match.group(1))

        if passed:
            message = f"Tests passed ({test_suite} suite, {elapsed:.1f}s)"
            if 'tests_passed' in details:
                message = f"{details['tests_passed']} tests passed ({test_suite} suite, {elapsed:.1f}s)"
        else:
            message = f"Tests failed ({test_suite} suite, {elapsed:.1f}s)"
            if 'tests_failed' in details:
                message = f"{details.get('tests_failed', 0)} tests failed ({test_suite} suite, {elapsed:.1f}s)"
            # Include error output in details
            details['stderr'] = stderr[-500:] if len(stderr) > 500 else stderr  # Last 500 chars

        return VerificationResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details=details
        )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            check_name=check_name,
            passed=False,
            message=f"Tests timed out after {timeout}s",
            details={'timeout': timeout, 'suite': test_suite}
        )
    except Exception as e:
        return VerificationResult(
            check_name=check_name,
            passed=False,
            message=f"Test execution failed: {str(e)}",
            details={'error': str(e), 'error_type': type(e).__name__}
        )


def verify_no_conflicts(modified_files: Dict[str, List[str]], verbose: bool = False) -> VerificationResult:
    """Check that no files were modified by multiple agents.

    File conflicts can indicate race conditions or coordination failures
    in parallel agent work.

    Args:
        modified_files: Dict mapping agent_id to list of files they modified
        verbose: Show detailed conflict information

    Returns:
        VerificationResult with conflicts listed in details if any
    """
    check_name = "conflicts"

    if not modified_files:
        return VerificationResult(
            check_name=check_name,
            passed=True,
            message="No modifications to check (skipped)",
            details={'agent_count': 0}
        )

    # Build reverse mapping: file -> list of agents that modified it
    file_to_agents: Dict[str, List[str]] = {}
    for agent_id, files in modified_files.items():
        for file_path in files:
            if file_path not in file_to_agents:
                file_to_agents[file_path] = []
            file_to_agents[file_path].append(agent_id)

    # Find conflicts (files modified by multiple agents)
    conflicts = {
        file_path: agents
        for file_path, agents in file_to_agents.items()
        if len(agents) > 1
    }

    details = {
        'agent_count': len(modified_files),
        'total_files': len(file_to_agents),
        'conflict_count': len(conflicts),
    }

    if conflicts:
        details['conflicts'] = conflicts
        conflict_list = '\n'.join(
            f"  {file_path}: {', '.join(agents)}"
            for file_path, agents in conflicts.items()
        )
        message = f"Found {len(conflicts)} file conflict(s):\n{conflict_list}"
        passed = False
    else:
        message = f"No file conflicts detected ({len(file_to_agents)} files checked across {len(modified_files)} agents)"
        passed = True

    return VerificationResult(
        check_name=check_name,
        passed=passed,
        message=message,
        details=details
    )


def verify_git_status(verbose: bool = False, quiet: bool = False) -> VerificationResult:
    """Check git status for uncommitted changes.

    Warns about uncommitted .py files but doesn't fail the check.
    Provides information about the current git state.

    Args:
        verbose: Show detailed git status
        quiet: Suppress warnings

    Returns:
        VerificationResult with status details
    """
    check_name = "git_status"

    try:
        # Run git status --short
        result = subprocess.run(
            ['git', 'status', '--short'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode != 0:
            return VerificationResult(
                check_name=check_name,
                passed=False,
                message="Failed to run git status",
                details={'error': result.stderr}
            )

        # Parse output
        status_lines = [line for line in result.stdout.strip().split('\n') if line]

        # Categorize changes
        modified = []
        added = []
        deleted = []
        untracked = []

        for line in status_lines:
            if not line.strip():
                continue

            status_code = line[:2]
            file_path = line[3:].strip()

            if status_code.startswith('M'):
                modified.append(file_path)
            elif status_code.startswith('A'):
                added.append(file_path)
            elif status_code.startswith('D'):
                deleted.append(file_path)
            elif status_code.startswith('??'):
                untracked.append(file_path)

        details = {
            'modified_count': len(modified),
            'added_count': len(added),
            'deleted_count': len(deleted),
            'untracked_count': len(untracked),
        }

        if verbose or (not quiet and untracked):
            details['modified'] = modified
            details['added'] = added
            details['deleted'] = deleted
            details['untracked'] = untracked

        # Check for untracked .py files (warning only)
        untracked_py = [f for f in untracked if f.endswith('.py')]
        if untracked_py:
            details['untracked_python'] = untracked_py

        # Build message
        if not status_lines:
            message = "Working directory clean (no uncommitted changes)"
            passed = True
        else:
            parts = []
            if modified:
                parts.append(f"{len(modified)} modified")
            if added:
                parts.append(f"{len(added)} added")
            if deleted:
                parts.append(f"{len(deleted)} deleted")
            if untracked:
                parts.append(f"{len(untracked)} untracked")

            message = f"Git status: {', '.join(parts)}"

            # Warn about untracked .py files but don't fail
            if untracked_py and not quiet:
                message += f"\n  ⚠️  Untracked Python files: {', '.join(untracked_py[:5])}"
                if len(untracked_py) > 5:
                    message += f" ... and {len(untracked_py) - 5} more"

            passed = True  # Git status check always passes (informational)

        return VerificationResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details=details
        )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            check_name=check_name,
            passed=False,
            message="Git status check timed out",
            details={'timeout': 10}
        )
    except FileNotFoundError:
        return VerificationResult(
            check_name=check_name,
            passed=False,
            message="Git not found (not a git repository?)",
            details={'error': 'git command not found'}
        )
    except Exception as e:
        return VerificationResult(
            check_name=check_name,
            passed=False,
            message=f"Git status check failed: {str(e)}",
            details={'error': str(e), 'error_type': type(e).__name__}
        )


def run_verification(
    quick: bool = True,
    modified_files: Optional[Dict[str, List[str]]] = None,
    checks: Optional[List[str]] = None,
    verbose: bool = False,
    quiet: bool = False
) -> BatchVerificationReport:
    """Run all verification checks and generate comprehensive report.

    Args:
        quick: Quick test mode (smoke tests only)
        modified_files: Optional agent file modifications to check
        checks: Optional list of specific checks to run (None = all)
        verbose: Show detailed output
        quiet: Show minimal output

    Returns:
        Comprehensive verification report with all check results
    """
    timestamp = datetime.now().isoformat()
    results = []

    # Determine which checks to run
    all_checks = ['tests', 'conflicts', 'git']
    if checks is None:
        checks_to_run = all_checks
    else:
        checks_to_run = checks

    if not quiet:
        print("\n" + "=" * 70)
        print(" Batch Verification Report")
        print("=" * 70)
        print(f" Timestamp: {timestamp}")
        print(f" Checks: {', '.join(checks_to_run)}")
        print(f" Mode: {'Quick (smoke tests)' if quick else 'Full (smoke + unit tests)'}")
        print("=" * 70)

    # Run tests check
    if 'tests' in checks_to_run:
        result = verify_tests(quick=quick, verbose=verbose, quiet=quiet)
        results.append(result)

        if not quiet:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} - {result.check_name}: {result.message}")

    # Run conflicts check
    if 'conflicts' in checks_to_run:
        if modified_files is None:
            modified_files = {}

        result = verify_no_conflicts(modified_files, verbose=verbose)
        results.append(result)

        if not quiet:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} - {result.check_name}: {result.message}")

    # Run git status check
    if 'git' in checks_to_run:
        result = verify_git_status(verbose=verbose, quiet=quiet)
        results.append(result)

        if not quiet:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} - {result.check_name}: {result.message}")

    # Determine overall pass/fail
    overall_passed = all(r.passed for r in results)

    # Generate summary
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    if overall_passed:
        summary = f"All {len(results)} verification checks passed"
    else:
        summary = f"{failed_count} of {len(results)} verification checks failed"

    if not quiet:
        print("\n" + "=" * 70)
        print(f" Summary: {summary}")
        print("=" * 70)

        if overall_passed:
            print("\n🎉 Batch verification PASSED!")
        else:
            print("\n💥 Batch verification FAILED!")
            print("\nFailed checks:")
            for r in results:
                if not r.passed:
                    print(f"  - {r.check_name}: {r.message}")

    return BatchVerificationReport(
        timestamp=timestamp,
        overall_passed=overall_passed,
        results=results,
        summary=summary
    )


def load_modified_files(file_path: str) -> Dict[str, List[str]]:
    """Load modified files JSON from disk.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary mapping agent IDs to file lists

    Raises:
        ValueError: If file format is invalid
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Validate format
        if not isinstance(data, dict):
            raise ValueError("Modified files must be a JSON object")

        for agent_id, files in data.items():
            if not isinstance(files, list):
                raise ValueError(f"Files for agent {agent_id} must be a list")
            if not all(isinstance(f, str) for f in files):
                raise ValueError(f"File paths for agent {agent_id} must be strings")

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except FileNotFoundError:
        raise ValueError(f"Modified files file not found: {file_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch verification for automated agent work',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run smoke tests only (default)'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Run smoke + unit tests'
    )

    parser.add_argument(
        '--check',
        choices=['tests', 'conflicts', 'git'],
        help='Run specific check only'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    parser.add_argument(
        '--modified-files',
        type=str,
        help='JSON file with agent file modifications'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Show minimal output'
    )

    args = parser.parse_args()

    # Determine test mode (default to quick)
    quick = True
    if args.full:
        quick = False

    # Load modified files if provided
    modified_files = None
    if args.modified_files:
        try:
            modified_files = load_modified_files(args.modified_files)
        except ValueError as e:
            print(f"Error loading modified files: {e}", file=sys.stderr)
            sys.exit(1)

    # Determine which checks to run
    checks = None
    if args.check:
        checks = [args.check]

    # Run verification
    try:
        report = run_verification(
            quick=quick,
            modified_files=modified_files,
            checks=checks,
            verbose=args.verbose,
            quiet=args.quiet or args.json  # Suppress normal output if JSON mode
        )

        # Output JSON if requested
        if args.json:
            print(report.to_json())

        # Exit with appropriate code
        sys.exit(0 if report.overall_passed else 1)

    except Exception as e:
        if args.json:
            error_report = {
                'timestamp': datetime.now().isoformat(),
                'overall_passed': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
            print(json.dumps(error_report, indent=2))
        else:
            print(f"\n💥 Verification failed with error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
