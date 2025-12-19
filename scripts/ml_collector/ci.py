"""
CI integration module for ML Data Collector

Handles auto-capture of CI results from GitHub Actions.
"""

import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def ci_autocapture() -> bool:
    """Auto-capture CI results from GitHub Actions environment.

    Reads environment variables set by GitHub Actions and records
    CI results for the current commit.

    Environment variables used:
        GITHUB_SHA: Commit hash
        CI_RESULT: pass/fail/error (set by workflow)
        CI_TESTS_PASSED: Number of tests passed
        CI_TESTS_FAILED: Number of tests failed
        CI_COVERAGE: Coverage percentage
        CI_DURATION: Duration in seconds

    Returns:
        True if CI result was recorded, False otherwise.
    """
    from .commit import update_commit_ci_result

    commit_hash = os.getenv('GITHUB_SHA')
    if not commit_hash:
        # Try to get from git if not in Actions
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
        except Exception:
            pass

    if not commit_hash:
        print("❌ No commit hash available")
        return False

    # Get CI result from environment
    result = os.getenv('CI_RESULT', 'unknown')
    if result not in ('pass', 'fail', 'error', 'pending', 'unknown'):
        result = 'unknown'

    # Collect details from environment
    details = {}

    tests_passed = os.getenv('CI_TESTS_PASSED')
    if tests_passed:
        try:
            details['tests_passed'] = int(tests_passed)
        except ValueError:
            pass

    tests_failed = os.getenv('CI_TESTS_FAILED')
    if tests_failed:
        try:
            details['tests_failed'] = int(tests_failed)
        except ValueError:
            pass

    coverage = os.getenv('CI_COVERAGE')
    if coverage:
        try:
            details['coverage'] = float(coverage)
        except ValueError:
            pass

    duration = os.getenv('CI_DURATION')
    if duration:
        try:
            details['duration_seconds'] = float(duration)
        except ValueError:
            pass

    # Get job name and workflow
    details['workflow'] = os.getenv('GITHUB_WORKFLOW', 'unknown')
    details['job'] = os.getenv('GITHUB_JOB', 'unknown')
    details['run_id'] = os.getenv('GITHUB_RUN_ID', '')
    details['run_number'] = os.getenv('GITHUB_RUN_NUMBER', '')

    # Update the commit
    success = update_commit_ci_result(commit_hash, result, details if details else None)

    if success:
        print(f"✅ CI result recorded for {commit_hash[:8]}: {result}")
        if details.get('coverage'):
            print(f"   Coverage: {details['coverage']}%")
        if details.get('tests_passed') is not None:
            print(f"   Tests: {details.get('tests_passed', 0)} passed, {details.get('tests_failed', 0)} failed")
    else:
        # Commit might not be in our data yet - that's OK
        print(f"⚠️  Commit {commit_hash[:8]} not found in ML data (may not be collected yet)")

    return success
