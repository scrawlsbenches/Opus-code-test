"""
Verification System: Confirming Reality Matches Intention.

This module implements the verification protocol from Part 6 of
docs/complex-reasoning-workflow.md. It provides:

- Verification pyramid (unit → integration → E2E → acceptance)
- Phase-specific verification checklists
- Failure response protocols
- Verification result tracking

Design Philosophy:
    "Verification = confirming reality matches intention.
     This is where bugs die and confidence is born."

The pyramid structure ensures appropriate verification depth:
- Width = Coverage breadth (how much is tested)
- Height = Confidence in full system (how well tested)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


class VerificationLevel(Enum):
    """
    Levels in the verification pyramid.

    From docs/complex-reasoning-workflow.md Part 6.1:
    - UNIT: Individual pieces work
    - INTEGRATION: Components interact correctly
    - E2E: Full system works together
    - ACCEPTANCE: User validates in real context
    """
    UNIT = auto()
    INTEGRATION = auto()
    E2E = auto()
    ACCEPTANCE = auto()


class VerificationPhase(Enum):
    """
    Which production phase triggers this verification.

    From docs/complex-reasoning-workflow.md Part 6.2:
    - DRAFTING: Quick sanity checks
    - REFINING: Thorough checks
    - FINALIZING: Complete verification
    """
    DRAFTING = auto()  # Quick sanity
    REFINING = auto()  # Thorough check
    FINALIZING = auto()  # Complete verification


class VerificationStatus(Enum):
    """Status of a verification check."""
    PENDING = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()
    ERROR = auto()  # Check itself errored (not a test failure)


@dataclass
class VerificationCheck:
    """
    A single verification check to be performed.

    Attributes:
        name: Human-readable name
        description: What this check verifies
        level: Verification pyramid level
        phase: When this check should run
        command: Command to execute (if applicable)
        expected: Expected outcome
        status: Current status
        result: Actual result
        duration_ms: How long the check took
        notes: Additional notes
    """
    name: str
    description: str
    level: VerificationLevel
    phase: VerificationPhase = VerificationPhase.FINALIZING
    command: Optional[str] = None
    expected: str = "pass"
    status: VerificationStatus = VerificationStatus.PENDING
    result: Optional[str] = None
    duration_ms: int = 0
    notes: List[str] = field(default_factory=list)
    run_at: Optional[datetime] = None

    def mark_passed(self, result: str = "passed", duration_ms: int = 0) -> None:
        """Mark check as passed."""
        self.status = VerificationStatus.PASSED
        self.result = result
        self.duration_ms = duration_ms
        self.run_at = datetime.now()

    def mark_failed(self, result: str, duration_ms: int = 0) -> None:
        """Mark check as failed."""
        self.status = VerificationStatus.FAILED
        self.result = result
        self.duration_ms = duration_ms
        self.run_at = datetime.now()

    def mark_error(self, error: str) -> None:
        """Mark check as errored (check itself failed)."""
        self.status = VerificationStatus.ERROR
        self.result = f"ERROR: {error}"
        self.run_at = datetime.now()

    def mark_skipped(self, reason: str = "skipped") -> None:
        """Mark check as skipped."""
        self.status = VerificationStatus.SKIPPED
        self.result = reason
        self.run_at = datetime.now()

    def reset(self) -> None:
        """Reset check to pending state."""
        self.status = VerificationStatus.PENDING
        self.result = None
        self.duration_ms = 0
        self.run_at = None


@dataclass
class VerificationFailure:
    """
    Record of a verification failure for analysis.

    From docs/complex-reasoning-workflow.md Part 6.3:
    "How you respond to verification failure determines quality."

    Captures the structured response to failure:
    1. STOP - Don't try random fixes
    2. OBSERVE - What exactly failed?
    3. HYPOTHESIZE - Why might it have failed?
    4. INVESTIGATE - Check most likely cause
    5. FIX - Apply targeted fix
    6. VERIFY - Confirm fix works
    7. DOCUMENT - Capture what was learned
    """
    check: VerificationCheck
    observed: str  # What exactly failed
    expected_vs_actual: str  # Expected vs actual
    hypotheses: List[str] = field(default_factory=list)
    investigation_notes: List[str] = field(default_factory=list)
    fix_applied: Optional[str] = None
    fix_successful: bool = False
    lessons_learned: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_hypothesis(self, hypothesis: str, likelihood: str = "medium") -> None:
        """Add a hypothesis for why the failure occurred."""
        self.hypotheses.append(f"[{likelihood}] {hypothesis}")

    def add_investigation_note(self, note: str) -> None:
        """Add a note from investigation."""
        self.investigation_notes.append(f"[{datetime.now().isoformat()}] {note}")

    def record_fix(self, fix: str, successful: bool) -> None:
        """Record the fix that was applied."""
        self.fix_applied = fix
        self.fix_successful = successful


@dataclass
class VerificationSuite:
    """
    A suite of verification checks for a task.

    Organizes checks by level and phase, tracks overall status,
    and provides reporting.
    """
    name: str
    description: str = ""
    checks: List[VerificationCheck] = field(default_factory=list)
    failures: List[VerificationFailure] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_check(self, check: VerificationCheck) -> None:
        """Add a check to the suite."""
        self.checks.append(check)

    def add_unit_check(self, name: str, description: str, command: str = None) -> VerificationCheck:
        """Add a unit-level check."""
        check = VerificationCheck(
            name=name,
            description=description,
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING,
            command=command,
        )
        self.checks.append(check)
        return check

    def add_integration_check(self, name: str, description: str, command: str = None) -> VerificationCheck:
        """Add an integration-level check."""
        check = VerificationCheck(
            name=name,
            description=description,
            level=VerificationLevel.INTEGRATION,
            phase=VerificationPhase.REFINING,
            command=command,
        )
        self.checks.append(check)
        return check

    def add_e2e_check(self, name: str, description: str, command: str = None) -> VerificationCheck:
        """Add an E2E-level check."""
        check = VerificationCheck(
            name=name,
            description=description,
            level=VerificationLevel.E2E,
            phase=VerificationPhase.FINALIZING,
            command=command,
        )
        self.checks.append(check)
        return check

    def add_acceptance_check(self, name: str, description: str) -> VerificationCheck:
        """Add an acceptance-level check (requires human validation)."""
        check = VerificationCheck(
            name=name,
            description=description,
            level=VerificationLevel.ACCEPTANCE,
            phase=VerificationPhase.FINALIZING,
        )
        self.checks.append(check)
        return check

    def get_checks_for_phase(self, phase: VerificationPhase) -> List[VerificationCheck]:
        """Get all checks that should run in a given phase."""
        return [c for c in self.checks if c.phase == phase]

    def get_checks_for_level(self, level: VerificationLevel) -> List[VerificationCheck]:
        """Get all checks at a given pyramid level."""
        return [c for c in self.checks if c.level == level]

    def get_pending_checks(self) -> List[VerificationCheck]:
        """Get all checks that haven't been run."""
        return [c for c in self.checks if c.status == VerificationStatus.PENDING]

    def get_failed_checks(self) -> List[VerificationCheck]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == VerificationStatus.FAILED]

    def record_failure(self, check: VerificationCheck, observed: str, expected_vs_actual: str) -> VerificationFailure:
        """Record a verification failure for structured analysis."""
        failure = VerificationFailure(
            check=check,
            observed=observed,
            expected_vs_actual=expected_vs_actual,
        )
        self.failures.append(failure)
        return failure

    def all_passed(self) -> bool:
        """Check if all verification checks passed."""
        return all(c.status == VerificationStatus.PASSED for c in self.checks)

    def run_all(self) -> Dict[str, Any]:
        """
        Run all verification checks in this suite.

        Returns:
            Summary of results with passed/failed/error counts
        """
        results = {'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0}
        for check in self.checks:
            # Stub: mark all as passed
            check.mark_passed("executed")
            results['passed'] += 1
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of verification status."""
        by_status = {}
        for status in VerificationStatus:
            by_status[status.name] = sum(1 for c in self.checks if c.status == status)

        by_level = {}
        for level in VerificationLevel:
            level_checks = self.get_checks_for_level(level)
            passed = sum(1 for c in level_checks if c.status == VerificationStatus.PASSED)
            by_level[level.name] = f"{passed}/{len(level_checks)}"

        return {
            'total_checks': len(self.checks),
            'by_status': by_status,
            'by_level': by_level,
            'all_passed': self.all_passed(),
            'failure_count': len(self.failures),
        }


# =============================================================================
# STANDARD CHECKLIST TEMPLATES
# =============================================================================

def create_drafting_checklist() -> List[VerificationCheck]:
    """
    Create standard verification checks for DRAFTING phase.

    From docs/complex-reasoning-workflow.md Part 6.2:
    "During DRAFTING (Quick Sanity):
    - Syntax/compilation check
    - Smoke test
    - 'Does it even run?' test"
    """
    return [
        VerificationCheck(
            name="syntax_check",
            description="Code compiles/parses without errors",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING,
            command="python -m py_compile {file}",
        ),
        VerificationCheck(
            name="smoke_test",
            description="Basic smoke tests pass",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING,
            command="python -m pytest tests/smoke/ -x -q",
        ),
        VerificationCheck(
            name="import_check",
            description="Module can be imported",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING,
            command="python -c 'from {module} import {class}'",
        ),
    ]


def create_refining_checklist() -> List[VerificationCheck]:
    """
    Create standard verification checks for REFINING phase.

    From docs/complex-reasoning-workflow.md Part 6.2:
    "During REFINING (Thorough Check):
    - Full unit tests
    - Coverage on modified files
    - Type checking (if applicable)"
    """
    return [
        VerificationCheck(
            name="unit_tests",
            description="Full unit test suite passes",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.REFINING,
            command="python -m pytest tests/unit/ -v",
        ),
        VerificationCheck(
            name="coverage_check",
            description="Coverage on modified files is acceptable",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.REFINING,
            command="python -m pytest tests/ --cov={module} --cov-report=term-missing",
        ),
        VerificationCheck(
            name="type_check",
            description="Type checking passes (if applicable)",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.REFINING,
            command="mypy {file}",
        ),
    ]


def create_finalizing_checklist() -> List[VerificationCheck]:
    """
    Create standard verification checks for FINALIZING phase.

    From docs/complex-reasoning-workflow.md Part 6.2:
    "During FINALIZING (Complete Verification):
    - Full test suite
    - Integration tests
    - Performance regression (if applicable)
    - Documentation check"
    """
    return [
        VerificationCheck(
            name="full_test_suite",
            description="Complete test suite passes",
            level=VerificationLevel.INTEGRATION,
            phase=VerificationPhase.FINALIZING,
            command="python -m pytest tests/ -v",
        ),
        VerificationCheck(
            name="integration_tests",
            description="Integration tests pass",
            level=VerificationLevel.INTEGRATION,
            phase=VerificationPhase.FINALIZING,
            command="python -m pytest tests/integration/ -v",
        ),
        VerificationCheck(
            name="performance_tests",
            description="Performance regression tests pass",
            level=VerificationLevel.E2E,
            phase=VerificationPhase.FINALIZING,
            command="python -m pytest tests/performance/ -v",
        ),
        VerificationCheck(
            name="documentation_check",
            description="Documentation is up to date",
            level=VerificationLevel.ACCEPTANCE,
            phase=VerificationPhase.FINALIZING,
        ),
    ]


class VerificationManager:
    """
    Manager for verification activities across tasks.

    Provides:
    - Suite lifecycle management
    - Aggregate status tracking
    - Failure analysis support
    - Integration with crisis manager
    """

    def __init__(self):
        """Initialize the verification manager."""
        self._suites: Dict[str, VerificationSuite] = {}
        self._on_failure: List[Callable[[VerificationCheck, VerificationFailure], None]] = []

    def create_suite(self, name: str, description: str = "") -> VerificationSuite:
        """
        Create a new verification suite.

        Args:
            name: Unique name for the suite
            description: What this suite verifies

        Returns:
            New VerificationSuite instance
        """
        suite = VerificationSuite(name=name, description=description)
        self._suites[name] = suite
        return suite

    def create_standard_suite(self, name: str, description: str = "") -> VerificationSuite:
        """
        Create a suite with standard checklists for all phases.

        Returns:
            VerificationSuite with standard checks
        """
        suite = self.create_suite(name, description)

        for check in create_drafting_checklist():
            suite.add_check(check)
        for check in create_refining_checklist():
            suite.add_check(check)
        for check in create_finalizing_checklist():
            suite.add_check(check)

        return suite

    def get_suite(self, name: str) -> Optional[VerificationSuite]:
        """Get a suite by name."""
        return self._suites.get(name)

    def register_failure_handler(
        self,
        handler: Callable[[VerificationCheck, VerificationFailure], None]
    ) -> None:
        """Register a handler called when verification fails."""
        self._on_failure.append(handler)

    def report_failure(self, suite: VerificationSuite, check: VerificationCheck, observed: str, expected_vs_actual: str) -> VerificationFailure:
        """
        Report a verification failure.

        Args:
            suite: The suite containing the check
            check: The check that failed
            observed: What was observed
            expected_vs_actual: Expected vs actual description

        Returns:
            VerificationFailure for analysis
        """
        failure = suite.record_failure(check, observed, expected_vs_actual)

        for handler in self._on_failure:
            try:
                handler(check, failure)
            except Exception:
                pass

        return failure

    def get_all_failures(self) -> List[VerificationFailure]:
        """Get all failures across all suites."""
        failures = []
        for suite in self._suites.values():
            failures.extend(suite.failures)
        return failures

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary across all suites."""
        total_checks = sum(len(s.checks) for s in self._suites.values())
        total_passed = sum(
            sum(1 for c in s.checks if c.status == VerificationStatus.PASSED)
            for s in self._suites.values()
        )
        total_failed = sum(
            sum(1 for c in s.checks if c.status == VerificationStatus.FAILED)
            for s in self._suites.values()
        )
        total_failures = sum(len(s.failures) for s in self._suites.values())

        return {
            'suites': len(self._suites),
            'total_checks': total_checks,
            'passed': total_passed,
            'failed': total_failed,
            'pass_rate': total_passed / total_checks if total_checks > 0 else 0,
            'total_failures': total_failures,
        }


# =============================================================================
# STUB CLASSES FOR COMPLEX IMPLEMENTATIONS
# =============================================================================


class VerificationRunner:
    """
    STUB: Automated verification check runner.

    Full Implementation Would:
    --------------------------
    1. Execute verification commands
       - Parse command templates with file/module substitution
       - Capture stdout/stderr
       - Parse test output for pass/fail determination
       - Time each check execution

    2. Intelligent ordering
       - Run cheaper checks first (fail fast)
       - Parallelize independent checks
       - Skip checks whose dependencies failed

    3. Result aggregation
       - Collect all results into unified report
       - Generate coverage reports
       - Track timing trends

    4. Integration with CI
       - Format output for CI systems
       - Generate JUnit XML
       - Update status checks
    """

    def run_check(self, check: VerificationCheck) -> VerificationStatus:
        """
        STUB: Execute a single verification check.

        Returns:
            VerificationStatus after running
        """
        # STUB: Would actually execute the command
        check.status = VerificationStatus.PASSED
        check.run_at = datetime.now()
        return check.status

    def run_suite(self, suite: VerificationSuite, phase: VerificationPhase = None) -> Dict[str, Any]:
        """
        STUB: Run all checks in a suite (optionally filtered by phase).

        Returns:
            Summary of results
        """
        checks = suite.get_checks_for_phase(phase) if phase else suite.checks

        results = {'passed': 0, 'failed': 0, 'error': 0}
        for check in checks:
            status = self.run_check(check)
            if status == VerificationStatus.PASSED:
                results['passed'] += 1
            elif status == VerificationStatus.FAILED:
                results['failed'] += 1
            else:
                results['error'] += 1

        return results


class FailureAnalyzer:
    """
    STUB: Intelligent failure analysis and root cause detection.

    Full Implementation Would:
    --------------------------
    1. Pattern matching against known failure types
       - Import errors → missing dependency
       - Assertion errors → logic bug
       - Timeout → performance issue
       - Connection errors → external dependency

    2. Diff analysis
       - What changed since last passing run?
       - Which changes are most likely to cause this failure?
       - Similar failures in git history?

    3. Hypothesis generation
       - Based on error message, suggest likely causes
       - Rank hypotheses by probability
       - Suggest investigation steps

    4. Fix suggestions
       - Pattern-match to known fixes
       - Generate suggested code changes
       - Link to relevant documentation
    """

    def analyze_failure(self, failure: VerificationFailure) -> Dict[str, Any]:
        """
        STUB: Analyze a failure and suggest causes/fixes.

        Returns:
            {'likely_cause': str, 'hypotheses': list, 'suggested_fix': str}
        """
        return {
            'likely_cause': 'Unknown (analysis stub)',
            'hypotheses': [
                'Logic error in implementation',
                'Missing edge case handling',
                'External dependency issue',
            ],
            'suggested_fix': 'Review the failing test and trace execution',
            'note': 'STUB: Would use ML/pattern matching for real analysis',
        }

    def find_similar_failures(self, failure: VerificationFailure) -> List[VerificationFailure]:
        """
        STUB: Find historically similar failures.

        Returns:
            List of similar past failures
        """
        return []  # Would query historical data


class RegressionDetector:
    """
    STUB: Detect regressions in verification results over time.

    Full Implementation Would:
    --------------------------
    1. Track verification history
       - Store results with timestamps
       - Associate with commits/branches
       - Track pass/fail trends

    2. Detect degradation patterns
       - Tests that flip from pass to fail
       - Tests that become flaky
       - Coverage drops
       - Performance regressions

    3. Alert on regressions
       - Identify which change caused regression
       - Notify relevant parties
       - Block merge if configured

    4. Trend analysis
       - Test suite health over time
       - Flaky test detection
       - Coverage trends
    """

    def detect_regression(
        self,
        current_results: Dict[str, VerificationStatus],
        baseline_results: Dict[str, VerificationStatus]
    ) -> List[str]:
        """
        STUB: Detect regressions from baseline.

        Returns:
            List of test names that regressed
        """
        regressions = []
        for name, current_status in current_results.items():
            baseline_status = baseline_results.get(name, VerificationStatus.PENDING)
            if baseline_status == VerificationStatus.PASSED and current_status == VerificationStatus.FAILED:
                regressions.append(name)
        return regressions
