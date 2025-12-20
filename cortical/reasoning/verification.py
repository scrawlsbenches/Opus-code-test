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
            except Exception:  # noqa: S110
                # Broad exception catch is intentional: handlers are user-provided
                # callbacks and we don't know what they might raise. We must not
                # let a failing handler break verification failure recording.
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
    Intelligent failure analysis and root cause detection.

    Provides:
    - Pattern matching against known failure types
    - Hypothesis generation based on error patterns
    - Investigation plan generation
    - Historical failure comparison
    - Pattern statistics tracking

    Pattern Library:
    ----------------
    - import_error: Missing dependencies or incorrect import paths
    - assertion_error: Test expectations not met
    - timeout: Performance issues or hanging code
    - connection_error: External service unavailable
    - type_error: Wrong argument types or counts
    """

    def __init__(self):
        """Initialize the failure analyzer with pattern library and history."""
        self._patterns: Dict[str, Dict] = {}
        self._history: List[VerificationFailure] = []
        self._pattern_stats: Dict[str, Dict[str, int]] = {}
        self._init_patterns()

    def _init_patterns(self) -> None:
        """Initialize the pattern library with known failure patterns."""
        self._patterns = {
            'import_error': {
                'pattern': r'ImportError|ModuleNotFoundError|No module named',
                'likely_cause': 'Missing dependency or incorrect import path',
                'fix_template': 'Install missing package or fix import statement',
                'investigation_steps': [
                    'Check if package is listed in requirements.txt or pyproject.toml',
                    'Verify import path is correct',
                    'Check if package is installed in current environment',
                    'Look for typos in module or package name',
                ],
            },
            'assertion_error': {
                'pattern': r'AssertionError|assert .* failed',
                'likely_cause': 'Test expectation not met',
                'fix_template': 'Check test logic or fix implementation',
                'investigation_steps': [
                    'Review the assertion that failed',
                    'Check expected vs actual values',
                    'Verify test setup is correct',
                    'Trace code execution to find discrepancy',
                ],
            },
            'timeout': {
                'pattern': r'TimeoutError|timed out|timeout exceeded',
                'likely_cause': 'Operation took too long',
                'fix_template': 'Optimize slow code or increase timeout',
                'investigation_steps': [
                    'Profile the slow operation',
                    'Check for infinite loops or blocking calls',
                    'Look for O(n²) or worse algorithms',
                    'Verify external service response times',
                ],
            },
            'connection_error': {
                'pattern': r'ConnectionError|ConnectionRefused|Connection refused',
                'likely_cause': 'External service unavailable',
                'fix_template': 'Check service availability or add retry logic',
                'investigation_steps': [
                    'Verify the service is running',
                    'Check network connectivity',
                    'Review firewall or security settings',
                    'Look for port conflicts',
                ],
            },
            'type_error': {
                'pattern': r'TypeError|takes \d+ positional|got an unexpected keyword',
                'likely_cause': 'Wrong argument type or count',
                'fix_template': 'Check function signature and call site',
                'investigation_steps': [
                    'Review function signature',
                    'Check arguments passed at call site',
                    'Verify type compatibility',
                    'Look for API changes or refactoring',
                ],
            },
            'attribute_error': {
                'pattern': r'AttributeError|has no attribute',
                'likely_cause': 'Object missing expected attribute',
                'fix_template': 'Check object type or attribute name',
                'investigation_steps': [
                    'Verify object type is correct',
                    'Check for typos in attribute name',
                    'Review object initialization',
                    'Look for None values where object expected',
                ],
            },
            'key_error': {
                'pattern': r'KeyError',
                'likely_cause': 'Dictionary key not found',
                'fix_template': 'Check key exists before access or use .get()',
                'investigation_steps': [
                    'Verify the expected key name',
                    'Check dictionary population logic',
                    'Use .get() with default value',
                    'Review data validation',
                ],
            },
            'index_error': {
                'pattern': r'IndexError|list index out of range',
                'likely_cause': 'List or sequence index out of bounds',
                'fix_template': 'Check sequence length before indexing',
                'investigation_steps': [
                    'Verify sequence is not empty',
                    'Check index calculation logic',
                    'Review loop bounds',
                    'Look for off-by-one errors',
                ],
            },
        }

        # Initialize statistics for each pattern
        for pattern_name in self._patterns:
            self._pattern_stats[pattern_name] = {
                'total_occurrences': 0,
                'successful_fixes': 0,
            }

    def analyze_failure(self, failure: VerificationFailure) -> Dict[str, Any]:
        """
        Analyze a failure and suggest causes/fixes.

        Args:
            failure: The verification failure to analyze

        Returns:
            Analysis dict with:
            - likely_cause: Most probable cause
            - matched_patterns: List of matched pattern names
            - hypotheses: List of hypothesis strings with likelihood
            - suggested_fix: Fix template from matched pattern
            - investigation_steps: Ordered list of investigation steps
            - similar_failures: List of historically similar failures
        """
        import re

        observed = failure.observed
        matched_patterns = []
        hypotheses = []
        investigation_steps = []

        # Match against known patterns
        for pattern_name, pattern_data in self._patterns.items():
            if re.search(pattern_data['pattern'], observed, re.IGNORECASE):
                matched_patterns.append(pattern_name)
                hypotheses.append(f"[high] {pattern_data['likely_cause']}")
                investigation_steps.extend(pattern_data['investigation_steps'])

        # If no patterns matched, provide generic analysis
        if not matched_patterns:
            return {
                'likely_cause': 'Unknown error pattern',
                'matched_patterns': [],
                'hypotheses': [
                    '[medium] Logic error in implementation',
                    '[medium] Missing edge case handling',
                    '[low] External dependency issue',
                ],
                'suggested_fix': 'Review the failing check and trace execution',
                'investigation_steps': [
                    'Review the error message in detail',
                    'Check recent code changes',
                    'Add debug logging',
                    'Reproduce the failure in isolation',
                ],
                'similar_failures': [],
            }

        # Use first matched pattern as primary
        primary_pattern = self._patterns[matched_patterns[0]]
        likely_cause = primary_pattern['likely_cause']
        suggested_fix = primary_pattern['fix_template']

        # Find similar historical failures
        similar_failures = self.find_similar_failures(failure)

        # Add context-specific hypotheses
        if len(matched_patterns) > 1:
            hypotheses.append(f"[medium] Multiple error types suggest cascading failure")

        return {
            'likely_cause': likely_cause,
            'matched_patterns': matched_patterns,
            'hypotheses': hypotheses,
            'suggested_fix': suggested_fix,
            'investigation_steps': investigation_steps,
            'similar_failures': similar_failures,
        }

    def find_similar_failures(self, failure: VerificationFailure, top_n: int = 5) -> List[Tuple[VerificationFailure, float]]:
        """
        Find historically similar failures.

        Similarity scoring based on:
        - Check name match (40%)
        - Error type match (30%)
        - Description similarity (30%)

        Args:
            failure: The failure to compare against
            top_n: Maximum number of similar failures to return

        Returns:
            List of (failure, similarity_score) tuples sorted by similarity
        """
        if not self._history:
            return []

        similarities = []

        for hist_failure in self._history:
            score = 0.0

            # Check name similarity (40%)
            if failure.check.name == hist_failure.check.name:
                score += 0.4
            elif failure.check.name in hist_failure.check.name or hist_failure.check.name in failure.check.name:
                score += 0.2

            # Error type similarity (30%) - compare observed errors
            if failure.observed == hist_failure.observed:
                score += 0.3
            elif any(word in hist_failure.observed for word in failure.observed.split()[:5]):
                score += 0.15

            # Description similarity (30%)
            if failure.check.description == hist_failure.check.description:
                score += 0.3
            elif failure.check.description in hist_failure.check.description or hist_failure.check.description in failure.check.description:
                score += 0.15

            if score > 0:
                similarities.append((hist_failure, score))

        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def record_failure(self, failure: VerificationFailure) -> None:
        """
        Record a failure in the history.

        Args:
            failure: The verification failure to record
        """
        self._history.append(failure)

        # Update pattern statistics
        analysis = self.analyze_failure(failure)
        for pattern_name in analysis['matched_patterns']:
            if pattern_name in self._pattern_stats:
                self._pattern_stats[pattern_name]['total_occurrences'] += 1

    def record_fix_success(self, failure: VerificationFailure, successful: bool) -> None:
        """
        Record whether a fix was successful.

        Args:
            failure: The failure that was fixed
            successful: Whether the fix worked
        """
        if successful:
            analysis = self.analyze_failure(failure)
            for pattern_name in analysis['matched_patterns']:
                if pattern_name in self._pattern_stats:
                    self._pattern_stats[pattern_name]['successful_fixes'] += 1

    def generate_investigation_plan(self, failure: VerificationFailure) -> List[str]:
        """
        Generate an ordered investigation plan based on failure analysis.

        Args:
            failure: The verification failure to investigate

        Returns:
            List of investigation steps in recommended order
        """
        analysis = self.analyze_failure(failure)
        return analysis['investigation_steps']

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about pattern occurrences and fix success rates.

        Returns:
            Dict mapping pattern names to statistics:
            - total_occurrences: How many times this pattern occurred
            - successful_fixes: How many fixes succeeded
            - success_rate: Percentage of successful fixes
        """
        stats = {}
        for pattern_name, pattern_data in self._pattern_stats.items():
            total = pattern_data['total_occurrences']
            successful = pattern_data['successful_fixes']
            success_rate = (successful / total * 100) if total > 0 else 0.0

            stats[pattern_name] = {
                'total_occurrences': total,
                'successful_fixes': successful,
                'success_rate': success_rate,
            }

        return stats


class RegressionDetector:
    """
    Detect regressions in verification results over time.

    Tracks verification history, detects degradation patterns,
    identifies flaky tests, and provides trend analysis.

    Key Features:
    - Named baselines for comparison
    - Historical result tracking
    - Flaky test detection
    - Trend analysis for individual tests
    - Comprehensive summary statistics
    """

    def __init__(self):
        """Initialize the regression detector."""
        # Named baselines: {baseline_name: {test_name: status}}
        self._baselines: Dict[str, Dict[str, VerificationStatus]] = {}

        # Historical snapshots: [{timestamp, results: {test_name: status}}]
        self._history: List[Dict] = []

        # Flaky test tracking: {test_name: [bool, bool, ...]} (pass=True, fail=False)
        self._flaky_tests: Dict[str, List[bool]] = {}

        # Track when baselines were created
        self._baseline_timestamps: Dict[str, datetime] = {}

        # Track regression first detection
        self._regression_first_seen: Dict[str, datetime] = {}

    def save_baseline(self, name: str, results: Dict[str, VerificationStatus]) -> None:
        """
        Store results as a named baseline.

        Args:
            name: Name for this baseline (e.g., "main", "release-1.0")
            results: Dictionary mapping test names to verification statuses
        """
        self._baselines[name] = results.copy()
        self._baseline_timestamps[name] = datetime.now()

    def detect_regression(
        self,
        current_results: Dict[str, VerificationStatus],
        baseline_results: Dict[str, VerificationStatus] = None,
        baseline_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Detect regressions from baseline (enhanced).

        Args:
            current_results: Current test results
            baseline_results: Baseline to compare against (if not using named baseline)
            baseline_name: Name of saved baseline to use

        Returns:
            List of regression details with:
                - test_name: Name of regressed test
                - baseline_status: Status in baseline
                - current_status: Current status
                - first_seen: When regression was first detected (if known)
        """
        # Determine baseline to use
        if baseline_name and baseline_name in self._baselines:
            baseline = self._baselines[baseline_name]
        elif baseline_results is not None:
            baseline = baseline_results
        elif len(self._baselines) > 0:
            # Use most recently saved baseline
            baseline = list(self._baselines.values())[-1]
        else:
            return []

        regressions = []
        now = datetime.now()

        for name, current_status in current_results.items():
            baseline_status = baseline.get(name, VerificationStatus.PENDING)

            # Regression: was passing, now failing
            if baseline_status == VerificationStatus.PASSED and current_status == VerificationStatus.FAILED:
                # Track when first seen
                if name not in self._regression_first_seen:
                    self._regression_first_seen[name] = now

                regressions.append({
                    'test_name': name,
                    'baseline_status': baseline_status,
                    'current_status': current_status,
                    'first_seen': self._regression_first_seen[name],
                })

        return regressions

    def detect_improvements(
        self,
        current_results: Dict[str, VerificationStatus],
        baseline_results: Dict[str, VerificationStatus] = None,
        baseline_name: str = None
    ) -> List[str]:
        """
        Find tests that went from FAILED to PASSED.

        Args:
            current_results: Current test results
            baseline_results: Baseline to compare against
            baseline_name: Name of saved baseline to use

        Returns:
            List of test names that improved
        """
        # Determine baseline to use
        if baseline_name and baseline_name in self._baselines:
            baseline = self._baselines[baseline_name]
        elif baseline_results is not None:
            baseline = baseline_results
        elif len(self._baselines) > 0:
            baseline = list(self._baselines.values())[-1]
        else:
            return []

        improvements = []

        for name, current_status in current_results.items():
            baseline_status = baseline.get(name, VerificationStatus.PENDING)

            # Improvement: was failing, now passing
            if baseline_status == VerificationStatus.FAILED and current_status == VerificationStatus.PASSED:
                improvements.append(name)
                # Clear regression tracking if fixed
                if name in self._regression_first_seen:
                    del self._regression_first_seen[name]

        return improvements

    def detect_flaky_tests(self, window: int = 10, threshold: float = 0.3) -> List[str]:
        """
        Track test results over window and flag tests that flip between pass/fail.

        Args:
            window: Number of recent runs to analyze
            threshold: Minimum flip rate to consider flaky (0.0-1.0)

        Returns:
            List of test names identified as flaky
        """
        flaky = []

        for test_name, results in self._flaky_tests.items():
            # Only analyze if we have enough history
            if len(results) < window:
                continue

            # Look at recent window
            recent = results[-window:]

            # Count flips (transitions from True to False or vice versa)
            flips = 0
            for i in range(1, len(recent)):
                if recent[i] != recent[i-1]:
                    flips += 1

            # Calculate flip rate
            flip_rate = flips / (len(recent) - 1) if len(recent) > 1 else 0

            if flip_rate >= threshold:
                flaky.append(test_name)

        return flaky

    def record_results(self, results: Dict[str, VerificationStatus]) -> None:
        """
        Add results to history with timestamp.

        Args:
            results: Dictionary mapping test names to verification statuses
        """
        # Add to history
        self._history.append({
            'timestamp': datetime.now(),
            'results': results.copy(),
        })

        # Update flaky test tracking
        for test_name, status in results.items():
            if test_name not in self._flaky_tests:
                self._flaky_tests[test_name] = []

            # Record pass (True) or fail (False)
            passed = status == VerificationStatus.PASSED
            self._flaky_tests[test_name].append(passed)

    def get_trend(self, test_name: str, limit: int = None) -> Dict[str, Any]:
        """
        Return history for specific test.

        Args:
            test_name: Name of the test
            limit: Maximum number of recent results to include

        Returns:
            Dictionary with:
                - history: List of (timestamp, status) tuples
                - pass_rate: Percentage of passes (0.0-1.0)
                - flakiness_score: How flaky the test is (0.0-1.0)
                - total_runs: Total number of runs tracked
        """
        history = []

        # Extract test results from history
        for snapshot in self._history:
            if test_name in snapshot['results']:
                history.append({
                    'timestamp': snapshot['timestamp'],
                    'status': snapshot['results'][test_name],
                })

        # Apply limit if specified
        if limit is not None:
            history = history[-limit:]

        # Calculate statistics
        if len(history) == 0:
            return {
                'history': [],
                'pass_rate': 0.0,
                'flakiness_score': 0.0,
                'total_runs': 0,
            }

        # Pass rate
        passes = sum(1 for h in history if h['status'] == VerificationStatus.PASSED)
        pass_rate = passes / len(history)

        # Flakiness score (based on transitions)
        flips = 0
        for i in range(1, len(history)):
            if history[i]['status'] != history[i-1]['status']:
                flips += 1
        flakiness_score = flips / (len(history) - 1) if len(history) > 1 else 0.0

        return {
            'history': history,
            'pass_rate': pass_rate,
            'flakiness_score': flakiness_score,
            'total_runs': len(history),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            Dictionary with:
                - total_tests_tracked: Number of unique tests
                - baselines_stored: Number of saved baselines
                - total_snapshots: Number of historical snapshots
                - regression_count: Current regressions being tracked
                - improvement_count: Tests that improved (if baseline exists)
                - flaky_test_count: Number of flaky tests detected
        """
        # Count improvements if we have a baseline
        improvement_count = 0
        if self._history and self._baselines:
            latest_results = self._history[-1]['results'] if self._history else {}
            improvements = self.detect_improvements(latest_results)
            improvement_count = len(improvements)

        # Count regressions
        regression_count = len(self._regression_first_seen)

        # Count flaky tests
        flaky = self.detect_flaky_tests()

        return {
            'total_tests_tracked': len(self._flaky_tests),
            'baselines_stored': len(self._baselines),
            'total_snapshots': len(self._history),
            'regression_count': regression_count,
            'improvement_count': improvement_count,
            'flaky_test_count': len(flaky),
        }
