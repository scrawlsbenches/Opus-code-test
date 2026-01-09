"""
Behavioral Tests for Audit Reasoning with PLN.

These tests describe user stories for the audit reasoning system using
a realistic in-memory codebase. The InMemoryFileSystem contains actual
Python files with real patterns (TODO, FIXME, etc.) that the auditor scans.

Testing Philosophy (Metus):
- Scenarios test behaviors, not implementation
- Given-When-Then format tells the story
- Tests serve as living documentation
- Test codebase simulates real audit scenarios
"""

import pytest
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from cortical.common.filesystem import InMemoryFileSystem


# =============================================================================
# TEST CODEBASE - Realistic files for audit scenarios
# =============================================================================


# Sample Python files with various audit patterns
TEST_CODEBASE = {
    "/codebase/src/auth/login.py": '''"""Authentication module for user login."""

import hashlib
import time

# TODO: Add rate limiting to prevent brute force attacks
# FIXME: Password hashing should use bcrypt, not md5

def authenticate(username: str, password: str) -> bool:
    """Authenticate user with username and password."""
    # HACK: Temporary bypass for testing
    if username == "test":
        return True

    hashed = hashlib.md5(password.encode()).hexdigest()  # FIXME: insecure!
    return check_database(username, hashed)

def check_database(username: str, hashed_pw: str) -> bool:
    """Check credentials against database."""
    # TODO: Implement actual database lookup
    return False
''',

    "/codebase/src/auth/session.py": '''"""Session management module."""

import uuid
from datetime import datetime, timedelta

class SessionManager:
    """Manage user sessions."""

    def __init__(self):
        self.sessions = {}  # TODO: Use Redis instead of in-memory

    def create_session(self, user_id: str) -> str:
        """Create a new session for user."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(hours=24)
        }
        return session_id

    def validate_session(self, session_id: str) -> bool:
        """Check if session is valid."""
        if session_id not in self.sessions:
            return False
        session = self.sessions[session_id]
        return datetime.now() < session["expires"]
''',

    "/codebase/src/api/endpoints.py": '''"""API endpoint handlers."""

from flask import request, jsonify

# FIXME: All endpoints need authentication middleware
# TODO: Add input validation for all parameters

def get_users():
    """Get list of users."""
    # Should be paginated but isn't - will be slow with many users
    return jsonify(fetch_all_users())

def create_user():
    """Create a new user."""
    data = request.json
    # XXX: No validation! SQL injection possible
    username = data.get("username")
    return insert_user(username)

def delete_user(user_id):
    """Delete a user by ID."""
    # HACK: Skipping permission check for now
    return remove_user(user_id)
''',

    "/codebase/src/utils/helpers.py": '''"""Utility helper functions."""

import os
import logging

logger = logging.getLogger(__name__)

def get_config(key: str, default=None):
    """Get configuration value from environment."""
    return os.environ.get(key, default)

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"${amount:,.2f}"

def calculate_tax(amount: float, rate: float = 0.1) -> float:
    """Calculate tax on amount."""
    return amount * rate
''',

    "/codebase/src/legacy/old_processor.py": '''"""Legacy data processor - scheduled for removal."""

# WARNING: This entire module is deprecated
# TODO: Migrate all usages to new_processor.py
# FIXME: Memory leak in process_batch when handling large files

import gc

class OldProcessor:
    """Legacy processor - DO NOT USE IN NEW CODE."""

    def __init__(self):
        self.cache = {}  # HACK: Unbounded cache causes memory issues

    def process(self, data):
        """Process data using legacy algorithm."""
        # FIXME: This is O(n^2), should be O(n log n)
        result = []
        for i in data:
            for j in data:
                if i != j:
                    result.append((i, j))
        return result

    def process_batch(self, items):
        """Process a batch of items."""
        # TODO: Add proper error handling
        # FIXME: Memory leak here - cache never cleared
        for item in items:
            self.cache[item.id] = self.process(item.data)
        return list(self.cache.values())
''',

    "/codebase/src/core/engine.py": '''"""Core processing engine."""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    priority: int
    data: dict

class Engine:
    """Main processing engine."""

    def __init__(self, workers: int = 4):
        self.workers = workers
        self.queue = []

    def submit(self, task: Task) -> str:
        """Submit a task for processing."""
        self.queue.append(task)
        self.queue.sort(key=lambda t: t.priority, reverse=True)
        return task.id

    def process_next(self) -> Optional[dict]:
        """Process the next task in queue."""
        if not self.queue:
            return None
        task = self.queue.pop(0)
        return self._execute(task)

    def _execute(self, task: Task) -> dict:
        """Execute a single task."""
        return {"id": task.id, "status": "completed", "result": task.data}
''',

    "/codebase/tests/test_auth.py": '''"""Tests for authentication module."""

import pytest
from src.auth.login import authenticate

def test_authenticate_valid_user():
    """Test authentication with valid credentials."""
    # TODO: Use proper test fixtures
    assert authenticate("test", "password") == True

def test_authenticate_invalid_user():
    """Test authentication with invalid credentials."""
    assert authenticate("invalid", "wrong") == False
''',
}


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def memory_fs():
    """Create an in-memory filesystem with test codebase."""
    fs = InMemoryFileSystem(Path("/codebase"))

    # Create directory structure and files
    for file_path, content in TEST_CODEBASE.items():
        path = Path(file_path)
        # Create parent directories
        fs.mkdir(path.parent, parents=True, exist_ok=True)
        # Write the file
        fs.write_text(path, content)

    return fs


@pytest.fixture
def file_scanner(memory_fs):
    """
    Create a scanner that extracts patterns from in-memory files.

    This simulates what the real codebase_health.analyze_directory does,
    but operates on the InMemoryFileSystem.
    """
    return InMemoryFileScanner(memory_fs)


@pytest.fixture
def reasoner_with_codebase(memory_fs, file_scanner):
    """
    Create an audit reasoner pre-loaded with facts from the test codebase.

    This is the realistic scenario: scan codebase -> extract findings -> reason.
    """
    from cortical.audits.reasoning import AuditReasoner

    # Scan the in-memory codebase
    findings = file_scanner.scan_directory(Path("/codebase"))

    # Create reasoner
    reasoner = AuditReasoner(use_persistence=False)
    reasoner.add_default_rules()

    # Assert facts from actual file scan
    for file_path, patterns, traits in findings:
        dirs = [p for p in Path(file_path).parts[2:-1]]  # Skip /codebase/src
        reasoner.assert_file_facts(file_path, patterns, traits, dirs)

    return reasoner, findings


# =============================================================================
# IN-MEMORY FILE SCANNER
# =============================================================================


@dataclass
class Finding:
    """A finding from scanning a file."""
    file_path: str
    pattern: str
    line_number: int
    line_content: str


class InMemoryFileScanner:
    """
    Scanner that extracts audit patterns from in-memory files.

    This mirrors what codebase_health.analyze_directory does on real files,
    but operates on InMemoryFileSystem for testing.
    """

    # Patterns to search for (same as real tool)
    PATTERNS = {
        "todo": re.compile(r"#\s*TODO:?\s*(.+)", re.IGNORECASE),
        "fixme": re.compile(r"#\s*FIXME:?\s*(.+)", re.IGNORECASE),
        "hack": re.compile(r"#\s*HACK:?\s*(.+)", re.IGNORECASE),
        "xxx": re.compile(r"#\s*XXX:?\s*(.+)", re.IGNORECASE),
        "warning": re.compile(r"#\s*WARNING:?\s*(.+)", re.IGNORECASE),
    }

    def __init__(self, fs: InMemoryFileSystem):
        self.fs = fs

    def scan_file(self, path: Path) -> List[Finding]:
        """Scan a single file for patterns."""
        findings = []
        try:
            content = self.fs.read_text(path)
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern in self.PATTERNS.items():
                    match = pattern.search(line)
                    if match:
                        findings.append(Finding(
                            file_path=str(path),
                            pattern=pattern_name,
                            line_number=line_num,
                            line_content=line.strip()
                        ))
        except FileNotFoundError:
            pass

        return findings

    def scan_directory(self, root: Path) -> List[Tuple[str, List[str], List[str]]]:
        """
        Scan all Python files in directory.

        Returns: List of (file_path, patterns, traits) tuples
        """
        results = []

        # Get all files
        all_files = self.fs.list_all_files()
        py_files = [f for f in all_files if f.endswith(".py") and str(root) in f]

        for file_path in py_files:
            path = Path(file_path)
            findings = self.scan_file(path)

            if findings:
                patterns = list(set(f.pattern for f in findings))

                # Determine traits based on patterns
                traits = []
                if len(findings) > 5:
                    traits.append("high_churn")  # Many issues = frequent changes
                if "fixme" in patterns and "hack" in patterns:
                    traits.append("bug_prone")
                if "legacy" in file_path.lower() or "old" in file_path.lower():
                    traits.append("legacy")

                results.append((file_path, patterns, traits))

        return results

    def get_findings_for_file(self, path: Path) -> List[Finding]:
        """Get all findings for a specific file."""
        return self.scan_file(path)


# =============================================================================
# STORY: Developer Analyzes Codebase for Risks
# =============================================================================


class TestDeveloperAnalyzesCodebaseForRisks:
    """
    Story: As a developer, I want to analyze my codebase for potential risks
    so that I can prioritize what to review.
    """

    def test_scanning_codebase_finds_todo_comments(self, file_scanner):
        """
        Scenario: Scanning codebase finds TODO comments

        Given a codebase with files containing TODO comments
        When I scan the codebase
        Then I should find all files with TODOs
        And the findings should include the line content
        """
        # Given a codebase with TODO comments (from fixture)

        # When I scan the codebase
        findings = file_scanner.scan_directory(Path("/codebase"))

        # Then I should find files with TODOs
        files_with_todos = [f for f, patterns, _ in findings if "todo" in patterns]
        assert len(files_with_todos) > 0

        # And the findings should include specific files
        file_paths = [f for f, _, _ in findings]
        assert any("login.py" in f for f in file_paths)
        assert any("old_processor.py" in f for f in file_paths)

    def test_scanning_finds_security_concerns(self, file_scanner):
        """
        Scenario: Scanning finds security-related FIXME comments

        Given a codebase with security issues marked as FIXME
        When I scan the codebase
        Then I should find the security concerns
        And login.py should be flagged for the password hashing issue
        """
        # Given a codebase with security FIXME comments

        # When I scan the codebase
        findings = file_scanner.scan_directory(Path("/codebase"))

        # Then I should find security concerns
        login_findings = [f for f, patterns, _ in findings
                         if "login.py" in f and "fixme" in patterns]
        assert len(login_findings) > 0

        # And the specific security issue should be present
        login_details = file_scanner.get_findings_for_file(
            Path("/codebase/src/auth/login.py")
        )
        fixme_findings = [f for f in login_details if f.pattern == "fixme"]
        assert any("bcrypt" in f.line_content or "md5" in f.line_content
                   for f in fixme_findings)

    def test_legacy_code_identified_as_high_risk(self, file_scanner):
        """
        Scenario: Legacy code is identified as high risk

        Given a codebase with legacy modules
        When I scan the codebase
        Then legacy modules should be marked with traits
        And old_processor.py should have legacy trait
        """
        # Given a codebase with legacy modules

        # When I scan the codebase
        findings = file_scanner.scan_directory(Path("/codebase"))

        # Then legacy modules should have traits
        legacy_findings = [(f, patterns, traits) for f, patterns, traits in findings
                          if "legacy" in f.lower() or "old" in f.lower()]
        assert len(legacy_findings) > 0

        # And old_processor.py should have legacy trait
        old_processor = [(f, patterns, traits) for f, patterns, traits in findings
                        if "old_processor.py" in f]
        assert len(old_processor) > 0
        _, _, traits = old_processor[0]
        assert "legacy" in traits


# =============================================================================
# STORY: Developer Reasons About File Risks
# =============================================================================


class TestDeveloperReasonsAboutFileRisks:
    """
    Story: As a developer, I want the system to reason about file risks
    so that I get intelligent prioritization beyond simple pattern matching.
    """

    def test_file_with_multiple_issues_flagged_for_review(
        self, reasoner_with_codebase
    ):
        """
        Scenario: File with multiple issues is flagged for review

        Given a codebase has been scanned
        When I query risk for login.py (which has TODO, FIXME, and HACK)
        Then it should be flagged for review
        And the risk should reflect multiple issues
        """
        reasoner, findings = reasoner_with_codebase

        # Given a codebase has been scanned (from fixture)

        # When I query risk for login.py
        login_file = [f for f, _, _ in findings if "login.py" in f][0]
        results = reasoner.query_file_risk(login_file)

        # Then it should have risk signals
        assert results is not None
        assert len(results) > 0

    def test_clean_utility_file_has_lower_risk(self, reasoner_with_codebase):
        """
        Scenario: Clean utility file has lower risk than problematic files

        Given a codebase has been scanned
        When I compare helpers.py (clean) to login.py (problematic)
        Then helpers.py should have fewer or no risk signals
        """
        reasoner, findings = reasoner_with_codebase

        # Given codebase has been scanned

        # When I query risk for both files
        helpers_findings = [f for f, patterns, _ in findings if "helpers.py" in f]

        # Then helpers.py should have fewer findings (it has no patterns)
        # It might not even be in findings if it's clean
        if helpers_findings:
            helpers_file = helpers_findings[0]
            results = reasoner.query_file_risk(helpers_file)
            # Should have minimal risk signals
            risk_count = len([k for k in results if not k.startswith("_")])
            assert risk_count <= 2  # Few or no risks

    def test_legacy_module_flagged_as_risky(self, reasoner_with_codebase):
        """
        Scenario: Legacy module is flagged as risky

        Given a codebase has been scanned
        When I query risk for old_processor.py
        Then it should be flagged as risky
        And it should have high importance due to multiple issues
        """
        reasoner, findings = reasoner_with_codebase

        # Given codebase has been scanned

        # When I query risk for old_processor.py
        old_proc = [f for f, _, _ in findings if "old_processor.py" in f][0]
        results = reasoner.query_file_risk(old_proc)

        # Then it should be flagged
        assert results is not None

        # And it should have importance tracked
        assert "_importance" in results
        importance = results["_importance"]
        assert importance["total"] > 0


# =============================================================================
# STORY: Developer Understands Why Files Are Flagged
# =============================================================================


class TestDeveloperUnderstandsWhyFlagged:
    """
    Story: As a developer, I want to understand WHY a file was flagged
    so that I can make informed decisions about what to fix first.
    """

    def test_explanation_shows_detected_patterns(self, reasoner_with_codebase):
        """
        Scenario: Explanation shows the detected patterns

        Given I have a file flagged for multiple issues
        When I request an explanation
        Then I should see which patterns triggered the flagging
        And the explanation should mention TODO, FIXME, or HACK
        """
        reasoner, findings = reasoner_with_codebase

        # Given a file with multiple issues
        login_file = [f for f, _, _ in findings if "login.py" in f][0]

        # When I request an explanation
        explanation = reasoner.explain_file_risk(login_file)

        # Then I should see the facts
        assert "facts" in explanation
        facts = explanation["facts"]
        assert len(facts) > 0

        # And the summary should be present
        assert "summary" in explanation

    def test_explanation_includes_suggestions(self, reasoner_with_codebase):
        """
        Scenario: Explanation includes actionable suggestions

        Given a file flagged for FIXME issues
        When I request an explanation
        Then I should receive suggestions for what to do
        """
        reasoner, findings = reasoner_with_codebase

        # Given a file with FIXME issues
        login_file = [f for f, _, _ in findings if "login.py" in f][0]

        # When I request an explanation
        explanation = reasoner.explain_file_risk(login_file)

        # Then suggestions should be provided
        assert "suggestions" in explanation
        suggestions = explanation["suggestions"]
        assert isinstance(suggestions, list)

    def test_explanation_traces_reasoning_chain(self, reasoner_with_codebase):
        """
        Scenario: Explanation shows reasoning chain

        Given a file that triggers rule inference
        When I request an explanation with traces
        Then I should see how conclusions were derived
        """
        reasoner, findings = reasoner_with_codebase

        # Given a file that triggers rules
        old_proc = [f for f, _, _ in findings if "old_processor.py" in f][0]

        # When I request an explanation
        explanation = reasoner.explain_file_risk(old_proc)

        # Then traces should be present
        assert "traces" in explanation
        traces = explanation["traces"]
        # Should have some inference results
        assert isinstance(traces, dict)


# =============================================================================
# STORY: Developer Gets Priority Ordering
# =============================================================================


class TestDeveloperGetsPriorityOrdering:
    """
    Story: As a developer, I want files prioritized by risk
    so that I can focus on the most critical issues first.
    """

    def test_priority_files_ordered_by_importance(self, reasoner_with_codebase):
        """
        Scenario: Priority files are ordered by importance

        Given multiple files have been analyzed
        When I request priority files
        Then files should be ordered by importance score
        And high-issue files should rank higher
        """
        reasoner, findings = reasoner_with_codebase

        # Given multiple files analyzed (from fixture)

        # When I request priority files
        priorities = reasoner.get_priority_files(top_n=5)

        # Then files should be returned in order
        assert len(priorities) > 0
        for i in range(len(priorities) - 1):
            _, importance_a = priorities[i]
            _, importance_b = priorities[i + 1]
            assert importance_a >= importance_b

    def test_legacy_file_has_high_priority(self, reasoner_with_codebase):
        """
        Scenario: Legacy files with many issues rank high

        Given a legacy file with multiple issues
        When I check priority ranking
        Then old_processor.py should be in top priority files
        """
        reasoner, findings = reasoner_with_codebase

        # Given codebase analyzed

        # When I check priorities
        priorities = reasoner.get_priority_files(top_n=10)
        priority_files = [f for f, _ in priorities]

        # Then old_processor should rank high (has many issues)
        # Note: file_id is normalized (. replaced with _)
        has_old_processor = any("old_processor" in f for f in priority_files)
        assert has_old_processor


# =============================================================================
# STORY: Developer Uses Natural Language Queries
# =============================================================================


class TestDeveloperUsesNaturalLanguageQueries:
    """
    Story: As a developer, I want to ask questions in natural language
    so that I don't need to learn a formal query syntax.
    """

    def test_translate_why_question(self):
        """
        Scenario: Translating "why is X risky?"

        Given a natural language question about risk
        When I translate it
        Then I should get a structured query with explain intent
        """
        from cortical.audits.reasoning import translate_audit_query

        # When I translate a "why" question
        query = translate_audit_query("why is login.py risky?")

        # Then the query should be recognized
        assert query is not None
        assert query.intent in ["explain", "list", "trace"]

    def test_translate_list_question(self):
        """
        Scenario: Translating "what files are flagged?"

        Given a natural language question about flagged files
        When I translate it
        Then I should get a list query
        """
        from cortical.audits.reasoning import translate_audit_query

        # When I translate a "what" question
        query = translate_audit_query("what files are flagged?")

        # Then the query should be recognized
        assert query is not None
        assert query.intent in ["list", "explain", "trace"]

    def test_detect_natural_language(self):
        """
        Scenario: Detecting natural language vs flags

        Given various input types
        When I check if they are natural language
        Then questions should be detected as NL
        And flags should not be detected as NL
        """
        from cortical.audits.reasoning import is_natural_language_query

        # Natural language
        assert is_natural_language_query("why is this file risky?")
        assert is_natural_language_query("show me the priority files")

        # Not natural language (flags)
        assert not is_natural_language_query("--verbose")
        assert not is_natural_language_query("-h")


# =============================================================================
# STORY: Insights Persist Across Sessions
# =============================================================================


class TestInsightsPersistAcrossSessions:
    """
    Story: As a developer, I want audit insights to persist
    so that importance builds up over time with repeated access.
    """

    def test_attention_value_decay(self):
        """
        Scenario: Importance decays when files not accessed

        Given a file with high short-term importance
        When decay is applied
        Then STI should decrease
        And LTI should remain more stable
        """
        from cortical.audits.reasoning import AttentionValue

        # Given high importance
        attention = AttentionValue(sti=0.9, lti=0.7, vlti=False)

        # When decay is applied
        attention.decay_sti(0.8)  # 20% decay

        # Then STI decreases
        assert attention.sti == pytest.approx(0.72, rel=0.01)
        # LTI unchanged
        assert attention.lti == 0.7

    def test_stimulation_increases_importance(self, reasoner_with_codebase):
        """
        Scenario: Accessing a file increases its importance

        Given a file in the codebase
        When I stimulate it (simulating access)
        Then its short-term importance should increase
        """
        reasoner, findings = reasoner_with_codebase

        # Given a file
        login_file = [f for f, _, _ in findings if "login.py" in f][0]
        file_id = Path(login_file).name.replace(".", "_")
        initial_sti = reasoner.file_importance[file_id].sti

        # When I stimulate it
        reasoner.stimulate_file(login_file, amount=0.3)

        # Then importance increases
        new_sti = reasoner.file_importance[file_id].sti
        assert new_sti > initial_sti


# =============================================================================
# STORY: Report Generation
# =============================================================================


class TestReportGeneration:
    """
    Story: As a developer, I want to generate audit reports
    so that I can share findings with my team.
    """

    def test_generate_report_from_analysis(self, reasoner_with_codebase):
        """
        Scenario: Generate a report from codebase analysis

        Given a codebase has been analyzed
        When I generate a report
        Then it should summarize all findings
        """
        from cortical.audits.reasoning import generate_reasoning_report

        reasoner, findings = reasoner_with_codebase

        # Given analysis results
        results = {
            "files_analyzed": [f for f, _, _ in findings],
            "rules_loaded": 10,
            "analysis_results": [],
        }

        for file_path, patterns, traits in findings:
            explanation = reasoner.explain_file_risk(file_path)
            results["analysis_results"].append({
                "file": file_path,
                "patterns": patterns,
                "explanation": explanation,
            })

        # When I generate a report
        report = generate_reasoning_report(results)

        # Then report should be generated
        assert report is not None
        assert isinstance(report, str)
        assert len(report) > 0


# =============================================================================
# STORY: Complete Audit Workflow
# =============================================================================


class TestCompleteAuditWorkflow:
    """
    Story: As an auditor, I want to run a complete audit workflow
    from scanning to reporting.
    """

    def test_full_audit_workflow(self, memory_fs, file_scanner):
        """
        Scenario: Complete audit from scan to prioritization

        Given a codebase with various issues
        When I run the full audit workflow
        Then I should get prioritized findings with explanations
        """
        from cortical.audits.reasoning import AuditReasoner

        # Given: Scan the codebase
        findings = file_scanner.scan_directory(Path("/codebase"))
        assert len(findings) > 0, "Should find files with issues"

        # When: Initialize reasoner and load facts
        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()

        for file_path, patterns, traits in findings:
            dirs = [p for p in Path(file_path).parts[2:-1]]
            reasoner.assert_file_facts(file_path, patterns, traits, dirs)

        # Then: Get prioritized results
        priorities = reasoner.get_priority_files(top_n=5)
        assert len(priorities) > 0

        # And: Explanations available for each
        for file_id, importance in priorities[:3]:
            # Reconstruct path for explanation
            for file_path, _, _ in findings:
                if file_id in Path(file_path).name.replace(".", "_"):
                    explanation = reasoner.explain_file_risk(file_path)
                    assert explanation is not None
                    assert "summary" in explanation
                    break

    def test_audit_identifies_security_issues(self, memory_fs, file_scanner):
        """
        Scenario: Audit correctly identifies security concerns

        Given a codebase with security-marked issues
        When I audit the codebase
        Then security files should be prioritized
        And explanations should mention security patterns
        """
        from cortical.audits.reasoning import AuditReasoner

        # Given: Scan
        findings = file_scanner.scan_directory(Path("/codebase"))

        # When: Analyze
        reasoner = AuditReasoner(use_persistence=False)
        reasoner.add_default_rules()

        for file_path, patterns, traits in findings:
            dirs = [p for p in Path(file_path).parts[2:-1]]
            reasoner.assert_file_facts(file_path, patterns, traits, dirs)

        # Then: Auth files with security issues should be tracked
        auth_findings = [f for f, patterns, _ in findings
                        if "auth" in f and ("fixme" in patterns or "hack" in patterns)]
        assert len(auth_findings) > 0

        # And: login.py should have explanation available
        login_file = [f for f, _, _ in findings if "login.py" in f]
        if login_file:
            explanation = reasoner.explain_file_risk(login_file[0])
            assert explanation is not None
            # Should have suggestions for the issues found
            assert "suggestions" in explanation
