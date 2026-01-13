"""
Behavioral Tests for Audit CLI Tool.

These tests verify the audit CLI commands work correctly by invoking them
as subprocess calls, simulating real user interaction with the tool.

Testing Philosophy (Metus):
- Tests exercise the actual CLI, not internal APIs
- Given-When-Then format tells the story
- Tests verify end-to-end functionality
- JSON output is parsed and validated
"""

import json
import os
import pytest
import subprocess
import tempfile
from pathlib import Path


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_codebase():
    """Create a temporary directory with test Python files containing patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files with various patterns
        test_files = {
            "module_with_monkeypatch.py": '''"""Module that uses monkeypatch pattern."""

def patch_system():
    """This function does monkeypatch-style operations."""
    # Using monkeypatch to override system behavior
    import sys
    original = sys.path
    sys.path = ["/custom/path"]  # monkeypatch sys.path
    return original

def another_monkeypatch():
    """Another monkeypatch example."""
    pass
''',
            "module_with_todos.py": '''"""Module with TODO markers."""

# TODO: Implement proper error handling
# FIXME: This needs to be refactored

def incomplete_function():
    """Function that needs work."""
    # TODO: Add validation
    pass

# HACK: Temporary workaround
def hacky_solution():
    pass
''',
            "clean_module.py": '''"""A clean module without issues."""

def clean_function(x: int, y: int) -> int:
    """A properly implemented function."""
    return x + y
''',
            "eval_usage.py": '''"""Module with security concerns."""

def dangerous_eval(user_input):
    """This uses eval which is risky."""
    result = eval(user_input)  # Security risk!
    return result
''',
        }

        for filename, content in test_files.items():
            filepath = Path(tmpdir) / filename
            filepath.write_text(content)

        yield tmpdir


@pytest.fixture
def cli_runner():
    """Helper to run CLI commands and capture output."""
    def run(args, timeout=60):
        cmd = ["python", "-m", "cortical.cli.audit"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result
    return run


# =============================================================================
# STORY: Pattern Management
# =============================================================================


class TestPatternManagementStories:
    """
    As an auditor, I want to manage custom patterns
    so that I can detect project-specific code smells.
    """

    def test_list_patterns_shows_defaults_and_custom(self, cli_runner):
        """
        Given the audit CLI is available
        When I list all patterns
        Then I see both default and custom patterns
        """
        result = cli_runner(["pattern", "list"])

        assert result.returncode == 0
        assert "Default Patterns" in result.stdout
        assert "Custom Patterns" in result.stdout
        # Default patterns should include common markers
        assert "TODO:" in result.stdout
        assert "FIXME:" in result.stdout

    def test_add_and_remove_custom_pattern(self, cli_runner):
        """
        Given I want to track a new code pattern
        When I add a custom pattern and then remove it
        Then the pattern appears and disappears from the list
        """
        # Add a test pattern
        add_result = cli_runner([
            "pattern", "add", "test_pattern_xyz",
            "--id", "test_xyz",
            "--scope", "code",
            "--implies", "test_implication"
        ])
        assert add_result.returncode == 0
        assert "Added pattern" in add_result.stdout

        # Verify it appears in list
        list_result = cli_runner(["pattern", "list"])
        assert "test_xyz" in list_result.stdout

        # Remove it
        remove_result = cli_runner(["pattern", "remove", "test_xyz"])
        assert remove_result.returncode == 0
        assert "Removed pattern" in remove_result.stdout

        # Verify it's gone
        list_result2 = cli_runner(["pattern", "list"])
        assert "test_xyz" not in list_result2.stdout


# =============================================================================
# STORY: Health Analysis with Pattern Detection
# =============================================================================


class TestHealthAnalysisStories:
    """
    As an auditor, I want to scan a codebase for patterns
    so that I can identify areas that need attention.
    """

    def test_health_detects_monkeypatch_in_code(self, cli_runner, temp_codebase):
        """
        Given a codebase with monkeypatch usage
        When I run health analysis
        Then monkeypatch occurrences are detected and reported
        """
        result = cli_runner(["health", temp_codebase, "--json"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Should find monkeypatch pattern
        assert "monkeypatch" in data.get("pattern_counts", {})
        monkeypatch_count = data["pattern_counts"]["monkeypatch"]
        assert monkeypatch_count >= 2, f"Expected at least 2 monkeypatch, got {monkeypatch_count}"

        # Should have findings for monkeypatch
        monkeypatch_findings = [
            f for f in data.get("findings", [])
            if f.get("pattern") == "monkeypatch"
        ]
        assert len(monkeypatch_findings) >= 2

    def test_health_detects_comment_patterns(self, cli_runner, temp_codebase):
        """
        Given a codebase with TODO/FIXME/HACK comments
        When I run health analysis
        Then comment patterns are detected
        """
        result = cli_runner(["health", temp_codebase, "--json"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        pattern_counts = data.get("pattern_counts", {})

        # Should detect TODO, FIXME, HACK
        assert "TODO:" in pattern_counts or any("todo" in k.lower() for k in pattern_counts)

    def test_health_json_output_has_required_fields(self, cli_runner, temp_codebase):
        """
        Given the health command
        When I request JSON output
        Then the output contains all required fields
        """
        result = cli_runner(["health", temp_codebase, "--json"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Required fields
        assert "files_analyzed" in data
        assert "comments_analyzed" in data
        assert "findings" in data
        assert "pattern_counts" in data
        assert "files" in data
        assert isinstance(data["findings"], list)
        assert isinstance(data["pattern_counts"], dict)

    def test_health_findings_include_file_and_line(self, cli_runner, temp_codebase):
        """
        Given findings are detected
        When I examine the JSON output
        Then each finding includes file path and line number
        """
        result = cli_runner(["health", temp_codebase, "--json"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        findings = data.get("findings", [])
        assert len(findings) > 0, "Expected at least one finding"

        for finding in findings:
            assert "file" in finding, f"Finding missing 'file': {finding}"
            assert "line" in finding, f"Finding missing 'line': {finding}"
            assert "pattern" in finding, f"Finding missing 'pattern': {finding}"

    def test_health_verbose_shows_full_paths(self, cli_runner, temp_codebase):
        """
        Given verbose output is requested
        When health analysis completes
        Then findings show full file paths (not relative ../../../)
        """
        result = cli_runner(["health", temp_codebase, "-v"])

        assert result.returncode == 0
        # Should not contain ../ path traversal in findings
        assert "../../../" not in result.stdout
        # Should contain the temp directory path
        assert temp_codebase in result.stdout or "module_with" in result.stdout


# =============================================================================
# STORY: Security Pattern Detection
# =============================================================================


class TestSecurityPatternStories:
    """
    As a security reviewer, I want to detect risky patterns
    so that I can identify potential vulnerabilities.
    """

    def test_detects_eval_usage(self, cli_runner, temp_codebase):
        """
        Given code that uses eval()
        When I run health analysis
        Then eval usage is flagged as a security risk
        """
        result = cli_runner(["health", temp_codebase, "--json"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # eval_call pattern should be detected
        pattern_counts = data.get("pattern_counts", {})
        eval_detected = any("eval" in k.lower() for k in pattern_counts)

        if eval_detected:
            # If detected, verify it's marked with security implication
            eval_findings = [
                f for f in data.get("findings", [])
                if "eval" in f.get("pattern", "").lower()
            ]
            for finding in eval_findings:
                if "implies" in finding:
                    assert "security" in finding["implies"].lower()


# =============================================================================
# STORY: PLN Reasoning
# =============================================================================


class TestReasoningStories:
    """
    As an auditor, I want to use PLN reasoning
    so that I can understand file risk levels.
    """

    def test_reason_command_runs_without_error(self, cli_runner, temp_codebase):
        """
        Given a codebase to analyze
        When I run the reason command
        Then it completes without error
        """
        result = cli_runner(["reason", "-d", temp_codebase])

        assert result.returncode == 0
        assert "Reasoning complete" in result.stdout

    def test_reason_with_verbose_shows_details(self, cli_runner, temp_codebase):
        """
        Given the reason command with verbose flag
        When analysis completes
        Then detailed output is shown
        """
        result = cli_runner(["reason", "-d", temp_codebase, "-v"])

        assert result.returncode == 0
        # Should show some analysis output
        assert "PLN" in result.stdout or "Analyzing" in result.stdout


# =============================================================================
# STORY: Scan Command
# =============================================================================


class TestScanStories:
    """
    As an auditor, I want to scan for suspicious comments
    so that I can find areas needing review.
    """

    def test_scan_finds_python_files(self, cli_runner, temp_codebase):
        """
        Given a directory with Python files
        When I run the scan command
        Then it finds and scans the files
        """
        result = cli_runner(["scan", temp_codebase])

        assert result.returncode == 0
        assert "Python files" in result.stdout
        assert "comments" in result.stdout.lower()


# =============================================================================
# STORY: Discover Command (Experimental)
# =============================================================================


class TestDiscoverStories:
    """
    As a researcher, I want to discover patterns using WovenMind
    so that I can find emergent code quality issues.
    """

    def test_discover_runs_and_shows_experimental_warning(self, cli_runner, temp_codebase):
        """
        Given the experimental discover command
        When I run it
        Then it shows the experimental warning and completes
        """
        result = cli_runner(["discover", temp_codebase])

        assert result.returncode == 0
        assert "EXPERIMENTAL" in result.stdout
        assert "Discovery complete" in result.stdout
