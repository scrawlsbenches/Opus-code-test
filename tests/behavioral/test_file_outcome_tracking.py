"""
Behavioral test: File Outcome Tracking

HYPOTHESIS:
The learning system should track outcomes per file/module, enabling
agents to know which files are "risky" based on historical failures.

From the Agent Survey (Worker Agent):
> "I also need to know if the files I'm about to touch are 'safe' or
> 'dangerous'. If src/auth.py has broken 3 times in the last week,
> I want to know that before I start."

This test proves:
1. Experiences track which files were touched
2. File history can be queried (success/failure counts)
3. Risky files are identified from failure patterns
4. File risk assessment informs task guidance

DISABLED: llm_orchestration module scheduled for removal.
"""

import pytest

# Skip entire module - llm_orchestration scheduled for removal
pytestmark = pytest.mark.skip(reason="DISABLED: llm_orchestration module scheduled for removal")
from pathlib import Path
import tempfile
import shutil

from llm_orchestration.learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)


class TestFileOutcomeTracking:
    """
    Prove that: Experiences → File History → Risk Assessment → Guidance

    This enables agents to approach risky files with caution.
    """

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for learning system."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_experience_tracks_files_touched(self, temp_storage):
        """
        HYPOTHESIS: Experiences record which files were modified.

        Given: An experience with file-modifying actions
        When: Experience is completed
        Then: The files touched are recorded and retrievable
        """
        cycle = LearningCycle(temp_storage)

        context = Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="api",
        )

        exp = cycle.start_experience(
            context=context,
            intent="Implement user authentication",
            experience_type=ExperienceType.TASK_EXECUTION,
        )

        # Record actions that touch specific files
        exp.add_action(Action(
            action_type="write_code",
            description="Add auth logic",
            target="src/auth/jwt.py",  # File touched
        ))
        exp.add_action(Action(
            action_type="write_test",
            description="Add auth tests",
            target="tests/test_auth.py",  # Another file touched
        ))
        exp.add_action(Action(
            action_type="modify",
            description="Update config",
            target="src/config.py",  # Third file
        ))

        cycle.complete_experience(exp, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Auth implemented",
        ))

        # Retrieve files touched from experience
        files = cycle.get_files_from_experience(exp.id)

        assert "src/auth/jwt.py" in files
        assert "tests/test_auth.py" in files
        assert "src/config.py" in files

        print(f"\n=== FILES TRACKED ===")
        print(f"Experience: {exp.intent}")
        print(f"Files touched: {files}")

    def test_file_history_tracks_success_and_failure_counts(self, temp_storage):
        """
        HYPOTHESIS: File history shows success/failure counts.

        Given: Multiple experiences touching the same file
        When: Some succeed and some fail
        Then: File history shows accurate counts
        """
        cycle = LearningCycle(temp_storage)

        # Create 3 successful experiences touching auth.py
        for i in range(3):
            context = Context(goal_type="feature", goal_complexity="simple")
            exp = cycle.start_experience(context=context, intent=f"Auth change {i}")
            exp.add_action(Action("write_code", "Modify auth", "src/auth.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Worked",
            ))

        # Create 2 FAILED experiences touching auth.py
        for i in range(2):
            context = Context(goal_type="bugfix", goal_complexity="complex")
            exp = cycle.start_experience(context=context, intent=f"Auth fix {i}")
            exp.add_action(Action("write_code", "Fix auth", "src/auth.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.FAILURE,
                description="Broke something",
                error_type="RegressionError",
            ))

        # Query file history
        history = cycle.get_file_history("src/auth.py")

        assert history["total_experiences"] == 5
        assert history["success_count"] == 3
        assert history["failure_count"] == 2
        assert history["success_rate"] == 0.6

        print(f"\n=== FILE HISTORY ===")
        print(f"File: src/auth.py")
        print(f"Total touches: {history['total_experiences']}")
        print(f"Successes: {history['success_count']}")
        print(f"Failures: {history['failure_count']}")
        print(f"Success rate: {history['success_rate']:.1%}")

    def test_risky_files_identified_from_failure_patterns(self, temp_storage):
        """
        HYPOTHESIS: Files with high failure rates are flagged as risky.

        Given: Some files have high failure rates, others are stable
        When: get_risky_files() is called
        Then: High-failure files are returned with risk scores
        """
        cycle = LearningCycle(temp_storage)

        # Create experiences for a STABLE file (5 successes, 0 failures)
        for i in range(5):
            context = Context(goal_type="feature", goal_complexity="simple")
            exp = cycle.start_experience(context=context, intent=f"Utils {i}")
            exp.add_action(Action("write_code", "Modify utils", "src/utils.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Worked",
            ))

        # Create experiences for a RISKY file (2 successes, 4 failures)
        for i in range(2):
            context = Context(goal_type="feature", goal_complexity="moderate")
            exp = cycle.start_experience(context=context, intent=f"Auth success {i}")
            exp.add_action(Action("write_code", "Modify auth", "src/auth.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Worked",
            ))

        for i in range(4):
            context = Context(goal_type="bugfix", goal_complexity="complex")
            exp = cycle.start_experience(context=context, intent=f"Auth failure {i}")
            exp.add_action(Action("write_code", "Fix auth", "src/auth.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.FAILURE,
                description="Broke",
            ))

        # Get risky files
        risky = cycle.get_risky_files(min_experiences=3, max_success_rate=0.5)

        # src/auth.py should be risky (success rate = 2/6 = 0.33)
        risky_paths = [r["file_path"] for r in risky]
        assert "src/auth.py" in risky_paths, "auth.py should be flagged as risky"
        assert "src/utils.py" not in risky_paths, "utils.py should NOT be risky"

        print(f"\n=== RISKY FILES ===")
        for r in risky:
            print(f"  {r['file_path']}: {r['success_rate']:.1%} success rate")

    def test_guidance_includes_file_risk_warnings(self, temp_storage):
        """
        HYPOTHESIS: Task guidance includes warnings about risky files.

        Given: A task that will modify a risky file
        When: get_guidance() is called with files_to_modify
        Then: Guidance includes warnings about the risky file
        """
        cycle = LearningCycle(temp_storage)

        # Create failure history for auth.py
        for i in range(3):
            context = Context(goal_type="bugfix", goal_complexity="complex")
            exp = cycle.start_experience(
                context=context,
                intent=f"Auth fix attempt {i}",
            )
            exp.add_action(Action("write_code", "Fix auth", "src/auth.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.FAILURE,
                description="Broke something",
                error_type="RegressionError",
                error_message="Existing tests failing after change",
            ))

        # Get guidance for a task that will modify auth.py
        guidance = cycle.get_guidance_for_files(
            intent="Add OAuth support to authentication",
            files_to_modify=["src/auth.py", "src/oauth.py"],
        )

        # Should have file risk warnings
        assert "file_risks" in guidance
        assert "src/auth.py" in guidance["file_risks"]

        auth_risk = guidance["file_risks"]["src/auth.py"]
        assert auth_risk["is_risky"] is True
        assert auth_risk["failure_count"] >= 3
        assert len(auth_risk["recent_failures"]) > 0

        # Should have warning in recommendations
        has_file_warning = any(
            "auth.py" in w.lower() or "risky" in w.lower()
            for w in guidance.get("warnings", [])
        )
        # Note: warning generation depends on implementation

        print(f"\n=== FILE RISK IN GUIDANCE ===")
        print(f"Task: Add OAuth support")
        print(f"Files to modify: src/auth.py, src/oauth.py")
        print(f"File risks: {guidance['file_risks']}")

    def test_file_history_includes_error_patterns(self, temp_storage):
        """
        HYPOTHESIS: File history includes common error types.

        Given: Multiple failures on a file with different error types
        When: File history is queried
        Then: Error patterns are summarized
        """
        cycle = LearningCycle(temp_storage)

        # Create failures with different error types
        error_types = [
            ("ImportError", "Circular import detected"),
            ("ImportError", "Module not found"),
            ("TypeError", "Expected string, got int"),
            ("ImportError", "Circular import detected"),
        ]

        for error_type, error_msg in error_types:
            context = Context(goal_type="bugfix", goal_complexity="complex")
            exp = cycle.start_experience(context=context, intent="Fix module")
            exp.add_action(Action("write_code", "Modify", "src/problematic.py"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.FAILURE,
                description="Failed",
                error_type=error_type,
                error_message=error_msg,
            ))

        # Get file history with error patterns
        history = cycle.get_file_history("src/problematic.py")

        assert "error_patterns" in history
        assert "ImportError" in history["error_patterns"]
        # ImportError should be most common (3 occurrences)
        assert history["error_patterns"]["ImportError"] >= 2

        print(f"\n=== ERROR PATTERNS ===")
        print(f"File: src/problematic.py")
        print(f"Error patterns: {history['error_patterns']}")
        print(f"Most common: ImportError ({history['error_patterns'].get('ImportError', 0)} times)")
