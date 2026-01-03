"""
Unit tests for GoT Learning Integration.

Tests the bridge between GoT tasks and LearningCycle, verifying:
- Experience capture from task completions
- Failure capture and reflection
- Guidance retrieval for task planning
"""

import pytest
import tempfile
from pathlib import Path

from cortical.got.learning_integration import GoTLearningBridge
from llm_orchestration.learning import OutcomeType, ExperienceType


@pytest.fixture
def temp_got_dir():
    """Create a temporary GoT directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def bridge(temp_got_dir):
    """Create a GoTLearningBridge instance for testing."""
    return GoTLearningBridge(temp_got_dir)


class TestExperienceCapture:
    """Test capturing task completions as experiences."""

    def test_capture_basic_completion(self, bridge):
        """Test capturing a simple task completion."""
        experience = bridge.capture_task_completion(
            task_id="T-test-001",
            task_title="Fix bug in authentication",
            task_category="bugfix",
            task_priority="high",
            retrospective="Used debugger to isolate issue. Fixed in 30 minutes.",
            files_changed=["cortical/auth.py", "tests/test_auth.py"]
        )

        # Verify experience was created
        assert experience is not None
        assert experience.id.startswith("exp_")
        assert experience.experience_type == ExperienceType.TASK_EXECUTION

        # Verify context mapping
        assert experience.context.goal_type == "debugging"
        assert experience.context.goal_complexity == "complex"

        # Verify actions were captured
        assert len(experience.actions) == 2
        assert any("auth.py" in a.target for a in experience.actions)

        # Verify outcome
        assert experience.outcome is not None
        assert experience.outcome.outcome_type == OutcomeType.SUCCESS
        assert experience.outcome.quality_score == 1.0

        # Verify tags
        assert "task:T-test-001" in experience.tags
        assert "category:bugfix" in experience.tags
        assert "priority:high" in experience.tags

    def test_capture_with_approach(self, bridge):
        """Test capturing completion with explicit approach/strategy."""
        experience = bridge.capture_task_completion(
            task_id="T-test-002",
            task_title="Implement user registration",
            task_category="feature",
            approach="test-first",
            retrospective="TDD approach worked well. All tests passed.",
            files_changed=["api/users.py", "tests/test_users.py"]
        )

        assert experience.strategy_used == "test-first"
        assert "approach:test-first" in experience.tags

    def test_capture_with_duration(self, bridge):
        """Test efficiency score calculation from duration."""
        # Short task (30 minutes)
        exp1 = bridge.capture_task_completion(
            task_id="T-test-003",
            task_title="Quick fix",
            duration_seconds=1800,  # 30 minutes
        )
        assert exp1.outcome.efficiency_score == 1.0

        # Long task (10 hours)
        exp2 = bridge.capture_task_completion(
            task_id="T-test-004",
            task_title="Complex refactor",
            duration_seconds=36000,  # 10 hours
        )
        assert exp2.outcome.efficiency_score == 0.4

    def test_retrospective_parsing(self, bridge):
        """Test structured parsing of retrospective text."""
        experience = bridge.capture_task_completion(
            task_id="T-test-005",
            task_title="Refactor database layer",
            retrospective=(
                "TDD approach worked well for this refactor. "
                "Had some issues with migration scripts. "
                "Next time, would validate migrations in staging first."
            )
        )

        # Check that retrospective was parsed into reflection
        assert len(experience.what_worked) > 0
        assert len(experience.what_didnt_work) > 0
        assert len(experience.would_do_differently) > 0


class TestFailureCapture:
    """Test capturing task failures."""

    def test_capture_basic_failure(self, bridge):
        """Test capturing a failed task attempt."""
        experience = bridge.capture_task_failure(
            task_id="T-test-006",
            task_title="Integrate new payment API",
            task_category="feature",
            task_priority="high",
            error_message="API authentication failed - missing credentials",
            attempted_approach="direct-integration",
            files_attempted=["payment/gateway.py"],
            blockers=["Need API credentials", "Documentation unclear"]
        )

        # Verify failure outcome
        assert experience.outcome is not None
        assert experience.outcome.outcome_type == OutcomeType.FAILURE
        assert "authentication failed" in experience.outcome.error_message
        assert experience.outcome.quality_score == 0.0

        # Verify reflection captures failure details
        assert len(experience.what_didnt_work) > 0
        assert any("direct-integration" in item for item in experience.what_didnt_work)

        # Verify tags
        assert "failure" in experience.tags
        assert "failed_approach:direct-integration" in experience.tags

        # Verify context has failure marker
        assert experience.context.prior_failures == 1
        assert len(experience.context.constraints) > 0

    def test_failure_without_approach(self, bridge):
        """Test failure capture when approach is unknown."""
        experience = bridge.capture_task_failure(
            task_id="T-test-007",
            task_title="Mystery bug",
            error_message="Segmentation fault",
        )

        assert experience.outcome.outcome_type == OutcomeType.FAILURE
        # Should handle None approach gracefully
        assert experience.strategy_used is None


class TestGuidanceRetrieval:
    """Test retrieving lessons and experiences for task planning."""

    def test_get_guidance_for_similar_task(self, bridge):
        """Test that guidance retrieves relevant past experiences."""
        # First, capture some experiences
        bridge.capture_task_completion(
            task_id="T-past-001",
            task_title="Add authentication to API",
            task_category="feature",
            task_priority="high",
            approach="test-first",
            retrospective="TDD worked great for auth implementation.",
            files_changed=["api/auth.py", "tests/test_auth.py"]
        )

        bridge.capture_task_completion(
            task_id="T-past-002",
            task_title="Fix auth token expiry",
            task_category="bugfix",
            approach="test-first",
            retrospective="Tests helped isolate the issue quickly.",
            files_changed=["api/auth.py"]
        )

        # Now request guidance for a new auth-related task
        guidance = bridge.get_guidance_for_task(
            task_title="Implement OAuth2 authentication",
            task_category="feature",
            task_priority="high",
            files_to_modify=["api/oauth.py"]
        )

        # Should have retrieved guidance
        assert 'lessons' in guidance
        assert 'recommendations' in guidance
        assert 'warnings' in guidance
        assert 'relevant_successes' in guidance
        assert 'relevant_failures' in guidance

    def test_guidance_includes_context(self, bridge):
        """Test that guidance context is properly built."""
        guidance = bridge.get_guidance_for_task(
            task_title="Refactor payment processing",
            task_category="refactor",
            task_priority="medium",
            files_to_modify=["payment/processor.py", "payment/models.py"]
        )

        # Guidance should be returned even if empty
        assert isinstance(guidance, dict)
        assert 'lessons' in guidance

    def test_empty_guidance_for_new_domain(self, bridge):
        """Test guidance when no relevant experiences exist."""
        guidance = bridge.get_guidance_for_task(
            task_title="Implement quantum encryption",
            task_category="feature",
            task_priority="critical"
        )

        # Should return empty but valid guidance
        assert len(guidance['relevant_successes']) == 0
        assert len(guidance['relevant_failures']) == 0


class TestExperienceLinking:
    """Test linking tasks to related experiences."""

    def test_link_by_category(self, bridge):
        """Test finding experiences by task category."""
        # Capture some bugfix experiences
        bridge.capture_task_completion(
            task_id="T-bug-001",
            task_title="Fix memory leak",
            task_category="bugfix",
            files_changed=["core/memory.py"]
        )

        bridge.capture_task_completion(
            task_id="T-bug-002",
            task_title="Fix race condition",
            task_category="bugfix",
            files_changed=["core/threading.py"]
        )

        # Link a new bugfix task
        related = bridge.link_task_to_experiences(
            task_id="T-bug-003",
            task_category="bugfix",
            task_title="Fix null pointer"
        )

        # Should find both previous bugfix experiences
        assert len(related) >= 2

    def test_link_with_no_matches(self, bridge):
        """Test linking when no related experiences exist."""
        related = bridge.link_task_to_experiences(
            task_id="T-new-001",
            task_category="feature",
            task_title="Brand new feature"
        )

        # Should return empty list without error
        assert isinstance(related, list)


class TestHelperMethods:
    """Test internal helper methods."""

    def test_category_to_goal_type_mapping(self, bridge):
        """Test category to goal type conversion."""
        assert bridge._map_category_to_goal_type("feature") == "implementation"
        assert bridge._map_category_to_goal_type("bugfix") == "debugging"
        assert bridge._map_category_to_goal_type("refactor") == "refactoring"
        assert bridge._map_category_to_goal_type("unknown") == "general"

    def test_priority_to_complexity_mapping(self, bridge):
        """Test priority to complexity conversion."""
        assert bridge._map_priority_to_complexity("critical") == "complex"
        assert bridge._map_priority_to_complexity("high") == "complex"
        assert bridge._map_priority_to_complexity("medium") == "moderate"
        assert bridge._map_priority_to_complexity("low") == "simple"

    def test_domain_inference_from_files(self, bridge):
        """Test domain extraction from file paths."""
        files = ["cortical/auth.py", "cortical/users.py"]
        domain = bridge._infer_domain_from_files(files)
        assert domain == "cortical"

        # Empty files
        assert bridge._infer_domain_from_files([]) == "general"

    def test_tools_inference_from_files(self, bridge):
        """Test tool extraction from file extensions."""
        files = [
            "api.py",
            "frontend.tsx",
            "README.md",
            "config.yaml",
            "deploy.sh"
        ]
        tools = bridge._infer_tools_from_files(files)

        assert "python" in tools
        assert "javascript" in tools
        assert "documentation" in tools
        assert "configuration" in tools
        assert "shell" in tools

    def test_action_type_inference(self, bridge):
        """Test action type inference from file paths."""
        assert bridge._infer_action_type("test_api.py") == "write_test"
        assert bridge._infer_action_type("README.md") == "write_documentation"
        assert bridge._infer_action_type("api.py") == "implement_api"
        assert bridge._infer_action_type("utils.py") == "write_code"
        assert bridge._infer_action_type("config.json") == "modify_file"


class TestPatternExtraction:
    """Test pattern extraction and lesson distillation."""

    def test_extract_patterns_from_experiences(self, bridge):
        """Test that patterns can be extracted from captured experiences."""
        # Capture multiple similar successful experiences
        for i in range(5):
            bridge.capture_task_completion(
                task_id=f"T-pattern-{i}",
                task_title=f"Feature {i}",
                task_category="feature",
                approach="test-first",
                retrospective="TDD approach worked well.",
                files_changed=["code.py", "test.py"]
            )

        # Extract patterns and lessons
        results = bridge.extract_patterns_and_lessons()

        assert isinstance(results, dict)
        assert 'sequence_patterns' in results
        assert 'strategy_patterns' in results
        assert 'lessons' in results

    def test_learning_stats(self, bridge):
        """Test retrieving learning statistics."""
        # Capture a few experiences
        bridge.capture_task_completion(
            task_id="T-stat-001",
            task_title="Task 1",
        )

        bridge.capture_task_failure(
            task_id="T-stat-002",
            task_title="Task 2",
            error_message="Failed"
        )

        # Get stats
        stats = bridge.get_learning_stats()

        assert isinstance(stats, dict)
        assert 'total_experiences' in stats
        assert stats['total_experiences'] >= 2
        assert 'total_patterns' in stats
        assert 'total_lessons' in stats


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_capture_with_minimal_data(self, bridge):
        """Test capturing with only required fields."""
        experience = bridge.capture_task_completion(
            task_id="T-minimal-001"
        )

        assert experience is not None
        assert experience.outcome.outcome_type == OutcomeType.SUCCESS

    def test_capture_with_empty_files_list(self, bridge):
        """Test capturing with empty files list."""
        experience = bridge.capture_task_completion(
            task_id="T-empty-files-001",
            files_changed=[]
        )

        assert experience is not None
        assert len(experience.actions) == 0

    def test_guidance_with_invalid_category(self, bridge):
        """Test guidance retrieval with unknown category."""
        guidance = bridge.get_guidance_for_task(
            task_title="Unknown task type",
            task_category="invalid_category"
        )

        # Should not raise error
        assert isinstance(guidance, dict)
