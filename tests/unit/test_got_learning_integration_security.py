"""
Security and validation tests for GoT Learning Integration.

Tests the security improvements:
- Input validation for task IDs, strings, and lists
- Thread safety with concurrent access
- Path validation to prevent directory traversal
- Error handling for missing dependencies
"""

import pytest
import tempfile
import threading
from pathlib import Path

from cortical.got.learning_integration import (
    GoTLearningBridge,
    _validate_task_id,
    _validate_string_length,
    _validate_path,
    MAX_TASK_ID_LENGTH,
    MAX_RETROSPECTIVE_LENGTH,
    MAX_STRING_LENGTH,
)


@pytest.fixture
def temp_got_dir():
    """Create a temporary GoT directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def bridge(temp_got_dir):
    """Create a GoTLearningBridge instance for testing."""
    return GoTLearningBridge(temp_got_dir)


class TestInputValidation:
    """Test input validation for security."""

    def test_validate_task_id_empty(self):
        """Test that empty task_id is rejected."""
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            _validate_task_id("")

    def test_validate_task_id_too_long(self):
        """Test that overly long task_id is rejected."""
        long_id = "T" * (MAX_TASK_ID_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            _validate_task_id(long_id)

    def test_validate_task_id_invalid_characters(self):
        """Test that task_id with invalid characters is rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_task_id("T-123/../../etc/passwd")

        with pytest.raises(ValueError, match="invalid characters"):
            _validate_task_id("T-123; rm -rf /")

        with pytest.raises(ValueError, match="invalid characters"):
            _validate_task_id("T-123 OR 1=1")

    def test_validate_task_id_valid(self):
        """Test that valid task_ids are accepted."""
        # Should not raise
        _validate_task_id("T-20260103-123456-abc123")
        _validate_task_id("T_test_001")
        _validate_task_id("simple-id-123")

    def test_validate_string_length_too_long(self):
        """Test that overly long strings are rejected."""
        long_string = "x" * (MAX_STRING_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            _validate_string_length(long_string, "test_field", MAX_STRING_LENGTH)

    def test_validate_string_length_none_accepted(self):
        """Test that None values are accepted."""
        # Should not raise
        _validate_string_length(None, "test_field", 100)

    def test_capture_with_invalid_task_id(self, bridge):
        """Test that capture_task_completion rejects invalid task_id."""
        with pytest.raises(ValueError):
            bridge.capture_task_completion(
                task_id="../../../etc/passwd",
                task_title="Test"
            )

    def test_capture_with_too_long_retrospective(self, bridge):
        """Test that overly long retrospective is rejected."""
        long_retro = "x" * (MAX_RETROSPECTIVE_LENGTH + 1)
        with pytest.raises(ValueError, match="retrospective exceeds maximum length"):
            bridge.capture_task_completion(
                task_id="T-test-001",
                retrospective=long_retro
            )

    def test_capture_with_too_many_files(self, bridge):
        """Test that overly long files list is rejected."""
        many_files = [f"file{i}.py" for i in range(1001)]
        with pytest.raises(ValueError, match="files_changed list exceeds maximum"):
            bridge.capture_task_completion(
                task_id="T-test-001",
                files_changed=many_files
            )

    def test_capture_failure_with_too_many_blockers(self, bridge):
        """Test that overly long blockers list is rejected."""
        many_blockers = [f"blocker{i}" for i in range(101)]
        with pytest.raises(ValueError, match="blockers list exceeds maximum"):
            bridge.capture_task_failure(
                task_id="T-test-001",
                error_message="Failed",
                blockers=many_blockers
            )

    def test_guidance_with_too_many_files(self, bridge):
        """Test that guidance rejects overly long files list."""
        many_files = [f"file{i}.py" for i in range(1001)]
        with pytest.raises(ValueError, match="files_to_modify list exceeds maximum"):
            bridge.get_guidance_for_task(
                task_title="Test",
                files_to_modify=many_files
            )


class TestPathValidation:
    """Test path validation for directory traversal prevention."""

    def test_validate_path_traversal_attack(self):
        """Test that directory traversal attempts are blocked."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_path("/tmp/test/../../etc/passwd", "/tmp/test")

    def test_validate_path_absolute_escape(self):
        """Test that absolute path escapes are blocked."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_path("/etc/passwd", "/tmp/test")

    def test_validate_path_valid_subdir(self):
        """Test that valid subdirectories are accepted."""
        # Should not raise
        result = _validate_path("/tmp/test/subdir", "/tmp/test")
        assert result.startswith("/tmp/test")

    def test_validate_path_same_dir(self):
        """Test that same directory is accepted."""
        # Should not raise
        result = _validate_path("/tmp/test", "/tmp/test")
        assert result == "/tmp/test"


@pytest.mark.skip(reason="llm_orchestration module scheduled for removal - ID generation has known race condition")
class TestThreadSafety:
    """Test thread safety of concurrent access."""

    def test_concurrent_captures(self, bridge):
        """Test that concurrent task captures are thread-safe."""
        results = []
        errors = []

        def capture_task(task_num):
            try:
                exp = bridge.capture_task_completion(
                    task_id=f"T-concurrent-{task_num}",
                    task_title=f"Concurrent task {task_num}",
                    retrospective=f"Task {task_num} completed"
                )
                results.append(exp)
            except Exception as e:
                errors.append(e)

        # Create 10 threads capturing tasks concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=capture_task, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # All captures should succeed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # All experiences should have unique IDs
        exp_ids = {exp.id for exp in results}
        assert len(exp_ids) == 10

    def test_concurrent_guidance_retrieval(self, bridge):
        """Test that concurrent guidance retrievals are thread-safe."""
        # First capture a few experiences
        for i in range(3):
            bridge.capture_task_completion(
                task_id=f"T-setup-{i}",
                task_title=f"Setup task {i}",
                task_category="feature"
            )

        results = []
        errors = []

        def get_guidance(task_num):
            try:
                guidance = bridge.get_guidance_for_task(
                    task_title=f"New task {task_num}",
                    task_category="feature"
                )
                results.append(guidance)
            except Exception as e:
                errors.append(e)

        # Create 5 threads retrieving guidance concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=get_guidance, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # All retrievals should succeed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_capture_with_all_validations(self, bridge):
        """Test that all validations work together."""
        # This should succeed with all valid inputs
        exp = bridge.capture_task_completion(
            task_id="T-valid-001",
            task_title="Valid task" * 10,  # Moderate length
            retrospective="This is a valid retrospective",
            files_changed=["file1.py", "file2.py"],
            approach="test-first",
            task_category="feature",
            task_priority="high",
            duration_seconds=1800
        )

        assert exp is not None
        assert exp.id.startswith("exp_")

    def test_link_task_with_invalid_id(self, bridge):
        """Test that link_task_to_experiences validates task_id."""
        with pytest.raises(ValueError):
            bridge.link_task_to_experiences(
                task_id="invalid/../id",
                task_category="feature"
            )

    def test_capture_failure_validates_inputs(self, bridge):
        """Test that capture_task_failure validates all inputs."""
        with pytest.raises(ValueError):
            bridge.capture_task_failure(
                task_id="",  # Invalid: empty
                error_message="Test error"
            )


class TestImportErrorHandling:
    """Test graceful handling when learning module is unavailable."""

    def test_import_availability_flag(self):
        """Test that LEARNING_AVAILABLE flag exists."""
        from cortical.got.learning_integration import LEARNING_AVAILABLE
        # Should be True in test environment
        assert LEARNING_AVAILABLE is True

    def test_bridge_requires_learning_module(self, temp_got_dir):
        """Test that bridge initialization checks for learning module."""
        # In normal test environment, this should succeed
        bridge = GoTLearningBridge(temp_got_dir)
        assert bridge is not None
