"""
Extended unit tests for CrisisManager module.
"""

import pytest
from cortical.reasoning import (
    CrisisLevel,
    RecoveryAction,
    CrisisEvent,
    FailureAttempt,
    RepeatedFailureTracker,
    ScopeCreepDetector,
    BlockedDependency,
    CrisisManager,
)


class TestCrisisLevel:
    """Tests for CrisisLevel enum."""

    def test_levels_ordered(self):
        """Test crisis levels are properly ordered."""
        assert CrisisLevel.HICCUP.value < CrisisLevel.OBSTACLE.value
        assert CrisisLevel.OBSTACLE.value < CrisisLevel.WALL.value
        assert CrisisLevel.WALL.value < CrisisLevel.CRISIS.value


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_all_actions_exist(self):
        """Test all recovery actions exist."""
        assert RecoveryAction.CONTINUE is not None
        assert RecoveryAction.ADAPT is not None
        assert RecoveryAction.ROLLBACK is not None
        assert RecoveryAction.PARTIAL_RECOVER is not None
        assert RecoveryAction.ESCALATE is not None
        assert RecoveryAction.STOP is not None


class TestCrisisEvent:
    """Tests for CrisisEvent dataclass."""

    def test_create_event(self):
        """Test creating a crisis event."""
        event = CrisisEvent(
            level=CrisisLevel.HICCUP,
            description="Minor issue"
        )

        assert event.id is not None
        assert event.level == CrisisLevel.HICCUP
        assert event.description == "Minor issue"

    def test_event_with_context(self):
        """Test event with context."""
        event = CrisisEvent(
            level=CrisisLevel.OBSTACLE,
            description="Issue",
            context={"task": "task1", "file": "main.py"}
        )

        assert event.context["task"] == "task1"

    def test_resolve_event(self):
        """Test resolving a crisis event."""
        event = CrisisEvent(
            level=CrisisLevel.HICCUP,
            description="Issue"
        )
        event.resolve(RecoveryAction.CONTINUE, "Fixed by retry")

        assert event.action_taken == RecoveryAction.CONTINUE
        assert event.resolution == "Fixed by retry"
        assert event.resolved_at is not None

    def test_add_lesson(self):
        """Test adding lessons to event."""
        event = CrisisEvent(
            level=CrisisLevel.OBSTACLE,
            description="Issue"
        )
        event.add_lesson("Always check input validation")

        assert "Always check input validation" in event.lessons_learned


class TestFailureAttempt:
    """Tests for FailureAttempt dataclass."""

    def test_create_attempt(self):
        """Test creating a failure attempt record."""
        attempt = FailureAttempt(
            attempt_number=1,
            hypothesis="Invalid configuration",
            action_taken="Updated config file",
            result="Still failing"
        )

        assert attempt.attempt_number == 1
        assert attempt.hypothesis == "Invalid configuration"
        assert attempt.action_taken == "Updated config file"
        assert attempt.result == "Still failing"


class TestRepeatedFailureTracker:
    """Tests for RepeatedFailureTracker class."""

    def test_create_tracker(self):
        """Test creating failure tracker."""
        tracker = RepeatedFailureTracker(issue_description="Tests failing")

        assert tracker is not None
        assert tracker.issue_description == "Tests failing"

    def test_record_attempt(self):
        """Test recording failure attempts."""
        tracker = RepeatedFailureTracker(issue_description="Tests failing")
        count = tracker.record_attempt(
            hypothesis="Wrong input",
            action="Fixed input",
            result="Still fails"
        )

        assert count == 1
        assert len(tracker.attempts) == 1

    def test_should_escalate_false(self):
        """Test escalation check before threshold."""
        tracker = RepeatedFailureTracker(issue_description="Tests failing")
        tracker.record_attempt("h1", "a1", "fail")
        tracker.record_attempt("h2", "a2", "fail")

        assert tracker.should_escalate() is False

    def test_should_escalate_true(self):
        """Test escalation check after threshold."""
        tracker = RepeatedFailureTracker(issue_description="Tests failing")
        tracker.record_attempt("h1", "a1", "fail")
        tracker.record_attempt("h2", "a2", "fail")
        tracker.record_attempt("h3", "a3", "fail")

        assert tracker.should_escalate() is True

    def test_get_escalation_report(self):
        """Test generating escalation report."""
        tracker = RepeatedFailureTracker(issue_description="Tests failing")
        tracker.record_attempt("h1", "a1", "fail")
        tracker.record_attempt("h2", "a2", "fail")
        tracker.record_attempt("h3", "a3", "fail")

        report = tracker.get_escalation_report()

        assert "Escalation" in report
        assert "Tests failing" in report
        assert "h1" in report


class TestScopeCreepDetector:
    """Tests for ScopeCreepDetector class."""

    def test_create_detector(self):
        """Test creating scope creep detector."""
        detector = ScopeCreepDetector(original_scope="Implement feature X")

        assert detector is not None
        assert detector.original_scope == "Implement feature X"

    def test_record_addition(self):
        """Test recording scope additions."""
        detector = ScopeCreepDetector(
            original_scope="Implement feature X",
            original_files=["file1.py", "file2.py"]
        )
        detector.record_addition("Also add caching")

        assert len(detector.additions) == 1
        assert "Also add caching" in detector.additions

    def test_record_unexpected_file(self):
        """Test recording unexpected file changes."""
        detector = ScopeCreepDetector(
            original_scope="Implement feature X",
            original_files=["file1.py", "file2.py"]
        )
        detector.record_unexpected_file("file3.py")

        assert "file3.py" in detector.unexpected_files

    def test_detect_creep_no_issues(self):
        """Test detecting no scope creep."""
        detector = ScopeCreepDetector(
            original_scope="Simple task",
            original_files=["file1.py"]
        )

        is_creeping, warnings = detector.detect_creep()
        assert is_creeping is False

    def test_detect_creep_many_additions(self):
        """Test detecting scope creep from additions."""
        detector = ScopeCreepDetector(
            original_scope="Simple task",
            original_files=["file1.py"]
        )
        detector.record_addition("Add A")
        detector.record_addition("Add B")
        detector.record_addition("Add C")

        is_creeping, warnings = detector.detect_creep()
        assert is_creeping is True
        assert any("one more thing" in w.lower() for w in warnings)

    def test_generate_alert(self):
        """Test generating scope creep alert."""
        detector = ScopeCreepDetector(
            original_scope="Simple task",
            original_files=["file1.py"]
        )
        detector.record_addition("Add A")
        detector.record_addition("Add B")
        detector.record_addition("Add C")

        alert = detector.generate_alert()

        assert "Scope Creep Alert" in alert
        assert "Simple task" in alert


class TestBlockedDependency:
    """Tests for BlockedDependency dataclass."""

    def test_create_blocked_dependency(self):
        """Test creating blocked dependency record."""
        blocked = BlockedDependency(
            description="Waiting for API spec",
            dependency_type="external"
        )

        assert blocked.description == "Waiting for API spec"
        assert blocked.dependency_type == "external"
        assert blocked.resolved is False

    def test_blocked_with_workaround(self):
        """Test blocked dependency with workaround."""
        blocked = BlockedDependency(
            description="Waiting for approval",
            dependency_type="internal",
            workaround_possible=True,
            workaround_description="Use mock data"
        )

        assert blocked.workaround_possible is True
        assert blocked.workaround_description == "Use mock data"


class TestCrisisManager:
    """Tests for CrisisManager class."""

    def test_create_manager(self):
        """Test creating crisis manager."""
        manager = CrisisManager()

        assert manager is not None

    def test_record_crisis(self):
        """Test recording a crisis."""
        manager = CrisisManager()
        event = manager.record_crisis(
            CrisisLevel.HICCUP,
            "Minor error",
            context={"task": "task1"}
        )

        assert event is not None
        assert event.level == CrisisLevel.HICCUP

    def test_get_unresolved_crises(self):
        """Test getting unresolved crises."""
        manager = CrisisManager()
        event1 = manager.record_crisis(CrisisLevel.HICCUP, "Error 1")
        event2 = manager.record_crisis(CrisisLevel.OBSTACLE, "Error 2")

        unresolved = manager.get_unresolved_crises()
        assert len(unresolved) >= 2

    def test_get_crises_by_level(self):
        """Test getting crises by level."""
        manager = CrisisManager()
        manager.record_crisis(CrisisLevel.HICCUP, "Minor")
        manager.record_crisis(CrisisLevel.OBSTACLE, "Medium")
        manager.record_crisis(CrisisLevel.HICCUP, "Another minor")

        hiccups = manager.get_crises_by_level(CrisisLevel.HICCUP)
        assert len(hiccups) == 2

    def test_register_crisis_handler(self):
        """Test registering crisis handler."""
        manager = CrisisManager()
        handled = []

        def handler(event):
            handled.append(event.id)

        manager.register_crisis_handler(handler)
        manager.record_crisis(CrisisLevel.HICCUP, "Test")

        assert len(handled) >= 1

    def test_recommend_action(self):
        """Test recommending recovery action."""
        manager = CrisisManager()
        event = manager.record_crisis(CrisisLevel.OBSTACLE, "Test issue")

        action = manager.recommend_action(event)
        assert action == RecoveryAction.ADAPT

    def test_create_failure_tracker(self):
        """Test creating failure tracker."""
        manager = CrisisManager()
        tracker = manager.create_failure_tracker("Tests failing")

        assert tracker is not None
        assert tracker.issue_description == "Tests failing"

    def test_create_scope_detector(self):
        """Test creating scope detector."""
        manager = CrisisManager()
        detector = manager.create_scope_detector(
            original_scope="Implement feature",
            original_files=["file1.py"]
        )

        assert detector is not None
        assert detector.original_scope == "Implement feature"

    def test_record_blocked(self):
        """Test recording blocked dependency."""
        manager = CrisisManager()
        blocked = manager.record_blocked("Waiting for API", "external")

        assert blocked is not None
        assert blocked.description == "Waiting for API"

    def test_get_active_blockers(self):
        """Test getting active blockers."""
        manager = CrisisManager()
        manager.record_blocked("Blocker 1", "external")
        manager.record_blocked("Blocker 2", "internal")

        blockers = manager.get_active_blockers()
        assert len(blockers) == 2

    def test_get_summary(self):
        """Test getting crisis summary."""
        manager = CrisisManager()
        manager.record_crisis(CrisisLevel.HICCUP, "Error 1")
        manager.record_crisis(CrisisLevel.OBSTACLE, "Error 2")

        summary = manager.get_summary()

        assert "total_events" in summary
        assert summary["total_events"] >= 2

    def test_get_lessons_learned(self):
        """Test getting lessons learned."""
        manager = CrisisManager()
        event = manager.record_crisis(CrisisLevel.OBSTACLE, "Issue")
        event.add_lesson("Always validate input")
        event.resolve(RecoveryAction.ADAPT, "Fixed by validation")

        lessons = manager.get_lessons_learned()
        assert "Always validate input" in lessons
