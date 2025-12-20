"""
Extended unit tests for CrisisManager module.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from cortical.reasoning.crisis_manager import (
    CrisisLevel,
    RecoveryAction,
    CrisisEvent,
    FailureAttempt,
    RepeatedFailureTracker,
    ScopeCreepDetector,
    BlockedDependency,
    CrisisManager,
    RecoveryProcedures,
    CrisisPredictor,
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

class TestRecoveryProcedures:
    """Tests for RecoveryProcedures class."""

    def test_create_recovery_procedures(self):
        """Test creating recovery procedures instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = RecoveryProcedures(memory_path=tmpdir)
            assert recovery is not None

    def test_suggest_recovery_hiccup(self):
        """Test recovery suggestions for HICCUP level crisis."""
        recovery = RecoveryProcedures()
        event = CrisisEvent(level=CrisisLevel.HICCUP, description="Minor error")
        suggestions = recovery.suggest_recovery(event)
        assert RecoveryAction.CONTINUE in suggestions

    def test_suggest_recovery_obstacle(self):
        """Test recovery suggestions for OBSTACLE level crisis."""
        recovery = RecoveryProcedures()
        event = CrisisEvent(level=CrisisLevel.OBSTACLE, description="Blocking issue")
        suggestions = recovery.suggest_recovery(event)
        assert RecoveryAction.ADAPT in suggestions
        assert RecoveryAction.ROLLBACK in suggestions

    def test_suggest_recovery_wall(self):
        """Test recovery suggestions for WALL level crisis."""
        recovery = RecoveryProcedures()
        event = CrisisEvent(level=CrisisLevel.WALL, description="Major problem")
        suggestions = recovery.suggest_recovery(event)
        assert RecoveryAction.ESCALATE in suggestions

    def test_suggest_recovery_crisis(self):
        """Test recovery suggestions for CRISIS level."""
        recovery = RecoveryProcedures()
        event = CrisisEvent(level=CrisisLevel.CRISIS, description="Critical failure")
        suggestions = recovery.suggest_recovery(event)
        assert RecoveryAction.STOP in suggestions

    def test_execute_recovery_continue(self):
        """Test executing CONTINUE action."""
        recovery = RecoveryProcedures()
        result = recovery.execute_recovery(RecoveryAction.CONTINUE, {})
        assert result is True

    def test_escalate_hiccup(self):
        """Test escalating HICCUP to OBSTACLE."""
        recovery = RecoveryProcedures()
        event = CrisisEvent(level=CrisisLevel.HICCUP, description="Minor")
        new_level = recovery.escalate(event)
        assert new_level == CrisisLevel.OBSTACLE

    def test_escalate_wall(self):
        """Test escalating WALL to CRISIS."""
        recovery = RecoveryProcedures()
        event = CrisisEvent(level=CrisisLevel.WALL, description="Problem")
        new_level = recovery.escalate(event)
        assert new_level == CrisisLevel.CRISIS

    def test_record_outcome(self):
        """Test recording recovery outcome."""
        recovery = RecoveryProcedures()
        recovery.record_outcome(RecoveryAction.CONTINUE, success=True)
        assert len(recovery._outcome_history) == 1

    def test_get_outcome_statistics(self):
        """Test getting outcome statistics."""
        recovery = RecoveryProcedures()
        recovery.record_outcome(RecoveryAction.CONTINUE, success=True)
        recovery.record_outcome(RecoveryAction.CONTINUE, success=False)
        stats = recovery.get_outcome_statistics()
        assert 'CONTINUE' in stats
        assert stats['CONTINUE']['total'] == 2

    def test_create_recovery_memory(self):
        """Test creating recovery memory document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = RecoveryProcedures(memory_path=tmpdir)
            file_path = recovery.create_recovery_memory(
                crisis="Test failure",
                actions=["Analyzed logs"],
                lessons=["Need better validation"]
            )
            assert os.path.exists(file_path)


class TestCrisisPredictor:
    """Tests for CrisisPredictor class."""

    def test_create_predictor(self):
        """Test creating crisis predictor."""
        predictor = CrisisPredictor()
        assert predictor is not None

    def test_analyze_patterns_empty(self):
        """Test pattern analysis with no events."""
        predictor = CrisisPredictor()
        patterns = predictor.analyze_patterns([])
        assert patterns == []

    def test_analyze_patterns_repeated_failures(self):
        """Test detecting repeated failure pattern."""
        events = [
            CrisisEvent(level=CrisisLevel.HICCUP, description="test failure"),
            CrisisEvent(level=CrisisLevel.HICCUP, description="test failure"),
        ]
        predictor = CrisisPredictor()
        patterns = predictor.analyze_patterns(events)
        assert any("Repeated" in p for p in patterns)

    def test_predict_risk_no_factors(self):
        """Test risk prediction with no risk factors."""
        predictor = CrisisPredictor()
        risk = predictor.predict_risk({})
        assert risk == 0.0

    def test_predict_risk_repeated_failures(self):
        """Test risk prediction with repeated failures."""
        predictor = CrisisPredictor()
        risk = predictor.predict_risk({'repeated_failures': 3})
        assert risk >= 0.4

    def test_predict_risk_high_complexity(self):
        """Test risk prediction with high complexity."""
        predictor = CrisisPredictor()
        risk = predictor.predict_risk({'complexity': 'high'})
        assert risk >= 0.25

    def test_predict_risk_multiple_factors(self):
        """Test risk prediction with multiple factors."""
        predictor = CrisisPredictor()
        risk = predictor.predict_risk({
            'repeated_failures': 3,
            'complexity': 'high',
        })
        assert risk > 0.5

    def test_suggest_prevention_empty(self):
        """Test prevention suggestions with no risk factors."""
        predictor = CrisisPredictor()
        suggestions = predictor.suggest_prevention([])
        assert suggestions == []

    def test_suggest_prevention_repeated_failures(self):
        """Test prevention for repeated failures."""
        predictor = CrisisPredictor()
        suggestions = predictor.suggest_prevention(['repeated_failures'])
        assert len(suggestions) > 0

    def test_get_similar_past_crises_empty(self):
        """Test finding similar crises with empty history."""
        predictor = CrisisPredictor()
        similar = predictor.get_similar_past_crises({'task': 'test'})
        assert similar == []

    def test_calculate_similarity_identical(self):
        """Test similarity with identical contexts."""
        predictor = CrisisPredictor()
        context = {'task': 'test', 'file': 'a.py'}
        sim = predictor._calculate_similarity(context, context)
        assert sim == 1.0
