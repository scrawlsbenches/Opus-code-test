"""
Comprehensive unit tests for collaboration.py module.

Uses unittest.TestCase pattern for consistency with existing tests.
Focus on achieving 80%+ coverage, particularly for:
- CollaborationManager methods
- StatusUpdate dataclass and to_markdown()
- Blocker/BlockerType
- DisagreementRecord
- ActiveWorkHandoff
- ConflictEvent/ConflictType
- QuestionBatcher (all methods)
"""

import unittest
from datetime import datetime
from cortical.reasoning.collaboration import (
    CollaborationMode,
    BlockerType,
    ConflictType,
    StatusUpdate,
    Blocker,
    DisagreementRecord,
    ParallelWorkBoundary,
    ConflictEvent,
    ActiveWorkHandoff,
    CollaborationManager,
    QuestionBatcher,
    BatchedQuestion,
)


class TestCollaborationMode(unittest.TestCase):
    """Test CollaborationMode enum."""

    def test_all_modes_exist(self):
        """Verify all collaboration modes are defined."""
        self.assertIsNotNone(CollaborationMode.SYNCHRONOUS)
        self.assertIsNotNone(CollaborationMode.ASYNCHRONOUS)
        self.assertIsNotNone(CollaborationMode.SEMI_SYNCHRONOUS)

    def test_modes_are_distinct(self):
        """Verify modes have unique values."""
        self.assertNotEqual(CollaborationMode.SYNCHRONOUS, CollaborationMode.ASYNCHRONOUS)
        self.assertNotEqual(CollaborationMode.ASYNCHRONOUS, CollaborationMode.SEMI_SYNCHRONOUS)
        self.assertNotEqual(CollaborationMode.SYNCHRONOUS, CollaborationMode.SEMI_SYNCHRONOUS)


class TestBlockerType(unittest.TestCase):
    """Test BlockerType enum."""

    def test_all_types_exist(self):
        """Verify all blocker types are defined."""
        self.assertIsNotNone(BlockerType.HARD)
        self.assertIsNotNone(BlockerType.SOFT)
        self.assertIsNotNone(BlockerType.INFO)

    def test_types_are_distinct(self):
        """Verify blocker types have unique values."""
        self.assertNotEqual(BlockerType.HARD, BlockerType.SOFT)
        self.assertNotEqual(BlockerType.SOFT, BlockerType.INFO)
        self.assertNotEqual(BlockerType.HARD, BlockerType.INFO)


class TestConflictType(unittest.TestCase):
    """Test ConflictType enum."""

    def test_all_types_exist(self):
        """Verify all conflict types are defined."""
        self.assertIsNotNone(ConflictType.FILE_CONFLICT)
        self.assertIsNotNone(ConflictType.LOGIC_CONFLICT)
        self.assertIsNotNone(ConflictType.DEPENDENCY_CONFLICT)
        self.assertIsNotNone(ConflictType.SCOPE_OVERLAP)

    def test_types_are_distinct(self):
        """Verify conflict types have unique values."""
        types = [
            ConflictType.FILE_CONFLICT,
            ConflictType.LOGIC_CONFLICT,
            ConflictType.DEPENDENCY_CONFLICT,
            ConflictType.SCOPE_OVERLAP
        ]
        self.assertEqual(len(types), len(set(types)))


class TestStatusUpdate(unittest.TestCase):
    """Test StatusUpdate dataclass."""

    def test_create_minimal(self):
        """Test creating with only required field."""
        update = StatusUpdate(task_name="Test Task")
        self.assertEqual(update.task_name, "Test Task")
        self.assertEqual(update.progress_percent, 0)

    def test_create_with_all_fields(self):
        """Test creating with all fields populated."""
        update = StatusUpdate(
            task_name="Complex Task",
            progress_percent=75,
            current_phase="Testing",
            eta_minutes=20,
            completed_items=["Item 1", "Item 2"],
            in_progress_items=["Item 3"],
            blockers=["Blocker 1"],
            concerns=["Concern 1"],
            needs_from_human=["Review needed"]
        )

        self.assertEqual(update.task_name, "Complex Task")
        self.assertEqual(update.progress_percent, 75)
        self.assertEqual(update.current_phase, "Testing")
        self.assertEqual(update.eta_minutes, 20)
        self.assertEqual(len(update.completed_items), 2)
        self.assertEqual(len(update.in_progress_items), 1)
        self.assertEqual(len(update.blockers), 1)
        self.assertEqual(len(update.concerns), 1)
        self.assertEqual(len(update.needs_from_human), 1)

    def test_default_values(self):
        """Test that default values are correct."""
        update = StatusUpdate(task_name="Task")

        self.assertEqual(update.progress_percent, 0)
        self.assertEqual(update.current_phase, "")
        self.assertIsNone(update.eta_minutes)
        self.assertEqual(update.completed_items, [])
        self.assertEqual(update.in_progress_items, [])
        self.assertEqual(update.blockers, [])
        self.assertEqual(update.concerns, [])
        self.assertEqual(update.needs_from_human, [])
        self.assertIsInstance(update.timestamp, datetime)

    def test_to_markdown_basic(self):
        """Test markdown generation with basic fields."""
        update = StatusUpdate(
            task_name="Feature Implementation",
            progress_percent=50,
            current_phase="Development"
        )

        md = update.to_markdown()
        self.assertIn("## Status: Feature Implementation", md)
        self.assertIn("50%", md)
        self.assertIn("Development", md)

    def test_to_markdown_with_eta(self):
        """Test markdown includes ETA when present."""
        update = StatusUpdate(
            task_name="Task",
            progress_percent=30,
            current_phase="Working",
            eta_minutes=45
        )

        md = update.to_markdown()
        self.assertIn("45 minutes", md)

    def test_to_markdown_without_eta(self):
        """Test markdown doesn't include ETA line when None."""
        update = StatusUpdate(
            task_name="Task",
            progress_percent=30,
            current_phase="Working"
        )

        md = update.to_markdown()
        self.assertNotIn("ETA:", md)

    def test_to_markdown_with_completed_items(self):
        """Test markdown formatting of completed items."""
        update = StatusUpdate(
            task_name="Task",
            completed_items=["Setup database", "Create models"]
        )

        md = update.to_markdown()
        self.assertIn("**Completed:**", md)
        self.assertIn("[x] Setup database", md)
        self.assertIn("[x] Create models", md)

    def test_to_markdown_with_in_progress_items(self):
        """Test markdown formatting of in-progress items."""
        update = StatusUpdate(
            task_name="Task",
            in_progress_items=["Writing tests", "Documenting API"]
        )

        md = update.to_markdown()
        self.assertIn("**In progress:**", md)
        self.assertIn("[ ] Writing tests", md)
        self.assertIn("[ ] Documenting API", md)

    def test_to_markdown_with_blockers(self):
        """Test markdown shows blockers when present."""
        update = StatusUpdate(
            task_name="Task",
            blockers=["Need API credentials", "Database schema unclear"]
        )

        md = update.to_markdown()
        self.assertIn("Need API credentials", md)
        self.assertIn("Database schema unclear", md)

    def test_to_markdown_no_blockers(self):
        """Test markdown shows 'None' when no blockers."""
        update = StatusUpdate(task_name="Task")

        md = update.to_markdown()
        self.assertIn("**Blockers:** None", md)

    def test_to_markdown_no_concerns(self):
        """Test markdown shows 'None' when no concerns."""
        update = StatusUpdate(task_name="Task")

        md = update.to_markdown()
        self.assertIn("**Concerns:** None", md)

    def test_to_markdown_needs_nothing(self):
        """Test markdown shows 'Nothing' when no needs."""
        update = StatusUpdate(task_name="Task")

        md = update.to_markdown()
        self.assertIn("**Need from you:** Nothing", md)

    def test_to_markdown_comprehensive(self):
        """Test markdown with all sections populated."""
        update = StatusUpdate(
            task_name="OAuth Implementation",
            progress_percent=60,
            current_phase="Integration Testing",
            eta_minutes=90,
            completed_items=["Auth flow", "Token refresh"],
            in_progress_items=["Session management"],
            blockers=["External API down"],
            concerns=["Performance under load"],
            needs_from_human=["Review security model"]
        )

        md = update.to_markdown()

        # Verify all sections present
        self.assertIn("OAuth Implementation", md)
        self.assertIn("60%", md)
        self.assertIn("Integration Testing", md)
        self.assertIn("90 minutes", md)
        self.assertIn("[x] Auth flow", md)
        self.assertIn("[ ] Session management", md)
        self.assertIn("External API down", md)
        self.assertIn("Performance under load", md)
        self.assertIn("Review security model", md)


class TestBlocker(unittest.TestCase):
    """Test Blocker dataclass."""

    def test_create_minimal(self):
        """Test creating with defaults."""
        blocker = Blocker()

        self.assertIsNotNone(blocker.id)
        self.assertEqual(blocker.description, "")
        self.assertEqual(blocker.blocker_type, BlockerType.SOFT)
        self.assertFalse(blocker.resolved)

    def test_create_with_fields(self):
        """Test creating with all fields."""
        blocker = Blocker(
            description="Database connection failed",
            blocker_type=BlockerType.HARD,
            resolution_needed="Fix connection string",
            workaround="Use local database",
            context={"host": "db.example.com", "port": 5432}
        )

        self.assertEqual(blocker.description, "Database connection failed")
        self.assertEqual(blocker.blocker_type, BlockerType.HARD)
        self.assertEqual(blocker.resolution_needed, "Fix connection string")
        self.assertEqual(blocker.workaround, "Use local database")
        self.assertEqual(blocker.context["host"], "db.example.com")

    def test_unique_ids(self):
        """Test that each blocker gets unique ID."""
        blocker1 = Blocker()
        blocker2 = Blocker()

        self.assertNotEqual(blocker1.id, blocker2.id)

    def test_resolve(self):
        """Test resolving a blocker."""
        blocker = Blocker(description="Issue")
        self.assertFalse(blocker.resolved)
        self.assertIsNone(blocker.resolution)
        self.assertIsNone(blocker.resolved_at)

        blocker.resolve("Fixed by updating configuration")

        self.assertTrue(blocker.resolved)
        self.assertEqual(blocker.resolution, "Fixed by updating configuration")
        self.assertIsInstance(blocker.resolved_at, datetime)

    def test_resolve_multiple_times(self):
        """Test that resolving multiple times updates fields."""
        blocker = Blocker(description="Issue")

        blocker.resolve("First resolution")
        first_resolution = blocker.resolution
        first_time = blocker.resolved_at

        blocker.resolve("Second resolution")

        self.assertEqual(blocker.resolution, "Second resolution")
        self.assertNotEqual(blocker.resolution, first_resolution)
        # Note: resolved_at may be same or different depending on timing

    def test_with_workaround(self):
        """Test blocker with workaround."""
        blocker = Blocker(
            description="API rate limited",
            blocker_type=BlockerType.SOFT,
            workaround="Cache responses for 1 hour"
        )

        self.assertEqual(blocker.workaround, "Cache responses for 1 hour")

    def test_without_workaround(self):
        """Test blocker without workaround defaults to None."""
        blocker = Blocker(description="Hard stop")
        self.assertIsNone(blocker.workaround)


class TestDisagreementRecord(unittest.TestCase):
    """Test DisagreementRecord dataclass."""

    def test_create_minimal(self):
        """Test creating with defaults."""
        record = DisagreementRecord()

        self.assertIsNotNone(record.id)
        self.assertEqual(record.instruction_given, "")
        self.assertEqual(record.concern_raised, "")
        self.assertEqual(record.evidence, [])
        self.assertIsNone(record.human_decision)
        self.assertIsNone(record.outcome)

    def test_create_full(self):
        """Test creating with all fields."""
        record = DisagreementRecord(
            instruction_given="Delete production database",
            concern_raised="This will cause data loss",
            evidence=["No backup exists", "Recovery takes 24 hours"],
            risk_if_proceed="Permanent data loss",
            alternative_suggested="Create backup first, then delete"
        )

        self.assertEqual(record.instruction_given, "Delete production database")
        self.assertEqual(record.concern_raised, "This will cause data loss")
        self.assertEqual(len(record.evidence), 2)
        self.assertEqual(record.risk_if_proceed, "Permanent data loss")
        self.assertEqual(record.alternative_suggested, "Create backup first, then delete")

    def test_unique_ids(self):
        """Test that each record gets unique ID."""
        record1 = DisagreementRecord()
        record2 = DisagreementRecord()

        self.assertNotEqual(record1.id, record2.id)

    def test_to_markdown_structure(self):
        """Test markdown has required structure."""
        record = DisagreementRecord(
            instruction_given="Use eval() for dynamic code",
            concern_raised="Security vulnerability",
            evidence=["OWASP Top 10", "Previous exploits"],
            risk_if_proceed="Remote code execution",
            alternative_suggested="Use ast.literal_eval() instead"
        )

        md = record.to_markdown()

        # Check major sections
        self.assertIn("## Respectful Disagreement", md)
        self.assertIn("**Your instruction:**", md)
        self.assertIn("**My concern:**", md)
        self.assertIn("**Evidence:**", md)
        self.assertIn("**Risk if we proceed as instructed:**", md)
        self.assertIn("**Alternative I'd suggest:**", md)
        self.assertIn("**However:**", md)
        self.assertIn("**Your call:**", md)

    def test_to_markdown_content(self):
        """Test markdown includes actual content."""
        record = DisagreementRecord(
            instruction_given="Skip unit tests",
            concern_raised="Code quality will suffer",
            evidence=["Past bugs from untested code", "Team policy requires tests"],
            risk_if_proceed="Bugs in production",
            alternative_suggested="Write minimal tests for critical paths"
        )

        md = record.to_markdown()

        self.assertIn("Skip unit tests", md)
        self.assertIn("Code quality will suffer", md)
        self.assertIn("Past bugs from untested code", md)
        self.assertIn("Team policy requires tests", md)
        self.assertIn("Bugs in production", md)
        self.assertIn("Write minimal tests for critical paths", md)

    def test_to_markdown_evidence_formatting(self):
        """Test evidence is formatted as bullet list."""
        record = DisagreementRecord(
            instruction_given="Instruction",
            concern_raised="Concern",
            evidence=["Evidence 1", "Evidence 2", "Evidence 3"],
            risk_if_proceed="Risk",
            alternative_suggested="Alternative"
        )

        md = record.to_markdown()

        self.assertIn("- Evidence 1", md)
        self.assertIn("- Evidence 2", md)
        self.assertIn("- Evidence 3", md)

    def test_to_markdown_empty_evidence(self):
        """Test markdown handles empty evidence list."""
        record = DisagreementRecord(
            instruction_given="Do X",
            concern_raised="X is bad",
            evidence=[],
            risk_if_proceed="Bad things",
            alternative_suggested="Do Y"
        )

        md = record.to_markdown()
        self.assertIn("**Evidence:**", md)


class TestConflictEvent(unittest.TestCase):
    """Test ConflictEvent dataclass."""

    def test_create_minimal(self):
        """Test creating with defaults."""
        event = ConflictEvent()

        self.assertIsNotNone(event.id)
        self.assertEqual(event.conflict_type, ConflictType.FILE_CONFLICT)
        self.assertEqual(event.agents_involved, [])
        self.assertFalse(event.escalated)
        self.assertIsNone(event.resolution)

    def test_create_full(self):
        """Test creating with all fields."""
        event = ConflictEvent(
            conflict_type=ConflictType.LOGIC_CONFLICT,
            agents_involved=["agent-1", "agent-2"],
            description="Incompatible approaches",
            files_affected=["src/main.py", "src/utils.py"]
        )

        self.assertEqual(event.conflict_type, ConflictType.LOGIC_CONFLICT)
        self.assertEqual(len(event.agents_involved), 2)
        self.assertEqual(event.description, "Incompatible approaches")
        self.assertEqual(len(event.files_affected), 2)

    def test_unique_ids(self):
        """Test that each event gets unique ID."""
        event1 = ConflictEvent()
        event2 = ConflictEvent()

        self.assertNotEqual(event1.id, event2.id)

    def test_default_conflict_type(self):
        """Test default conflict type is FILE_CONFLICT."""
        event = ConflictEvent()
        self.assertEqual(event.conflict_type, ConflictType.FILE_CONFLICT)

    def test_escalated_flag(self):
        """Test escalated flag defaults to False."""
        event = ConflictEvent()
        self.assertFalse(event.escalated)

    def test_timestamps(self):
        """Test timestamps are set."""
        event = ConflictEvent()

        self.assertIsInstance(event.detected_at, datetime)
        self.assertIsNone(event.resolved_at)


class TestActiveWorkHandoff(unittest.TestCase):
    """Test ActiveWorkHandoff dataclass."""

    def test_create_minimal(self):
        """Test creating with required fields only."""
        handoff = ActiveWorkHandoff(
            task_description="Implement feature X",
            status="In progress",
            urgency="medium"
        )

        self.assertEqual(handoff.task_description, "Implement feature X")
        self.assertEqual(handoff.status, "In progress")
        self.assertEqual(handoff.urgency, "medium")

    def test_default_values(self):
        """Test default values for optional fields."""
        handoff = ActiveWorkHandoff(
            task_description="Task",
            status="Status",
            urgency="low"
        )

        self.assertEqual(handoff.files_working, [])
        self.assertEqual(handoff.files_in_progress, {})
        self.assertEqual(handoff.known_issues, [])
        self.assertEqual(handoff.key_decisions, {})
        self.assertEqual(handoff.gotchas, [])
        self.assertEqual(handoff.files_to_read_first, [])
        self.assertEqual(handoff.immediate_next_steps, [])
        self.assertEqual(handoff.open_questions, [])
        self.assertEqual(handoff.verification_command, "")
        self.assertIsInstance(handoff.created_at, datetime)

    def test_create_comprehensive(self):
        """Test creating with all fields populated."""
        handoff = ActiveWorkHandoff(
            task_description="Implement OAuth",
            status="75% complete",
            urgency="high",
            files_working=["auth.py", "models.py"],
            files_in_progress={"routes.py": "Add logout endpoint"},
            known_issues=["Token refresh sometimes fails"],
            key_decisions={"Storage": "Use Redis", "Expiry": "1 hour"},
            gotchas=["Must call refresh before token expires"],
            files_to_read_first=["auth.py", "README.md", "config.py"],
            immediate_next_steps=["Fix token refresh", "Add tests"],
            open_questions=["Support multiple providers?"],
            verification_command="pytest tests/auth/ -v"
        )

        self.assertEqual(len(handoff.files_working), 2)
        self.assertEqual(len(handoff.files_in_progress), 1)
        self.assertEqual(len(handoff.known_issues), 1)
        self.assertEqual(len(handoff.key_decisions), 2)
        self.assertEqual(len(handoff.gotchas), 1)
        self.assertEqual(len(handoff.files_to_read_first), 3)
        self.assertEqual(len(handoff.immediate_next_steps), 2)
        self.assertEqual(len(handoff.open_questions), 1)
        self.assertEqual(handoff.verification_command, "pytest tests/auth/ -v")

    def test_to_markdown_structure(self):
        """Test markdown has required structure."""
        handoff = ActiveWorkHandoff(
            task_description="Task",
            status="Status",
            urgency="medium"
        )

        md = handoff.to_markdown()

        self.assertIn("## Active Work Handoff", md)
        self.assertIn("**Task:**", md)
        self.assertIn("**Status:**", md)
        self.assertIn("**Urgency:**", md)
        self.assertIn("### Current State", md)
        self.assertIn("**What's working:**", md)
        self.assertIn("**What's in progress:**", md)
        self.assertIn("**What's broken:**", md)
        self.assertIn("### Context You Need", md)
        self.assertIn("**Why we're doing it this way:**", md)
        self.assertIn("**Gotchas discovered:**", md)
        self.assertIn("**Files to read first:**", md)
        self.assertIn("### Immediate Next Steps", md)
        self.assertIn("### Questions Still Open", md)
        self.assertIn("### How to Verify You're On Track", md)

    def test_to_markdown_content(self):
        """Test markdown includes all content."""
        handoff = ActiveWorkHandoff(
            task_description="Add caching layer",
            status="50% complete",
            urgency="high",
            files_working=["cache.py"],
            files_in_progress={"redis_backend.py": "Add connection pooling"},
            known_issues=["Memory leak in cache eviction"],
            key_decisions={"Backend": "Redis over Memcached"},
            gotchas=["Must set max memory limit"],
            files_to_read_first=["cache.py", "config.py"],
            immediate_next_steps=["Fix memory leak", "Add monitoring"],
            open_questions=["What's the cache size limit?"],
            verification_command="pytest tests/cache/ && python scripts/test_cache.py"
        )

        md = handoff.to_markdown()

        # Verify content is present
        self.assertIn("Add caching layer", md)
        self.assertIn("50% complete", md)
        self.assertIn("high", md)
        self.assertIn("cache.py", md)
        self.assertIn("redis_backend.py: Add connection pooling", md)
        self.assertIn("Memory leak in cache eviction", md)
        self.assertIn("Backend: Redis over Memcached", md)
        self.assertIn("Must set max memory limit", md)
        self.assertIn("config.py", md)
        self.assertIn("Fix memory leak", md)
        self.assertIn("What's the cache size limit?", md)
        self.assertIn("pytest tests/cache/", md)

    def test_to_markdown_numbered_lists(self):
        """Test that files_to_read_first and steps are numbered."""
        handoff = ActiveWorkHandoff(
            task_description="Task",
            status="Status",
            urgency="medium",
            files_to_read_first=["file1.py", "file2.py", "file3.py"],
            immediate_next_steps=["Step 1", "Step 2", "Step 3"]
        )

        md = handoff.to_markdown()

        # Check files are numbered
        self.assertIn("1. file1.py", md)
        self.assertIn("2. file2.py", md)
        self.assertIn("3. file3.py", md)

        # Check steps are numbered
        self.assertIn("1. Step 1", md)
        self.assertIn("2. Step 2", md)
        self.assertIn("3. Step 3", md)

    def test_to_markdown_empty_sections(self):
        """Test markdown handles empty lists/dicts gracefully."""
        handoff = ActiveWorkHandoff(
            task_description="Minimal handoff",
            status="Just started",
            urgency="low"
        )

        md = handoff.to_markdown()

        # Should still have structure even with empty content
        self.assertIn("### Current State", md)
        self.assertIn("### Context You Need", md)
        self.assertIn("### Immediate Next Steps", md)


class TestCollaborationManager(unittest.TestCase):
    """Test CollaborationManager class."""

    def test_create_default_mode(self):
        """Test creating manager with default mode."""
        manager = CollaborationManager()
        self.assertEqual(manager.mode, CollaborationMode.SEMI_SYNCHRONOUS)

    def test_create_with_mode(self):
        """Test creating manager with specific mode."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)
        self.assertEqual(manager.mode, CollaborationMode.SYNCHRONOUS)

    def test_mode_property_getter(self):
        """Test mode property getter."""
        manager = CollaborationManager(mode=CollaborationMode.ASYNCHRONOUS)
        self.assertEqual(manager.mode, CollaborationMode.ASYNCHRONOUS)

    def test_mode_property_setter(self):
        """Test mode property setter."""
        manager = CollaborationManager()
        manager.mode = CollaborationMode.SYNCHRONOUS
        self.assertEqual(manager.mode, CollaborationMode.SYNCHRONOUS)

    def test_get_update_interval_synchronous(self):
        """Test update interval for synchronous mode."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)
        self.assertEqual(manager.get_update_interval(), 5)

    def test_get_update_interval_asynchronous(self):
        """Test update interval for asynchronous mode."""
        manager = CollaborationManager(mode=CollaborationMode.ASYNCHRONOUS)
        self.assertEqual(manager.get_update_interval(), 60)

    def test_get_update_interval_semi_synchronous(self):
        """Test update interval for semi-synchronous mode."""
        manager = CollaborationManager(mode=CollaborationMode.SEMI_SYNCHRONOUS)
        self.assertEqual(manager.get_update_interval(), 15)

    def test_post_status(self):
        """Test posting a status update."""
        manager = CollaborationManager()
        update = StatusUpdate(task_name="Test", progress_percent=25)

        manager.post_status(update)

        summary = manager.get_summary()
        self.assertEqual(summary['status_updates'], 1)

    def test_post_multiple_status_updates(self):
        """Test posting multiple status updates."""
        manager = CollaborationManager()

        manager.post_status(StatusUpdate(task_name="Task 1", progress_percent=10))
        manager.post_status(StatusUpdate(task_name="Task 1", progress_percent=50))
        manager.post_status(StatusUpdate(task_name="Task 1", progress_percent=100))

        summary = manager.get_summary()
        self.assertEqual(summary['status_updates'], 3)

    def test_raise_blocker_minimal(self):
        """Test raising blocker with minimal args."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker("Cannot proceed")

        self.assertEqual(blocker.description, "Cannot proceed")
        self.assertEqual(blocker.blocker_type, BlockerType.SOFT)

    def test_raise_blocker_hard(self):
        """Test raising hard blocker."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker(
            "Critical failure",
            blocker_type=BlockerType.HARD,
            resolution_needed="Fix immediately"
        )

        self.assertEqual(blocker.blocker_type, BlockerType.HARD)
        self.assertEqual(blocker.resolution_needed, "Fix immediately")

    def test_raise_blocker_with_context(self):
        """Test raising blocker with context dictionary."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker(
            "Database error",
            context={"table": "users", "error_code": 1045}
        )

        self.assertEqual(blocker.context["table"], "users")
        self.assertEqual(blocker.context["error_code"], 1045)

    def test_raise_blocker_with_workaround(self):
        """Test raising blocker with workaround."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker(
            "API unavailable",
            blocker_type=BlockerType.SOFT,
            workaround="Use cached data"
        )

        self.assertEqual(blocker.workaround, "Use cached data")

    def test_resolve_blocker(self):
        """Test resolving an existing blocker."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker("Issue")

        manager.resolve_blocker(blocker.id, "Fixed")

        self.assertTrue(blocker.resolved)
        self.assertEqual(blocker.resolution, "Fixed")

    def test_resolve_nonexistent_blocker(self):
        """Test resolving non-existent blocker doesn't raise error."""
        manager = CollaborationManager()

        # Should not raise exception
        manager.resolve_blocker("nonexistent-id", "Fixed")

    def test_get_active_blockers_empty(self):
        """Test getting active blockers when none exist."""
        manager = CollaborationManager()
        active = manager.get_active_blockers()

        self.assertEqual(len(active), 0)

    def test_get_active_blockers(self):
        """Test getting active blockers."""
        manager = CollaborationManager()

        blocker1 = manager.raise_blocker("Issue 1")
        blocker2 = manager.raise_blocker("Issue 2")
        blocker3 = manager.raise_blocker("Issue 3")

        # Resolve one
        manager.resolve_blocker(blocker2.id, "Fixed")

        active = manager.get_active_blockers()

        self.assertEqual(len(active), 2)
        descriptions = [b.description for b in active]
        self.assertIn("Issue 1", descriptions)
        self.assertIn("Issue 3", descriptions)
        self.assertNotIn("Issue 2", descriptions)

    def test_get_hard_blockers_empty(self):
        """Test getting hard blockers when none exist."""
        manager = CollaborationManager()
        manager.raise_blocker("Soft issue", blocker_type=BlockerType.SOFT)

        hard = manager.get_hard_blockers()
        self.assertEqual(len(hard), 0)

    def test_get_hard_blockers(self):
        """Test getting only hard blockers."""
        manager = CollaborationManager()

        manager.raise_blocker("Soft issue", blocker_type=BlockerType.SOFT)
        manager.raise_blocker("Info issue", blocker_type=BlockerType.INFO)
        manager.raise_blocker("Hard issue 1", blocker_type=BlockerType.HARD)
        manager.raise_blocker("Hard issue 2", blocker_type=BlockerType.HARD)

        hard = manager.get_hard_blockers()

        self.assertEqual(len(hard), 2)
        for blocker in hard:
            self.assertEqual(blocker.blocker_type, BlockerType.HARD)

    def test_get_hard_blockers_excludes_resolved(self):
        """Test that resolved hard blockers are not returned."""
        manager = CollaborationManager()

        blocker1 = manager.raise_blocker("Hard 1", blocker_type=BlockerType.HARD)
        blocker2 = manager.raise_blocker("Hard 2", blocker_type=BlockerType.HARD)

        manager.resolve_blocker(blocker1.id, "Fixed")

        hard = manager.get_hard_blockers()
        self.assertEqual(len(hard), 1)
        self.assertEqual(hard[0].description, "Hard 2")

    def test_record_disagreement(self):
        """Test recording a disagreement."""
        manager = CollaborationManager()

        record = manager.record_disagreement(
            instruction="Delete all logs",
            concern="We need logs for debugging",
            evidence=["Recent outage required logs", "Compliance requirement"],
            risk="Cannot debug future issues",
            alternative="Archive old logs instead"
        )

        self.assertEqual(record.instruction_given, "Delete all logs")
        self.assertEqual(record.concern_raised, "We need logs for debugging")
        self.assertEqual(len(record.evidence), 2)

    def test_record_multiple_disagreements(self):
        """Test recording multiple disagreements."""
        manager = CollaborationManager()

        manager.record_disagreement("Instr1", "Concern1", [], "Risk1", "Alt1")
        manager.record_disagreement("Instr2", "Concern2", [], "Risk2", "Alt2")

        summary = manager.get_summary()
        self.assertEqual(summary['disagreements'], 2)

    def test_create_boundary_minimal(self):
        """Test creating boundary with minimal args."""
        manager = CollaborationManager()
        boundary = manager.create_boundary("agent1", "Frontend work")

        self.assertEqual(boundary.agent_id, "agent1")
        self.assertEqual(boundary.scope_description, "Frontend work")
        self.assertEqual(len(boundary.files_owned), 0)

    def test_create_boundary_with_files(self):
        """Test creating boundary with files."""
        manager = CollaborationManager()
        files = {"src/app.js", "src/utils.js"}
        boundary = manager.create_boundary("agent1", "Frontend", files=files)

        self.assertEqual(len(boundary.files_owned), 2)
        self.assertIn("src/app.js", boundary.files_owned)
        self.assertIn("src/utils.js", boundary.files_owned)

    def test_check_conflicts_none(self):
        """Test checking conflicts when none exist."""
        manager = CollaborationManager()

        manager.create_boundary("agent1", "Work 1", {"file1.py"})
        manager.create_boundary("agent2", "Work 2", {"file2.py"})

        conflicts = manager.check_conflicts()
        self.assertEqual(len(conflicts), 0)

    def test_check_conflicts_single(self):
        """Test detecting a single conflict."""
        manager = CollaborationManager()

        manager.create_boundary("agent1", "Work 1", {"shared.py"})
        manager.create_boundary("agent2", "Work 2", {"shared.py"})

        conflicts = manager.check_conflicts()

        self.assertEqual(len(conflicts), 1)
        self.assertIn("shared.py", conflicts[0].files_affected)
        self.assertIn("agent1", conflicts[0].agents_involved)
        self.assertIn("agent2", conflicts[0].agents_involved)

    def test_check_conflicts_multiple_files(self):
        """Test detecting conflicts with multiple overlapping files."""
        manager = CollaborationManager()

        files1 = {"file1.py", "shared.py", "common.py"}
        files2 = {"file2.py", "shared.py", "common.py"}

        manager.create_boundary("agent1", "Work 1", files1)
        manager.create_boundary("agent2", "Work 2", files2)

        conflicts = manager.check_conflicts()

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(conflicts[0].files_affected), 2)
        self.assertIn("shared.py", conflicts[0].files_affected)
        self.assertIn("common.py", conflicts[0].files_affected)

    def test_check_conflicts_three_agents(self):
        """Test detecting conflicts among three agents."""
        manager = CollaborationManager()

        manager.create_boundary("agent1", "Work 1", {"shared.py"})
        manager.create_boundary("agent2", "Work 2", {"shared.py"})
        manager.create_boundary("agent3", "Work 3", {"other.py"})

        conflicts = manager.check_conflicts()

        # Should be 1 conflict (agent1 vs agent2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0].agents_involved), {"agent1", "agent2"})

    def test_create_handoff(self):
        """Test creating handoff document."""
        manager = CollaborationManager()

        handoff = manager.create_handoff(
            task="Implement caching",
            status="70% complete",
            urgency="medium"
        )

        self.assertEqual(handoff.task_description, "Implement caching")
        self.assertEqual(handoff.status, "70% complete")
        self.assertEqual(handoff.urgency, "medium")

    def test_create_multiple_handoffs(self):
        """Test creating multiple handoffs."""
        manager = CollaborationManager()

        manager.create_handoff("Task 1", "Status 1", "high")
        manager.create_handoff("Task 2", "Status 2", "low")

        summary = manager.get_summary()
        self.assertEqual(summary['handoffs'], 2)

    def test_get_summary_empty(self):
        """Test summary with no activity."""
        manager = CollaborationManager()
        summary = manager.get_summary()

        self.assertEqual(summary['mode'], 'SEMI_SYNCHRONOUS')
        self.assertEqual(summary['status_updates'], 0)
        self.assertEqual(summary['active_blockers'], 0)
        self.assertEqual(summary['hard_blockers'], 0)
        self.assertEqual(summary['disagreements'], 0)
        self.assertEqual(summary['boundaries'], 0)
        self.assertEqual(summary['conflicts'], 0)
        self.assertEqual(summary['handoffs'], 0)

    def test_get_summary_comprehensive(self):
        """Test summary with all activity types."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)

        # Add various activities
        manager.post_status(StatusUpdate(task_name="Task"))
        manager.raise_blocker("Issue 1", blocker_type=BlockerType.HARD)
        manager.raise_blocker("Issue 2", blocker_type=BlockerType.SOFT)
        manager.record_disagreement("I", "C", [], "R", "A")
        manager.create_boundary("agent1", "Work 1", {"file1.py"})
        manager.create_boundary("agent2", "Work 2", {"file1.py"})  # Conflict!
        manager.check_conflicts()  # Detect the conflict
        manager.create_handoff("Task", "Status", "high")

        summary = manager.get_summary()

        self.assertEqual(summary['mode'], 'SYNCHRONOUS')
        self.assertEqual(summary['update_interval_minutes'], 5)
        self.assertEqual(summary['status_updates'], 1)
        self.assertEqual(summary['active_blockers'], 2)
        self.assertEqual(summary['hard_blockers'], 1)
        self.assertEqual(summary['disagreements'], 1)
        self.assertEqual(summary['boundaries'], 2)
        self.assertEqual(summary['conflicts'], 1)
        self.assertEqual(summary['handoffs'], 1)

    def test_get_summary_resolved_conflicts_excluded(self):
        """Test that resolved conflicts are not counted in summary."""
        manager = CollaborationManager()

        manager.create_boundary("agent1", "Work", {"file.py"})
        manager.create_boundary("agent2", "Work", {"file.py"})

        conflicts = manager.check_conflicts()

        # Mark as resolved
        conflicts[0].resolved_at = datetime.now()

        summary = manager.get_summary()
        self.assertEqual(summary['conflicts'], 0)


class TestQuestionBatcher(unittest.TestCase):
    """Test QuestionBatcher class."""

    def test_create(self):
        """Test creating a batcher."""
        batcher = QuestionBatcher()
        self.assertIsNotNone(batcher)

    def test_add_question_minimal(self):
        """Test adding question with minimal args."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("What is the API endpoint?")

        self.assertIsNotNone(q_id)
        self.assertTrue(q_id.startswith("Q-"))

    def test_add_question_full(self):
        """Test adding question with all args."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question(
            question="Which database?",
            context="Choosing between MySQL and PostgreSQL",
            default="PostgreSQL",
            urgency="high",
            category="technical",
            blocking=True,
            related_ids=["Q-000"]
        )

        question = batcher.get_question(q_id)

        self.assertEqual(question.question, "Which database?")
        self.assertEqual(question.context, "Choosing between MySQL and PostgreSQL")
        self.assertEqual(question.default, "PostgreSQL")
        self.assertEqual(question.urgency, "high")
        self.assertEqual(question.category, "technical")
        self.assertTrue(question.blocking)
        self.assertIn("Q-000", question.related_ids)

    def test_add_question_sequential_ids(self):
        """Test that question IDs are sequential."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Question 1")
        q2 = batcher.add_question("Question 2")
        q3 = batcher.add_question("Question 3")

        self.assertEqual(q1, "Q-000")
        self.assertEqual(q2, "Q-001")
        self.assertEqual(q3, "Q-002")

    def test_get_question_exists(self):
        """Test getting an existing question."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("Test question")

        question = batcher.get_question(q_id)

        self.assertIsNotNone(question)
        self.assertEqual(question.question, "Test question")

    def test_get_question_not_exists(self):
        """Test getting non-existent question returns None."""
        batcher = QuestionBatcher()
        question = batcher.get_question("Q-999")

        self.assertIsNone(question)

    def test_get_all_questions(self):
        """Test getting all questions."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")
        batcher.add_question("Q2")
        batcher.add_question("Q3")

        all_questions = batcher.get_all_questions()

        self.assertEqual(len(all_questions), 3)

    def test_get_unanswered_questions(self):
        """Test getting unanswered questions."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")
        q3 = batcher.add_question("Q3")

        # Answer one
        batcher.mark_answered(q2, "Answer 2")

        unanswered = batcher.get_unanswered_questions()

        self.assertEqual(len(unanswered), 2)
        ids = [q.id for q in unanswered]
        self.assertIn(q1, ids)
        self.assertIn(q3, ids)
        self.assertNotIn(q2, ids)

    def test_mark_answered_exists(self):
        """Test marking existing question as answered."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("Test?")

        result = batcher.mark_answered(q_id, "My answer")

        self.assertTrue(result)
        question = batcher.get_question(q_id)
        self.assertTrue(question.answered)
        self.assertEqual(question.response, "My answer")

    def test_mark_answered_not_exists(self):
        """Test marking non-existent question returns False."""
        batcher = QuestionBatcher()

        result = batcher.mark_answered("Q-999", "Answer")

        self.assertFalse(result)

    def test_categorize_questions_empty(self):
        """Test categorizing with no questions."""
        batcher = QuestionBatcher()
        categorized = batcher.categorize_questions()

        self.assertEqual(len(categorized), 0)

    def test_categorize_questions_single_category(self):
        """Test categorizing questions in single category."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1", category="technical")
        batcher.add_question("Q2", category="technical")

        categorized = batcher.categorize_questions()

        self.assertEqual(len(categorized), 1)
        self.assertIn("technical", categorized)
        self.assertEqual(len(categorized["technical"]), 2)

    def test_categorize_questions_multiple_categories(self):
        """Test categorizing questions across categories."""
        batcher = QuestionBatcher()

        batcher.add_question("Tech Q", category="technical")
        batcher.add_question("Design Q", category="design")
        batcher.add_question("Approval Q", category="approval")

        categorized = batcher.categorize_questions()

        self.assertEqual(len(categorized), 3)
        self.assertIn("technical", categorized)
        self.assertIn("design", categorized)
        self.assertIn("approval", categorized)

    def test_categorize_questions_excludes_answered(self):
        """Test that answered questions are excluded."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Q1", category="technical")
        q2 = batcher.add_question("Q2", category="technical")

        batcher.mark_answered(q1, "Answer")

        categorized = batcher.categorize_questions()

        self.assertEqual(len(categorized["technical"]), 1)

    def test_categorize_questions_sorts_by_blocking(self):
        """Test that blocking questions come first."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Non-blocking", blocking=False, category="technical")
        q2 = batcher.add_question("Blocking", blocking=True, category="technical")

        categorized = batcher.categorize_questions()

        # Blocking should be first
        self.assertEqual(categorized["technical"][0].id, q2)
        self.assertEqual(categorized["technical"][1].id, q1)

    def test_categorize_questions_sorts_by_urgency(self):
        """Test that questions are sorted by urgency within category."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Low", urgency="low", category="technical")
        q2 = batcher.add_question("Critical", urgency="critical", category="technical")
        q3 = batcher.add_question("High", urgency="high", category="technical")
        q4 = batcher.add_question("Medium", urgency="medium", category="technical")

        categorized = batcher.categorize_questions()
        questions = categorized["technical"]

        # Should be ordered: critical, high, medium, low
        self.assertEqual(questions[0].urgency, "critical")
        self.assertEqual(questions[1].urgency, "high")
        self.assertEqual(questions[2].urgency, "medium")
        self.assertEqual(questions[3].urgency, "low")

    def test_categorize_questions_blocking_before_urgency(self):
        """Test that blocking status takes priority over urgency."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("High non-blocking", urgency="high", blocking=False, category="tech")
        q2 = batcher.add_question("Low blocking", urgency="low", blocking=True, category="tech")

        categorized = batcher.categorize_questions()

        # Blocking (even low urgency) should come before non-blocking
        self.assertEqual(categorized["tech"][0].id, q2)
        self.assertEqual(categorized["tech"][1].id, q1)

    def test_generate_batch_empty(self):
        """Test generating batch when no questions."""
        batcher = QuestionBatcher()
        md = batcher.generate_batch()

        self.assertIn("No pending questions", md)

    def test_generate_batch_basic(self):
        """Test generating batch with questions."""
        batcher = QuestionBatcher()

        batcher.add_question("What API version?")
        batcher.add_question("Use caching?")

        md = batcher.generate_batch()

        self.assertIn("## Question Request", md)
        self.assertIn("2 question(s)", md)
        self.assertIn("What API version?", md)
        self.assertIn("Use caching?", md)

    def test_generate_batch_with_blocking(self):
        """Test batch highlights blocking questions."""
        batcher = QuestionBatcher()

        batcher.add_question("Blocking Q", blocking=True)
        batcher.add_question("Normal Q", blocking=False)

        md = batcher.generate_batch()

        self.assertIn("URGENT", md)
        self.assertIn("1 blocking question(s)", md)

    def test_generate_batch_with_defaults(self):
        """Test batch shows default values."""
        batcher = QuestionBatcher()

        batcher.add_question("Which DB?", default="PostgreSQL")

        md = batcher.generate_batch()

        self.assertIn("Default if no response:", md)
        self.assertIn("PostgreSQL", md)

    def test_generate_batch_with_context(self):
        """Test batch includes context."""
        batcher = QuestionBatcher()

        batcher.add_question("Q", context="We need to decide this now")

        md = batcher.generate_batch()

        self.assertIn("Context:", md)
        self.assertIn("We need to decide this now", md)

    def test_generate_batch_with_related(self):
        """Test batch shows related questions."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2", related_ids=[q1])

        md = batcher.generate_batch()

        self.assertIn("Related to:", md)
        self.assertIn(q1, md)

    def test_generate_batch_urgency_markers(self):
        """Test batch shows urgency markers."""
        batcher = QuestionBatcher()

        batcher.add_question("Critical Q", urgency="critical")
        batcher.add_question("High Q", urgency="high")

        md = batcher.generate_batch()

        # Critical should have warning emoji
        self.assertIn("⚠️", md)
        # High should have up arrow
        self.assertIn("⬆️", md)

    def test_get_pending_blockers_empty(self):
        """Test getting pending blockers when none exist."""
        batcher = QuestionBatcher()

        batcher.add_question("Non-blocking", blocking=False)

        blockers = batcher.get_pending_blockers()
        self.assertEqual(len(blockers), 0)

    def test_get_pending_blockers(self):
        """Test getting unanswered blocking questions."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Blocker 1", blocking=True)
        q2 = batcher.add_question("Non-blocker", blocking=False)
        q3 = batcher.add_question("Blocker 2", blocking=True)

        blockers = batcher.get_pending_blockers()

        self.assertEqual(len(blockers), 2)
        ids = [q.id for q in blockers]
        self.assertIn(q1, ids)
        self.assertIn(q3, ids)
        self.assertNotIn(q2, ids)

    def test_get_pending_blockers_excludes_answered(self):
        """Test that answered blockers are not returned."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Blocker 1", blocking=True)
        q2 = batcher.add_question("Blocker 2", blocking=True)

        batcher.mark_answered(q1, "Answer")

        blockers = batcher.get_pending_blockers()

        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].id, q2)

    def test_process_responses_empty(self):
        """Test processing empty response."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")

        result = batcher.process_responses("")

        self.assertEqual(len(result['matched']), 0)
        self.assertEqual(len(result['unanswered_questions']), 1)

    def test_process_responses_simple_format(self):
        """Test processing Q-NNN: Answer format."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")

        response = """
        Q-000: Answer to first question
        Q-001: Answer to second question
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 2)
        self.assertEqual(result['matched']['Q-000'], "Answer to first question")
        self.assertEqual(result['matched']['Q-001'], "Answer to second question")
        self.assertEqual(len(result['unanswered_questions']), 0)

    def test_process_responses_numeric_format(self):
        """Test processing numeric format (1: Answer)."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")
        batcher.add_question("Q2")

        response = """
        1: First answer
        2: Second answer
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 2)
        self.assertEqual(result['matched']['Q-000'], "First answer")
        self.assertEqual(result['matched']['Q-001'], "Second answer")

    def test_process_responses_multiline_format(self):
        """Test processing Q-NNN on one line, answer on next."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")

        response = """
        Q-000
        This is the answer on the next line
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 1)
        self.assertEqual(result['matched']['Q-000'], "This is the answer on the next line")

    def test_process_responses_with_markdown(self):
        """Test processing skips markdown artifacts."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")

        response = """
        ```
        Q-000: Answer here
        ```
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 1)
        self.assertEqual(result['matched']['Q-000'], "Answer here")

    def test_process_responses_partial(self):
        """Test processing partial responses."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")
        batcher.add_question("Q2")
        batcher.add_question("Q3")

        response = """
        Q-000: Answer for first
        Q-002: Answer for third
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 2)
        self.assertEqual(len(result['unanswered_questions']), 1)
        self.assertIn('Q-001', result['unanswered_questions'])

    def test_process_responses_invalid_ids(self):
        """Test that invalid question IDs are unparsed."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")

        response = """
        Q-000: Valid answer
        Q-999: Invalid question ID
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 1)
        self.assertEqual(len(result['unparsed_lines']), 1)
        self.assertIn("Q-999: Invalid question ID", result['unparsed_lines'][0])

    def test_process_responses_updates_question_state(self):
        """Test that processing updates question answered state."""
        batcher = QuestionBatcher()

        q_id = batcher.add_question("Test question")

        response = f"{q_id}: My answer"
        batcher.process_responses(response)

        question = batcher.get_question(q_id)
        self.assertTrue(question.answered)
        self.assertEqual(question.response, "My answer")

    def test_process_responses_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")

        response = """

        Q-000: Answer


        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 1)
        self.assertEqual(len(result['unparsed_lines']), 0)

    def test_process_responses_skips_headers(self):
        """Test that markdown headers are skipped."""
        batcher = QuestionBatcher()

        batcher.add_question("Q1")

        response = """
        ## My Responses

        Q-000: Answer here

        ### Additional Notes
        """

        result = batcher.process_responses(response)

        self.assertEqual(len(result['matched']), 1)
        # Headers should not be in unparsed
        for line in result['unparsed_lines']:
            self.assertNotIn('#', line)


if __name__ == '__main__':
    unittest.main()
