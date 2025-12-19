"""
Extended unit tests for Collaboration module.
"""

import pytest
from cortical.reasoning import (
    CollaborationMode,
    BlockerType,
    StatusUpdate,
    CollaborationManager,
)
from cortical.reasoning.collaboration import (
    ConflictType,
    Blocker,
    DisagreementRecord,
    ParallelWorkBoundary,
    ConflictEvent,
    ActiveWorkHandoff,
    ParallelCoordinator,
    QuestionBatcher,
)


class TestCollaborationMode:
    """Tests for CollaborationMode enum."""

    def test_all_modes_exist(self):
        """Test all collaboration modes exist."""
        assert CollaborationMode.SYNCHRONOUS is not None
        assert CollaborationMode.ASYNCHRONOUS is not None
        assert CollaborationMode.SEMI_SYNCHRONOUS is not None

    def test_modes_are_distinct(self):
        """Test modes have distinct values."""
        assert CollaborationMode.SYNCHRONOUS != CollaborationMode.ASYNCHRONOUS
        assert CollaborationMode.ASYNCHRONOUS != CollaborationMode.SEMI_SYNCHRONOUS
        assert CollaborationMode.SYNCHRONOUS != CollaborationMode.SEMI_SYNCHRONOUS


class TestBlockerType:
    """Tests for BlockerType enum."""

    def test_all_types_exist(self):
        """Test all blocker types exist."""
        assert BlockerType.HARD is not None
        assert BlockerType.SOFT is not None
        assert BlockerType.INFO is not None

    def test_types_are_distinct(self):
        """Test blocker types have distinct values."""
        assert BlockerType.HARD != BlockerType.SOFT
        assert BlockerType.SOFT != BlockerType.INFO
        assert BlockerType.HARD != BlockerType.INFO


class TestConflictType:
    """Tests for ConflictType enum."""

    def test_all_types_exist(self):
        """Test all conflict types exist."""
        assert ConflictType.FILE_CONFLICT is not None
        assert ConflictType.LOGIC_CONFLICT is not None
        assert ConflictType.DEPENDENCY_CONFLICT is not None
        assert ConflictType.SCOPE_OVERLAP is not None


class TestStatusUpdate:
    """Tests for StatusUpdate dataclass."""

    def test_create_basic(self):
        """Test creating a basic status update."""
        update = StatusUpdate(
            task_name="Test Task",
            progress_percent=50,
            current_phase="testing"
        )

        assert update.task_name == "Test Task"
        assert update.progress_percent == 50
        assert update.current_phase == "testing"

    def test_default_values(self):
        """Test default values."""
        update = StatusUpdate(task_name="Task")

        assert update.progress_percent == 0
        assert update.current_phase == ""
        assert update.eta_minutes is None
        assert update.completed_items == []
        assert update.in_progress_items == []
        assert update.blockers == []
        assert update.concerns == []
        assert update.needs_from_human == []

    def test_to_markdown_basic(self):
        """Test markdown generation for basic update."""
        update = StatusUpdate(
            task_name="Implement Feature",
            progress_percent=25,
            current_phase="Planning"
        )

        md = update.to_markdown()
        assert "## Status: Implement Feature" in md
        assert "25%" in md
        assert "Planning" in md

    def test_to_markdown_with_eta(self):
        """Test markdown with ETA."""
        update = StatusUpdate(
            task_name="Task",
            progress_percent=50,
            current_phase="Working",
            eta_minutes=30
        )

        md = update.to_markdown()
        assert "30 minutes" in md

    def test_to_markdown_with_completed_items(self):
        """Test markdown with completed items."""
        update = StatusUpdate(
            task_name="Task",
            completed_items=["Item 1", "Item 2"]
        )

        md = update.to_markdown()
        assert "**Completed:**" in md
        assert "[x] Item 1" in md
        assert "[x] Item 2" in md

    def test_to_markdown_with_in_progress_items(self):
        """Test markdown with in-progress items."""
        update = StatusUpdate(
            task_name="Task",
            in_progress_items=["Working on A", "Working on B"]
        )

        md = update.to_markdown()
        assert "**In progress:**" in md
        assert "[ ] Working on A" in md
        assert "[ ] Working on B" in md

    def test_to_markdown_with_blockers(self):
        """Test markdown with blockers."""
        update = StatusUpdate(
            task_name="Task",
            blockers=["Need API key", "Waiting for approval"]
        )

        md = update.to_markdown()
        assert "Need API key" in md
        assert "Waiting for approval" in md

    def test_to_markdown_with_no_blockers(self):
        """Test markdown shows None when no blockers."""
        update = StatusUpdate(task_name="Task")

        md = update.to_markdown()
        assert "**Blockers:** None" in md

    def test_to_markdown_with_concerns(self):
        """Test markdown with concerns."""
        update = StatusUpdate(
            task_name="Task",
            concerns=["Performance might be slow"]
        )

        md = update.to_markdown()
        assert "Performance might be slow" in md

    def test_to_markdown_with_needs(self):
        """Test markdown with needs from human."""
        update = StatusUpdate(
            task_name="Task",
            needs_from_human=["Review the design"]
        )

        md = update.to_markdown()
        assert "Review the design" in md


class TestBlocker:
    """Tests for Blocker dataclass."""

    def test_create_basic(self):
        """Test creating a basic blocker."""
        blocker = Blocker(
            description="Cannot proceed",
            blocker_type=BlockerType.HARD,
            resolution_needed="Need API key"
        )

        assert blocker.description == "Cannot proceed"
        assert blocker.blocker_type == BlockerType.HARD
        assert blocker.resolution_needed == "Need API key"
        assert not blocker.resolved

    def test_default_values(self):
        """Test default values."""
        blocker = Blocker()

        assert blocker.id is not None
        assert blocker.description == ""
        assert blocker.blocker_type == BlockerType.SOFT
        assert blocker.resolved is False
        assert blocker.resolution is None
        assert blocker.resolved_at is None

    def test_resolve(self):
        """Test resolving a blocker."""
        blocker = Blocker(description="Issue")
        blocker.resolve("Fixed by updating config")

        assert blocker.resolved is True
        assert blocker.resolution == "Fixed by updating config"
        assert blocker.resolved_at is not None

    def test_with_workaround(self):
        """Test blocker with workaround."""
        blocker = Blocker(
            description="API rate limited",
            blocker_type=BlockerType.SOFT,
            workaround="Use cached data"
        )

        assert blocker.workaround == "Use cached data"


class TestDisagreementRecord:
    """Tests for DisagreementRecord dataclass."""

    def test_create_basic(self):
        """Test creating a disagreement record."""
        record = DisagreementRecord(
            instruction_given="Use eval() for parsing",
            concern_raised="Security vulnerability",
            evidence=["OWASP Top 10", "Security audit"],
            risk_if_proceed="Remote code execution",
            alternative_suggested="Use ast.literal_eval()"
        )

        assert record.instruction_given == "Use eval() for parsing"
        assert record.concern_raised == "Security vulnerability"
        assert len(record.evidence) == 2
        assert record.risk_if_proceed == "Remote code execution"
        assert record.alternative_suggested == "Use ast.literal_eval()"

    def test_default_values(self):
        """Test default values."""
        record = DisagreementRecord()

        assert record.id is not None
        assert record.instruction_given == ""
        assert record.evidence == []
        assert record.human_decision is None
        assert record.outcome is None

    def test_to_markdown(self):
        """Test markdown generation."""
        record = DisagreementRecord(
            instruction_given="Use eval()",
            concern_raised="Security risk",
            evidence=["OWASP", "Best practices"],
            risk_if_proceed="Code injection",
            alternative_suggested="Use safe parser"
        )

        md = record.to_markdown()
        assert "## Respectful Disagreement" in md
        assert "Use eval()" in md
        assert "Security risk" in md
        assert "OWASP" in md
        assert "Code injection" in md
        assert "Use safe parser" in md
        assert "Your call:" in md


class TestParallelWorkBoundary:
    """Tests for ParallelWorkBoundary dataclass."""

    def test_create_basic(self):
        """Test creating a boundary."""
        boundary = ParallelWorkBoundary(
            agent_id="agent1",
            scope_description="Frontend work"
        )

        assert boundary.agent_id == "agent1"
        assert boundary.scope_description == "Frontend work"
        assert len(boundary.files_owned) == 0

    def test_add_file_write_access(self):
        """Test adding file with write access."""
        boundary = ParallelWorkBoundary(
            agent_id="agent1",
            scope_description="Frontend"
        )

        boundary.add_file("src/app.js", write_access=True)
        assert "src/app.js" in boundary.files_owned

    def test_add_file_read_only(self):
        """Test adding file as read-only."""
        boundary = ParallelWorkBoundary(
            agent_id="agent1",
            scope_description="Frontend"
        )

        boundary.add_file("src/config.js", write_access=False)
        assert "src/config.js" in boundary.files_read_only
        assert "src/config.js" not in boundary.files_owned

    def test_can_modify(self):
        """Test checking modify permission."""
        boundary = ParallelWorkBoundary(
            agent_id="agent1",
            scope_description="Frontend",
            files_owned={"src/app.js", "src/utils.js"}
        )

        assert boundary.can_modify("src/app.js") is True
        assert boundary.can_modify("src/unknown.js") is False

    def test_conflicts_with(self):
        """Test detecting conflicts with another boundary."""
        boundary1 = ParallelWorkBoundary(
            agent_id="agent1",
            scope_description="Frontend",
            files_owned={"src/app.js", "src/shared.js"}
        )

        boundary2 = ParallelWorkBoundary(
            agent_id="agent2",
            scope_description="Backend",
            files_owned={"src/api.js", "src/shared.js"}
        )

        conflicts = boundary1.conflicts_with(boundary2)
        assert "src/shared.js" in conflicts

    def test_no_conflicts(self):
        """Test when no conflicts."""
        boundary1 = ParallelWorkBoundary(
            agent_id="agent1",
            scope_description="Frontend",
            files_owned={"src/app.js"}
        )

        boundary2 = ParallelWorkBoundary(
            agent_id="agent2",
            scope_description="Backend",
            files_owned={"src/api.js"}
        )

        conflicts = boundary1.conflicts_with(boundary2)
        assert len(conflicts) == 0


class TestConflictEvent:
    """Tests for ConflictEvent dataclass."""

    def test_create_basic(self):
        """Test creating a conflict event."""
        event = ConflictEvent(
            conflict_type=ConflictType.FILE_CONFLICT,
            agents_involved=["agent1", "agent2"],
            description="Both modified the same file",
            files_affected=["src/shared.js"]
        )

        assert event.conflict_type == ConflictType.FILE_CONFLICT
        assert "agent1" in event.agents_involved
        assert "src/shared.js" in event.files_affected
        assert event.escalated is False

    def test_default_values(self):
        """Test default values."""
        event = ConflictEvent()

        assert event.id is not None
        assert event.conflict_type == ConflictType.FILE_CONFLICT
        assert event.agents_involved == []
        assert event.resolution is None


class TestActiveWorkHandoff:
    """Tests for ActiveWorkHandoff dataclass."""

    def test_create_basic(self):
        """Test creating a handoff document."""
        handoff = ActiveWorkHandoff(
            task_description="Implement auth",
            status="In progress",
            urgency="high"
        )

        assert handoff.task_description == "Implement auth"
        assert handoff.status == "In progress"
        assert handoff.urgency == "high"

    def test_default_values(self):
        """Test default values."""
        handoff = ActiveWorkHandoff(
            task_description="Task",
            status="Status",
            urgency="medium"
        )

        assert handoff.files_working == []
        assert handoff.files_in_progress == {}
        assert handoff.known_issues == []
        assert handoff.key_decisions == {}
        assert handoff.gotchas == []
        assert handoff.open_questions == []

    def test_to_markdown(self):
        """Test markdown generation."""
        handoff = ActiveWorkHandoff(
            task_description="Implement OAuth",
            status="50% complete",
            urgency="high",
            files_working=["auth.py", "tokens.py"],
            files_in_progress={"routes.py": "Need to add logout"},
            known_issues=["Session timeout not handled"],
            key_decisions={"Token storage": "Use Redis"},
            gotchas=["Must refresh before expiry"],
            files_to_read_first=["auth.py", "README.md"],
            immediate_next_steps=["Add logout route", "Test tokens"],
            open_questions=["Should we support OAuth1?"],
            verification_command="pytest tests/auth/"
        )

        md = handoff.to_markdown()
        assert "## Active Work Handoff" in md
        assert "Implement OAuth" in md
        assert "50% complete" in md
        assert "high" in md
        assert "auth.py" in md
        assert "routes.py: Need to add logout" in md
        assert "Session timeout not handled" in md
        assert "Token storage: Use Redis" in md
        assert "Must refresh before expiry" in md
        assert "README.md" in md
        assert "Add logout route" in md
        assert "Should we support OAuth1?" in md
        assert "pytest tests/auth/" in md


class TestCollaborationManager:
    """Tests for CollaborationManager class."""

    def test_create_default(self):
        """Test creating with default mode."""
        manager = CollaborationManager()
        assert manager.mode == CollaborationMode.SEMI_SYNCHRONOUS

    def test_create_with_mode(self):
        """Test creating with specific mode."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)
        assert manager.mode == CollaborationMode.SYNCHRONOUS

    def test_set_mode(self):
        """Test setting mode."""
        manager = CollaborationManager()
        manager.mode = CollaborationMode.ASYNCHRONOUS
        assert manager.mode == CollaborationMode.ASYNCHRONOUS

    def test_update_interval_synchronous(self):
        """Test update interval for synchronous mode."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)
        assert manager.get_update_interval() == 5

    def test_update_interval_asynchronous(self):
        """Test update interval for asynchronous mode."""
        manager = CollaborationManager(mode=CollaborationMode.ASYNCHRONOUS)
        assert manager.get_update_interval() == 60

    def test_update_interval_semi_synchronous(self):
        """Test update interval for semi-synchronous mode."""
        manager = CollaborationManager(mode=CollaborationMode.SEMI_SYNCHRONOUS)
        assert manager.get_update_interval() == 15

    def test_post_status(self):
        """Test posting status updates."""
        manager = CollaborationManager()
        update = StatusUpdate(task_name="Test", progress_percent=25)

        manager.post_status(update)

        summary = manager.get_summary()
        assert summary['status_updates'] == 1

    def test_raise_blocker(self):
        """Test raising a blocker."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker(
            "Cannot proceed",
            blocker_type=BlockerType.HARD,
            resolution_needed="Need approval"
        )

        assert blocker.description == "Cannot proceed"
        assert blocker.blocker_type == BlockerType.HARD

    def test_raise_blocker_with_context(self):
        """Test raising blocker with context."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker(
            "Issue",
            context={"file": "main.py", "line": 42}
        )

        assert blocker.context["file"] == "main.py"
        assert blocker.context["line"] == 42

    def test_resolve_blocker(self):
        """Test resolving a blocker."""
        manager = CollaborationManager()
        blocker = manager.raise_blocker("Issue")

        manager.resolve_blocker(blocker.id, "Fixed")

        active = manager.get_active_blockers()
        assert len(active) == 0

    def test_resolve_nonexistent_blocker(self):
        """Test resolving non-existent blocker (no error)."""
        manager = CollaborationManager()
        manager.resolve_blocker("nonexistent", "Fixed")  # Should not raise

    def test_get_active_blockers(self):
        """Test getting active blockers."""
        manager = CollaborationManager()
        manager.raise_blocker("Issue 1")
        blocker2 = manager.raise_blocker("Issue 2")

        manager.resolve_blocker(blocker2.id, "Fixed")

        active = manager.get_active_blockers()
        assert len(active) == 1
        assert active[0].description == "Issue 1"

    def test_get_hard_blockers(self):
        """Test getting hard blockers."""
        manager = CollaborationManager()
        manager.raise_blocker("Soft issue", blocker_type=BlockerType.SOFT)
        manager.raise_blocker("Hard issue", blocker_type=BlockerType.HARD)

        hard = manager.get_hard_blockers()
        assert len(hard) == 1
        assert hard[0].description == "Hard issue"

    def test_record_disagreement(self):
        """Test recording a disagreement."""
        manager = CollaborationManager()
        record = manager.record_disagreement(
            instruction="Do X",
            concern="X is risky",
            evidence=["Evidence 1"],
            risk="Could break",
            alternative="Do Y instead"
        )

        assert record.instruction_given == "Do X"
        assert record.concern_raised == "X is risky"

    def test_create_boundary(self):
        """Test creating work boundary."""
        manager = CollaborationManager()
        boundary = manager.create_boundary(
            agent_id="agent1",
            scope="Frontend work",
            files={"src/app.js"}
        )

        assert boundary.agent_id == "agent1"
        assert "src/app.js" in boundary.files_owned

    def test_create_boundary_no_files(self):
        """Test creating boundary with no initial files."""
        manager = CollaborationManager()
        boundary = manager.create_boundary("agent1", "Work")

        assert len(boundary.files_owned) == 0

    def test_check_conflicts(self):
        """Test checking for conflicts."""
        manager = CollaborationManager()
        manager.create_boundary("agent1", "Work 1", {"shared.js"})
        manager.create_boundary("agent2", "Work 2", {"shared.js"})

        conflicts = manager.check_conflicts()
        assert len(conflicts) == 1
        assert "shared.js" in conflicts[0].files_affected

    def test_check_conflicts_none(self):
        """Test when no conflicts."""
        manager = CollaborationManager()
        manager.create_boundary("agent1", "Work 1", {"file1.js"})
        manager.create_boundary("agent2", "Work 2", {"file2.js"})

        conflicts = manager.check_conflicts()
        assert len(conflicts) == 0

    def test_create_handoff(self):
        """Test creating handoff document."""
        manager = CollaborationManager()
        handoff = manager.create_handoff(
            task="Implement feature",
            status="In progress",
            urgency="high"
        )

        assert handoff.task_description == "Implement feature"
        assert handoff.status == "In progress"
        assert handoff.urgency == "high"

    def test_get_summary(self):
        """Test getting collaboration summary."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)

        manager.post_status(StatusUpdate(task_name="Task"))
        manager.raise_blocker("Issue", blocker_type=BlockerType.HARD)
        manager.record_disagreement("Instr", "Concern", [], "Risk", "Alt")
        manager.create_boundary("agent1", "Work")
        manager.create_handoff("Task", "Status", "medium")

        summary = manager.get_summary()

        assert summary['mode'] == 'SYNCHRONOUS'
        assert summary['update_interval_minutes'] == 5
        assert summary['status_updates'] == 1
        assert summary['active_blockers'] == 1
        assert summary['hard_blockers'] == 1
        assert summary['disagreements'] == 1
        assert summary['boundaries'] == 1
        assert summary['handoffs'] == 1


class TestParallelCoordinator:
    """Tests for ParallelCoordinator stub class."""

    def test_spawn_parallel(self):
        """Test spawning parallel agents (stub)."""
        coordinator = ParallelCoordinator()
        result = coordinator.spawn_parallel(
            tasks=[{"goal": "Task 1"}, {"goal": "Task 2"}],
            boundaries=[]
        )

        assert 'agents' in result
        assert len(result['agents']) == 2
        assert 'note' in result

    def test_wait_for_completion(self):
        """Test waiting for completion (stub)."""
        coordinator = ParallelCoordinator()
        result = coordinator.wait_for_completion(
            agent_ids=["agent_0", "agent_1"],
            timeout_minutes=30
        )

        assert 'completed' in result
        assert result['completed'] == ["agent_0", "agent_1"]
        assert result['failed'] == []

    def test_merge_results(self):
        """Test merging results (stub)."""
        coordinator = ParallelCoordinator()
        result = coordinator.merge_results(["agent_0"])

        assert 'merged_files' in result
        assert 'conflicts' in result
        assert result['commit_ready'] is True


class TestQuestionBatcher:
    """Tests for QuestionBatcher stub class."""

    def test_create(self):
        """Test creating batcher."""
        batcher = QuestionBatcher()
        assert batcher is not None

    def test_add_question(self):
        """Test adding a question."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question(
            "What API version?",
            context="Building integration",
            default="v2",
            urgency="high"
        )

        assert q_id.startswith("Q-")

    def test_add_multiple_questions(self):
        """Test adding multiple questions."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Question 1")
        q2 = batcher.add_question("Question 2")

        assert q1 != q2

    def test_generate_batch_empty(self):
        """Test generating batch when empty."""
        batcher = QuestionBatcher()
        md = batcher.generate_batch()

        assert "No pending questions" in md

    def test_generate_batch_with_questions(self):
        """Test generating batch with questions."""
        batcher = QuestionBatcher()
        batcher.add_question("What API version?", default="v2")
        batcher.add_question("Use caching?")

        md = batcher.generate_batch()

        assert "## Question Request" in md
        assert "2 questions" in md
        assert "What API version?" in md
        assert "Use caching?" in md
        assert "Default if no response: v2" in md
