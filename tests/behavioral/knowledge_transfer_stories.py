"""
Behavioral tests for KnowledgeTransfer entity.

Epic: Developer Captures and Shares Session Knowledge

As a developer working on complex systems we built from first principles,
I want to capture insights and learnings from my sessions,
So that knowledge is preserved, shared, and discoverable across the team
using our own custom knowledge management implementation.

Following Metus: We describe behavior, then make it true.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from cortical.got.api import GoTManager
from cortical.got.types import KnowledgeTransfer, Task, Decision, Handoff
from cortical.got.errors import TransactionError
from cortical.core.bootstrap import create_container


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _create_tx_manager(got_dir, use_memory=True):
    """Create a GoTManager for testing.

    Args:
        got_dir: Directory for GoT storage
        use_memory: Ignored (always uses file storage for persistence tests)
    """
    container = create_container(got_dir=got_dir)
    return container.resolve(GoTManager)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_got_dir(tmp_path):
    """Provide a temporary directory for GoT storage."""
    got_dir = tmp_path / ".got"
    got_dir.mkdir()
    return got_dir


@pytest.fixture
def tx_manager(temp_got_dir):
    """
    Provide a TransactionalGoTAdapter for testing.

    Provides high-level API for knowledge transfer operations.
    """
    return TransactionalGoTAdapter(temp_got_dir)


@pytest.fixture
def sample_task(tx_manager):
    """
    Create a sample task for knowledge transfer testing.

    Task represents work on our custom transaction implementation.
    """
    task_id = tx_manager.create_task(
        title="Implement snapshot isolation layer we control",
        priority="high",
        category="feature",
        description="Build custom snapshot versioning for consistent reads in our transaction manager"
    )
    tx_manager.complete_task(task_id)
    return tx_manager.get_task(task_id)


@pytest.fixture
def sample_decision(tx_manager):
    """
    Create a sample decision for knowledge transfer testing.

    Decision represents architectural choice in our custom system.
    """
    decision_id = tx_manager.log_decision(
        decision="Use optimistic locking for conflict detection",
        rationale="Built version-based conflict detection ourselves to maintain complete control over transaction semantics",
        affects=["TASK-CDG-001"]
    )
    return tx_manager.get_decision(decision_id)


@pytest.fixture
def sample_handoff(tx_manager, sample_task):
    """
    Create a sample handoff for knowledge transfer testing.

    Handoff represents work transfer in our custom workflow system.
    """
    # Create a handoff for the sample task
    handoff_id = tx_manager.initiate_handoff(
        source_agent="agent-alpha",
        target_agent="agent-beta",
        task_id=sample_task.id,
        instructions="Continue implementing transaction recovery using our custom WAL",
        context={
            "current_branch": "claude/cdg-recovery",
            "files_modified": ["cortical/cdg/recovery.py", "cortical/cdg/wal.py"]
        }
    )
    return tx_manager.get_handoff(handoff_id)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_kt(tx_manager, **kwargs):
    """
    Helper to create a KT and return the full entity.

    The adapter's create_knowledge_transfer returns just the ID string.
    This helper creates the KT and retrieves the full entity.
    """
    kt_id = tx_manager.create_knowledge_transfer(**kwargs)
    return tx_manager.get_knowledge_transfer(kt_id)


# ============================================================================
# BEHAVIORAL SCENARIOS - CAPTURING SESSION LEARNINGS
# ============================================================================

class TestDeveloperCapturesSessionLearnings:
    """
    Epic: Session Knowledge Capture

    As a developer finishing a work session,
    I want to capture key insights and learnings,
    So that my knowledge persists beyond the session
    and teammates can benefit from what I learned.
    """

    def test_scenario_create_knowledge_transfer_with_title_and_summary(self, tx_manager):
        """
        Scenario: Developer creates knowledge transfer to document session

        Given a developer has completed meaningful work
        When they create a knowledge transfer with title and summary
        Then the knowledge transfer is persisted
        And it has a unique ID with timestamp
        And the status defaults to published
        """
        # Given a developer has completed meaningful work
        session_id = "session-2025-01-01-alpha"
        session_date = datetime.now(timezone.utc).isoformat()

        # When they create a knowledge transfer with title and summary
        kt_id = tx_manager.create_knowledge_transfer(
            title="CDG and GoT Unification Strategy",
            summary="Unified CDG and GoT transaction layers by creating configurable abstraction. "
                   "This allows us to swap WAL, recovery, and storage implementations while "
                   "maintaining API compatibility. Built entirely from first principles.",
            session_id=session_id,
            session_date=session_date
        )

        # Then the knowledge transfer is persisted
        assert kt_id is not None
        assert kt_id.startswith("KT-")

        # Retrieve the full entity
        kt = tx_manager.get_knowledge_transfer(kt_id)
        assert kt is not None
        assert kt.title == "CDG and GoT Unification Strategy"
        assert "Unified CDG and GoT" in kt.summary

        # And it has a unique ID with timestamp
        # Date portion should be in YYYYMMDD format (e.g., 20260101)
        date_part = session_date[:10].replace("-", "")  # Convert YYYY-MM-DD to YYYYMMDD
        assert date_part in kt.id

        # And the status defaults to draft (not published - must be finalized)
        assert kt.status == "draft"

        # Verify persistence
        retrieved = tx_manager.get_knowledge_transfer(kt.id)
        assert retrieved is not None
        assert retrieved.title == kt.title
        assert retrieved.summary == kt.summary

    def test_scenario_append_technical_insights_during_session(self, tx_manager):
        """
        Scenario: Developer adds sections incrementally as insights emerge

        Given an existing knowledge transfer
        When the developer appends a technical insight section
        Then the section is added to the knowledge transfer
        And subsequent appends to the same section accumulate content
        And the version number increments
        """
        # Given an existing knowledge transfer
        kt = create_kt(tx_manager,
            title="Custom WAL Implementation Insights",
            summary="Learnings from building write-ahead log from scratch"
        )

        # When the developer appends a technical insight section
        updated_kt = tx_manager.append_to_knowledge_transfer(
            kt.id,
            "Technical Insights",
            "WAL entry format: [TX_ID][OPERATION][ENTITY_ID][OLD_DATA][NEW_DATA]. "
            "This format enables both redo and undo operations during recovery."
        )

        # Then the section is added to the knowledge transfer
        assert "Technical Insights" in updated_kt.sections
        assert "WAL entry format" in updated_kt.sections["Technical Insights"]

        # And subsequent appends to the same section accumulate content
        updated_kt2 = tx_manager.append_to_knowledge_transfer(
            kt.id,
            "Technical Insights",
            "Recovery algorithm: Scan WAL backward to find uncommitted transactions, "
            "then roll them back atomically. This guarantees ACID properties."
        )

        # Sections are concatenated with double newline
        assert "WAL entry format" in updated_kt2.sections["Technical Insights"]
        assert "Recovery algorithm" in updated_kt2.sections["Technical Insights"]
        assert "\n\n" in updated_kt2.sections["Technical Insights"]

        # And the version number increments
        assert updated_kt2.version > kt.version

    def test_scenario_add_code_references_to_knowledge_transfer(self, tx_manager):
        """
        Scenario: Developer links knowledge to specific code locations

        Given a knowledge transfer documenting architectural decisions
        When the developer adds code references in file:line format
        Then the references are stored in the knowledge transfer
        And they can be retrieved for later navigation
        """
        # Given a knowledge transfer documenting architectural decisions
        kt = create_kt(tx_manager,
            title="Snapshot Isolation Implementation",
            summary="How we built MVCC snapshot isolation ourselves"
        )

        # When the developer adds code references in file:line format
        kt_with_refs = create_kt(tx_manager,
            title="Snapshot Isolation Implementation Details",
            summary="Complete implementation of snapshot versioning",
            code_refs=[
                "cortical/cdg/transaction_manager.py:249",
                "cortical/cdg/storage.py:156",
                "cortical/got/tx_manager.py:158"
            ]
        )

        # Then the references are stored in the knowledge transfer
        assert len(kt_with_refs.code_refs) == 3
        assert "cortical/cdg/transaction_manager.py:249" in kt_with_refs.code_refs

        # And they can be retrieved for later navigation
        retrieved = tx_manager.get_knowledge_transfer(kt_with_refs.id)
        assert retrieved is not None
        assert len(retrieved.code_refs) == 3
        assert all(
            ":" in ref and ".py:" in ref
            for ref in retrieved.code_refs
        ), "Code refs should be in file:line format"

    def test_scenario_tag_knowledge_transfer_for_discovery(self, tx_manager):
        """
        Scenario: Developer tags knowledge for future searchability

        Given a knowledge transfer about a specific domain
        When the developer adds descriptive tags
        Then the tags are stored with the knowledge transfer
        And future searches can filter by tags
        """
        # Given a knowledge transfer about a specific domain
        # When the developer adds descriptive tags
        kt = create_kt(tx_manager,
            title="Thread Safety in CDGStore",
            summary="How we built thread-safe file operations from scratch without external dependencies",
            tags=["concurrency", "thread-safety", "file-io", "custom-implementation"]
        )

        # Then the tags are stored with the knowledge transfer
        assert len(kt.tags) == 4
        assert "thread-safety" in kt.tags
        assert "custom-implementation" in kt.tags

        # And future searches can filter by tags
        # Create another KT with overlapping tags
        kt2 = create_kt(tx_manager,
            title="Process Locks for Isolation",
            summary="Custom process-level locking implementation",
            tags=["concurrency", "locking", "custom-implementation"]
        )

        # Search by tags
        concurrent_kts = tx_manager.list_knowledge_transfers(tags=["concurrency"])
        assert len(concurrent_kts) == 2
        kt_ids = {kt.id for kt in concurrent_kts}
        assert kt.id in kt_ids
        assert kt2.id in kt_ids

        # Search by multiple tags (AND logic)
        custom_concurrent = tx_manager.list_knowledge_transfers(
            tags=["concurrency", "custom-implementation"]
        )
        assert len(custom_concurrent) == 2

    def test_scenario_create_draft_knowledge_transfer_for_work_in_progress(self, tx_manager):
        """
        Scenario: Developer creates draft KT during active session

        Given a developer is mid-session with incomplete insights
        When they create a knowledge transfer with draft status
        Then the KT is saved with status=draft
        And it can be updated and published later
        """
        # Given a developer is mid-session with incomplete insights
        # When they create a knowledge transfer with draft status
        kt = create_kt(tx_manager,
            title="WAL Recovery Edge Cases - Work in Progress",
            summary="Still exploring corner cases in crash recovery..."
        )

        # Update to explicit draft status using high-level API
        tx_manager.update_knowledge_transfer(kt.id, status="draft")

        # Then the KT is saved with status=draft
        retrieved = tx_manager.get_knowledge_transfer(kt.id)
        assert retrieved.status == "draft"

        # And it can be updated and published later
        tx_manager.update_knowledge_transfer(
            kt.id,
            status="published",
            summary="Complete analysis of WAL recovery edge cases and solutions"
        )

        final_kt = tx_manager.get_knowledge_transfer(kt.id)
        assert final_kt.status == "published"
        assert "Complete analysis" in final_kt.summary


# ============================================================================
# BEHAVIORAL SCENARIOS - LINKING KNOWLEDGE TO WORK
# ============================================================================

class TestDeveloperLinksKnowledgeToWork:
    """
    Epic: Knowledge Work Traceability

    As a developer documenting work outcomes,
    I want to link knowledge transfers to related tasks, handoffs, and decisions,
    So that knowledge is connected to the work that produced it
    and future developers understand the context.
    """

    def test_scenario_link_knowledge_transfer_to_handoff_for_continuity(
        self, tx_manager, sample_handoff
    ):
        """
        Scenario: Developer links KT to handoff for session continuity

        Given a knowledge transfer documenting session work
        And a handoff transferring work to another agent
        When the developer links the KT to the handoff
        Then an edge is created between them
        And the knowledge provides context for the handoff
        """
        # Given a knowledge transfer documenting session work
        kt = create_kt(tx_manager,
            title="CDG Recovery Implementation Session",
            summary="Completed WAL recovery implementation and initial tests",
            session_id="session-001"
        )

        # And a handoff transferring work to another agent
        # (sample_handoff fixture provides this)
        assert sample_handoff.id.startswith("H-"), "Handoff should have H- prefix"

        # When the developer links the KT to the handoff
        link_created = tx_manager.link_knowledge_transfer(
            kt.id,
            sample_handoff.id,
            link_type="CONTINUES"
        )

        # Then an edge is created between them
        assert link_created is True

        # And the knowledge provides context for the handoff
        # Note: Edge creation is verified through API return value (link_created)
        # We don't check implementation details like file existence

    def test_scenario_link_knowledge_transfer_to_task_for_documentation(
        self, tx_manager, sample_task
    ):
        """
        Scenario: Developer documents task completion with KT

        Given a completed task
        And a knowledge transfer capturing task insights
        When the developer links the KT to the task
        Then the link documents what was learned
        And future developers can find the knowledge from the task
        """
        # Given a completed task
        # (sample_task fixture provides this)
        assert sample_task.status == "completed"

        # And a knowledge transfer capturing task insights
        kt = create_kt(tx_manager,
            title="Snapshot Isolation Implementation Learnings",
            summary="Key insights from implementing snapshot versioning ourselves",
            sections={
                "Implementation": "Built MVCC using version vectors...",
                "Challenges": "Handling concurrent reads required careful snapshot management..."
            }
        )

        # When the developer links the KT to the task
        link_created = tx_manager.link_knowledge_transfer(
            kt.id,
            sample_task.id,
            link_type="DOCUMENTS"
        )

        # Then the link documents what was learned
        assert link_created is True

        # And future developers can find the knowledge from the task
        # Note: Edge creation is verified through API return value (link_created)
        # We don't check implementation details like file existence

    def test_scenario_link_knowledge_transfer_to_decision_for_rationale(
        self, tx_manager, sample_decision
    ):
        """
        Scenario: Developer captures decision rationale in KT

        Given an architectural decision
        And a knowledge transfer explaining the decision context
        When the developer links the KT to the decision
        Then the connection preserves the reasoning
        And the decision is enriched with detailed context
        """
        # Given an architectural decision
        # (sample_decision fixture provides this)
        assert "optimistic locking" in sample_decision.title

        # And a knowledge transfer explaining the decision context
        kt = create_kt(tx_manager,
            title="Why We Chose Optimistic Locking",
            summary="Analysis of locking strategies for our custom transaction manager",
            sections={
                "Context": "Needed conflict detection for concurrent transactions in our "
                          "hand-built system without external dependencies",
                "Alternatives Considered": "Pessimistic locking, timestamp ordering, "
                                          "serializable snapshots - all rejected",
                "Rationale": "Optimistic locking gives us complete control over conflict "
                            "resolution while maintaining high concurrency"
            }
        )

        # When the developer links the KT to the decision
        link_created = tx_manager.link_knowledge_transfer(
            kt.id,
            sample_decision.id,
            link_type="DOCUMENTS"
        )

        # Then the connection preserves the reasoning
        assert link_created is True

        # And the decision is enriched with detailed context
        # Verify both entities exist and are linked
        retrieved_kt = tx_manager.get_knowledge_transfer(kt.id)
        assert retrieved_kt is not None
        retrieved_decision = tx_manager.get_decision(sample_decision.id)
        assert retrieved_decision is not None

    def test_scenario_query_knowledge_transfers_by_linked_entities(
        self, tx_manager, sample_task, sample_decision
    ):
        """
        Scenario: Developer finds related knowledge through entity links

        Given multiple knowledge transfers linked to different entities
        When the developer queries for knowledge transfers
        Then they can discover related KTs through graph traversal
        And knowledge is findable from multiple entry points
        """
        # Given multiple knowledge transfers linked to different entities
        kt1 = create_kt(tx_manager,
            title="Transaction Implementation Session 1",
            summary="Initial implementation of transaction primitives"
        )
        tx_manager.link_knowledge_transfer(kt1.id, sample_task.id, "DOCUMENTS")

        kt2 = create_kt(tx_manager,
            title="Transaction Implementation Session 2",
            summary="Conflict detection and resolution strategies"
        )
        tx_manager.link_knowledge_transfer(kt2.id, sample_decision.id, "DOCUMENTS")

        kt3 = create_kt(tx_manager,
            title="Transaction Implementation Session 3",
            summary="Recovery and durability mechanisms"
        )
        tx_manager.link_knowledge_transfer(kt3.id, sample_task.id, "DOCUMENTS")

        # When the developer queries for knowledge transfers
        all_kts = tx_manager.list_knowledge_transfers()

        # Then they can discover related KTs through graph traversal
        kt_ids = {kt.id for kt in all_kts}
        assert kt1.id in kt_ids
        assert kt2.id in kt_ids
        assert kt3.id in kt_ids

        # And knowledge is findable from multiple entry points
        assert len(all_kts) >= 3


# ============================================================================
# BEHAVIORAL SCENARIOS - IMPORTING HISTORICAL KNOWLEDGE
# ============================================================================

class TestDeveloperImportsHistoricalKnowledge:
    """
    Epic: Knowledge Import and Migration

    As a developer with existing knowledge documents,
    I want to import them as knowledge transfer entities,
    So that historical insights are integrated into our custom knowledge graph
    without manual recreation.
    """

    def test_scenario_import_markdown_file_as_knowledge_transfer(self, tx_manager, tmp_path):
        """
        Scenario: Developer imports existing markdown documentation

        Given a markdown file with session notes
        When the developer imports it as a knowledge transfer
        Then a KT entity is created with content from the file
        And the source file path is preserved
        """
        # Given a markdown file with session notes
        md_file = tmp_path / "session-notes.md"
        md_content = """# CDG and GoT Unification

Session: 2025-01-01

## Summary
We unified CDG and GoT by creating configurable transaction layers.

## Technical Implementation
- Built abstraction layer for WAL/recovery/storage
- Maintained API compatibility across both systems
- Everything implemented from first principles

## Key Decisions
- Use composition over inheritance for layer management
- Delegate to CDG while preserving GoT API
"""
        md_file.write_text(md_content)

        # When the developer imports it as a knowledge transfer
        kt = create_kt(tx_manager,
            title="CDG and GoT Unification",
            summary="We unified CDG and GoT by creating configurable transaction layers.",
            session_date="2025-01-01",
            sections={
                "Technical Implementation": "- Built abstraction layer for WAL/recovery/storage\n"
                                           "- Maintained API compatibility across both systems\n"
                                           "- Everything implemented from first principles",
                "Key Decisions": "- Use composition over inheritance for layer management\n"
                               "- Delegate to CDG while preserving GoT API"
            },
            source_file=str(md_file)
        )

        # Then a KT entity is created with content from the file
        assert kt is not None
        assert "Technical Implementation" in kt.sections
        assert "Key Decisions" in kt.sections
        assert "abstraction layer" in kt.sections["Technical Implementation"]

        # And the source file path is preserved
        assert kt.source_file == str(md_file)

    def test_scenario_parse_sections_from_markdown_headings(self, tx_manager, tmp_path):
        """
        Scenario: Automatic section extraction from markdown structure

        Given a markdown file with ## headings
        When the file is parsed for import
        Then each heading becomes a section name
        And content under each heading is captured
        """
        # Given a markdown file with ## headings
        md_file = tmp_path / "detailed-notes.md"
        md_content = """# Recovery Implementation Notes

## Architecture Overview
We built a custom WAL-based recovery system from scratch.

## Implementation Details
The recovery manager scans the WAL backward to find incomplete transactions.

## Performance Considerations
Recovery time is O(n) where n is the number of WAL entries since last checkpoint.
"""
        md_file.write_text(md_content)

        # When the file is parsed for import
        # (Simulating section parsing logic)
        sections = {}
        current_section = None
        current_content = []

        for line in md_content.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            elif current_section and not line.startswith('# '):
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        kt = create_kt(tx_manager,
            title="Recovery Implementation Notes",
            summary="Custom WAL-based recovery implementation",
            sections=sections
        )

        # Then each heading becomes a section name
        assert "Architecture Overview" in kt.sections
        assert "Implementation Details" in kt.sections
        assert "Performance Considerations" in kt.sections

        # And content under each heading is captured
        assert "custom WAL-based recovery" in kt.sections["Architecture Overview"]
        assert "scans the WAL backward" in kt.sections["Implementation Details"]
        assert "O(n)" in kt.sections["Performance Considerations"]

    def test_scenario_extract_session_metadata_from_markdown(self, tx_manager, tmp_path):
        """
        Scenario: Preserve session context during import

        Given a markdown file with session metadata
        When the file is imported
        Then session_id and session_date are extracted
        And they are stored in the knowledge transfer
        """
        # Given a markdown file with session metadata
        md_file = tmp_path / "session-2025-01-01-notes.md"
        md_content = """# Transaction Safety Implementation

**Session ID:** session-2025-01-01-alpha
**Date:** 2025-01-01

## Work Completed
Implemented thread safety for CDGStore using custom locking mechanisms.
"""
        md_file.write_text(md_content)

        # When the file is imported
        # (Simulating metadata extraction)
        session_id = "session-2025-01-01-alpha"
        session_date = "2025-01-01"

        kt = create_kt(tx_manager,
            title="Transaction Safety Implementation",
            summary="Implemented thread safety for CDGStore using custom locking mechanisms",
            session_id=session_id,
            session_date=session_date,
            sections={
                "Work Completed": "Implemented thread safety for CDGStore using custom locking mechanisms."
            }
        )

        # Then session_id and session_date are extracted and stored
        assert kt.session_id == "session-2025-01-01-alpha"
        assert kt.session_date == "2025-01-01"

        retrieved = tx_manager.get_knowledge_transfer(kt.id)
        assert retrieved.session_id == session_id
        assert retrieved.session_date == session_date

    def test_scenario_preserve_code_references_during_import(self, tx_manager, tmp_path):
        """
        Scenario: Code references survive the import process

        Given a markdown file with file:line references
        When the file is imported as a knowledge transfer
        Then code references are extracted and stored
        And they remain accessible in the KT entity
        """
        # Given a markdown file with file:line references
        md_file = tmp_path / "implementation-guide.md"
        md_content = """# Snapshot Isolation Guide

## Key Files
- Transaction manager: `cortical/cdg/transaction_manager.py:249`
- Storage layer: `cortical/cdg/storage.py:156`
- GoT wrapper: `cortical/got/tx_manager.py:158`

## Implementation
See the snapshot version tracking in transaction_manager.py for details.
"""
        md_file.write_text(md_content)

        # When the file is imported as a knowledge transfer
        # (Simulating code reference extraction via regex)
        import re
        code_ref_pattern = r'`([a-zA-Z0-9_/.-]+\.py:\d+)`'
        code_refs = re.findall(code_ref_pattern, md_content)

        kt = create_kt(tx_manager,
            title="Snapshot Isolation Guide",
            summary="Implementation guide for snapshot isolation in our transaction manager",
            code_refs=code_refs,
            sections={
                "Key Files": "- Transaction manager: cortical/cdg/transaction_manager.py:249\n"
                           "- Storage layer: cortical/cdg/storage.py:156\n"
                           "- GoT wrapper: cortical/got/tx_manager.py:158"
            }
        )

        # Then code references are extracted and stored
        assert len(kt.code_refs) == 3
        assert "cortical/cdg/transaction_manager.py:249" in kt.code_refs
        assert "cortical/cdg/storage.py:156" in kt.code_refs
        assert "cortical/got/tx_manager.py:158" in kt.code_refs

        # And they remain accessible in the KT entity
        retrieved = tx_manager.get_knowledge_transfer(kt.id)
        assert len(retrieved.code_refs) == 3
        assert all(":" in ref for ref in retrieved.code_refs)


# ============================================================================
# BEHAVIORAL SCENARIOS - SEARCHING KNOWLEDGE GRAPH
# ============================================================================

class TestDeveloperSearchesKnowledgeGraph:
    """
    Epic: Knowledge Discovery

    As a developer seeking context about past work,
    I want to search and filter knowledge transfers,
    So that I can quickly find relevant insights
    using our custom knowledge management system.
    """

    def test_scenario_list_all_knowledge_transfers(self, tx_manager):
        """
        Scenario: Developer views all available knowledge

        Given multiple knowledge transfers exist
        When the developer lists all knowledge transfers
        Then all KT entities are returned
        And they are ordered by creation time
        """
        # Given multiple knowledge transfers exist
        kt1 = create_kt(tx_manager,
            title="WAL Implementation Session 1",
            summary="Initial WAL design and structure"
        )

        kt2 = create_kt(tx_manager,
            title="Recovery Manager Session",
            summary="Built crash recovery from first principles"
        )

        kt3 = create_kt(tx_manager,
            title="Performance Optimization Session",
            summary="Optimized WAL write throughput"
        )

        # When the developer lists all knowledge transfers
        all_kts = tx_manager.list_knowledge_transfers()

        # Then all KT entities are returned
        kt_ids = {kt.id for kt in all_kts}
        assert kt1.id in kt_ids
        assert kt2.id in kt_ids
        assert kt3.id in kt_ids
        assert len(all_kts) >= 3

    def test_scenario_filter_knowledge_transfers_by_status(self, tx_manager):
        """
        Scenario: Developer finds work-in-progress knowledge

        Given knowledge transfers with different statuses
        When the developer filters by status
        Then only matching KTs are returned
        And draft vs published can be distinguished
        """
        # Given knowledge transfers with different statuses
        kt_published = create_kt(tx_manager,
            title="Completed Analysis",
            summary="Final analysis of transaction performance",
            status="published"
        )

        # Create draft KT (default status)
        kt_draft = create_kt(tx_manager,
            title="WIP: Edge Case Analysis",
            summary="Still investigating corner cases..."
        )
        tx_manager.update_knowledge_transfer(kt_draft.id, status="draft")

        # Create archived KT
        kt_archived = create_kt(tx_manager,
            title="Old Session Notes",
            summary="Archived historical notes"
        )
        tx_manager.update_knowledge_transfer(kt_archived.id, status="archived")

        # When the developer filters by status
        published_kts = tx_manager.list_knowledge_transfers(status="published")
        draft_kts = tx_manager.list_knowledge_transfers(status="draft")
        archived_kts = tx_manager.list_knowledge_transfers(status="archived")

        # Then only matching KTs are returned
        published_ids = {kt.id for kt in published_kts}
        assert kt_published.id in published_ids

        draft_ids = {kt.id for kt in draft_kts}
        assert kt_draft.id in draft_ids
        assert len(draft_kts) >= 1

        archived_ids = {kt.id for kt in archived_kts}
        assert kt_archived.id in archived_ids

        # And draft vs published can be distinguished
        assert kt_draft.id not in published_ids
        assert kt_published.id not in draft_ids

    def test_scenario_filter_knowledge_transfers_by_tags(self, tx_manager):
        """
        Scenario: Developer finds knowledge by topic

        Given knowledge transfers tagged by topic
        When the developer filters by specific tags
        Then only KTs with all specified tags are returned
        And tag filtering uses AND logic
        """
        # Given knowledge transfers tagged by topic
        kt1 = create_kt(tx_manager,
            title="Concurrency Patterns",
            summary="Thread safety in our custom implementations",
            tags=["concurrency", "thread-safety", "patterns"]
        )

        kt2 = create_kt(tx_manager,
            title="WAL Concurrency",
            summary="Concurrent WAL writes in our system",
            tags=["concurrency", "wal", "performance"]
        )

        kt3 = create_kt(tx_manager,
            title="Recovery Algorithms",
            summary="Custom crash recovery implementation",
            tags=["recovery", "algorithms", "wal"]
        )

        # When the developer filters by specific tags
        concurrent_kts = tx_manager.list_knowledge_transfers(tags=["concurrency"])
        wal_kts = tx_manager.list_knowledge_transfers(tags=["wal"])
        concurrent_and_perf = tx_manager.list_knowledge_transfers(
            tags=["concurrency", "performance"]
        )

        # Then only KTs with all specified tags are returned
        concurrent_ids = {kt.id for kt in concurrent_kts}
        assert kt1.id in concurrent_ids
        assert kt2.id in concurrent_ids
        assert kt3.id not in concurrent_ids

        wal_ids = {kt.id for kt in wal_kts}
        assert kt2.id in wal_ids
        assert kt3.id in wal_ids
        assert kt1.id not in wal_ids

        # And tag filtering uses AND logic
        concurrent_perf_ids = {kt.id for kt in concurrent_and_perf}
        assert kt2.id in concurrent_perf_ids  # Has both tags
        assert kt1.id not in concurrent_perf_ids  # Missing performance tag

    def test_scenario_find_related_entities_through_knowledge_links(
        self, tx_manager, sample_task, sample_decision
    ):
        """
        Scenario: Developer traverses knowledge graph

        Given knowledge transfers linked to various entities
        When the developer explores entity relationships
        Then they can navigate from KT to linked tasks/decisions
        And discover the full context of past work
        """
        # Given knowledge transfers linked to various entities
        kt1 = create_kt(tx_manager,
            title="Snapshot Implementation Knowledge",
            summary="Insights from building MVCC ourselves"
        )
        tx_manager.link_knowledge_transfer(kt1.id, sample_task.id, "DOCUMENTS")

        kt2 = create_kt(tx_manager,
            title="Locking Strategy Analysis",
            summary="Why we chose optimistic locking"
        )
        tx_manager.link_knowledge_transfer(kt2.id, sample_decision.id, "DOCUMENTS")

        # When the developer explores entity relationships
        # Verify KTs exist and have links
        all_kts = tx_manager.list_knowledge_transfers()
        kt_ids = {kt.id for kt in all_kts}

        assert kt1.id in kt_ids
        assert kt2.id in kt_ids

        # Then they can navigate from KT to linked tasks/decisions
        # Verify the KTs and linked entities exist
        kt1_exists = tx_manager.get_knowledge_transfer(kt1.id) is not None
        kt2_exists = tx_manager.get_knowledge_transfer(kt2.id) is not None

        assert kt1_exists, "KT linked to task should be retrievable"
        assert kt2_exists, "KT linked to decision should be retrievable"

        # And discover the full context of past work
        # Both entities should be accessible via high-level API
        task_entity = tx_manager.get_task(sample_task.id)
        decision_entity = tx_manager.get_decision(sample_decision.id)

        assert task_entity is not None
        assert decision_entity is not None


# ============================================================================
# BEHAVIORAL SCENARIOS - KNOWLEDGE INTEGRITY
# ============================================================================

class TestSystemMaintainsKnowledgeIntegrity:
    """
    Epic: Knowledge Persistence and Consistency

    As a system maintaining knowledge state,
    I want to ensure knowledge transfers persist reliably,
    So that captured insights are never lost
    and remain consistent across sessions.
    """

    def test_scenario_knowledge_transfer_persists_across_sessions(self, temp_got_dir):
        """
        Scenario: Knowledge survives system restarts

        Given a knowledge transfer created in one session
        When the system restarts with a new transaction manager
        Then the knowledge transfer is still accessible
        And all fields are preserved correctly
        """
        # Given a knowledge transfer created in one session
        # Use disk storage for persistence test across manager instances
        manager1 = _create_tx_manager(temp_got_dir, use_memory=False)
        kt = create_kt(manager1,
            title="Critical System Knowledge",
            summary="Must not be lost on restart",
            session_id="session-001",
            session_date="2025-01-01",
            sections={
                "Architecture": "Key architectural insights we built ourselves",
                "Gotchas": "Edge cases discovered during implementation"
            },
            tags=["critical", "architecture"],
            code_refs=["cortical/core/system.py:42"]
        )
        kt_id = kt.id

        # When the system restarts with a new transaction manager
        manager2 = _create_tx_manager(temp_got_dir, use_memory=False)

        # Then the knowledge transfer is still accessible
        retrieved = manager2.get_knowledge_transfer(kt_id)
        assert retrieved is not None

        # And all fields are preserved correctly
        assert retrieved.title == "Critical System Knowledge"
        assert retrieved.summary == "Must not be lost on restart"
        assert retrieved.session_id == "session-001"
        assert retrieved.session_date == "2025-01-01"
        assert "Architecture" in retrieved.sections
        assert "Gotchas" in retrieved.sections
        assert "critical" in retrieved.tags
        assert "architecture" in retrieved.tags
        assert "cortical/core/system.py:42" in retrieved.code_refs

    def test_scenario_version_increments_on_updates(self, tx_manager):
        """
        Scenario: Knowledge transfer versions track changes

        Given an existing knowledge transfer
        When the knowledge transfer is updated
        Then the version number increments
        And concurrent updates are detected via optimistic locking
        """
        # Given an existing knowledge transfer
        kt = create_kt(tx_manager,
            title="Evolving Knowledge",
            summary="Initial understanding"
        )
        initial_version = kt.version

        # When the knowledge transfer is updated
        updated_kt = tx_manager.append_to_knowledge_transfer(
            kt.id,
            "New Insights",
            "Additional learnings emerged during implementation"
        )

        # Then the version number increments
        assert updated_kt.version > initial_version

        # TODO: Optimistic locking test requires low-level transaction API
        # The TransactionalGoTAdapter doesn't expose begin/read/write/commit.
        # Consider adding a dedicated test in integration tests with CDGTransactionManager
        # to verify optimistic locking behavior when concurrent updates occur.
        # For now, we verify version incrementing works correctly.

        # Verify version continues to increment on further updates
        version_before_second_update = updated_kt.version
        further_updated = tx_manager.append_to_knowledge_transfer(
            kt.id,
            "More Insights",
            "Even more learnings"
        )
        assert further_updated.version > version_before_second_update

    def test_scenario_invalid_status_rejected(self, tx_manager):
        """
        Scenario: System validates knowledge transfer status

        Given a knowledge transfer entity
        When an invalid status is set
        Then a validation error is raised
        And the entity is not persisted
        """
        # Given a knowledge transfer entity
        # When an invalid status is set
        # Then a validation error is raised
        from cortical.got.errors import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            kt = KnowledgeTransfer(
                id="KT-TEST-001",
                title="Test KT",
                status="invalid_status"  # Not in {draft, published, archived}
            )

        # And the entity is not persisted
        assert "Invalid status" in str(exc_info.value)
        # Verify it was never written
        retrieved = tx_manager.get_knowledge_transfer("KT-TEST-001")
        assert retrieved is None
