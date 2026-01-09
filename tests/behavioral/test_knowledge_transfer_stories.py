"""
Behavioral tests for Knowledge Transfer operations.

As a developer documenting insights from development sessions,
I want to create, read, update and search knowledge transfers,
So that I can preserve and retrieve session learnings.

Tests demonstrate:
- KT creation persists data correctly
- KT can be read back with all fields intact
- KT can be updated (sections, status, etc.)
- KT can be searched and filtered

Following Metus: We describe behavior, then make it true.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got.api import GoTManager
from tests.conftest import _create_container


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_got_dir(tmp_path):
    """Provide a temporary directory for GoT operations."""
    got_dir = tmp_path / ".got"
    return got_dir


@pytest.fixture
def container(temp_got_dir):
    """Provide a DI container for each test."""
    return _create_container(temp_got_dir)  # Default in-memory for speed


# ============================================================================
# BEHAVIORAL SCENARIOS
# ============================================================================

class TestDeveloperCreatesKnowledgeTransfer:
    """
    Epic: Knowledge Transfer Lifecycle Management

    As a developer documenting session insights,
    I want to create knowledge transfer documents,
    So that I can preserve learnings for future reference.
    """

    def test_kt_creation_persists_all_fields(self, container):
        """
        Scenario: Creating a KT persists all provided fields

        Given a fresh GoTManager
        When I create a knowledge transfer with all fields populated
        Then I can read it back with all fields intact
        """
        got_manager = container.resolve(GoTManager)

        # When: Create KT with all fields
        kt = got_manager.create_knowledge_transfer(
            title="Session: CDG-GoT Unification Insights",
            summary="Unified CDG and GoT storage layers using dependency injection",
            session_id="session-2026-01-05-abc123",
            session_date="2026-01-05T10:30:00Z",
            sections={
                "Key Decisions": "Chose DI container pattern for testability",
                "Implementation Notes": "Used bootstrap.py for component wiring",
                "Lessons Learned": "Container-first approach enables mock isolation",
            },
            code_refs=[
                "cortical/core/bootstrap.py:45",
                "cortical/got/tx_manager.py:286"
            ],
            tags=["architecture", "dependency-injection"],
            status="draft"
        )

        # Then: Read back and verify all fields
        retrieved = got_manager.get_knowledge_transfer(kt.id)
        assert retrieved is not None
        assert retrieved.title == "Session: CDG-GoT Unification Insights"
        assert retrieved.summary == "Unified CDG and GoT storage layers using dependency injection"
        assert retrieved.session_id == "session-2026-01-05-abc123"
        assert "Key Decisions" in retrieved.sections
        assert "Implementation Notes" in retrieved.sections
        assert "Lessons Learned" in retrieved.sections
        assert len(retrieved.code_refs) == 2
        assert "architecture" in retrieved.tags
        assert retrieved.status == "draft"

    def test_kt_creation_with_minimal_fields(self, container):
        """
        Scenario: Creating a KT with only required fields succeeds

        Given a fresh GoTManager
        When I create a knowledge transfer with only title
        Then it is created with sensible defaults
        """
        got_manager = container.resolve(GoTManager)

        # When: Create KT with minimal fields
        kt = got_manager.create_knowledge_transfer(title="Minimal KT")

        # Then: Read back with defaults
        retrieved = got_manager.get_knowledge_transfer(kt.id)
        assert retrieved is not None
        assert retrieved.title == "Minimal KT"
        assert retrieved.status == "draft"  # Default status
        assert retrieved.sections == {} or retrieved.sections is None or len(retrieved.sections) == 0
        assert retrieved.tags == [] or retrieved.tags is None or len(retrieved.tags) == 0

    def test_kt_update_persists_changes(self, container):
        """
        Scenario: Updating a KT persists the changes

        Given an existing knowledge transfer
        When I update its status and summary
        Then the changes are persisted
        """
        got_manager = container.resolve(GoTManager)

        # Given: Create a KT
        kt = got_manager.create_knowledge_transfer(
            title="Work in Progress",
            summary="Initial notes",
            status="draft"
        )

        # When: Update it
        updated = got_manager.update_knowledge_transfer(
            kt.id,
            summary="Completed notes with full analysis",
            status="published"
        )

        # Then: Changes are persisted
        retrieved = got_manager.get_knowledge_transfer(kt.id)
        assert retrieved.summary == "Completed notes with full analysis"
        assert retrieved.status == "published"

    def test_kt_append_section_accumulates_content(self, container):
        """
        Scenario: Appending sections to a KT accumulates content

        Given a knowledge transfer with existing sections
        When I append new sections
        Then the content is accumulated
        """
        got_manager = container.resolve(GoTManager)

        # Given: Create KT with initial section
        kt = got_manager.create_knowledge_transfer(
            title="Evolving Documentation",
            sections={"Initial Notes": "First observations"}
        )

        # When: Append new sections
        got_manager.append_knowledge_transfer_section(
            kt.id, "Day 2 Insights", "Additional findings"
        )
        got_manager.append_knowledge_transfer_section(
            kt.id, "Day 3 Insights", "Final conclusions"
        )

        # Then: All sections are present
        retrieved = got_manager.get_knowledge_transfer(kt.id)
        assert "Initial Notes" in retrieved.sections
        assert "Day 2 Insights" in retrieved.sections
        assert "Day 3 Insights" in retrieved.sections

    def test_kt_append_to_existing_section_concatenates(self, container):
        """
        Scenario: Appending to an existing section concatenates content

        Given a knowledge transfer with a section
        When I append to the same section name
        Then the content is concatenated
        """
        got_manager = container.resolve(GoTManager)

        # Given: Create KT with a section
        kt = got_manager.create_knowledge_transfer(
            title="Ongoing Investigation",
            sections={"Findings": "First finding"}
        )

        # When: Append to same section
        got_manager.append_knowledge_transfer_section(
            kt.id, "Findings", "Second finding"
        )

        # Then: Content is concatenated
        retrieved = got_manager.get_knowledge_transfer(kt.id)
        assert "First finding" in retrieved.sections["Findings"]
        assert "Second finding" in retrieved.sections["Findings"]


class TestDeveloperSearchesKnowledgeTransfers:
    """
    Epic: Knowledge Transfer Discovery

    As a developer looking for past learnings,
    I want to search and filter knowledge transfers,
    So that I can find relevant documentation quickly.
    """

    def test_list_all_knowledge_transfers(self, container):
        """
        Scenario: Listing all knowledge transfers returns all created ones

        Given multiple knowledge transfers exist
        When I list all knowledge transfers
        Then all are returned
        """
        got_manager = container.resolve(GoTManager)

        # Given: Create multiple KTs
        kt1 = got_manager.create_knowledge_transfer(title="First KT")
        kt2 = got_manager.create_knowledge_transfer(title="Second KT")
        kt3 = got_manager.create_knowledge_transfer(title="Third KT")

        # When: List all
        all_kts = got_manager.list_knowledge_transfers()

        # Then: All are returned
        ids = [kt.id for kt in all_kts]
        assert kt1.id in ids
        assert kt2.id in ids
        assert kt3.id in ids

    def test_list_knowledge_transfers_by_status(self, container):
        """
        Scenario: Filtering by status returns matching KTs

        Given knowledge transfers with different statuses
        When I filter by status
        Then only matching ones are returned
        """
        got_manager = container.resolve(GoTManager)

        # Given: Create KTs with different statuses
        draft_kt = got_manager.create_knowledge_transfer(
            title="Draft KT", status="draft"
        )
        published_kt = got_manager.create_knowledge_transfer(
            title="Published KT", status="published"
        )

        # When: Filter by draft
        drafts = got_manager.list_knowledge_transfers(status="draft")

        # Then: Only draft is returned
        draft_ids = [kt.id for kt in drafts]
        assert draft_kt.id in draft_ids
        assert published_kt.id not in draft_ids

    def test_list_knowledge_transfers_by_tags(self, container):
        """
        Scenario: Filtering by tags returns matching KTs

        Given knowledge transfers with different tags
        When I filter by specific tags
        Then only KTs with all specified tags are returned
        """
        got_manager = container.resolve(GoTManager)

        # Given: Create KTs with different tags
        arch_kt = got_manager.create_knowledge_transfer(
            title="Architecture KT", tags=["architecture", "design"]
        )
        testing_kt = got_manager.create_knowledge_transfer(
            title="Testing KT", tags=["testing", "quality"]
        )
        both_kt = got_manager.create_knowledge_transfer(
            title="Both KT", tags=["architecture", "testing"]
        )

        # When: Filter by architecture tag
        arch_kts = got_manager.list_knowledge_transfers(tags=["architecture"])

        # Then: Only KTs with architecture tag are returned
        arch_ids = [kt.id for kt in arch_kts]
        assert arch_kt.id in arch_ids
        assert both_kt.id in arch_ids
        assert testing_kt.id not in arch_ids


class TestDeveloperUsesGoTManagerDirectly:
    """
    Epic: Direct GoTManager Usage

    As a developer building custom workflows,
    I want to create knowledge transfers via GoTManager,
    So that I can integrate KT creation into complex transactional operations.
    """

    def test_got_manager_creates_and_retrieves_kt(self, container):
        """
        Scenario: GoTManager creates and retrieves KT successfully

        Given I have a GoTManager from the DI container
        When I create a knowledge transfer using the GoT manager
        Then I can read it back with all fields intact
        """
        got_manager = container.resolve(GoTManager)

        # When: Create a KT
        kt = got_manager.create_knowledge_transfer(
            title="Direct GoTManager KT Creation",
            summary="Testing direct GoT manager usage",
            session_id="session-direct-got",
            sections={"Testing": "Direct creation through GoTManager"},
            tags=["transaction", "direct"]
        )

        # Then: Read it back with all fields
        retrieved = got_manager.get_knowledge_transfer(kt.id)
        assert retrieved is not None
        assert retrieved.title == "Direct GoTManager KT Creation"
        assert retrieved.summary == "Testing direct GoT manager usage"
        assert retrieved.session_id == "session-direct-got"
        assert "Testing" in retrieved.sections
        assert "transaction" in retrieved.tags
