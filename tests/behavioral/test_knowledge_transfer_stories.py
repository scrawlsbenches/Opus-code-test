"""
Behavioral tests for Knowledge Transfer operations.

As a developer documenting insights from development sessions,
I want knowledge transfers to be stored transactionally with checksums,
So that I can safely persist and retrieve session learnings without corruption.

Tests demonstrate:
- KT creation uses proper transactional storage
- KT files have checksum wrapper format
- KT can be read back without corruption errors
- KT validation passes after creation

Following Metus: We describe behavior, then make it true.

Regression Coverage:
- Bug: TransactionalGoTAdapter.create_knowledge_transfer() was bypassing CDGStore
  and writing files directly without checksums, causing CorruptionError on read
- Fix: Delegate to TransactionManager which uses CDGStore with checksums
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.core.bootstrap import create_container
from cortical.got import (
    TransactionManager,
    CorruptionError,
)
from cortical.got.types import KnowledgeTransfer
from scripts.got_utils import TransactionalGoTAdapter
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
def got_adapter(temp_got_dir):
    """Provide a TransactionalGoTAdapter for each test."""
    return TransactionalGoTAdapter(got_dir=temp_got_dir)


@pytest.fixture
def container(temp_got_dir):
    """Provide a DI container for each test."""
    return _create_container(temp_got_dir)


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

    def test_kt_creation_should_use_transactional_storage_with_checksums(
        self, got_adapter, temp_got_dir
    ):
        """
        Scenario: Creating a KT persists it with proper checksum protection

        Given a fresh GoT adapter
        When I create a knowledge transfer with title and sections
        Then the KT file has proper checksum wrapper format
        And the KT can be read back without corruption errors
        And validation passes after creation

        Regression: Previously, create_knowledge_transfer() bypassed CDGStore
        and wrote files directly without checksums, causing CorruptionError.
        """
        # Given: a fresh GoT adapter (provided by fixture)

        # When: I create a knowledge transfer with title and sections
        kt_id = got_adapter.create_knowledge_transfer(
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

        # Then: the KT file has proper checksum wrapper format
        entities_dir = temp_got_dir / "entities"
        kt_file = entities_dir / f"{kt_id}.json"

        assert kt_file.exists(), f"KT file should exist at {kt_file}"

        with open(kt_file, 'r') as f:
            wrapper = json.load(f)

        # Verify checksum wrapper structure
        assert "_checksum" in wrapper, "File must have _checksum field"
        assert isinstance(wrapper["_checksum"], str), "Checksum must be a string"
        assert len(wrapper["_checksum"]) == 16, \
            "Checksum must be 16 hex characters (truncated SHA256)"
        assert "_written_at" in wrapper, "File must have _written_at timestamp"
        assert "data" in wrapper, "File must have data field"

        # Verify data structure
        data = wrapper["data"]
        assert data["id"] == kt_id
        assert data["entity_type"] == "knowledge_transfer"
        assert data["title"] == "Session: CDG-GoT Unification Insights"
        assert data["summary"] == "Unified CDG and GoT storage layers using dependency injection"
        assert data["session_id"] == "session-2026-01-05-abc123"
        assert "Key Decisions" in data["sections"]
        assert len(data["code_refs"]) == 2
        assert "architecture" in data["tags"]
        assert data["status"] == "draft"

        # And: the KT can be read back without corruption errors
        # This verifies checksums are validated on read
        retrieved_kt = got_adapter.get_knowledge_transfer(kt_id)
        assert retrieved_kt is not None, "KT should be retrievable"
        assert retrieved_kt["title"] == "Session: CDG-GoT Unification Insights"
        assert retrieved_kt["status"] == "draft"

        # And: validation passes after creation
        # The file should pass checksum validation
        # We verify this by re-reading directly through CDGStore
        from cortical.cdg.storage import CDGStore
        from cortical.got.types import create_entity_from_dict

        store = CDGStore(entities_dir, entity_factory=create_entity_from_dict)
        # This will raise CorruptionError if checksum is invalid
        validated_kt = store.read(kt_id)
        assert validated_kt is not None
        assert validated_kt.id == kt_id


    def test_kt_with_corrupted_checksum_should_raise_corruption_error(
        self, got_adapter, temp_got_dir
    ):
        """
        Scenario: Corrupted KT file is detected via checksum validation

        Given I have created a knowledge transfer
        When the file is corrupted on disk
        And I attempt to read it
        Then a CorruptionError is raised
        And the corruption is detected before returning bad data

        This verifies that checksums are actually being validated on read.
        """
        # Given: I have created a knowledge transfer
        kt_id = got_adapter.create_knowledge_transfer(
            title="Test KT for corruption detection",
            summary="This KT will be corrupted to test checksum validation",
            session_id="session-test-corruption",
            tags=["test"]
        )

        # Verify it can be read initially
        kt = got_adapter.get_knowledge_transfer(kt_id)
        assert kt is not None

        # When: the file is corrupted on disk
        entities_dir = temp_got_dir / "entities"
        kt_file = entities_dir / f"{kt_id}.json"

        with open(kt_file, 'r') as f:
            wrapper = json.load(f)

        # Corrupt the checksum (use 16 hex chars like actual format)
        wrapper['_checksum'] = 'deadbeefcorrupt1'

        with open(kt_file, 'w') as f:
            json.dump(wrapper, f)

        # And: I attempt to read it
        # Then: a CorruptionError is raised
        from cortical.got.versioned_store import VersionedStore

        store = VersionedStore(entities_dir)

        with pytest.raises(CorruptionError) as exc_info:
            store.read(kt_id)

        # And: the corruption is detected before returning bad data
        assert kt_id in str(exc_info.value)


    def test_kt_update_should_preserve_checksum_integrity(
        self, got_adapter, temp_got_dir
    ):
        """
        Scenario: Updating a KT maintains checksum protection

        Given I have created a knowledge transfer
        When I update it by appending a section
        Then the updated file has a new valid checksum
        And the KT can be read back with the updates
        And validation passes for the updated data

        Regression Coverage: _update_kt_entity() now uses TransactionManager
        for proper transactional storage with checksums (fixed 2026-01-05).
        """
        # Given: I have created a knowledge transfer
        kt_id = got_adapter.create_knowledge_transfer(
            title="Evolving Session Documentation",
            summary="Initial summary",
            session_id="session-evolving-123",
            sections={
                "Initial Thoughts": "First section content"
            },
            tags=["evolving"]
        )

        # Read original checksum
        entities_dir = temp_got_dir / "entities"
        kt_file = entities_dir / f"{kt_id}.json"

        with open(kt_file, 'r') as f:
            original_wrapper = json.load(f)
        original_checksum = original_wrapper["_checksum"]

        # When: I update it by appending a section
        success = got_adapter.append_kt_section(
            kt_id,
            "New Insights",
            "Additional learnings discovered later"
        )
        assert success, "Section append should succeed"

        # Then: the updated file has a new valid checksum
        with open(kt_file, 'r') as f:
            updated_wrapper = json.load(f)
        updated_checksum = updated_wrapper["_checksum"]

        assert updated_checksum != original_checksum, \
            "Checksum must change when data changes"
        assert isinstance(updated_checksum, str) and len(updated_checksum) == 16, \
            "Updated checksum must still be 16 hex characters"

        # And: the KT can be read back with the updates
        retrieved = got_adapter.get_knowledge_transfer(kt_id)
        assert "New Insights" in retrieved["sections"]
        assert retrieved["sections"]["New Insights"] == "Additional learnings discovered later"

        # And: validation passes for the updated data
        from cortical.got.versioned_store import VersionedStore

        store = VersionedStore(entities_dir)
        validated_kt = store.read(kt_id)  # Will raise if checksum invalid
        assert validated_kt is not None
        assert "New Insights" in validated_kt.sections


class TestDeveloperUsesTransactionManagerDirectly:
    """
    Epic: Direct Transaction Manager Usage

    As a developer building custom workflows,
    I want to create knowledge transfers via TransactionManager,
    So that I can integrate KT creation into complex transactional operations.
    """

    def test_tx_manager_creates_kt_with_checksums(self, container, temp_got_dir):
        """
        Scenario: TransactionManager creates KT with proper checksum protection

        Given I have a TransactionManager from the DI container
        When I create a knowledge transfer using the transaction manager
        Then the KT is stored with checksum wrapper
        And I can read it back without corruption
        """
        # Given: I have a TransactionManager from the DI container
        tx_manager = container.resolve(TransactionManager)

        # When: I create a knowledge transfer using the transaction manager
        kt = tx_manager.create_knowledge_transfer(
            title="Direct TxManager KT Creation",
            summary="Testing direct transaction manager usage",
            session_id="session-direct-tx",
            sections={
                "Testing": "Direct creation through TxManager"
            },
            tags=["transaction", "direct"]
        )

        # Then: the KT is stored with checksum wrapper
        entities_dir = temp_got_dir / "entities"
        kt_file = entities_dir / f"{kt.id}.json"

        assert kt_file.exists()

        with open(kt_file, 'r') as f:
            wrapper = json.load(f)

        assert "_checksum" in wrapper
        assert isinstance(wrapper["_checksum"], str) and len(wrapper["_checksum"]) == 16
        assert wrapper["data"]["id"] == kt.id

        # And: I can read it back without corruption
        retrieved_kt = tx_manager.get_knowledge_transfer(kt.id)
        assert retrieved_kt is not None
        assert retrieved_kt.title == "Direct TxManager KT Creation"
        assert retrieved_kt.summary == "Testing direct transaction manager usage"
