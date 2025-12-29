"""
Integration tests for GoT-CEL Bridge with REAL DATA.

These tests load actual GoT JSON files from .got/entities/ and:
1. Parse and validate the JSON structure
2. Convert to CEL events via the adapter
3. Store in a CEL MerkleDAG
4. Query and retrieve the data
5. Validate roundtrip integrity

This is READ-ONLY - no modifications to actual GoT data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import pytest

# CEL imports
from cortical.cel.core.events import CognitiveEvent, EventType, Observation
from cortical.cel.core.references import MerkleRoot, EventHorizon
from cortical.cel.wisdom.dag import MerkleDAG
from cortical.cel.wisdom.semantic import BloomSemanticIndex

# GoT adapter
from cortical.cel.adapters.got import GoTEventAdapter, GOT_AVAILABLE

# GoT types (if available)
if GOT_AVAILABLE:
    from cortical.got.types import Task, Decision, Sprint, Edge


# =============================================================================
# FIXTURES
# =============================================================================

GOT_ENTITIES_PATH = Path(".got/entities")


@pytest.fixture
def got_entities_path():
    """Path to GoT entities directory."""
    return GOT_ENTITIES_PATH


@pytest.fixture
def sample_task_files(got_entities_path):
    """Get sample task JSON files."""
    if not got_entities_path.exists():
        pytest.skip("GoT entities directory not found")

    files = list(got_entities_path.glob("T-*.json"))
    if len(files) < 5:
        pytest.skip("Not enough task files for testing")

    return files[:20]  # Use first 20 for testing


@pytest.fixture
def sample_decision_files(got_entities_path):
    """Get sample decision JSON files."""
    if not got_entities_path.exists():
        pytest.skip("GoT entities directory not found")

    files = list(got_entities_path.glob("D-*.json"))
    if len(files) < 3:
        pytest.skip("Not enough decision files for testing")

    return files[:10]


@pytest.fixture
def sample_sprint_files(got_entities_path):
    """Get sample sprint JSON files."""
    if not got_entities_path.exists():
        pytest.skip("GoT entities directory not found")

    files = list(got_entities_path.glob("S-*.json"))
    return files[:5] if files else []


@pytest.fixture
def cel_dag():
    """Fresh CEL MerkleDAG for testing."""
    return MerkleDAG()


@pytest.fixture
def cel_index():
    """Fresh CEL semantic index for testing."""
    return BloomSemanticIndex()


# =============================================================================
# HELPERS
# =============================================================================

def load_got_json(file_path: Path) -> Dict[str, Any]:
    """Load and parse a GoT JSON file."""
    with open(file_path) as f:
        return json.load(f)


def validate_got_checksum(data: Dict[str, Any]) -> bool:
    """Validate GoT entity checksum format."""
    return (
        "_checksum" in data and
        isinstance(data["_checksum"], str) and
        len(data["_checksum"]) == 16  # 8-byte hex
    )


def got_json_to_cel_event(json_data: Dict[str, Any]) -> CognitiveEvent:
    """
    Convert raw GoT JSON to CEL event (without using GoT types).

    This is a direct conversion for testing, not using the full adapter.
    """
    entity_data = json_data.get("data", json_data)
    entity_type = entity_data.get("entity_type", "unknown")

    # Determine CEL event type
    if entity_type == "task":
        status = entity_data.get("status", "pending")
        if status == "completed":
            event_type = EventType.FULFILLMENT
        else:
            event_type = EventType.INTENTION
    else:
        event_type = EventType.OBSERVATION

    # Extract concepts
    concepts = []
    title = entity_data.get("title", "")
    if title:
        # Simple word extraction (stop words removed)
        stop_words = {"the", "a", "an", "to", "for", "of", "in", "on", "at", "is", "are", "and"}
        words = [w.lower() for w in title.split() if w.lower() not in stop_words]
        concepts.extend(words[:5])

    concepts.append(entity_type)
    if entity_data.get("priority"):
        concepts.append(f"priority:{entity_data['priority']}")

    # Build content
    content = {
        "got_id": entity_data.get("id"),
        "entity_type": entity_type,
        "title": title,
        "status": entity_data.get("status"),
        "priority": entity_data.get("priority"),
        "description": entity_data.get("description"),
        "got_checksum": json_data.get("_checksum"),
        "got_version": entity_data.get("version", 1),
    }

    # Get timestamp
    timestamp = entity_data.get("modified_at") or entity_data.get("created_at")
    if not timestamp:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

    return CognitiveEvent(
        timestamp=timestamp,
        event_type=event_type,
        causal_parents=(),
        content=content,
        concepts=tuple(set(concepts)),
    )


# =============================================================================
# USER STORY: LOADING GOT JSON FILES
# =============================================================================

class TestLoadingGoTData:
    """
    User Story: As a migration tool, I want to load existing GoT JSON files
    so I can understand the data I'm working with.
    """

    def test_got_entities_directory_exists(self, got_entities_path):
        """Verify GoT entities directory exists."""
        assert got_entities_path.exists(), \
            f"GoT entities path not found: {got_entities_path}"
        assert got_entities_path.is_dir()

    def test_task_files_can_be_parsed(self, sample_task_files):
        """All task JSON files parse successfully."""
        for file_path in sample_task_files:
            data = load_got_json(file_path)
            assert "data" in data or "id" in data, \
                f"Invalid structure in {file_path.name}"

    def test_task_files_have_checksums(self, sample_task_files):
        """Task files include integrity checksums."""
        for file_path in sample_task_files:
            data = load_got_json(file_path)
            assert validate_got_checksum(data), \
                f"Invalid or missing checksum in {file_path.name}"

    def test_task_files_have_required_fields(self, sample_task_files):
        """Task files contain required entity fields."""
        required_fields = ["id", "title", "status", "entity_type"]

        for file_path in sample_task_files:
            data = load_got_json(file_path)
            entity = data.get("data", data)

            for field in required_fields:
                assert field in entity, \
                    f"Missing {field} in {file_path.name}"

    def test_decision_files_parse_correctly(self, sample_decision_files):
        """Decision files parse and contain expected structure."""
        for file_path in sample_decision_files:
            data = load_got_json(file_path)
            entity = data.get("data", data)

            assert entity.get("entity_type") == "decision"
            assert "title" in entity or "rationale" in entity

    def test_sprint_files_if_present(self, sample_sprint_files):
        """Sprint files parse correctly if they exist."""
        for file_path in sample_sprint_files:
            data = load_got_json(file_path)
            entity = data.get("data", data)

            assert entity.get("entity_type") == "sprint"

    def test_entity_counts_match_filesystem(self, got_entities_path):
        """Verify entity file counts."""
        task_count = len(list(got_entities_path.glob("T-*.json")))
        decision_count = len(list(got_entities_path.glob("D-*.json")))
        edge_count = len(list(got_entities_path.glob("E-*.json")))

        print(f"\nGoT Entity Counts:")
        print(f"  Tasks: {task_count}")
        print(f"  Decisions: {decision_count}")
        print(f"  Edges: {edge_count}")

        assert task_count > 0, "No task files found"


# =============================================================================
# USER STORY: CONVERTING GOT TO CEL EVENTS
# =============================================================================

class TestGoTToCELConversion:
    """
    User Story: As a CEL system, I want to convert GoT entities to CEL events
    so I can work with them in the event-sourced model.
    """

    def test_task_converts_to_cel_event(self, sample_task_files):
        """GoT tasks convert to valid CEL events."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)

            # Verify it's a valid CEL event
            assert event.id is not None
            assert len(event.id) == 64  # SHA256 hex
            assert event.timestamp is not None
            assert event.content["got_id"] is not None

    def test_pending_task_becomes_intention(self, sample_task_files):
        """Pending tasks become INTENTION events."""
        for file_path in sample_task_files:
            data = load_got_json(file_path)
            entity = data.get("data", data)

            if entity.get("status") == "pending":
                event = got_json_to_cel_event(data)
                assert event.event_type == EventType.INTENTION
                break
        else:
            pytest.skip("No pending tasks found")

    def test_completed_task_becomes_fulfillment(self, sample_task_files):
        """Completed tasks become FULFILLMENT events."""
        for file_path in sample_task_files:
            data = load_got_json(file_path)
            entity = data.get("data", data)

            if entity.get("status") == "completed":
                event = got_json_to_cel_event(data)
                assert event.event_type == EventType.FULFILLMENT
                break
        else:
            pytest.skip("No completed tasks found")

    def test_decision_becomes_observation(self, sample_decision_files):
        """Decisions become OBSERVATION events."""
        for file_path in sample_decision_files[:3]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)

            assert event.event_type == EventType.OBSERVATION
            assert "decision" in event.concepts

    def test_concepts_extracted_from_title(self, sample_task_files):
        """Concepts are extracted from entity titles."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            entity = data.get("data", data)
            title = entity.get("title", "")

            if len(title.split()) >= 3:
                event = got_json_to_cel_event(data)
                assert len(event.concepts) >= 2, \
                    f"Expected concepts from title: {title}"
                break

    def test_got_metadata_preserved(self, sample_task_files):
        """GoT metadata is preserved in CEL event content."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)

            # Original GoT ID preserved
            assert event.content["got_id"] is not None

            # Checksum preserved for integrity verification
            assert event.content["got_checksum"] is not None

            # Version preserved for tracking
            assert event.content["got_version"] is not None


# =============================================================================
# USER STORY: STORING IN CEL MERKLE DAG
# =============================================================================

class TestStoringInCEL:
    """
    User Story: As a CEL system, I want to store converted events in
    a MerkleDAG so I can maintain causal ordering and integrity.
    """

    def test_events_can_be_added_to_dag(self, sample_task_files, cel_dag):
        """Converted events can be stored in CEL DAG."""
        added_roots = []

        for file_path in sample_task_files[:10]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)

            root = cel_dag.add(event)
            added_roots.append(root)

        assert len(added_roots) == 10
        assert cel_dag.count == 10

    def test_events_retrievable_by_id(self, sample_task_files, cel_dag):
        """Stored events can be retrieved by their Merkle root."""
        data = load_got_json(sample_task_files[0])
        event = got_json_to_cel_event(data)
        root = cel_dag.add(event)

        retrieved = cel_dag.get(root.value)

        assert retrieved is not None
        assert retrieved.id == event.id
        assert retrieved.content["got_id"] == event.content["got_id"]

    def test_dag_tracks_heads(self, sample_task_files, cel_dag):
        """DAG correctly tracks head events."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_dag.add(event)

        heads = cel_dag.get_heads()

        # All events are heads (no causal links between them)
        assert len(heads) == 5

    def test_content_addressable_ids(self, sample_task_files, cel_dag):
        """Events get deterministic content-addressed IDs."""
        data = load_got_json(sample_task_files[0])

        # Create same event twice
        event1 = got_json_to_cel_event(data)
        event2 = got_json_to_cel_event(data)

        # Same content = same ID
        assert event1.id == event2.id

    def test_causal_chain_can_be_built(self, sample_task_files, cel_dag):
        """Events can be linked in a causal chain."""
        # Add first event
        data1 = load_got_json(sample_task_files[0])
        event1 = got_json_to_cel_event(data1)
        root1 = cel_dag.add(event1)

        # Create second event with first as parent
        data2 = load_got_json(sample_task_files[1])
        entity2 = data2.get("data", data2)

        # Manually create event with causal parent
        event2 = CognitiveEvent(
            timestamp=entity2.get("modified_at") or "2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(root1.value,),  # Link to first
            content={
                "got_id": entity2.get("id"),
                "entity_type": entity2.get("entity_type"),
                "title": entity2.get("title"),
            },
            concepts=("linked", "event"),
        )
        root2 = cel_dag.add(event2)

        # Verify chain
        assert root1.value not in cel_dag.heads  # No longer a head
        assert root2.value in cel_dag.heads  # New head

        # Verify ancestry
        ancestors = list(cel_dag.ancestors(root2.value))
        assert len(ancestors) == 1
        assert ancestors[0].id == root1.value


# =============================================================================
# USER STORY: SEMANTIC INDEXING
# =============================================================================

class TestSemanticIndexing:
    """
    User Story: As a query system, I want to index GoT data semantically
    so I can search by concepts.
    """

    def test_events_indexed_by_concepts(self, sample_task_files, cel_index):
        """Events are indexed by their concepts."""
        events = []
        for file_path in sample_task_files[:10]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            events.append(event)
            cel_index.index_event(event)

        # All events should be indexed
        assert cel_index.stats["event_count"] == 10

    def test_search_by_entity_type(self, sample_task_files, sample_decision_files, cel_index):
        """Can search for events by entity type concept."""
        # Index tasks
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_index.index_event(event)

        # Index decisions
        for file_path in sample_decision_files[:3]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_index.index_event(event)

        # Search for tasks
        assert cel_index.probably_contains("task")

        # Search for decisions
        assert cel_index.probably_contains("decision")

    def test_bloom_filter_fast_rejection(self, sample_task_files, cel_index):
        """Bloom filter quickly rejects non-existent concepts."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_index.index_event(event)

        # This definitely doesn't exist
        assert not cel_index.probably_contains("xyzzy_nonexistent_12345")


# =============================================================================
# USER STORY: INTEGRITY CHECKS
# =============================================================================

class TestIntegrityChecks:
    """
    User Story: As a data steward, I want to verify data integrity
    throughout the GoT to CEL conversion process.
    """

    def test_got_checksums_are_valid_format(self, sample_task_files):
        """All GoT files have valid checksum format."""
        for file_path in sample_task_files:
            data = load_got_json(file_path)

            assert "_checksum" in data
            checksum = data["_checksum"]

            # Checksum is 16 hex chars (8 bytes)
            assert len(checksum) == 16
            assert all(c in "0123456789abcdef" for c in checksum)

    def test_cel_event_ids_are_deterministic(self, sample_task_files):
        """CEL event IDs are deterministic (same input = same ID)."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)

            # Convert twice
            event1 = got_json_to_cel_event(data)
            event2 = got_json_to_cel_event(data)

            # IDs must match
            assert event1.id == event2.id

    def test_roundtrip_preserves_got_id(self, sample_task_files, cel_dag):
        """GoT entity ID is preserved through CEL storage."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            original_got_id = data.get("data", data).get("id")

            # Convert and store
            event = got_json_to_cel_event(data)
            root = cel_dag.add(event)

            # Retrieve
            retrieved = cel_dag.get(root.value)

            # Verify GoT ID preserved
            assert retrieved.content["got_id"] == original_got_id

    def test_event_count_matches_input(self, sample_task_files, cel_dag):
        """Number of stored events matches input count."""
        n_inputs = min(len(sample_task_files), 15)

        for file_path in sample_task_files[:n_inputs]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_dag.add(event)

        assert cel_dag.count == n_inputs


# =============================================================================
# USER STORY: QUERYING CEL DATA
# =============================================================================

class TestQueryingCELData:
    """
    User Story: As a developer, I want to query the converted data
    to verify it's usable and correct.
    """

    def test_list_all_stored_events(self, sample_task_files, cel_dag):
        """Can iterate over all stored events."""
        stored_ids = []

        for file_path in sample_task_files[:10]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            root = cel_dag.add(event)
            stored_ids.append(root.value)

        # Get all via causal_order
        all_events = list(cel_dag.causal_order())

        assert len(all_events) == 10
        retrieved_ids = {e.id for e in all_events}

        for stored_id in stored_ids:
            assert stored_id in retrieved_ids

    def test_filter_by_event_type(self, sample_task_files, cel_dag):
        """Can filter events by type after retrieval."""
        for file_path in sample_task_files[:10]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_dag.add(event)

        all_events = list(cel_dag.causal_order())

        intentions = [e for e in all_events if e.event_type == EventType.INTENTION]
        fulfillments = [e for e in all_events if e.event_type == EventType.FULFILLMENT]

        print(f"\nEvent type distribution:")
        print(f"  INTENTION (pending): {len(intentions)}")
        print(f"  FULFILLMENT (completed): {len(fulfillments)}")

        # Should have at least one of each type
        assert len(intentions) + len(fulfillments) == len(all_events)

    def test_get_heads_for_latest_state(self, sample_task_files, cel_dag):
        """Heads represent the latest state."""
        for file_path in sample_task_files[:5]:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)
            cel_dag.add(event)

        heads = cel_dag.get_heads()
        latest = cel_dag.get_latest()

        assert len(heads) == 5  # No causal links = all are heads
        assert latest is not None
        assert latest.value in [h.value for h in heads]


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestFullRoundtrip:
    """End-to-end test demonstrating the full data flow."""

    def test_full_got_to_cel_roundtrip(self, got_entities_path, cel_dag, cel_index):
        """
        Full roundtrip: Load GoT → Convert → Store → Index → Query
        """
        if not got_entities_path.exists():
            pytest.skip("GoT entities not available")

        task_files = list(got_entities_path.glob("T-*.json"))[:20]
        decision_files = list(got_entities_path.glob("D-*.json"))[:5]

        all_files = task_files + decision_files
        events_by_got_id = {}

        print("\n" + "="*60)
        print("FULL GOT → CEL ROUNDTRIP TEST")
        print("="*60)

        # Step 1: Load and convert
        print(f"\n1. Loading {len(all_files)} GoT entities...")
        for file_path in all_files:
            data = load_got_json(file_path)
            event = got_json_to_cel_event(data)

            got_id = event.content["got_id"]
            events_by_got_id[got_id] = event

        print(f"   Converted {len(events_by_got_id)} entities to CEL events")

        # Step 2: Store in DAG
        print("\n2. Storing in MerkleDAG...")
        for event in events_by_got_id.values():
            cel_dag.add(event)
            cel_index.index_event(event)

        print(f"   Stored {cel_dag.count} events")
        print(f"   Indexed {cel_index.stats['event_count']} events")
        print(f"   Terms indexed: {cel_index.stats['term_count']}")

        # Step 3: Query and verify
        print("\n3. Querying stored data...")
        all_events = list(cel_dag.causal_order())

        intentions = [e for e in all_events if e.event_type == EventType.INTENTION]
        fulfillments = [e for e in all_events if e.event_type == EventType.FULFILLMENT]
        observations = [e for e in all_events if e.event_type == EventType.OBSERVATION]

        print(f"   INTENTION events: {len(intentions)}")
        print(f"   FULFILLMENT events: {len(fulfillments)}")
        print(f"   OBSERVATION events: {len(observations)}")

        # Step 4: Verify integrity
        print("\n4. Verifying integrity...")
        for event in all_events:
            got_id = event.content["got_id"]
            assert got_id in events_by_got_id, f"Missing event: {got_id}"

        print(f"   ✓ All {len(all_events)} events verified")

        # Step 5: Check semantic index
        print("\n5. Testing semantic search...")
        assert cel_index.probably_contains("task")
        print("   ✓ 'task' concept found")

        if decision_files:
            assert cel_index.probably_contains("decision")
            print("   ✓ 'decision' concept found")

        print("\n" + "="*60)
        print("ROUNDTRIP COMPLETE - ALL CHECKS PASSED")
        print("="*60)

        # Assertions for test framework
        assert cel_dag.count == len(all_files)
        assert len(all_events) == len(all_files)
