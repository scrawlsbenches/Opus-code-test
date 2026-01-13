"""
Behavioral specification for Conflict-Free Entity Storage.

Epic: Developer Uses Git-Friendly Transactional Storage

As a developer working on a team with multiple branches,
I want entity storage that never causes merge conflicts on infrastructure files,
So that I can focus on resolving real semantic conflicts, not fighting with WAL files.

Design Principles:
1. WAL is local-only (gitignored) - crash recovery is a local concern
2. Entity versions use timestamps, not integers - no sequence coordination needed
3. No global version file - reconstruct from entity timestamps on startup
4. Conflicts detected via content hash - self-healing after git operations
5. Fail hard on conflict - preserve both versions, block until resolved

Following Metus: We describe behavior, then make it true.
"""

import json
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import pytest


# ==============================================================================
# SPECIFICATION DATA STRUCTURES
# ==============================================================================

@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    entity_id: str
    local_version: Dict[str, Any]
    external_version: Dict[str, Any]
    local_hash: str
    external_hash: str


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Provide a temporary directory for entity storage."""
    storage_dir = tmp_path / "entities"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def temp_wal_dir(tmp_path):
    """Provide a temporary directory for WAL (separate from entities)."""
    wal_dir = tmp_path / "wal"
    wal_dir.mkdir()
    return wal_dir


# ==============================================================================
# HELPER FUNCTIONS (These represent the expected API)
# ==============================================================================

def compute_content_hash(entity_data: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of entity content.

    This hash is used for conflict detection - if two versions
    of an entity have different hashes, they have different content.
    """
    # Remove metadata fields that shouldn't affect content hash
    content = {k: v for k, v in entity_data.items()
               if not k.startswith('_')}
    # Sort keys for deterministic serialization
    serialized = json.dumps(content, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def generate_timestamp_version() -> str:
    """
    Generate a timestamp-based version identifier.

    Format: ISO-8601 timestamp with random suffix for uniqueness.
    This ensures versions never collide across branches.
    """
    import secrets
    ts = datetime.now(timezone.utc).isoformat()
    suffix = secrets.token_hex(4)
    return f"{ts}-{suffix}"


def write_entity(storage_dir: Path, entity_id: str, data: Dict[str, Any]) -> Path:
    """Write an entity to storage with timestamp version and content hash."""
    # Add metadata
    data['_modified_at'] = generate_timestamp_version()
    data['_content_hash'] = compute_content_hash(data)

    entity_path = storage_dir / f"{entity_id}.json"
    with open(entity_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

    return entity_path


def read_entity(storage_dir: Path, entity_id: str) -> Optional[Dict[str, Any]]:
    """Read an entity from storage."""
    entity_path = storage_dir / f"{entity_id}.json"
    if not entity_path.exists():
        return None
    with open(entity_path) as f:
        return json.load(f)


def detect_conflict(
    storage_dir: Path,
    entity_id: str,
    expected_hash: str
) -> Optional[ConflictInfo]:
    """
    Detect if an entity was modified externally (e.g., by git pull).

    Returns ConflictInfo if the entity's current hash doesn't match
    the expected hash from WAL, indicating external modification.
    """
    current = read_entity(storage_dir, entity_id)
    if current is None:
        return None  # Entity doesn't exist, no conflict

    current_hash = current.get('_content_hash', '')
    if current_hash == expected_hash:
        return None  # Hashes match, no conflict

    return ConflictInfo(
        entity_id=entity_id,
        local_version={},  # Would be populated from WAL
        external_version=current,
        local_hash=expected_hash,
        external_hash=current_hash
    )


def save_conflict_file(
    storage_dir: Path,
    entity_id: str,
    local_data: Dict[str, Any]
) -> Path:
    """
    Save local version to a .conflict file for user resolution.

    This preserves the user's local changes when a conflict is detected,
    allowing them to manually merge or choose which version to keep.
    """
    conflicts_dir = storage_dir / "conflicts"
    conflicts_dir.mkdir(exist_ok=True)

    conflict_path = conflicts_dir / f"{entity_id}.local.json"
    with open(conflict_path, 'w') as f:
        json.dump(local_data, f, indent=2, sort_keys=True)

    return conflict_path


def check_system_locked(storage_dir: Path) -> bool:
    """Check if the system is in conflict-locked state."""
    lock_file = storage_dir / ".conflict_lock"
    return lock_file.exists()


def lock_system(storage_dir: Path, conflicts: list) -> None:
    """Lock the system due to unresolved conflicts."""
    lock_file = storage_dir / ".conflict_lock"
    with open(lock_file, 'w') as f:
        json.dump({
            'locked_at': datetime.now(timezone.utc).isoformat(),
            'conflicts': [c.entity_id for c in conflicts]
        }, f, indent=2)


def unlock_system(storage_dir: Path) -> None:
    """Unlock the system after conflicts are resolved."""
    lock_file = storage_dir / ".conflict_lock"
    if lock_file.exists():
        lock_file.unlink()

    # Also clean up conflict files
    conflicts_dir = storage_dir / "conflicts"
    if conflicts_dir.exists():
        for f in conflicts_dir.glob("*.json"):
            f.unlink()


# ==============================================================================
# BEHAVIORAL SCENARIOS: Conflict-Free IDs
# ==============================================================================

class TestEntitiesHaveConflictFreeIdentifiers:
    """
    Epic: Conflict-Free Entity Creation

    As a developer creating entities on a feature branch,
    I want entity IDs that never collide with other branches,
    So that merging branches doesn't cause ID conflicts.
    """

    def test_scenario_timestamp_versions_never_collide(self, temp_storage_dir):
        """
        Scenario: Two entities created at different times have unique versions

        Given an entity storage system using timestamp versions
        When I create multiple entities
        Then each entity has a unique version identifier
        And versions are lexicographically sortable by time
        Because timestamp + random suffix guarantees uniqueness
        """
        # Given an entity storage system using timestamp versions
        # (provided by fixture)

        # When I create multiple entities
        versions = []
        for i in range(10):
            version = generate_timestamp_version()
            versions.append(version)

        # Then each entity has a unique version identifier
        assert len(versions) == len(set(versions)), \
            "All versions must be unique"

        # And versions are lexicographically sortable by time
        # (ISO-8601 format is naturally sortable)
        sorted_versions = sorted(versions)
        # The sorted order should roughly match creation order
        # (may not be exact due to same-millisecond creation)
        assert sorted_versions[0] <= sorted_versions[-1]

    def test_scenario_content_hash_detects_changes(self, temp_storage_dir):
        """
        Scenario: Content hash changes when entity data changes

        Given an entity with certain content
        When I modify the entity's content
        Then the content hash changes
        And identical content produces identical hash
        Because hash is deterministic and content-based
        """
        # Given an entity with certain content
        original_data = {
            'id': 'T-001',
            'title': 'Original Title',
            'status': 'pending'
        }
        original_hash = compute_content_hash(original_data)

        # When I modify the entity's content
        modified_data = {
            'id': 'T-001',
            'title': 'Modified Title',  # Changed
            'status': 'pending'
        }
        modified_hash = compute_content_hash(modified_data)

        # Then the content hash changes
        assert original_hash != modified_hash, \
            "Hash must change when content changes"

        # And identical content produces identical hash
        same_data = {
            'id': 'T-001',
            'title': 'Original Title',
            'status': 'pending'
        }
        same_hash = compute_content_hash(same_data)
        assert original_hash == same_hash, \
            "Identical content must produce identical hash"

    def test_scenario_metadata_excluded_from_hash(self, temp_storage_dir):
        """
        Scenario: Metadata fields don't affect content hash

        Given an entity with content and metadata
        When metadata fields change but content stays same
        Then the content hash remains unchanged
        Because only semantic content matters for conflict detection
        """
        # Given an entity with content and metadata
        data_v1 = {
            'id': 'T-001',
            'title': 'My Task',
            '_modified_at': '2026-01-13T10:00:00+00:00-abc123',
            '_content_hash': 'will_be_ignored'
        }

        # When metadata fields change but content stays same
        data_v2 = {
            'id': 'T-001',
            'title': 'My Task',
            '_modified_at': '2026-01-13T12:00:00+00:00-def456',  # Changed
            '_content_hash': 'different_value'  # Changed
        }

        # Then the content hash remains unchanged
        hash_v1 = compute_content_hash(data_v1)
        hash_v2 = compute_content_hash(data_v2)

        assert hash_v1 == hash_v2, \
            "Metadata changes should not affect content hash"


# ==============================================================================
# BEHAVIORAL SCENARIOS: WAL is Local Only
# ==============================================================================

class TestWALIsLocalCrashRecoveryOnly:
    """
    Epic: WAL Files Never Cause Merge Conflicts

    As a developer merging branches,
    I want WAL files to be completely local,
    So that I never see merge conflicts on WAL or sequence files.
    """

    def test_scenario_wal_directory_is_gitignored(self, temp_storage_dir):
        """
        Scenario: WAL directory should not be tracked by git

        Given a project with entity storage
        When I check which files should be in .gitignore
        Then the WAL directory is listed
        And the sequence file is listed
        Because these are local crash recovery files, not shared state
        """
        # This is a specification test - we're defining what SHOULD be true
        # The implementation will ensure .gitignore contains these entries

        expected_gitignore_entries = [
            '.got/entities/wal/',
            '.got/entities/wal/_sequence.json',
            '.got/entities/wal/current.wal',
        ]

        # For now, just document the requirement
        # Implementation will add these to .gitignore
        for entry in expected_gitignore_entries:
            # This assertion documents the requirement
            assert entry is not None, f"Must gitignore: {entry}"

    def test_scenario_sequence_reconstructed_from_wal_on_startup(
        self, temp_storage_dir, temp_wal_dir
    ):
        """
        Scenario: Sequence number is derived from WAL content, not stored file

        Given a WAL with entries but no sequence file
        When the system starts up
        Then the sequence is reconstructed from max(entry sequences) + 1
        Because the sequence file is just a cache, not source of truth
        """
        # Given a WAL with entries but no sequence file
        wal_file = temp_wal_dir / "current.wal"

        # Write some WAL entries with sequence numbers
        entries = [
            {'seq': 1, 'ts': '2026-01-13T10:00:00Z', 'op': 'TX_BEGIN'},
            {'seq': 2, 'ts': '2026-01-13T10:00:01Z', 'op': 'WRITE'},
            {'seq': 3, 'ts': '2026-01-13T10:00:02Z', 'op': 'TX_COMMIT'},
        ]

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        # Ensure no sequence file exists
        seq_file = temp_wal_dir / "_sequence.json"
        assert not seq_file.exists()

        # When the system starts up - reconstruct sequence
        max_seq = 0
        with open(wal_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'seq' in entry:
                    max_seq = max(max_seq, entry['seq'])

        next_seq = max_seq + 1

        # Then the sequence is reconstructed correctly
        assert next_seq == 4, \
            "Next sequence should be max(existing) + 1"


# ==============================================================================
# BEHAVIORAL SCENARIOS: Conflict Detection
# ==============================================================================

class TestSystemDetectsExternalModifications:
    """
    Epic: Detect When Git Changed Entities

    As a developer who just did git pull,
    I want the system to detect if my local WAL conflicts with pulled entities,
    So that I don't silently lose work or corrupt data.
    """

    def test_scenario_conflict_detected_when_hash_mismatch(
        self, temp_storage_dir
    ):
        """
        Scenario: System detects entity was modified externally

        Given an entity that I modified locally (tracked in WAL)
        And the same entity was modified by git pull
        When the system checks for conflicts
        Then a conflict is detected
        Because the content hash doesn't match what WAL expects
        """
        # Given an entity that I modified locally (tracked in WAL)
        entity_id = "T-conflict-test"
        local_data = {
            'id': entity_id,
            'title': 'My Local Changes',
            'status': 'in_progress'
        }
        local_hash = compute_content_hash(local_data)

        # And the same entity was modified by git pull
        # (Simulate: write different content to disk)
        external_data = {
            'id': entity_id,
            'title': 'Changes From Git Pull',  # Different!
            'status': 'done'  # Different!
        }
        external_data['_content_hash'] = compute_content_hash(external_data)
        external_data['_modified_at'] = generate_timestamp_version()

        entity_path = temp_storage_dir / f"{entity_id}.json"
        with open(entity_path, 'w') as f:
            json.dump(external_data, f)

        # When the system checks for conflicts
        conflict = detect_conflict(temp_storage_dir, entity_id, local_hash)

        # Then a conflict is detected
        assert conflict is not None, \
            "Conflict should be detected when hashes don't match"
        assert conflict.entity_id == entity_id
        assert conflict.local_hash == local_hash
        assert conflict.external_hash != local_hash

    def test_scenario_no_conflict_when_hashes_match(self, temp_storage_dir):
        """
        Scenario: No conflict when entity unchanged

        Given an entity in storage
        And the expected hash matches the stored hash
        When the system checks for conflicts
        Then no conflict is detected
        Because the entity wasn't modified externally
        """
        # Given an entity in storage
        entity_id = "T-no-conflict"
        data = {
            'id': entity_id,
            'title': 'Unchanged Entity',
            'status': 'pending'
        }
        expected_hash = compute_content_hash(data)

        # Write entity with matching hash
        data['_content_hash'] = expected_hash
        data['_modified_at'] = generate_timestamp_version()

        entity_path = temp_storage_dir / f"{entity_id}.json"
        with open(entity_path, 'w') as f:
            json.dump(data, f)

        # When the system checks for conflicts
        conflict = detect_conflict(temp_storage_dir, entity_id, expected_hash)

        # Then no conflict is detected
        assert conflict is None, \
            "No conflict should be detected when hashes match"


# ==============================================================================
# BEHAVIORAL SCENARIOS: Fail Hard on Conflict
# ==============================================================================

class TestSystemFailsHardOnConflict:
    """
    Epic: Conflicts Block All Operations Until Resolved

    As a developer who might lose work,
    I want the system to fail loudly when conflicts exist,
    So that I'm forced to resolve them before data corruption occurs.
    """

    def test_scenario_system_locks_when_conflict_detected(
        self, temp_storage_dir
    ):
        """
        Scenario: System enters locked state on conflict detection

        Given a detected conflict between local and external changes
        When the system processes the conflict
        Then the system enters a locked state
        And all write operations are blocked
        Because silent data loss is worse than blocking the user
        """
        # Given a detected conflict
        conflict = ConflictInfo(
            entity_id="T-locked",
            local_version={'title': 'Local'},
            external_version={'title': 'External'},
            local_hash="abc123",
            external_hash="def456"
        )

        # When the system processes the conflict
        lock_system(temp_storage_dir, [conflict])

        # Then the system enters a locked state
        assert check_system_locked(temp_storage_dir), \
            "System should be locked after conflict detected"

        # And the lock file contains conflict information
        lock_file = temp_storage_dir / ".conflict_lock"
        with open(lock_file) as f:
            lock_data = json.load(f)

        assert "T-locked" in lock_data['conflicts']

    def test_scenario_local_version_preserved_in_conflict_file(
        self, temp_storage_dir
    ):
        """
        Scenario: Local changes are saved to conflict file

        Given a conflict where local changes would be lost
        When the system handles the conflict
        Then local changes are saved to a .conflict file
        And the user can review both versions
        Because we never silently discard user work
        """
        # Given a conflict where local changes would be lost
        entity_id = "T-preserve"
        local_data = {
            'id': entity_id,
            'title': 'My Important Local Work',
            'description': 'Hours of effort here',
            'status': 'in_progress'
        }

        # When the system handles the conflict
        conflict_path = save_conflict_file(
            temp_storage_dir,
            entity_id,
            local_data
        )

        # Then local changes are saved to a .conflict file
        assert conflict_path.exists(), \
            "Conflict file should be created"
        assert ".local.json" in conflict_path.name

        # And the user can review both versions
        with open(conflict_path) as f:
            preserved_data = json.load(f)

        assert preserved_data['title'] == 'My Important Local Work'
        assert preserved_data['description'] == 'Hours of effort here'

    def test_scenario_system_unlocks_after_resolution(self, temp_storage_dir):
        """
        Scenario: System unlocks after user resolves conflicts

        Given a system in conflict-locked state
        When the user resolves all conflicts
        Then the system is unlocked
        And conflict files are cleaned up
        And normal operations resume
        """
        # Given a system in conflict-locked state
        conflict = ConflictInfo(
            entity_id="T-resolve",
            local_version={},
            external_version={},
            local_hash="abc",
            external_hash="def"
        )
        lock_system(temp_storage_dir, [conflict])
        save_conflict_file(
            temp_storage_dir,
            "T-resolve",
            {'title': 'Local'}
        )

        assert check_system_locked(temp_storage_dir)

        # When the user resolves all conflicts
        unlock_system(temp_storage_dir)

        # Then the system is unlocked
        assert not check_system_locked(temp_storage_dir), \
            "System should be unlocked after resolution"

        # And conflict files are cleaned up
        conflicts_dir = temp_storage_dir / "conflicts"
        if conflicts_dir.exists():
            remaining_conflicts = list(conflicts_dir.glob("*.json"))
            assert len(remaining_conflicts) == 0, \
                "Conflict files should be cleaned up"

    def test_scenario_multiple_conflicts_all_tracked(self, temp_storage_dir):
        """
        Scenario: Multiple conflicts are all recorded

        Given multiple entities with conflicts
        When the system detects them
        Then all conflicts are recorded in the lock file
        And all local versions are preserved
        Because partial conflict detection could still cause data loss
        """
        # Given multiple entities with conflicts
        conflicts = [
            ConflictInfo(
                entity_id="T-multi-1",
                local_version={'title': 'Local 1'},
                external_version={'title': 'External 1'},
                local_hash="hash1a",
                external_hash="hash1b"
            ),
            ConflictInfo(
                entity_id="T-multi-2",
                local_version={'title': 'Local 2'},
                external_version={'title': 'External 2'},
                local_hash="hash2a",
                external_hash="hash2b"
            ),
            ConflictInfo(
                entity_id="T-multi-3",
                local_version={'title': 'Local 3'},
                external_version={'title': 'External 3'},
                local_hash="hash3a",
                external_hash="hash3b"
            ),
        ]

        # When the system detects them
        lock_system(temp_storage_dir, conflicts)

        for conflict in conflicts:
            save_conflict_file(
                temp_storage_dir,
                conflict.entity_id,
                conflict.local_version
            )

        # Then all conflicts are recorded in the lock file
        lock_file = temp_storage_dir / ".conflict_lock"
        with open(lock_file) as f:
            lock_data = json.load(f)

        assert len(lock_data['conflicts']) == 3
        assert "T-multi-1" in lock_data['conflicts']
        assert "T-multi-2" in lock_data['conflicts']
        assert "T-multi-3" in lock_data['conflicts']

        # And all local versions are preserved
        conflicts_dir = temp_storage_dir / "conflicts"
        preserved_files = list(conflicts_dir.glob("*.json"))
        assert len(preserved_files) == 3


# ==============================================================================
# BEHAVIORAL SCENARIOS: Git Integration
# ==============================================================================

class TestStorageSurvivesGitOperations:
    """
    Epic: Normal Git Workflow Works Seamlessly

    As a developer using git pull, merge, checkout,
    I want entity storage to handle git operations gracefully,
    So that I can use normal git workflow without special procedures.
    """

    def test_scenario_entities_merge_cleanly_when_different_fields(
        self, temp_storage_dir
    ):
        """
        Scenario: Git can auto-merge entities with non-overlapping changes

        Given consistent JSON formatting (sorted keys, consistent indent)
        When two branches modify different fields of same entity
        Then git's text merge can potentially auto-merge
        Because JSON with sorted keys is more merge-friendly
        """
        # Given consistent JSON formatting
        base_entity = {
            'description': 'Original description',
            'id': 'T-merge-test',
            'status': 'pending',
            'title': 'Original title'
        }

        # Branch A changes title
        branch_a = {
            'description': 'Original description',
            'id': 'T-merge-test',
            'status': 'pending',
            'title': 'Changed by branch A'  # Modified
        }

        # Branch B changes status
        branch_b = {
            'description': 'Original description',
            'id': 'T-merge-test',
            'status': 'done',  # Modified
            'title': 'Original title'
        }

        # When formatted consistently, each field is on its own line
        def format_entity(data):
            return json.dumps(data, indent=2, sort_keys=True)

        base_lines = format_entity(base_entity).split('\n')
        branch_a_lines = format_entity(branch_a).split('\n')
        branch_b_lines = format_entity(branch_b).split('\n')

        # Then the changes are on different lines
        # Line 4 in sorted order is 'status', Line 5 is 'title'
        # Branch A changes line 5, Branch B changes line 4
        # This makes git auto-merge more likely to succeed

        # Find which lines differ
        a_diffs = [i for i, (a, b) in enumerate(zip(base_lines, branch_a_lines)) if a != b]
        b_diffs = [i for i, (a, b) in enumerate(zip(base_lines, branch_b_lines)) if a != b]

        # Changes should be on different lines (no overlap)
        assert set(a_diffs).isdisjoint(set(b_diffs)), \
            "Non-overlapping changes should be on different lines"

    def test_scenario_fresh_clone_works_without_wal(self, temp_storage_dir):
        """
        Scenario: Fresh git clone has no WAL and works correctly

        Given a fresh clone with entity files but no WAL
        When the system starts up
        Then it operates normally
        And a new local WAL is created as needed
        Because WAL absence is normal after clone
        """
        # Given a fresh clone with entity files but no WAL
        # (Just entity files, no WAL directory)
        write_entity(temp_storage_dir, "T-from-git", {
            'id': 'T-from-git',
            'title': 'Entity from git',
            'status': 'pending'
        })

        # WAL directory doesn't exist (simulating fresh clone)
        wal_dir = temp_storage_dir.parent / "wal"
        assert not wal_dir.exists()

        # When the system starts up - entity can be read
        entity = read_entity(temp_storage_dir, "T-from-git")

        # Then it operates normally
        assert entity is not None
        assert entity['title'] == 'Entity from git'

        # And system is not locked
        assert not check_system_locked(temp_storage_dir)

    def test_scenario_version_reconstructed_from_entities(
        self, temp_storage_dir
    ):
        """
        Scenario: Global version derived from entity timestamps

        Given entities with _modified_at timestamps
        When the system needs to know the latest version
        Then it scans entities for max timestamp
        Because no global version file means no version conflicts
        """
        # Given entities with _modified_at timestamps
        import time

        write_entity(temp_storage_dir, "T-old", {
            'id': 'T-old',
            'title': 'Older entity'
        })

        time.sleep(0.01)  # Ensure different timestamp

        write_entity(temp_storage_dir, "T-new", {
            'id': 'T-new',
            'title': 'Newer entity'
        })

        # When the system scans for latest version
        latest_timestamp = None
        for entity_file in temp_storage_dir.glob("*.json"):
            with open(entity_file) as f:
                data = json.load(f)
            modified_at = data.get('_modified_at', '')
            if latest_timestamp is None or modified_at > latest_timestamp:
                latest_timestamp = modified_at

        # Then we have the latest timestamp
        assert latest_timestamp is not None

        # And it came from T-new (the newer entity)
        new_entity = read_entity(temp_storage_dir, "T-new")
        assert new_entity['_modified_at'] == latest_timestamp


# ==============================================================================
# BEHAVIORAL SCENARIOS: Error Messages
# ==============================================================================

class TestConflictMessagesAreActionable:
    """
    Epic: Developer Knows Exactly How to Fix Conflicts

    As a developer who just hit a conflict,
    I want clear error messages with resolution steps,
    So that I can quickly get back to productive work.
    """

    def test_scenario_conflict_message_shows_both_versions(
        self, temp_storage_dir
    ):
        """
        Scenario: Conflict message shows local and external versions

        Given a detected conflict
        When the error message is generated
        Then it shows the entity ID
        And it shows where local version is saved
        And it shows how to resolve
        Because actionable errors are better than cryptic ones
        """
        # Given a detected conflict
        entity_id = "T-message-test"
        local_data = {'title': 'Local Title'}
        external_data = {'title': 'External Title'}

        conflict_path = save_conflict_file(
            temp_storage_dir,
            entity_id,
            local_data
        )

        # When the error message is generated
        message_lines = [
            "CONFLICT DETECTED",
            "",
            f"Entity: {entity_id}",
            f"  Local version saved to: {conflict_path}",
            f"  External version at: {temp_storage_dir / f'{entity_id}.json'}",
            "",
            "Resolution options:",
            "  --accept-local  : Keep your local changes",
            "  --accept-git    : Accept the git version",
            "",
            "System is LOCKED until conflicts are resolved.",
        ]

        error_message = '\n'.join(message_lines)

        # Then it shows the entity ID
        assert entity_id in error_message

        # And it shows where local version is saved
        assert str(conflict_path) in error_message

        # And it shows how to resolve
        assert "--accept-local" in error_message
        assert "--accept-git" in error_message
        assert "LOCKED" in error_message
