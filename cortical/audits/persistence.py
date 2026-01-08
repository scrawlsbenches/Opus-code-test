"""
Audit state persistence - Backend protocols and implementations.

This module provides persistence infrastructure for audit reasoning state,
including file importance tracking, attention focus, and PLN rules.

Implementations:
    - FilePersistenceBackend: Real filesystem persistence
    - NullPersistenceBackend: No-op for testing
    - InMemoryPersistenceBackend: In-memory for testing
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime

from cortical.common.filesystem import FileSystem, RealFileSystem


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_PERSISTENCE_FILE = Path(".got") / "audit_pln_state.json"
DEFAULT_RULES_FILE = Path(".got") / "audit_pln_rules.json"
DEFAULT_WOVEN_MIND_FILE = Path(".got") / "woven_audit_mind.json"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FileImportanceRecord:
    """
    Persistent record of a file's importance over time.

    Tracks STI (short-term importance), LTI (long-term importance),
    and VLTI (very long-term importance) for prioritization.
    """
    file_id: str
    sti: float
    lti: float
    vlti: bool
    last_seen: str  # ISO timestamp
    history: List[Dict[str, Any]]  # Historical snapshots

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_id": self.file_id,
            "sti": self.sti,
            "lti": self.lti,
            "vlti": self.vlti,
            "last_seen": self.last_seen,
            "history": self.history[-50:],  # Keep last 50 snapshots
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileImportanceRecord":
        """Create from dictionary."""
        return cls(
            file_id=data["file_id"],
            sti=data.get("sti", 0.3),
            lti=data.get("lti", 0.1),
            vlti=data.get("vlti", False),
            last_seen=data.get("last_seen", datetime.now().isoformat()),
            history=data.get("history", []),
        )


@dataclass
class AuditPersistenceState:
    """
    Complete persistent state for audit reasoning.

    Stores file importance records, attention focus, and global statistics
    across sessions.
    """
    version: int
    created: str
    updated: str
    session_count: int
    file_importance: Dict[str, FileImportanceRecord]
    attention_focus: List[str]  # Files currently in focus
    global_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "created": self.created,
            "updated": self.updated,
            "session_count": self.session_count,
            "file_importance": {
                k: v.to_dict() for k, v in self.file_importance.items()
            },
            "attention_focus": self.attention_focus,
            "global_stats": self.global_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditPersistenceState":
        """Create from dictionary."""
        file_importance = {}
        for k, v in data.get("file_importance", {}).items():
            file_importance[k] = FileImportanceRecord.from_dict(v)

        return cls(
            version=data.get("version", 1),
            created=data.get("created", datetime.now().isoformat()),
            updated=data.get("updated", datetime.now().isoformat()),
            session_count=data.get("session_count", 0),
            file_importance=file_importance,
            attention_focus=data.get("attention_focus", []),
            global_stats=data.get("global_stats", {}),
        )

    @classmethod
    def create_new(cls) -> "AuditPersistenceState":
        """Create a fresh state instance."""
        now = datetime.now().isoformat()
        return cls(
            version=1,
            created=now,
            updated=now,
            session_count=0,
            file_importance={},
            attention_focus=[],
            global_stats={},
        )


# =============================================================================
# PERSISTENCE BACKEND PROTOCOL
# =============================================================================

class PersistenceBackend(Protocol):
    """
    Protocol for audit state persistence.

    Implementations must provide load/save methods for both
    state and rules.
    """

    def load_state(self) -> AuditPersistenceState:
        """Load persistence state."""
        ...

    def save_state(self, state: AuditPersistenceState) -> None:
        """Save persistence state."""
        ...

    def load_rules(self) -> Dict[str, Any]:
        """Load PLN rules."""
        ...

    def save_rules(self, rules: Dict[str, Any]) -> None:
        """Save PLN rules."""
        ...


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================

class FilePersistenceBackend:
    """
    Real filesystem persistence backend.

    Stores state and rules as JSON files on disk.
    """

    def __init__(
        self,
        filesystem: FileSystem,
        persistence_file: Optional[Path] = None,
        rules_file: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize with filesystem and file paths.

        Args:
            filesystem: FileSystem implementation
            persistence_file: Path for state file (default: .got/audit_pln_state.json)
            rules_file: Path for rules file (default: .got/audit_pln_rules.json)
            base_dir: Base directory for relative paths (default: cwd)
        """
        self._fs = filesystem
        base = base_dir or Path.cwd()
        self._persistence_file = persistence_file or (base / DEFAULT_PERSISTENCE_FILE)
        self._rules_file = rules_file or (base / DEFAULT_RULES_FILE)

    def load_state(self) -> AuditPersistenceState:
        """Load persistence state from disk."""
        if self._fs.exists(self._persistence_file):
            try:
                data = json.loads(self._fs.read_text(self._persistence_file))
                return AuditPersistenceState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load persistence state: {e}")
        return AuditPersistenceState.create_new()

    def save_state(self, state: AuditPersistenceState) -> None:
        """Save persistence state to disk."""
        parent = self._persistence_file.parent
        if not self._fs.exists(parent):
            self._fs.mkdir(parent, parents=True, exist_ok=True)

        state.updated = datetime.now().isoformat()
        self._fs.write_text(
            self._persistence_file,
            json.dumps(state.to_dict(), indent=2)
        )

    def load_rules(self) -> Dict[str, Any]:
        """Load PLN rules from disk."""
        if self._fs.exists(self._rules_file):
            try:
                return json.loads(self._fs.read_text(self._rules_file))
            except (json.JSONDecodeError, KeyError):
                pass
        return self._create_empty_rules()

    def save_rules(self, rules: Dict[str, Any]) -> None:
        """Save PLN rules to disk."""
        parent = self._rules_file.parent
        if not self._fs.exists(parent):
            self._fs.mkdir(parent, parents=True, exist_ok=True)

        rules["updated"] = datetime.now().isoformat()
        self._fs.write_text(
            self._rules_file,
            json.dumps(rules, indent=2)
        )

    def _create_empty_rules(self) -> Dict[str, Any]:
        """Create empty rules structure."""
        return {
            "version": 1,
            "created": datetime.now().isoformat(),
            "rules": [],
            "manual_rules": [],
            "derived_rules": [],
        }


class NullPersistenceBackend:
    """
    No-op persistence backend for testing.

    Does not persist any state - useful for isolated tests.
    """

    def load_state(self) -> AuditPersistenceState:
        """Return fresh state (nothing to load)."""
        return AuditPersistenceState.create_new()

    def save_state(self, state: AuditPersistenceState) -> None:
        """No-op (don't save anything)."""
        pass

    def load_rules(self) -> Dict[str, Any]:
        """Return empty rules."""
        return {
            "version": 1,
            "created": datetime.now().isoformat(),
            "rules": [],
            "manual_rules": [],
            "derived_rules": [],
        }

    def save_rules(self, rules: Dict[str, Any]) -> None:
        """No-op (don't save anything)."""
        pass


class InMemoryPersistenceBackend:
    """
    In-memory persistence backend for testing.

    Stores state in memory - useful for tests that need to verify persistence
    behavior without disk I/O.
    """

    def __init__(self):
        self._state: Optional[AuditPersistenceState] = None
        self._rules: Dict[str, Any] = {
            "version": 1,
            "created": datetime.now().isoformat(),
            "rules": [],
            "manual_rules": [],
            "derived_rules": [],
        }
        # Tracking for test assertions
        self.save_state_calls = 0
        self.load_state_calls = 0
        self.save_rules_calls = 0
        self.load_rules_calls = 0

    def load_state(self) -> AuditPersistenceState:
        """Load state from memory."""
        self.load_state_calls += 1
        if self._state is None:
            return AuditPersistenceState.create_new()
        return self._state

    def save_state(self, state: AuditPersistenceState) -> None:
        """Save state to memory."""
        self.save_state_calls += 1
        state.updated = datetime.now().isoformat()
        self._state = state

    def load_rules(self) -> Dict[str, Any]:
        """Load rules from memory."""
        self.load_rules_calls += 1
        return self._rules.copy()

    def save_rules(self, rules: Dict[str, Any]) -> None:
        """Save rules to memory."""
        self.save_rules_calls += 1
        rules["updated"] = datetime.now().isoformat()
        self._rules = rules.copy()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_default_persistence(base_dir: Optional[Path] = None) -> PersistenceBackend:
    """
    Factory function for creating the default production persistence backend.

    This is the ONLY place in the module that knows about RealFileSystem.
    All other code should receive a PersistenceBackend via dependency injection.

    Args:
        base_dir: Base directory for persistence files (default: cwd)

    Returns:
        FilePersistenceBackend configured for production use
    """
    return FilePersistenceBackend(RealFileSystem(), base_dir=base_dir)
