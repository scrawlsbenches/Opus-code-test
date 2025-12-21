"""
Context Pool for Multi-Agent Coordination

Enables agents to publish and query shared findings with conflict detection.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable, Tuple, Any
from pathlib import Path
from collections import defaultdict
import hashlib


@dataclass(frozen=True)
class ContextFinding:
    """
    Immutable finding published to the context pool.

    Args:
        topic: Category/subject of the finding (e.g., "file_structure", "bug_analysis")
        content: The actual finding/discovery
        source_agent: Agent ID that published this finding
        timestamp: Unix timestamp when published
        confidence: Confidence score 0.0-1.0
        finding_id: Unique identifier for this finding
        metadata: Optional metadata (task_id, file_path, etc.)
    """
    topic: str
    content: str
    source_agent: str
    timestamp: float
    confidence: float
    finding_id: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ContextFinding':
        """Deserialize from dictionary."""
        return cls(**data)

    def conflicts_with(self, other: 'ContextFinding') -> bool:
        """
        Detect if this finding conflicts with another.

        Two findings conflict if they:
        1. Share the same topic
        2. Have significantly different content (not exact match)
        3. Are from different agents
        """
        if self.topic != other.topic:
            return False
        if self.source_agent == other.source_agent:
            return False

        # Simple content comparison (can be enhanced with semantic similarity)
        return self.content.strip().lower() != other.content.strip().lower()


class ConflictResolutionStrategy:
    """Strategies for resolving conflicts between findings."""

    MANUAL = "manual"  # Director must resolve
    LAST_WRITE_WINS = "last_write_wins"  # Most recent finding wins
    HIGHEST_CONFIDENCE = "highest_confidence"  # Highest confidence wins
    MERGE = "merge"  # Keep all conflicting findings


class ContextPool:
    """
    Shared context pool for multi-agent coordination.

    Features:
    - Publish immutable findings
    - Query by topic or all findings
    - TTL-based expiration
    - Conflict detection
    - Optional persistence
    - Subscription callbacks
    """

    def __init__(
        self,
        ttl_seconds: Optional[float] = None,
        conflict_strategy: str = ConflictResolutionStrategy.MANUAL,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize context pool.

        Args:
            ttl_seconds: Time-to-live for findings (None = no expiration)
            conflict_strategy: How to handle conflicts
            storage_dir: Directory for persistence (None = memory-only)
        """
        self._findings: Dict[str, List[ContextFinding]] = defaultdict(list)
        self._all_findings: List[ContextFinding] = []
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._conflicts: List[Tuple[ContextFinding, ContextFinding]] = []

        self.ttl_seconds = ttl_seconds
        self.conflict_strategy = conflict_strategy
        self.storage_dir = Path(storage_dir) if storage_dir else None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        topic: str,
        content: str,
        source_agent: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextFinding:
        """
        Publish a new finding to the pool.

        Args:
            topic: Category of the finding
            content: The finding content
            source_agent: Agent publishing this finding
            confidence: Confidence score 0.0-1.0
            metadata: Optional metadata (task_id, file_path, etc.)

        Returns:
            The created ContextFinding

        Raises:
            ValueError: If confidence not in [0.0, 1.0]
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        # Generate unique ID
        finding_id = self._generate_id(topic, content, source_agent)

        # Create immutable finding
        finding = ContextFinding(
            topic=topic,
            content=content,
            source_agent=source_agent,
            timestamp=time.time(),
            confidence=confidence,
            finding_id=finding_id,
            metadata=metadata or {}
        )

        # Check for conflicts
        existing = self._findings[topic]
        should_add = True
        for other in existing:
            if finding.conflicts_with(other):
                should_add = self._handle_conflict(finding, other)
                if not should_add:
                    break

        # Publish to pool (only if not rejected by conflict resolution)
        if should_add:
            self._findings[topic].append(finding)
            self._all_findings.append(finding)

            # Notify subscribers
            self._notify_subscribers(topic, finding)

            # Persist if configured
            if self.storage_dir:
                self._persist_finding(finding)

        return finding

    def query(self, topic: str) -> List[ContextFinding]:
        """
        Query findings by topic.

        Args:
            topic: Topic to query

        Returns:
            List of findings for this topic (may be empty)
        """
        self._prune_expired()
        return list(self._findings[topic])  # Return copy

    def query_all(self) -> List[ContextFinding]:
        """
        Query all findings across all topics.

        Returns:
            List of all findings
        """
        self._prune_expired()
        return list(self._all_findings)  # Return copy

    def subscribe(self, topic: str, callback: Callable[[ContextFinding], None]) -> None:
        """
        Subscribe to findings on a topic.

        Args:
            topic: Topic to subscribe to
            callback: Function called when new finding is published
        """
        self._subscribers[topic].append(callback)

    def get_conflicts(self) -> List[Tuple[ContextFinding, ContextFinding]]:
        """
        Get all detected conflicts.

        Returns:
            List of (finding1, finding2) conflict pairs
        """
        return list(self._conflicts)

    def get_topics(self) -> List[str]:
        """Get all topics with findings."""
        self._prune_expired()
        return list(self._findings.keys())

    def count(self, topic: Optional[str] = None) -> int:
        """
        Count findings.

        Args:
            topic: Count for specific topic, or all if None

        Returns:
            Number of findings
        """
        self._prune_expired()
        if topic:
            return len(self._findings[topic])
        return len(self._all_findings)

    def clear(self) -> None:
        """Clear all findings and conflicts."""
        self._findings.clear()
        self._all_findings.clear()
        self._conflicts.clear()

    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Save pool state to JSON.

        Args:
            filepath: File to save to (uses storage_dir if not specified)
        """
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage_dir configured")
            filepath = self.storage_dir / "context_pool.json"

        data = {
            "findings": [f.to_dict() for f in self._all_findings],
            "conflicts": [
                (f1.finding_id, f2.finding_id)
                for f1, f2 in self._conflicts
            ],
            "ttl_seconds": self.ttl_seconds,
            "conflict_strategy": self.conflict_strategy
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Optional[Path] = None) -> None:
        """
        Load pool state from JSON.

        Args:
            filepath: File to load from (uses storage_dir if not specified)
        """
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage_dir configured")
            filepath = self.storage_dir / "context_pool.json"

        if not filepath.exists():
            return  # Nothing to load

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore findings
        self._all_findings = [
            ContextFinding.from_dict(f) for f in data["findings"]
        ]

        # Rebuild topic index
        self._findings.clear()
        for finding in self._all_findings:
            self._findings[finding.topic].append(finding)

        # Restore conflicts
        finding_by_id = {f.finding_id: f for f in self._all_findings}
        self._conflicts = [
            (finding_by_id[id1], finding_by_id[id2])
            for id1, id2 in data.get("conflicts", [])
            if id1 in finding_by_id and id2 in finding_by_id
        ]

        self.ttl_seconds = data.get("ttl_seconds")
        self.conflict_strategy = data.get("conflict_strategy", ConflictResolutionStrategy.MANUAL)

    # Private methods

    def _generate_id(self, topic: str, content: str, source_agent: str) -> str:
        """Generate unique ID for a finding."""
        timestamp = time.time()
        data = f"{topic}:{content}:{source_agent}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _handle_conflict(
        self,
        new_finding: ContextFinding,
        existing: ContextFinding
    ) -> bool:
        """
        Handle a conflict between findings.

        Returns:
            True if new finding should be added, False otherwise
        """
        # Record the conflict
        self._conflicts.append((existing, new_finding))

        # Apply resolution strategy
        if self.conflict_strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            # Remove older finding, add new one
            self._findings[existing.topic].remove(existing)
            self._all_findings.remove(existing)
            return True

        elif self.conflict_strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            # Keep higher confidence finding
            if new_finding.confidence < existing.confidence:
                # Don't add the new one
                return False
            else:
                # Remove existing, add new one
                self._findings[existing.topic].remove(existing)
                self._all_findings.remove(existing)
                return True

        # MANUAL and MERGE: keep both findings
        return True

    def _notify_subscribers(self, topic: str, finding: ContextFinding) -> None:
        """Notify subscribers of a new finding."""
        for callback in self._subscribers[topic]:
            try:
                callback(finding)
            except Exception as e:
                # Don't let subscriber errors break publishing
                print(f"Subscriber error: {e}")

    def _prune_expired(self) -> None:
        """Remove expired findings based on TTL."""
        if self.ttl_seconds is None:
            return

        now = time.time()
        cutoff = now - self.ttl_seconds

        # Filter all findings
        self._all_findings = [
            f for f in self._all_findings
            if f.timestamp >= cutoff
        ]

        # Rebuild topic index
        self._findings.clear()
        for finding in self._all_findings:
            self._findings[finding.topic].append(finding)

        # Prune conflicts referencing expired findings
        valid_ids = {f.finding_id for f in self._all_findings}
        self._conflicts = [
            (f1, f2) for f1, f2 in self._conflicts
            if f1.finding_id in valid_ids and f2.finding_id in valid_ids
        ]

    def _persist_finding(self, finding: ContextFinding) -> None:
        """Persist a single finding to storage."""
        if not self.storage_dir:
            return

        # Write to topic-specific file
        topic_file = self.storage_dir / f"{finding.topic}.jsonl"
        with open(topic_file, 'a') as f:
            f.write(json.dumps(finding.to_dict()) + '\n')
