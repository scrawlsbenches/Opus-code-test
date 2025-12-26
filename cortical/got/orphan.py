"""
Orphan detection and auto-linking for Graph of Thought.

This module provides:
1. Orphan detection on task create - warns when tasks have no connections
2. Sprint assignment suggestions - recommends sprints based on context
3. Periodic orphan report - generates reports of unlinked entities
4. Auto-link based on context - suggests connections based on content similarity

Usage:
    from cortical.got.orphan import OrphanDetector

    detector = OrphanDetector(got_manager)

    # Check for orphans
    report = detector.generate_orphan_report()

    # Get suggestions for a task
    suggestions = detector.suggest_connections(task_id)

    # Auto-link orphans to current sprint
    linked = detector.auto_link_to_sprint(task_ids)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .api import GoTManager
    from .types import Task, Sprint, Edge

logger = logging.getLogger(__name__)


@dataclass
class OrphanReport:
    """Report of orphan entities in the GoT graph."""

    orphan_tasks: List[str] = field(default_factory=list)
    orphan_decisions: List[str] = field(default_factory=list)
    total_tasks: int = 0
    total_decisions: int = 0
    orphan_rate: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def orphan_count(self) -> int:
        """Total number of orphan entities."""
        return len(self.orphan_tasks) + len(self.orphan_decisions)

    @property
    def has_orphans(self) -> bool:
        """Whether there are any orphans."""
        return self.orphan_count > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'orphan_tasks': self.orphan_tasks,
            'orphan_decisions': self.orphan_decisions,
            'total_tasks': self.total_tasks,
            'total_decisions': self.total_decisions,
            'orphan_rate': self.orphan_rate,
            'orphan_count': self.orphan_count,
            'generated_at': self.generated_at,
        }


@dataclass
class ConnectionSuggestion:
    """A suggested connection for an entity."""

    source_id: str
    target_id: str
    edge_type: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'confidence': self.confidence,
            'reason': self.reason,
        }


@dataclass
class SprintSuggestion:
    """A suggested sprint for a task."""

    sprint_id: str
    sprint_title: str
    confidence: float
    reason: str
    is_current: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'sprint_id': self.sprint_id,
            'sprint_title': self.sprint_title,
            'confidence': self.confidence,
            'reason': self.reason,
            'is_current': self.is_current,
        }


class OrphanDetector:
    """
    Detects and manages orphan entities in the Graph of Thought.

    An "orphan" is an entity with no incoming or outgoing edges,
    meaning it's not connected to any sprint, epic, or other task.
    """

    # Keywords that suggest task relationships
    DEPENDENCY_KEYWORDS = {
        'requires', 'needs', 'depends', 'after', 'following', 'prerequisite'
    }
    BLOCKS_KEYWORDS = {
        'blocks', 'blocking', 'prevents', 'before', 'prior'
    }
    RELATED_KEYWORDS = {
        'related', 'similar', 'see also', 'cf', 'reference', 'like'
    }

    def __init__(self, manager: 'GoTManager'):
        """
        Initialize orphan detector.

        Args:
            manager: GoTManager instance
        """
        self.manager = manager

    def is_orphan(self, entity_id: str) -> bool:
        """
        Check if an entity is an orphan (has no edges).

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity has no connected edges
        """
        outgoing, incoming = self.manager.get_edges_for_task(entity_id)
        return len(outgoing) == 0 and len(incoming) == 0

    def find_orphan_tasks(self) -> List[str]:
        """
        Find all orphan tasks.

        Returns:
            List of task IDs with no connections
        """
        orphans = []
        tasks = self.manager.list_all_tasks()

        for task in tasks:
            if self.is_orphan(task.id):
                orphans.append(task.id)

        return orphans

    def generate_orphan_report(self) -> OrphanReport:
        """
        Generate a comprehensive orphan report.

        Returns:
            OrphanReport with statistics and orphan lists
        """
        tasks = self.manager.list_all_tasks()
        orphan_task_ids = []

        for task in tasks:
            if self.is_orphan(task.id):
                orphan_task_ids.append(task.id)

        total = len(tasks)
        orphan_count = len(orphan_task_ids)
        rate = (orphan_count / total * 100) if total > 0 else 0.0

        return OrphanReport(
            orphan_tasks=orphan_task_ids,
            orphan_decisions=[],  # TODO: Add decision tracking
            total_tasks=total,
            total_decisions=0,
            orphan_rate=rate,
        )

    def suggest_sprint(self, task_id: str) -> List[SprintSuggestion]:
        """
        Suggest sprints for a task based on context.

        Args:
            task_id: Task identifier

        Returns:
            List of SprintSuggestion sorted by confidence
        """
        task = self.manager.get_task(task_id)
        if task is None:
            return []

        suggestions = []

        # 1. Suggest current sprint (highest priority)
        current = self.manager.get_current_sprint()
        if current:
            suggestions.append(SprintSuggestion(
                sprint_id=current.id,
                sprint_title=current.title,
                confidence=0.9,
                reason="Current active sprint",
                is_current=True,
            ))

        # 2. Find sprints with similar tasks
        sprints = self.manager.list_sprints()
        task_keywords = self._extract_keywords(task.title + " " + task.description)

        for sprint in sprints:
            if current and sprint.id == current.id:
                continue  # Already added

            # Skip completed sprints
            if sprint.status == "completed":
                continue

            # Calculate similarity with sprint tasks
            sprint_tasks = self.manager.get_sprint_tasks(sprint.id)
            similarity = self._calculate_sprint_similarity(task, sprint_tasks)

            if similarity > 0.3:
                suggestions.append(SprintSuggestion(
                    sprint_id=sprint.id,
                    sprint_title=sprint.title,
                    confidence=similarity,
                    reason=f"Similar tasks in sprint ({int(similarity * 100)}% match)",
                    is_current=False,
                ))

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:5]  # Top 5

    def suggest_connections(self, task_id: str) -> List[ConnectionSuggestion]:
        """
        Suggest connections for a task based on content similarity.

        Args:
            task_id: Task identifier

        Returns:
            List of ConnectionSuggestion sorted by confidence
        """
        task = self.manager.get_task(task_id)
        if task is None:
            return []

        suggestions = []
        all_tasks = self.manager.list_all_tasks()
        task_keywords = self._extract_keywords(task.title + " " + task.description)

        for other in all_tasks:
            if other.id == task_id:
                continue

            # Calculate similarity
            other_keywords = self._extract_keywords(other.title + " " + other.description)
            similarity = self._keyword_similarity(task_keywords, other_keywords)

            if similarity < 0.2:
                continue

            # Determine edge type based on content analysis
            edge_type, reason = self._infer_edge_type(task, other)

            suggestions.append(ConnectionSuggestion(
                source_id=task_id,
                target_id=other.id,
                edge_type=edge_type,
                confidence=similarity,
                reason=reason,
            ))

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:10]  # Top 10

    def auto_link_to_sprint(
        self,
        task_ids: List[str],
        sprint_id: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Auto-link tasks to a sprint.

        Args:
            task_ids: List of task IDs to link
            sprint_id: Target sprint ID (uses current if None)

        Returns:
            List of (task_id, sprint_id) tuples that were linked
        """
        if sprint_id is None:
            current = self.manager.get_current_sprint()
            if current is None:
                logger.warning("No current sprint found for auto-linking")
                return []
            sprint_id = current.id

        linked = []
        for task_id in task_ids:
            # Check if already in a sprint
            outgoing, incoming = self.manager.get_edges_for_task(task_id)
            already_in_sprint = any(
                e.edge_type == "CONTAINS" and e.source_id.startswith("S-")
                for e in incoming
            )

            if already_in_sprint:
                continue

            try:
                self.manager.add_task_to_sprint(task_id, sprint_id)
                linked.append((task_id, sprint_id))
                logger.info(f"Linked task {task_id} to sprint {sprint_id}")
            except Exception as e:
                logger.warning(f"Failed to link task {task_id}: {e}")

        return linked

    def check_on_create(self, task_id: str) -> Dict:
        """
        Check a newly created task and provide suggestions.

        This is meant to be called immediately after task creation
        to warn about orphan status and suggest connections.

        Args:
            task_id: The newly created task ID

        Returns:
            Dictionary with warnings and suggestions
        """
        result = {
            'task_id': task_id,
            'is_orphan': True,  # New tasks are always orphans initially
            'warnings': [],
            'sprint_suggestions': [],
            'connection_suggestions': [],
        }

        # Get sprint suggestions
        sprint_suggestions = self.suggest_sprint(task_id)
        result['sprint_suggestions'] = [s.to_dict() for s in sprint_suggestions]

        if sprint_suggestions:
            best = sprint_suggestions[0]
            result['warnings'].append(
                f"Task created without sprint assignment. "
                f"Consider adding to '{best.sprint_title}' ({best.reason})"
            )
        else:
            result['warnings'].append(
                "Task created without sprint assignment. No active sprints found."
            )

        # Get connection suggestions
        connection_suggestions = self.suggest_connections(task_id)
        result['connection_suggestions'] = [c.to_dict() for c in connection_suggestions[:5]]

        if connection_suggestions:
            result['warnings'].append(
                f"Found {len(connection_suggestions)} potentially related tasks"
            )

        return result

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Normalize
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b[a-z]+\b', text)
        # Filter stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                      'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                      'did', 'will', 'would', 'could', 'should', 'may', 'might',
                      'this', 'that', 'these', 'those', 'it', 'its'}
        return {w for w in words if w not in stop_words and len(w) > 2}

    def _keyword_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """Calculate Jaccard similarity between keyword sets."""
        if not keywords1 or not keywords2:
            return 0.0
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        return intersection / union if union > 0 else 0.0

    def _calculate_sprint_similarity(
        self,
        task: 'Task',
        sprint_tasks: List['Task']
    ) -> float:
        """Calculate how similar a task is to tasks in a sprint."""
        if not sprint_tasks:
            return 0.0

        task_keywords = self._extract_keywords(task.title + " " + task.description)

        max_similarity = 0.0
        for sprint_task in sprint_tasks:
            other_keywords = self._extract_keywords(
                sprint_task.title + " " + sprint_task.description
            )
            similarity = self._keyword_similarity(task_keywords, other_keywords)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _infer_edge_type(
        self,
        source_task: 'Task',
        target_task: 'Task'
    ) -> Tuple[str, str]:
        """
        Infer the appropriate edge type between two tasks.

        Returns:
            Tuple of (edge_type, reason)
        """
        source_text = (source_task.title + " " + source_task.description).lower()
        target_text = (target_task.title + " " + target_task.description).lower()

        # Check for dependency keywords in source
        for keyword in self.DEPENDENCY_KEYWORDS:
            if keyword in source_text:
                return "DEPENDS_ON", f"Source mentions '{keyword}'"

        # Check for blocking keywords in source
        for keyword in self.BLOCKS_KEYWORDS:
            if keyword in source_text:
                return "BLOCKS", f"Source mentions '{keyword}'"

        # Default to related
        return "RELATES_TO", "Similar content"

    def get_orphan_summary(self) -> str:
        """
        Get a human-readable summary of orphan status.

        Returns:
            Formatted summary string
        """
        report = self.generate_orphan_report()

        lines = [
            "=" * 60,
            "ORPHAN DETECTION REPORT",
            "=" * 60,
            "",
            f"Total Tasks: {report.total_tasks}",
            f"Orphan Tasks: {len(report.orphan_tasks)}",
            f"Orphan Rate: {report.orphan_rate:.1f}%",
            "",
        ]

        if report.has_orphans:
            lines.append("Orphan Task IDs:")
            for task_id in report.orphan_tasks[:20]:  # Limit to 20
                task = self.manager.get_task(task_id)
                title = task.title if task else "Unknown"
                lines.append(f"  - {task_id}: {title[:50]}")

            if len(report.orphan_tasks) > 20:
                lines.append(f"  ... and {len(report.orphan_tasks) - 20} more")
        else:
            lines.append("No orphan tasks found!")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def check_orphan_on_create(manager: 'GoTManager', task_id: str) -> Dict:
    """
    Convenience function to check orphan status on task creation.

    Args:
        manager: GoTManager instance
        task_id: Task ID to check

    Returns:
        Check result dictionary
    """
    detector = OrphanDetector(manager)
    return detector.check_on_create(task_id)


def generate_orphan_report(manager: 'GoTManager') -> OrphanReport:
    """
    Convenience function to generate orphan report.

    Args:
        manager: GoTManager instance

    Returns:
        OrphanReport instance
    """
    detector = OrphanDetector(manager)
    return detector.generate_orphan_report()
