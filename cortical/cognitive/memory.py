#!/usr/bin/env python3
"""
Cognitive Memory System - CEL as AI Agent Memory

Uses the Cognitive Event Lattice as a persistent memory system for AI agents:
- Episodic memory: What happened (Observations)
- Working memory: Current goals (Intentions)
- Learning: Errors and solutions (MetaCognition)
- Memory consolidation: Session summaries (Compaction)
- Intent anchors: Sacred user requests that never decay
- Session hooks: Handoffs between sessions

Usage:
    from cortical.cognitive.memory import CognitiveMemory
    memory = CognitiveMemory.open()
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from cortical.cel.core.events import (
    CognitiveEvent,
    EventType,
    Observation,
    Intention,
    Fulfillment,
    MetaCognition,
    Compaction,
)
from cortical.cel.core.references import MerkleRoot
from cortical.cel.stores import MemoryEventStore
from cortical.cel.wisdom.dag import FileSystemEventStore
from cortical.cel.sanity.compaction import (
    TimeWindowCompactor,
    SemanticCompactor,
    CompactionResult,
)


# =============================================================================
# Cognitive Memory System
# =============================================================================

class CognitiveMemory:
    """
    AI agent memory built on CEL with persistent storage.

    Memory types:
    - Episodic: Observations of what happened
    - Working: Intentions (current tasks/goals)
    - Semantic: Learned facts (extracted from experience)
    - Meta: Self-awareness (errors, confusion, insights)

    Enhanced features:
    - Persistent storage via FileSystemEventStore
    - Concept indexing for O(1) lookups
    - Working memory tracking (pending vs completed intentions)
    - Associative recall by shared concepts
    - Importance scoring for memory prioritization

    Usage:
        # Create or load memory (persists to .cognitive/)
        memory = CognitiveMemory.open()

        # Or use in-memory only (for testing)
        memory = CognitiveMemory(persistent=False)
    """

    # Default storage location
    DEFAULT_STORAGE_PATH = Path(".cognitive")

    def __init__(
        self,
        session_id: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        persistent: bool = True,
    ):
        """
        Initialize cognitive memory.

        Args:
            session_id: Session identifier (auto-generated if None)
            storage_path: Where to persist events (default: .cognitive/)
            persistent: If True, use FileSystemEventStore. If False, use MemoryEventStore.
        """
        self._storage_path = Path(storage_path) if storage_path else self.DEFAULT_STORAGE_PATH
        self._persistent = persistent

        if persistent:
            self._store = FileSystemEventStore(self._storage_path / "events")
        else:
            self._store = MemoryEventStore()

        self._session_id = session_id or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._last_event: Optional[str] = None

        # Indexes for fast queries (rebuilt from events)
        self._concept_index: Dict[str, set] = {}  # concept -> event_ids
        self._pending_intentions: Dict[str, str] = {}  # event_id -> title
        self._importance_scores: Dict[str, float] = {}  # event_id -> importance
        self._preserved_events: Set[str] = set()  # events that should never be compacted

        # Rebuild indexes from existing events
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild in-memory indexes from persisted events."""
        for event in self._store.iterate():
            event_id = event.id

            # Track last event
            self._last_event = event_id

            # Rebuild concept index
            self._index_concepts(event_id, event.concepts)

            # Rebuild pending intentions
            if event.event_type == EventType.INTENTION:
                self._pending_intentions[event_id] = event.content.get('title', '')

            # Check for fulfillments that close intentions
            if event.event_type == EventType.FULFILLMENT:
                intention_id = event.content.get('intention_id')
                if intention_id:
                    self._pending_intentions.pop(intention_id, None)

            # Check for abandoned intentions
            if event.event_type == EventType.METACOGNITION:
                if event.content.get('observation_type') == 'abandoned_intention':
                    intention_id = event.content.get('metrics', {}).get('intention_id')
                    if intention_id:
                        self._pending_intentions.pop(intention_id, None)

            # Restore preserved events (intent anchors)
            if event.metadata.get('preserved'):
                self._preserved_events.add(event_id)

            # Restore importance from metadata, default 1.0
            self._importance_scores[event_id] = event.metadata.get('importance', 1.0)

    @classmethod
    def open(
        cls,
        storage_path: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
    ) -> 'CognitiveMemory':
        """
        Open or create a persistent cognitive memory.

        This is the recommended way to get a memory instance.
        Events persist across sessions.

        Args:
            storage_path: Where to store events (default: .cognitive/)
            session_id: Session identifier (auto-generated if None)

        Returns:
            CognitiveMemory instance with loaded events
        """
        return cls(
            session_id=session_id,
            storage_path=storage_path,
            persistent=True,
        )

    def _append(self, event: CognitiveEvent, importance: float = 1.0) -> str:
        """Append event and track causal chain."""
        if self._last_event and not event.causal_parents:
            event = event.with_parent(self._last_event)
        root = self._store.append(event)
        event_id = root.value
        self._last_event = event_id

        # Update indexes
        self._index_concepts(event_id, event.concepts)
        self._importance_scores[event_id] = importance

        return event_id

    def _index_concepts(self, event_id: str, concepts: tuple) -> None:
        """Add event to concept index for fast lookups."""
        for concept in concepts:
            normalized = concept.lower()
            if normalized not in self._concept_index:
                self._concept_index[normalized] = set()
            self._concept_index[normalized].add(event_id)

    def current_horizon(self) -> Optional[str]:
        """Get the ID of the most recent event (public API)."""
        return self._last_event

    # --- Episodic Memory (Observations) ---

    def observe(self, what: str, details: Dict[str, Any] = None) -> str:
        """Record an observation (something that happened)."""
        event = Observation(
            content={'observation': what, **(details or {})},
            concepts=self._extract_concepts(what),
            metadata={'session': self._session_id},
        )
        return self._append(event)

    def observe_user_request(self, request: str) -> str:
        """Record a user request."""
        return self.observe('user_request', {'request': request})

    def anchor_intent(self, prompt: str) -> str:
        """
        Create a sacred, never-decay user intent anchor.

        Intent anchors are:
        - Preserved from compaction (never deleted/merged)
        - Maximum importance (always surface in recovery)
        - The ground truth for "what the user asked for"

        Use this for user requests that should persist across all sessions
        and context compactions.

        Args:
            prompt: The user's request/intent

        Returns:
            Event ID of the anchor
        """
        # Create observation with special metadata
        event = Observation(
            content={
                'observation': 'intent_anchor',
                'prompt': prompt,
                'anchored_at': datetime.now(timezone.utc).isoformat(),
            },
            concepts=self._extract_concepts(prompt) + ('intent_anchor', 'user_request'),
            metadata={
                'session': self._session_id,
                'preserved': True,  # Signal to compaction: don't touch this
                'importance': 10.0,  # Maximum importance
                'anchor_type': 'user_intent',
            },
        )
        event_id = self._append(event, importance=10.0)

        # Mark as preserved (for in-memory tracking)
        self._preserved_events.add(event_id)

        return event_id

    def recall_intent_anchors(self) -> List[Dict]:
        """Get all intent anchors (sacred user requests)."""
        anchors = []
        for obs in self.recall_observations():
            if obs['content'].get('observation') == 'intent_anchor':
                anchors.append({
                    'id': obs['id'],
                    'prompt': obs['content'].get('prompt'),
                    'anchored_at': obs['content'].get('anchored_at'),
                })
        return anchors

    def observe_error(self, error: str, context: str = None) -> str:
        """Record an error encountered."""
        return self.observe('error_encountered', {
            'error': error,
            'context': context,
        })

    def observe_file_change(self, path: str, action: str) -> str:
        """Record a file modification."""
        return self.observe('file_change', {'path': path, 'action': action})

    # --- Working Memory (Intentions) ---

    def intend(self, goal: str, priority: str = 'medium') -> str:
        """Create an intention (task/goal) and track it as pending."""
        event = Intention(
            title=goal,
            priority=priority,
            concepts=self._extract_concepts(goal),
            metadata={'session': self._session_id},
        )
        # Higher importance for high-priority intentions
        importance = {'high': 2.0, 'medium': 1.0, 'low': 0.5}.get(priority, 1.0)
        event_id = self._append(event, importance=importance)

        # Track as pending
        self._pending_intentions[event_id] = goal
        return event_id

    def complete_intention(self, intention_id: str, result: str) -> str:
        """Mark an intention as fulfilled using proper Fulfillment event."""
        event = Fulfillment(
            intention_id=intention_id,
            result={'summary': result},
            metadata={'session': self._session_id},
            causal_parents=(intention_id,),
        )
        event_id = self._append(event, importance=1.5)  # Completions are important

        # Remove from pending
        self._pending_intentions.pop(intention_id, None)
        return event_id

    def pending_intentions(self) -> List[Dict]:
        """Get all pending (uncompleted) intentions - working memory."""
        results = []
        for eid, goal in self._pending_intentions.items():
            event = self._store.get(eid)
            # Priority is stored in content when retrieved from persistent store
            priority = 'medium'
            if event and hasattr(event, 'content') and isinstance(event.content, dict):
                priority = event.content.get('priority', 'medium')
            results.append({'id': eid, 'goal': goal, 'priority': priority})
        return results

    def abandon_intention(self, intention_id: str, reason: str) -> str:
        """Abandon an intention that won't be completed."""
        event = MetaCognition(
            observation_type='abandoned_intention',
            metrics={'intention_id': intention_id, 'reason': reason},
            conclusions=[f"Abandoned: {reason}"],
            metadata={'session': self._session_id},
            causal_parents=(intention_id,),
        )
        event_id = self._append(event)

        # Remove from pending
        self._pending_intentions.pop(intention_id, None)
        return event_id

    # --- Meta-Cognition (Self-Awareness) ---

    def reflect(self, insight: str, category: str = 'insight') -> str:
        """Record a meta-cognitive insight."""
        event = MetaCognition(
            observation_type=category,
            conclusions=[insight],
            metadata={'session': self._session_id},
        )
        return self._append(event)

    def learn(self, problem: str, solution: str) -> str:
        """Record a learned solution to a problem."""
        event = MetaCognition(
            observation_type='learning',
            metrics={'problem': problem, 'solution': solution},
            conclusions=[f"Learned: {solution}"],
            metadata={'session': self._session_id},
        )
        return self._append(event)

    def note_confusion(self, about: str) -> str:
        """Record confusion (useful for asking clarifying questions)."""
        return self.reflect(f"Confused about: {about}", category='confusion')

    # --- Memory Consolidation ---

    def summarize_session(self, summary: str) -> str:
        """Create a session summary (compaction)."""
        # Get all events from this session
        session_events = [
            e.id for e in self._store.iterate()
            if e.metadata.get('session') == self._session_id
        ]

        # Get current merkle root for verification
        latest = self._store.latest()
        merkle_root = latest.value if latest else "genesis"

        event = Compaction(
            compressed_events=session_events[:10],  # Reference first 10
            snapshot={
                'summary': summary,
                'event_count': len(session_events),
                'session': self._session_id,
            },
            preserved_merkle_root=merkle_root,
            metadata={'session': self._session_id},
        )
        return self._append(event)

    # --- Recovery Protocol (The Safety Net) ---

    def recover(self) -> str:
        """
        The safety net. Call this when confused/daydreaming.

        Synthesizes a recovery summary from:
        1. Intent anchors (sacred user requests - MOST important)
        2. Pending intentions (what's incomplete)
        3. Recent learnings (what was discovered)
        4. User requests (what was originally asked)
        5. Recent errors (what went wrong)

        Returns a formatted string that can restore context.
        """
        # Gather intent anchors (sacred, preserved user requests)
        intent_anchors = self.recall_intent_anchors()

        # Gather pending work
        pending = self.pending_intentions()

        # Gather learnings
        learnings = self.recall_learnings()[-5:]  # Last 5

        # Gather user requests (non-anchored)
        user_requests = []
        for obs in self.recall_observations():
            if obs['content'].get('observation') == 'user_request':
                user_requests.append(obs['content'].get('request', 'unknown'))
        user_requests = user_requests[-3:]  # Last 3

        # Gather recent errors
        errors = self.recall_errors()[-3:]  # Last 3

        # Build recovery summary
        lines = ["## Recovery Summary", ""]

        # Stats
        stats = self.stats
        lines.append(f"**Memory State:** {stats['total_events']} events, {stats['indexed_concepts']} concepts, {self.preserved_count} preserved")
        lines.append("")

        # Intent anchors FIRST (most sacred)
        if intent_anchors:
            lines.append(f"**Intent Anchors (Sacred):** {len(intent_anchors)}")
            for anchor in intent_anchors:
                lines.append(f"- {anchor['prompt']}")
            lines.append("")

        # Pending work
        if pending:
            lines.append(f"**Pending Work:** {len(pending)} tasks")
            for p in pending:
                lines.append(f"- [{p['priority']}] {p['goal']}")
            lines.append("")
        else:
            lines.append("**Pending Work:** None")
            lines.append("")

        # User requests (non-anchored, for context)
        if user_requests:
            lines.append("**Recent Requests:**")
            for req in user_requests:
                lines.append(f"- {req}")
            lines.append("")

        # Learnings
        if learnings:
            lines.append("**Recent Learnings:**")
            for l in learnings:
                lines.append(f"- {l['problem']} -> {l['solution']}")
            lines.append("")

        # Errors (if any)
        if errors:
            lines.append("**Recent Errors:**")
            for err in errors:
                lines.append(f"- {err.get('error', 'unknown')[:60]}")
            lines.append("")

        # Record that recovery happened
        self.reflect("Recovery protocol executed", category="recovery")

        return "\n".join(lines)

    def recall_user_requests(self) -> List[str]:
        """Get all user requests (for intent tracking)."""
        requests = []
        for obs in self.recall_observations():
            if obs['content'].get('observation') == 'user_request':
                requests.append(obs['content'].get('request', 'unknown'))
        return requests

    # --- Session Hooks (Handoff Between Sessions) ---

    def handoff(self, summary: str = None, focus: str = None) -> str:
        """
        Create a handoff for the next session.

        Call this at session end to capture state for continuation.
        The next session can check_handoff() to pick up where you left off.

        Args:
            summary: Optional summary of what was accomplished
            focus: Optional note about what to focus on next

        Returns:
            Event ID of the handoff
        """
        # Gather current state
        pending = self.pending_intentions()
        learnings = self.recall_learnings()[-3:]
        anchors = self.recall_intent_anchors()

        # Auto-generate summary if not provided
        if not summary:
            if pending:
                summary = f"Session ended with {len(pending)} pending tasks"
            else:
                summary = "Session ended with no pending tasks"

        # Create handoff event
        event = MetaCognition(
            observation_type='session_handoff',
            metrics={
                'pending_count': len(pending),
                'pending_tasks': [p['goal'] for p in pending[:5]],
                'recent_learnings': [l['problem'] for l in learnings],
                'intent_anchors': [a['prompt'] for a in anchors[:3]],
                'focus': focus,
            },
            conclusions=[summary],
            metadata={
                'session': self._session_id,
                'handoff_time': datetime.now(timezone.utc).isoformat(),
                'acknowledged': False,  # Will be set True when next session picks up
            },
        )
        return self._append(event)

    def check_handoff(self) -> Optional[Dict]:
        """
        Check for unacknowledged handoffs from previous sessions.

        Call this at session start to see if there's pending work
        from a previous session.

        Returns:
            Handoff info dict if found, None otherwise
        """
        # Find most recent unacknowledged handoff
        for event in reversed(list(self._store.iterate())):
            if event.event_type == EventType.METACOGNITION:
                if event.content.get('observation_type') == 'session_handoff':
                    if not event.metadata.get('acknowledged', False):
                        return {
                            'id': event.id,
                            'summary': event.content.get('conclusions', [''])[0],
                            'pending_tasks': event.content.get('metrics', {}).get('pending_tasks', []),
                            'focus': event.content.get('metrics', {}).get('focus'),
                            'handoff_time': event.metadata.get('handoff_time'),
                            'from_session': event.metadata.get('session'),
                        }
        return None

    def acknowledge_handoff(self, handoff_id: str) -> None:
        """
        Mark a handoff as acknowledged.

        Call this after processing a handoff so it won't surface again.
        """
        # Record acknowledgment (we can't modify the original event,
        # but we can record that we've seen it)
        self.reflect(
            f"Acknowledged handoff from previous session: {handoff_id[:12]}",
            category='handoff_ack',
        )
        # Note: In a real implementation, we might want to track acknowledged
        # handoffs in a separate index. For now, we rely on the handoff being
        # from a different session than the current one.

    def session_start(self) -> str:
        """
        Initialize a new session - check for handoffs and return status.

        Call this at the beginning of a session for a clean start.

        Returns:
            Formatted string with session status
        """
        lines = [f"## Session Start: {self._session_id}", ""]

        # Check for handoff
        handoff = self.check_handoff()
        if handoff:
            lines.append("**Handoff from previous session:**")
            lines.append(f"- {handoff['summary']}")
            if handoff['focus']:
                lines.append(f"- Focus: {handoff['focus']}")
            if handoff['pending_tasks']:
                lines.append(f"- Pending: {', '.join(handoff['pending_tasks'][:3])}")
            lines.append("")
            # Auto-acknowledge
            self.acknowledge_handoff(handoff['id'])
        else:
            lines.append("No pending handoff from previous session.")
            lines.append("")

        # Show current state
        stats = self.stats
        lines.append(f"**Memory State:** {stats['total_events']} events, {self.preserved_count} preserved")

        # Show intent anchors if any
        anchors = self.recall_intent_anchors()
        if anchors:
            lines.append(f"**Active Intent Anchors:** {len(anchors)}")
            for a in anchors[:3]:
                lines.append(f"- {a['prompt']}")

        return "\n".join(lines)

    # --- Querying Memory ---

    def recall_observations(self, concept: str = None) -> List[Dict]:
        """Recall observations, optionally filtered by concept (O(1) with index)."""
        if concept:
            # Fast path: use concept index
            event_ids = self._concept_index.get(concept.lower(), set())
            results = []
            for event_id in event_ids:
                event = self._store.get(event_id)
                if event and event.event_type == EventType.OBSERVATION:
                    results.append({
                        'id': event.id[:12],
                        'time': event.timestamp[:19],
                        'content': event.content,
                        'importance': self._importance_scores.get(event_id, 1.0),
                    })
            return sorted(results, key=lambda x: x['time'], reverse=True)

        # Full scan if no concept filter
        results = []
        for event in self._store.iterate():
            if event.event_type != EventType.OBSERVATION:
                continue
            results.append({
                'id': event.id[:12],
                'time': event.timestamp[:19],
                'content': event.content,
                'importance': self._importance_scores.get(event.id, 1.0),
            })
        return results

    def recall_learnings(self) -> List[Dict]:
        """Recall all learned solutions."""
        learnings = []
        for event in self._store.iterate():
            if event.event_type != EventType.METACOGNITION:
                continue
            if event.content.get('observation_type') == 'learning':
                learnings.append({
                    'problem': event.content.get('metrics', {}).get('problem', ''),
                    'solution': event.content.get('metrics', {}).get('solution', ''),
                })
        return learnings

    def recall_errors(self) -> List[Dict]:
        """Recall all errors encountered."""
        errors = []
        for event in self._store.iterate():
            if event.event_type != EventType.OBSERVATION:
                continue
            if event.content.get('observation') == 'error_encountered':
                errors.append(event.content)
        return errors

    def recall_by_concepts(self, concepts: List[str]) -> List[Dict]:
        """Associative recall: find memories that share any of the given concepts."""
        matching_ids: set = set()
        for concept in concepts:
            matching_ids.update(self._concept_index.get(concept.lower(), set()))

        results = []
        for event_id in matching_ids:
            event = self._store.get(event_id)
            if event:
                results.append({
                    'id': event.id[:12],
                    'type': event.event_type.name,
                    'time': event.timestamp[:19],
                    'concepts': event.concepts,
                    'importance': self._importance_scores.get(event_id, 1.0),
                })

        # Sort by importance (descending), then by time (most recent first)
        return sorted(results, key=lambda x: (-x['importance'], x['time']), reverse=True)

    def find_related(self, event_id: str, limit: int = 5) -> List[Dict]:
        """Find memories related to a given event through shared concepts."""
        event = self._store.get(event_id)
        if not event:
            return []

        # Find all events sharing concepts with this one
        related_ids: set = set()
        for concept in event.concepts:
            related_ids.update(self._concept_index.get(concept.lower(), set()))
        related_ids.discard(event_id)  # Don't include the source event

        results = []
        for rid in related_ids:
            related_event = self._store.get(rid)
            if related_event:
                # Calculate relevance based on shared concepts
                shared = set(c.lower() for c in event.concepts) & set(c.lower() for c in related_event.concepts)
                relevance = len(shared) / max(len(event.concepts), 1)
                results.append({
                    'id': related_event.id[:12],
                    'type': related_event.event_type.name,
                    'shared_concepts': list(shared),
                    'relevance': relevance,
                    'importance': self._importance_scores.get(rid, 1.0),
                })

        # Sort by relevance * importance
        results.sort(key=lambda x: x['relevance'] * x['importance'], reverse=True)
        return results[:limit]

    def state_at(self, horizon_id: str) -> Dict:
        """Get memory state at a specific point in time."""
        observations = 0
        intentions = 0
        learnings = 0

        for event in self._store.iterate(to_event=horizon_id):
            if event.event_type == EventType.OBSERVATION:
                observations += 1
            elif event.event_type == EventType.INTENTION:
                intentions += 1
            elif event.event_type == EventType.METACOGNITION:
                learnings += 1

        return {
            'observations': observations,
            'intentions': intentions,
            'learnings': learnings,
        }

    def _extract_concepts(self, text: str) -> tuple:
        """Extract key concepts from text for indexing."""
        # Simple extraction - in production would use NLP
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'and', 'in'}
        return tuple(w for w in words if w not in stopwords and len(w) > 3)

    @property
    def stats(self) -> Dict:
        """Get memory statistics."""
        by_type = {}
        for event in self._store.iterate():
            t = event.event_type.name
            by_type[t] = by_type.get(t, 0) + 1
        return {
            'total_events': self._store.count,
            'by_type': by_type,
            'session': self._session_id,
            'pending_intentions': len(self._pending_intentions),
            'indexed_concepts': len(self._concept_index),
        }

    def context_window(self, concepts: List[str] = None, limit: int = 10) -> List[Dict]:
        """
        Build a context window for the AI - most relevant recent memories.

        This is the key integration point: when the AI needs context,
        call this method to get the most relevant memories.
        """
        if concepts:
            # Associative recall by concepts
            memories = self.recall_by_concepts(concepts)
        else:
            # Just get recent memories sorted by importance
            memories = []
            for event in self._store.iterate():
                memories.append({
                    'id': event.id[:12],
                    'type': event.event_type.name,
                    'time': event.timestamp[:19],
                    'concepts': event.concepts,
                    'importance': self._importance_scores.get(event.id, 1.0),
                })

        # Sort by importance * recency
        memories.sort(key=lambda x: x['importance'], reverse=True)
        return memories[:limit]

    # --- Compaction (Memory Consolidation) ---

    def compact(
        self,
        strategy: str = 'semantic',
        min_age_days: int = 7,
    ) -> CompactionResult:
        """
        Compact memory using CEL's compaction infrastructure.

        Preserved events (intent anchors, learnings) are never compacted.

        Args:
            strategy: 'semantic' (concept overlap) or 'time' (age-based)
            min_age_days: Only compact events older than this

        Returns:
            CompactionResult with statistics
        """
        if strategy == 'time':
            compactor = TimeWindowCompactor(
                self._store,
                window_size=timedelta(hours=24),
                min_age=timedelta(days=min_age_days),
            )
        else:  # semantic
            compactor = SemanticCompactor(
                self._store,
                similarity_threshold=0.8,
                min_group_size=3,
            )

        # Mark all preserved events as non-compactable
        for event_id in self._preserved_events:
            compactor.preserve(event_id)

        # Also preserve all learnings (they're valuable)
        for event in self._store.iterate():
            if event.event_type == EventType.METACOGNITION:
                if event.content.get('observation_type') == 'learning':
                    compactor.preserve(event.id)

        # Execute compaction
        result = compactor.compact()

        # Record that compaction happened
        self.reflect(
            f"Memory compacted: {result.original_count} -> {result.compacted_count} events "
            f"({result.compression_ratio:.1%} ratio)",
            category='compaction',
        )

        return result

    def should_compact(self) -> bool:
        """Check if memory compaction is recommended."""
        total = sum(1 for _ in self._store.iterate())
        # Recommend if >100 events
        return total > 100

    @property
    def preserved_count(self) -> int:
        """Number of preserved (non-compactable) events."""
        return len(self._preserved_events)

    # --- Mega Prompt Generation (Consolidated Wisdom) ---

    def generate_mega_prompt(self, include_workflow: bool = True) -> str:
        """
        Generate a consolidated cognitive prompt from all learnings and insights.

        This creates a reusable summary that can be:
        - Injected into CLAUDE.md
        - Used as context for future sessions
        - Shared across projects

        Args:
            include_workflow: Include workflow patterns section

        Returns:
            Formatted markdown suitable for prompt injection
        """
        lines = [
            "# Cognitive Summary",
            "",
            f"*Generated from {self.stats['total_events']} events, "
            f"{len(self.recall_learnings())} learnings*",
            "",
        ]

        # Intent Anchors (Sacred User Requests)
        anchors = self.recall_intent_anchors()
        if anchors:
            lines.append("## Intent Anchors (Sacred)")
            lines.append("")
            lines.append("These are the core user requests that drive all work:")
            lines.append("")
            for anchor in anchors:
                lines.append(f"- **{anchor['prompt']}**")
            lines.append("")

        # Learnings - grouped by concept similarity
        learnings = self.recall_learnings()
        if learnings:
            lines.append("## Learnings")
            lines.append("")

            # Group learnings by shared concepts
            grouped = self._group_learnings_by_concept(learnings)

            for category, category_learnings in grouped.items():
                if category != "general":
                    lines.append(f"### {category.title()}")
                    lines.append("")

                for learning in category_learnings:
                    problem = learning['problem']
                    solution = learning['solution']
                    lines.append(f"- **{problem}**")
                    lines.append(f"  - {solution}")
                    lines.append("")

            # General learnings (no clear category)
            if "general" in grouped and grouped["general"]:
                lines.append("### General")
                lines.append("")
                for learning in grouped["general"]:
                    lines.append(f"- **{learning['problem']}** → {learning['solution']}")
                lines.append("")

        # Workflow patterns (if requested)
        if include_workflow:
            lines.append("## Workflow")
            lines.append("")
            lines.append("```")
            lines.append("SESSION START")
            lines.append("  ├─ Read CLAUDE.md (identity)")
            lines.append("  ├─ memory.session_start() - check handoffs")
            lines.append("  └─ Begin work")
            lines.append("")
            lines.append("DURING SESSION")
            lines.append("  ├─ anchor_intent() - sacred user requests")
            lines.append("  ├─ observe() / learn() - capture experience")
            lines.append("  ├─ intend() / complete() - track tasks")
            lines.append("  └─ recover() - if confused")
            lines.append("")
            lines.append("SESSION END")
            lines.append("  ├─ handoff(summary, focus) - prepare for next")
            lines.append("  └─ commit and push")
            lines.append("```")
            lines.append("")

        # Current state summary
        lines.append("## Current State")
        lines.append("")
        stats = self.stats
        lines.append(f"- **Events:** {stats['total_events']}")
        lines.append(f"- **Pending intentions:** {stats['pending_intentions']}")
        lines.append(f"- **Preserved (sacred):** {self.preserved_count}")
        lines.append(f"- **Indexed concepts:** {stats['indexed_concepts']}")
        lines.append("")

        # Record that mega prompt was generated
        self.reflect("Generated mega prompt for knowledge consolidation", category="mega_prompt")

        return "\n".join(lines)

    def _group_learnings_by_concept(self, learnings: List[Dict]) -> Dict[str, List[Dict]]:
        """Group learnings by their primary concept for better organization."""
        # Define concept categories
        categories = {
            'persistence': ['store', 'persist', 'save', 'load', 'file', 'storage'],
            'recovery': ['recover', 'restore', 'context', 'confused', 'daydream'],
            'workflow': ['session', 'handoff', 'start', 'end', 'hook'],
            'compaction': ['compact', 'compress', 'preserve', 'anchor'],
            'memory': ['memory', 'event', 'observe', 'learn', 'intent'],
        }

        grouped: Dict[str, List[Dict]] = {cat: [] for cat in categories}
        grouped['general'] = []

        for learning in learnings:
            problem_lower = learning['problem'].lower()
            solution_lower = learning['solution'].lower()
            combined = problem_lower + " " + solution_lower

            # Find best matching category
            best_category = 'general'
            best_score = 0

            for category, keywords in categories.items():
                score = sum(1 for kw in keywords if kw in combined)
                if score > best_score:
                    best_score = score
                    best_category = category

            grouped[best_category].append(learning)

        # Remove empty categories
        return {k: v for k, v in grouped.items() if v}

    def save_mega_prompt(self, path: str = None) -> str:
        """
        Generate and save mega prompt to a file.

        Args:
            path: Output path (default: .cognitive/mega_prompt.md)

        Returns:
            Path where file was saved
        """
        if path is None:
            path = str(self._storage_path / "mega_prompt.md")

        content = self.generate_mega_prompt()

        with open(path, 'w') as f:
            f.write(content)

        self.observe(f"Saved mega prompt to {path}")
        return path


# =============================================================================
# Demo: Simulated Agent Session
# =============================================================================

def run_demo():
    print("=" * 60)
    print("COGNITIVE MEMORY SYSTEM - AI Agent Memory Demo")
    print("=" * 60)

    # Use in-memory for demo (no disk writes)
    # For persistent memory across sessions, use:
    #   memory = CognitiveMemory.open()  # Stores in .cognitive/
    memory = CognitiveMemory(persistent=False)

    # --- Phase 1: User Request ---
    print("\n[Phase 1] Processing user request...")

    memory.observe_user_request("Fix the bug in the authentication module")
    task_id = memory.intend("Fix authentication bug", priority="high")

    # --- Phase 2: Investigation ---
    print("[Phase 2] Investigating...")

    memory.observe("Examined auth.py", {'file': 'auth.py', 'lines': 150})
    memory.observe_error(
        "TypeError: 'NoneType' has no attribute 'token'",
        context="auth.py:42 in validate_session()"
    )

    # Capture the horizon before we find the solution (using public API)
    pre_solution_horizon = memory.current_horizon()

    # --- Phase 3: Solution ---
    print("[Phase 3] Found and applied solution...")

    memory.learn(
        problem="NoneType error in validate_session",
        solution="Add null check before accessing token attribute"
    )
    memory.observe_file_change("auth.py", "modified")
    memory.complete_intention(task_id, "Fixed null check in validate_session")

    # --- Phase 4: Query Memory ---
    print("[Phase 4] Querying memory...")

    print(f"\n  Errors encountered: {len(memory.recall_errors())}")
    for err in memory.recall_errors():
        print(f"    - {err.get('error', 'unknown')[:50]}...")

    print(f"\n  Learnings recorded: {len(memory.recall_learnings())}")
    for learning in memory.recall_learnings():
        print(f"    - {learning['problem'][:30]} -> {learning['solution'][:30]}")

    # --- Phase 5: Working Memory ---
    print("\n[Phase 5] Working memory (pending intentions)...")

    pending = memory.pending_intentions()
    print(f"  Pending tasks: {len(pending)}")
    for p in pending:
        print(f"    - {p['goal'][:50]}")

    # --- Phase 6: Associative Recall ---
    print("\n[Phase 6] Associative recall by concept...")

    auth_memories = memory.recall_by_concepts(['authentication', 'auth.py'])
    print(f"  Memories related to 'authentication': {len(auth_memories)}")
    for mem in auth_memories[:3]:
        print(f"    - [{mem['type']}] concepts: {mem['concepts'][:3]}")

    # --- Phase 7: Temporal Query ---
    print("\n[Phase 7] Temporal query (state before solution)...")

    state_before = memory.state_at(pre_solution_horizon)
    state_now = memory.stats

    print(f"  Before solution: {state_before}")
    print(f"  After solution:  {state_now['by_type']}")

    # --- Phase 8: Context Window ---
    print("\n[Phase 8] Context window for AI...")

    # Use concepts that are actually in the indexed memories
    context = memory.context_window(concepts=['authentication', 'auth.py'], limit=5)
    print(f"  Top {len(context)} relevant memories for auth context:")
    for ctx in context:
        print(f"    - [{ctx['type']}] importance={ctx['importance']:.1f} concepts={ctx['concepts'][:2]}")

    # --- Phase 9: Session Handoff ---
    print("\n[Phase 9] Creating session handoff...")

    memory.summarize_session(
        "Fixed auth bug: added null check in validate_session(). "
        "User token was None when session expired."
    )

    # --- Final Stats ---
    print("\n" + "-" * 60)
    print("MEMORY STATISTICS")
    print("-" * 60)
    stats = memory.stats
    print(f"  Session: {stats['session']}")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Indexed concepts: {stats['indexed_concepts']}")
    print(f"  Pending intentions: {stats['pending_intentions']}")
    print(f"  By type:")
    for t, count in stats['by_type'].items():
        print(f"    {t}: {count}")

    print("\n" + "=" * 60)
    print("Demo complete. Memory persisted in CEL EventStore.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
