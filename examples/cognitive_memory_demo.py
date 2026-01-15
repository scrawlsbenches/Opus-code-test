#!/usr/bin/env python3
"""
Cognitive Memory System - CEL as AI Agent Memory

Demonstrates using the Cognitive Event Lattice as a memory system:
- Episodic memory: What happened (Observations)
- Working memory: Current goals (Intentions)
- Learning: Errors and solutions (MetaCognition)
- Memory consolidation: Session summaries (Compaction)

Run: python examples/cognitive_memory_demo.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.cel.core.events import (
    CognitiveEvent,
    EventType,
    Observation,
    Intention,
    MetaCognition,
    Compaction,
)
from cortical.cel.core.references import MerkleRoot
from cortical.cel.stores import MemoryEventStore
from cortical.cel.container import LatticeBuilder


# =============================================================================
# Cognitive Memory System
# =============================================================================

class CognitiveMemory:
    """
    AI agent memory built on CEL.

    Memory types:
    - Episodic: Observations of what happened
    - Working: Intentions (current tasks/goals)
    - Semantic: Learned facts (extracted from experience)
    - Meta: Self-awareness (errors, confusion, insights)
    """

    def __init__(self):
        lattice = LatticeBuilder().with_storage(MemoryEventStore).build()
        self._store = lattice.event_store
        self._session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._last_event: Optional[str] = None

    def _append(self, event: CognitiveEvent) -> str:
        """Append event and track causal chain."""
        if self._last_event and not event.causal_parents:
            event = event.with_parent(self._last_event)
        root = self._store.append(event)
        self._last_event = root.value
        return root.value

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
        """Create an intention (task/goal)."""
        event = Intention(
            title=goal,
            priority=priority,
            concepts=self._extract_concepts(goal),
            metadata={'session': self._session_id},
        )
        return self._append(event)

    def complete_intention(self, intention_id: str, result: str) -> str:
        """Mark an intention as fulfilled."""
        event = Observation(
            content={
                'type': 'fulfillment',
                'intention_id': intention_id,
                'result': result,
            },
            causal_parents=(intention_id,),
            metadata={'session': self._session_id},
        )
        return self._append(event)

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

    # --- Querying Memory ---

    def recall_observations(self, concept: str = None) -> List[Dict]:
        """Recall observations, optionally filtered by concept."""
        results = []
        for event in self._store.iterate():
            if event.event_type != EventType.OBSERVATION:
                continue
            if concept and concept.lower() not in ' '.join(event.concepts).lower():
                continue
            results.append({
                'id': event.id[:12],
                'time': event.timestamp[:19],
                'content': event.content,
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
        }


# =============================================================================
# Demo: Simulated Agent Session
# =============================================================================

def run_demo():
    print("=" * 60)
    print("COGNITIVE MEMORY SYSTEM - AI Agent Memory Demo")
    print("=" * 60)

    memory = CognitiveMemory()

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

    # Capture the horizon before we find the solution
    pre_solution_horizon = memory._last_event

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

    # --- Phase 5: Temporal Query ---
    print("\n[Phase 5] Temporal query (state before solution)...")

    state_before = memory.state_at(pre_solution_horizon)
    state_now = memory.stats

    print(f"  Before solution: {state_before}")
    print(f"  After solution:  {state_now['by_type']}")

    # --- Phase 6: Session Handoff ---
    print("\n[Phase 6] Creating session handoff...")

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
    print(f"  By type:")
    for t, count in stats['by_type'].items():
        print(f"    {t}: {count}")

    print("\n" + "=" * 60)
    print("Demo complete. Memory persisted in CEL EventStore.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
