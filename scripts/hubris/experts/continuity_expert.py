#!/usr/bin/env python3
"""
Continuity Expert

Micro-expert specialized in predicting which context items need to be
restored when starting a new agent session or recovering from context loss.

This expert learns from cognitive continuity patterns:
- Which knowledge transfers (KTs) were referenced after recovery
- Which decisions influenced subsequent work
- Which tasks were continued vs. abandoned
- What handoff patterns led to successful continuations

The Continuity Expert embodies the insight:
"I am not my context - I am the PATTERN of reasoning preserved across contexts."
"""

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class ContinuityExpert(MicroExpert):
    """
    Expert for predicting which context items to restore during session recovery.

    Learns patterns from:
    - Session recovery sequences (what was read after context loss)
    - Knowledge transfer usage (which KTs were actually consulted)
    - Decision chains (which decisions influenced subsequent work)
    - Task continuation patterns (completed vs. abandoned after handoff)
    - Recency-weighted entity importance

    Model Data Structure:
        kt_utility: Dict[str, float] - KT ID -> utility score based on usage
        decision_influence: Dict[str, float] - Decision ID -> influence score
        task_continuation: Dict[str, float] - Task patterns -> continuation rate
        entity_cooccurrence: Dict[str, Dict[str, int]] - Entity co-reference patterns
        recovery_sequences: List[Dict] - Successful recovery session traces
        topic_to_entities: Dict[str, Dict[str, int]] - Topic keyword -> relevant entities
        recency_weights: Dict[str, float] - Entity ID -> recency-weighted importance
        total_recoveries: int - Total recovery sessions analyzed
    """

    def __init__(
        self,
        expert_id: str = "continuity_expert",
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize ContinuityExpert.

        Args:
            expert_id: Unique identifier (default: "continuity_expert")
            version: Expert version (default: "1.0.0")
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        # Remove expert_type from kwargs if present (avoids conflict when loading)
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="continuity",
            version=version,
            **kwargs
        )

        # Ensure model_data has required keys
        if not self.model_data:
            self.model_data = {
                'kt_utility': {},           # KT ID -> utility score
                'decision_influence': {},   # Decision ID -> influence score
                'task_continuation': {},    # Pattern -> continuation rate
                'entity_cooccurrence': {},  # Entity -> {other_entity -> count}
                'recovery_sequences': [],   # List of recovery session traces
                'topic_to_entities': {},    # Keyword -> {entity_id -> count}
                'recency_weights': {},      # Entity ID -> recency weight
                'total_recoveries': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Predict which context items to restore for a session.

        Args:
            context: Dictionary with:
                - query (str): Session topic, task description, or recovery context
                - entity_pool (List[Dict]): Available entities to choose from
                    Each entity should have: id, type, title/summary, created_at
                - current_time (str, optional): ISO timestamp for recency calculation
                - top_n (int, optional): Number of predictions (default: 10)
                - prefer_types (List[str], optional): Preferred entity types
                    e.g., ["kt", "decision", "task", "handoff"]
                - min_confidence (float, optional): Minimum confidence threshold

        Returns:
            ExpertPrediction with ranked (entity_id, confidence) pairs
        """
        query = context.get('query', '')
        entity_pool = context.get('entity_pool', [])
        current_time = context.get('current_time', datetime.now().isoformat())
        top_n = context.get('top_n', 10)
        prefer_types = context.get('prefer_types', [])
        min_confidence = context.get('min_confidence', 0.1)

        if not query or not entity_pool:
            return ExpertPrediction(
                expert_id=self.expert_id,
                expert_type=self.expert_type,
                items=[],
                metadata={'reason': 'Empty query or entity pool'}
            )

        # Calculate scores for each entity
        scored_entities = []
        query_tokens = self._tokenize(query)

        for entity in entity_pool:
            entity_id = entity.get('id', '')
            entity_type = entity.get('type', '')
            title = entity.get('title', entity.get('summary', ''))
            created_at = entity.get('created_at', '')

            if not entity_id:
                continue

            # Base score from multiple signals
            score = 0.0
            signals_used = []

            # 1. Historical utility score
            if entity_type == 'kt':
                utility = self.model_data['kt_utility'].get(entity_id, 0.0)
                if utility > 0:
                    score += utility * 0.3
                    signals_used.append('kt_utility')
            elif entity_type == 'decision':
                influence = self.model_data['decision_influence'].get(entity_id, 0.0)
                if influence > 0:
                    score += influence * 0.3
                    signals_used.append('decision_influence')

            # 2. Topic relevance (keyword matching)
            entity_tokens = self._tokenize(title)
            topic_scores = []
            for token in query_tokens:
                if token in self.model_data['topic_to_entities']:
                    entity_count = self.model_data['topic_to_entities'][token].get(entity_id, 0)
                    if entity_count > 0:
                        topic_scores.append(min(entity_count / 10.0, 1.0))
            if topic_scores:
                score += (sum(topic_scores) / len(topic_scores)) * 0.25
                signals_used.append('topic_relevance')

            # 3. Keyword overlap (query <-> entity title)
            overlap = len(set(query_tokens) & set(entity_tokens))
            if overlap > 0:
                overlap_score = overlap / max(len(query_tokens), 1)
                score += overlap_score * 0.2
                signals_used.append('keyword_overlap')

            # 4. Recency weight
            recency = self.model_data['recency_weights'].get(entity_id, 0.0)
            if recency > 0:
                score += recency * 0.15
                signals_used.append('recency')
            elif created_at:
                # Calculate recency from created_at if not in model
                recency = self._calculate_recency(created_at, current_time)
                score += recency * 0.1
                signals_used.append('calculated_recency')

            # 5. Type preference boost
            if prefer_types and entity_type in prefer_types:
                type_rank = prefer_types.index(entity_type)
                type_boost = 0.1 * (1 - type_rank / len(prefer_types))
                score += type_boost
                signals_used.append('type_preference')

            # 6. Co-occurrence with other high-scoring entities (done in post-processing)

            if score >= min_confidence:
                scored_entities.append({
                    'id': entity_id,
                    'score': score,
                    'signals': signals_used,
                    'type': entity_type
                })

        # Sort by score and take top_n
        scored_entities.sort(key=lambda x: x['score'], reverse=True)
        top_entities = scored_entities[:top_n]

        # Convert to prediction format
        items = [(e['id'], e['score']) for e in top_entities]

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=items,
            metadata={
                'signals_by_entity': {e['id']: e['signals'] for e in top_entities},
                'query_tokens': query_tokens,
                'pool_size': len(entity_pool),
                'filtered_count': len(scored_entities)
            }
        )

    def train(self, recovery_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the expert on recovery session data.

        Args:
            recovery_sessions: List of session records, each containing:
                - session_id: Unique session identifier
                - recovered_from: Previous session ID or context loss type
                - entities_read: List of entity IDs read during recovery
                - entities_used: List of entity IDs that influenced work
                - topic: Session topic or task description
                - outcome: "success", "partial", "abandoned"
                - duration_to_productive: Seconds until productive work began

        Returns:
            Training statistics
        """
        if not recovery_sessions:
            return {'status': 'no_data', 'sessions': 0}

        stats = {
            'sessions_processed': 0,
            'entities_learned': set(),
            'topics_indexed': set()
        }

        for session in recovery_sessions:
            entities_read = session.get('entities_read', [])
            entities_used = session.get('entities_used', [])
            topic = session.get('topic', '')
            outcome = session.get('outcome', 'unknown')

            # Calculate outcome weight
            outcome_weights = {
                'success': 1.0,
                'partial': 0.5,
                'abandoned': -0.2,
                'unknown': 0.1
            }
            weight = outcome_weights.get(outcome, 0.1)

            # Update entity utility scores
            for entity_id in entities_used:
                # Entities actually used get high utility
                entity_type = self._infer_entity_type(entity_id)
                if entity_type == 'kt':
                    current = self.model_data['kt_utility'].get(entity_id, 0.0)
                    self.model_data['kt_utility'][entity_id] = current + weight * 0.1
                elif entity_type == 'decision':
                    current = self.model_data['decision_influence'].get(entity_id, 0.0)
                    self.model_data['decision_influence'][entity_id] = current + weight * 0.1
                stats['entities_learned'].add(entity_id)

            # Update topic-to-entity mappings
            topic_tokens = self._tokenize(topic)
            for token in topic_tokens:
                if token not in self.model_data['topic_to_entities']:
                    self.model_data['topic_to_entities'][token] = {}
                for entity_id in entities_used:
                    current = self.model_data['topic_to_entities'][token].get(entity_id, 0)
                    self.model_data['topic_to_entities'][token][entity_id] = current + 1
                stats['topics_indexed'].add(token)

            # Update entity co-occurrence
            for i, ent1 in enumerate(entities_used):
                if ent1 not in self.model_data['entity_cooccurrence']:
                    self.model_data['entity_cooccurrence'][ent1] = {}
                for ent2 in entities_used[i + 1:]:
                    current = self.model_data['entity_cooccurrence'][ent1].get(ent2, 0)
                    self.model_data['entity_cooccurrence'][ent1][ent2] = current + 1

            # Store recovery sequence for pattern analysis
            self.model_data['recovery_sequences'].append({
                'topic': topic,
                'entities': entities_used,
                'outcome': outcome
            })

            self.model_data['total_recoveries'] += 1
            stats['sessions_processed'] += 1

        stats['entities_learned'] = len(stats['entities_learned'])
        stats['topics_indexed'] = len(stats['topics_indexed'])

        return stats

    def update_recency_weights(self, entity_timestamps: Dict[str, str], current_time: str) -> int:
        """
        Update recency weights for entities based on their timestamps.

        Args:
            entity_timestamps: Dict of entity_id -> ISO timestamp
            current_time: Current time as ISO timestamp

        Returns:
            Number of entities updated
        """
        updated = 0
        for entity_id, timestamp in entity_timestamps.items():
            weight = self._calculate_recency(timestamp, current_time)
            self.model_data['recency_weights'][entity_id] = weight
            updated += 1
        return updated

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for topic matching.

        Args:
            text: Input text

        Returns:
            List of lowercase tokens (words)
        """
        import re
        # Split on non-alphanumeric, lowercase, filter short words
        tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 2]

    def _calculate_recency(self, timestamp: str, current_time: str) -> float:
        """
        Calculate recency weight (0-1) based on time difference.

        Uses exponential decay with half-life of 7 days.

        Args:
            timestamp: Entity timestamp (ISO format)
            current_time: Current time (ISO format)

        Returns:
            Recency weight (0-1, higher = more recent)
        """
        try:
            # Parse timestamps
            entity_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.fromisoformat(current_time.replace('Z', '+00:00'))

            # Calculate days difference
            delta = (now - entity_time).total_seconds() / 86400.0  # days

            # Exponential decay with 7-day half-life
            half_life_days = 7.0
            weight = math.exp(-0.693 * delta / half_life_days)

            return max(0.0, min(1.0, weight))
        except (ValueError, TypeError):
            return 0.0

    def _infer_entity_type(self, entity_id: str) -> str:
        """
        Infer entity type from ID prefix.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity type string
        """
        if entity_id.startswith('KT-'):
            return 'kt'
        elif entity_id.startswith('D-'):
            return 'decision'
        elif entity_id.startswith('T-'):
            return 'task'
        elif entity_id.startswith('H-'):
            return 'handoff'
        elif entity_id.startswith('S-'):
            return 'sprint'
        return 'unknown'

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContinuityExpert':
        """
        Load ContinuityExpert from dict.

        Args:
            data: Dictionary representation

        Returns:
            ContinuityExpert instance
        """
        metrics = ExpertMetrics.from_dict(data['metrics']) if data.get('metrics') else None

        return cls(
            expert_id=data['expert_id'],
            version=data['version'],
            created_at=data.get('created_at'),
            trained_on_commits=data.get('trained_on_commits', 0),
            trained_on_sessions=data.get('trained_on_sessions', 0),
            git_hash=data.get('git_hash', ''),
            model_data=data.get('model_data'),
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )


def suggest_recovery_context(
    query: str,
    got_manager,
    expert: Optional[ContinuityExpert] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    High-level function to suggest context for session recovery.

    Args:
        query: Topic or task description for the session
        got_manager: GoTManager instance to query entities
        expert: Optional trained ContinuityExpert (uses defaults if None)
        top_n: Number of suggestions

    Returns:
        List of entity dicts with id, type, title, and confidence
    """
    if expert is None:
        expert = ContinuityExpert()

    # Build entity pool from GoT
    entity_pool = []

    # Get recent KTs
    try:
        kts = got_manager.list_knowledge_transfers()
        for kt in kts[:50]:  # Recent 50
            entity_pool.append({
                'id': kt.id,
                'type': 'kt',
                'title': kt.title,
                'summary': kt.summary,
                'created_at': kt.created_at
            })
    except Exception:
        pass

    # Get recent decisions
    try:
        decisions = got_manager.list_decisions()
        for d in decisions[:50]:
            entity_pool.append({
                'id': d.id,
                'type': 'decision',
                'title': d.decision,
                'summary': d.rationale,
                'created_at': d.created_at
            })
    except Exception:
        pass

    # Get active tasks
    try:
        tasks = got_manager.list_tasks(status='in_progress')
        tasks.extend(got_manager.list_tasks(status='pending')[:20])
        for t in tasks:
            entity_pool.append({
                'id': t.id,
                'type': 'task',
                'title': t.title,
                'summary': t.description,
                'created_at': t.created_at
            })
    except Exception:
        pass

    if not entity_pool:
        return []

    # Get predictions
    prediction = expert.predict({
        'query': query,
        'entity_pool': entity_pool,
        'top_n': top_n,
        'prefer_types': ['kt', 'decision', 'task', 'handoff']
    })

    # Build result with full entity info
    result = []
    entity_lookup = {e['id']: e for e in entity_pool}
    for entity_id, confidence in prediction.items:
        entity = entity_lookup.get(entity_id, {})
        result.append({
            'id': entity_id,
            'type': entity.get('type', 'unknown'),
            'title': entity.get('title', ''),
            'confidence': confidence,
            'signals': prediction.metadata.get('signals_by_entity', {}).get(entity_id, [])
        })

    return result


if __name__ == '__main__':
    # Example usage
    expert = ContinuityExpert()

    # Simulate some training data
    training_data = [
        {
            'session_id': 'S001',
            'recovered_from': 'context_loss',
            'entities_read': ['KT-001', 'D-001', 'T-001'],
            'entities_used': ['KT-001', 'D-001'],
            'topic': 'cognitive continuity implementation',
            'outcome': 'success',
            'duration_to_productive': 120
        },
        {
            'session_id': 'S002',
            'recovered_from': 'session_handoff',
            'entities_read': ['KT-001', 'KT-002', 'T-002'],
            'entities_used': ['KT-002', 'T-002'],
            'topic': 'behavioral test development',
            'outcome': 'success',
            'duration_to_productive': 60
        }
    ]

    # Train
    stats = expert.train(training_data)
    print(f"Training stats: {stats}")

    # Predict
    entity_pool = [
        {'id': 'KT-001', 'type': 'kt', 'title': 'Cognitive Continuity Pattern'},
        {'id': 'KT-002', 'type': 'kt', 'title': 'Behavioral Testing Approach'},
        {'id': 'D-001', 'type': 'decision', 'title': 'Use GoT for context tracking'},
        {'id': 'T-001', 'type': 'task', 'title': 'Implement continuity bootstrap'}
    ]

    prediction = expert.predict({
        'query': 'continue cognitive continuity work',
        'entity_pool': entity_pool,
        'top_n': 5
    })

    print(f"\nPredictions for 'continue cognitive continuity work':")
    for entity_id, confidence in prediction.items:
        print(f"  {entity_id}: {confidence:.3f}")
