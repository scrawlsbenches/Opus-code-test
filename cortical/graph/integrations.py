"""
Cognitive Architecture Integration Adapters.

This module provides adapters for integrating the SemanticKnowledgeGraph
with CEL, GoT, WovenMind, PRISM, and SparkSLM.

Each adapter wraps the external system and provides a consistent interface
for the knowledge graph to interact with.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import time


# ============================================================================
# CEL Integration Adapter
# ============================================================================

@dataclass
class CELEvent:
    """Represents a CEL cognitive event."""
    event_id: str
    event_type: str  # 'document_added', 'graph_built', 'search_executed', etc.
    timestamp: datetime
    data: Dict[str, Any]
    parent_id: Optional[str] = None
    merkle_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'parent_id': self.parent_id,
            'merkle_hash': self.merkle_hash,
        }


class CELAdapter:
    """
    Adapter for CEL (Cognitive Event Lattice) integration.

    Provides event sourcing for knowledge graph operations.
    """

    def __init__(self):
        self._events: List[CELEvent] = []
        self._event_counter = 0
        self._last_event_id: Optional[str] = None

    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> CELEvent:
        """Log a new event."""
        import hashlib
        import uuid

        self._event_counter += 1
        event_id = f"evt_{uuid.uuid4().hex[:8]}"

        # Compute simple merkle hash
        content = f"{event_type}:{data}:{self._last_event_id or ''}"
        merkle_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        event = CELEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            parent_id=self._last_event_id,
            merkle_hash=merkle_hash,
        )

        self._events.append(event)
        self._last_event_id = event_id

        return event

    def get_events(
        self,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[CELEvent]:
        """Get events with optional filtering."""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def get_event_count(self) -> int:
        """Get total event count."""
        return len(self._events)

    def create_intention(self, title: str, description: str = "") -> CELEvent:
        """Create an intention event (for task-like goals)."""
        return self.log_event("intention_created", {
            "title": title,
            "description": description,
        })

    def create_observation(self, observation_type: str, details: Dict[str, Any]) -> CELEvent:
        """Create an observation event."""
        return self.log_event(f"observation_{observation_type}", details)

    def create_fulfillment(self, intention_id: str, result: Dict[str, Any]) -> CELEvent:
        """Create a fulfillment event (intention completed)."""
        return self.log_event("intention_fulfilled", {
            "intention_id": intention_id,
            "result": result,
        })


# ============================================================================
# GoT Integration Adapter
# ============================================================================

@dataclass
class LinkedTask:
    """A task linked to knowledge graph nodes."""
    task_id: str
    title: str
    description: str
    status: str  # 'pending', 'in_progress', 'completed'
    related_nodes: List[Any]  # GraphNode references
    related_query: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkedDecision:
    """A decision linked to expert consultation."""
    decision_id: str
    question: str
    options: List[str]
    chosen: Optional[str]
    rationale: str
    contributing_experts: List[str]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)


class GoTAdapter:
    """
    Adapter for GoT (Graph of Thought) integration.

    Provides task and decision tracking linked to knowledge graph.
    """

    def __init__(self):
        self._tasks: Dict[str, LinkedTask] = {}
        self._decisions: Dict[str, LinkedDecision] = {}
        self._task_counter = 0
        self._decision_counter = 0

    def create_task(
        self,
        title: str,
        related_nodes: List[Any],
        related_query: str,
        description: str = "",
    ) -> LinkedTask:
        """Create a task linked to graph nodes."""
        import uuid

        self._task_counter += 1
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        task = LinkedTask(
            task_id=task_id,
            title=title,
            description=description,
            status="pending",
            related_nodes=related_nodes,
            related_query=related_query,
        )

        self._tasks[task_id] = task
        return task

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed."""
        if task_id in self._tasks:
            self._tasks[task_id].status = "completed"
            self._tasks[task_id].metadata['result'] = result
            return True
        return False

    def create_decision(
        self,
        question: str,
        chosen: str,
        contributing_experts: List[str],
        confidence: float,
        rationale: str = "",
        options: Optional[List[str]] = None,
    ) -> LinkedDecision:
        """Create a decision record."""
        import uuid

        self._decision_counter += 1
        decision_id = f"dec_{uuid.uuid4().hex[:8]}"

        decision = LinkedDecision(
            decision_id=decision_id,
            question=question,
            options=options or [],
            chosen=chosen,
            rationale=rationale,
            contributing_experts=contributing_experts,
            confidence=confidence,
        )

        self._decisions[decision_id] = decision
        return decision

    def get_tasks(self, status: Optional[str] = None) -> List[LinkedTask]:
        """Get tasks with optional status filter."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def get_decisions(self) -> List[LinkedDecision]:
        """Get all decisions."""
        return list(self._decisions.values())


# ============================================================================
# WovenMind Integration Adapter
# ============================================================================

class ThinkingMode(Enum):
    """Thinking modes for dual-process cognition."""
    FAST = auto()  # System 1 - Hive
    SLOW = auto()  # System 2 - Cortex
    AUTO = auto()  # Automatic switching


@dataclass
class WovenMindResult:
    """Result from WovenMind processing."""
    mode: str  # 'FAST' or 'SLOW'
    activations: Dict[str, float]
    abstractions: List[str]
    explored_concepts: List[str]
    surprise_level: float
    processing_time_ms: float


@dataclass
class ConsolidationResult:
    """Result from memory consolidation."""
    patterns_transferred: int
    abstractions_formed: int
    decay_applied: bool
    high_frequency_patterns: List[str]


class WovenMindAdapter:
    """
    Adapter for WovenMind (dual-process cognition) integration.

    Provides System 1 (fast, pattern-based) and System 2 (slow, deliberate)
    processing modes with automatic switching based on surprise.
    """

    def __init__(self, surprise_threshold: float = 0.3):
        self._surprise_threshold = surprise_threshold
        self._current_mode = ThinkingMode.AUTO
        self._hive_patterns: Dict[str, int] = {}  # pattern -> frequency
        self._cortex_abstractions: Dict[str, List[str]] = {}  # abstraction -> components
        self._training_data: List[str] = []

    def train(self, text: str) -> None:
        """Train on text patterns."""
        self._training_data.append(text)

        # Extract and count patterns (simplified n-grams)
        words = text.lower().split()
        for i in range(len(words) - 1):
            pattern = f"{words[i]} {words[i+1]}"
            self._hive_patterns[pattern] = self._hive_patterns.get(pattern, 0) + 1

        for i in range(len(words) - 2):
            pattern = f"{words[i]} {words[i+1]} {words[i+2]}"
            self._hive_patterns[pattern] = self._hive_patterns.get(pattern, 0) + 1

    def process(
        self,
        context: List[str],
        mode: Optional[ThinkingMode] = None,
    ) -> WovenMindResult:
        """Process context through dual-process cognition."""
        start_time = time.time()

        # Calculate surprise based on pattern recognition
        text = " ".join(context).lower()
        words = text.split()

        recognized_patterns = 0
        total_patterns = 0

        for i in range(len(words) - 1):
            pattern = f"{words[i]} {words[i+1]}"
            total_patterns += 1
            if pattern in self._hive_patterns:
                recognized_patterns += 1

        surprise = 1.0 - (recognized_patterns / max(total_patterns, 1))

        # Determine mode
        if mode is None or mode == ThinkingMode.AUTO:
            actual_mode = ThinkingMode.SLOW if surprise > self._surprise_threshold else ThinkingMode.FAST
        else:
            actual_mode = mode

        # Process based on mode
        activations = {}
        abstractions = []
        explored_concepts = []

        if actual_mode == ThinkingMode.FAST:
            # Fast processing: activate known patterns
            for pattern, freq in self._hive_patterns.items():
                if pattern in text:
                    activations[pattern] = min(1.0, freq / 10)
        else:
            # Slow processing: explore and abstract
            for word in words:
                explored_concepts.append(word)

            # Look for abstraction opportunities
            for pattern, freq in self._hive_patterns.items():
                if freq >= 3:  # High frequency = potential abstraction
                    abstraction = f"concept:{pattern.replace(' ', '_')}"
                    abstractions.append(abstraction)

        processing_time = (time.time() - start_time) * 1000

        return WovenMindResult(
            mode="SLOW" if actual_mode == ThinkingMode.SLOW else "FAST",
            activations=activations,
            abstractions=abstractions,
            explored_concepts=explored_concepts,
            surprise_level=surprise,
            processing_time_ms=processing_time,
        )

    def consolidate(self) -> ConsolidationResult:
        """Consolidate patterns (memory "sleep")."""
        patterns_transferred = 0
        abstractions_formed = 0

        # Transfer high-frequency patterns to abstractions
        high_freq_patterns = []
        for pattern, freq in self._hive_patterns.items():
            if freq >= 5:
                high_freq_patterns.append(pattern)
                # Form abstraction
                abstraction = pattern.replace(" ", "_")
                if abstraction not in self._cortex_abstractions:
                    self._cortex_abstractions[abstraction] = pattern.split()
                    abstractions_formed += 1
                patterns_transferred += 1

        # Apply decay to low-frequency patterns
        decay_threshold = 2
        patterns_to_decay = [p for p, f in self._hive_patterns.items() if f < decay_threshold]
        for pattern in patterns_to_decay[:len(patterns_to_decay) // 2]:  # Decay half
            del self._hive_patterns[pattern]

        return ConsolidationResult(
            patterns_transferred=patterns_transferred,
            abstractions_formed=abstractions_formed,
            decay_applied=True,
            high_frequency_patterns=high_freq_patterns,
        )

    def get_surprise_threshold(self) -> float:
        """Get the surprise threshold."""
        return self._surprise_threshold

    def set_surprise_threshold(self, threshold: float) -> None:
        """Set the surprise threshold."""
        self._surprise_threshold = threshold


# ============================================================================
# PRISM Integration Adapter
# ============================================================================

@dataclass
class AttentionResult:
    """Result from attention computation."""
    focus_weights: Dict[str, float]
    attended_nodes: List[str]
    attention_mode: str  # 'who', 'what', 'where', 'when', 'why', 'how'


class PRISMAdapter:
    """
    Adapter for PRISM (Plasticity, Reasoning, Intelligence, Semantics, Mechanisms).

    Provides attention mechanisms and synaptic plasticity.
    """

    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.01):
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._connection_strengths: Dict[Tuple[str, str], float] = {}
        self._attention_history: List[AttentionResult] = []
        self._activation_counts: Dict[str, int] = {}

    def compute_attention(
        self,
        query: str,
        candidates: List[str],
        focus: Optional[str] = None,
    ) -> AttentionResult:
        """Compute attention weights for candidates."""
        query_words = set(query.lower().split())

        weights = {}
        for candidate in candidates:
            # Simple attention: word overlap
            candidate_words = set(candidate.lower().split())
            overlap = len(query_words & candidate_words)
            base_weight = overlap / max(len(query_words), 1)

            # Apply focus boost
            if focus and focus.lower() in candidate.lower():
                base_weight *= 1.5

            # Apply learned strength
            for word in query_words:
                key = (word, candidate)
                if key in self._connection_strengths:
                    base_weight *= self._connection_strengths[key]

            weights[candidate] = min(1.0, base_weight)

        # Normalize
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

        # Get top attended
        sorted_candidates = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        attended = [c for c, w in sorted_candidates if w > 0.1]

        result = AttentionResult(
            focus_weights=weights,
            attended_nodes=attended,
            attention_mode=focus or "general",
        )
        self._attention_history.append(result)

        return result

    def strengthen_connection(self, source: str, target: str, amount: float = 1.0) -> float:
        """Strengthen a synaptic connection (Hebbian learning)."""
        key = (source, target)
        current = self._connection_strengths.get(key, 1.0)
        new_strength = current + (amount * self._learning_rate)
        self._connection_strengths[key] = min(5.0, new_strength)  # Cap at 5x
        return self._connection_strengths[key]

    def record_activation(self, node_id: str) -> None:
        """Record that a node was activated."""
        self._activation_counts[node_id] = self._activation_counts.get(node_id, 0) + 1

    def apply_decay(self) -> int:
        """Apply decay to all connections."""
        decayed = 0
        for key in list(self._connection_strengths.keys()):
            self._connection_strengths[key] *= (1 - self._decay_rate)
            if self._connection_strengths[key] < 0.5:  # Prune weak connections
                del self._connection_strengths[key]
                decayed += 1
        return decayed

    def get_connection_strength(self, source: str, target: str) -> float:
        """Get the strength of a connection."""
        return self._connection_strengths.get((source, target), 1.0)

    def get_top_activated(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequently activated nodes."""
        sorted_nodes = sorted(
            self._activation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_nodes[:limit]


# ============================================================================
# SparkSLM Integration Adapter
# ============================================================================

@dataclass
class PrimeResult:
    """Result from SparkSLM priming."""
    primed_terms: List[str]
    predicted_next: List[Tuple[str, float]]  # (term, probability)
    topics: List[str]
    priming_confidence: float


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    is_anomalous: bool
    anomaly_score: float
    unusual_patterns: List[str]
    expected_patterns: List[str]


class SparkSLMAdapter:
    """
    Adapter for SparkSLM (Statistical Language Model) integration.

    Provides prediction, priming, and anomaly detection.
    """

    def __init__(self):
        self._ngram_counts: Dict[str, Dict[str, int]] = {}  # context -> {next -> count}
        self._unigram_counts: Dict[str, int] = {}
        self._total_tokens = 0
        self._trained = False

    def train(self, text: str) -> None:
        """Train on text."""
        words = text.lower().split()
        self._total_tokens += len(words)

        # Unigrams
        for word in words:
            self._unigram_counts[word] = self._unigram_counts.get(word, 0) + 1

        # Bigrams
        for i in range(len(words) - 1):
            context = words[i]
            next_word = words[i + 1]

            if context not in self._ngram_counts:
                self._ngram_counts[context] = {}
            self._ngram_counts[context][next_word] = self._ngram_counts[context].get(next_word, 0) + 1

        # Trigrams (context = 2 words)
        for i in range(len(words) - 2):
            context = f"{words[i]} {words[i+1]}"
            next_word = words[i + 2]

            if context not in self._ngram_counts:
                self._ngram_counts[context] = {}
            self._ngram_counts[context][next_word] = self._ngram_counts[context].get(next_word, 0) + 1

        self._trained = True

    def prime(self, query: str) -> PrimeResult:
        """Prime with a query, returning predicted expansions."""
        words = query.lower().split()

        primed_terms = []
        predicted_next = []
        topics = []

        # Get predictions based on last word
        if words:
            last_word = words[-1]
            if last_word in self._ngram_counts:
                next_counts = self._ngram_counts[last_word]
                total = sum(next_counts.values())
                for word, count in sorted(next_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    prob = count / total
                    predicted_next.append((word, prob))
                    primed_terms.append(word)

        # Get related terms (words that co-occur with query words)
        for word in words:
            if word in self._ngram_counts:
                for related in self._ngram_counts[word].keys():
                    if related not in primed_terms:
                        primed_terms.append(related)
                        if len(primed_terms) >= 10:
                            break

        # Extract topics (high-frequency terms)
        sorted_unigrams = sorted(self._unigram_counts.items(), key=lambda x: x[1], reverse=True)
        topics = [w for w, c in sorted_unigrams[:5] if c >= 2]

        confidence = 1.0 if predicted_next else 0.5

        return PrimeResult(
            primed_terms=primed_terms,
            predicted_next=predicted_next,
            topics=topics,
            priming_confidence=confidence,
        )

    def detect_anomalies(self, text: str) -> AnomalyResult:
        """Detect anomalous patterns in text."""
        if not self._trained:
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                unusual_patterns=[],
                expected_patterns=[],
            )

        words = text.lower().split()
        unusual_patterns = []
        expected_patterns = []
        total_transitions = 0
        unknown_transitions = 0

        for i in range(len(words) - 1):
            context = words[i]
            next_word = words[i + 1]
            total_transitions += 1

            if context in self._ngram_counts:
                if next_word in self._ngram_counts[context]:
                    expected_patterns.append(f"{context} {next_word}")
                else:
                    unusual_patterns.append(f"{context} {next_word}")
                    unknown_transitions += 1
            else:
                unusual_patterns.append(f"{context} {next_word}")
                unknown_transitions += 1

        anomaly_score = unknown_transitions / max(total_transitions, 1)
        is_anomalous = anomaly_score > 0.5

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            unusual_patterns=unusual_patterns[:5],
            expected_patterns=expected_patterns[:5],
        )

    def predict_next(self, context: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Predict the next word given context."""
        context = context.lower()

        if context in self._ngram_counts:
            next_counts = self._ngram_counts[context]
            total = sum(next_counts.values())
            predictions = [
                (word, count / total)
                for word, count in sorted(next_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            ]
            return predictions

        # Fall back to unigrams
        total = sum(self._unigram_counts.values()) or 1
        predictions = [
            (word, count / total)
            for word, count in sorted(self._unigram_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
        return predictions
