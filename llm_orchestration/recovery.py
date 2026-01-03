"""
Recovery System: Confusion Detection and State Restoration

This module addresses one of my most critical failure modes: confusion.
When I get confused, I can:
- Repeat failed approaches endlessly
- Contradict myself without noticing
- Generate plausible-sounding but wrong content
- Lose track of what I was doing

The key insight: I cannot reliably detect my own confusion from the inside.
I need external signals and structures to recognize and recover from it.

Recovery Layers:
    1. DETECTION - Identify confusion signals
    2. DIAGNOSIS - Understand what went wrong
    3. STABILIZATION - Stop harmful actions
    4. RESTORATION - Recover to known good state
    5. VERIFICATION - Confirm recovery was successful
    6. RESUMPTION - Continue with corrected approach

This integrates with cognitive_state.py for state management
and learning.py for improving recovery over time.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Protocol, TYPE_CHECKING
)
from pathlib import Path
import hashlib
import json

if TYPE_CHECKING:
    from cortical.reasoning.prism_got import SynapticMemoryGraph


# =============================================================================
# CONFUSION TYPES
# =============================================================================

class ConfusionType(Enum):
    """Categories of confusion I experience."""

    # Behavioral loops
    REPETITION_LOOP = auto()      # Repeating same failed action
    OSCILLATION = auto()          # Flip-flopping between approaches

    # State confusion
    CONTEXT_LOSS = auto()         # Lost track of what we're doing
    STATE_MISMATCH = auto()       # My beliefs don't match reality
    TEMPORAL_CONFUSION = auto()   # Confused about what's happened vs pending

    # Content problems
    CONTRADICTION = auto()        # Saying inconsistent things
    HALLUCINATION = auto()        # Generating false information
    PLACEHOLDER_CONTENT = auto()  # Generating obvious placeholders

    # Process failures
    BLOCKED = auto()              # Stuck, can't proceed
    WRONG_PATH = auto()           # Went down incorrect approach
    SCOPE_CREEP = auto()          # Lost sight of original goal

    # Unknown
    UNSPECIFIED = auto()          # Something's wrong but unclear what


class SeverityLevel(Enum):
    """How severe is the confusion?"""
    LOW = auto()       # Minor, self-correctable
    MEDIUM = auto()    # Needs intervention but not critical
    HIGH = auto()      # Blocking progress
    CRITICAL = auto()  # May cause damage if not stopped


@dataclass
class ConfusionSignal:
    """
    An indicator that confusion may be occurring.

    Signals are the raw observations that suggest something is wrong.
    Multiple signals may combine to indicate a specific confusion type.
    """
    signal_type: str           # What kind of signal
    description: str           # Human-readable description
    evidence: List[str]        # Specific evidence
    confidence: float          # How confident are we? (0-1)
    timestamp: datetime = field(default_factory=datetime.now)

    # Contextual info
    source: str = "unknown"    # What detected this
    related_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_type': self.signal_type,
            'description': self.description,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'related_action': self.related_action
        }


# =============================================================================
# CONFUSION DETECTION
# =============================================================================

class SignalDetector(Protocol):
    """Protocol for confusion signal detectors."""

    def detect(self, context: Dict[str, Any]) -> List[ConfusionSignal]:
        """Check for signals in the given context."""
        ...

    @property
    def signal_types(self) -> List[str]:
        """What types of signals does this detector find?"""
        ...


@dataclass
class ActionRecord:
    """Record of an action for pattern detection."""
    action_type: str
    target: str
    result: str  # success, failure, error
    timestamp: datetime
    parameters_hash: str  # Hash of parameters for dedup


class RepetitionDetector:
    """
    Detects when I'm repeating the same actions.

    This is one of the most common confusion patterns - trying the
    same thing that already failed.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.action_history: List[ActionRecord] = []

    @property
    def signal_types(self) -> List[str]:
        return ['repetition', 'failed_retry']

    def record_action(
        self,
        action_type: str,
        target: str,
        result: str,
        parameters: Dict[str, Any]
    ):
        """Record an action for analysis."""
        params_hash = hashlib.md5(
            json.dumps(parameters, sort_keys=True).encode()
        ).hexdigest()[:8]

        record = ActionRecord(
            action_type=action_type,
            target=target,
            result=result,
            timestamp=datetime.now(),
            parameters_hash=params_hash
        )
        self.action_history.append(record)

        # Keep window size
        if len(self.action_history) > self.window_size * 2:
            self.action_history = self.action_history[-self.window_size:]

    def detect(self, context: Dict[str, Any] = None) -> List[ConfusionSignal]:
        """Detect repetition patterns."""
        signals = []

        if len(self.action_history) < 3:
            return signals

        recent = self.action_history[-self.window_size:]

        # Check for exact repetition
        action_signatures = [
            f"{a.action_type}:{a.target}:{a.parameters_hash}"
            for a in recent
        ]

        from collections import Counter
        counts = Counter(action_signatures)
        for sig, count in counts.items():
            if count >= 3:
                signals.append(ConfusionSignal(
                    signal_type='repetition',
                    description=f"Action repeated {count} times",
                    evidence=[f"Signature: {sig}"],
                    confidence=min(0.9, 0.3 * count),
                    source='RepetitionDetector'
                ))

        # Check for failed retries (same action after failure)
        for i in range(1, len(recent)):
            if recent[i-1].result == 'failure':
                if (recent[i].action_type == recent[i-1].action_type and
                    recent[i].target == recent[i-1].target and
                    recent[i].parameters_hash == recent[i-1].parameters_hash):
                    signals.append(ConfusionSignal(
                        signal_type='failed_retry',
                        description="Retrying failed action without changes",
                        evidence=[
                            f"Failed: {recent[i-1].action_type} on {recent[i-1].target}",
                            f"Retried immediately with same parameters"
                        ],
                        confidence=0.8,
                        source='RepetitionDetector'
                    ))

        return signals


class ContradictionDetector:
    """
    Detects when I'm contradicting myself.

    Tracks statements/decisions and flags when new ones conflict
    with earlier ones.
    """

    def __init__(self):
        self.statements: List[Tuple[str, str, datetime]] = []  # (topic, content, time)

    @property
    def signal_types(self) -> List[str]:
        return ['contradiction', 'reversal']

    def record_statement(self, topic: str, content: str):
        """Record a statement on a topic."""
        self.statements.append((topic, content, datetime.now()))

        # Keep reasonable history
        if len(self.statements) > 100:
            self.statements = self.statements[-50:]

    def detect(self, context: Dict[str, Any] = None) -> List[ConfusionSignal]:
        """Detect contradictions."""
        signals = []

        # Group by topic
        by_topic: Dict[str, List[Tuple[str, datetime]]] = {}
        for topic, content, timestamp in self.statements:
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append((content, timestamp))

        # Look for contradictions on same topic
        for topic, entries in by_topic.items():
            if len(entries) < 2:
                continue

            # Simple heuristic: if statements differ significantly, flag
            contents = [e[0] for e in entries]
            unique = set(contents)

            if len(unique) > 1:
                signals.append(ConfusionSignal(
                    signal_type='contradiction',
                    description=f"Conflicting statements on '{topic}'",
                    evidence=list(unique)[:3],
                    confidence=min(0.9, 0.2 * len(unique)),
                    source='ContradictionDetector'
                ))

        return signals


class ProgressDetector:
    """
    Detects when progress has stalled.

    Monitors goal completion and flags when we're stuck.
    """

    def __init__(self, stall_threshold: timedelta = timedelta(minutes=5)):
        self.stall_threshold = stall_threshold
        self.last_progress: datetime = datetime.now()
        self.goals_completed: int = 0
        self.tasks_completed: int = 0

    @property
    def signal_types(self) -> List[str]:
        return ['stalled', 'blocked']

    def record_progress(self, progress_type: str):
        """Record that progress was made."""
        self.last_progress = datetime.now()
        if progress_type == 'goal':
            self.goals_completed += 1
        elif progress_type == 'task':
            self.tasks_completed += 1

    def detect(self, context: Dict[str, Any] = None) -> List[ConfusionSignal]:
        """Detect stalled progress."""
        signals = []

        time_since_progress = datetime.now() - self.last_progress
        if time_since_progress > self.stall_threshold:
            signals.append(ConfusionSignal(
                signal_type='stalled',
                description=f"No progress for {time_since_progress}",
                evidence=[
                    f"Last progress: {self.last_progress.isoformat()}",
                    f"Tasks completed: {self.tasks_completed}",
                    f"Goals completed: {self.goals_completed}"
                ],
                confidence=min(0.9, 0.1 * (time_since_progress.seconds // 60)),
                source='ProgressDetector'
            ))

        return signals


class StateVerifier:
    """
    Verifies that my beliefs match external reality.

    This is crucial - I can have incorrect beliefs about:
    - File contents
    - System state
    - What's been accomplished
    - What's pending
    """

    def __init__(self):
        self.beliefs: Dict[str, Tuple[Any, datetime]] = {}  # topic -> (belief, when_formed)
        self.verifiers: Dict[str, Callable[[], Any]] = {}  # topic -> verify_fn

    @property
    def signal_types(self) -> List[str]:
        return ['state_mismatch', 'stale_belief']

    def register_belief(self, topic: str, belief: Any):
        """Record a belief I hold."""
        self.beliefs[topic] = (belief, datetime.now())

    def register_verifier(self, topic: str, verifier: Callable[[], Any]):
        """Register a function that can check reality for a topic."""
        self.verifiers[topic] = verifier

    def detect(self, context: Dict[str, Any] = None) -> List[ConfusionSignal]:
        """Detect state mismatches."""
        signals = []

        for topic, (belief, formed_at) in self.beliefs.items():
            if topic in self.verifiers:
                try:
                    reality = self.verifiers[topic]()
                    if reality != belief:
                        signals.append(ConfusionSignal(
                            signal_type='state_mismatch',
                            description=f"Belief about '{topic}' doesn't match reality",
                            evidence=[
                                f"Believed: {belief}",
                                f"Reality: {reality}",
                                f"Belief formed: {formed_at.isoformat()}"
                            ],
                            confidence=0.9,
                            source='StateVerifier'
                        ))
                except Exception as e:
                    # Verification failed - could be a problem
                    signals.append(ConfusionSignal(
                        signal_type='state_mismatch',
                        description=f"Could not verify belief about '{topic}'",
                        evidence=[f"Error: {str(e)}"],
                        confidence=0.5,
                        source='StateVerifier'
                    ))

            # Check for stale beliefs
            belief_age = datetime.now() - formed_at
            if belief_age > timedelta(hours=1):
                signals.append(ConfusionSignal(
                    signal_type='stale_belief',
                    description=f"Belief about '{topic}' is {belief_age} old",
                    evidence=[f"Formed: {formed_at.isoformat()}"],
                    confidence=0.3,  # Low confidence - old isn't necessarily wrong
                    source='StateVerifier'
                ))

        return signals


class SynapticConfusionDetector:
    """
    Detects confusion using synaptic memory patterns.

    This detector bridges PRISM-GoT's synaptic memory with the recovery system,
    using activation patterns to detect cognitive confusion states that may
    not be obvious from action patterns alone.

    Detection capabilities:
    1. Activation loops: Same nodes repeatedly activated without progress
    2. Contradictory activations: Opposing concepts both strongly activated
    3. Stagnation: Activations fading without new learning
    4. Oscillation: Rapid switching between competing thought patterns

    Attributes:
        memory_graph: The synaptic memory graph to analyze
        loop_window: Number of recent activations to check for loops
        contradiction_threshold: Confidence threshold for contradictions
        stagnation_threshold: Minimum activation rate to avoid stagnation
    """

    def __init__(
        self,
        memory_graph: 'SynapticMemoryGraph',
        loop_window: int = 5,
        contradiction_threshold: float = 0.7,
        stagnation_threshold: float = 0.1
    ):
        """
        Initialize the synaptic confusion detector.

        Args:
            memory_graph: The synaptic memory graph to monitor
            loop_window: Window size for detecting activation loops
            contradiction_threshold: Threshold for detecting contradictions
            stagnation_threshold: Minimum activation rate (per minute)
        """
        self._memory = memory_graph
        self._loop_window = loop_window
        self._contradiction_threshold = contradiction_threshold
        self._stagnation_threshold = stagnation_threshold

        # Track activation sequences for loop detection
        self._activation_sequence: List[str] = []

    @property
    def signal_types(self) -> List[str]:
        return [
            'synaptic_loop',
            'synaptic_contradiction',
            'synaptic_stagnation',
            'synaptic_oscillation'
        ]

    def detect(self, context: Optional[Dict[str, Any]] = None) -> List[ConfusionSignal]:
        """
        Detect confusion from synaptic patterns.

        Args:
            context: Optional context with additional information

        Returns:
            List of confusion signals detected
        """
        signals = []

        # Check for activation loops
        loop_signal = self._detect_activation_loop()
        if loop_signal:
            signals.append(loop_signal)

        # Check for contradictory activations
        contradiction_signals = self._detect_contradictory_activations()
        signals.extend(contradiction_signals)

        # Check for stagnation
        stagnation_signal = self._detect_stagnation()
        if stagnation_signal:
            signals.append(stagnation_signal)

        # Check for oscillation
        oscillation_signal = self._detect_oscillation()
        if oscillation_signal:
            signals.append(oscillation_signal)

        return signals

    def record_activation(self, node_id: str):
        """
        Record a node activation for pattern tracking.

        Args:
            node_id: ID of the activated node
        """
        self._activation_sequence.append(node_id)

        # Keep bounded history
        max_history = self._loop_window * 10
        if len(self._activation_sequence) > max_history:
            self._activation_sequence = self._activation_sequence[-max_history:]

    def _detect_activation_loop(self) -> Optional[ConfusionSignal]:
        """
        Detect circular activation patterns (loops).

        Looks for repeated subsequences in recent activations,
        which indicate the reasoning is going in circles.

        Returns:
            ConfusionSignal if loop detected, None otherwise
        """
        if len(self._activation_sequence) < self._loop_window * 2:
            return None

        recent = self._activation_sequence[-self._loop_window * 2:]

        # Look for repeating patterns
        for pattern_length in range(2, self._loop_window + 1):
            for start in range(len(recent) - pattern_length * 2 + 1):
                pattern = recent[start:start + pattern_length]
                next_segment = recent[start + pattern_length:start + pattern_length * 2]

                if pattern == next_segment:
                    # Found a repeating pattern
                    node_contents = []
                    for node_id in pattern[:3]:  # Show first 3
                        if node_id in self._memory.nodes:
                            node_contents.append(
                                self._memory.nodes[node_id].content[:50]
                            )

                    return ConfusionSignal(
                        signal_type='synaptic_loop',
                        description=f'Activation loop detected: {pattern_length}-node pattern repeating',
                        evidence=[
                            f'Pattern: {pattern}',
                            f'Repeated {2} times in recent history',
                            f'Nodes: {node_contents}'
                        ],
                        confidence=min(0.9, 0.3 * pattern_length),
                        source='SynapticConfusionDetector'
                    )

        return None

    def _detect_contradictory_activations(self) -> List[ConfusionSignal]:
        """
        Detect contradictory activations (opposing concepts both strong).

        Looks for nodes with contradictory content that are both
        recently and strongly activated, indicating conflicting reasoning.

        Returns:
            List of contradiction signals
        """
        signals = []

        # Get recently active nodes (within last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        active_nodes = []

        for node_id, trace in self._memory.activation_traces.items():
            if not trace.history:
                continue

            last_activation = datetime.fromisoformat(trace.history[-1]['timestamp'])
            if last_activation >= cutoff:
                activation_freq = trace.get_frequency(window_minutes=60)
                if activation_freq > 0:
                    active_nodes.append((node_id, activation_freq))

        # Sort by frequency
        active_nodes.sort(key=lambda x: x[1], reverse=True)

        # Check for contradictions among top active nodes
        for i, (node_id_1, freq_1) in enumerate(active_nodes[:10]):
            for node_id_2, freq_2 in active_nodes[i+1:10]:
                # Check if these nodes are marked as contradictory
                contradiction_strength = self._check_contradiction(node_id_1, node_id_2)

                if contradiction_strength >= self._contradiction_threshold:
                    node_1 = self._memory.nodes.get(node_id_1)
                    node_2 = self._memory.nodes.get(node_id_2)

                    signals.append(ConfusionSignal(
                        signal_type='synaptic_contradiction',
                        description='Contradictory thoughts both strongly activated',
                        evidence=[
                            f'Node 1: {node_1.content[:50] if node_1 else node_id_1}',
                            f'Node 2: {node_2.content[:50] if node_2 else node_id_2}',
                            f'Activation rates: {freq_1:.2f}, {freq_2:.2f} per min',
                            f'Contradiction strength: {contradiction_strength:.2f}'
                        ],
                        confidence=min(0.9, contradiction_strength),
                        source='SynapticConfusionDetector'
                    ))

        return signals

    def _check_contradiction(self, node_id_1: str, node_id_2: str) -> float:
        """
        Check if two nodes represent contradictory concepts.

        Uses simple heuristics:
        1. Opposite node types (HYPOTHESIS vs REFUTATION)
        2. Negation in content
        3. Conflicting decisions

        Args:
            node_id_1: First node ID
            node_id_2: Second node ID

        Returns:
            Contradiction strength (0.0 to 1.0)
        """
        node_1 = self._memory.nodes.get(node_id_1)
        node_2 = self._memory.nodes.get(node_id_2)

        if not node_1 or not node_2:
            return 0.0

        strength = 0.0

        # Check node types (HYPOTHESIS vs EVIDENCE could indicate contradiction)
        from cortical.reasoning.graph_of_thought import NodeType
        if ((node_1.node_type == NodeType.HYPOTHESIS and
             node_2.node_type == NodeType.EVIDENCE) or
            (node_1.node_type == NodeType.EVIDENCE and
             node_2.node_type == NodeType.HYPOTHESIS)):
            # Check if one has negation - would indicate refuting evidence
            content_1_lower = node_1.content.lower()
            content_2_lower = node_2.content.lower()
            negation_words = ['not', 'no', 'never', 'dont', "don't", 'cannot', 'cant', "can't", 'false', 'wrong']
            has_neg_1 = any(word in content_1_lower for word in negation_words)
            has_neg_2 = any(word in content_2_lower for word in negation_words)
            if has_neg_1 or has_neg_2:
                strength += 0.4

        # Check for negation patterns
        content_1 = node_1.content.lower()
        content_2 = node_2.content.lower()

        negation_words = ['not', 'no', 'never', 'dont', "don't", 'cannot', 'cant', "can't"]
        has_negation_1 = any(word in content_1 for word in negation_words)
        has_negation_2 = any(word in content_2 for word in negation_words)

        if has_negation_1 != has_negation_2:
            # One has negation, other doesn't - possible contradiction
            # Check if they share similar words
            words_1 = set(content_1.split())
            words_2 = set(content_2.split())
            overlap = words_1 & words_2
            if len(overlap) > 2:
                strength += 0.3

        return min(strength, 1.0)

    def _detect_stagnation(self) -> Optional[ConfusionSignal]:
        """
        Detect stagnation (activations fading, no new learning).

        Checks if:
        1. Overall activation rate is declining
        2. No new nodes being created
        3. Edge weights are decaying without new strengthening

        Returns:
            ConfusionSignal if stagnation detected, None otherwise
        """
        # Calculate recent activation rate
        total_activations_recent = 0
        node_count = 0

        for trace in self._memory.activation_traces.values():
            freq = trace.get_frequency(window_minutes=10)
            total_activations_recent += freq
            node_count += 1

        if node_count == 0:
            return None

        avg_activation_rate = total_activations_recent / node_count

        # Check if below threshold
        if avg_activation_rate < self._stagnation_threshold:
            # Calculate average edge weight to see if connections are weakening
            if self._memory.synaptic_edges:
                avg_weight = sum(
                    e.weight for e in self._memory.synaptic_edges.values()
                ) / len(self._memory.synaptic_edges)
            else:
                avg_weight = 0.0

            return ConfusionSignal(
                signal_type='synaptic_stagnation',
                description='Synaptic activity is stagnating',
                evidence=[
                    f'Average activation rate: {avg_activation_rate:.3f} per min',
                    f'Threshold: {self._stagnation_threshold:.3f}',
                    f'Average edge weight: {avg_weight:.3f}',
                    f'Active nodes: {node_count}'
                ],
                confidence=min(0.8, (self._stagnation_threshold - avg_activation_rate) * 2),
                source='SynapticConfusionDetector'
            )

        return None

    def _detect_oscillation(self) -> Optional[ConfusionSignal]:
        """
        Detect oscillation (rapid switching between competing patterns).

        Looks for alternating activation of mutually exclusive options,
        which indicates indecision or flip-flopping.

        Returns:
            ConfusionSignal if oscillation detected, None otherwise
        """
        if len(self._activation_sequence) < 6:
            return None

        recent = self._activation_sequence[-10:]

        # Look for ABAB or ABCABC patterns
        for pattern_len in [2, 3]:
            if len(recent) < pattern_len * 2:
                continue

            pattern_a = recent[-pattern_len * 2:-pattern_len]
            pattern_b = recent[-pattern_len:]

            if pattern_a == pattern_b:
                # Possible oscillation
                oscillation_count = 1
                idx = len(recent) - pattern_len * 2

                while idx >= pattern_len:
                    prev_pattern = recent[idx - pattern_len:idx]
                    if prev_pattern == pattern_a:
                        oscillation_count += 1
                        idx -= pattern_len
                    else:
                        break

                if oscillation_count >= 2:
                    return ConfusionSignal(
                        signal_type='synaptic_oscillation',
                        description=f'Oscillating between {pattern_len} thought patterns',
                        evidence=[
                            f'Pattern: {pattern_a}',
                            f'Repeated {oscillation_count + 1} times',
                            f'Recent sequence: {recent}'
                        ],
                        confidence=min(0.85, 0.2 * oscillation_count),
                        source='SynapticConfusionDetector'
                    )

        return None


@dataclass
class ConfusionDiagnosis:
    """
    Complete diagnosis of a confusion state.

    Aggregates signals and determines:
    - What type of confusion
    - How severe
    - Likely cause
    - Recommended recovery action
    """
    confusion_type: ConfusionType
    severity: SeverityLevel
    signals: List[ConfusionSignal]
    likely_cause: str
    recommended_action: str

    # Aggregate confidence
    confidence: float = 0.0

    # When diagnosed
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'confusion_type': self.confusion_type.name,
            'severity': self.severity.name,
            'signals': [s.to_dict() for s in self.signals],
            'likely_cause': self.likely_cause,
            'recommended_action': self.recommended_action,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class ConfusionDiagnoser:
    """
    Analyzes signals to diagnose confusion.

    Takes signals from multiple detectors and synthesizes
    a coherent diagnosis.
    """

    def __init__(self):
        self.detectors: List[SignalDetector] = []

    def add_detector(self, detector: SignalDetector):
        """Add a detector to the diagnoser."""
        self.detectors.append(detector)

    def diagnose(self, context: Dict[str, Any] = None) -> Optional[ConfusionDiagnosis]:
        """
        Run all detectors and diagnose confusion.

        Returns None if no confusion detected.
        """
        all_signals: List[ConfusionSignal] = []

        for detector in self.detectors:
            signals = detector.detect(context)
            all_signals.extend(signals)

        if not all_signals:
            return None

        # Determine confusion type from signals
        signal_types = set(s.signal_type for s in all_signals)

        # Check for synaptic patterns first (they're more specific)
        if 'synaptic_loop' in signal_types:
            confusion_type = ConfusionType.REPETITION_LOOP
            likely_cause = "Circular reasoning pattern detected in synaptic activations"
            recommended_action = "PRUNE unsuccessful pathways, try alternative approach"

        elif 'synaptic_oscillation' in signal_types:
            confusion_type = ConfusionType.OSCILLATION
            likely_cause = "Rapid switching between competing thought patterns"
            recommended_action = "PRUNE oscillating pathways, commit to single approach"

        elif 'synaptic_contradiction' in signal_types:
            confusion_type = ConfusionType.CONTRADICTION
            likely_cause = "Contradictory concepts both strongly activated"
            recommended_action = "RESET synaptic state, reconcile conflicting thoughts"

        elif 'synaptic_stagnation' in signal_types:
            confusion_type = ConfusionType.BLOCKED
            likely_cause = "Synaptic activity declining without new learning"
            recommended_action = "REINFORCE successful pathways or try new approach"

        # Fall back to traditional detection
        elif 'repetition' in signal_types or 'failed_retry' in signal_types:
            confusion_type = ConfusionType.REPETITION_LOOP
            likely_cause = "Repeating actions without adjusting approach"
            recommended_action = "STOP current action, analyze why previous attempts failed"

        elif 'contradiction' in signal_types:
            confusion_type = ConfusionType.CONTRADICTION
            likely_cause = "Inconsistent mental model"
            recommended_action = "RELOAD state from checkpoint, reconcile beliefs"

        elif 'state_mismatch' in signal_types:
            confusion_type = ConfusionType.STATE_MISMATCH
            likely_cause = "Beliefs about state are incorrect"
            recommended_action = "VERIFY all assumptions against reality"

        elif 'stalled' in signal_types or 'blocked' in signal_types:
            confusion_type = ConfusionType.BLOCKED
            likely_cause = "Unable to make progress"
            recommended_action = "ESCALATE to higher level or request help"

        else:
            confusion_type = ConfusionType.UNSPECIFIED
            likely_cause = "Unknown confusion pattern"
            recommended_action = "PAUSE and report state to user"

        # Determine severity
        max_confidence = max(s.confidence for s in all_signals)
        if max_confidence >= 0.8:
            severity = SeverityLevel.HIGH
        elif max_confidence >= 0.5:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW

        # Critical if affecting core operations
        critical_types = {'state_mismatch', 'contradiction'}
        if signal_types.intersection(critical_types) and max_confidence >= 0.7:
            severity = SeverityLevel.CRITICAL

        return ConfusionDiagnosis(
            confusion_type=confusion_type,
            severity=severity,
            signals=all_signals,
            likely_cause=likely_cause,
            recommended_action=recommended_action,
            confidence=max_confidence
        )


# =============================================================================
# RECOVERY STRATEGIES
# =============================================================================

class RecoveryStrategy(Protocol):
    """Protocol for recovery strategies."""

    @property
    def name(self) -> str:
        """Name of this strategy."""
        ...

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        """What types of confusion this can recover from."""
        ...

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """
        Execute recovery.

        Returns True if recovery was successful.
        """
        ...


@dataclass
class RecoveryAction:
    """A single action taken during recovery."""
    action_type: str
    description: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    diagnosis: ConfusionDiagnosis
    strategy_used: str
    actions: List[RecoveryAction] = field(default_factory=list)
    success: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_action(self, action: RecoveryAction):
        self.actions.append(action)

    def complete(self, success: bool):
        self.success = success
        self.completed_at = datetime.now()


class StopAndAnalyzeStrategy:
    """
    Recovery by stopping and analyzing.

    Used for repetition loops - forces a pause and analysis
    of what's going wrong.
    """

    @property
    def name(self) -> str:
        return "stop_and_analyze"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {ConfusionType.REPETITION_LOOP, ConfusionType.OSCILLATION}

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute stop and analyze recovery."""
        attempt = context.get('recovery_attempt')

        # Step 1: Stop all current actions
        attempt.add_action(RecoveryAction(
            action_type='halt',
            description='Stopped all pending actions',
            success=True
        ))

        # Step 2: Collect what's been tried
        tried_approaches = context.get('tried_approaches', [])
        attempt.add_action(RecoveryAction(
            action_type='collect',
            description=f'Collected {len(tried_approaches)} tried approaches',
            success=True,
            details={'approaches': tried_approaches}
        ))

        # Step 3: Identify what hasn't been tried
        all_approaches = context.get('available_approaches', [])
        untried = [a for a in all_approaches if a not in tried_approaches]
        attempt.add_action(RecoveryAction(
            action_type='identify',
            description=f'Found {len(untried)} untried approaches',
            success=len(untried) > 0,
            details={'untried': untried}
        ))

        return len(untried) > 0


class CheckpointRestoreStrategy:
    """
    Recovery by restoring from checkpoint.

    Used for state confusion - reverts to known good state.
    """

    @property
    def name(self) -> str:
        return "checkpoint_restore"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {
            ConfusionType.STATE_MISMATCH,
            ConfusionType.CONTEXT_LOSS,
            ConfusionType.TEMPORAL_CONFUSION
        }

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute checkpoint restore recovery."""
        attempt = context.get('recovery_attempt')
        checkpoint_manager = context.get('checkpoint_manager')

        if not checkpoint_manager:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No checkpoint manager available',
                success=False
            ))
            return False

        # Step 1: Find most recent valid checkpoint
        checkpoint = checkpoint_manager.get_latest()
        if not checkpoint:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No valid checkpoint found',
                success=False
            ))
            return False

        attempt.add_action(RecoveryAction(
            action_type='found',
            description=f'Found checkpoint from {checkpoint.get("timestamp")}',
            success=True,
            details=checkpoint
        ))

        # Step 2: Restore state from checkpoint
        try:
            checkpoint_manager.restore(checkpoint)
            attempt.add_action(RecoveryAction(
                action_type='restore',
                description='Restored from checkpoint',
                success=True
            ))
        except Exception as e:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description=f'Restore failed: {e}',
                success=False
            ))
            return False

        # Step 3: Verify restoration
        try:
            checkpoint_manager.verify()
            attempt.add_action(RecoveryAction(
                action_type='verify',
                description='Verified restored state',
                success=True
            ))
            return True
        except Exception as e:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description=f'Verification failed: {e}',
                success=False
            ))
            return False


class EscalationStrategy:
    """
    Recovery by escalating to higher level.

    Used when local recovery isn't possible.
    """

    @property
    def name(self) -> str:
        return "escalation"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {
            ConfusionType.BLOCKED,
            ConfusionType.WRONG_PATH,
            ConfusionType.SCOPE_CREEP
        }

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute escalation recovery."""
        attempt = context.get('recovery_attempt')
        escalation_handler = context.get('escalation_handler')

        if not escalation_handler:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No escalation handler available',
                success=False
            ))
            return False

        # Step 1: Prepare escalation report
        report = {
            'diagnosis': diagnosis.to_dict(),
            'context_summary': context.get('summary', 'No summary available'),
            'attempted_actions': context.get('attempted_actions', []),
            'current_state': context.get('current_state', {})
        }

        attempt.add_action(RecoveryAction(
            action_type='prepare',
            description='Prepared escalation report',
            success=True,
            details={'report': report}
        ))

        # Step 2: Escalate
        try:
            escalation_handler.escalate(report)
            attempt.add_action(RecoveryAction(
                action_type='escalate',
                description='Escalated to higher level',
                success=True
            ))
            return True
        except Exception as e:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description=f'Escalation failed: {e}',
                success=False
            ))
            return False


class UserInterventionStrategy:
    """
    Recovery by requesting user intervention.

    Used for critical confusion that can't be auto-resolved.
    """

    @property
    def name(self) -> str:
        return "user_intervention"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {
            ConfusionType.HALLUCINATION,
            ConfusionType.CONTRADICTION,
            ConfusionType.UNSPECIFIED
        }

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute user intervention recovery."""
        attempt = context.get('recovery_attempt')
        user_handler = context.get('user_handler')

        if not user_handler:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No user handler available',
                success=False
            ))
            return False

        # Step 1: Prepare user message
        message = self._compose_user_message(diagnosis, context)

        attempt.add_action(RecoveryAction(
            action_type='compose',
            description='Composed message for user',
            success=True,
            details={'message': message}
        ))

        # Step 2: Request intervention
        try:
            user_handler.request_intervention(message)
            attempt.add_action(RecoveryAction(
                action_type='request',
                description='Requested user intervention',
                success=True
            ))
            return True
        except Exception as e:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description=f'Request failed: {e}',
                success=False
            ))
            return False

    def _compose_user_message(
        self,
        diagnosis: ConfusionDiagnosis,
        context: Dict[str, Any]
    ) -> str:
        """Compose a message explaining what's wrong and what help is needed."""
        lines = [
            "I've detected a problem that I cannot resolve automatically.",
            "",
            f"Type: {diagnosis.confusion_type.name}",
            f"Severity: {diagnosis.severity.name}",
            f"Likely cause: {diagnosis.likely_cause}",
            "",
            "Signals detected:"
        ]

        for signal in diagnosis.signals[:3]:  # Top 3
            lines.append(f"  - {signal.description}")

        lines.extend([
            "",
            "I need help with:",
            f"  {diagnosis.recommended_action}",
            "",
            "Please provide guidance on how to proceed."
        ])

        return "\n".join(lines)


class SynapticReinforcementStrategy:
    """
    Recovery by reinforcing successful synaptic pathways.

    Used when stagnation is detected - strengthens edges that led
    to successful outcomes to revive productive patterns.
    """

    @property
    def name(self) -> str:
        return "synaptic_reinforcement"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {ConfusionType.BLOCKED}

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute synaptic reinforcement recovery."""
        attempt = context.get('recovery_attempt')
        memory_graph = context.get('memory_graph')

        if not memory_graph:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No synaptic memory graph available',
                success=False
            ))
            return False

        # Step 1: Identify successful paths
        successful_paths = context.get('successful_paths', [])
        if not successful_paths:
            attempt.add_action(RecoveryAction(
                action_type='warning',
                description='No successful paths to reinforce',
                success=False
            ))
            return False

        # Step 2: Strengthen edges along successful paths
        reinforced_count = 0
        for path in successful_paths:
            try:
                memory_graph.apply_reward(path, reward=0.5)
                reinforced_count += 1
            except Exception as e:
                attempt.add_action(RecoveryAction(
                    action_type='error',
                    description=f'Failed to reinforce path: {e}',
                    success=False
                ))

        attempt.add_action(RecoveryAction(
            action_type='reinforce',
            description=f'Reinforced {reinforced_count} successful pathways',
            success=reinforced_count > 0,
            details={'paths': successful_paths}
        ))

        return reinforced_count > 0


class SynapticPruningStrategy:
    """
    Recovery by pruning unsuccessful synaptic pathways.

    Used for repetition loops - weakens edges that led to failures
    to discourage repeating unsuccessful approaches.
    """

    @property
    def name(self) -> str:
        return "synaptic_pruning"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {ConfusionType.REPETITION_LOOP, ConfusionType.OSCILLATION}

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute synaptic pruning recovery."""
        attempt = context.get('recovery_attempt')
        memory_graph = context.get('memory_graph')

        if not memory_graph:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No synaptic memory graph available',
                success=False
            ))
            return False

        # Step 1: Identify failed paths from signals
        failed_paths = []
        for signal in diagnosis.signals:
            if signal.signal_type in ['synaptic_loop', 'synaptic_oscillation']:
                # Extract pattern from evidence
                for evidence in signal.evidence:
                    if evidence.startswith('Pattern:'):
                        pattern_str = evidence.split('Pattern:')[1].strip()
                        # Parse pattern (e.g., "['A', 'B']" -> ['A', 'B'])
                        try:
                            import ast
                            pattern = ast.literal_eval(pattern_str)
                            if isinstance(pattern, list):
                                failed_paths.append(pattern)
                        except Exception:
                            pass

        if not failed_paths:
            attempt.add_action(RecoveryAction(
                action_type='warning',
                description='No failed paths identified for pruning',
                success=False
            ))
            return False

        # Step 2: Weaken edges along failed paths
        pruned_count = 0
        for path in failed_paths:
            try:
                memory_graph.apply_reward(path, reward=-0.3)
                pruned_count += 1
            except Exception as e:
                attempt.add_action(RecoveryAction(
                    action_type='error',
                    description=f'Failed to prune path: {e}',
                    success=False
                ))

        attempt.add_action(RecoveryAction(
            action_type='prune',
            description=f'Pruned {pruned_count} unsuccessful pathways',
            success=pruned_count > 0,
            details={'paths': failed_paths}
        ))

        return pruned_count > 0


class SynapticResetStrategy:
    """
    Recovery by resetting synaptic activation state.

    Used for severe confusion - clears recent activations and
    allows fresh reasoning without bias from confused state.
    """

    @property
    def name(self) -> str:
        return "synaptic_reset"

    @property
    def applicable_to(self) -> Set[ConfusionType]:
        return {
            ConfusionType.CONTRADICTION,
            ConfusionType.TEMPORAL_CONFUSION,
            ConfusionType.UNSPECIFIED
        }

    def execute(self, diagnosis: ConfusionDiagnosis, context: Dict[str, Any]) -> bool:
        """Execute synaptic reset recovery."""
        attempt = context.get('recovery_attempt')
        memory_graph = context.get('memory_graph')
        synaptic_detector = context.get('synaptic_detector')

        if not memory_graph:
            attempt.add_action(RecoveryAction(
                action_type='error',
                description='No synaptic memory graph available',
                success=False
            ))
            return False

        # Step 1: Clear recent activations
        memory_graph._recent_activations.clear()

        attempt.add_action(RecoveryAction(
            action_type='clear',
            description='Cleared recent activation state',
            success=True
        ))

        # Step 2: Clear activation sequence in detector (if available)
        if synaptic_detector:
            synaptic_detector._activation_sequence.clear()
            attempt.add_action(RecoveryAction(
                action_type='clear',
                description='Cleared activation sequence tracker',
                success=True
            ))

        # Step 3: Reset focus in reasoner (if available)
        reasoner = context.get('reasoner')
        if reasoner:
            reasoner.reset_focus()
            attempt.add_action(RecoveryAction(
                action_type='reset',
                description='Reset reasoning focus',
                success=True
            ))

        return True


# =============================================================================
# RECOVERY COORDINATOR
# =============================================================================

class RecoveryCoordinator:
    """
    Coordinates the complete recovery process.

    Integrates:
    - Confusion detection
    - Strategy selection
    - Recovery execution
    - Learning from recovery
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.diagnoser = ConfusionDiagnoser()
        self.strategies: Dict[str, RecoveryStrategy] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self.storage_dir = storage_dir

        # Set up default detectors
        self._setup_default_detectors()

        # Set up default strategies
        self._setup_default_strategies()

    def _setup_default_detectors(self):
        """Set up the default detection suite."""
        self.repetition_detector = RepetitionDetector()
        self.contradiction_detector = ContradictionDetector()
        self.progress_detector = ProgressDetector()
        self.state_verifier = StateVerifier()
        self.synaptic_detector: Optional[SynapticConfusionDetector] = None

        self.diagnoser.add_detector(self.repetition_detector)
        self.diagnoser.add_detector(self.contradiction_detector)
        self.diagnoser.add_detector(self.progress_detector)
        self.diagnoser.add_detector(self.state_verifier)

    def _setup_default_strategies(self):
        """Set up the default recovery strategies."""
        strategies = [
            StopAndAnalyzeStrategy(),
            CheckpointRestoreStrategy(),
            EscalationStrategy(),
            UserInterventionStrategy(),
            SynapticReinforcementStrategy(),
            SynapticPruningStrategy(),
            SynapticResetStrategy()
        ]

        for strategy in strategies:
            self.strategies[strategy.name] = strategy

    def record_action(
        self,
        action_type: str,
        target: str,
        result: str,
        parameters: Dict[str, Any]
    ):
        """Record an action for detection."""
        self.repetition_detector.record_action(
            action_type, target, result, parameters
        )

    def record_statement(self, topic: str, content: str):
        """Record a statement for contradiction detection."""
        self.contradiction_detector.record_statement(topic, content)

    def record_progress(self, progress_type: str = 'task'):
        """Record that progress was made."""
        self.progress_detector.record_progress(progress_type)

    def register_belief(self, topic: str, belief: Any):
        """Register a belief for state verification."""
        self.state_verifier.register_belief(topic, belief)

    def register_verifier(self, topic: str, verifier: Callable[[], Any]):
        """Register a way to verify a belief."""
        self.state_verifier.register_verifier(topic, verifier)

    def enable_synaptic_detection(
        self,
        memory_graph: 'SynapticMemoryGraph',
        loop_window: int = 5,
        contradiction_threshold: float = 0.7,
        stagnation_threshold: float = 0.1
    ):
        """
        Enable synaptic memory-based confusion detection.

        Args:
            memory_graph: The synaptic memory graph to monitor
            loop_window: Window size for detecting activation loops
            contradiction_threshold: Threshold for detecting contradictions
            stagnation_threshold: Minimum activation rate (per minute)
        """
        self.synaptic_detector = SynapticConfusionDetector(
            memory_graph=memory_graph,
            loop_window=loop_window,
            contradiction_threshold=contradiction_threshold,
            stagnation_threshold=stagnation_threshold
        )
        self.diagnoser.add_detector(self.synaptic_detector)

    def record_synaptic_activation(self, node_id: str):
        """
        Record a synaptic node activation.

        Args:
            node_id: ID of the activated node
        """
        if self.synaptic_detector:
            self.synaptic_detector.record_activation(node_id)

    def check_confusion(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ConfusionDiagnosis]:
        """
        Check for confusion.

        Returns diagnosis if confusion detected, None otherwise.
        """
        return self.diagnoser.diagnose(context or {})

    def recover(
        self,
        diagnosis: ConfusionDiagnosis,
        context: Dict[str, Any]
    ) -> RecoveryAttempt:
        """
        Attempt to recover from diagnosed confusion.

        Tries applicable strategies in order of preference.
        """
        # Find applicable strategies
        applicable = []
        for strategy in self.strategies.values():
            if diagnosis.confusion_type in strategy.applicable_to:
                applicable.append(strategy)

        if not applicable:
            # Fall back to user intervention
            applicable = [self.strategies.get('user_intervention')]
            applicable = [s for s in applicable if s is not None]

        # Create recovery attempt
        attempt = RecoveryAttempt(
            diagnosis=diagnosis,
            strategy_used='none'
        )
        context['recovery_attempt'] = attempt

        # Add synaptic detector to context if available
        if self.synaptic_detector:
            context['synaptic_detector'] = self.synaptic_detector
            # Also add the memory graph if not already present
            if 'memory_graph' not in context:
                context['memory_graph'] = self.synaptic_detector._memory

        # Try each strategy
        for strategy in applicable:
            attempt.strategy_used = strategy.name

            try:
                success = strategy.execute(diagnosis, context)
                if success:
                    attempt.complete(True)
                    break
            except Exception as e:
                attempt.add_action(RecoveryAction(
                    action_type='error',
                    description=f'Strategy {strategy.name} threw exception: {e}',
                    success=False
                ))

        if not attempt.completed_at:
            attempt.complete(False)

        self.recovery_history.append(attempt)
        self._save_attempt(attempt)

        return attempt

    def _save_attempt(self, attempt: RecoveryAttempt):
        """Save recovery attempt for learning."""
        if not self.storage_dir:
            return

        recovery_dir = self.storage_dir / "recovery_history"
        recovery_dir.mkdir(parents=True, exist_ok=True)

        filename = f"recovery_{attempt.started_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = recovery_dir / filename

        data = {
            'diagnosis': attempt.diagnosis.to_dict(),
            'strategy_used': attempt.strategy_used,
            'actions': [
                {
                    'action_type': a.action_type,
                    'description': a.description,
                    'success': a.success,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in attempt.actions
            ],
            'success': attempt.success,
            'started_at': attempt.started_at.isoformat(),
            'completed_at': attempt.completed_at.isoformat() if attempt.completed_at else None
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts."""
        if not self.recovery_history:
            return {
                'total_attempts': 0,
                'success_rate': 0.0,
                'by_type': {},
                'by_strategy': {}
            }

        total = len(self.recovery_history)
        successes = sum(1 for a in self.recovery_history if a.success)

        by_type: Dict[str, Dict[str, int]] = {}
        by_strategy: Dict[str, Dict[str, int]] = {}

        for attempt in self.recovery_history:
            type_name = attempt.diagnosis.confusion_type.name
            if type_name not in by_type:
                by_type[type_name] = {'total': 0, 'success': 0}
            by_type[type_name]['total'] += 1
            if attempt.success:
                by_type[type_name]['success'] += 1

            strat_name = attempt.strategy_used
            if strat_name not in by_strategy:
                by_strategy[strat_name] = {'total': 0, 'success': 0}
            by_strategy[strat_name]['total'] += 1
            if attempt.success:
                by_strategy[strat_name]['success'] += 1

        return {
            'total_attempts': total,
            'success_rate': successes / total if total > 0 else 0.0,
            'by_type': by_type,
            'by_strategy': by_strategy
        }


# =============================================================================
# CONTINUOUS MONITORING
# =============================================================================

class ConfusionMonitor:
    """
    Continuous background monitoring for confusion.

    Runs alongside normal operation and alerts when confusion
    signals accumulate.
    """

    def __init__(
        self,
        coordinator: RecoveryCoordinator,
        alert_threshold: float = 0.6,
        auto_recover: bool = False
    ):
        self.coordinator = coordinator
        self.alert_threshold = alert_threshold
        self.auto_recover = auto_recover

        self.is_monitoring: bool = False
        self.alert_callback: Optional[Callable[[ConfusionDiagnosis], None]] = None

    def set_alert_callback(self, callback: Callable[[ConfusionDiagnosis], None]):
        """Set callback for confusion alerts."""
        self.alert_callback = callback

    def check(self, context: Optional[Dict[str, Any]] = None) -> Optional[ConfusionDiagnosis]:
        """
        Perform a confusion check.

        Returns diagnosis if threshold exceeded.
        """
        diagnosis = self.coordinator.check_confusion(context)

        if diagnosis and diagnosis.confidence >= self.alert_threshold:
            if self.alert_callback:
                self.alert_callback(diagnosis)

            if self.auto_recover:
                self.coordinator.recover(diagnosis, context or {})

            return diagnosis

        return None

    def wrap_action(
        self,
        action_fn: Callable,
        action_type: str,
        target: str
    ) -> Callable:
        """
        Wrap an action with confusion monitoring.

        Records the action for detection and checks after completion.
        """
        def wrapped(*args, **kwargs):
            try:
                result = action_fn(*args, **kwargs)
                self.coordinator.record_action(
                    action_type, target, 'success',
                    {'args': str(args), 'kwargs': str(kwargs)}
                )
                self.coordinator.record_progress()
                return result
            except Exception as e:
                self.coordinator.record_action(
                    action_type, target, 'failure',
                    {'error': str(e)}
                )
                # Check for confusion after failure
                self.check()
                raise

        return wrapped
