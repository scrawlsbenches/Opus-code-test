"""
Learning System: Experience Capture and Pattern Extraction

This module implements the "memory" that I lack natively. It captures
experiences from executions, extracts patterns, and distills lessons
that can inform future behavior.

The key insight: I cannot learn through weight updates, but I CAN learn
through accumulated experiences stored externally and retrieved contextually.

Learning Flow:
    Execute → Experience → Pattern → Lesson → Retrieval → Apply

    1. EXECUTE: Run a goal through the orchestration system
    2. EXPERIENCE: Capture what happened (actions, outcomes, context)
    3. PATTERN: Identify recurring structures across experiences
    4. LESSON: Distill actionable insights from patterns
    5. RETRIEVAL: Find relevant lessons for current situation
    6. APPLY: Use lessons to inform decisions

This feeds into evolution.py for strategy improvement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Iterator, Protocol
)
from pathlib import Path
import json
import hashlib
import threading


# =============================================================================
# EXPERIENCE TYPES
# =============================================================================

class OutcomeType(Enum):
    """How did an action/goal turn out?"""
    SUCCESS = auto()       # Achieved intended result
    PARTIAL = auto()       # Partially achieved
    FAILURE = auto()       # Did not achieve
    UNEXPECTED = auto()    # Achieved something different
    BLOCKED = auto()       # Could not proceed
    ABANDONED = auto()     # Gave up intentionally


class ExperienceType(Enum):
    """What kind of experience is this?"""
    GOAL_EXECUTION = auto()     # A complete goal was executed
    TASK_EXECUTION = auto()     # A single task was executed
    DECISION_POINT = auto()     # A decision was made
    RECOVERY = auto()           # Recovered from confusion/error
    COLLABORATION = auto()      # Interaction between agents
    DISCOVERY = auto()          # Found unexpected information
    INSIGHT = auto()            # Realized something important


@dataclass
class Context:
    """
    The situation in which an experience occurred.

    Context is crucial for learning - the same action can have
    different outcomes in different contexts.
    """
    # What was the broader goal?
    goal_type: str
    goal_complexity: str  # simple, moderate, complex

    # What resources were available?
    available_tools: List[str] = field(default_factory=list)
    available_agents: int = 1

    # What was the state of the system?
    prior_failures: int = 0
    time_pressure: str = "none"  # none, moderate, high

    # What constraints applied?
    constraints: List[str] = field(default_factory=list)

    # Domain/topic
    domain: str = "general"

    # Free-form context notes
    notes: str = ""

    def similarity_to(self, other: 'Context') -> float:
        """How similar is this context to another?"""
        score = 0.0
        weights = {
            'goal_type': 0.3,
            'goal_complexity': 0.2,
            'domain': 0.2,
            'available_agents': 0.1,
            'prior_failures': 0.1,
            'time_pressure': 0.1
        }

        if self.goal_type == other.goal_type:
            score += weights['goal_type']
        if self.goal_complexity == other.goal_complexity:
            score += weights['goal_complexity']
        if self.domain == other.domain:
            score += weights['domain']
        if abs(self.available_agents - other.available_agents) <= 1:
            score += weights['available_agents']
        if abs(self.prior_failures - other.prior_failures) <= 2:
            score += weights['prior_failures']
        if self.time_pressure == other.time_pressure:
            score += weights['time_pressure']

        return score


@dataclass
class Action:
    """A single action taken during execution."""
    action_type: str          # What kind of action
    description: str          # What was done
    target: str              # What it was done to
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type,
            'description': self.description,
            'target': self.target,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms
        }


@dataclass
class Outcome:
    """The result of an action or goal."""
    outcome_type: OutcomeType
    description: str

    # What was achieved?
    achieved: List[str] = field(default_factory=list)

    # What was NOT achieved?
    not_achieved: List[str] = field(default_factory=list)

    # What was unexpected?
    unexpected: List[str] = field(default_factory=list)

    # Quality metrics
    quality_score: Optional[float] = None  # 0-1
    efficiency_score: Optional[float] = None  # 0-1

    # Error information if applicable
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def was_successful(self) -> bool:
        return self.outcome_type == OutcomeType.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            'outcome_type': self.outcome_type.name,
            'description': self.description,
            'achieved': self.achieved,
            'not_achieved': self.not_achieved,
            'unexpected': self.unexpected,
            'quality_score': self.quality_score,
            'efficiency_score': self.efficiency_score,
            'error_type': self.error_type,
            'error_message': self.error_message
        }


@dataclass
class Experience:
    """
    A complete record of something that happened.

    This is the fundamental unit of learning. Each experience captures:
    - What was the situation? (context)
    - What was attempted? (intent, actions)
    - What happened? (outcome)
    - What was learned? (reflection)
    """
    id: str
    experience_type: ExperienceType
    timestamp: datetime

    # The situation
    context: Context

    # What was intended
    intent: str
    strategy_used: Optional[str] = None

    # What happened
    actions: List[Action] = field(default_factory=list)
    outcome: Optional[Outcome] = None

    # Reflection (added after execution)
    what_worked: List[str] = field(default_factory=list)
    what_didnt_work: List[str] = field(default_factory=list)
    would_do_differently: List[str] = field(default_factory=list)

    # Connections to other experiences
    related_experiences: List[str] = field(default_factory=list)
    supersedes: Optional[str] = None  # If this replaces earlier experience

    # Tags for retrieval
    tags: Set[str] = field(default_factory=set)

    def add_action(self, action: Action):
        """Record an action taken during this experience."""
        self.actions.append(action)

    def complete(self, outcome: Outcome):
        """Mark this experience as complete with an outcome."""
        self.outcome = outcome

    def reflect(
        self,
        what_worked: List[str],
        what_didnt_work: List[str],
        would_do_differently: List[str]
    ):
        """Add post-execution reflection."""
        self.what_worked = what_worked
        self.what_didnt_work = what_didnt_work
        self.would_do_differently = would_do_differently

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'id': self.id,
            'experience_type': self.experience_type.name,
            'timestamp': self.timestamp.isoformat(),
            'context': {
                'goal_type': self.context.goal_type,
                'goal_complexity': self.context.goal_complexity,
                'domain': self.context.domain,
                'available_tools': self.context.available_tools,
                'available_agents': self.context.available_agents,
                'prior_failures': self.context.prior_failures,
                'time_pressure': self.context.time_pressure,
                'constraints': self.context.constraints,
                'notes': self.context.notes
            },
            'intent': self.intent,
            'strategy_used': self.strategy_used,
            'actions': [a.to_dict() for a in self.actions],
            'outcome': self.outcome.to_dict() if self.outcome else None,
            'what_worked': self.what_worked,
            'what_didnt_work': self.what_didnt_work,
            'would_do_differently': self.would_do_differently,
            'related_experiences': self.related_experiences,
            'supersedes': self.supersedes,
            'tags': list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Deserialize from persistence."""
        context = Context(
            goal_type=data['context']['goal_type'],
            goal_complexity=data['context']['goal_complexity'],
            domain=data['context'].get('domain', 'general'),
            available_tools=data['context'].get('available_tools', []),
            available_agents=data['context'].get('available_agents', 1),
            prior_failures=data['context'].get('prior_failures', 0),
            time_pressure=data['context'].get('time_pressure', 'none'),
            constraints=data['context'].get('constraints', []),
            notes=data['context'].get('notes', '')
        )

        experience = cls(
            id=data['id'],
            experience_type=ExperienceType[data['experience_type']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=context,
            intent=data['intent'],
            strategy_used=data.get('strategy_used')
        )

        # Reconstruct actions
        for action_data in data.get('actions', []):
            action = Action(
                action_type=action_data['action_type'],
                description=action_data['description'],
                target=action_data['target'],
                parameters=action_data.get('parameters', {}),
                timestamp=datetime.fromisoformat(action_data['timestamp']),
                duration_ms=action_data.get('duration_ms')
            )
            experience.actions.append(action)

        # Reconstruct outcome
        if data.get('outcome'):
            outcome_data = data['outcome']
            experience.outcome = Outcome(
                outcome_type=OutcomeType[outcome_data['outcome_type']],
                description=outcome_data['description'],
                achieved=outcome_data.get('achieved', []),
                not_achieved=outcome_data.get('not_achieved', []),
                unexpected=outcome_data.get('unexpected', []),
                quality_score=outcome_data.get('quality_score'),
                efficiency_score=outcome_data.get('efficiency_score'),
                error_type=outcome_data.get('error_type'),
                error_message=outcome_data.get('error_message')
            )

        experience.what_worked = data.get('what_worked', [])
        experience.what_didnt_work = data.get('what_didnt_work', [])
        experience.would_do_differently = data.get('would_do_differently', [])
        experience.related_experiences = data.get('related_experiences', [])
        experience.supersedes = data.get('supersedes')
        experience.tags = set(data.get('tags', []))

        return experience


# =============================================================================
# EXPERIENCE STORE
# =============================================================================

class ExperienceStore:
    """
    Persistent storage for experiences.

    Provides:
    - Save/load experiences to files
    - Query by context similarity
    - Query by tags
    - Query by outcome type
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Experience] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._load_index()

    def _load_index(self):
        """Load all experiences into memory index."""
        for exp_file in self.storage_dir.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                    experience = Experience.from_dict(data)
                    with self._lock:
                        self._index[experience.id] = experience
            except (json.JSONDecodeError, KeyError) as e:
                # Log but don't fail on corrupt files
                print(f"Warning: Could not load {exp_file}: {e}")

    def save(self, experience: Experience):
        """Save an experience to persistent storage."""
        with self._lock:
            self._index[experience.id] = experience

        filepath = self.storage_dir / f"{experience.id}.json"
        with open(filepath, 'w') as f:
            json.dump(experience.to_dict(), f, indent=2)

    def get(self, experience_id: str) -> Optional[Experience]:
        """Retrieve an experience by ID."""
        with self._lock:
            return self._index.get(experience_id)

    def find_similar_context(
        self,
        context: Context,
        min_similarity: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[Experience, float]]:
        """
        Find experiences with similar contexts.

        Returns list of (experience, similarity_score) tuples.
        """
        with self._lock:
            experiences = list(self._index.values())

        scored = []
        for experience in experiences:
            similarity = context.similarity_to(experience.context)
            if similarity >= min_similarity:
                scored.append((experience, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def find_by_tags(
        self,
        tags: Set[str],
        match_all: bool = False
    ) -> List[Experience]:
        """Find experiences by tags."""
        with self._lock:
            experiences = list(self._index.values())

        results = []
        for experience in experiences:
            if match_all:
                if tags.issubset(experience.tags):
                    results.append(experience)
            else:
                if tags.intersection(experience.tags):
                    results.append(experience)
        return results

    def find_by_outcome(
        self,
        outcome_type: OutcomeType
    ) -> List[Experience]:
        """Find experiences by outcome type."""
        with self._lock:
            experiences = list(self._index.values())

        return [
            exp for exp in experiences
            if exp.outcome and exp.outcome.outcome_type == outcome_type
        ]

    def find_successful_for_context(
        self,
        context: Context,
        limit: int = 5
    ) -> List[Experience]:
        """
        Find successful experiences with similar contexts.

        This is the key retrieval for learning - what worked before
        in situations like this?
        """
        similar = self.find_similar_context(context, limit=limit * 2)
        successful = [
            exp for exp, _ in similar
            if exp.outcome and exp.outcome.was_successful()
        ]
        return successful[:limit]

    def find_failures_for_context(
        self,
        context: Context,
        limit: int = 5
    ) -> List[Experience]:
        """
        Find failed experiences with similar contexts.

        This helps avoid repeating mistakes.
        """
        similar = self.find_similar_context(context, limit=limit * 2)
        failures = [
            exp for exp, _ in similar
            if exp.outcome and exp.outcome.outcome_type == OutcomeType.FAILURE
        ]
        return failures[:limit]

    def all_experiences(self) -> Iterator[Experience]:
        """Iterate over all stored experiences."""
        with self._lock:
            return iter(list(self._index.values()))

    def count(self) -> int:
        """Total number of stored experiences."""
        with self._lock:
            return len(self._index)


# =============================================================================
# PATTERN EXTRACTION
# =============================================================================

@dataclass
class Pattern:
    """
    A recurring structure identified across experiences.

    Patterns are the building blocks of lessons - they capture
    regularities that can inform future behavior.
    """
    id: str
    pattern_type: str  # sequence, association, outcome_predictor
    description: str

    # What contexts does this pattern apply to?
    applicable_contexts: List[Dict[str, Any]] = field(default_factory=list)

    # The pattern structure
    structure: Dict[str, Any] = field(default_factory=dict)

    # Evidence supporting this pattern
    supporting_experiences: List[str] = field(default_factory=list)

    # Statistical strength
    occurrence_count: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0

    def add_evidence(self, experience_id: str, was_successful: bool):
        """Add an experience as evidence for this pattern."""
        self.supporting_experiences.append(experience_id)
        self.occurrence_count += 1
        if was_successful:
            total_successes = self.success_rate * (self.occurrence_count - 1) + 1
            self.success_rate = total_successes / self.occurrence_count
        else:
            total_successes = self.success_rate * (self.occurrence_count - 1)
            self.success_rate = total_successes / self.occurrence_count

        # Confidence grows with evidence (logarithmically)
        import math
        self.confidence = min(0.95, math.log(self.occurrence_count + 1) / 5)


class PatternExtractor:
    """
    Extracts patterns from collections of experiences.

    Pattern Types:
    - SEQUENCE: Action A followed by Action B tends to succeed
    - ASSOCIATION: Context X often co-occurs with Outcome Y
    - STRATEGY: Strategy S works well for Goal Type G
    - ANTI-PATTERN: This combination tends to fail
    """

    def __init__(self, store: ExperienceStore):
        self.store = store
        self.patterns: Dict[str, Pattern] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def extract_sequence_patterns(
        self,
        min_occurrences: int = 3
    ) -> List[Pattern]:
        """
        Find recurring action sequences that correlate with success.
        """
        sequences: Dict[str, List[Tuple[str, bool]]] = {}

        for experience in self.store.all_experiences():
            if len(experience.actions) < 2:
                continue

            # Extract action type sequences
            action_types = [a.action_type for a in experience.actions]
            was_successful = (
                experience.outcome and
                experience.outcome.was_successful()
            )

            # Look at pairs and triples
            for i in range(len(action_types) - 1):
                pair = f"{action_types[i]} -> {action_types[i+1]}"
                if pair not in sequences:
                    sequences[pair] = []
                sequences[pair].append((experience.id, was_successful))

            for i in range(len(action_types) - 2):
                triple = f"{action_types[i]} -> {action_types[i+1]} -> {action_types[i+2]}"
                if triple not in sequences:
                    sequences[triple] = []
                sequences[triple].append((experience.id, was_successful))

        # Create patterns from frequent sequences
        patterns = []
        for seq, evidence in sequences.items():
            if len(evidence) >= min_occurrences:
                pattern_id = hashlib.md5(seq.encode()).hexdigest()[:12]
                pattern = Pattern(
                    id=f"seq_{pattern_id}",
                    pattern_type="sequence",
                    description=f"Action sequence: {seq}",
                    structure={'sequence': seq.split(' -> ')}
                )

                for exp_id, was_successful in evidence:
                    pattern.add_evidence(exp_id, was_successful)

                patterns.append(pattern)
                with self._lock:
                    self.patterns[pattern.id] = pattern

        return patterns

    def extract_strategy_patterns(
        self,
        min_occurrences: int = 3
    ) -> List[Pattern]:
        """
        Find which strategies work for which goal types.
        """
        strategy_outcomes: Dict[Tuple[str, str], List[Tuple[str, bool]]] = {}

        for experience in self.store.all_experiences():
            if not experience.strategy_used:
                continue

            key = (experience.strategy_used, experience.context.goal_type)
            if key not in strategy_outcomes:
                strategy_outcomes[key] = []

            was_successful = (
                experience.outcome and
                experience.outcome.was_successful()
            )
            strategy_outcomes[key].append((experience.id, was_successful))

        patterns = []
        for (strategy, goal_type), evidence in strategy_outcomes.items():
            if len(evidence) >= min_occurrences:
                pattern_id = hashlib.md5(
                    f"{strategy}_{goal_type}".encode()
                ).hexdigest()[:12]

                pattern = Pattern(
                    id=f"strat_{pattern_id}",
                    pattern_type="strategy",
                    description=f"Strategy '{strategy}' for goal type '{goal_type}'",
                    structure={'strategy': strategy, 'goal_type': goal_type}
                )

                for exp_id, was_successful in evidence:
                    pattern.add_evidence(exp_id, was_successful)

                patterns.append(pattern)
                with self._lock:
                    self.patterns[pattern.id] = pattern

        return patterns

    def extract_antipatterns(
        self,
        min_failures: int = 3
    ) -> List[Pattern]:
        """
        Find patterns that correlate with failure.

        These are things to AVOID doing.
        """
        failures = self.store.find_by_outcome(OutcomeType.FAILURE)

        # Look for common elements in failures
        failure_contexts: Dict[str, List[str]] = {
            'goal_types': [],
            'strategies': [],
            'action_patterns': []
        }

        for exp in failures:
            failure_contexts['goal_types'].append(exp.context.goal_type)
            if exp.strategy_used:
                failure_contexts['strategies'].append(exp.strategy_used)

            if len(exp.actions) >= 2:
                action_types = [a.action_type for a in exp.actions[:3]]
                failure_contexts['action_patterns'].append(
                    ' -> '.join(action_types)
                )

        patterns = []

        # Find frequently failing goal types
        from collections import Counter
        for element, count in Counter(failure_contexts['goal_types']).items():
            if count >= min_failures:
                pattern = Pattern(
                    id=f"anti_goal_{hashlib.md5(element.encode()).hexdigest()[:8]}",
                    pattern_type="antipattern",
                    description=f"Goal type '{element}' frequently fails",
                    structure={'problematic_goal_type': element}
                )
                pattern.occurrence_count = count
                pattern.success_rate = 0.0
                patterns.append(pattern)
                with self._lock:
                    self.patterns[pattern.id] = pattern

        return patterns

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve a specific pattern."""
        with self._lock:
            return self.patterns.get(pattern_id)

    def get_patterns_for_context(
        self,
        context: Context,
        pattern_type: Optional[str] = None
    ) -> List[Pattern]:
        """Get patterns applicable to a given context."""
        with self._lock:
            all_patterns = list(self.patterns.values())

        applicable = []

        for pattern in all_patterns:
            if pattern_type and pattern.pattern_type != pattern_type:
                continue

            # Check if pattern's context matches
            if pattern.pattern_type == "strategy":
                if pattern.structure.get('goal_type') == context.goal_type:
                    applicable.append(pattern)
            else:
                # Default: include if confidence is high enough
                if pattern.confidence >= 0.3:
                    applicable.append(pattern)

        return applicable


# =============================================================================
# LESSON SYSTEM
# =============================================================================

@dataclass
class Lesson:
    """
    A distilled, actionable insight from patterns.

    Lessons are the highest-level learning artifacts - they encode
    what to do (or not do) in specific situations.
    """
    id: str
    title: str
    description: str

    # When does this lesson apply?
    applicable_conditions: Dict[str, Any] = field(default_factory=dict)

    # What should be done?
    recommendations: List[str] = field(default_factory=list)

    # What should be avoided?
    warnings: List[str] = field(default_factory=list)

    # Evidence
    supporting_patterns: List[str] = field(default_factory=list)
    supporting_experiences: List[str] = field(default_factory=list)

    # Confidence and age
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    validation_count: int = 0

    # Lesson application tracking (for aging)
    last_applied: Optional[datetime] = None
    application_count: int = 0

    # Has this lesson been superseded?
    superseded_by: Optional[str] = None

    def is_applicable_to(self, context: Context) -> bool:
        """Check if this lesson applies to a given context."""
        conditions = self.applicable_conditions

        if 'goal_types' in conditions:
            if context.goal_type not in conditions['goal_types']:
                return False

        if 'domains' in conditions:
            if context.domain not in conditions['domains']:
                return False

        if 'complexity' in conditions:
            if context.goal_complexity not in conditions['complexity']:
                return False

        return True

    def validate(self, was_helpful: bool):
        """Record whether this lesson was helpful when applied."""
        self.last_validated = datetime.now()
        self.validation_count += 1

        if was_helpful:
            # Increase confidence
            self.confidence = min(0.95, self.confidence + 0.05)
        else:
            # Decrease confidence
            self.confidence = max(0.0, self.confidence - 0.1)

    def record_application(self):
        """Record that this lesson was applied."""
        self.last_applied = datetime.now()
        self.application_count += 1
        # Boost confidence slightly when applied
        self.confidence = min(0.95, self.confidence + 0.01)

    def apply_aging(self, days_since_last_use: int):
        """
        Apply aging to reduce confidence of unused lessons.

        Lessons lose confidence over time if not used, as the world changes
        and old lessons may become less relevant.
        """
        if days_since_last_use > 30:
            # Reduce confidence for lessons not used in 30+ days
            decay_factor = min(0.5, (days_since_last_use - 30) * 0.01)
            self.confidence = max(0.0, self.confidence - decay_factor)

    def similarity_to(self, other: 'Lesson') -> float:
        """
        Calculate similarity to another lesson.

        Used for consolidation to merge similar lessons.
        """
        score = 0.0

        # Compare applicable conditions (30%)
        if self.applicable_conditions == other.applicable_conditions:
            score += 0.3
        elif self.applicable_conditions and other.applicable_conditions:
            # Partial match on goal types
            my_goals = set(self.applicable_conditions.get('goal_types', []))
            other_goals = set(other.applicable_conditions.get('goal_types', []))
            if my_goals and other_goals:
                overlap = len(my_goals & other_goals) / len(my_goals | other_goals)
                score += 0.3 * overlap

        # Compare recommendations (40%)
        my_recs = set(self.recommendations)
        other_recs = set(other.recommendations)
        if my_recs and other_recs:
            rec_overlap = len(my_recs & other_recs) / len(my_recs | other_recs)
            score += 0.4 * rec_overlap

        # Compare warnings (20%)
        my_warns = set(self.warnings)
        other_warns = set(other.warnings)
        if my_warns and other_warns:
            warn_overlap = len(my_warns & other_warns) / len(my_warns | other_warns)
            score += 0.2 * warn_overlap

        # Compare supporting patterns (10%)
        my_patterns = set(self.supporting_patterns)
        other_patterns = set(other.supporting_patterns)
        if my_patterns and other_patterns:
            pattern_overlap = len(my_patterns & other_patterns) / len(my_patterns | other_patterns)
            score += 0.1 * pattern_overlap

        return score


class LessonDistiller:
    """
    Transforms patterns into actionable lessons.

    This is the "wisdom extraction" component - it takes raw patterns
    and creates lessons that can directly inform decisions.
    """

    def __init__(self, extractor: PatternExtractor, store: ExperienceStore):
        self.extractor = extractor
        self.store = store
        self.lessons: Dict[str, Lesson] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def distill_from_pattern(self, pattern: Pattern) -> Optional[Lesson]:
        """
        Create a lesson from a pattern.

        Only creates lessons for patterns with sufficient confidence.
        """
        if pattern.confidence < 0.4:
            return None

        if pattern.pattern_type == "sequence":
            return self._distill_sequence_lesson(pattern)
        elif pattern.pattern_type == "strategy":
            return self._distill_strategy_lesson(pattern)
        elif pattern.pattern_type == "antipattern":
            return self._distill_antipattern_lesson(pattern)

        return None

    def _distill_sequence_lesson(self, pattern: Pattern) -> Lesson:
        """Create a lesson from a sequence pattern."""
        sequence = pattern.structure.get('sequence', [])

        if pattern.success_rate > 0.6:
            lesson = Lesson(
                id=f"lesson_{pattern.id}",
                title=f"Effective sequence: {' → '.join(sequence)}",
                description=f"The action sequence '{' → '.join(sequence)}' "
                           f"has a {pattern.success_rate:.0%} success rate.",
                recommendations=[
                    f"Consider using the sequence: {' → '.join(sequence)}",
                    f"This pattern has worked in {pattern.occurrence_count} cases"
                ],
                supporting_patterns=[pattern.id],
                confidence=pattern.confidence * pattern.success_rate
            )
        else:
            lesson = Lesson(
                id=f"lesson_{pattern.id}",
                title=f"Risky sequence: {' → '.join(sequence)}",
                description=f"The action sequence '{' → '.join(sequence)}' "
                           f"has only a {pattern.success_rate:.0%} success rate.",
                warnings=[
                    f"The sequence {' → '.join(sequence)} often fails",
                    "Consider alternative approaches"
                ],
                supporting_patterns=[pattern.id],
                confidence=pattern.confidence * (1 - pattern.success_rate)
            )

        with self._lock:
            self.lessons[lesson.id] = lesson
        return lesson

    def _distill_strategy_lesson(self, pattern: Pattern) -> Lesson:
        """Create a lesson from a strategy pattern."""
        strategy = pattern.structure.get('strategy', 'unknown')
        goal_type = pattern.structure.get('goal_type', 'unknown')

        if pattern.success_rate > 0.6:
            lesson = Lesson(
                id=f"lesson_{pattern.id}",
                title=f"Strategy '{strategy}' works for '{goal_type}'",
                description=f"When facing '{goal_type}' goals, the '{strategy}' "
                           f"strategy succeeds {pattern.success_rate:.0%} of the time.",
                applicable_conditions={'goal_types': [goal_type]},
                recommendations=[
                    f"Use the '{strategy}' strategy for '{goal_type}' goals"
                ],
                supporting_patterns=[pattern.id],
                confidence=pattern.confidence * pattern.success_rate
            )
        else:
            lesson = Lesson(
                id=f"lesson_{pattern.id}",
                title=f"Avoid '{strategy}' for '{goal_type}'",
                description=f"The '{strategy}' strategy has poor results "
                           f"for '{goal_type}' goals ({pattern.success_rate:.0%} success).",
                applicable_conditions={'goal_types': [goal_type]},
                warnings=[
                    f"Avoid using '{strategy}' for '{goal_type}' goals"
                ],
                supporting_patterns=[pattern.id],
                confidence=pattern.confidence * (1 - pattern.success_rate)
            )

        with self._lock:
            self.lessons[lesson.id] = lesson
        return lesson

    def _distill_antipattern_lesson(self, pattern: Pattern) -> Lesson:
        """Create a lesson from an antipattern."""
        lesson = Lesson(
            id=f"lesson_{pattern.id}",
            title=f"Warning: {pattern.description}",
            description=pattern.description,
            warnings=[
                f"This has been observed in {pattern.occurrence_count} failures"
            ],
            supporting_patterns=[pattern.id],
            confidence=pattern.confidence
        )

        with self._lock:
            self.lessons[lesson.id] = lesson
        return lesson

    def get_lessons_for_context(
        self,
        context: Context,
        min_confidence: float = 0.3
    ) -> List[Lesson]:
        """Get all applicable lessons for a context."""
        with self._lock:
            all_lessons = list(self.lessons.values())

        applicable = []

        for lesson in all_lessons:
            if lesson.confidence < min_confidence:
                continue
            if lesson.superseded_by:
                continue
            if lesson.is_applicable_to(context):
                applicable.append(lesson)

        # Sort by confidence
        applicable.sort(key=lambda l: l.confidence, reverse=True)
        return applicable

    def distill_all(self, min_pattern_confidence: float = 0.4) -> List[Lesson]:
        """Distill lessons from all patterns."""
        lessons = []

        for pattern in self.extractor.patterns.values():
            if pattern.confidence >= min_pattern_confidence:
                lesson = self.distill_from_pattern(pattern)
                if lesson:
                    lessons.append(lesson)

        return lessons


# =============================================================================
# LEARNING CONSOLIDATION
# =============================================================================

@dataclass
class ConsolidationResult:
    """Results from a consolidation pass."""
    lessons_merged: int
    lessons_deprecated: int
    lessons_promoted: int
    new_patterns: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of consolidation."""
        return (
            f"Consolidation: {self.lessons_promoted} promoted, "
            f"{self.lessons_merged} merged, {self.lessons_deprecated} deprecated, "
            f"{len(self.new_patterns)} new patterns"
        )


class LearningConsolidator:
    """
    Consolidates lessons by promoting, merging, and deprecating.

    Over time, lessons accumulate. Consolidation:
    1. Promotes high-confidence lessons (marks them as validated)
    2. Merges similar lessons to avoid redundancy
    3. Deprecates low-confidence or outdated lessons
    4. Extracts meta-patterns from promoted lessons

    This keeps the lesson base clean and actionable.
    """

    def __init__(self, learning_cycle: 'LearningCycle'):
        self._cycle = learning_cycle
        self._confidence_threshold = 0.8  # High confidence for promotion
        self._deprecation_threshold = 0.3  # Low confidence for deprecation
        self._similarity_threshold = 0.85  # For merging similar lessons

    def consolidate(self) -> ConsolidationResult:
        """
        Run a full consolidation pass on all lessons.

        Returns a ConsolidationResult describing what was done.
        """
        result = ConsolidationResult(0, 0, 0, [])

        # 1. Apply aging to all lessons
        self._apply_aging_to_all()

        # 2. Identify and promote high-confidence lessons
        promoted = self._promote_high_confidence()
        result.lessons_promoted = len(promoted)

        # 3. Find and merge similar lessons
        merged = self._merge_similar_lessons()
        result.lessons_merged = merged

        # 4. Deprecate low-confidence lessons
        deprecated = self._deprecate_low_confidence()
        result.lessons_deprecated = deprecated

        # 5. Extract new patterns from promoted lessons
        result.new_patterns = self._extract_patterns(promoted)

        return result

    def _apply_aging_to_all(self):
        """Apply aging decay to all lessons based on usage."""
        now = datetime.now()

        for lesson in self._cycle.distiller.lessons.values():
            if lesson.superseded_by:
                # Don't age already deprecated lessons
                continue

            # Calculate days since last use
            if lesson.last_applied:
                days_since_use = (now - lesson.last_applied).days
            else:
                # Never applied - use creation date
                days_since_use = (now - lesson.created_at).days

            lesson.apply_aging(days_since_use)

    def _promote_high_confidence(self) -> List[Lesson]:
        """
        Identify high-confidence lessons for promotion.

        High-confidence lessons are those that:
        - Have confidence >= threshold
        - Have been validated multiple times
        - Are not already superseded
        """
        promoted = []

        for lesson in self._cycle.distiller.lessons.values():
            if lesson.superseded_by:
                continue

            if lesson.confidence >= self._confidence_threshold:
                # Mark as promoted by recording validation
                if lesson.validation_count == 0:
                    lesson.validate(was_helpful=True)
                promoted.append(lesson)

        return promoted

    def _merge_similar_lessons(self) -> int:
        """
        Find and merge similar lessons to reduce redundancy.

        Returns the number of lessons merged.
        """
        merged_count = 0
        lessons = [
            l for l in self._cycle.distiller.lessons.values()
            if not l.superseded_by
        ]

        # Find pairs of similar lessons
        already_merged = set()

        for i, lesson1 in enumerate(lessons):
            if lesson1.id in already_merged:
                continue

            for lesson2 in lessons[i + 1:]:
                if lesson2.id in already_merged:
                    continue

                similarity = lesson1.similarity_to(lesson2)

                if similarity >= self._similarity_threshold:
                    # Merge lesson2 into lesson1 (keep the one with higher confidence)
                    if lesson1.confidence >= lesson2.confidence:
                        self._merge_lesson_into(source=lesson2, target=lesson1)
                    else:
                        self._merge_lesson_into(source=lesson1, target=lesson2)

                    already_merged.add(lesson2.id)
                    merged_count += 1

        return merged_count

    def _merge_lesson_into(self, source: Lesson, target: Lesson):
        """
        Merge source lesson into target lesson.

        The source lesson is marked as superseded by the target.
        """
        # Mark source as superseded
        source.superseded_by = target.id

        # Merge evidence
        target.supporting_patterns.extend(
            p for p in source.supporting_patterns
            if p not in target.supporting_patterns
        )
        target.supporting_experiences.extend(
            e for e in source.supporting_experiences
            if e not in target.supporting_experiences
        )

        # Merge recommendations (avoid duplicates)
        for rec in source.recommendations:
            if rec not in target.recommendations:
                target.recommendations.append(rec)

        for warn in source.warnings:
            if warn not in target.warnings:
                target.warnings.append(warn)

        # Boost target confidence based on merged evidence
        combined_validations = source.validation_count + target.validation_count
        combined_applications = source.application_count + target.application_count

        # Weighted average of confidence
        if combined_validations > 0:
            target.confidence = (
                (source.confidence * source.validation_count +
                 target.confidence * target.validation_count) /
                combined_validations
            )

        target.validation_count = combined_validations
        target.application_count = combined_applications

        # Update timestamps
        if source.last_applied and target.last_applied:
            target.last_applied = max(source.last_applied, target.last_applied)
        elif source.last_applied:
            target.last_applied = source.last_applied

    def _deprecate_low_confidence(self) -> int:
        """
        Deprecate lessons with low confidence.

        Returns the number of lessons deprecated.
        """
        deprecated_count = 0

        for lesson in self._cycle.distiller.lessons.values():
            if lesson.superseded_by:
                # Already deprecated
                continue

            if lesson.confidence < self._deprecation_threshold:
                # Create a deprecation marker
                deprecation_id = f"deprecated_{lesson.id}"
                lesson.superseded_by = deprecation_id
                deprecated_count += 1

        return deprecated_count

    def _extract_patterns(self, promoted_lessons: List[Lesson]) -> List[str]:
        """
        Extract meta-patterns from promoted lessons.

        Looks for common themes across high-confidence lessons to create
        higher-order insights (patterns of patterns).
        """
        new_patterns = []

        if len(promoted_lessons) < 3:
            # Need at least 3 lessons to find meta-patterns
            return new_patterns

        # Group by applicable conditions
        from collections import defaultdict
        by_goal_type: Dict[str, List[Lesson]] = defaultdict(list)

        for lesson in promoted_lessons:
            goal_types = lesson.applicable_conditions.get('goal_types', [])
            for goal_type in goal_types:
                by_goal_type[goal_type].append(lesson)

        # Find goal types with multiple successful lessons
        for goal_type, goal_lessons in by_goal_type.items():
            if len(goal_lessons) >= 3:
                # Create a meta-pattern
                pattern_desc = (
                    f"Meta-pattern: Goal type '{goal_type}' has {len(goal_lessons)} "
                    f"high-confidence lessons, indicating a well-understood domain"
                )
                new_patterns.append(pattern_desc)

        # Look for common recommendations across lessons
        rec_counts: Dict[str, int] = defaultdict(int)
        for lesson in promoted_lessons:
            for rec in lesson.recommendations:
                rec_counts[rec] += 1

        # Recommendations appearing in multiple lessons become patterns
        for rec, count in rec_counts.items():
            if count >= 3:
                pattern_desc = (
                    f"Meta-pattern: '{rec}' appears in {count} high-confidence "
                    f"lessons, indicating a general best practice"
                )
                new_patterns.append(pattern_desc)

        return new_patterns


# =============================================================================
# LEARNING CYCLE
# =============================================================================

class LearningCycle:
    """
    The complete learning loop.

    Coordinates experience capture, pattern extraction, and lesson
    distillation to enable continuous improvement.

    Usage:
        cycle = LearningCycle(storage_dir)

        # During execution
        experience = cycle.start_experience(context, intent)
        experience.add_action(action)
        cycle.complete_experience(experience, outcome)

        # Before execution
        lessons = cycle.get_guidance(context)

        # Periodically
        cycle.extract_and_distill()
    """

    def __init__(self, storage_dir: Path):
        self.store = ExperienceStore(storage_dir / "experiences")
        self.extractor = PatternExtractor(self.store)
        self.distiller = LessonDistiller(self.extractor, self.store)

        # Load any saved patterns and lessons
        self._load_patterns(storage_dir / "patterns")
        self._load_lessons(storage_dir / "lessons")

    def _load_patterns(self, patterns_dir: Path):
        """Load previously extracted patterns."""
        if not patterns_dir.exists():
            return

        for pattern_file in patterns_dir.glob("*.json"):
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    pattern = Pattern(
                        id=data['id'],
                        pattern_type=data['pattern_type'],
                        description=data['description'],
                        structure=data.get('structure', {}),
                        supporting_experiences=data.get('supporting_experiences', []),
                        occurrence_count=data.get('occurrence_count', 0),
                        success_rate=data.get('success_rate', 0.0),
                        confidence=data.get('confidence', 0.0)
                    )
                    self.extractor.patterns[pattern.id] = pattern
            except (json.JSONDecodeError, KeyError):
                pass

    def _load_lessons(self, lessons_dir: Path):
        """Load previously distilled lessons."""
        if not lessons_dir.exists():
            return

        for lesson_file in lessons_dir.glob("*.json"):
            try:
                with open(lesson_file, 'r') as f:
                    data = json.load(f)
                    lesson = Lesson(
                        id=data['id'],
                        title=data['title'],
                        description=data['description'],
                        applicable_conditions=data.get('applicable_conditions', {}),
                        recommendations=data.get('recommendations', []),
                        warnings=data.get('warnings', []),
                        supporting_patterns=data.get('supporting_patterns', []),
                        confidence=data.get('confidence', 0.0),
                        validation_count=data.get('validation_count', 0)
                    )
                    self.distiller.lessons[lesson.id] = lesson
            except (json.JSONDecodeError, KeyError):
                pass

    def start_experience(
        self,
        context: Context,
        intent: str,
        experience_type: ExperienceType = ExperienceType.GOAL_EXECUTION,
        strategy: Optional[str] = None
    ) -> Experience:
        """Start tracking a new experience."""
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(context) % 10000:04d}"

        experience = Experience(
            id=exp_id,
            experience_type=experience_type,
            timestamp=datetime.now(),
            context=context,
            intent=intent,
            strategy_used=strategy
        )

        return experience

    def complete_experience(
        self,
        experience: Experience,
        outcome: Outcome,
        reflection: Optional[Dict[str, List[str]]] = None
    ):
        """Complete and save an experience."""
        experience.complete(outcome)

        if reflection:
            experience.reflect(
                what_worked=reflection.get('worked', []),
                what_didnt_work=reflection.get('didnt_work', []),
                would_do_differently=reflection.get('different', [])
            )

        # Auto-tag based on content
        experience.tags.add(experience.context.goal_type)
        experience.tags.add(experience.context.domain)
        experience.tags.add(outcome.outcome_type.name.lower())

        if experience.strategy_used:
            experience.tags.add(f"strategy:{experience.strategy_used}")

        self.store.save(experience)

    def get_guidance(
        self,
        context: Context,
        include_experiences: bool = True
    ) -> Dict[str, Any]:
        """
        Get guidance for a given context.

        Returns lessons, relevant experiences, and warnings.
        """
        guidance = {
            'lessons': [],
            'recommendations': [],
            'warnings': [],
            'relevant_successes': [],
            'relevant_failures': []
        }

        # Get applicable lessons
        lessons = self.distiller.get_lessons_for_context(context)
        guidance['lessons'] = lessons

        for lesson in lessons:
            guidance['recommendations'].extend(lesson.recommendations)
            guidance['warnings'].extend(lesson.warnings)

        if include_experiences:
            # Get relevant past experiences
            guidance['relevant_successes'] = self.store.find_successful_for_context(
                context, limit=3
            )
            guidance['relevant_failures'] = self.store.find_failures_for_context(
                context, limit=3
            )

        return guidance

    def extract_and_distill(self) -> Dict[str, int]:
        """
        Run pattern extraction and lesson distillation.

        Call this periodically to update learning.
        """
        results = {
            'sequence_patterns': 0,
            'strategy_patterns': 0,
            'antipatterns': 0,
            'lessons': 0
        }

        seq_patterns = self.extractor.extract_sequence_patterns()
        results['sequence_patterns'] = len(seq_patterns)

        strat_patterns = self.extractor.extract_strategy_patterns()
        results['strategy_patterns'] = len(strat_patterns)

        anti_patterns = self.extractor.extract_antipatterns()
        results['antipatterns'] = len(anti_patterns)

        lessons = self.distiller.distill_all()
        results['lessons'] = len(lessons)

        return results

    def validate_lesson(self, lesson_id: str, was_helpful: bool):
        """Record whether a lesson was helpful when applied."""
        with self.distiller._lock:
            lesson = self.distiller.lessons.get(lesson_id)
        if lesson:
            lesson.validate(was_helpful)

    def consolidate_lessons(self) -> ConsolidationResult:
        """
        Run consolidation on lessons.

        This should be called periodically (e.g., after extracting patterns)
        to keep the lesson base clean and actionable.
        """
        consolidator = LearningConsolidator(self)
        return consolidator.consolidate()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        return {
            'total_experiences': self.store.count(),
            'total_patterns': len(self.extractor.patterns),
            'total_lessons': len(self.distiller.lessons),
            'patterns_by_type': {
                'sequence': len([p for p in self.extractor.patterns.values()
                                if p.pattern_type == 'sequence']),
                'strategy': len([p for p in self.extractor.patterns.values()
                                if p.pattern_type == 'strategy']),
                'antipattern': len([p for p in self.extractor.patterns.values()
                                   if p.pattern_type == 'antipattern'])
            },
            'high_confidence_lessons': len([
                l for l in self.distiller.lessons.values()
                if l.confidence >= 0.7
            ]),
            'active_lessons': len([
                l for l in self.distiller.lessons.values()
                if not l.superseded_by
            ])
        }

    # =========================================================================
    # SEMANTIC INTENT MATCHING
    # =========================================================================
    # These methods enable finding experiences by semantic similarity of intent,
    # not just by categorical context matching. This is critical for useful
    # learning - agents need to find "what worked for JWT auth" not just
    # "what worked for features".

    # Stop words to filter out during keyword extraction
    _STOP_WORDS: Set[str] = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just',
        'don', 'now', 'and', 'or', 'but', 'if', 'this', 'that', 'these', 'those',
    }

    def extract_keywords(self, intent: str) -> Set[str]:
        """
        Extract meaningful keywords from a natural language intent string.

        This is the foundation of semantic matching. We extract terms that
        carry meaning and filter out common stop words.

        Args:
            intent: Natural language intent like "Implement JWT authentication"

        Returns:
            Set of keywords like {"implement", "jwt", "authentication"}

        Example:
            >>> cycle.extract_keywords("Implement JWT authentication for the API")
            {'implement', 'jwt', 'authentication', 'api'}
        """
        # Normalize: lowercase and split on word boundaries
        words = intent.lower().split()

        # Clean punctuation from words
        cleaned = []
        for word in words:
            # Remove leading/trailing punctuation
            clean = word.strip('.,!?;:()[]{}"\'-')
            if clean:
                cleaned.append(clean)

        # Filter stop words and very short words
        keywords = {
            word for word in cleaned
            if word not in self._STOP_WORDS and len(word) > 1
        }

        return keywords

    def intent_similarity(self, intent1: str, intent2: str) -> float:
        """
        Calculate semantic similarity between two intent strings.

        Uses Jaccard similarity on extracted keywords:
        similarity = |intersection| / |union|

        Args:
            intent1: First intent string
            intent2: Second intent string

        Returns:
            Similarity score between 0.0 and 1.0

        Example:
            >>> cycle.intent_similarity(
            ...     "Implement JWT authentication",
            ...     "Add JWT token verification"
            ... )
            0.4  # Both have 'jwt', one has 'authentication'/'verification'
        """
        keywords1 = self.extract_keywords(intent1)
        keywords2 = self.extract_keywords(intent2)

        if not keywords1 or not keywords2:
            return 0.0

        intersection = keywords1 & keywords2
        union = keywords1 | keywords2

        return len(intersection) / len(union)

    def find_by_intent(
        self,
        intent: str,
        min_similarity: float = 0.2,
        limit: int = 10
    ) -> List[Experience]:
        """
        Find experiences with semantically similar intents.

        This is the key method for semantic matching. It finds experiences
        where the intent text is similar, regardless of context categories.

        Args:
            intent: The intent to search for
            min_similarity: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results

        Returns:
            List of experiences sorted by intent similarity (highest first)

        Example:
            >>> cycle.find_by_intent("Add JWT token authentication")
            # Returns experiences with intents like:
            # - "Implement JWT authentication"
            # - "Fix token expiry verification"
            # - "Add OAuth token support"
        """
        experiences = list(self.store._index.values())

        scored = []
        for exp in experiences:
            similarity = self.intent_similarity(intent, exp.intent)
            if similarity >= min_similarity:
                scored.append((exp, similarity))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return just the experiences (not the scores)
        return [exp for exp, _ in scored[:limit]]

    def find_similar_context(
        self,
        context: Context,
        min_similarity: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[Experience, float]]:
        """
        Find experiences with similar contexts.

        This is a wrapper around ExperienceStore.find_similar_context
        for consistency in the API.

        Args:
            context: The context to match
            min_similarity: Minimum context similarity score
            limit: Maximum number of results

        Returns:
            List of (experience, similarity_score) tuples
        """
        return self.store.find_similar_context(context, min_similarity, limit)

    def find_by_context_and_intent(
        self,
        context: Context,
        intent: str,
        context_weight: float = 0.3,
        intent_weight: float = 0.7,
        min_combined_score: float = 0.3,
        limit: int = 10
    ) -> List[Tuple[Experience, float]]:
        """
        Find experiences using both context and intent similarity.

        This combines categorical context matching with semantic intent
        matching to get the best of both worlds.

        Args:
            context: The context to match
            intent: The intent to match
            context_weight: Weight for context similarity (0.0-1.0)
            intent_weight: Weight for intent similarity (0.0-1.0)
            min_combined_score: Minimum combined score to include
            limit: Maximum number of results

        Returns:
            List of (experience, combined_score) tuples

        Example:
            >>> cycle.find_by_context_and_intent(
            ...     context=Context(goal_type="feature", domain="api"),
            ...     intent="Add JWT authentication",
            ...     context_weight=0.3,
            ...     intent_weight=0.7,
            ... )
            # Returns experiences that match BOTH the context and intent,
            # weighted toward intent similarity
        """
        # Normalize weights
        total_weight = context_weight + intent_weight
        if total_weight == 0:
            total_weight = 1.0
        ctx_w = context_weight / total_weight
        int_w = intent_weight / total_weight

        experiences = list(self.store._index.values())

        scored = []
        for exp in experiences:
            # Calculate context similarity
            ctx_sim = context.similarity_to(exp.context)

            # Calculate intent similarity
            int_sim = self.intent_similarity(intent, exp.intent)

            # Combined weighted score
            combined = (ctx_w * ctx_sim) + (int_w * int_sim)

            if combined >= min_combined_score:
                scored.append((exp, combined))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:limit]

    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """
        Get an experience by ID.

        Convenience wrapper around ExperienceStore.get.

        Args:
            experience_id: The ID of the experience to retrieve

        Returns:
            The Experience if found, None otherwise
        """
        return self.store.get(experience_id)
