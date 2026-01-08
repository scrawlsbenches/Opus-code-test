"""
MicroExpert: Specialized experts in the Hubris MoE system.

A MicroExpert is a narrow specialist that knows what it knows—and crucially,
knows what it doesn't know. The name "Hubris" reminds us that overconfidence
is the enemy of good reasoning.

Design Philosophy:
    Experts should specialize deeply rather than broadly. A well-calibrated
    expert who admits uncertainty is more valuable than an overconfident
    generalist who confidently provides wrong answers.

Key Concepts:
    - Competencies: Specific skills the expert has developed
    - Domain: The broad area of expertise
    - Abstention: The ability to say "I don't know"
    - Confidence calibration: Matching confidence to actual accuracy

Example:
    >>> expert = MicroExpert(
    ...     name="sql_expert",
    ...     domain="databases",
    ...     competencies=["query_optimization", "schema_design"]
    ... )
    >>> response = expert.respond("How do I optimize this query?")
    >>> print(response.confidence)  # Should be high (in domain)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import uuid


@dataclass
class Competency:
    """
    A specific competency that an expert possesses.

    Competencies are more granular than domains. An expert in "databases"
    might have competencies in ["query_optimization", "schema_design"]
    but not ["distributed_transactions"].
    """
    name: str
    score: float = 0.5  # 0.0 to 1.0, learned over time
    experience_count: int = 0  # Number of queries handled
    success_count: int = 0  # Number of successful responses
    last_used: Optional[datetime] = None

    def accuracy(self) -> float:
        """Compute accuracy rate for this competency."""
        if self.experience_count == 0:
            return 0.5  # Prior
        return self.success_count / self.experience_count

    def update(self, success: bool) -> None:
        """Update competency based on outcome."""
        self.experience_count += 1
        if success:
            self.success_count += 1
        self.last_used = datetime.now()
        # Update score with exponential moving average
        alpha = 0.1
        outcome = 1.0 if success else 0.0
        self.score = alpha * outcome + (1 - alpha) * self.score


@dataclass
class ExpertResponse:
    """
    A response from an expert with associated metadata.

    The response includes not just the answer but also the expert's
    confidence, reasoning trace, and whether they chose to abstain.
    """
    expert: 'MicroExpert'
    answer: Any
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    abstained: bool = False
    competencies_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class CombinedResponse:
    """
    A combined response from multiple experts.

    When multiple experts contribute, their responses are weighted
    and combined into a final answer.
    """
    final_answer: Any
    confidence: float
    contributing_experts: List[ExpertResponse]
    combination_method: str = "weighted_average"
    dissent: List[ExpertResponse] = field(default_factory=list)  # Disagreeing experts


class MicroExpert:
    """
    A specialized micro-expert in the Hubris MoE system.

    MicroExperts are narrow specialists that:
    - Have specific competencies they excel at
    - Can abstain when facing unfamiliar queries
    - Track their own performance for calibration
    - Learn and improve from feedback

    The "micro" emphasizes specialization over breadth. A system with
    many well-calibrated micro-experts outperforms one with a few
    overconfident generalists.

    Attributes:
        name: Unique identifier for the expert
        domain: Broad area of expertise (e.g., "databases", "nlp")
        competencies: Specific skills within the domain
        abstention_threshold: Minimum confidence to provide answer
        response_handler: Optional custom handler for generating responses

    Example:
        >>> expert = MicroExpert(
        ...     name="sql_optimizer",
        ...     domain="databases",
        ...     competencies=["query_optimization", "index_design"]
        ... )
        >>> if expert.can_handle("optimize SQL query"):
        ...     response = expert.respond("optimize SQL query")
    """

    def __init__(
        self,
        name: str,
        domain: str,
        competencies: Optional[List[str]] = None,
        abstention_threshold: float = 0.3,
        response_handler: Optional[Callable[[str], ExpertResponse]] = None,
    ):
        """
        Initialize a MicroExpert.

        Args:
            name: Unique identifier
            domain: Broad expertise area
            competencies: List of specific skills
            abstention_threshold: Minimum confidence to respond (0.0-1.0)
            response_handler: Custom response generation function
        """
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.domain = domain
        self.abstention_threshold = abstention_threshold
        self._response_handler = response_handler
        self.created_at = datetime.now()

        # Competency tracking
        self._competencies: Dict[str, Competency] = {}
        for comp in (competencies or []):
            self._competencies[comp] = Competency(name=comp)

        # Performance tracking
        self._query_count = 0
        self._abstention_count = 0
        self._correct_count = 0

        # Keywords associated with this expert's domain
        self._domain_keywords: Set[str] = set()
        self._update_domain_keywords()

    def _update_domain_keywords(self) -> None:
        """Update domain keywords from domain and competencies."""
        # Add domain as keyword
        self._domain_keywords.add(self.domain.lower())

        # Add competencies as keywords
        for comp in self._competencies:
            # Split competency name into words
            words = comp.lower().replace("_", " ").split()
            self._domain_keywords.update(words)

    @property
    def competencies(self) -> List[str]:
        """Get list of competency names."""
        return list(self._competencies.keys())

    def get_competency_score(self, name: str) -> float:
        """Get score for a specific competency."""
        if name in self._competencies:
            return self._competencies[name].score
        return 0.0

    def add_competency(self, name: str, initial_score: float = 0.5) -> None:
        """Add a new competency."""
        if name not in self._competencies:
            self._competencies[name] = Competency(name=name, score=initial_score)
            self._update_domain_keywords()

    def estimate_competence(self, query: str) -> float:
        """
        Estimate competence level for a given query.

        Uses keyword matching and competency scores to estimate
        how well this expert can handle the query.

        Args:
            query: The query to evaluate

        Returns:
            Estimated competence score (0.0 to 1.0)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Check domain keyword overlap
        keyword_overlap = len(query_words & self._domain_keywords)
        keyword_score = min(keyword_overlap / max(len(self._domain_keywords), 1), 1.0)

        # Check competency relevance
        competency_scores = []
        for comp_name, comp in self._competencies.items():
            comp_words = set(comp_name.lower().replace("_", " ").split())
            if comp_words & query_words:
                competency_scores.append(comp.score)

        if competency_scores:
            avg_competency = sum(competency_scores) / len(competency_scores)
        else:
            avg_competency = 0.2  # Low default if no competency match

        # Combined score
        return 0.4 * keyword_score + 0.6 * avg_competency

    def can_handle(self, query: str) -> bool:
        """Check if expert can handle this query (above abstention threshold)."""
        return self.estimate_competence(query) >= self.abstention_threshold

    def respond(self, query: str) -> ExpertResponse:
        """
        Generate a response to a query.

        If the expert's estimated competence is below the abstention
        threshold, they will abstain rather than provide a low-quality answer.

        Args:
            query: The query to respond to

        Returns:
            ExpertResponse with answer, confidence, and metadata
        """
        self._query_count += 1
        competence = self.estimate_competence(query)

        # Check for abstention
        if competence < self.abstention_threshold:
            self._abstention_count += 1
            return ExpertResponse(
                expert=self,
                answer=None,
                confidence=competence,
                reasoning=f"Competence ({competence:.2f}) below threshold ({self.abstention_threshold})",
                abstained=True,
            )

        # Use custom handler if provided
        if self._response_handler:
            response = self._response_handler(query)
            response.expert = self
            return response

        # Default response generation
        relevant_competencies = [
            comp for comp in self._competencies
            if any(w in query.lower() for w in comp.lower().replace("_", " ").split())
        ]

        return ExpertResponse(
            expert=self,
            answer=f"Response from {self.name} regarding: {query[:50]}...",
            confidence=competence,
            reasoning=f"Based on competencies: {relevant_competencies or ['general domain knowledge']}",
            competencies_used=relevant_competencies,
        )

    def record_feedback(self, query: str, correct: bool) -> None:
        """
        Record feedback on a response for learning.

        Args:
            query: The query that was answered
            correct: Whether the response was correct
        """
        if correct:
            self._correct_count += 1

        # Update relevant competencies
        query_lower = query.lower()
        for comp_name, comp in self._competencies.items():
            comp_words = set(comp_name.lower().replace("_", " ").split())
            if any(w in query_lower for w in comp_words):
                comp.update(correct)

    def accuracy(self) -> float:
        """Compute overall accuracy rate."""
        answered = self._query_count - self._abstention_count
        if answered == 0:
            return 0.5  # Prior
        return self._correct_count / answered

    def abstention_rate(self) -> float:
        """Compute abstention rate."""
        if self._query_count == 0:
            return 0.0
        return self._abstention_count / self._query_count

    def get_stats(self) -> Dict[str, Any]:
        """Get expert statistics."""
        return {
            'name': self.name,
            'domain': self.domain,
            'competencies': self.competencies,
            'query_count': self._query_count,
            'abstention_count': self._abstention_count,
            'correct_count': self._correct_count,
            'accuracy': self.accuracy(),
            'abstention_rate': self.abstention_rate(),
        }


class MetaExpert(MicroExpert):
    """
    A meta-expert that coordinates multiple sub-experts.

    MetaExperts sit at higher levels in the expert hierarchy,
    routing queries to appropriate sub-experts and combining
    their responses.

    Example:
        >>> syntax = MicroExpert("syntax", "code", ["parsing"])
        >>> semantic = MicroExpert("semantic", "code", ["meaning"])
        >>> code_meta = MetaExpert("code_meta", "code", [syntax, semantic])
        >>> response = code_meta.respond("analyze this code")
    """

    def __init__(
        self,
        name: str,
        domain: str,
        sub_experts: Optional[List[MicroExpert]] = None,
        competencies: Optional[List[str]] = None,
        abstention_threshold: float = 0.3,
    ):
        """
        Initialize a MetaExpert.

        Args:
            name: Unique identifier
            domain: Broad expertise area
            sub_experts: List of subordinate experts to coordinate
            competencies: Direct competencies (in addition to sub-experts)
            abstention_threshold: Minimum confidence to respond
        """
        super().__init__(
            name=name,
            domain=domain,
            competencies=competencies,
            abstention_threshold=abstention_threshold,
        )
        self.sub_experts: List[MicroExpert] = sub_experts or []

        # Update keywords from sub-experts
        for expert in self.sub_experts:
            self._domain_keywords.update(expert._domain_keywords)

    def add_sub_expert(self, expert: MicroExpert) -> None:
        """Add a sub-expert."""
        self.sub_experts.append(expert)
        self._domain_keywords.update(expert._domain_keywords)

    def estimate_competence(self, query: str) -> float:
        """
        Estimate competence including sub-experts.

        A meta-expert's competence is the maximum of its own
        competence and its sub-experts' competences.
        """
        own_competence = super().estimate_competence(query)

        if not self.sub_experts:
            return own_competence

        sub_competences = [e.estimate_competence(query) for e in self.sub_experts]
        max_sub = max(sub_competences) if sub_competences else 0.0

        # Meta-expert is as competent as its best sub-expert
        return max(own_competence, max_sub)

    def respond(self, query: str) -> ExpertResponse:
        """
        Generate response by coordinating sub-experts.

        Routes query to relevant sub-experts and combines their responses.
        """
        self._query_count += 1
        competence = self.estimate_competence(query)

        if competence < self.abstention_threshold:
            self._abstention_count += 1
            return ExpertResponse(
                expert=self,
                answer=None,
                confidence=competence,
                abstained=True,
            )

        # Get responses from capable sub-experts
        sub_responses = []
        for expert in self.sub_experts:
            if expert.can_handle(query):
                response = expert.respond(query)
                if not response.abstained:
                    sub_responses.append(response)

        if not sub_responses:
            # Fall back to own response
            return super().respond(query)

        # Combine sub-expert responses
        # Weight by confidence
        total_confidence = sum(r.confidence for r in sub_responses)
        if total_confidence == 0:
            return super().respond(query)

        # Use highest confidence response as primary
        best_response = max(sub_responses, key=lambda r: r.confidence)

        return ExpertResponse(
            expert=self,
            answer=best_response.answer,
            confidence=competence,
            reasoning=f"Coordinated {len(sub_responses)} sub-experts; primary: {best_response.expert.name}",
            competencies_used=[r.expert.name for r in sub_responses],
        )


class ExpertTrainer:
    """
    Trainer for improving expert competencies.

    Provides methods for training experts on domain-specific data,
    enabling specialization through focused learning.
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize trainer.

        Args:
            learning_rate: Rate at which competencies improve
        """
        self.learning_rate = learning_rate
        self._training_history: List[Dict[str, Any]] = []

    def train(
        self,
        expert: MicroExpert,
        training_data: List[tuple],
    ) -> Dict[str, float]:
        """
        Train an expert on domain-specific data.

        Args:
            expert: The expert to train
            training_data: List of (query, domain_tag) tuples

        Returns:
            Dictionary of competency improvements
        """
        improvements = {}

        for query, domain_tag in training_data:
            # Add competency if new
            if domain_tag not in expert._competencies:
                expert.add_competency(domain_tag)

            # Simulate successful handling (training data is in-domain)
            comp = expert._competencies[domain_tag]
            old_score = comp.score
            comp.score = min(1.0, comp.score + self.learning_rate)
            comp.experience_count += 1
            comp.success_count += 1

            if domain_tag not in improvements:
                improvements[domain_tag] = 0.0
            improvements[domain_tag] += comp.score - old_score

        # Record training
        self._training_history.append({
            'expert': expert.name,
            'samples': len(training_data),
            'improvements': improvements,
            'timestamp': datetime.now(),
        })

        return improvements

    def get_training_history(self, expert_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get training history, optionally filtered by expert."""
        if expert_name:
            return [h for h in self._training_history if h['expert'] == expert_name]
        return self._training_history
