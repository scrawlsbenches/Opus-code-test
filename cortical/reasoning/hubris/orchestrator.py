"""
HubrisMoE: Mixture of Experts Orchestrator.

The HubrisMoE orchestrator coordinates multiple MicroExperts to solve
complex problems. It handles:
- Expert selection based on competence
- Response combination with calibrated weights
- Credit tracking and staking
- Value signal learning

Design Philosophy:
    The name "Hubris" reminds us that overconfidence destroys systems.
    A well-orchestrated MoE knows when to trust each expert, when to
    combine perspectives, and when to admit uncertainty.

Key Concepts:
    - Expert Pool: Collection of specialized micro-experts
    - Selection: Choosing which experts to consult
    - Combination: Merging expert responses intelligently
    - Calibration: Ensuring expert confidence matches accuracy

Example:
    >>> moe = HubrisMoE()
    >>> moe.register_expert(MicroExpert("nlp", "nlp", ["parsing"]))
    >>> moe.register_expert(MicroExpert("cv", "cv", ["detection"]))
    >>> result = moe.query("Parse this sentence")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math

from .expert import MicroExpert, MetaExpert, ExpertResponse, CombinedResponse
from .credit import CreditLedger
from .value import ValueSignal
from .staking import StakingManager


@dataclass
class QueryResult:
    """
    Result from a query to the MoE system.

    Contains the final answer along with metadata about which experts
    contributed and their confidence levels.
    """
    answer: Any
    confidence: float
    contributing_experts: List[str]
    expert_responses: List[ExpertResponse]
    combination_method: str
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    # Knowledge grounding
    grounding_docs: List[str] = field(default_factory=list)
    prediction_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))


class CombinationStrategy:
    """
    Strategies for combining expert responses.

    Different strategies suit different scenarios:
    - Weighted average for continuous/numeric outputs
    - Majority vote for discrete choices
    - Max confidence for selecting best single answer
    - Consensus for requiring agreement
    """

    @staticmethod
    def weighted_average(
        responses: List[ExpertResponse],
        value_signal: Optional[ValueSignal] = None,
    ) -> CombinedResponse:
        """
        Combine responses using confidence-weighted averaging.

        Weights are based on expert confidence, optionally adjusted
        by value signals from past performance.
        """
        if not responses:
            return CombinedResponse(
                final_answer=None,
                confidence=0.0,
                contributing_experts=[],
            )

        # Compute weights
        weights = []
        for r in responses:
            weight = r.confidence
            # Adjust by value signal if available
            if value_signal:
                value_boost = value_signal.get_value(r.expert.name)
                weight *= (1.0 + value_boost)
            weights.append(max(0.0, weight))

        total_weight = sum(weights)
        if total_weight == 0:
            return CombinedResponse(
                final_answer=responses[0].answer if responses else None,
                confidence=0.0,
                contributing_experts=[r.expert.name for r in responses],
            )

        # Normalize weights
        norm_weights = [w / total_weight for w in weights]

        # Select answer with highest weight
        best_idx = max(range(len(responses)), key=lambda i: weights[i])
        final_answer = responses[best_idx].answer

        # Combined confidence is weighted average
        combined_conf = sum(w * r.confidence for w, r in zip(norm_weights, responses))

        return CombinedResponse(
            final_answer=final_answer,
            confidence=combined_conf,
            contributing_experts=[r.expert for r in responses],
            combination_method="weighted_average",
        )

    @staticmethod
    def majority_vote(responses: List[ExpertResponse]) -> CombinedResponse:
        """
        Combine responses using majority voting.

        Each expert's vote is weighted by confidence.
        """
        if not responses:
            return CombinedResponse(
                final_answer=None,
                confidence=0.0,
                contributing_experts=[],
            )

        # Count votes per answer, weighted by confidence
        votes: Dict[Any, float] = {}
        for r in responses:
            if r.answer not in votes:
                votes[r.answer] = 0.0
            votes[r.answer] += r.confidence

        # Find winner
        winner = max(votes.items(), key=lambda x: x[1])
        total_votes = sum(votes.values())

        return CombinedResponse(
            final_answer=winner[0],
            confidence=winner[1] / total_votes if total_votes > 0 else 0.0,
            contributing_experts=[r.expert for r in responses],
            combination_method="majority_vote",
        )

    @staticmethod
    def max_confidence(responses: List[ExpertResponse]) -> CombinedResponse:
        """
        Select the response with highest confidence.

        Simple but effective when experts have clear competence boundaries.
        """
        if not responses:
            return CombinedResponse(
                final_answer=None,
                confidence=0.0,
                contributing_experts=[],
            )

        best = max(responses, key=lambda r: r.confidence)

        return CombinedResponse(
            final_answer=best.answer,
            confidence=best.confidence,
            contributing_experts=[best.expert],
            combination_method="max_confidence",
        )

    @staticmethod
    def consensus(
        responses: List[ExpertResponse],
        threshold: float = 0.7,
    ) -> CombinedResponse:
        """
        Require consensus among experts.

        Only returns confident answer if enough experts agree.
        """
        if not responses:
            return CombinedResponse(
                final_answer=None,
                confidence=0.0,
                contributing_experts=[],
            )

        # Group by answer
        answer_groups: Dict[Any, List[ExpertResponse]] = {}
        for r in responses:
            if r.answer not in answer_groups:
                answer_groups[r.answer] = []
            answer_groups[r.answer].append(r)

        # Find largest group
        if not answer_groups:
            return CombinedResponse(
                final_answer=None,
                confidence=0.0,
                contributing_experts=[],
            )

        largest_group = max(answer_groups.items(), key=lambda x: len(x[1]))
        consensus_fraction = len(largest_group[1]) / len(responses)

        if consensus_fraction >= threshold:
            avg_conf = sum(r.confidence for r in largest_group[1]) / len(largest_group[1])
            return CombinedResponse(
                final_answer=largest_group[0],
                confidence=avg_conf * consensus_fraction,
                contributing_experts=[r.expert for r in largest_group[1]],
                combination_method="consensus",
            )
        else:
            # No consensus
            return CombinedResponse(
                final_answer=None,
                confidence=0.0,
                contributing_experts=[r.expert for r in responses],
                combination_method="consensus",
                dissent=[r for r in responses if r.answer != largest_group[0]],
            )


class HubrisMoE:
    """
    Mixture of Experts orchestrator.

    Coordinates multiple specialized experts to answer queries,
    combining their responses intelligently based on competence
    and past performance.

    Features:
        - Expert registration and management
        - Competence-based expert selection
        - Multiple combination strategies
        - Credit/calibration tracking via CreditLedger
        - Value signal learning for expert weighting
        - Optional staking for commitment mechanism

    Example:
        >>> moe = HubrisMoE()
        >>>
        >>> # Register experts
        >>> moe.register_expert(MicroExpert("parser", "nlp", ["parsing", "syntax"]))
        >>> moe.register_expert(MicroExpert("semantic", "nlp", ["meaning", "intent"]))
        >>>
        >>> # Query
        >>> result = moe.query("What is the structure of this sentence?")
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        enable_credit_tracking: bool = True,
        enable_value_learning: bool = True,
        enable_staking: bool = False,
        enable_cel: bool = False,
        enable_got: bool = False,
        combination_strategy: str = "weighted_average",
        selection_threshold: float = 0.3,
        max_experts_per_query: int = 5,
        knowledge_graph: Optional[Any] = None,
    ):
        """
        Initialize the MoE orchestrator.

        Args:
            enable_credit_tracking: Track expert performance via CreditLedger
            enable_value_learning: Learn expert values via ValueSignal
            enable_staking: Enable staking mechanism
            enable_cel: Enable CEL event logging
            enable_got: Enable GoT decision tracking
            combination_strategy: Default combination strategy
            selection_threshold: Minimum competence for expert selection
            max_experts_per_query: Maximum experts to consult per query
            knowledge_graph: Optional SemanticKnowledgeGraph for grounding
        """
        self._experts: Dict[str, MicroExpert] = {}
        self._combination_strategy = combination_strategy
        self._selection_threshold = selection_threshold
        self._max_experts = max_experts_per_query

        # Knowledge graph integration
        self._knowledge_graph = knowledge_graph

        # Integration flags
        self._enable_cel = enable_cel
        self._enable_got = enable_got

        # CEL event log (simple list for CEL integration)
        self._cel_events: List[Dict[str, Any]] = []

        # GoT decisions
        self._decisions: List[Any] = []

        # Subsystems
        self._credit_ledger = CreditLedger() if enable_credit_tracking else None
        self._value_signal = ValueSignal() if enable_value_learning else None
        self._staking = StakingManager() if enable_staking else None

        # Statistics
        self._query_count = 0
        self._expert_usage: Dict[str, int] = {}

    def register_expert(self, expert: MicroExpert) -> None:
        """
        Register an expert with the MoE.

        Args:
            expert: The expert to register
        """
        self._experts[expert.name] = expert
        self._expert_usage[expert.name] = 0

    def unregister_expert(self, name: str) -> Optional[MicroExpert]:
        """
        Unregister an expert.

        Args:
            name: Name of expert to remove

        Returns:
            The removed expert, or None if not found
        """
        if name in self._experts:
            expert = self._experts.pop(name)
            return expert
        return None

    def get_expert(self, name: str) -> Optional[MicroExpert]:
        """Get expert by name."""
        return self._experts.get(name)

    def list_experts(self) -> List[str]:
        """List all registered expert names."""
        return list(self._experts.keys())

    def select_experts(
        self,
        query: str,
        threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[MicroExpert]:
        """
        Select experts competent for a query.

        Args:
            query: The query to find experts for
            threshold: Minimum competence (default: self._selection_threshold)
            max_count: Maximum experts to return (default: self._max_experts)

        Returns:
            List of selected experts, sorted by competence (highest first)
        """
        threshold = threshold or self._selection_threshold
        max_count = max_count or self._max_experts

        # Score all experts
        scored = []
        for expert in self._experts.values():
            competence = expert.estimate_competence(query)
            if competence >= threshold:
                scored.append((expert, competence))

        # Sort by competence (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top experts
        return [e for e, _ in scored[:max_count]]

    def query_expert(
        self,
        expert: MicroExpert,
        query: str,
    ) -> ExpertResponse:
        """
        Query a specific expert.

        Args:
            expert: Expert to query
            query: The query

        Returns:
            ExpertResponse from the expert
        """
        response = expert.respond(query)
        self._expert_usage[expert.name] = self._expert_usage.get(expert.name, 0) + 1
        return response

    def query(
        self,
        query: str,
        strategy: Optional[str] = None,
        experts: Optional[List[MicroExpert]] = None,
    ) -> QueryResult:
        """
        Query the MoE system.

        Selects appropriate experts, collects their responses, and
        combines them using the specified strategy.

        Args:
            query: The query to answer
            strategy: Combination strategy (default: self._combination_strategy)
            experts: Specific experts to use (default: auto-select)

        Returns:
            QueryResult with answer and metadata
        """
        import time
        start_time = time.time()
        self._query_count += 1

        strategy = strategy or self._combination_strategy

        # Select experts if not provided
        if experts is None:
            experts = self.select_experts(query)

        if not experts:
            return QueryResult(
                answer=None,
                confidence=0.0,
                contributing_experts=[],
                expert_responses=[],
                combination_method=strategy,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Collect responses
        responses = []
        for expert in experts:
            response = self.query_expert(expert, query)
            if not response.abstained:
                responses.append(response)

        # Combine responses
        combined = self._combine_responses(responses, strategy)

        # Update value signal based on expert performance
        if self._value_signal and responses:
            for r in responses:
                # Reward experts proportional to their contribution
                contribution = r.confidence / sum(resp.confidence for resp in responses)
                self._value_signal.reward(r.expert.name, contribution)

        return QueryResult(
            answer=combined.final_answer,
            confidence=combined.confidence,
            contributing_experts=[getattr(e, 'name', str(e)) for e in combined.contributing_experts],
            expert_responses=responses,
            combination_method=strategy,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _combine_responses(
        self,
        responses: List[ExpertResponse],
        strategy: str,
    ) -> CombinedResponse:
        """Combine responses using the specified strategy."""
        if strategy == "weighted_average":
            return CombinationStrategy.weighted_average(responses, self._value_signal)
        elif strategy == "majority_vote":
            return CombinationStrategy.majority_vote(responses)
        elif strategy == "max_confidence":
            return CombinationStrategy.max_confidence(responses)
        elif strategy == "consensus":
            return CombinationStrategy.consensus(responses)
        else:
            return CombinationStrategy.weighted_average(responses, self._value_signal)

    def combine_responses(self, responses: List[ExpertResponse]) -> CombinedResponse:
        """
        Public method to combine responses.

        Uses the default combination strategy.
        """
        return self._combine_responses(responses, self._combination_strategy)

    def record_outcome(
        self,
        result: QueryResult,
        correct: bool,
    ) -> None:
        """
        Record the outcome of a query for learning.

        Updates credit ledger, value signals, and expert competencies
        based on whether the answer was correct.

        Args:
            result: The query result to record
            correct: Whether the answer was correct
        """
        for response in result.expert_responses:
            expert = response.expert

            # Update credit ledger
            if self._credit_ledger:
                self._credit_ledger.record_prediction(
                    expert_id=expert.name,
                    confidence=response.confidence,
                    correct=correct,
                )

            # Update value signal
            if self._value_signal:
                reward = 1.0 if correct else -0.5
                # Scale by confidence (overconfident wrong = worse)
                if not correct:
                    reward *= response.confidence
                self._value_signal.reward(expert.name, reward)

            # Update expert competencies
            expert.record_feedback(result.answer, correct)

    def get_expert_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific expert."""
        stats = {
            'usage_count': self._expert_usage.get(name, 0),
        }

        if self._credit_ledger:
            stats['credit'] = self._credit_ledger.get_stats(name)

        if self._value_signal:
            stats['value'] = self._value_signal.get_value(name)
            stats['value_confidence'] = self._value_signal.get_confidence(name)

        expert = self._experts.get(name)
        if expert:
            stats['expert'] = expert.get_stats()

        return stats

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration report for all experts."""
        if not self._credit_ledger:
            return {'enabled': False}

        report = {
            'enabled': True,
            'experts': {},
        }

        for name in self._experts:
            stats = self._credit_ledger.get_stats(name)
            report['experts'][name] = {
                'ece': stats.get('ece', 0.0),
                'mce': stats.get('mce', 0.0),
                'accuracy': stats.get('accuracy', 0.5),
                'predictions': stats.get('predictions', 0),
            }

        return report

    def get_summary(self) -> Dict[str, Any]:
        """Get MoE system summary."""
        return {
            'total_experts': len(self._experts),
            'total_queries': self._query_count,
            'combination_strategy': self._combination_strategy,
            'selection_threshold': self._selection_threshold,
            'credit_tracking': self._credit_ledger is not None,
            'value_learning': self._value_signal is not None,
            'staking_enabled': self._staking is not None,
            'knowledge_graph': self._knowledge_graph is not None,
            'expert_usage': dict(sorted(
                self._expert_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10 most used
        }

    # =========================================================================
    # Knowledge Graph Integration
    # =========================================================================

    def query_with_grounding(
        self,
        query: str,
        strategy: Optional[str] = None,
    ) -> QueryResult:
        """
        Query with knowledge graph grounding.

        Searches the knowledge graph for relevant documents and uses them
        to ground the expert responses.

        Args:
            query: The query to answer
            strategy: Combination strategy

        Returns:
            QueryResult with grounding_docs populated
        """
        grounding_docs = []

        # Search knowledge graph for grounding
        if self._knowledge_graph is not None:
            try:
                results = self._knowledge_graph.search(query, limit=5)
                grounding_docs = [r.doc_id for r in results]
            except Exception:
                pass  # Gracefully handle if search fails

        # Perform normal query
        result = self.query(query, strategy=strategy)

        # Add grounding info
        result.grounding_docs = grounding_docs

        # Log CEL event
        if self._enable_cel:
            self._log_cel_event("expert_query", {
                "query": query,
                "confidence": result.confidence,
                "experts": result.contributing_experts,
                "grounding_docs": len(grounding_docs),
            })

        return result

    def get_knowledge_grounding(self, query: str, limit: int = 5) -> List[Any]:
        """
        Get knowledge grounding documents for a query.

        Args:
            query: Query to ground
            limit: Maximum documents to return

        Returns:
            List of SearchResult from knowledge graph
        """
        if self._knowledge_graph is None:
            return []

        try:
            return self._knowledge_graph.search(query, limit=limit)
        except Exception:
            return []

    # =========================================================================
    # CEL Integration
    # =========================================================================

    def _log_cel_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a CEL event."""
        if not self._enable_cel:
            return

        self._cel_events.append({
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
        })

    def get_cel_events(self) -> List[Dict[str, Any]]:
        """Get all logged CEL events."""
        return list(self._cel_events)

    def record_outcome(
        self,
        prediction_id_or_result: Any,
        correct: bool,
    ) -> None:
        """
        Record the outcome of a query for learning.

        Updates credit ledger, value signals, and expert competencies
        based on whether the answer was correct.

        Args:
            prediction_id_or_result: QueryResult or prediction_id string
            correct: Whether the answer was correct
        """
        # Handle both QueryResult and prediction_id
        if hasattr(prediction_id_or_result, 'expert_responses'):
            result = prediction_id_or_result
            for response in result.expert_responses:
                expert = response.expert

                # Update credit ledger
                if self._credit_ledger:
                    self._credit_ledger.record_prediction(
                        expert_id=expert.name,
                        confidence=response.confidence,
                        correct=correct,
                    )

                # Update value signal
                if self._value_signal:
                    reward = 1.0 if correct else -0.5
                    # Scale by confidence (overconfident wrong = worse)
                    if not correct:
                        reward *= response.confidence
                    self._value_signal.reward(expert.name, reward)

                # Update expert competencies
                expert.record_feedback(result.answer, correct)

            # Log CEL event
            if self._enable_cel:
                self._log_cel_event("prediction_outcome", {
                    "prediction_id": getattr(result, 'prediction_id', 'unknown'),
                    "correct": correct,
                    "confidence": result.confidence,
                })

    # =========================================================================
    # GoT Integration
    # =========================================================================

    def create_decision(
        self,
        question: str,
        chosen: Any,
        consultation_result: Optional[QueryResult] = None,
        rationale: str = "",
    ) -> Any:
        """
        Create a GoT decision record from expert consultation.

        Args:
            question: The decision question
            chosen: The chosen answer
            consultation_result: Optional QueryResult from consultation
            rationale: Explanation for the decision

        Returns:
            Decision object if GoT enabled
        """
        if not self._enable_got:
            return None

        from dataclasses import dataclass

        @dataclass
        class Decision:
            decision_id: str
            question: str
            chosen: Any
            rationale: str
            contributing_experts: List[str]
            confidence: float
            created_at: datetime

        contributing_experts = []
        confidence = 0.0

        if consultation_result:
            contributing_experts = consultation_result.contributing_experts
            confidence = consultation_result.confidence
            if not rationale:
                rationale = f"Based on {len(contributing_experts)} expert(s) with {confidence:.1%} confidence"

        decision = Decision(
            decision_id=f"dec_{datetime.now().timestamp()}",
            question=question,
            chosen=chosen,
            rationale=rationale,
            contributing_experts=contributing_experts,
            confidence=confidence,
            created_at=datetime.now(),
        )

        self._decisions.append(decision)

        # Log CEL event
        if self._enable_cel:
            self._log_cel_event("decision_created", {
                "decision_id": decision.decision_id,
                "question": question,
                "experts": len(contributing_experts),
                "confidence": confidence,
            })

        return decision

    def get_decisions(self) -> List[Any]:
        """Get all recorded decisions."""
        return list(self._decisions)
