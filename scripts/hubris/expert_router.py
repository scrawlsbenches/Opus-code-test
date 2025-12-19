#!/usr/bin/env python3
"""
Expert Router

Routes queries to appropriate micro-experts based on intent classification.

The router analyzes the query and context to determine which expert types
are most likely to provide value, similar to how the brain routes sensory
input to relevant cortical regions.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class RoutingDecision:
    """
    Result of routing a query to experts.

    Attributes:
        query: Original query text
        intent: Classified intent
        expert_types: List of expert types to consult
        confidence: Confidence in the routing decision (0-1)
        metadata: Additional routing context
    """
    query: str
    intent: str
    expert_types: List[str]
    confidence: float = 1.0
    metadata: Dict[str, any] = field(default_factory=dict)


class ExpertRouter:
    """
    Routes queries to appropriate experts based on intent classification.

    Uses keyword/pattern matching to classify user intent, then selects
    which expert types are most relevant.

    Intent Types:
        - implement_feature: Adding new functionality
        - fix_bug: Fixing broken behavior
        - debug_error: Diagnosing error messages
        - add_tests: Writing or updating tests
        - refactor: Restructuring code
        - update_docs: Documentation changes
        - code_review: Review-related queries
        - unknown: Cannot classify intent
    """

    # Intent → Expert type mapping
    INTENT_TO_EXPERTS = {
        'implement_feature': ['file', 'test', 'doc'],
        'fix_bug': ['file', 'error', 'test'],
        'debug_error': ['error', 'file'],
        'add_tests': ['test', 'file'],
        'refactor': ['file', 'refactor', 'test'],
        'update_docs': ['doc', 'file'],
        'code_review': ['review', 'test'],
        'unknown': ['file'],  # Default to file expert
    }

    # Intent classification patterns (ordered by specificity)
    INTENT_PATTERNS = {
        'debug_error': [
            r'\b(?:error|exception|traceback|stack\s+trace)\b',
            r'\b(?:failing|crashed|broken)\b',
            r'\b(?:debug|diagnose|investigate)\b.*\b(?:error|issue)\b',
            r'\b(?:why|what).*(?:failing|broken|error)',
        ],
        'fix_bug': [
            r'\bfix\b.*\b(?:bug|issue|problem)\b',
            r'\b(?:bug|issue).*\bfix\b',
            r'\brepair\b',
            r'\bresolve\b.*\b(?:bug|issue)\b',
        ],
        'add_tests': [
            r'\b(?:add|write|create)\b.*\btest',
            r'\btest\b.*\b(?:add|write|create)\b',
            r'\b(?:unit|integration|e2e)\s+test',
            r'\btest\s+coverage\b',
        ],
        'refactor': [
            r'\brefactor\b',
            r'\brestructure\b',
            r'\b(?:clean\s+up|cleanup)\b.*\bcode\b',
            r'\bextract\b.*\b(?:function|class|method)\b',
            r'\bsimplify\b',
        ],
        'update_docs': [
            r'\b(?:update|add|write)\b.*\b(?:doc|documentation|readme)\b',
            r'\b(?:doc|documentation)\b.*\b(?:update|add|write)\b',
            r'\bdocstring',
            r'\bcomment\b',
        ],
        'code_review': [
            r'\breview\b',
            r'\b(?:check|verify|validate)\b.*\b(?:code|implementation)\b',
            r'\bpull\s+request\b',
            r'\bpr\b',
        ],
        'implement_feature': [
            r'\b(?:add|implement|create)\b.*\b(?:feature|functionality)\b',
            r'\b(?:feature|functionality)\b.*\b(?:add|implement|create)\b',
            r'\bnew\b.*\b(?:feature|capability|function|class)\b',
            r'\bbuild\b.*\b(?:feature|component|module)\b',
            r'\badd\b.*\bnew\b',  # "Add new X" pattern
            r'^\s*implement\b(?!.*\b(?:test|doc|review)\b)',  # Implement (not test/doc/review)
        ],
    }

    # Keywords that boost specific expert types
    EXPERT_BOOST_KEYWORDS = {
        'test': [r'\btest', r'\bpytest\b', r'\bunittest\b', r'\bcoverage\b'],
        'doc': [r'\bdoc', r'\breadme\b', r'\bmarkdown\b', r'\.md\b'],
        'error': [r'\berror\b', r'\bexception\b', r'\btraceback\b', r'\bfail'],
        'refactor': [r'\brefactor\b', r'\bclean\b', r'\bsimplify\b'],
    }

    def __init__(self, expert_weights: Optional[Dict[str, float]] = None):
        """
        Initialize router.

        Args:
            expert_weights: Optional dict of expert_type -> weight for dynamic routing
        """
        self.expert_weights = expert_weights or {}

    def classify_intent(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Classify query intent using keyword/pattern matching.

        Args:
            query: User query string
            context: Optional context (e.g., files being modified, errors present)

        Returns:
            Intent string (one of INTENT_TO_EXPERTS keys)
        """
        query_lower = query.lower()

        # Check context for explicit signals
        if context:
            # If error message in context, likely debugging
            if context.get('error_message') or context.get('stack_trace'):
                return 'debug_error'

            # If test files in context, likely test-related
            if any('test' in f.lower() for f in context.get('files', [])):
                return 'add_tests'

            # If doc files in context, likely documentation
            if any(f.endswith('.md') or 'doc' in f.lower()
                   for f in context.get('files', [])):
                return 'update_docs'

        # Pattern-based classification (ordered by specificity)
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        # Default
        return 'unknown'

    def get_experts(
        self,
        query: str,
        context: Optional[Dict] = None,
        top_k: int = 3
    ) -> List[str]:
        """
        Get list of expert types to consult for a query.

        Args:
            query: User query string
            context: Optional context dict
            top_k: Maximum number of expert types to return

        Returns:
            List of expert types (e.g., ['file', 'test', 'doc'])
        """
        # Classify intent
        intent = self.classify_intent(query, context)

        # Get base expert types for this intent
        expert_types = self.INTENT_TO_EXPERTS.get(intent, ['file']).copy()

        # Apply keyword boosts
        query_lower = query.lower()
        boosted_experts: Set[str] = set()

        for expert_type, keywords in self.EXPERT_BOOST_KEYWORDS.items():
            for keyword in keywords:
                if re.search(keyword, query_lower):
                    boosted_experts.add(expert_type)

        # Merge boosted experts (preserve order, add new ones)
        for expert_type in boosted_experts:
            if expert_type not in expert_types:
                expert_types.append(expert_type)

        # Context overrides
        if context and context.get('explicit_experts'):
            expert_types = context['explicit_experts']

        # Apply dynamic routing weights if available
        if self.expert_weights:
            # Sort by weight (higher weight first)
            expert_types.sort(
                key=lambda e: self.expert_weights.get(e, 0.5),
                reverse=True
            )

        # Limit to top_k
        return expert_types[:top_k]

    def route(
        self,
        query: str,
        context: Optional[Dict] = None,
        top_k: int = 3
    ) -> RoutingDecision:
        """
        Route a query to appropriate experts with full decision info.

        Args:
            query: User query string
            context: Optional context dict
            top_k: Maximum number of expert types to return

        Returns:
            RoutingDecision with intent, experts, and confidence
        """
        intent = self.classify_intent(query, context)
        expert_types = self.get_experts(query, context, top_k)

        # Confidence based on pattern match strength
        confidence = 1.0 if intent != 'unknown' else 0.5

        # If context provided strong signals, boost confidence
        if context:
            if context.get('error_message') or context.get('explicit_experts'):
                confidence = min(1.0, confidence + 0.2)

        return RoutingDecision(
            query=query,
            intent=intent,
            expert_types=expert_types,
            confidence=confidence,
            metadata={
                'context_used': context is not None,
                'weights_applied': bool(self.expert_weights)
            }
        )

    def update_weights(self, expert_weights: Dict[str, float]) -> None:
        """
        Update dynamic routing weights.

        Weights represent learned performance of each expert type.
        Higher weights increase routing priority.

        Args:
            expert_weights: Dict of expert_type -> weight (0-1 typical)
        """
        self.expert_weights = expert_weights

    def get_weights(self) -> Dict[str, float]:
        """
        Get current routing weights.

        Returns:
            Dict of expert_type -> weight
        """
        return self.expert_weights.copy()
