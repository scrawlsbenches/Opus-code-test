"""
Unified Query Router for Cognitive Agent.

Routes natural language questions to appropriate backends:
- CDG queries (SQL-like: FROM task WHERE ...)
- Audit queries (risky files, why is X flagged)
- Code intent queries (where do we handle X)
- Semantic graph queries (what is X, associations)

This module integrates:
- cortical.cdg.query (CDG query language)
- cortical.audits.reasoning (AuditQuery, PLN reasoning)
- cortical.query.intent (code intent parsing)
- cortical.cognitive.nl_query (semantic associations)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# Import existing query infrastructure
from cortical.audits.reasoning import (
    AuditQuery,
    translate_audit_query,
    is_natural_language_query,
)

if TYPE_CHECKING:
    from cortical.cdg.query.ast import CDGQuery
    from cortical.query.intent import ParsedIntent


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnifiedQuery:
    """
    Unified query representation across all backends.

    Attributes:
        raw_question: Original user question
        query_type: Backend type (cdg, audit, code, semantic)
        parsed: Parsed query structure specific to backend
        confidence: Router confidence in this classification (0.0-1.0)
        metadata: Additional routing metadata
    """
    raw_question: str
    query_type: str  # cdg, audit, code, semantic
    parsed: Any  # Union[CDGQuery, AuditQuery, ParsedIntent, QueryIntent]
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryIntent:
    """
    Semantic query intent for graph-based queries.

    Used when query falls back to semantic associations.
    """
    question_type: str  # how, what, where, who, why, general
    concepts: List[str]  # extracted key concepts
    raw_question: str = ""


# =============================================================================
# Query Patterns
# =============================================================================

# CDG query patterns (SQL-like)
CDG_PATTERNS = [
    r'^\s*FROM\s+\w+',  # FROM task ...
    r'^\s*SELECT\s+',   # SELECT ...
    r'\bWHERE\s+\w+\s*=',  # ... WHERE status = ...
    r'^\s*blockers\s*\(',  # blockers('T-123')
    r'^\s*depends_on\s*\(',  # depends_on('T-123')
]

# Audit query patterns
AUDIT_PATTERNS = [
    r'\brisky\s+files?\b',
    r'\bwhy\s+is\s+\S+\s+(?:flagged|risky|marked)\b',
    r'\bexplain\s+\S+',
    r'\bhigh[_\s]?risk\b',
    r'\bcritical\s+files?\b',
    r'\bwith\s+(?:high[_\s]?churn|todo|fixme|hack)\b',
]

# Code intent patterns
CODE_PATTERNS = [
    r'\bwhere\s+(?:do\s+we|does?\s+\w+|is\s+\w+)\s+(?:handle|process|implement)\b',
    r'\bhow\s+(?:do\s+we|does?\s+\w+|is\s+\w+)\s+(?:handle|process|implement)\b',
    r'\bhow\s+does\s+.*\s+implement\b',  # "how does X implement Y"
    r'\bwho\s+(?:calls?|uses?|implements?)\b',
    r'\bwhat\s+(?:calls?|uses?|implements?)\b',
    r'\bfind\s+(?:all\s+)?(?:callers?|usages?|implementations?)\s+of\b',
]

# Stop words for concept extraction
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "what", "where", "why", "how", "who", "which", "when",
    "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "you", "your", "we", "our", "they", "their", "he", "she", "him", "her",
}


# =============================================================================
# Query Router
# =============================================================================

class QueryRouter:
    """
    Routes natural language questions to appropriate query backends.

    The router analyzes the question structure and content to determine
    which backend (CDG, Audit, Code, Semantic) is best suited to answer it.

    Usage:
        router = QueryRouter()
        unified = router.route("risky files in cortical/")
        # unified.query_type == "audit"
        # unified.parsed == AuditQuery(...)
    """

    def __init__(self):
        """Initialize the query router."""
        self._cdg_patterns = [re.compile(p, re.IGNORECASE) for p in CDG_PATTERNS]
        self._audit_patterns = [re.compile(p, re.IGNORECASE) for p in AUDIT_PATTERNS]
        self._code_patterns = [re.compile(p, re.IGNORECASE) for p in CODE_PATTERNS]

    def route(self, question: str) -> UnifiedQuery:
        """
        Route a question to the appropriate backend.

        Args:
            question: Natural language question

        Returns:
            UnifiedQuery with query_type and parsed query
        """
        question = question.strip()

        # 1. Check for CDG query pattern (highest priority - explicit SQL-like)
        if self._matches_cdg_pattern(question):
            return self._route_to_cdg(question)

        # 2. Check for audit query pattern
        if self._matches_audit_pattern(question):
            return self._route_to_audit(question)

        # 3. Check for code intent pattern
        if self._matches_code_pattern(question):
            return self._route_to_code(question)

        # 4. Fall back to semantic graph query
        return self._route_to_semantic(question)

    def _matches_cdg_pattern(self, question: str) -> bool:
        """Check if question matches CDG query patterns."""
        for pattern in self._cdg_patterns:
            if pattern.search(question):
                return True
        return False

    def _matches_audit_pattern(self, question: str) -> bool:
        """Check if question matches audit query patterns."""
        for pattern in self._audit_patterns:
            if pattern.search(question):
                return True
        return False

    def _matches_code_pattern(self, question: str) -> bool:
        """Check if question matches code intent patterns."""
        for pattern in self._code_patterns:
            if pattern.search(question):
                return True
        return False

    def _route_to_cdg(self, question: str) -> UnifiedQuery:
        """Route to CDG query backend."""
        # For now, return raw question - CDG parser will handle it
        # In Phase 2, we'll integrate cortical.cdg.query.Parser
        return UnifiedQuery(
            raw_question=question,
            query_type="cdg",
            parsed={"raw": question},  # Placeholder until CDG parser integrated
            confidence=0.9,
            metadata={"parser": "cdg"}
        )

    def _route_to_audit(self, question: str) -> UnifiedQuery:
        """Route to audit query backend."""
        audit_query = translate_audit_query(question)
        return UnifiedQuery(
            raw_question=question,
            query_type="audit",
            parsed=audit_query,
            confidence=0.8,
            metadata={"intent": audit_query.intent}
        )

    def _route_to_code(self, question: str) -> UnifiedQuery:
        """Route to code intent backend."""
        # Parse code intent
        intent = self._parse_code_intent(question)
        return UnifiedQuery(
            raw_question=question,
            query_type="code",
            parsed=intent,
            confidence=0.7,
            metadata={"action": intent.get("action")}
        )

    def _route_to_semantic(self, question: str) -> UnifiedQuery:
        """Route to semantic graph backend (fallback)."""
        query_intent = self._parse_semantic_intent(question)
        return UnifiedQuery(
            raw_question=question,
            query_type="semantic",
            parsed=query_intent,
            confidence=0.5,
            metadata={"question_type": query_intent.question_type}
        )

    def _parse_code_intent(self, question: str) -> Dict[str, Any]:
        """
        Parse code intent from question.

        Extracts action verbs and subjects for code search.
        """
        question_lower = question.lower()

        # Detect action
        action = None
        action_verbs = ["handle", "process", "implement", "call", "use", "create"]
        for verb in action_verbs:
            if verb in question_lower:
                action = verb
                break

        # Extract subject (words after the action verb)
        subject = None
        if action:
            match = re.search(rf'{action}\s+(\w+)', question_lower)
            if match:
                subject = match.group(1)

        # Detect intent type
        intent_type = "implementation"
        if question_lower.startswith("where"):
            intent_type = "location"
        elif question_lower.startswith("who"):
            intent_type = "attribution"
        elif question_lower.startswith("what"):
            intent_type = "definition"

        return {
            "action": action,
            "subject": subject,
            "intent": intent_type,
            "question_word": question.split()[0].lower() if question else None,
        }

    def _parse_semantic_intent(self, question: str) -> QueryIntent:
        """
        Parse semantic intent for graph-based queries.

        Extracts question type and key concepts.
        """
        question_lower = question.lower().strip()

        # Detect question type
        question_type = "general"
        for qword in ["how", "what", "where", "why", "who", "which", "when"]:
            if question_lower.startswith(qword):
                question_type = qword
                break

        # Extract concepts (non-stop words)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', question)
        concepts = [w.lower() for w in words if w.lower() not in STOP_WORDS and len(w) > 2]

        return QueryIntent(
            question_type=question_type,
            concepts=concepts,
            raw_question=question
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def route_question(question: str) -> UnifiedQuery:
    """
    Convenience function to route a question.

    Args:
        question: Natural language question

    Returns:
        UnifiedQuery with routing information
    """
    router = QueryRouter()
    return router.route(question)


def get_query_type(question: str) -> str:
    """
    Get the query type for a question.

    Args:
        question: Natural language question

    Returns:
        Query type string (cdg, audit, code, semantic)
    """
    return route_question(question).query_type
