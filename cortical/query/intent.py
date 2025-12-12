"""
Intent Query Module
==================

Intent-based query understanding for natural language code search.

This module handles:
- Parsing natural language queries to extract intent (where, how, what, etc.)
- Identifying action verbs and subjects in queries
- Intent-based search with weighted term scoring
"""

from typing import Dict, List, Tuple, Optional, TypedDict
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..code_concepts import get_related_terms


# Intent types for query understanding
class ParsedIntent(TypedDict):
    """Structured representation of a parsed query intent."""
    action: Optional[str]       # The verb/action (e.g., "handle", "implement")
    subject: Optional[str]      # The main subject (e.g., "authentication")
    intent: str                 # Query intent type (location, implementation, definition, etc.)
    question_word: Optional[str]  # Original question word if present
    expanded_terms: List[str]   # All searchable terms with synonyms


# Question word to intent mapping
QUESTION_INTENTS = {
    'where': 'location',      # Find location/file
    'how': 'implementation',  # Find implementation details
    'what': 'definition',     # Find definitions
    'why': 'rationale',       # Find comments/documentation explaining reasoning
    'when': 'lifecycle',      # Find when something happens (init, shutdown, etc.)
    'which': 'selection',     # Find choices/options
    'who': 'attribution',     # Find ownership/authorship (git blame territory)
}

# Common action verbs in code queries
ACTION_VERBS = frozenset([
    'handle', 'process', 'create', 'delete', 'update', 'fetch', 'get', 'set',
    'load', 'save', 'store', 'validate', 'check', 'parse', 'format', 'convert',
    'transform', 'render', 'display', 'show', 'hide', 'enable', 'disable',
    'start', 'stop', 'init', 'initialize', 'setup', 'configure', 'connect',
    'disconnect', 'send', 'receive', 'read', 'write', 'open', 'close',
    'authenticate', 'authorize', 'login', 'logout', 'register', 'subscribe',
    'publish', 'emit', 'listen', 'dispatch', 'trigger', 'call', 'invoke',
    'execute', 'run', 'build', 'compile', 'test', 'deploy', 'implement',
])


def parse_intent_query(query_text: str) -> ParsedIntent:
    """
    Parse a natural language query to extract intent and searchable terms.

    Analyzes queries like "where do we handle authentication?" to identify:
    - Question word (where) -> intent type (location)
    - Action verb (handle) -> search for handling code
    - Subject (authentication) -> main topic with synonyms

    Args:
        query_text: Natural language query string

    Returns:
        ParsedIntent with action, subject, intent type, and expanded terms

    Example:
        >>> parse_intent_query("where do we handle authentication?")
        {
            'action': 'handle',
            'subject': 'authentication',
            'intent': 'location',
            'question_word': 'where',
            'expanded_terms': ['handle', 'authentication', 'auth', 'login', ...]
        }
    """
    import re

    # Normalize query
    query_lower = query_text.lower().strip()
    query_lower = re.sub(r'[?!.,;:]', '', query_lower)  # Remove punctuation
    words = query_lower.split()

    if not words:
        return ParsedIntent(
            action=None,
            subject=None,
            intent='search',
            question_word=None,
            expanded_terms=[]
        )

    # Detect question word and intent
    question_word = None
    intent = 'search'  # Default intent

    for word in words:
        if word in QUESTION_INTENTS:
            question_word = word
            intent = QUESTION_INTENTS[word]
            break

    # Remove common filler words for parsing
    filler_words = {'do', 'we', 'i', 'you', 'the', 'a', 'an', 'is', 'are', 'was',
                    'were', 'can', 'could', 'should', 'would', 'does', 'did',
                    'have', 'has', 'had', 'be', 'been', 'being', 'will', 'to'}
    content_words = [w for w in words if w not in filler_words and w not in QUESTION_INTENTS]

    # Find action verb
    action = None
    for word in content_words:
        if word in ACTION_VERBS:
            action = word
            break

    # Find subject (first non-action content word, or last content word)
    subject = None
    for word in content_words:
        if word != action:
            subject = word
            break
    if not subject and content_words:
        subject = content_words[-1]

    # Build expanded terms list
    expanded_terms = []

    # Add action and its synonyms
    if action:
        expanded_terms.append(action)
        action_synonyms = get_related_terms(action, max_terms=5)
        expanded_terms.extend(action_synonyms)

    # Add subject and its synonyms
    if subject:
        expanded_terms.append(subject)
        subject_synonyms = get_related_terms(subject, max_terms=5)
        expanded_terms.extend(subject_synonyms)

    # Add remaining content words
    for word in content_words:
        if word not in expanded_terms:
            expanded_terms.append(word)

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return ParsedIntent(
        action=action,
        subject=subject,
        intent=intent,
        question_word=question_word,
        expanded_terms=unique_terms
    )


def search_by_intent(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: 'Tokenizer',
    top_n: int = 5
) -> List[Tuple[str, float, ParsedIntent]]:
    """
    Search the corpus using intent-based query understanding.

    Parses the query to understand intent, expands terms using code concepts,
    then searches with appropriate weighting based on intent type.

    Args:
        query_text: Natural language query string
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return

    Returns:
        List of (doc_id, score, parsed_intent) tuples

    Example:
        >>> search_by_intent("how do we validate user input?", layers, tokenizer)
        [('validation.py', 0.85, {...}), ('forms.py', 0.72, {...}), ...]
    """
    # Parse the query intent
    parsed = parse_intent_query(query_text)

    if not parsed['expanded_terms']:
        return []

    # Build weighted query from expanded terms
    layer0 = layers[CorticalLayer.TOKENS]

    # Score documents based on term matches
    doc_scores: Dict[str, float] = defaultdict(float)

    for i, term in enumerate(parsed['expanded_terms']):
        # Earlier terms (action, subject) get higher weight
        term_weight = 1.0 / (1 + i * 0.2)

        col = layer0.get_minicolumn(term)
        if col:
            for doc_id in col.document_ids:
                # Use TF-IDF if available
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_scores[doc_id] += term_weight * tfidf

    # Sort by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top results with parsed intent
    results = []
    for doc_id, score in sorted_docs[:top_n]:
        results.append((doc_id, score, parsed))

    return results
