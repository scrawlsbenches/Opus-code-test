"""
Query Module
============

Query expansion and search functionality.

Provides methods for expanding queries using lateral connections,
concept clusters, and word variants, then searching the corpus
using TF-IDF and graph-based scoring.
"""

from typing import Dict, List, Tuple, Optional, TypedDict
from collections import defaultdict
import re

from .layers import CorticalLayer, HierarchicalLayer
from .tokenizer import Tokenizer
from .code_concepts import expand_code_concepts, get_related_terms


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


def expand_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    max_expansions: int = 10,
    use_lateral: bool = True,
    use_concepts: bool = True,
    use_variants: bool = True,
    use_code_concepts: bool = False
) -> Dict[str, float]:
    """
    Expand a query using lateral connections and concept clusters.

    This mimics how the brain retrieves related memories when given a cue:
    - Lateral connections: direct word associations (like priming)
    - Concept clusters: semantic category membership
    - Word variants: stemming and synonym mapping
    - Code concepts: programming synonym groups (get/fetch/load)

    Args:
        query_text: Original query string
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        max_expansions: Maximum number of expansion terms to add
        use_lateral: Include terms from lateral connections
        use_concepts: Include terms from concept clusters
        use_variants: Try word variants when direct match fails
        use_code_concepts: Include programming synonym expansions

    Returns:
        Dict mapping terms to weights (original terms get weight 1.0)
    """
    tokens = tokenizer.tokenize(query_text)
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers.get(CorticalLayer.CONCEPTS)
    
    # Start with original terms at full weight
    expanded: Dict[str, float] = {}
    unmatched_tokens = []
    
    for token in tokens:
        col = layer0.get_minicolumn(token)
        if col:
            expanded[token] = 1.0
        else:
            unmatched_tokens.append(token)
    
    # Try to match unmatched tokens using variants
    if use_variants and unmatched_tokens:
        for token in unmatched_tokens:
            variants = tokenizer.get_word_variants(token)
            for variant in variants:
                col = layer0.get_minicolumn(variant)
                if col and variant not in expanded:
                    expanded[variant] = 0.8
                    break
    
    if not expanded:
        return expanded
    
    candidate_expansions: Dict[str, float] = defaultdict(float)
    
    # Method 1: Lateral connections (direct associations)
    if use_lateral:
        for token in list(expanded.keys()):
            col = layer0.get_minicolumn(token)
            if col:
                sorted_neighbors = sorted(
                    col.lateral_connections.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for neighbor_id, weight in sorted_neighbors:
                    # Use O(1) ID lookup instead of linear search
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor and neighbor.content not in expanded:
                        score = weight * neighbor.pagerank * 0.6
                        candidate_expansions[neighbor.content] = max(
                            candidate_expansions[neighbor.content], score
                        )
    
    # Method 2: Concept cluster membership
    if use_concepts and layer2 and layer2.column_count() > 0:
        for token in list(expanded.keys()):
            col = layer0.get_minicolumn(token)
            if col:
                for concept in layer2.minicolumns.values():
                    if col.id in concept.feedforward_sources:
                        for member_id in concept.feedforward_sources:
                            # Use O(1) ID lookup instead of linear search
                            member = layer0.get_by_id(member_id)
                            if member and member.content not in expanded:
                                score = concept.pagerank * member.pagerank * 0.4
                                candidate_expansions[member.content] = max(
                                    candidate_expansions[member.content], score
                                )

    # Method 3: Code concept groups (programming synonyms)
    if use_code_concepts:
        code_expansions = expand_code_concepts(
            list(expanded.keys()),
            max_expansions_per_term=3,
            weight=0.6
        )
        for term, weight in code_expansions.items():
            if term not in expanded:
                candidate_expansions[term] = max(
                    candidate_expansions[term], weight
                )

    # Select top expansions
    sorted_candidates = sorted(
        candidate_expansions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_expansions]
    
    for term, score in sorted_candidates:
        expanded[term] = score
    
    return expanded


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
    tokenizer: Tokenizer,
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
    layer3 = layers[CorticalLayer.DOCUMENTS]

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


# Valid relation chain patterns for multi-hop inference
# Key: (relation1, relation2) → validity score (0.0 = invalid, 1.0 = fully valid)
VALID_RELATION_CHAINS = {
    # Transitive hierarchies
    ('IsA', 'IsA'): 1.0,           # dog IsA animal IsA living_thing
    ('PartOf', 'PartOf'): 1.0,     # wheel PartOf car PartOf vehicle
    ('IsA', 'HasProperty'): 0.9,   # dog IsA animal HasProperty alive
    ('PartOf', 'HasProperty'): 0.8,  # wheel PartOf car HasProperty fast

    # Association chains
    ('RelatedTo', 'RelatedTo'): 0.6,
    ('SimilarTo', 'SimilarTo'): 0.7,
    ('CoOccurs', 'CoOccurs'): 0.5,
    ('RelatedTo', 'IsA'): 0.7,
    ('RelatedTo', 'SimilarTo'): 0.7,

    # Causal chains
    ('Causes', 'Causes'): 0.8,
    ('Causes', 'HasProperty'): 0.7,

    # Derivation chains
    ('DerivedFrom', 'DerivedFrom'): 0.8,
    ('DerivedFrom', 'IsA'): 0.7,

    # Usage chains
    ('UsedFor', 'UsedFor'): 0.6,
    ('UsedFor', 'RelatedTo'): 0.5,

    # Antonym - generally invalid for chaining
    ('Antonym', 'Antonym'): 0.3,   # Double negation, weak
    ('Antonym', 'IsA'): 0.1,       # Contradictory
}


def score_relation_path(path: List[str]) -> float:
    """
    Score a relation path by its semantic coherence.

    Args:
        path: List of relation types traversed (e.g., ['IsA', 'HasProperty'])

    Returns:
        Score from 0.0 (invalid) to 1.0 (fully valid)
    """
    if not path:
        return 1.0
    if len(path) == 1:
        return 1.0

    # Compute score as product of consecutive pair validities
    total_score = 1.0
    for i in range(len(path) - 1):
        pair = (path[i], path[i + 1])
        # Check both orderings
        pair_score = VALID_RELATION_CHAINS.get(pair, 0.4)  # Default: moderate validity
        total_score *= pair_score

    return total_score


def expand_query_multihop(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    semantic_relations: List[Tuple[str, str, str, float]],
    max_hops: int = 2,
    max_expansions: int = 15,
    decay_factor: float = 0.5,
    min_path_score: float = 0.2
) -> Dict[str, float]:
    """
    Expand query using multi-hop semantic inference.

    Unlike single-hop expansion that only follows direct connections,
    this follows relation chains to discover semantically related terms
    through transitive relationships.

    Example inference chains:
        "dog" → IsA → "animal" → HasProperty → "living"
        "car" → PartOf → "engine" → UsedFor → "transportation"

    Args:
        query_text: Original query string
        layers: Dictionary of layers (needs TOKENS)
        tokenizer: Tokenizer instance
        semantic_relations: List of (term1, relation, term2, weight) tuples
        max_hops: Maximum number of relation hops (default: 2)
        max_expansions: Maximum expansion terms to return
        decay_factor: Weight decay per hop (default: 0.5, so hop2 = 0.25)
        min_path_score: Minimum path validity score to include (default: 0.2)

    Returns:
        Dict mapping terms to weights (original terms get weight 1.0,
        expansions get decayed weights based on hop distance and path validity)

    Example:
        >>> expanded = expand_query_multihop("neural", layers, tokenizer, relations)
        >>> # Hop 1: networks (co-occur), learning (co-occur), brain (RelatedTo)
        >>> # Hop 2: deep (via learning), cortex (via brain), AI (via networks)
    """
    tokens = tokenizer.tokenize(query_text)
    layer0 = layers[CorticalLayer.TOKENS]

    # Start with original terms at full weight
    expanded: Dict[str, float] = {}
    for token in tokens:
        if layer0.get_minicolumn(token):
            expanded[token] = 1.0

    if not expanded or not semantic_relations:
        return expanded

    # Build bidirectional neighbor lookup with relation types
    # neighbors[term] = [(neighbor, relation_type, weight), ...]
    neighbors: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for t1, relation, t2, weight in semantic_relations:
        neighbors[t1].append((t2, relation, weight))
        neighbors[t2].append((t1, relation, weight))

    # Track expansions with their hop distance, weight, and relation path
    # (term, weight, hop, relation_path)
    candidates: Dict[str, Tuple[float, int, List[str]]] = {}

    # BFS-style expansion with hop tracking
    # frontier: [(term, current_weight, hop_count, relation_path)]
    frontier: List[Tuple[str, float, int, List[str]]] = [
        (term, 1.0, 0, []) for term in expanded.keys()
    ]

    visited_at_hop: Dict[str, int] = {term: 0 for term in expanded.keys()}

    while frontier:
        current_term, current_weight, hop, path = frontier.pop(0)

        if hop >= max_hops:
            continue

        next_hop = hop + 1

        for neighbor, relation, rel_weight in neighbors.get(current_term, []):
            # Skip if already in original query terms
            if neighbor in expanded:
                continue

            # Check if term exists in corpus
            if not layer0.get_minicolumn(neighbor):
                continue

            # Skip if we've visited this term at an earlier or equal hop
            if neighbor in visited_at_hop and visited_at_hop[neighbor] <= next_hop:
                continue

            # Compute new path and its validity
            new_path = path + [relation]
            path_score = score_relation_path(new_path)

            if path_score < min_path_score:
                continue

            # Compute weight with decay and path validity
            # weight = base_weight * relation_weight * decay^hop * path_validity
            hop_decay = decay_factor ** next_hop
            new_weight = current_weight * rel_weight * hop_decay * path_score

            # Update candidate if this path gives higher weight
            if neighbor not in candidates or candidates[neighbor][0] < new_weight:
                candidates[neighbor] = (new_weight, next_hop, new_path)
                visited_at_hop[neighbor] = next_hop

                # Add to frontier for further expansion
                if next_hop < max_hops:
                    frontier.append((neighbor, new_weight, next_hop, new_path))

    # Sort candidates by weight and take top expansions
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: x[1][0],  # Sort by weight
        reverse=True
    )[:max_expansions]

    # Add to expanded dict
    for term, (weight, hop, path) in sorted_candidates:
        expanded[term] = weight

    return expanded


def expand_query_semantic(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    semantic_relations: List[Tuple[str, str, str, float]],
    max_expansions: int = 10
) -> Dict[str, float]:
    """
    Expand query using semantic relations extracted from corpus.
    
    Args:
        query_text: Original query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        semantic_relations: List of (term1, relation, term2, weight) tuples
        max_expansions: Maximum expansions
        
    Returns:
        Dict mapping terms to weights
    """
    tokens = tokenizer.tokenize(query_text)
    layer0 = layers[CorticalLayer.TOKENS]
    
    # Build semantic neighbor lookup
    neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for t1, relation, t2, weight in semantic_relations:
        neighbors[t1].append((t2, weight))
        neighbors[t2].append((t1, weight))
    
    # Start with original terms
    expanded = {t: 1.0 for t in tokens if layer0.get_minicolumn(t)}
    
    if not expanded:
        return expanded
    
    # Add semantic neighbors
    candidates: Dict[str, float] = defaultdict(float)
    for token in list(expanded.keys()):
        for neighbor, weight in neighbors.get(token, []):
            if neighbor not in expanded and layer0.get_minicolumn(neighbor):
                candidates[neighbor] = max(candidates[neighbor], weight * 0.7)
    
    # Take top candidates
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    for term, score in sorted_candidates[:max_expansions]:
        expanded[term] = score
    
    return expanded


def get_expanded_query_terms(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True,
    max_expansions: int = 5,
    semantic_discount: float = 0.8
) -> Dict[str, float]:
    """
    Get expanded query terms with optional semantic expansion.

    This is a helper function that consolidates query expansion logic used
    by multiple search functions. It handles:
    - Lateral connection expansion via expand_query()
    - Semantic relation expansion via expand_query_semantic()
    - Merging of expansion results with appropriate weighting

    Args:
        query_text: Original query string
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        use_expansion: Whether to expand query terms using lateral connections
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion
        max_expansions: Maximum expansion terms per method (default 5)
        semantic_discount: Weight multiplier for semantic expansions (default 0.8)

    Returns:
        Dict mapping terms to weights (original terms get weight 1.0,
        expansions get lower weights based on connection strength)

    Example:
        >>> terms = get_expanded_query_terms("neural networks", layers, tokenizer)
        >>> # Returns: {'neural': 1.0, 'networks': 1.0, 'deep': 0.3, 'learning': 0.25, ...}
    """
    if use_expansion:
        # Start with lateral connection expansion
        query_terms = expand_query(query_text, layers, tokenizer, max_expansions=max_expansions)

        # Add semantic expansion if available
        if use_semantic and semantic_relations:
            semantic_terms = expand_query_semantic(
                query_text, layers, tokenizer, semantic_relations, max_expansions=max_expansions
            )
            # Merge semantic expansions (don't override stronger weights)
            for term, weight in semantic_terms.items():
                if term not in query_terms:
                    query_terms[term] = weight * semantic_discount
                else:
                    # Take the max weight
                    query_terms[term] = max(query_terms[term], weight * semantic_discount)
    else:
        tokens = tokenizer.tokenize(query_text)
        query_terms = {t: 1.0 for t in tokens}

    return query_terms


def find_documents_for_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[Tuple[str, float]]:
    """
    Find documents most relevant to a query using TF-IDF and optional expansion.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of documents to return
        use_expansion: Whether to expand query terms using lateral connections
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion (if available)

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    # Score each document
    doc_scores: Dict[str, float] = defaultdict(float)

    for term, term_weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            for doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_scores[doc_id] += tfidf * term_weight

    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
    return sorted_docs[:top_n]


def fast_find_documents(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    candidate_multiplier: int = 3,
    use_code_concepts: bool = True
) -> List[Tuple[str, float]]:
    """
    Fast document search using candidate filtering.

    Optimizes search by:
    1. Using set intersection to find candidate documents
    2. Only scoring top candidates fully
    3. Using code concept expansion for better recall

    This is ~2-3x faster than full search on large corpora while
    maintaining similar result quality.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return
        candidate_multiplier: Multiplier for candidate set size
        use_code_concepts: Whether to use code concept expansion

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Tokenize query
    tokens = tokenizer.tokenize(query_text)
    if not tokens:
        return []

    # Phase 1: Find candidate documents (fast set operations)
    # Get documents containing ANY query term
    candidate_docs: Dict[str, int] = defaultdict(int)  # doc_id -> match count

    for token in tokens:
        col = layer0.get_minicolumn(token)
        if col:
            for doc_id in col.document_ids:
                candidate_docs[doc_id] += 1

    # If no candidates, try code concept expansion for recall
    if not candidate_docs and use_code_concepts:
        for token in tokens:
            related = get_related_terms(token, max_terms=3)
            for related_term in related:
                col = layer0.get_minicolumn(related_term)
                if col:
                    for doc_id in col.document_ids:
                        candidate_docs[doc_id] += 0.5  # Lower weight for expansion

    if not candidate_docs:
        return []

    # Rank candidates by match count first (fast pre-filter)
    sorted_candidates = sorted(
        candidate_docs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Take top N * multiplier candidates for full scoring
    max_candidates = top_n * candidate_multiplier
    top_candidates = sorted_candidates[:max_candidates]

    # Phase 2: Full scoring only on top candidates
    doc_scores: Dict[str, float] = {}

    for doc_id, match_count in top_candidates:
        score = 0.0
        for token in tokens:
            col = layer0.get_minicolumn(token)
            if col and doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                score += tfidf

        # Boost by match coverage
        coverage_boost = match_count / len(tokens)
        doc_scores[doc_id] = score * (1 + 0.5 * coverage_boost)

    # Return top results
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


def build_document_index(
    layers: Dict[CorticalLayer, HierarchicalLayer]
) -> Dict[str, Dict[str, float]]:
    """
    Build an optimized inverted index for fast querying.

    Creates a term -> {doc_id: score} mapping that can be used
    for fast set operations during search.

    Args:
        layers: Dictionary of layers

    Returns:
        Dict mapping terms to {doc_id: tfidf_score} dicts
    """
    layer0 = layers.get(CorticalLayer.TOKENS)
    if not layer0:
        return {}

    index: Dict[str, Dict[str, float]] = {}

    for col in layer0.minicolumns.values():
        term = col.content
        term_index: Dict[str, float] = {}

        for doc_id in col.document_ids:
            tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
            term_index[doc_id] = tfidf

        if term_index:
            index[term] = term_index

    return index


def search_with_index(
    query_text: str,
    index: Dict[str, Dict[str, float]],
    tokenizer: Tokenizer,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Search using a pre-built inverted index.

    This is the fastest search method when the index is cached.

    Args:
        query_text: Search query
        index: Pre-built index from build_document_index()
        tokenizer: Tokenizer instance
        top_n: Number of results to return

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    tokens = tokenizer.tokenize(query_text)
    if not tokens:
        return []

    doc_scores: Dict[str, float] = defaultdict(float)

    for token in tokens:
        if token in index:
            for doc_id, score in index[token].items():
                doc_scores[doc_id] += score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


def query_with_spreading_activation(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 10,
    max_expansions: int = 8
) -> List[Tuple[str, float]]:
    """
    Query with automatic expansion using spreading activation.
    
    This is like the brain's spreading activation during memory retrieval:
    a cue activates not just direct matches but semantically related concepts.
    
    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return
        max_expansions: How many expansion terms to add
        
    Returns:
        List of (concept, score) tuples ranked by relevance
    """
    expanded_terms = expand_query(
        query_text, layers, tokenizer,
        max_expansions=max_expansions
    )
    
    if not expanded_terms:
        return []
    
    layer0 = layers[CorticalLayer.TOKENS]
    activated: Dict[str, float] = {}
    
    # Activate based on expanded query
    for term, term_weight in expanded_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            # Direct activation
            score = col.pagerank * col.activation * term_weight
            activated[col.content] = activated.get(col.content, 0) + score
            
            # Spread to neighbors using O(1) ID lookup
            for neighbor_id, conn_weight in col.lateral_connections.items():
                neighbor = layer0.get_by_id(neighbor_id)
                if neighbor:
                    spread_score = neighbor.pagerank * conn_weight * term_weight * 0.3
                    activated[neighbor.content] = activated.get(neighbor.content, 0) + spread_score
    
    sorted_concepts = sorted(activated.items(), key=lambda x: -x[1])
    return sorted_concepts[:top_n]


def find_related_documents(
    doc_id: str,
    layers: Dict[CorticalLayer, HierarchicalLayer]
) -> List[Tuple[str, float]]:
    """
    Find documents related to a given document via lateral connections.

    Args:
        doc_id: Source document ID
        layers: Dictionary of layers

    Returns:
        List of (doc_id, weight) tuples for related documents
    """
    layer3 = layers.get(CorticalLayer.DOCUMENTS)
    if not layer3:
        return []

    col = layer3.get_minicolumn(doc_id)
    if not col:
        return []

    related = []
    for neighbor_id, weight in col.lateral_connections.items():
        # Use O(1) ID lookup instead of linear search
        neighbor = layer3.get_by_id(neighbor_id)
        if neighbor:
            related.append((neighbor.content, weight))

    return sorted(related, key=lambda x: -x[1])


def create_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: Document text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if not text:
        return []

    chunks = []
    stride = max(1, chunk_size - overlap)
    text_len = len(text)

    for start in range(0, text_len, stride):
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))

        if end >= text_len:
            break

    return chunks


def score_chunk(
    chunk_text: str,
    query_terms: Dict[str, float],
    layer0: HierarchicalLayer,
    tokenizer: Tokenizer,
    doc_id: Optional[str] = None
) -> float:
    """
    Score a chunk against query terms using TF-IDF.

    Args:
        chunk_text: Text of the chunk
        query_terms: Dict mapping query terms to weights
        layer0: Token layer for TF-IDF lookups
        tokenizer: Tokenizer instance
        doc_id: Optional document ID for per-document TF-IDF

    Returns:
        Relevance score for the chunk
    """
    chunk_tokens = tokenizer.tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0

    # Count token occurrences in chunk
    token_counts: Dict[str, int] = {}
    for token in chunk_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    score = 0.0
    for term, term_weight in query_terms.items():
        if term in token_counts:
            col = layer0.get_minicolumn(term)
            if col:
                # Use per-document TF-IDF if available, otherwise global
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf) if doc_id else col.tfidf
                # Weight by occurrence in chunk and query weight
                score += tfidf * token_counts[term] * term_weight

    # Normalize by chunk length to avoid bias toward longer chunks
    return score / len(chunk_tokens) if chunk_tokens else 0.0


def find_passages_for_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    documents: Dict[str, str],
    top_n: int = 5,
    chunk_size: int = 512,
    overlap: int = 128,
    use_expansion: bool = True,
    doc_filter: Optional[List[str]] = None,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[Tuple[str, str, int, int, float]]:
    """
    Find text passages most relevant to a query.

    This is the key function for RAG systems - instead of returning document IDs,
    it returns actual text passages with position information for citations.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        documents: Dict mapping doc_id to document text
        top_n: Number of passages to return
        chunk_size: Size of each chunk in characters (default 512)
        overlap: Overlap between chunks in characters (default 128)
        use_expansion: Whether to expand query terms
        doc_filter: Optional list of doc_ids to restrict search to
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion (if available)

    Returns:
        List of (passage_text, doc_id, start_char, end_char, score) tuples
        ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Get expanded query terms
    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    if not query_terms:
        return []

    # First, get candidate documents (more than we need, since we'll rank passages)
    doc_scores = find_documents_for_query(
        query_text, layers, tokenizer,
        top_n=min(len(documents), top_n * 3),
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    # Apply document filter if provided
    if doc_filter:
        doc_scores = [(doc_id, score) for doc_id, score in doc_scores if doc_id in doc_filter]

    # Score passages within candidate documents
    passages: List[Tuple[str, str, int, int, float]] = []

    for doc_id, doc_score in doc_scores:
        if doc_id not in documents:
            continue

        text = documents[doc_id]
        chunks = create_chunks(text, chunk_size, overlap)

        for chunk_text, start_char, end_char in chunks:
            chunk_score = score_chunk(
                chunk_text, query_terms, layer0, tokenizer, doc_id
            )
            # Combine chunk score with document score for final ranking
            combined_score = chunk_score * (1 + doc_score * 0.1)

            passages.append((
                chunk_text,
                doc_id,
                start_char,
                end_char,
                combined_score
            ))

    # Sort by score and return top passages
    passages.sort(key=lambda x: x[4], reverse=True)
    return passages[:top_n]


def find_documents_batch(
    queries: List[str],
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[List[Tuple[str, float]]]:
    """
    Find documents for multiple queries efficiently.

    More efficient than calling find_documents_for_query() multiple times
    because it shares tokenization and expansion caching across queries.

    Args:
        queries: List of search query strings
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of documents to return per query
        use_expansion: Whether to expand query terms
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion

    Returns:
        List of results, one per query. Each result is a list of (doc_id, score) tuples.

    Example:
        >>> queries = ["neural networks", "machine learning", "data processing"]
        >>> results = find_documents_batch(queries, layers, tokenizer, top_n=3)
        >>> for query, docs in zip(queries, results):
        ...     print(f"{query}: {[doc_id for doc_id, _ in docs]}")
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Cache for expanded query terms to avoid redundant computation
    expansion_cache: Dict[str, Dict[str, float]] = {}

    all_results: List[List[Tuple[str, float]]] = []

    for query_text in queries:
        # Check cache first for expansion
        if query_text in expansion_cache:
            query_terms = expansion_cache[query_text]
        else:
            query_terms = get_expanded_query_terms(
                query_text, layers, tokenizer,
                use_expansion=use_expansion,
                semantic_relations=semantic_relations,
                use_semantic=use_semantic
            )
            expansion_cache[query_text] = query_terms

        # Score documents
        doc_scores: Dict[str, float] = defaultdict(float)
        for term, term_weight in query_terms.items():
            col = layer0.get_minicolumn(term)
            if col:
                for doc_id in col.document_ids:
                    tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                    doc_scores[doc_id] += tfidf * term_weight

        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        all_results.append(sorted_docs[:top_n])

    return all_results


def find_passages_batch(
    queries: List[str],
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    documents: Dict[str, str],
    top_n: int = 5,
    chunk_size: int = 512,
    overlap: int = 128,
    use_expansion: bool = True,
    doc_filter: Optional[List[str]] = None,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[List[Tuple[str, str, int, int, float]]]:
    """
    Find passages for multiple queries efficiently.

    More efficient than calling find_passages_for_query() multiple times
    because it shares chunk computation and expansion caching across queries.

    Args:
        queries: List of search query strings
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        documents: Dict mapping doc_id to document text
        top_n: Number of passages to return per query
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        use_expansion: Whether to expand query terms
        doc_filter: Optional list of doc_ids to restrict search to
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion

    Returns:
        List of results, one per query. Each result is a list of
        (passage_text, doc_id, start_char, end_char, score) tuples.

    Example:
        >>> queries = ["neural networks", "deep learning"]
        >>> results = find_passages_batch(queries, layers, tokenizer, documents)
        >>> for query, passages in zip(queries, results):
        ...     print(f"{query}: {len(passages)} passages found")
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Pre-compute chunks for all documents to avoid redundant chunking
    doc_chunks_cache: Dict[str, List[Tuple[str, int, int]]] = {}
    for doc_id, text in documents.items():
        if doc_filter and doc_id not in doc_filter:
            continue
        doc_chunks_cache[doc_id] = create_chunks(text, chunk_size, overlap)

    # Cache for expanded query terms
    expansion_cache: Dict[str, Dict[str, float]] = {}

    all_results: List[List[Tuple[str, str, int, int, float]]] = []

    for query_text in queries:
        # Get expanded query terms (with caching)
        if query_text in expansion_cache:
            query_terms = expansion_cache[query_text]
        else:
            query_terms = get_expanded_query_terms(
                query_text, layers, tokenizer,
                use_expansion=use_expansion,
                semantic_relations=semantic_relations,
                use_semantic=use_semantic
            )
            expansion_cache[query_text] = query_terms

        if not query_terms:
            all_results.append([])
            continue

        # Get candidate documents
        doc_scores = find_documents_for_query(
            query_text, layers, tokenizer,
            top_n=min(len(documents), top_n * 3),
            use_expansion=use_expansion,
            semantic_relations=semantic_relations,
            use_semantic=use_semantic
        )

        # Apply document filter
        if doc_filter:
            doc_scores = [(doc_id, score) for doc_id, score in doc_scores if doc_id in doc_filter]

        # Score passages using cached chunks
        passages: List[Tuple[str, str, int, int, float]] = []

        for doc_id, doc_score in doc_scores:
            if doc_id not in doc_chunks_cache:
                continue

            for chunk_text, start_char, end_char in doc_chunks_cache[doc_id]:
                chunk_score = score_chunk(
                    chunk_text, query_terms, layer0, tokenizer, doc_id
                )
                combined_score = chunk_score * (1 + doc_score * 0.1)
                passages.append((chunk_text, doc_id, start_char, end_char, combined_score))

        passages.sort(key=lambda x: x[4], reverse=True)
        all_results.append(passages[:top_n])

    return all_results


def find_relevant_concepts(
    query_terms: Dict[str, float],
    layers: Dict[CorticalLayer, HierarchicalLayer],
    top_n: int = 5
) -> List[Tuple[str, float, set]]:
    """
    Stage 1: Find concepts relevant to query terms.

    Args:
        query_terms: Dict mapping query terms to weights
        layers: Dictionary of layers
        top_n: Maximum number of concepts to return

    Returns:
        List of (concept_name, relevance_score, document_ids) tuples
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers.get(CorticalLayer.CONCEPTS)

    if not layer2 or layer2.column_count() == 0:
        return []

    concept_scores: Dict[str, float] = {}
    concept_docs: Dict[str, set] = {}

    for term, weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if not col:
            continue

        # Find concepts that contain this token
        for concept in layer2.minicolumns.values():
            if col.id in concept.feedforward_sources:
                # Score based on term weight, concept PageRank, and concept size
                score = weight * concept.pagerank * (1 + len(concept.feedforward_sources) * 0.01)
                concept_scores[concept.content] = concept_scores.get(concept.content, 0) + score
                if concept.content not in concept_docs:
                    concept_docs[concept.content] = set()
                concept_docs[concept.content].update(concept.document_ids)

    # Sort by score and return top concepts
    sorted_concepts = sorted(concept_scores.items(), key=lambda x: -x[1])[:top_n]
    return [(name, score, concept_docs.get(name, set())) for name, score in sorted_concepts]


def multi_stage_rank(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    documents: Dict[str, str],
    top_n: int = 5,
    chunk_size: int = 512,
    overlap: int = 128,
    concept_boost: float = 0.3,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[Tuple[str, str, int, int, float, Dict[str, float]]]:
    """
    Multi-stage ranking pipeline for improved RAG performance.

    Unlike flat ranking (TF-IDF → score), this uses a 4-stage pipeline:
    1. Concepts: Filter by topic relevance using Layer 2 clusters
    2. Documents: Rank documents within relevant topics
    3. Chunks: Rank passages within top documents
    4. Rerank: Combine all signals for final scoring

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        documents: Dict mapping doc_id to document text
        top_n: Number of passages to return
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        concept_boost: Weight for concept relevance in final score (0.0-1.0)
        use_expansion: Whether to expand query terms
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion

    Returns:
        List of (passage_text, doc_id, start_char, end_char, final_score, stage_scores) tuples.
        stage_scores dict contains: concept_score, doc_score, chunk_score, final_score

    Example:
        >>> results = multi_stage_rank(query, layers, tokenizer, documents)
        >>> for passage, doc_id, start, end, score, stages in results:
        ...     print(f"[{doc_id}] Score: {score:.3f}")
        ...     print(f"  Concept: {stages['concept_score']:.3f}")
        ...     print(f"  Doc: {stages['doc_score']:.3f}")
        ...     print(f"  Chunk: {stages['chunk_score']:.3f}")
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Get expanded query terms
    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    if not query_terms:
        return []

    # ========== STAGE 1: CONCEPTS ==========
    # Find relevant concepts to identify topic areas
    relevant_concepts = find_relevant_concepts(query_terms, layers, top_n=10)

    # Build concept score per document
    doc_concept_scores: Dict[str, float] = defaultdict(float)
    if relevant_concepts:
        max_concept_score = max(score for _, score, _ in relevant_concepts) if relevant_concepts else 1.0
        for concept_name, concept_score, doc_ids in relevant_concepts:
            normalized_score = concept_score / max_concept_score if max_concept_score > 0 else 0
            for doc_id in doc_ids:
                doc_concept_scores[doc_id] = max(doc_concept_scores[doc_id], normalized_score)

    # ========== STAGE 2: DOCUMENTS ==========
    # Score documents using TF-IDF (standard approach)
    doc_tfidf_scores: Dict[str, float] = defaultdict(float)
    for term, term_weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            for doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_tfidf_scores[doc_id] += tfidf * term_weight

    # Normalize TF-IDF scores
    max_tfidf = max(doc_tfidf_scores.values()) if doc_tfidf_scores else 1.0
    for doc_id in doc_tfidf_scores:
        doc_tfidf_scores[doc_id] /= max_tfidf if max_tfidf > 0 else 1.0

    # Combine concept and TF-IDF scores for document ranking
    combined_doc_scores: Dict[str, float] = {}
    all_docs = set(doc_concept_scores.keys()) | set(doc_tfidf_scores.keys())
    for doc_id in all_docs:
        concept_score = doc_concept_scores.get(doc_id, 0.0)
        tfidf_score = doc_tfidf_scores.get(doc_id, 0.0)
        # Weighted combination
        combined_doc_scores[doc_id] = (
            (1 - concept_boost) * tfidf_score +
            concept_boost * concept_score
        )

    # Get top documents for chunk scoring
    sorted_docs = sorted(combined_doc_scores.items(), key=lambda x: -x[1])
    top_docs = sorted_docs[:min(len(sorted_docs), top_n * 3)]

    # ========== STAGE 3: CHUNKS ==========
    # Score passages within top documents
    passages: List[Tuple[str, str, int, int, float, Dict[str, float]]] = []

    for doc_id, doc_score in top_docs:
        if doc_id not in documents:
            continue

        text = documents[doc_id]
        chunks = create_chunks(text, chunk_size, overlap)

        for chunk_text, start_char, end_char in chunks:
            chunk_score = score_chunk(chunk_text, query_terms, layer0, tokenizer, doc_id)

            # ========== STAGE 4: RERANK ==========
            # Combine all signals for final score
            concept_score = doc_concept_scores.get(doc_id, 0.0)
            tfidf_score = doc_tfidf_scores.get(doc_id, 0.0)

            # Normalize chunk score (avoid division by zero)
            normalized_chunk = chunk_score

            # Final score combines:
            # - Chunk-level relevance (primary signal)
            # - Document-level TF-IDF (context signal)
            # - Concept relevance (topic signal)
            final_score = (
                0.5 * normalized_chunk +
                0.3 * tfidf_score +
                0.2 * concept_score
            ) * (1 + doc_score * 0.1)  # Slight boost from combined doc score

            stage_scores = {
                'concept_score': concept_score,
                'doc_score': tfidf_score,
                'chunk_score': chunk_score,
                'combined_doc_score': doc_score,
                'final_score': final_score
            }

            passages.append((
                chunk_text,
                doc_id,
                start_char,
                end_char,
                final_score,
                stage_scores
            ))

    # Sort by final score and return top passages
    passages.sort(key=lambda x: x[4], reverse=True)
    return passages[:top_n]


def multi_stage_rank_documents(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    concept_boost: float = 0.3,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Multi-stage ranking for documents (without chunk scoring).

    Uses the first 2 stages of the pipeline:
    1. Concepts: Filter by topic relevance
    2. Documents: Rank by combined concept + TF-IDF scores

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of documents to return
        concept_boost: Weight for concept relevance (0.0-1.0)
        use_expansion: Whether to expand query terms
        semantic_relations: Optional list of semantic relations
        use_semantic: Whether to use semantic relations

    Returns:
        List of (doc_id, final_score, stage_scores) tuples.
        stage_scores dict contains: concept_score, tfidf_score, combined_score
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Get expanded query terms
    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    if not query_terms:
        return []

    # Stage 1: Concepts
    relevant_concepts = find_relevant_concepts(query_terms, layers, top_n=10)

    doc_concept_scores: Dict[str, float] = defaultdict(float)
    if relevant_concepts:
        max_concept_score = max(score for _, score, _ in relevant_concepts) if relevant_concepts else 1.0
        for concept_name, concept_score, doc_ids in relevant_concepts:
            normalized_score = concept_score / max_concept_score if max_concept_score > 0 else 0
            for doc_id in doc_ids:
                doc_concept_scores[doc_id] = max(doc_concept_scores[doc_id], normalized_score)

    # Stage 2: Documents
    doc_tfidf_scores: Dict[str, float] = defaultdict(float)
    for term, term_weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            for doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_tfidf_scores[doc_id] += tfidf * term_weight

    # Normalize TF-IDF
    max_tfidf = max(doc_tfidf_scores.values()) if doc_tfidf_scores else 1.0
    for doc_id in doc_tfidf_scores:
        doc_tfidf_scores[doc_id] /= max_tfidf if max_tfidf > 0 else 1.0

    # Combine scores
    results: List[Tuple[str, float, Dict[str, float]]] = []
    all_docs = set(doc_concept_scores.keys()) | set(doc_tfidf_scores.keys())

    for doc_id in all_docs:
        concept_score = doc_concept_scores.get(doc_id, 0.0)
        tfidf_score = doc_tfidf_scores.get(doc_id, 0.0)
        combined = (1 - concept_boost) * tfidf_score + concept_boost * concept_score

        stage_scores = {
            'concept_score': concept_score,
            'tfidf_score': tfidf_score,
            'combined_score': combined
        }
        results.append((doc_id, combined, stage_scores))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def find_relation_between(
    term_a: str,
    term_b: str,
    semantic_relations: List[Tuple[str, str, str, float]]
) -> List[Tuple[str, float]]:
    """
    Find semantic relations between two terms.

    Args:
        term_a: Source term
        term_b: Target term
        semantic_relations: List of (t1, relation, t2, weight) tuples

    Returns:
        List of (relation_type, weight) tuples
    """
    relations = []
    for t1, rel_type, t2, weight in semantic_relations:
        if t1 == term_a and t2 == term_b:
            relations.append((rel_type, weight))
        elif t2 == term_a and t1 == term_b:
            # Reverse direction
            relations.append((rel_type, weight * 0.9))  # Slight penalty for reverse

    return sorted(relations, key=lambda x: x[1], reverse=True)


def find_terms_with_relation(
    term: str,
    relation_type: str,
    semantic_relations: List[Tuple[str, str, str, float]],
    direction: str = 'forward'
) -> List[Tuple[str, float]]:
    """
    Find terms connected to a given term by a specific relation type.

    Args:
        term: Source term
        relation_type: Type of relation to follow
        semantic_relations: List of (t1, relation, t2, weight) tuples
        direction: 'forward' (term→x) or 'backward' (x→term)

    Returns:
        List of (target_term, weight) tuples
    """
    results = []
    for t1, rel_type, t2, weight in semantic_relations:
        if rel_type != relation_type:
            continue

        if direction == 'forward' and t1 == term:
            results.append((t2, weight))
        elif direction == 'backward' and t2 == term:
            results.append((t1, weight))

    return sorted(results, key=lambda x: x[1], reverse=True)


def complete_analogy(
    term_a: str,
    term_b: str,
    term_c: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    semantic_relations: List[Tuple[str, str, str, float]],
    embeddings: Optional[Dict[str, List[float]]] = None,
    top_n: int = 5,
    use_embeddings: bool = True,
    use_relations: bool = True
) -> List[Tuple[str, float, str]]:
    """
    Complete an analogy: "a is to b as c is to ?"

    Uses multiple strategies to find the best completion:
    1. Relation matching: Find what relation connects a→b, then find terms with
       the same relation from c
    2. Vector arithmetic: Use embeddings to compute d = c + (b - a)
    3. Pattern matching: Find terms that co-occur with c similar to how b co-occurs with a

    Example:
        "neural" is to "networks" as "knowledge" is to ?
        → "graphs" (both form compound technical terms with similar structure)

    Args:
        term_a: First term of the known pair
        term_b: Second term of the known pair
        term_c: First term of the query pair
        layers: Dictionary of layers
        semantic_relations: List of (t1, relation, t2, weight) tuples
        embeddings: Optional graph embeddings for vector arithmetic
        top_n: Number of candidates to return
        use_embeddings: Whether to use embedding-based completion
        use_relations: Whether to use relation-based completion

    Returns:
        List of (candidate_term, confidence, method) tuples, where method describes
        which approach found this candidate ('relation', 'embedding', 'pattern')
    """
    layer0 = layers[CorticalLayer.TOKENS]
    candidates: Dict[str, Tuple[float, str]] = {}  # term → (score, method)

    # Check that terms exist
    if not layer0.get_minicolumn(term_a) or not layer0.get_minicolumn(term_b):
        return []
    if not layer0.get_minicolumn(term_c):
        return []

    # Strategy 1: Relation-based completion
    if use_relations and semantic_relations:
        # Find relation between a and b
        relations_ab = find_relation_between(term_a, term_b, semantic_relations)

        for rel_type, rel_weight in relations_ab:
            # Find terms with same relation from c
            c_targets = find_terms_with_relation(
                term_c, rel_type, semantic_relations, direction='forward'
            )

            for target, target_weight in c_targets:
                # Don't include the input terms
                if target in {term_a, term_b, term_c}:
                    continue

                score = rel_weight * target_weight
                if target not in candidates or candidates[target][0] < score:
                    candidates[target] = (score, f'relation:{rel_type}')

    # Strategy 2: Embedding-based completion (vector arithmetic)
    if use_embeddings and embeddings:
        if term_a in embeddings and term_b in embeddings and term_c in embeddings:
            vec_a = embeddings[term_a]
            vec_b = embeddings[term_b]
            vec_c = embeddings[term_c]

            # d = c + (b - a)  (the analogy vector)
            vec_d = [
                c + (b - a)
                for a, b, c in zip(vec_a, vec_b, vec_c)
            ]

            # Find nearest terms to vec_d
            best_matches = []
            for term, vec in embeddings.items():
                if term in {term_a, term_b, term_c}:
                    continue

                # Cosine similarity
                dot = sum(d * v for d, v in zip(vec_d, vec))
                mag_d = sum(d * d for d in vec_d) ** 0.5
                mag_v = sum(v * v for v in vec) ** 0.5

                if mag_d > 0 and mag_v > 0:
                    similarity = dot / (mag_d * mag_v)
                    best_matches.append((term, similarity))

            # Sort by similarity and add to candidates
            best_matches.sort(key=lambda x: x[1], reverse=True)
            for term, sim in best_matches[:top_n * 2]:
                if sim > 0.5:  # Only include reasonably similar terms
                    if term not in candidates or candidates[term][0] < sim:
                        candidates[term] = (sim, 'embedding')

    # Strategy 3: Pattern matching (co-occurrence structure)
    col_a = layer0.get_minicolumn(term_a)
    col_b = layer0.get_minicolumn(term_b)
    col_c = layer0.get_minicolumn(term_c)

    if col_a and col_b and col_c:
        # Find terms that relate to c similarly to how b relates to a
        # I.e., if b co-occurs strongly with a, find terms that co-occur strongly with c

        a_neighbors = set(col_a.lateral_connections.keys())
        c_neighbors = set(col_c.lateral_connections.keys())

        # Look at c's neighbors that aren't a's neighbors (new context)
        for neighbor_id in c_neighbors:
            neighbor = layer0.get_by_id(neighbor_id)
            if not neighbor:
                continue

            term = neighbor.content
            if term in {term_a, term_b, term_c}:
                continue

            # Score based on how similar the neighbor's connection to c is
            # compared to b's connection to a
            c_weight = col_c.lateral_connections.get(neighbor_id, 0)
            b_to_a_weight = col_a.lateral_connections.get(col_b.id, 0)

            if c_weight > 0 and b_to_a_weight > 0:
                # The term should have similar connection strength pattern
                score = min(c_weight, b_to_a_weight) * 0.5
                if score > 0.1:
                    if term not in candidates or candidates[term][0] < score:
                        candidates[term] = (score, 'pattern')

    # Sort and return top candidates
    results = [
        (term, score, method)
        for term, (score, method) in candidates.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_n]


def complete_analogy_simple(
    term_a: str,
    term_b: str,
    term_c: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Simplified analogy completion using only term relationships.

    A lighter version of complete_analogy that doesn't require embeddings.
    Uses bigram patterns and co-occurrence to find analogies.

    Example:
        "neural" is to "networks" as "knowledge" is to ?
        → Looks for terms that form similar bigrams with "knowledge"

    Args:
        term_a: First term of the known pair
        term_b: Second term of the known pair
        term_c: First term of the query pair
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        semantic_relations: Optional semantic relations
        top_n: Number of candidates to return

    Returns:
        List of (candidate_term, confidence) tuples
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer1 = layers.get(CorticalLayer.BIGRAMS)

    candidates: Dict[str, float] = {}

    col_a = layer0.get_minicolumn(term_a)
    col_b = layer0.get_minicolumn(term_b)
    col_c = layer0.get_minicolumn(term_c)

    if not col_a or not col_b or not col_c:
        return []

    # Strategy 1: Bigram pattern matching
    if layer1:
        # Find bigrams containing "a b" pattern (bigrams use space separators)
        ab_bigram = f"{term_a} {term_b}"
        ba_bigram = f"{term_b} {term_a}"

        ab_col = layer1.get_minicolumn(ab_bigram)
        ba_col = layer1.get_minicolumn(ba_bigram)

        # If "a b" is a bigram, look for "c ?" bigrams
        if ab_col or ba_col:
            for bigram_col in layer1.minicolumns.values():
                bigram = bigram_col.content
                parts = bigram.split(' ')
                if len(parts) != 2:
                    continue

                first, second = parts

                # Look for bigrams starting with c
                if first == term_c and second not in {term_a, term_b, term_c}:
                    score = bigram_col.pagerank * 0.8
                    if second not in candidates or candidates[second] < score:
                        candidates[second] = score

                # Look for bigrams ending with c
                if second == term_c and first not in {term_a, term_b, term_c}:
                    score = bigram_col.pagerank * 0.6
                    if first not in candidates or candidates[first] < score:
                        candidates[first] = score

    # Strategy 2: Co-occurrence similarity
    # Find terms that co-occur with c like b co-occurs with a
    a_neighbors = col_a.lateral_connections
    c_neighbors = col_c.lateral_connections

    for neighbor_id, c_weight in c_neighbors.items():
        neighbor = layer0.get_by_id(neighbor_id)
        if not neighbor:
            continue

        term = neighbor.content
        if term in {term_a, term_b, term_c}:
            continue

        # Check if this term has similar connection pattern
        score = c_weight * 0.3
        if score > 0.05:
            candidates[term] = candidates.get(term, 0) + score

    # Strategy 3: Semantic relations (if available)
    if semantic_relations:
        relations_ab = find_relation_between(term_a, term_b, semantic_relations)
        for rel_type, rel_weight in relations_ab[:2]:  # Top 2 relations
            c_targets = find_terms_with_relation(
                term_c, rel_type, semantic_relations, direction='forward'
            )
            for target, target_weight in c_targets[:3]:  # Top 3 targets
                if target not in {term_a, term_b, term_c}:
                    score = rel_weight * target_weight
                    candidates[target] = candidates.get(target, 0) + score

    # Sort and return
    results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return results[:top_n]
