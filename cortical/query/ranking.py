"""
Ranking Module
=============

Multi-stage ranking and document type boosting for search results.

This module provides:
- Document type boosting (docs, code, tests)
- Conceptual vs implementation query detection
- Multi-stage ranking pipeline (concepts -> documents -> chunks)
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..tokenizer import Tokenizer

from .expansion import get_expanded_query_terms
from .search import find_documents_for_query


# Default boost factors for each document type
# Higher values make documents of that type rank higher
DOC_TYPE_BOOSTS = {
    'docs': 1.5,       # docs/ folder documentation
    'root_docs': 1.3,  # Root-level markdown (CLAUDE.md, README.md)
    'code': 1.0,       # Regular code files
    'test': 0.8,       # Test files (often less relevant for conceptual queries)
}

# Keywords that suggest a conceptual query (should boost documentation)
CONCEPTUAL_KEYWORDS = frozenset([
    'what', 'explain', 'describe', 'overview', 'introduction', 'concept',
    'architecture', 'design', 'pattern', 'algorithm', 'approach', 'method',
    'how does', 'why does', 'purpose', 'goal', 'rationale', 'theory',
    'understand', 'learn', 'documentation', 'guide', 'tutorial', 'example',
])

# Keywords that suggest an implementation query (should prefer code)
IMPLEMENTATION_KEYWORDS = frozenset([
    'where', 'implement', 'code', 'function', 'class', 'method', 'variable',
    'line', 'file', 'bug', 'fix', 'error', 'exception', 'call', 'invoke',
    'compute', 'calculate', 'return', 'parameter', 'argument',
])


def is_conceptual_query(query_text: str) -> bool:
    """
    Determine if a query is conceptual (should boost documentation).

    Conceptual queries ask about concepts, architecture, design, or
    explanations rather than specific code locations.

    Args:
        query_text: The search query

    Returns:
        True if the query appears to be conceptual
    """
    query_lower = query_text.lower()

    # Check for conceptual keywords
    conceptual_score = sum(
        1 for kw in CONCEPTUAL_KEYWORDS if kw in query_lower
    )

    # Check for implementation keywords
    implementation_score = sum(
        1 for kw in IMPLEMENTATION_KEYWORDS if kw in query_lower
    )

    # Boost if query starts with "what is" or "how does"
    if query_lower.startswith(('what is', 'what are', 'how does', 'explain')):
        conceptual_score += 2

    return conceptual_score > implementation_score


def get_doc_type_boost(
    doc_id: str,
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    custom_boosts: Optional[Dict[str, float]] = None
) -> float:
    """
    Get the boost factor for a document based on its type.

    Args:
        doc_id: Document ID
        doc_metadata: Optional metadata dict {doc_id: {doc_type: ..., ...}}
        custom_boosts: Optional custom boost factors

    Returns:
        Boost factor (1.0 = no boost)
    """
    boosts = custom_boosts or DOC_TYPE_BOOSTS

    # If we have metadata, use doc_type
    if doc_metadata and doc_id in doc_metadata:
        doc_type = doc_metadata[doc_id].get('doc_type', 'code')
        return boosts.get(doc_type, 1.0)

    # Fallback: infer from doc_id path
    if doc_id.endswith('.md'):
        if doc_id.startswith('docs/'):
            return boosts.get('docs', 1.5)
        return boosts.get('root_docs', 1.3)
    elif doc_id.startswith('tests/'):
        return boosts.get('test', 0.8)
    return boosts.get('code', 1.0)


def apply_doc_type_boost(
    results: List[Tuple[str, float]],
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    boost_docs: bool = True,
    custom_boosts: Optional[Dict[str, float]] = None
) -> List[Tuple[str, float]]:
    """
    Apply document type boosting to search results.

    Args:
        results: List of (doc_id, score) tuples
        doc_metadata: Optional metadata dict {doc_id: {doc_type: ..., ...}}
        boost_docs: Whether to apply boosting
        custom_boosts: Optional custom boost factors

    Returns:
        Re-ranked list of (doc_id, score) tuples
    """
    if not boost_docs:
        return results

    boosted = []
    for doc_id, score in results:
        boost = get_doc_type_boost(doc_id, doc_metadata, custom_boosts)
        boosted.append((doc_id, score * boost))

    # Re-sort by boosted scores
    boosted.sort(key=lambda x: -x[1])
    return boosted


def find_documents_with_boost(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    auto_detect_intent: bool = True,
    prefer_docs: bool = False,
    custom_boosts: Optional[Dict[str, float]] = None,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[Tuple[str, float]]:
    """
    Find documents with optional document-type boosting.

    This extends find_documents_for_query with doc_type boosting
    for improved ranking of documentation vs code.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return
        doc_metadata: Optional document metadata for boosting
        auto_detect_intent: If True, automatically boost docs for conceptual queries
        prefer_docs: If True, always boost documentation (overrides auto_detect)
        custom_boosts: Optional custom boost factors per doc_type
        use_expansion: Whether to expand query terms
        semantic_relations: Optional semantic relations for expansion
        use_semantic: Whether to use semantic relations

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    # Get base results (fetching more to allow re-ranking)
    base_results = find_documents_for_query(
        query_text, layers, tokenizer,
        top_n=top_n * 2,  # Get more candidates for re-ranking
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    # Determine if we should boost docs
    should_boost = prefer_docs or (auto_detect_intent and is_conceptual_query(query_text))

    if should_boost:
        boosted = apply_doc_type_boost(
            base_results, doc_metadata, True, custom_boosts
        )
        return boosted[:top_n]

    return base_results[:top_n]


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

    Unlike flat ranking (TF-IDF -> score), this uses a 4-stage pipeline:
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
    # Import here to avoid circular dependency
    from .passages import create_chunks, score_chunk

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
