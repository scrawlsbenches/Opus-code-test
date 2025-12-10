"""
Fingerprint Module
==================

Semantic fingerprinting for code comparison and similarity analysis.

A fingerprint is an interpretable representation of a text's semantic
content, including term weights, concept memberships, and relations.
Fingerprints can be compared to find similar code blocks or to explain
why two pieces of code are related.
"""

from typing import Dict, List, Tuple, Optional, TypedDict, Any
from collections import defaultdict
import math

from .layers import CorticalLayer, HierarchicalLayer
from .tokenizer import Tokenizer
from .code_concepts import get_concept_group


class SemanticFingerprint(TypedDict):
    """Structured representation of a text's semantic fingerprint."""
    terms: Dict[str, float]           # Term -> TF-IDF weight
    concepts: Dict[str, float]        # Concept group -> coverage score
    bigrams: Dict[str, float]         # Bigram -> weight
    top_terms: List[Tuple[str, float]]  # Top N terms by weight
    term_count: int                    # Total unique terms
    raw_text_hash: int                 # Hash of original text for identity check


def compute_fingerprint(
    text: str,
    tokenizer: Tokenizer,
    layers: Optional[Dict[CorticalLayer, HierarchicalLayer]] = None,
    top_n: int = 20
) -> SemanticFingerprint:
    """
    Compute the semantic fingerprint of a text.

    The fingerprint captures the semantic essence of the text in an
    interpretable format that can be compared with other fingerprints.

    Args:
        text: Input text to fingerprint
        tokenizer: Tokenizer instance
        layers: Optional corpus layers for TF-IDF weighting
        top_n: Number of top terms to include

    Returns:
        SemanticFingerprint with terms, concepts, bigrams, and metadata
    """
    # Tokenize
    tokens = tokenizer.tokenize(text)
    bigrams = tokenizer.extract_ngrams(tokens, n=2)

    # Compute term frequencies
    term_freq: Dict[str, int] = defaultdict(int)
    for token in tokens:
        term_freq[token] += 1

    # Compute bigram frequencies
    bigram_freq: Dict[str, int] = defaultdict(int)
    for bigram in bigrams:
        bigram_freq[bigram] += 1

    # Normalize to TF weights (or use corpus TF-IDF if available)
    total_terms = len(tokens) if tokens else 1
    term_weights: Dict[str, float] = {}

    for term, freq in term_freq.items():
        tf = freq / total_terms

        # If we have corpus layers, use IDF weighting
        if layers:
            layer0 = layers.get(CorticalLayer.TOKENS)
            if layer0:
                col = layer0.get_minicolumn(term)
                if col and col.tfidf > 0:
                    # Use corpus TF-IDF as weight
                    term_weights[term] = tf * col.tfidf
                else:
                    term_weights[term] = tf
            else:
                term_weights[term] = tf
        else:
            term_weights[term] = tf

    # Normalize bigram weights
    total_bigrams = len(bigrams) if bigrams else 1
    bigram_weights: Dict[str, float] = {}
    for bigram, freq in bigram_freq.items():
        bigram_weights[bigram] = freq / total_bigrams

    # Compute concept coverage
    concept_scores: Dict[str, float] = defaultdict(float)
    for term, weight in term_weights.items():
        groups = get_concept_group(term)
        for group in groups:
            concept_scores[group] += weight

    # Get top terms
    sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
    top_terms = sorted_terms[:top_n]

    return SemanticFingerprint(
        terms=term_weights,
        concepts=dict(concept_scores),
        bigrams=bigram_weights,
        top_terms=top_terms,
        term_count=len(term_weights),
        raw_text_hash=hash(text)
    )


def compare_fingerprints(
    fp1: SemanticFingerprint,
    fp2: SemanticFingerprint
) -> Dict[str, Any]:
    """
    Compare two fingerprints and compute similarity metrics.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint

    Returns:
        Dict with similarity scores and shared terms
    """
    # Check for identical text
    if fp1['raw_text_hash'] == fp2['raw_text_hash']:
        return {
            'identical': True,
            'term_similarity': 1.0,
            'concept_similarity': 1.0,
            'overall_similarity': 1.0,
            'shared_terms': list(fp1['terms'].keys()),
            'shared_concepts': list(fp1['concepts'].keys()),
        }

    # Compute cosine similarity for terms
    term_sim = _cosine_similarity(fp1['terms'], fp2['terms'])

    # Compute cosine similarity for concepts
    concept_sim = _cosine_similarity(fp1['concepts'], fp2['concepts'])

    # Compute bigram similarity
    bigram_sim = _cosine_similarity(fp1['bigrams'], fp2['bigrams'])

    # Find shared terms
    shared_terms = set(fp1['terms'].keys()) & set(fp2['terms'].keys())

    # Find shared concepts
    shared_concepts = set(fp1['concepts'].keys()) & set(fp2['concepts'].keys())

    # Compute overall similarity (weighted average)
    overall = 0.5 * term_sim + 0.3 * concept_sim + 0.2 * bigram_sim

    return {
        'identical': False,
        'term_similarity': term_sim,
        'concept_similarity': concept_sim,
        'bigram_similarity': bigram_sim,
        'overall_similarity': overall,
        'shared_terms': sorted(shared_terms),
        'shared_concepts': sorted(shared_concepts),
        'unique_to_fp1': sorted(set(fp1['terms'].keys()) - shared_terms),
        'unique_to_fp2': sorted(set(fp2['terms'].keys()) - shared_terms),
    }


def explain_fingerprint(
    fp: SemanticFingerprint,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Generate a human-readable explanation of a fingerprint.

    Args:
        fp: Fingerprint to explain
        top_n: Number of top items to include in explanation

    Returns:
        Dict with explanation components
    """
    # Get top terms
    top_terms = fp['top_terms'][:top_n]

    # Get top concepts
    sorted_concepts = sorted(
        fp['concepts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_concepts = sorted_concepts[:top_n]

    # Get top bigrams
    sorted_bigrams = sorted(
        fp['bigrams'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_bigrams = sorted_bigrams[:top_n]

    # Generate summary
    summary_parts = []
    if top_concepts:
        concept_names = [c[0] for c in top_concepts[:3]]
        summary_parts.append(f"Concepts: {', '.join(concept_names)}")

    if top_terms:
        term_names = [t[0] for t in top_terms[:5]]
        summary_parts.append(f"Key terms: {', '.join(term_names)}")

    return {
        'summary': ' | '.join(summary_parts) if summary_parts else 'No significant terms',
        'top_terms': top_terms,
        'top_concepts': top_concepts,
        'top_bigrams': top_bigrams,
        'term_count': fp['term_count'],
        'concept_coverage': len(fp['concepts']),
    }


def explain_similarity(
    fp1: SemanticFingerprint,
    fp2: SemanticFingerprint,
    comparison: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a human-readable explanation of why two fingerprints are similar.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        comparison: Optional pre-computed comparison result

    Returns:
        Human-readable explanation string
    """
    if comparison is None:
        comparison = compare_fingerprints(fp1, fp2)

    if comparison['identical']:
        return "These texts are identical."

    lines = []
    similarity = comparison['overall_similarity']

    if similarity > 0.8:
        lines.append("These texts are highly similar.")
    elif similarity > 0.5:
        lines.append("These texts have moderate similarity.")
    elif similarity > 0.2:
        lines.append("These texts have some common elements.")
    else:
        lines.append("These texts are quite different.")

    # Explain shared concepts
    shared_concepts = comparison.get('shared_concepts', [])
    if shared_concepts:
        lines.append(f"Shared concept domains: {', '.join(shared_concepts[:5])}")

    # Explain shared terms
    shared_terms = comparison.get('shared_terms', [])
    if shared_terms:
        # Get top shared terms by combined weight
        term_importance = []
        for term in shared_terms:
            weight = fp1['terms'].get(term, 0) + fp2['terms'].get(term, 0)
            term_importance.append((term, weight))
        term_importance.sort(key=lambda x: x[1], reverse=True)
        top_shared = [t[0] for t in term_importance[:5]]
        lines.append(f"Key shared terms: {', '.join(top_shared)}")

    # Note differences
    unique1 = comparison.get('unique_to_fp1', [])
    unique2 = comparison.get('unique_to_fp2', [])
    if unique1 or unique2:
        lines.append(f"First text has {len(unique1)} unique terms, second has {len(unique2)}.")

    return '\n'.join(lines)


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.

    Args:
        vec1: First vector as {dimension: value} dict
        vec2: Second vector as {dimension: value} dict

    Returns:
        Cosine similarity in range [0, 1]
    """
    if not vec1 or not vec2:
        return 0.0

    # Find common dimensions
    common_keys = set(vec1.keys()) & set(vec2.keys())

    if not common_keys:
        return 0.0

    # Compute dot product
    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

    # Compute magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)
