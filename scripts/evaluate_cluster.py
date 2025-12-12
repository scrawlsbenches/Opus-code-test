#!/usr/bin/env python3
"""
Cluster Coverage Evaluation Script
===================================

Task #127: Evaluate cluster quality and coverage for topic-based document groups.

This script helps determine if a document cluster has sufficient coverage
or needs more documents to form a coherent topic group.

Usage:
    # Find and evaluate documents matching a topic
    python scripts/evaluate_cluster.py --topic "customer service"

    # Evaluate specific documents
    python scripts/evaluate_cluster.py --documents customer_support_fundamentals,complaint_resolution

    # Find documents containing specific keywords
    python scripts/evaluate_cluster.py --keywords customer,ticket,escalation

    # Use existing corpus file
    python scripts/evaluate_cluster.py --corpus corpus_dev.pkl --topic "machine learning"

    # Verbose output with expansion suggestions
    python scripts/evaluate_cluster.py --topic "customer" --verbose --suggest
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortical import CorticalTextProcessor, CorticalLayer


def load_corpus(
    processor: CorticalTextProcessor,
    samples_dir: str = "samples"
) -> int:
    """Load all sample documents into the processor."""
    loaded = 0
    samples_path = Path(samples_dir)

    if not samples_path.is_dir():
        print(f"Samples directory not found: {samples_dir}")
        return 0

    for filepath in sorted(samples_path.glob("*.txt")):
        try:
            content = filepath.read_text(encoding='utf-8')
            doc_id = filepath.stem
            processor.process_document(doc_id, content)
            loaded += 1
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    return loaded


def find_documents_by_topic(
    processor: CorticalTextProcessor,
    topic: str,
    threshold: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Find documents related to a topic using semantic search.

    Returns list of (doc_id, score) tuples.
    """
    results = processor.find_documents_for_query(topic, top_n=50)
    # Filter by threshold
    return [(doc_id, score) for doc_id, score in results if score >= threshold]


def find_documents_by_keywords(
    processor: CorticalTextProcessor,
    keywords: List[str],
    min_keywords: int = 1
) -> List[str]:
    """
    Find documents containing at least min_keywords of the specified keywords.
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]
    doc_keyword_counts: Dict[str, int] = defaultdict(int)

    for keyword in keywords:
        keyword_lower = keyword.lower()
        col = layer0.get_minicolumn(keyword_lower)
        if col:
            for doc_id in col.document_ids:
                doc_keyword_counts[doc_id] += 1

    return [doc_id for doc_id, count in doc_keyword_counts.items()
            if count >= min_keywords]


def compute_document_similarity(
    processor: CorticalTextProcessor,
    doc1: str,
    doc2: str
) -> float:
    """
    Compute similarity between two documents based on shared terms.
    Uses Jaccard similarity of term sets weighted by TF-IDF.
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]

    # Get terms for each document
    terms1: Dict[str, float] = {}
    terms2: Dict[str, float] = {}

    for col in layer0.minicolumns.values():
        if doc1 in col.document_ids:
            terms1[col.content] = col.tfidf_per_doc.get(doc1, col.tfidf)
        if doc2 in col.document_ids:
            terms2[col.content] = col.tfidf_per_doc.get(doc2, col.tfidf)

    if not terms1 or not terms2:
        return 0.0

    # Compute weighted Jaccard
    common = set(terms1.keys()) & set(terms2.keys())
    if not common:
        return 0.0

    # Sum of minimum weights / sum of maximum weights
    min_sum = sum(min(terms1[t], terms2[t]) for t in common)
    all_terms = set(terms1.keys()) | set(terms2.keys())
    max_sum = sum(max(terms1.get(t, 0), terms2.get(t, 0)) for t in all_terms)

    return min_sum / max_sum if max_sum > 0 else 0.0


def compute_cluster_metrics(
    processor: CorticalTextProcessor,
    cluster_docs: List[str],
    all_docs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a document cluster.

    Returns:
        Dictionary with cohesion, separation, coverage, and diversity metrics.
    """
    if all_docs is None:
        all_docs = list(processor.documents.keys())

    outside_docs = [d for d in all_docs if d not in cluster_docs]

    layer0 = processor.layers[CorticalLayer.TOKENS]
    layer2 = processor.layers[CorticalLayer.CONCEPTS]

    # 1. Internal Cohesion: average similarity within cluster
    internal_similarities = []
    for i, doc1 in enumerate(cluster_docs):
        for doc2 in cluster_docs[i+1:]:
            sim = compute_document_similarity(processor, doc1, doc2)
            internal_similarities.append(sim)

    cohesion = sum(internal_similarities) / len(internal_similarities) if internal_similarities else 0.0

    # 2. External Separation: average similarity to outside documents
    external_similarities = []
    for cluster_doc in cluster_docs:
        for outside_doc in outside_docs[:20]:  # Sample for efficiency
            sim = compute_document_similarity(processor, cluster_doc, outside_doc)
            external_similarities.append(sim)

    separation = 1.0 - (sum(external_similarities) / len(external_similarities) if external_similarities else 0.0)

    # 3. Concept Coverage: unique concepts captured by cluster
    cluster_tokens: Set[str] = set()
    cluster_concepts: Set[str] = set()

    for doc_id in cluster_docs:
        for col in layer0.minicolumns.values():
            if doc_id in col.document_ids:
                cluster_tokens.add(col.content)

    # Find which concepts contain our tokens
    for concept_col in layer2.minicolumns.values():
        for token_id in concept_col.feedforward_connections:
            token_col = layer0.get_by_id(token_id)
            if token_col and token_col.content in cluster_tokens:
                cluster_concepts.add(concept_col.content)
                break

    # 4. Term Diversity: vocabulary richness (unique terms / total occurrences)
    total_term_occurrences = 0
    for doc_id in cluster_docs:
        for col in layer0.minicolumns.values():
            if doc_id in col.document_ids:
                total_term_occurrences += 1

    diversity = len(cluster_tokens) / total_term_occurrences if total_term_occurrences > 0 else 0.0

    # 5. Hub Document: document most connected to others in cluster
    hub_doc = cluster_docs[0] if cluster_docs else None  # Default to first doc
    if len(cluster_docs) > 1:
        max_avg_sim = -1.0
        for doc in cluster_docs:
            avg_sim = sum(
                compute_document_similarity(processor, doc, other)
                for other in cluster_docs if other != doc
            ) / (len(cluster_docs) - 1)
            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                hub_doc = doc

    # 6. Key Terms: most important terms in cluster by TF-IDF
    term_scores: Dict[str, float] = defaultdict(float)
    for col in layer0.minicolumns.values():
        docs_in_cluster = col.document_ids & set(cluster_docs)
        if docs_in_cluster:
            # Average TF-IDF across cluster documents
            for doc_id in docs_in_cluster:
                term_scores[col.content] += col.tfidf_per_doc.get(doc_id, col.tfidf)
            term_scores[col.content] /= len(docs_in_cluster)

    key_terms = sorted(term_scores.items(), key=lambda x: -x[1])[:15]

    return {
        'cohesion': cohesion,
        'separation': separation,
        'concept_count': len(cluster_concepts),
        'term_count': len(cluster_tokens),
        'diversity': diversity,
        'hub_document': hub_doc,
        'key_terms': key_terms,
        'cluster_tokens': cluster_tokens,
        'cluster_concepts': cluster_concepts,
    }


def find_expansion_suggestions(
    processor: CorticalTextProcessor,
    cluster_tokens: Set[str],
    cluster_docs: List[str],
    max_suggestions: int = 5
) -> List[Tuple[str, str]]:
    """
    Find topics that could expand the cluster coverage.

    Looks for:
    1. Related terms that appear in few/no cluster documents
    2. Concepts connected to cluster concepts but not well covered

    Returns:
        List of (suggestion, reason) tuples
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]
    suggestions = []

    # Find terms connected to cluster terms but not in cluster
    related_terms: Dict[str, float] = defaultdict(float)

    for token in list(cluster_tokens)[:50]:  # Sample for efficiency
        col = layer0.get_minicolumn(token)
        if col:
            for neighbor_id, weight in col.lateral_connections.items():
                neighbor = layer0.get_by_id(neighbor_id)
                if neighbor and neighbor.content not in cluster_tokens:
                    # Check if this term appears mostly outside our cluster
                    docs_in_cluster = neighbor.document_ids & set(cluster_docs)
                    docs_outside = neighbor.document_ids - set(cluster_docs)
                    if len(docs_outside) > len(docs_in_cluster):
                        related_terms[neighbor.content] += weight

    # Sort by connection strength
    top_related = sorted(related_terms.items(), key=lambda x: -x[1])[:max_suggestions * 2]

    for term, weight in top_related:
        if len(suggestions) >= max_suggestions:
            break

        col = layer0.get_minicolumn(term)
        if col:
            # Find what documents have this term
            example_docs = list(col.document_ids - set(cluster_docs))[:2]
            if example_docs:
                reason = f"Related to cluster terms, found in: {', '.join(example_docs[:2])}"
            else:
                reason = f"Strongly connected to cluster vocabulary"
            suggestions.append((term, reason))

    return suggestions


def assess_coverage(metrics: Dict[str, Any], num_docs: int) -> Tuple[str, str]:
    """
    Assess cluster coverage quality and provide recommendation.

    Returns:
        (assessment_label, explanation)
    """
    cohesion = metrics['cohesion']
    separation = metrics['separation']
    concept_count = metrics['concept_count']

    # Scoring
    score = 0
    issues = []
    strengths = []

    # Cohesion assessment
    if cohesion >= 0.3:
        score += 2
        strengths.append("strong internal connectivity")
    elif cohesion >= 0.15:
        score += 1
        strengths.append("moderate internal connectivity")
    else:
        issues.append("weak internal connectivity")

    # Separation assessment
    if separation >= 0.6:
        score += 2
        strengths.append("well-separated from other topics")
    elif separation >= 0.4:
        score += 1
    else:
        issues.append("overlaps significantly with other topics")

    # Coverage assessment
    if concept_count >= 5:
        score += 1
        strengths.append(f"covers {concept_count} concept clusters")
    elif concept_count < 2:
        issues.append("limited concept coverage")

    # Document count
    if num_docs >= 5:
        score += 1
    elif num_docs < 3:
        issues.append("too few documents")

    # Generate assessment
    if score >= 5:
        label = "STRONG"
        explanation = f"Cluster is well-formed with {', '.join(strengths)}."
    elif score >= 3:
        label = "ADEQUATE"
        if issues:
            explanation = f"Cluster is usable but could improve: {', '.join(issues)}."
        else:
            explanation = f"Cluster forms a coherent topic group with {', '.join(strengths)}."
    else:
        label = "NEEDS EXPANSION"
        explanation = f"Cluster needs more coverage: {', '.join(issues)}."

    return label, explanation


def print_cluster_analysis(
    cluster_name: str,
    cluster_docs: List[str],
    metrics: Dict[str, Any],
    suggestions: List[Tuple[str, str]],
    verbose: bool = False
) -> None:
    """Print formatted cluster analysis report."""

    assessment_label, assessment_explanation = assess_coverage(metrics, len(cluster_docs))

    # Header
    title = f"Cluster Analysis: {cluster_name} ({len(cluster_docs)} documents)"
    print(f"\n{title}")
    print("=" * len(title))

    # Documents
    print("\nDocuments:")
    hub = metrics.get('hub_document')
    for doc in sorted(cluster_docs):
        marker = " (hub)" if doc == hub else ""
        print(f"  * {doc}{marker}")

    # Metrics
    print("\nMetrics:")

    cohesion = metrics['cohesion']
    if cohesion >= 0.3:
        cohesion_label = "strong"
    elif cohesion >= 0.15:
        cohesion_label = "moderate"
    else:
        cohesion_label = "weak"
    print(f"  Internal Cohesion:    {cohesion:.2f} ({cohesion_label})")

    separation = metrics['separation']
    if separation >= 0.6:
        sep_label = "good"
    elif separation >= 0.4:
        sep_label = "moderate"
    else:
        sep_label = "low"
    print(f"  External Separation:  {separation:.2f} ({sep_label})")

    print(f"  Concept Coverage:     {metrics['concept_count']} concepts")
    print(f"  Term Diversity:       {metrics['diversity']:.2f}")
    print(f"  Unique Terms:         {metrics['term_count']}")

    # Assessment
    if assessment_label == "STRONG":
        symbol = "[OK]"
    elif assessment_label == "ADEQUATE":
        symbol = "[~]"
    else:
        symbol = "[!]"

    print(f"\nCoverage Assessment: {assessment_label} {symbol}")
    print(f"  {assessment_explanation}")

    # Key terms
    if verbose:
        print("\nKey Terms (by TF-IDF):")
        for term, score in metrics['key_terms'][:10]:
            print(f"  - {term}: {score:.3f}")

    # Expansion suggestions
    if suggestions:
        print("\nPotential Expansions:")
        for term, reason in suggestions:
            print(f"  * {term}")
            if verbose:
                print(f"    ({reason})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cluster quality and coverage for document groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --topic "customer service"
  %(prog)s --documents customer_support_fundamentals,complaint_resolution
  %(prog)s --keywords customer,ticket,escalation
  %(prog)s --corpus corpus_dev.pkl --topic "machine learning" --verbose
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--topic", "-t",
        help="Find documents by semantic topic search"
    )
    input_group.add_argument(
        "--documents", "-d",
        help="Comma-separated list of document IDs to evaluate"
    )
    input_group.add_argument(
        "--keywords", "-k",
        help="Comma-separated keywords to find documents containing them"
    )

    # Corpus options
    parser.add_argument(
        "--corpus", "-c",
        help="Path to saved corpus file (default: load from samples/)"
    )
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Directory containing sample documents (default: samples/)"
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including key terms"
    )
    parser.add_argument(
        "--suggest", "-s",
        action="store_true",
        help="Show expansion suggestions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum score threshold for topic search (default: 0.1)"
    )
    parser.add_argument(
        "--min-keywords",
        type=int,
        default=1,
        help="Minimum keywords required for document match (default: 1)"
    )

    args = parser.parse_args()

    # Load or create processor
    if args.corpus and os.path.exists(args.corpus):
        print(f"Loading corpus from {args.corpus}...")
        processor = CorticalTextProcessor.load(args.corpus)
        print(f"Loaded {len(processor.documents)} documents")
    else:
        print(f"Loading documents from {args.samples_dir}/...")
        processor = CorticalTextProcessor()
        num_loaded = load_corpus(processor, args.samples_dir)
        if num_loaded == 0:
            print("Error: No documents loaded")
            sys.exit(1)
        print(f"Loaded {num_loaded} documents, computing analysis...")
        processor.compute_all(verbose=False)

    # Find cluster documents
    cluster_name = ""
    cluster_docs: List[str] = []

    if args.topic:
        cluster_name = args.topic
        results = find_documents_by_topic(processor, args.topic, args.threshold)
        cluster_docs = [doc_id for doc_id, _ in results]
        if not cluster_docs:
            print(f"No documents found matching topic: {args.topic}")
            sys.exit(1)
        print(f"Found {len(cluster_docs)} documents matching topic '{args.topic}'")

    elif args.documents:
        doc_ids = [d.strip() for d in args.documents.split(",")]
        cluster_name = f"Selected ({len(doc_ids)} docs)"
        # Validate document IDs
        missing = [d for d in doc_ids if d not in processor.documents]
        if missing:
            print(f"Warning: Documents not found: {', '.join(missing)}")
        cluster_docs = [d for d in doc_ids if d in processor.documents]
        if not cluster_docs:
            print("Error: None of the specified documents were found")
            sys.exit(1)

    elif args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",")]
        cluster_name = f"Keywords: {', '.join(keywords[:3])}"
        cluster_docs = find_documents_by_keywords(processor, keywords, args.min_keywords)
        if not cluster_docs:
            print(f"No documents found containing keywords: {', '.join(keywords)}")
            sys.exit(1)
        print(f"Found {len(cluster_docs)} documents with keywords")

    # Compute metrics
    metrics = compute_cluster_metrics(processor, cluster_docs)

    # Find expansion suggestions if requested
    suggestions = []
    if args.suggest:
        suggestions = find_expansion_suggestions(
            processor,
            metrics['cluster_tokens'],
            cluster_docs
        )

    # Print analysis
    print_cluster_analysis(
        cluster_name,
        cluster_docs,
        metrics,
        suggestions,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
