#!/usr/bin/env python3
"""
Suggest Memory Consolidation Opportunities

Analyzes memory documents and suggests consolidation opportunities based on:
- Term overlap and semantic similarity
- Repeated concepts across multiple entries
- Memory age (old unconsolidated memories)
- Cluster analysis (memories discussing similar topics)

Usage:
    python scripts/suggest_consolidation.py
    python scripts/suggest_consolidation.py --threshold 0.7
    python scripts/suggest_consolidation.py --min-cluster 2 --output json
    python scripts/suggest_consolidation.py --min-age-days 30
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer


def parse_memory_date(doc_id: str) -> datetime:
    """
    Extract date from memory document ID.

    Supports formats:
    - samples/memories/2025-12-14-topic.md
    - samples/memories/2025-12-14_20-54-35_3b3a-topic.md
    - samples/memories/concept-*.md (returns very old date for concepts)

    Args:
        doc_id: Document ID path

    Returns:
        datetime object, or very old date if parsing fails
    """
    filename = doc_id.split('/')[-1]

    # Concept documents are considered "timeless" (very old)
    if filename.startswith('concept-'):
        return datetime(2000, 1, 1)

    # Try timestamp format first: YYYY-MM-DD_HH-MM-SS_XXXX-topic.md
    if '_' in filename:
        date_part = filename.split('_')[0]
        parts = date_part.split('-')
        if len(parts) >= 3:
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                return datetime(year, month, day)
            except (ValueError, IndexError):
                pass

    # Extract date from YYYY-MM-DD pattern (basic format)
    parts = filename.split('-')
    if len(parts) >= 3:
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            return datetime(year, month, day)
        except (ValueError, IndexError):
            pass

    # Default to very old if can't parse
    return datetime(2000, 1, 1)


def get_memory_age_days(doc_id: str) -> int:
    """Get the age of a memory in days from today."""
    memory_date = parse_memory_date(doc_id)
    today = datetime.now()
    return (today - memory_date).days


def is_concept_doc(doc_id: str) -> bool:
    """Check if document is a concept document."""
    filename = doc_id.split('/')[-1]
    return filename.startswith('concept-')


def compute_pairwise_similarity(
    processor: CorticalTextProcessor,
    doc_ids: List[str]
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise similarity between all documents using fingerprints.

    Args:
        processor: CorticalTextProcessor instance
        doc_ids: List of document IDs to compare

    Returns:
        Dictionary mapping (doc_id1, doc_id2) to similarity score
    """
    similarities = {}

    # Compute fingerprints for all documents
    fingerprints = {}
    for doc_id in doc_ids:
        content = processor.documents.get(doc_id, '')
        if content:
            fingerprints[doc_id] = processor.get_fingerprint(content, top_n=20)

    # Compute pairwise similarities
    for i, doc_id1 in enumerate(doc_ids):
        for doc_id2 in doc_ids[i+1:]:
            if doc_id1 in fingerprints and doc_id2 in fingerprints:
                fp1 = fingerprints[doc_id1]
                fp2 = fingerprints[doc_id2]
                comparison = processor.compare_fingerprints(fp1, fp2)
                similarity = comparison.get('overall_similarity', 0.0)
                similarities[(doc_id1, doc_id2)] = similarity
                similarities[(doc_id2, doc_id1)] = similarity  # Symmetric

    return similarities


def cluster_memories(
    processor: CorticalTextProcessor,
    memory_ids: List[str],
    min_cluster_size: int = 2,
    resolution: float = 1.0
) -> Dict[int, List[str]]:
    """
    Cluster memory documents using similarity-based grouping.

    Since we have a small number of memories, we'll use a simple
    similarity-based clustering approach instead of Louvain.

    Args:
        processor: CorticalTextProcessor instance
        memory_ids: List of memory document IDs
        min_cluster_size: Minimum documents per cluster
        resolution: Clustering resolution (higher = more clusters)

    Returns:
        Dictionary mapping cluster_id to list of document IDs
    """
    if len(memory_ids) < min_cluster_size:
        return {}

    # Compute fingerprints for all memories
    fingerprints = {}
    for doc_id in memory_ids:
        content = processor.documents.get(doc_id, '')
        if content:
            fingerprints[doc_id] = processor.get_fingerprint(content, top_n=20)

    # Build similarity graph
    # edges[doc1][doc2] = similarity
    edges: Dict[str, Dict[str, float]] = defaultdict(dict)

    for i, doc_id1 in enumerate(memory_ids):
        if doc_id1 not in fingerprints:
            continue

        for doc_id2 in memory_ids[i+1:]:
            if doc_id2 not in fingerprints:
                continue

            fp1 = fingerprints[doc_id1]
            fp2 = fingerprints[doc_id2]
            comparison = processor.compare_fingerprints(fp1, fp2)
            similarity = comparison.get('overall_similarity', 0.0)

            # Adjust threshold based on resolution
            # Higher resolution = higher threshold = more clusters
            threshold = 0.3 * resolution

            if similarity >= threshold:
                edges[doc_id1][doc_id2] = similarity
                edges[doc_id2][doc_id1] = similarity

    # Simple greedy clustering: find connected components
    visited = set()
    clusters = {}
    cluster_id = 0

    def dfs(doc_id: str, cluster: List[str]):
        """Depth-first search to find connected component."""
        if doc_id in visited:
            return
        visited.add(doc_id)
        cluster.append(doc_id)

        # Visit neighbors
        for neighbor in edges.get(doc_id, {}):
            if neighbor not in visited:
                dfs(neighbor, cluster)

    # Find all connected components
    for doc_id in memory_ids:
        if doc_id not in visited:
            cluster = []
            dfs(doc_id, cluster)

            if len(cluster) >= min_cluster_size:
                clusters[cluster_id] = cluster
                cluster_id += 1

    return clusters


def extract_cluster_topics(
    processor: CorticalTextProcessor,
    doc_ids: List[str],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Extract top terms representing a cluster of documents.

    Args:
        processor: CorticalTextProcessor instance
        doc_ids: List of document IDs in the cluster
        top_n: Number of top terms to extract

    Returns:
        List of (term, score) tuples sorted by importance
    """
    # Aggregate term weights across all documents in cluster
    term_scores: Dict[str, float] = defaultdict(float)

    layer0 = processor.layers[CorticalLayer.TOKENS]

    # For each document, get its top terms
    for doc_id in doc_ids:
        content = processor.documents.get(doc_id, '')
        if not content:
            continue

        fp = processor.get_fingerprint(content, top_n=20)
        terms = fp.get('terms', {})

        # Weight by document's PageRank in the cluster
        doc_col = processor.layers[CorticalLayer.DOCUMENTS].get_by_id(f"L3_{doc_id}")
        doc_weight = doc_col.pagerank if doc_col else 1.0

        for term, weight in terms.items():
            # Also consider the term's global importance
            term_col = layer0.get_minicolumn(term)
            term_importance = term_col.pagerank if term_col else 0.0

            # Combined score: term weight in doc * doc importance * term importance
            term_scores[term] += weight * doc_weight * (1 + term_importance)

    # Sort by score and return top N
    sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
    return sorted_terms[:top_n]


def suggest_consolidations(
    processor: CorticalTextProcessor,
    min_overlap: float = 0.5,
    min_cluster_size: int = 2,
    min_age_days: int = 30,
    resolution: float = 1.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze memories and suggest consolidation opportunities.

    Args:
        processor: CorticalTextProcessor instance
        min_overlap: Minimum similarity for pair suggestions (0.0-1.0)
        min_cluster_size: Minimum memories per cluster
        min_age_days: Minimum age in days for "old memory" warnings
        resolution: Louvain clustering resolution
        verbose: Print detailed information

    Returns:
        Dictionary with suggestions categorized by type
    """
    # Filter for memory documents (not concept docs, not decisions)
    memory_ids = [
        doc_id for doc_id in processor.documents.keys()
        if doc_id.startswith('samples/memories/') and not is_concept_doc(doc_id)
    ]

    concept_ids = [
        doc_id for doc_id in processor.documents.keys()
        if doc_id.startswith('samples/memories/') and is_concept_doc(doc_id)
    ]

    if verbose:
        print(f"Found {len(memory_ids)} memory entries")
        print(f"Found {len(concept_ids)} concept documents")

    suggestions = {
        'clusters': [],
        'similar_pairs': [],
        'old_memories': [],
        'stats': {
            'total_memories': len(memory_ids),
            'total_concepts': len(concept_ids),
            'analyzed_memories': len(memory_ids)
        }
    }

    if len(memory_ids) < 2:
        return suggestions

    # 1. Cluster analysis - find groups of related memories
    if verbose:
        print("\nClustering memories...")

    clusters = cluster_memories(
        processor,
        memory_ids,
        min_cluster_size=min_cluster_size,
        resolution=resolution
    )

    for cluster_id, doc_ids in clusters.items():
        # Extract topic terms for this cluster
        topics = extract_cluster_topics(processor, doc_ids, top_n=5)
        topic_terms = [term for term, _ in topics]

        # Suggest a concept name based on top terms
        concept_name = "-".join(topic_terms[:3])  # e.g., "security-testing-fuzzing"

        suggestions['clusters'].append({
            'cluster_id': cluster_id,
            'document_count': len(doc_ids),
            'documents': doc_ids,
            'suggested_concept': concept_name,
            'topics': topics,
            'message': f"These {len(doc_ids)} memories discuss '{concept_name}'. Consider creating samples/memories/concept-{concept_name}.md"
        })

    # 2. High similarity pairs - find memories with strong overlap
    if verbose:
        print("Computing pairwise similarities...")

    similarities = compute_pairwise_similarity(processor, memory_ids)

    for (doc_id1, doc_id2), similarity in similarities.items():
        if similarity >= min_overlap and doc_id1 < doc_id2:  # Avoid duplicates
            # Get shared terms
            fp1 = processor.get_fingerprint(processor.documents[doc_id1], top_n=20)
            fp2 = processor.get_fingerprint(processor.documents[doc_id2], top_n=20)
            comparison = processor.compare_fingerprints(fp1, fp2)
            shared = list(comparison.get('shared_terms', []))[:5]

            suggestions['similar_pairs'].append({
                'doc1': doc_id1,
                'doc2': doc_id2,
                'similarity': similarity,
                'shared_terms': shared,
                'message': f"{doc_id1.split('/')[-1]} and {doc_id2.split('/')[-1]} have {similarity:.1%} overlap (shared: {', '.join(shared[:3])}). Consider merging?"
            })

    # 3. Old memories - find memories that haven't been consolidated
    if verbose:
        print("Checking for old memories...")

    today = datetime.now()
    for doc_id in memory_ids:
        age_days = get_memory_age_days(doc_id)
        if age_days >= min_age_days:
            memory_date = parse_memory_date(doc_id)
            suggestions['old_memories'].append({
                'doc_id': doc_id,
                'age_days': age_days,
                'date': memory_date.strftime("%Y-%m-%d"),
                'message': f"{doc_id.split('/')[-1]} is {age_days} days old. Consider consolidating into a concept document?"
            })

    # Sort suggestions
    suggestions['clusters'].sort(key=lambda x: x['document_count'], reverse=True)
    suggestions['similar_pairs'].sort(key=lambda x: x['similarity'], reverse=True)
    suggestions['old_memories'].sort(key=lambda x: x['age_days'], reverse=True)

    return suggestions


def format_suggestions_text(suggestions: Dict[str, Any], verbose: bool = False) -> str:
    """Format suggestions as human-readable text."""
    lines = []

    lines.append("=" * 70)
    lines.append("MEMORY CONSOLIDATION SUGGESTIONS")
    lines.append("=" * 70)
    lines.append("")

    stats = suggestions['stats']
    lines.append(f"Analyzed {stats['total_memories']} memory entries")
    lines.append(f"Found {stats['total_concepts']} existing concept documents")
    lines.append("")

    # Cluster suggestions
    if suggestions['clusters']:
        lines.append(f"{'─' * 70}")
        lines.append(f"CLUSTER SUGGESTIONS ({len(suggestions['clusters'])})")
        lines.append(f"{'─' * 70}")
        lines.append("")

        for i, cluster in enumerate(suggestions['clusters'], 1):
            lines.append(f"[{i}] {cluster['message']}")
            if verbose:
                lines.append(f"    Documents ({cluster['document_count']}):")
                for doc_id in cluster['documents']:
                    filename = doc_id.split('/')[-1]
                    age = get_memory_age_days(doc_id)
                    lines.append(f"      - {filename} ({age} days old)")
                lines.append(f"    Key topics: {', '.join(t for t, _ in cluster['topics'][:5])}")
            lines.append("")
    else:
        lines.append("No cluster suggestions found.")
        lines.append("")

    # Similar pair suggestions
    if suggestions['similar_pairs']:
        lines.append(f"{'─' * 70}")
        lines.append(f"HIGH OVERLAP PAIRS ({len(suggestions['similar_pairs'])})")
        lines.append(f"{'─' * 70}")
        lines.append("")

        for i, pair in enumerate(suggestions['similar_pairs'][:10], 1):  # Limit to top 10
            lines.append(f"[{i}] {pair['message']}")
            if verbose:
                lines.append(f"    Similarity: {pair['similarity']:.1%}")
                lines.append(f"    Shared terms: {', '.join(pair['shared_terms'])}")
            lines.append("")
    else:
        lines.append("No high-overlap pairs found.")
        lines.append("")

    # Old memory suggestions
    if suggestions['old_memories']:
        lines.append(f"{'─' * 70}")
        lines.append(f"OLD MEMORIES ({len(suggestions['old_memories'])})")
        lines.append(f"{'─' * 70}")
        lines.append("")

        for i, old in enumerate(suggestions['old_memories'][:10], 1):  # Limit to top 10
            lines.append(f"[{i}] {old['message']}")
            lines.append("")
    else:
        lines.append("No old memories found.")
        lines.append("")

    lines.append("=" * 70)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")

    if suggestions['clusters']:
        lines.append("1. Review cluster suggestions and create concept documents:")
        for cluster in suggestions['clusters'][:3]:
            lines.append(f"   - Create: samples/memories/concept-{cluster['suggested_concept']}.md")
        lines.append("")

    if suggestions['similar_pairs']:
        lines.append("2. Review high-overlap pairs and consider merging:")
        for pair in suggestions['similar_pairs'][:3]:
            lines.append(f"   - Compare: {pair['doc1'].split('/')[-1]} vs {pair['doc2'].split('/')[-1]}")
        lines.append("")

    if suggestions['old_memories']:
        lines.append("3. Review old memories and consolidate into concepts:")
        for old in suggestions['old_memories'][:3]:
            lines.append(f"   - Review: {old['doc_id'].split('/')[-1]} ({old['age_days']} days)")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Suggest memory consolidation opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default analysis
  %(prog)s --threshold 0.7              # Higher similarity threshold
  %(prog)s --min-cluster 3              # Require 3+ memories per cluster
  %(prog)s --min-age-days 60            # Only flag memories older than 60 days
  %(prog)s --output json                # JSON output
  %(prog)s --verbose                    # Detailed output
        """
    )

    parser.add_argument(
        '--corpus', '-c',
        default='corpus_dev.pkl',
        help='Corpus file path (default: corpus_dev.pkl)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Minimum similarity for pair suggestions (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '--min-cluster',
        type=int,
        default=2,
        help='Minimum memories per cluster (default: 2)'
    )
    parser.add_argument(
        '--min-age-days',
        type=int,
        default=30,
        help='Minimum age in days for old memory warnings (default: 30)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='Louvain clustering resolution (default: 1.0, higher = more clusters)'
    )
    parser.add_argument(
        '--output', '-o',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed information'
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")

    if args.min_cluster < 2:
        parser.error("--min-cluster must be at least 2")

    base_path = Path(__file__).parent.parent
    corpus_path = base_path / args.corpus

    # Check if corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}", file=sys.stderr)
        print("Run 'python scripts/index_codebase.py' first to create it.", file=sys.stderr)
        sys.exit(1)

    # Load corpus
    if args.verbose or args.output == 'text':
        print(f"Loading corpus from {corpus_path}...")

    try:
        processor = CorticalTextProcessor.load(str(corpus_path))
    except Exception as e:
        print(f"Error loading corpus: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose or args.output == 'text':
        print(f"Loaded {len(processor.documents)} documents")

    # Ensure we have computed necessary features
    if processor.is_stale(processor.COMP_PAGERANK):
        if args.verbose:
            print("Computing PageRank...")
        processor.compute_importance()

    if processor.is_stale(processor.COMP_DOC_CONNECTIONS):
        if args.verbose:
            print("Computing document connections...")
        processor.compute_document_connections()

    # Generate suggestions
    suggestions = suggest_consolidations(
        processor,
        min_overlap=args.threshold,
        min_cluster_size=args.min_cluster,
        min_age_days=args.min_age_days,
        resolution=args.resolution,
        verbose=args.verbose and args.output == 'text'
    )

    # Output results
    if args.output == 'json':
        # Make suggestions JSON-serializable
        json_suggestions = suggestions.copy()
        for cluster in json_suggestions['clusters']:
            cluster['topics'] = [[term, float(score)] for term, score in cluster['topics']]

        print(json.dumps(json_suggestions, indent=2))
    else:
        output = format_suggestions_text(suggestions, verbose=args.verbose)
        print(output)


if __name__ == '__main__':
    main()
