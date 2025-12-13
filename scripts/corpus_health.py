#!/usr/bin/env python3
"""
Corpus Health Dashboard - Statistics and Status

Shows comprehensive statistics about the indexed corpus including:
- Document counts and coverage
- Layer statistics (tokens, bigrams, concepts)
- Computation staleness status
- Concept cluster quality metrics
- Semantic relation statistics

Usage:
    python scripts/corpus_health.py
    python scripts/corpus_health.py --verbose
    python scripts/corpus_health.py --check-staleness
    python scripts/corpus_health.py --concepts  # Show concept clusters
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer


def analyze_corpus_health(
    processor: CorticalTextProcessor,
    check_concepts: bool = False
) -> Dict[str, Any]:
    """
    Analyze corpus health and statistics.

    Args:
        processor: CorticalTextProcessor instance
        check_concepts: Perform detailed concept analysis

    Returns:
        Dict with health metrics and statistics
    """
    stats = {}

    # Document statistics
    stats['document_count'] = len(processor.documents)

    # Calculate document size distribution
    doc_sizes = [len(content) for content in processor.documents.values()]
    stats['total_chars'] = sum(doc_sizes)
    stats['avg_doc_size'] = sum(doc_sizes) / len(doc_sizes) if doc_sizes else 0
    stats['min_doc_size'] = min(doc_sizes) if doc_sizes else 0
    stats['max_doc_size'] = max(doc_sizes) if doc_sizes else 0

    # Document types
    doc_types = defaultdict(int)
    for doc_id in processor.documents:
        if doc_id.endswith('.py'):
            if doc_id.startswith('tests/'):
                doc_types['test'] += 1
            else:
                doc_types['code'] += 1
        elif doc_id.endswith('.md'):
            doc_types['docs'] += 1
        else:
            doc_types['other'] += 1
    stats['doc_types'] = dict(doc_types)

    # Layer statistics
    layer_stats = {}
    for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS,
                       CorticalLayer.CONCEPTS, CorticalLayer.DOCUMENTS]:
        layer = processor.layers[layer_enum]
        layer_stats[layer_enum.name.lower()] = {
            'count': layer.column_count(),
            'avg_connections': 0,
            'max_connections': 0
        }

        # Calculate connection statistics
        if layer.column_count() > 0:
            connection_counts = []
            for col in layer.minicolumns.values():
                conn_count = len(col.lateral_connections)
                connection_counts.append(conn_count)

            layer_stats[layer_enum.name.lower()]['avg_connections'] = (
                sum(connection_counts) / len(connection_counts) if connection_counts else 0
            )
            layer_stats[layer_enum.name.lower()]['max_connections'] = (
                max(connection_counts) if connection_counts else 0
            )

    stats['layers'] = layer_stats

    # Semantic relations
    stats['semantic_relations'] = len(processor.semantic_relations)

    # Relation types
    if processor.semantic_relations:
        relation_types = defaultdict(int)
        for _, rel_type, _, _ in processor.semantic_relations:
            relation_types[rel_type] += 1
        stats['relation_types'] = dict(relation_types)
    else:
        stats['relation_types'] = {}

    # Embeddings
    stats['has_embeddings'] = bool(processor.embeddings)
    stats['embedding_count'] = len(processor.embeddings)

    # Staleness status
    stale = processor.get_stale_computations()
    stats['stale_computations'] = list(stale)
    stats['is_fresh'] = len(stale) == 0

    # Concept cluster analysis
    if check_concepts:
        concept_stats = analyze_concepts(processor)
        stats['concept_analysis'] = concept_stats

    return stats


def analyze_concepts(processor: CorticalTextProcessor) -> Dict[str, Any]:
    """
    Analyze concept cluster quality.

    Args:
        processor: CorticalTextProcessor instance

    Returns:
        Dict with concept quality metrics
    """
    layer2 = processor.layers[CorticalLayer.CONCEPTS]

    concept_sizes = []
    concept_doc_coverage = []
    large_concepts = []

    for concept_col in layer2.minicolumns.values():
        size = concept_col.occurrence_count
        doc_count = len(concept_col.document_ids)

        concept_sizes.append(size)
        concept_doc_coverage.append(doc_count)

        # Track large/important concepts
        if doc_count >= 5:  # Appears in 5+ documents
            large_concepts.append({
                'content': concept_col.content[:60],
                'size': size,
                'documents': doc_count,
                'pagerank': concept_col.pagerank
            })

    # Sort large concepts by PageRank
    large_concepts.sort(key=lambda x: x['pagerank'], reverse=True)

    return {
        'total_concepts': len(concept_sizes),
        'avg_concept_size': sum(concept_sizes) / len(concept_sizes) if concept_sizes else 0,
        'avg_doc_coverage': sum(concept_doc_coverage) / len(concept_doc_coverage) if concept_doc_coverage else 0,
        'max_concept_size': max(concept_sizes) if concept_sizes else 0,
        'max_doc_coverage': max(concept_doc_coverage) if concept_doc_coverage else 0,
        'large_concepts': large_concepts[:10]
    }


def get_health_score(stats: Dict[str, Any]) -> Tuple[str, int]:
    """
    Calculate overall corpus health score.

    Args:
        stats: Statistics from analyze_corpus_health()

    Returns:
        Tuple of (status, score_0_100)
    """
    score = 0

    # Document count (max 20 points)
    doc_count = stats['document_count']
    score += min(20, doc_count // 5)  # 1 point per 5 documents, max 20

    # Layer coverage (max 20 points)
    layer_count = sum(1 for layer in stats['layers'].values() if layer['count'] > 0)
    score += layer_count * 5  # 5 points per populated layer

    # Semantic relations (max 20 points)
    if stats['semantic_relations'] > 0:
        score += min(20, stats['semantic_relations'] // 10)

    # Freshness (max 20 points)
    if stats['is_fresh']:
        score += 20
    else:
        stale_count = len(stats['stale_computations'])
        score += max(0, 20 - (stale_count * 3))

    # Embeddings (max 10 points)
    if stats['has_embeddings']:
        score += 10

    # Connection density (max 10 points)
    token_layer = stats['layers']['tokens']
    if token_layer['avg_connections'] > 0:
        # Good connection density is 5-20 connections per token
        density_score = min(10, int(token_layer['avg_connections']))
        score += density_score

    # Determine status
    if score >= 80:
        status = "Excellent"
    elif score >= 60:
        status = "Good"
    elif score >= 40:
        status = "Fair"
    else:
        status = "Needs Attention"

    return status, score


def display_health(stats: Dict[str, Any], verbose: bool = False, show_concepts: bool = False):
    """
    Display corpus health dashboard.

    Args:
        stats: Statistics from analyze_corpus_health()
        verbose: Show detailed statistics
        show_concepts: Show concept cluster details
    """
    # Header
    print(f"\n{'=' * 70}")
    print("📊 Corpus Health Dashboard")
    print(f"{'=' * 70}\n")

    # Health score
    status, score = get_health_score(stats)
    bar_length = score // 2  # 0-50 character bar
    bar = '█' * bar_length + '░' * (50 - bar_length)
    print(f"Overall Health: {status} ({score}/100)")
    print(f"{bar}\n")

    # Document statistics
    print("📚 Documents:")
    print(f"  Total documents: {stats['document_count']}")
    print(f"  Total size: {stats['total_chars']:,} characters")
    print(f"  Average doc size: {stats['avg_doc_size']:.0f} chars")

    if stats['doc_types']:
        print(f"\n  Document types:")
        for doc_type, count in sorted(stats['doc_types'].items()):
            pct = (count / stats['document_count']) * 100
            print(f"    {doc_type:8s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Layer statistics
    print("🧠 Layer Statistics:")
    for layer_name, layer_info in stats['layers'].items():
        print(f"  {layer_name.upper():12s}: {layer_info['count']:6d} minicolumns", end='')
        if layer_info['avg_connections'] > 0:
            print(f" (avg {layer_info['avg_connections']:.1f} connections)")
        else:
            print()
    print()

    # Semantic relations
    if stats['semantic_relations'] > 0:
        print("🔀 Semantic Relations:")
        print(f"  Total relations: {stats['semantic_relations']}")

        if stats['relation_types']:
            print(f"  Relation types:")
            for rel_type, count in sorted(stats['relation_types'].items(),
                                         key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {rel_type:20s}: {count:4d}")
        print()

    # Embeddings
    if stats['has_embeddings']:
        print("🎯 Embeddings:")
        print(f"  Embedded terms: {stats['embedding_count']}")
        print()

    # Staleness status
    print("⚡ Computation Status:")
    if stats['is_fresh']:
        print("  ✅ All computations are fresh")
    else:
        print(f"  ⚠️  {len(stats['stale_computations'])} stale computations:")
        for comp in stats['stale_computations']:
            print(f"    • {comp}")
    print()

    # Concept analysis
    if show_concepts and 'concept_analysis' in stats:
        concept_stats = stats['concept_analysis']
        print("💡 Concept Cluster Analysis:")
        print(f"  Total concepts: {concept_stats['total_concepts']}")
        print(f"  Average concept size: {concept_stats['avg_concept_size']:.1f}")
        print(f"  Average doc coverage: {concept_stats['avg_doc_coverage']:.1f} documents")
        print(f"  Largest concept: {concept_stats['max_concept_size']} occurrences")
        print()

        if concept_stats['large_concepts']:
            print("  Top concepts (by importance):")
            for i, concept in enumerate(concept_stats['large_concepts'][:10], 1):
                print(f"    {i:2d}. {concept['content']}")
                print(f"        Size: {concept['size']:4d} | Docs: {concept['documents']:3d} | "
                      f"PageRank: {concept['pagerank']:.4f}")
            print()

    # Verbose details
    if verbose:
        print("🔍 Detailed Statistics:")
        print(f"  Document size range: {stats['min_doc_size']} - {stats['max_doc_size']} chars")

        # Connection statistics
        for layer_name, layer_info in stats['layers'].items():
            if layer_info['count'] > 0:
                print(f"  {layer_name.upper()} max connections: {layer_info['max_connections']}")
        print()


def display_recommendations(stats: Dict[str, Any]):
    """
    Display recommendations based on corpus health.

    Args:
        stats: Statistics from analyze_corpus_health()
    """
    recommendations = []

    # Check for stale computations
    if not stats['is_fresh']:
        stale = stats['stale_computations']
        if 'tfidf' in stale or 'pagerank' in stale:
            recommendations.append(
                "⚠️  Core computations are stale. Run: processor.compute_all()"
            )
        elif stale:
            recommendations.append(
                f"⚠️  {len(stale)} computations need updating. Consider running compute_all()"
            )

    # Check for missing embeddings
    if not stats['has_embeddings']:
        recommendations.append(
            "💡 No embeddings found. Consider running: processor.compute_graph_embeddings()"
        )

    # Check for semantic relations
    if stats['semantic_relations'] == 0:
        recommendations.append(
            "💡 No semantic relations extracted. Run: processor.extract_corpus_semantics()"
        )

    # Check for low document count
    if stats['document_count'] < 10:
        recommendations.append(
            "📚 Low document count. Add more documents for better analysis."
        )

    # Check connection density
    token_layer = stats['layers']['tokens']
    if token_layer['avg_connections'] < 3:
        recommendations.append(
            "🔗 Low connection density. Run: processor.compute_bigram_connections()"
        )

    # Display recommendations
    if recommendations:
        print("💡 Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        print()
    else:
        print("✅ Corpus is healthy! No recommendations.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Display corpus health dashboard and statistics',
        epilog="""
Examples:
  %(prog)s                    # Basic health dashboard
  %(prog)s --verbose          # Detailed statistics
  %(prog)s --concepts         # Include concept analysis
  %(prog)s --check-staleness  # Show staleness status
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--corpus', '-c', default='corpus_dev.pkl',
                        help='Corpus file path (default: corpus_dev.pkl)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed statistics')
    parser.add_argument('--concepts', action='store_true',
                        help='Include detailed concept cluster analysis')
    parser.add_argument('--check-staleness', '-s', action='store_true',
                        help='Check computation staleness status')
    parser.add_argument('--recommendations', '-r', action='store_true',
                        help='Show recommendations for improvement')

    args = parser.parse_args()

    # Load corpus
    base_path = Path(__file__).parent.parent
    corpus_path = base_path / args.corpus

    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        print("Run 'python scripts/index_codebase.py' first to create it.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    processor = CorticalTextProcessor.load(str(corpus_path))

    # Analyze health
    print("Analyzing corpus health...")
    stats = analyze_corpus_health(
        processor,
        check_concepts=args.concepts
    )

    # Display dashboard
    display_health(
        stats,
        verbose=args.verbose,
        show_concepts=args.concepts
    )

    # Show recommendations
    if args.recommendations or not stats['is_fresh']:
        display_recommendations(stats)


if __name__ == '__main__':
    main()
