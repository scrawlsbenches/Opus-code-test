#!/usr/bin/env python3
"""
Louvain Resolution Parameter Analysis Script
=============================================

Task #126: Investigate optimal Louvain resolution for sample corpus.

This script analyzes how different resolution values affect:
1. Number of clusters
2. Cluster size distribution
3. Modularity score
4. Semantic coherence within clusters

The resolution parameter affects community detection:
- Lower values (<1.0): Fewer, larger clusters
- Higher values (>1.0): More, smaller clusters

Usage:
    python scripts/analyze_louvain_resolution.py
    python scripts/analyze_louvain_resolution.py --resolutions 0.5 1.0 2.0
    python scripts/analyze_louvain_resolution.py --verbose
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortical import CorticalTextProcessor, CorticalLayer


def compute_modularity(processor: CorticalTextProcessor) -> float:
    """
    Compute the modularity Q of the current clustering.

    Modularity measures the density of connections within clusters
    compared to connections between clusters.

    Q = (1/2m) * Σ [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

    where:
    - m = total edge weight
    - A_ij = edge weight between i and j
    - k_i = degree of node i
    - δ(c_i, c_j) = 1 if nodes i and j are in the same community, 0 otherwise

    Returns:
        Modularity score between -1 and 1 (higher is better)
        - Q > 0.3: Good community structure
        - Q > 0.5: Strong community structure
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]
    layer2 = processor.layers[CorticalLayer.CONCEPTS]

    if layer0.column_count() == 0 or layer2.column_count() == 0:
        return 0.0

    # Build token -> cluster mapping
    token_to_cluster: Dict[str, str] = {}
    for cluster_col in layer2.minicolumns.values():
        cluster_id = cluster_col.content
        for token_id in cluster_col.feedforward_connections:
            token_col = layer0.get_by_id(token_id)
            if token_col:
                token_to_cluster[token_col.content] = cluster_id

    # Compute total edge weight m
    total_weight = 0.0
    for col in layer0.minicolumns.values():
        for _, weight in col.lateral_connections.items():
            total_weight += weight

    m = total_weight / 2.0  # Each edge counted twice

    if m == 0:
        return 0.0

    # Compute node degrees k
    degrees: Dict[str, float] = {}
    for content, col in layer0.minicolumns.items():
        degrees[content] = sum(col.lateral_connections.values())

    # Compute modularity Q
    q = 0.0
    for content, col in layer0.minicolumns.items():
        c_i = token_to_cluster.get(content)
        if c_i is None:
            continue

        k_i = degrees.get(content, 0.0)

        for neighbor_id, weight in col.lateral_connections.items():
            neighbor_col = layer0.get_by_id(neighbor_id)
            if neighbor_col is None:
                continue

            neighbor_content = neighbor_col.content
            c_j = token_to_cluster.get(neighbor_content)
            if c_j is None:
                continue

            k_j = degrees.get(neighbor_content, 0.0)

            # δ(c_i, c_j) - same cluster indicator
            if c_i == c_j:
                # A_ij - k_i*k_j/(2m)
                q += weight - (k_i * k_j) / (2 * m)

    return q / (2 * m)


def compute_cluster_balance(cluster_sizes: List[int]) -> float:
    """
    Compute Gini coefficient for cluster size balance.

    Returns:
        Gini coefficient (0 = perfectly balanced, 1 = all in one cluster)
    """
    if not cluster_sizes or len(cluster_sizes) == 1:
        return 1.0

    sorted_sizes = sorted(cluster_sizes)
    n = len(sorted_sizes)
    total = sum(sorted_sizes)

    if total == 0:
        return 1.0

    # Standard Gini calculation using the formula:
    # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    weighted_sum = sum((i + 1) * size for i, size in enumerate(sorted_sizes))
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

    return max(0, min(1, gini))


def evaluate_semantic_coherence(processor: CorticalTextProcessor, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Evaluate semantic coherence of top clusters.

    For each of the top clusters, check if terms are semantically related
    by looking at their lateral connections.

    Returns:
        List of cluster evaluations with coherence scores
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]
    layer2 = processor.layers[CorticalLayer.CONCEPTS]

    evaluations = []

    # Get clusters sorted by size
    clusters = []
    for col in layer2.minicolumns.values():
        tokens = []
        for token_id in col.feedforward_connections:
            token_col = layer0.get_by_id(token_id)
            if token_col:
                tokens.append(token_col.content)
        clusters.append((col.content, tokens))

    clusters.sort(key=lambda x: len(x[1]), reverse=True)

    # Evaluate top N clusters
    for cluster_id, tokens in clusters[:top_n]:
        if len(tokens) < 2:
            continue

        # Compute intra-cluster connectivity
        intra_connections = 0
        possible_connections = 0

        token_set = set(tokens)
        for token in tokens:
            col = layer0.get_minicolumn(token)
            if col is None:
                continue

            for neighbor_id in col.lateral_connections:
                neighbor_col = layer0.get_by_id(neighbor_id)
                if neighbor_col and neighbor_col.content in token_set:
                    intra_connections += 1

            possible_connections += len(tokens) - 1

        coherence = intra_connections / max(possible_connections, 1)

        # Sample terms for display
        sample_terms = sorted(tokens, key=lambda t: layer0.get_minicolumn(t).pagerank if layer0.get_minicolumn(t) else 0, reverse=True)[:8]

        evaluations.append({
            'cluster_id': cluster_id,
            'size': len(tokens),
            'coherence': coherence,
            'sample_terms': sample_terms
        })

    return evaluations


def load_corpus(processor: CorticalTextProcessor, samples_dir: str = "samples") -> int:
    """Load all sample documents into the processor."""
    loaded = 0

    if not os.path.isdir(samples_dir):
        print(f"Samples directory not found: {samples_dir}")
        return 0

    for filename in sorted(os.listdir(samples_dir)):
        if not filename.endswith(('.txt', '.py')):
            continue

        filepath = os.path.join(samples_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            doc_id = os.path.splitext(filename)[0]
            processor.process_document(doc_id, content)
            loaded += 1
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return loaded


def analyze_resolution(
    resolution: float,
    samples_dir: str = "samples",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze clustering quality at a specific resolution value.

    Returns:
        Dictionary with analysis results
    """
    # Create fresh processor for each resolution
    processor = CorticalTextProcessor()

    # Load corpus
    num_docs = load_corpus(processor, samples_dir)
    if num_docs == 0:
        return {'error': 'No documents loaded'}

    # Build network with base computations (without default clustering)
    processor.compute_all(build_concepts=False, verbose=False)

    # Time the clustering with specified resolution
    start_time = time.perf_counter()

    # Build clusters with specified resolution
    clusters = processor.build_concept_clusters(
        clustering_method='louvain',
        resolution=resolution,
        verbose=False
    )

    # Compute concept connections for proper evaluation
    processor.compute_concept_connections(verbose=False)

    cluster_time = time.perf_counter() - start_time

    # Gather metrics
    layer0 = processor.layers[CorticalLayer.TOKENS]
    layer2 = processor.layers[CorticalLayer.CONCEPTS]

    total_tokens = layer0.column_count()
    num_clusters = layer2.column_count()

    # Cluster sizes
    cluster_sizes = []
    for col in layer2.minicolumns.values():
        cluster_sizes.append(len(col.feedforward_connections))

    if cluster_sizes:
        max_cluster_size = max(cluster_sizes)
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        min_cluster_size = min(cluster_sizes)
        max_cluster_pct = max_cluster_size / total_tokens * 100 if total_tokens > 0 else 0
    else:
        max_cluster_size = avg_cluster_size = min_cluster_size = max_cluster_pct = 0

    # Compute modularity
    modularity = compute_modularity(processor)

    # Compute balance
    balance = compute_cluster_balance(cluster_sizes)

    # Evaluate semantic coherence
    coherence_eval = evaluate_semantic_coherence(processor, top_n=5)
    avg_coherence = sum(c['coherence'] for c in coherence_eval) / len(coherence_eval) if coherence_eval else 0

    result = {
        'resolution': resolution,
        'num_documents': num_docs,
        'total_tokens': total_tokens,
        'num_clusters': num_clusters,
        'max_cluster_size': max_cluster_size,
        'max_cluster_pct': max_cluster_pct,
        'avg_cluster_size': avg_cluster_size,
        'min_cluster_size': min_cluster_size,
        'modularity': modularity,
        'balance_gini': balance,
        'avg_coherence': avg_coherence,
        'cluster_time_sec': cluster_time,
        'coherence_details': coherence_eval if verbose else None
    }

    return result


def print_results_table(results: List[Dict[str, Any]]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("LOUVAIN RESOLUTION ANALYSIS RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Resolution':>10} | {'Clusters':>8} | {'Max %':>8} | {'Avg Size':>8} | {'Modularity':>10} | {'Balance':>8} | {'Coherence':>10}")
    print("-" * 100)

    for r in results:
        if 'error' in r:
            print(f"{r.get('resolution', 'N/A'):>10} | ERROR: {r['error']}")
            continue

        print(f"{r['resolution']:>10.2f} | {r['num_clusters']:>8} | {r['max_cluster_pct']:>7.1f}% | {r['avg_cluster_size']:>8.1f} | {r['modularity']:>10.4f} | {r['balance_gini']:>8.3f} | {r['avg_coherence']:>10.3f}")

    print("-" * 100)


def print_detailed_analysis(results: List[Dict[str, Any]]):
    """Print detailed analysis and recommendations."""
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)

    # Find optimal by modularity
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print("No valid results to analyze.")
        return

    # Best by modularity (primary metric)
    best_modularity = max(valid_results, key=lambda x: x['modularity'])

    # Best by balance (secondary metric)
    best_balance = min(valid_results, key=lambda x: x['balance_gini'])

    # Best by coherence
    best_coherence = max(valid_results, key=lambda x: x['avg_coherence'])

    print("\nMETRIC INTERPRETATION:")
    print("-" * 50)
    print("  Modularity: Higher is better (>0.3 good, >0.5 strong)")
    print("  Balance (Gini): Lower is better (0=even, 1=skewed)")
    print("  Coherence: Higher is better (intra-cluster connectivity)")

    print("\nBEST RESULTS BY METRIC:")
    print("-" * 50)
    print(f"  Best Modularity:  res={best_modularity['resolution']:.2f} (Q={best_modularity['modularity']:.4f}, {best_modularity['num_clusters']} clusters)")
    print(f"  Best Balance:     res={best_balance['resolution']:.2f} (Gini={best_balance['balance_gini']:.3f}, {best_balance['num_clusters']} clusters)")
    print(f"  Best Coherence:   res={best_coherence['resolution']:.2f} (C={best_coherence['avg_coherence']:.3f}, {best_coherence['num_clusters']} clusters)")

    # Compute composite score (weighted)
    # Normalize each metric to 0-1 scale
    mod_max = max(r['modularity'] for r in valid_results)
    mod_min = min(r['modularity'] for r in valid_results)
    mod_range = mod_max - mod_min if mod_max > mod_min else 1

    bal_max = max(r['balance_gini'] for r in valid_results)
    bal_min = min(r['balance_gini'] for r in valid_results)
    bal_range = bal_max - bal_min if bal_max > bal_min else 1

    coh_max = max(r['avg_coherence'] for r in valid_results)
    coh_min = min(r['avg_coherence'] for r in valid_results)
    coh_range = coh_max - coh_min if coh_max > coh_min else 1

    for r in valid_results:
        # Normalize (invert balance so lower is better)
        norm_mod = (r['modularity'] - mod_min) / mod_range
        norm_bal = 1 - (r['balance_gini'] - bal_min) / bal_range
        norm_coh = (r['avg_coherence'] - coh_min) / coh_range

        # Mega-cluster penalty: heavily penalize if largest cluster > 30% of tokens
        max_cluster_penalty = 0.0
        if r['max_cluster_pct'] > 50:
            max_cluster_penalty = 0.5  # Severe penalty
        elif r['max_cluster_pct'] > 30:
            max_cluster_penalty = 0.3  # Moderate penalty
        elif r['max_cluster_pct'] > 20:
            max_cluster_penalty = 0.1  # Light penalty

        # Weighted composite (modularity important but balance matters for usability)
        # Penalize mega-clusters severely
        r['composite_score'] = 0.4 * norm_mod + 0.3 * norm_bal + 0.2 * norm_coh + 0.1 * (1 - max_cluster_penalty * 2)

    best_composite = max(valid_results, key=lambda x: x['composite_score'])

    print("\nCOMPOSITE SCORE (40% modularity + 30% balance + 20% coherence + 10% cluster size penalty):")
    print("-" * 50)
    for r in sorted(valid_results, key=lambda x: x['composite_score'], reverse=True):
        marker = " <-- RECOMMENDED" if r == best_composite else ""
        print(f"  res={r['resolution']:.2f}: score={r['composite_score']:.3f}{marker}")

    # Additional insights
    print("\nINSIGHTS:")
    print("-" * 50)

    # Check if default (1.0) is optimal
    default_result = next((r for r in valid_results if r['resolution'] == 1.0), None)
    if default_result:
        if default_result == best_composite:
            print("  * Default resolution (1.0) IS optimal for this corpus")
        else:
            diff_pct = (best_composite['composite_score'] - default_result['composite_score']) / default_result['composite_score'] * 100
            print(f"  * Default resolution (1.0) is NOT optimal")
            print(f"  * Resolution {best_composite['resolution']:.2f} is {diff_pct:.1f}% better")

    # Check for over-segmentation
    fine_results = [r for r in valid_results if r['resolution'] >= 2.0]
    coarse_results = [r for r in valid_results if r['resolution'] <= 0.75]

    if fine_results:
        avg_fine_clusters = sum(r['num_clusters'] for r in fine_results) / len(fine_results)
        print(f"  * High resolution (>=2.0): Average {avg_fine_clusters:.0f} clusters")

    if coarse_results:
        avg_coarse_clusters = sum(r['num_clusters'] for r in coarse_results) / len(coarse_results)
        print(f"  * Low resolution (<=0.75): Average {avg_coarse_clusters:.0f} clusters")

    # Recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print(f"\n  Optimal resolution for this corpus: {best_composite['resolution']:.2f}")
    print(f"  - Produces {best_composite['num_clusters']} clusters")
    print(f"  - Modularity: {best_composite['modularity']:.4f}")
    print(f"  - Largest cluster: {best_composite['max_cluster_pct']:.1f}% of tokens")

    if best_composite['resolution'] != 1.0:
        print(f"\n  Consider updating the default resolution from 1.0 to {best_composite['resolution']:.2f}")
        print(f"  in cortical/analysis.py:cluster_by_louvain() and cortical/processor.py:build_concept_clusters()")


def print_cluster_samples(results: List[Dict[str, Any]]):
    """Print sample clusters at different resolutions."""
    print("\n" + "=" * 100)
    print("SAMPLE CLUSTER CONTENTS")
    print("=" * 100)

    for r in results:
        if 'error' in r or not r.get('coherence_details'):
            continue

        print(f"\n--- Resolution {r['resolution']:.2f} ({r['num_clusters']} clusters) ---")

        for i, cluster in enumerate(r['coherence_details'][:3], 1):
            terms = ', '.join(cluster['sample_terms'][:6])
            print(f"  Cluster #{i} ({cluster['size']} tokens, coherence={cluster['coherence']:.3f}): {terms}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Louvain resolution parameter effects on clustering"
    )
    parser.add_argument(
        '--resolutions', '-r',
        type=float,
        nargs='+',
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0],
        help='Resolution values to test (default: 0.5-3.0 range)'
    )
    parser.add_argument(
        '--samples-dir', '-s',
        type=str,
        default='samples',
        help='Directory containing sample documents (default: samples)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed cluster contents'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Write results to file (markdown format)'
    )

    args = parser.parse_args()

    print("Louvain Resolution Parameter Analysis")
    print("=====================================\n")
    print(f"Testing resolutions: {args.resolutions}")
    print(f"Samples directory: {args.samples_dir}\n")

    results = []

    for resolution in sorted(args.resolutions):
        print(f"Analyzing resolution={resolution:.2f}...", end=' ', flush=True)
        result = analyze_resolution(resolution, args.samples_dir, verbose=args.verbose)
        results.append(result)

        if 'error' not in result:
            print(f"OK ({result['num_clusters']} clusters, Q={result['modularity']:.4f})")
        else:
            print(f"ERROR: {result['error']}")

    # Print results
    print_results_table(results)
    print_detailed_analysis(results)

    if args.verbose:
        print_cluster_samples(results)

    # Write to file if requested
    if args.output:
        write_markdown_report(results, args.output)
        print(f"\nReport written to: {args.output}")


def write_markdown_report(results: List[Dict[str, Any]], filepath: str):
    """Write analysis results to a markdown file."""
    valid_results = [r for r in results if 'error' not in r]

    with open(filepath, 'w') as f:
        f.write("# Louvain Resolution Parameter Analysis\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n\n")

        if valid_results:
            f.write(f"**Corpus:** {valid_results[0]['num_documents']} documents, {valid_results[0]['total_tokens']} tokens\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Resolution | Clusters | Max % | Avg Size | Modularity | Balance | Coherence |\n")
        f.write("|------------|----------|-------|----------|------------|---------|----------|\n")

        for r in results:
            if 'error' in r:
                f.write(f"| {r.get('resolution', 'N/A')} | ERROR | - | - | - | - | - |\n")
            else:
                f.write(f"| {r['resolution']:.2f} | {r['num_clusters']} | {r['max_cluster_pct']:.1f}% | {r['avg_cluster_size']:.1f} | {r['modularity']:.4f} | {r['balance_gini']:.3f} | {r['avg_coherence']:.3f} |\n")

        f.write("\n## Metric Interpretation\n\n")
        f.write("- **Modularity**: Higher is better (>0.3 good, >0.5 strong community structure)\n")
        f.write("- **Balance (Gini)**: Lower is better (0=even distribution, 1=all in one cluster)\n")
        f.write("- **Coherence**: Higher is better (measures intra-cluster connectivity)\n\n")

        # Compute composite scores
        if valid_results:
            mod_max = max(r['modularity'] for r in valid_results)
            mod_min = min(r['modularity'] for r in valid_results)
            mod_range = mod_max - mod_min if mod_max > mod_min else 1

            bal_max = max(r['balance_gini'] for r in valid_results)
            bal_min = min(r['balance_gini'] for r in valid_results)
            bal_range = bal_max - bal_min if bal_max > bal_min else 1

            coh_max = max(r['avg_coherence'] for r in valid_results)
            coh_min = min(r['avg_coherence'] for r in valid_results)
            coh_range = coh_max - coh_min if coh_max > coh_min else 1

            for r in valid_results:
                norm_mod = (r['modularity'] - mod_min) / mod_range
                norm_bal = 1 - (r['balance_gini'] - bal_min) / bal_range
                norm_coh = (r['avg_coherence'] - coh_min) / coh_range
                r['composite_score'] = 0.5 * norm_mod + 0.3 * norm_bal + 0.2 * norm_coh

            best = max(valid_results, key=lambda x: x['composite_score'])

            f.write("## Recommendation\n\n")
            f.write(f"**Optimal resolution: {best['resolution']:.2f}**\n\n")
            f.write(f"- Produces {best['num_clusters']} clusters\n")
            f.write(f"- Modularity: {best['modularity']:.4f}\n")
            f.write(f"- Largest cluster: {best['max_cluster_pct']:.1f}% of tokens\n")
            f.write(f"- Composite score: {best['composite_score']:.3f}\n")


if __name__ == '__main__':
    main()
