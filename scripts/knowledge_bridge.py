"""
Knowledge Bridge Pipeline
=========================

Takes JSON input from earlier pipeline stages (knowledge_analysis.py or
world_model_analysis.py), identifies knowledge gaps, and suggests bridges
between disconnected concepts.

This script can be used standalone or as part of a pipeline:

  python scripts/world_model_analysis.py --json | python scripts/knowledge_bridge.py
  python scripts/knowledge_bridge.py --input results.json --min-gap-distance 2
"""

import json
import sys
import argparse
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict


def detect_input_type(data: Dict[str, Any]) -> str:
    """
    Auto-detect the type of input JSON.

    Args:
        data: Parsed JSON data

    Returns:
        "world_model" or "knowledge_analysis" or "unknown"
    """
    # Check for world_model_analysis.py format
    if "suggestions" in data and "bridges" in data and "concepts" in data:
        return "world_model"

    # Check for knowledge_analysis.py format
    if "patterns" in data and "clusters" in data and "strength_scores" in data:
        return "knowledge_analysis"

    return "unknown"


def extract_clusters_world_model(data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Extract clusters from world_model_analysis.py output.

    Args:
        data: World model analysis JSON

    Returns:
        Dictionary mapping cluster_id -> set of terms
    """
    clusters = {}

    # Group concepts by their domains as pseudo-clusters
    for concept in data.get("concepts", []):
        for domain in concept.get("domains", []):
            cluster_id = f"domain_{domain}"
            if cluster_id not in clusters:
                clusters[cluster_id] = set()
            clusters[cluster_id].add(concept["term"])

    return clusters


def extract_clusters_knowledge_analysis(data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Extract clusters from knowledge_analysis.py output.

    Args:
        data: Knowledge analysis JSON

    Returns:
        Dictionary mapping cluster_id -> set of terms
    """
    clusters = {}

    for cluster in data.get("clusters", []):
        cluster_id = cluster.get("id", f"cluster_{len(clusters)}")
        terms = set(cluster.get("terms", []))
        clusters[cluster_id] = terms

    return clusters


def extract_connections(data: Dict[str, Any], input_type: str) -> Dict[str, Dict[str, float]]:
    """
    Extract connection strengths between terms.

    Args:
        data: Input JSON data
        input_type: Type of input ("world_model" or "knowledge_analysis")

    Returns:
        Dictionary mapping term -> {connected_term: weight}
    """
    connections = defaultdict(dict)

    if input_type == "world_model":
        # Extract from network structure
        network = data.get("network", {})
        for term, neighbors in network.items():
            for neighbor in neighbors:
                neighbor_term = neighbor.get("term")
                weight = neighbor.get("weight", 1.0)
                if neighbor_term:
                    connections[term][neighbor_term] = weight

    elif input_type == "knowledge_analysis":
        # Extract from strength_scores
        strength_scores = data.get("strength_scores", {})
        for key, score in strength_scores.items():
            if "-" in key:
                parts = key.split("-", 1)
                if len(parts) == 2:
                    term1, term2 = parts
                    connections[term1][term2] = score
                    connections[term2][term1] = score

    return dict(connections)


def find_cluster_gaps(
    clusters: Dict[str, Set[str]],
    connections: Dict[str, Dict[str, float]],
    min_gap_distance: int = 2
) -> List[Dict[str, Any]]:
    """
    Find gaps between clusters that have no direct connections.

    Args:
        clusters: Dictionary of cluster_id -> set of terms
        connections: Connection strengths between terms
        min_gap_distance: Minimum distance to consider a gap

    Returns:
        List of gap dictionaries with metadata
    """
    gaps = []
    cluster_list = list(clusters.items())

    for i, (cluster1_id, cluster1_terms) in enumerate(cluster_list):
        for cluster2_id, cluster2_terms in cluster_list[i+1:]:
            # Calculate minimum distance between clusters
            min_distance = float('inf')
            connecting_path = []

            for term1 in cluster1_terms:
                for term2 in cluster2_terms:
                    # Check if directly connected
                    if term1 in connections and term2 in connections[term1]:
                        min_distance = 1
                        connecting_path = [term1, term2]
                        break

                    # Check for 2-hop paths via intermediate terms
                    if term1 in connections:
                        for intermediate in connections[term1]:
                            if intermediate in connections and term2 in connections[intermediate]:
                                if min_distance > 2:
                                    min_distance = 2
                                    connecting_path = [term1, intermediate, term2]
                                break

                if min_distance == 1:
                    break

            # If no path found, estimate distance
            if min_distance == float('inf'):
                min_distance = 3
                connecting_path = []

            # Only report gaps that meet minimum distance
            if min_distance >= min_gap_distance:
                # Find potential bridges
                potential_bridges = find_potential_bridges(
                    cluster1_terms, cluster2_terms, connections
                )

                gaps.append({
                    "between": [cluster1_id, cluster2_id],
                    "distance": min_distance,
                    "potential_bridges": potential_bridges[:5],  # Top 5 bridges
                    "cluster1_size": len(cluster1_terms),
                    "cluster2_size": len(cluster2_terms),
                })

    return gaps


def find_potential_bridges(
    cluster1_terms: Set[str],
    cluster2_terms: Set[str],
    connections: Dict[str, Dict[str, float]]
) -> List[str]:
    """
    Find terms that could bridge two clusters.

    Args:
        cluster1_terms: Terms in first cluster
        cluster2_terms: Terms in second cluster
        connections: Connection strengths

    Returns:
        List of potential bridge terms, sorted by bridging score
    """
    bridge_scores = defaultdict(float)

    # Find terms connected to both clusters
    for term, neighbors in connections.items():
        if term in cluster1_terms or term in cluster2_terms:
            continue

        # Count connections to each cluster
        conn_to_c1 = sum(1 for n in neighbors if n in cluster1_terms)
        conn_to_c2 = sum(1 for n in neighbors if n in cluster2_terms)

        # Bridge score = product of connections (favors balanced bridges)
        if conn_to_c1 > 0 and conn_to_c2 > 0:
            bridge_scores[term] = conn_to_c1 * conn_to_c2

    # Sort by score
    sorted_bridges = sorted(
        bridge_scores.items(),
        key=lambda x: -x[1]
    )

    return [term for term, score in sorted_bridges]


def suggest_bridges(
    data: Dict[str, Any],
    input_type: str,
    connections: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    Suggest bridges between important but disconnected concepts.

    Args:
        data: Input JSON data
        input_type: Type of input
        connections: Connection strengths

    Returns:
        List of bridge suggestions
    """
    suggestions = []

    # Get important concepts (high PageRank or centrality)
    important_terms = set()

    if input_type == "world_model":
        # Use top concepts by PageRank
        for concept in data.get("concepts", [])[:30]:
            important_terms.add(concept["term"])

    # Check existing suggestions in input
    existing_suggestions = data.get("suggestions", [])
    for sugg in existing_suggestions[:20]:
        term1 = sugg.get("term1")
        term2 = sugg.get("term2")
        shared = sugg.get("shared_neighbors", 0)

        if not term1 or not term2:
            continue

        # Find best bridge path
        bridge_path = find_best_bridge_path(term1, term2, connections)

        # Estimate strength potential based on shared neighbors
        strength_potential = min(shared / 10.0, 0.9)

        suggestions.append({
            "concept1": term1,
            "concept2": term2,
            "via": bridge_path,
            "strength_potential": round(strength_potential, 2),
            "shared_neighbors": shared,
        })

    # Sort by strength potential
    suggestions.sort(key=lambda x: -x["strength_potential"])

    return suggestions


def find_best_bridge_path(
    term1: str,
    term2: str,
    connections: Dict[str, Dict[str, float]]
) -> Optional[str]:
    """
    Find the best intermediate term to bridge two concepts.

    Args:
        term1: First term
        term2: Second term
        connections: Connection strengths

    Returns:
        Best bridge term, or None if no bridge found
    """
    if term1 not in connections or term2 not in connections:
        return None

    # Find common neighbors
    neighbors1 = set(connections[term1].keys())
    neighbors2 = set(connections[term2].keys())
    common = neighbors1 & neighbors2

    if not common:
        return None

    # Select the best common neighbor (highest combined weight)
    best_bridge = None
    best_score = 0.0

    for bridge in common:
        score = connections[term1][bridge] + connections[term2][bridge]
        if score > best_score:
            best_score = score
            best_bridge = bridge

    return best_bridge


def identify_weak_links(
    connections: Dict[str, Dict[str, float]],
    threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Identify weak connections that could be strengthened.

    Args:
        connections: Connection strengths
        threshold: Threshold below which a connection is considered weak

    Returns:
        List of weak link dictionaries
    """
    weak_links = []
    seen = set()

    for term1, neighbors in connections.items():
        for term2, weight in neighbors.items():
            # Avoid duplicates (A->B and B->A)
            pair = tuple(sorted([term1, term2]))
            if pair in seen:
                continue
            seen.add(pair)

            if weight < threshold:
                # Suggest action based on weight
                if weight < 0.05:
                    action = "add document explicitly covering both concepts"
                elif weight < 0.1:
                    action = "add co-occurrence examples"
                else:
                    action = "strengthen with more context"

                weak_links.append({
                    "from": term1,
                    "to": term2,
                    "current_strength": round(weight, 3),
                    "suggested_action": action,
                })

    # Sort by current strength (weakest first)
    weak_links.sort(key=lambda x: x["current_strength"])

    return weak_links


def find_synthesis_opportunities(
    clusters: Dict[str, Set[str]],
    data: Dict[str, Any],
    input_type: str
) -> List[Dict[str, Any]]:
    """
    Find opportunities to synthesize concepts across domains.

    Args:
        clusters: Dictionary of clusters
        data: Input JSON data
        input_type: Type of input

    Returns:
        List of synthesis opportunities
    """
    opportunities = []

    if input_type == "world_model":
        # Look for cross-domain concepts
        domain_concepts = defaultdict(set)

        for concept in data.get("concepts", []):
            for domain in concept.get("domains", []):
                domain_concepts[domain].add(concept["term"])

        # Find overlapping concepts between domains
        domains = list(domain_concepts.keys())
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                overlap = domain_concepts[domain1] & domain_concepts[domain2]

                if len(overlap) >= 3:
                    opportunities.append({
                        "topics": [domain1, domain2],
                        "synthesis_type": "cross_domain",
                        "description": f"Synthesize {len(overlap)} shared concepts between {domain1} and {domain2}",
                        "shared_concepts": list(overlap)[:5],
                        "concept_count": len(overlap),
                    })

    # Look for concept clusters that could be merged
    cluster_list = list(clusters.items())
    for i, (cluster1_id, cluster1_terms) in enumerate(cluster_list):
        for cluster2_id, cluster2_terms in cluster_list[i+1:]:
            overlap = cluster1_terms & cluster2_terms

            if len(overlap) >= 2:
                opportunities.append({
                    "topics": [cluster1_id, cluster2_id],
                    "synthesis_type": "cluster_merge",
                    "description": f"Merge overlapping clusters {cluster1_id} and {cluster2_id}",
                    "shared_concepts": list(overlap)[:5],
                    "concept_count": len(overlap),
                })

    # Sort by concept count (most overlap first)
    opportunities.sort(key=lambda x: -x["concept_count"])

    return opportunities


def generate_summary(
    gaps: List[Dict[str, Any]],
    bridge_suggestions: List[Dict[str, Any]],
    weak_links: List[Dict[str, Any]],
    synthesis_opportunities: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a summary of the analysis.

    Args:
        gaps: List of detected gaps
        bridge_suggestions: List of bridge suggestions
        weak_links: List of weak links
        synthesis_opportunities: List of synthesis opportunities

    Returns:
        Summary dictionary
    """
    # Find priority bridges (high potential, bridge important gaps)
    priority_bridges = []
    for suggestion in bridge_suggestions[:10]:
        if suggestion["strength_potential"] >= 0.5:
            pair = f"{suggestion['concept1']}-{suggestion['concept2']}"
            priority_bridges.append(pair)

    # Count bridgeable gaps
    bridgeable = sum(1 for gap in gaps if len(gap["potential_bridges"]) > 0)

    return {
        "total_gaps": len(gaps),
        "bridgeable": bridgeable,
        "unbridgeable": len(gaps) - bridgeable,
        "priority_bridges": priority_bridges[:5],
        "weak_link_count": len(weak_links),
        "synthesis_opportunity_count": len(synthesis_opportunities),
    }


def process_knowledge_bridge(
    data: Dict[str, Any],
    min_gap_distance: int = 2,
    weak_link_threshold: float = 0.2,
    max_gaps: int = 50,
    max_bridges: int = 50,
    max_weak_links: int = 30,
    max_synthesis: int = 20
) -> Dict[str, Any]:
    """
    Main processing function for knowledge bridge analysis.

    Args:
        data: Input JSON data
        min_gap_distance: Minimum distance to consider a gap
        weak_link_threshold: Threshold for weak connections
        max_gaps: Maximum number of gaps to return
        max_bridges: Maximum number of bridge suggestions
        max_weak_links: Maximum number of weak links to return
        max_synthesis: Maximum number of synthesis opportunities

    Returns:
        Results dictionary with gaps, bridges, weak links, and synthesis opportunities
    """
    # Detect input type
    input_type = detect_input_type(data)

    if input_type == "unknown":
        return {
            "error": "Unknown input format. Expected world_model or knowledge_analysis JSON."
        }

    # Extract clusters
    if input_type == "world_model":
        clusters = extract_clusters_world_model(data)
    else:
        clusters = extract_clusters_knowledge_analysis(data)

    # Extract connections
    connections = extract_connections(data, input_type)

    # Find gaps
    gaps = find_cluster_gaps(clusters, connections, min_gap_distance)[:max_gaps]

    # Suggest bridges
    bridge_suggestions = suggest_bridges(data, input_type, connections)[:max_bridges]

    # Identify weak links
    weak_links = identify_weak_links(connections, weak_link_threshold)[:max_weak_links]

    # Find synthesis opportunities
    synthesis_opportunities = find_synthesis_opportunities(clusters, data, input_type)[:max_synthesis]

    # Generate summary
    summary = generate_summary(gaps, bridge_suggestions, weak_links, synthesis_opportunities)

    return {
        "input_type": input_type,
        "gaps": gaps,
        "bridge_suggestions": bridge_suggestions,
        "weak_links": weak_links,
        "synthesis_opportunities": synthesis_opportunities,
        "summary": summary,
    }


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Knowledge Bridge Analysis - Find gaps and suggest bridges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read from stdin (pipeline mode)
  python scripts/world_model_analysis.py --json | python scripts/knowledge_bridge.py

  # Read from file
  python scripts/knowledge_bridge.py --input results.json

  # Adjust gap distance threshold
  python scripts/knowledge_bridge.py --input results.json --min-gap-distance 3

  # Focus on weak links only
  python scripts/knowledge_bridge.py --input results.json --max-gaps 0 --max-weak-links 100

  # Prioritize bridge suggestions
  python scripts/knowledge_bridge.py --input results.json --bridge-priority novelty

  # Filter by domains
  python scripts/knowledge_bridge.py --input results.json --focus-domains cognitive_science,world_models

Input Format:
  Accepts JSON from world_model_analysis.py or knowledge_analysis.py.
  Auto-detects format based on structure.
  Also supports ThoughtChain format for pipeline context preservation.

Output Format:
  JSON with gaps, bridge_suggestions, weak_links, synthesis_opportunities, and summary.
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON file (default: read from stdin)"
    )

    parser.add_argument(
        "--min-gap-distance", "-d",
        type=int,
        default=2,
        help="Minimum distance to consider a gap (default: 2)"
    )

    parser.add_argument(
        "--weak-link-threshold", "-w",
        type=float,
        default=0.2,
        help="Threshold for weak connections (default: 0.2)"
    )

    parser.add_argument(
        "--max-gaps",
        type=int,
        default=50,
        help="Maximum number of gaps to return (default: 50)"
    )

    parser.add_argument(
        "--max-bridges",
        type=int,
        default=50,
        help="Maximum number of bridge suggestions (default: 50)"
    )

    parser.add_argument(
        "--max-weak-links",
        type=int,
        default=30,
        help="Maximum number of weak links (default: 30)"
    )

    parser.add_argument(
        "--max-synthesis",
        type=int,
        default=20,
        help="Maximum number of synthesis opportunities (default: 20)"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation"
    )

    # New configurable parameters
    parser.add_argument(
        "--bridge-priority",
        choices=["strength", "novelty", "coverage"],
        default="strength",
        help="How to prioritize bridge suggestions (default: strength)"
    )

    parser.add_argument(
        "--focus-domains",
        help="Comma-separated list of domains to focus on"
    )

    parser.add_argument(
        "--min-synthesis-overlap",
        type=int,
        default=3,
        help="Minimum shared concepts for synthesis opportunity (default: 3)"
    )

    parser.add_argument(
        "--include-actionable",
        action="store_true",
        help="Include actionable recommendations for each gap"
    )

    parser.add_argument(
        "--preserve-chain",
        action="store_true",
        help="Preserve thought chain metadata in output"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output to stderr"
    )

    args = parser.parse_args()

    # Import thought chain utilities
    try:
        from scripts.thought_chain import ThoughtChain, is_chain_format, extract_from_chain
        chain_available = True
    except ImportError:
        chain_available = False
        def is_chain_format(data): return False

    # Read input
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        raw_data = json.load(sys.stdin)

    # Check for thought chain format
    chain = None
    if chain_available and is_chain_format(raw_data):
        chain = ThoughtChain.from_dict(raw_data)
        # Try to extract previous stage results
        data = extract_from_chain(raw_data, 'knowledge_analysis')
        if data == raw_data:
            data = extract_from_chain(raw_data, 'question_connection')
        if data == raw_data:
            data = extract_from_chain(raw_data, 'world_model_analysis')
        if data == raw_data:
            data = raw_data.get('results', {}).get('knowledge_analysis', raw_data)
        if args.verbose:
            print(f"[knowledge_bridge] Loaded from chain, iteration {chain.iteration}", file=sys.stderr)
    else:
        data = raw_data

    # Process
    results = process_knowledge_bridge(
        data,
        min_gap_distance=args.min_gap_distance,
        weak_link_threshold=args.weak_link_threshold,
        max_gaps=args.max_gaps,
        max_bridges=args.max_bridges,
        max_weak_links=args.max_weak_links,
        max_synthesis=args.max_synthesis,
    )

    # Add parameters to output for traceability
    results['parameters'] = {
        'min_gap_distance': args.min_gap_distance,
        'weak_link_threshold': args.weak_link_threshold,
        'max_gaps': args.max_gaps,
        'max_bridges': args.max_bridges,
        'bridge_priority': args.bridge_priority,
        'min_synthesis_overlap': args.min_synthesis_overlap
    }

    # Add stage identifier
    results['stage'] = 'knowledge_bridge'

    # Filter by focus domains if specified
    if args.focus_domains:
        focus = set(d.strip() for d in args.focus_domains.split(','))
        results['gaps'] = [
            g for g in results['gaps']
            if any(f"domain_{d}" in str(g.get('between', [])) for d in focus)
        ]
        if args.verbose:
            print(f"[knowledge_bridge] Filtered to domains: {focus}", file=sys.stderr)

    # Add actionable recommendations if requested
    if args.include_actionable:
        for gap in results['gaps']:
            bridges = gap.get('potential_bridges', [])
            if bridges:
                gap['recommendation'] = f"Create content bridging via: {', '.join(bridges[:3])}"
            else:
                gap['recommendation'] = "Consider creating explicit bridge document"

    # Handle chain output
    if args.preserve_chain and chain:
        chain.add_result('knowledge_bridge', results)
        output_data = chain.to_dict()
    else:
        output_data = results

    # Output
    if args.pretty:
        print(json.dumps(output_data, indent=2))
    else:
        print(json.dumps(output_data))


if __name__ == "__main__":
    main()
