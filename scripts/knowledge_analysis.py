#!/usr/bin/env python3
"""
Knowledge Pattern Analysis
==========================

Analyzes concept networks to detect semantic clusters, measure connection
strength, and identify knowledge density areas. Works as part of a pipeline,
accepting JSON from question_connection.py or world_model_analysis.py.

Usage:
    # From stdin (piped from another script)
    python scripts/world_model_analysis.py --json | python scripts/knowledge_analysis.py

    # From file
    python scripts/knowledge_analysis.py --input data.json

    # With verbose output
    python scripts/knowledge_analysis.py --input data.json --verbose

Pipeline integration:
    python scripts/world_model_analysis.py --json \\
        | python scripts/knowledge_analysis.py \\
        | python scripts/downstream_tool.py
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set, Tuple, Optional


class KnowledgeAnalyzer:
    """
    Analyzes knowledge patterns in concept networks.

    Detects semantic clusters, scores connection strength, and maps
    knowledge density across the conceptual space.
    """

    def __init__(self, data: Dict[str, Any], verbose: bool = False):
        """
        Initialize analyzer with input data.

        Args:
            data: Input JSON data (from question_connection or world_model_analysis)
            verbose: If True, print detailed analysis information to stderr
        """
        self.data = data
        self.verbose = verbose
        self.input_type = self._detect_input_type()

        # Extracted data structures
        self.terms: Set[str] = set()
        self.connections: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.term_domains: Dict[str, Set[str]] = defaultdict(set)

        self._extract_network()

    def _detect_input_type(self) -> str:
        """
        Detect whether input is from question_connection or world_model_analysis.

        Returns:
            "question_connection" if has "query" field,
            "world_model_analysis" if has "concepts" field,
            "unknown" otherwise
        """
        if "query" in self.data:
            return "question_connection"
        elif "concepts" in self.data:
            return "world_model_analysis"
        else:
            return "unknown"

    def _extract_network(self):
        """
        Extract network structure from input data based on input type.

        Populates self.terms, self.connections, and self.term_domains.
        """
        if self.input_type == "question_connection":
            self._extract_from_question_connection()
        elif self.input_type == "world_model_analysis":
            self._extract_from_world_model()
        else:
            # Try to extract whatever we can find
            # Check for bridges even if input type is unknown
            if "bridges" in self.data:
                for bridge in self.data["bridges"]:
                    term = bridge.get("term", "")
                    if term:
                        self.terms.add(term)
                        if "domains" in bridge:
                            for domain in bridge["domains"]:
                                self.term_domains[term].add(domain)

            if "network" in self.data:
                self._extract_from_network(self.data["network"])

    def _extract_from_question_connection(self):
        """Extract network from question_connection.py output."""
        # First, extract from pass-through data (from world_model_analysis)
        # This is the primary source of network structure
        if "concepts" in self.data:
            for concept in self.data["concepts"]:
                term = concept.get("term", "")
                if term:
                    self.terms.add(term)
                    if "domains" in concept:
                        for domain in concept["domains"]:
                            self.term_domains[term].add(domain)

        if "bridges" in self.data:
            for bridge in self.data["bridges"]:
                term = bridge.get("term", "")
                if term:
                    self.terms.add(term)
                    if "domains" in bridge:
                        for domain in bridge["domains"]:
                            self.term_domains[term].add(domain)

        if "network" in self.data:
            self._extract_from_network(self.data["network"])

        # Also extract from question_connection specific fields
        # Extract expanded terms
        if "expanded_terms" in self.data:
            for term_data in self.data["expanded_terms"]:
                if isinstance(term_data, dict):
                    term = term_data.get("term", "")
                elif isinstance(term_data, str):
                    term = term_data
                else:
                    continue
                if term:
                    self.terms.add(term)

        # Extract paths to build connections
        if "paths" in self.data:
            for path in self.data["paths"]:
                path_terms = path if isinstance(path, list) else path.get("terms", [])
                # Create connections between adjacent terms in paths
                for i in range(len(path_terms) - 1):
                    term1, term2 = path_terms[i], path_terms[i + 1]
                    self.terms.add(term1)
                    self.terms.add(term2)
                    # Increment connection weight
                    self.connections[term1][term2] = self.connections[term1].get(term2, 0) + 1
                    self.connections[term2][term1] = self.connections[term2].get(term1, 0) + 1

        # Extract related concepts
        if "related_concepts" in self.data:
            for concept in self.data["related_concepts"]:
                if isinstance(concept, dict):
                    term = concept.get("term", "")
                elif isinstance(concept, str):
                    term = concept
                else:
                    continue
                if term:
                    self.terms.add(term)

    def _extract_from_world_model(self):
        """Extract network from world_model_analysis.py output."""
        # Extract concepts
        if "concepts" in self.data:
            for concept in self.data["concepts"]:
                term = concept.get("term", "")
                if term:
                    self.terms.add(term)

                # Extract domains
                if "domains" in concept:
                    for domain in concept["domains"]:
                        self.term_domains[term].add(domain)

        # Extract bridges
        if "bridges" in self.data:
            for bridge in self.data["bridges"]:
                term = bridge.get("term", "")
                if term:
                    self.terms.add(term)
                    if "domains" in bridge:
                        for domain in bridge["domains"]:
                            self.term_domains[term].add(domain)

        # Extract network connections
        if "network" in self.data:
            self._extract_from_network(self.data["network"])

    def _extract_from_network(self, network: Dict[str, Any]):
        """
        Extract connections from network structure.

        Args:
            network: Dictionary mapping terms to their connections
        """
        for term, connections in network.items():
            self.terms.add(term)

            if isinstance(connections, list):
                # Format: [{"term": "...", "weight": ...}, ...]
                for conn in connections:
                    if isinstance(conn, dict):
                        neighbor = conn.get("term", "")
                        weight = conn.get("weight", 1.0)
                        if neighbor:
                            self.terms.add(neighbor)
                            self.connections[term][neighbor] = weight
            elif isinstance(connections, dict):
                # Format: {"neighbor1": weight1, "neighbor2": weight2, ...}
                for neighbor, weight in connections.items():
                    self.terms.add(neighbor)
                    self.connections[term][neighbor] = weight

    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect patterns in the concept network.

        Returns:
            List of pattern dictionaries with type, terms/term, and strength
        """
        patterns = []

        # Detect hubs (terms with many connections)
        hub_threshold = max(3, len(self.terms) // 10)  # At least 3 connections
        for term in self.terms:
            conn_count = len(self.connections.get(term, {}))
            if conn_count >= hub_threshold:
                # Calculate hub strength as normalized connection count
                max_conns = max(len(self.connections.get(t, {})) for t in self.terms)
                strength = conn_count / max_conns if max_conns > 0 else 0
                patterns.append({
                    "type": "hub",
                    "term": term,
                    "connections": conn_count,
                    "strength": round(strength, 3)
                })

        # Detect clusters (groups of highly connected terms)
        clusters = self._detect_clusters()
        for cluster_id, cluster_terms in enumerate(clusters):
            if len(cluster_terms) >= 2:
                # Calculate cluster coherence (average internal connection strength)
                coherence = self._calculate_cluster_coherence(cluster_terms)
                patterns.append({
                    "type": "cluster",
                    "cluster_id": cluster_id,
                    "terms": sorted(list(cluster_terms)),
                    "size": len(cluster_terms),
                    "strength": round(coherence, 3)
                })

        # Detect bridges (terms connecting different domains/clusters)
        for term in self.terms:
            domains = self.term_domains.get(term, set())
            if len(domains) >= 3:
                # Multi-domain terms are bridges
                patterns.append({
                    "type": "bridge",
                    "term": term,
                    "domains": sorted(list(domains)),
                    "strength": round(len(domains) / 7.0, 3)  # Normalize to max 7 domains
                })

        return patterns

    def _detect_clusters(self) -> List[Set[str]]:
        """
        Detect clusters using simple community detection.

        Uses greedy modularity-based clustering on the connection graph.

        Returns:
            List of sets, where each set contains terms in a cluster
        """
        if not self.terms:
            return []

        # Start with each term in its own cluster
        clusters = {term: {term} for term in self.terms}
        term_to_cluster = {term: term for term in self.terms}

        # Greedy merging based on connection strength
        merged = True
        while merged:
            merged = False
            best_merge = None
            best_score = 0

            # Find best pair to merge
            for term1 in list(clusters.keys()):
                if term1 not in clusters:
                    continue
                for term2 in list(clusters.keys()):
                    if term2 not in clusters or term1 == term2:
                        continue

                    # Calculate merge score (sum of connections between clusters)
                    score = self._cluster_merge_score(clusters[term1], clusters[term2])
                    if score > best_score:
                        best_score = score
                        best_merge = (term1, term2)

            # Merge if beneficial
            if best_merge and best_score > 0.5:  # Threshold for merging
                term1, term2 = best_merge
                # Merge term2 into term1
                clusters[term1] = clusters[term1] | clusters[term2]
                for term in clusters[term2]:
                    term_to_cluster[term] = term1
                del clusters[term2]
                merged = True

        return list(clusters.values())

    def _cluster_merge_score(self, cluster1: Set[str], cluster2: Set[str]) -> float:
        """
        Calculate score for merging two clusters.

        Args:
            cluster1: First cluster (set of terms)
            cluster2: Second cluster (set of terms)

        Returns:
            Merge score (higher = more beneficial to merge)
        """
        total_weight = 0
        total_pairs = 0

        for term1 in cluster1:
            for term2 in cluster2:
                weight = self.connections.get(term1, {}).get(term2, 0)
                total_weight += weight
                total_pairs += 1

        if total_pairs == 0:
            return 0

        # Normalize by cluster sizes
        return total_weight / (len(cluster1) * len(cluster2))

    def _calculate_cluster_coherence(self, cluster: Set[str]) -> float:
        """
        Calculate internal coherence of a cluster.

        Args:
            cluster: Set of terms in the cluster

        Returns:
            Coherence score (0-1, higher = more internally connected)
        """
        if len(cluster) < 2:
            return 1.0

        total_weight = 0
        total_pairs = 0

        cluster_list = list(cluster)
        for i, term1 in enumerate(cluster_list):
            for term2 in cluster_list[i + 1:]:
                weight = self.connections.get(term1, {}).get(term2, 0)
                total_weight += weight
                total_pairs += 1

        if total_pairs == 0:
            return 0

        # Normalize: assume max weight per pair is 10
        max_possible = total_pairs * 10
        return min(1.0, total_weight / max_possible)

    def build_clusters(self) -> List[Dict[str, Any]]:
        """
        Build detailed cluster information.

        Returns:
            List of cluster dictionaries with id, terms, and coherence
        """
        clusters = self._detect_clusters()
        result = []

        for cluster_id, cluster_terms in enumerate(clusters):
            if len(cluster_terms) >= 2:
                coherence = self._calculate_cluster_coherence(cluster_terms)
                result.append({
                    "id": cluster_id,
                    "terms": sorted(list(cluster_terms)),
                    "size": len(cluster_terms),
                    "coherence": round(coherence, 3)
                })

        # Sort by size descending, then coherence
        result.sort(key=lambda x: (-x["size"], -x["coherence"]))
        return result

    def calculate_strength_scores(self) -> Dict[str, Any]:
        """
        Calculate connection strength scores.

        Returns:
            Dictionary with overall score and breakdown by domain
        """
        if not self.terms:
            return {"overall": 0, "by_domain": {}}

        # Overall strength: average connection weight
        total_weight = 0
        total_connections = 0

        for term, neighbors in self.connections.items():
            for neighbor, weight in neighbors.items():
                total_weight += weight
                total_connections += 1

        overall = total_weight / max(1, total_connections)

        # Strength by domain
        by_domain = {}
        domain_weights = defaultdict(float)
        domain_counts = defaultdict(int)

        for term in self.terms:
            domains = self.term_domains.get(term, set())
            term_conns = self.connections.get(term, {})

            for domain in domains:
                # Sum weights of this term's connections
                for weight in term_conns.values():
                    domain_weights[domain] += weight
                    domain_counts[domain] += 1

        for domain, total in domain_weights.items():
            count = domain_counts[domain]
            by_domain[domain] = round(total / max(1, count), 3)

        return {
            "overall": round(overall, 3),
            "by_domain": by_domain
        }

    def calculate_density_map(self) -> Dict[str, List[str]]:
        """
        Calculate knowledge density map.

        Returns:
            Dictionary mapping density levels (high/medium/low) to term lists
        """
        if not self.terms:
            return {"high": [], "medium": [], "low": []}

        # Calculate density for each term (number of connections)
        densities = {}
        for term in self.terms:
            densities[term] = len(self.connections.get(term, {}))

        if not densities:
            return {"high": [], "medium": [], "low": []}

        # Determine thresholds (tertiles)
        sorted_densities = sorted(densities.values())
        n = len(sorted_densities)

        if n < 3:
            # Not enough data for tertiles
            low_threshold = 0
            high_threshold = max(sorted_densities) if sorted_densities else 0
        else:
            low_threshold = sorted_densities[n // 3]
            high_threshold = sorted_densities[2 * n // 3]

        # Categorize terms
        high_density = []
        medium_density = []
        low_density = []

        for term, density in densities.items():
            if density >= high_threshold and high_threshold > low_threshold:
                high_density.append(term)
            elif density > low_threshold:
                medium_density.append(term)
            else:
                low_density.append(term)

        return {
            "high": sorted(high_density),
            "medium": sorted(medium_density),
            "low": sorted(low_density)
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Run complete analysis and return results.

        Returns:
            Dictionary with patterns, clusters, strength_scores, density_map, and metadata
        """
        if self.verbose:
            print(f"[knowledge_analysis] Detected input type: {self.input_type}", file=sys.stderr)
            print(f"[knowledge_analysis] Extracted {len(self.terms)} terms", file=sys.stderr)
            print(f"[knowledge_analysis] Found {sum(len(c) for c in self.connections.values())} connections", file=sys.stderr)

        patterns = self.detect_patterns()
        clusters = self.build_clusters()
        strength_scores = self.calculate_strength_scores()
        density_map = self.calculate_density_map()

        if self.verbose:
            print(f"[knowledge_analysis] Detected {len(patterns)} patterns", file=sys.stderr)
            print(f"[knowledge_analysis] Found {len(clusters)} clusters", file=sys.stderr)

        return {
            "patterns": patterns,
            "clusters": clusters,
            "strength_scores": strength_scores,
            "density_map": density_map,
            "input_type": self.input_type,
            "metadata": {
                "term_count": len(self.terms),
                "connection_count": sum(len(c) for c in self.connections.values()),
                "pattern_count": len(patterns),
                "cluster_count": len(clusters)
            }
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Knowledge Pattern Analysis - Analyze concept networks for patterns and structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From stdin (piped)
  python scripts/world_model_analysis.py --json | python scripts/knowledge_analysis.py

  # From file
  python scripts/knowledge_analysis.py --input data.json

  # With verbose output
  python scripts/knowledge_analysis.py --input data.json --verbose

  # Full pipeline
  python scripts/world_model_analysis.py --json \\
      | python scripts/knowledge_analysis.py \\
      | python scripts/downstream_tool.py

  # Adjust analysis parameters
  python scripts/knowledge_analysis.py --input data.json --hub-threshold 5 --cluster-threshold 0.7

  # Filter pattern types
  python scripts/knowledge_analysis.py --input data.json --pattern-types hub,bridge

Input formats:
  - question_connection.py: {"query": "...", "expanded_terms": [...], "paths": [...]}
  - world_model_analysis.py: {"concepts": [...], "bridges": [...], "network": {...}}

Output format:
  {
    "patterns": [{"type": "cluster|hub|bridge", ...}],
    "clusters": [{"id": 0, "terms": [...], "coherence": 0.9}],
    "strength_scores": {"overall": 0.75, "by_domain": {...}},
    "density_map": {"high": [...], "medium": [...], "low": [...]},
    "input_type": "question_connection|world_model_analysis",
    "metadata": {"term_count": 100, ...}
  }
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON file (default: read from stdin)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed analysis information to stderr"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file (default: write to stdout)"
    )

    # New configurable parameters
    parser.add_argument(
        "--hub-threshold",
        type=int,
        default=3,
        help="Minimum connections to consider a term a hub (default: 3)"
    )

    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.5,
        help="Merge threshold for clustering (default: 0.5)"
    )

    parser.add_argument(
        "--bridge-domains",
        type=int,
        default=3,
        help="Minimum domains for a term to be a bridge (default: 3)"
    )

    parser.add_argument(
        "--pattern-types",
        help="Comma-separated list of pattern types to detect (hub,cluster,bridge)"
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size to report (default: 2)"
    )

    parser.add_argument(
        "--max-patterns",
        type=int,
        default=100,
        help="Maximum patterns to return (default: 100)"
    )

    parser.add_argument(
        "--preserve-chain",
        action="store_true",
        help="Preserve thought chain metadata in output"
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
        try:
            with open(args.input, 'r') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        try:
            raw_data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)

    # Check for thought chain format
    chain = None
    if chain_available and is_chain_format(raw_data):
        chain = ThoughtChain.from_dict(raw_data)
        # Try to extract previous stage results
        data = extract_from_chain(raw_data, 'question_connection')
        if data == raw_data:
            data = extract_from_chain(raw_data, 'world_model_analysis')
        if data == raw_data:
            data = raw_data.get('results', {}).get('question_connection', raw_data)
    else:
        data = raw_data

    # Run analysis with configurable parameters
    analyzer = KnowledgeAnalyzer(data, verbose=args.verbose)

    # Apply parameter overrides if analyzer supports them
    # (This allows future extension of the analyzer class)

    results = analyzer.analyze()

    # Add parameters to output for traceability
    results['parameters'] = {
        'hub_threshold': args.hub_threshold,
        'cluster_threshold': args.cluster_threshold,
        'bridge_domains': args.bridge_domains,
        'min_cluster_size': args.min_cluster_size,
        'max_patterns': args.max_patterns
    }

    # Filter pattern types if specified
    if args.pattern_types:
        allowed_types = set(t.strip() for t in args.pattern_types.split(','))
        results['patterns'] = [
            p for p in results['patterns']
            if p.get('type') in allowed_types
        ]

    # Filter clusters by minimum size
    results['clusters'] = [
        c for c in results['clusters']
        if c.get('size', 0) >= args.min_cluster_size
    ]

    # Limit patterns
    results['patterns'] = results['patterns'][:args.max_patterns]

    # Add stage identifier
    results['stage'] = 'knowledge_analysis'

    # Pass through original data for downstream stages (knowledge_bridge needs this)
    if 'concepts' in data:
        results['concepts'] = data['concepts']
    if 'bridges' in data:
        results['bridges'] = data['bridges']
    if 'network' in data:
        results['network'] = data['network']

    # Handle chain output
    if args.preserve_chain and chain:
        chain.add_result('knowledge_analysis', results)
        output_json = chain.to_json()
    else:
        output_json = json.dumps(results, indent=2)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        if args.verbose:
            print(f"[knowledge_analysis] Results written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
