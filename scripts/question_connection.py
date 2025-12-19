#!/usr/bin/env python3
"""
Question Connection Pipeline
============================

Takes JSON output from world_model_analysis.py and explores concept connections.
Supports ThoughtChain format for loop-based reanalysis and context preservation.

This script:
1. Reads JSON from stdin or file (output of world_model_analysis.py --json)
2. Expands query terms using the Cortical library
3. Finds connection paths between related concepts
4. Outputs enriched JSON for further processing

Usage:
    # From stdin (pipeline mode)
    python scripts/world_model_analysis.py --json | python scripts/question_connection.py

    # From file
    python scripts/question_connection.py --input analysis.json

    # With query exploration
    python scripts/question_connection.py --input analysis.json --query "prediction model"

    # Find connection paths
    python scripts/question_connection.py --input analysis.json --explore

    # Show query expansion
    python scripts/question_connection.py --input analysis.json --query "learning" --expand

    # With thought chain parameters
    python scripts/question_connection.py --input chain.json --max-depth 5 --min-weight 0.3

Chain-Aware Features:
    - Auto-detects ThoughtChain format and preserves chain metadata
    - Reads query from chain context if not provided via CLI
    - Stores results in chain format for downstream stages
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer

# Import thought chain utilities
try:
    from scripts.thought_chain import ThoughtChain, is_chain_format, extract_from_chain
    CHAIN_AVAILABLE = True
except ImportError:
    CHAIN_AVAILABLE = False
    def is_chain_format(data): return False
    def extract_from_chain(data, stage): return data


def load_json_input(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load JSON input from stdin or file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If input is invalid JSON or missing
        FileNotFoundError: If input file doesn't exist
    """
    if args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Read from stdin
        if sys.stdin.isatty():
            raise ValueError(
                "No input provided. Use --input FILE or pipe JSON from stdin.\n"
                "Example: python scripts/world_model_analysis.py --json | "
                "python scripts/question_connection.py"
            )
        try:
            return json.load(sys.stdin)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")


def build_processor_from_json(data: Dict[str, Any]) -> CorticalTextProcessor:
    """
    Build a minimal CorticalTextProcessor from JSON data.

    Args:
        data: JSON data from world_model_analysis.py

    Returns:
        CorticalTextProcessor with concepts loaded
    """
    processor = CorticalTextProcessor()

    # Create synthetic documents from concepts to enable query expansion
    # Group concepts by domain for realistic document structure
    domain_concepts = defaultdict(list)

    if 'concepts' in data:
        for concept in data['concepts']:
            # Skip malformed concept entries
            if not isinstance(concept, dict) or 'term' not in concept:
                continue
            term = concept['term']
            domains = concept.get('domains', ['general'])
            for domain in domains:
                domain_concepts[domain].append(term)

    # Create a document per domain with its concepts
    for domain, terms in domain_concepts.items():
        doc_id = f"synthetic_{domain}"
        # Create text by repeating terms to simulate co-occurrence
        text = ' '.join(terms * 2)  # Repeat to create connections
        processor.process_document(doc_id, text)

    # Compute to enable query expansion
    processor.compute_all(verbose=False)

    return processor


def expand_query_terms(
    processor: CorticalTextProcessor,
    query: str,
    max_expansions: int = 10
) -> Dict[str, float]:
    """
    Expand query terms using lateral connections.

    Args:
        processor: CorticalTextProcessor instance
        query: Query text to expand
        max_expansions: Maximum number of expansion terms

    Returns:
        Dictionary of {term: weight} for expanded query
    """
    return processor.expand_query(query, max_expansions=max_expansions)


def find_connection_paths(
    data: Dict[str, Any],
    max_paths: int = 10,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    Find connection paths between concepts using network data.

    Uses BFS to find shortest paths through the concept network.

    Args:
        data: JSON data containing network connections
        max_paths: Maximum number of paths to return
        max_depth: Maximum path depth to explore

    Returns:
        List of path dictionaries with from, to, via, and total_weight
    """
    paths = []

    # Build adjacency list from network data
    adjacency = defaultdict(dict)
    if 'network' in data:
        for source_term, connections in data['network'].items():
            for conn in connections:
                target_term = conn['term']
                weight = conn['weight']
                adjacency[source_term][target_term] = weight

    # Get all nodes
    all_nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        all_nodes.update(neighbors.keys())

    nodes = list(all_nodes)

    # Find paths between interesting pairs (high-weight nodes)
    # Use concepts list if available for prioritization
    priority_nodes = nodes[:20]  # Limit to avoid combinatorial explosion

    if 'concepts' in data:
        # Sort by pagerank to prioritize important concepts
        sorted_concepts = sorted(
            data['concepts'],
            key=lambda x: x.get('pagerank', 0),
            reverse=True
        )
        priority_nodes = [c['term'] for c in sorted_concepts[:20]]

    for i, start in enumerate(priority_nodes):
        if start not in adjacency:
            continue

        for end in priority_nodes[i+1:]:
            if end == start:
                continue

            # BFS to find path
            path = _bfs_path(adjacency, start, end, max_depth)

            if path and len(path) > 1:  # Only keep paths with intermediate nodes
                # Calculate total weight
                total_weight = 0.0
                for j in range(len(path) - 1):
                    total_weight += adjacency.get(path[j], {}).get(path[j+1], 0)

                paths.append({
                    'from': path[0],
                    'to': path[-1],
                    'via': path[1:-1] if len(path) > 2 else [],
                    'total_weight': round(total_weight, 2),
                    'length': len(path) - 1
                })

            if len(paths) >= max_paths:
                break

        if len(paths) >= max_paths:
            break

    # Sort by total weight and path length
    paths.sort(key=lambda x: (-x['total_weight'], x['length']))

    return paths[:max_paths]


def _bfs_path(
    adjacency: Dict[str, Dict[str, float]],
    start: str,
    end: str,
    max_depth: int = 3
) -> Optional[List[str]]:
    """
    Find shortest path using BFS.

    Args:
        adjacency: Adjacency list {node: {neighbor: weight}}
        start: Start node
        end: End node
        max_depth: Maximum depth to search

    Returns:
        Path as list of nodes, or None if no path found
    """
    if start == end:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()

        if len(path) > max_depth:
            continue

        for neighbor in adjacency.get(node, {}):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def find_related_concepts(
    data: Dict[str, Any],
    query_terms: List[str],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Find concepts related to query terms.

    Args:
        data: JSON data with concepts
        query_terms: List of query terms
        top_k: Number of results to return

    Returns:
        List of related concept dictionaries with term and relevance
    """
    related = []

    # Handle invalid top_k
    if top_k <= 0:
        return related

    if 'concepts' not in data:
        return related

    # Score concepts by relevance to query
    for concept in data['concepts']:
        # Skip malformed concept entries
        if not isinstance(concept, dict) or 'term' not in concept:
            continue

        term = concept['term']

        # Skip if term is in query
        if term in query_terms:
            continue

        # Calculate relevance based on:
        # 1. PageRank (importance)
        # 2. Connection count (centrality)
        # 3. Domain overlap (if query terms have domains)

        pagerank = concept.get('pagerank', 0)
        conn_count = concept.get('connection_count', 0)

        # Simple scoring: weighted combination
        relevance = (pagerank * 0.6) + (min(conn_count / 100.0, 1.0) * 0.4)

        related.append({
            'term': term,
            'relevance': round(relevance, 4),
            'pagerank': round(pagerank, 6),
            'connections': conn_count,
            'domains': concept.get('domains', [])
        })

    # Sort by relevance
    related.sort(key=lambda x: -x['relevance'])

    return related[:top_k]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore concept connections from world model analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pipeline mode (from stdin)
  python scripts/world_model_analysis.py --json | python scripts/question_connection.py

  # From file with query
  python scripts/question_connection.py --input analysis.json --query "prediction"

  # Find connection paths
  python scripts/question_connection.py --input analysis.json --explore

  # Show query expansion
  python scripts/question_connection.py --input analysis.json --query "learning" --expand

  # Limit results
  python scripts/question_connection.py --input analysis.json --query "model" --top_k 5

  # Advanced: adjust path exploration
  python scripts/question_connection.py --input analysis.json --explore --max-depth 5 --min-weight 0.3

  # Chain-aware mode (preserves thought chain context)
  python scripts/cognitive_pipeline.py --query "learning" | python scripts/question_connection.py
        """
    )

    parser.add_argument(
        '--input', '-i',
        help='Input JSON file (default: read from stdin)'
    )

    parser.add_argument(
        '--query', '-q',
        help='Query terms to explore'
    )

    parser.add_argument(
        '--explore',
        action='store_true',
        help='Find connection paths between concepts'
    )

    parser.add_argument(
        '--expand',
        action='store_true',
        help='Show query expansion terms'
    )

    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )

    # New parameters for fine-grained control
    parser.add_argument(
        '--max-depth',
        type=int,
        default=3,
        help='Maximum path depth for exploration (default: 3)'
    )

    parser.add_argument(
        '--min-weight',
        type=float,
        default=0.0,
        help='Minimum connection weight threshold (default: 0.0)'
    )

    parser.add_argument(
        '--max-expansions',
        type=int,
        default=20,
        help='Maximum expansion terms per query (default: 20)'
    )

    parser.add_argument(
        '--focus-domains',
        help='Comma-separated list of domains to focus on'
    )

    parser.add_argument(
        '--preserve-chain',
        action='store_true',
        help='Preserve thought chain metadata in output'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output to stderr'
    )

    args = parser.parse_args()

    try:
        # Load input
        raw_data = load_json_input(args)

        # Check for thought chain format
        chain = None
        if CHAIN_AVAILABLE and is_chain_format(raw_data):
            chain = ThoughtChain.from_dict(raw_data)
            # Extract the relevant stage data
            data = extract_from_chain(raw_data, 'world_model_analysis')
            if not data or data == raw_data:
                # Try input stage
                data = extract_from_chain(raw_data, 'input')
                if not data or data == raw_data:
                    data = raw_data.get('results', {}).get('world_model_analysis', raw_data)

            # Use query from chain context if not provided via CLI
            if not args.query and chain.context.query:
                args.query = chain.context.query
                if args.verbose:
                    print(f"Using query from chain: {args.query}", file=sys.stderr)
        else:
            data = raw_data

        # Build output structure
        output = {
            'stage': 'question_connection',
            'query': args.query or None,
            'expanded_terms': [],
            'paths': [],
            'related_concepts': [],
            'parameters': {
                'top_k': args.top_k,
                'max_depth': args.max_depth,
                'min_weight': args.min_weight,
                'max_expansions': args.max_expansions
            },
            'input_summary': {
                'concepts_count': len(data.get('concepts', [])),
                'bridges_count': len(data.get('bridges', [])),
                'domains_count': len(data.get('domains', {})),
                'network_size': len(data.get('network', {}))
            }
        }

        # Filter by focus domains if specified
        focus_domains = None
        if args.focus_domains:
            focus_domains = set(d.strip() for d in args.focus_domains.split(','))
            if args.verbose:
                print(f"Focusing on domains: {focus_domains}", file=sys.stderr)

        # Process query if provided
        if args.query:
            # Build processor for query expansion
            if args.expand:
                processor = build_processor_from_json(data)
                expanded = expand_query_terms(processor, args.query, args.max_expansions)

                # Convert to list format, filtering by min weight
                output['expanded_terms'] = [
                    {'term': term, 'weight': round(weight, 4)}
                    for term, weight in sorted(
                        expanded.items(),
                        key=lambda x: -x[1]
                    )
                    if weight >= args.min_weight
                ][:args.top_k]

            # Find related concepts
            tokenizer = Tokenizer()
            query_terms = tokenizer.tokenize(args.query)
            related = find_related_concepts(data, query_terms, args.top_k * 2)

            # Filter by focus domains if specified
            if focus_domains:
                related = [
                    r for r in related
                    if any(d in focus_domains for d in r.get('domains', []))
                ]

            output['related_concepts'] = related[:args.top_k]

        # Find connection paths if requested
        if args.explore:
            output['paths'] = find_connection_paths(
                data, max_paths=args.top_k, max_depth=args.max_depth
            )

        # Wrap in chain format if requested or if input was chain
        if args.preserve_chain and chain:
            chain.add_result('question_connection', output)
            print(chain.to_json())
        elif chain and CHAIN_AVAILABLE:
            # Add to chain but output just this stage
            chain.add_result('question_connection', output)
            print(json.dumps(output, indent=2))
        else:
            # Output JSON
            print(json.dumps(output, indent=2))

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == '__main__':
    main()
