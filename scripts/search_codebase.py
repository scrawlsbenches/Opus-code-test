#!/usr/bin/env python3
"""
Search the indexed codebase using Cortical Text Processor.

This script provides semantic search over the codebase with file:line references.

Usage:
    python scripts/search_codebase.py "how does PageRank work"
    python scripts/search_codebase.py "bigram separator" --top 10
    python scripts/search_codebase.py --interactive
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor


def find_line_number(doc_content: str, passage_start: int) -> int:
    """Find the line number for a character position."""
    return doc_content[:passage_start].count('\n') + 1


def format_passage(passage: str, max_width: int = 80) -> str:
    """Format a passage for display, truncating long lines."""
    lines = passage.split('\n')
    formatted = []
    for line in lines[:10]:  # Limit to 10 lines
        if len(line) > max_width:
            line = line[:max_width - 3] + '...'
        formatted.append(line)
    if len(lines) > 10:
        formatted.append(f'  ... ({len(lines) - 10} more lines)')
    return '\n'.join(formatted)


def search_codebase(processor: CorticalTextProcessor, query: str,
                    top_n: int = 5, chunk_size: int = 400, fast: bool = False) -> list:
    """
    Search the codebase and return results with file:line references.

    Args:
        processor: CorticalTextProcessor instance
        query: Search query string
        top_n: Number of results to return
        chunk_size: Size of text chunks for passage extraction
        fast: Use fast search mode (documents only, no passages)

    Returns:
        List of result dicts with 'file', 'line', 'passage', 'score', 'reference'
    """
    if fast:
        # Fast mode: just find documents, return first lines
        doc_results = processor.fast_find_documents(query, top_n=top_n)
        formatted_results = []
        for doc_id, score in doc_results:
            doc_content = processor.documents.get(doc_id, '')
            # Get first 400 chars as passage
            passage = doc_content[:400] if doc_content else ''
            formatted_results.append({
                'file': doc_id,
                'line': 1,
                'passage': passage,
                'score': score,
                'reference': f"{doc_id}:1"
            })
        return formatted_results

    # Full passage search
    results = processor.find_passages_for_query(
        query,
        top_n=top_n,
        chunk_size=chunk_size,
        overlap=100
    )

    formatted_results = []
    for passage, doc_id, start, end, score in results:
        # Get the full document content to find line number
        doc_content = processor.documents.get(doc_id, '')
        line_num = find_line_number(doc_content, start)

        formatted_results.append({
            'file': doc_id,
            'line': line_num,
            'passage': passage,
            'score': score,
            'reference': f"{doc_id}:{line_num}"
        })

    return formatted_results


def display_results(results: list, verbose: bool = False):
    """Display search results."""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} relevant passages:\n")

    for i, result in enumerate(results, 1):
        print("=" * 60)
        print(f"[{i}] {result['reference']}")
        print(f"    Score: {result['score']:.3f}")
        print("-" * 60)

        if verbose:
            print(format_passage(result['passage']))
        else:
            # Show first 5 lines
            lines = result['passage'].split('\n')[:5]
            for line in lines:
                if len(line) > 76:
                    line = line[:73] + '...'
                print(f"  {line}")
            if len(result['passage'].split('\n')) > 5:
                print(f"  ... ({len(result['passage'].split(chr(10))) - 5} more lines)")
        print()


def expand_query_display(processor: CorticalTextProcessor, query: str):
    """Show expanded query terms."""
    expanded = processor.expand_query(query, max_expansions=10)
    print("\nQuery expansion:")
    for term, weight in sorted(expanded.items(), key=lambda x: -x[1])[:10]:
        print(f"  {term}: {weight:.3f}")


def interactive_mode(processor: CorticalTextProcessor):
    """Run interactive search mode."""
    print("\nInteractive Search Mode")
    print("=" * 40)
    print("Commands:")
    print("  /expand <query>  - Show query expansion")
    print("  /concepts        - List concept clusters")
    print("  /stats           - Show corpus statistics")
    print("  /help            - Show this help")
    print("  /quit            - Exit")
    print()

    while True:
        try:
            query = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.startswith('/'):
            cmd_parts = query.split(maxsplit=1)
            cmd = cmd_parts[0].lower()

            if cmd == '/quit' or cmd == '/exit':
                print("Goodbye!")
                break
            elif cmd == '/help':
                print("Commands: /expand, /concepts, /stats, /quit")
            elif cmd == '/stats':
                print(f"\nCorpus Statistics:")
                print(f"  Documents: {len(processor.documents)}")
                print(f"  Tokens: {processor.layers[0].column_count()}")
                print(f"  Bigrams: {processor.layers[1].column_count()}")
                print(f"  Concepts: {processor.layers[2].column_count()}")
                print(f"  Relations: {len(processor.semantic_relations)}")
            elif cmd == '/expand' and len(cmd_parts) > 1:
                expand_query_display(processor, cmd_parts[1])
            elif cmd == '/concepts':
                layer2 = processor.layers[2]
                concepts = list(layer2.minicolumns.values())[:10]
                print(f"\nTop concepts ({layer2.column_count()} total):")
                for c in concepts:
                    print(f"  {c.content[:50]}")
            else:
                print(f"Unknown command: {cmd}")
        else:
            results = search_codebase(processor, query, top_n=5)
            display_results(results, verbose=True)


def main():
    parser = argparse.ArgumentParser(description='Search the indexed codebase')
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--corpus', '-c', default='corpus_dev.pkl',
                        help='Corpus file path (default: corpus_dev.pkl)')
    parser.add_argument('--top', '-n', type=int, default=5,
                        help='Number of results (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show full passage text')
    parser.add_argument('--expand', '-e', action='store_true',
                        help='Show query expansion')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive search mode')
    parser.add_argument('--fast', '-f', action='store_true',
                        help='Fast search mode (document-level, ~2-3x faster)')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    corpus_path = base_path / args.corpus

    # Check if corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        print("Run 'python scripts/index_codebase.py' first to create it.")
        sys.exit(1)

    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    processor = CorticalTextProcessor.load(str(corpus_path))
    print(f"Loaded {len(processor.documents)} documents\n")

    if args.interactive:
        interactive_mode(processor)
    elif args.query:
        if args.expand:
            expand_query_display(processor, args.query)
            print()

        results = search_codebase(processor, args.query, top_n=args.top, fast=args.fast)
        if args.fast:
            print("(Fast mode: document-level results)")
        display_results(results, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
