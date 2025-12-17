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
from cortical.query import (
    apply_definition_boost,
    boost_definition_documents,
    detect_definition_query
)


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


def get_doc_type_label(doc_id: str) -> str:
    """Get a display label for document type."""
    if doc_id.startswith('samples/memories/'):
        # Check if it's a concept doc
        filename = doc_id.split('/')[-1]
        if filename.startswith('concept-'):
            return 'CON'
        return 'MEM'
    elif doc_id.startswith('samples/decisions/'):
        return 'ADR'
    elif doc_id.endswith('.md'):
        if doc_id.startswith('docs/'):
            return 'DOCS'
        return 'DOC'
    elif doc_id.startswith('tests/'):
        return 'TEST'
    return 'CODE'


def find_similar_code(
    processor: CorticalTextProcessor,
    target: str,
    top_n: int = 5,
    chunk_size: int = 400
) -> list:
    """
    Find code similar to a target reference or text.

    Args:
        processor: CorticalTextProcessor instance
        target: Either a file:line reference (e.g., "cortical/processor.py:100")
                or raw text to compare against
        top_n: Number of similar results to return
        chunk_size: Size of text chunks to compare

    Returns:
        List of result dicts with similarity scores
    """
    # Determine if target is a file reference or raw text
    if ':' in target and not target.startswith('http'):
        # Parse file:line reference
        parts = target.split(':')
        file_path = parts[0]
        try:
            line_num = int(parts[1]) if len(parts) > 1 else 1
        except ValueError:
            line_num = 1

        # Get content from indexed document
        doc_content = processor.documents.get(file_path, '')
        if not doc_content:
            # Try without extension
            for doc_id in processor.documents:
                if doc_id.endswith(file_path) or file_path.endswith(doc_id):
                    doc_content = processor.documents[doc_id]
                    file_path = doc_id
                    break

        if not doc_content:
            return []

        # Extract passage around the line
        lines = doc_content.split('\n')
        start_line = max(0, line_num - 1)
        end_line = min(len(lines), start_line + 20)  # ~20 lines of context
        target_text = '\n'.join(lines[start_line:end_line])
    else:
        target_text = target

    if not target_text.strip():
        return []

    # Get fingerprint of target
    target_fp = processor.get_fingerprint(target_text, top_n=20)

    # Compare against all documents
    results = []
    for doc_id, doc_content in processor.documents.items():
        # Skip the source document if we're searching from a file reference
        if ':' in target and doc_id in target:
            continue

        # Chunk the document and compare each chunk
        for start in range(0, len(doc_content), chunk_size // 2):
            end = min(start + chunk_size, len(doc_content))
            chunk = doc_content[start:end]

            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue

            chunk_fp = processor.get_fingerprint(chunk, top_n=20)
            comparison = processor.compare_fingerprints(target_fp, chunk_fp)

            similarity = comparison.get('overall_similarity', 0)
            if similarity > 0.1:  # Only include somewhat similar results
                line_num = find_line_number(doc_content, start)
                results.append({
                    'file': doc_id,
                    'line': line_num,
                    'passage': chunk,
                    'score': similarity,
                    'reference': f"{doc_id}:{line_num}",
                    'doc_type': get_doc_type_label(doc_id),
                    'shared_terms': list(comparison.get('shared_terms', []))[:5]
                })

    # Sort by similarity and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def display_similar_results(results: list, target: str, verbose: bool = False):
    """Display similar code results."""
    if not results:
        print(f"\nNo similar code found for: {target}")
        return

    print(f"\n{'─' * 60}")
    print(f"Code similar to: {target}")
    print(f"{'─' * 60}\n")

    for i, result in enumerate(results, 1):
        print(f"[{i}] [{result['doc_type']}] {result['reference']}")
        print(f"    Similarity: {result['score']:.1%}")
        if result.get('shared_terms'):
            print(f"    Shared: {', '.join(result['shared_terms'])}")
        print("─" * 50)

        if verbose:
            print(format_passage(result['passage']))
        else:
            lines = result['passage'].split('\n')[:5]
            for line in lines:
                if len(line) > 70:
                    line = line[:67] + '...'
                print(f"  {line}")
            if len(result['passage'].split('\n')) > 5:
                print(f"  ...")
        print()


def search_codebase(
    processor: CorticalTextProcessor,
    query: str,
    top_n: int = 5,
    chunk_size: int = 400,
    fast: bool = False,
    prefer_docs: bool = False,
    no_boost: bool = False
) -> list:
    """
    Search the codebase and return results with file:line references.

    Args:
        processor: CorticalTextProcessor instance
        query: Search query string
        top_n: Number of results to return
        chunk_size: Size of text chunks for passage extraction
        fast: Use fast search mode (documents only, no passages)
        prefer_docs: Always boost documentation files
        no_boost: Disable all document type boosting

    Returns:
        List of result dicts with 'file', 'line', 'passage', 'score', 'reference', 'doc_type'
    """
    if fast:
        # Fast mode with optional boosting
        if no_boost:
            doc_results = processor.fast_find_documents(query, top_n=top_n)
        else:
            doc_results = processor.find_documents_with_boost(
                query,
                top_n=top_n,
                auto_detect_intent=not prefer_docs,
                prefer_docs=prefer_docs
            )
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
                'reference': f"{doc_id}:1",
                'doc_type': get_doc_type_label(doc_id)
            })
        return formatted_results

    # Full passage search - first get top documents with boosting
    if no_boost:
        doc_results = processor.find_documents_for_query(query, top_n=top_n * 3)
    else:
        doc_results = processor.find_documents_with_boost(
            query,
            top_n=top_n * 3,  # Get more candidates for re-ranking
            auto_detect_intent=not prefer_docs,
            prefer_docs=prefer_docs
        )

    # Apply definition boost to documents (helps find class/def definitions)
    doc_results = boost_definition_documents(
        doc_results, query, processor.documents, boost_factor=2.0
    )

    # Then get passages from those documents
    doc_ids = [doc_id for doc_id, _ in doc_results]

    # Check if this is a definition query - if so, fetch more candidates
    definition_info = detect_definition_query(query)
    candidate_multiplier = 5 if definition_info['is_definition_query'] else 2

    # For definition queries, search fewer docs (boosted ones at top)
    # For regular queries, search more broadly
    doc_limit = top_n if definition_info['is_definition_query'] else top_n * 3

    results = processor.find_passages_for_query(
        query,
        top_n=top_n * candidate_multiplier,  # More candidates for definition queries
        chunk_size=chunk_size,
        overlap=100,
        doc_filter=doc_ids[:doc_limit] if doc_ids else None
    )

    # Apply definition boost for "class X" or "def X" queries
    # This helps find actual definitions instead of just usages
    results = apply_definition_boost(results, query, boost_factor=5.0)
    results = results[:top_n]  # Trim after boosting

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
            'reference': f"{doc_id}:{line_num}",
            'doc_type': get_doc_type_label(doc_id)
        })

    return formatted_results


def display_results(results: list, verbose: bool = False, show_doc_type: bool = True):
    """Display search results."""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} relevant passages:\n")

    for i, result in enumerate(results, 1):
        print("=" * 60)
        doc_type = result.get('doc_type', '')
        type_indicator = f"[{doc_type}] " if show_doc_type and doc_type else ""
        print(f"[{i}] {type_indicator}{result['reference']}")
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


def expand_query_display(processor: CorticalTextProcessor, query: str, use_code: bool = False):
    """Show expanded query terms."""
    if use_code:
        expanded = processor.expand_query_for_code(query, max_expansions=15)
        print("\nQuery expansion (code-aware):")
        print("  (includes programming synonyms: get/fetch/load, etc.)")
    else:
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
    print("  /code <query>    - Show code-aware expansion (programming synonyms)")
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
                print("Commands: /expand, /code, /concepts, /stats, /quit")
            elif cmd == '/stats':
                print(f"\nCorpus Statistics:")
                print(f"  Documents: {len(processor.documents)}")
                print(f"  Tokens: {processor.layers[0].column_count()}")
                print(f"  Bigrams: {processor.layers[1].column_count()}")
                print(f"  Concepts: {processor.layers[2].column_count()}")
                print(f"  Relations: {len(processor.semantic_relations)}")
            elif cmd == '/expand' and len(cmd_parts) > 1:
                expand_query_display(processor, cmd_parts[1])
            elif cmd == '/code' and len(cmd_parts) > 1:
                expand_query_display(processor, cmd_parts[1], use_code=True)
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
    parser = argparse.ArgumentParser(
        description='Search the indexed codebase',
        epilog="""
Examples:
  %(prog)s "PageRank algorithm"           # Search with auto-boost
  %(prog)s "what is PageRank" --prefer-docs  # Always boost docs
  %(prog)s "compute pagerank" --no-boost  # Disable boosting
  %(prog)s "architecture" --fast          # Fast document-level search
  %(prog)s "fetch data" --code            # Code-aware (also finds get/load/retrieve)
  %(prog)s "auth" --code --expand         # Show programming synonyms
  %(prog)s --similar-to cortical/processor.py:100  # Find similar code
  %(prog)s -s "def compute_pagerank"      # Find code similar to text
        """
    )
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--corpus', '-c', default='corpus_dev.json',
                        help='Corpus file path (default: corpus_dev.json, also supports .pkl)')
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
    parser.add_argument('--prefer-docs', '-d', action='store_true',
                        help='Always boost documentation files in results')
    parser.add_argument('--no-boost', action='store_true',
                        help='Disable document type boosting (raw TF-IDF)')
    parser.add_argument('--code', action='store_true',
                        help='Use code-aware query expansion (get→fetch/load/retrieve)')
    parser.add_argument('--similar-to', '-s', metavar='TARGET',
                        help='Find code similar to file:line reference or text')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    corpus_path = base_path / args.corpus

    # Check if corpus exists - try JSON first, then pkl for backward compatibility
    if not corpus_path.exists():
        # If specified path doesn't exist, try alternate extension
        if str(corpus_path).endswith('.json'):
            pkl_path = base_path / str(args.corpus).replace('.json', '.pkl')
            if pkl_path.exists():
                corpus_path = pkl_path
                print(f"Note: Using {pkl_path.name} (consider migrating to JSON format)")
            else:
                print(f"Error: Corpus file not found: {corpus_path}")
                print("Run 'python scripts/index_codebase.py' first to create it.")
                sys.exit(1)
        elif str(corpus_path).endswith('.pkl'):
            json_path = base_path / str(args.corpus).replace('.pkl', '.json')
            if json_path.exists():
                corpus_path = json_path
                print(f"Note: Using {json_path.name} (JSON format)")
            else:
                print(f"Error: Corpus file not found: {corpus_path}")
                print("Run 'python scripts/index_codebase.py' first to create it.")
                sys.exit(1)
        else:
            print(f"Error: Corpus file not found: {corpus_path}")
            print("Run 'python scripts/index_codebase.py' first to create it.")
            sys.exit(1)

    # Load corpus (auto-detects format)
    print(f"Loading corpus from {corpus_path}...")
    processor = CorticalTextProcessor.load(str(corpus_path))
    print(f"Loaded {len(processor.documents)} documents\n")

    if args.interactive:
        interactive_mode(processor)
    elif args.similar_to:
        # Find similar code mode
        results = find_similar_code(
            processor,
            args.similar_to,
            top_n=args.top
        )
        display_similar_results(results, args.similar_to, verbose=args.verbose)
    elif args.query:
        if args.expand:
            expand_query_display(processor, args.query, use_code=args.code)
            print()

        # Show query intent detection
        is_conceptual = processor.is_conceptual_query(args.query)
        if not args.no_boost:
            intent_str = "conceptual" if is_conceptual else "implementation"
            print(f"(Query type: {intent_str})")

        results = search_codebase(
            processor,
            args.query,
            top_n=args.top,
            fast=args.fast,
            prefer_docs=args.prefer_docs,
            no_boost=args.no_boost
        )
        if args.fast:
            print("(Fast mode: document-level results)")
        display_results(results, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
