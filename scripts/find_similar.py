#!/usr/bin/env python3
"""
Find Similar Code - Semantic Fingerprinting for Code Similarity

Uses semantic fingerprinting to find code blocks similar to a given file or text.
Compares term weights, concept coverage, and bigram patterns to identify similar code.

Usage:
    python scripts/find_similar.py path/to/file.py
    python scripts/find_similar.py path/to/file.py --top 10
    python scripts/find_similar.py path/to/file.py --verbose
    python scripts/find_similar.py --text "def compute_pagerank(graph):"
    python scripts/find_similar.py file.py --explain  # Show why they're similar
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer


def get_file_content(file_path: str, processor: CorticalTextProcessor) -> Tuple[str, str]:
    """
    Get file content from the indexed corpus.

    Args:
        file_path: Path to the file (relative or absolute)
        processor: CorticalTextProcessor instance

    Returns:
        Tuple of (matched_doc_id, content)

    Raises:
        FileNotFoundError: If file not found in corpus
    """
    # Normalize path
    file_path = file_path.replace('\\', '/')

    # Try exact match first
    if file_path in processor.documents:
        return file_path, processor.documents[file_path]

    # Try without leading './'
    if file_path.startswith('./'):
        clean_path = file_path[2:]
        if clean_path in processor.documents:
            return clean_path, processor.documents[clean_path]

    # Try matching by suffix (in case of absolute vs relative paths)
    for doc_id, content in processor.documents.items():
        if doc_id.endswith(file_path) or file_path.endswith(doc_id):
            return doc_id, content

    # Not found
    raise FileNotFoundError(
        f"File not found in corpus: {file_path}\n"
        f"Available files: {len(processor.documents)}\n"
        f"Try running: python scripts/index_codebase.py"
    )


def find_similar_code(
    processor: CorticalTextProcessor,
    query_text: str,
    query_file: str = None,
    top_n: int = 5,
    chunk_size: int = 400,
    min_similarity: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Find code similar to the query text.

    Args:
        processor: CorticalTextProcessor instance
        query_text: Text to find similar code for
        query_file: Optional source file (to exclude from results)
        top_n: Number of results to return
        chunk_size: Size of chunks to compare
        min_similarity: Minimum similarity threshold (0.0-1.0)

    Returns:
        List of result dicts with similarity scores, sorted by similarity
    """
    # Get fingerprint of query text
    query_fp = processor.get_fingerprint(query_text, top_n=20)

    results = []

    # Compare against all documents in chunks
    for doc_id, doc_content in processor.documents.items():
        # Skip the query file itself
        if query_file and doc_id == query_file:
            continue

        # Chunk the document and compare each chunk
        doc_len = len(doc_content)
        overlap = chunk_size // 4  # 25% overlap

        for start in range(0, doc_len, chunk_size - overlap):
            end = min(start + chunk_size, doc_len)
            chunk = doc_content[start:end]

            # Skip very short chunks
            if len(chunk.strip()) < 50:
                continue

            # Get fingerprint and compare
            chunk_fp = processor.get_fingerprint(chunk, top_n=20)
            comparison = processor.compare_fingerprints(query_fp, chunk_fp)

            similarity = comparison.get('overall_similarity', 0.0)

            # Only include results above threshold
            if similarity >= min_similarity:
                # Find line number for this chunk
                line_num = doc_content[:start].count('\n') + 1

                results.append({
                    'file': doc_id,
                    'line': line_num,
                    'passage': chunk,
                    'similarity': similarity,
                    'comparison': comparison,
                    'reference': f"{doc_id}:{line_num}"
                })

    # Sort by similarity descending
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]


def get_doc_type_label(doc_id: str) -> str:
    """Get a display label for document type."""
    if doc_id.endswith('.md'):
        return 'DOC' if doc_id.startswith('docs/') else 'MD'
    elif doc_id.startswith('tests/'):
        return 'TEST'
    elif doc_id.endswith('.py'):
        return 'CODE'
    return 'FILE'


def format_passage(passage: str, max_lines: int = 10, max_width: int = 80) -> str:
    """Format a passage for display."""
    lines = passage.split('\n')
    formatted = []

    for i, line in enumerate(lines[:max_lines]):
        if len(line) > max_width:
            line = line[:max_width - 3] + '...'
        formatted.append(f"  {line}")

    if len(lines) > max_lines:
        formatted.append(f"  ... ({len(lines) - max_lines} more lines)")

    return '\n'.join(formatted)


def display_results(
    results: List[Dict[str, Any]],
    query_source: str,
    verbose: bool = False,
    explain: bool = False
):
    """
    Display similarity results.

    Args:
        results: List of result dicts from find_similar_code()
        query_source: Description of the query (file path or "text snippet")
        verbose: Show full passage text
        explain: Show detailed explanation of similarity
    """
    if not results:
        print(f"\nNo similar code found for: {query_source}")
        return

    print(f"\n{'=' * 70}")
    print(f"Code similar to: {query_source}")
    print(f"{'=' * 70}\n")

    for i, result in enumerate(results, 1):
        doc_type = get_doc_type_label(result['file'])
        print(f"[{i}] [{doc_type}] {result['reference']}")
        print(f"    Similarity: {result['similarity']:.1%}")

        # Show shared terms
        comparison = result['comparison']
        shared_terms = list(comparison.get('shared_terms', []))[:5]
        if shared_terms:
            print(f"    Shared terms: {', '.join(shared_terms)}")

        print("─" * 70)

        # Show passage
        if verbose:
            print(format_passage(result['passage'], max_lines=20))
        else:
            print(format_passage(result['passage'], max_lines=5))

        # Show explanation if requested
        if explain:
            print("\n  Similarity breakdown:")
            print(f"    Term overlap: {comparison.get('term_similarity', 0):.1%}")
            print(f"    Concept overlap: {comparison.get('concept_similarity', 0):.1%}")
            print(f"    Bigram overlap: {comparison.get('bigram_similarity', 0):.1%}")

            shared_concepts = list(comparison.get('shared_concepts', []))[:3]
            if shared_concepts:
                print(f"    Shared concepts: {', '.join(shared_concepts)}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description='Find code similar to a file or text snippet',
        epilog="""
Examples:
  %(prog)s cortical/processor.py              # Find similar to file
  %(prog)s cortical/processor.py --top 10     # More results
  %(prog)s cortical/processor.py --verbose    # Show full passages
  %(prog)s cortical/processor.py --explain    # Show why they're similar
  %(prog)s --text "def compute_pagerank"      # Find similar to text
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('file', nargs='?', help='File path to find similar code for')
    parser.add_argument('--text', '-t', help='Text snippet to find similar code for')
    parser.add_argument('--corpus', '-c', default='corpus_dev.pkl',
                        help='Corpus file path (default: corpus_dev.pkl)')
    parser.add_argument('--top', '-n', type=int, default=5,
                        help='Number of results (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show full passage text')
    parser.add_argument('--explain', '-e', action='store_true',
                        help='Explain why results are similar')
    parser.add_argument('--min-similarity', type=float, default=0.1,
                        help='Minimum similarity threshold 0-1 (default: 0.1)')
    parser.add_argument('--chunk-size', type=int, default=400,
                        help='Size of text chunks to compare (default: 400)')

    args = parser.parse_args()

    # Validate inputs
    if not args.file and not args.text:
        parser.error("Must provide either FILE or --text")

    if args.file and args.text:
        parser.error("Cannot use both FILE and --text")

    # Load corpus
    base_path = Path(__file__).parent.parent
    corpus_path = base_path / args.corpus

    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        print("Run 'python scripts/index_codebase.py' first to create it.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    processor = CorticalTextProcessor.load(str(corpus_path))
    print(f"Loaded {len(processor.documents)} documents\n")

    # Get query text and source
    if args.file:
        try:
            matched_file, file_content = get_file_content(args.file, processor)
            query_text = file_content
            query_source = matched_file
            query_file = matched_file
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        query_text = args.text
        query_source = "text snippet"
        query_file = None

    # Find similar code
    print(f"Finding similar code...")
    results = find_similar_code(
        processor,
        query_text,
        query_file=query_file,
        top_n=args.top,
        chunk_size=args.chunk_size,
        min_similarity=args.min_similarity
    )

    # Display results
    display_results(results, query_source, verbose=args.verbose, explain=args.explain)


if __name__ == '__main__':
    main()
