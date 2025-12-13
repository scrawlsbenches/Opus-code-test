#!/usr/bin/env python3
"""
Explain This Code - Semantic Analysis and Context

Uses concept clusters, semantic relations, and term analysis to explain
what a code file is about, how it relates to other files, and its key concepts.

Usage:
    python scripts/explain_code.py path/to/file.py
    python scripts/explain_code.py path/to/file.py --verbose
    python scripts/explain_code.py path/to/file.py --relations  # Show semantic relations
    python scripts/explain_code.py --text "your code here"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer


def get_file_content(file_path: str, processor: CorticalTextProcessor) -> Tuple[str, str]:
    """
    Get file content from the indexed corpus.

    Args:
        file_path: Path to the file
        processor: CorticalTextProcessor instance

    Returns:
        Tuple of (matched_doc_id, content)

    Raises:
        FileNotFoundError: If file not found in corpus
    """
    # Normalize path
    file_path = file_path.replace('\\', '/')

    # Try exact match
    if file_path in processor.documents:
        return file_path, processor.documents[file_path]

    # Try without leading './'
    if file_path.startswith('./'):
        clean_path = file_path[2:]
        if clean_path in processor.documents:
            return clean_path, processor.documents[clean_path]

    # Try suffix matching
    for doc_id, content in processor.documents.items():
        if doc_id.endswith(file_path) or file_path.endswith(doc_id):
            return doc_id, content

    raise FileNotFoundError(f"File not found in corpus: {file_path}")


def analyze_code(
    processor: CorticalTextProcessor,
    doc_id: str,
    text: str
) -> Dict[str, Any]:
    """
    Analyze code and extract semantic information.

    Args:
        processor: CorticalTextProcessor instance
        doc_id: Document identifier
        text: Code text to analyze

    Returns:
        Dict with analysis results including:
        - key_terms: Important terms with weights
        - concepts: Concept memberships
        - related_docs: Related documents
        - semantic_relations: Relevant semantic relations
        - fingerprint: Semantic fingerprint
    """
    # Get semantic fingerprint
    fingerprint = processor.get_fingerprint(text, top_n=30)

    # Get key terms from fingerprint
    key_terms = fingerprint.get('top_terms', [])[:15]

    # Get concept coverage
    concepts = fingerprint.get('concepts', {})
    top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Find related documents using key terms
    # Build a query from top terms
    query_terms = [term for term, _ in key_terms[:10]]
    query = ' '.join(query_terms)

    # Find similar documents
    related_docs = processor.find_documents_for_query(query, top_n=10)

    # Filter out the source document
    related_docs = [(d, s) for d, s in related_docs if d != doc_id][:5]

    # Find relevant semantic relations
    # Check if any key terms appear in semantic relations
    key_term_set = {term for term, _ in key_terms}
    relevant_relations = []

    for t1, rel, t2, weight in processor.semantic_relations:
        if t1 in key_term_set or t2 in key_term_set:
            relevant_relations.append((t1, rel, t2, weight))

    # Sort by weight and limit
    relevant_relations.sort(key=lambda x: x[3], reverse=True)
    relevant_relations = relevant_relations[:15]

    # Get bigrams
    top_bigrams = fingerprint.get('bigrams', {})
    top_bigram_list = sorted(top_bigrams.items(), key=lambda x: x[1], reverse=True)[:10]

    # Find which concepts this file contributes to
    layer2 = processor.layers[CorticalLayer.CONCEPTS]
    file_concepts = []

    for concept_col in layer2.minicolumns.values():
        if doc_id in concept_col.document_ids:
            # Calculate contribution score (how important is this doc to this concept)
            # Use the concept's occurrence count vs this doc's contribution
            contribution = concept_col.doc_occurrence_counts.get(doc_id, 0)
            if contribution > 0:
                file_concepts.append({
                    'concept': concept_col.content[:80],  # Truncate long concepts
                    'contribution': contribution,
                    'total_occurrences': concept_col.occurrence_count
                })

    # Sort by contribution
    file_concepts.sort(key=lambda x: x['contribution'], reverse=True)

    return {
        'key_terms': key_terms,
        'concepts': top_concepts,
        'file_concepts': file_concepts[:10],
        'bigrams': top_bigram_list,
        'related_docs': related_docs,
        'semantic_relations': relevant_relations,
        'fingerprint': fingerprint,
        'term_count': fingerprint.get('term_count', 0)
    }


def display_analysis(
    analysis: Dict[str, Any],
    source: str,
    verbose: bool = False,
    show_relations: bool = False
):
    """
    Display code analysis results.

    Args:
        analysis: Analysis dict from analyze_code()
        source: Source identifier (file path)
        verbose: Show detailed information
        show_relations: Show semantic relations
    """
    print(f"\n{'=' * 70}")
    print(f"Code Analysis: {source}")
    print(f"{'=' * 70}\n")

    # Overview
    print("📊 Overview:")
    print(f"  Unique terms: {analysis['term_count']}")
    print(f"  Key terms identified: {len(analysis['key_terms'])}")
    print(f"  Concepts detected: {len(analysis['concepts'])}")
    print(f"  Related documents: {len(analysis['related_docs'])}")
    print()

    # Key Terms
    print("🔑 Key Terms (by importance):")
    for i, (term, weight) in enumerate(analysis['key_terms'][:10], 1):
        bar_length = int(weight * 30)
        bar = '█' * bar_length
        print(f"  {i:2d}. {term:20s} {bar} {weight:.3f}")
    print()

    # Concepts
    if analysis['concepts']:
        print("💡 Primary Concepts:")
        for concept, score in analysis['concepts'][:8]:
            print(f"  • {concept:30s} ({score:.3f})")
        print()

    # File's contribution to concepts (concept clusters it appears in)
    if analysis['file_concepts']:
        print("🎯 Concept Clusters (this file contributes to):")
        for fc in analysis['file_concepts'][:5]:
            pct = (fc['contribution'] / fc['total_occurrences']) * 100
            print(f"  • {fc['concept']}")
            print(f"    Contribution: {fc['contribution']}/{fc['total_occurrences']} ({pct:.1f}%)")
        print()

    # Bigrams (key phrases)
    if verbose and analysis['bigrams']:
        print("📝 Key Phrases (bigrams):")
        for bigram, weight in analysis['bigrams'][:8]:
            print(f"  • \"{bigram}\" ({weight:.3f})")
        print()

    # Related Documents
    if analysis['related_docs']:
        print("🔗 Related Documents:")
        for doc_id, score in analysis['related_docs']:
            doc_type = get_doc_type_label(doc_id)
            print(f"  [{doc_type}] {doc_id:50s} (score: {score:.2f})")
        print()

    # Semantic Relations
    if show_relations and analysis['semantic_relations']:
        print("🔀 Semantic Relations:")
        for t1, rel, t2, weight in analysis['semantic_relations'][:10]:
            print(f"  {t1} --[{rel}]--> {t2} ({weight:.2f})")
        print()

    # Fingerprint summary
    if verbose:
        fp = analysis['fingerprint']
        print("🔍 Semantic Fingerprint Summary:")
        print(f"  Terms in fingerprint: {len(fp.get('terms', {}))}")
        print(f"  Concepts in fingerprint: {len(fp.get('concepts', {}))}")
        print(f"  Bigrams in fingerprint: {len(fp.get('bigrams', {}))}")
        print()


def get_doc_type_label(doc_id: str) -> str:
    """Get a display label for document type."""
    if doc_id.endswith('.md'):
        return 'DOC' if doc_id.startswith('docs/') else 'MD'
    elif doc_id.startswith('tests/'):
        return 'TEST'
    elif doc_id.endswith('.py'):
        return 'CODE'
    return 'FILE'


def main():
    parser = argparse.ArgumentParser(
        description='Explain what a code file is about using semantic analysis',
        epilog="""
Examples:
  %(prog)s cortical/processor.py           # Analyze a file
  %(prog)s cortical/processor.py --verbose # Detailed analysis
  %(prog)s cortical/processor.py --relations  # Show semantic relations
  %(prog)s --text "your code snippet"      # Analyze text directly
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('file', nargs='?', help='File path to analyze')
    parser.add_argument('--text', '-t', help='Text to analyze directly')
    parser.add_argument('--corpus', '-c', default='corpus_dev.pkl',
                        help='Corpus file path (default: corpus_dev.pkl)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed information')
    parser.add_argument('--relations', '-r', action='store_true',
                        help='Show semantic relations')

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
    print(f"Loaded {len(processor.documents)} documents")

    # Get text and source
    if args.file:
        try:
            matched_file, file_content = get_file_content(args.file, processor)
            text = file_content
            source = matched_file
            doc_id = matched_file
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        text = args.text
        source = "text snippet"
        doc_id = None

    # Analyze code
    print(f"Analyzing {source}...")
    analysis = analyze_code(processor, doc_id, text)

    # Display results
    display_analysis(
        analysis,
        source,
        verbose=args.verbose,
        show_relations=args.relations
    )


if __name__ == '__main__':
    main()
