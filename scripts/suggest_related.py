#!/usr/bin/env python3
"""
Suggest Related Files - Find Contextually Related Code

Suggests files related to a given file based on:
- Shared concepts and terms
- Import relationships (files that import or are imported by this file)
- Semantic similarity via fingerprinting
- Co-occurrence in concept clusters

Usage:
    python scripts/suggest_related.py path/to/file.py
    python scripts/suggest_related.py path/to/file.py --top 10
    python scripts/suggest_related.py path/to/file.py --verbose
    python scripts/suggest_related.py path/to/file.py --imports-only
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
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


def extract_imports(file_content: str, file_path: str) -> Set[str]:
    """
    Extract import statements from Python code.

    Args:
        file_content: Python source code
        file_path: Path to the file (for relative imports)

    Returns:
        Set of module names that are imported
    """
    imports = set()

    # Match "import x", "import x as y", "import x.y.z"
    import_pattern = re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE)
    for match in import_pattern.finditer(file_content):
        module = match.group(1)
        imports.add(module)

    # Match "from x import y", "from x.y import z"
    from_pattern = re.compile(r'^\s*from\s+([\w.]+)\s+import', re.MULTILINE)
    for match in from_pattern.finditer(file_content):
        module = match.group(1)
        imports.add(module)

    return imports


def find_import_related_files(
    doc_id: str,
    doc_content: str,
    processor: CorticalTextProcessor
) -> List[Tuple[str, str]]:
    """
    Find files related by import relationships.

    Args:
        doc_id: Document ID
        doc_content: Document content
        processor: CorticalTextProcessor instance

    Returns:
        List of (related_file, relationship_type) tuples
    """
    related = []
    imports = extract_imports(doc_content, doc_id)

    # Convert import names to file paths
    # E.g., "cortical.processor" -> "cortical/processor.py"
    for imp in imports:
        # Convert dotted name to path
        potential_paths = [
            imp.replace('.', '/') + '.py',
            imp.replace('.', '/') + '/__init__.py'
        ]

        for path in potential_paths:
            if path in processor.documents:
                related.append((path, 'imports'))

    # Find files that import this file
    # Convert this file's path to a module name
    # E.g., "cortical/processor.py" -> "cortical.processor"
    if doc_id.endswith('.py'):
        module_name = doc_id[:-3].replace('/', '.')

        for other_id, other_content in processor.documents.items():
            if other_id == doc_id:
                continue

            other_imports = extract_imports(other_content, other_id)

            # Check if this module is imported
            for imp in other_imports:
                if imp == module_name or imp.startswith(module_name + '.'):
                    related.append((other_id, 'imported_by'))
                    break

    return related


def find_concept_related_files(
    doc_id: str,
    processor: CorticalTextProcessor,
    top_n: int = 10
) -> List[Tuple[str, float, str]]:
    """
    Find files that share concept clusters.

    Args:
        doc_id: Document ID
        processor: CorticalTextProcessor instance
        top_n: Number of results

    Returns:
        List of (file, score, concept) tuples
    """
    layer2 = processor.layers[CorticalLayer.CONCEPTS]

    # Find concepts this file appears in
    file_concepts = []
    for concept_col in layer2.minicolumns.values():
        if doc_id in concept_col.document_ids:
            file_concepts.append(concept_col)

    # Score other files by shared concepts
    file_scores = defaultdict(float)
    file_shared_concepts = defaultdict(list)

    for concept_col in file_concepts:
        concept_weight = concept_col.pagerank if concept_col.pagerank > 0 else 1.0

        for other_doc in concept_col.document_ids:
            if other_doc != doc_id:
                file_scores[other_doc] += concept_weight
                file_shared_concepts[other_doc].append(concept_col.content[:50])

    # Convert to list and sort
    results = []
    for file, score in file_scores.items():
        # Get the most significant shared concept
        concepts = file_shared_concepts[file]
        primary_concept = concepts[0] if concepts else "shared concept"
        results.append((file, score, primary_concept))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def find_semantic_related_files(
    doc_id: str,
    doc_content: str,
    processor: CorticalTextProcessor,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Find files with similar semantic fingerprints.

    Args:
        doc_id: Document ID
        doc_content: Document content
        processor: CorticalTextProcessor instance
        top_n: Number of results

    Returns:
        List of (file, similarity) tuples
    """
    # Get fingerprint of source file
    source_fp = processor.get_fingerprint(doc_content, top_n=20)

    results = []

    # Compare against all other documents
    for other_id, other_content in processor.documents.items():
        if other_id == doc_id:
            continue

        other_fp = processor.get_fingerprint(other_content, top_n=20)
        comparison = processor.compare_fingerprints(source_fp, other_fp)
        similarity = comparison.get('overall_similarity', 0.0)

        if similarity > 0.05:  # Minimum threshold
            results.append((other_id, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def suggest_related_files(
    doc_id: str,
    doc_content: str,
    processor: CorticalTextProcessor,
    top_n: int = 10,
    imports_only: bool = False
) -> Dict[str, List[Any]]:
    """
    Suggest files related to the given file.

    Args:
        doc_id: Document ID
        doc_content: Document content
        processor: CorticalTextProcessor instance
        top_n: Number of suggestions per category
        imports_only: Only show import-based relationships

    Returns:
        Dict with categories of related files
    """
    results = {}

    # Import relationships (always include)
    import_related = find_import_related_files(doc_id, doc_content, processor)
    results['imports'] = import_related

    if not imports_only:
        # Concept-based relationships
        concept_related = find_concept_related_files(doc_id, processor, top_n=top_n)
        results['concepts'] = concept_related

        # Semantic similarity
        semantic_related = find_semantic_related_files(doc_id, doc_content, processor, top_n=top_n)
        results['semantic'] = semantic_related

    return results


def get_doc_type_label(doc_id: str) -> str:
    """Get a display label for document type."""
    if doc_id.endswith('.md'):
        return 'DOC' if doc_id.startswith('docs/') else 'MD'
    elif doc_id.startswith('tests/'):
        return 'TEST'
    elif doc_id.endswith('.py'):
        return 'CODE'
    return 'FILE'


def display_suggestions(
    suggestions: Dict[str, List[Any]],
    source: str,
    verbose: bool = False
):
    """
    Display file suggestions.

    Args:
        suggestions: Dict from suggest_related_files()
        source: Source file path
        verbose: Show detailed information
    """
    print(f"\n{'=' * 70}")
    print(f"Related Files: {source}")
    print(f"{'=' * 70}\n")

    # Import relationships
    import_related = suggestions.get('imports', [])
    if import_related:
        imports = [f for f, rel in import_related if rel == 'imports']
        imported_by = [f for f, rel in import_related if rel == 'imported_by']

        if imports:
            print(f"📦 Imports ({len(imports)} files):")
            for file in imports[:10]:
                doc_type = get_doc_type_label(file)
                print(f"  [{doc_type}] {file}")
            if len(imports) > 10:
                print(f"  ... and {len(imports) - 10} more")
            print()

        if imported_by:
            print(f"📥 Imported By ({len(imported_by)} files):")
            for file in imported_by[:10]:
                doc_type = get_doc_type_label(file)
                print(f"  [{doc_type}] {file}")
            if len(imported_by) > 10:
                print(f"  ... and {len(imported_by) - 10} more")
            print()

    # Concept-based relationships
    concept_related = suggestions.get('concepts', [])
    if concept_related:
        print(f"💡 Shared Concepts ({len(concept_related)} files):")
        for file, score, concept in concept_related[:8]:
            doc_type = get_doc_type_label(file)
            if verbose:
                print(f"  [{doc_type}] {file}")
                print(f"    Score: {score:.2f} | Concept: {concept}")
            else:
                print(f"  [{doc_type}] {file:50s} ({score:.2f})")
        print()

    # Semantic similarity
    semantic_related = suggestions.get('semantic', [])
    if semantic_related:
        print(f"🔍 Semantically Similar ({len(semantic_related)} files):")
        for file, similarity in semantic_related[:8]:
            doc_type = get_doc_type_label(file)
            bar_length = int(similarity * 30)
            bar = '█' * bar_length
            if verbose:
                print(f"  [{doc_type}] {file}")
                print(f"    Similarity: {bar} {similarity:.1%}")
            else:
                print(f"  [{doc_type}] {file:50s} {similarity:.1%}")
        print()

    # Summary
    total = len(import_related) + len(concept_related) + len(semantic_related)
    print(f"📊 Total: {total} related files found")


def main():
    parser = argparse.ArgumentParser(
        description='Suggest files related to a given file',
        epilog="""
Examples:
  %(prog)s cortical/processor.py           # Find related files
  %(prog)s cortical/processor.py --top 15  # More suggestions
  %(prog)s cortical/processor.py --verbose # Detailed information
  %(prog)s cortical/processor.py --imports-only  # Only import relationships
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('file', help='File path to find related files for')
    parser.add_argument('--corpus', '-c', default='corpus_dev.pkl',
                        help='Corpus file path (default: corpus_dev.pkl)')
    parser.add_argument('--top', '-n', type=int, default=10,
                        help='Number of suggestions per category (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed information')
    parser.add_argument('--imports-only', '-i', action='store_true',
                        help='Only show import-based relationships')

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
    print(f"Loaded {len(processor.documents)} documents")

    # Get file content
    try:
        matched_file, file_content = get_file_content(args.file, processor)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Find related files
    print(f"Finding related files for {matched_file}...")
    suggestions = suggest_related_files(
        matched_file,
        file_content,
        processor,
        top_n=args.top,
        imports_only=args.imports_only
    )

    # Display results
    display_suggestions(suggestions, matched_file, verbose=args.verbose)


if __name__ == '__main__':
    main()
