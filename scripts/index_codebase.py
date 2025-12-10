#!/usr/bin/env python3
"""
Index the Cortical Text Processor codebase for dog-fooding.

This script indexes all Python files and documentation to enable
semantic search over the codebase using the Cortical Text Processor itself.

Usage:
    python scripts/index_codebase.py [--output corpus_dev.pkl]
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor


def get_python_files(base_path: Path) -> list:
    """Get all Python files in cortical/ and tests/ directories."""
    files = []
    for directory in ['cortical', 'tests']:
        dir_path = base_path / directory
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                if not py_file.name.startswith('__'):
                    files.append(py_file)
    return sorted(files)


def get_doc_files(base_path: Path) -> list:
    """Get documentation files from root and docs/ directory."""
    # Root documentation files
    root_docs = ['CLAUDE.md', 'TASK_LIST.md', 'README.md', 'KNOWLEDGE_TRANSFER.md']
    files = []
    for doc in root_docs:
        doc_path = base_path / doc
        if doc_path.exists():
            files.append(doc_path)

    # Intelligence documentation in docs/
    docs_dir = base_path / 'docs'
    if docs_dir.exists():
        for md_file in docs_dir.glob('*.md'):
            files.append(md_file)

    return files


def create_doc_id(file_path: Path, base_path: Path) -> str:
    """Create a document ID from file path."""
    rel_path = file_path.relative_to(base_path)
    return str(rel_path)


def index_file(processor: CorticalTextProcessor, file_path: Path, base_path: Path) -> dict:
    """Index a single file with line number metadata."""
    doc_id = create_doc_id(file_path, base_path)

    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  Warning: Could not read {doc_id}: {e}")
        return None

    # Create metadata with file info
    metadata = {
        'file_path': str(file_path),
        'relative_path': doc_id,
        'file_type': file_path.suffix,
        'line_count': content.count('\n') + 1,
    }

    # For Python files, extract additional metadata
    if file_path.suffix == '.py':
        metadata['language'] = 'python'
        # Count functions and classes
        metadata['function_count'] = content.count('\ndef ')
        metadata['class_count'] = content.count('\nclass ')

    processor.process_document(doc_id, content, metadata=metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Index the codebase for semantic search')
    parser.add_argument('--output', '-o', default='corpus_dev.pkl',
                        help='Output file path (default: corpus_dev.pkl)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    output_path = base_path / args.output

    print("Cortical Text Processor - Codebase Indexer")
    print("=" * 50)

    # Initialize processor
    processor = CorticalTextProcessor()

    # Get files to index
    python_files = get_python_files(base_path)
    doc_files = get_doc_files(base_path)
    all_files = python_files + doc_files

    print(f"\nFound {len(python_files)} Python files and {len(doc_files)} documentation files")

    # Index all files
    print("\nIndexing files...")
    indexed = 0
    total_lines = 0

    for file_path in all_files:
        if args.verbose:
            print(f"  Indexing: {create_doc_id(file_path, base_path)}")

        metadata = index_file(processor, file_path, base_path)
        if metadata:
            indexed += 1
            total_lines += metadata.get('line_count', 0)

    print(f"  Indexed {indexed} files ({total_lines:,} total lines)")

    # Compute all analysis
    print("\nComputing analysis...")
    processor.compute_all(
        build_concepts=True,
        pagerank_method='semantic',
        connection_strategy='hybrid',
        verbose=args.verbose
    )

    # Extract semantic relations
    print("Extracting semantic relations...")
    processor.extract_corpus_semantics(
        use_pattern_extraction=True,
        verbose=args.verbose
    )

    # Print statistics
    print("\nCorpus Statistics:")
    print(f"  Documents: {len(processor.documents)}")
    print(f"  Tokens (Layer 0): {processor.layers[0].column_count()}")
    print(f"  Bigrams (Layer 1): {processor.layers[1].column_count()}")
    print(f"  Concepts (Layer 2): {processor.layers[2].column_count()}")
    print(f"  Semantic relations: {len(processor.semantic_relations)}")

    # Save corpus
    print(f"\nSaving corpus to {output_path}...")
    processor.save(str(output_path))

    file_size = output_path.stat().st_size / 1024
    print(f"  Saved ({file_size:.1f} KB)")

    print("\nDone! Use search_codebase.py to query the indexed corpus.")


if __name__ == '__main__':
    main()
