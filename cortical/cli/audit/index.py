"""
Index command - Build search indexes for comments.

Builds:
- Inverted Index for term search
- Trie for marker search
- LSH index for similarity search
"""

import os
from typing import Any

from ._base import (
    INDEX_MODEL,
    TRIE_MODEL,
    LSH_MODEL,
    save_model,
    tokenize_comment,
    print_header,
    print_separator,
)


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'index',
        help='Build search indexes'
    )
    parser.add_argument('directory', help='Directory to index')


def run(args: Any) -> None:
    """Execute the index command."""
    from cortical.audits import (
        find_python_files,
        extract_comments_from_file,
        AuditInvertedIndex,
        CommentMarkerTrie,
        SimilarCommentFinder,
        COMMENT_MARKERS,
    )
    from cortical.tokenizer import Tokenizer

    print(f"Building search indexes for {args.directory}...")
    print_separator()

    # Initialize data structures
    tokenizer = Tokenizer()
    inverted_index = AuditInvertedIndex()
    marker_trie = CommentMarkerTrie()
    lsh_index = SimilarCommentFinder(num_hashes=100, num_bands=20)

    # Scan files
    python_files = find_python_files(args.directory)
    print(f"Found {len(python_files)} Python files\n")

    total_comments = 0

    for file_path in python_files:
        comments = extract_comments_from_file(file_path)

        for line_num, comment in comments:
            total_comments += 1

            # Create unique comment ID
            rel_path = os.path.relpath(file_path, args.directory)
            comment_id = f"{rel_path}:{line_num}"

            # Tokenize comment
            tokens = tokenize_comment(comment, tokenizer)

            # Add to inverted index
            for pos, token in enumerate(tokens):
                inverted_index.add(token, comment_id, pos)

            # Add to LSH index
            if tokens:
                lsh_index.add(comment_id, set(tokens))

            # Add markers to Trie
            comment_lower = comment.lower()
            for marker in COMMENT_MARKERS:
                if marker.lower() in comment_lower:
                    marker_trie.insert(marker.lower(), count=1)

    print(f"Indexed {total_comments} comments")

    # Save indexes
    print("\nSaving indexes...")
    save_model(inverted_index, INDEX_MODEL)
    save_model(marker_trie, TRIE_MODEL)
    save_model(lsh_index, LSH_MODEL)

    print_separator()
    print("Indexing complete!")
    print(f"Total comments indexed: {total_comments}")
