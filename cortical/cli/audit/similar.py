"""
Similar command - Find similar comments using LSH.

Uses Locality-Sensitive Hashing to find comments that are similar to
a query comment.
"""

from typing import Any

from ._base import (
    LSH_MODEL,
    load_model,
    tokenize_comment,
    print_header,
    print_separator,
)


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'similar',
        help='Find similar comments using LSH'
    )
    parser.add_argument('comment', help='Comment to find similar matches for')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Similarity threshold (default: 0.3)'
    )


def run(args: Any) -> None:
    """Execute the similar command."""
    from cortical.tokenizer import Tokenizer

    print(f'Finding comments similar to: "{args.comment}"')
    print_separator()

    # Load LSH index
    lsh_index = load_model(LSH_MODEL)
    if lsh_index is None:
        print("Error: No LSH index found.")
        print("Run 'audit index <directory>' first to build the index.")
        return

    # Tokenize query comment
    tokenizer = Tokenizer()
    query_tokens = set(tokenize_comment(args.comment, tokenizer))

    if not query_tokens:
        print("Error: No tokens in query comment")
        return

    print(f"Query tokens: {sorted(query_tokens)}\n")

    # Query LSH index
    threshold = getattr(args, 'threshold', 0.3)
    results = lsh_index.query(query_tokens, threshold=threshold)

    if not results:
        print(f"No similar comments found (threshold={threshold})")
        return

    print(f"Found {len(results)} similar comments:\n")
    print("SIMILAR COMMENTS:")
    print_separator()

    for comment_id, similarity in results:
        print(f"\n{comment_id}")
        print(f"  Similarity: {similarity:.1%}")

    print_separator()
