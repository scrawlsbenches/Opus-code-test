"""
Scan command - Scan directory for suspicious comments.

Uses Bloom Filter for fast pre-screening, Naive Bayes for classification,
and Trie for marker detection.
"""

import os
from typing import Any

from ._base import (
    NAIVE_BAYES_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_model,
    tokenize_comment,
    print_header,
    print_separator,
)


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'scan',
        help='Scan for suspicious comments'
    )
    parser.add_argument('directory', help='Directory to scan')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f'Minimum confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})'
    )


def run(args: Any) -> None:
    """Execute the scan command."""
    from cortical.audits import (
        find_python_files,
        extract_comments_from_file,
        SuspiciousCommentFilter,
        CommentMarkerTrie,
        SUSPICIOUS_PATTERNS,
        COMMENT_MARKERS,
    )
    from cortical.tokenizer import Tokenizer

    print(f"Scanning {args.directory} for suspicious comments...")
    print_separator()

    # Initialize tokenizer
    tokenizer = Tokenizer()

    # Initialize Bloom Filter for pre-screening
    bloom = SuspiciousCommentFilter(
        expected_patterns=len(SUSPICIOUS_PATTERNS),
        fp_rate=0.01
    )
    for pattern in SUSPICIOUS_PATTERNS:
        bloom.add(pattern.lower())

    # Try to load trained Naive Bayes model
    classifier = load_model(NAIVE_BAYES_MODEL)
    if classifier is None:
        print("Warning: No trained classifier found.")
        print("Run 'audit train' first for better results.")
        print("Proceeding with pattern-based detection only.\n")

    # Initialize Trie for marker detection
    trie = CommentMarkerTrie()
    for marker in COMMENT_MARKERS:
        trie.insert(marker.lower())

    # Scan files
    python_files = find_python_files(args.directory)
    print(f"Found {len(python_files)} Python files\n")

    suspicious_findings = []
    total_comments = 0

    for file_path in python_files:
        comments = extract_comments_from_file(file_path)
        total_comments += len(comments)

        for line_num, comment in comments:
            # Check Bloom filter for fast pre-screening
            comment_lower = comment.lower()
            if bloom.probably_suspicious(comment_lower):
                # Classify with Naive Bayes if available
                confidence = 0.0
                classification = "suspicious"

                if classifier is not None:
                    tokens = tokenize_comment(comment, tokenizer)
                    if tokens:
                        try:
                            probs = classifier.predict_proba(tokens)
                            if "misleading" in probs:
                                classification = "misleading"
                                confidence = probs["misleading"]
                            elif "suspicious" in probs:
                                classification = "suspicious"
                                confidence = probs["suspicious"]
                        except Exception:
                            # Classifier might not be trained on all classes
                            pass

                # Detect markers (check both with and without colon)
                markers = []
                for marker in COMMENT_MARKERS:
                    # Check for marker with colon (e.g., "FIXME:")
                    if marker.lower() in comment_lower:
                        markers.append(marker)
                    # Check for marker without colon (e.g., "FIXME" in "(FIXME)")
                    else:
                        marker_base = marker.rstrip(':').lower()
                        if marker_base in comment_lower:
                            markers.append(marker)

                # Only report if confidence >= min_confidence or has markers
                # Low confidence without markers is likely a false positive
                min_confidence = getattr(args, 'confidence', DEFAULT_CONFIDENCE_THRESHOLD)
                if confidence < min_confidence and not markers:
                    continue

                # Record finding
                finding = {
                    'file': file_path,
                    'line': line_num,
                    'comment': comment,
                    'classification': classification,
                    'confidence': confidence,
                    'markers': markers,
                }
                suspicious_findings.append(finding)

    # Report findings
    print(f"Scanned {total_comments} comments")
    print(f"Found {len(suspicious_findings)} suspicious comments\n")

    if suspicious_findings:
        print("SUSPICIOUS FINDINGS:")
        print_separator()

        for finding in suspicious_findings:
            rel_path = os.path.relpath(finding['file'], args.directory)
            print(f"\n[{finding['classification'].upper()}] {rel_path}:{finding['line']}")
            print(f"  Comment: {finding['comment']}")

            if finding['confidence'] > 0:
                print(f"  Classification: {finding['classification']} "
                      f"({finding['confidence']:.0%} confidence)")

            if finding['markers']:
                print(f"  Markers: {', '.join(finding['markers'])}")

    print_separator()
    print(f"Bloom filter false positive rate: {bloom.false_positive_rate():.2%}")
