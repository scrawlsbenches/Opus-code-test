#!/usr/bin/env python3
"""
Unified Audit Tool for Cortical Codebase Maintenance

This tool integrates all 11 algorithm implementations from cortical/audits/algorithms
to help maintain code quality during refactoring.

Commands:
    scan <directory>      - Scan for suspicious comments using Bloom filter + Naive Bayes
    train <findings_dir>  - Train classifiers from labeled findings
    patterns <directory>  - Find repeated patterns using Suffix Array + Count-Min Sketch
    similar <comment>     - Find similar comments using LSH
    index <directory>     - Build search index using Inverted Index + Trie

Examples:
    python scripts/audit_tool.py scan cortical/
    python scripts/audit_tool.py train docs/audits/
    python scripts/audit_tool.py patterns cortical/got/
    python scripts/audit_tool.py similar "FUTURE: When CDG index is implemented"
    python scripts/audit_tool.py index cortical/
"""

import argparse
import os
import re
import sys
import pickle
import json
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional

# Add project root to sys.path to allow imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import algorithm implementations
from cortical.audits.algorithms import (
    AuditInvertedIndex,
    CommentDecisionTree,
    CommentMarkerTrie,
    CommentClassifier,
    FindingCluster,
    SuspiciousCommentFilter,
    CommentPatternFinder,
    TaskDAG,
    CommentMarkovChain,
    SimilarCommentFinder,
    PatternFrequencySketch,
)

# Import tokenizer
from cortical.tokenizer import Tokenizer


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Suspicious patterns for Bloom filter pre-screening
SUSPICIOUS_PATTERNS = [
    "will be implemented",
    "will be done",
    "will be replaced",
    "will be handled",
    "when cdg index is implemented",
    "when feature is ready",
    "placeholder",
    "stub",
    "not implemented yet",
    "coming soon",
    "tbd",
    "fixme later",
    "hack",
    "temporary fix",
    "workaround",
]

# Comment markers to index
COMMENT_MARKERS = [
    "FUTURE:",
    "TODO:",
    "FIXME:",
    "HACK:",
    "XXX:",
    "NOTE:",
    "WARNING:",
    "BUG:",
    "OPTIMIZE:",
    "REFACTOR:",
]

# Model storage paths
MODEL_DIR = Path(".audit_models")
BLOOM_MODEL = MODEL_DIR / "bloom_filter.pkl"
NAIVE_BAYES_MODEL = MODEL_DIR / "naive_bayes.pkl"
LSH_MODEL = MODEL_DIR / "lsh_index.pkl"
INDEX_MODEL = MODEL_DIR / "inverted_index.pkl"
TRIE_MODEL = MODEL_DIR / "marker_trie.pkl"


# ==============================================================================
# UTILITIES
# ==============================================================================

def extract_comments_from_file(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract comments from a Python file.

    Returns:
        List of (line_number, comment_text) tuples
    """
    comments = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Find Python comments (# ...)
                match = re.search(r'#\s*(.+)$', line)
                if match:
                    comment_text = match.group(1).strip()
                    if comment_text:  # Skip empty comments
                        comments.append((line_num, comment_text))
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)

    return comments


def find_python_files(directory: str) -> List[str]:
    """
    Recursively find all Python files in directory.

    Returns:
        List of absolute file paths
    """
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def tokenize_comment(comment: str, tokenizer: Tokenizer) -> List[str]:
    """
    Tokenize a comment using the Cortical tokenizer.

    Returns:
        List of tokens
    """
    # Remove common punctuation that adds noise
    cleaned = re.sub(r'[^\w\s]', ' ', comment)
    tokens = tokenizer.tokenize(cleaned, split_identifiers=True)
    return tokens


def save_model(obj, path: Path) -> None:
    """Save a model using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Model saved to {path}")


def load_model(path: Path):
    """Load a model using pickle."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# COMMAND: SCAN
# ==============================================================================

def cmd_scan(args):
    """
    Scan directory for suspicious comments.

    Uses:
    - Bloom Filter for fast pre-screening
    - Naive Bayes for classification
    - Trie for marker detection
    """
    print(f"Scanning {args.directory} for suspicious comments...")
    print("=" * 70)

    # Initialize tokenizer
    tokenizer = Tokenizer()

    # Initialize Bloom Filter for pre-screening
    bloom = SuspiciousCommentFilter(expected_patterns=len(SUSPICIOUS_PATTERNS), fp_rate=0.01)
    for pattern in SUSPICIOUS_PATTERNS:
        bloom.add(pattern.lower())

    # Try to load trained Naive Bayes model
    classifier = load_model(NAIVE_BAYES_MODEL)
    if classifier is None:
        print("Warning: No trained classifier found. Run 'train' command first for better results.")
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
                        except Exception as e:
                            # Classifier might not be trained on all classes
                            pass

                # Detect markers
                markers = []
                for marker in COMMENT_MARKERS:
                    if trie.search(marker.lower()) and marker.lower() in comment_lower:
                        markers.append(marker)

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
        print("=" * 70)

        for finding in suspicious_findings:
            rel_path = os.path.relpath(finding['file'], args.directory)
            print(f"\n[{finding['classification'].upper()}] {rel_path}:{finding['line']}")
            print(f"  Comment: {finding['comment']}")

            if finding['confidence'] > 0:
                print(f"  Classification: {finding['classification']} ({finding['confidence']:.0%} confidence)")

            if finding['markers']:
                print(f"  Markers: {', '.join(finding['markers'])}")

    print("\n" + "=" * 70)
    print(f"Bloom filter false positive rate: {bloom.false_positive_rate():.2%}")


# ==============================================================================
# COMMAND: TRAIN
# ==============================================================================

def cmd_train(args):
    """
    Train classifiers from labeled findings.

    Expected format in findings_dir:
    - misleading.txt: One comment per line (misleading comments)
    - accurate.txt: One comment per line (accurate comments)
    """
    print(f"Training classifiers from {args.findings_dir}...")
    print("=" * 70)

    findings_dir = Path(args.findings_dir)

    # Load training data
    misleading_file = findings_dir / "misleading.txt"
    accurate_file = findings_dir / "accurate.txt"

    if not misleading_file.exists() or not accurate_file.exists():
        print("Error: Training data not found.")
        print(f"Expected files:")
        print(f"  - {misleading_file}")
        print(f"  - {accurate_file}")
        print("\nCreate these files with one comment per line.")
        return

    # Read training examples
    misleading_comments = []
    with open(misleading_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                misleading_comments.append(line)

    accurate_comments = []
    with open(accurate_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                accurate_comments.append(line)

    print(f"Loaded {len(misleading_comments)} misleading examples")
    print(f"Loaded {len(accurate_comments)} accurate examples")

    if len(misleading_comments) == 0 or len(accurate_comments) == 0:
        print("Error: Need at least one example of each class")
        return

    # Tokenize comments
    tokenizer = Tokenizer()

    all_comments = []
    all_labels = []

    for comment in misleading_comments:
        tokens = tokenize_comment(comment, tokenizer)
        if tokens:
            all_comments.append(tokens)
            all_labels.append("misleading")

    for comment in accurate_comments:
        tokens = tokenize_comment(comment, tokenizer)
        if tokens:
            all_comments.append(tokens)
            all_labels.append("accurate")

    print(f"\nTokenized {len(all_comments)} comments for training")

    # Train Naive Bayes classifier
    print("\nTraining Naive Bayes classifier...")
    classifier = CommentClassifier()
    classifier.fit(all_comments, all_labels)

    # Save model
    save_model(classifier, NAIVE_BAYES_MODEL)

    # Show most indicative words
    print("\nMost indicative words for 'misleading':")
    indicative = classifier.most_indicative_words("misleading", top_n=10)
    for word, prob in indicative:
        print(f"  {word}: {prob:.4f}")

    print("\nMost indicative words for 'accurate':")
    indicative = classifier.most_indicative_words("accurate", top_n=10)
    for word, prob in indicative:
        print(f"  {word}: {prob:.4f}")

    print("\n" + "=" * 70)
    print("Training complete!")


# ==============================================================================
# COMMAND: PATTERNS
# ==============================================================================

def cmd_patterns(args):
    """
    Find repeated patterns in comments.

    Uses:
    - Suffix Array to find copy-pasted text
    - Count-Min Sketch for frequency tracking
    """
    print(f"Finding repeated patterns in {args.directory}...")
    print("=" * 70)

    # Collect all comments
    python_files = find_python_files(args.directory)
    print(f"Found {len(python_files)} Python files\n")

    all_comments = []
    comment_map = {}  # Map position to (file, line)

    current_pos = 0
    for file_path in python_files:
        comments = extract_comments_from_file(file_path)

        for line_num, comment in comments:
            all_comments.append(comment)
            comment_map[current_pos] = (file_path, line_num)
            current_pos += len(comment) + 1  # +1 for separator

    if not all_comments:
        print("No comments found.")
        return

    # Build combined text with separator
    combined_text = " ".join(all_comments)

    print(f"Analyzing {len(all_comments)} comments ({len(combined_text)} characters)...")

    # Build Suffix Array
    print("\nBuilding suffix array...")
    pattern_finder = CommentPatternFinder(combined_text)

    # Find repeated substrings
    min_length = args.min_length if hasattr(args, 'min_length') else 15
    print(f"Finding patterns (min length: {min_length})...\n")

    repeated = pattern_finder.repeated_substrings(min_length=min_length)

    if not repeated:
        print(f"No repeated patterns of length >= {min_length} found.")
        return

    # Initialize Count-Min Sketch for frequency tracking
    sketch = PatternFrequencySketch(width=1000, depth=5)

    for pattern, count in repeated:
        sketch.add(pattern, count)

    # Report top patterns
    print(f"Found {len(repeated)} repeated patterns\n")
    print("TOP REPEATED PATTERNS:")
    print("=" * 70)

    # Show top 20 patterns
    for i, (pattern, count) in enumerate(repeated[:20], 1):
        # Clean up pattern for display
        display_pattern = pattern[:60] + "..." if len(pattern) > 60 else pattern
        estimated_freq = sketch.query(pattern)

        print(f"\n{i}. Pattern (length={len(pattern)}, count={count}, est={estimated_freq}):")
        print(f"   \"{display_pattern}\"")

        # Find where it appears
        positions = pattern_finder.search(pattern)
        if positions:
            print(f"   Appears at {len(positions)} locations")

    print("\n" + "=" * 70)


# ==============================================================================
# COMMAND: SIMILAR
# ==============================================================================

def cmd_similar(args):
    """
    Find similar comments using LSH.

    This command requires an existing LSH index built by the 'index' command.
    """
    print(f"Finding comments similar to: \"{args.comment}\"")
    print("=" * 70)

    # Load LSH index
    lsh_index = load_model(LSH_MODEL)
    if lsh_index is None:
        print("Error: No LSH index found. Run 'index' command first.")
        return

    # Tokenize query comment
    tokenizer = Tokenizer()
    query_tokens = set(tokenize_comment(args.comment, tokenizer))

    if not query_tokens:
        print("Error: No tokens in query comment")
        return

    print(f"Query tokens: {sorted(query_tokens)}\n")

    # Query LSH index
    threshold = args.threshold if hasattr(args, 'threshold') else 0.3
    results = lsh_index.query(query_tokens, threshold=threshold)

    if not results:
        print(f"No similar comments found (threshold={threshold})")
        return

    print(f"Found {len(results)} similar comments:\n")
    print("SIMILAR COMMENTS:")
    print("=" * 70)

    for comment_id, similarity in results:
        print(f"\n{comment_id}")
        print(f"  Similarity: {similarity:.1%}")

    print("\n" + "=" * 70)


# ==============================================================================
# COMMAND: INDEX
# ==============================================================================

def cmd_index(args):
    """
    Build search indexes.

    Builds:
    - Inverted Index for term search
    - Trie for marker search
    - LSH index for similarity search
    """
    print(f"Building search indexes for {args.directory}...")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("Indexing complete!")
    print(f"Total comments indexed: {total_comments}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Audit Tool for Cortical Codebase Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for suspicious comments')
    scan_parser.add_argument('directory', help='Directory to scan')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train classifiers')
    train_parser.add_argument('findings_dir', help='Directory with labeled findings')

    # Patterns command
    patterns_parser = subparsers.add_parser('patterns', help='Find repeated patterns')
    patterns_parser.add_argument('directory', help='Directory to analyze')
    patterns_parser.add_argument('--min-length', type=int, default=15,
                                 help='Minimum pattern length (default: 15)')

    # Similar command
    similar_parser = subparsers.add_parser('similar', help='Find similar comments')
    similar_parser.add_argument('comment', help='Comment to find similar matches for')
    similar_parser.add_argument('--threshold', type=float, default=0.3,
                                help='Similarity threshold (default: 0.3)')

    # Index command
    index_parser = subparsers.add_parser('index', help='Build search indexes')
    index_parser.add_argument('directory', help='Directory to index')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    if args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'patterns':
        cmd_patterns(args)
    elif args.command == 'similar':
        cmd_similar(args)
    elif args.command == 'index':
        cmd_index(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
