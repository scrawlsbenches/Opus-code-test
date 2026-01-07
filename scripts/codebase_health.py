#!/usr/bin/env python3
"""
Codebase Health Analyzer

Uses the algorithm implementations to analyze code quality:
1. Comment pattern detection (Trie + Inverted Index)
2. Duplicate detection (Suffix Array)
3. Similar comment clustering (LSH + Union-Find)
4. Comment classification (Naive Bayes - when trained)

Usage:
    python scripts/codebase_health.py [directory]
    python scripts/codebase_health.py cortical/got/
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.audits.algorithms.trie import CommentMarkerTrie
from cortical.audits.algorithms.inverted_index import AuditInvertedIndex
from cortical.audits.algorithms.suffix_array import CommentPatternFinder
from cortical.audits.algorithms.lsh import SimilarCommentFinder
from cortical.audits.algorithms.union_find import FindingCluster
from cortical.audits.algorithms.count_min_sketch import PatternFrequencySketch


def extract_comments(filepath: Path) -> List[Tuple[int, str]]:
    """Extract comments from a Python file with line numbers."""
    comments = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    comments.append((line_no, stripped[1:].strip()))
    except Exception:
        pass
    return comments


def analyze_directory(directory: str) -> Dict:
    """Run full health analysis on a directory."""
    root = Path(directory)
    if not root.exists():
        print(f"Error: {directory} does not exist")
        return {}

    # Initialize data structures
    marker_trie = CommentMarkerTrie()
    comment_index = AuditInvertedIndex()
    pattern_freq = PatternFrequencySketch(width=1000, depth=5)
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    clusters = FindingCluster()

    all_comments_text = []
    findings = []

    # Suspicious patterns to look for
    suspicious_patterns = [
        "FUTURE:", "TODO:", "FIXME:", "HACK:", "XXX:",
        "will be", "should be", "planned to", "eventually",
        "See:", "see docs/", "See docs/"
    ]

    print(f"Analyzing {directory}...")
    print("=" * 60)

    # Collect all Python files
    py_files = list(root.rglob("*.py"))
    print(f"Found {len(py_files)} Python files")

    # Process each file
    comment_count = 0
    for py_file in py_files:
        rel_path = py_file.relative_to(root)
        comments = extract_comments(py_file)

        for line_no, comment in comments:
            comment_count += 1
            finding_id = f"{rel_path}:{line_no}"

            # Index the comment
            comment_index.index_text(finding_id, comment)
            all_comments_text.append(comment)

            # Check for markers
            for pattern in suspicious_patterns:
                if pattern.lower() in comment.lower():
                    marker_trie.insert(pattern, accumulate=True)
                    pattern_freq.add(pattern.lower())
                    findings.append({
                        'id': finding_id,
                        'pattern': pattern,
                        'comment': comment[:100]
                    })

            # Add to LSH for similarity detection
            tokens = set(comment.lower().split())
            if len(tokens) >= 3:  # Only meaningful comments
                lsh.add(finding_id, tokens)
                clusters.make_set(finding_id)

    print(f"Processed {comment_count} comments")
    print()

    # Analysis Results
    results = {
        'files_analyzed': len(py_files),
        'comments_analyzed': comment_count,
        'findings': [],
        'pattern_counts': {},
        'similar_groups': [],
        'repeated_substrings': []
    }

    # 1. Pattern frequency analysis
    print("PATTERN FREQUENCY (Count-Min Sketch)")
    print("-" * 40)
    for pattern in suspicious_patterns:
        count = pattern_freq.query(pattern.lower())
        if count > 0:
            results['pattern_counts'][pattern] = count
            print(f"  {pattern:<15} ~ {count} occurrences")
    print()

    # 2. Marker grouping (Trie)
    print("MARKER GROUPS (Trie)")
    print("-" * 40)
    all_markers = marker_trie.all_markers()
    if all_markers:
        for marker in sorted(all_markers):
            count = marker_trie.get_count(marker)
            print(f"  {marker:<15} = {count}")
    else:
        print("  No markers found")
    print()

    # 3. Find similar comments (LSH)
    print("SIMILAR COMMENT DETECTION (LSH)")
    print("-" * 40)
    similar_pairs = []
    checked = set()

    for finding in findings[:50]:  # Check first 50 findings
        fid = finding['id']
        if fid in checked:
            continue

        tokens = set(finding['comment'].lower().split())
        if len(tokens) < 3:
            continue

        similar = lsh.query(tokens, threshold=0.6)
        if len(similar) > 1:
            group = [s[0] for s in similar if s[0] != fid]
            if group:
                similar_pairs.append((fid, group))
                # Cluster similar findings
                for other in group:
                    clusters.union(fid, other)
                checked.update(group)

    if similar_pairs:
        for fid, group in similar_pairs[:5]:  # Show top 5
            print(f"  {fid}")
            for other in group[:3]:
                print(f"    ~ similar to: {other}")
        results['similar_groups'] = similar_pairs
    else:
        print("  No highly similar comments detected")
    print()

    # 4. Find repeated substrings (Suffix Array)
    if all_comments_text:
        combined = " ".join(all_comments_text[:100])  # First 100 comments
        if len(combined) > 100:
            print("REPEATED PATTERNS (Suffix Array)")
            print("-" * 40)
            finder = CommentPatternFinder(combined)
            repeated = finder.repeated_substrings(min_length=15)[:5]

            if repeated:
                for pattern, count in repeated:
                    if len(pattern) < 80:
                        print(f"  [{count}x] \"{pattern}\"")
                    else:
                        print(f"  [{count}x] \"{pattern[:77]}...\"")
                results['repeated_substrings'] = repeated
            else:
                print("  No significant repeated patterns")
            print()

    # 5. Cluster summary (Union-Find)
    all_clusters = clusters.get_all_clusters()
    multi_clusters = [c for c in all_clusters if len(c) > 1]
    if multi_clusters:
        print("FINDING CLUSTERS (Union-Find)")
        print("-" * 40)
        print(f"  {len(multi_clusters)} clusters with related findings")
        for i, cluster in enumerate(multi_clusters[:3], 1):
            print(f"  Cluster {i}: {len(cluster)} related findings")
        print()

    # 6. Summary of findings
    print("SUSPICIOUS FINDINGS SUMMARY")
    print("-" * 40)
    for finding in findings[:10]:
        print(f"  [{finding['pattern']}] {finding['id']}")
        print(f"      {finding['comment'][:60]}...")

    if len(findings) > 10:
        print(f"  ... and {len(findings) - 10} more")

    results['findings'] = findings

    print()
    print("=" * 60)
    print(f"Analysis complete: {len(findings)} potential issues found")

    return results


def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "cortical/"

    analyze_directory(directory)


if __name__ == "__main__":
    main()
