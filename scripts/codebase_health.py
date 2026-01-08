#!/usr/bin/env python3
"""
Codebase Health Analyzer

DEPRECATION NOTE: This script is maintained for backward compatibility.
Prefer using the CLI: python -m cortical.cli audit health ...
Or import directly: from cortical.audits import analyze_directory, CodebaseAnalyzer

Uses the algorithm implementations to analyze code quality:
1. Comment pattern detection (Trie + Inverted Index)
2. Duplicate detection (Suffix Array)
3. Similar comment clustering (LSH + Union-Find)
4. Git history analysis (blame, churn, stale TODOs)
5. Import dependency analysis (DAG)

Usage:
    python scripts/codebase_health.py [directory]
    python scripts/codebase_health.py cortical/got/
    python scripts/codebase_health.py --with-git cortical/  # Include git analysis
    python scripts/codebase_health.py --git-only            # Only git analysis
"""

import sys
import os
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.audits.algorithms.trie import CommentMarkerTrie
from cortical.audits.algorithms.inverted_index import AuditInvertedIndex
from cortical.audits.algorithms.suffix_array import CommentPatternFinder
from cortical.audits.algorithms.lsh import SimilarCommentFinder
from cortical.audits.algorithms.union_find import FindingCluster
from cortical.audits.algorithms.count_min_sketch import PatternFrequencySketch
from cortical.audits.algorithms.dag import TaskDAG
from cortical.common.filesystem import FileSystem, RealFileSystem, InMemoryFileSystem


# =============================================================================
# CODEBASE ANALYZER - Uses FileSystem interface for DI
# =============================================================================


class CodebaseAnalyzer:
    """
    Analyzes codebases for health issues using dependency injection.

    This class accepts a FileSystem interface, enabling:
    - Real filesystem analysis in production
    - In-memory filesystem testing (fast, no disk I/O)
    - Consistent behavior across environments

    Example:
        # Production
        analyzer = CodebaseAnalyzer(RealFileSystem())
        results = analyzer.analyze(Path("/project/src"))

        # Testing
        fs = InMemoryFileSystem()
        fs.write_text(Path("/src/main.py"), "# TODO: fix this")
        analyzer = CodebaseAnalyzer(fs)
        results = analyzer.analyze(Path("/src"))
    """

    # Suspicious patterns to detect
    SUSPICIOUS_PATTERNS = [
        "FUTURE:", "TODO:", "FIXME:", "HACK:", "XXX:",
        "will be", "should be", "planned to", "eventually",
        "See:", "see docs/", "See docs/"
    ]

    def __init__(self, filesystem: FileSystem):
        """
        Initialize analyzer with a filesystem.

        Args:
            filesystem: FileSystem implementation (Real or InMemory)
        """
        self._fs = filesystem

    @property
    def filesystem(self) -> FileSystem:
        """Get the underlying filesystem."""
        return self._fs

    def find_python_files(self, root: Path) -> List[Path]:
        """
        Find all Python files in a directory.

        Args:
            root: Directory to search

        Returns:
            List of Python file paths
        """
        return self._fs.glob(root, "**/*.py")

    def extract_comments(self, filepath: Path) -> List[Tuple[int, str]]:
        """
        Extract comments from a Python file with line numbers.

        Args:
            filepath: Path to the Python file

        Returns:
            List of (line_number, comment_text) tuples
        """
        comments = []
        try:
            content = self._fs.read_text(filepath)
            for line_no, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    comments.append((line_no, stripped[1:].strip()))
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return comments

    def scan_for_patterns(
        self,
        root: Path,
        patterns: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Scan directory for suspicious patterns.

        Args:
            root: Directory to scan
            patterns: Patterns to look for (defaults to SUSPICIOUS_PATTERNS)

        Returns:
            List of findings with file, line, pattern info
        """
        if patterns is None:
            patterns = self.SUSPICIOUS_PATTERNS

        findings = []
        py_files = self.find_python_files(root)

        for py_file in py_files:
            try:
                rel_path = py_file.relative_to(root)
            except ValueError:
                rel_path = py_file

            comments = self.extract_comments(py_file)

            for line_no, comment in comments:
                for pattern in patterns:
                    if pattern.lower() in comment.lower():
                        findings.append({
                            "id": f"{rel_path}:{line_no}",
                            "file": str(rel_path),
                            "line": line_no,
                            "pattern": pattern.rstrip(":").lower(),
                            "comment": comment,
                        })

        return findings

    def analyze(
        self,
        root: Path,
        with_git: bool = False
    ) -> Dict[str, any]:
        """
        Run full analysis on a directory.

        Args:
            root: Directory to analyze
            with_git: Include git history analysis (requires real filesystem)

        Returns:
            Analysis results dictionary
        """
        if not self._fs.exists(root):
            return {"error": f"Directory does not exist: {root}"}

        if not self._fs.is_dir(root):
            return {"error": f"Path is not a directory: {root}"}

        # Find Python files
        py_files = self.find_python_files(root)

        results = {
            "files_analyzed": len(py_files),
            "comments_analyzed": 0,
            "findings": [],
            "pattern_counts": {},
            "files": [str(f) for f in py_files],
        }

        # Scan for patterns
        findings = self.scan_for_patterns(root)
        results["findings"] = findings

        # Count patterns
        pattern_counts = {}
        for finding in findings:
            pattern = finding["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        results["pattern_counts"] = pattern_counts

        # Count total comments
        total_comments = 0
        for py_file in py_files:
            comments = self.extract_comments(py_file)
            total_comments += len(comments)
        results["comments_analyzed"] = total_comments

        return results


# =============================================================================
# GIT HISTORY ANALYSIS
# =============================================================================

def run_git_command(args: List[str], cwd: str = ".") -> Optional[str]:
    """Run a git command and return output, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_git_blame_for_line(filepath: str, line_no: int, repo_root: str = ".") -> Optional[Dict]:
    """Get git blame info for a specific line."""
    output = run_git_command(
        ["blame", "-L", f"{line_no},{line_no}", "--porcelain", filepath],
        cwd=repo_root
    )
    if not output:
        return None

    info = {}
    lines = output.strip().split('\n')
    for line in lines:
        if line.startswith('author '):
            info['author'] = line[7:]
        elif line.startswith('author-time '):
            timestamp = int(line[12:])
            info['date'] = datetime.fromtimestamp(timestamp)
            info['age_days'] = (datetime.now() - info['date']).days
        elif line.startswith('summary '):
            info['commit_msg'] = line[8:]

    return info if info else None


def get_file_churn(repo_root: str = ".", days: int = 90) -> Dict[str, int]:
    """Get file change counts over the last N days."""
    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    output = run_git_command(
        ["log", "--since", since_date, "--name-only", "--pretty=format:"],
        cwd=repo_root
    )
    if not output:
        return {}

    churn = defaultdict(int)
    for line in output.strip().split('\n'):
        if line and line.endswith('.py'):
            churn[line] += 1

    return dict(churn)


def get_recent_commits_with_patterns(repo_root: str = ".", count: int = 100) -> List[Dict]:
    """Find recent commits with suspicious patterns in messages."""
    output = run_git_command(
        ["log", f"-{count}", "--pretty=format:%H|%s|%ai"],
        cwd=repo_root
    )
    if not output:
        return []

    suspicious_patterns = [
        (r'\bhack\b', 'hack'),
        (r'\bworkaround\b', 'workaround'),
        (r'\btemp\b', 'temporary'),
        (r'\bquick fix\b', 'quick fix'),
        (r'\bhotfix\b', 'hotfix'),
        (r'\bWIP\b', 'work-in-progress'),
        (r'\bFIXME\b', 'fixme'),
        (r'\bTODO\b', 'todo'),
    ]

    findings = []
    for line in output.strip().split('\n'):
        if '|' not in line:
            continue
        parts = line.split('|', 2)
        if len(parts) < 3:
            continue

        commit_hash, message, date = parts
        message_lower = message.lower()

        for pattern, label in suspicious_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                findings.append({
                    'hash': commit_hash[:8],
                    'message': message[:60],
                    'date': date[:10],
                    'pattern': label
                })
                break

    return findings


def analyze_import_dependencies(directory: str) -> TaskDAG:
    """Build a DAG of import dependencies between Python files."""
    dag = TaskDAG()
    root = Path(directory)

    import_pattern = re.compile(r'^(?:from|import)\s+([\w.]+)')

    # Map module names to files
    module_to_file = {}
    for py_file in root.rglob("*.py"):
        rel_path = py_file.relative_to(root)
        # Convert path to module name (e.g., cortical/got/api.py -> cortical.got.api)
        module_name = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
        module_to_file[module_name] = str(rel_path)
        dag.add_task(str(rel_path))

    # Parse imports
    for py_file in root.rglob("*.py"):
        rel_path = str(py_file.relative_to(root))
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = import_pattern.match(line.strip())
                    if match:
                        imported = match.group(1)
                        # Check if this is an internal import
                        for mod_name, mod_file in module_to_file.items():
                            if imported.startswith(mod_name.split('.')[0]):
                                if mod_file != rel_path:
                                    dag.add_dependency(mod_file, rel_path)
                                break
        except Exception:
            pass

    return dag


# =============================================================================
# COMMENT EXTRACTION
# =============================================================================

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


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_directory(directory: str, with_git: bool = False, git_only: bool = False) -> Dict:
    """Run full health analysis on a directory."""
    root = Path(directory)
    if not root.exists():
        print(f"Error: {directory} does not exist")
        return {}

    # Find repo root for git operations
    repo_root_str = run_git_command(["rev-parse", "--show-toplevel"], cwd=str(root.resolve()))
    repo_root = Path(repo_root_str.strip()) if repo_root_str else root.resolve()
    root = root.resolve()  # Make absolute for consistent comparisons

    results = {
        'files_analyzed': 0,
        'comments_analyzed': 0,
        'findings': [],
        'pattern_counts': {},
        'similar_groups': [],
        'repeated_substrings': [],
        'git_analysis': {}
    }

    print(f"Analyzing {directory}...")
    print("=" * 60)

    # Collect all Python files
    py_files = list(root.rglob("*.py"))
    results['files_analyzed'] = len(py_files)
    print(f"Found {len(py_files)} Python files")

    # ==========================================================================
    # GIT HISTORY ANALYSIS (if enabled)
    # ==========================================================================
    if with_git or git_only:
        print()
        print("GIT HISTORY ANALYSIS")
        print("-" * 40)

        # 1. File churn analysis
        print("\n[File Churn - Last 90 Days]")
        churn = get_file_churn(str(repo_root), days=90)
        if churn:
            # Filter to files in our directory
            try:
                rel_root = str(root.relative_to(repo_root)) if root != repo_root else ""
            except ValueError:
                rel_root = ""  # If paths don't share root, show all
            filtered_churn = {
                k: v for k, v in churn.items()
                if not rel_root or k.startswith(rel_root)
            }
            top_churn = sorted(filtered_churn.items(), key=lambda x: -x[1])[:10]
            if top_churn:
                print("  Most frequently changed files:")
                for filepath, count in top_churn:
                    indicator = "⚠️ " if count > 10 else "  "
                    print(f"  {indicator}{filepath}: {count} commits")
                results['git_analysis']['high_churn_files'] = top_churn
        else:
            print("  Could not retrieve file churn data")

        # 2. Suspicious commit messages
        print("\n[Suspicious Commit Patterns]")
        suspicious_commits = get_recent_commits_with_patterns(str(repo_root), count=200)
        if suspicious_commits:
            print(f"  Found {len(suspicious_commits)} commits with suspicious patterns:")
            for commit in suspicious_commits[:5]:
                print(f"    [{commit['pattern']}] {commit['hash']} - {commit['message']}")
            if len(suspicious_commits) > 5:
                print(f"    ... and {len(suspicious_commits) - 5} more")
            results['git_analysis']['suspicious_commits'] = suspicious_commits
        else:
            print("  No suspicious commit patterns found")

        # 3. Import dependency analysis (using DAG)
        print("\n[Import Dependencies - DAG Analysis]")
        dag = analyze_import_dependencies(str(root))
        roots = dag.roots()
        leaves = dag.leaves()
        print(f"  Entry points (no internal dependencies): {len(roots)}")
        print(f"  Leaf modules (nothing depends on them): {len(leaves)}")

        # Find most depended-upon files
        dep_counts = {}
        for node in dag._nodes:
            blocked = dag.blocks(node)
            if blocked:
                dep_counts[node] = len(blocked)

        if dep_counts:
            top_deps = sorted(dep_counts.items(), key=lambda x: -x[1])[:5]
            print("  Most critical modules (most dependents):")
            for filepath, count in top_deps:
                print(f"    {filepath}: {count} modules depend on it")
            results['git_analysis']['critical_modules'] = top_deps

        if git_only:
            print()
            print("=" * 60)
            print("Git-only analysis complete")
            return results

    # ==========================================================================
    # COMMENT ANALYSIS
    # ==========================================================================

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

                    finding = {
                        'id': finding_id,
                        'pattern': pattern,
                        'comment': comment[:100]
                    }

                    # Add git blame info if enabled
                    if with_git:
                        blame_info = get_git_blame_for_line(
                            str(py_file), line_no, str(repo_root)
                        )
                        if blame_info:
                            finding['age_days'] = blame_info.get('age_days', 0)
                            finding['author'] = blame_info.get('author', 'unknown')
                            # Flag stale TODOs (older than 180 days)
                            if pattern in ['TODO:', 'FIXME:'] and finding['age_days'] > 180:
                                finding['stale'] = True

                    findings.append(finding)

            # Add to LSH for similarity detection
            tokens = set(comment.lower().split())
            if len(tokens) >= 3:  # Only meaningful comments
                lsh.add(finding_id, tokens)
                clusters.make_set(finding_id)

    results['comments_analyzed'] = comment_count
    print(f"Processed {comment_count} comments")
    print()

    # ==========================================================================
    # ANALYSIS RESULTS
    # ==========================================================================

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

    # 3. Stale TODOs (if git enabled)
    if with_git:
        stale_findings = [f for f in findings if f.get('stale')]
        if stale_findings:
            print("STALE TODOs (> 180 days old)")
            print("-" * 40)
            for finding in stale_findings[:10]:
                print(f"  ⚠️  {finding['id']} ({finding['age_days']} days)")
                print(f"      {finding['comment'][:50]}...")
            if len(stale_findings) > 10:
                print(f"  ... and {len(stale_findings) - 10} more stale items")
            print()

    # 4. Find similar comments (LSH)
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

    # 5. Find repeated substrings (Suffix Array)
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

    # 6. Cluster summary (Union-Find)
    all_clusters = clusters.get_all_clusters()
    multi_clusters = [c for c in all_clusters if len(c) > 1]
    if multi_clusters:
        print("FINDING CLUSTERS (Union-Find)")
        print("-" * 40)
        print(f"  {len(multi_clusters)} clusters with related findings")
        for i, cluster in enumerate(multi_clusters[:3], 1):
            print(f"  Cluster {i}: {len(cluster)} related findings")
        print()

    # 7. Summary of findings
    print("SUSPICIOUS FINDINGS SUMMARY")
    print("-" * 40)
    for finding in findings[:10]:
        age_str = f" ({finding['age_days']}d)" if 'age_days' in finding else ""
        stale_marker = " ⚠️ STALE" if finding.get('stale') else ""
        print(f"  [{finding['pattern']}] {finding['id']}{age_str}{stale_marker}")
        print(f"      {finding['comment'][:60]}...")

    if len(findings) > 10:
        print(f"  ... and {len(findings) - 10} more")

    results['findings'] = findings

    print()
    print("=" * 60)
    print(f"Analysis complete: {len(findings)} potential issues found")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Codebase Health Analyzer")
    parser.add_argument("directory", nargs="?", default="cortical/",
                        help="Directory to analyze")
    parser.add_argument("--with-git", action="store_true",
                        help="Include git history analysis (blame, churn)")
    parser.add_argument("--git-only", action="store_true",
                        help="Only run git analysis")

    args = parser.parse_args()
    analyze_directory(args.directory, with_git=args.with_git, git_only=args.git_only)


if __name__ == "__main__":
    main()
