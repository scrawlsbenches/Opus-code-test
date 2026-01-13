"""
Codebase health analysis - Pattern detection and quality metrics.

Uses algorithm implementations to analyze code quality:
1. Comment pattern detection (Trie + Inverted Index)
2. Duplicate detection (Suffix Array)
3. Similar comment clustering (LSH + Union-Find)
4. Git history analysis (blame, churn, stale TODOs)
5. Import dependency analysis (DAG)
"""

import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from cortical.audits.algorithms.trie import CommentMarkerTrie
from cortical.audits.algorithms.inverted_index import AuditInvertedIndex
from cortical.audits.algorithms.suffix_array import CommentPatternFinder
from cortical.audits.algorithms.lsh import SimilarCommentFinder
from cortical.audits.algorithms.union_find import FindingCluster
from cortical.audits.algorithms.count_min_sketch import PatternFrequencySketch
from cortical.audits.algorithms.dag import TaskDAG
from cortical.common.filesystem import FileSystem, RealFileSystem


# =============================================================================
# CONSTANTS
# =============================================================================

# Default suspicious patterns to detect
DEFAULT_SUSPICIOUS_PATTERNS = [
    "FUTURE:", "TODO:", "FIXME:", "HACK:", "XXX:",
    "will be", "should be", "planned to", "eventually",
    "See:", "see docs/", "See docs/",
    "monkeypatch",  # Test isolation concern - may need DI refactor
]

# Patterns in commit messages that indicate potential issues
COMMIT_SUSPICIOUS_PATTERNS = [
    (r'\bhack\b', 'hack'),
    (r'\bworkaround\b', 'workaround'),
    (r'\btemp\b', 'temporary'),
    (r'\bquick fix\b', 'quick fix'),
    (r'\bhotfix\b', 'hotfix'),
    (r'\bWIP\b', 'work-in-progress'),
    (r'\bFIXME\b', 'fixme'),
    (r'\bTODO\b', 'todo'),
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HealthAnalysisResult:
    """Results from health analysis."""
    files_analyzed: int = 0
    comments_analyzed: int = 0
    findings: List[Dict[str, Any]] = field(default_factory=list)
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    similar_groups: List[Tuple[str, List[str]]] = field(default_factory=list)
    repeated_substrings: List[Tuple[str, int]] = field(default_factory=list)
    git_analysis: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'files_analyzed': self.files_analyzed,
            'comments_analyzed': self.comments_analyzed,
            'findings': self.findings,
            'pattern_counts': self.pattern_counts,
            'similar_groups': self.similar_groups,
            'repeated_substrings': self.repeated_substrings,
            'git_analysis': self.git_analysis,
            'files': self.files,
            'error': self.error,
        }


# =============================================================================
# GIT UTILITIES
# =============================================================================

def run_git_command(args: List[str], cwd: str = ".") -> Optional[str]:
    """
    Run a git command and return output, or None on failure.

    Args:
        args: Git command arguments (without 'git')
        cwd: Working directory

    Returns:
        Command output or None on failure
    """
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


def get_git_blame_for_line(
    filepath: str,
    line_no: int,
    repo_root: str = "."
) -> Optional[Dict[str, Any]]:
    """
    Get git blame info for a specific line.

    Args:
        filepath: Path to file
        line_no: Line number
        repo_root: Repository root directory

    Returns:
        Dict with author, date, age_days, commit_msg or None
    """
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
    """
    Get file change counts over the last N days.

    Args:
        repo_root: Repository root directory
        days: Number of days to look back

    Returns:
        Dict mapping filepath to commit count
    """
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


def get_recent_commits_with_patterns(
    repo_root: str = ".",
    count: int = 100
) -> List[Dict[str, str]]:
    """
    Find recent commits with suspicious patterns in messages.

    Args:
        repo_root: Repository root directory
        count: Number of commits to check

    Returns:
        List of dicts with hash, message, date, pattern
    """
    output = run_git_command(
        ["log", f"-{count}", "--pretty=format:%H|%s|%ai"],
        cwd=repo_root
    )
    if not output:
        return []

    findings = []
    for line in output.strip().split('\n'):
        if '|' not in line:
            continue
        parts = line.split('|', 2)
        if len(parts) < 3:
            continue

        commit_hash, message, date = parts

        for pattern, label in COMMIT_SUSPICIOUS_PATTERNS:
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
    """
    Build a DAG of import dependencies between Python files.

    Args:
        directory: Directory to analyze

    Returns:
        TaskDAG with import relationships
    """
    dag = TaskDAG()
    root = Path(directory)

    import_pattern = re.compile(r'^(?:from|import)\s+([\w.]+)')

    # Map module names to files
    module_to_file = {}
    for py_file in root.rglob("*.py"):
        rel_path = py_file.relative_to(root)
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
                        for mod_name, mod_file in module_to_file.items():
                            if imported.startswith(mod_name.split('.')[0]):
                                if mod_file != rel_path:
                                    dag.add_dependency(mod_file, rel_path)
                                break
        except Exception:
            pass

    return dag


# =============================================================================
# CODEBASE ANALYZER
# =============================================================================

class CodebaseAnalyzer:
    """
    Analyzes codebases for health issues using dependency injection.

    This class accepts a FileSystem interface, enabling:
    - Real filesystem analysis in production
    - In-memory filesystem testing (fast, no disk I/O)
    - Consistent behavior across environments
    """

    def __init__(
        self,
        filesystem: FileSystem,
        patterns: Optional[List[str]] = None
    ):
        """
        Initialize analyzer with a filesystem.

        Args:
            filesystem: FileSystem implementation (Real or InMemory)
            patterns: Suspicious patterns to detect (defaults to DEFAULT_SUSPICIOUS_PATTERNS)
        """
        self._fs = filesystem
        self._patterns = patterns or DEFAULT_SUSPICIOUS_PATTERNS

    @property
    def filesystem(self) -> FileSystem:
        """Get the underlying filesystem."""
        return self._fs

    @property
    def patterns(self) -> List[str]:
        """Get suspicious patterns."""
        return self._patterns

    def find_python_files(self, root: Path) -> List[Path]:
        """Find all Python files in a directory."""
        return self._fs.glob(root, "**/*.py")

    def extract_comments(self, filepath: Path) -> List[Tuple[int, str]]:
        """Extract comments from a Python file with line numbers."""
        comments = []
        try:
            content = self._fs.read_text(filepath)
            for line_no, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    comments.append((line_no, stripped[1:].strip()))
        except (FileNotFoundError, Exception):
            pass
        return comments

    def scan_for_patterns(
        self,
        root: Path,
        patterns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan directory for suspicious patterns.

        Args:
            root: Directory to scan
            patterns: Patterns to look for (defaults to self._patterns)

        Returns:
            List of findings with file, line, pattern info
        """
        if patterns is None:
            patterns = self._patterns

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
    ) -> HealthAnalysisResult:
        """
        Run full analysis on a directory.

        Args:
            root: Directory to analyze
            with_git: Include git history analysis (requires real filesystem)

        Returns:
            HealthAnalysisResult with analysis data
        """
        if not self._fs.exists(root):
            return HealthAnalysisResult(error=f"Directory does not exist: {root}")

        if not self._fs.is_dir(root):
            return HealthAnalysisResult(error=f"Path is not a directory: {root}")

        py_files = self.find_python_files(root)

        result = HealthAnalysisResult(
            files_analyzed=len(py_files),
            files=[str(f) for f in py_files],
        )

        # Scan for patterns
        findings = self.scan_for_patterns(root)
        result.findings = findings

        # Count patterns
        for finding in findings:
            pattern = finding["pattern"]
            result.pattern_counts[pattern] = result.pattern_counts.get(pattern, 0) + 1

        # Count total comments
        total_comments = 0
        for py_file in py_files:
            comments = self.extract_comments(py_file)
            total_comments += len(comments)
        result.comments_analyzed = total_comments

        return result


# =============================================================================
# FULL ANALYSIS FUNCTION
# =============================================================================

def analyze_directory(
    directory: str,
    with_git: bool = False,
    git_only: bool = False,
    verbose: bool = True,
) -> HealthAnalysisResult:
    """
    Run full health analysis on a directory.

    Args:
        directory: Directory to analyze
        with_git: Include git history analysis
        git_only: Only run git analysis
        verbose: Print progress information

    Returns:
        HealthAnalysisResult with all analysis data
    """
    root = Path(directory)
    if not root.exists():
        if verbose:
            print(f"Error: {directory} does not exist")
        return HealthAnalysisResult(error=f"Directory does not exist: {directory}")

    # Find repo root for git operations
    repo_root_str = run_git_command(["rev-parse", "--show-toplevel"], cwd=str(root.resolve()))
    repo_root = Path(repo_root_str.strip()) if repo_root_str else root.resolve()
    root = root.resolve()

    result = HealthAnalysisResult()

    if verbose:
        print(f"Analyzing {directory}...")
        print("=" * 60)

    # Collect all Python files
    py_files = list(root.rglob("*.py"))
    result.files_analyzed = len(py_files)
    result.files = [str(f) for f in py_files]

    if verbose:
        print(f"Found {len(py_files)} Python files")

    # Git history analysis
    if with_git or git_only:
        _analyze_git(root, repo_root, result, verbose)

        if git_only:
            if verbose:
                print()
                print("=" * 60)
                print("Git-only analysis complete")
            return result

    # Comment analysis
    _analyze_comments(root, repo_root, py_files, result, with_git, verbose)

    if verbose:
        print()
        print("=" * 60)
        print(f"Analysis complete: {len(result.findings)} potential issues found")

    return result


def _analyze_git(
    root: Path,
    repo_root: Path,
    result: HealthAnalysisResult,
    verbose: bool
) -> None:
    """Run git analysis and update result."""
    if verbose:
        print()
        print("GIT HISTORY ANALYSIS")
        print("-" * 40)

    # File churn analysis
    if verbose:
        print("\n[File Churn - Last 90 Days]")

    churn = get_file_churn(str(repo_root), days=90)
    if churn:
        try:
            rel_root = str(root.relative_to(repo_root)) if root != repo_root else ""
        except ValueError:
            rel_root = ""

        filtered_churn = {
            k: v for k, v in churn.items()
            if not rel_root or k.startswith(rel_root)
        }
        top_churn = sorted(filtered_churn.items(), key=lambda x: -x[1])[:10]

        if top_churn and verbose:
            print("  Most frequently changed files:")
            for filepath, count in top_churn:
                indicator = "⚠️ " if count > 10 else "  "
                print(f"  {indicator}{filepath}: {count} commits")

        result.git_analysis['high_churn_files'] = top_churn
    elif verbose:
        print("  Could not retrieve file churn data")

    # Suspicious commit messages
    if verbose:
        print("\n[Suspicious Commit Patterns]")

    suspicious_commits = get_recent_commits_with_patterns(str(repo_root), count=200)
    if suspicious_commits:
        if verbose:
            print(f"  Found {len(suspicious_commits)} commits with suspicious patterns:")
            for commit in suspicious_commits[:5]:
                print(f"    [{commit['pattern']}] {commit['hash']} - {commit['message']}")
            if len(suspicious_commits) > 5:
                print(f"    ... and {len(suspicious_commits) - 5} more")
        result.git_analysis['suspicious_commits'] = suspicious_commits
    elif verbose:
        print("  No suspicious commit patterns found")

    # Import dependency analysis
    if verbose:
        print("\n[Import Dependencies - DAG Analysis]")

    dag = analyze_import_dependencies(str(root))
    roots = dag.roots()
    leaves = dag.leaves()

    if verbose:
        print(f"  Entry points (no internal dependencies): {len(roots)}")
        print(f"  Leaf modules (nothing depends on them): {len(leaves)}")

    # Find most depended-upon files
    dep_counts = {}
    for node in dag.topological_sort():
        blocked = dag.blocks(node)
        if blocked:
            dep_counts[node] = len(blocked)

    if dep_counts:
        top_deps = sorted(dep_counts.items(), key=lambda x: -x[1])[:5]
        if verbose:
            print("  Most critical modules (most dependents):")
            for filepath, count in top_deps:
                print(f"    {filepath}: {count} modules depend on it")
        result.git_analysis['critical_modules'] = top_deps


def _analyze_comments(
    root: Path,
    repo_root: Path,
    py_files: List[Path],
    result: HealthAnalysisResult,
    with_git: bool,
    verbose: bool
) -> None:
    """Run comment analysis and update result."""
    # Initialize data structures
    marker_trie = CommentMarkerTrie()
    pattern_freq = PatternFrequencySketch(width=1000, depth=5)
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    clusters = FindingCluster()

    all_comments_text = []
    findings = []

    # Process each file
    comment_count = 0
    for py_file in py_files:
        rel_path = py_file.relative_to(root)
        comments = _extract_comments_from_file(py_file)

        for line_no, comment in comments:
            comment_count += 1
            finding_id = f"{rel_path}:{line_no}"

            all_comments_text.append(comment)

            # Check for markers
            for pattern in DEFAULT_SUSPICIOUS_PATTERNS:
                if pattern.lower() in comment.lower():
                    marker_trie.insert(pattern, accumulate=True)
                    pattern_freq.add(pattern.lower())

                    finding = {
                        'id': finding_id,
                        'file': str(rel_path),
                        'line': line_no,
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
                            if pattern in ['TODO:', 'FIXME:'] and finding['age_days'] > 180:
                                finding['stale'] = True

                    findings.append(finding)

            # Add to LSH for similarity detection
            tokens = set(comment.lower().split())
            if len(tokens) >= 3:
                lsh.add(finding_id, tokens)
                clusters.make_set(finding_id)

    result.comments_analyzed = comment_count
    result.findings = findings

    if verbose:
        print(f"Processed {comment_count} comments")
        print()

    # Pattern frequency
    if verbose:
        print("PATTERN FREQUENCY (Count-Min Sketch)")
        print("-" * 40)

    for pattern in DEFAULT_SUSPICIOUS_PATTERNS:
        count = pattern_freq.query(pattern.lower())
        if count > 0:
            result.pattern_counts[pattern] = count
            if verbose:
                print(f"  {pattern:<15} ~ {count} occurrences")

    if verbose:
        print()

    # Marker grouping
    if verbose:
        print("MARKER GROUPS (Trie)")
        print("-" * 40)

    all_markers = marker_trie.all_markers()
    if all_markers and verbose:
        for marker in sorted(all_markers):
            count = marker_trie.get_count(marker)
            print(f"  {marker:<15} = {count}")
    elif verbose:
        print("  No markers found")

    if verbose:
        print()

    # Similar comments
    if verbose:
        print("SIMILAR COMMENT DETECTION (LSH)")
        print("-" * 40)

    similar_pairs = []
    checked = set()

    for finding in findings[:50]:
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
                for other in group:
                    clusters.union(fid, other)
                checked.update(group)

    if similar_pairs:
        if verbose:
            for fid, group in similar_pairs[:5]:
                print(f"  {fid}")
                for other in group[:3]:
                    print(f"    ~ similar to: {other}")
        result.similar_groups = similar_pairs
    elif verbose:
        print("  No highly similar comments detected")

    if verbose:
        print()

    # Repeated substrings
    if all_comments_text:
        combined = " ".join(all_comments_text[:100])
        if len(combined) > 100:
            if verbose:
                print("REPEATED PATTERNS (Suffix Array)")
                print("-" * 40)

            finder = CommentPatternFinder(combined)
            repeated = finder.repeated_substrings(min_length=15)[:5]

            if repeated:
                if verbose:
                    for pattern, count in repeated:
                        if len(pattern) < 80:
                            print(f'  [{count}x] "{pattern}"')
                        else:
                            print(f'  [{count}x] "{pattern[:77]}..."')
                result.repeated_substrings = repeated
            elif verbose:
                print("  No significant repeated patterns")

            if verbose:
                print()

    # Summary
    if verbose:
        print("SUSPICIOUS FINDINGS SUMMARY")
        print("-" * 40)
        for finding in findings[:10]:
            age_str = f" ({finding['age_days']}d)" if 'age_days' in finding else ""
            stale_marker = " ⚠️ STALE" if finding.get('stale') else ""
            print(f"  [{finding['pattern']}] {finding['id']}{age_str}{stale_marker}")
            print(f"      {finding['comment'][:60]}...")

        if len(findings) > 10:
            print(f"  ... and {len(findings) - 10} more")


def _extract_comments_from_file(filepath: Path) -> List[Tuple[int, str]]:
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
