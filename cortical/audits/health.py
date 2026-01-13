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
import json

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

# Default suspicious patterns to detect (legacy - comment scope only)
DEFAULT_SUSPICIOUS_PATTERNS = [
    "FUTURE:", "TODO:", "FIXME:", "HACK:", "XXX:",
    "will be", "should be", "planned to", "eventually",
    "See:", "see docs/", "See docs/",
]

# Default pattern file location
DEFAULT_PATTERNS_FILE = Path(".got/audit_patterns.json")


@dataclass
class AuditPattern:
    """A pattern to detect in code with scope control."""
    id: str
    match: str
    scope: str = "comments"  # "comments", "code", or "all"
    implies: str = ""
    strength: float = 0.5
    regex: bool = False
    description: str = ""
    _compiled: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        if self.regex:
            try:
                self._compiled = re.compile(self.match, re.IGNORECASE)
            except re.error:
                self._compiled = None

    def matches(self, text: str) -> bool:
        """Check if pattern matches text."""
        if self.regex and self._compiled:
            return bool(self._compiled.search(text))
        return self.match.lower() in text.lower()


def load_custom_patterns(patterns_file: Optional[Path] = None) -> List[AuditPattern]:
    """Load custom patterns from JSON file."""
    path = patterns_file or DEFAULT_PATTERNS_FILE
    if not path.exists():
        return []

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        patterns = []
        for p in data.get('patterns', []):
            patterns.append(AuditPattern(
                id=p.get('id', ''),
                match=p.get('match', ''),
                scope=p.get('scope', 'comments'),
                implies=p.get('implies', ''),
                strength=p.get('strength', 0.5),
                regex=p.get('regex', False),
                description=p.get('description', ''),
            ))
        return patterns
    except (json.JSONDecodeError, KeyError):
        return []


def get_all_patterns(patterns_file: Optional[Path] = None) -> List[AuditPattern]:
    """Get combined default and custom patterns."""
    # Convert legacy patterns to AuditPattern (comments scope)
    patterns = [
        AuditPattern(id=p.replace(':', '').lower(), match=p, scope="comments")
        for p in DEFAULT_SUSPICIOUS_PATTERNS
    ]
    # Add custom patterns
    patterns.extend(load_custom_patterns(patterns_file))
    return patterns

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
    """Run comment and code analysis with scope-aware pattern detection."""
    # Initialize data structures
    marker_trie = CommentMarkerTrie()
    pattern_freq = PatternFrequencySketch(width=1000, depth=5)
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    clusters = FindingCluster()

    all_comments_text = []
    findings = []

    # Load all patterns (default + custom)
    all_patterns = get_all_patterns()

    # Separate patterns by scope for efficiency
    comment_patterns = [p for p in all_patterns if p.scope in ("comments", "all")]
    code_patterns = [p for p in all_patterns if p.scope in ("code", "all")]

    # Process each file
    comment_count = 0
    for py_file in py_files:
        rel_path = py_file.relative_to(root)

        # Extract both comments and code in one pass
        comments, code_lines = _extract_all_lines_from_file(py_file)

        # Check comments against comment-scope patterns
        for line_no, comment in comments:
            comment_count += 1
            finding_id = f"{rel_path}:{line_no}"
            all_comments_text.append(comment)

            for pattern in comment_patterns:
                if pattern.matches(comment):
                    marker_trie.insert(pattern.match, accumulate=True)
                    pattern_freq.add(pattern.id)

                    finding = {
                        'id': finding_id,
                        'file': str(rel_path),
                        'line': line_no,
                        'pattern': pattern.match,
                        'comment': comment[:100],
                        'scope': 'comment',
                    }
                    if pattern.implies:
                        finding['implies'] = pattern.implies
                        finding['strength'] = pattern.strength

                    # Add git blame info if enabled
                    if with_git:
                        blame_info = get_git_blame_for_line(
                            str(py_file), line_no, str(repo_root)
                        )
                        if blame_info:
                            finding['age_days'] = blame_info.get('age_days', 0)
                            finding['author'] = blame_info.get('author', 'unknown')
                            if pattern.match in ['TODO:', 'FIXME:'] and finding['age_days'] > 180:
                                finding['stale'] = True

                    findings.append(finding)

            # Add to LSH for similarity detection
            tokens = set(comment.lower().split())
            if len(tokens) >= 3:
                lsh.add(finding_id, tokens)
                clusters.make_set(finding_id)

        # Check code lines against code-scope patterns
        for line_no, code_line in code_lines:
            finding_id = f"{rel_path}:{line_no}"

            for pattern in code_patterns:
                if pattern.matches(code_line):
                    pattern_freq.add(pattern.id)

                    finding = {
                        'id': finding_id,
                        'file': str(rel_path),
                        'line': line_no,
                        'pattern': pattern.match,
                        'comment': code_line[:100],
                        'scope': 'code',
                    }
                    if pattern.implies:
                        finding['implies'] = pattern.implies
                        finding['strength'] = pattern.strength

                    findings.append(finding)

    result.comments_analyzed = comment_count
    result.findings = findings

    if verbose:
        print(f"Processed {comment_count} comments, {len(code_patterns)} code patterns")
        print()

    # Pattern frequency - check all patterns
    if verbose:
        print("PATTERN FREQUENCY (Count-Min Sketch)")
        print("-" * 40)

    for pattern in all_patterns:
        count = pattern_freq.query(pattern.id)
        if count > 0:
            result.pattern_counts[pattern.match] = count
            scope_marker = f"[{pattern.scope}]" if pattern.scope != "comments" else ""
            if verbose:
                print(f"  {pattern.match:<20} ~ {count} occurrences {scope_marker}")

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


def _extract_code_lines_from_file(filepath: Path) -> List[Tuple[int, str]]:
    """Extract non-comment code lines from a Python file with line numbers."""
    code_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                # Skip empty lines and pure comments
                if stripped and not stripped.startswith('#'):
                    code_lines.append((line_no, stripped))
    except Exception:
        pass
    return code_lines


def _extract_all_lines_from_file(filepath: Path) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Extract both comments and code lines from a Python file.

    Returns:
        Tuple of (comments, code_lines)
    """
    comments = []
    code_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    comments.append((line_no, stripped[1:].strip()))
                else:
                    code_lines.append((line_no, stripped))
    except Exception:
        pass
    return comments, code_lines


def _grep_for_pattern(
    directory: Path,
    pattern: 'AuditPattern',
    file_glob: str = "*.py"
) -> List[Dict[str, Any]]:
    """Use grep to efficiently find pattern matches in files.

    Args:
        directory: Directory to search
        pattern: AuditPattern to find
        file_glob: File pattern to search (default: *.py)

    Returns:
        List of findings with file, line, pattern info
    """
    findings = []

    try:
        # Build grep command
        if pattern.regex:
            # Use extended regex
            cmd = ["grep", "-r", "-n", "-E", "-i", "--include", file_glob, pattern.match, str(directory)]
        else:
            # Simple string search (faster)
            cmd = ["grep", "-r", "-n", "-i", "--include", file_glob, pattern.match, str(directory)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )

        # Parse grep output: file:line:content
        for line in result.stdout.strip().split('\n'):
            if not line or ':' not in line:
                continue

            # Split on first two colons: file:line:content
            parts = line.split(':', 2)
            if len(parts) < 3:
                continue

            filepath, line_no_str, content = parts
            try:
                line_no = int(line_no_str)
            except ValueError:
                continue

            # Make path relative
            try:
                rel_path = Path(filepath).relative_to(directory)
            except ValueError:
                rel_path = Path(filepath)

            # Skip comment lines - those are handled separately
            stripped = content.strip()
            if stripped.startswith('#'):
                continue

            finding = {
                'id': f"{rel_path}:{line_no}",
                'file': str(rel_path),
                'line': line_no,
                'pattern': pattern.match,
                'comment': stripped[:100],
                'scope': 'code',
            }
            if pattern.implies:
                finding['implies'] = pattern.implies
                finding['strength'] = pattern.strength

            findings.append(finding)

    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        # grep not available, fall back silently
        pass

    return findings


def _fast_code_pattern_scan(
    directory: Path,
    code_patterns: List['AuditPattern'],
    verbose: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Fast code pattern scanning using grep subprocess.

    Args:
        directory: Directory to scan
        code_patterns: List of code-scope patterns to find
        verbose: Print progress

    Returns:
        Tuple of (findings, pattern_counts)
    """
    all_findings = []
    pattern_counts = {}

    for pattern in code_patterns:
        if verbose:
            print(f"  Scanning for: {pattern.match}")

        findings = _grep_for_pattern(directory, pattern)
        all_findings.extend(findings)
        if findings:
            pattern_counts[pattern.match] = len(findings)

    return all_findings, pattern_counts
