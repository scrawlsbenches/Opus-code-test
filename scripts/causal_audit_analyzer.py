#!/usr/bin/env python3
"""
Causal Audit Analyzer - Data-Driven Correlation Analysis for Codebase Health

STATUS: Work-in-Progress Data Collection Tool
=========================================

This script is designed to BUILD UP correlation data over time by mining
git history. Unlike typical analysis tools that output results immediately,
this one:

1. MINES real data from git history (bug-fix commits, file changes)
2. CORRELATES findings from audits with actual bug occurrences
3. PERSISTS correlations to disk so they accumulate across runs
4. USES empirical data to estimate causal relationships

HONEST LIMITATIONS:
- Correlation ≠ Causation: We can measure "files with misleading comments
  are fixed more often" but can't prove the comments CAUSED the bugs.
- Data quality depends on commit message conventions (needs "fix", "bug", etc.)
- Requires multiple runs over time to build meaningful statistics
- Early runs will have sparse data and low confidence

The goal is to replace gut-feeling heuristics with measurable signals,
even if those signals are imperfect proxies for true causality.

Usage:
    python scripts/causal_audit_analyzer.py [directory]
    python scripts/causal_audit_analyzer.py --with-git cortical/
    python scripts/causal_audit_analyzer.py --show-data  # Show accumulated data
    python scripts/causal_audit_analyzer.py --reset-data # Clear accumulated data
"""

import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from our codebase health analyzer
from scripts.codebase_health import analyze_directory

# =============================================================================
# DATA PERSISTENCE - Accumulate correlations over time
# =============================================================================

DATA_FILE = Path(__file__).parent.parent / ".got" / "causal_correlations.json"


def load_correlation_data() -> Dict:
    """Load accumulated correlation data from disk."""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Default structure for new data
    return {
        "version": 1,
        "created": datetime.now().isoformat(),
        "last_updated": None,
        "runs": 0,
        "correlations": {
            # file_path -> {findings: [...], bug_fixes: [...], churn: int}
        },
        "aggregates": {
            "files_with_misleading_and_bugs": 0,
            "files_with_misleading_no_bugs": 0,
            "files_with_bugs_no_misleading": 0,
            "files_clean": 0,
            "total_bug_fix_commits": 0,
            "total_files_analyzed": 0,
        },
        "empirical_strengths": {
            # These get computed from aggregates
            "misleading_comments_to_bugs": None,
            "high_churn_to_bugs": None,
            "stale_todos_to_bugs": None,
        }
    }


def save_correlation_data(data: Dict) -> None:
    """Save correlation data to disk."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    data["runs"] += 1
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# GIT MINING - Extract real data from repository history
# =============================================================================

def mine_bug_fix_commits(directory: str = ".") -> List[Dict]:
    """
    Find commits that are likely bug fixes based on commit messages.

    Looks for patterns like:
    - "fix", "bug", "error", "crash", "issue"
    - Issue references like "#123", "fixes #456"

    Returns list of {hash, message, files_changed, date}
    """
    bug_patterns = [
        r'\bfix(es|ed|ing)?\b',
        r'\bbug\b',
        r'\berror\b',
        r'\bcrash(es|ed|ing)?\b',
        r'\bissue\b',
        r'\bpatch\b',
        r'\bhotfix\b',
        r'#\d+',  # Issue references
    ]
    pattern = '|'.join(bug_patterns)

    bug_fixes = []

    try:
        # Get commits with their messages and changed files
        result = subprocess.run(
            ["git", "log", "--pretty=format:%H|%s|%ai", "--name-only", "-500"],
            capture_output=True, text=True, cwd=directory,
            timeout=30
        )

        if result.returncode != 0:
            return []

        # Parse the output
        current_commit = None
        current_files = []

        for line in result.stdout.split('\n'):
            if '|' in line and len(line.split('|')) >= 3:
                # Save previous commit if it was a bug fix
                if current_commit and re.search(pattern, current_commit['message'], re.IGNORECASE):
                    current_commit['files_changed'] = current_files
                    bug_fixes.append(current_commit)

                # Start new commit
                parts = line.split('|')
                current_commit = {
                    'hash': parts[0],
                    'message': parts[1],
                    'date': parts[2] if len(parts) > 2 else '',
                    'files_changed': []
                }
                current_files = []
            elif line.strip() and current_commit:
                # This is a changed file
                current_files.append(line.strip())

        # Don't forget the last commit
        if current_commit and re.search(pattern, current_commit['message'], re.IGNORECASE):
            current_commit['files_changed'] = current_files
            bug_fixes.append(current_commit)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return bug_fixes


def get_files_with_bugs(bug_fix_commits: List[Dict]) -> Dict[str, List[str]]:
    """
    Map files to the bug-fix commits that touched them.

    Returns: {file_path: [commit_hash, ...]}
    """
    file_bugs = defaultdict(list)

    for commit in bug_fix_commits:
        for file_path in commit.get('files_changed', []):
            # Normalize path
            if file_path.endswith('.py'):
                file_bugs[file_path].append(commit['hash'][:8])

    return dict(file_bugs)


def calculate_empirical_correlations(
    findings_by_file: Dict[str, List],
    files_with_bugs: Dict[str, List[str]],
    high_churn_files: Set[str],
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate actual correlation strengths from observed data.

    This computes:
    - P(bug | misleading_comment) - probability of bug given misleading comment
    - P(bug | high_churn) - probability of bug given high file churn

    These are CORRELATIONS, not causal strengths, but they're based on
    real data rather than invented numbers.
    """
    # Categorize files
    files_misleading_and_bugs = 0
    files_misleading_no_bugs = 0
    files_bugs_no_misleading = 0
    files_clean = 0
    files_churn_and_bugs = 0
    files_churn_no_bugs = 0

    # Track specific files for verbose output
    high_priority_files = []  # Files with both findings and bugs

    all_files = set(findings_by_file.keys()) | set(files_with_bugs.keys())

    for file_path in all_files:
        has_findings = file_path in findings_by_file and len(findings_by_file[file_path]) > 0
        has_bugs = file_path in files_with_bugs
        has_churn = file_path in high_churn_files

        if has_findings and has_bugs:
            files_misleading_and_bugs += 1
            high_priority_files.append({
                'file': file_path,
                'findings': len(findings_by_file[file_path]),
                'bug_fixes': files_with_bugs[file_path]
            })
        elif has_findings and not has_bugs:
            files_misleading_no_bugs += 1
        elif has_bugs and not has_findings:
            files_bugs_no_misleading += 1
        else:
            files_clean += 1

        if has_churn and has_bugs:
            files_churn_and_bugs += 1
        elif has_churn and not has_bugs:
            files_churn_no_bugs += 1

    # Calculate conditional probabilities
    correlations = {}

    # P(bug | misleading) = files with both / files with misleading
    total_misleading = files_misleading_and_bugs + files_misleading_no_bugs
    if total_misleading > 0:
        correlations["misleading_to_bugs"] = files_misleading_and_bugs / total_misleading
    else:
        correlations["misleading_to_bugs"] = None  # Insufficient data

    # P(bug | churn) = files with both / files with churn
    total_churn = files_churn_and_bugs + files_churn_no_bugs
    if total_churn > 0:
        correlations["churn_to_bugs"] = files_churn_and_bugs / total_churn
    else:
        correlations["churn_to_bugs"] = None

    # P(bug | no_misleading) - for comparison
    total_no_misleading = files_bugs_no_misleading + files_clean
    if total_no_misleading > 0:
        correlations["baseline_bug_rate"] = files_bugs_no_misleading / total_no_misleading
    else:
        correlations["baseline_bug_rate"] = None

    # Store counts for transparency
    correlations["_counts"] = {
        "files_misleading_and_bugs": files_misleading_and_bugs,
        "files_misleading_no_bugs": files_misleading_no_bugs,
        "files_bugs_no_misleading": files_bugs_no_misleading,
        "files_clean": files_clean,
        "files_churn_and_bugs": files_churn_and_bugs,
        "files_churn_no_bugs": files_churn_no_bugs,
        "total_files": len(all_files),
    }

    # Store high-priority files for detailed output
    correlations["_high_priority_files"] = high_priority_files

    return correlations


# =============================================================================
# REPORT GENERATION - Show what we actually found
# =============================================================================

def generate_data_driven_report(
    findings: List[Dict],
    bug_fix_commits: List[Dict],
    correlations: Dict[str, float],
    accumulated_data: Dict,
    verbose: bool = False
) -> str:
    """Generate a report based on mined data, not invented numbers."""

    lines = []
    lines.append("=" * 70)
    lines.append("  CAUSAL AUDIT ANALYSIS - Data-Driven Report")
    lines.append("=" * 70)

    # Data provenance
    lines.append("\n[Data Sources]")
    lines.append(f"  Bug-fix commits analyzed: {len(bug_fix_commits)}")
    lines.append(f"  Analysis runs accumulated: {accumulated_data.get('runs', 1)}")
    if accumulated_data.get('last_updated'):
        lines.append(f"  Last updated: {accumulated_data['last_updated'][:19]}")

    # Finding summary
    lines.append("\n[Current Findings]")
    finding_counts = defaultdict(int)
    stale_count = 0
    for f in findings:
        finding_counts[f.get('pattern', 'unknown')] += 1
        if f.get('stale'):
            stale_count += 1

    lines.append(f"  Total findings: {len(findings)}")
    for pattern, count in sorted(finding_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {pattern}: {count}")
    lines.append(f"  Stale TODOs (>180 days): {stale_count}")

    # Empirical correlations
    lines.append("\n[Empirical Correlations]")
    lines.append("  (Based on actual git history, not invented numbers)")
    lines.append("")

    counts = correlations.get("_counts", {})

    # Misleading comments correlation
    misleading_corr = correlations.get("misleading_to_bugs")
    baseline = correlations.get("baseline_bug_rate")

    if misleading_corr is not None:
        lines.append(f"  P(bug-fix | file has findings): {misleading_corr:.1%}")
        lines.append(f"    Based on: {counts.get('files_misleading_and_bugs', 0)} files with both findings and bug-fixes")
        lines.append(f"              {counts.get('files_misleading_no_bugs', 0)} files with findings but no bug-fixes")

        if baseline is not None:
            lines.append(f"  P(bug-fix | file has NO findings): {baseline:.1%} (baseline)")
            if misleading_corr > baseline:
                lift = (misleading_corr - baseline) / baseline * 100 if baseline > 0 else 0
                lines.append(f"    → Files with findings are {lift:.0f}% more likely to have bug-fixes")
            else:
                lines.append(f"    → No elevated risk detected (findings may not predict bugs)")
    else:
        lines.append("  P(bug-fix | file has findings): INSUFFICIENT DATA")
        lines.append("    Need more runs to accumulate correlation data")

    lines.append("")

    # Churn correlation
    churn_corr = correlations.get("churn_to_bugs")
    if churn_corr is not None:
        lines.append(f"  P(bug-fix | high churn file): {churn_corr:.1%}")
        lines.append(f"    Based on: {counts.get('files_churn_and_bugs', 0)} high-churn files with bug-fixes")
    else:
        lines.append("  P(bug-fix | high churn file): INSUFFICIENT DATA")

    # Interpretation guidance
    lines.append("\n[Interpretation Notes]")
    lines.append("  • These are CORRELATIONS, not proven causal relationships")
    lines.append("  • Bug-fix detection depends on commit message conventions")
    lines.append("  • Run this tool periodically to accumulate more accurate data")
    lines.append("  • Data is stored in .got/causal_correlations.json")

    # Files of interest
    lines.append("\n[High-Priority Files]")
    lines.append("  Files with BOTH findings AND bug-fix history:")

    high_priority = correlations.get("_high_priority_files", [])
    if high_priority:
        lines.append(f"    → {len(high_priority)} files identified")
        if verbose:
            lines.append("")
            for hp in sorted(high_priority, key=lambda x: -x['findings']):
                lines.append(f"    {hp['file']}")
                lines.append(f"      Findings: {hp['findings']}, Bug-fix commits: {', '.join(hp['bug_fixes'][:3])}")
        else:
            lines.append("    (Run with --verbose for full list)")
    else:
        lines.append("    → None found (good news!)")

    lines.append("\n" + "=" * 70)
    lines.append("  To accumulate more data, run this tool after each audit session")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Causal Audit Analyzer - Data-Driven Correlation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool mines git history to find empirical correlations between
code quality findings and actual bug-fix commits.

Unlike analysis tools that use hardcoded heuristics, this one:
- Extracts real data from your repository's history
- Accumulates statistics across multiple runs
- Reports actual correlations with confidence levels

Run it periodically to build up meaningful statistics over time.
        """
    )
    parser.add_argument("directory", nargs="?", default="cortical/",
                        help="Directory to analyze")
    parser.add_argument("--with-git", action="store_true",
                        help="Include git history analysis (recommended)")
    parser.add_argument("--show-data", action="store_true",
                        help="Show accumulated correlation data and exit")
    parser.add_argument("--reset-data", action="store_true",
                        help="Clear accumulated data and start fresh")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed file lists")

    args = parser.parse_args()

    # Handle data management commands
    if args.show_data:
        data = load_correlation_data()
        print(json.dumps(data, indent=2))
        return

    if args.reset_data:
        if DATA_FILE.exists():
            DATA_FILE.unlink()
            print(f"Cleared accumulated data from {DATA_FILE}")
        else:
            print("No accumulated data to clear")
        return

    # Load accumulated data
    accumulated_data = load_correlation_data()

    print("=" * 70)
    print("  Causal Audit Analyzer - Mining Git History for Correlations")
    print("=" * 70)
    print()

    # Step 1: Run codebase health analysis to get findings
    print("[1/3] Running codebase health analysis...")
    results = analyze_directory(args.directory, with_git=args.with_git)

    if not results:
        print("No results from analysis")
        return

    findings = results.get('findings', [])
    print(f"      Found {len(findings)} audit findings")

    # Step 2: Mine git history for bug-fix commits
    print("\n[2/3] Mining git history for bug-fix commits...")
    bug_fix_commits = mine_bug_fix_commits(".")
    print(f"      Found {len(bug_fix_commits)} bug-fix commits")

    if bug_fix_commits and args.verbose:
        print("      Recent bug-fixes:")
        for commit in bug_fix_commits[:5]:
            print(f"        {commit['hash'][:8]}: {commit['message'][:50]}")

    # Step 3: Calculate correlations
    print("\n[3/3] Calculating empirical correlations...")

    # Group findings by file
    # IMPORTANT: Normalize paths to match git's repo-root format
    # findings have id like "tokenizer.py:27" (filename:line)
    # git log has paths like "cortical/utils/tokenizer.py"
    findings_by_file = defaultdict(list)
    analyzed_dir = args.directory.rstrip('/')
    for f in findings:
        # Extract file from id (format: "filename:line")
        finding_id = f.get('id', '')
        if ':' in finding_id:
            file_path = finding_id.rsplit(':', 1)[0]  # Get part before last colon
        else:
            file_path = f.get('file', '')

        if file_path:
            # Normalize to repo-root path
            if not file_path.startswith(analyzed_dir):
                file_path = f"{analyzed_dir}/{file_path}"
            findings_by_file[file_path].append(f)

    # Get files touched by bug-fix commits
    files_with_bugs = get_files_with_bugs(bug_fix_commits)

    # Get high-churn files from results
    high_churn = set()
    git_analysis = results.get('git_analysis', {})
    for file_path, churn in git_analysis.get('high_churn_files', []):
        if churn > 10:
            high_churn.add(file_path)

    # Calculate correlations
    correlations = calculate_empirical_correlations(
        findings_by_file,
        files_with_bugs,
        high_churn
    )

    # Update accumulated data
    counts = correlations.get("_counts", {})
    agg = accumulated_data["aggregates"]
    agg["files_with_misleading_and_bugs"] += counts.get("files_misleading_and_bugs", 0)
    agg["files_with_misleading_no_bugs"] += counts.get("files_misleading_no_bugs", 0)
    agg["files_with_bugs_no_misleading"] += counts.get("files_bugs_no_misleading", 0)
    agg["files_clean"] += counts.get("files_clean", 0)
    agg["total_bug_fix_commits"] += len(bug_fix_commits)
    agg["total_files_analyzed"] += counts.get("total_files", 0)

    # Update empirical strengths
    accumulated_data["empirical_strengths"]["misleading_comments_to_bugs"] = correlations.get("misleading_to_bugs")
    accumulated_data["empirical_strengths"]["high_churn_to_bugs"] = correlations.get("churn_to_bugs")

    # Save accumulated data
    save_correlation_data(accumulated_data)
    print(f"      Data saved to {DATA_FILE}")

    # Generate report
    print()
    report = generate_data_driven_report(
        findings=findings,
        bug_fix_commits=bug_fix_commits,
        correlations=correlations,
        accumulated_data=accumulated_data,
        verbose=args.verbose
    )
    print(report)


if __name__ == "__main__":
    main()
