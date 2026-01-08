#!/usr/bin/env python3
"""
Synthetic Audit Data Generator

Generates synthetic codebase health audit data with INTENTIONAL, DISCOVERABLE patterns
for testing WovenMind and PLN (Pattern Learning Network) with Full PLN features.

The synthetic data contains:
  - 100+ file findings with comment patterns
  - Git analysis (high churn files, bug-prone files)
  - Embedded patterns that should be discoverable by the system
  - Importance metadata (STI/LTI/VLTI) for Full PLN attention testing

INTENTIONAL PATTERNS EMBEDDED:
================================

PATTERN SET 1 - Directory Correlations:
  - All files in "legacy/" have TODO patterns (100% correlation)
  - Files in "api/" have "should be" patterns (80% correlation)
  - Files in "utils/" have "see docs" patterns (70% correlation)

PATTERN SET 2 - Trait Correlations:
  - High churn files tend to have "FIXME" patterns (75% correlation)
  - Bug-prone files tend to have multiple pattern types (3+ patterns per file)

PATTERN SET 3 - Cross-Cutting:
  - Files with BOTH "TODO:" and "HACK:" always have high churn (100% correlation)
  - Files with "FUTURE:" pattern are never bug-prone (100% anti-correlation)

PATTERN SET 4 - Surprising Outliers:
  - 2-3 files that break the normal patterns (for surprise detection)
    Example: legacy/clean_module.py has NO TODOs (surprising!)

IMPORTANCE METADATA (Full PLN):
================================
Each file now includes importance values for Full PLN testing:
  - STI (Short-Term Importance): Based on recent activity and traits
  - LTI (Long-Term Importance): Based on historical significance
  - VLTI (Very Long-Term Importance): Critical files that should never decay
  - Priority tier: Computed from combined importance

Usage:
    python scripts/generate_synthetic_audit.py                    # Print to stdout
    python scripts/generate_synthetic_audit.py --save             # Save to .got/synthetic_audit_data.json
    python scripts/generate_synthetic_audit.py --save --verbose   # Save with detailed output

Example - Using in Tests:
    # Generate and save synthetic data
    import subprocess
    subprocess.run(['python', 'scripts/generate_synthetic_audit.py', '--save'])

    # Load in your test
    import json
    with open('.got/synthetic_audit_data.json') as f:
        audit_data = json.load(f)

    # Verify pattern discovery
    # Expected: WovenMind should discover that legacy/ files → TODO correlation
    # Expected: PLN should learn that TODO+HACK → high churn

    # Access metadata for validation
    metadata = audit_data['_metadata']
    outliers = metadata['outlier_files']  # Files that break patterns
    expected_patterns = metadata['pattern_sets']  # Expected correlations
    importance_map = audit_data['file_importance']  # STI/LTI/VLTI per file
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

# Seed for reproducibility
random.seed(42)


# =============================================================================
# IMPORTANCE METADATA (Full PLN Support)
# =============================================================================

@dataclass
class FileImportance:
    """Importance metadata for Full PLN attention tracking."""
    sti: float  # Short-term importance (0.0 - 1.0)
    lti: float  # Long-term importance (0.0 - 1.0)
    vlti: bool  # Very long-term importance (pinned)
    priority_tier: str  # "critical", "high", "medium", "low"

    def total_importance(self) -> float:
        """Calculate combined importance (matching PLN formula)."""
        return 0.6 * self.sti + 0.4 * self.lti


def calculate_file_importance(
    filepath: str,
    is_high_churn: bool,
    is_bug_prone: bool,
    is_critical: bool,
    pattern_count: int
) -> FileImportance:
    """
    Calculate importance values for a file based on its characteristics.

    STI (Short-Term Importance):
      - Base: 0.2
      - +0.3 if high churn (recently active)
      - +0.2 if bug-prone (needs attention)
      - +0.1 per pattern beyond 2 (more issues = more important)

    LTI (Long-Term Importance):
      - Base: 0.1
      - +0.3 if critical module
      - +0.1 if bug-prone (historically problematic)

    VLTI (Very Long-Term Importance):
      - True only for critical modules (never decay)

    Priority Tier:
      - critical: VLTI=True or total >= 0.7
      - high: total >= 0.5
      - medium: total >= 0.3
      - low: total < 0.3
    """
    # Calculate STI
    sti = 0.2  # Base
    if is_high_churn:
        sti += 0.3
    if is_bug_prone:
        sti += 0.2
    if pattern_count > 2:
        sti += 0.1 * min(3, pattern_count - 2)  # Cap at +0.3
    sti = min(1.0, sti)

    # Calculate LTI
    lti = 0.1  # Base
    if is_critical:
        lti += 0.3
    if is_bug_prone:
        lti += 0.1
    lti = min(1.0, lti)

    # VLTI for critical modules only
    vlti = is_critical

    # Calculate priority tier
    total = 0.6 * sti + 0.4 * lti
    if vlti or total >= 0.7:
        tier = "critical"
    elif total >= 0.5:
        tier = "high"
    elif total >= 0.3:
        tier = "medium"
    else:
        tier = "low"

    return FileImportance(sti=sti, lti=lti, vlti=vlti, priority_tier=tier)


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

PATTERNS = {
    'TODO:': "TODO: Implement proper validation here",
    'FIXME:': "FIXME: This breaks under edge cases",
    'HACK:': "HACK: Temporary workaround for the race condition",
    'FUTURE:': "FUTURE: Will be replaced with new architecture",
    'XXX:': "XXX: This is a known limitation",
    'should be': "This should be refactored to use the new API",
    'will be': "This will be deprecated in the next release",
    'see docs': "See docs/architecture.md for details",
    'eventually': "This will eventually be moved to the utils module",
    'planned to': "We planned to optimize this but ran out of time",
}


# Directory structure for synthetic files
DIRECTORIES = {
    'legacy': 25,      # 25 files in legacy/
    'api': 30,         # 30 files in api/
    'utils': 20,       # 20 files in utils/
    'core': 15,        # 15 files in core/
    'services': 20,    # 20 files in services/
}


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_file_list() -> List[str]:
    """Generate a list of synthetic file paths."""
    files = []
    for directory, count in DIRECTORIES.items():
        for i in range(count):
            filename = f"module_{i:02d}.py"
            files.append(f"synthetic/{directory}/{filename}")
    return files


def should_apply_pattern(probability: float) -> bool:
    """Return True with given probability."""
    return random.random() < probability


def generate_findings_for_file(
    filepath: str,
    is_high_churn: bool,
    is_bug_prone: bool
) -> List[Dict]:
    """Generate findings for a single file according to the pattern rules."""
    findings = []
    directory = filepath.split('/')[1]  # Extract directory name

    patterns_to_add = []

    # ==========================================================================
    # PATTERN SET 1 - Directory Correlations
    # ==========================================================================

    if directory == 'legacy':
        # 100% of legacy files have TODOs (will remove some later for outliers)
        patterns_to_add.append('TODO:')

    if directory == 'api':
        # 80% of api files have "should be" patterns
        if should_apply_pattern(0.80):
            patterns_to_add.append('should be')

    if directory == 'utils':
        # 70% of utils files have "see docs" patterns
        if should_apply_pattern(0.70):
            patterns_to_add.append('see docs')

    # ==========================================================================
    # PATTERN SET 2 - Trait Correlations
    # ==========================================================================

    if is_high_churn:
        # 75% of high churn files have FIXME patterns
        if should_apply_pattern(0.75):
            patterns_to_add.append('FIXME:')

    if is_bug_prone:
        # Bug-prone files have multiple pattern types (3+ patterns)
        # Add extra patterns to reach at least 3
        extra_patterns = ['XXX:', 'HACK:', 'eventually']
        for pattern in extra_patterns:
            if pattern not in patterns_to_add:
                patterns_to_add.append(pattern)
                if len(patterns_to_add) >= 3:
                    break

    # ==========================================================================
    # PATTERN SET 3 - Cross-Cutting
    # ==========================================================================

    # Files with "FUTURE:" are never bug-prone (anti-correlation)
    # This means if we're bug-prone, we shouldn't add FUTURE
    if not is_bug_prone:
        # Add FUTURE to some non-bug-prone files
        if should_apply_pattern(0.20):
            patterns_to_add.append('FUTURE:')

    # Files with BOTH TODO and HACK always have high churn
    # If we have both, ensure high_churn flag is set (handled by caller)

    # ==========================================================================
    # Generate findings from patterns
    # ==========================================================================

    for pattern in patterns_to_add:
        line_no = random.randint(10, 500)
        finding_id = f"{filepath}:{line_no}"

        age_days = random.randint(1, 400)
        author = random.choice(['alice', 'bob', 'charlie', 'diana', 'eve'])

        finding = {
            'id': finding_id,
            'pattern': pattern,
            'comment': PATTERNS.get(pattern, f"Comment with {pattern}"),
            'age_days': age_days,
            'author': author,
        }

        # Mark stale TODOs/FIXMEs (> 180 days)
        if pattern in ['TODO:', 'FIXME:'] and age_days > 180:
            finding['stale'] = True

        findings.append(finding)

    return findings


def generate_git_analysis(files: List[str]) -> Dict:
    """Generate synthetic git analysis data."""

    # Select high churn files (20% of total)
    num_high_churn = int(len(files) * 0.20)
    high_churn_files = random.sample(files, num_high_churn)

    # Generate churn counts (higher for high churn files)
    high_churn_data = []
    for filepath in high_churn_files:
        churn_count = random.randint(15, 50)
        high_churn_data.append((filepath, churn_count))

    # Sort by churn count
    high_churn_data.sort(key=lambda x: -x[1])

    # Select bug-prone files (15% of total)
    # Bug-prone files are a subset that often overlaps with high churn
    num_bug_prone = int(len(files) * 0.15)
    # 70% overlap with high churn files
    overlap_count = int(num_bug_prone * 0.70)
    bug_prone_files = random.sample(high_churn_files, min(overlap_count, len(high_churn_files)))
    remaining = num_bug_prone - len(bug_prone_files)
    if remaining > 0:
        non_churn_files = [f for f in files if f not in high_churn_files]
        bug_prone_files.extend(random.sample(non_churn_files, min(remaining, len(non_churn_files))))

    # Generate bug counts
    bug_prone_data = []
    for filepath in bug_prone_files:
        bug_count = random.randint(3, 12)
        bug_prone_data.append((filepath, bug_count))

    bug_prone_data.sort(key=lambda x: -x[1])

    return {
        'high_churn_files': high_churn_data[:20],  # Top 20
        'bug_prone_files': bug_prone_data,
        'suspicious_commits': [
            {'hash': 'a1b2c3d4', 'message': 'Quick fix for login bug', 'date': '2025-12-15', 'pattern': 'quick fix'},
            {'hash': 'e5f6a7b8', 'message': 'TODO: Clean this up later', 'date': '2025-12-10', 'pattern': 'todo'},
            {'hash': 'c9d0e1f2', 'message': 'Temporary workaround for API', 'date': '2025-12-01', 'pattern': 'temporary'},
        ],
        'critical_modules': [
            ('synthetic/core/base.py', 15),
            ('synthetic/core/config.py', 12),
            ('synthetic/utils/helpers.py', 10),
        ]
    }


def apply_outliers(findings: List[Dict], files: List[str]) -> Tuple[List[Dict], Set[str]]:
    """
    Apply surprising outliers to the data.

    PATTERN SET 4 - Surprising Outliers:
    - Remove TODOs from 2-3 legacy files (breaks the 100% correlation)
    - These should be flagged as surprising by the pattern discovery system

    Returns:
        Tuple of (modified findings list, set of outlier file paths)
    """

    # Find legacy files with TODOs
    legacy_findings = defaultdict(list)
    for finding in findings:
        filepath = finding['id'].split(':')[0]
        if 'legacy/' in filepath:
            legacy_findings[filepath].append(finding)

    # Select 2-3 legacy files to make "clean" (surprising!)
    legacy_files_with_todos = [
        fp for fp, fds in legacy_findings.items()
        if any(f['pattern'] == 'TODO:' for f in fds)
    ]

    outlier_files = set()
    if len(legacy_files_with_todos) >= 3:
        num_outliers = random.choice([2, 3])  # 2 or 3 outliers
        outlier_files = set(random.sample(legacy_files_with_todos, num_outliers))

        # Remove ALL patterns from these files to make them truly clean
        findings = [
            f for f in findings
            if f['id'].split(':')[0] not in outlier_files
        ]

        print(f"[Outlier] Created {num_outliers} clean legacy files: {sorted(outlier_files)}")

    return findings, outlier_files


def generate_synthetic_audit_data(apply_outliers_flag: bool = True) -> Dict:
    """Generate complete synthetic audit data matching codebase_health.analyze_directory() format."""

    files = generate_file_list()
    git_analysis = generate_git_analysis(files)

    # Extract high churn and bug-prone file sets
    high_churn_set = {fp for fp, _ in git_analysis['high_churn_files']}
    bug_prone_set = {fp for fp, _ in git_analysis['bug_prone_files']}

    # ==========================================================================
    # PATTERN SET 3 - Enforce cross-cutting rules
    # ==========================================================================

    # Files with BOTH TODO and HACK should ALL be in high churn (100% correlation)
    # Let's identify some files and add both patterns + mark as high churn
    num_todo_hack_files = 10
    available_files = [f for f in files if f not in high_churn_set]
    todo_hack_files = set(random.sample(available_files, min(num_todo_hack_files, len(available_files))))

    # Add these files to high churn set
    for filepath in todo_hack_files:
        high_churn_set.add(filepath)

    # Add these to git analysis with high churn counts
    for filepath in todo_hack_files:
        churn_count = random.randint(25, 45)
        git_analysis['high_churn_files'].append((filepath, churn_count))

    # Re-sort
    git_analysis['high_churn_files'].sort(key=lambda x: -x[1])
    git_analysis['high_churn_files'] = git_analysis['high_churn_files'][:25]  # Keep top 25

    # ==========================================================================
    # Generate findings for all files
    # ==========================================================================

    all_findings = []
    pattern_counts = defaultdict(int)

    for filepath in files:
        is_high_churn = filepath in high_churn_set
        is_bug_prone = filepath in bug_prone_set

        findings = generate_findings_for_file(filepath, is_high_churn, is_bug_prone)

        # For TODO+HACK cross-cutting pattern files, ALWAYS add both patterns
        if filepath in todo_hack_files:
            # Ensure this file has both TODO and HACK
            has_todo = any(f['pattern'] == 'TODO:' for f in findings)
            has_hack = any(f['pattern'] == 'HACK:' for f in findings)

            if not has_todo:
                findings.append({
                    'id': f"{filepath}:{random.randint(10, 500)}",
                    'pattern': 'TODO:',
                    'comment': PATTERNS['TODO:'],
                    'age_days': random.randint(1, 400),
                    'author': random.choice(['alice', 'bob', 'charlie']),
                })

            if not has_hack:
                findings.append({
                    'id': f"{filepath}:{random.randint(10, 500)}",
                    'pattern': 'HACK:',
                    'comment': PATTERNS['HACK:'],
                    'age_days': random.randint(1, 400),
                    'author': random.choice(['alice', 'bob', 'charlie']),
                })

        all_findings.extend(findings)

        # Update pattern counts
        for finding in findings:
            pattern_counts[finding['pattern']] += 1

    # ==========================================================================
    # Apply outliers (surprising patterns)
    # ==========================================================================

    outlier_files = set()
    if apply_outliers_flag:
        all_findings, outlier_files = apply_outliers(all_findings, files)
        # Recalculate pattern counts after outlier removal
        pattern_counts = defaultdict(int)
        for finding in all_findings:
            pattern_counts[finding['pattern']] += 1

    # ==========================================================================
    # Build final result structure
    # ==========================================================================

    # Calculate file importance for each file
    file_importance: Dict[str, Dict[str, Any]] = {}
    file_patterns = defaultdict(set)
    for finding in all_findings:
        filepath = finding['id'].split(':')[0]
        file_patterns[filepath].add(finding['pattern'])

    # Get critical modules from git analysis
    critical_set = {fp for fp, _ in git_analysis.get('critical_modules', [])}

    for filepath in files:
        is_high_churn = filepath in high_churn_set
        is_bug_prone = filepath in bug_prone_set
        is_critical = filepath in critical_set
        pattern_count = len(file_patterns.get(filepath, set()))

        importance = calculate_file_importance(
            filepath, is_high_churn, is_bug_prone, is_critical, pattern_count
        )
        file_importance[filepath] = asdict(importance)
        file_importance[filepath]['total'] = importance.total_importance()

    # Count priority tiers
    tier_counts = defaultdict(int)
    for imp in file_importance.values():
        tier_counts[imp['priority_tier']] += 1

    result = {
        'files_analyzed': len(files),
        'comments_analyzed': len(all_findings),
        'findings': all_findings,
        'pattern_counts': dict(pattern_counts),
        'similar_groups': [],  # Not critical for WovenMind testing
        'repeated_substrings': [],  # Not critical for WovenMind testing
        'git_analysis': git_analysis,
        # Full PLN importance metadata
        'file_importance': file_importance,
        # Metadata about embedded patterns (for validation)
        '_metadata': {
            'pattern_sets': {
                'directory_correlations': {
                    'legacy_todo': 100.0,  # Expected percentage
                    'api_should_be': 80.0,
                    'utils_see_docs': 70.0,
                },
                'trait_correlations': {
                    'high_churn_fixme': 75.0,
                    'bug_prone_multi_pattern': 100.0,
                },
                'cross_cutting': {
                    'todo_hack_high_churn': 100.0,
                    'future_not_bug_prone': 100.0,
                },
            },
            'outlier_files': sorted(outlier_files),
            'todo_hack_files': sorted(todo_hack_files),
            # Full PLN metadata
            'importance_tiers': {
                'critical': tier_counts['critical'],
                'high': tier_counts['high'],
                'medium': tier_counts['medium'],
                'low': tier_counts['low'],
            },
            'vlti_files': [fp for fp, imp in file_importance.items() if imp['vlti']],
        }
    }

    return result


# =============================================================================
# PATTERN DOCUMENTATION
# =============================================================================

def print_pattern_summary(data: Dict) -> None:
    """Print a summary of the embedded patterns."""

    print()
    print("=" * 80)
    print("SYNTHETIC AUDIT DATA - EMBEDDED PATTERNS SUMMARY")
    print("=" * 80)
    print()

    # Build directory → pattern mapping
    dir_patterns = defaultdict(lambda: defaultdict(int))
    file_patterns = defaultdict(set)

    for finding in data['findings']:
        filepath = finding['id'].split(':')[0]
        directory = filepath.split('/')[1]
        pattern = finding['pattern']

        dir_patterns[directory][pattern] += 1
        file_patterns[filepath].add(pattern)

    # Get all files (including outliers) from the metadata
    # We need to reconstruct all files to properly count outliers
    all_files_with_findings = set()
    for finding in data['findings']:
        filepath = finding['id'].split(':')[0]
        all_files_with_findings.add(filepath)

    # Add outlier files (they won't be in findings)
    outlier_files = set(data['_metadata']['outlier_files'])
    all_files = all_files_with_findings | outlier_files

    # Count unique files per directory
    dir_unique_files = defaultdict(set)
    for filepath in all_files:
        directory = filepath.split('/')[1]
        dir_unique_files[directory].add(filepath)

    dir_file_counts = {d: len(files) for d, files in dir_unique_files.items()}

    # ==========================================================================
    # PATTERN SET 1 - Directory Correlations
    # ==========================================================================

    print("PATTERN SET 1 - Directory Correlations")
    print("-" * 80)
    print()

    # TODO in legacy/
    legacy_files = [f for f in all_files if '/legacy/' in f]
    legacy_with_todo = [f for f in legacy_files if 'TODO:' in file_patterns[f]]
    todo_percentage = (len(legacy_with_todo) / len(legacy_files) * 100) if legacy_files else 0
    print(f"  ✓ Files in legacy/ with TODO patterns:")
    print(f"    {len(legacy_with_todo)}/{len(legacy_files)} ({todo_percentage:.1f}%)")
    print(f"    Expected: 100% (with 2 outliers → ~92%)")
    print()

    # "should be" in api/
    api_files = [f for f in all_files if '/api/' in f]
    api_with_should = [f for f in api_files if 'should be' in file_patterns[f]]
    should_percentage = (len(api_with_should) / len(api_files) * 100) if api_files else 0
    print(f"  ✓ Files in api/ with 'should be' patterns:")
    print(f"    {len(api_with_should)}/{len(api_files)} ({should_percentage:.1f}%)")
    print(f"    Expected: ~80%")
    print()

    # "see docs" in utils/
    utils_files = [f for f in all_files if '/utils/' in f]
    utils_with_docs = [f for f in utils_files if 'see docs' in file_patterns[f]]
    docs_percentage = (len(utils_with_docs) / len(utils_files) * 100) if utils_files else 0
    print(f"  ✓ Files in utils/ with 'see docs' patterns:")
    print(f"    {len(utils_with_docs)}/{len(utils_files)} ({docs_percentage:.1f}%)")
    print(f"    Expected: ~70%")
    print()

    # ==========================================================================
    # PATTERN SET 2 - Trait Correlations
    # ==========================================================================

    print("PATTERN SET 2 - Trait Correlations")
    print("-" * 80)
    print()

    high_churn_files = {fp for fp, _ in data['git_analysis']['high_churn_files']}
    high_churn_with_fixme = [f for f in high_churn_files if 'FIXME:' in file_patterns[f]]
    fixme_percentage = (len(high_churn_with_fixme) / len(high_churn_files) * 100) if high_churn_files else 0
    print(f"  ✓ High churn files with FIXME patterns:")
    print(f"    {len(high_churn_with_fixme)}/{len(high_churn_files)} ({fixme_percentage:.1f}%)")
    print(f"    Expected: ~75%")
    print()

    bug_prone_files = {fp for fp, _ in data['git_analysis']['bug_prone_files']}
    bug_prone_multi = [f for f in bug_prone_files if len(file_patterns[f]) >= 3]
    multi_percentage = (len(bug_prone_multi) / len(bug_prone_files) * 100) if bug_prone_files else 0
    print(f"  ✓ Bug-prone files with 3+ pattern types:")
    print(f"    {len(bug_prone_multi)}/{len(bug_prone_files)} ({multi_percentage:.1f}%)")
    print(f"    Expected: ~100%")
    print()

    # ==========================================================================
    # PATTERN SET 3 - Cross-Cutting
    # ==========================================================================

    print("PATTERN SET 3 - Cross-Cutting Patterns")
    print("-" * 80)
    print()

    todo_hack_files = [
        f for f in all_files
        if 'TODO:' in file_patterns[f] and 'HACK:' in file_patterns[f]
    ]
    todo_hack_in_churn = [f for f in todo_hack_files if f in high_churn_files]
    cross_percentage = (len(todo_hack_in_churn) / len(todo_hack_files) * 100) if todo_hack_files else 0
    print(f"  ✓ Files with BOTH TODO and HACK that have high churn:")
    print(f"    {len(todo_hack_in_churn)}/{len(todo_hack_files)} ({cross_percentage:.1f}%)")
    print(f"    Expected: 100%")
    print()

    future_files = [f for f in all_files if 'FUTURE:' in file_patterns[f]]
    future_bug_prone = [f for f in future_files if f in bug_prone_files]
    anti_percentage = (len(future_bug_prone) / len(future_files) * 100) if future_files else 0
    print(f"  ✓ Files with FUTURE pattern that are bug-prone:")
    print(f"    {len(future_bug_prone)}/{len(future_files)} ({anti_percentage:.1f}%)")
    print(f"    Expected: 0% (anti-correlation)")
    print()

    # ==========================================================================
    # PATTERN SET 4 - Outliers
    # ==========================================================================

    print("PATTERN SET 4 - Surprising Outliers")
    print("-" * 80)
    print()

    # Outlier files are those removed from findings entirely
    legacy_no_findings = [f for f in legacy_files if f in outlier_files]
    print(f"  ✓ Legacy files WITHOUT any findings (surprising!):")
    print(f"    Count: {len(legacy_no_findings)}")
    print(f"    Expected: 2-3 outliers")
    if legacy_no_findings:
        for filepath in sorted(legacy_no_findings):
            print(f"      - {filepath}")
    else:
        print(f"      (Note: Check _metadata.outlier_files in saved JSON)")
    print()

    # ==========================================================================
    # Overall Statistics
    # ==========================================================================

    print("Overall Statistics")
    print("-" * 80)
    print()
    print(f"  Total files analyzed: {data['files_analyzed']}")
    print(f"  Total findings: {data['comments_analyzed']}")
    print(f"  High churn files: {len(high_churn_files)}")
    print(f"  Bug-prone files: {len(bug_prone_files)}")
    print()
    print("  Pattern distribution:")
    for pattern, count in sorted(data['pattern_counts'].items(), key=lambda x: -x[1]):
        print(f"    {pattern:<15} {count:>4} occurrences")
    print()

    # ==========================================================================
    # Full PLN Importance Statistics
    # ==========================================================================

    print("Full PLN Importance Statistics")
    print("-" * 80)
    print()

    importance_tiers = data['_metadata'].get('importance_tiers', {})
    print("  Priority tiers:")
    print(f"    Critical: {importance_tiers.get('critical', 0)} files")
    print(f"    High:     {importance_tiers.get('high', 0)} files")
    print(f"    Medium:   {importance_tiers.get('medium', 0)} files")
    print(f"    Low:      {importance_tiers.get('low', 0)} files")
    print()

    vlti_files = data['_metadata'].get('vlti_files', [])
    print(f"  VLTI (pinned) files: {len(vlti_files)}")
    for filepath in vlti_files[:5]:
        print(f"    • {filepath}")
    print()

    # Show top files by importance
    file_importance = data.get('file_importance', {})
    sorted_by_importance = sorted(
        file_importance.items(),
        key=lambda x: x[1].get('total', 0),
        reverse=True
    )
    print("  Top 5 files by importance:")
    for filepath, imp in sorted_by_importance[:5]:
        sti = imp.get('sti', 0)
        lti = imp.get('lti', 0)
        total = imp.get('total', 0)
        tier = imp.get('priority_tier', 'unknown')
        vlti_mark = " [VLTI]" if imp.get('vlti', False) else ""
        print(f"    {total:.2%} - {filepath} ({tier}){vlti_mark}")
        print(f"           STI: {sti:.2f}, LTI: {lti:.2f}")
    print()

    print("=" * 80)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic audit data with discoverable patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save to .got/synthetic_audit_data.json'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed pattern summary'
    )
    parser.add_argument(
        '--no-outliers',
        action='store_true',
        help='Do not include surprising outliers (for testing)'
    )

    args = parser.parse_args()

    # Generate data
    print("Generating synthetic audit data...")
    data = generate_synthetic_audit_data(apply_outliers_flag=not args.no_outliers)

    # Save if requested
    if args.save:
        output_path = Path('.got/synthetic_audit_data.json')
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved to {output_path}")
        print(f"  Files analyzed: {data['files_analyzed']}")
        print(f"  Findings generated: {data['comments_analyzed']}")

        # Show importance metadata summary
        importance_tiers = data['_metadata'].get('importance_tiers', {})
        vlti_files = data['_metadata'].get('vlti_files', [])
        print(f"\n  Full PLN Importance Metadata:")
        print(f"    Priority tiers: {importance_tiers}")
        print(f"    VLTI (pinned) files: {len(vlti_files)}")

    # Print pattern summary
    if args.verbose or not args.save:
        print_pattern_summary(data)

    # Print usage hint
    if args.save:
        print()
        print("Usage in tests:")
        print("  import json")
        print("  with open('.got/synthetic_audit_data.json') as f:")
        print("      audit_data = json.load(f)")
        print()


if __name__ == '__main__':
    main()
