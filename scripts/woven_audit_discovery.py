#!/usr/bin/env python3
"""
Woven Audit Discovery - Pattern Discovery for Codebase Audits using WovenMind

STATUS: Experimental Research Tool
==================================

This script uses the WovenMind dual-process cognitive architecture as a
pattern discovery layer for codebase audits. Unlike traditional analysis
that looks for predefined patterns, WovenMind discovers emergent patterns
through:

1. HEBBIAN LEARNING (Hive): Learns what audit findings tend to co-occur
2. ABSTRACTION FORMATION (Cortex): Creates higher-level concepts when
   patterns repeat (e.g., "files with X and Y together" becomes a concept)
3. SURPRISE DETECTION: Flags unusual combinations that deviate from learned
   patterns - these are potentially interesting anomalies to investigate

HONEST LIMITATIONS:
- This is EXPERIMENTAL - treat outputs as "hints to investigate" not truth
- Pattern discovery depends on data volume (needs many findings to learn)
- Abstractions are statistical, not semantic (co-occurrence, not causation)
- Single-machine, in-memory processing
- Early runs will show few patterns; it learns over time

The goal is to discover patterns we didn't think to look for.

Usage:
    python scripts/woven_audit_discovery.py [directory]
    python scripts/woven_audit_discovery.py --with-git cortical/
    python scripts/woven_audit_discovery.py --show-mind    # Show learned state
    python scripts/woven_audit_discovery.py --reset-mind   # Clear learned state
    python scripts/woven_audit_discovery.py --consolidate  # Run learning cycle
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig, WovenMindResult
from cortical.reasoning.loom import ThinkingMode

# Import from our codebase health analyzer
from scripts.codebase_health import analyze_directory

# =============================================================================
# STATE PERSISTENCE - WovenMind learns across sessions
# =============================================================================

MIND_STATE_FILE = Path(__file__).parent.parent / ".got" / "woven_audit_mind.json"
DISCOVERY_LOG_FILE = Path(__file__).parent.parent / ".got" / "woven_discoveries.json"


def load_mind_state() -> Optional[Dict[str, Any]]:
    """Load persisted WovenMind state."""
    if MIND_STATE_FILE.exists():
        try:
            with open(MIND_STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_mind_state(mind: WovenMind, metadata: Dict[str, Any]) -> None:
    """Save WovenMind state to disk."""
    MIND_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "mind": mind.to_dict(),
        "metadata": {
            **metadata,
            "saved_at": datetime.now().isoformat(),
        }
    }
    with open(MIND_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_discovery_log() -> Dict[str, Any]:
    """Load discovery log."""
    if DISCOVERY_LOG_FILE.exists():
        try:
            with open(DISCOVERY_LOG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "created": datetime.now().isoformat(),
        "sessions": 0,
        "discoveries": [],
        "surprising_files": [],
    }


def save_discovery_log(log: Dict[str, Any]) -> None:
    """Save discovery log."""
    DISCOVERY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log["updated_at"] = datetime.now().isoformat()
    with open(DISCOVERY_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


# =============================================================================
# FINDING TOKENIZATION - Convert audit findings to WovenMind tokens
# =============================================================================

def tokenize_finding(finding: Dict[str, Any], file_info: Dict[str, Any] = None) -> List[str]:
    """
    Convert an audit finding into tokens for WovenMind.

    We create semantic tokens that capture different aspects:
    - Pattern type (e.g., "pattern:should_be", "pattern:TODO")
    - File path components (e.g., "dir:got", "file:api.py")
    - File characteristics (e.g., "trait:high_churn", "trait:bug_prone")

    This lets WovenMind learn associations like:
    "files in got/ with TODO patterns tend to have high churn"
    """
    tokens = []

    # Pattern token - normalize by replacing spaces with underscores and stripping punctuation
    pattern = finding.get('pattern', 'unknown')
    normalized_pattern = pattern.replace(' ', '_').lower().rstrip(':')
    tokens.append(f"pattern:{normalized_pattern}")

    # Extract file info from finding ID (format: "path/file.py:line")
    finding_id = finding.get('id', '')
    if ':' in finding_id:
        file_path = finding_id.rsplit(':', 1)[0]

        # Directory components
        parts = file_path.split('/')
        if len(parts) > 1:
            for part in parts[:-1]:
                tokens.append(f"dir:{part}")

        # File name
        if parts:
            tokens.append(f"file:{parts[-1]}")

    # Add file characteristics if available
    if file_info:
        if file_info.get('high_churn'):
            tokens.append("trait:high_churn")
        if file_info.get('bug_prone'):
            tokens.append("trait:bug_prone")
        if file_info.get('stale'):
            tokens.append("trait:stale_code")

    return tokens


def tokenize_file_context(file_path: str, git_analysis: Dict[str, Any]) -> List[str]:
    """Create tokens representing a file's characteristics."""
    tokens = [f"file:{Path(file_path).name}"]

    # Directory context
    parts = file_path.split('/')
    for part in parts[:-1]:
        tokens.append(f"dir:{part}")

    # Check if high churn
    high_churn_files = {f for f, _ in git_analysis.get('high_churn_files', [])}
    if file_path in high_churn_files:
        tokens.append("trait:high_churn")

    return tokens


# =============================================================================
# NOVELTY DETECTION - Abstraction-based outlier detection
# =============================================================================

def compute_novelty_score(
    file_tokens: List[str],
    abstractions: List[Any],
    all_file_tokens: Dict[str, List[str]],
) -> float:
    """
    Compute novelty score for a file based on abstraction coverage.

    A file is novel/surprising if:
    1. Its tokens don't match well with any learned abstraction
    2. It contains unusual token combinations not seen in other files

    Args:
        file_tokens: Tokens for this file
        abstractions: List of formed abstractions from WovenMind
        all_file_tokens: Dict of file_path -> tokens for computing rarity

    Returns:
        Novelty score in [0, 1] where 1 = highly novel/surprising
    """
    if not file_tokens:
        return 0.0

    file_token_set = set(file_tokens)

    # Score 1: Abstraction coverage (how well do tokens match abstractions?)
    # Lower coverage = more novel
    abstraction_coverage = 0.0
    if abstractions:
        best_match = 0.0
        for abstraction in abstractions:
            source_nodes = set(abstraction.source_nodes)
            # Jaccard similarity
            intersection = len(file_token_set & source_nodes)
            union = len(file_token_set | source_nodes)
            if union > 0:
                similarity = intersection / union
                best_match = max(best_match, similarity)
        abstraction_coverage = best_match

    # Score 2: Token combination rarity
    # If this combination of tokens is rare across files, it's novel
    combination_rarity = 1.0
    if all_file_tokens:
        # Count how many files have similar token sets
        similar_files = 0
        for other_path, other_tokens in all_file_tokens.items():
            other_set = set(other_tokens)
            # Check if significant overlap
            if file_token_set and other_set:
                overlap = len(file_token_set & other_set) / len(file_token_set)
                if overlap > 0.5:  # >50% overlap = similar
                    similar_files += 1
        # Rarity = 1 - (proportion of similar files)
        combination_rarity = 1.0 - (similar_files / max(len(all_file_tokens), 1))

    # Score 3: Pattern diversity (key indicator of outliers)
    # Files with many different PATTERN types are unusual - normal files have 1-2 patterns
    pattern_tokens = [t for t in file_tokens if t.startswith('pattern:')]
    pattern_count = len(set(pattern_tokens))
    # 1 pattern = normal (0.0), 2 patterns = slightly unusual (0.33), 3+ patterns = unusual (0.67+)
    pattern_diversity = min((pattern_count - 1) / 3.0, 1.0) if pattern_count > 0 else 0.0

    # Score 4: Cross-domain signal (has traits like high_churn AND multiple patterns)
    has_trait = any(t.startswith('trait:') for t in file_tokens)
    cross_domain = 1.0 if (has_trait and pattern_count >= 2) else 0.0

    # Combine scores: low abstraction coverage + high rarity + pattern diversity = novel
    novelty = (
        (1.0 - abstraction_coverage) * 0.25 +  # Low abstraction match = novel
        combination_rarity * 0.25 +             # Rare combination = novel
        pattern_diversity * 0.35 +              # Multiple patterns = novel (increased weight)
        cross_domain * 0.15                     # Cross-domain signal = novel
    )

    return min(1.0, max(0.0, novelty))


# =============================================================================
# PATTERN FEEDING - Train WovenMind on audit data
# =============================================================================

def feed_findings_to_mind(
    mind: WovenMind,
    findings: List[Dict[str, Any]],
    git_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Feed audit findings to WovenMind for pattern learning.

    We feed findings at multiple granularities:
    1. Full file patterns (specific but rarely repeat)
    2. Generic patterns without file names (can form abstractions)
    3. Individual pattern types (for tracking frequency)

    Returns statistics about what was learned.
    """
    stats = {
        "findings_fed": 0,
        "patterns_observed": 0,
        "surprising_inputs": [],
        "mode_switches": 0,
    }

    # Track all file tokens for novelty detection
    all_file_tokens: Dict[str, List[str]] = {}

    # Group findings by file for richer context
    findings_by_file = defaultdict(list)
    for f in findings:
        finding_id = f.get('id', '')
        if ':' in finding_id:
            file_path = finding_id.rsplit(':', 1)[0]
            findings_by_file[file_path].append(f)

    # Prepare file characteristics
    high_churn = {f for f, _ in git_analysis.get('high_churn_files', [])}

    # Feed each file's findings as a combined pattern
    for file_path, file_findings in findings_by_file.items():
        # Build tokens for this file's pattern (specific)
        specific_tokens = []
        generic_tokens = []  # Without file name - for repeatable patterns

        # File context
        parts = file_path.split('/')
        for part in parts[:-1]:
            specific_tokens.append(f"dir:{part}")
            generic_tokens.append(f"dir:{part}")
        if parts:
            specific_tokens.append(f"file:{parts[-1]}")

        # File traits
        if file_path in high_churn:
            specific_tokens.append("trait:high_churn")
            generic_tokens.append("trait:high_churn")

        # All pattern types in this file
        for finding in file_findings:
            pattern = finding.get('pattern', 'unknown')
            normalized = pattern.replace(' ', '_').lower().rstrip(':')
            pattern_token = f"pattern:{normalized}"
            specific_tokens.append(pattern_token)
            generic_tokens.append(pattern_token)

        # Store tokens for novelty detection later
        all_file_tokens[file_path] = generic_tokens.copy()

        # Train Hive on the text representation
        text = " ".join(specific_tokens)
        mind.train(text)

        # Process through full system
        mind.process(specific_tokens)
        stats["findings_fed"] += len(file_findings)

        # Observe GENERIC pattern for Cortex abstraction (can repeat!)
        # This allows patterns like "dir:got + pattern:should_be" to form abstractions
        if len(generic_tokens) >= 2:
            mind.observe_pattern(generic_tokens)
            stats["patterns_observed"] += 1

        # Also observe individual pattern+directory combos
        # These are more likely to repeat across files
        for finding in file_findings:
            pattern = finding.get('pattern', 'unknown')
            normalized = pattern.replace(' ', '_').lower().rstrip(':')
            pattern_token = f"pattern:{normalized}"

            # Pattern + top-level directory
            if parts and len(parts) > 1:
                combo = [f"dir:{parts[0]}", pattern_token]
                mind.observe_pattern(combo)

            # Pattern + trait (if high churn)
            if file_path in high_churn:
                combo = [pattern_token, "trait:high_churn"]
                mind.observe_pattern(combo)

    # Run consolidation to form abstractions
    mind.consolidate()

    # Now detect surprising files using abstraction-based novelty
    # (WovenMind's built-in surprise is for sequential prediction, not outliers)
    abstractions = mind.cortex.get_abstractions()

    novelty_threshold = 0.6  # Files with novelty > 0.6 are "surprising"

    for file_path, tokens in all_file_tokens.items():
        novelty = compute_novelty_score(tokens, abstractions, all_file_tokens)

        if novelty > novelty_threshold:
            stats["surprising_inputs"].append({
                "file": file_path,
                "tokens": tokens,
                "surprise_magnitude": novelty,
                "mode": "NOVELTY",
            })

    return stats


# =============================================================================
# DISCOVERY EXTRACTION - What did WovenMind learn?
# =============================================================================

def extract_discoveries(mind: WovenMind) -> Dict[str, Any]:
    """
    Extract what WovenMind has discovered.

    Returns:
        Dictionary with discovered patterns, abstractions, and insights.
    """
    discoveries = {
        "abstractions": [],
        "hive_stats": {},
        "cortex_stats": {},
        "consolidation_stats": {},
    }

    # Get formed abstractions
    abstractions = mind.cortex.get_abstractions()
    for abstraction in abstractions:
        # Parse the source nodes to understand what was abstracted
        source_patterns = list(abstraction.source_nodes)

        discoveries["abstractions"].append({
            "id": abstraction.id,
            "level": abstraction.level,
            "frequency": abstraction.frequency,
            "strength": abstraction.strength,
            "source_nodes": source_patterns,
            "interpretation": interpret_abstraction(source_patterns),
        })

    # Sort by strength
    discoveries["abstractions"].sort(key=lambda x: -x["strength"])

    # Get system stats
    full_stats = mind.get_stats()
    discoveries["hive_stats"] = full_stats.get("hive", {})
    discoveries["cortex_stats"] = full_stats.get("cortex", {})
    discoveries["consolidation_stats"] = full_stats.get("consolidation", {})

    return discoveries


def interpret_abstraction(source_nodes: List[str]) -> str:
    """
    Attempt to interpret what an abstraction means in human terms.

    This is best-effort interpretation of pattern clusters.
    """
    dirs = [n.split(':')[1] for n in source_nodes if n.startswith('dir:')]
    files = [n.split(':')[1] for n in source_nodes if n.startswith('file:')]
    patterns = [n.split(':')[1] for n in source_nodes if n.startswith('pattern:')]
    traits = [n.split(':')[1] for n in source_nodes if n.startswith('trait:')]

    parts = []

    if dirs:
        parts.append(f"in {'/'.join(dirs)}")
    if files:
        parts.append(f"files like {', '.join(files[:2])}")
    if patterns:
        parts.append(f"with patterns [{', '.join(patterns)}]")
    if traits:
        parts.append(f"having traits [{', '.join(traits)}]")

    if not parts:
        return "Unknown pattern cluster"

    return " ".join(parts)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_discovery_report(
    findings_stats: Dict[str, Any],
    discoveries: Dict[str, Any],
    mind: WovenMind,
    session_num: int,
) -> str:
    """Generate a human-readable discovery report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  WOVEN AUDIT DISCOVERY - Pattern Discovery Report")
    lines.append("=" * 70)

    # Session info
    lines.append(f"\n[Session Info]")
    lines.append(f"  Session number: {session_num}")
    lines.append(f"  Findings processed: {findings_stats['findings_fed']}")
    lines.append(f"  Patterns observed: {findings_stats['patterns_observed']}")
    lines.append(f"  Current mode: {mind.get_current_mode().name}")

    # Surprising inputs
    surprising = findings_stats.get('surprising_inputs', [])
    if surprising:
        lines.append(f"\n[Surprising Files - Deviate from Learned Patterns]")
        lines.append(f"  Found {len(surprising)} files with unusual combinations:")
        for s in sorted(surprising, key=lambda x: -x['surprise_magnitude'])[:10]:
            lines.append(f"    ! {s['file']}")
            lines.append(f"      Surprise: {s['surprise_magnitude']:.2f}, Mode: {s['mode']}")
            # Show what made it surprising
            patterns = [t for t in s['tokens'] if t.startswith('pattern:')]
            if patterns:
                lines.append(f"      Patterns: {', '.join(patterns)}")
    else:
        lines.append(f"\n[Surprising Files]")
        lines.append(f"  None detected yet (need more data to establish baseline)")

    # Discovered abstractions
    abstractions = discoveries.get('abstractions', [])
    if abstractions:
        lines.append(f"\n[Discovered Abstractions - Emergent Patterns]")
        lines.append(f"  WovenMind has formed {len(abstractions)} abstractions:")
        for a in abstractions[:10]:
            lines.append(f"\n  [{a['id']}] (strength: {a['strength']:.2f}, seen {a['frequency']}x)")
            lines.append(f"    → {a['interpretation']}")
            lines.append(f"    Raw: {a['source_nodes'][:5]}{'...' if len(a['source_nodes']) > 5 else ''}")
    else:
        lines.append(f"\n[Discovered Abstractions]")
        lines.append(f"  None yet - need more data (min 3 observations per pattern)")
        lines.append(f"  Run multiple audit sessions to build up patterns")

    # System stats
    cortex = discoveries.get('cortex_stats', {})
    consolidation = discoveries.get('consolidation_stats', {})

    lines.append(f"\n[Learning Stats]")
    lines.append(f"  Total observations: {cortex.get('total_observations', 0)}")
    lines.append(f"  Unique patterns seen: {cortex.get('unique_patterns', 0)}")
    lines.append(f"  Patterns pending transfer: {consolidation.get('patterns_pending_transfer', 0)}")
    lines.append(f"  Consolidation cycles run: {consolidation.get('total_consolidations', 0)}")

    # Guidance
    lines.append(f"\n[How to Use These Discoveries]")
    lines.append(f"  • Surprising files: Investigate manually - unusual pattern combinations")
    lines.append(f"  • Abstractions: These are patterns that repeat - may indicate systemic issues")
    lines.append(f"  • Run --consolidate periodically to strengthen learned patterns")
    lines.append(f"  • Data accumulates across sessions - run regularly for better results")

    lines.append(f"\n[Data Files]")
    lines.append(f"  Mind state: {MIND_STATE_FILE}")
    lines.append(f"  Discovery log: {DISCOVERY_LOG_FILE}")

    lines.append("\n" + "=" * 70)
    lines.append("  This is EXPERIMENTAL - treat as hints to investigate, not ground truth")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Woven Audit Discovery - Pattern Discovery using WovenMind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses WovenMind's dual-process cognitive architecture to
discover emergent patterns in codebase audit findings.

Unlike traditional analysis that looks for predefined patterns,
WovenMind learns what patterns tend to co-occur and flags unusual
combinations as potentially interesting anomalies.

Run it multiple times to build up learned patterns.
        """
    )
    parser.add_argument("directory", nargs="?", default="cortical/",
                        help="Directory to analyze")
    parser.add_argument("--with-git", action="store_true",
                        help="Include git history analysis")
    parser.add_argument("--show-mind", action="store_true",
                        help="Show current WovenMind state and exit")
    parser.add_argument("--reset-mind", action="store_true",
                        help="Clear learned state and start fresh")
    parser.add_argument("--consolidate", action="store_true",
                        help="Run consolidation cycle (memory consolidation)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    # Handle state management commands
    if args.reset_mind:
        for f in [MIND_STATE_FILE, DISCOVERY_LOG_FILE]:
            if f.exists():
                f.unlink()
                print(f"Cleared: {f}")
        print("WovenMind state reset. Fresh start on next run.")
        return

    # Load or create WovenMind
    state = load_mind_state()
    if state:
        print(f"Loading existing WovenMind (trained on {state['metadata'].get('total_findings', 0)} findings)")
        mind = WovenMind.from_dict(state["mind"])
        session_num = state["metadata"].get("sessions", 0) + 1
    else:
        print("Creating new WovenMind instance")
        config = WovenMindConfig(
            surprise_threshold=0.3,  # Trigger SLOW mode at 30% surprise
            min_frequency=3,         # Need 3 observations for abstraction
            k_winners=10,            # Top 10 activations in lateral inhibition
            enable_auto_consolidation=True,
        )
        mind = WovenMind(config=config)
        session_num = 1

    # Handle show-mind
    if args.show_mind:
        print("\n" + "=" * 70)
        print("  WOVEN MIND STATE")
        print("=" * 70)

        stats = mind.get_stats()
        print(f"\nMode: {stats['mode']}")
        print(f"Surprise baseline: {stats['loom']['surprise_baseline']:.3f}")
        print(f"Mode transitions: {stats['loom']['transition_count']}")

        cortex = stats['cortex']
        print(f"\nCortex:")
        print(f"  Observations: {cortex.get('total_observations', 0)}")
        print(f"  Unique patterns: {cortex.get('unique_patterns', 0)}")
        print(f"  Abstractions: {cortex.get('total_abstractions', 0)}")

        abstractions = mind.cortex.get_abstractions()
        if abstractions:
            print(f"\nFormed Abstractions:")
            for a in sorted(abstractions, key=lambda x: -x.strength)[:10]:
                print(f"  [{a.id}] strength={a.strength:.2f}, freq={a.frequency}")
                print(f"    Nodes: {list(a.source_nodes)[:5]}...")

        return

    # Handle consolidation
    if args.consolidate:
        print("\nRunning consolidation cycle (memory consolidation)...")
        result = mind.consolidate()
        print(f"  Patterns transferred: {result.patterns_transferred}")
        print(f"  Abstractions formed: {result.abstractions_formed}")
        print(f"  Cycle duration: {result.cycle_duration_ms:.1f}ms")

        # Save updated state
        save_mind_state(mind, {
            "sessions": session_num,
            "total_findings": state["metadata"].get("total_findings", 0) if state else 0,
            "last_consolidation": datetime.now().isoformat(),
        })
        print(f"\nState saved to {MIND_STATE_FILE}")
        return

    # Normal analysis run
    print("=" * 70)
    print("  Woven Audit Discovery - Pattern Discovery Analysis")
    print("=" * 70)
    print()

    # Step 1: Run codebase health analysis
    print("[1/4] Running codebase health analysis...")
    results = analyze_directory(args.directory, with_git=args.with_git)

    if not results:
        print("No results from analysis")
        return

    findings = results.get('findings', [])
    git_analysis = results.get('git_analysis', {})
    print(f"      Found {len(findings)} audit findings")

    # Step 2: Feed findings to WovenMind
    print(f"\n[2/4] Feeding findings to WovenMind...")
    findings_stats = feed_findings_to_mind(mind, findings, git_analysis)
    print(f"      Processed {findings_stats['findings_fed']} findings")
    print(f"      Observed {findings_stats['patterns_observed']} file patterns")
    if findings_stats['surprising_inputs']:
        print(f"      Detected {len(findings_stats['surprising_inputs'])} surprising files")

    # Step 3: Extract discoveries
    print(f"\n[3/4] Extracting discovered patterns...")
    discoveries = extract_discoveries(mind)
    print(f"      Found {len(discoveries['abstractions'])} abstractions")

    # Step 4: Save state
    print(f"\n[4/4] Saving WovenMind state...")
    total_findings = (state["metadata"].get("total_findings", 0) if state else 0) + len(findings)
    save_mind_state(mind, {
        "sessions": session_num,
        "total_findings": total_findings,
    })

    # Update discovery log
    discovery_log = load_discovery_log()
    discovery_log["sessions"] = session_num
    if findings_stats['surprising_inputs']:
        discovery_log["surprising_files"].extend([
            s["file"] for s in findings_stats['surprising_inputs']
        ])
        # Keep unique
        discovery_log["surprising_files"] = list(set(discovery_log["surprising_files"]))
    save_discovery_log(discovery_log)

    print(f"      State saved to {MIND_STATE_FILE}")

    # Generate report
    print()
    report = generate_discovery_report(findings_stats, discoveries, mind, session_num)
    print(report)


if __name__ == "__main__":
    main()
