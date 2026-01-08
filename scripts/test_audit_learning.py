#!/usr/bin/env python3
"""
Test Audit Learning Pipeline - Validation Suite for WovenMind + PLN Integration

This script validates the entire audit learning pipeline on controlled synthetic data:

1. DATA GENERATION: Creates synthetic audit findings with embedded patterns
2. DISCOVERY: Runs WovenMind pattern discovery
3. REASONING: Runs PLN inference on discovered patterns
4. VALIDATION: Compares results against ground truth

EXPECTED PATTERNS (Embedded in Synthetic Data):
- Pattern A: "legacy" directory + "TODO:" pattern → frequent co-occurrence
- Pattern B: "api" directory + "should be" pattern → moderate co-occurrence
- Pattern C: high_churn trait + "FIXME:" pattern → high risk signal
- Outliers: Files with unusual combinations (for surprise detection)

VALIDATION CRITERIA:
- Abstraction formation: Did WovenMind discover the embedded patterns?
- Pattern precision: Are the discovered patterns meaningful?
- Risk inference: Did PLN assign higher risk to multi-signal files?
- Surprise detection: Were outlier files flagged?

Usage:
    python scripts/test_audit_learning.py
    python scripts/test_audit_learning.py --verbose
    python scripts/test_audit_learning.py --reset  # Clear state first
    python scripts/test_audit_learning.py --export # Save synthetic data
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
from cortical.reasoning.prism_pln import PLNReasoner, TruthValue

# Import the modules we're testing
sys.path.insert(0, str(Path(__file__).parent))
from woven_audit_discovery import (
    feed_findings_to_mind,
    extract_discoveries,
    MIND_STATE_FILE,
    DISCOVERY_LOG_FILE
)
from audit_reasoning import (
    AuditReasoner,
    load_woven_mind_abstractions,
    RULES_FILE
)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

SYNTHETIC_DATA_FILE = Path(__file__).parent.parent / ".got" / "synthetic_audit_data.json"


def generate_synthetic_audit_data() -> Dict[str, Any]:
    """
    Generate synthetic audit data with known embedded patterns.

    This creates a controlled dataset where we know what patterns should be
    discovered, allowing us to validate the learning pipeline.

    PATTERN DESIGN:
    1. Pattern A (Strong): legacy/ + TODO: (appears 10x)
    2. Pattern B (Moderate): api/ + should be (appears 6x)
    3. Pattern C (Risk): high_churn + FIXME: (appears 8x)
    4. Noise: Random combinations (appears 5x)
    5. Outliers: Unusual multi-signal files (appears 3x)
    """

    findings = []
    git_analysis = {
        "high_churn_files": [],
        "stale_todos": [],
        "critical_modules": []
    }

    # Pattern A: legacy + TODO (strong, frequent pattern)
    for i in range(10):
        findings.append({
            "id": f"legacy/module_{i}.py:{10 + i}",
            "pattern": "TODO:",
            "comment": f"TODO: Refactor this legacy code {i}",
            "age_days": 200 + i * 5,
            "author": "developer_1",
            "stale": True
        })

    # Pattern B: api + should_be (moderate pattern)
    for i in range(6):
        findings.append({
            "id": f"api/handler_{i}.py:{20 + i}",
            "pattern": "should be",
            "comment": f"This should be refactored {i}",
            "age_days": 100 + i * 10,
            "author": "developer_2"
        })

    # Pattern C: high_churn + FIXME (risk pattern)
    high_churn_files = []
    for i in range(8):
        file_path = f"core/processor_{i}.py"
        high_churn_files.append((file_path, 15 + i))
        findings.append({
            "id": f"{file_path}:{30 + i}",
            "pattern": "FIXME:",
            "comment": f"FIXME: This needs immediate attention {i}",
            "age_days": 50 + i * 5,
            "author": "developer_3"
        })

    git_analysis["high_churn_files"] = high_churn_files

    # Noise: Random combinations (shouldn't form strong patterns)
    noise_patterns = [
        ("utils/helper_1.py", "FUTURE:", "FUTURE: Add caching"),
        ("tests/test_1.py", "XXX:", "XXX: Flaky test"),
        ("docs/guide.py", "See:", "See: documentation"),
        ("scripts/tool_1.py", "HACK:", "HACK: Quick workaround"),
        ("config/settings.py", "planned to", "This is planned to be deprecated")
    ]

    for i, (file_path, pattern, comment) in enumerate(noise_patterns):
        findings.append({
            "id": f"{file_path}:{40 + i}",
            "pattern": pattern,
            "comment": comment,
            "age_days": 30 + i * 10,
            "author": f"developer_{i % 3 + 1}"
        })

    # Outliers: Unusual multi-signal combinations (should trigger surprise)
    # These combine multiple patterns in unexpected ways
    outlier_files = [
        "legacy/api_bridge.py",  # Combines legacy + api (unusual)
        "core/legacy_compat.py",  # Combines core + legacy (unusual)
        "api/experimental_core.py"  # Combines api + core (unusual)
    ]

    for i, file_path in enumerate(outlier_files):
        # Multiple patterns in same file (unusual combination)
        findings.append({
            "id": f"{file_path}:{50 + i}",
            "pattern": "TODO:",
            "comment": "TODO: Clean this up",
            "age_days": 300,
            "author": "developer_1",
            "stale": True
        })
        findings.append({
            "id": f"{file_path}:{51 + i}",
            "pattern": "FIXME:",
            "comment": "FIXME: Handle edge case",
            "age_days": 60,
            "author": "developer_2"
        })
        findings.append({
            "id": f"{file_path}:{52 + i}",
            "pattern": "should be",
            "comment": "This should be redesigned",
            "age_days": 120,
            "author": "developer_3"
        })

        # Mark as high churn too
        git_analysis["high_churn_files"].append((file_path, 25))

    return {
        "generated_at": datetime.now().isoformat(),
        "description": "Synthetic audit data with embedded patterns for testing",
        "expected_patterns": {
            "pattern_a": {
                "tokens": ["dir:legacy", "pattern:todo"],
                "frequency": 10,
                "description": "Legacy code with TODOs"
            },
            "pattern_b": {
                "tokens": ["dir:api", "pattern:should_be"],
                "frequency": 6,
                "description": "API handlers that need refactoring"
            },
            "pattern_c": {
                "tokens": ["trait:high_churn", "pattern:fixme"],
                "frequency": 8,
                "description": "High churn files with urgent fixes needed"
            }
        },
        "expected_outliers": outlier_files,
        "total_findings": len(findings),
        "findings": findings,
        "git_analysis": git_analysis
    }


def load_or_generate_synthetic_data(force_regenerate: bool = False) -> Dict[str, Any]:
    """Load existing synthetic data or generate fresh."""
    if not force_regenerate and SYNTHETIC_DATA_FILE.exists():
        try:
            with open(SYNTHETIC_DATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Generate fresh data
    data = generate_synthetic_audit_data()

    # Save it
    SYNTHETIC_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTHETIC_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    return data


# =============================================================================
# WOVEN MIND TESTING
# =============================================================================

def test_woven_mind_discovery(
    findings: List[Dict[str, Any]],
    git_analysis: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test WovenMind pattern discovery on synthetic data.

    Returns:
        Statistics and discovered patterns
    """
    # Create fresh WovenMind instance
    config = WovenMindConfig(
        surprise_threshold=0.3,
        min_frequency=3,  # Need 3 observations for abstraction
        k_winners=10,
        enable_auto_consolidation=True,
    )
    mind = WovenMind(config=config)

    # Feed findings multiple times to strengthen patterns
    # (In real usage, this would be multiple sessions)
    all_stats = {
        "sessions": 3,
        "total_findings_fed": 0,
        "surprising_inputs": [],
    }

    for session in range(3):
        if verbose:
            print(f"  Session {session + 1}/3...")

        stats = feed_findings_to_mind(mind, findings, git_analysis)
        all_stats["total_findings_fed"] += stats["findings_fed"]
        all_stats["surprising_inputs"].extend(stats.get("surprising_inputs", []))

    # Run consolidation to form abstractions
    consolidation = mind.consolidate()

    if verbose:
        print(f"  Consolidation: {consolidation.patterns_transferred} patterns transferred")
        print(f"                {consolidation.abstractions_formed} abstractions formed")

    # Extract discoveries
    discoveries = extract_discoveries(mind)

    return {
        "stats": all_stats,
        "consolidation": {
            "patterns_transferred": consolidation.patterns_transferred,
            "abstractions_formed": consolidation.abstractions_formed,
        },
        "discoveries": discoveries,
        "mind": mind,
    }


# =============================================================================
# PLN REASONING TESTING
# =============================================================================

def test_pln_reasoning(
    findings: List[Dict[str, Any]],
    git_analysis: Dict[str, Any],
    mind: WovenMind,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test PLN reasoning using WovenMind discoveries.

    Returns:
        Risk assessments and inference statistics
    """
    # Save WovenMind state temporarily so audit_reasoning can load it
    temp_mind_state = MIND_STATE_FILE.parent / "test_woven_audit_mind.json"
    state = {
        "mind": mind.to_dict(),
        "metadata": {
            "test_mode": True,
            "generated_at": datetime.now().isoformat(),
        }
    }
    temp_mind_state.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_mind_state, 'w') as f:
        json.dump(state, f)

    # Temporarily swap the state file
    original_state = None
    if MIND_STATE_FILE.exists():
        original_state = MIND_STATE_FILE.read_text()

    try:
        # Copy test state to expected location
        MIND_STATE_FILE.write_text(temp_mind_state.read_text())

        # Create reasoner and load rules
        reasoner = AuditReasoner()
        woven_rules = reasoner.load_rules_from_woven_mind()
        manual_rules = reasoner.load_manual_rules()
        reasoner.add_default_rules()

        # Assert facts from findings
        findings_by_file = defaultdict(list)
        for f in findings:
            finding_id = f.get("id", "")
            if ":" in finding_id:
                file_path = finding_id.rsplit(":", 1)[0]
                findings_by_file[file_path].append(f)

        high_churn = {f for f, _ in git_analysis.get("high_churn_files", [])}

        for file_path, file_findings in findings_by_file.items():
            patterns = [f.get("pattern", "") for f in file_findings]
            traits = []
            if file_path in high_churn:
                traits.append("high_churn")

            parts = file_path.split("/")
            dirs = parts[:-1] if len(parts) > 1 else []

            reasoner.assert_file_facts(file_path, patterns, traits, dirs)

        # Query risk for each file
        risk_assessments = []
        for file_path in findings_by_file.keys():
            risk = reasoner.query_file_risk(file_path)
            if risk:
                overall = max(
                    (r.get("probability", 0) for r in risk.values()),
                    default=0
                )
                risk_assessments.append({
                    "file": file_path,
                    "overall_risk": overall,
                    "details": risk,
                })

        risk_assessments.sort(key=lambda x: -x["overall_risk"])

        return {
            "rules_loaded": reasoner.get_stats()["rules"],
            "woven_rules": woven_rules,
            "manual_rules": manual_rules,
            "files_assessed": len(risk_assessments),
            "risk_assessments": risk_assessments,
        }

    finally:
        # Restore original state
        if original_state:
            MIND_STATE_FILE.write_text(original_state)
        elif MIND_STATE_FILE.exists():
            MIND_STATE_FILE.unlink()

        # Clean up temp file
        if temp_mind_state.exists():
            temp_mind_state.unlink()


# =============================================================================
# VALIDATION & SCORING
# =============================================================================

def validate_pattern_discovery(
    discoveries: Dict[str, Any],
    expected_patterns: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate that WovenMind discovered the expected patterns.

    Returns:
        Validation results with precision/recall metrics
    """
    abstractions = discoveries.get("abstractions", [])

    # Convert expected patterns to token sets for matching
    expected_token_sets = []
    for pattern_id, pattern_info in expected_patterns.items():
        expected_token_sets.append({
            "id": pattern_id,
            "tokens": set(pattern_info["tokens"]),
            "description": pattern_info["description"],
            "min_frequency": pattern_info.get("frequency", 1)
        })

    # Check which expected patterns were discovered
    discovered_expected = []
    discovered_other = []

    for abstraction in abstractions:
        source_nodes = set(abstraction["source_nodes"])

        # Check if this matches any expected pattern
        matched = False
        for expected in expected_token_sets:
            # Check if expected tokens are a subset of discovered nodes
            if expected["tokens"].issubset(source_nodes):
                discovered_expected.append({
                    "expected_id": expected["id"],
                    "abstraction_id": abstraction["id"],
                    "tokens": list(source_nodes),
                    "frequency": abstraction["frequency"],
                    "strength": abstraction["strength"],
                    "interpretation": abstraction.get("interpretation", "")
                })
                matched = True
                break

        if not matched:
            discovered_other.append({
                "abstraction_id": abstraction["id"],
                "tokens": list(source_nodes),
                "frequency": abstraction["frequency"],
                "strength": abstraction["strength"],
            })

    # Calculate metrics
    expected_count = len(expected_patterns)
    discovered_count = len(discovered_expected)

    precision = discovered_count / len(abstractions) if abstractions else 0
    recall = discovered_count / expected_count if expected_count else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "expected_patterns": expected_count,
        "discovered_patterns": len(abstractions),
        "matched_patterns": discovered_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "discovered_expected": discovered_expected,
        "discovered_other": discovered_other,
        "missing_patterns": [
            {"id": pat_id, "description": pat_info["description"]}
            for pat_id, pat_info in expected_patterns.items()
            if not any(d["expected_id"] == pat_id for d in discovered_expected)
        ]
    }


def validate_surprise_detection(
    discovery_stats: Dict[str, Any],
    expected_outliers: List[str],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate that surprise detection flagged expected outlier files.
    """
    surprising_inputs = discovery_stats.get("surprising_inputs", [])
    surprising_files = [s["file"] for s in surprising_inputs]

    # Check overlap with expected outliers
    detected_outliers = [f for f in expected_outliers if f in surprising_files]
    false_positives = [f for f in surprising_files if f not in expected_outliers]
    missed_outliers = [f for f in expected_outliers if f not in surprising_files]

    precision = len(detected_outliers) / len(surprising_files) if surprising_files else 0
    recall = len(detected_outliers) / len(expected_outliers) if expected_outliers else 0

    return {
        "expected_outliers": len(expected_outliers),
        "detected_surprising": len(surprising_files),
        "correctly_detected": len(detected_outliers),
        "false_positives": len(false_positives),
        "missed_outliers": len(missed_outliers),
        "precision": precision,
        "recall": recall,
        "detected_files": detected_outliers,
        "missed_files": missed_outliers,
    }


def validate_risk_inference(
    risk_assessments: List[Dict[str, Any]],
    expected_patterns: Dict[str, Any],
    git_analysis: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate that PLN correctly inferred higher risk for multi-signal files.
    """
    # Files with multiple signals should have higher risk
    high_churn_files = {f for f, _ in git_analysis.get("high_churn_files", [])}

    # Get top risk files
    top_risk_files = [a["file"] for a in risk_assessments[:10]]

    # Check if high-churn + FIXME files are ranked high
    high_risk_expected = [
        f for f in high_churn_files
        if any("FIXME" in str(a.get("file", "")) for a in risk_assessments if a["file"] == f)
    ]

    high_risk_in_top = [f for f in high_risk_expected if f in top_risk_files]

    # Calculate correlation between multi-signal and risk score
    multi_signal_files = []
    for assessment in risk_assessments:
        file_path = assessment["file"]
        signal_count = 0

        if file_path in high_churn_files:
            signal_count += 1

        # Count unique patterns (approximation)
        signal_count += len(assessment.get("details", {}))

        multi_signal_files.append({
            "file": file_path,
            "signals": signal_count,
            "risk": assessment["overall_risk"]
        })

    # Check if risk correlates with signal count
    avg_risk_by_signals = defaultdict(list)
    for item in multi_signal_files:
        avg_risk_by_signals[item["signals"]].append(item["risk"])

    correlation_positive = True
    prev_avg = 0
    for signals in sorted(avg_risk_by_signals.keys()):
        avg_risk = sum(avg_risk_by_signals[signals]) / len(avg_risk_by_signals[signals])
        if avg_risk < prev_avg:
            correlation_positive = False
            break
        prev_avg = avg_risk

    return {
        "total_files_assessed": len(risk_assessments),
        "high_risk_expected": len(high_risk_expected),
        "high_risk_in_top_10": len(high_risk_in_top),
        "correlation_positive": correlation_positive,
        "avg_risk_by_signal_count": {
            signals: sum(risks) / len(risks)
            for signals, risks in avg_risk_by_signals.items()
        }
    }


# =============================================================================
# REPORTING
# =============================================================================

def generate_scorecard(
    pattern_validation: Dict[str, Any],
    surprise_validation: Dict[str, Any],
    risk_validation: Dict[str, Any],
    verbose: bool = False
) -> str:
    """Generate a comprehensive scorecard report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  AUDIT LEARNING PIPELINE - VALIDATION SCORECARD")
    lines.append("=" * 80)
    lines.append("")

    # Overall summary
    lines.append("[OVERALL ASSESSMENT]")

    # Calculate overall score (0-100)
    scores = []

    # Pattern discovery score (40 points)
    pattern_score = pattern_validation["f1_score"] * 40
    scores.append(("Pattern Discovery", pattern_score, 40))

    # Surprise detection score (30 points)
    surprise_score = surprise_validation["recall"] * 30
    scores.append(("Surprise Detection", surprise_score, 30))

    # Risk inference score (30 points)
    risk_score = (
        (risk_validation["high_risk_in_top_10"] / max(risk_validation["high_risk_expected"], 1))
        * 15 +
        (15 if risk_validation["correlation_positive"] else 0)
    )
    scores.append(("Risk Inference", risk_score, 30))

    total_score = sum(s[1] for s in scores)
    total_possible = sum(s[2] for s in scores)

    lines.append(f"  Overall Score: {total_score:.1f} / {total_possible}")
    lines.append(f"  Grade: {get_grade(total_score / total_possible)}")
    lines.append("")

    for name, score, possible in scores:
        percentage = (score / possible * 100) if possible > 0 else 0
        lines.append(f"  • {name}: {score:.1f}/{possible} ({percentage:.0f}%)")

    lines.append("")

    # Pattern Discovery Details
    lines.append("[1. PATTERN DISCOVERY]")
    lines.append(f"  Expected patterns: {pattern_validation['expected_patterns']}")
    lines.append(f"  Discovered patterns: {pattern_validation['discovered_patterns']}")
    lines.append(f"  Matched patterns: {pattern_validation['matched_patterns']}")
    lines.append(f"  Precision: {pattern_validation['precision']:.1%}")
    lines.append(f"  Recall: {pattern_validation['recall']:.1%}")
    lines.append(f"  F1 Score: {pattern_validation['f1_score']:.1%}")

    if pattern_validation['discovered_expected']:
        lines.append("")
        lines.append("  ✓ Correctly Discovered:")
        for d in pattern_validation['discovered_expected']:
            lines.append(f"    • {d['expected_id']}: {d['interpretation']}")
            lines.append(f"      Frequency: {d['frequency']}, Strength: {d['strength']:.2f}")

    if pattern_validation['missing_patterns']:
        lines.append("")
        lines.append("  ✗ Missing Patterns:")
        for m in pattern_validation['missing_patterns']:
            lines.append(f"    • {m['id']}: {m['description']}")

    lines.append("")

    # Surprise Detection Details
    lines.append("[2. SURPRISE DETECTION]")
    lines.append(f"  Expected outliers: {surprise_validation['expected_outliers']}")
    lines.append(f"  Detected surprising files: {surprise_validation['detected_surprising']}")
    lines.append(f"  Correctly detected: {surprise_validation['correctly_detected']}")
    lines.append(f"  Precision: {surprise_validation['precision']:.1%}")
    lines.append(f"  Recall: {surprise_validation['recall']:.1%}")

    if surprise_validation['detected_files']:
        lines.append("")
        lines.append("  ✓ Correctly Detected Outliers:")
        for f in surprise_validation['detected_files']:
            lines.append(f"    • {f}")

    if surprise_validation['missed_files']:
        lines.append("")
        lines.append("  ✗ Missed Outliers:")
        for f in surprise_validation['missed_files']:
            lines.append(f"    • {f}")

    lines.append("")

    # Risk Inference Details
    lines.append("[3. RISK INFERENCE]")
    lines.append(f"  Files assessed: {risk_validation['total_files_assessed']}")
    lines.append(f"  High-risk expected: {risk_validation['high_risk_expected']}")
    lines.append(f"  High-risk in top 10: {risk_validation['high_risk_in_top_10']}")
    lines.append(f"  Risk/signal correlation: {'✓ POSITIVE' if risk_validation['correlation_positive'] else '✗ NEGATIVE'}")

    if risk_validation['avg_risk_by_signal_count']:
        lines.append("")
        lines.append("  Average Risk by Signal Count:")
        for signals, avg_risk in sorted(risk_validation['avg_risk_by_signal_count'].items()):
            lines.append(f"    {signals} signals → {avg_risk:.1%} risk")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def get_grade(percentage: float) -> str:
    """Convert percentage to letter grade."""
    if percentage >= 0.9:
        return "A (Excellent)"
    elif percentage >= 0.8:
        return "B (Good)"
    elif percentage >= 0.7:
        return "C (Acceptable)"
    elif percentage >= 0.6:
        return "D (Needs Work)"
    else:
        return "F (Failing)"


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test Audit Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script validates the entire audit learning pipeline on controlled data.
It tests WovenMind pattern discovery + PLN reasoning integration.
        """
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all state before running")
    parser.add_argument("--export", action="store_true",
                        help="Export synthetic data and exit")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate synthetic data")

    args = parser.parse_args()

    # Handle reset
    if args.reset:
        print("Clearing state files...")
        for f in [MIND_STATE_FILE, DISCOVERY_LOG_FILE, RULES_FILE, SYNTHETIC_DATA_FILE]:
            if f.exists():
                f.unlink()
                print(f"  Cleared: {f}")
        print("State reset complete.")
        if not args.export:
            return

    # Load or generate synthetic data
    print("Loading synthetic audit data...")
    synthetic_data = load_or_generate_synthetic_data(force_regenerate=args.regenerate)

    print(f"  Total findings: {synthetic_data['total_findings']}")
    print(f"  Expected patterns: {len(synthetic_data['expected_patterns'])}")
    print(f"  Expected outliers: {len(synthetic_data['expected_outliers'])}")

    if args.export:
        print(f"\nSynthetic data saved to: {SYNTHETIC_DATA_FILE}")
        return

    print()

    # Test 1: WovenMind Discovery
    print("[TEST 1/3] Testing WovenMind Pattern Discovery...")
    print("-" * 60)

    woven_results = test_woven_mind_discovery(
        synthetic_data["findings"],
        synthetic_data["git_analysis"],
        verbose=args.verbose
    )

    print(f"  ✓ Fed {woven_results['stats']['total_findings_fed']} findings across {woven_results['stats']['sessions']} sessions")
    print(f"  ✓ Formed {len(woven_results['discoveries']['abstractions'])} abstractions")
    print(f"  ✓ Detected {len(woven_results['stats']['surprising_inputs'])} surprising files")
    print()

    # Test 2: PLN Reasoning
    print("[TEST 2/3] Testing PLN Reasoning...")
    print("-" * 60)

    pln_results = test_pln_reasoning(
        synthetic_data["findings"],
        synthetic_data["git_analysis"],
        woven_results["mind"],
        verbose=args.verbose
    )

    print(f"  ✓ Loaded {pln_results['rules_loaded']} total rules")
    print(f"    • From WovenMind: {pln_results['woven_rules']}")
    print(f"    • Manual: {pln_results['manual_rules']}")
    print(f"    • Default: {pln_results['rules_loaded'] - pln_results['woven_rules'] - pln_results['manual_rules']}")
    print(f"  ✓ Assessed risk for {pln_results['files_assessed']} files")
    print()

    # Test 3: Validation
    print("[TEST 3/3] Validating Results...")
    print("-" * 60)

    pattern_validation = validate_pattern_discovery(
        woven_results["discoveries"],
        synthetic_data["expected_patterns"],
        verbose=args.verbose
    )

    surprise_validation = validate_surprise_detection(
        woven_results["stats"],
        synthetic_data["expected_outliers"],
        verbose=args.verbose
    )

    risk_validation = validate_risk_inference(
        pln_results["risk_assessments"],
        synthetic_data["expected_patterns"],
        synthetic_data["git_analysis"],
        verbose=args.verbose
    )

    print(f"  ✓ Pattern discovery validated: {pattern_validation['f1_score']:.1%} F1 score")
    print(f"  ✓ Surprise detection validated: {surprise_validation['recall']:.1%} recall")
    print(f"  ✓ Risk inference validated")
    print()

    # Generate scorecard
    print()
    scorecard = generate_scorecard(
        pattern_validation,
        surprise_validation,
        risk_validation,
        verbose=args.verbose
    )
    print(scorecard)


if __name__ == "__main__":
    main()
