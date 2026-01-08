#!/usr/bin/env python3
"""
Audit Reasoning - WovenMind Discovery → PLN Reasoning Pipeline

This tool bridges pattern discovery (WovenMind) with probabilistic reasoning
(PLN) to create a practical audit analysis pipeline:

1. DISCOVER: WovenMind finds patterns in audit data (unsupervised)
2. VALIDATE: Human reviews discovered patterns
3. ENCODE: Convert validated patterns to PLN rules
4. REASON: PLN infers conclusions about files with uncertainty

This is NOT theater - it's a real ML pipeline:
- Feature discovery (WovenMind abstractions)
- Feature selection (human validation)
- Model building (PLN rule encoding)
- Inference (PLN reasoning with uncertainty propagation)

Usage:
    # Discover patterns first
    python scripts/woven_audit_discovery.py cortical/ --with-git

    # Then run reasoning (uses discovered patterns as rules)
    python scripts/audit_reasoning.py cortical/ --with-git

    # Show what rules are active
    python scripts/audit_reasoning.py --show-rules

    # Add manual rule
    python scripts/audit_reasoning.py --add-rule "high_churn(X)" "risky(X)" 0.7
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.prism_pln import PLNReasoner, TruthValue
from cortical.reasoning.woven_mind import WovenMind

# Import from our other audit tools
from scripts.codebase_health import analyze_directory

# =============================================================================
# STATE FILES
# =============================================================================

RULES_FILE = Path(__file__).parent.parent / ".got" / "audit_pln_rules.json"
WOVEN_MIND_FILE = Path(__file__).parent.parent / ".got" / "woven_audit_mind.json"


def load_rules() -> Dict[str, Any]:
    """Load PLN rules from disk."""
    if RULES_FILE.exists():
        try:
            with open(RULES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "version": 1,
        "created": datetime.now().isoformat(),
        "rules": [],
        "manual_rules": [],  # Human-added rules
        "derived_rules": [],  # Rules derived from WovenMind
    }


def save_rules(rules: Dict[str, Any]) -> None:
    """Save PLN rules to disk."""
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    rules["updated"] = datetime.now().isoformat()
    with open(RULES_FILE, 'w') as f:
        json.dump(rules, f, indent=2)


# =============================================================================
# PATTERN → RULE CONVERSION
# =============================================================================

def abstraction_to_rule(abstraction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a WovenMind abstraction to a PLN rule.

    Abstractions like:
        {source_nodes: ["dir:reasoning", "pattern:should_be"], frequency: 8}

    Become rules like:
        in_dir(X, reasoning) ∧ has_pattern(X, should_be) → flagged(X) [strength=0.6]

    The strength is derived from frequency (more observations = higher confidence).
    """
    nodes = abstraction.get("source_nodes", [])
    frequency = abstraction.get("frequency", 0)
    strength = abstraction.get("strength", 0.5)

    if len(nodes) < 2:
        return None

    # Parse nodes into antecedent atoms
    antecedent_parts = []
    for node in nodes:
        if ":" in node:
            prefix, value = node.split(":", 1)
            if prefix == "dir":
                antecedent_parts.append(f"in_dir(X, {value})")
            elif prefix == "pattern":
                antecedent_parts.append(f"has_pattern(X, {value})")
            elif prefix == "trait":
                antecedent_parts.append(f"has_trait(X, {value})")
            elif prefix == "file":
                # Skip file-specific nodes (too specific for rules)
                continue

    if len(antecedent_parts) < 2:
        return None

    # Combine antecedents
    antecedent = " ∧ ".join(antecedent_parts)

    # Map frequency to confidence (logarithmic scale)
    import math
    confidence = min(0.9, 0.3 + 0.1 * math.log(frequency + 1))

    # Consequent depends on what we're detecting
    # If it has pattern nodes, it's about code quality
    consequent = "flagged(X)"

    return {
        "antecedent": antecedent,
        "consequent": consequent,
        "strength": min(0.8, strength + 0.2),  # Boost a bit for validated patterns
        "confidence": confidence,
        "source": abstraction.get("id", "unknown"),
        "frequency": frequency,
        "interpretation": abstraction.get("interpretation", ""),
    }


def load_woven_mind_abstractions() -> List[Dict[str, Any]]:
    """Load abstractions from WovenMind state file."""
    if not WOVEN_MIND_FILE.exists():
        return []

    try:
        with open(WOVEN_MIND_FILE, 'r') as f:
            state = json.load(f)

        mind_state = state.get("mind", {})
        cortex_state = mind_state.get("cortex_state", {})
        engine_state = cortex_state.get("engine_state", {})
        abstractions_data = engine_state.get("abstractions", {})

        abstractions = []
        for abs_id, abs_data in abstractions_data.items():
            abstractions.append({
                "id": abs_id,
                "source_nodes": list(abs_data.get("source_nodes", [])),
                "level": abs_data.get("level", 0),
                "frequency": abs_data.get("frequency", 0),
                "strength": abs_data.get("strength", 0.5),
            })

        return sorted(abstractions, key=lambda x: -x["strength"])
    except (json.JSONDecodeError, IOError, KeyError) as e:
        print(f"Warning: Could not load WovenMind state: {e}")
        return []


# =============================================================================
# PLN REASONING ENGINE
# =============================================================================

class AuditReasoner:
    """
    PLN-based reasoning for audit findings.

    Combines:
    - Facts from current audit (file has X, Y, Z)
    - Rules from WovenMind patterns and manual rules
    - Inference to determine risk levels
    """

    def __init__(self):
        self.pln = PLNReasoner()
        self.rules_config = load_rules()

    def load_rules_from_woven_mind(self) -> int:
        """Load and convert WovenMind abstractions to PLN rules."""
        abstractions = load_woven_mind_abstractions()
        count = 0

        for abstraction in abstractions:
            rule = abstraction_to_rule(abstraction)
            if rule:
                # For now, simplify to single antecedent rules
                # (PLN doesn't support conjunction in antecedent directly)
                for node in abstraction.get("source_nodes", []):
                    if ":" in node:
                        prefix, value = node.split(":", 1)
                        if prefix in ("dir", "pattern", "trait"):
                            atom = f"has_{prefix}(X, {value})"
                            self.pln.assert_rule(
                                atom,
                                "flagged(X)",
                                strength=rule["strength"] * 0.8,
                                confidence=rule["confidence"]
                            )
                            count += 1

        return count

    def load_manual_rules(self) -> int:
        """Load manually defined rules."""
        count = 0
        for rule in self.rules_config.get("manual_rules", []):
            self.pln.assert_rule(
                rule["antecedent"],
                rule["consequent"],
                strength=rule.get("strength", 0.7),
                confidence=rule.get("confidence", 0.8)
            )
            count += 1
        return count

    def add_default_rules(self) -> None:
        """Add sensible default rules for code audit."""
        defaults = [
            # High churn suggests instability
            ("has_trait(X, high_churn)", "needs_review(X)", 0.7),
            # TODO comments suggest incomplete work
            ("has_pattern(X, todo)", "incomplete(X)", 0.6),
            # "should be" suggests known issues
            ("has_pattern(X, should_be)", "has_known_issue(X)", 0.5),
            # Future markers suggest deferred work
            ("has_pattern(X, future)", "deferred_work(X)", 0.6),
            # Combine signals for risk
            ("has_known_issue(X)", "needs_review(X)", 0.6),
            ("incomplete(X)", "needs_review(X)", 0.5),
        ]

        for ant, cons, strength in defaults:
            self.pln.assert_rule(ant, cons, strength=strength, confidence=0.7)

    def assert_file_facts(
        self,
        file_path: str,
        patterns: List[str],
        traits: List[str],
        directories: List[str]
    ) -> None:
        """Assert facts about a file."""
        # Normalize file identifier
        file_id = Path(file_path).name.replace(".", "_")

        for pattern in patterns:
            pattern_clean = pattern.replace(" ", "_").lower()
            self.pln.assert_fact(
                f"has_pattern({file_id}, {pattern_clean})",
                strength=0.95,
                confidence=0.9
            )

        for trait in traits:
            self.pln.assert_fact(
                f"has_trait({file_id}, {trait})",
                strength=0.9,
                confidence=0.85
            )

        for dir_name in directories:
            self.pln.assert_fact(
                f"has_dir({file_id}, {dir_name})",
                strength=1.0,
                confidence=1.0
            )

    def query_file_risk(self, file_path: str) -> Dict[str, Any]:
        """Query the risk assessment for a file."""
        file_id = Path(file_path).name.replace(".", "_")

        results = {}

        # Query different risk aspects
        queries = [
            ("needs_review", f"needs_review({file_id})"),
            ("has_known_issue", f"has_known_issue({file_id})"),
            ("incomplete", f"incomplete({file_id})"),
            ("flagged", f"flagged({file_id})"),
        ]

        for name, query in queries:
            result = self.pln.query(query)
            if result:
                results[name] = {
                    "strength": result.strength,
                    "confidence": result.confidence,
                    "probability": result.to_probability(),
                }

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get reasoner statistics."""
        return {
            "facts": self.pln.fact_count,
            "rules": self.pln.rule_count,
        }


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def analyze_with_reasoning(
    directory: str,
    with_git: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Full analysis pipeline: Discover → Assert → Reason.
    """
    results = {
        "files_analyzed": 0,
        "rules_loaded": 0,
        "risk_assessments": [],
    }

    # Step 1: Run codebase analysis
    print("[1/4] Running codebase analysis...")
    analysis = analyze_directory(directory, with_git=with_git)

    if not analysis:
        print("No analysis results")
        return results

    findings = analysis.get("findings", [])
    git_analysis = analysis.get("git_analysis", {})
    print(f"      Found {len(findings)} findings")

    # Step 2: Initialize reasoner
    print("\n[2/4] Initializing PLN reasoner...")
    reasoner = AuditReasoner()

    # Load rules from various sources
    woven_rules = reasoner.load_rules_from_woven_mind()
    manual_rules = reasoner.load_manual_rules()
    reasoner.add_default_rules()

    stats = reasoner.get_stats()
    print(f"      Loaded {stats['rules']} rules")
    print(f"        From WovenMind: {woven_rules}")
    print(f"        Manual rules: {manual_rules}")
    print(f"        Default rules: {stats['rules'] - woven_rules - manual_rules}")

    results["rules_loaded"] = stats["rules"]

    # Step 3: Assert facts from findings
    print("\n[3/4] Asserting facts about files...")

    # Group findings by file
    findings_by_file = defaultdict(list)
    for f in findings:
        finding_id = f.get("id", "")
        if ":" in finding_id:
            file_path = finding_id.rsplit(":", 1)[0]
            findings_by_file[file_path].append(f)

    # Get high churn files
    high_churn = {f for f, _ in git_analysis.get("high_churn_files", [])}

    for file_path, file_findings in findings_by_file.items():
        patterns = [f.get("pattern", "") for f in file_findings]
        traits = []
        if file_path in high_churn:
            traits.append("high_churn")

        # Get directories
        parts = file_path.split("/")
        dirs = parts[:-1] if len(parts) > 1 else []

        reasoner.assert_file_facts(file_path, patterns, traits, dirs)

    results["files_analyzed"] = len(findings_by_file)
    print(f"      Asserted facts for {len(findings_by_file)} files")

    # Step 4: Query risk for each file
    print("\n[4/4] Running PLN inference...")

    risk_assessments = []
    for file_path in findings_by_file.keys():
        risk = reasoner.query_file_risk(file_path)
        if risk:
            # Calculate overall risk score
            overall = max(
                (r.get("probability", 0) for r in risk.values()),
                default=0
            )
            risk_assessments.append({
                "file": file_path,
                "overall_risk": overall,
                "details": risk,
            })

    # Sort by risk
    risk_assessments.sort(key=lambda x: -x["overall_risk"])
    results["risk_assessments"] = risk_assessments

    print(f"      Computed risk for {len(risk_assessments)} files")

    return results


def generate_reasoning_report(results: Dict[str, Any], verbose: bool = False) -> str:
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  AUDIT REASONING - PLN Probabilistic Risk Assessment")
    lines.append("=" * 70)

    lines.append(f"\n[Summary]")
    lines.append(f"  Files analyzed: {results['files_analyzed']}")
    lines.append(f"  Rules loaded: {results['rules_loaded']}")

    assessments = results.get("risk_assessments", [])

    if assessments:
        lines.append(f"\n[Risk Rankings]")
        lines.append(f"  Top files by inferred risk:")

        for i, assessment in enumerate(assessments[:15], 1):
            risk = assessment["overall_risk"]
            file_path = assessment["file"]

            # Risk level label
            if risk > 0.7:
                level = "HIGH"
            elif risk > 0.5:
                level = "MEDIUM"
            else:
                level = "LOW"

            lines.append(f"\n  {i}. [{level}] {file_path}")
            lines.append(f"     Overall risk: {risk:.1%}")

            if verbose:
                details = assessment.get("details", {})
                for aspect, values in details.items():
                    prob = values.get("probability", 0)
                    conf = values.get("confidence", 0)
                    lines.append(f"       {aspect}: {prob:.1%} (conf: {conf:.1%})")
    else:
        lines.append(f"\n[Risk Rankings]")
        lines.append(f"  No files with inferred risk")

    lines.append(f"\n[Interpretation]")
    lines.append(f"  • Risk scores combine multiple signals probabilistically")
    lines.append(f"  • Higher confidence = more evidence supporting the inference")
    lines.append(f"  • Rules come from WovenMind patterns + manual rules + defaults")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Audit Reasoning - PLN-based risk assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", nargs="?", default="cortical/",
                        help="Directory to analyze")
    parser.add_argument("--with-git", action="store_true",
                        help="Include git history analysis")
    parser.add_argument("--show-rules", action="store_true",
                        help="Show loaded rules and exit")
    parser.add_argument("--add-rule", nargs=3, metavar=("ANT", "CONS", "STRENGTH"),
                        help="Add manual rule: antecedent consequent strength")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    # Handle rule management
    if args.add_rule:
        rules = load_rules()
        ant, cons, strength = args.add_rule
        rules["manual_rules"].append({
            "antecedent": ant,
            "consequent": cons,
            "strength": float(strength),
            "confidence": 0.8,
            "added": datetime.now().isoformat(),
        })
        save_rules(rules)
        print(f"Added rule: {ant} → {cons} [{strength}]")
        return

    if args.show_rules:
        rules = load_rules()

        print("=" * 60)
        print("  PLN AUDIT RULES")
        print("=" * 60)

        manual = rules.get("manual_rules", [])
        print(f"\n[Manual Rules] ({len(manual)})")
        for r in manual:
            print(f"  {r['antecedent']} → {r['consequent']} [{r['strength']}]")

        # Also show what would be derived from WovenMind
        abstractions = load_woven_mind_abstractions()
        print(f"\n[WovenMind Abstractions Available] ({len(abstractions)})")
        for a in abstractions[:10]:
            nodes = a.get("source_nodes", [])
            freq = a.get("frequency", 0)
            strength = a.get("strength", 0)
            # Create readable interpretation
            parts = []
            for node in nodes:
                if ":" in node:
                    prefix, value = node.split(":", 1)
                    parts.append(f"{prefix}={value}")
            interp = " + ".join(parts) if parts else str(nodes)
            print(f"  {a.get('id', '?')}: {interp} (freq={freq}, str={strength:.2f})")

        return

    # Normal analysis
    print("=" * 70)
    print("  Audit Reasoning - WovenMind → PLN Pipeline")
    print("=" * 70)
    print()

    results = analyze_with_reasoning(
        args.directory,
        with_git=args.with_git,
        verbose=args.verbose
    )

    print()
    report = generate_reasoning_report(results, verbose=args.verbose)
    print(report)


if __name__ == "__main__":
    main()
