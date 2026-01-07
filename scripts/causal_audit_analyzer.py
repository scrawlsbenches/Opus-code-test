#!/usr/bin/env python3
"""
Causal Audit Analyzer - PRISM-Causal Integration for Codebase Health

Combines audit findings with causal reasoning to:
1. Build causal models of code quality issues
2. Analyze root causes of problems
3. Predict impact of fixes
4. Prioritize which issues to address first

Uses PRISM-Causal for do-calculus and counterfactual reasoning.

Usage:
    python scripts/causal_audit_analyzer.py [directory]
    python scripts/causal_audit_analyzer.py --with-git cortical/
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.prism_causal import CausalGraph, CausalExplainer

# Import from our codebase health analyzer
from scripts.codebase_health import analyze_directory, get_file_churn


# =============================================================================
# CAUSAL MODEL FOR CODE QUALITY
# =============================================================================

class CodeQualityCausalModel:
    """
    Causal model for understanding code quality issues.

    Models relationships like:
    - high_churn -> misleading_comments (frequently changed files accumulate stale docs)
    - misleading_comments -> developer_confusion
    - developer_confusion -> bugs
    - stale_todos -> technical_debt
    """

    def __init__(self):
        self.graph = CausalGraph()
        self._build_base_model()

    def _build_base_model(self):
        """Build the base causal model for code quality."""
        # File churn effects
        self.graph.add_cause("high_churn", "misleading_comments", strength=0.6)
        self.graph.add_cause("high_churn", "stale_todos", strength=0.5)
        self.graph.add_cause("high_churn", "code_complexity", strength=0.4)

        # Comment quality effects
        self.graph.add_cause("misleading_comments", "developer_confusion", strength=0.8)
        self.graph.add_cause("stale_todos", "technical_debt", strength=0.7)
        self.graph.add_cause("stale_todos", "developer_confusion", strength=0.3)

        # Confusion effects
        self.graph.add_cause("developer_confusion", "incorrect_assumptions", strength=0.7)
        self.graph.add_cause("incorrect_assumptions", "bugs", strength=0.6)
        self.graph.add_cause("incorrect_assumptions", "wasted_time", strength=0.8)

        # Complexity effects
        self.graph.add_cause("code_complexity", "bugs", strength=0.5)
        self.graph.add_cause("code_complexity", "developer_confusion", strength=0.4)

        # Technical debt effects
        self.graph.add_cause("technical_debt", "slow_development", strength=0.7)
        self.graph.add_cause("technical_debt", "bugs", strength=0.4)

    def add_finding_evidence(self, finding_type: str, count: int):
        """Add evidence from audit findings to adjust model strengths."""
        # More findings of a type increase its causal influence
        if finding_type == "misleading" and count > 5:
            self.graph.add_cause("misleading_comments", "developer_confusion",
                               strength=min(0.95, 0.8 + count * 0.02))
        elif finding_type == "stale_todo" and count > 3:
            self.graph.add_cause("stale_todos", "technical_debt",
                               strength=min(0.95, 0.7 + count * 0.03))

    def analyze_intervention(self, fix_target: str) -> Dict[str, float]:
        """
        Analyze the impact of fixing a specific issue type.

        Returns probabilities of outcomes if we intervene on the target.
        """
        outcomes = {}

        # Key outcomes we care about
        outcome_vars = ["bugs", "developer_confusion", "wasted_time",
                       "technical_debt", "slow_development"]

        for outcome in outcome_vars:
            # P(outcome | do(fix_target = False)) - what if we eliminate the issue?
            try:
                p_outcome = self.graph.intervene(outcome, do={fix_target: False})
                outcomes[outcome] = p_outcome
            except (KeyError, ValueError):
                pass

        return outcomes

    def get_root_causes(self, symptom: str) -> List[Tuple[str, float]]:
        """Find root causes of a symptom with their causal strengths."""
        causes = []

        # Get direct causes using the internal _edges structure
        # _edges is: cause -> [CausalEdge(cause, effect, strength), ...]
        for cause, edges in self.graph._edges.items():
            for edge in edges:
                if edge.effect == symptom:
                    causes.append((edge.cause, edge.strength))

        # Sort by strength
        causes.sort(key=lambda x: -x[1])
        return causes

    def prioritize_fixes(self, findings: Dict[str, int]) -> List[Tuple[str, float, str]]:
        """
        Prioritize which findings to fix based on causal impact.

        Returns: List of (finding_type, impact_score, reason)
        """
        priorities = []

        fix_targets = {
            "misleading_comments": "misleading",
            "stale_todos": "stale_todo",
            "high_churn": "high_churn",
        }

        for causal_var, finding_key in fix_targets.items():
            count = findings.get(finding_key, 0)
            if count == 0:
                continue

            # Get impact of fixing this issue
            impacts = self.analyze_intervention(causal_var)

            # Calculate aggregate impact score
            # Weight: bugs most important, then wasted_time, then technical_debt
            weights = {
                "bugs": 3.0,
                "wasted_time": 2.0,
                "developer_confusion": 1.5,
                "technical_debt": 1.0,
                "slow_development": 1.0,
            }

            impact_score = sum(
                (1 - p) * weights.get(outcome, 1.0)  # Lower p after fix = higher impact
                for outcome, p in impacts.items()
            )

            # Scale by count of findings
            impact_score *= min(count, 10) / 10  # Cap at 10 for normalization

            # Generate reason
            top_impact = max(impacts.items(), key=lambda x: (1-x[1]) * weights.get(x[0], 1))
            reason = f"Fixing {count} {finding_key} issues reduces {top_impact[0]} risk"

            priorities.append((finding_key, impact_score, reason))

        # Sort by impact score descending
        priorities.sort(key=lambda x: -x[1])
        return priorities


# =============================================================================
# CAUSAL ANALYSIS REPORT
# =============================================================================

def generate_causal_report(findings: List[Dict], churn_data: Dict[str, int]) -> str:
    """Generate a causal analysis report from audit findings."""

    # Count finding types
    finding_counts = defaultdict(int)
    stale_count = 0

    for f in findings:
        pattern = f.get('pattern', 'unknown')
        finding_counts[pattern] += 1
        if f.get('stale'):
            stale_count += 1

    # Count high-churn files
    high_churn_count = sum(1 for _, count in churn_data.items() if count > 10)

    # Build causal model with evidence
    model = CodeQualityCausalModel()
    model.add_finding_evidence("misleading", finding_counts.get("FUTURE:", 0))
    model.add_finding_evidence("stale_todo", stale_count)

    # Prepare findings dict for prioritization
    findings_dict = {
        "misleading": finding_counts.get("FUTURE:", 0) + finding_counts.get("will be", 0),
        "stale_todo": stale_count,
        "high_churn": high_churn_count,
    }

    # Generate report
    lines = []
    lines.append("=" * 60)
    lines.append("  CAUSAL AUDIT ANALYSIS (PRISM-Causal)")
    lines.append("=" * 60)

    # Finding summary
    lines.append("\n[Finding Summary]")
    lines.append(f"  Misleading comments: {findings_dict['misleading']}")
    lines.append(f"  Stale TODOs (>180 days): {findings_dict['stale_todo']}")
    lines.append(f"  High-churn files (>10 commits): {findings_dict['high_churn']}")

    # Root cause analysis
    lines.append("\n[Root Cause Analysis]")
    lines.append("  Causes of developer confusion:")
    for cause, strength in model.get_root_causes("developer_confusion"):
        lines.append(f"    • {cause}: {strength:.0%} causal strength")

    lines.append("\n  Causes of bugs:")
    for cause, strength in model.get_root_causes("bugs"):
        lines.append(f"    • {cause}: {strength:.0%} causal strength")

    # Impact analysis
    lines.append("\n[Intervention Analysis]")
    lines.append("  What if we fix misleading comments?")
    impacts = model.analyze_intervention("misleading_comments")
    for outcome, prob in sorted(impacts.items(), key=lambda x: x[1]):
        reduction = (1 - prob) * 100
        lines.append(f"    → {outcome}: {reduction:.0f}% risk reduction")

    # Prioritization
    lines.append("\n[Recommended Fix Priority]")
    priorities = model.prioritize_fixes(findings_dict)
    for i, (finding_type, score, reason) in enumerate(priorities, 1):
        lines.append(f"  {i}. {finding_type.upper()} (impact: {score:.1f})")
        lines.append(f"     {reason}")

    # Counterfactual
    lines.append("\n[Counterfactual Reasoning]")
    if findings_dict['misleading'] > 0:
        lines.append("  Q: Would bugs decrease if comments were accurate?")
        p_bugs_now = model.graph.observe("bugs", given={"misleading_comments": True})
        p_bugs_fixed = model.graph.intervene("bugs", do={"misleading_comments": False})
        lines.append(f"  A: Bug probability drops from {p_bugs_now:.0%} to {p_bugs_fixed:.0%}")
        lines.append(f"     → {(p_bugs_now - p_bugs_fixed) * 100:.0f}% fewer bugs expected")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Causal Audit Analyzer")
    parser.add_argument("directory", nargs="?", default="cortical/",
                        help="Directory to analyze")
    parser.add_argument("--with-git", action="store_true",
                        help="Include git history analysis")

    args = parser.parse_args()

    print("Running codebase health analysis...")
    print("-" * 60)

    # Run the base analysis
    results = analyze_directory(args.directory, with_git=args.with_git)

    if not results:
        print("No results from analysis")
        return

    # Get churn data if available
    churn_data = results.get('git_analysis', {}).get('high_churn_files', {})
    if isinstance(churn_data, list):
        churn_data = dict(churn_data)

    # Generate causal analysis report
    print()
    report = generate_causal_report(
        findings=results.get('findings', []),
        churn_data=churn_data
    )
    print(report)


if __name__ == "__main__":
    main()
