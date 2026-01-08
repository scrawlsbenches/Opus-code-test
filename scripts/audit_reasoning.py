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

from cortical.reasoning.prism_pln import (
    PLNReasoner, TruthValue, AttentionalFocus, AttentionValue,
    Term, TypeRegistry, aggregate_truth_values, AggregateStrategy
)
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
    PLN-based reasoning for audit findings with Full PLN capabilities.

    Combines:
    - Facts from current audit (file has X, Y, Z)
    - Rules from WovenMind patterns and manual rules
    - Multi-rule aggregation for combining evidence
    - Attention-based focus for prioritized inference
    - Importance weights (STI/LTI) for atom prioritization
    - Compound terms for complex multi-signal rules
    """

    def __init__(self, aggregate_strategy: AggregateStrategy = "revision"):
        self.pln = PLNReasoner()
        self.rules_config = load_rules()
        self.aggregate_strategy = aggregate_strategy

        # Attention tracking for files
        self.attention_focus = AttentionalFocus(max_size=50, default_boost=1.5)

        # Type registry for compound term constraints
        self.type_registry = TypeRegistry()
        self._setup_type_registry()

        # Track file importance
        self.file_importance: Dict[str, AttentionValue] = {}

    def _setup_type_registry(self) -> None:
        """Set up type constraints for audit domain."""
        # Register file types
        self.type_registry.register_type("File", [])  # Will be populated dynamically
        self.type_registry.register_type("Directory", [
            "legacy", "api", "utils", "core", "services", "reasoning", "cdg", "got"
        ])
        self.type_registry.register_type("Pattern", [
            "todo", "fixme", "hack", "future", "xxx", "should_be", "will_be",
            "see_docs", "eventually", "planned_to"
        ])
        self.type_registry.register_type("Trait", [
            "high_churn", "bug_prone", "complex", "critical", "stable"
        ])
        self.type_registry.register_type("RiskLevel", ["low", "medium", "high", "critical"])

    def load_rules_from_woven_mind(self) -> int:
        """Load and convert WovenMind abstractions to PLN rules with compound terms."""
        abstractions = load_woven_mind_abstractions()
        count = 0

        for abstraction in abstractions:
            rule = abstraction_to_rule(abstraction)
            if rule:
                # Use compound terms for multi-condition rules
                nodes = abstraction.get("source_nodes", [])
                if len(nodes) >= 2:
                    # Create compound antecedent using logical conjunction
                    # e.g., and(has_dir(X, legacy), has_pattern(X, todo))
                    parts = []
                    for node in nodes:
                        if ":" in node:
                            prefix, value = node.split(":", 1)
                            if prefix in ("dir", "pattern", "trait"):
                                parts.append(f"has_{prefix}(X, {value})")

                    if len(parts) >= 2:
                        # Assert as compound rule
                        compound_ant = f"and({', '.join(parts)})"
                        self.pln.assert_compound_rule(
                            compound_ant,
                            "flagged(X)",
                            strength=rule["strength"],
                            confidence=rule["confidence"]
                        )
                        count += 1
                else:
                    # Single antecedent - use simple rule
                    for node in nodes:
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
        """Load manually defined rules with aggregation support."""
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
        """Add sensible default rules for code audit with compound terms."""
        # Simple rules
        simple_rules = [
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
            # High churn + incomplete = high risk
            ("has_trait(X, high_churn)", "risky(X)", 0.6),
            ("incomplete(X)", "risky(X)", 0.5),
        ]

        for ant, cons, strength in simple_rules:
            self.pln.assert_rule(ant, cons, strength=strength, confidence=0.7)

        # Compound rules for multi-signal detection
        compound_rules = [
            # TODO + HACK = definitely needs review (compound evidence)
            ("and(has_pattern(X, todo), has_pattern(X, hack))", "needs_urgent_review(X)", 0.85),
            # Legacy + TODO = technical debt
            ("and(has_dir(X, legacy), has_pattern(X, todo))", "technical_debt(X)", 0.8),
            # High churn + FIXME = risky
            ("and(has_trait(X, high_churn), has_pattern(X, fixme))", "risky(X)", 0.75),
            # Bug prone + multiple patterns = critical
            ("and(has_trait(X, bug_prone), has_pattern(X, xxx))", "critical_review(X)", 0.8),
        ]

        for ant, cons, strength in compound_rules:
            self.pln.assert_compound_rule(ant, cons, strength=strength, confidence=0.75)

    def assert_file_facts(
        self,
        file_path: str,
        patterns: List[str],
        traits: List[str],
        directories: List[str],
        initial_importance: Optional[float] = None
    ) -> None:
        """Assert facts about a file with importance tracking."""
        # Normalize file identifier
        file_id = Path(file_path).name.replace(".", "_")

        # Register file in type system
        self.type_registry.register_type("File", [file_id])

        for pattern in patterns:
            pattern_clean = pattern.replace(" ", "_").replace(":", "").lower()
            self.pln.assert_fact(
                f"has_pattern({file_id}, {pattern_clean})",
                strength=0.95,
                confidence=0.9
            )
            # Also assert as compound fact for pattern matching
            self.pln.assert_compound_fact(
                f"file_pattern({file_id}, {pattern_clean})",
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

        # Set up importance tracking for this file
        if initial_importance is not None:
            sti = initial_importance
        else:
            # Calculate initial STI based on traits
            sti = 0.3  # Base importance
            if "high_churn" in traits:
                sti += 0.3
            if "bug_prone" in traits:
                sti += 0.2
            if len(patterns) > 2:
                sti += 0.1 * (len(patterns) - 2)

        self.file_importance[file_id] = AttentionValue(
            sti=min(1.0, sti),
            lti=0.2 if "critical" in traits else 0.1,
            vlti="critical" in traits
        )

        # Set attention in PLN reasoner
        self.pln.set_attention(file_id, self.file_importance[file_id])

    def focus_on_high_risk_files(self, threshold: float = 0.5) -> int:
        """Focus attention on files with high importance."""
        focused = []
        for file_id, attention in self.file_importance.items():
            if attention.total_importance() >= threshold:
                focused.append(file_id)

        if focused:
            self.attention_focus.focus_on(focused, boost=2.0)

        return len(focused)

    def query_file_risk(
        self,
        file_path: str,
        use_attention: bool = True,
        use_importance: bool = True
    ) -> Dict[str, Any]:
        """Query the risk assessment for a file with Full PLN features."""
        file_id = Path(file_path).name.replace(".", "_")

        results = {}

        # Query different risk aspects
        queries = [
            ("needs_review", f"needs_review({file_id})"),
            ("has_known_issue", f"has_known_issue({file_id})"),
            ("incomplete", f"incomplete({file_id})"),
            ("flagged", f"flagged({file_id})"),
            ("risky", f"risky({file_id})"),
            ("technical_debt", f"technical_debt({file_id})"),
            ("needs_urgent_review", f"needs_urgent_review({file_id})"),
            ("critical_review", f"critical_review({file_id})"),
        ]

        for name, query in queries:
            # Use attention-guided or importance-guided inference
            if use_attention and file_id in self.attention_focus._focused:
                result = self.pln.query_with_attention(
                    query,
                    self.attention_focus,
                    aggregate=self.aggregate_strategy
                )
            elif use_importance:
                result = self.pln.query_with_importance(
                    query,
                    spread_importance=True,
                    aggregate=self.aggregate_strategy
                )
            else:
                result = self.pln.query(query, aggregate=self.aggregate_strategy)

            if result:
                tv = result.truth_value if hasattr(result, 'truth_value') else result
                results[name] = {
                    "strength": tv.strength,
                    "confidence": tv.confidence,
                    "probability": tv.to_probability(),
                }

        # Add importance info to results
        if file_id in self.file_importance:
            attention = self.file_importance[file_id]
            results["_importance"] = {
                "sti": attention.sti,
                "lti": attention.lti,
                "vlti": attention.vlti,
                "total": attention.total_importance(),
            }

        return results

    def query_with_aggregation(
        self,
        query: str,
        strategies: List[AggregateStrategy] = None
    ) -> Dict[str, Any]:
        """Query using multiple aggregation strategies and compare results."""
        if strategies is None:
            strategies = ["first", "revision", "max", "or", "weighted"]

        results = {}
        for strategy in strategies:
            result = self.pln.query(query, aggregate=strategy)
            if result:
                results[strategy] = {
                    "strength": result.strength,
                    "confidence": result.confidence,
                    "probability": result.to_probability(),
                }

        return results

    def collect_rent(self, sti_decay: float = 0.9, lti_decay: float = 0.99) -> None:
        """Apply attention decay to all tracked files (rent collection)."""
        for file_id, attention in self.file_importance.items():
            attention.decay_sti(sti_decay)
            attention.decay_lti(lti_decay)
            self.pln.set_attention(file_id, attention)

    def stimulate_file(self, file_path: str, amount: float = 0.2) -> None:
        """Stimulate a file's importance (e.g., when it's accessed or modified)."""
        file_id = Path(file_path).name.replace(".", "_")
        if file_id in self.file_importance:
            self.file_importance[file_id].stimulate(amount)
            self.pln.set_attention(file_id, self.file_importance[file_id])

    def get_priority_files(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the top N files by total importance."""
        sorted_files = sorted(
            self.file_importance.items(),
            key=lambda x: x[1].total_importance(),
            reverse=True
        )
        return [(f, a.total_importance()) for f, a in sorted_files[:top_n]]

    def get_vlti_files(self) -> List[str]:
        """Get files marked as critically important (VLTI=True)."""
        return [f for f, a in self.file_importance.items() if a.vlti]

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner statistics including attention stats."""
        return {
            "facts": self.pln.fact_count,
            "rules": self.pln.rule_count,
            "files_tracked": len(self.file_importance),
            "vlti_files": len(self.get_vlti_files()),
            "aggregate_strategy": self.aggregate_strategy,
            "attention_focus_size": len(self.attention_focus._focused),
        }


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def analyze_with_reasoning(
    directory: str,
    with_git: bool = False,
    verbose: bool = False,
    aggregate_strategy: AggregateStrategy = "revision",
    enable_attention: bool = True,
    enable_importance: bool = True
) -> Dict[str, Any]:
    """
    Full analysis pipeline with Full PLN features: Discover → Assert → Reason.

    Features:
    - Multi-rule aggregation for combining evidence from multiple inference paths
    - Attention-based focus for prioritized inference
    - Importance weights (STI/LTI) for file prioritization
    - Compound terms for complex multi-signal rules
    """
    results = {
        "files_analyzed": 0,
        "rules_loaded": 0,
        "risk_assessments": [],
        "priority_files": [],
        "vlti_files": [],
        "aggregate_strategy": aggregate_strategy,
    }

    # Step 1: Run codebase analysis
    print("[1/5] Running codebase analysis...")
    analysis = analyze_directory(directory, with_git=with_git)

    if not analysis:
        print("No analysis results")
        return results

    findings = analysis.get("findings", [])
    git_analysis = analysis.get("git_analysis", {})
    print(f"      Found {len(findings)} findings")

    # Step 2: Initialize reasoner with Full PLN features
    print("\n[2/5] Initializing Full PLN reasoner...")
    reasoner = AuditReasoner(aggregate_strategy=aggregate_strategy)

    # Load rules from various sources
    woven_rules = reasoner.load_rules_from_woven_mind()
    manual_rules = reasoner.load_manual_rules()
    reasoner.add_default_rules()

    stats = reasoner.get_stats()
    print(f"      Loaded {stats['rules']} rules")
    print(f"        From WovenMind (compound terms): {woven_rules}")
    print(f"        Manual rules: {manual_rules}")
    print(f"        Default rules (simple + compound): {stats['rules'] - woven_rules - manual_rules}")
    print(f"      Aggregation strategy: {aggregate_strategy}")

    results["rules_loaded"] = stats["rules"]

    # Step 3: Assert facts from findings with importance tracking
    print("\n[3/5] Asserting facts with importance tracking...")

    # Group findings by file
    findings_by_file = defaultdict(list)
    for f in findings:
        finding_id = f.get("id", "")
        if ":" in finding_id:
            file_path = finding_id.rsplit(":", 1)[0]
            findings_by_file[file_path].append(f)

    # Get high churn and bug-prone files
    high_churn = {f for f, _ in git_analysis.get("high_churn_files", [])}
    bug_prone = {f for f, _ in git_analysis.get("bug_prone_files", [])}
    critical_modules = {f for f, _ in git_analysis.get("critical_modules", [])}

    for file_path, file_findings in findings_by_file.items():
        patterns = [f.get("pattern", "") for f in file_findings]
        traits = []
        if file_path in high_churn:
            traits.append("high_churn")
        if file_path in bug_prone:
            traits.append("bug_prone")
        if file_path in critical_modules:
            traits.append("critical")

        # Get directories
        parts = file_path.split("/")
        dirs = parts[:-1] if len(parts) > 1 else []

        # Calculate initial importance based on traits and pattern count
        initial_importance = None
        if enable_importance:
            importance = 0.3
            if "high_churn" in traits:
                importance += 0.25
            if "bug_prone" in traits:
                importance += 0.2
            if "critical" in traits:
                importance += 0.15
            if len(patterns) > 3:
                importance += 0.1
            initial_importance = min(1.0, importance)

        reasoner.assert_file_facts(file_path, patterns, traits, dirs, initial_importance)

    results["files_analyzed"] = len(findings_by_file)
    print(f"      Asserted facts for {len(findings_by_file)} files")
    print(f"      Files with importance tracking: {len(reasoner.file_importance)}")

    # Step 4: Focus attention on high-risk files
    if enable_attention:
        print("\n[4/5] Focusing attention on high-risk files...")
        focused_count = reasoner.focus_on_high_risk_files(threshold=0.5)
        print(f"      Focused on {focused_count} high-importance files")
    else:
        print("\n[4/5] Attention focusing disabled")

    # Step 5: Query risk for each file with Full PLN inference
    print("\n[5/5] Running Full PLN inference...")

    risk_assessments = []
    for file_path in findings_by_file.keys():
        risk = reasoner.query_file_risk(
            file_path,
            use_attention=enable_attention,
            use_importance=enable_importance
        )
        if risk:
            # Calculate overall risk score (excluding _importance key)
            risk_scores = [
                r.get("probability", 0)
                for key, r in risk.items()
                if not key.startswith("_") and isinstance(r, dict)
            ]
            overall = max(risk_scores, default=0)

            # Get importance info
            importance_info = risk.get("_importance", {})

            risk_assessments.append({
                "file": file_path,
                "overall_risk": overall,
                "importance": importance_info.get("total", 0),
                "sti": importance_info.get("sti", 0),
                "lti": importance_info.get("lti", 0),
                "vlti": importance_info.get("vlti", False),
                "details": {k: v for k, v in risk.items() if not k.startswith("_")},
            })

    # Sort by combined risk and importance
    risk_assessments.sort(key=lambda x: -(x["overall_risk"] * 0.7 + x["importance"] * 0.3))
    results["risk_assessments"] = risk_assessments

    # Get priority and VLTI files
    results["priority_files"] = reasoner.get_priority_files(top_n=10)
    results["vlti_files"] = reasoner.get_vlti_files()

    print(f"      Computed risk for {len(risk_assessments)} files")
    print(f"      Priority files (by importance): {len(results['priority_files'])}")
    print(f"      Critical (VLTI) files: {len(results['vlti_files'])}")

    # Store final stats
    results["stats"] = reasoner.get_stats()

    return results


def generate_reasoning_report(results: Dict[str, Any], verbose: bool = False) -> str:
    """Generate human-readable report with Full PLN features."""
    lines = []
    lines.append("=" * 70)
    lines.append("  AUDIT REASONING - Full PLN Probabilistic Risk Assessment")
    lines.append("=" * 70)

    lines.append(f"\n[Summary]")
    lines.append(f"  Files analyzed: {results['files_analyzed']}")
    lines.append(f"  Rules loaded: {results['rules_loaded']}")
    lines.append(f"  Aggregation strategy: {results.get('aggregate_strategy', 'revision')}")

    stats = results.get("stats", {})
    if stats:
        lines.append(f"  Files with importance tracking: {stats.get('files_tracked', 0)}")
        lines.append(f"  Critical (VLTI) files: {stats.get('vlti_files', 0)}")

    # Priority files section
    priority_files = results.get("priority_files", [])
    if priority_files:
        lines.append(f"\n[Priority Files by Importance]")
        lines.append(f"  Top {len(priority_files)} files by STI+LTI importance:")
        for i, (file_id, importance) in enumerate(priority_files[:5], 1):
            lines.append(f"    {i}. {file_id}: {importance:.2%}")

    # VLTI (critical) files
    vlti_files = results.get("vlti_files", [])
    if vlti_files:
        lines.append(f"\n[Critical Files (VLTI=True)]")
        lines.append(f"  These files are pinned and will not decay:")
        for file_id in vlti_files[:5]:
            lines.append(f"    • {file_id}")

    assessments = results.get("risk_assessments", [])

    if assessments:
        lines.append(f"\n[Risk Rankings]")
        lines.append(f"  Top files by combined risk (70%) + importance (30%):")

        for i, assessment in enumerate(assessments[:15], 1):
            risk = assessment["overall_risk"]
            importance = assessment.get("importance", 0)
            file_path = assessment["file"]

            # Risk level label
            if risk > 0.7:
                level = "HIGH  "
            elif risk > 0.5:
                level = "MEDIUM"
            else:
                level = "LOW   "

            vlti_marker = " [CRITICAL]" if assessment.get("vlti", False) else ""
            lines.append(f"\n  {i}. [{level}] {file_path}{vlti_marker}")
            lines.append(f"     Risk: {risk:.1%} | Importance: {importance:.1%}")

            if verbose:
                # Show STI/LTI details
                sti = assessment.get("sti", 0)
                lti = assessment.get("lti", 0)
                lines.append(f"     STI: {sti:.2f} | LTI: {lti:.2f}")

                details = assessment.get("details", {})
                for aspect, values in details.items():
                    if isinstance(values, dict):
                        prob = values.get("probability", 0)
                        conf = values.get("confidence", 0)
                        lines.append(f"       {aspect}: {prob:.1%} (conf: {conf:.1%})")
    else:
        lines.append(f"\n[Risk Rankings]")
        lines.append(f"  No files with inferred risk")

    lines.append(f"\n[Full PLN Features Used]")
    lines.append(f"  • Multi-rule aggregation: Combines evidence from multiple inference paths")
    lines.append(f"  • Attention focus: Prioritizes inference on high-importance atoms")
    lines.append(f"  • Importance weights: STI (short-term) + LTI (long-term) importance")
    lines.append(f"  • VLTI pinning: Critical files marked for permanent attention")
    lines.append(f"  • Compound terms: Complex multi-signal rules (e.g., TODO+HACK → urgent)")

    lines.append(f"\n[Interpretation]")
    lines.append(f"  • Risk scores combine multiple signals using '{results.get('aggregate_strategy', 'revision')}' aggregation")
    lines.append(f"  • Higher confidence = more evidence supporting the inference")
    lines.append(f"  • Importance decays over time (rent collection) unless VLTI=True")
    lines.append(f"  • Rules come from WovenMind patterns + manual rules + defaults")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Audit Reasoning - Full PLN-based risk assessment with attention and importance",
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
                        help="Show detailed output including STI/LTI values")

    # Full PLN feature flags
    parser.add_argument("--aggregate", choices=["first", "revision", "max", "or", "weighted"],
                        default="revision",
                        help="Multi-rule aggregation strategy (default: revision)")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable attention-based focus")
    parser.add_argument("--no-importance", action="store_true",
                        help="Disable importance weight tracking")

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
        print("  PLN AUDIT RULES (Full PLN)")
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

        # Show default compound rules
        print(f"\n[Default Compound Rules]")
        print(f"  and(has_pattern(X, todo), has_pattern(X, hack)) → needs_urgent_review(X) [0.85]")
        print(f"  and(has_dir(X, legacy), has_pattern(X, todo)) → technical_debt(X) [0.80]")
        print(f"  and(has_trait(X, high_churn), has_pattern(X, fixme)) → risky(X) [0.75]")
        print(f"  and(has_trait(X, bug_prone), has_pattern(X, xxx)) → critical_review(X) [0.80]")

        print(f"\n[Full PLN Features]")
        print(f"  • Aggregation strategies: first, revision, max, or, weighted")
        print(f"  • Attention focus: Prioritizes high-importance atoms")
        print(f"  • Importance weights: STI (short-term), LTI (long-term), VLTI (pinned)")
        print(f"  • Compound terms: Complex multi-condition rules")

        return

    # Normal analysis
    print("=" * 70)
    print("  Audit Reasoning - Full PLN Pipeline")
    print("=" * 70)
    print()

    results = analyze_with_reasoning(
        args.directory,
        with_git=args.with_git,
        verbose=args.verbose,
        aggregate_strategy=args.aggregate,
        enable_attention=not args.no_attention,
        enable_importance=not args.no_importance
    )

    print()
    report = generate_reasoning_report(results, verbose=args.verbose)
    print(report)


if __name__ == "__main__":
    main()
