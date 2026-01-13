"""
PLN-based audit reasoning - WovenMind Discovery → PLN Reasoning Pipeline.

This module provides probabilistic reasoning for audit analysis using PLN
(Probabilistic Logic Networks) with:
- Multi-rule aggregation for combining evidence
- Attention-based focus for prioritized inference
- Importance weights (STI/LTI/VLTI) for atom prioritization
- Natural language query parsing
- Explainability for inference chains

The pipeline:
1. DISCOVER: WovenMind finds patterns in audit data (unsupervised)
2. VALIDATE: Human reviews discovered patterns
3. ENCODE: Convert validated patterns to PLN rules
4. REASON: PLN infers conclusions about files with uncertainty
"""

import re
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from cortical.reasoning.prism_pln import (
    PLNReasoner, TruthValue, AttentionalFocus, AttentionValue,
    Term, TypeRegistry, aggregate_truth_values, AggregateStrategy,
    InferenceStep, InferenceTrace
)
from cortical.reasoning.woven_mind import WovenMind

from .persistence import (
    PersistenceBackend,
    FilePersistenceBackend,
    NullPersistenceBackend,
    InMemoryPersistenceBackend,
    AuditPersistenceState,
    FileImportanceRecord,
    create_default_persistence,
    DEFAULT_WOVEN_MIND_FILE,
)
from .health import analyze_directory, HealthAnalysisResult


# =============================================================================
# NATURAL LANGUAGE QUERY SUPPORT
# =============================================================================

@dataclass
class AuditQuery:
    """
    Structured representation of an audit query.

    Supports natural language queries like:
        "risky files in reasoning/ not tests"
        "why is prism_pln.py flagged"
        "files with high_churn"
    """
    # Scope
    directory: Optional[str] = None
    file_patterns: List[str] = field(default_factory=list)

    # Filters
    negations: List[str] = field(default_factory=list)  # Exclude these
    include_traits: List[str] = field(default_factory=list)  # Must have these traits

    # Intent
    intent: str = "list"  # list, explain, trace
    target_file: Optional[str] = None  # For "why is X flagged"

    # Thresholds
    min_risk: float = 0.0
    max_results: Optional[int] = None

    # Output
    explain: bool = False


def translate_audit_query(query: str) -> AuditQuery:
    """
    Translate natural language to AuditQuery.

    Pattern matching approach (no ML required).

    Examples:
        "risky files in reasoning/"
        → AuditQuery(directory="reasoning/", min_risk=0.5)

        "why is prism_pln.py flagged"
        → AuditQuery(intent="explain", target_file="prism_pln.py")

        "cortical/ not tests"
        → AuditQuery(directory="cortical/", negations=["tests"])
    """
    result = AuditQuery()
    query = query.strip()
    query_lower = query.lower()

    # Intent Detection
    why_match = re.search(r'why\s+is\s+(\S+)\s+(?:flagged|risky|marked)', query_lower)
    if why_match:
        result.intent = "explain"
        result.target_file = why_match.group(1)
        result.explain = True

    explain_match = re.search(r'explain\s+(\S+)', query_lower)
    if explain_match and result.intent != "explain":
        result.intent = "explain"
        result.target_file = explain_match.group(1)
        result.explain = True

    # Scope Extraction - order matters: more specific patterns first
    dir_patterns = [
        r'in\s+(\S+)',      # Captures full path like "cortical/audits"
        r'^(\S+/)\s',       # Path at start followed by space
        r'^(\S+/?)$',       # Path alone (with or without trailing /)
    ]

    for pattern in dir_patterns:
        match = re.search(pattern, query_lower)
        if match:
            potential_dir = match.group(1)
            if '/' in potential_dir or Path(potential_dir).exists():
                result.directory = potential_dir.rstrip('/') + '/'
                break

    # Negation Extraction
    negation_patterns = [
        r'not\s+(\w+)',
        r'without\s+(\w+)',
        r'exclude\s+(\w+)',
        r'excluding\s+(\w+)',
    ]

    for pattern in negation_patterns:
        for match in re.finditer(pattern, query_lower):
            negated = match.group(1)
            if negated not in result.negations:
                result.negations.append(negated)

    # Trait Extraction
    trait_patterns = [
        r'with\s+(high[_\s]?churn)',
        r'with\s+(todo|todos)',
        r'with\s+(fixme)',
        r'with\s+(hack|hacks)',
        r'with\s+([\w_]+)',
        r'has\s+([\w_]+)',
        r'having\s+([\w_]+)',
    ]

    for pattern in trait_patterns:
        for match in re.finditer(pattern, query_lower):
            trait = match.group(1).replace(' ', '_').lower()
            trait_map = {
                'high_churn': 'high_churn',
                'highchurn': 'high_churn',
                'todos': 'todo',
                'hacks': 'hack',
            }
            trait = trait_map.get(trait, trait)
            if trait not in result.include_traits:
                result.include_traits.append(trait)

    # Risk Level Extraction
    if 'critical' in query_lower:
        result.min_risk = 0.9
    elif 'high risk' in query_lower or 'high-risk' in query_lower:
        result.min_risk = 0.7
    elif 'risky' in query_lower:
        result.min_risk = 0.5
    elif 'medium risk' in query_lower:
        result.min_risk = 0.4

    # Result Limit
    limit_match = re.search(r'(?:top|first)\s+(\d+)', query_lower)
    if limit_match:
        result.max_results = int(limit_match.group(1))

    # Fallback directory detection
    if result.directory is None:
        first_word = query.split()[0] if query.split() else ""
        if '/' in first_word or Path(first_word).is_dir():
            result.directory = first_word.rstrip('/') + '/'

    return result


def is_natural_language_query(arg: str) -> bool:
    """Determine if input is a natural language query vs a path/flag."""
    if arg.startswith('--') or arg.startswith('-'):
        return False

    if Path(arg).exists() and ' ' not in arg:
        nlu_keywords = ['not ', 'with ', 'explain', 'why ', 'risky', 'top ']
        arg_lower = arg.lower()
        if not any(kw in arg_lower for kw in nlu_keywords):
            return False

    if ' ' in arg:
        return True

    nlu_keywords = ['not', 'with', 'explain', 'why', 'risky', 'top', 'critical']
    arg_lower = arg.lower()
    for kw in nlu_keywords:
        if kw in arg_lower and kw != arg_lower:
            return True

    return False


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
    """
    nodes = abstraction.get("source_nodes", [])
    frequency = abstraction.get("frequency", 0)
    strength = abstraction.get("strength", 0.5)

    if len(nodes) < 2:
        return None

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

    if len(antecedent_parts) < 2:
        return None

    antecedent = " ∧ ".join(antecedent_parts)
    confidence = min(0.9, 0.3 + 0.1 * math.log(frequency + 1))
    consequent = "flagged(X)"

    return {
        "antecedent": antecedent,
        "consequent": consequent,
        "strength": min(0.8, strength + 0.2),
        "confidence": confidence,
        "source": abstraction.get("id", "unknown"),
        "frequency": frequency,
        "interpretation": abstraction.get("interpretation", ""),
    }


def load_woven_mind_abstractions(
    woven_mind_file: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Load abstractions from WovenMind state file."""
    filepath = woven_mind_file or Path.cwd() / DEFAULT_WOVEN_MIND_FILE
    if not filepath.exists():
        return []

    try:
        with open(filepath, 'r') as f:
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
# DEFAULT RULES
# =============================================================================

# Simple rules: (antecedent, consequent, strength)
DEFAULT_SIMPLE_RULES = [
    ("has_trait(X, high_churn)", "needs_review(X)", 0.7),
    ("has_pattern(X, todo)", "incomplete(X)", 0.6),
    ("has_pattern(X, should_be)", "has_known_issue(X)", 0.5),
    ("has_pattern(X, future)", "deferred_work(X)", 0.6),
    ("has_known_issue(X)", "needs_review(X)", 0.6),
    ("incomplete(X)", "needs_review(X)", 0.5),
    ("has_trait(X, high_churn)", "risky(X)", 0.6),
    ("incomplete(X)", "risky(X)", 0.5),
]

# Compound rules: (antecedent, consequent, strength)
DEFAULT_COMPOUND_RULES = [
    ("and(has_pattern(X, todo), has_pattern(X, hack))", "needs_urgent_review(X)", 0.85),
    ("and(has_dir(X, legacy), has_pattern(X, todo))", "technical_debt(X)", 0.8),
    ("and(has_trait(X, high_churn), has_pattern(X, fixme))", "risky(X)", 0.75),
    ("and(has_trait(X, bug_prone), has_pattern(X, xxx))", "critical_review(X)", 0.8),
]


# =============================================================================
# AUDIT REASONER
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
    - Persistence across sessions for importance tracking
    """

    def __init__(
        self,
        persistence: Optional[PersistenceBackend] = None,
        aggregate_strategy: AggregateStrategy = "revision",
        apply_decay: bool = True,
        use_persistence: bool = True,
    ):
        """
        Initialize the audit reasoner.

        Args:
            persistence: Backend for state/rules persistence
            aggregate_strategy: Strategy for combining multiple rule results
            apply_decay: Whether to apply time-based decay to importance values
            use_persistence: Use persistence if no backend provided
        """
        self.pln = PLNReasoner()
        self.aggregate_strategy = aggregate_strategy

        # Set up persistence backend
        if persistence is not None:
            self._persistence = persistence
        elif use_persistence:
            self._persistence = create_default_persistence()
        else:
            self._persistence = NullPersistenceBackend()

        self.rules_config = self._persistence.load_rules()
        self.attention_focus = AttentionalFocus(max_size=50, default_boost=1.5)
        self.type_registry = TypeRegistry()
        self._setup_type_registry()
        self.file_importance: Dict[str, AttentionValue] = {}
        self._persistence_state: Optional[AuditPersistenceState] = None
        self._load_from_persistence(apply_decay=apply_decay)

    @property
    def persistence(self) -> PersistenceBackend:
        """Get the persistence backend."""
        return self._persistence

    def _load_from_persistence(self, apply_decay: bool = True) -> None:
        """Load importance values from persisted state."""
        self._persistence_state = self._persistence.load_state()

        for file_id, record in self._persistence_state.file_importance.items():
            sti = record.sti
            lti = record.lti

            if apply_decay:
                try:
                    last_seen = datetime.fromisoformat(record.last_seen)
                    hours_elapsed = (datetime.now() - last_seen).total_seconds() / 3600
                    if hours_elapsed > 0:
                        sti_decay = 0.9 ** min(hours_elapsed, 24)
                        lti_decay = 0.99 ** min(hours_elapsed, 168)
                        sti = sti * sti_decay
                        lti = lti * lti_decay
                except (ValueError, TypeError):
                    pass

            self.file_importance[file_id] = AttentionValue(
                sti=sti, lti=lti, vlti=record.vlti
            )
            self.pln.set_attention(file_id, self.file_importance[file_id])

        if self._persistence_state.attention_focus:
            self.attention_focus.focus_on(
                self._persistence_state.attention_focus, boost=1.5
            )

    def save_state(self) -> None:
        """Save current state to persistence."""
        if self._persistence_state is None:
            self._persistence_state = AuditPersistenceState.create_new()

        now = datetime.now().isoformat()
        self._persistence_state.session_count += 1

        for file_id, attention in self.file_importance.items():
            if file_id in self._persistence_state.file_importance:
                record = self._persistence_state.file_importance[file_id]
                record.history.append({
                    "timestamp": now,
                    "sti": record.sti,
                    "lti": record.lti,
                    "vlti": record.vlti,
                })
                record.sti = attention.sti
                record.lti = attention.lti
                record.vlti = attention.vlti
                record.last_seen = now
            else:
                self._persistence_state.file_importance[file_id] = FileImportanceRecord(
                    file_id=file_id,
                    sti=attention.sti,
                    lti=attention.lti,
                    vlti=attention.vlti,
                    last_seen=now,
                    history=[],
                )

        # Use public API for encapsulation
        self._persistence_state.attention_focus = self.attention_focus.get_focused_atoms()
        self._persistence_state.global_stats = {
            "last_aggregate_strategy": self.aggregate_strategy,
            "files_in_focus": len(self.attention_focus.get_focused_atoms()),
            "total_files_tracked": len(self.file_importance),
            "vlti_files": len(self.get_vlti_files()),
        }

        self._persistence.save_state(self._persistence_state)

    def _setup_type_registry(self) -> None:
        """Set up type constraints for audit domain."""
        self.type_registry.register_type("File", [])
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
        """Load and convert WovenMind abstractions to PLN rules."""
        abstractions = load_woven_mind_abstractions()
        count = 0

        for abstraction in abstractions:
            rule = abstraction_to_rule(abstraction)
            if rule:
                nodes = abstraction.get("source_nodes", [])
                if len(nodes) >= 2:
                    parts = []
                    for node in nodes:
                        if ":" in node:
                            prefix, value = node.split(":", 1)
                            if prefix in ("dir", "pattern", "trait"):
                                parts.append(f"has_{prefix}(X, {value})")

                    if len(parts) >= 2:
                        compound_ant = f"and({', '.join(parts)})"
                        self.pln.assert_compound_rule(
                            compound_ant,
                            "flagged(X)",
                            strength=rule["strength"],
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
        for ant, cons, strength in DEFAULT_SIMPLE_RULES:
            self.pln.assert_rule(ant, cons, strength=strength, confidence=0.7)

        for ant, cons, strength in DEFAULT_COMPOUND_RULES:
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
        file_id = Path(file_path).name.replace(".", "_")
        self.type_registry.register_type("File", [file_id])

        for pattern in patterns:
            pattern_clean = pattern.replace(" ", "_").replace(":", "").lower()
            self.pln.assert_fact(
                f"has_pattern({file_id}, {pattern_clean})",
                strength=0.95,
                confidence=0.9
            )
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

        # Set up importance tracking
        sti = initial_importance if initial_importance is not None else 0.3
        lti = 0.1

        # Trait-based boosts
        if "high_churn" in traits:
            sti += 0.3
        if "bug_prone" in traits:
            sti += 0.2
        if "critical" in traits:
            lti = 0.2  # Higher LTI for critical files

        # Pattern-based boosts
        risk_patterns = ["todo", "fixme", "hack", "xxx"]
        pattern_count = sum(1 for p in patterns if p in risk_patterns)
        if pattern_count > 0:
            sti += 0.1 + (pattern_count - 1) * 0.05  # 0.1 for first, +0.05 per extra
        sti = min(1.0, sti)

        # Set VLTI for critical files
        vlti = "critical" in traits

        if file_id not in self.file_importance:
            self.file_importance[file_id] = AttentionValue(sti=sti, lti=lti, vlti=vlti)
        else:
            current = self.file_importance[file_id]
            self.file_importance[file_id] = AttentionValue(
                sti=max(current.sti, sti),
                lti=max(current.lti, lti),
                vlti=current.vlti or vlti  # Preserve VLTI if already set
            )

        self.pln.set_attention(file_id, self.file_importance[file_id])

    def focus_on_high_importance(self, threshold: float = 0.5) -> List[str]:
        """Focus attention on high-importance files."""
        high_importance = [
            file_id for file_id, av in self.file_importance.items()
            if av.sti + av.lti >= threshold or av.vlti
        ]
        if high_importance:
            self.attention_focus.focus_on(high_importance, boost=2.0)
        return high_importance

    def focus_on_high_risk_files(self, threshold: float = 0.5) -> int:
        """
        Focus attention on files with high risk scores.

        Args:
            threshold: Minimum risk level to consider (0.0-1.0)

        Returns:
            Number of high-risk files focused on
        """
        high_risk_files = []
        for file_id in self.file_importance.keys():
            tv = self.query_risk(file_id, aggregate=True)
            if tv and tv.mean() >= threshold:
                high_risk_files.append(file_id)

        if high_risk_files:
            self.attention_focus.focus_on(high_risk_files, boost=2.0)

        return len(high_risk_files)

    def query_risk(
        self,
        file_id: str,
        aggregate: bool = True
    ) -> Optional[TruthValue]:
        """Query risk level for a file."""
        file_id_clean = file_id.replace(".", "_")

        risk_queries = [
            f"risky({file_id_clean})",
            f"needs_review({file_id_clean})",
            f"needs_urgent_review({file_id_clean})",
            f"critical_review({file_id_clean})",
        ]

        results = []
        for query in risk_queries:
            tv = self.pln.query(query)
            if tv:
                results.append(tv)

        if not results:
            return None

        if aggregate and len(results) > 1:
            return aggregate_truth_values(results, strategy=self.aggregate_strategy)

        return max(results, key=lambda tv: tv.mean())

    def get_vlti_files(self) -> List[str]:
        """Get files marked as very long-term important."""
        return [
            file_id for file_id, av in self.file_importance.items()
            if av.vlti
        ]

    def set_vlti(self, file_id: str, vlti: bool = True) -> None:
        """Mark a file as very long-term important (pinned)."""
        file_id_clean = file_id.replace(".", "_")
        if file_id_clean in self.file_importance:
            current = self.file_importance[file_id_clean]
            self.file_importance[file_id_clean] = AttentionValue(
                sti=current.sti,
                lti=max(current.lti, 0.5) if vlti else current.lti,
                vlti=vlti
            )

    def get_importance_history(self, file_id: str) -> List[Dict[str, Any]]:
        """Get the importance history for a specific file."""
        if self._persistence_state and file_id in self._persistence_state.file_importance:
            return self._persistence_state.file_importance[file_id].history
        return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the reasoner state.

        Returns dict with:
        - facts: Number of facts asserted in PLN graph
        - rules: Number of rules in the reasoner
        - aggregate_strategy: Current aggregation strategy
        """
        return {
            "facts": self.pln.fact_count,
            "rules": self.pln.rule_count,
            "aggregate_strategy": self.aggregate_strategy
        }

    def collect_rent(self, sti_decay: float = 0.9, lti_decay: float = 0.95) -> None:
        """
        Apply attention decay to all tracked files.

        Uses AttentionValue decay methods which protect VLTI files:
        - VLTI atoms decay STI slower (max 5% per cycle)
        - VLTI atoms don't decay LTI at all

        Args:
            sti_decay: Factor to multiply STI by (0 < factor < 1)
            lti_decay: Factor to multiply LTI by (typically close to 1)
        """
        for file_id, av in self.file_importance.items():
            av.decay_sti(sti_decay)
            av.decay_lti(lti_decay)

    def get_importance_trend(self, file_id: str) -> Optional[str]:
        """
        Analyze importance history to determine trend direction.

        Compares first and last history entries to determine if file
        importance is increasing, decreasing, or stable over time.

        Args:
            file_id: The file identifier to analyze

        Returns:
            "increasing" - Total importance has grown since first tracked
            "decreasing" - Total importance has declined since first tracked
            "stable" - Total importance unchanged
            None - No history available (fewer than 2 entries)
        """
        history = self.get_importance_history(file_id)
        if len(history) < 2:
            return None

        first = history[0]
        last = history[-1]

        first_total = first.get("sti", 0) + first.get("lti", 0)
        last_total = last.get("sti", 0) + last.get("lti", 0)

        if last_total > first_total:
            return "increasing"
        elif last_total < first_total:
            return "decreasing"
        else:
            return "stable"

    def explain_file_risk(
        self,
        file_id: str,
    ) -> Dict[str, Any]:
        """
        Explain why a file is flagged as risky.

        Returns dict with:
        - facts: List of facts asserted for this file
        - inferences: Inference chains that fired
        - risk_level: Overall risk assessment
        - traces: Reasoning traces (inference chains)
        - summary: Brief summary of the risk assessment
        """
        # Normalize file_id the same way as assert_file_facts
        file_id_clean = Path(file_id).name.replace(".", "_")

        explanation = {
            "file_id": file_id_clean,
            "facts": [],
            "inferences": [],
            "risk_level": None,
            "suggestions": [],
            "traces": {},
            "summary": "",
        }

        # Gather facts
        # Use public API to iterate atoms
        for atom_name, atom_obj in self.pln.graph.iter_atoms():
            if file_id_clean in atom_name:
                tv = atom_obj.truth_value
                explanation["facts"].append({
                    "atom": atom_name,
                    "strength": tv.strength,
                    "confidence": tv.confidence,
                })

        # Query risk and trace inference
        risk_tv = self.query_risk(file_id_clean, aggregate=True)
        if risk_tv:
            explanation["risk_level"] = {
                "strength": risk_tv.strength,
                "confidence": risk_tv.confidence,
                "mean": risk_tv.mean(),
            }

        # Generate suggestions based on facts
        for fact in explanation["facts"]:
            atom = fact["atom"]
            if "has_pattern" in atom and "todo" in atom:
                explanation["suggestions"].append("Review and address TODO comments")
            elif "has_pattern" in atom and "hack" in atom:
                explanation["suggestions"].append("Refactor HACK workarounds")
            elif "has_trait" in atom and "high_churn" in atom:
                explanation["suggestions"].append("Consider stabilizing this high-churn file")

        # Build traces from inference chain
        trace_steps = []
        for fact in explanation["facts"]:
            atom = fact["atom"]
            trace_steps.append({
                "step": "assert",
                "atom": atom,
                "source": "codebase_scan",
            })
        explanation["traces"] = {
            "steps": trace_steps,
            "count": len(trace_steps),
        }

        # Build summary
        risk_level = "unknown"
        if explanation["risk_level"]:
            mean = explanation["risk_level"]["mean"]
            if mean > 0.7:
                risk_level = "high"
            elif mean > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

        num_facts = len(explanation["facts"])
        num_suggestions = len(explanation["suggestions"])
        explanation["summary"] = (
            f"File {file_id_clean} has {risk_level} risk "
            f"({num_facts} facts, {num_suggestions} suggestions)"
        )

        return explanation

    def query_file_risk(
        self,
        file_id: str,
        use_attention: bool = True,
        use_importance: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Query risk assessment for a specific file.

        Args:
            file_id: The file identifier
            use_attention: Whether to include attention/importance data
            use_importance: Alias for use_attention (for backwards compatibility)

        Returns dict with risk score, evidence, and importance.
        """
        # Normalize file_id the same way as assert_file_facts
        file_id_clean = Path(file_id).name.replace(".", "_")
        tv = self.query_risk(file_id_clean, aggregate=True)
        if tv is None:
            return None

        result = {
            "file_id": file_id,
            "risk_score": tv.mean(),
            "strength": tv.strength,
            "confidence": tv.confidence,
        }

        # Include importance info only if use_attention and use_importance are True
        if use_attention and use_importance:
            av = self.file_importance.get(file_id_clean, AttentionValue())
            importance_total = av.sti + av.lti
            if av.vlti:
                importance_total += 0.5

            result["_importance"] = {
                "sti": av.sti,
                "lti": av.lti,
                "vlti": av.vlti,
                "total": importance_total,
            }

        return result

    def get_priority_files(
        self,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get files prioritized by risk and importance.

        Returns list of (file_id, priority_score) tuples.
        """
        priorities = []

        # Gather all known files from attention and facts
        known_files = set(self.file_importance.keys())
        # Use public API to get atom names
        for atom in self.pln.graph.get_atom_names():
            # Extract file IDs from atoms like "has_pattern(file_id, pattern)"
            if "(" in atom and ")" in atom:
                parts = atom.split("(")[1].split(")")[0].split(",")
                if parts:
                    known_files.add(parts[0].strip())

        for file_id in known_files:
            # Calculate priority from risk + importance
            risk_tv = self.query_risk(file_id, aggregate=True)
            risk_score = risk_tv.mean() if risk_tv else 0.0

            importance = self.file_importance.get(file_id, AttentionValue())
            importance_score = importance.sti * 0.3 + importance.lti * 0.5

            # VLTI files get priority boost
            if importance.vlti:
                importance_score += 0.3

            priority = risk_score * 0.6 + importance_score * 0.4
            if priority > 0:
                priorities.append((file_id, priority))

        # Sort by priority descending
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities[:top_n]

    def stimulate_file(
        self,
        file_id: str,
        amount: float = 0.1
    ) -> None:
        """
        Increase the short-term importance of a file.

        This simulates a file being accessed or modified.
        """
        # Normalize file_id the same way as assert_file_facts
        file_id_clean = Path(file_id).name.replace(".", "_")
        if file_id_clean in self.file_importance:
            current = self.file_importance[file_id_clean]
            new_sti = min(1.0, current.sti + amount)
            self.file_importance[file_id_clean] = AttentionValue(
                sti=new_sti,
                lti=current.lti,
                vlti=current.vlti
            )
        else:
            self.file_importance[file_id_clean] = AttentionValue(
                sti=amount,
                lti=0.0,
                vlti=False
            )

    def query_with_aggregation(
        self,
        query: str,
        strategies: List[str]
    ) -> Dict[str, Optional[TruthValue]]:
        """
        Query using multiple aggregation strategies and return results for each.

        Args:
            query: The PLN query string to execute
            strategies: List of aggregation strategy names to use

        Returns:
            Dict mapping strategy name to TruthValue result (or None if no result)
        """
        results = {}
        for strategy in strategies:
            tv = self.pln.query(query)
            if tv:
                results[strategy] = tv
        return results


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_reasoning_report(results: Dict[str, Any], verbose: bool = False) -> str:
    """
    Generate a formatted report from audit analysis results.

    Args:
        results: Dictionary containing:
            - files_analyzed: Number of files analyzed
            - rules_loaded: Number of rules loaded
            - risk_assessments: List of risk assessments per file
            - priority_files: List of (file_id, importance) tuples
            - vlti_files: List of critical file IDs
            - stats: Global statistics
        verbose: Whether to include detailed information

    Returns:
        Formatted markdown report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AUDIT REASONING REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary section
    files_analyzed = results.get("files_analyzed", 0)
    if isinstance(files_analyzed, list):
        files_analyzed = len(files_analyzed)
    lines.append(f"## Summary")
    lines.append(f"- Files analyzed: {files_analyzed}")
    lines.append(f"- Rules loaded: {results.get('rules_loaded', 0)}")
    if results.get("aggregate_strategy"):
        lines.append(f"- Aggregation: {results.get('aggregate_strategy')}")
    lines.append("")

    # Risk assessments
    risk_assessments = results.get("risk_assessments", [])
    if risk_assessments:
        lines.append("## Risk Assessments")
        lines.append("")
        for assessment in risk_assessments:
            file_path = assessment.get("file", "unknown")
            overall_risk = assessment.get("overall_risk", 0)

            # Determine risk level
            if overall_risk >= 0.7:
                level = "HIGH"
            elif overall_risk >= 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"

            lines.append(f"### {file_path}")
            lines.append(f"- Risk: {level} ({overall_risk:.1%})")

            if verbose and assessment.get("details"):
                lines.append("- Details:")
                for key, val in assessment.get("details", {}).items():
                    if isinstance(val, dict):
                        strength = val.get("strength", 0)
                        confidence = val.get("confidence", 0)
                        lines.append(f"  - {key}: strength={strength:.2f}, confidence={confidence:.2f}")

            importance = assessment.get("importance", 0)
            if importance > 0:
                lines.append(f"- Importance: {importance:.2f}")
            lines.append("")

    # Legacy analysis_results format (backwards compatibility)
    for result in results.get("analysis_results", []):
        file_path = result.get("file", "unknown")
        patterns = result.get("patterns", [])
        explanation = result.get("explanation", {})

        lines.append(f"### {file_path}")
        if patterns:
            lines.append(f"- Patterns: {', '.join(patterns)}")
        if explanation:
            risk = explanation.get("risk_level", {})
            if risk:
                lines.append(f"- Risk strength: {risk.get('strength', 0):.2f}")
                lines.append(f"- Confidence: {risk.get('confidence', 0):.2f}")
            suggestions = explanation.get("suggestions", [])
            if suggestions:
                lines.append("- Suggestions:")
                for s in suggestions:
                    lines.append(f"  - {s}")
        lines.append("")

    # Priority files section
    priority_files = results.get("priority_files", [])
    if priority_files:
        lines.append("## Priority Files")
        lines.append("")
        for item in priority_files:
            if isinstance(item, tuple):
                file_id, importance = item
                lines.append(f"- {file_id}: importance={importance:.2f}")
            else:
                lines.append(f"- {item}")
        lines.append("")

    # VLTI (Critical) files section
    vlti_files = results.get("vlti_files", [])
    if vlti_files:
        lines.append("## Critical Files (VLTI)")
        lines.append("")
        for file_id in vlti_files:
            lines.append(f"- {file_id}")
        lines.append("")

    # Stats section (verbose only)
    stats = results.get("stats", {})
    if verbose and stats:
        lines.append("## Statistics")
        lines.append("")
        for key, value in stats.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    return "\n".join(lines)


def analyze_with_reasoning(
    directory: str,
    use_persistence: bool = True,
    no_save: bool = False,
    with_git: bool = False,
    aggregate_strategy: str = "revision",
) -> Dict[str, Any]:
    """
    Analyze a directory with PLN reasoning.

    This is the programmatic API for audit reasoning, combining directory
    analysis with PLN-based risk assessment.

    Args:
        directory: Path to directory to analyze
        use_persistence: Whether to use persistent state
        no_save: If True, don't save state after analysis
        with_git: Include git history analysis
        aggregate_strategy: Strategy for combining evidence ('first', 'max', 'revision')

    Returns:
        Dict containing:
            - files_analyzed: Number of files analyzed
            - rules_loaded: Number of PLN rules loaded
            - risk_assessments: List of per-file risk assessments
            - aggregate_strategy: Strategy used
            - priority_files: High-priority files
            - vlti_files: Critical files marked as VLTI
    """
    import os

    # Initialize reasoner
    if use_persistence:
        reasoner = AuditReasoner()
    else:
        reasoner = AuditReasoner(use_persistence=False)

    reasoner.add_default_rules()
    reasoner.aggregate_strategy = aggregate_strategy

    # Analyze directory
    result = analyze_directory(
        directory=directory,
        with_git=with_git,
        verbose=False,
    )

    if result is None or (hasattr(result, 'findings') and not result.findings):
        print("No analysis results found.")
        return {
            "files_analyzed": 0,
            "rules_loaded": reasoner.pln.rule_count,
            "risk_assessments": [],
            "aggregate_strategy": aggregate_strategy,
            "priority_files": [],
            "vlti_files": [],
        }

    # Get findings from result
    findings = result.findings if hasattr(result, 'findings') else []
    git_analysis = result.git_analysis if hasattr(result, 'git_analysis') else {}
    files_analyzed = result.files_analyzed if hasattr(result, 'files_analyzed') else len(findings)

    # Process findings into file patterns
    file_patterns: Dict[str, Dict[str, List[str]]] = {}
    for finding in findings:
        finding_id = finding.get('id', finding.get('file', ''))
        filepath = finding_id.split(':')[0] if ':' in finding_id else finding_id
        pattern = finding.get('pattern', '')
        if filepath and filepath not in file_patterns:
            file_patterns[filepath] = {'patterns': [], 'traits': []}
        if filepath and pattern:
            file_patterns[filepath]['patterns'].append(pattern)

    # Add traits from git analysis
    if git_analysis:
        high_churn = git_analysis.get('high_churn_files', {})
        for filepath in file_patterns:
            if filepath in high_churn:
                file_patterns[filepath]['traits'].append('high_churn')

    # Assert facts and collect risk assessments
    risk_assessments = []
    for filepath, data in file_patterns.items():
        parts = filepath.split(os.sep)
        dirs = [p for p in parts[:-1] if p and p != '.']

        reasoner.assert_file_facts(
            file_path=filepath,
            patterns=data['patterns'],
            traits=data['traits'],
            directories=dirs,
        )

        # Query risk
        risk_result = reasoner.query_file_risk(filepath)
        if risk_result:
            risk_assessments.append({
                "file": filepath,
                "overall_risk": risk_result["risk_score"],
                "details": {
                    "strength": risk_result["strength"],
                    "confidence": risk_result["confidence"],
                },
                "importance": risk_result.get("_importance", {}).get("total", 0),
            })

    # Sort by risk
    risk_assessments.sort(key=lambda x: -x["overall_risk"])

    # Get priority and VLTI files
    priority_files = reasoner.get_priority_files(top_n=10)
    vlti_files = reasoner.get_vlti_files()

    # Save state unless no_save
    if not no_save and use_persistence:
        reasoner.save_state()

    return {
        "files_analyzed": files_analyzed,
        "rules_loaded": reasoner.pln.rule_count,
        "risk_assessments": risk_assessments,
        "aggregate_strategy": aggregate_strategy,
        "priority_files": priority_files,
        "vlti_files": vlti_files,
    }
