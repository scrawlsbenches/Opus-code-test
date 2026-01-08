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
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.prism_pln import (
    PLNReasoner, TruthValue, AttentionalFocus, AttentionValue,
    Term, TypeRegistry, aggregate_truth_values, AggregateStrategy,
    InferenceStep, InferenceTrace
)
from cortical.reasoning.woven_mind import WovenMind

# Import from our other audit tools
from scripts.codebase_health import analyze_directory

# =============================================================================
# STATE FILES
# =============================================================================

RULES_FILE = Path(__file__).parent.parent / ".got" / "audit_pln_rules.json"
WOVEN_MIND_FILE = Path(__file__).parent.parent / ".got" / "woven_audit_mind.json"
PERSISTENCE_FILE = Path(__file__).parent.parent / ".got" / "audit_pln_state.json"


# =============================================================================
# PERSISTENCE LAYER
# =============================================================================

@dataclass
class FileImportanceRecord:
    """Persistent record of a file's importance over time."""
    file_id: str
    sti: float
    lti: float
    vlti: bool
    last_seen: str  # ISO timestamp
    history: List[Dict[str, Any]]  # Historical snapshots

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "sti": self.sti,
            "lti": self.lti,
            "vlti": self.vlti,
            "last_seen": self.last_seen,
            "history": self.history[-50:],  # Keep last 50 snapshots
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileImportanceRecord":
        return cls(
            file_id=data["file_id"],
            sti=data.get("sti", 0.3),
            lti=data.get("lti", 0.1),
            vlti=data.get("vlti", False),
            last_seen=data.get("last_seen", datetime.now().isoformat()),
            history=data.get("history", []),
        )


@dataclass
class AuditPersistenceState:
    """Complete persistent state for audit reasoning."""
    version: int
    created: str
    updated: str
    session_count: int
    file_importance: Dict[str, FileImportanceRecord]
    attention_focus: List[str]  # Files currently in focus
    global_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created": self.created,
            "updated": self.updated,
            "session_count": self.session_count,
            "file_importance": {
                k: v.to_dict() for k, v in self.file_importance.items()
            },
            "attention_focus": self.attention_focus,
            "global_stats": self.global_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditPersistenceState":
        file_importance = {}
        for k, v in data.get("file_importance", {}).items():
            file_importance[k] = FileImportanceRecord.from_dict(v)

        return cls(
            version=data.get("version", 1),
            created=data.get("created", datetime.now().isoformat()),
            updated=data.get("updated", datetime.now().isoformat()),
            session_count=data.get("session_count", 0),
            file_importance=file_importance,
            attention_focus=data.get("attention_focus", []),
            global_stats=data.get("global_stats", {}),
        )

    @classmethod
    def create_new(cls) -> "AuditPersistenceState":
        now = datetime.now().isoformat()
        return cls(
            version=1,
            created=now,
            updated=now,
            session_count=0,
            file_importance={},
            attention_focus=[],
            global_stats={},
        )


# =============================================================================
# NATURAL LANGUAGE QUERY SUPPORT
# =============================================================================

@dataclass
class AuditQuery:
    """Structured representation of an audit query.

    Supports natural language queries like:
        "risky files in reasoning/ not tests"
        "why is prism_pln.py flagged"
        "files with high_churn"
    """
    # Scope
    directory: Optional[str] = None
    file_patterns: List[str] = None

    # Filters
    negations: List[str] = None  # Exclude these
    include_traits: List[str] = None  # Must have these traits

    # Intent
    intent: str = "list"  # list, explain, trace
    target_file: Optional[str] = None  # For "why is X flagged"

    # Thresholds
    min_risk: float = 0.0
    max_results: Optional[int] = None

    # Output
    explain: bool = False

    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = []
        if self.negations is None:
            self.negations = []
        if self.include_traits is None:
            self.include_traits = []


def translate_audit_query(query: str) -> AuditQuery:
    """
    Translate natural language to AuditQuery.

    Pattern matching approach (no ML required).

    Examples:
        "risky files in reasoning/"
        → AuditQuery(directory="reasoning/", min_risk=0.5)

        "why is prism_pln.py flagged"
        → AuditQuery(intent="explain", target_file="prism_pln.py")

        "files not tests with high churn"
        → AuditQuery(negations=["tests"], include_traits=["high_churn"])

        "cortical/ not tests"
        → AuditQuery(directory="cortical/", negations=["tests"])
    """
    import re

    # Initialize result
    result = AuditQuery()

    # Normalize input
    query = query.strip()
    query_lower = query.lower()
    original_query = query

    # =========================================================================
    # Intent Detection (check first - changes how we parse the rest)
    # =========================================================================

    # "why is <file> flagged" or "explain <file>"
    # NOTE: Don't return early - let scope extraction run too
    why_match = re.search(r'why\s+is\s+(\S+)\s+(?:flagged|risky|marked)', query_lower)
    if why_match:
        result.intent = "explain"
        result.target_file = why_match.group(1)
        result.explain = True

    explain_match = re.search(r'explain\s+(\S+)', query_lower)
    if explain_match and result.intent != "explain":  # Don't override why_match
        result.intent = "explain"
        result.target_file = explain_match.group(1)
        result.explain = True

    # =========================================================================
    # Scope Extraction
    # =========================================================================

    # Look for directory patterns: "in <dir>", "<dir>/", or just a path-like string
    dir_patterns = [
        r'in\s+(\S+/)',           # "in cortical/"
        r'in\s+(\S+)',            # "in cortical"
        r'^(\S+/)\s',             # "cortical/ not tests" (directory at start)
        r'^(\S+/?)$',             # Just a directory
    ]

    for pattern in dir_patterns:
        match = re.search(pattern, query_lower)
        if match:
            potential_dir = match.group(1)
            # Verify it looks like a directory (contains / or exists)
            if '/' in potential_dir or Path(potential_dir).exists():
                result.directory = potential_dir.rstrip('/') + '/'
                break

    # =========================================================================
    # Negation Extraction ("not X", "without X", "exclude X")
    # =========================================================================

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

    # =========================================================================
    # Trait/Filter Extraction ("with X", "has X", "having X")
    # =========================================================================

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
            # Normalize common variations
            trait_map = {
                'high_churn': 'high_churn',
                'highchurn': 'high_churn',
                'todos': 'todo',
                'hacks': 'hack',
            }
            trait = trait_map.get(trait, trait)
            if trait not in result.include_traits:
                result.include_traits.append(trait)

    # =========================================================================
    # Risk Level Extraction
    # =========================================================================

    if 'critical' in query_lower:
        result.min_risk = 0.9
    elif 'high risk' in query_lower or 'high-risk' in query_lower:
        result.min_risk = 0.7
    elif 'risky' in query_lower:
        result.min_risk = 0.5
    elif 'medium risk' in query_lower:
        result.min_risk = 0.4

    # =========================================================================
    # Result Limit Extraction ("top N", "first N")
    # =========================================================================

    limit_match = re.search(r'(?:top|first)\s+(\d+)', query_lower)
    if limit_match:
        result.max_results = int(limit_match.group(1))

    # =========================================================================
    # Fallback: If no directory found and query looks like a path
    # =========================================================================

    if result.directory is None:
        # Check if the first word looks like a directory
        first_word = query.split()[0] if query.split() else ""
        if '/' in first_word or Path(first_word).is_dir():
            result.directory = first_word.rstrip('/') + '/'

    return result


def is_natural_language_query(arg: str) -> bool:
    """Determine if input is a natural language query vs a path/flag."""
    # If it starts with --, it's a flag
    if arg.startswith('--') or arg.startswith('-'):
        return False

    # If it's an existing path with no spaces, treat as traditional
    if Path(arg).exists() and ' ' not in arg:
        # But only if it doesn't contain NLU keywords
        nlu_keywords = ['not ', 'with ', 'explain', 'why ', 'risky', 'top ']
        arg_lower = arg.lower()
        if not any(kw in arg_lower for kw in nlu_keywords):
            return False

    # If it contains spaces or NLU keywords, it's natural language
    if ' ' in arg:
        return True

    # If it contains NLU keywords, it's natural language
    nlu_keywords = ['not', 'with', 'explain', 'why', 'risky', 'top', 'critical']
    arg_lower = arg.lower()
    for kw in nlu_keywords:
        if kw in arg_lower and kw != arg_lower:  # keyword is part of query
            return True

    return False


def load_persistence_state() -> AuditPersistenceState:
    """Load persisted audit state from disk."""
    if PERSISTENCE_FILE.exists():
        try:
            with open(PERSISTENCE_FILE, 'r') as f:
                data = json.load(f)
                return AuditPersistenceState.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Warning: Could not load persistence state: {e}")
    return AuditPersistenceState.create_new()


def save_persistence_state(state: AuditPersistenceState) -> None:
    """Save audit state to disk."""
    PERSISTENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.updated = datetime.now().isoformat()
    with open(PERSISTENCE_FILE, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)


def show_persistence_status() -> None:
    """Display current persistence state."""
    state = load_persistence_state()

    print("=" * 60)
    print("  AUDIT PLN PERSISTENCE STATE")
    print("=" * 60)
    print(f"\n[State Info]")
    print(f"  Version: {state.version}")
    print(f"  Created: {state.created}")
    print(f"  Last updated: {state.updated}")
    print(f"  Session count: {state.session_count}")
    print(f"  Files tracked: {len(state.file_importance)}")
    print(f"  Files in focus: {len(state.attention_focus)}")

    if state.file_importance:
        print(f"\n[Top Files by Importance]")
        sorted_files = sorted(
            state.file_importance.values(),
            key=lambda x: x.sti + x.lti,
            reverse=True
        )
        for rec in sorted_files[:10]:
            total = rec.sti + rec.lti
            vlti_marker = " [VLTI]" if rec.vlti else ""
            history_len = len(rec.history)
            print(f"  {rec.file_id}: {total:.2%} (STI={rec.sti:.2f}, LTI={rec.lti:.2f}){vlti_marker}")
            print(f"    Last seen: {rec.last_seen}, History: {history_len} snapshots")

    if state.attention_focus:
        print(f"\n[Current Attention Focus]")
        for file_id in state.attention_focus[:10]:
            print(f"  • {file_id}")

    global_stats = state.global_stats
    if global_stats:
        print(f"\n[Global Statistics]")
        for key, value in global_stats.items():
            print(f"  {key}: {value}")

    print()


def clear_persistence_state() -> None:
    """Clear all persisted state (start fresh)."""
    if PERSISTENCE_FILE.exists():
        PERSISTENCE_FILE.unlink()
        print("Persistence state cleared.")
    else:
        print("No persistence state to clear.")


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
    - Persistence across sessions for importance tracking
    """

    def __init__(
        self,
        aggregate_strategy: AggregateStrategy = "revision",
        use_persistence: bool = True,
        apply_decay: bool = True
    ):
        self.pln = PLNReasoner()
        self.rules_config = load_rules()
        self.aggregate_strategy = aggregate_strategy
        self.use_persistence = use_persistence

        # Attention tracking for files
        self.attention_focus = AttentionalFocus(max_size=50, default_boost=1.5)

        # Type registry for compound term constraints
        self.type_registry = TypeRegistry()
        self._setup_type_registry()

        # Track file importance
        self.file_importance: Dict[str, AttentionValue] = {}

        # Load persisted state if available
        self._persistence_state: Optional[AuditPersistenceState] = None
        if use_persistence:
            self._load_from_persistence(apply_decay=apply_decay)

    def _load_from_persistence(self, apply_decay: bool = True) -> None:
        """Load importance values from persisted state."""
        self._persistence_state = load_persistence_state()

        # Restore importance values
        for file_id, record in self._persistence_state.file_importance.items():
            sti = record.sti
            lti = record.lti

            # Apply decay based on time since last seen
            if apply_decay:
                try:
                    last_seen = datetime.fromisoformat(record.last_seen)
                    hours_elapsed = (datetime.now() - last_seen).total_seconds() / 3600
                    # Decay STI by ~10% per hour, LTI by ~1% per hour
                    if hours_elapsed > 0:
                        sti_decay = 0.9 ** min(hours_elapsed, 24)  # Cap at 24 hours
                        lti_decay = 0.99 ** min(hours_elapsed, 168)  # Cap at 1 week
                        sti = sti * sti_decay
                        lti = lti * lti_decay
                except (ValueError, TypeError):
                    pass

            self.file_importance[file_id] = AttentionValue(
                sti=sti,
                lti=lti,
                vlti=record.vlti
            )
            # Also set in PLN reasoner
            self.pln.set_attention(file_id, self.file_importance[file_id])

        # Restore attention focus
        if self._persistence_state.attention_focus:
            self.attention_focus.focus_on(
                self._persistence_state.attention_focus,
                boost=1.5
            )

    def save_state(self) -> None:
        """Save current state to persistence."""
        if not self.use_persistence:
            return

        if self._persistence_state is None:
            self._persistence_state = AuditPersistenceState.create_new()

        now = datetime.now().isoformat()
        self._persistence_state.session_count += 1

        # Update file importance records
        for file_id, attention in self.file_importance.items():
            if file_id in self._persistence_state.file_importance:
                record = self._persistence_state.file_importance[file_id]
                # Add to history
                record.history.append({
                    "timestamp": now,
                    "sti": record.sti,
                    "lti": record.lti,
                    "vlti": record.vlti,
                })
                # Update current values
                record.sti = attention.sti
                record.lti = attention.lti
                record.vlti = attention.vlti
                record.last_seen = now
            else:
                # New file
                self._persistence_state.file_importance[file_id] = FileImportanceRecord(
                    file_id=file_id,
                    sti=attention.sti,
                    lti=attention.lti,
                    vlti=attention.vlti,
                    last_seen=now,
                    history=[],
                )

        # Save attention focus
        self._persistence_state.attention_focus = list(self.attention_focus._focused.keys())

        # Update global stats
        self._persistence_state.global_stats = {
            "last_aggregate_strategy": self.aggregate_strategy,
            "files_in_focus": len(self.attention_focus._focused),
            "total_files_tracked": len(self.file_importance),
            "vlti_files": len(self.get_vlti_files()),
        }

        save_persistence_state(self._persistence_state)

    def get_importance_history(self, file_id: str) -> List[Dict[str, Any]]:
        """Get the importance history for a specific file."""
        if self._persistence_state and file_id in self._persistence_state.file_importance:
            return self._persistence_state.file_importance[file_id].history
        return []

    def get_importance_trend(self, file_id: str) -> Optional[str]:
        """Determine if importance is trending up, down, or stable."""
        history = self.get_importance_history(file_id)
        if len(history) < 2:
            return None

        recent = history[-1]["sti"] + history[-1]["lti"]
        older = history[0]["sti"] + history[0]["lti"]

        diff = recent - older
        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"

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

    def _generate_suggestions(
        self,
        facts: Dict[str, Any],
        traces: Dict[str, "InferenceTrace"]
    ) -> List[str]:
        """
        Generate actionable suggestions based on detected patterns.

        Maps inference results to concrete actions the developer can take.
        """
        suggestions = []
        seen = set()

        # Pattern-to-suggestion mapping
        suggestion_map = {
            # Pattern markers
            "todo": "Review and resolve TODO comments",
            "fixme": "Address FIXME items - these indicate known bugs",
            "hack": "Refactor HACK workarounds into proper solutions",
            "future": "Plan implementation of FUTURE items or remove if obsolete",
            "should_be": "Investigate 'should be' comments - may indicate spec deviations",
            "will_be": "Verify 'will be' items are tracked or implemented",
            "see_docs": "Ensure referenced documentation is up to date",

            # Inferred states
            "incomplete": "Complete unfinished implementations",
            "needs_review": "Schedule code review for this file",
            "needs_urgent_review": "Prioritize review - multiple risk signals",
            "has_known_issue": "Triage known issues before adding features",
            "technical_debt": "Schedule refactoring to reduce technical debt",
            "risky": "Add tests before modifying this file",
            "critical_review": "Block merges until critical issues resolved",

            # Git-based
            "high_churn": "Consider splitting into smaller modules",
            "bug_prone": "Add regression tests for frequently-fixed areas",
        }

        # Check facts for pattern markers
        for fact_name in facts.keys():
            fact_lower = fact_name.lower()
            for pattern, suggestion in suggestion_map.items():
                if pattern in fact_lower and suggestion not in seen:
                    suggestions.append(suggestion)
                    seen.add(suggestion)

        # Check traces for inferred states
        for trace_name, trace in traces.items():
            if trace.final_result and trace.final_result.strength > 0.4:
                trace_lower = trace_name.lower()
                for pattern, suggestion in suggestion_map.items():
                    if pattern in trace_lower and suggestion not in seen:
                        suggestions.append(suggestion)
                        seen.add(suggestion)

        return suggestions[:5]  # Limit to top 5 most relevant

    def explain_file_risk(
        self,
        file_path: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Explain why a file is flagged with REAL PLN inference chains.

        This is Phase 2 explainability - shows actual rules that fired,
        not templated responses.

        Args:
            file_path: Path to the file to explain
            verbose: Include additional detail

        Returns:
            Dict with:
                - file: File path
                - file_id: Normalized file ID
                - facts: Facts asserted about this file
                - traces: Dict of query -> InferenceTrace
                - summary: Human-readable summary
                - raw_traces: Raw trace data for programmatic access
        """
        file_id = Path(file_path).name.replace(".", "_")

        # Collect facts asserted about this file
        file_facts = {}
        for atom_name, atom in self.pln.graph._atoms.items():
            # Check if this fact is about our file
            if f"({file_id}" in atom_name or f", {file_id})" in atom_name:
                file_facts[atom_name] = {
                    "strength": atom.truth_value.strength,
                    "confidence": atom.truth_value.confidence,
                }

        # Run traced inference for key risk queries
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

        traces = {}
        raw_traces = {}

        for name, query in queries:
            trace = self.pln.query_with_trace(
                query,
                max_depth=5,
                aggregate=self.aggregate_strategy
            )
            if trace.final_result is not None:
                traces[name] = trace
                raw_traces[name] = trace.to_dict()

        # Build human-readable summary
        summary_lines = []
        summary_lines.append(f"=== Explanation for: {file_path} ===")
        summary_lines.append(f"File ID: {file_id}")
        summary_lines.append("")

        # Facts section
        summary_lines.append("FACTS ASSERTED:")
        if file_facts:
            for fact, tv in file_facts.items():
                summary_lines.append(
                    f"  • {fact}: {tv['strength']:.0%} "
                    f"(confidence: {tv['confidence']:.0%})"
                )
        else:
            summary_lines.append("  (no facts found for this file)")
        summary_lines.append("")

        # Inference traces section
        summary_lines.append("INFERENCE CHAINS:")
        if traces:
            for name, trace in traces.items():
                if trace.final_result:
                    prob = trace.final_result.to_probability()
                    summary_lines.append(f"\n  [{name}] → {prob:.0%}")

                    # Show the inference chain
                    if trace.steps:
                        for step in trace.steps:
                            rule_str = (
                                f"{step.rule_antecedent} → {step.rule_consequent}"
                            )
                            subs = ", ".join(
                                f"{k}={v}" for k, v in step.substitutions.items()
                            )
                            if subs:
                                rule_str += f" [{subs}]"

                            summary_lines.append(f"    Rule: {rule_str}")
                            summary_lines.append(
                                f"      Rule strength: {step.rule_truth_value.strength:.0%}"
                            )
                            summary_lines.append(
                                f"      Antecedent: {step.antecedent_truth_value.strength:.0%}"
                            )
                            summary_lines.append(
                                f"      → Inferred: {step.result_truth_value.strength:.0%}"
                            )

                    # Show aggregation if multiple paths
                    if len(trace.aggregation_inputs) > 1:
                        summary_lines.append(
                            f"    Aggregation ({trace.aggregation_strategy}):"
                        )
                        for i, tv in enumerate(trace.aggregation_inputs, 1):
                            summary_lines.append(
                                f"      Path {i}: {tv.strength:.0%}"
                            )
        else:
            summary_lines.append("  (no inferences triggered for this file)")

        # Generate suggested actions based on what triggered
        suggestions = self._generate_suggestions(file_facts, traces)
        if suggestions:
            summary_lines.append("")
            summary_lines.append("SUGGESTED ACTIONS:")
            for suggestion in suggestions:
                summary_lines.append(f"  → {suggestion}")

        summary_lines.append("")
        summary_lines.append("=" * 50)

        # Build result
        result = {
            "file": file_path,
            "file_id": file_id,
            "facts": file_facts,
            "traces": {name: trace.explain() for name, trace in traces.items()},
            "summary": "\n".join(summary_lines),
            "raw_traces": raw_traces,
        }

        # Add importance info if tracked
        if file_id in self.file_importance:
            attention = self.file_importance[file_id]
            result["importance"] = {
                "sti": attention.sti,
                "lti": attention.lti,
                "vlti": attention.vlti,
                "total": attention.total_importance(),
            }

        return result

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
    enable_importance: bool = True,
    use_persistence: bool = True,
    no_save: bool = False
) -> Dict[str, Any]:
    """
    Full analysis pipeline with Full PLN features: Discover → Assert → Reason.

    Features:
    - Multi-rule aggregation for combining evidence from multiple inference paths
    - Attention-based focus for prioritized inference
    - Importance weights (STI/LTI) for file prioritization
    - Compound terms for complex multi-signal rules
    - Persistence across sessions for tracking importance over time
    """
    results = {
        "files_analyzed": 0,
        "rules_loaded": 0,
        "risk_assessments": [],
        "priority_files": [],
        "vlti_files": [],
        "aggregate_strategy": aggregate_strategy,
        "persistence_enabled": use_persistence,
    }

    # Step 1: Run codebase analysis
    print("[1/6] Running codebase analysis...")
    analysis = analyze_directory(directory, with_git=with_git)

    if not analysis:
        print("No analysis results")
        return results

    findings = analysis.get("findings", [])
    git_analysis = analysis.get("git_analysis", {})
    print(f"      Found {len(findings)} findings")

    # Step 2: Initialize reasoner with Full PLN features and persistence
    print("\n[2/6] Initializing Full PLN reasoner...")
    reasoner = AuditReasoner(
        aggregate_strategy=aggregate_strategy,
        use_persistence=use_persistence,
        apply_decay=True
    )

    # Show persistence status
    if use_persistence and reasoner._persistence_state:
        ps = reasoner._persistence_state
        if ps.session_count > 0:
            print(f"      Loaded state from session #{ps.session_count}")
            print(f"      Restored importance for {len(ps.file_importance)} files")
            if ps.attention_focus:
                print(f"      Restored attention focus on {len(ps.attention_focus)} files")
        else:
            print("      Starting fresh (no previous state)")

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
    print("\n[3/6] Asserting facts with importance tracking...")

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
        print("\n[4/6] Focusing attention on high-risk files...")
        focused_count = reasoner.focus_on_high_risk_files(threshold=0.5)
        print(f"      Focused on {focused_count} high-importance files")
    else:
        print("\n[4/6] Attention focusing disabled")

    # Step 5: Query risk for each file with Full PLN inference
    print("\n[5/6] Running Full PLN inference...")

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

    # Step 6: Save state for persistence
    if use_persistence and not no_save:
        print("\n[6/6] Saving state for next session...")
        reasoner.save_state()
        if reasoner._persistence_state:
            print(f"      State saved (session #{reasoner._persistence_state.session_count})")
            print(f"      Tracking {len(reasoner.file_importance)} files with history")
    else:
        print("\n[6/6] Persistence disabled or no-save mode")

    # Store final stats
    results["stats"] = reasoner.get_stats()

    # Add persistence info to results
    if use_persistence and reasoner._persistence_state:
        results["persistence"] = {
            "session_count": reasoner._persistence_state.session_count,
            "files_tracked": len(reasoner._persistence_state.file_importance),
            "created": reasoner._persistence_state.created,
            "updated": reasoner._persistence_state.updated,
        }

    # Include the reasoner for explain functionality (Phase 2)
    results["_reasoner"] = reasoner

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

    # Persistence info
    persistence = results.get("persistence", {})
    if persistence:
        lines.append(f"\n[Persistence]")
        lines.append(f"  • Session #{persistence.get('session_count', 0)}")
        lines.append(f"  • Tracking {persistence.get('files_tracked', 0)} files with history")
        lines.append(f"  • First tracked: {persistence.get('created', 'N/A')[:19]}")
        lines.append(f"  • Last updated: {persistence.get('updated', 'N/A')[:19]}")
        lines.append(f"  • View state: python scripts/audit_reasoning.py --show-state")
        lines.append(f"  • View file history: --file-history <filename>")

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
        epilog="""
Natural Language Queries (Phase 1):
  You can use natural language instead of flags:

  Examples:
    %(prog)s "cortical/ not tests"           # Analyze cortical/, exclude test files
    %(prog)s "risky files in reasoning/"     # Find risky files in reasoning/
    %(prog)s "files with high_churn"         # Files with high churn trait
    %(prog)s "why is prism_pln.py flagged"   # Explain why a file is flagged
    %(prog)s "top 10 risky"                  # Top 10 risky files

  Supported patterns:
    - "not <term>" / "without <term>"        Exclude files matching term
    - "with <trait>" / "has <trait>"         Include files with trait
    - "risky" / "high risk" / "critical"     Set minimum risk threshold
    - "top N" / "first N"                    Limit results
    - "why is <file> flagged"                Explain a file's risk
    - "in <dir>" / "<dir>/"                  Set directory scope
""",
    )
    parser.add_argument("directory", nargs="?", default="cortical/",
                        help="Directory to analyze (or natural language query in quotes)")
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

    # Persistence options
    parser.add_argument("--show-state", action="store_true",
                        help="Show persistence state and exit")
    parser.add_argument("--clear-state", action="store_true",
                        help="Clear all persistence state and start fresh")
    parser.add_argument("--no-persist", action="store_true",
                        help="Disable persistence for this run")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save state after analysis (read-only mode)")
    parser.add_argument("--file-history", metavar="FILE",
                        help="Show importance history for a specific file")

    args = parser.parse_args()

    # Handle persistence commands first
    if args.show_state:
        show_persistence_status()
        return

    if args.clear_state:
        clear_persistence_state()
        return

    if args.file_history:
        state = load_persistence_state()
        file_id = Path(args.file_history).name.replace(".", "_")
        if file_id in state.file_importance:
            record = state.file_importance[file_id]
            print(f"\n[Importance History for {file_id}]")
            print(f"  Current: STI={record.sti:.3f}, LTI={record.lti:.3f}, VLTI={record.vlti}")
            print(f"  Last seen: {record.last_seen}")
            print(f"\n  History ({len(record.history)} snapshots):")
            for entry in record.history[-10:]:
                ts = entry.get("timestamp", "?")[:19]
                sti = entry.get("sti", 0)
                lti = entry.get("lti", 0)
                total = sti + lti
                print(f"    {ts}: STI={sti:.3f}, LTI={lti:.3f} (total={total:.3f})")
        else:
            print(f"No history found for file: {file_id}")
            print(f"Available files: {list(state.file_importance.keys())[:10]}...")
        return

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

    # =========================================================================
    # Natural Language Query Detection and Translation
    # =========================================================================

    directory = args.directory
    nlu_query = None
    audit_query = None

    # Check if input is a natural language query
    if is_natural_language_query(args.directory):
        nlu_query = args.directory
        audit_query = translate_audit_query(nlu_query)

        print("=" * 70)
        print("  Audit Reasoning - Natural Language Query")
        print("=" * 70)
        print()
        print(f"  Query: \"{nlu_query}\"")
        print(f"  Parsed:")
        if audit_query.directory:
            print(f"    • Directory: {audit_query.directory}")
        if audit_query.negations:
            print(f"    • Exclude: {', '.join(audit_query.negations)}")
        if audit_query.include_traits:
            print(f"    • Traits: {', '.join(audit_query.include_traits)}")
        if audit_query.min_risk > 0:
            print(f"    • Min risk: {audit_query.min_risk:.0%}")
        if audit_query.max_results:
            print(f"    • Limit: {audit_query.max_results}")
        if audit_query.intent == "explain":
            print(f"    • Intent: explain file '{audit_query.target_file}'")
        print()

        # Use the parsed directory or default
        directory = audit_query.directory or "cortical/"

        # Handle "explain" intent - Phase 2: Real Explainability
        if audit_query.intent == "explain" and audit_query.target_file:
            print(f"[Explaining: {audit_query.target_file}]")
            print()
    else:
        print("=" * 70)
        print("  Audit Reasoning - Full PLN Pipeline")
        print("=" * 70)
        print()

    # Run analysis
    results = analyze_with_reasoning(
        directory,
        with_git=args.with_git,
        verbose=args.verbose,
        aggregate_strategy=args.aggregate,
        enable_attention=not args.no_attention,
        enable_importance=not args.no_importance,
        use_persistence=not args.no_persist,
        no_save=args.no_save
    )

    # =========================================================================
    # Apply NLU Filters (if natural language query)
    # =========================================================================

    if audit_query and results.get("risk_assessments"):
        assessments = results["risk_assessments"]
        original_count = len(assessments)

        # Filter by negations (exclude files matching these terms)
        if audit_query.negations:
            for negation in audit_query.negations:
                neg_lower = negation.lower()
                assessments = [
                    a for a in assessments
                    if neg_lower not in a["file"].lower()
                ]

        # Filter by minimum risk
        if audit_query.min_risk > 0:
            assessments = [
                a for a in assessments
                if a.get("overall_risk", 0) >= audit_query.min_risk
            ]

        # Filter by include traits
        if audit_query.include_traits:
            filtered_assessments = []
            for a in assessments:
                file_traits = set()
                # Collect traits from file details
                details = a.get("details", {})
                for key, val in details.items():
                    if isinstance(val, dict) and val.get("probability", 0) > 0.5:
                        # Extract trait name from key (e.g., "risky" from "risky(file)")
                        trait = key.split("(")[0].lower()
                        file_traits.add(trait)
                    if key.lower().startswith("has_trait"):
                        trait = key.split(",")[-1].rstrip(")").strip().lower()
                        file_traits.add(trait)
                # Check file path for traits too
                if "churn" in a["file"].lower():
                    file_traits.add("high_churn")
                # Check if file has any required traits
                if any(t in str(details).lower() for t in audit_query.include_traits):
                    filtered_assessments.append(a)
            assessments = filtered_assessments

        # Filter for explain intent (single file)
        if audit_query.intent == "explain" and audit_query.target_file:
            target = audit_query.target_file.lower()
            assessments = [
                a for a in assessments
                if target in a["file"].lower()
            ]

        # Apply max results limit (already sorted by risk in analyze_with_reasoning)
        if audit_query.max_results and len(assessments) > audit_query.max_results:
            assessments = assessments[:audit_query.max_results]

        # Show filter summary
        filtered_count = len(assessments)
        if original_count != filtered_count:
            print(f"[Filtered: {original_count} → {filtered_count} files]")
            print()

        # Update results with filtered assessments
        results["risk_assessments"] = assessments

    # =========================================================================
    # Handle Explain Intent (Phase 2: Real Explainability)
    # =========================================================================

    if audit_query and audit_query.intent == "explain" and audit_query.target_file:
        reasoner = results.get("_reasoner")
        if reasoner:
            # Find the matching file path from assessments
            target = audit_query.target_file.lower()
            matching_files = [
                a["file"] for a in results.get("risk_assessments", [])
                if target in a["file"].lower()
            ]

            if matching_files:
                file_path = matching_files[0]
                print()
                print("=" * 70)
                print("  REAL PLN INFERENCE EXPLANATION")
                print("  (Not templated - actual rules that fired)")
                print("=" * 70)
                print()

                explanation = reasoner.explain_file_risk(file_path, verbose=args.verbose)
                print(explanation["summary"])
                print()

                # Show detailed traces if verbose
                if args.verbose:
                    print("\n[Detailed Inference Traces]")
                    for name, trace_str in explanation.get("traces", {}).items():
                        print(f"\n--- {name} ---")
                        print(trace_str)
            else:
                print(f"\nNo matching file found for: {audit_query.target_file}")
                print("Files analyzed:")
                for a in results.get("risk_assessments", [])[:10]:
                    print(f"  • {a['file']}")
        else:
            print("\nWarning: Reasoner not available for explanation")

        # Don't show the full report for explain intent
        return

    print()
    report = generate_reasoning_report(results, verbose=args.verbose)
    print(report)


if __name__ == "__main__":
    main()
