"""
PRISM-PLN: Probabilistic Logic Networks with Synaptic Learning.

Combines probabilistic reasoning with synaptic plasticity for
uncertain knowledge representation and inference.

Key concepts:
- TruthValue: Probability with confidence (strength, confidence)
- Logical operations: NOT, AND, OR, implication with uncertainty
- Inference rules: Deduction, induction, abduction
- PLNGraph: Knowledge graph with probabilistic links
- SynapticTruthValue: Truth values that learn from evidence

Based on OpenCog PLN theory with PRISM synaptic extensions.

Example:
    from cortical.reasoning.prism_pln import PLNReasoner

    reasoner = PLNReasoner()
    reasoner.assert_fact("bird(tweety)", strength=0.99)
    reasoner.assert_rule("bird(X)", "canfly(X)", strength=0.85)

    result = reasoner.query("canfly(tweety)")
    print(f"Tweety can fly: {result.strength:.2f} (conf: {result.confidence:.2f})")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal, Union
from collections import defaultdict
from functools import reduce
import json
import math
import re


# Aggregation strategies for multi-rule inference
AggregateStrategy = Literal["first", "revision", "max", "or", "weighted"]


# =============================================================================
# TERMS AND UNIFICATION
# =============================================================================

@dataclass
class Term:
    """
    A logical term that can be an atom, variable, or compound term.

    Supports:
    - Atoms: constants like 'auth.py', 'high'
    - Variables: uppercase names like 'X', 'Y' (optionally typed: 'X:File')
    - Compound terms: functor(arg1, arg2, ...) like 'file(auth.py, metrics(high))'
    """
    functor: str = ""
    args: List[Any] = field(default_factory=list)  # List of Term or str
    is_variable: bool = False
    name: str = ""  # For variables
    type_constraint: Optional[str] = None  # For typed variables like X:File

    def __post_init__(self):
        """Normalize term representation."""
        if self.is_variable and not self.name:
            self.name = self.functor
            self.functor = ""

    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            return False
        if self.is_variable and other.is_variable:
            return self.name == other.name
        if self.is_variable or other.is_variable:
            return False  # Variable != non-variable
        if self.functor != other.functor:
            return False
        if len(self.args) != len(other.args):
            return False
        return all(
            _args_equal(a, b) for a, b in zip(self.args, other.args)
        )

    def __hash__(self) -> int:
        if self.is_variable:
            return hash(("var", self.name))
        args_tuple = tuple(
            a if isinstance(a, str) else hash(a) for a in self.args
        )
        return hash((self.functor, args_tuple))

    def __str__(self) -> str:
        if self.is_variable:
            if self.type_constraint:
                return f"{self.name}:{self.type_constraint}"
            return self.name
        if not self.args:
            return self.functor
        args_str = ", ".join(
            str(a) if isinstance(a, Term) else str(a) for a in self.args
        )
        return f"{self.functor}({args_str})"

    def __repr__(self) -> str:
        return f"Term({str(self)})"


def _args_equal(a: Any, b: Any) -> bool:
    """Compare two term arguments for equality."""
    if isinstance(a, Term) and isinstance(b, Term):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, str) and isinstance(b, Term):
        return not b.is_variable and b.functor == a and not b.args
    if isinstance(a, Term) and isinstance(b, str):
        return not a.is_variable and a.functor == b and not a.args
    return False


def parse_term(text: str) -> Term:
    """
    Parse a string into a Term structure.

    Supports:
    - Atoms: 'auth.py', 'high'
    - Variables: 'X', 'Y' (single uppercase letter or starting with uppercase)
    - Typed variables: 'X:File', 'F:Directory'
    - Compound terms: 'file(auth.py)', 'metrics(high, old)'
    - Nested terms: 'issue(file(auth.py), severity(high))'

    Args:
        text: String representation of term

    Returns:
        Parsed Term object
    """
    text = text.strip()

    # Check for typed variable: X:Type
    if ":" in text and "(" not in text:
        parts = text.split(":", 1)
        var_name = parts[0].strip()
        type_name = parts[1].strip()
        return Term(
            is_variable=True,
            name=var_name,
            type_constraint=type_name
        )

    # Check for variable (uppercase single letter or CamelCase starting uppercase)
    if text and text[0].isupper() and "(" not in text:
        # Single uppercase letter or word starting with uppercase = variable
        if len(text) == 1 or (text.isalnum() and text[0].isupper()):
            return Term(is_variable=True, name=text)

    # Check for compound term: functor(args)
    if "(" in text:
        # Find the functor and args
        paren_idx = text.index("(")
        functor = text[:paren_idx].strip()

        # Extract args string (handle nested parens)
        args_str = text[paren_idx + 1:-1]  # Remove outer parens
        args = _parse_args(args_str)

        return Term(functor=functor, args=args)

    # Simple atom
    return Term(functor=text, args=[])


def _parse_args(args_str: str) -> List[Any]:
    """Parse comma-separated arguments, handling nested parens."""
    args = []
    current = ""
    depth = 0

    for char in args_str:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            if current.strip():
                args.append(parse_term(current.strip()))
            current = ""
        else:
            current += char

    if current.strip():
        args.append(parse_term(current.strip()))

    return args


def unify(
    t1: Term,
    t2: Term,
    substitution: Optional[Dict[str, Any]] = None,
    type_registry: Optional["TypeRegistry"] = None
) -> Optional[Dict[str, Any]]:
    """
    Robinson's unification algorithm with occurs check.

    Finds a substitution that makes t1 and t2 identical, or returns None
    if unification fails.

    Args:
        t1: First term
        t2: Second term
        substitution: Optional initial substitution to extend
        type_registry: Optional type registry for type constraints

    Returns:
        Substitution dict mapping variable names to values, or None if fails
    """
    if substitution is None:
        substitution = {}
    else:
        substitution = dict(substitution)  # Copy to avoid mutation

    return _unify_impl(t1, t2, substitution, type_registry)


def _unify_impl(
    t1: Term,
    t2: Term,
    subst: Dict[str, Any],
    registry: Optional["TypeRegistry"]
) -> Optional[Dict[str, Any]]:
    """Internal unification implementation."""

    # Apply current substitution
    t1 = _apply_subst_to_term(t1, subst)
    t2 = _apply_subst_to_term(t2, subst)

    # Both are atoms/constants
    if not t1.is_variable and not t1.args and not t2.is_variable and not t2.args:
        if t1.functor == t2.functor:
            return subst
        return None

    # t1 is variable
    if t1.is_variable:
        return _unify_variable(t1, t2, subst, registry)

    # t2 is variable
    if t2.is_variable:
        return _unify_variable(t2, t1, subst, registry)

    # Both are compound terms
    if t1.functor != t2.functor:
        return None
    if len(t1.args) != len(t2.args):
        return None

    # Unify arguments
    for arg1, arg2 in zip(t1.args, t2.args):
        # Convert string args to terms
        if isinstance(arg1, str):
            arg1 = parse_term(arg1)
        if isinstance(arg2, str):
            arg2 = parse_term(arg2)

        result = _unify_impl(arg1, arg2, subst, registry)
        if result is None:
            return None
        subst = result

    return subst


def _unify_variable(
    var: Term,
    term: Term,
    subst: Dict[str, Any],
    registry: Optional["TypeRegistry"]
) -> Optional[Dict[str, Any]]:
    """Unify a variable with a term."""

    # Check if already bound
    if var.name in subst:
        bound_value = subst[var.name]
        if isinstance(bound_value, str):
            bound_term = parse_term(bound_value)
        elif isinstance(bound_value, Term):
            bound_term = bound_value
        else:
            bound_term = parse_term(str(bound_value))
        return _unify_impl(bound_term, term, subst, registry)

    # Variable unifying with itself
    if term.is_variable and term.name == var.name:
        return subst

    # Occurs check: var cannot appear in term
    if _occurs_in(var.name, term):
        return None

    # Type constraint check
    if var.type_constraint and registry:
        term_value = term.functor if not term.args and not term.is_variable else str(term)
        if not registry.is_type(term_value, var.type_constraint):
            return None

    # Bind variable
    if term.is_variable:
        subst[var.name] = term.name
    elif not term.args:
        subst[var.name] = term.functor
    else:
        subst[var.name] = term

    return subst


def _occurs_in(var_name: str, term: Term) -> bool:
    """Check if variable appears in term (occurs check)."""
    if term.is_variable:
        return term.name == var_name
    for arg in term.args:
        if isinstance(arg, Term):
            if _occurs_in(var_name, arg):
                return True
        elif isinstance(arg, str):
            if arg == var_name and arg[0].isupper():
                return True
    return False


def _apply_subst_to_term(term: Term, subst: Dict[str, Any]) -> Term:
    """Apply substitution to a term."""
    if term.is_variable:
        if term.name in subst:
            value = subst[term.name]
            if isinstance(value, Term):
                return value
            elif isinstance(value, str):
                return parse_term(value)
            else:
                return parse_term(str(value))
        return term

    if not term.args:
        return term

    new_args = []
    for arg in term.args:
        if isinstance(arg, Term):
            new_args.append(_apply_subst_to_term(arg, subst))
        elif isinstance(arg, str) and arg in subst:
            new_args.append(subst[arg])
        else:
            new_args.append(arg)

    return Term(functor=term.functor, args=new_args)


def apply_substitution(term: Term, substitution: Dict[str, Any]) -> Term:
    """
    Apply a substitution to a term, replacing variables with their bindings.

    Args:
        term: Term to apply substitution to
        substitution: Dict mapping variable names to values

    Returns:
        New term with variables replaced
    """
    return _apply_subst_to_term(term, substitution)


# =============================================================================
# TYPE REGISTRY
# =============================================================================

class TypeRegistry:
    """
    Registry for type constraints in unification.

    Supports:
    - Simple types: register_type("File", ["auth.py", "config.py"])
    - Type hierarchies: register_subtype("PythonFile", "File", [...])
    """

    def __init__(self):
        self._types: Dict[str, set] = {}
        self._subtypes: Dict[str, str] = {}  # subtype -> parent type

    def register_type(self, type_name: str, members: List[str]) -> None:
        """Register a type with its members."""
        self._types[type_name] = set(members)

    def register_subtype(
        self,
        subtype_name: str,
        parent_type: str,
        members: List[str]
    ) -> None:
        """Register a subtype of an existing type."""
        self._types[subtype_name] = set(members)
        self._subtypes[subtype_name] = parent_type

    def is_type(self, value: str, type_name: str) -> bool:
        """Check if value is of given type (including subtypes)."""
        # Direct membership
        if type_name in self._types:
            if value in self._types[type_name]:
                return True

        # Check subtypes
        for subtype, parent in self._subtypes.items():
            if parent == type_name:
                if subtype in self._types and value in self._types[subtype]:
                    return True

        return False

    def get_type_members(self, type_name: str) -> set:
        """Get all members of a type."""
        return self._types.get(type_name, set())


@dataclass
class TruthValue:
    """
    Probabilistic truth value with strength and confidence.

    - strength: Probability that the statement is true [0, 1]
    - confidence: How much evidence supports this estimate [0, 1]

    Based on PLN's indefinite probabilities.
    """
    strength: float = 0.5
    confidence: float = 0.0

    def __post_init__(self):
        # Clamp to valid range
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def mean(self, prior: float = 0.5) -> float:
        """
        Compute mean probability accounting for confidence.

        Low confidence → closer to prior
        High confidence → closer to strength
        """
        return self.confidence * self.strength + (1 - self.confidence) * prior

    def revise(self, other: "TruthValue") -> "TruthValue":
        """
        Revise this truth value with new evidence.

        Combines two independent estimates using PLN revision formula.
        """
        # Compute count from confidence (inverse of confidence formula)
        k = 1.0  # Confidence-to-count parameter

        c1 = self.confidence
        c2 = other.confidence

        # Avoid division by zero
        if c1 + c2 - c1 * c2 < 0.001:
            return TruthValue(
                strength=(self.strength + other.strength) / 2,
                confidence=max(c1, c2)
            )

        # Revision formula
        new_strength = (c1 * self.strength + c2 * other.strength - c1 * c2 * self.strength) / \
                       (c1 + c2 - c1 * c2)

        # Combined confidence is higher than either alone
        new_confidence = (c1 + c2 - c1 * c2)

        return TruthValue(strength=new_strength, confidence=new_confidence)

    def to_probability(self, prior: float = 0.5) -> float:
        """Convert to simple probability estimate."""
        return self.mean(prior)

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {"strength": self.strength, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TruthValue":
        """Deserialize from dictionary."""
        return cls(
            strength=data.get("strength", 0.5),
            confidence=data.get("confidence", 0.0)
        )

    def __repr__(self) -> str:
        return f"TV({self.strength:.2f}, {self.confidence:.2f})"


# =============================================================================
# LOGICAL OPERATIONS
# =============================================================================

def pln_not(tv: TruthValue) -> TruthValue:
    """
    Negation: NOT A.

    Inverts strength, preserves confidence.
    """
    return TruthValue(
        strength=1.0 - tv.strength,
        confidence=tv.confidence
    )


def pln_and(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    Conjunction: A AND B.

    Uses independence assumption: P(A ∧ B) = P(A) × P(B)
    """
    new_strength = tv1.strength * tv2.strength

    # Confidence is minimum (weakest link)
    new_confidence = min(tv1.confidence, tv2.confidence)

    return TruthValue(strength=new_strength, confidence=new_confidence)


def pln_or(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """
    Disjunction: A OR B.

    Uses independence assumption: P(A ∨ B) = P(A) + P(B) - P(A)P(B)
    """
    new_strength = tv1.strength + tv2.strength - tv1.strength * tv2.strength

    # Confidence is minimum (weakest link)
    new_confidence = min(tv1.confidence, tv2.confidence)

    return TruthValue(strength=new_strength, confidence=new_confidence)


def pln_implication(tv_antecedent: TruthValue, tv_implication: TruthValue) -> TruthValue:
    """
    Modus ponens with uncertainty: Given A and A→B, infer B.

    P(B) = P(A) × P(B|A) + P(¬A) × P(B|¬A)

    Simplified: assume P(B|¬A) ≈ prior
    """
    prior = 0.5

    # P(B) ≈ P(A) × P(A→B) + P(¬A) × prior
    p_a = tv_antecedent.strength
    p_impl = tv_implication.strength

    new_strength = p_a * p_impl + (1 - p_a) * prior

    # Confidence degrades through inference
    new_confidence = min(tv_antecedent.confidence, tv_implication.confidence) * 0.9

    return TruthValue(strength=new_strength, confidence=new_confidence)


# =============================================================================
# INFERENCE RULES
# =============================================================================

def deduce(tv_ab: TruthValue, tv_bc: TruthValue) -> TruthValue:
    """
    Deduction: A→B, B→C ⊢ A→C.

    PLN deduction formula for chaining implications.
    """
    s_ab = tv_ab.strength
    s_bc = tv_bc.strength

    # Deduction strength formula (simplified)
    # Assumes B probability is uncertain
    s_b = 0.5  # Prior for B

    if s_b < 0.001:
        s_b = 0.001

    s_ac = s_ab * s_bc + (1 - s_ab) * (s_bc - s_ab * s_bc) / (1 - s_ab + 0.001)
    s_ac = max(0.0, min(1.0, s_ac))

    # Confidence decreases through chain
    new_confidence = tv_ab.confidence * tv_bc.confidence * 0.9

    return TruthValue(strength=s_ac, confidence=new_confidence)


def induce(tv_ab: TruthValue, tv_ac: TruthValue) -> TruthValue:
    """
    Induction: A→B, A→C ⊢ B→C (with lower confidence).

    Infers correlation from shared cause.
    """
    s_ab = tv_ab.strength
    s_ac = tv_ac.strength

    # Induction assumes B and C are related through A
    s_bc = s_ab * s_ac + (1 - s_ab) * (1 - s_ac)  # Correlation estimate
    s_bc = max(0.0, min(1.0, s_bc))

    # Much lower confidence than deduction (weaker inference)
    new_confidence = tv_ab.confidence * tv_ac.confidence * 0.5

    return TruthValue(strength=s_bc, confidence=new_confidence)


def abduce(tv_ab: TruthValue, tv_b: TruthValue) -> TruthValue:
    """
    Abduction: A→B, B ⊢ A (with lower confidence).

    Reasoning from effect to cause.
    """
    s_ab = tv_ab.strength  # P(B|A)
    s_b = tv_b.strength     # P(B)

    # Bayes: P(A|B) = P(B|A) × P(A) / P(B)
    # Assume P(A) = 0.5 (prior)
    p_a = 0.5

    if s_b < 0.001:
        s_b = 0.001

    s_a_given_b = (s_ab * p_a) / s_b
    s_a_given_b = max(0.0, min(1.0, s_a_given_b))

    # Abduction has lower confidence (reasoning backwards)
    new_confidence = min(tv_ab.confidence, tv_b.confidence) * 0.6

    return TruthValue(strength=s_a_given_b, confidence=new_confidence)


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_truth_values(
    truth_values: List[TruthValue],
    strategy: AggregateStrategy = "revision"
) -> Optional[TruthValue]:
    """
    Aggregate multiple truth values into one using the specified strategy.

    Args:
        truth_values: List of truth values to aggregate
        strategy: Aggregation strategy:
            - "first": Return the first value (legacy behavior)
            - "revision": Combine using PLN revision (Bayesian update)
            - "max": Return the value with highest strength
            - "or": Combine using pln_or (disjunction)
            - "weighted": Weight by confidence, then average

    Returns:
        Aggregated truth value, or None if list is empty
    """
    if not truth_values:
        return None

    if len(truth_values) == 1:
        return truth_values[0]

    if strategy == "first":
        return truth_values[0]

    elif strategy == "revision":
        # Combine evidence using PLN revision formula
        # This is the principled Bayesian approach
        return reduce(lambda a, b: a.revise(b), truth_values)

    elif strategy == "max":
        # Return the strongest evidence
        return max(truth_values, key=lambda tv: tv.strength)

    elif strategy == "or":
        # Treat as independent evidence paths (disjunction)
        # P(A or B) = P(A) + P(B) - P(A)*P(B)
        return reduce(pln_or, truth_values)

    elif strategy == "weighted":
        # Weighted average by confidence
        total_conf = sum(tv.confidence for tv in truth_values)
        if total_conf < 0.001:
            # All have near-zero confidence, just average
            avg_strength = sum(tv.strength for tv in truth_values) / len(truth_values)
            avg_conf = sum(tv.confidence for tv in truth_values) / len(truth_values)
            return TruthValue(strength=avg_strength, confidence=avg_conf)

        weighted_strength = sum(
            tv.strength * tv.confidence for tv in truth_values
        ) / total_conf
        # Combined confidence increases with more evidence
        combined_conf = min(0.99, total_conf / (total_conf + 1))
        return TruthValue(strength=weighted_strength, confidence=combined_conf)

    else:
        # Unknown strategy, fall back to first
        return truth_values[0]


# =============================================================================
# ATTENTIONAL FOCUS
# =============================================================================

@dataclass
class AttentionalFocus:
    """
    Controls which atoms receive inference attention.

    Implements a bounded attention mechanism that:
    - Tracks which atoms are currently in focus
    - Assigns boost values to focused atoms
    - Supports temporal decay (unfocused atoms fade)
    - Maintains bounded size (can only focus on N things)

    For audit use cases:
    - Focus on files with multiple issues
    - Prioritize recent findings
    - Shift attention as audit progresses
    """
    max_size: int = 100
    default_boost: float = 1.0

    # Internal state
    _focused: Dict[str, float] = field(default_factory=dict)
    _access_order: List[str] = field(default_factory=list)

    def focus_on(self, atoms: List[str], boost: Optional[float] = None) -> None:
        """
        Add atoms to the attentional focus.

        Args:
            atoms: List of atom names to focus on
            boost: Boost factor for these atoms (default: default_boost)
        """
        if boost is None:
            boost = self.default_boost

        for atom in atoms:
            # Update or add
            self._focused[atom] = boost

            # Update access order
            if atom in self._access_order:
                self._access_order.remove(atom)
            self._access_order.append(atom)

        # Enforce max size by evicting oldest
        self._enforce_size_limit()

    def _enforce_size_limit(self) -> None:
        """Evict oldest atoms if over max_size."""
        while len(self._focused) > self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._focused:
                del self._focused[oldest]

    def is_focused(self, atom: str) -> bool:
        """Check if an atom is currently in focus."""
        return atom in self._focused

    def get_focus_strength(self, atom: str) -> float:
        """Get the focus strength (boost) for an atom."""
        return self._focused.get(atom, 0.0)

    def get_focused_atoms(self) -> List[str]:
        """Get list of all focused atoms."""
        return list(self._focused.keys())

    def set_boost(self, atom: str, boost: float) -> None:
        """Set the boost value for a specific atom."""
        if atom in self._focused:
            self._focused[atom] = boost

    def decay(self, factor: float = 0.9) -> None:
        """
        Apply decay to all focus strengths.

        Args:
            factor: Multiply all strengths by this factor (0 < factor < 1)
        """
        to_remove = []
        for atom in self._focused:
            self._focused[atom] *= factor
            # Remove atoms that decay below threshold
            if self._focused[atom] < 0.01:
                to_remove.append(atom)

        for atom in to_remove:
            del self._focused[atom]
            if atom in self._access_order:
                self._access_order.remove(atom)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_size": self.max_size,
            "default_boost": self.default_boost,
            "focused": dict(self._focused),
            "access_order": list(self._access_order),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionalFocus":
        """Deserialize from dictionary."""
        focus = cls(
            max_size=data.get("max_size", 100),
            default_boost=data.get("default_boost", 1.0),
        )
        focus._focused = dict(data.get("focused", {}))
        focus._access_order = list(data.get("access_order", []))
        return focus


# =============================================================================
# ATTENTION VALUE (STI/LTI/VLTI)
# =============================================================================

@dataclass
class AttentionValue:
    """
    Importance metadata for atoms (inspired by OpenCog ECAN).

    - STI (Short-Term Importance): Urgency, recent relevance [0, 1]
    - LTI (Long-Term Importance): Persistent, foundational relevance [0, 1]
    - VLTI (Very Long-Term Importance): If True, atom is pinned and never fully decays

    For audit use cases:
    - STI: Files with recent issues (just discovered bugs)
    - LTI: Files with persistent problems (known tech debt)
    - VLTI: Critical infrastructure that must always be reviewed
    """
    sti: float = 0.0
    lti: float = 0.0
    vlti: bool = False

    def __post_init__(self):
        """Clamp values to valid range."""
        self.sti = max(0.0, min(1.0, self.sti))
        self.lti = max(0.0, min(1.0, self.lti))

    def total_importance(self) -> float:
        """
        Compute total importance combining STI and LTI.

        Returns a weighted sum with VLTI floor protection.
        """
        # Base importance is weighted combination
        base = 0.6 * self.sti + 0.4 * self.lti

        # VLTI provides minimum floor
        if self.vlti:
            return max(base, 0.5)  # Never drop below 0.5 if pinned

        return base

    def decay_sti(self, factor: float = 0.9) -> None:
        """
        Decay STI (urgency fades over time).

        Args:
            factor: Multiply STI by this factor (0 < factor < 1)
        """
        if self.vlti:
            # VLTI atoms decay slower
            self.sti *= max(factor, 0.95)
        else:
            self.sti *= factor

        # Clamp to valid range
        self.sti = max(0.0, min(1.0, self.sti))

    def decay_lti(self, factor: float = 0.99) -> None:
        """
        Decay LTI (long-term importance fades very slowly).

        Args:
            factor: Multiply LTI by this factor (typically close to 1)
        """
        if self.vlti:
            # VLTI atoms don't decay LTI
            return

        self.lti *= factor
        self.lti = max(0.0, min(1.0, self.lti))

    def stimulate(self, amount: float = 0.1) -> None:
        """
        Increase STI (discovering/accessing an atom boosts urgency).

        Args:
            amount: Amount to increase STI by
        """
        self.sti = min(1.0, self.sti + amount)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sti": self.sti,
            "lti": self.lti,
            "vlti": self.vlti
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionValue":
        """Deserialize from dictionary."""
        return cls(
            sti=data.get("sti", 0.0),
            lti=data.get("lti", 0.0),
            vlti=data.get("vlti", False)
        )

    def __repr__(self) -> str:
        vlti_mark = " VLTI" if self.vlti else ""
        return f"AV(sti={self.sti:.2f}, lti={self.lti:.2f}{vlti_mark})"


# =============================================================================
# ATOMS
# =============================================================================

@dataclass
class Atom:
    """
    A probabilistic atom (statement) in the knowledge base.

    Can be a simple proposition or a predicate with arguments.
    """
    name: str = ""
    predicate: str = ""
    arguments: List[str] = field(default_factory=list)
    truth_value: TruthValue = field(default_factory=TruthValue)
    attention_value: AttentionValue = field(default_factory=AttentionValue)

    def __post_init__(self):
        if not self.name and self.predicate:
            args = ", ".join(self.arguments)
            self.name = f"{self.predicate}({args})"
        elif self.name and not self.predicate:
            # Parse name into predicate and arguments
            if "(" in self.name and ")" in self.name:
                self.predicate = self.name[:self.name.index("(")]
                args_str = self.name[self.name.index("(") + 1:self.name.index(")")]
                self.arguments = [a.strip() for a in args_str.split(",") if a.strip()]

    def matches(self, pattern: str) -> bool:
        """Check if this atom matches a pattern (with variables)."""
        # Simple pattern matching
        if pattern == self.name:
            return True

        # Check for variable patterns like "bird(X)"
        if "(" in pattern:
            pred = pattern[:pattern.index("(")]
            if pred != self.predicate:
                return False

            args_str = pattern[pattern.index("(") + 1:pattern.index(")")]
            pattern_args = [a.strip() for a in args_str.split(",")]

            if len(pattern_args) != len(self.arguments):
                return False

            for p_arg, s_arg in zip(pattern_args, self.arguments):
                # Variables (uppercase) match anything
                if p_arg.isupper() and len(p_arg) == 1:
                    continue
                if p_arg != s_arg:
                    return False

            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "predicate": self.predicate,
            "arguments": self.arguments,
            "truth_value": self.truth_value.to_dict(),
            "attention_value": self.attention_value.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atom":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", ""),
            predicate=data.get("predicate", ""),
            arguments=data.get("arguments", []),
            truth_value=TruthValue.from_dict(data.get("truth_value", {})),
            attention_value=AttentionValue.from_dict(data.get("attention_value", {}))
        )


# =============================================================================
# PLN GRAPH
# =============================================================================

@dataclass
class ImplicationLink:
    """A directed implication link between atoms."""
    antecedent: str
    consequent: str
    truth_value: TruthValue = field(default_factory=TruthValue)


class PLNGraph:
    """
    Probabilistic Logic Network knowledge graph.

    Stores atoms (statements) and implication links with truth values.
    """

    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._implications: Dict[Tuple[str, str], ImplicationLink] = {}

    @property
    def atom_count(self) -> int:
        return len(self._atoms)

    @property
    def link_count(self) -> int:
        return len(self._implications)

    def add_atom(self, name: str, truth_value: TruthValue) -> Atom:
        """Add an atom to the graph."""
        atom = Atom(name=name, truth_value=truth_value)
        self._atoms[name] = atom
        return atom

    def get_atom(self, name: str) -> Optional[Atom]:
        """Get an atom by name."""
        return self._atoms.get(name)

    def get_truth_value(self, name: str) -> Optional[TruthValue]:
        """Get the truth value of an atom."""
        atom = self._atoms.get(name)
        return atom.truth_value if atom else None

    def add_implication(
        self,
        antecedent: str,
        consequent: str,
        truth_value: TruthValue
    ) -> ImplicationLink:
        """Add an implication link: antecedent → consequent."""
        link = ImplicationLink(
            antecedent=antecedent,
            consequent=consequent,
            truth_value=truth_value
        )
        self._implications[(antecedent, consequent)] = link
        return link

    def get_implication(self, antecedent: str, consequent: str) -> Optional[ImplicationLink]:
        """Get an implication link."""
        return self._implications.get((antecedent, consequent))

    def find_implications_from(self, antecedent: str) -> List[ImplicationLink]:
        """Find all implications with the given antecedent."""
        return [
            link for (ant, _), link in self._implications.items()
            if ant == antecedent
        ]

    def find_implications_to(self, consequent: str) -> List[ImplicationLink]:
        """Find all implications with the given consequent."""
        return [
            link for (_, cons), link in self._implications.items()
            if cons == consequent
        ]

    def infer(
        self,
        query: str,
        max_depth: int = 3,
        aggregate: AggregateStrategy = "first"
    ) -> Optional[TruthValue]:
        """
        Infer the truth value of a query through backward chaining.

        Args:
            query: The atom to query
            max_depth: Maximum inference chain depth
            aggregate: Strategy for combining evidence from multiple rules:
                - "first": Return first matching rule (legacy behavior)
                - "revision": Combine using PLN revision (Bayesian update)
                - "max": Return the strongest evidence
                - "or": Combine as independent evidence (disjunction)
                - "weighted": Weight by confidence, then average

        Returns:
            Inferred truth value or None if no inference possible

        Example - Multi-rule aggregation for audit use cases:
            has_todo(X) → needs_review(X) [0.6]
            high_churn(X) → needs_review(X) [0.7]

            With aggregate="first": returns 0.6 (first match only)
            With aggregate="revision": returns ~0.82 (combined evidence)
            With aggregate="or": returns ~0.88 (either path)
        """
        # Direct lookup
        if query in self._atoms:
            return self._atoms[query].truth_value

        # Try to match patterns (query might match an existing atom)
        for name, atom in self._atoms.items():
            if atom.matches(query):
                return atom.truth_value

        # Backward chaining
        if max_depth <= 0:
            return None

        # Collect ALL matching rules and their inferred truth values
        matching_results: List[TruthValue] = []

        # Find implications that conclude the query
        for (ant, cons), link in self._implications.items():
            # Check if consequent pattern can match query
            # e.g., cons="canfly(X)" should match query="canfly(tweety)"
            query_matches = False
            substitutions = {}

            if cons == query:
                query_matches = True
            elif "(" in cons and "(" in query:
                # Pattern matching with variables
                cons_pred = cons[:cons.index("(")]
                query_pred = query[:query.index("(")]

                if cons_pred == query_pred:
                    cons_args = cons[cons.index("(") + 1:cons.index(")")].split(",")
                    query_args = query[query.index("(") + 1:query.index(")")].split(",")

                    if len(cons_args) == len(query_args):
                        query_matches = True
                        for c_arg, q_arg in zip(cons_args, query_args):
                            c_arg = c_arg.strip()
                            q_arg = q_arg.strip()
                            if c_arg.isupper() and len(c_arg) == 1:
                                substitutions[c_arg] = q_arg
                            elif c_arg != q_arg:
                                query_matches = False
                                break

            if query_matches:
                # Substitute variables in antecedent
                ant_query = ant
                for var, val in substitutions.items():
                    ant_query = ant_query.replace(var, val)

                # Try to infer antecedent (recursive call preserves aggregate strategy)
                ant_tv = self.infer(ant_query, max_depth - 1, aggregate)

                if ant_tv is not None:
                    # Apply modus ponens
                    result = pln_implication(ant_tv, link.truth_value)

                    # For "first" strategy, return immediately (legacy behavior)
                    if aggregate == "first":
                        return result

                    # Otherwise, collect for aggregation
                    matching_results.append(result)

        # Aggregate collected results
        return aggregate_truth_values(matching_results, aggregate)

    def _substitute_variables(self, pattern: str, source: str, target: str) -> str:
        """Substitute variables from source pattern into target."""
        # Simple variable substitution
        if "(" not in source or "(" not in target:
            return pattern

        # Extract arguments from source and target
        src_args = source[source.index("(") + 1:source.index(")")].split(",")
        tgt_args = target[target.index("(") + 1:target.index(")")].split(",")

        # Build substitution map
        subs = {}
        for s, t in zip(src_args, tgt_args):
            s = s.strip()
            t = t.strip()
            if s.isupper() and len(s) == 1:  # Variable
                subs[s] = t

        # Apply substitutions to pattern
        if "(" in pattern:
            pred = pattern[:pattern.index("(")]
            args = pattern[pattern.index("(") + 1:pattern.index(")")].split(",")
            new_args = []
            for arg in args:
                arg = arg.strip()
                new_args.append(subs.get(arg, arg))
            return f"{pred}({', '.join(new_args)})"

        return pattern

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        data = {
            "atoms": {name: atom.to_dict() for name, atom in self._atoms.items()},
            "implications": [
                {
                    "antecedent": link.antecedent,
                    "consequent": link.consequent,
                    "truth_value": link.truth_value.to_dict()
                }
                for link in self._implications.values()
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PLNGraph":
        """Load graph from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls()

        for name, atom_data in data.get("atoms", {}).items():
            graph._atoms[name] = Atom.from_dict(atom_data)

        for link_data in data.get("implications", []):
            ant = link_data["antecedent"]
            cons = link_data["consequent"]
            tv = TruthValue.from_dict(link_data.get("truth_value", {}))
            graph._implications[(ant, cons)] = ImplicationLink(ant, cons, tv)

        return graph


# =============================================================================
# SYNAPTIC TRUTH VALUE
# =============================================================================

class SynapticTruthValue(TruthValue):
    """
    Truth value that learns from evidence through synaptic plasticity.

    Integrates with PRISM learning mechanisms.
    """

    def __init__(
        self,
        strength: float = 0.5,
        confidence: float = 0.0,
        learning_rate: float = 0.1
    ):
        super().__init__(strength=strength, confidence=confidence)
        self.learning_rate = learning_rate
        self.positive_count = 0
        self.negative_count = 0

    def observe(self, positive: bool) -> None:
        """
        Observe evidence for or against this truth.

        Updates strength and confidence based on accumulated evidence.
        """
        if positive:
            self.positive_count += 1
        else:
            self.negative_count += 1

        total = self.positive_count + self.negative_count

        # Update strength (proportion of positive evidence)
        self.strength = (self.positive_count + 1) / (total + 2)  # Beta prior

        # Update confidence (based on evidence count)
        # Faster confidence growth
        self.confidence = total / (total + 2)  # Faster asymptotic approach

    def apply_decay(self, factor: float = 0.99) -> None:
        """Apply decay to confidence (uncertainty increases over time)."""
        self.confidence *= factor

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = super().to_dict()
        data["positive_count"] = self.positive_count
        data["negative_count"] = self.negative_count
        data["learning_rate"] = self.learning_rate
        return data


# =============================================================================
# PLN REASONER
# =============================================================================

class PLNReasoner:
    """
    High-level PLN reasoning engine.

    Provides a simple interface for asserting facts, rules, and querying.
    """

    def __init__(self):
        self.graph = PLNGraph()
        self._rules: Dict[Tuple[str, str], TruthValue] = {}

    @property
    def fact_count(self) -> int:
        return self.graph.atom_count

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def assert_fact(
        self,
        statement: str,
        strength: float = 0.9,
        confidence: float = 0.9
    ) -> None:
        """Assert a fact with given truth value."""
        tv = SynapticTruthValue(strength=strength, confidence=confidence)
        self.graph.add_atom(statement, tv)

    def assert_rule(
        self,
        antecedent: str,
        consequent: str,
        strength: float = 0.9,
        confidence: float = 0.9
    ) -> None:
        """Assert a rule (implication) with given truth value."""
        tv = SynapticTruthValue(strength=strength, confidence=confidence)
        self.graph.add_implication(antecedent, consequent, tv)
        self._rules[(antecedent, consequent)] = tv

        # Ensure atoms exist
        # TODO(encapsulation): Using internal _atoms dict instead of public API.
        # PROBLEM: PLNReasoner accesses PLNGraph._atoms directly, but PLNGraph
        #          provides get_atom(name) which returns None if not found.
        # SAME FILE: Both classes are in prism_pln.py, so this is less severe
        #          than cross-module violations, but still inconsistent.
        # FIX: Replace `antecedent not in self.graph._atoms` with
        #      `self.graph.get_atom(antecedent) is None`
        # RATIONALE: Using public API means if PLNGraph internals change
        #          (e.g., lazy loading, caching), this code still works.
        if antecedent not in self.graph._atoms:
            self.graph.add_atom(antecedent, TruthValue(1.0, 1.0))
        if consequent not in self.graph._atoms:
            self.graph.add_atom(consequent, TruthValue(1.0, 1.0))

    def get_rule_truth(self, antecedent: str, consequent: str) -> Optional[TruthValue]:
        """Get the truth value of a rule."""
        link = self.graph.get_implication(antecedent, consequent)
        return link.truth_value if link else None

    def query(
        self,
        statement: str,
        max_depth: int = 5,
        aggregate: AggregateStrategy = "first"
    ) -> Optional[TruthValue]:
        """
        Query the truth value of a statement.

        Args:
            statement: The statement to query
            max_depth: Maximum inference chain depth
            aggregate: Strategy for combining evidence from multiple rules:
                - "first": Return first matching rule (legacy behavior)
                - "revision": Combine using PLN revision (recommended for audits)
                - "max": Return the strongest evidence
                - "or": Combine as independent evidence paths
                - "weighted": Weight by confidence, then average

        Returns:
            Inferred truth value or None if no inference possible

        Example:
            # Multiple rules fire for needs_review
            reasoner.assert_rule("has_todo(X)", "needs_review(X)", strength=0.6)
            reasoner.assert_rule("high_churn(X)", "needs_review(X)", strength=0.7)
            reasoner.assert_fact("has_todo(file_a)", strength=0.95)
            reasoner.assert_fact("high_churn(file_a)", strength=0.9)

            # First match only (legacy)
            result = reasoner.query("needs_review(file_a)")  # ~0.57

            # Combined evidence (recommended)
            result = reasoner.query("needs_review(file_a)", aggregate="revision")  # ~0.82
        """
        return self.graph.infer(statement, max_depth=max_depth, aggregate=aggregate)

    def query_with_trace(
        self,
        statement: str,
        max_depth: int = 5,
        aggregate: AggregateStrategy = "revision"
    ) -> "InferenceTrace":
        """
        Query with full inference trace for explainability.

        Returns an InferenceTrace capturing:
        - Which facts were used
        - Which rules fired and their truth values
        - The inference chain from facts → rules → conclusions
        - How multiple inference paths were aggregated

        This is REAL explainability - not templated responses.

        Args:
            statement: The statement to query
            max_depth: Maximum inference chain depth
            aggregate: Strategy for combining evidence from multiple rules

        Returns:
            InferenceTrace with complete reasoning chain

        Example:
            trace = reasoner.query_with_trace("needs_review(file_a)")
            print(trace.explain())  # Human-readable explanation
        """
        trace = InferenceTrace(query=statement, aggregation_strategy=aggregate)

        # Perform inference with tracing
        result = self._infer_with_trace(
            statement, max_depth, aggregate, trace, depth=0
        )

        trace.final_result = result
        return trace

    def _infer_with_trace(
        self,
        query: str,
        max_depth: int,
        aggregate: AggregateStrategy,
        trace: "InferenceTrace",
        depth: int
    ) -> Optional[TruthValue]:
        """
        Internal inference with tracing.

        Records each inference step in the trace.
        """
        # Direct lookup - record as fact used
        atom = self.graph.get_atom(query)
        if atom is not None:
            trace.add_fact(query, atom.truth_value)
            return atom.truth_value

        # Try pattern matching
        for name, atom in self.graph._atoms.items():
            if atom.matches(query):
                trace.add_fact(name, atom.truth_value)
                return atom.truth_value

        if max_depth <= 0:
            return None

        # Collect matching rules with traces
        matching_results: List[TruthValue] = []

        for (ant, cons), link in self.graph._implications.items():
            # Pattern matching logic
            query_matches = False
            substitutions = {}

            if cons == query:
                query_matches = True
            elif "(" in cons and "(" in query:
                cons_pred = cons[:cons.index("(")]
                query_pred = query[:query.index("(")]

                if cons_pred == query_pred:
                    cons_args = cons[cons.index("(") + 1:cons.index(")")].split(",")
                    query_args = query[query.index("(") + 1:query.index(")")].split(",")

                    if len(cons_args) == len(query_args):
                        query_matches = True
                        for c_arg, q_arg in zip(cons_args, query_args):
                            c_arg = c_arg.strip()
                            q_arg = q_arg.strip()
                            if c_arg.isupper() and len(c_arg) == 1:
                                substitutions[c_arg] = q_arg
                            elif c_arg != q_arg:
                                query_matches = False
                                break

            if query_matches:
                # Substitute variables in antecedent
                ant_query = ant
                for var, val in substitutions.items():
                    ant_query = ant_query.replace(var, val)

                # Recursive inference (with tracing)
                ant_tv = self._infer_with_trace(
                    ant_query, max_depth - 1, aggregate, trace, depth + 1
                )

                if ant_tv is not None:
                    # Apply modus ponens
                    result = pln_implication(ant_tv, link.truth_value)

                    # Record inference step
                    step = InferenceStep(
                        rule_antecedent=ant,
                        rule_consequent=cons,
                        rule_truth_value=link.truth_value,
                        antecedent_truth_value=ant_tv,
                        result_truth_value=result,
                        substitutions=substitutions,
                        depth=depth
                    )
                    trace.add_step(step)

                    if aggregate == "first":
                        return result

                    matching_results.append(result)

        # Record aggregation inputs
        if matching_results:
            trace.aggregation_inputs = matching_results

        return aggregate_truth_values(matching_results, aggregate)

    def query_with_attention(
        self,
        statement: str,
        focus: "AttentionalFocus",
        max_depth: int = 5,
        aggregate: AggregateStrategy = "first",
        return_stats: bool = False
    ) -> Any:  # Returns TruthValue or Tuple[TruthValue, Dict] if return_stats
        """
        Query with attention-guided inference.

        Atoms in the attentional focus have their inference paths boosted,
        making focused atoms more influential in the final result.

        Args:
            statement: The statement to query
            focus: AttentionalFocus controlling which atoms to prioritize
            max_depth: Maximum inference chain depth
            aggregate: Strategy for combining evidence from multiple rules
            return_stats: If True, return (result, stats) tuple

        Returns:
            Inferred truth value, or (truth_value, stats) if return_stats=True

        Example:
            focus = AttentionalFocus()
            focus.focus_on(["has_bug(file_a)"], boost=2.0)
            result = reasoner.query_with_attention(
                "needs_review(file_a)",
                focus=focus,
                aggregate="weighted"
            )
        """
        stats = {"rules_explored": 0, "atoms_boosted": 0}

        # Get base inference result with attention boosting
        result = self._infer_with_attention(
            statement, focus, max_depth, aggregate, stats
        )

        if return_stats:
            return result, stats
        return result

    def _infer_with_attention(
        self,
        query: str,
        focus: "AttentionalFocus",
        max_depth: int,
        aggregate: AggregateStrategy,
        stats: Dict[str, int]
    ) -> Optional[TruthValue]:
        """
        Internal attention-guided inference.

        Modifies inference by boosting focused atoms' contributions.
        """
        # Direct lookup
        atom = self.graph.get_atom(query)
        if atom is not None:
            tv = atom.truth_value
            # Apply boost if focused
            if focus.is_focused(query):
                boost = focus.get_focus_strength(query)
                stats["atoms_boosted"] += 1
                # Boost confidence (more attention = more weight)
                boosted_conf = min(0.99, tv.confidence * boost)
                return TruthValue(strength=tv.strength, confidence=boosted_conf)
            return tv

        # Try pattern matching
        for name, atom in self.graph._atoms.items():
            if atom.matches(query):
                tv = atom.truth_value
                if focus.is_focused(name):
                    boost = focus.get_focus_strength(name)
                    stats["atoms_boosted"] += 1
                    boosted_conf = min(0.99, tv.confidence * boost)
                    return TruthValue(strength=tv.strength, confidence=boosted_conf)
                return tv

        if max_depth <= 0:
            return None

        # Collect matching rules with attention weighting
        matching_results: List[TruthValue] = []

        for (ant, cons), link in self.graph._implications.items():
            stats["rules_explored"] += 1

            # Check if focused atoms are involved - if so, prioritize
            rule_boost = 1.0
            if focus.is_focused(ant) or focus.is_focused(cons):
                rule_boost = max(
                    focus.get_focus_strength(ant),
                    focus.get_focus_strength(cons),
                    1.0
                )

            # Pattern matching
            query_matches = False
            substitutions = {}

            if cons == query:
                query_matches = True
            elif "(" in cons and "(" in query:
                cons_pred = cons[:cons.index("(")]
                query_pred = query[:query.index("(")]

                if cons_pred == query_pred:
                    cons_args = cons[cons.index("(") + 1:cons.index(")")].split(",")
                    query_args = query[query.index("(") + 1:query.index(")")].split(",")

                    if len(cons_args) == len(query_args):
                        query_matches = True
                        for c_arg, q_arg in zip(cons_args, query_args):
                            c_arg = c_arg.strip()
                            q_arg = q_arg.strip()
                            if c_arg.isupper() and len(c_arg) == 1:
                                substitutions[c_arg] = q_arg
                            elif c_arg != q_arg:
                                query_matches = False
                                break

            if query_matches:
                # Substitute variables
                ant_query = ant
                for var, val in substitutions.items():
                    ant_query = ant_query.replace(var, val)

                # Check if substituted antecedent is focused
                if focus.is_focused(ant_query):
                    rule_boost = max(rule_boost, focus.get_focus_strength(ant_query))

                # Recursive inference
                ant_tv = self._infer_with_attention(
                    ant_query, focus, max_depth - 1, aggregate, stats
                )

                if ant_tv is not None:
                    result = pln_implication(ant_tv, link.truth_value)

                    # Apply rule boost to confidence
                    if rule_boost > 1.0:
                        stats["atoms_boosted"] += 1
                        boosted_conf = min(0.99, result.confidence * rule_boost)
                        result = TruthValue(strength=result.strength, confidence=boosted_conf)

                    if aggregate == "first":
                        return result

                    matching_results.append(result)

        return aggregate_truth_values(matching_results, aggregate)

    def observe(self, statement: str, is_true: bool) -> None:
        """Observe evidence about a statement."""
        atom = self.graph.get_atom(statement)
        if atom and isinstance(atom.truth_value, SynapticTruthValue):
            atom.truth_value.observe(positive=is_true)

        # Update rules involving this statement
        for (ant, cons), tv in list(self._rules.items()):
            # Get the actual implication link from the graph
            link = self.graph.get_implication(ant, cons)
            if link and isinstance(link.truth_value, SynapticTruthValue):
                # Check for pattern matching with variables
                cons_matches = (cons == statement)
                if not cons_matches and "(" in cons:
                    cons_pred = cons[:cons.index("(")]
                    if "(" in statement:
                        stmt_pred = statement[:statement.index("(")]
                        cons_matches = (cons_pred == stmt_pred)

                if cons_matches:
                    # Evidence about consequent affects rule
                    # Find matching antecedent
                    ant_query = ant
                    if "(" in cons and "(" in statement:
                        # Substitute variables
                        cons_args = cons[cons.index("(") + 1:cons.index(")")].split(",")
                        stmt_args = statement[statement.index("(") + 1:statement.index(")")].split(",")
                        for c_arg, s_arg in zip(cons_args, stmt_args):
                            c_arg = c_arg.strip()
                            s_arg = s_arg.strip()
                            if c_arg.isupper() and len(c_arg) == 1:
                                ant_query = ant_query.replace(c_arg, s_arg)

                    ant_atom = self.graph.get_atom(ant_query)
                    if ant_atom and ant_atom.truth_value.strength > 0.5:
                        # If antecedent is true but consequent observed false,
                        # weaken the rule
                        if not is_true:
                            link.truth_value.observe(positive=False)
                            self._rules[(ant, cons)] = link.truth_value
                        else:
                            link.truth_value.observe(positive=True)
                            self._rules[(ant, cons)] = link.truth_value

    def explain(self, statement: str) -> List[str]:
        """Explain the reasoning chain for a statement."""
        explanations = []

        # Direct fact?
        if statement in self.graph._atoms:
            tv = self.graph._atoms[statement].truth_value
            explanations.append(f"Direct fact: {statement} = {tv}")
            return explanations

        # Find inference chain
        for (ant, cons), link in self.graph._implications.items():
            if cons == statement or Atom(name=cons).matches(statement):
                explanations.append(f"Rule: {ant} → {cons} ({link.truth_value})")

                # Check antecedent
                ant_tv = self.graph.infer(ant)
                if ant_tv:
                    explanations.append(f"  Antecedent: {ant} = {ant_tv}")

        return explanations

    # =========================================================================
    # IMPORTANCE (STI/LTI) METHODS
    # =========================================================================

    def get_attention(self, atom_name: str) -> AttentionValue:
        """
        Get the AttentionValue for an atom.

        Args:
            atom_name: Name of the atom

        Returns:
            AttentionValue for the atom (default if not found)
        """
        atom = self.graph.get_atom(atom_name)
        if atom is not None:
            return atom.attention_value
        return AttentionValue()  # Default

    def set_attention(self, atom_name: str, attention_value: AttentionValue) -> None:
        """
        Set the AttentionValue for an atom.

        Args:
            atom_name: Name of the atom
            attention_value: AttentionValue to set
        """
        atom = self.graph.get_atom(atom_name)
        if atom is not None:
            atom.attention_value = attention_value

    def stimulate(self, atom_name: str, amount: float = 0.1) -> None:
        """
        Stimulate an atom (increase its STI).

        Use when discovering or accessing an atom to boost its urgency.

        Args:
            atom_name: Name of the atom to stimulate
            amount: Amount to increase STI by
        """
        atom = self.graph.get_atom(atom_name)
        if atom is not None:
            atom.attention_value.stimulate(amount)

    def collect_rent(
        self,
        sti_decay: float = 0.9,
        lti_decay: float = 0.99
    ) -> None:
        """
        Apply decay to all atoms (rent collection cycle).

        Simulates importance fading over time. VLTI atoms are protected.

        Args:
            sti_decay: Factor to multiply STI by (0 < factor < 1)
            lti_decay: Factor to multiply LTI by (typically close to 1)
        """
        for atom in self.graph._atoms.values():
            atom.attention_value.decay_sti(sti_decay)
            atom.attention_value.decay_lti(lti_decay)

    def get_atoms_by_sti(self, min_sti: float = 0.5) -> List[str]:
        """
        Get atoms with STI above threshold.

        For audits: Find urgent items.

        Args:
            min_sti: Minimum STI threshold

        Returns:
            List of atom names with STI >= min_sti
        """
        return [
            name for name, atom in self.graph._atoms.items()
            if atom.attention_value.sti >= min_sti
        ]

    def get_atoms_by_lti(self, min_lti: float = 0.5) -> List[str]:
        """
        Get atoms with LTI above threshold.

        For audits: Find persistent issues.

        Args:
            min_lti: Minimum LTI threshold

        Returns:
            List of atom names with LTI >= min_lti
        """
        return [
            name for name, atom in self.graph._atoms.items()
            if atom.attention_value.lti >= min_lti
        ]

    def get_vlti_atoms(self) -> List[str]:
        """
        Get all VLTI (pinned) atoms.

        For audits: Get critical items that must always be reviewed.

        Returns:
            List of atom names with VLTI=True
        """
        return [
            name for name, atom in self.graph._atoms.items()
            if atom.attention_value.vlti
        ]

    def get_atoms_by_importance(self) -> List[str]:
        """
        Get all atoms sorted by total importance (descending).

        Returns:
            List of atom names sorted by total_importance()
        """
        atoms_with_importance = [
            (name, atom.attention_value.total_importance())
            for name, atom in self.graph._atoms.items()
        ]
        atoms_with_importance.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in atoms_with_importance]

    def query_with_importance(
        self,
        statement: str,
        spread_importance: bool = True,
        min_importance: float = 0.0,
        max_depth: int = 5,
        aggregate: AggregateStrategy = "first"
    ) -> Any:  # Returns TruthValue or List[TruthValue]
        """
        Query with importance spreading through inference chains.

        Args:
            statement: The statement to query
            spread_importance: If True, importance spreads to inferred atoms
            min_importance: Only return results from atoms with importance >= this
            max_depth: Maximum inference chain depth
            aggregate: Strategy for combining evidence

        Returns:
            TruthValue (if min_importance=0) or List[TruthValue] (if filtering)
        """
        # Perform inference
        result = self._infer_with_importance(
            statement,
            spread_importance=spread_importance,
            max_depth=max_depth,
            aggregate=aggregate,
            spread_factor=0.7  # How much importance attenuates per hop
        )

        # If filtering by importance, return list
        if min_importance > 0:
            # Get all atoms matching the pattern and filter
            filtered = []
            for name, atom in self.graph._atoms.items():
                if atom.matches(statement) and atom.attention_value.sti >= min_importance:
                    filtered.append(atom.truth_value)
            return filtered

        return result

    def _infer_with_importance(
        self,
        query: str,
        spread_importance: bool,
        max_depth: int,
        aggregate: AggregateStrategy,
        spread_factor: float,
        source_importance: Optional[float] = None
    ) -> Optional[TruthValue]:
        """
        Internal importance-spreading inference.

        Spreads importance from source atoms to inferred conclusions.
        """
        # Direct lookup
        atom = self.graph.get_atom(query)
        if atom is not None:
            if spread_importance and source_importance is not None:
                # Spread importance from source to this atom
                inherited = source_importance * spread_factor
                atom.attention_value.sti = max(
                    atom.attention_value.sti,
                    inherited
                )
            return atom.truth_value

        # Try pattern matching
        for name, atom in self.graph._atoms.items():
            if atom.matches(query):
                if spread_importance and source_importance is not None:
                    inherited = source_importance * spread_factor
                    atom.attention_value.sti = max(
                        atom.attention_value.sti,
                        inherited
                    )
                return atom.truth_value

        if max_depth <= 0:
            return None

        # Backward chaining with importance spreading
        matching_results: List[TruthValue] = []

        for (ant, cons), link in self.graph._implications.items():
            # Pattern matching
            query_matches = False
            substitutions = {}

            if cons == query:
                query_matches = True
            elif "(" in cons and "(" in query:
                cons_pred = cons[:cons.index("(")]
                query_pred = query[:query.index("(")]

                if cons_pred == query_pred:
                    cons_args = cons[cons.index("(") + 1:cons.index(")")].split(",")
                    query_args = query[query.index("(") + 1:query.index(")")].split(",")

                    if len(cons_args) == len(query_args):
                        query_matches = True
                        for c_arg, q_arg in zip(cons_args, query_args):
                            c_arg = c_arg.strip()
                            q_arg = q_arg.strip()
                            if c_arg.isupper() and len(c_arg) == 1:
                                substitutions[c_arg] = q_arg
                            elif c_arg != q_arg:
                                query_matches = False
                                break

            if query_matches:
                # Substitute variables
                ant_query = ant
                for var, val in substitutions.items():
                    ant_query = ant_query.replace(var, val)

                # Get source atom's importance for spreading
                ant_atom = self.graph.get_atom(ant_query)
                ant_importance = None
                if ant_atom is not None:
                    ant_importance = ant_atom.attention_value.sti

                # Recursive inference with importance spreading
                ant_tv = self._infer_with_importance(
                    ant_query,
                    spread_importance=spread_importance,
                    max_depth=max_depth - 1,
                    aggregate=aggregate,
                    spread_factor=spread_factor,
                    source_importance=source_importance
                )

                if ant_tv is not None:
                    result = pln_implication(ant_tv, link.truth_value)

                    # Spread importance to conclusion
                    if spread_importance and ant_importance is not None:
                        # Create/update atom for conclusion if needed
                        conclusion_atom = self.graph.get_atom(query)
                        if conclusion_atom is None:
                            conclusion_atom = self.graph.add_atom(
                                query, TruthValue(result.strength, result.confidence)
                            )
                        # Spread importance
                        inherited = ant_importance * spread_factor
                        conclusion_atom.attention_value.sti = max(
                            conclusion_atom.attention_value.sti,
                            inherited
                        )

                    if aggregate == "first":
                        return result

                    matching_results.append(result)

        return aggregate_truth_values(matching_results, aggregate)

    def query_by_importance(
        self,
        statement: str,
        return_stats: bool = False
    ) -> Any:  # Returns List[TruthValue] or Tuple[List[TruthValue], Dict]
        """
        Query with importance-guided exploration order.

        Explores high-importance atoms first.

        Args:
            statement: Pattern to query (can contain variables like X)
            return_stats: If True, return (results, stats) tuple

        Returns:
            List of matching TruthValues, or (list, stats) if return_stats=True
        """
        stats = {
            "first_explored": None,
            "exploration_order": [],
            "total_explored": 0
        }

        # Get atoms sorted by importance
        sorted_atoms = self.get_atoms_by_importance()

        results = []

        for atom_name in sorted_atoms:
            atom = self.graph.get_atom(atom_name)
            if atom is None:
                continue

            stats["total_explored"] += 1
            if stats["first_explored"] is None:
                stats["first_explored"] = atom_name
            stats["exploration_order"].append(atom_name)

            # Check if atom matches pattern
            if atom.matches(statement):
                results.append(atom.truth_value)

        if return_stats:
            return results, stats
        return results

    # =========================================================================
    # COMPOUND TERM METHODS
    # =========================================================================

    def set_type_registry(self, registry: TypeRegistry) -> None:
        """Set the type registry for type-constrained unification."""
        self._type_registry = registry

    def assert_compound_fact(
        self,
        term_str: str,
        strength: float = 0.9,
        confidence: float = 0.9
    ) -> None:
        """
        Assert a compound fact with the given truth value.

        Args:
            term_str: String representation of compound term
            strength: Truth value strength
            confidence: Truth value confidence

        Example:
            reasoner.assert_compound_fact(
                "file_info(auth.py, metrics(high_churn, security_critical))",
                strength=0.95
            )
        """
        # Store the term string and its parsed form
        term = parse_term(term_str)
        tv = SynapticTruthValue(strength=strength, confidence=confidence)
        self.graph.add_atom(term_str, tv)

        # Also store with canonical string representation
        canonical = str(term)
        if canonical != term_str:
            self.graph.add_atom(canonical, tv)

    def assert_compound_rule(
        self,
        antecedent_str: str,
        consequent_str: str,
        strength: float = 0.9,
        confidence: float = 0.9
    ) -> None:
        """
        Assert a rule with compound terms.

        Args:
            antecedent_str: Antecedent term string (can have variables)
            consequent_str: Consequent term string
            strength: Rule strength
            confidence: Rule confidence

        Example:
            reasoner.assert_compound_rule(
                "file_info(X, metrics(high_churn, security_critical))",
                "needs_immediate_review(X)",
                strength=0.95
            )
        """
        tv = SynapticTruthValue(strength=strength, confidence=confidence)
        self.graph.add_implication(antecedent_str, consequent_str, tv)
        self._rules[(antecedent_str, consequent_str)] = tv

    def query_compound(
        self,
        term_str: str,
        max_depth: int = 5
    ) -> Any:  # Returns QueryResult or List[QueryResult]
        """
        Query using compound term with unification.

        Args:
            term_str: Query term (can contain variables)
            max_depth: Maximum inference depth

        Returns:
            Single QueryResult if no variables, List[QueryResult] if variables

        Example:
            # Exact match query
            result = reasoner.query_compound(
                "file_info(auth.py, metrics(high_churn, security_critical))"
            )

            # Variable query - returns all matches
            results = reasoner.query_compound("file_info(X, metrics(high, Y))")
        """
        query_term = parse_term(term_str)
        has_variables = _term_has_variables(query_term)

        registry = getattr(self, '_type_registry', None)

        if has_variables:
            # Find all matching atoms
            results = []
            for name, atom in self.graph._atoms.items():
                atom_term = parse_term(name)
                subst = unify(query_term, atom_term, type_registry=registry)
                if subst is not None:
                    results.append(QueryResult(
                        truth_value=atom.truth_value,
                        bindings=subst
                    ))

            # Also try inference through rules
            inferred = self._infer_compound(
                query_term, max_depth, registry
            )
            results.extend(inferred)

            return results
        else:
            # Exact match or inference
            direct = self.graph.get_atom(term_str)
            if direct:
                return QueryResult(
                    truth_value=direct.truth_value,
                    bindings={}
                )

            # Try inference
            inferred = self._infer_compound(query_term, max_depth, registry)
            if inferred:
                return inferred[0]

            return None

    def _infer_compound(
        self,
        query_term: Term,
        max_depth: int,
        registry: Optional[TypeRegistry]
    ) -> List["QueryResult"]:
        """Internal compound inference with unification."""
        if max_depth <= 0:
            return []

        results = []

        for (ant_str, cons_str), link in self.graph._implications.items():
            cons_term = parse_term(cons_str)

            # Try to unify query with consequent
            subst = unify(query_term, cons_term, type_registry=registry)
            if subst is None:
                continue

            # Apply substitution to antecedent
            ant_term = parse_term(ant_str)
            ant_resolved = apply_substitution(ant_term, subst)
            ant_resolved_str = str(ant_resolved)

            # Check if antecedent is satisfied
            ant_result = None

            # Direct lookup
            ant_atom = self.graph.get_atom(ant_resolved_str)
            if ant_atom:
                ant_result = ant_atom.truth_value
            else:
                # Try to find matching atom with unification
                for name, atom in self.graph._atoms.items():
                    atom_term = parse_term(name)
                    match_subst = unify(ant_resolved, atom_term, subst, registry)
                    if match_subst is not None:
                        ant_result = atom.truth_value
                        subst = match_subst
                        break

            if ant_result is not None:
                # Apply modus ponens
                inferred_tv = pln_implication(ant_result, link.truth_value)
                results.append(QueryResult(
                    truth_value=inferred_tv,
                    bindings=subst
                ))

        return results


def _term_has_variables(term: Term) -> bool:
    """Check if a term contains any variables."""
    if term.is_variable:
        return True
    for arg in term.args:
        if isinstance(arg, Term) and _term_has_variables(arg):
            return True
    return False


@dataclass
class QueryResult:
    """Result of a compound query with variable bindings."""
    truth_value: TruthValue
    bindings: Dict[str, Any] = field(default_factory=dict)

    @property
    def strength(self) -> float:
        return self.truth_value.strength

    @property
    def confidence(self) -> float:
        return self.truth_value.confidence


# =============================================================================
# INFERENCE TRACING (Phase 2: Real Explainability)
# =============================================================================

@dataclass
class InferenceStep:
    """
    A single step in an inference chain.

    Captures:
    - The rule that fired (antecedent → consequent)
    - Truth values at each stage
    - Variable substitutions applied
    """
    rule_antecedent: str
    rule_consequent: str
    rule_truth_value: TruthValue
    antecedent_truth_value: TruthValue
    result_truth_value: TruthValue
    substitutions: Dict[str, str] = field(default_factory=dict)
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": f"{self.rule_antecedent} → {self.rule_consequent}",
            "rule_tv": self.rule_truth_value.to_dict(),
            "antecedent_tv": self.antecedent_truth_value.to_dict(),
            "result_tv": self.result_truth_value.to_dict(),
            "substitutions": self.substitutions,
            "depth": self.depth,
        }

    def __str__(self) -> str:
        subs = ", ".join(f"{k}={v}" for k, v in self.substitutions.items())
        subs_str = f" [{subs}]" if subs else ""
        return (
            f"  Rule: {self.rule_antecedent} → {self.rule_consequent}{subs_str}\n"
            f"    Rule strength: {self.rule_truth_value.strength:.2%} "
            f"(conf: {self.rule_truth_value.confidence:.2%})\n"
            f"    Antecedent: {self.antecedent_truth_value.strength:.2%} "
            f"(conf: {self.antecedent_truth_value.confidence:.2%})\n"
            f"    → Result: {self.result_truth_value.strength:.2%} "
            f"(conf: {self.result_truth_value.confidence:.2%})"
        )


@dataclass
class InferenceTrace:
    """
    Complete trace of an inference chain.

    Captures the full reasoning path from facts through rules to conclusions,
    enabling real explainability (not templated responses).
    """
    query: str
    steps: List[InferenceStep] = field(default_factory=list)
    facts_used: Dict[str, TruthValue] = field(default_factory=dict)
    final_result: Optional[TruthValue] = None
    aggregation_strategy: str = "first"
    aggregation_inputs: List[TruthValue] = field(default_factory=list)

    def add_step(self, step: InferenceStep) -> None:
        """Add an inference step to the trace."""
        self.steps.append(step)

    def add_fact(self, name: str, tv: TruthValue) -> None:
        """Record a fact that was used in inference."""
        self.facts_used[name] = tv

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "facts_used": {k: v.to_dict() for k, v in self.facts_used.items()},
            "final_result": self.final_result.to_dict() if self.final_result else None,
            "aggregation_strategy": self.aggregation_strategy,
            "aggregation_inputs": [tv.to_dict() for tv in self.aggregation_inputs],
        }

    def explain(self, verbose: bool = False) -> str:
        """
        Generate human-readable explanation of the inference.

        This is REAL explainability - shows actual rules that fired,
        not templated responses.
        """
        lines = []

        # Query
        lines.append(f"Query: {self.query}")
        lines.append("")

        # Facts used
        if self.facts_used:
            lines.append("Facts asserted:")
            for fact, tv in self.facts_used.items():
                lines.append(f"  • {fact}: {tv.strength:.2%} (conf: {tv.confidence:.2%})")
            lines.append("")

        # Inference steps
        if self.steps:
            lines.append("Inference chain:")
            for i, step in enumerate(self.steps, 1):
                lines.append(f"\n  Step {i} (depth {step.depth}):")
                lines.append(str(step))
            lines.append("")

        # Aggregation
        if len(self.aggregation_inputs) > 1:
            lines.append(f"Aggregation ({self.aggregation_strategy}):")
            lines.append(f"  Combining {len(self.aggregation_inputs)} inference paths:")
            for i, tv in enumerate(self.aggregation_inputs, 1):
                lines.append(f"    Path {i}: {tv.strength:.2%} (conf: {tv.confidence:.2%})")
            lines.append("")

        # Final result
        if self.final_result:
            lines.append(f"Final result: {self.final_result.strength:.2%} "
                        f"(confidence: {self.final_result.confidence:.2%})")
            lines.append(f"Probability: {self.final_result.to_probability():.2%}")
        else:
            lines.append("Final result: No inference possible")

        return "\n".join(lines)
