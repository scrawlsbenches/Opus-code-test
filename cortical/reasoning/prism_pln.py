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
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import math


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
            "truth_value": self.truth_value.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atom":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", ""),
            predicate=data.get("predicate", ""),
            arguments=data.get("arguments", []),
            truth_value=TruthValue.from_dict(data.get("truth_value", {}))
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

    def infer(self, query: str, max_depth: int = 3) -> Optional[TruthValue]:
        """
        Infer the truth value of a query through backward chaining.

        Args:
            query: The atom to query
            max_depth: Maximum inference chain depth

        Returns:
            Inferred truth value or None if no inference possible
        """
        # Direct lookup
        if query in self._atoms:
            return self._atoms[query].truth_value

        # Try to match patterns (query might match an existing atom)
        query_atom = Atom(name=query)
        for name, atom in self._atoms.items():
            if atom.matches(query):
                return atom.truth_value

        # Backward chaining
        if max_depth <= 0:
            return None

        # Find implications that conclude the query
        for (ant, cons), link in self._implications.items():
            # Check if consequent pattern can match query
            # e.g., cons="canfly(X)" should match query="canfly(tweety)"
            cons_atom = Atom(name=cons)
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

                # Try to infer antecedent
                ant_tv = self.infer(ant_query, max_depth - 1)

                if ant_tv is not None:
                    # Apply modus ponens
                    result = pln_implication(ant_tv, link.truth_value)
                    return result

        return None

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
        if antecedent not in self.graph._atoms:
            self.graph.add_atom(antecedent, TruthValue(1.0, 1.0))
        if consequent not in self.graph._atoms:
            self.graph.add_atom(consequent, TruthValue(1.0, 1.0))

    def get_rule_truth(self, antecedent: str, consequent: str) -> Optional[TruthValue]:
        """Get the truth value of a rule."""
        link = self.graph.get_implication(antecedent, consequent)
        return link.truth_value if link else None

    def query(self, statement: str, max_depth: int = 5) -> Optional[TruthValue]:
        """Query the truth value of a statement."""
        return self.graph.infer(statement, max_depth=max_depth)

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
