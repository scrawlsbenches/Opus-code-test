#!/usr/bin/env python3
"""
Probabilistic Logic Networks for Training Data Generation.

PLN combines logic programming with probability theory to generate
logically consistent training data with confidence scores.

Key concepts:
- Facts: "pagerank is_a algorithm" (confidence 1.0)
- Rules: "X is_a Y, Y has_property Z => X has_property Z"
- Inference: Derive new facts with propagated confidence
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Fact:
    """A logical fact with confidence."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "axiom"

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return (self.subject, self.predicate, self.object) == \
               (other.subject, other.predicate, other.object)

    def to_text(self) -> str:
        """Convert to natural language."""
        templates = {
            'is_a': f"{self.subject} is a type of {self.object}",
            'has_property': f"{self.subject} has {self.object}",
            'located_in': f"{self.subject} is located in {self.object}",
            'implements': f"{self.subject} implements {self.object}",
            'uses': f"{self.subject} uses {self.object}",
            'related_to': f"{self.subject} is related to {self.object}",
            'part_of': f"{self.subject} is part of {self.object}",
            'contains': f"{self.subject} contains {self.object}",
        }
        return templates.get(self.predicate, f"{self.subject} {self.predicate} {self.object}")


@dataclass
class Rule:
    """An inference rule with confidence decay."""
    name: str
    antecedent: List[Tuple[str, str, str]]  # [(subj_var, pred, obj_var), ...]
    consequent: Tuple[str, str, str]
    confidence_decay: float = 0.9  # How much confidence degrades per inference


class ProbabilisticLogicNetwork:
    """
    PLN for generating logically consistent training data.

    Uses forward chaining to derive new facts from axioms and rules.
    """

    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        self.inferred: Set[Fact] = set()

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0, source: str = "axiom"):
        """Add a base fact."""
        self.facts.add(Fact(subject, predicate, obj, confidence, source))

    def add_rule(self, rule: Rule):
        """Add an inference rule."""
        self.rules.append(rule)

    def load_knowledge_base(self):
        """Load domain knowledge about the codebase."""

        # Ontology: Type hierarchy
        type_hierarchy = {
            'algorithm': ['pagerank', 'tfidf', 'bm25', 'louvain', 'clustering'],
            'component': ['processor', 'tokenizer', 'gotmanager', 'wovenmind', 'sparksm'],
            'data_structure': ['minicolumn', 'layer', 'edge', 'graph'],
            'concept': ['hebbian_learning', 'lateral_connections', 'activation'],
            'file': ['pagerank.py', 'tfidf.py', 'api.py', 'tokenizer.py'],
        }

        for category, members in type_hierarchy.items():
            for member in members:
                self.add_fact(member, 'is_a', category)

        # Location facts
        locations = {
            'pagerank': 'cortical/analysis/pagerank.py',
            'tfidf': 'cortical/analysis/tfidf.py',
            'gotmanager': 'cortical/got/api.py',
            'tokenizer': 'cortical/tokenizer.py',
            'wovenmind': 'cortical/reasoning/woven_mind.py',
            'processor': 'cortical/processor/',
        }

        for component, location in locations.items():
            self.add_fact(component, 'located_in', location)

        # Property facts
        properties = {
            'algorithm': ['computes', 'analyzes', 'processes'],
            'component': ['initializes', 'configures', 'executes'],
            'data_structure': ['stores', 'indexes', 'connects'],
        }

        for category, props in properties.items():
            for prop in props:
                self.add_fact(category, 'has_property', prop)

        # Relationships
        relationships = [
            ('pagerank', 'uses', 'graph'),
            ('tfidf', 'uses', 'document'),
            ('louvain', 'uses', 'clustering'),
            ('processor', 'contains', 'tokenizer'),
            ('processor', 'uses', 'pagerank'),
            ('wovenmind', 'contains', 'hive'),
            ('wovenmind', 'contains', 'cortex'),
            ('gotmanager', 'manages', 'tasks'),
            ('gotmanager', 'manages', 'decisions'),
        ]

        for subj, pred, obj in relationships:
            self.add_fact(subj, pred, obj)

    def load_inference_rules(self):
        """Load inference rules for forward chaining."""

        # Rule: is_a transitivity
        # If X is_a Y and Y is_a Z, then X is_a Z
        self.add_rule(Rule(
            name="is_a_transitivity",
            antecedent=[('X', 'is_a', 'Y'), ('Y', 'is_a', 'Z')],
            consequent=('X', 'is_a', 'Z'),
            confidence_decay=0.85
        ))

        # Rule: Property inheritance
        # If X is_a Y and Y has_property P, then X has_property P
        self.add_rule(Rule(
            name="property_inheritance",
            antecedent=[('X', 'is_a', 'Y'), ('Y', 'has_property', 'P')],
            consequent=('X', 'has_property', 'P'),
            confidence_decay=0.9
        ))

        # Rule: Location inheritance
        # If X part_of Y and Y located_in L, then X located_in L
        self.add_rule(Rule(
            name="location_inheritance",
            antecedent=[('X', 'part_of', 'Y'), ('Y', 'located_in', 'L')],
            consequent=('X', 'located_in', 'L'),
            confidence_decay=0.95
        ))

        # Rule: Symmetric relationship
        # If X related_to Y, then Y related_to X
        self.add_rule(Rule(
            name="symmetry",
            antecedent=[('X', 'related_to', 'Y')],
            consequent=('Y', 'related_to', 'X'),
            confidence_decay=1.0
        ))

    def forward_chain(self, max_iterations: int = 5) -> List[Fact]:
        """
        Apply forward chaining to derive new facts.

        Returns list of newly inferred facts.
        """
        all_facts = set(self.facts)
        new_facts = []

        for iteration in range(max_iterations):
            iteration_new = []

            for rule in self.rules:
                # Try to match rule antecedent against all facts
                matches = self._find_matches(rule, all_facts)

                for bindings, min_confidence in matches:
                    # Apply consequent with bindings
                    new_subj = bindings.get(rule.consequent[0], rule.consequent[0])
                    new_pred = rule.consequent[1]
                    new_obj = bindings.get(rule.consequent[2], rule.consequent[2])

                    new_confidence = min_confidence * rule.confidence_decay

                    new_fact = Fact(
                        new_subj, new_pred, new_obj,
                        confidence=new_confidence,
                        source=f"inferred:{rule.name}"
                    )

                    if new_fact not in all_facts and new_confidence > 0.3:
                        iteration_new.append(new_fact)
                        all_facts.add(new_fact)

            if not iteration_new:
                break  # Fixed point reached

            new_facts.extend(iteration_new)
            print(f"  Iteration {iteration + 1}: inferred {len(iteration_new)} new facts")

        self.inferred = set(new_facts)
        return new_facts

    def _find_matches(self, rule: Rule, facts: Set[Fact]) -> List[Tuple[Dict, float]]:
        """Find all variable bindings that satisfy rule antecedent."""
        if not rule.antecedent:
            return []

        # Start with first antecedent pattern
        first_pattern = rule.antecedent[0]
        matches = []

        for fact in facts:
            bindings = self._match_pattern(first_pattern, fact)
            if bindings is not None:
                # Try to extend with remaining patterns
                extended = self._extend_bindings(
                    bindings, rule.antecedent[1:], facts, fact.confidence
                )
                matches.extend(extended)

        return matches

    def _match_pattern(self, pattern: Tuple[str, str, str], fact: Fact) -> Optional[Dict]:
        """Try to match pattern against fact, returning variable bindings."""
        subj_var, pred, obj_var = pattern

        # Predicate must match exactly
        if pred != fact.predicate:
            return None

        bindings = {}

        # Subject: variable (uppercase) or constant
        if subj_var.isupper():
            bindings[subj_var] = fact.subject
        elif subj_var != fact.subject:
            return None

        # Object: variable or constant
        if obj_var.isupper():
            bindings[obj_var] = fact.object
        elif obj_var != fact.object:
            return None

        return bindings

    def _extend_bindings(self, bindings: Dict, remaining: List,
                         facts: Set[Fact], min_conf: float) -> List[Tuple[Dict, float]]:
        """Extend bindings with remaining patterns."""
        if not remaining:
            return [(bindings, min_conf)]

        pattern = remaining[0]
        results = []

        for fact in facts:
            # Apply existing bindings to pattern
            subj = bindings.get(pattern[0], pattern[0])
            obj = bindings.get(pattern[2], pattern[2])
            bound_pattern = (subj, pattern[1], obj)

            new_bindings = self._match_pattern(bound_pattern, fact)
            if new_bindings is not None:
                # Merge bindings
                merged = {**bindings, **new_bindings}
                new_min = min(min_conf, fact.confidence)
                extended = self._extend_bindings(merged, remaining[1:], facts, new_min)
                results.extend(extended)

        return results

    def generate_training_patterns(self) -> List[Dict]:
        """Generate training patterns from facts and inferences."""
        patterns = []

        # Direct facts
        for fact in self.facts:
            patterns.append({
                'input': f"What is {fact.subject}?",
                'output': fact.to_text(),
                'confidence': fact.confidence,
                'source': fact.source,
                'type': 'pln_fact'
            })

            # Location queries
            if fact.predicate == 'located_in':
                patterns.append({
                    'input': f"Where is {fact.subject}?",
                    'output': f"{fact.subject} is located in {fact.object}",
                    'confidence': fact.confidence,
                    'source': fact.source,
                    'type': 'pln_location'
                })

            # Type queries
            if fact.predicate == 'is_a':
                patterns.append({
                    'input': f"What type is {fact.subject}?",
                    'output': f"{fact.subject} is a type of {fact.object}",
                    'confidence': fact.confidence,
                    'source': fact.source,
                    'type': 'pln_type'
                })

        # Inferred facts (lower weight)
        for fact in self.inferred:
            patterns.append({
                'input': f"Is {fact.subject} related to {fact.object}?",
                'output': fact.to_text(),
                'confidence': fact.confidence,
                'source': fact.source,
                'type': 'pln_inferred'
            })

        return patterns


def explore_pln():
    """Demonstrate PLN data generation."""
    print("=" * 70)
    print("PROBABILISTIC LOGIC NETWORKS FOR DATA GENERATION")
    print("=" * 70)

    pln = ProbabilisticLogicNetwork()

    print("\n1. Loading knowledge base...")
    pln.load_knowledge_base()
    print(f"   Loaded {len(pln.facts)} base facts")

    print("\n2. Loading inference rules...")
    pln.load_inference_rules()
    print(f"   Loaded {len(pln.rules)} rules")

    print("\n3. Running forward chaining...")
    inferred = pln.forward_chain(max_iterations=5)
    print(f"   Total inferred: {len(inferred)} new facts")

    print("\n4. Sample inferences:")
    for fact in list(inferred)[:10]:
        print(f"   [{fact.confidence:.2f}] {fact.to_text()} (via {fact.source})")

    print("\n5. Generating training patterns...")
    patterns = pln.generate_training_patterns()
    print(f"   Generated {len(patterns)} training patterns")

    print("\n6. Sample patterns:")
    for p in random.sample(patterns, min(10, len(patterns))):
        print(f"   Q: {p['input']}")
        print(f"   A: {p['output']}")
        print(f"   Confidence: {p['confidence']:.2f}, Type: {p['type']}")
        print()

    return patterns


def main():
    patterns = explore_pln()

    # Save patterns
    output_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "data" / "pln_patterns.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)

    print(f"\nSaved {len(patterns)} PLN patterns to {output_path}")

    print("\n" + "=" * 70)
    print("PLN BENEFITS FOR DATA GENERATION")
    print("=" * 70)
    print("""
1. LOGICAL CONSISTENCY
   - All inferred facts follow from axioms via rules
   - No contradictions in generated data
   - Inheritance works correctly (pagerank → algorithm → has computes property)

2. CONFIDENCE PROPAGATION
   - Direct facts have high confidence (1.0)
   - Inferred facts decay (0.9^n per inference step)
   - Training can weight by confidence

3. EXPLAINABILITY
   - Each fact has source (axiom vs inferred:rule_name)
   - Can trace why a fact was generated
   - Helps debug training data quality

4. EXTENSIBILITY
   - Add new facts → automatic inference of related facts
   - Add new rules → new inference patterns
   - Composable with other techniques

COMBINING WITH OTHER TECHNIQUES:
   - PLN facts → SparkSLM completion seeds
   - PLN confidence → Woven Mind surprise calibration
   - PLN type hierarchy → Training data organization
""")


if __name__ == "__main__":
    main()
