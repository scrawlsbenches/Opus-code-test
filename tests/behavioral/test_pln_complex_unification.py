"""
Behavioral tests for PLN Complex Unification.

These tests define the target behavior for advanced pattern matching,
inspired by Prolog's unification with extensions for PLN's probabilistic context.

Complex unification enables:
1. Compound terms - nested structures like file(name, metadata(churn, age))
2. Multi-argument predicates - relates(X, Y, relation_type)
3. Type constraints - X:File means X must be a File type
4. Proper Robinson's unification algorithm

For audit tooling:
- Query complex relationships: "files with high churn AND security issues"
- Match nested metadata: file(auth.py, metrics(churn(high), age(old)))
- Type-safe queries: "all File entities that need review"
"""

import pytest
from typing import List, Dict, Optional


class TestCompoundTerms:
    """
    Tests for compound (nested) term structures.
    """

    def test_simple_compound_term_creation(self):
        """Compound terms can be created and parsed."""
        from cortical.reasoning.prism_pln import Term, parse_term

        # Simple compound term
        term = parse_term("file(auth.py)")

        assert term.functor == "file"
        assert len(term.args) == 1
        # Args are Term objects
        assert term.args[0].functor == "auth.py"
        assert not term.is_variable

    def test_nested_compound_term(self):
        """Compound terms can contain other compound terms."""
        from cortical.reasoning.prism_pln import parse_term

        # Nested structure: file with metadata
        term = parse_term("file(auth.py, metadata(high_churn, old))")

        assert term.functor == "file"
        assert len(term.args) == 2
        # First arg is a Term representing an atom
        assert term.args[0].functor == "auth.py"
        # Second arg should be a nested term
        assert hasattr(term.args[1], 'functor')
        assert term.args[1].functor == "metadata"

    def test_deeply_nested_terms(self):
        """Terms can be arbitrarily nested."""
        from cortical.reasoning.prism_pln import parse_term

        # Deeply nested: issue(file(name), severity(level(critical)))
        term = parse_term("issue(file(auth.py), severity(level(critical)))")

        assert term.functor == "issue"
        assert term.args[0].functor == "file"
        assert term.args[1].functor == "severity"
        assert term.args[1].args[0].functor == "level"
        # Innermost arg is a Term representing an atom
        assert term.args[1].args[0].args[0].functor == "critical"

    def test_compound_term_with_variables(self):
        """Compound terms can contain variables."""
        from cortical.reasoning.prism_pln import parse_term

        term = parse_term("needs_review(X, reason(Y))")

        assert term.functor == "needs_review"
        assert term.args[0].is_variable
        assert term.args[0].name == "X"
        assert term.args[1].functor == "reason"
        assert term.args[1].args[0].is_variable

    def test_compound_term_to_string(self):
        """Compound terms can be converted back to string representation."""
        from cortical.reasoning.prism_pln import parse_term

        original = "file(auth.py, metadata(high, old))"
        term = parse_term(original)
        reconstructed = str(term)

        # Should reconstruct to equivalent string
        assert "file" in reconstructed
        assert "auth.py" in reconstructed
        assert "metadata" in reconstructed


class TestUnificationBasics:
    """
    Tests for basic unification operations.
    """

    def test_unify_identical_atoms(self):
        """Identical atoms unify with empty substitution."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("auth.py")
        t2 = parse_term("auth.py")

        result = unify(t1, t2)

        assert result is not None
        assert result == {}  # Empty substitution

    def test_unify_different_atoms_fails(self):
        """Different atoms do not unify."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("auth.py")
        t2 = parse_term("config.py")

        result = unify(t1, t2)

        assert result is None  # Unification fails

    def test_unify_variable_with_atom(self):
        """Variable unifies with any atom."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("X")
        t2 = parse_term("auth.py")

        result = unify(t1, t2)

        assert result is not None
        assert result["X"] == "auth.py"

    def test_unify_atom_with_variable(self):
        """Atom unifies with variable (symmetric)."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("auth.py")
        t2 = parse_term("Y")

        result = unify(t1, t2)

        assert result is not None
        assert result["Y"] == "auth.py"

    def test_unify_two_variables(self):
        """Two variables unify by binding one to the other."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("X")
        t2 = parse_term("Y")

        result = unify(t1, t2)

        assert result is not None
        # One variable bound to the other
        assert "X" in result or "Y" in result

    def test_unify_same_variable(self):
        """Same variable unifies with itself."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("X")
        t2 = parse_term("X")

        result = unify(t1, t2)

        assert result is not None
        # Should succeed (trivially)


class TestCompoundUnification:
    """
    Tests for unifying compound terms.
    """

    def test_unify_simple_compound_terms(self):
        """Compound terms with same functor and arity unify."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("file(auth.py)")
        t2 = parse_term("file(auth.py)")

        result = unify(t1, t2)

        assert result is not None
        assert result == {}

    def test_unify_compound_with_variable_arg(self):
        """Compound term with variable argument unifies."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("file(X)")
        t2 = parse_term("file(auth.py)")

        result = unify(t1, t2)

        assert result is not None
        assert result["X"] == "auth.py"

    def test_unify_nested_compounds(self):
        """Nested compound terms unify recursively."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("issue(file(X), severity(Y))")
        t2 = parse_term("issue(file(auth.py), severity(high))")

        result = unify(t1, t2)

        assert result is not None
        assert result["X"] == "auth.py"
        assert result["Y"] == "high"

    def test_unify_different_functors_fails(self):
        """Compound terms with different functors don't unify."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("file(auth.py)")
        t2 = parse_term("directory(src)")

        result = unify(t1, t2)

        assert result is None

    def test_unify_different_arity_fails(self):
        """Compound terms with different arities don't unify."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("file(auth.py)")
        t2 = parse_term("file(auth.py, old)")

        result = unify(t1, t2)

        assert result is None

    def test_unify_multiple_occurrences_of_variable(self):
        """Same variable appearing multiple times must unify consistently."""
        from cortical.reasoning.prism_pln import unify, parse_term

        # X must be the same in both positions
        t1 = parse_term("pair(X, X)")
        t2 = parse_term("pair(auth.py, auth.py)")

        result = unify(t1, t2)

        assert result is not None
        assert result["X"] == "auth.py"

    def test_unify_inconsistent_variable_fails(self):
        """Same variable with inconsistent bindings fails."""
        from cortical.reasoning.prism_pln import unify, parse_term

        # X can't be both auth.py and config.py
        t1 = parse_term("pair(X, X)")
        t2 = parse_term("pair(auth.py, config.py)")

        result = unify(t1, t2)

        assert result is None


class TestOccursCheck:
    """
    Tests for the occurs check (prevents infinite structures).
    """

    def test_occurs_check_prevents_infinite_structure(self):
        """Unifying X with f(X) should fail (occurs check)."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("X")
        t2 = parse_term("contains(X)")

        result = unify(t1, t2)

        # Should fail due to occurs check
        assert result is None

    def test_occurs_check_nested(self):
        """Occurs check works with deeply nested terms."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("X")
        t2 = parse_term("outer(inner(X))")

        result = unify(t1, t2)

        assert result is None


class TestTypeConstraints:
    """
    Tests for type-constrained variables.
    """

    def test_typed_variable_creation(self):
        """Variables can have type constraints."""
        from cortical.reasoning.prism_pln import parse_term

        term = parse_term("X:File")

        assert term.is_variable
        assert term.name == "X"
        assert term.type_constraint == "File"

    def test_typed_variable_in_compound(self):
        """Typed variables work in compound terms."""
        from cortical.reasoning.prism_pln import parse_term

        term = parse_term("needs_review(F:File, R:Reason)")

        assert term.functor == "needs_review"
        assert term.args[0].is_variable
        assert term.args[0].type_constraint == "File"
        assert term.args[1].type_constraint == "Reason"

    def test_type_constraint_unification_success(self):
        """Typed variable unifies with compatible value."""
        from cortical.reasoning.prism_pln import (
            unify, parse_term, TypeRegistry
        )

        registry = TypeRegistry()
        registry.register_type("File", ["auth.py", "config.py", "main.py"])

        t1 = parse_term("F:File")
        t2 = parse_term("auth.py")

        result = unify(t1, t2, type_registry=registry)

        assert result is not None
        assert result["F"] == "auth.py"

    def test_type_constraint_unification_fails(self):
        """Typed variable fails to unify with incompatible value."""
        from cortical.reasoning.prism_pln import (
            unify, parse_term, TypeRegistry
        )

        registry = TypeRegistry()
        registry.register_type("File", ["auth.py", "config.py"])

        t1 = parse_term("F:File")
        t2 = parse_term("not_a_file")  # Not in File type

        result = unify(t1, t2, type_registry=registry)

        assert result is None

    def test_type_hierarchy(self):
        """Types can have subtypes for hierarchical matching."""
        from cortical.reasoning.prism_pln import (
            unify, parse_term, TypeRegistry
        )

        registry = TypeRegistry()
        registry.register_type("Entity", ["auth.py", "src/"])
        registry.register_subtype("File", "Entity", ["auth.py"])
        registry.register_subtype("Directory", "Entity", ["src/"])

        # File is a subtype of Entity, so this should work
        t1 = parse_term("E:Entity")
        t2 = parse_term("auth.py")

        result = unify(t1, t2, type_registry=registry)

        assert result is not None
        assert result["E"] == "auth.py"


class TestPLNReasonerComplexQueries:
    """
    Tests for complex queries using the PLN reasoner.
    """

    def test_query_with_compound_fact(self):
        """Can assert and query compound facts."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Assert compound fact
        reasoner.assert_compound_fact(
            "file_info(auth.py, metrics(high_churn, security_critical))",
            strength=0.95
        )

        # Query exact match
        result = reasoner.query_compound(
            "file_info(auth.py, metrics(high_churn, security_critical))"
        )

        assert result is not None
        assert result.strength > 0.9

    def test_query_compound_with_variables(self):
        """Can query compound facts with variables."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Multiple files with different metrics
        reasoner.assert_compound_fact(
            "file_info(auth.py, metrics(high, critical))",
            strength=0.95
        )
        reasoner.assert_compound_fact(
            "file_info(config.py, metrics(low, normal))",
            strength=0.90
        )

        # Query for any file with high churn
        results = reasoner.query_compound(
            "file_info(X, metrics(high, Y))"
        )

        assert len(results) >= 1
        # Should find auth.py
        bindings = [r.bindings for r in results]
        assert any(b.get("X") == "auth.py" for b in bindings)

    def test_rule_with_compound_terms(self):
        """Rules can use compound terms."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Rule: files with high churn AND security flag need immediate review
        reasoner.assert_compound_rule(
            "file_info(X, metrics(high_churn, security_critical))",
            "needs_immediate_review(X)",
            strength=0.95
        )

        # Fact
        reasoner.assert_compound_fact(
            "file_info(auth.py, metrics(high_churn, security_critical))",
            strength=0.99
        )

        # Query should fire the rule
        result = reasoner.query_compound("needs_immediate_review(auth.py)")

        assert result is not None
        assert result.strength > 0.8

    def test_complex_audit_scenario(self):
        """
        Full audit scenario with compound terms and type constraints.

        Scenario: Find all security-critical files with high churn
        that need review.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, TypeRegistry

        reasoner = PLNReasoner()

        # Setup type registry
        registry = TypeRegistry()
        registry.register_type("File", ["auth.py", "payment.py", "utils.py"])
        registry.register_type("Severity", ["critical", "high", "medium", "low"])
        reasoner.set_type_registry(registry)

        # Rules
        reasoner.assert_compound_rule(
            "has_issue(F:File, severity(critical))",
            "needs_immediate_review(F)",
            strength=0.95
        )
        reasoner.assert_compound_rule(
            "has_issue(F:File, severity(high))",
            "needs_review(F)",
            strength=0.80
        )

        # Facts
        reasoner.assert_compound_fact(
            "has_issue(auth.py, severity(critical))",
            strength=0.99
        )
        reasoner.assert_compound_fact(
            "has_issue(payment.py, severity(high))",
            strength=0.95
        )
        reasoner.assert_compound_fact(
            "has_issue(utils.py, severity(low))",
            strength=0.90
        )

        # Helper to resolve binding chains
        def resolve_binding(bindings, var):
            """Resolve variable through binding chain."""
            value = bindings.get(var)
            while value in bindings:
                value = bindings[value]
            return value

        # Query for immediate review
        immediate = reasoner.query_compound("needs_immediate_review(X)")
        assert len(immediate) >= 1
        # X may be bound to F which is bound to auth.py
        assert any(
            resolve_binding(r.bindings, "X") == "auth.py" or
            resolve_binding(r.bindings, "F") == "auth.py"
            for r in immediate
        )

        # Query for regular review
        regular = reasoner.query_compound("needs_review(X)")
        assert len(regular) >= 1
        assert any(
            resolve_binding(r.bindings, "X") == "payment.py" or
            resolve_binding(r.bindings, "F") == "payment.py"
            for r in regular
        )


class TestSubstitutionApplication:
    """
    Tests for applying substitutions to terms.
    """

    def test_apply_substitution_to_variable(self):
        """Substitution replaces variable with value."""
        from cortical.reasoning.prism_pln import parse_term, apply_substitution

        term = parse_term("X")
        subst = {"X": "auth.py"}

        result = apply_substitution(term, subst)

        assert str(result) == "auth.py"

    def test_apply_substitution_to_compound(self):
        """Substitution applies to variables in compound terms."""
        from cortical.reasoning.prism_pln import parse_term, apply_substitution

        term = parse_term("file(X, status(Y))")
        subst = {"X": "auth.py", "Y": "needs_review"}

        result = apply_substitution(term, subst)

        assert "auth.py" in str(result)
        assert "needs_review" in str(result)

    def test_apply_substitution_preserves_unbound(self):
        """Unbound variables remain unchanged."""
        from cortical.reasoning.prism_pln import parse_term, apply_substitution

        term = parse_term("file(X, Y)")
        subst = {"X": "auth.py"}  # Y not bound

        result = apply_substitution(term, subst)

        assert "auth.py" in str(result)
        # Y should still be a variable
        assert result.args[1].is_variable


class TestUnificationAPI:
    """
    Tests for the unification API surface.
    """

    def test_term_equality(self):
        """Terms can be compared for equality."""
        from cortical.reasoning.prism_pln import parse_term

        t1 = parse_term("file(auth.py)")
        t2 = parse_term("file(auth.py)")
        t3 = parse_term("file(config.py)")

        assert t1 == t2
        assert t1 != t3

    def test_term_hashing(self):
        """Terms can be used in sets/dicts."""
        from cortical.reasoning.prism_pln import parse_term

        t1 = parse_term("file(auth.py)")
        t2 = parse_term("file(auth.py)")

        term_set = {t1, t2}

        assert len(term_set) == 1  # Same term

    def test_unification_with_existing_substitution(self):
        """Unification can extend an existing substitution."""
        from cortical.reasoning.prism_pln import unify, parse_term

        t1 = parse_term("pair(X, Y)")
        t2 = parse_term("pair(auth.py, config.py)")

        # Start with partial substitution
        initial_subst = {"Z": "other.py"}

        result = unify(t1, t2, initial_subst)

        assert result is not None
        assert result["X"] == "auth.py"
        assert result["Y"] == "config.py"
        assert result["Z"] == "other.py"  # Preserved
