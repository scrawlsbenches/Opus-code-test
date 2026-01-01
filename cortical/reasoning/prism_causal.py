"""
PRISM-Causal: Causal Reasoning with Probabilistic Inference.

Implements causal inference capabilities:
- CausalGraph: DAG-based causal model with do-calculus
- CausalWorld: Counterfactual reasoning ("What if?")
- CausalAnalyzer: Necessity/sufficiency analysis
- CausalDiscovery: Learning causal structure from data
- CausalPLN: Integration with PLN probabilistic reasoning
- CausalExplainer: Generate human-readable causal explanations

Key concepts:
- Intervention (do-calculus): P(Y|do(X)) differs from P(Y|X)
- Counterfactuals: Reasoning about alternative histories
- Causal chains: Tracing effects through cause-effect paths

"Correlation does not imply causation, but causation does imply correlation."

Example:
    from cortical.reasoning.prism_causal import CausalGraph

    causal = CausalGraph()
    causal.add_cause("drink_bottle", "shrink", strength=0.95)
    causal.add_cause("shrink", "fit_door", strength=0.99)

    # Interventional query: What if we MAKE someone drink?
    p_fit = causal.intervene("fit_door", do={"drink_bottle": True})
    print(f"P(fit_door | do(drink)) = {p_fit:.2f}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import math

from cortical.reasoning.prism_pln import TruthValue, PLNReasoner, deduce


@dataclass
class CausalTruthValue(TruthValue):
    """TruthValue extended with causal support information."""
    has_causal_support: bool = False
    causal_mechanism: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["has_causal_support"] = self.has_causal_support
        data["causal_mechanism"] = self.causal_mechanism
        return data


# =============================================================================
# CAUSAL EDGE
# =============================================================================

@dataclass
class CausalEdge:
    """
    A directed causal edge from cause to effect.

    Attributes:
        cause: The causing variable
        effect: The affected variable
        strength: Probability that cause produces effect P(effect|do(cause))
        mechanism: Optional description of the causal mechanism
    """
    cause: str
    effect: str
    strength: float = 0.9
    mechanism: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "mechanism": self.mechanism
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalEdge":
        return cls(
            cause=data["cause"],
            effect=data["effect"],
            strength=data.get("strength", 0.9),
            mechanism=data.get("mechanism")
        )


# =============================================================================
# CAUSAL GRAPH
# =============================================================================

class CausalGraph:
    """
    Directed Acyclic Graph representing causal relationships.

    Supports:
    - Observational queries: P(Y|X) - what we expect to see
    - Interventional queries: P(Y|do(X)) - what happens if we intervene
    - Total effect computation through causal chains
    """

    def __init__(self):
        # Adjacency list: cause -> [(effect, strength)]
        self._edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        # Reverse adjacency: effect -> [cause]
        self._parents: Dict[str, List[str]] = defaultdict(list)
        # All variables
        self._variables: Set[str] = set()
        # Causal mechanisms
        self._mechanisms: Dict[Tuple[str, str], str] = {}
        # Base probabilities (priors)
        self._priors: Dict[str, float] = defaultdict(lambda: 0.5)

    def add_cause(
        self,
        cause: str,
        effect: str,
        strength: float = 0.9,
        mechanism: Optional[str] = None
    ) -> None:
        """
        Add a causal relationship: cause -> effect.

        Args:
            cause: The causing variable
            effect: The affected variable
            strength: P(effect | do(cause)) - causal strength
            mechanism: Optional description of the mechanism
        """
        edge = CausalEdge(cause, effect, strength, mechanism)
        self._edges[cause].append(edge)
        self._parents[effect].append(cause)
        self._variables.add(cause)
        self._variables.add(effect)

        if mechanism:
            self._mechanisms[(cause, effect)] = mechanism

    def add_mechanism(self, cause: str, effect: str, mechanism: str) -> None:
        """Add a description of the causal mechanism."""
        self._mechanisms[(cause, effect)] = mechanism

    def set_prior(self, variable: str, probability: float) -> None:
        """Set the prior probability of a variable."""
        self._priors[variable] = probability
        self._variables.add(variable)

    def get_parents(self, variable: str) -> List[str]:
        """Get all direct causes of a variable."""
        return self._parents.get(variable, [])

    def get_children(self, variable: str) -> List[str]:
        """Get all direct effects of a variable."""
        return [e.effect for e in self._edges.get(variable, [])]

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestors (indirect causes) of a variable."""
        ancestors = set()
        queue = list(self._parents.get(variable, []))

        while queue:
            parent = queue.pop(0)
            if parent not in ancestors:
                ancestors.add(parent)
                queue.extend(self._parents.get(parent, []))

        return ancestors

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all descendants (indirect effects) of a variable."""
        descendants = set()
        queue = self.get_children(variable)

        while queue:
            child = queue.pop(0)
            if child not in descendants:
                descendants.add(child)
                queue.extend(self.get_children(child))

        return descendants

    def has_edge(self, cause: str, effect: str) -> bool:
        """Check if there's a direct causal edge from cause to effect."""
        return effect in self.get_children(cause)

    def _get_edge_strength(self, cause: str, effect: str) -> float:
        """Get the causal strength from cause to effect."""
        for edge in self._edges.get(cause, []):
            if edge.effect == effect:
                return edge.strength
        return 0.0

    def observe(
        self,
        target: str,
        given: Optional[Dict[str, bool]] = None
    ) -> float:
        """
        Observational probability P(target | given).

        This is CONFOUNDED by common causes - it's what we observe,
        not what we'd see if we intervened.

        Args:
            target: Variable to query
            given: Observed conditions

        Returns:
            P(target | given) - observational probability
        """
        given = given or {}

        # Base case: no parents, return prior
        parents = self.get_parents(target)
        if not parents:
            return self._priors[target]

        # For observational queries, conditioning creates selection bias
        # when there are common causes (confounders)

        # Find ancestors of target and given variables
        target_ancestors = self.get_ancestors(target)

        # Check for confounders (common ancestors of given var and target)
        # Confounding happens when there's a back-door path through a common cause
        confounding_boost = 0.0
        for var, val in given.items():
            if not val:
                continue  # Only consider when var=True

            # Get ancestors of the given variable
            var_ancestors = self.get_ancestors(var)

            # Find common ancestors that create back-door paths
            for common in var_ancestors:
                # Check if this common ancestor also affects target
                # through a path NOT going through var
                if common in target_ancestors:
                    # Common ancestor found - check for back-door path
                    # A back-door path exists if common -> target not through var
                    paths_to_target = self._find_all_paths(common, target)
                    for path in paths_to_target:
                        if var not in path:
                            # Back-door path! Confounding exists
                            path_strength = 1.0
                            for i in range(len(path) - 1):
                                path_strength *= self._get_edge_strength(path[i], path[i + 1])
                            # Observing var gives info about common cause
                            common_to_var = self.total_effect(common, var)
                            confounding_boost += path_strength * common_to_var * 0.3
                            break

        # Base probability from causal structure
        prob = self._compute_causal_probability(target, given)

        # Apply confounding (observation != intervention)
        # Confounding inflates observed probability
        prob = min(1.0, prob + confounding_boost)

        return prob

    def intervene(
        self,
        target: str,
        do: Optional[Dict[str, bool]] = None
    ) -> float:
        """
        Interventional probability P(target | do(variables)).

        do-calculus: We surgically set variables, breaking their
        natural causes. This removes confounding.

        Args:
            target: Variable to query
            do: Variables to intervene on (set to specific values)

        Returns:
            P(target | do(variables)) - interventional probability
        """
        do = do or {}

        # Intervention breaks incoming edges to intervened variables
        # Build mutilated graph

        if target in do:
            # Direct intervention on target
            return 1.0 if do[target] else 0.0

        # Compute P(target) in mutilated graph
        return self._compute_causal_probability(target, do, intervention=True)

    def _compute_causal_probability(
        self,
        target: str,
        conditions: Dict[str, bool],
        intervention: bool = False
    ) -> float:
        """
        Compute probability of target given conditions.

        For interventions, incoming edges to conditioned variables are cut.
        """
        # If target is directly conditioned
        if target in conditions:
            return 1.0 if conditions[target] else 0.0

        parents = self.get_parents(target)

        if not parents:
            return self._priors[target]

        # For each parent, compute their contribution
        total_strength = 0.0
        active_parents = 0

        for parent in parents:
            edge_strength = self._get_edge_strength(parent, target)

            if parent in conditions:
                # Parent is set (observed or intervened)
                if conditions[parent]:
                    total_strength += edge_strength
                    active_parents += 1
            else:
                # Parent has its own causes - recurse
                parent_prob = self._compute_causal_probability(
                    parent, conditions, intervention
                )
                total_strength += edge_strength * parent_prob
                active_parents += 1

        if active_parents == 0:
            return self._priors[target]

        # Combine using noisy-OR model
        # P(effect) = 1 - prod(1 - strength_i * P(cause_i))
        prob = total_strength / max(active_parents, 1)

        return min(1.0, max(0.0, prob))

    def total_effect(self, cause: str, effect: str) -> float:
        """
        Compute the total causal effect of cause on effect.

        This traces through all causal paths and computes the
        combined effect strength.

        Args:
            cause: The intervention variable
            effect: The outcome variable

        Returns:
            Total causal effect (product of strengths along paths)
        """
        # Find all paths from cause to effect
        paths = self._find_all_paths(cause, effect)

        if not paths:
            return 0.0

        # For single path, multiply strengths
        if len(paths) == 1:
            path = paths[0]
            total = 1.0
            for i in range(len(path) - 1):
                total *= self._get_edge_strength(path[i], path[i + 1])
            return total

        # For multiple paths, use noisy-OR combination
        # P(effect) = 1 - prod(1 - path_strength_i)
        path_effects = []
        for path in paths:
            path_strength = 1.0
            for i in range(len(path) - 1):
                path_strength *= self._get_edge_strength(path[i], path[i + 1])
            path_effects.append(path_strength)

        # Noisy-OR: probability of effect through at least one path
        prob_no_effect = 1.0
        for pe in path_effects:
            prob_no_effect *= (1.0 - pe)

        return 1.0 - prob_no_effect

    def _find_all_paths(
        self,
        source: str,
        target: str,
        visited: Optional[Set[str]] = None
    ) -> List[List[str]]:
        """Find all directed paths from source to target."""
        if visited is None:
            visited = set()

        if source == target:
            return [[source]]

        if source in visited:
            return []

        visited = visited | {source}
        paths = []

        for child in self.get_children(source):
            sub_paths = self._find_all_paths(child, target, visited)
            for path in sub_paths:
                paths.append([source] + path)

        return paths

    def is_d_separated(
        self,
        x: str,
        y: str,
        z: Optional[Set[str]] = None
    ) -> bool:
        """
        Check if X and Y are d-separated given Z.

        D-separation is the graphical criterion for conditional independence.
        If X ⊥ Y | Z in the graph, then X ⊥ Y | Z in any distribution
        that the graph represents.
        """
        z = z or set()

        # Find all paths between X and Y
        paths = self._find_undirected_paths(x, y)

        for path in paths:
            if not self._is_path_blocked(path, z):
                return False

        return True

    def _find_undirected_paths(
        self,
        source: str,
        target: str,
        visited: Optional[Set[str]] = None
    ) -> List[List[str]]:
        """Find all undirected paths (ignoring edge direction)."""
        if visited is None:
            visited = set()

        if source == target:
            return [[source]]

        if source in visited:
            return []

        visited = visited | {source}
        paths = []

        # Get both children and parents (undirected neighbors)
        neighbors = set(self.get_children(source)) | set(self.get_parents(source))

        for neighbor in neighbors:
            sub_paths = self._find_undirected_paths(neighbor, target, visited)
            for path in sub_paths:
                paths.append([source] + path)

        return paths

    def _is_path_blocked(self, path: List[str], z: Set[str]) -> bool:
        """Check if a path is blocked by conditioning set Z."""
        if len(path) < 3:
            # Path of length 1 or 2 is always unblocked
            return False

        for i in range(1, len(path) - 1):
            # Check the triplet (path[i-1], path[i], path[i+1])
            left = path[i - 1]
            middle = path[i]
            right = path[i + 1]

            is_collider = (
                middle in self._parents.get(left, []) and
                middle in self._parents.get(right, [])
            )

            if is_collider:
                # Collider: blocked unless middle or descendant in Z
                descendants = self.get_descendants(middle)
                if middle not in z and not (descendants & z):
                    return True
            else:
                # Non-collider: blocked if middle in Z
                if middle in z:
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        edges = []
        for cause, edge_list in self._edges.items():
            for edge in edge_list:
                edges.append(edge.to_dict())

        return {
            "edges": edges,
            "priors": dict(self._priors),
            "mechanisms": {f"{k[0]}->{k[1]}": v for k, v in self._mechanisms.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        """Deserialize from dictionary."""
        graph = cls()

        for edge_data in data.get("edges", []):
            edge = CausalEdge.from_dict(edge_data)
            graph.add_cause(edge.cause, edge.effect, edge.strength, edge.mechanism)

        for var, prob in data.get("priors", {}).items():
            graph.set_prior(var, prob)

        for key, mechanism in data.get("mechanisms", {}).items():
            parts = key.split("->")
            if len(parts) == 2:
                graph._mechanisms[(parts[0], parts[1])] = mechanism

        return graph


# =============================================================================
# CAUSAL WORLD (COUNTERFACTUALS)
# =============================================================================

@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""
    probability: float
    blocked_path: List[str] = field(default_factory=list)
    active_path: List[str] = field(default_factory=list)
    explanation: str = ""


class CausalWorld:
    """
    Causal model with actual observations for counterfactual reasoning.

    Counterfactuals answer "What if?" questions:
    - What would have happened if X had been different?
    - What if we had taken a different action?

    Based on Pearl's three-step counterfactual algorithm:
    1. Abduction: Use evidence to determine exogenous variables
    2. Action: Modify the model to reflect intervention
    3. Prediction: Compute outcome in modified model
    """

    def __init__(self):
        self._graph = CausalGraph()
        self._observations: Dict[str, bool] = {}

    def add_cause(
        self,
        cause: str,
        effect: str,
        strength: float = 0.9
    ) -> None:
        """Add a causal relationship."""
        self._graph.add_cause(cause, effect, strength)

    def observe(self, variable: str, value: bool) -> None:
        """Record an actual observation."""
        self._observations[variable] = value

    def counterfactual(
        self,
        intervention: Dict[str, bool],
        query: str
    ) -> CounterfactualResult:
        """
        Compute a counterfactual query.

        Given what actually happened, what would have happened
        if we had done intervention instead?

        Uses Pearl's three-step algorithm:
        1. Abduction: Infer exogenous variables from observations
        2. Action: Modify the model for the intervention
        3. Prediction: Compute the query in the modified model

        Args:
            intervention: Variables to set counterfactually
            query: Variable to query

        Returns:
            CounterfactualResult with probability and path info
        """
        # Find paths affected by intervention
        blocked = []
        active = []

        for var, val in intervention.items():
            if var in self._observations and self._observations[var] != val:
                # This intervention changes the actual world
                paths = self._graph._find_all_paths(var, query)
                if paths:
                    if val:
                        # Activating a cause that was off
                        active.extend(paths[0])
                    else:
                        # Deactivating a cause that was on - blocks the path
                        blocked.extend(paths[0])

        # Compute counterfactual probability
        # Key insight: if we intervene to set a cause to False,
        # and that cause was necessary for the effect, the effect won't happen

        # Check if any intervened variable blocks all paths to query
        all_paths_blocked = False

        for var, val in intervention.items():
            if not val and self._observations.get(var, False):
                # We're turning off something that was on
                # Check if this was on the causal path to query
                if query in self._graph.get_descendants(var):
                    # This variable causally affects query
                    # Check if there are alternative paths

                    # Get all paths from roots to query
                    ancestors = self._graph.get_ancestors(query)
                    roots = [a for a in ancestors if not self._graph.get_parents(a)]

                    # Check if any path doesn't go through var
                    has_alternative = False
                    for root in roots:
                        paths = self._graph._find_all_paths(root, query)
                        for path in paths:
                            if var not in path:
                                # Alternative path exists
                                has_alternative = True
                                break
                        if has_alternative:
                            break

                    if not has_alternative:
                        # No alternative path - blocking var blocks query
                        all_paths_blocked = True
                        break

        if all_paths_blocked:
            # Turning off the necessary cause blocks the effect
            prob = 0.0
        else:
            # Compute probability through remaining paths
            prob = self._graph.intervene(query, do=intervention)

            # If we're activating alternative paths
            for var, val in intervention.items():
                if val and not self._observations.get(var, False):
                    # We're turning on something that was off
                    effect = self._graph.total_effect(var, query)
                    if effect > 0:
                        prob = max(prob, effect)

        return CounterfactualResult(
            probability=prob,
            blocked_path=blocked,
            active_path=active,
            explanation=self._generate_explanation(intervention, query, prob)
        )

    def _generate_explanation(
        self,
        intervention: Dict[str, bool],
        query: str,
        probability: float
    ) -> str:
        """Generate human-readable explanation of counterfactual."""
        if probability > 0.5:
            outcome = f"{query} would have occurred"
        else:
            outcome = f"{query} would NOT have occurred"

        changes = ", ".join(
            f"{v}={'True' if val else 'False'}"
            for v, val in intervention.items()
        )

        return f"If {changes}, then {outcome} (P={probability:.2f})"


# =============================================================================
# CAUSAL ANALYZER
# =============================================================================

class CausalAnalyzer:
    """
    Analyze causal relationships for necessity and sufficiency.

    - Probability of Necessity (PN): Was X necessary for Y?
      "Would Y have occurred without X?"

    - Probability of Sufficiency (PS): Is X sufficient for Y?
      "Would Y occur if we did X?"
    """

    def __init__(self):
        self._graph = CausalGraph()

    def add_cause(
        self,
        cause: str,
        effect: str,
        strength: float = 0.9
    ) -> None:
        """Add a causal relationship."""
        self._graph.add_cause(cause, effect, strength)

    def probability_of_necessity(
        self,
        cause: str,
        effect: str,
        observed: Optional[Dict[str, bool]] = None
    ) -> float:
        """
        Compute P(Y_x'=0 | X=1, Y=1) - the probability of necessity.

        "Given that X and Y both occurred, would Y have NOT occurred
        if X had NOT occurred?"

        High PN means X was necessary for Y.

        Args:
            cause: The potential necessary cause
            effect: The effect that occurred
            observed: Additional observations

        Returns:
            Probability that cause was necessary for effect
        """
        observed = observed or {}

        # Given: cause and effect both occurred
        # Query: Would effect have occurred without cause?

        # Find all paths from cause to effect
        causal_paths = self._graph._find_all_paths(cause, effect)

        if not causal_paths:
            # cause doesn't affect effect causally
            return 0.0

        # Find alternative paths (paths to effect not going through cause)
        ancestors = self._graph.get_ancestors(effect)
        roots = [a for a in ancestors if not self._graph.get_parents(a)]
        roots = [r for r in roots if r != cause]  # Exclude cause as root

        alternative_prob = 0.0

        for root in roots:
            paths = self._graph._find_all_paths(root, effect)
            for path in paths:
                if cause not in path:
                    # This is an alternative path
                    # Check if it's blocked by observed variables
                    path_blocked = False
                    for var, val in observed.items():
                        if var in path and not val:
                            path_blocked = True
                            break

                    if not path_blocked:
                        # Compute effect through this path
                        path_strength = 1.0
                        for i in range(len(path) - 1):
                            path_strength *= self._graph._get_edge_strength(path[i], path[i + 1])
                        alternative_prob = max(alternative_prob, path_strength)

        # PN = probability effect wouldn't have happened without cause
        # If there are no alternative paths, necessity is 1.0
        # If there are alternative paths, necessity decreases
        necessity = 1.0 - alternative_prob

        return necessity

    def probability_of_sufficiency(
        self,
        cause: str,
        effect: str
    ) -> float:
        """
        Compute P(Y_x=1 | X=0, Y=0) - the probability of sufficiency.

        "Given that X and Y both did NOT occur, would Y have occurred
        if X HAD occurred?"

        High PS means X is sufficient for Y.

        Args:
            cause: The potential sufficient cause
            effect: The effect to analyze

        Returns:
            Probability that cause is sufficient for effect
        """
        # Probability of sufficiency = total causal effect
        # when neither cause nor effect are present

        return self._graph.total_effect(cause, effect)


# =============================================================================
# CAUSAL DISCOVERY
# =============================================================================

class CausalDiscovery:
    """
    Learn causal structure from observational data.

    Uses constraint-based methods to infer causal direction
    and detect hidden confounders.
    """

    def __init__(self):
        self._observations: List[Dict[str, bool]] = []
        self._variables: Set[str] = set()

    def observe(self, observation: Dict[str, bool]) -> None:
        """Add an observation (assignment of values to variables)."""
        self._observations.append(observation)
        self._variables.update(observation.keys())

    def infer_structure(self) -> CausalGraph:
        """
        Infer causal structure from accumulated observations.

        Uses:
        1. Correlation analysis to find associated variables
        2. Temporal/asymmetry heuristics for direction
        3. Conditional independence tests for confounders

        Returns:
            Inferred CausalGraph
        """
        graph = CausalGraph()

        if len(self._observations) < 2:
            return graph

        variables = list(self._variables)

        # Step 1: Compute correlations and conditional probabilities
        correlations = {}
        conditional_probs = {}

        for i, v1 in enumerate(variables):
            for v2 in variables[i + 1:]:
                corr = self._compute_correlation(v1, v2)
                if abs(corr) > 0.05:  # Lower threshold for weak associations
                    correlations[(v1, v2)] = corr

                    # Compute P(v2|v1) and P(v1|v2)
                    p_v2_given_v1 = self._conditional_probability(v2, v1)
                    p_v1_given_v2 = self._conditional_probability(v1, v2)
                    conditional_probs[(v1, v2)] = (p_v2_given_v1, p_v1_given_v2)

        # Step 2: Determine causal direction using multiple heuristics
        for (v1, v2), corr in correlations.items():
            p_v2_v1, p_v1_v2 = conditional_probs.get((v1, v2), (0.5, 0.5))

            # Heuristic 1: Cause occurs without effect more than vice versa
            direction = self._infer_direction(v1, v2)

            # Heuristic 2: If P(v1|v2)=1.0, v2 never happens without v1
            # This means v1 is necessary for v2, suggesting v1 -> v2
            if abs(direction) < 1:
                if p_v1_v2 >= 0.99 and p_v2_v1 < 0.99:
                    # v2 never happens without v1 -> v1 causes v2
                    direction = 1
                elif p_v2_v1 >= 0.99 and p_v1_v2 < 0.99:
                    # v1 never happens without v2 -> v2 causes v1
                    direction = -1
                elif p_v2_v1 > p_v1_v2 + 0.2:
                    # v2 is more likely given v1 than v1 given v2
                    # This is consistent with v1 -> v2
                    direction = 1
                elif p_v1_v2 > p_v2_v1 + 0.2:
                    direction = -1

            # Heuristic 3: Use occurrence asymmetry as tiebreaker
            if direction == 0:
                direction = self._infer_direction(v1, v2)
                if direction == 0 and corr > 0:
                    direction = 1  # Default to first variable as cause

            if direction > 0:
                strength = max(abs(corr), p_v2_v1)
                graph.add_cause(v1, v2, min(strength, 1.0))
            elif direction < 0:
                strength = max(abs(corr), p_v1_v2)
                graph.add_cause(v2, v1, min(strength, 1.0))

        # Step 3: Check for hidden confounders using conditional independence
        # If X and Y are correlated but become independent given Z,
        # then Z is a common cause (confounder) and X-Y edge should be removed
        edges_to_remove = []
        for (v1, v2), corr in correlations.items():
            # Check if conditioning on a third variable makes them independent
            for z in variables:
                if z == v1 or z == v2:
                    continue

                # Test conditional independence: X ⊥ Y | Z
                if self._conditionally_independent(v1, v2, z):
                    # Z confounds the relationship
                    # Add edges from Z to both v1 and v2
                    if not graph.has_edge(z, v1):
                        graph.add_cause(z, v1, 0.8)
                    if not graph.has_edge(z, v2):
                        graph.add_cause(z, v2, 0.8)

                    # Remove spurious edge between v1 and v2
                    if graph.has_edge(v1, v2):
                        edges_to_remove.append((v1, v2))
                    if graph.has_edge(v2, v1):
                        edges_to_remove.append((v2, v1))
                    break

        # Remove spurious edges
        for cause, effect in edges_to_remove:
            if cause in graph._edges:
                graph._edges[cause] = [e for e in graph._edges[cause] if e.effect != effect]
                if not graph._edges[cause]:
                    del graph._edges[cause]
            if effect in graph._parents:
                graph._parents[effect] = [p for p in graph._parents[effect] if p != cause]

        return graph

    def _conditionally_independent(self, v1: str, v2: str, given: str) -> bool:
        """
        Test if v1 and v2 are conditionally independent given a third variable.

        For small samples, uses a simpler heuristic: if both v1 and v2 are
        strongly correlated with 'given' but weakly correlated with each other,
        and 'given' precedes both in occurrence patterns, they're likely confounded.
        """
        # For small samples, use a simpler heuristic
        # Check if 'given' is a better predictor of both v1 and v2 than they are of each other

        # Correlation with confounder
        corr_v1_given = self._compute_correlation(v1, given)
        corr_v2_given = self._compute_correlation(v2, given)
        corr_v1_v2 = self._compute_correlation(v1, v2)

        # If both are strongly correlated with 'given' but weakly with each other
        strong_with_given = abs(corr_v1_given) > 0.5 and abs(corr_v2_given) > 0.5
        weak_with_each_other = abs(corr_v1_v2) < 0.4

        if strong_with_given and weak_with_each_other:
            # Check that 'given' precedes both in the sense that
            # given=True is necessary for both v1 and v2 to be True
            p_given_v1 = self._conditional_probability(given, v1)  # P(given|v1)
            p_given_v2 = self._conditional_probability(given, v2)  # P(given|v2)

            # If given is always true when v1 or v2 is true, given is likely the cause
            if p_given_v1 >= 0.99 and p_given_v2 >= 0.99:
                return True

        return False

    def _conditional_probability(self, target: str, given: str) -> float:
        """Compute P(target=True | given=True)."""
        given_true = 0
        both_true = 0

        for obs in self._observations:
            if given in obs and obs[given]:
                given_true += 1
                if target in obs and obs[target]:
                    both_true += 1

        if given_true == 0:
            return 0.0
        return both_true / given_true

    def _compute_correlation(self, v1: str, v2: str) -> float:
        """Compute correlation between two variables."""
        both_true = 0
        both_false = 0
        v1_only = 0
        v2_only = 0
        total = 0

        for obs in self._observations:
            if v1 in obs and v2 in obs:
                total += 1
                if obs[v1] and obs[v2]:
                    both_true += 1
                elif not obs[v1] and not obs[v2]:
                    both_false += 1
                elif obs[v1]:
                    v1_only += 1
                else:
                    v2_only += 1

        if total == 0:
            return 0.0

        # Phi coefficient (for binary variables)
        n11, n00, n10, n01 = both_true, both_false, v1_only, v2_only

        n1_ = n11 + n10
        n0_ = n01 + n00
        n_1 = n11 + n01
        n_0 = n10 + n00

        denom = math.sqrt(n1_ * n0_ * n_1 * n_0) if n1_ * n0_ * n_1 * n_0 > 0 else 1

        return (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0

    def _infer_direction(self, v1: str, v2: str) -> int:
        """
        Infer causal direction between correlated variables.

        Returns:
            1 if v1 -> v2
            -1 if v2 -> v1
            0 if undetermined
        """
        # Heuristic 1: The cause is more "stable"
        # (appears without effect more often than effect without cause)

        v1_without_v2 = sum(
            1 for obs in self._observations
            if obs.get(v1, False) and not obs.get(v2, False)
        )

        v2_without_v1 = sum(
            1 for obs in self._observations
            if obs.get(v2, False) and not obs.get(v1, False)
        )

        if v1_without_v2 > v2_without_v1 + 1:
            # v1 occurs without v2 more often -> v1 is likely cause
            return 1
        elif v2_without_v1 > v1_without_v2 + 1:
            # v2 occurs without v1 more often -> v2 is likely cause
            return -1

        return 0

    def _detect_confounders(self, graph: CausalGraph) -> None:
        """Detect and add hidden confounders to the graph."""
        # Look for variables that are correlated but have no direct edge
        # and both are caused by a third variable

        variables = list(self._variables)

        for i, v1 in enumerate(variables):
            for v2 in variables[i + 1:]:
                # Check if correlated
                corr = self._compute_correlation(v1, v2)

                if abs(corr) > 0.3:
                    # Check if there's already an edge
                    has_edge = (
                        v2 in graph.get_children(v1) or
                        v1 in graph.get_children(v2)
                    )

                    if not has_edge:
                        # Look for common cause
                        for v3 in variables:
                            if v3 != v1 and v3 != v2:
                                if (v1 in graph.get_children(v3) and
                                    v2 in graph.get_children(v3)):
                                    # v3 is a common cause (confounder)
                                    # Already in graph, no action needed
                                    pass

    def has_edge(self, cause: str, effect: str) -> bool:
        """Check if there's a causal edge from cause to effect."""
        graph = self.infer_structure()
        return effect in graph.get_children(cause)


# =============================================================================
# CAUSAL PLN INTEGRATION
# =============================================================================

class CausalPLN:
    """
    Integration of causal reasoning with PLN probabilistic logic.

    Combines:
    - PLN's uncertain knowledge representation (TruthValues)
    - Causal graph's intervention semantics (do-calculus)
    """

    def __init__(
        self,
        pln: Optional[PLNReasoner] = None,
        causal: Optional[CausalGraph] = None
    ):
        self.pln = pln or PLNReasoner()
        self.causal = causal or CausalGraph()

    def add_causal_rule(
        self,
        cause: str,
        effect: str,
        strength: float = 0.9,
        confidence: float = 0.9
    ) -> None:
        """
        Add a causal rule to both PLN and causal graph.

        This ensures both systems are synchronized.
        """
        # Add to PLN as implication
        self.pln.assert_rule(cause, effect, strength, confidence)

        # Add to causal graph
        self.causal.add_cause(cause, effect, strength)

    def query(
        self,
        target: str,
        given: Optional[Dict[str, bool]] = None,
        do: Optional[Dict[str, bool]] = None
    ) -> CausalTruthValue:
        """
        Query with both observational and interventional semantics.

        Args:
            target: Variable to query
            given: Observational conditions (passive)
            do: Interventional conditions (active)

        Returns:
            CausalTruthValue combining PLN inference with causal reasoning
        """
        given = given or {}

        # Check for causal support
        has_causal = len(self.causal._edges) > 0
        mechanism = None
        for (cause, effect), mech in self.causal._mechanisms.items():
            if effect == target:
                mechanism = mech
                break

        if do:
            # Interventional query - use causal graph with do-calculus
            # This breaks incoming edges to intervened variables
            prob = self.causal.intervene(target, do=do)

            # Combine with PLN confidence
            pln_result = self.pln.query(target)
            confidence = pln_result.confidence if pln_result else 0.5

            return CausalTruthValue(
                strength=prob,
                confidence=confidence * 0.9,
                has_causal_support=has_causal,
                causal_mechanism=mechanism
            )
        else:
            # Observational query
            if given:
                # Use causal graph for observation (includes confounding)
                prob = self.causal.observe(target, given=given)

                # Boost confidence and strength if we have causal support
                pln_result = self.pln.query(target)
                confidence = pln_result.confidence if pln_result else 0.5

                # Causal knowledge boosts the probability
                if has_causal:
                    for var, val in given.items():
                        if val:
                            causal_effect = self.causal.total_effect(var, target)
                            if causal_effect > 0:
                                prob = max(prob, causal_effect)

                return CausalTruthValue(
                    strength=prob,
                    confidence=confidence,
                    has_causal_support=has_causal,
                    causal_mechanism=mechanism
                )
            else:
                # Pure PLN query, but with causal info
                result = self.pln.query(target)
                if result:
                    return CausalTruthValue(
                        strength=result.strength,
                        confidence=result.confidence,
                        has_causal_support=has_causal,
                        causal_mechanism=mechanism
                    )
                return CausalTruthValue(has_causal_support=has_causal)


# =============================================================================
# CAUSAL EXPLAINER
# =============================================================================

@dataclass
class CausalExplanation:
    """Explanation of why something happened."""
    root_causes: List[str]
    proximate_causes: List[str]
    causal_chain: List[str]
    mechanisms: Dict[str, str]

    def to_narrative(self) -> str:
        """Generate human-readable narrative."""
        if not self.causal_chain:
            return "No causal explanation found."

        parts = []

        if self.root_causes:
            parts.append(f"The root cause was {', '.join(self.root_causes)}")

        if len(self.causal_chain) > 1:
            chain_str = " led to ".join(self.causal_chain)
            parts.append(f"This led to the following chain: {chain_str}")

        if self.proximate_causes:
            parts.append(
                f"The immediate cause was {', '.join(self.proximate_causes)}"
            )

        return ". ".join(parts) + "." if parts else "No explanation."


class CausalExplainer:
    """
    Generate causal explanations for events.

    Traces back through causal chains to identify:
    - Root causes (initial triggers)
    - Proximate causes (immediate predecessors)
    - Causal mechanisms (how causes produce effects)
    """

    def __init__(self):
        self._graph = CausalGraph()

    def add_cause(
        self,
        cause: str,
        effect: str,
        mechanism: Optional[str] = None
    ) -> None:
        """Add a causal relationship."""
        self._graph.add_cause(cause, effect, mechanism=mechanism)

    def explain(self, target: str) -> CausalExplanation:
        """
        Generate an explanation for why target occurred.

        Traces back through the causal graph to find root causes
        and the chain of events leading to the target.

        Args:
            target: The event to explain

        Returns:
            CausalExplanation with root causes, proximate causes, and chain
        """
        # Find proximate (direct) causes
        proximate_causes = self._graph.get_parents(target)

        # Find root causes (causes with no parents)
        root_causes = []
        all_ancestors = self._graph.get_ancestors(target)

        for ancestor in all_ancestors:
            if not self._graph.get_parents(ancestor):
                root_causes.append(ancestor)

        # Build causal chain from a root to target
        causal_chain = []
        if root_causes:
            # Find longest path from any root to target
            for root in root_causes:
                paths = self._graph._find_all_paths(root, target)
                for path in paths:
                    if len(path) > len(causal_chain):
                        causal_chain = path
        elif proximate_causes:
            causal_chain = [proximate_causes[0], target]
        else:
            causal_chain = [target]

        # Collect mechanisms
        mechanisms = {}
        for cause, mechanism in self._graph._mechanisms.items():
            if cause[1] in causal_chain or cause[0] in causal_chain:
                mechanisms[f"{cause[0]}->{cause[1]}"] = mechanism

        return CausalExplanation(
            root_causes=root_causes,
            proximate_causes=proximate_causes,
            causal_chain=causal_chain,
            mechanisms=mechanisms
        )
