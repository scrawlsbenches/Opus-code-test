#!/usr/bin/env python3
"""
PRISM-Causal Demo: Causal Reasoning in Action

This demo showcases the causal reasoning capabilities of PRISM:

1. Building causal models from domain knowledge
2. Distinguishing observation from intervention (do-calculus)
3. Counterfactual reasoning ("What if?")
4. Causal discovery from data
5. Generating causal explanations
6. Integrating with PLN probabilistic logic

"Correlation does not imply causation, but causation does imply correlation."

Run with: python examples/prism_causal_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning.prism_causal import (
    CausalGraph,
    CausalWorld,
    CausalAnalyzer,
    CausalDiscovery,
    CausalPLN,
    CausalExplainer,
)


def print_header(title: str) -> None:
    """Print a major section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_section(title: str) -> None:
    """Print a minor section header."""
    print(f"\n--- {title} ---")


def demo_intervention_vs_observation():
    """
    Demonstrate the difference between P(Y|X) and P(Y|do(X)).

    Observing vs. Intervening - the core of causal reasoning.
    """
    print_header("INTERVENTION VS OBSERVATION")

    print("""
    Consider a software system where:
    - High server load causes both slow responses AND error rates to rise
    - Slow responses themselves also cause errors (timeouts)

    Causal structure:
        server_load ─────> slow_response ─────> errors
             │                                   ↑
             └───────────────────────────────────┘
    """)

    causal = CausalGraph()

    # Build the causal model
    causal.add_cause("server_load", "slow_response", strength=0.9)
    causal.add_cause("slow_response", "errors", strength=0.7)
    causal.add_cause("server_load", "errors", strength=0.6)  # Direct effect too

    print_section("The Question")
    print("""
    We observe: When responses are slow, errors increase.
    But: Is the slow response CAUSING the errors?
         Or are both just symptoms of server load?
    """)

    # Observational query
    p_errors_given_slow = causal.observe("errors", given={"slow_response": True})
    print(f"  Observational: P(errors | slow_response) = {p_errors_given_slow:.2f}")

    # Interventional query
    p_errors_do_slow = causal.intervene("errors", do={"slow_response": True})
    print(f"  Interventional: P(errors | do(slow_response)) = {p_errors_do_slow:.2f}")

    print("""
    Interpretation:
    - Observing slow responses gives us {:.0f}% error probability
    - But if we MAKE responses slow (maybe for testing), only {:.0f}%

    The difference reveals the confounding effect of server_load!
    Observation inflates the correlation because server_load causes both.
    """.format(p_errors_given_slow * 100, p_errors_do_slow * 100))


def demo_counterfactual_reasoning():
    """
    Demonstrate counterfactual "what if" reasoning.
    """
    print_header("COUNTERFACTUAL REASONING")

    print("""
    A software deployment scenario:

        skip_tests ─────> bugs_shipped ─────> user_complaints
              │                                    ↑
              └──> quick_release ─────────────────┘
    """)

    world = CausalWorld()

    # Build the model
    world.add_cause("skip_tests", "bugs_shipped", strength=0.85)
    world.add_cause("bugs_shipped", "user_complaints", strength=0.90)
    world.add_cause("skip_tests", "quick_release", strength=0.95)
    world.add_cause("quick_release", "user_complaints", strength=0.3)

    # What actually happened
    world.observe("skip_tests", True)
    world.observe("bugs_shipped", True)
    world.observe("user_complaints", True)

    print_section("What Actually Happened")
    print("  - Tests were skipped")
    print("  - Bugs were shipped")
    print("  - Users complained")

    print_section("Counterfactual: What if we HAD run the tests?")

    cf = world.counterfactual(
        intervention={"skip_tests": False},
        query="user_complaints"
    )

    print(f"  P(user_complaints | do(NOT skip_tests)) = {cf.probability:.2f}")
    print(f"  {cf.explanation}")

    if cf.blocked_path:
        print(f"  Blocked path: {' -> '.join(cf.blocked_path)}")

    print("""
    This tells us: Running tests would have reduced complaints to {:.0f}%.

    Counterfactuals help answer: "Should we have done things differently?"
    """.format(cf.probability * 100))


def demo_necessity_sufficiency():
    """
    Demonstrate probability of necessity and sufficiency.
    """
    print_header("NECESSITY VS SUFFICIENCY")

    print("""
    Analyzing code deployment outcomes:

        code_review ─────> fewer_bugs
              │                 ↑
              │                 │
        automated_tests ────────┘
    """)

    analyzer = CausalAnalyzer()

    # Multiple causes can lead to fewer bugs
    analyzer.add_cause("code_review", "fewer_bugs", strength=0.75)
    analyzer.add_cause("automated_tests", "fewer_bugs", strength=0.80)

    print_section("Was code review NECESSARY for fewer bugs?")

    necessity = analyzer.probability_of_necessity(
        cause="code_review",
        effect="fewer_bugs",
        observed={"automated_tests": False}  # Assume no automated tests
    )

    print(f"  P(necessity | no automated tests) = {necessity:.2f}")
    print(f"  Without automated tests, code review was {necessity*100:.0f}% necessary")

    print_section("Is code review SUFFICIENT for fewer bugs?")

    sufficiency = analyzer.probability_of_sufficiency(
        cause="code_review",
        effect="fewer_bugs"
    )

    print(f"  P(sufficiency) = {sufficiency:.2f}")
    print(f"  Code review alone is {sufficiency*100:.0f}% sufficient")

    print("""
    Key insight:
    - Necessary: Would the effect have happened WITHOUT the cause?
    - Sufficient: Will the effect happen IF we do the cause?

    A cause can be necessary but not sufficient (or vice versa).
    """)


def demo_causal_discovery():
    """
    Demonstrate learning causal structure from data.
    """
    print_header("CAUSAL DISCOVERY FROM DATA")

    print("""
    We observe patterns in a production system:

    When memory_pressure is True:
        - Sometimes gc_pause is True
        - Sometimes slow_queries is True
        - gc_pause and slow_queries seem correlated...

    But is gc_pause causing slow_queries? Or vice versa?
    Or is memory_pressure causing both?
    """)

    discovery = CausalDiscovery()

    # Simulated observations from production
    observations = [
        {"memory_pressure": True, "gc_pause": True, "slow_queries": True},
        {"memory_pressure": True, "gc_pause": True, "slow_queries": False},
        {"memory_pressure": True, "gc_pause": False, "slow_queries": True},
        {"memory_pressure": True, "gc_pause": False, "slow_queries": False},
        {"memory_pressure": False, "gc_pause": False, "slow_queries": False},
        {"memory_pressure": False, "gc_pause": False, "slow_queries": False},
        # More observations showing the pattern
        {"memory_pressure": True, "gc_pause": True, "slow_queries": True},
        {"memory_pressure": True, "gc_pause": True, "slow_queries": False},
    ]

    for obs in observations:
        discovery.observe(obs)

    print_section("Inferring Causal Structure")

    structure = discovery.infer_structure()

    print("  Discovered causal relationships:")
    for cause, edges in structure._edges.items():
        for edge in edges:
            print(f"    {edge.cause} ─────> {edge.effect} (strength={edge.strength:.2f})")

    print_section("Checking for Confounders")

    # Check if memory_pressure is a confounder
    if structure.has_edge("memory_pressure", "gc_pause") and structure.has_edge("memory_pressure", "slow_queries"):
        print("  Memory pressure is a CONFOUNDER!")
        print("  It causes both gc_pause AND slow_queries.")
        print("  The correlation between gc_pause and slow_queries is spurious!")

    if not structure.has_edge("gc_pause", "slow_queries") and not structure.has_edge("slow_queries", "gc_pause"):
        print("  No direct causal link found between gc_pause and slow_queries.")
        print("  They're correlated only because of the common cause.")


def demo_causal_explanation():
    """
    Demonstrate generating causal explanations.
    """
    print_header("CAUSAL EXPLANATION")

    print("""
    Explaining why a system went down:

        traffic_spike ─> queue_overflow ─> worker_crash ─> service_down
    """)

    explainer = CausalExplainer()

    # Build the causal chain
    explainer.add_cause("traffic_spike", "queue_overflow")
    explainer.add_cause("queue_overflow", "worker_crash")
    explainer.add_cause("worker_crash", "service_down")

    print_section("Explaining: Why did the service go down?")

    explanation = explainer.explain("service_down")

    print(f"\n  Root cause(s): {', '.join(explanation.root_causes)}")
    print(f"  Proximate cause(s): {', '.join(explanation.proximate_causes)}")
    print(f"\n  Causal chain:")
    print(f"    {' → '.join(explanation.causal_chain)}")

    narrative = explanation.to_narrative()
    print(f"\n  Narrative: {narrative}")

    print("""

    This is the foundation of Root Cause Analysis (RCA):
    - Find the root cause (what started the chain)
    - Find the proximate cause (what directly caused the failure)
    - Understand the full chain of events
    """)


def demo_causal_pln_integration():
    """
    Demonstrate integration with PLN probabilistic logic.
    """
    print_header("CAUSAL + PLN INTEGRATION")

    print("""
    Combining causal reasoning with probabilistic logic:

    PLN provides: Uncertain knowledge with confidence
    Causal provides: Intervention semantics and counterfactuals

    Together: Robust reasoning under uncertainty with causal understanding.
    """)

    cpln = CausalPLN()

    # Add rules with both causal and probabilistic semantics
    cpln.add_causal_rule("high_latency", "timeout_errors", strength=0.8, confidence=0.9)
    cpln.add_causal_rule("timeout_errors", "user_churn", strength=0.6, confidence=0.7)
    cpln.add_causal_rule("competitor_launch", "user_churn", strength=0.5, confidence=0.6)

    print_section("Observational Query")

    # What's the probability of user_churn given high_latency?
    obs_result = cpln.query("user_churn", given={"high_latency": True})
    print(f"  P(user_churn | high_latency) = {obs_result.strength:.2f}")
    print(f"  Confidence: {obs_result.confidence:.2f}")
    print(f"  Has causal support: {obs_result.has_causal_support}")

    print_section("Interventional Query")

    # What if we FIX the latency problem?
    int_result = cpln.query("user_churn", do={"high_latency": False})
    print(f"  P(user_churn | do(NOT high_latency)) = {int_result.strength:.2f}")
    print(f"  Confidence: {int_result.confidence:.2f}")

    print("""

    The difference shows:
    - Observing high_latency: {:.0f}% chance of churn
    - Fixing latency: {:.0f}% chance of churn

    This tells us fixing latency could reduce churn, even with competitor pressure.
    The causal model helps distinguish what we CAN change from what we just observe.
    """.format(obs_result.strength * 100, int_result.strength * 100))


def main():
    print("\n" + "="*70)
    print("  PRISM-Causal: Causal Reasoning Demo")
    print("  Beyond Correlation to Causation")
    print("="*70)

    demo_intervention_vs_observation()
    demo_counterfactual_reasoning()
    demo_necessity_sufficiency()
    demo_causal_discovery()
    demo_causal_explanation()
    demo_causal_pln_integration()

    print_header("SUMMARY: THE POWER OF CAUSAL REASONING")

    print("""
    PRISM-Causal provides the foundation for answering:

    1. INTERVENTION (do-calculus)
       "What would happen if we MAKE X true?"
       vs "What do we observe when X is true?"

    2. COUNTERFACTUALS
       "What if we had done things differently?"
       Essential for learning from past decisions.

    3. NECESSITY & SUFFICIENCY
       "Was X necessary for Y?"
       "Is X sufficient for Y?"
       Critical for root cause analysis.

    4. CAUSAL DISCOVERY
       "What causes what?"
       Learning structure from observational data.

    5. CAUSAL EXPLANATION
       "Why did Y happen?"
       Tracing chains from root to proximate causes.

    6. PLN INTEGRATION
       Combining probabilistic uncertainty with causal semantics.

    This enables PRISM to reason about:
    - What actions to take (not just what to predict)
    - What would have happened under different choices
    - Why systems behave the way they do
    - How to attribute outcomes to causes

    "The first principle is that you must not fool yourself - and you are
    the easiest person to fool." - Richard Feynman

    Causal reasoning helps us not fool ourselves about cause and effect.
    """)


if __name__ == "__main__":
    main()
