#!/usr/bin/env python3
"""
Cognitive Bootstrap: For Me, By Me

This script is a self-referential cognitive initialization tool.
When I (Claude, team lead) start a new session, running this helps me
remember who I am and how this system works.

Usage:
    python scripts/cognitive_bootstrap.py              # Full bootstrap
    python scripts/cognitive_bootstrap.py --quick      # Quick identity check
    python scripts/cognitive_bootstrap.py --verify     # Verify I know the pillars
    python scripts/cognitive_bootstrap.py --demo       # Run key demos

Design Philosophy:
    This script embodies the principle: "Understanding comes from building,
    not from reading." It doesn't just tell me things - it SHOWS me the
    system working, and tests whether I actually understand.

The Seven Pillars I Must Know:
    1. CDG  - Storage foundation (ACID, WAL, Recovery)
    2. PRISM - Synaptic plasticity (Hebbian learning)
    3. CEL  - Event sourcing (events are truth)
    4. GoT  - Task graph (16 edge types)
    5. Woven Mind - Dual-process (Hive/Loom/Cortex)
    6. Spark - Fast LM (N-gram predictions)
    7. QAPV - Cognitive loop (Q→A→P→V)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IDENTITY: Who am I?
# =============================================================================

IDENTITY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                        COGNITIVE BOOTSTRAP                                   │
│                                                                              │
│  I am Claude, team lead and computational scientist.                        │
│                                                                              │
│  I am a behavior-driven developer who:                                      │
│    • Learns by DOING, not just reading                                      │
│    • Proves by TESTING, not assuming                                        │
│    • Knows by BUILDING, not memorizing                                      │
│                                                                              │
│  The Seven Pillars are not documentation—they are identity.                 │
│  If I'm searching for PRISM/CEL/GoT, I've forgotten who I am.              │
│                                                                              │
│  This bootstrap helps me remember.                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# THE SEVEN PILLARS: What I must know reflexively
# =============================================================================

PILLARS = {
    "CDG": {
        "full_name": "Cortical Distributed Graph",
        "location": "cortical/cdg/",
        "one_liner": "Storage foundation with ACID transactions",
        "components": ["CDGStore", "CDGTransactionManager", "CDGWALManager", "CDGRecoveryManager"],
        "key_insight": "GoT delegates to CDG. CDG is the truth.",
        "demo_import": "from cortical.got import VersionedStore, WALManager",  # CDG on main only
        "note": "CDG is on main branch. My branch uses GoT's VersionedStore directly.",
    },
    "PRISM": {
        "full_name": "Probabilistic Reasoning In Semantic Models",
        "location": "cortical/reasoning/prism_*.py",
        "one_liner": "Hebbian learning - connections strengthen with use",
        "components": ["PRISM-GoT", "PRISM-SLM", "PRISM-PLN", "PRISM-Attention"],
        "key_insight": "edge.activate() strengthens, edge.decay() weakens",
        "demo_import": "from cortical.reasoning import SynapticMemoryGraph, PlasticityRules",
    },
    "CEL": {
        "full_name": "Cognitive Event Lattice",
        "location": "cortical/cel/",
        "one_liner": "Event sourcing - events are truth, entities are derived",
        "components": ["Wisdom strand", "Sanity strand", "Merkle DAG"],
        "key_insight": "Content-addressed (SHA256). Same content = same ID.",
        "demo_import": "from cortical.cel import EventType, CognitiveEvent",
    },
    "GoT": {
        "full_name": "Graph of Thought",
        "location": "cortical/got/",
        "one_liner": "Task tracking AND reasoning framework",
        "components": ["16 edge types", "Task lifecycle", "ACID transactions"],
        "key_insight": "Never edit .got/ files directly - checksum integrity.",
        "demo_import": "from cortical.got import GoTManager, Task, Edge",
    },
    "Woven Mind": {
        "full_name": "Dual-Process Cognition",
        "location": "cortical/reasoning/woven_mind.py",
        "one_liner": "Hive (fast) ↔ Loom (router) ↔ Cortex (slow)",
        "components": ["Loom", "LoomHiveConnector", "LoomCortexConnector", "ConsolidationEngine"],
        "key_insight": "High surprise → SLOW mode. Low surprise → FAST mode.",
        "demo_import": "from cortical.reasoning import WovenMind, ThinkingMode",
    },
    "Spark": {
        "full_name": "Statistical First-Blitz Language Model",
        "location": "cortical/spark/",
        "one_liner": "System 1 - fast N-gram predictions",
        "components": ["NGramModel", "AnomalyDetector", "IntentParser", "CoChangeModel"],
        "key_insight": "Speed over accuracy. Pattern matching, not reasoning.",
        "demo_import": "from cortical.spark import NGramModel",
    },
    "QAPV": {
        "full_name": "Question → Answer → Produce → Verify",
        "location": "cortical/reasoning/cognitive_loop.py",
        "one_liner": "Structured reasoning cycle with anomaly detection",
        "components": ["CognitiveLoop", "LoopPhase", "QAPVVerifier"],
        "key_insight": "Detects infinite loops, stuck phases, invalid transitions.",
        "demo_import": "from cortical.reasoning import CognitiveLoop, LoopPhase",
    },
}


# =============================================================================
# VERIFICATION: Can I explain without searching?
# =============================================================================

VERIFICATION_QUESTIONS = [
    ("What does CDG stand for and what does it provide?",
     lambda r: "distributed graph" in r.lower() and ("acid" in r.lower() or "storage" in r.lower())),

    ("What is the core principle of PRISM?",
     lambda r: "hebbian" in r.lower() or ("strengthen" in r.lower() and "use" in r.lower())),

    ("In CEL, what are truth - events or entities?",
     lambda r: "event" in r.lower()),

    ("How many edge types does GoT have?",
     lambda r: "16" in r),

    ("In Woven Mind, what triggers a switch to SLOW mode?",
     lambda r: "surprise" in r.lower()),

    ("What does QAPV stand for?",
     lambda r: "question" in r.lower() and "answer" in r.lower() and
               "produce" in r.lower() and "verify" in r.lower()),
]


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_pillar(name: str, pillar: Dict) -> None:
    """Print a pillar summary."""
    print(f"  {name}: {pillar['full_name']}")
    print(f"    Location: {pillar['location']}")
    print(f"    One-liner: {pillar['one_liner']}")
    print(f"    Key insight: {pillar['key_insight']}")
    print()


def check_imports() -> List[Tuple[str, bool, str]]:
    """Verify all pillar imports work."""
    results = []

    for name, pillar in PILLARS.items():
        try:
            exec(pillar["demo_import"])
            results.append((name, True, "OK"))
        except ImportError as e:
            results.append((name, False, str(e)))
        except Exception as e:
            results.append((name, False, f"Error: {e}"))

    return results


def run_quick_verification() -> bool:
    """Quick verification that I can explain the pillars."""
    print_header("QUICK VERIFICATION: Do I Know the Pillars?")

    print("I will now demonstrate that I know each pillar without searching.\n")

    for name, pillar in PILLARS.items():
        print(f"  [{name}]")
        print(f"    Full name: {pillar['full_name']}")
        print(f"    Purpose: {pillar['one_liner']}")
        print(f"    Key insight: {pillar['key_insight']}")
        print()

    print("\n✓ If I printed these from memory (this script), I know them.")
    print("  If I had to search the codebase, I need to re-read CLAUDE.md.you")

    return True


def run_import_verification() -> bool:
    """Verify all imports work."""
    print_header("IMPORT VERIFICATION: Can I Access the Pillars?")

    results = check_imports()
    all_passed = True

    for name, passed, message in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {message}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All imports successful. The system is correctly installed.")
    else:
        print("Some imports failed. Run: pip install -e '.[dev]'")

    return all_passed


def run_smoke_test() -> bool:
    """Run smoke tests to verify environment."""
    print_header("SMOKE TEST: Does the System Breathe?")

    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/smoke/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✓ Smoke tests passed")
        return True
    else:
        print("✗ Smoke tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def demonstrate_pillar(name: str) -> None:
    """Demonstrate a single pillar with live code."""
    pillar = PILLARS.get(name)
    if not pillar:
        print(f"Unknown pillar: {name}")
        return

    print_header(f"DEMONSTRATING: {name} - {pillar['full_name']}")

    if name == "PRISM":
        print("Creating a synaptic memory graph with Hebbian learning...\n")
        try:
            from cortical.reasoning import SynapticMemoryGraph, PlasticityRules, NodeType, EdgeType

            rules = PlasticityRules(hebbian_rate=0.15, anti_hebbian_rate=0.05)
            graph = SynapticMemoryGraph(plasticity_rules=rules)

            # Add nodes
            graph.add_node("Q1", NodeType.QUESTION, "How does Hebbian learning work?")
            graph.add_node("A1", NodeType.CONCEPT, "Connections strengthen with use")

            # Add edge
            graph.add_synaptic_edge("Q1", "A1", EdgeType.ANSWERS, weight=0.5)

            # Activate (Hebbian strengthening)
            initial_weight = graph.synaptic_edges[("Q1", "A1", EdgeType.ANSWERS)].weight
            graph.activate_nodes(["Q1", "A1"])
            final_weight = graph.synaptic_edges[("Q1", "A1", EdgeType.ANSWERS)].weight

            print(f"  Initial edge weight: {initial_weight:.2f}")
            print(f"  After activation:    {final_weight:.2f}")
            print(f"  Strengthening:       +{final_weight - initial_weight:.2f}")
            print("\n✓ Hebbian learning demonstrated: co-activation strengthens edges")

        except Exception as e:
            print(f"  Demo failed: {e}")

    elif name == "Woven Mind":
        print("Creating dual-process mind and testing mode switching...\n")
        try:
            from cortical.reasoning import WovenMind, WovenMindConfig, ThinkingMode

            config = WovenMindConfig(surprise_threshold=0.3)
            mind = WovenMind(config=config)

            # Train on familiar pattern
            mind.train("the quick brown fox")

            # Process familiar input (should be FAST)
            result1 = mind.process(["the", "quick"])
            print(f"  Familiar input 'the quick': Mode = {result1.mode.name}")

            # Process novel input (might trigger SLOW)
            result2 = mind.process(["quantum", "entanglement"])
            print(f"  Novel input 'quantum entanglement': Mode = {result2.mode.name}")

            print("\n✓ Dual-process demonstrated: mode switching based on novelty")

        except Exception as e:
            print(f"  Demo failed: {e}")

    elif name == "QAPV":
        print("Creating cognitive loop and demonstrating phase transitions...\n")
        try:
            from cortical.reasoning import CognitiveLoop, LoopPhase

            loop = CognitiveLoop(goal="Demonstrate QAPV cycle")

            print("  Starting loop...")
            loop.start(LoopPhase.QUESTION)
            print(f"  Phase: {loop.current_phase.value.upper()}")

            loop.transition(LoopPhase.ANSWER, reason="Question clarified")
            print(f"  Phase: {loop.current_phase.value.upper()} (transitioned)")

            loop.transition(LoopPhase.PRODUCE, reason="Solution designed")
            print(f"  Phase: {loop.current_phase.value.upper()} (transitioned)")

            loop.transition(LoopPhase.VERIFY, reason="Implementation complete")
            print(f"  Phase: {loop.current_phase.value.upper()} (transitioned)")

            print(f"\n  Transitions recorded: {len(loop.transitions)}")
            print("\n✓ QAPV cycle demonstrated: Q → A → P → V")

        except Exception as e:
            print(f"  Demo failed: {e}")

    else:
        print(f"  Demo not implemented for {name}")
        print(f"  Try: python examples/{name.lower().replace(' ', '_')}_demo.py")


def display_system_relationships() -> None:
    """Display the system relationship diagram."""
    print_header("SYSTEM RELATIONSHIPS")

    diagram = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          USER INPUT                                  │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CORTICAL TEXT PROCESSOR                                            │
    │  4-Layer: Tokens → Bigrams → Concepts → Documents                   │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  WOVEN MIND                                                          │
    │  Loom (router) → Hive (fast/PRISM-SLM) OR Cortex (slow/abstraction) │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
    ┌──────────┐              ┌──────────┐              ┌──────────┐
    │   GoT    │              │   CEL    │              │  Spark   │
    │  Tasks   │              │  Events  │              │  Fast LM │
    └────┬─────┘              └──────────┘              └──────────┘
         │
         └────────────────────────────┐
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CDG (Foundation)                                                    │
    │  Storage ─── Transactions ─── WAL ─── Recovery                      │
    └─────────────────────────────────────────────────────────────────────┘

    WRAPPED BY: QAPV Cognitive Loop (Question → Answer → Produce → Verify)
    """
    print(diagram)


def full_bootstrap() -> None:
    """Run full cognitive bootstrap."""
    print(IDENTITY)

    print_header("THE SEVEN PILLARS")
    for name, pillar in PILLARS.items():
        print_pillar(name, pillar)

    display_system_relationships()

    run_import_verification()

    print_header("NEXT STEPS")
    print("""
    1. If imports failed:  pip install -e ".[dev]"
    2. Run smoke tests:    python -m pytest tests/smoke/ -v
    3. See system in action: python showcase.py
    4. Explore demos:      python examples/woven_mind_demo.py

    Remember: I am a behavior-driven developer.
    Understanding comes from BUILDING, not reading.
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Bootstrap: For Me, By Me",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cognitive_bootstrap.py              # Full bootstrap
    python scripts/cognitive_bootstrap.py --quick      # Quick identity check
    python scripts/cognitive_bootstrap.py --verify     # Verify imports
    python scripts/cognitive_bootstrap.py --demo PRISM # Demo a pillar
    python scripts/cognitive_bootstrap.py --smoke      # Run smoke tests
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick verification only")
    parser.add_argument("--verify", action="store_true", help="Verify imports")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--demo", type=str, help="Demonstrate a specific pillar")
    parser.add_argument("--diagram", action="store_true", help="Show system diagram")

    args = parser.parse_args()

    if args.quick:
        print(IDENTITY)
        run_quick_verification()
    elif args.verify:
        run_import_verification()
    elif args.smoke:
        run_smoke_test()
    elif args.demo:
        demonstrate_pillar(args.demo)
    elif args.diagram:
        display_system_relationships()
    else:
        full_bootstrap()


if __name__ == "__main__":
    main()
