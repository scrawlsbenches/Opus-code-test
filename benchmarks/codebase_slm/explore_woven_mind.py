#!/usr/bin/env python3
"""
Explore Woven Mind Capabilities for Repository Knowledge.

Woven Mind is a dual-process cognitive architecture that:
- FAST mode (Hive): Pattern matching for familiar inputs
- SLOW mode (Cortex): Deliberate abstraction for novel inputs

This script explores whether Woven Mind can:
1. Learn repository-specific patterns
2. Distinguish familiar vs novel queries
3. Build abstractions from observations
4. Provide better answers than pure n-gram approaches

Usage:
    python -m benchmarks.codebase_slm.explore_woven_mind
"""

import sys
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
from cortical.reasoning.loom import ThinkingMode


def create_training_corpus() -> List[str]:
    """Create a focused training corpus for repository knowledge."""
    return [
        # File locations (repeat for learning)
        "PageRank is implemented in cortical analysis pagerank py",
        "TF-IDF is implemented in cortical analysis tfidf py",
        "GoTManager is defined in cortical got api py",
        "The tokenizer is in cortical tokenizer py",
        "Clustering is in cortical analysis clustering py",
        "Query expansion is in cortical query expansion py",

        # Concepts
        "Hebbian learning means neurons that fire together wire together",
        "PRISM is a statistical language model for prediction",
        "Woven Mind is a dual process cognitive architecture",
        "GoT is Graph of Thought for task and decision tracking",
        "BM25 is a scoring algorithm optimized for code search",

        # How-to patterns
        "To create a task use python scripts got_utils.py task create",
        "To run tests use pytest or make test",
        "To search the codebase use python scripts search_codebase.py",
        "To index the codebase use python scripts index_codebase.py",

        # Import patterns
        "from cortical.got import GoTManager",
        "from cortical.processor import CorticalTextProcessor",
        "from cortical.analysis import compute_pagerank compute_tfidf",

        # Architecture patterns
        "The processor has four layers tokens bigrams concepts documents",
        "Layer 0 is TOKENS for individual words",
        "Layer 1 is BIGRAMS for word pairs",
        "Layer 2 is CONCEPTS for semantic clusters",
        "Layer 3 is DOCUMENTS for full documents",

        # Process patterns
        "Work priority order is Security Bugs Features Documentation",
        "TDD means Test Driven Development with red green refactor",
        "Always write tests first before implementing code",
    ]


def tokenize(text: str) -> List[str]:
    """Simple tokenizer."""
    return text.lower().split()


def explore_woven_mind():
    """Explore Woven Mind capabilities."""
    print("=" * 70)
    print("EXPLORING WOVEN MIND FOR REPOSITORY KNOWLEDGE")
    print("=" * 70)

    # Create Woven Mind with lower surprise threshold
    config = WovenMindConfig(
        surprise_threshold=0.2,  # Lower threshold for more SLOW mode
        k_winners=3,
        min_frequency=2,
        auto_switch=True,
        enable_auto_consolidation=True,
    )
    mind = WovenMind(config=config)

    # Train on corpus
    print("\n1. TRAINING ON REPOSITORY KNOWLEDGE")
    print("-" * 40)
    corpus = create_training_corpus()
    print(f"Training on {len(corpus)} patterns...")

    for text in corpus:
        mind.train(text)
        # Also observe pattern for abstraction
        mind.observe_pattern(tokenize(text))

    # Repeat training for stronger learning
    for _ in range(5):
        for text in corpus:
            mind.train(text)

    print(f"Training complete!")

    # Test queries
    print("\n2. TESTING QUERY PROCESSING")
    print("-" * 40)

    test_queries = [
        # Familiar patterns (should use FAST mode)
        (["pagerank", "implemented"], "File location query"),
        (["cortical", "got", "import"], "Import completion"),
        (["hebbian", "learning"], "Concept query"),

        # Novel patterns (should use SLOW mode)
        (["database", "migration"], "Novel concept"),
        (["authentication", "security"], "Novel security query"),
        (["performance", "optimization"], "Novel performance query"),

        # Mixed patterns
        (["pagerank", "performance"], "Mixed familiar/novel"),
        (["got", "database"], "Mixed familiar/novel"),
    ]

    for context, description in test_queries:
        result = mind.process(context)
        mode_str = "FAST ⚡" if result.mode == ThinkingMode.FAST else "SLOW 🐢"

        print(f"\nQuery: {context}")
        print(f"  Description: {description}")
        print(f"  Mode: {mode_str}")
        print(f"  Source: {result.source}")
        print(f"  Activations: {len(result.activations)} nodes")
        if result.surprise:
            print(f"  Surprise: {result.surprise.magnitude:.3f}")
        if result.predictions:
            top_preds = sorted(result.predictions.items(), key=lambda x: -x[1])[:5]
            print(f"  Top predictions: {top_preds}")

    # Test abstraction building
    print("\n3. ABSTRACTION BUILDING")
    print("-" * 40)

    # Observe patterns multiple times to build abstractions
    abstraction_patterns = [
        ["cortical", "analysis", "algorithm"],
        ["cortical", "query", "search"],
        ["cortical", "got", "task"],
    ]

    print("Observing patterns for abstraction...")
    for _ in range(5):
        for pattern in abstraction_patterns:
            abstractions = mind.observe_pattern(pattern)
            if abstractions:
                print(f"  Pattern {pattern} activated: {abstractions}")

    # Run consolidation
    print("\n4. CONSOLIDATION (Memory Transfer)")
    print("-" * 40)

    print("Running consolidation cycle...")
    consolidation_result = mind.consolidate()
    print(f"  Patterns transferred: {consolidation_result.patterns_transferred}")
    print(f"  Abstractions formed: {consolidation_result.abstractions_formed}")
    print(f"  Connections decayed: {consolidation_result.connections_decayed}")
    print(f"  Connections pruned: {consolidation_result.connections_pruned}")

    # Compare mode switching behavior
    print("\n5. MODE SWITCHING ANALYSIS")
    print("-" * 40)

    mode_counts = Counter()
    surprise_levels = []

    # Test many queries to see switching behavior
    test_words = [
        # Familiar
        "pagerank", "tfidf", "clustering", "tokenizer", "hebbian",
        "prism", "got", "task", "cortical", "analysis",
        # Novel
        "kubernetes", "docker", "microservices", "graphql", "redis",
        "elasticsearch", "kafka", "mongodb", "postgres", "nginx",
    ]

    for word in test_words:
        result = mind.process([word])
        mode_counts[result.mode.name] += 1
        if result.surprise:
            surprise_levels.append((word, result.surprise.magnitude))

    print(f"Mode distribution:")
    for mode, count in mode_counts.items():
        pct = count / len(test_words) * 100
        bar = "█" * int(pct / 5)
        print(f"  {mode}: {count}/{len(test_words)} ({pct:.0f}%) {bar}")

    print(f"\nHighest surprise words:")
    for word, level in sorted(surprise_levels, key=lambda x: -x[1])[:5]:
        print(f"  {word}: {level:.3f}")

    print(f"\nLowest surprise words:")
    for word, level in sorted(surprise_levels, key=lambda x: x[1])[:5]:
        print(f"  {word}: {level:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("WOVEN MIND EXPLORATION SUMMARY")
    print("=" * 70)
    print("""
Key Observations:
1. Woven Mind uses FAST mode for familiar patterns (trained concepts)
2. SLOW mode activates for novel/unfamiliar inputs (high surprise)
3. Abstractions form through repeated pattern observation
4. Consolidation transfers frequent patterns from Hive to Cortex

For Repository-Native SLM:
- Woven Mind excels at CLASSIFYING familiar vs novel queries
- It can route queries to appropriate processing modes
- But it doesn't directly GENERATE answers like PRISM-SLM
- Hybrid approach: Use Woven Mind for routing, PRISM for generation

Recommended Architecture:
  Query → Woven Mind (classify) → FAST: PRISM-SLM generation
                                → SLOW: Semantic search/retrieval
""")


if __name__ == '__main__':
    explore_woven_mind()
