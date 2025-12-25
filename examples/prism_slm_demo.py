#!/usr/bin/env python3
"""
PRISM-SLM: Statistical Language Model with Synaptic Learning Demo

This demo shows how PRISM-SLM learns from corpus text and generates
new text using biologically-inspired synaptic learning.

Key features:
- Transitions between words strengthen with repeated use (Hebbian learning)
- Unused transitions decay over time
- Temperature controls generation randomness
- Reward reinforces good generation paths

Run with: python examples/prism_slm_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning import PRISMLanguageModel


def load_corpus(samples_dir: Path) -> list:
    """Load text files from corpus."""
    texts = []
    for pattern in ["*.txt", "*.md"]:
        for f in samples_dir.glob(pattern):
            try:
                content = f.read_text(encoding="utf-8")
                if len(content) > 100:  # Skip tiny files
                    texts.append((f.name, content))
            except:
                pass
    return texts


def main():
    print("\n" + "=" * 60)
    print("  PRISM-SLM: Statistical Language Model with Synaptic Learning")
    print("=" * 60)

    samples_dir = Path(__file__).parent.parent / "samples"

    # Create model
    model = PRISMLanguageModel(context_size=3)

    print("\n1. TRAINING ON CORPUS")
    print("-" * 40)

    texts = load_corpus(samples_dir)
    print(f"   Found {len(texts)} text files")

    for name, content in texts:
        model.train(content)
        print(f"   Trained on: {name[:40]}...")

    stats = model.get_stats()
    print(f"\n   Model Statistics:")
    print(f"   - Vocabulary: {stats['vocab_size']:,} tokens")
    print(f"   - Transitions: {stats['transition_count']:,}")
    print(f"   - Total tokens seen: {stats['token_count']:,}")

    print("\n2. TEXT GENERATION")
    print("-" * 40)

    prompts = [
        "The neural",
        "Memory and",
        "Graph algorithms",
        "Machine learning",
        "The system",
    ]

    for prompt in prompts:
        print(f"\n   Prompt: \"{prompt}\"")

        # Generate with different temperatures
        for temp in [0.5, 1.0, 1.5]:
            generated = model.generate(
                prompt=prompt,
                max_tokens=12,
                temperature=temp,
            )
            print(f"   T={temp}: {generated}")

    print("\n3. PERPLEXITY EVALUATION")
    print("-" * 40)

    test_sentences = [
        "The neural network learns patterns.",
        "Memory consolidation occurs during sleep.",
        "Xyzzy foobar completely random nonsense words.",
    ]

    for sentence in test_sentences:
        perplexity = model.perplexity(sentence)
        label = "✓ Low" if perplexity < 50 else "⚠ High"
        print(f"   [{label:6}] PPL={perplexity:6.1f} | {sentence}")

    print("\n4. HEBBIAN LEARNING DEMONSTRATION")
    print("-" * 40)

    # Create a fresh model for this demo
    demo_model = PRISMLanguageModel(context_size=2)

    # Train on repeated phrase
    phrase = "the cat sat on the mat"
    print(f"   Training on: \"{phrase}\" (10 times)")

    for _ in range(10):
        demo_model.train(phrase)

    # Check transition strength
    transitions = demo_model.graph.get_transitions(("the",))
    print(f"\n   Transitions from 'the':")
    for trans in sorted(transitions, key=lambda t: -t.weight)[:5]:
        print(f"   - 'the' → '{trans.to_token}': weight={trans.weight:.1f}, count={trans.count}")

    # Apply decay
    print(f"\n   Applying decay (factor=0.8)...")
    demo_model.apply_decay(factor=0.8)

    transitions = demo_model.graph.get_transitions(("the",))
    print(f"\n   After decay:")
    for trans in sorted(transitions, key=lambda t: -t.weight)[:3]:
        print(f"   - 'the' → '{trans.to_token}': weight={trans.weight:.1f}")

    print("\n5. REWARD LEARNING")
    print("-" * 40)

    # Generate and reward
    result = demo_model.generate(prompt="the cat", max_tokens=5, return_path=True)
    path = result["path"]
    print(f"   Generated: {result['text']}")
    print(f"   Path: {path}")

    if len(path) >= 2:
        # Get weight before reward
        ctx = (path[0],)
        before_trans = demo_model.graph.get_transitions(ctx)
        before_weight = next(
            (t.weight for t in before_trans if t.to_token == path[1]), 0
        )

        # Apply positive reward
        demo_model.reward_path(path, reward=2.0)

        # Get weight after reward
        after_trans = demo_model.graph.get_transitions(ctx)
        after_weight = next(
            (t.weight for t in after_trans if t.to_token == path[1]), 0
        )

        print(f"\n   Rewarding path with +2.0:")
        print(f"   '{path[0]}' → '{path[1]}' weight: {before_weight:.2f} → {after_weight:.2f}")

    print("\n6. INTERACTIVE GENERATION")
    print("-" * 40)
    print("   Enter a prompt to generate text (or 'quit' to exit):\n")

    while True:
        try:
            prompt = input("   Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break

        generated = model.generate(prompt=prompt, max_tokens=20, temperature=1.0)
        print(f"   → {generated}\n")

    print("\n" + "=" * 60)
    print("   Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
