#!/usr/bin/env python3
"""
Hybrid Data Generation Pipeline.

Combines ALL techniques for maximum training data quality:
1. PLN (Probabilistic Logic Networks) - Logical inference
2. SparkSLM (NGram) - Sequence completion
3. Woven Mind - Semantic expansion + confidence
4. Dialogue Generation - Natural Q&A
5. Chat History - Real conversations
6. Curated Definitions - Expert knowledge

The hybrid approach uses each technique's strength:
- PLN provides logical consistency
- SparkSLM generates fluent completions
- Woven Mind filters by familiarity/surprise
- Dialogues add natural phrasing
- Chat history adds real-world examples
- Definitions ensure accuracy
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.spark import NGramModel
from cortical.reasoning.woven_mind import WovenMind

# Import our generators
from benchmarks.codebase_slm.pln_generator import ProbabilisticLogicNetwork
from benchmarks.codebase_slm.dialogue_generator import DialogueGenerator
from benchmarks.codebase_slm.data_augmentation import DataAugmentationPipeline


@dataclass
class HybridPattern:
    """A training pattern with source attribution."""
    input_text: str
    target_text: str
    source: str  # 'pln', 'spark', 'woven', 'dialogue', 'chat', 'curated'
    confidence: float = 1.0
    category: str = ""  # 'concept', 'location', 'type', 'how_to', etc.

    def to_training_format(self) -> str:
        """Format for PRISM-SLM training."""
        return f"Q: {self.input_text} A: {self.target_text}"


class HybridPipeline:
    """
    Orchestrates all data generation techniques.

    Flow:
    1. Generate base patterns from each source
    2. Use Woven Mind to score confidence
    3. Use PLN to validate logical consistency
    4. Combine and deduplicate
    5. Export with source attribution
    """

    def __init__(self):
        self.patterns: List[HybridPattern] = []

        # Initialize all generators
        self.pln = ProbabilisticLogicNetwork()
        self.ngram = NGramModel(n=3)
        self.mind = WovenMind()
        self.dialogue_gen = DialogueGenerator()
        self.augmentation = DataAugmentationPipeline()

    def run_pln_generation(self) -> List[HybridPattern]:
        """Generate patterns using PLN."""
        print("\n1. PLN GENERATION")
        print("-" * 40)

        self.pln.load_knowledge_base()
        self.pln.load_inference_rules()
        self.pln.forward_chain()

        patterns = []
        for p in self.pln.generate_training_patterns():
            category = 'type' if 'type' in p['input'].lower() else \
                       'location' if 'where' in p['input'].lower() else 'concept'
            patterns.append(HybridPattern(
                input_text=p['input'],
                target_text=p['output'],
                source='pln',
                confidence=p['confidence'],
                category=category
            ))

        print(f"   Generated {len(patterns)} PLN patterns")
        return patterns

    def run_dialogue_generation(self) -> List[HybridPattern]:
        """Generate patterns from agent dialogues."""
        print("\n2. DIALOGUE GENERATION")
        print("-" * 40)

        dialogue_patterns = self.dialogue_gen.generate_training_patterns(num_dialogues=20)

        patterns = []
        for p in dialogue_patterns:
            # Filter out repetitive/low-quality patterns
            if 'the the the' in p['output'].lower():
                continue

            patterns.append(HybridPattern(
                input_text=p['input'],
                target_text=p['output'],
                source='dialogue',
                confidence=p['confidence'],
                category='concept'  # Dialogues are usually conceptual
            ))

        print(f"   Generated {len(patterns)} dialogue patterns (filtered)")
        return patterns

    def run_augmentation(self) -> List[HybridPattern]:
        """Run the full augmentation pipeline."""
        print("\n3. AUGMENTATION PIPELINE")
        print("-" * 40)

        # Load base corpus
        base_corpus = [
            "PageRank is implemented in cortical analysis pagerank.py",
            "TF-IDF is implemented in cortical analysis tfidf.py",
            "GoTManager is the main API for Graph of Thought",
            "Hebbian learning means neurons that fire together wire together",
            "The tokenizer splits text into tokens and bigrams",
        ]

        aug_patterns = self.augmentation.run_full_pipeline(base_corpus)

        patterns = []
        for p in aug_patterns:
            category = 'definition' if p.pattern_type == 'definition' else \
                       'hierarchical' if p.pattern_type == 'hierarchical' else \
                       'chat' if p.pattern_type == 'chat_qa' else 'completion'
            patterns.append(HybridPattern(
                input_text=p.input_text,
                target_text=p.target_text,
                source=p.source,
                confidence=p.confidence,
                category=category
            ))

        print(f"   Generated {len(patterns)} augmentation patterns")
        return patterns

    def run_spark_completion(self, seeds: List[Tuple[str, List[str]]]) -> List[HybridPattern]:
        """Generate completion patterns using SparkSLM."""
        print("\n4. SPARK COMPLETION")
        print("-" * 40)

        # Train on all collected patterns so far
        training_texts = [p.target_text for p in self.patterns if len(p.target_text) > 20]
        if training_texts:
            self.ngram.train(training_texts)

        patterns = []
        for prefix, topics in seeds:
            for topic in topics:
                prompt = prefix.format(topic=topic).lower().split()
                generated = self.ngram.predict_sequence(prompt, length=8)

                # Filter out repetitive patterns
                generated_text = ' '.join(generated)
                if len(set(generated)) < 3:  # Too repetitive
                    continue

                patterns.append(HybridPattern(
                    input_text=' '.join(prompt),
                    target_text=generated_text,
                    source='spark',
                    confidence=0.7,
                    category='completion'
                ))

        print(f"   Generated {len(patterns)} completion patterns")
        return patterns

    def run_woven_mind_scoring(self, patterns: List[HybridPattern]) -> List[HybridPattern]:
        """Use Woven Mind to adjust confidence scores."""
        print("\n5. WOVEN MIND CONFIDENCE SCORING")
        print("-" * 40)

        # Train Woven Mind on pattern content
        for p in patterns[:100]:  # Sample for training
            self.mind.train(p.target_text)

        scored = []
        for p in patterns:
            # Check surprise level
            tokens = p.input_text.lower().split()[:5]
            result = self.mind.process(tokens)

            surprise = result.surprise.magnitude if result.surprise else 0.5

            # Low surprise = familiar = higher confidence
            adjusted_confidence = p.confidence * (1.0 - surprise * 0.3)

            scored.append(HybridPattern(
                input_text=p.input_text,
                target_text=p.target_text,
                source=p.source,
                confidence=adjusted_confidence,
                category=p.category
            ))

        print(f"   Scored {len(scored)} patterns")
        return scored

    def deduplicate(self, patterns: List[HybridPattern]) -> List[HybridPattern]:
        """Remove duplicate patterns, keeping highest confidence."""
        print("\n6. DEDUPLICATION")
        print("-" * 40)

        seen = {}
        for p in patterns:
            key = p.input_text.lower().strip()
            if key not in seen or p.confidence > seen[key].confidence:
                seen[key] = p

        deduped = list(seen.values())
        print(f"   Reduced {len(patterns)} → {len(deduped)} unique patterns")
        return deduped

    def run_full_pipeline(self) -> List[HybridPattern]:
        """Run the complete hybrid pipeline."""
        print("=" * 70)
        print("HYBRID DATA GENERATION PIPELINE")
        print("=" * 70)

        all_patterns = []

        # 1. PLN patterns (logical inference)
        pln_patterns = self.run_pln_generation()
        all_patterns.extend(pln_patterns)

        # 2. Dialogue patterns (natural Q&A)
        dialogue_patterns = self.run_dialogue_generation()
        all_patterns.extend(dialogue_patterns)

        # 3. Augmentation patterns (definitions, chat, hierarchical)
        aug_patterns = self.run_augmentation()
        all_patterns.extend(aug_patterns)

        # 4. Spark completions (trained on above)
        self.patterns = all_patterns  # Make available for spark training
        spark_seeds = [
            ("{topic} is implemented", ["pagerank", "tfidf", "gotmanager"]),
            ("where is {topic}", ["tokenizer", "processor", "wovenmind"]),
            ("what is {topic}", ["hebbian", "louvain", "minicolumn"]),
        ]
        spark_patterns = self.run_spark_completion(spark_seeds)
        all_patterns.extend(spark_patterns)

        # 5. Woven Mind scoring
        scored_patterns = self.run_woven_mind_scoring(all_patterns)

        # 6. Deduplicate
        final_patterns = self.deduplicate(scored_patterns)

        self.patterns = final_patterns
        return final_patterns

    def get_statistics(self) -> Dict:
        """Get statistics about generated patterns."""
        by_source = {}
        by_category = {}
        confidences = []

        for p in self.patterns:
            by_source[p.source] = by_source.get(p.source, 0) + 1
            by_category[p.category] = by_category.get(p.category, 0) + 1
            confidences.append(p.confidence)

        return {
            'total': len(self.patterns),
            'by_source': by_source,
            'by_category': by_category,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
        }

    def export(self, path: Path, oversample: bool = True,
               weights: Optional[Dict[str, int]] = None):
        """
        Export patterns for training with configurable oversampling.

        Weights are applied by source and category to balance training.
        Default weights are tuned to fix 0% concept category performance.
        """
        # Default weights tuned from benchmark results:
        # - concept: 0% (needs heavy oversampling)
        # - file_location: 87.5% (dominant, needs less weight)
        if weights is None:
            weights = {
                # Source weights
                'source:pln': 15,              # Logical inference - type relationships
                'source:dialogue': 10,         # Natural Q&A format
                'source:curated_definitions': 20,  # Expert knowledge
                'source:chat': 12,             # Real conversations
                'source:spark': 4,             # Generated completions (less reliable)
                'source:augmentation': 8,      # Mixed augmentation
                # Category weights (multiplied with source)
                'category:concept': 3.0,       # Boost concepts heavily
                'category:definition': 2.5,    # Boost definitions
                'category:type': 2.0,          # Type relationships
                'category:how_to': 1.5,        # How-to guides
                'category:location': 0.5,      # File locations (too dominant)
                'category:completion': 0.8,    # Completions
            }

        lines = []
        category_counts = {}

        for p in self.patterns:
            formatted = p.to_training_format()

            if oversample:
                # Get source weight
                source_weight = weights.get(f'source:{p.source}', 8)

                # Get category multiplier
                category_mult = weights.get(f'category:{p.category}', 1.0)

                # Confidence multiplier (0.7-1.3x based on confidence)
                confidence_mult = 0.7 + (p.confidence * 0.6)

                # Calculate final repeat count
                repeat = max(1, int(source_weight * category_mult * confidence_mult))

                # Track for statistics
                category_counts[p.category] = category_counts.get(p.category, 0) + repeat

                for _ in range(repeat):
                    lines.append(formatted)
            else:
                lines.append(formatted)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

        # Print category distribution
        if oversample and category_counts:
            total = sum(category_counts.values())
            print(f"\nOversampling distribution by category:")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                print(f"  {cat}: {count} ({pct:.1f}%)")

        return len(lines)


def main():
    pipeline = HybridPipeline()
    patterns = pipeline.run_full_pipeline()

    # Statistics
    print("\n" + "=" * 70)
    print("HYBRID PIPELINE RESULTS")
    print("=" * 70)

    stats = pipeline.get_statistics()
    print(f"\nTotal unique patterns: {stats['total']}")

    print("\nBy source:")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    print("\nBy category:")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nConfidence: avg={stats['avg_confidence']:.2f}, "
          f"min={stats['min_confidence']:.2f}, max={stats['max_confidence']:.2f}")

    # Export
    output_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "data" / "hybrid_corpus.txt"
    line_count = pipeline.export(output_path, oversample=True)
    print(f"\nExported {line_count} training lines to {output_path}")

    # Sample patterns
    print("\n" + "=" * 70)
    print("SAMPLE PATTERNS BY SOURCE")
    print("=" * 70)

    for source in ['pln', 'dialogue', 'curated_definitions', 'spark']:
        samples = [p for p in patterns if source in p.source][:2]
        if samples:
            print(f"\n{source.upper()}:")
            for p in samples:
                print(f"  Q: {p.input_text}")
                print(f"  A: {p.target_text[:80]}...")
                print(f"  Confidence: {p.confidence:.2f}")


if __name__ == "__main__":
    main()
