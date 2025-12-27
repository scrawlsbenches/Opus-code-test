#!/usr/bin/env python3
"""
Data Augmentation Pipeline for Repository-Native SLM.

Combines multiple strategies:
1. SparkSLM sequence generation
2. Woven Mind semantic expansion + confidence filtering
3. Concept definition patterns (to fix 0% concept score)
4. Hierarchical abstraction-based patterns
5. Chat history ingestion
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.spark import NGramModel
from cortical.reasoning.woven_mind import WovenMind


@dataclass
class AugmentedPattern:
    """A generated training pattern with metadata."""
    input_text: str
    target_text: str
    pattern_type: str  # 'completion', 'definition', 'hierarchical', 'chat_qa'
    confidence: float = 1.0
    source: str = ""  # Where this pattern came from

    def to_training_format(self) -> str:
        """Format for PRISM-SLM training."""
        if self.pattern_type == 'definition':
            return f"Q: What is {self.input_text}? A: {self.target_text}"
        elif self.pattern_type == 'chat_qa':
            return f"Q: {self.input_text} A: {self.target_text}"
        else:
            return f"{self.input_text} {self.target_text}"


class DataAugmentationPipeline:
    """
    Multi-strategy data augmentation for Repository-Native SLM.
    """

    def __init__(self):
        self.ngram = NGramModel(n=3)
        self.mind = WovenMind()
        self.patterns: List[AugmentedPattern] = []
        self.concept_definitions: Dict[str, str] = {}

    def train_base_models(self, texts: List[str]):
        """Train both underlying models on base corpus."""
        self.ngram.train(texts)
        for text in texts:
            self.mind.train(text)

    def load_concept_definitions(self):
        """Load concept definitions from knowledge base and codebase."""

        # Core concepts from the codebase (manually curated for quality)
        self.concept_definitions = {
            # Algorithms
            "pagerank": "PageRank is a graph algorithm that computes importance scores by analyzing link structure, where nodes with more incoming links from important nodes rank higher",
            "tfidf": "TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is to a document by multiplying term frequency by inverse document frequency across the corpus",
            "bm25": "BM25 is a ranking function that improves on TF-IDF by adding term frequency saturation and document length normalization for better search relevance",
            "louvain": "Louvain is a community detection algorithm that finds clusters by optimizing modularity, grouping nodes that are more densely connected to each other than to the rest of the network",

            # Core concepts
            "hebbian learning": "Hebbian learning is the principle that neurons that fire together wire together, meaning connections between co-activated neurons strengthen over time",
            "minicolumn": "A Minicolumn is the core data structure that stores a term with its connections, TF-IDF score, PageRank value, and document associations",
            "lateral connections": "Lateral connections are weighted links between terms in the same layer based on co-occurrence, representing semantic relationships",
            "typed connections": "Typed connections are semantic edges with explicit relationship types like IS_A, HAS_PROPERTY, or CAUSES between terms",

            # Components
            "gotmanager": "GoTManager is the main API for Graph of Thought, providing task creation, decision logging, sprint management, and dependency tracking",
            "woven mind": "Woven Mind is a dual-process cognitive architecture with FAST mode (Hive) for familiar patterns and SLOW mode (Cortex) for novel reasoning",
            "sparksm": "SparkSLM is a statistical n-gram language model for fast predictions and completions based on learned sequence patterns",
            "prism": "PRISM is the Pattern-Recognition Information Synthesis Model that combines multiple micro-models for intelligent text processing",

            # Architecture
            "cortical layers": "Cortical layers organize text hierarchically: Layer 0 for tokens, Layer 1 for bigrams, Layer 2 for concepts, Layer 3 for documents",
            "tokenizer": "The tokenizer splits text into tokens, applies stemming, removes stop words, and generates n-grams for analysis",
            "persistence": "Persistence handles saving and loading processor state in JSON format for reproducibility and caching",
            "query expansion": "Query expansion improves search recall by adding semantically related terms to the original query using lateral connections",

            # Processes
            "consolidation": "Consolidation is a sleep-like process that transfers frequent patterns from the fast Hive to the slow Cortex for long-term storage",
            "activation spreading": "Activation spreading propagates signal through the connection network, simulating how related concepts become active together",
            "staleness tracking": "Staleness tracking monitors which computations need rerunning after document changes to avoid unnecessary recomputation",
        }

    def generate_definition_patterns(self) -> List[AugmentedPattern]:
        """Generate patterns specifically for concept explanations (to fix 0% concept score)."""
        patterns = []

        for concept, definition in self.concept_definitions.items():
            # Multiple Q&A formats for each concept
            patterns.extend([
                AugmentedPattern(
                    input_text=concept,
                    target_text=definition,
                    pattern_type='definition',
                    confidence=1.0,
                    source='curated_definitions'
                ),
                AugmentedPattern(
                    input_text=f"explain {concept}",
                    target_text=definition,
                    pattern_type='chat_qa',
                    confidence=1.0,
                    source='curated_definitions'
                ),
                AugmentedPattern(
                    input_text=f"what does {concept} mean",
                    target_text=definition,
                    pattern_type='chat_qa',
                    confidence=1.0,
                    source='curated_definitions'
                ),
            ])

        return patterns

    def generate_completion_patterns(self, seeds: List[Tuple[List[str], int]]) -> List[AugmentedPattern]:
        """Generate completion patterns using SparkSLM."""
        patterns = []

        for seed_tokens, length in seeds:
            sequence = self.ngram.predict_sequence(seed_tokens, length=length)
            if sequence and sequence != seed_tokens:  # Avoid repetition
                patterns.append(AugmentedPattern(
                    input_text=' '.join(seed_tokens),
                    target_text=' '.join(sequence),
                    pattern_type='completion',
                    confidence=0.8,
                    source='ngram_generation'
                ))

        return patterns

    def generate_hierarchical_patterns(self) -> List[AugmentedPattern]:
        """Use Woven Mind abstractions to create hierarchical training data."""
        patterns = []

        # Define concept hierarchies
        hierarchies = {
            "algorithm": ["pagerank", "tfidf", "bm25", "louvain"],
            "data structure": ["minicolumn", "layer", "edge", "connection"],
            "component": ["gotmanager", "tokenizer", "processor", "persistence"],
            "cognitive mode": ["fast mode", "slow mode", "hive", "cortex"],
        }

        for category, members in hierarchies.items():
            for member in members:
                # "X is a type of Y" patterns
                patterns.append(AugmentedPattern(
                    input_text=f"what type is {member}",
                    target_text=f"{member} is a type of {category}",
                    pattern_type='hierarchical',
                    confidence=1.0,
                    source='hierarchy_definition'
                ))

                # "Y includes X" patterns
                patterns.append(AugmentedPattern(
                    input_text=f"{category} examples",
                    target_text=f"{category} includes {', '.join(members)}",
                    pattern_type='hierarchical',
                    confidence=1.0,
                    source='hierarchy_definition'
                ))

        # Use Woven Mind to find additional relationships
        for concept in list(self.concept_definitions.keys())[:10]:
            result = self.mind.process([concept])
            if result.predictions:
                related = [p for p, score in sorted(
                    result.predictions.items(),
                    key=lambda x: -x[1]
                )[:3]]

                if related:
                    patterns.append(AugmentedPattern(
                        input_text=f"related to {concept}",
                        target_text=f"{concept} is related to {', '.join(related)}",
                        pattern_type='hierarchical',
                        confidence=1.0 - (result.surprise.magnitude if result.surprise else 0.5),
                        source='woven_mind_association'
                    ))

        return patterns

    def ingest_chat_history(self, transcript_path: Optional[Path] = None) -> List[AugmentedPattern]:
        """
        Ingest chat history to extract Q&A patterns.

        Chat history contains real conversations about the codebase,
        making it excellent training data for concept explanations.
        """
        patterns = []

        # Check common locations for chat history
        possible_paths = [
            PROJECT_ROOT / ".git-ml" / "tracked" / "chunked",  # ML collected chats
            PROJECT_ROOT / ".git-ml" / "chats",
            PROJECT_ROOT / ".claude" / "transcripts",
            Path.home() / ".claude" / "transcripts",
        ]

        if transcript_path:
            possible_paths.insert(0, transcript_path)

        chat_files = []
        for path in possible_paths:
            if path.exists():
                if path.is_file():
                    chat_files.append(path)
                else:
                    chat_files.extend(path.glob("*.json"))
                    chat_files.extend(path.glob("*.jsonl"))

        print(f"Found {len(chat_files)} chat history files")

        for chat_file in chat_files:
            try:
                new_patterns = self._parse_chat_file(chat_file)
                patterns.extend(new_patterns)
                if new_patterns:
                    print(f"  Extracted {len(new_patterns)} Q&A from {chat_file.name}")
            except Exception as e:
                print(f"  Warning: Could not parse {chat_file}: {e}")

        return patterns

    def _parse_chat_file(self, path: Path) -> List[AugmentedPattern]:
        """Parse a chat transcript file for Q&A patterns."""
        patterns = []

        try:
            if path.suffix == '.jsonl':
                with open(path) as f:
                    entries = [json.loads(line) for line in f if line.strip()]
            else:
                with open(path) as f:
                    data = json.load(f)
                    entries = data if isinstance(data, list) else [data]
        except Exception:
            return patterns

        for entry in entries:
            # Handle ML data format (record_type + data)
            if 'record_type' in entry and entry.get('record_type') == 'chat':
                data = entry.get('data', {})
                query = data.get('query', '')
                response = data.get('response', '')

                # Skip compressed queries (start with eJz which is base64 zlib)
                if query.startswith('eJz') or not query or not response:
                    continue

                # Skip very short or code-heavy responses
                if len(response) < 50 or response.count('```') > 4:
                    continue

                # Clean and truncate
                query = query[:500].strip()
                response = response[:2000].strip()

                patterns.append(AugmentedPattern(
                    input_text=query,
                    target_text=response,
                    pattern_type='chat_qa',
                    confidence=0.95,  # High confidence - real conversations
                    source=f'ml_chat:{path.name}'
                ))
                continue

            # Handle direct transcript format
            role = entry.get('role', entry.get('type', ''))
            content = entry.get('content', entry.get('text', ''))

            if isinstance(content, list):
                content = ' '.join(
                    c.get('text', '') for c in content
                    if isinstance(c, dict)
                )

            # Simple Q&A extraction from sequential messages
            if not hasattr(self, '_pending_question'):
                self._pending_question = None

            if role in ('user', 'human') and content and len(content) > 10:
                self._pending_question = content[:500]
            elif role in ('assistant', 'ai') and self._pending_question and content:
                if len(content) > 50 and not content.startswith('```'):
                    patterns.append(AugmentedPattern(
                        input_text=self._pending_question,
                        target_text=content[:1000],
                        pattern_type='chat_qa',
                        confidence=0.9,
                        source=str(path.name)
                    ))
                self._pending_question = None

        return patterns

    def filter_by_confidence(self, patterns: List[AugmentedPattern],
                            min_confidence: float = 0.5) -> List[AugmentedPattern]:
        """Use Woven Mind surprise to filter low-quality patterns."""
        filtered = []

        for pattern in patterns:
            # Already has confidence from generation
            if pattern.confidence >= min_confidence:
                # Additional Woven Mind check for generated patterns
                if pattern.source in ('ngram_generation', 'woven_mind_association'):
                    tokens = pattern.target_text.lower().split()[:3]
                    result = self.mind.process(tokens)
                    surprise = result.surprise.magnitude if result.surprise else 0.5
                    pattern.confidence *= (1.0 - surprise * 0.5)  # Reduce if surprising

                if pattern.confidence >= min_confidence:
                    filtered.append(pattern)
            else:
                filtered.append(pattern)  # Keep curated patterns regardless

        return filtered

    def run_full_pipeline(self, base_corpus: List[str]) -> List[AugmentedPattern]:
        """Run the complete augmentation pipeline."""
        print("=" * 70)
        print("DATA AUGMENTATION PIPELINE")
        print("=" * 70)

        # Step 1: Train base models
        print("\n1. Training base models...")
        self.train_base_models(base_corpus)
        print(f"   Trained on {len(base_corpus)} base texts")

        # Step 2: Load concept definitions
        print("\n2. Loading concept definitions...")
        self.load_concept_definitions()
        print(f"   Loaded {len(self.concept_definitions)} concepts")

        # Step 3: Generate definition patterns (fixes concept category)
        print("\n3. Generating definition patterns...")
        definition_patterns = self.generate_definition_patterns()
        print(f"   Generated {len(definition_patterns)} definition patterns")

        # Step 4: Generate completion patterns
        print("\n4. Generating completion patterns...")
        seeds = [
            (["pagerank", "is"], 8),
            (["import", "from"], 6),
            (["the", "tokenizer"], 8),
            (["gotmanager", "provides"], 8),
            (["to", "create", "a"], 6),
            (["run", "python"], 6),
        ]
        completion_patterns = self.generate_completion_patterns(seeds)
        print(f"   Generated {len(completion_patterns)} completion patterns")

        # Step 5: Generate hierarchical patterns
        print("\n5. Generating hierarchical patterns...")
        hierarchical_patterns = self.generate_hierarchical_patterns()
        print(f"   Generated {len(hierarchical_patterns)} hierarchical patterns")

        # Step 6: Ingest chat history
        print("\n6. Ingesting chat history...")
        chat_patterns = self.ingest_chat_history()
        print(f"   Extracted {len(chat_patterns)} chat Q&A patterns")

        # Combine all patterns
        all_patterns = (
            definition_patterns +
            completion_patterns +
            hierarchical_patterns +
            chat_patterns
        )

        # Step 7: Filter by confidence
        print("\n7. Filtering by confidence...")
        filtered_patterns = self.filter_by_confidence(all_patterns, min_confidence=0.4)
        print(f"   Kept {len(filtered_patterns)} / {len(all_patterns)} patterns")

        self.patterns = filtered_patterns
        return filtered_patterns

    def export_training_corpus(self, output_path: Path,
                                weights: Optional[Dict[str, int]] = None) -> int:
        """
        Export augmented patterns as training corpus with configurable oversampling.

        Args:
            output_path: Where to write the corpus
            weights: Optional pattern type weights. Defaults to tuned weights.

        The default weights are tuned to fix the 0% concept category issue:
        - Concept patterns (definition, hierarchical) get 20x weight
        - Chat Q&A patterns get 10x weight (real conversations)
        - Completion patterns get 3x weight (generated, less reliable)
        """
        # Default weights tuned for balanced category performance
        # These were derived from benchmark results showing:
        # - file_location: 87.5% (too dominant)
        # - concept: 0% (needs heavy oversampling)
        if weights is None:
            weights = {
                'definition': 20,      # Concept explanations - critical for 0% concept
                'hierarchical': 15,    # Type relationships - "X is a type of Y"
                'chat_qa': 10,         # Real Q&A - high quality
                'completion': 3,       # Generated sequences - less reliable
                'pln_inference': 12,   # Logical inferences - good for relationships
                'dialogue': 8,         # Agent dialogues - natural Q&A style
            }

        training_lines = []
        for pattern in self.patterns:
            training_lines.append(pattern.to_training_format())

        # Oversample based on pattern type and confidence
        oversampled = []
        type_counts = {}

        for pattern in self.patterns:
            # Base weight from pattern type
            base_weight = weights.get(pattern.pattern_type, 5)

            # Confidence multiplier (0.5-1.5x based on confidence)
            confidence_mult = 0.5 + pattern.confidence

            # Calculate final repeat count
            repeat = max(1, int(base_weight * confidence_mult))

            # Track for statistics
            type_counts[pattern.pattern_type] = type_counts.get(pattern.pattern_type, 0) + repeat

            for _ in range(repeat):
                oversampled.append(pattern.to_training_format())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for line in oversampled:
                f.write(line + '\n')

        # Print weight distribution
        total = sum(type_counts.values())
        print(f"\nOversampling distribution:")
        for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {ptype}: {count} ({pct:.1f}%)")

        return len(oversampled)

    def get_statistics(self) -> Dict:
        """Get statistics about augmented patterns."""
        by_type = {}
        by_source = {}

        for p in self.patterns:
            by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1
            by_source[p.source] = by_source.get(p.source, 0) + 1

        return {
            'total_patterns': len(self.patterns),
            'by_type': by_type,
            'by_source': by_source,
            'avg_confidence': sum(p.confidence for p in self.patterns) / len(self.patterns) if self.patterns else 0,
        }


def explore_woven_abstractions():
    """Explore Woven Mind's abstraction mechanism in detail."""
    print("\n" + "=" * 70)
    print("WOVEN MIND ABSTRACTION EXPLORATION")
    print("=" * 70)

    mind = WovenMind()

    # Train on concept-rich patterns
    training_patterns = [
        "PageRank is a graph algorithm for importance",
        "TF-IDF is a scoring algorithm for relevance",
        "BM25 is a ranking algorithm for search",
        "Louvain is a clustering algorithm for communities",
        "GoTManager manages tasks decisions and sprints",
        "Tokenizer processes text into tokens",
        "Processor orchestrates the analysis pipeline",
        "Persistence saves and loads state",
        "Hebbian learning strengthens co-activated connections",
        "Lateral connections link related terms",
        "Typed connections have semantic relationships",
    ]

    print("\n1. Training Woven Mind on concepts...")
    for pattern in training_patterns:
        mind.train(pattern)
    print(f"   Trained on {len(training_patterns)} patterns")

    # Observe repeated patterns to build abstractions
    print("\n2. Building abstractions through repetition...")
    abstraction_seeds = [
        ["algorithm", "graph"],
        ["algorithm", "scoring"],
        ["algorithm", "ranking"],
        ["manager", "tasks"],
        ["connections", "related"],
        ["connections", "semantic"],
    ]

    for _ in range(5):  # Repeat to strengthen abstractions
        for seed in abstraction_seeds:
            result = mind.process(seed)

    # Run consolidation
    print("\n3. Running consolidation...")
    consolidation = mind.consolidate()
    print(f"   Patterns transferred: {consolidation.patterns_transferred}")
    print(f"   Abstractions formed: {consolidation.abstractions_formed}")
    print(f"   Connections decayed: {consolidation.connections_decayed}")

    # Explore what abstractions formed
    print("\n4. Testing abstraction activation...")

    test_queries = [
        ["algorithm"],
        ["connections"],
        ["manager"],
        ["learning"],
    ]

    abstraction_data = []
    for query in test_queries:
        result = mind.process(query)
        print(f"\n   Query: {query}")
        print(f"   Mode: {result.mode.name}")
        print(f"   Activations: {len(result.activations)}")
        if result.predictions:
            preds = sorted(result.predictions.items(), key=lambda x: -x[1])[:5]
            print(f"   Predictions: {preds}")
            abstraction_data.append({
                'seed': query[0],
                'predictions': [p[0] for p, _ in preds],
                'mode': result.mode.name
            })

    print("\n" + "=" * 70)
    print("ABSTRACTION INSIGHTS")
    print("=" * 70)
    print("""
Woven Mind builds abstractions by:
1. Observing repeated pattern co-occurrences
2. Strengthening frequently activated connections
3. Consolidating stable patterns from Hive to Cortex

For training data generation, this means:
- Concepts that frequently appear together form abstractions
- These abstractions can generate "X is related to Y" patterns
- The FAST/SLOW mode indicates familiarity (confidence)

Hierarchical training data from abstractions:
- "algorithm" → [graph, scoring, ranking, clustering]
- "connections" → [lateral, typed, semantic, weighted]
- These become training patterns like:
  "Q: What types of algorithms exist? A: graph, scoring, ranking, clustering"
""")

    return abstraction_data


def main():
    """Run data augmentation pipeline."""

    # First, explore abstractions
    abstraction_data = explore_woven_abstractions()

    # Base corpus from existing training
    base_corpus = [
        "PageRank is implemented in cortical analysis pagerank.py",
        "TF-IDF is implemented in cortical analysis tfidf.py",
        "GoTManager is the main API for Graph of Thought",
        "Import GoTManager from cortical.got",
        "Hebbian learning means neurons that fire together wire together",
        "The tokenizer splits text into tokens and bigrams",
        "Louvain clustering detects community structure",
        "Persistence saves and loads processor state",
        "Query expansion adds related terms to searches",
        "BM25 is the default scoring algorithm",
    ]

    # Run pipeline
    pipeline = DataAugmentationPipeline()
    patterns = pipeline.run_full_pipeline(base_corpus)

    # Show statistics
    print("\n" + "=" * 70)
    print("AUGMENTATION RESULTS")
    print("=" * 70)

    stats = pipeline.get_statistics()
    print(f"\nTotal patterns: {stats['total_patterns']}")
    print("\nBy type:")
    for ptype, count in stats['by_type'].items():
        print(f"  {ptype}: {count}")
    print("\nBy source:")
    for source, count in stats['by_source'].items():
        print(f"  {source}: {count}")
    print(f"\nAverage confidence: {stats['avg_confidence']:.2f}")

    # Export
    output_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "data" / "augmented_corpus.txt"
    count = pipeline.export_training_corpus(output_path)
    print(f"\nExported {count} training lines to {output_path}")

    # Show sample patterns
    print("\n" + "=" * 70)
    print("SAMPLE AUGMENTED PATTERNS")
    print("=" * 70)

    for ptype in ['definition', 'hierarchical', 'completion', 'chat_qa']:
        print(f"\n{ptype.upper()}:")
        samples = [p for p in patterns if p.pattern_type == ptype][:3]
        for p in samples:
            print(f"  Input: {p.input_text[:60]}...")
            print(f"  Target: {p.target_text[:60]}...")
            print(f"  Confidence: {p.confidence:.2f}")
            print()


if __name__ == "__main__":
    main()
