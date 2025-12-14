#!/usr/bin/env python3
"""
Cortical REPL Demo
==================

This script demonstrates the interactive REPL for the Cortical Text Processor.

The REPL provides a command-line interface for all processor operations with:
- Tab completion
- Command history (via readline)
- Built-in help system
- All processor operations accessible

Run this demo:
    python examples/repl_demo.py

Or start the REPL manually:
    python scripts/repl.py corpus_dev.pkl
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor


def create_sample_corpus():
    """Create a sample corpus for the demo."""
    print("Creating sample corpus...")

    processor = CorticalTextProcessor(enable_metrics=True)

    # Add some sample documents
    processor.process_document(
        "neural_networks.py",
        """
        class NeuralNetwork:
            def __init__(self, layers):
                self.layers = layers
                self.weights = []

            def train(self, data):
                # Train the network using backpropagation
                for epoch in range(100):
                    self.forward_pass(data)
                    self.backward_pass()

            def predict(self, input):
                return self.forward_pass(input)
        """
    )

    processor.process_document(
        "pagerank.py",
        """
        def compute_pagerank(graph, damping=0.85, iterations=100):
            # PageRank algorithm implementation
            n = len(graph)
            ranks = [1.0 / n] * n

            for _ in range(iterations):
                new_ranks = []
                for node in range(n):
                    rank = (1 - damping) / n
                    for neighbor in graph[node]:
                        rank += damping * ranks[neighbor]
                    new_ranks.append(rank)
                ranks = new_ranks

            return ranks
        """
    )

    processor.process_document(
        "README.md",
        """
        # Machine Learning Library

        This library provides implementations of common ML algorithms:

        ## Neural Networks
        - Feed-forward networks
        - Backpropagation training
        - Multiple activation functions

        ## Graph Algorithms
        - PageRank for importance ranking
        - Community detection
        - Graph embedding
        """
    )

    # Compute all analyses
    processor.compute_all()

    # Save corpus
    corpus_file = "demo_corpus.pkl"
    processor.save(corpus_file)
    print(f"✓ Created {corpus_file}")

    return corpus_file


def print_demo_commands():
    """Print example REPL commands."""
    print("\n" + "="*70)
    print("REPL DEMO - Example Commands")
    print("="*70)
    print("""
The REPL supports the following commands:

1. BASIC COMMANDS
   >>> stats                        # Show corpus statistics
   >>> search "neural network"      # Search documents
   >>> expand "neural"              # Show query expansion
   >>> concepts 10                  # List top 10 concept clusters

2. ADVANCED SEARCH
   >>> docs "what is pagerank"      # Search with documentation boost
   >>> code "train model"           # Code-aware search (synonyms)
   >>> passages "how does it work"  # Find relevant passages (RAG)
   >>> intent "where do we train"   # Intent-based search

3. CODE ANALYSIS
   >>> patterns neural_networks.py  # Detect code patterns
   >>> fingerprint "neural net"     # Get semantic fingerprint
   >>> similar pagerank.py:10       # Find similar code

4. INTROSPECTION
   >>> metrics                      # Show performance metrics
   >>> relations 10                 # Show semantic relations
   >>> stale                        # Show stale computations

5. COMPUTATION
   >>> compute                      # Compute all analyses
   >>> compute pagerank             # Compute specific analysis
   >>> compute concepts             # Build concept clusters

6. PERSISTENCE
   >>> save my_corpus.pkl           # Save corpus
   >>> export corpus_state json     # Export to JSON
   >>> load another.pkl             # Load different corpus

7. HELP SYSTEM
   >>> help                         # List all commands
   >>> help search                  # Help for specific command
   >>> quit                         # Exit REPL

Tab completion works for commands and file paths!
Command history is saved (use up/down arrows).
""")
    print("="*70)


def main():
    """Run the demo."""
    print(__doc__)

    # Check if demo corpus exists
    if not os.path.exists("demo_corpus.pkl"):
        print("\nDemo corpus not found. Creating one...")
        corpus_file = create_sample_corpus()
    else:
        corpus_file = "demo_corpus.pkl"
        print(f"\nUsing existing {corpus_file}")

    # Print example commands
    print_demo_commands()

    # Instructions to start REPL
    print("\nTo start the REPL with the demo corpus, run:")
    print(f"    python scripts/repl.py {corpus_file}")
    print("\nOr without a corpus:")
    print("    python scripts/repl.py")
    print("\nThen use 'load <file>' to load a corpus.")


if __name__ == '__main__':
    main()
