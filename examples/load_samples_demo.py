#!/usr/bin/env python3
"""
Load Samples Demo: Text-to-Atoms Bridge in Action.

This script demonstrates loading text files into a CognitiveAgent,
showing the complete pipeline from raw text to cognitive atoms.

Run this script to see:
    1. BPE tokenizer learning vocabulary from samples
    2. Text converted to WORD atoms
    3. Co-occurrence creating SIMILARITY links
    4. Agent processing the learned knowledge

Usage:
    python examples/load_samples_demo.py

    # Load specific number of files
    python examples/load_samples_demo.py --max-files 5

    # Show more details
    python examples/load_samples_demo.py --verbose

Design Philosophy:
    Start small, expand as needed. This demo uses a minimal subset
    of samples to show the concept clearly.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.cognitive.graph import (
    AtomType,
    CognitiveAgent,
    CognitiveGraph,
    EventBus,
    EventType,
)
from cortical.cognitive.text_bridge import (
    BPETokenizer,
    TextToAtomsBridge,
    load_directory_to_bridge,
)


# =============================================================================
# ANSI Colors for Pretty Output
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.RESET}\n")


def print_subheader(text: str) -> None:
    """Print a subsection header."""
    print(f"\n{Colors.CYAN}{'-' * 50}{Colors.RESET}")
    print(f"{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * 50}{Colors.RESET}")


def print_stat(label: str, value, color: str = Colors.GREEN) -> None:
    """Print a statistic."""
    print(f"  {Colors.DIM}{label}:{Colors.RESET} {color}{value}{Colors.RESET}")


# =============================================================================
# Demo Functions
# =============================================================================


def demo_tokenizer_learning(samples_dir: Path, max_files: int = 5, verbose: bool = False) -> BPETokenizer:
    """
    Demonstrate BPE tokenizer learning vocabulary.

    Shows:
        - How the tokenizer processes text
        - Vocabulary statistics
        - Most frequent word pairs (potential compounds)
    """
    print_header("Phase 1: BPE Tokenizer Learning")

    tokenizer = BPETokenizer()

    # Collect sample texts
    texts = []
    files_loaded = 0
    for txt_file in sorted(samples_dir.glob("*.txt"))[:max_files]:
        try:
            content = txt_file.read_text(encoding='utf-8')
            texts.append(content)
            files_loaded += 1
            if verbose:
                print(f"  {Colors.DIM}Loaded:{Colors.RESET} {txt_file.name}")
        except Exception as e:
            print(f"  {Colors.RED}Error loading {txt_file.name}: {e}{Colors.RESET}")

    print(f"\n{Colors.BOLD}Loaded {files_loaded} sample files{Colors.RESET}")

    # Learn vocabulary
    print(f"\n{Colors.YELLOW}Learning vocabulary...{Colors.RESET}")
    tokenizer.learn_from_texts(texts, n_merges=50)

    # Show statistics
    print_subheader("Vocabulary Statistics")
    print_stat("Unique words", len(tokenizer.vocab))
    print_stat("Total word occurrences", sum(tokenizer._word_counts.values()))
    print_stat("Learned merges (frequent pairs)", len(tokenizer.merges))

    # Show top words
    print_subheader("Top 10 Most Frequent Words")
    for word, count in tokenizer._word_counts.most_common(10):
        bar = "█" * min(30, count // 5)
        print(f"  {word:20} {count:5} {Colors.GREEN}{bar}{Colors.RESET}")

    # Show top pairs (potential compound concepts)
    print_subheader("Top 10 Word Pairs (Compound Candidates)")
    for (w1, w2), count in tokenizer.get_top_pairs(10):
        compound = f"{w1}_{w2}"
        print(f"  {w1} + {w2:15} = {Colors.CYAN}{compound}{Colors.RESET} ({count}x)")

    return tokenizer


def demo_text_to_atoms(agent: CognitiveAgent, samples_dir: Path, max_files: int = 3, verbose: bool = False):
    """
    Demonstrate converting text to cognitive graph atoms.

    Shows:
        - Text being fed into the agent
        - WORD atoms created
        - SIMILARITY links between co-occurring words
    """
    print_header("Phase 2: Text-to-Atoms Conversion")

    bridge = TextToAtomsBridge(agent.graph)

    # Load sample files
    txt_files = sorted(samples_dir.glob("*.txt"))[:max_files]

    # First, learn vocabulary from all files
    texts = []
    for txt_file in txt_files:
        try:
            texts.append(txt_file.read_text(encoding='utf-8'))
        except Exception:
            pass

    print(f"{Colors.YELLOW}Learning vocabulary from {len(texts)} files...{Colors.RESET}")
    bridge.learn_vocabulary(texts)

    # Now feed each file
    print_subheader("Feeding Documents")

    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding='utf-8')
            doc_id = txt_file.stem

            # Feed text
            atoms = bridge.feed_text(content, doc_id=doc_id)

            print(f"\n{Colors.BOLD}Document: {doc_id}{Colors.RESET}")
            print(f"  {Colors.DIM}Characters:{Colors.RESET} {len(content)}")
            print(f"  {Colors.DIM}Atoms created/updated:{Colors.RESET} {len(atoms)}")

            if verbose and atoms:
                # Show first few atoms
                word_atoms = [a for a in atoms if a.name][:5]
                print(f"  {Colors.DIM}Sample atoms:{Colors.RESET}")
                for atom in word_atoms:
                    print(f"    - {Colors.GREEN}{atom.name}{Colors.RESET} (LTI: {atom.lti:.2f})")

        except Exception as e:
            print(f"  {Colors.RED}Error: {e}{Colors.RESET}")

    # Show bridge statistics
    print_subheader("Bridge Statistics")
    stats = bridge.get_statistics()
    for key, value in stats.items():
        print_stat(key.replace('_', ' ').title(), value)

    return bridge


def demo_agent_exploration(agent: CognitiveAgent, verbose: bool = False):
    """
    Demonstrate the agent exploring its learned knowledge.

    Shows:
        - Agent stepping through attention cycles
        - Working memory contents
        - Connections being discovered
    """
    print_header("Phase 3: Agent Exploring Knowledge")

    # Get word atoms
    word_atoms = agent.graph.find_by_type(AtomType.WORD)
    if not word_atoms:
        print(f"{Colors.RED}No WORD atoms found. Run Phase 2 first.{Colors.RESET}")
        return

    print(f"{Colors.BOLD}Total WORD atoms in graph: {len(word_atoms)}{Colors.RESET}")

    # Find a seed word with connections
    print_subheader("Finding a Seed Word to Explore")

    # Pick a word with high LTI (common word)
    seed = max(word_atoms, key=lambda a: a.lti)
    print(f"  Seed word: {Colors.CYAN}{seed.name}{Colors.RESET} (LTI: {seed.lti:.2f})")

    # Boost attention on seed
    agent.boost_attention(seed.id, amount=0.5)

    # Run a few steps
    print_subheader("Agent Steps")
    for step in range(5):
        agent.step()

        # Show working memory
        wm_ids = list(agent.working_memory._items.keys())
        wm_names = []
        for atom_id in wm_ids[:5]:
            atom = agent.graph.get_atom(atom_id)
            if atom and atom.name:
                wm_names.append(atom.name)

        print(f"\n  {Colors.BOLD}Step {step + 1}{Colors.RESET}")
        print(f"    Working memory: {Colors.YELLOW}{', '.join(wm_names) or '(empty)'}{Colors.RESET}")
        print(f"    Exploration epsilon: {Colors.DIM}{agent.epsilon:.3f}{Colors.RESET}")

        if verbose:
            # Show top attention atoms
            top_atoms = agent.get_top_attention(n=3)
            print(f"    Top attention:")
            for atom, sti in top_atoms:
                name = atom.name or f"[link:{atom.id[:4]}]"
                print(f"      - {name}: {Colors.GREEN}{sti:.3f}{Colors.RESET}")


def demo_query_connections(agent: CognitiveAgent, query_word: str = None):
    """
    Demonstrate querying connections in the learned graph.

    Shows how words are connected through SIMILARITY links.
    """
    print_header("Phase 4: Querying Connections")

    # If no query word, pick a common one
    if not query_word:
        word_atoms = agent.graph.find_by_type(AtomType.WORD)
        if word_atoms:
            query_word = max(word_atoms, key=lambda a: a.lti).name
        else:
            print(f"{Colors.RED}No words in graph to query.{Colors.RESET}")
            return

    print(f"{Colors.BOLD}Query: What is connected to '{query_word}'?{Colors.RESET}")

    # Find the atom
    atom = agent.graph.get_node(query_word)
    if not atom:
        print(f"{Colors.RED}Word '{query_word}' not found in graph.{Colors.RESET}")
        return

    print(f"  {Colors.DIM}Atom ID:{Colors.RESET} {atom.id}")
    print(f"  {Colors.DIM}LTI:{Colors.RESET} {atom.lti:.2f}")

    # Find connections (links where this atom participates)
    incoming = agent.graph.get_incoming(atom.id)
    similarity_links = [l for l in incoming if l.atom_type == AtomType.SIMILARITY]

    print_subheader(f"Connected Words ({len(similarity_links)} SIMILARITY links)")

    if not similarity_links:
        print(f"  {Colors.DIM}No similarity links found.{Colors.RESET}")
        return

    # Show connected words sorted by link strength
    connections = []
    for link in similarity_links:
        # Find the other atom in the link
        for target_id in link.outgoing:
            if target_id != atom.id:
                other = agent.graph.get_atom(target_id)
                if other and other.name:
                    connections.append((other.name, link.tv.strength))

    # Sort by strength
    connections.sort(key=lambda x: x[1], reverse=True)

    for name, strength in connections[:10]:
        bar_len = int(strength * 20)
        bar = "█" * bar_len
        print(f"  {name:20} {strength:.2f} {Colors.GREEN}{bar}{Colors.RESET}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Load samples into CognitiveAgent")
    parser.add_argument("--max-files", type=int, default=5,
                        help="Maximum number of files to process")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show more details")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Path to samples directory")
    parser.add_argument("--query", type=str, default=None,
                        help="Word to query connections for")
    args = parser.parse_args()

    # Find samples directory
    if args.samples_dir:
        samples_dir = Path(args.samples_dir)
    else:
        samples_dir = PROJECT_ROOT / "samples"

    if not samples_dir.exists():
        print(f"{Colors.RED}Error: Samples directory not found: {samples_dir}{Colors.RESET}")
        sys.exit(1)

    print_header("CognitiveAgent Text Loading Demo")
    print(f"{Colors.DIM}Samples directory: {samples_dir}{Colors.RESET}")
    print(f"{Colors.DIM}Max files: {args.max_files}{Colors.RESET}")

    # Phase 1: Demonstrate tokenizer learning
    tokenizer = demo_tokenizer_learning(samples_dir, args.max_files, args.verbose)

    # Create agent
    print_subheader("Creating CognitiveAgent")
    agent = CognitiveAgent()
    print(f"  {Colors.GREEN}Agent created with empty graph{Colors.RESET}")

    # Phase 2: Text to atoms
    bridge = demo_text_to_atoms(agent, samples_dir, args.max_files, args.verbose)

    # Phase 3: Agent exploration
    demo_agent_exploration(agent, args.verbose)

    # Phase 4: Query connections
    demo_query_connections(agent, args.query)

    # Summary
    print_header("Summary")
    print(f"  {Colors.BOLD}The CognitiveAgent now has:{Colors.RESET}")
    print(f"    - {len(agent.graph._storage.all_atoms())} atoms in knowledge graph")
    print(f"    - {len(agent.graph.find_by_type(AtomType.WORD))} WORD atoms")
    print(f"    - {len(agent.graph.find_by_type(AtomType.SIMILARITY))} SIMILARITY links")
    print(f"\n  {Colors.DIM}The agent can now:{Colors.RESET}")
    print(f"    - Navigate between related concepts")
    print(f"    - Learn from surprise (prediction errors)")
    print(f"    - Build working memory of active concepts")
    print(f"    - Explore when stuck on a goal")

    print(f"\n{Colors.GREEN}Demo complete!{Colors.RESET}")


if __name__ == "__main__":
    main()
