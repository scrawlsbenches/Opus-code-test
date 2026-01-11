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
    # Quick demo (fast defaults)
    python examples/load_samples_demo.py

    # Load existing model and add more documents incrementally
    python examples/load_samples_demo.py --incremental --max-files 10

    # Train fresh and save
    python examples/load_samples_demo.py --max-files 20 --save

    # Custom performance tuning
    python examples/load_samples_demo.py --max-links 50 --window-size 2

    # Full training (slower)
    python examples/load_samples_demo.py --max-files 100 --max-links 500 --save

Defaults (optimized for speed):
    --max-files 5      Only process 5 files
    --max-links 100    Limit links per document (vs 500 default)
    --window-size 3    Smaller co-occurrence window (vs 5 default)

Design Philosophy:
    Start small, expand as needed. Fast iteration by default.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.cognitive.graph import (
    Atom,
    AtomType,
    CognitiveAgent,
    CognitiveGraph,
    EventBus,
    EventType,
    TruthValue,
)
from cortical.cognitive.text_bridge import (
    BPETokenizer,
    TextToAtomsBridge,
    load_directory_to_bridge,
)

# Default model path
DEFAULT_MODEL_DIR = PROJECT_ROOT / "trained_model"


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
# Model Loading/Saving
# =============================================================================


def load_model(model_dir: Path) -> tuple:
    """
    Load agent and tokenizer from model directory.

    Args:
        model_dir: Directory containing graph.json and tokenizer.json

    Returns:
        Tuple of (CognitiveAgent, BPETokenizer)
    """
    graph_path = model_dir / "graph.json"
    tokenizer_path = model_dir / "tokenizer.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"Model not found: {graph_path}")

    print(f"{Colors.YELLOW}Loading model from {model_dir}...{Colors.RESET}")

    # Load graph
    with open(graph_path) as f:
        data = json.load(f)

    agent = CognitiveAgent()
    atom_count = 0

    for atom_data in data.get("atoms", []):
        atom = Atom(
            id=atom_data["id"],
            atom_type=AtomType[atom_data["atom_type"]],
            name=atom_data.get("name", ""),
            outgoing=atom_data.get("outgoing", []),
            tv=TruthValue(
                atom_data.get("tv_strength", 1.0),
                atom_data.get("tv_confidence", 0.0),
            ),
            sti=atom_data.get("sti", 0.0),
            lti=atom_data.get("lti", 0.0),
        )
        agent.graph._storage.save(atom)
        atom_count += 1

    # Load tokenizer if exists
    tokenizer = BPETokenizer()
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"  {Colors.GREEN}Loaded tokenizer with {len(tokenizer.vocab)} words{Colors.RESET}")

    print(f"  {Colors.GREEN}Loaded {atom_count} atoms{Colors.RESET}")

    stats = data.get("stats", {})
    if stats:
        print(f"  {Colors.DIM}Previous training: {stats.get('documents_fed', 0)} documents{Colors.RESET}")

    return agent, tokenizer


def save_model(agent: CognitiveAgent, tokenizer: BPETokenizer, bridge: TextToAtomsBridge, model_dir: Path) -> None:
    """
    Save agent and tokenizer to model directory.

    Args:
        agent: The CognitiveAgent to save
        tokenizer: The BPETokenizer to save
        bridge: The TextToAtomsBridge (for statistics)
        model_dir: Directory to save to
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    graph_path = model_dir / "graph.json"
    tokenizer_path = model_dir / "tokenizer.json"

    print(f"{Colors.YELLOW}Saving model to {model_dir}...{Colors.RESET}")

    # Save graph with flat format (compatible with load_model)
    atoms_data = []
    for atom in agent.graph._storage.all_atoms():
        atoms_data.append({
            "id": atom.id,
            "name": atom.name,
            "atom_type": atom.atom_type.name,
            "tv_strength": atom.tv.strength,
            "tv_confidence": atom.tv.confidence,
            "sti": atom.sti,
            "lti": atom.lti,
            "outgoing": atom.outgoing,
        })

    graph_data = {
        "atoms": atoms_data,
        "stats": bridge.get_statistics(),
    }

    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    # Save tokenizer
    tokenizer.save(tokenizer_path)

    print(f"  {Colors.GREEN}Saved {len(atoms_data)} atoms to graph.json{Colors.RESET}")
    print(f"  {Colors.GREEN}Saved tokenizer with {len(tokenizer.vocab)} words{Colors.RESET}")


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


def demo_text_to_atoms(
    agent: CognitiveAgent,
    samples_dir: Path,
    max_files: int = 3,
    max_links: int = 100,
    window_size: int = 3,
    verbose: bool = False,
    existing_tokenizer: BPETokenizer = None,
    incremental: bool = False,
):
    """
    Demonstrate converting text to cognitive graph atoms.

    Shows:
        - Text being fed into the agent
        - WORD atoms created
        - SIMILARITY links between co-occurring words

    Args:
        agent: The CognitiveAgent to populate
        samples_dir: Directory with .txt files
        max_files: Maximum files to process
        max_links: Max links per document (lower = faster)
        window_size: Co-occurrence window (lower = faster)
        verbose: Show detailed output
        existing_tokenizer: Use existing tokenizer for incremental learning
        incremental: If True, add to existing vocabulary
    """
    print_header("Phase 2: Text-to-Atoms Conversion")

    # Use existing tokenizer or create new one
    tokenizer = existing_tokenizer if existing_tokenizer else BPETokenizer()

    # Create bridge with performance settings
    bridge = TextToAtomsBridge(
        graph=agent.graph,
        tokenizer=tokenizer,
        window_size=window_size,
        max_links_per_doc=max_links,
    )

    print(f"  {Colors.DIM}Window size: {window_size}, Max links/doc: {max_links}{Colors.RESET}")
    if incremental:
        print(f"  {Colors.CYAN}Incremental mode: adding to existing vocabulary{Colors.RESET}")

    # Load sample files
    txt_files = sorted(samples_dir.glob("*.txt"))[:max_files]

    # First, learn vocabulary from all files
    texts = []
    for txt_file in txt_files:
        try:
            texts.append(txt_file.read_text(encoding='utf-8'))
        except Exception:
            pass

    mode_str = "incrementally" if incremental else "from scratch"
    print(f"{Colors.YELLOW}Learning vocabulary {mode_str} from {len(texts)} files...{Colors.RESET}")
    bridge.learn_vocabulary(texts, incremental=incremental)

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
    agent.graph.stimulate(seed.id, amount=0.5)

    # Run a few steps
    print_subheader("Agent Steps")
    for step in range(5):
        agent.step()

        # Show working memory (uses contents() method)
        wm_atoms = agent.working_memory.contents()
        wm_names = [a.name for a in wm_atoms if a.name][:5]

        print(f"\n  {Colors.BOLD}Step {step + 1}{Colors.RESET}")
        print(f"    Working memory: {Colors.YELLOW}{', '.join(wm_names) or '(empty)'}{Colors.RESET}")
        print(f"    Exploration epsilon: {Colors.DIM}{agent.exploration.epsilon:.3f}{Colors.RESET}")

        if verbose:
            # Show top attention atoms
            top_atoms = agent.graph.get_attention_focus(top_k=3)
            print(f"    Top attention:")
            for atom in top_atoms:
                name = atom.name or f"[link:{atom.id[:4]}]"
                print(f"      - {name}: {Colors.GREEN}{atom.sti:.3f}{Colors.RESET}")


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

    # Show connected words sorted by link strength (deduplicated, keep max strength)
    connections_dict = {}
    for link in similarity_links:
        # Find the other atom in the link
        for target_id in link.outgoing:
            if target_id != atom.id:
                other = agent.graph.get_atom(target_id)
                if other and other.name:
                    # Keep the strongest connection for each word
                    if other.name not in connections_dict or link.tv.strength > connections_dict[other.name]:
                        connections_dict[other.name] = link.tv.strength

    # Sort by strength
    connections = sorted(connections_dict.items(), key=lambda x: x[1], reverse=True)

    for name, strength in connections[:10]:
        bar_len = int(strength * 20)
        bar = "█" * bar_len
        print(f"  {name:20} {strength:.2f} {Colors.GREEN}{bar}{Colors.RESET}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Load samples into CognitiveAgent with incremental training support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick demo:           python examples/load_samples_demo.py
  Incremental:          python examples/load_samples_demo.py -i --max-files 10
  Train and save:       python examples/load_samples_demo.py --max-files 20 -s
  Fast iteration:       python examples/load_samples_demo.py --max-links 50 --window-size 2
        """,
    )

    # Basic options
    parser.add_argument("--max-files", type=int, default=5,
                        help="Maximum number of files to process (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show more details")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Path to samples directory")
    parser.add_argument("--query", type=str, default=None,
                        help="Word to query connections for")

    # Incremental training options
    parser.add_argument("--incremental", "-i", action="store_true",
                        help="Load existing model and add new documents incrementally")
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save model after training")

    # Performance tuning (defaults optimized for speed)
    parser.add_argument("--max-links", type=int, default=100,
                        help="Max links per document (default: 100, lower=faster)")
    parser.add_argument("--window-size", type=int, default=3,
                        help="Co-occurrence window size (default: 3, lower=faster)")

    # Skip phases
    parser.add_argument("--skip-explore", action="store_true",
                        help="Skip agent exploration phase")
    parser.add_argument("--skip-query", action="store_true",
                        help="Skip query phase")

    args = parser.parse_args()

    # Resolve paths
    model_dir = Path(args.model_path) if args.model_path else DEFAULT_MODEL_DIR
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
    print(f"{Colors.DIM}Max links/doc: {args.max_links}, Window: {args.window_size}{Colors.RESET}")

    # Load or create agent
    existing_tokenizer = None
    if args.incremental and model_dir.exists() and (model_dir / "graph.json").exists():
        print_subheader("Loading Existing Model (Incremental Mode)")
        try:
            agent, existing_tokenizer = load_model(model_dir)
            print(f"  {Colors.GREEN}Existing model loaded, will add new documents{Colors.RESET}")
        except Exception as e:
            print(f"  {Colors.RED}Failed to load model: {e}{Colors.RESET}")
            print(f"  {Colors.YELLOW}Starting fresh instead{Colors.RESET}")
            agent = CognitiveAgent()
    else:
        if args.incremental:
            print(f"{Colors.YELLOW}No existing model found, starting fresh{Colors.RESET}")
        print_subheader("Creating CognitiveAgent")
        agent = CognitiveAgent()
        print(f"  {Colors.GREEN}Agent created with empty graph{Colors.RESET}")

    # Phase 1: Tokenizer learning (skip if incremental with existing tokenizer)
    if existing_tokenizer and args.incremental:
        print_subheader("Using Existing Tokenizer")
        print(f"  {Colors.DIM}Vocabulary: {len(existing_tokenizer.vocab)} words{Colors.RESET}")
        tokenizer = existing_tokenizer
    else:
        tokenizer = demo_tokenizer_learning(samples_dir, args.max_files, args.verbose)

    # Phase 2: Text to atoms
    bridge = demo_text_to_atoms(
        agent,
        samples_dir,
        max_files=args.max_files,
        max_links=args.max_links,
        window_size=args.window_size,
        verbose=args.verbose,
        existing_tokenizer=tokenizer,
        incremental=args.incremental,
    )

    # Phase 3: Agent exploration (optional)
    if not args.skip_explore:
        demo_agent_exploration(agent, args.verbose)

    # Phase 4: Query connections (optional)
    if not args.skip_query:
        demo_query_connections(agent, args.query)

    # Summary
    print_header("Summary")
    print(f"  {Colors.BOLD}The CognitiveAgent now has:{Colors.RESET}")
    print(f"    - {len(agent.graph._storage.all_atoms())} atoms in knowledge graph")
    print(f"    - {len(agent.graph.find_by_type(AtomType.WORD))} WORD atoms")
    print(f"    - {len(agent.graph.find_by_type(AtomType.SIMILARITY))} SIMILARITY links")

    # Save if requested
    if args.save:
        print_subheader("Saving Model")
        save_model(agent, tokenizer, bridge, model_dir)

    print(f"\n  {Colors.DIM}The agent can now:{Colors.RESET}")
    print(f"    - Navigate between related concepts")
    print(f"    - Learn from surprise (prediction errors)")
    print(f"    - Build working memory of active concepts")
    print(f"    - Explore when stuck on a goal")

    if not args.save:
        print(f"\n  {Colors.DIM}Tip: Use --save to persist the trained model{Colors.RESET}")

    print(f"\n{Colors.GREEN}Demo complete!{Colors.RESET}")


if __name__ == "__main__":
    main()
