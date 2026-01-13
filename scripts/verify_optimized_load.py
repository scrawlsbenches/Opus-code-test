#!/usr/bin/env python3
"""
Verify that the optimized graph loading works correctly and is fast.
"""

import time
from pathlib import Path

import sys
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.cognitive.graph import CognitiveAgent
from cortical.cognitive.training import IncrementalTrainer
from cortical.common.filesystem import RealFileSystem


def main():
    print("=" * 70)
    print("OPTIMIZED LOAD VERIFICATION")
    print("=" * 70)

    model_dir = Path("models/cognitive_agent")

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1

    # Verify files exist
    manifest_path = model_dir / "training_manifest.json"
    graph_path = model_dir / "bridge" / "graph.json"
    tokenizer_dir = model_dir / "tokenizer"

    for path in [manifest_path, graph_path, tokenizer_dir]:
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            return 1

    print(f"\nModel directory: {model_dir}")
    print(f"Graph file size: {graph_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Time the load
    print("\n--- Loading with optimized _load_graph_state() ---")
    start = time.perf_counter()

    filesystem = RealFileSystem(base_dir=Path.cwd())
    agent = CognitiveAgent(filesystem=filesystem)

    trainer = IncrementalTrainer(
        agent=agent,
        model_dir=model_dir,
        filesystem=filesystem,
    )

    bridge = trainer.bridge

    end = time.perf_counter()
    load_time = end - start

    print(f"\nLoad time: {load_time:.3f}s")

    # Verify graph is loaded correctly
    all_atoms = agent.graph._storage.all_atoms()
    nodes = [a for a in all_atoms if a.is_node()]
    links = [a for a in all_atoms if a.is_link()]

    print(f"\n--- Verification ---")
    print(f"Total atoms: {len(all_atoms):,}")
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Links: {len(links):,}")

    # Check some sample words
    sample_words = ["neural", "training", "data", "learning", "model"]
    found_words = []
    for word in sample_words:
        atom = agent.graph.get_node(word)
        if atom:
            found_words.append(word)

    print(f"\nSample words found: {len(found_words)}/{len(sample_words)}")
    print(f"  {', '.join(found_words)}")

    # Check manifest
    print(f"\nManifest: {len(trainer.manifest.documents)} documents tracked")

    # Check bridge stats
    print(f"\nBridge stats:")
    print(f"  Documents fed: {bridge._documents_fed}")
    print(f"  Atoms created: {bridge._atoms_created}")
    print(f"  Links created: {bridge._links_created}")

    # Performance check
    print("\n" + "=" * 70)
    if load_time < 1.0:
        print(f"✓ EXCELLENT: Load time {load_time:.3f}s (target < 1s)")
    elif load_time < 5.0:
        print(f"✓ GOOD: Load time {load_time:.3f}s (target < 5s)")
    else:
        print(f"✗ SLOW: Load time {load_time:.3f}s (need optimization)")

    # Scaling projection
    docs_trained = len(trainer.manifest.documents)
    docs_remaining = 525 - docs_trained
    if docs_trained > 0:
        time_per_doc = load_time / docs_trained
        projected_full = time_per_doc * 525
        print(f"\nScaling projection for 525 documents:")
        print(f"  Current: {docs_trained} docs @ {time_per_doc*1000:.1f}ms each")
        print(f"  Projected full load: {projected_full:.3f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
