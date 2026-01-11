#!/usr/bin/env python3
"""
Profile graph loading to identify performance bottlenecks.

This script provides detailed timing breakdown for each phase of graph loading
to understand where the 32+ second load time is being spent.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
import sys
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def profile_phase(name: str):
    """Context manager for timing phases."""
    class Timer:
        def __init__(self, phase_name: str):
            self.name = phase_name
            self.start = 0.0
            self.end = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            duration = self.end - self.start
            print(f"  {self.name}: {duration:.3f}s")
            return False

    return Timer(name)


def profile_graph_load():
    """Profile the graph loading process in detail."""
    model_dir = Path("models/cognitive_agent")
    graph_path = model_dir / "bridge" / "graph.json"

    if not graph_path.exists():
        print(f"ERROR: Graph file not found at {graph_path}")
        return

    print("=" * 70)
    print("GRAPH LOAD PROFILING")
    print("=" * 70)

    # Get file size
    file_size = graph_path.stat().st_size
    print(f"\nFile: {graph_path}")
    print(f"Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    total_start = time.perf_counter()

    # Phase 1: Read file from disk
    print("\n--- Phase 1: File I/O ---")
    with profile_phase("Read file from disk"):
        raw_content = graph_path.read_text()

    print(f"  Content length: {len(raw_content):,} characters")

    # Phase 2: JSON parsing
    print("\n--- Phase 2: JSON Parsing ---")
    with profile_phase("Parse JSON"):
        graph_data = json.loads(raw_content)

    atoms_data = graph_data.get("atoms", [])
    print(f"  Total atoms: {len(atoms_data):,}")

    # Phase 3: Categorize atoms
    print("\n--- Phase 3: Categorize Atoms ---")
    with profile_phase("Separate nodes from links"):
        nodes = [a for a in atoms_data if not a.get("outgoing")]
        links = [a for a in atoms_data if a.get("outgoing")]

    print(f"  Nodes: {len(nodes):,}")
    print(f"  Links: {len(links):,}")

    # Phase 4: Simulate node creation (what _load_graph_state does)
    print("\n--- Phase 4: Node Restoration Simulation ---")

    # 4a: Just the dictionary operations
    with profile_phase("Build id_map (dict operations only)"):
        id_map = {}
        for atom_data in nodes:
            # Simulate what we do per node (without actual graph operations)
            fake_id = atom_data["id"]
            id_map[fake_id] = atom_data  # Just dict assignment

    # 4b: Enum lookups
    from cortical.cognitive.graph import AtomType, TruthValue

    with profile_phase("AtomType enum lookups"):
        for atom_data in nodes:
            _ = AtomType[atom_data["atom_type"]]

    with profile_phase("TruthValue creation"):
        for atom_data in nodes:
            _ = TruthValue(atom_data["tv_strength"], atom_data["tv_confidence"])

    # Phase 5: Simulate actual graph node creation
    print("\n--- Phase 5: Graph Operations (The Real Cost) ---")

    from cortical.cognitive.graph import CognitiveGraph, InMemoryStorage

    # Fresh graph
    storage = InMemoryStorage()
    graph = CognitiveGraph(storage)

    with profile_phase("Create all nodes via graph.node()"):
        id_map_real = {}
        for atom_data in nodes:
            atom_type = AtomType[atom_data["atom_type"]]
            tv = TruthValue(atom_data["tv_strength"], atom_data["tv_confidence"])
            atom = graph.node(atom_data["name"], atom_type=atom_type, tv=tv)
            atom.sti = atom_data.get("sti", 0.0)
            atom.lti = atom_data.get("lti", 0.0)
            id_map_real[atom_data["id"]] = atom

    with profile_phase("Save all nodes to storage"):
        for atom in id_map_real.values():
            storage.save(atom)

    # Phase 6: Link creation
    print("\n--- Phase 6: Link Restoration ---")

    with profile_phase("Resolve link targets from id_map"):
        resolved_links = []
        for atom_data in links:
            targets = []
            for old_id in atom_data["outgoing"]:
                if old_id in id_map_real:
                    targets.append(id_map_real[old_id])
            if len(targets) == len(atom_data["outgoing"]):
                resolved_links.append((atom_data, targets))

    print(f"  Resolved links: {len(resolved_links):,}")

    with profile_phase("Create all links via graph.link()"):
        for atom_data, targets in resolved_links:
            atom_type = AtomType[atom_data["atom_type"]]
            tv = TruthValue(atom_data["tv_strength"], atom_data["tv_confidence"])
            link = graph.link(atom_type, targets, tv)
            link.sti = atom_data.get("sti", 0.0)
            link.lti = atom_data.get("lti", 0.0)
            storage.save(link)

    total_end = time.perf_counter()

    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {total_end - total_start:.3f}s")
    print(f"Final graph size: {len(storage.all_atoms()):,} atoms")
    print("=" * 70)

    # Breakdown analysis
    print("\n--- BOTTLENECK ANALYSIS ---")

    # Profile graph.node() internals
    print("\nDrilling into graph.node() overhead...")

    # Count unique names
    unique_names = set(a["name"] for a in nodes if a.get("name"))
    print(f"  Unique node names: {len(unique_names):,}")

    # Check for duplicate names
    name_counts: Dict[str, int] = {}
    for a in nodes:
        name = a.get("name", "")
        name_counts[name] = name_counts.get(name, 0) + 1

    duplicates = {k: v for k, v in name_counts.items() if v > 1}
    if duplicates:
        print(f"  Duplicate names found: {len(duplicates)}")
        top_dupes = sorted(duplicates.items(), key=lambda x: -x[1])[:5]
        for name, count in top_dupes:
            print(f"    '{name}': {count} times")

    # Profile find_by_name cost
    print("\n  Profiling find_by_name lookups...")
    storage2 = InMemoryStorage()

    # Pre-populate with nodes
    from cortical.cognitive.graph import Atom

    for i, atom_data in enumerate(nodes[:1000]):  # First 1000 only
        atom = Atom(
            id=f"test-{i}",
            atom_type=AtomType.WORD,
            name=atom_data.get("name", ""),
        )
        storage2.save(atom)

    with profile_phase("1000 find_by_name lookups"):
        for atom_data in nodes[:1000]:
            _ = storage2.find_by_name(atom_data.get("name", ""))

    # Profile link() internals - find_by_type is O(n)!
    print("\n  Profiling link creation overhead...")
    print("  WARNING: graph.link() calls find_by_type() which is O(n)!")

    # Count link types
    link_types: Dict[str, int] = {}
    for link in links:
        lt = link["atom_type"]
        link_types[lt] = link_types.get(lt, 0) + 1

    print(f"  Link types distribution:")
    for lt, count in sorted(link_types.items(), key=lambda x: -x[1]):
        print(f"    {lt}: {count:,}")


def profile_optimized_load():
    """Profile an optimized loading approach."""
    print("\n" + "=" * 70)
    print("OPTIMIZED LOAD PROFILING")
    print("=" * 70)

    model_dir = Path("models/cognitive_agent")
    graph_path = model_dir / "bridge" / "graph.json"

    from cortical.cognitive.graph import (
        CognitiveGraph, InMemoryStorage, Atom, AtomType, TruthValue
    )

    total_start = time.perf_counter()

    # Phase 1: Read and parse
    with profile_phase("Read + parse JSON"):
        graph_data = json.loads(graph_path.read_text())

    atoms_data = graph_data.get("atoms", [])

    # OPTIMIZED: Direct atom creation without graph.node() overhead
    with profile_phase("Direct atom creation (bypass graph.node)"):
        storage = InMemoryStorage()
        id_map = {}

        # Pass 1: Create all atoms directly
        for atom_data in atoms_data:
            atom = Atom(
                id=atom_data["id"],
                atom_type=AtomType[atom_data["atom_type"]],
                name=atom_data.get("name", ""),
                outgoing=atom_data.get("outgoing", []),
                tv=TruthValue(
                    atom_data["tv_strength"],
                    atom_data["tv_confidence"]
                ),
                sti=atom_data.get("sti", 0.0),
                lti=atom_data.get("lti", 0.0),
            )
            id_map[atom.id] = atom

    with profile_phase("Batch save all atoms"):
        for atom in id_map.values():
            storage.save(atom)

    total_end = time.perf_counter()

    print("\n" + "=" * 70)
    print(f"OPTIMIZED TOTAL: {total_end - total_start:.3f}s")
    print(f"Final atoms: {len(storage.all_atoms()):,}")
    print("=" * 70)

    return total_end - total_start


if __name__ == "__main__":
    profile_graph_load()
    optimized_time = profile_optimized_load()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. graph.node() overhead: Each call does find_by_name() which is O(1) but
   still has overhead. Direct Atom creation is faster.

2. graph.link() overhead: Each call does find_by_type() which is O(n)!
   For 23,653 links, this is O(n²) worst case.

3. Redundant saves: Current code saves atoms twice (once in graph.node,
   once explicitly). Only one save is needed.

4. Optimization strategy:
   - Create Atom objects directly (bypass graph.node/link methods)
   - Use batch save at the end
   - Avoid find_by_type during load (we have the data, use it directly)

5. Expected improvement: From ~32s to ~2-3s (10-15x faster)
""")
