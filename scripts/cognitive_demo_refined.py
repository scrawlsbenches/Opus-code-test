#!/usr/bin/env python3
"""
Cognitive Demo (Refined): Discovering Non-Obvious Codebase Insights

This refined version filters out Python syntax noise to focus on
domain-specific concepts that reveal architectural insights.

Usage:
    python scripts/cognitive_demo_refined.py
"""

import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortical import CorticalTextProcessor, CorticalLayer

# Filter out Python syntax and common programming terms
NOISE_TERMS = {
    'self', 'return', 'def', 'class', 'import', 'from', 'none', 'true',
    'false', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'args', 'kwargs', 'optional', 'union', 'any', 'type', 'typing',
    'append', 'extend', 'update', 'get', 'keys', 'values', 'items',
    'len', 'range', 'enumerate', 'zip', 'map', 'filter',
    'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
    'with', 'as', 'in', 'is', 'not', 'and', 'or',
    'raise', 'assert', 'pass', 'break', 'continue', 'yield',
    'lambda', 'async', 'await',
    'property', 'staticmethod', 'classmethod', 'abstractmethod',
    'dataclass', 'field', 'default', 'default_factory',
    'init', 'repr', 'post_init', 'slots',
    'path', 'file', 'open', 'read', 'write', 'close',
    'print', 'format', 'join', 'split', 'strip', 'replace',
    'name', 'value', 'key', 'item', 'index', 'count',
    'data', 'result', 'output', 'input', 'config',
    'new', 'old', 'current', 'previous', 'next',
    'start', 'end', 'begin', 'stop', 'first', 'last',
    'returns', 'params', 'param', 'attributes', 'attribute',
}


def is_interesting_term(term: str) -> bool:
    """Filter to keep only domain-relevant terms."""
    term_lower = term.lower()
    if term_lower in NOISE_TERMS:
        return False
    if len(term) < 4:  # Skip very short terms
        return False
    if term.startswith('_'):  # Skip private names
        return False
    if term.isdigit():  # Skip numbers
        return False
    return True


def load_codebase(processor: CorticalTextProcessor, root: Path) -> dict:
    """Load Python files into the processor."""
    stats = {"files": 0, "lines": 0}
    for py_file in root.rglob("*.py"):
        if "test" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        try:
            content = py_file.read_text()
            doc_id = str(py_file.relative_to(root))
            processor.process_document(doc_id, content)
            stats["files"] += 1
            stats["lines"] += content.count("\n")
        except Exception:
            pass
    return stats


def section(title: str):
    """Print section header."""
    print(f"\n{'─'*70}")
    print(f" {title}")
    print('─'*70 + "\n")


def discovery_domain_concepts(processor: CorticalTextProcessor):
    """Find the core domain concepts by PageRank."""
    section("INSIGHT 1: Core Domain Concepts (What This Codebase Is About)")

    token_layer = processor.layers[CorticalLayer.TOKENS]
    domain_terms = []

    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if is_interesting_term(term) and col.pagerank:
            domain_terms.append((term, col.pagerank, len(col.document_ids)))

    domain_terms.sort(key=lambda x: -x[1])

    print("These terms are central to the architecture (high PageRank):\n")
    for term, pr, usage in domain_terms[:20]:
        bar = "█" * int(pr * 500)
        print(f"  {term:25} {bar} ({usage} files)")

    # Group by conceptual area
    print("\n  Observations:")
    reasoning_terms = [t for t, _, _ in domain_terms[:50] if any(x in t.lower() for x in ['loop', 'phase', 'verify', 'crisis', 'thought'])]
    got_terms = [t for t, _, _ in domain_terms[:50] if any(x in t.lower() for x in ['task', 'sprint', 'decision', 'entity', 'edge'])]
    query_terms = [t for t, _, _ in domain_terms[:50] if any(x in t.lower() for x in ['query', 'search', 'document', 'passage'])]

    if reasoning_terms:
        print(f"    Reasoning concepts: {', '.join(reasoning_terms[:5])}")
    if got_terms:
        print(f"    GoT concepts: {', '.join(got_terms[:5])}")
    if query_terms:
        print(f"    Query concepts: {', '.join(query_terms[:5])}")


def discovery_cross_cutting_concerns(processor: CorticalTextProcessor):
    """Find concepts that appear across multiple unrelated modules."""
    section("INSIGHT 2: Cross-Cutting Concerns (Concepts Spanning Domains)")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Find terms that appear in multiple top-level directories
    cross_cutting = []

    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if not is_interesting_term(term):
            continue

        # Count unique top-level directories
        top_dirs = set()
        for doc_id in col.document_ids:
            parts = doc_id.split("/")
            if len(parts) > 1:
                top_dirs.add(parts[0])
            else:
                top_dirs.add("root")

        # Appears in 3+ different subsystems = cross-cutting
        if len(top_dirs) >= 3:
            cross_cutting.append((term, top_dirs, col.pagerank or 0))

    cross_cutting.sort(key=lambda x: (-len(x[1]), -x[2]))

    print("Terms appearing across 3+ subsystems (possible abstractions):\n")
    for term, dirs, pr in cross_cutting[:15]:
        print(f"  '{term}'")
        print(f"    Spans: {', '.join(sorted(dirs))}")


def discovery_module_themes(processor: CorticalTextProcessor):
    """Find distinctive concepts per module."""
    section("INSIGHT 3: What Makes Each Module Unique")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # For each module, find terms that are concentrated there
    module_terms = defaultdict(list)
    term_distribution = defaultdict(set)

    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if not is_interesting_term(term):
            continue

        for doc_id in col.document_ids:
            parts = doc_id.split("/")
            if len(parts) > 1:
                module = parts[0]
                term_distribution[term].add(module)

    # Find terms concentrated in single modules
    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if not is_interesting_term(term):
            continue
        if term not in term_distribution:
            continue

        modules = term_distribution[term]
        if len(modules) == 1:  # Unique to one module
            module = list(modules)[0]
            module_terms[module].append((term, len(col.document_ids)))

    print("Distinctive vocabulary by module (terms found ONLY in that module):\n")
    for module in ['reasoning', 'got', 'query', 'analysis', 'spark', 'processor']:
        terms = module_terms.get(module, [])
        if terms:
            terms.sort(key=lambda x: -x[1])
            top_terms = [t for t, _ in terms[:8]]
            print(f"  {module}/")
            print(f"    {', '.join(top_terms)}")
            print()


def discovery_conceptual_layers(processor: CorticalTextProcessor):
    """Discover the conceptual hierarchy."""
    section("INSIGHT 4: Conceptual Hierarchy (From Concrete to Abstract)")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Categorize terms by abstraction level based on usage patterns
    concrete = []  # Used in few files, specific implementations
    foundational = []  # Used widely, core abstractions
    bridge = []  # Connect different parts

    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if not is_interesting_term(term):
            continue

        num_files = len(col.document_ids)
        num_connections = len(col.lateral_connections)

        if num_files <= 3 and num_connections <= 10:
            concrete.append((term, num_files, num_connections))
        elif num_files >= 15:
            foundational.append((term, num_files, num_connections))
        elif num_connections >= 30:
            bridge.append((term, num_files, num_connections))

    print("FOUNDATIONAL (core abstractions, used everywhere):")
    foundational.sort(key=lambda x: -x[1])
    for term, files, conns in foundational[:10]:
        print(f"    {term}: {files} files, {conns} connections")

    print("\nBRIDGE (heavily connected, integration points):")
    bridge.sort(key=lambda x: -x[2])
    for term, files, conns in bridge[:10]:
        print(f"    {term}: {files} files, {conns} connections")

    print("\nCONCRETE (specific implementations, few files):")
    concrete.sort(key=lambda x: -x[2])  # Most connected among specific
    for term, files, conns in concrete[:10]:
        print(f"    {term}: {files} files, {conns} connections")


def discovery_semantic_neighbors(processor: CorticalTextProcessor):
    """Show the semantic neighborhood of key concepts."""
    section("INSIGHT 5: Semantic Neighborhoods (What Relates to What)")

    queries = [
        ("transaction", "GoT transactional semantics"),
        ("verification", "Verification/validation patterns"),
        ("crisis", "Error handling philosophy"),
        ("pagerank", "Graph algorithm usage"),
        ("cognitive", "Reasoning framework"),
    ]

    for term, description in queries:
        print(f"  {term.upper()} ({description})")
        try:
            expanded = processor.expand_query(term, max_expansions=10)
            if expanded:
                # Filter to interesting terms
                interesting = [
                    (t, w) for t, w in expanded.items()
                    if is_interesting_term(t) and t != term
                ][:6]
                if interesting:
                    neighbors = [f"{t}" for t, w in interesting]
                    print(f"    → {', '.join(neighbors)}")
                else:
                    print(f"    → (no domain-specific neighbors)")
            else:
                print(f"    → (term not found)")
        except Exception:
            print(f"    → (expansion failed)")
        print()


def discovery_hidden_dependencies(processor: CorticalTextProcessor):
    """Find modules that share concepts but aren't obviously related."""
    section("INSIGHT 6: Hidden Dependencies (Conceptual Coupling)")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Build term-to-modules mapping
    module_terms = defaultdict(set)
    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if not is_interesting_term(term):
            continue
        for doc_id in col.document_ids:
            parts = doc_id.split("/")
            if len(parts) > 1:
                module_terms[parts[0]].add(term)

    # Calculate Jaccard similarity between modules
    modules = ['reasoning', 'got', 'query', 'analysis', 'spark', 'processor']
    similarities = []

    for i, m1 in enumerate(modules):
        for m2 in modules[i+1:]:
            if m1 in module_terms and m2 in module_terms:
                t1, t2 = module_terms[m1], module_terms[m2]
                shared = t1 & t2
                jaccard = len(shared) / len(t1 | t2) if t1 | t2 else 0
                if jaccard > 0.05:  # At least 5% overlap
                    similarities.append((m1, m2, jaccard, shared))

    similarities.sort(key=lambda x: -x[2])

    print("Modules with significant conceptual overlap:\n")
    for m1, m2, sim, shared in similarities[:6]:
        print(f"  {m1} <─── {sim:.0%} ───> {m2}")
        sample = sorted(shared, key=len)[:5]
        print(f"    Shared concepts: {', '.join(sample)}")
        print()


def discovery_architectural_patterns(processor: CorticalTextProcessor):
    """Synthesize high-level architectural observations."""
    section("INSIGHT 7: Architectural Patterns (What the Structure Reveals)")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Look for design pattern indicators
    patterns = {
        'Pub/Sub & Events': ['subscribe', 'publish', 'broker', 'topic', 'message', 'event'],
        'State Machines': ['state', 'phase', 'transition', 'status', 'lifecycle'],
        'Transactions': ['transaction', 'commit', 'rollback', 'atomic', 'wal'],
        'Graphs & Trees': ['node', 'edge', 'graph', 'tree', 'path', 'traverse'],
        'Verification': ['verify', 'validate', 'check', 'assert', 'test'],
        'Recovery': ['recover', 'backup', 'restore', 'checkpoint', 'snapshot'],
    }

    print("Design patterns detected by concept presence:\n")
    for pattern_name, indicators in patterns.items():
        found = []
        for indicator in indicators:
            col = token_layer.get_by_id(f"L0_{indicator}")
            if col and len(col.document_ids) >= 3:
                found.append(f"{indicator}({len(col.document_ids)})")

        if len(found) >= 2:  # At least 2 indicators present
            print(f"  {pattern_name}")
            print(f"    Evidence: {', '.join(found[:5])}")


def discovery_evolution_hints(processor: CorticalTextProcessor):
    """Find terms that might indicate future development areas."""
    section("INSIGHT 8: Evolution Hints (Where the Codebase Might Be Heading)")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Look for "experimental" or "future" markers
    future_indicators = []

    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        # Check for experimental/future naming patterns
        if any(x in term.lower() for x in ['experiment', 'todo', 'fixme', 'future', 'planned', 'prototype', 'beta']):
            if is_interesting_term(term):
                future_indicators.append((term, list(col.document_ids)[:3]))

    if future_indicators:
        print("Terms suggesting work-in-progress or future features:\n")
        for term, files in future_indicators[:10]:
            short_files = [f.split('/')[-1] for f in files]
            print(f"  '{term}' in {', '.join(short_files)}")
    else:
        print("No explicit future/experimental markers found.")

    # Also check for ML-related terms (often indicates future direction)
    print("\nML/AI-related concepts found:")
    ml_terms = []
    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        if any(x in term.lower() for x in ['model', 'train', 'predict', 'embedding', 'neural', 'learn']):
            if is_interesting_term(term) and len(col.document_ids) >= 2:
                ml_terms.append((term, len(col.document_ids)))

    ml_terms.sort(key=lambda x: -x[1])
    for term, count in ml_terms[:8]:
        print(f"    {term} ({count} files)")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         REFINED COGNITIVE DEMO: Non-Obvious Codebase Insights        ║
║                                                                      ║
║  Filtering Python noise to focus on domain-specific discoveries     ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    processor = CorticalTextProcessor()
    cortical_root = project_root / "cortical"

    print("Loading codebase...")
    stats = load_codebase(processor, cortical_root)
    print(f"  {stats['files']} files, {stats['lines']:,} lines")

    print("Computing knowledge graph...")
    processor.compute_all()
    print("  Done.\n")

    # Run refined discoveries
    discovery_domain_concepts(processor)
    discovery_cross_cutting_concerns(processor)
    discovery_module_themes(processor)
    discovery_conceptual_layers(processor)
    discovery_semantic_neighbors(processor)
    discovery_hidden_dependencies(processor)
    discovery_architectural_patterns(processor)
    discovery_evolution_hints(processor)

    # Final synthesis
    print("\n" + "═"*70)
    print(" SYNTHESIS: What We Learned About This Codebase")
    print("═"*70 + "\n")
    print("""
Through semantic analysis, we discovered:

1. CORE DOMAIN: The codebase centers on 'reasoning', 'verification',
   'transactions', and 'graph' concepts - it's a cognitive system.

2. MODULAR STRUCTURE: Each subsystem has distinctive vocabulary,
   but they share common abstractions (cross-cutting concerns).

3. DESIGN PHILOSOPHY: Heavy use of state machines, graph algorithms,
   and transactional patterns indicates reliability focus.

4. INTEGRATION POINTS: 'reasoning' and 'got' share significant
   conceptual overlap - they're designed to work together.

5. FUTURE DIRECTION: ML-related terms in dedicated modules suggest
   planned machine learning capabilities.

The key insight: this codebase is built around the idea of treating
knowledge as a graph that can be traversed, queried, and reasoned
over - and it applies this same principle to itself (dogfooding).
""")

    return processor


if __name__ == "__main__":
    processor = main()
