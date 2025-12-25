#!/usr/bin/env python3
"""
Cognitive Demo: Exploring the Codebase with Cortical Text Processor

This script demonstrates the power of the knowledge graph by analyzing
the Cortical Text Processor codebase itself (dogfooding) and discovering
insights that wouldn't be obvious through casual browsing.

Usage:
    python scripts/cognitive_demo.py
"""

import sys
from pathlib import Path
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortical import CorticalTextProcessor, CorticalLayer


def load_codebase(processor: CorticalTextProcessor, root: Path) -> dict:
    """Load all Python files into the processor."""
    stats = {"files": 0, "lines": 0, "chars": 0}

    for py_file in root.rglob("*.py"):
        # Skip test files for core analysis, but we'll note them
        if "test" in py_file.parts:
            continue
        if "__pycache__" in py_file.parts:
            continue

        try:
            content = py_file.read_text()
            doc_id = str(py_file.relative_to(root))
            processor.process_document(doc_id, content)
            stats["files"] += 1
            stats["lines"] += content.count("\n")
            stats["chars"] += len(content)
        except Exception as e:
            print(f"  Skipped {py_file}: {e}")

    return stats


def print_section(title: str, char: str = "═"):
    """Print a formatted section header."""
    width = 70
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def demo_hidden_connections(processor: CorticalTextProcessor):
    """Find terms that connect otherwise unrelated modules."""
    print_section("DISCOVERY 1: Hidden Conceptual Bridges", "─")
    print("""
These terms act as bridges between different parts of the codebase.
High betweenness centrality means they connect clusters that would
otherwise be isolated. These are critical knowledge points.
""")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Find terms with high lateral connections spanning modules
    bridge_candidates = []
    for col in token_layer.minicolumns.values():
        if col.pagerank and col.pagerank > 0.001:
            # Count unique source documents for connections
            # lateral_connections is a dict: {target_id: weight}
            connected_docs = set()
            for target_id in col.lateral_connections.keys():
                target_col = token_layer.get_by_id(target_id)
                if target_col:
                    connected_docs.update(target_col.document_ids)

            if len(connected_docs) > 5:  # Connects 5+ different files
                bridge_candidates.append((
                    col.id.replace("L0_", ""),
                    col.pagerank,
                    len(connected_docs),
                    list(connected_docs)[:5]  # Sample of files
                ))

    # Sort by number of connections
    bridge_candidates.sort(key=lambda x: -x[2])

    print("Top Bridge Concepts (connect 5+ modules):\n")
    for term, pr, num_docs, sample_docs in bridge_candidates[:10]:
        print(f"  '{term}' (PageRank: {pr:.4f})")
        print(f"    Connects {num_docs} modules including:")
        for doc in sample_docs[:3]:
            # Shorten path for display
            short = "/".join(doc.split("/")[-2:]) if "/" in doc else doc
            print(f"      - {short}")
        print()


def demo_semantic_clusters(processor: CorticalTextProcessor):
    """Reveal emergent topic clusters without explicit labeling."""
    print_section("DISCOVERY 2: Emergent Topic Clusters", "─")
    print("""
The system automatically clusters related concepts using community
detection (Louvain algorithm). No manual labeling required - these
themes emerged from analyzing co-occurrence patterns.
""")

    concept_layer = processor.layers.get(CorticalLayer.CONCEPTS)
    if not concept_layer or not concept_layer.minicolumns:
        # Build clusters if not present
        processor.build_concept_clusters()
        concept_layer = processor.layers.get(CorticalLayer.CONCEPTS)

    if concept_layer and concept_layer.minicolumns:
        print("Discovered Conceptual Clusters:\n")

        clusters = list(concept_layer.minicolumns.values())[:8]
        for i, cluster in enumerate(clusters, 1):
            # Get top terms in cluster
            terms = [
                t.replace("L0_", "")
                for t in list(cluster.document_ids)[:8]
            ]
            print(f"  Cluster {i}: {', '.join(terms)}")

            # What files does this cluster primarily appear in?
            token_layer = processor.layers[CorticalLayer.TOKENS]
            cluster_files = set()
            for term_id in list(cluster.document_ids)[:5]:
                col = token_layer.get_by_id(term_id)
                if col:
                    cluster_files.update(col.document_ids)

            if cluster_files:
                sample = list(cluster_files)[:3]
                print(f"    Found primarily in: {', '.join(s.split('/')[-1] for s in sample)}")
            print()
    else:
        print("  (Clustering requires more document overlap)")


def demo_knowledge_gaps(processor: CorticalTextProcessor):
    """Find potential documentation or integration gaps."""
    print_section("DISCOVERY 3: Knowledge Gaps & Islands", "─")
    print("""
These are concepts that exist in isolation - they're used but not
well-connected to the broader codebase. They may indicate:
- Missing documentation
- Unused code paths
- Integration opportunities
- Specialized utilities that could be generalized
""")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Find isolated terms (used in code but few connections)
    isolated = []
    for col in token_layer.minicolumns.values():
        if len(col.document_ids) >= 2:  # Used in multiple places
            connection_count = len(col.lateral_connections)
            if connection_count < 3:  # But poorly connected
                isolated.append((
                    col.id.replace("L0_", ""),
                    len(col.document_ids),
                    connection_count,
                    list(col.document_ids)[:3]
                ))

    isolated.sort(key=lambda x: (-x[1], x[2]))  # Most used, least connected

    print("Isolated Concepts (used often, connected rarely):\n")
    for term, usage, connections, files in isolated[:10]:
        print(f"  '{term}'")
        print(f"    Used in {usage} files, only {connections} semantic connections")
        print(f"    Files: {', '.join(f.split('/')[-1] for f in files)}")
        print()


def demo_unexpected_relationships(processor: CorticalTextProcessor):
    """Find surprising term co-occurrences."""
    print_section("DISCOVERY 4: Unexpected Relationships", "─")
    print("""
These term pairs appear together more often than expected by chance.
Surprising co-occurrences often reveal hidden design patterns,
cross-cutting concerns, or architectural decisions.
""")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Look for high-weight edges between semantically distant terms
    surprising = []
    seen = set()

    for col in token_layer.minicolumns.values():
        term1 = col.id.replace("L0_", "")

        # lateral_connections is a dict: {target_id: weight}
        for target_id, weight in col.lateral_connections.items():
            if weight and weight > 0.3:
                pair_key = tuple(sorted([col.id, target_id]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                other_col = token_layer.get_by_id(target_id)
                if other_col:
                    term2 = target_id.replace("L0_", "")

                    # Skip obvious pairs (similar words)
                    if len(term1) > 3 and len(term2) > 3 and term1[:4] == term2[:4]:
                        continue
                    if term1 in term2 or term2 in term1:
                        continue

                    # Get shared context
                    shared_files = col.document_ids & other_col.document_ids

                    if shared_files:
                        surprising.append((
                            term1, term2,
                            weight,
                            list(shared_files)[:2]
                        ))

    surprising.sort(key=lambda x: -x[2])

    print("Strongly Correlated Term Pairs:\n")
    for t1, t2, weight, files in surprising[:12]:
        print(f"  '{t1}' <-> '{t2}' (correlation: {weight:.2f})")
        print(f"    Context: {', '.join(f.split('/')[-1] for f in files)}")
        print()


def demo_central_concepts(processor: CorticalTextProcessor):
    """Identify the most important concepts by PageRank."""
    print_section("DISCOVERY 5: Central Concepts (PageRank)", "─")
    print("""
PageRank reveals which concepts are most "important" in the semantic
network - not by frequency, but by how many other important concepts
reference them. High PageRank = foundational concept.
""")

    token_layer = processor.layers[CorticalLayer.TOKENS]

    # Get top PageRank terms
    ranked = []
    for col in token_layer.minicolumns.values():
        if col.pagerank and col.pagerank > 0:
            term = col.id.replace("L0_", "")
            # Skip very short terms
            if len(term) > 3:
                ranked.append((term, col.pagerank, len(col.document_ids)))

    ranked.sort(key=lambda x: -x[1])

    print("Most Central Concepts by PageRank:\n")
    print("  Term                 PageRank    Used In")
    print("  " + "-" * 45)
    for term, pr, usage in ranked[:15]:
        print(f"  {term:20} {pr:.4f}      {usage} files")


def demo_query_expansion(processor: CorticalTextProcessor):
    """Show how query expansion finds related concepts."""
    print_section("DISCOVERY 6: Query Expansion Examples", "─")
    print("""
Query expansion uses the semantic graph to find related terms.
This shows what the system "understands" about each concept.
""")

    queries = ["transaction", "graph", "verification", "crisis"]

    for query in queries:
        print(f"\n  Query: '{query}'")
        try:
            expanded = processor.expand_query(query, max_expansions=8)
            if expanded:
                terms = [f"{t} ({w:.2f})" for t, w in list(expanded.items())[:8]]
                print(f"    Expands to: {', '.join(terms)}")
            else:
                print(f"    (no expansions found)")
        except Exception as e:
            print(f"    (expansion failed: {e})")


def demo_module_similarity(processor: CorticalTextProcessor):
    """Find which modules are semantically similar."""
    print_section("DISCOVERY 7: Module Similarity Matrix", "─")
    print("""
By comparing semantic fingerprints, we can find modules that deal
with similar concepts - even if they're in different directories.
This reveals hidden architectural relationships.
""")

    doc_layer = processor.layers.get(CorticalLayer.DOCUMENTS)
    if not doc_layer:
        print("  (Document layer not populated)")
        return

    # Get fingerprints for each document
    docs = list(doc_layer.minicolumns.keys())[:20]  # Sample

    # Simple Jaccard similarity based on shared terms
    token_layer = processor.layers[CorticalLayer.TOKENS]
    doc_terms = defaultdict(set)

    for col in token_layer.minicolumns.values():
        for doc_id in col.document_ids:
            doc_terms[doc_id].add(col.id)

    # Find most similar pairs
    similarities = []
    for i, doc1 in enumerate(docs):
        for doc2 in docs[i+1:]:
            if doc1 in doc_terms and doc2 in doc_terms:
                t1, t2 = doc_terms[doc1], doc_terms[doc2]
                if t1 and t2:
                    jaccard = len(t1 & t2) / len(t1 | t2)
                    if jaccard > 0.2:  # At least 20% overlap
                        similarities.append((doc1, doc2, jaccard))

    similarities.sort(key=lambda x: -x[2])

    print("Semantically Similar Module Pairs:\n")
    for doc1, doc2, sim in similarities[:10]:
        # Shorten paths
        d1 = doc1.split("/")[-1]
        d2 = doc2.split("/")[-1]
        print(f"  {d1:25} <-> {d2:25} ({sim:.0%} overlap)")


def demo_architectural_insights(processor: CorticalTextProcessor):
    """High-level architectural observations."""
    print_section("DISCOVERY 8: Architectural Insights", "─")
    print("""
Synthesizing the analysis above, here are architectural patterns
that emerge from the semantic structure:
""")

    # Count terms by module prefix
    token_layer = processor.layers[CorticalLayer.TOKENS]
    module_concepts = defaultdict(set)

    for col in token_layer.minicolumns.values():
        term = col.id.replace("L0_", "")
        for doc_id in col.document_ids:
            parts = doc_id.split("/")
            if len(parts) > 1:
                module = parts[0]  # Top-level directory
                module_concepts[module].add(term)

    print("Concept Distribution by Module:\n")
    for module, concepts in sorted(module_concepts.items(),
                                    key=lambda x: -len(x[1]))[:8]:
        print(f"  {module}: {len(concepts)} unique concepts")
        # Sample distinguishing concepts (appear here but rarely elsewhere)
        unique_to_module = []
        for term in list(concepts)[:50]:
            term_id = f"L0_{term}"
            col = token_layer.get_by_id(term_id)
            if col:
                # Check if primarily in this module
                module_docs = [d for d in col.document_ids if d.startswith(module)]
                if len(module_docs) >= len(col.document_ids) * 0.7:
                    unique_to_module.append(term)

        if unique_to_module:
            print(f"    Distinctive terms: {', '.join(unique_to_module[:5])}")
        print()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              CORTICAL TEXT PROCESSOR - COGNITIVE DEMO                ║
║                                                                      ║
║  Dogfooding: Analyzing our own codebase to discover hidden insights ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Initialize processor
    processor = CorticalTextProcessor()

    # Load codebase
    print("Loading codebase...")
    project_root = Path(__file__).parent.parent
    cortical_root = project_root / "cortical"

    stats = load_codebase(processor, cortical_root)
    print(f"  Loaded {stats['files']} files ({stats['lines']:,} lines, {stats['chars']:,} chars)")

    # Compute all analyses
    print("\nComputing semantic analysis...")
    processor.compute_all()
    print("  Done. Knowledge graph constructed.")

    # Run discoveries
    demo_central_concepts(processor)
    demo_hidden_connections(processor)
    demo_semantic_clusters(processor)
    demo_unexpected_relationships(processor)
    demo_knowledge_gaps(processor)
    demo_query_expansion(processor)
    demo_module_similarity(processor)
    demo_architectural_insights(processor)

    # Summary
    print_section("SUMMARY: What We Learned", "═")
    print("""
This demo showed 8 types of insights the knowledge graph can surface:

1. CENTRAL CONCEPTS - Which terms are foundational (PageRank)
2. BRIDGE CONCEPTS - Terms connecting different modules
3. EMERGENT CLUSTERS - Topics that self-organized from patterns
4. UNEXPECTED PAIRS - Surprising correlations between concepts
5. KNOWLEDGE GAPS - Isolated concepts that might need attention
6. QUERY EXPANSION - How the system "understands" relationships
7. MODULE SIMILARITY - Which files deal with similar concepts
8. ARCHITECTURAL PATTERNS - High-level structure insights

The key insight: by treating code as interconnected knowledge rather
than isolated files, we can discover relationships and patterns that
aren't visible through traditional code browsing or grep searches.
""")

    return processor  # Return for interactive exploration


if __name__ == "__main__":
    processor = main()
