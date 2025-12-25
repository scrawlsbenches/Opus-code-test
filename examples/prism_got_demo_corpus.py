#!/usr/bin/env python3
"""
PRISM-GoT Corpus Demo: Synaptic Memory with Real Documents

This demo builds a PRISM-GoT reasoning graph from the actual samples/ corpus,
demonstrating how synaptic memory learns from real document relationships.

Run with: python examples/prism_got_demo_corpus.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer
from cortical.reasoning import (
    NodeType,
    EdgeType,
    SynapticMemoryGraph,
    IncrementalReasoner,
    ActivationTrace,
)


def load_corpus_samples(samples_dir: Path, max_docs: int = 20) -> dict:
    """Load sample documents from the corpus."""
    docs = {}

    # Find text and markdown files
    patterns = ["**/*.txt", "**/*.md"]
    files = []
    for pattern in patterns:
        files.extend(samples_dir.glob(pattern))

    # Sort by name and limit
    files = sorted(files)[:max_docs]

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            # Use relative path as doc_id
            doc_id = str(filepath.relative_to(samples_dir))
            docs[doc_id] = content
        except Exception:
            continue

    return docs


def extract_key_concepts(processor: CorticalTextProcessor, doc_id: str, top_n: int = 5) -> list:
    """Extract key concepts from a document using TF-IDF."""
    from cortical.layers import CorticalLayer

    layer0 = processor.get_layer(CorticalLayer.TOKENS)

    # Find terms in this document with highest TF-IDF
    doc_terms = []
    for col in layer0.minicolumns.values():
        if doc_id in col.document_ids:
            tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
            if tfidf > 0:
                doc_terms.append((col.content, tfidf))

    # Sort by TF-IDF and return top terms
    doc_terms.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in doc_terms[:top_n]]


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    print("\n" + "="*60)
    print("  PRISM-GoT Corpus Demo")
    print("  Learning from Real Documents")
    print("="*60)

    # Find samples directory
    samples_dir = Path(__file__).parent.parent / "samples"
    if not samples_dir.exists():
        print(f"Error: samples directory not found at {samples_dir}")
        return

    print_section("1. LOADING CORPUS")

    # Load documents
    docs = load_corpus_samples(samples_dir, max_docs=15)
    print(f"\nLoaded {len(docs)} documents from samples/")

    for doc_id in list(docs.keys())[:5]:
        preview = docs[doc_id][:60].replace('\n', ' ')
        print(f"  - {doc_id}: {preview}...")
    if len(docs) > 5:
        print(f"  ... and {len(docs) - 5} more")

    print_section("2. BUILDING CORTICAL INDEX")

    # Build cortical processor for concept extraction
    processor = CorticalTextProcessor()
    for doc_id, content in docs.items():
        processor.process_document(doc_id, content)
    processor.compute_all()

    print(f"\nIndexed {len(docs)} documents")
    print(f"  Tokens: {processor.get_layer(CorticalLayer.TOKENS).column_count()}")
    print(f"  Bigrams: {processor.get_layer(CorticalLayer.BIGRAMS).column_count()}")
    print(f"  Concepts: {processor.get_layer(CorticalLayer.CONCEPTS).column_count()}")

    print_section("3. BUILDING SYNAPTIC MEMORY GRAPH")

    # Create synaptic graph
    graph = SynapticMemoryGraph()
    reasoner = IncrementalReasoner(graph, auto_link_similar=True, similarity_threshold=0.3)

    # Add documents as CONTEXT nodes
    doc_nodes = {}
    for doc_id, content in docs.items():
        # Extract title from first line or filename
        lines = content.strip().split('\n')
        title = lines[0].strip('# ').strip()[:50] if lines else doc_id

        node = reasoner.process_thought(
            content=title,
            node_type=NodeType.CONTEXT,
            node_id=f"DOC:{doc_id}",
        )
        doc_nodes[doc_id] = node
        reasoner.reset_focus()  # Don't chain documents

    print(f"\nCreated {len(doc_nodes)} document nodes")

    # Extract concepts and create CONCEPT nodes
    print("\nExtracting key concepts from documents...")
    concept_nodes = {}

    for doc_id in docs.keys():
        concepts = extract_key_concepts(processor, doc_id, top_n=3)

        for concept in concepts:
            concept_key = concept.lower()

            # Create concept node if new
            if concept_key not in concept_nodes:
                concept_node = graph.add_node(
                    f"CONCEPT:{concept_key}",
                    NodeType.CONCEPT,
                    concept,
                )
                concept_nodes[concept_key] = concept_node

            # Link document to concept
            doc_node_id = f"DOC:{doc_id}"
            concept_node_id = f"CONCEPT:{concept_key}"

            if (doc_node_id, concept_node_id, EdgeType.CONTAINS) not in graph.synaptic_edges:
                graph.add_synaptic_edge(
                    doc_node_id,
                    concept_node_id,
                    EdgeType.CONTAINS,
                    weight=0.7,
                )

    print(f"Created {len(concept_nodes)} concept nodes")
    print(f"Total edges: {len(graph.synaptic_edges)}")

    # Show some concepts
    print("\nSample concepts found:")
    for concept in list(concept_nodes.keys())[:10]:
        print(f"  - {concept}")

    print_section("4. SIMULATING REASONING SESSIONS")

    # Simulate user exploring documents about specific topics
    exploration_paths = [
        ("neural", ["neural", "network", "learning"]),
        ("memory", ["memory", "consolidation", "cognitive"]),
        ("graph", ["graph", "pagerank", "network"]),
    ]

    for topic, related_concepts in exploration_paths:
        print(f"\n--- Exploring '{topic}' ---")

        # Find documents containing this topic
        topic_docs = []
        for doc_id, content in docs.items():
            if topic.lower() in content.lower():
                topic_docs.append(doc_id)

        if not topic_docs:
            print(f"  No documents found for '{topic}'")
            continue

        print(f"  Found {len(topic_docs)} related documents")

        # Simulate activation: user reads documents about this topic
        for doc_id in topic_docs[:3]:
            doc_node_id = f"DOC:{doc_id}"
            if doc_node_id in graph.nodes:
                graph.activate_node(doc_node_id, context={"topic": topic})

                # Also activate connected concepts
                for edge in graph.get_synaptic_edges_from(doc_node_id):
                    if edge.target_id.startswith("CONCEPT:"):
                        graph.activate_node(edge.target_id, context={"topic": topic})

        # Apply Hebbian learning
        strengthened = graph.apply_hebbian_learning(time_window_seconds=300)
        print(f"  Activated documents and concepts, strengthened {strengthened} edges")

    print_section("5. PREDICTIONS FROM LEARNED PATTERNS")

    # Pick a document and predict related content
    sample_docs = [d for d in docs.keys() if 'neural' in d.lower() or 'memory' in d.lower()]
    if not sample_docs:
        sample_docs = list(docs.keys())[:3]

    for doc_id in sample_docs[:2]:
        doc_node_id = f"DOC:{doc_id}"
        if doc_node_id not in graph.nodes:
            continue

        print(f"\nFrom document: {doc_id}")

        predictions = graph.predict_next_thoughts(doc_node_id, top_n=5)

        if predictions:
            print("  Predicted related concepts:")
            for pred in predictions:
                concept_name = pred.node_id.replace("CONCEPT:", "")
                print(f"    - {concept_name} (prob={pred.probability:.3f})")
        else:
            print("  No predictions available")

    print_section("6. CONCEPT SIMILARITY NETWORK")

    # Find concepts that got linked through SIMILAR edges
    similar_pairs = []
    for (src, tgt, etype), edge in graph.synaptic_edges.items():
        if etype == EdgeType.SIMILAR and src.startswith("CONCEPT:"):
            similar_pairs.append((
                src.replace("CONCEPT:", ""),
                tgt.replace("CONCEPT:", ""),
                edge.weight
            ))

    if similar_pairs:
        print("\nSimilar concept pairs discovered:")
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        for c1, c2, weight in similar_pairs[:10]:
            print(f"  {c1} <-> {c2} (similarity={weight:.2f})")
    else:
        print("\nNo similar concept pairs discovered (threshold may be too high)")

    print_section("7. GRAPH SUMMARY")

    summary = {
        "total_nodes": graph.node_count(),
        "total_edges": len(graph.synaptic_edges),
        "document_nodes": len([n for n in graph.nodes if n.startswith("DOC:")]),
        "concept_nodes": len([n for n in graph.nodes if n.startswith("CONCEPT:")]),
    }

    # Calculate average edge weight
    if graph.synaptic_edges:
        avg_weight = sum(e.weight for e in graph.synaptic_edges.values()) / len(graph.synaptic_edges)
        summary["avg_edge_weight"] = avg_weight

    # Count activations
    total_activations = sum(t.total_activations for t in graph.activation_traces.values())
    summary["total_activations"] = total_activations

    print(f"""
Graph Statistics:
  Documents:    {summary['document_nodes']}
  Concepts:     {summary['concept_nodes']}
  Total Nodes:  {summary['total_nodes']}
  Total Edges:  {summary['total_edges']}
  Avg Weight:   {summary.get('avg_edge_weight', 0):.3f}
  Activations:  {summary['total_activations']}
    """)

    print_section("SUMMARY")
    print("""
This demo showed PRISM-GoT learning from a real document corpus:

1. Loaded documents from samples/ directory
2. Built a cortical index for concept extraction
3. Created a synaptic memory graph with:
   - Document nodes (CONTEXT type)
   - Concept nodes extracted via TF-IDF
   - CONTAINS edges linking documents to concepts

4. Simulated exploration sessions:
   - Activated documents and their concepts
   - Applied Hebbian learning to strengthen co-activated paths

5. Made predictions based on learned patterns:
   - Documents predict related concepts
   - Frequently co-activated paths become preferred

The graph now reflects actual usage patterns and can predict
which concepts are most relevant given a starting document.
    """)


if __name__ == "__main__":
    main()
