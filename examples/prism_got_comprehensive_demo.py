#!/usr/bin/env python3
"""
PRISM-GoT Comprehensive Demo: Learning from the Full Corpus

This demo showcases the full power of Predictive Reasoning through
Incremental Synaptic Memory Graph of Thought by:

1. Loading the entire samples/ corpus recursively
2. Building a rich knowledge graph with multiple node types
3. Simulating realistic exploration sessions
4. Discovering emergent patterns and clusters
5. Demonstrating predictive capabilities
6. Showing how the system learns and improves

Run with: python examples/prism_got_comprehensive_demo.py
"""

import sys
from collections import defaultdict
from datetime import datetime, timedelta
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
    PlasticityRules,
)


# =============================================================================
# UTILITIES
# =============================================================================

def print_header(title: str) -> None:
    """Print a major section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_section(title: str) -> None:
    """Print a minor section header."""
    print(f"\n--- {title} ---")


def load_full_corpus(samples_dir: Path) -> dict:
    """Load all text and markdown files recursively."""
    docs = {}
    patterns = ["**/*.txt", "**/*.md"]

    for pattern in patterns:
        for filepath in samples_dir.glob(pattern):
            try:
                content = filepath.read_text(encoding="utf-8")
                # Use relative path as doc_id
                doc_id = str(filepath.relative_to(samples_dir))
                docs[doc_id] = content
            except Exception:
                continue

    return docs


def extract_title(content: str, fallback: str) -> str:
    """Extract title from document content."""
    lines = content.strip().split('\n')
    if lines:
        title = lines[0].strip('# ').strip()
        if len(title) > 60:
            title = title[:57] + "..."
        return title
    return fallback


def categorize_document(doc_id: str, content: str) -> str:
    """Categorize a document by its apparent domain."""
    doc_lower = doc_id.lower() + " " + content[:500].lower()

    categories = {
        "cognitive_science": ["cognitive", "memory", "neural", "brain", "learning", "attention"],
        "ai_development": ["agent", "ai", "model", "prediction", "algorithm", "machine"],
        "software_engineering": ["code", "software", "development", "testing", "api", "function"],
        "knowledge_management": ["knowledge", "document", "corpus", "index", "search", "retrieval"],
        "market_analysis": ["market", "trading", "financial", "economic", "price", "investor"],
        "philosophy": ["philosophy", "logic", "argument", "ethics", "reasoning", "epistem"],
        "religion": ["religious", "spiritual", "faith", "tradition", "sacred"],
        "archaeology": ["archaeolog", "artifact", "excavat", "ancient", "lithic"],
        "social_psychology": ["social", "influence", "persuasion", "behavior", "group"],
        "troubleshooting": ["troubleshoot", "debug", "error", "fix", "issue", "problem"],
    }

    for category, keywords in categories.items():
        if any(kw in doc_lower for kw in keywords):
            return category

    return "general"


def extract_key_terms(processor: CorticalTextProcessor, doc_id: str, top_n: int = 5) -> list:
    """Extract key terms from a document using TF-IDF."""
    layer0 = processor.get_layer(CorticalLayer.TOKENS)

    doc_terms = []
    for col in layer0.minicolumns.values():
        if doc_id in col.document_ids:
            tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
            if tfidf > 0 and len(col.content) > 2:  # Skip short terms
                doc_terms.append((col.content, tfidf))

    doc_terms.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in doc_terms[:top_n]]


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  PRISM-GoT: Comprehensive Knowledge Discovery Demo")
    print("  Predictive Reasoning through Incremental Synaptic Memory")
    print("="*70)

    samples_dir = Path(__file__).parent.parent / "samples"
    if not samples_dir.exists():
        print(f"Error: samples directory not found at {samples_dir}")
        return

    # =========================================================================
    # PHASE 1: LOAD CORPUS
    # =========================================================================
    print_header("PHASE 1: LOADING THE KNOWLEDGE CORPUS")

    docs = load_full_corpus(samples_dir)
    print(f"\nLoaded {len(docs)} documents from samples/")

    # Show directory structure
    subdirs = defaultdict(list)
    for doc_id in docs:
        parts = doc_id.split('/')
        if len(parts) > 1:
            subdirs[parts[0]].append(doc_id)
        else:
            subdirs['(root)'].append(doc_id)

    print("\nCorpus structure:")
    for subdir, files in sorted(subdirs.items()):
        print(f"  {subdir}/: {len(files)} files")

    # =========================================================================
    # PHASE 2: BUILD CORTICAL INDEX
    # =========================================================================
    print_header("PHASE 2: BUILDING CORTICAL TEXT INDEX")

    processor = CorticalTextProcessor()
    print("\nIndexing documents...")

    for i, (doc_id, content) in enumerate(docs.items()):
        processor.process_document(doc_id, content)
        if (i + 1) % 20 == 0:
            print(f"  Indexed {i + 1}/{len(docs)} documents...")

    print("\nComputing TF-IDF, PageRank, and concept clusters...")
    processor.compute_all()

    print(f"\nCortical index statistics:")
    print(f"  Unique tokens:  {processor.get_layer(CorticalLayer.TOKENS).column_count():,}")
    print(f"  Bigrams:        {processor.get_layer(CorticalLayer.BIGRAMS).column_count():,}")
    print(f"  Concept clusters: {processor.get_layer(CorticalLayer.CONCEPTS).column_count()}")

    # =========================================================================
    # PHASE 3: BUILD SYNAPTIC KNOWLEDGE GRAPH
    # =========================================================================
    print_header("PHASE 3: BUILDING SYNAPTIC KNOWLEDGE GRAPH")

    # Custom plasticity rules for knowledge discovery
    rules = PlasticityRules(
        hebbian_rate=0.12,       # Moderate strengthening
        anti_hebbian_rate=0.03,  # Slow forgetting
        reward_rate=0.20,        # Strong reward learning
    )

    graph = SynapticMemoryGraph(plasticity_rules=rules)
    reasoner = IncrementalReasoner(graph, auto_link_similar=True, similarity_threshold=0.4)

    # Create category nodes
    categories = set()
    for doc_id, content in docs.items():
        cat = categorize_document(doc_id, content)
        categories.add(cat)

    print(f"\nDiscovered {len(categories)} knowledge domains:")
    for cat in sorted(categories):
        graph.add_node(f"DOMAIN:{cat}", NodeType.CONTEXT, cat.replace("_", " ").title())
        print(f"  - {cat}")

    # Create document nodes with category links
    print(f"\nCreating document nodes and domain links...")
    doc_nodes = {}

    for doc_id, content in docs.items():
        title = extract_title(content, doc_id)
        category = categorize_document(doc_id, content)

        # Add document as ARTIFACT node
        node = graph.add_node(f"DOC:{doc_id}", NodeType.ARTIFACT, title)
        doc_nodes[doc_id] = node

        # Link to category
        graph.add_synaptic_edge(
            f"DOMAIN:{category}",
            f"DOC:{doc_id}",
            EdgeType.CONTAINS,
            weight=0.8
        )

    # Extract and link key concepts
    print("Extracting key concepts from documents...")
    concept_nodes = {}
    concept_doc_links = defaultdict(list)

    for doc_id in docs:
        terms = extract_key_terms(processor, doc_id, top_n=4)

        for term in terms:
            term_key = term.lower()

            if term_key not in concept_nodes:
                concept_nodes[term_key] = graph.add_node(
                    f"CONCEPT:{term_key}",
                    NodeType.CONCEPT,
                    term
                )

            concept_doc_links[term_key].append(doc_id)

            # Link document to concept
            if (f"DOC:{doc_id}", f"CONCEPT:{term_key}", EdgeType.CONTAINS) not in graph.synaptic_edges:
                graph.add_synaptic_edge(
                    f"DOC:{doc_id}",
                    f"CONCEPT:{term_key}",
                    EdgeType.CONTAINS,
                    weight=0.6
                )

    # Find cross-document concept bridges
    print("Discovering concept bridges between documents...")
    bridges_created = 0

    for concept, doc_list in concept_doc_links.items():
        if len(doc_list) >= 2:
            # This concept appears in multiple documents - create bridges
            for i, doc1 in enumerate(doc_list[:3]):  # Limit to avoid explosion
                for doc2 in doc_list[i+1:4]:
                    key = (f"DOC:{doc1}", f"DOC:{doc2}", EdgeType.SIMILAR)
                    if key not in graph.synaptic_edges:
                        graph.add_synaptic_edge(
                            f"DOC:{doc1}",
                            f"DOC:{doc2}",
                            EdgeType.SIMILAR,
                            weight=0.5,
                            bidirectional=True
                        )
                        bridges_created += 1

    print(f"\nKnowledge graph statistics:")
    print(f"  Domain nodes:   {len(categories)}")
    print(f"  Document nodes: {len(doc_nodes)}")
    print(f"  Concept nodes:  {len(concept_nodes)}")
    print(f"  Total nodes:    {graph.node_count()}")
    print(f"  Total edges:    {len(graph.synaptic_edges)}")
    print(f"  Cross-doc bridges: {bridges_created}")

    # =========================================================================
    # PHASE 4: SIMULATE KNOWLEDGE EXPLORATION
    # =========================================================================
    print_header("PHASE 4: SIMULATING KNOWLEDGE EXPLORATION SESSIONS")

    # Define exploration scenarios
    scenarios = [
        {
            "name": "Understanding Memory and Learning",
            "questions": [
                "How does memory consolidation work?",
                "What is the relationship between sleep and memory?",
                "How do neural networks learn from experience?",
            ],
            "keywords": ["memory", "learning", "neural", "cognitive", "consolidation"],
        },
        {
            "name": "AI Agent Development",
            "questions": [
                "How should agents coordinate on complex tasks?",
                "What patterns work for multi-agent systems?",
                "How do we handle agent handoffs?",
            ],
            "keywords": ["agent", "coordination", "task", "handoff", "parallel"],
        },
        {
            "name": "Knowledge Retrieval Systems",
            "questions": [
                "How does semantic search work?",
                "What is TF-IDF and how is it used?",
                "How do we rank search results?",
            ],
            "keywords": ["search", "retrieval", "semantic", "ranking", "index"],
        },
    ]

    session_insights = []

    for scenario in scenarios:
        print_section(f"Session: {scenario['name']}")

        # Find relevant documents
        relevant_docs = []
        for doc_id, content in docs.items():
            content_lower = content.lower()
            if any(kw in content_lower for kw in scenario["keywords"]):
                relevant_docs.append(doc_id)

        print(f"  Found {len(relevant_docs)} relevant documents")

        if not relevant_docs:
            continue

        # Simulate exploration
        for question in scenario["questions"]:
            # Create question node
            q_node = reasoner.process_thought(question, NodeType.QUESTION)

            # Activate relevant documents (user "reading" them)
            activated = 0
            for doc_id in relevant_docs[:5]:
                doc_node_id = f"DOC:{doc_id}"
                if doc_node_id in graph.nodes:
                    graph.activate_node(doc_node_id, context={"question": question})

                    # Also activate connected concepts
                    for edge in graph.get_synaptic_edges_from(doc_node_id):
                        if edge.target_id.startswith("CONCEPT:"):
                            graph.activate_node(edge.target_id)

                    activated += 1

            # Record insight
            if activated > 0:
                session_insights.append({
                    "scenario": scenario["name"],
                    "question": question,
                    "docs_explored": activated,
                })

        # Apply Hebbian learning for this session
        strengthened = graph.apply_hebbian_learning(time_window_seconds=300)
        print(f"  Explored {len(scenario['questions'])} questions")
        print(f"  Strengthened {strengthened} connections through co-activation")

    # =========================================================================
    # PHASE 5: DISCOVER EMERGENT PATTERNS
    # =========================================================================
    print_header("PHASE 5: EMERGENT PATTERNS AND INSIGHTS")

    # Find strongest connections (most reinforced paths)
    print_section("Strongest Knowledge Connections")

    strong_edges = sorted(
        graph.synaptic_edges.values(),
        key=lambda e: e.weight,
        reverse=True
    )[:15]

    for edge in strong_edges:
        src_label = edge.source_id.split(":")[-1][:25]
        tgt_label = edge.target_id.split(":")[-1][:25]
        print(f"  {src_label:25} --[{edge.edge_type.value}]--> {tgt_label:25} (w={edge.weight:.2f})")

    # Find hub concepts (most connected)
    print_section("Hub Concepts (Most Connected)")

    concept_connections = defaultdict(int)
    for (src, tgt, _) in graph.synaptic_edges.keys():
        if src.startswith("CONCEPT:"):
            concept_connections[src] += 1
        if tgt.startswith("CONCEPT:"):
            concept_connections[tgt] += 1

    top_concepts = sorted(concept_connections.items(), key=lambda x: x[1], reverse=True)[:10]
    for concept_id, count in top_concepts:
        concept_name = concept_id.replace("CONCEPT:", "")
        print(f"  {concept_name:20} - {count} connections")

    # Find document clusters
    print_section("Document Clusters (via Shared Concepts)")

    doc_similarities = defaultdict(float)
    for (src, tgt, etype), edge in graph.synaptic_edges.items():
        if etype == EdgeType.SIMILAR and src.startswith("DOC:") and tgt.startswith("DOC:"):
            doc_similarities[(src, tgt)] = edge.weight

    top_similar = sorted(doc_similarities.items(), key=lambda x: x[1], reverse=True)[:8]
    for (doc1, doc2), weight in top_similar:
        d1 = doc1.replace("DOC:", "")[:30]
        d2 = doc2.replace("DOC:", "")[:30]
        print(f"  {d1} <-> {d2} (sim={weight:.2f})")

    # =========================================================================
    # PHASE 6: PREDICTIVE REASONING
    # =========================================================================
    print_header("PHASE 6: PREDICTIVE REASONING IN ACTION")

    print_section("Query: 'What should I read to understand neural networks?'")

    # Find neural-related documents and predict from them
    neural_docs = [d for d in docs if "neural" in d.lower() or "neural" in docs[d].lower()[:500]]

    if neural_docs:
        start_doc = neural_docs[0]
        start_node = f"DOC:{start_doc}"

        print(f"\n  Starting from: {start_doc}")
        print(f"  Predictions for what to explore next:")

        predictions = graph.predict_next_thoughts(start_node, top_n=5)
        for i, pred in enumerate(predictions, 1):
            node_type = pred.node.node_type.value
            label = pred.node_id.split(":")[-1][:40]
            print(f"    {i}. [{node_type}] {label} (prob={pred.probability:.3f})")

    print_section("Query: 'How do agents coordinate?'")

    agent_docs = [d for d in docs if "agent" in d.lower() or "coordination" in docs[d].lower()[:500]]

    if agent_docs:
        start_doc = agent_docs[0]
        start_node = f"DOC:{start_doc}"

        print(f"\n  Starting from: {start_doc}")
        print(f"  Predictions for related knowledge:")

        predictions = graph.predict_next_thoughts(start_node, top_n=5)
        for i, pred in enumerate(predictions, 1):
            node_type = pred.node.node_type.value
            label = pred.node_id.split(":")[-1][:40]
            print(f"    {i}. [{node_type}] {label} (prob={pred.probability:.3f})")

    # =========================================================================
    # PHASE 7: LEARNING DEMONSTRATION
    # =========================================================================
    print_header("PHASE 7: LEARNING FROM FEEDBACK")

    print_section("Before Feedback: Edge Weights")

    # Pick a path to reinforce
    sample_path = []
    for domain_id in graph.nodes:
        if domain_id.startswith("DOMAIN:cognitive"):
            domain_edges = graph.get_synaptic_edges_from(domain_id)
            if domain_edges:
                doc_edge = domain_edges[0]
                doc_id = doc_edge.target_id

                doc_edges = graph.get_synaptic_edges_from(doc_id)
                if doc_edges:
                    concept_edge = doc_edges[0]
                    sample_path = [domain_id, doc_id, concept_edge.target_id]
                    break

    if sample_path:
        print(f"\n  Path to reinforce:")
        for i, node_id in enumerate(sample_path):
            label = node_id.split(":")[-1][:40]
            print(f"    {i+1}. {label}")

        # Show weights before
        print(f"\n  Edge weights before feedback:")
        for i in range(len(sample_path) - 1):
            src, tgt = sample_path[i], sample_path[i+1]
            for (s, t, _), edge in graph.synaptic_edges.items():
                if s == src and t == tgt:
                    print(f"    {src.split(':')[-1][:20]} -> {tgt.split(':')[-1][:20]}: {edge.weight:.3f}")

        # Apply reward
        graph.apply_reward(sample_path, reward=0.5)

        print(f"\n  After applying positive reward (+0.5):")
        for i in range(len(sample_path) - 1):
            src, tgt = sample_path[i], sample_path[i+1]
            for (s, t, _), edge in graph.synaptic_edges.items():
                if s == src and t == tgt:
                    print(f"    {src.split(':')[-1][:20]} -> {tgt.split(':')[-1][:20]}: {edge.weight:.3f}")

    # =========================================================================
    # PHASE 8: DECAY SIMULATION
    # =========================================================================
    print_header("PHASE 8: KNOWLEDGE DECAY OVER TIME")

    # Apply decay to simulate time passing
    print("\nSimulating 30 days without usage...")

    initial_avg = sum(e.weight for e in graph.synaptic_edges.values()) / len(graph.synaptic_edges)

    for day in range(30):
        graph.apply_global_decay()

    final_avg = sum(e.weight for e in graph.synaptic_edges.values()) / len(graph.synaptic_edges)

    print(f"\n  Average edge weight before: {initial_avg:.3f}")
    print(f"  Average edge weight after:  {final_avg:.3f}")
    print(f"  Weight retained: {(final_avg/initial_avg)*100:.1f}%")

    print("\n  Insight: Frequently used paths resist decay through activation,")
    print("  while neglected knowledge gradually fades - just like human memory.")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("SUMMARY: WHAT PRISM-GoT DISCOVERED")

    print(f"""
From analyzing {len(docs)} documents across {len(categories)} knowledge domains,
PRISM-GoT built a synaptic knowledge graph with:

  • {graph.node_count()} nodes (domains, documents, concepts)
  • {len(graph.synaptic_edges)} synaptic edges with learned weights
  • {len(concept_nodes)} key concepts extracted via TF-IDF
  • {bridges_created} cross-document bridges discovered

Key insights demonstrated:

  1. KNOWLEDGE ORGANIZATION
     Documents automatically clustered by domain (cognitive science,
     AI development, knowledge management, etc.)

  2. CONCEPT EXTRACTION
     Key terms extracted from each document enable semantic navigation
     beyond simple keyword matching

  3. CROSS-DOCUMENT BRIDGES
     Shared concepts create bridges between related documents,
     enabling serendipitous discovery

  4. HEBBIAN LEARNING
     Exploration sessions strengthen connections between co-activated
     documents and concepts ("what fires together wires together")

  5. PREDICTIVE REASONING
     The system predicts what you might want to explore next based
     on learned patterns from the corpus

  6. REWARD LEARNING
     Positive feedback on useful paths makes them more prominent
     in future recommendations

  7. GRACEFUL FORGETTING
     Unused knowledge slowly fades, keeping the graph focused on
     what's actually valuable

This creates a "second brain" that learns from how you explore knowledge,
becoming increasingly helpful at predicting what you need.
    """)


if __name__ == "__main__":
    main()
