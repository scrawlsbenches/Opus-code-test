"""
World Model Cognitive Analysis
==============================

Analyzes the cognitive science, world models, and cross-domain corpus
to explore how concepts connect across prediction, influence, and cognition.

Inspired by showcase.py but focused on supporting cognitive workflows.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def render_bar(value: float, max_value: float, width: int = 30) -> str:
    """Render a text-based progress bar."""
    if max_value == 0:
        return " " * width
    filled = int((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


class WorldModelAnalyzer:
    """
    Analyzes cognitive and world model concepts to support understanding
    of prediction, influence, and decision-making frameworks.
    """

    # Domain directories to analyze
    COGNITIVE_DOMAINS = [
        "cognitive_science",
        "world_models",
        "future_thinking",
        "social_influence",
        "ai_market_prediction",
        "workflow_practices",
        "cross_domain",
    ]

    # Key concepts to track across domains
    KEY_CONCEPTS = {
        "prediction": ["prediction", "forecast", "anticipate", "expect", "future"],
        "learning": ["learning", "adaptation", "update", "acquire", "experience"],
        "models": ["model", "representation", "simulation", "schema", "mental"],
        "decision": ["decision", "choice", "select", "judgment", "evaluate"],
        "influence": ["influence", "persuade", "social", "conform", "nudge"],
        "uncertainty": ["uncertainty", "probability", "risk", "confidence", "error"],
        "attention": ["attention", "focus", "salience", "awareness", "cognitive"],
        "memory": ["memory", "recall", "encoding", "consolidation", "retrieval"],
    }

    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = samples_dir
        self.tokenizer = Tokenizer(filter_code_noise=True)
        self.processor = CorticalTextProcessor(tokenizer=self.tokenizer)
        self.loaded_files = []
        self.domain_docs = defaultdict(list)
        self.start_time = None

    def run(self, mode: str = "full"):
        """
        Run the analysis.

        Args:
            mode: "full" for complete analysis, "quick" for overview only
        """
        self.print_intro()

        if not self.ingest_cognitive_corpus():
            print("No cognitive documents found!")
            return

        self.analyze_domain_structure()
        self.discover_core_concepts()
        self.analyze_concept_bridges()
        self.explore_world_model_network()

        if mode == "full":
            self.analyze_prediction_chain()
            self.demonstrate_cognitive_queries()
            self.find_knowledge_gaps()
            self.suggest_connections()

        self.print_cognitive_summary()

    def print_intro(self):
        """Print introduction."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║          🧠  WORLD MODEL COGNITIVE ANALYSIS  🧠                      ║
    ║                                                                      ║
    ║   Exploring prediction, influence, and cognition through            ║
    ║   hierarchical text analysis of world model concepts                ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
        """)

    def ingest_cognitive_corpus(self) -> bool:
        """Ingest cognitive-focused corpus from disk."""
        print_header("CORPUS INGESTION", "═")

        self.start_time = time.perf_counter()

        print(f"Loading cognitive domains from: {self.samples_dir}")
        print(f"Target domains: {', '.join(self.COGNITIVE_DOMAINS)}\n")

        total_docs = 0

        for domain in self.COGNITIVE_DOMAINS:
            domain_path = os.path.join(self.samples_dir, domain)

            if not os.path.exists(domain_path):
                print(f"  ⚠  {domain}: directory not found")
                continue

            files = sorted([f for f in os.listdir(domain_path) if f.endswith('.txt')])

            if not files:
                continue

            print(f"  📁 {domain}: ", end="")

            for filename in files:
                filepath = os.path.join(domain_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                doc_id = f"{domain}/{filename.replace('.txt', '')}"
                self.processor.process_document(doc_id, content)
                self.loaded_files.append((doc_id, len(content.split())))
                self.domain_docs[domain].append(doc_id)
                total_docs += 1

            print(f"{len(files)} documents")

        if total_docs == 0:
            return False

        # Compute cortical representations
        print("\nComputing cortical representations...")
        compute_start = time.perf_counter()

        self.processor.compute_all(
            verbose=False,
            connection_strategy='hybrid',
            cluster_strictness=0.5,
            bridge_weight=0.3
        )

        compute_time = time.perf_counter() - compute_start

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print(f"\n✓ Processed {total_docs} cognitive documents")
        print(f"✓ Created {layer0.column_count()} token minicolumns")
        print(f"✓ Created {layer1.column_count()} bigram minicolumns")
        print(f"✓ Discovered {layer2.column_count()} concept clusters")
        print(f"✓ Formed {total_conns:,} connections")
        print(f"\n⏱  Compute time: {compute_time:.2f}s")

        return True

    def analyze_domain_structure(self):
        """Analyze the structure of each cognitive domain."""
        print_header("DOMAIN STRUCTURE", "═")

        print("Analyzing conceptual coverage across cognitive domains:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        for domain in self.COGNITIVE_DOMAINS:
            if domain not in self.domain_docs:
                continue

            docs = self.domain_docs[domain]

            # Count unique terms in this domain
            domain_terms = set()
            for doc_id in docs:
                col = self.processor.get_layer(CorticalLayer.DOCUMENTS).get_minicolumn(doc_id)
                if col:
                    for conn_id in col.feedforward_connections:
                        term_col = layer0.get_by_id(conn_id)
                        if term_col:
                            domain_terms.add(term_col.content)

            print(f"  📊 {domain}")
            print(f"     Documents: {len(docs)}")
            print(f"     Unique terms: {len(domain_terms)}")

            # Find top terms by PageRank in this domain
            domain_top = []
            for term in domain_terms:
                col = layer0.get_minicolumn(term)
                if col:
                    domain_top.append((term, col.pagerank))

            domain_top.sort(key=lambda x: -x[1])
            top_terms = [t[0] for t in domain_top[:5]]
            print(f"     Key concepts: {', '.join(top_terms)}")
            print()

    def discover_core_concepts(self):
        """Discover core concepts via PageRank analysis."""
        print_header("CORE CONCEPTS (PageRank)", "═")

        print("Central concepts in the cognitive corpus:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Get top tokens by PageRank
        top_tokens = sorted(
            layer0.minicolumns.values(),
            key=lambda c: c.pagerank,
            reverse=True
        )[:20]

        if top_tokens:
            max_pr = top_tokens[0].pagerank
            print("  Rank  Concept            PageRank         Domains")
            print("  " + "─" * 60)

            for i, col in enumerate(top_tokens, 1):
                bar = render_bar(col.pagerank, max_pr, 15)

                # Find which domains this term appears in
                domains = set()
                for doc_id in col.document_ids:
                    domain = doc_id.split('/')[0] if '/' in doc_id else 'other'
                    domains.add(domain[:8])

                domain_str = ', '.join(sorted(domains)[:3])
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.pagerank:.4f}  {domain_str}")

    def analyze_concept_bridges(self):
        """Analyze how concepts bridge across domains."""
        print_header("CROSS-DOMAIN BRIDGES", "═")

        print("Concepts that connect multiple cognitive domains:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Find terms that appear in multiple domains
        term_domains = defaultdict(set)

        for col in layer0.minicolumns.values():
            for doc_id in col.document_ids:
                domain = doc_id.split('/')[0] if '/' in doc_id else 'other'
                if domain in self.COGNITIVE_DOMAINS:
                    term_domains[col.content].add(domain)

        # Filter to bridge terms (3+ domains)
        bridges = [
            (term, domains)
            for term, domains in term_domains.items()
            if len(domains) >= 3
        ]

        # Sort by domain count, then PageRank
        bridges.sort(key=lambda x: (-len(x[1]), -layer0.get_minicolumn(x[0]).pagerank))

        print("  Bridge Concept     Domains Connected")
        print("  " + "─" * 55)

        for term, domains in bridges[:15]:
            col = layer0.get_minicolumn(term)
            pr = col.pagerank if col else 0
            domain_list = ', '.join(sorted(domains))
            print(f"  {term:<18} ({len(domains)}) {domain_list}")

    def explore_world_model_network(self):
        """Explore the network of world model concepts."""
        print_header("WORLD MODEL NETWORK", "═")

        print("Exploring connections in the world model conceptual space:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Key world model terms to explore
        world_model_terms = [
            "model", "prediction", "simulation", "representation",
            "learning", "update", "belief", "inference"
        ]

        for term in world_model_terms:
            col = layer0.get_minicolumn(term)
            if not col or not col.lateral_connections:
                continue

            print_subheader(f"🌐 '{term}' network:")

            # Get top connections
            sorted_conns = sorted(
                col.lateral_connections.items(),
                key=lambda x: x[1],
                reverse=True
            )[:8]

            for neighbor_id, weight in sorted_conns:
                neighbor = layer0.get_by_id(neighbor_id)
                if neighbor:
                    bar_len = int(min(weight, 10) * 2)
                    bar = "─" * bar_len + ">"

                    # Check which domains this neighbor appears in
                    domains = set()
                    for doc_id in neighbor.document_ids:
                        domain = doc_id.split('/')[0] if '/' in doc_id else 'other'
                        if domain in self.COGNITIVE_DOMAINS:
                            domains.add(domain[:6])

                    domain_hint = f"[{','.join(sorted(domains)[:2])}]" if domains else ""
                    print(f"    {bar} {neighbor.content} ({weight:.1f}) {domain_hint}")
            print()

    def analyze_prediction_chain(self):
        """Analyze the prediction → decision → action chain."""
        print_header("PREDICTION-DECISION CHAIN", "═")

        print("Tracing the cognitive chain from prediction to action:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        chain_concepts = [
            ("perception", "Sensing the environment"),
            ("attention", "Focusing cognitive resources"),
            ("prediction", "Anticipating future states"),
            ("uncertainty", "Quantifying confidence"),
            ("decision", "Selecting actions"),
            ("learning", "Updating from outcomes"),
        ]

        print("  Stage           PageRank   Connections  Description")
        print("  " + "─" * 65)

        for concept, description in chain_concepts:
            col = layer0.get_minicolumn(concept)
            if col:
                conn_count = len(col.lateral_connections)
                bar = render_bar(col.pagerank, 0.02, 10)
                print(f"  {concept:<15} {bar} {col.pagerank:.4f}  {conn_count:>3} conns   {description}")
            else:
                print(f"  {concept:<15} (not found in corpus)")

        # Show how these concepts connect
        print("\n  Cross-stage connections:")
        for i, (concept1, _) in enumerate(chain_concepts[:-1]):
            concept2 = chain_concepts[i + 1][0]
            col1 = layer0.get_minicolumn(concept1)
            col2 = layer0.get_minicolumn(concept2)

            if col1 and col2:
                # Check if directly connected
                col2_id = f"L0_{concept2}"
                weight = col1.lateral_connections.get(col2_id, 0)

                if weight > 0:
                    print(f"    {concept1} ──({weight:.1f})──> {concept2}")
                else:
                    # Find intermediate connections
                    shared = set(col1.lateral_connections.keys()) & set(col2.lateral_connections.keys())
                    if shared:
                        bridge = layer0.get_by_id(list(shared)[0])
                        if bridge:
                            print(f"    {concept1} ···[{bridge.content}]···> {concept2}")
                    else:
                        print(f"    {concept1} ···(indirect)···> {concept2}")

    def demonstrate_cognitive_queries(self):
        """Demonstrate queries relevant to cognitive work."""
        print_header("COGNITIVE QUERIES", "═")

        print("Queries designed for cognitive science and world model work:\n")

        queries = [
            ("How do world models update from prediction errors?", "world_models"),
            ("What influences decision-making under uncertainty?", "cognitive_science"),
            ("How does social proof affect group behavior?", "social_influence"),
            ("What methods predict market regime changes?", "ai_market_prediction"),
            ("How does metacognition improve forecasting?", "cross_domain"),
        ]

        for query, expected_domain in queries:
            print_subheader(f"🔍 '{query}'")

            start = time.perf_counter()

            # Expand query
            expanded = self.processor.expand_query(query, max_expansions=8)
            original = set(self.tokenizer.tokenize(query))
            new_terms = [t for t in expanded.keys() if t not in original]

            if new_terms:
                print(f"    Expanded: +{', '.join(new_terms[:5])}")

            # Find documents
            results = self.processor.find_documents_for_query(query, top_n=5)
            elapsed = time.perf_counter() - start

            print(f"\n    Top results:")
            for doc_id, score in results:
                domain = doc_id.split('/')[0] if '/' in doc_id else 'other'
                match = "✓" if domain == expected_domain else " "
                print(f"    {match} {doc_id} ({score:.3f})")

            print(f"    ⏱  {elapsed*1000:.1f}ms\n")

    def find_knowledge_gaps(self):
        """Find gaps in the cognitive corpus."""
        print_header("KNOWLEDGE GAPS", "═")

        print("Analyzing coverage and identifying gaps:\n")

        gaps = self.processor.analyze_knowledge_gaps()

        print(f"  Coverage Score:     {gaps['coverage_score']:.1%}")
        print(f"  Connectivity Score: {gaps['connectivity_score']:.4f}")

        # Check concept category coverage
        print("\n  Concept Category Coverage:")
        print("  " + "─" * 50)

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        for category, terms in self.KEY_CONCEPTS.items():
            found = sum(1 for t in terms if layer0.get_minicolumn(t) is not None)
            coverage = found / len(terms)
            bar = render_bar(coverage, 1.0, 15)
            status = "✓" if coverage >= 0.6 else "⚠"
            print(f"  {status} {category:<12} {bar} {coverage:.0%} ({found}/{len(terms)})")

        # Show weak topics
        if gaps.get('weak_topics'):
            print("\n  Weak topics needing more coverage:")
            for topic in gaps['weak_topics'][:5]:
                print(f"    • '{topic['term']}' - only {topic['doc_count']} doc(s)")

    def suggest_connections(self):
        """Suggest potential new connections between concepts."""
        print_header("SUGGESTED CONNECTIONS", "═")

        print("Potential bridges between currently disconnected concepts:\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Find high-PageRank terms that could connect
        important_terms = sorted(
            layer0.minicolumns.values(),
            key=lambda c: c.pagerank,
            reverse=True
        )[:50]

        suggestions = []

        for i, col1 in enumerate(important_terms):
            for col2 in important_terms[i+1:]:
                # Check if they're NOT directly connected but share neighbors
                col2_id = f"L0_{col2.content}"
                if col2_id not in col1.lateral_connections:
                    shared = set(col1.lateral_connections.keys()) & set(col2.lateral_connections.keys())
                    if len(shared) >= 2:
                        # They share neighbors but aren't connected
                        combined_pr = col1.pagerank + col2.pagerank
                        suggestions.append((col1.content, col2.content, len(shared), combined_pr))

        # Sort by potential value
        suggestions.sort(key=lambda x: (-x[2], -x[3]))

        print("  Concepts that share neighbors but aren't directly connected:")
        print("  " + "─" * 55)

        for term1, term2, shared_count, _ in suggestions[:10]:
            print(f"    {term1} <···{shared_count} shared···> {term2}")

        print("\n  💡 These pairs might benefit from explicit bridging documents")

    def print_cognitive_summary(self):
        """Print final cognitive analysis summary."""
        print_header("COGNITIVE ANALYSIS SUMMARY", "═")

        elapsed = time.perf_counter() - self.start_time if self.start_time else 0

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        total_docs = len(self.loaded_files)
        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print("📊 CORPUS METRICS\n")
        print(f"  Cognitive domains:    {len(self.domain_docs)}")
        print(f"  Total documents:      {total_docs}")
        print(f"  Unique concepts:      {layer0.column_count()}")
        print(f"  Concept clusters:     {layer2.column_count()}")
        print(f"  Total connections:    {total_conns:,}")

        # Find the most central cognitive concept
        top_token = max(layer0.minicolumns.values(), key=lambda c: c.pagerank)
        print(f"\n  Most central concept: '{top_token.content}'")

        # Domain with most connections
        domain_conns = {}
        for domain, docs in self.domain_docs.items():
            conn_count = 0
            for doc_id in docs:
                col = self.processor.get_layer(CorticalLayer.DOCUMENTS).get_minicolumn(doc_id)
                if col:
                    conn_count += col.connection_count()
            domain_conns[domain] = conn_count

        if domain_conns:
            top_domain = max(domain_conns.items(), key=lambda x: x[1])
            print(f"  Most connected domain: '{top_domain[0]}' ({top_domain[1]} connections)")

        print(f"\n⏱  Total analysis time: {elapsed:.2f}s")

        print("\n" + "═" * 70)
        print("World model cognitive analysis complete!")
        print("  ✓ Analyzed cognitive domain structure")
        print("  ✓ Discovered core concepts via PageRank")
        print("  ✓ Found cross-domain bridge concepts")
        print("  ✓ Mapped world model conceptual network")
        print("  ✓ Traced prediction-decision chains")
        print("  ✓ Identified knowledge gaps")
        print("  ✓ Suggested potential new connections")
        print("═" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="World Model Cognitive Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/world_model_analysis.py                    # Full analysis
  python scripts/world_model_analysis.py --quick            # Quick overview
  python scripts/world_model_analysis.py --samples ./data   # Custom directory
        """
    )

    parser.add_argument(
        "--samples", "-s",
        default="samples",
        help="Path to samples directory (default: samples)"
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick analysis (skip detailed queries and gaps)"
    )

    args = parser.parse_args()

    analyzer = WorldModelAnalyzer(samples_dir=args.samples)
    analyzer.run(mode="quick" if args.quick else "full")


if __name__ == "__main__":
    main()
