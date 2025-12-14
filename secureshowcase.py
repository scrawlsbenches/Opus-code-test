"""
Security-Focused Cortical Text Processor Showcase
==================================================

This showcase demonstrates the cortical text processor's capabilities
for security-related content: threat analysis, vulnerability patterns,
secure coding practices, and security knowledge retrieval.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

from cortical import CorticalTextProcessor, CorticalLayer


class Timer:
    """Simple timer for measuring operation durations."""

    def __init__(self):
        self.times: Dict[str, float] = {}
        self._start: float = 0

    def start(self, name: str):
        """Start timing an operation."""
        self._start = time.perf_counter()
        self._current = name

    def stop(self) -> float:
        """Stop timing and record the duration."""
        elapsed = time.perf_counter() - self._start
        self.times[self._current] = elapsed
        return elapsed

    def get(self, name: str) -> float:
        """Get recorded time for an operation."""
        return self.times.get(name, 0)


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


class SecurityShowcase:
    """Security-focused showcase of the cortical text processor."""

    # Security-related sample files to load
    SECURITY_SAMPLES = [
        "application_security_fundamentals",
        "supply_chain_security",
        "secure_deserialization",
        "secrets_management",
        "threat_modeling",
        "secure_code_review",
        "secure_development_lifecycle",
        "devsecops_practices",
        "api_design_security",
        "dependency_management_practices",
        "configuration_management",
        "static_analysis_tools",
        "input_validation_patterns",
        "authentication_patterns",
        "network_security_fundamentals",
        "incident_response_procedures",
        "penetration_testing_methodology",
        "security_compliance_frameworks",
    ]

    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = samples_dir
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer(filter_code_noise=True)
        self.processor = CorticalTextProcessor(tokenizer=tokenizer)
        self.loaded_files = []
        self.timer = Timer()

    def run(self):
        """Run the complete security demo."""
        self.print_intro()

        if not self.ingest_security_corpus():
            print("No security documents found!")
            return

        self.analyze_hierarchy()
        self.discover_security_concepts()
        self.analyze_threat_categories()
        self.find_security_associations()
        self.demonstrate_vulnerability_queries()
        self.demonstrate_threat_modeling()
        self.demonstrate_secure_coding_search()
        self.demonstrate_compliance_queries()
        self.analyze_security_coverage()
        self.print_security_insights()

    def print_intro(self):
        """Print introduction."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║          🔒  SECURITY-FOCUSED CORTICAL SHOWCASE  🔒                  ║
    ║                                                                      ║
    ║     Semantic search and analysis for security knowledge bases        ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
        """)

    def ingest_security_corpus(self) -> bool:
        """Ingest security-focused documents from the corpus."""
        print_header("SECURITY CORPUS INGESTION", "═")

        print(f"Loading security documents from: {self.samples_dir}")
        print("Building security knowledge graph...\n")

        if not os.path.exists(self.samples_dir):
            print(f"  ❌ Directory not found: {self.samples_dir}")
            return False

        # Load security-related sample files
        self.timer.start('document_loading')
        loaded_count = 0

        for sample_name in self.SECURITY_SAMPLES:
            filepath = os.path.join(self.samples_dir, f"{sample_name}.txt")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                self.processor.process_document(sample_name, content)
                word_count = len(content.split())
                self.loaded_files.append((sample_name, word_count))
                print(f"  🔐 {sample_name:40} ({word_count:3} words)")
                loaded_count += 1
            else:
                print(f"  ⚠️  {sample_name:40} (not found)")

        load_time = self.timer.stop()

        if loaded_count == 0:
            return False

        # Compute all analysis
        print("\nBuilding security concept network...")
        self.timer.start('compute_all')
        self.processor.compute_all(
            verbose=False,
            connection_strategy='hybrid',
            cluster_strictness=0.5,
            bridge_weight=0.3
        )
        compute_time = self.timer.stop()

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print(f"\n✓ Processed {loaded_count} security documents")
        print(f"✓ Created {layer0.column_count()} security term minicolumns")
        print(f"✓ Created {layer1.column_count()} concept pair minicolumns")
        print(f"✓ Formed {total_conns:,} semantic connections")
        print(f"\n⏱  Document loading: {load_time:.2f}s")
        print(f"⏱  Network analysis: {compute_time:.2f}s")

        return True

    def analyze_hierarchy(self):
        """Show the hierarchical structure of security knowledge."""
        print_header("SECURITY KNOWLEDGE HIERARCHY", "═")

        print("Security concepts organized in hierarchical layers:\n")

        layers = [
            (CorticalLayer.TOKENS, "Terms", "Security terms (authentication, injection, etc.)"),
            (CorticalLayer.BIGRAMS, "Patterns", "Term pairs (SQL injection, access control, etc.)"),
            (CorticalLayer.CONCEPTS, "Categories", "Security domains (OWASP, STRIDE, etc.)"),
            (CorticalLayer.DOCUMENTS, "Knowledge", "Complete security documents"),
        ]

        for layer_enum, name, desc in layers:
            layer = self.processor.get_layer(layer_enum)
            count = layer.column_count()
            conns = layer.total_connections()
            print(f"  Layer {layer_enum.value}: {name}")
            print(f"         {count:,} nodes, {conns:,} connections")
            print(f"         {desc}")
            print()

    def discover_security_concepts(self):
        """Show most important security concepts via PageRank."""
        print_header("KEY SECURITY CONCEPTS (PageRank)", "═")

        print("Central security concepts - highly connected in the knowledge graph:")
        print("(Terms that bridge multiple security domains)\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Get top tokens by pagerank
        top_tokens = sorted(layer0.minicolumns.values(),
                           key=lambda c: c.pagerank, reverse=True)[:20]

        if top_tokens:
            max_pr = top_tokens[0].pagerank
            print("  Rank  Security Concept    PageRank")
            print("  " + "─" * 50)

            for i, col in enumerate(top_tokens, 1):
                bar = render_bar(col.pagerank, max_pr, 20)
                print(f"  {i:>3}.  {col.content:<18} {bar} {col.pagerank:.4f}")

    def analyze_threat_categories(self):
        """Analyze threat categories using TF-IDF."""
        print_header("THREAT & VULNERABILITY ANALYSIS (TF-IDF)", "═")

        print("Distinctive security terms - specific to certain threat domains:")
        print("(High TF-IDF = important in specific security contexts)\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Security-relevant terms to highlight
        security_terms = [
            "injection", "authentication", "deserialization", "pickle",
            "vulnerability", "exploit", "malicious", "tampering",
            "encryption", "credential", "token", "sanitize",
            "whitelist", "validation", "hmac", "signature"
        ]

        found_terms = []
        for term in security_terms:
            col = layer0.get_minicolumn(term)
            if col:
                found_terms.append((term, col.tfidf, len(col.document_ids)))

        if found_terms:
            found_terms.sort(key=lambda x: x[1], reverse=True)
            max_tfidf = found_terms[0][1] if found_terms else 1

            print("  Term               TF-IDF              Documents")
            print("  " + "─" * 55)

            for term, tfidf, doc_count in found_terms[:15]:
                bar = render_bar(tfidf, max_tfidf, 20)
                print(f"  {term:<18} {bar} {tfidf:.4f}  ({doc_count} docs)")

    def find_security_associations(self):
        """Show associations between security concepts."""
        print_header("SECURITY CONCEPT ASSOCIATIONS", "═")

        print("Semantic connections between security concepts:")
        print("(Terms that co-occur in security contexts)\n")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        # Key security concepts to explore
        security_concepts = ["authentication", "injection", "encryption", "validation"]

        for concept in security_concepts:
            col = layer0.get_minicolumn(concept)
            if col and col.lateral_connections:
                print_subheader(f"🔗 '{concept}' associates with:")

                sorted_conns = sorted(col.lateral_connections.items(),
                                     key=lambda x: x[1], reverse=True)[:6]

                for neighbor_id, weight in sorted_conns:
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor:
                        bar_len = int(min(weight, 10) * 3)
                        bar = "─" * bar_len + ">"
                        print(f"    {bar} {neighbor.content} (weight: {weight:.2f})")
                print()

    def demonstrate_vulnerability_queries(self):
        """Demonstrate vulnerability and attack pattern queries."""
        print_header("VULNERABILITY SEARCH", "═")

        print("Finding information about specific vulnerabilities:\n")

        vulnerability_queries = [
            "SQL injection prevention",
            "pickle deserialization attack",
            "cross-site scripting XSS",
            "authentication bypass",
        ]

        total_query_time = 0

        for query in vulnerability_queries:
            print_subheader(f"🔍 Query: '{query}'")

            start = time.perf_counter()

            # Show query expansion
            expanded = self.processor.expand_query(query, max_expansions=6)
            original = set(self.processor.tokenizer.tokenize(query))
            new_terms = [t for t in expanded.keys() if t not in original]

            if new_terms:
                print(f"    Expanded with: {', '.join(new_terms[:5])}")

            # Find relevant documents
            results = self.processor.find_documents_for_query(query, top_n=3)
            elapsed = time.perf_counter() - start
            total_query_time += elapsed

            print(f"\n    Relevant security documents:")
            for doc_id, score in results:
                print(f"      🔐 {doc_id} (relevance: {score:.3f})")
            print(f"    ⏱  {elapsed*1000:.1f}ms")
            print()

        self.timer.times['vuln_queries'] = total_query_time

    def demonstrate_threat_modeling(self):
        """Demonstrate STRIDE threat modeling queries."""
        print_header("STRIDE THREAT MODELING", "═")

        print("Searching for STRIDE threat categories:\n")
        print("  S - Spoofing    | T - Tampering   | R - Repudiation")
        print("  I - Info Disc   | D - Denial Svc  | E - Elev Privilege\n")

        stride_queries = [
            ("Spoofing", "identity spoofing authentication bypass"),
            ("Tampering", "data tampering integrity modification"),
            ("Repudiation", "repudiation audit logging non-repudiation"),
            ("Information Disclosure", "information disclosure data leak exposure"),
            ("Denial of Service", "denial of service availability rate limiting"),
            ("Elevation of Privilege", "privilege escalation authorization bypass"),
        ]

        for threat_name, query in stride_queries:
            results = self.processor.find_documents_for_query(query, top_n=2)

            print(f"  [{threat_name[0]}] {threat_name}:")
            if results:
                for doc_id, score in results:
                    print(f"      → {doc_id} ({score:.3f})")
            else:
                print(f"      → (no specific coverage)")
            print()

    def demonstrate_secure_coding_search(self):
        """Demonstrate secure coding practice searches."""
        print_header("SECURE CODING PRACTICES", "═")

        print("Retrieving secure coding guidance:\n")

        coding_queries = [
            "input validation sanitization",
            "password hashing bcrypt",
            "secure session management",
            "parameterized queries prepared statements",
        ]

        for query in coding_queries:
            print_subheader(f"🛡️  '{query}'")

            # Get passages for detailed guidance
            passages = self.processor.find_passages_for_query(
                query,
                top_n=2,
                chunk_size=250,
                overlap=30
            )

            if passages:
                for i, (text, doc_id, start, end, score) in enumerate(passages, 1):
                    print(f"\n    [{i}] From: {doc_id} (score: {score:.3f})")
                    print("    " + "─" * 45)

                    # Show truncated passage
                    lines = text.strip().split('\n')[:3]
                    for line in lines:
                        if len(line) > 55:
                            line = line[:52] + "..."
                        print(f"      {line}")
            print()

    def demonstrate_compliance_queries(self):
        """Demonstrate security compliance and best practice queries."""
        print_header("SECURITY BEST PRACTICES", "═")

        print("Finding security best practices and standards:\n")

        # Check for OWASP-related concepts
        print_subheader("📋 OWASP Top 10 Coverage")

        owasp_categories = [
            ("A01 - Broken Access Control", "access control authorization"),
            ("A02 - Cryptographic Failures", "encryption cryptography key management"),
            ("A03 - Injection", "injection SQL command LDAP"),
            ("A04 - Insecure Design", "threat modeling secure design"),
            ("A05 - Security Misconfiguration", "configuration hardening default"),
            ("A06 - Vulnerable Components", "dependency vulnerability supply chain"),
            ("A07 - Auth Failures", "authentication session management"),
            ("A08 - Integrity Failures", "deserialization integrity verification"),
            ("A09 - Logging Failures", "logging monitoring audit"),
            ("A10 - SSRF", "server-side request forgery SSRF"),
        ]

        coverage_count = 0
        for category, query in owasp_categories:
            results = self.processor.find_documents_for_query(query, top_n=1)
            if results and results[0][1] > 0.1:
                coverage_count += 1
                status = "✅"
                doc = results[0][0][:25]
            else:
                status = "⚠️"
                doc = "(limited coverage)"
            print(f"    {status} {category}: {doc}")

        print(f"\n    Coverage: {coverage_count}/10 OWASP categories addressed")

    def analyze_security_coverage(self):
        """Analyze coverage of security topics."""
        print_header("SECURITY KNOWLEDGE GAPS", "═")

        print("Analyzing security topic coverage:\n")

        gaps = self.processor.analyze_knowledge_gaps()

        print(f"  Knowledge Coverage: {gaps['coverage_score']:.1%}")
        print(f"  Topic Connectivity: {gaps['connectivity_score']:.4f}")

        summary = gaps['summary']
        print(f"\n  Total documents: {summary['total_documents']}")
        print(f"  Well-connected: {summary['well_connected_count']}")
        print(f"  Isolated topics: {summary['isolated_count']}")

        if gaps['weak_topics']:
            print("\n  📍 Topics needing more coverage:")
            for topic in gaps['weak_topics'][:5]:
                print(f"    • '{topic['term']}' - only {topic['doc_count']} doc(s)")

        # Check for missing security topics
        print_subheader("\n🔍 Security Topic Audit")

        essential_topics = [
            "authentication", "authorization", "encryption", "injection",
            "validation", "secrets", "vulnerability", "threat",
            "audit", "compliance", "firewall", "penetration"
        ]

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        present = []
        missing = []

        for topic in essential_topics:
            if layer0.get_minicolumn(topic):
                present.append(topic)
            else:
                missing.append(topic)

        print(f"    Present ({len(present)}): {', '.join(present[:8])}...")
        if missing:
            print(f"    Consider adding: {', '.join(missing)}")

    def print_security_insights(self):
        """Print final security insights and summary."""
        print_header("SECURITY KNOWLEDGE SUMMARY", "═")

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        total_conns = sum(
            layer.total_connections()
            for layer in self.processor.layers.values()
        )

        print("📊 SECURITY CORPUS ANALYSIS\n")

        print(f"  Security documents:      {len(self.loaded_files)}")
        print(f"  Security terms:          {layer0.column_count()}")
        print(f"  Term combinations:       {layer1.column_count()}")
        print(f"  Semantic connections:    {total_conns:,}")

        # Find most central security term
        top_token = max(layer0.minicolumns.values(), key=lambda c: c.pagerank)
        print(f"\n  Most central concept: '{top_token.content}'")

        # Find most connected document
        if layer3.column_count() > 0:
            top_doc = max(layer3.minicolumns.values(), key=lambda c: c.connection_count())
            print(f"  Most connected topic: '{top_doc.content}'")

        # Performance summary
        print("\n⏱  PERFORMANCE SUMMARY\n")
        if 'document_loading' in self.timer.times:
            print(f"  Corpus loading:      {self.timer.get('document_loading'):.2f}s")
        if 'compute_all' in self.timer.times:
            print(f"  Network analysis:    {self.timer.get('compute_all'):.2f}s")
        if 'vuln_queries' in self.timer.times:
            avg_query = self.timer.get('vuln_queries') / 4 * 1000
            print(f"  Avg query time:      {avg_query:.1f}ms")

        print("\n" + "═" * 70)
        print("Security showcase complete! The system successfully:")
        print("  ✓ Built security knowledge hierarchy")
        print("  ✓ Identified key security concepts via PageRank")
        print("  ✓ Analyzed threat and vulnerability patterns")
        print("  ✓ Found security concept associations")
        print("  ✓ Searched vulnerability documentation")
        print("  ✓ Mapped STRIDE threat categories")
        print("  ✓ Retrieved secure coding guidance")
        print("  ✓ Assessed OWASP Top 10 coverage")
        print("  ✓ Identified security knowledge gaps")
        print("═" * 70)

        print("\n💡 USE CASES FOR SECURITY TEAMS:\n")
        print("  • Query security knowledge base for threat info")
        print("  • Find relevant secure coding practices")
        print("  • Identify gaps in security documentation")
        print("  • Build RAG systems for security Q&A")
        print("  • Cross-reference vulnerabilities with mitigations")
        print("  • Support threat modeling exercises")
        print()


if __name__ == "__main__":
    showcase = SecurityShowcase(samples_dir="samples")
    showcase.run()
