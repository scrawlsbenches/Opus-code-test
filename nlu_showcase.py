#!/usr/bin/env python3
"""
NLU Showcase - Natural Language Understanding for Cortical Text Processor
=========================================================================

This showcase demonstrates:
1. Current NLU capabilities (intent parsing, query expansion)
2. Enhanced query understanding (negation, scoping, explanations)
3. Interactive Q&A mode for knowledge base exploration
4. Data/sample requirements for improving the system
5. Trajectory toward cognitive-inspired search

Run: python nlu_showcase.py [--interactive]
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.query.intent import parse_intent_query, ParsedIntent, QUESTION_INTENTS, ACTION_VERBS
from cortical.code_concepts import get_related_terms


# =============================================================================
# ENHANCED NLU COMPONENTS
# =============================================================================

@dataclass
class EnhancedQuery:
    """Extended query representation with negation, scoping, and explanation."""
    original: str
    intent: ParsedIntent

    # Enhanced parsing
    negations: List[str] = field(default_factory=list)      # Terms to exclude
    scope: Optional[str] = None                              # Module/file scope
    temporal: Optional[str] = None                           # Time context (recent, old)

    # Explanation components
    why_matched: Dict[str, List[str]] = field(default_factory=dict)  # doc_id -> reasons
    expansion_chain: Dict[str, str] = field(default_factory=dict)    # expanded -> source

    def explain(self) -> str:
        """Generate human-readable explanation of query understanding."""
        lines = [f"Query: \"{self.original}\"", ""]

        if self.intent['intent'] != 'search':
            lines.append(f"  Intent: {self.intent['intent']} (from '{self.intent['question_word']}')")

        if self.intent['action']:
            lines.append(f"  Action: {self.intent['action']}")

        if self.intent['subject']:
            lines.append(f"  Subject: {self.intent['subject']}")

        if self.negations:
            lines.append(f"  Exclude: {', '.join(self.negations)}")

        if self.scope:
            lines.append(f"  Scope: {self.scope}")

        if self.expansion_chain:
            lines.append("  Expansions:")
            for expanded, source in list(self.expansion_chain.items())[:5]:
                lines.append(f"    {source} → {expanded}")

        return "\n".join(lines)


def parse_enhanced_query(query_text: str) -> EnhancedQuery:
    """
    Parse query with enhanced NLU: negation, scoping, temporal.

    Examples:
        "authentication not tests" -> negations=['tests']
        "config in core module" -> scope='core'
        "recent changes to auth" -> temporal='recent'
    """
    # Get base intent parsing
    base_intent = parse_intent_query(query_text)

    enhanced = EnhancedQuery(
        original=query_text,
        intent=base_intent
    )

    query_lower = query_text.lower()

    # Parse negations: "not X", "without X", "exclude X", "except X"
    negation_patterns = [
        r'\bnot\s+(\w+)',
        r'\bwithout\s+(\w+)',
        r'\bexclude\s+(\w+)',
        r'\bexcept\s+(\w+)',
        r'\bno\s+(\w+)',
    ]
    for pattern in negation_patterns:
        matches = re.findall(pattern, query_lower)
        enhanced.negations.extend(matches)

    # Parse scope: "in X", "within X", "from X module/file/directory"
    scope_patterns = [
        r'\bin\s+(\w+)\s*(?:module|file|dir|directory|folder)?',
        r'\bwithin\s+(\w+)',
        r'\bfrom\s+(\w+)\s*(?:module|file)?',
    ]
    for pattern in scope_patterns:
        match = re.search(pattern, query_lower)
        if match:
            enhanced.scope = match.group(1)
            break

    # Parse temporal: "recent", "old", "new", "latest"
    if any(word in query_lower for word in ['recent', 'latest', 'new', 'today', 'yesterday']):
        enhanced.temporal = 'recent'
    elif any(word in query_lower for word in ['old', 'legacy', 'deprecated', 'original']):
        enhanced.temporal = 'old'

    # Build expansion chain for explainability
    if base_intent['action']:
        related = get_related_terms(base_intent['action'], max_terms=3)
        for term in related:
            enhanced.expansion_chain[term] = base_intent['action']

    if base_intent['subject']:
        related = get_related_terms(base_intent['subject'], max_terms=3)
        for term in related:
            enhanced.expansion_chain[term] = base_intent['subject']

    return enhanced


# =============================================================================
# EXPLAINABLE SEARCH
# =============================================================================

@dataclass
class ExplainedResult:
    """Search result with explanation of why it matched."""
    doc_id: str
    score: float
    matched_terms: List[Tuple[str, float]]  # (term, contribution)
    expansion_matches: List[Tuple[str, str, float]]  # (original, expanded, contribution)
    pagerank_boost: float = 0.0
    scope_match: bool = True
    negation_penalty: float = 0.0

    def explain(self) -> str:
        """Human-readable match explanation."""
        lines = [f"{self.doc_id} (score: {self.score:.3f})"]

        for term, contrib in self.matched_terms[:3]:
            lines.append(f"  ├─ '{term}' matched directly (weight: {contrib:.2f})")

        for orig, exp, contrib in self.expansion_matches[:2]:
            lines.append(f"  ├─ '{exp}' expanded from '{orig}' (weight: {contrib:.2f})")

        if self.pagerank_boost > 0.01:
            lines.append(f"  ├─ PageRank boost: +{self.pagerank_boost:.2f}")

        if self.negation_penalty > 0:
            lines.append(f"  └─ Negation penalty: -{self.negation_penalty:.2f}")

        return "\n".join(lines)


def search_with_explanation(
    query: EnhancedQuery,
    processor: CorticalTextProcessor,
    top_n: int = 5
) -> List[ExplainedResult]:
    """
    Search with full explanation of why results matched.
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]
    doc_results: Dict[str, ExplainedResult] = {}

    # Score from direct term matches
    for term in query.intent['expanded_terms']:
        col = layer0.get_minicolumn(term)
        if not col:
            continue

        for doc_id in col.document_ids:
            if doc_id not in doc_results:
                doc_results[doc_id] = ExplainedResult(
                    doc_id=doc_id,
                    score=0.0,
                    matched_terms=[],
                    expansion_matches=[]
                )

            tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
            result = doc_results[doc_id]

            # Check if this is an expansion or direct match
            if term in query.expansion_chain:
                source = query.expansion_chain[term]
                result.expansion_matches.append((source, term, tfidf * 0.6))
                result.score += tfidf * 0.6
            else:
                result.matched_terms.append((term, tfidf))
                result.score += tfidf

    # Apply PageRank boost
    for doc_id, result in doc_results.items():
        doc_col = processor.layers[CorticalLayer.DOCUMENTS].get_by_id(f"L3_{doc_id}")
        if doc_col:
            result.pagerank_boost = doc_col.pagerank * 0.5
            result.score += result.pagerank_boost

    # Apply negation penalties
    for negation in query.negations:
        neg_col = layer0.get_minicolumn(negation)
        if neg_col:
            for doc_id in neg_col.document_ids:
                if doc_id in doc_results:
                    doc_results[doc_id].negation_penalty = 0.5
                    doc_results[doc_id].score *= 0.5

    # Apply scope filtering
    if query.scope:
        for doc_id, result in list(doc_results.items()):
            if query.scope.lower() not in doc_id.lower():
                result.scope_match = False
                result.score *= 0.3

    # Sort and return top results
    sorted_results = sorted(doc_results.values(), key=lambda r: r.score, reverse=True)
    return sorted_results[:top_n]


# =============================================================================
# KNOWLEDGE BASE ANALYSIS
# =============================================================================

@dataclass
class KnowledgeGap:
    """Identified gap in the knowledge base."""
    category: str
    description: str
    suggested_samples: List[str]
    priority: str  # 'high', 'medium', 'low'


def analyze_knowledge_gaps(processor: CorticalTextProcessor) -> List[KnowledgeGap]:
    """
    Analyze the corpus to identify knowledge gaps and suggest improvements.
    """
    gaps = []

    layer0 = processor.layers[CorticalLayer.TOKENS]
    layer3 = processor.layers[CorticalLayer.DOCUMENTS]

    # Get corpus statistics
    total_docs = layer3.column_count()
    total_terms = layer0.column_count()

    # Check for domain coverage
    domains = defaultdict(int)
    for doc_id in [col.content for col in layer3.minicolumns.values()]:
        # Infer domain from path/name
        doc_lower = doc_id.lower()
        if 'test' in doc_lower:
            domains['testing'] += 1
        elif any(x in doc_lower for x in ['auth', 'login', 'security']):
            domains['security'] += 1
        elif any(x in doc_lower for x in ['api', 'rest', 'http']):
            domains['api'] += 1
        elif any(x in doc_lower for x in ['db', 'sql', 'query', 'model']):
            domains['database'] += 1
        elif any(x in doc_lower for x in ['ui', 'view', 'template', 'component']):
            domains['frontend'] += 1
        else:
            domains['general'] += 1

    # Identify underrepresented domains
    if total_docs > 10:
        avg_domain_size = total_docs / max(len(domains), 1)
        for domain, count in domains.items():
            if count < avg_domain_size * 0.3:
                gaps.append(KnowledgeGap(
                    category='domain_coverage',
                    description=f"Underrepresented domain: {domain} ({count} docs)",
                    suggested_samples=[
                        f"Add more {domain}-related documentation",
                        f"Include {domain} code examples",
                        f"Add {domain} best practices guides"
                    ],
                    priority='medium'
                ))

    # Check for semantic coverage
    # Look for terms with high TF but no lateral connections (isolated concepts)
    isolated_terms = []
    for col in layer0.minicolumns.values():
        if col.tfidf > 0.5 and len(col.lateral_connections) < 2:
            isolated_terms.append(col.content)

    if isolated_terms:
        gaps.append(KnowledgeGap(
            category='semantic_coverage',
            description=f"Isolated high-value terms: {', '.join(isolated_terms[:5])}",
            suggested_samples=[
                "Add documents that connect these concepts",
                "Include comparative analysis documents",
                "Add tutorial documents that span multiple concepts"
            ],
            priority='high'
        ))

    # Check for query coverage
    common_query_types = ['how', 'where', 'what', 'why']
    coverage_suggestions = {
        'how': "Add implementation guides and tutorials",
        'where': "Add architecture documentation with file references",
        'what': "Add glossary and definition documents",
        'why': "Add design decision documents (ADRs)"
    }

    for query_type in common_query_types:
        # Check if corpus can answer these query types
        gaps.append(KnowledgeGap(
            category='query_coverage',
            description=f"Ensure corpus can answer '{query_type}' questions",
            suggested_samples=[coverage_suggestions[query_type]],
            priority='low'
        ))

    return gaps


# =============================================================================
# MICRO-MODEL REQUIREMENTS
# =============================================================================

@dataclass
class MicroModelRequirements:
    """Requirements for training domain-specific micro-models."""
    model_type: str
    min_samples: int
    current_samples: int
    data_format: str
    training_approach: str
    feasibility: str  # 'ready', 'needs_data', 'not_feasible'


def assess_micromodel_feasibility(processor: CorticalTextProcessor) -> List[MicroModelRequirements]:
    """
    Assess what micro-models can be trained on current corpus.

    Note: True neural language models require ML libraries (transformers, torch).
    These assessments are for pattern-based statistical models that fit our
    zero-dependency philosophy.
    """
    layer0 = processor.layers[CorticalLayer.TOKENS]
    layer3 = processor.layers[CorticalLayer.DOCUMENTS]

    total_docs = layer3.column_count()
    total_terms = layer0.column_count()

    # Estimate total tokens (rough)
    total_tokens = sum(col.activation for col in layer0.minicolumns.values())

    models = []

    # 1. File Prediction Model (already exists in this project!)
    models.append(MicroModelRequirements(
        model_type="File Prediction",
        min_samples=500,  # commits
        current_samples=total_docs,  # Using doc count as proxy
        data_format="commit_message -> [file_paths]",
        training_approach="Co-occurrence matrix + TF-IDF keyword mapping",
        feasibility='ready' if total_docs >= 100 else 'needs_data'
    ))

    # 2. Query Expansion Model
    models.append(MicroModelRequirements(
        model_type="Query Expansion",
        min_samples=1000,  # term pairs
        current_samples=sum(len(col.lateral_connections) for col in layer0.minicolumns.values()),
        data_format="term -> [related_terms, weights]",
        training_approach="Lateral connection weights from co-occurrence",
        feasibility='ready'
    ))

    # 3. Intent Classification
    models.append(MicroModelRequirements(
        model_type="Intent Classification",
        min_samples=500,  # labeled queries
        current_samples=0,  # Would need labeled data
        data_format="query_text -> intent_label",
        training_approach="Pattern matching + keyword features (no ML needed)",
        feasibility='ready'  # Pattern-based, doesn't need training data
    ))

    # 4. Document Clustering
    models.append(MicroModelRequirements(
        model_type="Document Clustering",
        min_samples=50,
        current_samples=total_docs,
        data_format="document -> cluster_id",
        training_approach="Louvain community detection on term co-occurrence graph",
        feasibility='ready' if total_docs >= 50 else 'needs_data'
    ))

    # 5. True Language Model (HONEST ASSESSMENT)
    models.append(MicroModelRequirements(
        model_type="Neural Language Model",
        min_samples=100000,  # tokens minimum for tiny model
        current_samples=int(total_tokens),
        data_format="token sequences",
        training_approach="Requires: torch/tensorflow, GPU, weeks of training",
        feasibility='not_feasible'  # Violates zero-dependency philosophy
    ))

    return models


# =============================================================================
# COGNITIVE SCIENCE INTEGRATION
# =============================================================================

def print_cognitive_framework():
    """
    Print the cognitive science framework underlying the system.
    """
    framework = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COGNITIVE SCIENCE FRAMEWORK                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  The Cortical Text Processor uses metaphors from cognitive neuroscience:     ║
║                                                                              ║
║  VISUAL CORTEX ANALOGY                                                       ║
║  ════════════════════                                                        ║
║    Layer 0 (TOKENS)    ≈ V1 (edges)      - Individual features               ║
║    Layer 1 (BIGRAMS)   ≈ V2 (patterns)   - Local combinations                ║
║    Layer 2 (CONCEPTS)  ≈ V4 (shapes)     - Abstract groupings                ║
║    Layer 3 (DOCUMENTS) ≈ IT (objects)    - Complete entities                 ║
║                                                                              ║
║  HEBBIAN LEARNING                                                            ║
║  ════════════════                                                            ║
║    "Neurons that fire together wire together"                                ║
║    → Terms co-occurring in documents form lateral connections                ║
║    → Connection strength increases with co-occurrence frequency              ║
║                                                                              ║
║  SPREADING ACTIVATION                                                        ║
║  ═══════════════════                                                         ║
║    Query terms activate related concepts through connection weights          ║
║    → Similar to semantic priming in human memory                             ║
║    → PageRank = steady-state activation after infinite spreading             ║
║                                                                              ║
║  HONEST LIMITATIONS                                                          ║
║  ═════════════════                                                           ║
║    These are METAPHORS, not implementations of actual neural processes.      ║
║    We use standard IR algorithms (TF-IDF, BM25, PageRank, Louvain).          ║
║    The cognitive framing helps intuition but isn't scientifically rigorous.  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(framework)


def print_analytical_framework():
    """
    Print the analytical science framework.
    """
    framework = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ANALYTICAL SCIENCE FRAMEWORK                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INFORMATION RETRIEVAL FOUNDATIONS                                           ║
║  ═════════════════════════════════                                           ║
║    • TF-IDF: Term importance = frequency × inverse document frequency        ║
║    • BM25: Probabilistic relevance with saturation and length normalization  ║
║    • PageRank: Random walk probability as importance measure                 ║
║                                                                              ║
║  GRAPH ANALYSIS                                                              ║
║  ══════════════                                                              ║
║    • Louvain: Community detection for concept clustering                     ║
║    • Co-occurrence graphs: Term relationships from positional proximity      ║
║    • Spectral methods: Eigenvalue decomposition for embeddings               ║
║                                                                              ║
║  STATISTICAL NLP (No ML)                                                     ║
║  ═══════════════════════                                                     ║
║    • N-gram models: Bigram frequency and transition probabilities            ║
║    • Pattern extraction: Regex-based relation mining                         ║
║    • Edit distance: Fuzzy matching for typo tolerance                        ║
║                                                                              ║
║  WHAT WE CAN'T DO (Without ML Dependencies)                                  ║
║  ══════════════════════════════════════════                                  ║
║    ✗ True semantic similarity (needs embeddings from neural models)          ║
║    ✗ Question answering (needs language model inference)                     ║
║    ✗ Summarization (needs generative model)                                  ║
║    ✗ Named entity recognition (needs trained classifier)                     ║
║                                                                              ║
║  WHAT WE CAN DO WELL                                                         ║
║  ═══════════════════                                                         ║
║    ✓ Keyword and phrase search with intelligent expansion                    ║
║    ✓ Document similarity via term overlap and graph distance                 ║
║    ✓ Concept clustering via community detection                              ║
║    ✓ Intent classification via pattern matching                              ║
║    ✓ Query reformulation and suggestion                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(framework)


# =============================================================================
# SYSTEM TRAJECTORY
# =============================================================================

def print_trajectory():
    """
    Print the system trajectory and goals.
    """
    trajectory = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SYSTEM TRAJECTORY                                     ║
║                    "Plus Ultra and Beyond"                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CURRENT STATE (Where We Are)                                                ║
║  ════════════════════════════                                                ║
║    ✓ Hierarchical text processing (4 layers)                                 ║
║    ✓ Basic intent parsing (what/where/how/why)                               ║
║    ✓ Query expansion via co-occurrence                                       ║
║    ✓ Code-aware tokenization and concepts                                    ║
║    ✓ File prediction ML model (pattern-based)                                ║
║    ⚠ Search quality issues (stop words, over-expansion)                      ║
║                                                                              ║
║  NEAR-TERM GOALS (Next 2-4 Weeks)                                            ║
║  ════════════════════════════════                                            ║
║    1. Fix search quality fundamentals                                        ║
║       - Enable code stop word filtering                                      ║
║       - Apply test file penalties                                            ║
║       - Weight expansions by TF-IDF, not raw count                           ║
║                                                                              ║
║    2. Enhanced query understanding                                           ║
║       - Negation: "find X not in tests"                                      ║
║       - Scoping: "config in core module"                                     ║
║       - Temporal: "recent changes to auth"                                   ║
║                                                                              ║
║    3. Explainable results                                                    ║
║       - Show WHY each result matched                                         ║
║       - Display expansion chains                                             ║
║       - Highlight matching passages                                          ║
║                                                                              ║
║  MEDIUM-TERM VISION (1-3 Months)                                             ║
║  ═══════════════════════════════                                             ║
║    4. Mixture of Experts architecture                                        ║
║       - Lexical Expert (BM25 exact match)                                    ║
║       - Semantic Expert (graph-based similarity)                             ║
║       - Structural Expert (code patterns)                                    ║
║       - Temporal Expert (recency awareness)                                  ║
║       - Episodic Expert (session learning)                                   ║
║                                                                              ║
║    5. Session-aware intelligence                                             ║
║       - Track what user viewed/modified                                      ║
║       - Boost contextually relevant results                                  ║
║       - Learn from query refinements                                         ║
║                                                                              ║
║  LONG-TERM ASPIRATIONS                                                       ║
║  ═════════════════════                                                       ║
║    6. Self-improving knowledge base                                          ║
║       - Auto-detect knowledge gaps                                           ║
║       - Suggest documentation to write                                       ║
║       - Learn from user feedback                                             ║
║                                                                              ║
║    7. Reasoning support                                                      ║
║       - "Why is X related to Y?" explanations                                ║
║       - "What's the path from A to B?" traversals                            ║
║       - Analogy completion: "A is to B as C is to ?"                         ║
║                                                                              ║
║  PRINCIPLES (How We Get There)                                               ║
║  ═════════════════════════════                                               ║
║    • Zero dependencies: Build what we can, don't import complexity           ║
║    • Measure before optimizing: Profile, don't guess                         ║
║    • Dog-food everything: Use the system to develop the system               ║
║    • Honest assessment: Know our limits, work within them                    ║
║    • Incremental progress: Each commit should leave things better            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(trajectory)


# =============================================================================
# INTERACTIVE Q&A MODE
# =============================================================================

class InteractiveNLU:
    """Interactive Q&A mode for exploring NLU capabilities."""

    def __init__(self, processor: CorticalTextProcessor):
        self.processor = processor
        self.history: List[Tuple[str, List[ExplainedResult]]] = []

    def run(self):
        """Run interactive Q&A loop."""
        print("\n" + "="*70)
        print("INTERACTIVE NLU MODE")
        print("="*70)
        print("""
Commands:
  <query>           Search with NLU understanding
  /explain          Re-explain last results in detail
  /gaps             Analyze knowledge base gaps
  /models           Show micro-model feasibility
  /cognitive        Show cognitive science framework
  /analytical       Show analytical science framework
  /trajectory       Show system trajectory and goals
  /samples          Show what samples would improve the system
  /help             Show this help
  /quit             Exit interactive mode
        """)

        while True:
            try:
                query = input("\n🔍 Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not query:
                continue

            if query.startswith('/'):
                self._handle_command(query)
            else:
                self._handle_query(query)

    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        cmd_lower = cmd.lower()

        if cmd_lower in ['/quit', '/exit', '/q']:
            raise KeyboardInterrupt

        elif cmd_lower == '/help':
            print(__doc__)

        elif cmd_lower == '/explain':
            if self.history:
                query, results = self.history[-1]
                print(f"\nDetailed explanation for: {query}\n")
                for r in results:
                    print(r.explain())
                    print()
            else:
                print("No previous query to explain.")

        elif cmd_lower == '/gaps':
            gaps = analyze_knowledge_gaps(self.processor)
            print("\n📊 KNOWLEDGE BASE GAPS\n")
            for gap in gaps:
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[gap.priority]
                print(f"{priority_emoji} [{gap.category}] {gap.description}")
                for suggestion in gap.suggested_samples:
                    print(f"   → {suggestion}")
                print()

        elif cmd_lower == '/models':
            models = assess_micromodel_feasibility(self.processor)
            print("\n🤖 MICRO-MODEL FEASIBILITY\n")
            for m in models:
                status = {'ready': '✅', 'needs_data': '⚠️', 'not_feasible': '❌'}[m.feasibility]
                print(f"{status} {m.model_type}")
                print(f"   Samples: {m.current_samples:,} / {m.min_samples:,} needed")
                print(f"   Format: {m.data_format}")
                print(f"   Approach: {m.training_approach}")
                print()

        elif cmd_lower == '/cognitive':
            print_cognitive_framework()

        elif cmd_lower == '/analytical':
            print_analytical_framework()

        elif cmd_lower == '/trajectory':
            print_trajectory()

        elif cmd_lower == '/samples':
            self._suggest_samples()

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

    def _handle_query(self, query_text: str):
        """Handle a search query with NLU."""
        # Parse with enhanced NLU
        enhanced = parse_enhanced_query(query_text)

        # Show query understanding
        print("\n📝 QUERY UNDERSTANDING")
        print(enhanced.explain())

        # Search with explanation
        results = search_with_explanation(enhanced, self.processor, top_n=5)

        # Store in history
        self.history.append((query_text, results))

        # Display results
        if results:
            print("\n📄 RESULTS (with explanations)")
            print("-" * 50)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r.explain()}")
        else:
            print("\n❌ No results found.")
            print("Try:")
            print("  - Broader terms")
            print("  - Checking /gaps for knowledge base coverage")

    def _suggest_samples(self):
        """Suggest what samples would improve the system."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SAMPLE DATA RECOMMENDATIONS                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TO IMPROVE SEARCH QUALITY                                                   ║
║  ═════════════════════════                                                   ║
║    • More diverse documents covering different domains                        ║
║    • Documents that CONNECT concepts (tutorials, comparisons)                 ║
║    • Glossary/definition documents for "what is X?" queries                   ║
║    • Architecture docs with file references for "where is X?" queries         ║
║                                                                              ║
║  TO TRAIN FILE PREDICTION                                                    ║
║  ════════════════════════                                                    ║
║    • More commits with descriptive messages                                   ║
║    • Commits that follow patterns: feat:, fix:, refactor:, docs:              ║
║    • Multi-file commits showing which files change together                   ║
║    • Target: 500+ commits for reliable prediction                             ║
║                                                                              ║
║  TO IMPROVE INTENT UNDERSTANDING                                             ║
║  ═══════════════════════════════                                             ║
║    • Example queries with known correct answers (for evaluation)              ║
║    • Domain-specific vocabulary lists                                         ║
║    • Synonym groups for your specific codebase                                ║
║                                                                              ║
║  TO ENABLE SESSION LEARNING                                                  ║
║  ══════════════════════════                                                  ║
║    • Query logs with click-through data                                       ║
║    • User feedback on result relevance                                        ║
║    • Session transcripts showing query refinement patterns                    ║
║                                                                              ║
║  WHAT WE CANNOT TRAIN (Without ML Libraries)                                 ║
║  ═══════════════════════════════════════════                                 ║
║    ✗ Neural language models (need torch/tensorflow)                          ║
║    ✗ Transformer-based embeddings (need sentence-transformers)               ║
║    ✗ Fine-tuned classifiers (need scikit-learn minimum)                      ║
║                                                                              ║
║  HONEST ASSESSMENT                                                           ║
║  ════════════════                                                            ║
║    A true "small language model" requires:                                   ║
║    • 100K+ tokens of training data                                            ║
║    • ML framework (PyTorch, TensorFlow)                                       ║
║    • GPU training (hours to days)                                             ║
║    • This violates our zero-dependency philosophy                             ║
║                                                                              ║
║    What we CAN build:                                                        ║
║    • Statistical models (TF-IDF, co-occurrence)                               ║
║    • Pattern-based classifiers (regex, keyword matching)                      ║
║    • Graph-based similarity (PageRank, community detection)                   ║
║    • These are "micro-models" not "language models"                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# MAIN SHOWCASE
# =============================================================================

class NLUShowcase:
    """Main showcase demonstrating NLU capabilities."""

    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = samples_dir
        self.processor = CorticalTextProcessor()

    def run(self, interactive: bool = False):
        """Run the NLU showcase."""
        self._print_intro()

        if not self._load_corpus():
            print("No corpus loaded. Run with sample documents.")
            return

        self._demonstrate_intent_parsing()
        self._demonstrate_enhanced_queries()
        self._demonstrate_explainable_search()
        self._analyze_corpus_readiness()

        print_cognitive_framework()
        print_analytical_framework()
        print_trajectory()

        if interactive:
            nlu = InteractiveNLU(self.processor)
            nlu.run()

    def _print_intro(self):
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            🧠  NLU SHOWCASE - Natural Language Understanding  🧠             ║
║                                                                              ║
║     Demonstrating query understanding, explanation, and system trajectory    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

    def _load_corpus(self) -> bool:
        """Load corpus from samples or existing index."""
        # Try to load existing index first
        if os.path.exists("corpus_dev.json") or os.path.exists("corpus_dev"):
            print("Loading existing corpus index...")
            try:
                self.processor = CorticalTextProcessor.load("corpus_dev.json")
                print(f"  Loaded {self.processor.layers[CorticalLayer.DOCUMENTS].column_count()} documents")
                return True
            except Exception:
                pass

        if os.path.exists("corpus_dev.pkl"):
            print("Loading existing corpus index (pickle)...")
            try:
                self.processor = CorticalTextProcessor.load("corpus_dev.pkl")
                print(f"  Loaded {self.processor.layers[CorticalLayer.DOCUMENTS].column_count()} documents")
                return True
            except Exception:
                pass

        # Load from samples
        if not os.path.exists(self.samples_dir):
            return False

        print(f"Loading documents from {self.samples_dir}...")
        count = 0
        for root, dirs, files in os.walk(self.samples_dir):
            for f in files:
                if f.endswith(('.txt', '.md', '.py')):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', encoding='utf-8') as fp:
                            content = fp.read()
                        doc_id = os.path.relpath(path, self.samples_dir)
                        self.processor.process_document(doc_id, content)
                        count += 1
                    except Exception:
                        pass

        if count > 0:
            print(f"  Loaded {count} documents")
            print("  Computing indices...")
            self.processor.compute_all()
            return True

        return False

    def _demonstrate_intent_parsing(self):
        """Demonstrate intent parsing capabilities."""
        print("\n" + "="*70)
        print("INTENT PARSING DEMONSTRATION")
        print("="*70)

        test_queries = [
            "where do we handle authentication?",
            "how does the tokenizer work?",
            "what is PageRank?",
            "why do we use TF-IDF?",
            "find all validation functions",
        ]

        for query in test_queries:
            parsed = parse_intent_query(query)
            print(f"\nQuery: \"{query}\"")
            print(f"  Intent: {parsed['intent']}")
            print(f"  Action: {parsed['action']}")
            print(f"  Subject: {parsed['subject']}")
            print(f"  Expanded: {parsed['expanded_terms'][:5]}")

    def _demonstrate_enhanced_queries(self):
        """Demonstrate enhanced query parsing."""
        print("\n" + "="*70)
        print("ENHANCED QUERY UNDERSTANDING")
        print("="*70)

        test_queries = [
            "authentication not tests",
            "config in core module",
            "recent changes to processor",
            "find validation without test files",
        ]

        for query in test_queries:
            enhanced = parse_enhanced_query(query)
            print(f"\n{enhanced.explain()}")

    def _demonstrate_explainable_search(self):
        """Demonstrate search with explanations."""
        print("\n" + "="*70)
        print("EXPLAINABLE SEARCH")
        print("="*70)

        test_query = "how does query expansion work?"
        enhanced = parse_enhanced_query(test_query)

        print(f"\nSearching: \"{test_query}\"")
        print(enhanced.explain())

        results = search_with_explanation(enhanced, self.processor, top_n=3)

        print("\nResults with explanations:")
        print("-" * 50)
        for r in results:
            print(f"\n{r.explain()}")

    def _analyze_corpus_readiness(self):
        """Analyze corpus readiness for various features."""
        print("\n" + "="*70)
        print("CORPUS READINESS ANALYSIS")
        print("="*70)

        # Knowledge gaps
        print("\n📊 Knowledge Gaps:")
        gaps = analyze_knowledge_gaps(self.processor)
        for gap in gaps[:5]:
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[gap.priority]
            print(f"  {priority_emoji} {gap.description}")

        # Micro-model feasibility
        print("\n🤖 Micro-Model Readiness:")
        models = assess_micromodel_feasibility(self.processor)
        for m in models:
            status = {'ready': '✅', 'needs_data': '⚠️', 'not_feasible': '❌'}[m.feasibility]
            print(f"  {status} {m.model_type}: {m.current_samples:,}/{m.min_samples:,}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NLU Showcase")
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive Q&A mode')
    parser.add_argument('--samples', '-s', default='samples',
                       help='Samples directory')
    args = parser.parse_args()

    showcase = NLUShowcase(samples_dir=args.samples)
    showcase.run(interactive=args.interactive)


if __name__ == '__main__':
    main()
