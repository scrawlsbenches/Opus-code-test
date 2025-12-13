#!/usr/bin/env python3
"""
Cross-Domain Semantic Bridge Analysis
======================================

Task #131: Investigate how concepts bridge across domains.

This script analyzes the corpus to find:
1. Terms that appear in multiple domains
2. Semantic relations that connect domains
3. Concept clusters that span domains

Usage:
    python scripts/analyze_cross_domain_bridges.py
    python scripts/analyze_cross_domain_bridges.py --corpus corpus_dev.pkl
    python scripts/analyze_cross_domain_bridges.py --min-domains 3
    python scripts/analyze_cross_domain_bridges.py --output docs/research/cross-domain-bridges.md
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortical import CorticalTextProcessor, CorticalLayer


def categorize_documents(processor: CorticalTextProcessor) -> Dict[str, Set[str]]:
    """
    Categorize documents into domains based on filename patterns.

    Returns:
        Dictionary mapping domain names to sets of document IDs
    """
    domains = defaultdict(set)

    for doc_id in processor.documents.keys():
        doc_lower = doc_id.lower()

        # Customer service
        if any(kw in doc_lower for kw in ['customer', 'support', 'complaint', 'ticket', 'call', 'retention', 'satisfaction']):
            domains['customer_service'].add(doc_id)

        # Machine learning
        if any(kw in doc_lower for kw in ['neural', 'machine', 'learning', 'deep', 'ml', 'model', 'training', 'attention']):
            domains['machine_learning'].add(doc_id)

        # Database
        if any(kw in doc_lower for kw in ['database', 'sql', 'query', 'relational', 'index']):
            domains['database'].add(doc_id)

        # Algorithms
        if any(kw in doc_lower for kw in ['algorithm', 'sort', 'search', 'quicksort', 'data_structure']):
            domains['algorithms'].add(doc_id)

        # Software engineering
        if any(kw in doc_lower for kw in ['code', 'software', 'programming', 'debug', 'test', 'refactor', 'review', 'development', 'incremental']):
            domains['software_engineering'].add(doc_id)

        # Finance/Trading
        if any(kw in doc_lower for kw in ['market', 'trading', 'financial', 'portfolio', 'investment', 'factor', 'volatility', 'liquidity']):
            domains['finance'].add(doc_id)

        # Systems
        if any(kw in doc_lower for kw in ['network', 'distributed', 'system', 'protocol', 'microservice']):
            domains['systems'].add(doc_id)

        # Knowledge graphs
        if any(kw in doc_lower for kw in ['graph', 'knowledge', 'semantic', 'ontology', 'wordnet', 'conceptnet']):
            domains['knowledge_graphs'].add(doc_id)

        # If no domain matched, put in 'other'
        if not any(doc_id in docs for docs in domains.values()):
            domains['other'].add(doc_id)

    return dict(domains)


def find_bridging_terms(
    processor: CorticalTextProcessor,
    domains: Dict[str, Set[str]],
    min_domains: int = 2,
    min_pagerank: float = 0.001
) -> List[Tuple[str, Set[str], float, int]]:
    """
    Find terms that appear in multiple domains.

    Args:
        processor: CorticalTextProcessor instance
        domains: Domain categorization
        min_domains: Minimum number of domains for a term to be considered bridging
        min_pagerank: Minimum PageRank threshold (to filter out rare terms)

    Returns:
        List of (term, domains_set, pagerank, total_docs) sorted by number of domains
    """
    layer0 = processor.get_layer(CorticalLayer.TOKENS)

    bridging_terms = []

    for col in layer0:
        if col.pagerank < min_pagerank:
            continue

        # Find which domains this term appears in
        term_domains = set()
        for domain, doc_ids in domains.items():
            if col.document_ids & doc_ids:  # Intersection
                term_domains.add(domain)

        if len(term_domains) >= min_domains:
            bridging_terms.append((
                col.content,
                term_domains,
                col.pagerank,
                len(col.document_ids)
            ))

    # Sort by number of domains (descending), then by PageRank
    bridging_terms.sort(key=lambda x: (-len(x[1]), -x[2]))

    return bridging_terms


def find_cross_domain_relations(
    processor: CorticalTextProcessor,
    domains: Dict[str, Set[str]],
    min_confidence: float = 0.5
) -> List[Tuple[str, str, str, float, Set[str], Set[str]]]:
    """
    Find semantic relations that connect terms from different domains.

    Returns:
        List of (term1, relation, term2, confidence, term1_domains, term2_domains)
    """
    if not processor.semantic_relations:
        processor.extract_corpus_semantics(verbose=False)

    layer0 = processor.get_layer(CorticalLayer.TOKENS)

    # Build term -> domains mapping
    term_domains = {}
    for col in layer0:
        domains_for_term = set()
        for domain, doc_ids in domains.items():
            if col.document_ids & doc_ids:
                domains_for_term.add(domain)
        term_domains[col.content] = domains_for_term

    cross_domain_relations = []

    for term1, relation, term2, weight in processor.semantic_relations:
        # Get domains for each term
        domains1 = term_domains.get(term1, set())
        domains2 = term_domains.get(term2, set())

        # Only keep relations where terms are from different domains
        if domains1 and domains2 and not (domains1 & domains2):
            # No domain overlap - true cross-domain relation
            if weight >= min_confidence:
                cross_domain_relations.append((
                    term1, relation, term2, weight, domains1, domains2
                ))

    # Sort by confidence
    cross_domain_relations.sort(key=lambda x: -x[3])

    return cross_domain_relations


def find_spanning_concepts(
    processor: CorticalTextProcessor,
    domains: Dict[str, Set[str]],
    min_domains: int = 2
) -> List[Tuple[str, Set[str], List[str]]]:
    """
    Find concept clusters that span multiple domains.

    Returns:
        List of (concept_id, domains_spanned, member_terms)
    """
    layer0 = processor.get_layer(CorticalLayer.TOKENS)
    layer2 = processor.get_layer(CorticalLayer.CONCEPTS)

    spanning_concepts = []

    for concept_col in layer2:
        # Get member terms
        member_terms = []
        concept_domains = set()

        for token_id in concept_col.feedforward_connections:
            token_col = layer0.get_by_id(token_id)
            if token_col:
                member_terms.append(token_col.content)

                # Find domains for this term
                for domain, doc_ids in domains.items():
                    if token_col.document_ids & doc_ids:
                        concept_domains.add(domain)

        if len(concept_domains) >= min_domains:
            spanning_concepts.append((
                concept_col.content,
                concept_domains,
                member_terms
            ))

    # Sort by number of domains spanned
    spanning_concepts.sort(key=lambda x: -len(x[1]))

    return spanning_concepts


def generate_markdown_report(
    processor: CorticalTextProcessor,
    domains: Dict[str, Set[str]],
    bridging_terms: List,
    cross_domain_relations: List,
    spanning_concepts: List
) -> str:
    """Generate markdown report of findings."""

    report = []
    report.append("# Cross-Domain Semantic Bridges")
    report.append("")
    report.append("**Task #131**: Investigation of how concepts bridge across domains in the corpus.")
    report.append("")
    report.append("This analysis identifies terms, relations, and concept clusters that connect different topic domains.")
    report.append("")
    report.append("---")
    report.append("")

    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append(f"- **Total documents**: {len(processor.documents)}")
    report.append(f"- **Domains identified**: {len([d for d in domains.keys() if d != 'other'])}")
    report.append(f"- **Bridging terms found**: {len(bridging_terms)}")
    report.append(f"- **Cross-domain relations**: {len(cross_domain_relations)}")
    report.append(f"- **Spanning concepts**: {len(spanning_concepts)}")
    report.append("")

    # Domain breakdown
    report.append("### Domain Breakdown")
    report.append("")
    report.append("| Domain | Documents |")
    report.append("|--------|-----------|")
    for domain, docs in sorted(domains.items(), key=lambda x: -len(x[1])):
        report.append(f"| {domain} | {len(docs)} |")
    report.append("")
    report.append("---")
    report.append("")

    # Bridging terms
    report.append("## 1. Bridging Terms")
    report.append("")
    report.append("Terms that appear in multiple domains, ranked by number of domains spanned.")
    report.append("")

    # Group by number of domains
    by_domain_count = defaultdict(list)
    for term, term_domains, pagerank, doc_count in bridging_terms[:30]:
        by_domain_count[len(term_domains)].append((term, term_domains, pagerank, doc_count))

    for domain_count in sorted(by_domain_count.keys(), reverse=True):
        terms = by_domain_count[domain_count]
        report.append(f"### Spanning {domain_count} Domains")
        report.append("")
        report.append("| Term | Domains | PageRank | Documents |")
        report.append("|------|---------|----------|-----------|")
        for term, term_domains, pagerank, doc_count in terms[:10]:
            domains_str = ", ".join(sorted(term_domains))
            report.append(f"| {term} | {domains_str} | {pagerank:.4f} | {doc_count} |")
        report.append("")

    report.append("**Key Insights:**")
    report.append("")
    if bridging_terms:
        top_term, top_domains, top_pr, _ = bridging_terms[0]
        report.append(f"- Most connected term: **{top_term}** spanning {len(top_domains)} domains")
        report.append(f"- Bridging terms indicate common concepts across specializations")
        report.append(f"- High PageRank bridging terms are central to corpus vocabulary")
    else:
        report.append("- No significant bridging terms found (domains may be too isolated)")
    report.append("")
    report.append("---")
    report.append("")

    # Cross-domain relations
    report.append("## 2. Cross-Domain Semantic Relations")
    report.append("")
    report.append("Semantic relations connecting terms from different domains.")
    report.append("")

    if cross_domain_relations:
        report.append("### Top Cross-Domain Connections")
        report.append("")
        report.append("| Term 1 (Domains) | Relation | Term 2 (Domains) | Confidence |")
        report.append("|------------------|----------|------------------|------------|")

        for term1, rel, term2, conf, domains1, domains2 in cross_domain_relations[:15]:
            d1_str = ", ".join(sorted(domains1))
            d2_str = ", ".join(sorted(domains2))
            report.append(f"| {term1} ({d1_str}) | {rel} | {term2} ({d2_str}) | {conf:.3f} |")
        report.append("")

        report.append("**Key Insights:**")
        report.append("")
        report.append("- Cross-domain relations reveal unexpected connections between fields")
        report.append("- These links could enable analogical reasoning across domains")
        report.append("- High-confidence relations suggest strong semantic bridges")
    else:
        report.append("*No cross-domain semantic relations found. This may indicate:*")
        report.append("- Domains are well-separated in the corpus")
        report.append("- Semantic extraction parameters need tuning")
        report.append("- More documents needed to establish cross-domain patterns")
    report.append("")
    report.append("---")
    report.append("")

    # Spanning concepts
    report.append("## 3. Spanning Concept Clusters")
    report.append("")
    report.append("Concept clusters whose member terms come from multiple domains.")
    report.append("")

    if spanning_concepts:
        report.append("### Multi-Domain Concepts")
        report.append("")
        for concept_id, concept_domains, members in spanning_concepts[:10]:
            report.append(f"#### Concept: {concept_id}")
            report.append("")
            report.append(f"**Domains**: {', '.join(sorted(concept_domains))}")
            report.append("")
            report.append(f"**Member terms** ({len(members)}): {', '.join(members[:20])}")
            if len(members) > 20:
                report.append(f"... and {len(members) - 20} more")
            report.append("")

        report.append("**Key Insights:**")
        report.append("")
        report.append("- Spanning concepts represent abstract ideas used across domains")
        report.append("- These clusters could serve as semantic bridges in search")
        report.append("- Cross-domain concepts enable knowledge transfer between fields")
    else:
        report.append("*No spanning concept clusters found.*")
        report.append("")
        report.append("This suggests concepts are domain-specific, which may be expected for specialized corpora.")
    report.append("")
    report.append("---")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("Based on this analysis:")
    report.append("")

    if bridging_terms:
        report.append("1. **Use bridging terms for cross-domain search**")
        report.append("   - Terms spanning multiple domains can improve recall across topics")
        report.append(f"   - Focus on high-PageRank bridging terms like '{bridging_terms[0][0]}'")
        report.append("")

    if cross_domain_relations:
        report.append("2. **Leverage cross-domain relations for analogies**")
        report.append("   - Semantic relations connecting domains enable analogical reasoning")
        report.append("   - Could enhance query expansion across topic boundaries")
        report.append("")

    if spanning_concepts:
        report.append("3. **Exploit spanning concepts for knowledge transfer**")
        report.append("   - Concepts spanning domains represent shared abstractions")
        report.append("   - Use for finding similar problems in different contexts")
        report.append("")

    report.append("4. **Expand corpus for richer cross-domain connections**")
    report.append("   - More documents increase chances of discovering bridges")
    report.append("   - Diverse topics enhance cross-domain semantic richness")
    report.append("")

    report.append("---")
    report.append("")
    report.append("*Generated by `scripts/analyze_cross_domain_bridges.py`*")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-domain semantic bridges in the corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--corpus", "-c",
        help="Path to saved corpus file (default: load from samples/)"
    )
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Directory containing sample documents (default: samples/)"
    )
    parser.add_argument(
        "--min-domains",
        type=int,
        default=2,
        help="Minimum domains for bridging terms (default: 2)"
    )
    parser.add_argument(
        "--min-pagerank",
        type=float,
        default=0.001,
        help="Minimum PageRank for bridging terms (default: 0.001)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output markdown file (default: print to stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Load or create processor
    if args.corpus and os.path.exists(args.corpus):
        if args.verbose:
            print(f"Loading corpus from {args.corpus}...")
        processor = CorticalTextProcessor.load(args.corpus)
        if args.verbose:
            print(f"Loaded {len(processor.documents)} documents")
    else:
        if args.verbose:
            print(f"Loading documents from {args.samples_dir}/...")
        processor = CorticalTextProcessor()
        samples_path = Path(args.samples_dir)

        if not samples_path.is_dir():
            print(f"Error: Samples directory not found: {args.samples_dir}")
            sys.exit(1)

        num_loaded = 0
        for filepath in sorted(samples_path.glob("*.txt")):
            try:
                content = filepath.read_text(encoding='utf-8')
                processor.process_document(filepath.stem, content)
                num_loaded += 1
            except Exception as e:
                print(f"Warning: Error loading {filepath.name}: {e}")

        if num_loaded == 0:
            print("Error: No documents loaded")
            sys.exit(1)

        if args.verbose:
            print(f"Loaded {num_loaded} documents, computing analysis...")
        processor.compute_all(verbose=args.verbose)

    # Categorize documents into domains
    if args.verbose:
        print("Categorizing documents into domains...")
    domains = categorize_documents(processor)

    if args.verbose:
        print("Domain breakdown:")
        for domain, docs in sorted(domains.items(), key=lambda x: -len(x[1])):
            print(f"  {domain}: {len(docs)} docs")

    # Find bridging terms
    if args.verbose:
        print(f"Finding bridging terms (min_domains={args.min_domains})...")
    bridging_terms = find_bridging_terms(
        processor, domains, args.min_domains, args.min_pagerank
    )

    # Find cross-domain relations
    if args.verbose:
        print("Finding cross-domain semantic relations...")
    cross_domain_relations = find_cross_domain_relations(processor, domains)

    # Find spanning concepts
    if args.verbose:
        print("Finding spanning concept clusters...")
    spanning_concepts = find_spanning_concepts(processor, domains, args.min_domains)

    # Generate report
    report = generate_markdown_report(
        processor, domains, bridging_terms,
        cross_domain_relations, spanning_concepts
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
