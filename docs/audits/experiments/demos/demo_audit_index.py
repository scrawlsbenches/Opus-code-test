"""
Demonstration of AuditInvertedIndex with real audit finding examples.
"""

from audit_inverted_index import AuditInvertedIndex


def demo_real_audit_findings():
    """Demonstrate indexing real audit findings."""
    print("=" * 70)
    print("AuditInvertedIndex Demo - Real Audit Findings")
    print("=" * 70)

    idx = AuditInvertedIndex()

    # Index some real-style audit findings
    findings = {
        "F001": "FUTURE: When CDG index is implemented, this will be handled at the storage layer",
        "F002": "TODO: Add decision tracking to the GoT system. See: docs/design/got-decisions.md",
        "F003": "This will be replaced when the new API is ready",
        "F004": "NOTE: Performance optimization deferred. Will be addressed in Q2",
        "F005": "See: cortical/cdg/storage.py for implementation details",
        "F006": "HACK: Temporary workaround. This will be removed once proper solution exists",
    }

    print("\n📝 Indexing findings...")
    for finding_id, text in findings.items():
        idx.index_text(finding_id, text)
        print(f"  {finding_id}: {text[:60]}...")

    # Demo 1: Find all "will be" patterns (common misleading comment)
    print("\n🔍 Demo 1: Find 'will be' pattern (misleading future promises)")
    print("-" * 70)
    will_be_findings = idx.search_phrase(["will", "be"])
    print(f"Found {len(will_be_findings)} findings with 'will be':")
    for fid in will_be_findings:
        print(f"  ✓ {fid}: {findings[fid]}")

    # Demo 2: Find all "See:" references
    print("\n🔍 Demo 2: Find 'See:' documentation references")
    print("-" * 70)
    see_results = idx.search("see:")
    print(f"Found {len(see_results)} findings with 'see:':")
    for fid, positions in see_results:
        print(f"  ✓ {fid} (positions: {positions}): {findings[fid]}")

    # Demo 3: Term frequency analysis
    print("\n📊 Demo 3: Term frequency analysis")
    print("-" * 70)
    common_terms = ["the", "will", "be", "when", "this"]
    print("Term frequencies across all findings:")
    for term in common_terms:
        total_freq = sum(idx.term_frequency(term, fid) for fid in findings.keys())
        if total_freq > 0:
            print(f"  '{term}': {total_freq} occurrences")

    # Demo 4: Finding specific patterns
    print("\n🎯 Demo 4: Pattern detection")
    print("-" * 70)

    patterns = {
        "Future promises": ["will", "be"],
        "TODO markers": ["todo:"],
        "FUTURE markers": ["future:"],
        "Deferred work": ["will", "be", "handled"],
    }

    for pattern_name, terms in patterns.items():
        if len(terms) == 1:
            results = idx.search(terms[0])
            count = len(results)
        else:
            results = idx.search_phrase(terms)
            count = len(results)
        print(f"  {pattern_name} ({' '.join(terms)}): {count} findings")

    # Demo 5: Remove a finding and verify cleanup
    print("\n🗑️  Demo 5: Finding removal")
    print("-" * 70)
    print("Before removal:")
    results = idx.search("temporary")
    print(f"  'temporary' found in {len(results)} findings")

    idx.remove_finding("F006")
    print("\nAfter removing F006:")
    results = idx.search("temporary")
    print(f"  'temporary' found in {len(results)} findings")

    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_real_audit_findings()
