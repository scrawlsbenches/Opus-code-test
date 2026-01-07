"""
Verify AuditInvertedIndex with actual audit findings from the repository.
"""

from pathlib import Path
from audit_inverted_index import AuditInvertedIndex


def extract_comment_from_finding(finding_text: str) -> str:
    """Extract the actual comment text from a finding in the audit result."""
    lines = finding_text.split('\n')
    in_comment = False
    comment_lines = []

    for line in lines:
        if line.strip().startswith('**Comment:**'):
            in_comment = True
            continue
        if in_comment:
            if line.strip().startswith('```python'):
                continue
            if line.strip() == '```':
                break
            if line.strip():
                comment_lines.append(line.strip())

    return ' '.join(comment_lines)


def test_with_real_audit_findings():
    """Test indexing with real audit findings from the repository."""
    print("=" * 70)
    print("Real Audit Findings Verification")
    print("=" * 70)

    idx = AuditInvertedIndex()

    # Test with the exact finding from the experiment
    finding_1 = """FUTURE: When CDG index is implemented, this will be handled at the
storage layer with WAL-based recovery. See:
docs/design/cdg-transactional-indexing-design.md"""

    print("\n📁 Indexing real finding from experiment...")
    idx.index_text("FINDING-1", finding_1)
    print(f"  ✓ Indexed: {finding_1[:60]}...")

    # Test phrase search for "will be"
    print("\n🔍 Searching for 'will be' pattern...")
    will_be_results = idx.search_phrase(["will", "be"])
    assert "FINDING-1" in will_be_results, "Should find 'will be' in finding"
    print(f"  ✓ Found in: {will_be_results}")

    # Test search for "see:"
    print("\n🔍 Searching for 'see:' pattern...")
    see_results = idx.search("see:")
    assert len(see_results) == 1, "Should find one 'see:' reference"
    assert see_results[0][0] == "FINDING-1", "Should be in FINDING-1"
    print(f"  ✓ Found in: {[r[0] for r in see_results]}")

    # Test search for "FUTURE:"
    print("\n🔍 Searching for 'future:' marker...")
    future_results = idx.search("future:")
    assert len(future_results) == 1, "Should find one 'future:' marker"
    print(f"  ✓ Found in: {[r[0] for r in future_results]}")

    # Add more real-style findings
    findings = {
        "F-CDG-001": "FUTURE: WAL-based recovery will be implemented in Q2",
        "F-GOT-001": "TODO: Add edge validation. See: docs/design/got-schema.md",
        "F-CORE-001": "HACK: Temporary workaround for bootstrap issue",
        "F-PRISM-001": "This will be replaced when proper synaptic decay is implemented",
    }

    print("\n📁 Indexing additional real-style findings...")
    for fid, text in findings.items():
        idx.index_text(fid, text)
        print(f"  ✓ {fid}: {text[:50]}...")

    # Test pattern detection across all findings
    print("\n📊 Pattern Analysis Across All Findings")
    print("-" * 70)

    patterns = {
        "Future promises ('will be')": ["will", "be"],
        "TODO markers": ["todo:"],
        "FUTURE markers": ["future:"],
        "See references": ["see:"],
        "HACK markers": ["hack:"],
        "Temporary workarounds": ["temporary"],
    }

    for pattern_name, terms in patterns.items():
        if len(terms) == 1:
            results = idx.search(terms[0])
            findings_with_pattern = [r[0] for r in results]
        else:
            findings_with_pattern = idx.search_phrase(terms)

        if findings_with_pattern:
            print(f"\n  {pattern_name}:")
            print(f"    Found in {len(findings_with_pattern)} findings: {findings_with_pattern}")

    # Verify expected patterns
    print("\n✓ Verification Tests")
    print("-" * 70)

    # Should find multiple "will be" patterns
    will_be_all = idx.search_phrase(["will", "be"])
    assert len(will_be_all) >= 3, f"Should find at least 3 'will be' patterns, found {len(will_be_all)}"
    print(f"  ✓ Found {len(will_be_all)} 'will be' patterns (expected ≥3)")

    # Should find FUTURE markers
    future_all = idx.search("future:")
    assert len(future_all) >= 2, f"Should find at least 2 FUTURE markers, found {len(future_all)}"
    print(f"  ✓ Found {len(future_all)} FUTURE markers (expected ≥2)")

    # Should find See references
    see_all = idx.search("see:")
    assert len(see_all) == 2, f"Should find 2 See references, found {len(see_all)}"
    print(f"  ✓ Found {len(see_all)} See references (expected 2)")

    # Test term frequency on real data
    print("\n📈 Term Frequency Analysis")
    print("-" * 70)
    important_terms = ["will", "be", "implemented", "future:", "see:"]
    for term in important_terms:
        total_freq = sum(idx.term_frequency(term, fid) for fid in ["FINDING-1"] + list(findings.keys()))
        if total_freq > 0:
            print(f"  '{term}': {total_freq} total occurrences")

    print("\n" + "=" * 70)
    print("✓ All real-world verification tests PASSED!")
    print("=" * 70)
    print("\nConclusion:")
    print("  The AuditInvertedIndex successfully indexes and searches")
    print("  real audit findings from the Cortical codebase.")
    print("  Ready for integration with the audit system.")


def test_case_sensitivity_with_real_markers():
    """Verify case-insensitive search works with real markers."""
    print("\n" + "=" * 70)
    print("Case Sensitivity Test with Real Markers")
    print("=" * 70)

    idx = AuditInvertedIndex()

    # Index with various cases
    idx.index_text("F1", "FUTURE: Will be done")
    idx.index_text("F2", "future: Already planned")
    idx.index_text("F3", "Future: In progress")

    print("\n🔍 Testing case-insensitive search...")

    # All variations should find all 3 findings
    for variant in ["future:", "FUTURE:", "Future:", "FuTuRe:"]:
        results = idx.search(variant)
        assert len(results) == 3, f"Search for '{variant}' should find 3 findings, found {len(results)}"
        print(f"  ✓ '{variant}' → {len(results)} findings")

    print("\n✓ Case sensitivity test PASSED")


if __name__ == "__main__":
    test_with_real_audit_findings()
    test_case_sensitivity_with_real_markers()

    print("\n" + "=" * 70)
    print("✓✓✓ ALL REAL-WORLD TESTS PASSED ✓✓✓")
    print("=" * 70)
