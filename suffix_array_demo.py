"""
Demonstration: Using Suffix Array to Find Misleading Comment Patterns

This shows how the suffix array implementation can be used in the Cortical
codebase audit to detect copy-pasted misleading comments.
"""

from suffix_array_implementation import CommentPatternFinder


def demonstrate_comment_pattern_mining():
    """
    Simulate finding copy-pasted misleading patterns in actual code comments.
    """
    print("=" * 80)
    print("SUFFIX ARRAY FOR COMMENT PATTERN MINING - DEMONSTRATION")
    print("=" * 80)

    # Simulate audit comments from the codebase
    audit_comments = """
File: cortical/cdg/indexer.py
Comment: FUTURE: When CDG index is implemented this will be handled at storage layer

File: cortical/cdg/storage.py
Comment: See: docs/design/cdg-transactional-indexing-design.md for implementation details

File: cortical/cdg/transaction.py
Comment: FUTURE: When CDG index is implemented this will be replaced with proper indexing

File: cortical/got/manager.py
Comment: TODO: Add error handling for edge cases

File: cortical/cel/event_store.py
Comment: See: docs/design/cdg-transactional-indexing-design.md

File: cortical/cdg/query.py
Comment: FUTURE: When CDG index is implemented this will use batch operations

File: cortical/reasoning/prism.py
Comment: NOTE: This is a placeholder implementation

File: cortical/cdg/recovery.py
Comment: See: docs/design/cdg-transactional-indexing-design.md for recovery protocol
"""

    print("\n📝 AUDIT COMMENTS FROM CODEBASE")
    print("-" * 80)
    print(audit_comments.strip())

    # Build suffix array for all comments
    print("\n\n🔧 BUILDING SUFFIX ARRAY...")
    finder = CommentPatternFinder(audit_comments)
    print(f"✓ Suffix array built: {len(finder.suffixes)} suffixes")
    print(f"✓ LCP array computed: {len(finder.lcp_array())} values")

    # Find repeated patterns
    print("\n\n🔍 FINDING REPEATED PATTERNS (length >= 20)")
    print("-" * 80)
    repeated = finder.repeated_substrings(min_length=20)

    print(f"\nFound {len(repeated)} unique repeated patterns\n")

    # Categorize findings
    misleading_futures = []
    doc_references = []
    other_patterns = []

    for pattern, count in repeated:
        if "FUTURE:" in pattern and "CDG index is implemented" in pattern:
            misleading_futures.append((pattern, count))
        elif "docs/design/" in pattern:
            doc_references.append((pattern, count))
        else:
            other_patterns.append((pattern, count))

    # Report findings
    print("🚨 CATEGORY 1: Misleading FUTURE comments")
    print("-" * 80)
    if misleading_futures:
        for pattern, count in misleading_futures[:5]:
            print(f"\nPattern (count={count}):")
            print(f"  '{pattern}'")
            positions = finder.search(pattern)
            print(f"  Found at positions: {positions}")
    else:
        print("  (None found)")

    print("\n\n📄 CATEGORY 2: Documentation references")
    print("-" * 80)
    if doc_references:
        for pattern, count in doc_references[:5]:
            print(f"\nPattern (count={count}):")
            print(f"  '{pattern}'")
            positions = finder.search(pattern)
            print(f"  Found at positions: {positions}")
    else:
        print("  (None found)")

    print("\n\n📋 CATEGORY 3: Other repeated patterns")
    print("-" * 80)
    if other_patterns:
        for pattern, count in other_patterns[:5]:
            print(f"\nPattern (count={count}):")
            print(f"  '{pattern[:60]}{'...' if len(pattern) > 60 else ''}'")
    else:
        print("  (None found)")

    # Specific pattern searches
    print("\n\n🎯 TARGETED SEARCHES")
    print("-" * 80)

    searches = [
        ("FUTURE: When CDG index is implemented", "Copy-pasted FUTURE comments"),
        ("docs/design/cdg-transactional-indexing-design.md", "Non-existent doc references"),
        ("will be implemented", "Vague promises"),
        ("TODO:", "TODO markers"),
    ]

    for pattern, description in searches:
        positions = finder.search(pattern)
        print(f"\n{description}:")
        print(f"  Pattern: '{pattern}'")
        print(f"  Occurrences: {len(positions)}")
        if positions:
            print(f"  Positions: {positions}")

    # Analysis summary
    print("\n\n📊 ANALYSIS SUMMARY")
    print("=" * 80)

    future_count = len(finder.search("FUTURE: When CDG index is implemented"))
    doc_count = len(finder.search("docs/design/cdg-transactional-indexing-design.md"))

    print(f"""
FINDINGS:
1. Copy-pasted "FUTURE" comments: {future_count} occurrences
   - Pattern: "FUTURE: When CDG index is implemented"
   - Risk: Misleading comments suggesting unimplemented features
   - Action: Verify if CDG index is actually unimplemented

2. Repeated documentation references: {doc_count} occurrences
   - Pattern: "docs/design/cdg-transactional-indexing-design.md"
   - Risk: References to potentially non-existent documentation
   - Action: Verify file exists or remove references

3. Longest repeated substring: {repeated[0][0][:60]}... (length={len(repeated[0][0])})

CONCLUSION:
The suffix array successfully identifies copy-pasted comment patterns that may
indicate misleading or outdated documentation. These patterns should be audited
to ensure code comments accurately reflect the current implementation status.
""")


def demonstrate_algorithm_details():
    """
    Show detailed algorithm operation on a simple example.
    """
    print("\n\n" + "=" * 80)
    print("ALGORITHM DEEP DIVE: 'banana' Example")
    print("=" * 80)

    text = "banana"
    finder = CommentPatternFinder(text)

    print(f"\nInput Text: '{text}'")
    print(f"Length: {len(text)}")

    print("\n\n1️⃣  SUFFIX ARRAY CONSTRUCTION")
    print("-" * 80)
    print("\nAll suffixes:")
    for i in range(len(text)):
        print(f"  suffix[{i}] = '{text[i:]}'")

    print("\nSorted suffixes (lexicographically):")
    for pos, idx in enumerate(finder.suffixes):
        print(f"  position {pos}: suffix[{idx}] = '{text[idx:]}'")

    print(f"\nSuffix Array: {finder.suffixes}")

    print("\n\n2️⃣  LCP ARRAY (Kasai's Algorithm)")
    print("-" * 80)
    lcp = finder.lcp_array()
    print(f"LCP Array: {lcp}\n")

    print("Explanation:")
    for i in range(len(lcp)):
        if i == 0:
            print(f"  lcp[{i}] = {lcp[i]} (first element, by convention)")
        else:
            idx_prev = finder.suffixes[i-1]
            idx_curr = finder.suffixes[i]
            suf_prev = text[idx_prev:]
            suf_curr = text[idx_curr:]
            common = text[idx_curr:idx_curr+lcp[i]] if lcp[i] > 0 else "(none)"
            print(f"  lcp[{i}] = {lcp[i]}: '{suf_prev}' ∩ '{suf_curr}' = '{common}'")

    print("\n\n3️⃣  PATTERN SEARCH (Binary Search)")
    print("-" * 80)

    patterns = ["ana", "na", "ban", "xyz"]
    for pattern in patterns:
        positions = finder.search(pattern)
        print(f"\nSearch for '{pattern}':")
        if positions:
            print(f"  ✓ Found at positions: {positions}")
            for pos in positions:
                print(f"    text[{pos}:{pos+len(pattern)}] = '{text[pos:pos+len(pattern)]}'")
        else:
            print(f"  ✗ Not found")

    print("\n\n4️⃣  REPEATED SUBSTRINGS")
    print("-" * 80)
    repeated = finder.repeated_substrings(min_length=1)
    print(f"\nFound {len(repeated)} repeated substrings:\n")
    for substring, count in repeated:
        positions = finder.search(substring)
        print(f"  '{substring}' appears {count} times at positions {positions}")


if __name__ == "__main__":
    # Main demonstration
    demonstrate_comment_pattern_mining()

    # Algorithm details
    demonstrate_algorithm_details()

    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
