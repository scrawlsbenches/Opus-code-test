"""
Demonstration of Union-Find optimizations and edge cases
Shows path compression and union by rank in action
"""

from finding_cluster import FindingCluster


def demo_path_compression():
    """Demonstrate path compression optimization"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Path Compression")
    print("=" * 70)

    uf = FindingCluster()

    # Create a long chain
    findings = [f"F{i:03d}" for i in range(1, 11)]
    for f in findings:
        uf.make_set(f)

    print("\n1. Building a chain by sequential unions:")
    print("   F001 <- F002 <- F003 <- ... <- F010")

    for i in range(len(findings) - 1):
        uf.union(findings[i], findings[i + 1])

    print("\n2. Before path compression on F010:")
    print(f"   Parent structure (showing chain):")
    # Note: After union by rank, structure might be balanced, not a pure chain
    for f in findings[-3:]:
        print(f"   {f} -> {uf._parent[f]}")

    print("\n3. Calling find('F010')...")
    root = uf.find("F010")
    print(f"   Root found: {root}")

    print("\n4. After path compression:")
    print(f"   All nodes now point directly (or near) to root:")
    for f in findings[-3:]:
        print(f"   {f} -> {uf._parent[f]}")

    print("\n   ✓ Path compression flattens tree for O(α(n)) performance")


def demo_union_by_rank():
    """Demonstrate union by rank optimization"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Union by Rank")
    print("=" * 70)

    uf = FindingCluster()

    # Create two separate trees
    print("\n1. Creating two separate clusters:")

    # Cluster A: F001, F002, F003, F004 (will have higher rank)
    cluster_a = ["F001", "F002", "F003", "F004"]
    for f in cluster_a:
        uf.make_set(f)
    uf.union("F001", "F002")
    uf.union("F003", "F004")
    uf.union("F001", "F003")

    # Cluster B: F005, F006 (will have lower rank)
    cluster_b = ["F005", "F006"]
    for f in cluster_b:
        uf.make_set(f)
    uf.union("F005", "F006")

    root_a = uf.find("F001")
    root_b = uf.find("F005")

    print(f"   Cluster A (4 elements): root = {root_a}, rank = {uf._rank[root_a]}")
    print(f"   Cluster B (2 elements): root = {root_b}, rank = {uf._rank[root_b]}")

    print("\n2. Unioning the two clusters:")
    print(f"   union({cluster_a[0]}, {cluster_b[0]})")

    uf.union(cluster_a[0], cluster_b[0])

    final_root = uf.find("F001")
    print(f"\n3. Result:")
    print(f"   Smaller tree (B) attached under larger tree (A)")
    print(f"   New root: {final_root}")
    print(f"   Rank of root: {uf._rank[final_root]}")

    print("\n   ✓ Union by rank keeps trees balanced, preventing degeneration")


def demo_edge_cases():
    """Demonstrate edge case handling"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Edge Cases")
    print("=" * 70)

    print("\n1. Self-union (unioning an element with itself):")
    uf = FindingCluster()
    uf.make_set("F001")
    result = uf.union("F001", "F001")
    print(f"   union('F001', 'F001') returned: {result}")
    print(f"   ✓ Correctly returns False (no merge needed)")

    print("\n2. Multiple make_set calls (idempotent):")
    uf = FindingCluster()
    uf.make_set("F001")
    parent_before = uf._parent["F001"]
    rank_before = uf._rank["F001"]
    uf.make_set("F001")  # Second call
    uf.make_set("F001")  # Third call
    print(f"   Parent unchanged: {uf._parent['F001'] == parent_before}")
    print(f"   Rank unchanged: {uf._rank['F001'] == rank_before}")
    print(f"   ✓ make_set is idempotent (safe to call multiple times)")

    print("\n3. Auto-creation in union:")
    uf = FindingCluster()
    # Union of non-existent elements
    uf.union("NEW1", "NEW2")
    print(f"   union('NEW1', 'NEW2') auto-created both elements")
    print(f"   NEW1 in structure: {'NEW1' in uf._parent}")
    print(f"   NEW2 in structure: {'NEW2' in uf._parent}")
    print(f"   connected('NEW1', 'NEW2'): {uf.connected('NEW1', 'NEW2')}")
    print(f"   ✓ Convenience feature for easier usage")

    print("\n4. Error on accessing non-existent element:")
    uf = FindingCluster()
    try:
        uf.find("NONEXISTENT")
        print("   ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Raised ValueError: {e}")

    print("\n5. Empty cluster operations:")
    uf = FindingCluster()
    count = uf.cluster_count()
    clusters = uf.get_all_clusters()
    print(f"   cluster_count() on empty structure: {count}")
    print(f"   get_all_clusters() on empty structure: {clusters}")
    print(f"   ✓ Handles empty structure gracefully")

    print("\n6. Single-element cluster:")
    uf = FindingCluster()
    uf.make_set("SOLO")
    cluster = uf.get_cluster("SOLO")
    print(f"   Single element cluster: {cluster}")
    print(f"   ✓ Correctly returns set with one element")


def demo_practical_usage():
    """Demonstrate practical usage for audit findings"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Practical Audit Finding Clustering")
    print("=" * 70)

    uf = FindingCluster()

    # Simulate real audit findings with metadata
    audit_findings = {
        "F001": {
            "file": "cortical/got/indexer.py",
            "line": 478,
            "pattern": "See: missing-doc",
            "severity": "medium"
        },
        "F002": {
            "file": "cortical/got/indexer.py",
            "line": 508,
            "pattern": "See: missing-doc",
            "severity": "medium"
        },
        "F003": {
            "file": "cortical/got/api.py",
            "line": 123,
            "pattern": "TODO",
            "severity": "low"
        },
        "F004": {
            "file": "cortical/cdg/storage.py",
            "line": 89,
            "pattern": "will be",
            "severity": "low"
        },
        "F005": {
            "file": "cortical/cdg/transaction.py",
            "line": 234,
            "pattern": "will be",
            "severity": "low"
        },
        "F006": {
            "file": "cortical/cel/events.py",
            "line": 56,
            "pattern": "FIXME",
            "severity": "high"
        },
    }

    print("\n1. Initializing all findings:")
    for finding_id in audit_findings:
        uf.make_set(finding_id)
    print(f"   Created {len(audit_findings)} findings")

    print("\n2. Clustering by pattern type:")
    # Cluster same-pattern findings
    uf.union("F001", "F002")  # Both "See: missing-doc"
    uf.union("F004", "F005")  # Both "will be"

    print(f"   Merged F001 + F002 (missing-doc pattern)")
    print(f"   Merged F004 + F005 (will-be pattern)")

    print("\n3. Current clusters:")
    all_clusters = uf.get_all_clusters()
    for i, cluster in enumerate(sorted(all_clusters, key=len, reverse=True), 1):
        print(f"   Cluster {i}: {sorted(cluster)} ({len(cluster)} findings)")

        # Show what they have in common
        if len(cluster) > 1:
            patterns = [audit_findings[fid]["pattern"] for fid in cluster]
            print(f"      Pattern: {patterns[0]}")

    print(f"\n4. Query operations:")
    print(f"   Are F001 and F002 related? {uf.connected('F001', 'F002')}")
    print(f"   Are F001 and F003 related? {uf.connected('F001', 'F003')}")
    print(f"   Total clusters: {uf.cluster_count()}")

    print("\n5. Finding all related issues to F001:")
    related = uf.get_cluster("F001")
    print(f"   F001 is clustered with: {sorted(related)}")
    print(f"   When fixing F001, also check: {sorted(related - {'F001'})}")

    print("\n   ✓ Union-Find enables efficient finding grouping and batch operations")


if __name__ == "__main__":
    demo_path_compression()
    demo_union_by_rank()
    demo_edge_cases()
    demo_practical_usage()

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)
