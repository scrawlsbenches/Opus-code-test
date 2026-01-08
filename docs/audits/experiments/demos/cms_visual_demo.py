"""
Visual demonstration of Count-Min Sketch internals

Shows how hash functions, collision, and minimum work together.
"""

from pattern_frequency_sketch import PatternFrequencySketch
import hashlib


def visualize_hash_distribution():
    """Show how double hashing distributes patterns across rows"""
    print("\n" + "=" * 70)
    print("HASH DISTRIBUTION VISUALIZATION")
    print("=" * 70)

    cms = PatternFrequencySketch(width=20, depth=3)
    patterns = ["FUTURE:", "TODO:", "will be", "See:", "NOTE:"]

    print("\nConfiguration: width=20, depth=3")
    print("\nDouble hashing formula: h_i(x) = (hash1(x) + i × hash2(x)) % 20")

    for pattern in patterns:
        print(f"\nPattern: '{pattern}'")
        md5_hash = hashlib.md5(pattern.encode('utf-8')).hexdigest()
        hash1 = int(md5_hash[:8], 16)
        hash2 = int(md5_hash[8:16], 16)
        print(f"  MD5: {md5_hash}")
        print(f"  hash1: {hash1} (first 8 hex digits)")
        print(f"  hash2: {hash2} (next 8 hex digits)")

        for row in range(3):
            col = (hash1 + row * hash2) % 20
            print(f"  Row {row}: ({hash1} + {row} × {hash2}) % 20 = {col}")


def visualize_collision_and_minimum():
    """Show how collisions occur and why minimum helps"""
    print("\n" + "=" * 70)
    print("COLLISION & MINIMUM VISUALIZATION")
    print("=" * 70)

    cms = PatternFrequencySketch(width=10, depth=3)

    # Add patterns that might collide
    patterns = {
        "FUTURE:": 10,
        "TODO:": 5,
        "will be": 15,
    }

    print("\nConfiguration: width=10, depth=3 (small width → collisions likely)")
    print("\nAdding patterns:")

    for pattern, count in patterns.items():
        print(f"  {pattern}: count={count}")
        cms.add(pattern, count)

    # Show internal state after additions
    print("\n" + "-" * 70)
    print("Internal counter state (after all additions):")
    print("-" * 70)

    for row in range(3):
        print(f"\nRow {row}: ", end="")
        for col in range(10):
            count = cms._counters[row][col]
            if count > 0:
                print(f"[{col}]={count:2} ", end="")
            else:
                print(f"[{col}]={count:2} ", end="")
        print()

    # Query each pattern and show the process
    print("\n" + "-" * 70)
    print("Query process (showing why we take minimum):")
    print("-" * 70)

    for pattern in patterns.keys():
        print(f"\nQuerying '{pattern}':")
        estimates = []

        for row in range(3):
            col = cms._hash(pattern, row)
            estimate = cms._counters[row][col]
            estimates.append(estimate)
            print(f"  Row {row}: hash to col {col}, counter={estimate}")

        result = min(estimates)
        actual = patterns[pattern]
        print(f"  → min({estimates}) = {result}")
        print(f"  → Actual count: {actual}")
        print(f"  → Overestimate: {result - actual} (error: {((result/actual - 1) * 100):.1f}%)")


def visualize_merge_operation():
    """Show how merge combines two sketches"""
    print("\n" + "=" * 70)
    print("MERGE OPERATION VISUALIZATION")
    print("=" * 70)

    cms1 = PatternFrequencySketch(width=10, depth=3)
    cms2 = PatternFrequencySketch(width=10, depth=3)

    print("\nScenario: Analyzing two different code modules")
    print("\nModule 1 (cortical/got/):")
    cms1.add("FUTURE:", 5)
    cms1.add("TODO:", 3)
    print("  FUTURE: → 5")
    print("  TODO: → 3")

    print("\nModule 2 (cortical/cdg/):")
    cms2.add("FUTURE:", 3)
    cms2.add("See:", 4)
    print("  FUTURE: → 3")
    print("  See: → 4")

    print("\n" + "-" * 70)
    print("Merging: merged = cms1.merge(cms2)")
    print("-" * 70)

    merged = cms1.merge(cms2)

    print("\nElement-wise addition of counters:")
    print("(showing only non-zero buckets)\n")

    for row in range(3):
        print(f"Row {row}:")
        for col in range(10):
            c1 = cms1._counters[row][col]
            c2 = cms2._counters[row][col]
            cm = merged._counters[row][col]
            if c1 > 0 or c2 > 0 or cm > 0:
                print(f"  [{col}]: {c1} + {c2} = {cm}")

    print("\n" + "-" * 70)
    print("Query results after merge:")
    print("-" * 70)

    patterns = ["FUTURE:", "TODO:", "See:"]
    for pattern in patterns:
        result = merged.query(pattern)
        c1 = cms1.query(pattern)
        c2 = cms2.query(pattern)
        print(f"\n{pattern}:")
        print(f"  cms1.query() = {c1}")
        print(f"  cms2.query() = {c2}")
        print(f"  merged.query() = {result} (should be >= {c1 + c2})")


def visualize_accuracy_vs_width():
    """Show how width affects accuracy"""
    print("\n" + "=" * 70)
    print("ACCURACY vs WIDTH TRADE-OFF")
    print("=" * 70)

    configs = [
        (10, 3, "Small width → high collision"),
        (100, 3, "Medium width → moderate collision"),
        (1000, 3, "Large width → low collision"),
    ]

    patterns = {
        "FUTURE:": 10,
        "TODO:": 10,
        "will be": 10,
        "See:": 10,
        "NOTE:": 10,
    }

    print("\nAdding same 5 patterns (count=10 each) to different sketch sizes:")

    for width, depth, description in configs:
        print(f"\n{description}")
        print(f"Configuration: width={width}, depth={depth}")
        print(f"Expected avg collision: {50 / width:.1f} items per bucket\n")

        cms = PatternFrequencySketch(width=width, depth=depth)

        for pattern, count in patterns.items():
            cms.add(pattern, count)

        total_error = 0
        for pattern, actual in patterns.items():
            estimate = cms.query(pattern)
            error = estimate - actual
            error_pct = (error / actual * 100) if actual > 0 else 0
            total_error += error_pct
            print(f"  {pattern:12} → estimate={estimate:3}, actual={actual:2}, error={error:2} ({error_pct:5.1f}%)")

        avg_error = total_error / len(patterns)
        print(f"  Average error: {avg_error:.1f}%")


def visualize_streaming_scenario():
    """Show real-world streaming scenario"""
    print("\n" + "=" * 70)
    print("STREAMING COMMENT ANALYSIS SCENARIO")
    print("=" * 70)

    cms = PatternFrequencySketch(width=100, depth=5)

    # Simulate streaming through comments
    comments = [
        "# FUTURE: This will be implemented",
        "# TODO: Fix this bug",
        "# NOTE: See documentation",
        "# This will be refactored",
        "# FUTURE: Add caching",
        "# will be updated",
        "# See: related_module.py",
        "# FUTURE: Optimize",
        "# TODO: Add tests",
        "# will be deprecated",
    ]

    print("\nStreaming through 10 comments from code...")
    print("Tracking patterns: FUTURE:, TODO:, will be, See:\n")

    for i, comment in enumerate(comments, 1):
        print(f"Comment {i}: {comment}")

        # Extract patterns
        if "FUTURE:" in comment:
            cms.add("FUTURE:", 1)
            print("  → Added 'FUTURE:' +1")
        if "TODO:" in comment:
            cms.add("TODO:", 1)
            print("  → Added 'TODO:' +1")
        if "will be" in comment:
            cms.add("will be", 1)
            print("  → Added 'will be' +1")
        if "See:" in comment:
            cms.add("See:", 1)
            print("  → Added 'See:' +1")

    print("\n" + "-" * 70)
    print("Final frequency report:")
    print("-" * 70)

    pattern_counts = [
        ("FUTURE:", cms.query("FUTURE:")),
        ("will be", cms.query("will be")),
        ("TODO:", cms.query("TODO:")),
        ("See:", cms.query("See:")),
    ]

    # Sort by frequency
    pattern_counts.sort(key=lambda x: x[1], reverse=True)

    print("\nPattern frequencies (sorted by count):\n")
    for pattern, count in pattern_counts:
        bar = "█" * count
        print(f"  {pattern:12} {bar} ({count})")

    print(f"\nTotal patterns tracked: {cms.total_count}")
    print(f"Memory used: ~{3 * 100 * 5 * 4 / 1024:.1f} KB (width×depth×4 bytes)")

    # Analysis
    future_count = cms.query("FUTURE:")
    todo_count = cms.query("TODO:")
    speculation_count = cms.query("will be")

    print("\n" + "-" * 70)
    print("Audit insights:")
    print("-" * 70)

    if future_count > todo_count:
        print(f"⚠️  WARNING: More FUTURE: markers ({future_count}) than TODO: ({todo_count})")
        print("   This suggests promises without concrete actions.")

    if speculation_count > todo_count:
        print(f"⚠️  WARNING: High speculation ('will be': {speculation_count}) vs actions (TODO: {todo_count})")
        print("   Consider converting speculation into actionable tasks.")


if __name__ == "__main__":
    visualize_hash_distribution()
    visualize_collision_and_minimum()
    visualize_merge_operation()
    visualize_accuracy_vs_width()
    visualize_streaming_scenario()

    print("\n" + "=" * 70)
    print("END OF VISUALIZATION")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Double hashing creates different hash functions for each row")
    print("2. Minimum across rows reduces collision impact")
    print("3. Larger width → less collision → better accuracy")
    print("4. Merge enables distributed counting")
    print("5. Sub-linear space makes it practical for streaming large datasets")
    print("=" * 70)
