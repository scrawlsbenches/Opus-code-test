# Louvain Resolution Parameter Analysis

**Task #126 Research Report**

**Date:** 2025-12-11
**Author:** Research Agent
**Corpus:** 103 documents, 7,102 tokens

---

## Executive Summary

The Louvain algorithm's `resolution` parameter significantly affects clustering granularity. After systematic testing across 11 resolution values (0.5 to 3.0), we conclude:

1. **The current default of 1.0 is well-chosen** for general-purpose use
2. Resolution values from **0.9 to 1.5** all produce good results
3. All tested resolutions maintain modularity > 0.3 (good community structure)
4. Resolution should be tunable based on use case:
   - Coarse topics: 0.7-0.9
   - General purpose: 1.0 (default)
   - Fine-grained: 1.5-2.0
   - Ultra-specific: 3.0+

**Recommendation:** Keep default at 1.0, document resolution parameter for advanced users.

---

## Results Summary

| Resolution | Clusters | Max % | Avg Size | Modularity | Balance (Gini) | Coherence |
|------------|----------|-------|----------|------------|----------------|-----------|
| 0.50 | 38 | 64.4% | 186.9 | **0.7804** | 0.886 | 0.020 |
| 0.60 | 28 | 25.4% | 253.6 | 0.5217 | 0.765 | 0.010 |
| 0.70 | 33 | 21.6% | 215.2 | 0.5084 | 0.733 | 0.010 |
| 0.80 | 27 | 18.4% | 263.0 | 0.4753 | 0.586 | 0.011 |
| 0.90 | 28 | 12.2% | 253.6 | 0.4261 | 0.438 | 0.013 |
| **1.00** | **32** | **9.5%** | **221.9** | **0.4036** | **0.386** | **0.015** |
| 1.10 | 41 | 9.3% | 173.2 | 0.3885 | 0.399 | 0.023 |
| 1.20 | 44 | 8.0% | 161.4 | 0.3736 | 0.358 | 0.025 |
| 1.50 | 56 | 5.3% | 126.8 | 0.3467 | 0.282 | 0.032 |
| 2.00 | 79 | 4.2% | 89.9 | 0.3305 | 0.281 | 0.039 |
| 3.00 | 125 | 2.5% | 56.8 | 0.3064 | 0.277 | 0.050 |

---

## Metric Interpretation

### Modularity (Q)
- Measures density of connections within clusters vs between clusters
- **Q > 0.3**: Good community structure
- **Q > 0.5**: Strong community structure
- **All tested resolutions exceed the 0.3 threshold**

### Balance (Gini Coefficient)
- Measures how evenly sized clusters are
- **0 = perfectly balanced**, 1 = all tokens in one cluster
- Lower is better for even distribution

### Max Cluster %
- Percentage of total tokens in the largest cluster
- **< 50% is critical** to avoid mega-clusters that defeat clustering purpose
- **< 20% is ideal** for meaningful topic separation

### Coherence
- Measures intra-cluster connectivity
- Higher indicates tighter semantic grouping
- Increases with smaller clusters (higher resolution)

---

## Key Findings

### 1. Modularity vs. Cluster Size Trade-off

There is a clear trade-off between modularity score and cluster size distribution:

- **Low resolution (0.5)**: Highest modularity (0.78) but creates a 64% mega-cluster
- **Default resolution (1.0)**: Good modularity (0.40) with max cluster only 9.5%
- **High resolution (3.0)**: Lower modularity (0.31) but excellent balance (2.5% max)

The mathematically "best" modularity at low resolution is misleading because it concentrates most tokens in one cluster, which is semantically useless.

### 2. Resolution 1.0 is the Inflection Point

Looking at the data, resolution 1.0 is where the curves stabilize:

- Max cluster drops below 10% (from 64% at res=0.5)
- Balance improves significantly (0.386 vs 0.886)
- Modularity remains good (0.40 > 0.3 threshold)

### 3. Cluster Quality at Different Resolutions

**Resolution 0.5 (too coarse):**
```
Cluster #1 (4574 tokens): data, patterns, systems, code, knowledge, multiple
```
This cluster contains 64% of all tokens and mixes unrelated concepts.

**Resolution 1.0 (recommended default):**
```
Cluster #1 (672 tokens): properties, springs, cells, flow, energy, generation
Cluster #2 (660 tokens): tests, changes, system, behavior, prevents, problems
Cluster #3 (654 tokens): knowledge, concepts, structure, pagerank, graph
```
Clusters are semantically coherent with clear topic boundaries.

**Resolution 3.0 (fine-grained):**
```
Cluster #1 (181 tokens): self, test, content, record, results, def
Cluster #2 (165 tokens): fermentation, processes, activity, organic, material
Cluster #3 (150 tokens): springs, energy, storage, power, mechanical, lithium
```
Very specific clusters, good for detailed analysis but may be too granular for general use.

### 4. All Resolutions Maintain Good Structure

Importantly, **all tested resolutions maintain modularity > 0.3**, meaning the Louvain algorithm produces good community structure regardless of resolution. The resolution parameter primarily controls granularity, not quality.

---

## Use Case Recommendations

| Use Case | Resolution | Clusters | Notes |
|----------|------------|----------|-------|
| Coarse topic grouping | 0.7-0.9 | ~30 | Larger but distinct topics |
| **General purpose (default)** | **1.0** | **~32** | **Balanced trade-off** |
| Fine-grained topics | 1.5-2.0 | 56-79 | More specific groupings |
| Detailed analysis | 3.0+ | 100+ | Very specific clusters |

---

## Auto-Tuning Considerations

### Heuristic for Auto-Selection

A potential auto-tuning heuristic based on corpus characteristics:

```python
def suggest_resolution(processor):
    """Suggest resolution based on corpus characteristics."""
    layer0 = processor.layers[CorticalLayer.TOKENS]

    # Compute average connections per token
    total_connections = sum(
        len(col.lateral_connections)
        for col in layer0.minicolumns.values()
    )
    avg_connections = total_connections / layer0.column_count()

    # Dense graphs (many connections) → higher resolution
    # Sparse graphs (few connections) → lower resolution
    if avg_connections > 20:
        return 1.5  # Dense: finer clusters
    elif avg_connections > 10:
        return 1.0  # Moderate: default
    else:
        return 0.8  # Sparse: coarser clusters
```

### Recommendation

Auto-tuning adds complexity for marginal benefit. Since all resolutions produce good modularity, keeping a fixed default of 1.0 with documented tuning options is preferable.

---

## Final Recommendation

### Keep Default Resolution at 1.0

The current default of `resolution=1.0` is well-chosen because:

1. **Modularity 0.40** exceeds the 0.3 "good structure" threshold
2. **Max cluster 9.5%** prevents mega-clusters
3. **Balance 0.386** provides reasonable distribution
4. **Semantic coherence** produces meaningful topic groupings
5. **Standard interpretation** - resolution 1.0 is the standard Louvain default

### Document for Advanced Users

Add documentation explaining:
- Higher resolution (>1.0) → more, smaller clusters
- Lower resolution (<1.0) → fewer, larger clusters
- All values 0.5-3.0 produce valid community structure

### No Code Changes Required

The existing default values in `cortical/analysis.py:cluster_by_louvain()` and `cortical/processor/compute.py:build_concept_clusters()` should remain at `resolution=1.0`.

---

## Appendix: Reproducing This Analysis

```bash
# Run the analysis script
python scripts/analyze_louvain_resolution.py --verbose

# Test specific resolutions
python scripts/analyze_louvain_resolution.py -r 0.5 1.0 2.0

# Generate markdown report
python scripts/analyze_louvain_resolution.py -o docs/louvain_resolution_analysis.md
```

---

*Analysis completed for Task #126*
