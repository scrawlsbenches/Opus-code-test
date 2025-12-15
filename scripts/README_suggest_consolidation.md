# Memory Consolidation Suggestions

The `suggest_consolidation.py` script analyzes memory documents and suggests consolidation opportunities.

## Features

1. **Cluster Analysis**: Groups similar memories using semantic similarity and suggests creating concept documents
2. **High Overlap Detection**: Finds memory pairs with significant term overlap that could be merged
3. **Old Memory Tracking**: Identifies unconsolidated memories that are aging and should be reviewed
4. **Topic Extraction**: Automatically suggests concept names based on key terms from clusters

## Usage

### Basic Usage

```bash
# Default analysis (corpus_dev.pkl)
python scripts/suggest_consolidation.py

# Specify corpus file
python scripts/suggest_consolidation.py --corpus my_corpus.pkl

# Verbose output with details
python scripts/suggest_consolidation.py --verbose
```

### Advanced Options

```bash
# Higher similarity threshold (more strict)
python scripts/suggest_consolidation.py --threshold 0.7

# Require at least 3 memories per cluster
python scripts/suggest_consolidation.py --min-cluster 3

# Only flag memories older than 60 days
python scripts/suggest_consolidation.py --min-age-days 60

# Adjust clustering resolution (higher = more clusters)
python scripts/suggest_consolidation.py --resolution 1.5

# JSON output for programmatic use
python scripts/suggest_consolidation.py --output json
```

### Combined Options

```bash
# Detailed analysis with custom thresholds
python scripts/suggest_consolidation.py \
  --threshold 0.6 \
  --min-cluster 2 \
  --min-age-days 45 \
  --verbose
```

## Output Format

### Text Output (Default)

The script outputs three types of suggestions:

1. **Cluster Suggestions**: Groups of related memories that should be consolidated
   ```
   [1] These 3 memories discuss 'security-testing-fuzzing'.
       Consider creating samples/memories/concept-security-testing-fuzzing.md
   ```

2. **High Overlap Pairs**: Memory pairs with significant similarity
   ```
   [1] memory-1.md and memory-2.md have 78.5% overlap (shared: security, fuzzing, validation).
       Consider merging?
   ```

3. **Old Memories**: Unconsolidated memories past the age threshold
   ```
   [1] 2025-10-15-old-topic.md is 60 days old.
       Consider consolidating into a concept document?
   ```

### JSON Output

```bash
python scripts/suggest_consolidation.py --output json
```

Returns structured JSON with:
- `clusters`: Array of cluster suggestions with topics and document lists
- `similar_pairs`: Array of high-overlap pairs with similarity scores
- `old_memories`: Array of old memory entries with ages
- `stats`: Summary statistics

Example:
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "document_count": 3,
      "documents": ["samples/memories/2025-12-01-topic.md", ...],
      "suggested_concept": "security-testing-fuzzing",
      "topics": [["security", 0.85], ["testing", 0.67], ...],
      "message": "These 3 memories discuss ..."
    }
  ],
  "similar_pairs": [...],
  "old_memories": [...],
  "stats": {
    "total_memories": 15,
    "total_concepts": 3,
    "analyzed_memories": 12
  }
}
```

## Algorithm Details

### Clustering

The script uses fingerprint-based similarity clustering:
1. Computes semantic fingerprints for all memory documents
2. Calculates pairwise similarity using term overlap
3. Builds a similarity graph with threshold-based edges
4. Finds connected components (clusters) using depth-first search

The `--resolution` parameter adjusts the similarity threshold:
- `resolution = 1.0`: threshold = 0.3 (default)
- `resolution = 2.0`: threshold = 0.6 (stricter, more clusters)
- `resolution = 0.5`: threshold = 0.15 (looser, fewer clusters)

### Topic Extraction

For each cluster, the script:
1. Aggregates term weights across all documents
2. Weights by document PageRank importance
3. Considers global term importance
4. Extracts top 5 terms as representative topics
5. Suggests concept name as hyphenated combination of top 3 terms

### Similarity Calculation

Uses the processor's fingerprint comparison:
- Extracts top 20 terms from each document
- Computes cosine similarity between fingerprints
- Includes shared term analysis
- Considers both term weights and overlap

## Integration with Workflow

### Typical Workflow

1. **Regularly run analysis**:
   ```bash
   python scripts/suggest_consolidation.py --min-age-days 30
   ```

2. **Review cluster suggestions**: Create concept documents for major topics

3. **Merge high-overlap pairs**: Consolidate redundant memories

4. **Archive old memories**: Integrate learnings into concept docs

### Creating Concept Documents

When the script suggests creating a concept document:

```bash
# Script output:
# [1] These 3 memories discuss 'security-testing-fuzzing'.
#     Consider creating samples/memories/concept-security-testing-fuzzing.md

# Create the concept document
cat > samples/memories/concept-security-testing-fuzzing.md << 'EOF'
# Concept: Security Testing and Fuzzing

**Tags:** `security`, `testing`, `fuzzing`
**Related:** [[2025-12-01-topic.md]], [[2025-12-05-topic.md]]

## Overview

Consolidated learnings from multiple sessions on security testing...

## Key Insights

1. **Fuzzing finds edge cases**: Property-based testing...
2. **Validation is critical**: NaN and infinity...

## Patterns

- Always validate numeric inputs for NaN/inf
- Use Hypothesis for property-based testing
- Test with extreme values
EOF
```

## Command Reference

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--corpus` | `-c` | string | `corpus_dev.pkl` | Path to corpus file |
| `--threshold` | `-t` | float | `0.5` | Min similarity for pair suggestions (0.0-1.0) |
| `--min-cluster` | | int | `2` | Min memories per cluster |
| `--min-age-days` | | int | `30` | Min age for old memory warnings |
| `--resolution` | | float | `1.0` | Clustering resolution (higher = more clusters) |
| `--output` | `-o` | choice | `text` | Output format: `text` or `json` |
| `--verbose` | `-v` | flag | `false` | Detailed output with document lists |

## Examples

### Find duplicate content

```bash
# High threshold to find near-duplicates
python scripts/suggest_consolidation.py --threshold 0.8
```

### Review all memories quarterly

```bash
# Find memories older than 90 days
python scripts/suggest_consolidation.py --min-age-days 90
```

### Generate concept document candidates

```bash
# Loose clustering to find broad topic groups
python scripts/suggest_consolidation.py --resolution 0.5 --min-cluster 3
```

### Export for automation

```bash
# JSON output for scripting
python scripts/suggest_consolidation.py --output json > suggestions.json

# Process with jq
cat suggestions.json | jq '.clusters[] | .suggested_concept'
```

## Testing

Unit tests are in `tests/unit/test_suggest_consolidation.py`:

```bash
# Run tests
python -m unittest tests.unit.test_suggest_consolidation -v

# Or with pytest
pytest tests/unit/test_suggest_consolidation.py -v
```

Tests cover:
- Date parsing for various formats
- Memory age calculation
- Concept document detection
- Suggestion generation
- Edge cases (empty corpus, single memory, etc.)

## Limitations

- **Small corpus only**: Designed for personal memory management (10-100 memories)
- **No automatic merging**: Suggestions must be manually reviewed and actioned
- **Static analysis**: Does not consider semantic relationships or context
- **Filename-based dating**: Relies on consistent filename format

## Future Enhancements

Potential improvements (see task system):
- Interactive mode to act on suggestions
- Automatic concept document generation
- Timeline visualization of memory topics
- Semantic relation analysis between memories
- Cross-reference to decision records (ADRs)
