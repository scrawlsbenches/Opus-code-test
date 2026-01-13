# Design Decisions and Rationale

## Why These Choices Were Made

This document explains the reasoning behind key design decisions in the Cognitive Agent. When you're confused about why something works a certain way, check here first.

## IDF Weighting: Why It Matters

**Problem**: Common words like "the", "and", "is" appear everywhere but carry little meaning. Without weighting, these dominate associations.

**Solution**: IDF (Inverse Document Frequency) weighting.

**Formula**: `idf(word) = log((total_docs + 1) / (doc_frequency + 1))`

**Effect**:
- Rare words get high IDF (more meaningful)
- Common words get low IDF (less meaningful)
- "authentication" has higher weight than "the"

**Why smoothed formula?** Adding +1 prevents division by zero and log(0).

## Dual Value Storage: Raw vs IDF Strength

**Problem**: Sometimes you want raw co-occurrence counts, sometimes weighted.

**Solution**: Store both values on every link:
- `raw_strength`: Pure co-occurrence frequency
- `idf_strength`: Weighted by term rarity

**When to use which**:
- Raw: Analyzing what literally appears together
- IDF: Finding meaningful semantic relationships

## Incremental Training: Why Not Rebuild?

**Problem**: Full retraining takes too long as corpus grows.

**Solution**: Incremental training that:
- Only processes new/changed documents
- Preserves existing graph structure
- Tracks staleness for IDF refresh

**Trade-off**: IDF values become stale as vocabulary changes. Solution: periodic reindexing.

## Staleness Tracking: The 20% Threshold

**Problem**: How do you know when IDF weights need refreshing?

**Solution**: Track staleness as percentage growth since last reindex.

**Why 20%?** Empirical balance between accuracy and reindex cost. Below 20%, weights are close enough. Above 20%, associations may be skewed.

**Formula**: `staleness = (current_docs - last_reindex_docs) / last_reindex_docs`

## O(1) Indexes: _incoming and _outgoing

**Problem**: Finding links to/from an atom required scanning all links.

**Solution**: Maintain indexes:
- `_incoming[atom_id]` → links pointing TO this atom
- `_outgoing[atom_id]` → links originating FROM this atom

**Performance impact**:
- Before: O(n) scan of 248k links = 55-70ms
- After: O(1) lookup = <1ms

**Why both indexes?** Different queries need different directions:
- "What follows this word?" → _outgoing
- "What precedes this word?" → _incoming

## Sharded Storage: Git-Friendly Persistence

**Problem**: One big JSON file causes:
- Git merge conflicts
- GitHub size limits (50MB)
- Full rewrite on every save

**Solution**: Shard by atom type:
- `atoms_word.json` - vocabulary
- `atoms_similarity_00.json` through `_03.json` - subdivided large type
- `atoms_follows_00.json` through `_03.json` - subdivided large type

**Why subdivide large types?** SIMILARITY and FOLLOWS have 200k+ atoms each. Subdividing keeps each file under 20MB.

## Dirty Tracking: Incremental Saves

**Problem**: Saving 500k atoms takes 54 seconds even for small changes.

**Solution**: Track which atoms changed since last save:
- `_dirty_atoms` set tracks modified atom IDs
- Only rewrite shards containing dirty atoms
- Clear dirty state after successful save

**Performance impact**:
- No changes: 0.05s (was 54s)
- Small changes: ~10s (was 54s)

## FOLLOWS vs SIMILARITY Links

**FOLLOWS links** are directional:
- Structure: `[from_word, to_word]`
- Meaning: "to_word often follows from_word"
- Use: Next-word prediction

**SIMILARITY links** are bidirectional:
- Structure: `[word_a, word_b]` (order doesn't matter)
- Meaning: "these words co-occur in documents"
- Use: Finding related concepts

## Why Hypergraph, Not Regular Graph?

**Regular graph**: Nodes and edges. Edges connect nodes.

**Hypergraph**: Everything is an atom. Links ARE atoms that can be linked to.

**Why this matters**: Enables meta-reasoning:
- "This link is strong" (belief about a relationship)
- "Evidence X supports conclusion Y" (links about links)
- Future: "Agent A believes link L" (attribution)

## Container and Dependency Injection

**Problem**: Hardcoded dependencies make testing hard.

**Solution**: All components receive dependencies through constructor injection. A Container manages wiring.

**Why it matters**:
- Test isolation (inject mocks)
- Configuration flexibility
- Clear dependency graph
