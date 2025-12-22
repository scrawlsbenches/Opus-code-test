# Survey Results: Developer Feedback on Cortical Text Processor

**Date:** 2025-12-15
**Respondents:** 5 sub-agent developers with different use cases
**Method:** Each agent reviewed README.md, docs/, and CLAUDE.md before responding

---

## Respondent Profiles

| # | Use Case | Project Description |
|---|----------|---------------------|
| 1 | RAG Developer | Documentation chatbot for 500-1000 pages |
| 2 | Code Search | Semantic search for 500K+ line Python monorepo |
| 3 | Knowledge Graph | Auto-extract relationships from technical docs |
| 4 | Conversational Memory | Long-term memory for AI agent across sessions |
| 5 | Content Recommender | Related article suggestions for learning platform |

---

## Key Themes

### Theme 1: The "Neocortex" Metaphor Hurts More Than It Helps

**4 of 5 respondents** found the biological framing confusing or off-putting:

> "The biological metaphors are confusing in practice. 'Lateral connections', 'Hebbian learning', 'minicolumns' are evocative but don't map to implementation." — RAG Developer

> "Lead with value, not metaphor... The 'Hebbian learning' explanation for co-occurrence is backward." — Code Search Developer

> "Stop with the 'mimics the neocortex' angle and start with 'Here's how we get better recommendations than keyword matching.'" — Content Recommender

**Recommendation:** Rename concepts to standard IR terminology in user-facing docs. Keep biological metaphors for blog posts or "design philosophy" sections only.

---

### Theme 2: Missing Performance Benchmarks

**All 5 respondents** asked for performance data before considering adoption:

| What They Want | Current State |
|----------------|---------------|
| Indexing time at 1K, 10K, 100K docs | Only shows 92 docs |
| Query latency (p50, p99) | Not documented |
| Memory footprint per document | Not documented |
| Comparison to grep/Elasticsearch/embeddings | Not provided |

> "I'd trust 'searches 10K files in <500ms' more than '337 passing tests.'" — Code Search Developer

> "Would try if: You published a conversational memory example and performance benchmarks on 10K+ messages showing <100ms retrieval." — Memory Developer

**Recommendation:** Create `docs/benchmarks.md` with real numbers on realistic corpora.

---

### Theme 3: Passage Retrieval is the Killer Feature (But Underdocumented)

**4 of 5 respondents** identified `find_passages_for_query()` as the most valuable feature for their use case, but found documentation lacking:

> "The README says 'RAG System Support: Chunk-level passage retrieval' but it's unclear: What's the minimum context window size? How does passage ranking differ from document ranking?" — RAG Developer

> "If the library could guarantee sub-100ms retrieval from 50K messages with sensible defaults for conversational content, that alone would be worth adopting." — Memory Developer

**What's Missing:**
- End-to-end RAG example with LLM integration
- Chunk quality validation (are passages semantically coherent?)
- Performance at scale (100K+ passages)
- Passage-level metadata for linking back to source

**Recommendation:** Add `docs/rag-integration.md` with working LLM example code.

---

### Theme 4: No Clear "When to Use This vs. Embeddings" Positioning

**3 of 5 respondents** explicitly asked why they should use this over transformer embeddings:

> "Why should I use this instead of sentence-transformers + cosine similarity?" — Content Recommender

> "Embeddings (OpenAI, HuggingFace) are faster, more proven, work with any language. When would I pick this over embeddings? The docs don't say." — RAG Developer

**What Developers Need to Know:**
- Zero dependencies = easier deployment, no CUDA/PyTorch issues
- Incremental updates without re-embedding (unique advantage)
- Transparent graph-based approach vs. neural black box
- Trade-off: No semantic understanding of meaning, only co-occurrence

**Recommendation:** Add honest comparison table in README: "When to use Cortical vs. embeddings"

---

### Theme 5: Incremental Updates Are Valuable But Poorly Explained

**3 of 5 respondents** identified incremental indexing as a key differentiator, but found docs insufficient:

> "The docs mention `add_document_incremental()` with a `recompute` parameter, but: What's the default behavior? How stale does data get? Does incremental work for document removal?" — RAG Developer

> "Adding 50 new articles means rebuilding our entire search index. We need incremental indexing that doesn't break existing recommendations." — Content Recommender

**Recommendation:** Document incremental update guarantees clearly, with performance comparisons.

---

### Theme 6: Configuration is a Black Box

**All 5 respondents** expressed confusion about tuning parameters:

| Parameter | Question Asked |
|-----------|----------------|
| `cluster_strictness` | What's the default? What values for code vs. docs? |
| `bridge_weight` | How does this affect recommendation quality? |
| `max_bigrams_per_term` | When should I change this? |
| `connection_strategy` | Which one for my use case? |

> "There's no decision tree or heuristics for choosing connection strategy." — Knowledge Graph Developer

**Recommendation:** Create `docs/tuning-guide.md` with use-case-specific recommendations.

---

## Feature Requests (Ranked by Frequency)

| Feature | Requested By | Priority |
|---------|--------------|----------|
| Performance benchmarks | All 5 | **Critical** |
| RAG/LLM integration example | 4 of 5 | **High** |
| Parameter tuning guide | All 5 | **High** |
| "When to use this" positioning | 3 of 5 | **High** |
| Incremental update documentation | 3 of 5 | **Medium** |
| Failure case examples | 3 of 5 | **Medium** |
| Temporal/recency weighting | 2 of 5 | Low |
| Bidirectional graph queries | 1 of 5 | Low |

---

## Specific Documentation Gaps Identified

### Missing Examples
- [ ] End-to-end RAG with LLM (Claude/OpenAI)
- [ ] Code search on real monorepo
- [ ] Conversational memory system
- [ ] Content recommendation pipeline
- [ ] Knowledge graph extraction

### Missing Benchmarks
- [ ] Indexing time at various corpus sizes
- [ ] Query latency distribution
- [ ] Memory footprint scaling
- [ ] Comparison to baselines (grep, Elasticsearch, embeddings)

### Missing Guides
- [ ] Parameter tuning by use case
- [ ] When to use which connection strategy
- [ ] Incremental vs. full recomputation trade-offs
- [ ] Known limitations and failure modes

---

## Positive Feedback (What's Working)

Despite concerns, respondents identified genuine strengths:

| Strength | Mentioned By |
|----------|--------------|
| Zero dependencies (deployment simplicity) | All 5 |
| Well-engineered, solid test coverage | 4 of 5 |
| Incremental updates (unique vs. embeddings) | 3 of 5 |
| Transparent graph-based approach | 2 of 5 |
| Thorough CLAUDE.md development guide | 2 of 5 |
| Query expansion via co-occurrence | 2 of 5 |

> "Zero dependencies is genuinely valuable for deployment (fewer supply-chain risks, faster cold start)." — RAG Developer

> "The CLAUDE.md file is excellently detailed... That says something about your engineering culture." — Content Recommender

---

## Recommended Action Items

### Immediate (Week 1)
1. **Create performance benchmark doc** with real numbers
2. **Add RAG integration example** with working LLM code
3. **Write "When to use this" section** in README

### Short-term (Week 2-3)
4. **Create parameter tuning guide** by use case
5. **Document incremental update guarantees**
6. **Add failure case examples** (honest limitations)

### Medium-term (Month 1)
7. **Reduce biological jargon** in user-facing docs
8. **Add use-case-specific tutorials** (code search, recommendations, etc.)
9. **Publish comparison benchmarks** vs. alternatives

---

## Raw Survey Responses

Full responses from each sub-agent are preserved in the survey collection system for reference.

---

*Survey conducted 2025-12-15 using 5 sub-agents with different development contexts.*
