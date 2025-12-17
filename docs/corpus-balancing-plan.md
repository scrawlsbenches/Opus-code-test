# Corpus Balancing Plan: Building a Mixture of Experts Training Corpus

## Executive Summary

This document outlines the strategy for transforming the current code-dominated corpus into a balanced, diverse corpus capable of training ~100 specialized micro-models that can eventually form a Mixture of Experts (MoE) system.

## Current State Analysis

### Corpus Composition (212 documents)
| Source | Documents | % of Corpus | Issue |
|--------|-----------|-------------|-------|
| Python source files | ~80 | 38% | Dominates with test patterns |
| Test files | ~40 | 19% | `assert`, `mock`, `test` vocabulary |
| Sample documents | 179 | 43% | Imbalanced across domains |

### Sample Distribution (179 docs)
| Domain | Count | % | Status |
|--------|-------|---|--------|
| Finance/Trading | 31 | 17.3% | **OVERREPRESENTED** |
| NLP/IR | 18 | 10.1% | Project-specific bias |
| Software Engineering | 17 | 9.5% | Project-specific bias |
| Traditional Crafts | 14 | 7.8% | Good diversity |
| Computing Fundamentals | 12 | 6.7% | OK |
| Security | 11 | 6.1% | OK |
| ML/AI | 9 | 5.0% | Needs depth |
| Neuroscience/Cognitive | 8 | 4.5% | Needs depth |
| Science/Nature | 8 | 4.5% | **SEVERELY UNDERREPRESENTED** |
| Customer Service | 6 | 3.4% | Needs depth |
| Law | 2 | 1.1% | **SEVERELY UNDERREPRESENTED** |

### PageRank Analysis (Top Terms)
```
1. assert (0.0191) - TEST CODE
2. test (0.0178) - TEST CODE
3. processor (0.0147) - PROJECT INTERNAL
4. tokenizer (0.0067) - PROJECT INTERNAL
5. query (0.0065) - PROJECT INTERNAL
```

**Diagnosis**: The corpus is dominated by test infrastructure and project internals, not by domain knowledge.

---

## Target Architecture: Mixture of Experts

### Why MoE?
1. **Sparse Activation**: Only relevant experts process each query
2. **Specialization**: Small models excel at narrow domains
3. **Scalability**: Add new experts without retraining others
4. **Efficiency**: Better than one giant generalist model

### Training Requirements per Expert
| Metric | Minimum | Optimal |
|--------|---------|---------|
| Documents per domain | 20 | 50-100 |
| Unique terms per domain | 500 | 2000+ |
| Average doc length | 500 words | 1000+ words |
| Topic coherence | 0.4+ | 0.6+ |

### Target Corpus Size
- **Domains**: 80-100 distinct topics
- **Documents per domain**: 25-50 average
- **Total sample documents**: 2,000-5,000
- **Balance ratio**: Each domain = 0.5-2% of corpus

---

## Proposed Topic Taxonomy (96 Domains)

### 1. SCIENCES (16 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Physics - Mechanics | 1 | 30 | 29 |
| Physics - Quantum | 1 | 30 | 29 |
| Chemistry - Organic | 0 | 30 | 30 |
| Chemistry - Materials | 1 | 30 | 29 |
| Biology - Cellular | 1 | 30 | 29 |
| Biology - Ecology | 2 | 30 | 28 |
| Biology - Genetics | 0 | 30 | 30 |
| Astronomy | 1 | 30 | 29 |
| Geology | 1 | 30 | 29 |
| Meteorology | 0 | 30 | 30 |
| Oceanography | 0 | 30 | 30 |
| Pharmacology | 1 | 30 | 29 |
| Neuroscience | 5 | 30 | 25 |
| Psychology | 2 | 30 | 28 |
| Medicine - Clinical | 1 | 30 | 29 |
| Medicine - Surgery | 0 | 30 | 30 |

**Gap: ~465 documents needed**

### 2. ENGINEERING (12 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Mechanical Engineering | 2 | 30 | 28 |
| Electrical Engineering | 2 | 30 | 28 |
| Civil Engineering | 0 | 30 | 30 |
| Chemical Engineering | 0 | 30 | 30 |
| Aerospace Engineering | 0 | 30 | 30 |
| Robotics | 0 | 30 | 30 |
| Manufacturing | 0 | 30 | 30 |
| Quality Control | 0 | 30 | 30 |
| Industrial Design | 0 | 30 | 30 |
| Systems Engineering | 3 | 30 | 27 |
| Environmental Engineering | 1 | 30 | 29 |
| Biomedical Engineering | 0 | 30 | 30 |

**Gap: ~352 documents needed**

### 3. COMPUTING & TECHNOLOGY (12 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Algorithms & Data Structures | 4 | 30 | 26 |
| Operating Systems | 3 | 30 | 27 |
| Networking | 4 | 30 | 26 |
| Databases | 3 | 30 | 27 |
| Distributed Systems | 3 | 30 | 27 |
| Computer Graphics | 0 | 30 | 30 |
| Human-Computer Interaction | 0 | 30 | 30 |
| Cybersecurity | 11 | 30 | 19 |
| Cloud Computing | 2 | 30 | 28 |
| DevOps & SRE | 5 | 30 | 25 |
| Embedded Systems | 0 | 30 | 30 |
| Quantum Computing | 1 | 30 | 29 |

**Gap: ~324 documents needed**

### 4. ARTIFICIAL INTELLIGENCE (10 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Machine Learning Fundamentals | 5 | 30 | 25 |
| Deep Learning | 5 | 30 | 25 |
| Natural Language Processing | 10 | 30 | 20 |
| Computer Vision | 0 | 30 | 30 |
| Reinforcement Learning | 2 | 30 | 28 |
| Knowledge Graphs | 5 | 30 | 25 |
| Expert Systems | 0 | 30 | 30 |
| Neural Architecture | 4 | 30 | 26 |
| AI Ethics & Safety | 0 | 30 | 30 |
| Generative AI | 0 | 30 | 30 |

**Gap: ~269 documents needed**

### 5. BUSINESS & FINANCE (10 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Accounting | 0 | 30 | 30 |
| Financial Markets | 15 | 30 | 15 |
| Investment Banking | 2 | 30 | 28 |
| Corporate Finance | 0 | 30 | 30 |
| Risk Management | 8 | 30 | 22 |
| Marketing | 3 | 30 | 27 |
| Operations Management | 2 | 30 | 28 |
| Supply Chain | 2 | 30 | 28 |
| Human Resources | 0 | 30 | 30 |
| Project Management | 0 | 30 | 30 |

**Gap: ~268 documents needed**

### 6. LAW & GOVERNANCE (8 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Contract Law | 1 | 30 | 29 |
| Intellectual Property | 1 | 30 | 29 |
| Criminal Law | 0 | 30 | 30 |
| Corporate Law | 0 | 30 | 30 |
| International Law | 0 | 30 | 30 |
| Regulatory Compliance | 2 | 30 | 28 |
| Constitutional Law | 0 | 30 | 30 |
| Privacy Law | 0 | 30 | 30 |

**Gap: ~236 documents needed**

### 7. HUMANITIES (8 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Philosophy | 0 | 30 | 30 |
| History | 2 | 30 | 28 |
| Literature | 1 | 30 | 29 |
| Linguistics | 2 | 30 | 28 |
| Anthropology | 0 | 30 | 30 |
| Archaeology | 0 | 30 | 30 |
| Religious Studies | 0 | 30 | 30 |
| Art History | 0 | 30 | 30 |

**Gap: ~235 documents needed**

### 8. CREATIVE & PERFORMING ARTS (8 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Visual Arts | 1 | 30 | 29 |
| Music Theory & Composition | 1 | 30 | 29 |
| Film & Cinema | 0 | 30 | 30 |
| Theater & Drama | 0 | 30 | 30 |
| Architecture | 0 | 30 | 30 |
| Photography | 0 | 30 | 30 |
| Graphic Design | 1 | 30 | 29 |
| Creative Writing | 1 | 30 | 29 |

**Gap: ~236 documents needed**

### 9. PRACTICAL SKILLS & TRADES (8 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Culinary Arts | 5 | 30 | 25 |
| Agriculture & Horticulture | 3 | 30 | 27 |
| Woodworking | 1 | 30 | 29 |
| Metalworking | 0 | 30 | 30 |
| Textiles & Fashion | 0 | 30 | 30 |
| Construction | 0 | 30 | 30 |
| Automotive | 0 | 30 | 30 |
| Plumbing & Electrical | 0 | 30 | 30 |

**Gap: ~231 documents needed**

### 10. SPORTS & RECREATION (8 domains)
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Team Sports | 1 | 30 | 29 |
| Individual Sports | 2 | 30 | 28 |
| Martial Arts | 1 | 30 | 29 |
| Outdoor Recreation | 1 | 30 | 29 |
| Board Games & Strategy | 1 | 30 | 29 |
| Fitness & Training | 0 | 30 | 30 |
| Extreme Sports | 0 | 30 | 30 |
| Esports | 0 | 30 | 30 |

**Gap: ~234 documents needed**

---

## Total Gap Analysis

| Category | Current Docs | Target Docs | Gap |
|----------|-------------|-------------|-----|
| Sciences | ~16 | 480 | 464 |
| Engineering | ~8 | 360 | 352 |
| Computing | ~36 | 360 | 324 |
| AI/ML | ~31 | 300 | 269 |
| Business/Finance | ~32 | 300 | 268 |
| Law | ~4 | 240 | 236 |
| Humanities | ~5 | 240 | 235 |
| Arts | ~4 | 240 | 236 |
| Practical Skills | ~9 | 240 | 231 |
| Sports | ~6 | 240 | 234 |
| **TOTAL** | **~151** | **3,000** | **~2,849** |

**We need ~2,850 new sample documents to achieve balance.**

---

## Document Generation Strategy

### Phase 1: Quick Wins (500 docs)
Focus on domains that are easiest to generate and most distinct from code:

1. **Humanities & Arts** (200 docs)
   - Philosophy excerpts
   - Historical narratives
   - Literary analysis
   - Art technique guides

2. **Practical Skills** (150 docs)
   - Recipe collections
   - Craft tutorials
   - Trade procedures
   - Hobby guides

3. **Sports & Recreation** (150 docs)
   - Rules and strategies
   - Training protocols
   - Equipment guides
   - Competition formats

### Phase 2: Professional Domains (1000 docs)
Requires more structured, technical content:

1. **Law** (200 docs)
   - Case summaries
   - Legal procedures
   - Regulatory frameworks
   - Contract templates

2. **Business** (200 docs)
   - Business processes
   - Financial reports
   - Marketing strategies
   - HR procedures

3. **Sciences** (300 docs)
   - Research summaries
   - Lab procedures
   - Scientific concepts
   - Experiment descriptions

4. **Engineering** (300 docs)
   - Design specifications
   - Process documentation
   - Safety procedures
   - Technical standards

### Phase 3: Technical Depth (1000 docs)
Build out computing and AI domains with depth:

1. **AI/ML** (300 docs)
   - Algorithm explanations
   - Model architectures
   - Training procedures
   - Evaluation methods

2. **Computing** (300 docs)
   - System designs
   - Protocol specifications
   - Architecture patterns
   - Performance tuning

3. **Cross-Domain** (400 docs)
   - Interdisciplinary topics
   - Application case studies
   - Integration patterns

### Phase 4: Edge Cases & Depth (500 docs)
Fill remaining gaps and add depth where needed.

---

## Document Templates

### Template A: Concept Explanation
```markdown
# [Concept Name]

## Overview
[2-3 paragraphs explaining the core concept]

## Key Principles
1. [Principle 1]
2. [Principle 2]
3. [Principle 3]

## Applications
- [Application 1]
- [Application 2]

## Related Concepts
- [Related 1]
- [Related 2]

## Further Reading
[References]
```

### Template B: Procedural Guide
```markdown
# How to [Task]

## Prerequisites
- [Prerequisite 1]
- [Prerequisite 2]

## Steps
1. [Step 1 with details]
2. [Step 2 with details]
3. [Step 3 with details]

## Common Mistakes
- [Mistake 1]: [How to avoid]
- [Mistake 2]: [How to avoid]

## Tips for Success
- [Tip 1]
- [Tip 2]
```

### Template C: Domain Glossary
```markdown
# [Domain] Terminology

## Core Terms
**[Term 1]**: [Definition]
**[Term 2]**: [Definition]

## Advanced Concepts
**[Concept 1]**: [Explanation]
**[Concept 2]**: [Explanation]

## Relationships
- [Term A] relates to [Term B] because [reason]
```

---

## Implementation Tasks

### Infrastructure
- [ ] Create `scripts/generate_sample_docs.py` - Document generator
- [ ] Create `scripts/analyze_corpus_balance.py` - Balance analyzer
- [ ] Create `scripts/validate_topic_coverage.py` - Coverage validator
- [ ] Update `scripts/index_codebase.py` - Add `--samples-only` flag

### Phase 1 Tasks (Week 1-2)
- [ ] Generate 50 Philosophy documents
- [ ] Generate 50 History documents
- [ ] Generate 50 Culinary documents
- [ ] Generate 50 Sports documents
- [ ] Generate 50 Crafts documents
- [ ] Generate 50 Music/Art documents
- [ ] Run balance analysis
- [ ] Adjust generation strategy

### Phase 2 Tasks (Week 3-4)
- [ ] Generate Law domain (200 docs)
- [ ] Generate Business domain (200 docs)
- [ ] Generate Sciences domain (300 docs)
- [ ] Generate Engineering domain (300 docs)
- [ ] Run full reindex
- [ ] Validate topic separation

### Phase 3 Tasks (Week 5-6)
- [ ] Generate AI/ML depth (300 docs)
- [ ] Generate Computing depth (300 docs)
- [ ] Generate cross-domain (400 docs)
- [ ] Train prototype experts

### Phase 4 Tasks (Week 7-8)
- [ ] Fill remaining gaps
- [ ] Quality validation
- [ ] Expert routing tests
- [ ] Full MoE prototype

---

## Quality Metrics

### Per-Domain Quality
| Metric | Target |
|--------|--------|
| Documents | 25-50 |
| Unique vocabulary | 500+ terms |
| Average document length | 800+ words |
| Intra-domain coherence | 0.5+ |
| Inter-domain separation | 0.3+ (different from other domains) |

### Corpus-Wide Quality
| Metric | Target |
|--------|--------|
| Total documents | 3,000+ |
| Domains represented | 80+ |
| Gini coefficient (balance) | < 0.3 |
| Concept cluster modularity | > 0.5 |
| PageRank diversity | Top 20 terms from 10+ domains |

### Expert Routing Quality
| Metric | Target |
|--------|--------|
| Routing accuracy | 85%+ queries to correct expert |
| Expert coverage | Each expert handles 1-5% of queries |
| Fallback rate | < 10% queries to generalist |

---

## Risk Mitigation

### Risk: Generated content lacks authenticity
**Mitigation**: Use diverse templates, vary structure, include domain-specific terminology patterns

### Risk: Topic overlap causes routing confusion
**Mitigation**: Define clear domain boundaries, use discriminative terms, test with adversarial queries

### Risk: Some domains too small to train
**Mitigation**: Set minimum 20 docs/domain threshold, merge related micro-domains if needed

### Risk: Indexing becomes too slow
**Mitigation**: Use `--samples-only` mode, incremental indexing, parallel processing

---

## Success Criteria

### Milestone 1: Balanced Sample Set
- [ ] 1,000+ sample documents
- [ ] 50+ domains with 15+ docs each
- [ ] No domain > 5% of total
- [ ] PageRank top 20 shows domain diversity

### Milestone 2: Topic Separation
- [ ] Concept clusters align with domains
- [ ] Cross-domain terms identified and handled
- [ ] Query expansion respects domain boundaries

### Milestone 3: Expert Training Ready
- [ ] 2,500+ sample documents
- [ ] 80+ domains with 25+ docs each
- [ ] Clear domain vocabulary signatures
- [ ] Training/validation splits defined

### Milestone 4: MoE Prototype
- [ ] 10+ trained micro-experts
- [ ] Router correctly dispatches 85%+ queries
- [ ] Ensemble outperforms single model on diverse queries

---

## Next Steps

1. **Immediate**: Review this plan, prioritize domains
2. **This Week**: Build generation infrastructure
3. **Next Week**: Begin Phase 1 document generation
4. **Ongoing**: Monitor balance, adjust strategy

---

*Document created: 2025-12-17*
*Target completion: 8 weeks*
*Owner: [TBD]*
