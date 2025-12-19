# The Living Book: Vision for Intelligent Documentation

> *"Code tells you what. Comments tell you why. A living book tells you the journey."*

## Executive Summary

This document articulates the vision for transforming the Cortical Chronicles from **excellent technical documentation** into a **publishable narrative book** that writes itself during development.

The core insight: **Your ML data collection system already captures the development story**. Every query, investigation, commit, and CI result forms a narrative arc that publishers want. The missing piece is the **narrative synthesis layer** that transforms structured data into compelling reading.

---

## The Problem We're Solving

### What Publishers Want vs. What We Generate

| Dimension | Current Output | Publisher Requirement | Gap |
|-----------|---------------|----------------------|-----|
| **Accuracy** | Excellent | Excellent | None |
| **Completeness** | Comprehensive | Curated | Minor |
| **Narrative Flow** | Reference-style | Story arc | **Significant** |
| **Reader Journey** | Random access | Progressive | **Significant** |
| **Voice** | Technical | Engaging | **Moderate** |
| **Problem-Solving** | Implicit | Explicit stories | **Significant** |
| **Lessons Learned** | Scattered | Distilled wisdom | **Moderate** |
| **Exercises** | Missing | Required | **Complete** |

### The Opportunity

We already capture everything needed for publishable content:

```
┌─────────────────────────────────────────────────────────────────┐
│                 DATA WE ALREADY COLLECT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  .git-ml/chats/         → Developer intent, questions, reasoning │
│  .git-ml/sessions/      → Complete investigation journeys        │
│  .git-ml/commits.jsonl  → Solutions with diffs and context       │
│  samples/decisions/     → Architectural choices with rationale   │
│  samples/memories/      → Crystallized insights and patterns     │
│  git log               → Timeline of evolution                   │
│  tests/                → Examples waiting to become exercises    │
│                                                                  │
│  THIS IS A BOOK WAITING TO BE ASSEMBLED.                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Vision: Co-Generated Documentation

### Paradigm Shift

**Old Model**: Write code → Later, document it
**New Model**: Write code → Documentation generates alongside

Every development session becomes potential book content:

```
┌─────────────────────────────────────────────────────────────────┐
│              CO-GENERATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Developer Query     →   ML Captures Intent                      │
│  "Fix auth bug"          "Problem: Auth was failing"             │
│                                                                  │
│  Investigation       →   ML Captures Journey                     │
│  Read, Grep, Test        "Explored: auth.py, tokens.py"          │
│                                                                  │
│  Solution            →   ML Captures Resolution                  │
│  Edit, Commit            "Fix: Token expiry was wrong"           │
│                                                                  │
│  Verification        →   ML Captures Outcome                     │
│  CI Pass                 "Result: All 3000 tests pass"           │
│                                                                  │
│                      ↓                                           │
│          AUTOMATIC CHAPTER GENERATION                            │
│                                                                  │
│  "Chapter 12: Debugging Token Expiry"                            │
│  - The Problem (from query)                                      │
│  - The Investigation (from tool trace)                           │
│  - The Discovery (from response insights)                        │
│  - The Solution (from commit diff)                               │
│  - The Lesson (extracted pattern)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## New Generator Architecture

### Expanded Book Structure

```
book/
├── 00-preface/
│   ├── how-this-book-works.md        # Existing
│   └── the-living-book-vision.md     # NEW: This document as chapter
│
├── 01-foundations/                    # Existing algorithm chapters
│   └── alg-*.md
│
├── 02-architecture/                   # Existing module docs
│   └── mod-*.md
│
├── 03-decisions/                      # Existing ADRs
│   └── adr-*.md
│
├── 04-evolution/                      # Existing timeline/features
│   └── *.md
│
├── 05-case-studies/                   # NEW: Narrative problem-solving
│   ├── index.md
│   ├── case-performance-hunt.md      # "The Great Performance Hunt"
│   ├── case-bigram-mystery.md        # "The Bigram Separator Mystery"
│   └── case-*.md
│
├── 06-lessons/                        # NEW: Distilled wisdom
│   ├── index.md
│   ├── lessons-performance.md        # Performance lessons
│   ├── lessons-architecture.md       # Architecture lessons
│   ├── lessons-testing.md            # Testing lessons
│   └── lessons-debugging.md          # Debugging lessons
│
├── 07-concepts/                       # NEW: Concept evolution
│   ├── index.md
│   ├── concept-importance.md         # How "importance" evolved
│   ├── concept-semantic.md           # How "semantic" evolved
│   └── concept-*.md
│
├── 08-exercises/                      # NEW: Reader engagement
│   ├── index.md
│   ├── ex-foundations.md             # Algorithm exercises
│   ├── ex-search.md                  # Search exercises
│   └── ex-advanced.md                # Advanced challenges
│
├── 09-journey/                        # NEW: Reader learning path
│   ├── index.md
│   ├── journey-beginner.md           # "Start Here"
│   ├── journey-intermediate.md       # "Going Deeper"
│   └── journey-advanced.md           # "Mastery Path"
│
└── 10-future/                         # Existing roadmap
    └── index.md
```

---

## Generator Specifications

### 1. CaseStudyGenerator

**Purpose**: Transform ML session data into compelling problem-solving narratives.

**Data Sources**:
- `.git-ml/sessions/*.json` - Complete session transcripts
- `.git-ml/chats/*/chat-*.json` - Individual exchanges
- `.git-ml/tracked/commits.jsonl` - Linked commits

**Selection Criteria** (sessions that make good case studies):
```python
def is_narrative_worthy(session: SessionData) -> bool:
    """Identify sessions with compelling narrative arc."""
    return (
        session.exchange_count >= 5 and           # Substantial investigation
        session.tools_used_count >= 3 and         # Multiple approaches tried
        session.commits_made >= 1 and             # Resulted in solution
        has_problem_statement(session) and        # Clear "why" at start
        has_resolution(session)                   # Clear outcome
    )
```

**Output Structure**:
```markdown
# Case Study: [Title]

## The Problem
[Extracted from initial query - what triggered the investigation?]

## The Investigation
[Tool trace narrative - what was explored, in what order?]

### Dead Ends
[Failed approaches - what didn't work and why?]

### The Breakthrough
[The moment of discovery - what insight solved it?]

## The Solution
[Commit summary with key changes highlighted]

## The Lesson
[Generalized insight for future developers]

## Try It Yourself
[Exercise based on this case study]
```

**Example Output**:

> # Case Study: The Great Performance Hunt
>
> ## The Problem
>
> It started with a timeout. The `compute_all()` function was hanging on a corpus of just 125 documents—far smaller than our target of 10,000+. Something was fundamentally wrong.
>
> ## The Investigation
>
> The obvious suspect was Louvain clustering. It's our most complex algorithm—O(n log n) community detection with multiple passes. But assumptions are dangerous.
>
> We started with profiling:
>
> ```bash
> python scripts/profile_full_analysis.py
> ```
>
> The results surprised us...

---

### 2. LessonExtractor

**Purpose**: Distill wisdom from bugfixes, refactors, and performance commits.

**Data Sources**:
- Git commits with types: `fix:`, `refactor:`, `perf:`
- Linked session data showing debugging process
- ADR references in commit messages

**Classification Categories**:
```python
LESSON_CATEGORIES = {
    'performance': ['perf:', 'optimize', 'slow', 'timeout', 'O(n'],
    'correctness': ['fix:', 'bug', 'wrong', 'incorrect', 'fail'],
    'architecture': ['refactor:', 'extract', 'split', 'modular'],
    'testing': ['test:', 'coverage', 'assert', 'mock'],
    'security': ['security:', 'vulnerability', 'auth', 'token'],
    'documentation': ['docs:', 'readme', 'comment', 'explain']
}
```

**Output Structure**:
```markdown
# Lessons Learned: [Category]

## Overview
[Summary statistics: X lessons from Y commits over Z months]

## Lesson 1: [Title]
**Commit:** `abc123` | **Date:** 2025-12-15

### The Mistake
[What went wrong]

### The Fix
[What was changed]

### The Principle
[Generalized rule for avoiding this in future]

### Code Example
```python
# Before (wrong)
...

# After (correct)
...
```
```

**Lesson Templates by Category**:

| Category | Template Focus |
|----------|---------------|
| Performance | Profile first, measure always, watch for O(n²) |
| Correctness | Edge cases, assumptions, validation |
| Architecture | Single responsibility, dependency direction |
| Testing | Coverage gaps, assertion quality, fixtures |

---

### 3. ConceptEvolutionGenerator

**Purpose**: Track how key concepts emerged and strengthened over time.

**Data Sources**:
- Git commits with timestamps
- Cortical processor Layer 2 (CONCEPTS) clusters
- TF-IDF scores per time period

**Algorithm**:
```python
def track_concept_evolution(concept: str, commits: List[Commit]) -> ConceptTimeline:
    """
    Track how a concept grew in the codebase.

    Returns timeline with:
    - First mention
    - Growth curve (mentions per week)
    - Related concepts added over time
    - Key commits that shaped the concept
    """
    timeline = []
    for week in group_by_week(commits):
        mentions = count_mentions(concept, week)
        related = find_related_terms(concept, week)
        key_commits = filter_by_concept(concept, week)

        timeline.append({
            'week': week.start,
            'mentions': mentions,
            'related_terms': related,
            'key_commits': key_commits,
            'cluster_size': get_concept_cluster_size(concept, week)
        })

    return ConceptTimeline(concept, timeline)
```

**Output Structure**:
```markdown
# Concept Evolution: [Concept Name]

## Birth
[When and where this concept first appeared]

## Growth Timeline

### Week 1: Emergence
- First mention in `file.py`
- Initial meaning: [definition]

### Week 3: Expansion
- Connected to: [related concepts]
- Key commit: `abc123` - [what changed]

### Week 6: Maturation
- Cluster size: 12 related terms
- Central to: [which features]

## The Concept Today
[Current definition and relationships]

## Visualizing the Growth
[ASCII or Mermaid diagram of concept connections over time]
```

---

### 4. DecisionStoryGenerator

**Purpose**: Enrich ADRs with the conversation context that led to decisions.

**Data Sources**:
- `samples/decisions/adr-*.md` - Existing ADRs
- `.git-ml/chats/` - Conversations mentioning ADR topics
- Commit messages referencing ADR numbers

**Enrichment Process**:
```python
def enrich_adr(adr: ADR) -> EnrichedADR:
    """Add conversation context to an ADR."""
    # Find chats discussing this topic
    relevant_chats = search_chats(adr.title_keywords)

    # Extract the debate
    alternatives_discussed = extract_alternatives(relevant_chats)
    concerns_raised = extract_concerns(relevant_chats)
    resolution_reasoning = extract_resolution(relevant_chats)

    # Find implementation commits
    implementation = find_commits_referencing(adr.number)

    return EnrichedADR(
        original=adr,
        debate=alternatives_discussed,
        concerns=concerns_raised,
        reasoning=resolution_reasoning,
        implementation=implementation
    )
```

**Output Structure**:
```markdown
# Decision Story: [ADR Title]

## The Question
[What problem needed a decision?]

## The Debate

### Option A: [Name]
**Advocated because:** [reasoning from chats]
**Concerns:** [issues raised]

### Option B: [Name]
**Advocated because:** [reasoning]
**Concerns:** [issues]

## The Decision
[What was chosen and why - from ADR + chat context]

## In Hindsight
[If available: did it work out? Any follow-up changes?]

## Implementation
[Links to commits that implemented the decision]
```

---

### 5. ReaderJourneyGenerator

**Purpose**: Create progressive learning paths using PageRank concept ordering.

**Algorithm**:
```python
def generate_reader_journey() -> List[JourneyStep]:
    """
    Use PageRank to order concepts by prerequisite dependencies.

    Foundational concepts have:
    - High PageRank (many things depend on them)
    - Few incoming edges (don't require prior knowledge)

    Advanced concepts have:
    - Lower PageRank (more specialized)
    - Many incoming edges (require prerequisites)
    """
    processor = load_corpus()
    processor.compute_importance()

    concepts = []
    for col in processor.layers[CorticalLayer.CONCEPTS].minicolumns.values():
        incoming = len(col.feedback_connections)
        outgoing = len(col.feedforward_connections)

        concepts.append({
            'name': col.content,
            'pagerank': col.pagerank,
            'prerequisite_count': incoming,
            'enables_count': outgoing,
            'difficulty': compute_difficulty(incoming, col.pagerank)
        })

    # Sort by difficulty for progressive disclosure
    return sorted(concepts, key=lambda c: c['difficulty'])
```

**Output Structure**:
```markdown
# Your Learning Journey

## Start Here (Foundational)
*Concepts that unlock everything else*

1. **[Concept]** - [1-sentence explanation]
   - Chapter: [link]
   - Time: ~15 min
   - Prerequisites: None

2. **[Concept]** - [explanation]
   ...

## Going Deeper (Intermediate)
*Building on foundations*

5. **[Concept]** - [explanation]
   - Prerequisites: Concepts 1, 2
   ...

## Mastery Path (Advanced)
*For those who want expertise*

10. **[Concept]** - [explanation]
    - Prerequisites: Concepts 3, 5, 7
    ...

## Suggested Reading Order

```
Week 1: Foundations
├── Day 1-2: [Concept 1, 2]
├── Day 3-4: [Concept 3, 4]
└── Day 5: Exercises 1-5

Week 2: Core Skills
├── Day 1-2: [Concept 5, 6]
...
```
```

---

### 6. ExerciseGenerator

**Purpose**: Transform test cases into reader exercises.

**Data Sources**:
- `tests/` - All test files
- Test docstrings and assertions
- Fixture data

**Selection Criteria**:
```python
def is_good_exercise(test: TestCase) -> bool:
    """Identify tests that make good exercises."""
    return (
        has_clear_docstring(test) and
        not is_trivial(test) and           # More than just assert True
        has_educational_value(test) and    # Tests a learnable concept
        is_self_contained(test)            # Doesn't need complex setup
    )
```

**Transformation Process**:
```python
def test_to_exercise(test: TestCase) -> Exercise:
    """Transform a test into an exercise."""
    return Exercise(
        title=extract_title(test.docstring),
        concept=identify_concept(test),
        difficulty=estimate_difficulty(test),
        setup=extract_setup(test),
        prompt=generate_prompt(test),
        hints=[generate_hint(test, level) for level in [1, 2, 3]],
        solution=extract_solution(test),
        verification=generate_verification(test)
    )
```

**Output Structure**:
```markdown
# Exercise: [Title]

**Concept:** [What this teaches]
**Difficulty:** [Beginner/Intermediate/Advanced]
**Time:** ~[X] minutes

## Setup

```python
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()
# Your code here...
```

## Your Task

[Clear description of what to implement]

## Hints

<details>
<summary>Hint 1 (Approach)</summary>
[General direction without giving it away]
</details>

<details>
<summary>Hint 2 (API)</summary>
[Relevant methods to use]
</details>

<details>
<summary>Hint 3 (Almost there)</summary>
[Specific guidance]
</details>

## Solution

<details>
<summary>Click to reveal</summary>

```python
# Solution code from test
```

**Explanation:** [Why this works]
</details>

## Verify Your Solution

```python
# Verification code
assert your_result == expected
print("Success!")
```
```

---

## Narrative Voice Templates

### The Missing Ingredient

The generators above produce structured content. To make it publishable, we add **narrative voice templates**:

```python
NARRATIVE_TEMPLATES = {
    'case_study_opening': [
        "It started with {problem}. {context}.",
        "The first sign of trouble was {symptom}.",
        "Nobody expected {situation} to cause {consequence}.",
    ],

    'investigation_transition': [
        "The obvious suspect was {suspect}. But assumptions are dangerous.",
        "We started with the usual tools: {tools}.",
        "Before diving into code, we needed to understand {context}.",
    ],

    'breakthrough_moment': [
        "Then we saw it. {discovery}.",
        "The profiler revealed something unexpected: {finding}.",
        "A single line told the whole story: {code}.",
    ],

    'lesson_synthesis': [
        "The lesson? {principle}.",
        "This taught us: {wisdom}.",
        "Going forward, we always {practice}.",
    ]
}
```

### Example: Template Application

**Raw Data**:
```json
{
  "problem": "compute_all() timeout",
  "suspect": "Louvain clustering",
  "actual_cause": "O(n²) bigram connections",
  "tools_used": ["profiler", "cProfile", "line_profiler"],
  "discovery": "99% of time in bigram_connections(), not Louvain"
}
```

**Generated with Template**:

> It started with a timeout. The `compute_all()` function was hanging on a corpus of just 125 documents—far smaller than our target of 10,000+.
>
> The obvious suspect was Louvain clustering. It's our most complex algorithm—O(n log n) community detection with multiple passes. But assumptions are dangerous.
>
> We started with the usual tools: cProfile for high-level timing, line_profiler for hot spots.
>
> Then we saw it. 99% of execution time wasn't in Louvain at all—it was buried in `bigram_connections()`, creating O(n²) pairs for common terms like "self".
>
> The lesson? Profile before optimizing. The obvious culprit is often innocent.

---

## Implementation Phases

### Phase 1: Quick Wins (1-2 days)

| Task | Generator | Output |
|------|-----------|--------|
| Lesson extraction | `LessonExtractor` | `06-lessons/` chapters |
| ADR enrichment | `DecisionStoryGenerator` | Enhanced `03-decisions/` |
| Vision chapter | Manual | `00-preface/the-living-book-vision.md` |

### Phase 2: Narrative Generators (3-5 days)

| Task | Generator | Output |
|------|-----------|--------|
| Case studies | `CaseStudyGenerator` | `05-case-studies/` chapters |
| Concept evolution | `ConceptEvolutionGenerator` | `07-concepts/` chapters |
| Reader journey | `ReaderJourneyGenerator` | `09-journey/` chapters |

### Phase 3: Reader Engagement (2-3 days)

| Task | Generator | Output |
|------|-----------|--------|
| Exercise extraction | `ExerciseGenerator` | `08-exercises/` chapters |
| Narrative voice | Template system | All chapters enhanced |
| Cross-references | Link generator | Bidirectional links |

### Phase 4: Polish & Publish (2-3 days)

| Task | Component | Output |
|------|-----------|--------|
| PDF/EPUB export | pandoc integration | `book.pdf`, `book.epub` |
| Table of figures | Index generator | Front matter |
| Publisher package | Manual | Submission materials |

---

## Success Metrics

### Quantitative

| Metric | Current | Target |
|--------|---------|--------|
| Chapter types | 5 | 10 |
| Word count | ~50,000 | ~100,000 |
| Exercises | 0 | 50+ |
| Case studies | 0 | 10+ |
| Lessons extracted | 0 | 100+ |
| Cross-references | ~50 | ~500 |

### Qualitative

| Dimension | Current | Target |
|-----------|---------|--------|
| Reader can learn progressively | No | Yes |
| Stories explain "why" | Partial | Complete |
| Exercises reinforce learning | No | Yes |
| Voice is engaging | Technical | Conversational |
| Publishable quality | No | Yes |

---

## Conclusion

The infrastructure exists. The data is being captured. The generators can be built.

What transforms technical documentation into a publishable book is **narrative synthesis**: the art of weaving facts into stories, ordering concepts for progressive disclosure, and speaking to readers as fellow learners rather than reference-seekers.

This vision document serves two purposes:
1. **As a chapter**: Explaining to readers how the book creates itself
2. **As a specification**: Guiding the implementation of new generators

The Cortical Chronicles can become not just documentation, but a **story of building intelligence**—told by the system that embodies it.

---

## See Also

- [How This Book Works](../book/00-preface/how-this-book-works.md) - Current generation system
- [Product Vision](./VISION.md) - Overall product direction
- [ML Data Collection](./ml-milestone-thresholds.md) - Training data architecture
- [Text as Memories](./text-as-memories.md) - Knowledge crystallization philosophy

---

*This document is part of the Cortical Text Processor documentation and
will be included in The Cortical Chronicles as a preface chapter.*
