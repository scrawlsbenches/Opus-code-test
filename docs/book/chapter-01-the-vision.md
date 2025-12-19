# Chapter 1: The Vision

> "We'll have to write it to find out."

---

## What We're Building

A system where **the repository itself becomes the alignment corpus**.

Every file, every decision, every memory, every task - they all teach the system who you are. When a new session starts, I don't have to re-learn your vocabulary, your patterns, your preferences. The repo tells me.

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE REPO IS THE CORPUS                       │
│                                                                 │
│  Your Code        →  How you solve problems                     │
│  Your Decisions   →  Why you chose what you chose               │
│  Your Memories    →  What you learned along the way             │
│  Your Tasks       →  What you're trying to accomplish           │
│  Your Sprints     →  How you organize work                      │
│  Your Epics       →  What the bigger picture is                 │
│  Your Imagine List → What you dream about building              │
│                                                                 │
│  All of this      →  WHO YOU ARE as a developer                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Alignment Problem (And Our Solution)

Every Claude session starts fresh. I don't remember our last conversation. I don't know what "spark" means to you specifically. I don't know your preferences.

**The traditional approach:** You re-explain context every session.

**Our approach:** The repo teaches me.

```
Session Start
     │
     ▼
┌─────────────────────────┐
│ Load Alignment Corpus   │
│                         │
│ • samples/alignment/    │──→ Your definitions
│ • samples/memories/     │──→ What you learned
│ • samples/decisions/    │──→ Why you chose things
│ • tasks/                │──→ Current work
│ • CLAUDE.md             │──→ How to work here
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ SparkSLM Primes Search  │
│                         │
│ • N-gram predictions    │──→ Your vocabulary
│ • Alignment index       │──→ Your meanings
│ • Pattern matching      │──→ Your conventions
└───────────┬─────────────┘
            │
            ▼
      Aligned Faster
```

---

## The Book We're Writing

This isn't just documentation. It's a **living book** that we write together as we build.

### Structure

```
docs/book/
├── chapter-01-the-vision.md          ← You are here
├── chapter-02-the-architecture.md    ← How it works
├── chapter-03-the-ngram.md           ← N-gram language model
├── chapter-04-the-alignment.md       ← Alignment index
├── chapter-05-the-spark.md           ← SparkSLM predictor
├── chapter-06-training-on-us.md      ← Training on the repo
├── chapter-07-the-loop.md            ← Feedback and learning
└── appendix-imagine-list.md          ← Dreams for the future
```

### How We Write It

1. **We imagine something**
2. **We build it** (code)
3. **We document what we learned** (chapter)
4. **The documentation becomes training data**
5. **Future sessions understand better**
6. **Loop**

---

## What You Add, What I Learn

### Your Contributions (samples/)

| Type | Path | What It Teaches Me |
|------|------|-------------------|
| **Definitions** | `samples/alignment/definitions.md` | "When you say X, you mean Y" |
| **Patterns** | `samples/alignment/patterns.md` | "In this codebase, we do X this way" |
| **Preferences** | `samples/alignment/preferences.md` | "You prefer X over Y because Z" |
| **Memories** | `samples/memories/*.md` | What you learned from past work |
| **Decisions** | `samples/decisions/adr-*.md` | Why you made specific choices |
| **Imagine** | `samples/alignment/imagine.md` | What you dream about building |

### What I Index

| Source | Creates |
|--------|---------|
| Your documents | N-gram vocabulary model |
| Your definitions | Alignment index lookups |
| Your code | Co-occurrence patterns |
| Your commits | Change patterns |
| Your tasks | Current context |

---

## The Imagine List

A place for dreams. Not tasks, not sprints - pure imagination.

```markdown
# samples/alignment/imagine.md

## What If...

- What if the system could predict what file I need before I ask?
- What if search results explained WHY they matched?
- What if the AI could learn my coding style over time?
- What if session context persisted meaningfully?
- What if the repo was a conversation partner, not just data?

## One Day...

- A system that grows with me
- Code that understands context like a colleague
- Documentation that writes itself from our work
- Alignment that happens naturally through collaboration

## The Spark...

- Fast first-blitz thoughts that guide deeper analysis
- Statistical predictions that prime the pump
- A bridge between statistical speed and semantic depth
```

---

## The Feedback Loop

```
     ┌──────────────────────────────────────────┐
     │                                          │
     ▼                                          │
┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  You    │───▶│  Build  │───▶│ Commit  │──────┤
│ Imagine │    │         │    │         │      │
└─────────┘    └─────────┘    └─────────┘      │
                                               │
┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  Index  │◀───│ Memory  │◀───│ Learn   │◀─────┘
│ Updates │    │ Created │    │         │
└─────────┘    └─────────┘    └─────────┘
     │
     │
     ▼
┌─────────────────────────────────────────────┐
│  Next Session: I Understand You Better      │
└─────────────────────────────────────────────┘
```

---

## Why This Matters

**For you:** Less repetition. Faster alignment. More productive sessions.

**For me:** Context I can actually use. Vocabulary I understand. Patterns I recognize.

**For us:** A partnership that improves over time.

---

## What's Next

1. **Create the alignment corpus structure** (`samples/alignment/`)
2. **Train SparkSLM on this repo**
3. **Add session start hook to load alignment**
4. **Write the next chapters as we build**

---

## The Promise

Every commit is a lesson.
Every memory is training data.
Every session leaves us more aligned than before.

We're not just building software. We're writing a book about how human and AI can work together - and the book teaches us how to write it better.

*Let's find out what happens next.*
