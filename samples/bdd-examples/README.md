# Behavior-Driven Development Examples

This directory contains sample files demonstrating Behavior-Driven Development (BDD) patterns as practiced in the Metus philosophy.

## What is BDD?

Behavior-Driven Development bridges the gap between technical implementation and business value by:

1. **Starting with user stories** - Who needs what, and why?
2. **Defining scenarios** - How do we know when it works?
3. **Writing executable specifications** - Tests that document behavior
4. **Implementing to satisfy scenarios** - Code serves the story

## Directory Contents

```
bdd-examples/
├── README.md                           # This file
├── user_story_template.md              # Template for writing user stories
├── writing_guide.md                    # Comprehensive BDD writing guide
├── researcher_discovers_knowledge.py   # Sample: Search discovery scenarios
└── system_handles_failures.py          # Sample: Reliability scenarios
```

## The BDD Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   1. DISCOVER → What story does the user tell?                  │
│                                                                  │
│   2. FORMULATE → Write Given-When-Then scenarios                │
│                                                                  │
│   3. AUTOMATE → Make the scenarios executable                   │
│                                                                  │
│   4. IMPLEMENT → Write code to make scenarios pass              │
│                                                                  │
│   5. CERTIFY → CI must approve before merge                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Behavior Precedes Implementation

Never write production code until you have a failing scenario that demands it.

```python
# WRONG: "Let me implement caching, then test it"
# RIGHT: "Let me describe what caching should achieve, then implement"
```

### 2. Scenarios Are Living Documentation

Tests written as scenarios document what the system does. They never go stale because they're executed on every build.

### 3. User-Centric Language

Name tests after what users observe, not what the code does.

```python
# WRONG
def test_tfidf_computation(self): ...

# RIGHT
def test_relevant_documents_rank_higher_than_irrelevant_ones(self): ...
```

### 4. Given-When-Then Structure

Every scenario follows this pattern:

- **Given** - The initial context (setup)
- **When** - The action taken (what we're testing)
- **Then** - The observable outcome (assertion)
- **Because** - Why this matters (optional but encouraged)

## Quick Start

1. Read `user_story_template.md` to understand story format
2. Review `researcher_discovers_knowledge.py` for a complete example
3. Use `writing_guide.md` when creating new scenarios
4. Place your scenarios in `tests/behavioral/`

## Running Behavioral Tests

```bash
# Run all behavioral scenarios
python -m pytest tests/behavioral/ -v

# Run a specific story
python -m pytest tests/behavioral/test_document_freshness.py -v

# Run with verbose output for debugging
python -m pytest tests/behavioral/ -v --tb=long
```

## Related Files

- `CLAUDE.md` - The complete Metus philosophy
- `tests/behavioral/` - Production behavioral scenarios
- `tests/unit/specifications/` - Unit-level specifications

---

*"We describe behavior, then make it true."* - The Metus Way
