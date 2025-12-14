# Memory Manager Skill

Create, search, and manage knowledge memories stored in git.

## Overview

This skill helps you work with the text-as-memories system - treating documents as persistent, versioned, searchable memories that build institutional knowledge.

## Memory Types

### 1. Daily Memories (`samples/memories/YYYY-MM-DD-topic.md`)
Capture learnings, discoveries, and insights from development sessions.

### 2. Decision Records (`samples/decisions/adr-NNN-title.md`)
Document architectural decisions with context, options, and rationale.

### 3. Concept Documents (`samples/memories/concept-topic.md`)
Consolidated knowledge about specific topics, refined from multiple memories.

## Commands

### Create a Memory Entry

When the user wants to capture a learning or insight:

```bash
# Create the file
cat > samples/memories/$(date +%Y-%m-%d)-TOPIC.md << 'EOF'
# Memory Entry: $(date +%Y-%m-%d) TOPIC

**Tags:** `tag1`, `tag2`
**Related:** [[other-memory.md]], [[concept.md]]

---

## Context
What prompted this learning?

## What I Learned
- Key insight 1
- Key insight 2

## Connections Made
- How this relates to other knowledge

## Future Exploration
- [ ] Follow-up item

---
*Committed to memory at: $(date -Iseconds)*
EOF
```

### Create a Decision Record

When the user makes an architectural decision:

```bash
# Find next ADR number
NEXT_NUM=$(ls samples/decisions/adr-*.md 2>/dev/null | wc -l)
NEXT_NUM=$((NEXT_NUM + 1))
PADDED=$(printf "%03d" $NEXT_NUM)

cat > samples/decisions/adr-${PADDED}-TITLE.md << 'EOF'
# ADR-${PADDED}: TITLE

**Status:** Proposed | Accepted | Deprecated | Superseded
**Date:** $(date +%Y-%m-%d)
**Deciders:** Team/Person
**Tags:** `tag1`, `tag2`

---

## Context and Problem Statement
What is the issue?

## Decision Drivers
1. Driver 1
2. Driver 2

## Considered Options
### Option 1: Name
**Pros:** ...
**Cons:** ...

### Option 2: Name
**Pros:** ...
**Cons:** ...

## Decision Outcome
**Chosen Option:** Option X

**Rationale:** Why this option?

## Consequences
### Positive
- ...

### Negative
- ...

---
EOF
```

### Search Memories

```bash
# Search indexed memories (requires corpus_dev.pkl)
python scripts/search_codebase.py "query terms" --top 10

# Search with expansion to see related terms
python scripts/search_codebase.py "query" --expand
```

### Index New Memories

After creating memories, re-index for search:

```bash
python scripts/index_codebase.py --incremental
```

## Best Practices

1. **Write memories immediately** - Capture insights while fresh
2. **Use consistent tags** - Makes searching easier
3. **Link related memories** - Use `[[wiki-style]]` references
4. **Include context** - Future you won't remember why
5. **Consolidate periodically** - Merge related memories into concepts
6. **Commit memories** - They're only persistent once in git

## Integration with Tasks

When completing a task, consider creating a memory:

```markdown
## What I Learned (from Task T-XXXXX)
- The retrospective from the task
- Additional insights discovered
- Links to files modified
```

## Example Workflow

1. **During session**: Notice something interesting
2. **Capture**: Create a memory entry with the insight
3. **Connect**: Add `[[links]]` to related knowledge
4. **Commit**: `git add samples/ && git commit -m "memory: ..."`
5. **Index**: `python scripts/index_codebase.py --incremental`
6. **Later**: Search to recall: `python scripts/search_codebase.py "that thing"`

## File Locations

- Memories: `samples/memories/`
- Decisions: `samples/decisions/`
- Main guide: `docs/text-as-memories.md`
