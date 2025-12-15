---
description: Generate knowledge transfer document from current session
---
# Knowledge Transfer Generator

Generate a knowledge transfer document based on the current session context.

## Instructions

Analyze the conversation history and create a comprehensive knowledge transfer document. Save it to `samples/memories/` with today's date.

### Document Structure

Create the document with this structure:

```markdown
# Session Knowledge Transfer: [DATE] [BRIEF-TOPIC]

**Date:** YYYY-MM-DD
**Session:** [Describe the session focus]
**Branch:** [Current git branch if applicable]

## Summary

2-3 sentences capturing what was accomplished and why it matters.

## What Was Accomplished

### Completed Tasks
- List tasks that were completed with their IDs
- Include brief description of what each involved

### Code Changes
- Files created or modified
- Key functions/classes added
- Bug fixes applied

### Documentation Added
- New docs created
- Existing docs updated

## Key Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| ... | ... | ... |

## Problems Encountered & Solutions

### Problem 1: [Title]
**Symptom:** What was observed
**Root Cause:** What was actually wrong
**Solution:** How it was fixed
**Lesson:** What to remember for next time

## Technical Insights

- Key technical learnings from this session
- Non-obvious discoveries
- Performance findings
- Security considerations

## Context for Next Session

### Current State
- What's working
- What's in progress
- Any uncommitted changes

### Suggested Next Steps
1. Prioritized list of what to do next
2. Any blockers to be aware of
3. Related tasks to consider

### Files to Review
- Key files that were central to this work
- Entry points for understanding the changes

## Connections to Existing Knowledge

- Links to related memories: [[memory-name.md]]
- Related concepts: [[concept-name.md]]
- Relevant decisions: [[adr-NNN.md]]

## Tags

`tag1`, `tag2`, `tag3`
```

### Output Location

Save to: `samples/memories/YYYY-MM-DD-session-[topic].md`

Where `[topic]` is a brief kebab-case description of the main focus.

### After Creating

1. Show the user the generated document
2. Suggest: `git add samples/memories/ && git commit -m "memory: session knowledge transfer"`
3. Remind to re-index: `python scripts/index_codebase.py --incremental`

### Tips for Good Knowledge Transfers

- **Be specific** - Include file paths, function names, line numbers
- **Capture the "why"** - Decisions without rationale are less useful
- **Note surprises** - Things that were unexpected are often valuable
- **Link generously** - Cross-references strengthen the knowledge graph
- **Think forward** - What would help the next person (or future you)?
