# What is the Cognitive Agent?

## Identity and Purpose

The Cognitive Agent is a self-contained knowledge system that learns from text and code to help future agents understand and navigate the codebase. It exists because AI agents lose context between sessions and need a way to recover knowledge quickly.

Think of it as long-term memory for AI agents working on this codebase.

## Core Capabilities

The Cognitive Agent can:
- Learn word associations from documents and code
- Predict what words typically follow other words
- Find semantically related concepts
- Answer questions about the codebase
- Index code entities (files, classes, functions)

## Why Does It Exist?

When an AI agent starts a new session, it knows nothing about the specific codebase. The Cognitive Agent provides:

1. **Context Recovery** - "What was I working on?" becomes answerable
2. **Concept Exploration** - "What relates to storage?" returns relevant terms
3. **Code Navigation** - "Where is authentication handled?" finds locations
4. **Design Understanding** - "Why was this built this way?" has answers

## How It Works

The agent builds a hypergraph where:
- **WORD atoms** represent vocabulary terms
- **SIMILARITY links** connect co-occurring words (bidirectional)
- **FOLLOWS links** capture word sequences (directional)
- **IDF weighting** down-weights common words like "the" and "and"

When you query for associations, it traverses these links to find related concepts.

## Key Components

| Component | Purpose |
|-----------|---------|
| CognitiveGraph | Hypergraph storage for atoms and links |
| BPETokenizer | Vocabulary learning and IDF tracking |
| TextToAtomsBridge | Converts text to graph structure |
| IncrementalTrainer | Trains on new documents efficiently |

## Common Questions and Answers

**Q: How do I ask the cognitive agent a question?**
A: Use the CLI command `python -m cortical.cognitive ask "your question here"`

**Q: How do I find concepts related to a word?**
A: Use `python -m cortical.cognitive query "word"` or programmatically call `agent.get_associations("word")`

**Q: How do I train on new documents?**
A: Use `python -m cortical.cognitive train path/to/documents --incremental`

**Q: What if I'm confused about where to start?**
A: Run `python -m cortical.cognitive status` to see the model state, then explore with queries.

**Q: How do I know if the model needs reindexing?**
A: Check staleness with `status` command. If above 20%, run `python -m cortical.cognitive reindex`

## When to Use the Cognitive Agent

Use it when you need to:
- Understand unfamiliar parts of the codebase
- Recover context after losing your place
- Find related concepts or code
- Explore what the codebase knows about a topic

## When NOT to Use It

Don't use it for:
- Exact text search (use grep/ripgrep instead)
- Finding specific file paths (use glob)
- Real-time code execution
- Tasks requiring external knowledge
