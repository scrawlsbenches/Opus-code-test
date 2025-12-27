# Chat History Backfill Instructions

## For the Agent Running in Old Threads

You are helping to backfill chat history for ML training data collection. Your task is simple:

### Step 1: Navigate to the Project
```bash
cd /path/to/Opus-code-test  # Adjust to actual project path
```

### Step 2: Run the ML Session Capture
```bash
# Start a new ML session (if not already running)
python scripts/ml_data_collector.py session start

# The session will capture this conversation automatically
```

### Step 3: Discuss the Codebase
Have a natural conversation about the project. Ask and answer questions like:

**About Algorithms:**
- "What is PageRank and how does it work in this codebase?"
- "Explain TF-IDF and where it's implemented"
- "How does Louvain clustering detect communities?"
- "What is BM25 and why is it the default?"

**About Components:**
- "What is GoTManager and what does it do?"
- "Explain how Woven Mind works"
- "What does the tokenizer do?"
- "How does the CorticalTextProcessor work?"

**About Concepts:**
- "What is Hebbian learning in this context?"
- "Explain lateral connections"
- "What are minicolumns?"
- "How does query expansion work?"

**About Processes:**
- "How do I run the tests?"
- "What's the TDD workflow here?"
- "How do I index the codebase?"
- "How do I create a new task?"

### Step 4: End the Session
```bash
# End the session to trigger transcript capture
python scripts/ml_data_collector.py session end --summary "Backfill conversation about [topics covered]"
```

### Step 5: Verify Data Was Captured
```bash
# Check that chat data was saved
ls -la .git-ml/tracked/chunked/
python scripts/ml_data_collector.py stats
```

---

## What Happens Behind the Scenes

1. The `SessionStart` hook automatically starts ML collection
2. Every Q&A exchange is logged to `.git-ml/`
3. The `Stop` hook captures the full transcript
4. Data is stored in `.git-ml/tracked/chunked/` as JSONL
5. This data is gitignored (won't cause merge conflicts)
6. The data augmentation pipeline will find and use it

---

## Quick One-Liner Version

If you just want to trigger data capture without a full conversation:

```bash
cd /path/to/Opus-code-test && \
python scripts/ml_data_collector.py session start && \
echo "Session started - have your conversation - then run:" && \
echo "python scripts/ml_data_collector.py session end --summary 'backfill'"
```

---

## After Backfilling Multiple Threads

Once you've run this in several old threads, the main branch can pull in the chat data:

```bash
# On your current branch, run the augmentation to pick up new chat data
python -m benchmarks.codebase_slm.data_augmentation

# Check how many chat patterns were found
grep -c "ml_chat" benchmarks/codebase_slm/data/augmented_corpus.txt

# Retrain with the new data
python -m benchmarks.codebase_slm.train_augmented

# Run benchmarks to see improvement
python -m benchmarks.codebase_slm.benchmark_suite
```

The `.git-ml/` directory persists across branch switches, so all collected data accumulates locally.
