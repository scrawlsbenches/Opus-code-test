# Chat History Backfill Instructions

## For the Agent Running in Old Threads

You are helping to backfill chat history for ML training data collection. Your task is simple:

### Step 1: Navigate to the Project
```bash
cd /path/to/Opus-code-test  # Adjust to actual project path
```

### Step 2: Pull Latest from Main (CRITICAL)

You are on a historical branch from the past. Pull the latest ML collection infrastructure:

```bash
# Fetch and merge latest from main to get ML collection scripts
git fetch origin main
git merge origin/main --no-edit

# If merge conflicts occur in unrelated files, resolve them or:
git checkout --theirs .  # Accept main's version for conflicts
git add -A && git commit -m "merge: Pull latest ML infrastructure from main"
```

**Why this is needed:** Old branches don't have the ML data collection scripts. You need them to capture chat history.

### Step 3: Verify .gitignore Allows Chat Data to be Tracked

Check that chat data will be pushed to git (not ignored):

```bash
# Verify .git-ml/tracked/ is NOT in .gitignore (should see "!.git-ml/tracked/")
grep -n "git-ml/tracked" .gitignore

# You should see something like:
#   !.git-ml/tracked/  (the ! means NOT ignored)
# If missing, add it:
echo '!.git-ml/tracked/' >> .gitignore
echo '!.git-ml/tracked/**' >> .gitignore
```

**Expected .gitignore structure:**
```
# ML training data - tiered sharing strategy
# TRACKED IN GIT (small, shareable):
#   - .git-ml/tracked/       - JSONL files
# LOCAL ONLY (large or sensitive):
#   - .git-ml/chats/         - Full transcripts (too large)
.git-ml/chats/
.git-ml/sessions/
# ... but NOT .git-ml/tracked/
```

### Step 4: Run the ML Session Capture
```bash
# Start a new ML session (if not already running)
python scripts/ml_data_collector.py session start

# The session will capture this conversation automatically
```

### Step 5: Discuss the Codebase
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

### Step 6: End the Session
```bash
# End the session to trigger transcript capture
python scripts/ml_data_collector.py session end --summary "Backfill conversation about [topics covered]"
```

### Step 7: Verify Data Was Captured and Push
```bash
# Check that chat data was saved to the TRACKED location
ls -la .git-ml/tracked/chunked/
python scripts/ml_data_collector.py stats

# Commit and push the chat data
git add .git-ml/tracked/
git commit -m "ml: Backfill chat history from $(git branch --show-current)"
git push origin $(git branch --show-current)
```

**Important:** The data MUST be pushed to git for it to be available on other branches.

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

Once you've run this in several old threads, collect the data on your working branch:

```bash
# First, fetch all the backfilled data from remote branches
git fetch --all

# Merge chat data from each backfilled branch
# (the .git-ml/tracked/ directories will combine)
for branch in $(git branch -r | grep -v main | head -10); do
    git merge $branch --no-edit -m "merge: Pull ML data from $branch" || git merge --abort
done

# Or cherry-pick just the ML data commits
git log --oneline --all --grep="ml: Backfill" | head -10

# Run the augmentation to pick up new chat data
python -m benchmarks.codebase_slm.data_augmentation

# Check how many chat patterns were found
grep -c "ml_chat" benchmarks/codebase_slm/data/augmented_corpus.txt

# Retrain with the new data
python -m benchmarks.codebase_slm.train_augmented

# Run benchmarks to see improvement
python -m benchmarks.codebase_slm.benchmark_suite
```

**Note:** The `.git-ml/tracked/` data is in git, so you need to merge or cherry-pick from backfilled branches to get the data.
