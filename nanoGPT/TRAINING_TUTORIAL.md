# Training nanoGPT: A Step-by-Step Educational Tutorial

This tutorial teaches you how to train a GPT-style language model from scratch. Before each step, we explain **what** you're about to do and **why** it matters.

## Table of Contents

1. [Understanding the Goal](#1-understanding-the-goal)
2. [Prerequisites](#2-prerequisites)
3. [Step 1: Collect Training Data](#step-1-collect-training-data)
4. [Step 2: Build the Dataset](#step-2-build-the-dataset)
5. [Step 3: Understand the Model Architecture](#step-3-understand-the-model-architecture)
6. [Step 4: Configure Training](#step-4-configure-training)
7. [Step 5: Run Training](#step-5-run-training)
8. [Step 6: Generate Text](#step-6-generate-text)
9. [Step 7: Evaluate and Iterate](#step-7-evaluate-and-iterate)

---

## 1. Understanding the Goal

### What is a Language Model?

A language model learns to predict the next token (word or character) given previous tokens. For example:

```
Input:  "The cat sat on the"
Output: "mat" (predicted next word)
```

### Why Train Your Own?

- **Domain Adaptation**: Pre-trained models know general language but not your codebase
- **Privacy**: Your data stays local
- **Understanding**: You learn how LLMs actually work
- **Customization**: Full control over architecture and training

### What We're Building

We'll train a GPT-2 style model on git commit history to learn coding patterns. The model will learn:
- How to write commit messages
- What code changes look like
- Patterns in your development workflow

---

## 2. Prerequisites

### Required Software

```bash
# Python 3.8+
python --version

# PyTorch (CPU or CUDA)
pip install torch

# This repository
cd /path/to/Opus-code-test
```

### Hardware Requirements

| Model Size | Parameters | Minimum RAM | Recommended |
|------------|------------|-------------|-------------|
| Tiny | ~1M | 2 GB | CPU |
| Small | ~10M | 4 GB | CPU/GPU |
| Medium | ~100M | 8 GB | GPU |
| Large | ~1B+ | 32 GB+ | Multi-GPU |

---

## Step 1: Collect Training Data

### What You're About to Do

Extract commit data from git history, including:
- Commit messages (what the developer intended)
- Code diffs (what actually changed)
- Metadata (branch, author, timestamp)

### Why This Matters

Language models learn from examples. The quality and quantity of training data directly determines model capability:

- **More data** → Better generalization
- **Diverse data** → Broader capabilities
- **Clean data** → Fewer artifacts in output

Git history is ideal because:
1. It's structured (message + diff pairs)
2. It shows intent (commit message) and implementation (diff)
3. It's already on your machine

### How to Do It

```bash
# Collect the last 500 commits with full diffs
python scripts/ml_data_collector.py backfill -n 500
```

**What this command does:**

1. Runs `git log` to get commit hashes
2. For each commit, extracts:
   - Message, author, timestamp, branch
   - Files changed with insertions/deletions
   - Actual diff content (the code)
3. Saves structured JSON to `.git-ml/commits/`

### Verify the Collection

```bash
# Check how many commits were collected
ls .git-ml/commits/ | wc -l

# View a sample commit
head -50 .git-ml/commits/$(ls .git-ml/commits/ | head -1)
```

**Expected output:**
```json
{
  "hash": "abc123...",
  "message": "fix: Handle edge case in parser",
  "hunks": [
    {
      "file": "src/parser.py",
      "lines_added": ["+    if value is None:", "+        return default"],
      "lines_removed": ["-    return value"]
    }
  ]
}
```

---

## Step 2: Build the Dataset

### What You're About to Do

Transform raw commit data into weighted training examples with train/validation/test splits.

### Why Weighting Matters

Not all commits are equally valuable for training:

| Signal | Weight | Reasoning |
|--------|--------|-----------|
| Main branch | 1.0× | Production-quality code |
| Feature branch | 0.6× | Work in progress |
| Has tests | 1.1× | Higher quality |
| Was reverted | 0.1× | Known to be problematic |
| Recent | Higher | More relevant patterns |

**The weight formula:**
```
weight = branch_weight × quality_multipliers × temporal_decay
```

This makes the model learn more from high-quality, recent code.

### Why Train/Val/Test Splits

- **Training set (80%)**: Model learns from this
- **Validation set (10%)**: Tune hyperparameters, detect overfitting
- **Test set (10%)**: Final evaluation (never peek during training!)

If you train on all data, you can't tell if the model memorized or generalized.

### How to Do It

```bash
# Build the dataset with weighted sampling
python scripts/build_training_dataset.py
```

**What this command does:**

1. Loads commits from `.git-ml/commits/`
2. Detects quality signals (tests, merges, reverts)
3. Computes weight for each commit
4. Formats as training examples:
   ```
   ### Commit Message
   fix: Handle edge case in parser

   ### Files Changed
   - src/parser.py

   ### Code Changes
   ```diff
   -    return value
   +    if value is None:
   +        return default
   ```
   ```
5. Splits into train/val/test (80/10/10)
6. Saves to `datasets/git_training_data_{train,val,test}.jsonl`

### Verify the Dataset

```bash
# Check dataset sizes
wc -l datasets/git_training_data_*.jsonl

# View a training example
head -1 datasets/git_training_data_train.jsonl | python -m json.tool
```

### Understanding the Output Format

Each line in the JSONL file contains:

```json
{
  "text": "### Commit Message\n...\n### Code Changes\n...",
  "weight": 0.42,
  "metadata": {
    "hash": "abc123",
    "branch": "main",
    "has_tests": true,
    "weight_breakdown": {"branch": 1.0, "tested": 1.1, "temporal": 0.38}
  }
}
```

---

## Step 3: Understand the Model Architecture

### What You're About to Learn

The GPT architecture before you configure it. Understanding the components helps you make informed choices.

### The Transformer Architecture

```
Input Tokens
     ↓
[Token Embedding] + [Position Embedding]
     ↓
┌─────────────────────────────────────┐
│         Transformer Block           │ ← Repeated N times
│  ┌─────────────────────────────┐   │
│  │      Layer Norm             │   │
│  │           ↓                 │   │
│  │   Multi-Head Attention      │   │ ← "Look at other tokens"
│  │           ↓                 │   │
│  │      + Residual             │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │      Layer Norm             │   │
│  │           ↓                 │   │
│  │   Feed-Forward Network      │   │ ← "Think about what I saw"
│  │           ↓                 │   │
│  │      + Residual             │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
     ↓
[Final Layer Norm]
     ↓
[Language Model Head] → Predict next token
```

### Key Components Explained

#### Token Embeddings
**What**: Converts token IDs to dense vectors
**Why**: Neural networks need continuous numbers, not discrete symbols
```python
# Token 42 → [0.1, -0.3, 0.8, ...] (768 dimensions)
self.wte = nn.Embedding(vocab_size, n_embd)
```

#### Position Embeddings
**What**: Adds position information to each token
**Why**: Attention is position-agnostic; we need to know word order
```python
# Position 0 → [0.2, 0.1, ...], Position 1 → [-0.1, 0.3, ...]
self.wpe = nn.Embedding(block_size, n_embd)
```

#### Multi-Head Attention
**What**: Allows each token to "look at" other tokens
**Why**: Context matters - "bank" means different things in different contexts

```python
# Each head learns different relationships
# Head 1: Maybe learns syntax (subject-verb agreement)
# Head 2: Maybe learns semantics (word meanings)
# Head 3: Maybe learns formatting (indentation patterns)
```

**Causal Masking**: We mask future tokens so the model can't cheat:
```
Token:    The  cat  sat  on   the  mat
Can see:  [1]  [2]  [3]  [4]  [5]  [6]

"sat" can see: The, cat, sat
"sat" cannot see: on, the, mat
```

#### Feed-Forward Network
**What**: Two linear layers with GELU activation
**Why**: Adds non-linearity and "thinking" capacity

```python
# Expand to 4× size, apply non-linearity, project back
self.c_fc = nn.Linear(n_embd, 4 * n_embd)      # 768 → 3072
self.gelu = nn.GELU()
self.c_proj = nn.Linear(4 * n_embd, n_embd)    # 3072 → 768
```

#### Residual Connections
**What**: Add input directly to output (skip connection)
**Why**: Helps gradients flow during training, enables deeper networks

```python
x = x + attention(layer_norm(x))  # Don't lose the original!
x = x + ffn(layer_norm(x))
```

### Configuration Parameters

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257   # Number of unique tokens
    block_size: int = 1024    # Maximum sequence length
    n_layer: int = 12         # Number of transformer blocks
    n_head: int = 12          # Attention heads per block
    n_embd: int = 768         # Embedding dimension
    dropout: float = 0.1      # Regularization
```

**How to choose these:**

| Parameter | Smaller | Larger | Trade-off |
|-----------|---------|--------|-----------|
| n_layer | Faster, less capacity | Slower, more capacity | Depth vs speed |
| n_head | Fewer attention patterns | More diverse attention | Must divide n_embd |
| n_embd | Less expressive | More expressive | Memory vs quality |
| block_size | Shorter context | Longer context | Memory grows O(n²) |

---

## Step 4: Configure Training

### What You're About to Do

Set hyperparameters that control how the model learns.

### Learning Rate

**What**: How big a step to take when updating weights
**Why**: Too high → unstable, diverges. Too low → slow, gets stuck.

```python
learning_rate = 3e-4  # Good default for transformers
```

**The learning rate schedule:**
```
LR
 ↑
 │    /‾‾‾‾‾‾‾‾‾‾‾\
 │   /             \
 │  /               \________________
 │ /
 └────────────────────────────────────→ Steps
   Warmup    Peak         Decay
```

- **Warmup**: Start small, increase gradually (prevents early instability)
- **Peak**: Full learning rate
- **Decay**: Gradually decrease (fine-tune as we converge)

### Batch Size

**What**: How many examples to process before updating weights
**Why**: Larger batches = more stable gradients, but need more memory

```python
batch_size = 8          # Examples per forward pass
block_size = 256        # Tokens per example
# Total: 8 × 256 = 2048 tokens per step
```

**Gradient Accumulation** (if memory-limited):
```python
gradient_accumulation_steps = 4
# Process 4 mini-batches, accumulate gradients, then update
# Effective batch size: 8 × 4 = 32
```

### Regularization

**What**: Techniques to prevent overfitting (memorizing training data)

**Dropout**: Randomly zero out neurons during training
```python
dropout = 0.1  # 10% of neurons disabled each forward pass
```

**Weight Decay**: Penalize large weights
```python
weight_decay = 0.1  # Only on 2D+ parameters (not biases)
```

### Mixed Precision Training

**What**: Use 16-bit floats for most operations
**Why**: 2× faster, half the memory, minimal quality loss

```python
# PyTorch automatic mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(x, y)
```

---

## Step 5: Run Training

### What You're About to Do

Execute the training loop that teaches the model to predict next tokens.

### The Training Loop Explained

```python
for step in range(max_steps):
    # 1. Get a batch of data
    x, y = get_batch(train_data)
    # x: input tokens, y: target tokens (x shifted by 1)

    # 2. Forward pass: compute predictions
    logits, loss = model(x, targets=y)
    # logits: probability distribution over vocabulary
    # loss: how wrong we were (cross-entropy)

    # 3. Backward pass: compute gradients
    loss.backward()
    # PyTorch computes ∂loss/∂weight for every weight

    # 4. Update weights
    optimizer.step()
    # weight = weight - learning_rate × gradient

    # 5. Zero gradients for next iteration
    optimizer.zero_grad()
```

### Understanding Loss

**Cross-Entropy Loss**: Measures how wrong the predictions are

```
If true next token is "cat" (index 42):
  - Perfect prediction: loss ≈ 0
  - Random prediction: loss ≈ ln(vocab_size) ≈ 10.8
  - Confident wrong prediction: loss → ∞
```

**Perplexity**: A more intuitive metric
```python
perplexity = exp(loss)
# Interpretation: "How many tokens is the model confused between?"
# Perplexity 10 = model is choosing between ~10 likely tokens
# Perplexity 1 = model knows exactly what comes next
```

### How to Run Training

**Option A: Quick test (CPU, small model)**
```bash
cd nanoGPT
python nanogpt.py
```

**Option B: Full training (GPU)**
```bash
cd nanoGPT
python nanogpt.py --grad_accum=4
```

**Option C: Large model with FSDP (multi-GPU)**
```bash
cd nanoGPT
torchrun --nproc_per_node=4 nanogpt.py --fsdp
```

### Monitoring Training

Watch for these patterns:

**Healthy training:**
```
Step 0:    loss=10.82 (random initialization)
Step 100:  loss=6.54  (learning basic patterns)
Step 500:  loss=4.12  (learning structure)
Step 1000: loss=2.89  (learning content)
Step 2000: loss=2.31  (refining)
```

**Warning signs:**
- Loss goes up → Learning rate too high
- Loss stuck → Learning rate too low or dead neurons
- Loss NaN → Numerical instability
- Val loss up, train loss down → Overfitting

### Saving Checkpoints

```python
model.save_checkpoint(
    "checkpoint.pt",
    optimizer=optimizer,
    iter_num=step,
    best_val_loss=val_loss
)
```

**Why checkpoint?**
- Resume after interruption
- Compare different training runs
- Deploy the best version

---

## Step 6: Generate Text

### What You're About to Do

Use the trained model to generate new text.

### How Generation Works

```
Prompt:  "def calculate_"
Step 1:  Model predicts next token → "total"
Step 2:  Append "total" → "def calculate_total"
Step 3:  Model predicts next token → "("
Step 4:  Append "(" → "def calculate_total("
... continue until done
```

### Sampling Strategies

#### Temperature
**What**: Controls randomness of predictions
**Why**: Lower = more focused, higher = more creative

```python
# Temperature scales logits before softmax
logits = logits / temperature

temperature=0.1  # Almost deterministic (picks most likely)
temperature=1.0  # Balanced (natural distribution)
temperature=2.0  # Very random (creative but may be nonsense)
```

#### Top-K Sampling
**What**: Only consider the K most likely tokens
**Why**: Prevents sampling very unlikely tokens

```python
top_k=50  # Only consider top 50 tokens each step
```

#### Top-P (Nucleus) Sampling
**What**: Consider tokens until cumulative probability reaches P
**Why**: Adapts to the shape of the distribution

```python
top_p=0.9  # Consider tokens that make up 90% of probability mass

# If one token has 95% probability → only that token
# If many tokens are similar → consider many
```

### Generating with KV-Cache

**Why cache?** Without cache, generation is O(n²) because each token recomputes attention over all previous tokens.

```python
# Without cache (slow):
for i in range(100):
    output = model(all_tokens_so_far)  # Recomputes everything!

# With cache (fast):
for i in range(100):
    output = model(just_new_token, kv_cache=cache)  # Only new token
```

### How to Generate

```python
from nanoGPT.nanogpt import GPT, GPTConfig, CharTokenizer

# Load model
model, _ = GPT.load_checkpoint("checkpoint.pt")
model.eval()

# Tokenize prompt
tokenizer = CharTokenizer(training_text)
prompt = "def calculate_"
tokens = torch.tensor([tokenizer.encode(prompt)])

# Generate
with torch.no_grad():
    output = model.generate(
        tokens,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        use_cache=True
    )

# Decode and print
print(tokenizer.decode(output[0].tolist()))
```

---

## Step 7: Evaluate and Iterate

### What You're About to Do

Assess model quality and improve it.

### Quantitative Evaluation

**Validation Loss**: Primary metric during training
```bash
# Lower is better, but watch for overfitting
Train loss: 2.1, Val loss: 2.3  # Good (similar)
Train loss: 1.5, Val loss: 3.0  # Overfitting!
```

**Perplexity Benchmarks**:
```
Random:        ~50,000 (vocab size)
After 1 epoch: ~100-500
Well-trained:  ~10-50
State-of-art:  ~5-15
```

### Qualitative Evaluation

Generate samples and assess:

1. **Coherence**: Does it make sense?
2. **Relevance**: Is it on-topic?
3. **Correctness**: Is the code valid?
4. **Style**: Does it match training data?

### Common Issues and Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Underfitting | High loss, boring output | More capacity, more data, longer training |
| Overfitting | Train↓ Val↑, memorized output | More dropout, less capacity, more data |
| Mode collapse | Repetitive output | Higher temperature, nucleus sampling |
| Incoherence | Gibberish | More training, lower temperature |

### Iteration Strategies

1. **Data quality**: Clean your dataset, remove duplicates
2. **Architecture**: Try more layers/heads if underfitting
3. **Training**: Adjust learning rate, try different schedules
4. **Regularization**: Tune dropout if overfitting

---

## Appendix A: Full Training Script

```python
#!/usr/bin/env python3
"""Complete training script with best practices."""

import torch
from nanoGPT.nanogpt import GPT, GPTConfig, get_batch, get_lr

# Configuration
config = GPTConfig(
    vocab_size=256,      # Character-level
    block_size=256,      # Context length
    n_layer=6,           # Transformer blocks
    n_head=6,            # Attention heads
    n_embd=384,          # Embedding dimension
    dropout=0.1,
)

# Load data
with open("datasets/git_training_data_train.jsonl") as f:
    train_data = f.read()
train_tokens = torch.tensor([ord(c) for c in train_data])

# Initialize model
model = GPT(config)
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    device_type='cuda' if torch.cuda.is_available() else 'cpu'
)

# Training loop
max_steps = 5000
for step in range(max_steps):
    # Learning rate schedule
    lr = get_lr(step, warmup_iters=100, lr_decay_iters=max_steps,
                min_lr=3e-5, max_lr=3e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get batch
    x, y = get_batch(train_tokens, batch_size=8, block_size=256,
                     device=model.device)

    # Forward pass
    logits, loss, _ = model(x, targets=y)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update
    optimizer.step()
    optimizer.zero_grad()

    # Log
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, lr={lr:.2e}")

    # Checkpoint
    if step % 1000 == 0:
        model.save_checkpoint(f"ckpt_{step}.pt", optimizer=optimizer,
                              iter_num=step)

print("Training complete!")
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Token** | Smallest unit of text (character, word piece, or word) |
| **Embedding** | Dense vector representation of a token |
| **Attention** | Mechanism for tokens to exchange information |
| **Transformer** | Architecture using self-attention layers |
| **Forward pass** | Computing predictions from inputs |
| **Backward pass** | Computing gradients via chain rule |
| **Gradient** | Direction to adjust weights to reduce loss |
| **Loss** | Number measuring how wrong predictions are |
| **Epoch** | One complete pass through training data |
| **Batch** | Subset of data processed together |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Perplexity** | Exponential of loss; intuitive quality metric |

---

## Appendix C: Recommended Reading

1. **"Attention Is All You Need"** (Vaswani et al., 2017) - The original transformer paper
2. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019) - GPT-2 paper
3. **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022) - Scaling laws
4. **Andrej Karpathy's nanoGPT** - The inspiration for this implementation

---

*Tutorial created for the Opus-code-test project. Last updated: January 2026.*
