# Vocabulary Management Design

## Problem Statement

The current experiment CLI builds vocabulary per-document, creating a rigid coupling between trained model weights and the specific tokens seen during training. This prevents:
- Fine-tuning on new documents
- Iterative training across multiple documents
- Graceful handling of vocabulary changes over time

## Design Goals

1. **Extendable vocabulary** - Start with corpus, grow as needed
2. **Strict by default** - Error on OOV, don't silently degrade
3. **Transparent `<UNK>` handling** - If used, track when and how
4. **Model-vocabulary coupling** - Model knows its vocab, detects changes
5. **Future-proof** - Support code patterns and unseen tokens eventually

## Architecture

### Vocabulary File Format

```json
{
  "version": 1,
  "created_at": "2026-01-15T12:00:00Z",
  "source_files": [
    "samples/unix_evolution.txt",
    "samples/other_doc.txt"
  ],
  "config": {
    "min_freq": 1,
    "lowercase": true,
    "embedding_scale": 0.35
  },
  "tokens": {
    "token_to_id": {
      "unix": 0,
      "bell": 1,
      "labs": 2
    },
    "id_to_token": ["unix", "bell", "labs"]
  },
  "special_tokens": {
    "<PAD>": -1,
    "<UNK>": -2
  },
  "extensions": [
    {
      "added_at": "2026-01-16T10:00:00Z",
      "tokens": ["linux", "kernel"],
      "reason": "New document: samples/linux_history.txt"
    }
  ],
  "unk_usage_log": []
}
```

### Checkpoint-Vocabulary Relationship

```
checkpoint.pkl
├── parameters: {...}
├── optimizer_state: {...}
├── scheduler_state: {...}
├── vocab_reference:
│   ├── path: "experiments/vocab/corpus_v1.json"
│   ├── hash: "sha256:abc123..."
│   └── token_count: 76
└── epoch: 500
```

On resume:
1. Load checkpoint
2. Verify vocab file exists at referenced path
3. Verify hash matches (detect modifications)
4. If mismatch: ERROR with clear explanation

### OOV Handling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Token Encountered                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ In vocabulary?  │
                    └─────────────────┘
                       │           │
                      YES          NO
                       │           │
                       ▼           ▼
                 ┌─────────┐  ┌──────────────────┐
                 │ Use ID  │  │ --extend-vocab?  │
                 └─────────┘  └──────────────────┘
                                  │           │
                                 YES          NO
                                  │           │
                                  ▼           ▼
                           ┌───────────┐  ┌─────────────────┐
                           │ Add token │  │ --allow-unk?    │
                           │ + random  │  └─────────────────┘
                           │ embedding │      │           │
                           └───────────┘     YES          NO
                                              │           │
                                              ▼           ▼
                                        ┌──────────┐  ┌───────┐
                                        │ Map to   │  │ ERROR │
                                        │ <UNK> +  │  │ List  │
                                        │ LOG it   │  │ OOV   │
                                        └──────────┘  └───────┘
```

Default behavior: **ERROR** with list of OOV tokens
User must explicitly opt-in to `--extend-vocab` or `--allow-unk`

### `<UNK>` Transparency

When `--allow-unk` is used:
- Log every `<UNK>` substitution to `unk_usage_log` in vocab file
- Print summary after training: "X tokens mapped to <UNK>"
- Store in checkpoint for auditability

```json
"unk_usage_log": [
  {
    "timestamp": "2026-01-16T10:30:00Z",
    "document": "samples/new_doc.txt",
    "tokens": ["kubernetes", "containerization"],
    "count": 15
  }
]
```

## CLI Interface

### Creating Vocabulary

```bash
# From single file
python -m cortical.experiments.cli vocab create \
    --from samples/unix_evolution.txt \
    --output experiments/vocab/unix.json

# From directory (all .txt files)
python -m cortical.experiments.cli vocab create \
    --from samples/ \
    --output experiments/vocab/corpus.json \
    --min-freq 1

# Inspect vocabulary
python -m cortical.experiments.cli vocab inspect \
    experiments/vocab/corpus.json

# Extend existing vocabulary
python -m cortical.experiments.cli vocab extend \
    --vocab experiments/vocab/corpus.json \
    --from samples/new_doc.txt
```

### Training with Vocabulary

```bash
# Initial training
python -m cortical.experiments.cli run \
    --name my-experiment \
    --input samples/doc1.txt \
    --vocab experiments/vocab/corpus.json \
    --epochs 500

# Resume on same document (vocab unchanged)
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --epochs 1000

# Train on different document (same vocab)
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/doc2.txt \
    --epochs 500

# New document with OOV - will ERROR by default
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/new_doc.txt
# ERROR: 3 out-of-vocabulary tokens found: ['linux', 'kernel', 'gnu']
# Use --extend-vocab to add them, or --allow-unk to map to <UNK>

# Extend vocab and continue
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/new_doc.txt \
    --extend-vocab
# WARNING: Extended vocabulary with 3 new tokens (random embeddings)
# Vocabulary saved to: experiments/vocab/corpus.json
```

## Iterative Training Workflow

### Scenario: New Document Added

```bash
# 1. Check what's new
python -m cortical.experiments.cli vocab diff \
    --vocab experiments/vocab/corpus.json \
    --document samples/new_doc.txt
# New tokens: ['linux', 'kernel', 'gnu']
# Missing from document: ['pdp', 'minicomputer']

# 2. Decision point:
#    a) Extend vocab (recommended if new terms are meaningful)
#    b) Use <UNK> (if terms are noise/typos)
#    c) Edit document to use existing vocab

# 3. Extend and train
python -m cortical.experiments.cli vocab extend \
    --vocab experiments/vocab/corpus.json \
    --from samples/new_doc.txt

python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/new_doc.txt
```

### Scenario: Document Modified

```bash
# Check impact
python -m cortical.experiments.cli vocab diff \
    --vocab experiments/vocab/corpus.json \
    --document samples/modified_doc.txt

# If new tokens: extend vocab first
# If only removed tokens: no action needed (embeddings stay, unused)
```

### Scenario: Vocab Becomes Stale

```bash
# Audit vocabulary usage across corpus
python -m cortical.experiments.cli vocab audit \
    --vocab experiments/vocab/corpus.json \
    --corpus samples/

# Output:
# Tokens in vocab: 156
# Tokens used in corpus: 142
# Unused tokens: 14 (wasting 14 embedding slots)
# Tokens in corpus but not vocab: 0

# Optional: rebuild vocab (requires retraining from scratch)
python -m cortical.experiments.cli vocab rebuild \
    --from samples/ \
    --output experiments/vocab/corpus_v2.json
```

## Implementation Plan

### Phase 1: Core Vocabulary Management
- [ ] `Vocabulary` class with save/load/extend methods
- [ ] `vocab create` command
- [ ] `vocab inspect` command
- [ ] Modify `run` to accept `--vocab` parameter
- [ ] Store vocab reference in checkpoint

### Phase 2: OOV Handling
- [ ] OOV detection before training starts
- [ ] `--extend-vocab` flag with embedding expansion
- [ ] `--allow-unk` flag with logging
- [ ] `vocab diff` command

### Phase 3: Iterative Training
- [ ] `vocab extend` command
- [ ] `vocab audit` command
- [ ] Checkpoint-vocab hash verification
- [ ] Clear error messages for vocab mismatches

### Phase 4: Future Enhancements
- [ ] BPE integration (optional, for code/mixed content)
- [ ] Vocabulary compression (merge rare tokens)
- [ ] Pre-trained embeddings import

## Open Questions

1. **Embedding initialization for new tokens**: Random with same scale, or something smarter?
   - Could average embeddings of similar existing tokens
   - Could use character-level features
   - Start simple: random with `EMBEDDING_INIT_SCALE`

2. **Vocabulary versioning**: Extend in-place or create new file?
   - Proposal: Extend in-place but track history in `extensions` array
   - Hash changes, so checkpoints know vocab was modified

3. **Maximum vocab size**: Should there be a limit?
   - Larger vocab = larger embedding matrix = more memory
   - Could warn if vocab exceeds threshold (e.g., 10,000 tokens)

4. **Case sensitivity**: Currently lowercase by default
   - Code will need case sensitivity
   - Make configurable in vocab config

## Notes

- BPE exists in `cortical/cognitive/tokenizer_storage.py` but not integrated with experiments
- Current tokenizer is simple word-based (`cortical/experiments/__init__.py:tokenize`)
- `EMBEDDING_INIT_SCALE = 0.35` must be consistent across vocab extensions
