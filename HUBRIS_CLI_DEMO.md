# Hubris MoE CLI Demo

## Overview

The Hubris CLI (`scripts/hubris_cli.py`) provides a command-line interface for the Mixture of Experts system.

## Installation

No installation needed - the CLI is a standalone Python script.

```bash
python scripts/hubris_cli.py --help
```

## Commands

### 1. `train` - Train All Experts

Trains experts from collected ML data in `.git-ml/`.

```bash
# Train on all commits
python scripts/hubris_cli.py train

# Train on last 100 commits only
python scripts/hubris_cli.py train --commits 100

# Include transcript data for EpisodeExpert
python scripts/hubris_cli.py train --commits 50 --transcripts
```

**Sample Output:**
```
============================================================
TRAINING HUBRIS MoE EXPERTS
============================================================

Loading training data...
  Commits: 50

Creating fresh experts...
Created 4 fresh experts

Training experts...
Trained TestExpert on 50 commits

Training Results:
  file            ✗ Failed
  test            ✓ Success

Saving experts to .git-ml/models/hubris...
Saved 4 experts to .git-ml/models/hubris

Expert Statistics:
  file:
    Commits:  0
    Sessions: 0
    Version:  1.1.0
  test:
    Commits:  0
    Sessions: 0
    Version:  1.0.0

✓ Training complete!
```

### 2. `predict` - Get Predictions

Get file predictions for a task description.

```bash
# Predict from command line
python scripts/hubris_cli.py predict "Add authentication feature"

# Predict from file
python scripts/hubris_cli.py predict --file task.txt

# Show top 5 files only
python scripts/hubris_cli.py predict "Fix bug" --top 5
```

**Sample Output:**
```
🎯 MoE Prediction for: "Fix authentication bug"

Top Files (by confidence):
   1. cortical/auth.py                                       0.823  [FileExpert: 0.85, TestExpert: 0.80]
   2. tests/test_auth.py                                     0.756  [FileExpert: 0.72, TestExpert: 0.79]
   3. cortical/session.py                                    0.612  [FileExpert: 0.65, ErrorExpert: 0.57]

Expert Contributions:
  • file             weight: 0.42  balance:  156.3  boost: 1.05x
  • test             weight: 0.31  balance:  134.2  boost: 1.03x
  • error            weight: 0.27  balance:  109.5  boost: 1.01x

Overall Confidence: 0.730
Expert Disagreement: 0.125
```

### 3. `stats` - Show Expert Statistics

Display detailed statistics for all experts or a specific expert.

```bash
# Show all experts
python scripts/hubris_cli.py stats

# Show specific expert
python scripts/hubris_cli.py stats --expert file
```

**Sample Output:**
```
============================================================
HUBRIS MoE EXPERT STATISTICS
============================================================

FILE EXPERT
------------------------------------------------------------
  Expert ID:     file_expert
  Version:       1.1.0
  Created:       2025-12-18T03:07:08.928598

  Training Data:
    Commits:     403
    Sessions:    12

  Credit Account:
    Balance:     156.30
    Transactions: 24

  Recent Transactions:
    +15.0               correct_prediction              (balance: 156.3)
    -5.0                wrong_prediction                (balance: 141.3)
    +20.0               high_confidence_correct         (balance: 146.3)
    +10.0               correct_prediction              (balance: 126.3)
    -3.0                low_confidence_wrong            (balance: 116.3)

  Performance Metrics:
    MRR:          0.4321
    Recall@5:     0.6543
    Recall@10:    0.7812
    Precision@1:  0.3145
```

### 4. `leaderboard` - Expert Rankings

Show experts ranked by credit balance.

```bash
# Default: top 10 experts
python scripts/hubris_cli.py leaderboard

# Show top 5
python scripts/hubris_cli.py leaderboard --top 5
```

**Sample Output:**
```
============================================================
HUBRIS MoE EXPERT LEADERBOARD
============================================================

Rank   Expert               Balance      Transactions
------------------------------------------------------------
1      file                   156.30              24
2      test                   134.20              18
3      error                  109.50              15
4      episode                 87.30              12

------------------------------------------------------------
Total Credits in System: 487.30
Total Accounts: 4
```

### 5. `evaluate` - Evaluate Accuracy

Test expert predictions against actual commit data.

```bash
# Evaluate on last 20 commits
python scripts/hubris_cli.py evaluate --commits 20

# Evaluate on last 50 commits
python scripts/hubris_cli.py evaluate --commits 50
```

**Sample Output:**
```
============================================================
EVALUATING HUBRIS MoE EXPERTS
============================================================

Loading last 20 commits for evaluation...
Evaluating on 20 commits...

Evaluating file...
  Evaluated on 20 commits
  MRR:          0.4321
  Precision@1:  0.3145
  Recall@5:     0.6543
  Recall@10:    0.7812
  Credits:      +45.0

Evaluating test...
  Evaluated on 20 commits
  MRR:          0.3892
  Precision@1:  0.2750
  Recall@5:     0.5923
  Recall@10:    0.7234
  Credits:      +30.0

============================================================
EVALUATION SUMMARY
============================================================

file:
  MRR:          0.4321
  Precision@1:  0.3145
  Recall@5:     0.6543
  Recall@10:    0.7812

test:
  MRR:          0.3892
  Precision@1:  0.2750
  Recall@5:     0.5923
  Recall@10:    0.7234

✓ Evaluation complete! Credit balances updated.
```

## Features

### ANSI Color Support

The CLI uses color-coded output:
- **Green**: Success, positive credits
- **Red**: Failure, negative credits, warnings
- **Yellow**: Warnings
- **Cyan**: File paths
- **Bold**: Headers and important metrics

### Credit-Weighted Routing

Predictions use the `CreditRouter` which:
- Weights expert contributions by credit balance
- Applies softmax with temperature control
- Enforces minimum weight floors
- Provides confidence boosts to high-performing experts

### Performance-Based Credits

During evaluation, experts earn/lose credits:
- **+10 credits** per correct top-1 prediction
- **-5 credits** per incorrect prediction
- Credits affect future prediction weights

## Architecture

```
hubris_cli.py
├── cmd_train()       - Trains experts from .git-ml data
├── cmd_predict()     - Gets ensemble predictions
├── cmd_stats()       - Shows expert statistics
├── cmd_leaderboard() - Displays credit rankings
└── cmd_evaluate()    - Evaluates on recent commits
```

**Dependencies:**
- `scripts/hubris/expert_consolidator.py` - Expert management
- `scripts/hubris/credit_account.py` - Credit tracking
- `scripts/hubris/credit_router.py` - Credit-weighted routing

## Data Sources

| Command | Data Source |
|---------|-------------|
| `train` | `.git-ml/commits/*.json`, `.git-ml/chats/*.json` |
| `predict` | `.git-ml/models/hubris/*.json` |
| `stats` | `.git-ml/models/hubris/*.json`, `.git-ml/models/hubris/credit_ledger.json` |
| `leaderboard` | `.git-ml/models/hubris/credit_ledger.json` |
| `evaluate` | `.git-ml/commits/*.json`, `.git-ml/models/hubris/*.json` |

## Future Enhancements

- [ ] Add `--format json` for machine-readable output
- [ ] Support loading from specific model directories
- [ ] Add confidence threshold filtering in predictions
- [ ] Implement interactive mode for predictions
- [ ] Add expert comparison mode
- [ ] Export evaluation results to CSV/JSON
- [ ] Visualize credit balance history
