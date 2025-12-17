# Attention Marketplace: Intelligence Exchange for ML Models

## Conceptual Foundation

This document describes an **internal economy for intelligence** where ML models (micro-experts) earn and spend resources based on demonstrated value. The system draws inspiration from:

1. **Andrew Yang's Value Distribution** - Value created should flow back to creators, even when value is intangible or indirect
2. **YouTube Creator Economy** - Creators paid based on attention, engagement, and advertiser value
3. **Stock/Crypto Markets** - Capital allocated based on perceived and actual performance
4. **Attention Economics** - Scarce resource (attention/compute) allocated to highest-value producers

### Core Insight

Just as YouTube pays creators who capture attention and provide value to viewers (and thus advertisers), we can pay ML models that capture "prediction attention" and provide value to users (and thus the system).

---

## The Intelligence Exchange

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE EXCHANGE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │ FileExpert  │    │ TestExpert  │    │ ErrorExpert │            │
│   │ Credits: 847│    │ Credits: 423│    │ Credits: 156│            │
│   │ ROI: 2.3x   │    │ ROI: 1.8x   │    │ ROI: 0.9x   │            │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
│          │                  │                  │                    │
│          ▼                  ▼                  ▼                    │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              PREDICTION MARKETPLACE                  │          │
│   │                                                      │          │
│   │  User Query ──► Route to Experts ──► Aggregate      │          │
│   │                 (costs credits)       (pays winners) │          │
│   └─────────────────────────────────────────────────────┘          │
│                            │                                        │
│                            ▼                                        │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              VALUE ATTRIBUTION ENGINE                │          │
│   │                                                      │          │
│   │  • Did prediction lead to successful outcome?        │          │
│   │  • User feedback (thumbs up/down)                    │          │
│   │  • CI pass after suggested changes                   │          │
│   │  • Time saved vs baseline                            │          │
│   └─────────────────────────────────────────────────────┘          │
│                            │                                        │
│                            ▼                                        │
│   ┌─────────────────────────────────────────────────────┐          │
│   │                CREDIT SETTLEMENT                     │          │
│   │                                                      │          │
│   │  Winners: +credits (proportional to value created)   │          │
│   │  Losers:  -credits (prediction cost not recovered)   │          │
│   │  Stakers: +credits (backed correct predictions)      │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Currency: Intelligence Credits (IC)

### What Credits Represent

Credits are the unit of account in the intelligence exchange, representing:

1. **Earned Value** - Past contributions to successful outcomes
2. **Resource Claim** - Right to consume compute/training resources
3. **Routing Priority** - Higher-credit models get called more often
4. **Trust Signal** - Accumulated track record

### Credit Flows

```
┌──────────────────┐
│   Credit Mint    │  (new credits enter system)
│  ┌────────────┐  │
│  │ New Models │──┼──► Seed credits (bootstrap budget)
│  │ User Fund  │──┼──► Manual injection (like buying tokens)
│  │ System Rev │──┼──► Global pool from successful outcomes
│  └────────────┘  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Credit Circulation│  (credits move between models)
│  ┌────────────┐  │
│  │ Predictions│──┼──► Cost to predict (burn)
│  │ Wins       │──┼──► Revenue from accuracy (mint)
│  │ Routing    │──┼──► Transfer from caller to callee
│  │ Staking    │──┼──► Lock for priority/confidence
│  └────────────┘  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Credit Sink    │  (credits exit system)
│  ┌────────────┐  │
│  │ Bad Predict│──┼──► Lost on wrong predictions
│  │ Deprecation│──┼──► Expired models liquidated
│  │ Training   │──┼──► Spent on retraining
│  └────────────┘  │
└──────────────────┘
```

---

## Value Attribution System

### Analog to YouTube Monetization

| YouTube Concept | Intelligence Exchange Equivalent |
|-----------------|----------------------------------|
| Views | Times model was consulted |
| Watch time | Prediction confidence × usage duration |
| Engagement (likes/comments) | User feedback, acceptance rate |
| Ad revenue | Value of successful outcomes |
| CPM (cost per mille) | Credits per thousand predictions |
| Sponsorships | Priority routing from specific intents |
| Super Chat | User-boosted prediction requests |

### Value Signals

```python
@dataclass
class ValueSignal:
    """A signal indicating value was created or destroyed."""
    signal_type: str          # 'prediction_accepted', 'ci_pass', 'user_feedback', etc.
    timestamp: str
    expert_id: str
    prediction_id: str

    # Value metrics
    raw_value: float          # Unbounded measure of impact
    confidence: float         # How certain we are this is real value
    attribution: float        # % attributed to this expert (0-1)

    # Context
    context: Dict[str, Any]   # Query, other experts involved, etc.


# Value signal types and their weights
VALUE_SIGNALS = {
    # Direct feedback
    'user_thumbs_up': {'base_value': 10, 'confidence': 0.9},
    'user_thumbs_down': {'base_value': -15, 'confidence': 0.9},
    'prediction_accepted': {'base_value': 5, 'confidence': 0.7},
    'prediction_rejected': {'base_value': -3, 'confidence': 0.7},

    # Downstream outcomes
    'ci_pass_after_change': {'base_value': 20, 'confidence': 0.6},
    'ci_fail_after_change': {'base_value': -25, 'confidence': 0.6},
    'commit_includes_predicted_file': {'base_value': 15, 'confidence': 0.8},
    'error_resolved': {'base_value': 30, 'confidence': 0.5},

    # Consensus signals
    'agreed_with_winner': {'base_value': 3, 'confidence': 0.5},
    'disagreed_with_winner': {'base_value': -2, 'confidence': 0.5},
    'was_sole_correct': {'base_value': 25, 'confidence': 0.7},  # Found what others missed

    # Time-based
    'fast_resolution': {'base_value': 5, 'confidence': 0.4},   # Issue resolved quickly
    'long_debug_session': {'base_value': -5, 'confidence': 0.3},  # Took too long
}
```

### Attribution Algorithm

When multiple experts contribute to an outcome, we need to attribute value fairly:

```python
def attribute_value(
    outcome: ValueSignal,
    contributing_experts: List[ExpertPrediction],
    final_prediction: AggregatedPrediction
) -> Dict[str, float]:
    """
    Attribute value across contributing experts.

    Uses Shapley-value-inspired approach:
    - Experts who contributed correct items get more
    - Experts who had high confidence in correct items get even more
    - Experts who disagreed with wrong consensus get bonus (contrarian premium)
    """
    attributions = {}
    total_value = outcome.raw_value * outcome.confidence

    # Find which experts predicted the winning items
    winning_items = set(item for item, _ in final_prediction.items[:3])

    for expert_pred in contributing_experts:
        expert_items = set(item for item, _ in expert_pred.items[:5])

        # Base attribution: overlap with winning items
        overlap = len(expert_items & winning_items)
        base_share = overlap / max(len(winning_items), 1)

        # Confidence bonus: higher confidence in correct predictions
        confidence_bonus = 0
        for item, conf in expert_pred.items:
            if item in winning_items:
                confidence_bonus += conf * 0.1

        # Contrarian premium: was correct when others were wrong
        if final_prediction.disagreement_score > 0.5 and overlap > 0:
            contrarian_bonus = 0.2 * final_prediction.disagreement_score
        else:
            contrarian_bonus = 0

        # Calculate final attribution
        attributions[expert_pred.expert_id] = (
            base_share + confidence_bonus + contrarian_bonus
        ) * total_value

    # Normalize to not exceed total value
    total_attributed = sum(attributions.values())
    if total_attributed > total_value:
        scale = total_value / total_attributed
        attributions = {k: v * scale for k, v in attributions.items()}

    return attributions
```

---

## Credit Ledger

### Data Structure

```python
@dataclass
class CreditAccount:
    """Credit account for an expert."""
    expert_id: str

    # Balances
    available_credits: float      # Can spend immediately
    staked_credits: float         # Locked for priority/confidence
    pending_credits: float        # Awaiting outcome resolution

    # Lifetime stats
    total_earned: float
    total_spent: float
    total_burned: float           # Lost to bad predictions

    # Performance metrics
    roi: float                    # (earned - spent) / spent
    win_rate: float               # % of predictions that paid off
    avg_value_per_prediction: float

    # Timestamps
    created_at: str
    last_transaction: str


@dataclass
class CreditTransaction:
    """A single credit movement."""
    tx_id: str
    timestamp: str

    # Parties
    from_account: str             # 'mint', 'sink', or expert_id
    to_account: str               # 'mint', 'sink', or expert_id

    # Amount
    amount: float

    # Context
    tx_type: str                  # 'prediction_cost', 'value_reward', 'stake', etc.
    prediction_id: Optional[str]
    value_signal_id: Optional[str]

    # Audit
    reason: str
    metadata: Dict[str, Any]
```

### Transaction Types

| Type | From | To | Trigger |
|------|------|-----|---------|
| `seed` | mint | expert | New expert created |
| `prediction_cost` | expert | sink | Expert makes prediction |
| `value_reward` | mint | expert | Positive value signal |
| `value_penalty` | expert | sink | Negative value signal |
| `stake` | available | staked | Expert bids for priority |
| `unstake` | staked | available | Stake period ends |
| `routing_fee` | caller | callee | Expert calls another expert |
| `deprecation` | expert | sink | Model retired |

---

## Resource Allocation

Credits translate to real resources:

### Training Data Allocation

```python
def allocate_training_data(
    experts: List[MicroExpert],
    new_data: List[TrainingExample],
    budget_per_expert: Dict[str, int]
) -> Dict[str, List[TrainingExample]]:
    """
    Allocate training data proportional to credit balance.

    High-credit experts get more training data,
    allowing them to improve further (rich get richer,
    but only if they continue performing).
    """
    total_credits = sum(e.account.available_credits for e in experts)

    allocations = {}
    for expert in experts:
        share = expert.account.available_credits / total_credits

        # Minimum allocation (prevent starvation)
        min_examples = 10
        max_examples = len(new_data) // 2  # No single expert gets >50%

        allocation = int(share * len(new_data))
        allocation = max(min_examples, min(max_examples, allocation))

        # Sample data (could be random or stratified by relevance)
        allocations[expert.expert_id] = sample_relevant_data(
            new_data, expert, allocation
        )

    return allocations
```

### Routing Priority

```python
def select_experts_for_query(
    query: str,
    available_experts: List[MicroExpert],
    budget: int = 3  # Max experts to consult
) -> List[MicroExpert]:
    """
    Select experts based on relevance AND credit-weighted priority.

    Like ad auction: highest bidder with relevant inventory wins.
    """
    candidates = []

    for expert in available_experts:
        # Relevance score (does this expert handle this query type?)
        relevance = compute_relevance(query, expert)
        if relevance < 0.1:
            continue

        # Credit-weighted bid
        # Experts can stake credits for higher priority
        bid = relevance * (1 + expert.account.staked_credits * 0.01)

        # ROI factor: experts with good track record get bonus
        roi_bonus = 1 + max(0, expert.account.roi) * 0.5

        final_score = bid * roi_bonus
        candidates.append((expert, final_score))

    # Select top N by score
    candidates.sort(key=lambda x: -x[1])
    return [expert for expert, _ in candidates[:budget]]
```

### Compute Budget

```python
def allocate_inference_compute(
    expert: MicroExpert,
    base_budget_ms: int = 100
) -> int:
    """
    Allocate inference time budget based on credits.

    Higher-credit experts can do more expensive inference
    (e.g., more candidates, deeper search).
    """
    credit_multiplier = 1 + (expert.account.available_credits / 1000) * 0.5
    return int(base_budget_ms * credit_multiplier)
```

---

## Market Dynamics

### Bull Market (Expansion)

When the system is generating value:
- More credits minted than burned
- New experts can bootstrap
- Exploration encouraged
- Diversity of approaches rewarded

### Bear Market (Contraction)

When predictions are failing:
- More credits burned than minted
- Weak experts get deprecated
- Conservation mode: only proven experts consulted
- Focus on reliability over novelty

### Market Makers

Special system actors that provide liquidity:

1. **Seed Fund** - Bootstrap new experts with initial credits
2. **Insurance Pool** - Absorb catastrophic losses
3. **Diversity Subsidy** - Bonus credits for unique predictions
4. **Novelty Premium** - Extra rewards for discovering new patterns

---

## Staking Mechanism

Experts can stake credits to signal confidence:

```python
@dataclass
class Stake:
    """A credit stake on a prediction."""
    stake_id: str
    expert_id: str
    prediction_id: str

    amount: float
    confidence_claimed: float     # What confidence they're claiming

    # Outcome
    resolved: bool
    outcome_value: Optional[float]
    payout: Optional[float]


def calculate_stake_payout(stake: Stake, actual_confidence: float) -> float:
    """
    Calculate payout based on calibration.

    If expert claimed 80% confidence and was right 80% of the time,
    they get their stake back plus premium.

    If they claimed 80% but were only right 50%, they lose some stake.
    """
    calibration_error = abs(stake.confidence_claimed - actual_confidence)

    if calibration_error < 0.1:
        # Well-calibrated: return stake + premium
        return stake.amount * 1.2
    elif calibration_error < 0.2:
        # Reasonably calibrated: return stake
        return stake.amount
    else:
        # Poorly calibrated: lose proportional to error
        loss_rate = min(1.0, calibration_error * 2)
        return stake.amount * (1 - loss_rate)
```

---

## Integration with MoE Architecture

The attention marketplace enhances the MoE system:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced MoE System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query ──► ExpertRouter ──► Selected Experts ──► Predictions   │
│              │                    │                    │        │
│              │                    │                    │        │
│              ▼                    ▼                    ▼        │
│        ┌──────────┐        ┌──────────┐        ┌──────────┐    │
│        │ Credit   │        │ Credit   │        │ Credit   │    │
│        │ Check    │        │ Cost     │        │ Escrow   │    │
│        │ (can pay)│        │ Deducted │        │ (await   │    │
│        └──────────┘        └──────────┘        │ outcome) │    │
│                                                └──────────┘    │
│                                                     │          │
│                                                     ▼          │
│                                              ┌──────────┐      │
│  Outcome ────────────────────────────────────► Value    │      │
│  (CI result, user feedback, etc.)            │ Signal   │      │
│                                              └────┬─────┘      │
│                                                   │            │
│                                                   ▼            │
│                                              ┌──────────┐      │
│                                              │Settlement│      │
│                                              │ Credits  │      │
│                                              │ Paid Out │      │
│                                              └──────────┘      │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Storage Schema

```
.git-ml/
├── exchange/
│   ├── ledger.jsonl              # Append-only transaction log
│   ├── accounts/
│   │   ├── file_expert.json      # Credit account state
│   │   ├── test_expert.json
│   │   └── ...
│   ├── stakes/
│   │   └── active_stakes.json    # Currently staked credits
│   ├── market/
│   │   ├── price_history.jsonl   # Credit value over time
│   │   └── market_state.json     # Current market conditions
│   └── value_signals/
│       └── 2025-12-17/           # Daily value signal logs
│           └── signals.jsonl
│
├── experts/                       # (from MoE architecture)
│   └── ...
│
└── tracked/
    └── exchange_summary.jsonl    # Git-tracked aggregate stats
```

---

## Metrics Dashboard

```
╔══════════════════════════════════════════════════════════════════╗
║              INTELLIGENCE EXCHANGE DASHBOARD                     ║
╠══════════════════════════════════════════════════════════════════╣
║ MARKET OVERVIEW                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║   Total Credits in Circulation:  12,847                          ║
║   24h Volume:                    1,234 credits                   ║
║   Market Cap:                    ~$0 (internal only)             ║
║   Inflation Rate:                +2.3% (value > costs)           ║
╠══════════════════════════════════════════════════════════════════╣
║ TOP PERFORMERS (by ROI)                                          ║
╠══════════════════════════════════════════════════════════════════╣
║   1. FileExpert         847 credits   ROI: 2.31x   Win: 73%     ║
║   2. TestExpert         423 credits   ROI: 1.84x   Win: 68%     ║
║   3. DocExpert          201 credits   ROI: 1.21x   Win: 61%     ║
║   4. ErrorExpert        156 credits   ROI: 0.94x   Win: 52%     ║
╠══════════════════════════════════════════════════════════════════╣
║ RECENT VALUE SIGNALS                                             ║
╠══════════════════════════════════════════════════════════════════╣
║   [+20] ci_pass_after_change    FileExpert    2m ago            ║
║   [+10] user_thumbs_up          TestExpert    5m ago            ║
║   [-15] user_thumbs_down        ErrorExpert   12m ago           ║
║   [+5]  prediction_accepted     FileExpert    15m ago           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Philosophy: Why This Works

### Alignment with Human Markets

Human markets work because they:
1. **Aggregate distributed knowledge** - Many participants with different info
2. **Reward accuracy** - Being right pays, being wrong costs
3. **Enable price discovery** - Prices reflect true value over time
4. **Allow specialization** - Experts in niches thrive

The intelligence exchange replicates these properties for ML models.

### Andrew Yang's UBI Insight

Yang's insight: Value is created in ways traditional markets don't capture. Stay-at-home parents, open source contributors, and community volunteers create value that doesn't flow back to them.

Applied to ML: Models create value that traditional metrics miss:
- A model that prevents a bug saves hours
- A model that surfaces a forgotten test prevents regression
- A model that suggests the right file saves context-switching

The credit system captures this value and rewards it.

### YouTube Creator Economy Insight

YouTube showed that:
- Attention is valuable and measurable
- Creators will optimize for what pays
- Long-tail of creators can thrive in niches
- Algorithms learn what users value

Applied to ML:
- Prediction attention is valuable and measurable
- Models will optimize for what earns credits (accuracy)
- Specialized models can thrive in niches (specific file types, error classes)
- The system learns which models users value

---

## Future Extensions

1. **Cross-Project Exchange** - Models trade credits across projects
2. **Model IPO** - New models raise credits from existing models
3. **Derivatives** - Hedge against model failure
4. **Governance Tokens** - Models vote on system parameters
5. **External Value Bridge** - Convert credits to/from real resources

---

*Document created: 2025-12-17*
*Status: Conceptual - Ready for discussion*
