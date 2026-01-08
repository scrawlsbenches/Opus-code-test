"""
StakingManager: Commitment Mechanism for Expert Predictions.

Staking allows experts to commit resources to their predictions,
creating a skin-in-the-game incentive for honest confidence estimation.
Overconfident predictions that turn out wrong lose stake; accurate
predictions earn rewards.

Design Philosophy:
    Talk is cheap; commitment is expensive. When experts must stake
    resources on their predictions, they are incentivized to be
    honest about their uncertainty rather than always claiming
    high confidence.

Key Concepts:
    - Stake: Resources committed to a prediction
    - Slashing: Penalty for wrong high-confidence predictions
    - Reward: Bonus for correct predictions
    - Balance: Total available resources for staking

Example:
    >>> staking = StakingManager(initial_stake=100.0)
    >>> expert = MicroExpert(name="test", domain="test")
    >>> stake_id = staking.place_stake(expert, confidence=0.9, amount=50.0)
    >>> # If prediction is wrong...
    >>> staking.resolve_stake(expert, stake_id, correct=False)
    >>> print(staking.get_balance(expert))  # Less than 100.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum, auto
import uuid


class StakeStatus(Enum):
    """Status of a stake."""
    PENDING = auto()  # Stake placed, awaiting resolution
    WON = auto()  # Prediction correct, stake returned with reward
    LOST = auto()  # Prediction wrong, stake slashed
    CANCELLED = auto()  # Stake cancelled before resolution


@dataclass
class Stake:
    """
    A single stake on a prediction.

    Records the amount committed, confidence level, and outcome.
    """
    id: str
    expert_id: str
    amount: float
    confidence: float
    status: StakeStatus = StakeStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    outcome_correct: Optional[bool] = None
    payout: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


class StakingManager:
    """
    Manager for expert staking and commitment.

    The StakingManager maintains balances for experts and handles
    stake placement, resolution, and reward/penalty distribution.

    Staking Rules:
        - Stake amount is proportional to stated confidence
        - Correct predictions return stake + reward
        - Wrong predictions lose stake (slashing)
        - Higher confidence = higher risk/reward
        - Reward = stake * confidence * reward_multiplier
        - Slash = stake * confidence * slash_multiplier

    Example:
        >>> manager = StakingManager(initial_stake=100.0)
        >>> expert = MicroExpert(name="test", domain="test")
        >>>
        >>> # Place confident stake
        >>> stake_id = manager.place_stake(expert, confidence=0.8, amount=40.0)
        >>> print(f"Balance after staking: {manager.get_balance(expert)}")
        >>>
        >>> # Prediction was correct!
        >>> manager.resolve_stake(expert, stake_id, correct=True)
        >>> print(f"Balance after winning: {manager.get_balance(expert)}")
    """

    def __init__(
        self,
        initial_stake: float = 100.0,
        reward_multiplier: float = 0.5,
        slash_multiplier: float = 1.0,
        min_stake: float = 1.0,
        max_stake_fraction: float = 0.5,
    ):
        """
        Initialize the staking manager.

        Args:
            initial_stake: Starting balance for new experts
            reward_multiplier: Multiplier for rewards (stake * conf * this)
            slash_multiplier: Multiplier for slashing (stake * conf * this)
            min_stake: Minimum stake amount
            max_stake_fraction: Maximum fraction of balance that can be staked
        """
        self.initial_stake = initial_stake
        self.reward_multiplier = reward_multiplier
        self.slash_multiplier = slash_multiplier
        self.min_stake = min_stake
        self.max_stake_fraction = max_stake_fraction

        self._balances: Dict[str, float] = {}
        self._stakes: Dict[str, Stake] = {}
        self._expert_stakes: Dict[str, List[str]] = {}  # expert_id -> [stake_ids]
        self._history: List[Stake] = []

    def _get_expert_id(self, expert) -> str:
        """Get expert ID from expert object or string."""
        if isinstance(expert, str):
            return expert
        return getattr(expert, 'id', str(expert))

    def get_balance(self, expert) -> float:
        """
        Get current balance for an expert.

        Args:
            expert: Expert object or ID

        Returns:
            Current balance
        """
        expert_id = self._get_expert_id(expert)
        if expert_id not in self._balances:
            self._balances[expert_id] = self.initial_stake
        return self._balances[expert_id]

    def set_balance(self, expert, amount: float) -> None:
        """Set balance for an expert (for testing)."""
        expert_id = self._get_expert_id(expert)
        self._balances[expert_id] = max(0.0, amount)

    def compute_stake(
        self,
        expert,
        confidence: float,
        base_fraction: float = 0.1,
    ) -> float:
        """
        Compute recommended stake amount based on confidence.

        Higher confidence = higher stake (up to max_stake_fraction).

        Args:
            expert: Expert object or ID
            confidence: Confidence level (0.0 to 1.0)
            base_fraction: Base fraction of balance to stake

        Returns:
            Recommended stake amount
        """
        balance = self.get_balance(expert)

        # Stake scales with confidence
        # Low confidence (0.5) = base_fraction * balance
        # High confidence (1.0) = max_stake_fraction * balance
        fraction = base_fraction + (self.max_stake_fraction - base_fraction) * confidence

        stake = balance * fraction
        return max(self.min_stake, min(stake, balance))

    def place_stake(
        self,
        expert,
        confidence: float,
        amount: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Place a stake on a prediction.

        Args:
            expert: Expert making the prediction
            confidence: Stated confidence (0.0 to 1.0)
            amount: Stake amount (or None to auto-compute)
            context: Optional context for logging

        Returns:
            Stake ID for tracking

        Raises:
            ValueError: If insufficient balance
        """
        expert_id = self._get_expert_id(expert)
        balance = self.get_balance(expert)

        # Compute or validate amount
        if amount is None:
            amount = self.compute_stake(expert, confidence)
        else:
            amount = min(amount, balance * self.max_stake_fraction)
            amount = max(self.min_stake, amount)

        if amount > balance:
            raise ValueError(f"Insufficient balance: {balance} < {amount}")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Create stake
        stake_id = str(uuid.uuid4())[:8]
        stake = Stake(
            id=stake_id,
            expert_id=expert_id,
            amount=amount,
            confidence=confidence,
            context=context or {},
        )

        # Deduct from balance
        self._balances[expert_id] = balance - amount

        # Record stake
        self._stakes[stake_id] = stake
        if expert_id not in self._expert_stakes:
            self._expert_stakes[expert_id] = []
        self._expert_stakes[expert_id].append(stake_id)

        return stake_id

    def resolve_stake(
        self,
        expert,
        stake_id: str,
        correct: bool,
    ) -> float:
        """
        Resolve a stake based on prediction outcome.

        Args:
            expert: Expert who made the prediction
            stake_id: ID of the stake to resolve
            correct: Whether the prediction was correct

        Returns:
            Final payout (positive = gain, negative = loss)

        Raises:
            ValueError: If stake not found or already resolved
        """
        if stake_id not in self._stakes:
            raise ValueError(f"Stake not found: {stake_id}")

        stake = self._stakes[stake_id]
        if stake.status != StakeStatus.PENDING:
            raise ValueError(f"Stake already resolved: {stake.status}")

        expert_id = self._get_expert_id(expert)
        if stake.expert_id != expert_id:
            raise ValueError(f"Stake belongs to {stake.expert_id}, not {expert_id}")

        balance = self.get_balance(expert)

        if correct:
            # Return stake + reward
            # Reward scales with confidence and stake amount
            reward = stake.amount * stake.confidence * self.reward_multiplier
            payout = stake.amount + reward
            stake.status = StakeStatus.WON
        else:
            # Slash stake
            # Loss scales with confidence (overconfidence is penalized more)
            slash = stake.amount * stake.confidence * self.slash_multiplier
            # Return remaining stake minus slash
            remaining = stake.amount - slash
            payout = max(0.0, remaining)  # Can't lose more than staked
            stake.status = StakeStatus.LOST

        # Update balance
        self._balances[expert_id] = balance + payout

        # Record outcome
        stake.resolved_at = datetime.now()
        stake.outcome_correct = correct
        stake.payout = payout - stake.amount  # Net gain/loss

        self._history.append(stake)

        return stake.payout

    def cancel_stake(self, expert, stake_id: str) -> float:
        """
        Cancel a pending stake and return the amount.

        Args:
            expert: Expert who placed the stake
            stake_id: ID of the stake to cancel

        Returns:
            Refunded amount
        """
        if stake_id not in self._stakes:
            raise ValueError(f"Stake not found: {stake_id}")

        stake = self._stakes[stake_id]
        if stake.status != StakeStatus.PENDING:
            raise ValueError(f"Cannot cancel resolved stake: {stake.status}")

        expert_id = self._get_expert_id(expert)
        if stake.expert_id != expert_id:
            raise ValueError(f"Stake belongs to {stake.expert_id}")

        # Refund
        self._balances[expert_id] = self.get_balance(expert) + stake.amount
        stake.status = StakeStatus.CANCELLED
        stake.resolved_at = datetime.now()

        return stake.amount

    def get_stake(self, stake_id: str) -> Optional[Stake]:
        """Get stake by ID."""
        return self._stakes.get(stake_id)

    def get_expert_stakes(
        self,
        expert,
        status: Optional[StakeStatus] = None,
    ) -> List[Stake]:
        """
        Get all stakes for an expert.

        Args:
            expert: Expert object or ID
            status: Filter by status (optional)

        Returns:
            List of stakes
        """
        expert_id = self._get_expert_id(expert)
        stake_ids = self._expert_stakes.get(expert_id, [])
        stakes = [self._stakes[sid] for sid in stake_ids if sid in self._stakes]

        if status:
            stakes = [s for s in stakes if s.status == status]

        return stakes

    def get_pending_stakes(self, expert) -> List[Stake]:
        """Get all pending stakes for an expert."""
        return self.get_expert_stakes(expert, StakeStatus.PENDING)

    def get_total_staked(self, expert) -> float:
        """Get total amount currently staked (pending)."""
        pending = self.get_pending_stakes(expert)
        return sum(s.amount for s in pending)

    def get_stats(self, expert) -> Dict[str, Any]:
        """
        Get staking statistics for an expert.

        Returns:
            Dictionary of statistics
        """
        expert_id = self._get_expert_id(expert)
        stakes = self.get_expert_stakes(expert)

        won = [s for s in stakes if s.status == StakeStatus.WON]
        lost = [s for s in stakes if s.status == StakeStatus.LOST]
        pending = [s for s in stakes if s.status == StakeStatus.PENDING]

        total_won = sum(s.payout for s in won if s.payout)
        total_lost = sum(abs(s.payout) for s in lost if s.payout)

        return {
            'expert_id': expert_id,
            'balance': self.get_balance(expert),
            'total_stakes': len(stakes),
            'won': len(won),
            'lost': len(lost),
            'pending': len(pending),
            'total_won': total_won,
            'total_lost': total_lost,
            'net_pnl': total_won - total_lost,
            'win_rate': len(won) / len([s for s in stakes if s.status != StakeStatus.PENDING]) if stakes else 0.0,
            'pending_exposure': sum(s.amount for s in pending),
        }

    def get_leaderboard(self, metric: str = 'balance') -> List[tuple]:
        """
        Get expert leaderboard.

        Args:
            metric: Sorting metric ('balance', 'net_pnl', 'win_rate')

        Returns:
            List of (expert_id, metric_value) tuples
        """
        experts = list(self._balances.keys())
        stats = [(eid, self.get_stats(eid)) for eid in experts]

        if metric == 'balance':
            return sorted([(eid, s['balance']) for eid, s in stats], key=lambda x: x[1], reverse=True)
        elif metric == 'net_pnl':
            return sorted([(eid, s['net_pnl']) for eid, s in stats], key=lambda x: x[1], reverse=True)
        elif metric == 'win_rate':
            return sorted([(eid, s['win_rate']) for eid, s in stats], key=lambda x: x[1], reverse=True)
        else:
            return sorted([(eid, s['balance']) for eid, s in stats], key=lambda x: x[1], reverse=True)

    def get_summary(self) -> Dict[str, Any]:
        """Get overall staking summary."""
        total_balance = sum(self._balances.values())
        total_pending = sum(
            s.amount for s in self._stakes.values()
            if s.status == StakeStatus.PENDING
        )

        return {
            'total_experts': len(self._balances),
            'total_balance': total_balance,
            'total_pending': total_pending,
            'total_stakes': len(self._stakes),
            'resolved_stakes': len(self._history),
        }
