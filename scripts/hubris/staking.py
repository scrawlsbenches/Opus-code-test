#!/usr/bin/env python3
"""
Staking Mechanism for MoE System

Allows experts to stake credits on predictions to earn multiplied rewards
or suffer multiplied losses.

Classes:
    Stake: Individual stake on a prediction
    StakePool: Manager for all active stakes
    StakeStrategy: Risk/reward multiplier strategies
    AutoStaker: Automatic staking decisions based on confidence
"""

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from credit_account import CreditLedger, CreditAccount


@dataclass
class Stake:
    """
    Individual stake on a prediction.

    Attributes:
        stake_id: Unique identifier (UUID)
        expert_id: ID of the expert staking
        prediction_id: ID of the prediction being staked on
        amount: Credits staked
        multiplier: Risk/reward multiplier (1.0 to 3.0)
        timestamp: When stake was placed (Unix timestamp)
        status: Current status (pending, won, lost, cancelled)
        outcome_value: Credits won/lost after resolution (None until resolved)
    """
    stake_id: str
    expert_id: str
    prediction_id: str
    amount: float
    multiplier: float
    timestamp: float
    status: str  # pending, won, lost, cancelled
    outcome_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stake':
        """Load from dict."""
        return cls(**data)


class StakeStrategy(Enum):
    """
    Predefined staking strategies with risk/reward multipliers.

    Attributes:
        CONSERVATIVE: 1.0x multiplier, low risk
        MODERATE: 1.5x multiplier, balanced risk/reward
        AGGRESSIVE: 2.0x multiplier, high risk
        YOLO: 3.0x multiplier, maximum risk
    """
    CONSERVATIVE = 1.0
    MODERATE = 1.5
    AGGRESSIVE = 2.0
    YOLO = 3.0


class StakePool:
    """
    Manages all active stakes in the system.

    Handles stake placement, resolution, and escrow of staked credits.

    Attributes:
        ledger: CreditLedger for checking/deducting balances
        max_stake_ratio: Maximum % of balance that can be staked (default: 0.5)
        min_stake: Minimum stake amount (default: 5.0)
        stakes: Dict mapping stake_id to Stake
    """

    def __init__(
        self,
        ledger: CreditLedger,
        max_stake_ratio: float = 0.5,
        min_stake: float = 5.0
    ):
        """
        Initialize stake pool.

        Args:
            ledger: CreditLedger for managing expert balances
            max_stake_ratio: Max % of balance that can be staked (0.0-1.0)
            min_stake: Minimum stake amount (must be positive)

        Raises:
            ValueError: If max_stake_ratio or min_stake are invalid
        """
        if not 0.0 < max_stake_ratio <= 1.0:
            raise ValueError(f"max_stake_ratio must be in (0.0, 1.0], got {max_stake_ratio}")
        if min_stake <= 0:
            raise ValueError(f"min_stake must be positive, got {min_stake}")

        self.ledger = ledger
        self.max_stake_ratio = max_stake_ratio
        self.min_stake = min_stake
        self.stakes: Dict[str, Stake] = {}

    def place_stake(
        self,
        expert_id: str,
        prediction_id: str,
        amount: float,
        multiplier: float = 1.5
    ) -> Stake:
        """
        Place a stake on a prediction.

        Validates balance and deducts amount from expert's account (held in escrow).

        Args:
            expert_id: ID of expert placing stake
            prediction_id: ID of prediction to stake on
            amount: Credits to stake
            multiplier: Risk/reward multiplier (1.0-3.0)

        Returns:
            Stake object

        Raises:
            ValueError: If amount/multiplier invalid or insufficient balance
        """
        # Validate amount
        if amount < self.min_stake:
            raise ValueError(
                f"Stake amount {amount} below minimum {self.min_stake}"
            )

        # Validate multiplier
        if not 1.0 <= multiplier <= 3.0:
            raise ValueError(
                f"Multiplier must be in [1.0, 3.0], got {multiplier}"
            )

        # Get account and check balance
        account = self.ledger.get_or_create_account(expert_id)

        # Check if expert has sufficient balance
        if account.balance < amount:
            raise ValueError(
                f"Insufficient balance: {account.balance} < {amount}"
            )

        # Check stake ratio
        max_allowed = account.balance * self.max_stake_ratio
        if amount > max_allowed:
            raise ValueError(
                f"Stake amount {amount} exceeds max allowed {max_allowed:.2f} "
                f"({self.max_stake_ratio * 100}% of balance)"
            )

        # Deduct amount from account (escrow)
        account.debit(
            amount,
            'stake_placed',
            {
                'prediction_id': prediction_id,
                'multiplier': multiplier,
                'stake_type': 'escrow'
            }
        )

        # Create stake
        stake = Stake(
            stake_id=str(uuid.uuid4()),
            expert_id=expert_id,
            prediction_id=prediction_id,
            amount=amount,
            multiplier=multiplier,
            timestamp=time.time(),
            status='pending',
            outcome_value=None
        )

        self.stakes[stake.stake_id] = stake
        return stake

    def resolve_stake(self, stake_id: str, success: bool) -> float:
        """
        Resolve a stake after prediction outcome is known.

        If success: return amount * multiplier to account
        If failure: stake is forfeit (already deducted)

        Args:
            stake_id: ID of stake to resolve
            success: Whether prediction was successful

        Returns:
            Net gain/loss (positive = profit, negative = loss)

        Raises:
            ValueError: If stake not found or already resolved
        """
        if stake_id not in self.stakes:
            raise ValueError(f"Stake {stake_id} not found")

        stake = self.stakes[stake_id]

        if stake.status != 'pending':
            raise ValueError(
                f"Stake {stake_id} already resolved with status: {stake.status}"
            )

        account = self.ledger.get_or_create_account(stake.expert_id)

        if success:
            # Return original stake + winnings
            payout = stake.amount * stake.multiplier
            account.credit(
                payout,
                'stake_won',
                {
                    'stake_id': stake_id,
                    'prediction_id': stake.prediction_id,
                    'multiplier': stake.multiplier,
                    'original_stake': stake.amount
                }
            )
            stake.status = 'won'
            net_gain = payout - stake.amount  # Profit only
            stake.outcome_value = net_gain

        else:
            # Stake is forfeit (already deducted)
            stake.status = 'lost'
            net_gain = -stake.amount  # Loss
            stake.outcome_value = net_gain

        return net_gain

    def cancel_stake(self, stake_id: str) -> bool:
        """
        Cancel a pending stake and return credits to account.

        Only works if stake is still pending.

        Args:
            stake_id: ID of stake to cancel

        Returns:
            True if cancelled, False if stake not found or not pending

        Raises:
            ValueError: If stake not found
        """
        if stake_id not in self.stakes:
            raise ValueError(f"Stake {stake_id} not found")

        stake = self.stakes[stake_id]

        if stake.status != 'pending':
            return False

        # Return staked amount to account
        account = self.ledger.get_or_create_account(stake.expert_id)
        account.credit(
            stake.amount,
            'stake_cancelled',
            {
                'stake_id': stake_id,
                'prediction_id': stake.prediction_id
            }
        )

        stake.status = 'cancelled'
        stake.outcome_value = 0.0
        return True

    def get_active_stakes(self, expert_id: Optional[str] = None) -> List[Stake]:
        """
        Get all pending stakes, optionally filtered by expert.

        Args:
            expert_id: Optional expert ID to filter by

        Returns:
            List of pending stakes
        """
        active = [
            stake for stake in self.stakes.values()
            if stake.status == 'pending'
        ]

        if expert_id:
            active = [stake for stake in active if stake.expert_id == expert_id]

        return active

    def get_total_staked(self, expert_id: Optional[str] = None) -> float:
        """
        Get sum of all pending stake amounts.

        Args:
            expert_id: Optional expert ID to filter by

        Returns:
            Total credits currently staked
        """
        active = self.get_active_stakes(expert_id)
        return sum(stake.amount for stake in active)

    def get_stake_history(self, expert_id: str) -> List[Stake]:
        """
        Get all stakes (any status) for an expert.

        Args:
            expert_id: Expert ID to get history for

        Returns:
            List of all stakes by this expert, sorted by timestamp (newest first)
        """
        stakes = [
            stake for stake in self.stakes.values()
            if stake.expert_id == expert_id
        ]

        # Sort by timestamp descending (newest first)
        stakes.sort(key=lambda s: s.timestamp, reverse=True)
        return stakes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'max_stake_ratio': self.max_stake_ratio,
            'min_stake': self.min_stake,
            'stakes': {
                stake_id: stake.to_dict()
                for stake_id, stake in self.stakes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], ledger: CreditLedger) -> 'StakePool':
        """
        Load StakePool from dict.

        Args:
            data: Dictionary representation
            ledger: CreditLedger instance to use

        Returns:
            StakePool instance
        """
        pool = cls(
            ledger=ledger,
            max_stake_ratio=data['max_stake_ratio'],
            min_stake=data['min_stake']
        )

        # Restore stakes
        pool.stakes = {
            stake_id: Stake.from_dict(stake_data)
            for stake_id, stake_data in data['stakes'].items()
        }

        return pool

    def save(self, path: Path) -> None:
        """
        Save stake pool to JSON file (atomic write).

        Args:
            path: File path to save to
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file then rename
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix='.tmp_',
            suffix='.json'
        )

        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

            # Atomic rename
            os.replace(temp_path, path)
        except:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise

    @classmethod
    def load(cls, path: Path, ledger: CreditLedger) -> 'StakePool':
        """
        Load stake pool from JSON file.

        Args:
            path: File path to load from
            ledger: CreditLedger instance to use

        Returns:
            StakePool instance
        """
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data, ledger)


class AutoStaker:
    """
    Automatic staking decisions based on prediction confidence.

    Decides whether to stake and how much based on expert's confidence
    and chosen strategy.

    Attributes:
        pool: StakePool to place stakes in
    """

    def __init__(self, pool: StakePool):
        """
        Initialize auto-staker.

        Args:
            pool: StakePool to place stakes in
        """
        self.pool = pool

    def decide_stake(
        self,
        expert_id: str,
        prediction: Any,  # ExpertPrediction (avoiding circular import)
        strategy: StakeStrategy
    ) -> Optional[Tuple[float, float]]:
        """
        Decide whether to stake and how much based on confidence.

        Uses the confidence from the prediction's top item to determine
        stake amount and multiplier.

        Logic:
        - High confidence (>0.8): stake up to max_stake_ratio with strategy multiplier
        - Medium confidence (0.5-0.8): stake half the max with lower multiplier
        - Low confidence (<0.5): don't auto-stake

        Args:
            expert_id: ID of expert making prediction
            prediction: ExpertPrediction object
            strategy: StakeStrategy enum value

        Returns:
            Tuple of (amount, multiplier) or None if shouldn't stake
        """
        # Get confidence from top prediction item
        if not prediction.items:
            return None

        top_item, confidence = prediction.items[0]

        # Get expert's available balance
        account = self.pool.ledger.get_or_create_account(expert_id)
        available = account.balance
        max_stake_amount = available * self.pool.max_stake_ratio

        # Don't stake if below minimum
        if max_stake_amount < self.pool.min_stake:
            return None

        # Get strategy multiplier
        multiplier = strategy.value

        # Decision logic based on confidence
        if confidence > 0.8:
            # High confidence: stake up to max with full strategy multiplier
            amount = max_stake_amount
            return (amount, multiplier)

        elif confidence >= 0.5:
            # Medium confidence: stake half max with reduced multiplier
            amount = max_stake_amount * 0.5
            # Reduce multiplier proportionally to confidence
            # At 0.5 confidence: use 1.0x, at 0.8 confidence: use full multiplier
            confidence_factor = (confidence - 0.5) / 0.3  # 0 to 1 range
            adjusted_multiplier = 1.0 + (multiplier - 1.0) * confidence_factor
            adjusted_multiplier = max(1.0, min(adjusted_multiplier, multiplier))

            # Only stake if amount meets minimum
            if amount >= self.pool.min_stake:
                return (amount, adjusted_multiplier)

        # Low confidence or amount too small: don't stake
        return None
