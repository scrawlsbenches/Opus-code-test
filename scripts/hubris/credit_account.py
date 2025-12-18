#!/usr/bin/env python3
"""
Credit Account System

Tracks expert value through credit balances and transaction history.
Experts earn credits when their predictions help, lose credits when wrong.

Classes:
    CreditTransaction: Individual credit/debit transaction
    CreditAccount: Expert's account with balance and transaction history
    CreditLedger: Manager for all expert accounts
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class CreditTransaction:
    """
    Individual credit transaction.

    Attributes:
        timestamp: Unix timestamp when transaction occurred
        amount: Credit amount (positive = credit, negative = debit)
        expert_id: ID of the expert this transaction is for
        reason: Why this transaction occurred (e.g., "correct_prediction")
        context: Additional metadata about the transaction
        balance_after: Running balance after this transaction
    """
    timestamp: float
    amount: float
    expert_id: str
    reason: str
    context: Dict[str, Any]
    balance_after: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreditTransaction':
        """Load from dict."""
        return cls(**data)


class CreditAccount:
    """
    Credit account for a single expert.

    Tracks balance, transaction history, and provides query methods.

    Attributes:
        expert_id: ID of the expert this account belongs to
        balance: Current credit balance
        transactions: List of all transactions (oldest to newest)
        created_at: Unix timestamp when account was created
    """

    def __init__(
        self,
        expert_id: str,
        initial_balance: float = 100.0,
        created_at: Optional[float] = None
    ):
        """
        Initialize credit account.

        Args:
            expert_id: Expert ID this account belongs to
            initial_balance: Starting balance (default: 100.0)
            created_at: Creation timestamp (default: now)
        """
        self.expert_id = expert_id
        self.balance = initial_balance
        self.transactions: List[CreditTransaction] = []
        self.created_at = created_at if created_at is not None else time.time()

    def credit(
        self,
        amount: float,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CreditTransaction:
        """
        Add credits to account.

        Args:
            amount: Amount to add (must be positive)
            reason: Why credits are being added
            context: Additional metadata (optional)

        Returns:
            CreditTransaction record

        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError(f"Credit amount must be positive, got {amount}")

        self.balance += amount

        transaction = CreditTransaction(
            timestamp=time.time(),
            amount=amount,
            expert_id=self.expert_id,
            reason=reason,
            context=context or {},
            balance_after=self.balance
        )

        self.transactions.append(transaction)
        return transaction

    def debit(
        self,
        amount: float,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CreditTransaction:
        """
        Remove credits from account.

        Balance can go negative to track poorly performing experts.

        Args:
            amount: Amount to remove (must be positive)
            reason: Why credits are being removed
            context: Additional metadata (optional)

        Returns:
            CreditTransaction record

        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError(f"Debit amount must be positive, got {amount}")

        self.balance -= amount

        transaction = CreditTransaction(
            timestamp=time.time(),
            amount=-amount,
            expert_id=self.expert_id,
            reason=reason,
            context=context or {},
            balance_after=self.balance
        )

        self.transactions.append(transaction)
        return transaction

    def get_recent_transactions(self, n: int = 10) -> List[CreditTransaction]:
        """
        Get the N most recent transactions.

        Args:
            n: Number of transactions to return (default: 10)

        Returns:
            List of recent transactions (newest first)
        """
        return list(reversed(self.transactions[-n:]))

    def get_transactions_since(self, timestamp: float) -> List[CreditTransaction]:
        """
        Get all transactions after a given timestamp.

        Args:
            timestamp: Unix timestamp cutoff

        Returns:
            List of transactions after timestamp (oldest to newest)
        """
        return [t for t in self.transactions if t.timestamp > timestamp]

    def get_balance_history(self) -> List[Tuple[float, float]]:
        """
        Get balance over time.

        Returns:
            List of (timestamp, balance) tuples
        """
        # Include initial balance at creation time
        history = [(self.created_at, 100.0)]  # Assuming initial is always 100

        # Add each transaction's balance
        history.extend([
            (t.timestamp, t.balance_after)
            for t in self.transactions
        ])

        return history

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'expert_id': self.expert_id,
            'balance': self.balance,
            'transactions': [t.to_dict() for t in self.transactions],
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreditAccount':
        """
        Load account from dict.

        Args:
            data: Dictionary representation

        Returns:
            CreditAccount instance
        """
        account = cls(
            expert_id=data['expert_id'],
            initial_balance=0.0,  # Will be set from data
            created_at=data['created_at']
        )

        # Restore balance
        account.balance = data['balance']

        # Restore transactions
        account.transactions = [
            CreditTransaction.from_dict(t)
            for t in data['transactions']
        ]

        return account

    def save(self, path: Path) -> None:
        """
        Save account to JSON file (atomic write).

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
    def load(cls, path: Path) -> 'CreditAccount':
        """
        Load account from JSON file.

        Args:
            path: File path to load from

        Returns:
            CreditAccount instance
        """
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)


class CreditLedger:
    """
    Manages all expert credit accounts.

    Provides account management, transfers, and aggregate queries.

    Attributes:
        accounts: Dict mapping expert_id to CreditAccount
    """

    def __init__(self):
        """Initialize empty ledger."""
        self.accounts: Dict[str, CreditAccount] = {}

    def get_or_create_account(
        self,
        expert_id: str,
        initial_balance: float = 100.0
    ) -> CreditAccount:
        """
        Get existing account or create new one.

        Args:
            expert_id: Expert ID
            initial_balance: Initial balance for new accounts (default: 100.0)

        Returns:
            CreditAccount instance
        """
        if expert_id not in self.accounts:
            self.accounts[expert_id] = CreditAccount(
                expert_id=expert_id,
                initial_balance=initial_balance
            )

        return self.accounts[expert_id]

    def transfer(
        self,
        from_expert: str,
        to_expert: str,
        amount: float,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[CreditTransaction, CreditTransaction]:
        """
        Transfer credits between experts.

        Args:
            from_expert: Expert to debit
            to_expert: Expert to credit
            amount: Amount to transfer (must be positive)
            reason: Transfer reason
            context: Additional metadata (optional)

        Returns:
            Tuple of (debit_transaction, credit_transaction)

        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError(f"Transfer amount must be positive, got {amount}")

        # Get or create accounts
        from_account = self.get_or_create_account(from_expert)
        to_account = self.get_or_create_account(to_expert)

        # Build context with transfer details
        transfer_context = context or {}
        transfer_context['transfer_from'] = from_expert
        transfer_context['transfer_to'] = to_expert

        # Perform transfer
        debit_tx = from_account.debit(amount, reason, transfer_context)
        credit_tx = to_account.credit(amount, reason, transfer_context)

        return (debit_tx, credit_tx)

    def get_top_experts(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get experts with highest balances.

        Args:
            n: Number of experts to return (default: 5)

        Returns:
            List of (expert_id, balance) tuples, sorted by balance descending
        """
        balances = [
            (expert_id, account.balance)
            for expert_id, account in self.accounts.items()
        ]

        # Sort by balance descending
        balances.sort(key=lambda x: -x[1])

        return balances[:n]

    def get_total_credits(self) -> float:
        """
        Get sum of all account balances.

        Returns:
            Total credits across all accounts
        """
        return sum(account.balance for account in self.accounts.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'accounts': {
                expert_id: account.to_dict()
                for expert_id, account in self.accounts.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreditLedger':
        """
        Load ledger from dict.

        Args:
            data: Dictionary representation

        Returns:
            CreditLedger instance
        """
        ledger = cls()

        # Restore accounts
        ledger.accounts = {
            expert_id: CreditAccount.from_dict(account_data)
            for expert_id, account_data in data['accounts'].items()
        }

        return ledger

    def save(self, path: Path) -> None:
        """
        Save ledger to JSON file (atomic write).

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
    def load(cls, path: Path) -> 'CreditLedger':
        """
        Load ledger from JSON file.

        Args:
            path: File path to load from

        Returns:
            CreditLedger instance
        """
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)
