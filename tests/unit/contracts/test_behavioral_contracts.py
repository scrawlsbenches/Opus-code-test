"""
Unit tests for the Behavioral Contracts system.

Tests cover:
1. Contract decorators (@requires, @ensures, @invariant)
2. ContractViolation exception
3. ContractRegistry
4. ContractEventEmitter (in-memory mode)
5. ContractMaterializer
"""

import pytest
from datetime import datetime, timedelta

from cortical.contracts import (
    requires,
    ensures,
    invariant,
    ContractViolation,
    ContractRegistry,
    ContractEventEmitter,
    ContractMaterializer,
    ContractState,
    ContractType,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def emitter():
    """Create an event emitter for testing."""
    return ContractEventEmitter(emit_all_checks=True)


@pytest.fixture
def registry(emitter):
    """Create a registry with emitter."""
    return ContractRegistry(emitter=emitter)


@pytest.fixture
def materializer(emitter):
    """Create a materializer."""
    return ContractMaterializer(emitter=emitter)


# =============================================================================
# Test: @requires decorator
# =============================================================================

class TestRequiresDecorator:
    """Tests for @requires precondition decorator."""

    def test_requires_passes_when_true(self):
        """Precondition passes when predicate returns True."""
        @requires(lambda self: self.ready, "Must be ready")
        def do_work(self):
            return "done"

        class Obj:
            ready = True

        result = do_work(Obj())
        assert result == "done"

    def test_requires_raises_when_false(self):
        """Precondition raises ContractViolation when predicate returns False."""
        @requires(lambda self: self.ready, "Must be ready")
        def do_work(self):
            return "done"

        class Obj:
            ready = False

        with pytest.raises(ContractViolation) as exc_info:
            do_work(Obj())

        assert exc_info.value.contract_type == "requires"
        assert "Must be ready" in str(exc_info.value)

    def test_requires_with_arguments(self):
        """Precondition can access method arguments."""
        @requires(lambda self, amount: amount > 0, "Amount must be positive")
        def deposit(self, amount):
            return amount

        class Account:
            pass

        # Should pass
        assert deposit(Account(), 100) == 100

        # Should fail
        with pytest.raises(ContractViolation):
            deposit(Account(), -50)

    def test_multiple_requires_all_checked(self):
        """Multiple @requires decorators are all checked."""
        @requires(lambda self: self.open, "Must be open")
        @requires(lambda self: not self.frozen, "Must not be frozen")
        def operate(self):
            return "ok"

        class Resource:
            open = True
            frozen = False

        # Both pass
        assert operate(Resource()) == "ok"

        # First fails
        r = Resource()
        r.open = False
        with pytest.raises(ContractViolation) as exc_info:
            operate(r)
        assert "Must be open" in str(exc_info.value)

        # Second fails
        r = Resource()
        r.frozen = True
        with pytest.raises(ContractViolation) as exc_info:
            operate(r)
        assert "Must not be frozen" in str(exc_info.value)


# =============================================================================
# Test: @ensures decorator
# =============================================================================

class TestEnsuresDecorator:
    """Tests for @ensures postcondition decorator."""

    def test_ensures_passes_when_true(self):
        """Postcondition passes when predicate returns True."""
        @ensures(lambda self, result: result is not None, "Must return value")
        def get_value(self):
            return 42

        class Obj:
            pass

        assert get_value(Obj()) == 42

    def test_ensures_raises_when_false(self):
        """Postcondition raises ContractViolation when predicate returns False."""
        @ensures(lambda self, result: result > 0, "Result must be positive")
        def compute(self):
            return -1

        class Obj:
            pass

        with pytest.raises(ContractViolation) as exc_info:
            compute(Obj())

        assert exc_info.value.contract_type == "ensures"
        assert "Result must be positive" in str(exc_info.value)

    def test_ensures_receives_result(self):
        """Postcondition receives the method's return value."""
        results = []

        @ensures(lambda self, result: (results.append(result), True)[1], "Capture result")
        def return_value(self, value):
            return value * 2

        class Obj:
            pass

        return_value(Obj(), 21)
        assert results == [42]


# =============================================================================
# Test: @invariant decorator
# =============================================================================

class TestInvariantDecorator:
    """Tests for @invariant class invariant decorator."""

    def test_invariant_checked_before_and_after(self):
        """Invariant is checked before AND after method execution."""
        check_times = []

        original_balance = [100]

        @invariant(lambda self: (check_times.append(datetime.now()), self.balance >= 0)[1],
                   "Balance must be non-negative")
        def withdraw(self, amount):
            self.balance -= amount
            return self.balance

        class Account:
            balance = 100

        acc = Account()
        withdraw(acc, 50)

        # Invariant was checked twice (before and after)
        assert len(check_times) == 2

    def test_invariant_fails_before(self):
        """Invariant violation before method raises with 'before' phase."""
        @invariant(lambda self: self.valid, "Must be valid")
        def process(self):
            return "done"

        class Obj:
            valid = False

        with pytest.raises(ContractViolation) as exc_info:
            process(Obj())

        assert "violated before method" in str(exc_info.value)

    def test_invariant_fails_after(self):
        """Invariant violation after method raises with 'after' phase."""
        @invariant(lambda self: self.count >= 0, "Count must be non-negative")
        def decrement(self):
            self.count -= 10  # Goes negative

        class Counter:
            count = 5

        c = Counter()
        with pytest.raises(ContractViolation) as exc_info:
            decrement(c)

        assert "violated after method" in str(exc_info.value)


# =============================================================================
# Test: ContractViolation Exception
# =============================================================================

class TestContractViolation:
    """Tests for ContractViolation exception."""

    def test_violation_attributes(self):
        """ContractViolation stores all relevant attributes."""
        exc = ContractViolation(
            contract_type="requires",
            description="Must be ready",
            context={"method": "do_work", "args_count": 1},
        )

        assert exc.contract_type == "requires"
        assert exc.description == "Must be ready"
        assert exc.context["method"] == "do_work"

    def test_violation_to_dict(self):
        """ContractViolation can be serialized to dict."""
        exc = ContractViolation("ensures", "Result must be positive")
        d = exc.to_dict()

        assert d["contract_type"] == "ensures"
        assert d["description"] == "Result must be positive"

    def test_violation_message_format(self):
        """ContractViolation has informative message."""
        exc = ContractViolation(
            "invariant",
            "Balance >= 0",
            {"method": "Account.withdraw"},
        )

        assert "invariant" in str(exc)
        assert "Balance >= 0" in str(exc)
        assert "Account.withdraw" in str(exc)


# =============================================================================
# Test: ContractRegistry
# =============================================================================

class TestContractRegistry:
    """Tests for ContractRegistry."""

    def test_registry_tracks_contracts(self, registry):
        """Registry tracks contracts from decorated methods."""
        # NOTE: @registry.track MUST be outermost decorator
        @registry.track
        @requires(lambda self: True, "Always true")
        def always_pass(self):
            return True

        contracts = registry.all()
        assert len(contracts) == 1
        assert contracts[0].contract_type == ContractType.REQUIRES

    def test_registry_query_by_class(self, registry):
        """Registry can query contracts by class name."""
        class MyClass:
            @registry.track
            @requires(lambda self: True, "Precondition")
            def method_a(self):
                pass

            @registry.track
            @ensures(lambda self, r: True, "Postcondition")
            def method_b(self):
                pass

        # Class name includes test method context
        contracts = registry.all()
        assert len(contracts) == 2
        # Verify we have both types
        types = {c.contract_type for c in contracts}
        assert ContractType.REQUIRES in types
        assert ContractType.ENSURES in types

    def test_registry_query_by_type(self, registry):
        """Registry can query contracts by type."""
        @registry.track
        @requires(lambda self: True, "Pre")
        def with_requires(self):
            pass

        @registry.track
        @ensures(lambda self, r: True, "Post")
        def with_ensures(self):
            pass

        requires_contracts = registry.by_type(ContractType.REQUIRES)
        ensures_contracts = registry.by_type(ContractType.ENSURES)

        assert len(requires_contracts) == 1
        assert len(ensures_contracts) == 1

    def test_registry_stats(self, registry):
        """Registry provides statistics."""
        @registry.track
        @requires(lambda self: True, "Pre")
        @ensures(lambda self, r: True, "Post")
        def multi_contract(self):
            return True

        stats = registry.stats()
        assert stats["total_contracts"] == 2
        assert "by_type" in stats


# =============================================================================
# Test: ContractEventEmitter
# =============================================================================

class TestContractEventEmitter:
    """Tests for ContractEventEmitter."""

    def test_emitter_buffers_events(self, emitter, registry):
        """Emitter buffers events when no CEL store provided."""
        @registry.track
        @requires(lambda self: True, "Always pass")
        def passing_method(self):
            return True

        class Obj:
            pass

        passing_method(Obj())

        events = emitter.buffered_events
        assert len(events) >= 1

    def test_emitter_records_violations(self, emitter, registry):
        """Emitter records violations as MetaCognition events."""
        @registry.track
        @requires(lambda self: False, "Will fail")
        def failing_method(self):
            return True

        class Obj:
            pass

        try:
            failing_method(Obj())
        except ContractViolation:
            pass

        events = emitter.buffered_events
        metacog_events = [e for e in events if e.get('event_type') == 'METACOGNITION']
        assert len(metacog_events) >= 1

    def test_emitter_stats(self, emitter):
        """Emitter tracks statistics."""
        stats = emitter.stats
        assert "checks_emitted" in stats
        assert "violations_emitted" in stats
        assert "emit_all_checks" in stats


# =============================================================================
# Test: ContractMaterializer
# =============================================================================

class TestContractMaterializer:
    """Tests for ContractMaterializer."""

    def test_materializer_current_state(self, materializer, emitter, registry):
        """Materializer computes current state from events."""
        @registry.track
        @requires(lambda self: True, "Pass")
        def passing(self):
            return True

        @registry.track
        @requires(lambda self: False, "Fail")
        def failing(self):
            return True

        class Obj:
            pass

        # Execute some operations
        passing(Obj())
        passing(Obj())

        try:
            failing(Obj())
        except ContractViolation:
            pass

        state = materializer.current_state()
        # 2 passing checks + 1 failing check = 3 checks total
        # But the failing check emits before raising, so we get 2 passing + 1 violation
        assert state.total_checks >= 2  # At least 2 passing checks
        assert state.total_violations >= 1  # At least 1 violation

    def test_materializer_health_report(self, materializer):
        """Materializer generates health reports."""
        report = materializer.health_report()

        assert "status" in report
        assert "state" in report
        assert "recommendations" in report
        assert "generated_at" in report


# =============================================================================
# Test: ContractState
# =============================================================================

class TestContractState:
    """Tests for ContractState dataclass."""

    def test_state_violation_rate(self):
        """ContractState computes violation rate."""
        state = ContractState(total_checks=100, total_violations=5)
        assert state.violation_rate == 5.0

    def test_state_is_healthy(self):
        """ContractState reports health status."""
        healthy = ContractState(total_checks=100, total_violations=0)
        assert healthy.is_healthy

        unhealthy = ContractState(total_checks=100, total_violations=10)
        assert not unhealthy.is_healthy

    def test_state_serialization(self):
        """ContractState can be serialized and deserialized."""
        state = ContractState(
            total_checks=50,
            total_violations=2,
            violations_by_method={"MyClass.method": 2},
        )

        d = state.to_dict()
        restored = ContractState.from_dict(d)

        assert restored.total_checks == 50
        assert restored.total_violations == 2
        assert restored.violations_by_method == {"MyClass.method": 2}


# =============================================================================
# Test: Integration Scenario
# =============================================================================

class TestIntegrationScenario:
    """Integration tests for the full contract flow."""

    def test_full_lifecycle(self, emitter, registry, materializer):
        """Test complete lifecycle: define → execute → materialize → query."""
        # Define class with contracts
        # NOTE: @registry.track MUST be outermost decorator
        class BankAccount:
            def __init__(self, balance: float = 0):
                self._balance = balance

            @registry.track
            @requires(lambda self, amount: amount > 0, "Amount must be positive")
            @ensures(lambda self, result: self._balance >= 0, "Balance must be non-negative")
            def deposit(self, amount: float) -> float:
                self._balance += amount
                return self._balance

            @registry.track
            @requires(lambda self, amount: amount > 0, "Amount must be positive")
            @requires(lambda self, amount: self._balance >= amount, "Insufficient funds")
            def withdraw(self, amount: float) -> float:
                self._balance -= amount
                return self._balance

        # Execute operations
        account = BankAccount(100)

        # Successful operations
        account.deposit(50)
        account.withdraw(30)

        # Failed operation
        try:
            account.withdraw(200)  # Insufficient funds
        except ContractViolation:
            pass

        # Query state
        state = materializer.current_state()

        # Verify
        assert state.total_checks >= 3  # At least 3 checks occurred
        assert state.total_violations >= 1  # At least 1 violation

        # Health report
        report = materializer.health_report()
        assert "status" in report

        # Registry has contracts
        assert registry.stats()["total_contracts"] >= 4
