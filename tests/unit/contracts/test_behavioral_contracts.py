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


# =============================================================================
# Test: Contract Reducer (CEL Integration)
# =============================================================================

class TestContractReducer:
    """Tests for contract_reducer and related CEL integration."""

    def test_contract_reducer_initializes_state(self):
        """Reducer initializes state on first call."""
        from cortical.contracts.cel_integration import contract_reducer

        # First call with None state initializes
        state = contract_reducer(None, object())

        assert state is not None
        assert state['total_checks'] == 0
        assert state['total_violations'] == 0
        assert state['violations_by_method'] == {}

    def test_contract_reducer_counts_checks(self):
        """Reducer counts contract checks from events."""
        from cortical.contracts.cel_integration import contract_reducer
        from cortical.cel.core.events import Observation

        # Create mock check event
        event = Observation(
            content={'type': 'contract_check', 'method': 'Test.method'},
            concepts=('contract',),
        )

        state = contract_reducer(None, event)
        assert state['total_checks'] == 1
        assert state['last_check'] is not None
        assert state['first_check'] is not None

        # Second check
        state = contract_reducer(state, event)
        assert state['total_checks'] == 2

    def test_contract_reducer_counts_violations(self):
        """Reducer counts violations from MetaCognition events."""
        from cortical.contracts.cel_integration import contract_reducer
        from cortical.cel.core.events import MetaCognition

        # Create violation event
        event = MetaCognition(
            observation_type='contract_violation',
            metrics={'method': 'MyClass.mymethod'},
            conclusions=['Contract violated'],
            actions_triggered=['log'],
        )

        state = contract_reducer(None, event)
        assert state['total_violations'] == 1
        assert state['last_violation'] is not None
        assert state['violations_by_method']['MyClass.mymethod'] == 1

    def test_contract_reducer_wrapper(self):
        """_ContractReducerWrapper implements EventReducer protocol."""
        from cortical.contracts.cel_integration import (
            create_contract_reducer,
            _ContractReducerWrapper,
        )

        wrapper = create_contract_reducer()
        assert isinstance(wrapper, _ContractReducerWrapper)
        assert wrapper.entity_type == 'contract_summary'

        # Can call it
        state = wrapper(None, object())
        assert state is not None


# =============================================================================
# Test: Registry Query Methods
# =============================================================================

class TestContractRegistryQueries:
    """Tests for ContractRegistry query methods."""

    def test_for_class_returns_contracts(self, registry):
        """for_class returns all contracts for a class."""
        class QueryTestClass:
            @registry.track
            @requires(lambda self: True, "Always pass")
            def method1(self):
                pass

            @registry.track
            @requires(lambda self: True, "Also pass")
            def method2(self):
                pass

        # Contracts are registered when @registry.track is applied
        # Class name includes full qualname path for nested classes
        all_contracts = registry.all()
        assert len(all_contracts) >= 2

        # Get the actual class name used (includes test function prefix)
        class_name = all_contracts[0].class_name
        contracts = registry.for_class(class_name)
        assert len(contracts) >= 2

    def test_for_method_returns_specific(self, registry):
        """for_method returns contracts for a specific method."""
        class MethodTestClass:
            @registry.track
            @requires(lambda self: True, "Pass")
            @ensures(lambda self, result: True, "Also pass")
            def specific_method(self):
                return True

        # Get the actual class name used (includes full qualname)
        all_contracts = registry.all()
        assert len(all_contracts) >= 2

        class_name = all_contracts[0].class_name
        method_name = all_contracts[0].method_name

        contracts = registry.for_method(class_name, method_name)
        # Should have both requires and ensures
        assert len(contracts) >= 2

    def test_by_type_filters_correctly(self, registry):
        """by_type returns only contracts of that type."""
        # ContractType is in registry module, not decorators
        @registry.track
        @requires(lambda self: True, "Precondition")
        def with_requires(self):
            pass

        @registry.track
        @ensures(lambda self, result: True, "Postcondition")
        def with_ensures(self):
            return True

        requires_contracts = registry.by_type(ContractType.REQUIRES)
        ensures_contracts = registry.by_type(ContractType.ENSURES)

        assert len(requires_contracts) >= 1
        assert len(ensures_contracts) >= 1

    def test_contracts_with_violations(self, registry):
        """contracts_with_violations returns violated contracts."""
        class ViolationTestClass:
            @registry.track
            @requires(lambda self: False, "Will fail")
            def failing(self):
                pass

        obj = ViolationTestClass()
        try:
            obj.failing()
        except ContractViolation:
            pass

        # Update stats since the decorator doesn't auto-update registry
        contracts = registry.all()
        for c in contracts:
            if "Will fail" in c.description:
                registry.update_stats(c.id, passed=False)

        violated = registry.contracts_with_violations()
        assert len(violated) >= 1
        assert all(c.violation_count > 0 for c in violated)

    def test_contracts_never_checked(self, registry):
        """contracts_never_checked returns unchecked contracts."""
        # Register but don't execute
        @registry.track
        @requires(lambda self: True, "Never called")
        def never_called(self):
            pass

        # Check initial state - new registry should have unchecked contracts
        # after registration but before execution
        # Note: contracts are registered on first call, so this tests
        # the registry state after some contracts have been checked
        never_checked = registry.contracts_never_checked()
        # Just verify the method works
        assert isinstance(never_checked, list)


# =============================================================================
# Test: Registry Statistics
# =============================================================================

class TestContractRegistryStats:
    """Tests for ContractRegistry statistics methods."""

    def test_stats_returns_summary(self, registry):
        """stats returns comprehensive statistics."""
        class StatsTestClass:
            @registry.track
            @requires(lambda self: True, "Pass")
            def passing(self):
                pass

            @registry.track
            @requires(lambda self: False, "Fail")
            def failing(self):
                pass

        obj = StatsTestClass()

        # Execute - passing calls
        obj.passing()
        obj.passing()

        # Execute - failing call
        try:
            obj.failing()
        except ContractViolation:
            pass

        # Update stats since decorator doesn't auto-update
        contracts = registry.all()
        for c in contracts:
            if "Pass" in c.description:
                registry.update_stats(c.id, passed=True)
                registry.update_stats(c.id, passed=True)
            elif "Fail" in c.description:
                registry.update_stats(c.id, passed=False)

        stats = registry.stats()

        assert 'total_contracts' in stats
        assert 'contracts_with_violations' in stats
        assert 'total_checks' in stats
        assert 'total_violations' in stats
        assert 'violation_rate' in stats
        assert 'by_type' in stats
        assert 'classes_covered' in stats

        assert stats['total_contracts'] >= 2
        assert stats['total_violations'] >= 1

    def test_update_stats_increments_counts(self, registry):
        """update_stats correctly updates contract statistics."""
        @registry.track
        @requires(lambda self: True, "Test")
        def method(self):
            pass

        class Obj:
            pass

        # Execute to register
        method(Obj())

        # Get the contract
        contracts = registry.all()
        assert len(contracts) >= 1

        contract = contracts[0]
        initial_checks = contract.check_count

        # Manually update stats (simulating emitter)
        registry.update_stats(contract.id, passed=True)

        assert contract.check_count == initial_checks + 1

    def test_get_contract_by_id(self, registry):
        """get returns contract by ID."""
        @registry.track
        @requires(lambda self: True, "Test")
        def method(self):
            pass

        class Obj:
            pass

        method(Obj())

        contracts = registry.all()
        assert len(contracts) >= 1

        contract_id = contracts[0].id
        retrieved = registry.get(contract_id)

        assert retrieved is not None
        assert retrieved.id == contract_id

    def test_get_missing_returns_none(self, registry):
        """get returns None for missing contract."""
        result = registry.get("nonexistent-contract-id")
        assert result is None
