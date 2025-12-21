"""
Sample file with various comment markers for testing CommentCleaner.

This file contains examples of all marker types with different patterns.
"""


def example_function():
    """Example function with various comment markers."""
    # THINKING: This approach uses a hash map for O(1) lookups
    data = {}

    # TODO: Add input validation
    value = process_data(data)

    # QUESTION: Should we cache this result?
    result = compute_expensive_operation(value)

    # NOTE: This pattern matches what we do in module Y
    return result


def another_function():
    """Function with more marker variations."""
    # thinking Should we refactor this?
    x = 1

    # TODO handle edge case for empty lists
    items = []

    # QUESTION: Is this the right abstraction? Answered: Yes, decided in review.
    if items:
        process(items)

    # note: obvious implementation
    return x


class ExampleClass:
    """Class with performance and technical debt markers."""

    def method_one(self):
        # PERF: This is O(n²), acceptable for n < 1000
        for i in range(100):
            for j in range(100):
                calculate(i, j)

    def method_two(self):
        # HACK: Workaround for issue #123, remove when fixed
        return None

    def method_three(self):
        # TODO: Implement proper error handling (task #456)
        pass

    def method_four(self):
        # NOTE: Cross-reference to authentication module
        pass


# THINKING: Just a debug comment, temporary
def debug_function():
    # TODO Fix this later
    pass


# QUESTION: What's the best way to handle this edge case?
def edge_case_handler():
    # HACK: Quick fix for demo
    pass


# NOTE: This is self-explanatory
def simple_function():
    return 42


# PERF Edge case for performance tracking
def optimized_function():
    # NOTE: See also the caching module for related patterns
    return fast_compute()


def process_data(data):
    """Stub function."""
    return data


def compute_expensive_operation(value):
    """Stub function."""
    return value


def process(items):
    """Stub function."""
    pass


def calculate(i, j):
    """Stub function."""
    return i * j


def fast_compute():
    """Stub function."""
    return 100
