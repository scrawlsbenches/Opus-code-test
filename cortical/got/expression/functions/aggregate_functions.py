"""
Aggregation functions for GoT query expressions.

These functions provide data aggregation capabilities for analyzing
entity collections in the Graph of Thought system.

AVAILABLE FUNCTIONS
-------------------
aggregate(field, operation="count"):
    Groups entities by a field and performs aggregation.
    Currently supports "count" operation.
    Returns Dict[str, int] mapping field values to counts.

USAGE EXAMPLES
--------------
Count tasks by status:
    >>> aggregate('status')
    {'pending': 5, 'in_progress': 3, 'completed': 12}

Count tasks by priority:
    >>> aggregate('priority')
    {'high': 2, 'medium': 8, 'low': 15}
"""

from typing import Any, Dict, List

from cortical.got.expression.registry import FunctionRegistry, QueryFunction, FunctionSignature
from cortical.got.query_builder import Query


@FunctionRegistry.register("aggregate")
class AggregateFunction(QueryFunction):
    """Aggregate entities by a field."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="aggregate",
            description="Count or group entities by a field",
            required_args=["field"],
            optional_args={"operation": "count"},
            returns="Dict mapping field values to counts/results"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Dict[Any, int]:
        """
        Execute aggregate function.

        Groups entities by the specified field and performs the requested
        aggregation operation (currently only "count" is supported).

        Args:
            manager: GoTManager instance
            args: Positional arguments [field, operation]
            kwargs: Keyword arguments {field: str, operation: str}

        Returns:
            Dictionary mapping field values to aggregated results

        Raises:
            ValueError: If field is not provided
        """
        # Parse arguments
        if args:
            field = args[0]
            operation = args[1] if len(args) > 1 else kwargs.get('operation', 'count')
        else:
            field = kwargs.get('field')
            operation = kwargs.get('operation', 'count')

        if not field:
            raise ValueError("field is required")

        # Use Query builder's group_by and count functionality
        if operation == "count":
            result = Query(manager).tasks().group_by(field).count().execute()
            return result if isinstance(result, dict) else {}

        # Future operations (sum, avg, etc.) can be added here
        raise ValueError(f"Unsupported aggregation operation: {operation}")
