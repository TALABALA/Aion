"""
AION Condition Operators

Comparison operators for condition evaluation.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from aion.automation.types import ConditionOperator


class OperatorRegistry:
    """
    Registry of comparison operators.

    Provides standard operators and allows custom operator registration.
    """

    def __init__(self):
        self._operators: Dict[ConditionOperator, Callable[[Any, Any], bool]] = {}
        self._register_builtin_operators()

    def _register_builtin_operators(self) -> None:
        """Register built-in operators."""
        self._operators[ConditionOperator.EQUALS] = self._equals
        self._operators[ConditionOperator.NOT_EQUALS] = self._not_equals
        self._operators[ConditionOperator.GREATER_THAN] = self._greater_than
        self._operators[ConditionOperator.LESS_THAN] = self._less_than
        self._operators[ConditionOperator.GREATER_EQUAL] = self._greater_equal
        self._operators[ConditionOperator.LESS_EQUAL] = self._less_equal
        self._operators[ConditionOperator.CONTAINS] = self._contains
        self._operators[ConditionOperator.NOT_CONTAINS] = self._not_contains
        self._operators[ConditionOperator.STARTS_WITH] = self._starts_with
        self._operators[ConditionOperator.ENDS_WITH] = self._ends_with
        self._operators[ConditionOperator.MATCHES] = self._matches
        self._operators[ConditionOperator.IS_NULL] = self._is_null
        self._operators[ConditionOperator.IS_NOT_NULL] = self._is_not_null
        self._operators[ConditionOperator.IS_EMPTY] = self._is_empty
        self._operators[ConditionOperator.IS_NOT_EMPTY] = self._is_not_empty
        self._operators[ConditionOperator.IN] = self._in
        self._operators[ConditionOperator.NOT_IN] = self._not_in
        self._operators[ConditionOperator.IS_TRUE] = self._is_true
        self._operators[ConditionOperator.IS_FALSE] = self._is_false

    def register(
        self,
        operator: ConditionOperator,
        func: Callable[[Any, Any], bool],
    ) -> None:
        """Register a custom operator."""
        self._operators[operator] = func

    def evaluate(
        self,
        operator: ConditionOperator,
        left: Any,
        right: Any,
    ) -> bool:
        """Evaluate an operator."""
        func = self._operators.get(operator)
        if not func:
            raise ValueError(f"Unknown operator: {operator}")

        return func(left, right)

    # === Operator Implementations ===

    @staticmethod
    def _equals(left: Any, right: Any) -> bool:
        """Equality comparison."""
        # Handle type coercion for common cases
        if isinstance(left, str) and isinstance(right, (int, float)):
            try:
                left = float(left) if "." in left else int(left)
            except ValueError:
                pass
        elif isinstance(right, str) and isinstance(left, (int, float)):
            try:
                right = float(right) if "." in right else int(right)
            except ValueError:
                pass

        return left == right

    @staticmethod
    def _not_equals(left: Any, right: Any) -> bool:
        """Inequality comparison."""
        return not OperatorRegistry._equals(left, right)

    @staticmethod
    def _greater_than(left: Any, right: Any) -> bool:
        """Greater than comparison."""
        try:
            return float(left) > float(right)
        except (ValueError, TypeError):
            return str(left) > str(right)

    @staticmethod
    def _less_than(left: Any, right: Any) -> bool:
        """Less than comparison."""
        try:
            return float(left) < float(right)
        except (ValueError, TypeError):
            return str(left) < str(right)

    @staticmethod
    def _greater_equal(left: Any, right: Any) -> bool:
        """Greater than or equal comparison."""
        try:
            return float(left) >= float(right)
        except (ValueError, TypeError):
            return str(left) >= str(right)

    @staticmethod
    def _less_equal(left: Any, right: Any) -> bool:
        """Less than or equal comparison."""
        try:
            return float(left) <= float(right)
        except (ValueError, TypeError):
            return str(left) <= str(right)

    @staticmethod
    def _contains(left: Any, right: Any) -> bool:
        """Contains check (string or collection)."""
        if left is None:
            return False

        if isinstance(left, str):
            return str(right) in left

        if isinstance(left, (list, tuple, set)):
            return right in left

        if isinstance(left, dict):
            return right in left.keys()

        return False

    @staticmethod
    def _not_contains(left: Any, right: Any) -> bool:
        """Not contains check."""
        return not OperatorRegistry._contains(left, right)

    @staticmethod
    def _starts_with(left: Any, right: Any) -> bool:
        """String starts with check."""
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        return left.startswith(right)

    @staticmethod
    def _ends_with(left: Any, right: Any) -> bool:
        """String ends with check."""
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        return left.endswith(right)

    @staticmethod
    def _matches(left: Any, right: Any) -> bool:
        """Regex match check."""
        if left is None:
            return False

        try:
            pattern = re.compile(str(right))
            return bool(pattern.search(str(left)))
        except re.error:
            return False

    @staticmethod
    def _is_null(left: Any, right: Any) -> bool:
        """Null check."""
        return left is None

    @staticmethod
    def _is_not_null(left: Any, right: Any) -> bool:
        """Not null check."""
        return left is not None

    @staticmethod
    def _is_empty(left: Any, right: Any) -> bool:
        """Empty check (string, list, dict)."""
        if left is None:
            return True

        if isinstance(left, str):
            return left.strip() == ""

        if isinstance(left, (list, tuple, set, dict)):
            return len(left) == 0

        return False

    @staticmethod
    def _is_not_empty(left: Any, right: Any) -> bool:
        """Not empty check."""
        return not OperatorRegistry._is_empty(left, right)

    @staticmethod
    def _in(left: Any, right: Any) -> bool:
        """In collection check."""
        if right is None:
            return False

        if isinstance(right, (list, tuple, set)):
            return left in right

        if isinstance(right, str):
            # Treat as comma-separated list
            items = [item.strip() for item in right.split(",")]
            return str(left) in items

        return False

    @staticmethod
    def _not_in(left: Any, right: Any) -> bool:
        """Not in collection check."""
        return not OperatorRegistry._in(left, right)

    @staticmethod
    def _is_true(left: Any, right: Any) -> bool:
        """Boolean true check."""
        if isinstance(left, bool):
            return left is True
        if isinstance(left, str):
            return left.lower() in ("true", "yes", "1", "on")
        if isinstance(left, (int, float)):
            return left != 0
        return bool(left)

    @staticmethod
    def _is_false(left: Any, right: Any) -> bool:
        """Boolean false check."""
        return not OperatorRegistry._is_true(left, right)


# Global operator registry
_operator_registry = OperatorRegistry()


def compare(
    left: Any,
    operator: ConditionOperator,
    right: Any,
) -> bool:
    """
    Compare two values using an operator.

    Args:
        left: Left operand
        operator: Comparison operator
        right: Right operand

    Returns:
        Comparison result
    """
    return _operator_registry.evaluate(operator, left, right)


def register_operator(
    operator: ConditionOperator,
    func: Callable[[Any, Any], bool],
) -> None:
    """Register a custom operator."""
    _operator_registry.register(operator, func)
