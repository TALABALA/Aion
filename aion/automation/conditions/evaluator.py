"""
AION Condition Evaluator

Evaluates workflow conditions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import structlog

from aion.automation.types import Condition, ConditionOperator
from aion.automation.conditions.operators import compare, OperatorRegistry
from aion.automation.conditions.expressions import ExpressionParser

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class ConditionEvaluator:
    """
    Evaluates workflow conditions.

    Features:
    - Expression resolution
    - Operator evaluation
    - AND/OR logic
    - Negation support
    """

    def __init__(self):
        self._operator_registry = OperatorRegistry()
        self._expression_parser = ExpressionParser()

    async def evaluate(
        self,
        condition: Condition,
        context: "ExecutionContext",
    ) -> bool:
        """
        Evaluate a condition against a context.

        Args:
            condition: Condition to evaluate
            context: Execution context for variable resolution

        Returns:
            Boolean result
        """
        try:
            # Evaluate the main condition
            result = await self._evaluate_single(condition, context)

            # Evaluate AND conditions
            if condition.and_conditions:
                for and_condition in condition.and_conditions:
                    if not await self.evaluate(and_condition, context):
                        result = False
                        break

            # Evaluate OR conditions (only if main/AND failed)
            if not result and condition.or_conditions:
                for or_condition in condition.or_conditions:
                    if await self.evaluate(or_condition, context):
                        result = True
                        break

            # Apply negation
            if condition.negate:
                result = not result

            logger.debug(
                "condition_evaluated",
                condition_id=condition.id,
                result=result,
            )

            return result

        except Exception as e:
            logger.error(
                "condition_error",
                condition_id=condition.id,
                error=str(e),
            )
            # Default to False on error
            return False

    async def _evaluate_single(
        self,
        condition: Condition,
        context: "ExecutionContext",
    ) -> bool:
        """Evaluate a single condition (without AND/OR)."""
        # Resolve left side
        left = self._resolve_value(condition.left, context)

        # Resolve right side
        right = self._resolve_value(condition.right, context)

        # Apply operator
        result = compare(left, condition.operator, right)

        logger.debug(
            "condition_comparison",
            left=left,
            operator=condition.operator.value,
            right=right,
            result=result,
        )

        return result

    def _resolve_value(
        self,
        value: Any,
        context: "ExecutionContext",
    ) -> Any:
        """Resolve a value, handling expressions."""
        if value is None:
            return None

        if isinstance(value, str):
            # Check if it's an expression (starts with {{ or is a variable reference)
            if value.startswith("{{") and value.endswith("}}"):
                return context.resolve(value)

            # Check if it's a plain variable reference
            if "." in value and not value.startswith(("'", '"')):
                # Try to resolve as variable
                resolved = context.get(value)
                if resolved is not None:
                    return resolved

            # Check if it's a complex expression
            if any(op in value for op in ["+", "-", "*", "/", ">", "<", "==", "!="]):
                try:
                    self._expression_parser.set_variables(context.to_dict())
                    return self._expression_parser.parse(value)
                except Exception:
                    pass

            return value

        return value

    def evaluate_expression(
        self,
        expression: str,
        context: "ExecutionContext",
    ) -> Any:
        """
        Evaluate a complex expression.

        Args:
            expression: Expression string
            context: Execution context

        Returns:
            Evaluated result
        """
        self._expression_parser.set_variables(context.to_dict())
        return self._expression_parser.parse(expression)

    async def evaluate_all(
        self,
        conditions: list[Condition],
        context: "ExecutionContext",
        require_all: bool = True,
    ) -> bool:
        """
        Evaluate multiple conditions.

        Args:
            conditions: List of conditions
            context: Execution context
            require_all: If True, all must pass (AND). If False, any must pass (OR).

        Returns:
            Boolean result
        """
        if not conditions:
            return True

        results = []
        for condition in conditions:
            result = await self.evaluate(condition, context)
            results.append(result)

            # Short-circuit evaluation
            if require_all and not result:
                return False
            if not require_all and result:
                return True

        return all(results) if require_all else any(results)


# === Condition Builder ===


class ConditionBuilder:
    """
    Builder for creating conditions programmatically.

    Example:
        condition = (
            ConditionBuilder()
            .when("inputs.status")
            .equals("active")
            .and_when("inputs.priority")
            .greater_than(5)
            .build()
        )
    """

    def __init__(self):
        self._conditions: list[Condition] = []
        self._current: Optional[Condition] = None

    def when(self, left: str) -> "ConditionBuilder":
        """Start a condition with the left operand."""
        self._current = Condition(left=left)
        return self

    def equals(self, right: Any) -> "ConditionBuilder":
        """Set equals operator."""
        if self._current:
            self._current.operator = ConditionOperator.EQUALS
            self._current.right = right
            self._conditions.append(self._current)
            self._current = None
        return self

    def not_equals(self, right: Any) -> "ConditionBuilder":
        """Set not equals operator."""
        if self._current:
            self._current.operator = ConditionOperator.NOT_EQUALS
            self._current.right = right
            self._conditions.append(self._current)
            self._current = None
        return self

    def greater_than(self, right: Any) -> "ConditionBuilder":
        """Set greater than operator."""
        if self._current:
            self._current.operator = ConditionOperator.GREATER_THAN
            self._current.right = right
            self._conditions.append(self._current)
            self._current = None
        return self

    def less_than(self, right: Any) -> "ConditionBuilder":
        """Set less than operator."""
        if self._current:
            self._current.operator = ConditionOperator.LESS_THAN
            self._current.right = right
            self._conditions.append(self._current)
            self._current = None
        return self

    def contains(self, right: Any) -> "ConditionBuilder":
        """Set contains operator."""
        if self._current:
            self._current.operator = ConditionOperator.CONTAINS
            self._current.right = right
            self._conditions.append(self._current)
            self._current = None
        return self

    def matches(self, pattern: str) -> "ConditionBuilder":
        """Set regex matches operator."""
        if self._current:
            self._current.operator = ConditionOperator.MATCHES
            self._current.right = pattern
            self._conditions.append(self._current)
            self._current = None
        return self

    def is_null(self) -> "ConditionBuilder":
        """Set is null operator."""
        if self._current:
            self._current.operator = ConditionOperator.IS_NULL
            self._conditions.append(self._current)
            self._current = None
        return self

    def is_not_null(self) -> "ConditionBuilder":
        """Set is not null operator."""
        if self._current:
            self._current.operator = ConditionOperator.IS_NOT_NULL
            self._conditions.append(self._current)
            self._current = None
        return self

    def is_true(self) -> "ConditionBuilder":
        """Set is true operator."""
        if self._current:
            self._current.operator = ConditionOperator.IS_TRUE
            self._conditions.append(self._current)
            self._current = None
        return self

    def is_false(self) -> "ConditionBuilder":
        """Set is false operator."""
        if self._current:
            self._current.operator = ConditionOperator.IS_FALSE
            self._conditions.append(self._current)
            self._current = None
        return self

    def and_when(self, left: str) -> "ConditionBuilder":
        """Add an AND condition."""
        self._current = Condition(left=left)
        return self

    def or_when(self, left: str) -> "ConditionBuilder":
        """Add an OR condition (separate logic needed)."""
        # For OR, we'll handle it differently
        self._current = Condition(left=left)
        return self

    def negate(self) -> "ConditionBuilder":
        """Negate the last condition."""
        if self._conditions:
            self._conditions[-1].negate = True
        return self

    def build(self) -> Condition:
        """Build the final condition."""
        if not self._conditions:
            return Condition()

        if len(self._conditions) == 1:
            return self._conditions[0]

        # Combine with AND
        root = self._conditions[0]
        root.and_conditions = self._conditions[1:]
        return root

    def build_or(self) -> Condition:
        """Build with OR logic."""
        if not self._conditions:
            return Condition()

        if len(self._conditions) == 1:
            return self._conditions[0]

        # Combine with OR
        root = self._conditions[0]
        root.or_conditions = self._conditions[1:]
        return root


def condition(left: str) -> ConditionBuilder:
    """Create a condition builder."""
    return ConditionBuilder().when(left)
