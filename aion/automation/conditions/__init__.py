"""
AION Workflow Conditions

Condition evaluation for workflow branching:
- Expression parsing and evaluation
- Comparison operators
- Logical combinations (AND/OR)
"""

from aion.automation.conditions.evaluator import ConditionEvaluator
from aion.automation.conditions.operators import OperatorRegistry, compare
from aion.automation.conditions.expressions import ExpressionParser

__all__ = [
    "ConditionEvaluator",
    "OperatorRegistry",
    "compare",
    "ExpressionParser",
]
