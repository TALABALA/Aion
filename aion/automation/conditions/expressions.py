"""
AION Expression Parser

Expression language for workflow conditions.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ExpressionParser:
    """
    Parser for workflow expressions.

    Supports:
    - Variable references: inputs.name
    - Arithmetic: a + b, a * b
    - Comparisons: a > b, a == b
    - Boolean: a and b, a or b, not a
    - Function calls: len(items), sum(values)
    - Ternary: value if condition else default
    """

    # Safe built-in functions
    SAFE_FUNCTIONS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "all": all,
        "any": any,
        "filter": filter,
        "map": map,
        "isinstance": isinstance,
        "type": type,
    }

    # Safe operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Not: operator.not_,
        ast.Invert: operator.invert,
    }

    def __init__(self, variables: Dict[str, Any] = None):
        self.variables = variables or {}
        self.functions = self.SAFE_FUNCTIONS.copy()

    def register_function(self, name: str, func: Callable) -> None:
        """Register a custom function."""
        self.functions[name] = func

    def set_variables(self, variables: Dict[str, Any]) -> None:
        """Set variables for evaluation."""
        self.variables = variables

    def parse(self, expression: str) -> Any:
        """
        Parse and evaluate an expression.

        Args:
            expression: Expression string

        Returns:
            Evaluated result
        """
        if not expression or not expression.strip():
            return None

        try:
            # Parse the expression
            tree = ast.parse(expression.strip(), mode="eval")

            # Evaluate safely
            return self._eval_node(tree.body)

        except SyntaxError as e:
            logger.error("expression_syntax_error", expression=expression, error=str(e))
            raise ValueError(f"Invalid expression syntax: {expression}") from e
        except Exception as e:
            logger.error("expression_eval_error", expression=expression, error=str(e))
            raise ValueError(f"Expression evaluation error: {expression} - {e}") from e

    def _eval_node(self, node: ast.AST) -> Any:
        """Evaluate an AST node."""
        # Literal values
        if isinstance(node, ast.Constant):
            return node.value

        # For older Python compatibility
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Str):
            return node.s

        # Name (variable reference)
        if isinstance(node, ast.Name):
            name = node.id

            # Check built-in constants
            if name == "True":
                return True
            if name == "False":
                return False
            if name == "None":
                return None

            # Check functions
            if name in self.functions:
                return self.functions[name]

            # Check variables
            if name in self.variables:
                return self.variables[name]

            # Unknown name - return None instead of raising
            return None

        # Attribute access (e.g., inputs.name)
        if isinstance(node, ast.Attribute):
            value = self._eval_node(node.value)
            attr = node.attr

            if value is None:
                return None

            if isinstance(value, dict):
                return value.get(attr)

            return getattr(value, attr, None)

        # Subscript (e.g., items[0], data["key"])
        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            if value is None:
                return None

            # Handle slice vs index
            if isinstance(node.slice, ast.Slice):
                lower = self._eval_node(node.slice.lower) if node.slice.lower else None
                upper = self._eval_node(node.slice.upper) if node.slice.upper else None
                step = self._eval_node(node.slice.step) if node.slice.step else None
                return value[lower:upper:step]
            else:
                # Index (Python 3.9+) or directly the slice (Python 3.8)
                idx = self._eval_node(node.slice)
                try:
                    return value[idx]
                except (KeyError, IndexError, TypeError):
                    return None

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(left, right)
            raise ValueError(f"Unsupported operator: {type(node.op)}")

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(operand)
            raise ValueError(f"Unsupported operator: {type(node.op)}")

        # Comparisons
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)

            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                op_func = self.OPERATORS.get(type(op))
                if not op_func:
                    raise ValueError(f"Unsupported comparison: {type(op)}")
                if not op_func(left, right):
                    return False
                left = right

            return True

        # Boolean operations
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not self._eval_node(value):
                        return False
                return True
            elif isinstance(node.op, ast.Or):
                for value in node.values:
                    if self._eval_node(value):
                        return True
                return False

        # If expression (ternary)
        if isinstance(node, ast.IfExp):
            condition = self._eval_node(node.test)
            if condition:
                return self._eval_node(node.body)
            return self._eval_node(node.orelse)

        # Function call
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            if not callable(func):
                raise ValueError(f"Not callable: {node.func}")

            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords}

            return func(*args, **kwargs)

        # List
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        # Tuple
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)

        # Dict
        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k): self._eval_node(v)
                for k, v in zip(node.keys, node.values)
            }

        # Set
        if isinstance(node, ast.Set):
            return {self._eval_node(elt) for elt in node.elts}

        # Lambda (limited support)
        if isinstance(node, ast.Lambda):
            raise ValueError("Lambda expressions not supported for security reasons")

        raise ValueError(f"Unsupported expression type: {type(node)}")


def evaluate_expression(expression: str, variables: Dict[str, Any] = None) -> Any:
    """
    Evaluate an expression with variables.

    Args:
        expression: Expression string
        variables: Variable values

    Returns:
        Evaluated result
    """
    parser = ExpressionParser(variables)
    return parser.parse(expression)
