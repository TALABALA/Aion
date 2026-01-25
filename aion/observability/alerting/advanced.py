"""
Advanced Alerting System

SOTA alerting capabilities:
- PromQL-like query language
- Alert grouping and deduplication
- Correlation analysis
- Escalation policies with on-call rotation
- Maintenance windows
- Alert routing and silencing
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

from aion.observability.types import Alert, AlertRule, AlertSeverity, AlertState

logger = logging.getLogger(__name__)


# =============================================================================
# PromQL-Like Query Language
# =============================================================================

class MetricQueryTokenType(Enum):
    """Token types for the metric query language."""
    IDENTIFIER = "identifier"
    NUMBER = "number"
    STRING = "string"
    OPERATOR = "operator"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LBRACE = "lbrace"
    RBRACE = "rbrace"
    LBRACKET = "lbracket"
    RBRACKET = "rbracket"
    COMMA = "comma"
    COLON = "colon"
    EOF = "eof"


@dataclass
class Token:
    """A token in the query language."""
    type: MetricQueryTokenType
    value: str
    position: int = 0


class MetricQueryLexer:
    """Lexer for the metric query language."""

    OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "=~", "!~", "+", "-", "*", "/", "%", "^", "and", "or", "unless"}

    def __init__(self, query: str):
        self.query = query
        self.pos = 0
        self.length = len(query)

    def tokenize(self) -> List[Token]:
        """Tokenize the query string."""
        tokens = []

        while self.pos < self.length:
            char = self.query[self.pos]

            # Skip whitespace
            if char.isspace():
                self.pos += 1
                continue

            # String literals
            if char in ('"', "'"):
                tokens.append(self._read_string(char))
                continue

            # Numbers
            if char.isdigit() or (char == '.' and self.pos + 1 < self.length and self.query[self.pos + 1].isdigit()):
                tokens.append(self._read_number())
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                tokens.append(self._read_identifier())
                continue

            # Operators
            for op in sorted(self.OPERATORS, key=len, reverse=True):
                if self.query[self.pos:self.pos + len(op)] == op:
                    tokens.append(Token(MetricQueryTokenType.OPERATOR, op, self.pos))
                    self.pos += len(op)
                    break
            else:
                # Single character tokens
                token_map = {
                    "(": MetricQueryTokenType.LPAREN,
                    ")": MetricQueryTokenType.RPAREN,
                    "{": MetricQueryTokenType.LBRACE,
                    "}": MetricQueryTokenType.RBRACE,
                    "[": MetricQueryTokenType.LBRACKET,
                    "]": MetricQueryTokenType.RBRACKET,
                    ",": MetricQueryTokenType.COMMA,
                    ":": MetricQueryTokenType.COLON,
                }
                if char in token_map:
                    tokens.append(Token(token_map[char], char, self.pos))
                    self.pos += 1
                else:
                    # Unknown character, skip
                    self.pos += 1

        tokens.append(Token(MetricQueryTokenType.EOF, "", self.pos))
        return tokens

    def _read_string(self, quote: str) -> Token:
        """Read a string literal."""
        start = self.pos
        self.pos += 1  # Skip opening quote
        value = ""

        while self.pos < self.length:
            char = self.query[self.pos]
            if char == quote:
                self.pos += 1
                break
            if char == "\\" and self.pos + 1 < self.length:
                self.pos += 1
                value += self.query[self.pos]
            else:
                value += char
            self.pos += 1

        return Token(MetricQueryTokenType.STRING, value, start)

    def _read_number(self) -> Token:
        """Read a numeric literal."""
        start = self.pos
        value = ""

        while self.pos < self.length:
            char = self.query[self.pos]
            if char.isdigit() or char == "." or char in "eE+-":
                value += char
                self.pos += 1
            else:
                break

        return Token(MetricQueryTokenType.NUMBER, value, start)

    def _read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start = self.pos
        value = ""

        while self.pos < self.length:
            char = self.query[self.pos]
            if char.isalnum() or char == "_":
                value += char
                self.pos += 1
            else:
                break

        # Check if it's an operator keyword
        if value.lower() in ("and", "or", "unless"):
            return Token(MetricQueryTokenType.OPERATOR, value.lower(), start)

        return Token(MetricQueryTokenType.IDENTIFIER, value, start)


@dataclass
class LabelMatcher:
    """A label matcher for filtering metrics."""
    label: str
    operator: str  # ==, !=, =~, !~
    value: str
    _pattern: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        if self.operator in ("=~", "!~"):
            self._pattern = re.compile(self.value)

    def matches(self, labels: Dict[str, str]) -> bool:
        """Check if labels match this matcher."""
        actual = labels.get(self.label, "")

        if self.operator == "==":
            return actual == self.value
        elif self.operator == "!=":
            return actual != self.value
        elif self.operator == "=~":
            return bool(self._pattern and self._pattern.search(actual))
        elif self.operator == "!~":
            return not (self._pattern and self._pattern.search(actual))

        return False


@dataclass
class MetricSelector:
    """Selector for metrics with label matchers."""
    metric_name: str
    matchers: List[LabelMatcher] = field(default_factory=list)
    range_duration: Optional[str] = None  # e.g., "5m", "1h"

    def matches(self, name: str, labels: Dict[str, str]) -> bool:
        """Check if a metric matches this selector."""
        if name != self.metric_name:
            return False

        for matcher in self.matchers:
            if not matcher.matches(labels):
                return False

        return True


class AggregationType(Enum):
    """Aggregation functions."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STDDEV = "stddev"
    RATE = "rate"
    IRATE = "irate"
    INCREASE = "increase"
    DELTA = "delta"
    HISTOGRAM_QUANTILE = "histogram_quantile"
    TOPK = "topk"
    BOTTOMK = "bottomk"
    ABS = "abs"
    CEIL = "ceil"
    FLOOR = "floor"
    ROUND = "round"
    CLAMP = "clamp"
    CLAMP_MAX = "clamp_max"
    CLAMP_MIN = "clamp_min"


@dataclass
class AggregationExpr:
    """An aggregation expression."""
    function: AggregationType
    selector: MetricSelector
    by_labels: List[str] = field(default_factory=list)
    without_labels: List[str] = field(default_factory=list)
    args: List[float] = field(default_factory=list)  # For functions like topk(5, ...)


@dataclass
class BinaryExpr:
    """A binary expression (left op right)."""
    left: Union["QueryExpr", float]
    operator: str
    right: Union["QueryExpr", float]


QueryExpr = Union[MetricSelector, AggregationExpr, BinaryExpr, float]


class MetricQueryParser:
    """Parser for the PromQL-like query language."""

    AGGREGATION_FUNCTIONS = {f.value for f in AggregationType}

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> QueryExpr:
        """Parse the token stream into an expression."""
        return self._parse_or_expr()

    def _current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(MetricQueryTokenType.EOF, "", 0)

    def _advance(self) -> Token:
        token = self._current()
        self.pos += 1
        return token

    def _expect(self, token_type: MetricQueryTokenType) -> Token:
        token = self._advance()
        if token.type != token_type:
            raise ValueError(f"Expected {token_type}, got {token.type}")
        return token

    def _parse_or_expr(self) -> QueryExpr:
        left = self._parse_and_expr()

        while self._current().type == MetricQueryTokenType.OPERATOR and self._current().value == "or":
            self._advance()
            right = self._parse_and_expr()
            left = BinaryExpr(left, "or", right)

        return left

    def _parse_and_expr(self) -> QueryExpr:
        left = self._parse_comparison_expr()

        while self._current().type == MetricQueryTokenType.OPERATOR and self._current().value in ("and", "unless"):
            op = self._advance().value
            right = self._parse_comparison_expr()
            left = BinaryExpr(left, op, right)

        return left

    def _parse_comparison_expr(self) -> QueryExpr:
        left = self._parse_additive_expr()

        if self._current().type == MetricQueryTokenType.OPERATOR:
            op = self._current().value
            if op in ("==", "!=", ">", ">=", "<", "<="):
                self._advance()
                right = self._parse_additive_expr()
                return BinaryExpr(left, op, right)

        return left

    def _parse_additive_expr(self) -> QueryExpr:
        left = self._parse_multiplicative_expr()

        while self._current().type == MetricQueryTokenType.OPERATOR and self._current().value in ("+", "-"):
            op = self._advance().value
            right = self._parse_multiplicative_expr()
            left = BinaryExpr(left, op, right)

        return left

    def _parse_multiplicative_expr(self) -> QueryExpr:
        left = self._parse_power_expr()

        while self._current().type == MetricQueryTokenType.OPERATOR and self._current().value in ("*", "/", "%"):
            op = self._advance().value
            right = self._parse_power_expr()
            left = BinaryExpr(left, op, right)

        return left

    def _parse_power_expr(self) -> QueryExpr:
        left = self._parse_unary_expr()

        if self._current().type == MetricQueryTokenType.OPERATOR and self._current().value == "^":
            self._advance()
            right = self._parse_power_expr()
            return BinaryExpr(left, "^", right)

        return left

    def _parse_unary_expr(self) -> QueryExpr:
        if self._current().type == MetricQueryTokenType.OPERATOR and self._current().value == "-":
            self._advance()
            expr = self._parse_unary_expr()
            return BinaryExpr(0.0, "-", expr)

        return self._parse_primary_expr()

    def _parse_primary_expr(self) -> QueryExpr:
        token = self._current()

        # Number
        if token.type == MetricQueryTokenType.NUMBER:
            self._advance()
            return float(token.value)

        # Aggregation function or metric
        if token.type == MetricQueryTokenType.IDENTIFIER:
            name = token.value.lower()

            if name in self.AGGREGATION_FUNCTIONS:
                return self._parse_aggregation()

            return self._parse_metric_selector()

        # Parenthesized expression
        if token.type == MetricQueryTokenType.LPAREN:
            self._advance()
            expr = self._parse_or_expr()
            self._expect(MetricQueryTokenType.RPAREN)
            return expr

        raise ValueError(f"Unexpected token: {token}")

    def _parse_aggregation(self) -> AggregationExpr:
        func_name = self._advance().value
        func = AggregationType(func_name.lower())

        by_labels = []
        without_labels = []
        args = []

        # Check for 'by' or 'without' before arguments
        if self._current().type == MetricQueryTokenType.IDENTIFIER:
            modifier = self._current().value.lower()
            if modifier == "by":
                self._advance()
                by_labels = self._parse_label_list()
            elif modifier == "without":
                self._advance()
                without_labels = self._parse_label_list()

        self._expect(MetricQueryTokenType.LPAREN)

        # Parse arguments
        if self._current().type == MetricQueryTokenType.NUMBER:
            args.append(float(self._advance().value))
            if self._current().type == MetricQueryTokenType.COMMA:
                self._advance()

        # Parse inner expression
        inner = self._parse_or_expr()

        self._expect(MetricQueryTokenType.RPAREN)

        # Check for 'by' or 'without' after arguments
        if self._current().type == MetricQueryTokenType.IDENTIFIER:
            modifier = self._current().value.lower()
            if modifier == "by":
                self._advance()
                by_labels = self._parse_label_list()
            elif modifier == "without":
                self._advance()
                without_labels = self._parse_label_list()

        if isinstance(inner, MetricSelector):
            return AggregationExpr(
                function=func,
                selector=inner,
                by_labels=by_labels,
                without_labels=without_labels,
                args=args,
            )

        # Handle nested expressions
        return AggregationExpr(
            function=func,
            selector=MetricSelector(metric_name="__nested__"),
            by_labels=by_labels,
            without_labels=without_labels,
            args=args,
        )

    def _parse_metric_selector(self) -> MetricSelector:
        metric_name = self._advance().value
        matchers = []
        range_duration = None

        # Label matchers
        if self._current().type == MetricQueryTokenType.LBRACE:
            self._advance()
            matchers = self._parse_label_matchers()
            self._expect(MetricQueryTokenType.RBRACE)

        # Range vector
        if self._current().type == MetricQueryTokenType.LBRACKET:
            self._advance()
            range_duration = self._advance().value
            self._expect(MetricQueryTokenType.RBRACKET)

        return MetricSelector(
            metric_name=metric_name,
            matchers=matchers,
            range_duration=range_duration,
        )

    def _parse_label_matchers(self) -> List[LabelMatcher]:
        matchers = []

        while self._current().type != MetricQueryTokenType.RBRACE:
            label = self._expect(MetricQueryTokenType.IDENTIFIER).value
            operator = self._expect(MetricQueryTokenType.OPERATOR).value
            value = self._expect(MetricQueryTokenType.STRING).value

            matchers.append(LabelMatcher(label=label, operator=operator, value=value))

            if self._current().type == MetricQueryTokenType.COMMA:
                self._advance()

        return matchers

    def _parse_label_list(self) -> List[str]:
        self._expect(MetricQueryTokenType.LPAREN)
        labels = []

        while self._current().type != MetricQueryTokenType.RPAREN:
            labels.append(self._expect(MetricQueryTokenType.IDENTIFIER).value)
            if self._current().type == MetricQueryTokenType.COMMA:
                self._advance()

        self._expect(MetricQueryTokenType.RPAREN)
        return labels


class MetricQueryEngine:
    """
    Engine for evaluating PromQL-like queries.

    Evaluates queries against a metrics store.
    """

    def __init__(self, metrics_engine: Any):
        self.metrics_engine = metrics_engine

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        lexer = MetricQueryLexer(query_str)
        tokens = lexer.tokenize()
        parser = MetricQueryParser(tokens)
        expr = parser.parse()

        return self._evaluate(expr)

    def _evaluate(self, expr: QueryExpr) -> List[Dict[str, Any]]:
        """Evaluate an expression."""
        if isinstance(expr, float):
            return [{"labels": {}, "value": expr}]

        if isinstance(expr, MetricSelector):
            return self._evaluate_selector(expr)

        if isinstance(expr, AggregationExpr):
            return self._evaluate_aggregation(expr)

        if isinstance(expr, BinaryExpr):
            return self._evaluate_binary(expr)

        return []

    def _evaluate_selector(self, selector: MetricSelector) -> List[Dict[str, Any]]:
        """Evaluate a metric selector."""
        results = []

        # Get all time series matching the selector
        for name, series in self.metrics_engine._time_series.items():
            base_name = name.split("{")[0]
            if base_name != selector.metric_name:
                continue

            # Parse labels from key
            labels = {}
            if "{" in name:
                label_str = name.split("{")[1].rstrip("}")
                for pair in label_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        labels[k] = v.strip('"')

            # Check matchers
            match = True
            for matcher in selector.matchers:
                if not matcher.matches(labels):
                    match = False
                    break

            if match:
                # Get the latest value
                if series:
                    latest = series[-1]
                    results.append({
                        "labels": labels,
                        "value": latest[1] if isinstance(latest, tuple) else latest,
                        "timestamp": latest[0] if isinstance(latest, tuple) else None,
                    })

        return results

    def _evaluate_aggregation(self, expr: AggregationExpr) -> List[Dict[str, Any]]:
        """Evaluate an aggregation expression."""
        import numpy as np

        # Get input data
        inputs = self._evaluate(expr.selector)

        if not inputs:
            return []

        # Group by labels
        groups: Dict[str, List[float]] = defaultdict(list)

        for item in inputs:
            # Build group key
            if expr.by_labels:
                key = ",".join(f"{k}={item['labels'].get(k, '')}" for k in expr.by_labels)
            elif expr.without_labels:
                key = ",".join(
                    f"{k}={v}" for k, v in sorted(item['labels'].items())
                    if k not in expr.without_labels
                )
            else:
                key = ""

            groups[key].append(item["value"])

        # Apply aggregation function
        results = []

        for key, values in groups.items():
            labels = {}
            if key:
                for pair in key.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        labels[k] = v

            if expr.function == AggregationType.SUM:
                agg_value = sum(values)
            elif expr.function == AggregationType.AVG:
                agg_value = np.mean(values)
            elif expr.function == AggregationType.MIN:
                agg_value = min(values)
            elif expr.function == AggregationType.MAX:
                agg_value = max(values)
            elif expr.function == AggregationType.COUNT:
                agg_value = len(values)
            elif expr.function == AggregationType.STDDEV:
                agg_value = np.std(values)
            elif expr.function == AggregationType.TOPK:
                k = int(expr.args[0]) if expr.args else 10
                # Return top k values
                sorted_values = sorted(values, reverse=True)[:k]
                for v in sorted_values:
                    results.append({"labels": labels, "value": v})
                continue
            elif expr.function == AggregationType.ABS:
                agg_value = abs(values[0]) if values else 0
            elif expr.function == AggregationType.RATE:
                # Simple rate calculation
                if len(values) >= 2:
                    agg_value = (values[-1] - values[0]) / max(len(values), 1)
                else:
                    agg_value = 0
            else:
                agg_value = sum(values)  # Default fallback

            results.append({"labels": labels, "value": float(agg_value)})

        return results

    def _evaluate_binary(self, expr: BinaryExpr) -> List[Dict[str, Any]]:
        """Evaluate a binary expression."""
        left_results = self._evaluate(expr.left) if not isinstance(expr.left, float) else [{"labels": {}, "value": expr.left}]
        right_results = self._evaluate(expr.right) if not isinstance(expr.right, float) else [{"labels": {}, "value": expr.right}]

        results = []

        # Vector matching (simplified)
        for left in left_results:
            for right in right_results:
                # Check label matching for vector operations
                left_val = left["value"]
                right_val = right["value"]

                if expr.operator == "+":
                    result = left_val + right_val
                elif expr.operator == "-":
                    result = left_val - right_val
                elif expr.operator == "*":
                    result = left_val * right_val
                elif expr.operator == "/":
                    result = left_val / right_val if right_val != 0 else 0
                elif expr.operator == "%":
                    result = left_val % right_val if right_val != 0 else 0
                elif expr.operator == "^":
                    result = left_val ** right_val
                elif expr.operator == "==":
                    result = 1 if left_val == right_val else 0
                elif expr.operator == "!=":
                    result = 1 if left_val != right_val else 0
                elif expr.operator == ">":
                    result = 1 if left_val > right_val else 0
                elif expr.operator == ">=":
                    result = 1 if left_val >= right_val else 0
                elif expr.operator == "<":
                    result = 1 if left_val < right_val else 0
                elif expr.operator == "<=":
                    result = 1 if left_val <= right_val else 0
                elif expr.operator == "and":
                    result = left_val if left_val and right_val else 0
                elif expr.operator == "or":
                    result = left_val if left_val else right_val
                else:
                    result = 0

                results.append({
                    "labels": {**left["labels"], **right["labels"]},
                    "value": result,
                })

        return results


# =============================================================================
# Alert Grouping and Deduplication
# =============================================================================

@dataclass
class AlertGroup:
    """A group of related alerts."""
    group_key: str
    alerts: List[Alert] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    state: AlertState = AlertState.PENDING

    @property
    def count(self) -> int:
        return len(self.alerts)

    @property
    def severity(self) -> AlertSeverity:
        """Return highest severity in group."""
        if not self.alerts:
            return AlertSeverity.INFO
        severities = [a.severity for a in self.alerts]
        order = [AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO]
        for sev in order:
            if sev in severities:
                return sev
        return AlertSeverity.INFO

    def add_alert(self, alert: Alert) -> None:
        """Add an alert to the group."""
        self.alerts.append(alert)
        self.updated_at = datetime.utcnow()

        # Update group state
        if any(a.state == AlertState.FIRING for a in self.alerts):
            self.state = AlertState.FIRING
        elif any(a.state == AlertState.PENDING for a in self.alerts):
            self.state = AlertState.PENDING
        else:
            self.state = AlertState.RESOLVED


class AlertGrouper:
    """
    Groups related alerts together.

    Grouping is based on configurable labels and time windows.
    """

    def __init__(
        self,
        group_by: List[str] = None,
        group_wait: float = 30.0,  # seconds to wait before sending first notification
        group_interval: float = 300.0,  # seconds between notifications for same group
        repeat_interval: float = 3600.0,  # seconds before re-notifying for same alert
    ):
        self.group_by = group_by or ["alertname", "severity"]
        self.group_wait = group_wait
        self.group_interval = group_interval
        self.repeat_interval = repeat_interval

        self._groups: Dict[str, AlertGroup] = {}
        self._last_notification: Dict[str, float] = {}

    def get_group_key(self, alert: Alert) -> str:
        """Generate group key for an alert."""
        parts = []
        for label in self.group_by:
            value = alert.labels.get(label, alert.name if label == "alertname" else "")
            parts.append(f"{label}={value}")
        return ",".join(sorted(parts))

    def add_alert(self, alert: Alert) -> AlertGroup:
        """Add an alert to its group."""
        key = self.get_group_key(alert)

        if key not in self._groups:
            self._groups[key] = AlertGroup(group_key=key)

        group = self._groups[key]

        # Check for duplicate
        for existing in group.alerts:
            if self._is_duplicate(existing, alert):
                # Update existing instead of adding
                existing.state = alert.state
                existing.value = alert.value
                existing.last_updated = datetime.utcnow()
                group.updated_at = datetime.utcnow()
                return group

        group.add_alert(alert)
        return group

    def _is_duplicate(self, a1: Alert, a2: Alert) -> bool:
        """Check if two alerts are duplicates."""
        return (
            a1.name == a2.name and
            a1.labels == a2.labels and
            a1.severity == a2.severity
        )

    def should_notify(self, group: AlertGroup) -> bool:
        """Check if a notification should be sent for this group."""
        now = time.time()
        key = group.group_key

        # First notification - wait group_wait time
        if key not in self._last_notification:
            group_age = (datetime.utcnow() - group.created_at).total_seconds()
            if group_age >= self.group_wait:
                return True
            return False

        # Subsequent notifications
        last = self._last_notification[key]
        elapsed = now - last

        if group.state == AlertState.FIRING:
            return elapsed >= self.group_interval

        return False

    def mark_notified(self, group: AlertGroup) -> None:
        """Mark a group as having been notified."""
        self._last_notification[group.group_key] = time.time()

    def get_groups(self) -> List[AlertGroup]:
        """Get all alert groups."""
        return list(self._groups.values())

    def get_firing_groups(self) -> List[AlertGroup]:
        """Get groups with firing alerts."""
        return [g for g in self._groups.values() if g.state == AlertState.FIRING]

    def cleanup_resolved(self, max_age_seconds: float = 3600.0) -> int:
        """Remove old resolved groups."""
        now = datetime.utcnow()
        to_remove = []

        for key, group in self._groups.items():
            if group.state == AlertState.RESOLVED:
                age = (now - group.updated_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(key)

        for key in to_remove:
            del self._groups[key]
            self._last_notification.pop(key, None)

        return len(to_remove)


class AlertDeduplicator:
    """
    Deduplicates alerts based on content hash.

    Prevents sending the same alert multiple times.
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        max_entries: int = 10000,
    ):
        self.window_seconds = window_seconds
        self.max_entries = max_entries
        self._seen: Dict[str, float] = {}

    def _hash_alert(self, alert: Alert) -> str:
        """Generate hash for an alert."""
        content = f"{alert.name}:{alert.severity.value}:{sorted(alert.labels.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate."""
        self._cleanup()

        hash_key = self._hash_alert(alert)
        now = time.time()

        if hash_key in self._seen:
            return True

        self._seen[hash_key] = now
        return False

    def _cleanup(self) -> None:
        """Remove old entries."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove expired
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

        # Trim if too large
        if len(self._seen) > self.max_entries:
            sorted_items = sorted(self._seen.items(), key=lambda x: x[1])
            self._seen = dict(sorted_items[-self.max_entries // 2:])


# =============================================================================
# Alert Correlation
# =============================================================================

@dataclass
class CorrelatedAlertSet:
    """A set of correlated alerts."""
    alerts: List[Alert]
    correlation_type: str
    correlation_score: float
    root_cause_candidate: Optional[Alert] = None
    description: str = ""


class AlertCorrelator:
    """
    Correlates related alerts to identify root causes.

    Uses temporal proximity, label similarity, and causal relationships.
    """

    def __init__(
        self,
        time_window_seconds: float = 60.0,
        min_correlation_score: float = 0.5,
    ):
        self.time_window_seconds = time_window_seconds
        self.min_correlation_score = min_correlation_score
        self._alert_history: List[Tuple[float, Alert]] = []

    def add_alert(self, alert: Alert) -> None:
        """Add an alert for correlation analysis."""
        self._alert_history.append((time.time(), alert))

        # Trim old alerts
        cutoff = time.time() - self.time_window_seconds * 10
        self._alert_history = [(t, a) for t, a in self._alert_history if t > cutoff]

    def find_correlations(self, alert: Alert) -> List[CorrelatedAlertSet]:
        """Find alerts correlated with the given alert."""
        correlations = []
        now = time.time()

        # Get recent alerts
        recent = [
            (t, a) for t, a in self._alert_history
            if now - t <= self.time_window_seconds and a != alert
        ]

        if not recent:
            return correlations

        # Temporal correlation
        temporal_group = []
        for t, a in recent:
            score = 1.0 - (now - t) / self.time_window_seconds
            if score >= self.min_correlation_score:
                temporal_group.append((a, score))

        if temporal_group:
            alerts, scores = zip(*temporal_group)
            avg_score = sum(scores) / len(scores)

            correlations.append(CorrelatedAlertSet(
                alerts=list(alerts) + [alert],
                correlation_type="temporal",
                correlation_score=avg_score,
                description=f"Alerts occurring within {self.time_window_seconds}s window",
            ))

        # Label similarity correlation
        for t, a in recent:
            similarity = self._label_similarity(alert, a)
            if similarity >= self.min_correlation_score:
                correlations.append(CorrelatedAlertSet(
                    alerts=[a, alert],
                    correlation_type="label_similarity",
                    correlation_score=similarity,
                    description=f"Alerts with similar labels (score: {similarity:.2f})",
                ))

        # Service dependency correlation (if service labels exist)
        service_alerts = defaultdict(list)
        for t, a in recent + [(now, alert)]:
            service = a.labels.get("service", a.labels.get("instance", ""))
            if service:
                service_alerts[service].append(a)

        for service, svc_alerts in service_alerts.items():
            if len(svc_alerts) > 1:
                correlations.append(CorrelatedAlertSet(
                    alerts=svc_alerts,
                    correlation_type="service",
                    correlation_score=0.8,
                    description=f"Multiple alerts for service: {service}",
                ))

        return correlations

    def _label_similarity(self, a1: Alert, a2: Alert) -> float:
        """Calculate Jaccard similarity between alert labels."""
        keys1 = set(a1.labels.keys())
        keys2 = set(a2.labels.keys())

        if not keys1 and not keys2:
            return 1.0

        intersection = keys1 & keys2
        union = keys1 | keys2

        if not union:
            return 0.0

        # Check value similarity for common keys
        value_matches = sum(
            1 for k in intersection
            if a1.labels.get(k) == a2.labels.get(k)
        )

        return value_matches / len(union)

    def identify_root_cause(self, correlations: List[CorrelatedAlertSet]) -> Optional[Alert]:
        """Attempt to identify root cause from correlated alerts."""
        if not correlations:
            return None

        # Combine all correlated alerts
        all_alerts = []
        for corr in correlations:
            all_alerts.extend(corr.alerts)

        if not all_alerts:
            return None

        # Heuristics for root cause:
        # 1. First alert in time
        # 2. Highest severity
        # 3. Most common labels

        # Sort by timestamp
        sorted_alerts = sorted(
            all_alerts,
            key=lambda a: (a.fired_at or datetime.max, -a.severity.value if hasattr(a.severity, 'value') else 0)
        )

        return sorted_alerts[0] if sorted_alerts else None


# =============================================================================
# Escalation Policies
# =============================================================================

@dataclass
class EscalationLevel:
    """A single level in an escalation policy."""
    level: int
    delay_minutes: float
    notify_channels: List[str]
    notify_users: List[str] = field(default_factory=list)
    notify_teams: List[str] = field(default_factory=list)
    repeat_count: int = 1
    repeat_interval_minutes: float = 5.0


@dataclass
class OnCallSchedule:
    """On-call rotation schedule."""
    name: str
    users: List[str]
    rotation_interval_hours: float = 168.0  # Weekly default
    start_time: datetime = field(default_factory=datetime.utcnow)
    timezone: str = "UTC"

    def get_current_oncall(self) -> str:
        """Get the current on-call user."""
        if not self.users:
            return ""

        now = datetime.utcnow()
        elapsed = (now - self.start_time).total_seconds() / 3600
        rotation_count = int(elapsed / self.rotation_interval_hours)
        current_idx = rotation_count % len(self.users)

        return self.users[current_idx]

    def get_oncall_at(self, when: datetime) -> str:
        """Get the on-call user at a specific time."""
        if not self.users:
            return ""

        elapsed = (when - self.start_time).total_seconds() / 3600
        rotation_count = int(elapsed / self.rotation_interval_hours)
        current_idx = rotation_count % len(self.users)

        return self.users[current_idx]


@dataclass
class EscalationPolicy:
    """
    Escalation policy defining notification tiers and timing.

    Supports multi-level escalation with on-call rotation.
    """
    name: str
    description: str = ""
    levels: List[EscalationLevel] = field(default_factory=list)
    on_call_schedule: Optional[OnCallSchedule] = None
    enabled: bool = True

    def get_level_for_duration(self, minutes_since_fired: float) -> Optional[EscalationLevel]:
        """Get the appropriate escalation level based on time since alert fired."""
        if not self.levels:
            return None

        cumulative_delay = 0.0
        for level in sorted(self.levels, key=lambda l: l.level):
            cumulative_delay += level.delay_minutes
            if minutes_since_fired < cumulative_delay:
                return level

        # Return highest level if past all delays
        return max(self.levels, key=lambda l: l.level)


class EscalationManager:
    """
    Manages alert escalation and on-call routing.
    """

    def __init__(self):
        self._policies: Dict[str, EscalationPolicy] = {}
        self._schedules: Dict[str, OnCallSchedule] = {}
        self._escalation_state: Dict[str, Dict] = {}  # alert_id -> state

    def add_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy."""
        self._policies[policy.name] = policy

    def add_schedule(self, schedule: OnCallSchedule) -> None:
        """Add an on-call schedule."""
        self._schedules[schedule.name] = schedule

    def get_policy(self, name: str) -> Optional[EscalationPolicy]:
        """Get an escalation policy by name."""
        return self._policies.get(name)

    def get_on_call(self, schedule_name: str) -> Optional[str]:
        """Get current on-call user for a schedule."""
        schedule = self._schedules.get(schedule_name)
        if schedule:
            return schedule.get_current_oncall()
        return None

    def process_alert(
        self,
        alert: Alert,
        policy_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Process an alert through escalation policy.

        Returns list of notification targets.
        """
        policy = self._policies.get(policy_name)
        if not policy or not policy.enabled:
            return []

        alert_id = f"{alert.name}:{hash(frozenset(alert.labels.items()))}"

        # Initialize state
        if alert_id not in self._escalation_state:
            self._escalation_state[alert_id] = {
                "first_fired": datetime.utcnow(),
                "current_level": 0,
                "notification_count": 0,
                "last_notification": None,
            }

        state = self._escalation_state[alert_id]

        # Calculate time since fired
        minutes_since_fired = (datetime.utcnow() - state["first_fired"]).total_seconds() / 60

        # Get appropriate level
        level = policy.get_level_for_duration(minutes_since_fired)
        if not level:
            return []

        # Check if we should notify
        should_notify = False

        if state["current_level"] < level.level:
            # Escalated to new level
            state["current_level"] = level.level
            state["notification_count"] = 0
            should_notify = True
        elif state["notification_count"] < level.repeat_count:
            # Still in repeat window
            if state["last_notification"]:
                elapsed = (datetime.utcnow() - state["last_notification"]).total_seconds() / 60
                if elapsed >= level.repeat_interval_minutes:
                    should_notify = True
            else:
                should_notify = True

        if not should_notify:
            return []

        # Build notification targets
        targets = []

        for channel in level.notify_channels:
            targets.append({
                "type": "channel",
                "target": channel,
                "level": level.level,
            })

        for user in level.notify_users:
            targets.append({
                "type": "user",
                "target": user,
                "level": level.level,
            })

        # Add on-call if configured
        if policy.on_call_schedule:
            on_call = policy.on_call_schedule.get_current_oncall()
            if on_call:
                targets.append({
                    "type": "oncall",
                    "target": on_call,
                    "schedule": policy.on_call_schedule.name,
                    "level": level.level,
                })

        # Update state
        state["notification_count"] += 1
        state["last_notification"] = datetime.utcnow()

        return targets

    def resolve_alert(self, alert: Alert) -> None:
        """Mark an alert as resolved, stopping escalation."""
        alert_id = f"{alert.name}:{hash(frozenset(alert.labels.items()))}"
        self._escalation_state.pop(alert_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get escalation manager statistics."""
        return {
            "policies": len(self._policies),
            "schedules": len(self._schedules),
            "active_escalations": len(self._escalation_state),
        }


# =============================================================================
# Maintenance Windows
# =============================================================================

@dataclass
class MaintenanceWindow:
    """
    A scheduled maintenance window during which alerts are silenced.
    """
    id: str
    name: str
    start_time: datetime
    end_time: datetime
    matchers: List[LabelMatcher] = field(default_factory=list)
    created_by: str = ""
    comment: str = ""

    def is_active(self, now: Optional[datetime] = None) -> bool:
        """Check if the maintenance window is currently active."""
        now = now or datetime.utcnow()
        return self.start_time <= now <= self.end_time

    def matches_alert(self, alert: Alert) -> bool:
        """Check if an alert matches this maintenance window."""
        if not self.matchers:
            return True  # No matchers = match all

        for matcher in self.matchers:
            if not matcher.matches(alert.labels):
                return False

        return True


class MaintenanceManager:
    """Manages maintenance windows."""

    def __init__(self):
        self._windows: Dict[str, MaintenanceWindow] = {}

    def add_window(self, window: MaintenanceWindow) -> None:
        """Add a maintenance window."""
        self._windows[window.id] = window

    def remove_window(self, window_id: str) -> bool:
        """Remove a maintenance window."""
        if window_id in self._windows:
            del self._windows[window_id]
            return True
        return False

    def is_silenced(self, alert: Alert) -> Tuple[bool, Optional[MaintenanceWindow]]:
        """Check if an alert should be silenced due to maintenance."""
        now = datetime.utcnow()

        for window in self._windows.values():
            if window.is_active(now) and window.matches_alert(alert):
                return True, window

        return False, None

    def get_active_windows(self) -> List[MaintenanceWindow]:
        """Get all currently active maintenance windows."""
        now = datetime.utcnow()
        return [w for w in self._windows.values() if w.is_active(now)]

    def cleanup_expired(self) -> int:
        """Remove expired maintenance windows."""
        now = datetime.utcnow()
        expired = [
            wid for wid, w in self._windows.items()
            if w.end_time < now
        ]
        for wid in expired:
            del self._windows[wid]
        return len(expired)


# =============================================================================
# Advanced Alert Engine
# =============================================================================

class AdvancedAlertEngine:
    """
    Advanced alerting engine with full SOTA features.

    Combines all advanced alerting capabilities:
    - PromQL-like query evaluation
    - Alert grouping and deduplication
    - Correlation analysis
    - Escalation policies
    - Maintenance windows
    """

    def __init__(
        self,
        metrics_engine: Any,
        channels: Optional[Dict[str, Any]] = None,
    ):
        self.metrics_engine = metrics_engine
        self.channels = channels or {}

        self.query_engine = MetricQueryEngine(metrics_engine)
        self.grouper = AlertGrouper()
        self.deduplicator = AlertDeduplicator()
        self.correlator = AlertCorrelator()
        self.escalation = EscalationManager()
        self.maintenance = MaintenanceManager()

        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._running = False

    async def start(self) -> None:
        """Start the alert engine."""
        self._running = True
        logger.info("Advanced alert engine started")

    async def stop(self) -> None:
        """Stop the alert engine."""
        self._running = False
        logger.info("Advanced alert engine stopped")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.name] = rule

    async def evaluate_rules(self) -> List[Alert]:
        """Evaluate all alert rules."""
        alerts = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            try:
                alert = await self._evaluate_rule(rule)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        return alerts

    async def _evaluate_rule(self, rule: AlertRule) -> Optional[Alert]:
        """Evaluate a single alert rule."""
        # Build query from rule
        query = f"{rule.metric_name}"

        # Execute query
        try:
            results = self.query_engine.query(query)
        except Exception:
            # Fallback to direct metric lookup
            value = self.metrics_engine.get_current(rule.metric_name, {})
            results = [{"labels": {}, "value": value}] if value is not None else []

        if not results:
            return None

        # Check threshold condition
        for result in results:
            value = result["value"]
            labels = result["labels"]

            threshold_met = self._check_threshold(value, rule.condition, rule.threshold)

            if threshold_met:
                alert = Alert(
                    name=rule.name,
                    severity=rule.severity,
                    state=AlertState.FIRING,
                    value=value,
                    threshold=rule.threshold,
                    labels={**labels, **rule.labels},
                    annotations=rule.annotations,
                    message=f"Alert {rule.name}: {value} {rule.condition} {rule.threshold}",
                )

                # Check maintenance window
                is_silenced, window = self.maintenance.is_silenced(alert)
                if is_silenced:
                    logger.debug(f"Alert {rule.name} silenced by maintenance: {window.name}")
                    continue

                # Check deduplication
                if self.deduplicator.is_duplicate(alert):
                    continue

                # Add to correlator
                self.correlator.add_alert(alert)

                # Add to grouper
                self.grouper.add_alert(alert)

                return alert

        return None

    def _check_threshold(self, value: float, condition: str, threshold: float) -> bool:
        """Check if value meets threshold condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "neq":
            return value != threshold
        return False

    async def process_alerts(self) -> None:
        """Process alerts through notification pipeline."""
        groups = self.grouper.get_firing_groups()

        for group in groups:
            if self.grouper.should_notify(group):
                await self._notify_group(group)
                self.grouper.mark_notified(group)

    async def _notify_group(self, group: AlertGroup) -> None:
        """Send notification for an alert group."""
        # Find correlations
        if group.alerts:
            correlations = self.correlator.find_correlations(group.alerts[0])
            root_cause = self.correlator.identify_root_cause(correlations)

        # TODO: Send to channels
        logger.info(f"Notifying for alert group: {group.group_key} ({group.count} alerts)")

    def get_stats(self) -> Dict[str, Any]:
        """Get alert engine statistics."""
        return {
            "rules": len(self._rules),
            "active_alerts": len(self._active_alerts),
            "groups": len(self.grouper.get_groups()),
            "firing_groups": len(self.grouper.get_firing_groups()),
            "escalation": self.escalation.get_stats(),
            "maintenance_windows": len(self.maintenance.get_active_windows()),
        }
