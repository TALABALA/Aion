"""
Cardinality Management Policies.

Defines configurable policies for handling high-cardinality metrics.
"""

import re
import hashlib
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Pattern
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Base Policy
# =============================================================================

class CardinalityPolicy(ABC):
    """Base class for cardinality management policies."""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.enabled = True

    @abstractmethod
    def applies_to(self, metric_name: str, labels: Dict[str, str]) -> bool:
        """Check if this policy applies to the metric."""
        pass

    @abstractmethod
    def apply(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[str, Dict[str, str], float]]:
        """
        Apply the policy to a metric.

        Returns:
            (metric_name, labels, value) or None if metric should be dropped
        """
        pass


# =============================================================================
# Drop Policies
# =============================================================================

class DropHighCardinalityLabels(CardinalityPolicy):
    """Drop labels that exceed cardinality threshold."""

    def __init__(
        self,
        label_patterns: List[str] = None,
        cardinality_threshold: int = 10000,
        estimation_window: int = 1000
    ):
        super().__init__("drop_high_cardinality", priority=10)
        self.label_patterns = [
            re.compile(p) for p in (label_patterns or [])
        ]
        self.threshold = cardinality_threshold
        self.estimation_window = estimation_window

        # Track cardinalities
        self._label_values: Dict[str, set] = {}
        self._counts: Dict[str, int] = {}

    def applies_to(self, metric_name: str, labels: Dict[str, str]) -> bool:
        # Applies to all metrics
        return True

    def apply(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[str, Dict[str, str], float]]:
        filtered_labels = {}

        for label_name, label_value in labels.items():
            # Check pattern match
            should_check = not self.label_patterns or any(
                p.match(label_name) for p in self.label_patterns
            )

            if should_check:
                # Track cardinality
                key = f"{metric_name}:{label_name}"
                if key not in self._label_values:
                    self._label_values[key] = set()
                    self._counts[key] = 0

                self._label_values[key].add(label_value)
                self._counts[key] += 1

                # Check threshold
                if len(self._label_values[key]) > self.threshold:
                    logger.debug(f"Dropping high-cardinality label {label_name}")
                    continue  # Drop this label

                # Periodic cleanup
                if self._counts[key] > self.estimation_window:
                    self._label_values[key] = set()
                    self._counts[key] = 0

            filtered_labels[label_name] = label_value

        return (metric_name, filtered_labels, value)


# =============================================================================
# Hash Policies
# =============================================================================

class HashHighCardinalityLabels(CardinalityPolicy):
    """Hash label values to reduce cardinality."""

    def __init__(
        self,
        labels_to_hash: List[str],
        hash_buckets: int = 1000,
        hash_prefix: str = "bucket_"
    ):
        super().__init__("hash_high_cardinality", priority=20)
        self.labels_to_hash = set(labels_to_hash)
        self.hash_buckets = hash_buckets
        self.hash_prefix = hash_prefix

    def applies_to(self, metric_name: str, labels: Dict[str, str]) -> bool:
        return any(label in self.labels_to_hash for label in labels)

    def apply(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[str, Dict[str, str], float]]:
        new_labels = {}

        for label_name, label_value in labels.items():
            if label_name in self.labels_to_hash:
                # Hash to bucket
                h = hashlib.md5(label_value.encode()).hexdigest()
                bucket = int(h, 16) % self.hash_buckets
                new_labels[label_name] = f"{self.hash_prefix}{bucket}"
            else:
                new_labels[label_name] = label_value

        return (metric_name, new_labels, value)


# =============================================================================
# Sampling Policies
# =============================================================================

class SampleHighCardinalitySeries(CardinalityPolicy):
    """Sample metrics with high-cardinality labels."""

    def __init__(
        self,
        sample_rate: float = 0.1,
        label_patterns: List[str] = None,
        cardinality_threshold: int = 1000
    ):
        super().__init__("sample_high_cardinality", priority=5)
        self.sample_rate = sample_rate
        self.label_patterns = [
            re.compile(p) for p in (label_patterns or [".*"])
        ]
        self.threshold = cardinality_threshold

        self._label_values: Dict[str, set] = {}

    def applies_to(self, metric_name: str, labels: Dict[str, str]) -> bool:
        # Check if any label has high cardinality
        for label_name, label_value in labels.items():
            if any(p.match(label_name) for p in self.label_patterns):
                key = f"{metric_name}:{label_name}"
                if key not in self._label_values:
                    self._label_values[key] = set()

                self._label_values[key].add(label_value)

                if len(self._label_values[key]) > self.threshold:
                    return True

        return False

    def apply(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[str, Dict[str, str], float]]:
        if random.random() < self.sample_rate:
            return (metric_name, labels, value)
        return None


# =============================================================================
# Aggregation Policies
# =============================================================================

class AggregateHighCardinalitySeries(CardinalityPolicy):
    """Aggregate high-cardinality label values into buckets."""

    def __init__(
        self,
        aggregation_rules: Dict[str, List[Tuple[str, str]]] = None,
        default_bucket: str = "__other__"
    ):
        """
        Initialize aggregation policy.

        Args:
            aggregation_rules: Dict of label_name -> [(regex_pattern, replacement)]
            default_bucket: Value for non-matching values
        """
        super().__init__("aggregate_high_cardinality", priority=15)
        self.default_bucket = default_bucket

        # Compile rules
        self.rules: Dict[str, List[Tuple[Pattern, str]]] = {}
        for label, patterns in (aggregation_rules or {}).items():
            self.rules[label] = [
                (re.compile(p), r) for p, r in patterns
            ]

    def add_rule(self, label: str, pattern: str, replacement: str):
        """Add aggregation rule."""
        if label not in self.rules:
            self.rules[label] = []
        self.rules[label].append((re.compile(pattern), replacement))

    def applies_to(self, metric_name: str, labels: Dict[str, str]) -> bool:
        return any(label in self.rules for label in labels)

    def apply(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[str, Dict[str, str], float]]:
        new_labels = {}

        for label_name, label_value in labels.items():
            if label_name in self.rules:
                # Try to match rules
                matched = False
                for pattern, replacement in self.rules[label_name]:
                    if pattern.match(label_value):
                        new_labels[label_name] = replacement
                        matched = True
                        break

                if not matched:
                    new_labels[label_name] = self.default_bucket
            else:
                new_labels[label_name] = label_value

        return (metric_name, new_labels, value)


# =============================================================================
# Relabeling Rules (Prometheus-style)
# =============================================================================

@dataclass
class RelabelingRule:
    """
    Prometheus-style relabeling rule.

    Supports actions:
    - replace: Replace label value
    - keep: Keep metrics matching regex
    - drop: Drop metrics matching regex
    - labelkeep: Keep only matching labels
    - labeldrop: Drop matching labels
    - hashmod: Set label to hash modulo
    - labelmap: Map label names
    """

    source_labels: List[str] = field(default_factory=list)
    separator: str = ";"
    regex: str = "(.*)"
    target_label: str = ""
    replacement: str = "$1"
    action: str = "replace"
    modulus: int = 0  # For hashmod

    def __post_init__(self):
        self._regex = re.compile(self.regex)

    def apply(
        self,
        labels: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """
        Apply relabeling rule.

        Returns:
            Modified labels or None if metric should be dropped
        """
        # Get source value
        source_values = [
            labels.get(label, "") for label in self.source_labels
        ]
        source = self.separator.join(source_values)

        if self.action == "replace":
            match = self._regex.match(source)
            if match:
                replacement = self.replacement
                for i, group in enumerate(match.groups()):
                    replacement = replacement.replace(f"${i+1}", group or "")

                new_labels = dict(labels)
                if self.target_label:
                    new_labels[self.target_label] = replacement
                return new_labels
            return labels

        elif self.action == "keep":
            if self._regex.match(source):
                return labels
            return None

        elif self.action == "drop":
            if self._regex.match(source):
                return None
            return labels

        elif self.action == "labelkeep":
            return {
                k: v for k, v in labels.items()
                if self._regex.match(k)
            }

        elif self.action == "labeldrop":
            return {
                k: v for k, v in labels.items()
                if not self._regex.match(k)
            }

        elif self.action == "hashmod":
            if self.modulus > 0:
                h = hashlib.md5(source.encode()).hexdigest()
                mod_value = int(h, 16) % self.modulus
                new_labels = dict(labels)
                new_labels[self.target_label] = str(mod_value)
                return new_labels
            return labels

        elif self.action == "labelmap":
            new_labels = {}
            for k, v in labels.items():
                match = self._regex.match(k)
                if match:
                    new_key = self.replacement
                    for i, group in enumerate(match.groups()):
                        new_key = new_key.replace(f"${i+1}", group or "")
                    new_labels[new_key] = v
                else:
                    new_labels[k] = v
            return new_labels

        return labels


class RelabelingPipeline:
    """Pipeline of relabeling rules."""

    def __init__(self, rules: List[RelabelingRule] = None):
        self.rules = rules or []

    def add_rule(self, rule: RelabelingRule):
        """Add a rule to the pipeline."""
        self.rules.append(rule)

    def apply(self, labels: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Apply all rules in order."""
        current = labels

        for rule in self.rules:
            result = rule.apply(current)
            if result is None:
                return None
            current = result

        return current


# =============================================================================
# Policy Engine
# =============================================================================

class PolicyEngine:
    """
    Engine for applying cardinality management policies.

    Manages multiple policies with priority ordering.
    """

    def __init__(self):
        self._policies: List[CardinalityPolicy] = []
        self._relabeling = RelabelingPipeline()

    def add_policy(self, policy: CardinalityPolicy):
        """Add a policy."""
        self._policies.append(policy)
        # Sort by priority (lower first)
        self._policies.sort(key=lambda p: p.priority)

    def add_relabeling_rule(self, rule: RelabelingRule):
        """Add a relabeling rule."""
        self._relabeling.add_rule(rule)

    def process(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[str, Dict[str, str], float]]:
        """
        Process a metric through all applicable policies.

        Returns:
            (metric_name, labels, value) or None if dropped
        """
        # First apply relabeling
        labels = self._relabeling.apply(labels)
        if labels is None:
            return None

        current = (metric_name, labels, value)

        # Then apply policies
        for policy in self._policies:
            if not policy.enabled:
                continue

            if policy.applies_to(current[0], current[1]):
                result = policy.apply(current[0], current[1], current[2])
                if result is None:
                    return None
                current = result

        return current

    def configure_from_dict(self, config: Dict[str, Any]):
        """Configure engine from dictionary."""
        # Configure relabeling rules
        for rule_config in config.get("relabeling_rules", []):
            rule = RelabelingRule(**rule_config)
            self.add_relabeling_rule(rule)

        # Configure policies
        for policy_config in config.get("policies", []):
            policy_type = policy_config.pop("type")

            if policy_type == "drop_high_cardinality":
                self.add_policy(DropHighCardinalityLabels(**policy_config))
            elif policy_type == "hash_high_cardinality":
                self.add_policy(HashHighCardinalityLabels(**policy_config))
            elif policy_type == "sample_high_cardinality":
                self.add_policy(SampleHighCardinalitySeries(**policy_config))
            elif policy_type == "aggregate_high_cardinality":
                self.add_policy(AggregateHighCardinalitySeries(**policy_config))
