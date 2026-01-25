"""
Cardinality Management Implementation.

Provides tools for managing high-cardinality metrics:
- HyperLogLog for cardinality estimation
- Count-Min Sketch for frequency estimation
- Automatic cardinality limiting
- Adaptive sampling
- Label manipulation (dropping, hashing, aggregation)
"""

import math
import hashlib
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Probabilistic Data Structures
# =============================================================================

class HyperLogLog:
    """
    HyperLogLog for cardinality estimation.

    Provides approximate distinct count with O(1) space complexity.
    Standard error: 1.04 / sqrt(m) where m = 2^precision
    """

    def __init__(self, precision: int = 14):
        """
        Initialize HyperLogLog.

        Args:
            precision: Number of bits for register indexing (4-16)
                      Higher = more accuracy, more memory
                      14 bits = 16KB memory, ~0.81% error
        """
        self.precision = min(16, max(4, precision))
        self.m = 1 << self.precision  # Number of registers
        self.registers = [0] * self.m

        # Alpha constant for bias correction
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, value: Any) -> int:
        """Hash value to 64-bit integer."""
        h = hashlib.sha256(str(value).encode()).digest()
        return int.from_bytes(h[:8], 'big')

    def _get_register_index(self, hash_value: int) -> int:
        """Get register index from hash."""
        return hash_value >> (64 - self.precision)

    def _count_leading_zeros(self, hash_value: int) -> int:
        """Count leading zeros in the remaining bits."""
        # Mask out the index bits
        remaining = hash_value & ((1 << (64 - self.precision)) - 1)
        if remaining == 0:
            return 64 - self.precision

        count = 0
        mask = 1 << (63 - self.precision)
        while (remaining & mask) == 0 and count < 64 - self.precision:
            count += 1
            mask >>= 1

        return count + 1

    def add(self, value: Any):
        """Add a value to the set."""
        h = self._hash(value)
        idx = self._get_register_index(h)
        rank = self._count_leading_zeros(h)
        self.registers[idx] = max(self.registers[idx], rank)

    def count(self) -> int:
        """Estimate the cardinality."""
        # Harmonic mean of register values
        indicator = sum(2 ** (-r) for r in self.registers)
        estimate = self.alpha * self.m * self.m / indicator

        # Small range correction
        if estimate <= 2.5 * self.m:
            # Count zero registers
            zeros = self.registers.count(0)
            if zeros > 0:
                estimate = self.m * math.log(self.m / zeros)

        # Large range correction (not needed for 64-bit hash)

        return int(estimate)

    def merge(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """Merge two HyperLogLog structures."""
        if self.precision != other.precision:
            raise ValueError("Cannot merge HLLs with different precision")

        result = HyperLogLog(self.precision)
        result.registers = [
            max(a, b) for a, b in zip(self.registers, other.registers)
        ]
        return result

    def __len__(self) -> int:
        return self.count()


class CountMinSketch:
    """
    Count-Min Sketch for frequency estimation.

    Provides approximate frequency counts with O(1) space.
    """

    def __init__(self, width: int = 10000, depth: int = 5):
        """
        Initialize Count-Min Sketch.

        Args:
            width: Number of counters per row
            depth: Number of hash functions (rows)
        """
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self._seeds = [random.randint(0, 2**32 - 1) for _ in range(depth)]

    def _hash(self, value: Any, seed: int) -> int:
        """Hash value with seed."""
        h = hashlib.md5(f"{seed}:{value}".encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.width

    def add(self, value: Any, count: int = 1):
        """Add a value with count."""
        for i in range(self.depth):
            idx = self._hash(value, self._seeds[i])
            self.table[i][idx] += count

    def estimate(self, value: Any) -> int:
        """Estimate frequency of a value."""
        estimates = []
        for i in range(self.depth):
            idx = self._hash(value, self._seeds[i])
            estimates.append(self.table[i][idx])
        return min(estimates)  # Minimum to avoid over-counting

    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """Merge two Count-Min Sketches."""
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge sketches with different dimensions")

        result = CountMinSketch(self.width, self.depth)
        result._seeds = self._seeds
        result.table = [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.table, other.table)
        ]
        return result


# =============================================================================
# Cardinality Estimation
# =============================================================================

@dataclass
class CardinalityStats:
    """Statistics about label cardinality."""
    label_name: str
    estimated_cardinality: int
    sample_values: List[str]
    first_seen: datetime
    last_seen: datetime
    total_series: int
    is_high_cardinality: bool


class CardinalityEstimator:
    """
    Estimate and track cardinality of metric labels.

    Uses HyperLogLog for efficient cardinality estimation
    and tracks statistics per label.
    """

    def __init__(
        self,
        high_cardinality_threshold: int = 10000,
        estimation_precision: int = 14
    ):
        self.threshold = high_cardinality_threshold
        self.precision = estimation_precision

        # Per-label HyperLogLog
        self._estimators: Dict[str, HyperLogLog] = {}

        # Sample values for inspection
        self._samples: Dict[str, Set[str]] = defaultdict(set)
        self._sample_limit = 100

        # Tracking
        self._first_seen: Dict[str, datetime] = {}
        self._last_seen: Dict[str, datetime] = {}
        self._series_count: Dict[str, int] = defaultdict(int)

    def observe(self, labels: Dict[str, str]):
        """Observe a set of labels."""
        now = datetime.now()

        for label_name, label_value in labels.items():
            # Initialize if needed
            if label_name not in self._estimators:
                self._estimators[label_name] = HyperLogLog(self.precision)
                self._first_seen[label_name] = now

            # Update estimator
            self._estimators[label_name].add(label_value)
            self._last_seen[label_name] = now
            self._series_count[label_name] += 1

            # Keep samples
            if len(self._samples[label_name]) < self._sample_limit:
                self._samples[label_name].add(label_value)

    def get_cardinality(self, label_name: str) -> int:
        """Get estimated cardinality for a label."""
        if label_name not in self._estimators:
            return 0
        return self._estimators[label_name].count()

    def get_stats(self, label_name: str) -> Optional[CardinalityStats]:
        """Get full statistics for a label."""
        if label_name not in self._estimators:
            return None

        cardinality = self.get_cardinality(label_name)

        return CardinalityStats(
            label_name=label_name,
            estimated_cardinality=cardinality,
            sample_values=list(self._samples[label_name])[:10],
            first_seen=self._first_seen[label_name],
            last_seen=self._last_seen[label_name],
            total_series=self._series_count[label_name],
            is_high_cardinality=cardinality > self.threshold
        )

    def get_high_cardinality_labels(self) -> List[str]:
        """Get list of labels exceeding cardinality threshold."""
        return [
            name for name in self._estimators
            if self._estimators[name].count() > self.threshold
        ]

    def get_all_stats(self) -> Dict[str, CardinalityStats]:
        """Get statistics for all tracked labels."""
        return {
            name: self.get_stats(name)
            for name in self._estimators
        }


# =============================================================================
# Cardinality Limiting
# =============================================================================

class LimitAction(Enum):
    """Action to take when cardinality limit is exceeded."""
    DROP = "drop"  # Drop the metric entirely
    AGGREGATE = "aggregate"  # Aggregate to "other" bucket
    SAMPLE = "sample"  # Sample subset of values
    HASH = "hash"  # Hash values to reduce cardinality
    TRUNCATE = "truncate"  # Keep first N values, drop rest


@dataclass
class LimitConfig:
    """Configuration for cardinality limiting."""
    max_cardinality: int = 10000
    action: LimitAction = LimitAction.AGGREGATE
    hash_buckets: int = 1000  # For HASH action
    sample_rate: float = 0.1  # For SAMPLE action
    aggregate_label: str = "__other__"  # For AGGREGATE action


class CardinalityLimiter:
    """
    Limits cardinality of metric labels.

    Applies configured actions when labels exceed cardinality thresholds.
    """

    def __init__(self, default_config: LimitConfig = None):
        self.default_config = default_config or LimitConfig()
        self._configs: Dict[str, LimitConfig] = {}

        # Track allowed values per label
        self._allowed_values: Dict[str, Set[str]] = defaultdict(set)
        self._value_counts: Dict[str, CountMinSketch] = {}

        # Statistics
        self._dropped_count = 0
        self._aggregated_count = 0
        self._sampled_count = 0

    def configure_label(self, label_name: str, config: LimitConfig):
        """Configure limiting for a specific label."""
        self._configs[label_name] = config

    def _get_config(self, label_name: str) -> LimitConfig:
        """Get config for a label."""
        return self._configs.get(label_name, self.default_config)

    def process(self, labels: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Process labels and apply cardinality limiting.

        Returns:
            Modified labels, or None if metric should be dropped
        """
        result = {}

        for label_name, label_value in labels.items():
            config = self._get_config(label_name)

            # Initialize tracking
            if label_name not in self._value_counts:
                self._value_counts[label_name] = CountMinSketch()

            # Update count
            self._value_counts[label_name].add(label_value)

            # Check cardinality
            if len(self._allowed_values[label_name]) >= config.max_cardinality:
                if label_value not in self._allowed_values[label_name]:
                    # Apply limiting action
                    new_value = self._apply_limit(
                        label_name, label_value, config
                    )

                    if new_value is None:
                        self._dropped_count += 1
                        return None  # Drop entire metric

                    result[label_name] = new_value
                    continue

            # Add to allowed values
            self._allowed_values[label_name].add(label_value)
            result[label_name] = label_value

        return result

    def _apply_limit(
        self,
        label_name: str,
        label_value: str,
        config: LimitConfig
    ) -> Optional[str]:
        """Apply limiting action to a label value."""
        if config.action == LimitAction.DROP:
            return None

        elif config.action == LimitAction.AGGREGATE:
            self._aggregated_count += 1
            return config.aggregate_label

        elif config.action == LimitAction.SAMPLE:
            if random.random() < config.sample_rate:
                self._sampled_count += 1
                self._allowed_values[label_name].add(label_value)
                return label_value
            return None

        elif config.action == LimitAction.HASH:
            # Hash to a fixed number of buckets
            h = hashlib.md5(label_value.encode()).hexdigest()
            bucket = int(h, 16) % config.hash_buckets
            hashed = f"bucket_{bucket}"
            self._allowed_values[label_name].add(hashed)
            return hashed

        elif config.action == LimitAction.TRUNCATE:
            return config.aggregate_label

        return label_value

    def get_stats(self) -> Dict[str, Any]:
        """Get limiting statistics."""
        return {
            "dropped_count": self._dropped_count,
            "aggregated_count": self._aggregated_count,
            "sampled_count": self._sampled_count,
            "label_cardinalities": {
                name: len(values)
                for name, values in self._allowed_values.items()
            }
        }


# =============================================================================
# Label Manipulation
# =============================================================================

class LabelDropper:
    """Drop specified labels from metrics."""

    def __init__(self, labels_to_drop: List[str] = None):
        self.labels_to_drop = set(labels_to_drop or [])
        self._patterns: List[str] = []  # Glob patterns

    def add_label(self, label: str):
        """Add a label to drop."""
        self.labels_to_drop.add(label)

    def add_pattern(self, pattern: str):
        """Add a pattern for labels to drop (glob-style)."""
        self._patterns.append(pattern)

    def _matches_pattern(self, label: str) -> bool:
        """Check if label matches any pattern."""
        import fnmatch
        return any(fnmatch.fnmatch(label, p) for p in self._patterns)

    def process(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Drop configured labels."""
        return {
            k: v for k, v in labels.items()
            if k not in self.labels_to_drop and not self._matches_pattern(k)
        }


class LabelHasher:
    """Hash label values to reduce cardinality."""

    def __init__(
        self,
        labels_to_hash: List[str] = None,
        hash_length: int = 8,
        prefix: str = "h_"
    ):
        self.labels_to_hash = set(labels_to_hash or [])
        self.hash_length = hash_length
        self.prefix = prefix

    def add_label(self, label: str):
        """Add a label to hash."""
        self.labels_to_hash.add(label)

    def process(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Hash configured label values."""
        result = {}

        for k, v in labels.items():
            if k in self.labels_to_hash:
                h = hashlib.md5(v.encode()).hexdigest()[:self.hash_length]
                result[k] = f"{self.prefix}{h}"
            else:
                result[k] = v

        return result


class LabelAggregator:
    """Aggregate label values by grouping similar values."""

    def __init__(self):
        self._rules: Dict[str, List[Tuple[Callable, str]]] = defaultdict(list)

    def add_rule(
        self,
        label: str,
        condition: Callable[[str], bool],
        replacement: str
    ):
        """Add aggregation rule for a label."""
        self._rules[label].append((condition, replacement))

    def add_regex_rule(self, label: str, pattern: str, replacement: str):
        """Add regex-based aggregation rule."""
        import re
        regex = re.compile(pattern)
        self._rules[label].append((
            lambda v, r=regex: r.match(v) is not None,
            replacement
        ))

    def process(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Apply aggregation rules."""
        result = {}

        for k, v in labels.items():
            if k in self._rules:
                for condition, replacement in self._rules[k]:
                    if condition(v):
                        result[k] = replacement
                        break
                else:
                    result[k] = v
            else:
                result[k] = v

        return result


# =============================================================================
# Adaptive Sampling
# =============================================================================

class AdaptiveSampler:
    """
    Adaptive sampling based on metric characteristics.

    Adjusts sampling rate based on:
    - Metric importance
    - Recent activity
    - Cardinality
    - Available resources
    """

    def __init__(
        self,
        base_sample_rate: float = 1.0,
        min_sample_rate: float = 0.01,
        max_sample_rate: float = 1.0,
        adaptation_interval: timedelta = timedelta(minutes=1)
    ):
        self.base_rate = base_sample_rate
        self.min_rate = min_sample_rate
        self.max_rate = max_sample_rate
        self.adaptation_interval = adaptation_interval

        # Per-metric rates
        self._rates: Dict[str, float] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        self._last_adaptation = datetime.now()

        # Priority configuration
        self._priorities: Dict[str, float] = {}  # metric -> priority multiplier

    def set_priority(self, metric_name: str, priority: float):
        """Set priority for a metric (higher = more likely to sample)."""
        self._priorities[metric_name] = max(0.1, min(10.0, priority))

    def should_sample(self, metric_name: str) -> bool:
        """Determine if a metric should be sampled."""
        self._counts[metric_name] += 1

        # Get effective rate
        base = self._rates.get(metric_name, self.base_rate)
        priority = self._priorities.get(metric_name, 1.0)
        rate = min(self.max_rate, base * priority)

        # Sample decision
        sampled = random.random() < rate

        # Periodic adaptation
        if datetime.now() - self._last_adaptation > self.adaptation_interval:
            self._adapt_rates()

        return sampled

    def _adapt_rates(self):
        """Adapt sampling rates based on recent activity."""
        total_count = sum(self._counts.values())
        if total_count == 0:
            return

        # Calculate target samples per metric
        target_total = total_count * self.base_rate

        for metric, count in self._counts.items():
            if count == 0:
                continue

            # Reduce rate for high-frequency metrics
            frequency_ratio = count / total_count
            if frequency_ratio > 0.1:  # More than 10% of traffic
                # Reduce rate for high-frequency
                new_rate = self.base_rate / (frequency_ratio * 10)
            else:
                new_rate = self.base_rate

            self._rates[metric] = max(self.min_rate, min(self.max_rate, new_rate))

        # Reset counts
        self._counts.clear()
        self._last_adaptation = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            "current_rates": dict(self._rates),
            "priorities": dict(self._priorities),
            "counts": dict(self._counts)
        }


# =============================================================================
# Cardinality Manager
# =============================================================================

class CardinalityManager:
    """
    Comprehensive cardinality management.

    Combines all cardinality management features:
    - Estimation
    - Limiting
    - Label manipulation
    - Adaptive sampling
    """

    def __init__(
        self,
        max_cardinality: int = 10000,
        enable_estimation: bool = True,
        enable_limiting: bool = True,
        enable_sampling: bool = True
    ):
        self.max_cardinality = max_cardinality

        # Components
        self.estimator = CardinalityEstimator(
            high_cardinality_threshold=max_cardinality
        ) if enable_estimation else None

        self.limiter = CardinalityLimiter(
            LimitConfig(max_cardinality=max_cardinality)
        ) if enable_limiting else None

        self.sampler = AdaptiveSampler() if enable_sampling else None

        self.dropper = LabelDropper()
        self.hasher = LabelHasher()
        self.aggregator = LabelAggregator()

        # Pipeline configuration
        self._pipeline: List[Callable] = []
        self._build_pipeline()

        # Statistics
        self._processed = 0
        self._dropped = 0
        self._modified = 0

    def _build_pipeline(self):
        """Build processing pipeline."""
        self._pipeline = []

        # Drop first
        self._pipeline.append(self.dropper.process)

        # Then hash
        self._pipeline.append(self.hasher.process)

        # Then aggregate
        self._pipeline.append(self.aggregator.process)

        # Finally limit
        if self.limiter:
            self._pipeline.append(self.limiter.process)

    def configure_drop(self, labels: List[str]):
        """Configure labels to drop."""
        for label in labels:
            self.dropper.add_label(label)

    def configure_hash(self, labels: List[str]):
        """Configure labels to hash."""
        for label in labels:
            self.hasher.add_label(label)

    def configure_limit(self, label: str, config: LimitConfig):
        """Configure limiting for a label."""
        if self.limiter:
            self.limiter.configure_label(label, config)

    def process(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Tuple[Dict[str, str], float]]:
        """
        Process a metric through the cardinality management pipeline.

        Returns:
            (processed_labels, value) or None if dropped
        """
        self._processed += 1

        # Check sampling
        if self.sampler and not self.sampler.should_sample(metric_name):
            self._dropped += 1
            return None

        # Estimate cardinality
        if self.estimator:
            self.estimator.observe(labels)

        # Run pipeline
        current_labels = labels
        for processor in self._pipeline:
            result = processor(current_labels)
            if result is None:
                self._dropped += 1
                return None
            if result != current_labels:
                self._modified += 1
            current_labels = result

        return (current_labels, value)

    def get_high_cardinality_report(self) -> Dict[str, Any]:
        """Generate report on high-cardinality labels."""
        report = {
            "total_processed": self._processed,
            "total_dropped": self._dropped,
            "total_modified": self._modified,
            "high_cardinality_labels": [],
            "label_stats": {}
        }

        if self.estimator:
            stats = self.estimator.get_all_stats()
            report["label_stats"] = {
                name: {
                    "cardinality": s.estimated_cardinality,
                    "is_high_cardinality": s.is_high_cardinality,
                    "sample_values": s.sample_values
                }
                for name, s in stats.items()
            }
            report["high_cardinality_labels"] = self.estimator.get_high_cardinality_labels()

        if self.limiter:
            report["limiter_stats"] = self.limiter.get_stats()

        if self.sampler:
            report["sampler_stats"] = self.sampler.get_stats()

        return report

    def auto_configure(self):
        """
        Automatically configure cardinality management based on observed data.

        Should be called after some warm-up period.
        """
        if not self.estimator:
            return

        high_cardinality = self.estimator.get_high_cardinality_labels()

        for label in high_cardinality:
            stats = self.estimator.get_stats(label)
            if not stats:
                continue

            cardinality = stats.estimated_cardinality

            if cardinality > self.max_cardinality * 10:
                # Very high cardinality - hash
                self.hasher.add_label(label)
                logger.info(f"Auto-configuring hash for {label} (cardinality: {cardinality})")

            elif cardinality > self.max_cardinality:
                # High cardinality - limit with aggregation
                if self.limiter:
                    self.limiter.configure_label(label, LimitConfig(
                        max_cardinality=self.max_cardinality,
                        action=LimitAction.AGGREGATE
                    ))
                logger.info(f"Auto-configuring aggregation for {label} (cardinality: {cardinality})")

        # Rebuild pipeline
        self._build_pipeline()
