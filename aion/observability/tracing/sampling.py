"""
AION Trace Sampling Strategies

Implements various sampling strategies:
- Always on/off
- Probability-based
- Rate limiting
- Parent-based
- Adaptive (dynamic)
- Tail-based hints
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from aion.observability.types import (
    SpanContext, SpanKind, SpanLink, SamplingDecision,
)

logger = structlog.get_logger(__name__)


@dataclass
class SamplingResult:
    """Result of a sampling decision."""
    decision: SamplingDecision
    attributes: Dict[str, Any] = field(default_factory=dict)
    trace_state: str = ""


class Sampler(ABC):
    """Base class for trace samplers."""

    @abstractmethod
    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        """
        Determine if a span should be sampled.

        Args:
            trace_id: Trace ID
            parent_context: Parent span context (if any)
            name: Span name
            kind: Span kind
            attributes: Initial span attributes
            links: Span links

        Returns:
            Sampling decision
        """
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the sampler."""
        return self.__class__.__name__


class AlwaysOnSampler(Sampler):
    """Always sample all traces."""

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        return SamplingDecision.RECORD_AND_SAMPLE

    @property
    def description(self) -> str:
        return "AlwaysOnSampler"


class AlwaysOffSampler(Sampler):
    """Never sample any traces."""

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        return SamplingDecision.NOT_RECORD

    @property
    def description(self) -> str:
        return "AlwaysOffSampler"


class TraceIdRatioSampler(Sampler):
    """
    Sample traces based on trace ID ratio.

    Uses a deterministic algorithm based on trace ID to ensure
    all spans in a trace are sampled consistently.
    """

    def __init__(self, ratio: float = 1.0):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Ratio must be between 0.0 and 1.0")
        self.ratio = ratio
        self._bound = int(ratio * (1 << 64))

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        if self.ratio >= 1.0:
            return SamplingDecision.RECORD_AND_SAMPLE
        if self.ratio <= 0.0:
            return SamplingDecision.NOT_RECORD

        # Use last 16 hex chars (64 bits) for deterministic decision
        trace_id_int = int(trace_id[-16:], 16)

        if trace_id_int < self._bound:
            return SamplingDecision.RECORD_AND_SAMPLE
        return SamplingDecision.NOT_RECORD

    @property
    def description(self) -> str:
        return f"TraceIdRatioSampler(ratio={self.ratio})"


class ParentBasedSampler(Sampler):
    """
    Sample based on parent span's sampling decision.

    If parent is sampled, child is sampled.
    If parent is not sampled, child is not sampled.
    For root spans, delegates to another sampler.
    """

    def __init__(
        self,
        root_sampler: Sampler = None,
        remote_parent_sampled: Sampler = None,
        remote_parent_not_sampled: Sampler = None,
        local_parent_sampled: Sampler = None,
        local_parent_not_sampled: Sampler = None,
    ):
        self.root_sampler = root_sampler or AlwaysOnSampler()
        self.remote_parent_sampled = remote_parent_sampled or AlwaysOnSampler()
        self.remote_parent_not_sampled = remote_parent_not_sampled or AlwaysOffSampler()
        self.local_parent_sampled = local_parent_sampled or AlwaysOnSampler()
        self.local_parent_not_sampled = local_parent_not_sampled or AlwaysOffSampler()

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        if parent_context is None:
            # Root span
            return self.root_sampler.should_sample(
                trace_id, None, name, kind, attributes, links
            )

        if parent_context.is_remote:
            if parent_context.is_sampled:
                return self.remote_parent_sampled.should_sample(
                    trace_id, parent_context, name, kind, attributes, links
                )
            else:
                return self.remote_parent_not_sampled.should_sample(
                    trace_id, parent_context, name, kind, attributes, links
                )
        else:
            if parent_context.is_sampled:
                return self.local_parent_sampled.should_sample(
                    trace_id, parent_context, name, kind, attributes, links
                )
            else:
                return self.local_parent_not_sampled.should_sample(
                    trace_id, parent_context, name, kind, attributes, links
                )

    @property
    def description(self) -> str:
        return f"ParentBasedSampler(root={self.root_sampler.description})"


class RateLimitingSampler(Sampler):
    """
    Sample traces up to a maximum rate.

    Limits the number of traces sampled per second.
    """

    def __init__(self, max_traces_per_second: float = 100.0):
        self.max_traces_per_second = max_traces_per_second
        self._tokens = max_traces_per_second
        self._last_update = time.time()
        self._lock = threading.Lock()

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        # Respect parent's decision if present
        if parent_context and parent_context.is_sampled:
            return SamplingDecision.RECORD_AND_SAMPLE
        if parent_context and not parent_context.is_sampled:
            return SamplingDecision.NOT_RECORD

        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Replenish tokens
            self._tokens = min(
                self.max_traces_per_second,
                self._tokens + elapsed * self.max_traces_per_second
            )

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return SamplingDecision.RECORD_AND_SAMPLE

        return SamplingDecision.NOT_RECORD

    @property
    def description(self) -> str:
        return f"RateLimitingSampler(max={self.max_traces_per_second}/s)"


class AdaptiveSampler(Sampler):
    """
    Adaptive sampler that adjusts rate based on traffic.

    Increases sampling rate during low traffic.
    Decreases sampling rate during high traffic.
    """

    def __init__(
        self,
        target_samples_per_second: float = 100.0,
        min_ratio: float = 0.001,
        max_ratio: float = 1.0,
        window_seconds: float = 10.0,
    ):
        self.target_samples_per_second = target_samples_per_second
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.window_seconds = window_seconds

        self._current_ratio = 1.0
        self._request_count = 0
        self._sample_count = 0
        self._window_start = time.time()
        self._lock = threading.Lock()

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        # Respect parent's decision
        if parent_context and parent_context.is_sampled:
            return SamplingDecision.RECORD_AND_SAMPLE
        if parent_context and not parent_context.is_sampled:
            return SamplingDecision.NOT_RECORD

        with self._lock:
            now = time.time()
            elapsed = now - self._window_start

            # Update ratio at end of window
            if elapsed >= self.window_seconds:
                self._update_ratio()
                self._window_start = now
                self._request_count = 0
                self._sample_count = 0

            self._request_count += 1

            # Make sampling decision
            if random.random() < self._current_ratio:
                self._sample_count += 1
                return SamplingDecision.RECORD_AND_SAMPLE

        return SamplingDecision.NOT_RECORD

    def _update_ratio(self) -> None:
        """Update sampling ratio based on recent traffic."""
        if self._request_count == 0:
            return

        # Calculate actual samples per second
        actual_rate = self._sample_count / self.window_seconds

        # Adjust ratio to approach target
        if actual_rate > 0:
            adjustment = self.target_samples_per_second / actual_rate
            self._current_ratio = min(
                self.max_ratio,
                max(self.min_ratio, self._current_ratio * adjustment)
            )

    @property
    def current_ratio(self) -> float:
        """Get current sampling ratio."""
        return self._current_ratio

    @property
    def description(self) -> str:
        return f"AdaptiveSampler(target={self.target_samples_per_second}/s, ratio={self._current_ratio:.4f})"


class PrioritySampler(Sampler):
    """
    Priority-based sampler that always samples important traces.

    High-priority spans (errors, slow operations, etc.) are always sampled.
    Other spans use the fallback sampler.
    """

    def __init__(
        self,
        fallback_sampler: Sampler = None,
        always_sample_errors: bool = True,
        always_sample_slow: bool = True,
        slow_threshold_ms: float = 1000.0,
        priority_operations: List[str] = None,
    ):
        self.fallback_sampler = fallback_sampler or TraceIdRatioSampler(0.1)
        self.always_sample_errors = always_sample_errors
        self.always_sample_slow = always_sample_slow
        self.slow_threshold_ms = slow_threshold_ms
        self.priority_operations = priority_operations or []

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        attributes = attributes or {}

        # Check if this is a priority operation
        if name in self.priority_operations:
            return SamplingDecision.RECORD_AND_SAMPLE

        # Check for error indicators
        if self.always_sample_errors:
            if attributes.get("error") or attributes.get("http.status_code", 200) >= 500:
                return SamplingDecision.RECORD_AND_SAMPLE

        # Fallback to normal sampling
        return self.fallback_sampler.should_sample(
            trace_id, parent_context, name, kind, attributes, links
        )

    @property
    def description(self) -> str:
        return f"PrioritySampler(fallback={self.fallback_sampler.description})"


class CompositeSampler(Sampler):
    """
    Composite sampler with rule-based sampling.

    Applies different samplers based on operation name or attributes.
    """

    @dataclass
    class Rule:
        """Sampling rule."""
        name: str
        sampler: Sampler
        match_name: Optional[str] = None
        match_kind: Optional[SpanKind] = None
        match_attributes: Dict[str, Any] = field(default_factory=dict)

        def matches(
            self,
            name: str,
            kind: SpanKind,
            attributes: Dict[str, Any],
        ) -> bool:
            if self.match_name and name != self.match_name:
                return False
            if self.match_kind and kind != self.match_kind:
                return False
            for key, value in self.match_attributes.items():
                if attributes.get(key) != value:
                    return False
            return True

    def __init__(
        self,
        default_sampler: Sampler = None,
        rules: List["CompositeSampler.Rule"] = None,
    ):
        self.default_sampler = default_sampler or AlwaysOnSampler()
        self.rules = rules or []

    def add_rule(self, rule: "CompositeSampler.Rule") -> None:
        """Add a sampling rule."""
        self.rules.append(rule)

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[SpanContext],
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
    ) -> SamplingDecision:
        attributes = attributes or {}

        # Find matching rule
        for rule in self.rules:
            if rule.matches(name, kind, attributes):
                return rule.sampler.should_sample(
                    trace_id, parent_context, name, kind, attributes, links
                )

        # Use default
        return self.default_sampler.should_sample(
            trace_id, parent_context, name, kind, attributes, links
        )

    @property
    def description(self) -> str:
        return f"CompositeSampler(rules={len(self.rules)}, default={self.default_sampler.description})"


# Factory function for creating samplers
def create_sampler(config: Dict[str, Any]) -> Sampler:
    """
    Create a sampler from configuration.

    Config examples:
        {"type": "always_on"}
        {"type": "always_off"}
        {"type": "ratio", "ratio": 0.1}
        {"type": "rate_limit", "max_per_second": 100}
        {"type": "parent_based", "root": {"type": "ratio", "ratio": 0.1}}
    """
    sampler_type = config.get("type", "always_on")

    if sampler_type == "always_on":
        return AlwaysOnSampler()

    elif sampler_type == "always_off":
        return AlwaysOffSampler()

    elif sampler_type == "ratio":
        ratio = config.get("ratio", 1.0)
        return TraceIdRatioSampler(ratio)

    elif sampler_type == "rate_limit":
        max_per_second = config.get("max_per_second", 100.0)
        return RateLimitingSampler(max_per_second)

    elif sampler_type == "adaptive":
        return AdaptiveSampler(
            target_samples_per_second=config.get("target_per_second", 100.0),
            min_ratio=config.get("min_ratio", 0.001),
            max_ratio=config.get("max_ratio", 1.0),
        )

    elif sampler_type == "parent_based":
        root_config = config.get("root", {"type": "always_on"})
        root_sampler = create_sampler(root_config)
        return ParentBasedSampler(root_sampler=root_sampler)

    elif sampler_type == "priority":
        fallback_config = config.get("fallback", {"type": "ratio", "ratio": 0.1})
        fallback = create_sampler(fallback_config)
        return PrioritySampler(
            fallback_sampler=fallback,
            priority_operations=config.get("priority_operations", []),
        )

    else:
        logger.warning(f"Unknown sampler type: {sampler_type}, using AlwaysOn")
        return AlwaysOnSampler()
