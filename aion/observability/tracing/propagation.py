"""
AION Trace Context Propagation

Implements various trace context propagation formats:
- W3C Trace Context (traceparent, tracestate)
- B3 (Zipkin)
- Jaeger
- Custom propagation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

import structlog

from aion.observability.types import SpanContext

logger = structlog.get_logger(__name__)


class ContextPropagator(ABC):
    """Base class for context propagators."""

    @abstractmethod
    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject trace context into carrier (headers)."""
        pass

    @abstractmethod
    def extract(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """Extract trace context from carrier."""
        pass

    @property
    @abstractmethod
    def fields(self) -> List[str]:
        """List of header fields used by this propagator."""
        pass


class W3CTraceContextPropagator(ContextPropagator):
    """
    W3C Trace Context propagator.

    Implements the W3C Trace Context specification:
    - traceparent: version-trace_id-parent_id-trace_flags
    - tracestate: key1=value1,key2=value2
    """

    TRACEPARENT_HEADER = "traceparent"
    TRACESTATE_HEADER = "tracestate"

    # Regex for traceparent validation
    TRACEPARENT_REGEX = re.compile(
        r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
    )

    @property
    def fields(self) -> List[str]:
        return [self.TRACEPARENT_HEADER, self.TRACESTATE_HEADER]

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject W3C trace context into headers."""
        if not context or not context.is_valid:
            return carrier

        # Format: version-trace_id-span_id-trace_flags
        carrier[self.TRACEPARENT_HEADER] = context.to_traceparent()

        if context.trace_state:
            carrier[self.TRACESTATE_HEADER] = context.trace_state

        return carrier

    def extract(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """Extract W3C trace context from headers."""
        # Case-insensitive header lookup
        traceparent = self._get_header(carrier, self.TRACEPARENT_HEADER)
        if not traceparent:
            return None

        match = self.TRACEPARENT_REGEX.match(traceparent.strip())
        if not match:
            logger.warning(f"Invalid traceparent: {traceparent}")
            return None

        version, trace_id, span_id, trace_flags_str = match.groups()

        # Version 00 is the only supported version
        if version != "00":
            logger.warning(f"Unsupported traceparent version: {version}")
            # Continue anyway for forward compatibility

        # Check for invalid all-zeros
        if trace_id == "0" * 32 or span_id == "0" * 16:
            logger.warning("Invalid all-zeros trace_id or span_id")
            return None

        trace_flags = int(trace_flags_str, 16)
        trace_state = self._get_header(carrier, self.TRACESTATE_HEADER) or ""

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            trace_state=trace_state,
            is_remote=True,
        )

    def _get_header(self, carrier: Dict[str, str], key: str) -> Optional[str]:
        """Case-insensitive header lookup."""
        for k, v in carrier.items():
            if k.lower() == key.lower():
                return v
        return None


class B3Propagator(ContextPropagator):
    """
    B3 propagator (Zipkin format).

    Supports both single-header and multi-header formats:
    - Single: b3: {TraceId}-{SpanId}-{SamplingState}-{ParentSpanId}
    - Multi: X-B3-TraceId, X-B3-SpanId, X-B3-ParentSpanId, X-B3-Sampled, X-B3-Flags
    """

    B3_SINGLE_HEADER = "b3"
    B3_TRACE_ID = "x-b3-traceid"
    B3_SPAN_ID = "x-b3-spanid"
    B3_PARENT_SPAN_ID = "x-b3-parentspanid"
    B3_SAMPLED = "x-b3-sampled"
    B3_FLAGS = "x-b3-flags"

    def __init__(self, single_header: bool = True):
        self.single_header = single_header

    @property
    def fields(self) -> List[str]:
        if self.single_header:
            return [self.B3_SINGLE_HEADER]
        return [
            self.B3_TRACE_ID,
            self.B3_SPAN_ID,
            self.B3_PARENT_SPAN_ID,
            self.B3_SAMPLED,
            self.B3_FLAGS,
        ]

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject B3 trace context."""
        if not context or not context.is_valid:
            return carrier

        sampled = "1" if context.is_sampled else "0"

        if self.single_header:
            # Single header format: {trace_id}-{span_id}-{sampled}
            carrier[self.B3_SINGLE_HEADER] = f"{context.trace_id}-{context.span_id}-{sampled}"
        else:
            # Multi-header format
            carrier[self.B3_TRACE_ID] = context.trace_id
            carrier[self.B3_SPAN_ID] = context.span_id
            carrier[self.B3_SAMPLED] = sampled

        return carrier

    def extract(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """Extract B3 trace context."""
        # Try single header first
        b3_single = self._get_header(carrier, self.B3_SINGLE_HEADER)
        if b3_single:
            return self._extract_single(b3_single)

        # Try multi-header
        trace_id = self._get_header(carrier, self.B3_TRACE_ID)
        span_id = self._get_header(carrier, self.B3_SPAN_ID)

        if not trace_id or not span_id:
            return None

        # Handle 64-bit trace IDs (pad to 128-bit)
        if len(trace_id) == 16:
            trace_id = "0" * 16 + trace_id

        sampled = self._get_header(carrier, self.B3_SAMPLED)
        flags = self._get_header(carrier, self.B3_FLAGS)

        # Determine if sampled
        trace_flags = 0
        if flags == "1" or sampled in ("1", "true"):
            trace_flags = 1
        elif sampled in ("0", "false"):
            trace_flags = 0

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            is_remote=True,
        )

    def _extract_single(self, b3_value: str) -> Optional[SpanContext]:
        """Extract from single B3 header."""
        parts = b3_value.split("-")

        if len(parts) < 2:
            return None

        trace_id = parts[0]
        span_id = parts[1]

        # Handle 64-bit trace IDs
        if len(trace_id) == 16:
            trace_id = "0" * 16 + trace_id

        trace_flags = 0
        if len(parts) >= 3:
            if parts[2] in ("1", "d"):  # d = debug
                trace_flags = 1

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            is_remote=True,
        )

    def _get_header(self, carrier: Dict[str, str], key: str) -> Optional[str]:
        """Case-insensitive header lookup."""
        for k, v in carrier.items():
            if k.lower() == key.lower():
                return v
        return None


class JaegerPropagator(ContextPropagator):
    """
    Jaeger propagator.

    Uses uber-trace-id header:
    {trace-id}:{span-id}:{parent-span-id}:{flags}
    """

    TRACE_HEADER = "uber-trace-id"
    BAGGAGE_PREFIX = "uberctx-"

    @property
    def fields(self) -> List[str]:
        return [self.TRACE_HEADER]

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject Jaeger trace context."""
        if not context or not context.is_valid:
            return carrier

        # Format: {trace-id}:{span-id}:{parent-span-id}:{flags}
        # We don't have parent span ID at this level, so use 0
        carrier[self.TRACE_HEADER] = f"{context.trace_id}:{context.span_id}:0:{context.trace_flags}"

        return carrier

    def extract(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """Extract Jaeger trace context."""
        header = self._get_header(carrier, self.TRACE_HEADER)
        if not header:
            return None

        parts = header.split(":")
        if len(parts) != 4:
            logger.warning(f"Invalid Jaeger header: {header}")
            return None

        trace_id, span_id, _, flags_str = parts

        try:
            trace_flags = int(flags_str, 16)
        except ValueError:
            trace_flags = 0

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            is_remote=True,
        )

    def _get_header(self, carrier: Dict[str, str], key: str) -> Optional[str]:
        """Case-insensitive header lookup."""
        for k, v in carrier.items():
            if k.lower() == key.lower():
                return v
        return None


class CompositePropagator(ContextPropagator):
    """
    Composite propagator that tries multiple propagators.

    Useful for supporting multiple trace context formats.
    """

    def __init__(self, propagators: List[ContextPropagator] = None):
        self.propagators = propagators or [
            W3CTraceContextPropagator(),
            B3Propagator(),
        ]

    @property
    def fields(self) -> List[str]:
        fields = []
        for propagator in self.propagators:
            fields.extend(propagator.fields)
        return list(set(fields))

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject using all propagators."""
        for propagator in self.propagators:
            carrier = propagator.inject(context, carrier)
        return carrier

    def extract(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """Extract using first successful propagator."""
        for propagator in self.propagators:
            context = propagator.extract(carrier)
            if context and context.is_valid:
                return context
        return None


class BaggagePropagator(ContextPropagator):
    """
    W3C Baggage propagator.

    Propagates key-value pairs across service boundaries.
    """

    BAGGAGE_HEADER = "baggage"

    @property
    def fields(self) -> List[str]:
        return [self.BAGGAGE_HEADER]

    def inject(
        self,
        context: SpanContext,
        carrier: Dict[str, str],
        baggage: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """Inject baggage into headers."""
        if not baggage:
            return carrier

        # Format: key1=value1,key2=value2
        items = []
        for key, value in baggage.items():
            # URL-encode special characters
            from urllib.parse import quote
            encoded_key = quote(key, safe="")
            encoded_value = quote(value, safe="")
            items.append(f"{encoded_key}={encoded_value}")

        if items:
            carrier[self.BAGGAGE_HEADER] = ",".join(items)

        return carrier

    def extract(
        self,
        carrier: Dict[str, str],
    ) -> Dict[str, str]:
        """Extract baggage from headers."""
        baggage = {}

        header = self._get_header(carrier, self.BAGGAGE_HEADER)
        if not header:
            return baggage

        from urllib.parse import unquote

        for item in header.split(","):
            item = item.strip()
            if "=" in item:
                key, value = item.split("=", 1)
                baggage[unquote(key)] = unquote(value)

        return baggage

    def _get_header(self, carrier: Dict[str, str], key: str) -> Optional[str]:
        """Case-insensitive header lookup."""
        for k, v in carrier.items():
            if k.lower() == key.lower():
                return v
        return None


# Global propagator instance
_global_propagator: Optional[ContextPropagator] = None


def get_global_propagator() -> ContextPropagator:
    """Get the global context propagator."""
    global _global_propagator
    if _global_propagator is None:
        _global_propagator = CompositePropagator()
    return _global_propagator


def set_global_propagator(propagator: ContextPropagator) -> None:
    """Set the global context propagator."""
    global _global_propagator
    _global_propagator = propagator
