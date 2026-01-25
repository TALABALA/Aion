"""
W3C Baggage Propagation for OpenTelemetry.

Implements baggage propagation according to W3C specification.
"""

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from contextvars import ContextVar

# Context variable for current baggage
_current_baggage: ContextVar[Optional['Baggage']] = ContextVar('current_baggage', default=None)

# Constants
BAGGAGE_HEADER = "baggage"
MAX_PAIRS = 180
MAX_PAIR_LENGTH = 4096
MAX_TOTAL_LENGTH = 8192


@dataclass
class BaggageEntry:
    """A single baggage entry with value and metadata."""
    value: str
    metadata: Dict[str, str] = field(default_factory=dict)


class Baggage:
    """
    W3C Baggage for propagating context across service boundaries.

    Baggage allows passing key-value pairs alongside trace context.
    """

    def __init__(self, entries: Dict[str, BaggageEntry] = None):
        self._entries: Dict[str, BaggageEntry] = entries or {}

    def get(self, key: str) -> Optional[str]:
        """Get a baggage value by key."""
        entry = self._entries.get(key)
        return entry.value if entry else None

    def get_entry(self, key: str) -> Optional[BaggageEntry]:
        """Get full baggage entry by key."""
        return self._entries.get(key)

    def set(self, key: str, value: str, metadata: Dict[str, str] = None) -> 'Baggage':
        """Set a baggage value, returning new Baggage (immutable)."""
        new_entries = dict(self._entries)
        new_entries[key] = BaggageEntry(value=value, metadata=metadata or {})
        return Baggage(new_entries)

    def remove(self, key: str) -> 'Baggage':
        """Remove a baggage entry, returning new Baggage."""
        new_entries = {k: v for k, v in self._entries.items() if k != key}
        return Baggage(new_entries)

    def get_all(self) -> Dict[str, str]:
        """Get all baggage values as dict."""
        return {k: v.value for k, v in self._entries.items()}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(self._entries)

    @staticmethod
    def get_current() -> 'Baggage':
        """Get current baggage from context."""
        return _current_baggage.get() or Baggage()

    def make_current(self) -> 'Baggage':
        """Set this baggage as current."""
        _current_baggage.set(self)
        return self


class BaggagePropagator:
    """
    Propagator for W3C Baggage format.

    Serializes/deserializes baggage to/from HTTP headers.
    """

    # Regex patterns
    KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_\-*/]+$')
    TOKEN_PATTERN = re.compile(r'^[\x20-\x7E]+$')

    def inject(self, baggage: Baggage, carrier: Dict[str, str]):
        """Inject baggage into carrier (e.g., HTTP headers)."""
        if not baggage or len(baggage) == 0:
            return

        pairs = []
        total_length = 0

        for key, entry in baggage._entries.items():
            # Validate and encode
            if not self._is_valid_key(key):
                continue

            encoded_value = urllib.parse.quote(entry.value, safe='')
            pair = f"{key}={encoded_value}"

            # Add metadata
            for meta_key, meta_value in entry.metadata.items():
                pair += f";{meta_key}={urllib.parse.quote(meta_value, safe='')}"

            # Check limits
            if len(pair) > MAX_PAIR_LENGTH:
                continue
            if total_length + len(pair) + 1 > MAX_TOTAL_LENGTH:
                break
            if len(pairs) >= MAX_PAIRS:
                break

            pairs.append(pair)
            total_length += len(pair) + 1

        if pairs:
            carrier[BAGGAGE_HEADER] = ",".join(pairs)

    def extract(self, carrier: Dict[str, str]) -> Baggage:
        """Extract baggage from carrier."""
        header_value = carrier.get(BAGGAGE_HEADER) or carrier.get(BAGGAGE_HEADER.lower())

        if not header_value:
            return Baggage()

        entries = {}

        for pair in header_value.split(","):
            pair = pair.strip()
            if not pair:
                continue

            # Split key=value and metadata
            parts = pair.split(";")
            if not parts:
                continue

            # Parse key=value
            kv = parts[0].split("=", 1)
            if len(kv) != 2:
                continue

            key = kv[0].strip()
            value = urllib.parse.unquote(kv[1].strip())

            if not self._is_valid_key(key):
                continue

            # Parse metadata
            metadata = {}
            for meta_part in parts[1:]:
                meta_kv = meta_part.split("=", 1)
                if len(meta_kv) == 2:
                    meta_key = meta_kv[0].strip()
                    meta_value = urllib.parse.unquote(meta_kv[1].strip())
                    metadata[meta_key] = meta_value

            entries[key] = BaggageEntry(value=value, metadata=metadata)

        return Baggage(entries)

    def _is_valid_key(self, key: str) -> bool:
        """Validate baggage key."""
        return bool(key and len(key) <= 256 and self.KEY_PATTERN.match(key))


class BaggageManager:
    """Manager for working with baggage in applications."""

    def __init__(self):
        self.propagator = BaggagePropagator()

    def set(self, key: str, value: str, metadata: Dict[str, str] = None) -> Baggage:
        """Set a baggage value in current context."""
        current = Baggage.get_current()
        new_baggage = current.set(key, value, metadata)
        return new_baggage.make_current()

    def get(self, key: str) -> Optional[str]:
        """Get a baggage value from current context."""
        return Baggage.get_current().get(key)

    def remove(self, key: str) -> Baggage:
        """Remove a baggage value from current context."""
        current = Baggage.get_current()
        new_baggage = current.remove(key)
        return new_baggage.make_current()

    def clear(self) -> Baggage:
        """Clear all baggage."""
        return Baggage().make_current()

    def get_all(self) -> Dict[str, str]:
        """Get all baggage values."""
        return Baggage.get_current().get_all()

    def inject_headers(self, headers: Dict[str, str]):
        """Inject current baggage into headers."""
        self.propagator.inject(Baggage.get_current(), headers)

    def extract_headers(self, headers: Dict[str, str]) -> Baggage:
        """Extract baggage from headers and set as current."""
        baggage = self.propagator.extract(headers)
        return baggage.make_current()
