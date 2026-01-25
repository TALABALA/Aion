"""
AION DateTime Serializer

Consistent datetime handling with:
- Timezone support
- Multiple format support
- ISO 8601 compliance
- Relative time parsing
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
import re


class DateTimeSerializer:
    """
    Consistent datetime serialization and parsing.

    Features:
    - ISO 8601 format support
    - Timezone awareness
    - Multiple input format parsing
    - Relative time expressions
    """

    # Supported datetime formats for parsing
    FORMATS = [
        "%Y-%m-%dT%H:%M:%S.%f%z",  # Full ISO with microseconds and timezone
        "%Y-%m-%dT%H:%M:%S%z",      # ISO with timezone
        "%Y-%m-%dT%H:%M:%S.%f",     # ISO with microseconds
        "%Y-%m-%dT%H:%M:%S",        # Basic ISO
        "%Y-%m-%d %H:%M:%S.%f",     # Space separator with microseconds
        "%Y-%m-%d %H:%M:%S",        # Space separator
        "%Y-%m-%d",                 # Date only
        "%d/%m/%Y %H:%M:%S",        # European format
        "%m/%d/%Y %H:%M:%S",        # US format
    ]

    # Relative time patterns
    RELATIVE_PATTERNS = {
        r"now": lambda: datetime.now(timezone.utc),
        r"today": lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
        r"yesterday": lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
        r"tomorrow": lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1),
        r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago": None,  # Handled specially
        r"in\s*(\d+)\s*(second|minute|hour|day|week|month|year)s?": None,   # Handled specially
    }

    # Time unit multipliers (in seconds)
    TIME_UNITS = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
        "week": 604800,
        "month": 2592000,  # Approximate (30 days)
        "year": 31536000,  # Approximate (365 days)
    }

    def __init__(
        self,
        default_timezone: Optional[timezone] = None,
    ):
        self.default_timezone = default_timezone or timezone.utc

    def serialize(
        self,
        dt: datetime,
        include_timezone: bool = True,
        include_microseconds: bool = False,
    ) -> str:
        """
        Serialize a datetime to ISO 8601 string.

        Args:
            dt: Datetime to serialize
            include_timezone: Include timezone info
            include_microseconds: Include microseconds

        Returns:
            ISO 8601 formatted string
        """
        # Ensure timezone awareness
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.default_timezone)

        if include_microseconds:
            if include_timezone:
                return dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        else:
            if include_timezone:
                return dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def deserialize(
        self,
        value: Union[str, datetime, int, float, None],
        assume_utc: bool = True,
    ) -> Optional[datetime]:
        """
        Deserialize a value to datetime.

        Args:
            value: String, datetime, timestamp, or None
            assume_utc: Assume UTC if no timezone specified

        Returns:
            Datetime or None
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None and assume_utc:
                return value.replace(tzinfo=timezone.utc)
            return value

        if isinstance(value, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            value = value.strip()

            # Try relative time first
            relative_dt = self._parse_relative(value)
            if relative_dt:
                return relative_dt

            # Try standard formats
            for fmt in self.FORMATS:
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None and assume_utc:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue

            # Try ISO format parsing (Python 3.7+)
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if dt.tzinfo is None and assume_utc:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass

        return None

    def _parse_relative(self, value: str) -> Optional[datetime]:
        """Parse relative time expressions."""
        value_lower = value.lower().strip()

        # Check simple patterns
        if value_lower == "now":
            return datetime.now(timezone.utc)
        elif value_lower == "today":
            return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        elif value_lower == "yesterday":
            return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        elif value_lower == "tomorrow":
            return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        # Check "X ago" pattern
        ago_match = re.match(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago", value_lower)
        if ago_match:
            amount = int(ago_match.group(1))
            unit = ago_match.group(2)
            seconds = amount * self.TIME_UNITS.get(unit, 0)
            return datetime.now(timezone.utc) - timedelta(seconds=seconds)

        # Check "in X" pattern
        in_match = re.match(r"in\s*(\d+)\s*(second|minute|hour|day|week|month|year)s?", value_lower)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)
            seconds = amount * self.TIME_UNITS.get(unit, 0)
            return datetime.now(timezone.utc) + timedelta(seconds=seconds)

        return None

    def to_timestamp(self, dt: datetime) -> float:
        """Convert datetime to Unix timestamp."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.default_timezone)
        return dt.timestamp()

    def from_timestamp(self, ts: float) -> datetime:
        """Convert Unix timestamp to datetime."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def now(self) -> datetime:
        """Get current datetime with timezone."""
        return datetime.now(self.default_timezone)

    def format_relative(
        self,
        dt: datetime,
        reference: Optional[datetime] = None,
    ) -> str:
        """
        Format datetime as relative time (e.g., "2 hours ago").

        Args:
            dt: Datetime to format
            reference: Reference datetime (default: now)

        Returns:
            Relative time string
        """
        if reference is None:
            reference = self.now()

        # Ensure both are timezone aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.default_timezone)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=self.default_timezone)

        diff = reference - dt
        total_seconds = abs(diff.total_seconds())
        is_past = diff.total_seconds() > 0

        if total_seconds < 60:
            unit, value = "second", int(total_seconds)
        elif total_seconds < 3600:
            unit, value = "minute", int(total_seconds / 60)
        elif total_seconds < 86400:
            unit, value = "hour", int(total_seconds / 3600)
        elif total_seconds < 604800:
            unit, value = "day", int(total_seconds / 86400)
        elif total_seconds < 2592000:
            unit, value = "week", int(total_seconds / 604800)
        elif total_seconds < 31536000:
            unit, value = "month", int(total_seconds / 2592000)
        else:
            unit, value = "year", int(total_seconds / 31536000)

        plural = "s" if value != 1 else ""

        if is_past:
            return f"{value} {unit}{plural} ago"
        else:
            return f"in {value} {unit}{plural}"


# Default serializer instance
_default_serializer = DateTimeSerializer()


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO string."""
    return _default_serializer.serialize(dt)


def deserialize_datetime(value: Union[str, datetime, int, float, None]) -> Optional[datetime]:
    """Deserialize value to datetime."""
    return _default_serializer.deserialize(value)


def format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time."""
    return _default_serializer.format_relative(dt)
