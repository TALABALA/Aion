"""
Threat Detection for Security Observability.

Provides threat detection capabilities:
- Pattern-based threat detection
- Anomaly-based threat detection
- Threat intelligence integration
- IOC (Indicators of Compromise) matching
"""

import re
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Categories of security threats."""
    MALWARE = "malware"
    PHISHING = "phishing"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_CONTROL = "command_and_control"
    DENIAL_OF_SERVICE = "denial_of_service"
    INJECTION = "injection"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class ThreatIndicator:
    """Indicator of Compromise (IOC)."""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, url, email, etc.
    value: str
    category: ThreatCategory
    severity: ThreatSeverity
    confidence: float  # 0-1
    description: str = ""
    source: str = ""
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatPattern:
    """Detection pattern for threats."""
    pattern_id: str
    name: str
    description: str
    category: ThreatCategory
    severity: ThreatSeverity
    # Detection logic
    field_patterns: Dict[str, str] = field(default_factory=dict)  # field -> regex
    threshold_conditions: Dict[str, tuple] = field(default_factory=dict)  # field -> (op, value)
    time_window: Optional[timedelta] = None
    occurrence_threshold: int = 1
    # MITRE ATT&CK mapping
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    # Response
    recommended_actions: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ThreatAlert:
    """Alert generated from threat detection."""
    alert_id: str
    timestamp: datetime
    pattern: ThreatPattern
    matched_events: List[Dict[str, Any]]
    confidence: float
    indicators: List[ThreatIndicator] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class ThreatIntelFeed:
    """Threat intelligence feed for IOC updates."""

    def __init__(self, name: str, url: str = None, api_key: str = None):
        self.name = name
        self.url = url
        self.api_key = api_key
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.last_updated: Optional[datetime] = None

    async def refresh(self):
        """Refresh indicators from feed."""
        # In production, fetch from actual threat intel API
        logger.info(f"Refreshing threat intel feed: {self.name}")
        self.last_updated = datetime.now()

    def add_indicator(self, indicator: ThreatIndicator):
        """Add an indicator to the feed."""
        self.indicators[f"{indicator.indicator_type}:{indicator.value}"] = indicator

    def lookup(self, indicator_type: str, value: str) -> Optional[ThreatIndicator]:
        """Look up an indicator."""
        return self.indicators.get(f"{indicator_type}:{value}")


class ThreatDetector:
    """
    Main threat detection engine.

    Combines pattern matching, IOC lookup, and anomaly detection.
    """

    def __init__(self):
        self._patterns: Dict[str, ThreatPattern] = {}
        self._feeds: Dict[str, ThreatIntelFeed] = {}
        self._event_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._alerts: List[ThreatAlert] = []
        self._alert_handlers: List[Callable[[ThreatAlert], None]] = []

        # Built-in patterns
        self._register_builtin_patterns()

    def _register_builtin_patterns(self):
        """Register built-in threat detection patterns."""
        patterns = [
            ThreatPattern(
                pattern_id="brute_force_login",
                name="Brute Force Login Attempt",
                description="Multiple failed login attempts from same source",
                category=ThreatCategory.BRUTE_FORCE,
                severity=ThreatSeverity.HIGH,
                field_patterns={"event_type": r"login_failed"},
                time_window=timedelta(minutes=5),
                occurrence_threshold=10,
                mitre_tactics=["TA0006"],
                mitre_techniques=["T1110"],
                recommended_actions=["Block source IP", "Lock account", "Alert security team"]
            ),
            ThreatPattern(
                pattern_id="sql_injection",
                name="SQL Injection Attempt",
                description="Potential SQL injection in request",
                category=ThreatCategory.INJECTION,
                severity=ThreatSeverity.CRITICAL,
                field_patterns={
                    "request_body": r"(?i)(union\s+select|;\s*drop\s+|--\s*$|'\s*or\s+'1'\s*=\s*'1)",
                    "url": r"(?i)(union\s+select|;\s*drop\s+|--\s*$|'\s*or\s+'1'\s*=\s*'1)"
                },
                mitre_tactics=["TA0001"],
                mitre_techniques=["T1190"],
                recommended_actions=["Block request", "Review WAF rules", "Audit database"]
            ),
            ThreatPattern(
                pattern_id="data_exfiltration",
                name="Potential Data Exfiltration",
                description="Large outbound data transfer detected",
                category=ThreatCategory.DATA_EXFILTRATION,
                severity=ThreatSeverity.HIGH,
                threshold_conditions={"bytes_out": (">", 100_000_000)},
                time_window=timedelta(hours=1),
                mitre_tactics=["TA0010"],
                mitre_techniques=["T1048"],
                recommended_actions=["Block transfer", "Investigate source", "Review DLP policies"]
            ),
            ThreatPattern(
                pattern_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                description="User attempting to access admin functions",
                category=ThreatCategory.PRIVILEGE_ESCALATION,
                severity=ThreatSeverity.CRITICAL,
                field_patterns={
                    "url": r"(?i)(admin|root|sudo|privileged)",
                    "response_code": r"^(401|403)$"
                },
                occurrence_threshold=5,
                time_window=timedelta(minutes=10),
                mitre_tactics=["TA0004"],
                mitre_techniques=["T1068"],
            ),
        ]

        for pattern in patterns:
            self.register_pattern(pattern)

    def register_pattern(self, pattern: ThreatPattern):
        """Register a detection pattern."""
        self._patterns[pattern.pattern_id] = pattern

    def add_feed(self, feed: ThreatIntelFeed):
        """Add a threat intelligence feed."""
        self._feeds[feed.name] = feed

    def add_alert_handler(self, handler: Callable[[ThreatAlert], None]):
        """Add handler for threat alerts."""
        self._alert_handlers.append(handler)

    async def analyze(self, event: Dict[str, Any]) -> List[ThreatAlert]:
        """Analyze an event for threats."""
        alerts = []

        # Pattern matching
        for pattern in self._patterns.values():
            if not pattern.enabled:
                continue

            if self._matches_pattern(event, pattern):
                # Track for threshold-based patterns
                key = f"{pattern.pattern_id}:{event.get('source_ip', 'unknown')}"
                self._event_buffer[key].append({
                    "timestamp": datetime.now(),
                    "event": event
                })

                # Check threshold
                if self._check_threshold(key, pattern):
                    alert = self._create_alert(pattern, self._event_buffer[key])
                    alerts.append(alert)
                    self._event_buffer[key] = []

        # IOC lookup
        ioc_alerts = await self._check_iocs(event)
        alerts.extend(ioc_alerts)

        # Notify handlers
        for alert in alerts:
            self._alerts.append(alert)
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

        return alerts

    def _matches_pattern(self, event: Dict[str, Any], pattern: ThreatPattern) -> bool:
        """Check if event matches pattern."""
        # Field pattern matching
        for field_name, regex in pattern.field_patterns.items():
            value = str(event.get(field_name, ""))
            if not re.search(regex, value):
                return False

        # Threshold conditions
        for field_name, (op, threshold) in pattern.threshold_conditions.items():
            value = event.get(field_name, 0)
            try:
                value = float(value)
                if op == ">" and not value > threshold:
                    return False
                elif op == "<" and not value < threshold:
                    return False
                elif op == ">=" and not value >= threshold:
                    return False
                elif op == "<=" and not value <= threshold:
                    return False
                elif op == "=" and not value == threshold:
                    return False
            except (TypeError, ValueError):
                return False

        return True

    def _check_threshold(self, key: str, pattern: ThreatPattern) -> bool:
        """Check if occurrence threshold is met."""
        if not pattern.time_window:
            return len(self._event_buffer[key]) >= pattern.occurrence_threshold

        cutoff = datetime.now() - pattern.time_window
        recent = [e for e in self._event_buffer[key] if e["timestamp"] > cutoff]
        self._event_buffer[key] = recent

        return len(recent) >= pattern.occurrence_threshold

    async def _check_iocs(self, event: Dict[str, Any]) -> List[ThreatAlert]:
        """Check event against IOC feeds."""
        alerts = []

        # Extract potential IOCs from event
        potential_iocs = [
            ("ip", event.get("source_ip")),
            ("ip", event.get("destination_ip")),
            ("domain", event.get("host")),
            ("url", event.get("url")),
            ("hash", event.get("file_hash")),
        ]

        for ioc_type, value in potential_iocs:
            if not value:
                continue

            for feed in self._feeds.values():
                indicator = feed.lookup(ioc_type, value)
                if indicator:
                    pattern = ThreatPattern(
                        pattern_id=f"ioc_match_{indicator.indicator_id}",
                        name=f"IOC Match: {indicator.indicator_type}",
                        description=indicator.description,
                        category=indicator.category,
                        severity=indicator.severity,
                    )
                    alert = self._create_alert(pattern, [{"event": event}], [indicator])
                    alerts.append(alert)

        return alerts

    def _create_alert(self, pattern: ThreatPattern, events: List[Dict],
                      indicators: List[ThreatIndicator] = None) -> ThreatAlert:
        """Create a threat alert."""
        import uuid
        return ThreatAlert(
            alert_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            pattern=pattern,
            matched_events=[e.get("event", e) for e in events],
            confidence=0.8,
            indicators=indicators or [],
            context={"event_count": len(events)}
        )

    def get_alerts(self, since: datetime = None, category: ThreatCategory = None,
                   min_severity: ThreatSeverity = None) -> List[ThreatAlert]:
        """Get alerts with optional filters."""
        results = self._alerts

        if since:
            results = [a for a in results if a.timestamp >= since]
        if category:
            results = [a for a in results if a.pattern.category == category]
        if min_severity:
            results = [a for a in results if a.pattern.severity.value >= min_severity.value]

        return results
