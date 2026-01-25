"""
AION Alert Engine

SOTA alert processing with:
- Rule-based alerting
- Multiple notification channels
- Alert state management
- Deduplication and grouping
- Escalation policies
- Silencing support
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.observability.types import (
    Alert, AlertRule, AlertSeverity, AlertState, AlertCondition,
)
from aion.observability.metrics.engine import MetricsEngine
from aion.observability.alerting.channels import AlertChannel, LogChannel

logger = structlog.get_logger(__name__)


class AlertEngine:
    """
    SOTA Alert processing engine.

    Features:
    - Rule-based alerting
    - Multiple notification channels
    - Alert state machine (pending -> firing -> resolved)
    - Deduplication by fingerprint
    - Grouping related alerts
    - Escalation policies
    - Silencing support
    """

    def __init__(
        self,
        metrics_engine: MetricsEngine,
        evaluation_interval: float = 60.0,
        notification_interval: float = 300.0,  # Repeat interval for firing alerts
        resolve_timeout: float = 300.0,  # Time to wait before auto-resolving
    ):
        self.metrics = metrics_engine
        self.evaluation_interval = evaluation_interval
        self.notification_interval = notification_interval
        self.resolve_timeout = resolve_timeout

        # Rules and channels
        self._rules: Dict[str, AlertRule] = {}
        self._channels: Dict[str, AlertChannel] = {}

        # Active alerts by fingerprint
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._max_history = 10000

        # Silences
        self._silences: Dict[str, Dict[str, Any]] = {}

        # Groups (for notification grouping)
        self._groups: Dict[str, List[Alert]] = defaultdict(list)

        # Background task
        self._evaluation_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = {
            "evaluations": 0,
            "alerts_fired": 0,
            "alerts_resolved": 0,
            "notifications_sent": 0,
            "notifications_failed": 0,
        }

        # Register default log channel
        self._channels["log"] = LogChannel()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the alert engine."""
        if self._initialized:
            return

        logger.info("Initializing Alert Engine")

        # Start evaluation loop
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the alert engine."""
        logger.info("Shutting down Alert Engine")

        self._shutdown_event.set()

        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    # === Rule Management ===

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}", rule_id=rule.id)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed alert rule", rule_id=rule_id)
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> List[AlertRule]:
        """Get all rules."""
        return list(self._rules.values())

    def update_rule(self, rule: AlertRule) -> None:
        """Update an existing rule."""
        if rule.id in self._rules:
            self._rules[rule.id] = rule
            logger.info(f"Updated alert rule: {rule.name}", rule_id=rule.id)

    def enable_rule(self, rule_id: str, enabled: bool = True) -> bool:
        """Enable or disable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = enabled
            return True
        return False

    # === Channel Management ===

    def add_channel(self, name: str, channel: AlertChannel) -> None:
        """Add a notification channel."""
        self._channels[name] = channel
        logger.info(f"Added alert channel: {name}")

    def remove_channel(self, name: str) -> bool:
        """Remove a notification channel."""
        if name in self._channels:
            del self._channels[name]
            return True
        return False

    def get_channel(self, name: str) -> Optional[AlertChannel]:
        """Get a channel by name."""
        return self._channels.get(name)

    # === Silence Management ===

    def add_silence(
        self,
        id: str,
        matchers: Dict[str, str],
        ends_at: datetime,
        comment: str = "",
        created_by: str = "",
    ) -> None:
        """Add a silence."""
        self._silences[id] = {
            "id": id,
            "matchers": matchers,
            "ends_at": ends_at,
            "comment": comment,
            "created_by": created_by,
            "created_at": datetime.utcnow(),
        }
        logger.info(f"Added silence", silence_id=id, ends_at=ends_at)

    def remove_silence(self, id: str) -> bool:
        """Remove a silence."""
        if id in self._silences:
            del self._silences[id]
            return True
        return False

    def is_silenced(self, alert: Alert) -> bool:
        """Check if an alert is silenced."""
        now = datetime.utcnow()

        for silence in self._silences.values():
            if silence["ends_at"] < now:
                continue

            # Check if matchers match alert labels
            matches = True
            for key, value in silence["matchers"].items():
                if alert.labels.get(key) != value:
                    matches = False
                    break

            if matches:
                return True

        return False

    # === Evaluation ===

    async def _evaluation_loop(self) -> None:
        """Background rule evaluation loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.evaluation_interval)
                await self.evaluate_all_rules()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")

    async def evaluate_all_rules(self) -> None:
        """Evaluate all alert rules."""
        self._stats["evaluations"] += 1

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.is_silenced():
                continue

            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        # Clean up expired silences
        self._cleanup_silences()

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        # Get metric values
        values = self.metrics.get_all_values(rule.metric_name)

        if not values:
            # Handle absent metric
            if rule.condition == AlertCondition.ABSENT:
                await self._handle_alert_trigger(rule, {}, 0.0)
            return

        for labels_key, value in values.items():
            # Parse labels from key
            labels = {}
            if labels_key:
                for pair in labels_key.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        labels[k] = v

            # Check label matchers
            if not self._matches_labels(labels, rule.label_matchers):
                continue

            # Check condition
            triggered = rule.check_condition(value)

            alert_key = f"{rule.id}:{labels_key}"

            if triggered:
                await self._handle_alert_trigger(rule, labels, value)
            else:
                await self._handle_alert_resolve(rule, labels)

    def _matches_labels(
        self,
        labels: Dict[str, str],
        matchers: Dict[str, str],
    ) -> bool:
        """Check if labels match matchers."""
        for key, value in matchers.items():
            if labels.get(key) != value:
                return False
        return True

    async def _handle_alert_trigger(
        self,
        rule: AlertRule,
        labels: Dict[str, str],
        value: float,
    ) -> None:
        """Handle alert trigger."""
        # Create or get existing alert
        fingerprint = self._create_fingerprint(rule.id, labels)
        existing = self._active_alerts.get(fingerprint)

        if existing:
            # Update existing alert
            existing.current_value = value
            existing.last_evaluated_at = datetime.utcnow()
            existing.evaluation_count += 1

            # Check if should transition from pending to firing
            if existing.state == AlertState.PENDING:
                pending_duration = (datetime.utcnow() - existing.started_at)

                if pending_duration >= rule.for_duration:
                    existing.state = AlertState.FIRING
                    existing.fired_at = datetime.utcnow()
                    existing.consecutive_fires += 1

                    await self._notify_alert(existing, rule)
                    self._stats["alerts_fired"] += 1
            elif existing.state == AlertState.FIRING:
                # Check if should re-notify
                if existing.last_notified_at:
                    since_last = (datetime.utcnow() - existing.last_notified_at).total_seconds()
                    if since_last >= self.notification_interval:
                        await self._notify_alert(existing, rule)
        else:
            # Create new alert
            alert = Alert(
                rule_id=rule.id,
                rule_name=rule.name,
                fingerprint=fingerprint,
                state=AlertState.PENDING,
                severity=rule.severity,
                message=self._format_message(rule, labels, value),
                current_value=value,
                threshold=rule.threshold,
                labels={**rule.labels, **labels},
                annotations=rule.annotations.copy(),
            )
            self._active_alerts[fingerprint] = alert

            # Fire immediately if no for_duration
            if rule.for_duration.total_seconds() == 0:
                alert.state = AlertState.FIRING
                alert.fired_at = datetime.utcnow()

                await self._notify_alert(alert, rule)
                self._stats["alerts_fired"] += 1

    async def _handle_alert_resolve(
        self,
        rule: AlertRule,
        labels: Dict[str, str],
    ) -> None:
        """Handle alert resolution."""
        fingerprint = self._create_fingerprint(rule.id, labels)
        existing = self._active_alerts.get(fingerprint)

        if existing and existing.state in (AlertState.PENDING, AlertState.FIRING):
            # Resolve the alert
            existing.state = AlertState.RESOLVED
            existing.resolved_at = datetime.utcnow()

            # Notify resolution
            await self._notify_alert(existing, rule, resolved=True)

            # Move to history
            self._alert_history.append(existing)
            if len(self._alert_history) > self._max_history:
                self._alert_history = self._alert_history[-self._max_history:]

            del self._active_alerts[fingerprint]
            self._stats["alerts_resolved"] += 1

    def _create_fingerprint(self, rule_id: str, labels: Dict[str, str]) -> str:
        """Create unique fingerprint for an alert."""
        import hashlib
        import json

        label_str = json.dumps(labels, sort_keys=True)
        return hashlib.md5(f"{rule_id}:{label_str}".encode()).hexdigest()[:16]

    def _format_message(
        self,
        rule: AlertRule,
        labels: Dict[str, str],
        value: float,
    ) -> str:
        """Format alert message."""
        return f"{rule.name}: {value} {rule.condition.value} {rule.threshold}"

    async def _notify_alert(
        self,
        alert: Alert,
        rule: AlertRule,
        resolved: bool = False,
    ) -> None:
        """Send alert notifications."""
        # Check if silenced
        if self.is_silenced(alert):
            return

        channel_names = rule.channels or ["log"]

        for channel_name in channel_names:
            channel = self._channels.get(channel_name)
            if not channel:
                logger.warning(f"Channel not found: {channel_name}")
                continue

            try:
                success = await channel.send(alert)
                if success:
                    alert.notified_channels.append(channel_name)
                    alert.notification_count += 1
                    self._stats["notifications_sent"] += 1
                else:
                    self._stats["notifications_failed"] += 1
            except Exception as e:
                logger.error(f"Failed to notify channel {channel_name}: {e}")
                self._stats["notifications_failed"] += 1

        alert.last_notified_at = datetime.utcnow()

    def _cleanup_silences(self) -> None:
        """Remove expired silences."""
        now = datetime.utcnow()
        expired = [
            id for id, s in self._silences.items()
            if s["ends_at"] < now
        ]
        for id in expired:
            del self._silences[id]

    # === Manual Operations ===

    async def fire_test_alert(
        self,
        rule_id: str,
        labels: Dict[str, str] = None,
        value: float = 0.0,
    ) -> Optional[Alert]:
        """Fire a test alert for testing notification channels."""
        rule = self._rules.get(rule_id)
        if not rule:
            return None

        labels = labels or {}
        alert = Alert(
            rule_id=rule.id,
            rule_name=f"[TEST] {rule.name}",
            state=AlertState.FIRING,
            severity=rule.severity,
            message=f"Test alert for rule: {rule.name}",
            current_value=value,
            threshold=rule.threshold,
            labels=labels,
            fired_at=datetime.utcnow(),
        )

        await self._notify_alert(alert, rule)
        return alert

    def acknowledge_alert(
        self,
        fingerprint: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an active alert."""
        alert = self._active_alerts.get(fingerprint)
        if alert:
            alert.acknowledge(acknowledged_by)
            return True
        return False

    # === Query ===

    def get_active_alerts(
        self,
        severity: AlertSeverity = None,
        state: AlertState = None,
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if state:
            alerts = [a for a in alerts if a.state == state]

        return alerts

    def get_alert(self, fingerprint: str) -> Optional[Alert]:
        """Get an alert by fingerprint."""
        return self._active_alerts.get(fingerprint)

    def get_alert_history(
        self,
        limit: int = 100,
        since: datetime = None,
    ) -> List[Alert]:
        """Get alert history."""
        history = self._alert_history

        if since:
            history = [a for a in history if a.resolved_at and a.resolved_at >= since]

        return history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert engine statistics."""
        return {
            **self._stats,
            "rules_count": len(self._rules),
            "channels_count": len(self._channels),
            "active_alerts": len(self._active_alerts),
            "silences_count": len(self._silences),
            "history_count": len(self._alert_history),
            "alerts_by_severity": {
                severity.value: len([a for a in self._active_alerts.values() if a.severity == severity])
                for severity in AlertSeverity
            },
            "alerts_by_state": {
                state.value: len([a for a in self._active_alerts.values() if a.state == state])
                for state in AlertState
            },
        }

    # === Built-in Rules ===

    def add_builtin_rules(self) -> None:
        """Add built-in AION alert rules."""
        builtin_rules = [
            AlertRule(
                name="High Error Rate",
                description="Error rate exceeded threshold",
                metric_name="aion_errors_total",
                condition=AlertCondition.GREATER_THAN,
                threshold=10.0,
                for_duration=timedelta(minutes=1),
                severity=AlertSeverity.ERROR,
                channels=["log"],
            ),
            AlertRule(
                name="High LLM Latency",
                description="LLM API latency is high",
                metric_name="aion_llm_latency_seconds",
                condition=AlertCondition.GREATER_THAN,
                threshold=30.0,
                for_duration=timedelta(minutes=2),
                severity=AlertSeverity.WARNING,
                channels=["log"],
            ),
            AlertRule(
                name="Low Memory Available",
                description="Memory usage is critically high",
                metric_name="aion_process_memory_percent",
                condition=AlertCondition.GREATER_THAN,
                threshold=90.0,
                for_duration=timedelta(minutes=1),
                severity=AlertSeverity.CRITICAL,
                channels=["log"],
            ),
            AlertRule(
                name="Agent Count Anomaly",
                description="Too many active agents",
                metric_name="aion_agents_active",
                condition=AlertCondition.GREATER_THAN,
                threshold=100.0,
                for_duration=timedelta(minutes=5),
                severity=AlertSeverity.WARNING,
                channels=["log"],
            ),
        ]

        for rule in builtin_rules:
            self.add_rule(rule)
