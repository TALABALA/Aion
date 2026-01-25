"""
AION Alert Notification Channels

Multiple notification channels for alerts:
- Webhook
- Slack
- Email
- PagerDuty
- Discord
- Console/Log
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.observability.types import Alert, AlertSeverity, AlertState

logger = structlog.get_logger(__name__)


class AlertChannel(ABC):
    """Base class for alert notification channels."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """
        Send an alert notification.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        pass

    async def test(self) -> bool:
        """Test the channel connectivity."""
        return True


class WebhookChannel(AlertChannel):
    """
    Generic webhook notification channel.

    Sends JSON POST requests to a webhook URL.
    """

    def __init__(
        self,
        url: str,
        headers: Dict[str, str] = None,
        method: str = "POST",
        timeout: float = 30.0,
        include_labels: bool = True,
    ):
        self.url = url
        self.headers = headers or {}
        self.method = method
        self.timeout = timeout
        self.include_labels = include_labels

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import httpx

            payload = self._build_payload(alert)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    self.method,
                    self.url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        **self.headers,
                    },
                )

                if response.status_code >= 400:
                    logger.error(
                        f"Webhook failed: {response.status_code}",
                        url=self.url,
                    )
                    return False

                return True
        except Exception as e:
            logger.error(f"Webhook error: {e}", url=self.url)
            return False

    def _build_payload(self, alert: Alert) -> dict:
        """Build webhook payload."""
        payload = {
            "alert_id": alert.id,
            "fingerprint": alert.fingerprint,
            "rule_name": alert.rule_name,
            "state": alert.state.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "started_at": alert.started_at.isoformat(),
            "fired_at": alert.fired_at.isoformat() if alert.fired_at else None,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
        }

        if self.include_labels:
            payload["labels"] = alert.labels
            payload["annotations"] = alert.annotations

        return payload


class SlackChannel(AlertChannel):
    """
    Slack notification channel.

    Sends formatted messages to Slack via incoming webhook.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = None,
        username: str = "AION Alerts",
        icon_emoji: str = ":warning:",
    ):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

        self._severity_colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000",
        }

        self._severity_emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":fire:",
        }

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            import httpx

            payload = self._build_slack_payload(alert)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                )

                if response.status_code != 200:
                    logger.error(f"Slack send failed: {response.status_code}")
                    return False

                return True
        except Exception as e:
            logger.error(f"Slack error: {e}")
            return False

    def _build_slack_payload(self, alert: Alert) -> dict:
        """Build Slack message payload."""
        color = self._severity_colors.get(alert.severity, "#808080")
        emoji = self._severity_emoji.get(alert.severity, ":bell:")

        state_text = "RESOLVED" if alert.state == AlertState.RESOLVED else "FIRING"

        attachment = {
            "color": color,
            "title": f"{emoji} [{state_text}] {alert.rule_name}",
            "text": alert.message,
            "fields": [
                {
                    "title": "Severity",
                    "value": alert.severity.value.upper(),
                    "short": True,
                },
                {
                    "title": "Value",
                    "value": f"{alert.current_value:.2f}",
                    "short": True,
                },
            ],
            "footer": "AION Monitoring",
            "ts": int(alert.started_at.timestamp()),
        }

        # Add labels as fields
        for key, value in list(alert.labels.items())[:5]:
            attachment["fields"].append({
                "title": key,
                "value": value,
                "short": True,
            })

        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment],
        }

        if self.channel:
            payload["channel"] = self.channel

        return payload


class EmailChannel(AlertChannel):
    """
    Email notification channel.

    Sends emails via SMTP.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "",
        to_addrs: List[str] = None,
        use_tls: bool = True,
        subject_prefix: str = "[AION Alert]",
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.use_tls = use_tls
        self.subject_prefix = subject_prefix

    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            subject = f"{self.subject_prefix} [{alert.severity.value.upper()}] {alert.rule_name}"
            body = self._build_email_body(alert)

            message = MIMEMultipart()
            message["From"] = self.from_addr
            message["To"] = ", ".join(self.to_addrs)
            message["Subject"] = subject
            message.attach(MIMEText(body, "html"))

            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                use_tls=self.use_tls,
            )

            return True
        except ImportError:
            logger.warning("aiosmtplib not installed, email not sent")
            return False
        except Exception as e:
            logger.error(f"Email error: {e}")
            return False

    def _build_email_body(self, alert: Alert) -> str:
        """Build HTML email body."""
        state = "RESOLVED" if alert.state == AlertState.RESOLVED else "FIRING"
        color = "#ff0000" if state == "FIRING" else "#00ff00"

        labels_html = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in alert.labels.items()
        )

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {color};">[{state}] {alert.rule_name}</h2>
            <p><b>Severity:</b> {alert.severity.value.upper()}</p>
            <p><b>Message:</b> {alert.message}</p>
            <p><b>Value:</b> {alert.current_value} (threshold: {alert.threshold})</p>
            <p><b>Started:</b> {alert.started_at.isoformat()}</p>

            <h3>Labels</h3>
            <table border="1" cellpadding="5">
                {labels_html}
            </table>

            <hr>
            <p style="color: #888;">AION Monitoring System</p>
        </body>
        </html>
        """


class PagerDutyChannel(AlertChannel):
    """
    PagerDuty notification channel.

    Uses PagerDuty Events API v2.
    """

    EVENTS_API_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        routing_key: str,
        service_name: str = "AION",
    ):
        self.routing_key = routing_key
        self.service_name = service_name

    async def send(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        try:
            import httpx

            # Determine event action
            if alert.state == AlertState.RESOLVED:
                event_action = "resolve"
            else:
                event_action = "trigger"

            payload = {
                "routing_key": self.routing_key,
                "event_action": event_action,
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": f"[{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}",
                    "source": self.service_name,
                    "severity": self._map_severity(alert.severity),
                    "timestamp": alert.started_at.isoformat(),
                    "custom_details": {
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "labels": alert.labels,
                    },
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.EVENTS_API_URL,
                    json=payload,
                )

                if response.status_code != 202:
                    logger.error(f"PagerDuty failed: {response.status_code}")
                    return False

                return True
        except Exception as e:
            logger.error(f"PagerDuty error: {e}")
            return False

    def _map_severity(self, severity: AlertSeverity) -> str:
        """Map severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }
        return mapping.get(severity, "warning")


class DiscordChannel(AlertChannel):
    """
    Discord notification channel.

    Sends messages to Discord via webhook.
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "AION Alerts",
        avatar_url: str = None,
    ):
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url

        self._severity_colors = {
            AlertSeverity.INFO: 0x3498DB,
            AlertSeverity.WARNING: 0xF39C12,
            AlertSeverity.ERROR: 0xE74C3C,
            AlertSeverity.CRITICAL: 0x992D22,
        }

    async def send(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        try:
            import httpx

            color = self._severity_colors.get(alert.severity, 0x808080)
            state = "RESOLVED" if alert.state == AlertState.RESOLVED else "FIRING"

            embed = {
                "title": f"[{state}] {alert.rule_name}",
                "description": alert.message,
                "color": color,
                "fields": [
                    {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                    {"name": "Value", "value": f"{alert.current_value:.2f}", "inline": True},
                    {"name": "Threshold", "value": f"{alert.threshold}", "inline": True},
                ],
                "timestamp": alert.started_at.isoformat(),
                "footer": {"text": "AION Monitoring"},
            }

            payload = {
                "username": self.username,
                "embeds": [embed],
            }

            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                )

                if response.status_code not in (200, 204):
                    logger.error(f"Discord failed: {response.status_code}")
                    return False

                return True
        except Exception as e:
            logger.error(f"Discord error: {e}")
            return False


class OpsGenieChannel(AlertChannel):
    """
    OpsGenie notification channel.
    """

    API_URL = "https://api.opsgenie.com/v2/alerts"

    def __init__(
        self,
        api_key: str,
        responders: List[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.responders = responders or []

    async def send(self, alert: Alert) -> bool:
        """Send alert to OpsGenie."""
        try:
            import httpx

            if alert.state == AlertState.RESOLVED:
                # Close the alert
                return await self._close_alert(alert)

            payload = {
                "message": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                "description": alert.message,
                "alias": alert.fingerprint,
                "priority": self._map_priority(alert.severity),
                "details": {
                    "current_value": str(alert.current_value),
                    "threshold": str(alert.threshold),
                    **{k: str(v) for k, v in alert.labels.items()},
                },
            }

            if self.responders:
                payload["responders"] = self.responders

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    json=payload,
                    headers={"Authorization": f"GenieKey {self.api_key}"},
                )

                if response.status_code not in (200, 202):
                    logger.error(f"OpsGenie failed: {response.status_code}")
                    return False

                return True
        except Exception as e:
            logger.error(f"OpsGenie error: {e}")
            return False

    async def _close_alert(self, alert: Alert) -> bool:
        """Close an alert in OpsGenie."""
        try:
            import httpx

            url = f"{self.API_URL}/{alert.fingerprint}/close"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    params={"identifierType": "alias"},
                    json={"note": "Resolved by AION"},
                    headers={"Authorization": f"GenieKey {self.api_key}"},
                )

                return response.status_code in (200, 202)
        except Exception as e:
            logger.error(f"OpsGenie close error: {e}")
            return False

    def _map_priority(self, severity: AlertSeverity) -> str:
        """Map severity to OpsGenie priority."""
        mapping = {
            AlertSeverity.INFO: "P5",
            AlertSeverity.WARNING: "P3",
            AlertSeverity.ERROR: "P2",
            AlertSeverity.CRITICAL: "P1",
        }
        return mapping.get(severity, "P3")


class LogChannel(AlertChannel):
    """
    Log notification channel.

    Logs alerts using structlog (for testing and debugging).
    """

    def __init__(self, log_level: str = "warning"):
        self.log_level = log_level

    async def send(self, alert: Alert) -> bool:
        """Log the alert."""
        state = "RESOLVED" if alert.state == AlertState.RESOLVED else "FIRING"

        log_method = getattr(logger, self.log_level, logger.warning)
        log_method(
            f"ALERT [{state}] [{alert.severity.value.upper()}]: {alert.rule_name}",
            message=alert.message,
            current_value=alert.current_value,
            threshold=alert.threshold,
            labels=alert.labels,
        )
        return True


class ConsoleChannel(AlertChannel):
    """
    Console notification channel.

    Prints alerts to stdout (for development).
    """

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

        self._colors = {
            AlertSeverity.INFO: "\033[36m",     # Cyan
            AlertSeverity.WARNING: "\033[33m",  # Yellow
            AlertSeverity.ERROR: "\033[31m",    # Red
            AlertSeverity.CRITICAL: "\033[35m", # Magenta
        }
        self._reset = "\033[0m"

    async def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        state = "RESOLVED" if alert.state == AlertState.RESOLVED else "FIRING"
        color = self._colors.get(alert.severity, "") if self.use_colors else ""
        reset = self._reset if self.use_colors else ""

        print(f"{color}[{state}] [{alert.severity.value.upper()}] {alert.rule_name}{reset}")
        print(f"  Message: {alert.message}")
        print(f"  Value: {alert.current_value} (threshold: {alert.threshold})")
        print(f"  Labels: {alert.labels}")
        print()

        return True


class CompositeChannel(AlertChannel):
    """
    Composite channel that sends to multiple channels.
    """

    def __init__(
        self,
        channels: List[AlertChannel] = None,
        require_all: bool = False,
    ):
        self.channels = channels or []
        self.require_all = require_all

    def add_channel(self, channel: AlertChannel) -> None:
        """Add a channel."""
        self.channels.append(channel)

    async def send(self, alert: Alert) -> bool:
        """Send to all channels."""
        import asyncio

        results = await asyncio.gather(
            *[channel.send(alert) for channel in self.channels],
            return_exceptions=True,
        )

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Channel {i} failed: {result}")

        if self.require_all:
            return all(r is True for r in results)
        else:
            return any(r is True for r in results)
