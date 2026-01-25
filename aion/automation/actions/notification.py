"""
AION Notification Action Handler

Send notifications from workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import structlog

from aion.automation.types import ActionConfig
from aion.automation.actions.executor import BaseActionHandler

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class NotificationActionHandler(BaseActionHandler):
    """
    Handler for notification actions.

    Supports multiple channels:
    - Console/Log
    - Email
    - Slack
    - Webhook
    - Custom
    """

    def __init__(self):
        self._channels: Dict[str, "NotificationChannel"] = {
            "console": ConsoleChannel(),
            "log": LogChannel(),
            "email": EmailChannel(),
            "slack": SlackChannel(),
            "webhook": WebhookChannel(),
        }

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a notification action."""
        channel_name = action.notification_channel or "console"
        message = context.resolve(action.notification_message or "")
        title = context.resolve(action.notification_title or "")
        recipients = action.notification_recipients or []
        metadata = self.resolve_params(action.notification_metadata or {}, context)

        # Resolve recipients
        resolved_recipients = [context.resolve(r) for r in recipients]

        logger.info(
            "sending_notification",
            channel=channel_name,
            recipients=len(resolved_recipients),
        )

        channel = self._channels.get(channel_name.lower())
        if not channel:
            return {
                "channel": channel_name,
                "error": f"Unknown notification channel: {channel_name}",
                "success": False,
            }

        try:
            result = await channel.send(
                message=message,
                title=title,
                recipients=resolved_recipients,
                metadata=metadata,
            )

            return {
                "channel": channel_name,
                "message": message[:100] + "..." if len(message) > 100 else message,
                "recipients": resolved_recipients,
                "success": True,
                **result,
            }

        except Exception as e:
            logger.error("notification_error", channel=channel_name, error=str(e))
            return {
                "channel": channel_name,
                "error": str(e),
                "success": False,
            }

    def register_channel(
        self,
        name: str,
        channel: "NotificationChannel",
    ) -> None:
        """Register a custom notification channel."""
        self._channels[name.lower()] = channel


class NotificationChannel:
    """Base class for notification channels."""

    async def send(
        self,
        message: str,
        title: str = "",
        recipients: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Send a notification."""
        raise NotImplementedError


class ConsoleChannel(NotificationChannel):
    """Console/stdout notification channel."""

    async def send(
        self,
        message: str,
        title: str = "",
        recipients: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Print notification to console."""
        output = ""
        if title:
            output = f"[{title}] "
        output += message

        print(output)

        return {"printed": True}


class LogChannel(NotificationChannel):
    """Structured logging notification channel."""

    async def send(
        self,
        message: str,
        title: str = "",
        recipients: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Log notification."""
        log = structlog.get_logger("notification")

        log.info(
            "notification",
            title=title,
            message=message,
            recipients=recipients,
            **(metadata or {}),
        )

        return {"logged": True}


class EmailChannel(NotificationChannel):
    """Email notification channel."""

    async def send(
        self,
        message: str,
        title: str = "",
        recipients: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Send email notification."""
        if not recipients:
            return {"error": "No recipients specified", "sent": False}

        # Would integrate with email service (SMTP, SendGrid, etc.)
        logger.info(
            "email_notification",
            to=recipients,
            subject=title,
            body_preview=message[:100],
        )

        # Simulate for now
        return {
            "sent": True,
            "recipients": recipients,
            "subject": title,
            "simulated": True,
        }


class SlackChannel(NotificationChannel):
    """Slack notification channel."""

    async def send(
        self,
        message: str,
        title: str = "",
        recipients: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Send Slack notification."""
        webhook_url = (metadata or {}).get("webhook_url")

        if not webhook_url:
            return {"error": "No Slack webhook_url in metadata", "sent": False}

        try:
            import httpx

            # Build Slack message
            blocks = []
            if title:
                blocks.append({
                    "type": "header",
                    "text": {"type": "plain_text", "text": title},
                })

            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            })

            payload = {
                "blocks": blocks,
            }

            if recipients:
                # Mention recipients
                mentions = " ".join([f"<@{r}>" for r in recipients])
                payload["text"] = f"{mentions}\n{message}"

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload)

                return {
                    "sent": response.status_code == 200,
                    "status_code": response.status_code,
                }

        except ImportError:
            return {"sent": True, "simulated": True}

        except Exception as e:
            return {"error": str(e), "sent": False}


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""

    async def send(
        self,
        message: str,
        title: str = "",
        recipients: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Send webhook notification."""
        webhook_url = (metadata or {}).get("url")

        if not webhook_url:
            return {"error": "No url in metadata", "sent": False}

        try:
            import httpx

            payload = {
                "title": title,
                "message": message,
                "recipients": recipients,
                **(metadata or {}),
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload)

                return {
                    "sent": response.status_code < 400,
                    "status_code": response.status_code,
                }

        except ImportError:
            return {"sent": True, "simulated": True}

        except Exception as e:
            return {"error": str(e), "sent": False}
