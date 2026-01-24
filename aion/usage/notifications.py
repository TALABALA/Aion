"""
AION Usage Notification System

State-of-the-art notification system for:
- Email notifications at usage thresholds
- In-app notifications and banners
- Push notification support
- Webhook integrations
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import json

import structlog

from aion.usage.models import (
    UsageAlert,
    AlertType,
    AlertSeverity,
    UsageMetric,
    SubscriptionTier,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Notification Types
# =============================================================================

class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    IN_APP = "in_app"
    PUSH = "push"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationTemplate:
    """Template for notification content."""
    name: str
    subject_template: str
    body_template: str
    html_template: Optional[str] = None

    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context."""
        return {
            "subject": self.subject_template.format(**context),
            "body": self.body_template.format(**context),
            "html": self.html_template.format(**context) if self.html_template else None,
        }


@dataclass
class UsageNotification:
    """A notification to be sent to a user."""
    notification_id: str
    user_id: str
    channel: NotificationChannel
    priority: NotificationPriority

    # Content
    subject: str
    body: str
    html_body: Optional[str] = None

    # Metadata
    alert: Optional[UsageAlert] = None
    data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_for: Optional[datetime] = None
    sent_at: Optional[datetime] = None

    # Status
    status: str = "pending"  # pending, sent, failed, canceled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "channel": self.channel.value,
            "priority": self.priority.value,
            "subject": self.subject,
            "body": self.body,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
        }


# =============================================================================
# Notification Templates
# =============================================================================

NOTIFICATION_TEMPLATES = {
    AlertType.APPROACHING_LIMIT: NotificationTemplate(
        name="approaching_limit",
        subject_template="Usage Alert: {metric_name} at {percentage}%",
        body_template="""
Hi {user_name},

You've used {current} of your {limit} {metric_name} this month ({percentage}%).

To avoid service interruption, consider:
- Managing your usage
- Upgrading your plan for higher limits

View your usage: {usage_url}

Best,
The AION Team
        """.strip(),
        html_template="""
<!DOCTYPE html>
<html>
<head>
    <style>
        .container {{ max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ padding: 20px; }}
        .progress-bar {{ background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }}
        .progress-fill {{ background: #ffc107; height: 100%; width: {percentage}%; }}
        .cta {{ display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Usage Alert</h2>
        </div>
        <div class="content">
            <p>Hi {user_name},</p>
            <p>You've used <strong>{current}</strong> of your <strong>{limit}</strong> {metric_name} this month.</p>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <p style="text-align: center; color: #666;">{percentage}% used</p>
            <p style="text-align: center;">
                <a href="{usage_url}" class="cta">View Usage</a>
                <a href="{upgrade_url}" class="cta" style="background: #28a745; margin-left: 10px;">Upgrade Plan</a>
            </p>
        </div>
    </div>
</body>
</html>
        """.strip(),
    ),

    AlertType.NEAR_LIMIT: NotificationTemplate(
        name="near_limit",
        subject_template="Warning: {metric_name} at {percentage}% - Action Required",
        body_template="""
Hi {user_name},

You're running low on {metric_name}!

Current usage: {current} of {limit} ({percentage}%)
Remaining: {remaining}

At your current rate, you may reach your limit soon.

Upgrade now to avoid interruption: {upgrade_url}

View your usage: {usage_url}

Best,
The AION Team
        """.strip(),
        html_template="""
<!DOCTYPE html>
<html>
<head>
    <style>
        .container {{ max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif; }}
        .header {{ background: #fff3cd; padding: 20px; border-radius: 8px 8px 0 0; border-left: 4px solid #ffc107; }}
        .content {{ padding: 20px; }}
        .progress-bar {{ background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }}
        .progress-fill {{ background: #fd7e14; height: 100%; width: {percentage}%; }}
        .stats {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; }}
        .cta {{ display: inline-block; padding: 12px 24px; background: #28a745; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>⚠️ Usage Warning</h2>
        </div>
        <div class="content">
            <p>Hi {user_name},</p>
            <p>You're running low on <strong>{metric_name}</strong>!</p>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <div class="stats">
                <p><strong>Used:</strong> {current} / {limit}</p>
                <p><strong>Remaining:</strong> {remaining}</p>
                <p><strong>Days left in period:</strong> {days_remaining}</p>
            </div>
            <p style="text-align: center;">
                <a href="{upgrade_url}" class="cta">Upgrade Now</a>
            </p>
        </div>
    </div>
</body>
</html>
        """.strip(),
    ),

    AlertType.LIMIT_REACHED: NotificationTemplate(
        name="limit_reached",
        subject_template="🚫 {metric_name} Limit Reached",
        body_template="""
Hi {user_name},

You've reached your {metric_name} limit for this billing period.

Current plan: {tier}
Limit: {limit}

To continue using this feature, please upgrade your plan: {upgrade_url}

Your limit will reset on {reset_date}.

View your usage: {usage_url}

Best,
The AION Team
        """.strip(),
        html_template="""
<!DOCTYPE html>
<html>
<head>
    <style>
        .container {{ max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif; }}
        .header {{ background: #f8d7da; padding: 20px; border-radius: 8px 8px 0 0; border-left: 4px solid #dc3545; }}
        .content {{ padding: 20px; }}
        .limit-box {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; text-align: center; }}
        .cta {{ display: inline-block; padding: 12px 24px; background: #dc3545; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🚫 Limit Reached</h2>
        </div>
        <div class="content">
            <p>Hi {user_name},</p>
            <p>You've reached your <strong>{metric_name}</strong> limit for this billing period.</p>
            <div class="limit-box">
                <p style="font-size: 24px; margin: 0;"><strong>{limit}</strong></p>
                <p style="color: #666; margin: 5px 0;">Maximum {metric_name}</p>
                <p style="color: #666; margin: 5px 0;">Resets: {reset_date}</p>
            </div>
            <p style="text-align: center;">
                <a href="{upgrade_url}" class="cta">Upgrade for Unlimited Access</a>
            </p>
        </div>
    </div>
</body>
</html>
        """.strip(),
    ),

    AlertType.TIER_UPGRADE_SUGGESTED: NotificationTemplate(
        name="upgrade_suggested",
        subject_template="Unlock More with AION Pro",
        body_template="""
Hi {user_name},

Based on your usage patterns, you might benefit from upgrading to {recommended_tier}!

Benefits of {recommended_tier}:
{tier_benefits}

Your current usage shows you'd get great value from the upgrade.

Learn more: {upgrade_url}

Best,
The AION Team
        """.strip(),
        html_template=None,
    ),
}


# =============================================================================
# Notification Providers
# =============================================================================

class NotificationProvider(ABC):
    """Abstract notification provider."""

    @abstractmethod
    async def send(self, notification: UsageNotification) -> bool:
        """Send a notification. Returns True if successful."""
        pass

    @abstractmethod
    def supports_channel(self, channel: NotificationChannel) -> bool:
        """Check if this provider supports a channel."""
        pass


class EmailProvider(NotificationProvider):
    """
    Email notification provider.

    Supports SMTP and various email services.
    """

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: str = "noreply@aion.ai",
        from_name: str = "AION",
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls

    def supports_channel(self, channel: NotificationChannel) -> bool:
        return channel == NotificationChannel.EMAIL

    async def send(self, notification: UsageNotification) -> bool:
        """Send email notification."""
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Get user email from notification data
            user_email = notification.data.get("email")
            if not user_email:
                logger.warning("No email address for notification", user_id=notification.user_id)
                return False

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = notification.subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = user_email

            # Add text part
            msg.attach(MIMEText(notification.body, "plain"))

            # Add HTML part if available
            if notification.html_body:
                msg.attach(MIMEText(notification.html_body, "html"))

            # Send
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                use_tls=self.use_tls,
            )

            logger.info(
                "Email notification sent",
                user_id=notification.user_id,
                to=user_email,
            )
            return True

        except ImportError:
            logger.warning("aiosmtplib not installed, email not sent")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}", user_id=notification.user_id)
            return False


class InAppProvider(NotificationProvider):
    """
    In-app notification provider.

    Stores notifications for retrieval by the client.
    """

    def __init__(self):
        self._notifications: Dict[str, List[UsageNotification]] = {}
        self._lock = asyncio.Lock()

    def supports_channel(self, channel: NotificationChannel) -> bool:
        return channel == NotificationChannel.IN_APP

    async def send(self, notification: UsageNotification) -> bool:
        """Store in-app notification."""
        async with self._lock:
            if notification.user_id not in self._notifications:
                self._notifications[notification.user_id] = []

            self._notifications[notification.user_id].append(notification)

            # Keep only last 50 notifications per user
            self._notifications[notification.user_id] = self._notifications[notification.user_id][-50:]

        logger.info("In-app notification stored", user_id=notification.user_id)
        return True

    async def get_notifications(
        self,
        user_id: str,
        unread_only: bool = True,
    ) -> List[UsageNotification]:
        """Get notifications for a user."""
        async with self._lock:
            notifications = self._notifications.get(user_id, [])

            if unread_only:
                notifications = [n for n in notifications if n.status == "pending"]

            return notifications

    async def mark_read(self, user_id: str, notification_ids: List[str]) -> None:
        """Mark notifications as read."""
        async with self._lock:
            notifications = self._notifications.get(user_id, [])
            for n in notifications:
                if n.notification_id in notification_ids:
                    n.status = "read"


class WebhookProvider(NotificationProvider):
    """
    Webhook notification provider.

    Sends notifications to configured webhook URLs.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._webhooks: Dict[str, str] = {}  # user_id -> webhook_url

    def supports_channel(self, channel: NotificationChannel) -> bool:
        return channel == NotificationChannel.WEBHOOK

    def register_webhook(self, user_id: str, webhook_url: str) -> None:
        """Register a webhook URL for a user."""
        self._webhooks[user_id] = webhook_url

    async def send(self, notification: UsageNotification) -> bool:
        """Send webhook notification."""
        webhook_url = self._webhooks.get(notification.user_id)
        if not webhook_url:
            webhook_url = notification.data.get("webhook_url")

        if not webhook_url:
            logger.warning("No webhook URL for notification", user_id=notification.user_id)
            return False

        try:
            import aiohttp

            payload = {
                "type": "usage_notification",
                "notification": notification.to_dict(),
                "alert": notification.alert.to_dict() if notification.alert else None,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status < 400:
                        logger.info(
                            "Webhook notification sent",
                            user_id=notification.user_id,
                            url=webhook_url,
                        )
                        return True
                    else:
                        logger.warning(
                            f"Webhook returned {response.status}",
                            user_id=notification.user_id,
                        )
                        return False

        except ImportError:
            logger.warning("aiohttp not installed, webhook not sent")
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}", user_id=notification.user_id)
            return False


# =============================================================================
# Notification Service
# =============================================================================

class UsageNotificationService:
    """
    Central notification service coordinating all channels.

    Features:
    - Multi-channel notification delivery
    - Template-based content generation
    - Notification preferences
    - Rate limiting for notifications
    - Retry logic for failed notifications
    """

    def __init__(
        self,
        providers: Optional[Dict[NotificationChannel, NotificationProvider]] = None,
        default_channels: Optional[List[NotificationChannel]] = None,
        rate_limit_per_hour: int = 10,
    ):
        self.providers = providers or {}
        self.default_channels = default_channels or [
            NotificationChannel.IN_APP,
            NotificationChannel.EMAIL,
        ]
        self.rate_limit_per_hour = rate_limit_per_hour

        # User preferences
        self._user_preferences: Dict[str, Dict[str, Any]] = {}

        # Rate limiting
        self._sent_counts: Dict[str, List[datetime]] = {}

        # Notification ID counter
        self._notification_counter = 0

        # Initialize default providers if none provided
        if not self.providers:
            self.providers[NotificationChannel.IN_APP] = InAppProvider()

    def _generate_notification_id(self) -> str:
        """Generate unique notification ID."""
        self._notification_counter += 1
        return f"notif_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._notification_counter}"

    def set_user_preferences(
        self,
        user_id: str,
        channels: List[NotificationChannel],
        email: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        """Set notification preferences for a user."""
        self._user_preferences[user_id] = {
            "channels": channels,
            "email": email,
            "webhook_url": webhook_url,
        }

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within notification rate limit."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)

        # Get recent notifications for user
        recent = self._sent_counts.get(user_id, [])
        recent = [t for t in recent if t > cutoff]
        self._sent_counts[user_id] = recent

        return len(recent) < self.rate_limit_per_hour

    def _record_notification(self, user_id: str) -> None:
        """Record that a notification was sent."""
        if user_id not in self._sent_counts:
            self._sent_counts[user_id] = []
        self._sent_counts[user_id].append(datetime.utcnow())

    async def send_alert_notification(
        self,
        alert: UsageAlert,
        user_name: str = "there",
        user_email: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[UsageNotification]:
        """
        Send notifications for a usage alert.

        Returns list of notifications sent.
        """
        # Check rate limit
        if not self._check_rate_limit(alert.user_id):
            logger.info("Rate limit exceeded for notifications", user_id=alert.user_id)
            return []

        # Get template
        template = NOTIFICATION_TEMPLATES.get(alert.alert_type)
        if not template:
            logger.warning(f"No template for alert type: {alert.alert_type}")
            return []

        # Build context
        context = {
            "user_name": user_name,
            "metric_name": alert.metric.value.replace("_", " ").title(),
            "current": int(alert.current_value),
            "limit": int(alert.limit_value) if alert.limit_value else "unlimited",
            "remaining": int(alert.limit_value - alert.current_value) if alert.limit_value else 0,
            "percentage": int(alert.percentage),
            "tier": "Free",  # Would come from subscription
            "reset_date": self._get_reset_date(),
            "usage_url": "/usage",
            "upgrade_url": "/pricing",
            "days_remaining": self._get_days_remaining(),
        }

        if extra_context:
            context.update(extra_context)

        # Render template
        rendered = template.render(context)

        # Get user's preferred channels
        prefs = self._user_preferences.get(alert.user_id, {})
        channels = prefs.get("channels", self.default_channels)

        # Determine priority
        priority = NotificationPriority.NORMAL
        if alert.severity == AlertSeverity.CRITICAL:
            priority = NotificationPriority.URGENT
        elif alert.severity == AlertSeverity.WARNING:
            priority = NotificationPriority.HIGH

        # Send via each channel
        sent_notifications = []

        for channel in channels:
            provider = self.providers.get(channel)
            if not provider:
                continue

            notification = UsageNotification(
                notification_id=self._generate_notification_id(),
                user_id=alert.user_id,
                channel=channel,
                priority=priority,
                subject=rendered["subject"],
                body=rendered["body"],
                html_body=rendered.get("html"),
                alert=alert,
                data={
                    "email": user_email or prefs.get("email"),
                    "webhook_url": prefs.get("webhook_url"),
                },
            )

            try:
                success = await provider.send(notification)
                if success:
                    notification.status = "sent"
                    notification.sent_at = datetime.utcnow()
                    sent_notifications.append(notification)
                else:
                    notification.status = "failed"
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                notification.status = "failed"

        # Record for rate limiting
        if sent_notifications:
            self._record_notification(alert.user_id)

        return sent_notifications

    async def send_custom_notification(
        self,
        user_id: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> List[UsageNotification]:
        """Send a custom notification."""
        if not self._check_rate_limit(user_id):
            return []

        channels = channels or self.default_channels
        prefs = self._user_preferences.get(user_id, {})

        sent_notifications = []

        for channel in channels:
            provider = self.providers.get(channel)
            if not provider:
                continue

            notification = UsageNotification(
                notification_id=self._generate_notification_id(),
                user_id=user_id,
                channel=channel,
                priority=priority,
                subject=subject,
                body=body,
                html_body=html_body,
                data={
                    "email": prefs.get("email"),
                    "webhook_url": prefs.get("webhook_url"),
                },
            )

            success = await provider.send(notification)
            if success:
                notification.status = "sent"
                notification.sent_at = datetime.utcnow()
                sent_notifications.append(notification)

        if sent_notifications:
            self._record_notification(user_id)

        return sent_notifications

    async def get_in_app_notifications(
        self,
        user_id: str,
        unread_only: bool = True,
    ) -> List[UsageNotification]:
        """Get in-app notifications for a user."""
        provider = self.providers.get(NotificationChannel.IN_APP)
        if isinstance(provider, InAppProvider):
            return await provider.get_notifications(user_id, unread_only)
        return []

    async def mark_notifications_read(
        self,
        user_id: str,
        notification_ids: List[str],
    ) -> None:
        """Mark notifications as read."""
        provider = self.providers.get(NotificationChannel.IN_APP)
        if isinstance(provider, InAppProvider):
            await provider.mark_read(user_id, notification_ids)

    def _get_reset_date(self) -> str:
        """Get the next billing period reset date."""
        now = datetime.utcnow()
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        return next_month.strftime("%B %d, %Y")

    def _get_days_remaining(self) -> int:
        """Get days remaining in billing period."""
        now = datetime.utcnow()
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        return (next_month - now).days
