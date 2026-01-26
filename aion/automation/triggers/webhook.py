"""
AION Webhook Trigger Handler

HTTP webhook triggers.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from aion.automation.types import Trigger, TriggerType
from aion.automation.triggers.manager import BaseTriggerHandler

logger = structlog.get_logger(__name__)


class WebhookTriggerHandler(BaseTriggerHandler):
    """
    Handler for webhook triggers.

    Features:
    - Path-based routing
    - Secret validation (HMAC)
    - Method filtering
    - Header extraction
    """

    async def register(self, trigger: Trigger) -> None:
        """Register a webhook trigger."""
        config = trigger.config

        # Generate path if not specified
        path = config.webhook_path or f"/webhooks/{trigger.id}"

        logger.info(
            "webhook_registered",
            trigger_id=trigger.id,
            path=path,
            methods=config.webhook_methods,
            has_secret=bool(config.webhook_secret),
        )

    async def unregister(self, trigger: Trigger) -> None:
        """Unregister a webhook trigger."""
        logger.info("webhook_unregistered", trigger_id=trigger.id)

    @staticmethod
    def validate_signature(
        payload: bytes,
        signature: str,
        secret: str,
        algorithm: str = "sha256",
    ) -> bool:
        """
        Validate webhook signature.

        Supports HMAC-based signatures commonly used by:
        - GitHub (X-Hub-Signature-256)
        - Stripe (Stripe-Signature)
        - Slack (X-Slack-Signature)
        """
        try:
            if algorithm == "sha256":
                expected = hmac.new(
                    secret.encode(),
                    payload,
                    hashlib.sha256,
                ).hexdigest()
            elif algorithm == "sha1":
                expected = hmac.new(
                    secret.encode(),
                    payload,
                    hashlib.sha1,
                ).hexdigest()
            else:
                return False

            # Handle prefixed signatures (e.g., "sha256=...")
            if "=" in signature:
                signature = signature.split("=", 1)[1]

            return hmac.compare_digest(expected, signature)

        except Exception as e:
            logger.error("signature_validation_error", error=str(e))
            return False

    @staticmethod
    def extract_github_event(headers: Dict[str, str]) -> Optional[str]:
        """Extract GitHub webhook event type."""
        return headers.get("x-github-event")

    @staticmethod
    def extract_stripe_event(payload: Dict[str, Any]) -> Optional[str]:
        """Extract Stripe webhook event type."""
        return payload.get("type")

    @staticmethod
    def extract_slack_event(payload: Dict[str, Any]) -> Optional[str]:
        """Extract Slack webhook event type."""
        if "event" in payload:
            return payload["event"].get("type")
        return payload.get("type")

    @staticmethod
    def handle_verification(
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Handle webhook verification challenges.

        Many services send verification requests to confirm endpoint ownership.
        """
        # Slack URL verification
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}

        # Stripe test webhook
        if payload.get("type") == "checkout.session.completed":
            if payload.get("data", {}).get("object", {}).get("mode") == "test":
                logger.info("stripe_test_webhook_received")

        return None

    @staticmethod
    def parse_content_type(content_type: str) -> str:
        """Parse content type header."""
        if not content_type:
            return "application/json"

        # Handle charset and other parameters
        parts = content_type.split(";")
        return parts[0].strip().lower()

    @staticmethod
    def generate_webhook_path(
        workflow_id: str,
        prefix: str = "/webhooks",
    ) -> str:
        """Generate a unique webhook path."""
        # Create a short hash of the workflow ID
        hash_suffix = hashlib.sha256(workflow_id.encode()).hexdigest()[:8]
        return f"{prefix}/{hash_suffix}"

    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """Generate a secure webhook secret."""
        import secrets
        return secrets.token_hex(length)
