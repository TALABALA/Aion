"""
AION Webhook Action Handler

Call external webhooks from workflows.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import structlog

from aion.automation.types import ActionConfig
from aion.automation.actions.executor import BaseActionHandler

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class WebhookActionHandler(BaseActionHandler):
    """
    Handler for webhook actions.

    Makes HTTP requests to external URLs.
    """

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a webhook action."""
        url = context.resolve(action.webhook_url or "")
        if not url:
            return {"error": "No webhook URL provided", "success": False}

        method = (action.webhook_method or "POST").upper()

        # Resolve headers
        headers = {}
        if action.webhook_headers:
            for key, value in action.webhook_headers.items():
                headers[key] = context.resolve(value)

        # Resolve body
        body = None
        if action.webhook_body:
            body = self.resolve_params(action.webhook_body, context)

        logger.info("calling_webhook", url=url, method=method)

        try:
            import httpx

            timeout = action.webhook_timeout or 30.0

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if body else None,
                    timeout=timeout,
                )

                # Parse response
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        response_body = response.json()
                    except Exception:
                        response_body = response.text
                else:
                    response_body = response.text

                return {
                    "url": url,
                    "method": method,
                    "status_code": response.status_code,
                    "body": response_body,
                    "headers": dict(response.headers),
                    "success": response.status_code < 400,
                }

        except ImportError:
            logger.warning("httpx_not_available")
            # Simulate webhook call
            return {
                "url": url,
                "method": method,
                "status_code": 200,
                "body": {"simulated": True, "message": "Webhook simulated"},
                "headers": {},
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            logger.error("webhook_error", url=url, error=str(e))
            return {
                "url": url,
                "method": method,
                "error": str(e),
                "success": False,
            }

    @staticmethod
    def build_request(
        url: str,
        method: str = "POST",
        headers: Dict[str, str] = None,
        body: Dict[str, Any] = None,
        auth_type: str = None,
        auth_token: str = None,
    ) -> Dict[str, Any]:
        """
        Build a webhook request configuration.

        Helper for constructing webhook action configs.
        """
        config = {
            "url": url,
            "method": method,
            "headers": headers or {},
        }

        if body:
            config["body"] = body

        # Add authentication
        if auth_type and auth_token:
            if auth_type == "bearer":
                config["headers"]["Authorization"] = f"Bearer {auth_token}"
            elif auth_type == "basic":
                import base64
                encoded = base64.b64encode(auth_token.encode()).decode()
                config["headers"]["Authorization"] = f"Basic {encoded}"
            elif auth_type == "api_key":
                config["headers"]["X-API-Key"] = auth_token

        return config

    @staticmethod
    def create_github_webhook(
        repo: str,
        event_type: str,
        payload: Dict[str, Any],
        token: str,
    ) -> Dict[str, Any]:
        """Create a GitHub webhook dispatch request."""
        return {
            "url": f"https://api.github.com/repos/{repo}/dispatches",
            "method": "POST",
            "headers": {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            "body": {
                "event_type": event_type,
                "client_payload": payload,
            },
        }

    @staticmethod
    def create_slack_webhook(
        webhook_url: str,
        message: str,
        channel: str = None,
        username: str = None,
        icon_emoji: str = None,
    ) -> Dict[str, Any]:
        """Create a Slack webhook request."""
        body = {"text": message}

        if channel:
            body["channel"] = channel
        if username:
            body["username"] = username
        if icon_emoji:
            body["icon_emoji"] = icon_emoji

        return {
            "url": webhook_url,
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }

    @staticmethod
    def create_discord_webhook(
        webhook_url: str,
        content: str,
        username: str = None,
        embeds: list = None,
    ) -> Dict[str, Any]:
        """Create a Discord webhook request."""
        body = {"content": content}

        if username:
            body["username"] = username
        if embeds:
            body["embeds"] = embeds

        return {
            "url": webhook_url,
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }
