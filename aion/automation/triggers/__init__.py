"""
AION Workflow Triggers

Trigger types for workflow activation:
- Schedule (cron-based)
- Webhook (HTTP)
- Event (internal events)
- Data change (data mutations)
- Manual (user-initiated)
"""

from aion.automation.triggers.manager import TriggerManager
from aion.automation.triggers.schedule import ScheduleTriggerHandler
from aion.automation.triggers.webhook import WebhookTriggerHandler
from aion.automation.triggers.event import EventTriggerHandler
from aion.automation.triggers.data import DataChangeTriggerHandler
from aion.automation.triggers.manual import ManualTriggerHandler

__all__ = [
    "TriggerManager",
    "ScheduleTriggerHandler",
    "WebhookTriggerHandler",
    "EventTriggerHandler",
    "DataChangeTriggerHandler",
    "ManualTriggerHandler",
]
