"""
AION Schedule Trigger Handler

Cron-based schedule triggers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from aion.automation.types import Trigger, TriggerType
from aion.automation.triggers.manager import BaseTriggerHandler

logger = structlog.get_logger(__name__)


class ScheduleTriggerHandler(BaseTriggerHandler):
    """
    Handler for schedule (cron) triggers.

    Features:
    - Cron expression parsing
    - Timezone support
    - Next run calculation
    """

    async def register(self, trigger: Trigger) -> None:
        """Register a schedule trigger."""
        config = trigger.config

        if not config.cron_expression:
            logger.warning("schedule_missing_cron", trigger_id=trigger.id)
            return

        # Calculate next trigger time
        trigger.next_trigger_at = self._calculate_next(config.cron_expression, config.timezone)

        logger.info(
            "schedule_registered",
            trigger_id=trigger.id,
            cron=config.cron_expression,
            timezone=config.timezone,
            next_trigger=trigger.next_trigger_at,
        )

    async def unregister(self, trigger: Trigger) -> None:
        """Unregister a schedule trigger."""
        logger.info("schedule_unregistered", trigger_id=trigger.id)

    def _calculate_next(
        self,
        cron_expression: str,
        timezone: str = "UTC",
    ) -> Optional[datetime]:
        """Calculate next trigger time."""
        try:
            from croniter import croniter
            import pytz

            tz = pytz.timezone(timezone)
            now = datetime.now(tz)

            cron = croniter(cron_expression, now)
            next_time = cron.get_next(datetime)

            return next_time

        except ImportError:
            logger.warning("croniter_or_pytz_not_installed")
            return self._simple_next_minute()
        except Exception as e:
            logger.error("cron_calculation_error", error=str(e))
            return None

    def _simple_next_minute(self) -> datetime:
        """Simple fallback: next minute."""
        from datetime import timedelta
        now = datetime.now()
        return now.replace(second=0, microsecond=0) + timedelta(minutes=1)

    @staticmethod
    def parse_cron(expression: str) -> Dict[str, Any]:
        """
        Parse a cron expression for validation.

        Standard cron format:
        minute hour day_of_month month day_of_week

        Extended format (with seconds):
        second minute hour day_of_month month day_of_week
        """
        parts = expression.split()

        if len(parts) == 5:
            # Standard cron
            return {
                "minute": parts[0],
                "hour": parts[1],
                "day_of_month": parts[2],
                "month": parts[3],
                "day_of_week": parts[4],
            }
        elif len(parts) == 6:
            # Extended cron with seconds
            return {
                "second": parts[0],
                "minute": parts[1],
                "hour": parts[2],
                "day_of_month": parts[3],
                "month": parts[4],
                "day_of_week": parts[5],
            }
        else:
            raise ValueError(f"Invalid cron expression: {expression}")

    @staticmethod
    def validate_cron(expression: str) -> bool:
        """Validate a cron expression."""
        try:
            from croniter import croniter
            croniter(expression)
            return True
        except Exception:
            return False

    @staticmethod
    def describe_cron(expression: str) -> str:
        """Get human-readable description of cron expression."""
        # Common patterns
        descriptions = {
            "* * * * *": "Every minute",
            "0 * * * *": "Every hour",
            "0 0 * * *": "Every day at midnight",
            "0 0 * * 0": "Every Sunday at midnight",
            "0 0 1 * *": "First day of every month at midnight",
            "0 9 * * 1-5": "Every weekday at 9 AM",
            "*/5 * * * *": "Every 5 minutes",
            "*/15 * * * *": "Every 15 minutes",
            "0 */2 * * *": "Every 2 hours",
        }

        if expression in descriptions:
            return descriptions[expression]

        # Try to generate a description
        try:
            parts = ScheduleTriggerHandler.parse_cron(expression)
            desc_parts = []

            minute = parts.get("minute", "*")
            hour = parts.get("hour", "*")
            day_of_month = parts.get("day_of_month", "*")
            month = parts.get("month", "*")
            day_of_week = parts.get("day_of_week", "*")

            if minute.startswith("*/"):
                desc_parts.append(f"every {minute[2:]} minutes")
            elif minute != "*":
                desc_parts.append(f"at minute {minute}")

            if hour.startswith("*/"):
                desc_parts.append(f"every {hour[2:]} hours")
            elif hour != "*":
                desc_parts.append(f"at hour {hour}")

            if day_of_month != "*":
                desc_parts.append(f"on day {day_of_month}")

            if month != "*":
                desc_parts.append(f"in month {month}")

            if day_of_week != "*":
                days = {
                    "0": "Sunday", "1": "Monday", "2": "Tuesday",
                    "3": "Wednesday", "4": "Thursday", "5": "Friday",
                    "6": "Saturday", "7": "Sunday",
                }
                if day_of_week in days:
                    desc_parts.append(f"on {days[day_of_week]}")
                elif "-" in day_of_week:
                    desc_parts.append(f"on days {day_of_week}")

            return " ".join(desc_parts) if desc_parts else expression

        except Exception:
            return expression
