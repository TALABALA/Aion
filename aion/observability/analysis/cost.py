"""
AION Cost Tracker

Track and analyze resource costs:
- Token usage and costs
- API call costs
- Storage costs
- Compute costs
- Cost attribution to agents/goals
- Budget management
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.observability.types import CostRecord, CostBudget, ResourceType
from aion.observability.collector import TelemetryCollector

logger = structlog.get_logger(__name__)


# Default pricing (can be overridden)
DEFAULT_PRICING = {
    # Tokens (per 1K tokens)
    ResourceType.TOKENS_INPUT: {
        "claude-opus-4-5": 0.015,
        "claude-sonnet-4": 0.003,
        "claude-3-opus": 0.015,
        "claude-3-sonnet": 0.003,
        "claude-3-haiku": 0.00025,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.0005,
        "llama-3.3-70b": 0.0,  # Local/free
        "default": 0.003,
    },
    ResourceType.TOKENS_OUTPUT: {
        "claude-opus-4-5": 0.075,
        "claude-sonnet-4": 0.015,
        "claude-3-opus": 0.075,
        "claude-3-sonnet": 0.015,
        "claude-3-haiku": 0.00125,
        "gpt-4": 0.06,
        "gpt-4-turbo": 0.03,
        "gpt-3.5-turbo": 0.0015,
        "llama-3.3-70b": 0.0,
        "default": 0.015,
    },
    # API calls (per call)
    ResourceType.API_CALL: {
        "web_search": 0.01,
        "web_fetch": 0.001,
        "embedding": 0.0001,
        "default": 0.0,
    },
    # Storage (per GB per month)
    ResourceType.STORAGE: {
        "memory": 0.10,
        "knowledge": 0.10,
        "files": 0.05,
        "default": 0.10,
    },
    # Embedding (per 1K tokens)
    ResourceType.EMBEDDING: {
        "text-embedding-3-large": 0.00013,
        "text-embedding-3-small": 0.00002,
        "default": 0.0001,
    },
}


class CostTracker:
    """
    SOTA Cost tracking and budget management.

    Features:
    - Token cost tracking with model-specific pricing
    - API call costs
    - Storage costs
    - Cost attribution to agents/goals/users
    - Budget alerts and enforcement
    - Cost forecasting
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        pricing: Dict[ResourceType, Dict[str, float]] = None,
        budget_check_interval: float = 60.0,
    ):
        self.collector = collector
        self.pricing = pricing or DEFAULT_PRICING
        self.budget_check_interval = budget_check_interval

        # Cost aggregation
        self._costs_by_resource: Dict[str, float] = defaultdict(float)
        self._costs_by_model: Dict[str, float] = defaultdict(float)
        self._costs_by_agent: Dict[str, float] = defaultdict(float)
        self._costs_by_goal: Dict[str, float] = defaultdict(float)
        self._costs_by_user: Dict[str, float] = defaultdict(float)
        self._costs_by_day: Dict[str, float] = defaultdict(float)
        self._costs_by_hour: Dict[str, float] = defaultdict(float)

        # Token tracking
        self._tokens_by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

        # Cost records
        self._records: List[CostRecord] = []
        self._max_records = 100000

        # Budgets
        self._budgets: Dict[str, CostBudget] = {}

        # Alert callbacks
        self._budget_alert_callbacks: List[Callable[[CostBudget], None]] = []

        # Background task
        self._budget_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the cost tracker."""
        if self._initialized:
            return

        logger.info("Initializing Cost Tracker")

        # Start budget check loop
        self._budget_check_task = asyncio.create_task(self._budget_check_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the cost tracker."""
        self._shutdown_event.set()

        if self._budget_check_task:
            self._budget_check_task.cancel()
            try:
                await self._budget_check_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    def _get_unit_cost(
        self,
        resource_type: ResourceType,
        resource_name: str,
    ) -> float:
        """Get unit cost for a resource."""
        pricing = self.pricing.get(resource_type, {})
        return pricing.get(resource_name, pricing.get("default", 0.0))

    # === Recording Methods ===

    def record_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        trace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        goal_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> CostRecord:
        """
        Record token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            trace_id: Trace ID for correlation
            agent_id: Agent ID for attribution
            goal_id: Goal ID for attribution
            user_id: User ID for attribution

        Returns:
            Cost record
        """
        input_cost_per_k = self._get_unit_cost(ResourceType.TOKENS_INPUT, model)
        output_cost_per_k = self._get_unit_cost(ResourceType.TOKENS_OUTPUT, model)

        input_cost = (input_tokens / 1000) * input_cost_per_k
        output_cost = (output_tokens / 1000) * output_cost_per_k
        total_cost = input_cost + output_cost
        total_tokens = input_tokens + output_tokens

        record = CostRecord(
            resource_type=ResourceType.TOKENS_INPUT,  # Primary type
            resource_name=model,
            quantity=total_tokens,
            unit="tokens",
            unit_cost=total_cost / total_tokens if total_tokens > 0 else 0,
            total_cost=total_cost,
            trace_id=trace_id,
            agent_id=agent_id,
            goal_id=goal_id,
            user_id=user_id,
            request_id=request_id,
            labels={
                "model": model,
                "input_tokens": str(input_tokens),
                "output_tokens": str(output_tokens),
                "input_cost": f"{input_cost:.6f}",
                "output_cost": f"{output_cost:.6f}",
            },
        )

        self._add_record(record)

        # Update token tracking
        self._tokens_by_model[model]["input"] += input_tokens
        self._tokens_by_model[model]["output"] += output_tokens

        return record

    def record_api_call(
        self,
        api_name: str,
        trace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        goal_id: Optional[str] = None,
        user_id: Optional[str] = None,
        labels: Dict[str, str] = None,
    ) -> CostRecord:
        """Record an API call."""
        cost = self._get_unit_cost(ResourceType.API_CALL, api_name)

        record = CostRecord(
            resource_type=ResourceType.API_CALL,
            resource_name=api_name,
            quantity=1,
            unit="calls",
            unit_cost=cost,
            total_cost=cost,
            trace_id=trace_id,
            agent_id=agent_id,
            goal_id=goal_id,
            user_id=user_id,
            labels=labels or {},
        )

        self._add_record(record)
        return record

    def record_embedding(
        self,
        model: str,
        tokens: int,
        trace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> CostRecord:
        """Record embedding usage."""
        cost_per_k = self._get_unit_cost(ResourceType.EMBEDDING, model)
        total_cost = (tokens / 1000) * cost_per_k

        record = CostRecord(
            resource_type=ResourceType.EMBEDDING,
            resource_name=model,
            quantity=tokens,
            unit="tokens",
            unit_cost=cost_per_k / 1000,
            total_cost=total_cost,
            trace_id=trace_id,
            agent_id=agent_id,
            labels={"model": model},
        )

        self._add_record(record)
        return record

    def record_storage(
        self,
        storage_type: str,
        bytes_used: int,
        trace_id: Optional[str] = None,
    ) -> CostRecord:
        """Record storage usage."""
        gb_used = bytes_used / (1024 ** 3)
        cost_per_gb = self._get_unit_cost(ResourceType.STORAGE, storage_type)

        # Pro-rate to daily cost
        daily_cost = (gb_used * cost_per_gb) / 30

        record = CostRecord(
            resource_type=ResourceType.STORAGE,
            resource_name=storage_type,
            quantity=gb_used,
            unit="GB",
            unit_cost=cost_per_gb / 30,
            total_cost=daily_cost,
            trace_id=trace_id,
            labels={
                "bytes": str(bytes_used),
                "storage_type": storage_type,
            },
        )

        self._add_record(record)
        return record

    def record_custom(
        self,
        resource_type: str,
        resource_name: str,
        quantity: float,
        unit: str,
        unit_cost: float,
        **kwargs,
    ) -> CostRecord:
        """Record custom cost."""
        record = CostRecord(
            resource_type=ResourceType.API_CALL,  # Generic
            resource_name=resource_name,
            quantity=quantity,
            unit=unit,
            unit_cost=unit_cost,
            total_cost=quantity * unit_cost,
            **kwargs,
        )

        self._add_record(record)
        return record

    def _add_record(self, record: CostRecord) -> None:
        """Add a cost record and update aggregations."""
        self._records.append(record)

        # Trim if needed
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        # Update aggregations
        self._costs_by_resource[record.resource_type.value] += record.total_cost
        self._costs_by_model[record.resource_name] += record.total_cost

        if record.agent_id:
            self._costs_by_agent[record.agent_id] += record.total_cost

        if record.goal_id:
            self._costs_by_goal[record.goal_id] += record.total_cost

        if record.user_id:
            self._costs_by_user[record.user_id] += record.total_cost

        day_key = record.timestamp.strftime("%Y-%m-%d")
        hour_key = record.timestamp.strftime("%Y-%m-%d-%H")
        self._costs_by_day[day_key] += record.total_cost
        self._costs_by_hour[hour_key] += record.total_cost

        # Send to collector
        self.collector.collect_cost(record)

    # === Budget Management ===

    def set_budget(
        self,
        name: str,
        amount: float,
        period: str = "daily",
        scope_type: str = "global",
        scope_id: Optional[str] = None,
        warning_threshold: float = 80.0,
        critical_threshold: float = 95.0,
    ) -> CostBudget:
        """Set a budget."""
        budget = CostBudget(
            name=name,
            amount=amount,
            period=period,
            period_start=datetime.utcnow(),
            scope_type=scope_type,
            scope_id=scope_id,
            warning_threshold_percent=warning_threshold,
            critical_threshold_percent=critical_threshold,
        )
        self._budgets[name] = budget
        logger.info(f"Set budget: {name} = ${amount} ({period})")
        return budget

    def remove_budget(self, name: str) -> bool:
        """Remove a budget."""
        if name in self._budgets:
            del self._budgets[name]
            return True
        return False

    def add_budget_alert_callback(self, callback: Callable[[CostBudget], None]) -> None:
        """Add a callback for budget alerts."""
        self._budget_alert_callbacks.append(callback)

    async def _budget_check_loop(self) -> None:
        """Background loop to check budgets."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.budget_check_interval)
                self._check_budgets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Budget check error: {e}")

    def _check_budgets(self) -> None:
        """Check all budgets for threshold breaches."""
        for budget in self._budgets.values():
            # Update current usage
            budget.current_usage = self._get_budget_usage(budget)

            # Check thresholds
            if budget.usage_percent >= budget.critical_threshold_percent:
                logger.warning(
                    f"Budget CRITICAL: {budget.name}",
                    usage_percent=budget.usage_percent,
                    current=budget.current_usage,
                    budget=budget.amount,
                )
                for callback in self._budget_alert_callbacks:
                    try:
                        callback(budget)
                    except Exception as e:
                        logger.error(f"Budget callback error: {e}")

            elif budget.usage_percent >= budget.warning_threshold_percent:
                logger.warning(
                    f"Budget WARNING: {budget.name}",
                    usage_percent=budget.usage_percent,
                )

    def _get_budget_usage(self, budget: CostBudget) -> float:
        """Get current usage for a budget."""
        now = datetime.utcnow()

        # Get relevant records based on period
        if budget.period == "daily":
            day_key = now.strftime("%Y-%m-%d")
            return self._costs_by_day.get(day_key, 0.0)

        elif budget.period == "weekly":
            # Last 7 days
            total = 0.0
            for i in range(7):
                day = now - timedelta(days=i)
                day_key = day.strftime("%Y-%m-%d")
                total += self._costs_by_day.get(day_key, 0.0)
            return total

        elif budget.period == "monthly":
            month_start = now.replace(day=1).strftime("%Y-%m")
            return sum(
                cost for day, cost in self._costs_by_day.items()
                if day.startswith(month_start)
            )

        elif budget.period == "yearly":
            year_start = now.strftime("%Y")
            return sum(
                cost for day, cost in self._costs_by_day.items()
                if day.startswith(year_start)
            )

        return 0.0

    def check_budget(self, name: str) -> Dict[str, Any]:
        """Check a specific budget status."""
        budget = self._budgets.get(name)
        if not budget:
            return {"error": "Budget not found"}

        budget.current_usage = self._get_budget_usage(budget)
        return budget.to_dict()

    def get_all_budgets(self) -> List[Dict[str, Any]]:
        """Get all budgets with current usage."""
        result = []
        for budget in self._budgets.values():
            budget.current_usage = self._get_budget_usage(budget)
            result.append(budget.to_dict())
        return result

    # === Query Methods ===

    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        goal_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost summary with optional filtering."""
        records = self._records

        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]
        if goal_id:
            records = [r for r in records if r.goal_id == goal_id]
        if user_id:
            records = [r for r in records if r.user_id == user_id]

        total = sum(r.total_cost for r in records)

        by_resource = defaultdict(float)
        by_model = defaultdict(float)
        by_day = defaultdict(float)

        for r in records:
            by_resource[r.resource_type.value] += r.total_cost
            by_model[r.resource_name] += r.total_cost
            by_day[r.timestamp.strftime("%Y-%m-%d")] += r.total_cost

        return {
            "total_cost": total,
            "currency": "USD",
            "record_count": len(records),
            "by_resource_type": dict(by_resource),
            "by_model": dict(by_model),
            "by_day": dict(sorted(by_day.items())),
            "average_daily": total / len(by_day) if by_day else 0,
        }

    def get_token_summary(self) -> Dict[str, Any]:
        """Get token usage summary."""
        return {
            "by_model": dict(self._tokens_by_model),
            "total_input": sum(m["input"] for m in self._tokens_by_model.values()),
            "total_output": sum(m["output"] for m in self._tokens_by_model.values()),
        }

    def get_agent_costs(self, agent_id: str) -> Dict[str, Any]:
        """Get costs for a specific agent."""
        records = [r for r in self._records if r.agent_id == agent_id]

        return {
            "agent_id": agent_id,
            "total_cost": sum(r.total_cost for r in records),
            "record_count": len(records),
            "by_resource": {
                rt: sum(r.total_cost for r in records if r.resource_type.value == rt)
                for rt in set(r.resource_type.value for r in records)
            },
        }

    def get_goal_costs(self, goal_id: str) -> Dict[str, Any]:
        """Get costs for a specific goal."""
        records = [r for r in self._records if r.goal_id == goal_id]

        return {
            "goal_id": goal_id,
            "total_cost": sum(r.total_cost for r in records),
            "record_count": len(records),
        }

    def get_forecast(self, days: int = 30) -> Dict[str, Any]:
        """Forecast costs for the next N days."""
        # Calculate average daily cost
        now = datetime.utcnow()
        recent_days = 7

        recent_cost = sum(
            self._costs_by_day.get((now - timedelta(days=i)).strftime("%Y-%m-%d"), 0.0)
            for i in range(recent_days)
        )
        daily_average = recent_cost / recent_days if recent_days > 0 else 0

        return {
            "forecast_days": days,
            "daily_average": daily_average,
            "forecast_total": daily_average * days,
            "based_on_days": recent_days,
            "recent_total": recent_cost,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cost tracker statistics."""
        return {
            "total_records": len(self._records),
            "total_cost": sum(self._costs_by_resource.values()),
            "budgets_count": len(self._budgets),
            "models_tracked": len(self._tokens_by_model),
            "agents_tracked": len(self._costs_by_agent),
            "goals_tracked": len(self._costs_by_goal),
        }
