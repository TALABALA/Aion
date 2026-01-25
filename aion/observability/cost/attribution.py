"""
Cost Attribution and Chargeback Implementation.

Provides:
- Per-service cost allocation
- Resource usage tracking
- Chargeback reports
- Budget management
- Cost optimization recommendations
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of infrastructure costs."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    OBSERVABILITY = "observability"
    MESSAGING = "messaging"
    CACHE = "cache"
    CDN = "cdn"
    THIRD_PARTY = "third_party"
    OTHER = "other"


class AllocationMethod(Enum):
    """Methods for cost allocation."""
    DIRECT = "direct"  # Directly attributed
    PROPORTIONAL = "proportional"  # Based on usage proportion
    EVEN_SPLIT = "even_split"  # Split evenly
    WEIGHTED = "weighted"  # Custom weights
    ACTIVITY_BASED = "activity_based"  # Based on activity metrics


@dataclass
class CostMetric:
    """A metric used for cost calculation."""
    name: str
    value: float
    unit: str
    unit_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ServiceCost:
    """Cost breakdown for a single service."""
    service_name: str
    total_cost: float
    period_start: datetime
    period_end: datetime
    cost_by_category: Dict[CostCategory, float] = field(default_factory=dict)
    cost_by_resource: Dict[str, float] = field(default_factory=dict)
    metrics: List[CostMetric] = field(default_factory=list)
    trend_percent: float = 0.0  # % change from previous period


@dataclass
class CostCenter:
    """A cost center for grouping services."""
    cost_center_id: str
    name: str
    owner: str
    services: List[str] = field(default_factory=list)
    budget: float = 0.0
    allocation_method: AllocationMethod = AllocationMethod.DIRECT


@dataclass
class CostAllocation:
    """A single cost allocation record."""
    allocation_id: str
    timestamp: datetime
    service: str
    cost_center: str
    category: CostCategory
    amount: float
    resource_id: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostReport:
    """Cost report for a period."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_cost: float
    cost_by_service: Dict[str, float] = field(default_factory=dict)
    cost_by_category: Dict[str, float] = field(default_factory=dict)
    cost_by_cost_center: Dict[str, float] = field(default_factory=dict)
    service_details: List[ServiceCost] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Budget:
    """Budget configuration."""
    budget_id: str
    name: str
    amount: float
    period: str  # "monthly", "quarterly", "yearly"
    cost_center: Optional[str] = None
    service: Optional[str] = None
    category: Optional[CostCategory] = None
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 1.0])
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BudgetAlert:
    """Alert when budget threshold is reached."""
    alert_id: str
    budget: Budget
    threshold_percent: float
    current_spend: float
    projected_spend: float
    timestamp: datetime
    message: str


class CostAttributor:
    """
    Main cost attribution engine.

    Tracks resource usage and allocates costs to services/cost centers.
    """

    def __init__(self):
        self._allocations: List[CostAllocation] = []
        self._cost_centers: Dict[str, CostCenter] = {}
        self._unit_costs: Dict[str, float] = {
            # Default unit costs
            "cpu_core_hour": 0.05,
            "memory_gb_hour": 0.01,
            "storage_gb_month": 0.10,
            "network_gb": 0.02,
            "request": 0.000001,
            "trace": 0.00001,
            "metric_point": 0.000001,
            "log_gb": 0.50,
        }

        # Service to cost center mapping
        self._service_cost_center: Dict[str, str] = {}

    def set_unit_cost(self, metric: str, cost: float):
        """Set unit cost for a metric."""
        self._unit_costs[metric] = cost

    def register_cost_center(self, cost_center: CostCenter):
        """Register a cost center."""
        self._cost_centers[cost_center.cost_center_id] = cost_center
        for service in cost_center.services:
            self._service_cost_center[service] = cost_center.cost_center_id

    def record_usage(self, service: str, metric_name: str, value: float,
                     unit: str = "", category: CostCategory = CostCategory.COMPUTE,
                     tags: Dict[str, str] = None) -> CostAllocation:
        """Record resource usage for cost calculation."""
        import uuid

        unit_cost = self._unit_costs.get(metric_name, 0)
        amount = value * unit_cost

        cost_center = self._service_cost_center.get(service, "unassigned")

        allocation = CostAllocation(
            allocation_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            service=service,
            cost_center=cost_center,
            category=category,
            amount=amount,
            description=f"{value} {unit} of {metric_name}",
            tags=tags or {}
        )

        self._allocations.append(allocation)
        return allocation

    def get_service_cost(self, service: str, period_start: datetime,
                         period_end: datetime) -> ServiceCost:
        """Get cost breakdown for a service."""
        allocations = [a for a in self._allocations
                      if a.service == service
                      and period_start <= a.timestamp <= period_end]

        total = sum(a.amount for a in allocations)

        cost_by_category = defaultdict(float)
        for a in allocations:
            cost_by_category[a.category] += a.amount

        return ServiceCost(
            service_name=service,
            total_cost=total,
            period_start=period_start,
            period_end=period_end,
            cost_by_category=dict(cost_by_category),
        )

    def generate_report(self, period_start: datetime, period_end: datetime) -> CostReport:
        """Generate comprehensive cost report."""
        import uuid

        allocations = [a for a in self._allocations
                      if period_start <= a.timestamp <= period_end]

        total = sum(a.amount for a in allocations)

        cost_by_service = defaultdict(float)
        cost_by_category = defaultdict(float)
        cost_by_cost_center = defaultdict(float)

        for a in allocations:
            cost_by_service[a.service] += a.amount
            cost_by_category[a.category.value] += a.amount
            cost_by_cost_center[a.cost_center] += a.amount

        # Generate recommendations
        recommendations = self._generate_recommendations(allocations, cost_by_service)

        # Detect anomalies
        anomalies = self._detect_cost_anomalies(allocations)

        # Service details
        service_details = [
            self.get_service_cost(service, period_start, period_end)
            for service in cost_by_service.keys()
        ]

        return CostReport(
            report_id=str(uuid.uuid4())[:8],
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            total_cost=total,
            cost_by_service=dict(cost_by_service),
            cost_by_category=dict(cost_by_category),
            cost_by_cost_center=dict(cost_by_cost_center),
            service_details=service_details,
            recommendations=recommendations,
            anomalies=anomalies
        )

    def _generate_recommendations(self, allocations: List[CostAllocation],
                                  cost_by_service: Dict[str, float]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Find top spenders
        sorted_services = sorted(cost_by_service.items(), key=lambda x: x[1], reverse=True)
        if sorted_services:
            top_service, top_cost = sorted_services[0]
            recommendations.append(
                f"Review {top_service} - highest cost at ${top_cost:.2f}"
            )

        # Check for unassigned costs
        unassigned = sum(a.amount for a in allocations if a.cost_center == "unassigned")
        if unassigned > 0:
            recommendations.append(
                f"Assign ${unassigned:.2f} in unassigned costs to cost centers"
            )

        return recommendations

    def _detect_cost_anomalies(self, allocations: List[CostAllocation]) -> List[Dict]:
        """Detect unusual cost patterns."""
        anomalies = []

        # Group by service and detect spikes
        daily_costs = defaultdict(lambda: defaultdict(float))
        for a in allocations:
            day = a.timestamp.date()
            daily_costs[a.service][day] += a.amount

        for service, daily in daily_costs.items():
            if len(daily) < 2:
                continue

            values = list(daily.values())
            avg = sum(values) / len(values)
            for day, cost in daily.items():
                if cost > avg * 2:  # 2x spike
                    anomalies.append({
                        "service": service,
                        "date": str(day),
                        "cost": cost,
                        "average": avg,
                        "type": "spike"
                    })

        return anomalies


class ChargebackEngine:
    """
    Chargeback engine for billing internal teams.

    Handles:
    - Cost allocation to teams
    - Invoice generation
    - Usage-based billing
    """

    def __init__(self, cost_attributor: CostAttributor):
        self.cost_attributor = cost_attributor
        self._invoices: List[Dict] = []

    def generate_chargeback(self, cost_center_id: str, period_start: datetime,
                           period_end: datetime) -> Dict[str, Any]:
        """Generate chargeback for a cost center."""
        import uuid

        cost_center = self.cost_attributor._cost_centers.get(cost_center_id)
        if not cost_center:
            raise ValueError(f"Cost center not found: {cost_center_id}")

        allocations = [a for a in self.cost_attributor._allocations
                      if a.cost_center == cost_center_id
                      and period_start <= a.timestamp <= period_end]

        total = sum(a.amount for a in allocations)

        # Group by service
        by_service = defaultdict(float)
        for a in allocations:
            by_service[a.service] += a.amount

        # Group by category
        by_category = defaultdict(float)
        for a in allocations:
            by_category[a.category.value] += a.amount

        invoice = {
            "invoice_id": str(uuid.uuid4())[:8],
            "cost_center_id": cost_center_id,
            "cost_center_name": cost_center.name,
            "owner": cost_center.owner,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "total_amount": total,
            "currency": "USD",
            "line_items": [
                {"service": s, "amount": a}
                for s, a in by_service.items()
            ],
            "by_category": dict(by_category),
            "status": "pending"
        }

        self._invoices.append(invoice)
        return invoice

    def get_invoices(self, cost_center_id: str = None, status: str = None) -> List[Dict]:
        """Get invoices with optional filters."""
        results = self._invoices

        if cost_center_id:
            results = [i for i in results if i["cost_center_id"] == cost_center_id]
        if status:
            results = [i for i in results if i["status"] == status]

        return results


class BudgetManager:
    """
    Budget management for cost control.

    Features:
    - Budget creation and tracking
    - Alert generation
    - Forecast vs actual comparison
    """

    def __init__(self, cost_attributor: CostAttributor):
        self.cost_attributor = cost_attributor
        self._budgets: Dict[str, Budget] = {}
        self._alerts: List[BudgetAlert] = []
        self._alert_handlers: List[Callable[[BudgetAlert], None]] = []

    def create_budget(self, budget: Budget):
        """Create a budget."""
        self._budgets[budget.budget_id] = budget

    def add_alert_handler(self, handler: Callable[[BudgetAlert], None]):
        """Add alert handler."""
        self._alert_handlers.append(handler)

    def check_budgets(self) -> List[BudgetAlert]:
        """Check all budgets and generate alerts."""
        import uuid
        alerts = []

        for budget in self._budgets.values():
            # Get current spend
            period_start, period_end = self._get_budget_period(budget)
            current_spend = self._get_budget_spend(budget, period_start, period_end)

            # Check thresholds
            spend_percent = current_spend / budget.amount if budget.amount > 0 else 0

            for threshold in budget.alert_thresholds:
                if spend_percent >= threshold:
                    # Project end-of-period spend
                    days_elapsed = (datetime.now() - period_start).days + 1
                    total_days = (period_end - period_start).days
                    projected = current_spend * (total_days / days_elapsed) if days_elapsed > 0 else current_spend

                    alert = BudgetAlert(
                        alert_id=str(uuid.uuid4())[:8],
                        budget=budget,
                        threshold_percent=threshold,
                        current_spend=current_spend,
                        projected_spend=projected,
                        timestamp=datetime.now(),
                        message=f"Budget {budget.name} at {spend_percent*100:.1f}% "
                               f"(${current_spend:.2f} of ${budget.amount:.2f})"
                    )
                    alerts.append(alert)
                    self._alerts.append(alert)

                    for handler in self._alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Budget alert handler error: {e}")

                    break  # Only one alert per budget

        return alerts

    def _get_budget_period(self, budget: Budget) -> Tuple[datetime, datetime]:
        """Get current budget period."""
        now = datetime.now()

        if budget.period == "monthly":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif budget.period == "quarterly":
            quarter = (now.month - 1) // 3
            start = now.replace(month=quarter * 3 + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_month = quarter * 3 + 4
            if end_month > 12:
                end = start.replace(year=now.year + 1, month=end_month - 12)
            else:
                end = start.replace(month=end_month)
        else:  # yearly
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=now.year + 1)

        return start, end

    def _get_budget_spend(self, budget: Budget, period_start: datetime,
                          period_end: datetime) -> float:
        """Get current spend for a budget."""
        allocations = self.cost_attributor._allocations

        # Filter by period
        allocations = [a for a in allocations
                      if period_start <= a.timestamp <= period_end]

        # Filter by budget scope
        if budget.cost_center:
            allocations = [a for a in allocations if a.cost_center == budget.cost_center]
        if budget.service:
            allocations = [a for a in allocations if a.service == budget.service]
        if budget.category:
            allocations = [a for a in allocations if a.category == budget.category]

        return sum(a.amount for a in allocations)

    def get_budget_status(self, budget_id: str) -> Dict[str, Any]:
        """Get current status of a budget."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return {}

        period_start, period_end = self._get_budget_period(budget)
        current_spend = self._get_budget_spend(budget, period_start, period_end)

        days_elapsed = (datetime.now() - period_start).days + 1
        total_days = (period_end - period_start).days
        projected = current_spend * (total_days / days_elapsed) if days_elapsed > 0 else current_spend

        return {
            "budget_id": budget_id,
            "name": budget.name,
            "amount": budget.amount,
            "current_spend": current_spend,
            "projected_spend": projected,
            "percent_used": (current_spend / budget.amount * 100) if budget.amount > 0 else 0,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "days_remaining": (period_end - datetime.now()).days,
            "status": "on_track" if projected <= budget.amount else "over_budget"
        }
