"""
Agent Dashboard

Real-time dashboard for agent monitoring and management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import structlog

from aion.systems.agents.observability.tracing import TracingManager
from aion.systems.agents.observability.metrics import MetricsRegistry
from aion.systems.agents.observability.logging import LogAggregator, LogLevel

logger = structlog.get_logger()


class WidgetType(str, Enum):
    """Types of dashboard widgets."""

    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    LOG = "log"
    TRACE = "trace"
    STATUS = "status"
    ALERT = "alert"


@dataclass
class DashboardWidget:
    """A dashboard widget configuration."""

    id: str
    type: WidgetType
    title: str
    config: dict[str, Any] = field(default_factory=dict)
    position: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (1, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "config": self.config,
            "position": self.position,
            "size": self.size,
        }


@dataclass
class Alert:
    """An alert condition."""

    id: str
    name: str
    condition: str  # Simple expression like "error_rate > 0.1"
    severity: str  # info, warning, critical
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition,
            "severity": self.severity,
            "triggered": self.triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "message": self.message,
        }


class AgentDashboard:
    """
    Real-time agent monitoring dashboard.

    Features:
    - Customizable widgets
    - Live metrics and logs
    - Alert management
    - Agent status overview
    """

    def __init__(
        self,
        tracing: TracingManager,
        metrics: MetricsRegistry,
        logging: LogAggregator,
    ):
        self.tracing = tracing
        self.metrics = metrics
        self.logging = logging

        self._widgets: dict[str, DashboardWidget] = {}
        self._alerts: dict[str, Alert] = {}
        self._agent_status: dict[str, dict[str, Any]] = {}
        self._initialized = False

        # Create default widgets
        self._create_default_widgets()

    def _create_default_widgets(self) -> None:
        """Create default dashboard widgets."""
        default_widgets = [
            DashboardWidget(
                id="agent-status",
                type=WidgetType.STATUS,
                title="Agent Status",
                config={"show_count": True},
                position=(0, 0),
                size=(2, 1),
            ),
            DashboardWidget(
                id="tasks-counter",
                type=WidgetType.METRIC,
                title="Total Tasks",
                config={"metric": "total_tasks"},
                position=(2, 0),
                size=(1, 1),
            ),
            DashboardWidget(
                id="error-rate",
                type=WidgetType.METRIC,
                title="Error Rate",
                config={"metric": "error_rate", "format": "percent"},
                position=(3, 0),
                size=(1, 1),
            ),
            DashboardWidget(
                id="task-duration",
                type=WidgetType.CHART,
                title="Task Duration",
                config={"metric": "agent_task_duration_seconds", "type": "histogram"},
                position=(0, 1),
                size=(2, 2),
            ),
            DashboardWidget(
                id="recent-logs",
                type=WidgetType.LOG,
                title="Recent Logs",
                config={"level": "info", "limit": 20},
                position=(2, 1),
                size=(2, 2),
            ),
            DashboardWidget(
                id="recent-traces",
                type=WidgetType.TRACE,
                title="Recent Traces",
                config={"limit": 10},
                position=(0, 3),
                size=(4, 1),
            ),
            DashboardWidget(
                id="alerts",
                type=WidgetType.ALERT,
                title="Active Alerts",
                config={},
                position=(0, 4),
                size=(4, 1),
            ),
        ]

        for widget in default_widgets:
            self._widgets[widget.id] = widget

    async def initialize(self) -> None:
        """Initialize dashboard."""
        self._initialized = True
        logger.info("dashboard_initialized")

    async def shutdown(self) -> None:
        """Shutdown dashboard."""
        self._initialized = False
        logger.info("dashboard_shutdown")

    def add_widget(self, widget: DashboardWidget) -> None:
        """Add a widget to the dashboard."""
        self._widgets[widget.id] = widget

    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget."""
        if widget_id in self._widgets:
            del self._widgets[widget_id]
            return True
        return False

    def update_widget(self, widget_id: str, config: dict[str, Any]) -> bool:
        """Update widget configuration."""
        if widget_id in self._widgets:
            self._widgets[widget_id].config.update(config)
            return True
        return False

    def add_alert(self, alert: Alert) -> None:
        """Add an alert."""
        self._alerts[alert.id] = alert

    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert."""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            return True
        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.triggered = False
            alert.triggered_at = None
            return True
        return False

    def update_agent_status(
        self,
        agent_id: str,
        status: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update agent status."""
        self._agent_status[agent_id] = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "details": details or {},
        }

    def _check_alerts(self, data: dict[str, Any]) -> None:
        """Check alert conditions."""
        for alert in self._alerts.values():
            # Simple condition parsing
            try:
                triggered = self._evaluate_condition(alert.condition, data)

                if triggered and not alert.triggered:
                    alert.triggered = True
                    alert.triggered_at = datetime.now()
                    alert.message = f"Alert triggered: {alert.condition}"

                    logger.warning(
                        "alert_triggered",
                        alert_id=alert.id,
                        condition=alert.condition,
                    )

            except Exception as e:
                logger.warning("alert_evaluation_error", error=str(e))

    def _evaluate_condition(self, condition: str, data: dict[str, Any]) -> bool:
        """Evaluate a simple alert condition."""
        # Parse simple conditions like "error_rate > 0.1"
        operators = [">", "<", ">=", "<=", "==", "!="]

        for op in operators:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    threshold = float(parts[1].strip())

                    # Get metric value from data
                    value = data.get(metric_name, 0)

                    if op == ">":
                        return value > threshold
                    elif op == "<":
                        return value < threshold
                    elif op == ">=":
                        return value >= threshold
                    elif op == "<=":
                        return value <= threshold
                    elif op == "==":
                        return value == threshold
                    elif op == "!=":
                        return value != threshold

        return False

    def get_widget_data(self, widget_id: str) -> dict[str, Any]:
        """Get data for a specific widget."""
        if widget_id not in self._widgets:
            return {"error": "Widget not found"}

        widget = self._widgets[widget_id]

        if widget.type == WidgetType.STATUS:
            return self._get_status_data()
        elif widget.type == WidgetType.METRIC:
            return self._get_metric_data(widget.config)
        elif widget.type == WidgetType.CHART:
            return self._get_chart_data(widget.config)
        elif widget.type == WidgetType.LOG:
            return self._get_log_data(widget.config)
        elif widget.type == WidgetType.TRACE:
            return self._get_trace_data(widget.config)
        elif widget.type == WidgetType.ALERT:
            return self._get_alert_data()
        else:
            return {"error": f"Unknown widget type: {widget.type}"}

    def _get_status_data(self) -> dict[str, Any]:
        """Get agent status data."""
        return {
            "agents": self._agent_status,
            "count": len(self._agent_status),
            "healthy": sum(
                1 for s in self._agent_status.values()
                if s.get("status") == "healthy"
            ),
        }

    def _get_metric_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get metric data."""
        metric_name = config.get("metric", "")
        results = self.metrics.query(metric_name)

        if not results:
            return {"value": 0, "metric": metric_name}

        # Aggregate values
        total = sum(
            v.get("value", 0)
            for r in results
            for v in r.get("metric", {}).get("values", [])
        )

        return {
            "metric": metric_name,
            "value": total,
            "format": config.get("format", "number"),
        }

    def _get_chart_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get chart data."""
        metric_name = config.get("metric", "")
        chart_type = config.get("type", "line")

        results = self.metrics.query(metric_name)

        return {
            "metric": metric_name,
            "type": chart_type,
            "data": results,
        }

    def _get_log_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get log data."""
        level_str = config.get("level", "info")
        level = LogLevel(level_str)
        limit = config.get("limit", 20)

        entries = self.logging.search(level=level, limit=limit)

        return {
            "entries": [e.to_dict() for e in entries],
            "count": len(entries),
        }

    def _get_trace_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get trace data."""
        limit = config.get("limit", 10)
        traces = self.tracing.get_recent_traces(limit)

        return {
            "traces": traces,
            "count": len(traces),
        }

    def _get_alert_data(self) -> dict[str, Any]:
        """Get alert data."""
        return {
            "alerts": [a.to_dict() for a in self._alerts.values()],
            "active": sum(1 for a in self._alerts.values() if a.triggered),
        }

    def get_dashboard_state(self) -> dict[str, Any]:
        """Get complete dashboard state."""
        # Collect data for all widgets
        widget_data = {
            widget_id: self.get_widget_data(widget_id)
            for widget_id in self._widgets
        }

        # Get summary data for alert checking
        summary = self.metrics.get_summary()
        self._check_alerts(summary)

        return {
            "timestamp": datetime.now().isoformat(),
            "widgets": [w.to_dict() for w in self._widgets.values()],
            "data": widget_data,
            "summary": summary,
            "alerts": [a.to_dict() for a in self._alerts.values() if a.triggered],
        }

    def get_agent_detail(self, agent_id: str) -> dict[str, Any]:
        """Get detailed view for a specific agent."""
        collector = self.metrics.get_collector(agent_id)
        agent_logger = self.logging.get_logger(agent_id)

        return {
            "agent_id": agent_id,
            "status": self._agent_status.get(agent_id, {}),
            "metrics": collector.collect(),
            "recent_logs": [
                e.to_dict() for e in agent_logger.get_entries(limit=50)
            ],
            "traces": self.tracing.search_traces(
                service_name=agent_id,
                limit=20,
            ),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get dashboard statistics."""
        return {
            "widgets": len(self._widgets),
            "alerts_configured": len(self._alerts),
            "alerts_active": sum(1 for a in self._alerts.values() if a.triggered),
            "agents_tracked": len(self._agent_status),
            "tracing_stats": self.tracing.get_stats(),
            "metrics_summary": self.metrics.get_summary(),
            "logging_stats": self.logging.get_stats(),
        }
