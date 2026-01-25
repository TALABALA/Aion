"""
AION Dashboard Generation

Generates Grafana dashboards for workflow observability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GrafanaPanel:
    """A Grafana dashboard panel."""
    title: str
    panel_type: str  # graph, stat, gauge, table, heatmap
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]

    description: Optional[str] = None
    unit: Optional[str] = None
    thresholds: Optional[List[Dict[str, Any]]] = None
    legend: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        panel = {
            "id": uuid.uuid4().int % 1000000,
            "title": self.title,
            "type": self.panel_type,
            "targets": self.targets,
            "gridPos": self.grid_pos,
        }

        if self.description:
            panel["description"] = self.description

        if self.unit:
            panel["fieldConfig"] = {
                "defaults": {"unit": self.unit}
            }

        if self.thresholds:
            if "fieldConfig" not in panel:
                panel["fieldConfig"] = {"defaults": {}}
            panel["fieldConfig"]["defaults"]["thresholds"] = {
                "mode": "absolute",
                "steps": self.thresholds,
            }

        if self.legend:
            panel["options"] = panel.get("options", {})
            panel["options"]["legend"] = self.legend

        if self.options:
            panel["options"] = {**panel.get("options", {}), **self.options}

        return panel


@dataclass
class GrafanaRow:
    """A row in a Grafana dashboard."""
    title: str
    panels: List[GrafanaPanel] = field(default_factory=list)
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "row",
            "title": self.title,
            "collapsed": self.collapsed,
            "panels": [p.to_dict() for p in self.panels],
        }


@dataclass
class GrafanaDashboard:
    """A complete Grafana dashboard."""
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    rows: List[GrafanaRow] = field(default_factory=list)
    panels: List[GrafanaPanel] = field(default_factory=list)

    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"

    templating: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        all_panels = [p.to_dict() for p in self.panels]

        # Add row panels
        y_offset = 0
        for row in self.rows:
            row_dict = row.to_dict()
            row_dict["gridPos"] = {"x": 0, "y": y_offset, "w": 24, "h": 1}
            all_panels.append(row_dict)
            y_offset += 1

            for panel in row.panels:
                panel_dict = panel.to_dict()
                panel_dict["gridPos"]["y"] += y_offset
                all_panels.append(panel_dict)

            if row.panels:
                max_y = max(p.grid_pos.get("y", 0) + p.grid_pos.get("h", 0) for p in row.panels)
                y_offset += max_y

        return {
            "id": None,
            "uid": str(uuid.uuid4())[:8],
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "style": "dark",
            "timezone": "browser",
            "editable": True,
            "graphTooltip": 1,
            "panels": all_panels,
            "time": {
                "from": self.time_from,
                "to": self.time_to,
            },
            "refresh": self.refresh,
            "templating": {"list": self.templating},
            "annotations": {"list": self.annotations},
            "schemaVersion": 38,
            "version": 1,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export dashboard as JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save dashboard to file."""
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Dashboard saved to {path}")


class DashboardGenerator:
    """
    Generates Grafana dashboards for workflow observability.

    Creates pre-built dashboards for:
    - Workflow overview
    - Step execution details
    - Action performance
    - Approval tracking
    - Queue monitoring
    - Error analysis
    """

    def __init__(
        self,
        datasource: str = "Prometheus",
        metric_prefix: str = "aion_automation",
    ):
        self.datasource = datasource
        self.metric_prefix = metric_prefix

    def _prom_target(
        self,
        expr: str,
        legend: str = "",
        ref_id: str = "A",
    ) -> Dict[str, Any]:
        """Create a Prometheus query target."""
        return {
            "datasource": {"type": "prometheus", "uid": self.datasource},
            "expr": expr,
            "legendFormat": legend,
            "refId": ref_id,
        }

    def generate_workflow_overview(self) -> GrafanaDashboard:
        """Generate workflow overview dashboard."""
        prefix = self.metric_prefix

        dashboard = GrafanaDashboard(
            title="AION Workflow Automation - Overview",
            description="Overview of workflow automation metrics",
            tags=["aion", "automation", "workflows"],
        )

        # Summary row
        summary_row = GrafanaRow(title="Summary Statistics")

        # Total workflows stat
        summary_row.panels.append(GrafanaPanel(
            title="Total Workflows Started",
            panel_type="stat",
            targets=[self._prom_target(
                f"sum(increase({prefix}_workflows_started_total[24h]))",
                "Started",
            )],
            grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
            options={"colorMode": "background", "graphMode": "none"},
            thresholds=[
                {"color": "green", "value": None},
            ],
        ))

        # Success rate
        summary_row.panels.append(GrafanaPanel(
            title="Success Rate (24h)",
            panel_type="gauge",
            targets=[self._prom_target(
                f"sum(increase({prefix}_workflows_completed_total[24h])) / sum(increase({prefix}_workflows_started_total[24h])) * 100",
                "Success %",
            )],
            grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
            unit="percent",
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 80},
                {"color": "green", "value": 95},
            ],
        ))

        # Failed workflows
        summary_row.panels.append(GrafanaPanel(
            title="Failed Workflows (24h)",
            panel_type="stat",
            targets=[self._prom_target(
                f"sum(increase({prefix}_workflows_failed_total[24h]))",
                "Failed",
            )],
            grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
            options={"colorMode": "background"},
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 5},
                {"color": "red", "value": 20},
            ],
        ))

        # Active workflows
        summary_row.panels.append(GrafanaPanel(
            title="Active Workflows",
            panel_type="stat",
            targets=[self._prom_target(
                f"sum({prefix}_active_workflows)",
                "Active",
            )],
            grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
            options={"colorMode": "background"},
            thresholds=[
                {"color": "green", "value": None},
            ],
        ))

        # Avg duration
        summary_row.panels.append(GrafanaPanel(
            title="Avg Workflow Duration",
            panel_type="stat",
            targets=[self._prom_target(
                f"avg(rate({prefix}_workflow_duration_seconds_sum[5m]) / rate({prefix}_workflow_duration_seconds_count[5m]))",
                "Duration",
            )],
            grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
            unit="s",
            options={"colorMode": "background"},
            thresholds=[
                {"color": "green", "value": None},
            ],
        ))

        # Pending approvals
        summary_row.panels.append(GrafanaPanel(
            title="Pending Approvals",
            panel_type="stat",
            targets=[self._prom_target(
                f"sum({prefix}_pending_approvals)",
                "Pending",
            )],
            grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
            options={"colorMode": "background"},
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 5},
                {"color": "red", "value": 10},
            ],
        ))

        dashboard.rows.append(summary_row)

        # Workflow trends row
        trends_row = GrafanaRow(title="Workflow Trends")

        # Workflows over time
        trends_row.panels.append(GrafanaPanel(
            title="Workflows Over Time",
            panel_type="graph",
            targets=[
                self._prom_target(
                    f"sum(rate({prefix}_workflows_started_total[5m])) * 60",
                    "Started/min",
                    "A",
                ),
                self._prom_target(
                    f"sum(rate({prefix}_workflows_completed_total[5m])) * 60",
                    "Completed/min",
                    "B",
                ),
                self._prom_target(
                    f"sum(rate({prefix}_workflows_failed_total[5m])) * 60",
                    "Failed/min",
                    "C",
                ),
            ],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            legend={"show": True, "placement": "bottom"},
        ))

        # Duration histogram
        trends_row.panels.append(GrafanaPanel(
            title="Workflow Duration Distribution",
            panel_type="heatmap",
            targets=[self._prom_target(
                f"sum(rate({prefix}_workflow_duration_seconds_bucket[5m])) by (le)",
                "{{le}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
        ))

        dashboard.rows.append(trends_row)

        # By workflow row
        by_workflow_row = GrafanaRow(title="By Workflow")

        # Top workflows by count
        by_workflow_row.panels.append(GrafanaPanel(
            title="Top Workflows by Execution Count",
            panel_type="table",
            targets=[self._prom_target(
                f"topk(10, sum by (workflow_name) (increase({prefix}_workflows_started_total[24h])))",
                "{{workflow_name}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        ))

        # Top workflows by failure rate
        by_workflow_row.panels.append(GrafanaPanel(
            title="Workflows by Failure Rate",
            panel_type="table",
            targets=[self._prom_target(
                f"topk(10, sum by (workflow_name) (increase({prefix}_workflows_failed_total[24h])) / sum by (workflow_name) (increase({prefix}_workflows_started_total[24h])) * 100)",
                "{{workflow_name}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="percent",
        ))

        dashboard.rows.append(by_workflow_row)

        return dashboard

    def generate_step_details(self) -> GrafanaDashboard:
        """Generate step execution details dashboard."""
        prefix = self.metric_prefix

        dashboard = GrafanaDashboard(
            title="AION Workflow Automation - Step Details",
            description="Detailed step execution metrics",
            tags=["aion", "automation", "steps"],
        )

        # Step summary row
        step_row = GrafanaRow(title="Step Execution")

        step_row.panels.append(GrafanaPanel(
            title="Steps by Status",
            panel_type="graph",
            targets=[
                self._prom_target(
                    f"sum(rate({prefix}_steps_completed_total[5m])) * 60",
                    "Completed/min",
                    "A",
                ),
                self._prom_target(
                    f"sum(rate({prefix}_steps_failed_total[5m])) * 60",
                    "Failed/min",
                    "B",
                ),
                self._prom_target(
                    f"sum(rate({prefix}_steps_retried_total[5m])) * 60",
                    "Retried/min",
                    "C",
                ),
            ],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        ))

        step_row.panels.append(GrafanaPanel(
            title="Step Duration by Type",
            panel_type="graph",
            targets=[self._prom_target(
                f"avg by (step_type) (rate({prefix}_step_duration_seconds_sum[5m]) / rate({prefix}_step_duration_seconds_count[5m]))",
                "{{step_type}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="s",
        ))

        dashboard.rows.append(step_row)

        # Retries row
        retry_row = GrafanaRow(title="Retries")

        retry_row.panels.append(GrafanaPanel(
            title="Retry Rate by Step",
            panel_type="table",
            targets=[self._prom_target(
                f"topk(10, sum by (step_id) (increase({prefix}_steps_retried_total[24h])))",
                "{{step_id}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 24, "h": 8},
        ))

        dashboard.rows.append(retry_row)

        return dashboard

    def generate_action_performance(self) -> GrafanaDashboard:
        """Generate action performance dashboard."""
        prefix = self.metric_prefix

        dashboard = GrafanaDashboard(
            title="AION Workflow Automation - Action Performance",
            description="Action execution performance metrics",
            tags=["aion", "automation", "actions"],
        )

        action_row = GrafanaRow(title="Action Performance")

        action_row.panels.append(GrafanaPanel(
            title="Actions by Type",
            panel_type="graph",
            targets=[self._prom_target(
                f"sum by (action_type) (rate({prefix}_actions_executed_total[5m])) * 60",
                "{{action_type}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        ))

        action_row.panels.append(GrafanaPanel(
            title="Action Duration by Type",
            panel_type="graph",
            targets=[self._prom_target(
                f"avg by (action_type) (rate({prefix}_action_duration_seconds_sum[5m]) / rate({prefix}_action_duration_seconds_count[5m]))",
                "{{action_type}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="s",
        ))

        dashboard.rows.append(action_row)

        return dashboard

    def generate_approval_tracking(self) -> GrafanaDashboard:
        """Generate approval tracking dashboard."""
        prefix = self.metric_prefix

        dashboard = GrafanaDashboard(
            title="AION Workflow Automation - Approvals",
            description="Approval workflow tracking",
            tags=["aion", "automation", "approvals"],
        )

        approval_row = GrafanaRow(title="Approval Metrics")

        approval_row.panels.append(GrafanaPanel(
            title="Approval Status",
            panel_type="graph",
            targets=[
                self._prom_target(
                    f"sum(rate({prefix}_approvals_requested_total[5m])) * 60",
                    "Requested/min",
                    "A",
                ),
                self._prom_target(
                    f"sum(rate({prefix}_approvals_granted_total[5m])) * 60",
                    "Granted/min",
                    "B",
                ),
                self._prom_target(
                    f"sum(rate({prefix}_approvals_denied_total[5m])) * 60",
                    "Denied/min",
                    "C",
                ),
            ],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        ))

        approval_row.panels.append(GrafanaPanel(
            title="Approval Wait Time",
            panel_type="graph",
            targets=[self._prom_target(
                f"histogram_quantile(0.95, sum(rate({prefix}_approval_wait_time_seconds_bucket[5m])) by (le))",
                "p95 Wait Time",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            unit="s",
        ))

        dashboard.rows.append(approval_row)

        return dashboard

    def generate_error_analysis(self) -> GrafanaDashboard:
        """Generate error analysis dashboard."""
        prefix = self.metric_prefix

        dashboard = GrafanaDashboard(
            title="AION Workflow Automation - Errors",
            description="Error analysis and debugging",
            tags=["aion", "automation", "errors"],
        )

        error_row = GrafanaRow(title="Error Analysis")

        error_row.panels.append(GrafanaPanel(
            title="Errors by Type",
            panel_type="graph",
            targets=[self._prom_target(
                f"sum by (error_type) (rate({prefix}_workflows_failed_total[5m])) * 60",
                "{{error_type}}",
            )],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        ))

        error_row.panels.append(GrafanaPanel(
            title="Top Failing Workflows",
            panel_type="table",
            targets=[self._prom_target(
                f"topk(10, sum by (workflow_name) (increase({prefix}_workflows_failed_total[24h])))",
                "{{workflow_name}}",
            )],
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
        ))

        dashboard.rows.append(error_row)

        return dashboard

    def generate_all_dashboards(self, output_dir: str = ".") -> List[str]:
        """Generate all dashboards and save to files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        dashboards = [
            ("workflow_overview", self.generate_workflow_overview()),
            ("step_details", self.generate_step_details()),
            ("action_performance", self.generate_action_performance()),
            ("approval_tracking", self.generate_approval_tracking()),
            ("error_analysis", self.generate_error_analysis()),
        ]

        paths = []
        for name, dashboard in dashboards:
            path = os.path.join(output_dir, f"{name}.json")
            dashboard.save(path)
            paths.append(path)

        logger.info(f"Generated {len(paths)} dashboards")
        return paths
