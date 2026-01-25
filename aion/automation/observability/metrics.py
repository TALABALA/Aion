"""
AION Workflow Metrics

Comprehensive metrics collection for workflow automation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.automation.observability.telemetry import get_meter

logger = structlog.get_logger(__name__)


@dataclass
class MetricNames:
    """Standard metric names for workflows."""
    # Counters
    WORKFLOWS_STARTED = "workflows_started_total"
    WORKFLOWS_COMPLETED = "workflows_completed_total"
    WORKFLOWS_FAILED = "workflows_failed_total"
    WORKFLOWS_CANCELLED = "workflows_cancelled_total"

    STEPS_STARTED = "steps_started_total"
    STEPS_COMPLETED = "steps_completed_total"
    STEPS_FAILED = "steps_failed_total"
    STEPS_SKIPPED = "steps_skipped_total"
    STEPS_RETRIED = "steps_retried_total"

    ACTIONS_EXECUTED = "actions_executed_total"
    ACTIONS_FAILED = "actions_failed_total"

    TRIGGERS_FIRED = "triggers_fired_total"

    APPROVALS_REQUESTED = "approvals_requested_total"
    APPROVALS_GRANTED = "approvals_granted_total"
    APPROVALS_DENIED = "approvals_denied_total"
    APPROVALS_TIMEOUT = "approvals_timeout_total"

    # Histograms
    WORKFLOW_DURATION = "workflow_duration_seconds"
    STEP_DURATION = "step_duration_seconds"
    ACTION_DURATION = "action_duration_seconds"
    APPROVAL_WAIT_TIME = "approval_wait_time_seconds"

    # Gauges
    ACTIVE_WORKFLOWS = "active_workflows"
    ACTIVE_STEPS = "active_steps"
    PENDING_APPROVALS = "pending_approvals"
    QUEUE_DEPTH = "queue_depth"
    WORKERS_ACTIVE = "workers_active"


class MetricRegistry:
    """
    Registry for all workflow metrics.

    Provides a central place to access all metrics.
    """

    def __init__(self, prefix: str = "aion_automation"):
        self.prefix = prefix
        self._meter = get_meter()
        self._metrics: Dict[str, Any] = {}

        self._initialize_metrics()

    def _prefixed(self, name: str) -> str:
        """Add prefix to metric name."""
        return f"{self.prefix}_{name}"

    def _initialize_metrics(self) -> None:
        """Initialize all standard metrics."""
        meter = self._meter

        # Workflow counters
        self._metrics["workflows_started"] = meter.create_counter(
            self._prefixed(MetricNames.WORKFLOWS_STARTED),
            description="Total number of workflows started",
            unit="1",
        )
        self._metrics["workflows_completed"] = meter.create_counter(
            self._prefixed(MetricNames.WORKFLOWS_COMPLETED),
            description="Total number of workflows completed successfully",
            unit="1",
        )
        self._metrics["workflows_failed"] = meter.create_counter(
            self._prefixed(MetricNames.WORKFLOWS_FAILED),
            description="Total number of workflows failed",
            unit="1",
        )
        self._metrics["workflows_cancelled"] = meter.create_counter(
            self._prefixed(MetricNames.WORKFLOWS_CANCELLED),
            description="Total number of workflows cancelled",
            unit="1",
        )

        # Step counters
        self._metrics["steps_started"] = meter.create_counter(
            self._prefixed(MetricNames.STEPS_STARTED),
            description="Total number of steps started",
            unit="1",
        )
        self._metrics["steps_completed"] = meter.create_counter(
            self._prefixed(MetricNames.STEPS_COMPLETED),
            description="Total number of steps completed",
            unit="1",
        )
        self._metrics["steps_failed"] = meter.create_counter(
            self._prefixed(MetricNames.STEPS_FAILED),
            description="Total number of steps failed",
            unit="1",
        )
        self._metrics["steps_retried"] = meter.create_counter(
            self._prefixed(MetricNames.STEPS_RETRIED),
            description="Total number of step retries",
            unit="1",
        )

        # Action counters
        self._metrics["actions_executed"] = meter.create_counter(
            self._prefixed(MetricNames.ACTIONS_EXECUTED),
            description="Total number of actions executed",
            unit="1",
        )
        self._metrics["actions_failed"] = meter.create_counter(
            self._prefixed(MetricNames.ACTIONS_FAILED),
            description="Total number of actions failed",
            unit="1",
        )

        # Trigger counters
        self._metrics["triggers_fired"] = meter.create_counter(
            self._prefixed(MetricNames.TRIGGERS_FIRED),
            description="Total number of triggers fired",
            unit="1",
        )

        # Approval counters
        self._metrics["approvals_requested"] = meter.create_counter(
            self._prefixed(MetricNames.APPROVALS_REQUESTED),
            description="Total number of approvals requested",
            unit="1",
        )
        self._metrics["approvals_granted"] = meter.create_counter(
            self._prefixed(MetricNames.APPROVALS_GRANTED),
            description="Total number of approvals granted",
            unit="1",
        )
        self._metrics["approvals_denied"] = meter.create_counter(
            self._prefixed(MetricNames.APPROVALS_DENIED),
            description="Total number of approvals denied",
            unit="1",
        )

        # Histograms
        self._metrics["workflow_duration"] = meter.create_histogram(
            self._prefixed(MetricNames.WORKFLOW_DURATION),
            description="Workflow execution duration in seconds",
            unit="s",
        )
        self._metrics["step_duration"] = meter.create_histogram(
            self._prefixed(MetricNames.STEP_DURATION),
            description="Step execution duration in seconds",
            unit="s",
        )
        self._metrics["action_duration"] = meter.create_histogram(
            self._prefixed(MetricNames.ACTION_DURATION),
            description="Action execution duration in seconds",
            unit="s",
        )
        self._metrics["approval_wait_time"] = meter.create_histogram(
            self._prefixed(MetricNames.APPROVAL_WAIT_TIME),
            description="Time waiting for approval in seconds",
            unit="s",
        )

        # Gauges (using up-down counters for simplicity)
        self._metrics["active_workflows"] = meter.create_up_down_counter(
            self._prefixed(MetricNames.ACTIVE_WORKFLOWS),
            description="Number of currently active workflows",
            unit="1",
        )
        self._metrics["pending_approvals"] = meter.create_up_down_counter(
            self._prefixed(MetricNames.PENDING_APPROVALS),
            description="Number of pending approval requests",
            unit="1",
        )

    def get_metric(self, name: str):
        """Get a metric by name."""
        return self._metrics.get(name)


class WorkflowMetrics:
    """
    High-level metrics API for workflow operations.

    Provides convenient methods for recording workflow metrics.
    """

    def __init__(self, registry: Optional[MetricRegistry] = None):
        self.registry = registry or MetricRegistry()

    # Workflow metrics

    def workflow_started(
        self,
        workflow_id: str,
        workflow_name: str,
        trigger_type: Optional[str] = None,
    ) -> None:
        """Record workflow started."""
        attributes = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
        }
        if trigger_type:
            attributes["trigger_type"] = trigger_type

        self.registry.get_metric("workflows_started").add(1, attributes)
        self.registry.get_metric("active_workflows").add(1, attributes)

    def workflow_completed(
        self,
        workflow_id: str,
        workflow_name: str,
        duration_seconds: float,
    ) -> None:
        """Record workflow completed."""
        attributes = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
        }

        self.registry.get_metric("workflows_completed").add(1, attributes)
        self.registry.get_metric("active_workflows").add(-1, attributes)
        self.registry.get_metric("workflow_duration").record(duration_seconds, attributes)

    def workflow_failed(
        self,
        workflow_id: str,
        workflow_name: str,
        error_type: str,
        duration_seconds: float,
    ) -> None:
        """Record workflow failed."""
        attributes = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "error_type": error_type,
        }

        self.registry.get_metric("workflows_failed").add(1, attributes)
        self.registry.get_metric("active_workflows").add(-1, {"workflow_id": workflow_id, "workflow_name": workflow_name})
        self.registry.get_metric("workflow_duration").record(duration_seconds, attributes)

    def workflow_cancelled(
        self,
        workflow_id: str,
        workflow_name: str,
    ) -> None:
        """Record workflow cancelled."""
        attributes = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
        }

        self.registry.get_metric("workflows_cancelled").add(1, attributes)
        self.registry.get_metric("active_workflows").add(-1, attributes)

    # Step metrics

    def step_started(
        self,
        workflow_id: str,
        step_id: str,
        step_type: str,
    ) -> None:
        """Record step started."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
            "step_type": step_type,
        }

        self.registry.get_metric("steps_started").add(1, attributes)

    def step_completed(
        self,
        workflow_id: str,
        step_id: str,
        step_type: str,
        duration_seconds: float,
    ) -> None:
        """Record step completed."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
            "step_type": step_type,
        }

        self.registry.get_metric("steps_completed").add(1, attributes)
        self.registry.get_metric("step_duration").record(duration_seconds, attributes)

    def step_failed(
        self,
        workflow_id: str,
        step_id: str,
        step_type: str,
        error_type: str,
        duration_seconds: float,
    ) -> None:
        """Record step failed."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
            "step_type": step_type,
            "error_type": error_type,
        }

        self.registry.get_metric("steps_failed").add(1, attributes)
        self.registry.get_metric("step_duration").record(duration_seconds, attributes)

    def step_retried(
        self,
        workflow_id: str,
        step_id: str,
        retry_count: int,
    ) -> None:
        """Record step retry."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
            "retry_count": str(retry_count),
        }

        self.registry.get_metric("steps_retried").add(1, attributes)

    # Action metrics

    def action_executed(
        self,
        action_type: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record action execution."""
        attributes = {
            "action_type": action_type,
            "success": str(success).lower(),
        }

        self.registry.get_metric("actions_executed").add(1, attributes)
        self.registry.get_metric("action_duration").record(duration_seconds, attributes)

        if not success:
            self.registry.get_metric("actions_failed").add(1, {"action_type": action_type})

    # Trigger metrics

    def trigger_fired(
        self,
        trigger_type: str,
        workflow_id: str,
    ) -> None:
        """Record trigger fired."""
        attributes = {
            "trigger_type": trigger_type,
            "workflow_id": workflow_id,
        }

        self.registry.get_metric("triggers_fired").add(1, attributes)

    # Approval metrics

    def approval_requested(
        self,
        workflow_id: str,
        step_id: str,
        gate_type: str,
    ) -> None:
        """Record approval requested."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
            "gate_type": gate_type,
        }

        self.registry.get_metric("approvals_requested").add(1, attributes)
        self.registry.get_metric("pending_approvals").add(1, attributes)

    def approval_granted(
        self,
        workflow_id: str,
        step_id: str,
        wait_time_seconds: float,
    ) -> None:
        """Record approval granted."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
        }

        self.registry.get_metric("approvals_granted").add(1, attributes)
        self.registry.get_metric("pending_approvals").add(-1, attributes)
        self.registry.get_metric("approval_wait_time").record(wait_time_seconds, attributes)

    def approval_denied(
        self,
        workflow_id: str,
        step_id: str,
        wait_time_seconds: float,
    ) -> None:
        """Record approval denied."""
        attributes = {
            "workflow_id": workflow_id,
            "step_id": step_id,
        }

        self.registry.get_metric("approvals_denied").add(1, attributes)
        self.registry.get_metric("pending_approvals").add(-1, attributes)
        self.registry.get_metric("approval_wait_time").record(wait_time_seconds, attributes)


class MetricsTimer:
    """Context manager for timing operations and recording duration metrics."""

    def __init__(
        self,
        metrics: WorkflowMetrics,
        metric_type: str,
        **attributes,
    ):
        self.metrics = metrics
        self.metric_type = metric_type
        self.attributes = attributes
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if self.metric_type == "workflow":
            if exc_type:
                self.metrics.workflow_failed(
                    self.attributes.get("workflow_id", "unknown"),
                    self.attributes.get("workflow_name", "unknown"),
                    exc_type.__name__,
                    duration,
                )
            else:
                self.metrics.workflow_completed(
                    self.attributes.get("workflow_id", "unknown"),
                    self.attributes.get("workflow_name", "unknown"),
                    duration,
                )

        elif self.metric_type == "step":
            if exc_type:
                self.metrics.step_failed(
                    self.attributes.get("workflow_id", "unknown"),
                    self.attributes.get("step_id", "unknown"),
                    self.attributes.get("step_type", "unknown"),
                    exc_type.__name__,
                    duration,
                )
            else:
                self.metrics.step_completed(
                    self.attributes.get("workflow_id", "unknown"),
                    self.attributes.get("step_id", "unknown"),
                    self.attributes.get("step_type", "unknown"),
                    duration,
                )

        elif self.metric_type == "action":
            self.metrics.action_executed(
                self.attributes.get("action_type", "unknown"),
                duration,
                success=exc_type is None,
            )

        return False  # Don't suppress exceptions


# Global metrics instance
_workflow_metrics: Optional[WorkflowMetrics] = None


def get_workflow_metrics() -> WorkflowMetrics:
    """Get the global workflow metrics instance."""
    global _workflow_metrics
    if _workflow_metrics is None:
        _workflow_metrics = WorkflowMetrics()
    return _workflow_metrics
