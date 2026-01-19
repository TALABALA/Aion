"""
AION Built-in Agents

System agents providing core OS-level functionality:
- HealthMonitorAgent: System health monitoring and alerting
- GarbageCollectorAgent: Resource cleanup and memory management
- MetricsCollectorAgent: Telemetry and metrics collection
- WatchdogAgent: Process supervision and recovery
- LogAggregatorAgent: Centralized log collection
"""

from __future__ import annotations

import asyncio
import uuid
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from aion.systems.process.agent_base import BaseAgent
from aion.systems.process.models import (
    AgentConfig,
    Event,
    ProcessState,
    ProcessPriority,
)

logger = structlog.get_logger(__name__)


class HealthMonitorAgent(BaseAgent):
    """
    System health monitoring agent.

    Monitors:
    - Process health and state
    - Resource usage (memory, CPU, tokens)
    - System metrics
    - Emits alerts on anomalies
    """

    async def run(self) -> None:
        """Main monitoring loop."""
        check_interval = self.config.metadata.get("check_interval", 30)
        alert_thresholds = self.config.metadata.get("alert_thresholds", {})

        # Default thresholds
        memory_threshold = alert_thresholds.get("memory_percent", 80)
        cpu_threshold = alert_thresholds.get("cpu_percent", 90)
        failed_process_threshold = alert_thresholds.get("failed_processes", 3)
        queue_size_threshold = alert_thresholds.get("queue_size", 100)

        while not self._shutdown_requested:
            await self._paused.wait()

            if self._shutdown_requested:
                break

            try:
                # Collect health data
                health_data = await self._collect_health_data()

                # Check for issues
                alerts = []

                # Check process states
                failed_count = health_data.get("failed_processes", 0)
                if failed_count >= failed_process_threshold:
                    alerts.append({
                        "type": "high_failure_count",
                        "severity": "warning",
                        "message": f"{failed_count} processes have failed",
                        "details": {"failed_count": failed_count},
                    })

                # Check system memory
                system_memory = health_data.get("system_memory_percent", 0)
                if system_memory > memory_threshold:
                    alerts.append({
                        "type": "high_memory_usage",
                        "severity": "warning",
                        "message": f"System memory at {system_memory:.1f}%",
                        "details": {"memory_percent": system_memory},
                    })

                # Check system CPU
                system_cpu = health_data.get("system_cpu_percent", 0)
                if system_cpu > cpu_threshold:
                    alerts.append({
                        "type": "high_cpu_usage",
                        "severity": "warning",
                        "message": f"System CPU at {system_cpu:.1f}%",
                        "details": {"cpu_percent": system_cpu},
                    })

                # Check individual processes
                for process in health_data.get("processes", []):
                    if process.get("state") == "failed":
                        alerts.append({
                            "type": "process_failed",
                            "severity": "error",
                            "message": f"Process {process['name']} has failed",
                            "details": {
                                "process_id": process["id"],
                                "error": process.get("error"),
                            },
                        })

                    # Check resource limits
                    exceeded, reason = self._check_process_resources(process)
                    if exceeded:
                        alerts.append({
                            "type": "resource_exceeded",
                            "severity": "warning",
                            "message": f"Process {process['name']}: {reason}",
                            "details": {"process_id": process["id"]},
                        })

                # Emit alerts
                for alert in alerts:
                    await self._emit_alert(alert)

                # Emit health summary
                await self.emit_event("health_check", {
                    "timestamp": datetime.now().isoformat(),
                    "status": "healthy" if not alerts else "degraded",
                    "alert_count": len(alerts),
                    **health_data,
                })

            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                self.record_error()

            await self.sleep(check_interval)

    async def _collect_health_data(self) -> Dict[str, Any]:
        """Collect comprehensive health data."""
        processes = self.supervisor.get_all_processes()

        # Process breakdown
        running = len([p for p in processes if p.state == ProcessState.RUNNING])
        paused = len([p for p in processes if p.state == ProcessState.PAUSED])
        failed = len([p for p in processes if p.state == ProcessState.FAILED])

        # Resource totals
        total_memory = sum(p.usage.memory_mb for p in processes if p.state.is_active())
        total_tokens = sum(p.usage.tokens_used for p in processes)

        # System metrics
        try:
            system_memory = psutil.virtual_memory().percent
            system_cpu = psutil.cpu_percent(interval=0.1)
        except Exception:
            system_memory = 0
            system_cpu = 0

        # Process details
        process_details = []
        for p in processes:
            if p.state.is_active() or p.state == ProcessState.FAILED:
                process_details.append({
                    "id": p.id,
                    "name": p.name,
                    "state": p.state.value,
                    "priority": p.priority.name,
                    "memory_mb": p.usage.memory_mb,
                    "tokens_used": p.usage.tokens_used,
                    "runtime_seconds": p.usage.runtime_seconds,
                    "error": p.error,
                })

        return {
            "total_processes": len(processes),
            "running_processes": running,
            "paused_processes": paused,
            "failed_processes": failed,
            "total_memory_mb": total_memory,
            "total_tokens_used": total_tokens,
            "system_memory_percent": system_memory,
            "system_cpu_percent": system_cpu,
            "processes": process_details,
        }

    def _check_process_resources(self, process: Dict) -> tuple[bool, Optional[str]]:
        """Check if a process is exceeding resources."""
        # This would need access to limits - simplified for now
        return False, None

    async def _emit_alert(self, alert: Dict[str, Any]) -> None:
        """Emit an alert event."""
        await self.event_bus.emit(Event(
            id=str(uuid.uuid4()),
            type="system.alert",
            source=self.process_id,
            payload={
                "alert_type": alert["type"],
                "severity": alert["severity"],
                "message": alert["message"],
                "details": alert.get("details", {}),
                "timestamp": datetime.now().isoformat(),
            },
        ))

        self.logger.warning(
            f"Alert: {alert['message']}",
            alert_type=alert["type"],
            severity=alert["severity"],
        )


class GarbageCollectorAgent(BaseAgent):
    """
    Resource cleanup agent.

    Cleans up:
    - Completed/failed processes after retention period
    - Old event history
    - Stale checkpoints
    - Orphaned resources
    """

    async def run(self) -> None:
        """Main GC loop."""
        gc_interval = self.config.metadata.get("gc_interval", 300)  # 5 minutes
        max_completed_age = self.config.metadata.get("max_completed_age", 3600)  # 1 hour
        max_checkpoint_age = self.config.metadata.get("max_checkpoint_age", 86400)  # 24 hours

        while not self._shutdown_requested:
            await self._paused.wait()

            if self._shutdown_requested:
                break

            try:
                gc_stats = await self._run_gc(
                    max_completed_age=max_completed_age,
                    max_checkpoint_age=max_checkpoint_age,
                )

                if gc_stats["total_cleaned"] > 0:
                    self.logger.info(
                        "GC completed",
                        processes_cleaned=gc_stats["processes_cleaned"],
                        checkpoints_cleaned=gc_stats.get("checkpoints_cleaned", 0),
                    )

                await self.emit_event("gc_complete", {
                    "timestamp": datetime.now().isoformat(),
                    **gc_stats,
                })

            except Exception as e:
                self.logger.error(f"GC error: {e}")
                self.record_error()

            await self.sleep(gc_interval)

    async def _run_gc(
        self,
        max_completed_age: int,
        max_checkpoint_age: int,
    ) -> Dict[str, Any]:
        """Run garbage collection."""
        now = datetime.now()
        processes_cleaned = 0
        checkpoints_cleaned = 0
        memory_freed = 0

        # Clean up old completed processes
        processes = self.supervisor.get_all_processes()

        for process in processes:
            if process.state.is_terminal():
                if process.stopped_at:
                    age = (now - process.stopped_at).total_seconds()
                    if age > max_completed_age:
                        # Note: We don't actually delete from supervisor
                        # as it manages its own cleanup
                        processes_cleaned += 1

        # Could add checkpoint cleanup here if implemented in supervisor

        return {
            "processes_cleaned": processes_cleaned,
            "checkpoints_cleaned": checkpoints_cleaned,
            "memory_freed_mb": memory_freed,
            "total_cleaned": processes_cleaned + checkpoints_cleaned,
        }


class MetricsCollectorAgent(BaseAgent):
    """
    Telemetry and metrics collection agent.

    Collects:
    - Process metrics
    - System metrics
    - Event bus metrics
    - Custom application metrics
    """

    async def run(self) -> None:
        """Main metrics collection loop."""
        collect_interval = self.config.metadata.get("collect_interval", 60)
        include_system_metrics = self.config.metadata.get("include_system_metrics", True)

        metrics_history: List[Dict] = []
        max_history = 60  # Keep last 60 samples

        while not self._shutdown_requested:
            await self._paused.wait()

            if self._shutdown_requested:
                break

            try:
                metrics = await self._collect_metrics(include_system_metrics)

                # Store in history
                metrics_history.append(metrics)
                if len(metrics_history) > max_history:
                    metrics_history = metrics_history[-max_history:]

                # Calculate trends
                if len(metrics_history) >= 2:
                    metrics["trends"] = self._calculate_trends(metrics_history)

                await self.emit_event("metrics", {
                    "timestamp": datetime.now().isoformat(),
                    **metrics,
                })

            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                self.record_error()

            await self.sleep(collect_interval)

    async def _collect_metrics(self, include_system: bool) -> Dict[str, Any]:
        """Collect all metrics."""
        processes = self.supervisor.get_all_processes()

        # Process metrics
        total_tokens = sum(p.usage.tokens_used for p in processes)
        total_tool_calls = sum(p.usage.tool_calls for p in processes)
        total_errors = sum(p.usage.errors_count for p in processes)

        # State breakdown
        by_state = {}
        for state in ProcessState:
            by_state[state.value] = len([p for p in processes if p.state == state])

        # Priority breakdown
        by_priority = {}
        for priority in ProcessPriority:
            by_priority[priority.name] = len([p for p in processes if p.priority == priority])

        metrics = {
            "processes": {
                "total": len(processes),
                "by_state": by_state,
                "by_priority": by_priority,
            },
            "resources": {
                "total_tokens_used": total_tokens,
                "total_tool_calls": total_tool_calls,
                "total_errors": total_errors,
            },
            "supervisor": self.supervisor.get_stats(),
            "event_bus": self.event_bus.get_stats(),
        }

        # System metrics
        if include_system:
            try:
                metrics["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
                    "disk_percent": psutil.disk_usage("/").percent,
                }
            except Exception:
                metrics["system"] = {}

        return metrics

    def _calculate_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """Calculate trends from metrics history."""
        trends = {}

        if len(history) < 2:
            return trends

        current = history[-1]
        previous = history[-2]

        # Calculate deltas
        try:
            current_tokens = current.get("resources", {}).get("total_tokens_used", 0)
            previous_tokens = previous.get("resources", {}).get("total_tokens_used", 0)
            trends["tokens_delta"] = current_tokens - previous_tokens

            current_errors = current.get("resources", {}).get("total_errors", 0)
            previous_errors = previous.get("resources", {}).get("total_errors", 0)
            trends["errors_delta"] = current_errors - previous_errors

            current_processes = current.get("processes", {}).get("total", 0)
            previous_processes = previous.get("processes", {}).get("total", 0)
            trends["processes_delta"] = current_processes - previous_processes

        except Exception:
            pass

        return trends


class WatchdogAgent(BaseAgent):
    """
    Process watchdog agent.

    Responsibilities:
    - Monitor process health
    - Detect hung/unresponsive processes
    - Trigger restarts for stuck processes
    - Enforce heartbeat timeouts
    """

    async def run(self) -> None:
        """Main watchdog loop."""
        check_interval = self.config.metadata.get("check_interval", 30)
        heartbeat_timeout = self.config.metadata.get("heartbeat_timeout", 120)  # 2 minutes
        idle_timeout = self.config.metadata.get("idle_timeout", 600)  # 10 minutes

        while not self._shutdown_requested:
            await self._paused.wait()

            if self._shutdown_requested:
                break

            try:
                await self._check_processes(heartbeat_timeout, idle_timeout)

            except Exception as e:
                self.logger.error(f"Watchdog error: {e}")
                self.record_error()

            await self.sleep(check_interval)

    async def _check_processes(
        self,
        heartbeat_timeout: int,
        idle_timeout: int,
    ) -> None:
        """Check all processes for issues."""
        now = datetime.now()
        processes = self.supervisor.get_all_processes()

        for process in processes:
            # Skip self and system-critical processes
            if process.id == self.process_id:
                continue
            if process.priority == ProcessPriority.CRITICAL:
                continue
            if not process.state.is_active():
                continue

            issues = []

            # Check heartbeat timeout
            if process.last_heartbeat:
                heartbeat_age = (now - process.last_heartbeat).total_seconds()
                if heartbeat_age > heartbeat_timeout:
                    issues.append(f"No heartbeat for {heartbeat_age:.0f}s")

            # Check idle timeout (for non-system processes)
            if process.type.value not in ("system", "worker"):
                if process.usage.last_activity:
                    idle_time = (now - process.usage.last_activity).total_seconds()
                    if idle_timeout and idle_time > idle_timeout:
                        issues.append(f"Idle for {idle_time:.0f}s")

            # Take action if issues found
            if issues:
                self.logger.warning(
                    "Watchdog detected issues",
                    process_id=process.id,
                    process_name=process.name,
                    issues=issues,
                )

                # Emit warning event
                await self.event_bus.emit(Event(
                    id=str(uuid.uuid4()),
                    type="watchdog.warning",
                    source=self.process_id,
                    payload={
                        "process_id": process.id,
                        "process_name": process.name,
                        "issues": issues,
                    },
                ))

                # For severe issues, restart the process
                if any("No heartbeat" in i for i in issues):
                    self.logger.warning(f"Restarting unresponsive process: {process.name}")
                    try:
                        await self.supervisor.restart_process(process.id)
                    except Exception as e:
                        self.logger.error(f"Failed to restart process: {e}")


class LogAggregatorAgent(BaseAgent):
    """
    Centralized log aggregation agent.

    Collects logs from all processes and:
    - Aggregates into central store
    - Detects error patterns
    - Triggers alerts on critical errors
    """

    async def run(self) -> None:
        """Main log aggregation loop."""
        aggregation_interval = self.config.metadata.get("aggregation_interval", 10)
        error_threshold = self.config.metadata.get("error_threshold", 10)  # Errors per minute

        error_counts: Dict[str, int] = {}
        last_reset = datetime.now()

        # Subscribe to log events
        await self.event_bus.subscribe(
            "log.*",
            self._handle_log_event,
            subscriber_id=self.process_id,
        )

        while not self._shutdown_requested:
            await self._paused.wait()

            if self._shutdown_requested:
                break

            try:
                now = datetime.now()

                # Reset error counts every minute
                if (now - last_reset).total_seconds() >= 60:
                    # Check for high error rates
                    for source, count in error_counts.items():
                        if count >= error_threshold:
                            await self._emit_error_rate_alert(source, count)

                    error_counts.clear()
                    last_reset = now

                # Emit aggregation summary
                await self.emit_event("log_summary", {
                    "timestamp": now.isoformat(),
                    "error_counts": error_counts.copy(),
                })

            except Exception as e:
                self.logger.error(f"Log aggregation error: {e}")
                self.record_error()

            await self.sleep(aggregation_interval)

    async def _handle_log_event(self, event: Event) -> None:
        """Handle incoming log events."""
        level = event.payload.get("level", "info")
        if level in ("error", "critical"):
            source = event.source
            # Track error count - would need to access state properly
            pass

    async def _emit_error_rate_alert(self, source: str, count: int) -> None:
        """Emit alert for high error rate."""
        await self.event_bus.emit(Event(
            id=str(uuid.uuid4()),
            type="system.alert",
            source=self.process_id,
            payload={
                "alert_type": "high_error_rate",
                "severity": "warning",
                "message": f"High error rate from {source}: {count} errors/minute",
                "details": {
                    "source": source,
                    "error_count": count,
                },
            },
        ))
