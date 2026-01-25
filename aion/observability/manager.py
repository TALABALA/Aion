"""
AION Observability Manager

Central coordinator for all observability components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from aion.observability.collector import TelemetryCollector
from aion.observability.tracing.engine import TracingEngine
from aion.observability.tracing.sampling import Sampler, AlwaysOnSampler
from aion.observability.metrics.engine import MetricsEngine
from aion.observability.logging.engine import LoggingEngine
from aion.observability.alerting.engine import AlertEngine
from aion.observability.analysis.cost import CostTracker
from aion.observability.analysis.anomaly import AnomalyDetector
from aion.observability.analysis.profiler import Profiler
from aion.observability.health import HealthChecker
from aion.observability.types import LogLevel

logger = structlog.get_logger(__name__)


@dataclass
class ObservabilityConfig:
    """Configuration for the observability system."""
    service_name: str = "aion"
    service_version: str = ""
    environment: str = "development"

    # Collector settings
    collector_buffer_size: int = 10000
    collector_flush_interval: float = 5.0

    # Tracing settings
    trace_sample_rate: float = 1.0
    trace_sampler: Optional[Sampler] = None

    # Metrics settings
    metrics_aggregation_interval: float = 60.0
    metrics_max_cardinality: int = 10000

    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_debug_sample_rate: float = 0.1

    # Alerting settings
    alert_evaluation_interval: float = 60.0

    # Health check settings
    health_check_interval: float = 30.0

    # Profiling settings
    enable_cpu_profiling: bool = False
    enable_memory_tracking: bool = True
    hot_spot_threshold_ms: float = 100.0

    # Feature flags
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_alerting: bool = True
    enable_cost_tracking: bool = True
    enable_anomaly_detection: bool = True
    enable_profiling: bool = True
    enable_health_checks: bool = True


class ObservabilityManager:
    """
    Central manager for all observability components.

    Provides unified access to:
    - Metrics
    - Tracing
    - Logging
    - Alerting
    - Cost tracking
    - Anomaly detection
    - Profiling
    - Health checks
    """

    def __init__(
        self,
        config: ObservabilityConfig = None,
    ):
        self.config = config or ObservabilityConfig()

        # Initialize components
        self.collector = TelemetryCollector(
            buffer_size=self.config.collector_buffer_size,
            flush_interval=self.config.collector_flush_interval,
        )

        self.tracing: Optional[TracingEngine] = None
        self.metrics: Optional[MetricsEngine] = None
        self.logging: Optional[LoggingEngine] = None
        self.alerts: Optional[AlertEngine] = None
        self.costs: Optional[CostTracker] = None
        self.anomaly: Optional[AnomalyDetector] = None
        self.profiler: Optional[Profiler] = None
        self.health: Optional[HealthChecker] = None

        # Start time for uptime tracking
        self._start_time = datetime.utcnow()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all observability components."""
        if self._initialized:
            return

        logger.info(
            "Initializing Observability Manager",
            service=self.config.service_name,
            environment=self.config.environment,
        )

        # Initialize collector first
        await self.collector.initialize()

        # Initialize components based on config
        if self.config.enable_tracing:
            sampler = self.config.trace_sampler
            if sampler is None:
                from aion.observability.tracing.sampling import TraceIdRatioSampler
                sampler = TraceIdRatioSampler(self.config.trace_sample_rate)

            self.tracing = TracingEngine(
                collector=self.collector,
                service_name=self.config.service_name,
                sampler=sampler,
            )
            await self.tracing.initialize()

        if self.config.enable_metrics:
            self.metrics = MetricsEngine(
                collector=self.collector,
                aggregation_interval=self.config.metrics_aggregation_interval,
                max_cardinality=self.config.metrics_max_cardinality,
            )
            await self.metrics.initialize()

        if self.config.enable_logging:
            self.logging = LoggingEngine(
                collector=self.collector,
                service_name=self.config.service_name,
                default_level=self.config.log_level,
                debug_sample_rate=self.config.log_debug_sample_rate,
            )
            await self.logging.initialize()

        if self.config.enable_alerting and self.metrics:
            self.alerts = AlertEngine(
                metrics_engine=self.metrics,
                evaluation_interval=self.config.alert_evaluation_interval,
            )
            await self.alerts.initialize()
            self.alerts.add_builtin_rules()

        if self.config.enable_cost_tracking:
            self.costs = CostTracker(
                collector=self.collector,
            )
            await self.costs.initialize()

        if self.config.enable_anomaly_detection and self.metrics:
            self.anomaly = AnomalyDetector(
                metrics_engine=self.metrics,
            )
            await self.anomaly.initialize()

        if self.config.enable_profiling:
            self.profiler = Profiler(
                enable_cpu_profiling=self.config.enable_cpu_profiling,
                enable_memory_tracking=self.config.enable_memory_tracking,
                hot_spot_threshold_ms=self.config.hot_spot_threshold_ms,
            )
            await self.profiler.initialize()

        if self.config.enable_health_checks:
            self.health = HealthChecker(
                check_interval=self.config.health_check_interval,
            )
            await self.health.initialize()

        # Set global instance
        _set_global_observability(self)

        self._initialized = True
        logger.info("Observability Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown all components."""
        logger.info("Shutting down Observability Manager")

        # Shutdown in reverse order
        if self.health:
            await self.health.shutdown()
        if self.profiler:
            await self.profiler.shutdown()
        if self.anomaly:
            await self.anomaly.shutdown()
        if self.costs:
            await self.costs.shutdown()
        if self.alerts:
            await self.alerts.shutdown()
        if self.logging:
            await self.logging.shutdown()
        if self.metrics:
            await self.metrics.shutdown()
        if self.tracing:
            await self.tracing.shutdown()

        # Collector last (to flush remaining data)
        await self.collector.shutdown()

        self._initialized = False
        logger.info("Observability Manager shutdown complete")

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.utcnow() - self._start_time).total_seconds()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall observability statistics."""
        stats = {
            "service": self.config.service_name,
            "environment": self.config.environment,
            "uptime_seconds": self.uptime_seconds,
            "initialized": self._initialized,
            "collector": self.collector.get_stats() if self.collector else {},
        }

        if self.tracing:
            stats["tracing"] = self.tracing.get_stats()
        if self.metrics:
            stats["metrics"] = self.metrics.get_stats()
        if self.alerts:
            stats["alerts"] = self.alerts.get_stats()
        if self.costs:
            stats["costs"] = self.costs.get_stats()
        if self.anomaly:
            stats["anomaly"] = self.anomaly.get_stats()
        if self.profiler:
            stats["profiler"] = self.profiler.get_stats()
        if self.health:
            stats["health"] = self.health.get_stats()

        return stats

    def get_logger(self, name: str = "") -> Any:
        """Get a logger instance."""
        if self.logging:
            return self.logging.get_logger(name)
        return structlog.get_logger(name)


# Global instance
_observability: Optional[ObservabilityManager] = None


def _set_global_observability(obs: ObservabilityManager) -> None:
    """Set the global observability manager."""
    global _observability
    _observability = obs


def get_observability() -> ObservabilityManager:
    """Get the global observability manager."""
    global _observability
    if _observability is None:
        _observability = ObservabilityManager()
    return _observability


def get_tracing_engine() -> Optional[TracingEngine]:
    """Get the tracing engine."""
    obs = get_observability()
    return obs.tracing if obs else None


def get_metrics_engine() -> Optional[MetricsEngine]:
    """Get the metrics engine."""
    obs = get_observability()
    return obs.metrics if obs else None


def get_logging_engine() -> Optional[LoggingEngine]:
    """Get the logging engine."""
    obs = get_observability()
    return obs.logging if obs else None


def get_alert_engine() -> Optional[AlertEngine]:
    """Get the alert engine."""
    obs = get_observability()
    return obs.alerts if obs else None


def get_cost_tracker() -> Optional[CostTracker]:
    """Get the cost tracker."""
    obs = get_observability()
    return obs.costs if obs else None


def get_anomaly_detector() -> Optional[AnomalyDetector]:
    """Get the anomaly detector."""
    obs = get_observability()
    return obs.anomaly if obs else None


def get_profiler() -> Optional[Profiler]:
    """Get the profiler."""
    obs = get_observability()
    return obs.profiler if obs else None


def get_health_checker() -> Optional[HealthChecker]:
    """Get the health checker."""
    obs = get_observability()
    return obs.health if obs else None
