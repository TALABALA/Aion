"""
Chaos Engineering and Resilience Scoring.

Provides:
- Fault injection correlation with observability
- Resilience scoring based on system behavior
- Game day automation
- Steady state verification
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of faults for injection."""
    LATENCY = "latency"
    ERROR = "error"
    TIMEOUT = "timeout"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    NETWORK_PARTITION = "network_partition"
    PACKET_LOSS = "packet_loss"
    SERVICE_KILL = "service_kill"
    DNS_FAILURE = "dns_failure"
    CLOCK_SKEW = "clock_skew"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ExperimentStatus(Enum):
    """Status of a chaos experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class FaultConfig:
    """Configuration for a fault injection."""
    fault_type: FaultType
    target: str  # Service, host, container
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: timedelta = timedelta(minutes=5)
    probability: float = 1.0  # For probabilistic faults


@dataclass
class SteadyState:
    """Definition of steady state for verification."""
    name: str
    metric: str
    condition: str  # e.g., "< 100", "> 0.99"
    tolerance: float = 0.1


@dataclass
class FaultExperiment:
    """A chaos engineering experiment."""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    faults: List[FaultConfig]
    steady_states: List[SteadyState]
    duration: timedelta
    abort_conditions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""
    experiment_id: str
    status: ExperimentStatus
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    steady_state_before: Dict[str, bool] = field(default_factory=dict)
    steady_state_during: Dict[str, bool] = field(default_factory=dict)
    steady_state_after: Dict[str, bool] = field(default_factory=dict)
    hypothesis_validated: bool = False
    metrics_collected: Dict[str, List[float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)


@dataclass
class ResilienceMetric:
    """A metric contributing to resilience score."""
    name: str
    value: float
    weight: float = 1.0
    threshold: float = 0.9
    description: str = ""


@dataclass
class ResilienceScore:
    """Overall resilience score for a system."""
    score: float  # 0-100
    grade: str  # A, B, C, D, F
    metrics: List[ResilienceMetric]
    timestamp: datetime
    period: timedelta
    recommendations: List[str] = field(default_factory=list)


class FaultInjector:
    """
    Fault injection engine with observability integration.

    Injects faults and correlates system behavior with observability data.
    """

    def __init__(self):
        self._active_faults: Dict[str, FaultConfig] = {}
        self._experiment_history: List[Tuple[FaultExperiment, ExperimentResult]] = []
        self._observers: List['ChaosObserver'] = []

    def add_observer(self, observer: 'ChaosObserver'):
        """Add an observer for chaos events."""
        self._observers.append(observer)

    def _notify_observers(self, event_type: str, data: Dict[str, Any]):
        """Notify observers of chaos events."""
        for observer in self._observers:
            try:
                observer.on_chaos_event(event_type, data)
            except Exception as e:
                logger.error(f"Observer error: {e}")

    async def inject_fault(self, config: FaultConfig) -> str:
        """Inject a fault and return fault ID."""
        fault_id = str(uuid.uuid4())[:8]

        logger.info(f"Injecting fault {fault_id}: {config.fault_type.value} on {config.target}")
        self._notify_observers("fault_injected", {
            "fault_id": fault_id,
            "fault_type": config.fault_type.value,
            "target": config.target,
            "parameters": config.parameters
        })

        self._active_faults[fault_id] = config

        # Simulate fault injection
        if config.fault_type == FaultType.LATENCY:
            latency_ms = config.parameters.get("latency_ms", 100)
            logger.info(f"Adding {latency_ms}ms latency to {config.target}")

        elif config.fault_type == FaultType.ERROR:
            error_rate = config.parameters.get("error_rate", 0.1)
            logger.info(f"Injecting {error_rate*100}% error rate to {config.target}")

        elif config.fault_type == FaultType.CPU_STRESS:
            load = config.parameters.get("load_percent", 80)
            logger.info(f"Stressing CPU to {load}% on {config.target}")

        return fault_id

    async def remove_fault(self, fault_id: str):
        """Remove an injected fault."""
        if fault_id in self._active_faults:
            config = self._active_faults.pop(fault_id)
            logger.info(f"Removing fault {fault_id}: {config.fault_type.value}")
            self._notify_observers("fault_removed", {
                "fault_id": fault_id,
                "fault_type": config.fault_type.value
            })

    async def run_experiment(self, experiment: FaultExperiment,
                            metric_collector: Callable[[str], float] = None) -> ExperimentResult:
        """Run a complete chaos experiment."""
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.now()
        )

        self._notify_observers("experiment_started", {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name
        })

        try:
            # Verify steady state before
            result.steady_state_before = await self._verify_steady_states(
                experiment.steady_states, metric_collector
            )

            if not all(result.steady_state_before.values()):
                result.status = ExperimentStatus.FAILED
                result.errors.append("Steady state not achieved before experiment")
                return result

            # Inject faults
            fault_ids = []
            for fault_config in experiment.faults:
                fault_id = await self.inject_fault(fault_config)
                fault_ids.append(fault_id)

            # Wait for experiment duration
            start_time = datetime.now()
            while datetime.now() - start_time < experiment.duration:
                # Check abort conditions
                if await self._check_abort_conditions(experiment.abort_conditions):
                    result.status = ExperimentStatus.ABORTED
                    result.errors.append("Abort condition triggered")
                    break

                # Verify steady state during
                result.steady_state_during = await self._verify_steady_states(
                    experiment.steady_states, metric_collector
                )

                await asyncio.sleep(1)

            # Remove faults
            for fault_id in fault_ids:
                await self.remove_fault(fault_id)

            # Wait for recovery
            await asyncio.sleep(10)

            # Verify steady state after
            result.steady_state_after = await self._verify_steady_states(
                experiment.steady_states, metric_collector
            )

            # Evaluate hypothesis
            result.hypothesis_validated = all(result.steady_state_after.values())

            if result.status != ExperimentStatus.ABORTED:
                result.status = ExperimentStatus.COMPLETED

        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Experiment failed: {e}")

        result.ended_at = datetime.now()

        self._notify_observers("experiment_completed", {
            "experiment_id": experiment.experiment_id,
            "status": result.status.value,
            "hypothesis_validated": result.hypothesis_validated
        })

        self._experiment_history.append((experiment, result))
        return result

    async def _verify_steady_states(self, steady_states: List[SteadyState],
                                    metric_collector: Callable = None) -> Dict[str, bool]:
        """Verify all steady state conditions."""
        results = {}

        for ss in steady_states:
            if metric_collector:
                try:
                    value = metric_collector(ss.metric)
                    results[ss.name] = self._evaluate_condition(value, ss.condition, ss.tolerance)
                except Exception as e:
                    logger.error(f"Metric collection failed: {e}")
                    results[ss.name] = False
            else:
                # Default to passing without metric collector
                results[ss.name] = True

        return results

    def _evaluate_condition(self, value: float, condition: str, tolerance: float) -> bool:
        """Evaluate a condition string against a value."""
        import re
        match = re.match(r'([<>=!]+)\s*([\d.]+)', condition.strip())
        if not match:
            return False

        operator, threshold = match.groups()
        threshold = float(threshold)

        if operator == '<':
            return value < threshold * (1 + tolerance)
        elif operator == '<=':
            return value <= threshold * (1 + tolerance)
        elif operator == '>':
            return value > threshold * (1 - tolerance)
        elif operator == '>=':
            return value >= threshold * (1 - tolerance)
        elif operator in ('=', '=='):
            return abs(value - threshold) <= threshold * tolerance
        elif operator == '!=':
            return abs(value - threshold) > threshold * tolerance

        return False

    async def _check_abort_conditions(self, conditions: List[str]) -> bool:
        """Check if any abort condition is triggered."""
        # In real implementation, evaluate conditions against live metrics
        return False


class ResilienceScorer:
    """
    Calculate resilience scores based on chaos experiments and metrics.

    Uses multiple dimensions to evaluate system resilience:
    - Recovery time
    - Error rate during faults
    - Degradation patterns
    - Auto-healing capabilities
    """

    def __init__(self):
        self._weights = {
            "availability": 0.25,
            "recovery_time": 0.20,
            "error_rate": 0.20,
            "latency_impact": 0.15,
            "auto_healing": 0.10,
            "blast_radius": 0.10,
        }
        self._experiment_results: List[ExperimentResult] = []

    def add_experiment_result(self, result: ExperimentResult):
        """Add experiment result for scoring."""
        self._experiment_results.append(result)

    def calculate_score(self, service: str = None, period: timedelta = timedelta(days=30)) -> ResilienceScore:
        """Calculate resilience score."""
        metrics = []
        cutoff = datetime.now() - period

        # Filter results
        results = [r for r in self._experiment_results
                  if r.ended_at and r.ended_at > cutoff]

        if not results:
            return ResilienceScore(
                score=50.0,
                grade="C",
                metrics=[],
                timestamp=datetime.now(),
                period=period,
                recommendations=["Run chaos experiments to establish baseline"]
            )

        # Availability metric (% of experiments where steady state maintained)
        availability = sum(1 for r in results if all(r.steady_state_during.values())) / len(results)
        metrics.append(ResilienceMetric(
            name="availability",
            value=availability,
            weight=self._weights["availability"],
            description="Percentage of experiments with maintained steady state"
        ))

        # Recovery metric (% of experiments with successful recovery)
        recovery = sum(1 for r in results if all(r.steady_state_after.values())) / len(results)
        metrics.append(ResilienceMetric(
            name="recovery_time",
            value=recovery,
            weight=self._weights["recovery_time"],
            description="Percentage of experiments with successful recovery"
        ))

        # Hypothesis validation rate
        validation_rate = sum(1 for r in results if r.hypothesis_validated) / len(results)
        metrics.append(ResilienceMetric(
            name="hypothesis_validation",
            value=validation_rate,
            weight=self._weights["error_rate"],
            description="Percentage of experiments with validated hypothesis"
        ))

        # Calculate weighted score
        total_weight = sum(m.weight for m in metrics)
        weighted_sum = sum(m.value * m.weight for m in metrics)
        score = (weighted_sum / total_weight) * 100

        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        # Generate recommendations
        recommendations = []
        if availability < 0.9:
            recommendations.append("Improve fault tolerance - steady state was not maintained during faults")
        if recovery < 0.9:
            recommendations.append("Improve recovery mechanisms - system did not fully recover after faults")
        if validation_rate < 0.8:
            recommendations.append("Review system architecture - hypotheses frequently invalidated")

        return ResilienceScore(
            score=score,
            grade=grade,
            metrics=metrics,
            timestamp=datetime.now(),
            period=period,
            recommendations=recommendations
        )


class ChaosObserver(ABC):
    """Observer for chaos engineering events."""

    @abstractmethod
    def on_chaos_event(self, event_type: str, data: Dict[str, Any]):
        """Handle a chaos event."""
        pass


class ObservabilityChaosObserver(ChaosObserver):
    """Observer that correlates chaos events with observability data."""

    def __init__(self, trace_collector=None, metric_collector=None):
        self._trace_collector = trace_collector
        self._metric_collector = metric_collector
        self._chaos_events: List[Dict] = []

    def on_chaos_event(self, event_type: str, data: Dict[str, Any]):
        """Record chaos event for correlation."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        self._chaos_events.append(event)

        # Annotate traces if collector available
        if self._trace_collector and hasattr(self._trace_collector, 'annotate'):
            self._trace_collector.annotate(
                "chaos.event",
                event_type,
                {"chaos": data}
            )

        logger.info(f"Chaos event: {event_type} - {data}")

    def get_correlation_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate correlation report between chaos and observability."""
        events = [e for e in self._chaos_events
                 if e.get("experiment_id") == experiment_id]

        return {
            "experiment_id": experiment_id,
            "chaos_events": events,
            "event_count": len(events),
            # Would include correlated traces, metrics, etc.
        }
