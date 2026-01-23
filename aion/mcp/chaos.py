"""
AION MCP Chaos Engineering Module

Production-grade chaos engineering for resilience testing:
- Fault injection (errors, exceptions)
- Latency injection with distributions
- Circuit breaker simulation
- Rate limit simulation
- Network partition simulation
- Configurable attack schedules
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ============================================
# Chaos Configuration
# ============================================

class ChaosMode(str, Enum):
    """Chaos operation modes."""
    DISABLED = "disabled"
    DRY_RUN = "dry_run"  # Log but don't inject
    ENABLED = "enabled"
    ATTACK = "attack"  # Full chaos


class FaultType(str, Enum):
    """Types of injectable faults."""
    LATENCY = "latency"
    ERROR = "error"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMIT = "rate_limit"
    NETWORK_PARTITION = "network_partition"
    CORRUPTION = "corruption"


class LatencyDistribution(str, Enum):
    """Latency distribution types."""
    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    PARETO = "pareto"  # Heavy-tailed


@dataclass
class FaultConfig:
    """Configuration for a fault."""
    fault_type: FaultType
    probability: float = 0.1  # 10% chance
    enabled: bool = True

    # Latency config
    latency_ms: float = 100.0
    latency_max_ms: float = 1000.0
    latency_distribution: LatencyDistribution = LatencyDistribution.FIXED

    # Error config
    error_code: int = 500
    error_message: str = "Chaos error"
    exception_class: Optional[Type[Exception]] = None

    # Targeting
    targets: List[str] = field(default_factory=list)  # Empty = all
    exclude_targets: List[str] = field(default_factory=list)

    # Schedule
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    active_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def is_active(self) -> bool:
        """Check if fault is currently active."""
        if not self.enabled:
            return False

        now = datetime.now()

        # Check time window
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False

        # Check day of week
        if now.weekday() not in self.active_days:
            return False

        return True

    def should_trigger(self, target: Optional[str] = None) -> bool:
        """Check if fault should trigger for target."""
        if not self.is_active():
            return False

        # Check targeting
        if target:
            if self.exclude_targets and target in self.exclude_targets:
                return False
            if self.targets and target not in self.targets:
                return False

        # Random probability
        return random.random() < self.probability


@dataclass
class ChaosConfig:
    """Global chaos configuration."""
    mode: ChaosMode = ChaosMode.DISABLED
    faults: List[FaultConfig] = field(default_factory=list)
    global_probability: float = 1.0  # Master switch
    log_injections: bool = True
    metrics_enabled: bool = True
    safe_mode: bool = True  # Extra protections

    # Limits
    max_latency_ms: float = 30000.0
    max_concurrent_faults: int = 10


# ============================================
# Latency Injection
# ============================================

class LatencyInjector:
    """
    Injects configurable latency.

    Supports multiple distributions:
    - Fixed: Constant delay
    - Uniform: Random in range
    - Normal: Gaussian distribution
    - Exponential: For modeling network issues
    - Pareto: Heavy-tailed for realistic delays
    """

    def __init__(self, config: FaultConfig):
        self.config = config

    def calculate_delay(self) -> float:
        """Calculate delay in seconds."""
        dist = self.config.latency_distribution
        base = self.config.latency_ms
        max_val = self.config.latency_max_ms

        if dist == LatencyDistribution.FIXED:
            delay_ms = base

        elif dist == LatencyDistribution.UNIFORM:
            delay_ms = random.uniform(base, max_val)

        elif dist == LatencyDistribution.NORMAL:
            # Mean = base, stddev = (max - base) / 3
            stddev = (max_val - base) / 3
            delay_ms = max(0, random.gauss(base, stddev))

        elif dist == LatencyDistribution.EXPONENTIAL:
            # Exponential with mean = base
            delay_ms = random.expovariate(1 / base)

        elif dist == LatencyDistribution.PARETO:
            # Pareto distribution (heavy-tailed)
            alpha = 1.5  # Shape parameter
            delay_ms = base * (random.paretovariate(alpha) - 1)

        else:
            delay_ms = base

        # Cap at max
        return min(delay_ms, max_val) / 1000.0

    async def inject(self) -> float:
        """Inject latency, returns actual delay."""
        delay = self.calculate_delay()
        await asyncio.sleep(delay)
        return delay


# ============================================
# Error Injection
# ============================================

class ErrorInjector:
    """
    Injects errors and exceptions.

    Can inject:
    - Custom exceptions
    - HTTP-like error responses
    - Timeout errors
    """

    def __init__(self, config: FaultConfig):
        self.config = config

    def inject(self) -> None:
        """Inject an error."""
        if self.config.exception_class:
            raise self.config.exception_class(self.config.error_message)
        else:
            raise ChaosError(
                code=self.config.error_code,
                message=self.config.error_message,
                fault_type=self.config.fault_type,
            )


class ChaosError(Exception):
    """Exception raised by chaos injection."""

    def __init__(
        self,
        code: int,
        message: str,
        fault_type: FaultType,
    ):
        super().__init__(message)
        self.code = code
        self.fault_type = fault_type


# ============================================
# Chaos Monkey
# ============================================

class ChaosMonkey:
    """
    Central chaos engineering controller.

    Features:
    - Configurable fault injection
    - Target-based filtering
    - Metrics collection
    - Safe mode protections
    - Scheduled chaos attacks
    """

    def __init__(self, config: Optional[ChaosConfig] = None):
        """
        Initialize chaos monkey.

        Args:
            config: Chaos configuration
        """
        self.config = config or ChaosConfig()

        # Statistics
        self._stats = {
            "total_calls": 0,
            "faults_injected": 0,
            "latency_injected": 0,
            "errors_injected": 0,
            "skipped_dry_run": 0,
            "skipped_probability": 0,
        }

        # Active faults tracking
        self._active_faults: int = 0
        self._lock = asyncio.Lock()

        # Attack schedule
        self._scheduled_attacks: List[Tuple[datetime, FaultConfig]] = []

    def enable(self) -> None:
        """Enable chaos mode."""
        self.config.mode = ChaosMode.ENABLED
        logger.warning("Chaos monkey ENABLED")

    def disable(self) -> None:
        """Disable chaos mode."""
        self.config.mode = ChaosMode.DISABLED
        logger.info("Chaos monkey disabled")

    def set_dry_run(self) -> None:
        """Set dry-run mode (log only)."""
        self.config.mode = ChaosMode.DRY_RUN
        logger.info("Chaos monkey in dry-run mode")

    def add_fault(self, fault: FaultConfig) -> None:
        """Add a fault configuration."""
        self.config.faults.append(fault)
        logger.info(f"Added fault: {fault.fault_type.value}")

    def remove_fault(self, fault_type: FaultType) -> int:
        """Remove faults by type, returns count removed."""
        original = len(self.config.faults)
        self.config.faults = [
            f for f in self.config.faults
            if f.fault_type != fault_type
        ]
        removed = original - len(self.config.faults)
        logger.info(f"Removed {removed} faults of type {fault_type.value}")
        return removed

    def schedule_attack(
        self,
        fault: FaultConfig,
        start_at: datetime,
        duration: Optional[timedelta] = None,
    ) -> None:
        """Schedule a chaos attack."""
        fault.start_time = start_at
        if duration:
            fault.end_time = start_at + duration
        self._scheduled_attacks.append((start_at, fault))
        self.config.faults.append(fault)
        logger.info(
            "Scheduled chaos attack",
            fault_type=fault.fault_type.value,
            start_at=start_at.isoformat(),
        )

    async def maybe_inject(
        self,
        target: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Maybe inject a fault based on configuration.

        Args:
            target: Target identifier (e.g., server name, tool name)
            context: Additional context for logging

        Returns:
            Dict with injection details, or None if no injection
        """
        self._stats["total_calls"] += 1

        if self.config.mode == ChaosMode.DISABLED:
            return None

        # Global probability check
        if random.random() > self.config.global_probability:
            self._stats["skipped_probability"] += 1
            return None

        # Check concurrent fault limit
        if self._active_faults >= self.config.max_concurrent_faults:
            return None

        # Find matching fault
        for fault in self.config.faults:
            if fault.should_trigger(target):
                return await self._inject_fault(fault, target, context)

        return None

    async def _inject_fault(
        self,
        fault: FaultConfig,
        target: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Inject a specific fault."""
        async with self._lock:
            self._active_faults += 1

        try:
            result = {
                "fault_type": fault.fault_type.value,
                "target": target,
                "timestamp": datetime.now().isoformat(),
            }

            # Dry run - log only
            if self.config.mode == ChaosMode.DRY_RUN:
                self._stats["skipped_dry_run"] += 1
                if self.config.log_injections:
                    logger.info(
                        "CHAOS DRY-RUN",
                        fault_type=fault.fault_type.value,
                        target=target,
                    )
                result["dry_run"] = True
                return result

            # Actually inject
            self._stats["faults_injected"] += 1

            if self.config.log_injections:
                logger.warning(
                    "CHAOS INJECTION",
                    fault_type=fault.fault_type.value,
                    target=target,
                    context=context,
                )

            if fault.fault_type == FaultType.LATENCY:
                injector = LatencyInjector(fault)
                delay = await injector.inject()
                result["delay_seconds"] = delay
                self._stats["latency_injected"] += 1

            elif fault.fault_type in (FaultType.ERROR, FaultType.EXCEPTION):
                injector = ErrorInjector(fault)
                injector.inject()
                self._stats["errors_injected"] += 1

            elif fault.fault_type == FaultType.TIMEOUT:
                # Inject maximum latency to simulate timeout
                await asyncio.sleep(fault.latency_max_ms / 1000.0)
                raise asyncio.TimeoutError("Chaos-induced timeout")

            elif fault.fault_type == FaultType.CIRCUIT_OPEN:
                raise ChaosError(
                    code=503,
                    message="Circuit breaker is open (chaos)",
                    fault_type=fault.fault_type,
                )

            elif fault.fault_type == FaultType.RATE_LIMIT:
                raise ChaosError(
                    code=429,
                    message="Rate limit exceeded (chaos)",
                    fault_type=fault.fault_type,
                )

            elif fault.fault_type == FaultType.NETWORK_PARTITION:
                raise ChaosError(
                    code=503,
                    message="Network partition (chaos)",
                    fault_type=fault.fault_type,
                )

            return result

        finally:
            async with self._lock:
                self._active_faults -= 1

    def wrap(
        self,
        target: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to wrap functions with chaos injection.

        Args:
            target: Target identifier

        Returns:
            Decorator
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Pre-execution chaos
                await self.maybe_inject(target, {"phase": "pre"})

                try:
                    result = await func(*args, **kwargs)

                    # Post-execution chaos (e.g., corrupt response)
                    injection = await self.maybe_inject(
                        target,
                        {"phase": "post"},
                    )

                    return result

                except Exception as e:
                    # Could inject additional failures on error
                    raise

            return wrapper
        return decorator

    @asynccontextmanager
    async def chaos_context(
        self,
        target: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for chaos injection.

        Usage:
            async with chaos.chaos_context("my_service"):
                await do_something()
        """
        # Pre-injection
        await self.maybe_inject(target, {**(context or {}), "phase": "pre"})

        try:
            yield
        finally:
            pass  # Post-injection could go here

    def get_stats(self) -> Dict[str, Any]:
        """Get chaos injection statistics."""
        return {
            **self._stats,
            "mode": self.config.mode.value,
            "active_faults": self._active_faults,
            "configured_faults": len(self.config.faults),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        for key in self._stats:
            self._stats[key] = 0


# ============================================
# Chaos Experiments
# ============================================

@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""
    name: str
    started_at: datetime
    ended_at: datetime
    success: bool
    hypothesis_validated: bool
    observations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ChaosExperiment:
    """
    Structured chaos experiment.

    Implements the chaos engineering experiment loop:
    1. Define steady state hypothesis
    2. Introduce chaos
    3. Observe behavior
    4. Validate hypothesis
    """

    def __init__(
        self,
        name: str,
        description: str,
        chaos_monkey: ChaosMonkey,
        hypothesis: Callable[[], bool],
        fault: FaultConfig,
        duration: timedelta = timedelta(minutes=5),
        cooldown: timedelta = timedelta(minutes=2),
    ):
        """
        Initialize experiment.

        Args:
            name: Experiment name
            description: Description
            chaos_monkey: Chaos monkey instance
            hypothesis: Function that returns True if system is healthy
            fault: Fault to inject
            duration: How long to run chaos
            cooldown: Cooldown after experiment
        """
        self.name = name
        self.description = description
        self.chaos_monkey = chaos_monkey
        self.hypothesis = hypothesis
        self.fault = fault
        self.duration = duration
        self.cooldown = cooldown

        self._observations: List[Dict[str, Any]] = []

    async def run(self) -> ExperimentResult:
        """Run the chaos experiment."""
        started_at = datetime.now()
        logger.info(f"Starting chaos experiment: {self.name}")

        try:
            # 1. Verify steady state before chaos
            pre_check = self.hypothesis()
            if not pre_check:
                return ExperimentResult(
                    name=self.name,
                    started_at=started_at,
                    ended_at=datetime.now(),
                    success=False,
                    hypothesis_validated=False,
                    error="Pre-experiment steady state check failed",
                )

            self._observations.append({
                "phase": "pre_chaos",
                "hypothesis_valid": True,
                "timestamp": datetime.now().isoformat(),
            })

            # 2. Enable chaos
            original_mode = self.chaos_monkey.config.mode
            self.chaos_monkey.add_fault(self.fault)
            self.chaos_monkey.enable()

            # 3. Run for duration, observing
            end_time = datetime.now() + self.duration
            observation_interval = self.duration.total_seconds() / 10

            while datetime.now() < end_time:
                await asyncio.sleep(observation_interval)

                # Observe
                is_healthy = self.hypothesis()
                self._observations.append({
                    "phase": "during_chaos",
                    "hypothesis_valid": is_healthy,
                    "timestamp": datetime.now().isoformat(),
                    "chaos_stats": self.chaos_monkey.get_stats(),
                })

            # 4. Disable chaos
            self.chaos_monkey.remove_fault(self.fault.fault_type)
            self.chaos_monkey.config.mode = original_mode

            # 5. Cooldown
            logger.info(f"Chaos experiment cooldown: {self.cooldown}")
            await asyncio.sleep(self.cooldown.total_seconds())

            # 6. Verify steady state after chaos
            post_check = self.hypothesis()
            self._observations.append({
                "phase": "post_chaos",
                "hypothesis_valid": post_check,
                "timestamp": datetime.now().isoformat(),
            })

            # 7. Analyze results
            during_chaos_healthy = all(
                obs["hypothesis_valid"]
                for obs in self._observations
                if obs["phase"] == "during_chaos"
            )

            return ExperimentResult(
                name=self.name,
                started_at=started_at,
                ended_at=datetime.now(),
                success=True,
                hypothesis_validated=during_chaos_healthy and post_check,
                observations=self._observations,
                metrics=self.chaos_monkey.get_stats(),
            )

        except Exception as e:
            logger.error(f"Chaos experiment failed: {e}")
            return ExperimentResult(
                name=self.name,
                started_at=started_at,
                ended_at=datetime.now(),
                success=False,
                hypothesis_validated=False,
                observations=self._observations,
                error=str(e),
            )


# ============================================
# Pre-built Fault Profiles
# ============================================

class FaultProfiles:
    """Pre-configured fault profiles for common scenarios."""

    @staticmethod
    def network_latency(
        probability: float = 0.1,
        min_ms: float = 100,
        max_ms: float = 500,
    ) -> FaultConfig:
        """Simulate network latency."""
        return FaultConfig(
            fault_type=FaultType.LATENCY,
            probability=probability,
            latency_ms=min_ms,
            latency_max_ms=max_ms,
            latency_distribution=LatencyDistribution.PARETO,
        )

    @staticmethod
    def service_unavailable(probability: float = 0.05) -> FaultConfig:
        """Simulate service unavailability."""
        return FaultConfig(
            fault_type=FaultType.ERROR,
            probability=probability,
            error_code=503,
            error_message="Service temporarily unavailable",
        )

    @staticmethod
    def timeout(probability: float = 0.02, timeout_ms: float = 30000) -> FaultConfig:
        """Simulate request timeout."""
        return FaultConfig(
            fault_type=FaultType.TIMEOUT,
            probability=probability,
            latency_max_ms=timeout_ms,
        )

    @staticmethod
    def rate_limit(probability: float = 0.1) -> FaultConfig:
        """Simulate rate limiting."""
        return FaultConfig(
            fault_type=FaultType.RATE_LIMIT,
            probability=probability,
            error_code=429,
            error_message="Too many requests",
        )

    @staticmethod
    def circuit_breaker_open(probability: float = 0.05) -> FaultConfig:
        """Simulate open circuit breaker."""
        return FaultConfig(
            fault_type=FaultType.CIRCUIT_OPEN,
            probability=probability,
        )

    @staticmethod
    def network_partition(
        probability: float = 0.01,
        targets: Optional[List[str]] = None,
    ) -> FaultConfig:
        """Simulate network partition."""
        return FaultConfig(
            fault_type=FaultType.NETWORK_PARTITION,
            probability=probability,
            targets=targets or [],
        )

    @staticmethod
    def intermittent_failure(
        probability: float = 0.1,
        error_probability: float = 0.3,
    ) -> List[FaultConfig]:
        """Simulate intermittent failures (latency + errors)."""
        return [
            FaultConfig(
                fault_type=FaultType.LATENCY,
                probability=probability,
                latency_ms=200,
                latency_max_ms=2000,
                latency_distribution=LatencyDistribution.EXPONENTIAL,
            ),
            FaultConfig(
                fault_type=FaultType.ERROR,
                probability=probability * error_probability,
                error_code=500,
                error_message="Intermittent failure",
            ),
        ]


# ============================================
# Global Instance
# ============================================

_global_chaos_monkey: Optional[ChaosMonkey] = None


def get_chaos_monkey() -> ChaosMonkey:
    """Get global chaos monkey instance."""
    global _global_chaos_monkey
    if _global_chaos_monkey is None:
        _global_chaos_monkey = ChaosMonkey()
    return _global_chaos_monkey


def init_chaos_monkey(config: ChaosConfig) -> ChaosMonkey:
    """Initialize global chaos monkey with config."""
    global _global_chaos_monkey
    _global_chaos_monkey = ChaosMonkey(config)
    return _global_chaos_monkey
