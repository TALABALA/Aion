"""AION Stress Test Generator - Generate configurable stress test scenarios.

Provides:
- StressTestGenerator: Creates stress scenarios with configurable load
  profiles (constant, ramp, spike, wave) for throughput and latency testing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import structlog

from aion.simulation.types import Scenario, ScenarioType

logger = structlog.get_logger(__name__)


class LoadProfile:
    """Defines how load changes over time."""

    @staticmethod
    def constant(users: int, duration: float, rps: float = 10.0) -> List[Dict[str, Any]]:
        """Constant load."""
        events: List[Dict[str, Any]] = []
        interval = 1.0 / rps if rps > 0 else 1.0
        t = 0.0
        idx = 0
        while t < duration:
            for u in range(users):
                events.append({
                    "time": t + u * 0.001,
                    "type": "user_input",
                    "source": f"user_{u}",
                    "data": {"message": f"Request {idx}", "request_id": idx},
                })
                idx += 1
            t += interval
        return events

    @staticmethod
    def ramp(
        start_users: int,
        end_users: int,
        duration: float,
        rps: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Linearly ramping load."""
        events: List[Dict[str, Any]] = []
        interval = 1.0 / rps if rps > 0 else 1.0
        steps = int(duration / interval)
        idx = 0
        for step in range(steps):
            t = step * interval
            progress = t / duration
            users = int(start_users + (end_users - start_users) * progress)
            for u in range(users):
                events.append({
                    "time": t + u * 0.001,
                    "type": "user_input",
                    "source": f"user_{u}",
                    "data": {"message": f"Ramp request {idx}", "request_id": idx},
                })
                idx += 1
        return events

    @staticmethod
    def spike(
        base_users: int,
        spike_users: int,
        duration: float,
        spike_at: float = 0.5,
        spike_duration: float = 0.1,
        rps: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Load with a sudden spike."""
        events: List[Dict[str, Any]] = []
        interval = 1.0 / rps if rps > 0 else 1.0
        spike_start = duration * spike_at
        spike_end = spike_start + duration * spike_duration
        steps = int(duration / interval)
        idx = 0
        for step in range(steps):
            t = step * interval
            users = spike_users if spike_start <= t <= spike_end else base_users
            for u in range(users):
                events.append({
                    "time": t + u * 0.001,
                    "type": "user_input",
                    "source": f"user_{u}",
                    "data": {"message": f"Spike request {idx}", "request_id": idx},
                })
                idx += 1
        return events

    @staticmethod
    def wave(
        min_users: int,
        max_users: int,
        duration: float,
        period: float = 10.0,
        rps: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Sinusoidal load pattern."""
        events: List[Dict[str, Any]] = []
        interval = 1.0 / rps if rps > 0 else 1.0
        steps = int(duration / interval)
        amplitude = (max_users - min_users) / 2
        midpoint = (max_users + min_users) / 2
        idx = 0
        for step in range(steps):
            t = step * interval
            users = int(midpoint + amplitude * math.sin(2 * math.pi * t / period))
            users = max(1, users)
            for u in range(users):
                events.append({
                    "time": t + u * 0.001,
                    "type": "user_input",
                    "source": f"user_{u}",
                    "data": {"message": f"Wave request {idx}", "request_id": idx},
                })
                idx += 1
        return events


class StressTestGenerator:
    """Generates stress test scenarios with configurable load profiles.

    Features:
    - Multiple load profiles (constant, ramp, spike, wave).
    - Configurable user count, RPS, and duration.
    - Automatic success criteria based on SLA targets.
    - Soak test generation for long-running stability.
    """

    def __init__(self) -> None:
        self._sla_targets: Dict[str, float] = {
            "max_latency_ms": 1000.0,
            "p99_latency_ms": 500.0,
            "error_rate": 0.01,
            "min_throughput": 100.0,
        }

    def set_sla(self, **kwargs: float) -> None:
        """Set SLA targets."""
        self._sla_targets.update(kwargs)

    def generate_constant_load(
        self,
        users: int = 50,
        duration: float = 60.0,
        rps: float = 10.0,
        name: str = "constant_load_stress",
    ) -> Scenario:
        """Generate constant-load stress test."""
        events = LoadProfile.constant(users, duration, rps)
        return self._build_scenario(name, f"Constant load: {users} users, {rps} RPS", events, users)

    def generate_ramp(
        self,
        start_users: int = 1,
        end_users: int = 100,
        duration: float = 120.0,
        rps: float = 10.0,
        name: str = "ramp_stress",
    ) -> Scenario:
        """Generate ramping-load stress test."""
        events = LoadProfile.ramp(start_users, end_users, duration, rps)
        return self._build_scenario(
            name,
            f"Ramp load: {start_users} -> {end_users} users over {duration}s",
            events,
            end_users,
        )

    def generate_spike(
        self,
        base_users: int = 10,
        spike_users: int = 200,
        duration: float = 60.0,
        name: str = "spike_stress",
    ) -> Scenario:
        events = LoadProfile.spike(base_users, spike_users, duration)
        return self._build_scenario(
            name,
            f"Spike load: {base_users} base, {spike_users} spike",
            events,
            spike_users,
        )

    def generate_wave(
        self,
        min_users: int = 5,
        max_users: int = 100,
        duration: float = 120.0,
        name: str = "wave_stress",
    ) -> Scenario:
        events = LoadProfile.wave(min_users, max_users, duration)
        return self._build_scenario(
            name,
            f"Wave load: {min_users}-{max_users} users",
            events,
            max_users,
        )

    def generate_soak(
        self,
        users: int = 20,
        duration: float = 3600.0,
        rps: float = 5.0,
        name: str = "soak_stress",
    ) -> Scenario:
        """Generate long-running soak/stability test."""
        events = LoadProfile.constant(users, duration, rps)
        scenario = self._build_scenario(
            name, f"Soak test: {users} users for {duration}s", events, users,
        )
        scenario.tags.add("soak")
        scenario.max_time = duration * 1.1
        return scenario

    def generate_suite(self) -> List[Scenario]:
        """Generate a full stress test suite."""
        return [
            self.generate_constant_load(users=10, duration=30),
            self.generate_ramp(start_users=1, end_users=50, duration=60),
            self.generate_spike(base_users=5, spike_users=100, duration=30),
            self.generate_wave(min_users=2, max_users=50, duration=60),
        ]

    def _build_scenario(
        self,
        name: str,
        description: str,
        events: List[Dict[str, Any]],
        max_users: int,
    ) -> Scenario:
        entities = [
            {"type": "user", "name": f"stress_user_{i}"}
            for i in range(max_users)
        ]

        return Scenario(
            name=name,
            description=description,
            type=ScenarioType.STRESS,
            initial_entities=entities,
            scripted_events=events,
            success_criteria=[
                {"name": "all_processed", "condition": "all_requests_processed"},
                {
                    "name": "latency_sla",
                    "condition": f"p99_latency < {self._sla_targets['p99_latency_ms']}",
                },
                {
                    "name": "error_rate_sla",
                    "condition": f"error_rate < {self._sla_targets['error_rate']}",
                },
                {
                    "name": "throughput_sla",
                    "condition": f"throughput > {self._sla_targets['min_throughput']}",
                },
            ],
            config={
                "max_users": max_users,
                "total_events": len(events),
                "sla_targets": dict(self._sla_targets),
            },
            difficulty=min(1.0, 0.5 + max_users / 200),
            tags={"stress", "performance"},
        )
