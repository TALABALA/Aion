"""AION Simulation Configuration.

Centralized configuration for simulation environments, with sensible
defaults and environment-based overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aion.simulation.types import SimulationConfig, TimeMode


@dataclass
class SimulationEnvironmentConfig:
    """Top-level configuration for the simulation environment."""

    # Default simulation config
    default_sim_config: SimulationConfig = field(default_factory=SimulationConfig)

    # Snapshot settings
    max_snapshots: int = 10_000
    auto_snapshot_interval: int = 100
    deduplicate_snapshots: bool = True

    # Parallel execution
    max_parallel_simulations: int = 4
    worker_timeout_seconds: float = 600.0

    # Evaluation defaults
    default_pass_threshold: float = 0.5
    default_confidence_level: float = 0.95

    # Adversarial defaults
    default_fuzz_count: int = 50
    default_fuzz_seed: Optional[int] = None

    # Resource limits
    max_total_memory_mb: int = 4096
    max_entities_per_sim: int = 100_000
    max_events_per_sim: int = 1_000_000

    # Logging
    verbose: bool = False
    log_events: bool = False


def default_config() -> SimulationEnvironmentConfig:
    """Return default configuration."""
    return SimulationEnvironmentConfig()


def test_config() -> SimulationEnvironmentConfig:
    """Return configuration suitable for testing."""
    return SimulationEnvironmentConfig(
        default_sim_config=SimulationConfig(
            time_mode=TimeMode.STEP,
            max_ticks=1000,
            max_events=10_000,
            timeout_seconds=30.0,
            seed=42,
            deterministic=True,
            record_all_events=True,
            record_state_snapshots=True,
            snapshot_interval=10,
        ),
        max_snapshots=100,
        auto_snapshot_interval=10,
        max_parallel_simulations=1,
        verbose=True,
    )


def performance_config() -> SimulationEnvironmentConfig:
    """Return configuration optimized for performance."""
    return SimulationEnvironmentConfig(
        default_sim_config=SimulationConfig(
            time_mode=TimeMode.STEP,
            max_ticks=100_000,
            max_events=1_000_000,
            timeout_seconds=600.0,
            deterministic=False,
            record_all_events=False,
            record_state_snapshots=False,
            record_causal_graph=False,
        ),
        max_snapshots=100,
        deduplicate_snapshots=False,
        max_parallel_simulations=8,
    )
