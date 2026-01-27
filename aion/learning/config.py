"""
AION Reinforcement Learning Configuration

Centralised configuration for all learning subsystems with sensible
defaults tuned for online learning in an interactive AI system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RewardConfig:
    """Configuration for reward collection and processing."""

    explicit_weight: float = 1.0
    implicit_weight: float = 0.5
    outcome_weight: float = 0.8
    metric_weight: float = 0.3
    intrinsic_weight: float = 0.2
    reward_clip: float = 2.0
    normalisation_window: int = 1000
    discount_gamma: float = 0.99
    delay_penalty_per_minute: float = 0.01


@dataclass
class BufferConfig:
    """Configuration for experience replay buffer."""

    max_size: int = 100_000
    min_size_for_sampling: int = 100
    use_priority: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.001
    priority_epsilon: float = 1e-6
    n_step_returns: int = 3


@dataclass
class PolicyOptimizerConfig:
    """Configuration for the policy optimizer."""

    training_interval_seconds: float = 60.0
    batch_size: int = 32
    updates_per_interval: int = 10
    default_learning_rate: float = 0.001
    default_exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    gradient_clip: float = 1.0
    entropy_coefficient: float = 0.01


@dataclass
class BanditConfig:
    """Configuration for bandit algorithms."""

    default_prior_alpha: float = 1.0
    default_prior_beta: float = 1.0
    ucb_confidence: float = 2.0
    contextual_feature_dim: int = 10
    contextual_prior_variance: float = 1.0
    contextual_noise_variance: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing."""

    min_samples_per_variant: int = 100
    significance_level: float = 0.05
    min_effect_size: float = 0.01
    auto_stop_enabled: bool = True
    max_experiment_duration_hours: float = 168.0  # 1 week
    sequential_testing_enabled: bool = True
    alpha_spending_function: str = "obrien_fleming"


@dataclass
class PersistenceConfig:
    """Configuration for learning state persistence."""

    save_interval_seconds: float = 300.0  # 5 minutes
    checkpoint_dir: str = "data/learning/checkpoints"
    max_checkpoints: int = 10
    save_buffer: bool = True
    save_policies: bool = True
    save_bandits: bool = True
    save_experiments: bool = True


@dataclass
class LearningConfig:
    """Top-level learning configuration."""

    enabled: bool = True
    training_enabled: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    policy_optimizer: PolicyOptimizerConfig = field(default_factory=PolicyOptimizerConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningConfig":
        """Create configuration from dictionary."""
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key) and isinstance(value, dict):
                sub = getattr(cfg, key)
                for k, v in value.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
