"""
AION Reinforcement Learning Types

Core type definitions for the system-wide reinforcement learning loop.
Provides state-action-reward representations, policy configurations,
bandit statistics, and experiment types for adaptive learning across
all AION subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import uuid


# ---------------------------------------------------------------------------
# Reward signal taxonomy
# ---------------------------------------------------------------------------

class RewardSource(str, Enum):
    """Sources of reward signals."""

    # Explicit user feedback
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    EXPLICIT_RATING = "explicit_rating"
    EXPLICIT_CORRECTION = "explicit_correction"

    # Implicit behavioural signals
    IMPLICIT_COMPLETION = "implicit_completion"
    IMPLICIT_ABANDONMENT = "implicit_abandonment"
    IMPLICIT_RETRY = "implicit_retry"
    IMPLICIT_COPY = "implicit_copy"
    IMPLICIT_DWELL_TIME = "implicit_dwell_time"
    IMPLICIT_EDIT_DISTANCE = "implicit_edit_distance"

    # Outcome-based signals
    OUTCOME_SUCCESS = "outcome_success"
    OUTCOME_FAILURE = "outcome_failure"
    OUTCOME_PARTIAL = "outcome_partial"

    # Continuous metric signals
    METRIC_LATENCY = "metric_latency"
    METRIC_COST = "metric_cost"
    METRIC_QUALITY = "metric_quality"
    METRIC_SAFETY = "metric_safety"

    # Intrinsic motivation signals
    INTRINSIC_CURIOSITY = "intrinsic_curiosity"
    INTRINSIC_NOVELTY = "intrinsic_novelty"


class RewardType(str, Enum):
    """Temporal classification of reward signals."""

    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    SHAPED = "shaped"
    INTRINSIC = "intrinsic"
    HINDSIGHT = "hindsight"


# ---------------------------------------------------------------------------
# Reward signals
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    """A single reward signal with provenance tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: RewardSource = RewardSource.IMPLICIT_COMPLETION
    reward_type: RewardType = RewardType.IMMEDIATE
    value: float = 0.0
    confidence: float = 1.0
    interaction_id: str = ""
    action_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    delay_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def discounted_value(self, gamma: float = 0.99) -> float:
        """Return time-discounted, confidence-weighted reward."""
        discount = gamma ** (self.delay_seconds / 60.0)
        return self.value * discount * self.confidence


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class StateRepresentation:
    """Featurised representation of the system state for RL."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_embedding: Optional[np.ndarray] = None
    query_type: str = ""
    query_complexity: float = 0.0
    turn_count: int = 0
    user_sentiment: float = 0.0
    available_tools: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)
    context_features: Dict[str, float] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert state to a fixed-length feature vector."""
        features = [
            self.query_complexity,
            self.turn_count / 100.0,
            self.user_sentiment,
            len(self.available_tools) / 50.0,
            np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            np.std(self.recent_rewards) if len(self.recent_rewards) > 1 else 0.0,
            len(self.recent_actions) / 20.0,
        ]
        # Append arbitrary context features in sorted-key order
        for _key in sorted(self.context_features):
            features.append(self.context_features[_key])
        return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Categories of actions the RL loop can optimise."""

    TOOL_SELECTION = "tool_selection"
    PLANNING_STRATEGY = "planning_strategy"
    AGENT_ASSIGNMENT = "agent_assignment"
    RESPONSE_STYLE = "response_style"
    RESOURCE_ALLOCATION = "resource_allocation"


@dataclass
class Action:
    """An action taken by the system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType = ActionType.TOOL_SELECTION
    choice: str = ""
    alternatives: List[str] = field(default_factory=list)
    confidence: float = 0.0
    exploration: bool = False
    policy_version: str = ""
    state_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Experience tuples
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """A single experience tuple (s, a, r, s') with metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: StateRepresentation = field(default_factory=StateRepresentation)
    action: Action = field(default_factory=Action)
    reward: float = 0.0
    next_state: Optional[StateRepresentation] = None
    done: bool = False
    rewards: List[RewardSignal] = field(default_factory=list)
    cumulative_reward: float = 0.0
    priority: float = 1.0
    td_error: Optional[float] = None
    interaction_id: str = ""
    episode_id: str = ""
    step_index: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def compute_cumulative_reward(self, gamma: float = 0.99) -> float:
        """Aggregate all reward signals with temporal discounting."""
        total = self.reward
        for signal in self.rewards:
            total += signal.discounted_value(gamma)
        self.cumulative_reward = total
        return total


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Configuration for a learning policy."""

    name: str = ""
    version: str = "1.0.0"
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    batch_size: int = 32
    entropy_coefficient: float = 0.01
    gradient_clip: float = 1.0
    use_advantage_normalisation: bool = True


# ---------------------------------------------------------------------------
# Bandit arm statistics
# ---------------------------------------------------------------------------

@dataclass
class ArmStatistics:
    """Statistics for a bandit arm (Beta-Bernoulli model)."""

    arm_id: str = ""
    pulls: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    reward_variance: float = 0.0
    alpha: float = 1.0  # Beta distribution prior
    beta: float = 1.0
    last_pulled: Optional[datetime] = None

    def update(self, reward: float) -> None:
        """Online update of arm statistics (Welford's algorithm for variance)."""
        self.pulls += 1
        self.total_reward += reward
        delta = reward - self.avg_reward
        self.avg_reward += delta / self.pulls
        delta2 = reward - self.avg_reward
        self.reward_variance += delta * delta2
        # Beta-distribution posterior update
        if reward > 0:
            self.alpha += reward
        else:
            self.beta += abs(reward)
        self.last_pulled = datetime.now()

    @property
    def variance(self) -> float:
        return self.reward_variance / max(self.pulls - 1, 1) if self.pulls > 1 else 0.0


# ---------------------------------------------------------------------------
# Experiment types
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """A single variant (arm) in an A/B experiment."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    policy_config: Dict[str, Any] = field(default_factory=dict)
    traffic_percentage: float = 0.5
    sample_count: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    reward_variance: float = 0.0
    reward_samples: List[float] = field(default_factory=list)

    def record(self, reward: float) -> None:
        """Record a reward observation with Welford update."""
        n = self.sample_count
        old_mean = self.avg_reward
        self.sample_count += 1
        self.total_reward += reward
        delta = reward - old_mean
        self.avg_reward = old_mean + delta / self.sample_count
        self.reward_variance += delta * (reward - self.avg_reward)
        self.reward_samples.append(reward)
        # Keep bounded history
        if len(self.reward_samples) > 10000:
            self.reward_samples = self.reward_samples[-5000:]


@dataclass
class Experiment:
    """An A/B experiment comparing two policy variants."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    action_type: ActionType = ActionType.TOOL_SELECTION
    hypothesis: str = ""
    control: ExperimentVariant = field(default_factory=ExperimentVariant)
    treatment: ExperimentVariant = field(default_factory=ExperimentVariant)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    min_samples: int = 1000
    significance_level: float = 0.05
    power: float = 0.80
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    winner: Optional[str] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[tuple] = None
