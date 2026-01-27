"""
AION Distributed Experience Sharing

Shares reinforcement learning experiences across cluster nodes so that
every learner benefits from the collective exploration of the swarm.
Supports configurable share ratios, priority-based sharing of rare or
high-information experiences, deduplication, diversity metrics, and
importance-sampling weights for off-policy correction.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

import structlog

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from aion.distributed.types import DistributedTask, TaskPriority, TaskType

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager
    from aion.systems.agents.learning.reinforcement import Experience

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperienceSharingConfig:
    """Configuration for experience sharing."""

    share_ratio: float = 0.1  # Share 10% of experiences
    max_shared_buffer_size: int = 10000
    deduplication_enabled: bool = True
    deduplication_window: int = 5000
    priority_sharing_enabled: bool = True
    diversity_bonus: float = 0.2
    importance_sampling: bool = True
    importance_clip_range: float = 2.0
    broadcast_interval_seconds: float = 30.0
    min_experiences_to_share: int = 5
    max_experiences_per_broadcast: int = 100


# ---------------------------------------------------------------------------
# Diversity metrics helper
# ---------------------------------------------------------------------------


@dataclass
class DiversityMetrics:
    """Tracks diversity of the shared experience buffer."""

    unique_actions: Set[str] = field(default_factory=set)
    unique_state_hashes: Set[str] = field(default_factory=set)
    reward_histogram: Dict[str, int] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)
    total_experiences: int = 0

    def update(self, action: str, state_hash: str, reward: float) -> None:
        """Update diversity metrics with a new experience."""
        self.unique_actions.add(action)
        self.unique_state_hashes.add(state_hash)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.total_experiences += 1

        # Bucket reward into histogram bins
        bucket = self._reward_bucket(reward)
        self.reward_histogram[bucket] = self.reward_histogram.get(bucket, 0) + 1

    def diversity_score(self) -> float:
        """Compute an overall diversity score in [0, 1].

        Higher values indicate greater diversity across actions, states,
        and reward distribution.
        """
        if self.total_experiences == 0:
            return 0.0

        # Action entropy (normalised)
        action_entropy = self._normalised_entropy(self.action_counts)

        # State coverage fraction (capped at 1)
        state_coverage = min(
            1.0,
            len(self.unique_state_hashes) / max(1, self.total_experiences),
        )

        # Reward distribution entropy
        reward_entropy = self._normalised_entropy(self.reward_histogram)

        return (action_entropy + state_coverage + reward_entropy) / 3.0

    @staticmethod
    def _normalised_entropy(counts: Dict[str, int]) -> float:
        """Shannon entropy normalised to [0, 1]."""
        total = sum(counts.values())
        if total == 0 or len(counts) <= 1:
            return 0.0
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def _reward_bucket(reward: float) -> str:
        """Map a reward to a histogram bucket string."""
        if reward < -0.5:
            return "very_negative"
        if reward < 0.0:
            return "negative"
        if reward < 0.5:
            return "positive"
        return "very_positive"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_actions": len(self.unique_actions),
            "unique_state_hashes": len(self.unique_state_hashes),
            "total_experiences": self.total_experiences,
            "diversity_score": round(self.diversity_score(), 4),
            "action_counts": dict(self.action_counts),
        }


# ---------------------------------------------------------------------------
# ExperienceSharing
# ---------------------------------------------------------------------------


class ExperienceSharing:
    """
    Shares RL experiences across cluster nodes for distributed learning.

    Allows every learner in the cluster to benefit from the collective
    exploration performed by all agents.  Experiences are selected for
    sharing based on priority, diversity, and information content.

    Features:
        - Configurable share ratio (default 10%)
        - Priority-based sharing: share rare / high-reward experiences
        - Experience replay buffer integration
        - Deduplication of received experiences (hash-based)
        - Experience diversity metrics (action, state, reward entropy)
        - Importance-sampling weights for off-policy correction
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        config: Optional[ExperienceSharingConfig] = None,
    ) -> None:
        self._cluster = cluster_manager
        self._config = config or ExperienceSharingConfig()

        # Shared experience buffer (received from other nodes)
        self._shared_buffer: deque[Dict[str, Any]] = deque(
            maxlen=self._config.max_shared_buffer_size,
        )

        # Deduplication
        self._seen_hashes: deque[str] = deque(
            maxlen=self._config.deduplication_window,
        )

        # Diversity tracking
        self._diversity = DiversityMetrics()

        # Statistics
        self._total_shared: int = 0
        self._total_received: int = 0
        self._total_deduplicated: int = 0
        self._total_broadcasts: int = 0

        logger.info(
            "experience_sharing_created",
            share_ratio=self._config.share_ratio,
            max_buffer=self._config.max_shared_buffer_size,
            dedup=self._config.deduplication_enabled,
        )

    # ------------------------------------------------------------------
    # Share experiences to the cluster
    # ------------------------------------------------------------------

    async def share_experiences(
        self,
        experiences: List[Dict[str, Any]],
    ) -> int:
        """Select and broadcast a subset of experiences to the cluster.

        A fraction of the provided experiences (controlled by
        ``share_ratio``) is selected based on priority and diversity
        and sent to peer nodes via a distributed task.

        Args:
            experiences: List of experience dicts (matching
                ``Experience.to_dict()`` schema).

        Returns:
            Number of experiences actually shared.
        """
        if not experiences:
            return 0

        selected = self._select_experiences_to_share(experiences)

        if len(selected) < self._config.min_experiences_to_share:
            logger.debug(
                "too_few_experiences_to_share",
                selected=len(selected),
                minimum=self._config.min_experiences_to_share,
            )
            return 0

        # Cap per broadcast
        if len(selected) > self._config.max_experiences_per_broadcast:
            selected = selected[: self._config.max_experiences_per_broadcast]

        # Add importance-sampling weights if enabled
        if self._config.importance_sampling:
            selected = self._annotate_importance_weights(selected, experiences)

        task = DistributedTask(
            name="experience_share",
            task_type=TaskType.TRAINING_STEP.value,
            priority=TaskPriority.NORMAL,
            payload={
                "action": "share_experiences",
                "node_id": self._get_node_id(),
                "experiences": selected,
                "count": len(selected),
                "timestamp": time.time(),
            },
        )

        try:
            await self._cluster.submit_task(task)
            self._total_shared += len(selected)
            self._total_broadcasts += 1

            logger.info(
                "experiences_shared",
                count=len(selected),
                total=len(experiences),
                ratio=round(len(selected) / len(experiences), 3),
            )
        except Exception:
            logger.exception("experience_share_failed")
            return 0

        return len(selected)

    # ------------------------------------------------------------------
    # Receive experiences from peer nodes
    # ------------------------------------------------------------------

    async def receive_experiences(
        self,
        experiences: List[Dict[str, Any]],
    ) -> int:
        """Receive shared experiences from a peer node.

        Deduplicates, validates, and stores incoming experiences in the
        shared buffer for replay by the local learner.

        Args:
            experiences: List of experience dicts from a peer node.

        Returns:
            Number of experiences actually accepted (after dedup).
        """
        accepted = 0

        for exp in experiences:
            # Deduplication
            if self._config.deduplication_enabled:
                exp_hash = self._hash_experience(exp)
                if exp_hash in self._seen_hashes:
                    self._total_deduplicated += 1
                    continue
                self._seen_hashes.append(exp_hash)

            # Basic validation
            if not self._validate_experience(exp):
                continue

            self._shared_buffer.append(exp)
            accepted += 1

            # Update diversity metrics
            action = exp.get("action", "unknown")
            state_hash = self._hash_state(exp.get("state", {}))
            reward = float(exp.get("reward", 0.0))
            self._diversity.update(action, state_hash, reward)

        self._total_received += accepted

        logger.debug(
            "experiences_received",
            total=len(experiences),
            accepted=accepted,
            deduplicated=len(experiences) - accepted,
            buffer_size=len(self._shared_buffer),
        )

        return accepted

    # ------------------------------------------------------------------
    # Buffer access
    # ------------------------------------------------------------------

    def get_shared_buffer(self) -> List[Dict[str, Any]]:
        """Return a copy of the shared experience buffer.

        Returns:
            List of experience dicts from peer nodes.
        """
        return list(self._shared_buffer)

    def sample_shared(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch from the shared buffer for replay.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Randomly sampled experiences from the shared buffer.
        """
        buffer_list = list(self._shared_buffer)
        if not buffer_list:
            return []

        k = min(batch_size, len(buffer_list))
        return random.sample(buffer_list, k)

    # ------------------------------------------------------------------
    # Priority-based sharing
    # ------------------------------------------------------------------

    async def prioritized_sharing(
        self,
        experiences: List[Dict[str, Any]],
        priority_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> int:
        """Share experiences using a custom priority function.

        Experiences with higher priority scores are more likely to be
        selected for sharing.

        Args:
            experiences: List of experience dicts.
            priority_fn: Callable that maps an experience dict to a
                priority score (higher = more likely to be shared).
                Defaults to using the ``priority`` field.

        Returns:
            Number of experiences shared.
        """
        if not experiences:
            return 0

        fn = priority_fn or self._default_priority_fn

        # Score every experience
        scored: List[tuple[float, Dict[str, Any]]] = []
        for exp in experiences:
            try:
                score = fn(exp)
            except Exception:
                score = 0.0
            scored.append((score, exp))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select top fraction
        num_to_share = max(
            self._config.min_experiences_to_share,
            int(len(scored) * self._config.share_ratio),
        )
        num_to_share = min(num_to_share, self._config.max_experiences_per_broadcast)

        selected = [exp for _, exp in scored[:num_to_share]]

        return await self.share_experiences(selected)

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def _select_experiences_to_share(
        self,
        experiences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select a subset of experiences for sharing.

        Selection criteria:
        1. Base fraction determined by ``share_ratio``.
        2. Priority weighting: high-reward and high-TD-error experiences
           are preferred.
        3. Diversity bonus: experiences with rare actions or novel states
           receive a boost.

        Args:
            experiences: Full list of local experiences.

        Returns:
            Selected subset of experiences.
        """
        num_to_select = max(1, int(len(experiences) * self._config.share_ratio))

        if not self._config.priority_sharing_enabled:
            return random.sample(
                experiences, min(num_to_select, len(experiences)),
            )

        # Score each experience
        scored: List[tuple[float, int]] = []
        for idx, exp in enumerate(experiences):
            score = self._compute_share_score(exp)
            scored.append((score, idx))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        selected_indices = [idx for _, idx in scored[:num_to_select]]
        return [experiences[i] for i in selected_indices]

    def _compute_share_score(self, exp: Dict[str, Any]) -> float:
        """Compute the sharing priority score for an experience.

        Higher scores indicate more valuable experiences to share.
        """
        score = 0.0

        # Reward magnitude: prefer extreme rewards (both positive and negative)
        reward = abs(float(exp.get("reward", 0.0)))
        score += reward * 2.0

        # Explicit priority field
        priority = float(exp.get("priority", 1.0))
        score += priority

        # TD error (if available in info)
        info = exp.get("info", {})
        td_error = abs(float(info.get("td_error", 0.0)))
        score += td_error * 1.5

        # Diversity bonus: actions less frequently seen get a boost
        action = exp.get("action", "")
        action_count = self._diversity.action_counts.get(action, 0)
        if action_count == 0:
            score += self._config.diversity_bonus * 2.0
        elif self._diversity.total_experiences > 0:
            action_freq = action_count / self._diversity.total_experiences
            # Inverse frequency bonus
            score += self._config.diversity_bonus / max(action_freq, 0.01)

        # Novel state bonus
        state_hash = self._hash_state(exp.get("state", {}))
        if state_hash not in self._diversity.unique_state_hashes:
            score += self._config.diversity_bonus

        # Terminal state bonus (end-of-episode experiences are informative)
        if exp.get("done", False):
            score += 0.5

        return score

    # ------------------------------------------------------------------
    # Importance Sampling
    # ------------------------------------------------------------------

    def _annotate_importance_weights(
        self,
        selected: List[Dict[str, Any]],
        all_experiences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add importance-sampling weights for off-policy correction.

        When experiences are shared across nodes the receiving learner
        may have a different behaviour policy.  Importance weights allow
        compensating for this distribution shift.

        The weight is: w = P_target(a|s) / P_behaviour(a|s)
        Since we don't know the target policy of the receiver, we
        approximate with uniform correction based on selection probability.
        """
        n = len(all_experiences)
        k = len(selected)

        if n == 0 or k == 0:
            return selected

        # Selection probability for each experience
        selection_prob = k / n
        # Uniform policy probability
        uniform_prob = 1.0 / max(1, n)

        annotated: List[Dict[str, Any]] = []
        for exp in selected:
            exp_copy = dict(exp)
            # Importance weight clipped to configured range
            raw_weight = uniform_prob / max(selection_prob, 1e-8)
            clipped_weight = min(raw_weight, self._config.importance_clip_range)
            clipped_weight = max(clipped_weight, 1.0 / self._config.importance_clip_range)
            exp_copy["importance_weight"] = round(clipped_weight, 6)
            annotated.append(exp_copy)

        return annotated

    # ------------------------------------------------------------------
    # Deduplication & Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_experience(exp: Dict[str, Any]) -> str:
        """Compute a deterministic hash of an experience for dedup."""
        key_parts = [
            str(exp.get("id", "")),
            str(exp.get("action", "")),
            str(exp.get("reward", "")),
            str(exp.get("timestamp", "")),
        ]
        raw = "|".join(key_parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _hash_state(state: Any) -> str:
        """Hash a state dict for diversity tracking."""
        if isinstance(state, dict):
            raw = str(sorted(state.items()))
        else:
            raw = str(state)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _validate_experience(exp: Dict[str, Any]) -> bool:
        """Basic validation of an experience dict."""
        if not isinstance(exp, dict):
            return False
        if "action" not in exp:
            return False
        if "state" not in exp and "reward" not in exp:
            return False
        return True

    @staticmethod
    def _default_priority_fn(exp: Dict[str, Any]) -> float:
        """Default priority function based on reward magnitude."""
        reward = abs(float(exp.get("reward", 0.0)))
        priority = float(exp.get("priority", 1.0))
        return reward + priority

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_node_id(self) -> str:
        try:
            return self._cluster.node_id
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diversity_metrics(self) -> Dict[str, Any]:
        """Return the current diversity metrics."""
        return self._diversity.to_dict()

    def get_stats(self) -> Dict[str, Any]:
        """Return experience sharing statistics for monitoring."""
        return {
            "shared_buffer_size": len(self._shared_buffer),
            "total_shared": self._total_shared,
            "total_received": self._total_received,
            "total_deduplicated": self._total_deduplicated,
            "total_broadcasts": self._total_broadcasts,
            "share_ratio": self._config.share_ratio,
            "diversity": self._diversity.to_dict(),
        }
