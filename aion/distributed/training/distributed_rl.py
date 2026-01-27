"""
AION Distributed Reinforcement Learning Trainer

Production-grade distributed RL training coordinator that implements
the parameter server pattern for multi-node gradient aggregation.
Supports both synchronous and asynchronous gradient updates with
configurable sync intervals, gradient buffering, and automatic
leader/follower role handling via the cluster manager.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from aion.distributed.types import DistributedTask, NodeRole, TaskPriority, TaskType

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager
    from aion.systems.agents.learning.reinforcement import (
        Experience,
        Policy,
        ReinforcementLearner,
    )

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DistributedRLConfig:
    """Configuration for distributed RL training."""

    sync_interval_seconds: float = 60.0
    gradient_buffer_size: int = 64
    sync_mode: str = "periodic"  # "sync", "async", "periodic"
    learning_rate: float = 0.01
    momentum: float = 0.9
    max_gradient_norm: float = 1.0
    warmup_steps: int = 100
    parameter_broadcast_interval: int = 10
    staleness_threshold: int = 5
    enable_gradient_compression: bool = True
    compression_ratio: float = 0.1


# ---------------------------------------------------------------------------
# DistributedRLTrainer
# ---------------------------------------------------------------------------


class DistributedRLTrainer:
    """
    Distributed reinforcement learning training coordinator.

    Implements a parameter server pattern where the cluster leader
    maintains the global model, aggregates gradients from all nodes,
    and broadcasts updated parameters back. Follower nodes compute
    local gradients from their experiences and send them to the leader.

    Features:
        - Synchronous and asynchronous gradient aggregation
        - Leader-based parameter server pattern
        - Gradient buffering and batching for efficiency
        - Configurable sync intervals with warmup schedule
        - Automatic role detection via cluster manager
        - Staleness-aware gradient weighting
        - Momentum-based gradient accumulation
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        rl_learner: Optional[ReinforcementLearner] = None,
        config: Optional[DistributedRLConfig] = None,
    ) -> None:
        self._cluster = cluster_manager
        self._rl_learner = rl_learner
        self._config = config or DistributedRLConfig()

        # Gradient buffers
        self._gradient_buffer: List[Dict[str, Any]] = []
        self._momentum_buffer: Dict[str, Any] = {}
        self._global_parameters: Dict[str, Any] = {}

        # Synchronisation state
        self._sync_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._sync_round: int = 0
        self._last_sync_time: float = 0.0
        self._step_count: int = 0

        # Staleness tracking per node
        self._node_gradient_versions: Dict[str, int] = {}

        # Metrics
        self._total_gradients_sent: int = 0
        self._total_gradients_received: int = 0
        self._total_syncs: int = 0
        self._avg_sync_latency_ms: float = 0.0

        logger.info(
            "distributed_rl_trainer_created",
            sync_mode=self._config.sync_mode,
            sync_interval=self._config.sync_interval_seconds,
            has_learner=rl_learner is not None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the distributed RL training loop.

        On the leader node this launches the aggregation loop.
        On follower nodes this launches the gradient-send loop.
        """
        if self._running:
            logger.warning("distributed_rl_already_running")
            return

        self._running = True
        self._last_sync_time = time.monotonic()
        self._global_parameters = self._get_parameters()

        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            "distributed_rl_started",
            sync_mode=self._config.sync_mode,
        )

    async def stop(self) -> None:
        """Stop the distributed RL training loop gracefully."""
        if not self._running:
            return

        self._running = False

        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        logger.info(
            "distributed_rl_stopped",
            total_syncs=self._total_syncs,
            total_gradients_sent=self._total_gradients_sent,
            total_gradients_received=self._total_gradients_received,
        )

    # ------------------------------------------------------------------
    # Main sync loop
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Main synchronisation loop running at the configured interval.

        The leader aggregates gradients and broadcasts parameters.
        Followers compute and send local gradients to the leader.
        """
        logger.info("sync_loop_started", interval=self._config.sync_interval_seconds)

        try:
            while self._running:
                await asyncio.sleep(self._config.sync_interval_seconds)

                if not self._running:
                    break

                start = time.monotonic()

                try:
                    is_leader = self._is_leader()

                    if is_leader:
                        await self._aggregate_and_broadcast()
                    else:
                        await self._send_gradients_to_leader()

                    self._sync_round += 1
                    self._total_syncs += 1
                    elapsed_ms = (time.monotonic() - start) * 1000.0
                    self._avg_sync_latency_ms = (
                        self._avg_sync_latency_ms * 0.9 + elapsed_ms * 0.1
                    )

                    logger.debug(
                        "sync_round_completed",
                        round=self._sync_round,
                        is_leader=is_leader,
                        elapsed_ms=round(elapsed_ms, 2),
                    )
                except Exception:
                    logger.exception("sync_round_error", round=self._sync_round)

        except asyncio.CancelledError:
            logger.debug("sync_loop_cancelled")

    # ------------------------------------------------------------------
    # Leader operations
    # ------------------------------------------------------------------

    async def _aggregate_and_broadcast(self) -> None:
        """Leader: aggregate buffered gradients and broadcast new parameters.

        Collects gradients from all nodes, averages them with optional
        staleness weighting, applies momentum, clips by global norm,
        and then updates the global parameters that are broadcast to
        every follower in the cluster.
        """
        if not self._gradient_buffer:
            logger.debug("no_gradients_to_aggregate")
            return

        all_gradients = list(self._gradient_buffer)
        self._gradient_buffer.clear()

        # Weight gradients by staleness
        weighted_gradients: List[Dict[str, Any]] = []
        for grad_entry in all_gradients:
            node_id = grad_entry.get("node_id", "unknown")
            version = grad_entry.get("version", self._sync_round)
            staleness = max(0, self._sync_round - version)

            # Down-weight stale gradients
            if staleness > self._config.staleness_threshold:
                logger.debug(
                    "discarding_stale_gradient",
                    node_id=node_id,
                    staleness=staleness,
                )
                continue

            weight = 1.0 / (1.0 + staleness)
            gradients = grad_entry.get("gradients", {})
            weighted = {
                k: self._scale_value(v, weight) for k, v in gradients.items()
            }
            weighted_gradients.append(weighted)

        if not weighted_gradients:
            return

        # Average the gradients
        averaged = self._average_gradients(weighted_gradients)

        # Apply momentum
        averaged = self._apply_momentum(averaged)

        # Apply gradient clipping
        if self._config.max_gradient_norm > 0:
            averaged = self._clip_gradients(averaged, self._config.max_gradient_norm)

        # Apply gradients to the global model
        self._apply_gradients(averaged)

        # Broadcast updated parameters to followers
        await self._broadcast_parameters()

        self._total_gradients_received += len(all_gradients)
        logger.info(
            "gradients_aggregated",
            num_gradients=len(all_gradients),
            num_accepted=len(weighted_gradients),
            round=self._sync_round,
        )

    async def _broadcast_parameters(self) -> None:
        """Broadcast global model parameters to all follower nodes."""
        params = self._get_parameters()
        task = DistributedTask(
            name="parameter_broadcast",
            task_type=TaskType.TRAINING_STEP.value,
            priority=TaskPriority.HIGH,
            payload={
                "action": "update_parameters",
                "parameters": params,
                "round": self._sync_round,
                "timestamp": time.time(),
            },
        )
        try:
            await self._cluster.submit_task(task)
            logger.debug(
                "parameters_broadcast",
                round=self._sync_round,
                param_count=len(params),
            )
        except Exception:
            logger.exception("parameter_broadcast_failed")

    # ------------------------------------------------------------------
    # Follower operations
    # ------------------------------------------------------------------

    async def _send_gradients_to_leader(self) -> None:
        """Follower: compute local gradients and send to the leader."""
        gradients = self._get_local_gradients()

        if not gradients:
            return

        self._step_count += 1
        node_id = self._get_node_id()

        task = DistributedTask(
            name="gradient_upload",
            task_type=TaskType.TRAINING_STEP.value,
            priority=TaskPriority.NORMAL,
            payload={
                "action": "submit_gradients",
                "node_id": node_id,
                "gradients": gradients,
                "version": self._sync_round,
                "step": self._step_count,
                "timestamp": time.time(),
            },
        )

        try:
            await self._cluster.submit_task(task)
            self._total_gradients_sent += 1
            logger.debug(
                "gradients_sent",
                step=self._step_count,
                param_count=len(gradients),
            )
        except Exception:
            logger.exception("gradient_send_failed", step=self._step_count)

    # ------------------------------------------------------------------
    # Gradient computation helpers
    # ------------------------------------------------------------------

    def _get_local_gradients(self) -> Dict[str, Any]:
        """Compute local gradients from the RL learner's recent experiences.

        Returns a dictionary mapping parameter names to gradient values.
        If no learner is configured, returns an empty dict.
        """
        if self._rl_learner is None:
            return {}

        gradients: Dict[str, Any] = {}
        try:
            policy = self._rl_learner.get_policy()
            value_fn = self._rl_learner.get_value_function()

            # Policy gradients: difference from global parameters
            for action, pref in policy.action_preferences.items():
                global_pref = self._global_parameters.get(
                    f"policy.{action}", 0.0,
                )
                gradients[f"policy.{action}"] = pref - global_pref

            # Value function gradients
            for feature, weight in value_fn.feature_weights.items():
                global_weight = self._global_parameters.get(
                    f"value.{feature}", 0.0,
                )
                gradients[f"value.{feature}"] = weight - global_weight

        except Exception:
            logger.exception("local_gradient_computation_error")

        return gradients

    def _average_gradients(
        self, all_gradients: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Average a list of gradient dictionaries element-wise.

        Args:
            all_gradients: List of gradient dicts from different nodes.

        Returns:
            A single dict with each key mapped to the mean gradient value.
        """
        if not all_gradients:
            return {}

        accumulated: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for grad_dict in all_gradients:
            for key, value in grad_dict.items():
                numeric = self._to_float(value)
                if numeric is not None:
                    accumulated[key] = accumulated.get(key, 0.0) + numeric
                    counts[key] = counts.get(key, 0) + 1

        averaged: Dict[str, Any] = {}
        for key in accumulated:
            cnt = counts.get(key, 1)
            averaged[key] = accumulated[key] / cnt if cnt > 0 else 0.0

        return averaged

    def _apply_momentum(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Apply momentum to the gradient update."""
        momentum = self._config.momentum
        result: Dict[str, Any] = {}

        for key, value in gradients.items():
            numeric = self._to_float(value)
            if numeric is None:
                continue
            prev = self._to_float(self._momentum_buffer.get(key, 0.0)) or 0.0
            updated = momentum * prev + (1.0 - momentum) * numeric
            self._momentum_buffer[key] = updated
            result[key] = updated

        return result

    def _clip_gradients(
        self, gradients: Dict[str, Any], max_norm: float
    ) -> Dict[str, Any]:
        """Clip gradients by global L2 norm."""
        total_norm_sq = 0.0
        for value in gradients.values():
            numeric = self._to_float(value)
            if numeric is not None:
                total_norm_sq += numeric * numeric

        total_norm = total_norm_sq ** 0.5

        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / total_norm
            return {
                k: self._scale_value(v, scale) for k, v in gradients.items()
            }

        return gradients

    def _apply_gradients(self, gradients: Dict[str, Any]) -> None:
        """Apply averaged gradients to the global model parameters.

        Updates both the global parameter dict and, if available,
        the local RL learner's policy and value function.
        """
        lr = self._config.learning_rate

        # Warmup schedule
        if self._step_count < self._config.warmup_steps:
            lr = lr * (self._step_count + 1) / self._config.warmup_steps

        for key, grad_value in gradients.items():
            numeric = self._to_float(grad_value)
            if numeric is None:
                continue

            current = self._to_float(self._global_parameters.get(key, 0.0)) or 0.0
            self._global_parameters[key] = current - lr * numeric

        # Push parameters to local RL learner
        if self._rl_learner is not None:
            self._update_local_learner()

    def _update_local_learner(self) -> None:
        """Synchronise global parameters back into the local RL learner."""
        if self._rl_learner is None:
            return

        try:
            policy = self._rl_learner.get_policy()
            value_fn = self._rl_learner.get_value_function()

            for key, value in self._global_parameters.items():
                numeric = self._to_float(value)
                if numeric is None:
                    continue

                if key.startswith("policy."):
                    action = key[len("policy."):]
                    policy.action_preferences[action] = numeric
                elif key.startswith("value."):
                    feature = key[len("value."):]
                    value_fn.feature_weights[feature] = numeric
        except Exception:
            logger.exception("local_learner_update_error")

    def _get_parameters(self) -> Dict[str, Any]:
        """Extract current model parameters from the RL learner.

        Returns a flat dict mapping dotted parameter names to float values.
        """
        params: Dict[str, Any] = {}

        if self._rl_learner is None:
            return dict(self._global_parameters)

        try:
            policy = self._rl_learner.get_policy()
            value_fn = self._rl_learner.get_value_function()

            for action, pref in policy.action_preferences.items():
                params[f"policy.{action}"] = pref

            for feature, weight in value_fn.feature_weights.items():
                params[f"value.{feature}"] = weight
        except Exception:
            logger.exception("parameter_extraction_error")

        return params

    # ------------------------------------------------------------------
    # Receive interface (called by task handlers)
    # ------------------------------------------------------------------

    async def receive_gradients(self, payload: Dict[str, Any]) -> None:
        """Receive gradients from a follower node (leader only).

        This is intended to be called from the distributed task handler
        when a gradient_upload task arrives.
        """
        self._gradient_buffer.append(payload)

        node_id = payload.get("node_id", "unknown")
        version = payload.get("version", 0)
        self._node_gradient_versions[node_id] = version

        logger.debug(
            "gradient_received",
            node_id=node_id,
            version=version,
            buffer_size=len(self._gradient_buffer),
        )

        # In sync mode, aggregate immediately when all nodes reported
        if self._config.sync_mode == "sync":
            expected = self._get_cluster_size()
            if len(self._gradient_buffer) >= expected:
                await self._aggregate_and_broadcast()

    async def receive_parameters(self, payload: Dict[str, Any]) -> None:
        """Receive updated parameters from the leader (follower only).

        This is intended to be called from the distributed task handler
        when a parameter_broadcast task arrives.
        """
        parameters = payload.get("parameters", {})
        round_num = payload.get("round", 0)

        self._global_parameters.update(parameters)
        self._sync_round = round_num

        # Push to local learner
        self._update_local_learner()

        logger.debug(
            "parameters_received",
            round=round_num,
            param_count=len(parameters),
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _is_leader(self) -> bool:
        """Check whether this node is the cluster leader."""
        try:
            state = self._cluster.get_state()
            if state is not None:
                node_id = self._get_node_id()
                return state.leader_id == node_id
        except Exception:
            pass
        return False

    def _get_node_id(self) -> str:
        """Return this node's identifier."""
        try:
            return self._cluster.node_id
        except Exception:
            return "unknown"

    def _get_cluster_size(self) -> int:
        """Return the number of nodes in the cluster."""
        try:
            state = self._cluster.get_state()
            if state is not None:
                return len(state.nodes)
        except Exception:
            pass
        return 1

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Safely convert a value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _scale_value(value: Any, scale: float) -> Any:
        """Scale a numeric value, returning unchanged if not numeric."""
        if isinstance(value, (int, float)):
            return float(value) * scale
        return value

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return training statistics for monitoring."""
        return {
            "running": self._running,
            "sync_round": self._sync_round,
            "step_count": self._step_count,
            "sync_mode": self._config.sync_mode,
            "gradient_buffer_size": len(self._gradient_buffer),
            "global_param_count": len(self._global_parameters),
            "total_syncs": self._total_syncs,
            "total_gradients_sent": self._total_gradients_sent,
            "total_gradients_received": self._total_gradients_received,
            "avg_sync_latency_ms": round(self._avg_sync_latency_ms, 2),
            "is_leader": self._is_leader(),
        }
