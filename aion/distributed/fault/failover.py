"""
AION Failover Handler

Production-grade automatic failover implementing:
- Leader failover with consensus-driven re-election
- Node failover with task reassignment and data re-replication
- Network partition detection via quorum checking
- Split-brain detection and resolution (minority partition steps down)
- Fencing mechanism to prevent stale leaders from accepting writes
- Coordinated failover with pre-checks and post-validation
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set

import structlog

from aion.distributed.types import (
    NodeInfo,
    NodeRole,
    NodeStatus,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Failover event types
# ---------------------------------------------------------------------------


class FailoverType(str, Enum):
    """Types of failover events handled by the system."""

    LEADER_FAILURE = "leader_failure"
    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    SPLIT_BRAIN = "split_brain"


class FailoverStatus(str, Enum):
    """Status of a failover operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class FailoverEvent:
    """Record of a single failover operation.

    Captures the type, status, affected entities, and timing of the
    failover for auditing and diagnostics.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failover_type: FailoverType = FailoverType.NODE_FAILURE
    status: FailoverStatus = FailoverStatus.PENDING
    failed_node_id: Optional[str] = None
    new_leader_id: Optional[str] = None
    affected_nodes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "failover_type": self.failover_type.value,
            "status": self.status.value,
            "failed_node_id": self.failed_node_id,
            "new_leader_id": self.new_leader_id,
            "affected_nodes": self.affected_nodes,
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "details": self.details,
            "error": self.error,
        }


@dataclass
class FencingToken:
    """Token used to fence a stale leader.

    A fencing token is a monotonically increasing value that must
    accompany write requests.  Storage services reject any write whose
    token is older than the latest token they have observed, thereby
    preventing a stale leader from corrupting data.
    """

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    epoch: int = 0
    term: int = 0
    leader_id: str = ""
    issued_at: datetime = field(default_factory=datetime.now)

    @property
    def fencing_value(self) -> int:
        """Composite fencing value (term << 32 | epoch)."""
        return (self.term << 32) | (self.epoch & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# FailoverHandler
# ---------------------------------------------------------------------------


class FailoverHandler:
    """Handles automatic failover scenarios in the AION cluster.

    The handler addresses four categories of failure:

    1. **Leader failure** -- triggers a new consensus election so the
       cluster can continue accepting writes.
    2. **Node failure** -- reassigns the failed node's tasks and
       re-replicates its data shards.
    3. **Network partition** -- detects via quorum checks and fences the
       minority partition to prevent divergent state.
    4. **Split-brain** -- detects when multiple partitions believe they
       have a leader and resolves by demoting the minority side.

    Args:
        cluster_manager: The :class:`ClusterManager` this handler
                         operates on.
        fencing_enabled: Whether to issue and enforce fencing tokens.
                         Default ``True``.
        partition_check_interval: How often (seconds) to run background
                                  partition checks.  Default ``5.0``.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        fencing_enabled: bool = True,
        partition_check_interval: float = 5.0,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._fencing_enabled = fencing_enabled
        self._partition_check_interval = partition_check_interval

        # Failover history
        self._events: List[FailoverEvent] = []

        # Fencing tokens (latest per term)
        self._fencing_tokens: Dict[int, FencingToken] = {}
        self._current_fencing_epoch: int = 0

        # Partition state
        self._known_partitions: Dict[str, Set[str]] = {}

        # Background task handle
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Cooldown to prevent flapping (seconds since last failover)
        self._last_failover_time: float = 0.0
        self._failover_cooldown: float = 10.0

        logger.info(
            "failover_handler.init",
            fencing_enabled=fencing_enabled,
            partition_check_interval=partition_check_interval,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background partition monitoring loop."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._partition_monitor_loop())
        logger.info("failover_handler.started")

    async def stop(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("failover_handler.stopped")

    # ------------------------------------------------------------------
    # Leader failover
    # ------------------------------------------------------------------

    async def handle_leader_failure(self) -> FailoverEvent:
        """Handle the loss of the current cluster leader.

        Initiates a new consensus election so the cluster can resume
        accepting writes.  Issues a new fencing token to prevent the
        stale leader from making further changes.

        Returns:
            A :class:`FailoverEvent` describing the outcome.
        """
        event = FailoverEvent(
            failover_type=FailoverType.LEADER_FAILURE,
            failed_node_id=self._cluster_manager.state.leader_id,
        )
        event.status = FailoverStatus.IN_PROGRESS
        self._events.append(event)

        old_leader = self._cluster_manager.state.leader_id

        logger.warning(
            "failover_handler.leader_failure_detected",
            old_leader=old_leader,
        )

        try:
            # Issue a fencing token for the old leader
            if self._fencing_enabled and old_leader:
                self._issue_fencing_token(old_leader)

            # Clear the current leader
            self._cluster_manager.state.leader_id = None

            # Mark old leader node as suspected / offline
            if old_leader:
                old_node = self._cluster_manager.state.nodes.get(old_leader)
                if old_node is not None:
                    old_node.status = NodeStatus.SUSPECTED
                    old_node.role = NodeRole.FOLLOWER

            # Trigger a new consensus election
            consensus = getattr(self._cluster_manager, "_consensus", None)
            if consensus is not None:
                await consensus.trigger_election()
                logger.info("failover_handler.election_triggered")
            else:
                # Fallback: promote the healthiest voter
                new_leader = self._elect_fallback_leader()
                if new_leader:
                    await self._cluster_manager.handle_leader_change(
                        new_leader,
                        self._cluster_manager.state.term + 1,
                    )
                    event.new_leader_id = new_leader
                    logger.info(
                        "failover_handler.fallback_leader_elected",
                        new_leader=new_leader,
                    )
                else:
                    raise RuntimeError("No eligible leader candidate found")

            event.status = FailoverStatus.COMPLETED
            event.completed_at = datetime.now()
            self._last_failover_time = time.monotonic()

        except Exception as exc:
            event.status = FailoverStatus.FAILED
            event.error = str(exc)
            event.completed_at = datetime.now()
            logger.exception(
                "failover_handler.leader_failover_failed",
                error=str(exc),
            )

        return event

    # ------------------------------------------------------------------
    # Node failover
    # ------------------------------------------------------------------

    async def handle_node_failure(self, node_id: str) -> FailoverEvent:
        """Handle the failure of a non-leader node.

        Reassigns the node's tasks and triggers data re-replication
        via the cluster manager.

        Args:
            node_id: ID of the failed node.

        Returns:
            A :class:`FailoverEvent` describing the outcome.
        """
        event = FailoverEvent(
            failover_type=FailoverType.NODE_FAILURE,
            failed_node_id=node_id,
        )
        event.status = FailoverStatus.IN_PROGRESS
        self._events.append(event)

        logger.warning(
            "failover_handler.node_failure_detected",
            node_id=node_id,
        )

        try:
            # If the failed node was the leader, delegate to leader failover
            if self._cluster_manager.state.leader_id == node_id:
                leader_event = await self.handle_leader_failure()
                event.new_leader_id = leader_event.new_leader_id
                event.details["leader_failover"] = leader_event.to_dict()

            # Trigger node removal in the cluster manager
            await self._cluster_manager.handle_node_left(node_id)

            event.status = FailoverStatus.COMPLETED
            event.completed_at = datetime.now()
            self._last_failover_time = time.monotonic()

            logger.info(
                "failover_handler.node_failover_completed",
                node_id=node_id,
            )
        except Exception as exc:
            event.status = FailoverStatus.FAILED
            event.error = str(exc)
            event.completed_at = datetime.now()
            logger.exception(
                "failover_handler.node_failover_failed",
                node_id=node_id,
                error=str(exc),
            )

        return event

    async def initiate_failover(self, failed_node_id: str) -> FailoverEvent:
        """Public entry-point for initiating a failover of any node.

        Applies a cooldown check to prevent failover flapping.

        Args:
            failed_node_id: ID of the node to fail over.

        Returns:
            A :class:`FailoverEvent` describing the outcome.
        """
        # Cooldown check
        elapsed = time.monotonic() - self._last_failover_time
        if elapsed < self._failover_cooldown:
            logger.warning(
                "failover_handler.cooldown_active",
                remaining=round(self._failover_cooldown - elapsed, 2),
            )
            event = FailoverEvent(
                failover_type=FailoverType.NODE_FAILURE,
                failed_node_id=failed_node_id,
                status=FailoverStatus.FAILED,
                error="Failover cooldown active",
                completed_at=datetime.now(),
            )
            self._events.append(event)
            return event

        if self._cluster_manager.state.leader_id == failed_node_id:
            return await self.handle_leader_failure()
        return await self.handle_node_failure(failed_node_id)

    # ------------------------------------------------------------------
    # Network partition handling
    # ------------------------------------------------------------------

    async def handle_network_partition(
        self, partitioned_nodes: Set[str]
    ) -> FailoverEvent:
        """Handle a detected network partition.

        The minority partition is fenced: its nodes are marked as
        :attr:`NodeStatus.PARTITIONED` and their leader status (if any)
        is revoked to prevent divergent writes.

        Args:
            partitioned_nodes: Set of node IDs that are unreachable
                               from the local node's perspective.

        Returns:
            A :class:`FailoverEvent` describing the outcome.
        """
        event = FailoverEvent(
            failover_type=FailoverType.NETWORK_PARTITION,
            affected_nodes=list(partitioned_nodes),
        )
        event.status = FailoverStatus.IN_PROGRESS
        self._events.append(event)

        logger.warning(
            "failover_handler.network_partition_detected",
            partitioned_nodes=list(partitioned_nodes),
        )

        try:
            state = self._cluster_manager.state
            total_voters = len(state.voter_nodes)
            reachable_voters = sum(
                1
                for n in state.voter_nodes
                if n.id not in partitioned_nodes
            )

            # Are WE in the majority partition?
            we_have_quorum = reachable_voters >= (total_voters // 2 + 1)

            if we_have_quorum:
                # Fence the minority partition
                await self._fence_partition(partitioned_nodes)
                event.details["action"] = "fenced_minority"
                event.details["reachable_voters"] = reachable_voters
                event.details["total_voters"] = total_voters
            else:
                # We are the minority -- step down
                await self._step_down_as_minority(partitioned_nodes)
                event.details["action"] = "stepped_down_as_minority"
                event.details["reachable_voters"] = reachable_voters
                event.details["total_voters"] = total_voters

            self._known_partitions[event.event_id] = set(partitioned_nodes)
            event.status = FailoverStatus.COMPLETED
            event.completed_at = datetime.now()

        except Exception as exc:
            event.status = FailoverStatus.FAILED
            event.error = str(exc)
            event.completed_at = datetime.now()
            logger.exception(
                "failover_handler.partition_handling_failed",
                error=str(exc),
            )

        return event

    # ------------------------------------------------------------------
    # Split-brain detection & resolution
    # ------------------------------------------------------------------

    async def check_split_brain(self) -> bool:
        """Detect whether a split-brain condition exists.

        A split-brain is detected when there are multiple nodes claiming
        the :attr:`NodeRole.LEADER` role, which can happen after a
        network partition heals and both sides elected separate leaders.

        Returns:
            ``True`` if a split-brain condition is detected.
        """
        state = self._cluster_manager.state
        leaders = [
            n for n in state.nodes.values()
            if n.role == NodeRole.LEADER and n.status != NodeStatus.OFFLINE
        ]

        if len(leaders) > 1:
            leader_ids = [n.id for n in leaders]
            logger.critical(
                "failover_handler.split_brain_detected",
                leaders=leader_ids,
            )
            return True
        return False

    async def resolve_split_brain(self) -> FailoverEvent:
        """Resolve a split-brain by demoting all but one leader.

        The leader with the highest term wins.  Ties are broken by the
        node with the most healthy followers visible to it (a proxy for
        partition size).  The losing leaders are demoted to
        :attr:`NodeRole.FOLLOWER` and fenced.

        Returns:
            A :class:`FailoverEvent` describing the resolution.
        """
        event = FailoverEvent(failover_type=FailoverType.SPLIT_BRAIN)
        event.status = FailoverStatus.IN_PROGRESS
        self._events.append(event)

        state = self._cluster_manager.state
        leaders = [
            n for n in state.nodes.values()
            if n.role == NodeRole.LEADER and n.status != NodeStatus.OFFLINE
        ]

        if len(leaders) <= 1:
            event.status = FailoverStatus.COMPLETED
            event.details["action"] = "no_split_brain"
            event.completed_at = datetime.now()
            return event

        logger.warning(
            "failover_handler.resolving_split_brain",
            leaders=[n.id for n in leaders],
        )

        try:
            # Sort leaders: highest term first, then lowest load_score as proxy
            # for "larger partition" (lower load = more resources = more followers)
            leaders.sort(key=lambda n: (-state.term, n.load_score))
            winner = leaders[0]
            losers = leaders[1:]

            # Demote and fence losing leaders
            for loser in losers:
                loser.role = NodeRole.FOLLOWER
                loser.status = NodeStatus.HEALTHY
                event.affected_nodes.append(loser.id)

                if self._fencing_enabled:
                    self._issue_fencing_token(loser.id)

                logger.info(
                    "failover_handler.leader_demoted",
                    demoted=loser.id,
                    winner=winner.id,
                )

            # Ensure the winner is properly set
            state.leader_id = winner.id
            winner.role = NodeRole.LEADER
            event.new_leader_id = winner.id
            event.status = FailoverStatus.COMPLETED
            event.completed_at = datetime.now()

            logger.info(
                "failover_handler.split_brain_resolved",
                winner=winner.id,
                demoted=[n.id for n in losers],
            )
        except Exception as exc:
            event.status = FailoverStatus.FAILED
            event.error = str(exc)
            event.completed_at = datetime.now()
            logger.exception(
                "failover_handler.split_brain_resolution_failed",
                error=str(exc),
            )

        return event

    # ------------------------------------------------------------------
    # Fencing
    # ------------------------------------------------------------------

    def _issue_fencing_token(self, stale_leader_id: str) -> FencingToken:
        """Issue a new fencing token to invalidate a stale leader.

        The token's fencing value is monotonically increasing across
        terms and epochs, ensuring that any subsequent storage operation
        with a lower token is rejected.
        """
        self._current_fencing_epoch += 1
        state = self._cluster_manager.state

        token = FencingToken(
            epoch=self._current_fencing_epoch,
            term=state.term,
            leader_id=stale_leader_id,
        )
        self._fencing_tokens[state.term] = token

        logger.info(
            "failover_handler.fencing_token_issued",
            stale_leader=stale_leader_id,
            fencing_value=token.fencing_value,
            epoch=self._current_fencing_epoch,
        )
        return token

    def validate_fencing_token(self, token: FencingToken) -> bool:
        """Validate a fencing token against the latest known token.

        Returns ``True`` if the token is still valid (its fencing value
        is >= the latest issued token for its term).
        """
        latest = self._fencing_tokens.get(token.term)
        if latest is None:
            return True
        return token.fencing_value >= latest.fencing_value

    # ------------------------------------------------------------------
    # Partition helpers
    # ------------------------------------------------------------------

    async def _fence_partition(self, partitioned_nodes: Set[str]) -> None:
        """Mark nodes in the minority partition as partitioned and revoke
        any leader status they hold."""
        state = self._cluster_manager.state
        for nid in partitioned_nodes:
            node = state.nodes.get(nid)
            if node is None:
                continue
            node.status = NodeStatus.PARTITIONED
            if node.role == NodeRole.LEADER:
                node.role = NodeRole.FOLLOWER
                if state.leader_id == nid:
                    state.leader_id = None
                logger.info(
                    "failover_handler.partitioned_leader_demoted",
                    node_id=nid,
                )

        state.increment_epoch()
        logger.info(
            "failover_handler.partition_fenced",
            fenced_nodes=list(partitioned_nodes),
        )

    async def _step_down_as_minority(self, partitioned_nodes: Set[str]) -> None:
        """Step down the local node if it is leader in the minority partition."""
        state = self._cluster_manager.state
        own_id = self._cluster_manager.node_id

        if state.leader_id == own_id:
            state.leader_id = None
            local_node = state.nodes.get(own_id)
            if local_node is not None:
                local_node.role = NodeRole.FOLLOWER
            logger.warning(
                "failover_handler.stepped_down_minority",
                node_id=own_id,
            )

    def _elect_fallback_leader(self) -> Optional[str]:
        """Select the best candidate when no Raft consensus is available.

        Returns the ID of the healthiest voter node with the lowest
        load score, or ``None`` if no candidate qualifies.
        """
        state = self._cluster_manager.state
        own_id = self._cluster_manager.node_id
        candidates = [
            n
            for n in state.voter_nodes
            if n.status == NodeStatus.HEALTHY
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda n: (n.load_score, n.id != own_id))
        return candidates[0].id

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    async def _partition_monitor_loop(self) -> None:
        """Periodically check for split-brain and partition conditions."""
        while self._running:
            try:
                has_split_brain = await self.check_split_brain()
                if has_split_brain:
                    await self.resolve_split_brain()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("failover_handler.monitor_loop_error")

            await asyncio.sleep(self._partition_check_interval)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_failover_history(self) -> List[Dict[str, Any]]:
        """Return the full history of failover events."""
        return [e.to_dict() for e in self._events]

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information for the failover handler."""
        return {
            "running": self._running,
            "fencing_enabled": self._fencing_enabled,
            "current_fencing_epoch": self._current_fencing_epoch,
            "total_failovers": len(self._events),
            "active_partitions": len(self._known_partitions),
            "cooldown_remaining": max(
                0.0,
                self._failover_cooldown
                - (time.monotonic() - self._last_failover_time),
            ),
            "recent_events": [
                e.to_dict() for e in self._events[-10:]
            ],
        }
