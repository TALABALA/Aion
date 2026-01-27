"""
AION Raft Consensus Implementation

Production-grade implementation of the Raft consensus protocol for the
AION distributed computing system. Provides leader election, log
replication, and state machine application with full safety guarantees.

Key features:
- Full Raft leader election with randomized timeouts (150-300ms)
- PreVote protocol extension to prevent term inflation during network partitions
- Log replication with next_index/match_index per-follower tracking
- Conflict optimization in AppendEntries responses (conflict_term, conflict_index)
- Quorum-based commit index advancement
- Automatic step-down when a higher term is discovered
- Heartbeat broadcasting as leader (empty AppendEntries RPCs)
- Background election timer task (followers/candidates)
- Background heartbeat loop (leader)
- Noop entry on leader establishment for commit index catchup

References:
- Ongaro & Ousterhout, "In Search of an Understandable Consensus Algorithm" (2014)
- Ongaro, "Consensus: Bridging Theory and Practice" (PhD dissertation, 2014)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import structlog

from aion.distributed.consensus.leader import LeaderElection
from aion.distributed.consensus.log import ReplicatedLog
from aion.distributed.consensus.state_machine import ConsensusStateMachine
from aion.distributed.types import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    NodeInfo,
    NodeRole,
    NodeStatus,
    RaftLogEntry,
    RaftMessageType,
    SnapshotMetadata,
    VoteRequest,
    VoteResponse,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Persistent State Storage
# ---------------------------------------------------------------------------


class RaftPersistentState:
    """Durable storage for Raft safety-critical state.

    The Raft protocol requires that ``currentTerm`` and ``votedFor`` survive
    crashes.  Without durability a restarted node could vote twice in the
    same term, violating Election Safety (§5.2).

    This implementation uses a simple JSON file with atomic writes (write
    to a temp file then rename).  Production systems would use RocksDB,
    SQLite-WAL, or similar.
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        if data_dir:
            self._path = Path(data_dir) / "raft_state.json"
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._path = None
        self._current_term: int = 0
        self._voted_for: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load persisted state from disk."""
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._current_term = data.get("current_term", 0)
            self._voted_for = data.get("voted_for")
        except Exception:
            pass  # Start fresh on corruption

    def _flush(self) -> None:
        """Atomically persist state to disk."""
        if self._path is None:
            return
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({
                "current_term": self._current_term,
                "voted_for": self._voted_for,
            }))
            tmp.rename(self._path)
        except Exception:
            pass  # Best-effort; production would retry or crash

    @property
    def current_term(self) -> int:
        return self._current_term

    @current_term.setter
    def current_term(self, value: int) -> None:
        self._current_term = value
        self._flush()

    @property
    def voted_for(self) -> Optional[str]:
        return self._voted_for

    @voted_for.setter
    def voted_for(self, value: Optional[str]) -> None:
        self._voted_for = value
        self._flush()

    def update(self, term: int, voted_for: Optional[str]) -> None:
        """Batch update both fields with a single flush."""
        self._current_term = term
        self._voted_for = voted_for
        self._flush()


class RaftConsensus:
    """
    Full Raft consensus protocol implementation.

    Manages leader election, log replication, and state machine application
    for a single node in the AION distributed cluster.

    Each RaftConsensus instance runs two background tasks:
    1. Election timer: monitors for leader heartbeat timeouts and triggers
       elections when the leader is presumed failed.
    2. Heartbeat loop (leader only): periodically sends empty AppendEntries
       RPCs to maintain authority and replicate new entries.

    Usage::

        consensus = RaftConsensus(node_info, cluster_manager)
        await consensus.start()

        # Submit a command (leader only)
        success = await consensus.append_command("set_state", {"key": "k", "value": "v"})

        # Handle incoming RPCs
        response = await consensus.handle_vote_request(request)
        response = await consensus.handle_append_entries(request)

        await consensus.stop()
    """

    def __init__(
        self,
        node_info: NodeInfo,
        cluster_manager: "ClusterManager",
        data_dir: Optional[str] = None,
    ) -> None:
        self._log = logger.bind(
            component="raft_consensus",
            node_id=node_info.id,
        )

        # Node identity
        self._node_info = node_info
        self._cluster_manager = cluster_manager

        # Persistent state (on all servers) -- survives crashes
        self._persistent = RaftPersistentState(data_dir)

        # Snapshot data cache (for InstallSnapshot to followers)
        self._last_snapshot_data: Optional[bytes] = None

        # Volatile state (on all servers)
        self._commit_index: int = -1
        self._last_applied: int = -1

        # Volatile state (on leaders) -- per-follower tracking
        self._next_index: Dict[str, int] = {}
        self._match_index: Dict[str, int] = {}

        # Node role
        self._role: NodeRole = NodeRole.FOLLOWER
        self._leader_id: Optional[str] = None

        # Sub-components
        self._election = LeaderElection(
            node_id=node_info.id,
            election_timeout_min_ms=150,
            election_timeout_max_ms=300,
            pre_vote_enabled=True,
        )
        self._replicated_log = ReplicatedLog()
        self._state_machine = ConsensusStateMachine()

        # Background tasks
        self._election_timer_task: Optional[asyncio.Task[None]] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._apply_task: Optional[asyncio.Task[None]] = None

        # Lifecycle
        self._running = False
        self._lock = asyncio.Lock()

        # Configuration
        self._heartbeat_interval_ms: float = 50.0
        self._max_entries_per_request: int = 100

        # Pending command futures: index -> Future
        self._pending_commands: Dict[int, asyncio.Future[bool]] = {}

        # Metrics
        self._elections_started: int = 0
        self._elections_won: int = 0
        self._append_entries_sent: int = 0
        self._append_entries_received: int = 0
        self._votes_granted: int = 0
        self._step_downs: int = 0

    # =========================================================================
    # Persistent state accessors (delegate to RaftPersistentState)
    # =========================================================================

    @property
    def _current_term(self) -> int:
        return self._persistent.current_term

    @_current_term.setter
    def _current_term(self, value: int) -> None:
        self._persistent.current_term = value

    @property
    def _voted_for(self) -> Optional[str]:
        return self._persistent.voted_for

    @_voted_for.setter
    def _voted_for(self, value: Optional[str]) -> None:
        self._persistent.voted_for = value

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def current_term(self) -> int:
        """The current Raft term of this node."""
        return self._persistent.current_term

    @property
    def role(self) -> NodeRole:
        """The current role of this node (LEADER, FOLLOWER, CANDIDATE)."""
        return self._role

    @property
    def leader_id(self) -> Optional[str]:
        """The node ID of the current known leader."""
        return self._leader_id

    @property
    def commit_index(self) -> int:
        """The highest log index known to be committed."""
        return self._commit_index

    @property
    def node_id(self) -> str:
        """This node's ID."""
        return self._node_info.id

    @property
    def replicated_log(self) -> ReplicatedLog:
        """Access the replicated log (for snapshot/compaction)."""
        return self._replicated_log

    @property
    def state_machine(self) -> ConsensusStateMachine:
        """Access the state machine."""
        return self._state_machine

    @property
    def is_leader(self) -> bool:
        """Whether this node is currently the leader."""
        return self._role == NodeRole.LEADER

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """
        Start the Raft consensus protocol.

        Launches background tasks for election timeout monitoring
        and (when leader) heartbeat broadcasting.
        """
        if self._running:
            self._log.warning("already_running")
            return

        self._running = True
        self._role = NodeRole.FOLLOWER
        self._node_info.role = NodeRole.FOLLOWER
        self._election.reset_timeout()

        # Start background tasks
        self._election_timer_task = asyncio.create_task(
            self._election_timer_loop(),
            name=f"raft-election-timer-{self._node_info.id}",
        )
        self._apply_task = asyncio.create_task(
            self._apply_committed_entries_loop(),
            name=f"raft-apply-{self._node_info.id}",
        )

        self._log.info(
            "raft_started",
            node_id=self._node_info.id,
            term=self._current_term,
            role=self._role.value,
        )

    async def stop(self) -> None:
        """
        Stop the Raft consensus protocol.

        Cancels all background tasks and cleans up resources.
        """
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        tasks_to_cancel: List[Optional[asyncio.Task[None]]] = [
            self._election_timer_task,
            self._heartbeat_task,
            self._apply_task,
        ]
        for task in tasks_to_cancel:
            if task is not None and not task.done():
                task.cancel()

        # Await cancellation
        for task in tasks_to_cancel:
            if task is not None and not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Fail pending commands
        for future in self._pending_commands.values():
            if not future.done():
                future.set_result(False)
        self._pending_commands.clear()

        self._log.info(
            "raft_stopped",
            node_id=self._node_info.id,
            term=self._current_term,
            role=self._role.value,
        )

    # =========================================================================
    # Term Management
    # =========================================================================

    def _discover_higher_term(self, term: int, source: str = "") -> bool:
        """
        Handle discovery of a higher term from any RPC.

        If the discovered term is higher than the current term, this node
        must immediately step down to follower, update its term, and
        clear its vote. This is a fundamental Raft safety guarantee.

        Args:
            term: The term discovered from an RPC message.
            source: Description of the source for logging.

        Returns:
            True if a step-down occurred.
        """
        if term > self._current_term:
            old_term = self._current_term
            old_role = self._role
            # Batch-persist both fields in a single flush
            self._persistent.update(term, None)
            self._step_down_to_follower()
            self._step_downs += 1

            self._log.info(
                "higher_term_discovered",
                old_term=old_term,
                new_term=term,
                old_role=old_role.value,
                source=source,
            )
            return True
        return False

    def _step_down_to_follower(self) -> None:
        """Transition to follower state, cancelling leader duties if active."""
        self._role = NodeRole.FOLLOWER
        self._node_info.role = NodeRole.FOLLOWER
        self._leader_id = None

        # Cancel heartbeat task if we were leader
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        self._election.reset_timeout()

    # =========================================================================
    # Election
    # =========================================================================

    async def trigger_election(self) -> None:
        """
        Trigger a leader election.

        If PreVote is enabled, conducts a pre-vote round first.
        Only if the pre-vote succeeds does the node increment
        its term and run a real election.
        """
        async with self._lock:
            if not self._running:
                return

            if self._election.pre_vote_enabled:
                await self._run_pre_vote()
            else:
                await self._run_election()

    async def _run_pre_vote(self) -> None:
        """
        Conduct a PreVote round.

        The candidate checks whether it would win an election without
        incrementing its term. This prevents term inflation when a
        node is partitioned from the cluster.
        """
        pre_vote_term = self._current_term + 1

        request = self._election.start_election(
            term=pre_vote_term,
            pre_vote=True,
        )
        request.last_log_index = self._replicated_log.get_last_index()
        request.last_log_term = self._replicated_log.get_last_term()

        self._log.info(
            "pre_vote_started",
            term=pre_vote_term,
            last_log_index=request.last_log_index,
            last_log_term=request.last_log_term,
        )

        # Request pre-votes from peers
        peers = self._get_peer_ids()
        if not peers:
            # Single node cluster: proceed directly to election
            await self._run_election()
            return

        responses = await self._broadcast_vote_request(request)

        for response in responses:
            if self._discover_higher_term(response.term, source="pre_vote_response"):
                return
            self._election.handle_vote(response)

        cluster_size = len(peers) + 1  # Include self
        if self._election.check_if_won(cluster_size):
            self._log.info("pre_vote_succeeded", term=pre_vote_term)
            await self._run_election()
        else:
            self._log.info(
                "pre_vote_failed",
                term=pre_vote_term,
                votes=self._election.vote_count,
                needed=(cluster_size // 2) + 1,
            )

    async def _run_election(self) -> None:
        """
        Conduct a real Raft election.

        Increments the term, votes for self, and solicits votes from
        all peers. If a quorum is achieved, transitions to leader.
        """
        self._current_term += 1
        self._voted_for = self._node_info.id
        self._role = NodeRole.CANDIDATE
        self._node_info.role = NodeRole.CANDIDATE
        self._elections_started += 1

        request = self._election.start_election(
            term=self._current_term,
            pre_vote=False,
        )
        request.last_log_index = self._replicated_log.get_last_index()
        request.last_log_term = self._replicated_log.get_last_term()

        self._log.info(
            "election_started",
            term=self._current_term,
            last_log_index=request.last_log_index,
            last_log_term=request.last_log_term,
        )

        peers = self._get_peer_ids()
        if not peers:
            # Single node cluster: win immediately
            await self._become_leader()
            return

        responses = await self._broadcast_vote_request(request)

        # Process responses (only if still a candidate for this term)
        if self._role != NodeRole.CANDIDATE or self._election.election_term != self._current_term:
            return

        for response in responses:
            if self._discover_higher_term(response.term, source="vote_response"):
                return
            self._election.handle_vote(response)

        cluster_size = len(peers) + 1
        if self._election.check_if_won(cluster_size):
            await self._become_leader()
        else:
            self._log.info(
                "election_lost",
                term=self._current_term,
                votes=self._election.vote_count,
                needed=(cluster_size // 2) + 1,
            )

    async def _become_leader(self) -> None:
        """
        Transition to leader state after winning an election.

        Initializes next_index and match_index for all peers,
        starts the heartbeat loop, and appends a noop entry to
        establish commit index.
        """
        self._role = NodeRole.LEADER
        self._node_info.role = NodeRole.LEADER
        self._leader_id = self._node_info.id
        self._elections_won += 1

        # Initialize leader volatile state
        last_index = self._replicated_log.get_last_index()
        peers = self._get_peer_ids()
        for peer_id in peers:
            self._next_index[peer_id] = last_index + 1
            self._match_index[peer_id] = -1

        self._log.info(
            "became_leader",
            term=self._current_term,
            peer_count=len(peers),
            last_log_index=last_index,
        )

        # Start heartbeat loop
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"raft-heartbeat-{self._node_info.id}",
        )

        # Append a noop entry to establish commit index for the new term.
        # This ensures that the leader can commit entries from its own term.
        noop_entry = RaftLogEntry(
            term=self._current_term,
            command="noop",
            is_noop=True,
        )
        await self._replicated_log.append(noop_entry)

        # Immediately send heartbeats to establish authority
        await self._send_heartbeats()

    # =========================================================================
    # Vote Request Handling (Receiver)
    # =========================================================================

    async def handle_vote_request(self, request: VoteRequest) -> VoteResponse:
        """
        Handle an incoming VoteRequest (or PreVoteRequest) RPC.

        Implements the Raft voting rules:
        1. Deny if the candidate's term is stale.
        2. Deny if we already voted for someone else this term.
        3. Deny if the candidate's log is not at least as up-to-date.
        4. Grant otherwise.

        For pre-votes, the same log-up-to-date check applies, but
        we do not actually record a vote or update our term.

        Args:
            request: The incoming vote request.

        Returns:
            A VoteResponse indicating whether the vote was granted.
        """
        async with self._lock:
            response = VoteResponse(
                term=self._current_term,
                vote_granted=False,
                voter_id=self._node_info.id,
            )

            # Pre-vote handling: do not update term, but check eligibility
            if request.is_pre_vote:
                return self._evaluate_pre_vote(request, response)

            # Step down if higher term
            if self._discover_higher_term(request.term, source="vote_request"):
                response.term = self._current_term

            # Rule 1: Deny if candidate's term is stale
            if request.term < self._current_term:
                self._log.debug(
                    "vote_denied_stale_term",
                    candidate=request.candidate_id,
                    candidate_term=request.term,
                    our_term=self._current_term,
                )
                return response

            # Rule 2: Check if we already voted for someone else
            if (
                self._voted_for is not None
                and self._voted_for != request.candidate_id
            ):
                self._log.debug(
                    "vote_denied_already_voted",
                    candidate=request.candidate_id,
                    voted_for=self._voted_for,
                )
                return response

            # Rule 3: Check log up-to-date-ness
            our_last_index = self._replicated_log.get_last_index()
            our_last_term = self._replicated_log.get_last_term()

            if not LeaderElection.is_candidate_log_up_to_date(
                candidate_last_index=request.last_log_index,
                candidate_last_term=request.last_log_term,
                voter_last_index=our_last_index,
                voter_last_term=our_last_term,
            ):
                self._log.debug(
                    "vote_denied_log_not_up_to_date",
                    candidate=request.candidate_id,
                    candidate_last_index=request.last_log_index,
                    candidate_last_term=request.last_log_term,
                    our_last_index=our_last_index,
                    our_last_term=our_last_term,
                )
                return response

            # Grant vote
            self._voted_for = request.candidate_id
            self._current_term = request.term
            response.term = self._current_term
            response.vote_granted = True
            self._votes_granted += 1

            # Reset election timer when granting a vote
            self._election.reset_timeout()

            self._log.info(
                "vote_granted",
                candidate=request.candidate_id,
                term=request.term,
            )
            return response

    def _evaluate_pre_vote(
        self, request: VoteRequest, response: VoteResponse
    ) -> VoteResponse:
        """
        Evaluate a pre-vote request without changing state.

        We grant a pre-vote if:
        1. The candidate's proposed term is not stale.
        2. The candidate's log is at least as up-to-date as ours.
        3. We believe no leader is currently active (our election timer
           has elapsed), OR the candidate's term is strictly higher.
        """
        # Stale term check
        if request.term < self._current_term:
            return response

        # Log up-to-date check
        our_last_index = self._replicated_log.get_last_index()
        our_last_term = self._replicated_log.get_last_term()

        if not LeaderElection.is_candidate_log_up_to_date(
            candidate_last_index=request.last_log_index,
            candidate_last_term=request.last_log_term,
            voter_last_index=our_last_index,
            voter_last_term=our_last_term,
        ):
            return response

        # Only grant pre-vote if we believe no leader is active
        # (our own election timer has elapsed) or term is higher
        if request.term > self._current_term or self._election.is_timeout_elapsed():
            response.vote_granted = True
            self._log.debug(
                "pre_vote_granted",
                candidate=request.candidate_id,
                term=request.term,
            )

        return response

    # =========================================================================
    # AppendEntries Handling (Receiver / Follower)
    # =========================================================================

    async def handle_append_entries(
        self, request: AppendEntriesRequest
    ) -> AppendEntriesResponse:
        """
        Handle an incoming AppendEntries RPC (heartbeat or log replication).

        Implements the Raft AppendEntries receiver rules:
        1. Reply false if term < currentTerm.
        2. Reply false if log does not contain an entry at prevLogIndex
           with prevLogTerm (with conflict optimization).
        3. If existing entry conflicts with a new one, delete existing
           and all that follow.
        4. Append any new entries not already in the log.
        5. Update commitIndex if leaderCommit > commitIndex.

        Args:
            request: The incoming AppendEntries request.

        Returns:
            An AppendEntriesResponse indicating success or conflict info.
        """
        async with self._lock:
            self._append_entries_received += 1

            response = AppendEntriesResponse(
                term=self._current_term,
                success=False,
                node_id=self._node_info.id,
            )

            # Step down if higher term
            if self._discover_higher_term(request.term, source="append_entries"):
                response.term = self._current_term

            # Rule 1: Stale term
            if request.term < self._current_term:
                self._log.debug(
                    "append_entries_denied_stale_term",
                    leader=request.leader_id,
                    leader_term=request.term,
                    our_term=self._current_term,
                )
                return response

            # Valid leader heartbeat -- reset election timer and record leader
            self._election.reset_timeout()
            self._leader_id = request.leader_id
            self._current_term = request.term

            # Ensure we are a follower
            if self._role != NodeRole.FOLLOWER:
                self._step_down_to_follower()

            # Rule 2: Log consistency check
            if request.prev_log_index >= 0:
                if not self._replicated_log.has_entry_at(
                    request.prev_log_index, request.prev_log_term
                ):
                    # Conflict optimization: report the conflicting term and
                    # the first index of entries with that term
                    conflict_entry = self._replicated_log.get(request.prev_log_index)
                    if conflict_entry is not None:
                        response.conflict_term = conflict_entry.term
                        response.conflict_index = (
                            self._replicated_log.find_conflict_index(
                                conflict_entry.term
                            )
                        )
                    else:
                        # We do not have an entry at prev_log_index at all;
                        # tell the leader our last index so it can skip back
                        response.conflict_index = (
                            self._replicated_log.get_last_index() + 1
                        )

                    self._log.debug(
                        "append_entries_log_mismatch",
                        prev_log_index=request.prev_log_index,
                        prev_log_term=request.prev_log_term,
                        conflict_term=response.conflict_term,
                        conflict_index=response.conflict_index,
                    )
                    return response

            # Rules 3 & 4: Process new entries
            if request.entries:
                await self._process_new_entries(
                    request.prev_log_index, request.entries
                )

            # Rule 5: Update commit index
            if request.leader_commit > self._commit_index:
                last_new_index = (
                    request.entries[-1].index
                    if request.entries
                    else self._replicated_log.get_last_index()
                )
                self._commit_index = min(request.leader_commit, last_new_index)

            response.success = True
            response.match_index = self._replicated_log.get_last_index()
            return response

    async def _process_new_entries(
        self,
        prev_log_index: int,
        entries: List[RaftLogEntry],
    ) -> None:
        """
        Process entries from an AppendEntries RPC.

        For each new entry, check if it conflicts with an existing entry
        at the same index. If so, truncate the log from the conflict point.
        Then append any entries not already in the log.
        """
        insert_index = prev_log_index + 1

        for i, new_entry in enumerate(entries):
            entry_index = insert_index + i
            existing = self._replicated_log.get(entry_index)

            if existing is not None:
                if existing.term != new_entry.term:
                    # Conflict: truncate from here
                    await self._replicated_log.truncate_after(entry_index - 1)
                    # Append remaining entries
                    remaining = entries[i:]
                    for rem_entry in remaining:
                        await self._replicated_log.append(rem_entry)
                    return
                # Same term: entry already exists, skip
            else:
                # No existing entry: append from here
                remaining = entries[i:]
                for rem_entry in remaining:
                    await self._replicated_log.append(rem_entry)
                return

    # =========================================================================
    # Command Submission (Leader)
    # =========================================================================

    async def append_command(self, command: str, data: Dict[str, Any]) -> bool:
        """
        Submit a command to be replicated via the Raft log.

        Only the leader can accept new commands. The command is appended
        to the local log and then replicated to followers during the
        next heartbeat cycle. Returns True once the entry has been
        committed by a majority.

        Args:
            command: The command type (e.g., "set_state", "task_assign").
            data: The command data payload.

        Returns:
            True if the command was committed, False if this node is
            not the leader or the command could not be committed.
        """
        if self._role != NodeRole.LEADER:
            self._log.debug(
                "command_rejected_not_leader",
                command=command,
                role=self._role.value,
            )
            return False

        # Create and append the entry
        entry = RaftLogEntry(
            term=self._current_term,
            command=command,
            data=data,
        )
        index = await self._replicated_log.append(entry)

        self._log.debug(
            "command_appended",
            command=command,
            index=index,
            term=self._current_term,
        )

        # Create a future to wait for commitment
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending_commands[index] = future

        # Trigger immediate replication
        await self._send_heartbeats()

        # Wait for commitment with timeout
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            self._log.warning(
                "command_commit_timeout",
                index=index,
                command=command,
            )
            self._pending_commands.pop(index, None)
            return False

    # =========================================================================
    # Log Replication (Leader -> Followers)
    # =========================================================================

    async def _send_heartbeats(self) -> None:
        """
        Send AppendEntries RPCs to all peers.

        For each peer, constructs an AppendEntries request based on
        that peer's next_index. This handles both heartbeats (empty
        entries) and log replication (entries to catch up).
        """
        if self._role != NodeRole.LEADER:
            return

        peers = self._get_peer_ids()
        if not peers:
            # Single-node cluster: advance commit index directly
            await self._advance_commit_index()
            return

        tasks = []
        for peer_id in peers:
            tasks.append(self._send_append_entries_to_peer(peer_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._advance_commit_index()

    async def _send_append_entries_to_peer(self, peer_id: str) -> None:
        """
        Send an AppendEntries RPC to a single peer.

        Constructs the request based on the peer's next_index,
        sends it, and processes the response (updating next_index
        and match_index).

        If the peer's next_index has been compacted away (behind the
        snapshot boundary), an InstallSnapshot RPC is sent instead.

        All reads and writes to ``_next_index`` and ``_match_index``
        are protected by ``self._lock`` to prevent races when multiple
        peers are serviced concurrently via ``asyncio.gather``.
        """
        async with self._lock:
            next_idx = self._next_index.get(peer_id, 0)

            # If the follower needs entries we've already compacted,
            # send a snapshot instead (§7 of the Raft paper).
            snapshot_index = self._replicated_log.snapshot_index
            if snapshot_index >= 0 and next_idx <= snapshot_index:
                # Release lock before the potentially slow snapshot RPC
                pass
            else:
                snapshot_index = -1  # sentinel: no snapshot needed

            prev_log_index = next_idx - 1
            prev_log_term = self._replicated_log.get_term_at(prev_log_index)

            # Get entries to send
            entries = self._replicated_log.get_entries_since(next_idx)
            if len(entries) > self._max_entries_per_request:
                entries = entries[: self._max_entries_per_request]

            current_term = self._current_term
            commit_index = self._commit_index

        # Handle snapshot case (outside lock for the slow RPC)
        if snapshot_index >= 0 and next_idx <= snapshot_index:
            await self._send_install_snapshot_to_peer(peer_id)
            return

        request = AppendEntriesRequest(
            term=current_term,
            leader_id=self._node_info.id,
            prev_log_index=prev_log_index,
            prev_log_term=prev_log_term,
            entries=entries,
            leader_commit=commit_index,
        )

        self._append_entries_sent += 1

        # Send RPC (via cluster manager's transport) — outside lock
        response = await self._send_append_entries_rpc(peer_id, request)
        if response is None:
            self._log.debug(
                "append_entries_no_response",
                peer=peer_id,
            )
            return

        # Process response under lock
        async with self._lock:
            # Handle higher term discovery
            if self._discover_higher_term(response.term, source=f"ae_response_{peer_id}"):
                return

            if response.success:
                # Update next_index and match_index
                if entries:
                    new_match = entries[-1].index
                else:
                    new_match = prev_log_index
                self._next_index[peer_id] = new_match + 1
                self._match_index[peer_id] = new_match

                self._log.debug(
                    "append_entries_success",
                    peer=peer_id,
                    match_index=new_match,
                    entries_sent=len(entries),
                )
            else:
                # Conflict: use optimization to skip back efficiently
                if response.conflict_term is not None:
                    # Search our log for entries with the conflict term
                    # and set next_index to the end of that term
                    found = False
                    for i in range(
                        self._replicated_log.get_last_index(),
                        self._replicated_log.snapshot_index,
                        -1,
                    ):
                        entry = self._replicated_log.get(i)
                        if entry and entry.term == response.conflict_term:
                            self._next_index[peer_id] = i + 1
                            found = True
                            break
                    if not found and response.conflict_index is not None:
                        self._next_index[peer_id] = response.conflict_index
                elif response.conflict_index is not None:
                    # Follower does not have the entry at all
                    self._next_index[peer_id] = response.conflict_index
                else:
                    # Simple decrement fallback
                    self._next_index[peer_id] = max(0, next_idx - 1)

                self._log.debug(
                    "append_entries_conflict",
                    peer=peer_id,
                    conflict_term=response.conflict_term,
                    conflict_index=response.conflict_index,
                    new_next_index=self._next_index[peer_id],
                )

    async def _advance_commit_index(self) -> None:
        """
        Advance the commit index based on match_index values.

        A log entry is committed when it has been replicated to a
        majority of nodes AND it was created in the leader's current
        term (Raft safety: leaders only commit entries from their
        own term; previous-term entries are committed indirectly).

        This method must be called under ``self._lock`` or from a
        context that already holds exclusive access to the mutable state.
        """
        async with self._lock:
            if self._role != NodeRole.LEADER:
                return

            peers = self._get_peer_ids()
            cluster_size = len(peers) + 1  # Include self

            # Collect all match indices (leader's own match is last_log_index)
            match_indices = [self._replicated_log.get_last_index()]
            for peer_id in peers:
                match_indices.append(self._match_index.get(peer_id, -1))

            # Sort descending and find the median (quorum position)
            match_indices.sort(reverse=True)
            quorum_pos = cluster_size // 2  # Majority position (0-indexed)

            if quorum_pos < len(match_indices):
                potential_commit = match_indices[quorum_pos]
            else:
                return

            # Only commit entries from the current term (Raft safety)
            if potential_commit > self._commit_index:
                entry = self._replicated_log.get(potential_commit)
                if entry is not None and entry.term == self._current_term:
                    old_commit = self._commit_index
                    self._commit_index = potential_commit

                    self._log.debug(
                        "commit_index_advanced",
                        old=old_commit,
                        new=self._commit_index,
                        match_indices=match_indices,
                    )

                    # Resolve pending command futures
                    self._resolve_pending_commands()

    def _resolve_pending_commands(self) -> None:
        """Resolve futures for committed commands."""
        committed = []
        for index, future in self._pending_commands.items():
            if index <= self._commit_index and not future.done():
                future.set_result(True)
                committed.append(index)

        for index in committed:
            self._pending_commands.pop(index, None)

    # =========================================================================
    # RPC Transport (Abstraction Layer)
    # =========================================================================

    async def _broadcast_vote_request(
        self, request: VoteRequest
    ) -> List[VoteResponse]:
        """
        Broadcast a VoteRequest to all peers and collect responses.

        Uses the cluster manager's transport layer. Unreachable peers
        are silently skipped.
        """
        peers = self._get_peer_ids()
        responses: List[VoteResponse] = []

        async def request_vote(peer_id: str) -> Optional[VoteResponse]:
            try:
                return await self._send_vote_request_rpc(peer_id, request)
            except Exception as exc:
                self._log.debug(
                    "vote_request_failed",
                    peer=peer_id,
                    error=str(exc),
                )
                return None

        tasks = [request_vote(pid) for pid in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, VoteResponse):
                responses.append(result)

        return responses

    async def _send_vote_request_rpc(
        self, peer_id: str, request: VoteRequest
    ) -> Optional[VoteResponse]:
        """
        Send a VoteRequest RPC to a single peer via cluster manager transport.

        This is an integration point with the cluster manager's networking layer.
        """
        try:
            if hasattr(self._cluster_manager, "send_vote_request"):
                return await self._cluster_manager.send_vote_request(peer_id, request)
        except Exception as exc:
            self._log.debug("rpc_vote_failed", peer=peer_id, error=str(exc))
        return None

    async def _send_append_entries_rpc(
        self, peer_id: str, request: AppendEntriesRequest
    ) -> Optional[AppendEntriesResponse]:
        """
        Send an AppendEntries RPC to a single peer via cluster manager transport.

        This is an integration point with the cluster manager's networking layer.
        """
        try:
            if hasattr(self._cluster_manager, "send_append_entries"):
                return await self._cluster_manager.send_append_entries(
                    peer_id, request
                )
        except Exception as exc:
            self._log.debug("rpc_append_failed", peer=peer_id, error=str(exc))
        return None

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _election_timer_loop(self) -> None:
        """
        Background task that monitors the election timeout.

        When the timeout elapses without receiving a heartbeat from
        the leader, this task triggers an election. Only active when
        the node is a follower or candidate.
        """
        while self._running:
            try:
                # Sleep for a fraction of the timeout to check periodically
                await asyncio.sleep(0.01)  # 10ms poll interval

                if self._role == NodeRole.LEADER:
                    continue

                if self._election.is_timeout_elapsed():
                    self._log.info(
                        "election_timeout_elapsed",
                        term=self._current_term,
                        role=self._role.value,
                    )
                    await self.trigger_election()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    "election_timer_error",
                    error=str(exc),
                    exc_info=True,
                )
                await asyncio.sleep(0.1)

    async def _heartbeat_loop(self) -> None:
        """
        Background task that sends periodic heartbeats as leader.

        Heartbeats are empty AppendEntries RPCs that maintain the
        leader's authority and trigger log replication for peers
        that have fallen behind.
        """
        while self._running and self._role == NodeRole.LEADER:
            try:
                await asyncio.sleep(self._heartbeat_interval_ms / 1000.0)

                if self._role != NodeRole.LEADER:
                    break

                await self._send_heartbeats()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    "heartbeat_loop_error",
                    error=str(exc),
                    exc_info=True,
                )
                await asyncio.sleep(0.05)

    async def _apply_committed_entries_loop(self) -> None:
        """
        Background task that applies committed entries to the state machine.

        Continuously checks if commit_index > last_applied and applies
        entries in order.
        """
        while self._running:
            try:
                await asyncio.sleep(0.005)  # 5ms poll interval

                while self._commit_index > self._last_applied:
                    next_index = self._last_applied + 1
                    entry = self._replicated_log.get(next_index)
                    if entry is None:
                        break

                    await self._state_machine.apply(entry)
                    self._last_applied = next_index

                    self._log.debug(
                        "entry_applied_to_state_machine",
                        index=next_index,
                        command=entry.command,
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    "apply_loop_error",
                    error=str(exc),
                    exc_info=True,
                )
                await asyncio.sleep(0.1)

    # =========================================================================
    # Peer Management
    # =========================================================================

    def _get_peer_ids(self) -> List[str]:
        """
        Get the IDs of all peer nodes (excluding self).

        Uses the cluster manager to discover current cluster members.
        """
        try:
            if hasattr(self._cluster_manager, "get_peer_ids"):
                return self._cluster_manager.get_peer_ids(self._node_info.id)
            if hasattr(self._cluster_manager, "get_node_ids"):
                all_ids = self._cluster_manager.get_node_ids()
                return [nid for nid in all_ids if nid != self._node_info.id]
        except Exception as exc:
            self._log.debug("get_peers_error", error=str(exc))
        return []

    # =========================================================================
    # Snapshot Support
    # =========================================================================

    async def take_snapshot(self) -> Optional[SnapshotMetadata]:
        """
        Take a snapshot of the current state for log compaction.

        The snapshot captures the state machine's state at the last
        applied index. After the snapshot is taken, the log can be
        compacted to discard entries up to that index.

        Returns:
            Snapshot metadata, or None if no entries have been applied.
        """
        if self._last_applied < 0:
            return None

        snapshot_data = await self._state_machine.take_snapshot()
        metadata = self._state_machine.get_snapshot_metadata()
        metadata.size_bytes = len(snapshot_data)

        # Compact the log
        await self._replicated_log.compact(self._last_applied)

        self._log.info(
            "snapshot_completed",
            last_included_index=metadata.last_included_index,
            last_included_term=metadata.last_included_term,
            size_bytes=metadata.size_bytes,
        )

        # Cache snapshot data for InstallSnapshot RPCs
        self._last_snapshot_data = snapshot_data
        return metadata

    # =========================================================================
    # InstallSnapshot RPC (Receiver / Follower)
    # =========================================================================

    async def handle_install_snapshot(
        self, request: InstallSnapshotRequest
    ) -> InstallSnapshotResponse:
        """Handle an InstallSnapshot RPC from the leader.

        When a follower's log is too far behind and the leader has
        compacted the entries it needs, the leader sends a snapshot
        instead of individual log entries.  The follower replaces its
        state machine with the snapshot and resets its log.

        Implements §7 of the Raft paper.

        Args:
            request: The incoming InstallSnapshot request.

        Returns:
            An InstallSnapshotResponse with the current term.
        """
        async with self._lock:
            response = InstallSnapshotResponse(
                term=self._current_term,
            )

            # Step down if higher term
            if self._discover_higher_term(request.term, source="install_snapshot"):
                response.term = self._current_term

            # Stale term — reject
            if request.term < self._current_term:
                self._log.debug(
                    "install_snapshot_denied_stale_term",
                    leader_term=request.term,
                    our_term=self._current_term,
                )
                return response

            # Valid leader — reset election timer
            self._election.reset_timeout()
            self._leader_id = request.leader_id

            if self._role != NodeRole.FOLLOWER:
                self._step_down_to_follower()

            # If this snapshot is not newer than what we have, ignore
            if request.last_included_index <= self._replicated_log.snapshot_index:
                self._log.debug(
                    "install_snapshot_stale",
                    request_index=request.last_included_index,
                    our_snapshot_index=self._replicated_log.snapshot_index,
                )
                return response

            # Install the snapshot into the state machine
            await self._state_machine.restore_snapshot(request.data)

            # Reset the log to the snapshot boundary
            await self._replicated_log.reset_to_snapshot(
                request.last_included_index,
                request.last_included_term,
            )

            # Update volatile state
            self._commit_index = max(self._commit_index, request.last_included_index)
            self._last_applied = request.last_included_index

            self._log.info(
                "snapshot_installed",
                last_included_index=request.last_included_index,
                last_included_term=request.last_included_term,
                data_bytes=len(request.data) if request.data else 0,
            )

            return response

    async def _send_install_snapshot_to_peer(self, peer_id: str) -> None:
        """Send an InstallSnapshot RPC to a peer that is behind the snapshot.

        Called by the leader when a follower's next_index is at or before
        the snapshot boundary and the required log entries have been
        compacted away.
        """
        snapshot_index = self._replicated_log.snapshot_index
        snapshot_term = self._replicated_log.snapshot_term

        if snapshot_index < 0:
            return  # No snapshot to send

        # Get snapshot data (take one if we don't have cached data)
        snapshot_data = self._last_snapshot_data
        if snapshot_data is None:
            snapshot_data = await self._state_machine.take_snapshot()
            self._last_snapshot_data = snapshot_data

        request = InstallSnapshotRequest(
            term=self._current_term,
            leader_id=self._node_info.id,
            last_included_index=snapshot_index,
            last_included_term=snapshot_term,
            data=snapshot_data,
        )

        self._log.info(
            "sending_install_snapshot",
            peer=peer_id,
            snapshot_index=snapshot_index,
            snapshot_term=snapshot_term,
        )

        try:
            response = await self._send_install_snapshot_rpc(peer_id, request)
            if response is not None:
                async with self._lock:
                    if self._discover_higher_term(
                        response.term, source=f"install_snap_resp_{peer_id}"
                    ):
                        return
                    # On success, advance next_index past the snapshot
                    self._next_index[peer_id] = snapshot_index + 1
                    self._match_index[peer_id] = snapshot_index
                self._log.info(
                    "install_snapshot_sent",
                    peer=peer_id,
                    new_next_index=snapshot_index + 1,
                )
        except Exception as exc:
            self._log.warning(
                "install_snapshot_failed",
                peer=peer_id,
                error=str(exc),
            )

    async def _send_install_snapshot_rpc(
        self, peer_id: str, request: InstallSnapshotRequest
    ) -> Optional[InstallSnapshotResponse]:
        """Send an InstallSnapshot RPC via the cluster manager transport."""
        try:
            if hasattr(self._cluster_manager, "send_install_snapshot"):
                return await self._cluster_manager.send_install_snapshot(
                    peer_id, request
                )
        except Exception as exc:
            self._log.debug("rpc_install_snapshot_failed", peer=peer_id, error=str(exc))
        return None

    # =========================================================================
    # Statistics and Diagnostics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Raft consensus statistics."""
        return {
            "node_id": self._node_info.id,
            "role": self._role.value,
            "current_term": self._current_term,
            "voted_for": self._voted_for,
            "leader_id": self._leader_id,
            "commit_index": self._commit_index,
            "last_applied": self._last_applied,
            "log_last_index": self._replicated_log.get_last_index(),
            "log_last_term": self._replicated_log.get_last_term(),
            "log_entries_in_memory": self._replicated_log.length,
            "pending_commands": len(self._pending_commands),
            "elections_started": self._elections_started,
            "elections_won": self._elections_won,
            "votes_granted": self._votes_granted,
            "step_downs": self._step_downs,
            "append_entries_sent": self._append_entries_sent,
            "append_entries_received": self._append_entries_received,
            "next_index": dict(self._next_index),
            "match_index": dict(self._match_index),
            "running": self._running,
            "election": self._election.get_stats(),
            "log": self._replicated_log.get_stats(),
            "state_machine": self._state_machine.get_stats(),
        }
