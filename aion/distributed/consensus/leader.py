"""
AION Leader Election Helper

Encapsulates the leader election logic for the Raft consensus protocol.
Handles vote solicitation, vote counting, randomized election timeouts,
and the PreVote protocol extension.

The PreVote protocol (Section 9.6 of the Raft dissertation) prevents
nodes that are partitioned from the cluster from incrementing their
term when they start elections. A pre-vote round checks that the
candidate can win before it increments its term, preventing term
inflation that would disrupt the cluster upon partition healing.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, Optional, Set

import structlog

from aion.distributed.types import (
    NodeRole,
    VoteRequest,
    VoteResponse,
)

logger = structlog.get_logger(__name__)


class LeaderElection:
    """
    Manages the leader election process for a Raft node.

    Tracks the election state including votes received, election
    timeouts, and provides methods to conduct elections and
    determine winners based on quorum.

    Features:
    - Randomized election timeouts to prevent split votes
    - PreVote protocol support to avoid term inflation
    - Vote tracking and quorum detection
    - Election timeout management
    - Detailed election metrics
    """

    def __init__(
        self,
        node_id: str,
        election_timeout_min_ms: int = 150,
        election_timeout_max_ms: int = 300,
        pre_vote_enabled: bool = True,
    ) -> None:
        self._log = logger.bind(component="leader_election", node_id=node_id)
        self._node_id = node_id

        # Timeout configuration
        self._election_timeout_min_ms = election_timeout_min_ms
        self._election_timeout_max_ms = election_timeout_max_ms
        self._pre_vote_enabled = pre_vote_enabled

        # Election state
        self._votes_received: Set[str] = set()
        self._votes_denied: Set[str] = set()
        self._election_term: int = 0
        self._is_pre_vote: bool = False
        self._election_start_time: float = 0.0
        self._election_count: int = 0

        # Timeout tracking
        self._current_timeout_ms: float = self._random_timeout()
        self._last_reset_time: float = time.monotonic()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def votes_received(self) -> Set[str]:
        """Set of node IDs that have granted their vote."""
        return set(self._votes_received)

    @property
    def votes_denied(self) -> Set[str]:
        """Set of node IDs that have denied their vote."""
        return set(self._votes_denied)

    @property
    def vote_count(self) -> int:
        """Number of votes received (including self-vote)."""
        return len(self._votes_received)

    @property
    def election_term(self) -> int:
        """The term for the current or most recent election."""
        return self._election_term

    @property
    def is_pre_vote(self) -> bool:
        """Whether the current election round is a pre-vote."""
        return self._is_pre_vote

    @property
    def pre_vote_enabled(self) -> bool:
        """Whether the PreVote protocol extension is enabled."""
        return self._pre_vote_enabled

    @property
    def election_count(self) -> int:
        """Total number of elections started."""
        return self._election_count

    @property
    def current_timeout_ms(self) -> float:
        """The current randomized election timeout in milliseconds."""
        return self._current_timeout_ms

    # -------------------------------------------------------------------------
    # Timeout Management
    # -------------------------------------------------------------------------

    def _random_timeout(self) -> float:
        """Generate a randomized election timeout within the configured range."""
        return random.uniform(
            self._election_timeout_min_ms,
            self._election_timeout_max_ms,
        )

    def reset_timeout(self) -> float:
        """
        Reset the election timer with a new random timeout.

        Called when:
        - A heartbeat is received from the leader
        - A vote is granted to another candidate
        - An election starts

        Returns:
            The new timeout duration in milliseconds.
        """
        self._current_timeout_ms = self._random_timeout()
        self._last_reset_time = time.monotonic()
        return self._current_timeout_ms

    def is_timeout_elapsed(self) -> bool:
        """
        Check if the election timeout has elapsed.

        Returns:
            True if the timeout has elapsed and an election should start.
        """
        elapsed_ms = (time.monotonic() - self._last_reset_time) * 1000.0
        return elapsed_ms >= self._current_timeout_ms

    def time_until_timeout_ms(self) -> float:
        """
        Get the remaining time until the election timeout fires.

        Returns:
            Milliseconds remaining, or 0.0 if already elapsed.
        """
        elapsed_ms = (time.monotonic() - self._last_reset_time) * 1000.0
        remaining = self._current_timeout_ms - elapsed_ms
        return max(0.0, remaining)

    # -------------------------------------------------------------------------
    # Election Lifecycle
    # -------------------------------------------------------------------------

    def start_election(
        self,
        term: int,
        pre_vote: bool = False,
    ) -> VoteRequest:
        """
        Start a new election (or pre-vote round).

        Initializes the election state, records a self-vote (for real
        elections), and returns the VoteRequest to broadcast to peers.

        Args:
            term: The term for this election. For pre-votes, this is
                  the term the candidate *would* use. For real elections,
                  this is the candidate's incremented current term.
            pre_vote: If True, this is a pre-vote round that does not
                      increment the term.

        Returns:
            A VoteRequest to send to all peer nodes.
        """
        self._election_term = term
        self._is_pre_vote = pre_vote
        self._votes_received = set()
        self._votes_denied = set()
        self._election_start_time = time.monotonic()
        self._election_count += 1

        # Self-vote (only for real elections, not pre-votes)
        if not pre_vote:
            self._votes_received.add(self._node_id)

        self.reset_timeout()

        phase = "pre_vote" if pre_vote else "election"
        self._log.info(
            "election_started",
            phase=phase,
            term=term,
            election_number=self._election_count,
        )

        return VoteRequest(
            term=term,
            candidate_id=self._node_id,
            last_log_index=0,  # Caller must set these from the log
            last_log_term=0,   # Caller must set these from the log
            is_pre_vote=pre_vote,
        )

    def handle_vote(self, response: VoteResponse) -> None:
        """
        Process a vote response from a peer.

        Records whether the vote was granted or denied. Only responses
        for the current election term are considered; stale responses
        are discarded.

        Args:
            response: The vote response from a peer node.
        """
        if response.term != self._election_term:
            self._log.debug(
                "stale_vote_response",
                response_term=response.term,
                election_term=self._election_term,
                voter=response.voter_id,
            )
            return

        if response.vote_granted:
            self._votes_received.add(response.voter_id)
            self._log.debug(
                "vote_received",
                voter=response.voter_id,
                total_votes=len(self._votes_received),
            )
        else:
            self._votes_denied.add(response.voter_id)
            self._log.debug(
                "vote_denied",
                voter=response.voter_id,
                total_denied=len(self._votes_denied),
            )

    def check_if_won(self, cluster_size: int) -> bool:
        """
        Check if the candidate has received enough votes to win.

        A candidate wins when it has received votes from a strict
        majority (quorum) of the cluster.

        Args:
            cluster_size: Total number of voting members in the cluster.

        Returns:
            True if the candidate has a quorum of votes.
        """
        quorum = (cluster_size // 2) + 1
        won = len(self._votes_received) >= quorum

        if won:
            elapsed_ms = (time.monotonic() - self._election_start_time) * 1000.0
            phase = "pre_vote" if self._is_pre_vote else "election"
            self._log.info(
                "election_won",
                phase=phase,
                term=self._election_term,
                votes=len(self._votes_received),
                quorum=quorum,
                cluster_size=cluster_size,
                elapsed_ms=round(elapsed_ms, 2),
            )
        return won

    def check_if_lost(self, cluster_size: int) -> bool:
        """
        Check if the candidate has been denied by enough nodes
        that winning is impossible.

        Args:
            cluster_size: Total number of voting members.

        Returns:
            True if it is mathematically impossible to win.
        """
        quorum = (cluster_size // 2) + 1
        max_possible_votes = cluster_size - len(self._votes_denied)
        # Account for votes already received
        remaining_possible = max_possible_votes - len(self._votes_received)
        total_possible = len(self._votes_received) + max(0, remaining_possible)
        return total_possible < quorum

    def should_start_prevote(self) -> bool:
        """
        Determine if a pre-vote round should precede a real election.

        Returns True if:
        - PreVote is enabled in configuration
        - The current round is not already a pre-vote
        """
        return self._pre_vote_enabled and not self._is_pre_vote

    # -------------------------------------------------------------------------
    # Vote Eligibility
    # -------------------------------------------------------------------------

    @staticmethod
    def is_candidate_log_up_to_date(
        candidate_last_index: int,
        candidate_last_term: int,
        voter_last_index: int,
        voter_last_term: int,
    ) -> bool:
        """
        Check if a candidate's log is at least as up-to-date as the voter's.

        Raft defines "up-to-date" by comparing terms first, then indices.
        A candidate with a higher last term is more up-to-date. If terms
        are equal, the longer log is more up-to-date.

        Args:
            candidate_last_index: Last log index of the candidate.
            candidate_last_term: Last log term of the candidate.
            voter_last_index: Last log index of the voter.
            voter_last_term: Last log term of the voter.

        Returns:
            True if the candidate's log is at least as up-to-date.
        """
        if candidate_last_term != voter_last_term:
            return candidate_last_term > voter_last_term
        return candidate_last_index >= voter_last_index

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_election_duration_ms(self) -> float:
        """Get the duration of the current/last election in milliseconds."""
        if self._election_start_time == 0.0:
            return 0.0
        return (time.monotonic() - self._election_start_time) * 1000.0

    def get_stats(self) -> Dict[str, Any]:
        """Get election statistics."""
        return {
            "node_id": self._node_id,
            "election_term": self._election_term,
            "is_pre_vote": self._is_pre_vote,
            "pre_vote_enabled": self._pre_vote_enabled,
            "votes_received": len(self._votes_received),
            "votes_denied": len(self._votes_denied),
            "voters": sorted(self._votes_received),
            "election_count": self._election_count,
            "current_timeout_ms": round(self._current_timeout_ms, 2),
            "election_duration_ms": round(self.get_election_duration_ms(), 2),
            "timeout_min_ms": self._election_timeout_min_ms,
            "timeout_max_ms": self._election_timeout_max_ms,
        }
