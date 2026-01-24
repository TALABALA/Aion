"""
AION Consensus Engine

Voting mechanisms and conflict resolution for multi-agent decisions.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import structlog

from aion.systems.agents.types import (
    ConsensusVote,
    ConsensusResult,
    ConsensusMethod,
    Message,
    MessageType,
)

if TYPE_CHECKING:
    from aion.systems.agents.messaging import MessageBus
    from aion.systems.agents.pool import AgentPool

logger = structlog.get_logger(__name__)


class ConsensusEngine:
    """
    Consensus engine for multi-agent decision making.

    Features:
    - Multiple voting methods (majority, weighted, ranked, etc.)
    - Async vote collection with timeouts
    - Confidence-weighted voting
    - Conflict resolution strategies
    - Decision history tracking
    """

    def __init__(
        self,
        message_bus: "MessageBus",
        default_method: ConsensusMethod = ConsensusMethod.WEIGHTED,
        default_timeout: float = 60.0,
    ):
        self.bus = message_bus
        self.default_method = default_method
        self.default_timeout = default_timeout

        # Active consensus sessions
        self._sessions: dict[str, ConsensusResult] = {}

        # Decision history
        self._history: list[ConsensusResult] = []

        # Statistics
        self._total_decisions: int = 0
        self._unanimous_decisions: int = 0

    async def request_consensus(
        self,
        question: str,
        options: list[str],
        voter_ids: list[str],
        method: Optional[ConsensusMethod] = None,
        timeout: Optional[float] = None,
        min_votes: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ConsensusResult:
        """
        Request consensus from a group of agents.

        Args:
            question: The question to decide
            options: Available options to choose from
            voter_ids: Agent IDs who should vote
            method: Voting method to use
            timeout: Time to wait for votes
            min_votes: Minimum votes required
            context: Additional context for voters

        Returns:
            Consensus result
        """
        method = method or self.default_method
        timeout = timeout or self.default_timeout
        min_votes = min_votes or max(1, len(voter_ids) // 2 + 1)

        # Create result object
        result = ConsensusResult(
            question=question,
            options=options,
            method=method,
        )

        self._sessions[result.id] = result

        logger.info(
            "Starting consensus",
            consensus_id=result.id[:8],
            question=question[:50],
            voters=len(voter_ids),
            method=method.value,
        )

        # Request votes from all agents
        vote_futures = []
        for agent_id in voter_ids:
            future = asyncio.create_task(
                self._request_vote(
                    result.id,
                    agent_id,
                    question,
                    options,
                    context,
                    timeout,
                )
            )
            vote_futures.append(future)

        # Wait for votes with timeout
        try:
            votes = await asyncio.wait_for(
                asyncio.gather(*vote_futures, return_exceptions=True),
                timeout=timeout + 5,  # Extra buffer
            )

            # Collect valid votes
            for vote in votes:
                if isinstance(vote, ConsensusVote):
                    result.votes.append(vote)
                elif isinstance(vote, Exception):
                    logger.warning("Vote error", error=str(vote))

        except asyncio.TimeoutError:
            logger.warning("Consensus timeout", consensus_id=result.id[:8])

        # Calculate result
        if len(result.votes) >= min_votes:
            self._calculate_result(result, method)
        else:
            logger.warning(
                "Insufficient votes",
                consensus_id=result.id[:8],
                votes=len(result.votes),
                required=min_votes,
            )

        result.completed_at = datetime.now()
        self._sessions.pop(result.id, None)

        # Record to history
        self._history.append(result)
        self._total_decisions += 1
        if result.unanimous:
            self._unanimous_decisions += 1

        logger.info(
            "Consensus complete",
            consensus_id=result.id[:8],
            winner=result.winning_option,
            confidence=result.confidence,
            unanimous=result.unanimous,
        )

        return result

    async def _request_vote(
        self,
        consensus_id: str,
        agent_id: str,
        question: str,
        options: list[str],
        context: Optional[dict[str, Any]],
        timeout: float,
    ) -> Optional[ConsensusVote]:
        """Request a vote from an agent."""
        # Send vote request
        response = await self.bus.request(
            sender_id="consensus_engine",
            recipient_id=agent_id,
            content={
                "type": "vote_request",
                "consensus_id": consensus_id,
                "question": question,
                "options": options,
                "context": context,
            },
            subject=f"Vote requested: {question[:30]}",
            timeout=timeout,
        )

        if not response:
            return None

        # Parse vote from response
        try:
            vote_data = response.content
            if isinstance(vote_data, dict):
                return ConsensusVote(
                    agent_id=agent_id,
                    option=vote_data.get("option", options[0]),
                    confidence=vote_data.get("confidence", 1.0),
                    reasoning=vote_data.get("reasoning", ""),
                    rank=vote_data.get("rank"),
                )
        except Exception as e:
            logger.warning("Failed to parse vote", agent=agent_id[:8], error=str(e))

        return None

    def _calculate_result(
        self,
        result: ConsensusResult,
        method: ConsensusMethod,
    ) -> None:
        """Calculate consensus result based on method."""
        if method == ConsensusMethod.MAJORITY:
            self._calculate_majority(result)
        elif method == ConsensusMethod.SUPERMAJORITY:
            self._calculate_supermajority(result)
        elif method == ConsensusMethod.UNANIMOUS:
            self._calculate_unanimous(result)
        elif method == ConsensusMethod.WEIGHTED:
            self._calculate_weighted(result)
        elif method == ConsensusMethod.RANKED:
            self._calculate_ranked(result)
        elif method == ConsensusMethod.BORDA:
            self._calculate_borda(result)
        else:
            self._calculate_weighted(result)

    def _calculate_majority(self, result: ConsensusResult) -> None:
        """Simple majority wins."""
        counts: dict[str, int] = defaultdict(int)

        for vote in result.votes:
            counts[vote.option] += 1

        result.vote_counts = dict(counts)

        if counts:
            winner = max(counts, key=counts.get)
            result.winning_option = winner
            result.confidence = counts[winner] / len(result.votes)
            result.unanimous = len(counts) == 1

    def _calculate_supermajority(self, result: ConsensusResult) -> None:
        """2/3 majority required."""
        counts: dict[str, int] = defaultdict(int)
        threshold = len(result.votes) * 2 / 3

        for vote in result.votes:
            counts[vote.option] += 1

        result.vote_counts = dict(counts)

        for option, count in counts.items():
            if count >= threshold:
                result.winning_option = option
                result.confidence = count / len(result.votes)
                result.unanimous = count == len(result.votes)
                return

        # No supermajority - use plurality
        if counts:
            winner = max(counts, key=counts.get)
            result.winning_option = winner
            result.confidence = counts[winner] / len(result.votes)

    def _calculate_unanimous(self, result: ConsensusResult) -> None:
        """All must agree."""
        if not result.votes:
            return

        first_option = result.votes[0].option
        unanimous = all(v.option == first_option for v in result.votes)

        if unanimous:
            result.winning_option = first_option
            result.confidence = 1.0
            result.unanimous = True
        else:
            # Fall back to weighted
            self._calculate_weighted(result)
            result.unanimous = False

    def _calculate_weighted(self, result: ConsensusResult) -> None:
        """Votes weighted by confidence."""
        scores: dict[str, float] = defaultdict(float)
        total_confidence = 0.0

        for vote in result.votes:
            scores[vote.option] += vote.confidence
            total_confidence += vote.confidence

        result.weighted_scores = dict(scores)

        if scores:
            winner = max(scores, key=scores.get)
            result.winning_option = winner
            result.confidence = scores[winner] / total_confidence if total_confidence > 0 else 0

            # Check if unanimous
            result.unanimous = len(scores) == 1

        # Also track raw counts
        counts: dict[str, int] = defaultdict(int)
        for vote in result.votes:
            counts[vote.option] += 1
        result.vote_counts = dict(counts)

    def _calculate_ranked(self, result: ConsensusResult) -> None:
        """Instant-runoff voting with ranked preferences."""
        if not any(v.rank for v in result.votes):
            # No rankings provided, fall back to weighted
            self._calculate_weighted(result)
            return

        # Get all options
        remaining_options = set(result.options)
        rankings = [v.rank or [v.option] for v in result.votes]
        total_votes = len(rankings)

        while remaining_options:
            # Count first-choice votes
            counts: dict[str, int] = defaultdict(int)
            for ranking in rankings:
                for option in ranking:
                    if option in remaining_options:
                        counts[option] += 1
                        break

            if not counts:
                break

            # Check for majority
            max_option = max(counts, key=counts.get)
            if counts[max_option] > total_votes / 2:
                result.winning_option = max_option
                result.confidence = counts[max_option] / total_votes
                result.vote_counts = dict(counts)
                result.unanimous = counts[max_option] == total_votes
                return

            # Eliminate lowest
            min_option = min(counts, key=counts.get)
            remaining_options.remove(min_option)

        # Fallback
        if remaining_options:
            result.winning_option = list(remaining_options)[0]
            result.confidence = 0.5

    def _calculate_borda(self, result: ConsensusResult) -> None:
        """Borda count method."""
        if not any(v.rank for v in result.votes):
            # No rankings, fall back to weighted
            self._calculate_weighted(result)
            return

        scores: dict[str, float] = defaultdict(float)
        num_options = len(result.options)

        for vote in result.votes:
            ranking = vote.rank or [vote.option]
            for i, option in enumerate(ranking):
                # Higher rank = more points
                points = num_options - i
                scores[option] += points * vote.confidence

        result.weighted_scores = dict(scores)

        if scores:
            winner = max(scores, key=scores.get)
            max_possible = len(result.votes) * num_options
            result.winning_option = winner
            result.confidence = scores[winner] / max_possible if max_possible > 0 else 0

            # Check unanimous
            counts: dict[str, int] = defaultdict(int)
            for vote in result.votes:
                counts[vote.option] += 1
            result.vote_counts = dict(counts)
            result.unanimous = len(counts) == 1

    # === Conflict Resolution ===

    async def resolve_conflict(
        self,
        positions: list[dict[str, Any]],
        arbiter_ids: list[str],
        context: Optional[dict[str, Any]] = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """
        Resolve a conflict between competing positions.

        Args:
            positions: List of positions with {agent_id, position, reasoning}
            arbiter_ids: Agents who will arbitrate
            context: Additional context
            timeout: Resolution timeout

        Returns:
            Resolution result
        """
        # Format positions for voting
        options = [p["position"] for p in positions]
        position_lookup = {p["position"]: p for p in positions}

        # Add context with all positions
        full_context = {
            "type": "conflict_resolution",
            "positions": positions,
            **(context or {}),
        }

        # Request consensus from arbiters
        result = await self.request_consensus(
            question=f"Resolve conflict: {options[0][:50]}... vs others",
            options=options,
            voter_ids=arbiter_ids,
            method=ConsensusMethod.WEIGHTED,
            timeout=timeout,
            context=full_context,
        )

        winning_position = position_lookup.get(result.winning_option, {})

        return {
            "resolved": result.winning_option is not None,
            "winning_position": result.winning_option,
            "winning_agent": winning_position.get("agent_id"),
            "reasoning": winning_position.get("reasoning"),
            "confidence": result.confidence,
            "unanimous": result.unanimous,
            "votes": len(result.votes),
            "consensus_id": result.id,
        }

    async def merge_outputs(
        self,
        outputs: list[dict[str, Any]],
        merger_ids: list[str],
        criteria: Optional[list[str]] = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """
        Merge multiple agent outputs into a unified result.

        Args:
            outputs: List of outputs with {agent_id, output, quality_score}
            merger_ids: Agents who will merge
            criteria: Criteria for evaluation
            timeout: Merge timeout

        Returns:
            Merged result
        """
        if len(outputs) == 1:
            return {
                "merged": True,
                "output": outputs[0]["output"],
                "source_agents": [outputs[0]["agent_id"]],
                "method": "single_output",
            }

        # Create options based on outputs
        options = [f"output_{i}" for i in range(len(outputs))]
        output_lookup = {f"output_{i}": o for i, o in enumerate(outputs)}

        # Request evaluation
        context = {
            "type": "output_merge",
            "outputs": outputs,
            "criteria": criteria or ["quality", "completeness", "accuracy"],
        }

        result = await self.request_consensus(
            question="Select best output to use as base for merge",
            options=options,
            voter_ids=merger_ids,
            method=ConsensusMethod.WEIGHTED,
            timeout=timeout,
            context=context,
        )

        best_output = output_lookup.get(result.winning_option, outputs[0])

        return {
            "merged": True,
            "output": best_output["output"],
            "source_agents": [o["agent_id"] for o in outputs],
            "primary_agent": best_output["agent_id"],
            "confidence": result.confidence,
            "method": "consensus_selection",
        }

    # === Vote Submission ===

    def submit_vote(
        self,
        consensus_id: str,
        agent_id: str,
        option: str,
        confidence: float = 1.0,
        reasoning: str = "",
        rank: Optional[list[str]] = None,
    ) -> bool:
        """
        Submit a vote for an active consensus session.

        Args:
            consensus_id: Consensus session ID
            agent_id: Voting agent ID
            option: Selected option
            confidence: Confidence in vote (0-1)
            reasoning: Explanation for vote
            rank: Ranked preferences for ranked voting

        Returns:
            True if vote accepted
        """
        result = self._sessions.get(consensus_id)
        if not result:
            logger.warning("Consensus session not found", consensus_id=consensus_id[:8])
            return False

        if option not in result.options:
            logger.warning("Invalid option", option=option, consensus_id=consensus_id[:8])
            return False

        # Check for duplicate vote
        for existing in result.votes:
            if existing.agent_id == agent_id:
                logger.warning("Duplicate vote", agent=agent_id[:8])
                return False

        vote = ConsensusVote(
            agent_id=agent_id,
            option=option,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
            rank=rank,
        )

        result.votes.append(vote)

        logger.debug(
            "Vote submitted",
            agent=agent_id[:8],
            option=option,
            confidence=confidence,
        )

        return True

    # === History and Stats ===

    def get_history(
        self,
        limit: int = 100,
        method: Optional[ConsensusMethod] = None,
    ) -> list[ConsensusResult]:
        """Get consensus decision history."""
        history = self._history

        if method:
            history = [r for r in history if r.method == method]

        return history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get consensus engine statistics."""
        method_counts: dict[str, int] = defaultdict(int)
        for result in self._history:
            method_counts[result.method.value] += 1

        return {
            "total_decisions": self._total_decisions,
            "unanimous_decisions": self._unanimous_decisions,
            "unanimity_rate": (
                self._unanimous_decisions / self._total_decisions
                if self._total_decisions > 0 else 0
            ),
            "active_sessions": len(self._sessions),
            "history_size": len(self._history),
            "by_method": dict(method_counts),
        }
