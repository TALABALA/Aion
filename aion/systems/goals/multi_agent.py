"""
AION Goal System - Multi-Agent Coordination

SOTA multi-agent system for distributed goal pursuit.

Key capabilities:
- Agent registry and lifecycle management
- Distributed task allocation with auction mechanism
- Consensus protocols for shared decisions
- Conflict resolution and negotiation
- Hierarchical organization structures
- Communication and message passing
- Coalition formation for complex goals
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import heapq
import uuid

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
)

logger = structlog.get_logger()


class AgentRole(Enum):
    """Roles an agent can play in the system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    FAILED = "failed"


class MessageType(Enum):
    """Types of messages between agents."""
    TASK_ANNOUNCEMENT = "task_announcement"
    BID = "bid"
    AWARD = "award"
    ACCEPT = "accept"
    REJECT = "reject"
    PROGRESS = "progress"
    COMPLETE = "complete"
    FAILURE = "failure"
    HEARTBEAT = "heartbeat"
    CONSENSUS_PROPOSE = "consensus_propose"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_COMMIT = "consensus_commit"


@dataclass
class AgentCapability:
    """A capability an agent possesses."""

    name: str
    proficiency: float = 1.0  # 0-1
    cost_factor: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """An agent in the multi-agent system."""

    id: str
    name: str
    role: AgentRole = AgentRole.WORKER
    status: AgentStatus = AgentStatus.IDLE

    capabilities: List[AgentCapability] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3

    # Resource constraints
    compute_budget: float = 100.0
    memory_budget: float = 100.0

    # Performance tracking
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_completion_time: float = 0.0

    # Communication
    inbox: List["Message"] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.now)

    # Hierarchical organization
    supervisor_id: Optional[str] = None
    subordinate_ids: List[str] = field(default_factory=list)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a capability."""
        return any(c.name == capability_name for c in self.capabilities)

    def get_capability_proficiency(self, capability_name: str) -> float:
        """Get proficiency for a capability."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap.proficiency
        return 0.0

    def is_available(self) -> bool:
        """Check if agent can take more tasks."""
        return (
            self.status == AgentStatus.IDLE or
            (self.status == AgentStatus.BUSY and
             len(self.current_tasks) < self.max_concurrent_tasks)
        )

    def success_rate(self) -> float:
        """Get agent's success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.5
        return self.tasks_completed / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": [c.name for c in self.capabilities],
            "current_tasks": len(self.current_tasks),
            "tasks_completed": self.tasks_completed,
            "success_rate": self.success_rate(),
        }


@dataclass
class Message:
    """A message between agents."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # Empty for broadcast
    message_type: MessageType = MessageType.HEARTBEAT
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Bid:
    """A bid from an agent for a task."""

    agent_id: str
    task_id: str
    estimated_cost: float
    estimated_time: float
    confidence: float
    capabilities_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskAllocation:
    """An allocation of a task to an agent."""

    task_id: str
    agent_id: str
    allocated_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    status: str = "allocated"
    progress: float = 0.0


class AgentRegistry:
    """
    Registry of all agents in the system.

    Handles agent lifecycle and discovery.
    """

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._by_role: Dict[AgentRole, Set[str]] = defaultdict(set)
        self._by_capability: Dict[str, Set[str]] = defaultdict(set)

    def register(self, agent: Agent):
        """Register a new agent."""
        self._agents[agent.id] = agent
        self._by_role[agent.role].add(agent.id)

        for cap in agent.capabilities:
            self._by_capability[cap.name].add(agent.id)

        logger.info("agent_registered", agent_id=agent.id, role=agent.role.value)

    def unregister(self, agent_id: str):
        """Unregister an agent."""
        if agent_id not in self._agents:
            return

        agent = self._agents[agent_id]
        self._by_role[agent.role].discard(agent_id)

        for cap in agent.capabilities:
            self._by_capability[cap.name].discard(agent_id)

        del self._agents[agent_id]
        logger.info("agent_unregistered", agent_id=agent_id)

    def get(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_all(self) -> List[Agent]:
        """Get all agents."""
        return list(self._agents.values())

    def get_by_role(self, role: AgentRole) -> List[Agent]:
        """Get agents by role."""
        return [self._agents[aid] for aid in self._by_role[role] if aid in self._agents]

    def get_by_capability(self, capability: str) -> List[Agent]:
        """Get agents with a capability."""
        return [
            self._agents[aid]
            for aid in self._by_capability[capability]
            if aid in self._agents
        ]

    def get_available(self) -> List[Agent]:
        """Get available agents."""
        return [a for a in self._agents.values() if a.is_available()]

    def update_heartbeat(self, agent_id: str):
        """Update agent heartbeat."""
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = datetime.now()

    def get_stale_agents(self, threshold_seconds: float = 60) -> List[Agent]:
        """Get agents that haven't sent heartbeat recently."""
        threshold = datetime.now() - timedelta(seconds=threshold_seconds)
        return [
            a for a in self._agents.values()
            if a.last_heartbeat < threshold
        ]


class AuctionAllocator:
    """
    Contract Net Protocol based task allocation.

    Uses auction mechanism for optimal task distribution.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        bid_timeout: float = 5.0,
    ):
        self.registry = registry
        self.bid_timeout = bid_timeout

        self._active_auctions: Dict[str, Dict[str, Any]] = {}
        self._allocations: Dict[str, TaskAllocation] = {}

    async def allocate_task(
        self,
        goal: Goal,
        required_capabilities: List[str] = None,
    ) -> Optional[TaskAllocation]:
        """
        Allocate a task using auction mechanism.

        1. Announce task to eligible agents
        2. Collect bids
        3. Award to best bidder
        """
        auction_id = str(uuid.uuid4())
        required_capabilities = required_capabilities or []

        # Find eligible agents
        eligible = self.registry.get_available()

        if required_capabilities:
            eligible = [
                a for a in eligible
                if all(a.has_capability(c) for c in required_capabilities)
            ]

        if not eligible:
            logger.warning("no_eligible_agents", goal_id=goal.id)
            return None

        # Announce task
        self._active_auctions[auction_id] = {
            "goal": goal,
            "bids": [],
            "start_time": datetime.now(),
        }

        announcement = Message(
            message_type=MessageType.TASK_ANNOUNCEMENT,
            content={
                "auction_id": auction_id,
                "goal_id": goal.id,
                "goal_title": goal.title,
                "required_capabilities": required_capabilities,
                "priority": goal.priority.value,
            },
        )

        # Send to eligible agents
        for agent in eligible:
            agent.inbox.append(announcement)

        # Wait for bids
        await asyncio.sleep(self.bid_timeout)

        # Collect bids
        bids = self._active_auctions[auction_id]["bids"]

        if not bids:
            logger.warning("no_bids_received", goal_id=goal.id)
            del self._active_auctions[auction_id]
            return None

        # Select winner (lowest cost adjusted by capability)
        def bid_score(bid: Bid) -> float:
            agent = self.registry.get(bid.agent_id)
            if not agent:
                return float('inf')

            # Adjust cost by agent's success rate and proficiency
            adjusted_cost = bid.estimated_cost / (agent.success_rate() + 0.1)

            # Factor in confidence
            adjusted_cost /= (bid.confidence + 0.1)

            return adjusted_cost

        winner_bid = min(bids, key=bid_score)

        # Create allocation
        allocation = TaskAllocation(
            task_id=goal.id,
            agent_id=winner_bid.agent_id,
        )

        self._allocations[goal.id] = allocation

        # Notify winner
        winner = self.registry.get(winner_bid.agent_id)
        if winner:
            winner.current_tasks.append(goal.id)
            winner.status = AgentStatus.BUSY

            award_msg = Message(
                message_type=MessageType.AWARD,
                receiver_id=winner_bid.agent_id,
                content={"goal_id": goal.id, "auction_id": auction_id},
            )
            winner.inbox.append(award_msg)

        # Cleanup
        del self._active_auctions[auction_id]

        logger.info(
            "task_allocated",
            goal_id=goal.id,
            agent_id=winner_bid.agent_id,
            cost=winner_bid.estimated_cost,
        )

        return allocation

    def submit_bid(self, bid: Bid):
        """Submit a bid for an auction."""
        # Find active auction for this task
        for auction_id, auction in self._active_auctions.items():
            if auction["goal"].id == bid.task_id:
                auction["bids"].append(bid)
                return True
        return False

    def get_allocation(self, task_id: str) -> Optional[TaskAllocation]:
        """Get allocation for a task."""
        return self._allocations.get(task_id)

    def complete_task(self, task_id: str, success: bool):
        """Mark a task as complete."""
        if task_id not in self._allocations:
            return

        allocation = self._allocations[task_id]
        allocation.status = "completed" if success else "failed"

        agent = self.registry.get(allocation.agent_id)
        if agent:
            if task_id in agent.current_tasks:
                agent.current_tasks.remove(task_id)

            if success:
                agent.tasks_completed += 1
            else:
                agent.tasks_failed += 1

            if not agent.current_tasks:
                agent.status = AgentStatus.IDLE


class ConsensusProtocol:
    """
    Consensus protocol for distributed decisions.

    Implements Raft-like consensus for multi-agent agreement.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        quorum_fraction: float = 0.5,
        timeout: float = 10.0,
    ):
        self.registry = registry
        self.quorum_fraction = quorum_fraction
        self.timeout = timeout

        self._proposals: Dict[str, Dict[str, Any]] = {}
        self._current_term = 0
        self._leader_id: Optional[str] = None

    def get_quorum_size(self) -> int:
        """Get required quorum size."""
        total = len(self.registry.get_all())
        return max(1, int(total * self.quorum_fraction) + 1)

    async def propose(
        self,
        proposer_id: str,
        proposal: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Propose a decision for consensus.

        Returns (accepted, proposal_id).
        """
        proposal_id = str(uuid.uuid4())
        self._current_term += 1

        self._proposals[proposal_id] = {
            "term": self._current_term,
            "proposer_id": proposer_id,
            "proposal": proposal,
            "votes_for": set(),
            "votes_against": set(),
            "status": "pending",
        }

        # Send proposal to all agents
        agents = self.registry.get_all()
        proposal_msg = Message(
            sender_id=proposer_id,
            message_type=MessageType.CONSENSUS_PROPOSE,
            content={
                "proposal_id": proposal_id,
                "term": self._current_term,
                "proposal": proposal,
            },
        )

        for agent in agents:
            agent.inbox.append(proposal_msg)

        # Wait for votes
        await asyncio.sleep(self.timeout)

        # Count votes
        prop_data = self._proposals[proposal_id]
        votes_for = len(prop_data["votes_for"])
        votes_against = len(prop_data["votes_against"])
        quorum = self.get_quorum_size()

        accepted = votes_for >= quorum

        if accepted:
            prop_data["status"] = "accepted"

            # Send commit to all
            commit_msg = Message(
                message_type=MessageType.CONSENSUS_COMMIT,
                content={"proposal_id": proposal_id, "accepted": True},
            )
            for agent in agents:
                agent.inbox.append(commit_msg)
        else:
            prop_data["status"] = "rejected"

        logger.info(
            "consensus_result",
            proposal_id=proposal_id,
            accepted=accepted,
            votes_for=votes_for,
            votes_against=votes_against,
        )

        return accepted, proposal_id

    def vote(
        self,
        voter_id: str,
        proposal_id: str,
        vote: bool,
    ):
        """Submit a vote for a proposal."""
        if proposal_id not in self._proposals:
            return

        prop_data = self._proposals[proposal_id]

        if vote:
            prop_data["votes_for"].add(voter_id)
        else:
            prop_data["votes_against"].add(voter_id)

    def get_proposal_status(self, proposal_id: str) -> Optional[str]:
        """Get status of a proposal."""
        if proposal_id in self._proposals:
            return self._proposals[proposal_id]["status"]
        return None


class ConflictResolver:
    """
    Resolves conflicts between agents.

    Handles resource conflicts, goal conflicts, and priority disputes.
    """

    def __init__(self):
        self._conflict_history: List[Dict[str, Any]] = []

    def detect_conflict(
        self,
        agents: List[Agent],
        goal: Goal,
    ) -> Optional[Dict[str, Any]]:
        """Detect potential conflicts for a goal."""
        # Resource conflict
        total_required = len(goal.success_criteria) * 10
        total_available = sum(a.compute_budget for a in agents)

        if total_required > total_available:
            return {
                "type": "resource",
                "description": "Insufficient compute budget",
                "required": total_required,
                "available": total_available,
            }

        # Capability conflict
        required_caps = set()  # Would be extracted from goal
        agent_caps = set()
        for a in agents:
            agent_caps.update(c.name for c in a.capabilities)

        missing = required_caps - agent_caps
        if missing:
            return {
                "type": "capability",
                "description": f"Missing capabilities: {missing}",
                "missing": list(missing),
            }

        return None

    def resolve_priority_conflict(
        self,
        goals: List[Goal],
    ) -> List[Goal]:
        """Resolve conflicts by priority ordering."""
        # Sort by priority then deadline
        def sort_key(g: Goal) -> Tuple[int, datetime]:
            priority_order = {
                GoalPriority.CRITICAL: 0,
                GoalPriority.HIGH: 1,
                GoalPriority.MEDIUM: 2,
                GoalPriority.LOW: 3,
            }
            deadline = g.deadline or datetime.max
            return (priority_order.get(g.priority, 2), deadline)

        return sorted(goals, key=sort_key)

    def resolve_resource_conflict(
        self,
        agents: List[Agent],
        resource_name: str,
        requested: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Resolve resource allocation conflict.

        Uses proportional allocation based on need and priority.
        """
        total_requested = sum(requested.values())
        total_available = sum(
            getattr(a, f"{resource_name}_budget", 0)
            for a in agents
        )

        if total_requested <= total_available:
            return requested  # No conflict

        # Proportional allocation
        ratio = total_available / total_requested
        allocated = {
            agent_id: amount * ratio
            for agent_id, amount in requested.items()
        }

        self._conflict_history.append({
            "type": "resource",
            "resource": resource_name,
            "requested": requested,
            "allocated": allocated,
            "time": datetime.now().isoformat(),
        })

        return allocated

    def negotiate(
        self,
        agent_preferences: Dict[str, List[str]],  # agent_id -> ordered preferences
    ) -> Dict[str, str]:
        """
        Negotiate allocation using stable matching.

        Based on Gale-Shapley algorithm.
        """
        # Simplified stable matching
        assignments = {}
        unassigned_agents = list(agent_preferences.keys())
        assigned_items = set()

        while unassigned_agents:
            agent_id = unassigned_agents.pop(0)
            prefs = agent_preferences.get(agent_id, [])

            for item in prefs:
                if item not in assigned_items:
                    assignments[agent_id] = item
                    assigned_items.add(item)
                    break
            else:
                # No item available, agent unassigned
                assignments[agent_id] = None

        return assignments


class CoalitionFormation:
    """
    Forms coalitions of agents for complex goals.

    Uses game-theoretic approaches for optimal team formation.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._coalitions: Dict[str, Set[str]] = {}  # goal_id -> agent_ids

    def form_coalition(
        self,
        goal: Goal,
        required_capabilities: List[str],
        max_size: int = 5,
    ) -> Optional[Set[str]]:
        """
        Form a coalition for a goal.

        Finds minimal set of agents that covers required capabilities.
        """
        available = self.registry.get_available()

        if not available:
            return None

        # Greedy set cover
        coalition = set()
        uncovered = set(required_capabilities)

        while uncovered and len(coalition) < max_size:
            # Find agent that covers most uncovered capabilities
            best_agent = None
            best_coverage = 0

            for agent in available:
                if agent.id in coalition:
                    continue

                coverage = len(uncovered & {c.name for c in agent.capabilities})

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_agent = agent

            if best_agent is None or best_coverage == 0:
                break

            coalition.add(best_agent.id)
            uncovered -= {c.name for c in best_agent.capabilities}

        if uncovered:
            logger.warning(
                "incomplete_coalition",
                goal_id=goal.id,
                uncovered=list(uncovered),
            )

        self._coalitions[goal.id] = coalition
        return coalition

    def get_coalition_value(
        self,
        agent_ids: Set[str],
        goal: Goal,
    ) -> float:
        """
        Calculate coalition value using Shapley value approximation.
        """
        if not agent_ids:
            return 0.0

        agents = [self.registry.get(aid) for aid in agent_ids]
        agents = [a for a in agents if a is not None]

        if not agents:
            return 0.0

        # Base value from capability coverage
        all_caps = set()
        for a in agents:
            all_caps.update(c.name for c in a.capabilities)

        # More capabilities = higher value
        cap_value = len(all_caps) * 10

        # Synergy bonus for complementary capabilities
        synergy = 0
        for i, a1 in enumerate(agents):
            for a2 in agents[i + 1:]:
                caps1 = {c.name for c in a1.capabilities}
                caps2 = {c.name for c in a2.capabilities}
                if not caps1 & caps2:  # No overlap = complementary
                    synergy += 5

        # Success rate bonus
        avg_success = sum(a.success_rate() for a in agents) / len(agents)
        success_bonus = avg_success * 20

        return cap_value + synergy + success_bonus

    def dissolve_coalition(self, goal_id: str):
        """Dissolve a coalition."""
        if goal_id in self._coalitions:
            del self._coalitions[goal_id]

    def get_coalition(self, goal_id: str) -> Optional[Set[str]]:
        """Get coalition for a goal."""
        return self._coalitions.get(goal_id)


class MessageBroker:
    """
    Message broker for agent communication.

    Handles message routing and delivery.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._message_queue: List[Message] = []
        self._delivered: List[Message] = []
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> agent_ids

    def send(self, message: Message):
        """Send a message."""
        self._message_queue.append(message)

    def broadcast(self, message: Message, role: AgentRole = None):
        """Broadcast message to all agents or agents with specific role."""
        if role:
            agents = self.registry.get_by_role(role)
        else:
            agents = self.registry.get_all()

        for agent in agents:
            msg_copy = Message(
                sender_id=message.sender_id,
                receiver_id=agent.id,
                message_type=message.message_type,
                content=message.content.copy(),
                reply_to=message.reply_to,
                priority=message.priority,
            )
            self.send(msg_copy)

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic."""
        self._subscriptions[topic].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic."""
        self._subscriptions[topic].discard(agent_id)

    def publish(self, topic: str, message: Message):
        """Publish message to topic subscribers."""
        for agent_id in self._subscriptions[topic]:
            msg_copy = Message(
                sender_id=message.sender_id,
                receiver_id=agent_id,
                message_type=message.message_type,
                content={**message.content, "topic": topic},
                priority=message.priority,
            )
            self.send(msg_copy)

    async def process_messages(self):
        """Process and deliver pending messages."""
        while self._message_queue:
            message = heapq.heappop(self._message_queue) if hasattr(heapq, 'heappop') else self._message_queue.pop(0)

            if message.receiver_id:
                # Direct message
                agent = self.registry.get(message.receiver_id)
                if agent:
                    agent.inbox.append(message)
                    self._delivered.append(message)
            else:
                # Broadcast already handled
                self._delivered.append(message)

    def get_pending_count(self) -> int:
        """Get count of pending messages."""
        return len(self._message_queue)


class MultiAgentCoordinator:
    """
    Complete multi-agent coordination system.

    Integrates all components for distributed goal pursuit.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self.allocator = AuctionAllocator(self.registry)
        self.consensus = ConsensusProtocol(self.registry)
        self.conflict_resolver = ConflictResolver()
        self.coalition_formation = CoalitionFormation(self.registry)
        self.message_broker = MessageBroker(self.registry)

        self._goal_assignments: Dict[str, str] = {}  # goal_id -> agent_id or coalition_id
        self._initialized = False
        self._background_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the coordinator."""
        # Create default coordinator agent
        coordinator = Agent(
            id="coordinator_0",
            name="Primary Coordinator",
            role=AgentRole.COORDINATOR,
            capabilities=[
                AgentCapability("coordination", 1.0),
                AgentCapability("planning", 0.9),
            ],
        )
        self.registry.register(coordinator)

        # Start background tasks
        self._background_task = asyncio.create_task(self._background_loop())

        self._initialized = True
        logger.info("multi_agent_coordinator_initialized")

    async def shutdown(self):
        """Shutdown the coordinator."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("multi_agent_coordinator_shutdown")

    async def _background_loop(self):
        """Background loop for coordination tasks."""
        while True:
            try:
                await asyncio.sleep(5)

                # Process messages
                await self.message_broker.process_messages()

                # Check for stale agents
                stale = self.registry.get_stale_agents(60)
                for agent in stale:
                    agent.status = AgentStatus.OFFLINE
                    logger.warning("agent_stale", agent_id=agent.id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("background_loop_error", error=str(e))

    def register_agent(
        self,
        name: str,
        role: AgentRole = AgentRole.WORKER,
        capabilities: List[str] = None,
    ) -> Agent:
        """Register a new agent."""
        agent = Agent(
            id=str(uuid.uuid4()),
            name=name,
            role=role,
            capabilities=[
                AgentCapability(name=cap)
                for cap in (capabilities or [])
            ],
        )
        self.registry.register(agent)
        return agent

    async def assign_goal(
        self,
        goal: Goal,
        required_capabilities: List[str] = None,
        prefer_coalition: bool = False,
    ) -> Optional[str]:
        """
        Assign a goal to an agent or coalition.

        Returns agent_id or coalition_id.
        """
        required_capabilities = required_capabilities or []

        if prefer_coalition and len(required_capabilities) > 2:
            # Form coalition for complex goals
            coalition = self.coalition_formation.form_coalition(
                goal, required_capabilities
            )

            if coalition:
                coalition_id = f"coalition_{goal.id}"
                self._goal_assignments[goal.id] = coalition_id
                return coalition_id

        # Single agent allocation
        allocation = await self.allocator.allocate_task(
            goal, required_capabilities
        )

        if allocation:
            self._goal_assignments[goal.id] = allocation.agent_id
            return allocation.agent_id

        return None

    async def reach_consensus(
        self,
        proposer_id: str,
        proposal: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Reach consensus on a proposal."""
        return await self.consensus.propose(proposer_id, proposal)

    def resolve_conflicts(
        self,
        goals: List[Goal],
    ) -> List[Goal]:
        """Resolve goal conflicts."""
        return self.conflict_resolver.resolve_priority_conflict(goals)

    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
    ):
        """Send a message between agents."""
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
        )
        self.message_broker.send(message)

    def broadcast_message(
        self,
        sender_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        role: AgentRole = None,
    ):
        """Broadcast a message."""
        message = Message(
            sender_id=sender_id,
            message_type=message_type,
            content=content,
        )
        self.message_broker.broadcast(message, role)

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agents."""
        agents = self.registry.get_all()

        by_role = defaultdict(int)
        by_status = defaultdict(int)
        total_completed = 0
        total_failed = 0

        for agent in agents:
            by_role[agent.role.value] += 1
            by_status[agent.status.value] += 1
            total_completed += agent.tasks_completed
            total_failed += agent.tasks_failed

        return {
            "total_agents": len(agents),
            "by_role": dict(by_role),
            "by_status": dict(by_status),
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "avg_success_rate": (
                total_completed / (total_completed + total_failed)
                if (total_completed + total_failed) > 0
                else 0.0
            ),
        }

    def get_assignment(self, goal_id: str) -> Optional[str]:
        """Get assignment for a goal."""
        return self._goal_assignments.get(goal_id)

    def complete_goal(self, goal_id: str, success: bool):
        """Mark a goal as complete."""
        assignment = self._goal_assignments.get(goal_id)

        if assignment and assignment.startswith("coalition_"):
            # Coalition completion
            self.coalition_formation.dissolve_coalition(goal_id)
        elif assignment:
            # Single agent completion
            self.allocator.complete_task(goal_id, success)

        if goal_id in self._goal_assignments:
            del self._goal_assignments[goal_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "agents": self.get_agent_stats(),
            "active_assignments": len(self._goal_assignments),
            "active_coalitions": len(self.coalition_formation._coalitions),
            "pending_messages": self.message_broker.get_pending_count(),
            "consensus_term": self.consensus._current_term,
        }
