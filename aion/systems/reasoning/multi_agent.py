"""
AION Multi-Agent Coordination System

State-of-the-art multi-agent coordination with:
- Multi-Agent Debate (MAD) for improved reasoning
- Role-based specialization
- Hierarchical task decomposition
- Emergent coordination protocols
- Consensus mechanisms
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class AgentRole(Enum):
    """Specialized agent roles."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"
    PLANNER = "planner"
    SPECIALIST = "specialist"


class CoordinationStrategy(Enum):
    """Multi-agent coordination strategies."""
    CENTRALIZED = "centralized"  # Single coordinator
    DECENTRALIZED = "decentralized"  # Peer-to-peer
    HIERARCHICAL = "hierarchical"  # Tree structure
    MARKET = "market"  # Auction-based
    DEBATE = "debate"  # Multi-agent debate


@dataclass
class AgentMessage:
    """Message between agents."""
    id: str
    sender: str
    receiver: str  # Can be "broadcast" for all agents
    content: str
    message_type: str  # query, response, proposal, vote, critique
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Internal state of an agent."""
    beliefs: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    working_memory: list[dict] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class Agent:
    """
    An autonomous agent in the multi-agent system.

    Each agent has:
    - A specialized role
    - Internal beliefs and goals
    - Communication capabilities
    - Reasoning abilities
    """
    id: str
    name: str
    role: AgentRole
    capabilities: list[str] = field(default_factory=list)
    state: AgentState = field(default_factory=AgentState)

    # Communication
    inbox: list[AgentMessage] = field(default_factory=list)
    outbox: list[AgentMessage] = field(default_factory=list)

    # Performance tracking
    tasks_completed: int = 0
    success_rate: float = 1.0

    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.inbox.append(message)
        self.state.working_memory.append({
            "type": "received_message",
            "from": message.sender,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
        })

    def send_message(
        self,
        receiver: str,
        content: str,
        message_type: str,
        metadata: Optional[dict] = None,
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )
        self.outbox.append(message)
        return message

    def update_belief(self, key: str, value: Any, confidence: float = 0.8) -> None:
        """Update a belief with confidence."""
        self.state.beliefs[key] = {
            "value": value,
            "confidence": confidence,
            "updated": datetime.now().isoformat(),
        }


@dataclass
class DebateRound:
    """A round of multi-agent debate."""
    round_number: int
    topic: str
    arguments: list[dict] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    consensus: Optional[str] = None


class MultiAgentDebate:
    """
    Multi-Agent Debate (MAD) system.

    Implements debate-based reasoning where multiple agents:
    1. Present initial positions
    2. Critique each other's arguments
    3. Refine positions based on critiques
    4. Reach consensus or majority decision

    Based on "Improving Factuality and Reasoning in LLMs through Multiagent Debate"
    """

    def __init__(self, llm_adapter, num_rounds: int = 3):
        self.llm = llm_adapter
        self.num_rounds = num_rounds
        self.debate_history: list[DebateRound] = []

    async def debate(
        self,
        question: str,
        agents: list[Agent],
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Conduct a multi-agent debate on a question.

        Args:
            question: The question to debate
            agents: Participating agents
            context: Optional context

        Returns:
            Debate result with consensus and reasoning
        """
        from aion.core.llm import Message

        self.debate_history = []

        # Phase 1: Initial positions
        positions = {}
        for agent in agents:
            prompt = f"""You are {agent.name}, a {agent.role.value} agent.

Question: {question}

Context: {context or 'None provided'}

Provide your initial position on this question. Be specific and justify your reasoning.

Your capabilities: {agent.capabilities}
Your current beliefs: {json.dumps(agent.state.beliefs, default=str)}

POSITION: <your position>
REASONING: <step by step reasoning>
CONFIDENCE: <0-1>
"""
            try:
                response = await self.llm.complete([
                    Message(role="system", content=f"You are {agent.name} with role {agent.role.value}."),
                    Message(role="user", content=prompt),
                ])
                positions[agent.id] = {
                    "agent": agent.name,
                    "role": agent.role.value,
                    "position": response.content,
                    "round": 0,
                }
            except Exception as e:
                logger.warning(f"Agent {agent.name} failed to respond", error=str(e))
                positions[agent.id] = {
                    "agent": agent.name,
                    "position": "Unable to form position",
                    "round": 0,
                }

        # Phase 2: Debate rounds
        for round_num in range(1, self.num_rounds + 1):
            debate_round = DebateRound(
                round_number=round_num,
                topic=question,
            )

            # Each agent critiques others and refines position
            new_positions = {}
            for agent in agents:
                other_positions = {
                    aid: pos for aid, pos in positions.items()
                    if aid != agent.id
                }

                prompt = f"""You are {agent.name}, a {agent.role.value} agent in round {round_num} of a debate.

Question: {question}

Your current position:
{positions[agent.id]['position']}

Other agents' positions:
{json.dumps(other_positions, indent=2, default=str)}

Instructions:
1. Critique the other positions - identify flaws and strengths
2. Consider if your position should change based on valid criticisms
3. Provide your updated position

CRITIQUES:
<critique each other agent's position>

UPDATED_POSITION: <your refined position>
KEY_INSIGHT: <what changed your thinking, or why you held firm>
CONFIDENCE: <0-1>
"""
                try:
                    response = await self.llm.complete([
                        Message(role="system", content=f"You are engaged in a rational debate. Be open to changing your mind."),
                        Message(role="user", content=prompt),
                    ])
                    new_positions[agent.id] = {
                        "agent": agent.name,
                        "role": agent.role.value,
                        "position": response.content,
                        "round": round_num,
                    }
                    debate_round.arguments.append({
                        "agent": agent.name,
                        "content": response.content,
                    })
                except Exception as e:
                    logger.warning(f"Round {round_num}: Agent {agent.name} failed", error=str(e))
                    new_positions[agent.id] = positions[agent.id]

            positions = new_positions
            self.debate_history.append(debate_round)

        # Phase 3: Final voting and consensus
        votes = {}
        final_positions = []

        for agent in agents:
            # Extract final position
            import re
            pos_text = positions[agent.id]["position"]
            pos_match = re.search(r'UPDATED_POSITION:\s*(.+?)(?=KEY_INSIGHT|CONFIDENCE|$)', pos_text, re.DOTALL)
            final_pos = pos_match.group(1).strip() if pos_match else pos_text[:500]
            final_positions.append(final_pos)
            votes[agent.id] = final_pos

        # Synthesize consensus
        synthesis_prompt = f"""Synthesize the final positions from a multi-agent debate.

Question: {question}

Final positions from agents:
{json.dumps([{"agent": positions[a.id]["agent"], "position": votes[a.id]} for a in agents], indent=2)}

Create a synthesis that:
1. Identifies points of agreement
2. Resolves points of disagreement
3. Provides the best answer based on the collective reasoning

CONSENSUS: <synthesized answer>
AGREEMENT_LEVEL: <high/medium/low>
KEY_POINTS: <bullet points of main conclusions>
"""

        try:
            synthesis = await self.llm.complete([
                Message(role="system", content="You synthesize debate conclusions fairly."),
                Message(role="user", content=synthesis_prompt),
            ])

            return {
                "question": question,
                "num_rounds": self.num_rounds,
                "num_agents": len(agents),
                "final_positions": positions,
                "consensus": synthesis.content,
                "debate_history": [
                    {
                        "round": r.round_number,
                        "num_arguments": len(r.arguments),
                    }
                    for r in self.debate_history
                ],
            }
        except Exception as e:
            logger.warning("Consensus synthesis failed", error=str(e))
            return {
                "question": question,
                "final_positions": positions,
                "consensus": "Unable to synthesize consensus",
                "error": str(e),
            }


class TaskAllocator:
    """
    Intelligent task allocation for multi-agent systems.

    Uses capability matching, load balancing, and auction mechanisms.
    """

    def __init__(self):
        self.allocation_history: list[dict] = []

    def allocate_by_capability(
        self,
        task: dict,
        agents: list[Agent],
    ) -> Optional[Agent]:
        """Allocate task to most capable agent."""
        required_caps = set(task.get("required_capabilities", []))

        best_agent = None
        best_score = -1

        for agent in agents:
            agent_caps = set(agent.capabilities)
            match_score = len(required_caps & agent_caps) / max(len(required_caps), 1)

            # Factor in success rate and current load
            adjusted_score = match_score * agent.success_rate

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_agent = agent

        if best_agent:
            self.allocation_history.append({
                "task": task.get("id", "unknown"),
                "agent": best_agent.id,
                "score": best_score,
                "timestamp": datetime.now().isoformat(),
            })

        return best_agent

    async def auction_allocate(
        self,
        task: dict,
        agents: list[Agent],
        llm_adapter,
    ) -> Optional[Agent]:
        """
        Allocate task via auction mechanism.

        Agents bid based on their capability and availability.
        """
        from aion.core.llm import Message

        bids = []

        for agent in agents:
            prompt = f"""You are {agent.name}, a {agent.role.value} agent.

A task is available for bidding:
{json.dumps(task, default=str)}

Your capabilities: {agent.capabilities}
Your current goals: {agent.state.goals}
Your success rate: {agent.success_rate}

Should you bid on this task? If yes, how confident are you that you can complete it well?

BID: yes/no
CONFIDENCE: <0-1 if bidding>
REASONING: <brief explanation>
"""
            try:
                response = await llm_adapter.complete([
                    Message(role="system", content="Evaluate if you should bid on this task."),
                    Message(role="user", content=prompt),
                ])

                import re
                bid_match = re.search(r'BID:\s*(yes|no)', response.content.lower())
                conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response.content)

                if bid_match and bid_match.group(1) == "yes":
                    confidence = float(conf_match.group(1)) if conf_match else 0.5
                    bids.append({
                        "agent": agent,
                        "confidence": confidence,
                        "response": response.content,
                    })
            except Exception as e:
                logger.warning(f"Agent {agent.name} failed to bid", error=str(e))

        if not bids:
            return None

        # Select highest bidder
        winner = max(bids, key=lambda b: b["confidence"])

        self.allocation_history.append({
            "task": task.get("id", "unknown"),
            "agent": winner["agent"].id,
            "bid_confidence": winner["confidence"],
            "num_bidders": len(bids),
            "timestamp": datetime.now().isoformat(),
        })

        return winner["agent"]


class HierarchicalCoordinator:
    """
    Hierarchical task decomposition and coordination.

    Creates a tree of agents where:
    - Coordinator decomposes high-level tasks
    - Specialists handle leaf tasks
    - Results propagate up the hierarchy
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        self.task_tree: dict = {}

    async def decompose_task(
        self,
        task: str,
        available_agents: list[Agent],
    ) -> dict:
        """
        Decompose a complex task into subtasks for different agents.
        """
        from aion.core.llm import Message

        agent_desc = "\n".join([
            f"- {a.name} ({a.role.value}): {a.capabilities}"
            for a in available_agents
        ])

        prompt = f"""Decompose this complex task into subtasks for a multi-agent team.

Task: {task}

Available agents:
{agent_desc}

Create a hierarchical decomposition where:
1. Each subtask matches an agent's capabilities
2. Dependencies between subtasks are clear
3. The decomposition covers the full task

Format:
SUBTASK_1:
  description: <what to do>
  assigned_to: <agent name>
  dependencies: <list of subtask IDs or "none">

SUBTASK_2:
  ...

COORDINATION_STRATEGY: <how agents should coordinate>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You decompose complex tasks for multi-agent execution."),
                Message(role="user", content=prompt),
            ])

            # Parse subtasks
            import re
            subtasks = []
            current_subtask = {}

            for line in response.content.split('\n'):
                if re.match(r'SUBTASK_\d+:', line):
                    if current_subtask:
                        subtasks.append(current_subtask)
                    current_subtask = {"id": line.strip().rstrip(':')}
                elif 'description:' in line.lower():
                    current_subtask["description"] = line.split(':', 1)[1].strip()
                elif 'assigned_to:' in line.lower():
                    current_subtask["assigned_to"] = line.split(':', 1)[1].strip()
                elif 'dependencies:' in line.lower():
                    deps = line.split(':', 1)[1].strip()
                    current_subtask["dependencies"] = [] if deps.lower() == "none" else deps.split(',')

            if current_subtask:
                subtasks.append(current_subtask)

            self.task_tree = {
                "root_task": task,
                "subtasks": subtasks,
                "raw_decomposition": response.content,
            }

            return self.task_tree

        except Exception as e:
            logger.warning("Task decomposition failed", error=str(e))
            return {"error": str(e)}


class MultiAgentCoordinator:
    """
    Central coordinator for multi-agent systems.

    Implements SOTA coordination including:
    - Multi-Agent Debate for complex reasoning
    - Role-based agent management
    - Hierarchical task decomposition
    - Consensus mechanisms
    - Emergent coordination
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

        # Agent management
        self.agents: dict[str, Agent] = {}
        self.agent_teams: dict[str, list[str]] = {}

        # Coordination systems
        self.debate_system = MultiAgentDebate(llm_adapter)
        self.task_allocator = TaskAllocator()
        self.hierarchical_coordinator = HierarchicalCoordinator(llm_adapter)

        # Message bus
        self.message_queue: list[AgentMessage] = []
        self.message_history: list[AgentMessage] = []

        # Coordination strategy
        self.default_strategy = CoordinationStrategy.DEBATE

    def create_agent(
        self,
        name: str,
        role: AgentRole,
        capabilities: Optional[list[str]] = None,
    ) -> Agent:
        """Create and register a new agent."""
        agent = Agent(
            id=str(uuid.uuid4()),
            name=name,
            role=role,
            capabilities=capabilities or [],
        )
        self.agents[agent.id] = agent
        logger.info(f"Created agent: {name} ({role.value})")
        return agent

    def create_default_team(self) -> list[Agent]:
        """Create a default team with diverse roles."""
        team = [
            self.create_agent(
                "Planner",
                AgentRole.PLANNER,
                ["task_decomposition", "strategy", "prioritization"],
            ),
            self.create_agent(
                "Researcher",
                AgentRole.RESEARCHER,
                ["information_gathering", "analysis", "synthesis"],
            ),
            self.create_agent(
                "Critic",
                AgentRole.CRITIC,
                ["evaluation", "error_detection", "improvement_suggestions"],
            ),
            self.create_agent(
                "Executor",
                AgentRole.EXECUTOR,
                ["implementation", "action_execution", "tool_use"],
            ),
            self.create_agent(
                "Verifier",
                AgentRole.VERIFIER,
                ["validation", "testing", "quality_assurance"],
            ),
        ]

        self.agent_teams["default"] = [a.id for a in team]
        return team

    def form_team(
        self,
        team_name: str,
        agent_ids: list[str],
    ) -> list[Agent]:
        """Form a team from existing agents."""
        team_agents = [self.agents[aid] for aid in agent_ids if aid in self.agents]
        self.agent_teams[team_name] = agent_ids
        return team_agents

    async def coordinate(
        self,
        task: str,
        strategy: Optional[CoordinationStrategy] = None,
        team_name: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Coordinate agents to accomplish a task.

        Args:
            task: The task to accomplish
            strategy: Coordination strategy to use
            team_name: Team to use (or creates default)
            context: Optional context

        Returns:
            Coordination result
        """
        strategy = strategy or self.default_strategy

        # Get or create team
        if team_name and team_name in self.agent_teams:
            agents = [self.agents[aid] for aid in self.agent_teams[team_name]]
        elif self.agents:
            agents = list(self.agents.values())
        else:
            agents = self.create_default_team()

        logger.info(
            f"Coordinating task with {len(agents)} agents using {strategy.value} strategy"
        )

        if strategy == CoordinationStrategy.DEBATE:
            return await self._coordinate_debate(task, agents, context)
        elif strategy == CoordinationStrategy.HIERARCHICAL:
            return await self._coordinate_hierarchical(task, agents, context)
        elif strategy == CoordinationStrategy.MARKET:
            return await self._coordinate_market(task, agents, context)
        elif strategy == CoordinationStrategy.CENTRALIZED:
            return await self._coordinate_centralized(task, agents, context)
        else:
            return await self._coordinate_decentralized(task, agents, context)

    async def _coordinate_debate(
        self,
        task: str,
        agents: list[Agent],
        context: Optional[str],
    ) -> dict[str, Any]:
        """Use multi-agent debate for coordination."""
        result = await self.debate_system.debate(task, agents, context)

        # Update agent beliefs based on debate
        for agent in agents:
            agent.update_belief("last_debate_task", task)
            agent.update_belief("debate_consensus", result.get("consensus", ""))

        return {
            "strategy": "debate",
            "result": result,
            "agents_involved": [a.name for a in agents],
        }

    async def _coordinate_hierarchical(
        self,
        task: str,
        agents: list[Agent],
        context: Optional[str],
    ) -> dict[str, Any]:
        """Use hierarchical decomposition for coordination."""
        # Decompose task
        decomposition = await self.hierarchical_coordinator.decompose_task(task, agents)

        if "error" in decomposition:
            return {"strategy": "hierarchical", "error": decomposition["error"]}

        # Execute subtasks
        results = []
        for subtask in decomposition.get("subtasks", []):
            # Find assigned agent
            assigned_name = subtask.get("assigned_to", "")
            assigned_agent = None
            for agent in agents:
                if agent.name.lower() in assigned_name.lower():
                    assigned_agent = agent
                    break

            if assigned_agent:
                # Execute subtask
                result = await self._execute_agent_task(
                    assigned_agent,
                    subtask.get("description", ""),
                    context,
                )
                results.append({
                    "subtask": subtask.get("id"),
                    "agent": assigned_agent.name,
                    "result": result,
                })

        return {
            "strategy": "hierarchical",
            "decomposition": decomposition,
            "subtask_results": results,
        }

    async def _coordinate_market(
        self,
        task: str,
        agents: list[Agent],
        context: Optional[str],
    ) -> dict[str, Any]:
        """Use market/auction mechanism for coordination."""
        task_dict = {
            "id": str(uuid.uuid4()),
            "description": task,
            "context": context,
        }

        winner = await self.task_allocator.auction_allocate(
            task_dict,
            agents,
            self.llm,
        )

        if winner:
            result = await self._execute_agent_task(winner, task, context)
            return {
                "strategy": "market",
                "winning_agent": winner.name,
                "result": result,
            }
        else:
            return {
                "strategy": "market",
                "error": "No agent bid on the task",
            }

    async def _coordinate_centralized(
        self,
        task: str,
        agents: list[Agent],
        context: Optional[str],
    ) -> dict[str, Any]:
        """Use centralized coordination with a coordinator agent."""
        from aion.core.llm import Message

        # Find or create coordinator
        coordinator = None
        for agent in agents:
            if agent.role == AgentRole.COORDINATOR:
                coordinator = agent
                break

        if not coordinator:
            coordinator = self.create_agent(
                "Central Coordinator",
                AgentRole.COORDINATOR,
                ["coordination", "delegation", "synthesis"],
            )
            agents.append(coordinator)

        # Coordinator plans and delegates
        agent_desc = "\n".join([
            f"- {a.name} ({a.role.value}): {a.capabilities}"
            for a in agents if a.id != coordinator.id
        ])

        prompt = f"""You are the central coordinator. Plan how to accomplish this task.

Task: {task}

Available agents:
{agent_desc}

Context: {context or 'None'}

Create a coordination plan:
1. What should each agent do?
2. In what order?
3. How should results be combined?

PLAN:
<detailed coordination plan>

FINAL_SYNTHESIS_APPROACH:
<how to combine agent outputs>
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You are a coordinator agent."),
                Message(role="user", content=prompt),
            ])

            return {
                "strategy": "centralized",
                "coordinator": coordinator.name,
                "plan": response.content,
                "agents_available": [a.name for a in agents],
            }

        except Exception as e:
            return {"strategy": "centralized", "error": str(e)}

    async def _coordinate_decentralized(
        self,
        task: str,
        agents: list[Agent],
        context: Optional[str],
    ) -> dict[str, Any]:
        """Use decentralized peer-to-peer coordination."""
        # Each agent independently works on the task
        results = await asyncio.gather(*[
            self._execute_agent_task(agent, task, context)
            for agent in agents
        ])

        # Agents vote on best result
        from aion.core.llm import Message

        results_summary = "\n".join([
            f"Agent {agents[i].name}: {r.get('response', 'No response')[:200]}..."
            for i, r in enumerate(results)
        ])

        # Simple consensus - synthesize results
        prompt = f"""Multiple agents worked on this task independently.

Task: {task}

Results:
{results_summary}

Synthesize the best answer from these results.

SYNTHESIS: <best combined answer>
BEST_CONTRIBUTORS: <which agents contributed most>
"""

        try:
            synthesis = await self.llm.complete([
                Message(role="system", content="Synthesize multi-agent results."),
                Message(role="user", content=prompt),
            ])

            return {
                "strategy": "decentralized",
                "individual_results": [
                    {"agent": agents[i].name, "result": r}
                    for i, r in enumerate(results)
                ],
                "synthesis": synthesis.content,
            }

        except Exception as e:
            return {
                "strategy": "decentralized",
                "individual_results": [
                    {"agent": agents[i].name, "result": r}
                    for i, r in enumerate(results)
                ],
                "error": str(e),
            }

    async def _execute_agent_task(
        self,
        agent: Agent,
        task: str,
        context: Optional[str],
    ) -> dict[str, Any]:
        """Have an agent execute a task."""
        from aion.core.llm import Message

        prompt = f"""You are {agent.name}, a {agent.role.value} agent.

Your capabilities: {agent.capabilities}
Your beliefs: {json.dumps(agent.state.beliefs, default=str)}

Task: {task}

Context: {context or 'None'}

Complete this task using your capabilities.

RESPONSE: <your response to the task>
ACTIONS_TAKEN: <what you did>
CONFIDENCE: 0-1
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content=f"You are {agent.name}."),
                Message(role="user", content=prompt),
            ])

            agent.tasks_completed += 1

            return {
                "agent": agent.name,
                "response": response.content,
                "success": True,
            }

        except Exception as e:
            agent.success_rate *= 0.9  # Decrease success rate on failure
            return {
                "agent": agent.name,
                "error": str(e),
                "success": False,
            }

    def broadcast_message(
        self,
        sender_id: str,
        content: str,
        message_type: str,
    ) -> list[AgentMessage]:
        """Broadcast a message to all agents."""
        messages = []
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                msg = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender=sender_id,
                    receiver=agent_id,
                    content=content,
                    message_type=message_type,
                )
                agent.receive_message(msg)
                messages.append(msg)
                self.message_history.append(msg)

        return messages

    def get_agent_statuses(self) -> list[dict]:
        """Get status of all agents."""
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role.value,
                "tasks_completed": agent.tasks_completed,
                "success_rate": agent.success_rate,
                "num_beliefs": len(agent.state.beliefs),
                "inbox_size": len(agent.inbox),
            }
            for agent in self.agents.values()
        ]

    async def emergent_coordination(
        self,
        goal: str,
        max_rounds: int = 5,
    ) -> dict[str, Any]:
        """
        Allow agents to self-organize to achieve a goal.

        Agents communicate, form coalitions, and coordinate without
        explicit instruction through emergent behavior.
        """
        if not self.agents:
            self.create_default_team()

        agents = list(self.agents.values())
        from aion.core.llm import Message

        coordination_log = []

        for round_num in range(max_rounds):
            round_messages = []

            # Each agent decides what to communicate
            for agent in agents:
                # Get messages from other agents
                other_messages = [
                    m for m in self.message_history[-20:]
                    if m.receiver == agent.id or m.receiver == "broadcast"
                ]

                prompt = f"""You are {agent.name}, a {agent.role.value} in a self-organizing team.

Goal: {goal}

Recent messages:
{json.dumps([{"from": m.sender, "content": m.content[:100]} for m in other_messages], default=str)}

Your beliefs: {json.dumps(agent.state.beliefs, default=str)}

Round {round_num + 1} of {max_rounds}.

Decide:
1. What action should you take toward the goal?
2. What should you communicate to the team?
3. Do you need to coordinate with specific agents?

ACTION: <your action>
MESSAGE_TO_TEAM: <what to communicate (or "none")>
COORDINATE_WITH: <specific agent names or "none">
PROGRESS_ASSESSMENT: <how close is the team to the goal? 0-1>
"""

                try:
                    response = await self.llm.complete([
                        Message(role="system", content="You are part of a self-organizing team."),
                        Message(role="user", content=prompt),
                    ])

                    # Extract and broadcast message
                    import re
                    msg_match = re.search(r'MESSAGE_TO_TEAM:\s*(.+?)(?=COORDINATE_WITH|PROGRESS|$)',
                                         response.content, re.DOTALL)

                    if msg_match:
                        msg_content = msg_match.group(1).strip()
                        if msg_content.lower() != "none":
                            self.broadcast_message(agent.id, msg_content, "coordination")
                            round_messages.append({
                                "agent": agent.name,
                                "message": msg_content,
                            })

                    # Update agent belief about progress
                    prog_match = re.search(r'PROGRESS_ASSESSMENT:\s*([\d.]+)', response.content)
                    if prog_match:
                        agent.update_belief("goal_progress", float(prog_match.group(1)))

                except Exception as e:
                    logger.warning(f"Agent {agent.name} failed in emergent coordination", error=str(e))

            coordination_log.append({
                "round": round_num + 1,
                "messages": round_messages,
            })

            # Check if goal seems achieved (high average progress)
            progress_values = [
                a.state.beliefs.get("goal_progress", {}).get("value", 0.5)
                for a in agents
            ]
            avg_progress = sum(progress_values) / len(progress_values) if progress_values else 0

            if avg_progress > 0.9:
                logger.info(f"Goal appears achieved after {round_num + 1} rounds")
                break

        return {
            "goal": goal,
            "rounds_completed": len(coordination_log),
            "coordination_log": coordination_log,
            "final_agent_statuses": self.get_agent_statuses(),
        }
