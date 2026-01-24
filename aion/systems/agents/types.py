"""
AION Multi-Agent Orchestration Types

Core dataclasses and enums for multi-agent coordination.
Provides type-safe structures for agents, teams, messages, tasks, and consensus.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class AgentRole(str, Enum):
    """Predefined agent roles/specializations."""
    ORCHESTRATOR = "orchestrator"    # Coordinates other agents
    RESEARCHER = "researcher"        # Information gathering
    CODER = "coder"                  # Code generation/review
    ANALYST = "analyst"              # Data analysis
    WRITER = "writer"                # Content creation
    REVIEWER = "reviewer"            # Quality review
    PLANNER = "planner"              # Strategic planning
    EXECUTOR = "executor"            # Task execution
    GENERALIST = "generalist"        # General purpose
    CUSTOM = "custom"                # User-defined role

    def is_specialist(self) -> bool:
        """Check if this is a specialist role."""
        return self not in (AgentRole.GENERALIST, AgentRole.CUSTOM, AgentRole.ORCHESTRATOR)


class AgentStatus(str, Enum):
    """Status of an agent."""
    IDLE = "idle"                    # Available for tasks
    BUSY = "busy"                    # Working on a task
    WAITING = "waiting"              # Waiting for input/dependency
    PAUSED = "paused"                # Temporarily paused
    ERROR = "error"                  # In error state
    TERMINATED = "terminated"        # Shut down

    def is_available(self) -> bool:
        """Check if agent can accept new tasks."""
        return self == AgentStatus.IDLE

    def is_active(self) -> bool:
        """Check if agent is in an active state."""
        return self in (AgentStatus.IDLE, AgentStatus.BUSY, AgentStatus.WAITING, AgentStatus.PAUSED)


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK = "task"                    # Task assignment
    RESULT = "result"                # Task result
    QUERY = "query"                  # Information request
    RESPONSE = "response"            # Response to query
    BROADCAST = "broadcast"          # Message to all agents
    HANDOFF = "handoff"              # Task handoff to another agent
    STATUS = "status"                # Status update
    FEEDBACK = "feedback"            # Feedback on work
    VOTE = "vote"                    # Vote in consensus
    CONSENSUS = "consensus"          # Consensus result
    ERROR = "error"                  # Error notification
    HEARTBEAT = "heartbeat"          # Health check


class TeamStatus(str, Enum):
    """Status of a team."""
    FORMING = "forming"              # Being assembled
    ACTIVE = "active"                # Working on task
    PAUSED = "paused"                # Temporarily paused
    COMPLETED = "completed"          # Task completed
    FAILED = "failed"                # Task failed
    DISBANDED = "disbanded"          # Team dissolved

    def is_active(self) -> bool:
        """Check if team is in an active state."""
        return self in (TeamStatus.ACTIVE, TeamStatus.PAUSED)


class WorkflowPattern(str, Enum):
    """Multi-agent workflow patterns."""
    SEQUENTIAL = "sequential"        # Agents work one after another
    PARALLEL = "parallel"            # Agents work concurrently
    HIERARCHICAL = "hierarchical"    # Manager delegates to workers
    DEBATE = "debate"                # Agents debate/refine
    SWARM = "swarm"                  # Emergent coordination
    PIPELINE = "pipeline"            # Assembly line pattern
    CONSENSUS = "consensus"          # Vote-based decisions


class ConsensusMethod(str, Enum):
    """Methods for reaching consensus."""
    MAJORITY = "majority"            # Simple majority wins
    SUPERMAJORITY = "supermajority"  # 2/3 majority required
    UNANIMOUS = "unanimous"          # All must agree
    WEIGHTED = "weighted"            # Weighted by confidence
    RANKED = "ranked"                # Ranked choice voting
    BORDA = "borda"                  # Borda count method


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"            # Highest priority
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"        # Lowest priority

    def __lt__(self, other: "TaskPriority") -> bool:
        order = [TaskPriority.BACKGROUND, TaskPriority.LOW, TaskPriority.NORMAL,
                 TaskPriority.HIGH, TaskPriority.CRITICAL]
        return order.index(self) < order.index(other)


@dataclass
class AgentCapability:
    """A capability that an agent possesses."""
    name: str
    description: str
    proficiency: float = 1.0  # 0.0 to 1.0

    # What this capability enables
    can_handle_tasks: list[str] = field(default_factory=list)
    tools_required: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "proficiency": self.proficiency,
            "can_handle_tasks": self.can_handle_tasks.copy(),
            "tools_required": self.tools_required.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCapability":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentProfile:
    """Profile defining an agent's characteristics."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: AgentRole = AgentRole.GENERALIST

    # Personality and behavior
    system_prompt: str = ""
    personality_traits: list[str] = field(default_factory=list)
    communication_style: str = "professional"

    # Capabilities
    capabilities: list[AgentCapability] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # Allowed tools

    # Constraints
    max_concurrent_tasks: int = 1
    max_tokens_per_task: int = 100000
    timeout_seconds: float = 300.0

    # LLM Configuration
    model: Optional[str] = None
    temperature: float = 0.7

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle a task type."""
        task_type_lower = task_type.lower()
        for cap in self.capabilities:
            for task in cap.can_handle_tasks:
                if task.lower() in task_type_lower or task_type_lower in task.lower():
                    return True
        return False

    def get_proficiency(self, task_type: str) -> float:
        """Get proficiency for a task type."""
        best = 0.0
        task_type_lower = task_type.lower()
        for cap in self.capabilities:
            for task in cap.can_handle_tasks:
                if task.lower() in task_type_lower or task_type_lower in task.lower():
                    best = max(best, cap.proficiency)
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "system_prompt": self.system_prompt,
            "personality_traits": self.personality_traits.copy(),
            "communication_style": self.communication_style,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "tools": self.tools.copy(),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_tokens_per_task": self.max_tokens_per_task,
            "timeout_seconds": self.timeout_seconds,
            "model": self.model,
            "temperature": self.temperature,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentProfile":
        data = data.copy()
        if "role" in data:
            data["role"] = AgentRole(data["role"])
        if "capabilities" in data:
            data["capabilities"] = [AgentCapability.from_dict(c) for c in data["capabilities"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentInstance:
    """A running instance of an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile: AgentProfile = field(default_factory=AgentProfile)

    # State
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[str] = None
    current_team_id: Optional[str] = None

    # Communication
    inbox: list["Message"] = field(default_factory=list)
    outbox: list["Message"] = field(default_factory=list)

    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    total_runtime_seconds: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)

    # Internal state
    working_memory: dict[str, Any] = field(default_factory=dict)

    # Error tracking
    last_error: Optional[str] = None
    consecutive_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile": self.profile.to_dict(),
            "name": self.profile.name,
            "role": self.profile.role.value,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "current_team_id": self.current_team_id,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_tokens_used": self.total_tokens_used,
            "total_runtime_seconds": self.total_runtime_seconds,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "last_error": self.last_error,
            "consecutive_errors": self.consecutive_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentInstance":
        data = data.copy()
        if "profile" in data:
            data["profile"] = AgentProfile.from_dict(data["profile"])
        if "status" in data:
            data["status"] = AgentStatus(data["status"])
        for dt_field in ["created_at", "last_active_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        # Remove computed fields
        for key in ["name", "role"]:
            data.pop(key, None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update_activity(self) -> None:
        """Update last active timestamp."""
        self.last_active_at = datetime.now()


@dataclass
class Message:
    """A message between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Routing
    sender_id: str = ""
    recipient_id: str = ""  # Empty for broadcasts
    team_id: Optional[str] = None

    # Content
    message_type: MessageType = MessageType.QUERY
    subject: str = ""
    content: Any = None

    # Metadata
    priority: int = 5  # 1 (highest) to 10 (lowest)
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request/response pairing

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Delivery tracking
    delivered: bool = False
    delivered_at: Optional[datetime] = None
    read: bool = False
    read_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "team_id": self.team_id,
            "message_type": self.message_type.value,
            "subject": self.subject,
            "content": self.content,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "delivered": self.delivered,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read": self.read,
            "read_at": self.read_at.isoformat() if self.read_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        data = data.copy()
        if "message_type" in data:
            data["message_type"] = MessageType(data["message_type"])
        for dt_field in ["created_at", "expires_at", "delivered_at", "read_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def create_response(
        self,
        sender_id: str,
        content: Any,
        subject: Optional[str] = None,
    ) -> "Message":
        """Create a response to this message."""
        return Message(
            sender_id=sender_id,
            recipient_id=self.sender_id,
            team_id=self.team_id,
            message_type=MessageType.RESPONSE,
            subject=subject or f"Re: {self.subject}",
            content=content,
            correlation_id=self.correlation_id or self.id,
        )


@dataclass
class TeamTask:
    """A task assigned to a team."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Definition
    title: str = ""
    description: str = ""
    objective: str = ""
    success_criteria: list[str] = field(default_factory=list)

    # Assignment
    assigned_team_id: Optional[str] = None
    assigned_agent_ids: list[str] = field(default_factory=list)

    # Workflow
    workflow_pattern: WorkflowPattern = WorkflowPattern.SEQUENTIAL
    subtasks: list["TeamTask"] = field(default_factory=list)
    parent_task_id: Optional[str] = None

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs

    # Constraints
    deadline: Optional[datetime] = None
    max_iterations: int = 10
    priority: TaskPriority = TaskPriority.NORMAL

    # Status
    status: str = "pending"  # pending, active, completed, failed, cancelled
    progress: float = 0.0
    current_iteration: int = 0

    # Results
    result: Any = None
    artifacts: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Execution context
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "objective": self.objective,
            "success_criteria": self.success_criteria.copy(),
            "assigned_team_id": self.assigned_team_id,
            "assigned_agent_ids": self.assigned_agent_ids.copy(),
            "workflow_pattern": self.workflow_pattern.value,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "parent_task_id": self.parent_task_id,
            "depends_on": self.depends_on.copy(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "max_iterations": self.max_iterations,
            "priority": self.priority.value,
            "status": self.status,
            "progress": self.progress,
            "current_iteration": self.current_iteration,
            "result": self.result,
            "artifacts": self.artifacts.copy(),
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "context": self.context.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamTask":
        data = data.copy()
        if "workflow_pattern" in data:
            data["workflow_pattern"] = WorkflowPattern(data["workflow_pattern"])
        if "priority" in data:
            data["priority"] = TaskPriority(data["priority"])
        if "subtasks" in data:
            data["subtasks"] = [TeamTask.from_dict(st) for st in data["subtasks"]]
        for dt_field in ["deadline", "created_at", "started_at", "completed_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in ("completed", "failed", "cancelled")

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = "active"
        self.started_at = datetime.now()

    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        self.progress = 1.0
        if result is not None:
            self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.error = error


@dataclass
class Team:
    """A team of agents working together."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Definition
    name: str = ""
    purpose: str = ""

    # Composition
    agent_ids: list[str] = field(default_factory=list)
    leader_id: Optional[str] = None

    # Workflow
    workflow_pattern: WorkflowPattern = WorkflowPattern.SEQUENTIAL

    # Current work
    task_id: Optional[str] = None
    task_queue: list[str] = field(default_factory=list)  # Pending task IDs

    # Status
    status: TeamStatus = TeamStatus.FORMING

    # Communication
    shared_context: dict[str, Any] = field(default_factory=dict)
    message_history: list[Message] = field(default_factory=list)

    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    disbanded_at: Optional[datetime] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "agent_ids": self.agent_ids.copy(),
            "leader_id": self.leader_id,
            "workflow_pattern": self.workflow_pattern.value,
            "task_id": self.task_id,
            "task_queue": self.task_queue.copy(),
            "status": self.status.value,
            "shared_context": self.shared_context.copy(),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "created_at": self.created_at.isoformat(),
            "disbanded_at": self.disbanded_at.isoformat() if self.disbanded_at else None,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        data = data.copy()
        if "workflow_pattern" in data:
            data["workflow_pattern"] = WorkflowPattern(data["workflow_pattern"])
        if "status" in data:
            data["status"] = TeamStatus(data["status"])
        for dt_field in ["created_at", "disbanded_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        # Remove message_history from dict loading to avoid complexity
        data.pop("message_history", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def add_agent(self, agent_id: str) -> None:
        """Add an agent to the team."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the team."""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
        if self.leader_id == agent_id:
            self.leader_id = self.agent_ids[0] if self.agent_ids else None


@dataclass
class ConsensusVote:
    """A vote in a consensus decision."""
    agent_id: str
    option: str
    confidence: float = 1.0
    reasoning: str = ""
    rank: Optional[list[str]] = None  # For ranked voting
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "option": self.option,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "rank": self.rank.copy() if self.rank else None,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsensusVote":
        data = data.copy()
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConsensusResult:
    """Result of a consensus decision."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    options: list[str] = field(default_factory=list)
    votes: list[ConsensusVote] = field(default_factory=list)

    # Result
    winning_option: Optional[str] = None
    confidence: float = 0.0
    unanimous: bool = False

    # Method used
    method: ConsensusMethod = ConsensusMethod.MAJORITY

    # Details
    vote_counts: dict[str, int] = field(default_factory=dict)
    weighted_scores: dict[str, float] = field(default_factory=dict)

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options.copy(),
            "votes": [v.to_dict() for v in self.votes],
            "winning_option": self.winning_option,
            "confidence": self.confidence,
            "unanimous": self.unanimous,
            "method": self.method.value,
            "vote_counts": self.vote_counts.copy(),
            "weighted_scores": self.weighted_scores.copy(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsensusResult":
        data = data.copy()
        if "method" in data:
            data["method"] = ConsensusMethod(data["method"])
        if "votes" in data:
            data["votes"] = [ConsensusVote.from_dict(v) for v in data["votes"]]
        for dt_field in ["started_at", "completed_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkflowStep:
    """A step in a workflow execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    action: str = ""
    input_data: Any = None
    output_data: Any = None
    status: str = "pending"  # pending, active, completed, failed, skipped
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Token usage
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "action": self.action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
        }


@dataclass
class WorkflowExecution:
    """Record of a workflow execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_pattern: WorkflowPattern = WorkflowPattern.SEQUENTIAL
    team_id: str = ""
    task_id: str = ""

    # Steps
    steps: list[WorkflowStep] = field(default_factory=list)
    current_step_index: int = 0

    # Status
    status: str = "pending"  # pending, active, completed, failed

    # Results
    final_output: Any = None
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    # Resource usage
    total_tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_pattern": self.workflow_pattern.value,
            "team_id": self.team_id,
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status,
            "final_output": self.final_output,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "total_tokens_used": self.total_tokens_used,
        }


@dataclass
class OrchestratorStats:
    """Statistics for the multi-agent orchestrator."""
    # Agent statistics
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    agents_by_role: dict[str, int] = field(default_factory=dict)
    agents_by_status: dict[str, int] = field(default_factory=dict)

    # Team statistics
    total_teams: int = 0
    active_teams: int = 0
    teams_by_status: dict[str, int] = field(default_factory=dict)

    # Task statistics
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_pending: int = 0

    # Message statistics
    messages_sent: int = 0
    messages_delivered: int = 0
    broadcasts: int = 0

    # Resource usage
    total_tokens_used: int = 0
    total_runtime_seconds: float = 0.0

    # Timing
    uptime_seconds: float = 0.0
    started_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "active_agents": self.active_agents,
            "idle_agents": self.idle_agents,
            "agents_by_role": self.agents_by_role.copy(),
            "agents_by_status": self.agents_by_status.copy(),
            "total_teams": self.total_teams,
            "active_teams": self.active_teams,
            "teams_by_status": self.teams_by_status.copy(),
            "total_tasks": self.total_tasks,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_pending": self.tasks_pending,
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "broadcasts": self.broadcasts,
            "total_tokens_used": self.total_tokens_used,
            "total_runtime_seconds": self.total_runtime_seconds,
            "uptime_seconds": self.uptime_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }
