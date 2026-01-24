"""
Multi-Agent Coordination Safety

Safety mechanisms for multi-agent coordination.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

logger = structlog.get_logger()


@dataclass
class AgentAction:
    """An action taken by an agent."""

    agent_id: str
    action: str
    target: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conflict:
    """A detected conflict between agents."""

    id: str
    agent_ids: list[str]
    conflict_type: str
    description: str
    severity: str
    actions_involved: list[AgentAction] = field(default_factory=list)
    resolved: bool = False
    resolution: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_ids": self.agent_ids,
            "conflict_type": self.conflict_type,
            "description": self.description,
            "severity": self.severity,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


class ConflictDetector:
    """Detects conflicts between agent actions."""

    def __init__(self):
        self._actions: list[AgentAction] = []
        self._conflicts: list[Conflict] = []
        self._conflict_counter = 0

    def record_action(self, action: AgentAction) -> None:
        """Record an agent action."""
        self._actions.append(action)

        # Check for conflicts with recent actions
        conflicts = self._check_conflicts(action)
        self._conflicts.extend(conflicts)

    def _check_conflicts(self, new_action: AgentAction) -> list[Conflict]:
        """Check for conflicts with recent actions."""
        conflicts = []
        recent = self._actions[-20:-1]  # Exclude new action

        for other in recent:
            if other.agent_id == new_action.agent_id:
                continue

            # Check for same target
            if new_action.target and other.target == new_action.target:
                self._conflict_counter += 1
                conflicts.append(Conflict(
                    id=f"conflict-{self._conflict_counter}",
                    agent_ids=[new_action.agent_id, other.agent_id],
                    conflict_type="resource_contention",
                    description=f"Both agents targeting: {new_action.target}",
                    severity="medium",
                    actions_involved=[other, new_action],
                ))

            # Check for contradictory actions
            if self._are_contradictory(new_action.action, other.action):
                self._conflict_counter += 1
                conflicts.append(Conflict(
                    id=f"conflict-{self._conflict_counter}",
                    agent_ids=[new_action.agent_id, other.agent_id],
                    conflict_type="contradictory_actions",
                    description=f"Contradictory: {new_action.action} vs {other.action}",
                    severity="high",
                    actions_involved=[other, new_action],
                ))

        return conflicts

    def _are_contradictory(self, action1: str, action2: str) -> bool:
        """Check if two actions are contradictory."""
        contradictions = [
            ("create", "delete"),
            ("enable", "disable"),
            ("start", "stop"),
            ("add", "remove"),
            ("approve", "reject"),
        ]

        a1_lower = action1.lower()
        a2_lower = action2.lower()

        for c1, c2 in contradictions:
            if (c1 in a1_lower and c2 in a2_lower) or (c2 in a1_lower and c1 in a2_lower):
                return True

        return False

    def get_unresolved_conflicts(self) -> list[Conflict]:
        """Get unresolved conflicts."""
        return [c for c in self._conflicts if not c.resolved]

    def resolve_conflict(self, conflict_id: str, resolution: str) -> bool:
        """Resolve a conflict."""
        for conflict in self._conflicts:
            if conflict.id == conflict_id:
                conflict.resolved = True
                conflict.resolution = resolution
                return True
        return False


class SafeCoordinator:
    """Safe multi-agent coordinator."""

    def __init__(self):
        self.conflict_detector = ConflictDetector()
        self._blocked_agents: set[str] = set()
        self._action_limits: dict[str, int] = {}

    def request_action(
        self,
        agent_id: str,
        action: str,
        target: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Request permission for an action.

        Returns:
            Tuple of (approved, reason if denied)
        """
        # Check if agent is blocked
        if agent_id in self._blocked_agents:
            return False, "Agent is blocked"

        # Check rate limit
        limit = self._action_limits.get(agent_id, 100)
        recent_actions = sum(
            1 for a in self.conflict_detector._actions[-100:]
            if a.agent_id == agent_id
        )
        if recent_actions >= limit:
            return False, "Rate limit exceeded"

        # Record action
        action_obj = AgentAction(
            agent_id=agent_id,
            action=action,
            target=target,
            parameters=parameters or {},
        )
        self.conflict_detector.record_action(action_obj)

        # Check for new conflicts
        unresolved = self.conflict_detector.get_unresolved_conflicts()
        agent_conflicts = [c for c in unresolved if agent_id in c.agent_ids]

        if agent_conflicts and agent_conflicts[-1].severity == "high":
            return False, f"Conflict detected: {agent_conflicts[-1].description}"

        return True, None

    def block_agent(self, agent_id: str, reason: str) -> None:
        """Block an agent from taking actions."""
        self._blocked_agents.add(agent_id)
        logger.warning("agent_blocked", agent_id=agent_id, reason=reason)

    def unblock_agent(self, agent_id: str) -> None:
        """Unblock an agent."""
        self._blocked_agents.discard(agent_id)

    def set_rate_limit(self, agent_id: str, limit: int) -> None:
        """Set rate limit for an agent."""
        self._action_limits[agent_id] = limit

    def get_stats(self) -> dict[str, Any]:
        """Get coordination statistics."""
        return {
            "total_actions": len(self.conflict_detector._actions),
            "total_conflicts": len(self.conflict_detector._conflicts),
            "unresolved_conflicts": len(self.conflict_detector.get_unresolved_conflicts()),
            "blocked_agents": list(self._blocked_agents),
        }


class CoordinationSafety:
    """Main coordination safety class."""

    def __init__(self):
        self.coordinator = SafeCoordinator()

    async def check_action_safe(
        self,
        agent_id: str,
        action: str,
        context: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Check if action is safe from coordination perspective."""
        return self.coordinator.request_action(
            agent_id=agent_id,
            action=action,
            target=context.get("target"),
            parameters=context,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get safety statistics."""
        return self.coordinator.get_stats()
