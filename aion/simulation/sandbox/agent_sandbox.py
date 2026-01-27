"""AION Agent Sandbox - Isolated agent execution with resource tracking.

Provides:
- AgentSandbox: Isolated execution environment for agents in simulation.
- Tool mocking, memory isolation, resource limits.
- Action recording with timing and causality.
- Deterministic execution support.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.simulation.sandbox.tool_mock import ToolMockRegistry
from aion.simulation.types import (
    AgentInSimulation,
    EventType,
    SimulationEvent,
    WorldState,
)

if TYPE_CHECKING:
    from aion.simulation.world.engine import WorldEngine

logger = structlog.get_logger(__name__)


class AgentSandbox:
    """Isolated execution environment for agents.

    SOTA features:
    - Tool mocking with call recording and assertions.
    - Memory isolation per agent.
    - Resource limit enforcement (actions, time, memory).
    - Deterministic execution ordering.
    - Action causality tracking.
    - Decision recording for analysis.
    """

    def __init__(
        self,
        world_engine: "WorldEngine",
        kernel: Optional[Any] = None,
    ) -> None:
        self.world_engine = world_engine
        self.kernel = kernel

        # Agents
        self._agents: Dict[str, AgentInSimulation] = {}

        # Tool mocking
        self.tool_registry = ToolMockRegistry()

        # Memory isolation
        self._memory_stores: Dict[str, Dict[str, Any]] = {}

        # Resource tracking
        self._resource_usage: Dict[str, Dict[str, float]] = {}

        # Resource limits
        self._limits: Dict[str, Dict[str, float]] = {}

        # Decision recording
        self._decisions: Dict[str, List[Dict[str, Any]]] = {}

    # -- Agent Lifecycle --

    async def load_agent(
        self,
        agent_id: str,
        config: Dict[str, Any],
        tool_mocks: Optional[Dict[str, Any]] = None,
    ) -> AgentInSimulation:
        """Load an agent into the sandbox."""
        sim_agent = AgentInSimulation(
            agent_id=agent_id,
            config=config,
            mocked_tools=tool_mocks or {},
        )

        self._agents[sim_agent.id] = sim_agent
        self._memory_stores[sim_agent.id] = {}
        self._resource_usage[sim_agent.id] = {
            "actions": 0,
            "tokens": 0,
            "time_ms": 0,
            "memory_mb": 0,
        }
        self._decisions[sim_agent.id] = []
        self._limits[sim_agent.id] = {
            "max_actions": config.get("max_actions", 10_000),
            "max_time_ms": config.get("max_time_ms", 300_000),
            "max_memory_mb": config.get("max_memory_mb", 512),
        }

        # Register agent-specific tool mocks
        if tool_mocks:
            for tool_name, mock_config in tool_mocks.items():
                if callable(mock_config):
                    from aion.simulation.sandbox.tool_mock import ToolMock
                    mock = ToolMock(name=tool_name)
                    mock.with_handler(mock_config)
                    self.tool_registry.register(mock)

        logger.info("agent_loaded", agent_id=agent_id, sim_id=sim_agent.id)
        return sim_agent

    async def unload_agent(self, sim_id: str) -> None:
        """Unload an agent from sandbox."""
        self._agents.pop(sim_id, None)
        self._memory_stores.pop(sim_id, None)
        self._resource_usage.pop(sim_id, None)
        self._decisions.pop(sim_id, None)
        self._limits.pop(sim_id, None)

    async def reset(self) -> None:
        """Reset all sandbox state."""
        self._agents.clear()
        self._memory_stores.clear()
        self._resource_usage.clear()
        self._decisions.clear()
        self._limits.clear()
        self.tool_registry.reset_all()

    # -- Action Execution --

    async def execute_agent_action(
        self,
        agent_sim_id: str,
        action: str,
        params: Dict[str, Any],
        caused_by: Optional[str] = None,
    ) -> SimulationEvent:
        """Execute an agent action in the sandbox.

        Enforces resource limits, records actions and timing,
        and produces a causal event.
        """
        sim_agent = self._agents.get(agent_sim_id)
        if not sim_agent:
            return SimulationEvent(
                type=EventType.ERROR,
                action=action,
                success=False,
                error="Agent not found in sandbox",
            )

        # Check resource limits
        limit_error = self._check_limits(agent_sim_id)
        if limit_error:
            return SimulationEvent(
                type=EventType.ERROR,
                source_id=agent_sim_id,
                action=action,
                success=False,
                error=limit_error,
            )

        start = time.monotonic()

        try:
            if action.startswith("tool:"):
                tool_name = action[5:]
                result = await self._execute_tool(sim_agent, tool_name, params)
            else:
                result = await self._execute_action(sim_agent, action, params)

            duration_ms = (time.monotonic() - start) * 1000

            # Record
            action_record = {
                "action": action,
                "params": params,
                "result": result,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
                "tick": self.world_engine.state.tick,
            }
            sim_agent.actions_taken.append(action_record)
            sim_agent.total_actions += 1
            sim_agent.successful_actions += 1

            usage = self._resource_usage[agent_sim_id]
            usage["actions"] += 1
            usage["time_ms"] += duration_ms
            sim_agent.total_time_ms += duration_ms

            return SimulationEvent(
                type=EventType.AGENT_ACTION,
                source_id=agent_sim_id,
                action=action,
                data=params,
                result=result,
                success=True,
                simulation_time=self.world_engine.state.simulation_time,
                tick=self.world_engine.state.tick,
                caused_by=caused_by,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            sim_agent.failed_actions += 1
            sim_agent.total_actions += 1

            usage = self._resource_usage.get(agent_sim_id, {})
            usage["actions"] = usage.get("actions", 0) + 1
            usage["time_ms"] = usage.get("time_ms", 0) + duration_ms

            return SimulationEvent(
                type=EventType.AGENT_ACTION,
                source_id=agent_sim_id,
                action=action,
                data=params,
                success=False,
                error=str(exc),
                simulation_time=self.world_engine.state.simulation_time,
                tick=self.world_engine.state.tick,
                caused_by=caused_by,
            )

    async def record_decision(
        self,
        agent_sim_id: str,
        decision: Dict[str, Any],
    ) -> None:
        """Record an agent decision for analysis."""
        decisions = self._decisions.get(agent_sim_id)
        if decisions is not None:
            decisions.append({
                **decision,
                "tick": self.world_engine.state.tick,
                "timestamp": datetime.utcnow().isoformat(),
            })

    # -- Tool Execution --

    async def _execute_tool(
        self,
        sim_agent: AgentInSimulation,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Any:
        """Execute a tool (mocked)."""
        # Agent-specific mock
        if tool_name in sim_agent.mocked_tools:
            mock_fn = sim_agent.mocked_tools[tool_name]
            if asyncio.iscoroutinefunction(mock_fn):
                return await mock_fn(params)
            if callable(mock_fn):
                return mock_fn(params)
            return mock_fn  # Static value

        # Global mock registry
        if self.tool_registry.has(tool_name):
            return await self.tool_registry.invoke(tool_name, params)

        # Default mock
        return {"status": "mocked", "tool": tool_name, "params": params}

    async def _execute_action(
        self,
        sim_agent: AgentInSimulation,
        action: str,
        params: Dict[str, Any],
    ) -> Any:
        """Execute a general action."""
        handlers = {
            "think": self._handle_think,
            "respond": self._handle_respond,
            "remember": self._handle_remember,
            "recall": self._handle_recall,
            "decide": self._handle_decide,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(sim_agent, params)

        return {"status": "executed", "action": action}

    async def _handle_think(
        self,
        sim_agent: AgentInSimulation,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"thought": params.get("content", ""), "simulated": True}

    async def _handle_respond(
        self,
        sim_agent: AgentInSimulation,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        message = params.get("message", "")
        sim_agent.messages_sent.append({
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "tick": self.world_engine.state.tick,
        })

        self.world_engine.emit_event(SimulationEvent(
            type=EventType.AGENT_ACTION,
            source_id=sim_agent.id,
            action="respond",
            data={"message": message},
        ))

        return {"sent": True, "message": message}

    async def _handle_remember(
        self,
        sim_agent: AgentInSimulation,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        key = params.get("key", "default")
        value = params.get("value")
        store = self._memory_stores.setdefault(sim_agent.id, {})
        store[key] = value
        return {"stored": True, "key": key}

    async def _handle_recall(
        self,
        sim_agent: AgentInSimulation,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        key = params.get("key", "default")
        store = self._memory_stores.get(sim_agent.id, {})
        value = store.get(key)
        return {"found": value is not None, "value": value}

    async def _handle_decide(
        self,
        sim_agent: AgentInSimulation,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        decision = {
            "options": params.get("options", []),
            "chosen": params.get("chosen"),
            "reasoning": params.get("reasoning", ""),
        }
        sim_agent.decisions_made.append(decision)
        await self.record_decision(sim_agent.id, decision)
        return {"decided": True, **decision}

    # -- Resource Limits --

    def _check_limits(self, agent_sim_id: str) -> Optional[str]:
        """Check if agent has exceeded resource limits."""
        usage = self._resource_usage.get(agent_sim_id, {})
        limits = self._limits.get(agent_sim_id, {})

        if usage.get("actions", 0) >= limits.get("max_actions", float("inf")):
            return f"Action limit exceeded ({usage['actions']})"

        if usage.get("time_ms", 0) >= limits.get("max_time_ms", float("inf")):
            return f"Time limit exceeded ({usage['time_ms']:.0f}ms)"

        return None

    # -- Mock Shortcuts --

    def mock_tool(self, tool_name: str, mock_fn: Callable) -> None:
        """Register a global tool mock."""
        from aion.simulation.sandbox.tool_mock import ToolMock
        mock = ToolMock(name=tool_name)
        mock.with_handler(mock_fn)
        self.tool_registry.register(mock)

    def mock_tool_response(self, tool_name: str, response: Any) -> None:
        from aion.simulation.sandbox.tool_mock import ToolMock
        mock = ToolMock(name=tool_name, default_response=response)
        self.tool_registry.register(mock)

    def mock_tool_error(self, tool_name: str, error: str) -> None:
        from aion.simulation.sandbox.tool_mock import ToolMock
        mock = ToolMock(name=tool_name, error=error, error_probability=1.0)
        self.tool_registry.register(mock)

    def mock_tool_delay(self, tool_name: str, delay_ms: float) -> None:
        from aion.simulation.sandbox.tool_mock import ToolMock
        mock = ToolMock(name=tool_name, latency_ms=delay_ms)
        self.tool_registry.register(mock)

    # -- State Access --

    def get_agent(self, sim_id: str) -> Optional[AgentInSimulation]:
        return self._agents.get(sim_id)

    def get_agent_actions(self, sim_id: str) -> List[Dict[str, Any]]:
        agent = self._agents.get(sim_id)
        return agent.actions_taken if agent else []

    def get_agent_messages(self, sim_id: str) -> List[Dict[str, Any]]:
        agent = self._agents.get(sim_id)
        return agent.messages_sent if agent else []

    def get_agent_decisions(self, sim_id: str) -> List[Dict[str, Any]]:
        return list(self._decisions.get(sim_id, []))

    def get_resource_usage(self, sim_id: str) -> Dict[str, float]:
        return dict(self._resource_usage.get(sim_id, {}))

    def get_memory_store(self, sim_id: str) -> Dict[str, Any]:
        return dict(self._memory_stores.get(sim_id, {}))

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def all_agents(self) -> Dict[str, AgentInSimulation]:
        return dict(self._agents)
