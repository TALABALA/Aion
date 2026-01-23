"""
AION Autonomous Goal Manager

Central coordinator for the autonomous goal system.

SOTA Features:
- Learned components with neural goal evaluation
- Uncertainty quantification with Bayesian reasoning
- World model for outcome simulation
- Meta-learning for adaptive goal strategies
- Formal verification for safety guarantees
- Multi-agent coordination system
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
    GoalSource,
    GoalProposal,
    GoalEvent,
    Objective,
)
from aion.systems.goals.reasoner import GoalReasoner
from aion.systems.goals.registry import GoalRegistry
from aion.systems.goals.scheduler import GoalScheduler
from aion.systems.goals.executor import GoalExecutor
from aion.systems.goals.monitor import GoalMonitor
from aion.systems.goals.decomposer import GoalDecomposer
from aion.systems.goals.prioritizer import GoalPrioritizer
from aion.systems.goals.safety import SafetyBoundary
from aion.systems.goals.values import ValueSystem
from aion.systems.goals.triggers import GoalTriggers
from aion.systems.goals.persistence import GoalPersistence

# SOTA Components
from aion.systems.goals.learning import GoalLearningSystem
from aion.systems.goals.uncertainty import UncertaintyQuantifier
from aion.systems.goals.world_model import WorldModel, Action, WorldState
from aion.systems.goals.meta_learning import MetaLearningSystem
from aion.systems.goals.formal_verification import FormalVerificationSystem
from aion.systems.goals.multi_agent import MultiAgentCoordinator, AgentRole

logger = structlog.get_logger(__name__)


class AutonomousGoalManager:
    """
    Central manager for AION's autonomous goal system.

    Coordinates:
    - Goal generation and evaluation
    - Goal scheduling and execution
    - Safety boundary enforcement
    - Progress monitoring
    - Value alignment

    SOTA Capabilities:
    - Learned components with neural goal evaluation
    - Uncertainty quantification with Bayesian reasoning
    - World model for outcome simulation
    - Meta-learning for adaptive goal strategies
    - Formal verification for safety guarantees
    - Multi-agent coordination system
    """

    def __init__(
        self,
        # Core components
        reasoner: Optional[GoalReasoner] = None,
        registry: Optional[GoalRegistry] = None,
        scheduler: Optional[GoalScheduler] = None,
        executor: Optional[GoalExecutor] = None,
        monitor: Optional[GoalMonitor] = None,
        # Support components
        decomposer: Optional[GoalDecomposer] = None,
        prioritizer: Optional[GoalPrioritizer] = None,
        safety: Optional[SafetyBoundary] = None,
        values: Optional[ValueSystem] = None,
        triggers: Optional[GoalTriggers] = None,
        persistence: Optional[GoalPersistence] = None,
        # SOTA components
        learning_system: Optional[GoalLearningSystem] = None,
        uncertainty: Optional[UncertaintyQuantifier] = None,
        world_model: Optional[WorldModel] = None,
        meta_learning: Optional[MetaLearningSystem] = None,
        formal_verification: Optional[FormalVerificationSystem] = None,
        multi_agent: Optional[MultiAgentCoordinator] = None,
        # Configuration
        auto_generate_goals: bool = True,
        goal_generation_interval: float = 300.0,  # 5 minutes
        max_active_goals: int = 5,
        data_dir: str = "data/goals",
        enable_sota_features: bool = True,
    ):
        # Initialize persistence first
        self._persistence = persistence or GoalPersistence(data_dir=data_dir)

        # Core components
        self._registry = registry or GoalRegistry(persistence=self._persistence)
        self._reasoner = reasoner or GoalReasoner()
        self._executor = executor or GoalExecutor()
        self._scheduler = scheduler or GoalScheduler(
            self._registry, self._executor, max_concurrent_goals=max_active_goals
        )
        self._monitor = monitor or GoalMonitor(self._registry, self._reasoner)

        # Support components
        self._decomposer = decomposer or GoalDecomposer(self._reasoner)
        self._prioritizer = prioritizer or GoalPrioritizer()
        self._safety = safety or SafetyBoundary()
        self._values = values or ValueSystem()
        self._triggers = triggers or GoalTriggers(self._registry)

        # SOTA components
        self._enable_sota = enable_sota_features
        if enable_sota_features:
            self._learning = learning_system or GoalLearningSystem(data_dir=f"{data_dir}/learning")
            self._uncertainty = uncertainty or UncertaintyQuantifier()
            self._world_model = world_model or WorldModel()
            self._meta_learning = meta_learning or MetaLearningSystem(data_dir=f"{data_dir}/meta")
            self._formal_verification = formal_verification or FormalVerificationSystem()
            self._multi_agent = multi_agent or MultiAgentCoordinator()
        else:
            self._learning = None
            self._uncertainty = None
            self._world_model = None
            self._meta_learning = None
            self._formal_verification = None
            self._multi_agent = None

        # Configuration
        self._auto_generate_goals = auto_generate_goals
        self._goal_generation_interval = goal_generation_interval
        self._max_active_goals = max_active_goals

        # Background tasks
        self._generation_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Callbacks
        self._goal_callbacks: list[Callable[[str, Goal], None]] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the goal manager."""
        if self._initialized:
            return

        logger.info("Initializing Autonomous Goal Manager")

        # Initialize components in order
        await self._persistence.initialize()
        await self._registry.initialize()
        await self._reasoner.initialize()
        await self._executor.initialize()
        await self._values.initialize()
        await self._decomposer.initialize()
        await self._prioritizer.initialize()
        await self._scheduler.initialize()
        await self._monitor.initialize()
        await self._triggers.initialize()

        # Initialize SOTA components
        if self._enable_sota:
            logger.info("Initializing SOTA components")
            await self._learning.initialize()
            await self._uncertainty.initialize()
            await self._world_model.initialize()
            await self._meta_learning.initialize()
            await self._formal_verification.initialize()
            await self._multi_agent.initialize()
            logger.info("SOTA components initialized")

        # Wire up event handlers
        self._setup_event_handlers()

        # Start auto goal generation if enabled
        if self._auto_generate_goals:
            self._generation_task = asyncio.create_task(self._goal_generation_loop())

        self._initialized = True
        logger.info("Autonomous Goal Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the goal manager."""
        logger.info("Shutting down Autonomous Goal Manager")

        self._shutdown_event.set()

        if self._generation_task:
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass

        # Shutdown SOTA components first
        if self._enable_sota:
            logger.info("Shutting down SOTA components")
            await self._multi_agent.shutdown()
            await self._formal_verification.shutdown()
            await self._meta_learning.shutdown()
            await self._world_model.shutdown()
            await self._uncertainty.shutdown()
            await self._learning.shutdown()

        # Shutdown core components in reverse order
        await self._triggers.shutdown()
        await self._monitor.shutdown()
        await self._scheduler.shutdown()
        await self._prioritizer.shutdown()
        await self._decomposer.shutdown()
        await self._values.shutdown()
        await self._executor.shutdown()
        await self._reasoner.shutdown()
        await self._registry.shutdown()
        await self._persistence.shutdown()

        self._initialized = False
        logger.info("Autonomous Goal Manager shutdown complete")

    def _setup_event_handlers(self) -> None:
        """Set up internal event handlers."""
        # Subscribe to goal events
        self._registry.subscribe_to_events(self._on_goal_event)

        # Subscribe to scheduler completion
        self._scheduler.on_completion(self._on_goal_completed)

        # Subscribe to monitor alerts
        self._monitor.on_alert(self._on_monitor_alert)

        # Subscribe to safety approvals
        self._safety.on_approval_request(self._on_approval_request)

    def _on_goal_event(self, event: GoalEvent) -> None:
        """Handle goal events."""
        # Notify external callbacks
        for callback in self._goal_callbacks:
            try:
                goal = self._registry._goals.get(event.goal_id)
                if goal:
                    callback(event.event_type, goal)
            except Exception as e:
                logger.error(f"Goal callback error: {e}")

    def _on_goal_completed(self, goal: Goal, result: dict) -> None:
        """Handle goal completion."""
        logger.info(
            f"Goal completed: {goal.title}",
            goal_id=goal.id[:8],
            success=result.get("success"),
        )

        # Check for triggered follow-up goals
        asyncio.create_task(self._triggers.handle_event(
            "goal_completed",
            {"goal_id": goal.id, "success": result.get("success")},
        ))

    def _on_monitor_alert(self, alert_type: str, goal: Goal, data: dict) -> None:
        """Handle monitor alerts."""
        logger.warning(
            f"Goal alert: {alert_type}",
            goal_id=goal.id[:8],
            data=data,
        )

    def _on_approval_request(self, request: Any) -> None:
        """Handle safety approval requests."""
        logger.info(
            f"Approval requested",
            request_id=request.id[:8],
            goal_id=request.goal_id[:8],
        )

    async def _goal_generation_loop(self) -> None:
        """Background loop for autonomous goal generation."""
        # Initial delay
        await asyncio.sleep(60)

        while not self._shutdown_event.is_set():
            try:
                # Check if we have capacity for new goals
                active_count = len(self._scheduler.get_active_goals())
                pending_count = len(await self._registry.get_by_status(GoalStatus.PENDING))

                if active_count + pending_count < self._max_active_goals:
                    await self.generate_goals()

                await asyncio.sleep(self._goal_generation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Goal generation error: {e}")
                await asyncio.sleep(60)

    async def generate_goals(self, max_goals: int = 2) -> list[Goal]:
        """Generate new goals based on current context."""
        # Get current context
        context = await self._build_context()

        # Get active objectives
        objectives = await self._registry.get_active_objectives()

        # Get existing active goals
        existing_goals = await self._registry.get_active()

        # Generate proposals
        proposals = await self._reasoner.generate_goals(
            context=context,
            objectives=objectives,
            existing_goals=existing_goals,
            max_goals=max_goals,
        )

        # Evaluate and approve proposals
        approved_goals = []

        for proposal in proposals:
            # Safety check
            is_safe, concerns = await self._safety.check_goal_safety(proposal.goal)

            if not is_safe:
                logger.warning(
                    f"Goal rejected by safety: {proposal.goal.title}",
                    concerns=concerns,
                )
                continue

            # Value alignment check
            alignment = await self._values.check_alignment(proposal.goal)
            if alignment["overall_alignment"] < 0.4:
                logger.warning(
                    f"Goal rejected by value alignment: {proposal.goal.title}",
                    alignment=alignment["overall_alignment"],
                )
                continue

            # Evaluate goal
            evaluation = await self._reasoner.evaluate_goal(proposal.goal, context)

            if evaluation.get("should_pursue", True):
                # Approve goal
                proposal.goal.status = GoalStatus.PENDING
                await self._registry.create(proposal.goal)
                approved_goals.append(proposal.goal)

                logger.info(
                    f"Generated goal: {proposal.goal.title}",
                    goal_id=proposal.goal.id[:8],
                )

        return approved_goals

    async def submit_goal(
        self,
        title: str,
        description: str,
        success_criteria: list[str],
        priority: str = "medium",
        goal_type: str = "achievement",
        deadline: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        auto_decompose: bool = False,
    ) -> Goal:
        """
        Submit a user-requested goal.

        Args:
            title: Goal title
            description: Goal description
            success_criteria: List of success criteria
            priority: Priority level
            goal_type: Type of goal
            deadline: Optional deadline
            tags: Optional tags
            auto_decompose: Whether to automatically decompose

        Returns:
            Created goal
        """
        priority_map = {
            "critical": GoalPriority.CRITICAL,
            "high": GoalPriority.HIGH,
            "medium": GoalPriority.MEDIUM,
            "low": GoalPriority.LOW,
            "background": GoalPriority.BACKGROUND,
        }

        type_map = {
            "achievement": GoalType.ACHIEVEMENT,
            "maintenance": GoalType.MAINTENANCE,
            "improvement": GoalType.IMPROVEMENT,
            "learning": GoalType.LEARNING,
            "creation": GoalType.CREATION,
            "exploration": GoalType.EXPLORATION,
            "optimization": GoalType.OPTIMIZATION,
        }

        goal = Goal(
            title=title,
            description=description,
            success_criteria=success_criteria,
            source=GoalSource.USER,
            priority=priority_map.get(priority.lower(), GoalPriority.MEDIUM),
            goal_type=type_map.get(goal_type.lower(), GoalType.ACHIEVEMENT),
            status=GoalStatus.PENDING,
            deadline=deadline,
            tags=tags or [],
        )

        # Safety check
        is_safe, concerns = await self._safety.check_goal_safety(goal)

        if not is_safe:
            raise ValueError(f"Goal rejected by safety: {'; '.join(concerns)}")

        await self._registry.create(goal)

        # Auto-decompose if requested
        if auto_decompose:
            should_decompose, reason = await self._decomposer.should_decompose(goal)
            if should_decompose:
                await self.decompose_goal(goal.id)

        logger.info(
            f"User submitted goal: {goal.title}",
            goal_id=goal.id[:8],
        )

        return goal

    async def decompose_goal(self, goal_id: str) -> list[Goal]:
        """Decompose a goal into subgoals."""
        goal = await self._registry.get(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        subgoals = await self._decomposer.decompose(goal)

        # Register subgoals
        for subgoal in subgoals:
            await self._registry.create(subgoal)
            goal.subgoal_ids.append(subgoal.id)

        # Update parent goal
        goal.metrics.subgoals_total = len(subgoals)
        await self._registry.update(goal)

        logger.info(
            f"Decomposed goal into {len(subgoals)} subgoals",
            goal_id=goal_id[:8],
        )

        return subgoals

    async def prioritize_goals(self) -> list[Goal]:
        """Reprioritize all pending goals."""
        pending_goals = await self._registry.get_by_status(GoalStatus.PENDING)
        prioritized = await self._prioritizer.prioritize_goals(pending_goals)
        return prioritized

    async def pause_goal(self, goal_id: str) -> bool:
        """Pause a goal."""
        return await self._scheduler.pause_goal(goal_id)

    async def resume_goal(self, goal_id: str) -> bool:
        """Resume a paused goal."""
        return await self._scheduler.resume_goal(goal_id)

    async def abandon_goal(self, goal_id: str, reason: str) -> bool:
        """Abandon a goal."""
        return await self._scheduler.stop_goal(goal_id, reason)

    async def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return await self._registry.get(goal_id)

    async def get_all_goals(self) -> list[Goal]:
        """Get all goals."""
        return await self._registry.get_all()

    async def get_active_goals(self) -> list[Goal]:
        """Get active goals."""
        return await self._registry.get_active()

    async def get_progress(self, goal_id: str) -> dict[str, Any]:
        """Get detailed progress for a goal."""
        return await self._monitor.get_progress_report(goal_id)

    async def get_health(self, goal_id: str) -> dict[str, Any]:
        """Get health status for a goal."""
        return await self._monitor.get_goal_health(goal_id)

    async def get_system_health(self) -> dict[str, Any]:
        """Get overall system health."""
        return await self._monitor.get_system_health()

    # === Objectives ===

    async def create_objective(
        self,
        name: str,
        description: str,
        rationale: str = "",
    ) -> Objective:
        """Create a new objective."""
        objective = Objective(
            name=name,
            description=description,
            rationale=rationale,
        )

        await self._registry.create_objective(objective)

        logger.info(f"Created objective: {objective.name}")

        return objective

    async def get_objectives(self) -> list[Objective]:
        """Get all objectives."""
        return await self._registry.get_all_objectives()

    # === Safety ===

    async def approve_action(self, request_id: str, approver: str) -> bool:
        """Approve a safety request."""
        return await self._safety.approve(request_id, approver)

    async def deny_action(self, request_id: str, approver: str, reason: str = "") -> bool:
        """Deny a safety request."""
        return await self._safety.deny(request_id, approver, reason)

    def get_pending_approvals(self) -> list:
        """Get pending approval requests."""
        return self._safety.get_pending_approvals()

    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Activate emergency stop."""
        self._safety.emergency_stop(reason)
        self._scheduler.pause_scheduler()

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop."""
        self._safety.clear_emergency_stop()
        self._scheduler.resume_scheduler()

    # === Context Building ===

    async def _build_context(self) -> dict[str, Any]:
        """Build context for goal generation."""
        stats = self._registry.get_stats()

        return {
            "current_time": datetime.now().isoformat(),
            "active_goals": stats["by_status"].get("active", 0),
            "pending_goals": stats["by_status"].get("pending", 0),
            "completed_goals": stats["by_status"].get("completed", 0),
            "failed_goals": stats["by_status"].get("failed", 0),
            "total_goals": stats["total_goals"],
            "completion_rate": stats.get("completion_rate", 0.0),
            "active_objectives": len(await self._registry.get_active_objectives()),
            "system_status": "operational",
        }

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive goal system statistics."""
        stats = {
            "registry": self._registry.get_stats(),
            "scheduler": self._scheduler.get_stats(),
            "monitor": self._monitor.get_stats(),
            "safety": self._safety.get_stats(),
            "values": self._values.get_stats(),
            "triggers": self._triggers.get_stats(),
            "persistence": self._persistence.get_stats(),
            "config": {
                "auto_generate": self._auto_generate_goals,
                "generation_interval": self._goal_generation_interval,
                "max_active_goals": self._max_active_goals,
                "sota_enabled": self._enable_sota,
            },
        }

        # Add SOTA stats if enabled
        if self._enable_sota:
            stats["sota"] = self.get_sota_stats()

        return stats

    # === Callbacks ===

    def on_goal_event(self, callback: Callable[[str, Goal], None]) -> None:
        """Register callback for goal events."""
        self._goal_callbacks.append(callback)

    # === Properties ===

    @property
    def registry(self) -> GoalRegistry:
        """Get the goal registry."""
        return self._registry

    @property
    def scheduler(self) -> GoalScheduler:
        """Get the goal scheduler."""
        return self._scheduler

    @property
    def safety(self) -> SafetyBoundary:
        """Get the safety boundary."""
        return self._safety

    @property
    def values(self) -> ValueSystem:
        """Get the value system."""
        return self._values

    @property
    def triggers(self) -> GoalTriggers:
        """Get the trigger system."""
        return self._triggers

    @property
    def monitor(self) -> GoalMonitor:
        """Get the monitor."""
        return self._monitor

    @property
    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._initialized

    # === SOTA Component Properties ===

    @property
    def learning(self) -> Optional[GoalLearningSystem]:
        """Get the learning system."""
        return self._learning

    @property
    def uncertainty(self) -> Optional[UncertaintyQuantifier]:
        """Get the uncertainty quantifier."""
        return self._uncertainty

    @property
    def world_model(self) -> Optional[WorldModel]:
        """Get the world model."""
        return self._world_model

    @property
    def meta_learning(self) -> Optional[MetaLearningSystem]:
        """Get the meta-learning system."""
        return self._meta_learning

    @property
    def formal_verification(self) -> Optional[FormalVerificationSystem]:
        """Get the formal verification system."""
        return self._formal_verification

    @property
    def multi_agent(self) -> Optional[MultiAgentCoordinator]:
        """Get the multi-agent coordinator."""
        return self._multi_agent

    # === SOTA Features ===

    async def predict_goal_success(self, goal: Goal) -> Dict[str, Any]:
        """
        Predict goal success using learned models and uncertainty quantification.

        Returns comprehensive prediction with confidence intervals.
        """
        if not self._enable_sota:
            return {"success_probability": 0.5, "confidence": "low"}

        # Get learned prediction
        learned_prob = self._learning.predict_success(goal)

        # Get uncertainty estimate
        uncertainty_est = self._uncertainty.estimate_success_uncertainty(goal)

        # Get strategy recommendation
        strategy = self._meta_learning.select_strategy(goal)

        return {
            "success_probability": uncertainty_est.mean,
            "confidence_interval": uncertainty_est.confidence_interval,
            "epistemic_uncertainty": uncertainty_est.epistemic_uncertainty,
            "aleatoric_uncertainty": uncertainty_est.aleatoric_uncertainty,
            "confidence_level": self._uncertainty.get_confidence_level(goal),
            "learned_probability": learned_prob,
            "recommended_strategy": strategy.name,
            "should_gather_info": self._uncertainty.should_gather_more_info(goal),
        }

    async def simulate_goal_outcome(
        self,
        goal: Goal,
        actions: List[Action] = None,
    ) -> Dict[str, Any]:
        """
        Simulate goal outcome using the world model.

        Enables "what if" reasoning before committing to actions.
        """
        if not self._enable_sota:
            return {"simulated": False, "reason": "SOTA features disabled"}

        # Get current world state
        current_state = self._world_model.get_current_state()

        if actions:
            # Simulate specific action sequence
            trajectory = self._world_model.simulate_trajectory(actions, current_state)
            final_state, final_reward = trajectory[-1] if trajectory else (current_state, 0)
        else:
            # Plan and simulate best actions
            plan = self._world_model.plan_sequence(goal, horizon=5)
            if plan:
                actions = [action for action, _ in plan]
                trajectory = self._world_model.simulate_trajectory(actions, current_state)
                final_state, final_reward = trajectory[-1]
            else:
                final_state, final_reward = current_state, 0
                trajectory = []

        return {
            "simulated": True,
            "planned_actions": [a.name for a in actions] if actions else [],
            "expected_reward": final_reward,
            "trajectory_length": len(trajectory),
            "final_state_vars": len(final_state.variables) if final_state else 0,
        }

    async def verify_goal_safety_formally(self, goal: Goal) -> Dict[str, Any]:
        """
        Formally verify goal safety using formal methods.

        Provides provable safety guarantees.
        """
        if not self._enable_sota:
            return {"verified": False, "reason": "SOTA features disabled"}

        # Verify using formal verification system
        is_safe, issues = self._formal_verification.verify_goal_safety(goal)

        # Get safety report
        report = self._formal_verification.get_safety_report()

        return {
            "formally_verified": is_safe,
            "safety_issues": issues,
            "violation_count": report["violation_count"],
            "properties_checked": len(report["properties"]),
            "bounds_verified": list(report["bounds"].keys()),
        }

    async def assign_goal_to_agents(
        self,
        goal: Goal,
        required_capabilities: List[str] = None,
        prefer_coalition: bool = False,
    ) -> Dict[str, Any]:
        """
        Assign goal to agents using multi-agent coordination.

        Supports both single-agent and coalition-based assignment.
        """
        if not self._enable_sota:
            return {"assigned": False, "reason": "SOTA features disabled"}

        assignment = await self._multi_agent.assign_goal(
            goal,
            required_capabilities=required_capabilities,
            prefer_coalition=prefer_coalition,
        )

        if assignment:
            return {
                "assigned": True,
                "assignment_id": assignment,
                "is_coalition": assignment.startswith("coalition_"),
                "agent_stats": self._multi_agent.get_agent_stats(),
            }

        return {
            "assigned": False,
            "reason": "No suitable agents available",
            "agent_stats": self._multi_agent.get_agent_stats(),
        }

    async def adapt_to_goal_domain(
        self,
        goal: Goal,
        examples: List[Tuple[Goal, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Adapt the system to a new goal domain using meta-learning.

        Few-shot adaptation for new types of goals.
        """
        if not self._enable_sota:
            return {"adapted": False, "reason": "SOTA features disabled"}

        examples = examples or []

        # Adapt using meta-learning
        adapted_params = self._meta_learning.adapt_to_goal(goal, examples)

        # Get transferred knowledge
        transferred = self._meta_learning.transfer_knowledge(goal)

        # Get learning status
        status = self._meta_learning.get_learning_status()

        return {
            "adapted": True,
            "transferred_knowledge_keys": list(transferred.keys()),
            "current_skill_level": self._meta_learning.curriculum.get_current_skill_level(
                goal.goal_type.value
            ),
            "learning_plateaued": status["success_rate"]["plateaued"],
            "improvement_rate": status["success_rate"]["improvement_rate"],
        }

    async def record_goal_outcome(
        self,
        goal: Goal,
        success: bool,
        completion_time: float,
        quality_score: float = 1.0,
    ) -> None:
        """
        Record goal outcome for learning.

        Updates all learning components with the outcome.
        """
        if not self._enable_sota:
            return

        # Record in learning system
        await self._learning.record_outcome(
            goal=goal,
            success=success,
            completion_time=completion_time,
            quality_score=quality_score,
        )

        # Update uncertainty estimator
        self._uncertainty.update_from_outcome(goal, success)

        # Update meta-learning
        strategy = self._meta_learning.select_strategy(goal)
        reward = 1.0 if success else -0.5
        self._meta_learning.update_from_outcome(
            goal=goal,
            strategy_id=strategy.id,
            success=success,
            completion_time=completion_time,
            reward=reward,
        )

        # Complete in multi-agent if assigned
        assignment = self._multi_agent.get_assignment(goal.id)
        if assignment:
            self._multi_agent.complete_goal(goal.id, success)

        logger.info(
            "Goal outcome recorded for learning",
            goal_id=goal.id[:8],
            success=success,
        )

    def get_sota_stats(self) -> Dict[str, Any]:
        """Get statistics from all SOTA components."""
        if not self._enable_sota:
            return {"enabled": False}

        return {
            "enabled": True,
            "learning": self._learning.get_stats(),
            "uncertainty": self._uncertainty.get_stats(),
            "world_model": self._world_model.get_stats(),
            "meta_learning": self._meta_learning.get_stats(),
            "formal_verification": self._formal_verification.get_stats(),
            "multi_agent": self._multi_agent.get_stats(),
        }
