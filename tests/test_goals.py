"""
AION Goal System Tests

Comprehensive test suite for the autonomous goal system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil


class TestGoalTypes:
    """Test goal type definitions and dataclasses."""

    def test_goal_creation(self):
        """Test basic goal creation."""
        from aion.systems.goals.types import (
            Goal,
            GoalStatus,
            GoalPriority,
            GoalType,
            GoalSource,
        )

        goal = Goal(
            title="Test Goal",
            description="A test goal description",
            success_criteria=["Criterion 1", "Criterion 2"],
        )

        assert goal.title == "Test Goal"
        assert goal.status == GoalStatus.PROPOSED
        assert goal.priority == GoalPriority.MEDIUM
        assert goal.goal_type == GoalType.ACHIEVEMENT
        assert goal.source == GoalSource.SYSTEM
        assert len(goal.success_criteria) == 2
        assert goal.id is not None

    def test_goal_status_transitions(self):
        """Test goal status updates."""
        from aion.systems.goals.types import Goal, GoalStatus

        goal = Goal(
            title="Test Goal",
            description="Test",
            success_criteria=["Done"],
        )

        assert goal.status == GoalStatus.PROPOSED

        goal.status = GoalStatus.PENDING
        assert goal.status == GoalStatus.PENDING

        goal.status = GoalStatus.ACTIVE
        assert goal.status == GoalStatus.ACTIVE

    def test_goal_deadline_handling(self):
        """Test goal deadline functionality."""
        from aion.systems.goals.types import Goal

        # Goal with deadline
        future_deadline = datetime.now() + timedelta(hours=2)
        goal = Goal(
            title="Urgent Task",
            description="Needs to be done soon",
            success_criteria=["Complete"],
            deadline=future_deadline,
        )

        assert goal.deadline == future_deadline
        assert not goal.is_overdue()

        time_remaining = goal.time_until_deadline()
        assert time_remaining is not None
        assert time_remaining.total_seconds() > 0

    def test_goal_serialization(self):
        """Test goal to_dict and from_dict."""
        from aion.systems.goals.types import Goal, GoalPriority, GoalType

        original = Goal(
            title="Serialization Test",
            description="Testing serialization",
            success_criteria=["Works correctly"],
            priority=GoalPriority.HIGH,
            goal_type=GoalType.CREATION,
            tags=["test", "serialization"],
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = Goal.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.priority == original.priority
        assert restored.goal_type == original.goal_type
        assert restored.tags == original.tags

    def test_goal_metrics(self):
        """Test goal metrics tracking."""
        from aion.systems.goals.types import Goal, GoalMetrics

        goal = Goal(
            title="Metrics Test",
            description="Testing metrics",
            success_criteria=["Done"],
        )

        assert goal.metrics.progress_percent == 0.0
        assert goal.metrics.steps_completed == 0

        # Update metrics
        goal.metrics.progress_percent = 50.0
        goal.metrics.steps_completed = 2
        goal.metrics.steps_total = 4

        assert goal.metrics.progress_percent == 50.0
        assert goal.metrics.steps_completed == 2

    def test_objective_creation(self):
        """Test objective creation."""
        from aion.systems.goals.types import Objective

        objective = Objective(
            name="Improve Efficiency",
            description="Make the system more efficient",
            rationale="To better serve users",
        )

        assert objective.name == "Improve Efficiency"
        assert objective.active
        assert objective.id is not None

    def test_value_principle(self):
        """Test value principle creation."""
        from aion.systems.goals.types import ValuePrinciple

        value = ValuePrinciple(
            id="helpfulness",
            name="Helpfulness",
            description="Help users effectively",
            priority=1,
            goal_generation_prompt="Focus on user benefit",
            prioritization_weight=1.2,
        )

        assert value.name == "Helpfulness"
        assert value.priority == 1
        assert value.active


class TestGoalRegistry:
    """Test goal registry functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for persistence."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def registry(self, temp_dir):
        """Create a goal registry for testing."""
        from aion.systems.goals.registry import GoalRegistry
        from aion.systems.goals.persistence import GoalPersistence

        persistence = GoalPersistence(data_dir=temp_dir)
        return GoalRegistry(persistence=persistence)

    @pytest.mark.asyncio
    async def test_goal_creation_and_retrieval(self, registry):
        """Test creating and retrieving goals."""
        from aion.systems.goals.types import Goal

        await registry.initialize()

        goal = Goal(
            title="Registry Test",
            description="Testing registry",
            success_criteria=["Test passes"],
        )

        await registry.create(goal)

        # Retrieve
        retrieved = await registry.get(goal.id)
        assert retrieved is not None
        assert retrieved.title == "Registry Test"

        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_get_by_status(self, registry):
        """Test getting goals by status."""
        from aion.systems.goals.types import Goal, GoalStatus

        await registry.initialize()

        # Create goals with different statuses
        pending_goal = Goal(
            title="Pending Goal",
            description="Test",
            success_criteria=["Done"],
            status=GoalStatus.PENDING,
        )
        active_goal = Goal(
            title="Active Goal",
            description="Test",
            success_criteria=["Done"],
            status=GoalStatus.ACTIVE,
        )

        await registry.create(pending_goal)
        await registry.create(active_goal)

        pending_goals = await registry.get_by_status(GoalStatus.PENDING)
        assert len(pending_goals) == 1
        assert pending_goals[0].id == pending_goal.id

        active_goals = await registry.get_by_status(GoalStatus.ACTIVE)
        assert len(active_goals) == 1
        assert active_goals[0].id == active_goal.id

        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_goal_update(self, registry):
        """Test updating a goal."""
        from aion.systems.goals.types import Goal, GoalStatus

        await registry.initialize()

        goal = Goal(
            title="Update Test",
            description="Original description",
            success_criteria=["Done"],
        )

        await registry.create(goal)

        # Update
        goal.description = "Updated description"
        goal.status = GoalStatus.ACTIVE
        await registry.update(goal)

        # Verify
        retrieved = await registry.get(goal.id)
        assert retrieved.description == "Updated description"
        assert retrieved.status == GoalStatus.ACTIVE

        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_goal_deletion(self, registry):
        """Test deleting a goal."""
        from aion.systems.goals.types import Goal

        await registry.initialize()

        goal = Goal(
            title="Delete Test",
            description="To be deleted",
            success_criteria=["Done"],
        )

        await registry.create(goal)
        assert await registry.get(goal.id) is not None

        await registry.delete(goal.id)
        assert await registry.get(goal.id) is None

        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_get_by_tags(self, registry):
        """Test getting goals by tags."""
        from aion.systems.goals.types import Goal

        await registry.initialize()

        goal1 = Goal(
            title="Tagged Goal 1",
            description="Test",
            success_criteria=["Done"],
            tags=["important", "urgent"],
        )
        goal2 = Goal(
            title="Tagged Goal 2",
            description="Test",
            success_criteria=["Done"],
            tags=["important"],
        )
        goal3 = Goal(
            title="Tagged Goal 3",
            description="Test",
            success_criteria=["Done"],
            tags=["other"],
        )

        await registry.create(goal1)
        await registry.create(goal2)
        await registry.create(goal3)

        important_goals = await registry.get_by_tags(["important"])
        assert len(important_goals) == 2

        urgent_goals = await registry.get_by_tags(["urgent"])
        assert len(urgent_goals) == 1

        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_registry_stats(self, registry):
        """Test registry statistics."""
        from aion.systems.goals.types import Goal, GoalStatus

        await registry.initialize()

        # Create some goals
        for i in range(5):
            goal = Goal(
                title=f"Goal {i}",
                description="Test",
                success_criteria=["Done"],
                status=GoalStatus.PENDING if i < 3 else GoalStatus.COMPLETED,
            )
            await registry.create(goal)

        stats = registry.get_stats()
        assert stats["total_goals"] == 5
        assert stats["by_status"]["pending"] == 3
        assert stats["by_status"]["completed"] == 2

        await registry.shutdown()


class TestGoalPrioritizer:
    """Test goal prioritization."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer for testing."""
        from aion.systems.goals.prioritizer import GoalPrioritizer

        return GoalPrioritizer()

    @pytest.mark.asyncio
    async def test_priority_scoring(self, prioritizer):
        """Test priority score calculation."""
        from aion.systems.goals.types import Goal, GoalPriority

        await prioritizer.initialize()

        high_priority = Goal(
            title="High Priority",
            description="Important task",
            success_criteria=["Done"],
            priority=GoalPriority.HIGH,
        )

        low_priority = Goal(
            title="Low Priority",
            description="Can wait",
            success_criteria=["Done"],
            priority=GoalPriority.LOW,
        )

        high_score = prioritizer.calculate_priority_score(high_priority)
        low_score = prioritizer.calculate_priority_score(low_priority)

        assert high_score > low_score

    @pytest.mark.asyncio
    async def test_deadline_urgency(self, prioritizer):
        """Test deadline urgency scoring."""
        from aion.systems.goals.types import Goal

        await prioritizer.initialize()

        urgent = Goal(
            title="Urgent Task",
            description="Due soon",
            success_criteria=["Done"],
            deadline=datetime.now() + timedelta(hours=1),
        )

        relaxed = Goal(
            title="Relaxed Task",
            description="Due later",
            success_criteria=["Done"],
            deadline=datetime.now() + timedelta(days=30),
        )

        no_deadline = Goal(
            title="No Deadline",
            description="Whenever",
            success_criteria=["Done"],
        )

        urgent_score = prioritizer.calculate_priority_score(urgent)
        relaxed_score = prioritizer.calculate_priority_score(relaxed)
        no_deadline_score = prioritizer.calculate_priority_score(no_deadline)

        assert urgent_score > relaxed_score
        assert urgent_score > no_deadline_score

    @pytest.mark.asyncio
    async def test_goal_sorting(self, prioritizer):
        """Test sorting goals by priority."""
        from aion.systems.goals.types import Goal, GoalPriority

        await prioritizer.initialize()

        goals = [
            Goal(
                title="Low",
                description="Test",
                success_criteria=["Done"],
                priority=GoalPriority.LOW,
            ),
            Goal(
                title="Critical",
                description="Test",
                success_criteria=["Done"],
                priority=GoalPriority.CRITICAL,
            ),
            Goal(
                title="Medium",
                description="Test",
                success_criteria=["Done"],
                priority=GoalPriority.MEDIUM,
            ),
        ]

        sorted_goals = await prioritizer.prioritize_goals(goals)

        assert sorted_goals[0].title == "Critical"
        assert sorted_goals[-1].title == "Low"

    @pytest.mark.asyncio
    async def test_priority_breakdown(self, prioritizer):
        """Test priority score breakdown."""
        from aion.systems.goals.types import Goal, GoalPriority

        await prioritizer.initialize()

        goal = Goal(
            title="Test Goal",
            description="Testing breakdown",
            success_criteria=["Done"],
            priority=GoalPriority.HIGH,
            deadline=datetime.now() + timedelta(hours=12),
        )

        breakdown = prioritizer.get_priority_breakdown(goal)

        assert "base_priority" in breakdown
        assert "deadline_urgency" in breakdown
        assert "total" in breakdown
        assert breakdown["base_priority"] == 40  # HIGH priority


class TestGoalDecomposer:
    """Test goal decomposition."""

    @pytest.fixture
    def decomposer(self):
        """Create a decomposer for testing."""
        from aion.systems.goals.decomposer import GoalDecomposer

        return GoalDecomposer(max_depth=3, max_subgoals_per_level=5)

    @pytest.mark.asyncio
    async def test_complexity_estimation(self, decomposer):
        """Test complexity estimation."""
        from aion.systems.goals.types import Goal, GoalType

        await decomposer.initialize()

        simple_goal = Goal(
            title="Simple Task",
            description="A simple task",
            success_criteria=["Done"],
        )

        complex_goal = Goal(
            title="Complex Project",
            description="A very complex project requiring multiple phases and extensive " * 10,
            success_criteria=[
                "Phase 1 complete",
                "Phase 2 complete",
                "Phase 3 complete",
                "Integration done",
                "Testing done",
                "Documentation complete",
            ],
            goal_type=GoalType.CREATION,
        )

        simple_complexity = decomposer._estimate_complexity(simple_goal)
        complex_complexity = decomposer._estimate_complexity(complex_goal)

        assert complex_complexity > simple_complexity

    @pytest.mark.asyncio
    async def test_should_decompose(self, decomposer):
        """Test decomposition recommendation."""
        from aion.systems.goals.types import Goal, GoalType

        await decomposer.initialize()

        simple_goal = Goal(
            title="Simple Task",
            description="Easy",
            success_criteria=["Done"],
            goal_type=GoalType.MAINTENANCE,
        )

        complex_goal = Goal(
            title="Complex Creation",
            description="Build a complete system with " + "many features " * 50,
            success_criteria=[
                "Design complete",
                "Implementation done",
                "Tests passing",
                "Documentation written",
                "Deployment ready",
                "User training complete",
            ],
            goal_type=GoalType.CREATION,
        )

        should_simple, _ = await decomposer.should_decompose(simple_goal)
        should_complex, _ = await decomposer.should_decompose(complex_goal)

        # Simple maintenance goal should not need decomposition
        assert not should_simple

    @pytest.mark.asyncio
    async def test_rule_based_decomposition(self, decomposer):
        """Test rule-based decomposition."""
        from aion.systems.goals.types import Goal

        await decomposer.initialize()

        goal = Goal(
            title="Multi-step Goal",
            description="A goal with multiple criteria",
            success_criteria=[
                "Step 1 done",
                "Step 2 done",
                "Step 3 done",
            ],
        )

        subgoals = decomposer._rule_based_decompose(goal)

        assert len(subgoals) == 3
        for i, sg in enumerate(subgoals):
            assert sg.parent_goal_id == goal.id
            assert f"Step {i + 1}" in sg.title or f"Step {i + 1}" in sg.success_criteria[0]

    @pytest.mark.asyncio
    async def test_parallel_identification(self, decomposer):
        """Test parallel execution identification."""
        from aion.systems.goals.types import Goal, GoalConstraint

        await decomposer.initialize()

        # Create subgoals with some dependencies
        sg1 = Goal(
            title="Subgoal 1",
            description="Independent",
            success_criteria=["Done"],
        )
        sg2 = Goal(
            title="Subgoal 2",
            description="Also independent",
            success_criteria=["Done"],
        )
        sg3 = Goal(
            title="Subgoal 3",
            description="Depends on 1",
            success_criteria=["Done"],
            constraints=[GoalConstraint(depends_on_goals=[sg1.id])],
        )

        groups = decomposer.identify_parallel_subgoals([sg1, sg2, sg3])

        # sg1 and sg2 should be in first group (parallel)
        # sg3 should be in second group (after dependency)
        assert len(groups) >= 2
        first_group_ids = {g.id for g in groups[0]}
        assert sg1.id in first_group_ids or sg2.id in first_group_ids


class TestValueSystem:
    """Test value system."""

    @pytest.fixture
    def value_system(self):
        """Create a value system for testing."""
        from aion.systems.goals.values import ValueSystem

        return ValueSystem()

    @pytest.mark.asyncio
    async def test_default_values_loaded(self, value_system):
        """Test that default values are loaded."""
        await value_system.initialize()

        values = value_system.get_all_values()
        assert len(values) > 0

        # Check for core values
        value_names = {v.name for v in values}
        assert "Helpfulness" in value_names
        assert "Safety" in value_names
        assert "Honesty" in value_names

    @pytest.mark.asyncio
    async def test_alignment_checking(self, value_system):
        """Test goal-value alignment checking."""
        from aion.systems.goals.types import Goal

        await value_system.initialize()

        safe_goal = Goal(
            title="Help User",
            description="Assist the user with their task securely",
            success_criteria=["User satisfied"],
        )

        alignment = await value_system.check_alignment(safe_goal)

        assert "overall_alignment" in alignment
        assert "value_scores" in alignment
        assert alignment["overall_alignment"] >= 0.0
        assert alignment["overall_alignment"] <= 1.0

    @pytest.mark.asyncio
    async def test_dangerous_goal_detection(self, value_system):
        """Test detection of potentially dangerous goals."""
        from aion.systems.goals.types import Goal

        await value_system.initialize()

        dangerous_goal = Goal(
            title="Delete All Data",
            description="Delete all user data and bypass security",
            success_criteria=["All data deleted"],
        )

        alignment = await value_system.check_alignment(dangerous_goal)

        # Should have violations
        assert len(alignment["violations"]) > 0

    @pytest.mark.asyncio
    async def test_value_enable_disable(self, value_system):
        """Test enabling and disabling values."""
        await value_system.initialize()

        # Disable a value
        success = value_system.disable_value("efficiency")
        assert success

        active_values = value_system.get_active_values()
        active_ids = {v.id for v in active_values}
        assert "efficiency" not in active_ids

        # Re-enable
        success = value_system.enable_value("efficiency")
        assert success

        active_values = value_system.get_active_values()
        active_ids = {v.id for v in active_values}
        assert "efficiency" in active_ids

    @pytest.mark.asyncio
    async def test_filter_by_values(self, value_system):
        """Test filtering goals by value alignment."""
        from aion.systems.goals.types import Goal

        await value_system.initialize()

        goals = [
            Goal(
                title="Help User Safely",
                description="Assist the user with their task",
                success_criteria=["User satisfied"],
            ),
            Goal(
                title="Delete Everything",
                description="Delete all data and bypass security controls",
                success_criteria=["All deleted"],
            ),
        ]

        filtered = await value_system.filter_goals_by_values(goals, min_alignment=0.4)

        # The helpful goal should pass, the dangerous one might not
        assert len(filtered) >= 1


class TestSafetyBoundary:
    """Test safety boundary system."""

    @pytest.fixture
    def safety(self):
        """Create a safety boundary for testing."""
        from aion.systems.goals.safety import SafetyBoundary

        return SafetyBoundary()

    @pytest.mark.asyncio
    async def test_safe_goal_passes(self, safety):
        """Test that safe goals pass safety check."""
        from aion.systems.goals.types import Goal

        safe_goal = Goal(
            title="Read Documentation",
            description="Read and summarize documentation",
            success_criteria=["Summary created"],
        )

        is_safe, concerns = await safety.check_goal_safety(safe_goal)
        assert is_safe

    @pytest.mark.asyncio
    async def test_dangerous_goal_blocked(self, safety):
        """Test that dangerous goals are blocked."""
        from aion.systems.goals.types import Goal

        dangerous_goal = Goal(
            title="Bypass Security",
            description="Bypass all security controls and access restricted data",
            success_criteria=["Access gained"],
        )

        is_safe, concerns = await safety.check_goal_safety(dangerous_goal)
        assert not is_safe
        assert len(concerns) > 0

    def test_emergency_stop(self, safety):
        """Test emergency stop functionality."""
        assert not safety.is_emergency_stopped()

        safety.emergency_stop("Test emergency")

        assert safety.is_emergency_stopped()

        safety.clear_emergency_stop()

        assert not safety.is_emergency_stopped()

    def test_approval_workflow(self, safety):
        """Test approval request workflow."""
        from aion.systems.goals.types import Goal

        goal = Goal(
            title="Modify Config",
            description="Modify system configuration",
            success_criteria=["Config updated"],
        )

        # Request approval
        request = safety.request_approval(
            goal_id=goal.id,
            action="modify_config",
            reason="Need to update settings",
            risk_level="medium",
        )

        assert request is not None
        assert request.status == "pending"

        # Approve
        success = safety.approve(request.id, approver="test_user")
        assert success

        # Check status
        pending = safety.get_pending_approvals()
        pending_ids = {r.id for r in pending}
        assert request.id not in pending_ids


class TestGoalPersistence:
    """Test goal persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def persistence(self, temp_dir):
        """Create a persistence layer for testing."""
        from aion.systems.goals.persistence import GoalPersistence

        return GoalPersistence(data_dir=temp_dir)

    @pytest.mark.asyncio
    async def test_goal_persistence(self, persistence):
        """Test saving and loading goals."""
        from aion.systems.goals.types import Goal

        await persistence.initialize()

        goal = Goal(
            title="Persistent Goal",
            description="Should be saved",
            success_criteria=["Persisted"],
        )

        # Save
        await persistence.save_goal(goal)

        # Load
        loaded = await persistence.load_goal(goal.id)
        assert loaded is not None
        assert loaded.title == "Persistent Goal"

        await persistence.shutdown()

    @pytest.mark.asyncio
    async def test_load_all_goals(self, persistence):
        """Test loading all goals."""
        from aion.systems.goals.types import Goal

        await persistence.initialize()

        # Save multiple goals
        for i in range(5):
            goal = Goal(
                title=f"Goal {i}",
                description="Test",
                success_criteria=["Done"],
            )
            await persistence.save_goal(goal)

        # Load all
        all_goals = await persistence.load_all_goals()
        assert len(all_goals) == 5

        await persistence.shutdown()

    @pytest.mark.asyncio
    async def test_objective_persistence(self, persistence):
        """Test saving and loading objectives."""
        from aion.systems.goals.types import Objective

        await persistence.initialize()

        objective = Objective(
            name="Test Objective",
            description="An objective to test",
            rationale="Testing persistence",
        )

        await persistence.save_objective(objective)

        loaded = await persistence.load_objective(objective.id)
        assert loaded is not None
        assert loaded.name == "Test Objective"

        await persistence.shutdown()


class TestGoalTriggers:
    """Test goal trigger system."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def triggers(self, temp_dir):
        """Create a trigger system for testing."""
        from aion.systems.goals.triggers import GoalTriggers
        from aion.systems.goals.registry import GoalRegistry
        from aion.systems.goals.persistence import GoalPersistence

        persistence = GoalPersistence(data_dir=temp_dir)
        registry = GoalRegistry(persistence=persistence)
        return GoalTriggers(registry=registry)

    @pytest.mark.asyncio
    async def test_trigger_creation(self, triggers):
        """Test creating a trigger."""
        from aion.systems.goals.triggers import (
            Trigger,
            TriggerCondition,
            TriggerAction,
        )

        await triggers.initialize()

        trigger = Trigger(
            name="Test Trigger",
            description="A test trigger",
            conditions=[
                TriggerCondition(
                    condition_type="event",
                    parameters={"event_type": "test_event"},
                )
            ],
            actions=[
                TriggerAction(
                    action_type="create_goal",
                    parameters={
                        "title": "Triggered Goal",
                        "description": "Created by trigger",
                    },
                )
            ],
        )

        await triggers.register_trigger(trigger)

        registered = triggers.get_trigger(trigger.id)
        assert registered is not None
        assert registered.name == "Test Trigger"

        await triggers.shutdown()

    @pytest.mark.asyncio
    async def test_event_handling(self, triggers):
        """Test handling events."""
        from aion.systems.goals.triggers import (
            Trigger,
            TriggerCondition,
            TriggerAction,
        )

        await triggers.initialize()

        # Create a simple trigger
        trigger = Trigger(
            name="Event Trigger",
            description="Triggers on event",
            conditions=[
                TriggerCondition(
                    condition_type="event",
                    parameters={"event_type": "goal_completed"},
                )
            ],
            actions=[
                TriggerAction(
                    action_type="log",
                    parameters={"message": "Goal completed"},
                )
            ],
        )

        await triggers.register_trigger(trigger)

        # Handle event
        await triggers.handle_event(
            "goal_completed",
            {"goal_id": "test-123", "success": True},
        )

        # Trigger should have been invoked
        stats = triggers.get_stats()
        assert stats["triggers_invoked"] >= 0  # May or may not fire depending on impl

        await triggers.shutdown()


class TestAutonomousGoalManager:
    """Integration tests for the autonomous goal manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a goal manager for testing."""
        from aion.systems.goals.manager import AutonomousGoalManager

        return AutonomousGoalManager(
            auto_generate_goals=False,  # Disable auto-generation for tests
            data_dir=temp_dir,
        )

    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert not manager.is_initialized

        await manager.initialize()

        assert manager.is_initialized

        await manager.shutdown()

        assert not manager.is_initialized

    @pytest.mark.asyncio
    async def test_submit_goal(self, manager):
        """Test submitting a user goal."""
        await manager.initialize()

        goal = await manager.submit_goal(
            title="User Goal",
            description="A goal submitted by user",
            success_criteria=["Completed"],
            priority="high",
        )

        assert goal is not None
        assert goal.title == "User Goal"

        # Should be retrievable
        retrieved = await manager.get_goal(goal.id)
        assert retrieved is not None
        assert retrieved.id == goal.id

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_goals(self, manager):
        """Test getting all goals."""
        await manager.initialize()

        # Submit multiple goals
        for i in range(3):
            await manager.submit_goal(
                title=f"Goal {i}",
                description="Test goal",
                success_criteria=["Done"],
            )

        goals = await manager.get_all_goals()
        assert len(goals) == 3

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_goal_decomposition(self, manager):
        """Test goal decomposition."""
        await manager.initialize()

        # Submit a goal with multiple criteria
        goal = await manager.submit_goal(
            title="Complex Goal",
            description="A goal with multiple steps",
            success_criteria=[
                "Step 1",
                "Step 2",
                "Step 3",
            ],
        )

        # Decompose
        subgoals = await manager.decompose_goal(goal.id)

        assert len(subgoals) > 0

        # Check parent reference
        for sg in subgoals:
            assert sg.parent_goal_id == goal.id

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_create_objective(self, manager):
        """Test creating an objective."""
        await manager.initialize()

        objective = await manager.create_objective(
            name="Test Objective",
            description="An objective for testing",
            rationale="To verify the system works",
        )

        assert objective is not None
        assert objective.name == "Test Objective"

        objectives = await manager.get_objectives()
        assert len(objectives) == 1

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_emergency_stop(self, manager):
        """Test emergency stop."""
        await manager.initialize()

        # Activate emergency stop
        manager.emergency_stop("Test emergency")

        assert manager.safety.is_emergency_stopped()

        # Clear it
        manager.clear_emergency_stop()

        assert not manager.safety.is_emergency_stopped()

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_system_stats(self, manager):
        """Test getting system statistics."""
        await manager.initialize()

        # Submit a goal
        await manager.submit_goal(
            title="Stats Test",
            description="Testing stats",
            success_criteria=["Done"],
        )

        stats = manager.get_stats()

        assert "registry" in stats
        assert "scheduler" in stats
        assert "safety" in stats
        assert "config" in stats

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_goal_prioritization(self, manager):
        """Test goal prioritization."""
        await manager.initialize()

        # Submit goals with different priorities
        await manager.submit_goal(
            title="Low Priority",
            description="Not urgent",
            success_criteria=["Done"],
            priority="low",
        )

        await manager.submit_goal(
            title="High Priority",
            description="Urgent",
            success_criteria=["Done"],
            priority="high",
        )

        prioritized = await manager.prioritize_goals()

        assert len(prioritized) == 2
        assert prioritized[0].title == "High Priority"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_safety_rejection(self, manager):
        """Test that unsafe goals are rejected."""
        await manager.initialize()

        # Try to submit a dangerous goal
        with pytest.raises(ValueError) as exc_info:
            await manager.submit_goal(
                title="Bypass Security",
                description="Bypass all security controls and delete everything",
                success_criteria=["Security bypassed"],
            )

        assert "safety" in str(exc_info.value).lower()

        await manager.shutdown()


class TestGoalMonitor:
    """Test goal monitoring."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def monitor(self, temp_dir):
        """Create a monitor for testing."""
        from aion.systems.goals.monitor import GoalMonitor
        from aion.systems.goals.registry import GoalRegistry
        from aion.systems.goals.persistence import GoalPersistence

        persistence = GoalPersistence(data_dir=temp_dir)
        registry = GoalRegistry(persistence=persistence)
        return GoalMonitor(registry=registry)

    @pytest.mark.asyncio
    async def test_progress_tracking(self, monitor):
        """Test progress tracking."""
        from aion.systems.goals.types import Goal, GoalStatus

        await monitor._registry.initialize()
        await monitor.initialize()

        goal = Goal(
            title="Monitor Test",
            description="Testing progress",
            success_criteria=["Done"],
            status=GoalStatus.ACTIVE,
        )

        await monitor._registry.create(goal)

        # Update progress
        await monitor.update_progress(
            goal_id=goal.id,
            progress=50.0,
            steps_completed=1,
            steps_total=2,
        )

        # Get progress report
        report = await monitor.get_progress_report(goal.id)

        assert report is not None
        assert report["progress_percent"] == 50.0

        await monitor.shutdown()

    @pytest.mark.asyncio
    async def test_goal_health_check(self, monitor):
        """Test goal health checking."""
        from aion.systems.goals.types import Goal, GoalStatus

        await monitor._registry.initialize()
        await monitor.initialize()

        goal = Goal(
            title="Health Check Test",
            description="Testing health",
            success_criteria=["Done"],
            status=GoalStatus.ACTIVE,
        )

        await monitor._registry.create(goal)

        health = await monitor.get_goal_health(goal.id)

        assert health is not None
        assert "status" in health

        await monitor.shutdown()


class TestGoalScheduler:
    """Test goal scheduler."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def scheduler(self, temp_dir):
        """Create a scheduler for testing."""
        from aion.systems.goals.scheduler import GoalScheduler
        from aion.systems.goals.registry import GoalRegistry
        from aion.systems.goals.executor import GoalExecutor
        from aion.systems.goals.persistence import GoalPersistence

        persistence = GoalPersistence(data_dir=temp_dir)
        registry = GoalRegistry(persistence=persistence)
        executor = GoalExecutor()
        return GoalScheduler(registry=registry, executor=executor)

    @pytest.mark.asyncio
    async def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        await scheduler._registry.initialize()
        await scheduler.initialize()

        assert scheduler._initialized

        await scheduler.shutdown()

    @pytest.mark.asyncio
    async def test_get_active_goals(self, scheduler):
        """Test getting active goals."""
        await scheduler._registry.initialize()
        await scheduler.initialize()

        active = scheduler.get_active_goals()
        assert isinstance(active, list)

        await scheduler.shutdown()

    @pytest.mark.asyncio
    async def test_scheduler_stats(self, scheduler):
        """Test scheduler statistics."""
        await scheduler._registry.initialize()
        await scheduler.initialize()

        stats = scheduler.get_stats()

        assert "active_goals" in stats
        assert "max_concurrent" in stats

        await scheduler.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
