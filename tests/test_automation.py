"""
Tests for AION Workflow Automation System.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict

# Import types
from aion.automation.types import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    StepResult,
    TriggerConfig,
    ActionConfig,
    Condition,
    ApprovalRequest,
    WorkflowStatus,
    ExecutionStatus,
    TriggerType,
    ActionType,
    ConditionOperator,
    ApprovalStatus,
)

# Import components
from aion.automation.registry import WorkflowRegistry
from aion.automation.execution.context import ExecutionContext
from aion.automation.conditions.evaluator import ConditionEvaluator, ConditionBuilder
from aion.automation.conditions.operators import compare, OperatorRegistry
from aion.automation.conditions.expressions import ExpressionParser
from aion.automation.approval.manager import ApprovalManager, ApprovalResult
from aion.automation.approval.gates import (
    SingleApproverGate,
    MultiApproverGate,
    QuorumGate,
    create_gate,
)
from aion.automation.templates.builtin import get_builtin_templates


# === Type Tests ===


class TestWorkflowTypes:
    """Tests for workflow type definitions."""

    def test_workflow_creation(self):
        """Test creating a workflow."""
        workflow = Workflow(
            name="Test Workflow",
            description="A test workflow",
        )

        assert workflow.id is not None
        assert workflow.name == "Test Workflow"
        assert workflow.status == WorkflowStatus.DRAFT
        assert len(workflow.steps) == 0
        assert len(workflow.triggers) == 0

    def test_workflow_with_steps(self):
        """Test creating a workflow with steps."""
        step1 = WorkflowStep(
            id="step1",
            name="First Step",
            action=ActionConfig(
                action_type=ActionType.TOOL,
                tool_name="echo",
                tool_params={"message": "hello"},
            ),
        )

        step2 = WorkflowStep(
            id="step2",
            name="Second Step",
            action=ActionConfig(
                action_type=ActionType.NOTIFICATION,
                notification_channel="console",
                notification_message="Done",
            ),
        )

        step1.on_success = "step2"

        workflow = Workflow(
            name="Test",
            steps=[step1, step2],
            entry_step_id="step1",
        )

        assert len(workflow.steps) == 2
        assert workflow.get_step("step1") is not None
        assert workflow.get_step("step1").on_success == "step2"
        assert workflow.get_entry_step().id == "step1"

    def test_workflow_validation(self):
        """Test workflow validation."""
        # Empty workflow
        workflow = Workflow(name="")
        errors = workflow.validate()
        assert "Workflow name is required" in errors

        # Workflow without steps
        workflow = Workflow(name="Test")
        errors = workflow.validate()
        assert "Workflow must have at least one step" in errors

        # Valid workflow
        workflow = Workflow(
            name="Test",
            steps=[WorkflowStep(name="Step1", action=ActionConfig(action_type=ActionType.DELAY))],
        )
        errors = workflow.validate()
        assert len(errors) == 0

    def test_workflow_serialization(self):
        """Test workflow to_dict and from_dict."""
        workflow = Workflow(
            name="Test",
            description="Test workflow",
            triggers=[
                TriggerConfig(
                    trigger_type=TriggerType.SCHEDULE,
                    cron_expression="0 9 * * *",
                ),
            ],
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Step 1",
                    action=ActionConfig(
                        action_type=ActionType.TOOL,
                        tool_name="echo",
                    ),
                ),
            ],
            entry_step_id="step1",
        )

        # Serialize
        data = workflow.to_dict()
        assert data["name"] == "Test"
        assert len(data["triggers"]) == 1
        assert len(data["steps"]) == 1

        # Deserialize
        restored = Workflow.from_dict(data)
        assert restored.name == workflow.name
        assert len(restored.triggers) == 1
        assert len(restored.steps) == 1

    def test_trigger_config(self):
        """Test trigger configuration."""
        # Schedule trigger
        schedule = TriggerConfig(
            trigger_type=TriggerType.SCHEDULE,
            cron_expression="0 9 * * *",
            timezone="America/New_York",
        )
        assert schedule.trigger_type == TriggerType.SCHEDULE
        assert schedule.cron_expression == "0 9 * * *"

        # Webhook trigger
        webhook = TriggerConfig(
            trigger_type=TriggerType.WEBHOOK,
            webhook_path="/hooks/test",
            webhook_secret="secret123",
        )
        assert webhook.trigger_type == TriggerType.WEBHOOK
        assert webhook.webhook_path == "/hooks/test"

        # Event trigger
        event = TriggerConfig(
            trigger_type=TriggerType.EVENT,
            event_type="user.created",
            event_filter={"source": "api"},
        )
        assert event.trigger_type == TriggerType.EVENT
        assert event.event_type == "user.created"

    def test_action_config(self):
        """Test action configuration."""
        # Tool action
        tool = ActionConfig(
            action_type=ActionType.TOOL,
            tool_name="shell",
            tool_params={"command": "ls"},
            timeout_seconds=60.0,
        )
        assert tool.action_type == ActionType.TOOL
        assert tool.tool_name == "shell"

        # LLM action
        llm = ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Summarize: {{ inputs.text }}",
            llm_model="claude-3",
        )
        assert llm.action_type == ActionType.LLM
        assert llm.llm_prompt is not None

        # Approval action
        approval = ActionConfig(
            action_type=ActionType.APPROVAL,
            approval_message="Please approve this",
            approvers=["admin@example.com"],
            approval_timeout_hours=24.0,
        )
        assert approval.action_type == ActionType.APPROVAL
        assert len(approval.approvers) == 1

    def test_execution_status(self):
        """Test workflow execution."""
        execution = WorkflowExecution(
            workflow_id="wf-123",
            workflow_name="Test",
            inputs={"key": "value"},
        )

        assert execution.status == ExecutionStatus.PENDING
        assert not execution.is_terminal()

        # Start
        execution.start()
        assert execution.status == ExecutionStatus.RUNNING
        assert execution.started_at is not None

        # Complete
        execution.complete({"result": "success"})
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.is_terminal()
        assert execution.outputs["result"] == "success"


# === Condition Tests ===


class TestConditions:
    """Tests for condition evaluation."""

    def test_operators(self):
        """Test comparison operators."""
        # Equality
        assert compare("hello", ConditionOperator.EQUALS, "hello") is True
        assert compare("hello", ConditionOperator.NOT_EQUALS, "world") is True
        assert compare(5, ConditionOperator.EQUALS, 5) is True
        assert compare("5", ConditionOperator.EQUALS, 5) is True  # Type coercion

        # Numeric comparisons
        assert compare(10, ConditionOperator.GREATER_THAN, 5) is True
        assert compare(5, ConditionOperator.LESS_THAN, 10) is True
        assert compare(10, ConditionOperator.GREATER_EQUAL, 10) is True
        assert compare(5, ConditionOperator.LESS_EQUAL, 10) is True

        # Contains
        assert compare("hello world", ConditionOperator.CONTAINS, "world") is True
        assert compare([1, 2, 3], ConditionOperator.CONTAINS, 2) is True
        assert compare({"key": "value"}, ConditionOperator.CONTAINS, "key") is True

        # Null checks
        assert compare(None, ConditionOperator.IS_NULL, None) is True
        assert compare("value", ConditionOperator.IS_NOT_NULL, None) is True

        # Empty checks
        assert compare("", ConditionOperator.IS_EMPTY, None) is True
        assert compare([], ConditionOperator.IS_EMPTY, None) is True
        assert compare("hello", ConditionOperator.IS_NOT_EMPTY, None) is True

        # In/Not In
        assert compare(2, ConditionOperator.IN, [1, 2, 3]) is True
        assert compare(5, ConditionOperator.NOT_IN, [1, 2, 3]) is True

        # Regex
        assert compare("hello123", ConditionOperator.MATCHES, r"\d+") is True

        # Boolean
        assert compare(True, ConditionOperator.IS_TRUE, None) is True
        assert compare("true", ConditionOperator.IS_TRUE, None) is True
        assert compare(False, ConditionOperator.IS_FALSE, None) is True

    def test_expression_parser(self):
        """Test expression parsing."""
        parser = ExpressionParser({"x": 10, "y": 5, "name": "test"})

        # Simple values
        assert parser.parse("x") == 10
        assert parser.parse("name") == "test"

        # Arithmetic
        assert parser.parse("x + y") == 15
        assert parser.parse("x * y") == 50
        assert parser.parse("x - y") == 5
        assert parser.parse("x / y") == 2.0

        # Comparisons
        assert parser.parse("x > y") is True
        assert parser.parse("x == 10") is True
        assert parser.parse("x != y") is True

        # Boolean logic
        assert parser.parse("x > 5 and y < 10") is True
        assert parser.parse("x < 5 or y < 10") is True
        assert parser.parse("not (x < 5)") is True

        # Functions
        assert parser.parse("len([1, 2, 3])") == 3
        assert parser.parse("max(x, y)") == 10
        assert parser.parse("min(x, y)") == 5

        # Ternary
        assert parser.parse("'yes' if x > 5 else 'no'") == "yes"

    def test_condition_builder(self):
        """Test condition builder."""
        # Simple condition
        condition = ConditionBuilder().when("status").equals("active").build()
        assert condition.left == "status"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.right == "active"

        # AND conditions
        condition = (
            ConditionBuilder()
            .when("status").equals("active")
            .and_when("priority").greater_than(5)
            .build()
        )
        assert len(condition.and_conditions) == 1

        # OR conditions
        condition = (
            ConditionBuilder()
            .when("status").equals("active")
            .or_when("status").equals("pending")
            .build_or()
        )
        assert len(condition.or_conditions) == 1

    @pytest.mark.asyncio
    async def test_condition_evaluator(self):
        """Test full condition evaluation."""
        execution = WorkflowExecution(
            workflow_id="test",
            inputs={"status": "active", "count": 10},
        )
        workflow = Workflow(name="Test", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        context = ExecutionContext(execution, workflow)

        evaluator = ConditionEvaluator()

        # Simple condition
        condition = Condition(
            left="inputs.status",
            operator=ConditionOperator.EQUALS,
            right="active",
        )
        result = await evaluator.evaluate(condition, context)
        assert result is True

        # Numeric condition
        condition = Condition(
            left="inputs.count",
            operator=ConditionOperator.GREATER_THAN,
            right=5,
        )
        result = await evaluator.evaluate(condition, context)
        assert result is True

        # Negated condition
        condition = Condition(
            left="inputs.status",
            operator=ConditionOperator.EQUALS,
            right="inactive",
            negate=True,
        )
        result = await evaluator.evaluate(condition, context)
        assert result is True


# === Context Tests ===


class TestExecutionContext:
    """Tests for execution context."""

    def test_basic_operations(self):
        """Test basic context operations."""
        execution = WorkflowExecution(
            workflow_id="test",
            inputs={"name": "test", "count": 5},
            trigger_data={"source": "manual"},
        )
        workflow = Workflow(name="Test", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        context = ExecutionContext(execution, workflow)

        # Get built-in values
        assert context.get("inputs.name") == "test"
        assert context.get("inputs.count") == 5
        assert context.get("trigger.source") == "manual"

        # Set and get
        context.set("custom.value", "hello")
        assert context.get("custom.value") == "hello"

        # Delete
        context.delete("custom.value")
        assert context.get("custom.value") is None

    def test_expression_resolution(self):
        """Test expression resolution."""
        execution = WorkflowExecution(
            workflow_id="test",
            inputs={"name": "World"},
        )
        workflow = Workflow(name="Test", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        context = ExecutionContext(execution, workflow)

        # Simple expression
        result = context.resolve("{{ inputs.name }}")
        assert result == "World"

        # String interpolation
        result = context.resolve("Hello, {{ inputs.name }}!")
        assert result == "Hello, World!"

        # Non-expression passthrough
        result = context.resolve("plain string")
        assert result == "plain string"

        # Dict resolution
        result = context.resolve({"key": "{{ inputs.name }}"})
        assert result == {"key": "World"}

        # List resolution
        result = context.resolve(["{{ inputs.name }}", "static"])
        assert result == ["World", "static"]

    def test_step_outputs(self):
        """Test step output management."""
        execution = WorkflowExecution(workflow_id="test")
        workflow = Workflow(name="Test", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        context = ExecutionContext(execution, workflow)

        # Set step output
        context.set_step_output("step1", {"result": "success"})

        # Get step output
        assert context.get_step_output("step1") == {"result": "success"}

        # Access via expression
        assert context.get("steps.step1.output.result") == "success"

    def test_filters(self):
        """Test expression filters."""
        execution = WorkflowExecution(
            workflow_id="test",
            inputs={"text": "Hello World", "items": [1, 2, 3]},
        )
        workflow = Workflow(name="Test", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        context = ExecutionContext(execution, workflow)

        # Upper/lower
        assert context.resolve("{{ inputs.text | upper }}") == "HELLO WORLD"
        assert context.resolve("{{ inputs.text | lower }}") == "hello world"

        # Length
        assert context.resolve("{{ inputs.items | length }}") == 3

        # First/last
        assert context.resolve("{{ inputs.items | first }}") == 1
        assert context.resolve("{{ inputs.items | last }}") == 3

    def test_loop_context(self):
        """Test loop context management."""
        execution = WorkflowExecution(workflow_id="test")
        workflow = Workflow(name="Test", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        context = ExecutionContext(execution, workflow)

        context.enter_loop("item", "index")

        # First iteration
        context.set_loop_item("item", 0, "first")
        assert context.get("item") == "first"
        assert context.get("index") == 0

        # Second iteration
        context.set_loop_item("item", 1, "second")
        assert context.get("item") == "second"
        assert context.get("index") == 1

        context.exit_loop()
        assert context.get("item") is None


# === Registry Tests ===


class TestWorkflowRegistry:
    """Tests for workflow registry."""

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        """Test saving and retrieving workflows."""
        registry = WorkflowRegistry()
        await registry.initialize()

        workflow = Workflow(
            name="Test Workflow",
            steps=[WorkflowStep(name="Step1", action=ActionConfig())],
        )

        # Save
        workflow_id = await registry.save(workflow)
        assert workflow_id == workflow.id

        # Get
        retrieved = await registry.get(workflow_id)
        assert retrieved is not None
        assert retrieved.name == "Test Workflow"

        # Get by name
        by_name = await registry.get_by_name("Test Workflow")
        assert by_name is not None
        assert by_name.id == workflow_id

    @pytest.mark.asyncio
    async def test_list_and_filter(self):
        """Test listing workflows with filters."""
        registry = WorkflowRegistry()
        await registry.initialize()

        # Create test workflows
        for i in range(5):
            workflow = Workflow(
                name=f"Workflow {i}",
                status=WorkflowStatus.ACTIVE if i % 2 == 0 else WorkflowStatus.DRAFT,
                tags=["test", f"tag{i}"],
                steps=[WorkflowStep(name="S1", action=ActionConfig())],
            )
            await registry.save(workflow)

        # List all
        workflows = await registry.list()
        assert len(workflows) == 5

        # Filter by status
        active = await registry.list(status=WorkflowStatus.ACTIVE)
        assert len(active) == 3  # 0, 2, 4

        # Filter by tag
        tagged = await registry.list(tag="tag1")
        assert len(tagged) == 1

        # Search
        searched = await registry.list(search="Workflow 2")
        assert len(searched) == 1

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting workflows."""
        registry = WorkflowRegistry()
        await registry.initialize()

        workflow = Workflow(name="Delete Me", steps=[WorkflowStep(name="S1", action=ActionConfig())])
        workflow_id = await registry.save(workflow)

        # Delete
        deleted = await registry.delete(workflow_id)
        assert deleted is True

        # Verify deleted
        retrieved = await registry.get(workflow_id)
        assert retrieved is None

        # Delete non-existent
        deleted = await registry.delete("non-existent")
        assert deleted is False


# === Approval Tests ===


class TestApprovalManager:
    """Tests for approval manager."""

    @pytest.mark.asyncio
    async def test_create_request(self):
        """Test creating approval requests."""
        manager = ApprovalManager()
        await manager.initialize()

        request = await manager.create_request(
            execution_id="exec-123",
            step_id="step-1",
            message="Please approve this action",
            approvers=["admin@example.com"],
            timeout_hours=1.0,
        )

        assert request.id is not None
        assert request.status == ApprovalStatus.PENDING
        assert len(request.approvers) == 1
        assert request.expires_at is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_approve(self):
        """Test approving requests."""
        manager = ApprovalManager()
        await manager.initialize()

        request = await manager.create_request(
            execution_id="exec-123",
            step_id="step-1",
            message="Approve",
            approvers=["admin@example.com"],
        )

        # Approve
        success = await manager.approve(
            request.id,
            approver="admin@example.com",
            message="Looks good",
        )

        assert success is True

        # Check status
        updated = await manager.get_request(request.id)
        assert updated.status == ApprovalStatus.APPROVED
        assert updated.responded_by == "admin@example.com"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_reject(self):
        """Test rejecting requests."""
        manager = ApprovalManager()
        await manager.initialize()

        request = await manager.create_request(
            execution_id="exec-123",
            step_id="step-1",
            message="Approve",
            approvers=["admin@example.com"],
        )

        # Reject
        success = await manager.reject(
            request.id,
            approver="admin@example.com",
            message="Not approved",
        )

        assert success is True

        # Check status
        updated = await manager.get_request(request.id)
        assert updated.status == ApprovalStatus.REJECTED

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_decision(self):
        """Test waiting for approval decision."""
        manager = ApprovalManager()
        await manager.initialize()

        request = await manager.create_request(
            execution_id="exec-123",
            step_id="step-1",
            message="Approve",
        )

        # Start waiting in background
        async def wait():
            return await manager.wait_for_decision(request.id, timeout_hours=0.001)

        wait_task = asyncio.create_task(wait())

        # Approve after short delay
        await asyncio.sleep(0.1)
        await manager.approve(request.id, "admin", "OK")

        result = await wait_task
        assert result.approved is True

        await manager.shutdown()

    def test_approval_gates(self):
        """Test approval gate types."""
        # Single approver
        gate = create_gate("single")
        assert isinstance(gate, SingleApproverGate)

        # Multi approver
        gate = create_gate("multi", {"min_approvals": 3})
        assert isinstance(gate, MultiApproverGate)
        assert gate.min_approvals == 3

        # Quorum
        gate = create_gate("quorum", {"quorum_percentage": 0.75})
        assert isinstance(gate, QuorumGate)
        assert gate.quorum_percentage == 0.75


# === Template Tests ===


class TestTemplates:
    """Tests for workflow templates."""

    def test_builtin_templates(self):
        """Test built-in templates."""
        templates = get_builtin_templates()

        assert len(templates) > 0

        # Check each template is valid
        for template in templates:
            assert template.name is not None
            assert template.description is not None
            assert template.category is not None
            assert template.workflow is not None

            # Validate workflow
            errors = template.workflow.validate()
            assert len(errors) == 0, f"Template {template.name} has errors: {errors}"

    def test_template_instantiation(self):
        """Test instantiating templates."""
        templates = get_builtin_templates()
        template = templates[0]

        # Instantiate
        workflow = template.instantiate(
            name="My Custom Workflow",
            parameters={"custom": "value"},
        )

        assert workflow.name == "My Custom Workflow"
        assert workflow.id != template.workflow.id
        assert workflow.status == WorkflowStatus.DRAFT
        assert "template_id" in workflow.metadata

    def test_template_categories(self):
        """Test template categories."""
        templates = get_builtin_templates()

        categories = set()
        for template in templates:
            categories.add(template.category)

        # Should have multiple categories
        assert len(categories) > 1


# === Integration Tests ===


class TestWorkflowExecution:
    """Integration tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self):
        """Test executing a simple workflow."""
        from aion.automation.engine import WorkflowEngine

        engine = WorkflowEngine()
        await engine.initialize()

        # Create workflow
        workflow = Workflow(
            name="Simple Test",
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Transform",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="{{ inputs.value }}",
                        transform_output_key="result",
                    ),
                ),
            ],
            entry_step_id="step1",
        )

        await engine.register_workflow(workflow)

        # Execute
        execution = await engine.execute(
            workflow_id=workflow.id,
            inputs={"value": "hello"},
        )

        # Wait for completion
        await asyncio.sleep(0.5)

        # Check result
        updated = engine.get_execution(execution.id)
        assert updated.status == ExecutionStatus.COMPLETED

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_conditional_workflow(self):
        """Test workflow with conditions."""
        from aion.automation.engine import WorkflowEngine

        engine = WorkflowEngine()
        await engine.initialize()

        # Create workflow with conditional step
        workflow = Workflow(
            name="Conditional Test",
            steps=[
                WorkflowStep(
                    id="check",
                    name="Check Condition",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="{{ inputs.value }}",
                    ),
                    on_success="conditional",
                ),
                WorkflowStep(
                    id="conditional",
                    name="Conditional Step",
                    action=ActionConfig(
                        action_type=ActionType.TRANSFORM,
                        transform_expression="'executed'",
                    ),
                    condition=Condition(
                        left="{{ inputs.run_conditional }}",
                        operator=ConditionOperator.IS_TRUE,
                    ),
                ),
            ],
            entry_step_id="check",
        )

        await engine.register_workflow(workflow)

        # Execute with condition true
        execution = await engine.execute(
            workflow_id=workflow.id,
            inputs={"value": "test", "run_conditional": True},
        )

        await asyncio.sleep(0.5)

        updated = engine.get_execution(execution.id)
        assert updated.status == ExecutionStatus.COMPLETED
        assert "conditional" in updated.step_results

        await engine.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
