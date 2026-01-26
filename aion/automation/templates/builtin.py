"""
AION Built-in Workflow Templates

Pre-built workflow templates for common patterns.
"""

from aion.automation.types import (
    Workflow,
    WorkflowStep,
    TriggerConfig,
    ActionConfig,
    Condition,
    TriggerType,
    ActionType,
    ConditionOperator,
    WorkflowTemplate,
    WorkflowStatus,
)


def get_builtin_templates() -> list[WorkflowTemplate]:
    """Get all built-in workflow templates."""
    return [
        daily_report_template(),
        webhook_to_agent_template(),
        approval_workflow_template(),
        data_sync_template(),
        alert_handler_template(),
        scheduled_cleanup_template(),
        event_driven_pipeline_template(),
        multi_step_processing_template(),
    ]


def daily_report_template() -> WorkflowTemplate:
    """Daily report generation workflow."""
    gather_step = WorkflowStep(
        id="gather_data",
        name="Gather Data",
        description="Query data for the report",
        action=ActionConfig(
            action_type=ActionType.TOOL,
            tool_name="data_query",
            tool_params={"query": "{{ inputs.query }}"},
        ),
    )

    generate_step = WorkflowStep(
        id="generate_report",
        name="Generate Report",
        description="Use AI to generate the report",
        action=ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Generate a concise report from this data:\n\n{{ steps.gather_data.output }}",
            llm_system_prompt="You are a report generator. Create clear, professional reports.",
        ),
    )

    send_step = WorkflowStep(
        id="send_report",
        name="Send Report",
        description="Send the report via email",
        action=ActionConfig(
            action_type=ActionType.NOTIFICATION,
            notification_channel="email",
            notification_title="Daily Report - {{ now }}",
            notification_message="{{ steps.generate_report.output.response }}",
            notification_recipients=["{{ inputs.recipients }}"],
        ),
    )

    # Link steps
    gather_step.on_success = "generate_report"
    generate_step.on_success = "send_report"

    workflow = Workflow(
        name="Daily Report Generator",
        description="Generates and sends a daily report based on data query",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(
                trigger_type=TriggerType.SCHEDULE,
                cron_expression="0 9 * * *",  # 9 AM daily
                description="Run daily at 9 AM",
            ),
        ],
        steps=[gather_step, generate_step, send_step],
        entry_step_id="gather_data",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Data query"},
                "recipients": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query", "recipients"],
        },
    )

    return WorkflowTemplate(
        name="Daily Report Generator",
        description="Generates and sends a daily report based on data query",
        category="reporting",
        workflow=workflow,
        parameters={
            "query": "SELECT * FROM metrics WHERE date = CURRENT_DATE",
            "recipients": ["team@example.com"],
        },
        author="AION",
        tags=["report", "scheduled", "email"],
    )


def webhook_to_agent_template() -> WorkflowTemplate:
    """Webhook triggers agent workflow."""
    spawn_step = WorkflowStep(
        id="spawn_agent",
        name="Spawn Agent",
        description="Create an agent to handle the webhook",
        action=ActionConfig(
            action_type=ActionType.AGENT,
            agent_operation="spawn",
            agent_role="analyst",
        ),
    )

    process_step = WorkflowStep(
        id="process_data",
        name="Process Data",
        description="Analyze the webhook payload",
        action=ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Analyze this webhook data and extract key information:\n\n{{ trigger.body | json }}",
        ),
    )

    store_step = WorkflowStep(
        id="store_result",
        name="Store Result",
        description="Store analysis in memory",
        action=ActionConfig(
            action_type=ActionType.DATA,
            data_operation="write",
            data_source="memory",
            data_key="webhook_analysis_{{ execution.id }}",
            data_value="{{ steps.process_data.output.response }}",
        ),
    )

    spawn_step.on_success = "process_data"
    process_step.on_success = "store_result"

    workflow = Workflow(
        name="Webhook to Agent",
        description="Spawns an agent to handle webhook data",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(
                trigger_type=TriggerType.WEBHOOK,
                webhook_path="/webhooks/agent-handler",
                description="Receive webhook triggers",
            ),
        ],
        steps=[spawn_step, process_step, store_step],
        entry_step_id="spawn_agent",
    )

    return WorkflowTemplate(
        name="Webhook to Agent",
        description="Handles webhook data with an AI agent",
        category="integration",
        workflow=workflow,
        author="AION",
        tags=["webhook", "agent", "integration"],
    )


def approval_workflow_template() -> WorkflowTemplate:
    """Workflow with human approval."""
    prepare_step = WorkflowStep(
        id="prepare",
        name="Prepare Request",
        description="Prepare data for approval",
        action=ActionConfig(
            action_type=ActionType.TRANSFORM,
            transform_expression="{{ inputs }}",
            transform_output_key="request",
        ),
    )

    approve_step = WorkflowStep(
        id="approve",
        name="Request Approval",
        description="Wait for human approval",
        action=ActionConfig(
            action_type=ActionType.APPROVAL,
            approval_title="Action Approval Required",
            approval_message="Please review and approve the following action:\n\n{{ request | json }}",
            approvers=["{{ inputs.approvers }}"],
            approval_timeout_hours=24,
        ),
    )

    execute_step = WorkflowStep(
        id="execute",
        name="Execute Action",
        description="Execute the approved action",
        action=ActionConfig(
            action_type=ActionType.TOOL,
            tool_name="{{ inputs.tool }}",
            tool_params="{{ inputs.params }}",
        ),
    )

    notify_step = WorkflowStep(
        id="notify",
        name="Notify Completion",
        description="Send completion notification",
        action=ActionConfig(
            action_type=ActionType.NOTIFICATION,
            notification_channel="console",
            notification_message="Action completed: {{ steps.execute.output }}",
        ),
    )

    prepare_step.on_success = "approve"
    approve_step.on_success = "execute"
    execute_step.on_success = "notify"

    workflow = Workflow(
        name="Approval Workflow",
        description="Requires human approval before executing an action",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(trigger_type=TriggerType.MANUAL),
        ],
        steps=[prepare_step, approve_step, execute_step, notify_step],
        entry_step_id="prepare",
        input_schema={
            "type": "object",
            "properties": {
                "approvers": {"type": "array", "items": {"type": "string"}},
                "tool": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["tool"],
        },
    )

    return WorkflowTemplate(
        name="Approval Workflow",
        description="Requires human approval before executing an action",
        category="governance",
        workflow=workflow,
        parameters={
            "approvers": ["admin@example.com"],
            "tool": "echo",
            "params": {"message": "Hello"},
        },
        author="AION",
        tags=["approval", "human-in-the-loop", "governance"],
    )


def data_sync_template() -> WorkflowTemplate:
    """Data synchronization workflow."""
    fetch_step = WorkflowStep(
        id="fetch_source",
        name="Fetch from Source",
        description="Get data from source API",
        action=ActionConfig(
            action_type=ActionType.WEBHOOK,
            webhook_url="{{ inputs.source_url }}",
            webhook_method="GET",
        ),
    )

    transform_step = WorkflowStep(
        id="transform",
        name="Transform Data",
        description="Transform data to target format",
        action=ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Transform this data to the required format:\n\n{{ steps.fetch_source.output.body }}",
            llm_system_prompt="You are a data transformation assistant. Output valid JSON only.",
        ),
    )

    push_step = WorkflowStep(
        id="push_target",
        name="Push to Target",
        description="Send data to target API",
        action=ActionConfig(
            action_type=ActionType.WEBHOOK,
            webhook_url="{{ inputs.target_url }}",
            webhook_method="POST",
            webhook_body={"data": "{{ steps.transform.output.response }}"},
        ),
    )

    fetch_step.on_success = "transform"
    transform_step.on_success = "push_target"

    workflow = Workflow(
        name="Data Sync",
        description="Synchronizes data between two endpoints",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(
                trigger_type=TriggerType.SCHEDULE,
                cron_expression="*/15 * * * *",  # Every 15 minutes
                description="Sync every 15 minutes",
            ),
        ],
        steps=[fetch_step, transform_step, push_step],
        entry_step_id="fetch_source",
        input_schema={
            "type": "object",
            "properties": {
                "source_url": {"type": "string"},
                "target_url": {"type": "string"},
            },
            "required": ["source_url", "target_url"],
        },
    )

    return WorkflowTemplate(
        name="Data Sync",
        description="Periodically syncs data between two endpoints",
        category="integration",
        workflow=workflow,
        parameters={
            "source_url": "https://api.source.com/data",
            "target_url": "https://api.target.com/data",
        },
        author="AION",
        tags=["sync", "api", "scheduled"],
    )


def alert_handler_template() -> WorkflowTemplate:
    """Alert handling workflow."""
    triage_step = WorkflowStep(
        id="triage",
        name="AI Triage",
        description="Analyze and categorize the alert",
        action=ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Analyze this alert and provide:\n1. Severity (critical/high/medium/low)\n2. Category\n3. Suggested actions\n\nAlert: {{ trigger.data | json }}",
            llm_system_prompt="You are an alert triage system. Be concise and actionable.",
        ),
    )

    notify_step = WorkflowStep(
        id="notify",
        name="Notify Team",
        description="Send alert notification",
        action=ActionConfig(
            action_type=ActionType.NOTIFICATION,
            notification_channel="slack",
            notification_title="Alert: {{ trigger.data.name }}",
            notification_message="AI Analysis:\n{{ steps.triage.output.response }}",
            notification_metadata={"webhook_url": "{{ inputs.slack_webhook }}"},
        ),
    )

    create_goal_step = WorkflowStep(
        id="create_goal",
        name="Create Remediation Goal",
        description="Create a goal to track remediation",
        action=ActionConfig(
            action_type=ActionType.GOAL,
            goal_operation="create",
            goal_title="Remediate: {{ trigger.data.name }}",
            goal_description="{{ steps.triage.output.response }}",
            goal_config={"priority": "high"},
        ),
    )

    triage_step.on_success = "notify"
    notify_step.on_success = "create_goal"

    workflow = Workflow(
        name="Alert Handler",
        description="AI-powered alert triage and remediation",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(
                trigger_type=TriggerType.EVENT,
                event_type="alert.fired",
                description="Trigger on alert events",
            ),
        ],
        steps=[triage_step, notify_step, create_goal_step],
        entry_step_id="triage",
        input_schema={
            "type": "object",
            "properties": {
                "slack_webhook": {"type": "string"},
            },
        },
    )

    return WorkflowTemplate(
        name="Alert Handler",
        description="AI-powered alert triage and remediation",
        category="operations",
        workflow=workflow,
        parameters={
            "slack_webhook": "https://hooks.slack.com/services/...",
        },
        author="AION",
        tags=["alert", "triage", "operations"],
    )


def scheduled_cleanup_template() -> WorkflowTemplate:
    """Scheduled cleanup workflow."""
    check_step = WorkflowStep(
        id="check",
        name="Check Cleanup Needed",
        description="Determine if cleanup is needed",
        action=ActionConfig(
            action_type=ActionType.TOOL,
            tool_name="check_storage",
            tool_params={"threshold": "{{ inputs.threshold }}"},
        ),
    )

    cleanup_step = WorkflowStep(
        id="cleanup",
        name="Perform Cleanup",
        description="Execute cleanup operation",
        action=ActionConfig(
            action_type=ActionType.TOOL,
            tool_name="cleanup",
            tool_params={
                "target": "{{ inputs.target }}",
                "retention_days": "{{ inputs.retention_days }}",
            },
        ),
        condition=Condition(
            left="{{ steps.check.output.needs_cleanup }}",
            operator=ConditionOperator.IS_TRUE,
        ),
    )

    report_step = WorkflowStep(
        id="report",
        name="Report Results",
        description="Send cleanup report",
        action=ActionConfig(
            action_type=ActionType.NOTIFICATION,
            notification_channel="email",
            notification_title="Cleanup Report",
            notification_message="Cleanup completed:\n{{ steps.cleanup.output | json }}",
            notification_recipients=["{{ inputs.notify_email }}"],
        ),
    )

    check_step.on_success = "cleanup"
    cleanup_step.on_success = "report"

    workflow = Workflow(
        name="Scheduled Cleanup",
        description="Periodic cleanup of old data",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(
                trigger_type=TriggerType.SCHEDULE,
                cron_expression="0 2 * * 0",  # 2 AM every Sunday
                description="Weekly cleanup at 2 AM Sunday",
            ),
        ],
        steps=[check_step, cleanup_step, report_step],
        entry_step_id="check",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "threshold": {"type": "number"},
                "retention_days": {"type": "integer"},
                "notify_email": {"type": "string"},
            },
        },
    )

    return WorkflowTemplate(
        name="Scheduled Cleanup",
        description="Periodic cleanup of old data with conditional execution",
        category="maintenance",
        workflow=workflow,
        parameters={
            "target": "/data/logs",
            "threshold": 80,
            "retention_days": 30,
            "notify_email": "admin@example.com",
        },
        author="AION",
        tags=["cleanup", "maintenance", "scheduled"],
    )


def event_driven_pipeline_template() -> WorkflowTemplate:
    """Event-driven data pipeline."""
    validate_step = WorkflowStep(
        id="validate",
        name="Validate Data",
        description="Validate incoming data",
        action=ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Validate this data and return {valid: true/false, errors: [...]}:\n\n{{ trigger.data | json }}",
        ),
    )

    transform_step = WorkflowStep(
        id="transform",
        name="Transform Data",
        description="Transform validated data",
        action=ActionConfig(
            action_type=ActionType.TRANSFORM,
            transform_expression="{{ trigger.data }}",
            transform_output_key="transformed",
        ),
        condition=Condition(
            left="{{ steps.validate.output.response }}",
            operator=ConditionOperator.CONTAINS,
            right="true",
        ),
    )

    load_step = WorkflowStep(
        id="load",
        name="Load Data",
        description="Load data to destination",
        action=ActionConfig(
            action_type=ActionType.DATA,
            data_operation="write",
            data_source="memory",
            data_key="pipeline_{{ execution.id }}",
            data_value="{{ transformed }}",
        ),
    )

    validate_step.on_success = "transform"
    transform_step.on_success = "load"

    workflow = Workflow(
        name="Event-Driven Pipeline",
        description="ETL pipeline triggered by events",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(
                trigger_type=TriggerType.EVENT,
                event_type="data.received",
                description="Trigger on data events",
            ),
        ],
        steps=[validate_step, transform_step, load_step],
        entry_step_id="validate",
    )

    return WorkflowTemplate(
        name="Event-Driven Pipeline",
        description="ETL pipeline triggered by data events",
        category="data",
        workflow=workflow,
        author="AION",
        tags=["etl", "pipeline", "event-driven"],
    )


def multi_step_processing_template() -> WorkflowTemplate:
    """Multi-step processing with loops."""
    fetch_step = WorkflowStep(
        id="fetch_items",
        name="Fetch Items",
        description="Get list of items to process",
        action=ActionConfig(
            action_type=ActionType.DATA,
            data_operation="read",
            data_source="context",
            data_key="inputs.items",
        ),
    )

    process_step = WorkflowStep(
        id="process_items",
        name="Process Each Item",
        description="Process items in a loop",
        action=ActionConfig(
            action_type=ActionType.LLM,
            llm_prompt="Process this item: {{ item }}",
        ),
        loop_over="{{ inputs.items }}",
        loop_variable="item",
    )

    aggregate_step = WorkflowStep(
        id="aggregate",
        name="Aggregate Results",
        description="Combine all results",
        action=ActionConfig(
            action_type=ActionType.TRANSFORM,
            transform_expression="{{ steps.process_items.output }}",
            transform_output_key="results",
        ),
    )

    fetch_step.on_success = "process_items"
    process_step.on_success = "aggregate"

    workflow = Workflow(
        name="Multi-Step Processing",
        description="Process multiple items in a loop",
        status=WorkflowStatus.DRAFT,
        triggers=[
            TriggerConfig(trigger_type=TriggerType.MANUAL),
        ],
        steps=[fetch_step, process_step, aggregate_step],
        entry_step_id="fetch_items",
        input_schema={
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["items"],
        },
    )

    return WorkflowTemplate(
        name="Multi-Step Processing",
        description="Process multiple items with looping",
        category="processing",
        workflow=workflow,
        parameters={
            "items": [{"id": 1}, {"id": 2}, {"id": 3}],
        },
        author="AION",
        tags=["loop", "batch", "processing"],
    )
