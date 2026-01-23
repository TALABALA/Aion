"""
AION Tools Exposed via MCP

Registers AION's cognitive capabilities as MCP tools.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

import structlog

from aion.mcp.types import TextContent, PromptMessage

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel
    from aion.mcp.server.server import MCPServer

logger = structlog.get_logger(__name__)


def register_aion_tools(server: "MCPServer", kernel: "AIONKernel") -> None:
    """
    Register AION's tools with the MCP server.

    Args:
        server: MCP server instance
        kernel: AION kernel instance
    """
    # === Memory Tools ===

    async def memory_search(arguments: Dict[str, Any]) -> str:
        """Search AION's memory."""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)

        if kernel.memory:
            results = await kernel.memory.search(query=query, limit=limit)
            return json.dumps([
                {
                    "content": r.content,
                    "similarity": r.similarity,
                    "metadata": r.metadata,
                }
                for r in results
            ], indent=2)
        return "Memory system not available"

    server.register_tool(
        name="memory_search",
        description="Search AION's cognitive memory for relevant information",
        handler=memory_search,
        parameters={
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )

    async def memory_store(arguments: Dict[str, Any]) -> str:
        """Store information in AION's memory."""
        content = arguments.get("content", "")
        metadata = arguments.get("metadata", {})

        if kernel.memory:
            memory_id = await kernel.memory.store(
                content=content,
                metadata=metadata,
            )
            return f"Memory stored with ID: {memory_id}"
        return "Memory system not available"

    server.register_tool(
        name="memory_store",
        description="Store information in AION's cognitive memory",
        handler=memory_store,
        parameters={
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to store",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata",
                },
            },
            "required": ["content"],
        },
    )

    # === Planning Tools ===

    async def plan_create(arguments: Dict[str, Any]) -> str:
        """Create an execution plan."""
        goal = arguments.get("goal", "")
        context = arguments.get("context", {})

        if kernel.planning:
            plan = await kernel.planning.create_plan(
                goal=goal,
                context=context,
            )
            return json.dumps({
                "plan_id": plan.id,
                "steps": [
                    {
                        "id": step.id,
                        "action": step.action,
                        "status": step.status,
                    }
                    for step in plan.steps
                ],
            }, indent=2)
        return "Planning system not available"

    server.register_tool(
        name="plan_create",
        description="Create a deterministic execution plan for a goal",
        handler=plan_create,
        parameters={
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Goal to plan for",
                },
                "context": {
                    "type": "object",
                    "description": "Additional context",
                },
            },
            "required": ["goal"],
        },
    )

    async def plan_execute(arguments: Dict[str, Any]) -> str:
        """Execute a plan."""
        plan_id = arguments.get("plan_id")

        if kernel.planning:
            result = await kernel.planning.execute_plan(plan_id)
            return json.dumps(result, indent=2)
        return "Planning system not available"

    server.register_tool(
        name="plan_execute",
        description="Execute a previously created plan",
        handler=plan_execute,
        parameters={
            "properties": {
                "plan_id": {
                    "type": "string",
                    "description": "ID of the plan to execute",
                },
            },
            "required": ["plan_id"],
        },
    )

    # === Tool Orchestration ===

    async def tool_list(arguments: Dict[str, Any]) -> str:
        """List available tools."""
        if kernel.tools:
            tools = kernel.tools.registry.list_tools()
            return json.dumps([
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category.value,
                }
                for t in tools
            ], indent=2)
        return "Tool system not available"

    server.register_tool(
        name="tool_list",
        description="List all available AION tools",
        handler=tool_list,
        parameters={
            "properties": {},
        },
    )

    async def tool_execute(arguments: Dict[str, Any]) -> str:
        """Execute an AION tool."""
        tool_name = arguments.get("tool_name", "")
        params = arguments.get("params", {})

        if kernel.tools:
            result = await kernel.tools.execute(tool_name, params)
            return json.dumps(result, indent=2)
        return "Tool system not available"

    server.register_tool(
        name="tool_execute",
        description="Execute a specific AION tool",
        handler=tool_execute,
        parameters={
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute",
                },
                "params": {
                    "type": "object",
                    "description": "Tool parameters",
                },
            },
            "required": ["tool_name"],
        },
    )

    # === Vision Tools ===

    async def vision_analyze(arguments: Dict[str, Any]) -> str:
        """Analyze an image."""
        image_path = arguments.get("image_path", "")
        query = arguments.get("query", "Describe this image")

        if kernel.vision:
            result = await kernel.vision.process(
                image_path=image_path,
                query=query,
            )
            return json.dumps(result, indent=2)
        return "Vision system not available"

    server.register_tool(
        name="vision_analyze",
        description="Analyze an image using AION's visual cortex",
        handler=vision_analyze,
        parameters={
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file",
                },
                "query": {
                    "type": "string",
                    "description": "Question about the image",
                },
            },
            "required": ["image_path"],
        },
    )

    # === Audio Tools ===

    async def audio_transcribe(arguments: Dict[str, Any]) -> str:
        """Transcribe audio."""
        audio_path = arguments.get("audio_path", "")

        if kernel.audio:
            result = await kernel.audio.transcribe(audio=audio_path)
            return json.dumps({
                "text": result.text,
                "language": result.language,
            }, indent=2)
        return "Audio system not available"

    server.register_tool(
        name="audio_transcribe",
        description="Transcribe audio using AION's auditory cortex",
        handler=audio_transcribe,
        parameters={
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": "Path to the audio file",
                },
            },
            "required": ["audio_path"],
        },
    )

    # === Process Management ===

    async def process_list(arguments: Dict[str, Any]) -> str:
        """List running processes."""
        if kernel.supervisor:
            processes = kernel.supervisor.list_processes()
            return json.dumps([
                {
                    "id": p.id,
                    "name": p.name,
                    "status": p.status.value,
                }
                for p in processes
            ], indent=2)
        return "Process supervisor not available"

    server.register_tool(
        name="process_list",
        description="List all running AION processes and agents",
        handler=process_list,
        parameters={
            "properties": {},
        },
    )

    # === System Status ===

    async def system_status(arguments: Dict[str, Any]) -> str:
        """Get system status."""
        status = kernel.get_status()
        return json.dumps(status, indent=2, default=str)

    server.register_tool(
        name="system_status",
        description="Get AION system status and health information",
        handler=system_status,
        parameters={
            "properties": {},
        },
    )

    logger.info("Registered AION tools with MCP server")


def register_aion_resources(server: "MCPServer", kernel: "AIONKernel") -> None:
    """
    Register AION's resources with the MCP server.

    Args:
        server: MCP server instance
        kernel: AION kernel instance
    """
    # System status resource
    async def get_status() -> str:
        return json.dumps(kernel.get_status(), indent=2, default=str)

    server.register_resource(
        uri="aion://system/status",
        name="System Status",
        handler=get_status,
        description="Current AION system status",
        mime_type="application/json",
    )

    # Configuration resource
    async def get_config() -> str:
        return json.dumps(kernel.config.model_dump(), indent=2, default=str)

    server.register_resource(
        uri="aion://system/config",
        name="System Configuration",
        handler=get_config,
        description="AION configuration (read-only)",
        mime_type="application/json",
    )

    # Health resource
    async def get_health() -> str:
        health = kernel.get_health()
        return json.dumps({
            name: {
                "status": h.status.value,
                "message": h.message,
            }
            for name, h in health.items()
        }, indent=2)

    server.register_resource(
        uri="aion://system/health",
        name="System Health",
        handler=get_health,
        description="Health status of all subsystems",
        mime_type="application/json",
    )

    logger.info("Registered AION resources with MCP server")


def register_aion_prompts(server: "MCPServer", kernel: "AIONKernel") -> None:
    """
    Register AION's prompts with the MCP server.

    Args:
        server: MCP server instance
        kernel: AION kernel instance
    """
    # Task execution prompt
    async def task_prompt(arguments: Dict[str, str]) -> list:
        task = arguments.get("task", "")
        context = arguments.get("context", "")

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""Execute the following task using AION's capabilities:

Task: {task}

Context: {context}

Use the available tools to complete this task efficiently.""",
                },
            },
        ]

    server.register_prompt(
        name="task_execute",
        handler=task_prompt,
        description="Execute a task using AION",
        arguments=[
            {"name": "task", "description": "Task to execute", "required": True},
            {"name": "context", "description": "Additional context", "required": False},
        ],
    )

    # Memory recall prompt
    async def memory_recall_prompt(arguments: Dict[str, str]) -> list:
        topic = arguments.get("topic", "")

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""Search AION's memory for information about: {topic}

Use the memory_search tool to find relevant memories, then summarize the findings.""",
                },
            },
        ]

    server.register_prompt(
        name="memory_recall",
        handler=memory_recall_prompt,
        description="Recall information from AION's memory",
        arguments=[
            {"name": "topic", "description": "Topic to search for", "required": True},
        ],
    )

    # Analysis prompt
    async def analysis_prompt(arguments: Dict[str, str]) -> list:
        subject = arguments.get("subject", "")
        analysis_type = arguments.get("type", "general")

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""Perform a {analysis_type} analysis of: {subject}

Use AION's cognitive capabilities including:
- Memory search for relevant context
- Vision analysis if visual data is provided
- Planning for complex multi-step analysis

Provide a comprehensive analysis.""",
                },
            },
        ]

    server.register_prompt(
        name="analysis",
        handler=analysis_prompt,
        description="Perform analysis using AION",
        arguments=[
            {"name": "subject", "description": "Subject to analyze", "required": True},
            {"name": "type", "description": "Type of analysis", "required": False},
        ],
    )

    logger.info("Registered AION prompts with MCP server")


def setup_aion_mcp_server(
    server: "MCPServer",
    kernel: "AIONKernel",
) -> None:
    """
    Set up the MCP server with all AION capabilities.

    Args:
        server: MCP server instance
        kernel: AION kernel instance
    """
    register_aion_tools(server, kernel)
    register_aion_resources(server, kernel)
    register_aion_prompts(server, kernel)

    logger.info("AION MCP server fully configured")
